from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict

from .manifest import PairRecord


def weather_scene_weights(
    records: list[PairRecord], probabilities: dict[str, float]
) -> list[float]:
    """Sample a weather, then a scene uniformly, then one of its variants."""
    present = {record.weather for record in records}
    if present != set(probabilities):
        raise ValueError(f"Weather probabilities {set(probabilities)} must match data {present}")
    if any(not math.isfinite(p) or p <= 0 for p in probabilities.values()):
        raise ValueError("Weather probabilities must be finite and positive")
    if any(not record.group_id for record in records):
        raise ValueError("Weather/scene sampling requires an audited group_id for every pair")
    counts = Counter((record.weather, record.group_id) for record in records)
    groups = defaultdict(set)
    for weather, group in counts:
        groups[weather].add(group)
    total = sum(probabilities.values())
    return [
        probabilities[r.weather] / total / len(groups[r.weather]) / counts[r.weather, r.group_id]
        for r in records
    ]


class RestorationBatchSampler:
    """Build global batches once deterministically, then shard them across ranks.

    Balanced mode enforces weather quotas in every global batch and cycles
    shuffled background groups/variants, so repeated video frames do not dominate.
    ``batch_size`` and ``samples_per_epoch`` always mean global totals.
    """

    def __init__(
        self,
        records,
        batch_size,
        samples_per_epoch,
        *,
        probabilities=None,
        balanced=False,
        seed=42,
        rank=0,
        world_size=1,
        stratified_cycles=False,
        seeded_augmentations=False,
    ):
        if not records or batch_size <= 0 or samples_per_epoch <= 0:
            raise ValueError("Sampler needs records and positive batch/epoch sizes")
        if world_size <= 0 or batch_size % world_size or samples_per_epoch % batch_size:
            raise ValueError(
                "Global batch must divide across ranks; epoch must contain full batches"
            )
        if not 0 <= rank < world_size:
            raise ValueError("Invalid sampler rank")
        self.records = records
        self.batch_size, self.samples_per_epoch = batch_size, samples_per_epoch
        self.seed, self.rank, self.world_size, self.epoch = seed, rank, world_size, 0
        self.weights = weather_scene_weights(records, probabilities) if probabilities else None
        self.stratified_cycles = stratified_cycles
        self.seeded_augmentations = seeded_augmentations
        if stratified_cycles and (not balanced or any(not r.stratum for r in records)):
            raise ValueError("Stratified cycles require balanced batches and frozen strata")
        self.quotas = None
        if balanced:
            if not probabilities:
                raise ValueError("Balanced batches require weather_probabilities")
            total = sum(probabilities.values())
            quotas = {w: batch_size * p / total for w, p in probabilities.items()}
            if any(abs(q - round(q)) > 1e-8 or q < 1 for q in quotas.values()):
                raise ValueError("Each weather needs an integer, positive quota per global batch")
            self.quotas = {w: round(q) for w, q in sorted(quotas.items())}

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.samples_per_epoch // self.batch_size

    def __iter__(self):
        if self.stratified_cycles:
            yield from self._stratified_iterator()
            return
        rng = random.Random(self.seed + self.epoch)
        augmentation_rng = random.Random(f"augmentation/{self.seed}/{self.epoch}")

        def cycle(values):
            values = list(values)
            while True:
                rng.shuffle(values)
                yield from values

        if self.quotas:
            groups = defaultdict(lambda: defaultdict(list))
            for i, r in enumerate(self.records):
                groups[r.weather][r.group_id].append(i)
            group_cycles = {w: cycle(sorted(g)) for w, g in groups.items()}
            variants = {
                (w, g): cycle(indices) for w, gs in groups.items() for g, indices in gs.items()
            }
        else:
            indices = cycle(range(len(self.records)))
        local_size = self.batch_size // self.world_size
        for _ in range(len(self)):
            if self.quotas:
                batch = [
                    next(variants[w, next(group_cycles[w])])
                    for w, quota in self.quotas.items()
                    for _ in range(quota)
                ]
            elif self.weights:
                batch = rng.choices(
                    range(len(self.records)), weights=self.weights, k=self.batch_size
                )
            else:
                batch = [next(indices) for _ in range(self.batch_size)]
            rng.shuffle(batch)
            if self.seeded_augmentations:
                batch = [(i, augmentation_rng.getrandbits(64)) for i in batch]
            start = self.rank * local_size
            yield batch[start : start + local_size]

    def _stratified_iterator(self):
        """Continuous permutation streams; epoch boundaries do not restart coverage."""
        if self.epoch < 1:
            raise ValueError("Stratified sampling epochs start at one")
        groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for i, r in enumerate(self.records):
            groups[r.weather][r.group_id][r.stratum].append(i)
        group_keys = {w: sorted(g) for w, g in groups.items()}
        stratum_keys = {(w, g): sorted(s) for w, gs in groups.items() for g, s in gs.items()}
        permutations = {}

        def seed_for(*parts):
            message = ":".join(map(str, ("cycles-v1", self.seed, *parts)))
            return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], "big")

        def pick(values, position, namespace):
            cycle, offset = divmod(position, len(values))
            key = namespace, cycle
            if key not in permutations:
                permutation = list(values)
                random.Random(seed_for(namespace, cycle)).shuffle(permutation)
                permutations[key] = permutation
            return permutations[key][offset], cycle

        local_size = self.batch_size // self.world_size
        shuffle = random.Random(seed_for("batch-order", self.epoch))
        for batch_index in range(len(self)):
            batch = []
            for weather, quota in self.quotas.items():
                draws_per_epoch = len(self) * quota
                for j in range(quota):
                    draw = (self.epoch - 1) * draws_per_epoch + batch_index * quota + j
                    group, visit = pick(group_keys[weather], draw, weather)
                    strata = groups[weather][group]
                    stratum, stratum_visit = pick(
                        stratum_keys[weather, group], visit, f"{weather}/{group}"
                    )
                    index, _ = pick(strata[stratum], stratum_visit, f"{weather}/{group}/{stratum}")
                    batch.append((index, seed_for("augmentation", weather, draw)))
            shuffle.shuffle(batch)
            start = self.rank * local_size
            yield batch[start : start + local_size]
