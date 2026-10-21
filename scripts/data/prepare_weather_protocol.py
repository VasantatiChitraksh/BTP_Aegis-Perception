#!/usr/bin/env python3
"""Freeze v3: deduplicated train, broad validation, compact clustered test."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aegis_perception.data.catalog import sha256
from aegis_perception.data.manifest import load_manifest, write_manifest
from aegis_perception.reproducibility import write_json

SPLIT_SEED = 20261015


def ordered(values, key=str):
    return sorted(
        values, key=lambda v: hashlib.sha256(f"v3:{SPLIT_SEED}:{key(v)}".encode()).digest()
    )


def stratify(rows, scores, bins=4):
    """Within-background raw/target thumbnail MAE ranks; not physical density."""
    groups = defaultdict(list)
    for row in rows:
        groups[row.group_id].append(row)
    result = []
    for variants in groups.values():
        variants.sort(key=lambda r: (scores[r.sample_id], r.sample_id))
        count = min(bins, len(variants))
        for i, row in enumerate(variants):
            result.append(replace(row, stratum=f"q{min(count - 1, i * count // len(variants))}"))
    return result


def cap_strata(rows, per_stratum):
    groups = defaultdict(list)
    for row in rows:
        groups[row.group_id, row.stratum].append(row)
    return [
        r
        for key in sorted(groups)
        for r in ordered(groups[key], key=lambda r: r.sample_id)[:per_stratum]
    ]


def snow_panel(rows, groups_requested=256):
    """One background per known sequence, city-stratified sequence allocation."""
    cities = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        city = row.target_path.name.split("_")[0]
        cities[city][row.cluster_id][row.group_id].append(row)
    available = sum(len(v) for v in cities.values())
    if not len(cities) <= groups_requested <= available:
        raise ValueError("Invalid number of snow sequence groups requested")
    # At least one sequence per city, then largest quota deficit. No metric scores.
    quotas = dict.fromkeys(cities, 1)
    while sum(quotas.values()) < groups_requested:
        city = max(
            (c for c in sorted(cities) if quotas[c] < len(cities[c])),
            key=lambda c: groups_requested * len(cities[c]) / available - quotas[c],
        )
        quotas[city] += 1
    result = []
    for city in sorted(cities):
        for cluster in ordered(cities[city])[: quotas[city]]:
            groups = cities[city][cluster]
            group = ordered(groups)[0]
            # Normally three snow strengths; exact target aliases can give six.
            result.extend(cap_strata(groups[group], 1))
    return result


def prepare(output_dir: Path):
    if output_dir.exists():
        raise ValueError("Protocol output exists; frozen artifacts must not be overwritten")
    cache_path = ROOT / "artifacts/data/four_weather/image_cache.json"
    cache = json.loads(cache_path.read_text())
    seen_paths = set()

    def metadata(path):
        relative = path.relative_to(ROOT).as_posix()
        info = cache[relative]
        if path not in seen_paths:
            stat = path.stat()
            if info["signature"] != [stat.st_size, stat.st_mtime_ns]:
                raise ValueError(f"Stale image audit cache: {relative}")
            seen_paths.add(path)
        return info

    inputs = [
        ROOT / f"data/manifests/clean/{name}.csv"
        for name in ("realrain1k", "reside6k", "snow_cityscapes", "smokebench")
    ]
    rows = [
        replace(r, sample_id=f"{r.source}/{r.sample_id}") for p in inputs for r in load_manifest(p)
    ]
    group_splits = defaultdict(set)
    scores, decoded_pairs, unique, duplicates = {}, {}, [], []
    for r in sorted(rows, key=lambda r: r.sample_id):
        inp, gt = metadata(r.input_path), metadata(r.target_path)
        if inp["size"] != gt["size"]:
            raise ValueError(f"Unaligned dimensions: {r.sample_id}")
        group_splits[r.group_id].add(r.split)
        pair = inp["pixels_sha256"], gt["pixels_sha256"]
        if pair in decoded_pairs:
            previous = decoded_pairs[pair]
            if r.split != previous.split:
                raise ValueError("Identical decoded pair crosses partitions")
            duplicates.append(
                {
                    "excluded": r.sample_id,
                    "retained": previous.sample_id,
                    "split": r.split,
                    "weather": r.weather,
                }
            )
            continue
        decoded_pairs[pair] = r
        a, b = [
            np.frombuffer(base64.b64decode(info["thumbnail"]), dtype=np.uint8).astype(np.float32)
            for info in (inp, gt)
        ]
        scores[r.sample_id] = float(np.abs(a - b).mean() / 255)
        unique.append(r)
    if any(len(v) != 1 for v in group_splits.values()):
        raise ValueError("Input background groups cross partitions")

    # Union known Cityscapes sequence IDs and decoded-background aliases.
    parent = {r.group_id: r.group_id for r in unique}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    sequences = {}
    for r in unique:
        match = re.match(r"^([a-z]+_\d{6})_\d{6}_leftImg8bit", r.target_path.stem, re.I)
        if r.weather == "snow" and match:
            sequence = match[1].lower()
            if sequence in sequences:
                a, b = find(r.group_id), find(sequences[sequence])
                parent[max(a, b)] = min(a, b)
            sequences[sequence] = r.group_id
    unique = [replace(r, cluster_id=find(r.group_id)) for r in unique]
    cluster_splits = defaultdict(set)
    for r in unique:
        cluster_splits[r.cluster_id].add(r.split)
    if any(len(v) != 1 for v in cluster_splits.values()):
        raise ValueError("Known sequence cluster crosses partitions")

    annotated = []
    for weather in ("rain", "fog", "snow", "smoke"):
        for split in ("train", "val", "test"):
            subset = [r for r in unique if (r.weather, r.split) == (weather, split)]
            # Snow has three known variants per target; merge aliases before ranking.
            annotated.extend(stratify(subset, scores, bins=3 if weather == "snow" else 4))
    active = []
    for weather in ("rain", "fog", "snow", "smoke"):
        for split in ("train", "val", "test"):
            subset = [r for r in annotated if (r.weather, r.split) == (weather, split)]
            if weather == "smoke" and split != "train":
                subset = cap_strata(subset, 16)  # up to 64 observations/background
            elif weather == "snow" and split == "test":
                subset = snow_panel(subset)
            active.extend(subset)
    active_ids = {r.sample_id for r in active}
    reserve = [r for r in annotated if r.sample_id not in active_ids]
    # Fine-tuning is safe only if the initial model never trained on our held-out targets.
    original = load_manifest(ROOT / "data/manifests/clean/four_weather.csv")
    previous_train = {r.group_id for r in original if r.split == "train"}
    if previous_train & {r.group_id for r in active if r.split != "train"}:
        raise ValueError("Initialization training groups overlap the new held-out data")
    output_dir.mkdir(parents=True)

    def write(name, records):
        path = output_dir / name
        write_manifest(
            path,
            [
                replace(
                    r,
                    input_path=Path(os.path.relpath(r.input_path, output_dir)),
                    target_path=Path(os.path.relpath(r.target_path, output_dir)),
                )
                for r in sorted(records, key=lambda r: r.sample_id)
            ],
        )
        return sha256(path)

    hashes = {
        "active.csv": write("active.csv", active),
        "reserve.csv": write("reserve.csv", reserve),
    }
    counts = {}
    for weather in ("rain", "fog", "snow", "smoke"):
        counts[weather] = {}
        for split in ("train", "val", "test"):
            subset = [r for r in active if (r.weather, r.split) == (weather, split)]
            counts[weather][split] = {
                "images": len(subset),
                "backgrounds": len({r.group_id for r in subset}),
                "clusters": len({r.cluster_id for r in subset}),
            }
    report = {
        "protocol": "weather-v3",
        "split_seed": SPLIT_SEED,
        "training_seeds": [42, 43, 44],
        "input_manifests": {str(p.relative_to(ROOT)): sha256(p) for p in inputs},
        "cache_sha256": sha256(cache_path),
        "manifest_sha256": hashes,
        "counts": counts,
        "duplicate_pairs_excluded": duplicates,
        "reserve": dict(Counter(f"{r.weather}/{r.split}" for r in reserve)),
        "strata": (
            "Within-background thumbnail raw/target MAE rank bins; proxy, not physical density"
        ),
        "aggregation": "equal weather, background, degradation-rank stratum, then images",
        "snow_test": "256 known sequence clusters, city-proportional quotas, one background each",
        "limitations": [
            "Smoke has six validation and six test backgrounds; "
            "few-cluster intervals are exploratory.",
            "Renamed snow training files and rain files do not identify capture sequences.",
            "Decoded/near-copy grouping cannot certify all semantic scene "
            "or sequence independence.",
            "Test images were evaluated previously; this is not an untouched external benchmark.",
            "Training seeds vary fine-tuning from a shared seed-42 initialization.",
        ],
    }
    write_json(output_dir / "protocol.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/manifests/weather_v3")
    args = parser.parse_args()
    report = prepare(args.output_dir)
    print(
        json.dumps(
            {
                "counts": report["counts"],
                "reserve": report["reserve"],
                "exact_pairs_removed": len(report["duplicate_pairs_excluded"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
