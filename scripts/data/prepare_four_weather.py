#!/usr/bin/env python3
"""Freeze repaired manifests without modifying original CSVs or image files."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))
from aegis_perception.data.manifest import PairRecord, load_manifest, write_manifest


def image_metadata(path: Path) -> dict:
    """Decode all pixels; hash RGB content independently of file encoding."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        pixels = image.tobytes()
        content = hashlib.sha256(str(image.size).encode() + pixels).hexdigest()
        gray = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float32)
        spectrum = dctn(gray, norm="ortho")[:8, :8].flatten()
        bits = spectrum > np.median(spectrum[1:])
        phash = sum(int(bit) << index for index, bit in enumerate(bits))
        thumb = image.resize((16, 16), Image.Resampling.BILINEAR).tobytes()
        return {
            "size": list(image.size),
            "pixels_sha256": content,
            "phash": f"{phash:016x}",
            "thumbnail": base64.b64encode(thumb).decode(),
        }


def load_fog() -> list[PairRecord]:
    records = []
    for split in ("train", "test"):
        root = REPOSITORY / f"data/raw/fog/reside6k_{split}/{split}"
        inputs = {p.name: p for p in (root / "hazy").glob("*") if p.is_file()}
        targets = {p.name: p for p in (root / "GT").glob("*") if p.is_file()}
        if not inputs or inputs.keys() != targets.keys():
            raise ValueError(f"RESIDE-6K {split} missing or incorrectly paired")
        records.extend(
            PairRecord(f"{split}/{name}", inputs[name], targets[name], split, "fog", "reside6k")
            for name in sorted(inputs)
        )
    return records


def split_groups(groups: set[str], seed: int, test_fraction: float) -> dict[str, str]:
    ordered = sorted(groups, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).digest())
    n_test = max(1, round(len(ordered) * test_fraction)) if test_fraction else 0
    n_val = max(1, round(len(ordered) * 0.15))
    if len(ordered) <= n_test + n_val:
        raise ValueError("Insufficient independent groups for train/validation/test")
    return {
        g: "test" if i < n_test else "val" if i < n_test + n_val else "train"
        for i, g in enumerate(ordered)
    }


def remove_split_overlap(records: list[PairRecord]) -> tuple[list[PairRecord], list[dict]]:
    """Keep the highest-priority held-out partition; exclude its training copies."""
    priority = {"train": 0, "val": 1, "test": 2}
    selected = {}
    for record in records:
        current = selected.get(record.group_id, "train")
        if priority[record.split] >= priority[current]:
            selected[record.group_id] = record.split
    kept, dropped = [], []
    for record in records:
        if record.split == selected[record.group_id]:
            kept.append(record)
        else:
            dropped.append(
                {
                    "source": record.source,
                    "sample_id": record.sample_id,
                    "split": record.split,
                    "kept_split": selected[record.group_id],
                    "group_id": record.group_id,
                }
            )
    return kept, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = REPOSITORY / "data/manifests/clean"
    output.mkdir(parents=True, exist_ok=True)
    artifacts = REPOSITORY / "artifacts/data/four_weather"
    artifacts.mkdir(parents=True, exist_ok=True)
    cache_file = artifacts / "image_cache.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    names = [
        "realrain1k",
        "snow_cityscapes",
        "csd",
        "smokebench",
        "smoke_historical",
        "o_haze",
        "rain100l",
    ]
    datasets = {name: load_manifest(REPOSITORY / f"data/manifests/{name}.csv") for name in names}
    datasets["reside6k"] = load_fog()
    paths = sorted(
        {p for rows in datasets.values() for r in rows for p in (r.input_path, r.target_path)}
    )

    def inspect(path):
        relative = path.relative_to(REPOSITORY).as_posix()
        stat = path.stat()
        signature = [stat.st_size, stat.st_mtime_ns]
        old = cache.get(relative)
        if old and old["signature"] == signature:
            return relative, old
        return relative, {"signature": signature, **image_metadata(path)}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (path, entry) in enumerate(pool.map(inspect, paths), 1):
            cache[path] = entry
            if i % 2000 == 0:
                print(f"Decoded/checked {i}/{len(paths)} images", flush=True)
                cache_file.write_text(json.dumps(cache))
    cache_file.write_text(json.dumps(cache))

    def metadata(path):
        return cache[path.relative_to(REPOSITORY).as_posix()]

    representatives = {}
    for rows in datasets.values():
        for r in rows:
            if metadata(r.input_path)["size"] != metadata(r.target_path)["size"]:
                raise ValueError(f"Pair dimensions differ: {r.source}/{r.sample_id}")
            representatives.setdefault(metadata(r.target_path)["pixels_sha256"], r.target_path)
    parents = {key: key for key in representatives}

    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    # Conservative near-copy grouping: pHash distance <=2 AND RGB thumbnail MAE <=2/255.
    # Candidate search uses three disjoint hash chunks: distance <=2 must share one chunk.
    buckets = defaultdict(set)
    near_copies = []
    for key, path in sorted(representatives.items()):
        info = metadata(path)
        phash = int(info["phash"], 16)
        candidate_ids = set()
        chunks = [
            (0, phash & ((1 << 22) - 1)),
            (1, (phash >> 22) & ((1 << 21) - 1)),
            (2, phash >> 43),
        ]
        for chunk in chunks:
            candidate_ids.update(buckets[chunk])
        thumb = np.frombuffer(base64.b64decode(info["thumbnail"]), dtype=np.uint8).astype(float)
        for other in sorted(candidate_ids):
            candidate = metadata(representatives[other])
            if (phash ^ int(candidate["phash"], 16)).bit_count() > 2:
                continue
            a, b = info["size"], candidate["size"]
            if abs(a[0] / a[1] - b[0] / b[1]) > 0.01:
                continue
            other_thumb = np.frombuffer(
                base64.b64decode(candidate["thumbnail"]), dtype=np.uint8
            ).astype(float)
            mae = float(np.abs(thumb - other_thumb).mean())
            if mae <= 2.0:
                left, right = sorted((find(key), find(other)))
                parents[right] = left
                near_copies.append(
                    {
                        "first": str(path.relative_to(REPOSITORY)),
                        "second": str(representatives[other].relative_to(REPOSITORY)),
                        "thumbnail_mae_255": mae,
                    }
                )
        for chunk in chunks:
            buckets[chunk].add(key)

    repaired = {}
    for name, rows in datasets.items():
        rows = [replace(r, group_id=find(metadata(r.target_path)["pixels_sha256"])) for r in rows]
        if name in {"csd", "smokebench"}:
            assignment = split_groups({r.group_id for r in rows}, args.seed, 0.15)
            rows = [replace(r, split=assignment[r.group_id]) for r in rows]
        elif name == "reside6k":
            assignment = split_groups(
                {r.group_id for r in rows if r.split == "train"}, args.seed, 0.0
            )
            rows = [
                replace(r, split=assignment[r.group_id]) if r.split == "train" else r for r in rows
            ]
        repaired[name] = rows
    all_rows, excluded = remove_split_overlap([r for rows in repaired.values() for r in rows])
    retained = {(r.source, r.sample_id) for r in all_rows}
    report = {
        "seed": args.seed,
        "images_checked": len(paths),
        "input_manifest_sha256": {
            name: hashlib.sha256(
                (REPOSITORY / f"data/manifests/{name}.csv").read_bytes()
            ).hexdigest()
            for name in names
        },
        "protocol": "decoded-RGB identity plus conservative near-copy groups",
        "near_copy_rule": "pHash Hamming <=2; aspect difference <=0.01; 16x16 RGB MAE <=2/255",
        "limitations": "Does not certify all scene/sequence relationships or pair alignment.",
        "custom_resplit": ["csd", "smokebench"],
        "excluded": excluded,
        "near_copies": near_copies,
        "datasets": {},
    }
    for name, rows in repaired.items():
        rows = [r for r in rows if (r.source, r.sample_id) in retained]
        repaired[name] = rows
        target = output / f"{name}.csv"
        write_manifest(
            target,
            [
                replace(
                    r,
                    input_path=Path(os.path.relpath(r.input_path, output)),
                    target_path=Path(os.path.relpath(r.target_path, output)),
                )
                for r in rows
            ],
        )
        report["datasets"][name] = {
            "pairs": dict(Counter(r.split for r in rows)),
            "groups": {
                s: len({r.group_id for r in rows if r.split == s}) for s in ("train", "val", "test")
            },
            "manifest_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
        print(name, report["datasets"][name], flush=True)

    # Freeze a small, scene-balanced validation panel. Full validation remains in clean/*.csv.
    joint = []
    for name in ("realrain1k", "reside6k", "snow_cityscapes", "smokebench"):
        rows = repaired[name]
        validation = defaultdict(list)
        for r in rows:
            if r.split == "val":
                validation[r.group_id].append(r)
            else:
                joint.append(r)
        ordered = sorted(
            validation, key=lambda g: hashlib.sha256(f"panel:{args.seed}:{g}".encode()).digest()
        )
        variants_per_group = 8 if name == "smokebench" else 2
        for group in ordered[:64]:
            variants = sorted(
                validation[group],
                key=lambda r: hashlib.sha256(
                    f"variant:{args.seed}:{r.sample_id}".encode()
                ).digest(),
            )
            joint.extend(variants[:variants_per_group])
    joint = [
        replace(
            r,
            sample_id=f"{r.source}/{r.sample_id}",
            input_path=Path(os.path.relpath(r.input_path, output)),
            target_path=Path(os.path.relpath(r.target_path, output)),
        )
        for r in joint
    ]
    if {r.weather for r in joint if r.split == "train"} != {"rain", "fog", "snow", "smoke"}:
        raise ValueError("All four weather types must have training data")
    write_manifest(output / "four_weather.csv", joint)
    report["joint_counts"] = {
        s: dict(Counter(r.weather for r in joint if r.split == s)) for s in ("train", "val", "test")
    }
    report["joint_manifest_sha256"] = hashlib.sha256(
        (output / "four_weather.csv").read_bytes()
    ).hexdigest()
    (artifacts / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Combined:", report["joint_counts"], flush=True)


if __name__ == "__main__":
    main()
