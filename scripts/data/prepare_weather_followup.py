#!/usr/bin/env python3
"""Create a follow-up validation manifest and an equal-weather test panel."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aegis_perception.data.catalog import sha256
from aegis_perception.data.manifest import load_manifest, write_manifest
from aegis_perception.evaluation import balanced_panel
from aegis_perception.reproducibility import write_json


def prepare(source: Path, smoke: Path, output: Path, panel_path: Path, per_weather=224):
    records = load_manifest(source)
    if output.exists() or panel_path.exists():
        raise ValueError("Follow-up outputs already exist; keep frozen or choose new paths")
    smoke_val = [r for r in load_manifest(smoke, split="val")]
    groups = defaultdict(list)
    for row in smoke_val:
        groups[row.group_id].append(row)
    heldout = {r.group_id for r in records if r.split == "val" and r.weather == "smoke"}
    if not heldout or set(groups) != heldout:
        raise ValueError("Smoke validation backgrounds differ from the original protocol")
    selected_smoke = []
    for _group, rows in sorted(groups.items()):
        rows = sorted(
            rows, key=lambda r: hashlib.sha256(f"variant:42:{r.sample_id}".encode()).digest()
        )
        selected_smoke.extend(replace(r, sample_id=f"{r.source}/{r.sample_id}") for r in rows[:32])
    updated = [r for r in records if not (r.split == "val" and r.weather == "smoke")]
    updated.extend(selected_smoke)
    old_validation = {r.sample_id for r in records if r.split == "val"}
    if not old_validation <= {r.sample_id for r in updated if r.split == "val"}:
        raise ValueError("Expanded validation must contain the original panel")
    split_groups = defaultdict(set)
    for row in updated:
        split_groups[row.group_id].add(row.split)
    if any(len(splits) != 1 for splits in split_groups.values()):
        raise ValueError("Background leakage in follow-up manifest")
    for split in ("train", "test"):
        if {r for r in records if r.split == split} != {r for r in updated if r.split == split}:
            raise ValueError(f"The existing {split} split must stay unchanged")
    panel = balanced_panel([r for r in records if r.split == "test"], per_weather)
    write_manifest(
        output,
        [
            replace(
                r,
                input_path=Path(os.path.relpath(r.input_path, output.parent)),
                target_path=Path(os.path.relpath(r.target_path, output.parent)),
            )
            for r in updated
        ],
    )
    write_json(
        panel_path,
        {
            "seed": 42,
            "split": "test",
            "per_weather": per_weather,
            "selection": "ID-hash order, round-robin over backgrounds; no model scores used",
            "source_manifest_sha256": sha256(source),
            "compatible_manifest_sha256": [sha256(source), sha256(output)],
            "sample_ids": [r.sample_id for r in panel],
            "counts": dict(Counter(r.weather for r in panel)),
            "background_groups": {
                w: len({r.group_id for r in panel if r.weather == w})
                for w in sorted({r.weather for r in panel})
            },
        },
    )
    return {
        s: dict(Counter(r.weather for r in updated if r.split == s))
        for s in ("train", "val", "test")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("data/manifests/clean/four_weather.csv")
    )
    parser.add_argument("--smoke", type=Path, default=Path("data/manifests/clean/smokebench.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/manifests/clean/four_weather_v2.csv")
    )
    parser.add_argument(
        "--panel", type=Path, default=Path("data/manifests/clean/test_panel_224.json")
    )
    args = parser.parse_args()
    print(prepare(args.source, args.smoke, args.output, args.panel))


if __name__ == "__main__":
    main()
