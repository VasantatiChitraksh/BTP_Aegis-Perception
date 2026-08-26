#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from aegis_perception.data.manifest import PairRecord, deterministic_split, write_manifest

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic paired-image manifest")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weather", required=True)
    parser.add_argument("--source", default="local")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--group-regex",
        help="Regex applied to sample IDs; group 1 is kept together across splits",
    )
    return parser.parse_args()


def index_images(root: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            key = path.relative_to(root).with_suffix("").as_posix()
            if key in indexed:
                raise ValueError(f"Duplicate pair key {key!r} below {root}")
            indexed[key] = path.resolve()
    return indexed


def group_id(sample_id: str, expression: str | None) -> str:
    if not expression:
        return sample_id
    match = re.search(expression, sample_id)
    if not match or not match.groups():
        raise ValueError(f"group regex did not capture a group for {sample_id!r}")
    return match.group(1)


def main() -> None:
    args = parse_args()
    inputs = index_images(args.input_root)
    targets = index_images(args.target_root)
    missing_targets = sorted(inputs.keys() - targets.keys())
    missing_inputs = sorted(targets.keys() - inputs.keys())
    if missing_targets or missing_inputs:
        details = []
        if missing_targets:
            details.append(
                f"{len(missing_targets)} inputs lack targets; first={missing_targets[:5]}"
            )
        if missing_inputs:
            details.append(f"{len(missing_inputs)} targets lack inputs; first={missing_inputs[:5]}")
        raise SystemExit("Pairing failed: " + "; ".join(details))
    if not inputs:
        raise SystemExit("No image pairs found")

    records = []
    for sample_id in sorted(inputs):
        split = deterministic_split(
            group_id(sample_id, args.group_regex),
            seed=args.seed,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
        )
        records.append(
            PairRecord(
                sample_id=sample_id,
                input_path=inputs[sample_id],
                target_path=targets[sample_id],
                split=split,
                weather=args.weather,
                source=args.source,
            )
        )
    write_manifest(args.output, records)
    counts = {
        split: sum(record.split == split for record in records)
        for split in ("train", "val", "test")
    }
    print(f"wrote {len(records)} pairs to {args.output}: {counts}")


if __name__ == "__main__":
    main()
