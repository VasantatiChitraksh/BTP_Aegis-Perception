#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from aegis_perception.data.manifest import VALID_SPLITS, load_manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit paired data and detect split leakage")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--skip-hashes", action="store_true", help="Skip duplicate-target hashing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_manifest(args.manifest)
    errors: list[str] = []
    warnings: list[str] = []
    ids: set[str] = set()
    sizes: Counter[tuple[int, int]] = Counter()
    split_counts = Counter(record.split for record in records)
    weather_counts = Counter(record.weather for record in records)
    target_hash_splits: dict[str, set[str]] = defaultdict(set)

    for record in records:
        if record.sample_id in ids:
            errors.append(f"duplicate sample_id: {record.sample_id}")
        ids.add(record.sample_id)
        if record.split not in VALID_SPLITS:
            errors.append(f"invalid split {record.split!r}: {record.sample_id}")
        for role, path in (("input", record.input_path), ("target", record.target_path)):
            if not path.is_file():
                errors.append(f"missing {role}: {path}")
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                if role == "input":
                    with Image.open(path) as image:
                        sizes[image.size] += 1
            except Exception as exc:
                errors.append(f"invalid image {path}: {exc}")
        if not args.skip_hashes and record.target_path.is_file():
            target_hash_splits[sha256(record.target_path)].add(record.split)

    leaked = [splits for splits in target_hash_splits.values() if len(splits) > 1]
    if leaked:
        errors.append(f"{len(leaked)} clean target file(s) occur across multiple splits")
    for required in VALID_SPLITS:
        if split_counts[required] == 0:
            warnings.append(f"split {required!r} is empty")

    print(f"records: {len(records)}")
    print(f"splits: {dict(sorted(split_counts.items()))}")
    print(f"weather: {dict(sorted(weather_counts.items()))}")
    print(f"input sizes (top 10): {sizes.most_common(10)}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors[:50]:
        print(f"ERROR: {error}")
    if len(errors) > 50:
        print(f"ERROR: ... and {len(errors) - 50} more")
    if errors:
        raise SystemExit(1)
    print("manifest audit passed")


if __name__ == "__main__":
    main()
