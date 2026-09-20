#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Callable

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from aegis_perception.data.manifest import PairRecord, write_manifest

RAW = REPOSITORY / "data" / "raw"
MANIFESTS = REPOSITORY / "data" / "manifests"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build source-aware portable manifests for the reviewed collection"
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=(
            "smoke_historical",
            "smokebench",
            "rain100l",
            "realrain1k",
            "o_haze",
            "csd",
            "snow_cityscapes",
        ),
        help="Build only this dataset; may be repeated",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    return parser.parse_args()


def image_index(root: Path, key: Callable[[Path], str] | None = None) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    indexed: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        pair_key = key(path) if key else path.relative_to(root).with_suffix("").as_posix()
        if pair_key in indexed:
            raise ValueError(f"duplicate pair key {pair_key!r} below {root}")
        indexed[pair_key] = path.resolve()
    return indexed


def train_or_val(group: str, *, source: str, seed: int, val_fraction: float) -> str:
    if not 0 <= val_fraction < 1:
        raise ValueError("val-fraction must be in [0, 1)")
    digest = hashlib.sha256(f"{seed}:{source}:{group}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return "val" if value < val_fraction else "train"


def portable(path: Path, manifest: Path) -> Path:
    return Path(os.path.relpath(path.resolve(), manifest.resolve().parent))


def record(
    *,
    manifest: Path,
    sample_id: str,
    input_path: Path,
    target_path: Path,
    split: str,
    weather: str,
    source: str,
) -> PairRecord:
    return PairRecord(
        sample_id=sample_id,
        input_path=portable(input_path, manifest),
        target_path=portable(target_path, manifest),
        split=split,
        weather=weather,
        source=source,
    )


def add_exact_pairs(
    records: list[PairRecord],
    *,
    manifest: Path,
    input_root: Path,
    target_root: Path,
    split: str | Callable[[str], str],
    weather: str,
    source: str,
    prefix: str,
) -> None:
    inputs = image_index(input_root)
    targets = image_index(target_root)
    if inputs.keys() != targets.keys():
        missing_targets = sorted(inputs.keys() - targets.keys())
        missing_inputs = sorted(targets.keys() - inputs.keys())
        raise ValueError(
            f"{source} pairing mismatch: inputs without targets={missing_targets[:5]}, "
            f"targets without inputs={missing_inputs[:5]}"
        )
    for pair_key in sorted(inputs):
        record_split = split(pair_key) if callable(split) else split
        records.append(
            record(
                manifest=manifest,
                sample_id=f"{prefix}/{pair_key}",
                input_path=inputs[pair_key],
                target_path=targets[pair_key],
                split=record_split,
                weather=weather,
                source=source,
            )
        )


def build_smoke_historical(seed: int, val_fraction: float) -> tuple[Path, list[PairRecord]]:
    source = "smoke_historical"
    root = RAW / "smoke" / "historical"
    manifest = MANIFESTS / f"{source}.csv"
    records: list[PairRecord] = []
    add_exact_pairs(
        records,
        manifest=manifest,
        input_root=root / "train" / "hazy",
        target_root=root / "train" / "clean",
        split=lambda key: train_or_val(
            key, source=source, seed=seed, val_fraction=val_fraction
        ),
        weather="smoke",
        source=source,
        prefix="train",
    )
    add_exact_pairs(
        records,
        manifest=manifest,
        input_root=root / "test" / "hazy",
        target_root=root / "test" / "clean",
        split="test",
        weather="smoke",
        source=source,
        prefix="test",
    )
    return manifest, records


def build_smokebench(seed: int, val_fraction: float) -> tuple[Path, list[PairRecord]]:
    source = "smokebench_v2"
    root = RAW / "smoke" / "smokebench" / "dataset_version2"
    manifest = MANIFESTS / "smokebench.csv"
    records: list[PairRecord] = []
    add_exact_pairs(
        records,
        manifest=manifest,
        input_root=root / "Train" / "LQ",
        target_root=root / "Train" / "GT",
        split=lambda key: train_or_val(
            key, source=source, seed=seed, val_fraction=val_fraction
        ),
        weather="smoke",
        source=source,
        prefix="train",
    )
    add_exact_pairs(
        records,
        manifest=manifest,
        input_root=root / "Test" / "LQ",
        target_root=root / "Test" / "GT",
        split="test",
        weather="smoke",
        source=source,
        prefix="test",
    )
    return manifest, records


def build_rain100l(seed: int, val_fraction: float) -> tuple[Path, list[PairRecord]]:
    del seed, val_fraction
    source = "rain100l"
    root = RAW / "rain" / source / "Rain100L"
    manifest = MANIFESTS / f"{source}.csv"
    records: list[PairRecord] = []
    add_exact_pairs(
        records,
        manifest=manifest,
        input_root=root / "input",
        target_root=root / "target",
        split="test",
        weather="rain",
        source=source,
        prefix="test",
    )
    return manifest, records


def build_realrain1k(seed: int, val_fraction: float) -> tuple[Path, list[PairRecord]]:
    del seed, val_fraction
    source = "realrain1k_h"
    root = RAW / "rain" / "realrain1k" / "RealRain-1k" / "RealRain-1k-H"
    manifest = MANIFESTS / "realrain1k.csv"
    records: list[PairRecord] = []
    for native_split, split in (("train", "train"), ("validation", "val"), ("test", "test")):
        add_exact_pairs(
            records,
            manifest=manifest,
            input_root=root / native_split / "input",
            target_root=root / native_split / "target",
            split=split,
            weather="rain",
            source=source,
            prefix=native_split,
        )
    return manifest, records


def o_haze_key(path: Path) -> str:
    match = re.fullmatch(r"(\d+)_outdoor_(?:hazy|GT)", path.stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"unexpected O-HAZE filename: {path.name}")
    return match.group(1)


def build_o_haze(seed: int, val_fraction: float) -> tuple[Path, list[PairRecord]]:
    del seed, val_fraction
    source = "o_haze"
    root = RAW / "fog" / source / "# O-HAZY NTIRE 2018"
    manifest = MANIFESTS / f"{source}.csv"
    inputs = image_index(root / "hazy", o_haze_key)
    targets = image_index(root / "GT", o_haze_key)
    if inputs.keys() != targets.keys():
        raise ValueError("O-HAZE input/target scene IDs do not match")
    records = [
        record(
            manifest=manifest,
            sample_id=f"test/{pair_key}",
            input_path=inputs[pair_key],
            target_path=targets[pair_key],
            split="test",
            weather="fog",
            source=source,
        )
        for pair_key in sorted(inputs)
    ]
    return manifest, records


def build_snow_cityscapes(seed: int, val_fraction: float) -> tuple[Path, list[PairRecord]]:
    source = "snow_cityscapes"
    root = RAW / "snow" / source / "Cityscape-Dataset"
    manifest = MANIFESTS / f"{source}.csv"
    train_inputs = image_index(root / "train" / "synthetic")
    train_targets = image_index(root / "train" / "gt")
    records: list[PairRecord] = []
    for input_key, input_path in sorted(train_inputs.items()):
        match = re.fullmatch(r"cityscape-(\d+)", input_key)
        if not match:
            raise ValueError(f"unexpected SnowCityscapes train key: {input_key}")
        variant_index = int(match.group(1))
        scene_index = ((variant_index - 1) % 2000) + 1
        target_key = f"cityscape-{scene_index}"
        if target_key not in train_targets:
            raise ValueError(f"missing SnowCityscapes target: {target_key}")
        records.append(
            record(
                manifest=manifest,
                sample_id=f"train/{input_key}",
                input_path=input_path,
                target_path=train_targets[target_key],
                split=train_or_val(
                    target_key, source=source, seed=seed, val_fraction=val_fraction
                ),
                weather="snow",
                source=source,
            )
        )
    test_targets = image_index(root / "test" / "gt")
    for severity in ("smallSnow", "mediumSnow", "largeSnow"):
        inputs = image_index(root / "test" / severity)
        if inputs.keys() != test_targets.keys():
            raise ValueError(f"SnowCityscapes {severity} input/target scene IDs do not match")
        for pair_key in sorted(inputs):
            records.append(
                record(
                    manifest=manifest,
                    sample_id=f"test/{severity}/{pair_key}",
                    input_path=inputs[pair_key],
                    target_path=test_targets[pair_key],
                    split="test",
                    weather="snow",
                    source=source,
                )
            )
    return manifest, records


def build_csd(seed: int, val_fraction: float) -> tuple[Path, list[PairRecord]]:
    source = "csd"
    root = RAW / "snow" / source
    manifest = MANIFESTS / f"{source}.csv"
    records: list[PairRecord] = []
    add_exact_pairs(
        records,
        manifest=manifest,
        input_root=root / "Train" / "Snow",
        target_root=root / "Train" / "Gt",
        split=lambda key: train_or_val(
            key, source=source, seed=seed, val_fraction=val_fraction
        ),
        weather="snow",
        source=source,
        prefix="train",
    )
    add_exact_pairs(
        records,
        manifest=manifest,
        input_root=root / "Test" / "Snow",
        target_root=root / "Test" / "Gt",
        split="test",
        weather="snow",
        source=source,
        prefix="test",
    )
    return manifest, records


BUILDERS = {
    "smoke_historical": build_smoke_historical,
    "smokebench": build_smokebench,
    "rain100l": build_rain100l,
    "realrain1k": build_realrain1k,
    "o_haze": build_o_haze,
    "csd": build_csd,
    "snow_cityscapes": build_snow_cityscapes,
}


def main() -> None:
    args = parse_args()
    requested = args.dataset or list(BUILDERS)
    for dataset_id in requested:
        manifest, records = BUILDERS[dataset_id](args.seed, args.val_fraction)
        write_manifest(manifest, records)
        counts = {
            split: sum(item.split == split for item in records)
            for split in ("train", "val", "test")
        }
        print(f"wrote {len(records)} pairs to {manifest}: {counts}")


if __name__ == "__main__":
    main()
