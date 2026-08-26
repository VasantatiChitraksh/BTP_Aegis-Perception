from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

VALID_SPLITS = {"train", "val", "test"}
REQUIRED_COLUMNS = {"sample_id", "input_path", "target_path", "split", "weather", "source"}


@dataclass(frozen=True)
class PairRecord:
    sample_id: str
    input_path: Path
    target_path: Path
    split: str
    weather: str
    source: str


def _resolve(path_value: str, manifest_path: Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def load_manifest(path: str | Path, split: str | None = None) -> list[PairRecord]:
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest {manifest_path} is missing: {sorted(missing)}")
        records = [
            PairRecord(
                sample_id=row["sample_id"],
                input_path=_resolve(row["input_path"], manifest_path),
                target_path=_resolve(row["target_path"], manifest_path),
                split=row["split"],
                weather=row["weather"],
                source=row["source"],
            )
            for row in reader
            if split is None or row["split"] == split
        ]
    return records


def deterministic_split(
    group_id: str,
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> str:
    if train_fraction <= 0 or val_fraction < 0 or train_fraction + val_fraction >= 1:
        raise ValueError("Split fractions must leave a non-empty test fraction")
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < train_fraction:
        return "train"
    if value < train_fraction + val_fraction:
        return "val"
    return "test"


def write_manifest(path: str | Path, records: Iterable[PairRecord]) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    fieldnames = ["sample_id", "input_path", "target_path", "split", "weather", "source"]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "input_path": str(record.input_path),
                    "target_path": str(record.target_path),
                    "split": record.split,
                    "weather": record.weather,
                    "source": record.source,
                }
            )
    temporary.replace(output_path)
