"""Dataset selection and archive checks shared by the download and extraction tools."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..config import load_config


def load_catalog(path: Path) -> list[dict]:
    return load_config(path)["datasets"]


def select_datasets(
    datasets: list[dict], *, ids: list[str] | None = None, condition: str | None = None
) -> list[dict]:
    if ids:
        by_id = {item["id"]: item for item in datasets}
        missing = sorted(set(ids) - by_id.keys())
        if missing:
            raise ValueError(f"Unknown dataset ID(s): {', '.join(missing)}")
        return [by_id[dataset_id] for dataset_id in dict.fromkeys(ids)]
    return [
        item
        for item in datasets
        if item["selection"] == "core"
        and item.get("auto_download")
        and (condition is None or condition in item["conditions"])
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archive(path: Path, archive: dict) -> str:
    if not path.is_file():
        raise RuntimeError(f"missing archive: {path}")
    if path.with_suffix(path.suffix + ".aria2").exists():
        raise RuntimeError(f"download is still incomplete: {path}")
    expected_bytes = archive.get("expected_bytes")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"{path.name}: expected {expected_bytes} bytes, found {path.stat().st_size}"
        )
    actual_hash = sha256(path)
    if archive.get("sha256") and actual_hash != archive["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch: {path}")
    return actual_hash
