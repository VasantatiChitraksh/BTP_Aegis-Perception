#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import yaml

REPOSITORY = Path(__file__).resolve().parents[1]
CATALOG = REPOSITORY / "data" / "datasets.yaml"
ARCHIVE_DIR = REPOSITORY / "data" / "archives"
MARKER_NAME = ".aegis-extracted.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify or safely extract reviewed datasets")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--verify", action="store_true", help="Verify size and available hashes")
    action.add_argument("--extract", action="store_true", help="Verify and extract archives")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--condition", choices=("smoke", "rain", "fog", "snow"))
    selection.add_argument("--dataset", action="append", help="Dataset ID; may be repeated")
    return parser.parse_args()


def load_catalog() -> list[dict]:
    with CATALOG.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)["datasets"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_datasets(args: argparse.Namespace, datasets: list[dict]) -> list[dict]:
    core = [
        item
        for item in datasets
        if item["selection"] == "core" and item.get("auto_download")
    ]
    if args.condition:
        return [item for item in core if args.condition in item["conditions"]]
    if args.dataset:
        by_id = {item["id"]: item for item in datasets}
        missing = sorted(set(args.dataset) - by_id.keys())
        if missing:
            raise SystemExit(f"Unknown dataset ID(s): {', '.join(missing)}")
        return [by_id[dataset_id] for dataset_id in dict.fromkeys(args.dataset)]
    return core


def verify(dataset: dict) -> tuple[Path, str]:
    archive = dataset["archive"]
    path = ARCHIVE_DIR / archive["filename"]
    if not path.is_file():
        raise RuntimeError(f"missing archive: {path}")
    if path.with_suffix(path.suffix + ".aria2").exists():
        raise RuntimeError(f"download is still incomplete: {path}")
    expected_bytes = archive.get("expected_bytes")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"expected {expected_bytes} bytes, found {path.stat().st_size}: {path}"
        )
    actual_hash = sha256(path)
    expected_hash = archive.get("sha256")
    if expected_hash and actual_hash != expected_hash:
        raise RuntimeError(f"SHA-256 mismatch: {path}")
    return path, actual_hash


def safe_member(name: str) -> bool:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    return not path.is_absolute() and ".." not in path.parts


def extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as handle:
        unsafe = [
            member.filename
            for member in handle.infolist()
            if not safe_member(member.filename)
            or stat.S_ISLNK((member.external_attr >> 16) & 0xFFFF)
        ]
        if unsafe:
            raise RuntimeError(f"unsafe ZIP member: {unsafe[0]!r}")
        handle.extractall(destination)


def extract_tar(archive: Path, destination: Path) -> None:
    with tarfile.open(archive) as handle:
        unsafe = [
            member.name
            for member in handle.getmembers()
            if not safe_member(member.name) or not (member.isfile() or member.isdir())
        ]
        if unsafe:
            raise RuntimeError(f"unsafe TAR member: {unsafe[0]!r}")
        handle.extractall(destination)


def extract_rar(archive: Path, destination: Path) -> None:
    seven_zip = shutil.which("7zz") or shutil.which("7z")
    if not seven_zip:
        raise RuntimeError("7z is required to extract RAR archives")
    listing = subprocess.run(
        [seven_zip, "l", "-slt", str(archive)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    names = [line[7:] for line in listing.splitlines() if line.startswith("Path = ")]
    unsafe = [name for name in names[1:] if not safe_member(name)]
    if unsafe:
        raise RuntimeError(f"unsafe RAR member: {unsafe[0]!r}")
    subprocess.run(
        [
            seven_zip,
            "x",
            "-y",
            "-bso0",
            "-bsp0",
            "-bse1",
            f"-o{destination}",
            str(archive),
        ],
        check=True,
    )


def extract(dataset: dict, archive: Path, actual_hash: str) -> None:
    destination = REPOSITORY / dataset["extract_to"]
    marker = destination / MARKER_NAME
    if marker.is_file():
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if metadata.get("archive_sha256") == actual_hash:
            print(f"present   {dataset['id']}: {destination}")
            return
        raise RuntimeError(f"{destination} came from a different archive; preserving it")
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"{destination} is non-empty and untracked; preserving it")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{dataset['id']}-", dir=destination.parent) as tmp:
        staging = Path(tmp)
        archive_format = dataset["archive"]["format"]
        if archive_format == "zip":
            extract_zip(archive, staging)
        elif archive_format == "tar":
            extract_tar(archive, staging)
        elif archive_format == "rar":
            extract_rar(archive, staging)
        else:
            raise RuntimeError(f"unsupported archive format: {archive_format}")
        (staging / MARKER_NAME).write_text(
            json.dumps(
                {
                    "dataset_id": dataset["id"],
                    "archive": archive.name,
                    "archive_sha256": actual_hash,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            destination.rmdir()
        staging.rename(destination)
    print(f"extracted {dataset['id']}: {destination}")


def main() -> None:
    args = parse_args()
    failures = []
    for dataset in selected_datasets(args, load_catalog()):
        try:
            archive, actual_hash = verify(dataset)
            print(f"verified  {dataset['id']}: {actual_hash}")
            if args.extract:
                extract(dataset, archive, actual_hash)
        except Exception as exc:
            failures.append(f"{dataset['id']}: {exc}")
            print(f"ERROR     {failures[-1]}")
    if failures:
        raise SystemExit(f"{len(failures)} dataset(s) failed")


if __name__ == "__main__":
    main()
