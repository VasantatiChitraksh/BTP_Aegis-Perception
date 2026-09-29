#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from aegis_perception.data.catalog import load_catalog, select_datasets, validate_archive

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
    selected = select_datasets(load_catalog(CATALOG), ids=args.dataset, condition=args.condition)
    for dataset in selected:
        try:
            if "archive" not in dataset:
                raise RuntimeError("no downloadable archive is configured")
            archive = ARCHIVE_DIR / dataset["archive"]["filename"]
            actual_hash = validate_archive(archive, dataset["archive"])
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
