#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from aegis_perception.data.catalog import load_catalog, select_datasets, validate_archive

CATALOG = REPOSITORY / "data" / "datasets.yaml"
DEFAULT_ARCHIVE_DIR = REPOSITORY / "data" / "archives"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List or download the reviewed smoke/rain/fog/snow collection"
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--core", action="store_true", help="Download every core source")
    selection.add_argument("--condition", choices=("smoke", "rain", "fog", "snow"))
    selection.add_argument("--dataset", action="append", help="Dataset ID; may be repeated")
    parser.add_argument("--list", action="store_true", help="List the catalog and exit")
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    return parser.parse_args()


def aria2_download(url: str, destination: Path) -> None:
    command = [
        "aria2c",
        "--continue=true",
        "--max-connection-per-server=4",
        "--split=4",
        "--min-split-size=1M",
        "--file-allocation=none",
        "--auto-file-renaming=false",
        "--allow-overwrite=false",
        "--dir",
        str(destination.parent),
        "--out",
        destination.name,
        url,
    ]
    subprocess.run(command, check=True)


def urllib_download(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Aegis-Perception/1.0"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    partial.replace(destination)


def download_one(dataset: dict, archive_dir: Path) -> None:
    archive = dataset.get("archive")
    if not archive or not archive.get("url"):
        raise RuntimeError(f"{dataset['id']}: no direct reviewed download is configured")
    archive_dir.mkdir(parents=True, exist_ok=True)
    destination = archive_dir / archive["filename"]
    expected_bytes = archive.get("expected_bytes")

    if (
        destination.is_file()
        and (expected_bytes is None or destination.stat().st_size == expected_bytes)
        and not destination.with_suffix(destination.suffix + ".aria2").exists()
    ):
        validate_archive(destination, archive)
        print(f"verified  {dataset['id']}: {destination}")
        return
    if destination.is_file() and expected_bytes and destination.stat().st_size > expected_bytes:
        raise RuntimeError(
            f"{destination} is larger than expected; preserving it for manual inspection"
        )

    print(f"download  {dataset['id']}: {archive['url']}")
    if shutil.which("aria2c"):
        aria2_download(archive["url"], destination)
    else:
        if destination.exists():
            raise RuntimeError(
                f"{destination} is partial and aria2c is unavailable for a safe resume"
            )
        urllib_download(archive["url"], destination)
    validate_archive(destination, archive)
    print(f"complete  {dataset['id']}: {destination}")


def print_catalog(datasets: list[dict]) -> None:
    headings = ("ID", "conditions", "selection", "automatic", "tasks")
    rows = [
        (
            item["id"],
            ",".join(item["conditions"]),
            item["selection"],
            "yes" if item.get("auto_download") else "no",
            ",".join(item["tasks"]),
        )
        for item in datasets
    ]
    widths = [max(len(headings[index]), *(len(row[index]) for row in rows)) for index in range(5)]
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headings)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def main() -> None:
    args = parse_args()
    datasets = load_catalog(CATALOG)
    if args.list:
        print_catalog(datasets)
        return
    if not (args.core or args.condition or args.dataset):
        raise SystemExit("Choose --core, --condition, --dataset, or --list")
    selected = select_datasets(datasets, ids=args.dataset, condition=args.condition)
    failures = []
    for dataset in selected:
        try:
            download_one(dataset, args.archive_dir.expanduser().resolve())
        except Exception as exc:
            failures.append(f"{dataset['id']}: {exc}")
            print(f"ERROR     {failures[-1]}", file=sys.stderr)
    if failures:
        raise SystemExit(f"{len(failures)} download(s) failed")


if __name__ == "__main__":
    main()
