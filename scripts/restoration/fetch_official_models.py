#!/usr/bin/env python3
"""Fetch checksum-pinned official sources and pretrained weights; never train."""

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "configs/restoration/official_v1/sources.json"


def digest(path):
    with path.open("rb") as handle:
        h = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
        return h.hexdigest()


def ensure(entry, *, offline=False):
    destination = ROOT / entry["path"]
    if destination.exists():
        if digest(destination) != entry["sha256"]:
            raise ValueError(f"Checksum mismatch; refusing to overwrite: {destination}")
        return
    if offline:
        raise FileNotFoundError(f"Required official asset missing: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if "archive_member" in entry:
        import gdown

        archive = destination.with_suffix(".download.zip")
        result = gdown.download(entry["url"], str(archive), quiet=False)
        if result is None or digest(archive) != entry["archive_sha256"]:
            raise ValueError("Official archive download/checksum failed")
        with zipfile.ZipFile(archive) as bundle:
            with bundle.open(entry["archive_member"]) as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target)
        archive.unlink()
    else:
        with (
            urllib.request.urlopen(entry["url"], timeout=120) as source,
            partial.open("wb") as target,
        ):
            shutil.copyfileobj(source, target)
    if digest(partial) != entry["sha256"]:
        raise ValueError(f"Downloaded asset checksum mismatch: {partial}")
    partial.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Verify only; never download")
    parser.add_argument(
        "--sources-only", action="store_true",
        help="Fetch architecture sources/notices only; use with trained inference checkpoints",
    )
    args = parser.parse_args()
    for name, model in json.loads(LOCK.read_text())["models"].items():
        for source in model["sources"]:
            ensure(source, offline=args.offline)
        if not args.sources_only:
            ensure(model["weights"], offline=args.offline)
        assets = "sources" if args.sources_only else "sources and pretrained weights"
        print(f"Verified official {assets}: {name}", flush=True)


if __name__ == "__main__":
    main()
