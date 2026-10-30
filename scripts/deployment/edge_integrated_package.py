#!/usr/bin/env python3
"""Extend the verified frozen deployment kit with reviewed experiment tools."""

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_NAME = "aegis_restormer_edge_compression_20261024"
BASE_SHA256 = "ca712cc02e789719da395b1ce38d32b72bd08017eb68410efac744cb43f453c1"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def build(base_archive, destination):
    if digest(base_archive) != BASE_SHA256:
        raise ValueError("Base archive differs from the verified October 24 release")
    if destination.exists() or destination.with_suffix(".tar.gz").exists():
        raise FileExistsError("Use a fresh package destination")
    destination.mkdir(parents=True)
    # Copy only regular files under the expected archive root; never extract links.
    with tarfile.open(base_archive) as archive:
        for member in archive.getmembers():
            parts = Path(member.name).parts
            if not member.isfile() or parts[0] != BASE_NAME or ".." in parts:
                raise ValueError(f"Unexpected archive member: {member.name}")
            target = destination.joinpath(*parts[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    manifest = json.loads((destination / "PACKAGE.json").read_text())
    for relative, expected in manifest["files"].items():
        if digest(destination / relative) != expected:
            raise ValueError(f"Base file integrity failure: {relative}")
    shutil.copy2(destination / "PACKAGE.json", destination / "BASE_PACKAGE.json")
    additions = [
        "scripts/deployment/edge_custom_candidate.py",
        "scripts/deployment/edge_integrated_package.py",
        "edge_pipeline/quantization_scripts/custom_quantizer.py",
        "configs/deployment/edge_experiment_requirements.txt",
        "docs/edge_integrated_handoff.txt",
        "scripts/weather_testing/weather_suite.py",
        "scripts/weather_testing/requirements-workstation.txt",
        "scripts/weather_testing/README.md",
    ]
    for relative in additions:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    (destination / "START_INTEGRATED.txt").write_text(
        (ROOT / "docs/edge_integrated_handoff.txt").read_text()
    )
    files = {
        str(p.relative_to(destination)): digest(p)
        for p in sorted(destination.rglob("*"))
        if p.is_file() and p.name != "PACKAGE.json"
    }
    manifest.update(
        {
            "name": destination.name,
            "files": files,
            "base_archive_sha256": BASE_SHA256,
            "scope": "Trained deployment with custom weight-simulation and weather experiments. "
            "TensorRT engines and Jetson measurements remain pending.",
        }
    )
    (destination / "PACKAGE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    output = destination.with_suffix(".tar.gz")
    with tarfile.open(output, "w:gz", compresslevel=3) as archive:
        for relative in ["PACKAGE.json", *files]:
            archive.add(
                destination / relative, arcname=destination.name + "/" + relative, recursive=False
            )
    output.with_name(output.name + ".sha256").write_text(digest(output) + "  " + output.name + "\n")
    print(json.dumps({"archive": str(output), "files": len(files), "bytes": output.stat().st_size}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=ROOT / "artifacts/releases" / (BASE_NAME + ".tar.gz")
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/releases/aegis_edge_integrated_20261030"
    )
    args = parser.parse_args()
    build(args.base, args.output)
