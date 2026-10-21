#!/usr/bin/env python3
"""Package the shared Restormer run and its exact data dependencies for a GPU host."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aegis_perception.data.catalog import sha256
from aegis_perception.data.manifest import load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/transfer/aegis_four_weather.tar"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Bundle already exists: {output}; choose a new output path")
    files = set()
    for directory in ("src", "scripts", "configs", "tests"):
        files.update(
            p
            for p in (ROOT / directory).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        )
    files.update(
        ROOT / p
        for p in (
            "pyproject.toml",
            "README.md",
            "LICENSE",
            "data/datasets.yaml",
            "docs/datasets.md",
            "docs/experiments.md",
            "artifacts/pretrained/deraining.pth",
            "artifacts/data/four_weather/preparation_report.json",
            "artifacts/data/four_weather/preflight.json",
            "artifacts/data/four_weather/manifest_audit.txt",
        )
    )
    for name in ("four_weather", "realrain1k", "reside6k", "snow_cityscapes", "smokebench"):
        manifest = ROOT / f"data/manifests/clean/{name}.csv"
        files.add(manifest)
        for record in load_manifest(manifest):
            files.update((record.input_path, record.target_path))
    files = sorted(files)
    for path in files:
        if not path.is_file() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"Missing or external bundle dependency: {path}")
    total = sum(p.stat().st_size for p in files)
    print(f"Packaging {len(files)} files, {total / 2**30:.2f} GiB", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    # Images are already compressed; plain tar avoids a slow second compression pass.
    with tarfile.open(temporary, "w") as archive:
        for i, path in enumerate(files, 1):
            archive.add(path, arcname=path.relative_to(ROOT), recursive=False)
            if i % 5000 == 0:
                print(f"Packed {i}/{len(files)} files", flush=True)
    temporary.replace(output)
    digest = sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n")
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(
            {
                "archive": output.name,
                "sha256": digest,
                "archive_bytes": output.stat().st_size,
                "file_count": len(files),
                "payload_bytes": total,
                "files": [p.relative_to(ROOT).as_posix() for p in files],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Ready: {output}", flush=True)


if __name__ == "__main__":
    main()
