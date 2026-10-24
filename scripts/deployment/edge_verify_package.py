#!/usr/bin/env python3
"""Verify every packaged file before device engine compilation."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
manifest = json.loads((ROOT / "PACKAGE.json").read_text())
for relative, expected in manifest["files"].items():
    path = (ROOT / relative).resolve()
    if ROOT not in path.parents or not path.is_file():
        raise ValueError(f"Missing/unsafe package entry: {relative}")
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    if value.hexdigest() != expected:
        raise ValueError(f"Package checksum failed: {relative}")
print(f"Verified {len(manifest['files'])} packaged files")
