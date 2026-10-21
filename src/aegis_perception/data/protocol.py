"""Verify the versioned split before training or evaluation."""

import json
from pathlib import Path

from .catalog import sha256


def verify_protocol(data):
    if not data.get("protocol"):
        return
    path = Path(data["protocol"])
    if sha256(path) != data["protocol_sha256"]:
        raise ValueError("Frozen protocol hash mismatch")
    report = json.loads(path.read_text())
    manifest = Path(data["manifest"])
    if manifest.resolve() != (path.parent / "active.csv").resolve():
        raise ValueError("Training/evaluation must use the active protocol manifest")
    for name, digest in report["manifest_sha256"].items():
        if sha256(path.parent / name) != digest:
            raise ValueError(f"Frozen manifest hash mismatch: {name}")
