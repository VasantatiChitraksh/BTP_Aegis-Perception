#!/usr/bin/env python3
"""Audit and summarize saved predictions; does not rerun or change a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aegis_perception.data.catalog import sha256
from aegis_perception.data.manifest import load_manifest
from aegis_perception.evaluation import summarize_samples
from aegis_perception.reproducibility import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.metrics.read_text())
    manifest_hash = sha256(args.manifest)
    if payload["manifest_sha256"] != manifest_hash:
        raise ValueError("Result manifest hash does not match supplied manifest")
    expected = {r.sample_id: r for r in load_manifest(args.manifest, split=payload["split"])}
    samples = payload["samples"]
    if len(samples) != len(expected) or {r["sample_id"] for r in samples} != set(expected):
        raise ValueError("Metrics do not cover the complete frozen split exactly once")
    for row in samples:
        record = expected[row["sample_id"]]
        if (row["weather"], row["group_id"]) != (record.weather, record.group_id):
            raise ValueError(f"Weather/background mismatch for {record.sample_id}")
    panel = json.loads(args.panel.read_text())
    if (
        manifest_hash not in panel["compatible_manifest_sha256"]
        or panel["split"] != payload["split"]
    ):
        raise ValueError("Panel provenance mismatch")
    ids = set(panel["sample_ids"])
    if len(ids) != len(panel["sample_ids"]) or not ids <= set(expected):
        raise ValueError("Panel contains duplicate or unknown sample IDs")
    result = {
        "metrics_file_sha256": sha256(args.metrics),
        "manifest_sha256": manifest_hash,
        "panel_sha256": sha256(args.panel),
        "checkpoint_epoch": payload["checkpoint_epoch"],
        "full_test": summarize_samples(samples),
        "balanced_test_panel": summarize_samples([r for r in samples if r["sample_id"] in ids]),
        "note": "Same saved predictions; panel is a secondary subset, not an untouched test set.",
    }
    write_json(args.output, result)
    print(
        json.dumps(
            {key: result[key]["headline"] for key in ("full_test", "balanced_test_panel")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
