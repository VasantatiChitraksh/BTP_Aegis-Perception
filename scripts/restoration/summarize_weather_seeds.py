#!/usr/bin/env python3
"""Report all three declared fine-tuning seeds; do not select a seed on test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aegis_perception.data.catalog import sha256
from aegis_perception.reproducibility import write_json


def summarize(paths):
    reports = [json.loads(p.read_text()) for p in paths]
    if len(reports) != 3 or {r.get("training_seed") for r in reports} != {42, 43, 44}:
        raise ValueError("Require exactly seeds 42, 43, 44; report failed runs separately")
    for key in (
        "protocol_sha256",
        "manifest_sha256",
        "baseline_metrics_sha256",
        "condition_sha256",
    ):
        if len({r[key] for r in reports}) != 1:
            raise ValueError(f"Incompatible reports: {key}")
    scores = {}
    for weather in ("rain", "fog", "snow", "smoke", "macro_weather"):
        scores[weather] = {}
        for metric in ("psnr_db", "ssim"):
            values = [r["candidate"]["scene_balanced_summary"][weather][metric] for r in reports]
            scores[weather][metric] = {
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)),
                "by_seed": {
                    str(r["training_seed"]): v for r, v in zip(reports, values, strict=True)
                },
            }
    return {
        "scores": scores,
        "comparison_sha256": {str(p): sha256(p) for p in paths},
        "scope": "Three fine-tuning seeds from shared initialization; standard deviation "
        "describes training variability, not test-population confidence. "
        "Per-run cluster intervals remain in the comparison files.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparisons", type=Path, nargs=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, summarize(args.comparisons))


if __name__ == "__main__":
    main()
