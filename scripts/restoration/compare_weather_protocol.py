#!/usr/bin/env python3
"""Re-score saved baseline predictions on frozen v3; optionally compare a candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aegis_perception.data.catalog import sha256
from aegis_perception.data.manifest import load_manifest
from aegis_perception.evaluation import clustered_intervals, summarize_samples
from aegis_perception.reproducibility import write_json


def checked_results(path, manifest):
    payload = json.loads(path.read_text())
    if payload["split"] != "test" or payload["manifest_sha256"] != sha256(manifest):
        raise ValueError("Test result provenance does not match manifest")
    expected = {r.sample_id: r for r in load_manifest(manifest, "test")}
    samples = {r["sample_id"]: r for r in payload["samples"]}
    if set(expected) != set(samples) or len(samples) != len(payload["samples"]):
        raise ValueError("Results must cover the complete test split exactly once")
    for identifier, row in samples.items():
        record = expected[identifier]
        if (row["weather"], row["group_id"]) != (record.weather, record.group_id):
            raise ValueError("Result weather/background metadata mismatch")
        for field in ("stratum", "cluster_id"):
            if getattr(record, field) and row.get(field) != getattr(record, field):
                raise ValueError(f"Result {field} mismatch")
    return payload, expected, samples


def compare(baseline_path, baseline_manifest, active_manifest, candidate_path=None):
    protocol_path = active_manifest.parent / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["manifest_sha256"][active_manifest.name] != sha256(active_manifest):
        raise ValueError("Frozen protocol manifest changed")
    old, old_records, old_samples = checked_results(baseline_path, baseline_manifest)
    selected = load_manifest(active_manifest, "test")
    baseline = []
    for record in selected:
        previous = old_records[record.sample_id]
        if (previous.input_path, previous.target_path, previous.group_id, previous.weather) != (
            record.input_path,
            record.target_path,
            record.group_id,
            record.weather,
        ):
            raise ValueError("Baseline and active pairs differ")
        baseline.append(
            {
                **old_samples[record.sample_id],
                "stratum": record.stratum,
                "cluster_id": record.cluster_id,
            }
        )
    report = {
        "protocol_sha256": sha256(protocol_path),
        "manifest_sha256": sha256(active_manifest),
        "baseline_metrics_sha256": sha256(baseline_path),
        "baseline": summarize_samples(baseline),
        "baseline_uncertainty": clustered_intervals(baseline),
        "note": "Saved predictions re-aggregated on v3, not a model improvement or new test set.",
    }
    if candidate_path:
        candidate, _, samples = checked_results(candidate_path, active_manifest)
        for key in ("color", "data_range", "image_size", "spatial_mode", "precision", "mse_floor"):
            if old["protocol"].get(key) != candidate["protocol"].get(key):
                raise ValueError(f"Incompatible scoring protocol: {key}")
        for row in baseline:
            for metric in ("raw_psnr_db", "raw_ssim"):
                if abs(row[metric] - samples[row["sample_id"]][metric]) > 1e-5:
                    raise ValueError("Raw metric mismatch: input/target or scoring changed")
        new_rows = list(samples.values())
        report.update(
            {
                "candidate_metrics_sha256": sha256(candidate_path),
                "training_seed": candidate.get("training_seed"),
                "condition_sha256": candidate["condition_sha256"],
                "candidate": summarize_samples(new_rows),
                "paired_change": clustered_intervals(new_rows, reference=baseline),
            }
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("artifacts/restoration/four_weather_restormer_seed42/metrics_test.json"),
    )
    parser.add_argument(
        "--baseline-manifest", type=Path, default=Path("data/manifests/clean/four_weather.csv")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/weather_v3/active.csv")
    )
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.baseline_manifest, args.manifest, args.candidate)
    write_json(args.output, result)
    print(
        json.dumps(
            {k: result[k]["headline"] for k in ("baseline", "candidate") if k in result}, indent=2
        )
    )


if __name__ == "__main__":
    main()
