#!/usr/bin/env python3
"""Collect actual device benchmark outputs; never fill missing hardware measurements."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from aegis_perception.edge.common import write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", type=Path, default=Path("results"))
args = parser.parse_args()
rows = []
for path in sorted(args.results.glob("*/benchmark.json")):
    value = json.loads(path.read_text())
    rows.append(
        {
            "run": path.parent.name,
            "mode": value["mode"],
            "frames": value["frames"],
            "elapsed_seconds": value["elapsed_seconds"],
            "throughput_fps": value["measured_throughput_fps"],
            "pipeline_median_ms": value["timings_ms"]["pipeline_ms"]["median"],
            "pipeline_p95_ms": value["timings_ms"]["pipeline_ms"]["p95"],
            "frame_wall_p95_ms": value["timings_ms"]["frame_wall_ms"]["p95"],
            "detector_backend": value.get("detector_backend", "unspecified"),
            "restorer_backend": value.get("restorer_backend"),
            "device": value["environment"].get("jetson_model"),
            "tensorrt": value["environment"].get("tensorrt"),
            "detector_sha256": value["detector_sha256"],
            "restorer_sha256": value["restorer_sha256"],
        }
    )
if not rows:
    raise ValueError("No completed benchmark.json files found")
with (args.results / "device_comparison.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
write_json(
    args.results / "device_comparison.json",
    {
        "runs": rows,
        "note": (
            "Measured runs only. Do not compare differing power modes, inputs, "
            "precision or hardware as a controlled experiment. Accuracy, peak RAM "
            "and power are not inferred from FPS."
        ),
    },
)
print(args.results / "device_comparison.csv")
