#!/usr/bin/env python3
"""Add clean targets to validation fixtures and training-only INT8 calibration inputs."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from PIL import Image

from aegis_perception.edge.common import OUT, digest, preprocess, select_panel, write_json

directory = OUT / "exports/256x448/restormer"
with np.load(directory / "fixtures.npz", allow_pickle=False) as saved:
    fixtures = dict(saved)
rows = []
for record in select_panel("val", 1):
    with Image.open(record.target_path) as image:
        target, geometry = preprocess(image)
    width, height = geometry["resized_wh"]
    fixtures["target_" + record.weather] = target
    fixtures["valid_hw_" + record.weather] = np.array([height, width], dtype=np.int32)
    rows.append(
        {
            "sample_id": record.sample_id,
            "weather": record.weather,
            "split": "val",
            "input_sha256": digest(record.input_path),
            "target_sha256": digest(record.target_path),
        }
    )
np.savez_compressed(directory / "fixtures.npz", **fixtures)
write_json(
    directory / "fixture_provenance.json",
    {
        "samples": rows,
        "sha256": digest(directory / "fixtures.npz"),
        "selection": "One validation background per weather selected by hash, not metric score",
        "scope": "Four-image precision smoke test, not a statistical accuracy benchmark",
    },
)
inputs, rows = [], []
for record in select_panel("train", 16):
    with Image.open(record.input_path) as image:
        x, _ = preprocess(image)
    inputs.append(x)
    rows.append(
        {
            "sample_id": record.sample_id,
            "weather": record.weather,
            "group_id": record.group_id,
            "split": "train",
            "input_sha256": digest(record.input_path),
        }
    )
np.savez_compressed(directory / "calibration.npz", inputs=np.stack(inputs))
write_json(
    directory / "calibration_provenance.json",
    {
        "samples": rows,
        "sha256": digest(directory / "calibration.npz"),
        "count": len(rows),
        "selection": (
            "16 per weather; hash ordering and background round-robin; training split ONLY"
        ),
        "note": (
            "Representative initial PTQ calibration set, not evidence that "
            "INT8 retains quality; device validation required"
        ),
    },
)
print("Prepared four-weather validation targets and 64 training-only calibration tensors")
