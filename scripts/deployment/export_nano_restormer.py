#!/usr/bin/env python3
"""Workstation-only: export the trained Restormer at 128x224 with four references."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import onnx
import torch
from PIL import Image

from aegis_perception.checkpoints import load_generator
from aegis_perception.edge.common import digest, model_info, ort_session, preprocess, select_panel
from aegis_perception.official_runtime import activate


def main():
    output = ROOT / "artifacts/edge_v1/exports/128x224/restormer"
    output.mkdir(parents=True, exist_ok=False)
    checkpoint, _ = model_info("restormer")
    activate()
    torch.set_num_threads(4)
    model, state = load_generator(checkpoint, device="cpu")
    model.eval()
    graph = output / "restoration.onnx"
    print("Exporting trained epoch {} at 128x224".format(state["epoch"]), flush=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            torch.zeros(1, 3, 128, 224),
            str(graph),
            input_names=["degraded"],
            output_names=["restored"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    onnx.checker.check_model(str(graph), full_check=True)
    session = ort_session(graph)
    fixtures, rows = {}, []
    for record in select_panel("val", 1):
        with Image.open(record.input_path) as image:
            value, geometry = preprocess(image, 128, 224)
        with Image.open(record.target_path) as image:
            target, target_geometry = preprocess(image, 128, 224)
        if geometry != target_geometry:
            raise ValueError("Paired fixture geometry mismatch")
        with torch.inference_mode():
            expected = model(torch.from_numpy(value)).numpy()
        actual = session.run(None, {"degraded": value})[0]
        np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=1e-3)
        if actual.shape != value.shape or not np.isfinite(actual).all():
            raise ValueError("Invalid export output")
        weather = record.weather
        fixtures["input_" + weather] = value
        fixtures["expected_" + weather] = actual
        fixtures["target_" + weather] = target
        width, height = geometry["resized_wh"]
        fixtures["valid_hw_" + weather] = np.array([height, width], np.int32)
        rows.append(
            {
                "weather": weather,
                "sample_id": record.sample_id,
                "input_sha256": digest(record.input_path),
                "target_sha256": digest(record.target_path),
                "max_abs_error": float(np.max(np.abs(actual - expected))),
            }
        )
        print("{}: PyTorch/ONNX agreement PASS".format(weather), flush=True)
    np.savez_compressed(str(output / "fixtures.npz"), **fixtures)
    report = {
        "passed": True,
        "shape": [1, 3, 128, 224],
        "selected_epoch": state["epoch"],
        "checkpoint_sha256": digest(checkpoint),
        "onnx_sha256": digest(graph),
        "fixtures_sha256": digest(output / "fixtures.npz"),
        "samples": rows,
        "scope": "Four-image export agreement; Nano build/inference NOT measured. "
        "Lower resolution is a new operating point, not equivalent quality.",
        "exporter_sha256": digest(__file__),
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
