#!/usr/bin/env python3
"""Export exact selected checkpoints and validate static ONNX at deployment geometry."""

import argparse
import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import onnx
import torch
from PIL import Image

from aegis_perception.checkpoints import load_generator
from aegis_perception.edge.common import (
    HEIGHT,
    MODELS,
    OUT,
    WIDTH,
    digest,
    environment,
    model_info,
    ort_session,
    postprocess,
    preprocess,
    select_panel,
    write_json,
)
from aegis_perception.metrics import psnr, ssim
from aegis_perception.official_runtime import activate


def export(name, height, width):
    checkpoint, info = model_info(name)
    directory = OUT / "exports" / f"{height}x{width}" / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "restoration.onnx"
    report = directory / "validation.json"
    if report.exists() and path.exists() and (directory / "fixtures.npz").exists():
        import json

        previous = json.loads(report.read_text())
        if (
            previous.get("passed")
            and previous["checkpoint_sha256"] == digest(checkpoint)
            and previous["onnx_sha256"] == digest(path)
        ):
            print(f"Reusing verified export: {name}", flush=True)
            return
    activate()
    torch.set_num_threads(4)
    generator, state = load_generator(checkpoint, device="cpu")
    if height % 64 or width % 64:
        raise ValueError("Static export dimensions must be divisible by 64 for all six models")
    sample = torch.zeros(1, 3, height, width)
    started = time.monotonic()
    print(f"Exporting {name}: 1x3x{height}x{width}", flush=True)
    temporary = directory / "restoration.partial.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            generator,
            sample,
            temporary,
            input_names=["degraded"],
            output_names=["restored"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    onnx.checker.check_model(str(temporary), full_check=True)
    temporary.replace(path)
    session = ort_session(path)
    checks = []
    fixtures = {}
    for record in select_panel("val", 1):
        with Image.open(record.input_path) as image:
            x, geometry = preprocess(image, height, width)
        with torch.inference_mode():
            expected = generator(torch.from_numpy(x)).numpy()
        actual = session.run(["restored"], {"degraded": x})[0]
        fixtures[f"input_{record.weather}"] = x
        fixtures[f"expected_{record.weather}"] = actual
        difference = np.abs(actual - expected)
        np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=1e-3)
        if actual.shape != x.shape or not np.isfinite(actual).all():
            raise ValueError(f"Invalid exported output for {name}")
        # Same deployed geometry for both backends; these are not native benchmark scores.
        restored = np.asarray(postprocess(actual, geometry), dtype=np.float32) / 255
        reference = np.asarray(postprocess(expected, geometry), dtype=np.float32) / 255
        with Image.open(record.target_path) as image:
            target = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        checks.append(
            {
                "sample_id": record.sample_id,
                "weather": record.weather,
                "max_abs_error": float(difference.max()),
                "rmse": float(np.sqrt(np.mean(difference**2))),
                "pytorch_psnr": psnr(target, reference),
                "onnx_psnr": psnr(target, restored),
                "pytorch_ssim": ssim(target, reference),
                "onnx_ssim": ssim(target, restored),
            }
        )
        print(f"  {record.weather}: max error {difference.max():.6g}", flush=True)
    timings = []
    for _ in range(2):
        session.run(None, {"degraded": x})
    for _ in range(5):
        before = time.perf_counter()
        session.run(None, {"degraded": x})
        timings.append((time.perf_counter() - before) * 1000)
    write_json(
        report,
        {
            "passed": True,
            "name": name,
            "shape": [1, 3, height, width],
            "selected_epoch": state["epoch"],
            "parameters": info["parameters"],
            "checkpoint_sha256": digest(checkpoint),
            "source_best_sha256": info["best_sha256"],
            "onnx_sha256": digest(path),
            "opset": 17,
            "precision": "FP32",
            "tolerances": {"atol": 5e-4, "rtol": 1e-3},
            "validation_panel": checks,
            "cpu_smoke_timing_ms": {
                "runs": 5,
                "warmup": 2,
                "median": float(np.median(timings)),
                "p95": float(np.percentile(timings, 95)),
            },
            "timing_scope": "Local CPU ORT model call only; NOT Jetson or end-to-end FPS",
            "elapsed_seconds": time.monotonic() - started,
            "environment": environment(),
        },
    )
    # Portable backend-validation fixtures for on-device precision checks.
    np.savez_compressed(directory / "fixtures.npz", **fixtures)
    del generator, state, session
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["restormer"])
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    args = parser.parse_args()
    for name in args.models:
        export(name, args.height, args.width)


if __name__ == "__main__":
    main()
