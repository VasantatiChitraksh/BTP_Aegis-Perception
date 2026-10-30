#!/usr/bin/env python3
"""Screen Chitraksh's weight quantizer on the verified trained ONNX export.

This produces a FLOAT simulation graph, never an INT8 engine or a QDQ graph.
Run on the workstation; use edge_build.py separately for hardware compilation.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from aegis_perception.edge.common import digest, environment, ort_session, write_json
from aegis_perception.edge.quality import quality
from edge_pipeline.quantization_scripts.custom_quantizer import DevisedEdgeQuantizer


def verify_kit(kit):
    manifest = json.loads((kit / "PACKAGE.json").read_text())
    for relative, expected in manifest["files"].items():
        path = (kit / relative).resolve()
        if kit.resolve() not in path.parents or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Package integrity failure: {relative}")


def transform(model, bits=8, percentile=99.95, preserve_boundary=True):
    if bits not in (4, 8) or not 0 < percentile <= 100:
        raise ValueError("bits must be 4 or 8; percentile must be in (0, 100]")
    quantizer = DevisedEdgeQuantizer(num_bits=bits, clipping_percentile=percentile)
    weights = {node.input[1] for node in model.graph.node if node.op_type == "Conv"}
    boundary = {"network.patch_embed.proj.weight", "network.output.weight"}
    converted, preserved = [], []
    for tensor in model.graph.initializer:
        if tensor.name not in weights:
            continue
        if preserve_boundary and tensor.name in boundary:
            preserved.append(tensor.name)
            continue
        array = numpy_helper.to_array(tensor).copy()
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"Expected finite FP32 convolution weights: {tensor.name}")
        weight = torch.from_numpy(array)
        scale, zero = quantizer.compute_scale_zero_point(weight, ch_axis=0)
        result = quantizer.dequantize(quantizer.quantize(weight, scale, zero), scale, zero)
        tensor.CopyFrom(numpy_helper.from_array(result.numpy(), name=tensor.name))
        converted.append(tensor.name)
    if not converted:
        raise ValueError("No convolution weights converted")
    onnx.checker.check_model(model)
    return {"converted_weights": converted, "preserved_weights": preserved}


def screen(candidate, fixtures):
    session = ort_session(candidate)
    rows = []
    with np.load(fixtures, allow_pickle=False) as data:
        for key in sorted(k for k in data.files if k.startswith("input_")):
            suffix = key[6:]
            output = session.run(None, {session.get_inputs()[0].name: data[key]})[0]
            expected = data["expected_" + suffix]
            if output.shape != expected.shape or not np.isfinite(output).all():
                raise ValueError("Invalid candidate output")
            height, width = data["valid_hw_" + suffix]

            def rgb(value, height=height, width=width):
                return ((np.clip(value[0, :, :height, :width], -1, 1) + 1) / 2).transpose(1, 2, 0)

            baseline = quality(rgb(data["target_" + suffix]), rgb(expected))
            changed = quality(rgb(data["target_" + suffix]), rgb(output))
            delta_psnr = changed["psnr_db"] - baseline["psnr_db"]
            delta_ssim = changed["ssim"] - baseline["ssim"]
            rmse = float(np.sqrt(np.mean((output - expected) ** 2)))
            maximum = float(np.max(np.abs(output - expected)))
            rows.append(
                {
                    "sample": suffix,
                    "baseline_quality": baseline,
                    "candidate_quality": changed,
                    "psnr_delta_db": delta_psnr,
                    "ssim_delta": delta_ssim,
                    "rmse": rmse,
                    "max_abs_error": maximum,
                    "passed": bool(
                        delta_psnr >= -0.1
                        and delta_ssim >= -0.001
                        and rmse <= 0.004
                        and maximum <= 0.05
                    ),
                }
            )
    return {"passed": bool(rows) and all(row["passed"] for row in rows), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bits", type=int, choices=(4, 8), default=8)
    parser.add_argument("--percentile", type=float, default=99.95)
    parser.add_argument(
        "--quantize-boundary",
        action="store_true",
        help="Ablation: also simulate quantization of input/output convolutions",
    )
    args = parser.parse_args()
    verify_kit(args.kit)
    source = args.kit / "artifacts/edge_v1/exports/256x448/restormer"
    graph = source / "restoration.onnx"
    validation = json.loads((source / "validation.json").read_text())
    if not validation["passed"] or validation["onnx_sha256"] != digest(graph):
        raise ValueError("Expected the validated trained graph")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model = onnx.load(graph)
    stats = transform(model, args.bits, args.percentile, not args.quantize_boundary)
    candidate = args.output / "restoration_weight_simulation.onnx"
    onnx.save(model, candidate)
    del model
    result = screen(candidate, source / "fixtures.npz")
    report = {
        "scope": "Four-image rejection screen; not a full benchmark or hardware measurement",
        "execution": "FP32 ONNX Runtime; discretized float weights; no activation quantization",
        "algorithm": "DevisedEdgeQuantizer",
        "bits": args.bits,
        "percentile": args.percentile,
        "baseline_sha256": digest(graph),
        "candidate_sha256": digest(candidate),
        "fixtures_sha256": digest(source / "fixtures.npz"),
        "script_sha256": digest(__file__),
        "quantizer_sha256": digest(ROOT / "edge_pipeline/quantization_scripts/custom_quantizer.py"),
        "baseline_file_bytes": graph.stat().st_size,
        "candidate_file_bytes": candidate.stat().st_size,
        "environment": environment(),
        "layers": stats,
        "validation": result,
        "jetson_status": "Not built or measured",
    }
    write_json(args.output / "report.json", report)
    print(
        json.dumps({"report": str(args.output / "report.json"), "screen_passed": result["passed"]})
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
