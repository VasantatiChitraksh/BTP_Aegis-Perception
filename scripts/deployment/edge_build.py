#!/usr/bin/env python3
"""Build a hardware-local TensorRT engine, then gate it against reference fixtures."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from aegis_perception.edge.common import digest, environment, write_json
from aegis_perception.edge.quality import quality
from aegis_perception.edge.trt_runtime import Cuda, TrtSession


def validate(engine_path, fixtures, kind, precision):
    rows = []
    with np.load(fixtures, allow_pickle=False) as data, TrtSession(engine_path) as session:
        for key in sorted(k for k in data.files if k.startswith("input_")):
            expected = data[key.replace("input_", "expected_")]
            actual = session.run(data[key]).astype(np.float32)
            if expected.shape != actual.shape or not np.isfinite(actual).all():
                raise ValueError("Invalid engine output")
            difference = actual - expected
            row = {
                "sample": key,
                "max_abs_error": float(np.abs(difference).max()),
                "rmse": float(np.sqrt(np.mean(difference**2))),
            }
            if kind == "restormer":
                # [-1,1] output scale: 0.004 RMSE means ~54 dB output agreement.
                limit = 0.0005 if precision == "fp32" else 0.004
                max_limit = 0.005 if precision == "fp32" else 0.05
                row["passed"] = row["rmse"] <= limit and row["max_abs_error"] <= max_limit
                suffix = key[6:]
                height, width = data["valid_hw_" + suffix]

                def convert(a, height=height, width=width):
                    return ((np.clip(a[0, :, :height, :width], -1, 1) + 1) / 2).transpose(1, 2, 0)

                baseline = quality(convert(data["target_" + suffix]), convert(expected))
                candidate = quality(convert(data["target_" + suffix]), convert(actual))
                row["baseline_quality"] = baseline
                row["candidate_quality"] = candidate
                row["psnr_delta_db"] = candidate["psnr_db"] - baseline["psnr_db"]
                row["ssim_delta"] = candidate["ssim"] - baseline["ssim"]
                row["passed"] = (
                    row["passed"] and row["psnr_delta_db"] >= -0.1 and row["ssim_delta"] >= -0.001
                )
            else:
                # Box units are pixels, confidence units are probabilities.
                row["score_max_abs_error"] = float(np.abs(difference[:, 4:]).max())
                row["box_rmse_pixels"] = float(np.sqrt(np.mean(difference[:, :4] ** 2)))
                row["passed"] = row["score_max_abs_error"] <= 0.02 and row["box_rmse_pixels"] <= 0.5
            rows.append(row)
    return {
        "passed": bool(rows) and all(row["passed"] for row in rows),
        "rows": rows,
        "scope": "Small numerical smoke panel; does not establish full-dataset accuracy or safety",
    }


def build(args):
    import tensorrt as trt

    version = tuple(int(part) for part in trt.__version__.split(".")[:2])
    if not ((version[0] == 8 and version >= (8, 5)) or version[0] == 10):
        raise RuntimeError(
            f"TensorRT {trt.__version__} is outside the supported API range (8.5+ or 10.x). "
            "Use the TensorRT installation provided by JetPack."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    contract = {
        "onnx_sha256": digest(args.onnx),
        "fixtures_sha256": digest(args.fixtures),
        "precision": args.precision,
        "kind": args.kind,
        "workspace_mib": args.workspace_mib,
        "environment": environment(),
        "builder_sha256": digest(__file__),
    }
    if args.precision == "int8":
        if args.calibration is None:
            raise ValueError("INT8 requires training calibration")
        provenance_path = args.calibration.with_name("calibration_provenance.json")
        provenance = json.loads(provenance_path.read_text())
        if provenance["sha256"] != digest(args.calibration) or any(
            r["split"] != "train" for r in provenance["samples"]
        ):
            raise ValueError("Unverified or non-training calibration data")
        contract["calibration_sha256"] = digest(args.calibration)
    metadata_path = args.output.with_suffix(".json")
    layer_path = args.output.with_suffix(".layers.json")
    # A previous approved engine can be reused only with the same inputs/environment.
    if args.output.exists() and metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if (
            previous.get("contract") == contract
            and previous.get("validation", {}).get("passed")
            and previous.get("engine_sha256") == digest(args.output)
            and layer_path.exists()
            and previous.get("layer_information_sha256") == digest(layer_path)
        ):
            print(f"Reusing verified engine: {args.output}")
            return
        raise ValueError(
            "Existing engine does not match this build contract. Choose a new output path."
        )
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(args.onnx.read_bytes()):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    if network.num_inputs != 1 or network.num_outputs != 1:
        raise ValueError("Expected one input and one output")
    for tensor in (network.get_input(0), network.get_output(0)):
        tensor.dtype = trt.float32
        tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
    config = builder.create_builder_config()
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib * 1024**2)
    config.clear_flag(trt.BuilderFlag.TF32)
    if args.precision != "fp32":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("This GPU does not advertise fast FP16")
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    constrained = []
    if args.kind == "restormer" and args.precision != "fp32":
        # Preserve numerically sensitive transformer math. Convolutions may use FP16/INT8.
        sensitive = {
            "REDUCE",
            "SOFTMAX",
            "MATRIX_MULTIPLY",
            "ELEMENTWISE",
            "UNARY",
            "NORMALIZATION",
        }
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if str(layer.type).split(".")[-1] in sensitive:
                floating = [
                    j
                    for j in range(layer.num_outputs)
                    if layer.get_output(j).dtype in (trt.float32, trt.float16)
                    and not layer.get_output(j).is_shape_tensor
                ]
                if floating:
                    layer.precision = trt.float32
                    for j in floating:
                        layer.set_output_type(j, trt.float32)
                    constrained.append(layer.name)
    calibrator = None
    if args.precision == "int8":
        if args.kind != "restormer" or args.calibration is None:
            raise ValueError(
                "INT8 requires Restormer training-only calibration.npz; YOLO INT8 "
                "is not calibrated by this kit"
            )
        if not builder.platform_has_fast_int8:
            raise RuntimeError("This GPU does not advertise fast INT8")
        contract["calibration_sha256"] = digest(args.calibration)
        # No cache reuse: calibration always corresponds to the exact input archive.
        input_shape = tuple(network.get_input(0).shape)
        input_name = network.get_input(0).name

        class Calibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self):
                super().__init__()
                self.archive = np.load(args.calibration, allow_pickle=False)
                self.data = self.archive["inputs"]
                if self.data.shape[1:] != input_shape:
                    raise ValueError("Calibration tensor shape does not match graph")
                if not np.isfinite(self.data).all() or self.data.min() < -1 or self.data.max() > 1:
                    raise ValueError("Invalid calibration values")
                self.position = 0
                self.cuda = Cuda()
                self.pointer = self.cuda.malloc(self.data[0].nbytes)

            def get_batch_size(self):
                return 1

            def get_batch(self, names):
                if names != [input_name]:
                    raise ValueError("Unexpected calibration input names")
                if self.position == len(self.data):
                    return None
                batch = np.ascontiguousarray(self.data[self.position], dtype=np.float32)
                self.cuda.upload(self.pointer, batch)
                self.position += 1
                return [self.pointer]

            def read_calibration_cache(self):
                return None

            def write_calibration_cache(self, cache):
                args.output.with_suffix(".calibration.cache").write_bytes(cache)

            def close(self):
                self.cuda.free(self.pointer)
                self.archive.close()

        calibrator = Calibrator()
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator
    try:
        serialized = builder.build_serialized_network(network, config)
    finally:
        if calibrator is not None:
            calibrator.close()
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed. Read the preceding parser/tactic log.")
    temporary = args.output.with_suffix(".candidate.engine")
    temporary.write_bytes(bytes(serialized))
    # Free builder resources before allocating inference context on a 4 GB device.
    del serialized, config, parser, network, builder
    validation = validate(temporary, args.fixtures, args.kind, args.precision)
    # Flags permit precision choices; they do not prove all layers use INT8/FP16.
    # Preserve actual layer/tactic information for the compression experiment.
    with TrtSession(temporary) as session:
        inspector = session.engine.create_engine_inspector()
        inspector.execution_context = session.context
        layer_info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    layer_path.write_text(layer_info + "\n")
    record = {
        "contract": contract,
        "fp32_constrained_layers": constrained,
        "validation": validation,
        "engine_sha256": digest(temporary),
        "engine_bytes": temporary.stat().st_size,
        "layer_information_sha256": digest(layer_path),
        "precision_note": "Requested precision is not layer coverage; inspect .layers.json",
    }
    if validation["passed"]:
        temporary.replace(args.output)
        write_json(metadata_path, record)
        print(f"Validated engine: {args.output}", flush=True)
    else:
        write_json(args.output.with_suffix(".rejected.json"), record)
        raise RuntimeError(
            "Candidate failed numerical gate and was NOT promoted. Build FP32 "
            "with a different filename."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("restormer", "yolov8n"), required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp16")
    parser.add_argument("--workspace-mib", type=int, default=512)
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    if args.workspace_mib <= 0:
        parser.error("Workspace must be positive")
    build(args)
