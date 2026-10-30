#!/usr/bin/env python3
"""Build and check trained Restormer on TensorRT 8.2; Python 3.6 compatible.

Used with nano_yolo_demo.py. No training, PyTorch, or pip installs required.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from nano_yolo_demo import Engine, digest, write_json

GRAPH_SHA256 = "2761e778fc638727f4ae2bc9d0fb077ed414de84a314a6f12f6c578c061c026c"
FIXTURE_SHA256 = "01aea6e12d4cce8cc3b94e36730d223b6d3b305a2e93df04a71a2b468b06b6a4"
SMALL_FIXTURE_SHA256 = "992002fb730d3b0ec03153e96cc6ec904e51010b94e61ca59972ff86a73bd0e6"
PROFILES = {
    FIXTURE_SHA256: GRAPH_SHA256,
    SMALL_FIXTURE_SHA256: "283f9f437d74e06b256ddcc88a9a1fa150f94d58e96b47d3608de247fb7cb725",
}


def quality(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    psnr = -10 * np.log10(max(float(np.mean((x - y) ** 2)), 1e-12))

    def mean(value):
        return cv2.blur(value, (7, 7), borderType=cv2.BORDER_REFLECT)

    ux, uy = mean(x), mean(y)
    vx, vy = (mean(x * x) - ux * ux) * 49 / 48, (mean(y * y) - uy * uy) * 49 / 48
    covariance = (mean(x * y) - ux * uy) * 49 / 48
    score = ((2 * ux * uy + 0.01**2) * (2 * covariance + 0.03**2)) / (
        (ux * ux + uy * uy + 0.01**2) * (vx + vy + 0.03**2)
    )
    return float(psnr), float(score[3:-3, 3:-3].mean())


def validate_restorer(engine, fixtures):
    rows = []
    with np.load(str(fixtures), allow_pickle=False) as data:
        for key in sorted(k for k in data.files if k.startswith("input_")):
            actual = engine.run(data[key])
            suffix = key[6:]
            expected = data["expected_" + suffix]
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                raise ValueError("Invalid/nonfinite Restormer output")
            difference = actual - expected
            rmse = float(np.sqrt(np.mean(difference**2)))
            maximum = float(np.max(np.abs(difference)))
            height, width = data["valid_hw_" + suffix]

            def rgb(value, height=height, width=width):
                return ((np.clip(value[0, :, :height, :width], -1, 1) + 1) / 2).transpose(1, 2, 0)

            baseline = quality(rgb(data["target_" + suffix]), rgb(expected))
            candidate = quality(rgb(data["target_" + suffix]), rgb(actual))
            psnr_delta, ssim_delta = candidate[0] - baseline[0], candidate[1] - baseline[1]
            passed = (
                rmse <= 0.004 and maximum <= 0.05 and psnr_delta >= -0.1 and ssim_delta >= -0.001
            )
            rows.append(
                {
                    "sample": suffix,
                    "rmse": rmse,
                    "max_abs_error": maximum,
                    "psnr_delta_db": psnr_delta,
                    "ssim_delta": ssim_delta,
                    "passed": passed,
                }
            )
            print(
                "Restormer {}: {} (RMSE {:.6f})".format(suffix, "PASS" if passed else "FAIL", rmse),
                flush=True,
            )
    return {
        "passed": bool(rows) and all(r["passed"] for r in rows),
        "rows": rows,
        "scope": "Four-reference acceptance check, not a full benchmark",
    }


def check_restorer(engine, path, fixtures):
    fixture_hash = digest(fixtures)
    if fixture_hash not in PROFILES:
        raise ValueError("Restormer fixtures differ from the trained release")
    contract = {
        "engine_sha256": digest(path),
        "fixtures_sha256": fixture_hash,
        "validator_sha256": digest(__file__),
        "runtime_sha256": digest(Path(__file__).with_name("nano_yolo_demo.py")),
        "tensorrt": engine.version,
    }
    cache = path.with_suffix(".nano-validation.json")
    previous = json.loads(cache.read_text()) if cache.is_file() else {}
    if previous.get("passed") and previous.get("contract") == contract:
        print("Reusing matching Restormer acceptance check.", flush=True)
        return previous
    print("Checking Restormer on four saved inputs once...", flush=True)
    result = validate_restorer(engine, fixtures)
    result["contract"] = contract
    write_json(cache, result)
    return result


def preprocess(bgr, canvas_height=256, canvas_width=448):
    from PIL import Image

    image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    width, height = image.size
    scale = min(float(canvas_width) / width, float(canvas_height) / height)
    new_w = max(1, min(canvas_width, round(width * scale)))
    new_h = max(1, min(canvas_height, round(height * scale)))
    resized = np.asarray(image.resize((new_w, new_h), Image.BILINEAR), dtype=np.float32)
    padded = np.pad(
        resized, ((0, canvas_height - new_h), (0, canvas_width - new_w), (0, 0)), mode="edge"
    )
    value = np.ascontiguousarray((padded / 127.5 - 1).transpose(2, 0, 1)[None])
    return value, (width, height, new_w, new_h)


def postprocess(value, geometry):
    from PIL import Image

    if value.shape not in ((1, 3, 256, 448), (1, 3, 128, 224)) or not np.isfinite(value).all():
        raise ValueError("Invalid Restormer output")
    width, height, new_w, new_h = geometry
    rgb = (np.clip(value[0, :, :new_h, :new_w], -1, 1).transpose(1, 2, 0) + 1) * 127.5
    rgb = rgb.round().astype(np.uint8)
    result = np.asarray(Image.fromarray(rgb).resize((width, height), Image.BILINEAR))
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def restore(engine, frame):
    value, geometry = preprocess(frame, engine.input_shape[2], engine.input_shape[3])
    return postprocess(engine.run(value), geometry)


def build(args):
    import tensorrt as trt

    version = tuple(int(x) for x in trt.__version__.split(".")[:2])
    if version[0] != 8 or version < (8, 2):
        raise RuntimeError("This Nano builder targets TensorRT 8.2 through 8.x")
    if args.workspace < 1:
        raise ValueError("Workspace must be positive")
    fixture_hash, graph_hash = digest(args.fixtures), digest(args.onnx)
    if fixture_hash not in PROFILES or graph_hash != PROFILES[fixture_hash]:
        raise ValueError("Graph/fixtures must match the trained Restormer release")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        engine = Engine(args.output, kind="restormer")
        try:
            return 0 if check_restorer(engine, args.output, args.fixtures)["passed"] else 2
        finally:
            engine.close()
    candidate = args.output.with_suffix(".candidate.engine")
    if candidate.exists():
        raise FileExistsError("Candidate already exists; preserve it and choose a new output name")
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(args.onnx.read_bytes()):
        raise RuntimeError(
            "ONNX parsing failed:\n"
            + "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        )
    if network.num_inputs != 1 or network.num_outputs != 1:
        raise ValueError("Expected one Restormer input and output")
    for tensor in (network.get_input(0), network.get_output(0)):
        tensor.dtype = trt.float32
        tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
    config = builder.create_builder_config()
    config.max_workspace_size = args.workspace * 1024**2
    config.clear_flag(trt.BuilderFlag.TF32)
    if not builder.platform_has_fast_fp16:
        raise RuntimeError("GPU does not advertise fast FP16")
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    sensitive = {"REDUCE", "SOFTMAX", "MATRIX_MULTIPLY", "ELEMENTWISE", "UNARY", "NORMALIZATION"}
    constrained = []
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if str(layer.type).split(".")[-1] in sensitive:
            outputs = [
                j
                for j in range(layer.num_outputs)
                if layer.get_output(j).dtype in (trt.float16, trt.float32)
                and not layer.get_output(j).is_shape_tensor
            ]
            if outputs:
                layer.precision = trt.float32
                for j in outputs:
                    layer.set_output_type(j, trt.float32)
                constrained.append(layer.name)
    print(
        "Building trained Restormer; {} sensitive layers constrained to FP32.".format(
            len(constrained)
        ),
        flush=True,
    )
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Restormer engine build failed; preserve this log")
    candidate.write_bytes(bytes(serialized))
    write_json(
        args.output.with_suffix(".build.json"),
        {
            "onnx_sha256": graph_hash,
            "tensorrt": trt.__version__,
            "workspace_mib": args.workspace,
            "fp32_constrained_layers": constrained,
            "precision": "FP16 permitted; sensitive math FP32; not uniform FP16 or INT8",
        },
    )
    # Release builder allocations before reference inference on the 4GB Nano.
    del serialized, config, parser, network, builder, layer, tensor
    engine = Engine(candidate, kind="restormer")
    try:
        result = check_restorer(engine, candidate, args.fixtures)
    finally:
        engine.close()
    if not result["passed"]:
        print("REJECTED: retained candidate and validation report; no approved engine.", flush=True)
        return 2
    candidate.replace(args.output)
    candidate.with_suffix(".nano-validation.json").replace(
        args.output.with_suffix(".nano-validation.json")
    )
    print("READY: " + str(args.output), flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=int, default=256)
    raise SystemExit(build(parser.parse_args()))
