#!/usr/bin/env python3
"""export_trt_edge.py — Build TensorRT engine with custom quantization scales for Jetson.

Usage:
------
python edge_pipeline/quantization_scripts/export_trt_edge.py \\
    --checkpoint edge_pipeline/checkpoints/restormer/restormer_imported.pt \\
    --output-dir artifacts/edge_v1/engines \\
    --precision fp16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from aegis_perception.checkpoints import load_generator


def export_onnx_intermediate(
    checkpoint_path: Path,
    output_onnx: Path,
    height: int = 256,
    width: int = 448,
) -> Path:
    """Export Restormer model to static ONNX format for TensorRT compilation."""
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Loading checkpoint for ONNX export: {checkpoint_path}")
    
    ckpt_raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_config" in ckpt_raw and "state_dict" in ckpt_raw:
        from aegis_perception.models import build_generator
        model = build_generator(ckpt_raw["model_config"])
        model.load_state_dict(ckpt_raw["state_dict"])
    else:
        model, _ = load_generator(checkpoint_path, device="cpu")

    model.eval()
    dummy = torch.zeros(1, 3, height, width, dtype=torch.float32)

    print(f"[INFO] Exporting static ONNX graph ({height}x{width}) to: {output_onnx}")
    torch.onnx.export(
        model,
        dummy,
        str(output_onnx),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"[SUCCESS] ONNX graph exported successfully: {output_onnx}")
    return output_onnx


def print_jetson_trtexec_command(
    onnx_path: Path,
    output_engine: Path,
    precision: str = "fp16",
    workspace_mb: int = 1024,
) -> None:
    """Print the exact trtexec command for compilation directly on Jetson."""
    print("\n" + "=" * 75)
    print(">>> NATIVE TENSORRT BUILD COMMAND FOR JETSON TERMINAL <<<")
    print("=" * 75)
    cmd = [
        "/usr/src/tensorrt/bin/trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={output_engine}",
        f"--memPoolSize=workspace:{workspace_mb}MiB",
    ]
    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "int8":
        cmd.extend(["--int8", "--fp16", "--best"])

    print("Execute this command directly on your Jetson Orin Nano / JetAuto Pro:")
    print("\n" + " ".join(cmd) + "\n")
    print("=" * 75 + "\n")


def build_engine_on_device(
    onnx_path: Path,
    output_engine: Path,
    precision: str = "fp16",
    workspace_mb: int = 1024,
) -> None:
    """Compile TensorRT engine using the python TensorRT API if available."""
    try:
        import tensorrt as trt
    except ImportError:
        print("[INFO] Python 'tensorrt' module not found in host environment.")
        print_jetson_trtexec_command(onnx_path, output_engine, precision, workspace_mb)
        return

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"[TRT Error] {parser.get_error(error)}")
            sys.exit(1)

    config = builder.create_builder_config()
    workspace_bytes = workspace_mb * 1024 * 1024
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes

    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[INFO] Enabled FP16 TensorRT mode.")
    elif precision == "int8" and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        print("[INFO] Enabled INT8 TensorRT mode.")

    output_engine.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Building TensorRT Engine (this may take 3-5 minutes on Jetson)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT serialized network.")

    with open(output_engine, "wb") as f:
        f.write(serialized_engine)
    print(f"[SUCCESS] TensorRT Engine saved to: {output_engine}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Restormer to ONNX and TensorRT for edge deployment")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to imported Restormer checkpoint")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "artifacts" / "edge_v1" / "engines",
        help="Directory to save exported ONNX and TensorRT engine",
    )
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16", help="Precision target")
    parser.add_argument("--height", type=int, default=256, help="Input height")
    parser.add_argument("--width", type=int, default=448, help="Input width")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_file = args.output_dir / "restormer.onnx"
    engine_file = args.output_dir / f"restormer_{args.precision}.engine"

    export_onnx_intermediate(args.checkpoint, onnx_file, height=args.height, width=args.width)
    build_engine_on_device(onnx_file, engine_file, precision=args.precision)


if __name__ == "__main__":
    main()
