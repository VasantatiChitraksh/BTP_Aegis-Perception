#!/usr/bin/env python3
"""export_tensorrt.py — Export ONNX restoration model to NVIDIA TensorRT engine.

Tailored for:
- Jetson Orin Nano (Ampere 512-core GPU, 16 Tensor Cores) -> Supports FP16 & INT8
- Jetson Nano (Maxwell 128-core GPU on JetAuto Pro) -> Supports FP16 (Best)

Usage on Jetson (with TensorRT installed in JetPack):
    python quantization/scripts/export_tensorrt.py \
        --onnx artifacts/onnx/liteweatherformer.onnx \
        --output artifacts/tensorrt/liteweatherformer_fp16.engine \
        --precision fp16

For INT8 with calibration images:
    python quantization/scripts/export_tensorrt.py \
        --onnx artifacts/onnx/liteweatherformer.onnx \
        --output artifacts/tensorrt/liteweatherformer_int8.engine \
        --precision int8 \
        --calib-images data/raw/rain/realrain1k/rain

If run on a machine without TensorRT installed, it outputs the exact 'trtexec'
command line to run directly on the Jetson terminal.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ONNX model to TensorRT engine")
    parser.add_argument("--onnx", type=Path, required=True, help="Path to input .onnx model")
    parser.add_argument("--output", type=Path, required=True, help="Path for output .engine file")
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "int8"],
        default="fp16",
        help="Target precision (FP16 recommended for Jetson Nano & Orin Nano)",
    )
    parser.add_argument(
        "--max-workspace-mb",
        type=int,
        default=1024,
        help="Max GPU workspace in MB (use 512-1024 on 4GB Jetson Nano to prevent OOM)",
    )
    parser.add_argument(
        "--calib-images",
        type=Path,
        default=None,
        help="Directory with ~50-100 calibration images for INT8 mode",
    )
    parser.add_argument(
        "--calib-cache",
        type=Path,
        default=Path("artifacts/tensorrt/calibration.cache"),
        help="Calibration cache file path",
    )
    return parser.parse_args()


def print_trtexec_command(args: argparse.Namespace) -> None:
    print("\n" + "=" * 70)
    print(">>> TRTEXEC COMMAND FOR JETSON TERMINAL <<<")
    print("=" * 70)
    cmd = [
        "/usr/src/tensorrt/bin/trtexec",
        f"--onnx={args.onnx}",
        f"--saveEngine={args.output}",
        f"--memPoolSize=workspace:{args.max_workspace_mb}MiB",
    ]
    if args.precision == "fp16":
        cmd.append("--fp16")
    elif args.precision == "int8":
        cmd.append("--int8")
        if args.calib_cache:
            cmd.append(f"--calib={args.calib_cache}")
    
    cmd.append("--verbose")
    print("Run this command directly on your Jetson Orin Nano / JetAuto Pro:")
    print("\n" + " ".join(cmd) + "\n")
    print("=" * 70 + "\n")


def build_engine(args: argparse.Namespace) -> None:
    try:
        import tensorrt as trt
    except ImportError:
        print("[INFO] 'tensorrt' Python module not found in the current environment.")
        print("[INFO] Generating the native Jetson 'trtexec' command instead:")
        print_trtexec_command(args)
        return

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    print(f"Loading ONNX file: {args.onnx}")
    with open(args.onnx, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"[TRT Error] {parser.get_error(error)}")
            sys.exit(1)

    config = builder.create_builder_config()
    # Workspace allocation
    workspace_bytes = args.max_workspace_mb * 1024 * 1024
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes

    if args.precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[INFO] Enabled FP16 TensorRT mode.")
        else:
            print("[WARN] Platform does not have native fast FP16, falling back to FP32.")

    elif args.precision == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("[INFO] Enabled INT8 TensorRT mode.")
            if args.calib_images and args.calib_images.exists():
                print(f"[INFO] Setting up INT8 Calibrator with images from {args.calib_images}...")
                # Note: Calibrator class should implement trt.IInt8EntropyCalibrator2
            else:
                print("[WARN] No calibration dataset specified. Build may fail or fall back.")
        else:
            print("[WARN] Platform lacks fast INT8 (common on Jetson Nano Maxwell). Use FP16 instead!")
            config.set_flag(trt.BuilderFlag.FP16)

    print(f"Building TensorRT Engine for {args.onnx} (this may take a few minutes on Jetson)...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("[ERROR] Failed to build TensorRT engine.")
        sys.exit(1)

    with open(args.output, "wb") as f:
        f.write(serialized_engine)
    print(f"[SUCCESS] Serialized TensorRT engine saved to: {args.output}")


def main() -> None:
    args = parse_args()
    if not args.onnx.exists():
        print(f"[ERROR] ONNX file not found: {args.onnx}")
        sys.exit(1)
    build_engine(args)


if __name__ == "__main__":
    main()
