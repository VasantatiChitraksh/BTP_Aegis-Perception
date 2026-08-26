#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from aegis_perception.checkpoints import load_generator


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a restoration generator to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic-batch", action="store_true")
    args = parser.parse_args()

    import torch

    generator, checkpoint = load_generator(args.checkpoint, device="cpu")
    height, width = checkpoint["config"]["data"]["image_size"]
    example = torch.randn(1, 3, height, width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"degraded": {0: "batch"}, "restored": {0: "batch"}}
    torch.onnx.export(
        generator,
        example,
        args.output,
        input_names=["degraded"],
        output_names=["restored"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"exported {args.output}")


if __name__ == "__main__":
    main()
