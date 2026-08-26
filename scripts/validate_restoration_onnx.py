#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from aegis_perception.checkpoints import load_generator


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PyTorch and ONNX restoration outputs")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    import onnxruntime as ort
    import torch

    generator, checkpoint = load_generator(args.checkpoint, device="cpu")
    height, width = checkpoint["config"]["data"]["image_size"]
    rng = np.random.default_rng(42)
    sample = rng.uniform(-1, 1, size=(1, 3, height, width)).astype(np.float32)
    with torch.inference_mode():
        expected = generator(torch.from_numpy(sample)).numpy()
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: sample})[0]
    maximum = float(np.max(np.abs(expected - actual)))
    np.testing.assert_allclose(actual, expected, atol=args.atol, rtol=args.rtol)
    print(f"ONNX validation passed; max_abs_error={maximum:.8f}")


if __name__ == "__main__":
    main()
