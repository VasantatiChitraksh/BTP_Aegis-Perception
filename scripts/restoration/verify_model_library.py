#!/usr/bin/env python3
"""Verify the six report checkpoints, then strictly load and optionally smoke-test them."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Also run CPU FP32 on synthetic RGB")
    args = parser.parse_args()

    import torch

    from aegis_perception.checkpoints import load_generator
    from aegis_perception.official_runtime import activate

    torch.set_num_threads(2)
    activate()
    index = json.loads((ROOT / "artifacts/models/index.json").read_text())
    for name, entry in index["models"].items():
        path = ROOT / entry["library"] / "inference.pt"
        if digest(path) != entry["inference_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {name}")
        model, checkpoint = load_generator(path, device="cpu")
        if sum(p.numel() for p in model.parameters()) != entry["parameters"]:
            raise ValueError(f"Parameter count mismatch: {name}")
        expected_config = json.loads((path.parent / "config.json").read_text())
        if checkpoint["config"] != expected_config:
            raise ValueError(f"Configuration mismatch: {name}")
        if checkpoint["epoch"] != entry["best_epoch"]:
            raise ValueError(f"Selected epoch mismatch: {name}")
        if args.smoke:
            # 256 is compatible with all six architectures, including the Pix2Pix depth.
            with torch.inference_mode():
                output = model(torch.zeros(1, 3, 256, 256))
            if output.shape != (1, 3, 256, 256) or not torch.isfinite(output).all():
                raise ValueError(f"Invalid smoke output: {name}")
        print(f"Verified {name}: epoch {entry['best_epoch']}, strict load passed", flush=True)
        del model, checkpoint
    print("All six report checkpoints verified. Synthetic smoke checks are not quality benchmarks.")


if __name__ == "__main__":
    main()
