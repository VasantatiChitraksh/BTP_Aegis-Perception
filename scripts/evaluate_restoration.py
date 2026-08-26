#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from aegis_perception.checkpoints import load_generator
from aegis_perception.config import load_config, validate_restoration_config
from aegis_perception.data.paired import PairedImageDataset
from aegis_perception.metrics import psnr, ssim
from aegis_perception.reproducibility import environment_record, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate restoration on a fixed split")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    args = parse_args()
    config = load_config(args.config)
    validate_restoration_config(config)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    generator, checkpoint = load_generator(args.checkpoint, device=device)
    dataset = PairedImageDataset(
        config["data"]["manifest"],
        args.split,
        image_size=tuple(config["data"]["image_size"]),
        random_flip=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    per_weather: dict[str, list[dict[str, float]]] = defaultdict(list)
    per_sample: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch in loader:
            prediction = generator(batch["input"].to(device))[0].add(1).div(2).clamp(0, 1)
            target = batch["target"][0].add(1).div(2).clamp(0, 1)
            predicted_array = prediction.cpu().permute(1, 2, 0).numpy().astype(np.float32)
            target_array = target.cpu().permute(1, 2, 0).numpy().astype(np.float32)
            values = {
                "psnr_db": psnr(target_array, predicted_array),
                "ssim": ssim(target_array, predicted_array),
            }
            weather = batch["weather"][0]
            row: dict[str, object] = {
                "sample_id": batch["sample_id"][0],
                "weather": weather,
                **values,
            }
            per_sample.append(row)
            per_weather[weather].append(values)

    summary = {
        weather: {
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in ("psnr_db", "ssim")
        }
        for weather, rows in sorted(per_weather.items())
    }
    summary["all"] = {
        metric: float(np.mean([row[metric] for row in per_sample]))
        for metric in ("psnr_db", "ssim")
    }
    output = args.output or Path(config["run"]["output_dir"]) / f"metrics_{args.split}.json"
    payload = {
        "environment": environment_record(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
        "summary": summary,
        "samples": per_sample,
    }
    write_json(output, payload)
    print(summary)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
