#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from aegis_perception.checkpoints import load_generator, select_device
from aegis_perception.config import load_config, validate_restoration_config
from aegis_perception.data.catalog import sha256
from aegis_perception.data.paired import PairedImageDataset
from aegis_perception.inference import predict
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
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    config = load_config(args.config)
    validate_restoration_config(config)
    device = select_device(args.device)
    precision = config["train"].get("precision", "fp32") if device.startswith("cuda") else "fp32"
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("Configured evaluation precision is bf16, but the GPU does not support it")
    generator, checkpoint = load_generator(args.checkpoint, device=device)
    if config["model"] != checkpoint["config"]["model"]:
        raise SystemExit("Evaluation model config does not match checkpoint")
    manifest_hash = sha256(Path(config["data"]["manifest"]))
    if checkpoint.get("manifest_sha256", manifest_hash) != manifest_hash:
        raise SystemExit("Evaluation manifest differs from the checkpoint's split definition")
    dataset = PairedImageDataset(
        config["data"]["manifest"],
        args.split,
        image_size=tuple(config["data"]["image_size"]),
        random_flip=False,
        spatial_mode=config["data"].get("eval_mode", "resize"),
        random_crop=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    per_weather: dict[str, list[dict[str, float]]] = defaultdict(list)
    per_sample: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Evaluating {args.split}", unit="image"):
            prediction = (
                predict(generator, batch["input"].to(device), precision=precision)[0]
                .add(1)
                .div(2)
                .clamp(0, 1)
            )
            target = batch["target"][0].add(1).div(2).clamp(0, 1)
            raw = batch["input"][0].add(1).div(2).clamp(0, 1).permute(1, 2, 0).numpy()
            predicted_array = prediction.cpu().permute(1, 2, 0).numpy().astype(np.float32)
            target_array = target.cpu().permute(1, 2, 0).numpy().astype(np.float32)
            values = {
                "psnr_db": psnr(target_array, predicted_array),
                "ssim": ssim(target_array, predicted_array),
                "raw_psnr_db": psnr(target_array, raw),
                "raw_ssim": ssim(target_array, raw),
            }
            values["psnr_gain_db"] = values["psnr_db"] - values["raw_psnr_db"]
            weather = batch["weather"][0]
            row: dict[str, object] = {
                "sample_id": batch["sample_id"][0],
                "weather": weather,
                **values,
            }
            per_sample.append(row)
            per_weather[weather].append(values)

    summary = {
        weather: {metric: float(np.mean([row[metric] for row in rows])) for metric in values}
        for weather, rows in sorted(per_weather.items())
    }
    summary["all"] = {
        metric: float(np.mean([row[metric] for row in per_sample])) for metric in values
    }
    output = args.output or Path(config["run"]["output_dir"]) / f"metrics_{args.split}.json"
    payload = {
        "environment": environment_record(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "manifest_sha256": manifest_hash,
        "protocol": {
            "color": "RGB",
            "data_range": [0, 1],
            "spatial_mode": config["data"].get("eval_mode", "resize"),
            "precision": precision,
            "image_size": config["data"]["image_size"],
            "aggregation": "mean per-image",
            "mse_floor": 1e-12,
        },
        "split": args.split,
        "summary": summary,
        "samples": per_sample,
    }
    write_json(output, payload)
    print(summary)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
