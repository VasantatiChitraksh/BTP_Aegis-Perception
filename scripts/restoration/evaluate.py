#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from aegis_perception.checkpoints import load_generator
from aegis_perception.config import load_config, validate_restoration_config
from aegis_perception.data.catalog import sha256
from aegis_perception.data.paired import PairedImageDataset
from aegis_perception.data.protocol import verify_protocol
from aegis_perception.distributed import distributed_session, gather_objects
from aegis_perception.evaluation import clustered_intervals, summarize_samples
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
    parser.add_argument(
        "--panel", type=Path, help="Frozen subset of this split, selected without scores"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with distributed_session(args.device) as context:
        evaluate(args, *context)


def evaluate(args, device, rank, world_size):
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    config = load_config(args.config)
    validate_restoration_config(config)
    verify_protocol(config["data"])
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
    panel = None
    if args.panel:
        panel = json.loads(args.panel.read_text())
        if panel["split"] != args.split or manifest_hash not in panel["compatible_manifest_sha256"]:
            raise SystemExit("Panel split or manifest provenance mismatch")
        ids = panel["sample_ids"]
        available = {r.sample_id for r in dataset.records}
        if len(ids) != len(set(ids)) or not set(ids) <= available:
            raise SystemExit("Panel contains duplicate or unknown sample IDs")
        selected = set(ids)
        dataset.records = [r for r in dataset.records if r.sample_id in selected]
    dataset.records = dataset.records[rank::world_size]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    per_sample: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Evaluating {args.split}", unit="image", disable=rank != 0):
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
            values["ssim_gain"] = values["ssim"] - values["raw_ssim"]
            weather = batch["weather"][0]
            row: dict[str, object] = {
                "sample_id": batch["sample_id"][0],
                "group_id": batch["group_id"][0],
                "stratum": batch["stratum"][0],
                "cluster_id": batch["cluster_id"][0],
                "weather": weather,
                **values,
            }
            per_sample.append(row)
    per_sample = [row for part in gather_objects(per_sample) for row in part]
    if rank != 0:
        return
    per_sample.sort(key=lambda row: row["sample_id"])
    summaries = summarize_samples(per_sample)
    suffix = "_panel" if panel else ""
    output = args.output or Path(config["run"]["output_dir"]) / f"metrics_{args.split}{suffix}.json"
    payload = {
        "environment": environment_record(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "training_seed": checkpoint["config"]["run"]["seed"],
        "condition_sha256": hashlib.sha256(
            json.dumps(
                {k: checkpoint["config"][k] for k in ("model", "data", "train")}, sort_keys=True
            ).encode()
        ).hexdigest(),
        "manifest_sha256": manifest_hash,
        "protocol": {
            "color": "RGB",
            "data_range": [0, 1],
            "spatial_mode": config["data"].get("eval_mode", "resize"),
            "precision": precision,
            "image_size": config["data"]["image_size"],
            "aggregation": summaries["headline"]["aggregation"],
            "ssim": "RGB uniform 7x7, sample covariance, K1=0.01 K2=0.03",
            "mse_floor": 1e-12,
        },
        "split": args.split,
        **summaries,
        "uncertainty": clustered_intervals(per_sample),
        "panel_sha256": sha256(args.panel) if args.panel else None,
        "samples": per_sample,
    }
    write_json(output, payload)
    print("Primary score:", summaries["headline"])
    print("Per-weather counts and regressions:", summaries["counts"])
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
