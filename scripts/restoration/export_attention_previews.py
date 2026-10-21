#!/usr/bin/env python3
"""Export genuine CPU previews from the completed GAN; no DGX GPU allocation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aegis_perception.checkpoints import load_generator
from aegis_perception.data.catalog import sha256
from aegis_perception.data.manifest import load_manifest
from aegis_perception.data.paired import normalized_tensor_to_pil, pil_to_normalized_tensor
from aegis_perception.inference import predict
from aegis_perception.metrics import psnr, ssim


def main():
    run = ROOT / "artifacts/restoration/baselines_v1/attention_pix2pix_seed42"
    out = run / "report"
    out.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((run / "metrics_test.json").read_text())
    checkpoint = run / "best.pt"
    digest = sha256(checkpoint)
    if digest != metrics["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the evaluated checkpoint (or copy is incomplete)")
    manifest = ROOT / "data/manifests/weather_v3/active.csv"
    if sha256(manifest) != metrics["manifest_sha256"]:
        raise ValueError("Manifest differs from the evaluated split")
    records = {r.sample_id: r for r in load_manifest(manifest, "test")}
    selected = []
    for weather in ("rain", "fog", "snow", "smoke"):
        rows = sorted(
            (r for r in metrics["samples"] if r["weather"] == weather),
            key=lambda r: (r["psnr_gain_db"], r["sample_id"]),
        )
        selected.append({**rows[len(rows) // 2], "selection": "median PSNR gain within weather"})
    worst = min(
        (r for r in metrics["samples"] if r["weather"] == "smoke"),
        key=lambda r: (r["ssim_gain"], r["sample_id"]),
    )
    if worst["sample_id"] not in {r["sample_id"] for r in selected}:
        selected.append({**worst, "selection": "largest smoke SSIM loss"})
    torch.set_num_threads(4)
    model, state = load_generator(checkpoint, device="cpu")
    if state["epoch"] != metrics["checkpoint_epoch"]:
        raise ValueError("Checkpoint epoch does not match evaluation")
    del state  # Release discriminator/optimizer storage before native-resolution inference.
    previews = []
    for i, row in enumerate(selected):
        record = records[row["sample_id"]]
        with Image.open(record.input_path) as source:
            tensor = pil_to_normalized_tensor(source.convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            output = predict(model, tensor, precision="fp32")[0]
        array = output.clamp(-1, 1).add(1).div(2).permute(1, 2, 0).numpy()
        with Image.open(record.target_path) as target:
            reference = np.asarray(target.convert("RGB"), dtype=np.float32) / 255
        filename = f"restored_{i}_{record.weather}.png"
        normalized_tensor_to_pil(output).save(out / filename)
        previews.append(
            {
                **row,
                "prediction": filename,
                "preview_fp32_psnr_db": psnr(reference, array),
                "preview_fp32_ssim": ssim(reference, array),
            }
        )
        print(f"Exported {record.weather}: {record.sample_id}", flush=True)
    (out / "previews.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": digest,
                "precision": "CPU FP32",
                "note": "Visual previews only; reported test scores remain "
                "the saved DGX BF16 results.",
                "samples": previews,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
