#!/usr/bin/env python3
"""Verify downloaded runs and export matched native-size CPU visual examples."""

from __future__ import annotations

import gc
import hashlib
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
from aegis_perception.evaluation import summarize_samples
from aegis_perception.inference import predict
from aegis_perception.liteweather_runtime import activate
from aegis_perception.metrics import psnr, ssim

OUT = ROOT / "artifacts/reports/three_models_20261021"
RUNS = {
    "gan": ROOT / "artifacts/restoration/baselines_v1/attention_pix2pix_seed42",
    "restormer": ROOT / "artifacts/restoration/baselines_v1/restormer_seed42",
    "lite": ROOT / "artifacts/restoration/liteweather_v1/liteweatherformer_v2_seed42",
}


def main():
    activate()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    manifest = ROOT / "data/manifests/weather_v3/active.csv"
    records = {r.sample_id: r for r in load_manifest(manifest, "test")}
    results, histories = {}, {}
    for name, run in RUNS.items():
        m = json.loads((run / "metrics_test.json").read_text())
        h = json.loads((run / "history.json").read_text())
        assert m["manifest_sha256"] == sha256(manifest)
        assert m["checkpoint_sha256"] == sha256(run / "best.pt")
        assert [int(r["epoch"]) for r in h] == list(range(1, 61))
        assert len(m["samples"]) == len(records)
        assert {r["sample_id"] for r in m["samples"]} == set(records)
        for row in m["samples"]:
            record = records[row["sample_id"]]
            for key in ("weather", "group_id", "stratum", "cluster_id"):
                assert row[key] == getattr(record, key)
        recomputed = summarize_samples(m["samples"])
        for weather, values in recomputed["scene_balanced_summary"].items():
            for key, value in values.items():
                assert abs(value - m["scene_balanced_summary"][weather][key]) < 1e-9
        chosen = max(h, key=lambda r: r["val_macro_scene_psnr_db"])
        assert int(chosen["epoch"]) == m["checkpoint_epoch"]
        results[name], histories[name] = m, h
    reference = {r["sample_id"]: r for r in results["gan"]["samples"]}
    for m in results.values():
        assert m["protocol"] == results["gan"]["protocol"]
        for row in m["samples"]:
            for metric in ("raw_psnr_db", "raw_ssim"):
                assert abs(row[metric] - reference[row["sample_id"]][metric]) < 1e-9
    # Fixed score-blind selection; a size cap keeps native CPU visualization affordable.
    selected = []
    for weather in ("rain", "fog", "snow", "smoke"):
        candidates = []
        for record in records.values():
            if record.weather == weather:
                with Image.open(record.input_path) as im:
                    if im.width * im.height <= 500000:
                        candidates.append(record)
        record = min(candidates, key=lambda r: hashlib.sha256(r.sample_id.encode()).hexdigest())
        selected.append(record)
        for label, path in [("input", record.input_path), ("target", record.target_path)]:
            with Image.open(path) as im:
                im.convert("RGB").save(OUT / f"{weather}_{label}.png")
    audit = {
        "date": "2026-10-21",
        "manifest_sha256": sha256(manifest),
        "models": {},
        "selection": "One test image/weather: lowest SHA256(sample_id) among images "
        "<=500000 pixels; independent of model scores. Not a representative visual survey.",
        "preview_precision": "CPU FP32; official metrics remain native DGX BF16",
        "examples": [
            {
                "weather": r.weather,
                "sample_id": r.sample_id,
                "source_input": str(r.input_path),
                "source_target": str(r.target_path),
            }
            for r in selected
        ],
    }
    for name, run in RUNS.items():
        print(f"Loading verified {name} checkpoint", flush=True)
        model, state = load_generator(run / "best.pt", device="cpu")
        assert state["epoch"] == results[name]["checkpoint_epoch"]
        audit["models"][name] = {
            "checkpoint_sha256": results[name]["checkpoint_sha256"],
            "metrics_sha256": sha256(run / "metrics_test.json"),
            "history_sha256": sha256(run / "history.json"),
            "parameters": sum(p.numel() for p in model.parameters()),
            "best_pt_bytes": (run / "best.pt").stat().st_size,
            "epoch_train_val_hours": sum(r["epoch_seconds"] for r in histories[name]) / 3600,
            "config": state["config"],
            "previews": [],
        }
        del state
        for record in selected:
            path = OUT / f"{record.weather}_{name}.png"
            with Image.open(record.input_path) as im:
                x = pil_to_normalized_tensor(im.convert("RGB")).unsqueeze(0)
            with torch.inference_mode():
                output = predict(model, x)[0]
            assert output.shape == x[0].shape and torch.isfinite(output).all()
            array = output.clamp(-1, 1).add(1).div(2).permute(1, 2, 0).numpy()
            with Image.open(record.target_path) as im:
                target = np.asarray(im.convert("RGB"), dtype=np.float32) / 255
            normalized_tensor_to_pil(output).save(path)
            audit["models"][name]["previews"].append(
                {
                    "sample_id": record.sample_id,
                    "file": path.name,
                    "cpu_fp32_psnr_db": psnr(target, array),
                    "cpu_fp32_ssim": ssim(target, array),
                }
            )
            print(f"  {record.weather}: {record.sample_id}", flush=True)
            del x, output, array, target
        del model
        gc.collect()
    (OUT / "verified_report_data.json").write_text(
        json.dumps({"audit": audit, "results": results, "histories": histories}, indent=2) + "\n"
    )
    print(
        "Verified checkpoint hashes, identical test IDs/protocol/raw scores, complete histories, "
        "and recomputed scores. All 12 restored images exported.",
        flush=True,
    )


if __name__ == "__main__":
    main()
