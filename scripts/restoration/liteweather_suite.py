#!/usr/bin/env python3
"""One-GPU LiteWeatherTransformer v2 experiment on the frozen four-weather split."""

from __future__ import annotations

import gc
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from aegis_perception.liteweather_runtime import activate

activate()
spec = importlib.util.spec_from_file_location(
    "_liteweather_base", ROOT / "scripts/restoration/baseline_suite.py"
)
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)

suite.NAMES = ("liteweatherformer",)
suite.CONFIGS = (Path("configs/restoration/liteweatherformer_v2.yaml"),)
suite.OUTPUT = Path("artifacts/restoration/liteweather_v1")
suite.STAGE_SCRIPT = "scripts/restoration/train_liteweather_stage.sbatch"
suite.LOG_PREFIX = "aegis-lite"
suite.TRAIN_COMMAND = (sys.executable, "scripts/restoration/liteweather_entry.py", "train")
suite.EVALUATE_COMMAND = (sys.executable, "scripts/restoration/liteweather_entry.py", "evaluate")

_base_configs = suite.checked_configs
_base_preflight = suite.preflight
_base_report = suite.report
_base_package = suite.package
_base_stage = suite.run_stage
REFERENCE_CONFIGS = tuple(
    Path(f"configs/restoration/baselines_v1/{n}.yaml") for n in ("attention_pix2pix", "restormer")
)


def checked_configs():
    configs = _base_configs()
    c = configs[0]
    if c["model"].get("architecture_version") != 2 or c["model"]["name"] != "liteweatherformer":
        raise ValueError("This experiment requires the corrected Transformer v2")
    if c["model"].get("pretrained"):
        raise ValueError("No compatible pretrained checkpoint is declared for this custom model")
    for path in REFERENCE_CONFIGS:
        reference = suite.load_config(path)
        if c["data"] != reference["data"]:
            raise ValueError("LiteWeather must retain the exact completed baseline data protocol")
        for key in ("epochs", "batch_size"):
            if c["train"][key] != reference["train"][key]:
                raise ValueError(f"Baseline budget mismatch: {key}")
    return configs


def preflight(configs):
    _base_preflight(configs)
    stamp = suite.OUTPUT / "training_preflight.json"
    if stamp.exists():
        return
    import torch

    from aegis_perception.models import build_generator

    c = configs[0]
    model = build_generator(c["model"]).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["train"]["learning_rate"])
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(c["train"]["batch_size"], 3, *c["data"]["image_size"], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(x)
    loss = (output.float() - 0.8 * x).abs().mean()
    loss.backward()
    gradients = [p.grad for p in model.parameters()]
    if any(g is None or not torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("Missing or nonfinite gradient in GPU training preflight")
    optimizer.step()
    torch.cuda.synchronize()
    suite.write_json(
        stamp,
        {
            "parameters": sum(p.numel() for p in model.parameters()),
            "peak_training_mib": torch.cuda.max_memory_allocated() / 2**20,
            "batch_size": c["train"]["batch_size"],
            "crop": c["data"]["image_size"],
            "note": "Disposable synthetic-input forward/backward step; no saved training weights.",
        },
    )
    del model, optimizer, x, output, loss, gradients
    gc.collect()
    torch.cuda.empty_cache()


def report():
    configs = [suite.load_config(p) for p in REFERENCE_CONFIGS] + checked_configs()
    _base_report(configs, ("attention_pix2pix", "restormer", "liteweatherformer"))


def package(output):
    names = [
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "docs/liteweather_training.txt",
        "scripts/restoration/baseline_suite.py",
        "scripts/restoration/liteweather_suite.py",
        "scripts/restoration/liteweather_entry.py",
        "scripts/restoration/train.py",
        "scripts/restoration/evaluate.py",
        "scripts/restoration/train_liteweather_stage.sbatch",
        "scripts/restoration/submit_liteweather.sh",
        "scripts/data/audit_manifest.py",
        "data/manifests/weather_v3/active.csv",
        "data/manifests/weather_v3/reserve.csv",
        "data/manifests/weather_v3/protocol.json",
        "tests/conftest.py",
        "tests/test_liteweather.py",
        "artifacts/data/liteweather_v1/cpu_checks.json",
    ]
    names += [str(p) for p in REFERENCE_CONFIGS]
    names += [
        f"artifacts/restoration/baselines_v1/{n}_seed42/metrics_test.json"
        for n in ("attention_pix2pix", "restormer")
    ]
    for p in names[-2:]:
        data = json.loads((ROOT / p).read_text())
        if data["manifest_sha256"] != suite.sha256(ROOT / checked_configs()[0]["data"]["manifest"]):
            raise ValueError("Reference results are from a different test manifest")
    _base_package(output, file_names=names)


def run_stage(config_path, through_epoch):
    _base_stage(config_path, through_epoch)
    if through_epoch != suite.BOUNDARIES[-1]:
        return
    import torch

    from aegis_perception.checkpoints import save_checkpoint

    directory = Path(checked_configs()[0]["run"]["output_dir"])
    best = directory / "best.pt"
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    exported = {k: checkpoint[k] for k in ("generator", "config", "epoch", "manifest_sha256")}
    exported["source_checkpoint_sha256"] = suite.sha256(best)
    save_checkpoint(directory / "inference.pt", exported)
    print(
        f"Best checkpoint: {best}; compact inference copy: {directory / 'inference.pt'}", flush=True
    )


suite.checked_configs = checked_configs
suite.preflight = preflight
suite.report = report
suite.package = package
suite.run_stage = run_stage

if __name__ == "__main__":
    suite.main()
