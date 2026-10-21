from __future__ import annotations

import copy
import json
import runpy
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from aegis_perception.checkpoints import load_generator
from aegis_perception.config import ConfigError, load_config
from aegis_perception.data.manifest import PairRecord, write_manifest
from aegis_perception.inference import predict
from aegis_perception.liteweather_runtime import (
    activate,
    build_generator,
    validate_restoration_config,
)
from aegis_perception.models.restormer import Attention
from aegis_perception.training import train_restoration

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def single_thread(monkeypatch):
    activate()
    # Match CPU reduction order in the parent and staged CLI subprocesses.
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def tiny_model():
    return dict(
        name="liteweatherformer",
        architecture_version=2,
        dim=4,
        num_weather_types=4,
        num_blocks=[1, 1],
        heads=[1, 2],
    )


def test_real_attention_prompt_gradients_and_native_shape():
    model = build_generator(tiny_model())
    assert sum(isinstance(m, Attention) for m in model.modules()) == 3
    x = torch.randn(2, 3, 32, 40)
    (model(x) - 0.8 * x).abs().mean().backward()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert torch.all(model.weather_prompts.grad.norm(dim=1) > 0)
    assert model.prompt_router.weight.grad.abs().sum() > 0
    with torch.inference_mode():
        image = torch.randn(1, 3, 37, 43)
        out = predict(model.eval(), image)
        assert out.shape == image.shape and torch.isfinite(out).all()


def test_legacy_loading_remains_explicit_and_new_config_matches_frozen_budget(monkeypatch):
    monkeypatch.chdir(ROOT)
    legacy = build_generator(dict(name="liteweatherformer", dim=4, num_weather_types=4))
    assert not any(isinstance(m, Attention) for m in legacy.modules())
    suite = runpy.run_path(str(ROOT / "scripts/restoration/liteweather_suite.py"))["suite"]
    config = suite.checked_configs()[0]
    assert config["model"]["architecture_version"] == 2
    assert (
        config["data"]
        == load_config(ROOT / "configs/restoration/baselines_v1/restormer.yaml")["data"]
    )
    assert config["train"]["epochs"] * config["data"]["samples_per_epoch"] // 4 == 61440
    assert config["train"]["save_periodic_checkpoints"]
    assert config["train"]["checkpoint_every"] == 20
    broken = copy.deepcopy(config)
    broken["model"]["heads"] = [7, 8]
    with pytest.raises(ConfigError, match="heads"):
        validate_restoration_config(broken)


def fixture_config(tmp_path):
    rng = np.random.default_rng(42)
    records = []
    for weather in ("rain", "fog", "snow", "smoke"):
        for split in ("train", "val", "test"):
            name = f"{weather}_{split}"
            src, target = tmp_path / f"{name}.png", tmp_path / f"{name}_gt.png"
            pixels = rng.integers(0, 256, (35, 41, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(src)
            Image.fromarray(pixels // 2 + 50).save(target)
            records.append(PairRecord(name, src, target, split, weather, "fixture", name))
    manifest = tmp_path / "pairs.csv"
    write_manifest(manifest, records)
    return dict(
        run=dict(name="fixture", seed=42, device="cpu", output_dir=str(tmp_path / "staged")),
        data=dict(
            manifest=str(manifest),
            image_size=[32, 32],
            train_mode="crop",
            eval_mode="native",
            random_flip=True,
            num_workers=0,
            sampling="weighted_seeded",
            weather_probabilities=dict.fromkeys(("rain", "fog", "snow", "smoke"), 0.25),
            samples_per_epoch=4,
        ),
        model=tiny_model(),
        train=dict(
            epochs=2,
            batch_size=2,
            learning_rate=1e-4,
            betas=[0.9, 0.999],
            optimizer="adamw",
            scheduler="cosine",
            min_learning_rate=1e-6,
            gan_weight=0.0,
            l1_weight=1.0,
            validation_ssim=True,
            selection_metric="macro_scene_psnr_db",
            checkpoint_every=1,
        ),
    )


def test_actual_stages_resume_exactly_evaluate_and_export(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    config = fixture_config(tmp_path)
    continuous = copy.deepcopy(config)
    continuous["run"]["output_dir"] = str(tmp_path / "continuous")
    train_restoration(continuous)
    path = tmp_path / "fixture.yaml"
    path.write_text(yaml.safe_dump(config))
    ns = runpy.run_path(str(ROOT / "scripts/restoration/liteweather_suite.py"))
    suite = ns["suite"]
    for key, value in dict(
        CONFIGS=(path,),
        BOUNDARIES=(1, 2),
        OUTPUT=tmp_path / "summary",
        checked_configs=lambda: [config],
        verify_release=lambda: None,
        preflight=lambda _: None,
    ).items():
        monkeypatch.setattr(suite, key, value)
    # The report callback includes archived baselines only in the real package.
    monkeypatch.setattr(
        suite, "report", lambda: ns["_base_report"]([config], ["liteweatherformer"])
    )
    actual_run = subprocess.run
    calls = []

    def execute(command, **kwargs):
        command = list(command)
        if "evaluate" in command or "scripts/restoration/evaluate.py" in command:
            command[command.index("--device") + 1] = "cpu"
        calls.append(command)
        return actual_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", execute)
    # Export uses the same validated configuration as the stage.
    monkeypatch.setitem(ns["run_stage"].__globals__, "checked_configs", lambda: [config])
    suite.run_stage(path, 1)
    suite.run_stage(path, 2)
    a = torch.load(tmp_path / "continuous/last.pt", weights_only=False)
    b = torch.load(tmp_path / "staged/last.pt", weights_only=False)
    for key, value in a["generator"].items():
        torch.testing.assert_close(value, b["generator"][key], rtol=0, atol=0)
    assert a["schedulers"] == b["schedulers"]
    assert len(calls) == 3
    assert (tmp_path / "staged/epoch_001.pt").is_file()
    assert (tmp_path / "staged/epoch_002.pt").is_file()
    full, _ = load_generator(tmp_path / "staged/best.pt", device="cpu")
    compact, state = load_generator(tmp_path / "staged/inference.pt", device="cpu")
    assert "optimizer_g" not in state
    image = torch.randn(1, 3, 37, 43)
    with torch.inference_mode():
        torch.testing.assert_close(predict(full, image), predict(compact, image), rtol=0, atol=0)
    assert len(json.loads((tmp_path / "staged/metrics_test.json").read_text())["samples"]) == 4
    suite.run_stage(path, 2)
    assert len(calls) == 3


def test_submission_queues_only_three_liteweather_stages(tmp_path, monkeypatch):
    suite = runpy.run_path(str(ROOT / "scripts/restoration/liteweather_suite.py"))["suite"]
    monkeypatch.setattr(suite, "OUTPUT", tmp_path / "jobs")
    monkeypatch.setattr(suite, "verify_release", lambda: None)
    monkeypatch.setattr(
        suite,
        "checked_configs",
        lambda: [{"model": {}, "run": {"output_dir": str(tmp_path / "run")}}],
    )
    monkeypatch.setenv("USER", "fixture")
    calls = []

    def submit(command, **kwargs):
        calls.append(command)
        return f"{100 + len(calls)}\n"

    monkeypatch.setattr(subprocess, "check_output", submit)
    suite.submit()
    assert len(calls) == 3
    assert [c[-1] for c in calls] == ["20", "40", "60"]
    assert all("scripts/restoration/train_liteweather_stage.sbatch" in c for c in calls)
    assert "--dependency=afterok:101" in calls[1]
    assert "--dependency=afterok:102" in calls[2]
    with pytest.raises(ValueError, match="already submitted"):
        suite.submit()
