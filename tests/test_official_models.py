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
from aegis_perception.data.manifest import PairRecord, write_manifest
from aegis_perception.inference import predict
from aegis_perception.models.official import (
    OfficialGenerator,
    load_architecture,
    load_official_weights,
    source_info,
)
from aegis_perception.official_runtime import activate
from aegis_perception.training import train_restoration

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    activate()
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def need_assets(name):
    if not (ROOT / source_info(name)["weights"]["path"]).exists():
        pytest.skip("Fetch official sources/weights before running integration checks")


@pytest.mark.parametrize(
    "name,parameters",
    [
        ("transweather", 38052078),
        ("promptir", 35592263),
        ("ramit", 934651),
    ],
)
def test_official_weights_full_coverage_backward_and_native_shape(name, parameters):
    need_assets(name)
    suite = runpy.run_path(str(ROOT / "scripts/restoration/official_suite.py"))
    model = OfficialGenerator(name)
    load_official_weights(model, source_info(name)["weights"]["path"])
    assert sum(p.numel() for p in model.parameters()) == parameters
    image = torch.rand(1, 3, 128, 128) * 2 - 1
    loss = (model(image) - image * 0.8).abs().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert {n for n, p in model.named_parameters() if p.grad is None} == suite["UNUSED"][name]
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with torch.inference_mode():
        result = predict(model.eval(), torch.zeros(1, 3, 133, 145))
    assert result.shape == (1, 3, 133, 145) and torch.isfinite(result).all()


def test_transweather_sdpa_matches_upstream_attention_outputs_and_gradients():
    need_assets("transweather")
    original = load_architecture("transweather", efficient_attention=False)
    efficient = load_architecture("transweather", efficient_attention=True)
    for name in ("Attention", "Attention_dec"):
        a = getattr(original, name)(64, num_heads=4, sr_ratio=1)
        b = getattr(efficient, name)(64, num_heads=4, sr_ratio=1)
        b.load_state_dict(a.state_dict(), strict=True)
        x = torch.randn(2, 64, 64, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        left, right = a(x, 8, 8), b(y, 8, 8)
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
        left.square().sum().backward()
        right.square().sum().backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=1e-4, atol=1e-6)


def test_frozen_configs_and_eighteen_sequential_jobs(tmp_path, monkeypatch):
    suite = runpy.run_path(str(ROOT / "scripts/restoration/official_suite.py"))["suite"]
    configs = suite.checked_configs()
    assert len(configs) == 3
    assert all(c["train"]["epochs"] * c["data"]["samples_per_epoch"] // 4 == 61440 for c in configs)
    monkeypatch.setattr(suite, "OUTPUT", tmp_path)
    monkeypatch.setattr(suite, "verify_release", lambda: None)
    monkeypatch.setattr(suite, "ensure_initializations", lambda _: None)
    calls = []

    def output(command, **kwargs):
        if command[0] == "squeue":
            return "1\n"
        calls.append(command)
        return f"{len(calls)};cluster\n"

    monkeypatch.setattr(subprocess, "check_output", output)
    suite.submit()
    assert len(calls) == 18
    assert not any(c.startswith("--dependency") for c in calls[0])
    for i, command in enumerate(calls[1:], 1):
        assert f"--dependency=afterok:{i}" in command
    journal = json.loads((tmp_path / "jobs.json").read_text())
    assert len(journal["attempts"][0]) == 18
    with pytest.raises(ValueError, match="already submitted"):
        suite.submit()
    with pytest.raises(ValueError):
        suite.submit(resume=True)


def tiny_experiment(tmp_path):
    records = []
    rng = np.random.default_rng(9)
    for weather in ("rain", "fog", "snow", "smoke"):
        for split in ("train", "val", "test"):
            name = f"{weather}_{split}"
            src, target = tmp_path / f"{name}.png", tmp_path / f"{name}_gt.png"
            pixels = rng.integers(0, 256, (33, 35, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(src)
            Image.fromarray(pixels // 2 + 50).save(target)
            records.append(PairRecord(name, src, target, split, weather, "fixture", name))
    manifest = tmp_path / "pairs.csv"
    write_manifest(manifest, records)
    c = yaml.safe_load((ROOT / "configs/restoration/official_v1/ramit.yaml").read_text())
    c["run"].update(device="cpu", output_dir=str(tmp_path / "staged"))
    c["train"].update(epochs=2, batch_size=2, precision="fp32", checkpoint_every=1)
    c["data"].update(
        manifest=str(manifest), image_size=[32, 32], samples_per_epoch=4, num_workers=0
    )
    c["data"].pop("protocol")
    c["data"].pop("protocol_sha256")
    return c


def test_actual_pretrained_resume_evaluate_export(tmp_path, monkeypatch):
    need_assets("ramit")
    c = tiny_experiment(tmp_path)
    continuous = copy.deepcopy(c)
    continuous["run"]["output_dir"] = str(tmp_path / "continuous")
    train_restoration(continuous)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(c))
    suite = runpy.run_path(str(ROOT / "scripts/restoration/official_suite.py"))["suite"]
    for key, value in dict(
        CONFIGS=(path,),
        BOUNDARIES=(1, 2),
        checked_configs=lambda: [c],
        verify_release=lambda: None,
        ensure_initializations=lambda _: None,
        preflight=lambda _: None,
        report=lambda: None,
    ).items():
        monkeypatch.setattr(suite, key, value)
    actual = subprocess.run
    calls = []

    def execute(command, **kwargs):
        command = list(command)
        if "evaluate" in command:
            command[command.index("--device") + 1] = "cpu"
        calls.append(command)
        return actual(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", execute)
    suite.run_stage(path, 1)
    suite.run_stage(path, 2)
    a = torch.load(tmp_path / "continuous/last.pt", weights_only=False)
    b = torch.load(tmp_path / "staged/last.pt", weights_only=False)
    for key, value in a["generator"].items():
        torch.testing.assert_close(value, b["generator"][key], rtol=0, atol=0)
    assert a["schedulers"] == b["schedulers"]
    assert b["pretrained_sha256"] == c["train"]["expected_pretrained_sha256"]
    full, _ = load_generator(tmp_path / "staged/best.pt", device="cpu")
    compact, payload = load_generator(tmp_path / "staged/inference.pt", device="cpu")
    assert "optimizer_g" not in payload
    with torch.inference_mode():
        x = torch.zeros(1, 3, 37, 43)
        torch.testing.assert_close(predict(full, x), predict(compact, x), rtol=0, atol=0)
    assert len(json.loads((tmp_path / "staged/metrics_test.json").read_text())["samples"]) == 4
    suite.run_stage(path, 2)
    assert len(calls) == 3


def test_official_source_checksum_is_enforced(tmp_path, monkeypatch):
    from aegis_perception.models import official

    entry = copy.deepcopy(source_info("promptir"))
    destination = tmp_path / "net/model.py"
    destination.parent.mkdir(parents=True)
    destination.write_text("raise AssertionError('must never execute')")
    entry["sources"] = [{"path": str(destination), "sha256": "0" * 64}]
    monkeypatch.setattr(official, "source_info", lambda _: entry)
    with pytest.raises(ValueError, match="checksum"):
        official.load_architecture("promptir")


def test_large_depthwise_compatibility_keeps_outputs_and_gradients(monkeypatch):
    import types

    from aegis_perception.models import restormer
    from aegis_perception.models.official_support import safe_depthwise_forward

    a = torch.nn.Conv2d(8, 8, 3, padding=1, groups=8)
    b = copy.deepcopy(a)
    b.forward = types.MethodType(safe_depthwise_forward, b)
    monkeypatch.setattr(restormer, "_DEPTHWISE_INDEX_LIMIT", 1)
    monkeypatch.setattr(restormer, "_DEPTHWISE_CHUNK_ELEMENTS", 128)
    x = torch.randn(1, 8, 16, 16, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    left, right = a(x), b(y)
    torch.testing.assert_close(left, right)
    left.sum().backward()
    right.sum().backward()
    torch.testing.assert_close(x.grad, y.grad)
    torch.testing.assert_close(a.weight.grad, b.weight.grad)
