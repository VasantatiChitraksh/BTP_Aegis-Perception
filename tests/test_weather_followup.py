from __future__ import annotations

import copy
import json
import os
import runpy
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from aegis_perception.data.manifest import PairRecord, load_manifest, write_manifest
from aegis_perception.data.sampling import RestorationBatchSampler
from aegis_perception.evaluation import balanced_panel, summarize_samples
from aegis_perception.losses import reconstruction_objective, structural_similarity_per_image
from aegis_perception.metrics import ssim
from aegis_perception.training import passes_validation_guard

ROOT = Path(__file__).resolve().parents[1]


def records():
    return [
        PairRecord(f"{w}/{g}/{i}", Path("input"), Path("target"), "test", w, "fixture", str(g))
        for w in ("rain", "fog", "snow", "smoke")
        for g, count in enumerate((3, 17))
        for i in range(count)
    ]


def test_fixed_panel_is_order_independent_and_spreads_backgrounds():
    rows = records()
    panel = balanced_panel(rows, 6)
    assert [r.sample_id for r in panel] == [
        r.sample_id for r in balanced_panel(list(reversed(rows)), 6)
    ]
    assert Counter((r.weather, r.group_id) for r in panel) == {
        (w, str(g)): 3 for w in ("rain", "fog", "snow", "smoke") for g in range(2)
    }
    with pytest.raises(ValueError, match="fewer"):
        balanced_panel(rows, 21)


def test_global_batches_are_balanced_and_sharded_without_extra_draws():
    rows = records()
    options = dict(
        probabilities=dict.fromkeys(("rain", "fog", "snow", "smoke"), 0.25), balanced=True, seed=42
    )
    whole = RestorationBatchSampler(rows, 4, 32, **options)
    first = RestorationBatchSampler(rows, 4, 32, rank=0, world_size=2, **options)
    second = RestorationBatchSampler(rows, 4, 32, rank=1, world_size=2, **options)
    for batch, a, b in zip(whole, first, second, strict=True):
        assert batch == a + b
        assert Counter(rows[i].weather for i in batch) == dict.fromkeys(options["probabilities"], 1)
    total = Counter((rows[i].weather, rows[i].group_id) for batch in whole for i in batch)
    assert set(total.values()) == {4}
    old = list(whole)
    whole.set_epoch(2)
    assert list(whole) != old
    with pytest.raises(ValueError, match="divide"):
        RestorationBatchSampler(rows, 4, 32, world_size=3, **options)


def test_macro_reports_do_not_let_many_snow_frames_hide_smoke_regressions():
    rows = [
        dict(
            sample_id="rain",
            weather="rain",
            group_id="r",
            psnr_db=30,
            ssim=0.9,
            raw_psnr_db=20,
            raw_ssim=0.7,
        ),
        dict(
            sample_id="smoke",
            weather="smoke",
            group_id="s",
            psnr_db=10,
            ssim=0.5,
            raw_psnr_db=15,
            raw_ssim=0.6,
        ),
    ]
    expected = summarize_samples(rows)
    many = summarize_samples(rows + [{**rows[0], "sample_id": str(i)} for i in range(100)])
    for key, value in expected["headline"].items():
        if key != "aggregation":
            assert many["headline"][key] == pytest.approx(value)
    assert many["summary"]["all"]["psnr_db"] > expected["summary"]["all"]["psnr_db"]
    assert many["counts"]["smoke"]["both_worse_than_raw"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_samples(rows + [rows[0]])


def test_differentiable_ssim_matches_existing_metric_and_has_useful_gradients():
    torch.manual_seed(42)
    x = torch.rand(2, 3, 19, 23)
    y = (x * 0.85 + torch.rand_like(x) * 0.15).requires_grad_()
    scores = structural_similarity_per_image(x, y)
    expected = [
        ssim(a.permute(1, 2, 0).numpy(), b.detach().permute(1, 2, 0).numpy())
        for a, b in zip(x, y, strict=True)
    ]
    np.testing.assert_allclose(scores.detach().numpy(), expected, atol=3e-6, rtol=1e-5)
    identity = structural_similarity_per_image(x, x)
    torch.testing.assert_close(identity, torch.ones_like(identity), atol=1e-6, rtol=1e-6)
    loss = reconstruction_objective(
        y.mul(2).sub(1), x.mul(2).sub(1), ssim_weight=0.05, mse_weight=0.1
    )
    loss.backward()
    assert torch.isfinite(y.grad).all() and y.grad.abs().sum() > 0
    moved = y.detach() - 0.1 * y.grad
    assert structural_similarity_per_image(x, moved).mean() > scores.mean()


def test_guard_rejects_hidden_weather_or_ssim_regression():
    reference = {
        "rain_scene_psnr_db": 30.0,
        "smoke_scene_psnr_db": 20.0,
        "rain_scene_ssim": 0.95,
        "smoke_scene_ssim": 0.8,
        "macro_scene_psnr_db": 25.0,
    }
    current = {**reference, "rain_scene_psnr_db": 35.0, "smoke_scene_ssim": 0.79}
    tolerances = {"psnr_db": 0.0, "ssim": 0.0}
    assert not passes_validation_guard(current, reference, tolerances)
    current["smoke_scene_ssim"] = 0.81
    assert passes_validation_guard(current, reference, tolerances)


def test_guarded_training_does_not_publish_a_regressing_checkpoint(tmp_path, monkeypatch):
    from aegis_perception import training

    tiny_config = runpy.run_path(str(ROOT / "tests/test_restoration_pipeline.py"))["tiny_config"]
    config = tiny_config(tmp_path)
    config["train"].update(
        validation_ssim=True,
        validation_guard={"psnr_db": 0.0, "ssim": 0.0},
        selection_metric="macro_scene_psnr_db",
    )
    reference = dict(
        l1=0.1,
        psnr_db=30.0,
        raw_psnr_db=20.0,
        psnr_gain_db=10.0,
        rain_scene_psnr_db=30.0,
        rain_scene_ssim=0.95,
        macro_scene_psnr_db=30.0,
    )
    bad_ssim = {
        **reference,
        "rain_scene_psnr_db": 32.0,
        "macro_scene_psnr_db": 32.0,
        "rain_scene_ssim": 0.94,
    }
    validations = iter([reference, bad_ssim, bad_ssim])
    monkeypatch.setattr(training, "_validate", lambda *args, **kwargs: next(validations))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        training.train_restoration(config)
    finally:
        torch.set_num_threads(previous_threads)
    output = Path(config["run"]["output_dir"])
    assert not (output / "best.pt").exists()
    checkpoint = torch.load(output / "last.pt", weights_only=False)
    assert checkpoint["reference_validation"] == reference
    assert checkpoint["best_selection_score"] == 30.0
    assert not json.loads((output / "selection_status.json").read_text())["candidate_available"]


def test_preparation_preserves_train_and_test_and_expands_only_smoke_validation(tmp_path):
    prepare = runpy.run_path(str(ROOT / "scripts/data/prepare_weather_followup.py"))["prepare"]
    base = [replace(r, group_id=f"test:{r.weather}:{r.group_id}") for r in records()]
    smoke = [
        PairRecord(
            f"s/{i}",
            tmp_path / f"{i}.png",
            tmp_path / "target.png",
            "val",
            "smoke",
            "fixture",
            "smoke_val",
        )
        for i in range(40)
    ]
    base += [replace(r, sample_id=f"{r.source}/{r.sample_id}") for r in smoke[:8]]
    # Match the original panel's hash selection.
    import hashlib

    smoke.sort(key=lambda r: hashlib.sha256(f"variant:42:{r.sample_id}".encode()).digest())
    base = [r for r in base if r.split != "val"]
    base += [replace(r, sample_id=f"{r.source}/{r.sample_id}") for r in smoke[:8]]
    source, smoke_path = tmp_path / "base.csv", tmp_path / "smoke.csv"
    write_manifest(source, base)
    write_manifest(smoke_path, smoke)
    output, panel = tmp_path / "v2.csv", tmp_path / "panel.json"
    prepare(source, smoke_path, output, panel, per_weather=6)
    assert set(load_manifest(source, "test")) == set(load_manifest(output, "test"))
    assert len(load_manifest(output, "val")) == 32
    assert len(json.loads(panel.read_text())["sample_ids"]) == 24
    with pytest.raises(ValueError, match="already exist"):
        prepare(source, smoke_path, output, panel, per_weather=6)


@pytest.mark.parametrize("sampling", ["balanced_batches", "stratified_cycles"])
def test_two_process_training_resume_and_evaluation(tmp_path, sampling):
    """Exercise real DDP collectives, uneven validation shards and checkpoint IO."""
    rng = np.random.default_rng(42)
    rows = []
    for i, (split, weather) in enumerate(
        [
            ("train", "rain"),
            ("train", "smoke"),
            ("val", "rain"),
            ("val", "smoke"),
            ("val", "smoke"),
            ("test", "rain"),
        ]
    ):
        path = tmp_path / f"{i}.png"
        Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(path)
        rows.append(PairRecord(str(i), path, path, split, weather, "fixture", str(i), "q0", str(i)))
    manifest = tmp_path / "pairs.csv"
    write_manifest(manifest, rows)
    config = {
        "run": {
            "seed": 42,
            "name": "ddp-test",
            "device": "cpu",
            "output_dir": str(tmp_path / "run"),
        },
        "data": {
            "manifest": str(manifest),
            "image_size": [16, 16],
            "num_workers": 0,
            "train_mode": "crop",
            "eval_mode": "native",
            "sampling": sampling,
            "weather_probabilities": {"rain": 0.5, "smoke": 0.5},
            "samples_per_epoch": 4,
        },
        "model": {
            "name": "restormer",
            "dim": 8,
            "num_blocks": [1, 1, 1, 1],
            "num_refinement_blocks": 1,
        },
        "train": {
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 1e-4,
            "betas": [0.9, 0.999],
            "l1_weight": 1.0,
            "ssim_weight": 0.05,
            "validation_ssim": True,
            "precision": "fp32",
            "scheduler": "cosine",
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    launcher = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2"]

    def run(args, distributed=True):
        completed = subprocess.run(
            (launcher if distributed else [sys.executable]) + args,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr

    train = ["scripts/restoration/train.py", "--config", str(config_path)]
    run(train + ["--stop-after-epochs", "1"])
    last = tmp_path / "run/last.pt"
    run(train + ["--resume", str(last)])
    checkpoint = torch.load(last, weights_only=False)
    assert checkpoint["world_size"] == 2 and len(checkpoint["rng_states_by_rank"]) == 2
    assert "smoke_scene_ssim" in checkpoint["validation"]
    continuous = copy.deepcopy(config)
    continuous["run"]["output_dir"] = str(tmp_path / "continuous")
    other = tmp_path / "continuous.yaml"
    other.write_text(yaml.safe_dump(continuous))
    run(["scripts/restoration/train.py", "--config", str(other)])
    uninterrupted = torch.load(tmp_path / "continuous/last.pt", weights_only=False)
    for key, tensor in checkpoint["generator"].items():
        torch.testing.assert_close(tensor, uninterrupted["generator"][key], atol=0, rtol=0)
    # Test has one image, so rank 1 has an empty shard; it must still join the collective.
    evaluate = [
        "scripts/restoration/evaluate.py",
        "--config",
        str(config_path),
        "--checkpoint",
        str(last),
        "--device",
        "cpu",
    ]
    run(evaluate + ["--output", str(tmp_path / "ddp.json")])
    run(evaluate + ["--output", str(tmp_path / "single.json")], distributed=False)
    ddp, single = [
        json.loads((tmp_path / name).read_text()) for name in ("ddp.json", "single.json")
    ]
    assert ddp["samples"] == single["samples"]
    assert len(ddp["samples"]) == 1
