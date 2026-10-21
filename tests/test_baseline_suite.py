from __future__ import annotations

import copy
import json
import random
import runpy
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from aegis_perception.data.manifest import PairRecord, write_manifest
from aegis_perception.data.paired import PairedImageDataset
from aegis_perception.data.sampling import RestorationBatchSampler
from aegis_perception.training import train_restoration

ROOT = Path(__file__).resolve().parents[1]


def fixture_config(tmp_path, attention):
    records = []
    rng = np.random.default_rng(42)
    for i, split in enumerate(("train", "train", "val")):
        source, target = tmp_path / f"in{i}.png", tmp_path / f"gt{i}.png"
        pixels = rng.integers(0, 256, (73, 81, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(source)
        Image.fromarray(pixels // 2 + 50).save(target)
        records.append(PairRecord(str(i), source, target, split, "smoke", "fixture", str(i)))
    manifest = tmp_path / "pairs.csv"
    write_manifest(manifest, records)
    return dict(
        run=dict(name="fixture", seed=42, device="cpu", output_dir=str(tmp_path / "staged")),
        data=dict(
            manifest=str(manifest),
            image_size=[64, 64],
            train_mode="crop",
            eval_mode="native",
            num_workers=0,
            random_flip=True,
            sampling="weighted_seeded",
            weather_probabilities={"smoke": 1.0},
            samples_per_epoch=4,
        ),
        model=dict(name="attention_unet", features=2, attention=attention),
        train=dict(
            epochs=2,
            batch_size=2,
            learning_rate=1e-4,
            betas=[0.5, 0.999],
            l1_weight=100,
            gan_weight=1,
            scheduler="cosine",
            validation_ssim=True,
            selection_metric="macro_scene_psnr_db",
            save_periodic_checkpoints=False,
        ),
    )


@pytest.mark.parametrize("attention", [False, True])
def test_both_gans_resume_exactly_with_weighted_draws_dropout_and_native_validation(
    tmp_path, attention
):
    config = fixture_config(tmp_path, attention)
    continuous = copy.deepcopy(config)
    continuous["run"]["output_dir"] = str(tmp_path / "continuous")
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        train_restoration(continuous)
        train_restoration(config, stop_after_epochs=1)
        train_restoration(config, resume=tmp_path / "staged/last.pt", stop_after_epochs=1)
        a = torch.load(tmp_path / "continuous/last.pt", weights_only=False)
        b = torch.load(tmp_path / "staged/last.pt", weights_only=False)
        for component in ("generator", "discriminator"):
            for key, value in a[component].items():
                torch.testing.assert_close(value, b[component][key], rtol=0, atol=0)
        assert a["schedulers"] == b["schedulers"]
        assert a["optimizer_d"]["param_groups"] == b["optimizer_d"]["param_groups"]
        assert [x["val_macro_scene_psnr_db"] for x in a["history"]] == [
            x["val_macro_scene_psnr_db"] for x in b["history"]
        ]
        assert {p.name for p in (tmp_path / "staged").glob("*.pt")} == {"best.pt", "last.pt"}
    finally:
        torch.set_num_threads(before)


def test_training_draws_and_crops_do_not_depend_on_model_rng(tmp_path):
    config = fixture_config(tmp_path, False)
    dataset = PairedImageDataset(
        config["data"]["manifest"],
        "train",
        image_size=(64, 64),
        random_flip=True,
        spatial_mode="crop",
    )
    sampler = RestorationBatchSampler(
        dataset.records, 2, 16, probabilities={"smoke": 1.0}, seed=42, seeded_augmentations=True
    )
    sampler.set_epoch(1)
    first = list(sampler)
    a = [dataset[index]["input"] for batch in first for index in batch]
    random.seed(999)
    for _ in range(1000):
        random.random()
    torch.rand(1000)  # Different architectures/dropout consume different amounts of RNG.
    assert list(sampler) == first
    b = [dataset[index]["input"] for batch in sampler for index in batch]
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    sampler.set_epoch(2)
    assert list(sampler) != first


def suite_globals(monkeypatch, tmp_path):
    module = runpy.run_path(str(ROOT / "scripts/restoration/baseline_suite.py"))
    namespace = module["submit"].__globals__
    monkeypatch.setitem(namespace, "OUTPUT", tmp_path / "suite")
    monkeypatch.setitem(namespace, "verify_release", lambda: None)
    monkeypatch.setitem(
        namespace,
        "checked_configs",
        lambda: [
            {"run": {"output_dir": str(tmp_path / n)}, "model": {}} for n in namespace["NAMES"]
        ],
    )
    monkeypatch.setenv("USER", "fixture")
    return namespace


def test_submission_is_one_sequential_chain_and_rejects_active_duplicates(monkeypatch, tmp_path):
    ns = suite_globals(monkeypatch, tmp_path)
    calls = []

    def fake(command, **kwargs):
        if command[0] == "squeue":
            return "1001\n"
        calls.append(command)
        return f"{1000 + len(calls)};cluster\n"

    monkeypatch.setattr(subprocess, "check_output", fake)
    ns["submit"]()
    assert len(calls) == 6
    assert not any(s.startswith("--dependency") for s in calls[0])
    for i, command in enumerate(calls[1:], 1):
        assert f"--dependency=afterok:{1000 + i}" in command
        assert "--kill-on-invalid-dep=yes" in command
    assert [c[-1] for c in calls] == ["20", "40", "60"] * 2
    assert Path(calls[0][-2]).name == "attention_pix2pix.yaml"
    assert Path(calls[3][-2]).name == "restormer.yaml"
    assert all(Path(c[-2]).name != "pix2pix.yaml" for c in calls)
    with pytest.raises(ValueError, match="already submitted"):
        ns["submit"]()
    with pytest.raises(ValueError, match="still active"):
        ns["submit"](resume=True)


def test_partial_submission_records_jobs_before_failure(monkeypatch, tmp_path):
    ns = suite_globals(monkeypatch, tmp_path)
    count = 0

    def fake(command, **kwargs):
        nonlocal count
        count += 1
        if count == 3:
            raise subprocess.CalledProcessError(1, command)
        return f"{count}\n"

    monkeypatch.setattr(subprocess, "check_output", fake)
    with pytest.raises(subprocess.CalledProcessError):
        ns["submit"]()
    state = json.loads((tmp_path / "suite/jobs.json").read_text())
    assert [j["job_id"] for j in state["attempts"][0]] == ["1", "2"]


def test_suite_configs_are_frozen_and_match_data_budget(monkeypatch):
    monkeypatch.chdir(ROOT)
    ns = runpy.run_path(str(ROOT / "scripts/restoration/baseline_suite.py"))
    configs = ns["checked_configs"]()
    assert len(configs) == 2
    for c in configs:
        assert c["data"]["sampling"] == "weighted_seeded"
        assert c["data"]["eval_mode"] == "native"
        assert c["train"]["epochs"] * c["data"]["samples_per_epoch"] // 4 == 61440
        assert c["train"]["validation_ssim"]
    assert configs[0]["model"]["name"] == "attention_unet"
    assert configs[0]["model"]["attention"] is True
    assert "pretrained" not in configs[0]["model"]
    assert configs[1]["model"]["name"] == "restormer"
    original = yaml.safe_load(
        (ROOT / "configs/restoration/four_weather_restormer.yaml").read_text()
    )
    assert configs[1]["model"] == original["model"]
    for key in (
        "epochs",
        "batch_size",
        "optimizer",
        "learning_rate",
        "betas",
        "weight_decay",
        "scheduler",
        "min_learning_rate",
        "precision",
        "grad_clip",
        "gan_weight",
        "l1_weight",
    ):
        assert configs[1]["train"][key] == original["train"][key]
    assert configs[1]["data"]["weather_probabilities"] == original["data"]["weather_probabilities"]


def test_initialization_requires_verified_weights_and_reuses_existing_file(tmp_path, monkeypatch):
    import hashlib

    ns = runpy.run_path(str(ROOT / "scripts/restoration/baseline_suite.py"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AEGIS_PRETRAINED_DIR", str(tmp_path / "existing"))
    source = tmp_path / "existing/deraining.pth"
    destination = Path("release/deraining.pth")
    payload = b"fixture weights"
    config = {
        "model": {"pretrained": str(destination)},
        "train": {"expected_pretrained_sha256": hashlib.sha256(payload).hexdigest()},
    }
    with pytest.raises(ValueError, match="weights required"):
        ns["ensure_initializations"]([config])
    source.parent.mkdir()
    source.write_bytes(b"wrong checkpoint")
    with pytest.raises(ValueError, match="weights required"):
        ns["ensure_initializations"]([config])
    assert not destination.exists()
    source.write_bytes(payload)
    ns["ensure_initializations"]([config])
    assert destination.is_symlink()
    assert destination.resolve() == source
    source.write_bytes(b"changed after submission")
    with pytest.raises(ValueError, match="checksum mismatch"):
        ns["ensure_initializations"]([config])


def test_previous_comparison_flags_each_weather_and_metric_regression(tmp_path, monkeypatch):
    ns = runpy.run_path(str(ROOT / "scripts/restoration/baseline_suite.py"))
    baseline = dict.fromkeys(("macro_weather", "rain", "fog", "snow", "smoke"))
    baseline = {key: {"psnr_db": 30.0, "ssim": 0.9} for key in baseline}
    candidate = copy.deepcopy(baseline)
    candidate["macro_weather"]["psnr_db"] += 1
    candidate["smoke"]["ssim"] -= 0.01
    result = {
        "baseline": {"scene_balanced_summary": baseline},
        "candidate": {"scene_balanced_summary": candidate},
    }
    monkeypatch.setattr(runpy, "run_path", lambda _: {"compare": lambda *args: result})
    output = tmp_path / "comparison.json"
    ns["compare_previous"](tmp_path / "metrics.json", output)
    assessment = json.loads(output.read_text())["assessment"]
    assert not assessment["matches_or_exceeds_previous_on_all_reported_metrics"]
    assert len(assessment["regressions"]) == 1
    assert assessment["regressions"][0]["weather"] == "smoke"
    assert assessment["regressions"][0]["metric"] == "ssim"


def test_stages_train_resume_test_report_and_skip_finished_work(tmp_path, monkeypatch):
    """Actual CPU CLI training/evaluation; only GPU preflight and device are substituted."""
    monkeypatch.chdir(ROOT)
    config = fixture_config(tmp_path, False)
    from aegis_perception.data.manifest import load_manifest

    original = load_manifest(config["data"]["manifest"])
    records = []
    for weather in ("rain", "fog", "snow", "smoke"):
        for i, split in enumerate(("train", "train", "val", "test")):
            source = original[min(i, 2)]
            records.append(
                PairRecord(
                    f"{weather}/{i}",
                    source.input_path,
                    source.target_path,
                    split,
                    weather,
                    "fixture",
                    f"{weather}/{i}",
                )
            )
    # Fixture images intentionally reused; real release's preflight performs the leakage audit.
    write_manifest(config["data"]["manifest"], records)
    config["data"]["weather_probabilities"] = dict.fromkeys(("rain", "fog", "snow", "smoke"), 0.25)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    ns = runpy.run_path(str(ROOT / "scripts/restoration/baseline_suite.py"))[
        "run_stage"
    ].__globals__
    for key, value in {
        "OUTPUT": tmp_path / "summary",
        "CONFIGS": (path,),
        "NAMES": ("pix2pix",),
        "BOUNDARIES": (1, 2),
        "checked_configs": lambda: [config],
        "verify_release": lambda: None,
        "preflight": lambda _: None,
    }.items():
        monkeypatch.setitem(ns, key, value)
    actual_run = subprocess.run
    commands = []

    def execute(command, **kwargs):
        command = list(command)
        if "scripts/restoration/evaluate.py" in command:
            command[command.index("--device") + 1] = "cpu"
        commands.append(command)
        return actual_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", execute)
    ns["run_stage"](path, 1)
    ns["run_stage"](path, 2)
    assert len(commands) == 3  # Train first stage, resume second, evaluate once.
    assert "--resume" in commands[1]
    assert (tmp_path / "summary/comparison.csv").is_file()
    assert (tmp_path / "staged/metrics_test.json").is_file()
    ns["run_stage"](path, 2)
    assert len(commands) == 3
