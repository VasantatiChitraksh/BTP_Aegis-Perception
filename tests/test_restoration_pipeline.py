from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from PIL import Image

from aegis_perception.checkpoints import load_pretrained
from aegis_perception.config import ConfigError, validate_restoration_config
from aegis_perception.data.manifest import PairRecord, write_manifest
from aegis_perception.data.paired import PairedImageDataset


def tiny_config(tmp_path):
    rng = np.random.default_rng(42)
    records = []
    for i, split in enumerate(("train", "train", "val")):
        path = tmp_path / f"{i}.png"
        Image.fromarray(rng.integers(0, 256, (21, 29, 3), dtype=np.uint8)).save(path)
        records.append(PairRecord(str(i), path, path, split, "rain", "fixture"))
    manifest = tmp_path / "pairs.csv"
    write_manifest(manifest, records)
    return {
        "run": {
            "name": "fixture",
            "seed": 42,
            "device": "cpu",
            "output_dir": str(tmp_path / "run"),
        },
        "data": {
            "manifest": str(manifest),
            "image_size": [16, 16],
            "train_mode": "crop",
            "eval_mode": "native",
            "random_flip": True,
            "num_workers": 0,
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
            "optimizer": "adamw",
            "scheduler": "cosine",
            "l1_weight": 1,
            "gan_weight": 0,
            "precision": "fp32",
            "grad_clip": 1,
        },
    }


def test_crop_preserves_pair_alignment_and_native_geometry(tmp_path):
    torch = pytest.importorskip("torch")
    config = tiny_config(tmp_path)
    cropped = PairedImageDataset(
        config["data"]["manifest"],
        "train",
        image_size=(16, 16),
        spatial_mode="crop",
        random_flip=True,
    )[0]
    assert cropped["input"].shape == (3, 16, 16)
    torch.testing.assert_close(cropped["input"], cropped["target"])
    native = PairedImageDataset(config["data"]["manifest"], "val", spatial_mode="native")[0]
    assert native["input"].shape == (3, 21, 29)
    padded = PairedImageDataset(
        config["data"]["manifest"], "train", image_size=(32, 32), spatial_mode="crop"
    )[0]
    assert padded["input"].shape == (3, 32, 32)
    torch.testing.assert_close(padded["input"], padded["target"])


def test_upstream_weights_and_normalized_native_inference(tmp_path):
    torch = pytest.importorskip("torch")
    from aegis_perception.inference import predict
    from aegis_perception.models import build_generator

    config = tiny_config(tmp_path)
    generator = build_generator(config["model"]).eval()
    # A zero residual network must preserve the source, including padded borders.
    for parameter in generator.parameters():
        parameter.data.zero_()
    path = tmp_path / "upstream.pth"
    torch.save(
        {"params": {f"module.{k}": v for k, v in generator.network.state_dict().items()}}, path
    )
    restored = build_generator(config["model"]).eval()
    load_pretrained(restored, path)
    sample = torch.rand(1, 3, 21, 29).mul(2).sub(1)
    with torch.inference_mode():
        output = predict(restored, sample)
    torch.testing.assert_close(output, sample, atol=1e-6, rtol=1e-6)
    torch.save({"params": {"wrong_key": torch.zeros(1)}}, path)
    with pytest.raises(RuntimeError):
        load_pretrained(restored, path)


def test_resume_matches_uninterrupted_fixture_updates(tmp_path):
    torch = pytest.importorskip("torch")
    from aegis_perception.training import train_restoration

    config = tiny_config(tmp_path)
    continuous = copy.deepcopy(config)
    continuous["run"]["output_dir"] = str(tmp_path / "continuous")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        train_restoration(continuous)
        train_restoration(config, stop_after_epochs=1)
        last = tmp_path / "run" / "last.pt"
        train_restoration(config, resume=last)
        first = torch.load(tmp_path / "continuous" / "last.pt", weights_only=False)
        resumed = torch.load(last, weights_only=False)
        for key, value in first["generator"].items():
            torch.testing.assert_close(value, resumed["generator"][key], atol=0, rtol=0)
        assert first["schedulers"] == resumed["schedulers"]
        assert [r["epoch"] for r in resumed["history"]] == [1, 2]
        assert (tmp_path / "run" / "best.pt").is_file()
        assert json.loads((tmp_path / "run" / "run.json").read_text())["manifest_sha256"]
        with pytest.raises(ValueError, match="already exists"):
            train_restoration(config)
        changed = copy.deepcopy(config)
        changed["data"]["eval_mode"] = "resize"
        with pytest.raises(ValueError, match="Resume data config differs"):
            train_restoration(changed, resume=last)
    finally:
        torch.set_num_threads(previous_threads)


def test_rejects_unsupported_model_heads(tmp_path):
    config = tiny_config(tmp_path)
    config["model"]["heads"] = [3, 2, 4, 8]
    with pytest.raises(ConfigError, match="heads must divide"):
        validate_restoration_config(config)
