from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image

from aegis_perception.checkpoints import load_generator
from aegis_perception.data.manifest import PairRecord, write_manifest


@pytest.mark.parametrize("gan_weight", [0.0, 1.0])
def test_training_checkpoint_can_restore_images(tmp_path, monkeypatch, gan_weight):
    torch = pytest.importorskip("torch")
    from aegis_perception import training

    # The L1 ablation must train without allocating a discriminator at all.
    if gan_weight == 0:

        def unused_discriminator(**kwargs):
            pytest.fail("L1-only training constructed a discriminator")

        monkeypatch.setattr(training, "PatchDiscriminator", unused_discriminator)

    rng = np.random.default_rng(42)
    records = []
    for index, split in enumerate(("train", "train", "val")):
        image = tmp_path / f"{index}.png"
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(image)
        records.append(PairRecord(str(index), image, image, split, "smoke", "test"))
    manifest = tmp_path / "pairs.csv"
    write_manifest(manifest, records)
    config = {
        "run": {"name": "test", "seed": 42, "device": "cpu", "output_dir": str(tmp_path / "run")},
        "data": {"manifest": str(manifest), "image_size": [64, 64], "num_workers": 0},
        "model": {"attention": True, "features": 2},
        "train": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-4,
            "betas": [0.5, 0.999],
            "l1_weight": 100,
            "gan_weight": gan_weight,
        },
    }
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        history = training.train_restoration(config)
        model, checkpoint = load_generator(tmp_path / "run/best.pt", device="cpu")
        with torch.inference_mode():
            restored = model(torch.zeros(1, 3, 64, 64))
    finally:
        torch.set_num_threads(previous_threads)
    assert len(history) == 1 and math.isfinite(history[0]["val_psnr_db"])
    assert restored.shape == (1, 3, 64, 64) and torch.isfinite(restored).all()
    assert ("discriminator" in checkpoint) == (gan_weight > 0)
    assert ("optimizer_d" in checkpoint) == (gan_weight > 0)
