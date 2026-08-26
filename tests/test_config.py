from __future__ import annotations

import pytest

from aegis_perception.config import ConfigError, validate_restoration_config


def valid_config() -> dict:
    return {
        "run": {"name": "test", "seed": 42, "output_dir": "artifacts/test"},
        "data": {"manifest": "data/manifests/test.csv", "image_size": [256, 256]},
        "model": {"attention": True, "features": 64},
        "train": {
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 1e-4,
            "betas": [0.5, 0.999],
            "l1_weight": 100,
        },
    }


def test_valid_restoration_config() -> None:
    validate_restoration_config(valid_config())


def test_rejects_non_divisible_image_size() -> None:
    config = valid_config()
    config["data"]["image_size"] = [250, 250]
    with pytest.raises(ConfigError, match="divisible by 64"):
        validate_restoration_config(config)
