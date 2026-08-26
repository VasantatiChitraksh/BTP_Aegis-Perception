from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is incomplete or inconsistent."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError(f"Expected a YAML mapping in {config_path}")
    return config


def require_keys(mapping: dict[str, Any], keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ConfigError(f"Missing {context} key(s): {', '.join(missing)}")


def validate_restoration_config(config: dict[str, Any]) -> None:
    require_keys(config, ("run", "data", "model", "train"), "top-level")
    require_keys(config["run"], ("name", "seed", "output_dir"), "run")
    require_keys(config["data"], ("manifest", "image_size"), "data")
    require_keys(config["model"], ("attention", "features"), "model")
    require_keys(
        config["train"],
        ("epochs", "batch_size", "learning_rate", "betas", "l1_weight"),
        "train",
    )

    height, width = config["data"]["image_size"]
    if height != width or height < 64 or height % 64:
        raise ConfigError("data.image_size must be square and divisible by 64")
    if config["train"]["epochs"] <= 0 or config["train"]["batch_size"] <= 0:
        raise ConfigError("epochs and batch_size must be positive")
    if float(config["train"].get("gan_weight", 1.0)) < 0:
        raise ConfigError("train.gan_weight cannot be negative")
