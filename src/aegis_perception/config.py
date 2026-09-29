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
    model_name = config["model"].get("name", "attention_unet")
    if model_name == "attention_unet":
        require_keys(config["model"], ("attention", "features"), "model")
        multiple = 64
    elif model_name == "restormer":
        multiple = 8
        if float(config["train"].get("gan_weight", 0.0)) != 0:
            raise ConfigError("Restormer uses reconstruction training; set gan_weight to 0")
        dim = config["model"].get("dim", 48)
        heads = config["model"].get("heads", [1, 2, 4, 8])
        blocks = config["model"].get("num_blocks", [4, 6, 6, 8])
        if dim <= 0 or dim % 2 or len(heads) != 4 or len(blocks) != 4:
            raise ConfigError("Restormer needs a positive even dim and four heads/block counts")
        if any(h <= 0 or (dim * 2**i) % h for i, h in enumerate(heads)):
            raise ConfigError("Restormer heads must divide the stage channels")
        if any(b <= 0 for b in blocks):
            raise ConfigError("Restormer block counts must be positive")
        if config["model"].get("layer_norm_type", "WithBias") not in {"WithBias", "BiasFree"}:
            raise ConfigError("Restormer layer_norm_type must be WithBias or BiasFree")
    else:
        raise ConfigError(f"Unknown restoration model: {model_name!r}")
    require_keys(
        config["train"],
        ("epochs", "batch_size", "learning_rate", "betas", "l1_weight"),
        "train",
    )

    height, width = config["data"]["image_size"]
    if height < multiple or width < multiple or height % multiple or width % multiple:
        raise ConfigError(f"data.image_size must be divisible by {multiple}")
    if config["train"]["epochs"] <= 0 or config["train"]["batch_size"] <= 0:
        raise ConfigError("epochs and batch_size must be positive")
    if float(config["train"].get("gan_weight", 0.0)) < 0:
        raise ConfigError("train.gan_weight cannot be negative")
    if config["data"].get("train_mode", "resize") not in {"resize", "crop"}:
        raise ConfigError("data.train_mode must be resize or crop")
    if config["data"].get("eval_mode", "resize") not in {"resize", "crop", "native"}:
        raise ConfigError("data.eval_mode must be resize, crop, or native")
    if config["train"].get("precision", "fp32") not in {"fp32", "fp16", "bf16"}:
        raise ConfigError("train.precision must be fp32, fp16, or bf16")
    if config["train"].get("optimizer", "adam") not in {"adam", "adamw"}:
        raise ConfigError("train.optimizer must be adam or adamw")
    if config["train"].get("scheduler", "none") not in {"none", "cosine"}:
        raise ConfigError("train.scheduler must be none or cosine")
    if float(config["train"]["learning_rate"]) <= 0:
        raise ConfigError("train.learning_rate must be positive")
    if float(config["train"]["l1_weight"]) <= 0:
        raise ConfigError("train.l1_weight must be positive")
    if int(config["train"].get("checkpoint_every", 10)) <= 0:
        raise ConfigError("train.checkpoint_every must be positive")
    betas = config["train"]["betas"]
    if len(betas) != 2 or any(not 0 <= float(beta) < 1 for beta in betas):
        raise ConfigError("train.betas must contain two values in [0, 1)")
    if float(config["train"].get("grad_clip", 0)) < 0:
        raise ConfigError("train.grad_clip cannot be negative")
    for key in ("max_train_samples", "max_val_samples"):
        if key in config["data"] and int(config["data"][key]) <= 0:
            raise ConfigError(f"data.{key} must be positive")
