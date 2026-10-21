"""Explicit v2 integration without edits to the original model/config/trainer.

Call activate() before using shared training or checkpoint-loading APIs for v2.
Registration affects only this Python process; existing factories handle all other models.
"""

import sys

from . import config as config_module
from . import models
from .config import ConfigError

_original_factory = models.build_generator
_original_validator = config_module.validate_restoration_config


def build_generator(config):
    if config.get("name") != "liteweatherformer":
        return _original_factory(config)
    options = {k: v for k, v in config.items() if k not in {"name", "pretrained"}}
    version = options.pop("architecture_version", 1)
    if version == 1:
        return _original_factory({"name": "liteweatherformer", **options})
    if version != 2:
        raise ValueError(f"Unknown LiteWeatherFormer architecture version: {version}")
    from .models.liteweatherformer_v2 import LiteWeatherTransformerV2

    return LiteWeatherTransformerV2(**options)


def validate_restoration_config(config):
    _original_validator(config)
    model = config["model"]
    if model.get("name") != "liteweatherformer":
        return
    version = model.get("architecture_version", 1)
    if version not in {1, 2}:
        raise ConfigError("Unknown LiteWeatherFormer architecture version")
    if version == 1:
        return
    dim = model.get("dim", 24)
    heads = model.get("heads", [4, 8])
    blocks = model.get("num_blocks", [2, 4])
    if dim <= 0 or len(heads) != 2 or len(blocks) != 2:
        raise ConfigError("LiteWeatherFormer v2 needs positive dim and two stages")
    if any(h <= 0 or (dim * (4 << i)) % h for i, h in enumerate(heads)):
        raise ConfigError("LiteWeatherFormer heads must divide stage channels")
    if any(b <= 0 for b in blocks) or model.get("num_weather_types", 4) <= 0:
        raise ConfigError("LiteWeatherFormer needs positive block and prompt counts")
    if not float(model.get("ffn_expansion_factor", 2.0)) > 0:
        raise ConfigError("LiteWeatherFormer expansion must be positive")


def activate():
    """Register v2 for this process, including an already-imported trainer."""
    models.build_generator = build_generator
    config_module.validate_restoration_config = validate_restoration_config
    training = sys.modules.get("aegis_perception.training")
    if training is not None:
        training.build_generator = build_generator
        training.validate_restoration_config = validate_restoration_config
