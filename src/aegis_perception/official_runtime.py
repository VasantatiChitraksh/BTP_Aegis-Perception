"""Explicit registration of official models without editing protected shared files."""

import sys

from . import checkpoints, config, liteweather_runtime, models
from .models.official import OfficialGenerator, load_official_weights

_base_loader = checkpoints.load_pretrained
NAMES = ("transweather", "promptir", "ramit")


def build_generator(options):
    if options.get("implementation") == "official_v1":
        if options.get("name") not in NAMES:
            raise ValueError("Unknown official model")
        unexpected = set(options) - {"name", "implementation", "pretrained"}
        if unexpected:
            raise ValueError(f"Official checkpoint topology is fixed: {sorted(unexpected)}")
        return OfficialGenerator(options["name"])
    return liteweather_runtime.build_generator(options)


def validate_restoration_config(options):
    liteweather_runtime.validate_restoration_config(options)
    model = options["model"]
    if model.get("implementation") == "official_v1":
        if model.get("name") not in NAMES:
            raise config.ConfigError("Unknown official model")
        if set(model) - {"name", "implementation", "pretrained"}:
            raise config.ConfigError("Official model topology cannot be overridden")
        if model["name"] == "transweather" and any(v % 64 for v in options["data"]["image_size"]):
            raise config.ConfigError("TransWeather training crops must be multiples of 64")


def load_pretrained(generator, path):
    if isinstance(generator, OfficialGenerator):
        load_official_weights(generator, path)
    else:
        _base_loader(generator, path)


def activate():
    models.build_generator = build_generator
    config.validate_restoration_config = validate_restoration_config
    checkpoints.load_pretrained = load_pretrained
    trainer = sys.modules.get("aegis_perception.training")
    if trainer is not None:
        trainer.build_generator = build_generator
        trainer.validate_restoration_config = validate_restoration_config
        trainer.load_pretrained = load_pretrained
