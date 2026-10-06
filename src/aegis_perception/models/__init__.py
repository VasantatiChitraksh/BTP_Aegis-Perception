"""Restoration model components."""

from .pix2pix import AttentionUNetGenerator, PatchDiscriminator, initialize_pix2pix_weights


def build_generator(config: dict):
    """Build the same generator for training, inference, and export."""
    name = config.get("name", "attention_unet")
    if name == "attention_unet":
        return AttentionUNetGenerator(
            features=int(config["features"]), attention=bool(config["attention"])
        )
    if name == "restormer":
        from .restormer import RestormerGenerator
        options = {key: value for key, value in config.items() if key not in {"name", "pretrained"}}
        return RestormerGenerator(**options)
        
    if name == "transweather":
        from .transweather import TransWeatherGenerator
        options = {key: value for key, value in config.items() if key not in {"name", "pretrained"}}
        return TransWeatherGenerator(**options)
        
    if name == "promptir":
        from .promptir import PromptIRGenerator
        options = {key: value for key, value in config.items() if key not in {"name", "pretrained"}}
        return PromptIRGenerator(**options)
        
    if name == "ramit":
        from .ramit import RAMiTGenerator
        options = {key: value for key, value in config.items() if key not in {"name", "pretrained"}}
        return RAMiTGenerator(**options)
        
    if name == "liteweatherformer":
        from .liteweatherformer import LiteWeatherFormerGenerator
        options = {key: value for key, value in config.items() if key not in {"name", "pretrained"}}
        return LiteWeatherFormerGenerator(**options)
        
    raise ValueError(f"Unknown restoration model: {name!r}")


__all__ = [
    "AttentionUNetGenerator",
    "PatchDiscriminator",
    "build_generator",
    "initialize_pix2pix_weights",
]
