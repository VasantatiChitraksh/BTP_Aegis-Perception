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
    raise ValueError(f"Unknown restoration model: {name!r}")


__all__ = [
    "AttentionUNetGenerator",
    "PatchDiscriminator",
    "build_generator",
    "initialize_pix2pix_weights",
]
