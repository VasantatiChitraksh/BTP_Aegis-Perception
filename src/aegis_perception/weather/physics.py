from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CONDITIONS = {"fog", "rain", "snow", "smoke"}


def _validate(severity: float) -> None:
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be between 0 and 1")


def add_fog(image: Image.Image, severity: float, rng: np.random.Generator) -> Image.Image:
    del rng
    _validate(severity)
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    vertical = np.linspace(0.2, 1.0, height, dtype=np.float32)[:, None]
    horizontal = 0.9 + 0.1 * np.cos(np.linspace(-math.pi, math.pi, width, dtype=np.float32))
    transmission = 1.0 - (0.18 + 0.62 * severity) * vertical * horizontal[None, :]
    transmission = np.clip(transmission[..., None], 0.18, 1.0)
    atmospheric_light = np.array([0.92, 0.94, 0.96], dtype=np.float32)
    fogged = rgb * transmission + atmospheric_light * (1.0 - transmission)
    output = Image.fromarray(np.round(np.clip(fogged, 0, 1) * 255).astype(np.uint8))
    return output.filter(ImageFilter.GaussianBlur(radius=0.3 + severity * 1.2))


def add_rain(image: Image.Image, severity: float, rng: np.random.Generator) -> Image.Image:
    _validate(severity)
    base = image.convert("RGB")
    width, height = base.size
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    count = int((120 + width * height / 3500) * (0.25 + severity))
    length = max(6, int(min(width, height) * (0.018 + 0.04 * severity)))
    slant = int(length * (0.18 + 0.22 * severity))
    line_width = max(1, int(1 + severity * 2))
    alpha = int(55 + severity * 95)
    for _ in range(count):
        x = int(rng.integers(-slant, width))
        y = int(rng.integers(-length, height))
        draw.line((x, y, x + slant, y + length), fill=(190, 205, 220, alpha), width=line_width)
    layer = layer.filter(ImageFilter.GaussianBlur(radius=0.35 + severity * 0.7))
    return Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")


def add_snow(image: Image.Image, severity: float, rng: np.random.Generator) -> Image.Image:
    _validate(severity)
    base = image.convert("RGB")
    width, height = base.size
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    count = int((80 + width * height / 2600) * (0.2 + severity * 1.25))
    for _ in range(count):
        radius = float(rng.uniform(0.6, 1.8 + severity * 3.5))
        x = float(rng.uniform(0, width))
        y = float(rng.uniform(0, height))
        alpha = int(rng.integers(90, int(150 + severity * 105)))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(245, 248, 255, alpha))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=0.25 + severity * 0.6))
    result = Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")
    if severity > 0.55:
        snow_tint = Image.new("RGB", result.size, (225, 232, 238))
        result = Image.blend(result, snow_tint, 0.08 * severity)
    return result


def add_smoke(image: Image.Image, severity: float, rng: np.random.Generator) -> Image.Image:
    _validate(severity)
    base = image.convert("RGB")
    width, height = base.size
    low_width, low_height = max(4, width // 32), max(4, height // 32)
    noise = rng.random((low_height, low_width), dtype=np.float32)
    mask = Image.fromarray(np.round(noise * 255).astype(np.uint8), mode="L")
    mask = mask.resize((width, height), Image.Resampling.BICUBIC)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(width, height) * 0.035))
    mask_array = np.asarray(mask, dtype=np.float32) / 255.0
    alpha = np.clip((mask_array - 0.18) * (0.18 + severity * 0.62), 0.0, 0.72)
    smoke_rgba = np.empty((height, width, 4), dtype=np.uint8)
    smoke_rgba[..., :3] = np.array([188, 190, 192], dtype=np.uint8)
    smoke_rgba[..., 3] = np.round(alpha * 255).astype(np.uint8)
    layer = Image.fromarray(smoke_rgba, mode="RGBA")
    return Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")


def degrade(
    image: Image.Image,
    condition: str,
    *,
    severity: float,
    seed: int,
) -> Image.Image:
    """Apply seeded, geometry-preserving synthetic weather to an image."""
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {sorted(CONDITIONS)}")
    rng = np.random.default_rng(seed)
    functions = {"fog": add_fog, "rain": add_rain, "snow": add_snow, "smoke": add_smoke}
    return functions[condition](image, severity, rng)
