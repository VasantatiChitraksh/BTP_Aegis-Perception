from __future__ import annotations

from pathlib import Path
from typing import Any


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    import torch

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output_path)


def load_generator(path: str | Path, *, device: str):
    import torch

    from .models import build_generator

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_config = checkpoint["config"]["model"]
    generator = build_generator(model_config).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    return generator, checkpoint


def load_pretrained(generator, path: str | Path) -> None:
    """Strictly load official Restormer params or a project's generator weights."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "generator" in checkpoint:
        generator.load_state_dict(checkpoint["generator"], strict=True)
        return
    weights = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
    weights = {key.removeprefix("module."): value for key, value in weights.items()}
    target = getattr(generator, "network", generator)
    target.load_state_dict(weights, strict=True)


def select_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested
