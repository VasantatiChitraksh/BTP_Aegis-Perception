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

    from .models import AttentionUNetGenerator

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_config = checkpoint["config"]["model"]
    generator = AttentionUNetGenerator(
        features=int(model_config["features"]), attention=bool(model_config["attention"])
    ).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    return generator, checkpoint
