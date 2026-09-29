from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image

from .manifest import PairRecord, load_manifest


def pil_to_normalized_tensor(image: Image.Image):
    import torch

    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).mul(2.0).sub(1.0)


def normalized_tensor_to_pil(tensor) -> Image.Image:
    array = tensor.detach().cpu().clamp(-1, 1).add(1).div(2).permute(1, 2, 0).numpy()
    return Image.fromarray(np.round(array * 255).astype(np.uint8), mode="RGB")


class PairedImageDataset:
    """Paired restoration data with synchronized geometry transforms."""

    def __init__(
        self,
        manifest: str | Path,
        split: str,
        image_size: tuple[int, int] = (256, 256),
        random_flip: bool = False,
        spatial_mode: str = "resize",
        random_crop: bool = True,
        max_samples: int | None = None,
    ) -> None:
        self.records: list[PairRecord] = load_manifest(manifest, split=split)
        if not self.records:
            raise ValueError(f"No records for split={split!r} in {manifest}")
        self.image_size = tuple(image_size)
        self.random_flip = random_flip
        if spatial_mode not in {"resize", "crop", "native"}:
            raise ValueError(f"Unknown spatial mode: {spatial_mode!r}")
        self.spatial_mode = spatial_mode
        self.random_crop = random_crop
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            self.records = self.records[:max_samples]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.input_path) as image:
            degraded = image.convert("RGB")
        with Image.open(record.target_path) as image:
            clean = image.convert("RGB")

        if degraded.size != clean.size:
            raise ValueError(f"Pair dimensions differ: {record.sample_id}")
        if self.spatial_mode == "resize":
            resize_to = (self.image_size[1], self.image_size[0])
            degraded = degraded.resize(resize_to, Image.Resampling.BICUBIC)
            clean = clean.resize(resize_to, Image.Resampling.BICUBIC)
        elif self.spatial_mode == "crop":
            height, width = self.image_size
            pad_h = max(0, height - degraded.height)
            pad_w = max(0, width - degraded.width)
            if pad_h or pad_w:
                padding = ((0, pad_h), (0, pad_w), (0, 0))
                degraded = Image.fromarray(np.pad(np.asarray(degraded), padding, mode="edge"))
                clean = Image.fromarray(np.pad(np.asarray(clean), padding, mode="edge"))
            if self.random_crop:
                left = random.randint(0, degraded.width - width)
                top = random.randint(0, degraded.height - height)
            else:
                left = (degraded.width - width) // 2
                top = (degraded.height - height) // 2
            box = (left, top, left + width, top + height)
            degraded, clean = degraded.crop(box), clean.crop(box)
        if self.random_flip and random.random() < 0.5:
            degraded = degraded.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            clean = clean.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        return {
            "input": pil_to_normalized_tensor(degraded),
            "target": pil_to_normalized_tensor(clean),
            "sample_id": record.sample_id,
            "weather": record.weather,
        }
