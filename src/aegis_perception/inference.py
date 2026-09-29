"""Restoration inference at original resolution, shared by validation and scripts."""

from __future__ import annotations


def predict(generator, inputs):
    from torch.nn import functional as functional

    height, width = inputs.shape[-2:]
    multiple = getattr(generator, "spatial_multiple", 64)
    pad_h, pad_w = (-height) % multiple, (-width) % multiple
    if pad_h or pad_w:
        inputs = functional.pad(inputs, (0, pad_w, 0, pad_h), mode="replicate")
    return generator(inputs)[..., :height, :width]
