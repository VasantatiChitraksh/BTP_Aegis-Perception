"""Restoration inference at original resolution, shared by validation and scripts."""

from __future__ import annotations


def predict(generator, inputs, *, precision: str = "fp32"):
    import torch
    from torch.nn import functional as functional

    height, width = inputs.shape[-2:]
    multiple = getattr(generator, "spatial_multiple", 64)
    pad_h, pad_w = (-height) % multiple, (-width) % multiple
    if pad_h or pad_w:
        inputs = functional.pad(inputs, (0, pad_w, 0, pad_h), mode="replicate")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    with torch.autocast(inputs.device.type, dtype=amp_dtype, enabled=precision != "fp32"):
        predictions = generator(inputs)
    # Keep image conversion and metric calculations in FP32 after the model forward.
    return predictions[..., :height, :width].float()
