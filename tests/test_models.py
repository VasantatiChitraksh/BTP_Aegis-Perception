from __future__ import annotations

import importlib.util

import pytest

torch_available = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not torch_available, reason="PyTorch is not installed in this environment")
def test_generator_shape_and_attention_parameter_delta() -> None:
    import torch

    from aegis_perception.models import AttentionUNetGenerator

    attention = AttentionUNetGenerator(features=8, attention=True).eval()
    vanilla = AttentionUNetGenerator(features=8, attention=False).eval()
    sample = torch.randn(1, 3, 256, 256)
    with torch.inference_mode():
        assert attention(sample).shape == sample.shape
        assert vanilla(sample).shape == sample.shape
    attention_parameters = sum(parameter.numel() for parameter in attention.parameters())
    vanilla_parameters = sum(parameter.numel() for parameter in vanilla.parameters())
    assert attention_parameters > vanilla_parameters
