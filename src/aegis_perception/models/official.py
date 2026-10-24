"""Checkpoint-compatible adapters for pinned official model sources.

Sources live in ignored artifacts/official_sources and are verified before execution.
Only compatibility imports and equivalent attention execution are changed in memory.
The original contributor implementations are never imported for these experiments.
"""

import hashlib
import json
import sys
import types
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
LOCK = ROOT / "configs/restoration/official_v1/sources.json"


def source_info(name):
    return json.loads(LOCK.read_text())["models"][name]


def source_text(name, suffix):
    entry = next(e for e in source_info(name)["sources"] if e["path"].endswith(suffix))
    path = ROOT / entry["path"]
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"Official source checksum mismatch: {path}")
    return data.decode(), str(path)


def _module(name, text, filename, namespace=None):
    module = types.ModuleType(name)
    module.__file__ = filename
    module.__dict__.update(namespace or {})
    sys.modules[name] = module
    exec(compile(text, filename, "exec"), module.__dict__)
    return module


def load_architecture(name, *, efficient_attention=True):
    """Load verified code in private namespaces, avoiding upstream utils collisions."""
    prefix = "aegis_perception.models._upstream_"
    support = "aegis_perception.models.official_support"
    if name == "transweather":
        text, path = source_text(name, "base_networks.py")
        _module(prefix + "transweather_base", text, path)
        text, path = source_text(name, "transweather_model.py")
        text = text.replace(
            "from base_networks import *", f"from {prefix}transweather_base import *"
        )
        text = text.replace("import timm\n", "")
        text = text.replace("from timm.models.layers import", f"from {support} import")
        text = text.replace("from mmcv.cnn import ConvModule", "")  # Unused import upstream.
        if efficient_attention:
            # Full-image attention, not tiling: SDPA avoids a quadratic materialized matrix.
            old = """attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)"""
            if text.count(old) != 2:
                raise ValueError("Unexpected TransWeather attention source")
            text = text.replace(
                old,
                """attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale)""",
            )
            text = text.replace("(attn @ v).transpose", "attended.transpose")
        return _module(prefix + name, text, path)
    if name == "promptir":
        text, path = source_text(name, "net/model.py")
        return _module(prefix + name, text, path)
    if name == "ramit":
        text, path = source_text(name, "common/mean_std.py")
        mean_module = _module(prefix + "ramit_mean", text, path)
        text, path = source_text(name, "my_model/ramit.py")
        text = text.replace("from .common.mean_std import mean_std", "")
        text = text.replace(
            "from torchvision.transforms import functional as TF",
            f"from {support} import TensorTransforms as TF",
        )
        text = text.replace(
            "from utils.etc_utils import denormalize", f"from {support} import denormalize"
        )
        text = text.replace("from timm.models.fx_features import", f"from {support} import")
        text = text.replace("from timm.models.layers import", f"from {support} import")
        return _module(prefix + name, text, path, {"mean_std": mean_module.mean_std})
    raise ValueError(f"Unknown official architecture: {name}")


class OfficialGenerator(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        source = load_architecture(name)
        if name == "transweather":
            self.network = source.Transweather()
            self.spatial_multiple = 64
        elif name == "promptir":
            self.network = source.PromptIR(decoder=True)
            self.spatial_multiple = 8
            from .official_support import safe_depthwise_forward

            for layer in self.network.modules():
                if isinstance(layer, nn.Conv2d) and layer.groups == layer.in_channels > 1:
                    assert layer.stride == (1, 1) and layer.in_channels == layer.out_channels
                    layer.forward = types.MethodType(safe_depthwise_forward, layer)
        elif name == "ramit":
            self.network = source.RAMiT(target_mode="light_dr", mv_ver=2, helper=False)
            # RAMiT pads/crops internally using its official zero-padding convention.
            self.spatial_multiple = 1
        else:
            raise ValueError(name)

    def forward(self, inputs):
        # Shared pipeline images/targets are [-1,1]. Official target tensors are [0,1].
        native_inputs = inputs if self.name == "transweather" else (inputs + 1) / 2
        return self.network(native_inputs) * 2 - 1


def load_official_weights(generator, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    weights = checkpoint.get("state_dict", checkpoint)
    weights = {k.removeprefix("module.").removeprefix("net."): v for k, v in weights.items()}
    generator.network.load_state_dict(weights, strict=True)
