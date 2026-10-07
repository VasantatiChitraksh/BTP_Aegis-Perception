# MIT License
# Copyright (c) 2022 Syed Waqas Zamir and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Adapted from swz30/Restormer, revision 68dc6ac472db26f16361150cb7a96a1bc87da93f.
# https://github.com/swz30/Restormer/blob/main/basicsr/models/archs/restormer_arch.py
# Native PyTorch reshapes replace einops; parameter names/topology are preserved.

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_3d(x):
    return x.flatten(2).transpose(1, 2)


def to_4d(x, h, w):
    return x.transpose(1, 2).reshape(x.shape[0], -1, h, w)


# Older CUDA depthwise kernels require fewer than INT_MAX elements per call.
_DEPTHWISE_INDEX_LIMIT = 2**31 - 1
_DEPTHWISE_CHUNK_ELEMENTS = 2**28


def _depthwise_conv(conv, inputs):
    """Evaluate large depthwise tensors by independent channels, preserving spatial context."""
    if inputs.numel() < _DEPTHWISE_INDEX_LIMIT:
        return conv(inputs)

    batch, channels, height, width = inputs.shape
    chunk_channels = max(1, _DEPTHWISE_CHUNK_ELEMENTS // (batch * height * width))
    output = None
    for start in range(0, channels, chunk_channels):
        end = min(start + chunk_channels, channels)
        chunk = F.conv2d(
            inputs[:, start:end].contiguous(),
            conv.weight[start:end],
            None if conv.bias is None else conv.bias[start:end],
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=end - start,
        )
        if output is None:
            output = chunk.new_empty(inputs.shape)
        output[:, start:end] = chunk
    return output


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-05) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-05) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        (h, w) = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        (x1, x2) = _depthwise_conv(self.dwconv, x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        (b, c, h, w) = x.shape
        qkv = _depthwise_conv(self.qkv_dwconv, self.qkv(x))
        (q, k, v) = qkv.chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.reshape(b, c, h, w)
        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Restormer(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=(4, 6, 6, 8),
        num_refinement_blocks=4,
        heads=(1, 2, 4, 8),
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        dual_pixel_task=False,
    ):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )
        self.down2_3 = Downsample(int(dim * 2**1))
        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )
        self.down3_4 = Downsample(int(dim * 2**2))
        self.latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**3),
                    num_heads=heads[3],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[3])
            ]
        )
        self.up4_3 = Upsample(int(dim * 2**3))
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias
        )
        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )
        self.up3_2 = Upsample(int(dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )
        self.up2_1 = Upsample(int(dim * 2**1))
        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )
        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_refinement_blocks)
            ]
        )
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2**1), kernel_size=1, bias=bias)
        self.output = nn.Conv2d(
            int(dim * 2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img
        return out_dec_level1


class RestormerGenerator(nn.Module):
    """Official Restormer with this project's [-1, 1] input/output interface.

    The original network operates on RGB [0, 1]. Conversion happens here so
    upstream deraining weights work without changing legacy Pix2Pix data.
    Outputs remain unclamped during training.
    """

    spatial_multiple = 8

    def __init__(
        self,
        dim=48,
        num_blocks=(4, 6, 6, 8),
        num_refinement_blocks=4,
        heads=(1, 2, 4, 8),
        ffn_expansion_factor=2.66,
        bias=False,
        layer_norm_type="WithBias",
    ):
        super().__init__()
        self.network = Restormer(
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=layer_norm_type,
        )

    def forward(self, inputs):
        return self.network(inputs.add(1).div(2)).mul(2).sub(1)
