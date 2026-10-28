# Quantization Research — Aegis Perception

> **Project:** BTP Aegis Perception — All-Weather Image Restoration  
> **Target Hardware:** NVIDIA Jetson Orin Nano (primary) · Raspberry Pi 5 (secondary)  
> **Date:** 2026-10

---

## Table of Contents

1. [Motivation](#motivation)
2. [Background: Number Formats](#background-number-formats)
3. [Post-Training Quantization (PTQ)](#post-training-quantization-ptq)
   - [FP32 → FP16](#fp32--fp16)
   - [FP32 → INT8 Dynamic](#fp32--int8-dynamic)
   - [FP32 → INT8 Static (Calibration)](#fp32--int8-static-calibration)
4. [Quantization-Aware Training (QAT)](#quantization-aware-training-qat)
5. [NVIDIA TensorRT INT8 Path for Jetson Orin Nano](#nvidia-tensorrt-int8-path-for-jetson-orin-nano)
6. [PyTorch Quantization API Overview](#pytorch-quantization-api-overview)
7. [Model-Family Analysis](#model-family-analysis)
8. [Decision Matrix](#decision-matrix)
9. [Final Recommendation](#final-recommendation)
10. [References](#references)

---

## Motivation

Restoring rain-degraded images in real time on edge hardware imposes strict latency and memory budgets:

| Hardware | GPU VRAM | Peak Bandwidth | Target latency |
|---|---|---|---|
| Jetson Orin Nano | 4–8 GB shared | 68 GB/s | ≤ 50 ms / frame |
| Raspberry Pi 5 | — (CPU only) | ~25 GB/s LPDDR4X | ≤ 500 ms / frame |

FP32 inference on even a lightweight model (e.g., RAMiT at ~4 M params) consumes ~16 MB of weight memory; heavy models (Restormer, PromptIR, ~26 M params) consume ~100 MB. Quantization compresses both model size and arithmetic cost, unlocking deployment within these constraints.

---

## Background: Number Formats

| Format | Bits | Dynamic Range | Notes |
|---|---|---|---|
| FP32 | 32 | ±3.4 × 10³⁸ | Training baseline |
| FP16 | 16 | ±65504 | Native Tensor Core format on Ampere |
| BF16 | 16 | ±3.4 × 10³⁸ | Better range, less precision; Ampere supports |
| INT8 | 8 | −128 … 127 | 4× memory vs FP32; needs scale+zero-point |
| INT4 | 4 | −8 … 7 | 8× memory; experimental in PyTorch 2.x |

For image restoration the output quality is extremely sensitive to rounding. The quantization order of preference in terms of accuracy loss is:

```
FP16 ≈ BF16  >>>  INT8-dynamic > INT8-static  >>>  INT4
```

---

## Post-Training Quantization (PTQ)

PTQ converts a trained FP32 checkpoint without any re-training. It is the first step to attempt because it requires no changes to the training pipeline.

### FP32 → FP16

**Mechanism.** Every weight and activation tensor is cast from 32-bit float to 16-bit float. Modern PyTorch and TensorRT both support this natively.

```python
model = model.half()          # weights to FP16
input_tensor = input.half()   # inputs must also be FP16
```

On CUDA devices, `torch.autocast` (AMP) is preferred:

```python
with torch.autocast(device_type='cuda', dtype=torch.float16):
    output = model(input_tensor)   # model stays FP32 on disk
```

**Accuracy impact.** Usually < 0.1 dB PSNR drop for convolution-heavy models. Transformer attention softmax can encounter NaN/overflow with very long sequences due to the reduced FP16 range; BF16 (available on Ampere) is safer.

**Hardware benefit (Jetson Orin Nano).** The Ampere GPU executes FP16 Tensor Core MMA at 2× the FP32 throughput. Typical speed-up: **1.5–2×**.

**CPU (Raspberry Pi 5).** FP16 is not accelerated on Cortex-A76 (ARMv8.2 without dotprod for FP16). FP16 inference on CPU may be *slower* than FP32. Use INT8 dynamic for CPU targets.

---

### FP32 → INT8 Dynamic

**Mechanism.** Weights are quantized to INT8 at load time (offline). Activations are quantized on-the-fly per tensor, so no calibration dataset is required.

```python
import torch.ao.quantization as taq

model_int8 = taq.quantize_dynamic(
    model,
    qconfig_spec={torch.nn.Linear},   # target layer types
    dtype=torch.qint8,
)
```

**Limitations.**
- PyTorch's `quantize_dynamic` only targets `torch.nn.Linear` (and optionally `LSTM`) out of the box. `Conv2d` is **not** supported in dynamic mode via this API.
- For Conv2d-heavy architectures (Attention U-Net, RAMiT, LiteWeatherFormer) the practical impact of dynamic INT8 is limited to any MLP/attention-projection Linear layers.
- Transformer models (Restormer, TransWeather, PromptIR) use many `Linear` projections inside attention; they benefit more.

**Accuracy impact.** Typically 0.1–0.5 dB PSNR drop for transformers; minimal for conv-only.

**CPU benefit.** PyTorch's quantized linear kernel runs INT8 dot-products on ARM NEON, giving **1.3–2×** speed-up on Raspberry Pi 5.

---

### FP32 → INT8 Static (Calibration)

**Mechanism.** Both weights *and* activations are pre-quantized using statistics (min/max or percentile) collected from a representative calibration set. This requires inserting `QuantStub` / `DeQuantStub` around the model and calling `prepare` + `convert`.

```python
import torch.ao.quantization as taq

class QuantWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quant  = taq.QuantStub()
        self.model  = model
        self.dequant = taq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        return self.dequant(x)

wrapped = QuantWrapper(model)
wrapped.qconfig = taq.get_default_qconfig('fbgemm')   # CPU
# or 'qnnpack' for ARM; 'x86' for newer API
taq.prepare(wrapped, inplace=True)

# --- calibration ---
for img in calibration_loader:
    with torch.no_grad():
        wrapped(img)

taq.convert(wrapped, inplace=True)
```

**Calibration requirements.** A minimum of ~100 representative rain images is recommended for stable scale/zero-point estimates. A single image (as used in `quantize_compare.py`) is only valid for sanity-checking relative degradation; production calibration must use a proper subset of the training distribution.

**Limitations for transformer layers.**
Static INT8 quantization does not compose cleanly with `torch.nn.MultiheadAttention` because its `scaled_dot_product_attention` uses float-precision softmax internally. Attempting to convert these layers will either raise or silently produce incorrect results. **For Restormer, TransWeather, and PromptIR, dynamic INT8 is the recommended fallback for the attention sub-modules; conv projections can still be statically quantized.**

**Conv2d support.** PyTorch static INT8 *does* quantize `Conv2d` fully, making it ideal for Attention U-Net, RAMiT, and LiteWeatherFormer.

**Accuracy impact.** 0.2–1.0 dB PSNR drop. Quality depends heavily on calibration dataset representativeness. Using percentile (99.9th) rather than min/max reduces outlier distortion.

---

## Quantization-Aware Training (QAT)

QAT simulates quantization noise during training using *fake quantization* nodes (straight-through estimator). The model learns to compensate for the rounding errors.

```python
import torch.ao.quantization as taq

model.train()
model.qconfig = taq.get_default_qat_qconfig('fbgemm')
taq.prepare_qat(model, inplace=True)

# Fine-tune for 5–20 % of the original training epochs
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
for batch in dataloader:
    loss = criterion(model(batch['input']), batch['target'])
    loss.backward()
    optimizer.step()

model.eval()
taq.convert(model, inplace=True)   # produces final INT8 model
```

### When to use QAT

- PTQ accuracy drop > 0.5 dB PSNR on the validation set.
- The model has batch normalisation layers whose running stats shift under quantization.
- Target is INT8 *static* for a conv-heavy model on Jetson/Pi.
- Model has narrow dynamic range activations (common in residual blocks with ReLU).

### When PTQ is sufficient

- FP16 (always try first on Jetson Orin Nano).
- INT8 dynamic on transformer projections (< 0.3 dB typical).
- Rapid prototyping / benchmarking before committing training compute.

**Compute cost.** QAT requires re-running the training loop for at least 5 epochs. Budget ~20 % of original training time.

---

## NVIDIA TensorRT INT8 Path for Jetson Orin Nano

TensorRT is NVIDIA's inference optimiser, available on JetPack 5 / 6. It performs layer fusion, kernel auto-tuning, and INT8 calibration independent of PyTorch.

### Workflow

```
PyTorch FP32 model
       │
       ▼  torch.onnx.export()
    ONNX model (.onnx)
       │
       ▼  trtexec --onnx=model.onnx \
       │          --int8 \
       │          --calib=<calib_cache> \
       │          --saveEngine=model.trt
    TensorRT Engine (.trt)
       │
       ▼  tensorrt.Runtime  /  torch2trt  /  polygraphy
    Optimised inference on Jetson
```

### Key TensorRT flags for Jetson Orin Nano

```bash
trtexec \
  --onnx=model.onnx \
  --fp16 \               # FP16 Tensor Cores
  --int8 \               # INT8 for eligible layers
  --calib=calib.cache \  # from IInt8EntropyCalibrator2
  --minShapes=input:1x3x256x256 \
  --optShapes=input:1x3x512x512 \
  --maxShapes=input:1x3x1024x1024 \
  --saveEngine=model_jetson.trt \
  --best                 # trtexec chooses fastest per layer
```

### Calibration cache

The calibration cache can be generated using the `tensorrt.IInt8EntropyCalibrator2` interface or via `torch2trt`'s calibration helpers. The same ~100-image calibration set used for PyTorch static INT8 is appropriate.

### Expected TensorRT speed-up vs PyTorch FP32

| Model | FP32 (ms) | FP16 TRT (ms) | INT8 TRT (ms) | INT8 speed-up |
|---|---|---|---|---|
| Attention U-Net | ~35 | ~18 | ~11 | ~3.2× |
| RAMiT | ~28 | ~14 | ~9 | ~3.1× |
| LiteWeatherFormer | ~22 | ~11 | ~7 | ~3.1× |
| Restormer | ~80 | ~42 | ~28 | ~2.9× |
| TransWeather | ~65 | ~34 | ~22 | ~3.0× |
| PromptIR | ~90 | ~47 | ~31 | ~2.9× |

*(Estimates at 512×512 resolution on Jetson Orin Nano; actual results will vary.)*

---

## PyTorch Quantization API Overview

### Legacy API (`torch.quantization` — deprecated in 2.x)

```python
import torch.quantization as tq
tq.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

### Recommended API (`torch.ao.quantization`)

```python
import torch.ao.quantization as taq

# PTQ dynamic
model_q = taq.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

# PTQ static
model.qconfig = taq.get_default_qconfig('x86')   # or 'fbgemm', 'qnnpack'
taq.prepare(model, inplace=True)                  # insert observers
# ... run calibration data through model ...
taq.convert(model, inplace=True)                  # replace with quantized ops

# QAT
model.qconfig = taq.get_default_qat_qconfig('x86')
taq.prepare_qat(model, inplace=True)
# ... train for N epochs ...
model.eval()
taq.convert(model, inplace=True)
```

### New FX-Graph Mode (`torch.ao.quantization.quantize_fx`)

```python
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import get_default_qconfig_mapping

qconfig_mapping = get_default_qconfig_mapping('x86')
model_prepared  = prepare_fx(model, qconfig_mapping, example_inputs=(dummy,))
# calibrate...
model_int8      = convert_fx(model_prepared)
```

FX mode handles more operator patterns automatically (including some attention ops) and is the recommended path for PyTorch ≥ 2.0.

---

## Model-Family Analysis

### 1. Attention U-Net (Pix2Pix)

- **Architecture:** Pure CNN. 6-level encoder-decoder, skip connections with channel-attention gates.
- **Layer breakdown:** 100% Conv2d / ConvTranspose2d, BatchNorm2d, ReLU.
- **Quantization suitability:** **Excellent.** All layers fully supported by PyTorch static INT8. TensorRT INT8 is straightforward from ONNX.
- **Expected PSNR drop:** FP16 < 0.1 dB, INT8-static ~ 0.2–0.4 dB with proper calibration.
- **Recommended:** FP16 for Jetson (easiest, great speed-up); INT8-static for Pi via ONNX Runtime QNNPACK.

### 2. Restormer

- **Architecture:** Transformer-based. MDTA (multi-DConv head transposed attention) + GDFN (gated depthwise feed-forward). Uses LayerNorm.
- **Layer breakdown:** ~60% Conv2d (DWConv), ~40% attention + LayerNorm + Linear.
- **Quantization suitability:** **Moderate.** Linear projections in attention quantize well (dynamic INT8). DWConv can use static INT8. Softmax / LayerNorm accumulation sensitive to FP16 overflow with long sequences.
- **Expected PSNR drop:** FP16 < 0.15 dB (BF16 preferred); INT8-dynamic ~ 0.3–0.6 dB.
- **Recommended:** FP16/BF16 on Jetson; mixed dynamic INT8 (conv static + attn dynamic) for Pi.

### 3. TransWeather

- **Architecture:** Conv encoder-decoder with learnable weather query tokens (transformer bottleneck).
- **Layer breakdown:** ~70% Conv2d, ~30% MHA + Linear.
- **Quantization suitability:** **Good.** Conv path statically quantizable; MHA dynamic.
- **Expected PSNR drop:** FP16 < 0.1 dB; INT8-dynamic ~0.3 dB.
- **Recommended:** FP16 Jetson; dynamic INT8 Pi.

### 4. PromptIR

- **Architecture:** Prompt-based all-in-one restoration. Shared backbone + prompt banks (learned embeddings + cross-attention).
- **Layer breakdown:** ~55% Conv, ~45% Linear/attention.
- **Quantization suitability:** **Moderate.** Prompt embeddings (embedding tables) require special handling; not directly quantizable by `quantize_dynamic`. Cross-attention dynamic INT8.
- **Expected PSNR drop:** FP16 < 0.2 dB; INT8 ~ 0.4–0.7 dB.
- **Recommended:** FP16 Jetson; QAT may be needed if INT8 drop is excessive.

### 5. RAMiT

- **Architecture:** Lightweight spatial + channel attention hierarchy. Conv-dominant with lightweight SE blocks.
- **Layer breakdown:** ~90% Conv2d, ~10% Linear (SE reduction).
- **Quantization suitability:** **Excellent.** Very amenable to static INT8. Smallest model in the suite.
- **Expected PSNR drop:** FP16 < 0.05 dB; INT8-static ~ 0.15–0.3 dB.
- **Recommended:** INT8-static for both Jetson and Pi (best compute efficiency for smallest quality loss).

### 6. LiteWeatherFormer (Novel Model)

- **Architecture:** Conv Stem + depthwise encoder + weather prompt tokens + skip-fusion decoder.
- **Layer breakdown:** ~75% Depthwise/Pointwise Conv, ~15% Linear (prompt cross-attn), ~10% skip fusion.
- **Quantization suitability:** **Good.** DWConv + PWConv fully supported by static INT8. Prompt cross-attention uses dynamic INT8.
- **Expected PSNR drop:** FP16 < 0.1 dB; INT8-static ~ 0.2–0.35 dB (better than full transformers).
- **Recommended:** FP16 Jetson primary; INT8-static on Pi for CPU deployment.

---

## Decision Matrix

| Model | FP16 (Jetson) | INT8-Dyn (Pi) | INT8-Static (Jetson/Pi) | QAT needed? | TRT INT8 |
|---|---|---|---|---|---|
| Attention U-Net | ✅ Excellent | ⚠️ Partial (no conv) | ✅ Excellent | No (PTQ sufficient) | ✅ Full |
| Restormer | ✅ Good (BF16 safer) | ✅ Good (attn layers) | ⚠️ Partial (attn fallback) | Maybe (> 0.5 dB) | ✅ FP16+INT8 mixed |
| TransWeather | ✅ Excellent | ✅ Good | ⚠️ Partial | No | ✅ Full |
| PromptIR | ✅ Good | ✅ Good | ⚠️ Partial | Maybe | ✅ FP16+INT8 mixed |
| RAMiT | ✅ Excellent | ⚠️ Minimal gain | ✅ Excellent | No | ✅ Full |
| LiteWeatherFormer | ✅ Excellent | ✅ Good | ✅ Good | No | ✅ Full |

**Legend:** ✅ Recommended / works well · ⚠️ Works with caveats · ❌ Not recommended

---

## Final Recommendation

### For Jetson Orin Nano

1. **First pass: FP16 via `model.half()` or AMP.** Zero re-training, 1.5–2× speed-up, < 0.15 dB PSNR drop on all models. This alone may meet latency targets for most models.

2. **Second pass: TensorRT INT8 engine** (via ONNX export) for the three conv-heavy models (Attention U-Net, RAMiT, LiteWeatherFormer) where static INT8 is safe. Expected 3× speed-up over FP32.

3. **For transformer models (Restormer, TransWeather, PromptIR):** Use TensorRT with `--fp16` and let TRT decide per-layer precision. Do not force full INT8 without calibration validation.

### For Raspberry Pi 5 (CPU)

1. **INT8 Dynamic Quantization** on all `nn.Linear` layers (attention projections) — direct gain for transformer models.
2. **ONNX Runtime with QNNPACK / ARM backend** for Conv layers via INT8 static. Export with `torch.onnx.export` and run `onnxruntime` with `ExecutionProvider=['CPUExecutionProvider']`.
3. **Avoid FP16 on CPU** — no hardware acceleration; will be slower than FP32.

### Priority order for this project

```
LiteWeatherFormer → RAMiT → Attention U-Net → TransWeather → Restormer → PromptIR
```

LiteWeatherFormer and RAMiT deliver the best compression-quality tradeoff and are designed for edge deployment. Begin with FP16, validate PSNR, then attempt INT8-static with a 100-image calibration subset.

---

## References

1. Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*, CVPR 2018.
2. PyTorch documentation — [Quantization](https://pytorch.org/docs/stable/quantization.html)
3. NVIDIA TensorRT Developer Guide — [INT8 Calibration](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#enable_int8_c)
4. Zamir et al., *Restormer: Efficient Transformer for High-Resolution Image Restoration*, CVPR 2022.
5. Valanarasu et al., *TransWeather: Transformer-based Restoration of Images Degraded by Adverse Weather Conditions*, CVPR 2022.
6. Potlapalli et al., *PromptIR: Prompting for All-in-One Blind Image Restoration*, NeurIPS 2023.
7. JetPack 6 — Jetson Orin Nano Product Brief, NVIDIA 2024.
