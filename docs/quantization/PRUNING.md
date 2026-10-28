# Pruning Research — Aegis Perception

> **Project:** BTP Aegis Perception — All-Weather Image Restoration  
> **Target Hardware:** NVIDIA Jetson Orin Nano · Raspberry Pi 5  
> **Date:** 2026-10

---

## Table of Contents

1. [Why Prune?](#why-prune)
2. [Structured vs Unstructured Pruning](#structured-vs-unstructured-pruning)
3. [Magnitude-Based Pruning with PyTorch](#magnitude-based-pruning-with-pytorch)
4. [Global vs Local Pruning](#global-vs-local-pruning)
5. [Identifying Unimportant Layers](#identifying-unimportant-layers)
6. [Iterative Pruning-and-Fine-Tuning](#iterative-pruning-and-fine-tuning)
7. [Model-Specific Notes](#model-specific-notes)
8. [Recovery Fine-Tuning](#recovery-fine-tuning)
9. [Evaluation Protocol](#evaluation-protocol)
10. [References](#references)

---

## Why Prune?

Pruning removes redundant parameters from a trained neural network, reducing:

- **Memory footprint** — smaller model file, less DRAM bandwidth on Jetson/Pi.
- **FLOP count** — structured pruning removes entire filters, giving real speedup.
- **Inference latency** — fewer multiply-accumulate operations per forward pass.

Unlike quantization (which changes number precision), pruning changes the *topology* of the model. The two techniques are orthogonal and often combined (prune → quantize).

**Key trade-off.** Aggressive pruning causes accuracy degradation (PSNR drop). Fine-tuning after pruning recovers most of the loss.

---

## Structured vs Unstructured Pruning

### Unstructured Pruning

Individual scalar weights are zeroed based on a criterion (e.g., magnitude). The weight tensor retains its original shape but contains many zeros.

```
Dense weight matrix (4×4):
[ 0.8  -0.3   0.1   0.7 ]        [ 0.8   0    0    0.7 ]
[-0.6   0.4  -0.9   0.2 ]  --->  [-0.6   0.4  -0.9  0  ]   (50 % sparsity)
[ 0.5  -0.1   0.3  -0.4 ]        [ 0.5   0    0.3  -0.4]
[-0.2   0.6  -0.7   0.9 ]        [ 0     0.6  -0.7  0.9]
```

**Pros.** Fine-grained; achieves high sparsity with minimal accuracy loss.  
**Cons.** Sparse tensors are not efficiently executed on GPU/CPU without specialised sparse kernels (e.g., NVIDIA cuSPARSELt, TorchSparse). Standard dense inference ignores the zeros.

**When to use.** When targeting NVIDIA's sparse Tensor Core feature (Ampere 2:4 structured sparsity) or when the downstream runtime supports sparse execution (e.g., TensorRT sparse mode).

### Structured Pruning

Entire filters (output channels) or neurons are removed, yielding a smaller dense tensor.

```
Conv2d(in=3, out=64, k=3)  →  prune 16 filters  →  Conv2d(in=3, out=48, k=3)
```

**Pros.** Reduced model is a standard dense network → no specialised sparse kernels needed → immediate latency improvement on all hardware.  
**Cons.** Coarser granularity; same sparsity level causes more accuracy loss than unstructured.

**When to use.** When targeting Jetson Orin Nano or Raspberry Pi 5 where sparse acceleration is unavailable or undesirable.

> [!IMPORTANT]
> For this project, **unstructured magnitude pruning** is used as the research baseline (implemented in `prune_model.py`). Structured filter pruning is the recommended next step for production deployment but requires architecture-aware layer coupling analysis.

---

## Magnitude-Based Pruning with PyTorch

`torch.nn.utils.prune` provides a clean API for unstructured and structured pruning:

```python
import torch.nn.utils.prune as prune

# Unstructured L1 pruning on a single Conv2d layer
prune.l1_unstructured(layer, name='weight', amount=0.3)
# 'amount' is the fraction of weights to zero out

# Remove the pruning reparameterization (make permanent)
prune.remove(layer, 'weight')
```

Under the hood, `prune.l1_unstructured` does the following:
1. Computes the L1 norm of every scalar in the weight tensor.
2. Creates a binary mask where the `amount` smallest-norm weights are 0.
3. Registers a `weight_mask` buffer and a `weight_orig` parameter.
4. The effective `weight` = `weight_orig * weight_mask`.

### Supported Pruning Methods

| Method | API | Criterion |
|---|---|---|
| L1 Unstructured | `prune.l1_unstructured` | L1 magnitude of individual weights |
| Random Unstructured | `prune.random_unstructured` | Random |
| L1 Structured | `prune.ln_structured` | L1 norm of entire filters (along dim) |
| Random Structured | `prune.random_structured` | Random filters |
| Custom | Subclass `BasePruningMethod` | Any criterion |

---

## Global vs Local Pruning

### Local Pruning

Each layer is pruned independently to the same target sparsity:

```python
for module in model.modules():
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        prune.l1_unstructured(module, name='weight', amount=sparsity)
```

**Behaviour.** Every layer reaches exactly `sparsity` fraction zero. Useful for uniform budget allocation but ignores that some layers are more sensitive than others.

### Global Pruning

A global threshold is computed across *all* parameters jointly, and the least important weights across the entire network are zeroed regardless of which layer they come from:

```python
parameters_to_prune = [
    (module, 'weight')
    for module in model.modules()
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d))
]
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=sparsity,
)
```

**Behaviour.** Final per-layer sparsities will differ. Insensitive layers (large, redundant filters) get pruned more aggressively; sensitive layers (bottlenecks) retain more weights. Global pruning consistently outperforms local pruning at the same target sparsity.

> [!TIP]
> Always use **global** pruning as the first strategy. Switch to local only if specific layers must be constrained (e.g., first conv receiving raw pixel input is always sensitive).

---

## Identifying Unimportant Layers

Before pruning, it is useful to rank layers by their "importance" to avoid catastrophic accuracy drops.

### 1. L1 Norm of Weights

```python
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        l1 = module.weight.data.abs().mean().item()
        print(f"{name:60s}  L1={l1:.4f}")
```

Layers with low average L1 norm contain many near-zero weights and are natural pruning targets. A threshold of `L1 < 0.01` is a reasonable starting point.

### 2. Activation Statistics

Instrument hooks to capture mean absolute activation values:

```python
hooks = []
activations = {}

def make_hook(name):
    def hook(module, inp, out):
        activations[name] = out.detach().abs().mean().item()
    return hook

for name, module in model.named_modules():
    if isinstance(module, nn.ReLU):
        hooks.append(module.register_forward_hook(make_hook(name)))

with torch.no_grad():
    model(calibration_input)

for h in hooks:
    h.remove()
```

Layers whose post-ReLU activations are near-zero contribute little to the output and can be pruned aggressively.

### 3. Gradient-Based Saliency (Requires Training Data)

The Taylor expansion criterion scores each weight by `|weight × gradient|`. This is more accurate than magnitude alone but requires a forward+backward pass on calibration data.

```python
# Forward pass
loss = criterion(model(inputs), targets)
loss.backward()

for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d) and module.weight.grad is not None:
        saliency = (module.weight * module.weight.grad).abs().mean()
        print(f"{name}: saliency={saliency:.4f}")
```

---

## Iterative Pruning-and-Fine-Tuning

One-shot pruning to high sparsity causes large accuracy loss. **Iterative pruning** applies pruning in small steps with fine-tuning between each step:

```
Start: FP32 trained model (baseline PSNR)
│
├─► Prune 10% → Fine-tune 1 epoch → Evaluate PSNR
├─► Prune 10% more (20% total) → Fine-tune 1 epoch → Evaluate PSNR
├─► ...
└─► Prune to target (e.g. 50%) → Fine-tune 3 epochs → Final PSNR
```

### Recommended Schedule

| Total sparsity | Prune steps | Fine-tune epochs/step | Total epochs |
|---|---|---|---|
| 30% | 3 × 10% | 1 | 3 |
| 50% | 5 × 10% | 2 | 10 |
| 70% | 7 × 10% | 3 | 21 |

Fine-tuning at a learning rate 10× lower than original training (e.g., 1e-5 if original was 1e-4) avoids overwriting learned features.

> [!NOTE]
> The `prune_model.py` script applies one-shot pruning (for research/benchmarking). Iterative pruning requires integration with `scripts/restoration/train.py` and is the recommended production path.

---

## Model-Specific Notes

### 1. Attention U-Net

- **Sensitive layers:** First encoder conv (3→64), skip-connection attention gates (channel attention linear layers), last decoder conv (64→3).
- **Safe to prune:** Middle encoder/decoder convolutions (256→256, 512→512 blocks). These have the most parameters and highest redundancy.
- **Recommended sparsity:** 40–50% global unstructured; 20–30% structured (filter pruning on middle blocks).
- **Risk:** Attention gate Linear layers (SE blocks) have few parameters; prune with care or exclude.

### 2. Restormer

- **Sensitive layers:** DWConv3×3 in MDTA (directly computes spatial features), final projection conv.
- **Safe to prune:** Feed-forward expansion Linear layers in GDFN (4× expansion ratio has high redundancy). Sparsity 30–40% on GDFN FFN has negligible quality impact.
- **Recommended sparsity:** 30% global; exclude first and last conv.
- **Risk:** LayerNorm affine parameters (weight, bias) must NOT be pruned.

### 3. TransWeather

- **Sensitive layers:** Weather query cross-attention K, V projection (query tokens carry task-specific info).
- **Safe to prune:** Encoder conv layers at deeper levels (high channel count), FFN expansion layers.
- **Recommended sparsity:** 35% global unstructured.
- **Risk:** Learnable weather embeddings should be excluded from pruning.

### 4. PromptIR

- **Sensitive layers:** Prompt bank embeddings (discrete learned tokens), final synthesis conv.
- **Safe to prune:** Shared encoder conv layers (first half of encoder is task-agnostic).
- **Recommended sparsity:** 25% — PromptIR is most sensitive due to the multi-task nature; aggressive pruning breaks cross-weather generalisation.
- **Risk:** If pruning prompt-related Linear layers, validate on all weather types (rain, haze, snow).

### 5. RAMiT

- **Sensitive layers:** SE channel attention Linear layers (already few params), final reconstruction conv.
- **Safe to prune:** Intermediate spatial attention conv layers.
- **Recommended sparsity:** 50% — RAMiT's redundancy is high despite small size. Pruning to 50% retains most quality.
- **Risk:** Lowest parameter count; be cautious with local pruning (equal sparsity forces SE layers to zero which collapses channel attention).

### 6. LiteWeatherFormer (Novel Model)

- **Sensitive layers:** Conv Stem (learns weather-agnostic features), weather prompt cross-attention Q projection, skip-fusion conv (critical for multi-scale feature merging).
- **Safe to prune:** Depthwise encoder inner blocks, pointwise expansion channels.
- **Recommended sparsity:** 40% global — depthwise convs have naturally sparse weight distributions.
- **Risk:** Skip-fusion decoder uses element-wise addition of encoder/decoder features; pruning must be symmetric or use global mode to avoid shape mismatch.

---

## Recovery Fine-Tuning

After applying pruning (and making it permanent with `prune.remove()`), fine-tuning on the original training data recovers most of the accuracy loss.

### Fine-Tuning Protocol

```python
# 1. Load pruned model (weights already zeroed and masks removed)
model = torch.load('model_pruned.pt')
model.train()

# 2. Use a lower learning rate
optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-5,
    betas=(0.9, 0.999),
)

# 3. Use the same loss as training (e.g., Charbonnier + perceptual)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

# 4. Fine-tune
for epoch in range(5):
    for batch in dataloader:
        pred = model(batch['input'].cuda())
        loss = criterion(pred, batch['target'].cuda())
        optimizer.zero_grad()
        loss.backward()
        # Gradient masking: zeroed weights should stay zero
        # (no explicit mask needed after prune.remove(); gradients will update zeros)
        optimizer.step()
    scheduler.step()
```

> [!WARNING]
> After `prune.remove()`, the zero weights will receive gradients and may become non-zero again during fine-tuning. To maintain sparsity, re-apply pruning masks after each fine-tuning step (iterative approach) or accept slight sparsity reduction.

### Expected Recovery

| Sparsity | PSNR drop (raw) | PSNR drop (after 5-epoch FT) |
|---|---|---|
| 30% | 0.3–0.8 dB | 0.05–0.2 dB |
| 50% | 1.0–2.5 dB | 0.2–0.6 dB |
| 70% | 3.0–6.0 dB | 0.8–2.0 dB |

*(Representative estimates; exact values depend on model family and calibration quality.)*

---

## Evaluation Protocol

All pruning experiments must be evaluated consistently:

### Metrics

1. **PSNR** (dB) — primary quality metric. Computed on the clean/rain validation set (1000 pairs from RealRain-1k or synthetic rain).
2. **SSIM** — structural similarity; secondary quality metric.
3. **Latency** (ms) — median of 50 timed runs at 512×512 input, after 10 warm-up runs.
4. **Speedup ratio** — `latency_baseline / latency_pruned` (for unstructured, may be ≈ 1.0 on GPU).
5. **Model size** (MB) — `torch.save` to BytesIO buffer.
6. **Sparsity per layer** — fraction of zero weights post-pruning.

### Baseline

All comparisons use the FP32 un-pruned model evaluated on the validation set using `scripts/restoration/evaluate.py`. The PSNR reference for quantization experiments is the FP32 output (not GT) to isolate the quantization effect.

### Reporting Template

| Model | Sparsity | PSNR (base) | PSNR (pruned) | PSNR (FT) | PSNR drop (FT) | Latency (ms) | Speedup |
|---|---|---|---|---|---|---|---|
| Attention U-Net | 0% | 32.1 | — | — | — | 35 | 1.0× |
| Attention U-Net | 30% | 32.1 | 31.5 | 32.0 | 0.1 dB | 34 | 1.03× |
| Attention U-Net | 50% | 32.1 | 30.8 | 31.8 | 0.3 dB | 33 | 1.06× |
| … | … | … | … | … | … | … | … |

> [!NOTE]
> GPU speedup for unstructured pruning is usually < 1.1× because GPU kernels execute on dense tensors regardless of sparsity. Real speedup from unstructured pruning requires NVIDIA's Ampere 2:4 sparse Tensor Core feature. CPU speedup is 1.3–2× at 50%+ sparsity when using sparse INT8 runtimes (e.g., ARM Kleidi or OpenBLAS sparse).

---

## References

1. Han et al., *Learning both Weights and Connections for Efficient Neural Networks*, NeurIPS 2015.
2. Molchanov et al., *Pruning Convolutional Neural Networks for Resource Efficient Inference*, ICLR 2017.
3. Frankle & Carlin, *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks*, ICLR 2019.
4. PyTorch Pruning Tutorial — https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
5. NVIDIA Sparse Tensor Core — Ampere Architecture Whitepaper, 2020.
6. Zamir et al., *Restormer*, CVPR 2022.
7. Potlapalli et al., *PromptIR*, NeurIPS 2023.
