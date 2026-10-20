# `quantization/` — Model Compression Research

This folder contains all research documents, benchmark scripts, and report generators
for **quantization** and **pruning** of the Aegis Perception restoration models, targeting:

- **Primary:** NVIDIA Jetson Orin Nano (512-core Ampere GPU, 16 Tensor Cores)
- **Secondary:** Raspberry Pi 5 (Broadcom BCM2712, Cortex-A76, CPU only)

---

## Folder Structure

```
quantization/
├── QUANTIZATION.md          ← Research doc: PTQ, QAT, TensorRT, decision matrix
├── PRUNING.md               ← Research doc: structured/unstructured, iterative strategy
├── JETSON_ROVER_GUIDE.md    ← Hardware-specific deployment guide for JetAuto Pro car & Jetson
├── README.md                ← This file
└── scripts/
    ├── quantize_compare.py          ← Benchmark FP32/FP16/INT8 for one model
    ├── prune_model.py               ← Apply L1 pruning + measure PSNR/latency
    ├── export_tensorrt.py           ← Export ONNX to FP16/INT8 TensorRT engine
    ├── rover_infer_trt.py           ← Real-time TRT camera/image inference for rover
    ├── batch_quantize.sh            ← SLURM: quantize all 5 models
    ├── batch_prune.sh               ← SLURM: prune all 5 models at 30/50/70 %
    ├── generate_quantization_report.py  ← Aggregate JSON → Markdown + PDF
    └── generate_pruning_report.py       ← Aggregate JSON → Markdown + PDF
```

---

## Requirements

```bash
pip install fpdf2          # PDF generation
# All other deps (torch, PIL, numpy) are already in the project environment.
```

---

## Quick Start

### 1. Benchmark quantization modes for a single model

```bash
python quantization/scripts/quantize_compare.py \
    --checkpoint checkpoints/liteweatherformer/best.pt \
    --image data/raw/rain/realrain1k/rain/0001.jpg \
    --output-dir artifacts/quantization/liteweatherformer
```

**Outputs:**
- Console: formatted table of latency, size, PSNR for FP32 / FP16 / INT8-Dynamic / INT8-Static
- File: `artifacts/quantization/liteweatherformer/quantization_report.json`

**Raspberry Pi 5 simulation (CPU):**

```bash
python quantization/scripts/quantize_compare.py \
    --checkpoint checkpoints/ramit/best.pt \
    --image path/to/test_rain.jpg \
    --output-dir artifacts/quantization/ramit \
    --cpu
```

---

### 2. Run pruning on a single model

```bash
python quantization/scripts/prune_model.py \
    --checkpoint checkpoints/attention_unet/best.pt \
    --sparsity 0.5 \
    --method global \
    --image data/raw/rain/realrain1k/rain/0001.jpg \
    --output-dir artifacts/quantization/pruning/attention_unet
```

**Outputs:**
- Console: per-layer sparsity table
- File: `artifacts/quantization/pruning/attention_unet/attention_unet_pruned_s050.pt`
- File: `artifacts/quantization/pruning/attention_unet/attention_unet_pruned_s050_report.json`

---

### 3. Run all models on SLURM (DGX A100)

> Edit the checkpoint paths at the top of each script before submitting.

```bash
# Quantization sweep
sbatch quantization/scripts/batch_quantize.sh

# Pruning sweep (30%, 50%, 70% sparsity × 5 models)
sbatch quantization/scripts/batch_prune.sh
```

---

### 4. Generate aggregate reports

```bash
# After batch_quantize.sh completes:
python quantization/scripts/generate_quantization_report.py \
    --results-dir artifacts/quantization \
    --output-dir  artifacts/quantization

# After batch_prune.sh completes:
python quantization/scripts/generate_pruning_report.py \
    --results-dir artifacts/quantization/pruning \
    --output-dir  artifacts/quantization
```

**Outputs:**
- `artifacts/quantization/QUANTIZATION_RESULTS.md`
- `artifacts/quantization/quantization_report.pdf`
- `artifacts/quantization/PRUNING_RESULTS.md`
- `artifacts/quantization/pruning_report.pdf`

---

## Important Notes

### PSNR interpretation

All PSNR values in the quantization/pruning scripts measure **error relative to the
FP32 baseline output on a single test image** — not relative to clean ground truth.
This isolates the degradation introduced by quantization or pruning.

To measure PSNR / SSIM vs clean GT, run:

```bash
python scripts/restoration/evaluate.py --config <config.yaml> --checkpoint <path>
```

### Single-image calibration (INT8 Static)

`quantize_compare.py` uses the single input image for INT8 static calibration. This is
sufficient only for benchmarking relative quantization error. For production deployment,
calibrate with ≥ 100 representative rain/degraded images from the training set.

### FP16 on CPU

FP16 is not hardware-accelerated on Raspberry Pi 5 (Cortex-A76 lacks FP16 dot-product
instructions). Always use INT8 for CPU-only targets. The `--cpu` flag on
`quantize_compare.py` will emit a warning for FP16.

### Transformer models and INT8 Static

Restormer, TransWeather, and PromptIR contain MultiheadAttention layers whose softmax
is not fully quantizable by PyTorch eager-mode static INT8. These layers fall back to
FP32 inside the quantized graph. Use INT8 Dynamic or TensorRT for these models.

---

## Research Documents

| Document | Contents |
|---|---|
| [`QUANTIZATION.md`](QUANTIZATION.md) | PTQ (FP16/INT8), QAT, TensorRT pipeline, per-model analysis, decision matrix |
| [`PRUNING.md`](PRUNING.md) | Structured vs unstructured, magnitude pruning API, iterative strategy, per-model notes |
