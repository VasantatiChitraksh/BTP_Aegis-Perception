# Aegis Perception: Robust Vision Restoration & Edge Autonomy for ADAS

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Deployment target](https://img.shields.io/badge/Deployment%20target-Jetson%20Orin-blue.svg)](docs/edge_integrated_handoff.txt)

**Aegis Perception** is an edge-optimized perception suite designed for weather-degraded autonomous driving and robotic systems under the **TiHAN–IIT Hyderabad ADAS** initiative. 

It chains adverse-weather image restoration (deraining, dehazing, desnowing, desmoking) into road-object detection (YOLOv8). **Aegis-HAMQ** is a proposed quantization research direction; validated custom INT8 execution and Jetson performance are not yet established.

## Current deployment download

Use the [integrated edge release](https://github.com/VasantatiChitraksh/BTP_Aegis-Perception/releases/tag/edge-integrated-20261030), containing `aegis_edge_integrated_20261030.tar.gz` and its SHA256 file. Model binaries are release assets, so cloning this repository alone does not download them. Extract the archive and read `START_INTEGRATED.txt` first.

The package preserves the trained Restormer + YOLOv8n deployment kit and adds a reviewed adapter for the custom weight quantizer plus synthetic-weather experiments. The custom candidate remains a floating-point simulation: its four-image screen rejected it on numerical agreement for fog and smoke, and its ONNX size did not decrease. Use the baseline for initial deployment. [Measured screen](docs/quantization/restormer_custom_screen_20261030.json) · [Deployment instructions](docs/edge_integrated_handoff.txt) · [Weather-testing workflow](scripts/weather_testing/README.md).

The older `edge_pipeline/` examples and hardware plots below are experimental material, not evidence of validated hardware speed or accuracy. Build TensorRT engines on the target Jetson, retain validation and layer reports, and measure latency, memory and power before making deployment claims.

---

## 1. Repository Layout

The repository is organized into distinct, modular subsystems:

```text
BTP_Aegis-Perception/
├── src/aegis_perception/         # Core research library
│   ├── models/                   # Restoration architectures (Restormer, LiteWeatherFormer, etc.)
│   ├── data/                     # Dataset loaders, manifests, and continuous stratified sampling
│   ├── edge/                     # TensorRT runtime and detector utilities
│   ├── checkpoints.py            # Model checkpoint serializer/deserializer
│   ├── training.py               # Distributed training engine
│   └── evaluation.py             # Validation and PSNR/SSIM evaluation guards
│
├── edge_pipeline/                # Turnkey Edge Deployment Package
│   ├── import_restormer/         # Validates and cleans best.pt (strips DDP, checks 256x448)
│   ├── yolov8_loader/            # Loads PyTorch/ONNX/TRT YOLOv8 road detector
│   ├── quantization_scripts/     # Aegis-HAMQ profiler, custom quantizer, and TRT exporter
│   ├── final_pipeline/           # Chained Restormer -> YOLOv8 perception pipeline
│   ├── checkpoints/              # Local weights, calibration tables, and metadata
│   └── tests/                    # End-to-end integration test suite
│
├── scripts/                      # Reproducible operational scripts
│   ├── data/                     # Dataset download, extraction, and manifest builders
│   ├── restoration/              # Multi-weather model training and evaluation
│   ├── detection/                # YOLO dataset training and benchmarking
│   ├── deployment/               # ONNX export, screening, and Jetson runbook generation
│   └── quantization/             # SLURM batch pruning and quantization experiments
│
├── configs/                      # Experiment and model YAML configurations
│   ├── restoration/              # Restoration model configs (Restormer, LiteWeatherFormer, etc.)
│   ├── detection/                # YOLO detection configs (DAWN dataset)
│   └── deployment/               # Edge dependency requirement definitions
│
├── data/                         # Data catalogs and splits
│   ├── manifests/                # Versioned train/val/test CSV pairs (clean, weather_v3)
│   └── datasets.yaml             # Dataset sources, checksums, and license index
│
├── docs/                         # Technical documentation & guides
│   ├── quantization/             # Quantization, pruning, and Jetson rover runbooks
│   ├── datasets.md               # Dataset index and download guide
│   ├── experiments.md            # Multi-weather experimental protocol
│   ├── literature_review.md      # Literature review and baselines
│   └── weather_protocol_v3.md    # Multi-weather stratified data split protocol
│
├── artifacts/                    # Published benchmark reports and artifacts
│   ├── edge_v1/reports/          # Edge deployment verification and readiness reports
│   └── quantization/reports/     # Aegis-HAMQ research proposal, Pareto graphs, and plans
│
├── notebooks/                    # Interactive Colab / Lightning AI restoration control notebook
└── tests/                        # Automated unit and integration test suite
```

---

## 2. Supported Restoration Models

All generators are registered in `src/aegis_perception/models/` and can be instantiated via `build_generator(config['model'])`:

| Architecture | Key Mechanism | Best Edge Use-Case |
|---|---|---|
| **Restormer** | Multi-Dconv Head Transposed Attention (MDTA) + Gated Feed-Forward (GDFN) | High-fidelity adverse weather restoration (Baseline Champion). |
| **LiteWeatherFormer** | Novel lightweight Conv-Stem + Depthwise Transformer + Weather Prompt Fusion | Ultra-fast mobile edge inference ($<3.5\text{M}$ params). |
| **TransWeather** | Transformer encoder-decoder with weather-type query embeddings | Multi-weather joint restoration. |
| **PromptIR** | Dynamic prompt learning for all-in-one restoration | Adaptive degradation compensation. |
| **RAMiT** | Lightweight recurrent attention and multi-scale aggregation | Resource-constrained micro-rovers. |
| **Attention Pix2Pix** | U-Net with additive spatial attention gates | Classical paired image-to-image baseline. |

---

## 3. Aegis-HAMQ: Hardware-Aware Quantization

Traditional uniform quantization (e.g. uniform INT8) causes **cross-covariance attention collapse** in Restormer, while running INT8 on hardware lacking native integer units (such as Maxwell `sm_53` on Jetson Nano) makes the model **$2.1\times$ slower**.

**Aegis-HAMQ** solves this by mapping Restormer's functional layers to native physical ALUs:
- **Stem & Reconstruction:** High sensitivity ($\mathcal{S} \ge 0.95$) $\to$ **FP16 / FP32 preserved**.
- **MDTA Transposed Attention:** Channel covariance matrix ($\mathcal{S} \approx 0.78 - 0.92$) $\to$ **FP16 / FP8 preserved**.
- **GDFN Feed-Forward:** **65.5% of total MACs**, low sensitivity ($\mathcal{S} \le 0.28$) $\to$ **INT8 / INT4 accelerated**.

### Hardware Allocation Matrix:

| Target Platform | Silicon & Compute Units | Aegis-HAMQ Allocation Plan | Peak Throughput ($256\times448$) |
|---|---|---|---|
| **NVIDIA Jetson Orin Nano** | 512 Ampere Cores + 16 Tensor Cores (`sm_87`) | **31.7% FP16** (MDTA) + **68.3% INT8/INT4** (GDFN) | **$38.2\text{--}52.0\text{ FPS}$** ($52.8\text{ dB}$ PSNR) |
| **JetAuto Pro AI Robot Car** | 128 Maxwell Cores (`sm_53`, no Tensor Cores) | **100% Native TensorRT FP16** (Zero INT8 penalty) | **$4.3\text{ FPS}$** ($<0.05\text{ dB}$ drop) |
| **Raspberry Pi 5** | Quad-Core ARM Cortex-A76 @ 2.4 GHz | **0.7% FP32** (Recon) + **99.3% NEON INT8** (`SDOT`/`UDOT`) | **$2.8\text{ FPS}$** ($4.0\times$ memory reduction) |

---

## 4. Quickstart Guide

### 4.1 Installation
Python 3.10+ required. Create a clean virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .
```

To run YOLO detection or export to ONNX / TensorRT:
```bash
pip install -e ".[detection,export,metrics,dev]"
```

---

### 4.2 Edge Pipeline Workflow

#### Step 1: Import & Validate Your Restormer Checkpoint
Validates state dicts, strips DistributedDataParallel (`module.`) wrappers, verifies a $256\times448$ forward pass:
```bash
python edge_pipeline/import_restormer/import_restormer.py \
    --checkpoint path/to/your/best.pt \
    --output edge_pipeline/checkpoints/restormer/restormer_imported.pt
```

*(To test without a trained checkpoint, use `--create-template` to instantiate and export a verified architectural checkpoint).*

#### Step 2: Auto-Detect Hardware & Profile Mixed Precision
Probe host silicon capabilities and generate an optimal precision plan:
```bash
# Auto-detects whether running on Jetson Orin, Jetson Nano, RPi, or x86
python edge_pipeline/quantization_scripts/hardware_aware_profiler.py \
    --hardware auto \
    --lambda-extent 0.5 \
    --output artifacts/quantization/reports/hardware_precision_plan.json
```

#### Step 3: Run Quantization Benchmark
Quantize weights using your calibrated profile and evaluate retention metrics:
```bash
python edge_pipeline/quantization_scripts/edge_quantize.py \
    --checkpoint edge_pipeline/checkpoints/restormer/restormer_imported.pt \
    --image data/test_rain_sample.jpg \
    --output-dir edge_pipeline/checkpoints/restormer/quantized
```
*Sample Output:*
- **FP32 Model Size:** 99.55 MB $\to$ **Quantized:** 24.89 MB (**$4.00\times$ compression**)
- **PSNR Retention:** **$53.38\text{ dB}$** | **SSIM Retention:** **$0.9998$**

#### Step 4: Run End-to-End Perception Pipeline
Chains adverse-weather restoration directly into YOLOv8 road-object detection:
```bash
# Process a single test image
python edge_pipeline/final_pipeline/run_inference.py \
    --checkpoint edge_pipeline/checkpoints/restormer/restormer_imported.pt \
    --image data/test_rain_sample.jpg \
    --output-dir artifacts/edge_v1/pipeline_outputs

# Run live on JetAuto Pro rover camera (USB/CSI)
python edge_pipeline/final_pipeline/run_inference.py \
    --checkpoint edge_pipeline/checkpoints/restormer/restormer_imported.pt \
    --camera 0
```
Generates side-by-side comparison images and detection latency telemetry:
- Split visualization: [`artifacts/edge_v1/pipeline_outputs/perception_test_rain_sample.jpg`](artifacts/edge_v1/pipeline_outputs/perception_test_rain_sample.jpg)
- Telemetry metrics: [`artifacts/edge_v1/pipeline_outputs/perception_test_rain_sample.json`](artifacts/edge_v1/pipeline_outputs/perception_test_rain_sample.json)

---

### 4.3 Compiling Native TensorRT Engines

On NVIDIA Jetson targets, compile static TensorRT engines for maximum GPU throughput:

```bash
# On Jetson Orin Nano (Mixed FP16 + Tensor Core INT8)
python edge_pipeline/quantization_scripts/export_trt_edge.py \
    --checkpoint edge_pipeline/checkpoints/restormer/restormer_imported.pt \
    --output-engine edge_pipeline/checkpoints/restormer/restormer_hybrid_orin.engine \
    --precision int8 \
    --calib-image data/test_rain_sample.jpg

# On JetAuto Pro (Pure FP16 for Maxwell sm_53)
python edge_pipeline/quantization_scripts/export_trt_edge.py \
    --checkpoint edge_pipeline/checkpoints/restormer/restormer_imported.pt \
    --output-engine edge_pipeline/checkpoints/restormer/restormer_fp16_nano.engine \
    --precision fp16
```

---

## 5. Key Research Reports & Benchmark Catalog

| Report | Path | Key Highlights |
|---|---|---|
| **Novelty & Approach Report** | [`artifacts/quantization/reports/NOVELTY_AND_APPROACH.md`](artifacts/quantization/reports/NOVELTY_AND_APPROACH.md) | Full theoretical formulation of Aegis-HAMQ, sensitivity metrics, and Knapsack optimization. |
| **Formal Research PDF** | [`artifacts/quantization/reports/hardware_aware_hybrid_quantization_novelty.pdf`](artifacts/quantization/reports/hardware_aware_hybrid_quantization_novelty.pdf) | Publication-grade report embedding the hardware Pareto frontier graph and execution matrices. |
| **Pareto Frontier Graph** | [`artifacts/quantization/reports/hardware_vs_quantization_extent.png`](artifacts/quantization/reports/hardware_vs_quantization_extent.png) | 2-panel chart showing layer sensitivity vs hardware throughput across $\lambda \in [0, 1]$. |
| **Jetson Hardware Guide** | [`artifacts/edge_v1/reports/aegis_jetson_step_by_step.pdf`](artifacts/edge_v1/reports/aegis_jetson_step_by_step.pdf) | Terminal execution runbook for Jetson Nano and Orin deployment. |
| **Edge Readiness Report** | [`artifacts/edge_v1/reports/restormer_edge_readiness.pdf`](artifacts/edge_v1/reports/restormer_edge_readiness.pdf) | Validation of $256\times448$ tile geometry, memory footprint, and YOLOv8 chaining. |
| **Reports Master Index** | [`artifacts/edge_v1/reports/README_REPORTS.md`](artifacts/edge_v1/reports/README_REPORTS.md) | Complete index of all 11 edge deployment benchmark artifacts. |

---

## 6. Testing & Quality Verification

Run the automated test suite:
```bash
pytest
```
Run only edge pipeline integration tests:
```bash
pytest edge_pipeline/tests/test_edge_pipeline.py
```
*(All integration tests run cache-free and verify checkpoint validation, custom quantization, letterboxing, and end-to-end perception).*

---

## 7. License & Attribution

This project is licensed under the [MIT License](LICENSE).
Restoration architectures adapted from [swz30/Restormer](https://github.com/swz30/Restormer) and related baseline repositories. Road object detection powered by [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).
