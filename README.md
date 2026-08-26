# Aegis Perception

Weather-robust road-object perception for ADAS, developed under the TiHAN–IIT
Hyderabad project **Design and Development of Efficient Road Object Detection
Methods in Extreme Weather Conditions for Secure Autonomous Driving**.

The current 13-week scope is deliberately narrow:

1. reproduce the attention-gated Pix2Pix smoke baseline on a held-out split;
2. extend the same controlled baseline to rain and fog/snow;
3. measure whether restoration improves object detection, not only PSNR/SSIM;
4. export the restoration + YOLOv8n/s pipeline and benchmark FP32/FP16/INT8 on
   Jetson-class hardware.

Diffusion restoration, foundation-model detectors, domain adaptation, and
adversarial defence remain future work under the revised execution plan.

## Repository map

```text
configs/                 Versioned experiment configurations
data/                    Local dataset contracts and split manifests (no raw data in Git)
docs/                    Audit, literature review, protocols, and experiment roadmap
intern-work/             Historical internship reports (source evidence)
scripts/                 Dataset, training, evaluation, and export entry points
sem5-works/              Original proposal and semester-5 reports
src/aegis_perception/    Reusable Python package
tests/                   Lightweight tests
baseline.ipynb           Historical Colab smoke notebook; retained unchanged
```

Start with [docs/PROJECT_AUDIT.md](docs/PROJECT_AUDIT.md), then
[docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md). Dataset placement and the
important DAWN evaluation limitation are in [data/README.md](data/README.md).

## Setup

Python 3.10+ and a CUDA-capable PyTorch installation are recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[detection,metrics,dev]'
```

## Reproduce before extending

Create a leakage-safe manifest from paired smoke images:

```bash
python scripts/build_pair_manifest.py \
  --input-root data/raw/smoke/hazy \
  --target-root data/raw/smoke/clean \
  --output data/manifests/smoke.csv \
  --weather smoke

python scripts/audit_manifest.py data/manifests/smoke.csv
python scripts/train_restoration.py --config configs/restoration/smoke_attention.yaml
python scripts/evaluate_restoration.py \
  --config configs/restoration/smoke_attention.yaml \
  --checkpoint artifacts/restoration/smoke_attention_seed42/best.pt \
  --split test
```

Run the attention ablation by changing only `model.attention` and the run name,
or use `configs/restoration/smoke_vanilla.yaml`. Do not use the historical
notebook's metrics as a held-out baseline: it evaluates the training loader.

## Detection

Evaluate a detector on one prepared Ultralytics dataset YAML:

```bash
python scripts/evaluate_detection.py \
  --model artifacts/detection/yolov8n/best.pt \
  --data configs/detection/dawn.yaml \
  --condition dawn_raw
```

Use `scripts/evaluate_detection_matrix.py` for the clean/degraded/restored
matrix. That three-way comparison requires the same labelled scenes in all
three conditions; DAWN supports only raw-adverse vs restored because it has no
matched clean images.

## Reproducibility rules

- Every reported number must identify the config, Git commit, split manifest,
  checkpoint, dataset version, seed, and hardware.
- Keep the test split untouched until model choices are frozen.
- Report per-weather and per-class AP, not only a pooled score.
- Pair PSNR/SSIM with downstream mAP and failure cases.
- Report end-to-end latency, batch size 1, warm-up policy, and power mode.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the complete run
contract.
