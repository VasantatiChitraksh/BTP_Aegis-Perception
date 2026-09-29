# Aegis Perception

Weather-robust road-object perception for the TiHAN–IIT Hyderabad ADAS project.
The pipeline restores smoke, rain, fog, or snow images with attention/vanilla
Pix2Pix, evaluates the effect on YOLO detection, and exports restoration models
for edge benchmarking.

## Layout

```text
src/aegis_perception/   Models, training, data loading, metrics, and checkpoint utilities
scripts/data/          Download, extraction, manifest creation, and data audits
scripts/restoration/   Train, evaluate, and restore image folders
scripts/detection/     Train and evaluate YOLO
scripts/deployment/    Export, validate, and benchmark ONNX
configs/               Restoration experiments and detection dataset definitions
data/datasets.yaml     Dataset sources, checksums, access notes, and licenses
data/manifests/        Versioned CSV pairs and train/validation/test splits
data/archives/         Downloaded archives (ignored by Git)
data/raw/              Extracted originals (ignored by Git)
docs/                  Dataset guide, experiment protocol, and literature review
docs/reports/          Execution plan, meeting brief, and simulator selection
archive/               Historical notebook and internship/semester reports
tests/                 Automated checks
```

Generated models, metrics, and exports belong in `artifacts/`; derived images
and detector datasets belong in `data/processed/`. Both are ignored by Git.
Run the commands below from the repository root.

## Setup

Python 3.10+; install a suitable PyTorch build for your CPU/GPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[detection,metrics,dev]'
# For ONNX tools:
python -m pip install -e '.[export]'
```

## Smoke baseline

The existing smoke split has four clean targets shared between train and
validation; see [the dataset guide](docs/datasets.md) before training.

```bash
python scripts/data/download_datasets.py --dataset smoke_historical
python scripts/data/extract_datasets.py --extract --dataset smoke_historical
# Build only when establishing a split; preserve existing splits for comparisons.
python scripts/data/build_dataset_manifests.py --dataset smoke_historical
python scripts/data/audit_manifest.py data/manifests/smoke_historical.csv
python scripts/restoration/train.py --config configs/restoration/smoke_attention.yaml
python scripts/restoration/evaluate.py \
  --config configs/restoration/smoke_attention.yaml \
  --checkpoint artifacts/restoration/smoke_attention_seed42/best.pt --split test
```

Use `smoke_vanilla.yaml` and `smoke_attention_l1.yaml` for the attention and loss
ablations. Rain and snow configs use collected training manifests. The fog
config is a template: supply a paired fog training manifest first; O-HAZE is
reserved for testing.

## Detection and deployment

Prepare the labels and splits described in [the dataset guide](docs/datasets.md)
before using `configs/detection/dawn.yaml`. Evaluate one view or repeat `--view`
to compare several views with the same detector:

```bash
python scripts/detection/evaluate.py \
  --model artifacts/detection/training/yolov8n/weights/best.pt \
  --view dawn_raw=configs/detection/dawn.yaml

python scripts/deployment/export_onnx.py \
  --checkpoint artifacts/restoration/smoke_attention_seed42/best.pt \
  --output artifacts/restoration/smoke_attention.onnx
python scripts/deployment/validate_onnx.py \
  --checkpoint artifacts/restoration/smoke_attention_seed42/best.pt \
  --onnx artifacts/restoration/smoke_attention.onnx
python scripts/deployment/benchmark_onnx.py --model artifacts/restoration/smoke_attention.onnx
```

Every script supports `--help`. Detection evaluation writes `metrics.json` and
Ultralytics output under the selected output directory. ONNX benchmarking measures
model latency; TensorRT conversion and full Jetson pipeline timing remain future work.

## Project documents

- [Datasets and preparation](docs/datasets.md)
- [Experiments and reproducibility](docs/experiments.md)
- [Literature review](docs/literature_review.md)
- [Execution plan v2](docs/reports/execution_plan_v2.pdf)
- [Professor meeting brief](docs/reports/professor_meeting_brief.pdf)
- [Simulator selection](docs/reports/simulator_selection.docx)

Historical evidence lives in [the smoke notebook](archive/notebooks/smoke_baseline.ipynb),
[semester reports](archive/reports/semester_5/), and
[internship reports](archive/reports/internship/). The notebook evaluates its
training loader; its saved metrics are not a held-out benchmark. Original
notebook outputs and report contents are preserved.

## Checks

```bash
python -m pytest
ruff check .
```
