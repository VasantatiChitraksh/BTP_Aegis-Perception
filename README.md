# Aegis Perception

Weather-robust road-object perception for the TiHAN–IIT Hyderabad ADAS project.
The restoration pipeline supports official Restormer and attention/vanilla
Pix2Pix. It trains on paired images, saves resumable experiments, evaluates
held-out fidelity, and exports models for deployment checks. YOLO evaluation
and hardware benchmarking are subsequent experiments.

## Layout

```text
src/aegis_perception/   Models, training, data loading, metrics, and checkpoint utilities
scripts/data/          Download, extraction, manifest creation, and data audits
scripts/restoration/   Train, evaluate, and restore image folders
scripts/detection/     Train and evaluate YOLO
scripts/deployment/    Export, validate, and benchmark ONNX
configs/               Restoration experiments and detection dataset definitions
notebooks/             One restoration control notebook using the same scripts
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
python -m pip install --upgrade pip
python -m pip install -e '.[notebook,metrics,dev,export]'
```

## Restoration workflow

Open [notebooks/restoration.ipynb](notebooks/restoration.ipynb) in the updated
repository on Lightning AI or Colab. Start with `STAGE = "prepare"`: this
downloads/verifies RealRain-1k, audits the frozen manifest, previews image pairs,
and loads official pretrained deraining weights. It does **not** start training.
Then use `sanity`, `train`, `evaluate`, and `export` as separate stages. The
notebook contains no separate trainer or model implementation.

[rain_restormer.yaml](configs/restoration/rain_restormer.yaml) is the initial
fine-tuning recipe: full official architecture, 128-pixel training crops,
original-resolution validation, L1, AdamW, cosine decay, and bf16 on a supported
CUDA GPU. It is a baseline experiment, not a claimed reproduction of the
original training schedule. All legacy Pix2Pix configs still work.

After preparation, the same training can run directly in the terminal:

```bash
python scripts/restoration/train.py --config configs/restoration/rain_restormer.yaml
python scripts/restoration/train.py --config configs/restoration/rain_restormer.yaml \
  --resume artifacts/restoration/rain_restormer_seed42/last.pt
python scripts/restoration/evaluate.py --config configs/restoration/rain_restormer.yaml \
  --checkpoint artifacts/restoration/rain_restormer_seed42/best.pt --split val
```

Fresh training refuses to overwrite existing checkpoints. Resume restores the
optimizer, scheduler, mixed-precision scaler, RNG states, and history from the
last completed epoch. Changed configurations/split manifests require a new run.
Checkpoint selection uses mean per-image RGB validation PSNR; test is used only
after experiment selection. Record native/resized/crop preprocessing when
comparing metrics. ONNX export currently fixes the spatial size; full hardware
and INT8 validation remain later work.

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
python -m pip install -e '.[detection]'
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
