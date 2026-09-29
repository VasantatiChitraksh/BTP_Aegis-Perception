# Datasets

[data/datasets.yaml](../data/datasets.yaml) is the single catalog for source URLs,
archive names, expected sizes, SHA-256 values, access status, and license notes.
It also records optional/gated alternatives; they are excluded from automatic
core downloads. A code license does not establish dataset redistribution rights.

## Storage and commands

Archives go in `data/archives/`, extracted originals in `data/raw/`, generated
images/labels in `data/processed/`, and versioned pair lists in `data/manifests/`.
Keep the release's internal filenames and layout intact: the dataset-specific
manifest builders depend on them. Large data and generated images stay out of Git.
Run commands from the repository root:

```bash
python scripts/data/download_datasets.py --list
python scripts/data/download_datasets.py --core
# Or select --condition smoke|rain|fog|snow or repeat --dataset ID.
python scripts/data/extract_datasets.py --verify
python scripts/data/extract_datasets.py --extract
python scripts/data/build_dataset_manifests.py --dataset smoke_historical
python scripts/data/audit_manifest.py data/manifests/smoke_historical.csv
```

Existing archives are verified before reuse. `aria2c` supports resumable downloads;
without it, downloads use Python's standard library and restart partial `.part`
files. Extraction checks archive paths and uses a staging directory before
publishing the result. CSD's RAR5 archive needs a current `7zz`/7-Zip release;
legacy `p7zip 16.02` cannot extract it.

## Collected sources and split policy

| Dataset ID | Role | Manifest / limitation |
|---|---|---|
| `smoke_historical` | ACCV 2022 smoke reproduction; 110 training and 12 test pairs | `smoke_historical.csv`; hold validation out of original training |
| `smokebench` | Paired surveillance smoke; 9,875 training and 100 test pairs | `smokebench.csv`; validation from training; no road-object boxes |
| `rain100l` | Small paired synthetic rain benchmark | `rain100l.csv`; collected 100 pairs are test-only; public mirror provenance recorded in catalog |
| `realrain1k` | Paired rain training and evaluation | `realrain1k.csv`; preserve high-resolution release's train/validation/test; omit duplicate low-resolution copy |
| `o_haze` | 45 real outdoor haze pairs | `o_haze.csv`; test-only; not a fog training set |
| `rtts` | 4,322 real hazy traffic images with VOC boxes | No clean targets; prepare detector labels and splits separately |
| `foggy_driving` | 101 real foggy road images and annotations | No clean targets; verify task and class mapping |
| `csd` | Synthetic snow; 8,000 training and 2,000 test pairs plus masks | `csd.csv`; validation from training |
| `snow_cityscapes` | Synthetic road snow | `snow_cityscapes.csv`; group snow variants by clean scene; inherits Cityscapes terms |
| `dawn_fog`, `dawn_rain`, `dawn_snow` | Real-road detector evaluation; 300/200/204 images | No clean counterparts; prepare fixed splits and labels separately |

Dataset-specific builders preserve official test sets and use deterministic
scene-based validation splits where needed. Build manifests only when setting
up a dataset; keep them frozen for comparisons. Changing a training seed does
not mean changing the dataset split.

The cleanup audit found four identical clean targets shared between train and
validation in `smoke_historical.csv` (scene IDs 23–25, 26–28, 79–82, and 95–96).
The existing manifest is preserved; group these repeated targets before using
this split for a reported experiment. The audit command correctly rejects it.

Fog training still needs a suitable paired source and `data/manifests/fog.csv`.
RESIDE ITS, Haze4K, and Foggy Cityscapes are cataloged options with access
requirements. Do not turn the O-HAZE test set into training just to make the
fog configuration runnable.

Smoke boxes/masks usually label the smoke plume, not cars or people behind it.
The collected smoke sources support restoration metrics, but do not provide the
road-object ground truth needed to establish a smoke detection benefit.

## Custom paired data

```bash
python scripts/data/build_pair_manifest.py \
  --input-root data/raw/custom/degraded --target-root data/raw/custom/clean \
  --output data/manifests/custom.csv --weather smoke --source custom_v1
python scripts/data/audit_manifest.py data/manifests/custom.csv
```

Pairing uses relative paths without extensions. For several variants of the
same scene, use `--group-regex` with a first capture group identifying that
scene. CSV columns are `sample_id,input_path,target_path,split,weather,source`;
paths are relative to the CSV by default, and splits are `train`, `val`, `test`.
The audit checks missing/corrupt images, duplicate IDs, invalid splits, and exact
clean-target duplicates across splits. Manually inspect sequence/scene leakage too.

## Detection preparation and comparison

`configs/detection/dawn.yaml` is a template, not a ready-to-run dataset. Confirm
the annotation release and map car, bus, truck, person, motorcycle, and bicycle
indices consistently with the detector. Prepare YOLO images/labels and freeze
weather-stratified splits; record class imbalance. The DAWN v3 catalog entry
records CC BY-NC 3.0; sand is outside this project's weather scope.

For DAWN and other unpaired real images, compare the same raw images, restored
images, and a bicubic resize control. Use clean/degraded/restored comparisons
only where clean counterparts and compatible road-object annotations exist.
Restoration preserves image geometry, but labels must accompany each detector
view and PNG output filenames must agree with their label stems.

```bash
python scripts/restoration/restore_images.py \
  --config configs/restoration/smoke_attention.yaml \
  --checkpoint artifacts/restoration/smoke_attention_seed42/best.pt \
  --input-root data/processed/smoke_detection/images \
  --output-root data/processed/smoke_restored/images \
  --control-output-root data/processed/smoke_resize_control/images

# Each YAML must point to the corresponding prepared images and matching labels.
python scripts/detection/evaluate.py --model artifacts/detection/best.pt \
  --view raw=data/processed/smoke_detection/dataset.yaml \
  --view restored=data/processed/smoke_restored/dataset.yaml \
  --view control=data/processed/smoke_resize_control/dataset.yaml
```
