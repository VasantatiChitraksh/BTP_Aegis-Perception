# Datasets

**Current four-weather training protocol:** [frozen v3 split, sampling, statistics and DGX commands](weather_protocol_v3.md). This supersedes the v2 follow-up for new runs; existing results remain historical.


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

The 7 October 2026 audit also rejects `smokebench.csv`: its 9,975 pairs contain
43 distinct clean-target file hashes, all shared between train and validation.
Four target hashes also occur in test; all 100 test pairs share a clean target
with training. Pair count therefore overstates independent background diversity.
Use a documented background-disjoint protocol and independent evaluation before
reporting unseen-scene restoration results; preserve the original release.

Fog/haze training now uses RESIDE-6K from the
[DehazeFormer authors' release](https://github.com/IDKiro/DehazeFormer), with
6,000 training and 1,000 test pairs before repair. The archives and their
SHA-256 values are recorded as `reside6k_train` and `reside6k_test` in the catalog.
This is synthetic indoor/outdoor haze, a fog proxy rather than evidence of
real-driving fog generalization. O-HAZE remains held out.

## Repaired four-weather protocol

The first joint model uses RealRain-1k, RESIDE-6K, SnowCityscapes, and SmokeBench.
Original images and CSVs are preserved. Generate the repaired portable CSVs with:

```bash
python -m pip install -e '.[metrics,dev]'
python scripts/data/prepare_four_weather.py
python scripts/data/audit_manifest.py data/manifests/clean/four_weather.csv
```

Preparation fully decodes inputs and targets, checks paired dimensions, groups
identical decoded RGB targets regardless of file encoding, and conservatively
groups near copies. The near-copy rule requires pHash Hamming distance at most
2, aspect-ratio difference at most 0.01, and 16x16 RGB thumbnail mean absolute
difference at most 2/255. This detects some re-encoded copies; it does not prove
sequence disjointness or perfect alignment.

CSD and SmokeBench receive custom background-disjoint 70/15/15 group splits,
because their original partitions share clean backgrounds across training and
test. Their repaired scores must be labelled as custom-split results, not
compared directly against published original-split numbers. RESIDE-6K retains
its supplied test split and holds out 15% of training-background groups for
validation. Other sources retain their existing splits, with conflicting rows
removed in the order test > validation > training, checked across all sources.
CSD shares many backgrounds with RESIDE-6K's test set, so these exclusions also
substantially reduce CSD's retained data; the final fractions differ from the
initial 70/15/15 assignment. The provided report records the retained counts.
CSD and historical smoke are repaired for later use but are outside the first
joint training mixture.

`data/manifests/clean/four_weather.csv` freezes the joint training/test sets and
a compact validation panel. Each row includes `group_id` for sampling and
background-balanced scoring. Individual repaired CSVs retain the full
validation splits. Counts, exclusions, near-copy matches, and manifest hashes
are saved in `artifacts/data/four_weather/preparation_report.json`; decoded
metadata is cached separately to make repeat audits faster. Changing a training
seed must not regenerate the frozen split.

The completed repair checked 72,724 image files. Final counts for the first model:

| Weather / source | Train pairs | Full validation | Training validation panel | Test pairs |
|---|---:|---:|---:|---:|
| Rain / RealRain-1k | 775 | 112 | 64 | 224 |
| Fog/haze / RESIDE-6K | 5,102 | 897 | 64 | 1,000 |
| Snow / SnowCityscapes | 5,382 | 618 | 128 | 6,000 |
| Smoke / SmokeBench | 7,359 | 1,511 | 48 | 1,105 |

Rain excludes nine training pairs after exact/near-copy checks; RESIDE-6K
excludes one overlapping training pair. SmokeBench has just 31 training, six
validation, and six test background groups despite its large pair count.
Repeated frames cannot be treated as independent scenes in reported results.

Smoke boxes/masks usually label the smoke plume, not cars or people behind it.
The collected smoke sources support restoration metrics, but do not provide the
road-object ground truth needed to establish a smoke detection benefit.

## Follow-up sampling and evaluation: 15 October 2026

The 6,000 SnowCityscapes test pairs came from preserving its supplied test set;
they were not a decision to give snow 72% of the evaluation weight. The original
joint test has 8,329 pairs. Its pooled image average does give snow that weight,
so use the equal-weather, equal-background aggregate as the primary full-test
score. Keep per-weather scores, image counts, background counts, and counts of
images made worse than raw input alongside it.

`scripts/data/prepare_weather_followup.py` freezes two additional artifacts,
without rewriting the original joint manifest or any images:

| Weather | Training pairs (unchanged) | Follow-up validation pairs | Full test (unchanged) | Smaller test panel |
|---|---:|---:|---:|---:|
| Rain | 775 | 64 | 224 | 224 |
| Fog/haze | 5,102 | 64 | 1,000 | 224 |
| Snow | 5,382 | 128 | 6,000 | 224 |
| Smoke | 7,359 | 192 | 1,105 | 224 |
| Total | 18,618 | 448 | 8,329 | 896 |

`four_weather_v2.csv` expands smoke validation from eight to 32 hash-selected
frames per existing held-out background. It includes all previous validation
examples and keeps training/test membership exactly unchanged. This improves
coverage of the six validation backgrounds, not the number of backgrounds.
The original model must be evaluated again on this expanded validation panel;
the follow-up trainer does that before any optimization.

`test_panel_224.json` selects 224 existing test inputs per weather using ID hashes
and a round-robin over backgrounds before taking more variants. It contains no
model scores, records compatible manifest hashes, and is fixed across models.
It covers 224 rain, 224 fog, 224 snow, and six smoke background groups. Equal
image counts do not imply equal scene diversity. It is a secondary subset of an
already inspected test set, not a newly untouched test benchmark. Select models
on validation; use the panel to reduce final inference cost and retain the full
set for a fuller audit. Never move unused test images into training.

The source files were downloaded from existing datasets. No weather images are
generated by these preparation scripts. "Synthetic" describes the dataset
authors' simulated haze/snow effects, not images newly generated in this project.

### Source-data priorities for smoke

SmokeBench has 31/6/6 independent clean-background groups in the repaired
train/validation/test protocol. Its large frame count cannot remedy that narrow
scene diversity. The [authors' release](https://github.com/QianfengY/SmokeBench)
still advertises 9,875 training and 100 test pairs; downloading it again does
not supply new backgrounds. The existing custom background-disjoint split must
be retained instead of reverting to the leaking original split.

The [Smoke100K authors' page](https://bigmms.github.io/cheng_gcce19_smoke100k/)
provides synthetic smoke/clean pairs and masks, using LabelMe/NYU backgrounds.
It is a candidate for additional *training* diversity, not evidence of real
driving performance. The page announces expanded subsets, so verify the actual
archive and counts instead of assuming the name means exactly 100,000 pairs.
Before adding it, verify pairing, decode images, group clean backgrounds, check
exact/near overlaps against all existing held-out targets (especially RESIDE),
and freeze a versioned training-only addition. Its terms restrict redistribution;
do not include its images in a public or cross-organization transfer bundle.

Prioritize independently captured paired road-smoke scenes for a new external
restoration test, with capture-session/scene separation and aligned clear
references. Unpaired road images can support qualitative/detector evaluation,
but cannot provide reference PSNR/SSIM. Surgical smoke datasets are not a suitable
substitute for the project's road domain. No new source has been mixed into the
follow-up experiment; isolate sampling/loss changes first and audit additions.

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
