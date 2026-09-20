# Dataset collection and contract

Large datasets, derived images, and checkpoints are intentionally excluded from
Git. Only manifests, dataset cards, label mappings, and checksums should be
committed.

## Local layout

```text
data/
  datasets.yaml         Reviewed source registry and local archive metadata
  archives/             Immutable downloads (ignored by Git)
  raw/                  Immutable downloaded/extracted datasets
    smoke/<dataset_id>/
    rain/<dataset_id>/
    fog/<dataset_id>/
    snow/<dataset_id>/
    dawn/{fog,rain,snow}/
  derived/              Restored images and resize-control views
  manifests/            Committed paired split CSVs (when paths are portable)
  dataset_cards/        Source URL, license, version, checksum, and class mapping
```

Never edit files below `raw/`. Derived folder names should encode the source
dataset, restoration model/checkpoint, and evaluation split.

## Collection workflow

The reviewed catalog is [`datasets.yaml`](datasets.yaml). It deliberately covers
only smoke, rain, fog, and snow. A source is downloaded automatically only when
`selection: core` and `auto_download: true`; large, gated, redundant, or
task-mismatched sources remain documented without consuming disk.

```bash
# Show the reviewed collection without downloading anything.
python3 scripts/download_datasets.py --list

# Download all compact core sources, or one condition/source.
python3 scripts/download_datasets.py --core
python3 scripts/download_datasets.py --condition fog
python3 scripts/download_datasets.py --dataset o_haze

# Verify archives and extract them into source-specific immutable folders.
python3 scripts/prepare_datasets.py --verify
python3 scripts/prepare_datasets.py --extract

# Build portable, source-aware manifests after extraction.
python3 scripts/build_collection_manifests.py
python3 scripts/audit_manifest.py data/manifests/smoke_historical.csv
```

Downloads and extractions are restartable. Do not commit either directory.
Commit only the catalog, cards, checksums, and portable manifests. License notes
in the cards are conservative: a code-repository license is not assumed to grant
redistribution rights for its data.

CSD uses RAR5 compression and needs a current `7zz`/7-Zip release. The legacy
`p7zip 16.02` commonly shipped by older Linux distributions can list the
archive but cannot extract it; `prepare_datasets.py` prefers `7zz` when both
commands exist.

## Paired restoration manifest

CSV columns:

| Column | Meaning |
|---|---|
| `sample_id` | Stable scene identifier |
| `input_path` | Weather-degraded image |
| `target_path` | Matched clean image |
| `split` | `train`, `val`, or `test` |
| `weather` | `smoke`, `rain`, `fog`, or `snow` |
| `source` | Dataset/version identifier |

All versions of the same underlying clean scene must remain in one split. Run
`scripts/audit_manifest.py` before training; it detects exact clean-target
duplicates across splits.

## What each dataset can prove

| Dataset | Intended role | Important limitation |
|---|---|---|
| Historical smoke pairs + SmokeBench | Smoke restoration reproduction and modern paired surveillance evaluation | Historical archive provenance is unresolved; SmokeBench is not road-specific |
| Rain100L + RealRain-1k | Small standard benchmark plus diverse high-resolution paired rain | Rain100L mirror provenance and separate data rights require verification |
| O-HAZE | Real paired outdoor dehazing evaluation | Small and mostly not road-specific |
| RTTS + Foggy Driving | Real foggy road detection and driving imagery | No matched clean targets; native class taxonomies differ |
| CSD + SnowCityscapes | General paired desnowing plus road-oriented synthetic snow | Large synthetic sets; SnowCityscapes inherits Cityscapes restrictions |
| DAWN | Primary real adverse-weather detector test | No matched clean image; only raw-adverse vs restored is valid |
| ACDC | Optional real adverse qualitative/segmentation test | Native labels are semantic segmentation, not DAWN boxes |

## DAWN protocol

DAWN must be split once, stratified by weather and with its severe class
imbalance recorded. Freeze the held-out test list. Verify the exact annotation
release and class-index mapping before using `configs/detection/dawn.yaml`.

Do **not** label an unrelated clear dataset as DAWN's “clean baseline.” For DAWN,
report the same real images before and after restoration, plus a bicubic
down/up-sampled control at the restoration model's input resolution. Run the full
clean/degraded/restored comparison only on the same labelled clean scenes with
dataset-provided paired degraded views and compatible annotations.
