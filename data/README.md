# Dataset contract

Large datasets, derived images, and checkpoints are intentionally excluded from
Git. Only manifests, dataset cards, label mappings, and checksums should be
committed.

## Local layout

```text
data/
  raw/                  Immutable downloaded/extracted datasets
    smoke/{hazy,clean}/
    rain/{rainy,clean}/
    fog/{hazy,clean}/
    snow/{snowy,clean}/
    dawn/{images,labels}/
  derived/              Generated weather and restored images
  manifests/            Committed paired split CSVs (when paths are portable)
  dataset_cards/        Source URL, license, version, checksum, and class mapping
```

Never edit files below `raw/`. Derived folder names should encode the condition,
severity/model, and seed.

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
| Existing smoke pairs | Reproduce sem-5 Attention-Pix2Pix | Provenance/license and split are not in this checkout |
| Rain100H/L | Controlled paired deraining | Mostly non-driving; do not infer ADAS performance from PSNR alone |
| RainCityscapes | Paired driving-scene rain experiment | Synthetic weather; keep original scene split groups together |
| RESIDE | Paired dehazing pretraining/evaluation | Many scenes are not driving scenes |
| Foggy Cityscapes | Driving-scene fog and inherited labels | Synthetic fog, not an independent real-weather test |
| Snow100K/CSD | Controlled paired desnowing | Check exact license and split protocol |
| DAWN | Primary real adverse-weather detector test | No matched clean image; only raw-adverse vs restored is valid |
| BDD100K weather subset | Larger detector fine-tuning pool | Weather tags are coarse and class mappings differ |
| ACDC | Optional real adverse qualitative/segmentation test | Native labels are semantic segmentation, not DAWN boxes |

## DAWN protocol

DAWN must be split once, stratified by weather and with its severe class
imbalance recorded. Freeze the held-out test list. Verify the exact annotation
release and class-index mapping before using `configs/detection/dawn.yaml`.

Do **not** label an unrelated clear dataset as DAWN's “clean baseline.” For DAWN,
report the same real images before and after restoration, plus a bicubic
down/up-sampled control at the restoration model's input resolution. Run the full
clean/degraded/restored comparison only on the same labelled clean scenes with
geometry-preserving synthetic degradation.
