# Project audit and scope decision

Audit date: 26 August 2026

## Authority order

When documents conflict, use this order:

1. `Weather_Robust_ADAS_Execution_Plan_v2.pdf` — current 13-week delivery scope.
2. `sem5-works/Proposal 1 - TiHAN IITH _.pdf` — parent eight-month objectives.
3. Semester-5 reports and `baseline.ipynb` — evidence of the team's own work.
4. Internship reports — useful historical claims that must be reproduced before citation.

The v2 plan keeps multi-weather restoration, downstream object detection, and
edge deployment in scope. It explicitly defers diffusion restoration,
transformer/foundation-model detector comparisons, adversarial defence, and
domain adaptation. The repository follows that boundary.

The [official IEEE IV 2027 page](https://ieee-iv.org/2027/intelligent-vehicles-symposium-2027-perth-australia/)
confirms the v2 target dates: paper deadline 15 November 2026 and notification
15 January 2027.

## What is actually present

| Artifact | What it contains | Evidence status |
|---|---|---|
| Original TiHAN proposal | Broad aims: generative deweathering, robust detection, quantisation, adversarial robustness, domain adaptation | Authoritative parent scope |
| Revised execution plan v2 | 13-week smoke → rain + fog/snow extension, YOLOv8n/s, task-driven mAP, ONNX/TensorRT, Jetson | Current source of truth |
| Semester-5 progress report | Attention-gated Pix2Pix smoke method; claims PSNR 25.30 dB and SSIM 0.79 vs vanilla 21.45/0.72 | Result claim; raw run evidence absent |
| Semester-5 code report | Architecture/hyperparameters and the same smoke metrics | Result claim; linked Colab/checkpoint absent |
| `baseline.ipynb` | Colab implementation of the attention-gated smoke model | Executed prototype, but not a valid held-out benchmark |
| Internship report 1 | Claims separate rain, snow, smoke, fog Pix2Pix models, YOLOv5, and Raspberry Pi trials | Historical lead; code/checkpoints/logs absent |
| July monthly report | Repeats weather-removal and YOLOv5 claims; proposes YOLOv8/11/DETR and optimisation | Historical status report |
| Internship report 2 | Claims CycleGAN/Pix2Pix + DETR on Raspberry Pi | Internally inconsistent appendix; treat as unverified |
| Datasets | No dataset directory is present in this checkout | Blocking input for training |

The repository originally tracked only `README.md`, `baseline.ipynb`, and the
license. The PDFs and report directories were untracked at audit time. They have
not been renamed or modified.

## Notebook audit

The notebook is not an independent “baseline”; it is the semester-5 attention
model itself:

- paired `hazy/` and `clean/` folders on Google Drive, resized to 256×256;
- six encoder stages and five decoder stages;
- attention gates on four skip connections (`d5` through `d2`), with the
  highest-resolution `d1` skip left ungated;
- PatchGAN discriminator;
- `BCEWithLogitsLoss + 100 × L1` generator loss;
- Adam, learning rate `1e-4`, betas `(0.5, 0.999)`, 100 epochs;
- recorded output: PSNR 25.30 dB and SSIM 0.7874.

Those recorded metrics are not publication-ready because:

1. The same `dataloader` is used for training and evaluation; there is no
   train/validation/test split.
2. Evaluation is performed on the last in-memory epoch, not a checkpoint chosen
   on a validation set.
3. The evaluation loader is shuffled and no seed or deterministic split list is
   recorded.
4. No checkpoint, training curve, dataset checksum, environment, or per-image
   metric file is saved.
5. The notebook contains no vanilla Pix2Pix run, so it cannot substantiate the
   reported 21.45/0.72 comparison by itself.
6. It measures only image fidelity, not detector performance or latency.

The new package retains the same topology for reproduction, but separates
train/validation/test samples, audits duplicate targets across splits, selects
the checkpoint on validation PSNR, and emits machine-readable run metadata.

## Historical claims to reproduce

| Claim | Reported value | Required reproduction evidence |
|---|---:|---|
| Attention smoke | 25.30–25.31 dB / 0.7874–0.7906 SSIM | Fixed held-out split, 3 seeds, checkpoint and per-image metrics |
| Vanilla smoke | 21.45 dB / 0.72 SSIM | Same split, seed set, epochs, transform, and evaluation code as attention |
| Rain Pix2Pix | 24.92 dB; L1 0.0373 | Dataset version and held-out split; source report used only 861 pairs |
| Snow Pix2Pix | 23.07 dB / 0.7699 SSIM | Correct reported train/test counts and held-out manifest |
| Smoke internship model | 24.00 dB / 0.7646 SSIM | Determine whether this is a different run/data split |
| Raspberry Pi snow | 23.43 dB / 0.7516 SSIM | Checkpoint, 20-image list, device timing and software versions |
| GAN + DETR at 10–15 FPS | Claimed on Raspberry Pi | No usable DETR evidence; rerun or omit |
| Weather removal improves YOLO | Qualitative/unspecified | Same-scene clean/degraded/restored mAP matrix |

## Decisions made in this scaffold

- YOLOv8n is the primary detector; YOLOv8s is a capacity check only if schedule
  permits, as required by v2.
- Separate weather models are the guaranteed path. A unified conditional model
  is a stretch experiment after the core table exists.
- Smoke, rain, fog, and snow configurations are provided. The minimum paper
  claim is smoke + rain + one of fog/snow; run both fog and snow if verified
  paired data and compute are available.
- DAWN is a real-weather detection benchmark, not a paired restoration dataset.
  It supports raw-adverse vs restored comparisons only.
- Dataset-provided paired driving scenes support the same-scene three-way
  detection experiment without adding a local weather-generation pipeline.
- Diffusion restoration remains future work, in direct conformance with v2.

## Immediate external blockers

The code can be validated without large data, but experiments cannot begin until
the team records:

- exact dataset download/version/license/checksums;
- smoke data and any historical checkpoints/Colab notebooks;
- GPU model/VRAM allocation;
- Jetson model, JetPack/TensorRT versions, and power mode;
- confirmed DAWN class mapping and frozen split.
