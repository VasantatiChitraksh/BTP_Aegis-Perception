# Prioritized experiment plan

This is the pre-project experiment list requested by the revised execution plan.
Do not begin architecture expansion until Gate 0 and Gate 1 pass.

## Scientific questions

- Q1: Can the reported smoke attention improvement be reproduced on unseen scenes?
- Q2: Does attention help rain, fog, and snow consistently, or only smoke?
- Q3: Does restoration improve road-object mAP on the same labelled scenes?
- Q4: Does that gain transfer from synthetic/paired weather to real DAWN images?
- Q5: Is the gain still worthwhile after FP16/INT8 conversion and end-to-end edge timing?

## Gates

| Gate | Pass condition | If it fails |
|---|---|---|
| 0 — data | Sources/licenses/checksums recorded; pair audit passes; frozen split exists | Stop training and repair provenance/splits |
| 1 — smoke reproduction | Attention and vanilla complete for 3 seeds; held-out raw metrics saved | Report non-reproduction; inspect split and notebook provenance |
| 2 — motivating detector result | Weather causes a measurable mAP drop on same labelled scenes and DAWN raw baseline runs | Revisit dataset severity/label mapping before restoration |
| 3 — restoration utility | Restored mAP exceeds degraded mAP with a paired bootstrap interval; no critical class regresses unreported | Test task-oriented loss or no-restoration training |
| 4 — deployment | ONNX output matches PyTorch tolerance; FP16/INT8 latency and accuracy both recorded on device | Use FP16 as primary; report INT8 failure honestly |

## Experiments in execution order

### E0 — Data and historical artifact recovery (mandatory)

- Recover the exact smoke archive, Colab notebooks, checkpoints, and any
  internship source code from the report links.
- Record dataset version, license, URL, archive SHA-256, extraction date, image
  counts, dimensions, duplicates, pair failures, and class distribution.
- Create deterministic manifests grouped by underlying clean scene.
- Freeze `train/val/test` lists before inspecting test results.

Output: dataset cards, manifests, audit log, and a table classifying each old
claim as reproduced/not reproduced/not testable.

### E1 — Smoke reproduction and attention ablation (mandatory)

Run the same data splits and seeds `{42, 43, 44}` for:

| ID | Generator | Objective | Purpose |
|---|---|---|---|
| E1-A | Vanilla U-Net Pix2Pix | GAN + 100×L1 | True baseline |
| E1-B | Attention-gated U-Net Pix2Pix | GAN + 100×L1 | Reproduce sem-5 contribution |
| E1-C | Attention U-Net | L1 only | Determine whether adversarial texture helps or hallucinates |

Primary at this stage: held-out PSNR/SSIM with mean, standard deviation, and 95%
bootstrap confidence interval. Also save parameter count, latency, and at least
20 fixed qualitative examples/failures. Do not compare the new held-out values
directly to the notebook's training-set values as though protocols match.

### E2 — Detector degradation baseline (mandatory, before more restoration)

Use COCO-pretrained YOLOv8n at 640 px, batch 1:

1. clean labelled driving scenes;
2. the same scenes with seeded light/medium/heavy rain, fog, snow, and smoke;
3. DAWN raw images, per weather;
4. optionally YOLOv8s once as a capacity sensitivity check.

Output: per-weather/per-class mAP@50 and mAP@50:95, precision, recall, and a
severity curve. This proves the detector problem exists under the chosen data.

### E3 — Per-weather restoration extension (core)

Train attention and vanilla models for rain and fog/snow using exactly the E1
protocol. Prefer driving-scene paired data for the central claim; generic
Rain100/RESIDE/Snow100K results are supporting restoration benchmarks.

The v2 minimum is smoke plus rain plus one of fog/snow. The repository has
configs for both fog and snow so both can be run if verified data/compute are
available. Do not collapse fog and snow into one label.

Output: one table by model × weather × seed with PSNR, SSIM, parameters, FLOPs if
available, and GPU latency.

### E4 — Same-scene task-driven matrix (central paper evidence)

For each labelled clean test scene and each weather/severity, run the same frozen
detector on the three v2 views plus a resize control:

| View | Image | What it isolates |
|---|---|---|
| Clean | Original clean scene | Detector upper reference on that scene |
| Degraded | Geometry-preserving weather version | Weather damage |
| Degraded resize control | Degraded image down/up-sampled through the restoration input resolution | Resolution/codec cost without restoration |
| Restored | Degraded image after restoration | Restoration utility |

Keep labels identical because geometry must remain unchanged. Compare restoration
against the resize control, not only the native degraded image; otherwise the
historical 256 px restoration bottleneck confounds the result. Compare vanilla
and attention restoration without changing the detector. Report confidence
intervals over test images and AP for small/medium/large objects where supported.

Smoke has no DAWN subset; create its task-driven evaluation from labelled clear
driving scenes with controlled smoke synthesis, not from an unlabelled generic
smoke restoration test.

### E5 — Real-weather DAWN validation (mandatory)

For each DAWN weather subset, compare:

1. raw adverse image → detector;
2. the same down/up-sampled resize control → detector;
3. restored adverse image → the same detector.

There is no valid matched clean DAWN condition. Report the DAWN split, exact
class mapping, class imbalance, and per-weather results. Inspect hallucinated or
erased objects manually. A restoration gain on synthetic data but loss on DAWN
is a key failure mode, not a result to hide.

### E6 — Detector training strategy (core if E4 passes)

Fine-tune YOLOv8n with frozen test sets under these controlled regimes:

| ID | Training images | Test views |
|---|---|---|
| D0 | Clean only | Clean, degraded, DAWN raw |
| D1 | Clean + synthetic weather | Clean, degraded, DAWN raw |
| D2 | Clean + synthetic weather | Restored paired, DAWN restored |

This separates restoration benefit from ordinary adverse-weather augmentation.
Avoid training and testing on different renderings of the same base scene.

### E7 — Focused ablations (only after the core table)

Priority order:

1. attention off/on;
2. GAN + L1 versus L1-only;
3. loss weight `λ ∈ {50, 100, 200}` on one representative weather only;
4. restoration resolution 256 versus 512 on one representative weather;
5. per-weather models versus one conditional weather-embedding model;
6. task-aware detector feature/loss term;
7. optional pretrained TransWeather/PromptIR comparison;
8. optional ERUP-style differentiable filter baseline.

Do not run a grid across all weather types. Select on validation, then confirm
the frozen choice once on test.

### E8 — Export, quantisation, and edge benchmark (mandatory)

Validate in this order:

1. PyTorch FP32;
2. ONNX Runtime FP32;
3. TensorRT FP16;
4. TensorRT INT8 with a representative, disjoint calibration set.

For the restoration model, detector, and full pipeline report:

- mAP@50 and mAP@50:95 after conversion;
- PSNR/SSIM change after conversion;
- batch-1 latency p50/p95, FPS, peak memory;
- average power and joules/frame using `tegrastats`;
- image size, warm-up count, timed count, Jetson power mode, clocks, JetPack,
  CUDA, cuDNN, and TensorRT versions.

Use at least 100 warm-up frames and 1,000 timed frames (or the full fixed test
set repeated without data-loading time). Also report end-to-end latency including
preprocessing, restoration, detector, NMS, and transfers. The >30 FPS target is
an aspiration, not a reason to omit accuracy or power loss.

### E9 — Diffusion weather synthesis (optional augmentation ablation)

The provided adapter can generate training-only weather variants. Run it only
after the deterministic procedural baseline and use low image-to-image strength.

Before inheriting bounding boxes, verify label preservation with:

- detector agreement against the clean image;
- image registration/feature correspondence;
- manual review of a stratified sample including small pedestrians/cyclists;
- rejection of samples that move, invent, or erase objects.

Never use diffusion-generated images as the held-out test set. Diffusion-based
*restoration* remains outside the v2 delivery scope.

## Minimal result tables for the paper

1. Historical reproduction: attention vs vanilla, three seeds.
2. Restoration: weather × model with PSNR/SSIM and compute.
3. Task matrix: clean/degraded/restored mAP per weather and critical class.
4. Real DAWN: raw/restored mAP per weather.
5. Ablation: attention and loss choice.
6. Edge Pareto: precision × latency/FPS × power for FP32/FP16/INT8.
7. Qualitative successes and honestly selected failures.

## Stop rules

- Do not add a new architecture until E1 and E2 are reproducible.
- Do not claim restoration helps ADAS from PSNR/SSIM alone.
- Do not fine-tune on DAWN test images.
- Do not use an unmatched clear dataset as a clean counterpart to DAWN.
- Do not claim real-time from model-only GPU latency or Raspberry Pi training time.
- Do not claim INT8 success without post-quantisation accuracy.
