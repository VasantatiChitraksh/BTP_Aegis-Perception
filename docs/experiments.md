# Experiments and reproducibility

**Current four-weather training protocol:** [frozen v3 split, sampling, statistics and DGX commands](weather_protocol_v3.md). This supersedes the v2 follow-up for new runs; existing results remain historical.


The [execution plan v2](reports/execution_plan_v2.pdf) defines the current scope:
reproduce smoke restoration, extend to rain and fog/snow, measure downstream
object detection, and evaluate edge deployment. Diffusion, foundation-model
detectors, domain adaptation, and adversarial defence are future work.

## Experiment order

### Current priority: one shared four-weather Restormer

The immediate target is one model trained jointly on rain, fog/haze, snow, and
smoke. Detection, Jetson deployment, and additional architectures are deferred.
Use `configs/restoration/four_weather_restormer.yaml` after running the repair
procedure in [datasets.md](datasets.md). The original rain run remains historical;
its training split contains clean backgrounds duplicated in held-out splits.
Initialize the shared model from the official deraining weights, rather than
that fine-tuned checkpoint.

The first recipe uses the full Restormer, 256-pixel synchronized crops, AdamW,
L1 loss, BF16, and 60 epochs of 4,096 sampled pairs (61,440 optimizer updates at
batch size 4). Sampling chooses rain/fog/snow/smoke with probabilities
0.30/0.30/0.30/0.10, then an equal-probability clean-background group, then a
variant. Smoke receives less training weight because its many pairs contain
very few independent backgrounds. These proportions are an initial training
choice, not a demonstrated optimum.

Select checkpoints by validation PSNR averaged first over variants of a clean
background, then backgrounds within weather, then equally over the four weather
types. The frozen native-resolution validation panel includes up to 64 groups
and two variants per group (eight for smoke's few backgrounds), selected with a
fixed hash rather than taking adjacent frames; full validation splits remain available
in the individual repaired manifests. Final test evaluation reports per-image
PSNR/SSIM, per-weather results, raw-image baselines, and background-balanced
summaries. Inspect each weather curve: a higher average alone does not establish
that all four conditions improved. Sixty epochs is a starting budget, not a
claim of convergence; extend or adjust only using validation evidence.

On the configured DGX environment, from the repository root:

```bash
sbatch scripts/restoration/train_four_weather_dgx.sbatch
```

The job audits the manifest, archives the actual source, trains or resumes from
`last.pt`, and evaluates `best.pt` on the frozen test split after training.
The CPU fixture tests and preflight are functionality checks, not trained results.
The full-model CPU preflight uses 64-pixel crops; a 256-pixel FP32 attempt was
killed on the laptop. Production 256-pixel batch-4 BF16 GPU memory/throughput
remain unmeasured. The SLURM job validates native-resolution inference before
starting optimization; do not interpret local preflight as an A100 benchmark.

The transfer bundle `artifacts/transfer/aegis_four_weather.tar` contains the code,
official initialization, repaired CSVs, and paired images needed for these four
datasets, including their full validation sets. It excludes the laptop's CPU
environment, old runs, raw download archives, and the unused CSD/historical-smoke
datasets. Copy the `.tar` and `.tar.sha256` files to DGX scratch. Extract into an
empty directory so the existing rain experiment remains intact:

```bash
sha256sum -c aegis_four_weather.tar.sha256
mkdir aegis-four-weather
tar -xf aegis_four_weather.tar -C aegis-four-weather
cd aegis-four-weather
bash scripts/restoration/setup_four_weather_dgx.sh
sbatch scripts/restoration/train_four_weather_dgx.sbatch
```

The setup script installs the previously working CUDA 11.8/PyTorch 2.5.1 stack
and runs CPU functionality tests; GPU execution starts only through SLURM.
If reusing the already working project environment, skip setup and submit with
`AEGIS_VENV=/scratch/$USER/aegis-perception/.venv sbatch scripts/restoration/train_four_weather_dgx.sbatch`.
The bundle checksum verifies transfer integrity. Rebuild after changing code or
data with `python scripts/restoration/package_four_weather.py --output NEW_PATH.tar`.

### Completed shared-model baseline: 15 October 2026

DGX job 47470 completed 60 epochs and test evaluation. The downloaded run is
`artifacts/restoration/four_weather_restormer_seed42/`. Its 8,329 per-image test
records match the frozen test manifest, including background IDs; recomputed
per-image and background-balanced averages agree with the saved summaries.
Epoch 60 is the best checkpoint under the configured validation selection rule.

| Weather | Test pairs | Background groups | Raw PSNR | Restored PSNR | Restored SSIM |
|---|---:|---:|---:|---:|---:|
| Rain | 224 | 224 | 22.27 | 38.72 | 0.9772 |
| Fog/haze | 1,000 | 542 | 13.96 | 30.46 | 0.9673 |
| Snow | 6,000 | 1,999 | 18.64 | 39.00 | 0.9814 |
| Smoke | 1,105 | 6 | 19.48 | 21.86 | 0.8252 |

This table uses per-image RGB metrics at native resolution with BF16 inference.
Giving equal weight to backgrounds and then weather types yields 32.59 dB
PSNR and 0.9297 SSIM, versus raw-input 18.69 dB and 0.7192. The pooled per-image
35.70 dB score is dominated by the 6,000 snow pairs and should not be the sole
headline result. These are single-seed baseline results, not evidence of novelty
or superiority to published methods under different protocols.

Smoke is the main weakness: 73/1,105 test images lose PSNR relative to raw input,
and 153/1,105 lose SSIM. Mean PSNR gain is positive in each of the six test
background groups, but these remain only six backgrounds. Smoke validation
plateaus early and reaches its own highest score at epoch 38; the shared
checkpoint is selected using all four weather types. Investigate any next
training change using validation, keeping the test split frozen.

Training plus per-epoch validation took 10.74 hours, with a mean epoch time of
10.74 minutes and peak allocated GPU memory of 24.26 GiB on A100 40 GB. The
review directory contains the verified numeric summary and training/test plots.
The production GPU recipe is now exercised successfully; the earlier CPU
preflight limitation above remains a record of preparation, not the run status.

Native-resolution CPU FP32 previews in `review/` illustrate the results but do
not replace the saved BF16 evaluation. Fog and snow examples were selected near
their median test PSNR gain among images no larger than 300,000 pixels; smoke
includes a median-gain example and the lowest-gain eligible example. These smoke
examples retain substantial obscuration and show color/detail artifacts. Rain's
preview uses the smallest test image to limit CPU memory. Selection metadata is
saved alongside the previews; this is a small diagnostic gallery, not exhaustive
visual verification or a basis for tuning on the test set.

### Smoke follow-up and multi-GPU execution: 15 October 2026

The initial shared-model test includes all 8,329 images, including 6,000 snow
pairs. Per-weather results use only that weather's images. The existing
scene-balanced macro score already weighted weather equally (32.5886 dB,
0.929733 SSIM). A pooled 35.70 dB score should not replace it. Re-aggregating
saved predictions on the fixed 896-image panel gives 32.7889 dB / 0.932054;
this is a sampling difference, not improved model performance.

The follow-up configs initialize from the completed joint model's `best.pt`,
use a fresh optimizer at 2e-5 for 20 epochs, keep the global batch size at four
and 4,096 samples per epoch, and use `four_weather_v2.csv` (expanded smoke
validation, unchanged training/test membership). Every global batch contains
one example per weather. Sampling cycles shuffled background groups and then
variants. Smoke therefore receives 1,024 samples per epoch, versus an expected
409.6 previously. This is an experimental correction to exposure, not a claim
that its limited backgrounds have been solved by oversampling.

- `four_weather_balanced.yaml`: balanced sampling, original L1 objective.
- `four_weather_balanced_ssim.yaml`: the same setup with `L1 + 0.05*(1-SSIM)`.

The SSIM term uses RGB [0,1], uniform 7x7 windows and sample covariance, matching
the existing evaluation definition. SSIM computations remain FP32 with BF16
model forwards. Do not compare the scalar training loss across different loss
definitions as if it measured identical progress. Evaluate improvement on
held-out validation PSNR, SSIM and raw-input regression fractions.

Both runs measure their initialization on the expanded validation panel before
training. `best.pt` is saved only if macro background-balanced PSNR improves and
**each weather's background-balanced PSNR and SSIM is at least its initialization
value**. `last.pt` is always resumable. If no epoch meets the rule, no `best.pt`
is published and `selection_status.json` explicitly reports no qualified
candidate; retain the existing model. This rule does not guarantee every test
image will improve. Inspect smoke regression fractions and visual artifacts too.
Compare the two experiments using validation, and evaluate only the selected
candidate on test. The job deliberately does not automatically test every run.

Training, validation and standalone evaluation support single-node `torchrun`
with one process per GPU. Training shards each *global* batch across ranks;
validation/test use disjoint, unpadded shards and aggregate all observations
before scoring. Only rank zero writes checkpoints/reports. RNG states for every
rank are saved; exact epoch-boundary resume requires the same process count.
The existing GAN trainer remains single-device only.

On DGX, after copying the updated code/configs/manifests and retaining the
completed joint-model checkpoint at its configured path:

```bash
# Default: two full A100s, two images/GPU, global batch four.
sbatch scripts/restoration/train_weather_followup_dgx.sbatch
# The matched loss ablation, in its own output directory:
sbatch scripts/restoration/train_weather_followup_dgx.sbatch \
  configs/restoration/four_weather_balanced_ssim.yaml

# Four full GPUs keep global batch four (one image/GPU); scaling must be measured.
# Choose this instead of the two-GPU launch, never concurrently for the same run.
sbatch --gres=gpu:a100:4 scripts/restoration/train_weather_followup_dgx.sbatch
```

Do not increase global batch size or learning rate just because more GPUs are
available. Two GPUs are a starting configuration; four can suffer communication
overhead with only one image per rank. Independent ablations can also run on
separate allocations. More GPUs can reduce elapsed time, not guarantee higher
PSNR/SSIM. The previous run took 10.74 hours on one A100; no multi-GPU timing or
quality claim is available yet. CPU two-process tests exercise real distributed
training, exact resume and evaluation; NCCL/BF16 production execution still
needs the DGX run.

After choosing a validation-qualified candidate (replace config/checkpoint as
appropriate), run inside an allocated GPU job:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  scripts/restoration/evaluate.py \
  --config configs/restoration/four_weather_balanced.yaml \
  --checkpoint artifacts/restoration/four_weather_balanced_seed42/best.pt \
  --device cuda --panel data/manifests/clean/test_panel_224.json
```

This writes `metrics_test_panel.json`. Omit `--panel` for the full test, which
writes `metrics_test.json`; both headline equal-weather/equal-background scores
and retain per-image aggregates plus regression counts. To audit existing
predictions without inference:

```bash
python scripts/restoration/summarize_evaluation.py \
  --metrics artifacts/restoration/four_weather_restormer_seed42/metrics_test.json \
  --manifest data/manifests/clean/four_weather.csv \
  --panel data/manifests/clean/test_panel_224.json \
  --output artifacts/restoration/four_weather_restormer_seed42/review/fair_evaluation.json
```

### Broader project sequence (after the shared model)

1. **Prepare the first rain experiment.** Use `notebooks/restoration.ipynb` in
   prepare mode. Audit RealRain-1k, review alignment/background quality, preserve
   its official splits, and strictly load the official Restormer deraining model.
   No paid GPU training is started by prepare mode.
2. **Validate the restoration pipeline.** Use the explicit sanity stage, then
   fine-tune the standard Restormer with `rain_restormer.yaml`. Retain the
   untouched pretrained baseline, compare both against raw images, and select
   checkpoints on validation. This is an initial fine-tuning recipe, not a
   reproduction of the original Restormer 300k-step schedule. Repair historical
   smoke target leakage before any smoke reproduction. Existing Pix2Pix configs
   remain comparison baselines; use identical data/preprocessing for fair comparisons.
3. **Establish detection baselines.** Evaluate YOLOv8n on labelled clean/degraded
   pairs where available, and on real DAWN images per weather. Verify label
   mapping before comparing metrics. YOLOv8s is an optional capacity check.
4. **Extend restoration.** Add at least one of fog/snow after rain works.
   Generic restoration datasets provide supporting
   PSNR/SSIM evidence; driving scenes are needed for the road-perception claim.
5. **Measure restoration's effect.** Use the same detector, labels, and scenes
   for clean, degraded, resize-control, vanilla-restored, and attention-restored
   views. DAWN has no matched clean view: compare raw, resize-control, and restored.
6. **Compare detector training.** Compare a clean-trained detector with one
   trained on clean plus adverse images, then evaluate the latter on restored
   images. Keep all training views disjoint from held-out scenes.
7. **Run focused ablations.** Measure failure cases before selecting one
   architectural or loss change. Keep the data, budget, preprocessing, and
   initialization controlled. Compact Restormer variants do not inherit the
   full model's weights automatically. Unified models and task-aware losses are
   research extensions, and require a literature comparison before novelty claims.
8. **Deploy the frozen pipeline.** Validate PyTorch → ONNX outputs, then measure
   TensorRT FP16/INT8 accuracy and timing on the target device. The repository
   currently supplies ONNX tools; TensorRT and end-to-end timing are not implemented.

Build the complete restoration path before modifying architecture. If
restoration fails to improve detection, report and investigate that result
before adding complexity. Pretrained weights and ordinary fine-tuning are baselines.

## Split and metric rules

- Keep all variants of one scene in one split; split video by sequence/drive.
  Hash clean targets to catch exact duplicates and inspect scene IDs for
  re-encoded duplicates. Do not rebuild established test splits for a better score.
- Report mean and standard deviation over seeds 42/43/44. Preserve per-image
  results for paired comparisons and 95% bootstrap intervals over held-out scenes.
- Restoration: RGB PSNR/SSIM on `[0,1]` with identical resize/crop/native rules,
  plus fixed success and failure images. Training and evaluation use mean
  per-image RGB PSNR with an MSE floor of `1e-12` (120 dB ceiling for exact RGB
  matches), including raw-image baselines. Native validation pads to the model's
  required multiple, then removes padding before scoring. RGB scores are not
  directly comparable to published Y-channel scores.
- Detection: mAP@50, mAP@50:95, precision/recall, per-weather and per-class AP.
  Inspect erased or hallucinated objects. PSNR alone does not establish detector benefit.
- Keep label mappings, split definitions, preprocessing, and detector weights
  fixed across a view comparison. Never substitute unrelated clear images for DAWN.

## Run records

Use descriptive output directories, for example `smoke_attention_seed42`. Change
one experimental factor at a time and use a separate directory for every run.
The restoration trainer writes:

```text
run.json           Config, Git commit, Python/PyTorch/CUDA/device
resume.json        Environment of the latest resumed run
history.json       Training losses and validation metrics by epoch
best.pt            Validation-selected checkpoint with config and optimizer state
last.pt            Latest completed epoch, with scheduler/scaler/RNG/history
epoch_*.pt         Periodic checkpoints
```

Restoration evaluation writes `metrics_test.json` with per-image and aggregate
metrics. Detection evaluation writes `metrics.json` and native Ultralytics output.
The trainer records source-code, manifest, and pretrained-weight SHA-256,
parameter count, precision, throughput, epoch time, and peak allocated GPU memory.
Archive verification uses the checksums in the dataset catalog. Retain dataset
version, hardware settings, and fixed qualitative sample IDs. The v3 evaluator generates cluster-based statistical summaries; end-to-end
hardware measurements require separate benchmarking.

## Edge measurements

Use batch 1, at least 100 warm-up and 1,000 timed frames. Record image size,
precision, p50/p95 latency, FPS, peak memory, power, and joules/frame. State
Jetson model, power mode, clocks, JetPack, CUDA, cuDNN, and TensorRT versions.
Synchronize GPU work when timing it. Separate model timing from full pipeline
timing including preprocessing, restoration, detector, NMS, and transfers.

INT8 calibration must be disjoint from validation/test. Check numerical output
and task accuracy after each export or precision change. Report latency gains
alongside accuracy changes; do not infer real-time operation from model-only timing.

## Historical results

The [semester report](../archive/reports/semester_5/research_report.pdf) claims
attention smoke PSNR/SSIM around 25.30/0.79 versus vanilla 21.45/0.72.
The [notebook](../archive/notebooks/smoke_baseline.ipynb) records 25.30/0.7874,
but evaluates its training loader without a held-out split or validation-selected
checkpoint, and contains no vanilla comparison. Those values are historical
claims to reproduce, not verified baselines. Internship claims similarly lack
runnable checkpoints and raw logs in the repository.
