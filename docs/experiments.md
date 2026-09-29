# Experiments and reproducibility

The [execution plan v2](reports/execution_plan_v2.pdf) defines the current scope:
reproduce smoke restoration, extend to rain and fog/snow, measure downstream
object detection, and evaluate edge deployment. Diffusion, foundation-model
detectors, domain adaptation, and adversarial defence are future work.

## Experiment order

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
version, hardware settings, and fixed qualitative sample IDs. Statistical
summaries and end-to-end hardware measurements are not generated automatically.

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
