# Reproducibility and run contract

## Required run artifacts

Each run directory must contain:

```text
run.json                 Full config, Git commit, Python/PyTorch/CUDA/device
history.json             Epoch-level training and validation metrics
best.pt                  Validation-selected checkpoint
metrics_test.json        Per-sample and aggregate held-out restoration metrics
detection_*.json         Raw detector metrics for every evaluated view
samples/                 Fixed qualitative success/failure sample IDs
```

Record the manifest SHA-256 and source dataset archive SHA-256 in the associated
dataset card. The current scripts capture the config and environment; dataset
cards/checksums remain a team responsibility because the data is not present.

## Run naming

Use:

```text
{task}_{weather}_{model}_{loss}_seed{seed}_{precision}
```

Example: `restore_smoke_attnpix2pix_gan-l1_seed42_fp32`.

Change one experimental factor at a time. Copy the YAML config and commit it
before starting a result intended for the paper.

## Split discipline

- Group all variants of an underlying clean scene into one split.
- Use train for optimization, validation for checkpoint/model selection, and
  test once after choices are frozen.
- Hash clean targets to catch exact leakage; also inspect filename-derived scene
  IDs to catch re-encoded duplicates.
- If video frames are used, split by drive/sequence, never by individual frame.
- Keep detector fine-tuning and restoration evaluation test sets disjoint.

## Seed and statistics policy

- Main training comparisons: seeds 42, 43, and 44.
- Report mean and standard deviation across training seeds.
- Add 95% bootstrap confidence intervals over held-out scenes for the main mAP
  differences and restoration metrics.
- Preserve per-sample results so paired tests remain possible.

## Metric policy

Restoration: PSNR and SSIM on RGB `[0,1]`, same resize/crop, plus visual failure
cases. Detection: mAP@50, mAP@50:95, precision, recall, per class, per weather,
and where possible AP by object size. Efficiency: p50/p95 batch-1 latency, FPS,
peak memory, watts, and joules/frame.

Never compare metrics across different splits, resizing rules, or label maps
without stating the protocol mismatch.

## Edge protocol

Pin and report Jetson power mode/clocks. Warm up first, synchronize CUDA around
timed regions, exclude dataset disk I/O from model latency, and separately report
end-to-end application latency. Use a calibration set disjoint from validation
and test for INT8. Validate numerical output after every export step.

## Historical notebook policy

`baseline.ipynb` is preserved as an immutable historical artifact. Its saved
outputs are evidence of a prototype run, not a held-out baseline. New results
must be produced through the versioned scripts/configs in this repository.
