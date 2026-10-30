# Synthetic dashcam weather experiment

Use recorded clear footage, create reproducible weather corruptions, and replay the
same frames through validated Jetson engines. This workflow does not require real
rain/snow/fog/smoke, a simulator installation, or retraining.

The Jetson itself does not provide a dashcam recording or a camera. A USB camera
must be attached, or a video must be supplied separately. CSI/MIPI cameras may need
a different capture pipeline; the recorder below uses OpenCV camera indices.

These are **screen-space synthetic stress tests**, not calibrated atmospheric
rendering. Fog uses a drifting smooth opacity field without scene depth; smoke
uses a drifting plume proxy; rain/snow use coherent moving particles. Severity
1/2/3 means light/medium/heavy *within this generator*, not mm/hour or visibility
in metres. There is no basis to claim real-weather robustness from these alone.

## Experiment fixed before testing

Conditions: clear + four weather types x three levels = **13 conditions**.

| Candidate | Input and processing | Question |
|---|---|---|
| raw | Degraded frames directly to frozen YOLO | Starting detection performance |
| resize_control | Deployment resize/padding round trip, then YOLO | Effect of resizing alone |
| fp32 | Restormer FP32, then the same YOLO | Restoration reference |
| fp16 | Restormer mixed FP16, then the same YOLO | Initial deployment candidate |
| int8 | Restormer calibrated mixed INT8, then the same YOLO | Quantization trade-off |

That is **65 condition/candidate cells per clip** when all three Restormer
engines passed validation. A first FP16-only run has 39 cells. Failed engine
builds are unavailable candidates, not zero-quality or zero-latency results.
Use the same detector engine for every candidate. Include clear inputs to detect
unnecessary restoration damage. Fix severity, seed, labels, detector confidence
and splits before looking at the final test results.

## 1. LAPTOP: create a dedicated preparation environment

From the repository root:

```bash
mkdir -p artifacts/weather_testing
python3 -m venv artifacts/weather_testing/venv
artifacts/weather_testing/venv/bin/python -m pip install \
  -r scripts/weather_testing/requirements-workstation.txt
```

This avoids changing the Jetson's CUDA/TensorRT installation. The workstation
script needs no PyTorch. Commands below assume this repository is the working
directory on the laptop.

## 2. Obtain a clear clip

Option A: use an existing permitted dashcam video. Copy it to
`artifacts/weather_testing/scene01.mp4`. Use clear footage with relevant road
objects; avoid assuming the input is a clean reference if it already has weather.

Option B: record a camera attached to the Jetson. First copy the standalone tool:

```bash
# LAPTOP terminal (not inside SSH)
read -r -p 'Jetson login (username@IP): ' JETSON_TARGET
ssh "$JETSON_TARGET" 'mkdir -p ~/aegis-weather/tools ~/aegis-weather/clips'
scp scripts/weather_testing/weather_suite.py \
  "$JETSON_TARGET:~/aegis-weather/tools/"
```

Then on the Jetson, with the verified deployment kit already installed:

```bash
# JETSON terminal
KIT="$HOME/aegis-edge/aegis_restormer_edge_compression_20261024"
WEATHER_PYTHON="$KIT/.venv/bin/python"
ls /dev/video*
"$WEATHER_PYTHON" "$HOME/aegis-weather/tools/weather_suite.py" record \
  --camera 0 --frames 300 --fps 30 \
  --output "$HOME/aegis-weather/clips/scene01.mp4"
```

A `/dev/video*` node is only a hint; the recorder confirms whether frames can be
read. If no camera is available, stop here and supply a video later. No weather
or model processing is applied while recording. `--fps` is a camera request;
actual frame timestamps are saved in the `.capture.json` sidecar. Keep the
sidecar beside the video. Use a new filename for another scene.

```bash
# LAPTOP terminal, same JETSON_TARGET as above
scp "$JETSON_TARGET:~/aegis-weather/clips/scene01.mp4" \
  artifacts/weather_testing/
scp "$JETSON_TARGET:~/aegis-weather/clips/scene01.capture.json" \
  artifacts/weather_testing/
```

Repeat with independent scenes for meaningful evaluation. Do not use one
stationary clip as evidence of general road-weather robustness.

## 3. LAPTOP: generate the paired evaluation inputs

```bash
artifacts/weather_testing/venv/bin/python \
  scripts/weather_testing/weather_suite.py prepare \
  --video artifacts/weather_testing/scene01.mp4 \
  --output artifacts/weather_testing/scene01 \
  --scene-id scene01 --split development \
  --seed 42 --frames 60 --stride 5 --max-width 960
```

This samples up to 60 frames, one every five source frames, and preserves aspect
ratio while limiting width to 960 pixels. It writes lossless PNGs for clear and
12 weather variants. All variants retain the same frame IDs and geometry.
PNG evaluation avoids introducing different video-encoding errors per weather.
For consecutive-frame evaluation use `--stride 1`; the particle motion is tied
to source timestamps. `--start` is a nominal video-time offset in seconds.

Outputs include `manifest.json`, `frames/<condition>/*.png`, and
`annotations.template.json`. The manifest records source/generator hashes,
seed, sampling and checksums of every generated image. A changed input fails
verification. Existing output directories are never overwritten.

Generator strengths are fixed in `weather_suite.py`. Changing that file changes
its recorded hash. These presets have not been empirically validated for realism.
Do not describe them as validated physical fog/rain/snow/smoke.

## 4. LAPTOP: prepare ground-truth labels

Annotate the **prepared clear PNGs**, at their saved resolution, using your
annotation tool. Export COCO-format bounding boxes; adapt the export to the
image IDs/filenames in `annotations.template.json`. Save as
`scene01/annotations.reviewed.json`.

Use the project's zero-based COCO category IDs:
`0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck`.
Each annotation needs a unique `id`, `image_id`, `category_id`, and `bbox` in
`[x, y, width, height]` pixels. `area` and `iscrowd=0` are filled if absent.
Keep crowd/ignore annotations explicit when applicable.

Add top-level `reviewed_image_ids` containing every sampled image ID, including
frames reviewed and confirmed to contain no target objects. Do this only after
review. The empty template deliberately cannot be scored as labelled data.

Labels describe the original underlying objects; they are inherited by every
synthetic condition because the geometry is unchanged. Dense weather may obscure
some objects. Predeclare this as an underlying-object robustness task, preserve
the same target set, and report severe occlusion limitations. Do not selectively
delete missed-object labels after looking at predictions. A visible-only task
would require separately reviewed labels per condition and a different evaluator.

You can run image quality/runtime tests before labels are ready. Detection mAP
will remain blank, not be inferred from confidence or another model's boxes.

## 5. LAPTOP: transfer inputs and the tool to the Jetson

```bash
read -r -p 'Jetson login (username@IP): ' JETSON_TARGET
ssh "$JETSON_TARGET" 'mkdir -p ~/aegis-weather/tools ~/aegis-weather/datasets'
scp scripts/weather_testing/weather_suite.py \
  "$JETSON_TARGET:~/aegis-weather/tools/"
scp -r artifacts/weather_testing/scene01 \
  "$JETSON_TARGET:~/aegis-weather/datasets/"
```

Keep this tool outside the deployment kit. It imports the verified kit runtime
but does not edit its files. If updating prepared inputs, use a new dataset name
instead of merging different revisions into the same remote folder.

## 6. JETSON: first FP16 matrix

The engines must already have passed `edge_device.sh prepare` in the kit.

```bash
cd "$HOME/aegis-weather"
KIT="$HOME/aegis-edge/aegis_restormer_edge_compression_20261024"
WEATHER_PYTHON="$KIT/.venv/bin/python"
"$WEATHER_PYTHON" tools/weather_suite.py run \
  --dataset datasets/scene01 --kit "$KIT" \
  --detector "$KIT/engines/yolov8n_fp16.engine" \
  --restorer "fp16=$KIT/engines/restormer_fp16.engine" \
  --output results/scene01_fp16_rep1
```

The runner includes raw and resize_control automatically. It verifies input
hashes and kit integrity; approved TensorRT engine hashes/validation records are
checked by the kit loader. It never silently substitutes a mock model.

For a no-inference inspection, append `--plan-only` and choose a separate output
folder. This produces `plan.json` only, not measurements. Use a fresh output
folder for the real run.

## 7. JETSON: full FP32/FP16/INT8 comparison

First build comparison engines using the kit's existing compression driver:

```bash
cd "$KIT"
export AEGIS_PYTHON="$KIT/.venv/bin/python"
bash scripts/deployment/edge_compression_device.sh
```

Inspect status/logs. Only if all candidates passed, run:

```bash
cd "$HOME/aegis-weather"
"$WEATHER_PYTHON" tools/weather_suite.py run \
  --dataset datasets/scene01 --kit "$KIT" \
  --detector "$KIT/engines/comparison_yolov8n_fp16.engine" \
  --restorer "fp32=$KIT/engines/comparison_restormer_fp32.engine" \
  --restorer "fp16=$KIT/engines/comparison_restormer_fp16.engine" \
  --restorer "int8=$KIT/engines/comparison_restormer_int8.engine" \
  --output results/scene01_full_rep1
```

If INT8 fails, omit its `--restorer` argument and record it as unavailable. Never
rename a failed candidate to an approved engine. Repeat runs with rep2/rep3 output
names under the same power/cooling conditions. This script executes candidates
in the supplied order; consider counterbalancing that order across repeated runs
if thermal/order effects are material.

Outputs: `plan.json`, per-candidate model/environment identity JSON,
per-frame detections/metrics JSONL, `summary.csv`, `comparisons.csv`, and `complete.json`.
The comparison CSV computes quality gains versus raw/resize-control and quality
changes/speed ratios versus the candidate labelled `fp32` when supplied. Absence of
`complete.json` means an interrupted/incomplete matrix; preserve error logs.

## 8. LAPTOP: collect results and score detection

```bash
# LAPTOP terminal
mkdir -p artifacts/weather_testing/jetson_results
scp -r "$JETSON_TARGET:~/aegis-weather/results/scene01_full_rep1" \
  artifacts/weather_testing/jetson_results/
artifacts/weather_testing/venv/bin/python \
  scripts/weather_testing/weather_suite.py score \
  --dataset artifacts/weather_testing/scene01 \
  --results artifacts/weather_testing/jetson_results/scene01_full_rep1 \
  --annotations artifacts/weather_testing/scene01/annotations.reviewed.json
```

Use the corresponding FP16-only result folder if that is what you ran. Detection
scoring uses pycocotools on the workstation, not on the Jetson. It requires all
frames reviewed, matching geometry, complete predictions and a matching dataset
hash. Output: `detection_metrics.json`, with COCO mAP50-95, mAP50 and recall@100
separately for each condition/candidate. Empty predictions are scored as misses;
a dataset with no ground-truth objects has undefined AP and is rejected.

## How to interpret the tables

- `psnr_db` / `ssim`: output versus the paired prepared clean frame, using the
  deployment's full resize/restore round trip. These are synthetic restoration
  scores, not the original native-resolution training report. Identical clear
  inputs have a finite 120 dB ceiling in the existing metric implementation.
- `pipeline_median_ms`, `pipeline_p95_ms`, `pipeline_only_fps`: measured model
  pipeline including its preprocessing/postprocessing. File reads, synthetic
  generation, quality scoring and result writing are outside this timer. This
  is not camera-to-display FPS or sustained real-time throughput; CPU scoring
  creates gaps between calls. Use the existing video runner separately for a
  sustained end-to-end timing experiment.
- Model identity JSON: actual engine bytes, hashes, requested precision and
  software environment. A precision request is not measured layer coverage.
- `detection_metrics.json`: accuracy against manually reviewed labels. The CSV
  leaves mAP blank until this separate scoring step; it is not updated in place.
- Fixed detector candidate threshold defaults to 0.001 to retain predictions for
  AP evaluation; NMS/max detections follow the frozen kit. Report these settings.
  Do not compare these timings against a 0.25-threshold demo without noting that
  the postprocessing workload differs.

Compare restoration against both raw and resize_control; compare quantized
restoration against FP32. Report deterioration as well as improvement. Synthetic
heavy occlusion can destroy information that restoration cannot recover.

## Before claiming a publishable result

Use independent scenes and separate development/validation/test clips. Keep all
weather variants and adjacent frames of a clip in the same split. Splitting one
long drive into near-identical neighbouring clips is not independent sampling.
The split field records your assignment; it does not detect overlapping scenes
across separate datasets. Maintain a separate source/scene register.

Freeze the renderer/settings before the held-out test. Repeat on different seeds
and scenes without selecting favourable examples. Bootstrap by scene/clip, not
individual adjacent frames. This tool scores each clip; a paper-wide AP must be
computed from pooled predictions with unique IDs, not an unqualified average of
per-clip AP. That multi-scene statistical aggregation is not implemented here.

Include simulator/generator details, samples, code version, seeds, severity,
scene counts, sample counts, model hashes, thresholds, hardware and limitations.
Without real-weather evaluation, phrase conclusions as robustness to these
synthetic corruptions, not proof of real-world fog/smoke/rain/snow performance.
No experiment guarantees a particular publication outcome.

Physics-based follow-up references (different from this lightweight generator):
- Foggy Cityscapes: https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/
- Physics-based rain: https://github.com/astra-vision/rain-rendering
- Controlled and real weather: https://github.com/princeton-computational-imaging/SeeingThroughFog
