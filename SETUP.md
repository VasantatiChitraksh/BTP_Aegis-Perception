# Restormer inference setup

This guide runs the **frozen Restormer → frozen YOLOv8n** pipeline. It does not
retrain either model. Start with FP16 on the Jetson; INT8 is an optional later
experiment. Engine compilation and hardware performance have **not yet been
verified on a Jetson**.

## Results and current assessment

- [Four-page report, including example images](artifacts/edge_v1/reports/restormer_edge_readiness.pdf)
- [Detection metrics CSV](artifacts/edge_v1/reports/detection_pilot72.csv)
- [Detailed results and export provenance](artifacts/edge_v1/reports/restormer_edge_readiness.json)
- [Local verification summary](artifacts/edge_v1/local_verification.json)

The completed detection experiment is a **72-image DAWN pilot**, with 24 images
per weather selected before scoring. It compares original images, resizing alone,
and Restormer restoration using the same detector and annotations.

| Detection mAP50–95 (%) | Original images | Restormer | Change (percentage points) |
| --- | ---: | ---: | ---: |
| Overall | 29.97 | **32.86** | +2.89 |
| Fog | 26.99 | **37.11** | +10.12 |
| Rain | **38.87** | 37.21 | −1.66 |
| Snow | **36.27** | 35.62 | −0.65 |

**Assessment:** promising for fog, mixed across weather. This justifies further
evaluation, but does not establish reliable all-weather detection or real-time
edge performance. The pilot is small, has no confidence intervals, and has no
smoke detection labels. Reported precision/recall use each evaluation's maximum-F1
operating point; they are not fixed-threshold comparisons. The full 704-image
detection evaluation remains optional and has not been run.

Restormer's earlier **32.58 dB PSNR / 0.9297 SSIM** describes its native-resolution
restoration benchmark. Those numbers are not detection accuracy and do not
describe this resized deployment pipeline.

## 1. Obtain the inference bundle

Git contains the source, tests, documentation and small result files. **A Git
clone alone does not contain the model graphs, fixtures or sample images.**

On the project maintainer's workstation, the prepared files are:

```text
artifacts/releases/aegis_restormer_edge_20261024.tar.gz
artifacts/releases/aegis_restormer_edge_20261024.tar.gz.sha256
```

The archive is approximately **136 MiB**. Obtain both files from the maintainer
and copy them to `~/aegis-edge/` on the Jetson. No public model-download URL is
provided by this repository. The archive includes ONNX graphs, numerical
fixtures, calibration tensors and three research image samples; their source
licenses are recorded in `THIRD_PARTY_NOTICES.txt`.

On the Jetson:

```bash
cd ~/aegis-edge
sha256sum -c aegis_restormer_edge_20261024.tar.gz.sha256
tar -xzf aegis_restormer_edge_20261024.tar.gz
cd aegis_restormer_edge_20261024
```

**Run the remaining device commands from this extracted directory**, not from a
bare repository clone. It contains the `PACKAGE.json` integrity manifest required
by the device scripts. Do not overlay newer source files onto a checksummed kit;
rebuild a new kit if the runtime changes.

## 2. Prepare the Jetson environment

Use the JetPack installation's CUDA, TensorRT and OpenCV. The code targets
TensorRT **8.5/8.6 or 10.x** and includes compatibility checks. An original Jetson
Nano is different from an Orin Nano and may have an unsupported software stack.

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r configs/deployment/edge_runtime_requirements.txt
export AEGIS_PYTHON="$PWD/.venv/bin/python"

"$AEGIS_PYTHON" scripts/deployment/edge_verify_package.py
"$AEGIS_PYTHON" scripts/deployment/edge_preflight.py --output results/preflight.json
```

If `venv` or `cv2` is missing, install the JetPack-compatible OS packages
`python3-venv` or `python3-opencv`. Do not replace JetPack's CUDA/TensorRT with
desktop pip packages. The inference runner does not require PyTorch or the
Ultralytics Python package on the device.

In a new terminal, return to the extracted directory and repeat the
`export AEGIS_PYTHON=...` command.

## 3. Build and validate the engines

```bash
bash scripts/deployment/edge_device.sh prepare
```

This builds Restormer and YOLOv8n sequentially on the device, then validates their
outputs against the supplied FP32 references. Successful builds produce:

```text
engines/restormer_fp16.engine
engines/restormer_fp16.json
engines/yolov8n_fp16.engine
engines/yolov8n_fp16.json
results/build_restormer.log
results/build_yolov8n.log
```

Restormer uses FP16 where allowed and preserves sensitive transformer operations
in FP32. Its four-image validation panel also checks PSNR/SSIM against clean
targets at the deployment geometry. An engine that fails the acceptance gates
is **not promoted** to the approved filename. These small checks do not replace
full accuracy evaluation.

Build engines on the actual target and rebuild after relevant hardware/software
changes. The configured 512 MiB builder workspace is not a cap on total RAM use.
Close memory-heavy applications, especially on a 4 GB device; compilation may
need more memory than inference.

## 4. Measure raw, resize-only and restored inference

```bash
bash scripts/deployment/edge_device.sh benchmark
```

Each mode processes the three supplied samples 100 times, after 20 warmup calls.
The script records:

- Per-run `benchmark.json`: frame count, elapsed throughput, median/p95 timings,
  model hashes and environment.
- `results/device_comparison.csv` and `.json`: completed measured runs.
- `results/tegrastats_<timestamp>.log`, when `tegrastats` is available.

This is a throughput experiment on repeated images, not a new accuracy test or
camera-latency measurement. Keep the same input, power mode and cooling when
comparing modes. No Jetson FPS, peak memory or power result is claimed in advance.

## 5. Run an image, video or camera

First try the bundled image:

```bash
"$AEGIS_PYTHON" scripts/deployment/edge_run.py \
  --restorer engines/restormer_fp16.engine \
  --detector engines/yolov8n_fp16.engine \
  --source samples/fog.jpg --mode restored \
  --warmup 10 --save-visuals --output results/image_restored_01
```

Replace `--source samples/fog.jpg` with your video path or `--source camera:0`.
For a camera, add `--max-frames 300` to make the first run finite. USB/V4L2 capture
is expected; CSI cameras may need a separately configured capture pipeline.

To compare the same recorded input without restoration, use `--mode raw` and a
different output directory, such as `results/image_raw_01`. `--mode resize_control`
isolates the effect of the restoration input resizing. Existing output
directories are deliberately rejected so previous results are preserved.

Outputs include detection coordinates in original-image pixels, a timing JSON,
and images or annotated video when `--save-visuals` is enabled. Visual overlays
default to confidence 0.25; they are not the mAP evaluation, which uses a
confidence sweep. Saved video FPS metadata is not measured processing FPS.

## 6. If FP16 validation fails

Read the build log and `.rejected.json`. Do not loosen tolerances simply to get
a passing engine. For a Restormer numerical failure, build a separate FP32 engine:

```bash
"$AEGIS_PYTHON" scripts/deployment/edge_build.py \
  --onnx artifacts/edge_v1/exports/256x448/restormer/restoration.onnx \
  --fixtures artifacts/edge_v1/exports/256x448/restormer/fixtures.npz \
  --kind restormer --precision fp32 --output engines/restormer_fp32.engine
```

Use `--restorer engines/restormer_fp32.engine` with `edge_run.py`. The convenience
benchmark wrapper uses FP16 filenames; use explicit `edge_run.py` commands for
FP32 comparisons. The same fallback pattern is available for YOLOv8n using its
ONNX graph, fixtures and `--kind yolov8n`.

Optional INT8 instructions, gate thresholds, dataset provenance and source links
are in [the detailed handoff](docs/restormer_edge_handoff.txt). INT8 is implemented
as an experiment with 64 training-only calibration inputs; it has not yet been
built or measured on the Jetson. Unstructured pruning has not been applied.

## Source and reproduction

| Location | Purpose |
| --- | --- |
| `src/aegis_perception/edge/` | Image geometry, detection decoding, quality metrics and TensorRT runtime |
| `scripts/deployment/edge_*.py` | Export, audit, evaluation, build, inference, reporting and packaging tools |
| `scripts/deployment/edge_device.sh` | Sequential device preparation and benchmark commands |
| `configs/deployment/` | Separate local-export and device-runtime dependencies |
| `tests/test_edge_pipeline.py` | Geometry, label mapping, metric and precision-gate tests |

The recorded local validation passed **14 tests**. Export/reproduction commands
in the [detailed handoff](docs/restormer_edge_handoff.txt) require the maintainer's
existing checkpoints, model index and licensed local datasets; the Git clone
does not supply those artifacts. A full DAWN run can be started from that local
project with `.venv/bin/python scripts/deployment/edge_dawn.py all --full`.

Generated graphs, engines, checkpoints, calibration tensors, datasets, logs and
release archives stay outside ordinary Git. The published report, CSV and JSON
summaries retain the actual measurements and their limitations.
