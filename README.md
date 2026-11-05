# Aegis Perception

Image restoration and downstream object-detection evaluation for adverse-weather
perception. The project compares six trained restoration models on a fixed
four-weather protocol and prepares the trained Restormer + YOLOv8n pipeline for
edge testing. Sustained full-pipeline device performance is not yet established.

## Writing the paper: start here

**[Manuscript evidence pack](docs/manuscript/README.md)** is the reference for
current results, dataset details, report PDFs, CSV/JSON tables and limitations.
Give [AI_GUIDE.md](docs/manuscript/AI_GUIDE.md) to the assistant helping write the
draft. It explains how to extract and verify evidence without model weights.

| What you need | Reference |
|---|---|
| All six models, scores and training methods | [Model comparison](docs/manuscript/MODEL_RESULTS.md) · [PDF](docs/manuscript/reports/six_model_benchmark.pdf) · [CSV](docs/manuscript/tables/six_models.csv) |
| Dataset split, sources, real/synthetic provenance and sampling | [Dataset guide](docs/manuscript/DATASET.md) · [Audit](docs/manuscript/evidence/data_audit.json) |
| Detection pilot, compression and actual hardware status | [Deployment evidence](docs/manuscript/DEPLOYMENT.md) |
| Completed architecture/loss experiments | [Report](docs/manuscript/reports/joint_research_report.pdf) · [Results](docs/manuscript/tables/joint_test.csv) |
| Changes needed in the earlier rain-only manuscript | [Draft corrections](docs/manuscript/DRAFT_CORRECTIONS.md) |

The older rain-only report and earlier planning documents remain historical.
Use saved run configurations to describe completed runs; do not infer experimental
status from an implementation filename or an example command.

## Completed restoration comparison

All six models were trained for 60 epochs with seed 42. Checkpoints were selected
by validation. These are native-resolution, equally weather/background/stratum
weighted test scores on the same 2,334 pairs; pretraining and model recipes differ.

| Model | Parameters | Test PSNR (dB) ↑ | Test SSIM ↑ |
|---|---:|---:|---:|
| Attention Pix2Pix | 29,854,795 | 24.7281 | 0.820120 |
| **Restormer** | **26,126,644** | **32.5831** | **0.929690** |
| LiteWeather v2 | 2,189,475 | 27.2071 | 0.864443 |
| TransWeather | 38,052,078 | 29.0444 | 0.900040 |
| PromptIR | 35,592,263 | 31.3899 | 0.918225 |
| RAMiT | 934,651 | 28.6586 | 0.899470 |

**Restormer is the current quality-first deployment candidate**, using its
trained epoch-51 weights. It is not the smallest model. The later eight-arm
Restormer study did not pass the predefined validation replacement criteria.
This is an empirical comparison under our protocol, not a claim of architectural
novelty, general superiority or measured real-time edge performance.

The trained LiteWeather v2 has transformer attention and input-derived prompt
routing. Official TransWeather/PromptIR/RAMiT adapters use pinned upstream
sources and weights; preserved legacy prototypes are different implementations.
See the [source lock](configs/restoration/official_v1/sources.json) and
[model library index](artifacts/models/index.json).

## Frozen dataset

| Weather / source | Train | Validation | Test |
|---|---:|---:|---:|
| Rain / RealRain-1k-H | 773 | 112 | 224 |
| Fog proxy / RESIDE-6K | 5,102 | 897 | 1,000 |
| Snow / SnowCityscapes | 5,382 | 618 | 768 |
| Smoke / SmokeBench repaired split | 7,353 | 384 | 342 |
| **Total pairs** | **18,610** | **2,011** | **2,334** |

Smoke has only 31/6/6 background groups across train/validation/test; thousands
of related frames do not establish broad scene coverage. The repaired protocol
removes known reference overlap, but cannot certify all unknown sequence overlap.
Exact manifests live in [data/manifests/weather_v3](data/manifests/weather_v3).
Source-provided synthetic haze/snow were downloaded; these runs did not generate
new training images with diffusion or GANs. See the [source explanation](docs/manuscript/DATASET.md).

## Detection and deployment status

A completed 72-image DAWN pilot compares raw, resize-only and Restormer-restored
views using one frozen YOLOv8n detector. Overall mAP50–95 changes from **29.97%
to 32.86%**. Fog improves; rain and snow decline slightly against raw input.
This exploratory pilot has no smoke detection labels or confidence intervals.
It is not a quantized-device benchmark. [Full results and limits](docs/manuscript/DEPLOYMENT.md).

FP32 ONNX export checks passed on small panels. FP16-permitted engine builders,
FP32 fallbacks and optional INT8 calibration experiments are prepared. The tested
custom weight-quantization simulation failed numerical gates and did not reduce
ONNX size; zero-shot pruning candidates also lost quality. Contributor research
implementations remain preserved, but their projections are not measured results.

### Select the correct hardware path

| Hardware | Status and instructions |
|---|---|
| **Original Jetson Nano 4GB** | Latest operator notes report L4T 32.7.3 / Python 3.6.9 / TensorRT 8.2.1.8. Use the [separate Nano adapter guide](docs/nano_yolo_demo.txt). Limited detector/build observations exist; Restormer fit and sustained combined performance remain unvalidated. |
| **Jetson Orin family / earlier Orin Nano plan** | Use the [integrated handoff](docs/edge_integrated_handoff.txt) and [SETUP.md](SETUP.md). Do not use its environment instructions on the original Nano. No accepted Orin device benchmark is recorded here. |

Record the exact device and evidence in
[hardware_status.json](docs/manuscript/hardware_status.json), and update
[DEPLOYMENT.md](docs/manuscript/DEPLOYMENT.md) with the underlying logs when
measurements arrive. That JSON is a documentation record, not a runtime switch.
The two hardware families and their precision support must not be conflated.

### Download the trained inference package

Use the [integrated release](https://github.com/VasantatiChitraksh/BTP_Aegis-Perception/releases/tag/edge-integrated-20261030)
and its SHA256 file; extract it and read `START_INTEGRATED.txt`. For original-Nano
memory trials, follow the separate Nano guide and its resolution-specific release
links. Smaller static exports use the same trained weights; their quality and
memory requirements must be measured separately.

A Git clone contains code and evidence, not weights, raw datasets or TensorRT
engines. Obtain binaries from [Releases](https://github.com/VasantatiChitraksh/BTP_Aegis-Perception/releases).
Build and validate engines on the actual device. Do not describe a prepared
script or a successful export as accepted deployment or native INT8 acceleration.

## Repository map and local development

- `docs/manuscript/`: curated paper evidence, report PDFs, tables and AI guide.
- `src/aegis_perception/`: models, data loading, training, evaluation and edge runtime.
- `scripts/`: dataset preparation, restoration experiments and deployment commands.
- `configs/`: exact experiment and deployment configurations.
- `data/manifests/`: versioned pair/group membership; raw images remain local.
- `edge_pipeline/`, `quantization/`: preserved contributor prototypes and experiments;
  consult the current evidence pack before citing their results or projections.
- `artifacts/`: original local outputs, with selected compact evidence tracked in Git.

For development on a supported Python 3.10+ workstation:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[detection,export,metrics,dev]'
```

For a documentation-only verification, no GPU or extra Python packages are needed:

```bash
python3 docs/manuscript/verify_evidence.py
```

Use `pytest` for code checks in an appropriately prepared development environment.
Device validation is separate. Do not install this workstation environment over
an existing original-Nano JetPack/ROS stack.

## License and attribution

Project code is under the [MIT License](LICENSE). Upstream models, datasets and
Ultralytics YOLO have their own terms. See the [dataset catalog](data/datasets.yaml),
[official model source lock](configs/restoration/official_v1/sources.json) and
release notices. Contributor implementations are preserved; the manuscript pack
identifies which versions were actually evaluated.
