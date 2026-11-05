# Instructions for the manuscript assistant

Give the following prompt to the assistant helping write the paper:

> Read docs/manuscript/README.md, MODEL_RESULTS.md, DATASET.md, DEPLOYMENT.md and
> DRAFT_CORRECTIONS.md before editing the manuscript. Run verify_evidence.py.
> Use the accompanying JSON/CSV records as the numerical evidence. For every
> table or factual claim, retain the source path, experiment/model identity,
> split, metric definition and evidence level. Separate completed results,
> historical runs, user-reported device observations and proposed work. Do not
> infer performance from filenames, example output, theoretical FLOPs or a
> script existing. Do not invent TODO values. If evidence conflicts, report the
> discrepancy and inspect the run configuration; do not merge incompatible
> experiments. Preserve contributors' implementation files and trained weights.

## Reproduce the tables without GPUs or model weights

Run from the repository root after pulling the current branch:

```bash
python3 docs/manuscript/verify_evidence.py
python3 - <<'PY'
import csv, json
from pathlib import Path
root = Path('docs/manuscript')
for row in csv.DictReader((root / 'tables/six_models.csv').open()):
    if row['weather'] == 'macro_weather':
        print(row['model'], row['psnr_db'], row['ssim'])
data = json.loads((root / 'evidence/six_models.json').read_text())
for name, model in data['models'].items():
    print(name, model['run']['config']['model'])
    print('Training:', model['run']['config']['train'])
    print('Sampling:', model['run']['config']['data'])
PY
```

Read `reports/*.txt` when PDF extraction is unavailable. PDF figures carry
sample-selection and precision caveats; preserve those captions. Do not treat a
chosen visual example as a statistically representative survey.

For per-image analysis, `evidence/per_image_restoration.csv.gz` contains all six
models' 2,334 test rows each. Open with Python `gzip.open(path, 'rt')` and
`csv.DictReader`. The metric summaries retain regression counts and bootstrap
intervals. Recompute aggregate PSNR by the documented nested grouping; never
average all images and call it the primary score. Additional paired significance
tests are new analyses, not results already recorded in the PDF.

## Evidence precedence and locations

| Question | Read first | Code / original references |
|---|---|---|
| Current six-model scores | `tables/six_models.csv`, `evidence/six_models.json` | [evaluation.py](../../src/aegis_perception/evaluation.py) |
| Which images and groups? | `evidence/data_protocol.json`, `tables/dataset_split.csv` | [active.csv](../../data/manifests/weather_v3/active.csv), [reserve.csv](../../data/manifests/weather_v3/reserve.csv) |
| What data sources? | DATASET.md, `evidence/dataset_sources.json` | [catalog](../../data/datasets.yaml), [historical dataset guide](../datasets.md) |
| Actual sampling/training | Each saved `run.config`, `history` | [sampling.py](../../src/aegis_perception/data/sampling.py), [training.py](../../src/aegis_perception/training.py) |
| LiteWeather implementation | Run `architecture_version: 2` | [v2 implementation](../../src/aegis_perception/models/liteweatherformer_v2.py), [runtime](../../src/aegis_perception/liteweather_runtime.py) |
| Official baseline identity | Run `implementation: official_v1` | [source lock](../../configs/restoration/official_v1/sources.json), [adapters](../../src/aegis_perception/models/official.py), [runtime registration](../../src/aegis_perception/official_runtime.py) |
| Trained weight identity | `model_library_identity` in six-model evidence | [model index](../../artifacts/models/index.json) |
| Detection pilot | `tables/detection_pilot72.csv`, `evidence/edge_readiness.json` | [edge evaluation](../../scripts/deployment/edge_dawn.py) |
| Architecture/loss experiments | `evidence/joint_experiments.json`, joint CSVs/PDF | Preserved configs/history in the evidence JSON; original local paths/hashes in provenance.json |
| Hardware | `hardware_status.json`, DEPLOYMENT.md | [later Nano notes](../nano_yolo_demo.txt), [integrated guide](../edge_integrated_handoff.txt) |

The `provenance.json` source commit identifies the repository state read when
this pack was assembled. Some joint-research implementation files were local
untracked work at that time; their source hashes and saved run configs are
evidence identifiers, not a claim that the full source archive is in this clone.
This is a manuscript evidence release, not a new training submission package.

Use current run JSON over historical plans when reporting what ran. Preserve
historical PDFs unchanged. Later operator notes supplement older local-only
deployment reports; a missing raw device log limits independent verification.
If refreshing from the maintainer's workstation, `build_evidence.py --date
YYYY-MM-DD` recreates copied/derived evidence from local original artifacts.
It requires PyYAML and pdftotext; ordinary reading/verification does not.

## Required distinctions in the manuscript

- Rain-only 37.2565 dB and four-weather 32.5831 dB are different estimands.
- Six-model baseline Restormer, older four-weather Restormer, upstream pretrained
  Restormer and joint-v3 variants are distinct checkpoints.
- Frozen means trained weights are used unchanged during inference.
- Small model parameter count does not establish edge speed or memory fit.
- Raw/resize/restored detection views are three inputs to one frozen detector,
  not three trained detectors.
- Output-agreement PSNR for quantization is not PSNR against clean targets.
- FP32 ONNX, BF16 training, simulated weight quantization and actual TensorRT
  FP16/INT8 execution are separate facts.
- The original Nano and Orin Nano have separate deployment paths. Report the
  measured board and software versions, not an earlier planned target.
- Existing source-provided synthetic images were downloaded. Do not claim that
  our team trained a diffusion generator or synthesized this training dataset.
- Legacy prototypes and unverified hardware projection tables are not results
  for the completed official-model experiments.

Keep missing experiments explicit: multi-seed replication, broader labelled
detection evaluation, additional independent smoke backgrounds, detector-aware
training and accepted sustained full-pipeline hardware measurements.
