# Manuscript evidence pack

**Updated 5 November 2026. Start here when writing the paper.** This folder
collects the completed experiments and their limits. It supersedes the earlier
rain-only draft as a description of project status; it does not replace or
rewrite historical experiment records.

## Reading order

1. [Six-model results and experimental methods](MODEL_RESULTS.md): the main
   restoration comparison, model identities, selection and limitations.
2. [Dataset sources, splits and sampling](DATASET.md): the actual four-weather
   mixture, real/synthetic provenance, background counts and leakage audit.
3. [Detection, compression and hardware status](DEPLOYMENT.md): why Restormer
   is the current deployment candidate and what has actually been tested.
4. [Corrections to the rain-only draft](DRAFT_CORRECTIONS.md): section-specific
   corrections, including implementation versions and device identity.
5. [Instructions for an AI writing/coding assistant](AI_GUIDE.md): extraction
   commands, evidence precedence and claim boundaries.

## Reports and numerical evidence

| Material | PDF | Machine-readable evidence |
|---|---|---|
| Main six-model comparison | [Six-model benchmark](reports/six_model_benchmark.pdf) | [Full-precision results and intervals](tables/six_models.csv), [run configurations, histories and metric summaries](evidence/six_models.json) |
| Restormer deployment preparation and detection pilot | [Edge readiness](reports/restormer_edge_readiness.pdf) | [Detection table](tables/detection_pilot72.csv), [export and pilot evidence](evidence/edge_readiness.json) |
| Completed eight-arm Restormer investigation | [Architecture/loss experiments](reports/joint_research_report.pdf) | [Test table](tables/joint_test.csv), [validation table](tables/joint_validation.csv), [runs and selection decision](evidence/joint_experiments.json) |
| Earlier rain-only experiment, historical | [Rain baseline](reports/rain_baseline_historical.pdf) | [Original run and scores](evidence/rain_baseline_historical.json) |
| Current data audit | — | [Audit](evidence/data_audit.json), [frozen protocol](evidence/data_protocol.json), [counts](tables/dataset_split.csv) |
| Compression rejection screens | — | [Pruning](tables/pruning_screen.csv), [custom weight quantizer](evidence/custom_quantization_screen.json) |

Each PDF has a matching `.txt` extraction in `reports/` for assistants without
PDF support. PDFs are unchanged copies of the source reports, so their original
dates and status statements remain historical. Read DEPLOYMENT.md for the later
Nano notes; the October 24 PDF predates those attempts.

The main findings are:

- Six models completed 60-epoch runs on the same frozen four-weather protocol.
  Restormer leads this comparison at **32.5831 dB / 0.929690 SSIM**.
- The current deployment candidate is the **epoch-51 Restormer**, followed by
  frozen pretrained YOLOv8n. Frozen means its trained weights are retained and
  no longer updated during inference.
- The 72-image detection pilot is mixed: aggregate mAP50–95 improves, fog
  improves, rain and snow decline against raw input. It is not a six-detector
  comparison or a measured quantized-device result.
- Eight further Restormer arms gave small gains; none passed the predefined
  validation replacement rule. No new architecture is established as superior.
- Full Restormer + detector deployment on the target device is not validated.
  Later notes contain limited original-Nano detector/build observations, which
  must not be described as sustained end-to-end performance.

## Scope and provenance

The proposed paper can be framed as **Aegis Perception: A Controlled
Multi-Weather Restoration Benchmark and Downstream Evaluation for Edge
Perception**. The earlier rain result can be an appendix. A new algorithmic
contribution, detector-aware training and successful edge acceleration remain
unestablished.

This pack contains reports and compact evidence, not weights or raw datasets.
The exact pair lists are already tracked under
[data/manifests/weather_v3](../../data/manifests/weather_v3).
Model identities and download paths are described in
[the model index](../../artifacts/models/index.json) and
[deployment status](DEPLOYMENT.md).

From the repository root, verify this snapshot with:

```bash
python3 docs/manuscript/verify_evidence.py
```

[provenance.json](provenance.json) records original source paths and SHA-256
hashes, including the derivations of compact JSON/CSV files. Some original
artifact paths are intentionally local-only; the copied evidence here is what
a fresh clone needs. Original manifests and protected contributor code are
unchanged. Update this folder's summaries and checksums whenever new verified
results become available; do not silently overwrite historical measurements.
