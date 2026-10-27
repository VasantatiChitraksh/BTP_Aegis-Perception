AEGIS / VERIFIED MODEL LIBRARY

Locally verified: 6/6 completed benchmark models.
best.pt: original validation-selected checkpoint, linked to its training run.
last.pt: last resumable training state, where locally available.
inference.pt: generator + config + selected epoch; optimizer/discriminator omitted.
training_run/: complete original run, including downloaded periodic snapshots.
metrics_test.json / history.json: evaluated scores and all 60 training epochs.
config.json / SHA256SUMS: exact model configuration and local integrity records.

Links save disk space. To copy a model folder elsewhere, use cp -L or rsync -L;
omit training_run and last.pt for a small inference-only handoff.
Weights also require this project's model code. Call official_runtime.activate()
before load_generator for all six architectures. Official models additionally
require artifacts/official_sources and configs/restoration/official_v1/sources.json.
inference.pt is not an ONNX or TensorRT engine and has not been quantized.
The original older Restormer is retained outside this six-model comparison.

Loading example (run with this repository's src on PYTHONPATH):
  from aegis_perception.official_runtime import activate
  from aegis_perception.checkpoints import load_generator
  activate()
  model, metadata = load_generator(
      'artifacts/models/restormer/inference.pt', device='cpu')
Inputs: NCHW RGB tensors normalized to [-1, 1]; use the project's predict()
helper for native shapes/padding. Keep input/output normalization unchanged.
