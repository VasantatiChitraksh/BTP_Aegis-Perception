#!/usr/bin/env bash
# Compare Restormer precisions with the same FP16 detector and the same inputs.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN="${AEGIS_PYTHON:-python3}"
"$PYTHON_BIN" scripts/deployment/edge_verify_package.py
RUN_ID="$(date +%Y%m%d_%H%M%S)_compression"
RUN_DIR="results/$RUN_ID"
mkdir -p "$RUN_DIR" engines
"$PYTHON_BIN" scripts/deployment/edge_preflight.py --output "$RUN_DIR/preflight.json"
REST_DIR=artifacts/edge_v1/exports/256x448/restormer
YOLO_DIR=artifacts/edge_v1/exports/yolov8n
"$PYTHON_BIN" scripts/deployment/edge_build.py \
  --onnx "$YOLO_DIR/yolov8n.onnx" --fixtures "$YOLO_DIR/fixtures.npz" \
  --kind yolov8n --precision fp16 --output engines/comparison_yolov8n_fp16.engine \
  2>&1 | tee "$RUN_DIR/build_detector.log"
printf 'precision\tstatus\n' > "$RUN_DIR/status.tsv"
STATS_PID=''
trap 'if [ -n "$STATS_PID" ]; then kill "$STATS_PID" 2>/dev/null || true; fi' EXIT
COMPLETED=0
for PRECISION in fp32 fp16 int8; do
  EXTRA=()
  if [ "$PRECISION" = int8 ]; then EXTRA=(--calibration "$REST_DIR/calibration.npz"); fi
  ENGINE="engines/comparison_restormer_$PRECISION.engine"
  if ! "$PYTHON_BIN" scripts/deployment/edge_build.py \
    --onnx "$REST_DIR/restoration.onnx" --fixtures "$REST_DIR/fixtures.npz" \
    --kind restormer --precision "$PRECISION" --output "$ENGINE" "${EXTRA[@]}" \
    2>&1 | tee "$RUN_DIR/build_$PRECISION.log"; then
    printf '%s\tbuild_or_validation_failed_see_log\n' "$PRECISION" >> "$RUN_DIR/status.tsv"
    continue
  fi
  if command -v tegrastats >/dev/null 2>&1; then
    tegrastats --interval 1000 > "$RUN_DIR/tegrastats_$PRECISION.log" &
    STATS_PID=$!
  fi
  if "$PYTHON_BIN" scripts/deployment/edge_run.py \
    --restorer "$ENGINE" --detector engines/comparison_yolov8n_fp16.engine \
    --source samples --mode restored --warmup 20 --repeat 100 \
    --output "$RUN_DIR/$PRECISION" 2>&1 | tee "$RUN_DIR/run_$PRECISION.log"; then
    printf '%s\tcompleted\n' "$PRECISION" >> "$RUN_DIR/status.tsv"
    COMPLETED=$((COMPLETED + 1))
  else
    printf '%s\tbenchmark_failed_see_log\n' "$PRECISION" >> "$RUN_DIR/status.tsv"
  fi
  if [ -n "$STATS_PID" ]; then
    kill "$STATS_PID" 2>/dev/null || true
    wait "$STATS_PID" 2>/dev/null || true
    STATS_PID=''
  fi
done
if [ "$COMPLETED" -gt 0 ]; then
  "$PYTHON_BIN" scripts/deployment/edge_summarize_device.py --results "$RUN_DIR"
fi
cat "$RUN_DIR/status.tsv"
printf 'Keep build logs, layer JSONs, validation JSONs and tegrastats with the comparison.\n'
[ "$COMPLETED" -eq 3 ]
