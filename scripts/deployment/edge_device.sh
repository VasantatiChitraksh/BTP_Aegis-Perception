#!/usr/bin/env bash
# Run from the extracted package. Engines are built on this device, never copied from DGX.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN="${AEGIS_PYTHON:-python3}"
MODE="${1:-prepare}"
REST_DIR=artifacts/edge_v1/exports/256x448/restormer
YOLO_DIR=artifacts/edge_v1/exports/yolov8n
mkdir -p results engines
case "$MODE" in
  prepare)
    "$PYTHON_BIN" scripts/deployment/edge_verify_package.py
    "$PYTHON_BIN" scripts/deployment/edge_preflight.py --output results/preflight.json
    "$PYTHON_BIN" scripts/deployment/edge_build.py \
      --onnx "$REST_DIR/restoration.onnx" --fixtures "$REST_DIR/fixtures.npz" \
      --kind restormer --precision fp16 --output engines/restormer_fp16.engine \
      2>&1 | tee results/build_restormer.log
    "$PYTHON_BIN" scripts/deployment/edge_build.py \
      --onnx "$YOLO_DIR/yolov8n.onnx" --fixtures "$YOLO_DIR/fixtures.npz" \
      --kind yolov8n --precision fp16 --output engines/yolov8n_fp16.engine \
      2>&1 | tee results/build_yolov8n.log
    ;;
  benchmark)
    "$PYTHON_BIN" scripts/deployment/edge_verify_package.py
    RUN_ID="$(date +%Y%m%d_%H%M%S)"
    STATS_PID=''
    if command -v tegrastats >/dev/null 2>&1; then
      tegrastats --interval 1000 > "results/tegrastats_$RUN_ID.log" &
      STATS_PID=$!
    fi
    trap 'if [ -n "$STATS_PID" ]; then kill "$STATS_PID" 2>/dev/null || true; fi' EXIT
    for VIEW in raw resize_control restored; do
      "$PYTHON_BIN" scripts/deployment/edge_run.py \
        --restorer engines/restormer_fp16.engine --detector engines/yolov8n_fp16.engine \
        --source samples --mode "$VIEW" --warmup 20 --repeat 100 \
        --output "results/${RUN_ID}_$VIEW"
    done
    "$PYTHON_BIN" scripts/deployment/edge_summarize_device.py --results results
    ;;
  *)
    printf 'Usage: bash scripts/deployment/edge_device.sh prepare|benchmark\n' >&2
    exit 2
    ;;
esac
