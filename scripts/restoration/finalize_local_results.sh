#!/usr/bin/env bash
# Run locally after the DGX rsync completes. No training or remote mutation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON="${AEGIS_LOCAL_PYTHON:-.venv/bin/python}"
"$PYTHON" scripts/restoration/archive_completed_models.py
"$PYTHON" scripts/restoration/export_six_model_examples.py
"$PYTHON" scripts/restoration/build_benchmark_report.py
"$PYTHON" scripts/restoration/report_six_models.py --require-complete --no-pdf
printf '\nModels: %s/artifacts/models\nReport: %s/artifacts/reports/six_models_20261024/aegis_six_model_benchmark.pdf\n' "$PWD" "$PWD"
