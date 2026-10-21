#!/usr/bin/env bash
# Run on the DGX login node; only submits jobs, no GPU work on the login node.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
. /etc/profile.d/modules.sh
module load anaconda/2023.03-1
eval "$(conda shell.bash hook)"
conda activate "${AEGIS_CONDA_ENV:-pytorch_12.1}"
. "${AEGIS_VENV:-.venv}/bin/activate"
set -u
mkdir -p artifacts/restoration/baselines_v1
flock -n artifacts/restoration/baselines_v1/submission.lock \
    python scripts/restoration/baseline_suite.py submit "$@"
