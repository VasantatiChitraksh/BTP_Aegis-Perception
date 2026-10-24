#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
. /etc/profile.d/modules.sh
module load anaconda/2023.03-1
eval "$(conda shell.bash hook)"
conda activate "${AEGIS_CONDA_ENV:-pytorch_12.1}"
. "${AEGIS_VENV:-.venv}/bin/activate"
set -u
bash scripts/restoration/setup_official.sh
mkdir -p artifacts/restoration/official_v1
flock -n artifacts/restoration/official_v1/submission.lock \
    python scripts/restoration/official_suite.py submit "$@"
