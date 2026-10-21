#!/usr/bin/env bash
# Run from the extracted bundle root on the DGX login node.
set -eo pipefail
. /etc/profile.d/modules.sh
module load anaconda/2023.03-1
eval "$(conda shell.bash hook)"
conda activate "${AEGIS_CONDA_ENV:-pytorch_12.1}"
set -u
if [[ ! -x .venv/bin/python ]]; then
    python -m venv .venv
fi
.venv/bin/python -m pip install 'torch==2.5.1+cu118' --index-url https://download.pytorch.org/whl/cu118
.venv/bin/python -m pip install -e '.[metrics,dev]' 'numpy==1.26.4'
.venv/bin/python -m pip check
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest \
    tests/test_four_weather.py tests/test_restoration_pipeline.py
printf '%s\n' 'Environment ready. Submit: sbatch scripts/restoration/train_four_weather_dgx.sbatch'
