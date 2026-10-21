#!/usr/bin/env bash
# One GPU per stage. Dependency chain keeps the three stages sequential.
#SBATCH --job-name=aegis-liteweather
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/%u/aegis-lite-%j.out
#SBATCH --error=/scratch/%u/aegis-lite-%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the packaged repository root}"
. /etc/profile.d/modules.sh
module load anaconda/2023.03-1
eval "$(conda shell.bash hook)"
conda activate "${AEGIS_CONDA_ENV:-pytorch_12.1}"
. "${AEGIS_VENV:-.venv}/bin/activate"
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
mkdir -p artifacts/restoration/liteweather_v1
python -m pip freeze > "artifacts/restoration/liteweather_v1/environment_${SLURM_JOB_ID}.txt"
flock -n artifacts/restoration/liteweather_v1/execution.lock \
    python scripts/restoration/liteweather_suite.py stage "${1:?config}" "${2:?through_epoch}"
