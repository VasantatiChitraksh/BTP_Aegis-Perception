#!/usr/bin/env bash

#SBATCH --job-name=aegis_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --mem=30000M
#SBATCH --output=/scratch/%u/aegis_eval-%j.out
#SBATCH --error=/scratch/%u/aegis_eval-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

if [[ $# -lt 2 ]]; then
    echo "Usage: sbatch scripts/evaluate_on_dgx.sh <config_yaml> <checkpoint_pt> [evaluation options]" >&2
    exit 2
fi

. /etc/profile.d/modules.sh
module load anaconda/2023.03-1

eval "$(conda shell.bash hook)"
conda activate "${AEGIS_CONDA_ENV:-pytorch_12.1}"
. .venv/bin/activate

python -u scripts/restoration/evaluate.py \
    --config "$1" --checkpoint "$2" --split test --device cuda "${@:3}"

echo "Evaluation completed."
