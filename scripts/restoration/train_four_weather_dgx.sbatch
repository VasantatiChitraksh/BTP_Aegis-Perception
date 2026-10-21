#!/usr/bin/env bash
# Submit from the repository root on the DGX login node.
#SBATCH --job-name=aegis-four-weather
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30000M
#SBATCH --time=40:00:00
#SBATCH --output=/scratch/%u/aegis-four-weather-%j.out
#SBATCH --error=/scratch/%u/aegis-four-weather-%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
. /etc/profile.d/modules.sh
module load anaconda/2023.03-1
eval "$(conda shell.bash hook)"
conda activate "${AEGIS_CONDA_ENV:-pytorch_12.1}"
. "${AEGIS_VENV:-.venv}/bin/activate"
set -u

python - <<'PY'
import torch
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise RuntimeError("This recipe requires a CUDA GPU with BF16 support")
print(torch.__version__, torch.cuda.get_device_name(0), flush=True)
PY

config=configs/restoration/four_weather_restormer.yaml
run_dir=artifacts/restoration/four_weather_restormer_seed42
python scripts/data/audit_manifest.py data/manifests/clean/four_weather.csv
mkdir -p "$run_dir"
# Capture the actual source, including uncommitted changes, before optimization.
if [[ ! -f "$run_dir/source_code.tar.gz" ]]; then
    tar -czf "$run_dir/source_code.tar.gz" src scripts configs pyproject.toml data/manifests/clean
    python -m pip freeze > "$run_dir/environment.txt"
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git diff --binary > "$run_dir/working_tree.patch"
    fi
fi
if [[ -f "$run_dir/last.pt" ]]; then
    python scripts/restoration/train.py --config "$config" --resume "$run_dir/last.pt"
else
    python scripts/restoration/train.py --config "$config"
fi
python scripts/restoration/evaluate.py --config "$config" --checkpoint "$run_dir/best.pt" \
    --split test --device cuda
