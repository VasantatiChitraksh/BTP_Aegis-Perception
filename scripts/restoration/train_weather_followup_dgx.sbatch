#!/usr/bin/env bash
# Submit from the repository root; override --gres for 1, 2 or 4 full GPUs.
#SBATCH --job-name=aegis-weather-followup
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/%u/aegis-followup-%j.out
#SBATCH --error=/scratch/%u/aegis-followup-%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
. /etc/profile.d/modules.sh
module load anaconda/2023.03-1
eval "$(conda shell.bash hook)"
conda activate "${AEGIS_CONDA_ENV:-pytorch_12.1}"
. "${AEGIS_VENV:-.venv}/bin/activate"
set -u
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
config="${1:-configs/restoration/four_weather_balanced.yaml}"
mapfile -t settings < <(python - "$config" <<'PY'
import sys
import torch
import yaml
c = yaml.safe_load(open(sys.argv[1]))
print(c['run']['output_dir'])
print(c['data']['manifest'])
print(torch.cuda.device_count())
PY
)
run_dir="${settings[0]}"
manifest="${settings[1]}"
gpus="${settings[2]}"
if [[ "$gpus" -lt 1 ]]; then
    echo 'No allocated GPUs visible.' >&2
    exit 1
fi
python scripts/data/audit_manifest.py "$manifest"
mkdir -p "$run_dir"
# Capture each launch, including the implementation actually used on resume.
tar -czf "$run_dir/source_${SLURM_JOB_ID}.tar.gz" \
    src scripts configs tests pyproject.toml data/manifests/clean
python -m pip freeze > "$run_dir/environment_${SLURM_JOB_ID}.txt"
resume_args=()
if [[ -f "$run_dir/last.pt" ]]; then
    resume_args=(--resume "$run_dir/last.pt")
fi
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$gpus" \
    scripts/restoration/train.py --config "$config" "${resume_args[@]}"
if [[ ! -f "$run_dir/best.pt" ]]; then
    echo 'No candidate improved the selection score while retaining every weather PSNR and SSIM.'
    echo "Inspect $run_dir/history.json and selection_status.json; retain the existing model."
    exit 0
fi
echo "Validation-qualified candidate: $run_dir/best.pt"
echo 'Compare the sampling-only and SSIM experiments on validation before final test evaluation.'
echo "Panel: python -m torch.distributed.run --standalone --nproc-per-node=$gpus scripts/restoration/evaluate.py --config $config --checkpoint $run_dir/best.pt --device cuda --panel data/manifests/clean/test_panel_224.json"
echo 'Omit --panel for the full, equally weighted final test report.'
