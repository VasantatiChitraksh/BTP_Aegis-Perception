#!/usr/bin/env bash
# Submit from the repository root after preparing the dataset and weights.
# Create /scratch/$USER before submitting; SLURM opens the log files first.
#SBATCH --job-name=aegis-rain
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/aegis-rain-%j.out
#SBATCH --error=/scratch/%u/aegis-rain-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

. /etc/profile.d/modules.sh
module load anaconda/2023.03-1
eval "$(conda shell.bash hook)"
conda activate "${AEGIS_CONDA_ENV:-pytorch_gpu1}"

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in the selected environment")
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("The rain config uses bf16; check the allocated GPU")
print("PyTorch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0), flush=True)
PY

test -s artifacts/pretrained/deraining.pth || {
    echo "Missing official pretrained weights: artifacts/pretrained/deraining.pth" >&2
    exit 1
}
python scripts/data/audit_manifest.py data/manifests/realrain1k.csv

checkpoint=artifacts/restoration/rain_restormer_seed42/last.pt
if [[ -f "$checkpoint" ]]; then
    python scripts/restoration/train.py \
        --config configs/restoration/rain_restormer.yaml --resume "$checkpoint"
else
    python scripts/restoration/train.py \
        --config configs/restoration/rain_restormer.yaml
fi
