#!/bin/bash
# =============================================================================
# batch_prune.sh — SLURM batch script to prune all 6 Aegis model families at
#                  sparsities [0.3, 0.5, 0.7].
#
# Usage:
#   sbatch quantization/scripts/batch_prune.sh
#
# Results are written to artifacts/quantization/pruning/<model_name>/
# =============================================================================

#SBATCH --job-name=aegis_prune
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --mem=32G
#SBATCH --output=/scratch/%u/aegis_prune-%j.out
#SBATCH --error=/scratch/%u/aegis_prune-%j.err

# ---------------------------------------------------------------------------
# Environment setup (mirrors train_on_dgx.sh)
# ---------------------------------------------------------------------------
. /etc/profile.d/modules.sh
module load anaconda/2023.03-1

eval "$(conda shell.bash hook)"
conda activate pytorch_gpu1

echo "===== NODE INFO ====="
hostname
echo "====================="

echo "===== GPU INFO ====="
nvidia-smi
echo "===================="

# ---------------------------------------------------------------------------
# User-configurable paths — edit these before submitting.
# ---------------------------------------------------------------------------
REPO_ROOT="${AEGIS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "$REPO_ROOT"

# Representative rain/degraded test image.
TEST_IMAGE="${AEGIS_TEST_IMAGE:?Set AEGIS_TEST_IMAGE to a representative degraded validation image}"

# Checkpoint paths — update to match your trained checkpoints.
CKPT_ATTENTION_UNET="${REPO_ROOT}/artifacts/models/attention_pix2pix/inference.pt"
CKPT_RESTORMER="${REPO_ROOT}/artifacts/models/restormer/inference.pt"
CKPT_TRANSWEATHER="${REPO_ROOT}/artifacts/models/transweather/inference.pt"
CKPT_PROMPTIR="${REPO_ROOT}/artifacts/models/promptir/inference.pt"
CKPT_RAMIT="${REPO_ROOT}/artifacts/models/ramit/inference.pt"
CKPT_LITEWEATHERFORMER="${REPO_ROOT}/artifacts/models/liteweatherformer/inference.pt"

# Output root.
OUTPUT_ROOT="${REPO_ROOT}/artifacts/quantization/pruning"

# Pruning method (global recommended — see PRUNING.md).
PRUNE_METHOD="global"

# Sparsities to evaluate.
SPARSITIES=(0.3 0.5 0.7)

# ---------------------------------------------------------------------------
PRUNE_SCRIPT="${REPO_ROOT}/quantization/scripts/prune_model.py"

declare -A CHECKPOINTS
CHECKPOINTS["attention_unet"]="${CKPT_ATTENTION_UNET}"
CHECKPOINTS["restormer"]="${CKPT_RESTORMER}"
CHECKPOINTS["transweather"]="${CKPT_TRANSWEATHER}"
CHECKPOINTS["promptir"]="${CKPT_PROMPTIR}"
CHECKPOINTS["ramit"]="${CKPT_RAMIT}"
CHECKPOINTS["liteweatherformer"]="${CKPT_LITEWEATHERFORMER}"

echo "===== Starting Pruning Benchmarks ====="
echo "Method: ${PRUNE_METHOD}"
echo "Sparsities: ${SPARSITIES[*]}"
echo "Test image: ${TEST_IMAGE}"
echo "Output root: ${OUTPUT_ROOT}"
echo ""

for MODEL_NAME in attention_unet restormer transweather promptir ramit liteweatherformer; do
    CKPT="${CHECKPOINTS[$MODEL_NAME]}"
    OUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}"

    if [ ! -f "${CKPT}" ]; then
        echo "[WARNING] Checkpoint not found: ${CKPT} — skipping ${MODEL_NAME}."
        continue
    fi

    for SPARSITY in "${SPARSITIES[@]}"; do
        echo "--------------------------------------------------"
        echo "Model: ${MODEL_NAME} | Sparsity: ${SPARSITY}"
        echo "--------------------------------------------------"

        python "${PRUNE_SCRIPT}" \
            --checkpoint "${CKPT}" \
            --sparsity "${SPARSITY}" \
            --method "${PRUNE_METHOD}" \
            --image "${TEST_IMAGE}" \
            --output-dir "${OUT_DIR}"

        EXIT_CODE=$?
        if [ ${EXIT_CODE} -ne 0 ]; then
            echo "[ERROR] prune_model.py failed for ${MODEL_NAME} s=${SPARSITY} (exit ${EXIT_CODE})."
        else
            echo "[OK] ${MODEL_NAME} @ s=${SPARSITY} complete."
        fi
        echo ""
    done
done

echo "===== All pruning benchmarks complete ====="
echo "Run quantization/scripts/generate_pruning_report.py to build the final report."
