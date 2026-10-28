#!/bin/bash
# =============================================================================
# batch_quantize.sh — SLURM batch script to quantize all 6 Aegis model families.
#
# Usage:
#   sbatch quantization/scripts/batch_quantize.sh
#
# Before submitting, set the MODEL_CHECKPOINTS and TEST_IMAGE variables below
# to point to your actual checkpoint files.
#
# Results are written to artifacts/quantization/<model_name>/
# =============================================================================

#SBATCH --job-name=aegis_quantize
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --mem=32G
#SBATCH --output=/scratch/%u/aegis_quantize-%j.out
#SBATCH --error=/scratch/%u/aegis_quantize-%j.err

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

# Path to a representative rain/degraded test image.
# Replace with your actual path once datasets are downloaded, e.g.:
#   data/raw/rain/realrain1k/rain/0001.jpg
TEST_IMAGE="${AEGIS_TEST_IMAGE:?Set AEGIS_TEST_IMAGE to a representative degraded validation image}"

# Checkpoint paths for each of the 6 model families.
# Update these to match your trained checkpoint locations.
CKPT_ATTENTION_UNET="${REPO_ROOT}/artifacts/models/attention_pix2pix/inference.pt"
CKPT_RESTORMER="${REPO_ROOT}/artifacts/models/restormer/inference.pt"
CKPT_TRANSWEATHER="${REPO_ROOT}/artifacts/models/transweather/inference.pt"
CKPT_PROMPTIR="${REPO_ROOT}/artifacts/models/promptir/inference.pt"
CKPT_RAMIT="${REPO_ROOT}/artifacts/models/ramit/inference.pt"
CKPT_LITEWEATHERFORMER="${REPO_ROOT}/artifacts/models/liteweatherformer/inference.pt"

# Output root for all quantization artefacts.
OUTPUT_ROOT="${REPO_ROOT}/artifacts/quantization"

# Timing parameters.
N_WARMUP=10
N_TIMED=50

# ---------------------------------------------------------------------------
# Script path
# ---------------------------------------------------------------------------
QUANTIZE_SCRIPT="${REPO_ROOT}/quantization/scripts/quantize_compare.py"

# ---------------------------------------------------------------------------
# Run quantization for each model
# ---------------------------------------------------------------------------
declare -A CHECKPOINTS
CHECKPOINTS["attention_unet"]="${CKPT_ATTENTION_UNET}"
CHECKPOINTS["restormer"]="${CKPT_RESTORMER}"
CHECKPOINTS["transweather"]="${CKPT_TRANSWEATHER}"
CHECKPOINTS["promptir"]="${CKPT_PROMPTIR}"
CHECKPOINTS["ramit"]="${CKPT_RAMIT}"
CHECKPOINTS["liteweatherformer"]="${CKPT_LITEWEATHERFORMER}"

echo "===== Starting Quantization Benchmarks ====="
echo "Test image: ${TEST_IMAGE}"
echo "Output root: ${OUTPUT_ROOT}"
echo ""

for MODEL_NAME in attention_unet restormer transweather promptir ramit liteweatherformer; do
    CKPT="${CHECKPOINTS[$MODEL_NAME]}"
    OUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}"

    echo "--------------------------------------------------"
    echo "Model: ${MODEL_NAME}"
    echo "Checkpoint: ${CKPT}"
    echo "Output dir: ${OUT_DIR}"
    echo "--------------------------------------------------"

    # Validate checkpoint exists before running.
    if [ ! -f "${CKPT}" ]; then
        echo "[WARNING] Checkpoint not found: ${CKPT} — skipping ${MODEL_NAME}."
        continue
    fi

    python "${QUANTIZE_SCRIPT}" \
        --checkpoint "${CKPT}" \
        --image "${TEST_IMAGE}" \
        --output-dir "${OUT_DIR}" \
        --n-warmup "${N_WARMUP}" \
        --n-timed "${N_TIMED}"

    EXIT_CODE=$?
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "[ERROR] quantize_compare.py failed for ${MODEL_NAME} (exit code ${EXIT_CODE})."
    else
        echo "[OK] ${MODEL_NAME} complete — results at ${OUT_DIR}/quantization_report.json"
    fi

    echo ""
done

echo "===== All quantization benchmarks complete ====="
echo "Run quantization/scripts/generate_quantization_report.py to build the final report."
