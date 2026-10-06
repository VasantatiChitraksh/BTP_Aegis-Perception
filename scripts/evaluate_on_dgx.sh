#!/bin/bash

#SBATCH --job-name=aegis_eval
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --mem=16G
#SBATCH --output=/scratch/%u/aegis_eval-%j.out
#SBATCH --error=/scratch/%u/aegis_eval-%j.err

. /etc/profile.d/modules.sh
module load anaconda/2023.03-1

# Activate conda
eval "$(conda shell.bash hook)"
conda activate pytorch_gpu1

CONFIG_FILE=$1
CHECKPOINT_FILE=$2

if [ -z "$CONFIG_FILE" ] || [ -z "$CHECKPOINT_FILE" ]; then
    echo "Error: Config or Checkpoint file not specified."
    echo "Usage: sbatch evaluate_on_dgx.sh <path_to_config_yaml> <path_to_checkpoint_pt>"
    exit 1
fi

echo "Starting evaluation with config: $CONFIG_FILE and checkpoint: $CHECKPOINT_FILE"

# Run the evaluation script
python scripts/restoration/evaluate.py --config "$CONFIG_FILE" --checkpoint "$CHECKPOINT_FILE" --split test

echo "Evaluation completed."
