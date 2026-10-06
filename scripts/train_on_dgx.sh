#!/bin/bash

#SBATCH --job-name=aegis_train
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=40:00:00
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --mem=16G
#SBATCH --output=/scratch/%u/aegis_train-%j.out
#SBATCH --error=/scratch/%u/aegis_train-%j.err

. /etc/profile.d/modules.sh
module load anaconda/2023.03-1

# Activate conda
eval "$(conda shell.bash hook)"
conda activate pytorch_gpu1

echo "===== NODE INFO ====="
hostname
echo "====================="

echo "===== GPU INFO ====="
nvidia-smi
echo "===================="

# The first argument to the script will be the config file
CONFIG_FILE=$1

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file specified."
    echo "Usage: sbatch train_on_dgx.sh <path_to_config_yaml>"
    exit 1
fi

echo "Starting training with config: $CONFIG_FILE"

# Run the training script
python scripts/restoration/train.py --config "$CONFIG_FILE"

echo "Training completed."
