#!/usr/bin/env bash
#SBATCH --job-name=accv_t2
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/accv_t2_%A_%a.out
#SBATCH --error=logs/accv_t2_%A_%a.err
# T2: adaptation budget sweep, 3 methods x {50, 250, 1000} images, 6000 steps each.
set -euo pipefail
cd "$HOME/medsam-vpt"
mkdir -p logs
source /software/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate medsam-vpt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANG=en_US.UTF-8
export PYTHONUNBUFFERED=1   # stream [train] prints into the .out log

CONFIGS=(
  budget/lora_n50_seed0.yaml
  budget/lora_n250_seed0.yaml
  budget/lora_n1000_seed0.yaml
  budget/decoder_only_n50_seed0.yaml
  budget/decoder_only_n250_seed0.yaml
  budget/decoder_only_n1000_seed0.yaml
  budget/lora_encoder_only_n50_seed0.yaml
  budget/lora_encoder_only_n250_seed0.yaml
  budget/lora_encoder_only_n1000_seed0.yaml
)
CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "[accv_t2] task $SLURM_ARRAY_TASK_ID -> $CFG on $(hostname) at $(date -Iseconds)"
python -m src.train --config "configs/$CFG"
