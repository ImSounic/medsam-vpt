#!/usr/bin/env bash
#SBATCH --job-name=accv_t6
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:lovelace:1   # 48 GB L40/L40S only
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-17
#SBATCH --output=logs/accv_t6_%A_%a.out
#SBATCH --error=logs/accv_t6_%A_%a.err
# T6: adaptation-budget sweep, seeds 1 and 2 (same image subsets as seed 0).
# Set REPO to the medsam-vpt checkout to use (defaults to ~/medsam-vpt).
set -euo pipefail
REPO="${REPO:-$HOME/medsam-vpt}"
cd "$REPO"
mkdir -p logs
source /software/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate medsam-vpt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANG=en_US.UTF-8
export PYTHONUNBUFFERED=1

CONFIGS=(
  budget/lora_n50_seed1.yaml
  budget/lora_n250_seed1.yaml
  budget/lora_n1000_seed1.yaml
  budget/decoder_only_n50_seed1.yaml
  budget/decoder_only_n250_seed1.yaml
  budget/decoder_only_n1000_seed1.yaml
  budget/lora_encoder_only_n50_seed1.yaml
  budget/lora_encoder_only_n250_seed1.yaml
  budget/lora_encoder_only_n1000_seed1.yaml
  budget/lora_n50_seed2.yaml
  budget/lora_n250_seed2.yaml
  budget/lora_n1000_seed2.yaml
  budget/decoder_only_n50_seed2.yaml
  budget/decoder_only_n250_seed2.yaml
  budget/decoder_only_n1000_seed2.yaml
  budget/lora_encoder_only_n50_seed2.yaml
  budget/lora_encoder_only_n250_seed2.yaml
  budget/lora_encoder_only_n1000_seed2.yaml
)
CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "[accv_t6] task $SLURM_ARRAY_TASK_ID -> $CFG on $(hostname) at $(date -Iseconds)"
python -m src.train --config "configs/$CFG"
