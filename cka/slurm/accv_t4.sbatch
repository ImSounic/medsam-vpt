#!/usr/bin/env bash
#SBATCH --job-name=accv_t4
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:lovelace:1   # 48 GB L40/L40S only; 22 GB cards OOM on the CKA probe
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/accv_t4_%A_%a.out
#SBATCH --error=logs/accv_t4_%A_%a.err
# T4: retrain two seed-0 checkpoints lost from the HPC on 10 Sep 2026:
#   0: CKA late_l10 pm=0 (needed for the pm=0 bbox robustness sweep)
#   1: encoder-only LoRA r=28 (needed for the T3 detector dumps and budget curve)
set -euo pipefail
cd "$HOME/medsam-vpt"
mkdir -p logs
source /software/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate medsam-vpt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANG=en_US.UTF-8
export PYTHONUNBUFFERED=1   # stream [train] prints into the .out log

CONFIGS=(
  lora_cka_oodonly_late_l10_seed0.yaml
  lora_encoder_only.yaml
)
CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "[accv_t4] task $SLURM_ARRAY_TASK_ID -> $CFG on $(hostname) at $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m src.train --config "configs/$CFG"
