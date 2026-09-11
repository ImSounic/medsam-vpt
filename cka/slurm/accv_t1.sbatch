#!/usr/bin/env bash
#SBATCH --job-name=accv_t1
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:lovelace:1   # 48 GB L40/L40S only; 22 GB cards OOM on the CKA probe
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/accv_t1_%A_%a.out
#SBATCH --error=logs/accv_t1_%A_%a.err
# T1: CKA multi-seed (seeds 1, 2) and hook-placement ablation (enc, both).
# About 6 h each on an L40S; QOS runs two at a time.
set -euo pipefail
cd "$HOME/medsam-vpt"
mkdir -p logs
source /software/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate medsam-vpt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANG=en_US.UTF-8
export PYTHONUNBUFFERED=1   # stream [train] prints into the .out log

CONFIGS=(
  lora_cka_oodonly_late_l10_pm20_seed1.yaml
  lora_cka_oodonly_late_l10_pm20_seed2.yaml
  lora_cka_oodonly_enc_l10_pm20_seed0.yaml
  lora_cka_oodonly_both_l10_pm20_seed0.yaml
)
CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "[accv_t1] task $SLURM_ARRAY_TASK_ID -> $CFG on $(hostname) at $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m src.train --config "configs/$CFG"
