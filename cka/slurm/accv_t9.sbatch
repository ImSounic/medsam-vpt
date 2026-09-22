#!/usr/bin/env bash
#SBATCH --job-name=accv_t9
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:lovelace:1   # 48 GB L40/L40S only
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/accv_t9_%A_%a.out
#SBATCH --error=logs/accv_t9_%A_%a.err
# T9: retrain the seed-1 and seed-2 tight-box checkpoints of the five ladder
# methods (the originals were trained on JupyterLab and are not on the HPC).
# Writes checkpoints/runs_seed{1,2}/<method>_seed{1,2}/best.pth, the paths T8
# evaluates. Configs are the Phase-8 seed configs (6 epochs, decoder-only 10).
# About 1 to 2 h each on an L40S; submit T8 with --dependency=afterany on this.
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
  decoder_only_seed1.yaml
  vpt_shallow_seed1.yaml
  vpt_deep_seed1.yaml
  lora_seed1.yaml
  full_ft_seed1.yaml
  decoder_only_seed2.yaml
  vpt_shallow_seed2.yaml
  vpt_deep_seed2.yaml
  lora_seed2.yaml
  full_ft_seed2.yaml
)
CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "[accv_t9] task $SLURM_ARRAY_TASK_ID -> $CFG on $(hostname) at $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m src.train --config "configs/$CFG"
echo "[accv_t9] done at $(date -Iseconds)"
