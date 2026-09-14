#!/usr/bin/env bash
#SBATCH --job-name=accv_t6_eval
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:lovelace:1   # 48 GB L40/L40S only
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/accv_t6_eval_%j.out
#SBATCH --error=logs/accv_t6_eval_%j.err
# T6 evals: tight boxes, one src.eval per checkpoint, appended to results/accv/runs_t6.csv.
set -euo pipefail
REPO="${REPO:-$HOME/medsam-vpt}"
cd "$REPO"
mkdir -p logs results/accv
source /software/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate medsam-vpt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANG=en_US.UTF-8
export PYTHONUNBUFFERED=1

python scripts/eval_all_checkpoints.py \
  --config configs/accv_eval.yaml \
  --checkpoint-glob 'checkpoints/runs_accv_t6/*/best.pth' \
  --eval-args "--results-csv results/accv/runs_t6.csv --per-image-dir results/accv/raw_t6"
echo "[accv_t6_eval] done at $(date -Iseconds)"
