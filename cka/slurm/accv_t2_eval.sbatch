#!/usr/bin/env bash
#SBATCH --job-name=accv_t2_eval
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/accv_t2_eval_%j.out
#SBATCH --error=logs/accv_t2_eval_%j.err
# T2 evals: tight boxes only, one src.eval per checkpoint (eval_all_methods.py
# would skip all but the first decoder_only checkpoint).
set -euo pipefail
cd "$HOME/medsam-vpt"
mkdir -p logs results/accv
source /software/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate medsam-vpt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANG=en_US.UTF-8

python scripts/eval_all_checkpoints.py \
  --config configs/accv_eval.yaml \
  --checkpoint-glob 'checkpoints/runs_accv_t2/*/best.pth' \
  --eval-args "--results-csv results/accv/runs_t2.csv --per-image-dir results/accv/raw_t2"
echo "[accv_t2_eval] done at $(date -Iseconds)"
