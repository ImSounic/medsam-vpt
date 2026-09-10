#!/usr/bin/env bash
#SBATCH --job-name=accv_t1_eval
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/accv_t1_eval_%j.out
#SBATCH --error=logs/accv_t1_eval_%j.err
# T1 evals: tight-box eval of the four new checkpoints, bbox robustness sweep
# (20/50/100/200 px) for them, and the never-run sweep of the pm=0 late_l10 checkpoint.
set -euo pipefail
cd "$HOME/medsam-vpt"
mkdir -p logs cka/results
source /software/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate medsam-vpt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANG=en_US.UTF-8

python scripts/eval_all_checkpoints.py \
  --config configs/accv_eval.yaml \
  --checkpoint-glob 'checkpoints/runs_accv_t1/*/best.pth' \
  --eval-args "--results-csv cka/results/runs_accv_t1.csv --per-image-dir results/accv/raw_t1"

python bbox_robustness/eval_bbox_robust.py \
  --config configs/zero_shot.yaml \
  --checkpoint-glob 'checkpoints/runs_accv_t1/*/best.pth' \
  --n-samples 1 \
  --out-dir bbox_robustness/results_accv_t1

PM0="checkpoints/runs_cka_oodonly_late/lora_cka_oodonly_late_l10_seed0/best.pth"
if [[ -f "$PM0" ]]; then
  python bbox_robustness/eval_bbox_robust.py \
    --config configs/zero_shot.yaml \
    --checkpoint-glob "$PM0" \
    --n-samples 1 \
    --out-dir bbox_robustness/results_cka_oodonly_late_l10_pm0
else
  echo "[accv_t1_eval] WARNING: $PM0 not found; pm=0 robustness sweep skipped"
fi
echo "[accv_t1_eval] done at $(date -Iseconds)"
