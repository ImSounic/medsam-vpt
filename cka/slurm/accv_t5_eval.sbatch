#!/usr/bin/env bash
#SBATCH --job-name=accv_t5_eval
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:lovelace:1   # 48 GB L40/L40S only; 22 GB cards OOM on the CKA probe
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/accv_t5_eval_%j.out
#SBATCH --error=logs/accv_t5_eval_%j.err
# T5 evals: tight-box eval and bbox sweep of the four T5 checkpoints, plus a
# tight-box eval of the retrained CKA pm=0 checkpoint so its robustness curve
# comes from a single model.
set -euo pipefail
cd "$HOME/medsam-vpt"
mkdir -p logs cka/results results/accv
source /software/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate medsam-vpt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANG=en_US.UTF-8
export PYTHONUNBUFFERED=1

PM0="checkpoints/runs_cka_oodonly_late/lora_cka_oodonly_late_l10_seed0/best.pth"
if [[ -f "$PM0" ]]; then
  python -m src.eval --config configs/accv_eval.yaml --checkpoint "$PM0" \
    --results-csv results/accv/runs_cka_pm0_retrained.csv --per-image-dir results/accv/raw_cka_pm0_retrained
fi

python scripts/eval_all_checkpoints.py \
  --config configs/accv_eval.yaml \
  --checkpoint-glob 'checkpoints/runs_accv_t5/*/best.pth' \
  --eval-args "--results-csv cka/results/runs_accv_t5.csv --per-image-dir results/accv/raw_t5"

python bbox_robustness/eval_bbox_robust.py \
  --config configs/zero_shot.yaml \
  --checkpoint-glob 'checkpoints/runs_accv_t5/*/best.pth' \
  --n-samples 1 \
  --out-dir bbox_robustness/results_accv_t5
echo "[accv_t5_eval] done at $(date -Iseconds)"
