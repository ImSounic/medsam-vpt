#!/usr/bin/env bash
#SBATCH --job-name=accv_t4_eval
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:lovelace:1   # 48 GB L40/L40S only; 22 GB cards OOM on the CKA probe
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/accv_t4_eval_%j.out
#SBATCH --error=logs/accv_t4_eval_%j.err
# T4 evals after the retrains: pm=0 bbox robustness sweep of the CKA pm=0
# checkpoint (sixth curve), tight-box eval of the encoder-only checkpoint, and
# its T3 detector dumps at 0 and 50 px (appended to the existing T3 CSVs).
set -euo pipefail
cd "$HOME/medsam-vpt"
mkdir -p logs results/accv
source /software/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate medsam-vpt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANG=en_US.UTF-8
export PYTHONUNBUFFERED=1

PM0="checkpoints/runs_cka_oodonly_late/lora_cka_oodonly_late_l10_seed0/best.pth"
ENC="checkpoints/runs_perfect_bboxes/lora_encoder_only_r28_all_seed0/best.pth"

if [[ -f "$PM0" ]]; then
  python bbox_robustness/eval_bbox_robust.py \
    --config configs/zero_shot.yaml \
    --checkpoint-glob "$PM0" \
    --n-samples 1 \
    --out-dir bbox_robustness/results_cka_oodonly_late_l10_pm0
else
  echo "[accv_t4_eval] WARNING: $PM0 not found; pm=0 robustness sweep skipped"
fi

if [[ -f "$ENC" ]]; then
  python -m src.eval --config configs/accv_eval.yaml --checkpoint "$ENC" \
    --results-csv results/accv/runs_encoder_only_seed0.csv --per-image-dir results/accv/raw_encoder_only
  for PM in 0 50; do
    python -m src.eval --config configs/accv_eval.yaml --checkpoint "$ENC" --drift \
      --bbox-perturb "$PM" --results-csv "results/accv/runs_t3_pm${PM}.csv" \
      --per-image-dir "results/accv/raw_t3_pm${PM}"
  done
else
  echo "[accv_t4_eval] WARNING: $ENC not found; encoder-only evals skipped"
fi
echo "[accv_t4_eval] done at $(date -Iseconds)"
