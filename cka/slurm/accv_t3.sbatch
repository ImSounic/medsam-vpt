#!/usr/bin/env bash
#SBATCH --job-name=accv_t3
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/accv_t3_%A_%a.out
#SBATCH --error=logs/accv_t3_%A_%a.err
# T3: iou_pred and decoder-drift dumps for the seven seed-0 pm=0 methods on the
# four-dataset ladder, at tight boxes (task 0) and 50 px jitter (task 1).
set -euo pipefail
cd "$HOME/medsam-vpt"
mkdir -p logs results/accv
source /software/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate medsam-vpt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANG=en_US.UTF-8
export PYTHONUNBUFFERED=1   # stream [train] prints into the .out log

PERTURBS=(0 50)
PM="${PERTURBS[$SLURM_ARRAY_TASK_ID]}"
OUT_CSV="results/accv/runs_t3_pm${PM}.csv"
RAW_DIR="results/accv/raw_t3_pm${PM}"

CHECKPOINTS=(
  checkpoints/runs/decoder_only_seed0/best.pth
  checkpoints/runs/vpt_shallow_seed0/best.pth
  checkpoints/runs/vpt_deep_seed0/best.pth
  checkpoints/runs/lora_seed0/best.pth
  checkpoints/runs_perfect_bboxes/lora_encoder_only_r28_all_seed0/best.pth
  checkpoints/runs/full_ft_seed0/best.pth
)

echo "[accv_t3] pm=$PM zero_shot"
python -m src.eval --config configs/accv_eval.yaml --drift --bbox-perturb "$PM" \
  --results-csv "$OUT_CSV" --per-image-dir "$RAW_DIR"
for CK in "${CHECKPOINTS[@]}"; do
  if [[ ! -f "$CK" ]]; then
    echo "[accv_t3] WARNING: missing $CK, skipped"
    continue
  fi
  echo "[accv_t3] pm=$PM $CK"
  python -m src.eval --config configs/accv_eval.yaml --checkpoint "$CK" --drift \
    --bbox-perturb "$PM" --results-csv "$OUT_CSV" --per-image-dir "$RAW_DIR"
done
echo "[accv_t3] done at $(date -Iseconds)"
