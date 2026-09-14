#!/usr/bin/env bash
#SBATCH --job-name=accv_t7
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:lovelace:1   # 48 GB L40/L40S only
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/accv_t7_%j.out
#SBATCH --error=logs/accv_t7_%j.err
# T7: representation-drift dumps (decoder upscaled embedding and encoder neck,
# 1 - CKA to base) for the nine budget checkpoints and the three full-data seed-0
# checkpoints, tight boxes, four datasets. Answers the reviewer request for drift
# curves across subset sizes.
set -euo pipefail
REPO="${REPO:-$HOME/medsam-vpt}"
cd "$REPO"
mkdir -p logs results/accv
source /software/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate medsam-vpt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANG=en_US.UTF-8
export PYTHONUNBUFFERED=1

OUT_CSV=results/accv/runs_t7.csv
RAW_DIR=results/accv/raw_t7

CHECKPOINTS=(
  checkpoints/runs_accv_t2/lora_n50_seed0/best.pth
  checkpoints/runs_accv_t2/lora_n250_seed0/best.pth
  checkpoints/runs_accv_t2/lora_n1000_seed0/best.pth
  checkpoints/runs_accv_t2/decoder_only_n50_seed0/best.pth
  checkpoints/runs_accv_t2/decoder_only_n250_seed0/best.pth
  checkpoints/runs_accv_t2/decoder_only_n1000_seed0/best.pth
  checkpoints/runs_accv_t2/lora_encoder_only_n50_seed0/best.pth
  checkpoints/runs_accv_t2/lora_encoder_only_n250_seed0/best.pth
  checkpoints/runs_accv_t2/lora_encoder_only_n1000_seed0/best.pth
  checkpoints/runs/lora_seed0/best.pth
  checkpoints/runs/decoder_only_seed0/best.pth
  checkpoints/runs_perfect_bboxes/lora_encoder_only_r28_all_seed0/best.pth
)
for CK in "${CHECKPOINTS[@]}"; do
  if [[ ! -f "$CK" ]]; then
    echo "[accv_t7] WARNING: missing $CK, skipped"
    continue
  fi
  echo "[accv_t7] $CK"
  python -m src.eval --config configs/accv_eval.yaml --checkpoint "$CK" --drift \
    --results-csv "$OUT_CSV" --per-image-dir "$RAW_DIR"
done
echo "[accv_t7] done at $(date -Iseconds)"
