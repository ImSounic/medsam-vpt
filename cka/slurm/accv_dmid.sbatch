#!/usr/bin/env bash
#SBATCH --job-name=accv_dmid
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:lovelace:1   # 48 GB L40/L40S only
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/accv_dmid_%j.out
#SBATCH --error=logs/accv_dmid_%j.err
# DMID held-out evaluation (spec section 7): zero-shot first (sanity check), then
# every available seed-0 and CKA checkpoint, tight boxes, with decoder drift.
# Set REPO to the medsam-vpt checkout to use (defaults to ~/medsam-vpt).
set -euo pipefail
REPO="${REPO:-$HOME/medsam-vpt}"
cd "$REPO"
mkdir -p logs results/accv
source /software/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate medsam-vpt
# Compute nodes have no internet: install once on the head node with
#   pip install tifffile imagecodecs
python -c "import tifffile, imagecodecs" || { echo "[accv_dmid] tifffile/imagecodecs missing in the env; install them on the head node first"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANG=en_US.UTF-8
export PYTHONUNBUFFERED=1

OUT_CSV=results/accv/runs_dmid.csv
RAW_DIR=results/accv/raw_dmid

echo "[accv_dmid] zero-shot"
python -m src.eval --config configs/accv_dmid_eval.yaml --drift --results-csv "$OUT_CSV" --per-image-dir "$RAW_DIR"

CHECKPOINTS=(
  checkpoints/runs/decoder_only_seed0/best.pth
  checkpoints/runs/vpt_shallow_seed0/best.pth
  checkpoints/runs/vpt_deep_seed0/best.pth
  checkpoints/runs/lora_seed0/best.pth
  checkpoints/runs_perfect_bboxes/lora_encoder_only_r28_all_seed0/best.pth
  checkpoints/runs/full_ft_seed0/best.pth
  checkpoints/runs_cka_oodonly_late/lora_cka_oodonly_late_l10_seed0/best.pth
  checkpoints/runs_accv_t1/lora_cka_oodonly_late_l10_pm20_seed1/best.pth
  checkpoints/runs_accv_t1/lora_cka_oodonly_late_l10_pm20_seed2/best.pth
  checkpoints/runs_accv_t1/lora_cka_oodonly_enc_l10_pm20_seed0/best.pth
  checkpoints/runs_accv_t1/lora_cka_oodonly_both_l10_pm20_seed0/best.pth
)
for CK in "${CHECKPOINTS[@]}"; do
  if [[ ! -f "$CK" ]]; then
    echo "[accv_dmid] WARNING: missing $CK, skipped"
    continue
  fi
  echo "[accv_dmid] $CK"
  python -m src.eval --config configs/accv_dmid_eval.yaml --checkpoint "$CK" --drift \
    --results-csv "$OUT_CSV" --per-image-dir "$RAW_DIR"
done
echo "[accv_dmid] done at $(date -Iseconds)"
