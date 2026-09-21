#!/usr/bin/env bash
#SBATCH --job-name=accv_t8
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:lovelace:1   # 48 GB L40/L40S only
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/accv_t8_%A_%a.out
#SBATCH --error=logs/accv_t8_%A_%a.err
# T8: iou_pred and decoder-drift dumps for the seed-1 and seed-2 tight-box
# checkpoints (task 0 = seed 1, task 1 = seed 2), on the four-dataset ladder and
# on DMID, so the failure-detection table can carry a std over three seeds.
# Seed-0 dumps already exist (T3: raw_t3_pm0, DMID job: raw_dmid).
# Prompt levels default to tight boxes; PERTURBS="0 50" adds the 50 px dumps.
# Set REPO to the medsam-vpt checkout to use (defaults to ~/medsam-vpt).
set -euo pipefail
REPO="${REPO:-$HOME/medsam-vpt}"
cd "$REPO"
mkdir -p logs results/accv
source /software/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate medsam-vpt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANG=en_US.UTF-8
export PYTHONUNBUFFERED=1

SEEDS=(1 2)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
PERTURBS="${PERTURBS:-0}"

CHECKPOINTS=(
  checkpoints/runs_seed${SEED}/decoder_only_seed${SEED}/best.pth
  checkpoints/runs_seed${SEED}/vpt_shallow_seed${SEED}/best.pth
  checkpoints/runs_seed${SEED}/vpt_deep_seed${SEED}/best.pth
  checkpoints/runs_seed${SEED}/lora_seed${SEED}/best.pth
  checkpoints/runs_perfect_bboxes/lora_encoder_only_r28_all_seed${SEED}/best.pth
  checkpoints/runs_seed${SEED}/full_ft_seed${SEED}/best.pth
)

for PM in $PERTURBS; do
  OUT_CSV="results/accv/runs_t8_pm${PM}.csv"
  RAW_DIR="results/accv/raw_t8_pm${PM}"
  for CK in "${CHECKPOINTS[@]}"; do
    if [[ ! -f "$CK" ]]; then
      echo "[accv_t8] WARNING: missing $CK, skipped"
      continue
    fi
    echo "[accv_t8] seed=$SEED pm=$PM $CK"
    python -m src.eval --config configs/accv_eval.yaml --checkpoint "$CK" --drift \
      --bbox-perturb "$PM" --results-csv "$OUT_CSV" --per-image-dir "$RAW_DIR"
  done
done

# DMID (tight boxes only). Needs tifffile/imagecodecs in the env (head-node install).
if python -c "import tifffile, imagecodecs" 2>/dev/null; then
  OUT_CSV="results/accv/runs_t8_dmid.csv"
  RAW_DIR="results/accv/raw_t8_dmid"
  for CK in "${CHECKPOINTS[@]}"; do
    [[ -f "$CK" ]] || continue
    echo "[accv_t8] seed=$SEED dmid $CK"
    python -m src.eval --config configs/accv_dmid_eval.yaml --checkpoint "$CK" --drift \
      --results-csv "$OUT_CSV" --per-image-dir "$RAW_DIR"
  done
else
  echo "[accv_t8] WARNING: tifffile/imagecodecs missing; DMID skipped"
fi
echo "[accv_t8] done at $(date -Iseconds)"
