#!/usr/bin/env bash
#SBATCH --job-name=accv_t10
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:lovelace:1   # 48 GB L40/L40S only
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/accv_t10_%A_%a.out
#SBATCH --error=logs/accv_t10_%A_%a.err
# T10 (camera-ready controls requested by reviewers), tight boxes, seeds 0-2:
#   lora_r8_frozendec: the standard rank-8 QKV LoRA with the mask decoder frozen,
#     so it differs from standard LoRA only in decoder training (kTc3 W1, SNjd).
#   lora_declr1e4: standard LoRA with the mask decoder at 1e-4, the decoder-only
#     LR, instead of 5e-4 (SNjd learning-rate confound).
# Each task trains, then evaluates its checkpoint on the ladder and on DMID with
# decoder drift. Per-task CSVs avoid concurrent appends.
# Set REPO to the medsam-vpt checkout to use (defaults to ~/medsam-vpt).
set -euo pipefail
REPO="${REPO:-$HOME/medsam-vpt}"
cd "$REPO"
mkdir -p logs results/accv
source /software/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate medsam-vpt
python -c "import tifffile, imagecodecs" || { echo "[accv_t10] tifffile/imagecodecs missing in the env; install them on the head node first"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANG=en_US.UTF-8
export PYTHONUNBUFFERED=1

CONFIGS=(
  lora_r8_frozendec_seed0.yaml
  lora_r8_frozendec_seed1.yaml
  lora_r8_frozendec_seed2.yaml
  lora_declr1e4_seed0.yaml
  lora_declr1e4_seed1.yaml
  lora_declr1e4_seed2.yaml
)
CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
NAME="${CFG%.yaml}"
CKPT="checkpoints/runs_accv_t10/${NAME}/best.pth"
echo "[accv_t10] task $SLURM_ARRAY_TASK_ID -> $CFG on $(hostname) at $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ ! -f "$CKPT" ]]; then
  python -m src.train --config "configs/$CFG"
fi

python -m src.eval --config configs/accv_eval.yaml --checkpoint "$CKPT" --drift \
  --results-csv "results/accv/runs_t10_${NAME}.csv" --per-image-dir results/accv/raw_t10
python -m src.eval --config configs/accv_dmid_eval.yaml --checkpoint "$CKPT" --drift \
  --results-csv "results/accv/runs_t10_dmid_${NAME}.csv" --per-image-dir results/accv/raw_t10_dmid
echo "[accv_t10] done at $(date -Iseconds)"
