#!/usr/bin/env bash
# Push code and configs (no data, checkpoints or results) to the HPC.
# Needs eduVPN. Usage from the repo root on the laptop: bash cka/slurm/sync_to_hpc.sh
set -euo pipefail
HOST="${HPC_HOST:-s3702111@hpc-head2.ewi.utwente.nl}"
DEST="${HPC_DEST:-~/medsam-vpt/}"
cd "$(dirname "$0")/../.."
rsync -avz --delete-excluded \
  --exclude '.git' --exclude 'data' --exclude 'checkpoints' --exclude 'results' \
  --exclude 'bbox_robustness/results*' --exclude 'logs' --exclude '__pycache__' \
  --exclude 'figures' --exclude 'colab' --exclude '*.zip' \
  ./ "$HOST:$DEST"
echo "[sync] done. Next: ssh $HOST 'cd ~/medsam-vpt && bash cka/slurm/submit_accv.sh'"
