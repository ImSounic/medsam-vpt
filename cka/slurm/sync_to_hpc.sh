#!/usr/bin/env bash
# Push committed code and configs to the HPC without rsync and without ever
# deleting anything on the remote side: git archive of the current branch,
# restricted to code paths, extracted over ~/medsam-vpt. Needs eduVPN.
# Usage from the repo root on the laptop: bash cka/slurm/sync_to_hpc.sh
set -euo pipefail
HOST="${HPC_HOST:-s3702111@hpc-head2.ewi.utwente.nl}"
DEST="${HPC_DEST:-medsam-vpt}"
cd "$(dirname "$0")/../.."
BRANCH="$(git branch --show-current)"
PATHS=(
  src scripts tests configs docs/superpowers
  cka/__init__.py cka/hooks.py cka/probe.py cka/generate_cka_configs.py
  cka/analysis cka/slurm
  bbox_robustness/eval_bbox_robust.py
  pyproject.toml requirements.txt
)
echo "[sync] branch $BRANCH -> $HOST:~/$DEST (${#PATHS[@]} paths, no deletions)"
git archive --format=tar "$BRANCH" -- "${PATHS[@]}" \
  | ssh "$HOST" "mkdir -p ~/$DEST && tar -xf - -C ~/$DEST && echo '[sync] extracted on' \$(hostname)"
echo "[sync] done. Next: ssh $HOST 'cd ~/$DEST && bash cka/slurm/submit_accv.sh'"
