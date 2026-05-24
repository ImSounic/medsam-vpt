#!/usr/bin/env bash
# Run the full seed-1 + seed-2 training + eval pipeline.
#
# Estimated wall-clock on A10:
#   Seed 1 training (15 methods): ~16h
#   Seed 2 training (15 methods): ~16h
#   Seed 1 + Seed 2 evals (12 invocations total): ~17h
#   Aggregation: ~30s
#   ----------------------------------------------------
#   Total: ~50h
#
# Designed to be run inside tmux so it survives terminal disconnects:
#   tmux new -s seeds
#   bash scripts/run_seed_pipeline.sh
#   # Ctrl-b then d to detach
#   # tmux attach -t seeds to reattach
#
# Pipeline stops on first error (set -e). Per-step logs are persisted via tee
# so a failure can be diagnosed without re-running upstream successful steps.

set -e
set -u
set -o pipefail

cd "$(dirname "$0")/.."   # repo root
echo "[pipeline] working dir: $(pwd)"
echo "[pipeline] started at:  $(date -Iseconds)"
echo

# ----------------------------------------------------------------------------
# Output directories
# ----------------------------------------------------------------------------
mkdir -p \
    checkpoints/runs_seed1 checkpoints/runs_seed1_pm20 checkpoints/runs_seed1_rand100 \
    checkpoints/runs_seed2 checkpoints/runs_seed2_pm20 checkpoints/runs_seed2_rand100 \
    bbox_robustness/results_seed1 bbox_robustness/results_seed1_pm20 bbox_robustness/results_seed1_rand100 \
    bbox_robustness/results_seed2 bbox_robustness/results_seed2_pm20 bbox_robustness/results_seed2_rand100

# Helper for nicer section markers
section () {
    echo
    echo "============================================================"
    echo "[pipeline] $1  ($(date -Iseconds))"
    echo "============================================================"
}

# Per-step training runner. Usage: train <config_name> <log_dir>
train () {
    local cfg="$1"
    local log_dir="$2"
    local stem
    stem="$(basename "$cfg" .yaml)"
    echo "[pipeline] -> training $cfg (log: $log_dir/$stem.log)"
    python -m src.train --config "configs/$cfg" 2>&1 | tee "$log_dir/$stem.log"
}

# Per-step eval runners.
eval_standard () {
    local glob="$1"
    local out_csv="$2"
    local log_path="$3"
    echo "[pipeline] -> standard eval glob=$glob out=$out_csv"
    python scripts/eval_all_methods.py \
        --config configs/zero_shot.yaml \
        --checkpoint-glob "$glob" \
        --out-csv "$out_csv" 2>&1 | tee "$log_path"
}

eval_bbox () {
    local glob="$1"
    local out_dir="$2"
    local log_path="$3"
    echo "[pipeline] -> bbox-robustness eval glob=$glob out=$out_dir"
    python bbox_robustness/eval_bbox_robust.py \
        --config configs/zero_shot.yaml \
        --checkpoint-glob "$glob" \
        --n-samples 1 \
        --out-dir "$out_dir" 2>&1 | tee "$log_path"
}

# ----------------------------------------------------------------------------
# Seed 1 — training (15 methods, ~16h on A10)
# ----------------------------------------------------------------------------
section "SEED 1 — training pm=0 (5 methods)"
train decoder_only_seed1.yaml  checkpoints/runs_seed1
train vpt_shallow_seed1.yaml   checkpoints/runs_seed1
train vpt_deep_seed1.yaml      checkpoints/runs_seed1
train lora_seed1.yaml          checkpoints/runs_seed1
train full_ft_seed1.yaml       checkpoints/runs_seed1

section "SEED 1 — training pm=20 (5 methods)"
train decoder_only_seed1_pm20.yaml  checkpoints/runs_seed1_pm20
train vpt_shallow_seed1_pm20.yaml   checkpoints/runs_seed1_pm20
train vpt_deep_seed1_pm20.yaml      checkpoints/runs_seed1_pm20
train lora_seed1_pm20.yaml          checkpoints/runs_seed1_pm20
train full_ft_seed1_pm20.yaml       checkpoints/runs_seed1_pm20

section "SEED 1 — training rand100 (5 methods)"
train decoder_only_seed1_rand100.yaml  checkpoints/runs_seed1_rand100
train vpt_shallow_seed1_rand100.yaml   checkpoints/runs_seed1_rand100
train vpt_deep_seed1_rand100.yaml      checkpoints/runs_seed1_rand100
train lora_seed1_rand100.yaml          checkpoints/runs_seed1_rand100
train full_ft_seed1_rand100.yaml       checkpoints/runs_seed1_rand100

# ----------------------------------------------------------------------------
# Seed 2 — training (15 methods, ~16h on A10)
# ----------------------------------------------------------------------------
section "SEED 2 — training pm=0 (5 methods)"
train decoder_only_seed2.yaml  checkpoints/runs_seed2
train vpt_shallow_seed2.yaml   checkpoints/runs_seed2
train vpt_deep_seed2.yaml      checkpoints/runs_seed2
train lora_seed2.yaml          checkpoints/runs_seed2
train full_ft_seed2.yaml       checkpoints/runs_seed2

section "SEED 2 — training pm=20 (5 methods)"
train decoder_only_seed2_pm20.yaml  checkpoints/runs_seed2_pm20
train vpt_shallow_seed2_pm20.yaml   checkpoints/runs_seed2_pm20
train vpt_deep_seed2_pm20.yaml      checkpoints/runs_seed2_pm20
train lora_seed2_pm20.yaml          checkpoints/runs_seed2_pm20
train full_ft_seed2_pm20.yaml       checkpoints/runs_seed2_pm20

section "SEED 2 — training rand100 (5 methods)"
train decoder_only_seed2_rand100.yaml  checkpoints/runs_seed2_rand100
train vpt_shallow_seed2_rand100.yaml   checkpoints/runs_seed2_rand100
train vpt_deep_seed2_rand100.yaml      checkpoints/runs_seed2_rand100
train lora_seed2_rand100.yaml          checkpoints/runs_seed2_rand100
train full_ft_seed2_rand100.yaml       checkpoints/runs_seed2_rand100

# ----------------------------------------------------------------------------
# Seed 1 — eval (~8.5h on A10)
# ----------------------------------------------------------------------------
section "SEED 1 — standard tight-bbox eval (3 invocations)"
eval_standard 'checkpoints/runs_seed1/*/best.pth' \
              results/runs_seed1.csv \
              results/eval_seed1.log
eval_standard 'checkpoints/runs_seed1_pm20/*/best.pth' \
              results/runs_seed1_pm20.csv \
              results/eval_seed1_pm20.log
eval_standard 'checkpoints/runs_seed1_rand100/*/best.pth' \
              results/runs_seed1_rand100.csv \
              results/eval_seed1_rand100.log

section "SEED 1 — bbox robustness eval (3 invocations)"
eval_bbox 'checkpoints/runs_seed1/*/best.pth' \
          bbox_robustness/results_seed1 \
          bbox_robustness/results_seed1/eval.log
eval_bbox 'checkpoints/runs_seed1_pm20/*/best.pth' \
          bbox_robustness/results_seed1_pm20 \
          bbox_robustness/results_seed1_pm20/eval.log
eval_bbox 'checkpoints/runs_seed1_rand100/*/best.pth' \
          bbox_robustness/results_seed1_rand100 \
          bbox_robustness/results_seed1_rand100/eval.log

# ----------------------------------------------------------------------------
# Seed 2 — eval (~8.5h on A10)
# ----------------------------------------------------------------------------
section "SEED 2 — standard tight-bbox eval (3 invocations)"
eval_standard 'checkpoints/runs_seed2/*/best.pth' \
              results/runs_seed2.csv \
              results/eval_seed2.log
eval_standard 'checkpoints/runs_seed2_pm20/*/best.pth' \
              results/runs_seed2_pm20.csv \
              results/eval_seed2_pm20.log
eval_standard 'checkpoints/runs_seed2_rand100/*/best.pth' \
              results/runs_seed2_rand100.csv \
              results/eval_seed2_rand100.log

section "SEED 2 — bbox robustness eval (3 invocations)"
eval_bbox 'checkpoints/runs_seed2/*/best.pth' \
          bbox_robustness/results_seed2 \
          bbox_robustness/results_seed2/eval.log
eval_bbox 'checkpoints/runs_seed2_pm20/*/best.pth' \
          bbox_robustness/results_seed2_pm20 \
          bbox_robustness/results_seed2_pm20/eval.log
eval_bbox 'checkpoints/runs_seed2_rand100/*/best.pth' \
          bbox_robustness/results_seed2_rand100 \
          bbox_robustness/results_seed2_rand100/eval.log

# ----------------------------------------------------------------------------
# Aggregate everything
# ----------------------------------------------------------------------------
section "AGGREGATING across seeds"
python scripts/aggregate_seeds.py

echo
echo "============================================================"
echo "[pipeline] DONE at $(date -Iseconds)"
echo "============================================================"
echo
echo "Generated files:"
echo "  summary_full_multiseed.csv"
echo "  bbox_robustness/comparison/seed_summary_table.md"
echo "  bbox_robustness/comparison/seed_error_bars.png"
echo
echo "Next: commit + push the new eval CSVs and aggregated outputs."
