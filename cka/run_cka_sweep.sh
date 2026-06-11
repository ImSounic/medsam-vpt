#!/usr/bin/env bash
# Run CKA-aware LoRA sweep: 3 positions x 3 lambdas = 9 trainings + 3 evals + aggregation (~93h on A10); halts on first error, per-step logs via tee.

set -e
set -u
set -o pipefail

cd "$(dirname "$0")/.."   # repo root
echo "[pipeline] working dir: $(pwd)"
echo "[pipeline] started at:  $(date -Iseconds)"
echo

# Output directories
mkdir -p \
    checkpoints/runs_cka_early checkpoints/runs_cka_mid checkpoints/runs_cka_late \
    cka/results cka/figures

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

# Per-position eval runner. Usage: eval_position <position>
eval_position () {
    local position="$1"
    local glob="checkpoints/runs_cka_${position}/*/best.pth"
    local out_csv="cka/results/runs_cka_${position}.csv"
    local log_path="cka/results/eval_cka_${position}.log"
    echo "[pipeline] -> eval position=$position glob=$glob out=$out_csv"
    python scripts/eval_all_methods.py \
        --config configs/zero_shot.yaml \
        --checkpoint-glob "$glob" \
        --out-csv "$out_csv" 2>&1 | tee "$log_path"
}

# Trainings: 3 positions x 3 lambdas = 9 runs
section "CKA SWEEP - early position (3 lambdas)"
train lora_cka_early_l01_seed0.yaml  checkpoints/runs_cka_early
train lora_cka_early_l1_seed0.yaml   checkpoints/runs_cka_early
train lora_cka_early_l10_seed0.yaml  checkpoints/runs_cka_early

section "CKA SWEEP - mid position (3 lambdas)"
train lora_cka_mid_l01_seed0.yaml    checkpoints/runs_cka_mid
train lora_cka_mid_l1_seed0.yaml     checkpoints/runs_cka_mid
train lora_cka_mid_l10_seed0.yaml    checkpoints/runs_cka_mid

section "CKA SWEEP - late position (3 lambdas)"
train lora_cka_late_l01_seed0.yaml   checkpoints/runs_cka_late
train lora_cka_late_l1_seed0.yaml    checkpoints/runs_cka_late
train lora_cka_late_l10_seed0.yaml   checkpoints/runs_cka_late

# Evals: one invocation per position covers all 3 lambdas via glob
section "CKA SWEEP - standard tight-bbox eval (3 invocations)"
eval_position early
eval_position mid
eval_position late

# Aggregate across positions x lambdas
section "AGGREGATING CKA sweep results"
python cka/analysis/aggregate_cka_sweep.py

echo
echo "============================================================"
echo "[pipeline] DONE at $(date -Iseconds)"
echo "============================================================"
echo
echo "Generated files:"
echo "  cka/results/cka_sweep_summary.csv"
echo "  cka/results/cka_sweep_summary.md"
echo "  cka/figures/cka_sweep_grid.png"
echo "  cka/figures/cka_id_vs_far_ood_tradeoff.png"
echo
echo "Next: commit + push the new eval CSVs and aggregated outputs."
