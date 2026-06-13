#!/usr/bin/env bash
# Full seed-1 + seed-2 train/eval pipeline. Run inside tmux; stops on first
# error and keeps one log per step.

set -euo pipefail

cd "$(dirname "$0")/.."

METHODS=(
    decoder_only
    vpt_shallow
    vpt_deep
    lora
    lora_encoder_only
    full_ft
)
REGIMES=(clean pm20 rand100)
BBOX_EVAL_SCRIPT="bbox_robustness/eval_bbox_robust.py"

section() {
    echo
    echo "============================================================"
    echo "[pipeline] $1  ($(date -Iseconds))"
    echo "============================================================"
}

regime_label() {
    case "$1" in
        clean) echo "pm=0" ;;
        pm20) echo "pm=20" ;;
        rand100) echo "rand100" ;;
        *)
            echo "unknown regime: $1" >&2
            return 1
            ;;
    esac
}

regime_suffix() {
    case "$1" in
        clean) echo "" ;;
        pm20) echo "_pm20" ;;
        rand100) echo "_rand100" ;;
        *)
            echo "unknown regime: $1" >&2
            return 1
            ;;
    esac
}

checkpoint_dir_for() {
    local seed="$1"
    local regime="$2"
    local suffix
    suffix="$(regime_suffix "$regime")"
    echo "checkpoints/runs_seed${seed}${suffix}"
}

results_csv_for() {
    local seed="$1"
    local regime="$2"
    local suffix
    suffix="$(regime_suffix "$regime")"
    echo "results/runs_seed${seed}${suffix}.csv"
}

train_one() {
    local cfg="$1"
    local log_dir="$2"
    local stem
    stem="$(basename "$cfg" .yaml)"
    echo "[pipeline] -> training $cfg"
    python -m src.train --config "configs/$cfg" 2>&1 | tee "$log_dir/$stem.log"
}

eval_standard() {
    local glob="$1"
    local out_csv="$2"
    local log_path="$3"
    echo "[pipeline] -> standard eval glob=$glob out=$out_csv"
    python scripts/eval_all_methods.py \
        --config configs/zero_shot.yaml \
        --checkpoint-glob "$glob" \
        --out-csv "$out_csv" 2>&1 | tee "$log_path"
}

eval_bbox() {
    local glob="$1"
    local out_dir="$2"
    local log_path="$3"
    echo "[pipeline] -> bbox eval glob=$glob out=$out_dir"
    python "$BBOX_EVAL_SCRIPT" \
        --config configs/zero_shot.yaml \
        --checkpoint-glob "$glob" \
        --n-samples 1 \
        --out-dir "$out_dir" 2>&1 | tee "$log_path"
}

train_regime() {
    local seed="$1"
    local regime="$2"
    local suffix
    local checkpoint_dir
    suffix="$(regime_suffix "$regime")"
    checkpoint_dir="$(checkpoint_dir_for "$seed" "$regime")"

    mkdir -p "$checkpoint_dir"
    section "SEED ${seed} - training $(regime_label "$regime") (${#METHODS[@]} methods)"
    for method in "${METHODS[@]}"; do
        train_one "${method}_seed${seed}${suffix}.yaml" "$checkpoint_dir"
    done
}

eval_regime() {
    local seed="$1"
    local regime="$2"
    local suffix
    local checkpoint_dir
    suffix="$(regime_suffix "$regime")"
    checkpoint_dir="$(checkpoint_dir_for "$seed" "$regime")"

    mkdir -p results
    section "SEED ${seed} - standard eval $(regime_label "$regime")"
    eval_standard \
        "${checkpoint_dir}/*/best.pth" \
        "$(results_csv_for "$seed" "$regime")" \
        "results/eval_seed${seed}${suffix}.log"
}

eval_bbox_regime() {
    local seed="$1"
    local regime="$2"
    local suffix
    local checkpoint_dir
    local out_dir
    suffix="$(regime_suffix "$regime")"
    checkpoint_dir="$(checkpoint_dir_for "$seed" "$regime")"
    out_dir="bbox_robustness/results_seed${seed}${suffix}"

    mkdir -p "$out_dir"
    eval_bbox \
        "${checkpoint_dir}/*/best.pth" \
        "$out_dir" \
        "$out_dir/eval.log"
}

echo "[pipeline] working dir: $(pwd)"
echo "[pipeline] started at:  $(date -Iseconds)"

for seed in 1 2; do
    for regime in "${REGIMES[@]}"; do
        train_regime "$seed" "$regime"
    done
done

for seed in 1 2; do
    for regime in "${REGIMES[@]}"; do
        eval_regime "$seed" "$regime"
    done
done

if [[ -f "$BBOX_EVAL_SCRIPT" ]]; then
    for seed in 1 2; do
        section "SEED ${seed} - bbox robustness eval"
        for regime in "${REGIMES[@]}"; do
            eval_bbox_regime "$seed" "$regime"
        done
    done
else
    section "SKIPPING bbox robustness eval"
    echo "[pipeline] ${BBOX_EVAL_SCRIPT} is not present in this trimmed repo."
    echo "[pipeline] Standard eval, aggregation, and paper visuals will still run."
fi

section "AGGREGATING across seeds"
python scripts/aggregate_seeds.py

section "RENDERING paper visuals"
python scripts/plot_paper_visuals.py

echo
echo "============================================================"
echo "[pipeline] DONE at $(date -Iseconds)"
echo "============================================================"
echo
echo "Generated files:"
echo "  results/summary_full_multiseed.csv"
echo "  results/summary_multiseed.md"
echo "  figures/paper_visuals/degradation_curves.png"
echo "  figures/paper_visuals/degradation_heatmap.png"
echo "  figures/paper_visuals/performance_histograms.png"
