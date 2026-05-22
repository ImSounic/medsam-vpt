"""Plot bbox-robustness degradation curves.

Reads:
  bbox_robustness/results/runs.csv     (this study's perturbed evals, levels 20/50/100/200)
  ../results/runs.csv                  (main eval — used for the 0-px tight baseline)

Produces:
  bbox_robustness/results/figures/degradation_curves.png   # 4 panels (one per dataset)
  bbox_robustness/results/figures/degradation_heatmap.png  # method x perturb, faceted by dataset
  bbox_robustness/results/figures/relative_drop.png        # % drop from 0-px to 200-px

Plus a pivoted summary CSV:
  bbox_robustness/results/summary.csv

Run after eval_bbox_robust.py finishes (so runs.csv exists).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "bbox_robustness" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
BASELINE_RUNS_CSV = REPO_ROOT / "results" / "runs.csv"

# Method display order + colours (consistent with the main plots.py)
METHOD_ORDER = [
    "zero_shot",
    "decoder_only",
    "vpt_shallow",
    "vpt_deep",
    "lora",
    "full_ft",
]
METHOD_COLORS = {
    "zero_shot":    "#808080",
    "decoder_only": "#1f77b4",
    "vpt_shallow":  "#ff7f0e",
    "vpt_deep":     "#d62728",
    "lora":         "#9467bd",
    "full_ft":      "#2ca02c",
}
METHOD_LABELS = {
    "zero_shot":    "Zero-shot",
    "decoder_only": "Decoder-only",
    "vpt_shallow":  "VPT-shallow",
    "vpt_deep":     "VPT-deep",
    "lora":         "LoRA",
    "full_ft":      "Full FT",
}
DATASET_LABELS = {
    "isic2018_test": "ISIC 2018 (ID)",
    "ph2":           "PH² (near-OOD)",
    "busi":          "BUSI (ultrasound, far-OOD)",
    "cbis_ddsm":     "CBIS-DDSM (mammography, far-OOD)",
}
DATASET_ORDER = ["isic2018_test", "ph2", "busi", "cbis_ddsm"]


def load_data() -> pd.DataFrame:
    """Load robustness runs + baseline, combine into a single long-form DataFrame.

    Each row: (method, dataset, perturb_max_px, dice_mean, dice_std, ...)
    Baseline rows have perturb_max_px = 0.
    """
    if not (RESULTS_DIR / "runs.csv").exists():
        sys.exit(f"[plot] missing {RESULTS_DIR / 'runs.csv'} — run eval_bbox_robust.py first.")
    robust = pd.read_csv(RESULTS_DIR / "runs.csv")

    # Baseline: 0-px tight bbox from the main eval
    if not BASELINE_RUNS_CSV.exists():
        print(f"[plot] WARNING: {BASELINE_RUNS_CSV} not found — plots will skip the 0-px baseline.")
        return robust

    base = pd.read_csv(BASELINE_RUNS_CSV)
    # Some columns may differ between the two CSVs; keep only what we need
    base = base[["run_name", "method", "dataset", "seed",
                 "dice_mean", "dice_std", "iou_mean", "hd95_mean",
                 "trainable_params"]].copy()
    base["perturb_max_px"] = 0
    base["n_samples"] = 1
    base["n_images"] = ""
    base["peak_mem_mb"] = ""
    base["wall_clock_s"] = ""
    base["timestamp"] = ""

    # Align column order with robust
    base = base[robust.columns]
    combined = pd.concat([base, robust], ignore_index=True)
    return combined


def plot_degradation_curves(df: pd.DataFrame, out_path: Path) -> None:
    """4-panel grid (one per dataset). x = perturb level, y = Dice, line per method."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=False)
    axes_flat = axes.flatten()

    for ax, dataset in zip(axes_flat, DATASET_ORDER):
        sub = df[df["dataset"] == dataset]
        if sub.empty:
            ax.set_title(f"{DATASET_LABELS.get(dataset, dataset)}\n(no data)")
            continue

        for method in METHOD_ORDER:
            sub_m = sub[sub["method"] == method].sort_values("perturb_max_px")
            if sub_m.empty:
                continue
            x = sub_m["perturb_max_px"].values
            y = sub_m["dice_mean"].values
            yerr = sub_m["dice_std"].values
            ax.plot(x, y, marker="o", linewidth=2, color=METHOD_COLORS.get(method),
                    label=METHOD_LABELS.get(method, method))
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.12,
                            color=METHOD_COLORS.get(method))

        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        ax.set_xlabel("Max bbox expansion per side (px)")
        ax.set_ylabel("Dice")
        ax.grid(alpha=0.3)
        ax.set_xticks([0, 20, 50, 100, 200])

    # Single shared legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(METHOD_ORDER),
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("Segmentation Dice vs bbox imprecision  (5 random samples per image)",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_degradation_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """Heatmap: rows=method, cols=perturb_max_px, one panel per dataset."""
    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(18, 5), sharey=True)

    for ax, dataset in zip(axes, DATASET_ORDER):
        sub = df[df["dataset"] == dataset]
        # Pivot: rows=method (in METHOD_ORDER), cols=perturb_max_px
        pivot = sub.pivot_table(
            index="method", columns="perturb_max_px",
            values="dice_mean", aggfunc="first",
        )
        # Reorder rows
        pivot = pivot.reindex([m for m in METHOD_ORDER if m in pivot.index])
        im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in pivot.index])
        ax.set_xlabel("Max bbox expansion (px)")
        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        # Cell-value annotations
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if pd.isna(val):
                    continue
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        color="black" if val > 0.6 else "white", fontsize=8)
        if ax is axes[-1]:
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Dice")

    fig.suptitle("Dice heatmap: method × bbox imprecision", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_relative_drop(df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of % Dice drop from 0-px baseline to 200-px (the most extreme)."""
    # Only plot if we have both baseline and 200-px rows
    baseline = df[df["perturb_max_px"] == 0]
    extreme = df[df["perturb_max_px"] == 200]
    if baseline.empty or extreme.empty:
        print("[plot] skipping relative_drop — missing baseline or pm=200 rows")
        return

    merged = baseline.merge(
        extreme, on=["method", "dataset"],
        suffixes=("_base", "_200"),
    )
    merged["drop_pct"] = 100.0 * (
        merged["dice_mean_base"] - merged["dice_mean_200"]
    ) / merged["dice_mean_base"]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(DATASET_ORDER))
    bar_w = 0.13
    for i, method in enumerate(METHOD_ORDER):
        sub_m = merged[merged["method"] == method]
        if sub_m.empty:
            continue
        # Order by DATASET_ORDER
        vals = []
        for ds in DATASET_ORDER:
            row = sub_m[sub_m["dataset"] == ds]
            vals.append(float(row["drop_pct"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + (i - len(METHOD_ORDER) / 2) * bar_w + bar_w / 2, vals,
               width=bar_w, color=METHOD_COLORS.get(method),
               label=METHOD_LABELS.get(method, method))

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(ds, ds) for ds in DATASET_ORDER],
                       rotation=15, ha="right")
    ax.set_ylabel("% Dice drop (0 px → 200 px)")
    ax.set_title("Sensitivity to bbox imprecision: relative Dice loss at max perturbation",
                 fontsize=12)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def write_summary_csv(df: pd.DataFrame, out_path: Path) -> None:
    """Pivoted summary: method × (dataset, perturb_max_px) → dice_mean (± dice_std)."""
    df = df.copy()
    df["cell"] = df.apply(
        lambda r: f"{r['dice_mean']:.4f} ± {r['dice_std']:.4f}", axis=1
    )
    pivot = df.pivot_table(
        index="method", columns=["dataset", "perturb_max_px"],
        values="cell", aggfunc="first",
    )
    # Reorder rows + columns
    pivot = pivot.reindex([m for m in METHOD_ORDER if m in pivot.index])
    # Bring datasets into our preferred order
    if isinstance(pivot.columns, pd.MultiIndex):
        present_ds = [ds for ds in DATASET_ORDER if ds in pivot.columns.get_level_values(0)]
        pivot = pivot.reindex(columns=present_ds, level=0)
    pivot.to_csv(out_path)
    print(f"[plot] wrote {out_path}")


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    print(f"[plot] loaded {len(df)} rows "
          f"({df['method'].nunique()} methods × "
          f"{df['dataset'].nunique()} datasets × "
          f"{df['perturb_max_px'].nunique()} perturb levels)")

    plot_degradation_curves(df, FIGURES_DIR / "degradation_curves.png")
    plot_degradation_heatmap(df, FIGURES_DIR / "degradation_heatmap.png")
    plot_relative_drop(df, FIGURES_DIR / "relative_drop.png")
    write_summary_csv(df, RESULTS_DIR / "summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
