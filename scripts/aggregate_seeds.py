"""Aggregate multi-seed eval results into mean ± std across seeds.

After running the seed=1 and seed=2 trainings + their evals, we'll have:

  Tight-bbox eval CSVs (one per seed):
    results/runs.csv             (seed 0, pm=0)
    results/runs_pm20.csv        (seed 0, pm=20)
    results/runs_rand100.csv     (seed 0, rand100)
    results/runs_seed1.csv       (seed 1, pm=0)
    results/runs_seed1_pm20.csv  (seed 1, pm=20)
    results/runs_seed1_rand100.csv
    results/runs_seed2.csv
    results/runs_seed2_pm20.csv
    results/runs_seed2_rand100.csv

  Bbox robustness CSVs (one per seed × training):
    bbox_robustness/results/runs.csv
    bbox_robustness/results_pm20/runs.csv
    bbox_robustness/results_rand100/runs.csv
    bbox_robustness/results_seed1/runs.csv
    bbox_robustness/results_seed1_pm20/runs.csv
    bbox_robustness/results_seed1_rand100/runs.csv
    bbox_robustness/results_seed2/runs.csv
    bbox_robustness/results_seed2_pm20/runs.csv
    bbox_robustness/results_seed2_rand100/runs.csv

This script aggregates them into:
  summary_full_multiseed.csv         — mean ± std per (method, training, dataset, perturb)
  bbox_robustness/comparison/seed_error_bars.png — overlay curves with shaded bands
  bbox_robustness/comparison/seed_summary_table.md — markdown table with ± std

Run after all 30 seed-1/seed-2 trainings + their 12 eval invocations complete.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Source CSV discovery — handles existing seed=0 paths plus seed{1,2} siblings
# ---------------------------------------------------------------------------

TRAININGS = ("pm=0", "pm=20", "rand100")
SEEDS = (0, 1, 2)

# tight-bbox CSV paths
def tight_csv_for_seed(seed: int, training: str) -> Path:
    """results/runs[_pm20|_rand100].csv (seed 0) or
       results/runs_seedN[_pm20|_rand100].csv (seed > 0)."""
    suffix = {"pm=0": "", "pm=20": "_pm20", "rand100": "_rand100"}[training]
    if seed == 0:
        if suffix == "":
            return REPO_ROOT / "results" / "runs.csv"
        return REPO_ROOT / "results" / f"runs{suffix}.csv"
    return REPO_ROOT / "results" / f"runs_seed{seed}{suffix}.csv"


def bbox_csv_for_seed(seed: int, training: str) -> Path:
    """bbox_robustness/results[_pm20|_rand100]/runs.csv (seed 0) or
       bbox_robustness/results_seedN[_pm20|_rand100]/runs.csv."""
    suffix = {"pm=0": "", "pm=20": "_pm20", "rand100": "_rand100"}[training]
    if seed == 0:
        if suffix == "":
            return REPO_ROOT / "bbox_robustness" / "results" / "runs.csv"
        return REPO_ROOT / "bbox_robustness" / f"results{suffix}" / "runs.csv"
    return REPO_ROOT / "bbox_robustness" / f"results_seed{seed}{suffix}" / "runs.csv"


METHODS = ["zero_shot", "decoder_only", "vpt_shallow", "vpt_deep", "lora", "full_ft"]
DATASETS = ["isic2018_test", "ph2", "busi", "cbis_ddsm"]
PERTURBS = [0, 20, 50, 100, 200]

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
    "busi":          "BUSI (far-OOD ultrasound)",
    "cbis_ddsm":     "CBIS-DDSM (far-OOD mammography)",
}
METHOD_COLORS = {
    "zero_shot":    "#808080",
    "decoder_only": "#1f77b4",
    "vpt_shallow":  "#ff7f0e",
    "vpt_deep":     "#d62728",
    "lora":         "#9467bd",
    "full_ft":      "#2ca02c",
}
TRAINING_LINESTYLES = {"pm=0": "-", "pm=20": "--", "rand100": ":"}


# ---------------------------------------------------------------------------
# Load every (seed, training) pair into one long-form DataFrame
# ---------------------------------------------------------------------------

def load_all_seeds() -> pd.DataFrame:
    """Returns DataFrame with columns: seed, method, training, dataset,
       perturb_max_px, dice_mean. Includes both tight (pm=0 eval) and
       bbox-robustness (pm=20..200) rows."""
    rows = []

    for seed in SEEDS:
        for training in TRAININGS:
            # Tight-bbox rows (perturb_max_px = 0)
            tight_path = tight_csv_for_seed(seed, training)
            if tight_path.exists():
                tight_df = pd.read_csv(tight_path)
                for _, r in tight_df.iterrows():
                    rows.append({
                        "seed": seed,
                        "method": r["method"],
                        "training": training,
                        "dataset": r["dataset"],
                        "perturb_max_px": 0,
                        "dice_mean": float(r["dice_mean"]),
                    })

            # Bbox-robustness rows (perturb_max_px ∈ {20, 50, 100, 200})
            bbox_path = bbox_csv_for_seed(seed, training)
            if bbox_path.exists():
                bbox_df = pd.read_csv(bbox_path)
                for _, r in bbox_df.iterrows():
                    rows.append({
                        "seed": seed,
                        "method": r["method"],
                        "training": training,
                        "dataset": r["dataset"],
                        "perturb_max_px": int(r["perturb_max_px"]),
                        "dice_mean": float(r["dice_mean"]),
                    })

    df = pd.DataFrame(rows)
    # zero_shot results are identical across trainings (no checkpoint difference)
    # — drop duplicates so we keep just one copy per (seed, dataset, perturb).
    df = df.drop_duplicates(
        subset=["seed", "method", "training", "dataset", "perturb_max_px"],
        keep="first",
    )
    return df


# ---------------------------------------------------------------------------
# Aggregate: mean ± std per (method, training, dataset, perturb)
# ---------------------------------------------------------------------------

def aggregate_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (method, training, dataset, perturb), compute mean + std across seeds."""
    agg = (
        df.groupby(["method", "training", "dataset", "perturb_max_px"])
          .agg(
              dice_mean_seeds=("dice_mean", "mean"),
              dice_std_seeds=("dice_mean", "std"),
              n_seeds=("seed", "nunique"),
              seeds=("seed", lambda s: ",".join(map(str, sorted(s.unique())))),
          )
          .reset_index()
    )
    # If only one seed contributed, std is NaN — keep but flag
    agg["dice_std_seeds"] = agg["dice_std_seeds"].fillna(0.0)
    return agg


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(agg: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = agg.copy()
    for c in ("dice_mean_seeds", "dice_std_seeds"):
        df[c] = df[c].round(4)
    df.to_csv(out_path, index=False)
    print(f"[agg-seeds] wrote {out_path}")


def write_summary_markdown(agg: pd.DataFrame, out_path: Path) -> None:
    """Per-dataset table: method × training, cell = mean ± std @ pm=0 and pm=200."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Multi-seed summary  (mean ± std across seeds 0, 1, 2)\n"]

    for ds in DATASETS:
        lines.append(f"\n## {DATASET_LABELS.get(ds, ds)}\n")
        lines.append("| Method | Training | Dice @ pm=0 (tight) | Dice @ pm=200 (extreme) | n_seeds |")
        lines.append("|---|---|---:|---:|---:|")
        for m in METHODS:
            if m == "zero_shot":
                tight = agg[(agg["method"] == m) & (agg["dataset"] == ds) & (agg["perturb_max_px"] == 0)]
                extreme = agg[(agg["method"] == m) & (agg["dataset"] == ds) & (agg["perturb_max_px"] == 200)]
                if tight.empty:
                    continue
                # zero_shot only has pm=0 training entries (same checkpoint reused)
                r_t = tight.iloc[0]
                r_e = extreme.iloc[0] if not extreme.empty else None
                t_cell = f"{r_t['dice_mean_seeds']:.4f} ± {r_t['dice_std_seeds']:.4f}"
                e_cell = (f"{r_e['dice_mean_seeds']:.4f} ± {r_e['dice_std_seeds']:.4f}"
                          if r_e is not None else "—")
                lines.append(f"| {METHOD_LABELS[m]} | — | {t_cell} | {e_cell} | {int(r_t['n_seeds'])} |")
            else:
                for tr in TRAININGS:
                    tight = agg[(agg["method"] == m) & (agg["training"] == tr) &
                                (agg["dataset"] == ds) & (agg["perturb_max_px"] == 0)]
                    extreme = agg[(agg["method"] == m) & (agg["training"] == tr) &
                                  (agg["dataset"] == ds) & (agg["perturb_max_px"] == 200)]
                    if tight.empty:
                        continue
                    r_t = tight.iloc[0]
                    r_e = extreme.iloc[0] if not extreme.empty else None
                    t_cell = f"{r_t['dice_mean_seeds']:.4f} ± {r_t['dice_std_seeds']:.4f}"
                    e_cell = (f"{r_e['dice_mean_seeds']:.4f} ± {r_e['dice_std_seeds']:.4f}"
                              if r_e is not None else "—")
                    lines.append(f"| {METHOD_LABELS[m]} | {tr} | {t_cell} | {e_cell} | {int(r_t['n_seeds'])} |")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[agg-seeds] wrote {out_path}")


def plot_curves_with_seed_bands(agg: pd.DataFrame, out_path: Path) -> None:
    """Overlay degradation curves with shaded ±std bands across seeds."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True)
    axes_flat = axes.flatten()

    for ax, ds in zip(axes_flat, DATASETS):
        for m in METHODS:
            if m == "zero_shot":
                sub = agg[(agg["method"] == m) & (agg["dataset"] == ds)].sort_values("perturb_max_px")
                if not sub.empty:
                    ax.plot(sub["perturb_max_px"], sub["dice_mean_seeds"],
                             marker="o", linewidth=2.5, color=METHOD_COLORS[m],
                             label=METHOD_LABELS[m])
                    ax.fill_between(sub["perturb_max_px"],
                                     sub["dice_mean_seeds"] - sub["dice_std_seeds"],
                                     sub["dice_mean_seeds"] + sub["dice_std_seeds"],
                                     color=METHOD_COLORS[m], alpha=0.10)
                continue
            for tr in TRAININGS:
                sub = agg[(agg["method"] == m) & (agg["training"] == tr) & (agg["dataset"] == ds)].sort_values("perturb_max_px")
                if sub.empty:
                    continue
                ax.plot(sub["perturb_max_px"], sub["dice_mean_seeds"],
                         color=METHOD_COLORS[m],
                         linestyle=TRAINING_LINESTYLES[tr], linewidth=1.6, marker="o", markersize=4,
                         alpha=0.9, label="_nolegend_")
                ax.fill_between(sub["perturb_max_px"],
                                 sub["dice_mean_seeds"] - sub["dice_std_seeds"],
                                 sub["dice_mean_seeds"] + sub["dice_std_seeds"],
                                 color=METHOD_COLORS[m], alpha=0.10)
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.set_xlabel("Eval bbox max expansion (px)")
        ax.set_ylabel("Dice (mean ± std across seeds)")
        ax.grid(alpha=0.3)
        ax.set_xticks(PERTURBS)

    method_handles = [
        plt.Line2D([], [], color=METHOD_COLORS[m], marker="o", linewidth=2, label=METHOD_LABELS[m])
        for m in METHODS
    ]
    style_handles = [
        plt.Line2D([], [], color="black", linewidth=2, linestyle="-", label="pm=0 train"),
        plt.Line2D([], [], color="black", linewidth=2, linestyle="--", label="pm=20 train"),
        plt.Line2D([], [], color="black", linewidth=2, linestyle=":", label="rand100 train"),
    ]
    fig.legend(handles=method_handles + style_handles,
                loc="lower center", ncol=5, frameon=False,
                bbox_to_anchor=(0.5, -0.04), fontsize=10)
    fig.suptitle(
        "Multi-seed curves with ±std bands (n_seeds varies per cell — see summary CSV)",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[agg-seeds] wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("[agg-seeds] scanning for seed CSVs...")
    df = load_all_seeds()
    if df.empty:
        print("[agg-seeds] No CSVs found. Have you trained and evaluated seed=1/seed=2 yet?")
        return 1

    print(f"[agg-seeds] loaded {len(df)} raw rows across {df['seed'].nunique()} seed(s): {sorted(df['seed'].unique())}")
    agg = aggregate_mean_std(df)
    print(f"[agg-seeds] aggregated to {len(agg)} unique (method, training, dataset, perturb) cells")

    write_csv(agg, REPO_ROOT / "summary_full_multiseed.csv")
    write_summary_markdown(agg, REPO_ROOT / "bbox_robustness" / "comparison" / "seed_summary_table.md")
    plot_curves_with_seed_bands(agg, REPO_ROOT / "bbox_robustness" / "comparison" / "seed_error_bars.png")

    # Quick console summary of n_seeds per cell — surfaces incomplete runs
    n_seeds_dist = agg["n_seeds"].value_counts().sort_index()
    print(f"\n[agg-seeds] n_seeds distribution across {len(agg)} cells:")
    for n, count in n_seeds_dist.items():
        print(f"             {n} seed(s): {count} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
