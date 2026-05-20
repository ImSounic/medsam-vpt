"""Generate report figures + summary tables from results/runs.csv.

Run:
    python scripts/plots.py

Outputs to results/figures/ (300 dpi PNG) and results/summary_table.csv.

Figures produced:
  1_dice_per_dataset.png          — 3 sub-panels (ISIC | PH² | BUSI); zoomed bars
  2_pareto_id_vs_ood.png          — 2 sub-panels (ID Dice vs params | OOD Dice vs params)
  3_drift_gap.png                 — bar chart, value-labelled, log-style for visibility
  4_hd95_split.png                — 2 panels (skin domain | ultrasound domain) at native scales
  5_id_vs_ood_scatter.png         — 2 panels (ISIC vs PH² | ISIC vs BUSI) with y=x reference
  6_drift_curves.png              — line plot, no error bars (handled in fig 1)
  7_rank_reversal.png             — bumps chart showing method ranks across datasets
  8_summary_heatmap.png           — methods × datasets × Dice heatmap with cell labels
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_CSV = REPO_ROOT / "results" / "runs.csv"
FIG_DIR = REPO_ROOT / "results" / "figures"
SUMMARY_CSV = REPO_ROOT / "results" / "summary_table.csv"

# Canonical method order (rough order of trainable parameter count)
METHODS = ["zero_shot", "decoder_only", "vpt_shallow", "vpt_deep", "lora", "full_ft"]
METHOD_LABELS = {
    "zero_shot":     "Zero-shot",
    "decoder_only":  "Decoder-only FT",
    "vpt_shallow":   "VPT-shallow",
    "vpt_deep":      "VPT-deep",
    "lora":          "LoRA",
    "full_ft":       "Full FT",
}
METHOD_COLORS = {
    "zero_shot":     "#7f7f7f",  # gray
    "decoder_only":  "#1f77b4",  # blue
    "vpt_shallow":   "#ff7f0e",  # orange
    "vpt_deep":      "#d62728",  # red
    "lora":          "#2ca02c",  # green
    "full_ft":       "#9467bd",  # purple
}

DATASETS = ["isic2018_test", "ph2", "busi", "cbis_ddsm"]
DATASET_LABELS = {
    "isic2018_test": "ISIC (ID)",
    "ph2":           "PH² (near-OOD)",
    "busi":          "BUSI (far-OOD)",
    "cbis_ddsm":     "CBIS-DDSM (far-OOD)",
}

# Save figures at 300 dpi for paper-quality output
DPI = 300


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Latest non-quick row per (method, dataset)."""
    df = pd.read_csv(RUNS_CSV)
    df["notes"] = df["notes"].fillna("")
    df = df[~df["notes"].str.contains("quick", case=False, na=False)]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = (
        df.sort_values("timestamp")
        .drop_duplicates(["method", "dataset"], keep="last")
        .reset_index(drop=True)
    )
    return df


def get(df, method, dataset, col):
    row = df[(df["method"] == method) & (df["dataset"] == dataset)]
    return np.nan if row.empty else row[col].iloc[0]


# ----------------------------------------------------------------------------
# Figure 1 — 3-panel Dice per dataset, zoomed per-panel y-axes, value labels
# ----------------------------------------------------------------------------
def plot_dice_per_dataset(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(6 * len(DATASETS), 5.5))
    method_names = [METHOD_LABELS[m] for m in METHODS]
    colors = [METHOD_COLORS[m] for m in METHODS]

    for ax, ds in zip(axes, DATASETS):
        values = [get(df, m, ds, "dice_mean") for m in METHODS]
        stds = [get(df, m, ds, "dice_std") for m in METHODS]
        bars = ax.bar(
            method_names, values, color=colors, alpha=0.85,
            edgecolor="black", linewidth=0.6,
        )
        # Value labels above each bar
        for bar, val, std in zip(bars, values, stds):
            if np.isnan(val):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2, val + 0.005,
                f"{val:.3f}\n±{std:.3f}",
                ha="center", va="bottom", fontsize=8.5,
            )
        # Auto y-axis: use the data range with some padding
        finite_vals = [v for v in values if not np.isnan(v)]
        ymin = max(0, min(finite_vals) - 0.06)
        ymax = min(1.0, max(finite_vals) + 0.08)
        ax.set_ylim(ymin, ymax)
        ax.set_title(DATASET_LABELS[ds], fontsize=12, fontweight="bold")
        ax.set_ylabel("Dice")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    fig.suptitle("Segmentation Dice across the drift ladder", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
# Figure 2 — Pareto: 2 sub-panels (ID + far-OOD Dice vs trainable params)
# ----------------------------------------------------------------------------
def plot_pareto_id_vs_ood(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=False)

    for ax, ds, title in zip(
        axes,
        ["isic2018_test", "busi"],
        ["In-distribution: ISIC", "Far-OOD: BUSI"],
    ):
        for m in METHODS:
            params = get(df, m, ds, "trainable_params")
            dice = get(df, m, ds, "dice_mean")
            if np.isnan(dice):
                continue
            # Place zero-shot at a special x position to the left
            x = max(params, 0.5)
            ax.scatter(
                x, dice, color=METHOD_COLORS[m], s=300, alpha=0.9,
                edgecolors="black", linewidth=1.2, zorder=3,
            )
            # Method label next to each point
            ax.annotate(
                METHOD_LABELS[m], (x, dice),
                xytext=(8, 8), textcoords="offset points",
                fontsize=9.5, fontweight="bold",
            )

        ax.set_xscale("log")
        ax.set_xlabel("Trainable parameters (log scale)")
        ax.set_ylabel("Dice")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, which="both", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        # Mark zero-shot region explicitly
        ax.axvspan(0.3, 1.5, alpha=0.08, color="gray", zorder=0)
        ax.text(0.7, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else 0.5,
                "zero-\nshot", ha="center", fontsize=8, color="gray", alpha=0.7)

    fig.suptitle(
        "Performance vs. trainable parameter count — ID vs. far-OOD\n"
        "(ranking inverts on BUSI: methods that modify the encoder lose to those that don't)",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
# Figure 3 — Drift gap with value labels (linear scale; PH² gets value labels
# even though bars are tiny)
# ----------------------------------------------------------------------------
def plot_drift_gap(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6.5))
    bar_w = 0.27
    x = np.arange(len(METHODS))

    ph2_gaps, busi_gaps, ddsm_gaps = [], [], []
    for m in METHODS:
        id_d = get(df, m, "isic2018_test", "dice_mean")
        ph2_d = get(df, m, "ph2", "dice_mean")
        busi_d = get(df, m, "busi", "dice_mean")
        ddsm_d = get(df, m, "cbis_ddsm", "dice_mean")
        ph2_gaps.append(id_d - ph2_d if not np.isnan(ph2_d) else np.nan)
        busi_gaps.append(id_d - busi_d if not np.isnan(busi_d) else np.nan)
        ddsm_gaps.append(id_d - ddsm_d if not np.isnan(ddsm_d) else np.nan)

    b1 = ax.bar(x - bar_w, ph2_gaps, bar_w, color="#1f77b4", alpha=0.85,
                edgecolor="black", linewidth=0.6, label="PH² (near-OOD)")
    b2 = ax.bar(x, busi_gaps, bar_w, color="#d62728", alpha=0.85,
                edgecolor="black", linewidth=0.6, label="BUSI (far-OOD)")
    b3 = ax.bar(x + bar_w, ddsm_gaps, bar_w, color="#8c564b", alpha=0.85,
                edgecolor="black", linewidth=0.6, label="CBIS-DDSM (far-OOD)")

    # Value labels
    for bar, val in zip(b1, ph2_gaps):
        if np.isnan(val):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.005 if val >= 0 else val - 0.012,
                f"{val:+.3f}", ha="center", fontsize=7.5)
    for bar, val in zip(b2, busi_gaps):
        if np.isnan(val):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.005,
                f"{val:+.3f}", ha="center", fontsize=8, fontweight="bold")
    for bar, val in zip(b3, ddsm_gaps):
        if np.isnan(val):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.005,
                f"{val:+.3f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=15)
    ax.set_ylabel("Drift gap (ID Dice − OOD Dice)")
    ax.set_title("Distribution drift cost — lower bar = more robust", fontsize=13)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
# Figure 4 — HD95 split panel (skin domain vs ultrasound, linear scales)
# ----------------------------------------------------------------------------
def plot_hd95_split(df: pd.DataFrame, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5),
                                    gridspec_kw={"width_ratios": [1.3, 1]})

    method_names = [METHOD_LABELS[m] for m in METHODS]
    x = np.arange(len(METHODS))
    bar_w = 0.4

    # Left panel: ISIC + PH² (skin domain, all values < 20 px)
    isic_v = [get(df, m, "isic2018_test", "hd95_mean") for m in METHODS]
    ph2_v = [get(df, m, "ph2", "hd95_mean") for m in METHODS]
    b1 = ax1.bar(x - bar_w / 2, isic_v, bar_w, color="#1f77b4", alpha=0.85,
                 edgecolor="black", linewidth=0.6, label="ISIC (ID)")
    b2 = ax1.bar(x + bar_w / 2, ph2_v, bar_w, color="#ff7f0e", alpha=0.85,
                 edgecolor="black", linewidth=0.6, label="PH² (near-OOD)")
    for bar, val in zip(list(b1) + list(b2), isic_v + ph2_v):
        if np.isnan(val):
            continue
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.4,
                 f"{val:.1f}", ha="center", fontsize=8.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(method_names, rotation=15)
    ax1.set_ylabel("HD95 (pixels)")
    ax1.set_title("Skin domain (ISIC + PH²)", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper right")

    # Right panel: BUSI + CBIS-DDSM (both far-OOD, large scale, up to ~300 px)
    busi_v = [get(df, m, "busi", "hd95_mean") for m in METHODS]
    ddsm_v = [get(df, m, "cbis_ddsm", "hd95_mean") for m in METHODS]
    b3 = ax2.bar(x - bar_w / 2, busi_v, bar_w, color="#d62728", alpha=0.85,
                 edgecolor="black", linewidth=0.6, label="BUSI (ultrasound)")
    b4 = ax2.bar(x + bar_w / 2, ddsm_v, bar_w, color="#8c564b", alpha=0.85,
                 edgecolor="black", linewidth=0.6, label="CBIS-DDSM (mammography)")
    for bar, val in zip(list(b3) + list(b4), busi_v + ddsm_v):
        if np.isnan(val):
            continue
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 4,
                 f"{val:.1f}", ha="center", fontsize=8.5, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(method_names, rotation=15)
    ax2.set_ylabel("HD95 (pixels)")
    ax2.set_title("Far-OOD domain (BUSI ultrasound + CBIS-DDSM mammography)",
                  fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper right")

    fig.suptitle("Boundary error (HD95) — separate scales for skin vs far-OOD domains",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
# Figure 5 — 2-panel scatter: ISIC vs PH² | ISIC vs BUSI, with y=x diagonal
# ----------------------------------------------------------------------------
def plot_id_vs_ood_scatter(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    for ax, ood_ds, ood_label in zip(
        axes,
        ["ph2", "busi", "cbis_ddsm"],
        ["PH² Dice (near-OOD)", "BUSI Dice (far-OOD)", "CBIS-DDSM Dice (far-OOD)"],
    ):
        for m in METHODS:
            id_d = get(df, m, "isic2018_test", "dice_mean")
            ood_d = get(df, m, ood_ds, "dice_mean")
            if np.isnan(id_d) or np.isnan(ood_d):
                continue
            ax.scatter(
                id_d, ood_d, color=METHOD_COLORS[m], s=300, alpha=0.9,
                edgecolors="black", linewidth=1.2, zorder=3,
            )
            ax.annotate(
                METHOD_LABELS[m], (id_d, ood_d),
                xytext=(10, 6), textcoords="offset points",
                fontsize=10, fontweight="bold",
            )

        # Diagonal (y = x) showing "perfect transfer"
        all_id = [get(df, m, "isic2018_test", "dice_mean") for m in METHODS]
        all_ood = [get(df, m, ood_ds, "dice_mean") for m in METHODS]
        lo = min(min(all_id), min(all_ood)) - 0.05
        hi = max(max(all_id), max(all_ood)) + 0.05
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=1, zorder=1,
                label="y = x (no drift)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("ISIC Dice (in-distribution)")
        ax.set_ylabel(ood_label)
        ax.set_title(f"ID vs {DATASET_LABELS[ood_ds]}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.legend(loc="lower right")

    fig.suptitle(
        "Distance below the y=x diagonal = drift cost\n"
        "(PH² hugs the diagonal; BUSI shows big drops for encoder-modifying methods; "
        "CBIS-DDSM shows the largest drops)",
        fontsize=13, y=1.03,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
# Figure 6 — Drift curves (no error bars; cleaner)
# ----------------------------------------------------------------------------
def plot_drift_curves(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    xs = list(range(len(DATASETS)))

    for m in METHODS:
        ys = [get(df, m, ds, "dice_mean") for ds in DATASETS]
        if all(np.isnan(y) for y in ys):
            continue
        ax.plot(
            xs, ys, label=METHOD_LABELS[m], color=METHOD_COLORS[m],
            marker="o", markersize=11, linewidth=2.5,
            markeredgecolor="black", markeredgewidth=0.7,
        )
        # Value annotations at each point
        for x, y in zip(xs, ys):
            if not np.isnan(y):
                ax.annotate(f"{y:.3f}", (x, y),
                            xytext=(0, 8 if m != "zero_shot" else -14),
                            textcoords="offset points",
                            fontsize=8, ha="center", color=METHOD_COLORS[m])

    ax.set_xticks(xs)
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in DATASETS], fontsize=11)
    ax.set_xlabel("Increasing distribution shift →", fontsize=11)
    ax.set_ylabel("Dice", fontsize=11)
    ax.set_title("Method robustness across the drift ladder", fontsize=13)
    ax.legend(loc="lower left", title="Method", framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
# Figure 7 — Bumps chart: method ranking across datasets (NEW)
# ----------------------------------------------------------------------------
def plot_rank_reversal(df: pd.DataFrame, out_path: Path) -> None:
    """Slopechart showing how each method's rank changes across datasets."""
    fig, ax = plt.subplots(figsize=(11, 7))

    # For each dataset, rank methods 1 (best) to 6 (worst) by Dice
    ranks_per_dataset = {}
    for ds in DATASETS:
        method_scores = [(m, get(df, m, ds, "dice_mean")) for m in METHODS]
        # Sort descending: rank 1 = highest Dice
        method_scores.sort(key=lambda x: -x[1] if not np.isnan(x[1]) else 99)
        ranks_per_dataset[ds] = {m: rank + 1 for rank, (m, _) in enumerate(method_scores)}

    xs = list(range(len(DATASETS)))
    for m in METHODS:
        ranks = [ranks_per_dataset[ds][m] for ds in DATASETS]
        ax.plot(
            xs, ranks, marker="o", markersize=14, linewidth=3,
            color=METHOD_COLORS[m], alpha=0.9, label=METHOD_LABELS[m],
            markeredgecolor="black", markeredgewidth=1,
        )
        # Label at the rightmost point
        ax.text(xs[-1] + 0.05, ranks[-1], METHOD_LABELS[m],
                va="center", fontsize=10, fontweight="bold",
                color=METHOD_COLORS[m])

    ax.set_xticks(xs)
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in DATASETS], fontsize=11)
    ax.set_yticks(range(1, 7))
    ax.set_ylabel("Rank by Dice (1 = best)", fontsize=11)
    ax.invert_yaxis()
    ax.set_title(
        "Method ranking flips across the drift ladder\n"
        "LoRA: best PEFT on ID → worst on far-OOD;  Decoder-only: middling on ID → 2nd on far-OOD",
        fontsize=12,
    )
    ax.set_xlim(-0.2, len(DATASETS) + 0.5)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
# Figure 8 — Summary heatmap: methods × datasets × Dice
# ----------------------------------------------------------------------------
def plot_summary_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    matrix = np.array(
        [[get(df, m, ds, "dice_mean") for ds in DATASETS] for m in METHODS]
    )
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Dice", fontsize=11)

    # Cell annotations
    for i in range(len(METHODS)):
        for j in range(len(DATASETS)):
            val = matrix[i, j]
            if np.isnan(val):
                txt = "—"
            else:
                txt = f"{val:.4f}"
            # Text color: black on light, white on dark (for readability)
            txt_color = "black" if 0.7 <= val <= 0.9 else "white" if val < 0.7 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    color=txt_color, fontsize=11, fontweight="bold")

    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in DATASETS], fontsize=11)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=11)
    ax.set_title("Dice — methods × drift ladder (summary)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
# Summary table CSV
# ----------------------------------------------------------------------------
def write_summary_table(df: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for m in METHODS:
        row = {"method": METHOD_LABELS[m]}
        for ds in DATASETS:
            d = get(df, m, ds, "dice_mean")
            s = get(df, m, ds, "dice_std")
            h = get(df, m, ds, "hd95_mean")
            i = get(df, m, ds, "iou_mean")
            row[f"{ds}_dice"] = f"{d:.4f} ± {s:.4f}" if not np.isnan(d) else "—"
            row[f"{ds}_iou"] = f"{i:.4f}" if not np.isnan(i) else "—"
            row[f"{ds}_hd95"] = f"{h:.2f}" if not np.isnan(h) else "—"
        p = get(df, m, "isic2018_test", "trainable_params")
        row["trainable_params"] = int(p) if not np.isnan(p) else 0
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


# ----------------------------------------------------------------------------
def main() -> int:
    if not RUNS_CSV.exists():
        print(f"[plots] {RUNS_CSV} not found — nothing to do")
        return 1

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    print(f"[plots] Loaded {len(df)} unique (method, dataset) rows")
    print(f"[plots] Methods:  {sorted(df['method'].unique())}")
    print(f"[plots] Datasets: {sorted(df['dataset'].unique())}")

    plot_dice_per_dataset(df, FIG_DIR / "1_dice_per_dataset.png")
    plot_pareto_id_vs_ood(df, FIG_DIR / "2_pareto_id_vs_ood.png")
    plot_drift_gap(df, FIG_DIR / "3_drift_gap.png")
    plot_hd95_split(df, FIG_DIR / "4_hd95_split.png")
    plot_id_vs_ood_scatter(df, FIG_DIR / "5_id_vs_ood_scatter.png")
    plot_drift_curves(df, FIG_DIR / "6_drift_curves.png")
    plot_rank_reversal(df, FIG_DIR / "7_rank_reversal.png")
    plot_summary_heatmap(df, FIG_DIR / "8_summary_heatmap.png")
    write_summary_table(df, SUMMARY_CSV)

    print(f"[plots] Wrote 8 figures to {FIG_DIR}/")
    print(f"[plots] Wrote summary table to {SUMMARY_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
