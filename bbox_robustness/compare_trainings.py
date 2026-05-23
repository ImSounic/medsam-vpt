"""Three-way comparison: pm=0, pm=20, rand100 trained models.

Reads:
  bbox_robustness/results/runs.csv          (pm=0 trained, bbox perturbations)
  bbox_robustness/results_pm20/runs.csv     (pm=20 trained, bbox perturbations)
  bbox_robustness/results_rand100/runs.csv  (rand100 trained, bbox perturbations)
  ../results/runs.csv                       (pm=0 trained, tight bbox baseline)
  ../results/runs_pm20.csv                  (pm=20 trained, tight bbox baseline)
  ../results/runs_rand100.csv               (rand100 trained, tight bbox baseline)

Produces in bbox_robustness/comparison/:
  comparison_curves_3way.png    — overlay curves (solid pm=0, dashed pm=20, dotted rand100)
  delta_heatmap_3way.png        — two columns of delta heatmaps (vs pm=0 baseline)
  tradeoff_id_vs_perturbation.png — scatter showing ISIC tight vs ISIC pm=200 trade-off
  tradeoff_isic_vs_cbis.png     — scatter showing modality-transfer trade-off
  comparison_summary_3way.csv   — long-form summary

Also regenerates summary_full.csv at the repo root.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

# Training -> (bbox_robust runs.csv, tight runs.csv) — order matters for plot styling
TRAININGS = [
    ("pm=0",     REPO_ROOT / "bbox_robustness" / "results"          / "runs.csv",
                 REPO_ROOT / "results" / "runs.csv"),
    ("pm=20",    REPO_ROOT / "bbox_robustness" / "results_pm20"     / "runs.csv",
                 REPO_ROOT / "results" / "runs_pm20.csv"),
    ("rand100",  REPO_ROOT / "bbox_robustness" / "results_rand100"  / "runs.csv",
                 REPO_ROOT / "results" / "runs_rand100.csv"),
]

OUT_DIR = REPO_ROOT / "bbox_robustness" / "comparison"
OUT_SUMMARY_3WAY = OUT_DIR / "comparison_summary_3way.csv"
OUT_SUMMARY_FULL = REPO_ROOT / "summary_full.csv"

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
TRAINING_STYLES = {
    "pm=0":    {"linestyle": "-",  "marker": "o", "alpha": 1.00},
    "pm=20":   {"linestyle": "--", "marker": "s", "alpha": 0.85},
    "rand100": {"linestyle": ":",  "marker": "^", "alpha": 0.75, "linewidth": 2.5},
}
DATASET_LABELS = {
    "isic2018_test": "ISIC 2018 (ID)",
    "ph2":           "PH² (near-OOD)",
    "busi":          "BUSI (far-OOD ultrasound)",
    "cbis_ddsm":     "CBIS-DDSM (far-OOD mammography)",
}
DATASET_ORDER = ["isic2018_test", "ph2", "busi", "cbis_ddsm"]
PERTURBS = [0, 20, 50, 100, 200]


# ---------------------------------------------------------------------------
# Load + combine all 6 CSVs into one long-form DataFrame
# ---------------------------------------------------------------------------

def load_combined() -> pd.DataFrame:
    """Returns DataFrame with columns: method, dataset, perturb_max_px, training, dice_mean, iou_mean, hd95_mean."""
    frames = []

    for training, bbox_csv, tight_csv in TRAININGS:
        if not bbox_csv.exists():
            print(f"[compare] WARNING: {bbox_csv} not found — skipping {training} bbox data")
            continue
        if not tight_csv.exists():
            print(f"[compare] WARNING: {tight_csv} not found — skipping {training} tight data")
            continue

        # Bbox-robustness rows (perturb_max_px ∈ {20, 50, 100, 200})
        bbox_df = pd.read_csv(bbox_csv)
        bbox_df["training"] = training
        frames.append(bbox_df[[
            "method", "dataset", "perturb_max_px", "training",
            "dice_mean", "iou_mean", "hd95_mean",
        ]])

        # Tight-bbox rows (perturb_max_px = 0)
        tight_df = pd.read_csv(tight_csv)
        tight_df["perturb_max_px"] = 0
        tight_df["training"] = training
        frames.append(tight_df[[
            "method", "dataset", "perturb_max_px", "training",
            "dice_mean", "iou_mean", "hd95_mean",
        ]])

    df = pd.concat(frames, ignore_index=True)
    # zero_shot is identical across all three trainings (no checkpoint to differ).
    # Drop duplicates — keep first occurrence (pm=0's zero_shot row).
    df = df.drop_duplicates(
        subset=["method", "dataset", "perturb_max_px", "training"],
        keep="first",
    )
    return df


# ---------------------------------------------------------------------------
# Plot 1: overlay degradation curves — 3 line styles per method
# ---------------------------------------------------------------------------

def plot_overlay_curves(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True, sharey=False)
    axes_flat = axes.flatten()

    for ax, ds in zip(axes_flat, DATASET_ORDER):
        sub = df[df["dataset"] == ds]
        for m in METHOD_ORDER:
            if m == "zero_shot":
                # Single line — same across trainings
                sub_m = sub[(sub["method"] == m) & (sub["training"] == "pm=0")].sort_values("perturb_max_px")
                if not sub_m.empty:
                    ax.plot(sub_m["perturb_max_px"], sub_m["dice_mean"],
                             marker="o", linewidth=2.5, color=METHOD_COLORS[m],
                             label=METHOD_LABELS[m], zorder=1)
                continue

            for training in ("pm=0", "pm=20", "rand100"):
                sub_t = sub[(sub["method"] == m) & (sub["training"] == training)].sort_values("perturb_max_px")
                if sub_t.empty:
                    continue
                style = TRAINING_STYLES[training]
                ax.plot(sub_t["perturb_max_px"], sub_t["dice_mean"],
                         color=METHOD_COLORS[m],
                         linestyle=style["linestyle"],
                         marker=style["marker"],
                         linewidth=style.get("linewidth", 1.6),
                         alpha=style["alpha"],
                         markersize=5,
                         zorder=2 + ("pm=0", "pm=20", "rand100").index(training),
                         label="_nolegend_")

        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=12)
        ax.set_xlabel("Eval-time bbox max expansion (px)")
        ax.set_ylabel("Dice")
        ax.grid(alpha=0.3)
        ax.set_xticks(PERTURBS)

    # Method legend (colors)
    method_handles = [
        plt.Line2D([], [], color=METHOD_COLORS[m], marker="o", linewidth=2,
                    label=METHOD_LABELS[m])
        for m in METHOD_ORDER
    ]
    # Training-style legend (linestyles)
    style_handles = [
        plt.Line2D([], [], color="black", marker="o", linewidth=2, linestyle="-",
                    label="pm=0 trained (tight)"),
        plt.Line2D([], [], color="black", marker="s", linewidth=2, linestyle="--",
                    label="pm=20 trained (fixed jitter)"),
        plt.Line2D([], [], color="black", marker="^", linewidth=2.5, linestyle=":",
                    label="rand100 trained (random jitter)"),
    ]
    fig.legend(handles=method_handles + style_handles,
                loc="lower center", ncol=5, frameon=False,
                bbox_to_anchor=(0.5, -0.04), fontsize=10)
    fig.suptitle(
        "Bbox robustness across three training conditions  "
        "(solid = pm=0, dashed = pm=20 fixed jitter, dotted = rand100 random jitter)",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: dual delta heatmap — pm=20 vs pm=0, and rand100 vs pm=0
# ---------------------------------------------------------------------------

def plot_delta_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """Two side-by-side rows of heatmaps:
       - Top row: ΔDice (pm=20 trained − pm=0 trained), 4 panels per dataset
       - Bottom row: ΔDice (rand100 trained − pm=0 trained), 4 panels per dataset
    """
    fig, axes = plt.subplots(2, len(DATASET_ORDER), figsize=(20, 9), sharex=True, sharey=True)

    DELTA_MAX = 0.30  # color scale bound

    for row_i, training_target in enumerate(("pm=20", "rand100")):
        for col_i, ds in enumerate(DATASET_ORDER):
            ax = axes[row_i, col_i]
            sub = df[df["dataset"] == ds]
            pivot = sub.pivot_table(
                index="method", columns=["training", "perturb_max_px"],
                values="dice_mean", aggfunc="first",
            )
            delta_grid = pd.DataFrame(
                index=[m for m in METHOD_ORDER if m != "zero_shot"],
                columns=PERTURBS, dtype=float,
            )
            for m in delta_grid.index:
                for p in PERTURBS:
                    try:
                        v_target = pivot.loc[m, (training_target, p)]
                        v_base = pivot.loc[m, ("pm=0", p)]
                        delta_grid.loc[m, p] = v_target - v_base
                    except KeyError:
                        delta_grid.loc[m, p] = np.nan

            im = ax.imshow(delta_grid.values.astype(float),
                            cmap="RdBu_r", vmin=-DELTA_MAX, vmax=DELTA_MAX,
                            aspect="auto")
            ax.set_xticks(range(len(PERTURBS)))
            ax.set_xticklabels([str(p) for p in PERTURBS])
            ax.set_yticks(range(len(delta_grid.index)))
            ax.set_yticklabels([METHOD_LABELS[m] for m in delta_grid.index])
            if row_i == 1:
                ax.set_xlabel("Eval bbox max expansion (px)")
            if col_i == 0:
                ax.set_ylabel(f"{training_target} train\n− pm=0 train", fontsize=11, fontweight="bold")
            if row_i == 0:
                ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=11)
            for i in range(delta_grid.shape[0]):
                for j in range(delta_grid.shape[1]):
                    val = float(delta_grid.values[i, j])
                    if np.isnan(val):
                        continue
                    ax.text(j, i, f"{val:+.2f}",
                             ha="center", va="center",
                             color="white" if abs(val) > 0.20 else "black",
                             fontsize=7.5)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.018, pad=0.02)
    cbar.set_label("ΔDice (target training − pm=0 baseline)\nblue = worse than pm=0, red = better")
    fig.suptitle(
        "Effect of bbox jitter training on Dice  "
        "(top: pm=20 fixed jitter, bottom: rand100 random jitter)",
        fontsize=13, y=0.99,
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: trade-off scatter — ID prompt robustness
# ---------------------------------------------------------------------------

def plot_tradeoff_id_vs_perturbation(df: pd.DataFrame, out_path: Path) -> None:
    """For ISIC: scatter of (Dice at pm=0) vs (Dice at pm=200) for each (method, training)
       — visualizes the ID vs prompt-robustness trade-off.
    """
    fig, ax = plt.subplots(figsize=(9, 8))

    sub = df[df["dataset"] == "isic2018_test"]

    for m in METHOD_ORDER:
        # Collect points across all trainings for this method
        points = []
        for training in ("pm=0", "pm=20", "rand100"):
            tight = sub[(sub["method"] == m) & (sub["training"] == training) & (sub["perturb_max_px"] == 0)]
            perturbed = sub[(sub["method"] == m) & (sub["training"] == training) & (sub["perturb_max_px"] == 200)]
            if tight.empty or perturbed.empty:
                continue
            points.append((training, float(tight["dice_mean"].iloc[0]), float(perturbed["dice_mean"].iloc[0])))

        if not points:
            continue

        for training, x, y in points:
            style = TRAINING_STYLES.get(training, {"marker": "o"})
            ax.scatter(x, y, color=METHOD_COLORS[m],
                        marker=style["marker"], s=180, edgecolors="black", linewidth=1.2,
                        zorder=5, label="_nolegend_")
            ax.annotate(training, (x, y), xytext=(7, -2),
                         textcoords="offset points", fontsize=8, color="#333")

        # Connect points for this method to show the trajectory across trainings
        if len(points) > 1:
            xs = [p[1] for p in points]
            ys = [p[2] for p in points]
            ax.plot(xs, ys, color=METHOD_COLORS[m], linestyle="-", linewidth=1.2,
                     alpha=0.5, zorder=3)

    # Diagonal reference: y=x would mean equal accuracy at tight and loose bbox
    lims = [0.65, 0.97]
    ax.plot(lims, lims, "k:", alpha=0.4, label="y = x (perfectly prompt-invariant)")
    ax.set_xlim(*lims); ax.set_ylim(0.65, 0.90)
    ax.set_xlabel("ISIC Dice at tight bbox (pm=0)", fontsize=12)
    ax.set_ylabel("ISIC Dice at extreme perturbation (pm=200)", fontsize=12)
    ax.set_title(
        "Trade-off: tight-bbox accuracy vs prompt robustness (ISIC only)\n"
        "↑ and → towards upper-right corner = better at both",
        fontsize=12,
    )

    method_handles = [
        plt.Line2D([], [], color=METHOD_COLORS[m], marker="o", linestyle="",
                    markersize=10, label=METHOD_LABELS[m], markeredgecolor="black")
        for m in METHOD_ORDER
    ]
    style_handles = [
        plt.Line2D([], [], color="gray", marker=TRAINING_STYLES[t]["marker"],
                    linestyle="", markersize=10, label=f"{t} trained",
                    markeredgecolor="black")
        for t in ("pm=0", "pm=20", "rand100")
    ]
    ax.legend(handles=method_handles + style_handles, loc="lower left", fontsize=9,
               frameon=True, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


def plot_tradeoff_modality(df: pd.DataFrame, out_path: Path) -> None:
    """Scatter of (ISIC tight Dice) vs (CBIS-DDSM tight Dice) — modality-transfer trade-off."""
    fig, ax = plt.subplots(figsize=(9, 8))

    for m in METHOD_ORDER:
        points = []
        for training in ("pm=0", "pm=20", "rand100"):
            isic_tight = df[(df["method"] == m) & (df["training"] == training) &
                             (df["dataset"] == "isic2018_test") & (df["perturb_max_px"] == 0)]
            cbis_tight = df[(df["method"] == m) & (df["training"] == training) &
                             (df["dataset"] == "cbis_ddsm") & (df["perturb_max_px"] == 0)]
            if isic_tight.empty or cbis_tight.empty:
                continue
            points.append((training, float(isic_tight["dice_mean"].iloc[0]),
                            float(cbis_tight["dice_mean"].iloc[0])))

        if not points:
            continue

        for training, x, y in points:
            style = TRAINING_STYLES.get(training, {"marker": "o"})
            ax.scatter(x, y, color=METHOD_COLORS[m],
                        marker=style["marker"], s=180, edgecolors="black", linewidth=1.2,
                        zorder=5, label="_nolegend_")
            ax.annotate(training, (x, y), xytext=(7, -2),
                         textcoords="offset points", fontsize=8, color="#333")

        if len(points) > 1:
            xs = [p[1] for p in points]
            ys = [p[2] for p in points]
            ax.plot(xs, ys, color=METHOD_COLORS[m], linestyle="-", linewidth=1.2,
                     alpha=0.5, zorder=3)

    ax.set_xlim(0.85, 0.97); ax.set_ylim(0.10, 0.90)
    ax.set_xlabel("ISIC Dice at tight bbox (ID, dermoscopy)", fontsize=12)
    ax.set_ylabel("CBIS-DDSM Dice at tight bbox (far-OOD, mammography)", fontsize=12)
    ax.set_title(
        "Modality-transfer trade-off: ID accuracy vs far-OOD accuracy at tight bbox\n"
        "Right-and-down = ID-specialized; up = better modality transfer",
        fontsize=12,
    )

    method_handles = [
        plt.Line2D([], [], color=METHOD_COLORS[m], marker="o", linestyle="",
                    markersize=10, label=METHOD_LABELS[m], markeredgecolor="black")
        for m in METHOD_ORDER
    ]
    style_handles = [
        plt.Line2D([], [], color="gray", marker=TRAINING_STYLES[t]["marker"],
                    linestyle="", markersize=10, label=f"{t} trained",
                    markeredgecolor="black")
        for t in ("pm=0", "pm=20", "rand100")
    ]
    ax.legend(handles=method_handles + style_handles, loc="lower left", fontsize=9,
               frameon=True, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


# ---------------------------------------------------------------------------
# Plot 5: multi-heatmap — 4 datasets × 3 trainings, absolute Dice
# ---------------------------------------------------------------------------

def plot_multi_heatmap_3way(df: pd.DataFrame, out_path: Path) -> None:
    """4 rows (datasets) × 3 columns (trainings). Each cell is a small heatmap
    with methods on the y-axis and perturb levels on the x-axis. All cells
    share the same Dice color scale so you can read the entire experiment
    matrix in one glance.
    """
    n_rows = len(DATASET_ORDER)
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.2 * n_rows),
                              sharex=True, sharey=True)

    trainings = ("pm=0", "pm=20", "rand100")
    DICE_MIN, DICE_MAX = 0.10, 0.96

    for row_i, ds in enumerate(DATASET_ORDER):
        for col_i, training in enumerate(trainings):
            ax = axes[row_i, col_i]
            sub = df[(df["dataset"] == ds) & (df["training"] == training)]
            grid = pd.DataFrame(
                index=METHOD_ORDER, columns=PERTURBS, dtype=float,
            )
            for m in METHOD_ORDER:
                for p in PERTURBS:
                    matched = sub[(sub["method"] == m) & (sub["perturb_max_px"] == p)]
                    grid.loc[m, p] = float(matched["dice_mean"].iloc[0]) if not matched.empty else np.nan

            im = ax.imshow(grid.values.astype(float),
                            cmap="RdYlGn", vmin=DICE_MIN, vmax=DICE_MAX,
                            aspect="auto")
            ax.set_xticks(range(len(PERTURBS)))
            ax.set_xticklabels([str(p) for p in PERTURBS])
            ax.set_yticks(range(len(METHOD_ORDER)))
            ax.set_yticklabels([METHOD_LABELS[m] for m in METHOD_ORDER])
            if row_i == 0:
                ax.set_title(f"Trained: {training}", fontsize=12, fontweight="bold")
            if col_i == 0:
                ax.set_ylabel(DATASET_LABELS.get(ds, ds), fontsize=11, fontweight="bold")
            if row_i == n_rows - 1:
                ax.set_xlabel("Eval bbox max expansion (px)")
            # Cell-value annotations
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    val = float(grid.values[i, j])
                    if np.isnan(val):
                        continue
                    ax.text(j, i, f"{val:.2f}",
                             ha="center", va="center",
                             color="black" if val > 0.55 else "white",
                             fontsize=8.5)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.018, pad=0.02)
    cbar.set_label("Dice", fontsize=11)
    fig.suptitle(
        "Full experiment matrix: Dice across all methods × datasets × perturb levels × trainings",
        fontsize=13, y=1.00,
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


# ---------------------------------------------------------------------------
# Plot 6 / 7: grouped bar charts at tight bbox and extreme perturb
# ---------------------------------------------------------------------------

def _grouped_bars_panel(ax, df_panel, perturb: int) -> None:
    """One panel: x=method, hue=training, y=Dice at the given perturb level."""
    n_trainings = 3
    bar_w = 0.26
    x_positions = np.arange(len(METHOD_ORDER))
    training_palette = {"pm=0": "#1f77b4", "pm=20": "#ff7f0e", "rand100": "#2ca02c"}

    for ti, training in enumerate(("pm=0", "pm=20", "rand100")):
        vals = []
        for m in METHOD_ORDER:
            if m == "zero_shot" and training != "pm=0":
                vals.append(np.nan)  # skip — same as pm=0
                continue
            sub = df_panel[(df_panel["method"] == m) &
                           (df_panel["training"] == training) &
                           (df_panel["perturb_max_px"] == perturb)]
            vals.append(float(sub["dice_mean"].iloc[0]) if not sub.empty else np.nan)
        offsets = (ti - 1) * bar_w  # center the group
        bars = ax.bar(x_positions + offsets, vals, width=bar_w,
                       label=training, color=training_palette[training],
                       edgecolor="black", linewidth=0.5)
        # Annotate Dice values on top of each bar
        for x, v in zip(x_positions + offsets, vals):
            if not np.isnan(v):
                ax.text(x, v + 0.01, f"{v:.2f}",
                         ha="center", va="bottom", fontsize=7.5, rotation=0)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylabel("Dice")


def plot_bars_perturb(df: pd.DataFrame, out_path: Path, perturb: int,
                       title_suffix: str) -> None:
    """4 dataset panels, each showing grouped bars (method × training) at one perturb level."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes_flat = axes.flatten()

    for ax, ds in zip(axes_flat, DATASET_ORDER):
        sub = df[df["dataset"] == ds]
        _grouped_bars_panel(ax, sub, perturb)
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=12)

    # One shared legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#1f77b4", ec="black", label="pm=0 trained"),
        plt.Rectangle((0, 0), 1, 1, color="#ff7f0e", ec="black", label="pm=20 trained"),
        plt.Rectangle((0, 0), 1, 1, color="#2ca02c", ec="black", label="rand100 trained"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                bbox_to_anchor=(0.5, -0.02), fontsize=11)
    fig.suptitle(
        f"Dice by method × training condition  —  {title_suffix}",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


# ---------------------------------------------------------------------------
# Plot 8: Δ summary bars — average ΔDice from pm=0 baseline, per (method, dataset)
# ---------------------------------------------------------------------------

def plot_delta_summary_bars(df: pd.DataFrame, out_path: Path) -> None:
    """For each method × dataset, two grouped bars:
       - ΔDice (pm=20 vs pm=0), averaged across all 5 perturb levels
       - ΔDice (rand100 vs pm=0), averaged across all 5 perturb levels
       Shows which (method, dataset) cells benefit / regress from jitter training,
       and whether widening the jitter (rand100) amplifies the effect.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    axes_flat = axes.flatten()

    plot_methods = [m for m in METHOD_ORDER if m != "zero_shot"]
    bar_w = 0.35
    palette = {"pm=20": "#ff7f0e", "rand100": "#2ca02c"}

    for ax, ds in zip(axes_flat, DATASET_ORDER):
        sub = df[df["dataset"] == ds]

        deltas_pm20 = []
        deltas_rand100 = []
        for m in plot_methods:
            base = sub[(sub["method"] == m) & (sub["training"] == "pm=0")].set_index("perturb_max_px")["dice_mean"]
            t20  = sub[(sub["method"] == m) & (sub["training"] == "pm=20")].set_index("perturb_max_px")["dice_mean"]
            t100 = sub[(sub["method"] == m) & (sub["training"] == "rand100")].set_index("perturb_max_px")["dice_mean"]
            deltas_pm20.append((t20 - base).mean() if not t20.empty else np.nan)
            deltas_rand100.append((t100 - base).mean() if not t100.empty else np.nan)

        x_positions = np.arange(len(plot_methods))
        ax.bar(x_positions - bar_w / 2, deltas_pm20, width=bar_w,
                color=palette["pm=20"], edgecolor="black", linewidth=0.5,
                label="pm=20 − pm=0")
        ax.bar(x_positions + bar_w / 2, deltas_rand100, width=bar_w,
                color=palette["rand100"], edgecolor="black", linewidth=0.5,
                label="rand100 − pm=0")
        for x, v in zip(x_positions - bar_w / 2, deltas_pm20):
            if not np.isnan(v):
                ax.text(x, v + (0.005 if v >= 0 else -0.012), f"{v:+.2f}",
                         ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)
        for x, v in zip(x_positions + bar_w / 2, deltas_rand100):
            if not np.isnan(v):
                ax.text(x, v + (0.005 if v >= 0 else -0.012), f"{v:+.2f}",
                         ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([METHOD_LABELS[m] for m in plot_methods],
                            rotation=20, ha="right")
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=12)
        ax.set_ylabel("ΔDice (averaged across 5 perturb levels)")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(-0.35, 0.15)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#ff7f0e", ec="black", label="pm=20 trained vs pm=0 baseline"),
        plt.Rectangle((0, 0), 1, 1, color="#2ca02c", ec="black", label="rand100 trained vs pm=0 baseline"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
                bbox_to_anchor=(0.5, -0.02), fontsize=11)
    fig.suptitle(
        "Average ΔDice from pm=0 baseline (positive = jitter training helped; negative = it hurt)",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


# ---------------------------------------------------------------------------
# Long-form 3-way summary CSV
# ---------------------------------------------------------------------------

def write_summary_3way(df: pd.DataFrame, out_path: Path) -> None:
    wide = df.pivot_table(
        index=["method", "dataset", "perturb_max_px"],
        columns="training",
        values="dice_mean",
        aggfunc="first",
    ).reset_index()
    wide = wide.rename(columns={
        "pm=0": "dice_pm0_train", "pm=20": "dice_pm20_train", "rand100": "dice_rand100_train",
    })
    wide["delta_pm20_vs_pm0"] = wide["dice_pm20_train"] - wide["dice_pm0_train"]
    wide["delta_rand100_vs_pm0"] = wide["dice_rand100_train"] - wide["dice_pm0_train"]

    method_idx = {m: i for i, m in enumerate(METHOD_ORDER)}
    dataset_idx = {d: i for i, d in enumerate(DATASET_ORDER)}
    wide["_m"] = wide["method"].map(method_idx)
    wide["_d"] = wide["dataset"].map(dataset_idx)
    wide = wide.sort_values(["_m", "_d", "perturb_max_px"]).drop(columns=["_m", "_d"])
    for c in ("dice_pm0_train", "dice_pm20_train", "dice_rand100_train",
              "delta_pm20_vs_pm0", "delta_rand100_vs_pm0"):
        if c in wide.columns:
            wide[c] = wide[c].astype(float).round(4)
    wide.to_csv(out_path, index=False)
    print(f"[compare] wrote {out_path}")


# ---------------------------------------------------------------------------
# Regenerate summary_full.csv at the repo root (now includes rand100 column block)
# ---------------------------------------------------------------------------

def write_summary_full(df: pd.DataFrame, out_path: Path) -> None:
    """Per-dataset / per-method / per-training row, with all metrics × all perturb levels columns."""
    cols = ["dataset", "method", "training"]
    for metric in ("dice", "iou", "hd95"):
        for p in PERTURBS:
            cols.append(f"{metric}_pm{p}")

    # Build a lookup
    lookup = {}
    for _, r in df.iterrows():
        key = (r["dataset"], r["method"], r["training"], int(r["perturb_max_px"]))
        lookup[key] = (r["dice_mean"], r["iou_mean"], r["hd95_mean"])

    rows = []
    for ds in DATASET_ORDER:
        for m in METHOD_ORDER:
            if m == "zero_shot":
                row = {"dataset": ds, "method": m, "training": "n/a"}
                for metric_i, metric in enumerate(("dice", "iou", "hd95")):
                    for p in PERTURBS:
                        v = lookup.get((ds, m, "pm=0", p))
                        row[f"{metric}_pm{p}"] = f"{v[metric_i]:.4f}" if v else ""
                rows.append(row)
            else:
                for training in ("pm=0", "pm=20", "rand100"):
                    row = {"dataset": ds, "method": m, "training": training}
                    for metric_i, metric in enumerate(("dice", "iou", "hd95")):
                        for p in PERTURBS:
                            v = lookup.get((ds, m, training, p))
                            row[f"{metric}_pm{p}"] = f"{v[metric_i]:.4f}" if v else ""
                    rows.append(row)

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[compare] wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_combined()
    print(f"[compare] loaded {len(df)} rows "
          f"({df['method'].nunique()} methods × {df['dataset'].nunique()} datasets × "
          f"{df['perturb_max_px'].nunique()} perturb levels × {df['training'].nunique()} trainings)")

    # Original (kept for backward compat — busy but complete)
    plot_overlay_curves(df, OUT_DIR / "comparison_curves_3way.png")
    plot_delta_heatmap(df, OUT_DIR / "delta_heatmap_3way.png")
    plot_tradeoff_id_vs_perturbation(df, OUT_DIR / "tradeoff_id_vs_perturbation.png")
    plot_tradeoff_modality(df, OUT_DIR / "tradeoff_isic_vs_cbis.png")

    # New cleaner figures (the ones to actually look at)
    plot_multi_heatmap_3way(df, OUT_DIR / "multi_heatmap_3way.png")
    plot_bars_perturb(df, OUT_DIR / "bars_tight_bbox.png",
                       perturb=0, title_suffix="at tight bbox (pm=0)")
    plot_bars_perturb(df, OUT_DIR / "bars_extreme_perturb.png",
                       perturb=200, title_suffix="at extreme perturbation (pm=200)")
    plot_delta_summary_bars(df, OUT_DIR / "delta_summary_bars.png")

    write_summary_3way(df, OUT_SUMMARY_3WAY)
    write_summary_full(df, OUT_SUMMARY_FULL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
