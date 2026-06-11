"""Three-way comparison of pm=0, pm=20, rand100 trained models, mean over seeds 0,1,2 with +/-1 sigma bands."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = REPO_ROOT / "summary_full_multiseed.csv"

OUT_DIR = REPO_ROOT / "bbox_robustness" / "comparison"
OUT_SUMMARY_3WAY = OUT_DIR / "comparison_summary_3way_multiseed.csv"

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
    "pm=0":    {"linestyle": "-",  "marker": "o", "alpha": 1.00, "linewidth": 1.8},
    "pm=20":   {"linestyle": "--", "marker": "s", "alpha": 0.90, "linewidth": 1.6},
    "rand100": {"linestyle": ":",  "marker": "^", "alpha": 0.85, "linewidth": 2.4},
}
DATASET_LABELS = {
    "isic2018_test": "ISIC 2018 (ID)",
    "ph2":           "PH² (near-OOD)",
    "busi":          "BUSI (far-OOD ultrasound)",
    "cbis_ddsm":     "CBIS-DDSM (far-OOD mammography)",
}
DATASET_ORDER = ["isic2018_test", "ph2", "busi", "cbis_ddsm"]
PERTURBS = [0, 20, 50, 100, 200]


def load_multiseed() -> pd.DataFrame:
    """Returns multiseed DataFrame with dice_mean/dice_std renamed from the seed-aggregated columns."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"{INPUT_CSV} not found; run `python scripts/aggregate_seeds.py` first."
        )
    df = pd.read_csv(INPUT_CSV)
    df = df.rename(columns={
        "dice_mean_seeds": "dice_mean",
        "dice_std_seeds":  "dice_std",
    })
    df["perturb_max_px"] = df["perturb_max_px"].astype(int)
    return df


def lookup(df: pd.DataFrame, method: str, training: str, dataset: str,
           perturb: int) -> tuple[float, float] | None:
    """Returns (mean, std) or None if not present."""
    sub = df[(df["method"] == method) & (df["training"] == training) &
             (df["dataset"] == dataset) & (df["perturb_max_px"] == perturb)]
    if sub.empty:
        return None
    return float(sub["dice_mean"].iloc[0]), float(sub["dice_std"].iloc[0])


def plot_overlay_curves(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True, sharey=False)
    axes_flat = axes.flatten()

    for ax, ds in zip(axes_flat, DATASET_ORDER):
        sub = df[df["dataset"] == ds]
        for m in METHOD_ORDER:
            if m == "zero_shot":
                sub_m = sub[(sub["method"] == m) & (sub["training"] == "pm=0")].sort_values("perturb_max_px")
                if not sub_m.empty:
                    x = sub_m["perturb_max_px"].values
                    y = sub_m["dice_mean"].values
                    sd = sub_m["dice_std"].values
                    ax.plot(x, y, marker="o", linewidth=2.5, color=METHOD_COLORS[m],
                            label=METHOD_LABELS[m], zorder=4)
                    ax.fill_between(x, y - sd, y + sd, color=METHOD_COLORS[m], alpha=0.12, zorder=1)
                continue

            for training in ("pm=0", "pm=20", "rand100"):
                sub_t = sub[(sub["method"] == m) & (sub["training"] == training)].sort_values("perturb_max_px")
                if sub_t.empty:
                    continue
                style = TRAINING_STYLES[training]
                x = sub_t["perturb_max_px"].values
                y = sub_t["dice_mean"].values
                sd = sub_t["dice_std"].values
                ax.plot(x, y, color=METHOD_COLORS[m],
                        linestyle=style["linestyle"], marker=style["marker"],
                        linewidth=style["linewidth"], alpha=style["alpha"],
                        markersize=5, label="_nolegend_",
                        zorder=2 + ("pm=0", "pm=20", "rand100").index(training))
                ax.fill_between(x, y - sd, y + sd, color=METHOD_COLORS[m], alpha=0.08, zorder=1)

        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=12)
        ax.set_xlabel("Eval-time bbox max expansion (px)")
        ax.set_ylabel("Dice (mean +/- std over 3 seeds)")
        ax.grid(alpha=0.3)
        ax.set_xticks(PERTURBS)

    method_handles = [
        plt.Line2D([], [], color=METHOD_COLORS[m], marker="o", linewidth=2,
                   label=METHOD_LABELS[m])
        for m in METHOD_ORDER
    ]
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
        "Bbox robustness across three training conditions (3 seeds, +/-std shaded)  "
        "(solid pm=0, dashed pm=20, dotted rand100)",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


def plot_delta_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """Delta-Dice heatmaps vs pm=0 baseline (top pm=20, bottom rand100); cells with |delta| > 2x pooled sigma are bold."""
    fig, axes = plt.subplots(2, len(DATASET_ORDER), figsize=(20, 9),
                             sharex=True, sharey=True)
    DELTA_MAX = 0.30

    for row_i, target in enumerate(("pm=20", "rand100")):
        for col_i, ds in enumerate(DATASET_ORDER):
            ax = axes[row_i, col_i]

            methods_no_zs = [m for m in METHOD_ORDER if m != "zero_shot"]
            delta_grid = np.full((len(methods_no_zs), len(PERTURBS)), np.nan)
            sig_grid   = np.full((len(methods_no_zs), len(PERTURBS)), False)

            for i, m in enumerate(methods_no_zs):
                for j, p in enumerate(PERTURBS):
                    a = lookup(df, m, target, ds, p)
                    b = lookup(df, m, "pm=0",  ds, p)
                    if a is None or b is None:
                        continue
                    delta = a[0] - b[0]
                    pooled = float(np.sqrt(a[1]**2 + b[1]**2))
                    delta_grid[i, j] = delta
                    # 2-sigma ~ 95% CI excludes zero
                    sig_grid[i, j] = abs(delta) > 2 * max(pooled, 1e-9)

            im = ax.imshow(delta_grid, cmap="RdBu_r",
                           vmin=-DELTA_MAX, vmax=DELTA_MAX, aspect="auto")
            ax.set_xticks(range(len(PERTURBS)))
            ax.set_xticklabels([str(p) for p in PERTURBS])
            ax.set_yticks(range(len(methods_no_zs)))
            ax.set_yticklabels([METHOD_LABELS[m] for m in methods_no_zs])
            if row_i == 1:
                ax.set_xlabel("Eval bbox max expansion (px)")
            if col_i == 0:
                ax.set_ylabel(f"{target} train\n- pm=0 train", fontsize=11, fontweight="bold")
            if row_i == 0:
                ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=11)

            for i in range(delta_grid.shape[0]):
                for j in range(delta_grid.shape[1]):
                    val = delta_grid[i, j]
                    if np.isnan(val):
                        continue
                    weight = "bold" if sig_grid[i, j] else "normal"
                    ax.text(j, i, f"{val:+.2f}",
                            ha="center", va="center",
                            color="white" if abs(val) > 0.20 else "black",
                            fontsize=7.8, fontweight=weight)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.018, pad=0.02)
    cbar.set_label("delta Dice (target - pm=0 baseline)  -  bold = |delta| > 2 sigma_pooled")
    fig.suptitle(
        "Effect of bbox jitter training on Dice (3 seeds, bold cells exceed 2 sigma)  "
        "(top: pm=20, bottom: rand100)",
        fontsize=13, y=0.99,
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


def _scatter_with_errorbars(ax, df, x_query, y_query, lims_x, lims_y,
                            xlabel, ylabel, title, draw_diag: bool):
    """Generic scatter: x_query/y_query return (mean, std) per (method, training)."""
    for m in METHOD_ORDER:
        pts = []  # (training, x_mean, y_mean, x_std, y_std)
        for tr in ("pm=0", "pm=20", "rand100"):
            x = x_query(df, m, tr)
            y = y_query(df, m, tr)
            if x is None or y is None:
                continue
            pts.append((tr, x[0], y[0], x[1], y[1]))
        if not pts:
            continue
        for tr, xm, ym, xs, ys in pts:
            style = TRAINING_STYLES[tr]
            ax.errorbar(xm, ym, xerr=xs, yerr=ys,
                        fmt=style["marker"], color=METHOD_COLORS[m],
                        markersize=10, markeredgecolor="black", markeredgewidth=1.0,
                        ecolor=METHOD_COLORS[m], capsize=3, alpha=0.9, zorder=5)
            ax.annotate(tr, (xm, ym), xytext=(8, -3),
                        textcoords="offset points", fontsize=8, color="#333")
        if len(pts) > 1:
            xs = [p[1] for p in pts]
            ys = [p[2] for p in pts]
            ax.plot(xs, ys, color=METHOD_COLORS[m], linestyle="-",
                    linewidth=1.2, alpha=0.45, zorder=3)

    if draw_diag:
        d = [min(lims_x[0], lims_y[0]), max(lims_x[1], lims_y[1])]
        ax.plot(d, d, "k:", alpha=0.4, label="y = x")

    ax.set_xlim(*lims_x)
    ax.set_ylim(*lims_y)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12)

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


def plot_tradeoff_id_vs_perturbation(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    _scatter_with_errorbars(
        ax, df,
        x_query=lambda d, m, tr: lookup(d, m, tr, "isic2018_test", 0),
        y_query=lambda d, m, tr: lookup(d, m, tr, "isic2018_test", 200),
        lims_x=(0.65, 0.97), lims_y=(0.65, 0.90),
        xlabel="ISIC Dice at tight bbox (pm=0)",
        ylabel="ISIC Dice at extreme perturbation (pm=200)",
        title="Trade-off: tight-bbox accuracy vs prompt robustness (ISIC, 3 seeds, error bars = +/-1 sigma)\n"
              "upper-right corner = better at both",
        draw_diag=True,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


def plot_tradeoff_modality(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    _scatter_with_errorbars(
        ax, df,
        x_query=lambda d, m, tr: lookup(d, m, tr, "isic2018_test", 0),
        y_query=lambda d, m, tr: lookup(d, m, tr, "cbis_ddsm",     0),
        lims_x=(0.85, 0.97), lims_y=(0.10, 0.90),
        xlabel="ISIC Dice at tight bbox (ID, dermoscopy)",
        ylabel="CBIS-DDSM Dice at tight bbox (far-OOD, mammography)",
        title="Modality-transfer trade-off (3 seeds, error bars = +/-1 sigma)\n"
              "right-down = ID-specialised, up = better modality transfer",
        draw_diag=False,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


def plot_multi_heatmap_3way(df: pd.DataFrame, out_path: Path) -> None:
    n_rows = len(DATASET_ORDER)
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.2 * n_rows),
                             sharex=True, sharey=True)
    trainings = ("pm=0", "pm=20", "rand100")
    DICE_MIN, DICE_MAX = 0.10, 0.96

    for row_i, ds in enumerate(DATASET_ORDER):
        for col_i, tr in enumerate(trainings):
            ax = axes[row_i, col_i]
            grid_mean = np.full((len(METHOD_ORDER), len(PERTURBS)), np.nan)
            grid_std  = np.full_like(grid_mean, np.nan)
            for i, m in enumerate(METHOD_ORDER):
                for j, p in enumerate(PERTURBS):
                    src = "pm=0" if m == "zero_shot" else tr
                    v = lookup(df, m, src, ds, p)
                    if v is not None:
                        grid_mean[i, j] = v[0]
                        grid_std[i, j]  = v[1]

            im = ax.imshow(grid_mean, cmap="RdYlGn",
                           vmin=DICE_MIN, vmax=DICE_MAX, aspect="auto")
            ax.set_xticks(range(len(PERTURBS)))
            ax.set_xticklabels([str(p) for p in PERTURBS])
            ax.set_yticks(range(len(METHOD_ORDER)))
            ax.set_yticklabels([METHOD_LABELS[m] for m in METHOD_ORDER])
            if row_i == 0:
                ax.set_title(f"Trained: {tr}", fontsize=12, fontweight="bold")
            if col_i == 0:
                ax.set_ylabel(DATASET_LABELS.get(ds, ds), fontsize=11, fontweight="bold")
            if row_i == n_rows - 1:
                ax.set_xlabel("Eval bbox max expansion (px)")

            for i in range(grid_mean.shape[0]):
                for j in range(grid_mean.shape[1]):
                    val = grid_mean[i, j]
                    sd  = grid_std[i, j]
                    if np.isnan(val):
                        continue
                    col = "black" if val > 0.55 else "white"
                    ax.text(j, i - 0.12, f"{val:.2f}",
                            ha="center", va="center", color=col, fontsize=8.5)
                    if not np.isnan(sd) and sd > 1e-4:
                        ax.text(j, i + 0.20, f"±{sd:.2f}",
                                ha="center", va="center", color=col, fontsize=6.5,
                                fontstyle="italic")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.018, pad=0.02)
    cbar.set_label("Dice (mean over 3 seeds)", fontsize=11)
    fig.suptitle(
        "Full experiment matrix: Dice across methods x datasets x perturb x trainings (3 seeds)",
        fontsize=13, y=1.00,
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


def _grouped_bars_panel(ax, df, ds: str, perturb: int) -> None:
    bar_w = 0.26
    x_positions = np.arange(len(METHOD_ORDER))
    palette = {"pm=0": "#1f77b4", "pm=20": "#ff7f0e", "rand100": "#2ca02c"}

    for ti, tr in enumerate(("pm=0", "pm=20", "rand100")):
        means = []
        stds  = []
        for m in METHOD_ORDER:
            if m == "zero_shot" and tr != "pm=0":
                means.append(np.nan); stds.append(np.nan)
                continue
            v = lookup(df, m, tr if m != "zero_shot" else "pm=0", ds, perturb)
            means.append(v[0] if v else np.nan)
            stds.append(v[1] if v else np.nan)
        offsets = (ti - 1) * bar_w
        ax.bar(x_positions + offsets, means, width=bar_w,
               yerr=stds, label=tr, color=palette[tr],
               edgecolor="black", linewidth=0.5, capsize=3,
               error_kw={"elinewidth": 0.8, "ecolor": "#333"})
        for x, v in zip(x_positions + offsets, means):
            if not np.isnan(v):
                ax.text(x, v + 0.02, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=20, ha="right")
    ax.set_ylim(0, 1.10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylabel("Dice")


def plot_bars_perturb(df: pd.DataFrame, out_path: Path, perturb: int,
                      title_suffix: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes_flat = axes.flatten()
    for ax, ds in zip(axes_flat, DATASET_ORDER):
        _grouped_bars_panel(ax, df, ds, perturb)
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=12)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#1f77b4", ec="black", label="pm=0 trained"),
        plt.Rectangle((0, 0), 1, 1, color="#ff7f0e", ec="black", label="pm=20 trained"),
        plt.Rectangle((0, 0), 1, 1, color="#2ca02c", ec="black", label="rand100 trained"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=11)
    fig.suptitle(
        f"Dice by method x training condition  -  {title_suffix}  (3 seeds, +/-1 sigma error bars)",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


def plot_delta_summary_bars(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    axes_flat = axes.flatten()
    plot_methods = [m for m in METHOD_ORDER if m != "zero_shot"]
    bar_w = 0.35
    palette = {"pm=20": "#ff7f0e", "rand100": "#2ca02c"}

    for ax, ds in zip(axes_flat, DATASET_ORDER):
        d_pm20 = []
        d_r100 = []
        s_pm20 = []
        s_r100 = []
        for m in plot_methods:
            deltas20 = []
            deltas100 = []
            var20 = []
            var100 = []
            for p in PERTURBS:
                base = lookup(df, m, "pm=0", ds, p)
                t20  = lookup(df, m, "pm=20", ds, p)
                t100 = lookup(df, m, "rand100", ds, p)
                if base and t20:
                    deltas20.append(t20[0] - base[0])
                    var20.append(t20[1]**2 + base[1]**2)
                if base and t100:
                    deltas100.append(t100[0] - base[0])
                    var100.append(t100[1]**2 + base[1]**2)
            d_pm20.append(np.mean(deltas20) if deltas20 else np.nan)
            d_r100.append(np.mean(deltas100) if deltas100 else np.nan)
            # error bar = sqrt(mean of pooled variances) across perturbs
            s_pm20.append(np.sqrt(np.mean(var20)) if var20 else np.nan)
            s_r100.append(np.sqrt(np.mean(var100)) if var100 else np.nan)

        x_positions = np.arange(len(plot_methods))
        ax.bar(x_positions - bar_w / 2, d_pm20, width=bar_w, yerr=s_pm20,
               color=palette["pm=20"], edgecolor="black", linewidth=0.5,
               capsize=3, error_kw={"elinewidth": 0.8, "ecolor": "#333"},
               label="pm=20 - pm=0")
        ax.bar(x_positions + bar_w / 2, d_r100, width=bar_w, yerr=s_r100,
               color=palette["rand100"], edgecolor="black", linewidth=0.5,
               capsize=3, error_kw={"elinewidth": 0.8, "ecolor": "#333"},
               label="rand100 - pm=0")
        for x, v in zip(x_positions - bar_w / 2, d_pm20):
            if not np.isnan(v):
                ax.text(x, v + (0.005 if v >= 0 else -0.012), f"{v:+.2f}",
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)
        for x, v in zip(x_positions + bar_w / 2, d_r100):
            if not np.isnan(v):
                ax.text(x, v + (0.005 if v >= 0 else -0.012), f"{v:+.2f}",
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([METHOD_LABELS[m] for m in plot_methods],
                           rotation=20, ha="right")
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=12)
        ax.set_ylabel("delta Dice (averaged across 5 perturb levels)")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(-0.45, 0.20)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#ff7f0e", ec="black", label="pm=20 trained - pm=0 baseline"),
        plt.Rectangle((0, 0), 1, 1, color="#2ca02c", ec="black", label="rand100 trained - pm=0 baseline"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=11)
    fig.suptitle(
        "Average delta Dice from pm=0 baseline (3 seeds, error bars = pooled sigma)  "
        "(positive = jitter helped, negative = it hurt)",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


def write_summary_3way(df: pd.DataFrame, out_path: Path) -> None:
    """One row per (method, dataset, perturb_max_px) with mean+std per training plus delta columns."""
    methods = sorted(df["method"].unique())
    datasets = sorted(df["dataset"].unique())
    perturbs = sorted(df["perturb_max_px"].unique())

    rows = []
    for m in methods:
        for ds in datasets:
            for p in perturbs:
                row = {"method": m, "dataset": ds, "perturb_max_px": int(p)}
                vals = {}
                for tr in ("pm=0", "pm=20", "rand100"):
                    v = lookup(df, m, tr, ds, p)
                    if v is None:
                        row[f"dice_{tr.replace('=','').replace('rand','rand')}_mean"] = ""
                        row[f"dice_{tr.replace('=','').replace('rand','rand')}_std"] = ""
                        vals[tr] = None
                    else:
                        row[f"dice_{tr.replace('=','').replace('rand','rand')}_mean"] = round(v[0], 4)
                        row[f"dice_{tr.replace('=','').replace('rand','rand')}_std"] = round(v[1], 4)
                        vals[tr] = v
                if vals["pm=0"]:
                    if vals["pm=20"]:
                        row["delta_pm20_vs_pm0"] = round(vals["pm=20"][0] - vals["pm=0"][0], 4)
                    if vals["rand100"]:
                        row["delta_rand100_vs_pm0"] = round(vals["rand100"][0] - vals["pm=0"][0], 4)
                rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"[compare] wrote {out_path}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_multiseed()
    n_cells = len(df)
    print(f"[compare] loaded {n_cells} multiseed cells "
          f"({df['method'].nunique()} methods x {df['dataset'].nunique()} datasets x "
          f"{df['perturb_max_px'].nunique()} perturbs x {df['training'].nunique()} trainings)")

    plot_overlay_curves(df,            OUT_DIR / "comparison_curves_3way.png")
    plot_delta_heatmap(df,             OUT_DIR / "delta_heatmap_3way.png")
    plot_tradeoff_id_vs_perturbation(df, OUT_DIR / "tradeoff_id_vs_perturbation.png")
    plot_tradeoff_modality(df,         OUT_DIR / "tradeoff_isic_vs_cbis.png")
    plot_multi_heatmap_3way(df,        OUT_DIR / "multi_heatmap_3way.png")
    plot_bars_perturb(df,              OUT_DIR / "bars_tight_bbox.png",
                      perturb=0, title_suffix="at tight bbox (pm=0)")
    plot_bars_perturb(df,              OUT_DIR / "bars_extreme_perturb.png",
                      perturb=200, title_suffix="at extreme perturbation (pm=200)")
    plot_delta_summary_bars(df,        OUT_DIR / "delta_summary_bars.png")

    write_summary_3way(df, OUT_SUMMARY_3WAY)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
