"""Compare CKA-aware LoRA across 3 bbox-jitter training regimes ({no CKA, CKA late_l10} x {pm=0, pm=20, rand100})."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "cka" / "results"
FIGURES_DIR = REPO_ROOT / "cka" / "figures"
BBOX_ROBUST_DIR = REPO_ROOT / "bbox_robustness"

DATASETS = ["isic2018_test", "ph2", "busi", "cbis_ddsm"]
DATASET_LABELS = {
    "isic2018_test": "ISIC 2018 (ID)",
    "ph2":           "PH2 (near-OOD)",
    "busi":          "BUSI (far-OOD ultrasound)",
    "cbis_ddsm":     "CBIS-DDSM (far-OOD X-ray)",
}
TRAININGS = ["pm=0", "pm=20", "rand100"]
TRAINING_COLORS = {"pm=0": "#1f77b4", "pm=20": "#ff7f0e", "rand100": "#2ca02c"}
PERTURBS = [0, 20, 50, 100, 200]


# Load no-CKA baseline LoRA across the 3 trainings (multi-seed if available)
def load_no_cka_tight() -> dict[tuple[str, str], float]:
    """Returns {(training, dataset) -> dice} at tight bbox (pm=0 eval) for plain LoRA."""
    multiseed_csv = REPO_ROOT / "summary_full_multiseed.csv"
    if multiseed_csv.exists():
        df = pd.read_csv(multiseed_csv)
        df = df.rename(columns={"dice_mean_seeds": "dice_mean"})
        df = df[(df["method"] == "lora") & (df["perturb_max_px"] == 0)]
        return {
            (r["training"], r["dataset"]): float(r["dice_mean"])
            for _, r in df.iterrows()
        }
    # Fallback: hardcoded multi-seed averages
    return {
        ("pm=0",    "isic2018_test"): 0.9556, ("pm=0",    "ph2"): 0.9570,
        ("pm=0",    "busi"): 0.7801,           ("pm=0",    "cbis_ddsm"): 0.5094,
        ("pm=20",   "isic2018_test"): 0.9492, ("pm=20",   "ph2"): 0.9511,
        ("pm=20",   "busi"): 0.7745,           ("pm=20",   "cbis_ddsm"): 0.4370,
        ("rand100", "isic2018_test"): 0.9411, ("rand100", "ph2"): 0.9496,
        ("rand100", "busi"): 0.7467,           ("rand100", "cbis_ddsm"): 0.2143,
    }


def load_no_cka_bbox_robust() -> dict[tuple[str, str, int], float]:
    """{(training, dataset, perturb) -> dice} for no-CKA LoRA bbox robustness."""
    out = {}
    for training, suffix in [("pm=0", ""), ("pm=20", "_pm20"), ("rand100", "_rand100")]:
        for seed_dir in ["", "_seed0"]:
            csv_path = BBOX_ROBUST_DIR / f"results{suffix}{seed_dir}" / "runs.csv"
            if csv_path.exists():
                break
        else:
            # Try multi-seed average if present, else fall back to whatever exists
            csv_path = None
            for s in ("results_seed0", "results"):
                p = BBOX_ROBUST_DIR / f"{s}{suffix}" / "runs.csv"
                if p.exists():
                    csv_path = p
                    break
        if csv_path is None or not csv_path.exists():
            print(f"[compare] WARNING: no bbox robustness CSV for no-CKA training={training}")
            continue
        df = pd.read_csv(csv_path)
        df = df[df["method"] == "lora"]
        for _, r in df.iterrows():
            out[(training, r["dataset"], int(r["perturb_max_px"]))] = float(r["dice_mean"])
    return out


# Load CKA late_l10 results across the 3 trainings
def _extract_cka_tight_from_eval_csv(csv_path: Path, run_name_substr: str) -> dict[str, float]:
    """{dataset -> dice} for the single CKA run inside an eval CSV."""
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    df = df[df["run_name"].astype(str).str.contains(run_name_substr, regex=False)]
    if df.empty:
        return {}
    return {r["dataset"]: float(r["dice_mean"]) for _, r in df.iterrows()}


def load_cka_tight() -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    # pm=0 trained: from the OOD-only late CSV (3 lambdas, take l10)
    pm0 = _extract_cka_tight_from_eval_csv(
        RESULTS_DIR / "runs_cka_oodonly_late.csv",
        "lora_cka_oodonly_late_l10_seed0",
    )
    for ds, d in pm0.items():
        out[("pm=0", ds)] = d
    # pm=20 trained
    pm20 = _extract_cka_tight_from_eval_csv(
        RESULTS_DIR / "runs_cka_oodonly_late_l10_pm20.csv",
        "lora_cka_oodonly_late_l10_pm20_seed0",
    )
    for ds, d in pm20.items():
        out[("pm=20", ds)] = d
    # rand100 trained
    rand100 = _extract_cka_tight_from_eval_csv(
        RESULTS_DIR / "runs_cka_oodonly_late_l10_rand100.csv",
        "lora_cka_oodonly_late_l10_rand100_seed0",
    )
    for ds, d in rand100.items():
        out[("rand100", ds)] = d
    return out


def load_cka_bbox_robust() -> dict[tuple[str, str, int], float]:
    """{(training, dataset, perturb) -> dice} for CKA-aware late_l10 bbox robustness."""
    out: dict[tuple[str, str, int], float] = {}
    for training, dir_name in [
        ("pm=20",   "results_cka_oodonly_late_l10_pm20"),
        ("rand100", "results_cka_oodonly_late_l10_rand100"),
    ]:
        csv_path = BBOX_ROBUST_DIR / dir_name / "runs.csv"
        if not csv_path.exists():
            print(f"[compare] WARNING: no bbox robustness for CKA training={training} ({csv_path})")
            continue
        df = pd.read_csv(csv_path)
        df = df[df["method"] == "lora"]
        for _, r in df.iterrows():
            out[(training, r["dataset"], int(r["perturb_max_px"]))] = float(r["dice_mean"])
    return out


# Plot 1: tight-bbox grouped bars per dataset
def plot_tight_bars(no_cka: dict, cka: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    x = np.arange(len(TRAININGS))
    width = 0.36

    for col_i, ds in enumerate(DATASETS):
        ax = axes[col_i]
        no_cka_vals = [no_cka.get((t, ds), float("nan")) for t in TRAININGS]
        cka_vals    = [cka.get((t, ds), float("nan")) for t in TRAININGS]

        b1 = ax.bar(x - width/2, no_cka_vals, width,
                    color="#7f7f7f", edgecolor="black", label="No CKA")
        b2 = ax.bar(x + width/2, cka_vals, width,
                    color="#d62728", edgecolor="black", label="CKA late_l10")

        for bars in (b1, b2):
            for b, v in zip(bars, [no_cka_vals if bars is b1 else cka_vals][0]):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}",
                            ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(TRAININGS)
        ax.set_title(DATASET_LABELS[ds], fontsize=11)
        ax.set_ylabel("Dice")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        if col_i == 0:
            ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("CKA-aware late_l10 vs baseline LoRA across 3 bbox-jitter trainings\n"
                 "(tight-bbox eval, single seed for CKA, multi-seed avg for no-CKA)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


# Plot 2: bbox robustness curves per dataset
def plot_robustness_curves(no_cka_robust: dict, cka_robust: dict,
                           no_cka_tight: dict, cka_tight: dict,
                           out_path: Path) -> None:
    """4 panels (one per dataset). x = perturb level. 6 lines (3 trainings x CKA on/off)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes_flat = axes.flatten()

    for ax, ds in zip(axes_flat, DATASETS):
        for training in TRAININGS:
            # No-CKA line
            ys_nocka = []
            for p in PERTURBS:
                if p == 0:
                    ys_nocka.append(no_cka_tight.get((training, ds), float("nan")))
                else:
                    ys_nocka.append(no_cka_robust.get((training, ds, p), float("nan")))
            ax.plot(PERTURBS, ys_nocka,
                    color=TRAINING_COLORS[training], linestyle="--", marker="o",
                    linewidth=1.5, alpha=0.6,
                    label=f"No CKA, {training}")

            # CKA line
            ys_cka = []
            for p in PERTURBS:
                if p == 0:
                    ys_cka.append(cka_tight.get((training, ds), float("nan")))
                else:
                    ys_cka.append(cka_robust.get((training, ds, p), float("nan")))
            ax.plot(PERTURBS, ys_cka,
                    color=TRAINING_COLORS[training], linestyle="-", marker="s",
                    linewidth=2.0,
                    label=f"CKA late_l10, {training}")

        ax.set_xlabel("Eval bbox max expansion (px)")
        ax.set_ylabel("Dice")
        ax.set_title(DATASET_LABELS[ds], fontsize=12)
        ax.grid(alpha=0.3)
        ax.set_xticks(PERTURBS)
        ax.legend(loc="best", fontsize=7, ncol=2)

    fig.suptitle("Bbox robustness curves: CKA-aware vs baseline LoRA, across 3 trainings",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


# Plot 3: ID vs far-OOD trade-off scatter (ISIC tight vs CBIS tight)
def plot_tradeoff(no_cka_tight: dict, cka_tight: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 9))

    for training in TRAININGS:
        # No CKA point
        x = no_cka_tight.get((training, "isic2018_test"), float("nan"))
        y = no_cka_tight.get((training, "cbis_ddsm"), float("nan"))
        if not (np.isnan(x) or np.isnan(y)):
            ax.scatter(x, y, marker="o", s=240, color=TRAINING_COLORS[training],
                       edgecolors="black", linewidth=1.5, alpha=0.7, zorder=3)
            ax.annotate(f"NoCKA\n{training}", (x, y), xytext=(7, 7),
                        textcoords="offset points", fontsize=9)

        # CKA point
        x = cka_tight.get((training, "isic2018_test"), float("nan"))
        y = cka_tight.get((training, "cbis_ddsm"), float("nan"))
        if not (np.isnan(x) or np.isnan(y)):
            ax.scatter(x, y, marker="*", s=400, color=TRAINING_COLORS[training],
                       edgecolors="black", linewidth=1.5, zorder=5)
            ax.annotate(f"CKA\n{training}", (x, y), xytext=(7, -22),
                        textcoords="offset points", fontsize=9, fontweight="bold")

        # Connect the matched pair with a thin arrow
        x0 = no_cka_tight.get((training, "isic2018_test"), float("nan"))
        y0 = no_cka_tight.get((training, "cbis_ddsm"), float("nan"))
        x1 = cka_tight.get((training, "isic2018_test"), float("nan"))
        y1 = cka_tight.get((training, "cbis_ddsm"), float("nan"))
        if not any(np.isnan(v) for v in (x0, y0, x1, y1)):
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color=TRAINING_COLORS[training],
                                         lw=1.5, alpha=0.5))

    ax.set_xlabel("ISIC 2018 Dice (in-domain)", fontsize=12)
    ax.set_ylabel("CBIS-DDSM Dice (far-OOD mammography)", fontsize=12)
    ax.set_title("CKA effect on ID vs far-OOD trade-off across bbox-jitter trainings\n"
                 "Circle = no CKA, star = CKA late_l10. Arrow shows the CKA effect.",
                 fontsize=11)
    ax.grid(alpha=0.3)

    # Legend
    color_legend = [
        plt.Line2D([], [], color=TRAINING_COLORS[t], marker="o", linestyle="",
                   markersize=11, label=f"{t} training", markeredgecolor="black")
        for t in TRAININGS
    ]
    marker_legend = [
        plt.Line2D([], [], color="gray", marker="o", linestyle="",
                   markersize=11, label="No CKA", markeredgecolor="black"),
        plt.Line2D([], [], color="gray", marker="*", linestyle="",
                   markersize=15, label="CKA late_l10", markeredgecolor="black"),
    ]
    ax.legend(handles=color_legend + marker_legend, loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


# Plot 4: delta heatmap (CKA - no_CKA) per (training, dataset)
def plot_delta_heatmap(no_cka_tight: dict, cka_tight: dict, out_path: Path) -> None:
    grid = np.zeros((len(TRAININGS), len(DATASETS)))
    for i, training in enumerate(TRAININGS):
        for j, ds in enumerate(DATASETS):
            v_no = no_cka_tight.get((training, ds), float("nan"))
            v_ck = cka_tight.get((training, ds), float("nan"))
            grid[i, j] = (v_ck - v_no) if not (np.isnan(v_no) or np.isnan(v_ck)) else np.nan

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-0.10, vmax=0.10, aspect="auto")
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], rotation=15, ha="right")
    ax.set_yticks(range(len(TRAININGS)))
    ax.set_yticklabels(TRAININGS)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid[i, j]
            if np.isnan(val):
                continue
            color = "white" if abs(val) > 0.07 else "black"
            ax.text(j, i, f"{val:+.4f}", ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Delta Dice (CKA - no CKA)")
    ax.set_title("Delta Dice from adding CKA late_l10, per training x dataset\n"
                 "Red = CKA helps, Blue = CKA hurts",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


def write_summary(no_cka_tight: dict, cka_tight: dict, out_csv: Path, out_md: Path) -> None:
    rows = []
    for training in TRAININGS:
        for ds in DATASETS:
            v_no = no_cka_tight.get((training, ds), float("nan"))
            v_ck = cka_tight.get((training, ds), float("nan"))
            delta = (v_ck - v_no) if not (np.isnan(v_no) or np.isnan(v_ck)) else float("nan")
            rows.append({
                "training": training,
                "dataset":  ds,
                "no_cka_dice": round(v_no, 4) if not np.isnan(v_no) else "",
                "cka_dice":    round(v_ck, 4) if not np.isnan(v_ck) else "",
                "delta":       round(delta, 4) if not np.isnan(delta) else "",
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[compare] wrote {out_csv}")

    lines = ["# CKA late_l10 effect across bbox-jitter trainings\n",
             "Single seed for CKA, multi-seed avg for no-CKA baseline.\n\n",
             "| Training | Dataset | No-CKA Dice | CKA Dice | Delta |",
             "|---|---|---:|---:|---:|"]
    for r in rows:
        delta_str = (f"{float(r['delta']):+.4f}" if r['delta'] != "" else "-")
        lines.append(
            f"| {r['training']} | {DATASET_LABELS[r['dataset']]} | "
            f"{r['no_cka_dice']} | {r['cka_dice']} | {delta_str} |"
        )

    # Per-training summary
    lines.append("\n## Per-training summary\n")
    lines.append("| Training | CKA helps far-OOD? | Best at far-OOD overall? |")
    lines.append("|---|---|---|")
    for training in TRAININGS:
        busi_d = next((float(r['delta']) for r in rows
                       if r['training'] == training and r['dataset'] == 'busi'
                       and r['delta'] != ''), float('nan'))
        cbis_d = next((float(r['delta']) for r in rows
                       if r['training'] == training and r['dataset'] == 'cbis_ddsm'
                       and r['delta'] != ''), float('nan'))
        helps = "yes" if (not np.isnan(cbis_d) and cbis_d > 0) else "no"
        notes = ""
        if training == "rand100":
            notes = "CKA + heavy random jitter overfits"
        lines.append(f"| {training} | BUSI {busi_d:+.4f}, CBIS {cbis_d:+.4f} -> {helps} | {notes} |")

    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"[compare] wrote {out_md}")


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    no_cka_tight  = load_no_cka_tight()
    cka_tight     = load_cka_tight()
    no_cka_robust = load_no_cka_bbox_robust()
    cka_robust    = load_cka_bbox_robust()

    print(f"[compare] loaded {len(no_cka_tight)} no-CKA tight cells, "
          f"{len(cka_tight)} CKA tight cells, "
          f"{len(no_cka_robust)} no-CKA robust cells, "
          f"{len(cka_robust)} CKA robust cells")

    plot_tight_bars(no_cka_tight, cka_tight, FIGURES_DIR / "cka_jitter_tight_bbox_bars.png")
    plot_robustness_curves(no_cka_robust, cka_robust, no_cka_tight, cka_tight,
                           FIGURES_DIR / "cka_jitter_robustness_curves.png")
    plot_tradeoff(no_cka_tight, cka_tight, FIGURES_DIR / "cka_jitter_tradeoff_scatter.png")
    plot_delta_heatmap(no_cka_tight, cka_tight, FIGURES_DIR / "cka_jitter_delta_heatmap.png")
    write_summary(no_cka_tight, cka_tight,
                  RESULTS_DIR / "cka_jitter_comparison.csv",
                  RESULTS_DIR / "cka_jitter_comparison.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
