"""Compare CKA sweep results across two probe compositions (original vs OOD-only)."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "cka" / "results"
FIGURES_DIR = REPO_ROOT / "cka" / "figures"
BASELINE_CSV = REPO_ROOT / "results" / "runs.csv"

POSITIONS = ("early", "mid", "late")
LAMBDAS = {"01": 0.1, "1": 1.0, "10": 10.0}
LAMBDA_LABELS = {0.1: "lambda=0.1", 1.0: "lambda=1.0", 10.0: "lambda=10.0"}
DATASETS = ["isic2018_test", "ph2", "busi", "cbis_ddsm"]
DATASET_LABELS = {
    "isic2018_test": "ISIC 2018 (ID)",
    "ph2":           "PH2 (near-OOD)",
    "busi":          "BUSI (far-OOD US)",
    "cbis_ddsm":     "CBIS-DDSM (far-OOD X-ray)",
}
POSITION_COLORS = {"early": "#1f77b4", "mid": "#ff7f0e", "late": "#d62728"}
PROBE_COLORS = {"original": "#0072B2", "oodonly": "#D55E00"}
PROBE_LABELS = {
    "original": "Original probe (12 ISIC + 10 BUSI + 10 CBIS)",
    "oodonly":  "OOD-only probe (0 ISIC + 16 BUSI + 16 CBIS)",
}


def parse_run_name(run_name: str) -> tuple[str, str, float] | None:
    """Extract (probe, position, lambda) from a run_name."""
    m = re.match(r"^lora_cka_(oodonly_)?(early|mid|late)_l([0-9]+)_seed\d+$", run_name)
    if not m:
        return None
    probe = "oodonly" if m.group(1) else "original"
    position = m.group(2)
    lambda_str = m.group(3)
    if lambda_str not in LAMBDAS:
        return None
    return probe, position, LAMBDAS[lambda_str]


def load_all_results() -> pd.DataFrame:
    """Combine all 6 CSVs (2 probes x 3 positions) into one long-form DataFrame."""
    frames = []
    for position in POSITIONS:
        for probe_tag, suffix in (("original", ""), ("oodonly", "_oodonly")):
            csv_path = RESULTS_DIR / f"runs_cka{suffix}_{position}.csv"
            if not csv_path.exists():
                print(f"[compare] WARNING: {csv_path} missing, skipping ({probe_tag}, {position})")
                continue
            df = pd.read_csv(csv_path)
            df = df[df["method"] == "lora"].copy()
            parsed = df["run_name"].apply(parse_run_name)
            df = df[parsed.notna()].copy()
            df["probe"] = parsed.apply(lambda x: x[0])
            df["position"] = parsed.apply(lambda x: x[1])
            df["lambda"] = parsed.apply(lambda x: x[2])
            frames.append(df[["probe", "position", "lambda", "dataset",
                              "dice_mean", "iou_mean", "hd95_mean"]])
    if not frames:
        raise RuntimeError("No CKA result CSVs found under cka/results/.")
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"dice_mean": "dice", "iou_mean": "iou", "hd95_mean": "hd95"})
    return out


def load_baseline() -> dict[str, float]:
    """Baseline LoRA dice per dataset (seed 0, no CKA)."""
    if not BASELINE_CSV.exists():
        # Fall back to known multiseed averages
        return {
            "isic2018_test": 0.9556,
            "ph2":           0.9570,
            "busi":          0.7801,
            "cbis_ddsm":     0.5094,
        }
    df = pd.read_csv(BASELINE_CSV)
    df = df[(df["method"] == "lora") & (df["run_name"] == "lora_seed0")]
    if df.empty:
        return {
            "isic2018_test": 0.9556,
            "ph2":           0.9570,
            "busi":          0.7801,
            "cbis_ddsm":     0.5094,
        }
    return {row["dataset"]: float(row["dice_mean"]) for _, row in df.iterrows()}


def load_zero_shot() -> dict[str, float]:
    """Zero-shot dice per dataset (from any CKA results CSV, same everywhere)."""
    for f in RESULTS_DIR.glob("runs_cka_*.csv"):
        df = pd.read_csv(f)
        zs = df[df["method"] == "zero_shot"]
        if zs.empty:
            continue
        return {row["dataset"]: float(row["dice_mean"]) for _, row in zs.iterrows()}
    return {ds: float("nan") for ds in DATASETS}


# Plot 1: 2x4 grid, probe rows x dataset cols, x=lambda, lines=position
def plot_lambda_grid(df: pd.DataFrame, baseline: dict, zero_shot: dict, out_path: Path) -> None:
    """4 datasets x 2 probes: 8 panels showing dice vs lambda for each position."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True)
    probes = ("original", "oodonly")
    lams = sorted(LAMBDAS.values())

    for row_i, probe in enumerate(probes):
        for col_i, ds in enumerate(DATASETS):
            ax = axes[row_i, col_i]
            sub = df[(df["probe"] == probe) & (df["dataset"] == ds)]
            for position in POSITIONS:
                line = sub[sub["position"] == position].sort_values("lambda")
                if line.empty:
                    continue
                ax.plot(line["lambda"], line["dice"],
                        marker="o", linewidth=2, markersize=9,
                        color=POSITION_COLORS[position], label=position)

            # Baseline LoRA dashed line
            base = baseline.get(ds, float("nan"))
            if not np.isnan(base):
                ax.axhline(base, color="black", linestyle="--", linewidth=1.2,
                           alpha=0.7, label=f"LoRA baseline ({base:.3f})")
            # Zero-shot dotted line
            zs = zero_shot.get(ds, float("nan"))
            if not np.isnan(zs):
                ax.axhline(zs, color="gray", linestyle=":", linewidth=1.2,
                           alpha=0.7, label=f"Zero-shot ({zs:.3f})")

            ax.set_xscale("log")
            ax.set_xticks(lams)
            ax.set_xticklabels([f"{lam:g}" for lam in lams])
            if row_i == 1:
                ax.set_xlabel("lambda (CKA loss weight)")
            if col_i == 0:
                ax.set_ylabel(f"{PROBE_LABELS[probe].split(' (')[0]}\nDice", fontweight="bold")
            if row_i == 0:
                ax.set_title(DATASET_LABELS[ds], fontsize=12)
            ax.grid(alpha=0.3)
            if row_i == 0 and col_i == 0:
                ax.legend(loc="best", fontsize=8, frameon=True)

    fig.suptitle("CKA-aware LoRA: Dice vs lambda for each (probe, position)",
                 fontsize=14, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


# Plot 2: delta-vs-baseline heatmap, 4 datasets x 2 probes
def plot_delta_heatmaps(df: pd.DataFrame, baseline: dict, out_path: Path) -> None:
    """For each (probe, dataset), a 3x3 grid (position x lambda) of delta Dice vs baseline."""
    fig, axes = plt.subplots(4, 2, figsize=(11, 16))
    probes = ("original", "oodonly")
    lams = sorted(LAMBDAS.values())
    DELTA_MAX = 0.50

    for row_i, ds in enumerate(DATASETS):
        for col_i, probe in enumerate(probes):
            ax = axes[row_i, col_i]
            grid = np.full((len(POSITIONS), len(lams)), np.nan)
            for i, position in enumerate(POSITIONS):
                for j, lam in enumerate(lams):
                    sub = df[(df["probe"] == probe) & (df["position"] == position) &
                             (df["lambda"] == lam) & (df["dataset"] == ds)]
                    if not sub.empty:
                        grid[i, j] = float(sub["dice"].iloc[0]) - baseline.get(ds, 0.0)

            im = ax.imshow(grid, cmap="RdBu_r", vmin=-DELTA_MAX, vmax=DELTA_MAX,
                           aspect="auto")
            ax.set_xticks(range(len(lams)))
            ax.set_xticklabels([f"lambda={lam:g}" for lam in lams])
            ax.set_yticks(range(len(POSITIONS)))
            ax.set_yticklabels(POSITIONS)
            if row_i == 0:
                ax.set_title(PROBE_LABELS[probe].split(' (')[0], fontweight="bold")
            if col_i == 0:
                ax.set_ylabel(DATASET_LABELS[ds], fontweight="bold", fontsize=10)
            # Annotate values
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    val = grid[i, j]
                    if np.isnan(val):
                        continue
                    txt_color = "white" if abs(val) > 0.30 else "black"
                    ax.text(j, i, f"{val:+.3f}",
                            ha="center", va="center", color=txt_color, fontsize=10)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Delta Dice vs baseline LoRA")
    fig.suptitle("CKA-aware LoRA: Delta Dice from baseline by (position, lambda, probe)",
                 fontsize=13, y=0.995)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


# Plot 3: best-config bar chart per dataset, per probe
def plot_best_bars(df: pd.DataFrame, baseline: dict, zero_shot: dict, out_path: Path) -> None:
    """For each dataset, show baseline LoRA, zero-shot, best original-probe CKA, best OOD-only CKA."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=False)

    for col_i, ds in enumerate(DATASETS):
        ax = axes[col_i]
        cats = []
        vals = []
        cols = []

        zs = zero_shot.get(ds, float("nan"))
        if not np.isnan(zs):
            cats.append("Zero-shot")
            vals.append(zs)
            cols.append("#808080")

        base = baseline.get(ds, float("nan"))
        if not np.isnan(base):
            cats.append("LoRA\nbaseline")
            vals.append(base)
            cols.append("#404040")

        # Best per probe (max dice on this dataset)
        for probe in ("original", "oodonly"):
            sub = df[(df["probe"] == probe) & (df["dataset"] == ds)]
            if sub.empty:
                continue
            best = sub.loc[sub["dice"].idxmax()]
            label = f"{probe}\n{best['position']}_l{best['lambda']:g}"
            cats.append(label)
            vals.append(float(best["dice"]))
            cols.append(PROBE_COLORS[probe])

        x = np.arange(len(cats))
        bars = ax.bar(x, vals, color=cols, edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=0, fontsize=8)
        ax.set_ylim(0, max(vals) * 1.12 if vals else 1.0)
        ax.set_ylabel("Dice")
        ax.set_title(DATASET_LABELS[ds], fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Best CKA-aware LoRA per probe vs baselines (single seed)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


# Plot 4: ID vs CBIS-DDSM trade-off scatter
def plot_tradeoff(df: pd.DataFrame, baseline: dict, zero_shot: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 9))

    for probe in ("original", "oodonly"):
        sub = df[df["probe"] == probe]
        isic = sub[sub["dataset"] == "isic2018_test"].set_index(["position", "lambda"])
        cbis = sub[sub["dataset"] == "cbis_ddsm"].set_index(["position", "lambda"])
        for position in POSITIONS:
            for lam in sorted(LAMBDAS.values()):
                try:
                    x = float(isic.loc[(position, lam), "dice"])
                    y = float(cbis.loc[(position, lam), "dice"])
                except KeyError:
                    continue
                marker = {"early": "o", "mid": "s", "late": "^"}[position]
                ax.scatter(x, y, marker=marker, s=180,
                           color=PROBE_COLORS[probe],
                           edgecolors="black", linewidth=1.2, alpha=0.9, zorder=5)
                # Label point with lambda
                ax.annotate(f"{position[0]}{lam:g}",
                            (x, y), xytext=(7, -4),
                            textcoords="offset points", fontsize=7)

    # Baselines as star markers
    if "isic2018_test" in baseline and "cbis_ddsm" in baseline:
        ax.scatter([baseline["isic2018_test"]], [baseline["cbis_ddsm"]],
                   marker="*", s=500, color="black", zorder=10,
                   label=f"LoRA baseline ({baseline['isic2018_test']:.3f}, {baseline['cbis_ddsm']:.3f})",
                   edgecolors="white", linewidth=1.5)
    if "isic2018_test" in zero_shot and "cbis_ddsm" in zero_shot:
        ax.scatter([zero_shot["isic2018_test"]], [zero_shot["cbis_ddsm"]],
                   marker="*", s=500, color="gray", zorder=10,
                   label=f"Zero-shot ({zero_shot['isic2018_test']:.3f}, {zero_shot['cbis_ddsm']:.3f})",
                   edgecolors="white", linewidth=1.5)

    # Probe color legend (manual handles)
    probe_handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=13,
                   color=PROBE_COLORS["original"], markeredgecolor="black",
                   label="Original probe"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=13,
                   color=PROBE_COLORS["oodonly"], markeredgecolor="black",
                   label="OOD-only probe"),
    ]
    position_handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=11,
                   color="gray", markeredgecolor="black", label="early"),
        plt.Line2D([], [], marker="s", linestyle="", markersize=11,
                   color="gray", markeredgecolor="black", label="mid"),
        plt.Line2D([], [], marker="^", linestyle="", markersize=11,
                   color="gray", markeredgecolor="black", label="late"),
    ]
    leg1 = ax.legend(handles=probe_handles + position_handles,
                     loc="upper left", fontsize=9, frameon=True)
    ax.add_artist(leg1)
    ax.legend(loc="lower right", fontsize=9, frameon=True)

    ax.set_xlabel("ISIC 2018 Dice (in-domain)", fontsize=12)
    ax.set_ylabel("CBIS-DDSM Dice (far-OOD mammography)", fontsize=12)
    ax.set_title("CKA-aware LoRA: ID vs far-OOD trade-off\n"
                 "Upper-right = better at both", fontsize=12)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


# Plot 5: side-by-side dice heatmap (4 rows x 2 cols)
def plot_dice_heatmaps(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(11, 16))
    lams = sorted(LAMBDAS.values())

    # Per-dataset min/max for better color contrast within each dataset
    for row_i, ds in enumerate(DATASETS):
        sub_ds = df[df["dataset"] == ds]
        if sub_ds.empty:
            continue
        v_min = sub_ds["dice"].min()
        v_max = sub_ds["dice"].max()

        for col_i, probe in enumerate(("original", "oodonly")):
            ax = axes[row_i, col_i]
            grid = np.full((len(POSITIONS), len(lams)), np.nan)
            for i, position in enumerate(POSITIONS):
                for j, lam in enumerate(lams):
                    sel = df[(df["probe"] == probe) & (df["position"] == position) &
                             (df["lambda"] == lam) & (df["dataset"] == ds)]
                    if not sel.empty:
                        grid[i, j] = float(sel["dice"].iloc[0])

            im = ax.imshow(grid, cmap="RdYlGn", vmin=v_min, vmax=v_max, aspect="auto")
            ax.set_xticks(range(len(lams)))
            ax.set_xticklabels([f"lambda={lam:g}" for lam in lams])
            ax.set_yticks(range(len(POSITIONS)))
            ax.set_yticklabels(POSITIONS)
            if row_i == 0:
                ax.set_title(PROBE_LABELS[probe].split(' (')[0], fontweight="bold")
            if col_i == 0:
                ax.set_ylabel(DATASET_LABELS[ds], fontweight="bold", fontsize=10)
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    val = grid[i, j]
                    if np.isnan(val):
                        continue
                    ax.text(j, i, f"{val:.3f}",
                            ha="center", va="center",
                            color="black", fontsize=10)
            # Per-dataset colorbar
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.suptitle("CKA-aware LoRA Dice heatmap by (position, lambda, probe)\n"
                 "Color range scaled per dataset",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


# Plot 6: robustness rank chart, count wins / catastrophes per probe
def plot_robustness(df: pd.DataFrame, baseline: dict, out_path: Path) -> None:
    """Counts per probe of: configs that beat baseline, that lose >0.1, that lose >0.3."""
    cats = ["beat_baseline", "loss<0.1", "loss>0.1", "loss>0.3"]
    cat_labels = ["Beats LoRA\nbaseline", "Mild loss\n(0-0.1)", "Moderate loss\n(0.1-0.3)", "Catastrophic\n(>0.3)"]
    counts = {probe: {c: 0 for c in cats} for probe in ("original", "oodonly")}

    for _, row in df.iterrows():
        probe = row["probe"]
        delta = float(row["dice"]) - baseline.get(row["dataset"], 0.0)
        if delta >= 0:
            counts[probe]["beat_baseline"] += 1
        elif delta >= -0.1:
            counts[probe]["loss<0.1"] += 1
        elif delta >= -0.3:
            counts[probe]["loss>0.1"] += 1
        else:
            counts[probe]["loss>0.3"] += 1

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(cats))
    width = 0.38
    bars1 = ax.bar(x - width/2, [counts["original"][c] for c in cats],
                   width, color=PROBE_COLORS["original"], edgecolor="black",
                   label="Original probe")
    bars2 = ax.bar(x + width/2, [counts["oodonly"][c] for c in cats],
                   width, color=PROBE_COLORS["oodonly"], edgecolor="black",
                   label="OOD-only probe")
    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.1, f"{int(h)}",
                    ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=9)
    ax.set_ylabel("# (config x dataset) cells out of 36 (9 configs x 4 datasets)")
    ax.set_title("Robustness comparison: how many cells fall in each Delta Dice bucket?",
                 fontsize=11)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


def write_summary(df: pd.DataFrame, baseline: dict, zero_shot: dict,
                  csv_path: Path, md_path: Path) -> None:
    rows = []
    for _, r in df.iterrows():
        base = baseline.get(r["dataset"], float("nan"))
        rows.append({
            "probe":    r["probe"],
            "position": r["position"],
            "lambda":   float(r["lambda"]),
            "dataset":  r["dataset"],
            "dice":     round(float(r["dice"]), 4),
            "iou":      round(float(r["iou"]), 4),
            "hd95":     round(float(r["hd95"]), 2),
            "baseline_dice": round(base, 4) if not np.isnan(base) else "",
            "delta_vs_baseline": round(float(r["dice"]) - base, 4) if not np.isnan(base) else "",
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(csv_path, index=False)
    print(f"[compare] wrote {csv_path}")

    # Markdown: best per (probe, dataset)
    lines = ["# CKA probe comparison: best (position, lambda) per dataset per probe\n"]
    lines.append("Baseline LoRA dice (seed 0, no CKA): "
                 + ", ".join(f"{ds}={baseline.get(ds, float('nan')):.4f}" for ds in DATASETS) + "\n")
    lines.append("Zero-shot dice: "
                 + ", ".join(f"{ds}={zero_shot.get(ds, float('nan')):.4f}" for ds in DATASETS) + "\n")
    lines.append("\n## Best CKA config per dataset per probe\n")
    lines.append("| Dataset | Probe | Best position | Best lambda | Dice | Delta vs baseline |")
    lines.append("|---|---|---|---:|---:|---:|")
    for ds in DATASETS:
        for probe in ("original", "oodonly"):
            sub = out_df[(out_df["probe"] == probe) & (out_df["dataset"] == ds)]
            if sub.empty:
                continue
            best = sub.loc[sub["dice"].idxmax()]
            delta = best["delta_vs_baseline"] if best["delta_vs_baseline"] != "" else float("nan")
            delta_str = f"{delta:+.4f}" if not (isinstance(delta, float) and np.isnan(delta)) else "-"
            lines.append(f"| {DATASET_LABELS[ds]} | {probe} | {best['position']} | "
                         f"{best['lambda']:g} | {best['dice']:.4f} | {delta_str} |")

    lines.append("\n## Robustness: how many of 36 (config x dataset) cells fall in each bucket?\n")
    lines.append("| Bucket | Original probe | OOD-only probe |")
    lines.append("|---|---:|---:|")
    counts = {p: {"beat": 0, "mild": 0, "mod": 0, "cat": 0} for p in ("original", "oodonly")}
    for _, r in out_df.iterrows():
        if r["delta_vs_baseline"] == "":
            continue
        d = float(r["delta_vs_baseline"])
        p = r["probe"]
        if d >= 0: counts[p]["beat"] += 1
        elif d >= -0.1: counts[p]["mild"] += 1
        elif d >= -0.3: counts[p]["mod"] += 1
        else: counts[p]["cat"] += 1
    for label, key in (("Beats baseline (delta >= 0)", "beat"),
                        ("Mild loss (0 > delta >= -0.1)", "mild"),
                        ("Moderate loss (-0.1 > delta >= -0.3)", "mod"),
                        ("Catastrophic (delta < -0.3)", "cat")):
        lines.append(f"| {label} | {counts['original'][key]} | {counts['oodonly'][key]} |")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[compare] wrote {md_path}")


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all_results()
    baseline = load_baseline()
    zero_shot = load_zero_shot()
    print(f"[compare] loaded {len(df)} rows from "
          f"{df['probe'].nunique()} probes x {df['position'].nunique()} positions x "
          f"{df['lambda'].nunique()} lambdas x {df['dataset'].nunique()} datasets")
    print(f"[compare] baseline: {baseline}")
    print(f"[compare] zero-shot: {zero_shot}")

    plot_lambda_grid(df, baseline, zero_shot, FIGURES_DIR / "cka_probe_compare_grid.png")
    plot_dice_heatmaps(df, FIGURES_DIR / "cka_probe_compare_heatmap.png")
    plot_delta_heatmaps(df, baseline, FIGURES_DIR / "cka_probe_compare_delta_heatmap.png")
    plot_best_bars(df, baseline, zero_shot, FIGURES_DIR / "cka_probe_compare_best_bars.png")
    plot_tradeoff(df, baseline, zero_shot, FIGURES_DIR / "cka_probe_compare_tradeoff.png")
    plot_robustness(df, baseline, FIGURES_DIR / "cka_probe_compare_robustness.png")
    write_summary(df, baseline, zero_shot,
                  RESULTS_DIR / "cka_probe_compare_summary.csv",
                  RESULTS_DIR / "cka_probe_compare_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
