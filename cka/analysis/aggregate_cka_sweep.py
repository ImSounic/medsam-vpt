"""Aggregate the 9 CKA-sweep eval CSVs into one comparison table + plots.

Reads runs_cka_{early,mid,late}.csv plus results/runs.csv (LoRA pm=0 baseline).
Each CSV has 3 trainings x 4 datasets; we keep LoRA rows, parse (position, lambda)
from run_name, and join with the baseline.

Writes to cka/results/: cka_sweep_summary.csv (long-form) and .md (grouped by dataset).
Writes to cka/figures/: cka_sweep_grid.png (dice vs lambda per dataset, line per
position, baseline dashed) and cka_id_vs_far_ood_tradeoff.png (ISIC vs CBIS scatter).
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CKA_RESULTS = REPO_ROOT / "cka" / "results"
CKA_FIGS = REPO_ROOT / "cka" / "figures"
BASELINE_CSV = REPO_ROOT / "results" / "runs.csv"

POSITIONS = ("early", "mid", "late")
LAMBDAS = {"01": 0.1, "1": 1.0, "10": 10.0}
DATASET_ORDER = ["isic2018_test", "ph2", "busi", "cbis_ddsm"]
DATASET_LABELS = {
    "isic2018_test": "ISIC 2018 (ID)",
    "ph2":           "PH2 (near-OOD)",
    "busi":          "BUSI (far-OOD ultrasound)",
    "cbis_ddsm":     "CBIS-DDSM (far-OOD mammography)",
}
POSITION_COLORS = {"early": "#1f77b4", "mid": "#ff7f0e", "late": "#d62728"}


def parse_cka_run_name(run_name: str) -> tuple[str, float] | None:
    """lora_cka_<position>_l<lambda_str>_seed0 to (position, lambda)."""
    m = re.match(r"^lora_cka_(early|mid|late)_l([0-9]+)_seed\d+$", run_name)
    if not m:
        return None
    position, lambda_str = m.group(1), m.group(2)
    if lambda_str not in LAMBDAS:
        return None
    return position, LAMBDAS[lambda_str]


def load_cka_runs() -> pd.DataFrame:
    """DataFrame: position, lambda, dataset, dice_mean, iou_mean, hd95_mean."""
    frames = []
    for position in POSITIONS:
        csv_path = CKA_RESULTS / f"runs_cka_{position}.csv"
        if not csv_path.exists():
            print(f"[agg-cka] WARNING: {csv_path} not found, skipping {position}")
            continue
        df = pd.read_csv(csv_path)
        # Keep only LoRA rows (sweep is LoRA-only)
        df = df[df["method"] == "lora"].copy()
        # Parse (position, lambda) from run_name
        parsed = df["run_name"].apply(parse_cka_run_name)
        df = df[parsed.notna()].copy()
        df["position"] = parsed.apply(lambda x: x[0])
        df["lambda"] = parsed.apply(lambda x: x[1])
        frames.append(df[["position", "lambda", "dataset",
                          "dice_mean", "iou_mean", "hd95_mean"]])
    if not frames:
        raise RuntimeError("No CKA runs found in cka/results/. Run the sweep first.")
    return pd.concat(frames, ignore_index=True)


def load_lora_baseline() -> pd.DataFrame:
    """Baseline LoRA (pm=0, seed 0) dice per dataset, as a small DataFrame."""
    if not BASELINE_CSV.exists():
        print(f"[agg-cka] WARNING: {BASELINE_CSV} not found, baseline unavailable")
        return pd.DataFrame()
    df = pd.read_csv(BASELINE_CSV)
    df = df[(df["method"] == "lora") & (df["run_name"] == "lora_seed0")]
    if df.empty:
        return pd.DataFrame()
    return df[["dataset", "dice_mean", "iou_mean", "hd95_mean"]].copy()


def write_summary_csv(cka_df: pd.DataFrame, baseline_df: pd.DataFrame,
                      out_path: Path) -> None:
    """Long-form CSV: one row per (position, lambda, dataset) plus baseline rows."""
    base_lookup = {}
    if not baseline_df.empty:
        for _, r in baseline_df.iterrows():
            base_lookup[r["dataset"]] = float(r["dice_mean"])

    rows = []
    for _, r in cka_df.iterrows():
        baseline = base_lookup.get(r["dataset"], float("nan"))
        delta = float(r["dice_mean"]) - baseline if not np.isnan(baseline) else float("nan")
        rows.append({
            "position":  r["position"],
            "lambda":    r["lambda"],
            "dataset":   r["dataset"],
            "dice":      round(float(r["dice_mean"]), 4),
            "iou":       round(float(r["iou_mean"]), 4),
            "hd95":      round(float(r["hd95_mean"]), 2),
            "baseline_dice": round(baseline, 4) if not np.isnan(baseline) else "",
            "delta_dice":    round(delta, 4) if not np.isnan(delta) else "",
        })
    out = pd.DataFrame(rows)
    out["dataset"] = pd.Categorical(out["dataset"], categories=DATASET_ORDER, ordered=True)
    out["position"] = pd.Categorical(out["position"], categories=POSITIONS, ordered=True)
    out = out.sort_values(["dataset", "position", "lambda"]).reset_index(drop=True)
    out.to_csv(out_path, index=False)
    print(f"[agg-cka] wrote {out_path}")


def write_summary_md(cka_df: pd.DataFrame, baseline_df: pd.DataFrame,
                     out_path: Path) -> None:
    """Per-dataset markdown table grouped by (position, lambda)."""
    base_lookup = {}
    if not baseline_df.empty:
        for _, r in baseline_df.iterrows():
            base_lookup[r["dataset"]] = float(r["dice_mean"])

    lines = ["# CKA-aware LoRA sweep - Dice results\n",
             "Single seed (seed=0), 6 epochs each. Position = which decoder layers "
             "are CKA-regularised. lambda = strength of the CKA loss term.\n",
             "Delta vs baseline = CKA-aware Dice minus baseline LoRA (pm=0, seed 0).\n"]

    for ds in DATASET_ORDER:
        sub = cka_df[cka_df["dataset"] == ds]
        if sub.empty:
            continue
        lines.append(f"\n## {DATASET_LABELS.get(ds, ds)}\n")
        baseline = base_lookup.get(ds, float("nan"))
        if not np.isnan(baseline):
            lines.append(f"Baseline LoRA dice = **{baseline:.4f}**\n")
        lines.append("| Position | lambda | Dice | Delta vs baseline | HD95 (px) | IoU |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for position in POSITIONS:
            for lam_val in sorted(LAMBDAS.values()):
                row = sub[(sub["position"] == position) & (sub["lambda"] == lam_val)]
                if row.empty:
                    continue
                r = row.iloc[0]
                dice = float(r["dice_mean"])
                delta = dice - baseline if not np.isnan(baseline) else float("nan")
                delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "-"
                lines.append(
                    f"| {position} | {lam_val:g} | {dice:.4f} | {delta_str} | "
                    f"{float(r['hd95_mean']):.2f} | {float(r['iou_mean']):.4f} |"
                )
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[agg-cka] wrote {out_path}")


def plot_sweep_grid(cka_df: pd.DataFrame, baseline_df: pd.DataFrame,
                    out_path: Path) -> None:
    """4 subplots (one per dataset). x = lambda (log), y = Dice. 3 lines (positions)."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True)
    axes_flat = axes.flatten()

    base_lookup = {}
    if not baseline_df.empty:
        for _, r in baseline_df.iterrows():
            base_lookup[r["dataset"]] = float(r["dice_mean"])

    for ax, ds in zip(axes_flat, DATASET_ORDER):
        sub = cka_df[cka_df["dataset"] == ds]
        for position in POSITIONS:
            line = sub[sub["position"] == position].sort_values("lambda")
            if line.empty:
                continue
            ax.plot(line["lambda"], line["dice_mean"],
                    marker="o", linewidth=2, color=POSITION_COLORS[position],
                    label=f"{position}", markersize=8)
        baseline = base_lookup.get(ds, float("nan"))
        if not np.isnan(baseline):
            ax.axhline(baseline, color="black", linestyle="--", linewidth=1.2,
                       alpha=0.6, label=f"LoRA baseline ({baseline:.3f})")
        ax.set_xscale("log")
        ax.set_xlabel("lambda (CKA loss weight)")
        ax.set_ylabel("Dice")
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    fig.suptitle("LoRA + CKA regularisation: Dice vs lambda for each decoder-hook position",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[agg-cka] wrote {out_path}")


def plot_tradeoff(cka_df: pd.DataFrame, baseline_df: pd.DataFrame,
                  out_path: Path) -> None:
    """Scatter: x = ISIC dice (in-domain), y = CBIS dice (far-OOD).
    One point per (position, lambda), plus the baseline LoRA point."""
    fig, ax = plt.subplots(figsize=(9, 8))

    isic = cka_df[cka_df["dataset"] == "isic2018_test"].set_index(["position", "lambda"])
    cbis = cka_df[cka_df["dataset"] == "cbis_ddsm"].set_index(["position", "lambda"])

    for position in POSITIONS:
        xs, ys, labels = [], [], []
        for lam in sorted(LAMBDAS.values()):
            try:
                x = float(isic.loc[(position, lam), "dice_mean"])
                y = float(cbis.loc[(position, lam), "dice_mean"])
            except KeyError:
                continue
            xs.append(x); ys.append(y); labels.append(f"lambda={lam:g}")
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", linewidth=1.4, markersize=12,
                color=POSITION_COLORS[position], label=position,
                markeredgecolor="black", alpha=0.85)
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab, (x, y), xytext=(8, -4),
                        textcoords="offset points", fontsize=8, color="#333")

    if not baseline_df.empty:
        b_isic = baseline_df[baseline_df["dataset"] == "isic2018_test"]
        b_cbis = baseline_df[baseline_df["dataset"] == "cbis_ddsm"]
        if not b_isic.empty and not b_cbis.empty:
            bx = float(b_isic["dice_mean"].iloc[0])
            by = float(b_cbis["dice_mean"].iloc[0])
            ax.scatter([bx], [by], marker="*", s=400, color="black",
                       zorder=10, label=f"LoRA baseline ({bx:.3f}, {by:.3f})",
                       edgecolors="white", linewidth=1.5)

    ax.set_xlabel("ISIC Dice (in-domain)", fontsize=12)
    ax.set_ylabel("CBIS-DDSM Dice (far-OOD)", fontsize=12)
    ax.set_title("CKA-aware LoRA: ID vs far-OOD trade-off\n"
                 "Upper-right corner = better at both", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[agg-cka] wrote {out_path}")


def main() -> int:
    CKA_FIGS.mkdir(parents=True, exist_ok=True)
    CKA_RESULTS.mkdir(parents=True, exist_ok=True)

    cka_df = load_cka_runs()
    baseline_df = load_lora_baseline()
    print(f"[agg-cka] loaded {len(cka_df)} CKA rows, "
          f"{len(baseline_df)} baseline rows")

    write_summary_csv(cka_df, baseline_df, CKA_RESULTS / "cka_sweep_summary.csv")
    write_summary_md(cka_df, baseline_df, CKA_RESULTS / "cka_sweep_summary.md")
    plot_sweep_grid(cka_df, baseline_df, CKA_FIGS / "cka_sweep_grid.png")
    plot_tradeoff(cka_df, baseline_df, CKA_FIGS / "cka_id_vs_far_ood_tradeoff.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
