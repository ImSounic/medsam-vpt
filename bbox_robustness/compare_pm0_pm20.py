"""Compare bbox-robustness curves between pm=0 trained and pm=20 trained models."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PM0_BBOX_CSV   = REPO_ROOT / "bbox_robustness" / "results" / "runs.csv"
PM20_BBOX_CSV  = REPO_ROOT / "bbox_robustness" / "results_pm20" / "runs.csv"
PM0_TIGHT_CSV  = REPO_ROOT / "results" / "runs.csv"
PM20_TIGHT_CSV = REPO_ROOT / "results" / "runs_pm20.csv"

OUT_FIG_DIR = REPO_ROOT / "bbox_robustness" / "results_pm20" / "figures"
OUT_SUMMARY = REPO_ROOT / "bbox_robustness" / "results_pm20" / "comparison_summary.csv"

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
    "busi":          "BUSI (far-OOD ultrasound)",
    "cbis_ddsm":     "CBIS-DDSM (far-OOD mammography)",
}
DATASET_ORDER = ["isic2018_test", "ph2", "busi", "cbis_ddsm"]
PERTURBS = [0, 20, 50, 100, 200]


def load_combined() -> pd.DataFrame:
    """Return combined long-form DataFrame across pm=0 and pm=20 trainings."""
    frames = []

    bbox0 = pd.read_csv(PM0_BBOX_CSV)
    bbox0["training"] = "pm=0"
    frames.append(bbox0[["method", "dataset", "perturb_max_px", "dice_mean", "training"]])

    bbox20 = pd.read_csv(PM20_BBOX_CSV)
    bbox20["training"] = "pm=20"
    frames.append(bbox20[["method", "dataset", "perturb_max_px", "dice_mean", "training"]])

    # Tight baselines at perturb=0
    tight0 = pd.read_csv(PM0_TIGHT_CSV)
    tight0["perturb_max_px"] = 0
    tight0["training"] = "pm=0"
    frames.append(tight0[["method", "dataset", "perturb_max_px", "dice_mean", "training"]])

    tight20 = pd.read_csv(PM20_TIGHT_CSV)
    tight20["perturb_max_px"] = 0
    tight20["training"] = "pm=20"
    frames.append(tight20[["method", "dataset", "perturb_max_px", "dice_mean", "training"]])

    df = pd.concat(frames, ignore_index=True)

    # zero_shot appears in both the pm20 bbox CSV and the tight baseline with identical values; drop the dupes.
    df = df.drop_duplicates(
        subset=["method", "dataset", "perturb_max_px", "training"],
        keep="first",
    )
    return df


def plot_overlay_curves(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes_flat = axes.flatten()

    for ax, ds in zip(axes_flat, DATASET_ORDER):
        sub = df[df["dataset"] == ds]
        for m in METHOD_ORDER:
            if m == "zero_shot":
                # zero_shot is the same model in both trainings; plot once
                sub_m = sub[(sub["method"] == m) & (sub["training"] == "pm=0")].sort_values("perturb_max_px")
                if not sub_m.empty:
                    ax.plot(sub_m["perturb_max_px"], sub_m["dice_mean"],
                             marker="o", linewidth=2, color=METHOD_COLORS[m],
                             label=METHOD_LABELS[m], zorder=1)
                continue

            # pm=0 trained: solid line
            sub_0 = sub[(sub["method"] == m) & (sub["training"] == "pm=0")].sort_values("perturb_max_px")
            if not sub_0.empty:
                ax.plot(sub_0["perturb_max_px"], sub_0["dice_mean"],
                         marker="o", linewidth=2, color=METHOD_COLORS[m],
                         label=f"{METHOD_LABELS[m]} (pm=0 trained)", zorder=2)

            # pm=20 trained: dashed line, same colour
            sub_20 = sub[(sub["method"] == m) & (sub["training"] == "pm=20")].sort_values("perturb_max_px")
            if not sub_20.empty:
                ax.plot(sub_20["perturb_max_px"], sub_20["dice_mean"],
                         marker="s", linewidth=2, linestyle="--",
                         color=METHOD_COLORS[m], alpha=0.85,
                         label=f"{METHOD_LABELS[m]} (pm=20 trained)", zorder=3)

        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.set_xlabel("Eval-time bbox max expansion (px)")
        ax.set_ylabel("Dice")
        ax.grid(alpha=0.3)
        ax.set_xticks(PERTURBS)

    # Shared legend: methods only; training shows up in the linestyle
    method_handles = [
        plt.Line2D([], [], color=METHOD_COLORS[m], marker="o", linewidth=2,
                    label=METHOD_LABELS[m])
        for m in METHOD_ORDER
    ]
    style_handles = [
        plt.Line2D([], [], color="black", marker="o", linewidth=2, linestyle="-",
                    label="trained at pm=0 (tight bbox)"),
        plt.Line2D([], [], color="black", marker="s", linewidth=2, linestyle="--",
                    label="trained at pm=20 (jittered bbox)"),
    ]
    fig.legend(handles=method_handles + style_handles,
                loc="lower center", ncol=4, frameon=False,
                bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "Bbox robustness: pm=0 trained vs pm=20 trained models  "
        "(solid = pm=0 train, dashed = pm=20 train)",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


def plot_delta_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """Heatmap of Dice(pm=20 trained) minus Dice(pm=0 trained), one panel per dataset."""
    wide = df.pivot_table(
        index=["method", "perturb_max_px"],
        columns="training",
        values="dice_mean",
        aggfunc="first",
    ).reset_index()
    wide["delta"] = wide["pm=20"] - wide["pm=0"]

    # zero_shot is the same model in both, so its delta is meaningless
    wide = wide[wide["method"] != "zero_shot"]

    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(20, 5), sharey=True)

    delta_min = -0.25
    delta_max = 0.25

    for ax, ds in zip(axes, DATASET_ORDER):
        sub_df = df[df["dataset"] == ds]
        pivot = sub_df.pivot_table(
            index="method", columns=["training", "perturb_max_px"],
            values="dice_mean", aggfunc="first",
        )
        # delta = pm=20 minus pm=0
        delta_grid = pd.DataFrame(index=[m for m in METHOD_ORDER if m != "zero_shot"],
                                   columns=PERTURBS, dtype=float)
        for m in delta_grid.index:
            for p in PERTURBS:
                try:
                    v20 = pivot.loc[m, ("pm=20", p)]
                    v0  = pivot.loc[m, ("pm=0",  p)]
                    delta_grid.loc[m, p] = v20 - v0
                except KeyError:
                    delta_grid.loc[m, p] = np.nan

        im = ax.imshow(delta_grid.values.astype(float),
                        cmap="RdBu_r", vmin=delta_min, vmax=delta_max,
                        aspect="auto")
        ax.set_xticks(range(len(PERTURBS)))
        ax.set_xticklabels([str(p) for p in PERTURBS])
        ax.set_yticks(range(len(delta_grid.index)))
        ax.set_yticklabels([METHOD_LABELS[m] for m in delta_grid.index])
        ax.set_xlabel("Eval bbox max expansion (px)")
        ax.set_title(DATASET_LABELS.get(ds, ds))
        for i in range(delta_grid.shape[0]):
            for j in range(delta_grid.shape[1]):
                val = float(delta_grid.values[i, j])
                if np.isnan(val):
                    continue
                ax.text(j, i, f"{val:+.3f}",
                         ha="center", va="center",
                         color="white" if abs(val) > 0.15 else "black",
                         fontsize=8)
        if ax is axes[-1]:
            cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cbar.set_label("delta Dice (pm=20 trained - pm=0 trained)\nblue = pm=20 worse, red = pm=20 better")

    fig.suptitle(
        "Effect of pm=20 jitter training on bbox-robustness Dice  "
        "(red = jitter training helped, blue = jitter training hurt)",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


def write_summary_csv(df: pd.DataFrame, out_path: Path) -> None:
    """Long-form summary: method, dataset, perturb_max_px, dice_pm0, dice_pm20, delta."""
    wide = df.pivot_table(
        index=["method", "dataset", "perturb_max_px"],
        columns="training",
        values="dice_mean",
        aggfunc="first",
    ).reset_index()
    if "pm=0" in wide.columns:
        wide = wide.rename(columns={"pm=0": "dice_pm0", "pm=20": "dice_pm20"})
        wide["delta"] = wide["dice_pm20"] - wide["dice_pm0"]
    method_idx = {m: i for i, m in enumerate(METHOD_ORDER)}
    dataset_idx = {d: i for i, d in enumerate(DATASET_ORDER)}
    wide["_m"] = wide["method"].map(method_idx)
    wide["_d"] = wide["dataset"].map(dataset_idx)
    wide = wide.sort_values(["_m", "_d", "perturb_max_px"]).drop(columns=["_m", "_d"])
    for c in ("dice_pm0", "dice_pm20", "delta"):
        if c in wide.columns:
            wide[c] = wide[c].astype(float).round(4)
    wide.to_csv(out_path, index=False)
    print(f"[compare] wrote {out_path}")


def main() -> int:
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_combined()
    print(f"[compare] loaded {len(df)} rows "
          f"({df['method'].nunique()} methods x {df['dataset'].nunique()} datasets x "
          f"{df['perturb_max_px'].nunique()} perturb levels x {df['training'].nunique()} trainings)")

    plot_overlay_curves(df, OUT_FIG_DIR / "comparison_curves.png")
    plot_delta_heatmap(df, OUT_FIG_DIR / "delta_heatmap.png")
    write_summary_csv(df, OUT_SUMMARY)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
