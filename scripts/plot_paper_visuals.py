"""Generate the three kept paper visuals from summary_full_multiseed.csv."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = REPO_ROOT / "summary_full_multiseed.csv"
OUT_DIR = REPO_ROOT / "figures" / "paper_visuals"

METHOD_ORDER = [
    "zero_shot",
    "decoder_only",
    "vpt_shallow",
    "vpt_deep",
    "lora",
    "full_ft",
]
METHOD_LABELS = {
    "zero_shot": "Zero-shot",
    "decoder_only": "Decoder-only",
    "vpt_shallow": "VPT-shallow",
    "vpt_deep": "VPT-deep",
    "lora": "LoRA",
    "full_ft": "Full FT",
}
METHOD_COLORS = {
    "zero_shot": "#6c757d",
    "decoder_only": "#1f77b4",
    "vpt_shallow": "#ff7f0e",
    "vpt_deep": "#d62728",
    "lora": "#2ca02c",
    "full_ft": "#9467bd",
}
TRAINING_ORDER = ["pm=0", "pm=20", "rand100"]
TRAINING_LABELS = {
    "pm=0": "pm=0",
    "pm=20": "pm=20",
    "rand100": "rand100",
}
TRAINING_STYLES = {
    "pm=0": {"linestyle": "-", "marker": "o"},
    "pm=20": {"linestyle": "--", "marker": "s"},
    "rand100": {"linestyle": ":", "marker": "^"},
}
TRAINING_COLORS = {
    "pm=0": "#264653",
    "pm=20": "#2a9d8f",
    "rand100": "#e76f51",
}
DATASET_ORDER = ["isic2018_test", "ph2", "busi", "cbis_ddsm"]
DATASET_LABELS = {
    "isic2018_test": "ISIC",
    "ph2": "PH2",
    "busi": "BUSI",
    "cbis_ddsm": "CBIS-DDSM",
}
PERTURB_ORDER = [0, 20, 50, 100, 200]


def load_summary() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    df = df.rename(
        columns={
            "dice_mean_seeds": "dice_mean",
            "dice_std_seeds": "dice_std",
        }
    )
    df["perturb_max_px"] = df["perturb_max_px"].astype(int)
    df = df[df["method"].isin(METHOD_ORDER)].copy()
    return df


def plot_degradation_curves(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, dataset in zip(axes, DATASET_ORDER):
        subset = df[df["dataset"] == dataset]
        for method in METHOD_ORDER:
            for training in TRAINING_ORDER:
                if method == "zero_shot" and training != "pm=0":
                    continue
                series = subset[
                    (subset["method"] == method) & (subset["training"] == training)
                ].sort_values("perturb_max_px")
                if series.empty:
                    continue
                ax.plot(
                    series["perturb_max_px"],
                    series["dice_mean"],
                    color=METHOD_COLORS[method],
                    linestyle=TRAINING_STYLES[training]["linestyle"],
                    marker=TRAINING_STYLES[training]["marker"],
                    linewidth=2.0 if training != "rand100" else 2.4,
                    markersize=4.5,
                    alpha=0.95 if training == "pm=0" else 0.9,
                )
                ax.fill_between(
                    series["perturb_max_px"],
                    series["dice_mean"] - series["dice_std"],
                    series["dice_mean"] + series["dice_std"],
                    color=METHOD_COLORS[method],
                    alpha=0.08,
                )
        ax.set_title(DATASET_LABELS[dataset], fontsize=12, fontweight="bold")
        ax.set_xlabel("Eval bbox perturbation (px)")
        ax.set_ylabel("Dice")
        ax.set_xticks(PERTURB_ORDER)
        ax.grid(alpha=0.25)

    method_handles = [
        plt.Line2D(
            [],
            [],
            color=METHOD_COLORS[method],
            linewidth=2,
            label=METHOD_LABELS[method],
        )
        for method in METHOD_ORDER
    ]
    style_handles = [
        plt.Line2D(
            [],
            [],
            color="black",
            linestyle=TRAINING_STYLES[training]["linestyle"],
            marker=TRAINING_STYLES[training]["marker"],
            linewidth=2,
            label=TRAINING_LABELS[training],
        )
        for training in TRAINING_ORDER
    ]
    fig.legend(
        handles=method_handles + style_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle("Prompt perturbation degradation curves", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "degradation_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_degradation_heatmap(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        len(TRAINING_ORDER),
        len(DATASET_ORDER),
        figsize=(18, 10),
        sharex=True,
        sharey=True,
    )
    image = None

    for row, training in enumerate(TRAINING_ORDER):
        for col, dataset in enumerate(DATASET_ORDER):
            ax = axes[row, col]
            subset = df[
                (df["training"] == training) & (df["dataset"] == dataset)
            ].pivot_table(
                index="method",
                columns="perturb_max_px",
                values="dice_mean",
                aggfunc="first",
            )
            subset = subset.reindex(METHOD_ORDER)
            subset = subset[PERTURB_ORDER]
            image = ax.imshow(
                subset.values,
                cmap="RdYlGn",
                vmin=0.3,
                vmax=1.0,
                aspect="auto",
            )
            ax.set_xticks(range(len(PERTURB_ORDER)))
            ax.set_xticklabels(PERTURB_ORDER)
            ax.set_yticks(range(len(METHOD_ORDER)))
            ax.set_yticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], fontsize=9)
            if row == 0:
                ax.set_title(DATASET_LABELS[dataset], fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(training, fontsize=11, fontweight="bold")
            for i in range(subset.shape[0]):
                for j in range(subset.shape[1]):
                    value = subset.iloc[i, j]
                    if pd.isna(value):
                        continue
                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color="black" if value > 0.62 else "white",
                        fontsize=7,
                    )

    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.018, pad=0.02)
    fig.suptitle("Degradation heatmaps", fontsize=14, y=0.98)
    fig.subplots_adjust(top=0.92, wspace=0.18, hspace=0.18)
    fig.savefig(OUT_DIR / "degradation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_performance_histograms(df: pd.DataFrame) -> None:
    tight = df[df["perturb_max_px"] == 0].copy()
    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(20, 5), sharey=True)
    bar_width = 0.24
    x = np.arange(len(METHOD_ORDER))

    for ax, dataset in zip(axes, DATASET_ORDER):
        subset = tight[tight["dataset"] == dataset]
        for offset, training in enumerate(TRAINING_ORDER):
            values = []
            for method in METHOD_ORDER:
                row = subset[
                    (subset["method"] == method) & (subset["training"] == training)
                ]
                if method == "zero_shot":
                    row = subset[
                        (subset["method"] == method) & (subset["training"] == "pm=0")
                    ]
                values.append(np.nan if row.empty else float(row["dice_mean"].iloc[0]))
            ax.bar(
                x + (offset - 1) * bar_width,
                values,
                width=bar_width,
                color=TRAINING_COLORS[training],
                label=TRAINING_LABELS[training],
                edgecolor="black",
                linewidth=0.5,
            )
        ax.set_title(DATASET_LABELS[dataset], fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=25)
        ax.set_ylabel("Dice at tight bbox")
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    fig.suptitle("Performance histograms at perturbation 0", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "performance_histograms.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_summary()
    plot_degradation_curves(df)
    plot_degradation_heatmap(df)
    plot_performance_histograms(df)
    print(f"Wrote paper visuals to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
