"""Multi-seed CKA vs no-CKA LoRA (pm=20): seed table, paired Wilcoxon on bbox-sweep per-image Dice, six-curve robustness figure."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASETS = ["isic2018_test", "ph2", "busi", "cbis_ddsm"]
LEVELS = (20, 50, 100, 200)
_CKA_RE = re.compile(r"^lora_cka_oodonly_late_l10_pm20_seed(\d)$")
_NO_RE = re.compile(r"^lora_seed(\d)_pm20$")
_PER_IMAGE_RE = re.compile(
    r"^(?P<run>.+)_(?P<ds>isic2018_test|ph2|busi|cbis_ddsm)_pm(?P<level>\d+)\.csv$"
)

# Default inputs (relative to the repo root).
TIGHT_PM20 = [
    "cka/results/runs_cka_oodonly_late_l10_pm20.csv",
    "cka/results/runs_accv_t1.csv",
    "results/runs_pm20.csv",
    "results/runs_seed1_pm20.csv",
    "results/runs_seed2_pm20.csv",
]
SWEEP_PM20 = [
    "bbox_robustness/results_cka_oodonly_late_l10_pm20",
    "bbox_robustness/results_accv_t1",
    "bbox_robustness/results_pm20",
    "bbox_robustness/results_seed1_pm20",
    "bbox_robustness/results_seed2_pm20",
]

# Six robustness curves: label -> (run-name regex with the seed as group 1, tight CSVs, sweep dirs)
CURVES = {
    "no-CKA pm0": (
        r"^lora_seed(\d)$",
        ["results/runs.csv", "results/runs_seed1.csv", "results/runs_seed2.csv"],
        [
            "bbox_robustness/results",
            "bbox_robustness/results_seed1",
            "bbox_robustness/results_seed2",
        ],
    ),
    "no-CKA pm20": (
        r"^lora_seed(\d)_pm20$",
        [
            "results/runs_pm20.csv",
            "results/runs_seed1_pm20.csv",
            "results/runs_seed2_pm20.csv",
        ],
        [
            "bbox_robustness/results_pm20",
            "bbox_robustness/results_seed1_pm20",
            "bbox_robustness/results_seed2_pm20",
        ],
    ),
    "no-CKA rand100": (
        r"^lora_seed(\d)_rand100$",
        [
            "results/runs_rand100.csv",
            "results/runs_seed1_rand100.csv",
            "results/runs_seed2_rand100.csv",
        ],
        [
            "bbox_robustness/results_rand100",
            "bbox_robustness/results_seed1_rand100",
            "bbox_robustness/results_seed2_rand100",
        ],
    ),
    "CKA pm0": (
        r"^lora_cka_oodonly_late_l10_seed(\d)$",
        ["results/accv/runs_cka_pm0_retrained.csv"],  # same model as the sweep
        ["bbox_robustness/results_cka_oodonly_late_l10_pm0"],
    ),
    "CKA pm20": (
        r"^lora_cka_oodonly_late_l10_pm20_seed(\d)$",
        [
            "cka/results/runs_cka_oodonly_late_l10_pm20.csv",
            "cka/results/runs_accv_t1.csv",
        ],
        [
            "bbox_robustness/results_cka_oodonly_late_l10_pm20",
            "bbox_robustness/results_accv_t1",
        ],
    ),
    "CKA rand100": (
        r"^lora_cka_oodonly_late_l10_rand100_seed(\d)$",
        ["cka/results/runs_cka_oodonly_late_l10_rand100.csv"],
        ["bbox_robustness/results_cka_oodonly_late_l10_rand100"],
    ),
}


def tag_group(run_name: str):
    m = _CKA_RE.match(run_name)
    if m:
        return "cka", int(m.group(1))
    m = _NO_RE.match(run_name)
    if m:
        return "no_cka", int(m.group(1))
    return None


def _read_tight(paths) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths if Path(p).exists()]
    if not frames:
        return pd.DataFrame(
            columns=["run_name", "dataset", "dice_mean", "iou_mean", "hd95_mean"]
        )
    return pd.concat(frames, ignore_index=True)


def seed_table(tight_paths) -> pd.DataFrame:
    """One row per dataset: mean/std/n per group over seeds, and CKA minus no-CKA deltas."""
    df = _read_tight(tight_paths)
    tags = df.run_name.map(tag_group)
    df = df[tags.notna()].copy()
    df["group"] = [t[0] for t in tags[tags.notna()]]
    df["seed"] = [t[1] for t in tags[tags.notna()]]
    df = df.drop_duplicates(["group", "seed", "dataset"], keep="last")
    rows = []
    for ds in DATASETS:
        row = {"dataset": ds}
        for g in ("cka", "no_cka"):
            sub = df[(df.group == g) & (df.dataset == ds)]
            row[f"{g}_n"] = int(len(sub))
            for m in ("dice", "iou", "hd95"):
                vals = sub[f"{m}_mean"].to_numpy(dtype=float)
                row[f"{g}_{m}_mean"] = float(vals.mean()) if len(vals) else np.nan
                row[f"{g}_{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        for m in ("dice", "iou", "hd95"):
            row[f"delta_{m}"] = row[f"cka_{m}_mean"] - row[f"no_cka_{m}_mean"]
        rows.append(row)
    return pd.DataFrame(rows)


def load_sweep_per_image(dirs, levels=LEVELS) -> pd.DataFrame:
    """Long table (group, seed, dataset, level, image_id, dice) from bbox-sweep per_image files."""
    rows = []
    for d in dirs:
        per_dir = Path(d) / "per_image"
        if not per_dir.is_dir():
            continue
        for p in sorted(per_dir.glob("*.csv")):
            m = _PER_IMAGE_RE.match(p.name)
            if not m or int(m.group("level")) not in levels:
                continue
            tag = tag_group(m.group("run"))
            if tag is None:
                continue
            df = pd.read_csv(p, usecols=["image_id", "dice_mean"])
            df = df.rename(columns={"dice_mean": "dice"})
            df["group"], df["seed"] = tag
            df["dataset"] = m.group("ds")
            df["level"] = int(m.group("level"))
            rows.append(df)
    if not rows:
        return pd.DataFrame(
            columns=["group", "seed", "dataset", "level", "image_id", "dice"]
        )
    return pd.concat(rows, ignore_index=True)


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values, same order as the input."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def paired_tests(per: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, level): Wilcoxon signed-rank on CKA minus no-CKA Dice, pairs matched on (seed, image_id), Holm over datasets within a level."""
    rows = []
    for level, g in per.groupby("level"):
        level_rows = []
        for ds in DATASETS:
            a = g[(g.group == "cka") & (g.dataset == ds)][["seed", "image_id", "dice"]]
            b = g[(g.group == "no_cka") & (g.dataset == ds)][
                ["seed", "image_id", "dice"]
            ]
            pair = a.merge(b, on=["seed", "image_id"], suffixes=("_cka", "_no"))
            if len(pair) < 5:
                continue
            delta = (pair.dice_cka - pair.dice_no).to_numpy()
            try:
                p_raw = float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0
            except ValueError:
                p_raw = 1.0
            level_rows.append(
                {
                    "dataset": ds,
                    "level": int(level),
                    "n_pairs": int(len(pair)),
                    "n_seeds": int(pair.seed.nunique()),
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(np.median(delta)),
                    "frac_improved": float((delta > 0).mean()),
                    "p_raw": p_raw,
                }
            )
        if level_rows:
            ps = holm(np.array([r["p_raw"] for r in level_rows]))
            for r, p_adj in zip(level_rows, ps):
                r["p_holm"] = float(p_adj)
            rows.extend(level_rows)
    return pd.DataFrame(rows)


def curve_series(run_regex: str, tight_paths, sweep_dirs) -> pd.DataFrame:
    """(seed, dataset, level, dice): level 0 from tight-box CSVs, other levels from sweep runs.csv files."""
    rx = re.compile(run_regex)
    rows = []
    tight = _read_tight([REPO_ROOT / p for p in tight_paths])
    for _, r in tight.iterrows():
        m = rx.match(str(r.run_name))
        if m and r.dataset in DATASETS:
            rows.append(
                {
                    "seed": int(m.group(1)),
                    "dataset": r.dataset,
                    "level": 0,
                    "dice": float(r.dice_mean),
                }
            )
    for d in sweep_dirs:
        p = REPO_ROOT / d / "runs.csv"
        if not p.exists():
            continue
        sw = pd.read_csv(p)
        for _, r in sw.iterrows():
            m = rx.match(str(r.run_name))
            if m and r.dataset in DATASETS:
                rows.append(
                    {
                        "seed": int(m.group(1)),
                        "dataset": r.dataset,
                        "level": int(r.perturb_max_px),
                        "dice": float(r.dice_mean),
                    }
                )
    df = pd.DataFrame(rows, columns=["seed", "dataset", "level", "dice"])
    return df.drop_duplicates(["seed", "dataset", "level"], keep="last")


def plot_six_curves(out: Path) -> pd.DataFrame:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharey=False)
    colors = {"no-CKA": "tab:grey", "CKA": "tab:red"}
    styles = {"pm0": "-", "pm20": "--", "rand100": ":"}
    summary = []
    for label, (rx, tight, sweeps) in CURVES.items():
        df = curve_series(rx, tight, sweeps)
        if df.empty:
            print(f"[curves] {label}: no data, skipped")
            continue
        fam, regime = label.split(" ")
        for ax, ds in zip(axes, DATASETS):
            g = (
                df[df.dataset == ds]
                .groupby("level")["dice"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            ax.plot(
                g.level,
                g["mean"],
                styles[regime],
                color=colors[fam],
                marker="o",
                ms=3,
                label=f"{label} (n={int(g['count'].max())})",
            )
            if (g["count"] > 1).any():
                sd = g["std"].fillna(0.0)
                ax.fill_between(
                    g.level,
                    g["mean"] - sd,
                    g["mean"] + sd,
                    color=colors[fam],
                    alpha=0.12,
                )
            for _, r in g.iterrows():
                summary.append(
                    {
                        "curve": label,
                        "dataset": ds,
                        "level": int(r.level),
                        "dice_mean": r["mean"],
                        "dice_std": r["std"],
                        "n_seeds": int(r["count"]),
                    }
                )
    for ax, ds in zip(axes, DATASETS):
        ax.set_title(ds)
        ax.set_xlabel("bbox expansion (px)")
        ax.set_xticks([0, 20, 50, 100, 200])
    axes[0].set_ylabel("Dice")
    axes[-1].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return pd.DataFrame(summary)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "results/accv/cka_multiseed"
    )
    ap.add_argument(
        "--figure",
        type=Path,
        default=REPO_ROOT / "figures/accv/robustness_six_curves.png",
    )
    args = ap.parse_args(argv)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    table = seed_table([REPO_ROOT / p for p in TIGHT_PM20])
    table.to_csv(out / "seed_table.csv", index=False)
    print("Tight boxes, pm=20 training, mean over seeds (CKA n / no-CKA n):")
    show = table[
        [
            "dataset",
            "cka_n",
            "no_cka_n",
            "cka_dice_mean",
            "cka_dice_std",
            "no_cka_dice_mean",
            "no_cka_dice_std",
            "delta_dice",
            "delta_iou",
            "delta_hd95",
        ]
    ]
    print(show.round(4).to_string(index=False))

    per = load_sweep_per_image([REPO_ROOT / d for d in SWEEP_PM20])
    tests = paired_tests(per)
    tests.to_csv(out / "paired_wilcoxon.csv", index=False)
    print(
        "\nPaired Wilcoxon (CKA minus no-CKA per-image Dice, pairs matched on seed and image, Holm over datasets per level):"
    )
    print(tests.round(4).to_string(index=False))

    summary = plot_six_curves(args.figure)
    summary.to_csv(out / "robustness_curves.csv", index=False)
    print(f"\n[cka-multiseed] figure -> {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
