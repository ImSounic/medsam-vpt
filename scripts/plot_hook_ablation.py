"""Hook-placement ablation over seeds: no CKA vs decoder / encoder / both hooks (LoRA, pm=20 training), tight-box means with std, per-seed points, paired Wilcoxon between arms on the bbox-sweep per-image files."""

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
FAR_OOD = ["busi", "cbis_ddsm"]
ORDER = ["no_cka", "late", "enc", "both"]
LABELS = {
    "no_cka": "no CKA",
    "late": "decoder hooks",
    "enc": "encoder hooks",
    "both": "both",
}
_POS_RE = re.compile(
    r"^lora_cka_oodonly_(?P<pos>late|enc|both)_l10_pm20_seed(?P<seed>\d+)$"
)
_NO_RE = re.compile(r"^lora_seed(?P<seed>\d+)_pm20$")
_PER_IMAGE_RE = re.compile(
    r"^(?P<run>.+)_(?P<ds>isic2018_test|ph2|busi|cbis_ddsm)_pm(?P<level>\d+)\.csv$"
)

DEFAULT_CSVS = [
    "cka/results/runs_cka_oodonly_late_l10_pm20.csv",
    "cka/results/runs_accv_t1.csv",
    "cka/results/runs_accv_t5.csv",
]
DEFAULT_BASELINES = [
    "results/runs_pm20.csv",
    "results/runs_seed1_pm20.csv",
    "results/runs_seed2_pm20.csv",
]
DEFAULT_SWEEPS = [
    "bbox_robustness/results_cka_oodonly_late_l10_pm20",
    "bbox_robustness/results_accv_t1",
    "bbox_robustness/results_accv_t1_hooks",
    "bbox_robustness/results_accv_t5",
    "bbox_robustness/results_pm20",
    "bbox_robustness/results_seed1_pm20",
    "bbox_robustness/results_seed2_pm20",
]


def parse_position(run_name: str) -> str | None:
    m = _POS_RE.match(run_name)
    return m.group("pos") if m else None


def arm_of(run_name: str):
    m = _POS_RE.match(run_name)
    if m:
        return m.group("pos"), int(m.group("seed"))
    m = _NO_RE.match(run_name)
    if m:
        return "no_cka", int(m.group("seed"))
    return None


def seed_table_from_csvs(paths) -> pd.DataFrame:
    """One row per (arm, dataset): Dice mean, std and n over seeds, plus the per-seed values."""
    frames = [pd.read_csv(p) for p in paths if Path(p).exists()]
    df = pd.concat(frames, ignore_index=True)
    tags = df.run_name.map(arm_of)
    df = df[tags.notna()].copy()
    df["arm"] = [t[0] for t in tags[tags.notna()]]
    df["seed"] = [t[1] for t in tags[tags.notna()]]
    df = df.drop_duplicates(["arm", "seed", "dataset"], keep="last")
    rows = []
    for arm in ORDER:
        for ds in DATASETS:
            sub = df[(df.arm == arm) & (df.dataset == ds)].sort_values("seed")
            if len(sub) == 0:
                continue
            vals = sub.dice_mean.to_numpy(dtype=float)
            rows.append(
                {
                    "arm": arm,
                    "dataset": ds,
                    "n": int(len(vals)),
                    "dice_mean": float(vals.mean()),
                    "dice_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "per_seed": ";".join(
                        f"{s}:{v:.4f}" for s, v in zip(sub.seed, vals)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def _load_sweep(dirs, level: int) -> pd.DataFrame:
    rows = []
    for d in dirs:
        per = Path(d) / "per_image"
        if not per.is_dir():
            continue
        for p in sorted(per.glob(f"*_pm{level}.csv")):
            m = _PER_IMAGE_RE.match(p.name)
            if not m:
                continue
            tag = arm_of(m.group("run"))
            if tag is None:
                continue
            df = pd.read_csv(p, usecols=["image_id", "dice_mean"]).rename(
                columns={"dice_mean": "dice"}
            )
            df["arm"], df["seed"] = tag
            df["dataset"] = m.group("ds")
            rows.append(df)
    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["image_id", "dice", "arm", "seed", "dataset"])
    )


def paired_arm_tests(
    sweep_dirs,
    level: int = 20,
    pairs=(("late", "no_cka"), ("both", "late"), ("both", "no_cka"), ("enc", "no_cka")),
) -> pd.DataFrame:
    """Paired Wilcoxon on per-image Dice (pairs on seed and image) for arm_a minus arm_b, Holm over datasets within a pair."""
    per = _load_sweep(sweep_dirs, level)
    rows = []
    for a, b in pairs:
        pair_rows = []
        for ds in DATASETS:
            x = per[(per.arm == a) & (per.dataset == ds)][["seed", "image_id", "dice"]]
            y = per[(per.arm == b) & (per.dataset == ds)][["seed", "image_id", "dice"]]
            m = x.merge(y, on=["seed", "image_id"], suffixes=("_a", "_b"))
            if len(m) < 5:
                continue
            delta = (m.dice_a - m.dice_b).to_numpy()
            try:
                p = float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0
            except ValueError:
                p = 1.0
            pair_rows.append(
                {
                    "arm_a": a,
                    "arm_b": b,
                    "dataset": ds,
                    "level": level,
                    "n_pairs": int(len(m)),
                    "n_seeds": int(m.seed.nunique()),
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(np.median(delta)),
                    "frac_improved": float((delta > 0).mean()),
                    "p_raw": p,
                }
            )
        if pair_rows:
            for r, adj in zip(pair_rows, _holm([r["p_raw"] for r in pair_rows])):
                r["p_holm"] = float(adj)
            rows.extend(pair_rows)
    return pd.DataFrame(rows)


def make_figure(table: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [a for a in ORDER if a in set(table.arm)]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    x = np.arange(len(arms))
    w = 0.38
    for k, ds in enumerate(FAR_OOD):
        means, stds = [], []
        for arm in arms:
            r = table[(table.arm == arm) & (table.dataset == ds)]
            means.append(float(r.dice_mean.iloc[0]) if len(r) else np.nan)
            stds.append(float(r.dice_std.iloc[0]) if len(r) else 0.0)
        ax.bar(x + (k - 0.5) * w, means, w, yerr=stds, capsize=3, label=ds)
        for i, arm in enumerate(arms):
            r = table[(table.arm == arm) & (table.dataset == ds)]
            if len(r):
                vals = [float(v.split(":")[1]) for v in r.per_seed.iloc[0].split(";")]
                ax.scatter(
                    [x[i] + (k - 0.5) * w] * len(vals),
                    vals,
                    s=10,
                    color="black",
                    zorder=3,
                )
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in arms])
    ax.set_ylabel("Dice (tight boxes)")
    ax.set_title("CKA hook placement, pm=20 training, seeds 0 to 2")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv", type=Path, nargs="+", default=[REPO_ROOT / p for p in DEFAULT_CSVS]
    )
    ap.add_argument(
        "--baseline-csv",
        type=Path,
        nargs="+",
        default=[REPO_ROOT / p for p in DEFAULT_BASELINES],
    )
    ap.add_argument(
        "--baseline-run", default=None, help="Ignored; kept for compatibility"
    )
    ap.add_argument(
        "--sweep-dirs",
        type=Path,
        nargs="*",
        default=[REPO_ROOT / p for p in DEFAULT_SWEEPS],
    )
    ap.add_argument(
        "--out", type=Path, default=REPO_ROOT / "figures/accv/hook_ablation.png"
    )
    args = ap.parse_args(argv)

    table = seed_table_from_csvs(list(args.csv) + list(args.baseline_csv))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out.with_suffix(".csv"), index=False)
    print(table.round(4).to_string(index=False))
    tests = paired_arm_tests(args.sweep_dirs) if args.sweep_dirs else pd.DataFrame()
    if len(tests):
        tests.to_csv(args.out.with_name(args.out.stem + "_wilcoxon.csv"), index=False)
        print(
            "\nPaired Wilcoxon at 20 px (arm_a minus arm_b, pairs on seed and image, Holm over datasets):"
        )
        print(tests.round(4).to_string(index=False))
    make_figure(table, args.out)
    print(f"[hook-ablation] figure -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
