"""Failure self-detection: AUROC/AUPRC of iou_pred and drift against Dice-threshold failure labels, plus iou_pred calibration."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASETS = ["isic2018_test", "ph2", "busi", "cbis_ddsm"]
FAR_OOD = ["busi", "cbis_ddsm"]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUROC (ties get half credit). nan if only one class."""
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(np.asarray(scores, dtype=np.float64))
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision (area under the precision-recall step curve)."""
    labels = np.asarray(labels, dtype=bool)
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    hits = labels[order]
    tp = np.cumsum(hits)
    precision = tp / np.arange(1, len(hits) + 1)
    return float(precision[hits].sum() / labels.sum())


def _bin_index(conf: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)


def ece(conf: np.ndarray, truth: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error of a confidence (iou_pred) against the true value (iou)."""
    conf = np.asarray(conf, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    idx = _bin_index(conf, n_bins)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(conf[m].mean() - truth[m].mean())
    return float(total)


def reliability_bins(conf, truth, n_bins: int = 10) -> pd.DataFrame:
    conf = np.asarray(conf, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = _bin_index(conf, n_bins)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append(
            {
                "bin_lo": edges[b],
                "bin_hi": edges[b + 1],
                "n": int(m.sum()),
                "conf_mean": float(conf[m].mean()) if m.any() else float("nan"),
                "true_mean": float(truth[m].mean()) if m.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def load_per_image(raw_dir: Path, run_name: str, dataset: str) -> pd.DataFrame:
    return pd.read_csv(Path(raw_dir) / f"{run_name}_{dataset}_per_image.csv")


def discover_runs(raw_dir: Path) -> list[str]:
    pat = re.compile(r"^(.+)_(" + "|".join(DATASETS) + r")_per_image\.csv$")
    names = set()
    for p in Path(raw_dir).glob("*_per_image.csv"):
        m = pat.match(p.name)
        if m:
            names.add(m.group(1))
    return sorted(names)


def build_table(frames: dict, thresholds=(0.5, 0.7)) -> pd.DataFrame:
    rows = []
    for (run_name, dataset), df in frames.items():
        for thr in thresholds:
            fail = df["dice"].to_numpy() < thr
            detectors = {"iou_pred": -df["iou_pred"].to_numpy()}
            if "drift" in df.columns:
                detectors["drift"] = df["drift"].to_numpy()
            for det, score in detectors.items():
                rows.append(
                    {
                        "run_name": run_name,
                        "dataset": dataset,
                        "detector": det,
                        "threshold": thr,
                        "auroc": auroc(score, fail),
                        "auprc": auprc(score, fail),
                        "n": int(len(df)),
                        "n_fail": int(fail.sum()),
                    }
                )
    return pd.DataFrame(rows)


def build_calibration(frames: dict, n_bins: int = 10) -> pd.DataFrame:
    rows = []
    for (run_name, dataset), df in frames.items():
        rows.append(
            {
                "run_name": run_name,
                "dataset": dataset,
                "n": int(len(df)),
                "ece": ece(df["iou_pred"].to_numpy(), df["iou"].to_numpy(), n_bins),
                "iou_pred_mean": float(df["iou_pred"].mean()),
                "iou_mean": float(df["iou"].mean()),
            }
        )
    return pd.DataFrame(rows)


def make_figure(
    table: pd.DataFrame,
    frames: dict,
    out_path: Path,
    threshold: float,
    scatter_run: str | None,
    scatter_dataset: str = "cbis_ddsm",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = table[(table.threshold == threshold) & (table.dataset.isin(FAR_OOD))]
    datasets = [ds for ds in FAR_OOD if ds in set(sub.dataset)]
    runs = sorted(sub.run_name.unique())
    detectors = sorted(sub.detector.unique())
    n_panels = max(len(datasets), 1) + (1 if scatter_run else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 3.6))
    axes = np.atleast_1d(axes)
    width = 0.8 / max(len(detectors), 1)
    x = np.arange(len(runs))
    for ax, ds in zip(axes[: len(datasets)], datasets):
        for k, det in enumerate(detectors):
            vals = []
            for r in runs:
                cell = sub[
                    (sub.run_name == r) & (sub.dataset == ds) & (sub.detector == det)
                ]
                vals.append(float(cell.auroc.iloc[0]) if len(cell) else np.nan)
            ax.bar(x + (k - (len(detectors) - 1) / 2) * width, vals, width, label=det)
        ax.axhline(0.5, color="grey", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(runs, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(f"{ds}: AUROC (Dice < {threshold})")
        ax.legend(fontsize=7)
    if scatter_run:
        ax = axes[-1]
        df = frames.get((scatter_run, scatter_dataset))
        if df is not None:
            ax.scatter(df["iou_pred"], df["dice"], s=8, alpha=0.6)
            ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--")
            ax.set_xlabel("iou_pred")
            ax.set_ylabel("Dice")
            ax.set_title(f"{scatter_run} on {scatter_dataset}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "results/accv/raw")
    ap.add_argument(
        "--runs", nargs="*", default=None, help="Run names (default: discover)"
    )
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--thresholds", nargs="*", type=float, default=[0.5, 0.7])
    ap.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "results/accv/failure_detection"
    )
    ap.add_argument("--figure", type=Path, default=None)
    ap.add_argument(
        "--scatter-run", default=None, help="Run for the iou_pred vs Dice scatter"
    )
    args = ap.parse_args(argv)

    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else REPO_ROOT / args.raw_dir
    runs = args.runs or discover_runs(raw_dir)
    frames = {}
    for r in runs:
        for ds in args.datasets:
            p = raw_dir / f"{r}_{ds}_per_image.csv"
            if p.exists():
                frames[(r, ds)] = pd.read_csv(p)
    if not frames:
        print(f"[failure-detection] no per-image CSVs under {raw_dir}")
        return 1

    table = build_table(frames, tuple(args.thresholds))
    calib = build_calibration(frames)
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "detector_metrics.csv", index=False)
    calib.to_csv(out_dir / "calibration.csv", index=False)
    for (r, ds), df in frames.items():
        reliability_bins(df["iou_pred"], df["iou"]).to_csv(
            out_dir / f"reliability_{r}_{ds}.csv", index=False
        )

    primary = table[table.threshold == args.thresholds[0]]
    print(f"AUROC, failure = Dice < {args.thresholds[0]}")
    print(
        primary.pivot_table(
            index=["run_name", "dataset"], columns="detector", values="auroc"
        )
        .round(3)
        .to_string()
    )
    print("\nn_fail / n per run and dataset")
    print(
        primary[primary.detector == "iou_pred"][
            ["run_name", "dataset", "n_fail", "n"]
        ].to_string(index=False)
    )
    print("\niou_pred calibration (ECE, 10 bins)")
    print(calib.round(3).to_string(index=False))
    print(f"\n[failure-detection] wrote {out_dir / 'detector_metrics.csv'}")

    fig_path = args.figure or (out_dir / "failure_detection.png")
    make_figure(table, frames, fig_path, args.thresholds[0], args.scatter_run)
    print(f"[failure-detection] figure -> {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
