"""Risk-coverage and review-budget analysis for the per-image failure signals.

For each run, dataset and detector (drift, or the negated IoU estimate), images
are ranked by the signal, most suspicious first. Flagging the top c of them for
review gives: recall of failures among the flagged, precision of the flag, and
the failure rate that remains among the unflagged predictions (selective risk at
coverage 1 - c). The full curve over c yields the area under the risk-coverage
curve (AURC); a random ranking has AURC equal to the base failure rate.

Inputs are the per-image CSVs written by src.eval (--drift), so nothing here
needs a GPU.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.failure_detection import discover_runs  # noqa: E402

FAR_OOD = ["busi", "cbis_ddsm"]
DETECTORS = ["drift", "iou_pred"]


def _scores(df: pd.DataFrame, detector: str) -> np.ndarray | None:
    """Higher score = more suspicious. None when the column is absent."""
    if detector == "drift":
        if "drift" not in df.columns:
            return None
        return df["drift"].to_numpy(dtype=np.float64)
    if detector == "iou_pred":
        return -df["iou_pred"].to_numpy(dtype=np.float64)
    raise ValueError(detector)


def risk_coverage_curve(score: np.ndarray, fail: np.ndarray) -> pd.DataFrame:
    """One row per number of flagged images k = 0..n.

    Columns: k, flagged (k / n), recall (flagged failures / all failures),
    precision (flagged failures / k), residual_risk (failure rate among the
    n - k unflagged images; the selective risk at coverage 1 - flagged).
    """
    n = len(score)
    order = np.argsort(-score, kind="stable")
    hits = np.concatenate([[0], np.cumsum(fail[order].astype(np.int64))])
    k = np.arange(n + 1)
    n_fail = int(fail.sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = hits / n_fail if n_fail else np.full(n + 1, np.nan)
        precision = np.where(k > 0, hits / np.maximum(k, 1), np.nan)
        remaining = n - k
        residual = np.where(
            remaining > 0, (n_fail - hits) / np.maximum(remaining, 1), np.nan
        )
    return pd.DataFrame(
        {
            "k": k,
            "flagged": k / n,
            "recall": recall,
            "precision": precision,
            "residual_risk": residual,
        }
    )


def aurc(curve: pd.DataFrame) -> float:
    """Area under residual risk vs coverage (coverage = 1 - flagged), trapezoidal."""
    c = curve[curve.residual_risk.notna()]
    coverage = (1.0 - c.flagged).to_numpy()[::-1]
    risk = c.residual_risk.to_numpy()[::-1]
    trap = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2.0
    return float(trap(risk, coverage)) if len(c) > 1 else float("nan")


def review_points(curve: pd.DataFrame, coverages: list[float]) -> list[dict]:
    """Recall, precision and residual risk when the top c of images are flagged."""
    n = int(curve.k.max())
    out = []
    for c in coverages:
        k = min(n, max(1, math.ceil(c * n)))
        row = curve.iloc[k]
        out.append(
            {
                "flagged": c,
                "k": k,
                "recall": float(row.recall),
                "precision": float(row.precision),
                "residual_risk": float(row.residual_risk),
            }
        )
    return out


def build(
    frames: dict, threshold: float, coverages: list[float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    points, curves = [], []
    for (run, ds), df in frames.items():
        fail = df["dice"].to_numpy(dtype=np.float64) < threshold
        base = float(fail.mean())
        for det in DETECTORS:
            score = _scores(df, det)
            if score is None:
                continue
            curve = risk_coverage_curve(score, fail)
            area = aurc(curve)
            for p in review_points(curve, coverages):
                points.append(
                    {
                        "run_name": run,
                        "dataset": ds,
                        "detector": det,
                        "threshold": threshold,
                        "n": int(len(df)),
                        "n_fail": int(fail.sum()),
                        "base_rate": base,
                        "aurc": area,
                        **p,
                    }
                )
            curve = curve.assign(run_name=run, dataset=ds, detector=det)
            curves.append(curve)
    return pd.DataFrame(points), pd.concat(curves, ignore_index=True)


def make_figure(curves: pd.DataFrame, out: Path, datasets: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(datasets), figsize=(4.4 * len(datasets), 3.4))
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        sub = curves[curves.dataset == ds]
        for run, g_run in sub.groupby("run_name"):
            color = None
            for det, ls in [("drift", "-"), ("iou_pred", "--")]:
                g = g_run[g_run.detector == det]
                if not len(g):
                    continue
                (line,) = ax.plot(
                    g.flagged,
                    g.residual_risk,
                    ls=ls,
                    color=color,
                    label=f"{run} ({det})",
                )
                color = line.get_color()
        ax.set_xlabel("fraction flagged for review")
        ax.set_ylabel("failure rate among unflagged")
        ax.set_title(ds)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=5)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--runs", nargs="*", default=None, help="default: discover")
    ap.add_argument("--datasets", nargs="*", default=FAR_OOD)
    ap.add_argument("--threshold", type=float, default=0.5, help="failure: Dice <")
    ap.add_argument("--coverages", nargs="*", type=float, default=[0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--figure", type=Path, default=None)
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
        print(f"[risk-coverage] no per-image CSVs under {raw_dir}")
        return 1

    points, curves = build(frames, args.threshold, args.coverages)
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    points.to_csv(out_dir / "review_points.csv", index=False)
    curves.to_csv(out_dir / "risk_coverage_curves.csv", index=False)

    print(f"failure = Dice < {args.threshold}; AURC (random = base_rate)")
    print(
        points.drop_duplicates(["run_name", "dataset", "detector"])
        .pivot_table(
            index=["run_name", "dataset"],
            columns="detector",
            values=["aurc", "base_rate"],
        )
        .round(3)
        .to_string()
    )
    for c in args.coverages:
        print(f"\nflag top {c:.0%}: recall of failures / residual failure rate")
        sub = points[points.flagged == c]
        print(
            sub.pivot_table(
                index=["run_name", "dataset"],
                columns="detector",
                values=["recall", "residual_risk"],
            )
            .round(3)
            .to_string()
        )
    if args.figure is not None:
        make_figure(curves, args.figure, args.datasets)
        print(f"[risk-coverage] figure -> {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
