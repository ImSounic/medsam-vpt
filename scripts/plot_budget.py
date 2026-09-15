"""Adaptation budget vs Dice: in-domain and far-OOD (mean of BUSI, CBIS-DDSM), one line per method.

Aggregates over seeds: budget points (50/250/1000) come from one CSV per seed
(seed 0 in runs_t2.csv, seeds 1-2 in runs_t6.csv), plotted as mean with a
+/- std shaded band. The full-data (2595) point and the zero-shot reference are
seed-0 only (no band).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FULL_BUDGET = 2595
FAR_OOD = ["busi", "cbis_ddsm"]
ID_SET = "isic2018_test"
# Full-data seed-0 pm=0 run names; read from the --full-csv inputs (encoder-only
# lives in its own results/accv/runs_encoder_only_seed0.csv).
FULL_RUNS = {
    "lora": "lora_seed0",
    "decoder_only": "decoder_only_seed0",
    "lora_encoder_only": "lora_encoder_only_r28_all_seed0",
}
_BUDGET_RE = re.compile(r"^(?P<method>.+)_n(?P<budget>\d+)_seed(?P<seed>\d+)$")


def parse_budget_run(run_name: str) -> tuple[str, int, int] | None:
    m = _BUDGET_RE.match(run_name)
    return (
        (m.group("method"), int(m.group("budget")), int(m.group("seed")))
        if m
        else None
    )


def _id_and_far(df: pd.DataFrame) -> tuple[float, float]:
    """One row per dataset expected; returns (in-domain Dice, far-OOD mean Dice)."""
    by_ds = df.groupby("dataset")["dice_mean"].mean()
    far = float(by_ds.reindex(FAR_OOD).mean())
    return float(by_ds.get(ID_SET, float("nan"))), far


def _read_concat(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths if p.exists()]
    if not frames:
        raise FileNotFoundError(f"none of these CSVs exist: {paths}")
    return pd.concat(frames, ignore_index=True)


def budget_long(budget_csvs: list[Path], full_csvs: list[Path]) -> pd.DataFrame:
    """Long table: one row per (method, budget, seed) with id_dice and far_ood_dice."""
    bud = _read_concat(budget_csvs)
    full = _read_concat(full_csvs)
    rows = []
    for run, g in bud.groupby("run_name"):
        parsed = parse_budget_run(run)
        if parsed is None:
            continue
        method, budget, seed = parsed
        idd, far = _id_and_far(g)
        rows.append(
            {
                "method": method,
                "budget": budget,
                "seed": seed,
                "id_dice": idd,
                "far_ood_dice": far,
            }
        )
    # Full-data 2595 point (seed 0 only).
    for method, run in FULL_RUNS.items():
        g = full[full.run_name == run]
        if len(g):
            idd, far = _id_and_far(g)
            rows.append(
                {
                    "method": method,
                    "budget": FULL_BUDGET,
                    "seed": 0,
                    "id_dice": idd,
                    "far_ood_dice": far,
                }
            )
    long = pd.DataFrame(rows)
    zs = full[full.run_name == "zero_shot"]
    zs_id, zs_far = _id_and_far(zs) if len(zs) else (float("nan"), float("nan"))
    long.attrs["zero_shot"] = {"id_dice": zs_id, "far_ood_dice": zs_far}
    return long


def aggregate_seeds(long: pd.DataFrame) -> pd.DataFrame:
    """Mean and (sample) std over seeds per (method, budget); std is NaN when n_seeds == 1."""
    agg = (
        long.groupby(["method", "budget"])
        .agg(
            n_seeds=("seed", "nunique"),
            id_dice_mean=("id_dice", "mean"),
            id_dice_std=("id_dice", "std"),
            far_ood_dice_mean=("far_ood_dice", "mean"),
            far_ood_dice_std=("far_ood_dice", "std"),
        )
        .reset_index()
    )
    agg.attrs["zero_shot"] = long.attrs.get("zero_shot", {})
    return agg


def first_drop_below_zero_shot(agg: pd.DataFrame) -> dict:
    zs_far = agg.attrs.get("zero_shot", {}).get("far_ood_dice", float("nan"))
    out: dict = {}
    for method, g in agg.groupby("method"):
        below = g[g.far_ood_dice_mean < zs_far].sort_values("budget")
        out[method] = int(below.budget.iloc[0]) if len(below) else None
    return out


def make_figure(agg: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zs = agg.attrs.get("zero_shot", {})
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    panels = [
        ("id_dice", "In-domain (ISIC test)"),
        ("far_ood_dice", "Far-OOD (mean BUSI, CBIS-DDSM)"),
    ]
    for ax, (col, title) in zip(axes, panels):
        for method, g in agg.groupby("method"):
            g = g.sort_values("budget")
            mean = g[f"{col}_mean"].to_numpy()
            std = g[f"{col}_std"].fillna(0.0).to_numpy()
            (line,) = ax.plot(g.budget, mean, marker="o", label=method)
            ax.fill_between(
                g.budget, mean - std, mean + std, color=line.get_color(), alpha=0.18
            )
        if zs.get(col) == zs.get(col):  # not NaN
            ax.axhline(zs[col], color="grey", ls="--", lw=0.9, label="zero-shot")
        ax.set_xscale("log")
        ax.set_xlabel("training images")
        ax.set_ylabel("Dice")
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--budget-csv",
        type=Path,
        nargs="+",
        default=[
            REPO_ROOT / "results/accv/runs_t2.csv",
            REPO_ROOT / "results/accv/runs_t6.csv",
        ],
        help="one CSV per seed of budget runs (seed 0 = runs_t2, seeds 1-2 = runs_t6)",
    )
    ap.add_argument(
        "--full-csv",
        type=Path,
        nargs="+",
        default=[
            REPO_ROOT / "results/runs.csv",
            REPO_ROOT / "results/accv/runs_encoder_only_seed0.csv",
        ],
        help="CSVs holding the 2595-image seed-0 runs and zero_shot",
    )
    ap.add_argument(
        "--out", type=Path, default=REPO_ROOT / "figures/accv/budget_curves.png"
    )
    args = ap.parse_args(argv)

    long = budget_long(args.budget_csv, args.full_csv)
    agg = aggregate_seeds(long)
    first_drop = first_drop_below_zero_shot(agg)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    agg.sort_values(["method", "budget"]).to_csv(
        args.out.with_suffix(".csv"), index=False
    )
    print(agg.sort_values(["method", "budget"]).round(4).to_string(index=False))
    zs = agg.attrs.get("zero_shot", {})
    print(f"[budget] zero-shot: id={zs.get('id_dice'):.4f} far={zs.get('far_ood_dice'):.4f}")
    for method, b in first_drop.items():
        print(f"[budget] {method}: first budget below zero-shot far-OOD = {b}")
    make_figure(agg, args.out)
    print(f"[budget] figure -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
