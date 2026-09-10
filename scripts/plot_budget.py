"""Adaptation budget vs Dice: in-domain and far-OOD (mean of BUSI, CBIS-DDSM), one line per method, zero-shot as reference."""

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
# Full-data seed-0 pm=0 run names in results/runs.csv (encoder-only may be absent locally).
FULL_RUNS = {
    "lora": "lora_seed0",
    "decoder_only": "decoder_only_seed0",
    "lora_encoder_only": "lora_encoder_only_r28_all_seed0",
}
_BUDGET_RE = re.compile(r"^(?P<method>.+)_n(?P<budget>\d+)_seed\d+$")


def parse_budget_run(run_name: str) -> tuple[str, int] | None:
    m = _BUDGET_RE.match(run_name)
    return (m.group("method"), int(m.group("budget"))) if m else None


def _id_and_far(df: pd.DataFrame) -> tuple[float, float]:
    by_ds = df.groupby("dataset")["dice_mean"].mean()
    far = float(by_ds.reindex(FAR_OOD).mean())
    return float(by_ds.get(ID_SET, float("nan"))), far


def budget_summary(budget_csv: Path, full_csv: Path):
    bud = pd.read_csv(budget_csv)
    full = pd.read_csv(full_csv)
    rows = []
    for run, g in bud.groupby("run_name"):
        parsed = parse_budget_run(run)
        if parsed is None:
            continue
        method, budget = parsed
        idd, far = _id_and_far(g)
        rows.append(
            {"method": method, "budget": budget, "id_dice": idd, "far_ood_dice": far}
        )
    for method, run in FULL_RUNS.items():
        g = full[full.run_name == run]
        if len(g):
            idd, far = _id_and_far(g)
            rows.append(
                {
                    "method": method,
                    "budget": FULL_BUDGET,
                    "id_dice": idd,
                    "far_ood_dice": far,
                }
            )
    zs = full[full.run_name == "zero_shot"]
    zs_id, zs_far = _id_and_far(zs) if len(zs) else (float("nan"), float("nan"))
    summary = pd.DataFrame(rows)
    summary.attrs["zero_shot"] = {"id_dice": zs_id, "far_ood_dice": zs_far}
    first_drop: dict = {}
    for method, g in summary.groupby("method"):
        below = g[g.far_ood_dice < zs_far].sort_values("budget")
        first_drop[method] = int(below.budget.iloc[0]) if len(below) else None
    return summary, first_drop


def make_figure(summary: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zs = summary.attrs.get("zero_shot", {})
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    panels = [
        ("id_dice", "In-domain (ISIC test)"),
        ("far_ood_dice", "Far-OOD (mean BUSI, CBIS-DDSM)"),
    ]
    for ax, (col, title) in zip(axes, panels):
        for method, g in summary.groupby("method"):
            g = g.sort_values("budget")
            ax.plot(g.budget, g[col], marker="o", label=method)
        if zs.get(col) == zs.get(col):
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
        "--budget-csv", type=Path, default=REPO_ROOT / "results/accv/runs_t2.csv"
    )
    ap.add_argument("--full-csv", type=Path, default=REPO_ROOT / "results/runs.csv")
    ap.add_argument(
        "--out", type=Path, default=REPO_ROOT / "figures/accv/budget_curves.png"
    )
    args = ap.parse_args(argv)
    summary, first_drop = budget_summary(args.budget_csv, args.full_csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.sort_values(["method", "budget"]).to_csv(
        args.out.with_suffix(".csv"), index=False
    )
    print(summary.sort_values(["method", "budget"]).round(4).to_string(index=False))
    for method, b in first_drop.items():
        print(f"[budget] {method}: first budget below zero-shot far-OOD = {b}")
    make_figure(summary, args.out)
    print(f"[budget] figure -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
