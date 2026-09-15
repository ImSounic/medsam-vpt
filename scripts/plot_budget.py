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


def drift_summary(raw_dir: Path) -> pd.DataFrame:
    """Mean per-image decoder and encoder drift per (method, budget): in-domain and far-OOD (mean of BUSI, CBIS-DDSM)."""
    rows = {}
    for p in sorted(Path(raw_dir).glob("*_per_image.csv")):
        name = p.name[: -len("_per_image.csv")]
        for ds in [ID_SET] + FAR_OOD:
            if name.endswith("_" + ds):
                run = name[: -len(ds) - 1]
                parsed = parse_budget_run(run)
                if parsed is None:
                    continue
                df = pd.read_csv(p)
                key = parsed
                rows.setdefault(key, {})[ds] = (
                    float(df["drift"].mean()),
                    (
                        float(df["drift_enc"].mean())
                        if "drift_enc" in df.columns
                        else float("nan")
                    ),
                )
    out = []
    for (method, budget), per_ds in rows.items():
        far = [per_ds[d] for d in FAR_OOD if d in per_ds]
        out.append(
            {
                "method": method,
                "budget": budget,
                "id_drift": per_ds.get(ID_SET, (float("nan"), float("nan")))[0],
                "id_drift_enc": per_ds.get(ID_SET, (float("nan"), float("nan")))[1],
                "far_ood_drift": (
                    sum(v[0] for v in far) / len(far) if far else float("nan")
                ),
                "far_ood_drift_enc": (
                    sum(v[1] for v in far) / len(far) if far else float("nan")
                ),
            }
        )
    return pd.DataFrame(out)


def make_drift_figure(drift: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    for ax, col, title in zip(
        axes,
        ["far_ood_drift", "far_ood_drift_enc"],
        ["Decoder drift (far-OOD)", "Encoder drift (far-OOD)"],
    ):
        for method, g in drift.groupby("method"):
            g = g.sort_values("budget")
            ax.plot(g.budget, g[col], marker="o", label=method)
        ax.set_xscale("log")
        ax.set_xlabel("training images")
        ax.set_ylabel("1 - CKA to base")
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def make_figure(
    summary: pd.DataFrame, out: Path, drift: pd.DataFrame | None = None
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zs = summary.attrs.get("zero_shot", {})
    n_panels = 3 if drift is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 3.4))
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
    if drift is not None:
        ax = axes[2]
        for method, g in drift.groupby("method"):
            g = g.sort_values("budget")
            (line,) = ax.plot(
                g.budget, g.far_ood_drift, marker="o", label=f"{method} (decoder)"
            )
            ax.plot(
                g.budget,
                g.far_ood_drift_enc,
                marker="s",
                ls="--",
                color=line.get_color(),
                label=f"{method} (encoder)",
            )
        ax.set_xscale("log")
        ax.set_xlabel("training images")
        ax.set_ylabel("1 - CKA to base (far-OOD)")
        ax.set_title("Representation drift")
        ax.legend(fontsize=6)
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
    ap.add_argument(
        "--drift-raw-dir",
        type=Path,
        default=None,
        help="Per-image files with drift columns (T7) for the drift panel",
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
    drift = None
    if args.drift_raw_dir is not None:
        drift = drift_summary(args.drift_raw_dir)
        drift.sort_values(["method", "budget"]).to_csv(
            args.out.with_name(args.out.stem + "_drift.csv"), index=False
        )
        print(drift.sort_values(["method", "budget"]).round(4).to_string(index=False))
    make_figure(summary, args.out, drift=drift)
    print(f"[budget] figure -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
