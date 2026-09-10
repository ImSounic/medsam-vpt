"""Hook-placement ablation: far-OOD Dice for no-CKA vs decoder / encoder / both hooks (seed 0, pm=20)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def parse_position(run_name: str) -> str | None:
    m = _POS_RE.match(run_name)
    return m.group("pos") if m else None


def ablation_table(csv: Path, baseline_csv: Path, baseline_run: str) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df["position"] = df.run_name.map(parse_position)
    df = df[df.position.notna() & df.run_name.str.endswith("seed0")]
    base = pd.read_csv(baseline_csv)
    base = base[base.run_name == baseline_run].copy()
    base["position"] = "no_cka"
    both = pd.concat([base, df], ignore_index=True)
    both = both[both.dataset.isin(FAR_OOD)]
    return both.pivot_table(
        index="position", columns="dataset", values="dice_mean"
    ).reindex(ORDER)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv", type=Path, default=REPO_ROOT / "cka/results/runs_accv_t1.csv"
    )
    ap.add_argument(
        "--baseline-csv", type=Path, default=REPO_ROOT / "results/runs_pm20.csv"
    )
    ap.add_argument("--baseline-run", default="lora_seed0_pm20")
    ap.add_argument(
        "--out", type=Path, default=REPO_ROOT / "figures/accv/hook_ablation.png"
    )
    args = ap.parse_args(argv)

    table = ablation_table(args.csv, args.baseline_csv, args.baseline_run)
    print(table.round(4).to_string())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out.with_suffix(".csv"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.4))
    x = np.arange(len(table.index))
    w = 0.38
    for k, ds in enumerate(FAR_OOD):
        ax.bar(x + (k - 0.5) * w, table[ds].to_numpy(), w, label=ds)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[p] for p in table.index])
    ax.set_ylabel("Dice (tight boxes)")
    ax.set_title("CKA hook placement, seed 0, pm=20")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"[hook-ablation] figure -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
