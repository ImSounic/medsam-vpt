"""Failure-detection metrics over seeds: AUROC / AUPRC per seed, then mean and std per method.

Reads per-image files from several raw directories (seed 0 from T3 and the DMID
job, seeds 1-2 from T8), scores each run with scripts.failure_detection, maps run
names such as lora_seed1 or lora_encoder_only_r28_all_seed2 to a method, and
aggregates over seeds. Zero-shot has one run and no std.
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

from scripts.failure_detection import build_table  # noqa: E402

_SEED_RE = re.compile(r"^(?P<method>.+?)(?:_r28_all)?_seed(?P<seed>\d+)$")


def split_run(run_name: str) -> tuple[str, int]:
    """'lora_encoder_only_r28_all_seed2' -> ('lora_encoder_only', 2); 'zero_shot' -> ('zero_shot', 0)."""
    m = _SEED_RE.match(run_name)
    return (m.group("method"), int(m.group("seed"))) if m else (run_name, 0)


def load_frames(raw_dirs: list[Path], datasets: list[str]) -> dict:
    frames = {}
    for d in raw_dirs:
        for p in sorted(Path(d).glob("*_per_image.csv")):
            for ds in datasets:
                suffix = f"_{ds}_per_image.csv"
                if p.name.endswith(suffix):
                    run = p.name[: -len(suffix)]
                    frames[(run, ds)] = pd.read_csv(p)
                    break
    return frames


def per_seed_table(frames: dict, threshold: float) -> pd.DataFrame:
    t = build_table(frames, (threshold,))
    if not len(t):
        return t
    ms = t.run_name.map(split_run)
    t = t.assign(method=[m for m, _ in ms], seed=[s for _, s in ms])
    return t


def aggregate(per_seed: pd.DataFrame, min_fail: int = 20) -> pd.DataFrame:
    """Mean and std of AUROC / AUPRC over seeds whose failure count reaches min_fail."""
    ok = per_seed[per_seed.n_fail >= min_fail]
    agg = (
        ok.groupby(["method", "dataset", "detector"])
        .agg(
            n_seeds=("seed", "nunique"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
            n_fail_mean=("n_fail", "mean"),
        )
        .reset_index()
    )
    return agg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw-dirs",
        nargs="+",
        type=Path,
        default=[
            REPO_ROOT / "results/accv/raw_t3_pm0",
            REPO_ROOT / "results/accv/raw_t8_pm0",
        ],
    )
    ap.add_argument("--datasets", nargs="*", default=["busi", "cbis_ddsm", "dmid"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-fail", type=int, default=20)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results/accv/failure_detection_multiseed",
    )
    args = ap.parse_args(argv)

    frames = load_frames(args.raw_dirs, args.datasets)
    if not frames:
        print(f"[detector-multiseed] no per-image CSVs under {args.raw_dirs}")
        return 1
    per_seed = per_seed_table(frames, args.threshold)
    agg = aggregate(per_seed, args.min_fail)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(args.out_dir / "per_seed.csv", index=False)
    agg.to_csv(args.out_dir / "detector_metrics_multiseed.csv", index=False)
    print(f"AUROC mean (std) over seeds, failure = Dice < {args.threshold}")
    view = agg.assign(
        cell=lambda d: d.apply(
            lambda r: (
                f"{r.auroc_mean:.2f} ({r.auroc_std:.2f}, n={int(r.n_seeds)})"
                if r.n_seeds > 1
                else f"{r.auroc_mean:.2f} (n=1)"
            ),
            axis=1,
        )
    )
    print(
        view.pivot_table(
            index=["method", "dataset"],
            columns="detector",
            values="cell",
            aggfunc="first",
        ).to_string()
    )
    print(f"[detector-multiseed] wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
