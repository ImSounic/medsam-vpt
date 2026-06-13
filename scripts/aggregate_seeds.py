"""Aggregate multi-seed eval results into mean +/- std across seeds 0, 1, 2."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

TRAININGS = ("pm=0", "pm=20", "rand100")
SEEDS = (0, 1, 2)


def tight_csv_for_seed(seed: int, training: str) -> Path:
    """Tight-bbox runs CSV for (seed, training)."""
    suffix = {"pm=0": "", "pm=20": "_pm20", "rand100": "_rand100"}[training]
    if seed == 0:
        if suffix == "":
            return REPO_ROOT / "results" / "runs.csv"
        return REPO_ROOT / "results" / f"runs{suffix}.csv"
    return REPO_ROOT / "results" / f"runs_seed{seed}{suffix}.csv"


def bbox_csv_for_seed(seed: int, training: str) -> Path:
    """Bbox-robustness runs CSV for (seed, training)."""
    suffix = {"pm=0": "", "pm=20": "_pm20", "rand100": "_rand100"}[training]
    if seed == 0:
        if suffix == "":
            return REPO_ROOT / "bbox_robustness" / "results" / "runs.csv"
        return REPO_ROOT / "bbox_robustness" / f"results{suffix}" / "runs.csv"
    return REPO_ROOT / "bbox_robustness" / f"results_seed{seed}{suffix}" / "runs.csv"


METHODS = [
    "zero_shot",
    "decoder_only",
    "vpt_shallow",
    "vpt_deep",
    "lora",
    "lora_encoder_only",
    "full_ft",
]
DATASETS = ["isic2018_test", "ph2", "busi", "cbis_ddsm"]
METHOD_LABELS = {
    "zero_shot": "Zero-shot",
    "decoder_only": "Decoder-only",
    "vpt_shallow": "VPT-shallow",
    "vpt_deep": "VPT-deep",
    "lora": "LoRA",
    "lora_encoder_only": "Encoder-only LoRA",
    "full_ft": "Full FT",
}
DATASET_LABELS = {
    "isic2018_test": "ISIC 2018 (ID)",
    "ph2": "PH2 (near-OOD)",
    "busi": "BUSI (far-OOD ultrasound)",
    "cbis_ddsm": "CBIS-DDSM (far-OOD mammography)",
}


def canonical_method_name(method: str, run_name: str) -> str:
    """Recover encoder-only LoRA identity from older CSVs if needed."""
    if method == "lora" and run_name.startswith("lora_encoder_only"):
        return "lora_encoder_only"
    return method


def load_all_seeds() -> pd.DataFrame:
    """Long-form DataFrame of tight (perturb 0) and bbox-robustness (perturb>0) dice rows."""
    rows = []

    for seed in SEEDS:
        for training in TRAININGS:
            # Tight-bbox rows (perturb_max_px = 0)
            tight_path = tight_csv_for_seed(seed, training)
            if tight_path.exists():
                tight_df = pd.read_csv(tight_path)
                for _, r in tight_df.iterrows():
                    rows.append(
                        {
                            "seed": seed,
                            "method": canonical_method_name(
                                str(r["method"]), str(r.get("run_name", ""))
                            ),
                            "training": training,
                            "dataset": r["dataset"],
                            "perturb_max_px": 0,
                            "dice_mean": float(r["dice_mean"]),
                        }
                    )

            # Bbox-robustness rows (perturb_max_px in 20, 50, 100, 200)
            bbox_path = bbox_csv_for_seed(seed, training)
            if bbox_path.exists():
                bbox_df = pd.read_csv(bbox_path)
                for _, r in bbox_df.iterrows():
                    rows.append(
                        {
                            "seed": seed,
                            "method": canonical_method_name(
                                str(r["method"]), str(r.get("run_name", ""))
                            ),
                            "training": training,
                            "dataset": r["dataset"],
                            "perturb_max_px": int(r["perturb_max_px"]),
                            "dice_mean": float(r["dice_mean"]),
                        }
                    )

    df = pd.DataFrame(rows)
    # zero_shot is identical across trainings (no checkpoint), so dedupe to one copy per (seed, dataset, perturb).
    df = df.drop_duplicates(
        subset=["seed", "method", "training", "dataset", "perturb_max_px"],
        keep="first",
    )
    return df


def aggregate_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    """Mean + std across seeds per (method, training, dataset, perturb)."""
    agg = (
        df.groupby(["method", "training", "dataset", "perturb_max_px"])
        .agg(
            dice_mean_seeds=("dice_mean", "mean"),
            dice_std_seeds=("dice_mean", "std"),
            n_seeds=("seed", "nunique"),
            seeds=("seed", lambda s: ",".join(map(str, sorted(s.unique())))),
        )
        .reset_index()
    )
    # Single-seed cells have NaN std; set to 0.
    agg["dice_std_seeds"] = agg["dice_std_seeds"].fillna(0.0)
    return agg


def write_csv(agg: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = agg.copy()
    for c in ("dice_mean_seeds", "dice_std_seeds"):
        df[c] = df[c].round(4)
    df.to_csv(out_path, index=False)
    print(f"[agg-seeds] wrote {out_path}")


def write_summary_markdown(agg: pd.DataFrame, out_path: Path) -> None:
    """Per-dataset method x training table, cell = mean +/- std at pm=0 and pm=200."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Multi-seed summary  (mean +/- std across seeds 0, 1, 2)\n"]

    for ds in DATASETS:
        lines.append(f"\n## {DATASET_LABELS.get(ds, ds)}\n")
        lines.append(
            "| Method | Training | Dice @ pm=0 (tight) | Dice @ pm=200 (extreme) | n_seeds |"
        )
        lines.append("|---|---|---:|---:|---:|")
        for m in METHODS:
            if m == "zero_shot":
                tight = agg[
                    (agg["method"] == m)
                    & (agg["dataset"] == ds)
                    & (agg["perturb_max_px"] == 0)
                ]
                extreme = agg[
                    (agg["method"] == m)
                    & (agg["dataset"] == ds)
                    & (agg["perturb_max_px"] == 200)
                ]
                if tight.empty:
                    continue
                # zero_shot only has pm=0 training entries (same checkpoint reused)
                r_t = tight.iloc[0]
                r_e = extreme.iloc[0] if not extreme.empty else None
                t_cell = f"{r_t['dice_mean_seeds']:.4f} +/- {r_t['dice_std_seeds']:.4f}"
                e_cell = (
                    f"{r_e['dice_mean_seeds']:.4f} +/- {r_e['dice_std_seeds']:.4f}"
                    if r_e is not None
                    else "-"
                )
                lines.append(
                    f"| {METHOD_LABELS[m]} | - | {t_cell} | {e_cell} | {int(r_t['n_seeds'])} |"
                )
            else:
                for tr in TRAININGS:
                    tight = agg[
                        (agg["method"] == m)
                        & (agg["training"] == tr)
                        & (agg["dataset"] == ds)
                        & (agg["perturb_max_px"] == 0)
                    ]
                    extreme = agg[
                        (agg["method"] == m)
                        & (agg["training"] == tr)
                        & (agg["dataset"] == ds)
                        & (agg["perturb_max_px"] == 200)
                    ]
                    if tight.empty:
                        continue
                    r_t = tight.iloc[0]
                    r_e = extreme.iloc[0] if not extreme.empty else None
                    t_cell = (
                        f"{r_t['dice_mean_seeds']:.4f} +/- {r_t['dice_std_seeds']:.4f}"
                    )
                    e_cell = (
                        f"{r_e['dice_mean_seeds']:.4f} +/- {r_e['dice_std_seeds']:.4f}"
                        if r_e is not None
                        else "-"
                    )
                    lines.append(
                        f"| {METHOD_LABELS[m]} | {tr} | {t_cell} | {e_cell} | {int(r_t['n_seeds'])} |"
                    )

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[agg-seeds] wrote {out_path}")


def main() -> int:
    print("[agg-seeds] scanning for seed CSVs...")
    df = load_all_seeds()
    if df.empty:
        print(
            "[agg-seeds] No CSVs found. Have you trained and evaluated seed=1/seed=2 yet?"
        )
        return 1

    print(
        f"[agg-seeds] loaded {len(df)} raw rows across {df['seed'].nunique()} seed(s): {sorted(df['seed'].unique())}"
    )
    agg = aggregate_mean_std(df)
    print(
        f"[agg-seeds] aggregated to {len(agg)} unique (method, training, dataset, perturb) cells"
    )

    write_csv(agg, REPO_ROOT / "results" / "summary_full_multiseed.csv")
    write_summary_markdown(agg, REPO_ROOT / "results" / "summary_multiseed.md")

    # n_seeds per cell surfaces incomplete runs
    n_seeds_dist = agg["n_seeds"].value_counts().sort_index()
    print(f"\n[agg-seeds] n_seeds distribution across {len(agg)} cells:")
    for n, count in n_seeds_dist.items():
        print(f"             {n} seed(s): {count} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
