"""Paired significance tests for the three headline claims: across-seed paired t (n=3, low power) plus per-image Wilcoxon on dice pooled over 3 seeds."""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "bbox_robustness" / "comparison"
OUT_MD = OUT_DIR / "seed_significance.md"
OUT_CSV = OUT_DIR / "seed_significance.csv"

SEEDS = (0, 1, 2)

# Source CSV paths, mirrors scripts/aggregate_seeds.py
def tight_csv_for_seed(seed: int, training: str) -> Path:
    suffix = {"pm=0": "", "pm=20": "_pm20", "rand100": "_rand100"}[training]
    if seed == 0:
        if suffix == "":
            return REPO_ROOT / "results" / "runs.csv"
        return REPO_ROOT / "results" / f"runs{suffix}.csv"
    return REPO_ROOT / "results" / f"runs_seed{seed}{suffix}.csv"


def bbox_runs_csv_for_seed(seed: int, training: str) -> Path:
    suffix = {"pm=0": "", "pm=20": "_pm20", "rand100": "_rand100"}[training]
    if seed == 0:
        if suffix == "":
            return REPO_ROOT / "bbox_robustness" / "results" / "runs.csv"
        return REPO_ROOT / "bbox_robustness" / f"results{suffix}" / "runs.csv"
    return REPO_ROOT / "bbox_robustness" / f"results_seed{seed}{suffix}" / "runs.csv"


def per_image_csv_for_seed(seed: int, training: str, method: str,
                           dataset: str, perturb: int) -> Path:
    """Per-image CSV for one (seed, training, method, dataset, perturb>0); zero_shot file is seed/training-independent."""
    suffix = {"pm=0": "", "pm=20": "_pm20", "rand100": "_rand100"}[training]
    if method == "zero_shot":
        return (REPO_ROOT / "bbox_robustness" / "results" / "per_image"
                / f"zero_shot_{dataset}_pm{perturb}.csv")
    stem = f"{method}_seed{seed}{suffix}_{dataset}_pm{perturb}"
    if seed == 0:
        ckpt_dir = REPO_ROOT / "bbox_robustness" / f"results{suffix}" / "per_image"
    else:
        ckpt_dir = REPO_ROOT / "bbox_robustness" / f"results_seed{seed}{suffix}" / "per_image"
    return ckpt_dir / f"{stem}.csv"


# Loaders

def load_per_seed_means(method: str, training: str, dataset: str,
                        perturb: int) -> list[float]:
    """Returns a list of dice means (one per seed) for one cell."""
    means = []
    for seed in SEEDS:
        if perturb == 0:
            csv_path = tight_csv_for_seed(seed, training)
        else:
            csv_path = bbox_runs_csv_for_seed(seed, training)
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "perturb_max_px" in df.columns:
            df = df[df["perturb_max_px"] == perturb]
        sub = df[(df["method"] == method) & (df["dataset"] == dataset)]
        if not sub.empty:
            means.append(float(sub["dice_mean"].iloc[0]))
    return means


def load_per_image_dice(method: str, training: str, dataset: str,
                        perturb: int) -> pd.DataFrame | None:
    """DataFrame of per-image dice with one column per seed (same series for zero_shot)."""
    if perturb == 0:
        return None  # tight-bbox per-image only exists for seed-0 + zero_shot

    per_seed = {}
    for seed in SEEDS:
        csv_path = per_image_csv_for_seed(seed, training, method, dataset, perturb)
        if not csv_path.exists():
            return None
        sub = pd.read_csv(csv_path)
        per_seed[seed] = sub.set_index("image_id")["dice_mean"]

    df = pd.DataFrame(per_seed)
    df.columns = [f"dice_seed{s}" for s in per_seed.keys()]
    df = df.dropna()
    return df


def per_image_seed_avg(method: str, training: str, dataset: str,
                       perturb: int) -> pd.Series | None:
    """Returns a Series indexed by image_id, value = mean dice across 3 seeds."""
    df = load_per_image_dice(method, training, dataset, perturb)
    if df is None or df.empty:
        return None
    return df.mean(axis=1)


# Tests

@dataclass
class ClaimResult:
    name: str
    description: str
    seed_diffs: list[float]
    seed_t: float | None
    seed_p_one_sided: float | None
    image_n: int | None
    image_w: float | None
    image_p_one_sided: float | None
    image_median_diff: float | None
    pass_seed: bool | None
    pass_image: bool | None


def run_paired_t_seeds(diffs: list[float], greater_is_better: bool) -> tuple[float, float]:
    """Paired t-test on per-seed differences (n=3); returns (t-stat, one-sided p)."""
    a = np.asarray(diffs, dtype=float)
    if len(a) < 2 or a.std(ddof=1) < 1e-12:
        return float("nan"), float("nan")
    # one-sided: test diff > 0 if greater_is_better else diff < 0
    alt = "greater" if greater_is_better else "less"
    res = stats.ttest_1samp(a, 0.0, alternative=alt)
    return float(res.statistic), float(res.pvalue)


def run_paired_wilcoxon(a: pd.Series, b: pd.Series,
                        greater_is_better: bool) -> tuple[int, float, float, float]:
    """One-sided Wilcoxon signed-rank on per-image dice; returns (n, W, one-sided p, median diff)."""
    common = a.index.intersection(b.index)
    diff = (a.loc[common] - b.loc[common]).dropna()
    if len(diff) < 5 or (diff == 0).all():
        return len(diff), float("nan"), float("nan"), float(diff.median()) if len(diff) else float("nan")
    alt = "greater" if greater_is_better else "less"
    try:
        res = stats.wilcoxon(diff, alternative=alt, zero_method="wilcox")
        return len(diff), float(res.statistic), float(res.pvalue), float(diff.median())
    except ValueError:
        return len(diff), float("nan"), float("nan"), float(diff.median())


# Each claim tests diff = (A - B) > 0 one-sided; for ">=" claims we check diff < 0 is not rejected
CLAIM_DEFINITIONS = [
    {
        "name": "C1a-isic",
        "description": "Across all 5 PEFT methods, **rand100 training > pm=0 training on ISIC at pm=200** "
                       "(rand100 produces better tight-bbox-trained method robustness on ISIC).",
        "cells": [
            ("decoder_only", "rand100", "isic2018_test", 200),
            ("vpt_shallow",  "rand100", "isic2018_test", 200),
            ("vpt_deep",     "rand100", "isic2018_test", 200),
            ("lora",         "rand100", "isic2018_test", 200),
            ("full_ft",      "rand100", "isic2018_test", 200),
        ],
        "baseline_cells": [
            ("decoder_only", "pm=0", "isic2018_test", 200),
            ("vpt_shallow",  "pm=0", "isic2018_test", 200),
            ("vpt_deep",     "pm=0", "isic2018_test", 200),
            ("lora",         "pm=0", "isic2018_test", 200),
            ("full_ft",      "pm=0", "isic2018_test", 200),
        ],
        "greater_is_better": True,
    },
    {
        "name": "C1b-ph2",
        "description": "Across all 5 PEFT methods, **rand100 training > pm=0 training on PH2 at pm=200**.",
        "cells": [
            ("decoder_only", "rand100", "ph2", 200),
            ("vpt_shallow",  "rand100", "ph2", 200),
            ("vpt_deep",     "rand100", "ph2", 200),
            ("lora",         "rand100", "ph2", 200),
            ("full_ft",      "rand100", "ph2", 200),
        ],
        "baseline_cells": [
            ("decoder_only", "pm=0", "ph2", 200),
            ("vpt_shallow",  "pm=0", "ph2", 200),
            ("vpt_deep",     "pm=0", "ph2", 200),
            ("lora",         "pm=0", "ph2", 200),
            ("full_ft",      "pm=0", "ph2", 200),
        ],
        "greater_is_better": True,
    },
    {
        "name": "C2-cbis",
        "description": "**Full FT > LoRA on CBIS-DDSM tight bbox (pm=0)**, "
                       "Full FT keeps far-OOD modality transfer that LoRA loses.",
        "cells":         [("full_ft", "pm=0", "cbis_ddsm", 0)],
        "baseline_cells": [("lora",    "pm=0", "cbis_ddsm", 0)],
        "greater_is_better": True,
    },
    {
        "name": "C3-cbis-zeroshot-ge-rand100",
        "description": "**Zero-shot >= rand100-trained on CBIS-DDSM at pm=200** "
                       "(averaged across 5 PEFT methods). Tests that "
                       "rand100-trained methods do NOT beat zero-shot here.",
        # Test (rand100 - zero_shot) > 0; not rejected (p>=0.05) is consistent with zero_shot >= rand100
        "cells": [
            ("decoder_only", "rand100", "cbis_ddsm", 200),
            ("vpt_shallow",  "rand100", "cbis_ddsm", 200),
            ("vpt_deep",     "rand100", "cbis_ddsm", 200),
            ("lora",         "rand100", "cbis_ddsm", 200),
            ("full_ft",      "rand100", "cbis_ddsm", 200),
        ],
        "baseline_cells": [
            ("zero_shot", "pm=0", "cbis_ddsm", 200),
            ("zero_shot", "pm=0", "cbis_ddsm", 200),
            ("zero_shot", "pm=0", "cbis_ddsm", 200),
            ("zero_shot", "pm=0", "cbis_ddsm", 200),
            ("zero_shot", "pm=0", "cbis_ddsm", 200),
        ],
        "greater_is_better": False,  # we WANT this to fail (rand100 NOT > zero_shot)
        "invert_for_pass": True,     # passes if p >= 0.05 in the greater direction
    },
]


def evaluate_claim(claim: dict) -> ClaimResult:
    # across-seed test
    diffs = []
    for cell, base in zip(claim["cells"], claim["baseline_cells"]):
        m_a, t_a, ds_a, p_a = cell
        m_b, t_b, ds_b, p_b = base
        a_means = load_per_seed_means(m_a, t_a, ds_a, p_a)
        b_means = load_per_seed_means(m_b, t_b, ds_b, p_b)
        n = min(len(a_means), len(b_means))
        for i in range(n):
            diffs.append(a_means[i] - b_means[i])

    if claim.get("invert_for_pass", False):
        # Test a - b > 0 and expect it to fail (data does not show a > b)
        seed_t, seed_p = run_paired_t_seeds(diffs, greater_is_better=True)
        pass_seed = (seed_p >= 0.05) if not np.isnan(seed_p) else None
    else:
        seed_t, seed_p = run_paired_t_seeds(diffs, greater_is_better=claim["greater_is_better"])
        pass_seed = (seed_p < 0.05) if not np.isnan(seed_p) else None

    # per-image test
    all_a, all_b = [], []
    for cell, base in zip(claim["cells"], claim["baseline_cells"]):
        if cell[3] == 0:
            continue  # tight bbox: only seed-0 per-image data exists, skip per-image
        sa = per_image_seed_avg(*cell)
        sb = per_image_seed_avg(*base)
        if sa is None or sb is None:
            continue
        common = sa.index.intersection(sb.index)
        all_a.append(sa.loc[common])
        all_b.append(sb.loc[common])

    if all_a:
        a_pool = pd.concat(all_a)
        b_pool = pd.concat(all_b)
        # duplicate image_ids across the 5 PEFT methods are fine: each row is a distinct (image, method) pair
        if claim.get("invert_for_pass", False):
            n_img, w_stat, img_p, med = run_paired_wilcoxon(a_pool, b_pool, greater_is_better=True)
            pass_image = (img_p >= 0.05) if not np.isnan(img_p) else None
        else:
            n_img, w_stat, img_p, med = run_paired_wilcoxon(a_pool, b_pool,
                                                            greater_is_better=claim["greater_is_better"])
            pass_image = (img_p < 0.05) if not np.isnan(img_p) else None
    else:
        n_img, w_stat, img_p, med, pass_image = None, None, None, None, None

    return ClaimResult(
        name=claim["name"],
        description=claim["description"],
        seed_diffs=diffs,
        seed_t=seed_t,
        seed_p_one_sided=seed_p,
        image_n=n_img,
        image_w=w_stat,
        image_p_one_sided=img_p,
        image_median_diff=med,
        pass_seed=pass_seed,
        pass_image=pass_image,
    )


def write_markdown(results: list[ClaimResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Statistical reliability of headline claims",
        "",
        "All claims are tested **two ways**:",
        "",
        "  * **(A) Across-seed paired t-test** on per-seed dice means "
        "(n = number of (method, seed) pairs contributing to the claim). "
        "Tests one-sided H1: mean (a - b) > 0 (or < 0 where the claim is a "
        "negative). Reported as `t_stat`, `p_one_sided`, and the empirical "
        "mean +/- std of the diffs.",
        "",
        "  * **(B) Per-image paired Wilcoxon signed-rank** on dice values "
        "averaged across 3 seeds. n = number of unique (image, method) pairs. "
        "This test has high power; the seed-level test is the conservative "
        "n=3 sanity check.",
        "",
        "A claim **passes** when the relevant test rejects the null at p < 0.05. "
        "For claim C3 (a negative claim: rand100 does NOT beat zero-shot on "
        "CBIS-DDSM), it passes when the test in the opposite direction fails "
        "to reject (p >= 0.05).",
        "",
        "| Claim | seed n | mean+/-std diff | seed t | seed p (1-sided) | seed pass | img n | Wilcoxon W | img p | img median diff | img pass |",
        "|---|---:|---|---:|---:|:--:|---:|---:|---:|---:|:--:|",
    ]
    for r in results:
        diffs_arr = np.asarray(r.seed_diffs, dtype=float)
        mean_diff = diffs_arr.mean() if diffs_arr.size else float("nan")
        std_diff = diffs_arr.std(ddof=1) if diffs_arr.size > 1 else float("nan")
        ms = f"{mean_diff:+.4f} +/- {std_diff:.4f}" if not np.isnan(mean_diff) else "-"
        seed_t = f"{r.seed_t:+.3f}" if r.seed_t is not None and not np.isnan(r.seed_t) else "-"
        seed_p = f"{r.seed_p_one_sided:.4f}" if r.seed_p_one_sided is not None and not np.isnan(r.seed_p_one_sided) else "-"
        seed_ok = "PASS" if r.pass_seed else ("FAIL" if r.pass_seed is False else "-")
        if r.image_n is not None:
            img_n = str(r.image_n)
            img_w = f"{r.image_w:.0f}" if r.image_w is not None and not np.isnan(r.image_w) else "-"
            img_p = f"{r.image_p_one_sided:.2e}" if r.image_p_one_sided is not None and not np.isnan(r.image_p_one_sided) else "-"
            img_md = f"{r.image_median_diff:+.4f}" if r.image_median_diff is not None else "-"
            img_ok = "PASS" if r.pass_image else ("FAIL" if r.pass_image is False else "-")
        else:
            img_n = img_w = img_p = img_md = img_ok = "-"
        lines.append(f"| **{r.name}** | {len(r.seed_diffs)} | {ms} | {seed_t} | {seed_p} | {seed_ok} | {img_n} | {img_w} | {img_p} | {img_md} | {img_ok} |")
    lines.append("")
    lines.append("### Claim descriptions")
    lines.append("")
    for r in results:
        lines.append(f"- **{r.name}**: {r.description}")
    lines.append("")
    lines.append("### Reading the table")
    lines.append("")
    lines.append("- For C1a/C1b/C2: we expect both `seed pass` and `img pass` to be PASS.")
    lines.append("- For C3: we expect both to be PASS, meaning the data does NOT support "
                 "'rand100 > zero_shot on CBIS-DDSM at pm=200', consistent with the "
                 "claim 'zero_shot >= rand100'.")
    lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[significance] wrote {out_path}")


def write_csv(results: list[ClaimResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["claim", "n_seeds_diffs", "mean_diff", "std_diff",
                    "seed_t", "seed_p_one_sided", "seed_pass",
                    "image_n", "image_w", "image_p_one_sided",
                    "image_median_diff", "image_pass"])
        for r in results:
            diffs_arr = np.asarray(r.seed_diffs, dtype=float)
            mean_diff = diffs_arr.mean() if diffs_arr.size else ""
            std_diff = diffs_arr.std(ddof=1) if diffs_arr.size > 1 else ""
            w.writerow([
                r.name, len(r.seed_diffs), mean_diff, std_diff,
                r.seed_t, r.seed_p_one_sided, r.pass_seed,
                r.image_n, r.image_w, r.image_p_one_sided,
                r.image_median_diff, r.pass_image,
            ])
    print(f"[significance] wrote {out_path}")


def main() -> int:
    print(f"[significance] running paired tests on {len(CLAIM_DEFINITIONS)} claims")
    results = [evaluate_claim(c) for c in CLAIM_DEFINITIONS]
    for r in results:
        diffs = np.asarray(r.seed_diffs, dtype=float)
        print(f"  {r.name}: n_diffs={len(r.seed_diffs)}, "
              f"mean_diff={diffs.mean():+.4f}, "
              f"seed_t={r.seed_t}, seed_p={r.seed_p_one_sided}, "
              f"image_n={r.image_n}, image_p={r.image_p_one_sided}, "
              f"pass_seed={r.pass_seed}, pass_image={r.pass_image}")
    write_markdown(results, OUT_MD)
    write_csv(results, OUT_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
