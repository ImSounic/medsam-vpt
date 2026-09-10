"""Regenerate camera-ready tables, figures, and statistics from committed seed CSVs.

Produces (no GPU / no HPC needed, all from existing eval outputs):
  results/summary_full.csv                     rebuilt wide dice/iou/hd95 x 5 jitter (item 7)
  results/camera_ready/boundary_metrics.csv    HD95 + IoU tight-box table, seed mean+/-std (item 1)
  results/camera_ready/boundary_metrics.md
  results/camera_ready/method_significance.md  pairwise Wilcoxon per dataset, Holm-corrected (item 3)
  results/camera_ready/deployment.md           method trade-off recommendation table (item 4)
  figures/camera_ready/hd95_robustness.png     HD95 vs jitter curves (item 2)
  figures/camera_ready/iou_robustness.png      IoU vs jitter curves (item 2)

Single-seed (seed 0) drives the per-image stats and robustness curves; the tight-box
boundary table additionally aggregates seeds 0/1/2 where available.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "results" / "raw"
OUT = REPO / "results" / "camera_ready"
FIG = REPO / "figures" / "camera_ready"

METHODS = ["zero_shot", "decoder_only", "vpt_shallow", "vpt_deep", "lora", "full_ft"]
DATASETS = ["isic2018_test", "ph2", "busi", "cbis_ddsm"]
TRAININGS = ["pm=0", "pm=20", "rand100"]
PERTURBS = [0, 20, 50, 100, 200]

METHOD_LABEL = {
    "zero_shot": "Zero-shot",
    "decoder_only": "Decoder-only FT",
    "vpt_shallow": "VPT-shallow",
    "vpt_deep": "VPT-deep",
    "lora": "LoRA",
    "full_ft": "Full FT",
}
DATASET_LABEL = {
    "isic2018_test": "ISIC 2018 (ID)",
    "ph2": "PH2 (near-OOD)",
    "busi": "BUSI (far-OOD US)",
    "cbis_ddsm": "CBIS-DDSM (far-OOD X-ray)",
}
TRAIN_SUFFIX = {"pm=0": "", "pm=20": "_pm20", "rand100": "_rand100"}


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
def tight_runs(training: str, seed: int) -> Path:
    suf = TRAIN_SUFFIX[training]
    if seed == 0:
        return REPO / "results" / (f"runs{suf}.csv" if suf else "runs.csv")
    return REPO / "results" / f"runs_seed{seed}{suf}.csv"


def bbox_runs(training: str, seed: int) -> Path:
    suf = TRAIN_SUFFIX[training]
    if seed == 0:
        return REPO / "bbox_robustness" / (f"results{suf}" if suf else "results") / "runs.csv"
    return REPO / "bbox_robustness" / f"results_seed{seed}{suf}" / "runs.csv"


# --------------------------------------------------------------------------- #
# Item 7: rebuild wide summary_full.csv (seed 0, all metrics, all jitter)
# --------------------------------------------------------------------------- #
def rebuild_summary_full() -> pd.DataFrame:
    rows = []
    for training in TRAININGS:
        long = {}  # (method, dataset, perturb) -> {metric: val}
        tp = tight_runs(training, 0)
        if tp.exists():
            for _, r in pd.read_csv(tp).iterrows():
                long[(r["method"], r["dataset"], 0)] = {
                    "dice": r["dice_mean"], "iou": r["iou_mean"], "hd95": r["hd95_mean"],
                }
        bp = bbox_runs(training, 0)
        if bp.exists():
            for _, r in pd.read_csv(bp).iterrows():
                long[(r["method"], r["dataset"], int(r["perturb_max_px"]))] = {
                    "dice": r["dice_mean"], "iou": r["iou_mean"], "hd95": r["hd95_mean"],
                }
        for method in METHODS:
            for dataset in DATASETS:
                if not any((method, dataset, p) in long for p in PERTURBS):
                    continue
                row = {"dataset": dataset, "method": method, "training": training}
                for metric in ("dice", "iou", "hd95"):
                    for p in PERTURBS:
                        cell = long.get((method, dataset, p))
                        row[f"{metric}_pm{p}"] = (
                            round(float(cell[metric]), 4) if cell else np.nan
                        )
                rows.append(row)
    df = pd.DataFrame(rows)
    out = REPO / "results" / "summary_full.csv"
    df.to_csv(out, index=False)
    print(f"[item7] wrote {out}  ({len(df)} rows)")
    return df


# --------------------------------------------------------------------------- #
# Item 1: boundary-metric table (tight box, seed mean+/-std + per-image median)
# --------------------------------------------------------------------------- #
def per_image(method: str, dataset: str) -> pd.DataFrame:
    # zero_shot has no checkpoint/seed, so its per-image files omit the seed tag
    stem = f"{method}_{dataset}" if method == "zero_shot" else f"{method}_seed0_{dataset}"
    return pd.read_csv(RAW / f"{stem}_per_image.csv")


def boundary_metrics() -> pd.DataFrame:
    seed_runs = {s: pd.read_csv(tight_runs("pm=0", s)) for s in (0, 1, 2)
                 if tight_runs("pm=0", s).exists()}
    rows = []
    for method in METHODS:
        for dataset in DATASETS:
            # seed-level mean +/- std of the per-run means
            hd95_seed, iou_seed = [], []
            for df in seed_runs.values():
                sel = df[(df["method"] == method) & (df["dataset"] == dataset)]
                if not sel.empty:
                    hd95_seed.append(float(sel["hd95_mean"].iloc[0]))
                    iou_seed.append(float(sel["iou_mean"].iloc[0]))
            # per-image distribution (seed 0) for median / IQR (HD95 is heavy-tailed)
            pi = per_image(method, dataset)
            hd95_med = float(pi["hd95"].median())
            hd95_q1, hd95_q3 = np.percentile(pi["hd95"], [25, 75])
            rows.append({
                "method": METHOD_LABEL[method],
                "dataset": DATASET_LABEL[dataset],
                "n_seeds": len(hd95_seed),
                "iou_mean": round(np.mean(iou_seed), 4),
                "iou_std": round(np.std(iou_seed, ddof=1) if len(iou_seed) > 1 else 0.0, 4),
                "hd95_mean": round(np.mean(hd95_seed), 2),
                "hd95_std": round(np.std(hd95_seed, ddof=1) if len(hd95_seed) > 1 else 0.0, 2),
                "hd95_median": round(hd95_med, 2),
                "hd95_iqr": f"[{hd95_q1:.2f}, {hd95_q3:.2f}]",
            })
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "boundary_metrics.csv", index=False)

    lines = ["# Boundary metrics (tight box, pm=0)\n",
             "HD95 in pixels at 1024x1024. Mean +/- std across seeds; median [IQR] over "
             "per-image scores (seed 0). HD95 is heavy-tailed, so the median is the more "
             "reliable central estimate.\n"]
    for ds in DATASETS:
        lbl = DATASET_LABEL[ds]
        sub = df[df["dataset"] == lbl]
        lines.append(f"\n## {lbl}\n")
        lines.append("| Method | IoU (mean +/- std) | HD95 mean +/- std | HD95 median [IQR] |")
        lines.append("|---|---:|---:|---:|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['method']} | {r['iou_mean']:.4f} +/- {r['iou_std']:.4f} "
                f"| {r['hd95_mean']:.2f} +/- {r['hd95_std']:.2f} | "
                f"{r['hd95_median']:.2f} {r['hd95_iqr']} |"
            )
    (OUT / "boundary_metrics.md").write_text("\n".join(lines))
    print(f"[item1] wrote {OUT/'boundary_metrics.csv'} and .md")
    return df


# --------------------------------------------------------------------------- #
# Item 3: pairwise Wilcoxon between methods per dataset (Holm-corrected)
# --------------------------------------------------------------------------- #
def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def method_significance() -> None:
    lines = ["# Pairwise method comparisons (paired Wilcoxon signed-rank, per-image Dice)\n",
             "Tight box (pm=0), seed 0. Paired over the same images. p-values Holm-corrected "
             "within each dataset. Effect = median(Dice_A - Dice_B); positive means A > B.\n"]
    for ds in DATASETS:
        pi = {m: per_image(m, ds).set_index("image_id")["dice"] for m in METHODS}
        pairs = list(itertools.combinations(METHODS, 2))
        raw_p, recs = [], []
        for a, b in pairs:
            common = pi[a].index.intersection(pi[b].index)
            da, db = pi[a].loc[common], pi[b].loc[common]
            diff = da - db
            if np.allclose(diff, 0):
                stat, p = np.nan, 1.0
            else:
                stat, p = stats.wilcoxon(da, db)
            raw_p.append(p)
            recs.append((a, b, float(np.median(diff))))
        adj = holm(raw_p)
        lines.append(f"\n## {DATASET_LABEL[ds]}\n")
        lines.append("| A vs B | median Dice diff (A-B) | p (Holm) | sig |")
        lines.append("|---|---:|---:|:--:|")
        for (a, b, eff), p in sorted(zip(recs, adj), key=lambda x: x[1]):
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            lines.append(
                f"| {METHOD_LABEL[a]} vs {METHOD_LABEL[b]} | {eff:+.4f} | {p:.3g} | {star} |"
            )
    (OUT / "method_significance.md").write_text("\n".join(lines))
    print(f"[item3] wrote {OUT/'method_significance.md'}")


# --------------------------------------------------------------------------- #
# Item 4: deployment recommendation table
# --------------------------------------------------------------------------- #
def deployment() -> None:
    st = pd.read_csv(REPO / "results" / "summary_table.csv")
    st["method"] = st["method"].str.strip()

    def val(method, col):
        row = st[st["method"] == method]
        return row[col].iloc[0] if not row.empty else "-"

    order = ["Zero-shot", "Decoder-only FT", "VPT-shallow", "VPT-deep", "LoRA", "Full FT"]
    lines = [
        "# Deployment recommendation (tight box, multi-seed)\n",
        "Trade-off view: parameter cost vs in-domain vs far-OOD. Far-OOD = mean of BUSI "
        "and CBIS-DDSM Dice. ID = ISIC 2018 test Dice.\n",
        "| Method | Trainable params | ID Dice | Far-OOD Dice | When to use |",
        "|---|---:|---:|---:|---|",
    ]
    when = {
        "Zero-shot": "Baseline; still competitive under severe far-OOD prompt jitter.",
        "Decoder-only FT": "Cheap, robust far-OOD; safe default when compute is limited.",
        "VPT-shallow": "Not recommended; collapses far-OOD.",
        "VPT-deep": "Not recommended; collapses far-OOD.",
        "LoRA": "Best ID/near-OOD; avoid if far-OOD transfer matters.",
        "Full FT": "Best overall incl. far-OOD, but 93.7M params / most expensive.",
    }
    for m in order:
        params = val(m, "trainable_params")
        idd = str(val(m, "isic2018_test_dice")).split("+/-")[0].split("±")[0].strip()
        busi = float(str(val(m, "busi_dice")).split("±")[0].strip())
        cbis = float(str(val(m, "cbis_ddsm_dice")).split("±")[0].strip())
        far = (busi + cbis) / 2
        pstr = f"{int(params):,}" if str(params).isdigit() else params
        lines.append(f"| {m} | {pstr} | {idd} | {far:.4f} | {when[m]} |")
    (OUT / "deployment.md").write_text("\n".join(lines))
    print(f"[item4] wrote {OUT/'deployment.md'}")


# --------------------------------------------------------------------------- #
# Item 2: HD95 and IoU robustness curves
# --------------------------------------------------------------------------- #
def robustness_curves(summary: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    # rand100-trained models are the paper's Fig.2 regime (variable training jitter)
    sub = summary[summary["training"] == "rand100"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(METHODS)))
    cmap = dict(zip(METHODS, colors))

    for metric, fname, ylabel, logy in [
        ("hd95", "hd95_robustness.png", "HD95 (px)", True),
        ("iou", "iou_robustness.png", "IoU", False),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, ds in zip(axes.ravel(), DATASETS):
            for method in METHODS:
                r = sub[(sub["method"] == method) & (sub["dataset"] == ds)]
                if r.empty:
                    continue
                ys = [r[f"{metric}_pm{p}"].iloc[0] for p in PERTURBS]
                ax.plot(PERTURBS, ys, marker="o", ms=4, color=cmap[method],
                        label=METHOD_LABEL[method])
            if logy:
                ax.set_yscale("log")
            ax.set_title(DATASET_LABEL[ds], fontsize=10)
            ax.set_xlabel("Evaluation jitter (px)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
        axes.ravel()[0].legend(fontsize=7, ncol=2)
        fig.suptitle(f"{ylabel} vs bounding-box jitter (rand100-trained)", fontsize=12)
        fig.tight_layout()
        fig.savefig(FIG / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[item2] wrote {FIG/fname}")


def main() -> None:
    summary = rebuild_summary_full()
    boundary_metrics()
    method_significance()
    deployment()
    robustness_curves(summary)
    print("\n[camera-ready] done.")


if __name__ == "__main__":
    main()
