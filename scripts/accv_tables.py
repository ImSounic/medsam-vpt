"""Emit the paper's LaTeX tables from the result CSVs (multi-seed CKA table, detector AUROC table)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASET_LABEL = {
    "isic2018_test": "ISIC (ID)",
    "ph2": "PH2 (near-OOD)",
    "busi": "BUSI (far-OOD)",
    "cbis_ddsm": "CBIS-DDSM (far-OOD)",
    "dmid": "DMID (held-out)",
}
DATASET_ORDER = ["isic2018_test", "ph2", "busi", "cbis_ddsm"]
METHOD_LABEL = {
    "zero_shot": "Zero-shot",
    "decoder_only_seed0": "Decoder-only FT",
    "vpt_shallow_seed0": "VPT-shallow",
    "vpt_deep_seed0": "VPT-deep",
    "lora_seed0": "LoRA",
    "lora_encoder_only_r28_all_seed0": "Encoder-only LoRA",
    "full_ft_seed0": "Full FT",
}
METHOD_ORDER = list(METHOD_LABEL)


def _p(p: float) -> str:
    if p != p:
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def multiseed_table_tex(
    seed_table_csv: Path, wilcoxon_csv: Path, level: int = 20
) -> str:
    seed = pd.read_csv(seed_table_csv).set_index("dataset")
    wil = pd.read_csv(wilcoxon_csv)
    wil = wil[wil.level == level].set_index("dataset")
    lines = [
        r"\begin{tabular}{lcccrr}",
        r"\toprule",
        r"Dataset & no-CKA Dice & CKA Dice & $\Delta$Dice & $\Delta$HD95 (px) & $p$ (Holm) \\",
        r"\midrule",
    ]
    for ds in DATASET_ORDER:
        if ds not in seed.index:
            continue
        r = seed.loc[ds]
        p = _p(float(wil.loc[ds, "p_holm"])) if ds in wil.index else "--"
        dh = float(r["cka_hd95_mean"]) - float(r["no_cka_hd95_mean"])
        lines.append(
            f"{DATASET_LABEL.get(ds, ds)} & "
            f"{r['no_cka_dice_mean']:.3f} $\\pm$ {r['no_cka_dice_std']:.3f} & "
            f"{r['cka_dice_mean']:.3f} $\\pm$ {r['cka_dice_std']:.3f} & "
            f"{r['delta_dice']:+.3f} & {dh:+.1f} & {p} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _read_metrics(csvs) -> pd.DataFrame:
    """Concatenate detector_metrics.csv files (e.g. the ladder run and the DMID run)."""
    paths = [csvs] if isinstance(csvs, (str, Path)) else list(csvs)
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def detector_table_tex(
    metrics_csv,
    threshold: float = 0.5,
    datasets=("busi", "cbis_ddsm"),
    min_fail: int = 20,
    with_auprc: bool = False,
) -> str:
    """metrics_csv: one detector_metrics.csv or a list of them (merged)."""
    m = _read_metrics(metrics_csv)
    m = m[m.threshold == threshold]
    cols = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{{DATASET_LABEL.get(d, d)}}}" for d in datasets
    )
    sub = " & ".join("drift & IoU head" for _ in datasets)
    lines = [
        r"\begin{tabular}{l" + "cc" * len(datasets) + "}",
        r"\toprule",
        f"Method & {cols} \\\\",
        f" & {sub} \\\\",
        r"\midrule",
    ]
    runs = [r for r in METHOD_ORDER if r in set(m.run_name)]
    for run in runs:
        cells = []
        for ds in datasets:
            for det in ("drift", "iou_pred"):
                cell = m[(m.run_name == run) & (m.dataset == ds) & (m.detector == det)]
                if (
                    len(cell) == 0
                    or int(cell.n_fail.iloc[0]) < min_fail
                    or run == "zero_shot"
                    and det == "drift"
                ):
                    cells.append("--")
                elif with_auprc:
                    cells.append(
                        f"{float(cell.auroc.iloc[0]):.2f} / {float(cell.auprc.iloc[0]):.2f}"
                    )
                else:
                    cells.append(f"{float(cell.auroc.iloc[0]):.2f}")
        lines.append(f"{METHOD_LABEL.get(run, run)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def threshold_sweep_table_tex(
    metrics_csv,
    runs=("lora_seed0", "vpt_shallow_seed0", "vpt_deep_seed0"),
    datasets=("busi", "cbis_ddsm"),
    detector: str = "drift",
) -> str:
    """Supplement: AUROC of one detector per (method, dataset) at every failure threshold."""
    m = _read_metrics(metrics_csv)
    m = m[(m.detector == detector) & m.run_name.isin(runs) & m.dataset.isin(datasets)]
    thresholds = sorted(m.threshold.unique())
    lines = [
        r"\begin{tabular}{ll" + "c" * len(thresholds) + "}",
        r"\toprule",
        "Method & Dataset & " + " & ".join(f"{t:.1f}" for t in thresholds) + r" \\",
        r"\midrule",
    ]
    for run in [r for r in METHOD_ORDER if r in runs]:
        for ds in datasets:
            cells = []
            for t in thresholds:
                cell = m[(m.run_name == run) & (m.dataset == ds) & (m.threshold == t)]
                cells.append(f"{float(cell.auroc.iloc[0]):.2f}" if len(cell) else "--")
            lines.append(
                f"{METHOD_LABEL.get(run, run)} & {DATASET_LABEL.get(ds, ds)} & "
                + " & ".join(cells)
                + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def risk_coverage_table_tex(
    points_csv: Path,
    runs=("lora_seed0", "vpt_shallow_seed0", "vpt_deep_seed0"),
    datasets=("busi", "cbis_ddsm"),
    detector: str = "drift",
) -> str:
    """Supplement: base failure rate, AURC, and recall / residual failure rate when the top c images by drift are flagged."""
    p = pd.read_csv(points_csv)
    p = p[(p.detector == detector) & p.run_name.isin(runs) & p.dataset.isin(datasets)]
    coverages = sorted(p.flagged.unique())
    lines = [
        r"\begin{tabular}{llcc" + "c" * len(coverages) + "}",
        r"\toprule",
        "Method & Dataset & Base rate & AURC & "
        + " & ".join(f"top {c:.0%}".replace("%", r"\%") for c in coverages)
        + r" \\",
        r"\midrule",
    ]
    for run in [r for r in METHOD_ORDER if r in runs]:
        for ds in datasets:
            g = p[(p.run_name == run) & (p.dataset == ds)]
            if not len(g):
                continue
            cells = []
            for c in coverages:
                row = g[g.flagged == c]
                cells.append(
                    f"{float(row.recall.iloc[0]):.2f} / {float(row.residual_risk.iloc[0]):.3f}"
                    if len(row)
                    else "--"
                )
            lines.append(
                f"{METHOD_LABEL.get(run, run)} & {DATASET_LABEL.get(ds, ds)} & "
                f"{float(g.base_rate.iloc[0]):.3f} & {float(g.aurc.iloc[0]):.3f} & "
                + " & ".join(cells)
                + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def robustness_levels_table_tex(
    curves_csv: Path, datasets=("isic2018_test", "ph2", "busi", "cbis_ddsm")
) -> str:
    """Supplement: Dice mean +- std over seeds per (curve, dataset) at every box-expansion level."""
    c = pd.read_csv(curves_csv)
    levels = sorted(c.level.unique())
    lines = [
        r"\begin{tabular}{ll" + "c" * len(levels) + "}",
        r"\toprule",
        "Model & Dataset & " + " & ".join(f"{int(l)}\\,px" for l in levels) + r" \\",
        r"\midrule",
    ]
    for curve in list(dict.fromkeys(c.curve)):
        for ds in datasets:
            g = c[(c.curve == curve) & (c.dataset == ds)]
            if not len(g):
                continue
            cells = []
            for l in levels:
                r = g[g.level == l]
                if not len(r):
                    cells.append("--")
                    continue
                m, sd, n = (
                    float(r.dice_mean.iloc[0]),
                    float(r.dice_std.iloc[0]),
                    int(r.n_seeds.iloc[0]),
                )
                cells.append(
                    f"{m:.3f} $\\pm$ {sd:.3f}" if n > 1 and sd == sd else f"{m:.3f}"
                )
            lines.append(
                f"{curve} & {DATASET_LABEL.get(ds, ds)} & " + " & ".join(cells) + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def wilcoxon_levels_table_tex(
    wilcoxon_csv: Path, datasets=("isic2018_test", "ph2", "busi", "cbis_ddsm")
) -> str:
    """Supplement: paired Wilcoxon (CKA minus no-CKA, pm20 training) per dataset and expansion level."""
    w = pd.read_csv(wilcoxon_csv)
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Dataset & Level (px) & Pairs & Mean $\Delta$ & Median $\Delta$ & Improved & $p$ (Holm) \\",
        r"\midrule",
    ]
    for ds in datasets:
        for _, r in w[w.dataset == ds].sort_values("level").iterrows():
            lines.append(
                f"{DATASET_LABEL.get(ds, ds)} & {int(r.level)} & {int(r.n_pairs)} & "
                f"{r.mean_delta:+.3f} & {r.median_delta:+.3f} & {100 * r.frac_improved:.0f}\\% & {_p(r.p_holm)} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def calibration_table_tex(calibration_csv: Path, datasets=("busi", "cbis_ddsm")) -> str:
    """Supplement: ECE of the IoU estimate, its mean, and the true mean IoU per method and far-OOD set."""
    c = pd.read_csv(calibration_csv)
    cols = " & ".join(
        f"\\multicolumn{{3}}{{c}}{{{DATASET_LABEL.get(d, d)}}}" for d in datasets
    )
    sub = " & ".join("ECE & $\\hat{u}$ & IoU" for _ in datasets)
    lines = [
        r"\begin{tabular}{l" + "ccc" * len(datasets) + "}",
        r"\toprule",
        f"Method & {cols} \\\\",
        f" & {sub} \\\\",
        r"\midrule",
    ]
    for run in [r for r in METHOD_ORDER if r in set(c.run_name)]:
        cells = []
        for ds in datasets:
            r = c[(c.run_name == run) & (c.dataset == ds)]
            if len(r):
                r = r.iloc[0]
                cells += [f"{r.ece:.2f}", f"{r.iou_pred_mean:.2f}", f"{r.iou_mean:.2f}"]
            else:
                cells += ["--"] * 3
        lines.append(f"{METHOD_LABEL.get(run, run)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


CKA_LABEL = [
    (
        r"^lora_cka_oodonly_late_l10_seed\d+$",
        "LoRA + CKA (decoder hooks, tight training)",
    ),
    (r"^lora_cka_oodonly_late_l10_pm20_seed\d+$", "LoRA + CKA (decoder hooks, 20 px)"),
    (r"^lora_cka_oodonly_enc_l10_pm20_seed\d+$", "LoRA + CKA (encoder hooks, 20 px)"),
    (r"^lora_cka_oodonly_both_l10_pm20_seed\d+$", "LoRA + CKA (both, 20 px)"),
]


def dmid_table_tex(runs_csv: Path) -> str:
    """DMID held-out set: Dice and HD95 per method (seed 0) and per CKA variant (mean over available seeds)."""
    import re

    df = pd.read_csv(runs_csv)
    df = df[df.dataset == "dmid"]
    rows = []
    for run in METHOD_ORDER:
        sub = df[df.run_name == run]
        if len(sub):
            rows.append(
                (
                    METHOD_LABEL[run],
                    sub.dice_mean.mean(),
                    sub.hd95_mean.mean(),
                    len(sub),
                )
            )
    for pattern, label in CKA_LABEL:
        sub = df[df.run_name.str.match(pattern)]
        if len(sub):
            rows.append((label, sub.dice_mean.mean(), sub.hd95_mean.mean(), len(sub)))
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & Dice & HD95 (px) \\",
        r"\midrule",
    ]
    for label, dice, hd, n in rows:
        seeds = f" ($n={n}$)" if n > 1 else ""
        lines.append(f"{label}{seeds} & {dice:.3f} & {hd:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "paper/tables")
    ap.add_argument(
        "--multiseed-dir", type=Path, default=REPO_ROOT / "results/accv/cka_multiseed"
    )
    ap.add_argument(
        "--detector-csv",
        type=Path,
        default=REPO_ROOT / "results/accv/failure_detection_pm50/detector_metrics.csv",
    )
    ap.add_argument(
        "--detector-csv-tight",
        type=Path,
        default=REPO_ROOT / "results/accv/failure_detection_pm0/detector_metrics.csv",
    )
    ap.add_argument(
        "--detector-csv-dmid",
        type=Path,
        default=REPO_ROOT / "results/accv/failure_detection_dmid/detector_metrics.csv",
    )
    ap.add_argument(
        "--sweep-csv",
        type=Path,
        default=REPO_ROOT
        / "results/accv/failure_detection_pm0_sweep/detector_metrics.csv",
    )
    ap.add_argument(
        "--sweep-csv-pm50",
        type=Path,
        default=REPO_ROOT
        / "results/accv/failure_detection_pm50_sweep/detector_metrics.csv",
    )
    ap.add_argument(
        "--risk-coverage-csv",
        type=Path,
        default=REPO_ROOT / "results/accv/risk_coverage_pm0/review_points.csv",
    )
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cka_multiseed.tex").write_text(
        multiseed_table_tex(
            args.multiseed_dir / "seed_table.csv",
            args.multiseed_dir / "paired_wilcoxon.csv",
        )
    )
    (args.out_dir / "detectors_pm50.tex").write_text(
        detector_table_tex(args.detector_csv, with_auprc=True)
    )
    dmid_csv = REPO_ROOT / "results/accv/runs_dmid.csv"
    if dmid_csv.exists():
        (args.out_dir / "dmid.tex").write_text(dmid_table_tex(dmid_csv))
    if args.detector_csv_tight.exists():
        # Main-text table: tight boxes, AUROC / AUPRC, with the held-out DMID column.
        csvs = [args.detector_csv_tight]
        datasets = ["busi", "cbis_ddsm"]
        if args.detector_csv_dmid.exists():
            csvs.append(args.detector_csv_dmid)
            datasets.append("dmid")
        (args.out_dir / "detectors_pm0.tex").write_text(
            detector_table_tex(csvs, datasets=datasets, with_auprc=True)
        )
    if args.sweep_csv.exists():
        (args.out_dir / "supp_threshold_sweep_pm0.tex").write_text(
            threshold_sweep_table_tex(args.sweep_csv)
        )
    if args.sweep_csv_pm50.exists():
        (args.out_dir / "supp_threshold_sweep_pm50.tex").write_text(
            threshold_sweep_table_tex(args.sweep_csv_pm50)
        )
    if args.risk_coverage_csv.exists():
        (args.out_dir / "supp_risk_coverage_pm0.tex").write_text(
            risk_coverage_table_tex(args.risk_coverage_csv)
        )
    curves = args.multiseed_dir / "robustness_curves.csv"
    if curves.exists():
        (args.out_dir / "supp_robustness_levels.tex").write_text(
            robustness_levels_table_tex(curves)
        )
    wilcoxon = args.multiseed_dir / "paired_wilcoxon.csv"
    if wilcoxon.exists():
        (args.out_dir / "supp_wilcoxon_levels.tex").write_text(
            wilcoxon_levels_table_tex(wilcoxon)
        )
    for tag, csv in [
        ("pm0", args.detector_csv_tight),
        ("pm50", args.detector_csv),
    ]:
        cal = csv.with_name("calibration.csv")
        if cal.exists():
            (args.out_dir / f"supp_calibration_{tag}.tex").write_text(
                calibration_table_tex(cal)
            )
    for p in sorted(args.out_dir.glob("*.tex")):
        print(f"[tables] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
