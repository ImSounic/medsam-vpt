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


def detector_table_tex(
    metrics_csv: Path,
    threshold: float = 0.5,
    datasets=("busi", "cbis_ddsm"),
    min_fail: int = 20,
    with_auprc: bool = False,
) -> str:
    m = pd.read_csv(metrics_csv)
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
    if args.detector_csv_tight.exists():
        (args.out_dir / "detectors_pm0.tex").write_text(
            detector_table_tex(args.detector_csv_tight)
        )
    for p in sorted(args.out_dir.glob("*.tex")):
        print(f"[tables] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
