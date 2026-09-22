"""LaTeX tables for the paper are generated from the result CSVs."""

import pandas as pd

from scripts.accv_tables import (
    calibration_table_tex,
    detector_table_tex,
    multiseed_table_tex,
    risk_coverage_table_tex,
    robustness_levels_table_tex,
    threshold_sweep_table_tex,
    wilcoxon_levels_table_tex,
)


def test_multiseed_table_has_rows_and_pvalues(tmp_path):
    seed = tmp_path / "seed_table.csv"
    pd.DataFrame(
        [
            {
                "dataset": "busi",
                "cka_n": 3,
                "no_cka_n": 3,
                "cka_dice_mean": 0.8258,
                "cka_dice_std": 0.0075,
                "no_cka_dice_mean": 0.7751,
                "no_cka_dice_std": 0.0320,
                "delta_dice": 0.0513,
                "cka_hd95_mean": 41.0,
                "no_cka_hd95_mean": 137.0,
            },
        ]
    ).to_csv(seed, index=False)
    wil = tmp_path / "paired_wilcoxon.csv"
    pd.DataFrame(
        [
            {
                "dataset": "busi",
                "level": 20,
                "n_pairs": 1941,
                "p_holm": 3e-7,
                "median_delta": 0.0115,
            }
        ]
    ).to_csv(wil, index=False)
    tex = multiseed_table_tex(seed, wil, level=20)
    assert r"\begin{tabular}" in tex
    assert "BUSI" in tex and "0.826" in tex and "0.775" in tex and "+0.051" in tex
    assert "$<$0.001" in tex


def test_detector_table_marks_low_failure_counts(tmp_path):
    m = tmp_path / "detector_metrics.csv"
    rows = []
    for run, ds, det, auroc, nfail in [
        ("lora_seed0", "busi", "drift", 0.912, 120),
        ("lora_seed0", "busi", "iou_pred", 0.692, 120),
        ("lora_seed0", "cbis_ddsm", "drift", 0.757, 270),
        ("lora_seed0", "cbis_ddsm", "iou_pred", 0.444, 270),
        ("full_ft_seed0", "busi", "drift", 0.227, 3),
        ("full_ft_seed0", "busi", "iou_pred", 0.868, 3),
    ]:
        rows.append(
            {
                "run_name": run,
                "dataset": ds,
                "detector": det,
                "threshold": 0.5,
                "auroc": auroc,
                "auprc": 0.5,
                "n": 647,
                "n_fail": nfail,
            }
        )
    pd.DataFrame(rows).to_csv(m, index=False)
    tex = detector_table_tex(m, threshold=0.5, min_fail=20)
    assert "LoRA" in tex and "0.91" in tex and "0.44" in tex
    assert "Full FT" in tex and "0.23" not in tex.split("Full FT")[1].split("\\\\")[
        0
    ].replace(
        "--", ""
    )  # low-count cells blanked
    assert "--" in tex


def test_detector_table_can_show_auprc(tmp_path):
    m = tmp_path / "detector_metrics.csv"
    pd.DataFrame(
        [
            {
                "run_name": "lora_seed0",
                "dataset": "busi",
                "detector": "drift",
                "threshold": 0.5,
                "auroc": 0.912,
                "auprc": 0.747,
                "n": 647,
                "n_fail": 120,
            },
            {
                "run_name": "lora_seed0",
                "dataset": "busi",
                "detector": "iou_pred",
                "threshold": 0.5,
                "auroc": 0.692,
                "auprc": 0.189,
                "n": 647,
                "n_fail": 120,
            },
            {
                "run_name": "lora_seed0",
                "dataset": "cbis_ddsm",
                "detector": "drift",
                "threshold": 0.5,
                "auroc": 0.757,
                "auprc": 0.831,
                "n": 362,
                "n_fail": 270,
            },
            {
                "run_name": "lora_seed0",
                "dataset": "cbis_ddsm",
                "detector": "iou_pred",
                "threshold": 0.5,
                "auroc": 0.444,
                "auprc": 0.405,
                "n": 362,
                "n_fail": 270,
            },
        ]
    ).to_csv(m, index=False)
    tex = detector_table_tex(m, threshold=0.5, with_auprc=True)
    assert "0.91 / 0.75" in tex and "0.44 / 0.41" in tex


def test_dmid_table_lists_methods_in_order(tmp_path):
    from scripts.accv_tables import dmid_table_tex

    m = tmp_path / "runs_dmid.csv"
    rows = [
        ("zero_shot", "dmid", 0.6369, 0.2579, 30.1),
        ("lora_seed0", "dmid", 0.4016, 0.31, 319.0),
        ("full_ft_seed0", "dmid", 0.7273, 0.2578, 25.4),
        ("lora_cka_oodonly_late_l10_seed0", "dmid", 0.6973, 0.2651, 65.4),
        ("lora_cka_oodonly_both_l10_pm20_seed0", "dmid", 0.6513, 0.2751, 94.4),
    ]
    pd.DataFrame(
        rows, columns=["run_name", "dataset", "dice_mean", "dice_std", "hd95_mean"]
    ).to_csv(m, index=False)
    tex = dmid_table_tex(m)
    assert (
        tex.index("Zero-shot")
        < tex.index("LoRA")
        < tex.index("Full FT")
        < tex.index("CKA")
    )
    assert "0.402" in tex and "0.637" in tex and "319" in tex


def _metrics_rows(run, ds, det, thresholds, auroc):
    return [
        {
            "run_name": run,
            "dataset": ds,
            "detector": det,
            "threshold": t,
            "auroc": auroc,
            "auprc": auroc - 0.1,
            "n": 100,
            "n_fail": 30,
        }
        for t in thresholds
    ]


def test_detector_table_merges_dmid_csv(tmp_path):
    ladder = tmp_path / "ladder.csv"
    dmid = tmp_path / "dmid.csv"
    pd.DataFrame(
        _metrics_rows("lora_seed0", "busi", "drift", [0.5], 0.95)
        + _metrics_rows("lora_seed0", "busi", "iou_pred", [0.5], 0.72)
    ).to_csv(ladder, index=False)
    pd.DataFrame(
        _metrics_rows("lora_seed0", "dmid", "drift", [0.5], 0.76)
        + _metrics_rows("lora_seed0", "dmid", "iou_pred", [0.5], 0.60)
    ).to_csv(dmid, index=False)
    tex = detector_table_tex([ladder, dmid], datasets=["busi", "dmid"], with_auprc=True)
    assert "DMID (held-out)" in tex
    assert "0.95 / 0.85" in tex and "0.76 / 0.66" in tex


def test_threshold_sweep_table_one_column_per_threshold(tmp_path):
    m = tmp_path / "m.csv"
    pd.DataFrame(
        _metrics_rows("lora_seed0", "busi", "drift", [0.3, 0.5, 0.7], 0.9)
        + _metrics_rows("lora_seed0", "busi", "iou_pred", [0.3, 0.5, 0.7], 0.5)
    ).to_csv(m, index=False)
    tex = threshold_sweep_table_tex(m, runs=("lora_seed0",), datasets=("busi",))
    assert "0.3 & 0.5 & 0.7" in tex
    assert tex.count("0.90") == 3 and "0.50" not in tex


def test_risk_coverage_table_recall_and_residual(tmp_path):
    pts = tmp_path / "points.csv"
    pd.DataFrame(
        [
            {
                "run_name": "lora_seed0",
                "dataset": "busi",
                "detector": "drift",
                "threshold": 0.5,
                "n": 647,
                "n_fail": 61,
                "base_rate": 0.094,
                "aurc": 0.010,
                "flagged": c,
                "k": int(c * 647),
                "recall": r,
                "precision": 0.5,
                "residual_risk": rr,
            }
            for c, r, rr in [(0.1, 0.705, 0.031), (0.2, 0.869, 0.015)]
        ]
    ).to_csv(pts, index=False)
    tex = risk_coverage_table_tex(pts, runs=("lora_seed0",), datasets=("busi",))
    assert r"top 10\%" in tex and r"top 20\%" in tex
    assert "0.87 / 0.015" in tex and "0.094 & 0.010" in tex


def test_robustness_levels_table_has_pm_columns_and_std(tmp_path):
    csv = tmp_path / "curves.csv"
    pd.DataFrame(
        [
            {
                "curve": "CKA pm20",
                "dataset": "busi",
                "level": 0,
                "dice_mean": 0.826,
                "dice_std": 0.008,
                "n_seeds": 3,
            },
            {
                "curve": "CKA pm20",
                "dataset": "busi",
                "level": 20,
                "dice_mean": 0.80,
                "dice_std": 0.01,
                "n_seeds": 3,
            },
            {
                "curve": "CKA pm0",
                "dataset": "busi",
                "level": 0,
                "dice_mean": 0.81,
                "dice_std": float("nan"),
                "n_seeds": 1,
            },
        ]
    ).to_csv(csv, index=False)
    tex = robustness_levels_table_tex(csv, datasets=("busi",))
    assert "0\\,px & 20\\,px" in tex
    assert "0.826 $\\pm$ 0.008" in tex and "& 0.810 &" in tex


def test_wilcoxon_levels_table(tmp_path):
    csv = tmp_path / "w.csv"
    pd.DataFrame(
        [
            {
                "dataset": "busi",
                "level": 20,
                "n_pairs": 1941,
                "n_seeds": 3,
                "mean_delta": 0.0526,
                "median_delta": 0.0118,
                "frac_improved": 0.58,
                "p_raw": 1e-31,
                "p_holm": 5e-31,
            },
        ]
    ).to_csv(csv, index=False)
    tex = wilcoxon_levels_table_tex(csv, datasets=("busi",))
    assert "BUSI (far-OOD) & 20 & 1941 & +0.053 & +0.012 & 58\\%" in tex


def test_calibration_table(tmp_path):
    csv = tmp_path / "cal.csv"
    pd.DataFrame(
        [
            {
                "run_name": "lora_seed0",
                "dataset": "busi",
                "n": 647,
                "ece": 0.158,
                "iou_pred_mean": 0.511,
                "iou_mean": 0.669,
            },
            {
                "run_name": "lora_seed0",
                "dataset": "cbis_ddsm",
                "n": 362,
                "ece": 0.129,
                "iou_pred_mean": 0.50,
                "iou_mean": 0.379,
            },
        ]
    ).to_csv(csv, index=False)
    tex = calibration_table_tex(csv)
    assert "LoRA & 0.16 & 0.51 & 0.67 & 0.13 & 0.50 & 0.38" in tex
