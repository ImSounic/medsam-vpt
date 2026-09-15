"""LaTeX tables for the paper are generated from the result CSVs."""

import pandas as pd

from scripts.accv_tables import detector_table_tex, multiseed_table_tex


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
