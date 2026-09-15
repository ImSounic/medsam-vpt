"""Hook ablation over seeds: mean/std per arm from several runs CSVs, paired Wilcoxon between arms on sweep per-image files."""

import numpy as np
import pandas as pd

from scripts.plot_hook_ablation import arm_of, paired_arm_tests, seed_table_from_csvs

DS = ["isic2018_test", "ph2", "busi", "cbis_ddsm"]


def test_arm_of():
    assert arm_of("lora_seed1_pm20") == ("no_cka", 1)
    assert arm_of("lora_cka_oodonly_late_l10_pm20_seed0") == ("late", 0)
    assert arm_of("lora_cka_oodonly_both_l10_pm20_seed2") == ("both", 2)
    assert arm_of("lora_cka_oodonly_enc_l10_pm20_seed1") == ("enc", 1)
    assert arm_of("zero_shot") is None


def _csv(path, rows):
    pd.DataFrame(rows, columns=["run_name", "dataset", "dice_mean"]).to_csv(
        path, index=False
    )


def test_seed_table_aggregates_arms(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _csv(
        a,
        [
            ("lora_cka_oodonly_both_l10_pm20_seed0", "busi", 0.86),
            ("lora_seed0_pm20", "busi", 0.81),
        ],
    )
    _csv(
        b,
        [
            ("lora_cka_oodonly_both_l10_pm20_seed1", "busi", 0.80),
            ("lora_cka_oodonly_both_l10_pm20_seed2", "busi", 0.83),
            ("lora_seed1_pm20", "busi", 0.75),
            ("lora_seed2_pm20", "busi", 0.76),
        ],
    )
    t = seed_table_from_csvs([a, b])
    both = t[(t.arm == "both") & (t.dataset == "busi")].iloc[0]
    assert both.n == 3 and abs(both.dice_mean - 0.83) < 1e-9 and both.dice_std > 0
    none = t[(t.arm == "no_cka") & (t.dataset == "busi")].iloc[0]
    assert none.n == 3 and abs(none.dice_mean - (0.81 + 0.75 + 0.76) / 3) < 1e-9


def _sweep(root, run, level, dice):
    d = root / "per_image"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"image_id": [f"i{k}" for k in range(len(dice))], "dice_mean": dice}
    ).to_csv(d / f"{run}_busi_pm{level}.csv", index=False)


def test_paired_arm_tests(tmp_path):
    rng = np.random.default_rng(1)
    base = rng.uniform(0.3, 0.9, 40)
    for s in (0, 1):
        _sweep(
            tmp_path / "x", f"lora_cka_oodonly_both_l10_pm20_seed{s}", 20, base + 0.05
        )
        _sweep(tmp_path / "y", f"lora_cka_oodonly_late_l10_pm20_seed{s}", 20, base)
        _sweep(tmp_path / "z", f"lora_seed{s}_pm20", 20, base - 0.05)
    res = paired_arm_tests([tmp_path / "x", tmp_path / "y", tmp_path / "z"], level=20)
    row = res[
        (res.arm_a == "both") & (res.arm_b == "late") & (res.dataset == "busi")
    ].iloc[0]
    assert row.n_pairs == 80 and row.median_delta > 0 and row.p_holm < 1e-6
