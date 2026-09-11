"""Multi-seed CKA vs no-CKA: seed table from tight-box CSVs, paired Wilcoxon over bbox-sweep per-image files."""

import numpy as np
import pandas as pd

from scripts.accv_cka_multiseed import (
    holm,
    load_sweep_per_image,
    paired_tests,
    seed_table,
    tag_group,
)

DS = ["isic2018_test", "ph2", "busi", "cbis_ddsm"]


def test_tag_group():
    assert tag_group("lora_cka_oodonly_late_l10_pm20_seed1") == ("cka", 1)
    assert tag_group("lora_seed2_pm20") == ("no_cka", 2)
    assert tag_group("zero_shot") is None
    assert tag_group("lora_cka_oodonly_enc_l10_pm20_seed0") is None


def _tight(path, rows):
    pd.DataFrame(
        rows, columns=["run_name", "dataset", "dice_mean", "iou_mean", "hd95_mean"]
    ).to_csv(path, index=False)


def test_seed_table_means_and_delta(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _tight(
        a,
        [
            ("lora_cka_oodonly_late_l10_pm20_seed0", "busi", 0.80, 0.70, 10.0),
            ("lora_cka_oodonly_late_l10_pm20_seed1", "busi", 0.84, 0.74, 12.0),
        ],
    )
    _tight(
        b,
        [
            ("lora_seed0_pm20", "busi", 0.76, 0.66, 20.0),
            ("lora_seed1_pm20", "busi", 0.74, 0.64, 22.0),
            ("zero_shot", "busi", 0.82, 0.70, 15.0),
        ],
    )
    t = seed_table([a, b])
    row = t[(t.dataset == "busi")].iloc[0]
    assert abs(row.cka_dice_mean - 0.82) < 1e-9 and row.cka_n == 2
    assert abs(row.no_cka_dice_mean - 0.75) < 1e-9 and row.no_cka_n == 2
    assert abs(row.delta_dice - 0.07) < 1e-9
    assert abs(row.cka_hd95_mean - 11.0) < 1e-9


def _sweep_dir(root, run, seed, level, dice):
    d = root / "per_image"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"image_id": [f"im{i}" for i in range(len(dice))], "dice_mean": dice}
    ).to_csv(d / f"{run}_busi_pm{level}.csv", index=False)


def test_paired_tests_pool_seeds_and_holm(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.uniform(0.3, 0.9, 30)
    cka_root = tmp_path / "cka"
    no_root = tmp_path / "no"
    for seed in (0, 1):
        _sweep_dir(
            cka_root,
            f"lora_cka_oodonly_late_l10_pm20_seed{seed}",
            seed,
            20,
            base + 0.05,
        )
        _sweep_dir(no_root, f"lora_seed{seed}_pm20", seed, 20, base)
    per = load_sweep_per_image([cka_root, no_root], levels=(20,))
    assert set(per.group) == {"cka", "no_cka"}
    assert len(per) == 120
    res = paired_tests(per)
    row = res.iloc[0]
    assert row.dataset == "busi" and row.level == 20 and row.n_pairs == 60
    assert row.p_raw < 1e-6 and row.p_holm < 1e-6 and row.median_delta > 0


def test_holm_monotone():
    p = holm(np.array([0.01, 0.04, 0.03]))
    assert list(np.round(p, 4)) == [0.03, 0.06, 0.06]
