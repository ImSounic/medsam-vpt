"""Budget and hook-ablation plots run end to end on synthetic runs CSVs."""

import pandas as pd

from scripts.plot_budget import aggregate_seeds, budget_long, first_drop_below_zero_shot
from scripts.plot_budget import main as budget_main
from scripts.plot_budget import parse_budget_run
from scripts.plot_hook_ablation import main as hook_main
from scripts.plot_hook_ablation import parse_position

DATASETS = ["isic2018_test", "ph2", "busi", "cbis_ddsm"]


def _runs_csv(path, run_dice, method_of=lambda r: r.split("_n")[0]):
    rows = []
    for run, per_ds in run_dice.items():
        for ds, dice in per_ds.items():
            rows.append(
                {
                    "run_name": run,
                    "method": method_of(run),
                    "dataset": ds,
                    "seed": 0,
                    "dice_mean": dice,
                    "dice_std": 0.01,
                    "iou_mean": dice - 0.05,
                    "hd95_mean": 10.0,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_parse_budget_run():
    assert parse_budget_run("lora_encoder_only_n250_seed0") == (
        "lora_encoder_only",
        250,
        0,
    )
    assert parse_budget_run("lora_n50_seed2") == ("lora", 50, 2)
    assert parse_budget_run("lora_seed0") is None


def test_budget_summary_and_first_drop(tmp_path):
    budget_csv = tmp_path / "runs_t2.csv"
    seeds_csv = tmp_path / "runs_t6.csv"
    full_csv = tmp_path / "runs.csv"
    _runs_csv(
        budget_csv,
        {
            "lora_n50_seed0": dict(
                zip(DATASETS, [0.90, 0.90, 0.85, 0.70])
            ),  # far 0.775 > zs 0.755
            "lora_n250_seed0": dict(zip(DATASETS, [0.93, 0.93, 0.75, 0.60])),
            "lora_n1000_seed0": dict(zip(DATASETS, [0.95, 0.95, 0.70, 0.50])),
        },
    )
    # Seeds 1-2 only at n=50: mean stays 0.775, std over the three seeds = 0.01.
    _runs_csv(
        seeds_csv,
        {
            "lora_n50_seed1": dict(zip(DATASETS, [0.90, 0.90, 0.86, 0.71])),
            "lora_n50_seed2": dict(zip(DATASETS, [0.90, 0.90, 0.84, 0.69])),
        },
    )
    _runs_csv(
        full_csv,
        {
            "zero_shot": dict(zip(DATASETS, [0.90, 0.90, 0.82, 0.69])),
            "lora_seed0": dict(zip(DATASETS, [0.95, 0.95, 0.78, 0.50])),
        },
        method_of=lambda r: r.replace("_seed0", ""),
    )
    long = budget_long([budget_csv, seeds_csv], [full_csv])
    assert len(long[(long.method == "lora") & (long.budget == 50)]) == 3
    agg = aggregate_seeds(long)
    first_drop = first_drop_below_zero_shot(agg)
    lora = agg[agg.method == "lora"].sort_values("budget")
    assert list(lora.budget) == [50, 250, 1000, 2595]
    assert list(lora.n_seeds) == [3, 1, 1, 1]
    assert abs(lora.iloc[0].far_ood_dice_mean - 0.775) < 1e-9
    assert abs(lora.iloc[0].far_ood_dice_std - 0.01) < 1e-9
    assert lora.iloc[1].far_ood_dice_std != lora.iloc[1].far_ood_dice_std  # NaN
    assert first_drop["lora"] == 250  # 0.675 < zero-shot 0.755
    out = tmp_path / "budget.png"
    assert (
        budget_main(
            [
                "--budget-csv",
                str(budget_csv),
                str(seeds_csv),
                "--full-csv",
                str(full_csv),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert out.exists()


def test_parse_position():
    assert parse_position("lora_cka_oodonly_enc_l10_pm20_seed0") == "enc"
    assert parse_position("lora_cka_oodonly_late_l10_pm20_seed1") == "late"
    assert parse_position("lora_seed0_pm20") is None


def test_hook_ablation_plot(tmp_path):
    t1 = tmp_path / "runs_accv_t1.csv"
    base = tmp_path / "runs_pm20.csv"
    _runs_csv(
        t1,
        {
            "lora_cka_oodonly_late_l10_pm20_seed0": dict(
                zip(DATASETS, [0.95, 0.95, 0.82, 0.48])
            ),
            "lora_cka_oodonly_enc_l10_pm20_seed0": dict(
                zip(DATASETS, [0.95, 0.95, 0.78, 0.44])
            ),
            "lora_cka_oodonly_both_l10_pm20_seed0": dict(
                zip(DATASETS, [0.95, 0.95, 0.82, 0.48])
            ),
        },
        method_of=lambda r: "lora",
    )
    _runs_csv(
        base,
        {"lora_seed0_pm20": dict(zip(DATASETS, [0.95, 0.95, 0.77, 0.44]))},
        method_of=lambda r: "lora",
    )
    out = tmp_path / "hooks.png"
    rc = hook_main(["--csv", str(t1), "--baseline-csv", str(base), "--out", str(out)])
    assert rc == 0 and out.exists() and out.with_suffix(".csv").exists()
