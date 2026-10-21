"""Methods-by-datasets Dice table (camera-ready Table 1)."""

import pandas as pd

from scripts.accv_tables import method_table_tex


def _runs(tmp_path):
    rows = []
    for seed, (b, c) in enumerate([(0.80, 0.50), (0.82, 0.54), (0.84, 0.58)]):
        for ds, v in [("isic2018_test", 0.95), ("ph2", 0.95), ("busi", b), ("cbis_ddsm", c)]:
            rows.append({"run_name": f"lora_seed{seed}_pm20", "dataset": ds, "dice_mean": v})
    for ds, v in [("isic2018_test", 0.907), ("ph2", 0.905), ("busi", 0.823), ("cbis_ddsm", 0.692)]:
        rows.append({"run_name": "zero_shot", "dataset": ds, "dice_mean": v})
        rows.append({"run_name": "zero_shot", "dataset": ds, "dice_mean": v})  # repeated across files
    p = tmp_path / "runs.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_mean_std_over_seeds_and_single_run(tmp_path):
    tex = method_table_tex(
        [_runs(tmp_path)],
        [("Zero-shot", r"^zero_shot$"), ("LoRA", r"^lora_seed\d_pm20$")],
    )
    assert r"\begin{tabular}{lcccc}" in tex
    zs = next(l for l in tex.splitlines() if l.startswith("Zero-shot"))
    assert "0.823" in zs and r"\pm" not in zs  # one distinct run: no std
    lora = next(l for l in tex.splitlines() if l.startswith("LoRA"))
    assert "0.820 $\\pm$ 0.020" in lora and "0.540 $\\pm$ 0.040" in lora


def test_missing_rows_are_skipped_and_midrule_kept(tmp_path):
    tex = method_table_tex(
        [_runs(tmp_path)],
        [("LoRA", r"^lora_seed\d_pm20$"), None, ("Absent", r"^nothing$")],
    )
    assert "Absent" not in tex
    assert tex.count(r"\midrule") == 2
