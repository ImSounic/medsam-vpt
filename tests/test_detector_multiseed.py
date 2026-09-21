"""Multi-seed detector aggregation: run-name parsing and mean/std over seeds."""

import numpy as np
import pandas as pd

from scripts.detector_multiseed import aggregate, load_frames, per_seed_table, split_run


def test_split_run_handles_seed_suffixes():
    assert split_run("lora_seed1") == ("lora", 1)
    assert split_run("lora_encoder_only_r28_all_seed2") == ("lora_encoder_only", 2)
    assert split_run("vpt_shallow_seed0") == ("vpt_shallow", 0)
    assert split_run("zero_shot") == ("zero_shot", 0)


def _dump(d, run, ds, seed):
    rng = np.random.default_rng(seed)
    dice = rng.random(80)
    pd.DataFrame(
        {
            "image_id": [f"i{i}" for i in range(80)],
            "dice": dice,
            "iou": dice * 0.9,
            "hd95": 1.0,
            "iou_pred": rng.random(80),
            "drift": 1.0 - dice + rng.random(80) * 0.05,
        }
    ).to_csv(d / f"{run}_{ds}_per_image.csv", index=False)


def test_aggregate_over_three_seeds(tmp_path):
    a, b = tmp_path / "raw_t3_pm0", tmp_path / "raw_t8_pm0"
    a.mkdir(), b.mkdir()
    _dump(a, "lora_seed0", "busi", 0)
    _dump(b, "lora_seed1", "busi", 1)
    _dump(b, "lora_seed2", "busi", 2)
    _dump(a, "zero_shot", "busi", 3)
    frames = load_frames([a, b], ["busi", "cbis_ddsm"])
    assert len(frames) == 4
    ps = per_seed_table(frames, 0.5)
    assert set(ps.seed) == {0, 1, 2}
    agg = aggregate(ps, min_fail=20)
    lora = agg[(agg.method == "lora") & (agg.detector == "drift")].iloc[0]
    assert lora.n_seeds == 3 and lora.auroc_mean > 0.9 and lora.auroc_std >= 0
    zs = agg[(agg.method == "zero_shot") & (agg.detector == "iou_pred")].iloc[0]
    assert zs.n_seeds == 1 and np.isnan(zs.auroc_std)
