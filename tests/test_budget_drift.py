"""Budget drift panel: mean decoder and encoder drift per (method, budget) from per-image files."""

import pandas as pd

from scripts.plot_budget import drift_summary


def _raw(dirpath, run, ds, drift, drift_enc):
    dirpath.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "image_id": ["a", "b"],
            "dice": [0.5, 0.6],
            "iou": [0.4, 0.5],
            "hd95": [1, 2],
            "iou_pred": [0.5, 0.5],
            "drift": [drift, drift],
            "drift_enc": [drift_enc, drift_enc],
        }
    ).to_csv(dirpath / f"{run}_{ds}_per_image.csv", index=False)


def test_drift_summary_means_far_ood(tmp_path):
    raw = tmp_path / "raw_t7"
    _raw(raw, "lora_n50_seed0", "busi", 0.10, 0.01)
    _raw(raw, "lora_n50_seed0", "cbis_ddsm", 0.30, 0.03)
    _raw(raw, "lora_n50_seed0", "isic2018_test", 0.05, 0.01)
    _raw(raw, "decoder_only_n250_seed0", "busi", 0.20, 0.0)
    _raw(raw, "decoder_only_n250_seed0", "cbis_ddsm", 0.40, 0.0)
    df = drift_summary(raw)
    lora = df[(df.method == "lora") & (df.budget == 50)].iloc[0]
    assert (
        abs(lora.far_ood_drift - 0.20) < 1e-9
        and abs(lora.far_ood_drift_enc - 0.02) < 1e-9
    )
    assert abs(lora.id_drift - 0.05) < 1e-9
    dec = df[(df.method == "decoder_only") & (df.budget == 250)].iloc[0]
    assert abs(dec.far_ood_drift - 0.30) < 1e-9
