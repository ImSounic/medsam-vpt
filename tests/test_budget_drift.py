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


def test_budget_figure_with_drift_panel(tmp_path):
    from scripts.plot_budget import make_figure

    summary = pd.DataFrame(
        [
            {"method": "lora", "budget": 50, "id_dice": 0.94, "far_ood_dice": 0.82},
            {"method": "lora", "budget": 250, "id_dice": 0.95, "far_ood_dice": 0.63},
        ]
    )
    summary.attrs["zero_shot"] = {"id_dice": 0.907, "far_ood_dice": 0.758}
    drift = pd.DataFrame(
        [
            {
                "method": "lora",
                "budget": 50,
                "far_ood_drift": 0.12,
                "far_ood_drift_enc": 0.02,
                "id_drift": 0.1,
                "id_drift_enc": 0.3,
            },
            {
                "method": "lora",
                "budget": 250,
                "far_ood_drift": 0.36,
                "far_ood_drift_enc": 0.11,
                "id_drift": 0.1,
                "id_drift_enc": 0.4,
            },
        ]
    )
    out = tmp_path / "b.png"
    make_figure(summary, out, drift=drift)
    assert out.exists()
