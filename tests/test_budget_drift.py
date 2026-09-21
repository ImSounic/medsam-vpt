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
    # A second seed at the same (method, budget) is averaged in.
    _raw(raw, "decoder_only_n250_seed1", "busi", 0.40, 0.0)
    _raw(raw, "decoder_only_n250_seed1", "cbis_ddsm", 0.60, 0.0)
    df = drift_summary(raw)
    lora = df[(df.method == "lora") & (df.budget == 50)].iloc[0]
    assert (
        abs(lora.far_ood_drift - 0.20) < 1e-9
        and abs(lora.far_ood_drift_enc - 0.02) < 1e-9
    )
    assert abs(lora.id_drift - 0.05) < 1e-9
    assert lora.n_seeds == 1
    dec = df[(df.method == "decoder_only") & (df.budget == 250)].iloc[0]
    assert dec.n_seeds == 2
    assert abs(dec.far_ood_drift - 0.40) < 1e-9  # mean of 0.30 and 0.50


def test_budget_figure_with_drift_panel(tmp_path):
    from scripts.plot_budget import make_figure

    summary = pd.DataFrame(
        [
            {
                "method": "lora",
                "budget": 50,
                "n_seeds": 3,
                "id_dice_mean": 0.94,
                "id_dice_std": 0.002,
                "far_ood_dice_mean": 0.82,
                "far_ood_dice_std": 0.02,
            },
            {
                "method": "lora",
                "budget": 250,
                "n_seeds": 1,
                "id_dice_mean": 0.95,
                "id_dice_std": float("nan"),
                "far_ood_dice_mean": 0.63,
                "far_ood_dice_std": float("nan"),
            },
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
