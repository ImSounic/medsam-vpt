"""Risk-coverage: recall, residual risk and AURC of a failure signal on synthetic per-image files."""

import numpy as np
import pandas as pd

from scripts.risk_coverage import aurc, main, review_points, risk_coverage_curve


def test_perfect_ranking_curve():
    # 4 images, 2 failures; score ranks both failures first.
    score = np.array([0.9, 0.8, 0.2, 0.1])
    fail = np.array([True, True, False, False])
    c = risk_coverage_curve(score, fail)
    assert list(c.k) == [0, 1, 2, 3, 4]
    assert abs(c.recall.iloc[2] - 1.0) < 1e-12
    assert abs(c.precision.iloc[2] - 1.0) < 1e-12
    assert abs(c.residual_risk.iloc[0] - 0.5) < 1e-12  # nothing flagged: base rate
    assert abs(c.residual_risk.iloc[2] - 0.0) < 1e-12  # both failures removed
    assert np.isnan(c.residual_risk.iloc[4])  # nothing left
    pts = review_points(c, [0.5])
    assert pts[0]["k"] == 2 and abs(pts[0]["recall"] - 1.0) < 1e-12


def test_aurc_perfect_below_random():
    rng = np.random.default_rng(0)
    fail = rng.random(200) < 0.3
    perfect = fail.astype(float) + rng.random(200) * 0.1
    random_score = rng.random(200)
    a_perfect = aurc(risk_coverage_curve(perfect, fail))
    a_random = aurc(risk_coverage_curve(random_score, fail))
    assert a_perfect < a_random
    assert abs(a_random - fail.mean()) < 0.06  # random ranking ~ base rate


def test_main_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(1)
    for ds in ["busi", "cbis_ddsm"]:
        dice = rng.random(50)
        pd.DataFrame(
            {
                "image_id": [f"i{i}" for i in range(50)],
                "dice": dice,
                "iou": dice * 0.9,
                "hd95": 1.0,
                "iou_pred": rng.random(50),
                "drift": 1.0 - dice + rng.random(50) * 0.05,
            }
        ).to_csv(raw / f"lora_seed0_{ds}_per_image.csv", index=False)
    out = tmp_path / "out"
    fig = tmp_path / "rc.png"
    assert (
        main(
            [
                "--raw-dir",
                str(raw),
                "--out-dir",
                str(out),
                "--figure",
                str(fig),
                "--coverages",
                "0.2",
            ]
        )
        == 0
    )
    pts = pd.read_csv(out / "review_points.csv")
    assert set(pts.detector) == {"drift", "iou_pred"}
    drift = pts[(pts.detector == "drift") & (pts.dataset == "busi")].iloc[0]
    assert drift.aurc < drift.base_rate  # drift tracks Dice by construction
    assert (out / "risk_coverage_curves.csv").exists() and fig.exists()
