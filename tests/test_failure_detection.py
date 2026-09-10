"""AUROC / AUPRC / ECE from per-image CSVs; higher drift and lower iou_pred mean failure."""

import csv
import math

import numpy as np

from scripts.failure_detection import auprc, auroc, build_table, ece, load_per_image


def test_auroc_perfect_and_reversed():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([False, False, True, True])
    assert auroc(scores, labels) == 1.0
    assert auroc(-scores, labels) == 0.0


def test_auroc_handles_ties_and_degenerate():
    assert auroc(np.array([1.0, 1.0, 1.0]), np.array([True, False, True])) == 0.5
    assert math.isnan(auroc(np.array([0.1, 0.9]), np.array([True, True])))


def test_auprc_perfect_is_one():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([True, True, False, False])
    assert abs(auprc(scores, labels) - 1.0) < 1e-9


def test_ece_perfectly_calibrated_is_zero():
    conf = np.linspace(0.05, 0.95, 10)
    assert ece(conf, conf, n_bins=10) < 1e-9
    assert abs(ece(np.full(10, 0.9), np.full(10, 0.4), n_bins=10) - 0.5) < 1e-9


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_build_table_from_per_image_csvs(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        dice = float(rng.uniform(0.1, 1.0))
        rows.append(
            {
                "image_id": f"img{i}",
                "dice": dice,
                "iou": dice / (2 - dice),
                "hd95": 5.0,
                "iou_pred": dice + rng.normal(0, 0.05),
                "drift": (1 - dice) + rng.normal(0, 0.05),
            }
        )
    _write(tmp_path / "lora_seed0_cbis_ddsm_per_image.csv", rows)
    df = load_per_image(tmp_path, "lora_seed0", "cbis_ddsm")
    table = build_table({("lora_seed0", "cbis_ddsm"): df}, thresholds=(0.5, 0.7))
    assert {
        "run_name",
        "dataset",
        "detector",
        "threshold",
        "auroc",
        "auprc",
        "n",
        "n_fail",
    } <= set(table.columns)
    sub = table[(table.threshold == 0.5)]
    assert set(sub.detector) == {"iou_pred", "drift"}
    assert (sub.auroc > 0.9).all()
