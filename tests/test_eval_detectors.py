"""iou_pred is returned per image; drift is 1 - CKA on the upscaled mask embedding; CSV has the new columns."""

import csv

import torch

from src.drift import decoder_drift
from src.eval import (
    predict_from_embeddings,
    predict_from_embeddings_with_iou,
    write_per_image_csv,
)


def test_predict_with_iou_shapes(make_sam):
    sam = make_sam()
    emb = torch.randn(2, 256, 64, 64)
    boxes = torch.tensor([[100.0, 100.0, 500.0, 500.0], [10.0, 10.0, 900.0, 900.0]])
    masks, ious = predict_from_embeddings_with_iou(sam, emb, boxes, 1024, 1024)
    assert masks.shape == (2, 1024, 1024) and masks.dtype == torch.uint8
    assert ious.shape == (2,) and torch.isfinite(ious).all()


def test_predict_from_embeddings_still_returns_masks_only(make_sam):
    sam = make_sam()
    emb = torch.randn(1, 256, 64, 64)
    boxes = torch.tensor([[100.0, 100.0, 500.0, 500.0]])
    out = predict_from_embeddings(sam, emb, boxes, 256, 256)
    assert out.shape == (1, 256, 256)


def test_drift_zero_for_identical_and_scaled():
    a = torch.randn(32, 256, 256)
    assert abs(decoder_drift(a, a)) < 1e-5
    assert abs(decoder_drift(a, 3.0 * a)) < 1e-5


def test_drift_near_one_for_independent():
    a = torch.randn(32, 256, 256)
    b = torch.randn(32, 256, 256)
    assert decoder_drift(a, b) > 0.9


def test_drift_accepts_leading_batch_dim():
    a = torch.randn(1, 32, 256, 256)
    assert abs(decoder_drift(a, a)) < 1e-5


def test_per_image_csv_columns(tmp_path):
    rows = [
        {
            "image_id": "x",
            "dice": 0.9,
            "iou": 0.8,
            "hd95": 3.0,
            "iou_pred": 0.85,
            "drift": 0.1,
        },
        {
            "image_id": "y",
            "dice": 0.2,
            "iou": 0.1,
            "hd95": 40.0,
            "iou_pred": 0.9,
            "drift": 0.5,
        },
    ]
    p = write_per_image_csv(tmp_path / "a" / "b.csv", rows)
    with open(p) as f:
        header = next(csv.reader(f))
    assert header == ["image_id", "dice", "iou", "hd95", "iou_pred", "drift"]


def test_drift_works_on_encoder_neck_shape():
    a = torch.randn(1, 256, 64, 64)
    b = torch.randn(1, 256, 64, 64)
    assert abs(decoder_drift(a, a)) < 1e-5
    assert decoder_drift(a, b) > 0.5


def test_per_image_csv_keeps_encoder_drift_column(tmp_path):
    rows = [
        {
            "image_id": "x",
            "dice": 0.9,
            "iou": 0.8,
            "hd95": 3.0,
            "iou_pred": 0.85,
            "drift": 0.1,
            "drift_enc": 0.02,
        }
    ]
    p = write_per_image_csv(tmp_path / "c.csv", rows)
    with open(p) as f:
        header = next(csv.reader(f))
    assert header[-2:] == ["drift", "drift_enc"]
