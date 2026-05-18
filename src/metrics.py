"""Segmentation metrics: Dice, IoU, HD95, plus bootstrap CIs.

All functions accept binary numpy arrays / torch tensors of the same shape.
Predictions and targets are expected to be 0/1.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.uint8)


def dice_score(pred, target, eps: float = 1e-6) -> float:
    """Soft-Dice: 2|A∩B| / (|A|+|B|). Scalar."""
    pred = _to_numpy(pred).astype(bool)
    target = _to_numpy(target).astype(bool)
    inter = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0  # both empty — convention: perfect score
    return float((2.0 * inter + eps) / (denom + eps))


def iou_score(pred, target, eps: float = 1e-6) -> float:
    """Intersection-over-Union. Scalar."""
    pred = _to_numpy(pred).astype(bool)
    target = _to_numpy(target).astype(bool)
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0
    return float((inter + eps) / (union + eps))


def hd95(pred, target) -> float:
    """95th-percentile symmetric Hausdorff distance over mask boundaries, in pixels.

    Pure scipy implementation — deliberately avoids monai to keep import-time
    dependencies minimal (monai pulls transformers→tensorflow which conflicts
    with numpy 2.x on some clusters). Mathematically equivalent to
    monai.metrics.utils.get_surface_distance with euclidean metric.

    Algorithm:
      1. Extract 1-pixel-thick boundaries via mask XOR (mask AND NOT eroded(mask)).
      2. For each pred-boundary pixel, distance to nearest target-boundary pixel.
      3. For each target-boundary pixel, distance to nearest pred-boundary pixel.
      4. Return 95th percentile of the concatenated distances.

    Returns inf if either mask is empty (substituted with finite max in aggregation).
    """
    pred = _to_numpy(pred).astype(bool)
    target = _to_numpy(target).astype(bool)

    if pred.sum() == 0 or target.sum() == 0:
        return float("inf")

    from scipy.ndimage import binary_erosion, distance_transform_edt

    pred_boundary = pred & ~binary_erosion(pred)
    target_boundary = target & ~binary_erosion(target)

    # Edge case: mask is a single pixel — erosion produces empty, boundary == mask
    if pred_boundary.sum() == 0:
        pred_boundary = pred
    if target_boundary.sum() == 0:
        target_boundary = target

    # distance_transform_edt(~M) returns, at each pixel, the distance to the
    # nearest True pixel of M (because EDT computes distance to nearest 0,
    # which after inversion is nearest 1 of the original).
    dist_to_target_bd = distance_transform_edt(~target_boundary)
    dist_to_pred_bd = distance_transform_edt(~pred_boundary)

    d_pred_to_target = dist_to_target_bd[pred_boundary]
    d_target_to_pred = dist_to_pred_bd[target_boundary]

    d_all = np.concatenate([d_pred_to_target, d_target_to_pred])
    if len(d_all) == 0:
        return 0.0
    return float(np.percentile(d_all, 95))


def aggregate_metrics(per_image: list[dict]) -> dict:
    """Aggregate per-image metric dicts into mean ± std + bootstrap CI."""
    if not per_image:
        return {}

    out = {}
    for key in per_image[0]:
        vals = np.array([d[key] for d in per_image], dtype=np.float64)
        # Replace inf HD95 (empty masks) with finite max for aggregation
        if np.any(np.isinf(vals)):
            finite = vals[np.isfinite(vals)]
            replacement = finite.max() if len(finite) else 0.0
            vals = np.where(np.isinf(vals), replacement, vals)
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        ci_lo, ci_hi = bootstrap_ci(vals)
        out[f"{key}_mean"] = mean
        out[f"{key}_std"] = std
        out[f"{key}_ci_lo"] = ci_lo
        out[f"{key}_ci_hi"] = ci_hi
    return out


def bootstrap_ci(
    values: Iterable[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(list(values), dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0
    boot_means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means[i] = sample.mean()
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi
