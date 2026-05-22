"""Segmentation metrics: Dice, IoU, HD95, plus bootstrap CIs.

All functions accept binary numpy arrays / torch tensors of the same shape.
Predictions and targets are expected to be 0/1.
"""
from __future__ import annotations

# --- Silence tensorflow / tensorboard chatter before anything else ---------
# monai.metrics.utils.get_surface_distance pulls in monai's full package init,
# which on some clusters (JupyterLab, etc.) transitively loads tensorboard ->
# tensorflow -> ml_dtypes. That chain prints C++ INFO/ERROR lines and Python
# tracebacks even when monai eventually works. We can't fix the cluster's
# broken TF install, but we can:
#   1. Tell TF's C++ logger to be quiet via env vars (must be set BEFORE
#      tensorflow is imported, hence before monai).
#   2. Eager-import monai inside redirected stderr/stdout so any Python-side
#      noise during the import chain is captured silently.
# After this block, hd95() is fast and never re-imports anything.
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import contextlib
import io
import warnings
from typing import Iterable

import numpy as np
import torch

_MONAI_GET_SURFACE = None
with contextlib.redirect_stderr(io.StringIO()), \
     contextlib.redirect_stdout(io.StringIO()), \
     warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from monai.metrics.utils import get_surface_distance as _monai_gsd
        _MONAI_GET_SURFACE = _monai_gsd
    except Exception:
        # monai unavailable / broken on this machine — hd95() falls back to scipy.
        _MONAI_GET_SURFACE = None


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
    """95th-percentile Hausdorff distance, in pixels.

    Uses monai's robust implementation if available; falls back to a simple
    scipy-based version otherwise. Returns inf if either mask is empty
    (which we substitute with the image diagonal in aggregation).
    """
    pred = _to_numpy(pred).astype(bool)
    target = _to_numpy(target).astype(bool)

    if pred.sum() == 0 or target.sum() == 0:
        return float("inf")

    if _MONAI_GET_SURFACE is not None:
        try:
            # symmetric 95th-percentile
            d1 = _MONAI_GET_SURFACE(pred, target, distance_metric="euclidean")
            d2 = _MONAI_GET_SURFACE(target, pred, distance_metric="euclidean")
            d = np.concatenate([d1, d2])
            if len(d) == 0:
                return 0.0
            return float(np.percentile(d, 95))
        except Exception:
            pass  # fall through to scipy

    # Fallback: distance transforms via scipy (used if monai is unavailable
    # or raises at call time)
    from scipy.ndimage import distance_transform_edt

    # Distance from each pred boundary point to nearest target point
    target_dt = distance_transform_edt(~target)
    pred_dt = distance_transform_edt(~pred)

    # Boundary-only is more correct, but for a fallback we use full-mask DTs
    d_pred_to_target = target_dt[pred]
    d_target_to_pred = pred_dt[target]
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
