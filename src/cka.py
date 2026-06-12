"""Differentiable linear CKA."""

from __future__ import annotations

import torch


def _center_columns(X: torch.Tensor) -> torch.Tensor:
    return X - X.mean(dim=0, keepdim=True)


def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Linear CKA between two (n, d) feature matrices; differentiable wrt both."""
    if X.dim() != 2 or Y.dim() != 2:
        raise ValueError(
            f"linear_cka expects 2D inputs (n, d). Got X.shape={tuple(X.shape)}, "
            f"Y.shape={tuple(Y.shape)}. Flatten higher-dim features first."
        )
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"linear_cka: sample dim must match. Got X[{X.shape[0]}] vs Y[{Y.shape[0]}]."
        )

    Xc = _center_columns(X)
    Yc = _center_columns(Y)

    n = X.shape[0]
    if max(Xc.shape[1], Yc.shape[1]) > n:
        # (n, n)
        Kx = Xc @ Xc.T  # (n, n)
        Ky = Yc @ Yc.T  # (n, n)
        num = (Kx * Ky).sum()
        denom = torch.sqrt((Kx * Kx).sum() * (Ky * Ky).sum())
    else:
        XtY = Xc.T @ Yc  # (d_X, d_Y)
        XtX = Xc.T @ Xc  # (d_X, d_X)
        YtY = Yc.T @ Yc  # (d_Y, d_Y)
        num = (XtY * XtY).sum()
        denom = torch.sqrt((XtX * XtX).sum() * (YtY * YtY).sum())

    return num / (denom + eps)


def flatten_for_cka(activation: torch.Tensor) -> torch.Tensor:
    if activation.dim() < 2:
        raise ValueError(
            f"flatten_for_cka: activation must have batch dim. Got shape "
            f"{tuple(activation.shape)}."
        )
    B = activation.shape[0]
    return activation.reshape(B, -1)
