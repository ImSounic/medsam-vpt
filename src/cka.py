"""Linear CKA (Kornblith et al., 2019), differentiable, for training-time use.

For column-centered feature matrices X (n, d_X) and Y (n, d_Y):

    CKA(X, Y) = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

Scalar in [0, 1]: 1.0 = identical up to orthogonal transform, 0.0 = uncorrelated.
Differentiable wrt X and Y (no detach), invariant to orthogonal transforms and
isotropic scaling. Linear (not RBF) avoids the n^2 kernel memory.
"""
from __future__ import annotations

import torch


def _center_columns(X: torch.Tensor) -> torch.Tensor:
    """Subtract the per-column mean from X. Returns same shape."""
    return X - X.mean(dim=0, keepdim=True)


def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Linear CKA between two (n, d) feature matrices.

    Args:
        X: tensor of shape (n_samples, d_X). Feature matrix from one model.
        Y: tensor of shape (n_samples, d_Y). Feature matrix from the other model.
            n_samples must match X. d_X and d_Y can differ.
        eps: small constant to keep the denominator > 0 in pathological cases
            (e.g. zero-variance features).

    Returns:
        Scalar tensor in [0, 1]. Differentiable wrt both X and Y.

    Raises:
        ValueError: if shapes don't match in the sample dimension or X, Y aren't 2D.
    """
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

    # Two routes give the same CKA; pick whichever keeps the intermediate small.
    # When d > n (our case: n~32, d in the hundreds of thousands) the (n, n) Gram
    # route is far cheaper than forming (d_X, d_Y).
    n = X.shape[0]
    if max(Xc.shape[1], Yc.shape[1]) > n:
        # (n, n) Gram route: CKA = trace(Kx Ky) / sqrt(trace(Kx Kx) trace(Ky Ky))
        Kx = Xc @ Xc.T          # (n, n)
        Ky = Yc @ Yc.T          # (n, n)
        num = (Kx * Ky).sum()
        denom = torch.sqrt((Kx * Kx).sum() * (Ky * Ky).sum())
    else:
        # (d_X, d_Y) feature-product route (cheaper when d < n)
        XtY = Xc.T @ Yc         # (d_X, d_Y)
        XtX = Xc.T @ Xc         # (d_X, d_X)
        YtY = Yc.T @ Yc         # (d_Y, d_Y)
        num = (XtY * XtY).sum()
        denom = torch.sqrt((XtX * XtX).sum() * (YtY * YtY).sum())

    return num / (denom + eps)


def flatten_for_cka(activation: torch.Tensor) -> torch.Tensor:
    """Flatten a (B, ...) hook activation to (B, D). Layout doesn't matter for
    linear CKA as long as both models get the same flattening.
    """
    if activation.dim() < 2:
        raise ValueError(
            f"flatten_for_cka: activation must have batch dim. Got shape "
            f"{tuple(activation.shape)}."
        )
    B = activation.shape[0]
    return activation.reshape(B, -1)
