"""Centered Kernel Alignment (CKA) — differentiable, batch-level, for training-time use.

Implements *linear* CKA (Kornblith et al., 2019, "Similarity of Neural Network
Representations Revisited"). Linear CKA between two feature matrices

    X : (n_samples, d_X)
    Y : (n_samples, d_Y)

is

    CKA(X, Y) = ||X^T Y||_F^2  /  ( ||X^T X||_F  *  ||Y^T Y||_F )

after column-centering both X and Y. The result is a scalar in [0, 1]:
  - 1.0  : the two feature spaces are identical up to an orthogonal transform
  - 0.0  : the two feature spaces are uncorrelated under linear similarity

Useful properties for our use:
  - Differentiable (no .detach() needed) — gradients flow through it
  - Invariant to (i) orthogonal transforms on either side and (ii) isotropic scaling
  - Cheap: O(n * d_X * d_Y) compute; no kernel matrix needed
  - The (n, d) layout matches what forward hooks produce after flattening
    spatial / channel dimensions: e.g. a (B, C, H, W) activation flattens to
    (B, C*H*W), then linear_cka measures how similar each pair of images looks
    in feature space between the two models.

We use linear (not RBF) CKA because:
  - It has a closed-form derivative and is faster
  - The original CKA paper showed it correlates strongly with the kernel
    variant in practice
  - For training-time use, the kernel variant adds n^2 memory which is
    prohibitive at batch sizes > ~64

Typical use in this project:

    from src.cka import linear_cka

    # X = activations from base MedSAM at some hook layer, on a probe batch
    # Y = activations from current trainable model at the same hook layer
    similarity = linear_cka(X, Y)
    loss = 1.0 - similarity         # to be minimized
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

    # Numerator: ||X^T Y||_F^2   (use trace identity: ||A||_F^2 = trace(A^T A))
    # Equivalent to torch.linalg.norm(Xc.T @ Yc, ord="fro") ** 2 but avoids a
    # potentially large (d_X, d_Y) intermediate when d_X*d_Y > n^2.
    # Using (Y^T X X^T Y).trace() = ||X^T Y||_F^2 keeps the intermediate (n, n)
    # whenever d_X and d_Y are larger than n (which is our typical case:
    # probe batch n=32, feature dim d in the hundreds-of-thousands).
    n = X.shape[0]
    if max(Xc.shape[1], Yc.shape[1]) > n:
        # (n, n) Gram matrices route
        Kx = Xc @ Xc.T          # (n, n)
        Ky = Yc @ Yc.T          # (n, n)
        # Linear CKA = ||K_x K_y||_F / sqrt(||K_x K_x||_F * ||K_y K_y||_F)
        # equivalently = trace(K_x K_y) / sqrt(trace(K_x K_x) * trace(K_y K_y))
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
    """Reshape a forward-hook activation to a (B, D) matrix suitable for CKA.

    Supported shapes:
      - (B, D)            -> returned as-is
      - (B, C, H, W)      -> reshaped to (B, C*H*W)
      - (B, T, D)         -> reshaped to (B, T*D)        (transformer token outputs)
      - (B, *whatever)    -> reshaped to (B, prod(*whatever))

    The exact layout doesn't matter for linear CKA — what matters is that the
    same flattening is applied to BOTH models' activations so the per-image
    feature vectors live in the same space and dimension.
    """
    if activation.dim() < 2:
        raise ValueError(
            f"flatten_for_cka: activation must have batch dim. Got shape "
            f"{tuple(activation.shape)}."
        )
    B = activation.shape[0]
    return activation.reshape(B, -1)
