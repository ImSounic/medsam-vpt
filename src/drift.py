"""Per-image decoder drift: 1 - linear CKA between base and adapted upscaled mask embeddings."""

from __future__ import annotations

import torch

from src.cka import linear_cka


def _positions_by_channels(act: torch.Tensor) -> torch.Tensor:
    """(1, C, H, W) or (C, H, W) -> (H*W, C) float32 on the same device."""
    if act.dim() == 4:
        if act.shape[0] != 1:
            raise ValueError(f"expected a single image, got batch {act.shape[0]}")
        act = act[0]
    if act.dim() != 3:
        raise ValueError(f"expected (C, H, W), got {tuple(act.shape)}")
    c = act.shape[0]
    return act.reshape(c, -1).transpose(0, 1).float()


@torch.no_grad()
def decoder_drift(base_act: torch.Tensor, cur_act: torch.Tensor) -> float:
    """Spatial positions are samples, channels are features (256*256 x 32 for SAM)."""
    x = _positions_by_channels(base_act)
    y = _positions_by_channels(cur_act)
    return float(1.0 - linear_cka(x, y).item())
