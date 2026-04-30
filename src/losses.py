"""Segmentation losses.

DiceBCELoss = (1-w)*BCE + w*Dice on a single binary mask channel.
Standard recipe for medical segmentation. BCE pushes pixel-wise correctness;
Dice pushes overlap, which matters when the foreground is small (most ISIC
lesions cover <30% of the image).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss.

    pred_logits: (B, 1, H, W) or (B, H, W)
    target:      (B, H, W) with 0/1 values
    """
    if pred_logits.dim() == 4:
        pred_logits = pred_logits.squeeze(1)
    pred = torch.sigmoid(pred_logits)
    target_f = target.float()
    inter = (pred * target_f).sum(dim=(-2, -1))
    denom = pred.sum(dim=(-2, -1)) + target_f.sum(dim=(-2, -1))
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self, pred_logits: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if pred_logits.dim() == 4:
            pred_logits = pred_logits.squeeze(1)
        target_f = target.float()
        bce = self.bce(pred_logits, target_f)
        d = dice_loss(pred_logits, target)
        total = (1.0 - self.dice_weight) * bce + self.dice_weight * d
        return total, {"bce": bce.item(), "dice_loss": d.item()}
