"""LoRA fine-tuning for MedSAM: one rank-r adapter wraps the fused qkv (shared across Q/K/V), no peft lib."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from segment_anything.modeling import Sam


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank
        self.lora_dropout: nn.Module = (
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        # A Kaiming, B zeros: residual is 0 at start, model matches frozen base.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.lora_B(
            self.lora_A(self.lora_dropout(x))
        )


def apply_lora(
    sam: Sam,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    **_kwargs,
) -> None:
    """Configure SAM for LoRA in place: wrap encoder qkv with LoRALinear, freeze prompt encoder, train mask decoder."""
    for p in sam.parameters():
        p.requires_grad = False

    device = next(sam.parameters()).device

    for block in sam.image_encoder.blocks:
        block.attn.qkv = LoRALinear(
            block.attn.qkv,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        ).to(device)

    for p in sam.mask_decoder.parameters():
        p.requires_grad = True
