"""LoRA fine-tuning for MedSAM. Self-contained — no peft dependency.

Implements rank-r LoRA on the SAM image encoder's attention qkv projections
by replacing each block.attn.qkv (nn.Linear) with a LoRALinear wrapper that
adds a low-rank residual to the base linear's output.

Mathematical form (Hu et al., 2021):
    y = W·x + (alpha/r) * (B·A·x)
    where:
      W is the frozen base linear (768 -> 2304)
      A is (r, 768), B is (2304, r) — both trainable
      A initialized with Kaiming, B with zeros (so initial residual = 0
      and the model starts identical to the frozen base)

Trainable param budget (rank=8 on 12 blocks):
    Per block: 8*768 + 2304*8 = 24,576
    LoRA total: 12 * 24,576 = 294,912
    + mask decoder:        4,058,340
    Grand total trainable: 4,353,252  (~4.65% of MedSAM)

This implementation deliberately avoids the peft library. peft pulls
transformers, which pulls tensorflow, which conflicts with numpy>=2 on
some Linux clusters and crashes at import time.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from segment_anything.modeling import Sam


class LoRALinear(nn.Module):
    """Wraps an nn.Linear with a trainable low-rank residual.

    The wrapped (frozen) linear is kept at `self.base`. New trainable
    parameters live at `self.lora_A` and `self.lora_B`.
    """

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

        # Use nn.Linear (no bias) for A and B so they inherit standard
        # initialization tooling. Same memory layout as raw nn.Parameter.
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank
        self.lora_dropout: nn.Module = (
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        # Standard LoRA init: A ~ Kaiming, B ~ zeros. With B=0 the residual
        # is exactly 0 at start, so the model behaves identically to the
        # frozen base until training kicks in.
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
    """Configure SAM for LoRA fine-tuning. Modifies sam in place.

    - Image encoder: every block's attention.qkv replaced with LoRALinear
      (base weights frozen, low-rank residual trainable)
    - Prompt encoder: frozen
    - Mask decoder: fully trainable (standard SAM-PEFT recipe)

    Args:
        sam: SAM model to configure.
        rank: LoRA rank r. Standard value is 8.
        alpha: LoRA scaling factor. Convention is alpha = 2 * rank.
        dropout: LoRA-side dropout. Keep 0 for small datasets like ISIC.
    """
    # Freeze everything to start
    for p in sam.parameters():
        p.requires_grad = False

    device = next(sam.parameters()).device

    # Wrap each encoder block's qkv linear with a LoRALinear.
    # SAM's attention has a single fused qkv: Linear(768, 3*768=2304) which
    # projects to concatenated Q,K,V. We apply LoRA to this combined projection.
    for block in sam.image_encoder.blocks:
        block.attn.qkv = LoRALinear(
            block.attn.qkv,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        ).to(device)

    # Train mask decoder fully (decoder is small enough that full training
    # adds useful expressiveness without breaking parameter-efficiency).
    for p in sam.mask_decoder.parameters():
        p.requires_grad = True
