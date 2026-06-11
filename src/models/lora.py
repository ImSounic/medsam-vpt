"""LoRA fine-tuning for MedSAM. Self-contained, no peft dependency.

Rank-r LoRA on the SAM image encoder's attention qkv projections: each
block.attn.qkv (nn.Linear) is replaced with a LoRALinear that adds a
low-rank residual to the base output.

Form (Hu et al., 2021): y = W*x + (alpha/r) * (B*A*x). W is frozen
(768 -> 2304); A (r,768) and B (2304,r) are trainable; A Kaiming-init,
B zero-init so the initial residual is 0.

Note: we apply LoRA to SAM's single fused qkv projection (768 -> 2304),
not separate Q/K/V matrices.

Trainable budget (rank=8, 12 blocks): 294,912 LoRA + 4,058,340 decoder
= 4,353,252 (~4.65% of MedSAM).

We avoid the peft library on purpose: it pulls transformers -> tensorflow,
which conflicts with numpy>=2 on some Linux clusters and crashes at import.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from segment_anything.modeling import Sam


class LoRALinear(nn.Module):
    """Wraps an nn.Linear with a trainable low-rank residual.

    Frozen base at self.base; trainable params at self.lora_A / self.lora_B.
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

        # nn.Linear (no bias) for A and B to inherit standard init tooling.
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
    """Configure SAM for LoRA fine-tuning, in place.

    Encoder qkv linears wrapped with LoRALinear (base frozen, residual
    trainable); prompt encoder frozen; mask decoder fully trainable.

    Args:
        sam: SAM model to configure.
        rank: LoRA rank r. Standard value is 8.
        alpha: LoRA scaling factor. Convention is alpha = 2 * rank.
        dropout: LoRA-side dropout. Keep 0 for small datasets like ISIC.
    """
    for p in sam.parameters():
        p.requires_grad = False

    device = next(sam.parameters()).device

    # Wrap each block's fused qkv (Linear 768 -> 2304) with LoRALinear.
    for block in sam.image_encoder.blocks:
        block.attn.qkv = LoRALinear(
            block.attn.qkv,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        ).to(device)

    # Mask decoder fully trainable (small enough not to break param-efficiency).
    for p in sam.mask_decoder.parameters():
        p.requires_grad = True
