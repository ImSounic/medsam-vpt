"""LoRA fine-tuning for MedSAM, including encoder-only variants."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
from segment_anything.modeling import Sam

_BLOCK_LINEAR_TARGETS = {
    "qkv": "attn.qkv",
    "proj": "attn.proj",
    "mlp_lin1": "mlp.lin1",
    "mlp_lin2": "mlp.lin2",
}
_TARGET_PRESETS = {
    "qkv": ("qkv",),
    "all": ("qkv", "proj", "mlp_lin1", "mlp_lin2"),
}


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
    train_mask_decoder: bool = True,
    target_modules: str | Iterable[str] = "qkv",
    **_kwargs,
) -> None:
    """Configure SAM for LoRA in place."""
    for p in sam.parameters():
        p.requires_grad = False

    device = next(sam.parameters()).device

    if isinstance(target_modules, str):
        selected = _TARGET_PRESETS.get(target_modules, (target_modules,))
    else:
        selected = tuple(target_modules)

    invalid = [name for name in selected if name not in _BLOCK_LINEAR_TARGETS]
    if invalid:
        valid = ", ".join(sorted(_BLOCK_LINEAR_TARGETS))
        raise ValueError(
            f"Unknown LoRA target_modules: {invalid}. Valid names: {valid}."
        )

    def _replace_linear(root: nn.Module, dotted_name: str) -> None:
        parts = dotted_name.split(".")
        parent = root
        for part in parts[:-1]:
            parent = getattr(parent, part)
        leaf = parts[-1]
        base = getattr(parent, leaf)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"Expected nn.Linear at {dotted_name}, found {type(base)}")
        setattr(
            parent,
            leaf,
            LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout).to(device),
        )

    for block in sam.image_encoder.blocks:
        for key in selected:
            _replace_linear(block, _BLOCK_LINEAR_TARGETS[key])

    if train_mask_decoder:
        for p in sam.mask_decoder.parameters():
            p.requires_grad = True


def apply_lora_encoder_only(sam: Sam, **kwargs) -> None:
    """LoRA on the image encoder only; the mask decoder stays frozen."""
    kwargs.pop("train_mask_decoder", None)
    apply_lora(sam, train_mask_decoder=False, **kwargs)
    for p in sam.mask_decoder.parameters():
        p.requires_grad = False
