"""Decoder-only fine-tuning.

Image encoder frozen. Prompt encoder frozen (it processes deterministic
bbox inputs and adapting it would mean ~6k extra params with negligible
benefit). Mask decoder trainable.

This is MedSAM's documented fine-tuning recipe and the strongest
non-PEFT baseline for the comparison.
"""
from __future__ import annotations

from segment_anything.modeling import Sam


def apply_decoder_only(sam: Sam, **_kwargs) -> None:
    """Configure SAM in place for decoder-only fine-tuning."""
    for p in sam.parameters():
        p.requires_grad = False
    for p in sam.mask_decoder.parameters():
        p.requires_grad = True
