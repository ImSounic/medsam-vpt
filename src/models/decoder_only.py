"""Decoder-only fine-tuning: image+prompt encoder frozen, mask decoder trainable."""
from __future__ import annotations

from segment_anything.modeling import Sam


def apply_decoder_only(sam: Sam, **_kwargs) -> None:
    """Configure SAM in place for decoder-only fine-tuning."""
    for p in sam.parameters():
        p.requires_grad = False
    for p in sam.mask_decoder.parameters():
        p.requires_grad = True
