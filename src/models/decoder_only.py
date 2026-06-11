"""Decoder-only fine-tuning.

Image encoder and prompt encoder frozen, mask decoder trainable. Prompt
encoder stays frozen because it only handles deterministic bbox inputs
(~6k params, no benefit). This is MedSAM's documented recipe and the
strongest non-PEFT baseline here.
"""
from __future__ import annotations

from segment_anything.modeling import Sam


def apply_decoder_only(sam: Sam, **_kwargs) -> None:
    """Configure SAM in place for decoder-only fine-tuning."""
    for p in sam.parameters():
        p.requires_grad = False
    for p in sam.mask_decoder.parameters():
        p.requires_grad = True
