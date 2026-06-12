"""Full fine-tuning: image encoder + mask decoder trainable, prompt encoder frozen."""

from __future__ import annotations

from segment_anything.modeling import Sam


def apply_full_ft(sam: Sam, **_kwargs) -> None:
    for p in sam.parameters():
        p.requires_grad = False
    for p in sam.image_encoder.parameters():
        p.requires_grad = True
    for p in sam.mask_decoder.parameters():
        p.requires_grad = True
    # Prompt encoder stays frozen.
