"""Full fine-tuning.

Image encoder + mask decoder trainable, prompt encoder frozen (deterministic
bbox inputs, ~6k params, no benefit).

Memory-heavy: ~89M-param encoder needs gradients + Adam moments, ~1 GB on top
of activations. On 8 GB use batch=1 + grad accumulation or run on Colab T4.
"""
from __future__ import annotations

from segment_anything.modeling import Sam


def apply_full_ft(sam: Sam, **_kwargs) -> None:
    """Configure SAM in place for full fine-tuning."""
    # Freeze first, then unfreeze what we train.
    for p in sam.parameters():
        p.requires_grad = False
    for p in sam.image_encoder.parameters():
        p.requires_grad = True
    for p in sam.mask_decoder.parameters():
        p.requires_grad = True
    # Prompt encoder stays frozen.
