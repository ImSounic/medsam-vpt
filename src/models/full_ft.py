"""Full fine-tuning.

Image encoder + mask decoder both trainable. Prompt encoder stays frozen
(it processes deterministic bbox inputs; training it adds ~6k params for
no measurable benefit and risks destabilising the prompt embedding space).

Memory-heavy: the ~89M-param encoder needs gradients and Adam moments,
which adds ~1 GB on top of the activation cost. On 8 GB this requires
either batch=1 + gradient accumulation, or pushing to Colab T4. Plan to
run this last (after the laptop-feasible methods are locked in).
"""
from __future__ import annotations

from segment_anything.modeling import Sam


def apply_full_ft(sam: Sam, **_kwargs) -> None:
    """Configure SAM in place for full fine-tuning."""
    # Freeze first to be defensive — then explicitly unfreeze.
    for p in sam.parameters():
        p.requires_grad = False
    for p in sam.image_encoder.parameters():
        p.requires_grad = True
    for p in sam.mask_decoder.parameters():
        p.requires_grad = True
    # Prompt encoder stays frozen.
