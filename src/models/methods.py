"""Adaptation method dispatcher.

Each method's setup logic lives in its own module under src/models/.
This file routes a method name to the corresponding `apply_*` function
and reports parameter counts.

Methods supported:
    zero_shot      — everything frozen (no training, just inference)
    decoder_only   — encoder frozen, mask decoder trainable
    vpt_shallow    — encoder frozen + VPT prompts at input + decoder
    vpt_deep       — encoder frozen + per-layer VPT prompts + decoder
    full_ft        — encoder + decoder trainable
    lora           — (placeholder — to be added)
"""
from __future__ import annotations

from segment_anything.modeling import Sam

from .decoder_only import apply_decoder_only
from .full_ft import apply_full_ft
from .vpt import apply_vpt


def setup_method(sam: Sam, method: str, **kwargs) -> dict:
    """Configure `sam` in place for `method`. Returns parameter count summary."""
    if method == "zero_shot":
        for p in sam.parameters():
            p.requires_grad = False
    elif method == "decoder_only":
        apply_decoder_only(sam, **kwargs)
    elif method == "vpt_shallow":
        apply_vpt(sam, n_prompts=kwargs.get("n_prompts", 10), mode="shallow")
    elif method == "vpt_deep":
        apply_vpt(sam, n_prompts=kwargs.get("n_prompts", 10), mode="deep")
    elif method == "full_ft":
        apply_full_ft(sam, **kwargs)
    elif method == "lora":
        # Imported lazily so the project runs without peft installed for non-LoRA methods.
        from .lora import apply_lora
        apply_lora(sam, **kwargs)
    else:
        raise ValueError(
            f"Unknown method: {method!r}. "
            "Supported: zero_shot, decoder_only, vpt_shallow, vpt_deep, full_ft, lora."
        )

    total = sum(p.numel() for p in sam.parameters())
    trainable = sum(p.numel() for p in sam.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": 100.0 * trainable / max(total, 1),
    }


def encoder_in_grad_path(method: str) -> bool:
    """Whether the encoder forward must be inside the autograd graph.

    True for any method whose trainable parameters live inside the encoder
    (full_ft, vpt_*, lora). False for decoder-only and zero-shot, where we
    can wrap the encoder forward in torch.no_grad() to skip activation
    caching and roughly halve training memory.
    """
    return method in {"full_ft", "vpt_shallow", "vpt_deep", "lora"}
