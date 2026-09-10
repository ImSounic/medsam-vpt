"""Adaptation method dispatcher: routes a method name to its apply_* function and reports parameter counts."""

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
        # Lazy import to keep startup light for non-LoRA runs.
        from .lora import apply_lora

        apply_lora(sam, **kwargs)
    elif method == "lora_encoder_only":
        from .lora import apply_lora_encoder_only

        apply_lora_encoder_only(sam, **kwargs)
    else:
        raise ValueError(
            f"Unknown method: {method!r}. "
            "Supported: zero_shot, decoder_only, vpt_shallow, vpt_deep, full_ft, "
            "lora, lora_encoder_only."
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
    """Whether the encoder forward must be inside the autograd graph (true for full_ft, vpt_*, lora)."""
    return method in {"full_ft", "vpt_shallow", "vpt_deep", "lora", "lora_encoder_only"}
