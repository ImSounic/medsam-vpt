"""LoRA fine-tuning for MedSAM.

Uses the peft library to inject low-rank adapters into the SAM image
encoder's attention qkv projections. The decoder is trained alongside
(standard PEFT-on-SAM recipe — decoder is only ~4M params, full training
adds expressiveness without breaking the parameter-efficiency story).

SAM's attention layer has a single fused qkv linear (Linear(768, 3*768))
rather than separate q_proj/k_proj/v_proj. LoRA applied to this linear
produces low-rank updates to all three projections simultaneously, which
is a standard simplification used in SAM-LoRA literature.

Trainable parameter budget (rank=8):
    LoRA A: (8, 768) = 6,144 per block
    LoRA B: (2304, 8) = 18,432 per block
    Per block:                24,576
    12 blocks total:         294,912 LoRA params
    + mask decoder:        4,058,340
    Total trainable:      ~4,353,000  (~4.65% of MedSAM)
"""
from __future__ import annotations

from segment_anything.modeling import Sam


def apply_lora(
    sam: Sam,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    **_kwargs,
) -> None:
    """Configure SAM for LoRA fine-tuning. Modifies sam in place.

    Args:
        sam: the SAM model to configure.
        rank: LoRA rank r. Standard value is 8.
        alpha: LoRA scaling factor. Convention is alpha = 2*rank.
        dropout: LoRA dropout. Keep 0 for small datasets like ISIC.
    """
    # Lazy import so the project runs without peft installed for non-LoRA methods.
    from peft import LoraConfig, inject_adapter_in_model

    # Freeze everything to start
    for p in sam.parameters():
        p.requires_grad = False

    # Inject LoRA into the encoder's attention qkv projections.
    # target_modules=["qkv"] matches sam.image_encoder.blocks.{0..11}.attn.qkv
    # (no qkv modules exist elsewhere in SAM, so this is unambiguous).
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["qkv"],
        lora_dropout=dropout,
        bias="none",
    )
    inject_adapter_in_model(lora_config, sam.image_encoder)

    # peft sets LoRA params (lora_A, lora_B) to requires_grad=True automatically.
    # Defensively re-freeze any non-LoRA encoder params (e.g. the wrapped base
    # qkv weights, pos_embed, layer norms, etc).
    for name, param in sam.image_encoder.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    # Train mask decoder fully — standard SAM-PEFT recipe.
    for p in sam.mask_decoder.parameters():
        p.requires_grad = True

    # Prompt encoder stays frozen (matches every other method in this project).
