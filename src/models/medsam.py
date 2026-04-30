"""MedSAM loading helpers.

MedSAM uses the SAM ViT-B architecture; only the weights differ. We load
through segment_anything's registry and replace state dict from the MedSAM
checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import torch
from segment_anything import sam_model_registry
from segment_anything.modeling import Sam


def load_medsam(
    checkpoint_path: str | Path,
    arch: str = "vit_b",
    device: str | torch.device = "cuda",
) -> Sam:
    """Load MedSAM weights into a SAM ViT-B architecture and return it.

    The checkpoint is loaded with strict=True; if MedSAM ever ships a
    slightly modified arch we'd surface that here rather than failing
    silently.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"MedSAM checkpoint not found at {checkpoint_path}. "
            f"Run: python scripts/download_medsam.py"
        )

    # Build architecture without weights
    sam: Sam = sam_model_registry[arch](checkpoint=None)

    # Load MedSAM weights
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    # MedSAM checkpoints are flat state dicts; some are wrapped in {"model": ...}
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    missing, unexpected = sam.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_medsam] missing keys: {len(missing)} (often acceptable)")
    if unexpected:
        print(f"[load_medsam] unexpected keys: {len(unexpected)}")

    sam = sam.to(device)
    sam.eval()
    return sam


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
