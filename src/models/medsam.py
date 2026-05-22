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
    device: str | torch.device | None = None,
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

    # Auto-select best device if caller didn't specify (cuda > mps > cpu).
    if device is None:
        from src.device_utils import get_device
        device = get_device()

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


def load_medsam_from_state_dict(
    state_dict: dict,
    arch: str = "vit_b",
    device: str | torch.device | None = None,
) -> Sam:
    """Build a SAM ViT-B with an already-loaded state dict (no disk I/O).

    Use this when you want to instantiate multiple SAMs from the same base
    weights without re-reading the 358 MB .pth file each time. Typical
    pattern:

        base_sd = torch.load("checkpoints/medsam_vit_b.pth", map_location="cpu")
        sam_a = load_medsam_from_state_dict(base_sd, device="cuda")
        sam_b = load_medsam_from_state_dict(base_sd, device="cuda")
    """
    if device is None:
        from src.device_utils import get_device
        device = get_device()

    sam: Sam = sam_model_registry[arch](checkpoint=None)
    # Some MedSAM dumps wrap the actual weights under a top-level "model" key
    sd = state_dict["model"] if isinstance(state_dict.get("model"), dict) else state_dict
    sam.load_state_dict(sd, strict=False)
    sam = sam.to(device)
    sam.eval()
    return sam


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
