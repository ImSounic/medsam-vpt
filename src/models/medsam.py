"""MedSAM loading helpers: SAM ViT-B architecture with MedSAM weights loaded via segment_anything's registry."""

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
    """Load MedSAM weights into a SAM ViT-B architecture and return it (strict=False)."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"MedSAM checkpoint not found at {checkpoint_path}. "
            f"Run: python scripts/download_medsam.py"
        )

    if device is None:
        from src.device_utils import get_device

        device = get_device()

    sam: Sam = sam_model_registry[arch](checkpoint=None)

    state_dict = torch.load(checkpoint_path, map_location="cpu")

    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    missing, unexpected = sam.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_medsam] missing keys: {len(missing)} (usually fine)")
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
    if device is None:
        from src.device_utils import get_device

        device = get_device()

    sam: Sam = sam_model_registry[arch](checkpoint=None)

    sd = (
        state_dict["model"] if isinstance(state_dict.get("model"), dict) else state_dict
    )
    sam.load_state_dict(sd, strict=False)
    sam = sam.to(device)
    sam.eval()
    return sam


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
