"""Probe-set construction and base-feature caching for CKA-aware training; base activations cached once with no grad."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from segment_anything.modeling import Sam

import torch.utils.checkpoint as cp

from src.data.busi import BUSI
from src.data.cbis_ddsm import CBISDDSM
from src.data.isic import ISIC2018, PIXEL_MEAN, PIXEL_STD
from cka.hooks import register_hooks


def _center_bbox(image_size: int, frac: float = 0.5) -> torch.Tensor:
    """Synthetic centered bbox covering frac of the image in each dimension."""
    side = int(image_size * frac)
    x1 = (image_size - side) // 2
    y1 = (image_size - side) // 2
    x2 = x1 + side
    y2 = y1 + side
    return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)


def _load_and_preprocess(img_path: Path, image_size: int) -> torch.Tensor:
    """Mirror the dataset loaders: PIL->RGB->resize->float->ImageNet normalize."""
    pil = Image.open(img_path).convert("RGB").resize(
        (image_size, image_size), Image.BILINEAR
    )
    arr = np.asarray(pil, dtype=np.float32)
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return (t - PIXEL_MEAN) / PIXEL_STD


def build_probe_batch(
    repo_root: Path,
    image_size: int = 1024,
    n_isic: int = 12,
    n_busi: int = 10,
    n_cbis: int = 10,
    probe_seed: int = 42,
    device: torch.device | str = "cpu",
) -> dict:
    """Construct a deterministic multi-modal probe batch."""
    rng = random.Random(probe_seed)
    items: list[tuple[str, Path]] = []  # (image_id, image_path)

    # ISIC train split, disjoint from test_images/
    isic_ds = ISIC2018(
        root=repo_root / "data",
        split="train",
        image_size=image_size,
        bbox_perturb_pixels=0,
    )
    isic_idx = rng.sample(range(len(isic_ds.items)), k=min(n_isic, len(isic_ds.items)))
    for i in isic_idx:
        img_path, _msk_path, stem = isic_ds.items[i]
        items.append((f"isic_{stem}", img_path))

    # BUSI: sample any class; bbox is synthetic so masks don't matter
    busi_ds = BUSI(
        root=repo_root / "data" / "busi",
        image_size=image_size,
        bbox_perturb_pixels=0,
    )
    busi_idx = rng.sample(range(len(busi_ds.items)), k=min(n_busi, len(busi_ds.items)))
    for i in busi_idx:
        img_path, _mask_paths, stem = busi_ds.items[i]
        items.append((f"busi_{stem}", img_path))

    # CBIS-DDSM: use train split so we don't leak into test eval
    try:
        cbis_ds = CBISDDSM(
            root=repo_root / "data" / "cbis-ddsm",
            split="train",
            image_size=image_size,
            bbox_perturb_pixels=0,
        )
        cbis_idx = rng.sample(range(len(cbis_ds.items)), k=min(n_cbis, len(cbis_ds.items)))
        for i in cbis_idx:
            full_path, _mask_paths, stem = cbis_ds.items[i]
            items.append((f"cbis_{stem}", full_path))
    except (FileNotFoundError, RuntimeError) as e:
        # If CBIS train is unavailable, fall back to extra BUSI images to keep probe size consistent.
        print(f"[probe] CBIS-DDSM train not available ({e}); "
              f"filling {n_cbis} probe slots with extra BUSI images.")
        avail = [i for i in range(len(busi_ds.items)) if i not in busi_idx]
        extra_idx = rng.sample(avail, k=min(n_cbis, len(avail)))
        for i in extra_idx:
            img_path, _mask_paths, stem = busi_ds.items[i]
            items.append((f"busi_extra_{stem}", img_path))

    # Stack into a batch tensor
    images = []
    image_ids = []
    for image_id, img_path in items:
        images.append(_load_and_preprocess(img_path, image_size))
        image_ids.append(image_id)

    image_tensor = torch.stack(images).to(device)
    bbox_tensor = torch.stack(
        [_center_bbox(image_size) for _ in items]
    ).to(device)

    return {
        "image": image_tensor,    # (N, 3, H, W)
        "bbox":  bbox_tensor,     # (N, 4)
        "image_id": image_ids,
    }


def _encoder_forward_checkpointed(encoder, x: torch.Tensor) -> torch.Tensor:
    """encoder(x) with per-block gradient checkpointing on the probe encoder to bound memory."""
    if hasattr(encoder, "base") and hasattr(encoder, "_add_prompts"):
        # VPT wrapper: delegate to its own checkpointing path
        encoder.gradient_checkpointing = True
        try:
            return encoder(x)
        finally:
            encoder.gradient_checkpointing = False
    x = encoder.patch_embed(x)
    if encoder.pos_embed is not None:
        x = x + encoder.pos_embed
    for blk in encoder.blocks:
        x = cp.checkpoint(blk, x, use_reentrant=False)
    x = encoder.neck(x.permute(0, 3, 1, 2))
    return x


@torch.no_grad()
def cache_base_activations(
    base_sam: Sam,
    probe_batch: dict,
    layer_names: Iterable[str],
    encoder_chunk: int = 4,
) -> dict[str, torch.Tensor]:
    """Forward the probe through base MedSAM once and capture activations, cached with no grad."""
    base_sam.eval()
    with register_hooks(base_sam, layer_names, detach=True, accumulate=True) as hh:
        _run_probe_forward(base_sam, probe_batch, hook_handle=hh,
                           encoder_chunk=encoder_chunk)
        # Clone so later forwards cannot mutate the cached tensors.
        return {name: act.clone() for name, act in hh.stacked().items()}


def _run_probe_forward(
    sam: Sam,
    probe_batch: dict,
    *,
    hook_handle=None,
    encoder_chunk: int = 4,
    use_grad_checkpoint: bool = False,
) -> None:
    """Forward pass through the full SAM pipeline on the probe batch."""
    images = probe_batch["image"]
    bboxes = probe_batch["bbox"]
    B = images.shape[0]

    # Image encoder micro-batched to bound peak activation memory
    emb_chunks = []
    for start in range(0, B, encoder_chunk):
        chunk = images[start : start + encoder_chunk]
        if use_grad_checkpoint:
            emb_chunks.append(_encoder_forward_checkpointed(sam.image_encoder, chunk))
        else:
            emb_chunks.append(sam.image_encoder(chunk))
    image_emb = torch.cat(emb_chunks, dim=0)  # (B, 256, H/16, W/16)

    # Per-sample prompt encoder plus mask decoder
    for i in range(B):
        sparse, dense = sam.prompt_encoder(
            points=None,
            boxes=bboxes[i : i + 1],
            masks=None,
        )
        sam.mask_decoder(
            image_embeddings=image_emb[i : i + 1],
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )


def run_current_probe_forward(
    sam: Sam,
    probe_batch: dict,
    hook_handle=None,
    encoder_chunk: int = 4,
    use_grad_checkpoint: bool = True,
) -> None:
    """Forward the probe through the current (trainable) model for CKA."""
    if hook_handle is not None:
        hook_handle.clear()
    _run_probe_forward(sam, probe_batch, hook_handle=hook_handle,
                       encoder_chunk=encoder_chunk,
                       use_grad_checkpoint=use_grad_checkpoint)
