"""Probe-set construction and base-feature caching for CKA-aware training.

We need a small, fixed batch of images that gets forwarded through both the
base MedSAM model (frozen) and the current trainable model at every training
step, so we can compute CKA between their decoder activations.

Design choices:

  - Probe is constructed **once** at training startup. The same 32 images are
    used at every step throughout training. Reproducible via `probe_seed`.

  - Multi-modal composition: 12 ISIC train + 10 BUSI + 10 CBIS-DDSM (train
    splits, disjoint from any test data we evaluate on). This tells the
    optimizer "preserve decoder structure on dermoscopy AND ultrasound AND
    mammography images" — directly targeting the far-OOD modalities that
    fine-tuning otherwise breaks.

  - Base-model features are **cached once**. We forward the probe through the
    frozen base MedSAM at startup, capture activations at the hooked layers,
    and stash them in a dict. Every subsequent training step only needs to
    forward the probe through the *current* (trainable) model — the base
    activations are already in memory.

  - We use **fake bounding boxes** for the probe (image-center 50% box) because
    we don't have ground-truth masks for the BUSI/CBIS train images we're
    sampling, and the goal here is to preserve decoder *representational
    structure*, not segmentation accuracy. The exact bbox doesn't matter as
    long as both models see the same one.

Public API:

  build_probe_batch(...)        -> dict with "image", "bbox", "image_id"
  cache_base_activations(...)   -> dict[layer_name -> Tensor]
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from segment_anything.modeling import Sam

from src.data.busi import BUSI
from src.data.cbis_ddsm import CBISDDSM
from src.data.isic import ISIC2018, PIXEL_MEAN, PIXEL_STD
from cka.hooks import register_hooks


def _center_bbox(image_size: int, frac: float = 0.5) -> torch.Tensor:
    """Synthetic centered bbox covering `frac` of the image in each dimension."""
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
    """Construct a deterministic multi-modal probe batch.

    Returns a dict with:
        image:     (N, 3, H, W) float32 — ImageNet-normalised
        bbox:      (N, 4)       float32 — synthetic centered boxes
        image_id:  list[str]    — provenance strings (modality_<id>)
    where N = n_isic + n_busi + n_cbis.

    Picks images by **deterministic random index** into each dataset's items
    list, using `probe_seed`. Same probe_seed always returns the same images.
    """
    rng = random.Random(probe_seed)
    items: list[tuple[str, Path]] = []  # (image_id, image_path)

    # ISIC train (use the train split — disjoint from test_images/)
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

    # BUSI — sample any class with masks; bbox is synthetic so we don't need them
    busi_ds = BUSI(
        root=repo_root / "data" / "busi",
        image_size=image_size,
        bbox_perturb_pixels=0,
    )
    busi_idx = rng.sample(range(len(busi_ds.items)), k=min(n_busi, len(busi_ds.items)))
    for i in busi_idx:
        img_path, _mask_paths, stem = busi_ds.items[i]
        items.append((f"busi_{stem}", img_path))

    # CBIS-DDSM — use train split so we don't leak into our test eval
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
        # If CBIS train isn't available, fall back to extra BUSI images so the
        # probe size stays consistent and training doesn't crash.
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


@torch.no_grad()
def cache_base_activations(
    base_sam: Sam,
    probe_batch: dict,
    layer_names: Iterable[str],
) -> dict[str, torch.Tensor]:
    """Forward the probe through base MedSAM once, capture activations.

    The captured tensors are detached (no graph) and ready to use as fixed
    reference targets in the CKA loss for every training step.

    Args:
        base_sam: frozen base MedSAM (eval mode).
        probe_batch: output of `build_probe_batch`.
        layer_names: which layers to capture; must match the names used for
            hooking the trainable model so we compare like-for-like.

    Returns:
        dict mapping layer_name -> Tensor (batch-first, no grad).
    """
    base_sam.eval()
    with register_hooks(base_sam, layer_names, detach=True) as hh:
        _run_probe_forward(base_sam, probe_batch)
        # Clone so we don't keep references to internal tensors that might be
        # mutated by later forward passes.
        return {name: act.clone() for name, act in hh.activations.items()}


def _run_probe_forward(sam: Sam, probe_batch: dict) -> None:
    """Forward pass through the full SAM pipeline on the probe batch.

    Runs image_encoder, then for each image: prompt_encoder + mask_decoder.
    This matches the per-sample iteration used in `src/train.forward_with_prompt`,
    so the layer activations captured here exactly correspond to what would be
    captured during normal training-time forwards.
    """
    images = probe_batch["image"]
    bboxes = probe_batch["bbox"]

    image_emb = sam.image_encoder(images)
    B = images.shape[0]
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


def run_current_probe_forward(sam: Sam, probe_batch: dict) -> None:
    """Forward the probe through the *current* (trainable) model.

    Use during training: hooks installed on `sam` will populate their
    activation dict. The caller is responsible for setting up hooks beforehand
    and reading from `hook_handle.activations` after this call returns.
    """
    _run_probe_forward(sam, probe_batch)
