"""BUSI dataset — Breast UltraSound Images (Cairo Univ., Al-Dhabyani et al. 2020).

780 ultrasound images across 3 classes: benign (487), malignant (210), normal (133).
For each abnormal image there is a binary segmentation mask. Normal images have no
masks and are excluded from segmentation evaluation.

Used as a *far-OOD* test set for our skin-trained models: completely different
imaging modality (greyscale ultrasound vs RGB dermoscopy), different anatomy,
different acquisition physics. Tests whether trained methods preserve MedSAM's
general medical-imaging features.

Original distribution layout (from Kaggle / Cairo Univ. mirror):

    Dataset_BUSI_with_GT/
    ├── benign/
    │   ├── benign (1).png
    │   ├── benign (1)_mask.png
    │   └── ...                  (some images have multiple instance masks:
    │                             benign (X)_mask_1.png, _mask_2.png — we OR them
    │                             together into a single binary mask)
    ├── malignant/
    │   ├── malignant (1).png
    │   ├── malignant (1)_mask.png
    │   └── ...
    └── normal/                  (no masks — skipped)

This loader handles the multi-instance-mask case automatically by OR-ing them.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .isic import PIXEL_MEAN, PIXEL_STD, _bbox_from_mask


class BUSI(Dataset):
    """BUSI breast ultrasound dataset (segmentation subset).

    Args:
        root: path to the directory containing `benign/`, `malignant/`, `normal/`.
        include_normal: include normal-class images (no lesion). Default False
            because they have no masks and don't contribute to segmentation metrics.
        image_size: square size (typically 1024 to match training resolution).
        bbox_perturb_pixels: jitter on bbox prompts (0 for eval).
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["test"] = "test",  # accepted for API symmetry
        image_size: int = 1024,
        bbox_perturb_pixels: int = 0,
        include_normal: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.image_size = image_size
        self.bbox_perturb_pixels = bbox_perturb_pixels

        # BUSI image-naming pattern: "<class> (N).png", mask "<class> (N)_mask[_K].png"
        # We pair images with all matching mask files and OR them at load time.
        classes = ["benign", "malignant"]
        if include_normal:
            classes.append("normal")

        self.items: list[tuple[Path, list[Path], str]] = []
        for cls in classes:
            cls_dir = self.root / cls
            if not cls_dir.is_dir():
                # Some redistributions use "Dataset_BUSI_with_GT/benign" — caller
                # should pass the correct root. Skip missing classes silently.
                continue
            # Index files by stem (without _mask suffix)
            mask_re = re.compile(r"^(.+?)_mask(?:_\d+)?$")
            images = []
            masks_by_stem: dict[str, list[Path]] = {}
            for p in cls_dir.iterdir():
                if p.suffix.lower() != ".png":
                    continue
                m = mask_re.match(p.stem)
                if m:
                    masks_by_stem.setdefault(m.group(1), []).append(p)
                else:
                    images.append(p)
            for img_path in sorted(images):
                stem = img_path.stem
                masks = sorted(masks_by_stem.get(stem, []))
                if not masks and not include_normal:
                    continue
                self.items.append((img_path, masks, f"{cls}_{stem}"))

        if not self.items:
            raise RuntimeError(
                f"No image/mask pairs found under {self.root}. "
                f"Expected BUSI layout with benign/ and malignant/ subdirs containing "
                f"<class> (N).png images and <class> (N)_mask.png masks."
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_paths, stem = self.items[idx]
        img_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img_pil.size

        # Combine multiple instance masks via logical OR
        if mask_paths:
            mask_arrs = []
            for mp in mask_paths:
                m = np.asarray(Image.open(mp).convert("L"), dtype=np.uint8) > 127
                mask_arrs.append(m)
            mask_full = np.any(np.stack(mask_arrs, axis=0), axis=0).astype(np.uint8)
        else:
            mask_full = np.zeros((orig_h, orig_w), dtype=np.uint8)

        img_pil = img_pil.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask_pil = Image.fromarray(mask_full * 255).resize(
            (self.image_size, self.image_size), Image.NEAREST
        )

        img_np = np.asarray(img_pil, dtype=np.float32)
        msk_np = (np.asarray(mask_pil, dtype=np.uint8) > 127).astype(np.uint8)

        bbox = _bbox_from_mask(msk_np, perturb_px=self.bbox_perturb_pixels)

        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
        img_tensor = (img_tensor - PIXEL_MEAN) / PIXEL_STD

        return {
            "image": img_tensor,
            "mask": torch.from_numpy(msk_np),
            "bbox": torch.from_numpy(bbox),
            "image_id": stem,
            "orig_size": (orig_h, orig_w),
        }
