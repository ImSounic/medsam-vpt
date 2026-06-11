"""ISIC 2018 Task 1 (Lesion Boundary Segmentation) dataset.

Item dict: image (3,H,W float32, ImageNet-normalised), mask (H,W uint8, 0/1),
bbox (4, float32 [x1,y1,x2,y2] in image_size coords), image_id, orig_size (H,W).
bbox comes from the GT mask; jittered during training, tight box at eval.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# MedSAM uses ImageNet-style normalisation.
PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)


def _bbox_from_mask(
    mask: np.ndarray,
    perturb_px: int = 0,
    random_perturb: bool = False,
) -> np.ndarray:
    """Tight bbox around mask>0, optionally perturbed. Returns [x1,y1,x2,y2]
    inclusive in mask coords; full image if mask is empty.

    perturb_px: max +/- displacement per corner (0 = exact tight box).
    random_perturb: if True, sample actual magnitude from uniform [0, perturb_px]
        per call (wide prompt-quality distribution); if False, use perturb_px directly.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        H, W = mask.shape
        return np.array([0, 0, W - 1, H - 1], dtype=np.float32)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    if perturb_px > 0:
        rng = np.random.default_rng()
        actual = int(rng.integers(0, perturb_px + 1)) if random_perturb else perturb_px
        if actual > 0:
            dx = rng.integers(-actual, actual + 1, size=2)
            dy = rng.integers(-actual, actual + 1, size=2)
            x1 = max(0, x1 + dx[0])
            y1 = max(0, y1 + dy[0])
            x2 = min(mask.shape[1] - 1, x2 + dx[1])
            y2 = min(mask.shape[0] - 1, y2 + dy[1])
    return np.array([x1, y1, x2, y2], dtype=np.float32)


class ISIC2018(Dataset):
    """ISIC 2018 Task 1 segmentation dataset.

    root has {split}_images and {split}_masks subdirs. split in train/val/test.
    image_size resize target (1024 for native MedSAM). bbox_perturb_pixels = max
    jitter (0 = exact). random_perturb: sample jitter magnitude per item (training).
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test"] = "test",
        image_size: int = 1024,
        bbox_perturb_pixels: int = 0,
        random_perturb: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.bbox_perturb_pixels = bbox_perturb_pixels
        self.random_perturb = random_perturb

        img_dir = self.root / f"{split}_images"
        msk_dir = self.root / f"{split}_masks"
        if not img_dir.is_dir() or not msk_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {img_dir} and {msk_dir}. "
                f"Check that ISIC 2018 is unzipped under {self.root}."
            )

        # Pair images and masks by stem; ignore any non-image files.
        img_files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() == ".jpg")
        self.items: list[tuple[Path, Path, str]] = []
        missing = 0
        for img_path in img_files:
            stem = img_path.stem
            msk_path = msk_dir / f"{stem}_segmentation.png"
            if not msk_path.exists():
                missing += 1
                continue
            self.items.append((img_path, msk_path, stem))

        if not self.items:
            raise RuntimeError(f"No image/mask pairs found in {self.root} / {split}.")
        if missing > 0:
            print(f"[ISIC2018:{split}] warning: {missing} images had no matching mask")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        img_path, msk_path, stem = self.items[idx]

        img_pil = Image.open(img_path).convert("RGB")
        msk_pil = Image.open(msk_path).convert("L")
        orig_w, orig_h = img_pil.size

        img_pil = img_pil.resize((self.image_size, self.image_size), Image.BILINEAR)
        msk_pil = msk_pil.resize((self.image_size, self.image_size), Image.NEAREST)

        img_np = np.asarray(img_pil, dtype=np.float32)  # (H, W, 3) in [0, 255]
        msk_np = (np.asarray(msk_pil, dtype=np.uint8) > 127).astype(np.uint8)

        bbox = _bbox_from_mask(
            msk_np,
            perturb_px=self.bbox_perturb_pixels,
            random_perturb=self.random_perturb,
        )

        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
        img_tensor = (img_tensor - PIXEL_MEAN) / PIXEL_STD

        return {
            "image": img_tensor,
            "mask": torch.from_numpy(msk_np),
            "bbox": torch.from_numpy(bbox),
            "image_id": stem,
            "orig_size": (orig_h, orig_w),
        }


def isic_collate(batch: list[dict]) -> dict:
    """Collate that keeps metadata fields as lists."""
    out = {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "bbox": torch.stack([b["bbox"] for b in batch]),
        "image_id": [b["image_id"] for b in batch],
        "orig_size": [b["orig_size"] for b in batch],
    }
    return out
