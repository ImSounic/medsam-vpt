"""BUSI ultrasound segmentation dataset.

Expected raw layout:
    Dataset_BUSI_with_GT/
      benign/
        benign (1).png
        benign (1)_mask.png
        benign (100)_mask_1.png   # optional extra lesion masks
      malignant/
      normal/
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.isic import PIXEL_MEAN, PIXEL_STD, _bbox_from_mask


class BUSI(Dataset):
    """BUSI breast ultrasound dataset with masks stored next to each image."""

    def __init__(
        self,
        root: str | Path,
        image_size: int = 1024,
        bbox_perturb_pixels: int = 0,
        classes: tuple[str, ...] = ("benign", "malignant", "normal"),
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.image_size = image_size
        self.bbox_perturb_pixels = bbox_perturb_pixels
        self.classes = classes

        if not self.root.is_dir():
            raise FileNotFoundError(f"BUSI root not found: {self.root}")

        self.items: list[tuple[Path, list[Path], str]] = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Expected BUSI class directory: {class_dir}")

            for img_path in sorted(class_dir.glob("*.png")):
                if "_mask" in img_path.stem:
                    continue
                mask_paths = sorted(class_dir.glob(f"{img_path.stem}_mask*.png"))
                if not mask_paths:
                    raise RuntimeError(f"No mask found for BUSI image: {img_path}")
                image_id = f"{class_name}/{img_path.stem}"
                self.items.append((img_path, mask_paths, image_id))

        if not self.items:
            raise RuntimeError(f"No BUSI image/mask pairs found in {self.root}.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_paths, image_id = self.items[idx]

        img_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img_pil.size
        img_pil = img_pil.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_np = np.asarray(img_pil, dtype=np.float32)

        mask_union = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        for mask_path in mask_paths:
            mask_pil = Image.open(mask_path).convert("L")
            mask_pil = mask_pil.resize((self.image_size, self.image_size), Image.NEAREST)
            mask_np = (np.asarray(mask_pil, dtype=np.uint8) > 127).astype(np.uint8)
            mask_union = np.maximum(mask_union, mask_np)

        bbox = _bbox_from_mask(mask_union, perturb_px=self.bbox_perturb_pixels)

        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
        img_tensor = (img_tensor - PIXEL_MEAN) / PIXEL_STD

        return {
            "image": img_tensor,
            "mask": torch.from_numpy(mask_union),
            "bbox": torch.from_numpy(bbox),
            "image_id": image_id,
            "orig_size": (orig_h, orig_w),
        }
