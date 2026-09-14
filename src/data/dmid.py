"""DMID mammography held-out set (evaluation only): 8-bit RGBA TIFF mammograms paired by stem with LZW-compressed binary ROI masks; images without a mask (normals) are skipped."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import tifffile
import torch
from PIL import Image
from torch.utils.data import Dataset

from .isic import PIXEL_MEAN, PIXEL_STD, _bbox_from_mask


class DMID(Dataset):
    """DMID dataset: data/dmid/TIFF Images/TIFF Images/IMG###.tif + ROI Masks/ROI Masks/IMG###.tif."""

    def __init__(
        self,
        root: str | Path,
        split: Literal["test"] = "test",  # API symmetry; DMID has no splits
        image_size: int = 1024,
        bbox_perturb_pixels: int = 0,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.image_size = image_size
        self.bbox_perturb_pixels = bbox_perturb_pixels

        img_dir = self._find_dir("TIFF Images")
        roi_dir = self._find_dir("ROI Masks")
        if img_dir is None or roi_dir is None:
            raise FileNotFoundError(
                f"Expected {self.root}/'TIFF Images'[/'TIFF Images'] and "
                f"{self.root}/'ROI Masks'[/'ROI Masks'] containing IMG###.tif files."
            )

        self.items: list[tuple[Path, Path, str]] = []
        for img_path in sorted(img_dir.glob("*.tif")):
            mask_path = roi_dir / img_path.name
            if mask_path.exists():
                self.items.append((img_path, mask_path, img_path.stem))
        if not self.items:
            raise RuntimeError(f"No image/mask pairs found under {self.root}.")

    def _find_dir(self, name: str) -> Path | None:
        """The Kaggle zip nests each folder inside a folder of the same name."""
        nested = self.root / name / name
        flat = self.root / name
        if nested.is_dir():
            return nested
        if flat.is_dir():
            return flat
        return None

    @staticmethod
    def _read_mask(path: Path) -> np.ndarray:
        arr = tifffile.imread(path)
        if arr.ndim == 3:
            arr = arr[..., 0]
        return (arr > 127).astype(np.uint8)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path, stem = self.items[idx]
        img_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img_pil.size

        mask_full = self._read_mask(mask_path)
        if mask_full.shape != (orig_h, orig_w):
            mask_full = (
                np.array(
                    Image.fromarray(mask_full * 255).resize(
                        (orig_w, orig_h), Image.NEAREST
                    )
                )
                > 127
            )
            mask_full = mask_full.astype(np.uint8)

        img_pil = img_pil.resize((self.image_size, self.image_size), Image.BILINEAR)
        msk_pil = Image.fromarray(mask_full * 255).resize(
            (self.image_size, self.image_size), Image.NEAREST
        )
        img_np = np.asarray(img_pil, dtype=np.float32)
        msk_np = (np.asarray(msk_pil, dtype=np.uint8) > 127).astype(np.uint8)

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
