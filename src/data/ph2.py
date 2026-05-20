"""PH² dataset — dermoscopy images from ADDI (Univ. of Porto).

200 dermoscopy images, each with a manually-annotated lesion mask. Used as
a *near-OOD* test set for skin lesion segmentation: same modality as ISIC
(dermoscopy) but different acquisition setup (Hospital Pedro Hispano,
Matosinhos, Portugal) — different cameras, lighting, and patient cohort.

Original distribution layout (from the ADDI zip):

    PH2 Dataset images/
    ├── IMD002/
    │   ├── IMD002_Dermoscopic_Image/IMD002.bmp
    │   └── IMD002_lesion/IMD002_lesion.bmp
    ├── IMD003/
    │   ├── ...
    └── ...

This loader walks the directory structure and pairs images with masks
automatically. Place the unzipped `PH2 Dataset images/` folder under
`data/ph2/` (or anywhere — pass the path via the dataset's root argument).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .isic import PIXEL_MEAN, PIXEL_STD, _bbox_from_mask


class PH2(Dataset):
    """PH² dermoscopy dataset.

    Args:
        root: path to the parent directory containing per-image subfolders
            (e.g. `data/ph2/PH2 Dataset images`), OR a directory containing
            preprocessed `images/` and `masks/` subfolders.
        image_size: square size to resize to (typically 1024 for MedSAM).
        bbox_perturb_pixels: max pixel jitter on bbox prompts (0 for eval).
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["test"] = "test",  # accepted for API symmetry; PH² has no splits
        image_size: int = 1024,
        bbox_perturb_pixels: int = 0,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.image_size = image_size
        self.bbox_perturb_pixels = bbox_perturb_pixels

        # Try a few common flat layouts first; fall back to the original
        # per-image folder structure if none match.
        self.items = self._collect_flat()
        if not self.items:
            self.items = self._collect_original()

        if not self.items:
            raise RuntimeError(
                f"No image/mask pairs found under {self.root}. "
                f"Supported layouts:\n"
                f"  (a) {self.root}/images/ + {self.root}/masks/\n"
                f"  (b) {self.root}/trainx/ + {self.root}/trainy/ (Kaggle redistribution)\n"
                f"  (c) {self.root}/IMD###/IMD###_Dermoscopic_Image/IMD###.bmp + .../IMD###_lesion/IMD###_lesion.bmp (ADDI original)"
            )

    def _collect_flat(self) -> list[tuple[Path, Path, str]]:
        """Try multiple flat-folder naming conventions.

        Handles:
          - images/ + masks/
          - trainx/ + trainy/   (e.g. Kaggle 'athina123/ph2dataset')
        Plus several mask filename conventions:
          - IMD002.bmp (paired by stem)
          - IMD002_lesion.bmp (PH²'s standard "lesion" suffix)
          - IMD002_mask.png
        """
        img_dir = None
        msk_dir = None
        for img_name, msk_name in (("images", "masks"), ("trainx", "trainy")):
            i, m = self.root / img_name, self.root / msk_name
            if i.is_dir() and m.is_dir():
                img_dir, msk_dir = i, m
                break
        if img_dir is None:
            return []

        valid_exts = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        items = []
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in valid_exts:
                continue
            stem = img_path.stem  # e.g. "IMD002"
            mask_path = self._find_mask(msk_dir, stem, valid_exts)
            if mask_path is not None:
                items.append((img_path, mask_path, stem))
        return items

    @staticmethod
    def _find_mask(
        msk_dir: Path, image_stem: str, valid_exts: set[str]
    ) -> Path | None:
        """Try mask filename conventions, in order."""
        candidate_stems = (
            image_stem,                 # IMD002.bmp
            f"{image_stem}_lesion",     # IMD002_lesion.bmp (PH² standard)
            f"{image_stem}_mask",       # IMD002_mask.png
            f"{image_stem}_segmentation",
        )
        for stem in candidate_stems:
            for ext in valid_exts:
                p = msk_dir / f"{stem}{ext}"
                if p.exists():
                    return p
        return None

    def _collect_original(self) -> list[tuple[Path, Path, str]]:
        # Look for per-image folders matching IMD\d+
        items = []
        for case_dir in sorted(self.root.iterdir()):
            if not case_dir.is_dir():
                continue
            case_id = case_dir.name
            img_subdir = case_dir / f"{case_id}_Dermoscopic_Image"
            msk_subdir = case_dir / f"{case_id}_lesion"
            if not img_subdir.is_dir() or not msk_subdir.is_dir():
                continue
            img_files = [p for p in img_subdir.iterdir() if p.suffix.lower() == ".bmp"]
            msk_files = [p for p in msk_subdir.iterdir() if p.suffix.lower() == ".bmp"]
            if img_files and msk_files:
                items.append((img_files[0], msk_files[0], case_id))
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        img_path, msk_path, stem = self.items[idx]
        img_pil = Image.open(img_path).convert("RGB")
        msk_pil = Image.open(msk_path).convert("L")
        orig_w, orig_h = img_pil.size

        img_pil = img_pil.resize((self.image_size, self.image_size), Image.BILINEAR)
        msk_pil = msk_pil.resize((self.image_size, self.image_size), Image.NEAREST)

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
