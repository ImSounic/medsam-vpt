"""CBIS-DDSM mammography far-OOD test set; mask PatientID has a trailing abnormality index that the full-mammogram PID drops, and multiple masks per mammogram are OR'd into one lesion mask."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .isic import PIXEL_MEAN, PIXEL_STD, _bbox_from_mask


class CBISDDSM(Dataset):
    """CBIS-DDSM mammography dataset."""

    def __init__(
        self,
        root: str | Path,
        split: Literal["test", "train"] = "test",
        image_size: int = 1024,
        bbox_perturb_pixels: int = 0,
        abnormality_type: Literal["all", "mass", "calc"] = "all",
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.image_size = image_size
        self.bbox_perturb_pixels = bbox_perturb_pixels

        dinfo_path = self.root / "csv" / "dicom_info.csv"
        if not dinfo_path.exists():
            raise FileNotFoundError(
                f"Expected {dinfo_path}. CBIS-DDSM root should contain csv/dicom_info.csv"
                f" and jpeg/<SeriesUID>/*.jpg files."
            )

        # Use stdlib csv, not pandas: pandas + PyTorch on Windows can hit an OpenMP DLL conflict that silently kills the process.
        split_tag = "-Test_" if split == "test" else "-Training_"
        fulls: list[tuple[str, str]] = []  # (PatientID, image_path)
        masks_rows: list[tuple[str, str]] = []
        with open(dinfo_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("PatientID") or ""
                if split_tag not in pid:
                    continue
                if abnormality_type == "mass" and not pid.startswith("Mass-"):
                    continue
                if abnormality_type == "calc" and not pid.startswith("Calc-"):
                    continue
                desc = row.get("SeriesDescription") or ""
                ipath = row.get("image_path") or ""
                if not ipath:
                    continue
                if desc == "full mammogram images":
                    fulls.append((pid, ipath))
                elif desc == "ROI mask images":
                    masks_rows.append((pid, ipath))

        # Build index: full_mammogram_PID -> list[mask_path].
        mask_by_pid: dict = {}
        for mpid, mpath in masks_rows:
            # Drop trailing "_<digits>" to recover the full mammogram PID.
            parts = mpid.rsplit("_", 1)
            base_pid = parts[0] if len(parts) == 2 and parts[1].isdigit() else mpid
            mask_by_pid.setdefault(base_pid, []).append(self._resolve_path(mpath))

        # Pair each full mammogram with its mask(s).
        self.items = []
        for full_pid, fpath in fulls:
            mask_paths = mask_by_pid.get(full_pid)
            if not mask_paths:
                continue
            self.items.append((self._resolve_path(fpath), mask_paths, full_pid))

        if not self.items:
            raise RuntimeError(
                f"No mammogram/mask pairs found under {self.root} for split={split}. "
                f"Check that jpeg/ and csv/ are present and PatientIDs contain '{split_tag}'."
            )

    def _resolve_path(self, csv_path: str) -> Path:
        """Map dicom_info.csv image_path to <root>/jpeg/<UID>/X.jpg."""
        s = csv_path.strip()
        # Strip prefixes used by different Kaggle redistributions.
        for prefix in ("CBIS-DDSM/", "cbis-ddsm/"):
            if s.startswith(prefix):
                s = s[len(prefix) :]
                break
        return self.root / s

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        full_path, mask_paths, stem = self.items[idx]

        img_pil = Image.open(full_path).convert("RGB")
        orig_w, orig_h = img_pil.size

        # OR all instance masks into one binary mask.
        combined = np.zeros((orig_h, orig_w), dtype=bool)
        for mp in mask_paths:
            m = np.array(Image.open(mp).convert("L"))
            # Some masks differ slightly in size; resize to match the image.
            if m.shape != (orig_h, orig_w):
                m = np.array(Image.fromarray(m).resize((orig_w, orig_h), Image.NEAREST))
            combined |= m > 127

        img_pil = img_pil.resize((self.image_size, self.image_size), Image.BILINEAR)
        msk_pil = Image.fromarray(combined.astype(np.uint8) * 255).resize(
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
