"""Shared fixtures: repo root on sys.path, a random-init SAM ViT-B on CPU, a fake ISIC tree."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def make_sam():
    """Factory: fresh SAM ViT-B with random weights on CPU (about 2 s each)."""
    from segment_anything import sam_model_registry

    def _make():
        sam = sam_model_registry["vit_b"](checkpoint=None)
        sam.eval()
        return sam

    return _make


@pytest.fixture
def fake_isic_root(tmp_path: Path) -> Path:
    """20 train pairs, 4 val pairs, 8x8 images, ISIC naming conventions."""
    for split, n in (("train", 20), ("val", 4)):
        img_dir = tmp_path / f"{split}_images"
        msk_dir = tmp_path / f"{split}_masks"
        img_dir.mkdir()
        msk_dir.mkdir()
        for i in range(n):
            stem = f"ISIC_{split}_{i:04d}"
            Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8)).save(
                img_dir / f"{stem}.jpg"
            )
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[2:6, 2:6] = 255
            Image.fromarray(mask).save(msk_dir / f"{stem}_segmentation.png")
    return tmp_path
