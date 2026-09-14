"""DMID loader: pairs TIFF images with LZW ROI masks by stem, drops alpha, derives tight bboxes, skips images without masks."""

import numpy as np
import pytest
import tifffile
from PIL import Image

from src.data.dmid import DMID


@pytest.fixture
def fake_dmid_root(tmp_path):
    img_dir = tmp_path / "TIFF Images" / "TIFF Images"
    roi_dir = tmp_path / "ROI Masks" / "ROI Masks"
    img_dir.mkdir(parents=True)
    roi_dir.mkdir(parents=True)
    # three images, two with masks (IMG002 has none, like the normal cases)
    for i in (1, 2, 3):
        rgba = np.zeros((40, 30, 4), dtype=np.uint8)
        rgba[..., :3] = 90
        rgba[..., 3] = 255
        Image.fromarray(rgba, "RGBA").save(img_dir / f"IMG{i:03d}.tif")
    for i in (1, 3):
        mask = np.zeros((40, 30, 4), dtype=np.uint8)
        mask[10:20, 5:15, :3] = 255
        mask[..., 3] = 255
        tifffile.imwrite(roi_dir / f"IMG{i:03d}.tif", mask, compression="lzw")
    (img_dir / "Info.txt").write_text("IMG001  G  CIRC  B  10  15  5\n")
    return tmp_path


def test_pairs_only_images_with_masks(fake_dmid_root):
    ds = DMID(fake_dmid_root, image_size=64)
    assert [stem for _, _, stem in ds.items] == ["IMG001", "IMG003"]


def test_item_shapes_and_bbox(fake_dmid_root):
    ds = DMID(fake_dmid_root, image_size=64)
    item = ds[0]
    assert item["image"].shape == (3, 64, 64)
    assert (
        item["mask"].shape == (64, 64) and item["mask"].dtype.is_floating_point is False
    )
    assert item["image_id"] == "IMG001"
    assert item["orig_size"] == (40, 30)
    x1, y1, x2, y2 = item["bbox"].tolist()
    # mask columns 5..14 of 30 -> about 10..31 of 64; rows 10..19 of 40 -> about 16..31 of 64
    assert 8 <= x1 <= 12 and 29 <= x2 <= 33 and 14 <= y1 <= 18 and 29 <= y2 <= 33
    assert int(item["mask"].sum()) > 0


def test_missing_layout_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DMID(tmp_path, image_size=64)


def test_eval_build_dataset_knows_dmid(fake_dmid_root):
    from src.eval import build_dataset

    cfg = {"data": {"root": "data"}, "eval": {"bbox_perturb_pixels": 0}}
    ds = build_dataset(
        cfg, {"name": "dmid", "kind": "dmid", "root": str(fake_dmid_root)}, 64
    )
    assert isinstance(ds, DMID) and len(ds) == 2
