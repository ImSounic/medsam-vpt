"""Download the MedSAM ViT-B checkpoint into checkpoints/medsam_vit_b.pth.

Pulls from Zenodo, the academic mirror linked from the official
bowang-lab/MedSAM repo. ~358 MB. The earlier HF path I tried
(wanglab/medsam-vit-base) doesn't host the segment_anything-format .pth
file we need; the HF Transformers-format weights live there but require
a different loading path.
"""
from __future__ import annotations

import sys
from pathlib import Path

URL = "https://zenodo.org/records/10689643/files/medsam_vit_b.pth"
TARGET_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
TARGET = TARGET_DIR / "medsam_vit_b.pth"


def main() -> int:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET.exists():
        size_mb = TARGET.stat().st_size / 1024 / 1024
        if size_mb > 100:  # sanity: real weights are ~358 MB
            print(f"[skip] {TARGET} already exists ({size_mb:.1f} MB)")
            return 0
        print(f"[warn] {TARGET} exists but only {size_mb:.1f} MB — re-downloading")
        TARGET.unlink()

    print(f"[download] {URL}")
    print(f"[download] -> {TARGET}")

    try:
        import requests
        from tqdm import tqdm
    except ImportError as e:
        print(f"[err] missing dependency: {e}. Run: pip install -r requirements.txt")
        return 1

    with requests.get(URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(TARGET, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024,
            desc="medsam_vit_b.pth"
        ) as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    size_mb = TARGET.stat().st_size / 1024 / 1024
    print(f"[done] saved {TARGET} ({size_mb:.1f} MB)")

    if size_mb < 100:
        print(f"[warn] file is only {size_mb:.1f} MB; that's too small. "
              "Check the URL and try again.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
