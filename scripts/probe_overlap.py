"""Overlap of the CKA probe with the evaluation sets, and its effect on reported Dice.

BUSI has no official split, so the 16 ultrasound probe images are drawn from the same
647 images that are evaluated; CBIS-DDSM probe images come from the training split, but
one probe patient also has test images. For every per-image result file this writes the
mean Dice with and without the overlapping images.

Usage: python scripts/probe_overlap.py [--out results/accv/probe_overlap.csv]
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PATTERNS = (
    "results/accv/raw_*/*_{ds}_per_image.csv",
    "bbox_robustness/results*/per_image/*_{ds}_pm20.csv",
)
_PATIENT = re.compile(r"P_\d+")


def probe_ids(probe_seed: int = 42, n_busi: int = 16, n_cbis: int = 16) -> list[str]:
    """Probe image ids exactly as training builds them (images are not loaded)."""
    import cka.probe as probe

    real = probe._load_and_preprocess
    probe._load_and_preprocess = lambda path, size: probe.torch.zeros(3, 1, 1)
    try:
        batch = probe.build_probe_batch(
            REPO_ROOT, n_isic=0, n_busi=n_busi, n_cbis=n_cbis, probe_seed=probe_seed
        )
    finally:
        probe._load_and_preprocess = real
    return list(batch["image_id"])


def overlap_masks(ids: list[str]) -> dict[str, callable]:
    """Per dataset, a predicate on evaluation image ids that marks overlap with the probe."""
    busi = {i[len("busi_") :] for i in ids if i.startswith("busi_")}
    cbis_patients = {
        _PATIENT.search(i).group() for i in ids if i.startswith("cbis_") and _PATIENT.search(i)
    }

    def cbis(s: pd.Series) -> pd.Series:
        return s.astype(str).str.extract(r"(P_\d+)")[0].isin(cbis_patients)

    return {"busi": lambda s: s.astype(str).isin(busi), "cbis_ddsm": cbis}


def summarise(masks: dict) -> pd.DataFrame:
    rows = []
    for ds, is_overlap in masks.items():
        for pat in PATTERNS:
            for f in sorted(glob.glob(str(REPO_ROOT / pat.format(ds=ds)))):
                if "/._" in f:
                    continue
                d = pd.read_csv(f)
                col = "dice" if "dice" in d else "dice_mean"
                if "image_id" not in d or col not in d:
                    continue
                m = is_overlap(d["image_id"])
                rows.append(
                    {
                        "dataset": ds,
                        "file": str(Path(f).relative_to(REPO_ROOT)),
                        "regularised": "cka" in Path(f).name,
                        "n": len(d),
                        "n_overlap": int(m.sum()),
                        "dice_all": d[col].mean(),
                        "dice_excl": d.loc[~m, col].mean(),
                    }
                )
    out = pd.DataFrame(rows)
    out["delta"] = out["dice_excl"] - out["dice_all"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results/accv/probe_overlap.csv")
    args = ap.parse_args()
    df = summarise(overlap_masks(probe_ids()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    for (ds, reg), g in df.groupby(["dataset", "regularised"]):
        print(
            f"{ds:10s} regularised={reg!s:5s} files={len(g):3d} overlap={g.n_overlap.max():2d} "
            f"max|delta|={g.delta.abs().max():.4f} mean delta={g.delta.mean():+.4f}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
