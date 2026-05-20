"""Qualitative prediction visualizations.

Produces figures showing input image, ground-truth mask, model prediction,
and an error breakdown (TP / FP / FN) per example. One figure per
(method, dataset) pair, saved to results/figures/qualitative/.

Usage:
    python scripts/visualize_predictions.py                          # all methods × all datasets
    python scripts/visualize_predictions.py --method lora full_ft    # only those methods
    python scripts/visualize_predictions.py --dataset busi cbis_ddsm # only those datasets
    python scripts/visualize_predictions.py --n 6                    # 6 examples per dataset
    python scripts/visualize_predictions.py --strategy spread        # easy/medium/hard examples by Dice

Strategies for sample selection (--strategy):
    first    — first N items in the dataset (default, deterministic, fast)
    spread   — examples spanning best/mid/worst Dice from results/raw/<run>_<ds>_per_image.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Make the repo root importable so `from src...` works when this script is
# run directly (i.e. `python scripts/visualize_predictions.py`).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.data.busi import BUSI
from src.data.cbis_ddsm import CBISDDSM
from src.data.isic import PIXEL_MEAN, PIXEL_STD, ISIC2018
from src.data.ph2 import PH2
from src.eval import predict_batch
from src.models.medsam import load_medsam
from src.models.methods import setup_method

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "results" / "figures" / "qualitative"

METHODS = {
    "zero_shot":    {"checkpoint": None,                                          "label": "Zero-shot"},
    "decoder_only": {"checkpoint": "checkpoints/runs/decoder_only_seed0/best.pth", "label": "Decoder-only FT"},
    "vpt_shallow":  {"checkpoint": "checkpoints/runs/vpt_shallow_seed0/best.pth",  "label": "VPT-shallow"},
    "vpt_deep":     {"checkpoint": "checkpoints/runs/vpt_deep_seed0/best.pth",     "label": "VPT-deep"},
    "lora":         {"checkpoint": "checkpoints/runs/lora_seed0/best.pth",         "label": "LoRA"},
    "full_ft":      {"checkpoint": "checkpoints/runs/full_ft_seed0/best.pth",      "label": "Full FT"},
}

DATASET_BUILDERS = {
    "isic":      lambda: ISIC2018(root=REPO_ROOT / "data", split="test", image_size=1024),
    "ph2":       lambda: PH2(root=REPO_ROOT / "data" / "ph2", image_size=1024),
    "busi":      lambda: BUSI(root=REPO_ROOT / "data" / "busi", image_size=1024),
    "cbis_ddsm": lambda: CBISDDSM(root=REPO_ROOT / "data" / "cbis-ddsm", split="test", image_size=1024),
}

DATASET_TO_CSV_NAME = {
    "isic":      "isic2018_test",
    "ph2":       "ph2",
    "busi":      "busi",
    "cbis_ddsm": "cbis_ddsm",
}


# ----------------------------------------------------------------------------
# Sample selection
# ----------------------------------------------------------------------------
def pick_indices_first(dataset, n: int) -> list[int]:
    """First N items deterministically."""
    return list(range(min(n, len(dataset))))


def pick_indices_spread(run_name: str, ds_csv_name: str, dataset, n: int) -> list[int]:
    """Pick N indices spanning the Dice quality distribution (best, ..., worst)."""
    csv_path = REPO_ROOT / "results" / "raw" / f"{run_name}_{ds_csv_name}_per_image.csv"
    if not csv_path.exists():
        print(f"  [spread] per-image csv not found ({csv_path}); falling back to first-N")
        return pick_indices_first(dataset, n)
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((r["image_id"], float(r["dice"])))
    if not rows:
        return pick_indices_first(dataset, n)
    # Sort high → low Dice
    rows.sort(key=lambda x: x[1], reverse=True)
    # Pick evenly spaced indices
    idxs = np.linspace(0, len(rows) - 1, n).astype(int).tolist()
    target_ids = {rows[i][0] for i in idxs}
    # Now find those IDs in the dataset
    out = []
    for di, item in enumerate(getattr(dataset, "items", [])):
        # ds.items is a list of tuples whose last element is image_id
        sid = item[-1] if isinstance(item, tuple) else None
        if sid in target_ids:
            out.append(di)
        if len(out) >= n:
            break
    if not out:
        return pick_indices_first(dataset, n)
    return out


# ----------------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------------
def denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert preprocessed (3,H,W) tensor back to a uint8 RGB array."""
    arr = img_tensor.cpu().clone() * PIXEL_STD + PIXEL_MEAN
    arr = arr.permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8)
    return arr


def overlay_mask(img: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = img.astype(np.float32).copy()
    c = np.array(color, dtype=np.float32)
    for k in range(3):
        out[..., k] = np.where(mask, out[..., k] * (1 - alpha) + c[k] * alpha, out[..., k])
    return out.clip(0, 255).astype(np.uint8)


def error_breakdown(img: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """TP = green, FP = red, FN = blue."""
    out = img.copy()
    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt
    out = overlay_mask(out, tp, (0, 200, 0), alpha=0.5)
    out = overlay_mask(out, fp, (220, 30, 30), alpha=0.55)
    out = overlay_mask(out, fn, (30, 60, 220), alpha=0.55)
    return out


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    dice = float((2.0 * inter) / max(denom, 1))
    iou = float(inter / max(union, 1))
    return dice, iou


# ----------------------------------------------------------------------------
# Per-(method, dataset) figure
# ----------------------------------------------------------------------------
@torch.no_grad()
def render_figure(
    method_name: str,
    method_cfg: dict,
    dataset_name: str,
    dataset,
    n_samples: int,
    strategy: str,
    device: str,
    out_path: Path,
) -> None:
    label = method_cfg["label"]
    print(f"[viz] {label} × {dataset_name} -> {out_path.name}")

    # Set up SAM with the right wrapper / weights
    sam = load_medsam(REPO_ROOT / "checkpoints" / "medsam_vit_b.pth", device=device)
    method_kwargs: dict = {}
    if method_cfg["checkpoint"]:
        ckpt = torch.load(
            REPO_ROOT / method_cfg["checkpoint"],
            map_location="cpu",
            weights_only=False,
        )
        method_kwargs = ckpt.get("config", {}).get("method_kwargs", {}) or {}
        setup_method(sam, ckpt["method"], **method_kwargs)
        state = {k: v.to(device) for k, v in ckpt["trainable_state"].items()}
        sam.load_state_dict(state, strict=False)
    else:
        setup_method(sam, "zero_shot")
    sam.eval()

    # Pick indices
    if strategy == "spread":
        run_name = f"{method_name}_seed0" if method_name != "zero_shot" else "zero_shot"
        ds_csv_name = DATASET_TO_CSV_NAME[dataset_name]
        idxs = pick_indices_spread(run_name, ds_csv_name, dataset, n_samples)
    else:
        idxs = pick_indices_first(dataset, n_samples)
    if not idxs:
        print(f"  [viz] no samples available — skipping")
        return

    fig, axes = plt.subplots(
        len(idxs), 4,
        figsize=(20, 5 * len(idxs)),
        squeeze=False,
    )
    for row, idx in enumerate(idxs):
        item = dataset[idx]
        img_rgb = denormalize_image(item["image"])
        gt = item["mask"].numpy().astype(bool)
        bbox = item["bbox"].numpy()

        images = item["image"].unsqueeze(0).to(device)
        bboxes = item["bbox"].unsqueeze(0).to(device)
        pred_t = predict_batch(sam, images, bboxes)
        pred = pred_t.squeeze().cpu().numpy().astype(bool)

        dice, iou = dice_iou(pred, gt)

        # Column 0: input image with bbox prompt
        ax = axes[row, 0]
        ax.imshow(img_rgb)
        x1, y1, x2, y2 = bbox
        ax.add_patch(
            plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor="cyan", linewidth=2.0,
            )
        )
        ax.set_title(f"Input + bbox prompt\nID: {item['image_id']}", fontsize=10)
        ax.axis("off")

        # Column 1: ground truth overlay
        axes[row, 1].imshow(overlay_mask(img_rgb, gt, (0, 200, 0)))
        axes[row, 1].set_title("Ground truth (green)", fontsize=10)
        axes[row, 1].axis("off")

        # Column 2: prediction overlay
        axes[row, 2].imshow(overlay_mask(img_rgb, pred, (220, 30, 30)))
        axes[row, 2].set_title(f"Prediction (red)\nDice = {dice:.3f}, IoU = {iou:.3f}", fontsize=10)
        axes[row, 2].axis("off")

        # Column 3: error breakdown
        axes[row, 3].imshow(error_breakdown(img_rgb, pred, gt))
        axes[row, 3].set_title("TP green / FP red / FN blue", fontsize=10)
        axes[row, 3].axis("off")

    fig.suptitle(f"{label} on {dataset_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--method", nargs="+", choices=list(METHODS.keys()),
        default=list(METHODS.keys()),
        help="Methods to visualize (default: all).",
    )
    p.add_argument(
        "--dataset", nargs="+", choices=list(DATASET_BUILDERS.keys()),
        default=list(DATASET_BUILDERS.keys()),
        help="Datasets to visualize (default: all).",
    )
    p.add_argument("--n", type=int, default=4, help="Examples per (method, dataset) figure.")
    p.add_argument(
        "--strategy", choices=["first", "spread"], default="first",
        help="Sample selection: 'first' = first N items, 'spread' = best/mid/worst by Dice.",
    )
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    # Build each dataset once (avoid re-reading CSVs repeatedly)
    datasets: dict = {}
    for ds_name in args.dataset:
        try:
            datasets[ds_name] = DATASET_BUILDERS[ds_name]()
        except Exception as e:
            print(f"[viz] could not build {ds_name}: {e}")

    for method_name in args.method:
        method_cfg = METHODS[method_name]
        # Skip methods whose checkpoint doesn't exist (other than zero-shot)
        if method_cfg["checkpoint"]:
            ckpt = REPO_ROOT / method_cfg["checkpoint"]
            if not ckpt.exists():
                print(f"[viz] skipping {method_name}: {ckpt} not found")
                continue
        for ds_name, ds in datasets.items():
            out = FIG_DIR / f"{method_name}__{ds_name}.png"
            try:
                render_figure(
                    method_name, method_cfg, ds_name, ds,
                    n_samples=args.n, strategy=args.strategy,
                    device=device, out_path=out,
                )
            except Exception as e:
                print(f"  [viz] failed for {method_name} × {ds_name}: {e}")

    print(f"\n[viz] All figures in {FIG_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
