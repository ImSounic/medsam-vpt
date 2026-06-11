"""Qualitative bbox-robustness visualizations.

One figure per (method, dataset, image): the same image across perturbation
levels, one row per level, four columns:

    Col 0: input + bbox (cyan = perturbed, yellow dashed = tight ref)
    Col 1: ground-truth mask overlay (green)
    Col 2: prediction overlay (red) with Dice / IoU
    Col 3: TP green / FP red / FN blue breakdown

Rows are perturb_max = 20, 50, 100, 200 px by default. Each row's bbox is the
deterministic sample_idx=0 draw (same RNG scheme as eval_bbox_robust.py), so
it matches what was scored. The encoder runs once per figure; only the
prompt+decoder re-runs per row.

Output: results/figures/qualitative/<method>__<dataset>__<image_id>.png

Usage:
    python bbox_robustness/visualize_bbox_robust.py
    python bbox_robustness/visualize_bbox_robust.py --method lora full_ft \\
        --dataset cbis_ddsm --n 3
    python bbox_robustness/visualize_bbox_robust.py --strategy spread
    python bbox_robustness/visualize_bbox_robust.py --perturb-levels 10 50 150
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from bbox_robustness.eval_bbox_robust import (  # noqa: E402
    PERTURB_LEVELS_DEFAULT,
    decode_image_with_prompts,
    expand_bbox,
    make_rng,
)
from src.data.busi import BUSI  # noqa: E402
from src.data.cbis_ddsm import CBISDDSM  # noqa: E402
from src.data.isic import PIXEL_MEAN, PIXEL_STD, ISIC2018  # noqa: E402
from src.data.ph2 import PH2  # noqa: E402
from src.device_utils import device_name, get_device  # noqa: E402
from src.models.medsam import load_medsam_from_state_dict  # noqa: E402
from src.models.methods import setup_method  # noqa: E402


FIG_DIR = REPO_ROOT / "bbox_robustness" / "results" / "figures" / "qualitative"
PER_IMAGE_DIR = REPO_ROOT / "bbox_robustness" / "results" / "per_image"

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

IMAGE_SIZE = 1024


def pick_indices_first(dataset, n: int) -> list[int]:
    return list(range(min(n, len(dataset))))


def pick_indices_spread(
    run_name: str, ds_csv_name: str, perturb_max: int, dataset, n: int,
) -> list[int]:
    """Pick indices spanning the per-image Dice distribution at this perturb level.

    Reads results/per_image/{run}_{ds}_pm{N}.csv; falls back to first-N if it
    isn't there. Ranks at the largest perturb level, where methods differ most.
    """
    csv_path = PER_IMAGE_DIR / f"{run_name}_{ds_csv_name}_pm{perturb_max}.csv"
    if not csv_path.exists():
        print(f"  [spread] {csv_path.name} not found; using first-{n}")
        return pick_indices_first(dataset, n)
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((r["image_id"], float(r["dice_mean"])))
    if not rows:
        return pick_indices_first(dataset, n)
    rows.sort(key=lambda x: x[1], reverse=True)  # best Dice first
    idxs = np.linspace(0, len(rows) - 1, n).astype(int).tolist()
    target_ids = {rows[i][0] for i in idxs}
    out = []
    for di, item in enumerate(getattr(dataset, "items", [])):
        sid = item[-1] if isinstance(item, tuple) else None
        if sid in target_ids:
            out.append(di)
        if len(out) >= n:
            break
    return out if out else pick_indices_first(dataset, n)


# Rendering helpers, mirrored from scripts/visualize_predictions.py
def denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    arr = img_tensor.cpu().clone() * PIXEL_STD + PIXEL_MEAN
    return arr.permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8)


def overlay_mask(img: np.ndarray, mask: np.ndarray, color: tuple,
                  alpha: float = 0.45) -> np.ndarray:
    out = img.astype(np.float32).copy()
    c = np.array(color, dtype=np.float32)
    for k in range(3):
        out[..., k] = np.where(mask, out[..., k] * (1 - alpha) + c[k] * alpha,
                                out[..., k])
    return out.clip(0, 255).astype(np.uint8)


def error_breakdown(img: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    out = img.copy()
    out = overlay_mask(out, pred & gt,     (0, 200, 0),   alpha=0.5)   # TP
    out = overlay_mask(out, pred & ~gt,    (220, 30, 30), alpha=0.55)  # FP
    out = overlay_mask(out, ~pred & gt,    (30, 60, 220), alpha=0.55)  # FN
    return out


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    dice = float((2.0 * inter) / max(denom, 1))
    iou = float(inter / max(union, 1))
    return dice, iou


def safe_filename(s: str) -> str:
    """Keep alphanumerics, dashes, underscores; collapse the rest to underscores."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", s)
    return cleaned.strip("_")


@torch.no_grad()
def render_image_degradation(
    sam, method_label: str, dataset_name: str, item: dict,
    perturb_levels: list, device: str, out_path: Path,
) -> None:
    """One image, all perturb levels stacked vertically. Each row uses a
    different perturbed bbox; the embedding is computed once and reused.
    """
    img_rgb = denormalize_image(item["image"])
    gt = item["mask"].numpy().astype(bool)
    tight_bbox = item["bbox"].numpy()
    img_id = item["image_id"]

    # Encoder runs once, reused across all perturb levels (same image)
    images_t = item["image"].unsqueeze(0).to(device)
    embedding = sam.image_encoder(images_t)  # (1, 256, 64, 64)

    fig, axes = plt.subplots(
        len(perturb_levels), 4,
        figsize=(20, 5 * len(perturb_levels)),
        squeeze=False,
    )

    for row, pm in enumerate(perturb_levels):
        # Deterministic perturbed bbox for this image at this level
        rng = make_rng(img_id, pm, sample_idx=0)
        perturbed = expand_bbox(tight_bbox, pm, IMAGE_SIZE, rng)

        boxes_t = torch.from_numpy(perturbed).unsqueeze(0).to(device).float()  # (1, 4)
        pred_t = decode_image_with_prompts(
            sam, embedding, boxes_t, IMAGE_SIZE, IMAGE_SIZE,
        )  # (1, H, W) uint8
        pred = pred_t.squeeze(0).cpu().numpy().astype(bool)
        dice, iou = dice_iou(pred, gt)

        # Col 0: input + bboxes
        ax = axes[row, 0]
        ax.imshow(img_rgb)
        # Tight reference box: yellow dashed
        x1t, y1t, x2t, y2t = tight_bbox
        ax.add_patch(plt.Rectangle(
            (x1t, y1t), x2t - x1t, y2t - y1t,
            fill=False, edgecolor="yellow", linewidth=1.2,
            linestyle="--", alpha=0.7,
        ))
        # Perturbed box actually used: cyan
        x1, y1, x2, y2 = perturbed
        ax.add_patch(plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor="cyan", linewidth=2.0,
        ))
        ax.set_title(
            f"Input + bbox (perturb_max = 0-{pm} px per side)",
            fontsize=11, fontweight="bold",
        )
        # Y-axis label shows the perturb level on the leftmost panel
        ax.set_ylabel(f"0-{pm} px", fontsize=12, fontweight="bold", rotation=0,
                      labelpad=40, va="center")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Col 1: ground truth
        axes[row, 1].imshow(overlay_mask(img_rgb, gt, (0, 200, 0)))
        axes[row, 1].set_title("Ground truth (green)", fontsize=11)
        axes[row, 1].axis("off")

        # Col 2: prediction
        axes[row, 2].imshow(overlay_mask(img_rgb, pred, (220, 30, 30)))
        axes[row, 2].set_title(
            f"Prediction (red)  |  Dice = {dice:.3f}, IoU = {iou:.3f}",
            fontsize=11,
        )
        axes[row, 2].axis("off")

        # Col 3: error breakdown
        axes[row, 3].imshow(error_breakdown(img_rgb, pred, gt))
        axes[row, 3].set_title("TP green / FP red / FN blue", fontsize=11)
        axes[row, 3].axis("off")

    fig.suptitle(
        f"{method_label}  -  {dataset_name}  -  image {img_id}\n"
        f"Degradation across bbox imprecision (top to bottom: looser bbox)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--method", nargs="+", choices=list(METHODS.keys()),
        default=list(METHODS.keys()),
    )
    p.add_argument(
        "--dataset", nargs="+", choices=list(DATASET_BUILDERS.keys()),
        default=list(DATASET_BUILDERS.keys()),
    )
    p.add_argument(
        "--perturb-levels", type=int, nargs="+",
        default=PERTURB_LEVELS_DEFAULT,
        help=f"Perturbation levels as rows in the figure (default: {PERTURB_LEVELS_DEFAULT}).",
    )
    p.add_argument(
        "--n", type=int, default=4,
        help="Number of sample IMAGES per (method, dataset). Each gets its own figure.",
    )
    p.add_argument(
        "--strategy", choices=["first", "spread"], default="first",
        help="Sample selection: 'first' = first N dataset items, "
             "'spread' = best/mid/worst by per-image Dice at the LARGEST perturb level "
             "(needs per_image CSVs from eval_bbox_robust.py).",
    )
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = get_device(prefer=args.device)
    print(f"[viz] device={device} ({device_name(device)})")
    print(f"[viz] perturb levels per figure (rows): {args.perturb_levels}")
    print(f"[viz] sample images per (method, dataset): {args.n}")

    # Build datasets once
    datasets: dict = {}
    for ds_name in args.dataset:
        try:
            datasets[ds_name] = DATASET_BUILDERS[ds_name]()
        except Exception as e:
            print(f"[viz] could not build {ds_name}: {e}")

    # Load base MedSAM weights once
    base_ckpt_path = REPO_ROOT / "checkpoints" / "medsam_vit_b.pth"
    base_sd = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    if "model" in base_sd and isinstance(base_sd["model"], dict):
        base_sd = base_sd["model"]

    largest_pm = max(args.perturb_levels)

    for method_name in args.method:
        method_cfg = METHODS[method_name]
        label = method_cfg["label"]

        if method_cfg["checkpoint"]:
            ckpt_path = REPO_ROOT / method_cfg["checkpoint"]
            if not ckpt_path.exists():
                print(f"[viz] skipping {method_name}: {ckpt_path} not found")
                continue

        # One SAM per method, reused across datasets
        sam = load_medsam_from_state_dict(base_sd, device=device)
        method_kwargs: dict = {}
        if method_cfg["checkpoint"]:
            ckpt = torch.load(
                REPO_ROOT / method_cfg["checkpoint"],
                map_location="cpu", weights_only=False,
            )
            method_kwargs = ckpt.get("config", {}).get("method_kwargs", {}) or {}
            setup_method(sam, ckpt["method"], **method_kwargs)
            state = {k: v.to(device) for k, v in ckpt["trainable_state"].items()}
            sam.load_state_dict(state, strict=False)
        else:
            setup_method(sam, "zero_shot")
        sam.eval()

        for ds_name, ds in datasets.items():
            run_name = (f"{method_name}_seed0"
                        if method_name != "zero_shot" else "zero_shot")
            ds_csv_name = DATASET_TO_CSV_NAME[ds_name]
            if args.strategy == "spread":
                idxs = pick_indices_spread(
                    run_name, ds_csv_name, largest_pm, ds, args.n,
                )
            else:
                idxs = pick_indices_first(ds, args.n)

            for idx in idxs:
                item = ds[idx]
                img_id_safe = safe_filename(item["image_id"])
                out_path = FIG_DIR / f"{method_name}__{ds_name}__{img_id_safe}.png"
                print(f"[viz] {label} x {ds_name} x {item['image_id']} -> {out_path.name}")
                try:
                    render_image_degradation(
                        sam, label, ds_name, item,
                        args.perturb_levels, device, out_path,
                    )
                except Exception as e:
                    print(f"  [viz] failed: {e}")

        del sam
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n[viz] all figures in {FIG_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
