"""Qualitative bbox-robustness figures: rows are examples, columns are jitter levels, each panel shows image + jittered bbox (cyan) + GT outline (green) + prediction (red)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `from src...` work when run directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.busi import BUSI
from src.data.cbis_ddsm import CBISDDSM
from src.data.isic import ISIC2018, PIXEL_MEAN, PIXEL_STD
from src.data.ph2 import PH2
from src.eval import predict_batch
from src.models.medsam import load_medsam
from src.models.methods import setup_method

OUT_DIR = _REPO_ROOT / "results" / "paper" / "bbox_robustness" / "qualitative"

METHODS = {
    "zero_shot":    {"checkpoint": None,                                          "label": "Zero-shot"},
    "decoder_only": {"checkpoint": "checkpoints/runs/decoder_only_seed0/best.pth", "label": "Decoder-only FT"},
    "vpt_shallow":  {"checkpoint": "checkpoints/runs/vpt_shallow_seed0/best.pth",  "label": "VPT-shallow"},
    "vpt_deep":     {"checkpoint": "checkpoints/runs/vpt_deep_seed0/best.pth",     "label": "VPT-deep"},
    "lora":         {"checkpoint": "checkpoints/runs/lora_seed0/best.pth",         "label": "LoRA"},
    "full_ft":      {"checkpoint": "checkpoints/runs/full_ft_seed0/best.pth",      "label": "Full FT"},
}

DATASET_BUILDERS = {
    # built at perturb=0; bbox is jittered manually below for reproducibility
    "isic":      lambda: ISIC2018(root=_REPO_ROOT / "data", split="test", image_size=1024, bbox_perturb_pixels=0),
    "ph2":       lambda: PH2(root=_REPO_ROOT / "data" / "ph2", image_size=1024, bbox_perturb_pixels=0),
    "busi":      lambda: BUSI(root=_REPO_ROOT / "data" / "busi", image_size=1024, bbox_perturb_pixels=0, include_normal=False),
    "cbis_ddsm": lambda: CBISDDSM(root=_REPO_ROOT / "data" / "cbis-ddsm", split="test", image_size=1024, bbox_perturb_pixels=0, abnormality_type="all"),
}


# rendering helpers (duplicated from visualize_predictions.py)
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


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    dice = float((2.0 * inter) / max(denom, 1))
    iou = float(inter / max(union, 1))
    return dice, iou


# reproducible bbox jittering: fixed seed per (image_id, jitter) so figures are deterministic
def jitter_bbox(base_bbox: np.ndarray, perturb_px: int, image_size: int, image_id: str) -> np.ndarray:
    """Expand bbox outward (never shrink) by a random amount in [0, perturb_px], deterministic per (image_id, perturb_px)."""
    if perturb_px == 0:
        return base_bbox.copy().astype(np.float32)
    x1, y1, x2, y2 = base_bbox.astype(np.float32)
    seed = abs(hash((image_id, perturb_px))) % (2**31)
    rng = np.random.RandomState(seed=seed)
    pad_left, pad_top, pad_right, pad_bottom = rng.uniform(0, perturb_px, size=4)
    new_x1 = max(0.0, x1 - pad_left)
    new_y1 = max(0.0, y1 - pad_top)
    new_x2 = min(image_size - 1.0, x2 + pad_right)
    new_y2 = min(image_size - 1.0, y2 + pad_bottom)
    return np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float32)


def render_panel(ax, img_rgb, gt_mask, pred_mask, bbox, jitter, dice, iou, image_size):
    """Render one (example, jitter) panel: prediction in red, GT contour in green, jittered bbox in cyan."""
    rendered = overlay_mask(img_rgb, pred_mask, (220, 30, 30), alpha=0.45)
    ax.imshow(rendered)

    # GT outline in green
    if gt_mask.any():
        ax.contour(
            gt_mask.astype(float),
            levels=[0.5],
            colors=["#00FF40"],
            linewidths=1.8,
        )

    # Jittered bbox in cyan
    x1, y1, x2, y2 = bbox
    ax.add_patch(
        plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor="cyan", linewidth=2.0,
        )
    )

    ax.set_title(f"jitter = {jitter} px\nDice = {dice:.3f}, IoU = {iou:.3f}", fontsize=10)
    ax.axis("off")
    # render the whole image even if the bbox runs out of frame
    ax.set_xlim(0, image_size)
    ax.set_ylim(image_size, 0)


@torch.no_grad()
def render_figure(
    method_name: str,
    method_cfg: dict,
    dataset_name: str,
    dataset,
    sam: torch.nn.Module,
    n_examples: int,
    jitter_levels: list[int],
    image_size: int,
    device: str,
    out_path: Path,
) -> None:
    print(f"[bbox_viz] {method_cfg['label']:14s} x {dataset_name:9s} -> {out_path.name}")

    n_examples = min(n_examples, len(dataset))
    if n_examples == 0:
        print("  [bbox_viz] no examples available, skipping")
        return

    fig, axes = plt.subplots(
        n_examples, len(jitter_levels),
        figsize=(3.5 * len(jitter_levels), 4.0 * n_examples),
        squeeze=False,
    )

    for row in range(n_examples):
        item = dataset[row]
        img_rgb = denormalize_image(item["image"])
        gt = item["mask"].numpy().astype(bool)
        base_bbox = item["bbox"].numpy()
        image_id = item.get("image_id", f"idx_{row}")

        img_tensor = item["image"].unsqueeze(0).to(device)

        for col, jitter in enumerate(jitter_levels):
            jittered = jitter_bbox(base_bbox, jitter, image_size, str(image_id))
            jittered_t = torch.from_numpy(jittered).unsqueeze(0).to(device)

            pred = predict_batch(sam, img_tensor, jittered_t)
            pred_np = pred.squeeze().cpu().numpy().astype(bool)

            dice, iou = dice_iou(pred_np, gt)
            render_panel(
                axes[row, col],
                img_rgb, gt, pred_np, jittered,
                jitter, dice, iou, image_size,
            )

    fig.suptitle(
        f"{method_cfg['label']} on {dataset_name}: robustness to bbox jitter\n"
        f"(cyan = jittered bbox prompt, green outline = GT, red overlay = prediction)",
        fontsize=13, fontweight="bold", y=1.0,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()


def load_model(method_name: str, method_cfg: dict, device: str) -> torch.nn.Module:
    """Load MedSAM and apply the given fine-tuning method's weights."""
    sam = load_medsam(_REPO_ROOT / "checkpoints" / "medsam_vit_b.pth", device=device)
    if method_cfg["checkpoint"] is None:
        setup_method(sam, "zero_shot")
        sam.eval()
        return sam

    ckpt = torch.load(
        _REPO_ROOT / method_cfg["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    method_kwargs = ckpt.get("config", {}).get("method_kwargs", {}) or {}
    setup_method(sam, ckpt["method"], **method_kwargs)
    state = {k: v.to(device) for k, v in ckpt["trainable_state"].items()}
    sam.load_state_dict(state, strict=False)
    sam.eval()
    return sam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods", nargs="+",
        default=["zero_shot", "decoder_only", "vpt_shallow", "vpt_deep", "lora", "full_ft"],
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["isic", "ph2", "busi", "cbis_ddsm"],
    )
    parser.add_argument(
        "--jitter", nargs="+", type=int,
        default=[0, 20, 40, 80, 120, 200],
    )
    parser.add_argument("--n-examples", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"[bbox_viz] device={args.device}  n_examples={args.n_examples}")
    print(f"[bbox_viz] jitter levels: {args.jitter}")
    print(f"[bbox_viz] methods: {args.methods}")
    print(f"[bbox_viz] datasets: {args.datasets}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # outer loop is method since model load is the expensive part
    for method_name in args.methods:
        if method_name not in METHODS:
            print(f"[bbox_viz] unknown method '{method_name}', skipping")
            continue
        method_cfg = METHODS[method_name]
        print(f"\n[bbox_viz] === Loading {method_cfg['label']} ===")
        sam = load_model(method_name, method_cfg, args.device)

        for ds_name in args.datasets:
            if ds_name not in DATASET_BUILDERS:
                print(f"[bbox_viz] unknown dataset '{ds_name}', skipping")
                continue
            dataset = DATASET_BUILDERS[ds_name]()
            out_path = OUT_DIR / f"{method_name}__{ds_name}__bbox_jitter.png"
            render_figure(
                method_name, method_cfg,
                ds_name, dataset, sam,
                args.n_examples, args.jitter, args.image_size,
                args.device, out_path,
            )

        del sam
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n[bbox_viz] All figures written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
