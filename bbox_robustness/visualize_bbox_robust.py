"""Qualitative bbox-robustness visualizations.

Same 4-column format as results/figures/qualitative/ (input+bbox, ground
truth, prediction, TP/FP/FN diff) — but generates one figure per
(method, dataset, perturb_level) so you can flip between e.g.
lora__cbis_ddsm__pm20.png and lora__cbis_ddsm__pm200.png to see how
predictions collapse as the prompt becomes sloppier.

The bbox used for inference and shown in column 0 is one deterministic
sample (sample_idx=0) drawn with the same RNG seed scheme as
eval_bbox_robust.py — so what you see is a representative example of
what was scored.

Usage:
    # All methods × all datasets × all default perturb levels
    python bbox_robustness/visualize_bbox_robust.py

    # Subset
    python bbox_robustness/visualize_bbox_robust.py --method lora full_ft \\
        --dataset cbis_ddsm --perturb-levels 20 200

    # Pick samples by per-image Dice quality (best/mid/worst), needs the
    # per_image CSVs that eval_bbox_robust.py writes
    python bbox_robustness/visualize_bbox_robust.py --strategy spread
"""
from __future__ import annotations

import argparse
import csv
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


# ----------------------------------------------------------------------------
# Sample selection
# ----------------------------------------------------------------------------
def pick_indices_first(dataset, n: int) -> list[int]:
    return list(range(min(n, len(dataset))))


def pick_indices_spread(run_name: str, ds_csv_name: str, perturb_max: int,
                         dataset, n: int) -> list[int]:
    """Pick indices spanning the dice distribution at this perturb level.

    Uses bbox_robustness/results/per_image/{run_name}_{ds}_pm{N}.csv if present.
    Falls back to first-N if the file isn't available.
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
    rows.sort(key=lambda x: x[1], reverse=True)  # best dice first
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


# ----------------------------------------------------------------------------
# Rendering helpers (mirrored from scripts/visualize_predictions.py for
# consistency — same colour palette, same overlay style)
# ----------------------------------------------------------------------------
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
    out = overlay_mask(out, pred & gt,     (0, 200, 0),   alpha=0.5)   # TP green
    out = overlay_mask(out, pred & ~gt,    (220, 30, 30), alpha=0.55)  # FP red
    out = overlay_mask(out, ~pred & gt,    (30, 60, 220), alpha=0.55)  # FN blue
    return out


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    dice = float((2.0 * inter) / max(denom, 1))
    iou = float(inter / max(union, 1))
    return dice, iou


# ----------------------------------------------------------------------------
# Per (method, dataset, perturb_level) figure
# ----------------------------------------------------------------------------
@torch.no_grad()
def render_figure(
    sam, method_label: str, dataset_name: str, dataset,
    perturb_max: int, sample_idxs: list, device: str, out_path: Path,
) -> None:
    fig, axes = plt.subplots(
        len(sample_idxs), 4,
        figsize=(20, 5 * len(sample_idxs)),
        squeeze=False,
    )
    for row, idx in enumerate(sample_idxs):
        item = dataset[idx]
        img_rgb = denormalize_image(item["image"])
        gt = item["mask"].numpy().astype(bool)
        tight_bbox = item["bbox"].numpy()
        img_id = item["image_id"]

        # Pick the same "sample_idx=0" perturbed bbox the eval saw first
        rng = make_rng(img_id, perturb_max, sample_idx=0)
        perturbed = expand_bbox(tight_bbox, perturb_max, IMAGE_SIZE, rng)

        # Inference with the perturbed bbox
        images_t = item["image"].unsqueeze(0).to(device)
        boxes_t = torch.from_numpy(perturbed).unsqueeze(0).to(device).float()  # (1, 4)
        embedding = sam.image_encoder(images_t)  # (1, 256, 64, 64)
        pred_t = decode_image_with_prompts(
            sam, embedding, boxes_t, IMAGE_SIZE, IMAGE_SIZE
        )  # (1, H, W) uint8
        pred = pred_t.squeeze(0).cpu().numpy().astype(bool)

        dice, iou = dice_iou(pred, gt)

        # Column 0: input + perturbed bbox in cyan; tight bbox in dim yellow for ref
        ax = axes[row, 0]
        ax.imshow(img_rgb)
        x1t, y1t, x2t, y2t = tight_bbox
        ax.add_patch(plt.Rectangle(
            (x1t, y1t), x2t - x1t, y2t - y1t,
            fill=False, edgecolor="yellow", linewidth=1.2,
            linestyle="--", alpha=0.7,
        ))
        x1, y1, x2, y2 = perturbed
        ax.add_patch(plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor="cyan", linewidth=2.0,
        ))
        ax.set_title(
            f"Input + bbox (cyan = perturbed, yellow dash = tight)\n"
            f"ID: {img_id}  |  perturb_max = {perturb_max} px",
            fontsize=10,
        )
        ax.axis("off")

        # Column 1: ground truth overlay
        axes[row, 1].imshow(overlay_mask(img_rgb, gt, (0, 200, 0)))
        axes[row, 1].set_title("Ground truth (green)", fontsize=10)
        axes[row, 1].axis("off")

        # Column 2: prediction overlay
        axes[row, 2].imshow(overlay_mask(img_rgb, pred, (220, 30, 30)))
        axes[row, 2].set_title(
            f"Prediction (red)\nDice = {dice:.3f}, IoU = {iou:.3f}",
            fontsize=10,
        )
        axes[row, 2].axis("off")

        # Column 3: TP/FP/FN diff
        axes[row, 3].imshow(error_breakdown(img_rgb, pred, gt))
        axes[row, 3].set_title("TP green / FP red / FN blue", fontsize=10)
        axes[row, 3].axis("off")

    fig.suptitle(
        f"{method_label} on {dataset_name}  —  bbox max-expansion {perturb_max} px",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
# CLI + main
# ----------------------------------------------------------------------------
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
    )
    p.add_argument(
        "--n", type=int, default=4,
        help="Sample images per (method, dataset, level) figure.",
    )
    p.add_argument(
        "--strategy", choices=["first", "spread"], default="first",
        help="Sample selection: 'first' = first N dataset items, "
             "'spread' = best/mid/worst by per-image Dice (needs per_image CSVs).",
    )
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = get_device(prefer=args.device)
    print(f"[viz] device={device} ({device_name(device)})")

    # Pre-build datasets once
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

    for method_name in args.method:
        method_cfg = METHODS[method_name]
        label = method_cfg["label"]

        # Skip if checkpoint is missing
        if method_cfg["checkpoint"]:
            ckpt_path = REPO_ROOT / method_cfg["checkpoint"]
            if not ckpt_path.exists():
                print(f"[viz] skipping {method_name}: {ckpt_path} not found")
                continue

        # Build SAM once per method, reuse across datasets and levels
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
            for pm in args.perturb_levels:
                # Per (method, dataset, perturb_level) — sample selection can
                # depend on the perturb level when using 'spread' strategy
                run_name = (f"{method_name}_seed0"
                            if method_name != "zero_shot" else "zero_shot")
                ds_csv_name = DATASET_TO_CSV_NAME[ds_name]
                if args.strategy == "spread":
                    idxs = pick_indices_spread(
                        run_name, ds_csv_name, pm, ds, args.n,
                    )
                else:
                    idxs = pick_indices_first(ds, args.n)
                if not idxs:
                    continue

                out_path = FIG_DIR / f"{method_name}__{ds_name}__pm{pm}.png"
                print(f"[viz] {label} × {ds_name} × pm={pm} -> {out_path.name}")
                try:
                    render_figure(
                        sam, label, ds_name, ds, pm, idxs, device, out_path,
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
