"""Side-by-side qualitative comparison of pm=0 vs pm=20 trained models; the deterministic sample_idx=0 bbox is fed to both so the comparison is fair."""
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


FIG_DIR = REPO_ROOT / "bbox_robustness" / "results_pm20" / "figures" / "comparison"
# Per-image CSVs from the pm=0 eval, used by --strategy spread
PM0_PER_IMAGE_DIR = REPO_ROOT / "bbox_robustness" / "results" / "per_image"

# Checkpoint paths for both trainings; zero_shot is excluded (same model in both conditions)
METHODS = {
    "decoder_only": {
        "pm0":   "checkpoints/runs/decoder_only_seed0/best.pth",
        "pm20":  "checkpoints/runs_pm20/decoder_only_seed0_pm20/best.pth",
        "label": "Decoder-only FT",
    },
    "vpt_shallow": {
        "pm0":   "checkpoints/runs/vpt_shallow_seed0/best.pth",
        "pm20":  "checkpoints/runs_pm20/vpt_shallow_seed0_pm20/best.pth",
        "label": "VPT-shallow",
    },
    "vpt_deep": {
        "pm0":   "checkpoints/runs/vpt_deep_seed0/best.pth",
        "pm20":  "checkpoints/runs_pm20/vpt_deep_seed0_pm20/best.pth",
        "label": "VPT-deep",
    },
    "lora": {
        "pm0":   "checkpoints/runs/lora_seed0/best.pth",
        "pm20":  "checkpoints/runs_pm20/lora_seed0_pm20/best.pth",
        "label": "LoRA",
    },
    "full_ft": {
        "pm0":   "checkpoints/runs/full_ft_seed0/best.pth",
        "pm20":  "checkpoints/runs_pm20/full_ft_seed0_pm20/best.pth",
        "label": "Full FT",
    },
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
    pm0_run_name: str, ds_csv_name: str, perturb_max: int, dataset, n: int,
) -> list[int]:
    """Pick indices spanning the Dice distribution from the pm=0 per_image CSV; same images are rendered for both trainings."""
    csv_path = PM0_PER_IMAGE_DIR / f"{pm0_run_name}_{ds_csv_name}_pm{perturb_max}.csv"
    if not csv_path.exists():
        print(f"  [spread] {csv_path.name} not found; using first-{n}")
        return pick_indices_first(dataset, n)
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((r["image_id"], float(r["dice_mean"])))
    if not rows:
        return pick_indices_first(dataset, n)
    rows.sort(key=lambda x: x[1], reverse=True)
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


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    dice = float((2.0 * inter) / max(denom, 1))
    iou = float(inter / max(union, 1))
    return dice, iou


def safe_filename(s: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", s)
    return cleaned.strip("_")


# Build a SAM with the given method + checkpoint loaded
def build_sam_for_method(base_sd: dict, ckpt_path: Path, device: str) -> torch.nn.Module:
    sam = load_medsam_from_state_dict(base_sd, device=device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    method_kwargs = ckpt.get("config", {}).get("method_kwargs", {}) or {}
    setup_method(sam, ckpt["method"], **method_kwargs)
    state = {k: v.to(device) for k, v in ckpt["trainable_state"].items()}
    sam.load_state_dict(state, strict=False)
    sam.eval()
    return sam


@torch.no_grad()
def render_comparison(
    sam_pm0, sam_pm20, method_label: str, dataset_name: str,
    item: dict, perturb_levels: list, device: str, out_path: Path,
) -> None:
    img_rgb = denormalize_image(item["image"])
    gt = item["mask"].numpy().astype(bool)
    tight_bbox = item["bbox"].numpy()
    img_id = item["image_id"]

    # Encode with both models (different weights give different embeddings)
    images_t = item["image"].unsqueeze(0).to(device)
    emb_pm0 = sam_pm0.image_encoder(images_t)
    emb_pm20 = sam_pm20.image_encoder(images_t)

    n_rows = len(perturb_levels)
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows), squeeze=False)

    for row, pm in enumerate(perturb_levels):
        # Deterministic perturbed bbox, same one fed to both models
        rng = make_rng(img_id, pm, sample_idx=0)
        perturbed = expand_bbox(tight_bbox, pm, IMAGE_SIZE, rng)
        boxes_t = torch.from_numpy(perturbed).unsqueeze(0).to(device).float()

        pred_pm0 = decode_image_with_prompts(
            sam_pm0, emb_pm0, boxes_t, IMAGE_SIZE, IMAGE_SIZE,
        ).squeeze(0).cpu().numpy().astype(bool)
        pred_pm20 = decode_image_with_prompts(
            sam_pm20, emb_pm20, boxes_t, IMAGE_SIZE, IMAGE_SIZE,
        ).squeeze(0).cpu().numpy().astype(bool)

        dice_pm0, iou_pm0 = dice_iou(pred_pm0, gt)
        dice_pm20, iou_pm20 = dice_iou(pred_pm20, gt)
        delta = dice_pm20 - dice_pm0

        # Col 0: input + perturbed bbox
        ax = axes[row, 0]
        ax.imshow(img_rgb)
        x1t, y1t, x2t, y2t = tight_bbox
        ax.add_patch(plt.Rectangle((x1t, y1t), x2t - x1t, y2t - y1t,
                                     fill=False, edgecolor="yellow",
                                     linewidth=1.2, linestyle="--", alpha=0.7))
        x1, y1, x2, y2 = perturbed
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     fill=False, edgecolor="cyan", linewidth=2.0))
        ax.set_title(f"Input + bbox (perturb_max = 0-{pm} px)", fontsize=11, fontweight="bold")
        ax.set_ylabel(f"0-{pm} px", fontsize=12, fontweight="bold", rotation=0,
                       labelpad=40, va="center")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Col 1: ground truth
        axes[row, 1].imshow(overlay_mask(img_rgb, gt, (0, 200, 0)))
        axes[row, 1].set_title("Ground truth (green)", fontsize=11)
        axes[row, 1].axis("off")

        # Col 2: pm=0 trained prediction
        axes[row, 2].imshow(overlay_mask(img_rgb, pred_pm0, (220, 30, 30)))
        axes[row, 2].set_title(
            f"pm=0 trained  |  Dice = {dice_pm0:.3f}, IoU = {iou_pm0:.3f}",
            fontsize=11,
        )
        axes[row, 2].axis("off")

        # Col 3: pm=20 trained prediction, with delta annotation
        delta_color = "#2ca02c" if delta > 0.01 else ("#d62728" if delta < -0.01 else "#666666")
        delta_sign = "+" if delta >= 0 else ""
        axes[row, 3].imshow(overlay_mask(img_rgb, pred_pm20, (220, 30, 30)))
        axes[row, 3].set_title(
            f"pm=20 trained  |  Dice = {dice_pm20:.3f}  |  delta = {delta_sign}{delta:.3f}",
            fontsize=11,
            color=delta_color,
            fontweight="bold" if abs(delta) > 0.05 else "normal",
        )
        axes[row, 3].axis("off")

    fig.suptitle(
        f"{method_label}  -  {dataset_name}  -  image {img_id}\n"
        f"Comparison: pm=0 trained (col 3) vs pm=20 trained (col 4).  "
        f"Green delta = pm=20 better; red delta = pm=20 worse.",
        fontsize=13, fontweight="bold",
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
    )
    p.add_argument(
        "--n", type=int, default=4,
        help="Sample images per (method, dataset) figure.",
    )
    p.add_argument(
        "--strategy", choices=["first", "spread"], default="first",
        help="Sample selection: 'first' = first N dataset items, "
             "'spread' = best/mid/worst by per-image Dice at the largest perturb "
             "level from the PM=0 robustness eval.",
    )
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = get_device(prefer=args.device)
    print(f"[viz-compare] device={device} ({device_name(device)})")

    # Build datasets once
    datasets: dict = {}
    for ds_name in args.dataset:
        try:
            datasets[ds_name] = DATASET_BUILDERS[ds_name]()
        except Exception as e:
            print(f"[viz-compare] could not build {ds_name}: {e}")

    # Load base MedSAM once
    base_ckpt_path = REPO_ROOT / "checkpoints" / "medsam_vit_b.pth"
    base_sd = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    if "model" in base_sd and isinstance(base_sd["model"], dict):
        base_sd = base_sd["model"]

    largest_pm = max(args.perturb_levels)

    for method_name in args.method:
        method_cfg = METHODS[method_name]
        label = method_cfg["label"]

        ckpt_pm0  = REPO_ROOT / method_cfg["pm0"]
        ckpt_pm20 = REPO_ROOT / method_cfg["pm20"]

        if not ckpt_pm0.exists():
            print(f"[viz-compare] skipping {method_name}: pm=0 ckpt missing ({ckpt_pm0})")
            continue
        if not ckpt_pm20.exists():
            print(f"[viz-compare] skipping {method_name}: pm=20 ckpt missing ({ckpt_pm20})")
            continue

        # Both SAMs in memory at once (~750 MB)
        sam_pm0  = build_sam_for_method(base_sd, ckpt_pm0,  device)
        sam_pm20 = build_sam_for_method(base_sd, ckpt_pm20, device)

        for ds_name, ds in datasets.items():
            pm0_run_name = f"{method_name}_seed0"
            ds_csv_name = DATASET_TO_CSV_NAME[ds_name]
            if args.strategy == "spread":
                idxs = pick_indices_spread(pm0_run_name, ds_csv_name, largest_pm, ds, args.n)
            else:
                idxs = pick_indices_first(ds, args.n)

            for idx in idxs:
                item = ds[idx]
                img_id_safe = safe_filename(item["image_id"])
                out_path = FIG_DIR / f"{method_name}__{ds_name}__{img_id_safe}.png"
                print(f"[viz-compare] {label} x {ds_name} x {item['image_id']} -> {out_path.name}")
                try:
                    render_comparison(
                        sam_pm0, sam_pm20, label, ds_name, item,
                        args.perturb_levels, device, out_path,
                    )
                except Exception as e:
                    print(f"  [viz-compare] failed: {e}")

        del sam_pm0, sam_pm20
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n[viz-compare] all figures in {FIG_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
