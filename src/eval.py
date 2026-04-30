"""Evaluation entry point — handles zero-shot and any trained checkpoint.

Usage:
    # Zero-shot
    python -m src.eval --config configs/zero_shot.yaml

    # Trained checkpoint (method is auto-detected from the checkpoint)
    python -m src.eval --config configs/zero_shot.yaml \
        --checkpoint checkpoints/runs/decoder_only_seed0/best.pth

    # Smoke test on 8 images
    python -m src.eval --config configs/zero_shot.yaml --quick

The config provides everything *not* tied to the trained method (test sets,
batch size, base MedSAM checkpoint, output paths). The --checkpoint flag,
when set, overrides the config's `method` and `method_kwargs` with whatever
was used at training time.
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.isic import ISIC2018, isic_collate
from src.metrics import aggregate_metrics, dice_score, hd95, iou_score
from src.models.medsam import load_medsam
from src.models.methods import setup_method

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Trained best.pth. Method/kwargs come from the checkpoint.",
    )
    p.add_argument("--quick", action="store_true", help="Run on first 8 images only")
    p.add_argument("--device", default=None)
    return p.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dataset(cfg: dict, ts_cfg: dict, image_size: int):
    kind = ts_cfg["kind"]
    if kind == "isic":
        # Try data/isic2018/ first, then data/ as fallback
        candidate = REPO_ROOT / cfg["data"]["root"] / "isic2018"
        root = candidate if candidate.is_dir() else REPO_ROOT / cfg["data"]["root"]
        return ISIC2018(
            root=root,
            split=ts_cfg.get("split", "test"),
            image_size=image_size,
            bbox_perturb_pixels=cfg["eval"].get("bbox_perturb_pixels", 0),
        )
    raise ValueError(f"Unknown dataset kind: {kind}")


@torch.no_grad()
def predict_batch(sam, images: torch.Tensor, bboxes: torch.Tensor) -> torch.Tensor:
    """Method-agnostic forward. Works for zero-shot or any trained method —
    sam.image_encoder is whatever wrapper was applied at setup time.

    Returns (B, H, W) uint8 mask predictions.
    """
    H, W = images.shape[-2:]
    image_embeddings = sam.image_encoder(images)  # (B, 256, H/16, W/16)

    masks_out = []
    for i in range(images.shape[0]):
        sparse_embed, dense_embed = sam.prompt_encoder(
            points=None,
            boxes=bboxes[i : i + 1],
            masks=None,
        )
        low_res, _ = sam.mask_decoder(
            image_embeddings=image_embeddings[i : i + 1],
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embed,
            dense_prompt_embeddings=dense_embed,
            multimask_output=False,
        )
        mask = torch.nn.functional.interpolate(
            low_res, size=(H, W), mode="bilinear", align_corners=False
        )
        masks_out.append((mask > 0).to(torch.uint8).squeeze(0).squeeze(0))
    return torch.stack(masks_out, dim=0)


def evaluate(cfg: dict, args: argparse.Namespace) -> int:
    device = args.device or cfg["eval"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("[eval] CUDA not available, falling back to CPU.")
        device = "cpu"

    image_size = cfg["model"]["image_size"]
    print(f"[eval] device={device} image_size={image_size}")

    # Load base MedSAM (always — even for trained methods, this provides the
    # frozen backbone that the trainable_state will overlay onto).
    base_ckpt = REPO_ROOT / cfg["model"]["checkpoint"]
    sam = load_medsam(base_ckpt, arch=cfg["model"]["arch"], device=device)

    # Decide method + run name
    ckpt = None
    if args.checkpoint is not None:
        # weights_only=False because we save the config dict alongside the tensors
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        method = ckpt["method"]
        method_kwargs = ckpt.get("config", {}).get("method_kwargs", {}) or {}
        run_name = ckpt.get("config", {}).get("name", method)
        seed = int(ckpt.get("config", {}).get("seed", 0))
        print(f"[eval] checkpoint: {args.checkpoint}")
        print(f"[eval] method: {method} (from checkpoint)")
        print(f"[eval] training: epoch={ckpt['epoch']} val_dice={ckpt['val_dice']:.4f}")
    else:
        method = cfg["method"]
        method_kwargs = cfg.get("method_kwargs", {}) or {}
        run_name = cfg["name"]
        seed = int(cfg.get("seed", 0))
        print(f"[eval] method: {method} (zero-shot from config)")

    # Apply method setup (creates VPT wrapper if needed)
    info = setup_method(sam, method, **method_kwargs)
    print(f"[eval] params total={info['total']:,} trainable={info['trainable']:,}")

    # Load trainable weights from checkpoint, if any
    if ckpt is not None:
        trainable_state = {k: v.to(device) for k, v in ckpt["trainable_state"].items()}
        result = sam.load_state_dict(trainable_state, strict=False)
        # PyTorch returns IncompatibleKeys(missing_keys, unexpected_keys)
        unexpected = list(result.unexpected_keys)
        loaded = len(trainable_state) - len(unexpected)
        print(
            f"[eval] loaded {loaded}/{len(trainable_state)} tensors from checkpoint"
            f" (unexpected keys: {len(unexpected)})"
        )
        if unexpected:
            preview = ", ".join(unexpected[:3])
            tail = "..." if len(unexpected) > 3 else ""
            print(f"[eval] WARNING unexpected keys: {preview}{tail}")
            print("[eval] check that --checkpoint matches the method in --config")

    sam.eval()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rows_for_csv: list[dict] = []

    for ts_cfg in cfg["data"]["test_sets"]:
        ds_name = ts_cfg["name"]
        print(f"\n[eval] === {ds_name} ===")
        ds = build_dataset(cfg, ts_cfg, image_size)
        if args.quick:
            ds.items = ds.items[:8]
        loader = DataLoader(
            ds,
            batch_size=cfg["eval"]["batch_size"],
            num_workers=cfg["eval"].get("num_workers", 2),
            collate_fn=isic_collate,
            shuffle=False,
        )
        per_image: list[dict] = []
        per_image_rows: list[dict] = []
        t0 = time.time()
        for batch in tqdm(loader, desc=ds_name):
            images = batch["image"].to(device)
            bboxes = batch["bbox"].to(device)
            masks_gt = batch["mask"].cpu().numpy()
            preds = predict_batch(sam, images, bboxes)
            preds_np = preds.cpu().numpy()
            for j in range(preds_np.shape[0]):
                pj = preds_np[j]
                gj = masks_gt[j]
                d = dice_score(pj, gj)
                i_ = iou_score(pj, gj)
                h_ = hd95(pj, gj)
                per_image.append({"dice": d, "iou": i_, "hd95": h_})
                per_image_rows.append({
                    "image_id": batch["image_id"][j],
                    "dice": d,
                    "iou": i_,
                    "hd95": h_,
                })
        elapsed = time.time() - t0

        agg = aggregate_metrics(per_image)
        peak_mb = (
            torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0.0
        )
        print(
            f"[eval] {ds_name}: dice={agg['dice_mean']:.4f}±{agg['dice_std']:.4f} "
            f"iou={agg['iou_mean']:.4f} hd95={agg['hd95_mean']:.2f}px "
            f"n={len(per_image)} time={elapsed:.1f}s peak={peak_mb:.0f}MB"
        )

        rows_for_csv.append({
            "run_name": run_name,
            "method": method,
            "dataset": ds_name,
            "seed": seed,
            "dice_mean": f"{agg['dice_mean']:.4f}",
            "dice_std": f"{agg['dice_std']:.4f}",
            "iou_mean": f"{agg['iou_mean']:.4f}",
            "hd95_mean": f"{agg['hd95_mean']:.4f}",
            "trainable_params": info["trainable"],
            "peak_mem_mb": f"{peak_mb:.0f}",
            "wall_clock_s": f"{elapsed:.1f}",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "notes": "quick" if args.quick else "",
        })

        # Per-image CSV (one per dataset)
        per_image_path = REPO_ROOT / cfg["output"].get(
            "per_image_csv", "results/raw/per_image.csv"
        )
        per_image_path = per_image_path.with_name(
            f"{run_name}_{ds_name}_per_image.csv"
        )
        per_image_path.parent.mkdir(parents=True, exist_ok=True)
        with open(per_image_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["image_id", "dice", "iou", "hd95"])
            w.writeheader()
            w.writerows(per_image_rows)
        print(f"[eval] per-image -> {per_image_path}")

    # Append to runs.csv
    runs_path = REPO_ROOT / cfg["output"]["results_csv"]
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = runs_path.exists() and runs_path.stat().st_size > 0
    with open(runs_path, "a", newline="") as f:
        fields = list(rows_for_csv[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            w.writeheader()
        w.writerows(rows_for_csv)
    print(f"[eval] appended {len(rows_for_csv)} rows to {runs_path}")
    return 0


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    return evaluate(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
