"""Evaluation entry point for zero-shot or any trained checkpoint (method auto-detected)."""

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

from src.data.busi import BUSI
from src.data.cbis_ddsm import CBISDDSM
from src.data.dmid import DMID
from src.data.isic import ISIC2018, isic_collate
from src.data.ph2 import PH2
from src.device_utils import (
    device_name,
    get_device,
    peak_memory_mb,
    reset_peak_memory,
)
from src.drift import decoder_drift
from src.metrics import aggregate_metrics, dice_score, hd95, iou_score
from src.models.medsam import load_medsam
from src.models.methods import encoder_in_grad_path, setup_method

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
    p.add_argument("--limit", type=int, default=None, help="First N images per dataset")
    p.add_argument(
        "--drift",
        action="store_true",
        help="Load base MedSAM alongside and write per-image decoder drift "
        "(1 - CKA of the upscaled mask embedding).",
    )
    p.add_argument(
        "--bbox-perturb",
        type=int,
        default=None,
        help="Override eval.bbox_perturb_pixels",
    )
    p.add_argument(
        "--results-csv", type=Path, default=None, help="Override output.results_csv"
    )
    p.add_argument(
        "--per-image-dir", type=Path, default=None, help="Directory for per-image CSVs"
    )
    return p.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dataset(cfg: dict, ts_cfg: dict, image_size: int):
    kind = ts_cfg["kind"]
    perturb = cfg["eval"].get("bbox_perturb_pixels", 0)
    if kind == "isic":
        candidate = REPO_ROOT / cfg["data"]["root"] / "isic2018"
        root = candidate if candidate.is_dir() else REPO_ROOT / cfg["data"]["root"]
        return ISIC2018(
            root=root,
            split=ts_cfg.get("split", "test"),
            image_size=image_size,
            bbox_perturb_pixels=perturb,
        )
    if kind == "ph2":
        # Default location data/ph2/, overridable via ts_cfg["root"].
        root = REPO_ROOT / ts_cfg.get("root", "data/ph2")
        return PH2(root=root, image_size=image_size, bbox_perturb_pixels=perturb)
    if kind == "busi":
        # Default location data/busi/, overridable via ts_cfg["root"].
        root = REPO_ROOT / ts_cfg.get("root", "data/busi")
        return BUSI(
            root=root,
            image_size=image_size,
            bbox_perturb_pixels=perturb,
            include_normal=ts_cfg.get("include_normal", False),
        )
    if kind == "cbis_ddsm":
        root = REPO_ROOT / ts_cfg.get("root", "data/cbis-ddsm")
        return CBISDDSM(
            root=root,
            split=ts_cfg.get("split", "test"),
            image_size=image_size,
            bbox_perturb_pixels=perturb,
            abnormality_type=ts_cfg.get("abnormality_type", "all"),
        )
    if kind == "dmid":
        root_cfg = ts_cfg.get("root", "data/dmid")
        root = Path(root_cfg) if Path(root_cfg).is_absolute() else REPO_ROOT / root_cfg
        return DMID(root=root, image_size=image_size, bbox_perturb_pixels=perturb)
    raise ValueError(f"Unknown dataset kind: {kind}")


@torch.no_grad()
def predict_from_embeddings_with_iou(
    sam,
    image_embeddings: torch.Tensor,
    bboxes: torch.Tensor,
    H: int,
    W: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prompt encoder + mask decoder per image; returns (B, H, W) uint8 masks and (B,) iou_pred."""
    masks_out, ious = [], []
    for i in range(image_embeddings.shape[0]):
        sparse_embed, dense_embed = sam.prompt_encoder(
            points=None,
            boxes=bboxes[i : i + 1],
            masks=None,
        )
        low_res, iou_pred = sam.mask_decoder(
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
        ious.append(iou_pred.reshape(-1)[0].float())
    return torch.stack(masks_out, dim=0), torch.stack(ious, dim=0)


@torch.no_grad()
def predict_from_embeddings(
    sam,
    image_embeddings: torch.Tensor,
    bboxes: torch.Tensor,
    H: int,
    W: int,
) -> torch.Tensor:
    """Masks only; kept for scripts/eval_all_methods.py and bbox_robustness."""
    masks, _ = predict_from_embeddings_with_iou(sam, image_embeddings, bboxes, H, W)
    return masks


def write_per_image_csv(path: Path, rows: list[dict]) -> Path:
    """Columns follow the first row's keys (image_id, dice, iou, hd95, iou_pred[, drift])."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        list(rows[0].keys())
        if rows
        else ["image_id", "dice", "iou", "hd95", "iou_pred"]
    )
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return path


@torch.no_grad()
def predict_batch(sam, images: torch.Tensor, bboxes: torch.Tensor) -> torch.Tensor:
    """Method-agnostic forward over sam.image_encoder; returns (B, H, W) uint8 predictions."""
    H, W = images.shape[-2:]
    image_embeddings = sam.image_encoder(images)  # (B, 256, H/16, W/16)
    return predict_from_embeddings(sam, image_embeddings, bboxes, H, W)


def evaluate(cfg: dict, args: argparse.Namespace) -> int:
    # --device, else config preference, else auto-pick; treat config "cuda" as absent if no CUDA.
    preferred = args.device or cfg["eval"].get("device")
    if preferred == "cuda" and not torch.cuda.is_available():
        preferred = None
    device = get_device(prefer=preferred)
    if args.bbox_perturb is not None:
        cfg["eval"]["bbox_perturb_pixels"] = int(args.bbox_perturb)
    limit = args.limit if args.limit is not None else (8 if args.quick else None)

    image_size = cfg["model"]["image_size"]
    print(f"[eval] device={device} ({device_name(device)}) image_size={image_size}")

    # Always load base MedSAM; trained methods overlay trainable_state onto it.
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

    base_sam = None
    cur_hook = base_hook = None
    shares_encoder = False
    if args.drift:
        from cka.hooks import register_hooks

        base_sam = load_medsam(base_ckpt, arch=cfg["model"]["arch"], device=device)
        setup_method(base_sam, "zero_shot")
        base_sam.eval()
        shares_encoder = not encoder_in_grad_path(method)
        # Encoder drift (neck output) is only defined when the encoder was adapted.
        hook_layers = ["decoder_upscaling"] + (
            [] if shares_encoder else ["encoder_neck"]
        )
        cur_hook = register_hooks(sam, hook_layers, detach=True, accumulate=True)
        base_hook = register_hooks(base_sam, hook_layers, detach=True, accumulate=True)
        print(f"[eval] drift enabled (base encoder reused: {shares_encoder})")

    reset_peak_memory(device)

    rows_for_csv: list[dict] = []

    for ts_cfg in cfg["data"]["test_sets"]:
        ds_name = ts_cfg["name"]
        print(f"\n[eval] === {ds_name} ===")
        ds = build_dataset(cfg, ts_cfg, image_size)
        if limit is not None:
            ds.items = ds.items[:limit]
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
        with torch.no_grad():
            for batch in tqdm(loader, desc=ds_name):
                images = batch["image"].to(device)
                bboxes = batch["bbox"].to(device)
                masks_gt = batch["mask"].cpu().numpy()
                H, W = images.shape[-2:]
                if cur_hook is not None:
                    cur_hook.clear()
                embeddings = sam.image_encoder(images)
                preds, iou_preds = predict_from_embeddings_with_iou(
                    sam, embeddings, bboxes, H, W
                )
                drifts = drifts_enc = None
                if base_sam is not None:
                    if shares_encoder:
                        base_emb = embeddings
                    else:
                        base_hook.clear()
                        base_emb = base_sam.image_encoder(images)
                    predict_from_embeddings_with_iou(base_sam, base_emb, bboxes, H, W)
                    cur_acts = cur_hook.stacked()
                    base_acts = base_hook.stacked()
                    dec_c, dec_b = (
                        cur_acts["decoder_upscaling"],
                        base_acts["decoder_upscaling"],
                    )
                    drifts = [
                        decoder_drift(dec_b[j], dec_c[j]) for j in range(dec_c.shape[0])
                    ]
                    if shares_encoder:
                        drifts_enc = [0.0] * len(drifts)
                    else:
                        enc_c, enc_b = (
                            cur_acts["encoder_neck"],
                            base_acts["encoder_neck"],
                        )
                        drifts_enc = [
                            decoder_drift(enc_b[j], enc_c[j])
                            for j in range(enc_c.shape[0])
                        ]
                    cur_hook.clear()
                    base_hook.clear()
                preds_np = preds.cpu().numpy()
                iou_np = iou_preds.cpu().numpy()
                for j in range(preds_np.shape[0]):
                    pj = preds_np[j]
                    gj = masks_gt[j]
                    d = dice_score(pj, gj)
                    i_ = iou_score(pj, gj)
                    h_ = hd95(pj, gj)
                    per_image.append({"dice": d, "iou": i_, "hd95": h_})
                    row = {
                        "image_id": batch["image_id"][j],
                        "dice": d,
                        "iou": i_,
                        "hd95": h_,
                        "iou_pred": float(iou_np[j]),
                    }
                    if drifts is not None:
                        row["drift"] = float(drifts[j])
                        row["drift_enc"] = float(drifts_enc[j])
                    per_image_rows.append(row)
        elapsed = time.time() - t0

        agg = aggregate_metrics(per_image)
        peak_mb = peak_memory_mb(device)
        print(
            f"[eval] {ds_name}: dice={agg['dice_mean']:.4f}±{agg['dice_std']:.4f} "
            f"iou={agg['iou_mean']:.4f} hd95={agg['hd95_mean']:.2f}px "
            f"n={len(per_image)} time={elapsed:.1f}s peak={peak_mb:.0f}MB"
        )

        rows_for_csv.append(
            {
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
                "notes": ";".join(
                    t
                    for t in (
                        "quick" if args.quick else "",
                        f"limit{limit}" if limit is not None else "",
                        "drift" if args.drift else "",
                        f"pm{cfg['eval'].get('bbox_perturb_pixels', 0)}",
                    )
                    if t
                ),
            }
        )

        # Per-image CSV (one per dataset)
        per_image_dir = (
            REPO_ROOT / args.per_image_dir
            if args.per_image_dir is not None
            else (
                REPO_ROOT
                / cfg["output"].get("per_image_csv", "results/raw/per_image.csv")
            ).parent
        )
        per_image_path = write_per_image_csv(
            per_image_dir / f"{run_name}_{ds_name}_per_image.csv", per_image_rows
        )
        print(f"[eval] per-image -> {per_image_path}")

    # Append to runs.csv
    runs_path = REPO_ROOT / (args.results_csv or cfg["output"]["results_csv"])
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = runs_path.exists() and runs_path.stat().st_size > 0
    with open(runs_path, "a", newline="") as f:
        fields = list(rows_for_csv[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            w.writeheader()
        w.writerows(rows_for_csv)
    print(f"[eval] appended {len(rows_for_csv)} rows to {runs_path}")
    if cur_hook is not None:
        cur_hook.remove()
        base_hook.remove()
    return 0


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    return evaluate(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
