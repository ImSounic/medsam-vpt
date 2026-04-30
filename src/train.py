"""Training entry point. Method-agnostic — drives decoder-only, VPT, LoRA, full FT.

Usage:
    python -m src.train --config configs/decoder_only.yaml

The script saves only trainable parameters in the checkpoint, not the full
93M-param state dict — so decoder-only checkpoints are ~16 MB, VPT
checkpoints will be ~16 MB (decoder + tiny prompts), and full FT will be
~370 MB.
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.isic import ISIC2018, isic_collate
from src.losses import DiceBCELoss
from src.metrics import aggregate_metrics, dice_score, iou_score
from src.models.medsam import load_medsam
from src.models.methods import encoder_in_grad_path, setup_method

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--seed", type=int, default=None, help="Override config seed")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest.pth in the run directory if present.",
    )
    return p.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def forward_with_prompt(
    sam,
    images: torch.Tensor,
    bboxes: torch.Tensor,
    encoder_grad: bool,
) -> torch.Tensor:
    """Run MedSAM forward with bbox prompts. Returns logits (B, 1, H, W)."""
    if encoder_grad:
        image_emb = sam.image_encoder(images)
    else:
        with torch.no_grad():
            image_emb = sam.image_encoder(images)
        image_emb = image_emb.detach()

    B, _, H, W = images.shape
    masks_out = []
    for i in range(B):
        sparse, dense = sam.prompt_encoder(
            points=None, boxes=bboxes[i : i + 1], masks=None
        )
        low_res, _ = sam.mask_decoder(
            image_embeddings=image_emb[i : i + 1],
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        # Upsample low-res (256x256) to image size (HxW) for loss / metric
        masks_out.append(
            F.interpolate(low_res, size=(H, W), mode="bilinear", align_corners=False)
        )
    return torch.cat(masks_out, dim=0)  # (B, 1, H, W)


def train_one_epoch(
    sam, loader, optimizer, scaler, criterion, device, *, encoder_grad: bool, amp: bool
) -> dict:
    sam.train()
    losses, bces, dlosses = [], [], []
    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        bboxes = batch["bbox"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=amp):
            logits = forward_with_prompt(sam, images, bboxes, encoder_grad=encoder_grad)
            loss, parts = criterion(logits, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        bces.append(parts["bce"])
        dlosses.append(parts["dice_loss"])
        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "dice_l": f"{parts['dice_loss']:.3f}",
        })
    return {
        "loss": float(np.mean(losses)),
        "bce": float(np.mean(bces)),
        "dice_loss": float(np.mean(dlosses)),
    }


@torch.no_grad()
def validate(sam, loader, device, *, amp: bool) -> dict:
    sam.eval()
    per_image = []
    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        bboxes = batch["bbox"].to(device, non_blocking=True)
        masks_gt = batch["mask"].cpu().numpy()
        with torch.autocast(device_type="cuda", enabled=amp):
            logits = forward_with_prompt(sam, images, bboxes, encoder_grad=False)
        preds = (logits.squeeze(1) > 0).cpu().numpy().astype("uint8")
        for j in range(preds.shape[0]):
            per_image.append({
                "dice": dice_score(preds[j], masks_gt[j]),
                "iou": iou_score(preds[j], masks_gt[j]),
            })
    agg = aggregate_metrics(per_image)
    return agg


def save_checkpoint(
    sam,
    ckpt_path: Path,
    epoch: int,
    val_dice: float,
    cfg: dict,
    *,
    optimizer=None,
    scheduler=None,
    best_val: float | None = None,
) -> None:
    """Save trainable parameters + (optionally) optimizer/scheduler state for resume."""
    trainable_state = {
        n: p.detach().cpu().clone()
        for n, p in sam.named_parameters()
        if p.requires_grad
    }
    payload = {
        "epoch": epoch,
        "method": cfg["method"],
        "val_dice": val_dice,
        "best_val": best_val if best_val is not None else val_dice,
        "trainable_state": trainable_state,
        "config": cfg,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, ckpt_path)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    method = cfg["method"]
    print(f"[train] device={device} method={method} seed={seed}")
    if device == "cuda":
        print(f"[train] gpu={torch.cuda.get_device_name(0)}")

    # Model
    ckpt_path = REPO_ROOT / cfg["model"]["checkpoint"]
    sam = load_medsam(ckpt_path, arch=cfg["model"]["arch"], device=device)
    method_kwargs = cfg.get("method_kwargs", {})
    info = setup_method(sam, method, **method_kwargs)
    print(
        f"[train] params total={info['total']:,} "
        f"trainable={info['trainable']:,} ({info['trainable_pct']:.3f}%)"
    )

    enc_grad = encoder_in_grad_path(method)

    # Data
    image_size = cfg["model"]["image_size"]
    train_ds = ISIC2018(
        root=REPO_ROOT / cfg["data"]["root"],
        split="train",
        image_size=image_size,
        bbox_perturb_pixels=cfg["data"].get("bbox_perturb_pixels", 0),
    )
    val_ds = ISIC2018(
        root=REPO_ROOT / cfg["data"]["root"],
        split="val",
        image_size=image_size,
        bbox_perturb_pixels=0,
    )
    print(f"[train] train_n={len(train_ds)} val_n={len(val_ds)} image_size={image_size}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 2),
        collate_fn=isic_collate,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["eval"].get("num_workers", 2),
        collate_fn=isic_collate,
        pin_memory=(device == "cuda"),
    )

    # Optimizer / scheduler
    trainable = [p for p in sam.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable,
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    epochs = int(cfg["train"]["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    amp = bool(cfg["train"].get("amp", True)) and device == "cuda"
    scaler = torch.amp.GradScaler() if amp else None
    print(f"[train] amp={amp} epochs={epochs} batch={cfg['train']['batch_size']}")

    criterion = DiceBCELoss(dice_weight=float(cfg["train"].get("dice_weight", 0.5)))

    # Output paths
    run_name = cfg["name"]
    run_dir = REPO_ROOT / cfg["output"]["checkpoint_dir"] / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.csv"
    # Append on resume so we don't lose previous epoch records
    log_mode = "a" if (args.resume and log_path.exists() and log_path.stat().st_size > 0) else "w"
    log_fh = open(log_path, log_mode, newline="")
    log_w = csv.writer(log_fh)
    if log_mode == "w":
        log_w.writerow(["epoch", "train_loss", "train_bce", "train_dice_loss",
                        "val_dice", "val_iou", "lr", "epoch_s"])

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    best_val = 0.0
    start_epoch = 1
    latest_path = run_dir / "latest.pth"
    if args.resume and latest_path.exists():
        ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
        # Restore trainable params
        trainable_state = {k: v.to(device) for k, v in ckpt["trainable_state"].items()}
        sam.load_state_dict(trainable_state, strict=False)
        # Restore optimizer state (Adam moments, not the LR — scheduler will set that)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        # IMPORTANT: do NOT load the saved scheduler state directly. If the saved
        # checkpoint came from a run with a different `epochs` value (e.g. a 1-epoch
        # smoke test), its T_max baked into the state would override our new T_max
        # and break cosine annealing. Instead, freshly-constructed scheduler is
        # advanced to the right position by stepping `completed_epochs` times,
        # which gives the correct LR for the new T_max.
        completed_epochs = int(ckpt["epoch"])
        for _ in range(completed_epochs):
            scheduler.step()
        start_epoch = completed_epochs + 1
        best_val = float(ckpt.get("best_val", ckpt.get("val_dice", 0.0)))
        cur_lr = scheduler.get_last_lr()[0]
        print(
            f"[train] resumed from {latest_path} "
            f"(completed epoch {completed_epochs}, best_val={best_val:.4f}). "
            f"Continuing at epoch {start_epoch} with lr={cur_lr:.2e}."
        )

    t_total = time.time()
    cooldown_s = float(cfg["train"].get("cooldown_seconds", 0))
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(
            sam, train_loader, optimizer, scaler, criterion, device,
            encoder_grad=enc_grad, amp=amp,
        )
        # Thermal cooldown between train and val. ViT-B encoder gradient
        # backward at 1024x1024 saturates mobile GPUs; the val pass kicks off
        # without checkpointing and spikes memory, which causes driver crashes
        # on hot hardware. Letting the GPU cool for a minute fixes this.
        if cooldown_s > 0:
            print(f"[train] cooldown {cooldown_s:.0f}s before val")
            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            time.sleep(cooldown_s)
        val_stats = validate(sam, val_loader, device, amp=amp)
        scheduler.step()
        elapsed = time.time() - t0
        cur_lr = scheduler.get_last_lr()[0]

        val_dice = val_stats["dice_mean"]
        print(
            f"[train] ep {epoch:3d}/{epochs} "
            f"loss={train_stats['loss']:.4f} "
            f"val_dice={val_dice:.4f} val_iou={val_stats['iou_mean']:.4f} "
            f"lr={cur_lr:.2e} t={elapsed:.0f}s"
        )
        log_w.writerow([
            epoch,
            f"{train_stats['loss']:.4f}",
            f"{train_stats['bce']:.4f}",
            f"{train_stats['dice_loss']:.4f}",
            f"{val_dice:.4f}",
            f"{val_stats['iou_mean']:.4f}",
            f"{cur_lr:.2e}",
            f"{elapsed:.0f}",
        ])
        log_fh.flush()

        # Always save latest (with optimizer/scheduler for resume), save best when val improves
        if val_dice > best_val:
            best_val = val_dice
            save_checkpoint(
                sam, run_dir / "best.pth", epoch, val_dice, cfg, best_val=best_val
            )
            print(f"[train]   ✓ new best val_dice={best_val:.4f}")
        save_checkpoint(
            sam, run_dir / "latest.pth", epoch, val_dice, cfg,
            optimizer=optimizer, scheduler=scheduler, best_val=best_val,
        )

    log_fh.close()
    total_min = (time.time() - t_total) / 60
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0
    print(f"[train] done in {total_min:.1f} min. best val_dice={best_val:.4f} peak={peak_mb:.0f}MB")
    print(f"[train] best checkpoint -> {run_dir / 'best.pth'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
