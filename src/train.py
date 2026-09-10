"""Method-agnostic training entry point; checkpoints store only trainable params."""

from __future__ import annotations

import argparse
import csv
import random
import time
from datetime import datetime
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.cka import flatten_for_cka, linear_cka
from src.data.isic import ISIC2018, isic_collate
from src.device_utils import (
    autocast_device_type,
    device_name,
    empty_cache,
    get_device,
    peak_memory_mb,
    reset_peak_memory,
    seed_all,
    supports_amp,
    supports_pin_memory,
    synchronize,
)
from src.losses import DiceBCELoss
from src.metrics import aggregate_metrics, dice_score, iou_score
from src.models.medsam import load_medsam
from src.models.methods import encoder_in_grad_path, setup_method
from src.train_schedule import (
    apply_quick_overrides,
    cycle_loader,
    plan_segments,
    resolve_budget,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

_AUTOCAST_DEVICE = "cuda"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--seed", type=int, default=None, help="Override config seed")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest.pth in the run directory if present.",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: 8 train / 4 val images, 1 epoch or 6 steps, tiny CKA probe.",
    )
    return p.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    seed_all(seed)  # handles torch + cuda + mps


def forward_with_prompt(
    sam,
    images: torch.Tensor,
    bboxes: torch.Tensor,
    encoder_grad: bool,
    grad_checkpoint: bool = False,
) -> torch.Tensor:
    """Run MedSAM forward with bbox prompts. Returns logits (B, 1, H, W)."""
    if encoder_grad:
        if grad_checkpoint:
            # Per-block checkpointing on the encoder (same maths, less memory).
            from cka.probe import _encoder_forward_checkpointed

            image_emb = _encoder_forward_checkpointed(sam.image_encoder, images)
        else:
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
        masks_out.append(
            F.interpolate(low_res, size=(H, W), mode="bilinear", align_corners=False)
        )
    return torch.cat(masks_out, dim=0)  # (B, 1, H, W)


def train_one_epoch(
    sam,
    loader,
    optimizer,
    scaler,
    criterion,
    device,
    *,
    encoder_grad: bool,
    amp: bool,
    cka_ctx: dict | None = None,
    max_steps: int | None = None,
    step_scheduler=None,
    grad_checkpoint: bool = False,
) -> dict:
    sam.train()
    losses, bces, dlosses, cka_losses = [], [], [], []
    per_layer_cka_history: dict[str, list[float]] = (
        {n: [] for n in cka_ctx["base_acts"]} if cka_ctx else {}
    )
    cka_every_n = cka_ctx["every_n_steps"] if cka_ctx else 1
    batches = islice(loader, max_steps) if max_steps is not None else loader
    total = max_steps if max_steps is not None else len(loader)
    pbar = tqdm(batches, desc="train", leave=False, total=total)
    for step_idx, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        bboxes = batch["bbox"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=_AUTOCAST_DEVICE, enabled=amp):
            logits = forward_with_prompt(
                sam,
                images,
                bboxes,
                encoder_grad=encoder_grad,
                grad_checkpoint=grad_checkpoint,
            )
            task_loss, parts = criterion(logits, masks)

        task_loss_val = float(task_loss.detach().item())
        if scaler is not None:
            scaler.scale(task_loss).backward()
        else:
            task_loss.backward()
        del logits, task_loss
        if cka_ctx is not None:
            # Hooks also fire during the task forward; drop those activations now.
            cka_ctx["hook_handle"].clear()

        cka_loss_val = 0.0
        do_cka_this_step = (cka_ctx is not None) and (step_idx % cka_every_n == 0)
        if do_cka_this_step:
            from cka.probe import run_current_probe_forward

            effective_lambda = float(cka_ctx["lambda_cka"]) * cka_every_n
            with torch.autocast(device_type=_AUTOCAST_DEVICE, enabled=amp):
                run_current_probe_forward(
                    sam,
                    cka_ctx["probe_batch"],
                    hook_handle=cka_ctx["hook_handle"],
                    encoder_chunk=cka_ctx["encoder_chunk"],
                    use_grad_checkpoint=cka_ctx["use_grad_checkpoint"],
                )
            cur_acts = cka_ctx["hook_handle"].stacked()

            with torch.autocast(device_type=_AUTOCAST_DEVICE, enabled=False):
                per_layer_losses = []
                for name, base_act in cka_ctx["base_acts"].items():
                    cur_act = cur_acts.get(name)
                    if cur_act is None:
                        continue
                    X = flatten_for_cka(base_act.float())
                    Y = flatten_for_cka(cur_act.float())
                    sim = linear_cka(X, Y)
                    w = float(cka_ctx["layer_weights"].get(name, 1.0))
                    per_layer_losses.append(w * (1.0 - sim))
                    per_layer_cka_history[name].append(float(sim.detach().item()))
                if per_layer_losses:
                    cka_loss = torch.stack(per_layer_losses).sum()
                    scaled_cka = effective_lambda * cka_loss
                    cka_loss_val = float(cka_loss.detach().item())
                else:
                    scaled_cka = None

            if scaled_cka is not None:
                if scaler is not None:
                    scaler.scale(scaled_cka).backward()
                else:
                    scaled_cka.backward()

            cka_ctx["hook_handle"].clear()
            del cur_acts
            if scaled_cka is not None:
                del cka_loss, scaled_cka

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if step_scheduler is not None:
            step_scheduler()

        total_loss_val = task_loss_val + (
            float(cka_ctx["lambda_cka"]) * cka_loss_val
            if (cka_ctx and do_cka_this_step)
            else 0.0
        )
        losses.append(total_loss_val)
        bces.append(parts["bce"])
        dlosses.append(parts["dice_loss"])
        if cka_ctx is not None:
            if do_cka_this_step:
                cka_losses.append(cka_loss_val)
        postfix = {
            "loss": f"{total_loss_val:.3f}",
            "dice_l": f"{parts['dice_loss']:.3f}",
        }
        if cka_ctx is not None:
            # "-" on skipped steps
            postfix["cka_l"] = f"{cka_loss_val:.3f}" if do_cka_this_step else "-"
        pbar.set_postfix(postfix)

    out = {
        "loss": float(np.mean(losses)),
        "bce": float(np.mean(bces)),
        "dice_loss": float(np.mean(dlosses)),
    }
    if cka_ctx is not None:
        out["cka_loss"] = float(np.mean(cka_losses))
        for name, hist in per_layer_cka_history.items():
            out[f"cka_{name}"] = float(np.mean(hist)) if hist else float("nan")
    return out


@torch.no_grad()
def validate(sam, loader, device, *, amp: bool) -> dict:
    sam.eval()
    per_image = []
    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        bboxes = batch["bbox"].to(device, non_blocking=True)
        masks_gt = batch["mask"].cpu().numpy()
        with torch.autocast(device_type=_AUTOCAST_DEVICE, enabled=amp):
            logits = forward_with_prompt(sam, images, bboxes, encoder_grad=False)
        preds = (logits.squeeze(1) > 0).cpu().numpy().astype("uint8")
        for j in range(preds.shape[0]):
            per_image.append(
                {
                    "dice": dice_score(preds[j], masks_gt[j]),
                    "iou": iou_score(preds[j], masks_gt[j]),
                }
            )
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
    step: int = 0,
) -> None:
    """Save trainable params, optionally optimizer/scheduler state for resume."""
    trainable_state = {
        n: p.detach().cpu().clone()
        for n, p in sam.named_parameters()
        if p.requires_grad
    }
    payload = {
        "epoch": epoch,
        "step": step,
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
    if args.quick:
        apply_quick_overrides(cfg)
        print("[train] --quick: tiny data, budget and probe")
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    set_seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = get_device()  # cuda > mps > cpu
    method = cfg["method"]
    print(
        f"[train] device={device} ({device_name(device)}) method={method} seed={seed}"
    )

    global _AUTOCAST_DEVICE
    _AUTOCAST_DEVICE = autocast_device_type(device)

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
    grad_ckpt = bool(cfg["train"].get("grad_checkpoint", False))

    # Data
    image_size = cfg["model"]["image_size"]
    train_ds = ISIC2018(
        root=REPO_ROOT / cfg["data"]["root"],
        split="train",
        image_size=image_size,
        bbox_perturb_pixels=cfg["data"].get("bbox_perturb_pixels", 0),
        random_perturb=bool(cfg["data"].get("random_perturb", False)),
        max_train_samples=cfg["data"].get("max_train_samples"),
        subset_seed=int(cfg["data"].get("subset_seed", 0)),
    )
    val_ds = ISIC2018(
        root=REPO_ROOT / cfg["data"]["root"],
        split="val",
        image_size=image_size,
        bbox_perturb_pixels=0,
    )
    if args.quick:
        train_ds.items = train_ds.items[:8]
        val_ds.items = val_ds.items[:4]
    print(
        f"[train] train_n={len(train_ds)} val_n={len(val_ds)} image_size={image_size}"
    )

    train_nw = cfg["train"].get("num_workers", 8)
    eval_nw = cfg["eval"].get("num_workers", 8)
    train_extra = (
        {"persistent_workers": True, "prefetch_factor": 4} if train_nw > 0 else {}
    )
    eval_extra = (
        {"persistent_workers": True, "prefetch_factor": 4} if eval_nw > 0 else {}
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=train_nw,
        collate_fn=isic_collate,
        pin_memory=supports_pin_memory(device),
        **train_extra,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=eval_nw,
        collate_fn=isic_collate,
        pin_memory=supports_pin_memory(device),
        **eval_extra,
    )

    # Optimizer / scheduler
    trainable = [p for p in sam.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable,
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    budget = resolve_budget(cfg["train"])
    if budget.mode == "steps":
        scheduler = CosineAnnealingLR(optimizer, T_max=budget.max_steps)
        segments = plan_segments(budget.max_steps, budget.val_every_steps)
        per_step_sched = scheduler.step
        seg_label = "seg"
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=budget.n_segments)
        segments = [len(train_loader)] * budget.n_segments
        per_step_sched = None
        seg_label = "ep"
    n_segments = len(segments)
    stream = cycle_loader(train_loader)

    amp = bool(cfg["train"].get("amp", True)) and supports_amp(device)
    scaler = torch.amp.GradScaler() if amp else None
    print(
        f"[train] amp={amp} mode={budget.mode} segments={n_segments} "
        f"max_steps={budget.max_steps} batch={cfg['train']['batch_size']}"
    )

    criterion = DiceBCELoss(dice_weight=float(cfg["train"].get("dice_weight", 0.5)))

    # Optional CKA-aware training setup
    cka_cfg = cfg.get("cka_regularization") or {}
    cka_ctx = None
    if cka_cfg.get("enabled"):
        from cka.hooks import register_hooks
        from cka.probe import build_probe_batch, cache_base_activations

        layer_names = list(cka_cfg["hook_layers"])
        lambda_cka = float(cka_cfg["lambda"])
        layer_weights = dict(cka_cfg.get("weights") or {})
        probe_seed = int(cka_cfg.get("probe_seed", 42))
        n_isic = int(cka_cfg.get("n_isic", 12))
        n_busi = int(cka_cfg.get("n_busi", 10))
        n_cbis = int(cka_cfg.get("n_cbis", 10))

        print(f"[train][cka] enabled  lambda={lambda_cka}  layers={layer_names}")
        print(
            f"[train][cka] probe: ISIC={n_isic} BUSI={n_busi} CBIS={n_cbis} "
            f"seed={probe_seed}"
        )

        probe_batch = build_probe_batch(
            repo_root=REPO_ROOT,
            image_size=image_size,
            n_isic=n_isic,
            n_busi=n_busi,
            n_cbis=n_cbis,
            probe_seed=probe_seed,
            device=device,
        )
        print(f"[train][cka] probe batch built: {probe_batch['image'].shape}")

        base_for_probe = load_medsam(
            ckpt_path, arch=cfg["model"]["arch"], device=device
        )
        for p in base_for_probe.parameters():
            p.requires_grad = False
        base_acts = cache_base_activations(
            base_for_probe,
            probe_batch,
            layer_names,
            encoder_chunk=int(cka_cfg.get("encoder_chunk", 4)),
        )
        print(
            f"[train][cka] base activations cached: "
            f"{ {n: tuple(a.shape) for n, a in base_acts.items()} }"
        )
        del base_for_probe  # only needed for one forward
        empty_cache(device)

        hook_handle = register_hooks(sam, layer_names, detach=False, accumulate=True)

        cka_ctx = {
            "probe_batch": probe_batch,
            "base_acts": base_acts,
            "hook_handle": hook_handle,
            "lambda_cka": lambda_cka,
            "layer_weights": layer_weights,
            "encoder_chunk": int(cka_cfg.get("encoder_chunk", 16)),
            "use_grad_checkpoint": bool(cka_cfg.get("use_grad_checkpoint", True)),
            "every_n_steps": int(cka_cfg.get("every_n_steps", 1)),
        }

    run_name = cfg["name"]
    run_dir = REPO_ROOT / cfg["output"]["checkpoint_dir"] / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.csv"
    log_mode = (
        "a"
        if (args.resume and log_path.exists() and log_path.stat().st_size > 0)
        else "w"
    )
    log_fh = open(log_path, log_mode, newline="")
    log_w = csv.writer(log_fh)
    log_header = [
        "epoch",
        "step",
        "train_loss",
        "train_bce",
        "train_dice_loss",
        "val_dice",
        "val_iou",
        "lr",
        "epoch_s",
    ]
    if cka_ctx is not None:
        log_header.append("cka_loss")
        for name in cka_ctx["base_acts"]:
            log_header.append(f"cka_{name}")
    if log_mode == "w":
        log_w.writerow(log_header)

    reset_peak_memory(device)

    best_val = 0.0
    start_segment = 1
    global_step = 0
    latest_path = run_dir / "latest.pth"
    if args.resume and latest_path.exists():
        ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
        trainable_state = {k: v.to(device) for k, v in ckpt["trainable_state"].items()}
        sam.load_state_dict(trainable_state, strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        completed = int(ckpt["epoch"])
        global_step = int(ckpt.get("step", 0))
        n_sched = global_step if budget.mode == "steps" else completed
        for _ in range(n_sched):
            scheduler.step()
        start_segment = completed + 1
        best_val = float(ckpt.get("best_val", ckpt.get("val_dice", 0.0)))
        cur_lr = scheduler.get_last_lr()[0]
        print(
            f"[train] resumed from {latest_path} "
            f"(completed {seg_label} {completed}, step {global_step}, "
            f"best_val={best_val:.4f}). Continuing at {seg_label} {start_segment} "
            f"with lr={cur_lr:.2e}."
        )

    t_total = time.time()
    cooldown_s = float(cfg["train"].get("cooldown_seconds", 0))
    for seg in range(start_segment, n_segments + 1):
        n_steps = segments[seg - 1]
        t0 = time.time()
        train_stats = train_one_epoch(
            sam,
            stream,
            optimizer,
            scaler,
            criterion,
            device,
            encoder_grad=enc_grad,
            amp=amp,
            cka_ctx=cka_ctx,
            max_steps=n_steps,
            step_scheduler=per_step_sched,
            grad_checkpoint=grad_ckpt,
        )
        global_step += n_steps
        if cooldown_s > 0:
            print(f"[train] cooldown {cooldown_s:.0f}s before val")
            synchronize(device)
            empty_cache(device)
            import gc

            gc.collect()
            time.sleep(cooldown_s)
        val_stats = validate(sam, val_loader, device, amp=amp)
        if per_step_sched is None:
            scheduler.step()
        elapsed = time.time() - t0
        cur_lr = scheduler.get_last_lr()[0]

        val_dice = val_stats["dice_mean"]
        print(
            f"[train] {seg_label} {seg:3d}/{n_segments} step={global_step} "
            f"loss={train_stats['loss']:.4f} "
            f"val_dice={val_dice:.4f} val_iou={val_stats['iou_mean']:.4f} "
            f"lr={cur_lr:.2e} t={elapsed:.0f}s"
        )
        row = [
            seg,
            global_step,
            f"{train_stats['loss']:.4f}",
            f"{train_stats['bce']:.4f}",
            f"{train_stats['dice_loss']:.4f}",
            f"{val_dice:.4f}",
            f"{val_stats['iou_mean']:.4f}",
            f"{cur_lr:.2e}",
            f"{elapsed:.0f}",
        ]
        if cka_ctx is not None:
            row.append(f"{train_stats.get('cka_loss', 0.0):.4f}")
            for name in cka_ctx["base_acts"]:
                v = train_stats.get(f"cka_{name}", float("nan"))
                row.append(f"{v:.4f}" if v == v else "nan")
        log_w.writerow(row)
        log_fh.flush()

        if val_dice > best_val:
            best_val = val_dice
            save_checkpoint(
                sam,
                run_dir / "best.pth",
                seg,
                val_dice,
                cfg,
                best_val=best_val,
                step=global_step,
            )
            print(f"[train]   new best val_dice={best_val:.4f}")
        save_checkpoint(
            sam,
            run_dir / "latest.pth",
            seg,
            val_dice,
            cfg,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val=best_val,
            step=global_step,
        )

    log_fh.close()
    if cka_ctx is not None:
        cka_ctx["hook_handle"].remove()
    total_min = (time.time() - t_total) / 60
    peak_mb = peak_memory_mb(device)
    print(
        f"[train] done in {total_min:.1f} min. best val_dice={best_val:.4f} peak={peak_mb:.0f}MB"
    )
    print(f"[train] best checkpoint -> {run_dir / 'best.pth'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
