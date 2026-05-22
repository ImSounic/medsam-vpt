"""In-process orchestrator: evaluate zero_shot + every trained checkpoint
across all test sets in a single Python process.

Replaces the two-command sequence:
    python -m src.eval --config configs/zero_shot.yaml
    python scripts/eval_all_checkpoints.py --config configs/zero_shot.yaml

Optimizations vs. the old subprocess-based flow:

  1. Single Python startup + import. monai/tensorflow only loads once
     (vs once per checkpoint subprocess), saving ~5s × N_checkpoints.

  2. Single load of base MedSAM weights from disk. The 358 MB .pth file
     reads once into a CPU state dict; every method's SAM is built from
     that in-memory dict (saves ~2-3s × N_checkpoints).

  3. SHARED ENCODER for zero_shot + decoder_only. Both methods use the
     unmodified base MedSAM image encoder (decoder_only freezes it,
     zero_shot doesn't touch it). We compute the encoder forward ONCE per
     batch and run two different mask decoders on the same embeddings.
     The encoder is the dominant per-batch cost for these PEFT-style
     methods, so this roughly halves their combined wall-clock time.

The output (results/runs.csv rows, per-image CSVs, log lines) is
intentionally indistinguishable from the old commands — same numeric
values, same column order — so plots.py / summary_table.csv consumers
don't care which orchestrator produced the rows.

Usage:
    python scripts/eval_all_methods.py --config configs/zero_shot.yaml
    python scripts/eval_all_methods.py --config configs/zero_shot.yaml --quick

For VPT-shallow, VPT-deep, LoRA, full_ft — the encoder differs per method
(different prompts / adapters / weights), so we fall back to a per-method
loop. Still cheaper than subprocess relaunches.
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.isic import isic_collate
from src.device_utils import device_name, get_device, peak_memory_mb, reset_peak_memory
from src.eval import build_dataset, predict_from_embeddings
from src.metrics import aggregate_metrics, dice_score, hd95, iou_score
from src.models.medsam import load_medsam_from_state_dict
from src.models.methods import setup_method

REPO_ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------------
# CLI + config
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument(
        "--checkpoint-glob",
        default="checkpoints/runs/*/best.pth",
        help="Glob (relative to repo root) for trained checkpoints to evaluate.",
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Run on first 8 images per dataset — smoke-test mode.",
    )
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    return p.parse_args()


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------------
# Per-checkpoint helpers
# ----------------------------------------------------------------------------

def _read_checkpoint(ckpt_path: Path) -> dict:
    """Pull method/method_kwargs/run_name/seed/trainable_state out of a .pth."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_cfg = ckpt.get("config", {}) or {}
    return {
        "method": ckpt["method"],
        "method_kwargs": saved_cfg.get("method_kwargs", {}) or {},
        "run_name": saved_cfg.get("name", ckpt["method"]),
        "seed": int(saved_cfg.get("seed", 0)),
        "trainable_state": ckpt["trainable_state"],
        "epoch": ckpt.get("epoch"),
        "val_dice": ckpt.get("val_dice"),
    }


def _apply_method_and_weights(sam, info: dict, device: str) -> dict:
    """Wire up the method on `sam` and load its trainable params. Returns param info."""
    param_info = setup_method(sam, info["method"], **info["method_kwargs"])
    trainable_state = {k: v.to(device) for k, v in info["trainable_state"].items()}
    sam.load_state_dict(trainable_state, strict=False)
    return param_info


# ----------------------------------------------------------------------------
# Per-dataset loader + result helpers
# ----------------------------------------------------------------------------

def _make_loader(ds, cfg: dict, quick: bool) -> DataLoader:
    if quick:
        ds.items = ds.items[:8]
    return DataLoader(
        ds,
        batch_size=cfg["eval"]["batch_size"],
        num_workers=cfg["eval"].get("num_workers", 2),
        collate_fn=isic_collate,
        shuffle=False,
    )


def _save_per_image_csv(rows, run_name: str, ds_name: str, cfg: dict) -> Path:
    per_image_path = REPO_ROOT / cfg["output"].get(
        "per_image_csv", "results/raw/per_image.csv"
    )
    per_image_path = per_image_path.with_name(f"{run_name}_{ds_name}_per_image.csv")
    per_image_path.parent.mkdir(parents=True, exist_ok=True)
    with open(per_image_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "dice", "iou", "hd95"])
        w.writeheader()
        w.writerows(rows)
    return per_image_path


def _build_runs_row(
    run_name: str, method: str, ds_name: str, seed: int,
    agg: dict, param_info: dict, peak_mb: float, elapsed: float, args,
) -> dict:
    return {
        "run_name": run_name,
        "method": method,
        "dataset": ds_name,
        "seed": seed,
        "dice_mean": f"{agg['dice_mean']:.4f}",
        "dice_std": f"{agg['dice_std']:.4f}",
        "iou_mean": f"{agg['iou_mean']:.4f}",
        "hd95_mean": f"{agg['hd95_mean']:.4f}",
        "trainable_params": param_info["trainable"],
        "peak_mem_mb": f"{peak_mb:.0f}",
        "wall_clock_s": f"{elapsed:.1f}",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "notes": "quick" if args.quick else "",
    }


def _metric_triple(pred, gt):
    return dice_score(pred, gt), iou_score(pred, gt), hd95(pred, gt)


# ----------------------------------------------------------------------------
# Group 1: zero_shot + decoder_only (shared encoder)
# ----------------------------------------------------------------------------

def eval_shared_encoder_group(
    cfg: dict, args, device: str, base_state_dict: dict, do_ckpt_path: Path,
) -> list[dict]:
    """Run zero_shot and decoder_only over each dataset, sharing the encoder pass.

    Both methods share the base MedSAM image encoder (decoder_only freezes it,
    zero_shot doesn't modify it), so we encode once per batch and run two
    different mask decoders on the same embeddings.
    """
    print("\n[multi-eval] === GROUP: zero_shot + decoder_only (shared encoder) ===")
    arch = cfg["model"]["arch"]
    image_size = cfg["model"]["image_size"]

    # SAM for zero_shot
    sam_zs = load_medsam_from_state_dict(base_state_dict, arch=arch, device=device)
    info_zs = setup_method(sam_zs, "zero_shot")
    sam_zs.eval()
    print(f"[multi-eval] zero_shot ready (trainable={info_zs['trainable']:,})")

    # SAM for decoder_only
    sam_do = load_medsam_from_state_dict(base_state_dict, arch=arch, device=device)
    do_info = _read_checkpoint(do_ckpt_path)
    info_do = _apply_method_and_weights(sam_do, do_info, device)
    sam_do.eval()
    print(
        f"[multi-eval] decoder_only ({do_ckpt_path.parent.name}) ready "
        f"(trainable={info_do['trainable']:,})"
    )

    rows = []
    for ts_cfg in cfg["data"]["test_sets"]:
        ds_name = ts_cfg["name"]
        print(f"\n[multi-eval] --- {ds_name} (zero_shot + decoder_only) ---")
        ds = build_dataset(cfg, ts_cfg, image_size)
        loader = _make_loader(ds, cfg, args.quick)
        reset_peak_memory(device)

        zs_per, do_per = [], []
        zs_rows, do_rows = [], []
        t0 = time.time()
        with torch.no_grad():
            for batch in tqdm(loader, desc=ds_name):
                images = batch["image"].to(device)
                bboxes = batch["bbox"].to(device)
                masks_gt = batch["mask"].cpu().numpy()
                H, W = images.shape[-2:]

                # ONE encoder pass — both methods share base encoder weights
                embeddings = sam_zs.image_encoder(images)

                # Two cheap decoders on the same embeddings
                zs_preds = predict_from_embeddings(
                    sam_zs, embeddings, bboxes, H, W
                ).cpu().numpy()
                do_preds = predict_from_embeddings(
                    sam_do, embeddings, bboxes, H, W
                ).cpu().numpy()

                for j in range(zs_preds.shape[0]):
                    gj = masks_gt[j]
                    img_id = batch["image_id"][j]
                    for preds_j, per_list, row_list in (
                        (zs_preds[j], zs_per, zs_rows),
                        (do_preds[j], do_per, do_rows),
                    ):
                        d, i_, h_ = _metric_triple(preds_j, gj)
                        per_list.append({"dice": d, "iou": i_, "hd95": h_})
                        row_list.append({
                            "image_id": img_id, "dice": d, "iou": i_, "hd95": h_,
                        })
        elapsed = time.time() - t0
        peak_mb = peak_memory_mb(device)

        for per_image, per_rows, info, run_name, method, seed in (
            (zs_per, zs_rows, info_zs, "zero_shot", "zero_shot", 0),
            (do_per, do_rows, info_do, do_info["run_name"], "decoder_only", do_info["seed"]),
        ):
            agg = aggregate_metrics(per_image)
            print(
                f"[multi-eval] {ds_name} / {method}: "
                f"dice={agg['dice_mean']:.4f}±{agg['dice_std']:.4f} "
                f"iou={agg['iou_mean']:.4f} hd95={agg['hd95_mean']:.2f}px "
                f"n={len(per_image)}"
            )
            rows.append(_build_runs_row(
                run_name, method, ds_name, seed,
                agg, info, peak_mb, elapsed, args,
            ))
            csv_path = _save_per_image_csv(per_rows, run_name, ds_name, cfg)
            print(f"[multi-eval]   per-image -> {csv_path}")
        print(f"[multi-eval] {ds_name} group time: {elapsed:.1f}s peak={peak_mb:.0f}MB")

    del sam_zs, sam_do
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows


# ----------------------------------------------------------------------------
# Groups 2+: per-method eval (VPT-shallow, VPT-deep, LoRA, full_ft)
# ----------------------------------------------------------------------------

def eval_one_checkpoint(
    cfg: dict, args, device: str, base_state_dict: dict, ckpt_path: Path,
) -> list[dict]:
    arch = cfg["model"]["arch"]
    image_size = cfg["model"]["image_size"]

    sam = load_medsam_from_state_dict(base_state_dict, arch=arch, device=device)
    info = _read_checkpoint(ckpt_path)
    param_info = _apply_method_and_weights(sam, info, device)
    sam.eval()
    print(
        f"\n[multi-eval] === {info['method']} ({ckpt_path.parent.name}) === "
        f"trainable={param_info['trainable']:,}"
    )
    if info["epoch"] is not None:
        print(
            f"[multi-eval]   ckpt: epoch={info['epoch']} val_dice={info['val_dice']:.4f}"
        )

    rows = []
    for ts_cfg in cfg["data"]["test_sets"]:
        ds_name = ts_cfg["name"]
        print(f"[multi-eval] --- {ds_name} ---")
        ds = build_dataset(cfg, ts_cfg, image_size)
        loader = _make_loader(ds, cfg, args.quick)
        reset_peak_memory(device)

        per_image, per_rows = [], []
        t0 = time.time()
        with torch.no_grad():
            for batch in tqdm(loader, desc=ds_name):
                images = batch["image"].to(device)
                bboxes = batch["bbox"].to(device)
                masks_gt = batch["mask"].cpu().numpy()
                H, W = images.shape[-2:]
                embeddings = sam.image_encoder(images)
                preds = predict_from_embeddings(
                    sam, embeddings, bboxes, H, W
                ).cpu().numpy()
                for j in range(preds.shape[0]):
                    pj, gj = preds[j], masks_gt[j]
                    d, i_, h_ = _metric_triple(pj, gj)
                    per_image.append({"dice": d, "iou": i_, "hd95": h_})
                    per_rows.append({
                        "image_id": batch["image_id"][j],
                        "dice": d, "iou": i_, "hd95": h_,
                    })
        elapsed = time.time() - t0
        peak_mb = peak_memory_mb(device)
        agg = aggregate_metrics(per_image)
        print(
            f"[multi-eval] {ds_name}: dice={agg['dice_mean']:.4f}±{agg['dice_std']:.4f} "
            f"iou={agg['iou_mean']:.4f} hd95={agg['hd95_mean']:.2f}px "
            f"n={len(per_image)} time={elapsed:.1f}s peak={peak_mb:.0f}MB"
        )
        rows.append(_build_runs_row(
            info["run_name"], info["method"], ds_name, info["seed"],
            agg, param_info, peak_mb, elapsed, args,
        ))
        csv_path = _save_per_image_csv(per_rows, info["run_name"], ds_name, cfg)
        print(f"[multi-eval]   per-image -> {csv_path}")

    del sam
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows


# ----------------------------------------------------------------------------
# Zero-shot fallback (used only when no decoder_only checkpoint exists)
# ----------------------------------------------------------------------------

def eval_zero_shot_only(cfg, args, device, base_state_dict):
    arch = cfg["model"]["arch"]
    image_size = cfg["model"]["image_size"]
    sam = load_medsam_from_state_dict(base_state_dict, arch=arch, device=device)
    info = setup_method(sam, "zero_shot")
    sam.eval()

    rows = []
    for ts_cfg in cfg["data"]["test_sets"]:
        ds_name = ts_cfg["name"]
        ds = build_dataset(cfg, ts_cfg, image_size)
        loader = _make_loader(ds, cfg, args.quick)
        reset_peak_memory(device)
        per_image, per_rows = [], []
        t0 = time.time()
        with torch.no_grad():
            for batch in tqdm(loader, desc=ds_name):
                images = batch["image"].to(device)
                bboxes = batch["bbox"].to(device)
                masks_gt = batch["mask"].cpu().numpy()
                H, W = images.shape[-2:]
                embeddings = sam.image_encoder(images)
                preds = predict_from_embeddings(sam, embeddings, bboxes, H, W).cpu().numpy()
                for j in range(preds.shape[0]):
                    pj, gj = preds[j], masks_gt[j]
                    d, i_, h_ = _metric_triple(pj, gj)
                    per_image.append({"dice": d, "iou": i_, "hd95": h_})
                    per_rows.append({
                        "image_id": batch["image_id"][j],
                        "dice": d, "iou": i_, "hd95": h_,
                    })
        elapsed = time.time() - t0
        peak_mb = peak_memory_mb(device)
        agg = aggregate_metrics(per_image)
        rows.append(_build_runs_row(
            "zero_shot", "zero_shot", ds_name, 0,
            agg, info, peak_mb, elapsed, args,
        ))
        _save_per_image_csv(per_rows, "zero_shot", ds_name, cfg)
    del sam
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows


# ----------------------------------------------------------------------------
# Results CSV append
# ----------------------------------------------------------------------------

def append_to_runs_csv(rows: list[dict], runs_path: Path) -> None:
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = runs_path.exists() and runs_path.stat().st_size > 0
    with open(runs_path, "a", newline="") as f:
        fields = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            w.writeheader()
        w.writerows(rows)
    print(f"\n[multi-eval] appended {len(rows)} rows to {runs_path}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device(prefer=args.device)
    image_size = cfg["model"]["image_size"]
    batch_size = cfg["eval"]["batch_size"]
    print(
        f"[multi-eval] device={device} ({device_name(device)}) "
        f"image_size={image_size} batch={batch_size}"
    )

    # Load base MedSAM into CPU state dict ONCE. Every method-specific SAM
    # is built from this in-memory dict — no repeated 358 MB disk reads.
    base_ckpt = REPO_ROOT / cfg["model"]["checkpoint"]
    print(f"[multi-eval] loading base weights from {base_ckpt}")
    t0 = time.time()
    base_state_dict = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    print(f"[multi-eval] base loaded in {time.time() - t0:.1f}s")

    # Discover trained checkpoints
    checkpoints = sorted(REPO_ROOT.glob(args.checkpoint_glob))
    print(f"[multi-eval] found {len(checkpoints)} trained checkpoint(s):")
    for c in checkpoints:
        print(f"             {c.relative_to(REPO_ROOT)}")

    all_rows: list[dict] = []
    t_wall = time.time()

    # ---- Group 1: zero_shot + decoder_only (shared encoder) ----
    do_ckpt = next(
        (c for c in checkpoints if "decoder_only" in c.parent.name),
        None,
    )
    if do_ckpt is not None:
        all_rows.extend(eval_shared_encoder_group(
            cfg, args, device, base_state_dict, do_ckpt,
        ))
    else:
        print("[multi-eval] no decoder_only checkpoint — running zero_shot standalone")
        all_rows.extend(eval_zero_shot_only(cfg, args, device, base_state_dict))

    # ---- Groups 2+: standard per-method eval ----
    for ckpt in checkpoints:
        if "decoder_only" in ckpt.parent.name:
            continue  # already handled in Group 1
        all_rows.extend(eval_one_checkpoint(
            cfg, args, device, base_state_dict, ckpt,
        ))

    # Write all rows at the end (single append, ordered by group)
    runs_path = REPO_ROOT / cfg["output"]["results_csv"]
    append_to_runs_csv(all_rows, runs_path)

    total_min = (time.time() - t_wall) / 60
    print(f"[multi-eval] done. {len(all_rows)} rows in {total_min:.1f} min.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
