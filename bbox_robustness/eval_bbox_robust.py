"""Bbox prompt robustness evaluation: expand each tight-bbox side by a random offset in [0, perturb_max_px], N=5 deterministically-seeded samples per image."""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.isic import isic_collate  # noqa: E402
from src.device_utils import (  # noqa: E402
    device_name,
    get_device,
    peak_memory_mb,
    reset_peak_memory,
)
from src.eval import build_dataset  # noqa: E402
from src.metrics import aggregate_metrics, dice_score, hd95, iou_score  # noqa: E402
from src.models.medsam import load_medsam_from_state_dict  # noqa: E402
from src.models.methods import setup_method  # noqa: E402


PERTURB_LEVELS_DEFAULT = [20, 50, 100, 200]
N_SAMPLES_DEFAULT = 5
# Module-level global so the save helpers see --out-dir overrides set in main().
OUT_DIR = REPO_ROOT / "bbox_robustness" / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument(
        "--checkpoint-glob",
        default="checkpoints/runs/*/best.pth",
        help="Glob for trained checkpoints (relative to repo root).",
    )
    p.add_argument(
        "--perturb-levels", type=int, nargs="+", default=PERTURB_LEVELS_DEFAULT,
        help=f"Max expansion per side, in pixels (default: {PERTURB_LEVELS_DEFAULT}).",
    )
    p.add_argument(
        "--n-samples", type=int, default=N_SAMPLES_DEFAULT,
        help=f"Random bbox samples per image per level (default: {N_SAMPLES_DEFAULT}).",
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Smoke-test mode: use only the first 8 images per dataset.",
    )
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    p.add_argument(
        "--out-dir",
        type=Path, default=None,
        help="Override the output directory (runs.csv + per_image/*.csv). "
             "Useful when evaluating a separate set of checkpoints (e.g. pm=20 "
             "trained) without clobbering the baseline results. "
             "Defaults to bbox_robustness/results/.",
    )
    return p.parse_args()


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_rng(image_id: str, perturb_max: int, sample_idx: int) -> np.random.Generator:
    """Deterministic RNG keyed on (image_id, perturb level, sample idx), independent of dataloader ordering or worker count."""
    seed_str = f"{image_id}|{perturb_max}|{sample_idx}"
    seed = int(hashlib.sha1(seed_str.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def expand_bbox(
    tight_bbox: np.ndarray,
    perturb_max: int,
    image_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Expand each side independently by a random offset in [0, perturb_max], clamped to [0, image_size-1] so it always contains the input bbox."""
    x1, y1, x2, y2 = tight_bbox
    dx_left = rng.integers(0, perturb_max + 1)
    dy_top = rng.integers(0, perturb_max + 1)
    dx_right = rng.integers(0, perturb_max + 1)
    dy_bottom = rng.integers(0, perturb_max + 1)
    new_x1 = max(0, x1 - dx_left)
    new_y1 = max(0, y1 - dy_top)
    new_x2 = min(image_size - 1, x2 + dx_right)
    new_y2 = min(image_size - 1, y2 + dy_bottom)
    return np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float32)


def read_checkpoint(ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_cfg = ckpt.get("config", {}) or {}
    return {
        "method": ckpt["method"],
        "method_kwargs": saved_cfg.get("method_kwargs", {}) or {},
        "run_name": saved_cfg.get("name", ckpt["method"]),
        "seed": int(saved_cfg.get("seed", 0)),
        "trainable_state": ckpt["trainable_state"],
    }


def apply_method_and_weights(sam, info: dict, device: str) -> dict:
    param_info = setup_method(sam, info["method"], **info["method_kwargs"])
    if info["trainable_state"]:
        ts = {k: v.to(device) for k, v in info["trainable_state"].items()}
        sam.load_state_dict(ts, strict=False)
    return param_info


@torch.no_grad()
def decode_image_with_prompts(
    sam, image_embedding: torch.Tensor, bboxes: torch.Tensor, H: int, W: int
) -> torch.Tensor:
    """Run prompt encoder + mask decoder for K box prompts on one image via SAM's native batching; bit-identical to K separate single-prompt calls but ~3x faster."""
    sparse, dense = sam.prompt_encoder(points=None, boxes=bboxes, masks=None)
    low_res, _ = sam.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    masks = F.interpolate(low_res, size=(H, W), mode="bilinear", align_corners=False)
    return (masks > 0).to(torch.uint8).squeeze(1)


def make_loader(ds, cfg: dict, quick: bool) -> DataLoader:
    if quick:
        ds.items = ds.items[:8]
    return DataLoader(
        ds,
        batch_size=cfg["eval"]["batch_size"],
        num_workers=cfg["eval"].get("num_workers", 2),
        collate_fn=isic_collate,
        shuffle=False,
    )


def build_runs_row(
    run_name: str, method: str, dataset: str, seed: int,
    perturb_max: int, n_samples: int, agg: dict, n_images: int,
    param_info: dict, peak_mb: float, elapsed: float,
) -> dict:
    """Row schema for bbox_robustness/results/runs.csv."""
    return {
        "run_name": run_name,
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "perturb_max_px": perturb_max,
        "n_samples": n_samples,
        "dice_mean": f"{agg['dice_mean']:.4f}",
        "dice_std": f"{agg['dice_std']:.4f}",
        "iou_mean": f"{agg['iou_mean']:.4f}",
        "hd95_mean": f"{agg['hd95_mean']:.4f}",
        "n_images": n_images,
        "trainable_params": param_info["trainable"],
        "peak_mem_mb": f"{peak_mb:.0f}",
        "wall_clock_s": f"{elapsed:.1f}",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def save_per_image_csv(rows: list, run_name: str, ds_name: str, perturb_max: int) -> Path:
    """Per-image CSV: one row per image, mean+std across N samples."""
    p = OUT_DIR / "per_image" / f"{run_name}_{ds_name}_pm{perturb_max}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "dice_mean", "dice_std",
        "iou_mean", "iou_std",
        "hd95_mean", "hd95_std",
    ]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return p


def eval_method_on_datasets(
    sam, run_name: str, method: str, seed: int, param_info: dict,
    cfg: dict, args, device: str, image_size: int,
    perturb_levels: list, n_samples: int,
) -> list:
    """Evaluate one (already loaded) method across all datasets and all levels."""
    rows = []
    for ts_cfg in cfg["data"]["test_sets"]:
        ds_name = ts_cfg["name"]
        print(f"\n[bbox-robust] --- {ds_name} / {method} ---")
        ds = build_dataset(cfg, ts_cfg, image_size)
        loader = make_loader(ds, cfg, args.quick)
        reset_peak_memory(device)

        # Per-perturb-level accumulators: means feed runs.csv aggregation, csv_rows (mean+std) go to per_image/*.csv
        per_image_means = {pm: [] for pm in perturb_levels}
        per_image_csv_rows = {pm: [] for pm in perturb_levels}

        t0 = time.time()
        for batch in tqdm(loader, desc=f"{ds_name}"):
            images = batch["image"].to(device)
            tight_bboxes = batch["bbox"].cpu().numpy()  # (B, 4)
            masks_gt = batch["mask"].cpu().numpy()
            image_ids = batch["image_id"]
            B, _, H, W = images.shape

            with torch.no_grad():
                embeddings = sam.image_encoder(images)  # (B, 256, H/16, W/16)

            for i in range(B):
                img_id = image_ids[i]
                tight = tight_bboxes[i]
                gt = masks_gt[i]
                emb_i = embeddings[i:i + 1]  # (1, 256, 64, 64)

                for pm in perturb_levels:
                    # N expanded bboxes, deterministic by (image, pm, sample)
                    boxes_np = np.stack([
                        expand_bbox(tight, pm, image_size, make_rng(img_id, pm, s))
                        for s in range(n_samples)
                    ])  # (N, 4)
                    boxes_t = torch.from_numpy(boxes_np).to(device).float()

                    # All N samples in one batched decoder call
                    preds = decode_image_with_prompts(
                        sam, emb_i, boxes_t, H, W
                    ).cpu().numpy()  # (N, H, W)

                    # Per-sample metrics, then aggregate to per-image
                    dices = np.empty(n_samples, dtype=np.float64)
                    ious = np.empty(n_samples, dtype=np.float64)
                    hds = np.empty(n_samples, dtype=np.float64)
                    for s in range(n_samples):
                        dices[s] = dice_score(preds[s], gt)
                        ious[s] = iou_score(preds[s], gt)
                        hds[s] = hd95(preds[s], gt)

                    # Per-image mean: drives dataset-level aggregation
                    per_image_means[pm].append({
                        "dice": float(dices.mean()),
                        "iou": float(ious.mean()),
                        "hd95": float(np.where(np.isinf(hds), np.nan, hds).mean()
                                      if not np.all(np.isinf(hds))
                                      else float("inf")),
                    })
                    # Per-image mean+std: written to CSV
                    per_image_csv_rows[pm].append({
                        "image_id": img_id,
                        "dice_mean": f"{float(dices.mean()):.6f}",
                        "dice_std":  f"{float(dices.std(ddof=1) if n_samples > 1 else 0.0):.6f}",
                        "iou_mean":  f"{float(ious.mean()):.6f}",
                        "iou_std":   f"{float(ious.std(ddof=1) if n_samples > 1 else 0.0):.6f}",
                        "hd95_mean": f"{float(hds.mean()):.6f}",
                        "hd95_std":  f"{float(hds.std(ddof=1) if n_samples > 1 else 0.0):.6f}",
                    })

        elapsed = time.time() - t0
        peak_mb = peak_memory_mb(device)

        for pm in perturb_levels:
            agg = aggregate_metrics(per_image_means[pm])
            n_images = len(per_image_means[pm])
            row = build_runs_row(
                run_name, method, ds_name, seed, pm, n_samples,
                agg, n_images, param_info, peak_mb, elapsed,
            )
            rows.append(row)
            csv_path = save_per_image_csv(per_image_csv_rows[pm], run_name, ds_name, pm)
            print(
                f"[bbox-robust]   pm={pm:>3}: "
                f"dice={agg['dice_mean']:.4f}±{agg['dice_std']:.4f} "
                f"iou={agg['iou_mean']:.4f} hd95={agg['hd95_mean']:.2f}px "
                f"n_images={n_images} -> {csv_path.name}"
            )
        print(f"[bbox-robust] {ds_name} time={elapsed:.1f}s peak={peak_mb:.0f}MB")

    return rows


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device(prefer=args.device)
    image_size = cfg["model"]["image_size"]
    batch_size = cfg["eval"]["batch_size"]

    print(
        f"[bbox-robust] device={device} ({device_name(device)}) "
        f"image_size={image_size} batch={batch_size}"
    )
    print(f"[bbox-robust] perturb levels (max px per side): {args.perturb_levels}")
    print(f"[bbox-robust] samples per image per level: {args.n_samples}")

    # Apply --out-dir override via the module-level global before any save_per_image_csv calls
    global OUT_DIR
    if args.out_dir is not None:
        OUT_DIR = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir
    print(f"[bbox-robust] output dir: {OUT_DIR}")

    # Load base MedSAM weights once into CPU memory
    base_ckpt_path = REPO_ROOT / cfg["model"]["checkpoint"]
    print(f"[bbox-robust] loading base weights from {base_ckpt_path}")
    t0 = time.time()
    base_sd = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    if "model" in base_sd and isinstance(base_sd["model"], dict):
        base_sd = base_sd["model"]
    print(f"[bbox-robust] base loaded in {time.time() - t0:.1f}s")

    # Discover trained checkpoints
    checkpoints = sorted(REPO_ROOT.glob(args.checkpoint_glob))
    print(f"[bbox-robust] found {len(checkpoints)} trained checkpoint(s):")
    for c in checkpoints:
        print(f"             {c.relative_to(REPO_ROOT)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list = []
    t_wall = time.time()
    arch = cfg["model"]["arch"]

    # zero_shot first: unmodified base encoder, no checkpoint
    print("\n[bbox-robust] ============================================")
    print("[bbox-robust] === zero_shot ===")
    print("[bbox-robust] ============================================")
    sam = load_medsam_from_state_dict(base_sd, arch=arch, device=device)
    info_zs = setup_method(sam, "zero_shot")
    sam.eval()
    print(f"[bbox-robust] zero_shot ready (trainable={info_zs['trainable']:,})")
    all_rows.extend(eval_method_on_datasets(
        sam, "zero_shot", "zero_shot", 0, info_zs,
        cfg, args, device, image_size, args.perturb_levels, args.n_samples,
    ))
    del sam
    if device == "cuda":
        torch.cuda.empty_cache()

    # Trained checkpoints
    for ckpt_path in checkpoints:
        print("\n[bbox-robust] ============================================")
        print(f"[bbox-robust] === {ckpt_path.parent.name} ===")
        print("[bbox-robust] ============================================")
        sam = load_medsam_from_state_dict(base_sd, arch=arch, device=device)
        ckpt_info = read_checkpoint(ckpt_path)
        param_info = apply_method_and_weights(sam, ckpt_info, device)
        sam.eval()
        print(
            f"[bbox-robust] {ckpt_info['method']} ({ckpt_info['run_name']}): "
            f"trainable={param_info['trainable']:,}"
        )
        all_rows.extend(eval_method_on_datasets(
            sam, ckpt_info["run_name"], ckpt_info["method"],
            ckpt_info["seed"], param_info,
            cfg, args, device, image_size,
            args.perturb_levels, args.n_samples,
        ))
        del sam
        if device == "cuda":
            torch.cuda.empty_cache()

    # Write runs.csv; complete experiment, overwrite any previous run
    runs_path = OUT_DIR / "runs.csv"
    fieldnames = list(all_rows[0].keys())
    with open(runs_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n[bbox-robust] wrote {len(all_rows)} rows to {runs_path}")
    total_min = (time.time() - t_wall) / 60
    print(f"[bbox-robust] done in {total_min:.1f} min.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
