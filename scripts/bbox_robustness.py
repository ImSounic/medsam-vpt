"""Bbox-robustness study: train at 0px, eval with expand-only outward bbox jitter to test if models learned segmentation or just memorized that the lesion fills the box."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# Make `from src...` work when run as a script
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.busi import BUSI
from src.data.cbis_ddsm import CBISDDSM
from src.data.isic import ISIC2018, isic_collate
from src.data.ph2 import PH2
from src.eval import predict_batch
from src.metrics import aggregate_metrics, dice_score, hd95, iou_score
from src.models.medsam import load_medsam
from src.models.methods import setup_method

# Paper experiments live under results/paper/ to keep them separate from the dissertation results.
OUT_DIR = _REPO_ROOT / "results" / "paper" / "bbox_robustness"
CSV_PATH = OUT_DIR / "bbox_robustness.csv"
FIG_PATH = OUT_DIR / "bbox_robustness.png"

METHODS = {
    "zero_shot":    {"checkpoint": None,                                              "label": "Zero-shot",       "color": "#7f7f7f"},
    "decoder_only": {"checkpoint": "checkpoints/runs/decoder_only_seed0/best.pth",   "label": "Decoder-only",    "color": "#1f77b4"},
    "vpt_shallow":  {"checkpoint": "checkpoints/runs/vpt_shallow_seed0/best.pth",    "label": "VPT-shallow",     "color": "#ff7f0e"},
    "vpt_deep":     {"checkpoint": "checkpoints/runs/vpt_deep_seed0/best.pth",       "label": "VPT-deep",        "color": "#d62728"},
    "lora":         {"checkpoint": "checkpoints/runs/lora_seed0/best.pth",           "label": "LoRA",            "color": "#2ca02c"},
    "full_ft":      {"checkpoint": "checkpoints/runs/full_ft_seed0/best.pth",        "label": "Full FT",         "color": "#9467bd"},
}


def build_dataset(name: str, image_size: int):
    """Build a dataset with tight (unperturbed) bboxes; jitter is applied manually in evaluate()."""
    if name == "isic":
        return ISIC2018(
            root=_REPO_ROOT / "data",
            split="test",
            image_size=image_size,
            bbox_perturb_pixels=0,
        )
    if name == "ph2":
        return PH2(root=_REPO_ROOT / "data/ph2", image_size=image_size, bbox_perturb_pixels=0)
    if name == "busi":
        return BUSI(
            root=_REPO_ROOT / "data/busi",
            image_size=image_size,
            bbox_perturb_pixels=0,
            include_normal=False,
        )
    if name == "cbis_ddsm":
        return CBISDDSM(
            root=_REPO_ROOT / "data/cbis-ddsm",
            split="test",
            image_size=image_size,
            bbox_perturb_pixels=0,
            abnormality_type="all",
        )
    raise ValueError(f"Unknown dataset: {name}")


def expand_bbox(tight_bbox: np.ndarray, perturb_px: int, image_size: int, image_id: str, trial: int = 0) -> np.ndarray:
    """Expand bbox outward (never shrink) by a random amount in [0, perturb_px], deterministic per (image_id, perturb_px, trial)."""
    if perturb_px == 0:
        return tight_bbox.copy().astype(np.float32)
    x1, y1, x2, y2 = tight_bbox.astype(np.float32)
    seed = abs(hash((image_id, perturb_px, trial))) % (2**31)
    rng = np.random.RandomState(seed=seed)
    pad_left, pad_top, pad_right, pad_bottom = rng.uniform(0, perturb_px, size=4)
    new_x1 = max(0.0, x1 - pad_left)
    new_y1 = max(0.0, y1 - pad_top)
    new_x2 = min(image_size - 1.0, x2 + pad_right)
    new_y2 = min(image_size - 1.0, y2 + pad_bottom)
    return np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float32)


def load_model(method: str, device: str):
    """Load MedSAM with the given fine-tuning method's trained weights."""
    sam = load_medsam(
        _REPO_ROOT / "checkpoints/medsam_vit_b.pth",
        arch="vit_b",
        device=device,
    )
    ckpt_path = METHODS[method]["checkpoint"]
    if ckpt_path is None:
        # zero-shot: no method setup or weights
        sam.eval()
        return sam

    ckpt = torch.load(_REPO_ROOT / ckpt_path, map_location="cpu", weights_only=False)
    method_kwargs = ckpt.get("config", {}).get("method_kwargs", {}) or {}
    setup_method(sam, method, **method_kwargs)
    state = {k: v.to(device) for k, v in ckpt["trainable_state"].items()}
    sam.load_state_dict(state, strict=False)
    sam.eval()
    return sam


@torch.no_grad()
def _decode_with_cached_features(sam, image_embedding, bbox, image_size):
    """Decode from a precomputed image embedding (reused across jitter/trial combos since the encoder is ~80% of inference cost)."""
    sparse_embed, dense_embed = sam.prompt_encoder(
        points=None,
        boxes=bbox.unsqueeze(0),
        masks=None,
    )
    low_res, _ = sam.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embed,
        dense_prompt_embeddings=dense_embed,
        multimask_output=False,
    )
    mask = torch.nn.functional.interpolate(
        low_res, size=(image_size, image_size), mode="bilinear", align_corners=False
    )
    return (mask > 0).to(torch.uint8).squeeze(0).squeeze(0)


def evaluate_all_jitters(sam, dataset, jitter_levels, image_size: int, device: str, n_trials: int = 1) -> dict:
    """Eval all jitter levels in one dataset pass, caching encoder features per image; returns {jitter_px: aggregate metrics}."""
    loader = DataLoader(
        dataset, batch_size=1, num_workers=0, collate_fn=isic_collate, shuffle=False
    )

    # per-image metrics per jitter level, averaged across trials
    per_jitter_per_image: dict = {j: [] for j in jitter_levels}

    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            images = batch["image"].to(device)
            tight_bboxes = batch["bbox"].numpy()  # (B=1, 4)
            masks_gt = batch["mask"].cpu().numpy()
            image_ids = batch["image_id"]
            B = tight_bboxes.shape[0]
            assert B == 1, "Encoder caching assumes batch_size=1"

            # run encoder once per image
            image_embedding = sam.image_encoder(images)

            for j in range(B):
                gt_j = masks_gt[j]
                image_id = str(image_ids[j])
                tight_bbox = tight_bboxes[j]

                # loop over (jitter, trial) reusing the cached encoder features
                for jitter in jitter_levels:
                    n_t = 1 if jitter == 0 else n_trials
                    trial_dice, trial_iou, trial_hd95 = [], [], []
                    for t in range(n_t):
                        jittered = expand_bbox(tight_bbox, jitter, image_size, image_id, trial=t)
                        bbox_t = torch.from_numpy(jittered).to(device)
                        pred = _decode_with_cached_features(
                            sam, image_embedding, bbox_t, image_size,
                        ).cpu().numpy()
                        trial_dice.append(dice_score(pred, gt_j))
                        trial_iou.append(iou_score(pred, gt_j))
                        trial_hd95.append(hd95(pred, gt_j))

                    per_jitter_per_image[jitter].append({
                        "dice": float(np.mean(trial_dice)),
                        "iou": float(np.mean(trial_iou)),
                        "hd95": float(np.mean(trial_hd95)),
                    })

    # Aggregate per jitter level
    return {j: aggregate_metrics(per_jitter_per_image[j]) for j in jitter_levels}


CSV_FIELDS = [
    "method", "dataset", "jitter_px", "n_images", "n_trials",
    "dice_mean", "dice_std", "iou_mean", "hd95_mean", "wall_clock_s",
]


def _coerce_row(row: dict) -> dict:
    """Convert CSV string fields back to numeric types for plotting."""
    return {
        "method": row["method"],
        "dataset": row["dataset"],
        "jitter_px": int(row["jitter_px"]),
        "n_images": int(row["n_images"]),
        "n_trials": int(row["n_trials"]),
        "dice_mean": float(row["dice_mean"]),
        "dice_std": float(row["dice_std"]),
        "iou_mean": float(row["iou_mean"]),
        "hd95_mean": float(row["hd95_mean"]),
        "wall_clock_s": float(row["wall_clock_s"]),
    }


def save_csv(all_rows: list[dict]) -> None:
    """Overwrite the whole CSV with all rows; called after each pair so partial progress survives Ctrl+C."""
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(all_rows)


def render_figure(all_rows: list[dict], methods_in_run: list[str]) -> None:
    """Render the Dice-vs-jitter facet plot from whatever rows are present."""
    if not all_rows:
        return
    datasets_present = sorted({r["dataset"] for r in all_rows})
    n_datasets = len(datasets_present)

    fig, axes = plt.subplots(
        1, n_datasets,
        figsize=(5.5 * n_datasets, 5.5),
        squeeze=False,
    )
    axes = axes[0]

    for ax, ds in zip(axes, datasets_present):
        for method in methods_in_run:
            if method not in METHODS:
                continue
            method_rows = [r for r in all_rows if r["method"] == method and r["dataset"] == ds]
            if not method_rows:
                continue
            method_rows.sort(key=lambda r: r["jitter_px"])
            xs = [r["jitter_px"] for r in method_rows]
            ys = [r["dice_mean"] for r in method_rows]
            ax.plot(
                xs, ys,
                marker="o", markersize=9, linewidth=2.5,
                color=METHODS[method]["color"],
                label=METHODS[method]["label"],
                markeredgecolor="black", markeredgewidth=0.5,
            )

        ax.set_xlabel("Bbox jitter (pixels), wider = more realistic", fontsize=11)
        ax.set_ylabel("Dice", fontsize=11)
        ax.set_title(f"{ds}: Dice vs bbox jitter", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.legend(loc="lower left", framealpha=0.95, fontsize=10)

    fig.suptitle(
        "Robustness to bbox imprecision: fine-tuned models trained at 0px jitter\n"
        "(steeper drop = model learned to over-trust the bbox during training)",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods", nargs="+",
        default=["zero_shot", "decoder_only", "lora"],
        help="Which methods to evaluate (subset of: zero_shot decoder_only vpt_shallow vpt_deep lora full_ft)",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["isic"],
        help="Which datasets to evaluate (subset of: isic ph2 busi cbis_ddsm)",
    )
    parser.add_argument(
        "--jitter", nargs="+", type=int,
        default=[0, 5, 10, 20, 40, 80],
        help="Bbox jitter levels in pixels",
    )
    parser.add_argument(
        "--subset", type=int, default=200,
        help="Eval on first N images per dataset (0 = full dataset)",
    )
    parser.add_argument(
        "--n-trials", type=int, default=3,
        help="Number of random expansion trials per image. Per-image metrics are "
             "averaged across trials. Use 1 for single-sample (faster), 3-5 for "
             "the paper (more stable estimates), or higher for tighter confidence.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Delete existing CSV and start fresh. Default behavior is to resume "
             "from any (method, dataset) pairs already in the CSV.",
    )
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"[bbox_robust] device={args.device} subset={args.subset or 'full'} jitter={args.jitter} n_trials={args.n_trials}")
    print(f"[bbox_robust] methods={args.methods}")
    print(f"[bbox_robust] datasets={args.datasets}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # resume support: load existing CSV unless --overwrite
    all_rows: list[dict] = []
    done_pairs: set = set()
    if args.overwrite and CSV_PATH.exists():
        CSV_PATH.unlink()
        print(f"[bbox_robust] --overwrite given; deleted existing CSV")
    elif CSV_PATH.exists():
        with open(CSV_PATH, "r", newline="") as f:
            for r in csv.DictReader(f):
                all_rows.append(_coerce_row(r))
                done_pairs.add((r["method"], r["dataset"]))
        if done_pairs:
            print(f"[bbox_robust] Resuming: {len(done_pairs)} (method, dataset) pairs already in CSV; will skip those")

    try:
        for method in args.methods:
            if method not in METHODS:
                print(f"[bbox_robust] WARNING: unknown method {method}, skipping")
                continue

            sam = None  # lazy-load only when we have work to do
            for ds_name in args.datasets:
                if (method, ds_name) in done_pairs:
                    print(f"[bbox_robust] Skip {METHODS[method]['label']:14s} | {ds_name:9s} (already done)")
                    continue

                if sam is None:
                    print(f"\n[bbox_robust] === Loading {method} ===")
                    sam = load_model(method, args.device)

                # Build dataset once (tight bboxes); we'll expand them per-jitter inside evaluate()
                dataset = build_dataset(ds_name, args.image_size)
                if args.subset > 0 and hasattr(dataset, "items"):
                    dataset.items = dataset.items[:args.subset]
                n = len(dataset)

                # single pass for all jitter levels; encoder features cached per image
                t0 = time.time()
                metrics_by_jitter = evaluate_all_jitters(
                    sam, dataset, args.jitter, args.image_size, args.device, n_trials=args.n_trials,
                )
                elapsed_total = time.time() - t0
                per_jitter_elapsed = elapsed_total / max(len(args.jitter), 1)

                new_rows = []
                for jitter in args.jitter:
                    metrics = metrics_by_jitter[jitter]
                    trials_here = 1 if jitter == 0 else args.n_trials
                    row = {
                        "method": method,
                        "dataset": ds_name,
                        "jitter_px": jitter,
                        "n_images": n,
                        "n_trials": trials_here,
                        "dice_mean": round(metrics["dice_mean"], 4),
                        "dice_std": round(metrics["dice_std"], 4),
                        "iou_mean": round(metrics["iou_mean"], 4),
                        "hd95_mean": round(metrics["hd95_mean"], 2),
                        "wall_clock_s": round(per_jitter_elapsed, 1),
                    }
                    new_rows.append(row)
                    print(
                        f"[bbox_robust] {METHODS[method]['label']:14s} | "
                        f"{ds_name:9s} | jitter={jitter:3d}px | n={n:4d} | k={trials_here} | "
                        f"dice={row['dice_mean']:.4f} ± {row['dice_std']:.4f} | "
                        f"hd95={row['hd95_mean']:6.1f}px"
                    )
                print(f"[bbox_robust] {METHODS[method]['label']:14s} | {ds_name:9s} | total: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")

                # IMMEDIATELY persist to disk so Ctrl+C doesn't lose this pair
                all_rows.extend(new_rows)
                done_pairs.add((method, ds_name))
                save_csv(all_rows)
                print(f"[bbox_robust] >>> saved {len(all_rows)} rows to {CSV_PATH.name} ({len(done_pairs)} pairs complete)")

            if sam is not None:
                del sam
                if args.device == "cuda":
                    torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n[bbox_robust] Interrupted by user. Partial results saved to CSV.")
        print(f"[bbox_robust] Completed pairs so far: {len(done_pairs)}")

    finally:
        # Always regenerate the figure from whatever rows we have
        if all_rows:
            render_figure(all_rows, args.methods)
            print(f"[bbox_robust] CSV: {CSV_PATH}")
            print(f"[bbox_robust] Figure: {FIG_PATH}")
            print(f"[bbox_robust] Tip: re-run the same command to resume the remaining (method, dataset) pairs.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
