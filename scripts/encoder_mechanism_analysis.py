"""Encoder weight/feature-shift analysis over 15 checkpoints (5 methods x 3 trainings); tests whether PEFT trades modality transfer for prompt robustness via feature drift growing pm=0 < pm=20 < rand100."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.busi import BUSI  # noqa: E402
from src.data.cbis_ddsm import CBISDDSM  # noqa: E402
from src.data.isic import ISIC2018, isic_collate  # noqa: E402
from src.data.ph2 import PH2  # noqa: E402
from src.device_utils import device_name, get_device  # noqa: E402
from src.models.medsam import load_medsam_from_state_dict  # noqa: E402
from src.models.methods import setup_method  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "mechanism"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Number of probe images per dataset
PROBE_N = 32

METHODS = ["decoder_only", "vpt_shallow", "vpt_deep", "lora", "full_ft"]
TRAININGS = [
    # (label, checkpoint_dir_relative_to_repo)
    ("pm=0",     "checkpoints/runs"),
    ("pm=20",    "checkpoints/runs_pm20"),
    ("rand100",  "checkpoints/runs_rand100"),
]
DATASETS = {
    "isic2018_test": lambda: ISIC2018(root=REPO_ROOT / "data", split="test", image_size=1024),
    "ph2":           lambda: PH2(root=REPO_ROOT / "data" / "ph2", image_size=1024),
    "busi":          lambda: BUSI(root=REPO_ROOT / "data" / "busi", image_size=1024),
    "cbis_ddsm":     lambda: CBISDDSM(root=REPO_ROOT / "data" / "cbis-ddsm", split="test", image_size=1024),
}

METHOD_LABELS = {
    "decoder_only": "Decoder-only",
    "vpt_shallow":  "VPT-shallow",
    "vpt_deep":     "VPT-deep",
    "lora":         "LoRA",
    "full_ft":      "Full FT",
}
DATASET_LABELS = {
    "isic2018_test": "ISIC (ID)",
    "ph2":           "PH2 (near-OOD)",
    "busi":          "BUSI (far-OOD US)",
    "cbis_ddsm":     "CBIS-DDSM (far-OOD X-ray)",
}
TRAINING_PALETTE = {
    "pm=0":    "#1f77b4",
    "pm=20":   "#ff7f0e",
    "rand100": "#2ca02c",
}


def checkpoint_path(method: str, training_label: str, base_dir: str) -> Path:
    """Resolve the path for a (method, training) checkpoint."""
    base = REPO_ROOT / base_dir
    if training_label == "pm=0":
        return base / f"{method}_seed0" / "best.pth"
    suffix = training_label.replace("=", "").replace("(", "").replace(")", "").replace(",", "_").replace(" ", "")
    # Hardcoded paths for the trained suffixes
    if training_label == "pm=20":
        return base / f"{method}_seed0_pm20" / "best.pth"
    if training_label == "rand100":
        return base / f"{method}_seed0_rand100" / "best.pth"
    raise ValueError(f"unhandled training label: {training_label}")


def build_sam_with_checkpoint(base_sd: dict, ckpt_path: Path, device: str) -> torch.nn.Module:
    sam = load_medsam_from_state_dict(base_sd, device=device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    method_kwargs = ckpt.get("config", {}).get("method_kwargs", {}) or {}
    setup_method(sam, ckpt["method"], **method_kwargs)
    state = {k: v.to(device) for k, v in ckpt["trainable_state"].items()}
    sam.load_state_dict(state, strict=False)
    sam.eval()
    return sam, ckpt


def measure_weight_delta(ckpt: dict, base_sd: dict) -> float:
    """Mean relative L2 encoder weight change; 0 for frozen-encoder methods (LoRA/VPT)."""
    deltas = []
    for k, v in ckpt["trainable_state"].items():
        if "image_encoder" not in k:
            continue
        if k not in base_sd:
            continue
        base_w = base_sd[k].float()
        ft_w = v.float()
        delta = (ft_w - base_w).norm() / (base_w.norm() + 1e-9)
        deltas.append(delta.item())
    return float(np.mean(deltas)) if deltas else 0.0


@torch.no_grad()
def compute_features(sam, images: torch.Tensor, batch_size: int = 2) -> torch.Tensor:
    """Run sam.image_encoder over a stack of images, return flattened features."""
    feats = []
    for i in range(0, images.shape[0], batch_size):
        batch = images[i : i + batch_size]
        out = sam.image_encoder(batch).flatten(1)
        feats.append(out)
    return torch.cat(feats, dim=0)


def measure_feature_shift(ft_feats: torch.Tensor, base_feats: torch.Tensor) -> tuple:
    """Mean and std per-sample relative L2 distance across N samples."""
    per_image = (ft_feats - base_feats).norm(dim=1) / base_feats.norm(dim=1)
    return float(per_image.mean()), float(per_image.std())


def build_probe_set(device: str, n: int = PROBE_N) -> dict:
    """Returns {dataset_name: image tensor on device}."""
    out = {}
    for ds_name, builder in DATASETS.items():
        try:
            ds = builder()
        except Exception as e:
            print(f"[mech] could not build {ds_name}: {e}")
            continue
        ds.items = ds.items[: n]
        loader = DataLoader(ds, batch_size=2, num_workers=0, collate_fn=isic_collate, shuffle=False)
        images = torch.cat([b["image"] for b in loader], dim=0).to(device)
        print(f"[mech] probe set {ds_name}: {images.shape[0]} images")
        out[ds_name] = images
    return out


def main() -> int:
    device = get_device()
    print(f"[mech] device={device} ({device_name(device)})")

    # Load base MedSAM weights once
    base_ckpt_path = REPO_ROOT / "checkpoints" / "medsam_vit_b.pth"
    base_sd = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    if "model" in base_sd and isinstance(base_sd["model"], dict):
        base_sd = base_sd["model"]

    # Build probe set (one fixed set of images per dataset)
    probe = build_probe_set(device)

    # Cache base-model features per dataset (saves a forward per checkpoint)
    print("\n[mech] computing base MedSAM features on probe sets...")
    base_sam = load_medsam_from_state_dict(base_sd, device=device)
    setup_method(base_sam, "zero_shot")
    base_sam.eval()
    base_feats = {}
    for ds_name, images in probe.items():
        base_feats[ds_name] = compute_features(base_sam, images)
        print(f"[mech]   {ds_name}: base feats {base_feats[ds_name].shape}, "
              f"mean ||f|| = {base_feats[ds_name].norm(dim=1).mean().item():.2f}")
    del base_sam
    if device == "cuda":
        torch.cuda.empty_cache()

    # Iterate over the 15 checkpoints
    rows = []
    for method in METHODS:
        for training_label, ckpt_dir in TRAININGS:
            ckpt_path = checkpoint_path(method, training_label, ckpt_dir)
            if not ckpt_path.exists():
                print(f"[mech] SKIP {method} x {training_label}: {ckpt_path} not found")
                continue

            print(f"\n[mech] === {method} x {training_label} ===")
            sam, ckpt = build_sam_with_checkpoint(base_sd, ckpt_path, device)

            # Weight-space delta (encoder only)
            wdelta = measure_weight_delta(ckpt, base_sd)
            print(f"[mech]   weight delta (encoder, relative L2): {wdelta:.4f}")

            # Feature-space shift on each dataset
            for ds_name, images in probe.items():
                ft = compute_features(sam, images)
                fmean, fstd = measure_feature_shift(ft, base_feats[ds_name])
                print(f"[mech]   {ds_name}: feature shift = {fmean:.4f} +/- {fstd:.4f}")
                rows.append({
                    "method": method,
                    "training": training_label,
                    "dataset": ds_name,
                    "weight_delta": f"{wdelta:.6f}",
                    "feature_shift_mean": f"{fmean:.6f}",
                    "feature_shift_std":  f"{fstd:.6f}",
                    "n_probe_images": images.shape[0],
                })

            del sam
            if device == "cuda":
                torch.cuda.empty_cache()

    # Save CSV
    csv_path = OUT_DIR / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[mech] wrote {csv_path} ({len(rows)} rows)")

    # plots
    df = pd.DataFrame(rows)
    df["weight_delta"] = df["weight_delta"].astype(float)
    df["feature_shift_mean"] = df["feature_shift_mean"].astype(float)
    df["feature_shift_std"] = df["feature_shift_std"].astype(float)

    plot_weight_delta_bars(df, OUT_DIR / "weight_delta_bars.png")
    plot_feature_shift_heatmap(df, OUT_DIR / "feature_shift_heatmap.png")
    plot_feature_shift_bars(df, OUT_DIR / "feature_shift_bars.png")
    plot_shift_vs_dice_scatter(df, OUT_DIR / "shift_vs_dice_scatter.png")
    plot_method_conditional_trajectories(df, OUT_DIR / "shift_vs_dice_trajectories.png")
    return 0


def plot_weight_delta_bars(df: pd.DataFrame, out_path: Path) -> None:
    """Per (method, training) mean encoder weight delta; only full_ft is non-zero."""
    # One row per (method, training); weight delta is the same across datasets
    sub = df.drop_duplicates(subset=["method", "training"])[["method", "training", "weight_delta"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_w = 0.25
    x_pos = np.arange(len(METHODS))

    for ti, training in enumerate(("pm=0", "pm=20", "rand100")):
        vals = []
        for m in METHODS:
            r = sub[(sub["method"] == m) & (sub["training"] == training)]
            vals.append(float(r["weight_delta"].iloc[0]) if not r.empty else np.nan)
        offsets = (ti - 1) * bar_w
        ax.bar(x_pos + offsets, vals, width=bar_w, color=TRAINING_PALETTE[training],
                edgecolor="black", linewidth=0.5, label=f"{training} trained")
        for x, v in zip(x_pos + offsets, vals):
            if not np.isnan(v) and v > 0.0005:
                ax.text(x, v + 0.0005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS])
    ax.set_ylabel("Mean relative encoder weight change\n||W_finetuned - W_base|| / ||W_base||")
    ax.set_title("Encoder weight delta by method x training\n"
                  "(only full_ft mutates encoder weights; frozen-encoder methods sit at 0)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[mech] wrote {out_path}")


def plot_feature_shift_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """One heatmap: rows = method x training (15 cells), cols = dataset."""
    fig, ax = plt.subplots(figsize=(8, 9))

    # Build matrix: 15 rows (method x training), 4 cols (datasets)
    row_labels = []
    row_keys = []
    for m in METHODS:
        for t in ("pm=0", "pm=20", "rand100"):
            row_labels.append(f"{METHOD_LABELS[m]} - {t}")
            row_keys.append((m, t))

    cols = list(DATASETS.keys())
    matrix = np.full((len(row_keys), len(cols)), np.nan)
    for ri, (m, t) in enumerate(row_keys):
        for ci, ds in enumerate(cols):
            r = df[(df["method"] == m) & (df["training"] == t) & (df["dataset"] == ds)]
            if not r.empty:
                matrix[ri, ci] = float(r["feature_shift_mean"].iloc[0])

    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([DATASET_LABELS[d] for d in cols], rotation=15, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                         color="white" if v < matrix[~np.isnan(matrix)].mean() else "black",
                         fontsize=8.5)
    # Group separators (every 3 rows is one method block)
    for i in range(3, len(row_labels), 3):
        ax.axhline(i - 0.5, color="white", linewidth=1.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Mean relative feature shift\n||f_ft(x) - f_base(x)|| / ||f_base(x)||")
    ax.set_title("Encoder feature-space drift by (method x training) x dataset")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[mech] wrote {out_path}")


def plot_feature_shift_bars(df: pd.DataFrame, out_path: Path) -> None:
    """4 dataset panels, each with grouped bars (5 methods x 3 trainings)."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=False)
    axes_flat = axes.flatten()
    bar_w = 0.25
    x_pos = np.arange(len(METHODS))

    for ax, ds in zip(axes_flat, DATASETS.keys()):
        sub = df[df["dataset"] == ds]
        for ti, training in enumerate(("pm=0", "pm=20", "rand100")):
            vals = []
            for m in METHODS:
                r = sub[(sub["method"] == m) & (sub["training"] == training)]
                vals.append(float(r["feature_shift_mean"].iloc[0]) if not r.empty else np.nan)
            offsets = (ti - 1) * bar_w
            ax.bar(x_pos + offsets, vals, width=bar_w, color=TRAINING_PALETTE[training],
                    edgecolor="black", linewidth=0.5, label=f"{training} trained")
            for x, v in zip(x_pos + offsets, vals):
                if not np.isnan(v):
                    ax.text(x, v + max(vals) * 0.01, f"{v:.2f}",
                             ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=15, ha="right")
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=11)
        ax.set_ylabel("Relative feature shift")
        ax.grid(axis="y", alpha=0.3)

    handles = [plt.Rectangle((0, 0), 1, 1, color=TRAINING_PALETTE[t], ec="black", label=f"{t} trained")
                for t in ("pm=0", "pm=20", "rand100")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                bbox_to_anchor=(0.5, -0.02), fontsize=11)
    fig.suptitle(
        "Encoder feature-space drift per dataset (higher = output diverges more from base MedSAM)",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[mech] wrote {out_path}")


def plot_shift_vs_dice_scatter(df: pd.DataFrame, out_path: Path) -> None:
    """Feature shift vs delta-Dice (vs pm=0 baseline at tight bbox) per (method, training, dataset)."""
    # Read Dice values from the tight-bbox eval CSVs
    pm0_dice = _load_tight_dice(REPO_ROOT / "results" / "runs.csv")
    pm20_dice = _load_tight_dice(REPO_ROOT / "results" / "runs_pm20.csv")
    rand_dice = _load_tight_dice(REPO_ROOT / "results" / "runs_rand100.csv")
    dice_by_training = {"pm=0": pm0_dice, "pm=20": pm20_dice, "rand100": rand_dice}

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    for ax, ds in zip(axes, DATASETS.keys()):
        xs, ys, labels, colors = [], [], [], []
        for m in METHODS:
            for t in ("pm=0", "pm=20", "rand100"):
                row = df[(df["method"] == m) & (df["training"] == t) & (df["dataset"] == ds)]
                if row.empty:
                    continue
                shift = float(row["feature_shift_mean"].iloc[0])
                ft_d = dice_by_training[t].get((m, ds))
                base_d = pm0_dice.get((m, ds))
                if ft_d is None or base_d is None:
                    continue
                delta_dice = ft_d - base_d
                xs.append(shift); ys.append(delta_dice)
                labels.append(f"{METHOD_LABELS[m]}\n{t}")
                colors.append(TRAINING_PALETTE[t])

        ax.scatter(xs, ys, c=colors, s=120, edgecolors="black", linewidth=1.0, zorder=3)
        # Annotate with method initials
        for x, y, lab in zip(xs, ys, labels):
            init = lab.split("\n")[0][:3]
            ax.annotate(init, (x, y), xytext=(5, 0), textcoords="offset points",
                         fontsize=7, alpha=0.7)

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Feature shift  (||f_ft - f_base|| / ||f_base||)")
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("delta Dice (training vs pm=0 baseline) at tight bbox")

        # Pearson correlation
        if len(xs) >= 3:
            r = np.corrcoef(xs, ys)[0, 1]
            ax.text(0.02, 0.02, f"r = {r:+.2f}", transform=ax.transAxes,
                     fontsize=10, va="bottom", ha="left",
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    handles = [plt.Line2D([], [], marker="o", linestyle="", color=TRAINING_PALETTE[t],
                            markersize=10, markeredgecolor="black", label=f"{t} trained")
                for t in ("pm=0", "pm=20", "rand100")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                bbox_to_anchor=(0.5, -0.02), fontsize=11)
    fig.suptitle(
        "Does encoder drift predict Dice regression? Feature shift vs delta-Dice (tight bbox), per dataset",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[mech] wrote {out_path}")


def plot_method_conditional_trajectories(df: pd.DataFrame, out_path: Path) -> None:
    """Per-method (feature_shift, delta-Dice) trajectories; pm=0 anchors at delta-Dice=0, per-method view avoids global-fit anchoring toward zero."""
    pm0_dice = _load_tight_dice(REPO_ROOT / "results" / "runs.csv")
    pm20_dice = _load_tight_dice(REPO_ROOT / "results" / "runs_pm20.csv")
    rand_dice = _load_tight_dice(REPO_ROOT / "results" / "runs_rand100.csv")
    dice_by_training = {"pm=0": pm0_dice, "pm=20": pm20_dice, "rand100": rand_dice}

    dataset_colors = {
        "isic2018_test": "#1f77b4",
        "ph2":           "#ff7f0e",
        "busi":          "#9467bd",
        "cbis_ddsm":     "#d62728",
    }
    training_markers = {"pm=0": "o", "pm=20": "s", "rand100": "^"}

    fig, axes = plt.subplots(1, len(METHODS), figsize=(22, 5.5), sharey=True)

    for ax, m in zip(axes, METHODS):
        # Non-baseline points for per-method Pearson r; exclude pm=0 which sits at delta-Dice=0 and drags r toward zero
        nonbaseline_shifts: list = []
        nonbaseline_deltas: list = []

        for ds, ds_color in dataset_colors.items():
            points = []
            for t in ("pm=0", "pm=20", "rand100"):
                row = df[(df["method"] == m) & (df["training"] == t) & (df["dataset"] == ds)]
                if row.empty:
                    continue
                shift = float(row["feature_shift_mean"].iloc[0])
                ft_d = dice_by_training[t].get((m, ds))
                base_d = pm0_dice.get((m, ds))
                if ft_d is None or base_d is None:
                    continue
                d_dice = ft_d - base_d
                points.append((t, shift, d_dice))
                if t != "pm=0":
                    nonbaseline_shifts.append(shift)
                    nonbaseline_deltas.append(d_dice)
            if len(points) < 2:
                continue

            xs = [p[1] for p in points]
            ys = [p[2] for p in points]
            # Trajectory line pm=0 -> pm=20 -> rand100
            ax.plot(xs, ys, color=ds_color, linewidth=1.8, alpha=0.55, zorder=2)
            # Markers per training
            for t, x, y in points:
                ax.scatter(x, y, color=ds_color, marker=training_markers[t],
                            s=160, edgecolors="black", linewidth=1.0, zorder=5)

        ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
        ax.set_title(METHOD_LABELS[m], fontsize=12, fontweight="bold")
        ax.set_xlabel("Feature shift  ||f_ft - f_base|| / ||f_base||")
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("delta Dice (training vs pm=0 baseline), tight bbox")

        # Per-method Pearson r over non-baseline points (8 max: 4 datasets x 2 non-pm=0 trainings)
        if len(nonbaseline_shifts) >= 3:
            xs_arr = np.array(nonbaseline_shifts)
            ys_arr = np.array(nonbaseline_deltas)
            if xs_arr.std() > 1e-9:
                r = float(np.corrcoef(xs_arr, ys_arr)[0, 1])
                ax.text(0.02, 0.02, f"r = {r:+.2f}  (n={len(nonbaseline_shifts)})",
                         transform=ax.transAxes, fontsize=9,
                         va="bottom", ha="left",
                         bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
            else:
                # Feature shift constant across all non-baseline trainings (e.g. decoder_only)
                ax.text(0.02, 0.02, f"r undefined  (shift constant)\nn={len(nonbaseline_shifts)}",
                         transform=ax.transAxes, fontsize=8,
                         va="bottom", ha="left",
                         bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    dataset_handles = [
        plt.Line2D([], [], marker="o", linestyle="-", color=dataset_colors[d],
                    markersize=10, markeredgecolor="black",
                    label=DATASET_LABELS[d])
        for d in dataset_colors
    ]
    training_handles = [
        plt.Line2D([], [], marker=training_markers[t], linestyle="", color="gray",
                    markersize=10, markeredgecolor="black", label=t)
        for t in ("pm=0", "pm=20", "rand100")
    ]
    fig.legend(handles=dataset_handles + training_handles, loc="lower center",
                ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=10)
    fig.suptitle(
        "Per-method trajectories in (feature shift, delta-Dice) space "
        "as training jitter widens",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[mech] wrote {out_path}")


def _load_tight_dice(path: Path) -> dict:
    """Returns {(method, dataset): dice_mean} from a results/runs*.csv."""
    out = {}
    if not path.exists():
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            out[(r["method"], r["dataset"])] = float(r["dice_mean"])
    return out


if __name__ == "__main__":
    raise SystemExit(main())
