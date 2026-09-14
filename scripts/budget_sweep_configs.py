"""Emit the nine adaptation-budget configs (3 methods x {50, 250, 1000} images, 6000 steps)."""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "configs" / "budget"

BUDGETS = [50, 250, 1000]
# lr and kwargs follow the existing per-method configs.
METHODS = {
    "lora": {"lr": "5.0e-4", "kwargs": {"rank": 8, "alpha": 16, "dropout": 0.0}},
    "decoder_only": {"lr": "1.0e-4", "kwargs": None},
    "lora_encoder_only": {
        "lr": "5.0e-4",
        "kwargs": {"rank": 28, "alpha": 56, "dropout": 0.0, "target_modules": "all"},
    },
}

TEMPLATE = """# Adaptation-budget run (ACCV TrustFMI spec 5.1): {method} on {budget} ISIC images,
# 6000 optimiser steps at batch 1, cosine over steps, val every 500 steps.

name: {run_name}
method: {method}
seed: {seed}
{kwargs_block}
model:
  arch: vit_b
  checkpoint: checkpoints/medsam_vit_b.pth
  image_size: 1024

data:
  root: data
  bbox_perturb_pixels: 0
  max_train_samples: {budget}
  subset_seed: 0

train:
  batch_size: 1
  num_workers: 8
  max_steps: 6000
  val_every_steps: 500
  lr: {lr}
  weight_decay: 0.0
  dice_weight: 0.5
  amp: true
  cooldown_seconds: 0

eval:
  batch_size: 1
  num_workers: 4

output:
  checkpoint_dir: {checkpoint_dir}
"""


def _kwargs_block(kwargs: dict | None) -> str:
    if not kwargs:
        return ""
    lines = ["", "method_kwargs:"]
    for k, v in kwargs.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines) + "\n"


def emit_budget_configs(out_dir: Path = OUT_DIR, seeds=(0,)) -> list[Path]:
    """Seed 0 goes to checkpoints/runs_accv_t2 (the original sweep); extra seeds to runs_accv_t6."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for seed in seeds:
        ckpt_dir = "checkpoints/runs_accv_t2" if seed == 0 else "checkpoints/runs_accv_t6"
        for method, spec in METHODS.items():
            for budget in BUDGETS:
                run_name = f"{method}_n{budget}_seed{seed}"
                text = TEMPLATE.format(
                    method=method,
                    budget=budget,
                    run_name=run_name,
                    seed=seed,
                    lr=spec["lr"],
                    kwargs_block=_kwargs_block(spec["kwargs"]),
                    checkpoint_dir=ckpt_dir,
                )
                p = out_dir / f"{run_name}.yaml"
                p.write_text(text)
                written.append(p)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args(argv)
    for p in emit_budget_configs(args.out_dir, tuple(args.seeds)):
        rel = p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p
        print(f"  wrote {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
