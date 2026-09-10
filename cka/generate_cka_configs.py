"""Generate 9 LoRA + CKA configs: 3 positions x 3 lambdas x seed 0."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"

POSITION_LAYERS = {
    "early": ["decoder_transformer"],
    "mid": ["decoder_transformer", "decoder_upscaling"],
    "late": ["decoder_upscaling", "decoder_iou_head", "decoder_mask_logits"],
}
# Per-layer weight in the CKA loss sum; higher where the paper's correlation is stronger.
POSITION_WEIGHTS = {
    "early": {"decoder_transformer": 1.0},
    "mid": {"decoder_transformer": 1.0, "decoder_upscaling": 0.8},
    "late": {
        "decoder_upscaling": 0.8,
        "decoder_iou_head": 1.5,
        "decoder_mask_logits": 1.0,
    },
}

# ACCV 2026 TrustFMI (spec section 4.2): encoder-hook ablation positions.
POSITION_LAYERS["enc"] = ["encoder_neck", "encoder_block_10", "encoder_block_11"]
POSITION_LAYERS["both"] = POSITION_LAYERS["late"] + POSITION_LAYERS["enc"]
POSITION_WEIGHTS["enc"] = {
    "encoder_neck": 1.0,
    "encoder_block_10": 1.0,
    "encoder_block_11": 1.0,
}
POSITION_WEIGHTS["both"] = {**POSITION_WEIGHTS["late"], **POSITION_WEIGHTS["enc"]}

# OOD-only probe (0 ISIC + 16 BUSI + 16 CBIS), lambda 10. The first two runs
# already exist on the HPC and are listed so their configs live in git; the
# remaining four are the new T1 array.
ACCV_RUNS = [
    {
        "name": "lora_cka_oodonly_late_l10_seed0",
        "position": "late",
        "seed": 0,
        "perturb": 0,
        "checkpoint_dir": "checkpoints/runs_cka_oodonly_late",
    },
    {
        "name": "lora_cka_oodonly_late_l10_pm20_seed0",
        "position": "late",
        "seed": 0,
        "perturb": 20,
        "checkpoint_dir": "checkpoints/runs_cka_oodonly_late_pm20",
    },
    {
        "name": "lora_cka_oodonly_late_l10_pm20_seed1",
        "position": "late",
        "seed": 1,
        "perturb": 20,
        "checkpoint_dir": "checkpoints/runs_accv_t1",
    },
    {
        "name": "lora_cka_oodonly_late_l10_pm20_seed2",
        "position": "late",
        "seed": 2,
        "perturb": 20,
        "checkpoint_dir": "checkpoints/runs_accv_t1",
    },
    {
        "name": "lora_cka_oodonly_enc_l10_pm20_seed0",
        "position": "enc",
        "seed": 0,
        "perturb": 20,
        "checkpoint_dir": "checkpoints/runs_accv_t1",
    },
    {
        "name": "lora_cka_oodonly_both_l10_pm20_seed0",
        "position": "both",
        "seed": 0,
        "perturb": 20,
        "checkpoint_dir": "checkpoints/runs_accv_t1",
    },
]

ACCV_TEMPLATE = """# LoRA + CKA (OOD-only probe), ACCV TrustFMI; position={position} lambda=10 seed={seed} pm={perturb}

name: {run_name}
method: lora
seed: {seed}

method_kwargs:
  rank: 8
  alpha: 16
  dropout: 0.0

model:
  arch: vit_b
  checkpoint: checkpoints/medsam_vit_b.pth
  image_size: 1024

data:
  root: data
  bbox_perturb_pixels: {perturb}

train:
  batch_size: 1
  num_workers: 8
  epochs: 6
  lr: 5.0e-4
  weight_decay: 0.0
  dice_weight: 0.5
  amp: true
  cooldown_seconds: 0

eval:
  batch_size: 1
  num_workers: 4

cka_regularization:
  enabled: true
  lambda: 10.0
  probe_seed: 42
  n_isic: 0
  n_busi: 16
  n_cbis: 16
  encoder_chunk: 8               # L40S 48 GB
  use_grad_checkpoint: true
  every_n_steps: 4
  hook_layers:
{hook_layers_yaml}
  weights:
{weights_yaml}

output:
  checkpoint_dir: {checkpoint_dir}
"""


def emit_accv_configs(out_dir: Path = CONFIGS_DIR) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for run in ACCV_RUNS:
        layers = POSITION_LAYERS[run["position"]]
        weights = POSITION_WEIGHTS[run["position"]]
        text = ACCV_TEMPLATE.format(
            position=run["position"],
            seed=run["seed"],
            perturb=run["perturb"],
            run_name=run["name"],
            hook_layers_yaml="\n".join(f"    - {layer}" for layer in layers),
            weights_yaml="\n".join(f"    {k}: {v}" for k, v in weights.items()),
            checkpoint_dir=run["checkpoint_dir"],
        )
        p = out_dir / f"{run['name']}.yaml"
        p.write_text(text)
        written.append(p)
    return written


# Lambda values to sweep; the string suffix is used in filenames and run names.
LAMBDAS = [
    ("01", 0.1),
    ("1", 1.0),
    ("10", 10.0),
]


CONFIG_TEMPLATE = """# LoRA + CKA auto-generated; position={position} lambda={lambda_val} seed=0 hook layers={layers} weights={weights}

name: {run_name}
method: lora
seed: 0

method_kwargs:
  rank: 8
  alpha: 16
  dropout: 0.0

model:
  arch: vit_b
  checkpoint: checkpoints/medsam_vit_b.pth
  image_size: 1024

data:
  root: data
  bbox_perturb_pixels: 0

train:
  batch_size: 1
  num_workers: 8              # bumped from 2 to keep GPU fed (PIL decode/resize is CPU-bound)
  epochs: 6
  lr: 5.0e-4
  weight_decay: 0.0
  dice_weight: 0.5
  amp: true
  cooldown_seconds: 0         # 0 (was 60); A10 datacenter needs no thermal cooldown

eval:
  batch_size: 1
  num_workers: 4

cka_regularization:
  enabled: true
  lambda: {lambda_val}
  probe_seed: 42
  n_isic: 12
  n_busi: 10
  n_cbis: 10
  encoder_chunk: 4               # probe encoder micro-batch; 4 fits 22 GB A10 with checkpointing, 8/16 OOM
  use_grad_checkpoint: true      # per-block gradient checkpointing on the probe encoder
  every_n_steps: 4               # compute CKA loss every N steps (lambda scaled by N), ~4x speedup
  hook_layers:
{hook_layers_yaml}
  weights:
{weights_yaml}

output:
  checkpoint_dir: checkpoints/runs_cka_{position}
"""


def emit_config(position: str, lambda_str: str, lambda_val: float) -> Path:
    layers = POSITION_LAYERS[position]
    weights = POSITION_WEIGHTS[position]
    run_name = f"lora_cka_{position}_l{lambda_str}_seed0"

    hook_layers_yaml = "\n".join(f"    - {layer}" for layer in layers)
    weights_yaml = "\n".join(f"    {k}: {v}" for k, v in weights.items())

    out = CONFIG_TEMPLATE.format(
        position=position,
        lambda_val=lambda_val,
        layers=layers,
        weights=weights,
        run_name=run_name,
        hook_layers_yaml=hook_layers_yaml,
        weights_yaml=weights_yaml,
    )
    out_path = CONFIGS_DIR / f"{run_name}.yaml"
    out_path.write_text(out)
    return out_path


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=CONFIGS_DIR)
    ap.add_argument(
        "--accv-only", action="store_true", help="Skip the original 9-config sweep"
    )
    args = ap.parse_args(argv)

    written = []
    if not args.accv_only:
        print("[gen-cka-configs] original sweep: 9 configs (3 positions x 3 lambdas)")
        for position in ("early", "mid", "late"):
            for lambda_str, lambda_val in LAMBDAS:
                written.append(emit_config(position, lambda_str, lambda_val))
    print("[gen-cka-configs] ACCV T1 set: 6 configs")
    written.extend(emit_accv_configs(args.out_dir))
    for p in written:
        print(f"  wrote {p.name}")
    print(f"[gen-cka-configs] done: {len(written)} configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
