"""Generate seed-1 and seed-2 training configs from the seed-0 ones.

For each of the 15 base configs (5 methods x 3 trainings), produce a seed-1
and seed-2 variant, e.g. decoder_only_pm20.yaml -> decoder_only_seedN_pm20.yaml.

Each new config differs in three fields:
  - seed: 0 -> N
  - name: <method>_seed0[_suffix] -> <method>_seedN[_suffix]
  - output.checkpoint_dir: checkpoints/runs[_suffix] -> checkpoints/runs_seedN[_suffix]

Writes 15 yaml files per seed to configs/. Text-mode rewrite to keep comments.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"

# (source_config_name, perturb_suffix); suffix is "" for pm=0
BASE_CONFIGS = [
    # pm=0 (no suffix in filename or run_name)
    ("decoder_only.yaml",  ""),
    ("vpt_shallow.yaml",   ""),
    ("vpt_deep.yaml",      ""),
    ("lora.yaml",          ""),
    ("full_ft.yaml",       ""),
    # pm=20 fixed jitter
    ("decoder_only_pm20.yaml", "_pm20"),
    ("vpt_shallow_pm20.yaml",  "_pm20"),
    ("vpt_deep_pm20.yaml",     "_pm20"),
    ("lora_pm20.yaml",         "_pm20"),
    ("full_ft_pm20.yaml",      "_pm20"),
    # rand100 random jitter
    ("decoder_only_rand100.yaml", "_rand100"),
    ("vpt_shallow_rand100.yaml",  "_rand100"),
    ("vpt_deep_rand100.yaml",     "_rand100"),
    ("lora_rand100.yaml",         "_rand100"),
    ("full_ft_rand100.yaml",      "_rand100"),
]

EXTRA_SEEDS = [1, 2]


def transform_config_text(text: str, seed: int, suffix: str) -> str:
    """Rewrite seed/name/checkpoint_dir in a config's raw YAML text. Text-mode
    (not YAML round-trip) to keep comments and whitespace.
    """
    out = text

    # 1. seed: 0 -> seed: N (whole-line match for safety)
    out = re.sub(
        r"^seed:\s*0\b",
        f"seed: {seed}",
        out,
        count=1,
        flags=re.MULTILINE,
    )

    # 2. name: <something>_seed0<suffix> -> name: <something>_seedN<suffix>
    out = re.sub(
        r"^(name:\s*\w+)_seed0",
        rf"\1_seed{seed}",
        out,
        count=1,
        flags=re.MULTILINE,
    )

    # 3. output.checkpoint_dir: checkpoints/runs[_suffix] -> runs_seedN[_suffix].
    # Match only the checkpoint_dir line; the medsam_vit_b.pth path also has
    # "checkpoints/" in it.
    def repl_ckptdir(m):
        prefix = m.group(1)
        old_value = m.group(2)  # "runs", "runs_pm20", etc.
        new_value = old_value.replace("runs", f"runs_seed{seed}", 1)
        return f"{prefix}{new_value}"

    out = re.sub(
        r"(^\s*checkpoint_dir:\s*checkpoints/)(runs(?:_pm20|_rand100)?)\b",
        repl_ckptdir,
        out,
        count=1,
        flags=re.MULTILINE,
    )

    return out


def new_config_name(source_name: str, seed: int, suffix: str) -> str:
    """decoder_only_pm20.yaml + seed=1 -> decoder_only_seed1_pm20.yaml"""
    method = source_name.replace(".yaml", "")
    if suffix:
        # method was e.g. "decoder_only_pm20"; strip the suffix to get "decoder_only"
        method = method[: -len(suffix)]
    return f"{method}_seed{seed}{suffix}.yaml"


def main() -> int:
    print(f"[generate-seeds] generating configs for seeds {EXTRA_SEEDS}")
    print(f"[generate-seeds] base configs: {len(BASE_CONFIGS)}")
    n_written = 0
    n_skipped = 0
    for source, suffix in BASE_CONFIGS:
        source_path = CONFIGS_DIR / source
        if not source_path.exists():
            print(f"[generate-seeds] SKIP missing source: {source_path}")
            continue
        with open(source_path) as f:
            source_text = f.read()
        for seed in EXTRA_SEEDS:
            new_name = new_config_name(source, seed, suffix)
            out_path = CONFIGS_DIR / new_name
            new_text = transform_config_text(source_text, seed, suffix)
            with open(out_path, "w") as f:
                f.write(new_text)
            n_written += 1
            print(f"[generate-seeds] wrote {out_path.name}")
    print(f"\n[generate-seeds] done: {n_written} configs written, {n_skipped} skipped")
    print("\nNext steps:")
    print("  1. Commit + push the 30 new configs")
    print("  2. On JupyterLab: pull, then launch the 30 trainings (~32h)")
    print("  3. After training: run evals with --checkpoint-glob 'checkpoints/runs_seedN*/*/best.pth'")
    print("  4. Aggregate across seeds with scripts/aggregate_seeds.py (TBD)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
