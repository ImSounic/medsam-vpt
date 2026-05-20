"""Run src.eval for every checkpoint that matches a glob.

Examples:
    python scripts/eval_all_checkpoints.py --config configs/busi_eval.yaml
    python scripts/eval_all_checkpoints.py --config configs/busi_eval.yaml --quick
    python scripts/eval_all_checkpoints.py --config configs/busi_eval.yaml \
        --checkpoint-glob 'checkpoints/runs/*/latest.pth'
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument(
        "--checkpoint-glob",
        default="checkpoints/runs/*/best.pth"
    )
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pattern = Path(args.checkpoint_glob)
    if pattern.is_absolute():
        checkpoints = sorted(pattern.parent.glob(pattern.name))
    else:
        checkpoints = sorted(REPO_ROOT.glob(args.checkpoint_glob))

    if not checkpoints:
        return 1

    print(f"[eval-all] evaluating {len(checkpoints)} checkpoints")
    for ckpt in checkpoints:
        rel = ckpt.relative_to(REPO_ROOT) if ckpt.is_relative_to(REPO_ROOT) else ckpt
        print(f"\n[eval-all] === {rel} ===")
        cmd = [
            sys.executable,
            "-m",
            "src.eval",
            "--config",
            str(args.config),
            "--checkpoint",
            str(ckpt),
        ]
        if args.device:
            cmd.extend(["--device", args.device])
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
