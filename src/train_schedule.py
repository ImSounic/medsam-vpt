"""Schedule helpers for src.train: epoch budgets vs fixed optimiser-step budgets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass(frozen=True)
class Budget:
    mode: str  # "epochs" or "steps"
    n_segments: int  # epochs, or validation segments in steps mode
    max_steps: int | None
    val_every_steps: int | None


def resolve_budget(train_cfg: dict) -> Budget:
    """train.max_steps (with train.val_every_steps) overrides train.epochs."""
    max_steps = train_cfg.get("max_steps")
    if max_steps is None:
        return Budget("epochs", int(train_cfg["epochs"]), None, None)
    max_steps = int(max_steps)
    if max_steps <= 0:
        raise ValueError(f"train.max_steps must be positive, got {max_steps}")
    val_every = int(train_cfg.get("val_every_steps", max_steps))
    if val_every <= 0:
        raise ValueError(f"train.val_every_steps must be positive, got {val_every}")
    return Budget("steps", math.ceil(max_steps / val_every), max_steps, val_every)


def plan_segments(max_steps: int, val_every_steps: int) -> list[int]:
    """Optimiser steps per validation segment; the last segment takes the remainder."""
    segs = [val_every_steps] * (max_steps // val_every_steps)
    rem = max_steps % val_every_steps
    if rem:
        segs.append(rem)
    return segs


def cycle_loader(loader: Iterable) -> Iterator:
    """Iterate a DataLoader forever; each pass re-shuffles because iter() is called again."""
    while True:
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded:
            raise RuntimeError("cycle_loader: the loader yielded nothing")


def apply_quick_overrides(cfg: dict) -> None:
    """--quick: smallest run that exercises every code path (8 GB laptop GPU)."""
    train = cfg["train"]
    if train.get("max_steps") is not None:
        train["max_steps"] = 6
        train["val_every_steps"] = 3
    else:
        train["epochs"] = 1
    # Encoder backprop at 1024 px does not fit 8 GB without checkpointing, and
    # a smoke run must never touch a real run directory.
    train["grad_checkpoint"] = True
    cfg.setdefault("output", {})["checkpoint_dir"] = "checkpoints/quick"
    cka = cfg.get("cka_regularization") or {}
    if cka.get("enabled"):
        cka["n_isic"] = 0
        cka["n_busi"] = 2
        cka["n_cbis"] = 2
        cka["encoder_chunk"] = 1
        cka["every_n_steps"] = 1
