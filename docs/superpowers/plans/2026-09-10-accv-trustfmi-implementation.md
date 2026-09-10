# ACCV TrustFMI code changes and compute plan: implementation plan

> Work through the tasks in order; each task is test-first and ends with a commit. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement spec section 8 (eight code changes) and section 9 (compute plan) of `docs/superpowers/specs/2026-09-10-accv-trustfmi-paper-design.md` so that the T1 CKA array, the T2 budget array and the T3 detector dumps can be submitted to the UT HPC, and run the day-one `iou_pred` check locally.

**Architecture:** All work is on branch `accv-trustfmi` in `/home/imsounic/Projects/medsam-vpt`. The training loop gains a step-budget mode by treating both epochs and step segments as "segments" over a cycling loader; eval gains `iou_pred` and an optional base-model decoder-drift pass using the restored `cka.hooks` package; config generators emit the T1 and T2 YAMLs; SLURM arrays chain with `afterany` dependencies; new analysis scripts are pure numpy/pandas/matplotlib. Every change is unit-tested under `tests/` with pytest and smoke-tested on the laptop GPU with `--quick`.

**Tech Stack:** Python 3.12 in the existing virtualenv `~/mlenv` (torch 2.11 + CUDA 12.8, segment_anything, monai, pandas, scipy, matplotlib). pytest is added to that env. SLURM on `hpc-head2.ewi.utwente.nl`, account `s3702111`, QOS 2 concurrent GPU jobs.

---

## Facts established while planning (read before starting)

- The working directory `/home/imsounic/Projects/accv-project` is empty; the repo is `/home/imsounic/Projects/medsam-vpt`, already on branch `accv-trustfmi`. Run every command from that directory. `PY=~/mlenv/bin/python`.
- Commit `ec000e0` is the commit that **deleted** `cka/` and `bbox_robustness/eval_bbox_robust.py`. The last commit that contains them is its parent, `569464a`. Restore from `569464a`, not `ec000e0`.
- `configs/lora_encoder_only.yaml` already exists (rank 28, alpha 56, `target_modules: all`, `train_mask_decoder: false`, `method: lora`). Task 3 adds the named method and switches the config to it.
- The local seed-0 checkpoints are `checkpoints/runs/{decoder_only,full_ft,lora,vpt_deep,vpt_shallow}_seed0/best.pth`. No encoder-only or CKA checkpoint is local.
- Local data: ISIC train 2595 / val 101 / test 1000, PH2, BUSI, CBIS-DDSM (csv + jpeg), DMID.
- The HPC head node is unreachable from this laptop right now (SSH times out without eduVPN, and `~/.ssh` has no key). Task 9 produces the SLURM files and a submit script; actual submission needs the VPN. Do not block on it.
- `scripts/eval_all_methods.py` skips every checkpoint whose directory name contains `decoder_only` except the first one. The T2 array has three `decoder_only_n*` checkpoints, so T2 eval must use `scripts/eval_all_checkpoints.py` (one `src.eval` subprocess per checkpoint).
- Non-reentrant gradient checkpointing (used on the probe encoder) re-runs the forward during backward, so forward hooks on encoder blocks fire twice. The CKA loss is computed from `stacked()` before backward, and `clear()` runs afterwards, so this is harmless. The task forward also fires the hooks; Task 2 adds a `clear()` after the task backward.

## File structure

Create:

| File | Responsibility |
|---|---|
| `tests/conftest.py` | `sys.path` setup, `make_sam()` factory (ViT-B, random init, CPU), fake ISIC tree fixture |
| `tests/test_cka_hooks.py` | encoder_neck hook resolution and capture |
| `tests/test_methods.py` | `lora_encoder_only` freezing and dispatch |
| `tests/test_isic_subset.py` | deterministic prefix subset |
| `tests/test_train_schedule.py` | budget resolution, segment planning, cycling loader, quick overrides |
| `tests/test_eval_detectors.py` | `iou_pred` return, drift math, per-image CSV columns |
| `tests/test_config_generators.py` | T1 and T2 config emission |
| `tests/test_failure_detection.py` | AUROC, AUPRC, ECE, table building |
| `tests/test_plots.py` | budget and hook-ablation plot scripts on synthetic CSVs |
| `tests/test_slurm.py` | array ranges match config lists; bash syntax |
| `src/train_schedule.py` | pure helpers: `Budget`, `resolve_budget`, `plan_segments`, `cycle_loader`, `apply_quick_overrides` |
| `src/drift.py` | `decoder_drift(base_act, cur_act)` |
| `scripts/budget_sweep_configs.py` | emits the nine T2 configs into `configs/budget/` |
| `scripts/failure_detection.py` | AUROC/AUPRC/ECE table and figure from per-image CSVs |
| `scripts/plot_budget.py` | two-panel budget figure |
| `scripts/plot_hook_ablation.py` | hook ablation bars |
| `configs/accv_eval.yaml` | four-dataset eval config used by T1/T2/T3 evals and the day-one check |
| `configs/budget/*.yaml` | generated, nine files |
| `configs/lora_cka_oodonly_*.yaml` | generated, six files (four T1 runs plus the two existing HPC reference runs) |
| `cka/slurm/accv_t1.sbatch`, `accv_t1_eval.sbatch`, `accv_t2.sbatch`, `accv_t2_eval.sbatch`, `accv_t3.sbatch`, `submit_accv.sh`, `sync_to_hpc.sh` | compute plan |

Modify:

| File | Change |
|---|---|
| `cka/hooks.py` (restored) | `encoder_neck` position |
| `cka/generate_cka_configs.py` (restored) | `emit_accv_config`, `ACCV_RUNS`, `--out-dir` |
| `src/models/lora.py` | `apply_lora_encoder_only` |
| `src/models/methods.py` | dispatch `lora_encoder_only`; `encoder_in_grad_path` |
| `configs/lora_encoder_only.yaml` | `method: lora_encoder_only` |
| `src/data/isic.py` | `max_train_samples`, `subset_seed` |
| `src/train.py` | `--quick`, step budget, per-step scheduler, `step` in log and checkpoint, hook clear |
| `src/eval.py` | `iou_pred`, `--drift`, `--limit`, `--bbox-perturb`, `--results-csv`, `--per-image-dir` |
| `scripts/eval_all_checkpoints.py` | `--eval-args` passthrough |
| `pyproject.toml` | pytest config |

---

### Task 0: Test scaffold and environment

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/__init__.py`, `tests/conftest.py`

- [ ] **Step 1: Install pytest into mlenv and confirm CUDA**

Run:
```bash
cd /home/imsounic/Projects/medsam-vpt && uv pip install --python ~/mlenv/bin/python pytest && ~/mlenv/bin/python -c "import torch; print(torch.cuda.is_available())"
```
Expected: `True`.

- [ ] **Step 2: Add pytest config to pyproject.toml**

Append to `pyproject.toml`:
```toml

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
filterwarnings = ["ignore::DeprecationWarning"]
```

- [ ] **Step 3: Write tests/conftest.py**

```python
"""Shared fixtures: repo root on sys.path, a random-init SAM ViT-B on CPU, a fake ISIC tree."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def make_sam():
    """Factory: fresh SAM ViT-B with random weights on CPU (about 2 s each)."""
    from segment_anything import sam_model_registry

    def _make():
        sam = sam_model_registry["vit_b"](checkpoint=None)
        sam.eval()
        return sam

    return _make


@pytest.fixture
def fake_isic_root(tmp_path: Path) -> Path:
    """20 train pairs, 4 val pairs, 8x8 images, ISIC naming conventions."""
    for split, n in (("train", 20), ("val", 4)):
        img_dir = tmp_path / f"{split}_images"
        msk_dir = tmp_path / f"{split}_masks"
        img_dir.mkdir()
        msk_dir.mkdir()
        for i in range(n):
            stem = f"ISIC_{split}_{i:04d}"
            Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8)).save(
                img_dir / f"{stem}.jpg"
            )
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[2:6, 2:6] = 255
            Image.fromarray(mask).save(msk_dir / f"{stem}_segmentation.png")
    return tmp_path
```

Also create an empty `tests/__init__.py`.

- [ ] **Step 4: Run pytest to verify the scaffold collects nothing and exits 5**

Run: `~/mlenv/bin/python -m pytest`
Expected: `no tests ran` (exit code 5).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/__init__.py tests/conftest.py
git commit -m "test: add pytest scaffold for accv-trustfmi work"
```

---

### Task 1: Restore the CKA package and the bbox robustness script

**Files:**
- Restore: `cka/` (whole tree), `bbox_robustness/eval_bbox_robust.py` from `569464a`

- [ ] **Step 1: Restore from the parent of ec000e0**

```bash
git checkout 569464a -- cka/ bbox_robustness/eval_bbox_robust.py
git status --short | head
```
Expected: `A  cka/...` lines (35 files) and `A  bbox_robustness/eval_bbox_robust.py`.

- [ ] **Step 2: Verify imports still resolve**

Run:
```bash
~/mlenv/bin/python -c "import cka.hooks, cka.probe, src.cka, src.train; print('ok')"
~/mlenv/bin/python -m src.train --help | head -3
~/mlenv/bin/python bbox_robustness/eval_bbox_robust.py --help | head -3
```
Expected: `ok`, then argparse usage text for both scripts.

- [ ] **Step 3: Commit**

```bash
git add cka bbox_robustness/eval_bbox_robust.py
git commit -m "restore: cka package and bbox robustness eval from 569464a"
```

---

### Task 2: Encoder hook position `encoder_neck`

**Files:**
- Modify: `cka/hooks.py` (`_resolve_module`, error message)
- Modify: `src/train.py` (clear hooks after task backward)
- Test: `tests/test_cka_hooks.py`

- [ ] **Step 1: Write the failing tests**

```python
"""encoder_neck hook position resolves on raw and VPT-wrapped encoders and captures the neck output."""

import torch

from cka.hooks import _resolve_module, register_hooks


def test_encoder_neck_resolves_on_raw_encoder(make_sam):
    sam = make_sam()
    assert _resolve_module(sam, "encoder_neck") is sam.image_encoder.neck


def test_encoder_neck_resolves_on_vpt_wrapped_encoder(make_sam):
    from src.models.vpt import apply_vpt

    sam = make_sam()
    apply_vpt(sam, n_prompts=2, mode="shallow")
    assert _resolve_module(sam, "encoder_neck") is sam.image_encoder.base.neck


def test_encoder_neck_hook_captures_neck_output(make_sam):
    sam = make_sam()
    with register_hooks(sam, ["encoder_neck"], detach=True, accumulate=True) as hh:
        x = torch.randn(2, 64, 64, 768)
        with torch.no_grad():
            sam.image_encoder.neck(x.permute(0, 3, 1, 2))
            sam.image_encoder.neck(x.permute(0, 3, 1, 2))
        acts = hh.stacked()
    assert acts["encoder_neck"].shape == (4, 256, 64, 64)


def test_unknown_layer_lists_encoder_neck(make_sam):
    sam = make_sam()
    try:
        _resolve_module(sam, "bogus")
    except ValueError as e:
        assert "encoder_neck" in str(e)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run to verify failure**

Run: `~/mlenv/bin/python -m pytest tests/test_cka_hooks.py -v`
Expected: 3 failures with `ValueError: Unknown layer name: 'encoder_neck'` and one failure on the message assertion.

- [ ] **Step 3: Implement in cka/hooks.py**

In `_resolve_module`, before the `encoder_block_` branch, add:
```python
    if name == "encoder_neck":
        enc = sam.image_encoder
        base = getattr(enc, "base", None)
        return base.neck if base is not None and hasattr(base, "neck") else enc.neck
```
Replace the final `raise ValueError(...)` message with:
```python
    raise ValueError(
        f"Unknown layer name: {name!r}. "
        f"Supported: encoder_block_<i>, encoder_neck, decoder_transformer, "
        f"decoder_upscaling, decoder_iou_head, decoder_mask_logits."
    )
```

- [ ] **Step 4: Clear accumulated task-forward activations in src/train.py**

In `train_one_epoch`, directly after `del logits, task_loss`, add:
```python
        if cka_ctx is not None:
            # Hooks also fire during the task forward; drop those activations now.
            cka_ctx["hook_handle"].clear()
```

- [ ] **Step 5: Run tests**

Run: `~/mlenv/bin/python -m pytest tests/test_cka_hooks.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add cka/hooks.py src/train.py tests/test_cka_hooks.py
git commit -m "feat(cka): add encoder_neck hook position; clear task-forward activations"
```

---

### Task 3: `lora_encoder_only` method

**Files:**
- Modify: `src/models/lora.py`, `src/models/methods.py`, `configs/lora_encoder_only.yaml`
- Test: `tests/test_methods.py`

- [ ] **Step 1: Write the failing tests**

```python
"""lora_encoder_only: LoRA on the encoder, mask decoder frozen, encoder in the grad path."""

import pytest

from src.models.methods import encoder_in_grad_path, setup_method


def _decoder_trainable(sam) -> int:
    return sum(p.numel() for p in sam.mask_decoder.parameters() if p.requires_grad)


def test_encoder_only_freezes_decoder(make_sam):
    sam = make_sam()
    info = setup_method(sam, "lora_encoder_only", rank=28, alpha=56, target_modules="all")
    assert _decoder_trainable(sam) == 0
    assert info["trainable"] > 0


def test_encoder_only_ignores_train_mask_decoder_kwarg(make_sam):
    sam = make_sam()
    setup_method(sam, "lora_encoder_only", rank=8, train_mask_decoder=True)
    assert _decoder_trainable(sam) == 0


def test_encoder_only_matches_lora_without_decoder(make_sam):
    a = make_sam()
    b = make_sam()
    ia = setup_method(a, "lora_encoder_only", rank=8, alpha=16)
    ib = setup_method(b, "lora", rank=8, alpha=16, train_mask_decoder=False)
    assert ia["trainable"] == ib["trainable"]


def test_encoder_only_in_grad_path():
    assert encoder_in_grad_path("lora_encoder_only") is True


def test_unknown_method_message_lists_encoder_only(make_sam):
    sam = make_sam()
    with pytest.raises(ValueError, match="lora_encoder_only"):
        setup_method(sam, "bogus")
```

- [ ] **Step 2: Run to verify failure**

Run: `~/mlenv/bin/python -m pytest tests/test_methods.py -v`
Expected: 5 failures (`Unknown method: 'lora_encoder_only'`, and the grad-path assertion).

- [ ] **Step 3: Add apply_lora_encoder_only to src/models/lora.py**

Append at the end of the file:
```python
def apply_lora_encoder_only(sam: Sam, **kwargs) -> None:
    """LoRA on the image encoder only; the mask decoder stays frozen."""
    kwargs.pop("train_mask_decoder", None)
    apply_lora(sam, train_mask_decoder=False, **kwargs)
    for p in sam.mask_decoder.parameters():
        p.requires_grad = False
```

- [ ] **Step 4: Dispatch in src/models/methods.py**

Replace the `elif method == "lora":` block and the `else` with:
```python
    elif method == "lora":
        # Lazy import to keep startup light for non-LoRA runs.
        from .lora import apply_lora

        apply_lora(sam, **kwargs)
    elif method == "lora_encoder_only":
        from .lora import apply_lora_encoder_only

        apply_lora_encoder_only(sam, **kwargs)
    else:
        raise ValueError(
            f"Unknown method: {method!r}. "
            "Supported: zero_shot, decoder_only, vpt_shallow, vpt_deep, full_ft, "
            "lora, lora_encoder_only."
        )
```
And in `encoder_in_grad_path`:
```python
    return method in {"full_ft", "vpt_shallow", "vpt_deep", "lora", "lora_encoder_only"}
```

- [ ] **Step 5: Point the config at the named method**

In `configs/lora_encoder_only.yaml` change `method: lora` to `method: lora_encoder_only`. Leave `method_kwargs` as they are (the `train_mask_decoder: false` line is now redundant but harmless).

- [ ] **Step 6: Run tests**

Run: `~/mlenv/bin/python -m pytest tests/test_methods.py -v`
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add src/models/lora.py src/models/methods.py configs/lora_encoder_only.yaml tests/test_methods.py
git commit -m "feat: add lora_encoder_only method (encoder LoRA, decoder frozen)"
```

---

### Task 4: ISIC deterministic prefix subset

**Files:**
- Modify: `src/data/isic.py` (`ISIC2018.__init__`)
- Modify: `src/train.py` (pass the two config keys)
- Test: `tests/test_isic_subset.py`

- [ ] **Step 1: Write the failing tests**

```python
"""max_train_samples selects a deterministic subset; smaller budgets are prefixes of larger ones."""

from src.data.isic import ISIC2018


def _ids(ds):
    return [stem for _, _, stem in ds.items]


def test_no_subset_by_default(fake_isic_root):
    assert len(ISIC2018(fake_isic_root, split="train", image_size=8)) == 20


def test_subset_size(fake_isic_root):
    ds = ISIC2018(fake_isic_root, split="train", image_size=8, max_train_samples=5)
    assert len(ds) == 5


def test_subset_is_prefix_of_larger_subset(fake_isic_root):
    small = ISIC2018(fake_isic_root, split="train", image_size=8, max_train_samples=5)
    large = ISIC2018(fake_isic_root, split="train", image_size=8, max_train_samples=12)
    assert _ids(small) == _ids(large)[:5]


def test_subset_is_deterministic_and_seed_dependent(fake_isic_root):
    a = ISIC2018(fake_isic_root, split="train", image_size=8, max_train_samples=10, subset_seed=0)
    b = ISIC2018(fake_isic_root, split="train", image_size=8, max_train_samples=10, subset_seed=0)
    c = ISIC2018(fake_isic_root, split="train", image_size=8, max_train_samples=10, subset_seed=1)
    assert _ids(a) == _ids(b)
    assert _ids(a) != _ids(c)


def test_subset_larger_than_dataset_keeps_all(fake_isic_root):
    ds = ISIC2018(fake_isic_root, split="train", image_size=8, max_train_samples=999)
    assert len(ds) == 20
```

- [ ] **Step 2: Run to verify failure**

Run: `~/mlenv/bin/python -m pytest tests/test_isic_subset.py -v`
Expected: 4 failures with `TypeError: ... unexpected keyword argument 'max_train_samples'`.

- [ ] **Step 3: Implement in src/data/isic.py**

Change the constructor signature to:
```python
    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test"] = "test",
        image_size: int = 1024,
        bbox_perturb_pixels: int = 0,
        random_perturb: bool = False,
        max_train_samples: int | None = None,
        subset_seed: int = 0,
    ) -> None:
```
After the `if missing > 0:` warning block at the end of `__init__`, add:
```python
        # Deterministic subset: one fixed permutation per seed, so a smaller
        # budget is always a prefix of a larger one.
        if max_train_samples is not None and max_train_samples < len(self.items):
            perm = np.random.default_rng(subset_seed).permutation(len(self.items))
            keep = sorted(int(i) for i in perm[:max_train_samples])
            self.items = [self.items[i] for i in perm[:max_train_samples]]
            print(
                f"[ISIC2018:{split}] subset: {len(self.items)} of {len(perm)} "
                f"(subset_seed={subset_seed})"
            )
```
(Remove the unused `keep` line if the linter complains; the order must follow `perm`, not sorted order, for the prefix property.)

- [ ] **Step 4: Pass the config keys in src/train.py**

In `main()`, the `train_ds = ISIC2018(...)` call becomes:
```python
    train_ds = ISIC2018(
        root=REPO_ROOT / cfg["data"]["root"],
        split="train",
        image_size=image_size,
        bbox_perturb_pixels=cfg["data"].get("bbox_perturb_pixels", 0),
        random_perturb=bool(cfg["data"].get("random_perturb", False)),
        max_train_samples=cfg["data"].get("max_train_samples"),
        subset_seed=int(cfg["data"].get("subset_seed", 0)),
    )
```

- [ ] **Step 5: Run tests**

Run: `~/mlenv/bin/python -m pytest tests/test_isic_subset.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/data/isic.py src/train.py tests/test_isic_subset.py
git commit -m "feat(data): deterministic prefix subset via max_train_samples/subset_seed"
```

---

### Task 5: Step budget in the trainer (`train.max_steps`, `train.val_every_steps`, `--quick`)

**Files:**
- Create: `src/train_schedule.py`
- Modify: `src/train.py` (imports, `parse_args`, `train_one_epoch`, `save_checkpoint`, `main`)
- Test: `tests/test_train_schedule.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Pure schedule helpers used by src.train: budgets, segments, cycling loader, quick overrides."""

import itertools

import pytest

from src.train_schedule import (
    Budget,
    apply_quick_overrides,
    cycle_loader,
    plan_segments,
    resolve_budget,
)


def test_epoch_mode_when_no_max_steps():
    b = resolve_budget({"epochs": 6})
    assert b == Budget(mode="epochs", n_segments=6, max_steps=None, val_every_steps=None)


def test_steps_mode_segments():
    b = resolve_budget({"epochs": 6, "max_steps": 6000, "val_every_steps": 500})
    assert b.mode == "steps"
    assert b.n_segments == 12
    assert b.max_steps == 6000
    assert b.val_every_steps == 500


def test_steps_mode_defaults_val_every_to_max_steps():
    b = resolve_budget({"max_steps": 10})
    assert b.val_every_steps == 10 and b.n_segments == 1


def test_steps_mode_rejects_nonpositive():
    with pytest.raises(ValueError):
        resolve_budget({"max_steps": 0})
    with pytest.raises(ValueError):
        resolve_budget({"max_steps": 10, "val_every_steps": 0})


def test_plan_segments_remainder():
    assert plan_segments(6000, 500) == [500] * 12
    assert plan_segments(7, 3) == [3, 3, 1]


def test_cycle_loader_reiterates():
    loader = [1, 2, 3]
    got = list(itertools.islice(cycle_loader(loader), 7))
    assert got == [1, 2, 3, 1, 2, 3, 1]


def test_cycle_loader_empty_raises():
    with pytest.raises(RuntimeError):
        next(cycle_loader([]))


def test_quick_overrides_epoch_mode():
    cfg = {"train": {"epochs": 6}, "cka_regularization": {"enabled": True, "n_isic": 0,
           "n_busi": 16, "n_cbis": 16, "encoder_chunk": 8, "every_n_steps": 4}}
    apply_quick_overrides(cfg)
    assert cfg["train"]["epochs"] == 1
    assert cfg["cka_regularization"]["n_busi"] == 2
    assert cfg["cka_regularization"]["n_cbis"] == 2
    assert cfg["cka_regularization"]["encoder_chunk"] == 1
    assert cfg["cka_regularization"]["every_n_steps"] == 1


def test_quick_overrides_steps_mode():
    cfg = {"train": {"max_steps": 6000, "val_every_steps": 500}}
    apply_quick_overrides(cfg)
    assert cfg["train"]["max_steps"] == 6
    assert cfg["train"]["val_every_steps"] == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `~/mlenv/bin/python -m pytest tests/test_train_schedule.py -v`
Expected: `ModuleNotFoundError: No module named 'src.train_schedule'`.

- [ ] **Step 3: Create src/train_schedule.py**

```python
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
    cka = cfg.get("cka_regularization") or {}
    if cka.get("enabled"):
        cka["n_isic"] = 0
        cka["n_busi"] = 2
        cka["n_cbis"] = 2
        cka["encoder_chunk"] = 1
        cka["every_n_steps"] = 1
```

- [ ] **Step 4: Run tests**

Run: `~/mlenv/bin/python -m pytest tests/test_train_schedule.py -v`
Expected: 9 passed.

- [ ] **Step 5: Wire the trainer (src/train.py)**

(a) Imports: add `from itertools import islice` and
```python
from src.train_schedule import (
    apply_quick_overrides,
    cycle_loader,
    plan_segments,
    resolve_budget,
)
```

(b) `parse_args`: add
```python
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: 8 train / 4 val images, 1 epoch or 6 steps, tiny CKA probe.",
    )
```

(c) `train_one_epoch`: change the signature to
```python
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
) -> dict:
```
Replace `pbar = tqdm(loader, desc="train", leave=False)` with
```python
    batches = islice(loader, max_steps) if max_steps is not None else loader
    total = max_steps if max_steps is not None else len(loader)
    pbar = tqdm(batches, desc="train", leave=False, total=total)
```
Directly after the optimizer step block (`scaler.step(optimizer); scaler.update()` / `optimizer.step()`), add
```python
        if step_scheduler is not None:
            step_scheduler()
```

(d) `save_checkpoint`: add keyword `step: int = 0` after `best_val` and store `"step": step,` in `payload` next to `"epoch": epoch`.

(e) `main()`, right after `cfg = load_config(args.config)`:
```python
    if args.quick:
        apply_quick_overrides(cfg)
        print("[train] --quick: tiny data, budget and probe")
```
After both datasets are built and before the loaders:
```python
    if args.quick:
        train_ds.items = train_ds.items[:8]
        val_ds.items = val_ds.items[:4]
```

(f) Replace the optimizer/scheduler block
```python
    epochs = int(cfg["train"]["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
```
with
```python
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
```
and change the `print(f"[train] amp={amp} epochs={epochs} ...")` line to
```python
    print(
        f"[train] amp={amp} mode={budget.mode} segments={n_segments} "
        f"max_steps={budget.max_steps} batch={cfg['train']['batch_size']}"
    )
```

(g) Log header: insert `"step",` after `"epoch",`.

(h) Resume block: replace from `best_val = 0.0` to the end of the `if args.resume ...` block with
```python
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
```

(i) Main loop: replace `for epoch in range(start_epoch, epochs + 1):` and its body up to and including the two `save_checkpoint` calls with
```python
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
                sam, run_dir / "best.pth", seg, val_dice, cfg,
                best_val=best_val, step=global_step,
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
```
Epoch-mode behaviour is unchanged: one full loader pass per segment, scheduler stepped once per epoch.

- [ ] **Step 6: Run the whole test suite, then a quick epoch-mode smoke on the GPU**

Run: `~/mlenv/bin/python -m pytest`
Expected: all passed.

Run:
```bash
cd /home/imsounic/Projects/medsam-vpt && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ~/mlenv/bin/python -m src.train --config configs/lora_encoder_only.yaml --quick 2>&1 | tail -8
```
Expected: `[train] ep   1/1 step=8 ...` then `[train] done in ...` and a `best.pth` under `checkpoints/runs_perfect_bboxes/lora_encoder_only_r28_all_seed0/`. Delete that smoke checkpoint directory afterwards so it is not mistaken for a real run:
```bash
rm -rf checkpoints/runs_perfect_bboxes/lora_encoder_only_r28_all_seed0
```

- [ ] **Step 7: Commit**

```bash
git add src/train_schedule.py src/train.py tests/test_train_schedule.py
git commit -m "feat(train): fixed optimiser-step budget with periodic validation; --quick smoke mode"
```

---

### Task 6: Eval writes `iou_pred`, optional `--drift`, and path/limit overrides

**Files:**
- Create: `src/drift.py`
- Modify: `src/eval.py`
- Modify: `scripts/eval_all_checkpoints.py` (`--eval-args`)
- Create: `configs/accv_eval.yaml`
- Test: `tests/test_eval_detectors.py`

- [ ] **Step 1: Write the failing tests**

```python
"""iou_pred is returned per image; drift is 1 - CKA on the upscaled mask embedding; CSV has the new columns."""

import csv

import torch

from src.drift import decoder_drift
from src.eval import predict_from_embeddings, predict_from_embeddings_with_iou, write_per_image_csv


def test_predict_with_iou_shapes(make_sam):
    sam = make_sam()
    emb = torch.randn(2, 256, 64, 64)
    boxes = torch.tensor([[100.0, 100.0, 500.0, 500.0], [10.0, 10.0, 900.0, 900.0]])
    masks, ious = predict_from_embeddings_with_iou(sam, emb, boxes, 1024, 1024)
    assert masks.shape == (2, 1024, 1024) and masks.dtype == torch.uint8
    assert ious.shape == (2,) and torch.isfinite(ious).all()


def test_predict_from_embeddings_still_returns_masks_only(make_sam):
    sam = make_sam()
    emb = torch.randn(1, 256, 64, 64)
    boxes = torch.tensor([[100.0, 100.0, 500.0, 500.0]])
    out = predict_from_embeddings(sam, emb, boxes, 256, 256)
    assert out.shape == (1, 256, 256)


def test_drift_zero_for_identical_and_scaled():
    a = torch.randn(32, 256, 256)
    assert abs(decoder_drift(a, a)) < 1e-5
    assert abs(decoder_drift(a, 3.0 * a)) < 1e-5


def test_drift_near_one_for_independent():
    a = torch.randn(32, 256, 256)
    b = torch.randn(32, 256, 256)
    assert decoder_drift(a, b) > 0.9


def test_drift_accepts_leading_batch_dim():
    a = torch.randn(1, 32, 256, 256)
    assert abs(decoder_drift(a, a)) < 1e-5


def test_per_image_csv_columns(tmp_path):
    rows = [
        {"image_id": "x", "dice": 0.9, "iou": 0.8, "hd95": 3.0, "iou_pred": 0.85, "drift": 0.1},
        {"image_id": "y", "dice": 0.2, "iou": 0.1, "hd95": 40.0, "iou_pred": 0.9, "drift": 0.5},
    ]
    p = write_per_image_csv(tmp_path / "a" / "b.csv", rows)
    with open(p) as f:
        header = next(csv.reader(f))
    assert header == ["image_id", "dice", "iou", "hd95", "iou_pred", "drift"]
```

- [ ] **Step 2: Run to verify failure**

Run: `~/mlenv/bin/python -m pytest tests/test_eval_detectors.py -v`
Expected: `ImportError` for `src.drift` / `predict_from_embeddings_with_iou`.

- [ ] **Step 3: Create src/drift.py**

```python
"""Per-image decoder drift: 1 - linear CKA between base and adapted upscaled mask embeddings."""

from __future__ import annotations

import torch

from src.cka import linear_cka


def _positions_by_channels(act: torch.Tensor) -> torch.Tensor:
    """(1, C, H, W) or (C, H, W) -> (H*W, C) float32 on the same device."""
    if act.dim() == 4:
        if act.shape[0] != 1:
            raise ValueError(f"expected a single image, got batch {act.shape[0]}")
        act = act[0]
    if act.dim() != 3:
        raise ValueError(f"expected (C, H, W), got {tuple(act.shape)}")
    c = act.shape[0]
    return act.reshape(c, -1).transpose(0, 1).float()


@torch.no_grad()
def decoder_drift(base_act: torch.Tensor, cur_act: torch.Tensor) -> float:
    """Spatial positions are samples, channels are features (256*256 x 32 for SAM)."""
    x = _positions_by_channels(base_act)
    y = _positions_by_channels(cur_act)
    return float(1.0 - linear_cka(x, y).item())
```

- [ ] **Step 4: Modify src/eval.py**

(a) Imports: add `from src.drift import decoder_drift` and `from src.models.methods import encoder_in_grad_path, setup_method` (replacing the existing `setup_method` import).

(b) `parse_args`: add after `--device`
```python
    p.add_argument("--limit", type=int, default=None, help="First N images per dataset")
    p.add_argument(
        "--drift",
        action="store_true",
        help="Load base MedSAM alongside and write per-image decoder drift "
        "(1 - CKA of the upscaled mask embedding).",
    )
    p.add_argument("--bbox-perturb", type=int, default=None, help="Override eval.bbox_perturb_pixels")
    p.add_argument("--results-csv", type=Path, default=None, help="Override output.results_csv")
    p.add_argument("--per-image-dir", type=Path, default=None, help="Directory for per-image CSVs")
```

(c) Replace `predict_from_embeddings` with the pair:
```python
@torch.no_grad()
def predict_from_embeddings_with_iou(
    sam,
    image_embeddings: torch.Tensor,
    bboxes: torch.Tensor,
    H: int,
    W: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prompt encoder + mask decoder per image; returns (B, H, W) uint8 masks and (B,) iou_pred."""
    masks_out, ious = [], []
    for i in range(image_embeddings.shape[0]):
        sparse_embed, dense_embed = sam.prompt_encoder(
            points=None,
            boxes=bboxes[i : i + 1],
            masks=None,
        )
        low_res, iou_pred = sam.mask_decoder(
            image_embeddings=image_embeddings[i : i + 1],
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embed,
            dense_prompt_embeddings=dense_embed,
            multimask_output=False,
        )
        mask = torch.nn.functional.interpolate(
            low_res, size=(H, W), mode="bilinear", align_corners=False
        )
        masks_out.append((mask > 0).to(torch.uint8).squeeze(0).squeeze(0))
        ious.append(iou_pred.reshape(-1)[0].float())
    return torch.stack(masks_out, dim=0), torch.stack(ious, dim=0)


@torch.no_grad()
def predict_from_embeddings(
    sam,
    image_embeddings: torch.Tensor,
    bboxes: torch.Tensor,
    H: int,
    W: int,
) -> torch.Tensor:
    """Masks only; kept for scripts/eval_all_methods.py and bbox_robustness."""
    masks, _ = predict_from_embeddings_with_iou(sam, image_embeddings, bboxes, H, W)
    return masks


def write_per_image_csv(path: Path, rows: list[dict]) -> Path:
    """Columns follow the first row's keys (image_id, dice, iou, hd95, iou_pred[, drift])."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["image_id", "dice", "iou", "hd95", "iou_pred"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return path
```

(d) In `evaluate()`, right after `device = get_device(prefer=preferred)`:
```python
    if args.bbox_perturb is not None:
        cfg["eval"]["bbox_perturb_pixels"] = int(args.bbox_perturb)
    limit = args.limit if args.limit is not None else (8 if args.quick else None)
```
After `sam.eval()` (the adapted model), add the drift setup:
```python
    base_sam = None
    cur_hook = base_hook = None
    shares_encoder = False
    if args.drift:
        from cka.hooks import register_hooks

        base_sam = load_medsam(base_ckpt, arch=cfg["model"]["arch"], device=device)
        setup_method(base_sam, "zero_shot")
        base_sam.eval()
        cur_hook = register_hooks(sam, ["decoder_upscaling"], detach=True, accumulate=True)
        base_hook = register_hooks(base_sam, ["decoder_upscaling"], detach=True, accumulate=True)
        shares_encoder = not encoder_in_grad_path(method)
        print(f"[eval] drift enabled (base encoder reused: {shares_encoder})")
```

(e) Replace `if args.quick: ds.items = ds.items[:8]` with `if limit is not None: ds.items = ds.items[:limit]`.

(f) Replace the batch loop body from `preds = predict_batch(sam, images, bboxes)` through the `per_image_rows.append(...)` with:
```python
            H, W = images.shape[-2:]
            embeddings = sam.image_encoder(images)
            if cur_hook is not None:
                cur_hook.clear()
            preds, iou_preds = predict_from_embeddings_with_iou(sam, embeddings, bboxes, H, W)
            drifts = None
            if base_sam is not None:
                base_emb = embeddings if shares_encoder else base_sam.image_encoder(images)
                base_hook.clear()
                predict_from_embeddings_with_iou(base_sam, base_emb, bboxes, H, W)
                cur_acts = cur_hook.stacked()["decoder_upscaling"]
                base_acts = base_hook.stacked()["decoder_upscaling"]
                drifts = [
                    decoder_drift(base_acts[j], cur_acts[j]) for j in range(cur_acts.shape[0])
                ]
                cur_hook.clear()
                base_hook.clear()
            preds_np = preds.cpu().numpy()
            iou_np = iou_preds.cpu().numpy()
            for j in range(preds_np.shape[0]):
                pj = preds_np[j]
                gj = masks_gt[j]
                d = dice_score(pj, gj)
                i_ = iou_score(pj, gj)
                h_ = hd95(pj, gj)
                per_image.append({"dice": d, "iou": i_, "hd95": h_})
                row = {
                    "image_id": batch["image_id"][j],
                    "dice": d,
                    "iou": i_,
                    "hd95": h_,
                    "iou_pred": float(iou_np[j]),
                }
                if drifts is not None:
                    row["drift"] = float(drifts[j])
                per_image_rows.append(row)
```
The whole loop must sit inside `with torch.no_grad():` (the encoder call is no longer wrapped by `predict_batch`). Wrap the `for batch in tqdm(loader, ...)` loop in `with torch.no_grad():`.

(g) `notes`: replace `"notes": "quick" if args.quick else ""` with
```python
                "notes": ";".join(
                    t for t in (
                        "quick" if args.quick else "",
                        f"limit{limit}" if limit is not None else "",
                        "drift" if args.drift else "",
                        f"pm{cfg['eval'].get('bbox_perturb_pixels', 0)}",
                    ) if t
                ),
```

(h) Per-image CSV block: replace it with
```python
        per_image_dir = (
            REPO_ROOT / args.per_image_dir
            if args.per_image_dir is not None
            else (REPO_ROOT / cfg["output"].get("per_image_csv", "results/raw/per_image.csv")).parent
        )
        per_image_path = write_per_image_csv(
            per_image_dir / f"{run_name}_{ds_name}_per_image.csv", per_image_rows
        )
        print(f"[eval] per-image -> {per_image_path}")
```

(i) Results CSV: replace `runs_path = REPO_ROOT / cfg["output"]["results_csv"]` with
```python
    runs_path = REPO_ROOT / (args.results_csv or cfg["output"]["results_csv"])
```
Finally, at the end of `evaluate()` before `return 0`, remove hooks:
```python
    if cur_hook is not None:
        cur_hook.remove()
        base_hook.remove()
```

- [ ] **Step 5: `--eval-args` passthrough in scripts/eval_all_checkpoints.py**

Add `import shlex`, the argument
```python
    p.add_argument(
        "--eval-args",
        default="",
        help='Extra arguments passed verbatim to src.eval, e.g. "--drift --limit 100".',
    )
```
and after `cmd.extend(["--device", args.device])`:
```python
        if args.eval_args:
            cmd.extend(shlex.split(args.eval_args))
```

- [ ] **Step 6: Create configs/accv_eval.yaml**

```yaml
# ACCV TrustFMI eval config: four-dataset drift ladder, tight boxes unless
# overridden with --bbox-perturb. Outputs default under results/accv/.
name: zero_shot
method: zero_shot

model:
  arch: vit_b
  checkpoint: checkpoints/medsam_vit_b.pth
  image_size: 1024

data:
  root: data
  test_sets:
    - name: isic2018_test
      kind: isic
      split: test
    - name: ph2
      kind: ph2
      root: data/ph2
    - name: busi
      kind: busi
      root: data/busi
    - name: cbis_ddsm
      kind: cbis_ddsm
      root: data/cbis-ddsm
      split: test
      abnormality_type: all

eval:
  batch_size: 2
  num_workers: 4
  bbox_perturb_pixels: 0
  device: cuda

output:
  results_csv: results/accv/runs.csv
  per_image_csv: results/accv/raw/per_image.csv
```

- [ ] **Step 7: Run tests and a GPU smoke with drift**

Run: `~/mlenv/bin/python -m pytest tests/test_eval_detectors.py -v`
Expected: 6 passed.

Run:
```bash
cd /home/imsounic/Projects/medsam-vpt && ~/mlenv/bin/python -m src.eval --config configs/accv_eval.yaml --checkpoint checkpoints/runs/lora_seed0/best.pth --quick --drift --results-csv results/accv/smoke_runs.csv --per-image-dir results/accv/smoke_raw 2>&1 | grep -E "^\[eval\]" | tail -12
head -3 results/accv/smoke_raw/lora_seed0_cbis_ddsm_per_image.csv
```
Expected: four dataset lines with Dice numbers, per-image CSV header `image_id,dice,iou,hd95,iou_pred,drift`, drift values in (0, 1). Then delete the smoke outputs: `rm -rf results/accv/smoke_runs.csv results/accv/smoke_raw`.

- [ ] **Step 8: Commit**

```bash
git add src/drift.py src/eval.py scripts/eval_all_checkpoints.py configs/accv_eval.yaml tests/test_eval_detectors.py
git commit -m "feat(eval): per-image iou_pred, --drift decoder drift, path and limit overrides"
```

---

### Task 7: Config generators (T1 CKA runs, T2 budget sweep)

**Files:**
- Modify: `cka/generate_cka_configs.py`
- Create: `scripts/budget_sweep_configs.py`
- Test: `tests/test_config_generators.py`
- Generated: `configs/lora_cka_oodonly_*.yaml` (6), `configs/budget/*.yaml` (9)

- [ ] **Step 1: Write the failing tests**

```python
"""Generators emit the T1 (CKA ablation, multi-seed) and T2 (budget sweep) configs."""

import yaml

from cka.generate_cka_configs import ACCV_RUNS, emit_accv_configs
from scripts.budget_sweep_configs import BUDGETS, METHODS, emit_budget_configs


def _load(p):
    with open(p) as f:
        return yaml.safe_load(f)


def test_accv_runs_are_the_spec_set():
    names = [r["name"] for r in ACCV_RUNS]
    assert names == [
        "lora_cka_oodonly_late_l10_seed0",
        "lora_cka_oodonly_late_l10_pm20_seed0",
        "lora_cka_oodonly_late_l10_pm20_seed1",
        "lora_cka_oodonly_late_l10_pm20_seed2",
        "lora_cka_oodonly_enc_l10_pm20_seed0",
        "lora_cka_oodonly_both_l10_pm20_seed0",
    ]


def test_accv_configs_content(tmp_path):
    paths = emit_accv_configs(tmp_path)
    assert len(paths) == 6
    cfgs = {p.stem: _load(p) for p in paths}

    enc = cfgs["lora_cka_oodonly_enc_l10_pm20_seed0"]
    assert enc["cka_regularization"]["hook_layers"] == [
        "encoder_neck", "encoder_block_10", "encoder_block_11"
    ]
    assert enc["data"]["bbox_perturb_pixels"] == 20
    assert enc["seed"] == 0
    assert enc["cka_regularization"]["n_isic"] == 0
    assert enc["cka_regularization"]["n_busi"] == 16
    assert enc["cka_regularization"]["lambda"] == 10.0
    assert enc["output"]["checkpoint_dir"] == "checkpoints/runs_accv_t1"

    both = cfgs["lora_cka_oodonly_both_l10_pm20_seed0"]
    assert set(both["cka_regularization"]["hook_layers"]) == {
        "decoder_upscaling", "decoder_iou_head", "decoder_mask_logits",
        "encoder_neck", "encoder_block_10", "encoder_block_11",
    }
    assert set(both["cka_regularization"]["weights"]) == set(both["cka_regularization"]["hook_layers"])

    s2 = cfgs["lora_cka_oodonly_late_l10_pm20_seed2"]
    assert s2["seed"] == 2 and s2["name"] == "lora_cka_oodonly_late_l10_pm20_seed2"

    pm0 = cfgs["lora_cka_oodonly_late_l10_seed0"]
    assert pm0["data"]["bbox_perturb_pixels"] == 0
    assert pm0["output"]["checkpoint_dir"] == "checkpoints/runs_cka_oodonly_late"
    ref = cfgs["lora_cka_oodonly_late_l10_pm20_seed0"]
    assert ref["output"]["checkpoint_dir"] == "checkpoints/runs_cka_oodonly_late_pm20"


def test_budget_configs(tmp_path):
    paths = emit_budget_configs(tmp_path)
    assert len(paths) == 9
    assert BUDGETS == [50, 250, 1000]
    assert list(METHODS) == ["lora", "decoder_only", "lora_encoder_only"]
    cfgs = {p.stem: _load(p) for p in paths}
    c = cfgs["lora_encoder_only_n250_seed0"]
    assert c["method"] == "lora_encoder_only"
    assert c["method_kwargs"]["rank"] == 28 and c["method_kwargs"]["target_modules"] == "all"
    assert c["data"]["max_train_samples"] == 250 and c["data"]["subset_seed"] == 0
    assert c["train"]["max_steps"] == 6000 and c["train"]["val_every_steps"] == 500
    assert c["train"]["batch_size"] == 1
    assert c["data"]["bbox_perturb_pixels"] == 0
    assert c["output"]["checkpoint_dir"] == "checkpoints/runs_accv_t2"
    assert cfgs["decoder_only_n50_seed0"]["train"]["lr"] == 1.0e-4
    assert cfgs["lora_n1000_seed0"]["train"]["lr"] == 5.0e-4
    assert "method_kwargs" not in cfgs["decoder_only_n50_seed0"]
```

- [ ] **Step 2: Run to verify failure**

Run: `~/mlenv/bin/python -m pytest tests/test_config_generators.py -v`
Expected: `ImportError` (no `ACCV_RUNS`, no `scripts.budget_sweep_configs`). Note `scripts/` has no `__init__.py`; add an empty `scripts/__init__.py` so the test can import it.

- [ ] **Step 3: Extend cka/generate_cka_configs.py**

Add after `POSITION_WEIGHTS`:
```python
# ACCV 2026 TrustFMI (spec section 4.2): encoder-hook ablation positions.
POSITION_LAYERS["enc"] = ["encoder_neck", "encoder_block_10", "encoder_block_11"]
POSITION_LAYERS["both"] = POSITION_LAYERS["late"] + POSITION_LAYERS["enc"]
POSITION_WEIGHTS["enc"] = {"encoder_neck": 1.0, "encoder_block_10": 1.0, "encoder_block_11": 1.0}
POSITION_WEIGHTS["both"] = {**POSITION_WEIGHTS["late"], **POSITION_WEIGHTS["enc"]}

# OOD-only probe (0 ISIC + 16 BUSI + 16 CBIS), lambda 10. The first two runs
# already exist on the HPC and are listed so their configs live in git; the
# remaining four are the new T1 array.
ACCV_RUNS = [
    {"name": "lora_cka_oodonly_late_l10_seed0", "position": "late", "seed": 0,
     "perturb": 0, "checkpoint_dir": "checkpoints/runs_cka_oodonly_late"},
    {"name": "lora_cka_oodonly_late_l10_pm20_seed0", "position": "late", "seed": 0,
     "perturb": 20, "checkpoint_dir": "checkpoints/runs_cka_oodonly_late_pm20"},
    {"name": "lora_cka_oodonly_late_l10_pm20_seed1", "position": "late", "seed": 1,
     "perturb": 20, "checkpoint_dir": "checkpoints/runs_accv_t1"},
    {"name": "lora_cka_oodonly_late_l10_pm20_seed2", "position": "late", "seed": 2,
     "perturb": 20, "checkpoint_dir": "checkpoints/runs_accv_t1"},
    {"name": "lora_cka_oodonly_enc_l10_pm20_seed0", "position": "enc", "seed": 0,
     "perturb": 20, "checkpoint_dir": "checkpoints/runs_accv_t1"},
    {"name": "lora_cka_oodonly_both_l10_pm20_seed0", "position": "both", "seed": 0,
     "perturb": 20, "checkpoint_dir": "checkpoints/runs_accv_t1"},
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
```
Change `main()` to take `--accv-only` / `--out-dir` and emit both sets by default:
```python
def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=CONFIGS_DIR)
    ap.add_argument("--accv-only", action="store_true", help="Skip the original 9-config sweep")
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
```
(`emit_config` keeps writing into `CONFIGS_DIR`; only the ACCV set honours `--out-dir`.)

- [ ] **Step 4: Create scripts/budget_sweep_configs.py**

```python
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
seed: 0
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
  checkpoint_dir: checkpoints/runs_accv_t2
"""


def _kwargs_block(kwargs: dict | None) -> str:
    if not kwargs:
        return ""
    lines = ["", "method_kwargs:"]
    for k, v in kwargs.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines) + "\n"


def emit_budget_configs(out_dir: Path = OUT_DIR) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for method, spec in METHODS.items():
        for budget in BUDGETS:
            run_name = f"{method}_n{budget}_seed0"
            text = TEMPLATE.format(
                method=method,
                budget=budget,
                run_name=run_name,
                lr=spec["lr"],
                kwargs_block=_kwargs_block(spec["kwargs"]),
            )
            p = out_dir / f"{run_name}.yaml"
            p.write_text(text)
            written.append(p)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    for p in emit_budget_configs(args.out_dir):
        print(f"  wrote {p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests, then generate the real configs**

Run: `~/mlenv/bin/python -m pytest tests/test_config_generators.py -v`
Expected: 3 passed.

Run:
```bash
cd /home/imsounic/Projects/medsam-vpt && ~/mlenv/bin/python cka/generate_cka_configs.py --accv-only && ~/mlenv/bin/python scripts/budget_sweep_configs.py && git status --short configs | wc -l
```
Expected: 15 new config files (6 + 9).

- [ ] **Step 6: GPU smoke of a CKA encoder-hook config and a budget config with --quick**

```bash
cd /home/imsounic/Projects/medsam-vpt && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ~/mlenv/bin/python -m src.train --config configs/lora_cka_oodonly_both_l10_pm20_seed0.yaml --quick 2>&1 | grep -E "^\[train\]" | tail -8
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ~/mlenv/bin/python -m src.train --config configs/budget/lora_encoder_only_n50_seed0.yaml --quick 2>&1 | grep -E "^\[train\]" | tail -6
```
Expected for the CKA run: `[train][cka] base activations cached: {... 'encoder_neck': (4, 256, 64, 64), 'encoder_block_10': (4, 64, 64, 768) ...}` and one epoch line with a finite loss. Expected for the budget run: `[ISIC2018:train] subset: 50 of 2595`, `mode=steps segments=2 max_steps=6`, two `seg` lines (`step=3`, `step=6`). Then remove the smoke checkpoints:
```bash
rm -rf checkpoints/runs_accv_t1 checkpoints/runs_accv_t2
```
If the CKA smoke OOMs on 8 GB even with `--quick`, lower `n_busi`/`n_cbis` to 1 in `apply_quick_overrides` and retry; do not change the real configs.

- [ ] **Step 7: Commit**

```bash
git add cka/generate_cka_configs.py scripts/budget_sweep_configs.py scripts/__init__.py configs/lora_cka_oodonly_*.yaml configs/budget tests/test_config_generators.py
git commit -m "feat(configs): T1 CKA ablation/multi-seed and T2 budget-sweep generators"
```

---

### Task 8: Analysis scripts (failure detection, budget plot, hook-ablation plot)

**Files:**
- Create: `scripts/failure_detection.py`, `scripts/plot_budget.py`, `scripts/plot_hook_ablation.py`
- Test: `tests/test_failure_detection.py`, `tests/test_plots.py`

- [ ] **Step 1: Write the failing detector tests**

```python
"""AUROC / AUPRC / ECE from per-image CSVs; higher drift and lower iou_pred mean failure."""

import csv
import math

import numpy as np

from scripts.failure_detection import auprc, auroc, build_table, ece, load_per_image


def test_auroc_perfect_and_reversed():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([False, False, True, True])
    assert auroc(scores, labels) == 1.0
    assert auroc(-scores, labels) == 0.0


def test_auroc_handles_ties_and_degenerate():
    assert auroc(np.array([1.0, 1.0, 1.0]), np.array([True, False, True])) == 0.5
    assert math.isnan(auroc(np.array([0.1, 0.9]), np.array([True, True])))


def test_auprc_perfect_is_one():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([True, True, False, False])
    assert abs(auprc(scores, labels) - 1.0) < 1e-9


def test_ece_perfectly_calibrated_is_zero():
    conf = np.linspace(0.05, 0.95, 10)
    assert ece(conf, conf, n_bins=10) < 1e-9
    assert abs(ece(np.full(10, 0.9), np.full(10, 0.4), n_bins=10) - 0.5) < 1e-9


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_build_table_from_per_image_csvs(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        dice = float(rng.uniform(0.1, 1.0))
        rows.append({
            "image_id": f"img{i}", "dice": dice, "iou": dice / (2 - dice), "hd95": 5.0,
            "iou_pred": dice + rng.normal(0, 0.05),   # informative
            "drift": (1 - dice) + rng.normal(0, 0.05),  # informative
        })
    _write(tmp_path / "lora_seed0_cbis_ddsm_per_image.csv", rows)
    df = load_per_image(tmp_path, "lora_seed0", "cbis_ddsm")
    table = build_table({("lora_seed0", "cbis_ddsm"): df}, thresholds=(0.5, 0.7))
    assert {"run_name", "dataset", "detector", "threshold", "auroc", "auprc", "n", "n_fail"} <= set(table.columns)
    sub = table[(table.threshold == 0.5)]
    assert set(sub.detector) == {"iou_pred", "drift"}
    assert (sub.auroc > 0.9).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `~/mlenv/bin/python -m pytest tests/test_failure_detection.py -v`
Expected: `ModuleNotFoundError: No module named 'scripts.failure_detection'`.

- [ ] **Step 3: Create scripts/failure_detection.py**

```python
"""Failure self-detection: AUROC/AUPRC of iou_pred and drift against Dice-threshold failure labels, plus iou_pred calibration."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASETS = ["isic2018_test", "ph2", "busi", "cbis_ddsm"]
FAR_OOD = ["busi", "cbis_ddsm"]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUROC (ties get half credit). nan if only one class."""
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(np.asarray(scores, dtype=np.float64))
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision (area under the precision-recall step curve)."""
    labels = np.asarray(labels, dtype=bool)
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    hits = labels[order]
    tp = np.cumsum(hits)
    precision = tp / np.arange(1, len(hits) + 1)
    return float(precision[hits].sum() / labels.sum())


def ece(conf: np.ndarray, truth: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error of a confidence (iou_pred) against the true value (iou)."""
    conf = np.asarray(conf, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(conf[m].mean() - truth[m].mean())
    return float(total)


def reliability_bins(conf, truth, n_bins: int = 10) -> pd.DataFrame:
    conf = np.asarray(conf, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({
            "bin_lo": edges[b], "bin_hi": edges[b + 1], "n": int(m.sum()),
            "conf_mean": float(conf[m].mean()) if m.any() else float("nan"),
            "true_mean": float(truth[m].mean()) if m.any() else float("nan"),
        })
    return pd.DataFrame(rows)


def load_per_image(raw_dir: Path, run_name: str, dataset: str) -> pd.DataFrame:
    return pd.read_csv(Path(raw_dir) / f"{run_name}_{dataset}_per_image.csv")


def discover_runs(raw_dir: Path) -> list[str]:
    pat = re.compile(r"^(.+)_(" + "|".join(DATASETS) + r")_per_image\.csv$")
    names = sorted({m.group(1) for p in Path(raw_dir).glob("*_per_image.csv") if (m := pat.match(p.name))})
    return names


def build_table(frames: dict[tuple[str, str], pd.DataFrame], thresholds=(0.5, 0.7)) -> pd.DataFrame:
    rows = []
    for (run_name, dataset), df in frames.items():
        for thr in thresholds:
            fail = df["dice"].to_numpy() < thr
            detectors = {"iou_pred": -df["iou_pred"].to_numpy()}
            if "drift" in df.columns:
                detectors["drift"] = df["drift"].to_numpy()
            for det, score in detectors.items():
                rows.append({
                    "run_name": run_name, "dataset": dataset, "detector": det,
                    "threshold": thr, "auroc": auroc(score, fail), "auprc": auprc(score, fail),
                    "n": int(len(df)), "n_fail": int(fail.sum()),
                })
    return pd.DataFrame(rows)


def build_calibration(frames: dict[tuple[str, str], pd.DataFrame], n_bins: int = 10) -> pd.DataFrame:
    rows = []
    for (run_name, dataset), df in frames.items():
        rows.append({
            "run_name": run_name, "dataset": dataset, "n": int(len(df)),
            "ece": ece(df["iou_pred"].to_numpy(), df["iou"].to_numpy(), n_bins),
            "iou_pred_mean": float(df["iou_pred"].mean()), "iou_mean": float(df["iou"].mean()),
        })
    return pd.DataFrame(rows)


def make_figure(table: pd.DataFrame, frames: dict, out_path: Path, threshold: float,
                scatter_run: str | None, scatter_dataset: str = "cbis_ddsm") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = table[(table.threshold == threshold) & (table.dataset.isin(FAR_OOD))]
    runs = sorted(sub.run_name.unique())
    detectors = sorted(sub.detector.unique())
    n_panels = len(FAR_OOD) + (1 if scatter_run else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 3.6))
    axes = np.atleast_1d(axes)
    width = 0.8 / max(len(detectors), 1)
    x = np.arange(len(runs))
    for ax, ds in zip(axes[: len(FAR_OOD)], FAR_OOD):
        for k, det in enumerate(detectors):
            vals = [
                float(sub[(sub.run_name == r) & (sub.dataset == ds) & (sub.detector == det)].auroc.iloc[0])
                if len(sub[(sub.run_name == r) & (sub.dataset == ds) & (sub.detector == det)]) else np.nan
                for r in runs
            ]
            ax.bar(x + (k - (len(detectors) - 1) / 2) * width, vals, width, label=det)
        ax.axhline(0.5, color="grey", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(runs, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(f"{ds}: AUROC (Dice < {threshold})")
        ax.legend(fontsize=7)
    if scatter_run:
        ax = axes[-1]
        df = frames.get((scatter_run, scatter_dataset))
        if df is not None:
            ax.scatter(df["iou_pred"], df["dice"], s=8, alpha=0.6)
            ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--")
            ax.set_xlabel("iou_pred")
            ax.set_ylabel("Dice")
            ax.set_title(f"{scatter_run} on {scatter_dataset}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "results/accv/raw")
    ap.add_argument("--runs", nargs="*", default=None, help="Run names (default: discover)")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--thresholds", nargs="*", type=float, default=[0.5, 0.7])
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results/accv/failure_detection")
    ap.add_argument("--figure", type=Path, default=None)
    ap.add_argument("--scatter-run", default=None, help="Run for the iou_pred vs Dice scatter")
    args = ap.parse_args(argv)

    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else REPO_ROOT / args.raw_dir
    runs = args.runs or discover_runs(raw_dir)
    frames = {}
    for r in runs:
        for ds in args.datasets:
            p = raw_dir / f"{r}_{ds}_per_image.csv"
            if p.exists():
                frames[(r, ds)] = pd.read_csv(p)
    if not frames:
        print(f"[failure-detection] no per-image CSVs under {raw_dir}")
        return 1

    table = build_table(frames, tuple(args.thresholds))
    calib = build_calibration(frames)
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "detector_metrics.csv", index=False)
    calib.to_csv(out_dir / "calibration.csv", index=False)
    for (r, ds), df in frames.items():
        reliability_bins(df["iou_pred"], df["iou"]).to_csv(out_dir / f"reliability_{r}_{ds}.csv", index=False)

    primary = table[table.threshold == args.thresholds[0]]
    print(primary.pivot_table(index=["run_name", "dataset"], columns="detector", values="auroc").round(3).to_string())
    print(f"\n[failure-detection] wrote {out_dir / 'detector_metrics.csv'}")

    fig_path = args.figure or (out_dir / "failure_detection.png")
    make_figure(table, frames, fig_path, args.thresholds[0], args.scatter_run)
    print(f"[failure-detection] figure -> {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run detector tests**

Run: `~/mlenv/bin/python -m pytest tests/test_failure_detection.py -v`
Expected: 5 passed.

- [ ] **Step 5: Write the failing plot tests**

```python
"""Budget and hook-ablation plots run end to end on synthetic runs CSVs."""

import pandas as pd

from scripts.plot_budget import budget_summary, main as budget_main, parse_budget_run
from scripts.plot_hook_ablation import main as hook_main, parse_position

DATASETS = ["isic2018_test", "ph2", "busi", "cbis_ddsm"]


def _runs_csv(path, run_dice: dict[str, dict[str, float]], method_of=lambda r: r.split("_n")[0]):
    rows = []
    for run, per_ds in run_dice.items():
        for ds, dice in per_ds.items():
            rows.append({"run_name": run, "method": method_of(run), "dataset": ds, "seed": 0,
                         "dice_mean": dice, "dice_std": 0.01, "iou_mean": dice - 0.05, "hd95_mean": 10.0})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_parse_budget_run():
    assert parse_budget_run("lora_encoder_only_n250_seed0") == ("lora_encoder_only", 250)
    assert parse_budget_run("lora_seed0") is None


def test_budget_summary_and_first_drop(tmp_path):
    budget_csv = tmp_path / "runs_t2.csv"
    full_csv = tmp_path / "runs.csv"
    _runs_csv(budget_csv, {
        "lora_n50_seed0": dict(zip(DATASETS, [0.90, 0.90, 0.80, 0.70])),
        "lora_n250_seed0": dict(zip(DATASETS, [0.93, 0.93, 0.75, 0.60])),
        "lora_n1000_seed0": dict(zip(DATASETS, [0.95, 0.95, 0.70, 0.50])),
    })
    _runs_csv(full_csv, {
        "zero_shot": dict(zip(DATASETS, [0.90, 0.90, 0.82, 0.69])),
        "lora_seed0": dict(zip(DATASETS, [0.95, 0.95, 0.78, 0.50])),
    }, method_of=lambda r: r.replace("_seed0", ""))
    summary, first_drop = budget_summary(budget_csv, full_csv)
    lora = summary[summary.method == "lora"].sort_values("budget")
    assert list(lora.budget) == [50, 250, 1000, 2595]
    assert abs(lora.iloc[0].far_ood_dice - 0.75) < 1e-9
    assert first_drop["lora"] == 250  # 0.675 < zero-shot 0.755
    out = tmp_path / "budget.png"
    assert budget_main(["--budget-csv", str(budget_csv), "--full-csv", str(full_csv), "--out", str(out)]) == 0
    assert out.exists()


def test_parse_position():
    assert parse_position("lora_cka_oodonly_enc_l10_pm20_seed0") == "enc"
    assert parse_position("lora_cka_oodonly_late_l10_pm20_seed1") == "late"
    assert parse_position("lora_seed0_pm20") is None


def test_hook_ablation_plot(tmp_path):
    t1 = tmp_path / "runs_accv_t1.csv"
    base = tmp_path / "runs_pm20.csv"
    _runs_csv(t1, {
        "lora_cka_oodonly_late_l10_pm20_seed0": dict(zip(DATASETS, [0.95, 0.95, 0.82, 0.48])),
        "lora_cka_oodonly_enc_l10_pm20_seed0": dict(zip(DATASETS, [0.95, 0.95, 0.78, 0.44])),
        "lora_cka_oodonly_both_l10_pm20_seed0": dict(zip(DATASETS, [0.95, 0.95, 0.82, 0.48])),
    }, method_of=lambda r: "lora")
    _runs_csv(base, {"lora_seed0_pm20": dict(zip(DATASETS, [0.95, 0.95, 0.77, 0.44]))},
              method_of=lambda r: "lora")
    out = tmp_path / "hooks.png"
    rc = hook_main(["--csv", str(t1), "--baseline-csv", str(base), "--out", str(out)])
    assert rc == 0 and out.exists() and out.with_suffix(".csv").exists()
```

- [ ] **Step 6: Run to verify failure**

Run: `~/mlenv/bin/python -m pytest tests/test_plots.py -v`
Expected: `ModuleNotFoundError` for `scripts.plot_budget`.

- [ ] **Step 7: Create scripts/plot_budget.py**

```python
"""Adaptation budget vs Dice: in-domain and far-OOD (mean of BUSI, CBIS-DDSM), one line per method, zero-shot as reference."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FULL_BUDGET = 2595
FAR_OOD = ["busi", "cbis_ddsm"]
ID_SET = "isic2018_test"
# Full-data seed-0 pm=0 run names in results/runs.csv (encoder-only may be absent locally).
FULL_RUNS = {
    "lora": "lora_seed0",
    "decoder_only": "decoder_only_seed0",
    "lora_encoder_only": "lora_encoder_only_r28_all_seed0",
}
_BUDGET_RE = re.compile(r"^(?P<method>.+)_n(?P<budget>\d+)_seed\d+$")


def parse_budget_run(run_name: str) -> tuple[str, int] | None:
    m = _BUDGET_RE.match(run_name)
    return (m.group("method"), int(m.group("budget"))) if m else None


def _id_and_far(df: pd.DataFrame) -> tuple[float, float]:
    by_ds = df.groupby("dataset")["dice_mean"].mean()
    far = float(by_ds.reindex(FAR_OOD).mean())
    return float(by_ds.get(ID_SET, float("nan"))), far


def budget_summary(budget_csv: Path, full_csv: Path) -> tuple[pd.DataFrame, dict[str, int | None]]:
    bud = pd.read_csv(budget_csv)
    full = pd.read_csv(full_csv)
    rows = []
    for run, g in bud.groupby("run_name"):
        parsed = parse_budget_run(run)
        if parsed is None:
            continue
        method, budget = parsed
        idd, far = _id_and_far(g)
        rows.append({"method": method, "budget": budget, "id_dice": idd, "far_ood_dice": far})
    for method, run in FULL_RUNS.items():
        g = full[full.run_name == run]
        if len(g):
            idd, far = _id_and_far(g)
            rows.append({"method": method, "budget": FULL_BUDGET, "id_dice": idd, "far_ood_dice": far})
    zs = full[full.run_name == "zero_shot"]
    zs_id, zs_far = _id_and_far(zs) if len(zs) else (float("nan"), float("nan"))
    summary = pd.DataFrame(rows)
    summary.attrs["zero_shot"] = {"id_dice": zs_id, "far_ood_dice": zs_far}
    first_drop: dict[str, int | None] = {}
    for method, g in summary.groupby("method"):
        below = g[g.far_ood_dice < zs_far].sort_values("budget")
        first_drop[method] = int(below.budget.iloc[0]) if len(below) else None
    return summary, first_drop


def make_figure(summary: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zs = summary.attrs.get("zero_shot", {})
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    for ax, col, title in zip(axes, ["id_dice", "far_ood_dice"], ["In-domain (ISIC test)", "Far-OOD (mean BUSI, CBIS-DDSM)"]):
        for method, g in summary.groupby("method"):
            g = g.sort_values("budget")
            ax.plot(g.budget, g[col], marker="o", label=method)
        if zs.get(col) == zs.get(col):
            ax.axhline(zs[col], color="grey", ls="--", lw=0.9, label="zero-shot")
        ax.set_xscale("log")
        ax.set_xlabel("training images")
        ax.set_ylabel("Dice")
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-csv", type=Path, default=REPO_ROOT / "results/accv/runs_t2.csv")
    ap.add_argument("--full-csv", type=Path, default=REPO_ROOT / "results/runs.csv")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "figures/accv/budget_curves.png")
    args = ap.parse_args(argv)
    summary, first_drop = budget_summary(args.budget_csv, args.full_csv)
    summary.sort_values(["method", "budget"]).to_csv(args.out.with_suffix(".csv"), index=False)
    print(summary.sort_values(["method", "budget"]).round(4).to_string(index=False))
    for method, b in first_drop.items():
        print(f"[budget] {method}: first budget below zero-shot far-OOD = {b}")
    make_figure(summary, args.out)
    print(f"[budget] figure -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 8: Create scripts/plot_hook_ablation.py**

```python
"""Hook-placement ablation: far-OOD Dice for no-CKA vs decoder / encoder / both hooks (seed 0, pm=20)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FAR_OOD = ["busi", "cbis_ddsm"]
ORDER = ["no_cka", "late", "enc", "both"]
LABELS = {"no_cka": "no CKA", "late": "decoder hooks", "enc": "encoder hooks", "both": "both"}
_POS_RE = re.compile(r"^lora_cka_oodonly_(?P<pos>late|enc|both)_l10_pm20_seed(?P<seed>\d+)$")


def parse_position(run_name: str) -> str | None:
    m = _POS_RE.match(run_name)
    return m.group("pos") if m else None


def ablation_table(csv: Path, baseline_csv: Path, baseline_run: str) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df["position"] = df.run_name.map(parse_position)
    df = df[df.position.notna() & df.run_name.str.endswith("seed0")]
    base = pd.read_csv(baseline_csv)
    base = base[base.run_name == baseline_run].copy()
    base["position"] = "no_cka"
    both = pd.concat([base, df], ignore_index=True)
    both = both[both.dataset.isin(FAR_OOD)]
    return both.pivot_table(index="position", columns="dataset", values="dice_mean").reindex(ORDER)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=REPO_ROOT / "cka/results/runs_accv_t1.csv")
    ap.add_argument("--baseline-csv", type=Path, default=REPO_ROOT / "results/runs_pm20.csv")
    ap.add_argument("--baseline-run", default="lora_seed0_pm20")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "figures/accv/hook_ablation.png")
    args = ap.parse_args(argv)

    table = ablation_table(args.csv, args.baseline_csv, args.baseline_run)
    print(table.round(4).to_string())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out.with_suffix(".csv"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.4))
    x = np.arange(len(table.index))
    w = 0.38
    for k, ds in enumerate(FAR_OOD):
        ax.bar(x + (k - 0.5) * w, table[ds].to_numpy(), w, label=ds)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[p] for p in table.index])
    ax.set_ylabel("Dice (tight boxes)")
    ax.set_title("CKA hook placement, seed 0, pm=20")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"[hook-ablation] figure -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 9: Run tests**

Run: `~/mlenv/bin/python -m pytest tests/test_plots.py tests/test_failure_detection.py -v`
Expected: 9 passed.

- [ ] **Step 10: Commit**

```bash
git add scripts/failure_detection.py scripts/plot_budget.py scripts/plot_hook_ablation.py tests/test_failure_detection.py tests/test_plots.py
git commit -m "feat(analysis): failure-detection metrics, budget and hook-ablation plots"
```

---

### Task 9: SLURM arrays, eval chains, submit and sync scripts

**Files:**
- Create: `cka/slurm/accv_t1.sbatch`, `cka/slurm/accv_t1_eval.sbatch`, `cka/slurm/accv_t2.sbatch`, `cka/slurm/accv_t2_eval.sbatch`, `cka/slurm/accv_t3.sbatch`, `cka/slurm/submit_accv.sh`, `cka/slurm/sync_to_hpc.sh`
- Test: `tests/test_slurm.py`

Conventions (from PROJECT_STATE.md): repo at `~/medsam-vpt` on the HPC, `partition main-gpu`, conda env `medsam-vpt` activated via `source /software/anaconda3/2025.06/etc/profile.d/conda.sh`, env vars `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `LANG=en_US.UTF-8`, logs under `logs/`. QOS allows 2 concurrent GPU jobs, so a 4-task array runs two at a time.

- [ ] **Step 1: Write the failing tests**

```python
"""SLURM array ranges match their config lists, referenced configs exist, and bash parses."""

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SLURM = REPO / "cka" / "slurm"


def _array_len(text: str) -> int:
    m = re.search(r"^#SBATCH --array=(\d+)-(\d+)$", text, re.M)
    assert m, "missing --array"
    return int(m.group(2)) - int(m.group(1)) + 1


def _bash_array(text: str, name: str) -> list[str]:
    m = re.search(rf"^{name}=\((.*?)\)", text, re.M | re.S)
    assert m, f"missing {name}=( ... )"
    return [t.strip().strip('"') for t in m.group(1).split() if t.strip()]


@pytest.mark.parametrize("script,var,n", [("accv_t1.sbatch", "CONFIGS", 4), ("accv_t2.sbatch", "CONFIGS", 9)])
def test_train_arrays_match_configs(script, var, n):
    text = (SLURM / script).read_text()
    items = _bash_array(text, var)
    assert len(items) == n
    assert _array_len(text) == n
    for cfg in items:
        assert (REPO / "configs" / cfg).exists(), cfg


def test_t3_array_covers_two_prompt_levels():
    text = (SLURM / "accv_t3.sbatch").read_text()
    assert _bash_array(text, "PERTURBS") == ["0", "50"]
    assert _array_len(text) == 2
    assert len(_bash_array(text, "CHECKPOINTS")) == 6  # plus zero-shot handled separately
    assert "--drift" in text


@pytest.mark.parametrize("script", sorted(p.name for p in SLURM.glob("accv_*")) + ["submit_accv.sh", "sync_to_hpc.sh"])
def test_bash_syntax(script):
    subprocess.run(["bash", "-n", str(SLURM / script)], check=True)


def test_submit_chain_dependencies():
    text = (SLURM / "submit_accv.sh").read_text()
    assert "afterany:$T1" in text and "afterany:$T2" in text
    assert text.index("accv_t1.sbatch") < text.index("accv_t2.sbatch")
```

- [ ] **Step 2: Run to verify failure**

Run: `~/mlenv/bin/python -m pytest tests/test_slurm.py -v`
Expected: `FileNotFoundError` for the sbatch files.

- [ ] **Step 3: Create cka/slurm/accv_t1.sbatch**

```bash
#!/usr/bin/env bash
#SBATCH --job-name=accv_t1
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/accv_t1_%A_%a.out
#SBATCH --error=logs/accv_t1_%A_%a.err
# T1: CKA multi-seed (seeds 1, 2) and hook-placement ablation (enc, both).
# About 6 h each on an L40S; QOS runs two at a time.
set -euo pipefail
cd "$HOME/medsam-vpt"
mkdir -p logs
source /software/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate medsam-vpt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANG=en_US.UTF-8

CONFIGS=(
  lora_cka_oodonly_late_l10_pm20_seed1.yaml
  lora_cka_oodonly_late_l10_pm20_seed2.yaml
  lora_cka_oodonly_enc_l10_pm20_seed0.yaml
  lora_cka_oodonly_both_l10_pm20_seed0.yaml
)
CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "[accv_t1] task $SLURM_ARRAY_TASK_ID -> $CFG on $(hostname) at $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m src.train --config "configs/$CFG"
```

- [ ] **Step 4: Create cka/slurm/accv_t1_eval.sbatch**

```bash
#!/usr/bin/env bash
#SBATCH --job-name=accv_t1_eval
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/accv_t1_eval_%j.out
#SBATCH --error=logs/accv_t1_eval_%j.err
# T1 evals: tight-box eval of the four new checkpoints, bbox robustness sweep
# (20/50/100/200 px) for them, and the never-run sweep of the pm=0 late_l10 checkpoint.
set -euo pipefail
cd "$HOME/medsam-vpt"
mkdir -p logs cka/results
source /software/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate medsam-vpt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANG=en_US.UTF-8

python scripts/eval_all_checkpoints.py \
  --config configs/accv_eval.yaml \
  --checkpoint-glob 'checkpoints/runs_accv_t1/*/best.pth' \
  --eval-args "--results-csv cka/results/runs_accv_t1.csv --per-image-dir results/accv/raw_t1"

python bbox_robustness/eval_bbox_robust.py \
  --config configs/zero_shot.yaml \
  --checkpoint-glob 'checkpoints/runs_accv_t1/*/best.pth' \
  --n-samples 1 \
  --out-dir bbox_robustness/results_accv_t1

PM0="checkpoints/runs_cka_oodonly_late/lora_cka_oodonly_late_l10_seed0/best.pth"
if [[ -f "$PM0" ]]; then
  python bbox_robustness/eval_bbox_robust.py \
    --config configs/zero_shot.yaml \
    --checkpoint-glob "$PM0" \
    --n-samples 1 \
    --out-dir bbox_robustness/results_cka_oodonly_late_l10_pm0
else
  echo "[accv_t1_eval] WARNING: $PM0 not found; pm=0 robustness sweep skipped"
fi
echo "[accv_t1_eval] done at $(date -Iseconds)"
```

- [ ] **Step 5: Create cka/slurm/accv_t2.sbatch**

```bash
#!/usr/bin/env bash
#SBATCH --job-name=accv_t2
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/accv_t2_%A_%a.out
#SBATCH --error=logs/accv_t2_%A_%a.err
# T2: adaptation budget sweep, 3 methods x {50, 250, 1000} images, 6000 steps each.
set -euo pipefail
cd "$HOME/medsam-vpt"
mkdir -p logs
source /software/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate medsam-vpt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANG=en_US.UTF-8

CONFIGS=(
  budget/lora_n50_seed0.yaml
  budget/lora_n250_seed0.yaml
  budget/lora_n1000_seed0.yaml
  budget/decoder_only_n50_seed0.yaml
  budget/decoder_only_n250_seed0.yaml
  budget/decoder_only_n1000_seed0.yaml
  budget/lora_encoder_only_n50_seed0.yaml
  budget/lora_encoder_only_n250_seed0.yaml
  budget/lora_encoder_only_n1000_seed0.yaml
)
CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "[accv_t2] task $SLURM_ARRAY_TASK_ID -> $CFG on $(hostname) at $(date -Iseconds)"
python -m src.train --config "configs/$CFG"
```

- [ ] **Step 6: Create cka/slurm/accv_t2_eval.sbatch**

```bash
#!/usr/bin/env bash
#SBATCH --job-name=accv_t2_eval
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/accv_t2_eval_%j.out
#SBATCH --error=logs/accv_t2_eval_%j.err
# T2 evals: tight boxes only, one src.eval per checkpoint (eval_all_methods.py
# would skip all but the first decoder_only checkpoint).
set -euo pipefail
cd "$HOME/medsam-vpt"
mkdir -p logs results/accv
source /software/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate medsam-vpt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANG=en_US.UTF-8

python scripts/eval_all_checkpoints.py \
  --config configs/accv_eval.yaml \
  --checkpoint-glob 'checkpoints/runs_accv_t2/*/best.pth' \
  --eval-args "--results-csv results/accv/runs_t2.csv --per-image-dir results/accv/raw_t2"
echo "[accv_t2_eval] done at $(date -Iseconds)"
```

- [ ] **Step 7: Create cka/slurm/accv_t3.sbatch**

```bash
#!/usr/bin/env bash
#SBATCH --job-name=accv_t3
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/accv_t3_%A_%a.out
#SBATCH --error=logs/accv_t3_%A_%a.err
# T3: iou_pred and decoder-drift dumps for the seven seed-0 pm=0 methods on the
# four-dataset ladder, at tight boxes (task 0) and 50 px jitter (task 1).
set -euo pipefail
cd "$HOME/medsam-vpt"
mkdir -p logs results/accv
source /software/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate medsam-vpt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LANG=en_US.UTF-8

PERTURBS=(0 50)
PM="${PERTURBS[$SLURM_ARRAY_TASK_ID]}"
OUT_CSV="results/accv/runs_t3_pm${PM}.csv"
RAW_DIR="results/accv/raw_t3_pm${PM}"

CHECKPOINTS=(
  checkpoints/runs/decoder_only_seed0/best.pth
  checkpoints/runs/vpt_shallow_seed0/best.pth
  checkpoints/runs/vpt_deep_seed0/best.pth
  checkpoints/runs/lora_seed0/best.pth
  checkpoints/runs_perfect_bboxes/lora_encoder_only_r28_all_seed0/best.pth
  checkpoints/runs/full_ft_seed0/best.pth
)

echo "[accv_t3] pm=$PM zero_shot"
python -m src.eval --config configs/accv_eval.yaml --drift --bbox-perturb "$PM" \
  --results-csv "$OUT_CSV" --per-image-dir "$RAW_DIR"
for CK in "${CHECKPOINTS[@]}"; do
  if [[ ! -f "$CK" ]]; then
    echo "[accv_t3] WARNING: missing $CK, skipped"
    continue
  fi
  echo "[accv_t3] pm=$PM $CK"
  python -m src.eval --config configs/accv_eval.yaml --checkpoint "$CK" --drift \
    --bbox-perturb "$PM" --results-csv "$OUT_CSV" --per-image-dir "$RAW_DIR"
done
echo "[accv_t3] done at $(date -Iseconds)"
```

- [ ] **Step 8: Create cka/slurm/submit_accv.sh and cka/slurm/sync_to_hpc.sh**

`submit_accv.sh`:
```bash
#!/usr/bin/env bash
# Submit the ACCV compute plan on the HPC head node: T1 array first, its evals
# after it, T2 queued behind T1, T2 evals after T2, T3 after T2 evals.
# Usage (on hpc-head2, from ~/medsam-vpt): bash cka/slurm/submit_accv.sh
set -euo pipefail
cd "$HOME/medsam-vpt"
mkdir -p logs

T1=$(sbatch --parsable cka/slurm/accv_t1.sbatch)
echo "T1 array:      $T1"
T1E=$(sbatch --parsable --dependency=afterany:$T1 cka/slurm/accv_t1_eval.sbatch)
echo "T1 eval:       $T1E (afterany:$T1)"
T2=$(sbatch --parsable --dependency=afterany:$T1 cka/slurm/accv_t2.sbatch)
echo "T2 array:      $T2 (afterany:$T1)"
T2E=$(sbatch --parsable --dependency=afterany:$T2 cka/slurm/accv_t2_eval.sbatch)
echo "T2 eval:       $T2E (afterany:$T2)"
T3=$(sbatch --parsable --dependency=afterany:$T2E cka/slurm/accv_t3.sbatch)
echo "T3 dumps:      $T3 (afterany:$T2E)"
echo
squeue -u "$USER" --array
```

`sync_to_hpc.sh`:
```bash
#!/usr/bin/env bash
# Push code and configs (no data, checkpoints or results) to the HPC.
# Needs eduVPN. Usage from the repo root on the laptop: bash cka/slurm/sync_to_hpc.sh
set -euo pipefail
HOST="${HPC_HOST:-s3702111@hpc-head2.ewi.utwente.nl}"
DEST="${HPC_DEST:-~/medsam-vpt/}"
cd "$(dirname "$0")/../.."
rsync -avz --delete-excluded \
  --exclude '.git' --exclude 'data' --exclude 'checkpoints' --exclude 'results' \
  --exclude 'bbox_robustness/results*' --exclude 'logs' --exclude '__pycache__' \
  --exclude 'figures' --exclude 'colab' --exclude '*.zip' \
  ./ "$HOST:$DEST"
echo "[sync] done. Next: ssh $HOST 'cd ~/medsam-vpt && bash cka/slurm/submit_accv.sh'"
```

- [ ] **Step 9: Run tests**

Run: `~/mlenv/bin/python -m pytest tests/test_slurm.py -v`
Expected: all passed (config existence checks need Task 7's generated files).

- [ ] **Step 10: Local dry run of the T3 loop body on the laptop (one checkpoint, --quick)**

```bash
cd /home/imsounic/Projects/medsam-vpt && ~/mlenv/bin/python scripts/eval_all_checkpoints.py --config configs/accv_eval.yaml --checkpoint-glob 'checkpoints/runs/decoder_only_seed0/best.pth' --eval-args "--quick --drift --bbox-perturb 50 --results-csv results/accv/smoke_t3.csv --per-image-dir results/accv/smoke_t3" 2>&1 | grep -E "^\[eval\]" | tail -6
rm -rf results/accv/smoke_t3.csv results/accv/smoke_t3
```
Expected: four dataset lines, `notes` containing `quick;drift;pm50`, and for decoder_only the message `base encoder reused: True`.

- [ ] **Step 11: Commit**

```bash
git add cka/slurm tests/test_slurm.py
git commit -m "feat(slurm): ACCV T1/T2/T3 arrays with afterany chain, submit and sync scripts"
```

---

### Task 10: Day-one `iou_pred` check (spec 6.3)

**Files:**
- Output: `results/accv/dayone/runs.csv`, `results/accv/dayone/raw/*_cbis_ddsm_per_image.csv`, `results/accv/dayone/failure_detection/*`

- [ ] **Step 1: Dump iou_pred and drift for zero-shot and the five local seed-0 checkpoints on 100 CBIS-DDSM images**

Create `configs/accv_dayone_cbis.yaml` by copying `configs/accv_eval.yaml` and keeping only the `cbis_ddsm` entry in `test_sets`, with outputs `results/accv/dayone/runs.csv` and `results/accv/dayone/raw/per_image.csv`. Then:
```bash
cd /home/imsounic/Projects/medsam-vpt
~/mlenv/bin/python -m src.eval --config configs/accv_dayone_cbis.yaml --limit 100 --drift 2>&1 | grep -E "^\[eval\] cbis"
for m in decoder_only vpt_shallow vpt_deep lora full_ft; do
  ~/mlenv/bin/python -m src.eval --config configs/accv_dayone_cbis.yaml --checkpoint checkpoints/runs/${m}_seed0/best.pth --limit 100 --drift 2>&1 | grep -E "^\[eval\] cbis"
done
```
Expected: six lines of CBIS Dice (zero-shot about 0.69, LoRA about 0.50, decoder-only about 0.83 on the full set; the 100-image prefix will differ somewhat). Full FT loads 93 M trainable weights; if it OOMs at batch 2, rerun that one with `--device cuda` after setting `eval.batch_size: 1` in the config.

- [ ] **Step 2: Compute AUROC**

```bash
~/mlenv/bin/python scripts/failure_detection.py --raw-dir results/accv/dayone/raw --datasets cbis_ddsm --out-dir results/accv/dayone/failure_detection --scatter-run lora_seed0
```
Expected: a pivot of AUROC per run for `iou_pred` and `drift` at Dice < 0.5, and `detector_metrics.csv`. Decision rule from the spec: if `iou_pred` AUROC < 0.6 for every method, it is reported as a one-sentence negative result and drift is the sole detector. Record the numbers in the final report.

- [ ] **Step 3: Commit the day-one artefacts**

```bash
git add configs/accv_dayone_cbis.yaml results/accv/dayone
git commit -m "results: day-one iou_pred/drift check on 100 CBIS-DDSM images, seed-0 checkpoints"
```

---

### Task 11: Full test run, formatting, final commit

- [ ] **Step 1: Format and run everything**

```bash
cd /home/imsounic/Projects/medsam-vpt
~/mlenv/bin/python -m black src scripts/budget_sweep_configs.py scripts/failure_detection.py scripts/plot_budget.py scripts/plot_hook_ablation.py scripts/eval_all_checkpoints.py tests cka/hooks.py cka/generate_cka_configs.py 2>&1 | tail -2
~/mlenv/bin/python -m isort src scripts tests cka/hooks.py cka/generate_cka_configs.py 2>&1 | tail -1
~/mlenv/bin/python -m pytest
```
(black/isort are in requirements.txt; install into mlenv with `uv pip install --python ~/mlenv/bin/python black isort` if missing.) Expected: all tests pass.

- [ ] **Step 2: Commit formatting if anything changed**

```bash
git add -A src scripts tests cka/hooks.py cka/generate_cka_configs.py
git commit -m "style: black/isort over accv-trustfmi changes" || true
```

- [ ] **Step 3: HPC submission (only if the head node is reachable)**

```bash
timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=5 s3702111@hpc-head2.ewi.utwente.nl hostname
```
If it prints a hostname: `bash cka/slurm/sync_to_hpc.sh`, then `ssh s3702111@hpc-head2.ewi.utwente.nl 'cd ~/medsam-vpt && bash cka/slurm/submit_accv.sh'`, and paste the job ids into the final report. If it times out (no eduVPN) or asks for a password, stop and report that the user must run those two commands from a VPN-connected shell before 13 September. Do not push to GitHub.

---

## Self-review against the spec

| Spec item | Task |
|---|---|
| 8.1 restore cka/, verify imports, extend generator, encoder hook position | 1, 2, 7 |
| 8.2 `lora_encoder_only` method and config | 3 |
| 8.3 ISIC `max_train_samples`, `subset_seed` | 4 |
| 8.4 `train.max_steps`, `train.val_every_steps`, scheduler over steps | 5 |
| 8.5 eval `iou_pred`, `--drift` | 6 |
| 8.6 `budget_sweep_configs.py`, `plot_budget.py`, `failure_detection.py`, `plot_hook_ablation.py` | 7, 8 |
| 8.7 restore `eval_bbox_robust.py` | 1 |
| 8.8 SLURM T1 (array of 4), T2 (array of 9), T3, evals with `afterany` | 9 |
| 4.2 four T1 runs plus the pm=0 robustness sweep | 7, 9 |
| 5.1 nine budget runs, 6000 steps, val every 500, tight boxes | 7 |
| 6.1 to 6.3 detectors, 7 methods x 4 datasets x 2 prompt levels, day-one check | 6, 9, 10 |
| 9 compute plan and `--quick` smoke tests | 5, 6, 7, 9, 11 |
| 7 DMID loader | not in scope of section 8; separate plan after 13 Sep |

Type consistency checked: `predict_from_embeddings_with_iou` (Task 6) is the only new eval entry point; `register_hooks(..., detach=True, accumulate=True)` and `stacked()` match the restored `cka/hooks.py`; `decoder_drift` takes `(C, H, W)` or `(1, C, H, W)`; `resolve_budget` returns `Budget(mode, n_segments, max_steps, val_every_steps)` and both tests and `train.py` use `n_segments`; generator functions `emit_accv_configs(out_dir)` and `emit_budget_configs(out_dir)` return lists of `Path`.
