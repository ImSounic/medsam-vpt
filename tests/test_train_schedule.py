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
    assert b == Budget(
        mode="epochs", n_segments=6, max_steps=None, val_every_steps=None
    )


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
    cfg = {
        "train": {"epochs": 6},
        "cka_regularization": {
            "enabled": True,
            "n_isic": 0,
            "n_busi": 16,
            "n_cbis": 16,
            "encoder_chunk": 8,
            "every_n_steps": 4,
        },
    }
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


def test_quick_overrides_enable_grad_checkpoint_and_scratch_dir():
    cfg = {"train": {"epochs": 6}, "output": {"checkpoint_dir": "checkpoints/runs"}}
    apply_quick_overrides(cfg)
    assert cfg["train"]["grad_checkpoint"] is True
    assert cfg["output"]["checkpoint_dir"] == "checkpoints/quick"
