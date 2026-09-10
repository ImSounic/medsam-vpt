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
        "encoder_neck",
        "encoder_block_10",
        "encoder_block_11",
    ]
    assert enc["data"]["bbox_perturb_pixels"] == 20
    assert enc["seed"] == 0
    assert enc["cka_regularization"]["n_isic"] == 0
    assert enc["cka_regularization"]["n_busi"] == 16
    assert enc["cka_regularization"]["lambda"] == 10.0
    assert enc["output"]["checkpoint_dir"] == "checkpoints/runs_accv_t1"

    both = cfgs["lora_cka_oodonly_both_l10_pm20_seed0"]
    assert set(both["cka_regularization"]["hook_layers"]) == {
        "decoder_upscaling",
        "decoder_iou_head",
        "decoder_mask_logits",
        "encoder_neck",
        "encoder_block_10",
        "encoder_block_11",
    }
    assert set(both["cka_regularization"]["weights"]) == set(
        both["cka_regularization"]["hook_layers"]
    )

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
    assert (
        c["method_kwargs"]["rank"] == 28
        and c["method_kwargs"]["target_modules"] == "all"
    )
    assert c["data"]["max_train_samples"] == 250 and c["data"]["subset_seed"] == 0
    assert c["train"]["max_steps"] == 6000 and c["train"]["val_every_steps"] == 500
    assert c["train"]["batch_size"] == 1
    assert c["data"]["bbox_perturb_pixels"] == 0
    assert c["output"]["checkpoint_dir"] == "checkpoints/runs_accv_t2"
    assert cfgs["decoder_only_n50_seed0"]["train"]["lr"] == 1.0e-4
    assert cfgs["lora_n1000_seed0"]["train"]["lr"] == 5.0e-4
    assert "method_kwargs" not in cfgs["decoder_only_n50_seed0"]
