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
