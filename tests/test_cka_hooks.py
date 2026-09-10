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
