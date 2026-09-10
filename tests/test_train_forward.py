"""forward_with_prompt with encoder gradient checkpointing matches the plain forward."""

import torch

from src.train import forward_with_prompt


def test_grad_checkpoint_forward_matches_plain(make_sam):
    sam = make_sam()
    torch.manual_seed(0)
    images = torch.randn(1, 3, 1024, 1024)
    bboxes = torch.tensor([[200.0, 200.0, 700.0, 700.0]])
    with torch.no_grad():
        plain = forward_with_prompt(sam, images, bboxes, encoder_grad=False)
    ckpt = forward_with_prompt(
        sam, images, bboxes, encoder_grad=True, grad_checkpoint=True
    )
    assert ckpt.shape == plain.shape == (1, 1, 1024, 1024)
    assert torch.allclose(ckpt.detach(), plain, atol=1e-4)
    assert ckpt.requires_grad
