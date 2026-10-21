"""Optional separate learning rate for the mask decoder (train.decoder_lr)."""

import torch.nn as nn

from src.train import build_param_groups


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Linear(4, 4)
        self.mask_decoder = nn.Linear(4, 2)
        self.prompt_encoder = nn.Linear(2, 2)
        for p in self.prompt_encoder.parameters():
            p.requires_grad = False


def test_single_group_without_decoder_lr():
    groups = build_param_groups(_Toy(), lr=5e-4)
    assert len(groups) == 1
    assert groups[0]["lr"] == 5e-4
    assert len(groups[0]["params"]) == 4  # encoder + decoder weight/bias


def test_decoder_gets_its_own_lr():
    m = _Toy()
    groups = build_param_groups(m, lr=5e-4, decoder_lr=1e-4)
    by_lr = {g["lr"]: g["params"] for g in groups}
    assert set(by_lr) == {5e-4, 1e-4}
    dec = {id(p) for p in m.mask_decoder.parameters()}
    assert {id(p) for p in by_lr[1e-4]} == dec
    assert not dec & {id(p) for p in by_lr[5e-4]}


def test_frozen_params_excluded_and_empty_groups_dropped():
    m = _Toy()
    for p in m.mask_decoder.parameters():
        p.requires_grad = False
    groups = build_param_groups(m, lr=5e-4, decoder_lr=1e-4)
    assert len(groups) == 1 and groups[0]["lr"] == 5e-4
    assert len(groups[0]["params"]) == 2
