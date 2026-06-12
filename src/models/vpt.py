"""Visual Prompt Tuning for MedSAM ViT-B: adds learnable vectors at fixed spatial positions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from segment_anything.modeling import Sam


class VPTSAMEncoder(nn.Module):
    """Wraps SAM's image encoder with VPT-style learnable input perturbations."""

    def __init__(
        self,
        base_encoder: nn.Module,
        n_prompts: int = 10,
        mode: str = "deep",
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if mode not in {"shallow", "deep"}:
            raise ValueError(f"mode must be 'shallow' or 'deep', got {mode!r}")
        self.base = base_encoder
        for p in self.base.parameters():
            p.requires_grad = False

        self.mode = mode
        self.n_prompts = n_prompts
        self.gradient_checkpointing = gradient_checkpointing
        embed_dim = base_encoder.pos_embed.shape[-1]  # 768 for ViT-B
        depth = len(base_encoder.blocks)  # 12 for ViT-B

        if mode == "shallow":
            self.prompts = nn.Parameter(torch.zeros(n_prompts, embed_dim))
        else:
            self.layer_prompts = nn.ParameterList(
                [nn.Parameter(torch.zeros(n_prompts, embed_dim)) for _ in range(depth)]
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        params = [self.prompts] if self.mode == "shallow" else list(self.layer_prompts)
        for p in params:
            nn.init.normal_(p, std=0.02)

    def _add_prompts(self, x: torch.Tensor, prompts: torch.Tensor) -> torch.Tensor:
        """Add prompts (n_prompts, C) to the first n_prompts row-major spatial positions of x (B, H, W, C)."""
        B, H, W, C = x.shape
        N = prompts.shape[0]
        x_flat = x.reshape(B, H * W, C)

        pert = torch.zeros_like(x_flat)
        pert[:, :N, :] = prompts.unsqueeze(0).expand(B, -1, -1)
        x_flat = x_flat + pert
        return x_flat.reshape(B, H, W, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base.patch_embed(x)
        if self.base.pos_embed is not None:
            x = x + self.base.pos_embed

        if self.mode == "shallow":
            x = self._add_prompts(x, self.prompts)

        for i, blk in enumerate(self.base.blocks):
            if self.mode == "deep":
                x = self._add_prompts(x, self.layer_prompts[i])
            if self.gradient_checkpointing and self.training:
                x = cp.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        x = self.base.neck(x.permute(0, 3, 1, 2))
        return x


def apply_vpt(
    sam: Sam,
    n_prompts: int = 10,
    mode: str = "deep",
    gradient_checkpointing: bool = False,
    **_kwargs,
) -> None:
    for p in sam.parameters():
        p.requires_grad = False

    base_encoder = sam.image_encoder
    device = next(sam.parameters()).device
    sam.image_encoder = VPTSAMEncoder(
        base_encoder,
        n_prompts=n_prompts,
        mode=mode,
        gradient_checkpointing=gradient_checkpointing,
    ).to(device)
    for p in sam.mask_decoder.parameters():
        p.requires_grad = True
