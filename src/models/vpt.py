"""Visual Prompt Tuning for MedSAM ViT-B.

Non-canonical design: standard VPT (Jia et al., ECCV 2022) prepends N
learnable tokens to a 1D ViT sequence, but SAM's encoder is 2D throughout
(patches as (B,H,W,C), windowed attention, 2D rel pos embeds), so
token-prepending doesn't fit. Instead we add N learnable vectors to the
first N spatial positions of each layer's input (additive perturbation).

Param budget matches canonical VPT (shallow: N*768 = 7,680 at N=10;
deep: 12*N*768 = 92,160 at N=10) and window attention / rel pos embeds stay
intact. Trade-off: prompts perturb fixed spatial positions rather than
acting as attention tokens, so slightly less expressive. Documented in paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from segment_anything.modeling import Sam


class VPTSAMEncoder(nn.Module):
    """Wraps SAM's image encoder with VPT-style learnable input perturbations.

    Args:
        base_encoder: a frozen `ImageEncoderViT` instance.
        n_prompts: number of prompt vectors per layer (per-layer for deep,
            single set for shallow). Same as standard VPT's N.
        mode: "shallow" | "deep".
    """

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
        """Add prompts (n_prompts, C) to the first n_prompts row-major spatial
        positions of x (B, H, W, C). Returns same shape.
        """
        B, H, W, C = x.shape
        N = prompts.shape[0]
        x_flat = x.reshape(B, H * W, C)
        # Add via a zero tensor instead of in-place index assignment to keep
        # the autograd graph clean.
        pert = torch.zeros_like(x_flat)
        pert[:, :N, :] = prompts.unsqueeze(0).expand(B, -1, -1)
        x_flat = x_flat + pert
        return x_flat.reshape(B, H, W, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embed + pos embed (frozen).
        x = self.base.patch_embed(x)
        if self.base.pos_embed is not None:
            x = x + self.base.pos_embed

        if self.mode == "shallow":
            x = self._add_prompts(x, self.prompts)

        for i, blk in enumerate(self.base.blocks):
            if self.mode == "deep":
                x = self._add_prompts(x, self.layer_prompts[i])
            if self.gradient_checkpointing and self.training:
                # Recompute block activations in backward instead of caching:
                # ~50% less activation memory, ~30% slower.
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
    """Configure SAM in place for VPT.

    Wraps sam.image_encoder in VPTSAMEncoder; freezes everything except the
    prompt parameters and the mask decoder.

    gradient_checkpointing: recompute encoder activations in backward instead
        of caching (~50% less activation memory, ~30% slower). Use on 8 GB GPUs.
    """
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
    # Prompts are nn.Parameter (requires_grad=True by default); base encoder
    # frozen in __init__.

    # Mask decoder trainable: it must turn prompt-modulated features into
    # masks, and freezing it costs Dice for no param savings.
    for p in sam.mask_decoder.parameters():
        p.requires_grad = True
