"""Visual Prompt Tuning for MedSAM ViT-B.

Standard VPT (Jia et al., ECCV 2022) prepends N learnable tokens to a
1D ViT input sequence. SAM's image encoder keeps a 2D spatial layout
throughout — patches are arranged as (B, H, W, C) tensors, attention is
windowed, and relative positional embeddings are 2D. Token-prepending
breaks all three.

Our adaptation: **additive perturbation**. We hold N learnable vectors
(matching standard VPT's parameter count) and add them to the first N
spatial positions of each layer's input. This:

  - keeps the parameter budget identical to canonical VPT
    (shallow:  N * 768   = 7,680  params with N=10)
    (deep:     12 * N * 768 = 92,160 params with N=10)
  - leaves SAM's window attention and relative positional embeddings intact
  - injects per-layer learnable bias for VPT-deep, mirroring the original
    paper's "fresh prompts at every block" recipe
  - is conceptually defensible: prompts modulate input features at every
    layer, the frozen backbone does the heavy lifting

We document this clearly in the report as a 2D-architecture-compatible
form of VPT. The trade-off vs token-prepending is a small loss in
expressiveness (prompts perturb fixed spatial positions rather than
participating in attention as full tokens), in exchange for full
compatibility with SAM's window attention.
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
        depth = len(base_encoder.blocks)              # 12 for ViT-B

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
        """Add `prompts` to the first n_prompts spatial positions of `x`.

        x:        (B, H, W, C)
        prompts:  (n_prompts, C)
        returns:  (B, H, W, C) with prompts added to positions [0, ..., n_prompts-1]
                  in row-major order.
        """
        B, H, W, C = x.shape
        N = prompts.shape[0]
        x_flat = x.reshape(B, H * W, C)
        # Build a zero tensor and place prompts in the first N positions, then add.
        # Using addition + zeros (rather than in-place index assignment) keeps the
        # autograd graph clean.
        pert = torch.zeros_like(x_flat)
        pert[:, :N, :] = prompts.unsqueeze(0).expand(B, -1, -1)
        x_flat = x_flat + pert
        return x_flat.reshape(B, H, W, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embed + pos embed (frozen)
        x = self.base.patch_embed(x)
        if self.base.pos_embed is not None:
            x = x + self.base.pos_embed

        if self.mode == "shallow":
            x = self._add_prompts(x, self.prompts)

        for i, blk in enumerate(self.base.blocks):
            if self.mode == "deep":
                x = self._add_prompts(x, self.layer_prompts[i])
            if self.gradient_checkpointing and self.training:
                # Don't cache the block's activations; recompute during backward.
                # ~50% memory savings on encoder activations, ~30% slower.
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

    Replaces sam.image_encoder with a VPT-wrapped version, freezes everything
    except the new prompt parameters and the mask decoder.

    gradient_checkpointing: if True, encoder blocks recompute activations
        during backward instead of caching them. Cuts activation memory ~50%
        at ~30% time cost. Recommended on 8 GB GPUs.
    """
    # Freeze everything to start
    for p in sam.parameters():
        p.requires_grad = False

    # Wrap the image encoder
    base_encoder = sam.image_encoder
    device = next(sam.parameters()).device
    sam.image_encoder = VPTSAMEncoder(
        base_encoder,
        n_prompts=n_prompts,
        mode=mode,
        gradient_checkpointing=gradient_checkpointing,
    ).to(device)
    # The VPT prompts are nn.Parameter so they auto-have requires_grad=True;
    # the wrapped base encoder is frozen via __init__.

    # Mask decoder is trainable for VPT (standard PEFT-on-SAM recipe — the
    # decoder has to translate prompt-modulated features into masks, and
    # leaving it frozen costs Dice with no parameter savings).
    for p in sam.mask_decoder.parameters():
        p.requires_grad = True
