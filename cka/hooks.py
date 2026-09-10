"""Forward hooks capturing decoder layer activations for CKA; detach frozen base, keep grads on trainable model."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from segment_anything.modeling import Sam


def _resolve_module(sam: Sam, name: str) -> nn.Module:
    """Resolve a layer-name string to the nn.Module to hook on sam."""
    if name == "decoder_transformer":
        return sam.mask_decoder.transformer
    if name == "decoder_upscaling":
        return sam.mask_decoder.output_upscaling
    if name == "decoder_iou_head":
        return sam.mask_decoder.iou_prediction_head
    if name == "decoder_mask_logits":
        # Hook whole mask_decoder; forward returns (masks, iou_pred), keep masks.
        return sam.mask_decoder
    if name == "encoder_neck":
        enc = sam.image_encoder
        base = getattr(enc, "base", None)
        return base.neck if base is not None and hasattr(base, "neck") else enc.neck
    if name.startswith("encoder_block_"):
        try:
            idx = int(name.removeprefix("encoder_block_"))
        except ValueError:
            raise ValueError(f"Bad encoder block index in {name!r}")
        # VPT-wrapped encoder keeps blocks at .base.blocks; raw at .blocks.
        enc = sam.image_encoder
        blocks = getattr(getattr(enc, "base", None), "blocks", None) or enc.blocks
        if idx < 0 or idx >= len(blocks):
            raise ValueError(
                f"encoder_block_{idx} out of range (encoder has {len(blocks)} blocks)"
            )
        return blocks[idx]
    raise ValueError(
        f"Unknown layer name: {name!r}. "
        f"Supported: encoder_block_<i>, encoder_neck, decoder_transformer, "
        f"decoder_upscaling, decoder_iou_head, decoder_mask_logits."
    )


def _normalize_output(name: str, output) -> torch.Tensor:
    """Unwrap layer-specific output formats into a single tensor."""
    if name == "decoder_transformer":
        if isinstance(output, (tuple, list)):
            return output[0]
        return output
    if name == "decoder_mask_logits":
        if isinstance(output, (tuple, list)):
            return output[0]
        return output
    return output


class HookHandle:
    """Registers forward hooks and exposes activations; accumulate=True fires once per probe sample in the per-sample decoder loop."""

    def __init__(
        self,
        sam: Sam,
        layer_names: Iterable[str],
        detach: bool = False,
        accumulate: bool = False,
    ) -> None:
        self._handles = []
        self._detach = detach
        self._accumulate = accumulate
        # non-accumulate: name -> last Tensor; accumulate: name -> list, .stacked() concats.
        self.activations: dict[str, torch.Tensor] = {}
        self._accum: dict[str, list[torch.Tensor]] = {}

        for name in layer_names:
            module = _resolve_module(sam, name)
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)
            if self._accumulate:
                self._accum[name] = []

    def _make_hook(self, name: str):
        def _hook(_module, _inputs, output):
            t = _normalize_output(name, output)
            t = t.detach() if self._detach else t
            if self._accumulate:
                self._accum[name].append(t)
            else:
                self.activations[name] = t

        return _hook

    def clear(self) -> None:
        """Reset accumulators; call before each fresh forward pass when accumulating."""
        if self._accumulate:
            for name in self._accum:
                self._accum[name].clear()
        self.activations.clear()

    def stacked(self) -> dict[str, torch.Tensor]:
        """Accumulate mode: name -> tensor concatenated along dim 0."""
        if not self._accumulate:
            raise RuntimeError(
                "stacked() requires accumulate=True; use .activations instead"
            )
        out = {}
        for name, parts in self._accum.items():
            if not parts:
                continue
            out[name] = torch.cat(parts, dim=0)
        return out

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.activations.clear()
        self._accum.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()


def register_hooks(
    sam: Sam,
    layer_names: Iterable[str],
    detach: bool = False,
    accumulate: bool = False,
) -> HookHandle:
    """Register forward hooks on sam at the requested layers."""
    return HookHandle(sam, layer_names, detach=detach, accumulate=accumulate)
