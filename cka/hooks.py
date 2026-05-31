"""Forward-hook helpers for capturing decoder layer activations.

We need to compute CKA at specific intermediate layers of the MedSAM mask
decoder during training. PyTorch's forward hooks let us capture an activation
without modifying the model code; this module provides:

  - `register_hooks(model, layer_names) -> HookHandle` — attach hooks to a model
  - `HookHandle.activations` — dict mapping name -> Tensor of last forward's output
  - `HookHandle.remove()` — clean up the hooks when training ends

Supported layer names (relative to a `Sam` instance):

  encoder_block_6           - image_encoder.blocks[6] output
  encoder_block_11          - image_encoder.blocks[11] output  (last encoder block)
  decoder_transformer       - mask_decoder.transformer output (tuple, we keep [0])
  decoder_upscaling         - mask_decoder.output_upscaling output
  decoder_iou_head          - mask_decoder.iou_prediction_head output
  decoder_mask_logits       - mask_decoder full output (the low-res masks tensor)

The actual mapping to nn.Module objects happens in `_resolve_module`. Add new
mappings there if a layer not in the list above is needed.

Activations are detached only if we're capturing the BASE model (no gradients
needed). For the current trainable model we keep gradients so the CKA loss
backpropagates through the activations.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from segment_anything.modeling import Sam


# Maps a human-readable layer name to a function that returns the nn.Module
# whose forward output we want to capture. Each function takes the Sam model.
def _resolve_module(sam: Sam, name: str) -> nn.Module:
    """Resolve a layer-name string to the actual nn.Module to hook on `sam`."""
    if name == "decoder_transformer":
        return sam.mask_decoder.transformer
    if name == "decoder_upscaling":
        return sam.mask_decoder.output_upscaling
    if name == "decoder_iou_head":
        return sam.mask_decoder.iou_prediction_head
    if name == "decoder_mask_logits":
        # Hook the entire mask_decoder; its forward returns (masks, iou_pred).
        # We capture the masks tensor in the post-process step below.
        return sam.mask_decoder
    if name.startswith("encoder_block_"):
        try:
            idx = int(name.removeprefix("encoder_block_"))
        except ValueError:
            raise ValueError(f"Bad encoder block index in {name!r}")
        # When VPT wraps the encoder, the blocks live at .base.blocks; otherwise
        # at .blocks directly. Support both.
        enc = sam.image_encoder
        blocks = getattr(getattr(enc, "base", None), "blocks", None) or enc.blocks
        if idx < 0 or idx >= len(blocks):
            raise ValueError(
                f"encoder_block_{idx} out of range (encoder has {len(blocks)} blocks)"
            )
        return blocks[idx]
    raise ValueError(
        f"Unknown layer name: {name!r}. "
        f"Supported: encoder_block_<i>, decoder_transformer, "
        f"decoder_upscaling, decoder_iou_head, decoder_mask_logits."
    )


def _normalize_output(name: str, output) -> torch.Tensor:
    """Unwrap layer-specific output formats into a single tensor.

    - decoder_transformer returns a (hs, src) tuple — keep hs (token outputs).
    - decoder_mask_logits is the full mask_decoder return: (low_res_masks, iou_pred)
      — keep low_res_masks.
    - Other layers return tensors directly.
    """
    if name == "decoder_transformer":
        # TwoWayTransformer.forward returns (queries, keys) tuple
        if isinstance(output, (tuple, list)):
            return output[0]
        return output
    if name == "decoder_mask_logits":
        if isinstance(output, (tuple, list)):
            return output[0]
        return output
    return output


class HookHandle:
    """Manages a set of registered forward hooks and exposes their activations.

    Two modes (toggled at construction):

    - accumulate=False  : the latest forward pass's activation overwrites the
                          previous one. Use when the hooked module is called
                          ONCE per forward (e.g., encoder blocks called on a
                          batched (B, 3, H, W) input).

    - accumulate=True   : each forward call's activation is appended to a list
                          per layer. Call `.stacked()` to get the concatenated
                          (B_total, ...) tensor. Use when the hooked module is
                          called MULTIPLE times in a per-sample loop (e.g., SAM's
                          mask_decoder, which is called once per probe image so
                          its hook fires 32 times for a 32-image probe).

    The accumulate mode is essential for decoder-layer hooks on SAM because the
    mask decoder is invoked inside a `for i in range(B)` loop, not on a batched
    input. Without accumulation we'd only see the last sample's activations and
    CKA would be computed on a 1-image probe.
    """

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
        # In non-accumulate mode: name -> Tensor (last call only).
        # In accumulate mode:     name -> list[Tensor] (one per call), .stacked() returns concat.
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
        """Reset accumulators. Call before each fresh forward pass when accumulating."""
        if self._accumulate:
            for name in self._accum:
                self._accum[name].clear()
        self.activations.clear()

    def stacked(self) -> dict[str, torch.Tensor]:
        """In accumulate mode, return name -> concatenated tensor (along batch dim).

        Concatenation is along dim 0 so a per-sample loop that calls the hooked
        module on B individual 1-sample inputs results in a (B, ...) tensor —
        exactly what we'd have gotten from a single batched forward.

        Raises:
            RuntimeError if not in accumulate mode (call `.activations` instead).
        """
        if not self._accumulate:
            raise RuntimeError("stacked() requires accumulate=True; use .activations instead")
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
    """Register forward hooks on `sam` at the requested layers.

    Args:
        sam: a Sam model.
        layer_names: which layers to hook (see `_resolve_module`).
        detach: True for the frozen BASE model (no grad needed); False for the
            trainable current model so CKA loss backpropagates.
        accumulate: True for hooks on per-sample-looped modules (mask_decoder
            and its sub-layers in SAM's pipeline). Call `.clear()` before each
            forward pass and `.stacked()` after to read the (B, ...) tensor.

    Returns:
        HookHandle. Remember to `.remove()` when done.
    """
    return HookHandle(sam, layer_names, detach=detach, accumulate=accumulate)
