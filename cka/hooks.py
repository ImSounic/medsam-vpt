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

    After each forward pass on the hooked model, `self.activations[name]` holds
    the latest captured tensor for that named layer. Tensors are kept in their
    raw (B, ...) shape — flatten to (B, D) before passing to `linear_cka()`.
    """

    def __init__(
        self,
        sam: Sam,
        layer_names: Iterable[str],
        detach: bool = False,
    ) -> None:
        self._handles = []
        self.activations: dict[str, torch.Tensor] = {}
        self._detach = detach

        for name in layer_names:
            module = _resolve_module(sam, name)
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def _make_hook(self, name: str):
        # Captured-name closure so each hook stores under the right key
        def _hook(_module, _inputs, output):
            t = _normalize_output(name, output)
            self.activations[name] = t.detach() if self._detach else t
        return _hook

    def remove(self) -> None:
        """Detach all hooks. Call when training is done or model is being torn down."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.activations.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()


def register_hooks(
    sam: Sam,
    layer_names: Iterable[str],
    detach: bool = False,
) -> HookHandle:
    """Register forward hooks on `sam` at the requested layers.

    Args:
        sam: a Sam model (possibly wrapped with VPT/LoRA — those don't affect
            decoder hooks; encoder hooks resolve through the VPT base when present).
        layer_names: iterable of names; see `_resolve_module` for the supported set.
        detach: if True, captured activations are .detach()-ed. Use for the BASE
            (frozen) model where gradients aren't needed. Keep False for the
            trainable model so CKA loss can backpropagate through the activations.

    Returns:
        HookHandle. Hooks remain active until `.remove()` is called.
    """
    return HookHandle(sam, layer_names, detach=detach)
