"""Device selection and per-backend helpers.

Single source of truth for picking between CUDA, Apple Silicon (MPS) and CPU.
Auto-selects in priority order: CUDA > MPS > CPU, so the same code runs on:
  - Windows / Linux NVIDIA box → CUDA
  - Mac Mini / MacBook (M-series) → MPS
  - anywhere else → CPU

Why a helper module: train.py / eval.py have lots of `torch.cuda.*` calls
(peak memory, synchronize, empty_cache, manual_seed_all, pin_memory). Each
of those needs an MPS equivalent (or a no-op). Centralising avoids
sprinkling `if device == ...` branches everywhere.

Conventions:
  - `device` is always a string: "cuda" | "mps" | "cpu".
  - AMP / fp16 autocast is only enabled on CUDA. MPS autocast is still
    rough (some ops fall back to fp32 silently, GradScaler isn't supported);
    we keep MPS in fp32 for correctness.
  - pin_memory is CUDA-only. DataLoader on MPS warns if pin_memory=True.
"""
from __future__ import annotations

import torch


def get_device(prefer: str | None = None) -> str:
    """Return the best available device string.

    Args:
        prefer: optional user override ("cuda", "mps", "cpu"). If the
            requested backend is unavailable we fall back automatically
            and print a notice. None = auto-pick.
    """
    if prefer is not None:
        prefer = prefer.lower()
        if prefer == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            print("[device] CUDA requested but unavailable; falling back.")
        elif prefer == "mps":
            if _mps_available():
                return "mps"
            print("[device] MPS requested but unavailable; falling back.")
        elif prefer == "cpu":
            return "cpu"
        else:
            print(f"[device] unknown preference {prefer!r}; auto-selecting.")

    if torch.cuda.is_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"


def _mps_available() -> bool:
    """True if Apple Metal backend is built into this torch and the chip supports it."""
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def device_name(device: str) -> str:
    """Human-readable name for printing."""
    if device == "cuda":
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "CUDA"
    if device == "mps":
        return "Apple Silicon (MPS)"
    return "CPU"


def seed_all(seed: int) -> None:
    """Seed every backend we might touch."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if _mps_available() and hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def reset_peak_memory(device: str) -> None:
    """Reset peak-memory counter on the active accelerator. No-op on CPU."""
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # MPS exposes `torch.mps.driver_allocated_memory()` (current) but no
    # reset-style API for a separate peak counter — we just report current
    # usage at the end via peak_memory_mb().


def peak_memory_mb(device: str) -> float:
    """Return peak (or current) accelerator memory in MB. 0.0 on CPU."""
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    if device == "mps" and hasattr(torch, "mps"):
        # Best signal we have on MPS — current driver allocation in bytes.
        # Underreports vs. true peak but better than nothing.
        if hasattr(torch.mps, "driver_allocated_memory"):
            return torch.mps.driver_allocated_memory() / (1024 * 1024)
    return 0.0


def synchronize(device: str) -> None:
    """Block until pending kernels finish. No-op on CPU."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def empty_cache(device: str) -> None:
    """Hint to the allocator to release cached blocks. No-op on CPU."""
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def supports_amp(device: str) -> bool:
    """True if torch.autocast + GradScaler are safe on this backend.

    We restrict to CUDA. MPS autocast exists in recent PyTorch but
    GradScaler does not support MPS, and several segment_anything ops
    silently fall back to fp32 anyway, so the speedup is unreliable.
    """
    return device == "cuda"


def supports_pin_memory(device: str) -> bool:
    """DataLoader pin_memory is only meaningful for CUDA transfer."""
    return device == "cuda"


def autocast_device_type(device: str) -> str:
    """Argument for `torch.autocast(device_type=...)`. Falls back to 'cuda' for AMP-disabled calls (the kwarg is required even when enabled=False)."""
    return "cuda" if device == "cuda" else "cpu"
