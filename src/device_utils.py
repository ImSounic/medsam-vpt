"""Device selection (CUDA > MPS > CPU) and per-backend helpers; AMP/pin_memory are CUDA-only."""
from __future__ import annotations

import torch


def get_device(prefer: str | None = None) -> str:
    """Return the best available device string, honoring `prefer` then auto-picking."""
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
    # MPS has no peak-reset API; peak_memory_mb() reports current usage instead.


def peak_memory_mb(device: str) -> float:
    """Return peak (or current) accelerator memory in MB. 0.0 on CPU."""
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    if device == "mps" and hasattr(torch, "mps"):
        # Current driver allocation; underreports true peak but it's all MPS gives.
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
    """CUDA only; MPS has no GradScaler and several SAM ops fall back to fp32."""
    return device == "cuda"


def supports_pin_memory(device: str) -> bool:
    """DataLoader pin_memory is only meaningful for CUDA transfer."""
    return device == "cuda"


def autocast_device_type(device: str) -> str:
    """device_type arg for torch.autocast (required even when enabled=False)."""
    return "cuda" if device == "cuda" else "cpu"
