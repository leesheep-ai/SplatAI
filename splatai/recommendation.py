from __future__ import annotations

from dataclasses import dataclass

from .backends.registry import ProductMode
from .runtime import RuntimeHealth


@dataclass(frozen=True)
class ModeRecommendation:
    mode: ProductMode
    reason: str


_GIB = 1024**3


def recommend_mode(health: RuntimeHealth) -> ModeRecommendation:
    """Choose a conservative default mode from runtime capabilities.

    This policy intentionally prefers predictability over chasing the most
    advanced backend. Expert users may override the result in the desktop UI.
    """

    if not (health.python_ok and health.torch_ok and health.gsplat_ok):
        return ModeRecommendation(
            ProductMode.COMPATIBLE,
            "gsplat runtime is incomplete; use the broad-compatibility backend",
        )

    if not health.cuda_available:
        return ModeRecommendation(
            ProductMode.COMPATIBLE,
            "CUDA is unavailable; use the broad-compatibility backend",
        )

    vram = health.vram_bytes or 0
    if vram >= 16 * _GIB:
        return ModeRecommendation(
            ProductMode.ULTIMATE,
            "CUDA runtime is healthy and at least 16 GiB VRAM is available",
        )

    if vram >= 8 * _GIB:
        return ModeRecommendation(
            ProductMode.QUALITY,
            "CUDA runtime is healthy and at least 8 GiB VRAM is available",
        )

    return ModeRecommendation(
        ProductMode.COMPATIBLE,
        "CUDA is available but detected VRAM is below the initial gsplat quality threshold",
    )
