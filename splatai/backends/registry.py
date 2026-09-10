from __future__ import annotations

from enum import Enum

from .base import TrainingBackend
from .brush import BrushBackend
from .gsplat import GsplatBackend


class ProductMode(str, Enum):
    COMPATIBLE = "compatible"
    FAST = "fast"
    QUALITY = "quality"
    ULTIMATE = "ultimate"
    GEOMETRY = "geometry"
    LARGE_SCENE = "large_scene"


class BackendUnavailable(RuntimeError):
    pass


class BackendRegistry:
    def __init__(self) -> None:
        self._backends: dict[str, TrainingBackend] = {}

    def register(self, backend: TrainingBackend) -> None:
        name = backend.capabilities.name
        if name in self._backends:
            raise ValueError(f"backend already registered: {name}")
        self._backends[name] = backend

    def get(self, name: str) -> TrainingBackend:
        try:
            return self._backends[name]
        except KeyError as exc:
            raise BackendUnavailable(f"backend is not registered: {name}") from exc

    def select(self, mode: ProductMode) -> TrainingBackend:
        if mode == ProductMode.COMPATIBLE:
            return self.get("brush")
        if mode == ProductMode.QUALITY:
            return self.get("gsplat")
        if mode == ProductMode.ULTIMATE:
            # Phase 1 maps Ultimate to gsplat + 3DGUT. 3DGRUT/NHT will become
            # a distinct backend in a later milestone.
            return self.get("gsplat")
        raise BackendUnavailable(f"mode is planned but not implemented yet: {mode.value}")


def default_registry() -> BackendRegistry:
    registry = BackendRegistry()
    registry.register(BrushBackend())
    registry.register(GsplatBackend())
    return registry
