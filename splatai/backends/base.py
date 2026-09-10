from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping, Protocol, Sequence


class ArtifactKind(str, Enum):
    GAUSSIAN_PLY = "gaussian_ply"
    CHECKPOINT = "checkpoint"
    MESH = "mesh"
    RUNTIME_ASSET = "runtime_asset"


@dataclass(frozen=True)
class TrainingArtifact:
    kind: ArtifactKind
    path: Path
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingJob:
    dataset: Path
    output_dir: Path
    iterations: int
    max_resolution: int | None = None
    device: str | None = None
    enable_3dgut: bool = False
    extra_args: Sequence[str] = ()


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    supports_cpu: bool
    supports_cuda: bool
    supports_mcmc: bool = False
    supports_3dgut: bool = False
    supports_mesh: bool = False


@dataclass(frozen=True)
class BackendCommand:
    executable: str
    args: tuple[str, ...]
    cwd: Path | None = None
    environment: Mapping[str, str] = field(default_factory=dict)


class TrainingBackend(Protocol):
    @property
    def capabilities(self) -> BackendCapabilities:
        ...

    def build_command(self, job: TrainingJob) -> BackendCommand:
        ...

    def discover_artifacts(self, job: TrainingJob) -> tuple[TrainingArtifact, ...]:
        ...
