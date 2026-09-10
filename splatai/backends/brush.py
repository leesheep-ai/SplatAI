from __future__ import annotations

from pathlib import Path

from .base import (
    ArtifactKind,
    BackendCapabilities,
    BackendCommand,
    TrainingArtifact,
    TrainingJob,
)


class BrushBackend:
    def __init__(self, executable: str = "brush_app") -> None:
        self.executable = executable

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="brush",
            supports_cpu=True,
            supports_cuda=True,
        )

    def build_command(self, job: TrainingJob) -> BackendCommand:
        args: list[str] = [
            "--total-steps",
            str(job.iterations),
        ]
        if job.max_resolution is not None:
            args.extend(["--max-resolution", str(job.max_resolution)])
        args.extend(
            [
                "--export-every",
                str(job.iterations),
                "--export-path",
                str(job.output_dir),
                "--export-name",
                "final.ply.tmp",
            ]
        )
        args.extend(job.extra_args)
        args.append(str(job.dataset))
        return BackendCommand(
            executable=self.executable,
            args=tuple(args),
            cwd=job.output_dir,
        )

    def discover_artifacts(self, job: TrainingJob) -> tuple[TrainingArtifact, ...]:
        candidates = (
            job.output_dir / "final.ply.tmp",
            job.output_dir / "final.ply.tmp.ply",
        )
        for candidate in candidates:
            if candidate.is_file():
                return (TrainingArtifact(ArtifactKind.GAUSSIAN_PLY, candidate),)
        return ()
