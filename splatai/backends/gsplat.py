from __future__ import annotations

from pathlib import Path

from .base import (
    ArtifactKind,
    BackendCapabilities,
    BackendCommand,
    TrainingArtifact,
    TrainingJob,
)


class GsplatBackend:
    """Adapter for gsplat's example trainer.

    The adapter intentionally treats gsplat as an external runtime. SplatAI's
    desktop core should not import torch/CUDA directly; it launches a pinned
    Python runtime and consumes artifacts through this contract.
    """

    def __init__(
        self,
        python_executable: str = "python",
        trainer_script: str = "engines/gsplat/simple_trainer.py",
    ) -> None:
        self.python_executable = python_executable
        self.trainer_script = trainer_script

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="gsplat",
            supports_cpu=False,
            supports_cuda=True,
            supports_mcmc=True,
            supports_3dgut=True,
        )

    def build_command(self, job: TrainingJob) -> BackendCommand:
        if job.iterations <= 0:
            raise ValueError("iterations must be positive")

        # Current gsplat simple_trainer exposes MCMC as a tyro subcommand.
        args: list[str] = [
            self.trainer_script,
            "mcmc",
            "--data-dir",
            str(job.dataset),
            "--result-dir",
            str(job.output_dir),
            "--max-steps",
            str(job.iterations),
            "--save-ply",
            "True",
            "--ply-steps",
            str(job.iterations),
            "--disable-viewer",
            "True",
            "--disable-video",
            "True",
        ]

        if job.max_resolution is not None:
            # gsplat's COLMAP loader uses an integer downsample factor rather
            # than a max-resolution switch. Resolution normalization belongs
            # in SplatAI's dataset preparation layer, so this value is not
            # translated here.
            pass

        if job.enable_3dgut:
            # gsplat documents full 3DGUT mode as with_ut + with_eval3d and
            # requires MCMCStrategy rather than DefaultStrategy for eval3d.
            args.extend(["--with-ut", "True", "--with-eval3d", "True"])

        args.extend(job.extra_args)

        environment: dict[str, str] = {}
        if job.device and job.device.startswith("cuda:"):
            environment["CUDA_VISIBLE_DEVICES"] = job.device.split(":", 1)[1]

        return BackendCommand(
            executable=self.python_executable,
            args=tuple(args),
            cwd=Path.cwd(),
            environment=environment,
        )

    def discover_artifacts(self, job: TrainingJob) -> tuple[TrainingArtifact, ...]:
        artifacts: list[TrainingArtifact] = []

        ply_dir = job.output_dir / "ply"
        if ply_dir.is_dir():
            ply_candidates = sorted(ply_dir.glob("*.ply"), key=lambda p: p.stat().st_mtime)
            if ply_candidates:
                artifacts.append(
                    TrainingArtifact(
                        ArtifactKind.GAUSSIAN_PLY,
                        ply_candidates[-1],
                        {"strategy": "mcmc", "3dgut": job.enable_3dgut},
                    )
                )

        ckpt_dir = job.output_dir / "ckpts"
        if ckpt_dir.is_dir():
            checkpoint_candidates = sorted(
                ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime
            )
            if checkpoint_candidates:
                artifacts.append(
                    TrainingArtifact(
                        ArtifactKind.CHECKPOINT,
                        checkpoint_candidates[-1],
                        {"strategy": "mcmc"},
                    )
                )

        return tuple(artifacts)
