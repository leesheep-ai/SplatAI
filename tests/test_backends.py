from pathlib import Path

import pytest

from splatai.backends import (
    BackendUnavailable,
    BrushBackend,
    GsplatBackend,
    ProductMode,
    TrainingJob,
    default_registry,
)


def test_brush_command_matches_compatibility_contract(tmp_path: Path) -> None:
    job = TrainingJob(
        dataset=tmp_path / "dataset",
        output_dir=tmp_path / "out",
        iterations=30_000,
        max_resolution=1600,
    )
    command = BrushBackend(executable="brush_app").build_command(job)

    assert command.executable == "brush_app"
    assert command.args[:4] == (
        "--total-steps",
        "30000",
        "--max-resolution",
        "1600",
    )
    assert "--export-name" in command.args
    assert "final.ply.tmp" in command.args
    assert command.args[-1] == str(job.dataset)


def test_gsplat_defaults_to_mcmc_and_can_enable_3dgut(tmp_path: Path) -> None:
    job = TrainingJob(
        dataset=tmp_path / "dataset",
        output_dir=tmp_path / "out",
        iterations=30_000,
        device="cuda:1",
        enable_3dgut=True,
    )
    command = GsplatBackend(
        python_executable="python",
        trainer_script="engines/gsplat/simple_trainer.py",
    ).build_command(job)

    assert command.args[1] == "mcmc"
    assert ("--max-steps", "30000") == (
        command.args[command.args.index("--max-steps")],
        command.args[command.args.index("--max-steps") + 1],
    )
    assert "--with-ut" in command.args
    assert "--with-eval3d" in command.args
    assert command.environment["CUDA_VISIBLE_DEVICES"] == "1"


def test_registry_maps_quality_and_ultimate_to_gsplat() -> None:
    registry = default_registry()

    assert registry.select(ProductMode.COMPATIBLE).capabilities.name == "brush"
    assert registry.select(ProductMode.QUALITY).capabilities.name == "gsplat"
    assert registry.select(ProductMode.ULTIMATE).capabilities.name == "gsplat"

    with pytest.raises(BackendUnavailable):
        registry.select(ProductMode.FAST)
