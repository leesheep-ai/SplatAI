from subprocess import CompletedProcess

from splatai.backends import ProductMode
from splatai.recommendation import recommend_mode
from splatai.runtime import RuntimeHealth, probe_runtime


def test_probe_runtime_parses_structured_health() -> None:
    def fake_runner(*args, **kwargs):
        return CompletedProcess(
            args=args,
            returncode=0,
            stdout='{"python_ok": true, "torch_ok": true, "gsplat_ok": true, "cuda_available": true, "torch_version": "2.x", "cuda_version": "13.x", "gsplat_version": "1.x", "gpu_name": "Test GPU", "vram_bytes": 17179869184, "compute_capability": "9.0"}\n',
            stderr="",
        )

    health = probe_runtime(runner=fake_runner)

    assert health.cuda_available is True
    assert health.gsplat_ok is True
    assert health.gpu_name == "Test GPU"
    assert health.compute_capability == "9.0"


def test_recommendation_prefers_ultimate_at_16_gib() -> None:
    health = RuntimeHealth(
        python_ok=True,
        torch_ok=True,
        gsplat_ok=True,
        cuda_available=True,
        vram_bytes=16 * 1024**3,
    )
    assert recommend_mode(health).mode == ProductMode.ULTIMATE


def test_recommendation_uses_quality_at_8_gib() -> None:
    health = RuntimeHealth(
        python_ok=True,
        torch_ok=True,
        gsplat_ok=True,
        cuda_available=True,
        vram_bytes=8 * 1024**3,
    )
    assert recommend_mode(health).mode == ProductMode.QUALITY


def test_recommendation_falls_back_without_gsplat() -> None:
    health = RuntimeHealth(
        python_ok=True,
        torch_ok=True,
        gsplat_ok=False,
        cuda_available=True,
        vram_bytes=24 * 1024**3,
    )
    assert recommend_mode(health).mode == ProductMode.COMPATIBLE
