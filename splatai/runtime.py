from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class RuntimeHealth:
    python_ok: bool
    torch_ok: bool
    gsplat_ok: bool
    cuda_available: bool
    torch_version: str | None = None
    cuda_version: str | None = None
    gsplat_version: str | None = None
    gpu_name: str | None = None
    vram_bytes: int | None = None
    compute_capability: str | None = None
    error: str | None = None


_PROBE_SCRIPT = r'''
import json
out = {
    "python_ok": True,
    "torch_ok": False,
    "gsplat_ok": False,
    "cuda_available": False,
}
try:
    import torch
    out["torch_ok"] = True
    out["torch_version"] = getattr(torch, "__version__", None)
    out["cuda_version"] = getattr(torch.version, "cuda", None)
    out["cuda_available"] = bool(torch.cuda.is_available())
    if out["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        out["gpu_name"] = props.name
        out["vram_bytes"] = int(props.total_memory)
        out["compute_capability"] = f"{props.major}.{props.minor}"
except Exception as exc:
    out["error"] = f"torch: {exc}"
try:
    import gsplat
    out["gsplat_ok"] = True
    out["gsplat_version"] = getattr(gsplat, "__version__", None)
except Exception as exc:
    if not out.get("error"):
        out["error"] = f"gsplat: {exc}"
print(json.dumps(out))
'''


def build_probe_command(python_executable: str = "python") -> tuple[str, ...]:
    return (python_executable, "-c", _PROBE_SCRIPT)


def probe_runtime(
    python_executable: str = "python",
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> RuntimeHealth:
    command: Sequence[str] = build_probe_command(python_executable)
    try:
        completed = runner(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        return RuntimeHealth(
            python_ok=False,
            torch_ok=False,
            gsplat_ok=False,
            cuda_available=False,
            error=str(exc),
        )

    if completed.returncode != 0:
        return RuntimeHealth(
            python_ok=False,
            torch_ok=False,
            gsplat_ok=False,
            cuda_available=False,
            error=(completed.stderr or completed.stdout or "probe failed").strip(),
        )

    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return RuntimeHealth(
            python_ok=True,
            torch_ok=False,
            gsplat_ok=False,
            cuda_available=False,
            error=f"invalid probe output: {exc}",
        )

    return RuntimeHealth(
        python_ok=bool(payload.get("python_ok")),
        torch_ok=bool(payload.get("torch_ok")),
        gsplat_ok=bool(payload.get("gsplat_ok")),
        cuda_available=bool(payload.get("cuda_available")),
        torch_version=payload.get("torch_version"),
        cuda_version=payload.get("cuda_version"),
        gsplat_version=payload.get("gsplat_version"),
        gpu_name=payload.get("gpu_name"),
        vram_bytes=payload.get("vram_bytes"),
        compute_capability=payload.get("compute_capability"),
        error=payload.get("error"),
    )
