"""Cross-platform host/container compatibility for OpenReef workers.

CUDA and ROCm remain **separate images** (different base stacks). What is
auto-detectable:

1. **Inside the container (runtime):** WSL2 / Docker Desktop / native Linux
   cues → safe PyTorch env (disable fragile Triton native JIT, etc.).
2. **On the provider host (selection):** NVIDIA vs AMD → which compose/image
   tag to use.
3. **On a developer machine (build):** which Dockerfile to build; works from
   Linux, macOS, or Windows (Docker Desktop / Git Bash).

macOS note: you can *build/push* images with Docker, but Apple Silicon cannot
run CUDA or ROCm training. Training providers must be Linux or Windows+WSL2
with a discrete NVIDIA/AMD GPU.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Container / training-process detection
# ---------------------------------------------------------------------------


def _read_text(path: str, max_chars: int = 4096) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")[:max_chars]
    except OSError:
        return ""


def detect_container_host_context() -> dict[str, Any]:
    """Infer how this *Linux container* is hosted (WSL2, Docker Desktop, bare metal).

    Runs inside the worker image (always Linux). Host OS is inferred from
    kernel strings, cgroup, and optional OPENREEF_HOST_* overrides.
    """
    uname = platform.uname()
    proc_version = _read_text("/proc/version").lower()
    osrelease = _read_text("/proc/sys/kernel/osrelease").lower()
    cgroup = _read_text("/proc/1/cgroup").lower()

    # Explicit overrides from compose / provider-app (best signal on Windows).
    host_os_hint = (os.environ.get("OPENREEF_HOST_OS") or "").strip().lower()
    host_virt_hint = (os.environ.get("OPENREEF_HOST_VIRT") or "").strip().lower()

    is_wsl = any(
        marker in proc_version or marker in osrelease
        for marker in ("microsoft", "wsl", "wsl2")
    )
    if host_virt_hint in ("wsl", "wsl2"):
        is_wsl = True
    if host_os_hint in ("windows", "win32", "win"):
        # Windows Docker Desktop almost always means WSL2 Linux containers.
        is_wsl = True

    is_docker = Path("/.dockerenv").exists() or "docker" in cgroup or "containerd" in cgroup
    is_docker_desktop = is_docker and (
        is_wsl
        or "docker-desktop" in proc_version
        or bool(os.environ.get("DOCKER_DESKTOP") or os.environ.get("DESKTOP_SESSION_DOCKER"))
    )

    # "Native" = Linux kernel without Microsoft/WSL markers.
    is_native_linux = (not is_wsl) and sys.platform.startswith("linux")

    host_family = "unknown"
    if host_os_hint in ("linux", "darwin", "macos", "mac", "windows", "win32", "win"):
        if host_os_hint in ("darwin", "macos", "mac"):
            host_family = "macos"
        elif host_os_hint in ("windows", "win32", "win"):
            host_family = "windows"
        else:
            host_family = "linux"
    elif is_wsl:
        host_family = "windows"  # WSL2 guest under Windows
    elif is_native_linux:
        host_family = "linux"

    return {
        "kernel": uname.system,
        "kernel_release": uname.release,
        "machine": uname.machine,
        "python": platform.python_version(),
        "is_linux_container": sys.platform.startswith("linux"),
        "is_docker": is_docker,
        "is_wsl": is_wsl,
        "is_docker_desktop": is_docker_desktop,
        "is_native_linux": is_native_linux,
        "host_family": host_family,
        "host_os_hint": host_os_hint or None,
        "host_virt_hint": host_virt_hint or None,
        "proc_version_snippet": (proc_version[:120] if proc_version else None),
    }


def recommended_train_env(
    device: str,
    host: dict[str, Any] | None = None,
    *,
    force: bool | None = None,
) -> dict[str, str]:
    """Return env vars that should be set for stable training on this host.

    Policy:
    - NVIDIA CUDA: always disable PyTorch 2.9 native Triton JIT. Failures were
      observed on Windows/WSL2 (CudaUtils compile) and are harmless on bare
      Linux (eager CUDA path).
    - AMD ROCm: do not force CUDA triton knobs; leave ROCm experimental flags
      to compose if needed.
    - OPENREEF_FORCE_EAGER_TORCH=0|1 overrides.
    """
    host = host or detect_container_host_context()
    device = (device or "").strip().lower()
    out: dict[str, str] = {}

    if force is None:
        raw = (os.environ.get("OPENREEF_FORCE_EAGER_TORCH") or "").strip().lower()
        if raw in ("0", "false", "no", "off"):
            force = False
        elif raw in ("1", "true", "yes", "on"):
            force = True
        else:
            force = None

    want_eager = force
    if want_eager is None:
        # Default ON for CUDA everywhere (cross-platform safe).
        # Default ON extra-hard for WSL/Docker Desktop regardless of device label.
        want_eager = device in ("nvidia_cuda", "cuda", "nvidia") or bool(
            host.get("is_wsl") or host.get("is_docker_desktop")
        )

    if want_eager:
        out["TORCH_DISABLE_NATIVE_JIT"] = "1"
        out["TORCHDYNAMO_DISABLE"] = "1"
        out["TORCH_COMPILE_DISABLE"] = "1"
        reasons = []
        if device in ("nvidia_cuda", "cuda", "nvidia"):
            reasons.append("cuda_default")
        if host.get("is_wsl"):
            reasons.append("wsl")
        if host.get("is_docker_desktop"):
            reasons.append("docker_desktop")
        if force is True:
            reasons.append("force")
        out["OPENREEF_EAGER_TORCH_REASON"] = ",".join(reasons) or "default"

    return out


def apply_train_compat_env(env: dict[str, str], device: str) -> dict[str, str]:
    """Merge recommended compat vars into a train subprocess env (setdefault)."""
    host = detect_container_host_context()
    recommended = recommended_train_env(device, host)
    for key, value in recommended.items():
        env.setdefault(key, value)
    # Always expose detection for logs / live_data.
    env.setdefault("OPENREEF_DETECTED_HOST_FAMILY", str(host.get("host_family") or "unknown"))
    if host.get("is_wsl"):
        env.setdefault("OPENREEF_DETECTED_WSL", "1")
    return env


# ---------------------------------------------------------------------------
# Host-side backend selection (runs on provider / developer machine)
# ---------------------------------------------------------------------------


def _run(cmd: list[str], timeout: float = 5.0) -> tuple[int, str, str]:
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return p.returncode, p.stdout or "", p.stderr or ""
    except FileNotFoundError:
        return 127, "", "command not found"
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"


def detect_host_gpu_backend() -> dict[str, Any]:
    """Detect NVIDIA / AMD / none on the *host* (or inside a GPU-enabled container)."""
    code, out, err = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    nvidia_ok = code == 0 and bool(out.strip())
    nvidia_gpus: list[dict[str, Any]] = []
    if nvidia_ok:
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if not parts or not parts[0]:
                continue
            vram = None
            if len(parts) > 1:
                try:
                    vram = round(float(parts[1]) / 1024.0, 2)
                except ValueError:
                    vram = None
            nvidia_gpus.append({"name": parts[0], "vram_gb": vram})

    rocm_code, rocm_out, _ = _run(["rocminfo"])
    rocm_ok = rocm_code == 0 and bool(re.search(r"\bgfx[0-9a-fA-F]+", rocm_out))
    amd_names = re.findall(r"Marketing Name:\s*(.+)", rocm_out) if rocm_ok else []
    amd_names = [
        n.strip()
        for n in amd_names
        if n.strip()
        and not any(x in n.lower() for x in ("cpu", "processor", "apu"))
        and n.strip() != "N/A"
    ]

    # Prefer discrete GPU presence; if both exist, prefer NVIDIA for OpenReef
    # (qlora path) unless OPENREEF_PREFER_BACKEND is set.
    prefer = (os.environ.get("OPENREEF_PREFER_BACKEND") or "").strip().lower()
    backend = "cpu"
    if prefer in ("nvidia", "cuda", "nvidia_cuda") and nvidia_ok:
        backend = "nvidia_cuda"
    elif prefer in ("amd", "rocm", "amd_rocm") and rocm_ok:
        backend = "amd_rocm"
    elif nvidia_ok and not rocm_ok:
        backend = "nvidia_cuda"
    elif rocm_ok and not nvidia_ok:
        backend = "amd_rocm"
    elif nvidia_ok and rocm_ok:
        backend = "nvidia_cuda"  # dual-GPU default

    host_system = platform.system().lower()  # Linux, Darwin, Windows
    return {
        "backend": backend,
        "host_system": host_system,
        "host_machine": platform.machine(),
        "nvidia": {"ok": nvidia_ok, "gpus": nvidia_gpus, "error": None if nvidia_ok else (err or out)[:200]},
        "amd": {"ok": rocm_ok, "gpus": [{"name": n} for n in amd_names], "error": None if rocm_ok else "rocminfo not ok"},
        "compose_file": {
            "nvidia_cuda": "docker-compose-nvidia.yml",
            "amd_rocm": "docker-compose-amd.yml",
            "cpu": None,
        }.get(backend),
        "image_tag": {
            "nvidia_cuda": "ghcr.io/asphyksia/finetune-worker:cuda-latest",
            "amd_rocm": "ghcr.io/asphyksia/finetune-worker:rocm-latest",
            "cpu": None,
        }.get(backend),
        "dockerfile": {
            "nvidia_cuda": "Dockerfile",
            "amd_rocm": "Dockerfile.rocm",
            "cpu": None,
        }.get(backend),
        "notes": _host_notes(host_system, backend),
    }


def _host_notes(host_system: str, backend: str) -> list[str]:
    notes: list[str] = []
    if host_system == "darwin":
        notes.append(
            "macOS: Docker can build/push images, but CUDA/ROCm training needs "
            "a Linux or Windows+WSL2 host with a discrete GPU."
        )
    if host_system == "windows" and backend == "nvidia_cuda":
        notes.append(
            "Windows: use Docker Desktop with WSL2 + GPU support. "
            "Compose sets TORCH_DISABLE_NATIVE_JIT for Triton safety."
        )
    if host_system == "windows" and backend == "amd_rocm":
        notes.append(
            "Windows + AMD: ROCm in Docker on Windows is limited; prefer a Linux host for ROCm providers."
        )
    if backend == "cpu":
        notes.append("No NVIDIA/AMD GPU detected. Cannot select a finetune worker image.")
    return notes


def main() -> int:
    """CLI: print host backend selection as JSON."""
    import json

    mode = (sys.argv[1] if len(sys.argv) > 1 else "host").strip().lower()
    if mode in ("container", "runtime", "train-env"):
        host = detect_container_host_context()
        device = (os.environ.get("OPENREEF_PROVIDER_ENV") or "nvidia_cuda").strip()
        payload = {
            "mode": "container",
            "host": host,
            "recommended_train_env": recommended_train_env(device, host),
        }
    else:
        payload = {"mode": "host", **detect_host_gpu_backend()}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("backend", "cpu") != "cpu" or mode != "host" else 1


if __name__ == "__main__":
    raise SystemExit(main())
