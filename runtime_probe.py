"""Provider runtime diagnostics for OpenReef fine-tune workers.

The probe is intentionally small and dependency-light: it uses bounded shell
commands when available, then validates the Python training runtime with torch
and import checks. The resulting JSON can be logged by the worker or published
through provider live_data by the surrounding provider stack.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


PROVIDER_ENV_ALIASES = {
    "cuda": "nvidia_cuda",
    "nvidia": "nvidia_cuda",
    "nvidia_cuda": "nvidia_cuda",
    "rocm": "amd_rocm",
    "amd": "amd_rocm",
    "amd_rocm": "amd_rocm",
    "cpu": "cpu",
}

GPU_BACKENDS = {"nvidia_cuda", "amd_rocm"}


def normalise_provider_env(raw: str | None) -> str | None:
    value = (raw or "").strip().lower()
    if not value:
        return None
    expected = PROVIDER_ENV_ALIASES.get(value)
    if expected is None:
        raise RuntimeError(
            "Invalid OPENREEF_PROVIDER_ENV. Expected one of: "
            f"{', '.join(sorted(PROVIDER_ENV_ALIASES))}"
        )
    return expected


def _run_command(args: list[str], timeout_seconds: int = 3) -> dict[str, Any]:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError:
        return {
            "available": False,
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": "command not found",
        }
    except subprocess.TimeoutExpired:
        return {
            "available": True,
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": f"timeout after {timeout_seconds}s",
        }

    return {
        "available": True,
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
        "error": None if result.returncode == 0 else (result.stderr or result.stdout or "").strip()[:500],
    }


def _parse_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _probe_nvidia_smi() -> dict[str, Any]:
    query = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        timeout_seconds=3,
    )
    gpus: list[dict[str, Any]] = []
    driver_version = None

    if query["ok"]:
        for line in query["stdout"].splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            name = parts[0]
            memory_mib = _parse_float(parts[1])
            if len(parts) >= 3:
                driver_version = parts[2] or driver_version
            gpus.append({
                "name": name,
                "vram_gb": round(memory_mib / 1024, 2) if memory_mib else None,
            })

    fallback = None
    if not gpus:
        fallback = _run_command(["nvidia-smi", "-L"], timeout_seconds=3)
        if fallback["ok"]:
            for line in fallback["stdout"].splitlines():
                match = re.search(r"GPU\s+\d+:\s+(.+?)(?:\s+\(|$)", line)
                if match:
                    gpus.append({"name": match.group(1).strip(), "vram_gb": None})

    return {
        "available": bool(query["available"] or (fallback and fallback["available"])),
        "ok": bool(gpus),
        "driver_version": driver_version,
        "gpus": gpus,
        "error": None if gpus else query.get("error") or (fallback or {}).get("error"),
    }


def _read_rocm_version() -> str | None:
    for candidate in (
        "/opt/rocm/.info/version",
        "/opt/rocm/.info/version-dev",
        "/usr/share/rocm/.info/version",
    ):
        path = Path(candidate)
        try:
            if path.exists():
                value = path.read_text(encoding="utf-8", errors="replace").strip()
                if value:
                    return value
        except OSError:
            continue
    return None


def _parse_rocminfo(output: str) -> dict[str, Any]:
    gfx_arches: list[str] = []
    marketing_names: list[str] = []

    for line in output.splitlines():
        name_match = re.search(r"\bName:\s*(gfx[0-9a-zA-Z]+)", line)
        if name_match:
            arch = name_match.group(1)
            if arch not in gfx_arches:
                gfx_arches.append(arch)

        marketing_match = re.search(r"\bMarketing Name:\s*(.+)$", line)
        if marketing_match:
            name = marketing_match.group(1).strip()
            is_cpu_name = any(marker in name.lower() for marker in ("processor", "cpu", "apu"))
            if name and name != "N/A" and not is_cpu_name and name not in marketing_names:
                marketing_names.append(name)

    return {
        "gfx_arches": gfx_arches,
        "marketing_names": marketing_names,
    }


def _probe_rocm() -> dict[str, Any]:
    rocminfo = _run_command(["rocminfo"], timeout_seconds=5)
    parsed = _parse_rocminfo(rocminfo["stdout"]) if rocminfo["ok"] else {
        "gfx_arches": [],
        "marketing_names": [],
    }
    hipinfo = _run_command(["hipinfo"], timeout_seconds=5)

    return {
        "available": bool(rocminfo["available"] or hipinfo["available"] or _read_rocm_version()),
        "ok": bool(parsed["gfx_arches"] or hipinfo["ok"] or _read_rocm_version()),
        "version": _read_rocm_version(),
        "gfx_arches": parsed["gfx_arches"],
        "marketing_names": parsed["marketing_names"],
        "hipinfo_ok": bool(hipinfo["ok"]),
        "error": None if rocminfo["ok"] or hipinfo["ok"] else rocminfo.get("error") or hipinfo.get("error"),
    }


def detect_device_from_torch() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            if torch.version.hip is not None:
                return "amd_rocm"
            if torch.version.cuda is not None:
                return "nvidia_cuda"
    except ImportError:
        pass
    return "cpu"


def _probe_torch() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        return {
            "ok": False,
            "import_ok": False,
            "error": str(exc),
            "version": None,
            "cuda_version": None,
            "hip_version": None,
            "cuda_available": False,
            "device_count": 0,
            "device_name": None,
            "vram_gb": None,
            "tensor_probe_ok": False,
        }

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    device_name = None
    vram_gb = None
    tensor_probe_ok = False
    tensor_error = None

    if cuda_available:
        try:
            props = torch.cuda.get_device_properties(0)
            device_name = props.name
            vram_gb = round(float(props.total_memory) / (1024 ** 3), 2)
            probe = torch.ones((2, 2), device="cuda")
            tensor_probe_ok = float(probe.sum().cpu()) == 4.0
        except Exception as exc:  # pragma: no cover - hardware dependent
            tensor_error = str(exc)

    return {
        "ok": True,
        "import_ok": True,
        "error": tensor_error,
        "version": getattr(torch, "__version__", None),
        "cuda_version": getattr(torch.version, "cuda", None),
        "hip_version": getattr(torch.version, "hip", None),
        "cuda_available": cuda_available,
        "device_count": device_count,
        "device_name": device_name,
        "vram_gb": vram_gb,
        "tensor_probe_ok": tensor_probe_ok,
    }


def _probe_import(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {"ok": False, "version": None, "error": str(exc)}
    return {
        "ok": True,
        "version": getattr(module, "__version__", None),
        "error": None,
    }


def build_runtime_report(expected_device: str | None = None) -> dict[str, Any]:
    expected_error = None
    try:
        expected = normalise_provider_env(expected_device)
    except RuntimeError as exc:
        expected = None
        expected_error = str(exc)
    torch_probe = _probe_torch()
    detected = detect_device_from_torch()
    nvidia_probe = _probe_nvidia_smi()
    rocm_probe = _probe_rocm()
    axolotl_probe = _probe_import("axolotl.cli.train")
    bnb_probe = _probe_import("bitsandbytes") if detected == "nvidia_cuda" or expected == "nvidia_cuda" else {
        "ok": None,
        "version": None,
        "error": None,
    }

    issues: list[str] = []
    if expected_error:
        issues.append(expected_error)
    if expected and detected != expected:
        issues.append(f"expected {expected}, detected {detected}")
    if expected in GPU_BACKENDS and not torch_probe["cuda_available"]:
        issues.append(f"{expected} selected but torch cannot access a GPU")
    if expected == "amd_rocm" and torch_probe["hip_version"] is None:
        issues.append("AMD ROCm selected but PyTorch is not a ROCm build")
    if expected == "nvidia_cuda" and torch_probe["cuda_version"] is None:
        issues.append("NVIDIA CUDA selected but PyTorch is not a CUDA build")
    if detected in GPU_BACKENDS and not torch_probe["tensor_probe_ok"]:
        issues.append("GPU tensor probe failed")
    if not axolotl_probe["ok"]:
        issues.append(f"Axolotl import failed: {axolotl_probe['error']}")
    if (detected == "nvidia_cuda" or expected == "nvidia_cuda") and not bnb_probe["ok"]:
        issues.append(f"bitsandbytes import failed: {bnb_probe['error']}")

    gpu_name = torch_probe["device_name"]
    vram_gb = torch_probe["vram_gb"]
    if not gpu_name and nvidia_probe["gpus"]:
        gpu_name = nvidia_probe["gpus"][0]["name"]
        vram_gb = nvidia_probe["gpus"][0]["vram_gb"]
    if not gpu_name and rocm_probe["marketing_names"]:
        gpu_name = rocm_probe["marketing_names"][0]

    try:
        from platform_compat import detect_container_host_context, recommended_train_env

        host_ctx = detect_container_host_context()
        train_env = recommended_train_env(detected or expected or "cpu", host_ctx)
    except Exception as exc:  # pragma: no cover - optional module during partial deploys
        host_ctx = {"error": str(exc)}
        train_env = {}

    report = {
        "schema": "openreef.runtime.v1",
        "ready": not issues,
        "expected_device": expected,
        "expected_device_raw": expected_device,
        "detected_device": detected,
        "qlora_supported": detected == "nvidia_cuda" and bool(bnb_probe["ok"]),
        "gpu": {
            "name": gpu_name,
            "vram_gb": vram_gb,
            "gfx_arch": rocm_probe["gfx_arches"][0] if rocm_probe["gfx_arches"] else None,
        },
        "host": host_ctx,
        "recommended_train_env": train_env,
        "torch": torch_probe,
        "nvidia": nvidia_probe,
        "rocm": rocm_probe,
        "imports": {
            "axolotl": axolotl_probe,
            "bitsandbytes": bnb_probe,
        },
        "issues": issues,
        "env": {
            "OPENREEF_PROVIDER_ENV": os.environ.get("OPENREEF_PROVIDER_ENV"),
            "OPENREEF_HOST_OS": os.environ.get("OPENREEF_HOST_OS"),
            "OPENREEF_HOST_VIRT": os.environ.get("OPENREEF_HOST_VIRT"),
            "HSA_OVERRIDE_GFX_VERSION": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "TORCH_DISABLE_NATIVE_JIT": os.environ.get("TORCH_DISABLE_NATIVE_JIT"),
        },
    }
    return report


def verify_runtime(expected_device: str | None = None) -> dict[str, Any]:
    report = build_runtime_report(expected_device=expected_device)
    if not report["ready"]:
        raise RuntimeError("OpenReef runtime probe failed: " + "; ".join(report["issues"]))
    return report


def main() -> int:
    expected = os.environ.get("OPENREEF_PROVIDER_ENV")
    with contextlib.redirect_stdout(sys.stderr):
        report = build_runtime_report(expected_device=expected)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
