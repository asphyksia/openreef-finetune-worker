"""Protect interactive displays when training shares the GPU.

Consumer workstations often run the compositor (KDE/GNOME/Wayland) and the
training process on the same discrete GPU. Unbounded VRAM use freezes or
crashes the desktop. Dedicated headless nodes should not pay that tax.

Policy (env-driven, no hardware-specific defaults):

- ``OPENREEF_PROTECT_DISPLAY``:
    ``auto`` (default) → protect when a graphical session is detected
    ``1`` / ``true``   → always protect
    ``0`` / ``false``  → never protect (dedicated training GPU)

- ``OPENREEF_DISPLAY_VRAM_RESERVE_MB``:
    Absolute VRAM reserved for the display stack (default 2048 MiB).
    Converted to a process memory fraction from probed total VRAM.

- ``OPENREEF_VRAM_FRACTION``:
    Optional explicit fraction (0 disables). Overrides reserve math when set.

Applies to both AMD ROCm and NVIDIA CUDA (``torch.cuda`` / HIP share the API).
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Mapping

logger = logging.getLogger(__name__)

# Absolute headroom for compositor + desktop chrome on shared GPUs.
DEFAULT_DISPLAY_RESERVE_MB = 2048
# Never starve training below this even on small cards (4–6 GB).
MIN_TRAINING_FRACTION = 0.50
# Never claim more than this even when reserve is tiny relative to huge GPUs.
MAX_TRAINING_FRACTION = 0.95
# If we cannot probe total VRAM, use a conservative fraction.
FALLBACK_FRACTION_WHEN_UNKNOWN = 0.85


@dataclass(frozen=True)
class DisplayGpuGuard:
    """Resolved guard decision for one training launch."""

    protect: bool
    reason: str
    vram_fraction: float | None  # None → no process memory cap
    reserve_mb: int
    total_vram_bytes: int | None
    device_type: str

    @property
    def enabled(self) -> bool:
        return self.protect and self.vram_fraction is not None and 0 < self.vram_fraction < 1.0


def graphical_session_present(env: Mapping[str, str] | None = None) -> bool:
    """True when the process environment looks like an interactive desktop."""
    e = env if env is not None else os.environ
    if e.get("WAYLAND_DISPLAY"):
        return True
    display = (e.get("DISPLAY") or "").strip()
    # ":0", ":1", "localhost:10.0" — ignore empty / "invalid"
    if display and display.lower() not in ("none", "null"):
        return True
    # systemd user graphical target hint (optional)
    if e.get("XDG_SESSION_TYPE", "").lower() in ("wayland", "x11", "mir"):
        return True
    return False


def should_protect_display(env: Mapping[str, str] | None = None) -> tuple[bool, str]:
    """Return (protect, reason) from OPENREEF_PROTECT_DISPLAY + session detection."""
    e = env if env is not None else os.environ
    raw = (e.get("OPENREEF_PROTECT_DISPLAY") or "auto").strip().lower()
    if raw in ("0", "false", "no", "off", "never"):
        return False, "OPENREEF_PROTECT_DISPLAY=off"
    if raw in ("1", "true", "yes", "on", "always"):
        return True, "OPENREEF_PROTECT_DISPLAY=on"
    # auto
    if graphical_session_present(e):
        return True, "auto:graphical_session_detected"
    return False, "auto:no_graphical_session"


def _parse_positive_float(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fraction_from_reserve(
    total_vram_bytes: int,
    reserve_mb: int = DEFAULT_DISPLAY_RESERVE_MB,
    *,
    min_fraction: float = MIN_TRAINING_FRACTION,
    max_fraction: float = MAX_TRAINING_FRACTION,
) -> float:
    """Map absolute display reserve to a torch memory fraction."""
    if total_vram_bytes <= 0:
        return FALLBACK_FRACTION_WHEN_UNKNOWN
    reserve_bytes = max(0, int(reserve_mb)) * 1024 * 1024
    usable = max(0, total_vram_bytes - reserve_bytes)
    frac = usable / float(total_vram_bytes)
    return max(min_fraction, min(max_fraction, frac))


def probe_total_vram_bytes(device_type: str = "") -> int | None:
    """Best-effort total VRAM probe without importing torch (fast, subprocess)."""
    device_type = (device_type or "").lower()
    probes: list[tuple[str, list[str]]] = []
    if device_type in ("", "amd_rocm", "rocm", "hip"):
        probes.append(("rocm-smi", ["rocm-smi", "--showmeminfo", "vram"]))
        probes.append(("rocm-smi-opt", ["/opt/rocm/bin/rocm-smi", "--showmeminfo", "vram"]))
    if device_type in ("", "nvidia_cuda", "cuda", "nvidia"):
        probes.append(
            (
                "nvidia-smi",
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
            )
        )

    for name, cmd in probes:
        try:
            out = subprocess.check_output(
                cmd,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            continue

        if "nvidia-smi" in name:
            # "16384" or "16384 MiB"
            for line in out.splitlines():
                m = re.search(r"(\d+(?:\.\d+)?)", line)
                if m:
                    # nvidia-smi nounits is MiB
                    return int(float(m.group(1)) * 1024 * 1024)
        else:
            # "VRAM Total Memory (B): 17095983104"
            m = re.search(r"Total Memory \(B\):\s*(\d+)", out, re.I)
            if m:
                return int(m.group(1))
            m = re.search(r"total memory[^:]*:\s*(\d+)", out, re.I)
            if m:
                return int(m.group(1))
    return None


def resolve_display_gpu_guard(
    device_type: str,
    env: Mapping[str, str] | None = None,
    *,
    total_vram_bytes: int | None = None,
) -> DisplayGpuGuard:
    """Resolve whether/how to cap training GPU memory for display safety."""
    e = env if env is not None else os.environ
    device_type = (device_type or "cpu").lower()

    if device_type == "cpu":
        return DisplayGpuGuard(
            protect=False,
            reason="cpu_device",
            vram_fraction=None,
            reserve_mb=0,
            total_vram_bytes=None,
            device_type=device_type,
        )

    protect, reason = should_protect_display(e)

    reserve_raw = e.get("OPENREEF_DISPLAY_VRAM_RESERVE_MB")
    try:
        reserve_mb = int(reserve_raw) if reserve_raw not in (None, "") else DEFAULT_DISPLAY_RESERVE_MB
    except ValueError:
        reserve_mb = DEFAULT_DISPLAY_RESERVE_MB
    reserve_mb = max(0, reserve_mb)

    # Explicit fraction always wins when set (including 0 = disable cap while protect may still log).
    explicit = _parse_positive_float(e.get("OPENREEF_VRAM_FRACTION"))
    # Back-compat with earlier local-only knob
    if explicit is None:
        explicit = _parse_positive_float(e.get("OPENREEF_ROCM_VRAM_FRACTION"))

    total = total_vram_bytes
    if total is None and protect:
        total = probe_total_vram_bytes(device_type)

    vram_fraction: float | None = None
    if explicit is not None:
        if explicit <= 0:
            vram_fraction = None
            reason = f"{reason};OPENREEF_VRAM_FRACTION=0"
        elif explicit >= 1.0:
            vram_fraction = MAX_TRAINING_FRACTION
            reason = f"{reason};OPENREEF_VRAM_FRACTION={explicit}"
        else:
            vram_fraction = explicit
            reason = f"{reason};OPENREEF_VRAM_FRACTION={explicit}"
    elif protect:
        if total and total > 0:
            vram_fraction = fraction_from_reserve(total, reserve_mb)
            reason = (
                f"{reason};reserve_mb={reserve_mb};"
                f"total_gb={total / (1024**3):.1f};fraction={vram_fraction:.3f}"
            )
        else:
            vram_fraction = FALLBACK_FRACTION_WHEN_UNKNOWN
            reason = f"{reason};vram_probe_failed;fallback_fraction={vram_fraction}"

    return DisplayGpuGuard(
        protect=protect,
        reason=reason,
        vram_fraction=vram_fraction,
        reserve_mb=reserve_mb,
        total_vram_bytes=total,
        device_type=device_type,
    )


def apply_guard_to_env(env: dict[str, str], guard: DisplayGpuGuard) -> dict[str, str]:
    """Mutate a subprocess env dict with allocator knobs + fraction for the wrapper."""
    if not guard.enabled or guard.vram_fraction is None:
        # Clear sticky caps so dedicated nodes never inherit a stale fraction.
        env.pop("OPENREEF_VRAM_FRACTION", None)
        env.pop("OPENREEF_ROCM_VRAM_FRACTION", None)
        return env

    frac = f"{guard.vram_fraction:.4f}".rstrip("0").rstrip(".")
    env["OPENREEF_VRAM_FRACTION"] = frac
    # Keep legacy name in sync for older wrappers / logs
    env["OPENREEF_ROCM_VRAM_FRACTION"] = frac

    if guard.device_type == "amd_rocm":
        env.setdefault(
            "PYTORCH_HIP_ALLOC_CONF",
            "garbage_collection_threshold:0.7,max_split_size_mb:128",
        )
    elif guard.device_type == "nvidia_cuda":
        env.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            "garbage_collection_threshold:0.7,max_split_size_mb:128,expandable_segments:True",
        )
    return env


def write_vram_cap_wrapper(
    work_path,
    *,
    training_python: str,
    config_path: str,
) -> str:
    """Write a small bootstrap that caps process GPU memory then runs Axolotl.

    Returns path to the wrapper script.
    """
    work_path = __import__("pathlib").Path(work_path)
    wrapper = work_path / "run_train_with_vram_cap.py"
    wrapper.write_text(
        "import os\n"
        "import runpy\n"
        "import sys\n"
        "\n"
        "def _cap_vram() -> None:\n"
        "    '''Cap torch HIP/CUDA caching allocator so the display keeps VRAM.'''\n"
        "    try:\n"
        "        raw = os.environ.get('OPENREEF_VRAM_FRACTION') or os.environ.get('OPENREEF_ROCM_VRAM_FRACTION') or '0'\n"
        "        frac = float(raw or '0')\n"
        "        if not (0.0 < frac < 1.0):\n"
        "            return\n"
        "        import torch\n"
        "        if not torch.cuda.is_available():\n"
        "            return\n"
        "        # Works for CUDA and ROCm (HIP is exposed via torch.cuda).\n"
        "        local_rank = int(os.environ.get('LOCAL_RANK') or '0')\n"
        "        torch.cuda.set_device(local_rank)\n"
        "        torch.cuda.set_per_process_memory_fraction(frac, local_rank)\n"
        "        print(\n"
        "            f'[openreef] GPU memory fraction cap: {frac:.3f} '\n"
        "            f'(rank={local_rank} device={torch.cuda.get_device_name(local_rank)!r})',\n"
        "            flush=True,\n"
        "        )\n"
        "    except Exception as exc:\n"
        "        print(f'[openreef] GPU memory cap skipped: {exc}', flush=True)\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    _cap_vram()\n"
        f"    sys.argv = {[training_python, '-m', 'axolotl.cli.train', str(config_path)]!r}\n"
        "    runpy.run_module('axolotl.cli.train', run_name='__main__', alter_sys=True)\n",
        encoding="utf-8",
    )
    return str(wrapper)
