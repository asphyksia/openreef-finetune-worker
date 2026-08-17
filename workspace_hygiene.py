"""Purge and verify per-job workspace/GPU leftovers before the next claim.

The provider volume is persistent. A finished job must not leave the customer's
dataset, tokenizer cache or adapter output on disk, and must not keep a large
GPU allocation, or the next claim can train on stale data or OOM.

Logs and operator control markers are kept. This module never deletes
``openreef-control/``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_WORKSPACE = "/workspace"
KEEP_DIR_NAMES = frozenset({"logs", "openreef-control", ".openreef-control"})
KEEP_FILE_NAMES = frozenset(
    {
        "OPENREEF_LOGS.txt",
        "openreef_runtime.json",
        "paused",
        "updating",
        "training.active",
    }
)
TRANSIENT_DIR_NAMES = frozenset({"prepared", "output", "checkpoints"})
TRANSIENT_FILE_NAMES = frozenset(
    {
        "dataset.jsonl",
        "unsloth.log",
        "axolotl.log",
        "openreef_adapter.zip",
    }
)
TRANSIENT_FILE_PREFIXES = ("dataset.raw.",)
# CUDA/HIP context leftovers after empty_cache are normal; leftover tensors are not.
MAX_ALLOCATED_BYTES = 256 * 1024 * 1024


def workspace_dir() -> Path:
    return Path(os.environ.get("OPENREEF_WORKSPACE", DEFAULT_WORKSPACE)).expanduser()


def _is_transient_file(path: Path) -> bool:
    name = path.name
    if name in KEEP_FILE_NAMES:
        return False
    if name in TRANSIENT_FILE_NAMES:
        return True
    if any(name.startswith(prefix) for prefix in TRANSIENT_FILE_PREFIXES):
        return True
    return path.suffix == ".zip"


def inspect_workspace(work_dir: Path | None = None) -> dict[str, Any]:
    root = Path(work_dir or workspace_dir())
    leftovers: list[str] = []
    if not root.is_dir():
        return {"ok": True, "leftovers": leftovers, "root": str(root)}
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if child.name in KEEP_DIR_NAMES or child.name in KEEP_FILE_NAMES:
            continue
        if child.is_dir() and child.name in TRANSIENT_DIR_NAMES:
            leftovers.append(f"{child.name}/")
            continue
        if child.is_file() and _is_transient_file(child):
            leftovers.append(child.name)
    return {"ok": not leftovers, "leftovers": leftovers, "root": str(root)}


def purge_workspace(work_dir: Path | None = None) -> dict[str, Any]:
    import shutil

    root = Path(work_dir or workspace_dir())
    removed: list[str] = []
    errors: list[str] = []
    if root.is_dir():
        for child in list(root.iterdir()):
            if child.name in KEEP_DIR_NAMES or child.name in KEEP_FILE_NAMES:
                continue
            transient_dir = child.is_dir() and child.name in TRANSIENT_DIR_NAMES
            transient_file = child.is_file() and _is_transient_file(child)
            if not (transient_dir or transient_file):
                continue
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                    removed.append(f"{child.name}/")
                else:
                    child.unlink()
                    removed.append(child.name)
            except OSError as exc:
                errors.append(f"{child.name}:{exc}")
    inspection = inspect_workspace(root)
    return {
        "ok": inspection["ok"] and not errors,
        "removed": removed,
        "errors": errors,
        **inspection,
    }


def inspect_gpu_memory() -> dict[str, Any]:
    allocated = 0
    reserved = 0
    available = False
    backend = None
    try:
        import torch

        if not torch.cuda.is_available():
            return {
                "ok": True,
                "available": False,
                "allocated_bytes": 0,
                "reserved_bytes": 0,
                "backend": None,
            }
        available = True
        backend = "hip" if getattr(torch.version, "hip", None) else "cuda"
        allocated = int(torch.cuda.memory_allocated())
        reserved = int(torch.cuda.memory_reserved())
    except Exception as exc:
        return {
            "ok": True,
            "available": False,
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "backend": None,
            "error": str(exc)[:200],
        }
    return {
        "ok": allocated <= MAX_ALLOCATED_BYTES,
        "available": available,
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "backend": backend,
        "limit_bytes": MAX_ALLOCATED_BYTES,
    }


def purge_gpu_memory() -> dict[str, Any]:
    try:
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
    except Exception as exc:
        report = inspect_gpu_memory()
        report["purge_error"] = str(exc)[:200]
        return report
    return inspect_gpu_memory()


def sanitize_for_next_job(work_dir: Path | None = None) -> dict[str, Any]:
    """Purge job leftovers and report whether the node may admit another claim."""
    workspace = purge_workspace(work_dir)
    gpu = purge_gpu_memory()
    ok = bool(workspace.get("ok")) and bool(gpu.get("ok"))
    return {"ok": ok, "workspace": workspace, "gpu": gpu}
