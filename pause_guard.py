"""Persistent provider pause marker for recreated OpenReef workers.

The marker lives on the provider's ``/data`` bind mount, which is mounted as
``/workspace`` in the worker image. It therefore survives container recreation
without putting provider state in the image.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_PAUSE_FILE = "/workspace/openreef-control/paused"
DEFAULT_UPDATE_FILE = "/workspace/openreef-control/updating"
DEFAULT_ACTIVE_FILE = "/workspace/openreef-control/training.active"


def pause_file_path() -> Path:
    """Return the operator-controlled pause marker path."""
    return Path(os.environ.get("OPENREEF_PAUSE_FILE", DEFAULT_PAUSE_FILE)).expanduser()


def update_file_path() -> Path:
    return Path(os.environ.get("OPENREEF_UPDATE_FILE", DEFAULT_UPDATE_FILE)).expanduser()


def active_training_file_path() -> Path:
    return Path(os.environ.get("OPENREEF_ACTIVE_FILE", DEFAULT_ACTIVE_FILE)).expanduser()


def provider_pause_reason() -> str | None:
    """Return the marker reason, or ``None`` when the provider is active."""
    for marker, fallback in (
        (pause_file_path(), "operator pause marker is present"),
        (update_file_path(), "provider update is draining new work"),
    ):
        if not marker.is_file():
            continue
        try:
            reason = marker.read_text(encoding="utf-8").strip()
        except OSError:
            reason = fallback
        return (reason or fallback)[:240]
    return None


def mark_training_active(task: str | None = None) -> Path:
    marker = active_training_file_path()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"pid={os.getpid()} task={task or 'unknown'}\n", encoding="utf-8")
    return marker


def clear_training_active() -> None:
    try:
        active_training_file_path().unlink(missing_ok=True)
    except OSError:
        pass
