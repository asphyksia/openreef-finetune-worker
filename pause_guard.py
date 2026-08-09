"""Persistent provider pause marker for recreated OpenReef workers.

The marker lives on the provider's ``/data`` bind mount, which is mounted as
``/workspace`` in the worker image. It therefore survives container recreation
without putting provider state in the image.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_PAUSE_FILE = "/workspace/openreef-control/paused"


def pause_file_path() -> Path:
    """Return the operator-controlled pause marker path."""
    return Path(os.environ.get("OPENREEF_PAUSE_FILE", DEFAULT_PAUSE_FILE)).expanduser()


def provider_pause_reason() -> str | None:
    """Return the marker reason, or ``None`` when the provider is active."""
    marker = pause_file_path()
    if not marker.is_file():
        return None
    try:
        reason = marker.read_text(encoding="utf-8").strip()
    except OSError:
        reason = "operator pause marker is present"
    return (reason or "operator pause marker is present")[:240]
