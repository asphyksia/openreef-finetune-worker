import os

from pause_guard import DEFAULT_PAUSE_FILE, pause_file_path, provider_pause_reason


def test_pause_marker_defaults_to_workspace(monkeypatch):
    monkeypatch.delenv("OPENREEF_PAUSE_FILE", raising=False)
    assert pause_file_path().as_posix() == DEFAULT_PAUSE_FILE


def test_pause_marker_is_persistent_operator_signal(tmp_path, monkeypatch):
    marker = tmp_path / ".openreef-paused"
    marker.write_text("maintenance window", encoding="utf-8")
    monkeypatch.setenv("OPENREEF_PAUSE_FILE", str(marker))

    assert provider_pause_reason() == "maintenance window"


def test_missing_pause_marker_keeps_provider_active(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENREEF_PAUSE_FILE", os.fspath(tmp_path / "missing"))
    assert provider_pause_reason() is None
