import os

from pause_guard import (
    DEFAULT_PAUSE_FILE,
    active_training_file_path,
    clear_training_active,
    mark_training_active,
    pause_file_path,
    provider_pause_reason,
)


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


def test_update_marker_drains_new_work(tmp_path, monkeypatch):
    marker = tmp_path / "updating"
    marker.write_text("qualified image update", encoding="utf-8")
    monkeypatch.setenv("OPENREEF_UPDATE_FILE", str(marker))
    monkeypatch.setenv("OPENREEF_PAUSE_FILE", str(tmp_path / "missing"))

    assert provider_pause_reason() == "qualified image update"


def test_active_training_marker_lifecycle(tmp_path, monkeypatch):
    marker = tmp_path / "training.active"
    monkeypatch.setenv("OPENREEF_ACTIVE_FILE", str(marker))

    assert mark_training_active("task-123") == marker
    assert "task=task-123" in marker.read_text(encoding="utf-8")
    assert active_training_file_path() == marker

    clear_training_active()
    assert not marker.exists()
