"""Post-job workspace/GPU hygiene before the next claim is admitted."""

from pathlib import Path

from workspace_hygiene import (
    inspect_workspace,
    purge_workspace,
    sanitize_for_next_job,
)

from test_finetune_worker import _load_worker


def _seed_job_workspace(root: Path) -> None:
    (root / "dataset.jsonl").write_text('{"messages":[]}\n', encoding="utf-8")
    (root / "dataset.raw.jsonl").write_text("raw\n", encoding="utf-8")
    (root / "unsloth.log").write_text("log\n", encoding="utf-8")
    (root / "openreef_adapter.zip").write_bytes(b"zip")
    (root / "prepared").mkdir()
    (root / "prepared" / "cache.bin").write_bytes(b"x")
    (root / "output").mkdir()
    (root / "output" / "adapter.safetensors").write_bytes(b"y")
    (root / "logs").mkdir()
    (root / "logs" / "worker.log").write_text("keep\n", encoding="utf-8")
    (root / "openreef-control").mkdir()
    (root / "openreef-control" / "paused").write_text("operator\n", encoding="utf-8")
    (root / "OPENREEF_LOGS.txt").write_text("pointer\n", encoding="utf-8")
    (root / "openreef_runtime.json").write_text("{}\n", encoding="utf-8")


def test_purge_removes_dataset_and_keeps_control(tmp_path):
    _seed_job_workspace(tmp_path)
    before = inspect_workspace(tmp_path)
    assert before["ok"] is False
    assert "dataset.jsonl" in before["leftovers"]
    assert "prepared/" in before["leftovers"]

    report = purge_workspace(tmp_path)
    assert report["ok"] is True
    assert not (tmp_path / "dataset.jsonl").exists()
    assert not (tmp_path / "dataset.raw.jsonl").exists()
    assert not (tmp_path / "prepared").exists()
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "openreef_adapter.zip").exists()
    assert (tmp_path / "logs" / "worker.log").is_file()
    assert (tmp_path / "openreef-control" / "paused").is_file()
    assert (tmp_path / "OPENREEF_LOGS.txt").is_file()
    assert inspect_workspace(tmp_path)["ok"] is True


def test_sanitize_reports_ok_on_clean_workspace(tmp_path):
    (tmp_path / "logs").mkdir()
    report = sanitize_for_next_job(tmp_path)
    assert report["ok"] is True
    assert report["workspace"]["ok"] is True
    assert report["gpu"]["ok"] is True


def test_admit_refuses_when_dataset_cannot_be_removed(monkeypatch, tmp_path):
    worker = _load_worker(monkeypatch)
    monkeypatch.setenv("OPENREEF_WORKSPACE", str(tmp_path))
    leftover = tmp_path / "dataset.jsonl"
    leftover.write_text("secret\n", encoding="utf-8")

    def fake_sanitize(work_dir=None):
        return {
            "ok": False,
            "workspace": {"ok": False, "leftovers": ["dataset.jsonl"], "removed": []},
            "gpu": {"ok": True, "allocated_bytes": 0},
        }

    monkeypatch.setattr(worker, "sanitize_for_next_job", fake_sanitize)
    assert worker._admit_job("task-a") == "dirty"
    assert worker._try_reserve_job("task-a") is True
    worker._release_job("task-a")


def test_release_if_clean_holds_reservation_when_dirty(monkeypatch, tmp_path):
    worker = _load_worker(monkeypatch)
    monkeypatch.setenv("OPENREEF_WORKSPACE", str(tmp_path))
    assert worker._try_reserve_job("task-a") is True

    monkeypatch.setattr(
        worker,
        "sanitize_for_next_job",
        lambda work_dir=None: {
            "ok": False,
            "workspace": {"ok": False, "leftovers": ["dataset.jsonl"], "removed": []},
            "gpu": {"ok": True, "allocated_bytes": 0},
        },
    )
    assert worker._release_job_if_clean("task-a") is False
    assert worker._try_reserve_job("task-b") is False
    worker._release_job("task-a")


def test_release_if_clean_opens_node_after_purge(monkeypatch, tmp_path):
    worker = _load_worker(monkeypatch)
    monkeypatch.setenv("OPENREEF_WORKSPACE", str(tmp_path))
    _seed_job_workspace(tmp_path)
    assert worker._try_reserve_job("task-a") is True
    assert worker._release_job_if_clean("task-a") is True
    assert not (tmp_path / "dataset.jsonl").exists()
    assert worker._try_reserve_job("task-b") is True
    worker._release_job("task-b")
