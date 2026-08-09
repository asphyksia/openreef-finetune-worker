"""Tests for the provider-side fine-tune worker contract."""

import hashlib
import importlib.util
import logging
import sys
import types
from pathlib import Path
from zipfile import ZipFile

import pytest


WORKER_PATH = Path(__file__).resolve().parents[1] / "worker.py"


def _load_worker(monkeypatch):
    """Load worker.py with a small ogpu.service stub."""
    ogpu_mod = types.ModuleType("ogpu")
    service_mod = types.ModuleType("ogpu.service")
    service_mod.init = lambda: lambda fn: fn
    service_mod.expose = lambda **_: lambda fn: fn
    service_mod.logger = logging.getLogger("test.openreef.worker")
    service_mod.exception = service_mod.logger.exception
    service_mod.start = lambda: None
    ogpu_mod.service = service_mod

    monkeypatch.setitem(sys.modules, "ogpu", ogpu_mod)
    monkeypatch.setitem(sys.modules, "ogpu.service", service_mod)
    monkeypatch.syspath_prepend(str(WORKER_PATH.parent))

    spec = importlib.util.spec_from_file_location("openreef_finetune_worker", WORKER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_provider_env_rejects_wrong_hardware(monkeypatch):
    worker = _load_worker(monkeypatch)
    monkeypatch.setenv("OPENREEF_PROVIDER_ENV", "nvidia_cuda")

    with pytest.raises(RuntimeError, match="expects nvidia_cuda"):
        worker._verify_expected_device("amd_rocm")


def test_provider_env_accepts_alias(monkeypatch):
    worker = _load_worker(monkeypatch)
    monkeypatch.setenv("OPENREEF_PROVIDER_ENV", "rocm")

    assert worker._verify_expected_device("amd_rocm") == "amd_rocm"


def test_package_adapter_output_includes_manifest_and_sha(monkeypatch, tmp_path):
    worker = _load_worker(monkeypatch)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    adapter = output_dir / "adapter_model.safetensors"
    adapter.write_bytes(b"0" * 2048)
    (output_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    package = worker._package_adapter_output(adapter)

    assert worker._sha256_file(package) == hashlib.sha256(package.read_bytes()).hexdigest()
    with ZipFile(package) as zf:
        names = set(zf.namelist())

    assert "adapter_model.safetensors" in names
    assert "adapter_config.json" in names
    assert "openreef_manifest.json" in names


def test_download_dataset_rejects_redirect(monkeypatch, tmp_path):
    worker = _load_worker(monkeypatch)
    monkeypatch.setattr(worker, "_validate_public_https_url", lambda url: None)

    class RedirectResponse:
        status_code = 302
        headers = {"location": "https://example.com/next"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            raise AssertionError("redirect should be rejected before raise_for_status")

    def fake_get(*args, **kwargs):
        assert kwargs["allow_redirects"] is False
        assert kwargs["stream"] is True
        return RedirectResponse()

    monkeypatch.setattr(worker.requests, "get", fake_get)

    with pytest.raises(ValueError, match="redirects are not allowed"):
        worker._download_dataset("https://example.com/dataset.jsonl", tmp_path / "dataset.jsonl")


def test_upload_via_presigned_url_sends_zip_content_type(monkeypatch, tmp_path):
    worker = _load_worker(monkeypatch)
    package = tmp_path / "openreef_adapter.zip"
    package.write_bytes(b"0" * 2048)

    class PutResponse:
        status_code = 200

        def raise_for_status(self):
            return None

    def fake_put(*args, **kwargs):
        assert kwargs["headers"] == {"Content-Type": "application/zip"}
        assert kwargs["allow_redirects"] is False
        return PutResponse()

    monkeypatch.setattr(worker.requests, "put", fake_put)

    worker._upload_via_presigned_url(package, "https://example.com/upload")
