"""Tests for the provider-side fine-tune worker contract."""

import hashlib
import importlib.util
import logging
import sys
import types
from pathlib import Path
from zipfile import ZipFile

import pytest
import yaml


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


def test_classify_train_failure_gpu_page_fault(monkeypatch):
    worker = _load_worker(monkeypatch)
    fc, msg, retryable = worker.classify_train_failure(
        error_text=(
            "Memory access fault by GPU node-1 (Agent handle: 0x4a32ea60) "
            "on address 0x75ece8600000. Reason: Page not present or supervisor privilege."
        ),
        returncode=-6,
        device="amd_rocm",
    )
    assert fc == "gpu_fault"
    assert retryable is False
    assert msg.startswith("gpu_fault:")


def test_classify_train_failure_sigabrt_on_amd_without_markers(monkeypatch):
    worker = _load_worker(monkeypatch)
    fc, msg, retryable = worker.classify_train_failure(
        error_text="tokens/total': 8710, 'epoch': '0.03988'}",
        returncode=-6,
        device="amd_rocm",
    )
    assert fc == "gpu_fault"
    assert retryable is False
    assert "dataset" in msg.lower()


def test_classify_train_failure_ordinary_error_retryable(monkeypatch):
    worker = _load_worker(monkeypatch)
    fc, msg, retryable = worker.classify_train_failure(
        error_text="RuntimeError: loss is NaN",
        returncode=1,
        device="amd_rocm",
    )
    assert fc == "train_error"
    assert retryable is True
    assert "NaN" in msg


def test_parse_axolotl_train_loss_accepts_quoted_metric(monkeypatch, tmp_path):
    worker = _load_worker(monkeypatch)
    log = tmp_path / "axolotl.log"
    log.write_text(
        "{'loss': '0.125', 'epoch': '1.0'}\n"
        "{'loss': '0.02525', 'epoch': '1.993'}\n",
        encoding="utf-8",
    )

    assert worker._parse_axolotl_train_loss(log) == pytest.approx(0.02525)


def test_parse_axolotl_train_loss_prefers_aggregate_metric(monkeypatch, tmp_path):
    worker = _load_worker(monkeypatch)
    log = tmp_path / "axolotl.log"
    log.write_text(
        "{'loss': '0.000708', 'epoch': '3.985'}\n"
        "{'train_runtime': '1243', 'train_loss': '0.3243', 'epoch': '3.997'}\n",
        encoding="utf-8",
    )

    assert worker._parse_axolotl_train_loss(log) == pytest.approx(0.3243)


def test_select_train_engine_matrix(monkeypatch):
    worker = _load_worker(monkeypatch)
    fake = types.ModuleType("unsloth_train")
    fake.unsloth_available = lambda: True
    monkeypatch.setitem(sys.modules, "unsloth_train", fake)
    monkeypatch.delenv("OPENREEF_TRAIN_ENGINE", raising=False)

    monkeypatch.setattr(worker, "_gpu_count", lambda: 1)
    assert worker._select_train_engine("nvidia_cuda") == "unsloth"
    assert worker._select_train_engine("amd_rocm") == "axolotl"

    monkeypatch.setattr(worker, "_gpu_count", lambda: 2)
    assert worker._select_train_engine("nvidia_cuda") == "axolotl"
    assert worker._select_train_engine("amd_rocm") == "axolotl"


def test_custom_engine_is_single_gpu_nvidia_only(monkeypatch):
    worker = _load_worker(monkeypatch)
    fake = types.ModuleType("unsloth_train")
    fake.unsloth_available = lambda: True
    monkeypatch.setitem(sys.modules, "unsloth_train", fake)
    monkeypatch.delenv("OPENREEF_TRAIN_ENGINE", raising=False)

    monkeypatch.setattr(worker, "_gpu_count", lambda: 1)
    assert worker._select_train_engine("nvidia_cuda", "custom") == "unsloth"
    with pytest.raises(RuntimeError, match="one NVIDIA GPU"):
        worker._select_train_engine("amd_rocm", "custom")

    monkeypatch.setattr(worker, "_gpu_count", lambda: 2)
    with pytest.raises(RuntimeError, match="one NVIDIA GPU"):
        worker._select_train_engine("nvidia_cuda", "custom")


def test_custom_hyperparameters_are_resolved_without_floating_fields():
    from training_config import resolve_training_hyperparams

    custom = {
        "num_epochs": 4,
        "learning_rate": 5e-5,
        "lora_r": 64,
        "lora_alpha": 128,
        "sequence_len": 4096,
        "weight_decay": 0.03,
        "save_steps": 20,
        "save_total_limit": 2,
    }
    resolved = resolve_training_hyperparams("custom", custom_config=custom)

    assert resolved["num_epochs"] == 4
    assert resolved["learning_rate"] == pytest.approx(5e-5)
    assert resolved["lora_r"] == 64
    assert resolved["sequence_len"] == 4096
    assert resolved["weight_decay"] == pytest.approx(0.03)
    assert resolved["save_steps"] == 20
    assert resolved["save_total_limit"] == 2

    with pytest.raises(ValueError, match="Unsupported Custom fields"):
        resolve_training_hyperparams("custom", custom_config={"unknown": 1})


def test_axolotl_train_command_single_and_multi_gpu(monkeypatch):
    worker = _load_worker(monkeypatch)

    assert worker._axolotl_train_command(
        "python", "/workspace/config.yml", gpu_count=1
    ) == ["python", "-m", "axolotl.cli.train", "/workspace/config.yml"]

    assert worker._axolotl_train_command(
        "python",
        "/workspace/config.yml",
        gpu_count=2,
        wrapper_path="/workspace/cap.py",
    ) == [
        "python",
        "-m",
        "accelerate.commands.launch",
        "--multi_gpu",
        "--num_processes",
        "2",
        "/workspace/cap.py",
    ]


def test_gfx1200_balanced_uses_safe_rank_without_changing_other_profiles(monkeypatch):
    worker = _load_worker(monkeypatch)
    balanced = {
        "lora_r": 32,
        "lora_alpha": 64,
        "num_epochs": 2,
        "learning_rate": 1e-4,
    }

    changed = worker._apply_rdna4_balanced_safety(
        balanced,
        device="amd_rocm",
        gfx_target="gfx1200:sramecc+:xnack-",
        preset="balanced",
    )

    assert changed is True
    assert balanced == {
        "lora_r": 64,
        "lora_alpha": 128,
        "num_epochs": 2,
        "learning_rate": 1e-4,
    }

    for device, gfx_target, preset in (
        ("nvidia_cuda", "gfx1200", "balanced"),
        ("amd_rocm", "gfx1100", "balanced"),
        ("amd_rocm", "gfx1200", "fast"),
        ("amd_rocm", "gfx1200", "custom"),
    ):
        config = {"lora_r": 32, "lora_alpha": 64}
        assert worker._apply_rdna4_balanced_safety(
            config,
            device=device,
            gfx_target=gfx_target,
            preset=preset,
        ) is False
        assert config == {"lora_r": 32, "lora_alpha": 64}


def test_detect_gfx_target_prefers_explicit_hint(monkeypatch):
    worker = _load_worker(monkeypatch)
    monkeypatch.setenv("OPENREEF_GFX_TARGET", "gfx1200:sramecc+:xnack-")

    assert worker._detect_gfx_target() == "gfx1200"


def test_build_config_applies_gfx1200_safety_after_claim_overrides(
    monkeypatch, tmp_path
):
    worker = _load_worker(monkeypatch)
    monkeypatch.setattr(worker, "_detect_device", lambda: "amd_rocm")
    monkeypatch.setattr(worker, "_detect_vram_gb", lambda: 15.92)
    monkeypatch.setattr(worker, "_detect_gfx_target", lambda: "gfx1200")
    monkeypatch.setattr(worker, "_probe_tokenizer_has_chat_template", lambda _: True)
    job = worker.FineTuneInput(
        base_model="Qwen/Qwen3-1.7B",
        preset="balanced",
        adapter="lora",
        lora_r=32,
        lora_alpha=64,
        val_set_size=0.05,
    )

    path = worker._build_axolotl_config(job, tmp_path, num_dataset_rows=1056)
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    assert config["lora_r"] == 64
    assert config["lora_alpha"] == 128
    assert config["num_epochs"] == 2
    assert config["learning_rate"] == 1e-4


def test_provider_env_rejects_wrong_hardware(monkeypatch):
    worker = _load_worker(monkeypatch)
    monkeypatch.setenv("OPENREEF_PROVIDER_ENV", "nvidia_cuda")

    with pytest.raises(RuntimeError, match="expects nvidia_cuda"):
        worker._verify_expected_device("amd_rocm")


def test_provider_env_accepts_alias(monkeypatch):
    worker = _load_worker(monkeypatch)
    monkeypatch.setenv("OPENREEF_PROVIDER_ENV", "rocm")

    assert worker._verify_expected_device("amd_rocm") == "amd_rocm"


def test_main_keeps_service_alive_while_provider_is_paused(
    monkeypatch, tmp_path, capsys
):
    worker = _load_worker(monkeypatch)
    pause_file = tmp_path / "paused"
    pause_file.write_text("operator maintenance", encoding="utf-8")
    monkeypatch.setenv("OPENREEF_PAUSE_FILE", str(pause_file))
    calls = []
    monkeypatch.setattr(worker, "clear_training_active", lambda: calls.append("clear"))
    monkeypatch.setattr(worker, "_reset_progress", lambda: calls.append("reset"))
    monkeypatch.setattr(
        worker,
        "_install_ogpu_task_address_env_patch",
        lambda: calls.append("patch"),
    )
    monkeypatch.setattr(worker.ogpu.service, "start", lambda: calls.append("start"))

    worker.main()

    assert calls == ["clear", "reset", "patch", "start"]
    assert "PHASE=provider_paused reason=operator maintenance" in capsys.readouterr().err


def test_finetune_rejects_paused_provider_before_marking_training_active(
    monkeypatch, tmp_path
):
    worker = _load_worker(monkeypatch)
    pause_file = tmp_path / "paused"
    pause_file.write_text("operator maintenance", encoding="utf-8")
    monkeypatch.setenv("OPENREEF_PAUSE_FILE", str(pause_file))
    monkeypatch.setattr(worker, "_setup_workspace_file_logging", lambda: None)
    monkeypatch.setattr(
        worker,
        "mark_training_active",
        lambda *_: pytest.fail("paused provider must not mark training active"),
    )

    result = worker.finetune(worker.FineTuneInput())

    assert result.status == "failed"
    assert result.error == "OpenReef provider is paused"


def test_direct_input_is_rejected_without_explicit_local_escape_hatch(monkeypatch):
    worker = _load_worker(monkeypatch)
    monkeypatch.delenv("OPENREEF_ALLOW_DIRECT_INPUT", raising=False)
    monkeypatch.setattr(worker, "_setup_workspace_file_logging", lambda: None)
    monkeypatch.setattr(
        worker,
        "mark_training_active",
        lambda *_: pytest.fail("unclaimed input must fail before marking active"),
    )

    result = worker.finetune(
        worker.FineTuneInput(
            base_model="attacker/model",
            dataset_url="https://example.com/dataset.jsonl",
        )
    )

    assert result.status == "failed"
    assert result.error == "OpenReef claim required"


def test_direct_input_escape_hatch_is_explicit(monkeypatch):
    worker = _load_worker(monkeypatch)
    monkeypatch.delenv("OPENREEF_ALLOW_DIRECT_INPUT", raising=False)
    assert worker._direct_input_allowed() is False

    monkeypatch.setenv("OPENREEF_ALLOW_DIRECT_INPUT", "true")
    assert worker._direct_input_allowed() is True


def test_single_job_reservation_is_owner_safe(monkeypatch):
    worker = _load_worker(monkeypatch)

    assert worker._try_reserve_job("task-one") is True
    assert worker._try_reserve_job("task-two") is False

    worker._release_job("task-two")
    assert worker._try_reserve_job("task-three") is False

    worker._release_job("task-one")
    assert worker._try_reserve_job("task-three") is True
    worker._release_job("task-three")


def test_tokenizer_probe_disables_remote_code(monkeypatch):
    worker = _load_worker(monkeypatch)
    calls = []

    class Tokenizer:
        chat_template = "{{ messages }}"

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model, **kwargs):
            calls.append((model, kwargs))
            return Tokenizer()

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    assert worker._probe_tokenizer_has_chat_template("trusted/model") is True
    assert calls == [
        (
            "trusted/model",
            {"trust_remote_code": False, "use_fast": True},
        )
    ]


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


def test_read_training_metrics_returns_bounded_summary(monkeypatch, tmp_path):
    worker = _load_worker(monkeypatch)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "openreef_training_metrics.json").write_text(
        """{
          "train_loss": 0.42,
          "eval_loss": null,
          "global_step": 120,
          "epoch": 2.0,
          "best_checkpoint": "checkpoint-100",
          "log_history": [{"loss": 9}],
          "unknown": "ignored"
        }""",
        encoding="utf-8",
    )

    assert worker._read_training_metrics(output_dir) == {
        "train_loss": 0.42,
        "global_step": 120,
        "epoch": 2.0,
        "best_checkpoint": "checkpoint-100",
    }


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
