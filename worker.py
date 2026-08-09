"""OGPU Source worker — runs inside the provider's container.

Receives a fine-tuning task from OGPU, detects the available hardware
(NVIDIA CUDA or AMD ROCm), runs Axolotl training with hardware-specific
configuration, and returns the trained adapter via presigned PUT URL or base64.

Hardware detection is automatic — the same worker.py works on both
NVIDIA and AMD providers without modification.
"""

import base64
import csv
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import ogpu.service
import requests
import yaml
from pydantic import BaseModel

from runtime_probe import (
    build_runtime_report,
    detect_device_from_torch,
    normalise_provider_env,
    verify_runtime,
)
from training_config import build_axolotl_config_dict
from pause_guard import provider_pause_reason

# Host-visible logs: provider-app bind-mounts host data dir → /data → /workspace.
# Windows/Linux providers can open these files even when `docker logs` is empty.
_FILE_LOG_CONFIGURED = False
_PHASE_LOGGER = logging.getLogger("openreef.phase")

# Shared pipeline progress for claim heartbeats (Lab % is not time-only).
_PROGRESS_LOCK = None  # set lazily to avoid import-time threading issues
_PROGRESS_STATE: dict[str, object] = {
    "phase": None,
    "progress_pct": None,
    "detail": None,
    "step": None,
    "max_steps": None,
    "train_log": None,
}

# Floors mirror backend app.services.job_progress.PHASE_PROGRESS
_PHASE_PROGRESS_FLOOR: dict[str, int] = {
    "heartbeat_loop_started": 10,
    "claim_start": 10,
    "claim_ok": 12,
    "device_ok": 14,
    "dataset_start": 16,
    "dataset_ok": 24,
    "workspace_reset": 25,
    "config_start": 26,
    "config_ok": 28,
    "host_compat": 29,
    "display_guard": 30,
    "model_download": 35,
    "train_start": 50,
    "train": 50,
    "train_exit": 90,
    "artifact_start": 92,
    "upload_start": 95,
    "upload_ok": 97,
    "job_done": 98,
}

_PHASE_LABELS: dict[str, str] = {
    "claim_ok": "Claimed; preparing workspace",
    "device_ok": "GPU ready",
    "dataset_start": "Downloading dataset…",
    "dataset_ok": "Dataset ready",
    "config_ok": "Training config ready",
    "train_start": "Training started",
    "train": "Training…",
    "train_exit": "Training finished",
    "artifact_start": "Packaging adapter…",
    "upload_start": "Uploading adapter…",
    "upload_ok": "Upload complete",
    "job_done": "Job finishing…",
}


def _progress_lock():
    global _PROGRESS_LOCK
    if _PROGRESS_LOCK is None:
        import threading

        _PROGRESS_LOCK = threading.Lock()
    return _PROGRESS_LOCK


def _set_progress(
    *,
    phase: str | None = None,
    progress_pct: int | None = None,
    detail: str | None = None,
    step: int | None = None,
    max_steps: int | None = None,
    train_log: str | None = None,
) -> None:
    with _progress_lock():
        if phase is not None:
            _PROGRESS_STATE["phase"] = phase
            floor = _PHASE_PROGRESS_FLOOR.get(phase)
            if floor is not None:
                cur = _PROGRESS_STATE.get("progress_pct")
                cur_i = int(cur) if isinstance(cur, int) else 0
                _PROGRESS_STATE["progress_pct"] = max(cur_i, min(99, floor))
            if detail is None and phase in _PHASE_LABELS:
                _PROGRESS_STATE["detail"] = _PHASE_LABELS[phase]
        if progress_pct is not None:
            cur = _PROGRESS_STATE.get("progress_pct")
            cur_i = int(cur) if isinstance(cur, int) else 0
            _PROGRESS_STATE["progress_pct"] = max(cur_i, min(99, int(progress_pct)))
        if detail is not None:
            _PROGRESS_STATE["detail"] = str(detail)[:500]
        if step is not None:
            _PROGRESS_STATE["step"] = int(step)
        if max_steps is not None:
            _PROGRESS_STATE["max_steps"] = int(max_steps)
        if train_log is not None:
            _PROGRESS_STATE["train_log"] = train_log
        # Blend train steps into 50–90 when known
        st = _PROGRESS_STATE.get("step")
        ms = _PROGRESS_STATE.get("max_steps")
        if isinstance(st, int) and isinstance(ms, int) and ms > 0:
            ratio = min(1.0, max(0.0, st / ms))
            train_pct = int(50 + ratio * 40)
            cur = _PROGRESS_STATE.get("progress_pct")
            cur_i = int(cur) if isinstance(cur, int) else 0
            _PROGRESS_STATE["progress_pct"] = max(cur_i, min(90, train_pct))
            if not _PROGRESS_STATE.get("detail") or str(_PROGRESS_STATE.get("phase")) in (
                "train",
                "train_start",
            ):
                _PROGRESS_STATE["detail"] = f"Training… ({st}/{ms})"


def _snapshot_progress() -> dict[str, object]:
    with _progress_lock():
        return dict(_PROGRESS_STATE)


def _parse_train_steps_from_log(log_path: str | None) -> tuple[int | None, int | None]:
    """Best-effort parse of HuggingFace/Axolotl progress bars: ``5/50 [``."""
    if not log_path:
        return None, None
    try:
        path = Path(log_path)
        if not path.is_file():
            return None, None
        # Tail last ~32 KiB
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > 32768:
                fh.seek(size - 32768)
            text = fh.read().decode("utf-8", errors="replace")
    except Exception:
        return None, None
    import re

    matches = re.findall(r"\|\s*(\d+)\s*/\s*(\d+)\s*\[", text)
    if not matches:
        matches = re.findall(r"\b(\d+)\s*/\s*(\d+)\s*\[", text)
    if not matches:
        return None, None
    step_s, max_s = matches[-1]
    try:
        step_i, max_i = int(step_s), int(max_s)
    except ValueError:
        return None, None
    if max_i <= 0 or step_i < 0:
        return None, None
    return step_i, max_i


class _FlushFileHandler(logging.FileHandler):
    """FileHandler that flushes every record so Windows tail / Explorer stay fresh."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def _phase(name: str, **fields: object) -> None:
    """Structured job timeline — searchable in Docker Desktop Logs as PHASE=..."""
    parts = [f"PHASE={name}"]
    for key, value in fields.items():
        if value is None:
            continue
        text = str(value).replace("\n", " ").strip()
        if len(text) > 240:
            text = text[:239] + "…"
        parts.append(f"{key}={text}")
    line = " ".join(parts)
    # ogpu logger → docker logs + file handler; phase logger is attached to file too.
    ogpu.service.logger.info("%s", line)
    _PHASE_LOGGER.info("%s", line)
    # Hard guarantee for Docker Desktop when logger wiring is incomplete.
    try:
        print(line, file=sys.stderr, flush=True)
    except Exception:
        pass
    # Feed Lab progress (heartbeat) — ignore noisy claim-retry phases.
    if name.startswith("claim_") and name not in ("claim_ok", "claim_start"):
        return
    if name.startswith("heartbeat_"):
        if name == "heartbeat_loop_started":
            _set_progress(phase=name)
        return
    train_log = fields.get("axolotl_log")
    detail_bits = []
    if "rows" in fields:
        detail_bits.append(f"rows={fields['rows']}")
    if "base_model" in fields:
        detail_bits.append(str(fields["base_model"])[:48])
    detail = None
    if name in _PHASE_LABELS:
        detail = _PHASE_LABELS[name]
        if detail_bits:
            detail = f"{detail} ({', '.join(detail_bits)})"
    _set_progress(
        phase=name,
        detail=detail,
        train_log=str(train_log) if train_log else None,
    )


def _compiler_status() -> dict[str, object]:
    gcc = shutil.which("gcc") or shutil.which("cc")
    gxx = shutil.which("g++") or shutil.which("c++")
    return {
        "has_c_compiler": bool(gcc),
        "gcc": gcc,
        "gxx": gxx,
    }


def _setup_workspace_file_logging() -> Path | None:
    """Mirror worker logs into /workspace/logs for host-side debugging."""
    global _FILE_LOG_CONFIGURED
    log_dir = Path(os.environ.get("OPENREEF_LOG_DIR", "/workspace/logs"))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "openreef_worker.log"
        # Also drop a one-line pointer at workspace root (easy to spot on Windows Explorer).
        pointer = Path("/workspace/OPENREEF_LOGS.txt")
        pointer.write_text(
            "OpenReef worker logs (live on Windows):\n"
            f"  1) Docker Desktop → container → Logs  (or: docker logs -f <name>)\n"
            f"  2) This data folder → logs/openreef_worker.log\n"
            f"     full path in container: {log_path}\n"
            "Look for lines starting with PHASE= for job progress.\n",
            encoding="utf-8",
        )
    except Exception:
        return None

    if _FILE_LOG_CONFIGURED:
        return log_path

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = _FlushFileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    # Attach to root + ogpu + phase loggers so claim/train errors land in the file.
    for name in ("", "ogpu", "ogpu.service", "openreef.phase", __name__):
        logger = logging.getLogger(name) if name else logging.getLogger()
        # Avoid duplicate handlers on re-init
        if not any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == str(log_path)
            for h in logger.handlers
        ):
            logger.addHandler(file_handler)
        if name:
            logger.setLevel(logging.DEBUG)

    # Ensure unbuffered stderr still shows in docker logs
    try:
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    _FILE_LOG_CONFIGURED = True
    logging.getLogger(__name__).info("OpenReef file logging enabled at %s", log_path)
    return log_path


class FineTuneInput(BaseModel):
    """Training input.

    Real OpenReef jobs publish an opaque ``openreef`` claim on IPFS. The worker
    exchanges it for short-lived dataset/upload URLs after the task is attempted.
    Legacy / mock / local modes may still send full fields directly.
    """

    model_config = {"extra": "ignore"}

    base_model: str = ""
    dataset_url: str = ""
    preset: str = "fast"
    adapter: str = "lora"
    # Optional legacy fields — worker re-resolves from preset + hardware.
    num_epochs: int = 2
    learning_rate: float = 1e-4
    batch_size: int = 2
    param_count: int = 0
    dataset_format: str = "jsonl"
    timeout_seconds: int = 7200
    output_prefix: str = ""
    upload_url: str = ""  # presigned PUT URL for artifact upload
    output_key: str = ""  # R2 key authorized by upload_url
    heartbeat_url: str = ""
    heartbeat_token: str = ""
    heartbeat_interval_seconds: int = 0
    lora_r: int | None = None
    lora_alpha: int | None = None
    sequence_len: int | None = None
    val_set_size: float | None = None
    # Opaque claim (public on IPFS) — resolved via OpenReef claim API
    openreef: dict | None = None

    @classmethod
    def model_validate(cls, obj, *args, **kwargs):  # type: ignore[override]
        """Accept OGPU wrappers: ``{function_name, data: {...}}`` or ``{data: {...}}``."""
        if isinstance(obj, dict) and "openreef" not in obj and isinstance(obj.get("data"), dict):
            inner = obj["data"]
            if any(k in inner for k in ("openreef", "dataset_url", "base_model")):
                obj = inner
        return super().model_validate(obj, *args, **kwargs)


class FineTuneOutput(BaseModel):
    status: str
    output_key: str | None = None
    adapter_base64: str | None = None
    artifact_format: str | None = None
    artifact_sha256: str | None = None
    artifact_size_bytes: int | None = None
    gpu_type: str | None = None
    error: str | None = None


def _detect_device() -> str:
    """Detect the available compute device.

    Returns:
        "nvidia_cuda" if NVIDIA GPU with CUDA is available.
        "amd_rocm" if AMD GPU with ROCm is available.
        "cpu" if no GPU is detected.
    """
    return detect_device_from_torch()


def _expected_device_from_env() -> str | None:
    return normalise_provider_env(os.environ.get("OPENREEF_PROVIDER_ENV"))


def _verify_expected_device(device: str | None = None) -> str:
    """Fail fast when a provider starts the wrong image for its hardware."""
    detected = device or _detect_device()
    expected = _expected_device_from_env()
    if expected and detected != expected:
        raise RuntimeError(
            f"Provider environment expects {expected}, but detected {detected}. "
            "Check the selected compose file, GPU drivers, and Docker GPU runtime."
        )
    return detected


def _validate_runtime(device: str) -> None:
    """Fail fast if the provider image cannot run the selected backend."""
    verify_runtime(expected_device=device)


def _write_runtime_report(report: dict) -> None:
    report_path = Path(os.environ.get("OPENREEF_RUNTIME_REPORT_PATH", "/workspace/openreef_runtime.json"))
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True),
            encoding="utf-8",
        )
    except Exception as exc:
        ogpu.service.logger.warning("Could not write OpenReef runtime report: %s", exc)


@ogpu.service.init()
def load_model():
    _setup_workspace_file_logging()
    _phase("runtime_probe_start")
    report = build_runtime_report(expected_device=os.environ.get("OPENREEF_PROVIDER_ENV"))
    _write_runtime_report(report)
    cc = _compiler_status()
    host = report.get("host") if isinstance(report.get("host"), dict) else {}
    _phase(
        "runtime_probe_done",
        ready=report.get("ready"),
        device=report.get("detected_device"),
        expected=report.get("expected_device"),
        host_family=host.get("host_family"),
        wsl=host.get("is_wsl"),
        has_c_compiler=cc.get("has_c_compiler"),
        issues=";".join(report.get("issues") or []) or None,
    )
    ogpu.service.logger.info(
        "OpenReef runtime report: %s",
        json.dumps(report, sort_keys=True, ensure_ascii=True),
    )
    if not report["ready"]:
        _phase("runtime_probe_failed", issues=";".join(report.get("issues") or []))
        raise RuntimeError("OpenReef runtime probe failed: " + "; ".join(report["issues"]))
    _phase("runtime_ready")
    device = report["detected_device"]
    ogpu.service.logger.info(
        "OpenReef fine-tune worker initialized — device: %s", device
    )


def _validate_public_https_url(url: str) -> None:
    """Validate a URL before provider-side fetches to reduce SSRF risk."""
    import ipaddress
    import socket
    import urllib.parse

    parsed = urllib.parse.urlparse(url)

    if not parsed.scheme or parsed.scheme not in ("http", "https"):
        raise ValueError(f"Only HTTP/HTTPS URLs allowed for dataset download: {url}")

    # Only allow HTTPS to prevent credential leakage
    if parsed.scheme != "https":
        raise ValueError(f"Only HTTPS URLs allowed for dataset download: {url}")

    # Resolve DNS and check IP is not private
    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"Invalid URL — no hostname: {url}")

    try:
        resolved_ip = socket.gethostbyname(hostname)
    except socket.gaierror as e:
        raise ValueError(f"Cannot resolve hostname {hostname}: {e}")

    ip = ipaddress.ip_address(resolved_ip)
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        raise ValueError(
            f"Dataset URL resolves to a private/internal IP ({resolved_ip}) — SSRF blocked"
        )


def _download_dataset(url: str, target: Path, max_size_bytes: int = 500 * 1024 * 1024) -> Path:
    """Download dataset from R2 (or public URL) to local path.

    Redirects are rejected instead of followed automatically so a public URL
    cannot bounce the worker into internal metadata endpoints.
    """
    _validate_public_https_url(url)

    with requests.get(
        url,
        stream=True,
        timeout=(10, 300),
        allow_redirects=False,
    ) as resp:
        if 300 <= resp.status_code < 400:
            raise ValueError("Dataset URL redirects are not allowed")
        resp.raise_for_status()

        content_length = resp.headers.get("content-length")
        if content_length and int(content_length) > max_size_bytes:
            raise ValueError(
                f"Dataset exceeds maximum size: {int(content_length) / 1024 / 1024:.1f} MB "
                f"(limit: {max_size_bytes / 1024 / 1024:.0f} MB)"
            )

        downloaded = 0
        with target.open("wb") as out:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > max_size_bytes:
                    raise ValueError(
                        f"Dataset exceeds maximum size: {downloaded / 1024 / 1024:.1f} MB "
                        f"(limit: {max_size_bytes / 1024 / 1024:.0f} MB)"
                    )
                out.write(chunk)

    return target


def _messages_to_pair(messages: object) -> tuple[str, str] | None:
    if not isinstance(messages, list):
        return None

    instruction = ""
    output = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).lower()
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if role == "user":
            instruction = content.strip()
        elif role == "assistant":
            output = content.strip()

    if output:
        return instruction, output
    return None


def _first_text_value(obj: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_pair(obj: dict) -> tuple[str, str] | None:
    messages_pair = _messages_to_pair(obj.get("messages"))
    if messages_pair:
        return messages_pair

    output = _first_text_value(obj, ("output", "completion", "response", "answer"))
    instruction = _first_text_value(obj, ("instruction", "prompt", "input", "question"))
    if output:
        return instruction, output

    text = obj.get("text")
    if isinstance(text, str) and text.strip():
        return "", text.strip()

    return None


def _write_pair(out, instruction: str, output: str) -> None:
    out.write(json.dumps(
        {"instruction": instruction, "output": output},
        ensure_ascii=False,
    ) + "\n")


def _normalise_jsonl(source: Path, target: Path) -> int:
    count = 0
    with source.open("r", encoding="utf-8") as src, target.open("w", encoding="utf-8") as out:
        for line_num, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_num}: expected a JSON object")
            pair = _extract_pair(obj)
            if pair is None:
                raise ValueError(
                    f"Line {line_num}: expected instruction/output, prompt/completion, messages, or text"
                )
            _write_pair(out, *pair)
            count += 1
    return count


def _normalise_csv(source: Path, target: Path) -> int:
    count = 0
    with source.open("r", encoding="utf-8", newline="") as src, target.open("w", encoding="utf-8") as out:
        reader = csv.DictReader(src)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        fieldnames = {name.lower(): name for name in reader.fieldnames}
        output_name = next((fieldnames[k] for k in ("output", "completion", "response", "answer", "text") if k in fieldnames), None)
        instruction_name = next((fieldnames[k] for k in ("instruction", "prompt", "input", "question") if k in fieldnames), None)
        if output_name is None:
            raise ValueError("CSV must include output, completion, response, answer, or text column")
        for row_num, row in enumerate(reader, start=2):
            output = (row.get(output_name) or "").strip()
            instruction = (row.get(instruction_name) or "").strip() if instruction_name else ""
            if not output:
                raise ValueError(f"Line {row_num}: output column is empty")
            _write_pair(out, instruction, output)
            count += 1
    return count


def _normalise_txt(source: Path, target: Path) -> int:
    count = 0
    with source.open("r", encoding="utf-8") as src, target.open("w", encoding="utf-8") as out:
        for line in src:
            text = line.strip()
            if not text:
                continue
            _write_pair(out, "", text)
            count += 1
    return count


def _normalise_dataset(source: Path, target: Path, fmt: str) -> Path:
    fmt = fmt.lower()
    if fmt == "jsonl":
        count = _normalise_jsonl(source, target)
    elif fmt == "csv":
        count = _normalise_csv(source, target)
    elif fmt == "txt":
        count = _normalise_txt(source, target)
    else:
        raise ValueError(f"Unsupported dataset format: {fmt}")
    if count == 0:
        raise ValueError("Dataset is empty")
    ogpu.service.logger.info("Normalised %d dataset rows to %s", count, target)
    return target


def _count_jsonl_rows(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def _detect_vram_gb() -> float | None:
    """Best-effort device VRAM (GB) for packing / sequence clamps."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return round(float(props.total_memory) / (1024**3), 2)
    except Exception:
        pass
    try:
        report_path = Path(os.environ.get("OPENREEF_RUNTIME_REPORT_PATH", "/workspace/openreef_runtime.json"))
        if report_path.is_file():
            data = json.loads(report_path.read_text(encoding="utf-8"))
            gpu = data.get("gpu") or {}
            if gpu.get("vram_gb") is not None:
                return float(gpu["vram_gb"])
            torch_info = data.get("torch") or {}
            if torch_info.get("vram_gb") is not None:
                return float(torch_info["vram_gb"])
    except Exception:
        pass
    return None


def _build_axolotl_config(
    job_input: FineTuneInput,
    config_dir: Path,
    *,
    num_dataset_rows: int | None = None,
) -> str:
    """Generate an Axolotl YAML config from the job parameters.

    Uses shared training_config (mirrored from backend) so local and OGPU
    paths share the same presets, batch clamps, and sequence length.
    QLoRA falls back to LoRA on AMD inside build_axolotl_config_dict.
    """
    device = _detect_device()
    adapter = job_input.adapter
    if adapter == "qlora" and device == "amd_rocm":
        ogpu.service.logger.warning(
            "QLoRA requested on AMD ROCm — falling back to LoRA "
            "(bitsandbytes ROCm support is version-dependent and fragile)"
        )
        adapter = "lora"

    vram_gb = _detect_vram_gb()
    config = build_axolotl_config_dict(
        base_model=job_input.base_model,
        dataset_path="/workspace/dataset.jsonl",
        device=device,
        adapter=adapter,
        output_dir="/workspace/output",
        preset=job_input.preset,
        param_count=job_input.param_count,
        dataset_prepared_path="/workspace/prepared",
        num_dataset_rows=num_dataset_rows,
        vram_gb=vram_gb,
    )
    # Optional *non-hardware* overrides from claim/backend.
    # Never apply sequence_len / batch / packing from the claim: claim hyperparams
    # are device-agnostic (CUDA defaults) and would undo AMD/tiny-VRAM clamps.
    if job_input.lora_r:
        config["lora_r"] = job_input.lora_r
    if job_input.lora_alpha:
        config["lora_alpha"] = job_input.lora_alpha
    # val_set_size: allow override only when dataset is large enough to split.
    if job_input.val_set_size is not None and (
        num_dataset_rows is None or int(num_dataset_rows) >= 32
    ):
        config["val_set_size"] = job_input.val_set_size
    if job_input.sequence_len:
        ogpu.service.logger.info(
            "Ignoring claim sequence_len=%s; using hardware-resolved seq=%s",
            job_input.sequence_len,
            config.get("sequence_len"),
        )

    config_path = config_dir / "axolotl.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, default_flow_style=False))
    ogpu.service.logger.info(
        "Axolotl config: preset=%s device=%s adapter=%s seq=%s batch=%s packing=%s "
        "rows=%s vram_gb=%s lora_r=%s val=%.2f gc=%s",
        job_input.preset,
        device,
        config.get("adapter"),
        config.get("sequence_len"),
        config.get("micro_batch_size"),
        config.get("sample_packing"),
        num_dataset_rows,
        vram_gb,
        config.get("lora_r"),
        float(config.get("val_set_size") or 0),
        config.get("gradient_checkpointing"),
    )
    return str(config_path)


def _extract_training_error(log_path: Path, *, max_chars: int = 2500) -> str:
    """Prefer the *real* Axolotl traceback; skip accelerate's wrapper re-raise.

    ``accelerate launch`` often ends the log with CalledProcessError after the
    child already printed a useful traceback. Taking rfind(Traceback) alone
    returns only the useless wrapper (~530s later in the dapp).
    """
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return "Training failed (could not read log)"
    if not text.strip():
        return "Training failed (empty log)"

    marker = "Traceback (most recent call last):"
    positions = []
    start = 0
    while True:
        idx = text.find(marker, start)
        if idx < 0:
            break
        positions.append(idx)
        start = idx + len(marker)

    def _slice_tb(idx: int) -> str:
        # Next traceback or end of file
        next_idxs = [p for p in positions if p > idx]
        end = next_idxs[0] if next_idxs else len(text)
        return text[idx:end].strip()

    def _is_accelerate_wrapper(tb: str) -> bool:
        low = tb.lower()
        return (
            "accelerate/commands/launch.py" in low
            or "calledprocesserror" in low
            or ("accelerate" in low and "returned non-zero exit status" in low)
        )

    # Prefer last non-wrapper traceback (actual axolotl/torch failure)
    for idx in reversed(positions):
        tb = _slice_tb(idx)
        if tb and not _is_accelerate_wrapper(tb):
            return tb[-max_chars:] if len(tb) > max_chars else tb

    # Fall back to any traceback
    if positions:
        tb = _slice_tb(positions[-1])
        return tb[-max_chars:] if len(tb) > max_chars else tb

    # Skip pure progress / deprecation noise when possible
    lines = [ln for ln in text.splitlines() if ln.strip()]
    useful = [
        ln
        for ln in lines
        if not ln.startswith("W0")
        and "torch/utils/_pytree" not in ln
        and "register_constant" not in ln
        and "Saving the dataset" not in ln
        and "huggingface/tokenizers" not in ln
    ]
    # Prefer lines that look like hard failures
    hard = [
        ln
        for ln in useful
        if any(
            k in ln
            for k in (
                "Error",
                "Exception",
                "OOM",
                "CUDA",
                "IndexError",
                "RuntimeError",
                "ValueError",
                "out of memory",
                "FAILED",
            )
        )
    ]
    blob = "\n".join(hard[-40:] if hard else (useful if useful else lines))
    return blob[-max_chars:]


def _find_adapter_file(output_dir: Path) -> Path | None:
    """Find the adapter safetensors file in the output directory.

    Searches recursively for adapter_model.safetensors, adapter_merged.safetensors,
    or any *.safetensors file. Returns the first match found.
    """
    # Priority order: specific names first, then any safetensors
    priority_names = ["adapter_model.safetensors", "adapter_merged.safetensors"]

    for name in priority_names:
        for f in output_dir.rglob(name):
            if f.is_file():
                return f

    # Fallback: any safetensors file
    for f in output_dir.rglob("*.safetensors"):
        if f.is_file():
            return f

    return None


def _package_adapter_output(adapter_path: Path, max_size_bytes: int = 2 * 1024 * 1024 * 1024) -> Path:
    """Package the final trained adapter (exclude checkpoints/prepared) as a zip."""
    package_path = adapter_path.parent / "openreef_adapter.zip"
    include_suffixes = {".safetensors", ".bin", ".json", ".model", ".txt", ".md"}
    total = 0
    root = adapter_path.parent

    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if path == package_path or path.name == "openreef_adapter.zip":
                continue
            if path.suffix not in include_suffixes:
                continue
            rel_parts = path.relative_to(root).parts
            # Intermediate training artifacts are not part of the downloadable adapter
            if any(part.startswith("checkpoint-") for part in rel_parts):
                continue
            if "prepared" in rel_parts:
                continue
            total += path.stat().st_size
            if total > max_size_bytes:
                raise ValueError(
                    f"Packaged adapter too large: {total / 1024 / 1024:.1f} MB "
                    f"(limit: {max_size_bytes / 1024 / 1024:.0f} MB)"
                )
            zf.write(path, path.relative_to(root))

        manifest = {
            "format": "openreef-adapter",
            "primary_file": adapter_path.name,
        }
        zf.writestr("openreef_manifest.json", json.dumps(manifest, indent=2))

    return package_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_text_tail(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as f:
        f.seek(max(0, size - max_chars))
        data = f.read()
    return data.decode("utf-8", errors="replace")


def _encode_adapter_base64(adapter_path: Path, max_size_bytes: int = 100 * 1024 * 1024) -> str | None:
    """Read packaged adapter bytes and encode as base64.

    Returns None if the file is too large (>100MB) to encode safely.
    """
    file_size = adapter_path.stat().st_size
    if file_size > max_size_bytes:
        ogpu.service.logger.warning(
            "Adapter file too large for base64 encoding: %.1f MB (limit: %.0f MB)",
            file_size / 1024 / 1024, max_size_bytes / 1024 / 1024,
        )
        return None

    with adapter_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _upload_via_presigned_url(adapter_path: Path, upload_url: str, max_size_bytes: int = 2 * 1024 * 1024 * 1024) -> None:
    """Upload the adapter file to R2 using a presigned PUT URL.

    No R2 credentials needed — the URL is pre-authorized by the backend.

    Args:
        adapter_path: Path to the adapter safetensors file.
        upload_url: Presigned PUT URL from the backend.
        max_size_bytes: Maximum file size (default 2 GB).
    """
    file_size = adapter_path.stat().st_size
    if file_size > max_size_bytes:
        raise ValueError(
            f"Adapter file too large: {file_size / 1024 / 1024:.1f} MB "
            f"(limit: {max_size_bytes / 1024 / 1024:.0f} MB)"
        )

    with adapter_path.open("rb") as f:
        resp = requests.put(
            upload_url,
            data=f,
            headers={"Content-Type": "application/zip"},
            timeout=300,
            allow_redirects=False,
        )
        if 300 <= resp.status_code < 400:
            raise ValueError("Presigned upload URL redirects are not allowed")
        resp.raise_for_status()

    ogpu.service.logger.info(
        "Uploaded adapter to R2 via presigned URL: %d bytes", file_size
    )


def _provider_private_key() -> str:
    """Provider wallet key injected by provider-app at runtime (never baked into image)."""
    for name in (
        "PROVIDER_PRIVATE_KEY",
        "OPENREEF_PROVIDER_PRIVATE_KEY",
        "OGPU_PROVIDER_PRIVATE_KEY",
    ):
        raw = (os.environ.get(name) or "").strip()
        if raw:
            return raw
    return ""


def _public_claim_error(exc: BaseException) -> str:
    """Short, non-actionable claim error for OGPU task response / management dapps.

    Full diagnostics stay in docker logs / PHASE= lines only — never env names,
    compose hints, API paths, or HTTP bodies.
    """
    text = str(exc or "").lower()
    if any(
        k in text
        for k in (
            "sign",
            "private_key",
            "provider_private",
            "missing_provider",
            "eth_account",
            "invalid key",
        )
    ):
        return "Failed to sign OpenReef claim"
    if "task_address" in text or "missing_task" in text:
        return "OpenReef claim failed"
    if "http" in text or "409" in text or "401" in text or "claim failed" in text:
        return "OpenReef claim failed"
    if "training fields" in text or "dataset_url" in text or "base_model" in text:
        return "OpenReef claim incomplete"
    return "OpenReef claim failed"


def _open_gpu_chain_id() -> int:
    """Match OpenReef backend defaults (mainnet 1071 / testnet 200820172034)."""
    raw = (os.environ.get("OPENREEF_CHAIN_ID") or os.environ.get("OGPU_CHAIN_ID") or "").strip()
    if raw.isdigit():
        return int(raw)
    # Default mainnet; set OPENREEF_CHAIN_ID=200820172034 for testnet.
    flag = (os.environ.get("OGPU_USE_TESTNET") or "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return 200820172034
    return 1071


def _build_and_sign_claim(
    *,
    job_id: str,
    task_address: str,
    api_base: str,
) -> tuple[str, str, str]:
    """Return (provider_address, message, signature) for D2 claim."""
    from eth_account import Account
    from eth_account.messages import encode_defunct

    key = _provider_private_key()
    if not key:
        _phase("claim_sign_fail", reason="missing_PROVIDER_PRIVATE_KEY")
        raise RuntimeError(
            "PROVIDER_PRIVATE_KEY is not set in the finetune container. "
            "OpenReef D2 claim requires the provider wallet key at runtime "
            "(injected by provider-app compose — not baked into the image)."
        )
    if not key.startswith("0x"):
        key = "0x" + key
    account = Account.from_key(key)
    address = account.address.lower()
    chain_id = _open_gpu_chain_id()
    # Canonical message (must match backend claim_signature.py)
    import secrets
    from datetime import datetime, timedelta, timezone

    issued = datetime.now(timezone.utc)
    expires = issued + timedelta(minutes=5)

    def _iso(dt: datetime) -> str:
        return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    task_norm = str(task_address or "").strip().lower()
    nonce = secrets.token_urlsafe(18)
    message = "\n".join(
        [
            "OpenReef Job Claim",
            f"Address: {address}",
            f"Chain ID: {chain_id}",
            "Purpose: claim job training secrets",
            f"Job ID: {job_id}",
            f"Task Address: {task_norm}",
            f"API Base: {api_base.rstrip('/')}",
            f"Issued At: {_iso(issued)}",
            f"Expires At: {_iso(expires)}",
            f"Nonce: {nonce}",
        ]
    )
    signed = Account.sign_message(encode_defunct(text=message), private_key=key)
    signature = signed.signature.hex()
    if not signature.startswith("0x"):
        signature = "0x" + signature
    # Visible in `docker logs` — never log the key or full signature.
    _phase(
        "claim_signed",
        job_id=job_id[:8],
        wallet=address[:10] + "…",
        chain_id=chain_id,
        task=(task_norm[:12] + "…") if len(task_norm) > 12 else task_norm,
        nonce=nonce[:8] + "…",
        sig_len=len(signature),
    )
    return address, message, signature


def _signer_base_url() -> str:
    """If set, D2 claim is performed by the external openreef-signer service."""
    return (os.environ.get("OPENREEF_SIGNER_URL") or "").strip().rstrip("/")


class _ClaimantHeartbeat:
    """Lifecycle-bound heartbeat for one claimed training attempt."""

    def __init__(self, data: FineTuneInput):
        self.url = str(data.heartbeat_url or "").strip()
        self.token = str(data.heartbeat_token or "").strip()
        self.interval = max(15, int(data.heartbeat_interval_seconds or 60))
        self._stop = None
        self._thread = None

    def start(self) -> None:
        if not self.url or not self.token:
            return

        import threading
        import urllib.error
        import urllib.request

        self._stop = threading.Event()

        def _post() -> bool:
            try:
                snap = _snapshot_progress()
                # During train, lift step/max from axolotl log when possible.
                train_log = snap.get("train_log")
                if isinstance(train_log, str) and train_log:
                    step, max_steps = _parse_train_steps_from_log(train_log)
                    if step is not None and max_steps is not None:
                        _set_progress(phase="train", step=step, max_steps=max_steps)
                        snap = _snapshot_progress()

                payload: dict[str, object] = {"token": self.token}
                phase = snap.get("phase")
                if isinstance(phase, str) and phase:
                    payload["phase"] = phase
                pct = snap.get("progress_pct")
                if isinstance(pct, int):
                    payload["progress_pct"] = pct
                detail = snap.get("detail")
                if isinstance(detail, str) and detail:
                    payload["detail"] = detail[:500]
                step = snap.get("step")
                max_steps = snap.get("max_steps")
                if isinstance(step, int) and isinstance(max_steps, int) and max_steps > 0:
                    payload["step"] = step
                    payload["max_steps"] = max_steps

                body = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    self.url,
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "User-Agent": "openreef-finetune-worker/1.0",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=20) as resp:
                    resp.read()
                return True
            except urllib.error.HTTPError as exc:
                if exc.code in (400, 401, 403, 404, 409, 410):
                    _phase("heartbeat_rejected", http=exc.code)
                    return False
                _phase("heartbeat_error", http=exc.code)
                return True
            except Exception as e:
                _phase(
                    "heartbeat_error",
                    error=type(e).__name__,
                    msg=str(e)[:120],
                )
                return True

        def _loop() -> None:
            if not _post():
                return
            while not self._stop.wait(self.interval):
                if not _post():
                    return

        self._thread = threading.Thread(
            target=_loop,
            name="openreef-claimant-heartbeat",
            daemon=True,
        )
        self._thread.start()
        _phase("heartbeat_loop_started", interval_s=self.interval)

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._thread = None


def _resolve_openreef_claim_via_signer(
    data: FineTuneInput,
    *,
    job_id: str,
    api_base: str,
    task: str,
    claim_v: int | str,
) -> FineTuneInput:
    """Delegate claim signing + OpenReef HTTP to the sidecar (no PK in this process)."""
    import urllib.error
    import urllib.request

    signer = _signer_base_url()
    token = (os.environ.get("OPENREEF_SIGNER_TOKEN") or "").strip()
    _phase(
        "claim_start",
        job_id=job_id[:8],
        v=claim_v or "?",
        api_base=api_base,
        task_set=bool(task),
        via_signer=True,
        signer=signer[:48],
        key_env_set=False,
    )
    if not task:
        _phase("claim_fail", reason="missing_task_address")
        raise RuntimeError("OpenReef claim failed")

    body = json.dumps(
        {
            "job_id": job_id,
            "task_address": task,
            "api_base": api_base,
            "job_kind": "finetune",
        }
    ).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "openreef-finetune-worker/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    last_err: Exception | None = None
    for attempt in range(36):
        req = urllib.request.Request(
            f"{signer}/v1/claim",
            data=body,
            headers=headers,
            method="POST",
        )
        _phase("claim_signer_post", attempt=attempt + 1, of=36)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                http_status = getattr(resp, "status", 200)
                payload = json.loads(resp.read().decode("utf-8"))
            training = payload.get("training") or {}
            if not training.get("dataset_url") or not training.get("base_model"):
                _phase("claim_bad_response", fields=list(training.keys())[:12], via_signer=True)
                raise RuntimeError("OpenReef claim incomplete")
            wallet = str(payload.get("provider_address") or "")[:10]
            _phase(
                "claim_ok",
                job_id=job_id[:8],
                wallet=wallet + "…" if wallet else "?",
                http=http_status,
                via_signer=True,
                has_dataset_url=True,
                has_upload_url=bool(training.get("upload_url")),
                base_model=str(training.get("base_model") or "")[:48],
                preset=training.get("preset"),
                timeout_s=training.get("timeout_seconds"),
            )
            ogpu.service.logger.info(
                "Resolved OpenReef claim via signer for job %s (presigned secrets not logged)",
                job_id,
            )
            return FineTuneInput(
                **{**data.model_dump(exclude={"openreef"}), **training, "openreef": None}
            )
        except urllib.error.HTTPError as e:
            last_err = e
            detail = e.read().decode("utf-8", errors="replace")[:200]
            # Signer maps OpenReef 409 to retries internally; 502/503 may mean wait or misconfig.
            if e.code in (429, 502, 503):
                _phase(
                    "claim_signer_retry",
                    attempt=attempt + 1,
                    http=e.code,
                    detail=detail[:80],
                )
                time.sleep(5)
                continue
            _phase("claim_signer_error", attempt=attempt + 1, http=e.code)
            raise RuntimeError("OpenReef claim failed") from e
        except Exception as e:
            last_err = e
            _phase(
                "claim_signer_error",
                attempt=attempt + 1,
                error=type(e).__name__,
                msg=str(e)[:120],
            )
            time.sleep(5)
    _phase("claim_exhausted", attempts=36, via_signer=True)
    raise RuntimeError(f"OpenReef claim failed after retries: {last_err}")


def _resolve_openreef_claim(data: FineTuneInput, task_address: str | None = None) -> FineTuneInput:
    """If payload is an OpenReef claim, fetch private training secrets from the API."""
    claim = data.openreef
    if not isinstance(claim, dict):
        return data
    api_base = str(claim.get("api_base") or os.environ.get("OPENREEF_API_BASE") or "").rstrip("/")
    job_id = str(claim.get("job_id") or "").strip()
    claim_v = int(claim.get("v") or 0)
    if not api_base or not job_id:
        _phase("claim_fail", reason="invalid_payload", has_api_base=bool(api_base), has_job_id=bool(job_id))
        raise RuntimeError("Invalid openreef claim payload (api_base/job_id required)")

    # Hard cut: v1 claim_token is no longer accepted by the API.
    if claim.get("claim_token") and claim_v < 2:
        _phase("claim_legacy_token_ignored", job_id=job_id[:8])
        ogpu.service.logger.warning(
            "Legacy claim_token present in payload; D2 uses wallet signature only"
        )

    task = (
        (task_address or "").strip()
        or str(claim.get("task_address") or "").strip()
        or (os.environ.get("OGPU_TASK_ADDRESS") or "").strip()
        or (os.environ.get("TASK_ADDRESS") or "").strip()
        or (os.environ.get("OGPU_CURRENT_TASK") or "").strip()
    )

    # Preferred path: external signer (PK never in this container).
    if _signer_base_url():
        return _resolve_openreef_claim_via_signer(
            data,
            job_id=job_id,
            api_base=api_base,
            task=task,
            claim_v=claim_v or "?",
        )

    _phase(
        "claim_start",
        job_id=job_id[:8],
        v=claim_v or "?",
        api_base=api_base,
        task_set=bool(task),
        via_signer=False,
        key_env_set=bool(_provider_private_key()),
    )

    import urllib.error
    import urllib.request

    url = f"{api_base}/api/provider/v1/jobs/{job_id}/claim"
    chain_id = _open_gpu_chain_id()
    if not task:
        _phase("claim_fail", reason="missing_task_address")
        raise RuntimeError(
            "OpenReef claim requires task_address (OGPU URL path / env OGPU_TASK_ADDRESS). "
            "Empty task cannot be signed for D2."
        )

    last_err: Exception | None = None
    # Wait until the provider attempt is visible on-chain (claim returns 409 until then).
    for attempt in range(36):
        try:
            provider_address, message, signature = _build_and_sign_claim(
                job_id=job_id,
                task_address=str(task),
                api_base=api_base,
            )
        except Exception as e:
            _phase("claim_sign_error", attempt=attempt + 1, error=type(e).__name__)
            raise RuntimeError(f"Failed to sign OpenReef claim: {e}") from e

        body = json.dumps(
            {
                "provider_address": provider_address,
                "message": message,
                "signature": signature,
                "task_address": task,
                "chain_id": chain_id,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "openreef-finetune-worker/1.0",
            },
            method="POST",
        )
        _phase("claim_http_post", attempt=attempt + 1, of=36, url_path=f"/api/provider/v1/jobs/{job_id[:8]}…/claim")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                http_status = getattr(resp, "status", 200)
                payload = json.loads(resp.read().decode("utf-8"))
            training = payload.get("training") or {}
            if not training.get("dataset_url") or not training.get("base_model"):
                _phase("claim_bad_response", fields=list(training.keys())[:12])
                raise RuntimeError(f"Claim response missing training fields: {list(training.keys())}")
            _phase(
                "claim_ok",
                job_id=job_id[:8],
                wallet=provider_address[:10] + "…",
                http=http_status,
                has_dataset_url=True,
                has_upload_url=bool(training.get("upload_url")),
                base_model=str(training.get("base_model") or "")[:48],
                preset=training.get("preset"),
                timeout_s=training.get("timeout_seconds"),
            )
            ogpu.service.logger.info(
                "Resolved OpenReef claim for job %s claimant=%s… (presigned secrets not logged)",
                job_id,
                provider_address[:10],
            )
            return FineTuneInput(**{**data.model_dump(exclude={"openreef"}), **training, "openreef": None})
        except urllib.error.HTTPError as e:
            last_err = e
            detail = e.read().decode("utf-8", errors="replace")[:300]
            if e.code == 409:
                _phase("claim_wait_attempt", attempt=attempt + 1, of=36, http=409, detail=detail[:120])
                ogpu.service.logger.info(
                    "Claim not ready yet (attempt %s/36): %s", attempt + 1, detail
                )
                time.sleep(5)
                continue
            _phase("claim_http_error", attempt=attempt + 1, http=e.code, detail=detail[:160])
            raise RuntimeError(f"OpenReef claim failed HTTP {e.code}: {detail}") from e
        except Exception as e:
            last_err = e
            _phase("claim_error", attempt=attempt + 1, error=type(e).__name__, msg=str(e)[:120])
            ogpu.service.logger.warning("Claim request error (attempt %s/36): %s", attempt + 1, e)
            time.sleep(5)
    _phase("claim_exhausted", attempts=36)
    raise RuntimeError(f"OpenReef claim failed after retries: {last_err}")


@ogpu.service.expose(timeout=86400)
def finetune(data: FineTuneInput) -> FineTuneOutput:
    """Execute the fine-tuning job.

    Automatically detects hardware and configures Axolotl accordingly.
    Returns the adapter as base64 (for small adapters) or uploads to R2
    and returns the output key.
    """
    _setup_workspace_file_logging()
    job_t0 = time.monotonic()
    heartbeat: _ClaimantHeartbeat | None = None

    def _finish(output: FineTuneOutput) -> FineTuneOutput:
        if heartbeat is not None:
            heartbeat.stop()
        return output

    # Resolve opaque claim before logging model/dataset details.
    task_address = (
        os.environ.get("OGPU_TASK_ADDRESS")
        or os.environ.get("TASK_ADDRESS")
        or os.environ.get("OGPU_CURRENT_TASK")
    )
    has_claim = bool(getattr(data, "openreef", None))
    _phase(
        "job_start",
        task=task_address,
        openreef_claim=has_claim,
        base_model=getattr(data, "base_model", "") or None,
        dataset_url_set=bool(getattr(data, "dataset_url", None)),
        preset=getattr(data, "preset", None),
        adapter=getattr(data, "adapter", None),
    )
    ogpu.service.logger.info(
        "finetune() entered task=%s openreef=%s base_model=%r dataset_url_set=%s",
        task_address,
        has_claim,
        getattr(data, "base_model", ""),
        bool(getattr(data, "dataset_url", None)),
    )

    try:
        if has_claim:
            _phase("claim_start", task=task_address)
            data = _resolve_openreef_claim(data, task_address=task_address)
            _phase(
                "claim_ok",
                base_model=data.base_model,
                dataset_url_set=bool(data.dataset_url),
                upload_url_set=bool(data.upload_url),
            )
            heartbeat = _ClaimantHeartbeat(data)
            heartbeat.start()
        else:
            _phase("claim_skip", reason="no_openreef_payload")
    except Exception as e:
        # Keep full exception in logs/PHASE; public OGPU response stays minimal.
        _phase("claim_fail", error=str(e)[:240])
        ogpu.service.logger.exception("Claim failed: %s", e)
        return _finish(FineTuneOutput(status="failed", error=_public_claim_error(e)))

    # Fail closed: never train without a real dataset URL (claim must have succeeded).
    if not (data.dataset_url or "").strip() or not (data.base_model or "").strip():
        _phase(
            "job_failed",
            reason="missing_training_fields",
            dataset_url_set=bool((data.dataset_url or "").strip()),
            base_model_set=bool((data.base_model or "").strip()),
        )
        ogpu.service.logger.error(
            "Missing base_model/dataset_url after claim resolution "
            "(public config must include openreef job_id; claim API must succeed)"
        )
        return _finish(FineTuneOutput(status="failed", error="OpenReef claim incomplete"))

    try:
        device = _verify_expected_device()
        _validate_runtime(device)
        _phase("device_ok", device=device, base_model=data.base_model, preset=data.preset, adapter=data.adapter)
    except Exception as e:
        _phase("device_fail", error=str(e))
        return _finish(FineTuneOutput(status="failed", error=str(e)))

    ogpu.service.logger.info(
        "Starting fine-tune: %s / %s / %s (device: %s)",
        data.base_model, data.preset, data.adapter, device,
    )

    work_dir = Path("/workspace")
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Download dataset
        _phase("dataset_start")
        ogpu.service.logger.info("Downloading dataset...")
        raw_suffix = data.dataset_format.lower() if data.dataset_format else "jsonl"
        raw_dataset_path = _download_dataset(data.dataset_url, work_dir / f"dataset.raw.{raw_suffix}")
        dataset_path = work_dir / "dataset.jsonl"
        _normalise_dataset(raw_dataset_path, dataset_path, raw_suffix)
        num_rows = _count_jsonl_rows(dataset_path)
        _phase("dataset_ok", rows=num_rows, format=raw_suffix, bytes=raw_dataset_path.stat().st_size)

        # Drop Axolotl prepared-cache from prior jobs on this provider volume.
        # Persistent /workspace/prepared can be reused across jobs; a stale 3-row
        # smoke cache was observed to force max_steps=1 on a 539-row expert job
        # (OpenReef expert eval 2026-07-20). Always re-tokenize the current download.
        prepared_dir = work_dir / "prepared"
        if prepared_dir.exists():
            shutil.rmtree(prepared_dir, ignore_errors=True)
        prepared_dir.mkdir(parents=True, exist_ok=True)
        # Also clear previous adapter output so packaging never picks stale weights.
        output_dir = work_dir / "output"
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        _phase("workspace_reset", prepared_cleared=True, output_cleared=True, dataset_rows=num_rows)

        # 2. Build Axolotl config (hardware-aware; packing gated by dataset size)
        _phase("config_start", rows=num_rows, device=device)
        ogpu.service.logger.info("Building Axolotl config for %s (rows=%s)...", device, num_rows)
        config_path = _build_axolotl_config(data, work_dir, num_dataset_rows=num_rows)
        # Re-read key knobs for the phase line (already logged in _build_axolotl_config).
        try:
            cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        except Exception:
            cfg = {}
        _phase(
            "config_ok",
            packing=cfg.get("sample_packing"),
            seq=cfg.get("sequence_len"),
            batch=cfg.get("micro_batch_size"),
            optimizer=cfg.get("optimizer"),
            epochs=cfg.get("num_epochs"),
            bf16=cfg.get("bf16"),
            fp16=cfg.get("fp16"),
        )

        # 3. Run Axolotl training
        ogpu.service.logger.info("Running Axolotl training...")
        timeout_seconds = max(1800, min(int(data.timeout_seconds or 7200), 86400))
        log_path = work_dir / "axolotl.log"
        env = os.environ.copy()
        env["AXOLOTL_DO_NOT_TRACK"] = "1"
        env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        env.setdefault("WANDB_MODE", "disabled")
        # Cross-platform train env (WSL2/Windows Docker Desktop, native Linux, …).
        # Auto-disables fragile PyTorch 2.9 Triton native JIT on CUDA / WSL.
        from platform_compat import apply_train_compat_env, detect_container_host_context

        host_ctx = detect_container_host_context()
        apply_train_compat_env(env, device)
        cc = _compiler_status()
        _phase(
            "host_compat",
            host_family=host_ctx.get("host_family"),
            wsl=host_ctx.get("is_wsl"),
            docker_desktop=host_ctx.get("is_docker_desktop"),
            eager_torch=env.get("TORCH_DISABLE_NATIVE_JIT"),
            eager_reason=env.get("OPENREEF_EAGER_TORCH_REASON"),
            has_c_compiler=cc.get("has_c_compiler"),
            gcc=cc.get("gcc"),
        )
        ogpu.service.logger.info(
            "Host compat: family=%s wsl=%s docker_desktop=%s eager_torch=%s reason=%s has_c_compiler=%s",
            host_ctx.get("host_family"),
            host_ctx.get("is_wsl"),
            host_ctx.get("is_docker_desktop"),
            env.get("TORCH_DISABLE_NATIVE_JIT"),
            env.get("OPENREEF_EAGER_TORCH_REASON"),
            cc.get("has_c_compiler"),
        )

        # If this host also runs a desktop compositor on the same GPU, reserve VRAM.
        # Headless provider containers: no DISPLAY → no cap (full card for training).
        from display_gpu_guard import (
            apply_guard_to_env,
            resolve_display_gpu_guard,
            write_vram_cap_wrapper,
        )

        guard = resolve_display_gpu_guard(device, env)
        apply_guard_to_env(env, guard)
        if guard.protect:
            _phase(
                "display_guard",
                protect=guard.protect,
                fraction=guard.vram_fraction,
                reason=guard.reason,
            )
            ogpu.service.logger.info(
                "Display GPU guard: protect=%s fraction=%s reason=%s",
                guard.protect,
                guard.vram_fraction,
                guard.reason,
            )

        # Prefer single-process `python -m axolotl.cli.train` over `accelerate launch`.
        # Accelerate re-raises CalledProcessError and hides the real child traceback
        # in response IPFS (seen as ~530s fail with only the wrapper message).
        training_python = env.get("OPENREEF_TRAINING_PYTHON") or "python"
        if guard.enabled:
            # Single-process train with hard VRAM ceiling (desktop-safe).
            wrapper = write_vram_cap_wrapper(
                work_dir,
                training_python=training_python,
                config_path=str(config_path),
            )
            train_cmd = [training_python, wrapper]
        else:
            train_cmd = [
                training_python, "-m", "axolotl.cli.train",
                str(config_path),
            ]

        _phase(
            "train_start",
            cmd=" ".join(train_cmd),
            timeout_s=timeout_seconds,
            axolotl_log=str(log_path),
        )
        train_t0 = time.monotonic()
        with log_path.open("w", encoding="utf-8") as train_log:
            result = subprocess.run(
                train_cmd,
                stdout=train_log,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=timeout_seconds,
            )
        train_s = round(time.monotonic() - train_t0, 1)
        _phase("train_exit", code=result.returncode, seconds=train_s)

        if result.returncode != 0:
            error_tail = _extract_training_error(log_path, max_chars=2500)
            # Help providers correlate "missing c compiler" with our runtime image.
            low = error_tail.lower()
            if (
                "c compiler" in low
                or "compiler cannot be found" in low
                or "unable to find a compatible compiler" in low
                or ("triton" in low and "compile" in low)
            ) and not cc.get("has_c_compiler"):
                hint = (
                    "HINT: image has no C compiler (CUDA runtime image by design). "
                    "OpenReef sets lora_*_kernel=false so Axolotl must not use Triton "
                    "LoRA/SwiGLU kernels. If you still see this: (1) pull latest "
                    "finetune-worker:cuda-latest, (2) TORCH_DISABLE_NATIVE_JIT=1 only "
                    "covers torch.compile — not Axolotl kernels, (3) do not force "
                    "lora_mlp_kernel/lora_qkv_kernel true without installing gcc."
                )
                ogpu.service.logger.error("%s", hint)
                error_tail = f"{error_tail}\n{hint}"
            # Last lines of axolotl.log for Docker Desktop (full log is on disk).
            try:
                ax_tail = _read_text_tail(log_path, max_chars=1200)
                if ax_tail.strip():
                    _phase("train_log_tail", text=ax_tail.replace("\n", " | "))
            except Exception:
                pass
            ogpu.service.logger.error("Training failed: %s", error_tail)
            _phase(
                "train_failed",
                code=result.returncode,
                seconds=train_s,
                error=(error_tail[:200] + "…") if len(error_tail) > 200 else error_tail,
            )
            # Prefer the *end* of the traceback (root exception line) when clipping.
            if len(error_tail) > 2200:
                error_tail = "…" + error_tail[-2199:]
            _phase("job_failed", reason="train", total_s=round(time.monotonic() - job_t0, 1))
            return _finish(FineTuneOutput(status="failed", error=error_tail, gpu_type=device))

        # 4. Find and return the adapter
        _phase("artifact_start")
        ogpu.service.logger.info("Locating trained adapter...")
        output_dir = work_dir / "output"
        adapter_path = _find_adapter_file(output_dir)

        if adapter_path is None:
            ogpu.service.logger.error("No adapter safetensors file found in %s", output_dir)
            _phase("job_failed", reason="no_adapter")
            return _finish(FineTuneOutput(status="failed", error="No adapter file found after training", gpu_type=device))

        ogpu.service.logger.info("Found adapter: %s (%.1f MB)", adapter_path, adapter_path.stat().st_size / 1024 / 1024)
        package_path = _package_adapter_output(adapter_path)
        ogpu.service.logger.info("Packaged adapter: %s (%.1f MB)", package_path, package_path.stat().st_size / 1024 / 1024)
        artifact_sha256 = _sha256_file(package_path)
        artifact_size = package_path.stat().st_size
        _phase(
            "artifact_ok",
            path=str(package_path),
            mb=round(artifact_size / (1024 * 1024), 2),
            sha256=artifact_sha256[:16] + "…",
        )

        if data.upload_url:
            _phase("upload_start")
            ogpu.service.logger.info("Uploading packaged adapter via presigned URL...")
            _upload_via_presigned_url(package_path, data.upload_url)
            ogpu.service.logger.info("Training complete. Artifact uploaded to R2 via presigned URL.")
            _phase("upload_ok")
            _phase("job_done", total_s=round(time.monotonic() - job_t0, 1), train_s=train_s)
            return _finish(FineTuneOutput(
                status="completed",
                output_key=data.output_key or None,
                artifact_format="zip",
                artifact_sha256=artifact_sha256,
                artifact_size_bytes=artifact_size,
                gpu_type=device,
            ))

        # Try to encode as base64 when no upload URL is available.
        adapter_b64 = _encode_adapter_base64(package_path)
        if adapter_b64:
            ogpu.service.logger.info("Returning adapter as base64 (%.1f MB encoded)", len(adapter_b64) / 1024 / 1024)
            _phase("job_done", total_s=round(time.monotonic() - job_t0, 1), mode="base64")
            return _finish(FineTuneOutput(
                status="completed",
                adapter_base64=adapter_b64,
                artifact_format="zip",
                artifact_sha256=artifact_sha256,
                artifact_size_bytes=artifact_size,
                gpu_type=device,
            ))

        ogpu.service.logger.error("No upload_url provided for large adapter upload")
        _phase("job_failed", reason="no_upload_url")
        return _finish(FineTuneOutput(status="failed", error="No upload URL provided for artifact upload", gpu_type=device))

    except subprocess.TimeoutExpired:
        _phase("job_failed", reason="timeout", timeout_s=data.timeout_seconds)
        return _finish(FineTuneOutput(
            status="failed",
            error=f"Training exceeded timeout ({data.timeout_seconds}s)",
            gpu_type=device,
        ))
    except Exception as e:
        ogpu.service.logger.exception("Training failed: %s", e)
        _phase("job_failed", reason="exception", error=str(e))
        return _finish(FineTuneOutput(status="failed", error=str(e), gpu_type=device))


def _install_ogpu_task_address_env_patch() -> None:
    """OGPU FastAPI exposes /run/{fn}/{task_address} but never passes task_address
    into the handler — only ``handler(data)``. Inject it into env for the job
    so D2 claim can sign the correct task (H-02 / claim e2e fix).

    Important: ``ogpu.service.start`` is a *bound import* of ``server.start`` at
    package import time. Patching only ``server.start`` leaves
    ``ogpu.service.start()`` on the original function (seen in e2e: log still
    said "Starting OpenGPU Service server..." without the OpenReef marker).
    """
    import ogpu.service as svc_mod
    import ogpu.service.server as server_mod

    if getattr(server_mod, "_openreef_task_env_patched", False):
        return

    def start_patched() -> None:  # type: ignore[no-untyped-def]
        from contextlib import asynccontextmanager
        from typing import AsyncIterator

        import uvicorn
        from fastapi import BackgroundTasks, FastAPI, HTTPException

        from ogpu.service.config import CALLBACK_URL, SERVICE_HOST, SERVICE_PORT
        from ogpu.service.handler import get_handlers, get_init_handler
        from ogpu.service.logger import logger
        from ogpu.service.server import send_callback

        logger.info("Starting OpenGPU Service server (OpenReef task_address env patch)...")

        @asynccontextmanager
        async def lifespan(_: FastAPI) -> AsyncIterator[None]:
            init_handler = get_init_handler()
            if init_handler:
                try:
                    logger.info(f"Executing init function: `{init_handler.__name__}`")
                    init_handler()
                    logger.info(
                        f"Init function `{init_handler.__name__}` completed successfully"
                    )
                except Exception as e:
                    logger.error(f"Init function `{init_handler.__name__}` failed: {e}")
                    raise e
            logger.info("Connected to OpenGPU Service 🔵")
            logger.info(f"API docs: http://{SERVICE_HOST}:{SERVICE_PORT}/docs")
            yield

        app = FastAPI(title="OpenGPU Service", version="0.1.0", lifespan=lifespan)

        def create_endpoint(handler, input_model, function_name):
            async def endpoint(
                task_address: str,
                data: input_model,  # type: ignore[valid-type]
                background_tasks: BackgroundTasks,
            ):
                paused_reason = provider_pause_reason()
                if paused_reason:
                    _phase("provider_paused", reason=paused_reason)
                    raise HTTPException(
                        status_code=503,
                        detail="OpenReef provider is paused by its operator",
                    )

                def runner():
                    prev_ogpu = os.environ.get("OGPU_TASK_ADDRESS")
                    prev_task = os.environ.get("TASK_ADDRESS")
                    try:
                        # Visible to finetune() / claim without changing handler signature.
                        os.environ["OGPU_TASK_ADDRESS"] = task_address
                        os.environ["TASK_ADDRESS"] = task_address
                        _phase(
                            "task_env_set",
                            task=task_address[:14] + "…"
                            if len(task_address) > 14
                            else task_address,
                        )
                        result = handler(data)
                        if result:
                            logger.task_success(  # type: ignore[attr-defined]
                                f"[{task_address}] Function: `{function_name}` completed successfully"
                            )
                            send_callback(task_address, result.model_dump())
                    except Exception as e:
                        logger.task_fail(  # type: ignore[attr-defined]
                            f"[{task_address}] Error in `{function_name}`: {e}"
                        )
                    finally:
                        if prev_ogpu is None:
                            os.environ.pop("OGPU_TASK_ADDRESS", None)
                        else:
                            os.environ["OGPU_TASK_ADDRESS"] = prev_ogpu
                        if prev_task is None:
                            os.environ.pop("TASK_ADDRESS", None)
                        else:
                            os.environ["TASK_ADDRESS"] = prev_task

                background_tasks.add_task(runner)
                return {"task_address": task_address, "status": "accepted"}

            return endpoint

        for handler, input_model, _output_model in get_handlers():
            function_name = handler.__name__
            path = f"/run/{function_name}/{{task_address}}"
            endpoint = create_endpoint(handler, input_model, function_name)
            app.post(path, status_code=202)(endpoint)
            logger.info(f"Registered endpoint → /run/{function_name}/{{task_address}}")

        uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT, log_level="warning")

    # Patch both the module attribute and the package re-export used by worker main.
    server_mod.start = start_patched  # type: ignore[assignment]
    svc_mod.start = start_patched  # type: ignore[assignment]
    server_mod._openreef_task_env_patched = True  # type: ignore[attr-defined]
    svc_mod.logger.info(
        "OpenReef: patched OGPU server + ogpu.service.start to inject OGPU_TASK_ADDRESS"
    )


if __name__ == "__main__":
    paused_reason = provider_pause_reason()
    if paused_reason:
        print(
            f"PHASE=provider_paused reason={paused_reason}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(0)
    _install_ogpu_task_address_env_patch()
    ogpu.service.start()
