# OpenReef Fine-Tune Worker — NVIDIA CUDA (Profile A)
#
# One NVIDIA source for essentially all modern GPUs. Keep this simpler than
# the ROCm image: no gfx-specific forks, just a reproducible CUDA stack.
#
# All versions pinned via ARGs below (the lock is the Dockerfile itself;
# pins-cuda.env + automated refresh land in a follow-up PR).
#
# Build-time checks do NOT require a GPU. torch.cuda.is_available() may be
# false on CI runners; we only assert the wheel is a CUDA build + package set.
# NOTE: `import unsloth` needs a visible accelerator, so the SFT API smoke
# runs with GPU access (see scripts/smoke_unsloth_rocm.py pattern / Vast).
# Merge gate for pin bumps also requires a 1-step GPU mini-train (house/Vast).

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# --- complete lock (override from pins-cuda.env) ---
ARG TORCH_VERSION=2.9.1
ARG TORCH_CUDA_CHANNEL=cu126
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
ARG XFORMERS_VERSION=0.0.33.post2
ARG XFORMERS_INDEX_URL=https://download.pytorch.org/whl/cu126
ARG UNSLOTH_VERSION=2026.8.10
ARG UNSLOTH_ZOO_VERSION=2026.8.7
ARG TRANSFORMERS_VERSION=4.56.2
ARG TRL_VERSION=0.22.2
ARG DATASETS_VERSION=4.3.0
# Unsloth requires peft>=0.18.0 (!=0.11.0); force-reinstall must not pull us below that.
ARG PEFT_VERSION=0.20.0
ARG ACCELERATE_VERSION=1.14.0
ARG BITSANDBYTES_VERSION=0.50.0
ARG OGPU_VERSION=0.2.1

# Keep TORCH_* out of early ENV layers so dependency layers stay cacheable.
# Runtime compat (incl. Windows/WSL Triton safety) is applied by platform_compat
# + optional compose env; defaults are also set after COPY below.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir packaging setuptools wheel

# CUDA torch from the official channel only — never the default PyPI resolver.
RUN pip install --no-cache-dir \
    "torch==${TORCH_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"

# xformers must match cu126 + this torch line (not a random PyPI build).
RUN pip install --no-cache-dir \
    "xformers==${XFORMERS_VERSION}" \
    --index-url "${XFORMERS_INDEX_URL}"

# Pinned HF/TRL stack first so Unsloth cannot float us onto a breaking TRL.
RUN pip install --no-cache-dir \
    "transformers==${TRANSFORMERS_VERSION}" \
    "trl==${TRL_VERSION}" \
    "datasets==${DATASETS_VERSION}" \
    "peft==${PEFT_VERSION}" \
    "accelerate==${ACCELERATE_VERSION}" \
    "bitsandbytes==${BITSANDBYTES_VERSION}" \
    s3fs \
    fsspec \
    --extra-index-url "${TORCH_INDEX_URL}"

# Unsloth + zoo at lock versions. Then re-assert pins only if deps drifted
# (avoids blind ~2GB torch/xformers re-download when versions already match).
RUN pip install --no-cache-dir \
    "unsloth_zoo==${UNSLOTH_ZOO_VERSION}" \
    "unsloth==${UNSLOTH_VERSION}" \
    --extra-index-url "${TORCH_INDEX_URL}" \
    && TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION}" \
       TRL_VERSION="${TRL_VERSION}" \
       DATASETS_VERSION="${DATASETS_VERSION}" \
       PEFT_VERSION="${PEFT_VERSION}" \
       ACCELERATE_VERSION="${ACCELERATE_VERSION}" \
       BITSANDBYTES_VERSION="${BITSANDBYTES_VERSION}" \
       UNSLOTH_VERSION="${UNSLOTH_VERSION}" \
       UNSLOTH_ZOO_VERSION="${UNSLOTH_ZOO_VERSION}" \
       XFORMERS_VERSION="${XFORMERS_VERSION}" \
       TORCH_VERSION="${TORCH_VERSION}" \
       TORCH_INDEX_URL="${TORCH_INDEX_URL}" \
       XFORMERS_INDEX_URL="${XFORMERS_INDEX_URL}" \
       python - <<'PY'
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version as pkg_version

from packaging.version import Version


def installed(name: str) -> str | None:
    try:
        return pkg_version(name)
    except PackageNotFoundError:
        return None


def matches_pin(got: str | None, want: str) -> bool:
    """True if installed version equals pin (torch may be 2.9.1+cu126)."""
    if got is None:
        return False
    base = got.split("+", 1)[0]
    if base == want or got == want:
        return True
    try:
        return Version(base) == Version(want)
    except Exception:
        return False


def pip(*args: str) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


# Small stack: reinstall with --no-deps only when drifted.
nodeps_pins = {
    "transformers": os.environ["TRANSFORMERS_VERSION"],
    "trl": os.environ["TRL_VERSION"],
    "datasets": os.environ["DATASETS_VERSION"],
    "peft": os.environ["PEFT_VERSION"],
    "accelerate": os.environ["ACCELERATE_VERSION"],
    "bitsandbytes": os.environ["BITSANDBYTES_VERSION"],
    "unsloth": os.environ["UNSLOTH_VERSION"],
    "unsloth_zoo": os.environ["UNSLOTH_ZOO_VERSION"],
}
drifted = []
for name, want in nodeps_pins.items():
    got = installed(name)
    if matches_pin(got, want):
        print(f"pin guard: {name} ok ({got})")
    else:
        print(f"pin guard: {name} drift got={got!r} want={want!r}")
        drifted.append(f"{name}=={want}")
if drifted:
    pip("--force-reinstall", "--no-deps", *drifted)
else:
    print("pin guard: HF/TRL/unsloth stack already at pins")

# Heavy wheels: only reinstall when version actually changed.
for name, env_key, index_key in (
    ("xformers", "XFORMERS_VERSION", "XFORMERS_INDEX_URL"),
    ("torch", "TORCH_VERSION", "TORCH_INDEX_URL"),
):
    want = os.environ[env_key]
    index = os.environ[index_key]
    got = installed(name)
    if matches_pin(got, want):
        print(f"pin guard: {name} ok ({got}) — skip re-download")
    else:
        print(f"pin guard: {name} drift got={got!r} want={want!r} — reinstall")
        pip("--force-reinstall", f"{name}=={want}", "--index-url", index)

try:
    subprocess.check_call([sys.executable, "-m", "pip", "check"])
except subprocess.CalledProcessError:
    print("pin guard: pip check reported issues (non-fatal if pins match)", flush=True)
PY

RUN pip install --no-cache-dir "ogpu[service]>=${OGPU_VERSION}"

# Reproducibility gate (no GPU required on the build host).
# `import unsloth` needs a visible accelerator (unsloth_zoo device probe at
# import time), so the SFTConfig smoke lives outside the build.
RUN python - <<'PY'
import importlib.metadata as md
import torch

cuda_ver = getattr(torch.version, "cuda", None)
print("torch", torch.__version__, "cuda", cuda_ver, "hip", getattr(torch.version, "hip", None))
if not cuda_ver:
    raise SystemExit(
        "FATAL: torch.version.cuda is None — this is not a CUDA wheel. "
        "Refuse to publish a CPU/generic torch as the NVIDIA worker image."
    )
for pkg in ("transformers", "trl", "datasets", "peft", "accelerate",
            "bitsandbytes", "unsloth", "unsloth_zoo", "xformers"):
    print(pkg, md.version(pkg))
print("build-time sanity OK: CUDA torch + unsloth stack installed")
PY

# Job I/O is bind-mounted at /workspace (/data on the provider host).
# Keep application code outside that mount so provider volumes never shadow
# the entrypoint (a previous bug wiped /workspace/worker.py at runtime).
WORKDIR /workspace

COPY worker.py runtime_probe.py training_config.py display_gpu_guard.py platform_compat.py pause_guard.py unsloth_train.py /app/
COPY scripts/ /app/scripts/
# Defaults safe on Linux + Windows/WSL2 Docker Desktop. Worker may reinforce via
# platform_compat autodetection (OPENREEF_FORCE_EAGER_TORCH=0 to opt out).
ENV PYTHONPATH=/app \
    TORCH_DISABLE_NATIVE_JIT=1 \
    TORCHDYNAMO_DISABLE=1 \
    TORCH_COMPILE_DISABLE=1

ENTRYPOINT ["python", "/app/worker.py"]
