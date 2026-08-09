# OpenReef Fine-Tune Worker — NVIDIA CUDA (Profile A)
#
# One NVIDIA source for essentially all modern GPUs. Keep this simpler than
# the ROCm image: no gfx-specific forks, just a reproducible CUDA stack.
#
# Pins (do not use floating `pip install torch`):
#   - CUDA base 12.6 runtime (matches torch cu126 wheels)
#   - torch 2.9.1+cu126 from the official PyTorch index
#   - axolotl 0.17.0 (requires torch>=2.9.1)
#
# Build-time checks do NOT require a GPU. torch.cuda.is_available() may be
# false on CI runners; we only assert the wheel is a CUDA build.

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ARG TORCH_VERSION=2.9.1
ARG TORCH_CUDA_CHANNEL=cu126
ARG AXOLOTL_VERSION=0.17.0
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
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_CHANNEL}"

# Axolotl without flash-attn (SDPA is the stable baseline for all NVIDIA GPUs).
# Keep the PyTorch CUDA index as extra so dependency resolution does not replace
# our pinned CUDA wheel with a CPU/generic PyPI build.
RUN pip install --no-cache-dir \
    "axolotl==${AXOLOTL_VERSION}" \
    s3fs \
    fsspec \
    --extra-index-url "https://download.pytorch.org/whl/${TORCH_CUDA_CHANNEL}"

RUN pip install --no-cache-dir "ogpu>=${OGPU_VERSION}"

# Reproducibility gate (no GPU required on the build host).
RUN python - <<'PY'
import importlib
import torch

cuda_ver = getattr(torch.version, "cuda", None)
print("torch", torch.__version__, "cuda", cuda_ver, "hip", getattr(torch.version, "hip", None))
if not cuda_ver:
    raise SystemExit(
        "FATAL: torch.version.cuda is None — this is not a CUDA wheel. "
        "Refuse to publish a CPU/generic torch as the NVIDIA worker image."
    )
if "+cu" not in torch.__version__ and "cuda" not in torch.__version__.lower():
    # Some wheels encode CUDA only in torch.version.cuda; version string is advisory.
    print("note: torch.__version__ has no +cu tag; relying on torch.version.cuda")

importlib.import_module("axolotl.cli.train")
print("runtime sanity OK: CUDA torch + axolotl.cli.train")
PY

# Job I/O is bind-mounted at /workspace (/data on the provider host).
# Keep application code outside that mount so provider volumes never shadow
# the entrypoint (a previous bug wiped /workspace/worker.py at runtime).
WORKDIR /workspace

COPY worker.py runtime_probe.py training_config.py display_gpu_guard.py platform_compat.py /app/
# Defaults safe on Linux + Windows/WSL2 Docker Desktop. Worker may reinforce via
# platform_compat autodetection (OPENREEF_FORCE_EAGER_TORCH=0 to opt out).
ENV PYTHONPATH=/app \
    TORCH_DISABLE_NATIVE_JIT=1 \
    TORCHDYNAMO_DISABLE=1 \
    TORCH_COMPILE_DISABLE=1

ENTRYPOINT ["python", "/app/worker.py"]
