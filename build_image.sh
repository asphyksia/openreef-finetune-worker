#!/usr/bin/env bash
# Build (and optionally push) the OpenReef finetune-worker image with autodetection.
#
# Works on: Linux, macOS (Docker), Windows (Git Bash / WSL) with Docker installed.
# Does NOT require a GPU on the build machine — only Docker.
#
# Usage:
#   ./build_image.sh                 # auto-detect NVIDIA vs AMD on this host
#   ./build_image.sh --cuda          # force CUDA image
#   ./build_image.sh --rocm          # force ROCm image
#   ./build_image.sh --cuda-multigpu # force CUDA Axolotl multi-GPU image
#   ./build_image.sh --cuda --tag cuda-0123456789abcdef --push
#   ./build_image.sh --cuda --tag cuda-0123456789abcdef --push \
#     --publish-latest --confirm-gpu-smoke
#   OPENREEF_IMAGE_REPO=ghcr.io/you/finetune-worker ./build_image.sh --cuda
#
# Note: one image cannot contain both CUDA and ROCm stacks. Autodetect chooses
# which Dockerfile to build, not a mythical universal GPU image.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

REPO="${OPENREEF_IMAGE_REPO:-ghcr.io/asphyksia/finetune-worker}"
BACKEND=""
PUSH=0
EXTRA_TAG=""
PUBLISH_LATEST=0
CONFIRM_GPU_SMOKE=0

usage() {
  sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda|--nvidia)       BACKEND=nvidia_cuda; shift ;;
    --rocm|--amd)          BACKEND=amd_rocm; shift ;;
    --cuda-multigpu)       BACKEND=nvidia_cuda_multigpu; shift ;;
    --push)          PUSH=1; shift ;;
    --tag)           EXTRA_TAG="${2:-}"; shift 2 ;;
    --publish-latest) PUBLISH_LATEST=1; shift ;;
    --confirm-gpu-smoke) CONFIRM_GPU_SMOKE=1; shift ;;
    -h|--help)       usage 0 ;;
    *) echo "Unknown arg: $1" >&2; usage 1 ;;
  esac
done

if [[ -z "$BACKEND" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    BACKEND="$(python3 platform_compat.py host 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin).get("backend",""))' 2>/dev/null || true)"
  fi
  if [[ -z "$BACKEND" || "$BACKEND" == "cpu" ]]; then
    # Fallbacks without Python JSON path
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
      BACKEND=nvidia_cuda
    elif command -v rocminfo >/dev/null 2>&1 && rocminfo >/dev/null 2>&1; then
      BACKEND=amd_rocm
    else
      echo "No GPU backend detected. Pass --cuda or --rocm explicitly." >&2
      echo "  (macOS build machines should use --cuda or --rocm; training needs Linux/WSL2 GPU.)" >&2
      exit 1
    fi
  fi
fi

case "$BACKEND" in
  nvidia_cuda|cuda|nvidia)
    DOCKERFILE=Dockerfile
    TAG_ROLLING="${REPO}:cuda-latest"
    LABEL=cuda
    ;;
  amd_rocm|rocm|amd)
    DOCKERFILE=Dockerfile.rocm
    TAG_ROLLING="${REPO}:rocm-latest"
    LABEL=rocm
    ;;
  nvidia_cuda_multigpu|cuda-multigpu)
    DOCKERFILE=Dockerfile.axolotl.cuda
    TAG_ROLLING="${REPO}:cuda-multigpu-latest"
    LABEL=cuda-multigpu
    ;;
  *)
    echo "Unsupported backend: $BACKEND" >&2
    exit 1
    ;;
esac

if [[ "$PUBLISH_LATEST" -eq 1 && "$PUSH" -ne 1 ]]; then
  echo "--publish-latest requires --push." >&2
  exit 1
fi
if [[ "$PUSH" -eq 1 ]]; then
  if [[ ! "$EXTRA_TAG" =~ ^${LABEL}-[0-9a-f]{7,40}$ ]]; then
    echo "--push requires an immutable SHA tag: --tag ${LABEL}-<7-40 lowercase hex>." >&2
    exit 1
  fi
  TAGS=(-t "${REPO}:${EXTRA_TAG}")
  if [[ "$PUBLISH_LATEST" -eq 1 ]]; then
    if [[ "$CONFIRM_GPU_SMOKE" -ne 1 ]]; then
      echo "Rolling-tag publication requires --confirm-gpu-smoke." >&2
      exit 1
    fi
    TAGS+=(-t "$TAG_ROLLING")
  fi
else
  LOCAL_TAG="${REPO}:${LABEL}-local"
  TAGS=(-t "$LOCAL_TAG")
  if [[ -n "$EXTRA_TAG" ]]; then
    TAGS+=(-t "${REPO}:${EXTRA_TAG}")
  fi
fi

echo "==> Building OpenReef finetune-worker ($LABEL)"
echo "    dockerfile=$DOCKERFILE"
echo "    tags=${TAGS[*]}"
echo "    host=$(uname -s)/$(uname -m) docker=$(docker version --format '{{.Server.Os}}/{{.Server.Arch}}' 2>/dev/null || echo unknown)"

PIN_ARGS=()
if [[ "$LABEL" == "cuda" ]]; then
  while IFS= read -r pin; do
    [[ -n "$pin" ]] && PIN_ARGS+=(--build-arg "$pin")
  done < <(scripts/load_cuda_pins.sh --gha)
fi

BUILD_ARGS=(buildx build -f "$DOCKERFILE" "${TAGS[@]}" "${PIN_ARGS[@]}" --progress=plain)
if [[ "$PUSH" -eq 1 ]]; then
  BUILD_ARGS+=(--push)
else
  BUILD_ARGS+=(--load)
fi
BUILD_ARGS+=(.)

docker "${BUILD_ARGS[@]}"

echo "==> Done: ${TAGS[*]}"
if [[ "$PUSH" -eq 1 ]]; then
  echo "    Pushed immutable tag: ${REPO}:${EXTRA_TAG}"
  if [[ "$PUBLISH_LATEST" -eq 1 ]]; then
    echo "    Promoted rolling tag after explicit GPU-smoke confirmation: $TAG_ROLLING"
  fi
fi
