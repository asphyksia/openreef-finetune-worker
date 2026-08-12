#!/usr/bin/env bash
# Load pins-cuda.env into the current shell and emit docker --build-arg flags.
# Usage:
#   source scripts/load_cuda_pins.sh           # exports KEY=VAL
#   eval "$(scripts/load_cuda_pins.sh --args)" # prints: --build-arg K=V ...
#   scripts/load_cuda_pins.sh --export-file /tmp/pins.env

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PINS="${OPENREEF_CUDA_PINS:-$ROOT/pins-cuda.env}"

if [[ ! -f "$PINS" ]]; then
  echo "load_cuda_pins: missing $PINS" >&2
  exit 1
fi

# Keys the Dockerfile accepts as ARG (complete lock).
PIN_KEYS=(
  UNSLOTH_VERSION
  UNSLOTH_ZOO_VERSION
  TORCH_VERSION
  TORCH_CUDA_CHANNEL
  TORCH_INDEX_URL
  XFORMERS_VERSION
  XFORMERS_INDEX_URL
  TRANSFORMERS_VERSION
  TRL_VERSION
  DATASETS_VERSION
  PEFT_VERSION
  ACCELERATE_VERSION
  BITSANDBYTES_VERSION
  OGPU_VERSION
)

# shellcheck disable=SC1090
set -a
# shellcheck source=/dev/null
source "$PINS"
set +a

MODE="${1:-}"
if [[ "$MODE" == "--args" ]]; then
  out=()
  for k in "${PIN_KEYS[@]}"; do
    if [[ -n "${!k:-}" ]]; then
      out+=(--build-arg "${k}=${!k}")
    fi
  done
  printf '%q ' "${out[@]}"
  echo
  exit 0
fi

if [[ "$MODE" == "--export-file" ]]; then
  dest="${2:?export-file path required}"
  : >"$dest"
  for k in "${PIN_KEYS[@]}"; do
    if [[ -n "${!k:-}" ]]; then
      printf '%s=%s\n' "$k" "${!k}" >>"$dest"
    fi
  done
  exit 0
fi

if [[ "$MODE" == "--gha" ]]; then
  # Emit multiline build-args for docker/build-push-action.
  for k in "${PIN_KEYS[@]}"; do
    if [[ -n "${!k:-}" ]]; then
      printf '%s=%s\n' "$k" "${!k}"
    fi
  done
  exit 0
fi

# Default when sourced: variables already exported.
return 0 2>/dev/null || true
