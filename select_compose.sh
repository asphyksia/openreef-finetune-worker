#!/usr/bin/env bash
# Print which compose file / image a provider should use (autodetect GPU).
#
# Usage:
#   ./select_compose.sh
#   ./select_compose.sh --multi-gpu
#   ./select_compose.sh --url-base https://openreef.network/source
#
# Exit 0 if a GPU backend was selected; 1 if none.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URL_BASE=""
MULTI_GPU=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url-base) URL_BASE="${2:-}"; shift 2 ;;
    --multi-gpu) MULTI_GPU=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--multi-gpu] [--url-base https://openreef.network/source]"
      exit 0
      ;;
    *) echo "Unknown: $1" >&2; exit 1 ;;
  esac
done

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 required" >&2
  exit 1
fi

JSON="$(python3 "$ROOT/platform_compat.py" host)"
BACKEND="$(printf '%s' "$JSON" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("backend",""))')"
COMPOSE="$(printf '%s' "$JSON" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("compose_file") or "")')"
IMAGE="$(printf '%s' "$JSON" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("image_tag") or "")')"

if [[ "$MULTI_GPU" -eq 1 && "$BACKEND" == "nvidia_cuda" ]]; then
  COMPOSE="docker-compose-nvidia-multigpu.yml"
  IMAGE="ghcr.io/asphyksia/finetune-worker:cuda-multigpu-latest"
fi

echo "backend=$BACKEND"
echo "compose_file=$COMPOSE"
echo "image=$IMAGE"
if [[ -n "$URL_BASE" && -n "$COMPOSE" ]]; then
  echo "compose_url=${URL_BASE%/}/$COMPOSE"
fi
if [[ "$MULTI_GPU" -eq 1 && "$BACKEND" == "amd_rocm" ]]; then
  echo "note: AMD already uses the Axolotl ROCm image; ensure all intended GPUs are visible."
fi
printf '%s' "$JSON" | python3 -c 'import sys,json; notes=json.load(sys.stdin).get("notes") or [];
[print("note:", n) for n in notes]'

[[ "$BACKEND" != "cpu" && -n "$COMPOSE" ]]
