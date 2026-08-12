#!/usr/bin/env bash
set -euo pipefail

compose_file="${OPENREEF_COMPOSE_FILE:-}"
project_dir="${OPENREEF_PROJECT_DIR:-}"
control_dir="${OPENREEF_CONTROL_DIR:-/data/openreef-control}"
drain_grace="${OPENREEF_UPDATE_DRAIN_GRACE_SECONDS:-10}"
health_wait="${OPENREEF_UPDATE_HEALTH_SECONDS:-90}"

usage() {
  echo "Usage: $0 --compose-file PATH [--project-directory PATH]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --compose-file) compose_file="${2:?compose path required}"; shift 2 ;;
    --project-directory) project_dir="${2:?project directory required}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

[[ -n "$compose_file" && -f "$compose_file" ]] || { usage; exit 2; }
[[ "$drain_grace" =~ ^[0-9]+$ ]] || { echo "Invalid drain grace" >&2; exit 2; }
[[ "$health_wait" =~ ^[0-9]+$ ]] || { echo "Invalid health wait" >&2; exit 2; }

mkdir -p "$control_dir"
updating="$control_dir/updating"
active="$control_dir/training.active"
exec 9>"$control_dir/update.lock"
flock -n 9 || { echo "Another provider update is running" >&2; exit 75; }

cleanup() { rm -f "$updating"; }
trap cleanup EXIT INT TERM
printf 'Automated provider update in progress\n' >"$updating"
sleep "$drain_grace"

compose=(docker compose -f "$compose_file")
if [[ -n "$project_dir" ]]; then
  compose+=(--project-directory "$project_dir")
fi

if [[ -f "$active" ]]; then
  current_worker="$("${compose[@]}" ps -q finetune 2>/dev/null || true)"
  if [[ -n "$current_worker" ]] \
    && [[ "$(docker inspect -f '{{.State.Running}}' "$current_worker" 2>/dev/null || true)" == "true" ]]; then
    echo "Training is active; deferring update" >&2
    exit 75
  fi
  echo "Removing orphaned training marker from stopped worker" >&2
  rm -f "$active"
fi

old_finetune="$("${compose[@]}" images -q finetune 2>/dev/null || true)"
old_signer="$("${compose[@]}" images -q openreef-signer 2>/dev/null || true)"
rollback_suffix="$(date -u +%Y%m%d%H%M%S)"
rollback_finetune="openreef/finetune-rollback:$rollback_suffix"
rollback_signer="openreef/signer-rollback:$rollback_suffix"

rollback() {
  echo "Update health check failed; restoring previous images" >&2
  rollback_env=(env)
  rollback_services=()
  if [[ -n "$old_signer" ]]; then
    rollback_env+=("OPENREEF_SIGNER_IMAGE=$rollback_signer")
    rollback_services+=(openreef-signer)
  fi
  if [[ -n "$old_finetune" ]]; then
    rollback_env+=("FINETUNE_IMAGE=$rollback_finetune")
    rollback_services+=(finetune)
  fi
  if (( ${#rollback_services[@]} > 0 )); then
    "${rollback_env[@]}" "${compose[@]}" up -d --no-deps "${rollback_services[@]}"
  fi
}

prune_old_rollback_tags() {
  local repository="$1"
  mapfile -t old_refs < <(
    docker image ls "$repository" --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
      | sort -r \
      | tail -n +4
  )
  if (( ${#old_refs[@]} > 0 )); then
    docker image rm "${old_refs[@]}" >/dev/null 2>&1 || true
  fi
}

if ! "${compose[@]}" pull openreef-signer finetune; then
  echo "Image pull failed; existing provider containers were not replaced" >&2
  exit 1
fi
[[ -n "$old_finetune" ]] && docker image tag "$old_finetune" "$rollback_finetune"
[[ -n "$old_signer" ]] && docker image tag "$old_signer" "$rollback_signer"
if ! "${compose[@]}" up -d --no-deps openreef-signer finetune; then
  rollback
  exit 1
fi

deadline=$((SECONDS + health_wait))
while (( SECONDS < deadline )); do
  if [[ -f "$active" ]]; then
    echo "Unexpected training start during update" >&2
    rollback
    exit 1
  fi
  signer_id="$("${compose[@]}" ps -q openreef-signer)"
  worker_id="$("${compose[@]}" ps -q finetune)"
  if [[ -n "$signer_id" && -n "$worker_id" ]] \
    && [[ "$(docker inspect -f '{{.State.Running}}' "$signer_id")" == "true" ]] \
    && [[ "$(docker inspect -f '{{.State.Running}}' "$worker_id")" == "true" ]] \
    && docker exec "$signer_id" python -c \
      "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8787/health', timeout=3).read(1)" \
      >/dev/null 2>&1 \
    && docker exec "$worker_id" python -c \
      "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5555/docs', timeout=3).read(1)" \
      >/dev/null 2>&1; then
    echo "Provider images updated; rollback tags retained: $rollback_suffix"
    prune_old_rollback_tags openreef/finetune-rollback
    prune_old_rollback_tags openreef/signer-rollback
    exit 0
  fi
  sleep 5
done

rollback
exit 1
