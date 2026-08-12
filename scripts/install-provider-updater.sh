#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compose_file=""
project_dir=""
control_dir="/data/openreef-control"

usage() {
  echo "Usage: sudo $0 --compose-file PATH --project-directory PATH [--control-directory PATH]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --compose-file) compose_file="${2:?compose path required}"; shift 2 ;;
    --project-directory) project_dir="${2:?project directory required}"; shift 2 ;;
    --control-directory) control_dir="${2:?control directory required}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

if [[ "$EUID" -ne 0 ]]; then
  echo "Run this installer with sudo." >&2
  exit 1
fi
[[ -n "$compose_file" && "$compose_file" == /* ]] || { usage; exit 2; }
[[ -n "$project_dir" && "$project_dir" == /* ]] || { usage; exit 2; }
[[ "$control_dir" == /* ]] || { usage; exit 2; }
[[ -f "$compose_file" ]] || { echo "Compose file not found: $compose_file" >&2; exit 2; }

install -d -m 0755 /usr/local/lib/openreef /etc/openreef "$control_dir"
install -m 0755 "$root/scripts/provider-safe-update.sh" /usr/local/lib/openreef/provider-safe-update.sh
install -m 0644 "$root/systemd/openreef-provider-update.service" /etc/systemd/system/
install -m 0644 "$root/systemd/openreef-provider-update.timer" /etc/systemd/system/

env_file=/etc/openreef/provider-updater.env
if [[ ! -e "$env_file" ]]; then
  {
    printf 'OPENREEF_COMPOSE_FILE=%s\n' "$compose_file"
    printf 'OPENREEF_PROJECT_DIR=%s\n' "$project_dir"
    printf 'OPENREEF_CONTROL_DIR=%s\n' "$control_dir"
    printf 'OPENREEF_UPDATE_DRAIN_GRACE_SECONDS=10\n'
    printf 'OPENREEF_UPDATE_HEALTH_SECONDS=90\n'
  } >"$env_file"
  chmod 0644 "$env_file"
else
  echo "Preserving existing $env_file"
fi

systemctl daemon-reload
systemctl enable --now openreef-provider-update.timer
systemctl start openreef-provider-update.service
systemctl --no-pager status openreef-provider-update.timer
