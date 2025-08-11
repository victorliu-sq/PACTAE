#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# If you're inside the venv, exit it so we can delete .venv on Linux/macOS safely.
deactivate 2>/dev/null || true

DIRS=(
  "${ROOT_DIR}/.venv"
  "${ROOT_DIR}/cmake-build-release"
  "${ROOT_DIR}/.ae"
  "${ROOT_DIR}/tmp"
  "${ROOT_DIR}/data"
)

for d in "${DIRS[@]}"; do
  if [[ -e "$d" ]]; then
    echo "[clean] Removing: $d"
    rm -rf "$d"
  else
    echo "[clean] Not found: $d"
  fi
done

echo "[clean] Done."
