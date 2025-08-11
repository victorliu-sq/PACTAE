#!/usr/bin/env bash
# Only enable strict mode when this file is executed, NOT when it's sourced.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
REQ_FILE="${ROOT_DIR}/requirements.txt"

echo "[setup] Root: $ROOT_DIR"
echo "[setup] Venv: $VENV_DIR"
echo

# Create venv if missing (do this in a subshell with strict mode so failures don't nuke your interactive shell)
(
  set -euo pipefail
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "[setup] Creating venv..."
    python3 -m venv "$VENV_DIR"
  fi
)

# Activate venv in the CURRENT shell iff we're being sourced.
# If we're executed (not sourced), we still activate for the duration of this script.
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# Upgrade tools (don't hide errors; show output so you can debug)
python3 -m pip install --upgrade pip wheel

# Install requirements if present, otherwise tell the user
if [[ -f "$REQ_FILE" ]]; then
  echo "[setup] Installing Python dependencies from requirements.txt..."
  python3 -m pip install -r "$REQ_FILE"
else
  echo "[setup] No requirements.txt found — skipping pip install."
  echo "         (You can create one with: python3 -m pip freeze > requirements.txt)"
fi

# Helpful confirmation
echo
echo "[setup] Python: $(command -v python3)"
echo "[setup] Pip:    $(command -v pip3)"
echo "[setup] VENV:   ${VIRTUAL_ENV:-<not active>}"
