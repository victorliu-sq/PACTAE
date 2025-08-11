#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

echo "[check] Running prerequisite checks..."
bash "${ROOT_DIR}/scripts/check_requirements.sh"

echo "=== BambooSMPAE: bootstrap ==="

# 3) Create/populate venv
echo "[venv] Setting up Python environment..."
bash "${ROOT_DIR}/workflow/setup_venv.sh"

# 4) Activate venv for this shell so child scripts use it
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
echo "[venv] Using $(python3 --version) at $(command -v python3)"

# 0) Re-create directories that workflows expect
echo "[prep] Ensuring directory layout exists"
mkdir -p \
  "${ROOT_DIR}/.ae" \
  "${ROOT_DIR}/tmp/logs" \
  "${ROOT_DIR}/tmp/metrics" \
  "${ROOT_DIR}/data/figures" \
  "${ROOT_DIR}/data/parsed" \
  "${ROOT_DIR}/data/workloads"

# 1) Install native deps (e.g., glog) via your script
if [[ -x "${ROOT_DIR}/scripts/install_dep.sh" ]]; then
  echo "[deps] Running scripts/install_dep.sh ..."
  bash "${ROOT_DIR}/scripts/install_dep.sh"
else
  echo "[deps] WARNING: scripts/install_dep.sh not found or not executable; skipping."
fi

# 2) Build the selected benchmark targets (Release)
echo "[build] Building release targets..."
bash "${ROOT_DIR}/workflow/build_release.sh"


# 5) Run all figure workflows
echo "[run] Executing workflows..."
bash "${ROOT_DIR}/workflow/all_figures.sh"

echo "=== Done ==="