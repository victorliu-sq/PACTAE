#!/usr/bin/env bash
set -euo pipefail

# -------- config / paths --------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# top (now split)
EXE_TOP_LEFT="${ROOT_DIR}/cmake-build-release/benchmark/figure_8_top_left"
EXE_TOP_RIGHT="${ROOT_DIR}/cmake-build-release/benchmark/figure_8_top_right"

PARSE_TOP="${ROOT_DIR}/scripts/f8_parse_top_times.py"
COLLECT_BOTTOM="${ROOT_DIR}/scripts/f8_collect_and_parse_bottom.sh"
VISUALIZE="${ROOT_DIR}/scripts/f8_visualize.py"

LOG_DIR="${ROOT_DIR}/tmp/logs"
LOG_LEFT="${LOG_DIR}/figure_8_top_left.INFO"
LOG_RIGHT="${LOG_DIR}/figure_8_top_right.INFO"

METRICS_DIR="${ROOT_DIR}/tmp/metrics"
FIG_DIR="${ROOT_DIR}/data/figures"

# -------- helpers --------
die() { echo "ERROR: $*" >&2; exit 1; }

echo "==> BambooSMPAE Figure 8 workflow"
echo "ROOT:        ${ROOT_DIR}"
echo "LOG_DIR:     ${LOG_DIR}"
echo "METRICS_DIR: ${METRICS_DIR}"
echo "FIG_DIR:     ${FIG_DIR}"
echo

# Ensure dirs exist
mkdir -p "${LOG_DIR}" "${METRICS_DIR}" "${FIG_DIR}"

# Sanity checks
[[ -x "${EXE_TOP_LEFT}"  ]] || die "Missing or non-executable: ${EXE_TOP_LEFT}"
[[ -x "${EXE_TOP_RIGHT}" ]] || die "Missing or non-executable: ${EXE_TOP_RIGHT}"
command -v python3 >/dev/null 2>&1 || die "python3 not found in PATH"
[[ -f "${PARSE_TOP}" ]] || die "Missing script: ${PARSE_TOP}"
[[ -f "${COLLECT_BOTTOM}" ]] || die "Missing script: ${COLLECT_BOTTOM}"
[[ -f "${VISUALIZE}" ]] || die "Missing script: ${VISUALIZE}"

# -------- step 1: generate logs for top data --------
echo "==> [1/4] Running top executables to generate timing logs..."
( cd "${ROOT_DIR}" && "${EXE_TOP_LEFT}" )
( cd "${ROOT_DIR}" && "${EXE_TOP_RIGHT}" )
[[ -s "${LOG_LEFT}"  ]] || die "Expected log not found (or empty): ${LOG_LEFT}"
[[ -s "${LOG_RIGHT}" ]] || die "Expected log not found (or empty): ${LOG_RIGHT}"
echo "[OK] Top logs:"
echo "     - ${LOG_LEFT}"
echo "     - ${LOG_RIGHT}"
echo

# -------- step 2: parse top timing logs --------
echo "==> [2/4] Parsing top timing logs -> tmp/metrics/f8_top_times.json ..."
# Pass both INFO files explicitly; parser combines them into one JSON.
python3 "${PARSE_TOP}" --left "${LOG_LEFT}" --right "${LOG_RIGHT}"
echo "[OK] Parsed: tmp/metrics/f8_top_times.json"
echo

# -------- step 3: collect + parse bottom metrics (ncu) --------
echo "==> [3/4] Running bottom collection & parsing (this may take a while)..."
bash "${COLLECT_BOTTOM}"
echo "[OK] Bottom metrics collected and parsed into ${METRICS_DIR}"
echo

# -------- step 4: visualize --------
echo "==> [4/4] Visualizing to ${FIG_DIR}/figure_8.png ..."
python3 "${VISUALIZE}"
echo "[OK] Figure saved to ${FIG_DIR}/figure_8.png"
echo
echo "✅ Done."