#!/usr/bin/env bash
set -euo pipefail

# Minimum versions
MIN_GCC="11.4.0"
MIN_CMAKE="3.22.1"
MIN_PYTHON="3.10.0"
MIN_CUDA="12.6"   # parsed from nvcc
# perf/ncu/wget/curl don't have strict mins here—just require presence
# but we'll print detected versions.

# --- helpers ---
need_cmd() { command -v "$1" >/dev/null 2>&1; }

# return 0 if $1 >= $2 (semantic-ish compare)
version_ge() {
  # handles 1.2.3 vs 1.10.0 correctly
  [ "$(printf '%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]
}

fail() { echo "[check] ERROR: $*" >&2; exit 1; }

echo "[check] Verifying software prerequisites..."

# bash (we're already running under bash, just print version)
echo "[check] bash: $BASH_VERSION"

# wget or curl
if need_cmd wget; then
  WGET_VER=$(wget --version 2>/dev/null | head -n1 | awk '{print $3}')
  echo "[check] wget: ${WGET_VER:-found}"
elif need_cmd curl; then
  CURL_VER=$(curl --version 2>/dev/null | head -n1 | awk '{print $2}')
  echo "[check] curl: ${CURL_VER:-found}"
else
  fail "Need either 'wget' or 'curl' in PATH."
fi

# perf
need_cmd perf || fail "perf not found in PATH."
PERF_VER=$(perf --version 2>&1 | grep -oE '[0-9]+(\.[0-9]+)+' | head -n1 || true)
echo "[check] perf: ${PERF_VER:-found}"

# ncu
need_cmd ncu || fail "ncu (NVIDIA Nsight Compute CLI) not found in PATH."
NCU_VER=$(ncu --version 2>&1 | grep -oE '[0-9]+(\.[0-9]+)+' | head -n1 || true)
echo "[check] ncu: ${NCU_VER:-found}"

# gcc >= 13.2.0
need_cmd gcc || fail "gcc not found in PATH."
GCC_VER=$(gcc -dumpfullversion -dumpversion 2>/dev/null || echo "")
[ -n "$GCC_VER" ] || fail "Unable to determine gcc version."
version_ge "$GCC_VER" "$MIN_GCC" || fail "gcc >= $MIN_GCC required (found $GCC_VER)."
echo "[check] gcc: $GCC_VER (OK)"

# cmake >= 3.25.2
need_cmd cmake || fail "cmake not found in PATH."
CMAKE_VER=$(cmake --version | head -n1 | awk '{print $3}')
version_ge "$CMAKE_VER" "$MIN_CMAKE" || fail "CMake >= $MIN_CMAKE required (found $CMAKE_VER)."
echo "[check] cmake: $CMAKE_VER (OK)"

# python3 >= 3.10
need_cmd python3 || fail "python3 not found in PATH."
PY_VER=$(python3 -V 2>&1 | awk '{print $2}')
version_ge "$PY_VER" "$MIN_PYTHON" || fail "Python3 >= $MIN_PYTHON required (found $PY_VER)."
echo "[check] python3: $PY_VER (OK)"

# CUDA Toolkit (nvcc) >= 12.6.2
need_cmd nvcc || fail "nvcc (CUDA Toolkit) not found in PATH."
# Try to pull a clean version like 12.6.2 from `nvcc --version`
NVCC_VER=$(nvcc --version | grep -oE 'V[0-9.]+|release[[:space:]][0-9.]+' | head -n1 | tr -d 'V' | sed 's/release[[:space:]]//' )
[ -n "$NVCC_VER" ] || fail "Unable to determine CUDA (nvcc) version."
version_ge "$NVCC_VER" "$MIN_CUDA" || fail "CUDA Toolkit >= $MIN_CUDA required (found $NVCC_VER)."
echo "[check] nvcc: $NVCC_VER (OK)"

echo "[check] All prerequisites satisfied."