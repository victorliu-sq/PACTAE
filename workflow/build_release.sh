#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/cmake-build-release"

echo "=== Configure & Build (Release) ==="
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure (you can add EXTRA_CMAKE_ARGS when calling this script)
cmake -DCMAKE_BUILD_TYPE=Release ${EXTRA_CMAKE_ARGS:-} ..

# Explicit benchmark targets only (must match your CMake target names)
TARGETS=(
  figure_3
  figure_5
  figure_7_bottom
  smp_executor_f7
  figure_8_bottom
  figure_8_top_left
  figure_8_top_right
  figure_9
  table_1
)

# Build only these targets; no --config flag
JOBS="$(command -v nproc >/dev/null 2>&1 && nproc || echo 8)"
cmake --build . -j"${JOBS}" --target "${TARGETS[@]}"

echo "=== Build complete ==="
