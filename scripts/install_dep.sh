#!/usr/bin/env bash
set -euo pipefail

# ---- Config (user-overridable via env) ----
AE_HOME="${AE_HOME:-$PWD/.ae}"              # where we keep stamps
AE_TMP_DIR="${AE_TMP_DIR:-$AE_HOME/tmp}"    # temp downloads/build area
AE_DEPS_DIR="${AE_DEPS_DIR:-$AE_HOME/deps}" # install prefix for deps
GLOG_VER="${GLOG_VER:-0.6.0}"
GFLAGS_VER="${GFLAGS_VER:-2.2.2}"

STAMP="$AE_HOME/.glog-$GLOG_VER"
GFLAGS_STAMP="$AE_HOME/.gflags-$GFLAGS_VER"

mkdir -p "$AE_HOME" "$AE_TMP_DIR" "$AE_DEPS_DIR"

# downloader
download() {
  local url="$1" out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$out"
  elif command -v wget >/dev/null 2>&1; then
    wget "$url" -O "$out"
  else
    echo "ERROR: need curl or wget" >&2
    exit 1
  fi
}

# parallelism
JOBS=1
if command -v nproc >/dev/null 2>&1; then
  JOBS="$(nproc)"
elif [[ "$(uname -s)" == "Darwin" ]]; then
  JOBS="$(sysctl -n hw.ncpu)"
fi

# --- Install gflags (once) ----------------------------------------------------
if [[ ! -f "$GFLAGS_STAMP" ]]; then
  echo "[gflags] Installing gflags v$GFLAGS_VER to $AE_DEPS_DIR ..."
  pushd "$AE_TMP_DIR" >/dev/null

  TARBALL="gflags-$GFLAGS_VER.tar.gz"
  URL="https://github.com/gflags/gflags/archive/refs/tags/v$GFLAGS_VER.tar.gz"
  download "$URL" "$TARBALL"

  rm -rf "gflags-$GFLAGS_VER"
  tar zxf "$TARBALL"
  pushd "gflags-$GFLAGS_VER" >/dev/null

  rm -rf build
  mkdir -p build && cd build

  cmake -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_INSTALL_PREFIX="$AE_DEPS_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        ..
  cmake --build . --target install -j "$JOBS"

  popd >/dev/null # gflags-$GFLAGS_VER
  popd >/dev/null # tmp

  touch "$GFLAGS_STAMP"
  echo "[gflags] Installed."
else
  echo "[gflags] Found stamp $GFLAGS_STAMP, skipping install."
fi

# --- Install glog (using .ae/deps so it finds gflags) -------------------------
if [[ ! -f "$STAMP" ]]; then
  echo "[glog] Installing glog v$GLOG_VER to $AE_DEPS_DIR ..."
  pushd "$AE_TMP_DIR" >/dev/null

  TARBALL="glog-$GLOG_VER.tar.gz"
  URL="https://github.com/google/glog/archive/refs/tags/v$GLOG_VER.tar.gz"
  download "$URL" "$TARBALL"

  rm -rf "glog-$GLOG_VER"
  tar zxf "$TARBALL"
  pushd "glog-$GLOG_VER" >/dev/null

  rm -rf build
  mkdir -p build && cd build

  cmake \
    -DBUILD_SHARED_LIBS=ON \
    -DWITH_GTEST=OFF \
    -DCMAKE_INSTALL_PREFIX="$AE_DEPS_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$AE_DEPS_DIR" \
    ..

  # --- Print whether glog found gflags before building ---
  echo "[check] glog CMakeCache (gflags):"
  grep -i -E 'WITH_GFLAGS|GFLAGS|gflags_DIR' CMakeCache.txt || true
  echo

  cmake --build . --target install -j "$JOBS"

  popd >/dev/null # glog-$GLOG_VER
  popd >/dev/null # tmp

  touch "$STAMP"
  echo "[glog] Installed."
else
  echo "[glog] Found stamp $STAMP, skipping install."
fi

# --- Post-install sanity: does installed glog require gflags & is it present? -
REQ=$([[ -d "$AE_DEPS_DIR/include/gflags" ]] && echo "present" || echo "missing")
INC=$([[ -n "$(grep -R "gflags/gflags.h" "$AE_DEPS_DIR/include/glog" 2>/dev/null || true)" ]] && echo "YES" || echo "NO")
echo
echo "[summary]"
echo "  gflags headers in prefix: $REQ ($AE_DEPS_DIR/include/gflags)"
echo "  glog headers include <gflags/gflags.h>: $INC"

echo
echo "[glog] To build your project, set CMAKE_PREFIX_PATH or export it:"
echo "  cmake -S . -B build -DCMAKE_PREFIX_PATH=\"$AE_DEPS_DIR\" -DCMAKE_BUILD_TYPE=Release"
echo "  cmake --build build -j"