#!/usr/bin/env bash
set -euo pipefail

# ---- Config (user-overridable via env) ----
AE_HOME="${AE_HOME:-$PWD/.ae}"              # where we keep stamps
AE_TMP_DIR="${AE_TMP_DIR:-$AE_HOME/tmp}"    # temp downloads/build area
AE_DEPS_DIR="${AE_DEPS_DIR:-$AE_HOME/deps}" # install prefix for deps
GLOG_VER="${GLOG_VER:-0.6.0}"
STAMP="$AE_HOME/.glog-$GLOG_VER"

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
    ..

  cmake --build . --target install -j "$JOBS"

  popd >/dev/null # glog-$GLOG_VER
  popd >/dev/null # tmp

  touch "$STAMP"
  echo "[glog] Installed."
else
  echo "[glog] Found stamp $STAMP, skipping install."
fi

echo
echo "[glog] To build your project, set CMAKE_PREFIX_PATH or export it:"
echo "  cmake -S . -B build -DCMAKE_PREFIX_PATH=\"$AE_DEPS_DIR\" -DCMAKE_BUILD_TYPE=Release"
echo "  cmake --build build -j"