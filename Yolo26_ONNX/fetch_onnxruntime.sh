#!/usr/bin/env bash
# Download official ONNX Runtime CPU (C/C++ API) for this machine.
# Orange Pi Zero 3W = aarch64. Also works on x86_64 Linux for a desktop smoke test.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
VER="${ONNXRUNTIME_VERSION:-1.24.4}"
DEST="${ROOT}/third_party/onnxruntime"
ARCH="$(uname -m)"

case "${ARCH}" in
  aarch64|arm64) PKG="onnxruntime-linux-aarch64-${VER}" ;;
  x86_64|amd64)  PKG="onnxruntime-linux-x64-${VER}" ;;
  *)
    echo "No official ONNX Runtime tarball for ${ARCH}."
    echo "Set ONNXRUNTIME_ROOT to a built install (include/ + lib/)."
    exit 1
    ;;
esac

URL="https://github.com/microsoft/onnxruntime/releases/download/v${VER}/${PKG}.tgz"
TMP="${ROOT}/third_party"
mkdir -p "${TMP}"

if [[ -f "${DEST}/include/onnxruntime_cxx_api.h" && -e "${DEST}/lib/libonnxruntime.so" ]]; then
  echo "ONNX Runtime already present: ${DEST}"
  exit 0
fi

echo "==> ${URL}"
ARCHIVE="${TMP}/${PKG}.tgz"
if command -v wget >/dev/null 2>&1; then
  wget -q --show-progress -O "${ARCHIVE}" "${URL}"
else
  curl -L --fail -o "${ARCHIVE}" "${URL}"
fi

rm -rf "${TMP}/${PKG}" "${DEST}"
tar -xzf "${ARCHIVE}" -C "${TMP}"
mv "${TMP}/${PKG}" "${DEST}"
rm -f "${ARCHIVE}"

if [[ ! -f "${DEST}/include/onnxruntime_cxx_api.h" ]]; then
  echo "Extracted tree is missing include/onnxruntime_cxx_api.h"
  exit 1
fi
echo "Installed ${PKG} -> ${DEST}"
echo "Headers: ${DEST}/include"
echo "Library: ${DEST}/lib/libonnxruntime.so"
