#!/usr/bin/env bash
# Build on the Orange Pi Zero 3W (or any aarch64 board with VIPLite + OpenCV).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "${ROOT}/lib" "${ROOT}/build"
cd "${ROOT}/build"

# Optional: copy board libs next to the project
#   cp /usr/lib/libNBGlinker.so /usr/lib/libVIPhal.so ../lib/

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  ${VIPLITE_ROOT:+-DVIPLITE_ROOT=$VIPLITE_ROOT} \
  ${VIP_INIT_HAS_SIZE:+-DVIP_INIT_HAS_SIZE=$VIP_INIT_HAS_SIZE}

cmake --build . -j"$(nproc)"
echo
echo "Run:"
echo "  ./yolo26_npu /path/to/yolo26n_6_fp16_a733.nb /path/to/bus.jpg --no-show --save result.jpg"
