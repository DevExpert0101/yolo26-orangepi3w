#!/usr/bin/env bash
# Build YOLO26 ONNX Runtime C++ app on Orange Pi Zero 3W (aarch64).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
ORT="${ONNXRUNTIME_ROOT:-${ROOT}/third_party/onnxruntime}"

if [[ ! -f "${ORT}/include/onnxruntime_cxx_api.h" || ! -e "${ORT}/lib/libonnxruntime.so" ]]; then
  echo "ONNX Runtime missing under ${ORT}"
  echo "On the board:  cd .. && ./setup_onnx.sh"
  echo "Or:            ./fetch_onnxruntime.sh"
  exit 1
fi

mkdir -p "${ROOT}/build"
cd "${ROOT}/build"
cmake .. -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT="${ORT}"
cmake --build . -j"$(nproc)"
echo
echo "Run:"
echo "  export LD_LIBRARY_PATH=${ORT}/lib:\${LD_LIBRARY_PATH}"
echo "  ./yolo26_onnx /path/to/yolo26n.onnx /path/to/bus.jpg --no-show --save result.jpg"
echo "Or from the project root:  ./run_onnx.sh"
