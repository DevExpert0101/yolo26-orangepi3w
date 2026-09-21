#!/usr/bin/env bash
# Run the YOLO26 ONNX C++ app. Extra flags are forwarded (e.g. --conf 0.3 --threads 2).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
ORT="${ONNXRUNTIME_ROOT:-${ROOT}/Yolo26_ONNX/third_party/onnxruntime}"
BIN="${ROOT}/Yolo26_ONNX/build/yolo26_onnx"
MODEL="${ROOT}/export_onnx/yolo26n.onnx"
SOURCE="${ROOT}/export_onnx/bus.jpg"

if [[ $# -ge 1 && "${1}" != --* ]]; then
  MODEL="$1"
  shift
fi
if [[ $# -ge 1 && "${1}" != --* ]]; then
  SOURCE="$1"
  shift
fi

if [[ ! -x "${BIN}" ]]; then
  echo "Missing ${BIN}"
  echo "On the board:  ./setup_onnx.sh && cd Yolo26_ONNX && ./build.sh"
  exit 1
fi
if [[ ! -f "${MODEL}" ]]; then
  echo "Missing model ${MODEL}"
  echo "On a PC:  python prepare_onnx.py --model yolo26n.pt --outdir export_onnx"
  echo "Then copy export_onnx/ to this folder."
  exit 1
fi

export LD_LIBRARY_PATH="${ORT}/lib:${LD_LIBRARY_PATH:-}"
exec "${BIN}" "${MODEL}" "${SOURCE}" --no-show --save "${ROOT}/result.jpg" "$@"
