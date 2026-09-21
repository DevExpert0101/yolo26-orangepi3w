#!/usr/bin/env bash
# Run the YOLO26 ONNX C++ app. Extra flags are forwarded (e.g. --conf 0.3 --threads 2).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
ORT="${ONNXRUNTIME_ROOT:-${ROOT}/Yolo26_ONNX/third_party/onnxruntime}"
BIN="${ROOT}/Yolo26_ONNX/build/yolo26_onnx"

pick_first() {
  local p
  for p in "$@"; do
    if [[ -f "${p}" ]]; then
      printf '%s' "${p}"
      return 0
    fi
  done
  return 1
}

MODEL=""
SOURCE=""
if [[ $# -ge 1 && "${1}" != --* ]]; then
  MODEL="$1"
  shift
fi
if [[ $# -ge 1 && "${1}" != --* ]]; then
  SOURCE="$1"
  shift
fi

if [[ -z "${MODEL}" ]]; then
  MODEL="$(pick_first \
    "${ROOT}/export_onnx/yolo26n.onnx" \
    "${ROOT}/export/yolo26n_6.onnx" \
    "${ROOT}/export_nb/yolo26n_6.onnx" \
    "${ROOT}/export/yolo26s_6.onnx" \
    "${ROOT}/export_nb/yolo26s_6.onnx" || true)"
fi
if [[ -z "${SOURCE}" ]]; then
  SOURCE="$(pick_first \
    "${ROOT}/export_onnx/bus.jpg" \
    "${ROOT}/export/calib/bus.jpg" \
    "${ROOT}/export/calib/zidane.jpg" \
    "${ROOT}/export_nb/calib/bus.jpg" || true)"
fi

if [[ ! -x "${BIN}" ]]; then
  echo "Missing ${BIN}"
  echo "On the board:  ./setup_onnx.sh && cd Yolo26_ONNX && ./build.sh"
  exit 1
fi
if [[ -z "${MODEL}" || ! -f "${MODEL}" ]]; then
  echo "Missing ONNX model."
  echo "Use an existing 6-head file:"
  echo "  ./run_onnx.sh export/yolo26n_6.onnx export/calib/bus.jpg"
  echo "Or export e2e ONNX on a PC:"
  echo "  python prepare_onnx.py --model yolo26n.pt --outdir export_onnx"
  exit 1
fi
if [[ -z "${SOURCE}" || ! -f "${SOURCE}" ]]; then
  echo "Missing image. Pass a path:  ./run_onnx.sh ${MODEL} /path/to.jpg"
  exit 1
fi

echo "model=${MODEL}"
echo "source=${SOURCE}"
export LD_LIBRARY_PATH="${ORT}/lib:${LD_LIBRARY_PATH:-}"
exec "${BIN}" "${MODEL}" "${SOURCE}" --no-show --save "${ROOT}/result.jpg" "$@"
