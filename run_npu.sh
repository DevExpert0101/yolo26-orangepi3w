#!/usr/bin/env bash
# Run the YOLO26 NPU C++ app. Extra flags are forwarded (e.g. --loop 50 --conf 0.3).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
BIN="${ROOT}/Yolo26_NPU/build/yolo26_npu"
LIB="${ROOT}/Yolo26_NPU/lib"

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
    "${ROOT}/export_nb/yolo26n_fp16.nb" \
    "${ROOT}/export/yolo26n_6_fp16_a733.nb" \
    "${ROOT}/export_nb/wksp/yolo26n_6_fp16_nbg_unify/network_binary.nb" \
    "${ROOT}/export/wksp/yolo26n_6_fp16_nbg_unify/network_binary.nb" \
    "${ROOT}/export_nb/yolo26n.nb" \
    "${ROOT}/export_nb/wksp/yolo26n_6_pcq_nbg_unify/network_binary.nb" \
    "${ROOT}/export/yolo26n_6_pcq_a733.nb" \
    "${ROOT}/export_nb/yolo26s.nb" || true)"
fi
if [[ -z "${SOURCE}" ]]; then
  SOURCE="$(pick_first \
    "${ROOT}/export/calib/bus.jpg" \
    "${ROOT}/export_nb/calib/bus.jpg" \
    "${ROOT}/export/calib/zidane.jpg" || true)"
fi

if [[ ! -x "${BIN}" ]]; then
  echo "Missing ${BIN}"
  echo "On the board:  ./setup_board.sh && cd Yolo26_NPU && ./build.sh"
  exit 1
fi
if [[ -z "${MODEL}" || ! -f "${MODEL}" ]]; then
  echo "Missing .nb model. Copy from the PC, then:"
  echo "  ./run_npu.sh export_nb/yolo26n.nb export/calib/bus.jpg"
  exit 1
fi
if [[ -z "${SOURCE}" || ! -f "${SOURCE}" ]]; then
  echo "Missing image. Pass a path:  ./run_npu.sh ${MODEL} /path/to.jpg"
  exit 1
fi

echo "model=${MODEL}"
echo "source=${SOURCE}"
export LD_LIBRARY_PATH="${LIB}:/usr/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"
exec "${BIN}" "${MODEL}" "${SOURCE}" --no-show --save "${ROOT}/result_npu.jpg" "$@"
