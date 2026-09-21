#!/usr/bin/env bash
# Time YOLO26 ONNX on the Orange Pi. Reports ORT infer vs preprocess / postprocess.
#   ./bench_onnx.sh
#   ./bench_onnx.sh export/yolo26n_6.onnx export/calib/bus.jpg
#   ./bench_onnx.sh --loop 50 --threads 2
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
LOOP=30
THREADS=4
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --loop) LOOP="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --warmup) EXTRA+=(--warmup "$2"); shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
chmod +x "${ROOT}/run_onnx.sh"
exec "${ROOT}/run_onnx.sh" "${EXTRA[@]}" --loop "${LOOP}" --warmup 5 --threads "${THREADS}" --no-show
