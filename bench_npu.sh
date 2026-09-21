#!/usr/bin/env bash
# Time YOLO26 NPU on the Orange Pi. Reports VIPLite infer vs preprocess / postprocess.
#   ./bench_npu.sh
#   ./bench_npu.sh export_nb/yolo26n.nb export/calib/bus.jpg
#   ./bench_npu.sh --loop 50
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
LOOP=30
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --loop) LOOP="$2"; shift 2 ;;
    --warmup) EXTRA+=(--warmup "$2"); shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
chmod +x "${ROOT}/run_npu.sh"
exec "${ROOT}/run_npu.sh" "${EXTRA[@]}" --loop "${LOOP}" --warmup 5 --no-show
