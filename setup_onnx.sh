#!/usr/bin/env bash
# Prepare the ONNX Runtime environment on Orange Pi Zero 3W (Debian/Ubuntu aarch64).
# This path does not need VIPLite or /dev/vipcore.
#
#   ./setup_onnx.sh           C++ toolchain + OpenCV + ONNX Runtime libs
#   ./setup_onnx.sh --python  also a venv for infer.py
#   ./setup_onnx.sh --swap    also 2G swap (1 GB boards)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="${ROOT}/.venv"
SWAP=0
PYTHON=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --swap) SWAP=1; shift ;;
    --python) PYTHON=1; shift ;;
    *) echo "Usage: $0 [--swap] [--python]"; exit 1 ;;
  esac
done

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "Warning: this script is meant for the Orange Pi (aarch64). Current: $(uname -m)"
fi

echo "==> apt packages for ONNX C++ (g++, CMake, OpenCV)"
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  g++ \
  pkg-config \
  git \
  wget \
  curl \
  ca-certificates \
  libopencv-dev \
  libopencv-core-dev \
  libopencv-imgproc-dev \
  libopencv-imgcodecs-dev \
  libopencv-videoio-dev \
  libopencv-highgui-dev \
  libgl1 \
  libglib2.0-0 \
  libgtk-3-0 \
  libjpeg-dev \
  libpng-dev \
  libtiff-dev \
  libopenblas-dev \
  libgomp1 \
  libatomic1 \
  v4l-utils

if [[ "${PYTHON}" -eq 1 ]]; then
  sudo apt-get install -y python3 python3-venv python3-pip python3-opencv python3-numpy
fi

if [[ "${SWAP}" -eq 1 && ! -f /swapfile ]]; then
  echo "==> 2G swap (needed on 1 GB boards before yolo26s / cmake -j)"
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

echo "==> official ONNX Runtime C/C++ CPU package"
chmod +x "${ROOT}/setup_onnx.sh" "${ROOT}/run_onnx.sh" \
  "${ROOT}/Yolo26_ONNX/fetch_onnxruntime.sh" "${ROOT}/Yolo26_ONNX/build.sh"
"${ROOT}/Yolo26_ONNX/fetch_onnxruntime.sh"

ORT="${ROOT}/Yolo26_ONNX/third_party/onnxruntime"
echo
echo "==> environment probe"
echo "arch       $(uname -m)"
echo "g++        $(g++ --version | head -n1)"
echo "cmake      $(cmake --version | head -n1)"
echo "opencv     $(pkg-config --modversion opencv4 2>/dev/null || pkg-config --modversion opencv 2>/dev/null || echo missing)"
echo "ort header $(test -f "${ORT}/include/onnxruntime_cxx_api.h" && echo OK || echo MISSING)"
echo "ort lib    $(test -e "${ORT}/lib/libonnxruntime.so" && echo OK || echo MISSING)"

if [[ "${PYTHON}" -eq 1 ]]; then
  echo "==> Python venv with onnxruntime"
  python3 -m venv --system-site-packages "${VENV}"
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  python -m pip install -U pip wheel
  python -m pip install -U -r "${ROOT}/requirements-onnx.txt"
  python - <<'PY'
import sys
import numpy
import cv2
import onnxruntime as ort
print("python     ", sys.version.split()[0])
print("numpy      ", numpy.__version__)
print("opencv     ", cv2.__version__)
print("onnxruntime", ort.__version__)
print("providers  ", ort.get_available_providers())
print("OK: Python ONNX environment is ready")
PY
fi

echo
echo "Board ONNX environment installed."
echo "C++:   cd ${ROOT}/Yolo26_ONNX && ./build.sh"
echo "Run:   ${ROOT}/run_onnx.sh"
if [[ "${PYTHON}" -eq 1 ]]; then
  echo "Python: source ${VENV}/bin/activate"
  echo "        python infer.py --backend onnx --model export_onnx/yolo26n.onnx --source export_onnx/bus.jpg"
fi
