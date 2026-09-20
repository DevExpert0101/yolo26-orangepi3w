#!/usr/bin/env bash
# Install packages and NPU libraries on Orange Pi Zero 3W (Debian/Ubuntu aarch64).
# Run on the board, not on Windows.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="${ROOT}/.venv"
SWAP=0
PYTHON=0
FULL=0
NPU=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --swap) SWAP=1; shift ;;
    --python) PYTHON=1; shift ;;
    --full) PYTHON=1; FULL=1; shift ;;
    --cpu-only) NPU=0; shift ;;
    *) echo "Usage: $0 [--swap] [--python] [--full] [--cpu-only]"; exit 1 ;;
  esac
done

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This script is meant for the Orange Pi (aarch64). Current: $(uname -m)"
fi

echo "==> apt packages for the C++ NPU app"
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  g++ \
  pkg-config \
  git \
  wget \
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
  libatomic1 \
  v4l-utils

if [[ "${PYTHON}" -eq 1 ]]; then
  sudo apt-get install -y python3 python3-venv python3-pip python3-opencv
fi

if [[ "${SWAP}" -eq 1 && ! -f /swapfile ]]; then
  echo "==> 2G swap"
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

if [[ "${NPU}" -eq 1 ]]; then
  echo "==> NPU device"
  if [[ -e /dev/vipcore ]]; then
    sudo chmod 666 /dev/vipcore || true
    echo 'KERNEL=="vipcore", MODE="0666"' | sudo tee /etc/udev/rules.d/99-vipcore.rules >/dev/null
    sudo udevadm control --reload-rules || true
    echo "NPU device present: /dev/vipcore"
  else
    echo "NPU device /dev/vipcore not found. Flash the vendor Orange Pi image."
  fi

  echo "==> VIPLite libraries"
  mapfile -t HITS < <(find /usr /opt /lib "${HOME}" -name 'libNBGlinker.so' -o -name 'libVIPhal.so' 2>/dev/null | head -n 20)
  if [[ ${#HITS[@]} -eq 0 ]]; then
    echo "libNBGlinker.so / libVIPhal.so not found."
    echo "Copy A733 VIPLite .so files into ${ROOT}/Yolo26_NPU/lib"
  else
    printf '%s\n' "${HITS[@]}"
    mkdir -p "${ROOT}/Yolo26_NPU/lib"
    for f in "${HITS[@]}"; do
      cp -n "$f" "${ROOT}/Yolo26_NPU/lib/" 2>/dev/null || true
    done
    echo "Copied available VIPLite libs into ${ROOT}/Yolo26_NPU/lib"
  fi
  sudo ldconfig || true
fi

if [[ "${PYTHON}" -eq 1 ]]; then
  echo "==> Python venv"
  python3 -m venv "${VENV}"
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  python -m pip install -U pip wheel
  if [[ "${FULL}" -eq 1 ]]; then
    python -m pip install -U ultralytics ncnn onnxruntime
  else
    python -m pip install -r "${ROOT}/requirements-board.txt"
  fi
  (cd "${ROOT}" && python npu_runtime.py --probe || true)
fi

echo
echo "Board packages installed."
echo "OpenCV: $(pkg-config --modversion opencv4 2>/dev/null || pkg-config --modversion opencv 2>/dev/null || echo missing)"
echo "Next:  cd ${ROOT}/Yolo26_NPU && ./build.sh"
if [[ "${PYTHON}" -eq 1 ]]; then
  echo "Python: source ${VENV}/bin/activate"
fi
