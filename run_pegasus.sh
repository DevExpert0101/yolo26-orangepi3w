#!/usr/bin/env bash
# Run ACUITY pegasus on a 6-head YOLO26 ONNX. CWD must contain NAME.onnx.
# Args: NAME QUANT BITS OPTIMIZE IMGSZ
set -euo pipefail

NAME="$1"
QUANT="$2"
BITS="$3"
OPTIMIZE="$4"
IMGSZ="${5:-640}"

export PATH="${ACUITY_PATH:-$HOME/acuity-toolkit-whl-6.30.22/bin}:${PATH}"
export VIV_SDK="${VIV_SDK:-$HOME/Vivante_IDE/VivanteIDE5.11.0/cmdtools}"

# ACUITY wheels are Python 3.8 (onnx_cpp2py_export.cpython-38). Host 3.10 cannot load them.
# Put host glibc first: the image libc is 2.31 and poisons Ubuntu 22.04 if it wins LD_LIBRARY_PATH.
if [[ -n "${ACUITY_ROOTFS:-}" && -f "${ACUITY_ROOTFS}/usr/bin/python3.8" ]]; then
  export PYTHONHOME="${ACUITY_ROOTFS}/usr"
  export PYTHONPATH="${ACUITY_ROOTFS}/usr/local/lib/python3.8/dist-packages${PYTHONPATH:+:$PYTHONPATH}"
  export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${ACUITY_ROOTFS}/usr/lib/x86_64-linux-gnu:${ACUITY_ROOTFS}/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  PY="${ACUITY_ROOTFS}/usr/bin/python3.8"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:--1}"
  export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
  # NBG `make` is often missing on WSL. Image make is enough; gcc must be the host compiler.
  WRAP="${XDG_CACHE_HOME:-$HOME/.cache}/orangepi-acuity/bin"
  mkdir -p "${WRAP}"
  rm -f "${WRAP}/gcc" "${WRAP}/cc"
  cp -f "${ACUITY_ROOTFS}/usr/bin/make" "${WRAP}/make"
  chmod +x "${WRAP}/make" 2>/dev/null || true
  export PATH="${WRAP}:${PATH}"
  echo "Using extracted Python 3.8: ${PY}"
  case "${ACUITY_ROOTFS}" in
    /mnt/*) echo "Note: ACUITY on /mnt is 9p/NTFS; first TensorFlow import can take a few minutes." ;;
  esac
else
  PY=python3
  command -v python3.8 >/dev/null 2>&1 && PY=python3.8
fi

pegasus_run() {
  if [[ -f "${ACUITY_PATH}/pegasus.py" ]]; then
    "${PY}" "${ACUITY_PATH}/pegasus.py" "$@"
  elif [[ -f "${ACUITY_PATH}/pegasus" ]]; then
    "${PY}" "${ACUITY_PATH}/pegasus" "$@"
  elif command -v pegasus >/dev/null 2>&1; then
    pegasus "$@"
  else
    echo "pegasus.py not found in ACUITY_PATH=${ACUITY_PATH:-unset}"
    return 1
  fi
}

if [[ ! -d "${VIV_SDK}" ]]; then
  echo "VIV_SDK is not a directory: ${VIV_SDK}"
  echo "Point it at Vivante IDE cmdtools (needed for pegasus export ovxlib)."
  exit 1
fi

echo "Using ACUITY_PATH=${ACUITY_PATH}"
echo "VIV_SDK=${VIV_SDK}"
echo "Import ${NAME}.onnx  imgsz=${IMGSZ}  quant=${QUANT}  target=${OPTIMIZE}"

PREPARE="$(cd "$(dirname "$0")" && pwd)/prepare_onnx_acuity.py"
if [[ -f "${PREPARE}" ]]; then
  "${PY}" "${PREPARE}" "${NAME}.onnx" || echo "prepare_onnx_acuity skipped"
fi

# Do not pass --input-size-list 1,3,H,W. ACUITY prepends batch itself, then
# conv_shape IndexError. Zoo ONNX import uses CHW without the batch dim.
if grep -q 'AcuityVersion' "${NAME}.json" 2>/dev/null && [[ -f "${NAME}.data" ]]; then
  echo "Reusing imported ${NAME}.json"
else
  pegasus_run import onnx \
    --model "${NAME}.onnx" \
    --output-model "${NAME}.json" \
    --output-data "${NAME}.data" \
    --inputs images \
    --input-size-list "3,${IMGSZ},${IMGSZ}"
fi

PATCH="$(cd "$(dirname "$0")" && pwd)/patch_inputmeta.py"
if [[ -f "${PATCH}" ]]; then
  "${PY}" "${PATCH}" "${NAME}"
fi

META=""
[[ -f "${NAME}_inputmeta.yml" ]] && META="--with-input-meta ${NAME}_inputmeta.yml"

QUANTIZER="asymmetric_affine"
QTYPE="${QUANT}"
if [[ "${QUANT}" == "pcq" ]]; then
  QUANTIZER="perchannel_symmetric_affine"
  QTYPE="int8"
fi
ITERS="${BITS:-12}"

if [[ -f "${NAME}_${QUANT}.quantize" ]]; then
  echo "Reusing ${NAME}_${QUANT}.quantize"
else
  # shellcheck disable=SC2086
  pegasus_run quantize \
    --model "${NAME}.json" \
    --model-data "${NAME}.data" \
    --device CPU \
    --quantizer "${QUANTIZER}" \
    --qtype "${QTYPE}" \
    --iterations "${ITERS}" \
    --rebuild \
    --model-quantize "${NAME}_${QUANT}.quantize" \
    ${META}
fi

mkdir -p wksp
if ! command -v gcc >/dev/null 2>&1; then
  echo "NBG pack needs a host gcc (the image gcc cannot link against Ubuntu 22.04)."
  echo "Quantize already saved ${NAME}_${QUANT}.quantize. Install a compiler, then re-run convert:"
  echo "  sudo apt-get update && sudo apt-get install -y build-essential"
  echo "  bash ./convert_npu.sh --image-tar ../linux_aw_npu/docker/ubuntu-npu_v2.0.10.2.tar --onnx export/${NAME}.onnx"
  exit 1
fi
export EXTRALFLAGS="${EXTRALFLAGS:-} -lNNArchPerf -lArchModelSw -ldl -lpthread -lrt"
POST=""
[[ -f "${NAME}_postprocess_file.yml" ]] && POST="--postprocess-file ${NAME}_postprocess_file.yml"
# shellcheck disable=SC2086
pegasus_run export ovxlib \
  --model "${NAME}.json" \
  --model-data "${NAME}.data" \
  --dtype quantized \
  --model-quantize "${NAME}_${QUANT}.quantize" \
  --target-ide-project linux64 \
  --optimize "${OPTIMIZE}" \
  --viv-sdk "${VIV_SDK}" \
  --pack-nbg-unify \
  --output-path "./wksp/${NAME}_${QUANT}/${NAME}_${QUANT}" \
  ${META} ${POST}

# --pack-nbg-unify always writes network_binary.nb. Zoo renames it to NAME_pcq_a733.nb.
NB_OUT="${NAME}_${QUANT}_a733.nb"
nb_src="$(find wksp -name network_binary.nb 2>/dev/null | head -1 || true)"
[[ -z "${nb_src}" ]] && nb_src="$(find wksp -name '*.nb' 2>/dev/null | head -1 || true)"
if [[ -n "${nb_src}" ]]; then
  cp -f "${nb_src}" "${NB_OUT}"
  echo "NBG: ${nb_src} -> ${NB_OUT}"
else
  echo "No network_binary.nb under wksp/. NBG pack did not finish."
  exit 1
fi
ls -l "${NB_OUT}"
