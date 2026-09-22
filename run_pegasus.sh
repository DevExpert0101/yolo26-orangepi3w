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

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PREPARE="${SCRIPT_DIR}/prepare_onnx_acuity.py"
if [[ -f "${PREPARE}" ]]; then
  "${PY}" "${PREPARE}" "${NAME}.onnx" || echo "prepare_onnx_acuity skipped"
fi

# Official zoo pegasus_import.sh always deletes json/data then re-imports.
# Stale graphs (wrong input size, no preproc node) were the source of garbage NBGs.
REUSE="${PEGASUS_REUSE:-0}"
if [[ "${REUSE}" == "1" ]] && grep -q 'AcuityVersion' "${NAME}.json" 2>/dev/null && [[ -f "${NAME}.data" ]]; then
  echo "PEGASUS_REUSE=1: keeping imported ${NAME}.json"
else
  rm -f "${NAME}.json" "${NAME}.data"
  rm -f "${NAME}_inputmeta.yml" "${NAME}_postprocess_file.yml"
  rm -f "${NAME}_pcq.quantize" "${NAME}_uint8.quantize" "${NAME}_int16.quantize"
  # Zoo import_onnx_network: model/json/data only. Do not pass --input-size-list
  # 1,3,H,W — ACUITY prepends batch and then conv_shape IndexError.
  if pegasus_run import onnx \
      --model "${NAME}.onnx" \
      --output-model "${NAME}.json" \
      --output-data "${NAME}.data"; then
    :
  else
    echo "ONNX import without sizes failed; retry 3,${IMGSZ},${IMGSZ}"
    pegasus_run import onnx \
      --model "${NAME}.onnx" \
      --output-model "${NAME}.json" \
      --output-data "${NAME}.data" \
      --inputs images \
      --input-size-list "3,${IMGSZ},${IMGSZ}"
  fi
fi

if [[ ! -f "${NAME}.json" || ! -f "${NAME}.data" ]]; then
  echo "pegasus import onnx failed (${NAME}.json / ${NAME}.data missing)"
  exit 1
fi

# Same two generate steps as pegasus_import.sh, then config_yml.py equivalent.
pegasus_run generate inputmeta \
  --model "${NAME}.json" \
  --separated-database \
  --input-meta-output "${NAME}_inputmeta.yml"
pegasus_run generate postprocess-file \
  --model "${NAME}.json" \
  --postprocess-file-output "${NAME}_postprocess_file.yml"

PATCH="${SCRIPT_DIR}/patch_inputmeta.py"
if [[ -f "${PATCH}" ]]; then
  "${PY}" "${PATCH}" "${NAME}"
else
  echo "missing ${PATCH}"
  exit 1
fi

if [[ ! -f "${NAME}_inputmeta.yml" ]]; then
  echo "missing ${NAME}_inputmeta.yml after generate+patch"
  exit 1
fi
if ! grep -qi 'add_preproc_node: true' "${NAME}_inputmeta.yml"; then
  echo "ERROR: ${NAME}_inputmeta.yml must have add_preproc_node: true (IMAGE_RGB)."
  echo "Without it the NBG expects a float tensor, not HWC uint8, and boxes are garbage."
  exit 1
fi
if [[ ! -f "${NAME}_postprocess_file.yml" ]]; then
  echo "missing ${NAME}_postprocess_file.yml after generate"
  exit 1
fi
if ! grep -qi 'add_postproc_node: true' "${NAME}_postprocess_file.yml"; then
  echo "ERROR: ${NAME}_postprocess_file.yml must have add_postproc_node: true"
  exit 1
fi

META="--with-input-meta ${NAME}_inputmeta.yml"
POST="--postprocess-file ${NAME}_postprocess_file.yml"

case "${QUANT}" in
  fp16|float16|float) QUANT="fp16" ;;
  pcq|int8) QUANT="pcq" ;;
  int16) QUANT="int16" ;;
esac

QUANTIZER="asymmetric_affine"
QTYPE="${QUANT}"
DTYPE="quantized"
OUT_STEM="${NAME}_${QUANT}"
if [[ "${QUANT}" == "pcq" ]]; then
  QUANTIZER="perchannel_symmetric_affine"
  QTYPE="int8"
elif [[ "${QUANT}" == "int16" ]]; then
  # Frigate A733 notes: int16 stays near float; uint8/pcq drops scores too far.
  QUANTIZER="dynamic_fixed_point"
  QTYPE="int16"
elif [[ "${QUANT}" == "fp16" ]]; then
  # VIP9000 has no FP32 MACs. ACUITY --dtype float packs an FP16 NBG (zoo: wksp/NAME_fp16).
  DTYPE="float"
  OUT_STEM="${NAME}_fp16"
  QTYPE=""
fi
ITERS="${BITS:-12}"

if [[ "${QUANT}" == "fp16" ]]; then
  echo "Skipping pegasus quantize (FP16 / --dtype float). No INT8 calibration."
else
  # Do not reuse an old .quantize: it was built against the previous inputmeta.
  rm -f "${NAME}_${QUANT}.quantize"
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
  if [[ ! -f "${NAME}_${QUANT}.quantize" ]]; then
    echo "pegasus quantize failed"
    exit 1
  fi
fi


mkdir -p wksp
if ! command -v gcc >/dev/null 2>&1; then
  echo "NBG pack needs a host gcc (the image gcc cannot link against Ubuntu 22.04)."
  echo "Install a compiler, then re-run convert:"
  echo "  sudo apt-get update && sudo apt-get install -y build-essential"
  echo "  bash ./convert_npu.sh --image-tar ../linux_aw_npu/docker/ubuntu-npu_v2.0.10.2.tar --onnx export/${NAME}.onnx --quant ${QUANT}"
  exit 1
fi
export EXTRALFLAGS="${EXTRALFLAGS:-} -lNNArchPerf -lArchModelSw -ldl -lpthread -lrt"
rm -rf "wksp/${OUT_STEM}" "wksp/${OUT_STEM}_nbg_unify"
# shellcheck disable=SC2086
# Zoo pegasus_export_ovx_nbg.sh always passes --with-input-meta and --postprocess-file.
if [[ "${DTYPE}" == "float" ]]; then
  pegasus_run export ovxlib \
    --model "${NAME}.json" \
    --model-data "${NAME}.data" \
    --dtype float \
    --target-ide-project linux64 \
    --optimize "${OPTIMIZE}" \
    --viv-sdk "${VIV_SDK}" \
    --pack-nbg-unify \
    --output-path "./wksp/${OUT_STEM}/${OUT_STEM}" \
    ${META} ${POST}
else
  pegasus_run export ovxlib \
    --model "${NAME}.json" \
    --model-data "${NAME}.data" \
    --dtype quantized \
    --model-quantize "${NAME}_${QUANT}.quantize" \
    --target-ide-project linux64 \
    --optimize "${OPTIMIZE}" \
    --viv-sdk "${VIV_SDK}" \
    --pack-nbg-unify \
    --output-path "./wksp/${OUT_STEM}/${OUT_STEM}" \
    ${META} ${POST}
fi


# --pack-nbg-unify writes wksp/OUT_STEM_nbg_unify/network_binary.nb.
# Never `find wksp` globally: a leftover yolo26n .nb would be copied as yolo26x.
NB_OUT="${NAME}_${QUANT}_a733.nb"
nb_src=""
for cand in \
  "wksp/${OUT_STEM}_nbg_unify/network_binary.nb" \
  "wksp/${OUT_STEM}/network_binary.nb"
do
  if [[ -f "${cand}" ]]; then
    nb_src="${cand}"
    break
  fi
done
if [[ -z "${nb_src}" ]]; then
  for dir in "wksp/${OUT_STEM}_nbg_unify" "wksp/${OUT_STEM}"; do
    if [[ -d "${dir}" ]]; then
      nb_src="$(find "${dir}" -name '*.nb' 2>/dev/null | head -1 || true)"
      [[ -n "${nb_src}" ]] && break
    fi
  done
fi
if [[ -z "${nb_src}" || ! -f "${nb_src}" ]]; then
  echo "No .nb under wksp/${OUT_STEM}*_nbg_unify/. NBG pack did not finish for ${NAME}."
  echo "Other models in wksp/ are ignored on purpose."
  exit 1
fi
cp -f "${nb_src}" "${NB_OUT}"
echo "NBG: ${nb_src} -> ${NB_OUT}  ($(wc -c < "${NB_OUT}") bytes)"
if [[ -f "wksp/${OUT_STEM}_nbg_unify/nbg_meta.json" ]]; then
  cp -f "wksp/${OUT_STEM}_nbg_unify/nbg_meta.json" "${NAME}_${QUANT}_nbg_meta.json"
  echo "meta: ${NAME}_${QUANT}_nbg_meta.json"
fi
echo "inputmeta add_preproc_node:"
grep -E 'add_preproc_node|preproc_type|scale:' "${NAME}_inputmeta.yml" | head -20
ls -l "${NB_OUT}"
