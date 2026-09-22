#!/usr/bin/env bash
# Compile the 6-head YOLO26 ONNX to an A733 NBG for the 3 TOPS VIP9000.
#
# Default: native ACUITY (no Docker). Needs x86_64 Linux or WSL + pegasus.
#
#   python export_npu.py --model yolo26n.pt --outdir export
#   export ACUITY_PATH=$HOME/acuity-toolkit-whl-6.30.22/bin
#   export VIV_SDK=$HOME/Vivante_IDE/VivanteIDE5.11.0/cmdtools
#   ./convert_npu.sh --onnx export/yolo26n_6.onnx              # FP16 (default)
#   ./convert_npu.sh --onnx export/yolo26n_6.onnx --quant pcq  # INT8
#
# Docker is optional:  ./convert_npu.sh --docker --onnx export/yolo26n_6.onnx
set -euo pipefail

if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
  echo "Do not run this with sudo. ACUITY lives in your user home; sudo looks in /root."
  echo "Re-run as:  ./convert_npu.sh --onnx export/yolo26n_6.onnx"
  exit 1
fi

IMAGE="${NPU_DOCKER_IMAGE:-ubuntu-npu:v2.0.10.2}"
ONNX=""
NAME=""
QUANT="${NPU_QUANT:-fp16}"
BITS="${NPU_QUANT_BITS:-12}"
OPTIMIZE="VIP9000NANODI_PLUS_PID0X1000003B"
ZOO=""
MODE="auto"
IMGSZ=""
SDK=""
IMAGE_TAR=""
CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/orangepi-acuity"

usage() {
  cat <<EOF
Usage: $0 --onnx PATH [options]

  --onnx PATH     6-head ONNX from export_npu.py
  --name NAME     stem (default: ONNX filename without .onnx)
  --imgsz N       input size (default: NAME_npu.json, else 640)
  --quant TYPE    fp16 (default, no INT8) | pcq | int16 | uint8
  --native        use host pegasus (no Docker)
  --docker        use ${IMAGE}
  --image NAME    Docker image if --docker
  --image-tar T   unpack pegasus from ubuntu-npu_*.tar (no Docker daemon)
  --sdk DIR       linux_aw_npu tree (finds pegasus or the docker tar)
  --zoo DIR       optional awnpu_model_zoo (yolo26 convert wrappers)

Env: ACUITY_PATH  toolkit bin/  (native)
     VIV_SDK      VivanteIDE*/cmdtools  (native, required for export)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --onnx) ONNX="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --quant) QUANT="$2"; shift 2 ;;
    --zoo) ZOO="$2"; shift 2 ;;
    --native) MODE="native"; shift ;;
    --docker) MODE="docker"; shift ;;
    --image-tar) IMAGE_TAR="$2"; shift 2 ;;
    --sdk) SDK="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg $1"; usage; exit 1 ;;
  esac
done

ROOT="$(cd "$(dirname "$0")" && pwd)"
ONNX="$(realpath "${ONNX:-$ROOT/export/yolo26n_6.onnx}")"
[[ -f "${ONNX}" ]] || { echo "Missing ONNX: ${ONNX}. Run python export_npu.py first."; exit 1; }
NAME="${NAME:-$(basename "${ONNX}" .onnx)}"
WORKDIR="$(dirname "${ONNX}")"
RUNNER="${ROOT}/run_pegasus.sh"
# /mnt/d is NTFS: chmod +x often fails in WSL. Always invoke with bash.

if [[ -z "${IMGSZ}" ]] && command -v python3 >/dev/null 2>&1; then
  for jf in "${WORKDIR}/${NAME}_npu.json" "${WORKDIR}/${NAME}.json"; do
    if [[ -f "${jf}" ]]; then
      IMGSZ="$(python3 -c "import json; print(json.load(open('${jf}')).get('imgsz', ''))" 2>/dev/null || true)"
      [[ -n "${IMGSZ}" ]] && break
    fi
  done
fi
IMGSZ="${IMGSZ:-640}"
case "${QUANT}" in
  fp16|float16|float) QUANT="fp16" ;;
  pcq|int8) QUANT="pcq" ;;
  int16) QUANT="int16" ;;
esac
ZOO_QUANT="${QUANT}"
[[ "${QUANT}" == "fp16" ]] && ZOO_QUANT="float"

pegasus_exists() {
  local p="$1"
  [[ -n "${p}" && -f "${p}" && ! -d "${p}" ]]
}

acuity_bin_ok() {
  local dir="${1:-}"
  [[ -n "${dir}" && -d "${dir}" ]] || return 1
  [[ -f "${dir}/pegasus" || -f "${dir}/pegasus.py" ]]
}

have_pegasus() {
  command -v pegasus >/dev/null 2>&1 && return 0
  acuity_bin_ok "${ACUITY_PATH:-}" && return 0
  acuity_bin_ok "${HOME}/acuity-toolkit-whl-6.30.22/bin" && return 0
  local d
  for d in "${HOME}"/acuity-toolkit-*/bin "${HOME}"/acuity-toolkit-binary-*/bin; do
    acuity_bin_ok "${d}" && return 0
  done
  return 1
}

activate_native_path() {
  if [[ -f "${CACHE}/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${CACHE}/env.sh"
  fi
  if acuity_bin_ok "${ACUITY_PATH:-}" && [[ -d "${VIV_SDK:-}" ]]; then
    export PATH="${ACUITY_PATH}:${PATH}"
    return 0
  fi
  if command -v pegasus >/dev/null 2>&1 && [[ -d "${VIV_SDK:-}" ]]; then
    return 0
  fi
  local d
  for d in \
    "${HOME}/acuity-toolkit-whl-6.30.22/bin" \
    "${HOME}"/acuity-toolkit-whl-*/bin \
    "${HOME}"/acuity-toolkit-binary-*/bin
  do
    if acuity_bin_ok "${d}"; then
      export ACUITY_PATH="${d}"
      export PATH="${d}:${PATH}"
      return 0
    fi
  done
  return 1
}

native_help() {
  cat <<EOF
ACUITY pegasus is not on this WSL/Linux user account. The 6-head ONNX is ready;
conversion cannot start until the Allwinner toolkit is present.

Do not use sudo.

1. Download linux_aw_npu from the Orange Pi Zero 3W support / Official Tools
   pack (Baidu / Google Drive on the board page), or KickPi's copy:
   https://doc.kickpi.com/products/linux_customization/linux_npu/

2. You want one of:
     linux_aw_npu/docker/ubuntu-npu_v2.0.10.2.tar.zip
     acuity-toolkit-whl-6.30.22/bin/pegasus + VivanteIDE*/cmdtools

3. Unpack pegasus WITHOUT Docker:
     unzip ubuntu-npu_v2.0.10.2.tar.zip
     $0 --image-tar ubuntu-npu_v2.0.10.2.tar --onnx ${ONNX}

   or if the toolkit is already extracted:
     export ACUITY_PATH=\$HOME/acuity-toolkit-whl-6.30.22/bin
     export VIV_SDK=\$HOME/Vivante_IDE/VivanteIDE5.11.0/cmdtools
     $0 --native --onnx ${ONNX}

A733 target: ${OPTIMIZE}
EOF
}

use_acuity_dirs() {
  local peg="$1"
  local viv="$2"
  pegasus_exists "${peg}" || return 1
  [[ -n "${viv}" && -d "${viv}" ]] || return 1
  export ACUITY_PATH="$(cd "$(dirname "${peg}")" && pwd)"
  export VIV_SDK="$(cd "${viv}" && pwd)"
  export PATH="${ACUITY_PATH}:${PATH}"
  echo "ACUITY_PATH=${ACUITY_PATH}"
  echo "VIV_SDK=${VIV_SDK}"
}

discover_acuity() {
  local root="$1"
  [[ -d "${root}" ]] || return 1
  local peg viv
  peg="$(find "${root}" -type f -path '*/acuity-toolkit*/bin/pegasus.py' 2>/dev/null | head -1 || true)"
  [[ -n "${peg}" ]] || peg="$(find "${root}" -type f \( -name pegasus -o -name pegasus.py \) 2>/dev/null | head -1 || true)"
  viv="$(find "${root}" -type d -name cmdtools 2>/dev/null | head -1 || true)"
  use_acuity_dirs "${peg}" "${viv}"
}

unpack_layer_blob() {
  local blob="$1"
  local rootfs="$2"
  local magic
  magic="$(od -An -tx1 -N2 "${blob}" 2>/dev/null | tr -d ' \n')"
  if [[ "${magic}" == "1f8b" ]]; then
    tar -xzf "${blob}" -C "${rootfs}" || true
  else
    tar -xf "${blob}" -C "${rootfs}" 2>/dev/null || true
  fi
}

extract_image_tar() {
  local tarfile="$1"
  tarfile="$(realpath "${tarfile}")"
  [[ -f "${tarfile}" ]] || { echo "Missing --image-tar ${tarfile}"; return 1; }
  CACHE="$(dirname "${tarfile}")/.acuity-extract"
  mkdir -p "${CACHE}"
  if [[ -f "${CACHE}/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${CACHE}/env.sh"
    if acuity_bin_ok "${ACUITY_PATH:-}" && [[ -d "${VIV_SDK:-}" ]]; then
      export PATH="${ACUITY_PATH}:${PATH}"
      echo "Using cached ACUITY in ${CACHE}"
      return 0
    fi
  fi
  echo "Unpacking ${tarfile} (OCI image, no Docker). This can take several minutes and several GB."
  local work rootfs
  work="$(mktemp -d "${CACHE}/unpack.XXXXXX")"
  rootfs="${CACHE}/rootfs"
  mkdir -p "${rootfs}"
  echo "Extracting image blobs..."
  tar -xf "${tarfile}" -C "${work}"
  local blob layers=()
  if [[ -f "${work}/manifest.json" ]] && command -v python3 >/dev/null 2>&1; then
    mapfile -t layers < <(python3 -c "import json,sys; m=json.load(open(sys.argv[1]));
print('\\n'.join(m[0]['Layers'] if isinstance(m,list) else []))
" "${work}/manifest.json")
  fi
  if [[ ${#layers[@]} -gt 0 ]]; then
    local layer
    for layer in "${layers[@]}"; do
      blob="${work}/${layer}"
      echo "  layer ${layer##*/}"
      unpack_layer_blob "${blob}" "${rootfs}"
    done
  else
    echo "No Docker/OCI Layers in manifest; trying every blob."
    while IFS= read -r blob; do
      echo "  blob $(basename "${blob}")"
      unpack_layer_blob "${blob}" "${rootfs}"
    done < <(find "${work}/blobs" -type f 2>/dev/null)
  fi
  if ! find "${rootfs}" -type f \( -name pegasus -o -name pegasus.py \) | grep -q .; then
    echo "No pegasus inside ${tarfile} after unpacking OCI layers."
    echo "Looked under ${rootfs}. First entries:"
    ls -la "${rootfs}" | head || true
    rm -rf "${work}"
    return 1
  fi
  discover_acuity "${rootfs}" || {
    echo "Found pegasus but not Vivante cmdtools. Need both for export."
    find "${rootfs}" -type f \( -name pegasus -o -name pegasus.py \) | head
    find "${rootfs}" -type d -name cmdtools | head
    rm -rf "${work}"
    return 1
  }
  {
    printf 'export ACUITY_PATH=%q\n' "${ACUITY_PATH}"
    printf 'export VIV_SDK=%q\n' "${VIV_SDK}"
    printf 'export ACUITY_ROOTFS=%q\n' "${rootfs}"
    printf 'export PATH="%s:$PATH"\n' "${ACUITY_PATH}"
  } > "${CACHE}/env.sh"
  rm -rf "${work}"
}

if [[ -n "${IMAGE_TAR}" ]]; then
  extract_image_tar "${IMAGE_TAR}" || exit 1
  MODE="native"
fi
if [[ -n "${SDK}" ]]; then
  SDK="$(realpath "${SDK}")"
  if discover_acuity "${SDK}"; then
    MODE="native"
  else
    found_tar="$(find "${SDK}" -name 'ubuntu-npu*.tar' 2>/dev/null | head -1 || true)"
    if [[ -n "${found_tar}" ]]; then
      extract_image_tar "${found_tar}"
      MODE="native"
    fi
  fi
fi

if [[ "${MODE}" == "auto" ]]; then
  if have_pegasus; then
    MODE="native"
  else
    echo "No host pegasus found. Native ACUITY is the default (no Docker)."
    native_help
    exit 1
  fi
fi

run_zoo_native() {
  local convert="${ZOO}/examples/yolo26/convert_model"
  [[ -n "${ZOO}" && -x "${convert}/pegasus_import.sh" ]] || return 1
  echo "Using model-zoo wrappers in ${convert}"
  cp -f "${ONNX}" "${convert}/${NAME}.onnx"
  (
    cd "${convert}"
    ./convert_model_env.sh || true
    ./pegasus_import.sh "${NAME}"
    if [[ "${ZOO_QUANT}" != "float" ]]; then
      ./pegasus_quantize.sh "${NAME}" "${ZOO_QUANT}" "${BITS}"
    fi
    ./pegasus_export_ovx_nbg.sh "${NAME}" "${ZOO_QUANT}" a733
  )
  find "${ZOO}" -name "*${NAME}*a733*.nb" -exec cp -f {} "${WORKDIR}/" \;
  return 0
}

if [[ "${MODE}" == "native" ]]; then
  if ! activate_native_path; then
    native_help
    exit 1
  fi
  if run_zoo_native; then
    :
  else
    (
      cd "${WORKDIR}"
      bash "${RUNNER}" "${NAME}" "${QUANT}" "${BITS}" "${OPTIMIZE}" "${IMGSZ}"
    )
  fi
else
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is not installed. Use --native instead."
    native_help
    exit 1
  fi
  if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "Docker image ${IMAGE} is not loaded."
    echo "  docker load -i ubuntu-npu_v2.0.10.2.tar"
    echo "Or skip Docker: install ACUITY and run $0 --native --onnx ${ONNX}"
    exit 1
  fi
  INNER="/workspace/export"
  ZOO_MOUNT=()
  if [[ -n "${ZOO}" ]]; then
    ZOO_MOUNT=(-v "$(realpath "${ZOO}"):/workspace/zoo")
  fi
  docker run --rm --ipc=host \
    -v "${WORKDIR}:${INNER}" \
    -v "${RUNNER}:/workspace/run_pegasus.sh:ro" \
    -v "${ROOT}/patch_inputmeta.py:/workspace/patch_inputmeta.py:ro" \
    -v "${ROOT}/prepare_onnx_acuity.py:/workspace/prepare_onnx_acuity.py:ro" \
    "${ZOO_MOUNT[@]}" \
    "${IMAGE}" \
    bash -lc "cd ${INNER} && bash /workspace/run_pegasus.sh '${NAME}' '${QUANT}' '${BITS}' '${OPTIMIZE}' '${IMGSZ}'"
fi

NB_FILE="${WORKDIR}/${NAME}_${QUANT}_a733.nb"
if [[ -f "${NB_FILE}" ]]; then
  mkdir -p "${ROOT}/export_nb"
  short="${NAME%_6}"
  dest="${ROOT}/export_nb/${short}_${QUANT}.nb"
  cp -f "${NB_FILE}" "${dest}"
  echo "Also copied ${dest}  ($(wc -c < "${dest}") bytes)"
  [[ -f "${WORKDIR}/${NAME}_inputmeta.yml" ]] && cp -f "${WORKDIR}/${NAME}_inputmeta.yml" "${ROOT}/export_nb/"
  [[ -f "${WORKDIR}/${NAME}_postprocess_file.yml" ]] && cp -f "${WORKDIR}/${NAME}_postprocess_file.yml" "${ROOT}/export_nb/"
  echo "NBG input is IMAGE_RGB (HWC uint8 0-255). Rebuild yolo26_npu after git pull."
  # Same-size as another stem usually means the wrong wksp .nb was copied.
  while IFS= read -r other; do
    [[ "${other}" == "${dest}" ]] && continue
    if [[ -f "${other}" ]] && cmp -s "${dest}" "${other}"; then
      echo "ERROR: ${dest} is a byte-for-byte copy of ${other}."
      echo "That is the old 'find wksp | head' bug. Re-run convert for ${NAME} after git pull."
      exit 1
    fi
  done < <(find "${ROOT}/export_nb" -maxdepth 1 -name "*_${QUANT}.nb" 2>/dev/null)
fi
echo
echo "If conversion succeeded, copy this file to the Orange Pi:"
echo "  ${NB_FILE}"
echo "ACUITY's pack name is network_binary.nb; convert_npu.sh renames it."
echo "  ./Yolo26_NPU/build/yolo26_npu ${NB_FILE} export/calib/bus.jpg --no-show --save result.jpg"
ls -l "${NB_FILE}" "${WORKDIR}"/*.nb 2>/dev/null || true
