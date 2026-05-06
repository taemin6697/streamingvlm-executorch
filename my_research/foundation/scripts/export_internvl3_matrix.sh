#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  export_internvl3_matrix.sh [all|qnn|xnnpack|vulkan]

Exports InternVL3 1B/2B/8B artifacts for context lengths:
  512 1024 2048 4096 8192 16384

Environment overrides:
  ROOT_DIR                 Project root. Default: auto-detected /workspace/streamingvlm
  EXPORT_LENGTHS           Space-separated lengths. Default: 512 1024 2048 4096 8192 16384
  EXPORT_MODELS            Space-separated models. Default: internvl3_1b internvl3_2b internvl3_8b
  HF_MODEL_ROOT            Local HF/checkpoint root. Default: $ROOT_DIR/my_research/foundation/results/model/hf
  QNN_ARTIFACT_BASE        QNN output root. Default: $ROOT_DIR/my_research/foundation/results/model/qnn
  XNNPACK_ARTIFACT_BASE    XNNPACK output root. Default: $ROOT_DIR/my_research/foundation/results/model/xnnpack
  VULKAN_ARTIFACT_BASE     Vulkan output root. Default: $ROOT_DIR/my_research/foundation/results/model/vulkan
  SKIP_EXISTING            If 1, skip artifact dirs that already contain manifest.json. Default: 0

QNN overrides:
  QNN_BUILD_PATH           Default: $ROOT_DIR/executorch/build-android
  QNN_DEVICE               Default: R3KYC01FW1P
  QNN_SOC_MODEL            Default: SM8750
  QNN_MODEL_MODE           Default: hybrid
  QNN_PREFILL_AR_LEN       Default: 16
  QNN_QUANT                Default for all QNN components. Default: fp16 (QNN HTP fp16 compile)
  QNN_VISION_QUANT         QNN vision quant mode. Default: $QNN_QUANT
  QNN_DECODER_QUANT        QNN decoder quant mode. Default: $QNN_QUANT
  QNN_EMBEDDING_QUANT      QNN embedding quant mode. Default: $QNN_QUANT
  QNN_EXTRA_ARGS           Extra args appended to each QNN export command
  QNN_PROMPT               Default: Can you describe this image?
  QNN_IMAGE_PATH           Default: COCO sample image URL

XNNPACK overrides:
  XNNPACK_DTYPE            Default: fp16
  XNNPACK_VISION_QUANT     Default: fp16
  XNNPACK_DECODER_QUANT    Default: fp16
  XNNPACK_EMBEDDING_QUANT  Default: fp16
  XNNPACK_EXTRA_ARGS       Extra args appended to each XNNPACK export command
  DYNAMIC_SHAPE            If 0/false, disable XNNPACK dynamic sequence shapes. Default: 1

Vulkan overrides:
  VULKAN_DTYPE             Default: fp16
  VULKAN_VISION_QUANT      Default: fp16
  VULKAN_DECODER_QUANT     Default: fp16
  VULKAN_EMBEDDING_QUANT   Default: fp16
  VULKAN_DECODER_INPUT_MODE Default: embeddings
  VULKAN_USE_SDPA_WITH_KV_CACHE If 0/false, pass --no-use_sdpa_with_kv_cache. Default: 1
  VULKAN_ARTIFACT_LABEL     Override artifact quant/tag suffix after length.
                            Default: $VULKAN_DECODER_QUANT
  VULKAN_EXTRA_ARGS        Extra args appended to each Vulkan export command
EOF
}

log() {
  printf '\n==> %s\n' "$*"
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FOUNDATION_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd -- "${FOUNDATION_DIR}/../.." && pwd)}"
FOUNDATION_MODULE="my_research.foundation.cli"

BACKEND_MODE="${1:-all}"
case "${BACKEND_MODE}" in
  all|qnn|xnnpack|vulkan)
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    die "unknown mode: ${BACKEND_MODE}"
    ;;
esac

if [[ -n "${EXPORT_LENGTHS:-}" ]]; then
  read -r -a LENGTHS <<< "${EXPORT_LENGTHS}"
else
  LENGTHS=(512 1024 2048 4096 8192 16384)
fi

if [[ -n "${EXPORT_MODELS:-}" ]]; then
  read -r -a DECODER_MODELS <<< "${EXPORT_MODELS}"
else
  DECODER_MODELS=(internvl3_1b internvl3_2b internvl3_8b)
fi

HF_MODEL_ROOT="${HF_MODEL_ROOT:-${ROOT_DIR}/my_research/foundation/results/model/hf}"
QNN_ARTIFACT_BASE="${QNN_ARTIFACT_BASE:-${ROOT_DIR}/my_research/foundation/results/model/qnn}"
XNNPACK_ARTIFACT_BASE="${XNNPACK_ARTIFACT_BASE:-${ROOT_DIR}/my_research/foundation/results/model/xnnpack}"
VULKAN_ARTIFACT_BASE="${VULKAN_ARTIFACT_BASE:-${ROOT_DIR}/my_research/foundation/results/model/vulkan}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

QNN_BUILD_PATH="${QNN_BUILD_PATH:-${ROOT_DIR}/executorch/build-android}"
QNN_DEVICE="${QNN_DEVICE:-R3KYC01FW1P}"
QNN_SOC_MODEL="${QNN_SOC_MODEL:-SM8750}"
QNN_MODEL_MODE="${QNN_MODEL_MODE:-hybrid}"
QNN_PREFILL_AR_LEN="${QNN_PREFILL_AR_LEN:-16}"
QNN_QUANT="${QNN_QUANT:-fp16}"
QNN_VISION_QUANT="${QNN_VISION_QUANT:-${QNN_QUANT}}"
QNN_DECODER_QUANT="${QNN_DECODER_QUANT:-${QNN_QUANT}}"
QNN_EMBEDDING_QUANT="${QNN_EMBEDDING_QUANT:-${QNN_QUANT}}"
QNN_PROMPT="${QNN_PROMPT:-Can you describe this image?}"
QNN_IMAGE_PATH="${QNN_IMAGE_PATH:-http://images.cocodataset.org/val2017/000000039769.jpg}"
read -r -a QNN_EXTRA_ARGS_ARRAY <<< "${QNN_EXTRA_ARGS:-}"

XNNPACK_DTYPE="${XNNPACK_DTYPE:-fp16}"
XNNPACK_VISION_QUANT="${XNNPACK_VISION_QUANT:-fp16}"
XNNPACK_DECODER_QUANT="${XNNPACK_DECODER_QUANT:-fp16}"
XNNPACK_EMBEDDING_QUANT="${XNNPACK_EMBEDDING_QUANT:-fp16}"
read -r -a XNNPACK_EXTRA_ARGS_ARRAY <<< "${XNNPACK_EXTRA_ARGS:-}"
DYNAMIC_SHAPE="${DYNAMIC_SHAPE:-1}"

VULKAN_DTYPE="${VULKAN_DTYPE:-fp16}"
VULKAN_VISION_QUANT="${VULKAN_VISION_QUANT:-fp16}"
VULKAN_DECODER_QUANT="${VULKAN_DECODER_QUANT:-fp16}"
VULKAN_EMBEDDING_QUANT="${VULKAN_EMBEDDING_QUANT:-fp16}"
VULKAN_DECODER_INPUT_MODE="${VULKAN_DECODER_INPUT_MODE:-embeddings}"
VULKAN_USE_SDPA_WITH_KV_CACHE="${VULKAN_USE_SDPA_WITH_KV_CACHE:-1}"
VULKAN_ARTIFACT_LABEL="${VULKAN_ARTIFACT_LABEL:-${VULKAN_DECODER_QUANT}}"
read -r -a VULKAN_EXTRA_ARGS_ARRAY <<< "${VULKAN_EXTRA_ARGS:-}"

length_tag() {
  local length="$1"
  if (( length % 1024 == 0 )); then
    echo "$((length / 1024))k"
  else
    echo "${length}"
  fi
}

model_size_tag() {
  local decoder_model="$1"
  echo "${decoder_model#internvl3_}"
}

qnn_quant_tag_value() {
  local quant="${1,,}"
  case "${quant}" in
    fp16) echo "fp16" ;;
    16a16w) echo "16a16w" ;;
    16a8w|16a4w|16a4w_block|8a8w|8a4w) echo "${quant}" ;;
    *) die "unsupported QNN quant mode for artifact tag: ${1}" ;;
  esac
}

qnn_artifact_quant_tag() {
  local vision
  local decoder
  local embedding
  vision="$(qnn_quant_tag_value "${QNN_VISION_QUANT}")"
  decoder="$(qnn_quant_tag_value "${QNN_DECODER_QUANT}")"
  embedding="$(qnn_quant_tag_value "${QNN_EMBEDDING_QUANT}")"

  if [[ "${vision}" == "${decoder}" && "${decoder}" == "${embedding}" ]]; then
    echo "${decoder}"
  else
    echo "v${vision}_d${decoder}_e${embedding}"
  fi
}

hf_model_dirname() {
  local decoder_model="$1"
  case "${decoder_model}" in
    internvl3_1b) echo "InternVL3-1B-hf" ;;
    internvl3_2b) echo "InternVL3-2B-hf" ;;
    internvl3_8b) echo "InternVL3-8B-hf" ;;
    *) die "unsupported decoder model: ${decoder_model}" ;;
  esac
}

local_model_path() {
  local decoder_model="$1"
  echo "${HF_MODEL_ROOT}/$(hf_model_dirname "${decoder_model}")"
}

local_checkpoint_path() {
  local decoder_model="$1"
  echo "${HF_MODEL_ROOT}/${decoder_model}_meta_cpu.pth"
}

maybe_skip_existing() {
  local artifact_root="$1"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${artifact_root}/manifest.json" ]]; then
    log "Skipping existing artifact: ${artifact_root}"
    return 0
  fi
  return 1
}

append_local_model_args() {
  local -n cmd_ref="$1"
  local decoder_model="$2"
  local model_path
  local checkpoint
  model_path="$(local_model_path "${decoder_model}")"
  checkpoint="$(local_checkpoint_path "${decoder_model}")"

  if [[ -d "${model_path}" ]]; then
    cmd_ref+=(--model_path "${model_path}")
  fi
  if [[ -f "${checkpoint}" ]]; then
    cmd_ref+=(--checkpoint "${checkpoint}")
  fi
}

run_qnn_export() {
  local decoder_model="$1"
  local length="$2"
  local model_size
  local tag
  local quant_tag
  local artifact_root
  model_size="$(model_size_tag "${decoder_model}")"
  tag="$(length_tag "${length}")"
  quant_tag="$(qnn_artifact_quant_tag)"
  artifact_root="${QNN_ARTIFACT_BASE}/internvl3_${model_size}_${QNN_MODEL_MODE}_${QNN_PREFILL_AR_LEN}p_${tag}_${quant_tag}"

  maybe_skip_existing "${artifact_root}" && return 0

  local -a cmd=(
    python -m "${FOUNDATION_MODULE}" export
    --backend qnn
    --artifact_root "${artifact_root}"
    --decoder_model "${decoder_model}"
    --build_path "${QNN_BUILD_PATH}"
    --device "${QNN_DEVICE}"
    --model "${QNN_SOC_MODEL}"
    --model_mode "${QNN_MODEL_MODE}"
    --prefill_ar_len "${QNN_PREFILL_AR_LEN}"
    --max_seq_len "${length}"
    --max_context_len "${length}"
    --dtype fp32
    --vision_quant "${QNN_VISION_QUANT}"
    --decoder_quant "${QNN_DECODER_QUANT}"
    --embedding_quant "${QNN_EMBEDDING_QUANT}"
    --prompts "${QNN_PROMPT}"
    --image_path "${QNN_IMAGE_PATH}"
    "${QNN_EXTRA_ARGS_ARRAY[@]}"
  )
  append_local_model_args cmd "${decoder_model}"

  log "QNN export: ${decoder_model}, ${length} (${artifact_root})"
  "${cmd[@]}"
}

run_xnnpack_export() {
  local decoder_model="$1"
  local length="$2"
  local model_size
  local tag
  local artifact_root
  model_size="$(model_size_tag "${decoder_model}")"
  tag="$(length_tag "${length}")"
  artifact_root="${XNNPACK_ARTIFACT_BASE}/internvl3_xnnpack_${model_size}_${tag}_${XNNPACK_DTYPE}"

  local shape_label="dynamic"
  local -a dynamic_shape_args=(--dynamic_shape)
  if [[ "${DYNAMIC_SHAPE}" == "0" || "${DYNAMIC_SHAPE}" == "false" ]]; then
    shape_label="static"
    dynamic_shape_args=(--disable_dynamic_shape)
    artifact_root="${artifact_root}_static"
  fi

  maybe_skip_existing "${artifact_root}" && return 0

  local -a cmd=(
    python -m "${FOUNDATION_MODULE}" export
    --backend xnnpack
    --artifact_root "${artifact_root}"
    --decoder_model "${decoder_model}"
    --max_seq_len "${length}"
    --max_context_len "${length}"
    --dtype "${XNNPACK_DTYPE}"
    --vision_quant "${XNNPACK_VISION_QUANT}"
    --decoder_quant "${XNNPACK_DECODER_QUANT}"
    --embedding_quant "${XNNPACK_EMBEDDING_QUANT}"
    "${dynamic_shape_args[@]}"
    "${XNNPACK_EXTRA_ARGS_ARRAY[@]}"
  )
  append_local_model_args cmd "${decoder_model}"

  log "XNNPACK export: ${decoder_model}, ${length}, ${shape_label} shape (${artifact_root})"
  "${cmd[@]}"
}

run_vulkan_export() {
  local decoder_model="$1"
  local length="$2"
  local model_size
  local tag
  local artifact_root
  model_size="$(model_size_tag "${decoder_model}")"
  tag="$(length_tag "${length}")"
  artifact_root="${VULKAN_ARTIFACT_BASE}/internvl3_vulkan_${model_size}_${tag}_${VULKAN_ARTIFACT_LABEL}"

  maybe_skip_existing "${artifact_root}" && return 0

  local -a sdpa_args=(--use_sdpa_with_kv_cache)
  if [[ "${VULKAN_USE_SDPA_WITH_KV_CACHE}" == "0" || "${VULKAN_USE_SDPA_WITH_KV_CACHE}" == "false" ]]; then
    sdpa_args=(--no-use_sdpa_with_kv_cache)
  fi

  local -a cmd=(
    python -m "${FOUNDATION_MODULE}" export
    --backend vulkan
    --artifact_root "${artifact_root}"
    --decoder_model "${decoder_model}"
    --max_seq_len "${length}"
    --max_context_len "${length}"
    --dtype "${VULKAN_DTYPE}"
    --vision_quant "${VULKAN_VISION_QUANT}"
    --decoder_quant "${VULKAN_DECODER_QUANT}"
    --embedding_quant "${VULKAN_EMBEDDING_QUANT}"
    --decoder_input_mode "${VULKAN_DECODER_INPUT_MODE}"
    --dynamic_shape
    "${sdpa_args[@]}"
    "${VULKAN_EXTRA_ARGS_ARRAY[@]}"
  )
  append_local_model_args cmd "${decoder_model}"

  log "Vulkan export: ${decoder_model}, ${length} (${artifact_root})"
  "${cmd[@]}"
}

cd "${ROOT_DIR}"

for decoder_model in "${DECODER_MODELS[@]}"; do
  for length in "${LENGTHS[@]}"; do
    if [[ "${BACKEND_MODE}" == "all" || "${BACKEND_MODE}" == "qnn" ]]; then
      run_qnn_export "${decoder_model}" "${length}"
    fi
    if [[ "${BACKEND_MODE}" == "all" || "${BACKEND_MODE}" == "xnnpack" ]]; then
      run_xnnpack_export "${decoder_model}" "${length}"
    fi
    if [[ "${BACKEND_MODE}" == "all" || "${BACKEND_MODE}" == "vulkan" ]]; then
      run_vulkan_export "${decoder_model}" "${length}"
    fi
  done
done
