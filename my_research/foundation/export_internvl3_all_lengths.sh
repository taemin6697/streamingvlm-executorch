#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/streamingvlm}"
FOUNDATION_MODULE="my_research.foundation.cli"

BACKEND_MODE="${1:-all}"
if [[ -n "${EXPORT_LENGTHS:-}" ]]; then
  read -r -a LENGTHS <<< "${EXPORT_LENGTHS}"
else
  LENGTHS=(1024 2048 4096 8192 16384 32768)
fi
if [[ -n "${EXPORT_MODELS:-}" ]]; then
  read -r -a DECODER_MODELS <<< "${EXPORT_MODELS}"
elif [[ -n "${DECODER_MODEL:-}" ]]; then
  DECODER_MODELS=("${DECODER_MODEL}")
else
  DECODER_MODELS=(internvl3_1b internvl3_2b internvl3_8b)
fi

XNN_ARTIFACT_BASE="${XNN_ARTIFACT_BASE:-${ROOT_DIR}/my_save/save_model/cpu}"
QNN_ARTIFACT_BASE="${QNN_ARTIFACT_BASE:-${ROOT_DIR}/my_save/save_model/qnn}"

QNN_BUILD_PATH="${QNN_BUILD_PATH:-${ROOT_DIR}/executorch/build-android}"
QNN_DEVICE="${QNN_DEVICE:-R3KYC01FW1P}"
QNN_SOC_MODEL="${QNN_SOC_MODEL:-SM8750}"
QNN_MODEL_MODE="${QNN_MODEL_MODE:-hybrid}"
QNN_PREFILL_AR_LEN="${QNN_PREFILL_AR_LEN:-16}"
QNN_PROMPT="${QNN_PROMPT:-Can you describe this image?}"
QNN_IMAGE_PATH="${QNN_IMAGE_PATH:-http://images.cocodataset.org/val2017/000000039769.jpg}"

case "${BACKEND_MODE}" in
  all|xnnpack|qnn)
    ;;
  *)
    echo "Usage: $0 [all|xnnpack|qnn]" >&2
    exit 1
    ;;
esac

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

default_model_path() {
  local decoder_model="$1"
  case "${decoder_model}" in
    internvl3_1b) echo "${ROOT_DIR}/my_save/save_model/meta/InternVL3-1B-hf" ;;
    internvl3_2b) echo "${ROOT_DIR}/my_save/save_model/meta/InternVL3-2B-hf" ;;
    internvl3_8b) echo "${ROOT_DIR}/my_save/save_model/meta/InternVL3-8B-hf" ;;
    *)
      echo "Unsupported decoder model: ${decoder_model}" >&2
      exit 1
      ;;
  esac
}

default_checkpoint_path() {
  local decoder_model="$1"
  case "${decoder_model}" in
    internvl3_1b) echo "${ROOT_DIR}/my_save/save_model/meta/internvl3_1b_meta_cpu.pth" ;;
    internvl3_2b) echo "${ROOT_DIR}/my_save/save_model/meta/internvl3_2b_meta_cpu.pth" ;;
    internvl3_8b) echo "${ROOT_DIR}/my_save/save_model/meta/internvl3_8b_meta_cpu.pth" ;;
    *)
      echo "Unsupported decoder model: ${decoder_model}" >&2
      exit 1
      ;;
  esac
}

run_xnnpack_export() {
  local decoder_model="$1"
  local model_size
  model_size="$(model_size_tag "${decoder_model}")"
  local length="$2"
  local tag
  tag="$(length_tag "${length}")"
  local model_path="${MODEL_PATH:-$(default_model_path "${decoder_model}")}"
  local checkpoint="${CHECKPOINT:-$(default_checkpoint_path "${decoder_model}")}"
  local artifact_root="${XNN_ARTIFACT_BASE}/internvl3_xnnpack_${model_size}_${tag}"

  echo "==> XNNPACK export: ${decoder_model}, ${length} (${artifact_root})"
  python -m "${FOUNDATION_MODULE}" export \
    --backend xnnpack \
    --artifact_root "${artifact_root}" \
    --decoder_model "${decoder_model}" \
    --model_path "${model_path}" \
    --checkpoint "${checkpoint}" \
    --max_seq_len "${length}" \
    --max_context_len "${length}" \
    --dtype fp16 \
    --vision_quant fp16 \
    --decoder_quant fp16 \
    --embedding_quant fp16
}

run_qnn_export() {
  local decoder_model="$1"
  local model_size
  model_size="$(model_size_tag "${decoder_model}")"
  local length="$2"
  local tag
  tag="$(length_tag "${length}")"
  local artifact_root="${QNN_ARTIFACT_BASE}/internvl3_${model_size}_${QNN_MODEL_MODE}_${QNN_PREFILL_AR_LEN}p_${tag}"

  echo "==> QNN export: ${decoder_model}, ${length} (${artifact_root})"
  python -m "${FOUNDATION_MODULE}" export \
    --backend qnn \
    --artifact_root "${artifact_root}" \
    --decoder_model "${decoder_model}" \
    --build_path "${QNN_BUILD_PATH}" \
    --device "${QNN_DEVICE}" \
    --model "${QNN_SOC_MODEL}" \
    --model_mode "${QNN_MODEL_MODE}" \
    --prefill_ar_len "${QNN_PREFILL_AR_LEN}" \
    --max_seq_len "${length}" \
    --max_context_len "${length}" \
    --dtype fp32 \
    --vision_quant fp16 \
    --decoder_quant fp16 \
    --embedding_quant fp16 \
    --prompts "${QNN_PROMPT}" \
    --image_path "${QNN_IMAGE_PATH}"
}

cd "${ROOT_DIR}"

for decoder_model in "${DECODER_MODELS[@]}"; do
  for length in "${LENGTHS[@]}"; do
    case "${BACKEND_MODE}" in
      all)
        run_xnnpack_export "${decoder_model}" "${length}"
        run_qnn_export "${decoder_model}" "${length}"
        ;;
      xnnpack)
        run_xnnpack_export "${decoder_model}" "${length}"
        ;;
      qnn)
        run_qnn_export "${decoder_model}" "${length}"
        ;;
    esac
  done
done
