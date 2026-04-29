#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/streamingvlm}
SEQ_LEN=${SEQ_LEN:-512}
TAG=${TAG:-sdpa_kv_fp32_island}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-"${ROOT}/my_research/foundation/results/model/vulkan/internvl3_vulkan_1b_${SEQ_LEN}_fp16_${TAG}"}

cd "${ROOT}"

PYTHONPATH="${ROOT}:${ROOT}/executorch" \
python -m my_research.foundation.cli export \
  --backend vulkan \
  --artifact_root "${ARTIFACT_ROOT}" \
  --decoder_model internvl3_1b \
  --model_path "${ROOT}/my_research/foundation/results/model/hf/InternVL3-1B-hf" \
  --checkpoint "${ROOT}/my_research/foundation/results/model/hf/internvl3_1b_meta_cpu.pth" \
  --max_seq_len "${SEQ_LEN}" \
  --max_context_len "${SEQ_LEN}" \
  --dtype fp16 \
  --vision_quant fp16 \
  --decoder_quant fp16 \
  --embedding_quant fp16 \
  --decoder_input_mode embeddings \
  --dynamic_shape \
  --use_sdpa_with_kv_cache \
  --vulkan_debug_fp32_kv_cache
