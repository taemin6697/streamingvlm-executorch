#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python3}"
VISION="${VISION:-my_research/foundation_llamacpp/results/vision_models/internvl3_1b_vision_tower_preproj_qnn_realweights_sm8750/vision_tower_preproj_qnn.pte}"
LLAMA_BUILD_DIR="${LLAMA_BUILD_DIR:-my_research/foundation_llamacpp/build-hybrid-android-opencl}"
MODEL="${MODEL:-llama.cpp/models/InternVL3-2B-Instruct-GGUF/InternVL3-2B-Instruct-Q8_0.gguf}"
MMPROJ="${MMPROJ:-llama.cpp/models/InternVL3-2B-Instruct-GGUF/mmproj-InternVL3-2B-Instruct-Q8_0.gguf}"
VIDEO="${VIDEO:-my_research/foundation_llamacpp/sample_images/surveil_8.mp4}"
RESULTS_ROOT="${RESULTS_ROOT:-my_research/foundation_llamacpp/results/log/dynamic_kv_grow_sweep_2b_q8_30s_3prompt}"
REMOTE_ROOT="${REMOTE_ROOT:-/data/local/tmp/streamingvlm_2b_kv_grow_sweep}"

KV_INIT="${KV_INIT:-512}"
KV_STEPS=(${KV_STEPS:-512 1024 2048 4096 8192 16384 32768})
TIMES="${TIMES:-[5.0, 15.0, 25.0]}"
PROMPTS="${PROMPTS:-[\"What is happening in the scene?\", \"What changed recently?\", \"Summarize the situation so far.\"]}"
MAX_VIDEO_TIME="${MAX_VIDEO_TIME:-30.0}"
N_PREDICT="${N_PREDICT:-64}"

RUN_NAME="InternVL3-2B-Instruct-Q8_0_hybrid_ctx_32768_streaming_vision_prefill_kv16_dynamic_online"
SUMMARY="${RESULTS_ROOT}/sweep_summary.tsv"
mkdir -p "${RESULTS_ROOT}"
printf "kv_init\tkv_grow_step\trc\tresult_dir\tcommitted_cache_updates\tcommitted_cache_fps\tcache_worker_fps\tprompt_decode_total_s\tdynamic_kv_grow_count\n" > "${SUMMARY}"

for step in "${KV_STEPS[@]}"; do
  step_root="${RESULTS_ROOT}/kv_grow_${step}"
  result_dir="${step_root}/${RUN_NAME}"
  echo "===== dynamic KV grow sweep: init=${KV_INIT}, grow_step=${step} ====="

  "${PYTHON}" my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
    --processor hybrid \
    --vision "${VISION}" \
    --llama-build-dir "${LLAMA_BUILD_DIR}" \
    --model "${MODEL}" \
    --mmproj "${MMPROJ}" \
    --streaming-video "${VIDEO}" \
    --stream-mode vision-prefill \
    --sampling-fps 1.0 \
    --max-video-time "${MAX_VIDEO_TIME}" \
    --max-num 1 \
    --time "${TIMES}" \
    --prompt "${PROMPTS}" \
    --n-predict "${N_PREDICT}" \
    --dynamic-kv-cache \
    --kv-init-size "${KV_INIT}" \
    --kv-grow-step "${step}" \
    --threads 4 \
    --gpu-layers 99 \
    --device GPUOpenCL \
    --ctx-size 32768 \
    --batch-size 1024 \
    --ubatch-size 512 \
    --temperature 0.0 \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --fit off \
    --soc-model SM8750 \
    --baseline-window 0.5 \
    --remote-root "${REMOTE_ROOT}" \
    --results-root "${step_root}" \
    --online-buffer

  rc=$?
  if [[ -f "${result_dir}/txt_json/foundation_exit_code.txt" ]]; then
    rc="$(tr -d '\r\n' < "${result_dir}/txt_json/foundation_exit_code.txt")"
  fi

  committed_cache_updates=""
  committed_cache_fps=""
  cache_worker_fps=""
  prompt_decode_total_s=""
  if [[ -f "${result_dir}/txt_json/stream_buffer_summary.txt" ]]; then
    committed_cache_updates="$(awk -F= '$1=="committed_cache_updates"{print $2}' "${result_dir}/txt_json/stream_buffer_summary.txt")"
    committed_cache_fps="$(awk -F= '$1=="committed_cache_fps"{print $2}' "${result_dir}/txt_json/stream_buffer_summary.txt")"
    cache_worker_fps="$(awk -F= '$1=="cache_worker_fps"{print $2}' "${result_dir}/txt_json/stream_buffer_summary.txt")"
    prompt_decode_total_s="$(awk -F= '$1=="prompt_decode_total_s"{print $2}' "${result_dir}/txt_json/stream_buffer_summary.txt")"
  fi

  dynamic_kv_grow_count=""
  if [[ -f "${result_dir}/csv/foundation_proc.csv" ]]; then
    dynamic_kv_grow_count="$(awk -F, '$1=="DynamicKVGrow"{count++} END{print count+0}' "${result_dir}/csv/foundation_proc.csv")"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${KV_INIT}" \
    "${step}" \
    "${rc}" \
    "${result_dir}" \
    "${committed_cache_updates}" \
    "${committed_cache_fps}" \
    "${cache_worker_fps}" \
    "${prompt_decode_total_s}" \
    "${dynamic_kv_grow_count}" >> "${SUMMARY}"

  echo "===== completed grow_step=${step}, rc=${rc} ====="
done

echo "Sweep summary: ${SUMMARY}"
