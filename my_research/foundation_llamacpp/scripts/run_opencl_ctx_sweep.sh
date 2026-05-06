#!/usr/bin/env bash
# Sweep llama.cpp Android multimodal baseline over ctx sizes (golden_gate sample).
# Default model: InternVL3-1B-Instruct Q8_0 + mmproj Q8_0.
# CTX_SIZES: 512 … 32768 (batch=min(ctx,2048), ubatch=min(batch,512)).
#
# Usage (repo root): ./my_research/foundation_llamacpp/scripts/run_opencl_ctx_sweep.sh
#
# Processor:
#   PROCESSOR=gpu   — OpenCL via opencl_phase_mtmd (default)
#   PROCESSOR=cpu   — CPU via llama-mtmd-cli (needs LLAMA_BUILD_CPU)
#   PROCESSOR=both  — full GPU sweep then full CPU sweep
#
# Builds:
#   LLAMA_BUILD_GPU — default: my_research/foundation_llamacpp/build-hybrid-android-opencl
#   LLAMA_BUILD_CPU — default: llama.cpp/build-android-cpu-noomp (see README.md)
#
# Remote dirs (avoid GPU/CPU binary cache clashes when running both):
#   REMOTE_ROOT_GPU — default /data/local/tmp/streamingvlm_unified
#   REMOTE_ROOT_CPU — default /data/local/tmp/streamingvlm_cpu_vlm
#
# Override artifacts:
#   MODEL=... MMPROJ=... ./my_research/foundation_llamacpp/scripts/run_opencl_ctx_sweep.sh
# 2B Q4 example:
#   MODEL="${REPO_ROOT}/llama.cpp/models/InternVL3-2B-Instruct-GGUF/InternVL3-2B-Instruct-Q4_K_M.gguf" \
#   MMPROJ="${REPO_ROOT}/llama.cpp/models/InternVL3-2B-Instruct-GGUF/mmproj-InternVL3-2B-Instruct-Q8_0.gguf" \
#     ./my_research/foundation_llamacpp/scripts/run_opencl_ctx_sweep.sh
#
# If the device reused a truncated GGUF ("tensor ... not within the file bounds"), refresh once:
#   MODEL_PUSH=1 ./my_research/foundation_llamacpp/scripts/run_opencl_ctx_sweep.sh
# Full remote workdir reset (heavy): FORCE_PUSH=1 MODEL_PUSH=1 ./my_research/.../run_opencl_ctx_sweep.sh
#
# KV cache dtype (passed through as llama.cpp --cache-type-k / --cache-type-v).
# Default f16 (baseline). For quantized KV experiments rebuild llama.cpp with OpenCL FA patch;
# see `foundation_llamacpp/docs/for_cursor_llm_llamacpp.md` (PR #21313 cherry-pick). Example:
#   CACHE_TYPE_K=q8_0 CACHE_TYPE_V=q8_0 PROCESSOR=gpu ./my_research/foundation_llamacpp/scripts/run_opencl_ctx_sweep.sh
#
# Requires: adb device, Android binaries pushed per README.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

CTX_SIZES=(512 1024 2048 4096 8192 16384 32768)

MODEL="${MODEL:-${REPO_ROOT}/llama.cpp/models/InternVL3-1B-Instruct-GGUF/InternVL3-1B-Instruct-Q8_0.gguf}"
MMPROJ="${MMPROJ:-${REPO_ROOT}/llama.cpp/models/InternVL3-1B-Instruct-GGUF/mmproj-InternVL3-1B-Instruct-Q8_0.gguf}"
MODEL_PUSH="${MODEL_PUSH:-0}"
FORCE_PUSH="${FORCE_PUSH:-0}"
PROCESSOR="${PROCESSOR:-gpu}"

LLAMA_BUILD_GPU="${LLAMA_BUILD_GPU:-${REPO_ROOT}/my_research/foundation_llamacpp/build-hybrid-android-opencl}"
LLAMA_BUILD_CPU="${LLAMA_BUILD_CPU:-${REPO_ROOT}/llama.cpp/build-android-cpu-noomp}"
REMOTE_ROOT_GPU="${REMOTE_ROOT_GPU:-/data/local/tmp/streamingvlm_unified}"
REMOTE_ROOT_CPU="${REMOTE_ROOT_CPU:-/data/local/tmp/streamingvlm_cpu_vlm}"
THREADS="${THREADS:-4}"
CACHE_TYPE_K="${CACHE_TYPE_K:-f16}"
CACHE_TYPE_V="${CACHE_TYPE_V:-f16}"

runner_py="${REPO_ROOT}/my_research/foundation_llamacpp/run_android_hybrid_bridge.py"

run_one() {
  local backend="$1"
  local ctx="$2"
  local batch ubatch
  local push_flags=()
  batch=$((ctx < 2048 ? ctx : 2048))
  ubatch=$((batch < 512 ? batch : 512))
  [[ "${MODEL_PUSH}" == "1" ]] && push_flags+=(--model-push)
  [[ "${FORCE_PUSH}" == "1" ]] && push_flags+=(--force-push)

  local build_dir remote_root label
  if [[ "${backend}" == "gpu" ]]; then
    build_dir="${LLAMA_BUILD_GPU}"
    remote_root="${REMOTE_ROOT_GPU}"
    label="GPU OpenCL"
  else
    build_dir="${LLAMA_BUILD_CPU}"
    remote_root="${REMOTE_ROOT_CPU}"
    label="CPU"
  fi

  echo "======== [${label}] ctx=${ctx} batch=${batch} ubatch=${ubatch} model=$(basename "${MODEL}") build=$(basename "${build_dir}") ========"
  if [[ "${backend}" == "gpu" ]]; then
    python3 "${runner_py}" \
      --processor gpu \
      --llama-build-dir "${build_dir}" \
      --model "${MODEL}" \
      --mmproj "${MMPROJ}" \
      --image "${REPO_ROOT}/my_research/foundation_llamacpp/sample_images/golden_gate_bridge_448.jpg" \
      --prompt "Describe this image briefly." \
      --n-predict 32 \
      --force-generation 64 \
      --threads "${THREADS}" \
      --gpu-layers 99 \
      --device GPUOpenCL \
      --ctx-size "${ctx}" \
      --batch-size "${batch}" \
      --ubatch-size "${ubatch}" \
      --temperature 0.0 \
      --cache-type-k "${CACHE_TYPE_K}" \
      --cache-type-v "${CACHE_TYPE_V}" \
      --baseline-window 5.0 \
      --remote-root "${remote_root}" \
      --results-root "${REPO_ROOT}/my_research/foundation_llamacpp/results/log" \
      "${push_flags[@]}"
  else
    python3 "${runner_py}" \
      --processor cpu \
      --llama-build-dir "${build_dir}" \
      --model "${MODEL}" \
      --mmproj "${MMPROJ}" \
      --image "${REPO_ROOT}/my_research/foundation_llamacpp/sample_images/golden_gate_bridge_448.jpg" \
      --prompt "Describe this image briefly." \
      --n-predict 32 \
      --force-generation 64 \
      --threads "${THREADS}" \
      --ctx-size "${ctx}" \
      --batch-size "${batch}" \
      --ubatch-size "${ubatch}" \
      --temperature 0.0 \
      --cache-type-k "${CACHE_TYPE_K}" \
      --cache-type-v "${CACHE_TYPE_V}" \
      --baseline-window 5.0 \
      --remote-root "${remote_root}" \
      --results-root "${REPO_ROOT}/my_research/foundation_llamacpp/results/log" \
      "${push_flags[@]}"
  fi
}

failed=()

run_sweep() {
  local backend="$1"
  local ctx
  for ctx in "${CTX_SIZES[@]}"; do
    if ! run_one "${backend}" "${ctx}"; then
      failed+=("${backend}:${ctx}")
      echo "FAILED ${backend} ctx=${ctx}" >&2
    fi
  done
}

case "${PROCESSOR}" in
  gpu)
    run_sweep gpu
    ;;
  cpu)
    run_sweep cpu
    ;;
  both)
    run_sweep gpu
    run_sweep cpu
    ;;
  *)
    printf 'error: PROCESSOR must be gpu, cpu, or both (got %s)\n' "${PROCESSOR}" >&2
    exit 2
    ;;
esac

if ((${#failed[@]})); then
  echo "Sweep finished with failures: ${failed[*]}" >&2
  exit 1
fi
echo "Sweep finished OK (PROCESSOR=${PROCESSOR}, ${#CTX_SIZES[@]} ctx steps per backend)."
