#!/usr/bin/env bash
# Sweep OpenCL GPU baseline over ctx sizes (matches README OpenCL example).
# Usage: from repo root (streamingvlm): ./my_research/foundation_llamacpp/scripts/run_opencl_ctx_sweep.sh
# Requires: adb device, Android binaries pushed per README.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

CTX_SIZES=(512 1024 2048 4096 8192 16384 32768)

runner_py="${REPO_ROOT}/my_research/foundation_llamacpp/run_android_hybrid_bridge.py"

run_one() {
  local ctx="$1"
  local batch ubatch
  batch=$((ctx < 2048 ? ctx : 2048))
  ubatch=$((batch < 512 ? batch : 512))

  echo "======== ctx=${ctx} batch=${batch} ubatch=${ubatch} ========"
  python3 "${runner_py}" \
    --processor gpu \
    --llama-build-dir "${REPO_ROOT}/my_research/foundation_llamacpp/build-hybrid-android-opencl" \
    --model "${REPO_ROOT}/llama.cpp/models/InternVL3-8B-Instruct-GGUF/InternVL3-8B-Instruct-Q4_K_M.gguf" \
    --mmproj "${REPO_ROOT}/llama.cpp/models/InternVL3-8B-Instruct-GGUF/mmproj-InternVL3-8B-Instruct-Q8_0.gguf" \
    --image "${REPO_ROOT}/my_research/foundation_llamacpp/sample_images/golden_gate_bridge_448.jpg" \
    --prompt "Describe this image briefly." \
    --n-predict 32 \
    --force-generation 64 \
    --threads 4 \
    --gpu-layers 99 \
    --device GPUOpenCL \
    --ctx-size "${ctx}" \
    --batch-size "${batch}" \
    --ubatch-size "${ubatch}" \
    --temperature 0.0 \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --baseline-window 5.0 \
    --remote-root /data/local/tmp/streamingvlm_unified \
    --results-root "${REPO_ROOT}/my_research/foundation_llamacpp/results/log"
}

failed=()
for ctx in "${CTX_SIZES[@]}"; do
  if ! run_one "${ctx}"; then
    failed+=("${ctx}")
    echo "FAILED ctx=${ctx}" >&2
  fi
done

if ((${#failed[@]})); then
  echo "Sweep finished with failures: ${failed[*]}" >&2
  exit 1
fi
echo "Sweep finished OK (${#CTX_SIZES[@]} runs)."
