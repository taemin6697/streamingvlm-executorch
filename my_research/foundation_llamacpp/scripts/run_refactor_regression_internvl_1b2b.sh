#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR}"
cd "$ROOT_DIR"

SERIAL="${ADB_SERIAL:-}"
REMOTE_ROOT="${REMOTE_ROOT:-/data/local/tmp/streamingvlm_refactor_regression}"
INTERNVL_1B_REMOTE_ROOT="${INTERNVL_1B_REMOTE_ROOT:-$REMOTE_ROOT}"
INTERNVL_2B_REMOTE_ROOT="${INTERNVL_2B_REMOTE_ROOT:-$REMOTE_ROOT}"
RESULTS_ROOT="${RESULTS_ROOT:-my_research/foundation_llamacpp/results/log/refactor_regression_internvl_1b2b_$(date +%Y%m%d_%H%M%S)}"
BUILD_DIR="${BUILD_DIR:-my_research/foundation_llamacpp/build-hybrid-android-opencl}"
CTX_SIZE="${CTX_SIZE:-32768}"
N_PREDICT="${N_PREDICT:-48}"
STREAM_SECONDS="${STREAM_SECONDS:-16.0}"
STREAM_TIMES="${STREAM_TIMES:-[5.0, 8.0, 11.0, 14.0]}"
STREAM_PROMPTS="${STREAM_PROMPTS:-[\"What is this situation?\", \"In the conversation history above, what was the user first question? Repeat that question exactly and output nothing else.\", \"What changed in the scene?\", \"Summarize the full situation so far.\"]}"
STREAM_UBATCH="${STREAM_UBATCH:-64}"
OFFLINE_UBATCH="${OFFLINE_UBATCH:-512}"
KV_INIT_SIZE="${KV_INIT_SIZE:-512}"
KV_GROW_STEP="${KV_GROW_STEP:-512}"
KV_TYPE="${KV_TYPE:-f16}"
ADB_WAIT_SECONDS="${ADB_WAIT_SECONDS:-180}"

IMAGE_A="${IMAGE_A:-$DATA_ROOT/my_research/foundation_llamacpp/sample_images/golden_gate_bridge_448.jpg}"
IMAGE_B="${IMAGE_B:-$DATA_ROOT/my_research/foundation_llamacpp/sample_images/sample_coco_cats_448.jpg}"
VIDEO="${VIDEO:-$DATA_ROOT/my_research/foundation_llamacpp/sample_images/surveil_8_20sec.mp4}"
VISION="${VISION:-$DATA_ROOT/my_research/foundation_llamacpp/results/vision_models/internvl3_1b_vision_tower_preproj_qnn_realweights_sm8750/vision_tower_preproj_qnn.pte}"

INTERNVL_1B_MODEL="${INTERNVL_1B_MODEL:-$DATA_ROOT/llama.cpp/models/InternVL3-1B-Instruct-GGUF/InternVL3-1B-Instruct-Q8_0.gguf}"
INTERNVL_1B_MMPROJ="${INTERNVL_1B_MMPROJ:-$DATA_ROOT/llama.cpp/models/InternVL3-1B-Instruct-GGUF/mmproj-InternVL3-1B-Instruct-Q8_0.gguf}"
INTERNVL_2B_MODEL="${INTERNVL_2B_MODEL:-$DATA_ROOT/llama.cpp/models/InternVL3-2B-Instruct-GGUF/InternVL3-2B-Instruct-Q8_0.gguf}"
INTERNVL_2B_MMPROJ="${INTERNVL_2B_MMPROJ:-$DATA_ROOT/llama.cpp/models/InternVL3-2B-Instruct-GGUF/mmproj-InternVL3-2B-Instruct-Q8_0.gguf}"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export QNN_SDK_ROOT="${QNN_SDK_ROOT:-$DATA_ROOT/executorch/backends/qualcomm/sdk/qnn/qairt/2.37.0.250724}"
export EXECUTORCH_ROOT="${EXECUTORCH_ROOT:-$DATA_ROOT/executorch}"

ADB=(adb)
if [[ -n "$SERIAL" ]]; then
  ADB=(adb -s "$SERIAL")
fi

wait_for_adb() {
  local deadline=$((SECONDS + ADB_WAIT_SECONDS))
  while (( SECONDS < deadline )); do
    if "${ADB[@]}" wait-for-device >/dev/null 2>&1 && "${ADB[@]}" shell true >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "ADB did not become ready within ${ADB_WAIT_SECONDS}s" >&2
  exit 4
}

require_local_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing local input: $1" >&2
    exit 2
  fi
}

remote_has_file() {
  local remote_root="$1"
  local file="$2"
  local name
  name="$(basename "$file")"
  "${ADB[@]}" shell "test -f '$remote_root/$name'"
}

require_remote_model() {
  local remote_root="$1"
  local file="$2"
  wait_for_adb
  if ! remote_has_file "$remote_root" "$file"; then
    echo "Missing remote model artifact: $remote_root/$(basename "$file")" >&2
    echo "This regression script avoids model pushes. Populate the remote root once, then re-run." >&2
    exit 2
  fi
}

verify_run_dir() {
  local run_dir="$1"
  [[ -d "$run_dir/csv" ]]
  [[ -d "$run_dir/png" ]]
  [[ -d "$run_dir/txt_json" ]]
  [[ -s "$run_dir/txt_json/run_command.txt" ]]
  [[ -s "$run_dir/txt_json/foundation_exit_code.txt" ]]
  if [[ "$(tr -d '\r\n' < "$run_dir/txt_json/foundation_exit_code.txt")" != "0" ]]; then
    echo "Non-zero foundation exit code in $run_dir" >&2
    exit 3
  fi
  if [[ -f "$run_dir/txt_json/foundation_inference_tokens.txt" ]]; then
    grep -q -- "--- PREFILL" "$run_dir/txt_json/foundation_inference_tokens.txt"
  fi
  if [[ -f "$run_dir/csv/foundation_proc.csv" ]]; then
    grep -q "Decode" "$run_dir/csv/foundation_proc.csv"
  fi
}

run_common() {
  local remote_root="$1"
  local model="$2"
  local mmproj="$3"
  local ubatch="$4"
  shift 4
  wait_for_adb
  python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
    --processor hybrid \
    --vision "$VISION" \
    --llama-build-dir "$BUILD_DIR" \
    --vision-build-dir "$BUILD_DIR" \
    --model "$model" \
    --mmproj "$mmproj" \
    --prompt-format internvl3 \
    --n-predict "$N_PREDICT" \
    --threads 4 \
    --gpu-layers 99 \
    --device GPUOpenCL \
    --ctx-size "$CTX_SIZE" \
    --batch-size 1024 \
    --ubatch-size "$ubatch" \
    --temperature 0.0 \
    --cache-type-k "$KV_TYPE" \
    --cache-type-v "$KV_TYPE" \
    --fit off \
    --dynamic-kv-cache \
    --kv-init-size "$KV_INIT_SIZE" \
    --kv-grow-step "$KV_GROW_STEP" \
    --soc-model SM8750 \
    --baseline-window 0.5 \
    --remote-root "$remote_root" \
    --results-root "$RESULTS_ROOT" \
    "$@"
}

run_suite() {
  local label="$1"
  local remote_root="$2"
  local model="$3"
  local mmproj="$4"

  echo "== $label =="
  require_local_file "$model"
  require_local_file "$mmproj"
  require_local_file "$VISION"
  require_remote_model "$remote_root" "$model"
  require_remote_model "$remote_root" "$mmproj"
  require_remote_model "$remote_root" "$VISION"

  run_common "$remote_root" "$model" "$mmproj" "$OFFLINE_UBATCH" \
    --image "$IMAGE_A" \
    --prompt "Describe this image briefly."

  run_common "$remote_root" "$model" "$mmproj" "$OFFLINE_UBATCH" \
    --multi-image "$IMAGE_A" "$IMAGE_B" \
    --prompt "Compare these two images briefly."

  run_common "$remote_root" "$model" "$mmproj" "$OFFLINE_UBATCH" \
    --video "$VIDEO" \
    --num-segments 4 \
    --max-num 1 \
    --prompt "What is happening in this video?"

  run_common "$remote_root" "$model" "$mmproj" "$STREAM_UBATCH" \
    --streaming-video "$VIDEO" \
    --stream-mode on-demand \
    --online-buffer \
    --sampling-fps 1.0 \
    --max-video-time "$STREAM_SECONDS" \
    --max-num 1 \
    --time "$STREAM_TIMES" \
    --prompt "$STREAM_PROMPTS"

  run_common "$remote_root" "$model" "$mmproj" "$STREAM_UBATCH" \
    --streaming-video "$VIDEO" \
    --stream-mode sliding-window \
    --online-buffer \
    --sampling-fps 1.0 \
    --max-video-time "$STREAM_SECONDS" \
    --window-sec 4.0 \
    --window-max-frames 8 \
    --max-num 1 \
    --time "$STREAM_TIMES" \
    --prompt "$STREAM_PROMPTS"

  run_common "$remote_root" "$model" "$mmproj" "$STREAM_UBATCH" \
    --streaming-video "$VIDEO" \
    --stream-mode vision-prefill \
    --online-buffer \
    --latest-frame-only \
    --partial-vision-kv \
    --sampling-fps 1.0 \
    --max-video-time "$STREAM_SECONDS" \
    --max-num 1 \
    --time "$STREAM_TIMES" \
    --prompt "$STREAM_PROMPTS"
}

require_local_file "$BUILD_DIR/hybrid_streaming_decode"
require_local_file "$BUILD_DIR/opencl_phase_mtmd"
require_local_file "$IMAGE_A"
require_local_file "$IMAGE_B"
require_local_file "$VIDEO"
mkdir -p "$RESULTS_ROOT"

run_suite "InternVL3-1B" "$INTERNVL_1B_REMOTE_ROOT" "$INTERNVL_1B_MODEL" "$INTERNVL_1B_MMPROJ"
run_suite "InternVL3-2B" "$INTERNVL_2B_REMOTE_ROOT" "$INTERNVL_2B_MODEL" "$INTERNVL_2B_MMPROJ"

while IFS= read -r -d '' run_dir; do
  verify_run_dir "$run_dir"
done < <(find "$RESULTS_ROOT" -mindepth 1 -maxdepth 1 -type d -print0)

find "$RESULTS_ROOT" -maxdepth 2 -type d | sort
