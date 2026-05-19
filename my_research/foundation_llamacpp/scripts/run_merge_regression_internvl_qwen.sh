#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

SERIAL="${ADB_SERIAL:-}"
REMOTE_ROOT="${REMOTE_ROOT:-/data/local/tmp/streamingvlm_merge_regression}"
INTERNVL_REMOTE_ROOT="${INTERNVL_REMOTE_ROOT:-$REMOTE_ROOT}"
QWEN_REMOTE_ROOT="${QWEN_REMOTE_ROOT:-$REMOTE_ROOT}"
RESULTS_ROOT="${RESULTS_ROOT:-my_research/foundation_llamacpp/results/log/merge_regression_internvl_qwen_$(date +%Y%m%d_%H%M%S)}"
BUILD_DIR="${BUILD_DIR:-my_research/foundation_llamacpp/build-hybrid-android-opencl}"
CTX_SIZE="${CTX_SIZE:-32768}"
N_PREDICT="${N_PREDICT:-16}"
STREAM_SECONDS="${STREAM_SECONDS:-6.0}"
STREAM_TIMES="${STREAM_TIMES:-[2.0, 4.0, 6.0]}"
STREAM_PROMPTS="${STREAM_PROMPTS:-[\"What is happening?\", \"What did I ask earlier?\", \"Summarize the scene so far.\"]}"
STREAM_UBATCH="${STREAM_UBATCH:-64}"
OFFLINE_UBATCH="${OFFLINE_UBATCH:-512}"
KV_INIT_SIZE="${KV_INIT_SIZE:-512}"
KV_GROW_STEP="${KV_GROW_STEP:-512}"
KV_TYPE="${KV_TYPE:-f16}"

IMAGE_A="${IMAGE_A:-my_research/foundation_llamacpp/sample_images/golden_gate_bridge_448.jpg}"
IMAGE_B="${IMAGE_B:-my_research/foundation_llamacpp/sample_images/sample_coco_cats_448.jpg}"
VIDEO="${VIDEO:-my_research/foundation_llamacpp/sample_images/surveil_8_20sec.mp4}"

INTERNVL_MODEL="${INTERNVL_MODEL:-llama.cpp/models/InternVL3-1B-Instruct-GGUF/InternVL3-1B-Instruct-Q8_0.gguf}"
INTERNVL_MMPROJ="${INTERNVL_MMPROJ:-llama.cpp/models/InternVL3-1B-Instruct-GGUF/mmproj-InternVL3-1B-Instruct-Q8_0.gguf}"
INTERNVL_VISION="${INTERNVL_VISION:-my_research/foundation_llamacpp/results/vision_models/internvl3_1b_vision_tower_preproj_qnn_realweights_sm8750/vision_tower_preproj_qnn.pte}"

QWEN_MODEL="${QWEN_MODEL:-llama.cpp/models/llama.cpp.models.preserved/llama.cpp.models.preserved/Qwen2.5-VL-3B-Instruct-GGUF/Qwen2.5-VL-3B-Instruct-Q8_0.gguf}"
QWEN_MMPROJ="${QWEN_MMPROJ:-llama.cpp/models/llama.cpp.models.preserved/llama.cpp.models.preserved/Qwen2.5-VL-3B-Instruct-GGUF/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf}"
QWEN_VISION="${QWEN_VISION:-my_research/foundation_llamacpp/results/vision_models/qwen2_5_vl_3b_vision_encoder_premerger_qnn_1024tok_sm8750/vision_encoder_qnn.pte}"

export QNN_SDK_ROOT="${QNN_SDK_ROOT:-$ROOT_DIR/executorch/backends/qualcomm/sdk/qnn/qairt/2.37.0.250724}"
export EXECUTORCH_ROOT="${EXECUTORCH_ROOT:-$ROOT_DIR/executorch}"

ADB=(adb)
if [[ -n "$SERIAL" ]]; then
  ADB=(adb -s "$SERIAL")
fi

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
}

run_common() {
  local remote_root="$1"
  local model="$2"
  local mmproj="$3"
  local vision="$4"
  local ubatch="$5"
  shift 5
  python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
    --processor hybrid \
    --vision "$vision" \
    --llama-build-dir "$BUILD_DIR" \
    --vision-build-dir "$BUILD_DIR" \
    --model "$model" \
    --mmproj "$mmproj" \
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
  local vision="$5"

  echo "== $label =="
  require_local_file "$model"
  require_local_file "$mmproj"
  require_local_file "$vision"
  require_remote_model "$remote_root" "$model"
  require_remote_model "$remote_root" "$mmproj"
  require_remote_model "$remote_root" "$vision"

  run_common "$remote_root" "$model" "$mmproj" "$vision" "$OFFLINE_UBATCH" \
    --image "$IMAGE_A" \
    --prompt "Describe this image briefly."

  run_common "$remote_root" "$model" "$mmproj" "$vision" "$OFFLINE_UBATCH" \
    --multi-image "$IMAGE_A" "$IMAGE_B" \
    --prompt "Compare these two images briefly."

  run_common "$remote_root" "$model" "$mmproj" "$vision" "$OFFLINE_UBATCH" \
    --video "$VIDEO" \
    --num-segments 4 \
    --max-num 1 \
    --prompt "What is happening in this video?"

  for mode in on-demand sliding-window; do
    run_common "$remote_root" "$model" "$mmproj" "$vision" "$STREAM_UBATCH" \
      --streaming-video "$VIDEO" \
      --stream-mode "$mode" \
      --online-buffer \
      --sampling-fps 1.0 \
      --max-video-time "$STREAM_SECONDS" \
      --window-sec 4.0 \
      --window-max-frames 8 \
      --max-num 1 \
      --time "$STREAM_TIMES" \
      --prompt "$STREAM_PROMPTS"
  done

  run_common "$remote_root" "$model" "$mmproj" "$vision" "$STREAM_UBATCH" \
    --streaming-video "$VIDEO" \
    --stream-mode vision-prefill \
    --online-buffer \
    --latest-frame-only \
    --partial-vision-kv \
    --sampling-fps 1.0 \
    --max-video-time "$STREAM_SECONDS" \
    --window-sec 4.0 \
    --window-max-frames 8 \
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
run_suite "InternVL3-1B" "$INTERNVL_REMOTE_ROOT" "$INTERNVL_MODEL" "$INTERNVL_MMPROJ" "$INTERNVL_VISION"
run_suite "Qwen2.5-VL-3B" "$QWEN_REMOTE_ROOT" "$QWEN_MODEL" "$QWEN_MMPROJ" "$QWEN_VISION"

while IFS= read -r -d '' run_dir; do
  verify_run_dir "$run_dir"
done < <(find "$RESULTS_ROOT" -mindepth 1 -maxdepth 1 -type d -print0)

find "$RESULTS_ROOT" -maxdepth 2 -type d | sort
