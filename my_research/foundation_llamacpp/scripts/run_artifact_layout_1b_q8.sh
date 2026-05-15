#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

SERIAL="${ADB_SERIAL:-}"
REMOTE_ROOT="${REMOTE_ROOT:-/data/local/tmp/streamingvlm_smoke_modes}"
RESULTS_ROOT="${RESULTS_ROOT:-my_research/foundation_llamacpp/results/log/artifact_layout_1b_q8}"
BUILD_DIR="${BUILD_DIR:-my_research/foundation_llamacpp/build-hybrid-android-opencl}"
MODEL="${MODEL:-llama.cpp/models/InternVL3-1B-Instruct-GGUF/InternVL3-1B-Instruct-Q8_0.gguf}"
MMPROJ="${MMPROJ:-llama.cpp/models/InternVL3-1B-Instruct-GGUF/mmproj-InternVL3-1B-Instruct-Q8_0.gguf}"
VISION="${VISION:-my_research/foundation_llamacpp/results/vision_models/internvl3_1b_vision_tower_preproj_qnn_realweights_sm8750/vision_tower_preproj_qnn.pte}"
IMAGE_A="${IMAGE_A:-my_research/foundation_llamacpp/sample_images/golden_gate_bridge_448.jpg}"
IMAGE_B="${IMAGE_B:-my_research/foundation_llamacpp/sample_images/sample_coco_cats_448.jpg}"
VIDEO="${VIDEO:-my_research/foundation_llamacpp/sample_images/red-panda_448.mp4}"

export QNN_SDK_ROOT="${QNN_SDK_ROOT:-$ROOT_DIR/executorch/backends/qualcomm/sdk/qnn/qairt/2.37.0.250724}"
export EXECUTORCH_ROOT="${EXECUTORCH_ROOT:-$ROOT_DIR/executorch}"

ADB=(adb)
if [[ -n "$SERIAL" ]]; then
  ADB=(adb -s "$SERIAL")
fi

remote_has_file() {
  local name
  name="$(basename "$1")"
  "${ADB[@]}" shell "test -f '$REMOTE_ROOT/$name'"
}

require_remote_model() {
  if ! remote_has_file "$1"; then
    echo "Missing remote model artifact: $REMOTE_ROOT/$(basename "$1")" >&2
    echo "This smoke script intentionally avoids model pushes. Populate the remote root once, then re-run." >&2
    exit 2
  fi
}

require_local_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing local input: $1" >&2
    exit 2
  fi
}

verify_layout() {
  local run_dir="$1"
  [[ -d "$run_dir/csv" ]]
  [[ -d "$run_dir/png" ]]
  [[ -d "$run_dir/txt_json" ]]
  if find "$run_dir" -maxdepth 1 -type f | grep -q .; then
    echo "Unexpected root-level files in $run_dir" >&2
    find "$run_dir" -maxdepth 1 -type f >&2
    exit 3
  fi
}

run_common() {
  python3 my_research/foundation_llamacpp/run_android_hybrid_bridge.py \
    --processor hybrid \
    --vision "$VISION" \
    --llama-build-dir "$BUILD_DIR" \
    --model "$MODEL" \
    --mmproj "$MMPROJ" \
    --n-predict 8 \
    --threads 4 \
    --gpu-layers 99 \
    --device GPUOpenCL \
    --ctx-size 4096 \
    --batch-size 1024 \
    --ubatch-size 512 \
    --temperature 0.0 \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --fit off \
    --soc-model SM8750 \
    --baseline-window 0.5 \
    --remote-root "$REMOTE_ROOT" \
    --results-root "$RESULTS_ROOT" \
    "$@"
}

require_local_file "$MODEL"
require_local_file "$MMPROJ"
require_local_file "$VISION"
require_local_file "$IMAGE_A"
require_local_file "$IMAGE_B"
require_local_file "$VIDEO"
require_remote_model "$MODEL"
require_remote_model "$MMPROJ"
require_remote_model "$VISION"

rm -rf "$RESULTS_ROOT"

run_common \
  --image "$IMAGE_A" \
  --prompt "Describe this image briefly."

run_common \
  --multi-image "$IMAGE_A" "$IMAGE_B" \
  --prompt "Compare these two images briefly."

run_common \
  --video "$VIDEO" \
  --num-segments 4 \
  --max-num 1 \
  --prompt "What is happening in this video?"

for mode in on-demand sliding-window; do
  run_common \
    --streaming-video "$VIDEO" \
    --stream-mode "$mode" \
    --sampling-fps 1.0 \
    --max-video-time 2.0 \
    --window-sec 2.0 \
    --window-max-frames 4 \
    --max-num 1 \
    --time '[1.0, 2.0]' \
    --prompt '["What is happening in this video?", "What changed?"]'
done

run_common \
  --streaming-video "$VIDEO" \
  --stream-mode vision-prefill \
  --sampling-fps 1.0 \
  --max-video-time 2.0 \
  --window-sec 2.0 \
  --window-max-frames 4 \
  --max-num 1 \
  --time '[1.0, 2.0]' \
  --prompt '["What is happening in this video?", "What changed?"]'

run_common \
  --streaming-video "$VIDEO" \
  --stream-mode vision-prefill \
  --sampling-fps 1.0 \
  --max-video-time 2.0 \
  --window-sec 2.0 \
  --window-max-frames 4 \
  --max-num 1 \
  --time '[1.0, 2.0]' \
  --prompt '["What is happening in this video?", "What changed?"]' \
  --dynamic-kv-cache \
  --kv-init-size 512 \
  --kv-grow-step 512

run_common \
  --streaming-video "$VIDEO" \
  --stream-mode vision-prefill \
  --sampling-fps 1.0 \
  --max-video-time 2.0 \
  --window-sec 2.0 \
  --window-max-frames 4 \
  --max-num 1 \
  --time '[1.0, 2.0]' \
  --prompt '["What is happening in this video?", "What changed?"]' \
  --dynamic-kv-cache \
  --kv-init-size 512 \
  --kv-grow-step 512 \
  --online-buffer

STEM="InternVL3-1B-Instruct-Q8_0_hybrid_ctx_4096"
verify_layout "$RESULTS_ROOT/${STEM}_image_kv16"
verify_layout "$RESULTS_ROOT/${STEM}_multi_image_kv16"
verify_layout "$RESULTS_ROOT/${STEM}_video_kv16"
verify_layout "$RESULTS_ROOT/${STEM}_streaming_on_demand_kv16"
verify_layout "$RESULTS_ROOT/${STEM}_streaming_sliding_window_kv16"
verify_layout "$RESULTS_ROOT/${STEM}_streaming_vision_prefill_kv16"
verify_layout "$RESULTS_ROOT/${STEM}_streaming_vision_prefill_kv16_dynamic"
verify_layout "$RESULTS_ROOT/${STEM}_streaming_vision_prefill_kv16_dynamic_online"

find "$RESULTS_ROOT" -maxdepth 2 -type d | sort
