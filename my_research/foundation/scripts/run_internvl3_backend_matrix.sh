#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_internvl3_backend_matrix.sh [all|xnnpack|vulkan]

Runs InternVL3-1B XNNPACK/Vulkan artifacts on Android with the unified runner.
The default mode is all, covering:
  512 1k 2k 4k 8k 16k for XNNPACK
  512 1k 2k 4k 8k 16k for Vulkan

Environment overrides:
  ROOT_DIR                Project root. Default: auto-detected /workspace/streamingvlm
  DEVICE                  Android device id. Default: R3KYC01FW1P
  RUNNER_BINARY           Default: $ROOT_DIR/executorch/build-android-unified/foundation/xnnpack_qnn_runner
  IMAGE                   Default: COCO cat sample URL
  QUESTION                Default: Describe this image briefly using around 10 words.
  FORCE_GENERATE_TOKEN    Default: 128
  TEMPERATURE             Default: 0.0
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

MODE="${1:-all}"
case "${MODE}" in
  all|xnnpack|vulkan)
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    die "unknown mode: ${MODE}"
    ;;
esac

DEVICE="${DEVICE:-R3KYC01FW1P}"
RUNNER_BINARY="${RUNNER_BINARY:-${ROOT_DIR}/executorch/build-android-unified/foundation/xnnpack_qnn_runner}"
IMAGE="${IMAGE:-http://images.cocodataset.org/val2017/000000039769.jpg}"
QUESTION="${QUESTION:-Describe this image briefly using around 10 words.}"
FORCE_GENERATE_TOKEN="${FORCE_GENERATE_TOKEN:-128}"
TEMPERATURE="${TEMPERATURE:-0.0}"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/executorch${PYTHONPATH:+:${PYTHONPATH}}"

[[ -x "${RUNNER_BINARY}" ]] || die "runner not executable: ${RUNNER_BINARY}"

run_manifest() {
  local backend="$1"
  local tag="$2"
  local manifest="${ROOT_DIR}/my_research/foundation/results/model/${backend}/internvl3_${backend}_1b_${tag}_fp16/manifest.json"

  [[ -f "${manifest}" ]] || die "missing manifest: ${manifest}"

  log "Running ${backend} ${tag} with force_generate_token=${FORCE_GENERATE_TOKEN}"
  python -m my_research.foundation.cli run \
    --manifest "${manifest}" \
    --runner_binary "${RUNNER_BINARY}" \
    --device "${DEVICE}" \
    --image "${IMAGE}" \
    --questions "${QUESTION}" \
    --force_generate_token "${FORCE_GENERATE_TOKEN}" \
    --temperature "${TEMPERATURE}" \
    --force_push \
    --save_log
}

run_backend() {
  local backend="$1"
  for tag in 512 1k 2k 4k 8k 16k; do
    run_manifest "${backend}" "${tag}"
  done
}

case "${MODE}" in
  all)
    run_backend xnnpack
    run_backend vulkan
    ;;
  xnnpack|vulkan)
    run_backend "${MODE}"
    ;;
esac

log "Completed ${MODE} inference matrix."
