#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_internvl3_backend_matrix.sh [all|xnnpack|vulkan|qnn]

Runs InternVL3-1B XNNPACK/Vulkan artifacts on Android with the unified runner.
The default mode is all, covering:
  512 1k 2k 4k 8k 16k for XNNPACK
  512 1k 2k 4k 8k 16k for Vulkan

Mode qnn runs QNN manifests under results/model/qnn/ (needs QNN_SDK_ROOT, -b, -m via env).

Environment overrides:
  ROOT_DIR                Project root. Default: auto-detected /workspace/streamingvlm
  DEVICE                  Android device id. Default: R3KYC01FW1P
  RUNNER_BINARY           Default: $ROOT_DIR/executorch/build-android-unified/foundation/xnnpack_qnn_runner
  BUILD_PATH              QNN only: ExecuTorch Android build tree (-b). Default: $ROOT_DIR/executorch/build-android-unified
  SOC_MODEL               QNN only: SoC for launcher (-m). Default: SM8750
  QNN_ARTIFACT_BASE       QNN only: Directory containing artifact folders. Default: $ROOT_DIR/my_research/foundation/results/model/qnn
  QNN_ARTIFACT_DIRS       QNN only: Space-separated artifact folder names under QNN_ARTIFACT_BASE.
                          Default: internvl3_1b_hybrid_16p_* 512/1k/…/16k with 16a4w
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
  all|xnnpack|vulkan|qnn)
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
BUILD_PATH="${BUILD_PATH:-${ROOT_DIR}/executorch/build-android-unified}"
SOC_MODEL="${SOC_MODEL:-SM8750}"
QNN_ARTIFACT_BASE="${QNN_ARTIFACT_BASE:-${ROOT_DIR}/my_research/foundation/results/model/qnn}"
DEFAULT_QNN_ARTIFACT_DIRS="internvl3_1b_hybrid_16p_512_16a4w internvl3_1b_hybrid_16p_1k_16a4w internvl3_1b_hybrid_16p_2k_16a4w internvl3_1b_hybrid_16p_4k_16a4w internvl3_1b_hybrid_16p_8k_16a4w internvl3_1b_hybrid_16p_16k_16a4w"
read -r -a QNN_ARTIFACT_DIR_LIST <<< "${QNN_ARTIFACT_DIRS:-${DEFAULT_QNN_ARTIFACT_DIRS}}"
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

run_qnn_manifest() {
  local artifact_dir="$1"
  local manifest="${QNN_ARTIFACT_BASE}/${artifact_dir}/manifest.json"

  [[ -f "${manifest}" ]] || die "missing manifest: ${manifest}"

  log "Running QNN ${artifact_dir} with force_generate_token=${FORCE_GENERATE_TOKEN}"
  python -m my_research.foundation.cli run \
    --manifest "${manifest}" \
    --runner_binary "${RUNNER_BINARY}" \
    -b "${BUILD_PATH}" \
    -s "${DEVICE}" \
    -m "${SOC_MODEL}" \
    --image "${IMAGE}" \
    --questions "${QUESTION}" \
    --force_generate_token "${FORCE_GENERATE_TOKEN}" \
    --temperature "${TEMPERATURE}" \
    --force_push \
    --save_log
}

run_qnn_matrix() {
  for artifact_dir in "${QNN_ARTIFACT_DIR_LIST[@]}"; do
    run_qnn_manifest "${artifact_dir}"
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
  qnn)
    run_qnn_matrix
    ;;
esac

log "Completed ${MODE} inference matrix."
