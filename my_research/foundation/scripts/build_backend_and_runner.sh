#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  build_backend_and_runner.sh <xnnpack-vulkan|qnn|all>

Builds the upstream ExecuTorch backend tree first, then builds the
project-local foundation runner against that tree.

Environment overrides:
  ROOT_DIR            Project root. Default: auto-detected /workspace/streamingvlm
  EXECUTORCH_ROOT     ExecuTorch checkout. Default: $ROOT_DIR/executorch
  ANDROID_NDK_ROOT    Android NDK path. Default: /opt/android-ndk-r26c
  CMAKE_BUILD_TYPE    CMake build type. Default: Release
  JOBS                Parallel build jobs. Default: nproc
  SKIP_ET_BUILD       If 1, skip ExecuTorch backend build and build runner only.
  SKIP_RUNNER_BUILD   If 1, skip foundation runner build.

Output build trees:
  xnnpack-vulkan      $EXECUTORCH_ROOT/build-android-xnnpack-vulkan
  qnn                 $EXECUTORCH_ROOT/build-android
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
EXECUTORCH_ROOT="${EXECUTORCH_ROOT:-${ROOT_DIR}/executorch}"
ANDROID_NDK_ROOT="${ANDROID_NDK_ROOT:-/opt/android-ndk-r26c}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"
SKIP_ET_BUILD="${SKIP_ET_BUILD:-0}"
SKIP_RUNNER_BUILD="${SKIP_RUNNER_BUILD:-0}"

MODE="${1:-}"
case "${MODE}" in
  xnnpack-vulkan|qnn|all)
    ;;
  -h|--help|"")
    usage
    exit 0
    ;;
  *)
    usage >&2
    die "unknown mode: ${MODE}"
    ;;
esac

[[ -d "${EXECUTORCH_ROOT}" ]] || die "ExecuTorch checkout not found: ${EXECUTORCH_ROOT}"
[[ -d "${ANDROID_NDK_ROOT}" ]] || die "ANDROID_NDK_ROOT not found: ${ANDROID_NDK_ROOT}"

build_et_xnnpack_vulkan() {
  local build_dir="${EXECUTORCH_ROOT}/build-android-xnnpack-vulkan"
  log "Configuring ExecuTorch XNNPACK+Vulkan: ${build_dir}"
  cmake -S "${EXECUTORCH_ROOT}" -B "${build_dir}" \
    -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-30 \
    -DCMAKE_INSTALL_PREFIX="${build_dir}" \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
    -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
    -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
    -DEXECUTORCH_ENABLE_LOGGING=1 \
    -DPYTHON_EXECUTABLE=python \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_VULKAN=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
    -DSUPPORT_REGEX_LOOKAHEAD=ON

  log "Building ExecuTorch XNNPACK+Vulkan"
  cmake --build "${build_dir}" -j"${JOBS}" --target install --config "${CMAKE_BUILD_TYPE}"
}

build_et_qnn() {
  log "Building ExecuTorch QNN"
  (cd "${EXECUTORCH_ROOT}" && ./backends/qualcomm/scripts/build.sh --skip_x86_64)
}

build_runner() {
  local backend_build_dir="$1"
  local runner_build_dir="${backend_build_dir}/foundation"
  local cmake_prefix="${backend_build_dir};${backend_build_dir}/third-party/gflags"

  log "Configuring foundation runner: ${runner_build_dir}"
  cmake -S "${FOUNDATION_DIR}" -B "${runner_build_dir}" \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
    -DEXECUTORCH_ROOT="${EXECUTORCH_ROOT}" \
    -DEXECUTORCH_BUILD_DIR="${backend_build_dir}" \
    -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-30 \
    -DCMAKE_PREFIX_PATH="${cmake_prefix}" \
    -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
    -Dgflags_DIR="${backend_build_dir}/third-party/gflags"

  log "Building foundation runner"
  cmake --build "${runner_build_dir}" -j"${JOBS}" --target xnnpack_qnn_runner
  log "Runner ready: ${runner_build_dir}/xnnpack_qnn_runner"
}

run_xnnpack_vulkan() {
  local build_dir="${EXECUTORCH_ROOT}/build-android-xnnpack-vulkan"
  if [[ "${SKIP_ET_BUILD}" != "1" ]]; then
    build_et_xnnpack_vulkan
  fi
  if [[ "${SKIP_RUNNER_BUILD}" != "1" ]]; then
    build_runner "${build_dir}"
  fi
}

run_qnn() {
  local build_dir="${EXECUTORCH_ROOT}/build-android"
  if [[ "${SKIP_ET_BUILD}" != "1" ]]; then
    build_et_qnn
  fi
  if [[ "${SKIP_RUNNER_BUILD}" != "1" ]]; then
    build_runner "${build_dir}"
  fi
}

case "${MODE}" in
  xnnpack-vulkan)
    run_xnnpack_vulkan
    ;;
  qnn)
    run_qnn
    ;;
  all)
    run_xnnpack_vulkan
    run_qnn
    ;;
esac
