# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
project(foundation_runner)

if(NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()

get_filename_component(_STREAMINGVLM_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(
  EXECUTORCH_ROOT
  "${_STREAMINGVLM_ROOT}/executorch"
  CACHE PATH "Path to a clean upstream ExecuTorch checkout"
)
if(NOT IS_DIRECTORY "${EXECUTORCH_ROOT}")
  message(FATAL_ERROR "EXECUTORCH_ROOT does not exist: ${EXECUTORCH_ROOT}")
endif()

set(
  EXECUTORCH_BUILD_DIR
  "${EXECUTORCH_ROOT}/build-android-xnnpack"
  CACHE PATH "Path to the installed ExecuTorch CMake build directory"
)
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(_common_compile_options -Wno-deprecated-declarations -fPIC)
set(
  _common_include_directories
  ${CMAKE_CURRENT_SOURCE_DIR}/runner
  ${_STREAMINGVLM_ROOT}
  ${EXECUTORCH_ROOT}/..
  ${EXECUTORCH_ROOT}/third-party/json/include
)

if(NOT gflags_DIR)
  set(gflags_DIR ${EXECUTORCH_BUILD_DIR}/third-party/gflags)
endif()
find_package(gflags REQUIRED NO_CMAKE_FIND_ROOT_PATH)

list(APPEND CMAKE_FIND_ROOT_PATH ${EXECUTORCH_BUILD_DIR})
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
executorch_target_link_options_shared_lib(executorch)

if(TARGET qnn_executorch_backend AND NOT TARGET portable_ops_lib)
  executorch_load_build_variables()
  gen_selected_ops(
    LIB_NAME "full_portable_ops_lib"
    ROOT_OPS
      "aten::expand_copy.out"
      "aten::embedding.out"
  )
  generate_bindings_for_kernels(
    LIB_NAME "full_portable_ops_lib" FUNCTIONS_YAML
    ${EXECUTORCH_ROOT}/kernels/portable/functions.yaml
  )
  gen_operators_lib(
    LIB_NAME "full_portable_ops_lib" KERNEL_LIBS portable_kernels DEPS executorch
  )
  target_compile_options(
    full_portable_ops_lib INTERFACE -DET_EVENT_TRACER_ENABLED
  )
endif()

add_executable(
  xnnpack_qnn_runner
  runner/xnnpack_qnn_runner.cpp
  runner/foundation_qnn_multimodal_runner.cpp
  runner/qnn_backend.cpp
  runner/xnnpack_backend.cpp
)

target_include_directories(
  xnnpack_qnn_runner PUBLIC ${_common_include_directories}
)
target_include_directories(
  xnnpack_qnn_runner PUBLIC ${EXECUTORCH_ROOT}/extension/llm/tokenizers/include
)

target_link_libraries(
  xnnpack_qnn_runner
  PUBLIC
  executorch
  gflags
  extension_llm_runner
)

# xnnpack_backend: internvl3 runner와 동일한 방식으로 링크한다.
if(TARGET xnnpack_backend)
  if(TARGET qnn_executorch_backend)
    target_link_options(xnnpack_qnn_runner PRIVATE "LINKER:--allow-multiple-definition")
  endif()
  set(_foundation_xnnpack_backend_libs xnnpack_backend XNNPACK xnnpack-microkernels-prod)
  if(TARGET kleidiai)
    list(APPEND _foundation_xnnpack_backend_libs kleidiai)
  endif()
  executorch_target_link_options_shared_lib(xnnpack_backend)
  target_link_libraries(
    xnnpack_qnn_runner
    PUBLIC
    ${_foundation_xnnpack_backend_libs}
  )
endif()

if(TARGET vulkan_backend)
  executorch_target_link_options_shared_lib(vulkan_backend)
  target_link_libraries(xnnpack_qnn_runner PUBLIC vulkan_backend)
endif()

if(TARGET portable_ops_lib)
  target_link_libraries(xnnpack_qnn_runner PUBLIC portable_ops_lib portable_kernels)
endif()

if(TARGET quantized_ops_lib)
  executorch_target_link_options_shared_lib(quantized_ops_lib)
  target_link_libraries(xnnpack_qnn_runner PUBLIC quantized_ops_lib quantized_kernels)
endif()

if(TARGET custom_ops AND NOT TARGET qnn_executorch_backend)
  target_link_libraries(
    xnnpack_qnn_runner PUBLIC $<LINK_LIBRARY:WHOLE_ARCHIVE,custom_ops>
  )
endif()

if(TARGET qnn_executorch_backend)
  target_compile_definitions(xnnpack_qnn_runner PRIVATE FOUNDATION_ENABLE_QNN=1)
  if(NOT TARGET custom_ops)
    add_library(
      custom_ops
      ${EXECUTORCH_ROOT}/extension/llm/custom_ops/op_fallback.cpp
    )
    target_include_directories(custom_ops PUBLIC ${_common_include_directories})
    target_link_libraries(custom_ops PUBLIC executorch_core)
    executorch_target_link_options_shared_lib(custom_ops)
  endif()

  set(_foundation_qnn_runner_srcs
      ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/decoder_runner.cpp
      ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/prompt_processor.cpp
      ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/token_generator.cpp
      ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/lhd_token_generator.cpp
      ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/rpc_mem.cpp
      ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/kv_manager.cpp
      ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/attention_sink_rope_runner.cpp
      ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/encoder.cpp
      ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/tok_embedding_runner.cpp
      ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/tok_embedding_processor.cpp
      ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_embedding_merger.cpp
      ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_prompt_processor.cpp
      ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_token_generator.cpp
      ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_lhd_token_generator.cpp
      ${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_runner.cpp
      ${EXECUTORCH_ROOT}/examples/models/llama/runner/runner.cpp
  )
  target_sources(xnnpack_qnn_runner PRIVATE ${_foundation_qnn_runner_srcs})
  target_link_libraries(xnnpack_qnn_runner PUBLIC qnn_executorch_backend
    executorch_core
    extension_data_loader
    extension_flat_tensor
    extension_llm_runner
    extension_module
    extension_tensor
    tokenizers::tokenizers)
  if(TARGET custom_ops)
    target_link_libraries(
      xnnpack_qnn_runner PUBLIC $<LINK_LIBRARY:WHOLE_ARCHIVE,custom_ops>
    )
  endif()
  if(NOT TARGET portable_ops_lib AND TARGET full_portable_ops_lib)
    add_dependencies(xnnpack_qnn_runner full_portable_ops_lib)
    target_link_options(
      xnnpack_qnn_runner PRIVATE
      "LINKER:--whole-archive"
      "$<TARGET_FILE:full_portable_ops_lib>"
      "LINKER:--no-whole-archive"
    )
    target_link_libraries(
      xnnpack_qnn_runner PUBLIC
      portable_kernels
    )
  endif()
  if(TARGET quantized_ops_lib)
    target_link_libraries(xnnpack_qnn_runner PUBLIC quantized_ops_lib quantized_kernels)
  endif()
endif()
target_compile_options(xnnpack_qnn_runner PUBLIC ${_common_compile_options})

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_gc_sections(xnnpack_qnn_runner)
  if(NOT APPLE)
    target_link_options(xnnpack_qnn_runner PRIVATE "LINKER:-s")
  endif()
endif()
