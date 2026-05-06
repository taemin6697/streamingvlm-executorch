cmake_minimum_required(VERSION 3.24)
project(streamingvlm_hybrid_bridge CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

option(HYBRID_BRIDGE_BUILD_LLAMA_DECODER "Build llama.cpp external embedding decoder bridge" ON)
option(HYBRID_BRIDGE_BUILD_EXECUTORCH_VISION "Build ExecuTorch QNN vision embedding dumper" OFF)

get_filename_component(STREAMINGVLM_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

if(HYBRID_BRIDGE_BUILD_LLAMA_DECODER)
  set(LLAMA_CPP_ROOT "${STREAMINGVLM_ROOT}/llama.cpp" CACHE PATH "llama.cpp checkout")
  set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
  add_subdirectory("${LLAMA_CPP_ROOT}" "${CMAKE_BINARY_DIR}/llama.cpp" EXCLUDE_FROM_ALL)

  find_package(Threads REQUIRED)
  add_executable(
    hybrid_decode
    hybrid_decode.cpp
    hybrid_embedding_file.cpp
  )
  add_executable(opencl_phase_mtmd opencl_phase_mtmd.cpp)
  target_include_directories(
    hybrid_decode
    PRIVATE
      "${CMAKE_CURRENT_LIST_DIR}"
      "${LLAMA_CPP_ROOT}/common"
      "${LLAMA_CPP_ROOT}/include"
      "${LLAMA_CPP_ROOT}/tools/mtmd"
      "${LLAMA_CPP_ROOT}/vendor"
      "${LLAMA_CPP_ROOT}/ggml/include"
      "${LLAMA_CPP_ROOT}/ggml/src"
  )
  target_include_directories(
    opencl_phase_mtmd
    PRIVATE
      "${CMAKE_CURRENT_LIST_DIR}"
      "${LLAMA_CPP_ROOT}/common"
      "${LLAMA_CPP_ROOT}/include"
      "${LLAMA_CPP_ROOT}/tools/mtmd"
      "${LLAMA_CPP_ROOT}/vendor"
      "${LLAMA_CPP_ROOT}/ggml/include"
      "${LLAMA_CPP_ROOT}/ggml/src"
  )
  target_link_libraries(hybrid_decode PRIVATE llama-common mtmd Threads::Threads)
  target_link_libraries(opencl_phase_mtmd PRIVATE llama-common mtmd Threads::Threads)
  target_compile_features(hybrid_decode PRIVATE cxx_std_17)
  target_compile_features(opencl_phase_mtmd PRIVATE cxx_std_17)
endif()

if(HYBRID_BRIDGE_BUILD_EXECUTORCH_VISION)
  set(EXECUTORCH_ROOT "${STREAMINGVLM_ROOT}/executorch" CACHE PATH "ExecuTorch checkout")
  set(
    EXECUTORCH_BUILD_DIR
    "${EXECUTORCH_ROOT}/build-android-xnnpack"
    CACHE PATH
    "Installed ExecuTorch CMake build directory"
  )
  if(NOT IS_DIRECTORY "${EXECUTORCH_ROOT}")
    message(FATAL_ERROR "EXECUTORCH_ROOT does not exist: ${EXECUTORCH_ROOT}")
  endif()

  include("${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake")
  list(APPEND CMAKE_FIND_ROOT_PATH "${EXECUTORCH_BUILD_DIR}")
  if(NOT gflags_DIR)
    set(gflags_DIR "${EXECUTORCH_BUILD_DIR}/third-party/gflags")
  endif()
  find_package(gflags REQUIRED NO_CMAKE_FIND_ROOT_PATH)
  find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
  executorch_target_link_options_shared_lib(executorch)

  add_executable(
    hybrid_vision_dump
    hybrid_vision_dump.cpp
    hybrid_embedding_file.cpp
    "${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/encoder.cpp"
  )
  target_include_directories(
    hybrid_vision_dump
    PRIVATE
      "${CMAKE_CURRENT_LIST_DIR}"
      "${STREAMINGVLM_ROOT}"
      "${EXECUTORCH_ROOT}/.."
      "${EXECUTORCH_ROOT}/third-party/json/include"
  )
  target_link_libraries(
    hybrid_vision_dump
    PRIVATE
      executorch
      executorch_core
      extension_data_loader
      extension_flat_tensor
      extension_llm_runner
      extension_module
      extension_tensor
      gflags
  )
  if(TARGET qnn_executorch_backend)
    target_link_libraries(hybrid_vision_dump PRIVATE qnn_executorch_backend)
    target_compile_definitions(hybrid_vision_dump PRIVATE FOUNDATION_ENABLE_QNN=1)
  else()
    message(WARNING "qnn_executorch_backend target was not found; QNN delegated PTEs may not run.")
  endif()
  if(TARGET portable_ops_lib)
    target_link_libraries(hybrid_vision_dump PRIVATE portable_ops_lib portable_kernels)
  endif()
  if(TARGET quantized_ops_lib)
    target_link_libraries(hybrid_vision_dump PRIVATE quantized_ops_lib quantized_kernels)
  endif()
  target_compile_options(hybrid_vision_dump PRIVATE -Wno-deprecated-declarations -fPIC)
  target_compile_features(hybrid_vision_dump PRIVATE cxx_std_17)
endif()
