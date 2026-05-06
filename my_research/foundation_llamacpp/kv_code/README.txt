Source: https://github.com/ggml-org/llama.cpp/pull/21313 (Draft)
Fetched PR head ref: refs/pull/21313/head
Commit: 48d85b9af5833abe378052b9c5e139c9c818e773

Cherry-picked onto workspace nested llama.cpp master as cd6a04a01 (rebuild hybrid Android OpenCL after pulling).

Files (paths relative to llama.cpp repo root):
  ggml/src/ggml-opencl/CMakeLists.txt
  ggml/src/ggml-opencl/ggml-opencl.cpp
  ggml/src/ggml-opencl/kernels/flash_attn_f16.cl
  ggml/src/ggml-opencl/kernels/flash_attn_f32.cl
  ggml/src/ggml-opencl/kernels/flash_attn_f32_f16.cl
  ggml/src/ggml-opencl/kernels/flash_attn_pre_f16.cl

Also bundled: pr21313.patch (full PR diff).

Refresh:
  cd /path/to/llama.cpp && git fetch origin pull/21313/head:refs/tmp/pr21313-opencl-fa
  # then git show refs/tmp/pr21313-opencl-fa:<path> > kv_code/<path>
