#ifndef GGML_OPENCL_H
#define GGML_OPENCL_H

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

//
// backend API
//
struct ggml_tensor;

GGML_BACKEND_API ggml_backend_t ggml_backend_opencl_init(void);
GGML_BACKEND_API bool ggml_backend_is_opencl(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_opencl_buffer_type(void);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_opencl_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_opencl_reg(void);
GGML_BACKEND_API bool ggml_backend_opencl_tensor_copy_bytes(const struct ggml_tensor * src, struct ggml_tensor * dst, size_t src_offset, size_t dst_offset, size_t size);

#ifdef  __cplusplus
}
#endif

#endif // GGML_OPENCL_H
