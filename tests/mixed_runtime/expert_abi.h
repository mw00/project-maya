#pragma once
#include <stddef.h>
#include <stdint.h>
/* Diagnostic ABI, not a production cache. Synchronous; all pointers host-owned.
 * blobs: k consecutive [gate | up | down] experts with common types and layout.
 * output: n_embd floats. No model routing decisions are made by this module. */
#ifdef __cplusplus
extern "C" {
#endif
typedef int (*maya_expert_probe_fn)(uint32_t abi, int device, int gu_type, int down_type,
    int n_embd, int n_ff, int k, const void *blobs, size_t bytes, size_t stride,
    size_t down_offset, const float *routing, const float *input, float clamp,
    float *output, char *error, size_t error_size);
#ifdef __cplusplus
}
#endif
