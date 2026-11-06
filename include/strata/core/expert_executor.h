#pragma once
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Synchronous experimental ABI. Methods serialize per context. Host arrays remain
 * caller-owned through return. Destroy must be exclusive with all other callers.
 * No runtime handles or device pointers cross the boundary. */
typedef struct maya_context_config {
    uint32_t size, version;
    int32_t device, gu_type, down_type, width, intermediate;
    uint32_t slots;
    uint64_t byte_budget;
} maya_context_config;
typedef struct maya_context_stats {
    uint64_t allocated_bytes, upload_bytes, uploads, computes;
} maya_context_stats;
typedef struct maya_context_api {
    uint32_t size, version;
    int (*create)(const maya_context_config *, void **, char *, size_t);
    int (*upload)(void *, uint32_t slot, uint64_t expected_generation,
                  const void *, size_t, uint64_t *new_generation, char *, size_t);
    int (*compute)(void *, uint32_t count, const uint32_t *slots,
                   const uint64_t *generations, const float *weights, const float *input,
                   float clamp, float *output, char *, size_t);
    int (*stats)(void *, maya_context_stats *, char *, size_t);
    /* Consumes *context and sets it NULL even if GPU cleanup reports an error. */
    int (*destroy)(void **context, char *, size_t);
} maya_context_api;
typedef const maya_context_api *(*maya_context_get_api_fn)(uint32_t);
/* Optional v1 readback extension. Synchronous, non-destructive byte-range read.
 * The destination remains caller-owned through return; discard it on failure.
 * Uses the same serialization and generation checks as compute/upload. */
typedef int (*maya_context_read_fn)(void *, uint32_t slot, uint64_t generation,
                                   size_t offset, void *output, size_t bytes, char *, size_t);
/* Optional unweighted expert-row extension: 1..16 independent input/output rows,
 * each width floats, with the same lifetime and generation rules as compute. */
typedef int (*maya_context_rows_fn)(void *, uint32_t count, const uint32_t *slots,
                                  const uint64_t *generations, const float *input,
                                  float clamp, float *output, char *, size_t);
#ifdef __cplusplus
}
#endif
