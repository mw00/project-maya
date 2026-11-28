// src/ds4/ds4_ernie_rx.h - the C interface between maya-ds4 (ERNIE-5.5) and a helper GPU's expert executor
// (libmaya_ds4_ernie_rx.so, built from ds4_ernie_rx.cu with the helper GPU's own runtime - HIP on an AMD card - and
// loaded with dlopen): routed experts held in that GPU's memory and computed there for a decoded token while the CPU
// lane computes the others.  No runtime handle or device pointer crosses the boundary; host arrays are the caller's.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// One pool of experts of one layout (the same three ggml types and blob layout): `slots` blobs on the helper GPU, an
/// expert's blob gate | up | down at off[0..2].
typedef struct ernx_config {
    uint32_t size, version;   // sizeof, 1
    int32_t device;           // the helper runtime's own ordinal
    int32_t dim, ff;          // the experts' input / output width and their hidden width (multiples of 256)
    int32_t type[3];          // gate, up, down: ggml type ids (gate and up of one type)
    uint64_t off[3];          // each role's byte offset in a blob
    uint64_t blob_bytes;
    uint32_t slots;
    float beta, lin;          // SiTU-GLU's beta and linear beta
} ernx_config;

typedef struct ernx_stats {
    uint64_t pool_bytes, uploads, submits, experts;
} ernx_stats;

typedef struct ernx_api {
    uint32_t size, version;
    int (*create)(const ernx_config*, void** ctx, char* err, size_t n);
    int (*upload)(void* ctx, uint32_t slot, const void* blob, size_t bytes, char* err, size_t n);
    /// Starts out[dim] = sum_i w[i] * down(situ(gate(x), up(x))) over the 1..16 experts in slots[] (x [dim] F32, read
    /// before the call returns); returns at once.  One submit in flight a context: wait() before the next.
    int (*submit)(void* ctx, uint32_t count, const uint32_t* slots, const float* w, const float* x, char* err, size_t n);
    /// Waits for the last submit and writes its out[dim].
    int (*wait)(void* ctx, float* out, char* err, size_t n);
    int (*stats)(void* ctx, ernx_stats* out, char* err, size_t n);
    int (*destroy)(void** ctx, char* err, size_t n);
} ernx_api;

typedef const ernx_api* (*ernx_get_api_fn)(uint32_t version);

#ifdef __cplusplus
}
#endif
