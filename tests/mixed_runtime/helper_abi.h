#pragma once
#include <stddef.h>
#include <stdint.h>

/* PB1 transport probe only, not the production expert ABI. No runtime handles,
 * allocator ownership or C++ objects cross this boundary. All calls synchronous.
 * Caller owns error storage and host arrays for the duration of each call. */
#ifdef __cplusplus
extern "C" {
#endif
#define MAYA_PROBE_ABI 1u
typedef struct maya_probe_api {
    uint32_t version;
    uint32_t struct_size;
    int (*create)(int device, size_t capacity, void **context, char *error, size_t error_size);
    int (*twice)(void *context, const float *input, float *output, size_t count,
                 char *error, size_t error_size);
    int (*destroy)(void *context, char *error, size_t error_size);
} maya_probe_api;
typedef const maya_probe_api *(*maya_probe_get_api_fn)(uint32_t version);
#ifdef __cplusplus
}
#endif
