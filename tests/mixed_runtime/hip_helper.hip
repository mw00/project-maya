#include "helper_abi.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <new>
#include <limits>

namespace {
struct Context { int device; size_t capacity; float *data = nullptr; hipStream_t stream = nullptr; };
int fail(const char *text, char *error, size_t n) {
    if (error && n) std::snprintf(error, n, "%s", text);
    return 1;
}
int check(hipError_t e, char *error, size_t n) {
    return e == hipSuccess ? 0 : fail(hipGetErrorString(e), error, n);
}
__global__ void twice_kernel(float *p, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) p[i] *= 2.0f;
}
int destroy(void *opaque, char *error, size_t n) {
    auto *c = static_cast<Context *>(opaque);
    if (!c) return 0;
    // A failure selecting the device must not free allocations in another context.
    if (check(hipSetDevice(c->device), error, n)) return 1;
    int rc = 0;
    if (c->stream) {
        rc |= check(hipStreamSynchronize(c->stream), error, n);
        rc |= check(hipStreamDestroy(c->stream), error, n);
    }
    if (c->data) rc |= check(hipFree(c->data), error, n);
    delete c;
    return rc;
}
int create(int device, size_t capacity, void **out, char *error, size_t n) {
    if (!out) return fail("null context output", error, n);
    *out = nullptr;
    if (!capacity || capacity > 16 * 1024 * 1024)
        return fail("capacity outside probe limit", error, n);
    if (check(hipSetDevice(device), error, n)) return 1;
    hipDeviceProp_t prop{};
    if (check(hipGetDeviceProperties(&prop, device), error, n)) return 1;
    std::printf("HIP device %d: %s (%s)\n", device, prop.name, prop.gcnArchName);
    auto *c = new (std::nothrow) Context{device, capacity};
    if (!c) return fail("context allocation failed", error, n);
    if (check(hipStreamCreateWithFlags(&c->stream, hipStreamNonBlocking), error, n) ||
        check(hipMalloc(reinterpret_cast<void **>(&c->data), capacity * sizeof(float)), error, n)) {
        char ignored[256]; destroy(c, ignored, sizeof(ignored)); return 1;
    }
    *out = c;
    return 0;
}
int twice(void *opaque, const float *in, float *out, size_t count, char *error, size_t n) {
    auto *c = static_cast<Context *>(opaque);
    if (!c || !in || !out || !count || count > c->capacity)
        return fail("invalid job or capacity exceeded", error, n);
    if (check(hipSetDevice(c->device), error, n)) return 1;
    if (check(hipMemcpyAsync(c->data, in, count * sizeof(float), hipMemcpyHostToDevice, c->stream), error, n))
        return 1;
    twice_kernel<<<(count + 255) / 256, 256, 0, c->stream>>>(c->data, count);
    int rc = check(hipGetLastError(), error, n);
    if (!rc) rc = check(hipMemcpyAsync(out, c->data, count * sizeof(float), hipMemcpyDeviceToHost, c->stream), error, n);
    // Always drain submitted reads/writes before returning host buffer ownership.
    int drained = check(hipStreamSynchronize(c->stream), error, n);
    return rc || drained;
}
const maya_probe_api api{MAYA_PROBE_ABI, sizeof(maya_probe_api), create, twice, destroy};
}
extern "C" __attribute__((visibility("default")))
const maya_probe_api *maya_probe_get_api(uint32_t version) {
    return version == MAYA_PROBE_ABI ? &api : nullptr;
}
