// src/ik2/ik2_off.cu - maya-ik2's offload token path (ik2_off.cuh): the GPU fetch of a token's missing experts.
#include "ik2_off.cuh"

namespace ik2::k {
namespace {

__global__ void off_fetch_plan_kernel(const uint8_t** blob, const int* __restrict__ top, int K, int key0,
                                      const uint8_t* host, size_t hstride, OffStage st, OffFetch* fl, OffCount* cnt,
                                      uint32_t* heat) {
    const int lane = threadIdx.x;
    uint32_t seq = 0;
    if (lane == 0) seq = ++cnt->seq;
    seq = __shfl_sync(0xffffffffu, seq, 0);
    if (heat && lane < K) atomicAdd(heat + key0 + top[lane], 1u);
    int n_fetch = 0, n_tier = 0, n_stage = 0;
    for (int i = 0; i < K; ++i) {
        if (blob[i]) {   // in the tier
            ++n_tier;
            continue;
        }
        const int key = key0 + top[i];
        int found = -1;
        for (int b = 0; b < st.n; b += 32) {
            const int s = b + lane;
            const unsigned m = __ballot_sync(0xffffffffu, s < st.n && st.tag[s] == key);
            if (m) {
                found = b + __ffs(m) - 1;
                break;
            }
        }
        if (lane == 0) {
            if (found >= 0) {   // still in staging from an earlier visit
                st.used[found] = seq;
                blob[i] = st.data + (size_t) found * st.sb;
                ++n_stage;
            } else {            // the clock's next slot this visit does not use (st.n >= K: there is one)
                int h = *st.hand, s = h;
                for (int t = 0; t < st.n; ++t) {
                    s = h;
                    h = h + 1 == st.n ? 0 : h + 1;
                    if (st.used[s] != seq) break;
                }
                *st.hand = h;
                st.tag[s] = key;
                st.used[s] = seq;
                uint8_t* dst = st.data + (size_t) s * st.sb;
                blob[i] = dst;
                fl->src[n_fetch] = host + (size_t) top[i] * hstride;
                fl->dst[n_fetch] = dst;
                ++n_fetch;
            }
        }
        __syncwarp();
    }
    if (lane == 0) {
        fl->n = n_fetch;
        cnt->picks += (unsigned long long) K;
        cnt->tier += (unsigned long long) n_tier;
        cnt->stage += (unsigned long long) n_stage;
        cnt->fetched += (unsigned long long) n_fetch;
    }
}

// 4 x 16 bytes in flight a thread: the bus's latency (a microsecond or two) hidden by many reads at once
__global__ void off_fetch_copy_kernel(const OffFetch* __restrict__ fl, size_t n16) {
    const int n = fl->n;
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    const size_t t0 = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = 0; j < n; ++j) {
        const uint4* src = (const uint4*) fl->src[j];
        uint4* dst = (uint4*) fl->dst[j];
        size_t i = t0;
        for (; i + 3 * stride < n16; i += 4 * stride) {
            const uint4 a = src[i], b = src[i + stride], c = src[i + 2 * stride], d = src[i + 3 * stride];
            dst[i] = a, dst[i + stride] = b, dst[i + 2 * stride] = c, dst[i + 3 * stride] = d;
        }
        for (; i < n16; i += stride) dst[i] = src[i];
    }
}

}  // namespace

void off_fetch_plan(const uint8_t** blob, const int* top, int K, int key0, const uint8_t* host, size_t hstride,
                    OffStage st, OffFetch* fl, OffCount* cnt, uint32_t* heat, cudaStream_t s) {
    off_fetch_plan_kernel<<<1, 32, 0, s>>>(blob, top, K, key0, host, hstride, st, fl, cnt, heat);
}

void off_fetch_copy(const OffFetch* fl, size_t bytes, int blocks, cudaStream_t s) {
    off_fetch_copy_kernel<<<blocks, 256, 0, s>>>(fl, bytes / 16);
}

}  // namespace ik2::k
