// src/ik2/ik2_off.cu - maya-ik2's offload token path (ik2_off.cuh): the GPU fetch of a token's missing experts.
#include "ik2_off.cuh"

#include "ik2_exact.h"

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
        cnt->bytes += (unsigned long long) n_fetch * hstride;
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

// ---- the lane-parity mode

// the staging slot holding key, -1: none (the whole warp)
__device__ int stage_find(const OffStage& st, int key) {
    const int lane = threadIdx.x;
    for (int b = 0; b < st.n; b += 32) {
        const unsigned m = __ballot_sync(0xffffffffu, b + lane < st.n && st.tag[b + lane] == key);
        if (m) return b + __ffs(m) - 1;
    }
    return -1;
}
// (lane 0) the clock's next slot this visit does not use, now key's
__device__ int stage_take(const OffStage& st, int key, uint32_t seq) {
    int h = *st.hand, s = h;
    for (int t = 0; t < st.n; ++t) {
        s = h;
        h = h + 1 == st.n ? 0 : h + 1;
        if (st.used[s] != seq) break;
    }
    *st.hand = h;
    st.tag[s] = key;
    st.used[s] = seq;
    return s;
}

__global__ void off_lane_plan_kernel(const uint8_t** blob, const int* __restrict__ top, int K, int key0,
                                     const uint8_t* host, size_t hstride, OffStage st, OffFetch* fl, OffCount* cnt,
                                     uint32_t* heat, const LaneCost* __restrict__ cost, int fmt,
                                     const float* __restrict__ xin, int C, const float* __restrict__ sg,
                                     const float* __restrict__ su, int layer, LaneJob* job, LaneState* ls, int* cls) {
    const int lane = threadIdx.x;
    uint32_t seq = 0;
    if (lane == 0) seq = ++cnt->seq;
    seq = __shfl_sync(0xffffffffu, seq, 0);
    if (heat && lane < K) atomicAdd(heat + key0 + top[lane], 1u);
    int miss[kOffMaxK], nm = 0, n_gpu = 0, n_stage = 0;
    for (int i = 0; i < K; ++i) {
        if (blob[i]) {   // in the tier
            ++n_gpu;
            continue;
        }
        const int f = stage_find(st, key0 + top[i]);
        if (f >= 0) {    // in staging from an earlier visit
            if (lane == 0) st.used[f] = seq, blob[i] = st.data + (size_t) f * st.sb;
            ++n_gpu, ++n_stage;
        } else {
            miss[nm++] = i;
        }
        __syncwarp();
    }
    // the split: n_cpu of the nm misses on the lane, the rest fetched - the one whose slower side ends first
    const LaneCost cs = *cost;
    int n_cpu = 0;
    if (cs.force == 1) {
        n_cpu = 0;
    } else if (cs.force == 2) {
        n_cpu = nm;
    } else {
        float best = 3.4e38f;
        for (int n = 0; n <= nm; ++n) {
            const float tc = n ? cs.lat + n * cs.cpu[fmt] : 0.0f;
            const float tg = (n_gpu + nm - n) * cs.gpu[fmt] + (nm - n) * cs.fetch[fmt];
            const float t = fmaxf(tc, tg);
            if (t < best) best = t, n_cpu = n;
        }
    }
    if (lane == 0) {
        for (int i = 0; i < K; ++i) cls[i] = 0;
        int n_fetch = 0;
        for (int m = 0; m < nm - n_cpu; ++m) {
            const int i = miss[m];
            const int s = stage_take(st, key0 + top[i], seq);
            uint8_t* dst = st.data + (size_t) s * st.sb;
            blob[i] = dst;
            fl->src[n_fetch] = host + (size_t) top[i] * hstride;
            fl->dst[n_fetch] = dst;
            ++n_fetch;
        }
        fl->n = n_fetch;
        for (int k = 0; k < n_cpu; ++k) {
            const int i = miss[nm - n_cpu + k];
            cls[i] = 1;
            job->slot[k] = i, job->e[k] = top[i], job->sg[k] = sg[i], job->su[k] = su[i];
            ls->slot[k] = i;
        }
        job->n = (uint32_t) n_cpu;
        job->layer = layer;
        ls->n = n_cpu;
        ls->seq = seq;
        cnt->picks += (unsigned long long) K;
        cnt->tier += (unsigned long long) (n_gpu - n_stage);
        cnt->stage += (unsigned long long) n_stage;
        cnt->fetched += (unsigned long long) n_fetch;
        cnt->bytes += (unsigned long long) n_fetch * hstride;
        cnt->cpu += (unsigned long long) n_cpu;
    }
    if (n_cpu > 0) {   // the input beside the job, then the flag the lane waits for
        float* jx = (float*) ((uint8_t*) job + kLaneJobX);
        for (int i = lane; i < C; i += 32) jx[i] = xin[i];
        __threadfence_system();
        __syncwarp();
        if (lane == 0) *(volatile uint32_t*) &job->seq = seq;
    }
}

// an NVFP4 code's value (ik2_exact.h's table: 0 1 2 3 4 6 8 12, bit 3 the sign) without a table in local memory
__device__ __forceinline__ float kv4(uint32_t mags, uint32_t code) {
    const float m = (float) ((mags >> (4 * (code & 7u))) & 0xFu);
    return (code & 8u) ? -m : m;
}
// 16 exact weights of sub-block s of a row (ik2_exact.h's sub_weights, with wider loads on the device)
__device__ __forceinline__ void sub_w(int ty, const uint8_t* row, int s, float* w) {
    if (ty == x::kNVFP4) {   // (a 36-byte block at a multiple of 4: its scales, then the four sub-blocks' 8 bytes)
        const uint8_t* b = row + (size_t) (s >> 2) * 36;
        const int c = s & 3;
        const uint32_t sc = *(const uint32_t*) b;
        const float d = x::ue4m3((uint8_t) (sc >> (8 * c)));
        const uint32_t q0 = *(const uint32_t*) (b + 4 + 8 * c), q1 = *(const uint32_t*) (b + 8 + 8 * c);
        const uint32_t mags = 0xC8643210u;   // (a code's magnitude by its low 3 bits, a nibble each; bit 3: the sign)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint32_t by = ((j < 4 ? q0 : q1) >> (8 * (j & 3))) & 0xFFu;
            w[j] = x::mul(kv4(mags, by & 0xF), d);
            w[j + 8] = x::mul(kv4(mags, by >> 4), d);
        }
    } else {                 // (Q8_0: 34-byte blocks, at a multiple of 2)
        const uint8_t* b = row + (size_t) (s >> 1) * 34;
        const float d = x::h2f(*(const uint16_t*) b);
        const uint16_t* q = (const uint16_t*) (b + 2 + (s & 1) * 16);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint16_t v = q[j];
            w[2 * j] = x::mul((float) (int8_t) (v & 0xFF), d);
            w[2 * j + 1] = x::mul((float) (int8_t) (v >> 8), d);
        }
    }
}
// chain c (0 .. 3) of a row's dot: sub-blocks c, c + 4, .. into acc [16]
__device__ __forceinline__ void chain_dot(int ty, const uint8_t* row, int K, const float* xs, int c, float* acc) {
#pragma unroll
    for (int j = 0; j < 16; ++j) acc[j] = 0.0f;
    for (int s = c; s < K / 16; s += 4) {
        float w[16];
        sub_w(ty, row, s, w);
        const float* xp = xs + 16 * s;
#pragma unroll
        for (int j = 0; j < 16; ++j) acc[j] = x::fma(xp[j], w[j], acc[j]);
    }
}
// the four chains' threads (lanes t, t^1, t^2, t^3) to the row's sum: (c0 + c1) + (c2 + c3), then the tree
__device__ __forceinline__ float chains_sum(float* acc) {
#pragma unroll
    for (int j = 0; j < 16; ++j) acc[j] = x::add(acc[j], __shfl_xor_sync(0xffffffffu, acc[j], 1));
#pragma unroll
    for (int j = 0; j < 16; ++j) acc[j] = x::add(acc[j], __shfl_xor_sync(0xffffffffu, acc[j], 2));
    return x::tree16(acc);
}

// a block: 32 rows of gate and up, 8 threads a row (4 chains of gate, 4 of up)
__global__ void __launch_bounds__(256) off_gate_up_x_kernel(const uint8_t* const* blob, const int* cls, int ty,
                                                             size_t eb0, const float* __restrict__ xin, int C, int F,
                                                             const float* __restrict__ sg, const float* __restrict__ su,
                                                             float gcap, float ucap, float* __restrict__ h) {
    const int slot = blockIdx.y;
    const uint8_t* base = blob[slot];
    if (cls[slot] || !base) return;
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < C; i += blockDim.x) xs[i] = xin[i];
    __syncthreads();
    const int r = blockIdx.x * 32 + (threadIdx.x >> 3), t = threadIdx.x & 7;
    const uint8_t* row = base + (t >= 4 ? eb0 : 0) + (size_t) r * x::row_bytes(ty, C);
    float acc[16];
    chain_dot(ty, row, C, xs, t & 3, acc);
    const float d = chains_sum(acc);
    const float du = __shfl_down_sync(0xffffffffu, d, 4);
    if (t == 0 && r < F) h[(size_t) slot * F + r] = x::swiglu(x::mul(d, sg[slot]), x::mul(du, su[slot]), gcap, ucap);
}

// a block: 64 rows of down, 4 threads (chains) a row
__global__ void __launch_bounds__(256) off_down_x_kernel(const uint8_t* const* blob, const int* cls, int ty,
                                                          size_t off_down, const float* __restrict__ h, int C, int F,
                                                          float* __restrict__ y) {
    const int slot = blockIdx.y;
    const uint8_t* base = blob[slot];
    if (cls[slot] || !base) return;
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < F; i += blockDim.x) xs[i] = h[(size_t) slot * F + i];
    __syncthreads();
    const int r = blockIdx.x * 64 + (threadIdx.x >> 2);
    const uint8_t* row = base + off_down + (size_t) r * x::row_bytes(ty, F);
    float acc[16];
    chain_dot(ty, row, F, xs, threadIdx.x & 3, acc);
    const float d = chains_sum(acc);
    if ((threadIdx.x & 3) == 0 && r < C) y[(size_t) slot * C + r] = d;
}

__device__ __forceinline__ unsigned long long globaltimer() {
#ifdef STRATA_USE_HIP   // (AMD builds the offload path but never takes it: no PTX timer there, the wall clock instead)
    return (unsigned long long) wall_clock64();
#else
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
#endif
}
__global__ void off_lane_wait_kernel(LaneState* ls, const LaneDone* done, int C, float* __restrict__ y) {
    __shared__ int n;
    if (threadIdx.x == 0) {
        n = ls->n;
        if (n > 0) {
            const uint32_t seq = ls->seq;
            const unsigned long long t0 = globaltimer();
            while (*(const volatile uint32_t*) &done->seq != seq) {
                __nanosleep(256);
                if (globaltimer() - t0 > 20000000000ull) {   // (20 s: the lane is gone - the engine reports it)
                    ls->err = 1;
                    break;
                }
            }
        }
    }
    __syncthreads();
    if (n <= 0) return;
    __threadfence();
    const float* yd = (const float*) ((const uint8_t*) done + kLaneDoneY);
    for (int k = 0; k < n; ++k) {
        const int slot = ls->slot[k];
#ifdef STRATA_USE_HIP   // (AMD: no __ldcv - a volatile read of the host's rows; the path is never taken there)
        for (int r = threadIdx.x; r < C; r += blockDim.x)
            y[(size_t) slot * C + r] = *(const volatile float*) (yd + (size_t) k * C + r);
#else
        for (int r = threadIdx.x; r < C; r += blockDim.x) y[(size_t) slot * C + r] = __ldcv(yd + (size_t) k * C + r);
#endif
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

void off_lane_plan(const uint8_t** blob, const int* top, int K, int key0, const uint8_t* host, size_t hstride,
                   OffStage st, OffFetch* fl, OffCount* cnt, uint32_t* heat, const LaneCost* cost, int fmt,
                   const float* x, int C, const float* sg, const float* su, int layer, LaneJob* job, LaneState* ls,
                   int* cls, cudaStream_t s) {
    off_lane_plan_kernel<<<1, 32, 0, s>>>(blob, top, K, key0, host, hstride, st, fl, cnt, heat, cost, fmt, x, C, sg, su,
                                          layer, job, ls, cls);
}

void off_gate_up_x(const uint8_t* const* blob, const int* cls, int K, int ty, size_t eb0, const float* x, int C, int F,
                   const float* sg, const float* su, float gcap, float ucap, float* h, cudaStream_t s) {
    off_gate_up_x_kernel<<<dim3((F + 31) / 32, K), 256, (size_t) C * 4, s>>>(blob, cls, ty, eb0, x, C, F, sg, su, gcap,
                                                                              ucap, h);
}

void off_down_x(const uint8_t* const* blob, const int* cls, int K, int ty, size_t off_down, const float* h, int C,
                int F, float* y, cudaStream_t s) {
    off_down_x_kernel<<<dim3((C + 63) / 64, K), 256, (size_t) F * 4, s>>>(blob, cls, ty, off_down, h, C, F, y);
}

void off_lane_wait(LaneState* ls, const LaneDone* done, int C, float* y, cudaStream_t s) {
    off_lane_wait_kernel<<<1, 256, 0, s>>>(ls, done, C, y);
}

}  // namespace ik2::k
