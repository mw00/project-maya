// src/ik2/ik2_off.cu - maya-ik2's offload token path (ik2_off.cuh): the GPU fetch of a token's missing experts.
#include "ik2_off.cuh"

#include "ik2_exact.h"

namespace ik2::k {
namespace {

// an expert's place in pinned host memory as the device sees it: the layer's base + e * stride, or (stride 0: Windows,
// only the experts the VRAM tier does not hold pinned, a run's device address of its own) the layer's row of the table
__device__ __forceinline__ const uint8_t* off_src(const uint8_t* host, size_t hstride, int e) {
    return hstride ? host + (size_t) e * hstride : ((const uint8_t* const*) host)[e];
}

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
                fl->src[n_fetch] = off_src(host, hstride, top[i]);
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
        cnt->bytes += (unsigned long long) n_fetch * (hstride ? hstride : st.sb);
    }
}

__global__ void off_cache_plan_kernel(const uint8_t** blob, const int* __restrict__ top, int K, int key0,
                                      const uint8_t* host, size_t hstride, OffCache dc, OffStage st, bool use_st,
                                      OffFetch* fl, OffCount* cnt, uint32_t* heat, int* cls) {
    const int lane = threadIdx.x;
    if (heat && lane < K) atomicAdd(heat + key0 + top[lane], 1u);
    if (lane != 0) return;
    const uint32_t seq = ++cnt->seq;
    int slot[kOffMaxK];
    // Pass 1: resolve all hits and pin them for this visit so misses cannot evict a sibling pick.
    for (int i = 0; i < K; ++i) {
        const int s = dc.loc[key0 + top[i]];
        slot[i] = s;
        if (s >= 0) dc.used[s] = seq;
    }
    int n_hit = 0, n_stage = 0, n_fetch = 0, n_staged = 0;
    // Pass 2: increment hit frequencies; allocate Clock-LFU victims (or staging bypass for cold !is_ghost misses).
    for (int i = 0; i < K; ++i) {
        int s = slot[i];
        const int key = key0 + top[i];
        if (s >= 0) {
            const uint8_t f = dc.freq[s];
            if ((int) f < dc.max_freq) dc.freq[s] = (uint8_t) (f + 1u);
            blob[i] = dc.data + (size_t) s * dc.sb;
            if (cls) cls[i] = 0;
            ++n_hit;
            continue;
        }
        // Check if an earlier pick in the same visit already allocated this key (defensive against duplicate top[i]).
        s = dc.loc[key];
        if (s >= 0) {
            blob[i] = dc.data + (size_t) s * dc.sb;
            if (cls) cls[i] = 1;   // (an earlier pick of this visit fetched it: its bytes arrive with the copy)
            ++n_hit;
            continue;
        }
        if (use_st && st.n > 0) {
            int found = -1;
            for (int j = 0; j < st.n; ++j) {
                if (st.tag[j] == key) { found = j; break; }
            }
            if (found >= 0) {
                const bool same_visit = (st.used[found] == seq);
                st.used[found] = seq;
                blob[i] = st.data + (size_t) found * st.sb;
                if (cls) cls[i] = same_visit ? 1 : 0;
                ++n_stage;
                continue;
            }
            const uint32_t g0 = dc.ghost[key];
            const bool is_ghost0 = g0 != 0 && (*dc.evict_seq - g0) <= (uint32_t) dc.ghost_win;
            if (!is_ghost0 && n_staged < st.n) {
                int sh = *st.hand, ss = sh;
                bool found_free = false;
                for (int t = 0; t < st.n; ++t) {
                    ss = sh;
                    sh = sh + 1 == st.n ? 0 : sh + 1;
                    if (st.used[ss] != seq) { found_free = true; break; }
                }
                if (found_free) {
                    *st.hand = sh;
                    st.tag[ss] = key;
                    st.used[ss] = seq;
                    const uint32_t es = ++(*dc.evict_seq);
                    dc.ghost[key] = es;
                    uint8_t* dst = st.data + (size_t) ss * st.sb;
                    blob[i] = dst;
                    fl->src[n_fetch] = off_src(host, hstride, top[i]);
                    fl->dst[n_fetch] = dst;
                    if (cls) cls[i] = 1;
                    ++n_fetch;
                    ++n_staged;
                    continue;
                }
            }
        }
        int h = *dc.hand, v = -1;
        const int max_steps = (dc.max_freq + 2) * dc.n;
        for (int step = 0; step < max_steps; ++step) {
            const int c = h;
            h = (h + 1 == dc.n) ? 0 : (h + 1);
            if (dc.used[c] == seq) continue;
            const int oc = dc.owner[c];   // (PF-B': the pin - the decode never evicts the pinned pairs)
            if (oc >= 0 && dc.pin && dc.pin[oc]) continue;
            if (oc < 0) {
                v = c;
                break;
            }
            const uint8_t f = dc.freq[c];
            if (f > 0) {
                dc.freq[c] = (uint8_t) (f - 1u);
                continue;
            }
            v = c;
            break;
        }
        *dc.hand = h;
        const int old_k = dc.owner[v];
        if (old_k >= 0) {
            dc.loc[old_k] = -1;
            const uint32_t es = ++(*dc.evict_seq);
            dc.ghost[old_k] = es;
        }
        const uint32_t g = dc.ghost[key];
        const bool is_ghost = g != 0 && (*dc.evict_seq - g) <= (uint32_t) dc.ghost_win;
        dc.owner[v] = key;
        dc.loc[key] = v;
        dc.used[v] = seq;
        dc.freq[v] = (uint8_t) (is_ghost ? 2 : 0);
        uint8_t* dst = dc.data + (size_t) v * dc.sb;
        blob[i] = dst;
        fl->src[n_fetch] = off_src(host, hstride, top[i]);
        fl->dst[n_fetch] = dst;
        if (cls) cls[i] = 1;
        ++n_fetch;
    }
    fl->n = n_fetch;
    cnt->picks += (unsigned long long) K;
    cnt->tier += (unsigned long long) n_hit;
    cnt->stage += (unsigned long long) n_stage;
    cnt->fetched += (unsigned long long) n_fetch;
    cnt->bytes += (unsigned long long) n_fetch * (hstride ? hstride : dc.sb);
}


__global__ void off_cache_decay_kernel(uint32_t* score, const int* __restrict__ loc, int n_keys, int shift,
                                       uint32_t seed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_keys) return;
    if (shift >= 32) {
        score[i] = (loc && loc[i] >= 0) ? seed : 0u;
    } else if (shift > 0) {
        uint32_t v = score[i] >> shift;
        if (loc && loc[i] >= 0 && v < seed) v = seed;
        score[i] = v;
    }
}

__global__ void off_cache_plan_lfu_kernel(const uint8_t** blob, const int* __restrict__ top, int K, int key0,
                                          const uint8_t* host, size_t hstride, OffCache dc, OffStage st, bool use_st,
                                          OffFetch* fl, OffCount* cnt, uint32_t* heat, int* cls) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    __shared__ uint32_t s_seq;
    __shared__ int s_slot[kOffMaxK];
    __shared__ int s_miss_idx[kOffMaxK];
    __shared__ int s_n_miss;
    __shared__ int s_need_scan;
    __shared__ int s_cur_i;
    __shared__ int s_cur_key;
    __shared__ unsigned long long s_wmet[8];
    __shared__ int s_wslot[8];
    __shared__ int s_n_hit, s_n_stage, s_n_fetch, s_n_staged;

    if (tid < K) {
        const int key = key0 + top[tid];
        if (heat) atomicAdd(heat + key, 1u);
        atomicAdd(dc.score + key, 1u);
    }
    if (tid == 0) {
        s_seq = ++cnt->seq;
        s_n_miss = 0;
        s_n_hit = 0;
        s_n_stage = 0;
        s_n_fetch = 0;
        s_n_staged = 0;
    }
    __syncthreads();

    const uint32_t seq = s_seq;
    if (tid < K) {
        const int s = dc.loc[key0 + top[tid]];
        s_slot[tid] = s;
        if (s >= 0) dc.used[s] = seq;
    }
    __syncthreads();

    if (tid == 0) {
        for (int i = 0; i < K; ++i) {
            const int s = s_slot[i];
            if (s >= 0) {
                blob[i] = dc.data + (size_t) s * dc.sb;
                if (cls) cls[i] = 0;
                ++s_n_hit;
            } else {
                s_miss_idx[s_n_miss++] = i;
            }
        }
        if (s_n_miss == 0) {
            fl->n = 0;
            cnt->picks += (unsigned long long) K;
            cnt->tier += (unsigned long long) s_n_hit;
        }
    }
    __syncthreads();
    if (s_n_miss == 0) return;

    const int nm = s_n_miss;
    for (int m = 0; m < nm; ++m) {
        if (tid == 0) {
            const int i = s_miss_idx[m];
            const int key = key0 + top[i];
            s_cur_i = i;
            s_cur_key = key;
            s_need_scan = 0;
            const int s = dc.loc[key];
            if (s >= 0) {
                blob[i] = dc.data + (size_t) s * dc.sb;
                if (cls) cls[i] = 1;
                ++s_n_hit;
            } else {
                int found = -1;
                if (use_st && st.n > 0) {
                    for (int j = 0; j < st.n; ++j) {
                        if (st.tag[j] == key) { found = j; break; }
                    }
                }
                if (found >= 0) {
                    const bool same_visit = (st.used[found] == seq);
                    st.used[found] = seq;
                    blob[i] = st.data + (size_t) found * st.sb;
                    if (cls) cls[i] = same_visit ? 1 : 0;
                    ++s_n_stage;
                } else {
                    s_need_scan = 1;
                }
            }
        }
        __syncthreads();
        if (!s_need_scan) continue;

        unsigned long long best_met = ~0ull;
        int best_slot = -1;
        for (int c = tid; c < dc.n; c += 256) {
            if (dc.used[c] == seq) continue;
            const int oc = dc.owner[c];
            if (oc >= 0 && dc.pin && dc.pin[oc]) continue;
            const unsigned long long met = (oc < 0)
                ? 0ull
                : (((unsigned long long) dc.score[oc] << 32) | (unsigned long long) dc.used[c]);
            if (met < best_met) {
                best_met = met;
                best_slot = c;
            }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const unsigned long long o_met = __shfl_down_sync(0xffffffffu, best_met, off);
            const int o_slot = __shfl_down_sync(0xffffffffu, best_slot, off);
            if (o_met < best_met) {
                best_met = o_met;
                best_slot = o_slot;
            }
        }
        if (lane == 0) {
            s_wmet[warp] = best_met;
            s_wslot[warp] = best_slot;
        }
        __syncthreads();

        if (tid == 0) {
            unsigned long long g_met = s_wmet[0];
            int v = s_wslot[0];
            for (int w = 1; w < 8; ++w) {
                if (s_wmet[w] < g_met) {
                    g_met = s_wmet[w];
                    v = s_wslot[w];
                }
            }
            const int i = s_cur_i;
            const int key = s_cur_key;
            const uint32_t k_sc = dc.score[key];
            const uint32_t v_sc = (uint32_t) (g_met >> 32);
            bool bypassed = false;
            if (use_st && st.n > 0 && s_n_staged < st.n && g_met != 0ull && k_sc < v_sc) {
                int sh = *st.hand, ss = sh;
                bool found_free = false;
                for (int t = 0; t < st.n; ++t) {
                    ss = sh;
                    sh = sh + 1 == st.n ? 0 : sh + 1;
                    if (st.used[ss] != seq) { found_free = true; break; }
                }
                if (found_free) {
                    *st.hand = sh;
                    st.tag[ss] = key;
                    st.used[ss] = seq;
                    uint8_t* dst = st.data + (size_t) ss * st.sb;
                    blob[i] = dst;
                    fl->src[s_n_fetch] = off_src(host, hstride, top[i]);
                    fl->dst[s_n_fetch] = dst;
                    if (cls) cls[i] = 1;
                    ++s_n_fetch;
                    ++s_n_staged;
                    bypassed = true;
                }
            }
            if (!bypassed) {
                const int old_k = dc.owner[v];
                if (old_k >= 0) dc.loc[old_k] = -1;
                dc.owner[v] = key;
                dc.loc[key] = v;
                dc.used[v] = seq;
                uint8_t* dst = dc.data + (size_t) v * dc.sb;
                blob[i] = dst;
                fl->src[s_n_fetch] = off_src(host, hstride, top[i]);
                fl->dst[s_n_fetch] = dst;
                if (cls) cls[i] = 1;
                ++s_n_fetch;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        fl->n = s_n_fetch;
        cnt->picks += (unsigned long long) K;
        cnt->tier += (unsigned long long) s_n_hit;
        cnt->stage += (unsigned long long) s_n_stage;
        cnt->fetched += (unsigned long long) s_n_fetch;
        cnt->bytes += (unsigned long long) s_n_fetch * (hstride ? hstride : dc.sb);
    }
}

// 4 x 16 bytes in flight a thread: the bus's latency (a microsecond or two) hidden by many reads at once
__global__ void off_fetch_copy_kernel(const OffFetch* __restrict__ fl, size_t n16) {
    const int n = fl->n;
    if (n <= 0) return;
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
            fl->src[n_fetch] = off_src(host, hstride, top[i]);
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
        cnt->bytes += (unsigned long long) n_fetch * (hstride ? hstride : st.sb);
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

void off_cache_decay(uint32_t* score, const int* loc, int n_keys, int shift, uint32_t seed, cudaStream_t s) {
    if (!score || n_keys <= 0) return;
    off_cache_decay_kernel<<<(n_keys + 255) / 256, 256, 0, s>>>(score, loc, n_keys, shift, seed);
}

void off_cache_plan(const uint8_t** blob, const int* top, int K, int key0, const uint8_t* host, size_t hstride,
                    OffCache dc, OffFetch* fl, OffCount* cnt, uint32_t* heat, int* cls, cudaStream_t s,
                    const OffStage* st) {
    OffStage dummy{};
    if (dc.lfu_mode && dc.score) {
        off_cache_plan_lfu_kernel<<<1, 256, 0, s>>>(blob, top, K, key0, host, hstride, dc, st ? *st : dummy,
                                                    st != nullptr, fl, cnt, heat, cls);
        return;
    }
    off_cache_plan_kernel<<<1, kOffMaxK, 0, s>>>(blob, top, K, key0, host, hstride, dc, st ? *st : dummy,
                                                 st != nullptr, fl, cnt, heat, cls);
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
