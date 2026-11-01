// src/kernels/cuda/g4_kernels.cu - see include/strata/kernels/g4_kernels.hpp.
#include "strata/kernels/g4_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
unsigned blocks(int64_t n, int b) { return (unsigned) ((n + b - 1) / b); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (int) (blockDim.x >> 5);
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < nw; ++i) t += red[i];
    return t;
}
__device__ __forceinline__ float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

// block (token, head), D threads: q / k: rms * weight, NeoX rope over D; v: rms
// a cache position's cell: the position itself, or its slot in a ring of kv.ring cells (a sliding-window layer)
__device__ __forceinline__ int g4_cell(const G4Kv& kv, int pos) { return kv.ring > 0 ? pos % kv.ring : pos; }

__global__ void prep_kernel(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                            int n_head, int n_kv, const float* __restrict__ q_norm, const float* __restrict__ k_norm,
                            float eps, float rope_base, const float* __restrict__ ff, int p0, float* __restrict__ q_out,
                            G4Kv kv, int rope_mode, int v_norm) {
    __shared__ float red[32];
    __shared__ float row[512];
    const int D = blockDim.x, t = blockIdx.x, hh = blockIdx.y, d = threadIdx.x, pos = p0 + t;
    const int nh = n_head + n_kv;
    const bool is_q = hh < n_head, is_k = !is_q && hh < nh, is_v = hh >= nh;
    const int head = is_q ? hh : is_k ? hh - n_head : hh - nh;
    const float* src = is_q ? q + ((size_t) t * n_head + head) * D : (is_k ? k : v) + ((size_t) t * n_kv + head) * D;
    const float val = src[d];
    const float inv = rsqrtf(block_sum(val * val, red) / (float) D + eps);
    if (is_v) {
        kv.v[((size_t) head * kv.max_ctx + g4_cell(kv, pos)) * D + d] = __half_as_ushort(__float2half(v_norm ? val * inv : val));
        return;
    }
    row[d] = val * inv * (is_q ? q_norm[d] : k_norm[d]);
    __syncthreads();
    float r = row[d];
    if (rope_mode == 1) {   // NeoX: (i, i + D / 2)
        const int half = D / 2, pair = d < half ? d : d - half;
        float theta = (float) pos * powf(rope_base, -2.0f * (float) pair / (float) D);
        if (ff) theta /= ff[pair];
        float s, c;
        sincosf(theta, &s, &c);
        const float a = row[pair], b = row[pair + half];
        r = d < half ? a * c - b * s : a * s + b * c;
    } else if (rope_mode == 2) {   // the plain form: (2 i, 2 i + 1)
        const int pair = d >> 1;
        float theta = (float) pos * powf(rope_base, -2.0f * (float) pair / (float) D);
        if (ff) theta /= ff[pair];
        float s, c;
        sincosf(theta, &s, &c);
        const float a = row[2 * pair], b = row[2 * pair + 1];
        r = (d & 1) == 0 ? a * c - b * s : a * s + b * c;
    }
    if (is_q) q_out[((size_t) t * n_head + head) * D + d] = r;
    else kv.k[((size_t) head * kv.max_ctx + g4_cell(kv, pos)) * D + d] = __half_as_ushort(__float2half(r));
}

// ---- the attention: q35_kernels.cu's attn_tc_kernel, generic in the head size (AD), the output dims a block
// writes (VD: AD / VD blocks share a row tile at D 512, the registers' limit), a window and a scale; no gate.
// grid (row tiles, n_kv, splits * AD / VD), AW warps of 16 rows (rows: token-major, G heads a token)
constexpr int AK = 32, LDS = AK + 4, LDP = AK + 8;
template <int AD, int VD, int AW>
constexpr size_t attn_smem() {
    return (size_t) AW * 16 * (AD + 8) * 2 + (size_t) AK * (AD + 8) * 2 + (size_t) AK * (VD + 8) * 2 +
           (size_t) AW * 16 * LDS * 4 + (size_t) AW * 16 * LDP * 2;
}
template <int AD, int VD, int AW>
__global__ void __launch_bounds__(AW * 32) attn_kernel(const float* __restrict__ q, G4Kv kv, int p0, int T, int n_head,
                                                       int G, int window, int kbase, int split_len, float scale,
                                                       float* __restrict__ out, float* __restrict__ part_o,
                                                       float* __restrict__ part_ml) {
    using namespace nvcuda;
    constexpr int AR = AW * 16, LDQ = AD + 8, LDV = VD + 8, NV = AD / VD;
    extern __shared__ __align__(32) unsigned char smem[];
    __half* sQ = (__half*) smem;
    __half* sK = sQ + AR * LDQ;
    __half* sV = sK + AK * LDQ;
    float* sS = (float*) (sV + AK * LDV);
    __half* sP = (__half*) (sS + AW * 16 * LDS);
    const int kvh = blockIdx.y, vpart = blockIdx.z % NV, split = blockIdx.z / NV, n_split = gridDim.z / NV;
    const int voff = vpart * VD;
    const int rows = T * G, r0 = blockIdx.x * AR;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int qld = n_head * AD;
    for (int i = tid; i < AR * (AD / 4); i += AW * 32) {
        const int r = i / (AD / 4), d = (i % (AD / 4)) * 4, row = r0 + r;
        float4 f = make_float4(0.f, 0.f, 0.f, 0.f);
        if (row < rows) {
            const int tok = row / G, head = kvh * G + row % G;
            f = *(const float4*) (q + (size_t) tok * qld + (size_t) head * AD + d);
        }
        __half2* dst = (__half2*) (sQ + r * LDQ + d);
        dst[0] = __floats2half2_rn(f.x * scale, f.y * scale);
        dst[1] = __floats2half2_rn(f.z * scale, f.w * scale);
    }
    const int last_row = min(r0 + AR, rows) - 1;
    const int kend = p0 + last_row / G + 1;
    const int klo = window > 0 ? max(0, p0 + r0 / G - window + 1) : 0;   // the block's first visible key
    const int ks = kbase + split * split_len, ke = min(kend, ks + split_len);
    const int kstart = max(ks, ks + (klo - ks) / AK * AK);
    const int wr0 = r0 + w * 16;
    const bool w_on = wr0 < rows;
    const int wlast = w_on ? p0 + (min(wr0 + 16, rows) - 1) / G : -1;
    const int ri = lane >> 1, hi = lane & 1;
    const int row = wr0 + ri;
    const bool row_ok = row < rows;
    const int pos = row_ok ? p0 + row / G : -1;
    const int lo = window > 0 ? pos - window + 1 : 0;
    float* Sw = sS + w * 16 * LDS;
    __half* Pw = sP + w * 16 * LDP;
    constexpr float kLazy = 8.0f;
    float m = -INFINITY, l = 0.0f;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fo[VD / 16];
#pragma unroll
    for (int n = 0; n < VD / 16; ++n) wmma::fill_fragment(fo[n], 0.0f);
    for (int k0 = kstart; k0 < ke; k0 += AK) {
        __syncthreads();
        for (int i = tid; i < AK * (AD / 8); i += AW * 32) {
            const int r = i / (AD / 8), d = (i % (AD / 8)) * 8, key = k0 + r;
            uint4 v4 = make_uint4(0, 0, 0, 0);
            if (key < ke) v4 = __ldg((const uint4*) (kv.k + ((size_t) kvh * kv.max_ctx + g4_cell(kv, key)) * AD + d));
            *(uint4*) (sK + r * LDQ + d) = v4;
        }
        for (int i = tid; i < AK * (VD / 8); i += AW * 32) {
            const int r = i / (VD / 8), d = (i % (VD / 8)) * 8, key = k0 + r;
            uint4 v4 = make_uint4(0, 0, 0, 0);
            if (key < ke) v4 = __ldg((const uint4*) (kv.v + ((size_t) kvh * kv.max_ctx + g4_cell(kv, key)) * AD + voff + d));
            *(uint4*) (sV + r * LDV + d) = v4;
        }
        __syncthreads();
        if (!w_on || k0 > wlast) continue;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[AK / 16];
#pragma unroll
        for (int n = 0; n < AK / 16; ++n) wmma::fill_fragment(acc[n], 0.0f);
#pragma unroll 4
        for (int kk = 0; kk < AD / 16; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fq;
            wmma::load_matrix_sync(fq, sQ + (w * 16) * LDQ + kk * 16, LDQ);
#pragma unroll
            for (int n = 0; n < AK / 16; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::load_matrix_sync(fb, sK + (n * 16) * LDQ + kk * 16, LDQ);
                wmma::mma_sync(acc[n], fq, fb, acc[n]);
            }
        }
#pragma unroll
        for (int n = 0; n < AK / 16; ++n) wmma::store_matrix_sync(Sw + n * 16, acc[n], LDS, wmma::mem_row_major);
        __syncwarp();
        constexpr int KH = AK / 2;
        float sv[KH], mt = -INFINITY;
#pragma unroll
        for (int j = 0; j < KH; ++j) {
            const int c = hi * KH + j, key = k0 + c;
            sv[j] = row_ok && key < ke && key <= pos && key >= lo ? Sw[ri * LDS + c] : -INFINITY;
            mt = fmaxf(mt, sv[j]);
        }
        mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 1));
        const bool need = mt > m + kLazy;
        float corr = 1.0f;
        if (need) {
            corr = m == -INFINITY ? 0.0f : __expf(m - mt);
            l *= corr;
            m = mt;
        }
        float ps = 0.0f;
#pragma unroll
        for (int j = 0; j < KH; ++j) {
            const float pj = sv[j] == -INFINITY ? 0.0f : __expf(sv[j] - m);
            ps += pj;
            Pw[ri * LDP + hi * KH + j] = __float2half(pj);
        }
        l += ps + __shfl_xor_sync(0xffffffffu, ps, 1);
        __syncwarp();
        if (__any_sync(0xffffffffu, need)) {
#pragma unroll
            for (int n = 0; n < VD / 16; ++n) {
                wmma::store_matrix_sync(Sw, fo[n], LDS, wmma::mem_row_major);
                __syncwarp();
#pragma unroll
                for (int j = 0; j < 8; ++j) Sw[ri * LDS + hi * 8 + j] *= corr;
                __syncwarp();
                wmma::load_matrix_sync(fo[n], Sw, LDS, wmma::mem_row_major);
                __syncwarp();
            }
        }
#pragma unroll
        for (int kk = 0; kk < AK / 16; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fp;
            wmma::load_matrix_sync(fp, Pw + kk * 16, LDP);
#pragma unroll
            for (int n = 0; n < VD / 16; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fv;
                wmma::load_matrix_sync(fv, sV + (kk * 16) * LDV + n * 16, LDV);
                wmma::mma_sync(fo[n], fp, fv, fo[n]);
            }
        }
        __syncwarp();
    }
    if (!w_on) return;
    const int tok = row_ok ? row / G : 0, head = kvh * G + (row_ok ? row % G : 0);
    const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
#pragma unroll
    for (int n = 0; n < VD / 16; ++n) {
        wmma::store_matrix_sync(Sw, fo[n], LDS, wmma::mem_row_major);
        __syncwarp();
        if (row_ok) {
            const int d0 = voff + n * 16 + hi * 8;
            if (n_split == 1) {
                float* o = out + (size_t) tok * qld + (size_t) head * AD + d0;
#pragma unroll
                for (int j = 0; j < 8; ++j) o[j] = Sw[ri * LDS + hi * 8 + j] * inv_l;
            } else {
                float* o = part_o + (((size_t) split * gridDim.y + kvh) * rows + row) * AD + d0;
#pragma unroll
                for (int j = 0; j < 8; ++j) o[j] = Sw[ri * LDS + hi * 8 + j];
            }
        }
        __syncwarp();
    }
    if (n_split > 1 && row_ok && hi == 0 && vpart == 0) {
        part_ml[(((size_t) split * gridDim.y + kvh) * rows + row) * 2] = m;
        part_ml[(((size_t) split * gridDim.y + kvh) * rows + row) * 2 + 1] = l;
    }
}

// block (row, kv head), D threads: the splits' partials folded
__global__ void merge_kernel(const float* __restrict__ part_o, const float* __restrict__ part_ml, int rows, int n_split,
                             int G, int n_head, float* __restrict__ out) {
    const int D = blockDim.x, row = blockIdx.x, d = threadIdx.x;
    const int kvh = blockIdx.y, tok = row / G, head = kvh * G + row % G;
    const int n_kvh = gridDim.y;
    float M = -INFINITY;
    for (int s = 0; s < n_split; ++s) {
        const float* ml = part_ml + (((size_t) s * n_kvh + kvh) * rows + row) * 2;
        if (ml[1] > 0.0f) M = fmaxf(M, ml[0]);
    }
    float a = 0.0f, L = 0.0f;
    for (int s = 0; s < n_split; ++s) {
        const float* ml = part_ml + (((size_t) s * n_kvh + kvh) * rows + row) * 2;
        if (!(ml[1] > 0.0f)) continue;
        const float f = __expf(ml[0] - M);
        a += part_o[(((size_t) s * n_kvh + kvh) * rows + row) * D + d] * f;
        L += ml[1] * f;
    }
    out[((size_t) tok * n_head + head) * D + d] = L > 0.0f ? a / L : 0.0f;
}

__global__ void gelu_mul_kernel(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ o, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = gelu_tanh(g[i]) * u[i];
}
// block (token, layer), 256 threads
__global__ void ple_kernel(const float* __restrict__ proj, const float* __restrict__ tok, const float* __restrict__ w,
                           float eps, float pscale, int n_layer, float* __restrict__ out) {
    __shared__ float red[32];
    const int t = blockIdx.x, l = blockIdx.y, j = threadIdx.x;
    const size_t i = ((size_t) t * n_layer + l) * 256 + j;
    const float x = proj[i] * pscale;
    const float inv = rsqrtf(block_sum(x * x, red) / 256.0f + eps);
    out[i] = (x * inv * w[j] + tok[i] * 16.0f) * 0.7071067811865476f;
}
__global__ void gelu_ple_kernel(float* __restrict__ g, const float* __restrict__ ple, int l, int n_layer, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * 256) return;
    const int t = i / 256, j = i % 256;
    g[i] = gelu_tanh(g[i]) * ple[((size_t) t * n_layer + l) * 256 + j];
}
__global__ void add_scale_kernel(float* __restrict__ x, const float* __restrict__ y, float s, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = (x[i] + y[i]) * s;
}
__global__ void sigmoid_mul_kernel(float* __restrict__ x, const float* __restrict__ g, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 1.0f / (1.0f + __expf(-g[i]));
}
__global__ void scale_kernel(float* __restrict__ x, float s, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}
__global__ void softcap_kernel(float* __restrict__ x, int64_t ld, int n, float cap) {
    float* r = x + (size_t) blockIdx.y * ld;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) r[i] = cap * tanhf(r[i] / cap);
}

constexpr int kSplitLen = 128;

// ---- one token (rows = G <= 8 heads of a kv head): split-K over the keys, a warp a key at a time (lane: D / 32
// dims), the online softmax in registers; partials as attn_kernel's (merge_kernel folds them).  grid (n_kv, splits),
// 4 warps; the block's warps fold through shared memory.
template <int D, int G>
__global__ void __launch_bounds__(128) dec_kernel(const float* __restrict__ q, G4Kv kv, int pos, int n_head, int window,
                                                  int kbase, int split_len, float scale, float* __restrict__ out,
                                                  float* __restrict__ part_o, float* __restrict__ part_ml, bool round_h,
                                                  int rep) {
    constexpr int PL = D / 32;   // dims a lane: 4, 8 or 16 (D 128: lane l the 4 dims 4 l ..; else chunks of 8)
    __shared__ float sm_m[4][G], sm_l[4][G];
    __shared__ float sm_o[4][G][D];
    const int kvh = blockIdx.x, split = blockIdx.y, n_split = gridDim.y;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int lo = window > 0 ? max(0, pos - window + 1) : 0;
    const int ks = max(lo, kbase + split * split_len), ke = min(pos + 1, kbase + (split + 1) * split_len);
    // the lane's dims: chunk c (8 halves) at c * 256 + lane * 8 (D 512: 2 chunks; D 256: 1)
    auto dim = [&](int j) { return D == 128 ? lane * 4 + j : (j / 8) * 256 + lane * 8 + (j % 8); };   // lane's j-th dim
    float qv[G][PL];
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int j = 0; j < PL; ++j) {
            const float x = q[(size_t) (kvh * G + g) * D + dim(j)] * scale;
            qv[g][j] = round_h ? __half2float(__float2half(x)) : x;
        }
    float m[G], l[G], acc[G][PL];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        m[g] = -INFINITY;
        l[g] = 0.0f;
#pragma unroll
        for (int j = 0; j < PL; ++j) acc[g][j] = 0.0f;
    }
    for (int key = ks + w; key < ke; key += 4) {
        // (rep > 1: blockIdx.x is a virtual kv head of G of the real one's rep x G query heads)
        const __half* kr = (const __half*) kv.k + ((size_t) (kvh / rep) * kv.max_ctx + g4_cell(kv, key)) * D;
        const __half* vr = (const __half*) kv.v + ((size_t) (kvh / rep) * kv.max_ctx + g4_cell(kv, key)) * D;
        float kf[PL], vf[PL];
        if (D == 128) {
            const uint2 ku = __ldg((const uint2*) (kr + lane * 4)), vu = __ldg((const uint2*) (vr + lane * 4));
            const __half2* k2 = (const __half2*) &ku;
            const __half2* v2 = (const __half2*) &vu;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const float2 a = __half22float2(k2[j]), b = __half22float2(v2[j]);
                kf[2 * j] = a.x; kf[2 * j + 1] = a.y;
                vf[2 * j] = b.x; vf[2 * j + 1] = b.y;
            }
        } else {
#pragma unroll
            for (int c = 0; c < (PL >= 8 ? PL / 8 : 1); ++c) {
                const uint4 ku = __ldg((const uint4*) (kr + c * 256 + lane * 8));
                const uint4 vu = __ldg((const uint4*) (vr + c * 256 + lane * 8));
                const __half2* k2 = (const __half2*) &ku;
                const __half2* v2 = (const __half2*) &vu;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 a = __half22float2(k2[j]), b = __half22float2(v2[j]);
                    kf[c * 8 + 2 * j] = a.x; kf[c * 8 + 2 * j + 1] = a.y;
                    vf[c * 8 + 2 * j] = b.x; vf[c * 8 + 2 * j + 1] = b.y;
                }
            }
        }
#pragma unroll
        for (int g = 0; g < G; ++g) {
            float sv = 0.0f;
#pragma unroll
            for (int j = 0; j < PL; ++j) sv += qv[g][j] * kf[j];
            sv = warp_sum(sv);
            const float mn = fmaxf(m[g], sv), corr = __expf(m[g] - mn);
            float p = __expf(sv - mn);
            if (round_h) p = __half2float(__float2half(p));
            l[g] = l[g] * corr + p;
            m[g] = mn;
#pragma unroll
            for (int j = 0; j < PL; ++j) acc[g][j] = acc[g][j] * corr + p * vf[j];
        }
    }
    // fold the 4 warps
    if (lane == 0)
#pragma unroll
        for (int g = 0; g < G; ++g) { sm_m[w][g] = m[g]; sm_l[w][g] = l[g]; }
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int j = 0; j < PL; ++j) sm_o[w][g][dim(j)] = acc[g][j];
    __syncthreads();
    for (int i = threadIdx.x; i < G * D; i += 128) {
        const int g = i / D, d = i % D;
        float M = -INFINITY;
#pragma unroll
        for (int ww = 0; ww < 4; ++ww) if (sm_l[ww][g] > 0.0f) M = fmaxf(M, sm_m[ww][g]);
        float a = 0.0f, L = 0.0f;
#pragma unroll
        for (int ww = 0; ww < 4; ++ww) {
            if (!(sm_l[ww][g] > 0.0f)) continue;
            const float f = __expf(sm_m[ww][g] - M);
            a += sm_o[ww][g][d] * f;
            L += sm_l[ww][g] * f;
        }
        const int head = kvh * G + g;
        if (n_split == 1) {
            out[(size_t) head * D + d] = L > 0.0f ? a / L : 0.0f;
        } else {
            part_o[(((size_t) split * gridDim.x + kvh) * G + g) * D + d] = a;
            if (d == 0) {
                part_ml[(((size_t) split * gridDim.x + kvh) * G + g) * 2] = M;
                part_ml[(((size_t) split * gridDim.x + kvh) * G + g) * 2 + 1] = L;
            }
        }
    }
}

template <int D, int G>
void launch_dec(const float* q, const G4Kv& kv, int p0, int n_head, int n_kv_real, int window, float scale, float* out,
                float* scratch, cudaStream_t s, int rep = 1) {
    const int n_kv = n_kv_real * rep;   // (rep > 1: each kv head's query heads in rep blocks of G)
    const int kbase = window > 0 ? std::max(0, p0 - window + 1) : 0;
    const int n_keys = p0 + 1 - kbase;
    // splits: ~4 blocks a SM over the kv heads, at least 64 keys a split (16 a warp)
    // (at most g4_attn_scratch_floats' splits: one a kSplitLen keys + 1)
    int n_split = std::max(1, std::min({320 / n_kv, (n_keys + 63) / 64, (n_keys + kSplitLen - 1) / kSplitLen + 1}));
    const int split_len = (n_keys + n_split - 1) / n_split;
    n_split = (n_keys + split_len - 1) / split_len;
    float* part_o = scratch;
    float* part_ml = scratch + (size_t) n_split * G * n_kv * D;
    // q and p rounded to FP16 as llama.cpp's (FP16 GEMM) attention and the tile form: Gemma's scale-1 scores make the
    // greedy tokens at 28K follow that rounding (measured); STRATA_G4_DEC_F32=1 keeps them in FP32
    static const bool round_h = std::getenv("STRATA_G4_DEC_F32") == nullptr;
    dec_kernel<D, G><<<dim3(n_kv, n_split), 128, 0, s>>>(q, kv, p0, n_head, window, kbase, split_len, scale, out, part_o,
                                                        part_ml, round_h, rep);
    check("g4_attn_dec");
    if (n_split > 1) {
        merge_kernel<<<dim3(G, n_kv), D, 0, s>>>(part_o, part_ml, G, n_split, G, n_head, out);
        check("g4_attn_dec_merge");
    }
}

// ---- head 512 prompts: one block (32 rows, 4 warps) computes the scores ONCE and both halves of the output (the
// generic form splits V's 512 dims over two blocks, each paying the whole q k^T).  Tiles of 16 keys: warp w scores
// rows (w & 1) * 16 .. over dims (w >> 1) * 256 .., the halves summed by warps 0 / 1, which keep the rows' online
// softmax and publish P and the rescale; warp w then accumulates rows (w & 1) * 16 .., output dims (w >> 1) * 256 ..
constexpr int P5_AK = 16, P5_LDQ = 512 + 8, P5_LDS = P5_AK + 4, P5_LDP = P5_AK + 8;
// DIRECT (the default for a layer without a ring; STRATA_G4_P5_DIRECT=0: staged): the K / V fragments read in place from the cache (L1) instead of
// staged in shared memory - 40 instead of 73 KB a block, two blocks an SM; the cache is zeroed with 16 spare cells, so
// a tile's keys past the last one are finite (masked to P = 0)
constexpr size_t p5_smem(bool direct) {
    return (size_t) 32 * P5_LDQ * 2 + (direct ? 0 : (size_t) 2 * P5_AK * P5_LDQ * 2) + (size_t) 4 * 16 * P5_LDS * 4 +
           (size_t) 32 * P5_LDP * 2 + (size_t) 64 * 4;
}
template <bool DIRECT>
__global__ void __launch_bounds__(128) attn512_kernel(const float* __restrict__ q, G4Kv kv, int p0, int T, int n_head,
                                                      int G, int window, int kbase, int split_len, float scale,
                                                      float* __restrict__ out, float* __restrict__ part_o,
                                                      float* __restrict__ part_ml) {
    using namespace nvcuda;
    constexpr int AD = 512, AR = 32;
    extern __shared__ __align__(32) unsigned char smem[];
    __half* sQ = (__half*) smem;
    __half* sK = sQ + AR * P5_LDQ;
    __half* sV = sK + P5_AK * P5_LDQ;
    float* sS = (float*) (DIRECT ? sK : sV + P5_AK * P5_LDQ);
    __half* sP = (__half*) (sS + 4 * 16 * P5_LDS);
    float* sCorr = (float*) (sP + 32 * P5_LDP);     // 32 rows' rescale, then (the end) their sums
    float* sL = sCorr + 32;
    const int kvh = blockIdx.y, split = blockIdx.z, n_split = gridDim.z;
    const int rows = T * G, r0 = blockIdx.x * AR;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, rt = w & 1, half = w >> 1;
    const int qld = n_head * AD;
    for (int i = tid; i < AR * (AD / 4); i += 128) {
        const int r = i / (AD / 4), d = (i % (AD / 4)) * 4, row = r0 + r;
        float4 f = make_float4(0.f, 0.f, 0.f, 0.f);
        if (row < rows) f = *(const float4*) (q + (size_t) (row / G) * qld + (size_t) (kvh * G + row % G) * AD + d);
        __half2* dst = (__half2*) (sQ + r * P5_LDQ + d);
        dst[0] = __floats2half2_rn(f.x * scale, f.y * scale);
        dst[1] = __floats2half2_rn(f.z * scale, f.w * scale);
    }
    const int last_row = min(r0 + AR, rows) - 1;
    const int kend = p0 + last_row / G + 1;
    const int klo = window > 0 ? max(0, p0 + r0 / G - window + 1) : 0;
    const int ks = kbase + split * split_len, ke = min(kend, ks + split_len);
    const int kstart = max(ks, ks + (klo - ks) / P5_AK * P5_AK);
    // the softmax rows (warps 0 / 1): lane pair (ri, hi) one row, 8 keys each
    const int ri = lane >> 1, hi = lane & 1, srow = r0 + rt * 16 + ri;
    const bool srow_ok = srow < rows;
    const int pos = srow_ok ? p0 + srow / G : -1, lo = window > 0 ? pos - window + 1 : 0;
    float m = -INFINITY, l = 0.0f;
    constexpr float kLazy = 8.0f;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fo[16];
#pragma unroll
    for (int n = 0; n < 16; ++n) wmma::fill_fragment(fo[n], 0.0f);
    for (int k0 = kstart; k0 < ke; k0 += P5_AK) {
        __syncthreads();
        // (DIRECT: the tile's K / V rows in the cache, ld = AD)
        const __half* gK = (const __half*) kv.k + ((size_t) kvh * kv.max_ctx + k0) * AD;
        const __half* gV = (const __half*) kv.v + ((size_t) kvh * kv.max_ctx + k0) * AD;
        if (!DIRECT)
        for (int i = tid; i < P5_AK * (AD / 8); i += 128) {
            const int r = i / (AD / 8), d = (i % (AD / 8)) * 8, key = k0 + r;
            uint4 kq = make_uint4(0, 0, 0, 0), vq = kq;
            if (key < ke) {
                const size_t cell = ((size_t) kvh * kv.max_ctx + g4_cell(kv, key)) * AD + d;
                kq = __ldg((const uint4*) (kv.k + cell));
                vq = __ldg((const uint4*) (kv.v + cell));
            }
            *(uint4*) (sK + r * P5_LDQ + d) = kq;
            *(uint4*) (sV + r * P5_LDQ + d) = vq;
        }
        __syncthreads();
        {   // scores: rows rt * 16 .., dims half * 256 ..
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
#pragma unroll 4
            for (int kk = 0; kk < 16; ++kk) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fq;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::load_matrix_sync(fq, sQ + rt * 16 * P5_LDQ + half * 256 + kk * 16, P5_LDQ);
                if (DIRECT) wmma::load_matrix_sync(fb, gK + half * 256 + kk * 16, AD);
                else wmma::load_matrix_sync(fb, sK + half * 256 + kk * 16, P5_LDQ);
                wmma::mma_sync(acc, fq, fb, acc);
            }
            wmma::store_matrix_sync(sS + w * 16 * P5_LDS, acc, P5_LDS, wmma::mem_row_major);
        }
        __syncthreads();
        if (w < 2) {   // the softmax of rows rt * 16 .. (both halves' scores summed)
            const float* Sa = sS + w * 16 * P5_LDS;
            const float* Sb = sS + (w + 2) * 16 * P5_LDS;
            constexpr int KH = P5_AK / 2;
            float sv[KH], mt = -INFINITY;
#pragma unroll
            for (int j = 0; j < KH; ++j) {
                const int c = hi * KH + j, key = k0 + c;
                sv[j] = srow_ok && key < ke && key <= pos && key >= lo ? Sa[ri * P5_LDS + c] + Sb[ri * P5_LDS + c] : -INFINITY;
                mt = fmaxf(mt, sv[j]);
            }
            mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 1));
            float corr = 1.0f;
            if (mt > m + kLazy) {
                corr = m == -INFINITY ? 0.0f : __expf(m - mt);
                l *= corr;
                m = mt;
            }
            float ps = 0.0f;
#pragma unroll
            for (int j = 0; j < KH; ++j) {
                const float pj = sv[j] == -INFINITY ? 0.0f : __expf(sv[j] - m);
                ps += pj;
                sP[(rt * 16 + ri) * P5_LDP + hi * KH + j] = __float2half(pj);
            }
            l += ps + __shfl_xor_sync(0xffffffffu, ps, 1);
            if (hi == 0) sCorr[rt * 16 + ri] = corr;
        }
        __syncthreads();
        {   // P . V: rows rt * 16 .., output dims half * 256 ..
            const float cr = sCorr[rt * 16 + (lane & 15)];
            if (__any_sync(0xffffffffu, cr != 1.0f)) {
                float* Sw = sS + w * 16 * P5_LDS;   // (the scores are read: scratch for the rescale)
#pragma unroll
                for (int n = 0; n < 16; ++n) {
                    wmma::store_matrix_sync(Sw, fo[n], P5_LDS, wmma::mem_row_major);
                    __syncwarp();
                    for (int i = lane; i < 256; i += 32) Sw[(i / 16) * P5_LDS + i % 16] *= sCorr[rt * 16 + i / 16];
                    __syncwarp();
                    wmma::load_matrix_sync(fo[n], Sw, P5_LDS, wmma::mem_row_major);
                    __syncwarp();
                }
            }
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fp;
            wmma::load_matrix_sync(fp, sP + rt * 16 * P5_LDP, P5_LDP);
#pragma unroll
            for (int n = 0; n < 16; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fv;
                if (DIRECT) wmma::load_matrix_sync(fv, gV + half * 256 + n * 16, AD);
                else wmma::load_matrix_sync(fv, sV + half * 256 + n * 16, P5_LDQ);
                wmma::mma_sync(fo[n], fp, fv, fo[n]);
            }
        }
    }
    __syncthreads();
    if (w < 2 && hi == 0) { sL[rt * 16 + ri] = l; sCorr[rt * 16 + ri] = m; }   // (sums, maxima)
    __syncthreads();
    float* Sw = sS + w * 16 * P5_LDS;
#pragma unroll
    for (int n = 0; n < 16; ++n) {
        wmma::store_matrix_sync(Sw, fo[n], P5_LDS, wmma::mem_row_major);
        __syncwarp();
        for (int i = lane; i < 256; i += 32) {
            const int rr = i / 16, c = i % 16, row = r0 + rt * 16 + rr;
            if (row >= rows) continue;
            const int tok = row / G, head = kvh * G + row % G, d = half * 256 + n * 16 + c;
            const float v = Sw[rr * P5_LDS + c];
            if (n_split == 1) {
                const float L = sL[rt * 16 + rr];
                out[(size_t) tok * qld + (size_t) head * AD + d] = L > 0.0f ? v / L : 0.0f;
            } else {
                part_o[(((size_t) split * gridDim.y + kvh) * rows + row) * AD + d] = v;
            }
        }
        __syncwarp();
    }
    if (n_split > 1 && tid < 32) {
        const int row = r0 + tid;
        if (row < rows) {
            part_ml[(((size_t) split * gridDim.y + kvh) * rows + row) * 2] = sCorr[tid];
            part_ml[(((size_t) split * gridDim.y + kvh) * rows + row) * 2 + 1] = sL[tid];
        }
    }
}

void launch_attn512(const float* q, const G4Kv& kv, int p0, int T, int n_head, int n_kv, int window, float scale,
                    float* out, float* scratch, cudaStream_t s) {
    // (31B 28K prompt 62.0 / 62.5 -> 53.5 / 53.5 s, E4B 9.36 -> 7.74 s; needles 3/3, greedy the same, cooled decode the same)
    static const bool direct_env = std::getenv("STRATA_G4_P5_DIRECT") == nullptr || std::atoi(std::getenv("STRATA_G4_P5_DIRECT")) != 0;
    const bool direct = direct_env && kv.ring == 0;
    const size_t sm = p5_smem(direct);
    static const bool attr = cudaFuncSetAttribute(attn512_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  (int) p5_smem(false)) == cudaSuccess &&
                             cudaFuncSetAttribute(attn512_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  (int) p5_smem(true)) == cudaSuccess;
    if (!attr) throw std::runtime_error("g4_attn512: shared memory attribute");
    const int G = n_head / n_kv, rows = T * G, tiles = (rows + 31) / 32;
    const int kbase = window > 0 ? std::max(0, p0 - window + 1) : 0;
    const int n_keys = p0 + T - kbase;
    const int want = 160 / std::max(1, tiles * n_kv);
    int n_split = std::max(1, std::min(want, (n_keys + 127) / 128));
    const int split_len = n_split == 1 ? n_keys : ((n_keys + n_split - 1) / n_split + P5_AK - 1) / P5_AK * P5_AK;
    n_split = (n_keys + split_len - 1) / split_len;
    float* part_o = scratch;
    float* part_ml = scratch + (size_t) n_split * rows * n_kv * 512;
    if (direct)
        attn512_kernel<true><<<dim3(tiles, n_kv, n_split), 128, sm, s>>>(q, kv, p0, T, n_head, G, window, kbase, split_len,
                                                                         scale, out, part_o, part_ml);
    else
        attn512_kernel<false><<<dim3(tiles, n_kv, n_split), 128, sm, s>>>(q, kv, p0, T, n_head, G, window, kbase, split_len,
                                                                          scale, out, part_o, part_ml);
    check("g4_attn512");
    if (n_split > 1) {
        merge_kernel<<<dim3(rows, n_kv), 512, 0, s>>>(part_o, part_ml, rows, n_split, G, n_head, out);
        check("g4_attn512_merge");
    }
}

template <int AD, int VD, int AW>
void launch_attn(const float* q, const G4Kv& kv, int p0, int T, int n_head, int n_kv, int window, float scale,
                 float* out, float* scratch, cudaStream_t s) {
    constexpr size_t sm = attn_smem<AD, VD, AW>();
    static const bool attr = cudaFuncSetAttribute(attn_kernel<AD, VD, AW>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  (int) sm) == cudaSuccess;
    if (!attr) throw std::runtime_error("g4_attn: shared memory attribute");
    const int G = n_head / n_kv, rows = T * G, AR = AW * 16, tiles = (rows + AR - 1) / AR;
    const int kbase = window > 0 ? std::max(0, p0 - window + 1) : 0;   // (the first token's first key)
    const int n_keys = p0 + T - kbase;
    const int want = 160 / std::max(1, tiles * n_kv * (AD / VD));
    int n_split = std::max(1, std::min(want, (n_keys + kSplitLen - 1) / kSplitLen));
    const int split_len = n_split == 1 ? n_keys : ((n_keys + n_split - 1) / n_split + AK - 1) / AK * AK;
    n_split = (n_keys + split_len - 1) / split_len;
    float* part_o = scratch;
    float* part_ml = scratch + (size_t) n_split * rows * n_kv * AD;
    attn_kernel<AD, VD, AW><<<dim3(tiles, n_kv, n_split * (AD / VD)), AW * 32, sm, s>>>(
        q, kv, p0, T, n_head, G, window, kbase, split_len, scale, out, part_o, part_ml);
    check("g4_attn");
    if (n_split > 1) {
        merge_kernel<<<dim3(rows, n_kv), AD, 0, s>>>(part_o, part_ml, rows, n_split, G, n_head, out);
        check("g4_attn_merge");
    }
}

}  // namespace

void g4_attn_prep(const float* q, const float* k, const float* v, int T, int n_head, int n_kv, int D,
                  const float* q_norm, const float* k_norm, float eps, float rope_base, const float* ff, int p0,
                  float* q_out, const G4Kv& kv, void* stream, int rope_mode, bool v_norm) {
    if (D != 128 && D != 256 && D != 512) throw std::invalid_argument("g4_attn_prep: head size 128, 256 or 512");
    const int heads = n_head + (kv.k ? 2 * n_kv : 0);
    prep_kernel<<<dim3(T, heads), D, 0, (cudaStream_t) stream>>>(q, k, v, n_head, n_kv, q_norm, k_norm, eps, rope_base,
                                                                 ff, p0, q_out, kv, rope_mode, v_norm ? 1 : 0);
    check("g4_attn_prep");
}

int64_t g4_attn_scratch_floats(int T, int n_head, int D, int max_ctx) {
    const int64_t splits = (max_ctx + kSplitLen - 1) / kSplitLen + 1;
    return splits * (int64_t) T * n_head * (D + 2);
}

void g4_attn(const float* q, const G4Kv& kv, int p0, int T, int n_head, int n_kv, int D, int window, float scale,
             float* out, float* scratch, void* stream) {
    if (n_head % n_kv != 0 || n_head / n_kv > 16) throw std::invalid_argument("g4_attn: geometry");
    cudaStream_t s = (cudaStream_t) stream;
    static const bool dec = std::getenv("STRATA_G4_DEC") == nullptr || std::atoi(std::getenv("STRATA_G4_DEC")) != 0;
    // eight query heads a kv head at head 512 (Gemma 4 31B's full layers): two blocks of four a kv head (one block of
    // eight would hold 256 registers of q and output a thread); STRATA_G4_DEC8=0: the tile form as before
    static const bool dec8 = std::getenv("STRATA_G4_DEC8") == nullptr || std::atoi(std::getenv("STRATA_G4_DEC8")) != 0;
    if (T == 1 && dec && dec8 && D == 512 && n_head / n_kv == 8) {
        static bool said = false;
        if (!said) { std::fprintf(stderr, "gemma4: head-512 decode, 8 query heads a K/V head: two decode blocks a K/V head\n"); said = true; }
        return launch_dec<512, 4>(q, kv, p0, n_head, n_kv, window, scale, out, scratch, s, 2);
    }
    if (T == 1 && dec && D == 128 && n_head / n_kv == 16)
        return launch_dec<128, 16>(q, kv, p0, n_head, n_kv, window, scale, out, scratch, s);
    if (T == 1 && dec && (n_head / n_kv == 4 || n_head / n_kv == 2)) {   // one token: the split-K SIMT form
        if (n_head / n_kv == 4) {                                         // (the tile form pads the rows to 16+)
            if (D == 256) return launch_dec<256, 4>(q, kv, p0, n_head, n_kv, window, scale, out, scratch, s);
            if (D == 512) return launch_dec<512, 4>(q, kv, p0, n_head, n_kv, window, scale, out, scratch, s);
        } else {
            if (D == 256) return launch_dec<256, 2>(q, kv, p0, n_head, n_kv, window, scale, out, scratch, s);
            if (D == 512) return launch_dec<512, 2>(q, kv, p0, n_head, n_kv, window, scale, out, scratch, s);
        }
    }
    if (D == 128) launch_attn<128, 128, 4>(q, kv, p0, T, n_head, n_kv, window, scale, out, scratch, s);
    else if (D == 256) launch_attn<256, 256, 4>(q, kv, p0, T, n_head, n_kv, window, scale, out, scratch, s);
    else if (D == 512) {
        static const bool p5 = std::getenv("STRATA_G4_ATTN512") == nullptr || std::atoi(std::getenv("STRATA_G4_ATTN512")) != 0;
        if (p5 && T > 1) launch_attn512(q, kv, p0, T, n_head, n_kv, window, scale, out, scratch, s);
        else launch_attn<512, 256, 2>(q, kv, p0, T, n_head, n_kv, window, scale, out, scratch, s);
    }
    else throw std::invalid_argument("g4_attn: head size 128, 256 or 512");
}

void g4_gelu_mul(const float* g, const float* u, float* out, int64_t n, void* stream) {
    gelu_mul_kernel<<<blocks(n, 256), 256, 0, (cudaStream_t) stream>>>(g, u, out, n);
    check("g4_gelu_mul");
}
void g4_ple(const float* proj, const float* tok, const float* w, float eps, int n_embd, int n_layer, int T, float* out,
            void* stream) {
    ple_kernel<<<dim3(T, n_layer), 256, 0, (cudaStream_t) stream>>>(proj, tok, w, eps, 1.0f / sqrtf((float) n_embd),
                                                                    n_layer, out);
    check("g4_ple");
}
void g4_gelu_ple(float* g, const float* ple, int l, int n_layer, int T, void* stream) {
    gelu_ple_kernel<<<blocks((int64_t) T * 256, 256), 256, 0, (cudaStream_t) stream>>>(g, ple, l, n_layer, T);
    check("g4_gelu_ple");
}
void g4_add_scale(float* x, const float* y, float s, int64_t n, void* stream) {
    add_scale_kernel<<<blocks(n, 256), 256, 0, (cudaStream_t) stream>>>(x, y, s, n);
    check("g4_add_scale");
}
void g4_sigmoid_mul(float* x, const float* g, int64_t n, void* stream) {
    sigmoid_mul_kernel<<<blocks(n, 256), 256, 0, (cudaStream_t) stream>>>(x, g, n);
    check("g4_sigmoid_mul");
}
void g4_scale(float* x, float s, int64_t n, void* stream) {
    scale_kernel<<<blocks(n, 256), 256, 0, (cudaStream_t) stream>>>(x, s, n);
    check("g4_scale");
}
void g4_softcap(float* x, int64_t ld, int n, int rows, float cap, void* stream) {
    softcap_kernel<<<dim3(blocks(n, 256), rows), 256, 0, (cudaStream_t) stream>>>(x, ld, n, cap);
    check("g4_softcap");
}

}  // namespace strata::kernels
