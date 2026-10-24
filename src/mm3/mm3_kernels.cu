// src/mm3/mm3_kernels.cu - maya-mm3's kernels around the EXL3 matrices (the interface: mm3_kernels.cuh).
#include "mm3_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>

namespace mm3::k {
namespace {

__device__ __forceinline__ float bf2f(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ uint16_t f2h(float f) { return __half_as_ushort(__float2half_rn(f)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// the sum of v over the block (blockDim.x a multiple of 32, at most 1024), every thread gets it
__device__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float t = l < nw ? red[l] : 0.0f;
    if (w == 0) t = warp_sum(t);
    if (threadIdx.x == 0) red[32] = t;
    __syncthreads();
    return red[32];
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) std::fprintf(stderr, "mm3 %s: %s\n", what, cudaGetErrorString(e));
}

// ------------------------------------------------------------------ RMSNorm (Gemma: the weight is 1 + w, baked)
__global__ void rms_kernel(const float* x, const float* w1, int D, float eps, float* y, uint16_t* y16) {
    __shared__ float red[33];
    const float* xr = x + (size_t) blockIdx.x * D;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) ss = fmaf(xr[i], xr[i], ss);
    const float r = rsqrtf(block_sum(ss, red) / (float) D + eps);
    float* yr = y + (size_t) blockIdx.x * D;
    uint16_t* hr = y16 ? y16 + (size_t) blockIdx.x * D : nullptr;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float v = xr[i] * r * w1[i];
        yr[i] = v;
        if (hr) hr[i] = f2h(v);
    }
}

// ------------------------------------------------------------------ one head: RMSNorm over hd then NeoX RoPE
// (blockDim.x = hd = 128; the rotary on dims < n_rot: out[d] = x[d] c - x[d + n_rot/2] s, out[d + n_rot/2] =
//  x[d + n_rot/2] c + x[d] s for d < n_rot / 2, as transformers' rotate_half)
__device__ float norm_rope(float x, const float* w1, int pos, int n_rot, const float* inv, float eps, float* sh,
                           float* red) {
    const int d = threadIdx.x, hd = blockDim.x;
    const float r = rsqrtf(block_sum(x * x, red) / (float) hd + eps);
    const float v = x * r * w1[d];
    sh[d] = v;
    __syncthreads();
    float o = v;
    const int half = n_rot / 2;
    if (d < n_rot) {
        const int i = d < half ? d : d - half;
        const float a = (float) pos * inv[i];
        const float c = cosf(a), s = sinf(a);
        o = d < half ? v * c - sh[d + half] * s : v * c + sh[d - half] * s;
    }
    __syncthreads();
    return o;
}

__global__ void qk_kernel(float* q, const float* k, const float* v, const float* qn1, const float* kn1,
                          const int* pos, int Hq, int Hkv, int n_rot, const float* inv, float eps, uint16_t* kc,
                          uint16_t* vc) {
    __shared__ float red[33], sh[128];
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x, hd = blockDim.x;
    const int p = pos[t];
    if (h < Hq) {
        float* x = q + ((size_t) t * Hq + h) * hd;
        x[d] = norm_rope(x[d], qn1, p, n_rot, inv, eps, sh, red);
    } else if (h < Hq + Hkv) {
        const int kh = h - Hq;
        const float o = norm_rope(k[((size_t) t * Hkv + kh) * hd + d], kn1, p, n_rot, inv, eps, sh, red);
        kc[((size_t) p * Hkv + kh) * hd + d] = f2bf(o);
    } else {
        const int vh = h - Hq - Hkv;
        vc[((size_t) p * Hkv + vh) * hd + d] = f2bf(v[((size_t) t * Hkv + vh) * hd + d]);
    }
}

__global__ void index_kernel(float* iq, const float* ik, const float* qn1, const float* kn1, const int* pos, int Hi,
                             int n_rot, const float* inv, float eps, uint16_t* ic) {
    __shared__ float red[33], sh[128];
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x, D = blockDim.x;
    const int p = pos[t];
    if (h < Hi) {
        float* x = iq + ((size_t) t * Hi + h) * D;
        x[d] = bf2f(f2bf(norm_rope(x[d], qn1, p, n_rot, inv, eps, sh, red)));
    } else {
        ic[(size_t) p * D + d] = f2bf(norm_rope(ik[(size_t) t * D + d], kn1, p, n_rot, inv, eps, sh, red));
    }
}

// ------------------------------------------------------------------ MSA: block scores
// A block of 256 threads takes 4 queries x Hi (4) heads = 16 vectors and walks the key blocks up to the last query's:
// thread (v = tid / 16, s = tid % 16) the keys s + 16 i (i < 8) of each key block, their max, then the max over s.
constexpr int kBsQ = 4, kBsV = 16, kKP = 136;   // (key rows padded to 136 halves in shared memory)
__global__ void __launch_bounds__(256) block_scores_kernel(const float* iq, const uint16_t* ic, const int* pos, int T,
                                                           int Hi, int nblk, int blk, float* bs) {
    __shared__ float qv[kBsV][129];
    __shared__ __align__(16) uint16_t kt[128][kKP];
    const int t0 = blockIdx.x * kBsQ, tid = threadIdx.x;
    const int nq = min(kBsQ, T - t0);
    for (int i = tid; i < kBsV * 128; i += 256) {
        const int v = i / 128, d = i % 128, qi = v / Hi, h = v % Hi;
        qv[v][d] = qi < nq ? iq[((size_t) (t0 + qi) * Hi + h) * 128 + d] : 0.0f;
    }
    int pmax = 0;
    for (int qi = 0; qi < nq; ++qi) pmax = max(pmax, pos[t0 + qi]);
    const int v = tid >> 4, s = tid & 15, qi = v / Hi, h = v % Hi;
    const int myp = qi < nq ? pos[t0 + qi] : -1;
    const int last = pmax / blk;
    for (int b = 0; b <= last; ++b) {
        __syncthreads();
        for (int i = tid; i < 128 * 16; i += 256) {   // 128 keys x 16 chunks of 8 halves
            const int j = i >> 4, c = i & 15;
            const int kp = b * blk + j;
            uint4 w = make_uint4(0, 0, 0, 0);
            if (kp <= pmax) w = *(const uint4*) (ic + (size_t) kp * 128 + c * 8);
            *(uint4*) &kt[j][c * 8] = w;
        }
        __syncthreads();
        float best = -INFINITY;
        if (qi < nq) {
#pragma unroll 1
            for (int i = 0; i < 8; ++i) {
                const int j = s + 16 * i, kp = b * blk + j;
                if (kp > myp) continue;
                float dot = 0.0f;
#pragma unroll 8
                for (int d = 0; d < 128; ++d) dot = fmaf(qv[v][d], bf2f(kt[j][d]), dot);
                best = fmaxf(best, dot);
            }
        }
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) best = fmaxf(best, __shfl_xor_sync(0xffffffffu, best, o));
        if (s == 0 && qi < nq) bs[((size_t) h * T + t0 + qi) * nblk + b] = best;
    }
    // (blocks past the tile's last query's stay unwritten: select_blocks reads only blocks <= its query's)
}

// ------------------------------------------------------------------ MSA: the kept blocks
// one warp per (head, query): its own block, then topk - 1 rounds of the best block after the previous pick in the
// order (score desc, block asc) - the set torch.topk keeps, ties to the lower block
__device__ __forceinline__ bool better(float a, int ia, float b, int ib) { return a > b || (a == b && ia < ib); }

__global__ void select_kernel(const float* bs, const int* pos, int T, int Hi, int nblk, int blk, int topk, int* kept) {
    const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (wid >= T * Hi) return;
    const int h = wid / T, t = wid % T;
    const int qb = pos[t] / blk;
    int* out = kept + ((size_t) h * T + t) * topk;
    const float* row = bs + ((size_t) h * T + t) * nblk;
    if (qb + 1 <= topk) {
        for (int i = lane; i < topk; i += 32) out[i] = i <= qb ? i : -1;
        return;
    }
    if (lane == 0) out[0] = qb;
    float ps = INFINITY;
    int pi = -1;
    for (int r = 1; r < topk; ++r) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int b = lane; b < qb; b += 32) {
            const float sv = row[b];
            if (better(ps, pi, sv, b) && better(sv, b, bv, bi)) {
                bv = sv;
                bi = b;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (better(ov, oi, bv, bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) out[r] = bi;
        ps = bv;
        pi = bi;
    }
}

// ------------------------------------------------------------------ MSA: attention over the kept blocks
// one block of 128 threads per (query, KV group): its 16 query heads over 64-key tiles of the kept blocks, online
// softmax in F32
constexpr int kG = 16, kTile = 64, kRow = 130;   // (K / V rows padded to 130 halves)
__global__ void __launch_bounds__(128) sparse_attn_kernel(const float* q, const uint16_t* kc, const uint16_t* vc,
                                                          const int* kept, const int* pos, int T, int Hq, int Hkv,
                                                          int topk, int blk, float scale, float* out) {
    __shared__ float qs[kG][128];
    __shared__ __align__(4) uint16_t ks[kTile][kRow], vs[kTile][kRow];
    __shared__ float sc[kG][kTile + 1];
    __shared__ float mrow[kG], lrow[kG], arow[kG];
    const int t = blockIdx.x, g = blockIdx.y, tid = threadIdx.x;
    const int p = pos[t];
    for (int i = tid; i < kG * 128; i += 128) {
        const int h = i / 128, d = i % 128;
        qs[h][d] = q[((size_t) t * Hq + g * kG + h) * 128 + d] * scale;
    }
    if (tid < kG) {
        mrow[tid] = -INFINITY;
        lrow[tid] = 0.0f;
    }
    float acc[kG];
#pragma unroll
    for (int h = 0; h < kG; ++h) acc[h] = 0.0f;
    const int* kb = kept + ((size_t) g * T + t) * topk;
    for (int bi = 0; bi < topk; ++bi) {
        const int b = kb[bi];
        if (b < 0) continue;
        for (int sub = 0; sub < blk / kTile; ++sub) {
            const int k0 = b * blk + sub * kTile;
            if (k0 > p) break;
            __syncthreads();
            for (int i = tid; i < kTile * 64; i += 128) {   // 64 keys x 64 pairs of halves, K and V
                const int j = i >> 6, c = i & 63;
                const int kp = k0 + j;
                uint32_t kw = 0, vw = 0;
                if (kp <= p) {
                    kw = *(const uint32_t*) (kc + ((size_t) kp * Hkv + g) * 128 + c * 2);
                    vw = *(const uint32_t*) (vc + ((size_t) kp * Hkv + g) * 128 + c * 2);
                }
                *(uint32_t*) &ks[j][c * 2] = kw;
                *(uint32_t*) &vs[j][c * 2] = vw;
            }
            __syncthreads();
            {   // scores: thread (j = tid % 64) for the heads (tid / 64) * 8 .. + 8
                const int j = tid & 63, h0 = (tid >> 6) * 8;
                const bool ok = k0 + j <= p;
                float s8[8];
#pragma unroll
                for (int h = 0; h < 8; ++h) s8[h] = 0.0f;
                if (ok) {
#pragma unroll 4
                    for (int d = 0; d < 128; ++d) {
                        const float kv = bf2f(ks[j][d]);
#pragma unroll
                        for (int h = 0; h < 8; ++h) s8[h] = fmaf(qs[h0 + h][d], kv, s8[h]);
                    }
                }
#pragma unroll
                for (int h = 0; h < 8; ++h) sc[h0 + h][j] = ok ? s8[h] : -INFINITY;
            }
            __syncthreads();
            if (tid < kG) {   // the row's new max, the old sum's factor, the tile's probabilities
                const int h = tid;
                float m = mrow[h];
                for (int j = 0; j < kTile; ++j) m = fmaxf(m, sc[h][j]);
                const float a = mrow[h] == -INFINITY ? 0.0f : expf(mrow[h] - m);
                float l = lrow[h] * a;
                for (int j = 0; j < kTile; ++j) {
                    const float e = sc[h][j] == -INFINITY ? 0.0f : expf(sc[h][j] - m);
                    sc[h][j] = e;
                    l += e;
                }
                mrow[h] = m;
                lrow[h] = l;
                arow[h] = a;
            }
            __syncthreads();
            {   // the values: thread d over the 16 heads
                const int d = tid;
#pragma unroll
                for (int h = 0; h < kG; ++h) {
                    float s = acc[h] * arow[h];
                    for (int j = 0; j < kTile; ++j) s = fmaf(sc[h][j], bf2f(vs[j][d]), s);
                    acc[h] = s;
                }
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < kG; ++h)
        out[((size_t) t * Hq + g * kG + h) * 128 + tid] = lrow[h] > 0.0f ? acc[h] / lrow[h] : 0.0f;
}

// ------------------------------------------------------------------ the router (one warp a token, E <= 256)
__global__ void route_kernel(const float* logits, const float* bias, int T, int E, int K, int* top, float* w) {
    const int t = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (t >= T) return;
    float sg[8], ch[8];
    const int per = (E + 31) / 32;
    for (int i = 0; i < per; ++i) {
        const int e = lane + 32 * i;
        sg[i] = e < E ? 1.0f / (1.0f + expf(-logits[(size_t) t * E + e])) : 0.0f;
        ch[i] = e < E ? sg[i] + bias[e] : -INFINITY;
    }
    float ps = INFINITY;
    int pi = -1;
    float sum = 0.0f;
    for (int r = 0; r < K; ++r) {
        float bv = -INFINITY, bsg = 0.0f;
        int bi = 0x7fffffff;
        for (int i = 0; i < per; ++i) {
            const int e = lane + 32 * i;
            if (e < E && better(ps, pi, ch[i], e) && better(ch[i], e, bv, bi)) {
                bv = ch[i];
                bi = e;
                bsg = sg[i];
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            const float os = __shfl_xor_sync(0xffffffffu, bsg, o);
            if (better(ov, oi, bv, bi)) {
                bv = ov;
                bi = oi;
                bsg = os;
            }
        }
        if (lane == 0) {
            top[(size_t) t * K + r] = bi;
            w[(size_t) t * K + r] = bsg;
        }
        sum += bsg;
        ps = bv;
        pi = bi;
    }
    __syncwarp();
    if (lane < K) w[(size_t) t * K + lane] /= sum;
}

__device__ __forceinline__ float swoai(float g, float u, float alpha, float limit) {
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
    return g / (1.0f + expf(-alpha * g)) * (u + 1.0f);
}

__global__ void swigluoai_kernel(const float* g, const float* u, size_t n, float alpha, float limit, float* hf,
                                 uint16_t* h16) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const float v = swoai(g[i], u[i], alpha, limit);
        if (hf) hf[i] = v;
        if (h16) h16[i] = f2h(v);
    }
}

__global__ void swigluoai_pairs_kernel(const float* gu, int ff, float alpha, float limit, uint16_t* h16) {
    const float* r = gu + (size_t) blockIdx.x * 2 * ff;
    for (int i = threadIdx.x; i < ff; i += blockDim.x)
        h16[(size_t) blockIdx.x * ff + i] = f2h(swoai(r[i], r[ff + i], alpha, limit));
}

__global__ void to_f16_kernel(const float* x, int cols, int ld, uint16_t* y, int ldo) {
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        y[(size_t) blockIdx.x * ldo + i] = f2h(x[(size_t) blockIdx.x * ld + i]);
}

__global__ void combine_kernel(const float* res, const float* y, const int* inv, const float* w, const float* sh, int K,
                               int D, float scale, float* out) {
    const int t = blockIdx.x;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float a = 0.0f;
        for (int i = 0; i < K; ++i) a = fmaf(w[(size_t) t * K + i], y[(size_t) inv[t * K + i] * D + d], a);
        float m = a * scale;
        if (sh) m += sh[(size_t) t * D + d];
        out[(size_t) t * D + d] = res[(size_t) t * D + d] + m;
    }
}

__global__ void add_kernel(float* x, const float* a, size_t n) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        x[i] += a[i];
}

unsigned grid_for(size_t n) { return (unsigned) std::min<size_t>(4096, (n + 255) / 256); }

}  // namespace

void rms(const float* x, const float* w1, int rows, int D, float eps, float* y, uint16_t* y16, cudaStream_t s) {
    if (rows <= 0) return;
    rms_kernel<<<rows, 256, 0, s>>>(x, w1, D, eps, y, y16);
    check("rms");
}

void rope_inv_freq(float theta, int n_rot, float* inv) {
    for (int i = 0; i < n_rot / 2; ++i)
        inv[i] = (float) (1.0 / std::pow((double) theta, (double) (float) (2 * i) / (double) n_rot));
}

void qk_norm_rope(float* q, const float* k, const float* v, const float* qn1, const float* kn1, const int* pos, int T,
                  int Hq, int Hkv, int hd, int n_rot, const float* inv, float eps, uint16_t* kc, uint16_t* vc,
                  cudaStream_t s) {
    if (T <= 0) return;
    qk_kernel<<<dim3(T, Hq + 2 * Hkv), hd, 0, s>>>(q, k, v, qn1, kn1, pos, Hq, Hkv, n_rot, inv, eps, kc, vc);
    check("qk_norm_rope");
}

void index_norm_rope(float* iq, const float* ik, const float* qn1, const float* kn1, const int* pos, int T, int Hi,
                     int D, int n_rot, const float* inv, float eps, uint16_t* ic, cudaStream_t s) {
    if (T <= 0) return;
    index_kernel<<<dim3(T, Hi + 1), D, 0, s>>>(iq, ik, qn1, kn1, pos, Hi, n_rot, inv, eps, ic);
    check("index_norm_rope");
}

void block_scores(const float* iq, const uint16_t* ic, const int* pos, int T, int Hi, int nblk, int blk, float* bs,
                  cudaStream_t s) {
    if (T <= 0) return;
    block_scores_kernel<<<(T + kBsQ - 1) / kBsQ, 256, 0, s>>>(iq, ic, pos, T, Hi, nblk, blk, bs);
    check("block_scores");
}

void select_blocks(const float* bs, const int* pos, int T, int Hi, int nblk, int blk, int topk, int* kept,
                   cudaStream_t s) {
    if (T <= 0) return;
    const int warps = T * Hi;
    select_kernel<<<(warps + 7) / 8, 256, 0, s>>>(bs, pos, T, Hi, nblk, blk, topk, kept);
    check("select_blocks");
}

void sparse_attn(const float* q, const uint16_t* kc, const uint16_t* vc, const int* kept, const int* pos, int T,
                 int Hq, int Hkv, int topk, int blk, float scale, float* out, cudaStream_t s) {
    if (T <= 0) return;
    sparse_attn_kernel<<<dim3(T, Hkv), 128, 0, s>>>(q, kc, vc, kept, pos, T, Hq, Hkv, topk, blk, scale, out);
    check("sparse_attn");
}

void route(const float* logits, const float* bias, int T, int E, int K, int* top, float* w, cudaStream_t s) {
    if (T <= 0) return;
    route_kernel<<<(T + 7) / 8, 256, 0, s>>>(logits, bias, T, E, K, top, w);
    check("route");
}

void swigluoai(const float* g, const float* u, size_t n, float alpha, float limit, float* hf, uint16_t* h16,
               cudaStream_t s) {
    if (n == 0) return;
    swigluoai_kernel<<<grid_for(n), 256, 0, s>>>(g, u, n, alpha, limit, hf, h16);
    check("swigluoai");
}

void swigluoai_pairs(const float* gu, int rows, int ff, float alpha, float limit, uint16_t* h16, cudaStream_t s) {
    if (rows <= 0) return;
    swigluoai_pairs_kernel<<<rows, 256, 0, s>>>(gu, ff, alpha, limit, h16);
    check("swigluoai_pairs");
}

void to_f16(const float* x, int rows, int cols, int ld, uint16_t* y, int ldo, cudaStream_t s) {
    if (rows <= 0) return;
    to_f16_kernel<<<rows, 256, 0, s>>>(x, cols, ld, y, ldo);
    check("to_f16");
}

void moe_combine(const float* res, const float* y, const int* inv, const float* w, const float* sh, int T, int K,
                 int D, float scale, float* out, cudaStream_t s) {
    if (T <= 0) return;
    combine_kernel<<<T, 256, 0, s>>>(res, y, inv, w, sh, K, D, scale, out);
    check("moe_combine");
}

void add(float* x, const float* a, size_t n, cudaStream_t s) {
    if (n == 0) return;
    add_kernel<<<grid_for(n), 256, 0, s>>>(x, a, n);
    check("add");
}

}  // namespace mm3::k
