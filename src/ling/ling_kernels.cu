// src/ling/ling_kernels.cu - Ling-3.2-air (bailingmoe3) CUDA kernels.
#include "ling_kernels.cuh"

#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>

namespace ling::k {
namespace {

__device__ __forceinline__ float warp_sum_f(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ double warp_sum_d(double v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max_f(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ __forceinline__ float sigmoid_exact(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ void q8_store_warp(float v, size_t at, int8_t* __restrict__ q, float* __restrict__ d) {
    const float am = warp_max_f(fabsf(v));
    const float sc = am / 127.0f;
    q[at] = (int8_t) (am > 0.0f ? __float2int_rn(v / sc) : 0);
    if ((threadIdx.x & 31) == 0) d[at >> 5] = sc;
}

__device__ __forceinline__ void rope_pair_d(float& a, float& b, int pos, int i, int n_rot, float base) {
    const double inv = pow((double) base, -2.0 * (double) i / (double) n_rot);
    const double theta = (double) pos * inv;
    double sd, cd;
    sincos(theta, &sd, &cd);
    const float c = (float) cd, s = (float) sd;
    const float x0 = a, x1 = b;
    a = x0 * c - x1 * s;
    b = x0 * s + x1 * c;
}

// ---- KDA prep (grid: (n_head, 4), block: 128) ----
__global__ void __launch_bounds__(128)
    kda_prep_full_kernel(const float* __restrict__ q_in, const float* __restrict__ k_in,
                         const float* __restrict__ v_in, const float* __restrict__ f_in,
                         const float* __restrict__ conv_q, const float* __restrict__ conv_k,
                         const float* __restrict__ conv_v, float* __restrict__ conv_state,
                         const float* __restrict__ ssm_a, float lb, float eps, int n_head, int head_dim, int d_conv,
                         float* __restrict__ q_out, float* __restrict__ k_out, float* __restrict__ v_out,
                         float* __restrict__ g1_out) {
    const int h = blockIdx.x;
    const int role = blockIdx.y;
    const int tid = threadIdx.x;
    const int d_inner = n_head * head_dim;
    if (tid >= head_dim) return;
    const int c = h * head_dim + tid;

    if (role == 3) {
        const float v = f_in[c] * ssm_a[h];
        g1_out[c] = lb * sigmoid_exact(v);
        return;
    }

    const float* in = role == 0 ? q_in : (role == 1 ? k_in : v_in);
    const float* cw = role == 0 ? conv_q : (role == 1 ? conv_k : conv_v);
    float* out = role == 0 ? q_out : (role == 1 ? k_out : v_out);
    const int hist = d_conv - 1;
    float* st = conv_state + (size_t) role * d_inner * hist + (size_t) hist * c;
    const float* w = cw + (size_t) d_conv * c;

    const float xp = in[c];
    float acc = 0.0f;
    for (int tap = 0; tap < hist; ++tap) acc += w[tap] * st[tap];
    acc += w[hist] * xp;
    float y = acc * sigmoid_exact(acc);
    for (int j = 0; j + 1 < hist; ++j) st[j] = st[j + 1];
    st[hist - 1] = xp;

    if (role < 2) {
        __shared__ double ssum[4];
        const double wv = warp_sum_d((double) y * (double) y);
        if ((tid & 31) == 0) ssum[tid >> 5] = wv;
        __syncthreads();
        const double total = ssum[0] + ssum[1] + ssum[2] + ssum[3];
        y *= 1.0f / fmaxf(sqrtf((float) total), eps);
    }
    out[c] = y;
}

// ---- KDA recurrence + out gate + Q8 (grid: n_head, block: 256, HD == 128) ----
__global__ void __launch_bounds__(256)
    kda_rec_q8_kernel(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                      const float* __restrict__ g1, const float* __restrict__ beta_raw, float* __restrict__ state,
                      const float* __restrict__ g2, const float* __restrict__ norm_w, float eps,
                      int8_t* __restrict__ xq_out, float* __restrict__ xd_out, float* __restrict__ raw_o_opt) {
    constexpr int HD = 128;
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    const int j = t & (HD - 1);
    const int half = t >> 7;

    __shared__ float sq[HD], sk[HD], sdec[HD], sdel[HD];
    __shared__ float sacc[2][HD];
    __shared__ float sred[4];

    if (t < HD) {
        sq[t] = q[h * HD + t];
        sk[t] = k[h * HD + t];
        sdec[t] = expf(g1[h * HD + t]);
    }
    const float b = sigmoid_exact(beta_raw[h]);
    __syncthreads();

    float* Sh = state + (size_t) h * HD * HD;
    float reg[64];
    float A = 0.0f;
#pragma unroll
    for (int ii = 0; ii < 64; ++ii) {
        const int i = half * 64 + ii;
        const float s = Sh[i * HD + j] * sdec[i];
        reg[ii] = s;
        A += s * sk[i];
    }
    sacc[half][j] = A;
    __syncthreads();

    if (half == 0) sdel[j] = b * (v[h * HD + j] - (sacc[0][j] + sacc[1][j]));
    __syncthreads();

    const float dl = sdel[j];
    float O = 0.0f;
#pragma unroll
    for (int ii = 0; ii < 64; ++ii) {
        const int i = half * 64 + ii;
        const float s2 = reg[ii] + sk[i] * dl;
        Sh[i * HD + j] = s2;
        O += s2 * sq[i];
    }
    sacc[half][j] = O;
    __syncthreads();

    if (half == 0) {
        const float o = (sacc[0][j] + sacc[1][j]) * rsqrtf((float) HD);
        if (raw_o_opt != nullptr) raw_o_opt[h * HD + j] = o;
        const float wsum = warp_sum_f(o * o);
        if ((j & 31) == 0) sred[j >> 5] = wsum;
        __syncthreads();
        const float ss = sred[0] + sred[1] + sred[2] + sred[3];
        const float inv = rsqrtf(ss / (float) HD + eps);
        const float gv = o * inv * norm_w[j] * sigmoid_exact(g2[h * HD + j]);
        q8_store_warp(gv, (size_t) h * HD + j, xq_out, xd_out);
    }
}

// ---- Grouped sigmoid router (grid: T, block: 512) ----
__device__ __forceinline__ bool r_better(float av, int ai, float bv, int bi) {
    return av > bv || (av == bv && ai < bi);
}

__global__ void __launch_bounds__(512)
    router_grouped_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int n_exp, int n_used,
                          int n_group, int n_group_used, int w_norm, float w_scale, int* __restrict__ ids_out,
                          float* __restrict__ w_out) {
    __shared__ float sp[512], sv[512];
    __shared__ int si[512];
    __shared__ float gscore[16];
    __shared__ int gkeep[16];

    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const float* lr = logits + (size_t) t * n_exp;

    if (tid < n_exp) {
        const float p = sigmoid_exact(lr[tid]);
        sp[tid] = p;
        sv[tid] = p + (bias != nullptr ? bias[tid] : 0.0f);
        si[tid] = tid;
    } else {
        sp[tid] = 0.0f;
        sv[tid] = -INFINITY;
        si[tid] = 0x7fffffff;
    }
    __syncthreads();

    if (n_group > 1) {
        const int per = n_exp / n_group;
        if (tid < n_group) {
            float b1 = -INFINITY, b2 = -INFINITY;
            const int e0 = tid * per, e1 = e0 + per;
            for (int e = e0; e < e1; ++e) {
                const float v = sv[e];
                if (v > b1) {
                    b2 = b1;
                    b1 = v;
                } else if (v > b2) {
                    b2 = v;
                }
            }
            gscore[tid] = b1 + b2;
            gkeep[tid] = 0;
        }
        __syncthreads();
        if (tid == 0) {
            for (int k = 0; k < n_group_used; ++k) {
                int best = -1;
                float best_v = -INFINITY;
                for (int g = 0; g < n_group; ++g) {
                    if (gkeep[g]) continue;
                    if (best < 0 || r_better(gscore[g], g, best_v, best)) {
                        best = g;
                        best_v = gscore[g];
                    }
                }
                if (best >= 0) gkeep[best] = 1;
            }
        }
        __syncthreads();
        if (tid < n_exp && !gkeep[tid / per]) sv[tid] = -INFINITY;
        __syncthreads();
    }

    // 512-element bitonic sort descending by (sv, -si)
    for (int k = 2; k <= 512; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int ixj = tid ^ j;
            if (ixj > tid) {
                const float a = sv[tid], b = sv[ixj];
                const int ia = si[tid], ib = si[ixj];
                const bool asc = (tid & k) != 0;   // false: descending (better first)
                const bool b_better = r_better(b, ib, a, ia);
                if (asc ? !b_better : b_better) {
                    sv[tid] = b;
                    sv[ixj] = a;
                    si[tid] = ib;
                    si[ixj] = ia;
                }
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        float sum = 0.0f;
        int* idr = ids_out + (size_t) t * n_used;
        float* wr = w_out + (size_t) t * n_used;
        for (int j = 0; j < n_used; ++j) {
            const int e = si[j];
            idr[j] = e;
            const float pj = sp[e];
            wr[j] = pj;
            sum += pj;
        }
        const float denom = w_norm ? fmaxf(sum, 6.103515625e-5f) : 1.0f;
        const float mult = (w_scale != 0.0f && w_scale != 1.0f) ? (w_scale / denom) : (1.0f / denom);
        if (w_norm || (w_scale != 0.0f && w_scale != 1.0f)) {
            for (int j = 0; j < n_used; ++j) wr[j] *= mult;
        }
    }
}

// ---- Split KV cache store (grid: T, block: 256) ----
__global__ void __launch_bounds__(256)
    kv_store_split_kernel(const float* __restrict__ kv_in, int kv_lora, int n_rot, const float* __restrict__ w,
                          float eps, const int* __restrict__ pos_dev, int p0, float base, void* __restrict__ lat,
                          int lat_q8, uint16_t* __restrict__ k_pe) {
    __shared__ float sred[8];
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int row = kv_lora + n_rot;
    const float* x = kv_in + (size_t) t * row;
    const int p = pos_dev != nullptr ? pos_dev[t] : (p0 + t);

    float ss = 0.0f;
    for (int i = tid; i < kv_lora; i += blockDim.x) ss += x[i] * x[i];
    ss = warp_sum_f(ss);
    if ((tid & 31) == 0) sred[tid >> 5] = ss;
    __syncthreads();
    if (tid < 32) {
        float v = tid < 8 ? sred[tid] : 0.0f;
        v = warp_sum_f(v);
        if (tid == 0) sred[0] = v;
    }
    __syncthreads();
    const float r = rsqrtf(sred[0] / (float) kv_lora + eps);

    if (!lat_q8) {
        __half* dst = (__half*) lat + (size_t) p * kv_lora;
        for (int i = tid; i < kv_lora; i += blockDim.x) dst[i] = __float2half_rn(x[i] * r * w[i]);
    } else {
        // glmd INT8 record: kv_lora int8 codes + (kv_lora / 32) FP16 scales
        const size_t rec_bytes = (size_t) kv_lora + (size_t) (kv_lora / 32) * 2;
        uint8_t* rec = (uint8_t*) lat + (size_t) p * rec_bytes;
        __half* scales = (__half*) (rec + kv_lora);
        for (int i = tid; i < kv_lora; i += blockDim.x) {
            const float y = x[i] * r * w[i];
            const float am = warp_max_f(fabsf(y));
            const __half d16 = __float2half(am / 127.0f);
            const float d = __half2float(d16);
            ((int8_t*) rec)[i] = (int8_t) (d > 0.0f ? max(-127, min(127, __float2int_rn(y / d))) : 0);
            if ((i & 31) == 0) scales[i >> 5] = d16;
        }
    }

    __half* kp = (__half*) k_pe + (size_t) p * n_rot;
    for (int i = tid; i < n_rot / 2; i += blockDim.x) {
        float a = x[kv_lora + 2 * i];
        float b = x[kv_lora + 2 * i + 1];
        rope_pair_d(a, b, p, i, n_rot, base);
        kp[2 * i] = __float2half_rn(a);
        kp[2 * i + 1] = __float2half_rn(b);
    }
}

// ---- MLA query prep: Q8/FP16 q_nope + in-place RoPE on q_pe (grid: (H, T), block: 256) ----
__global__ void __launch_bounds__(256)
    mla_q_prep_kernel(float* __restrict__ qm, int T, int n_head, int qk_head, int qk_nope, int n_rot,
                      const int* __restrict__ pos_dev, int p0, float base, int8_t* __restrict__ q8n,
                      float* __restrict__ d8n, uint16_t* __restrict__ qm16) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int c = threadIdx.x;
    const int p = pos_dev != nullptr ? pos_dev[t] : (p0 + t);
    float* qh = qm + ((size_t) t * n_head + h) * qk_head;

    if (c < qk_nope) {
        const float v = qh[c];
        if (q8n != nullptr) q8_store_warp(v, ((size_t) h * T + t) * qk_nope + c, q8n, d8n);
        if (qm16 != nullptr) ((__half*) qm16)[((size_t) t * n_head + h) * qk_head + c] = __float2half_rn(v);
    } else if (c < qk_nope + (n_rot >> 1)) {
        const int i = c - qk_nope;
        float a = qh[qk_nope + 2 * i];
        float b = qh[qk_nope + 2 * i + 1];
        rope_pair_d(a, b, p, i, n_rot, base);
        qh[qk_nope + 2 * i] = a;
        qh[qk_nope + 2 * i + 1] = b;
    }
}

// ---- MLA per-head output gate (+ optional fused Q8 or FP16) ----
__global__ void head_gate_out_kernel(float* __restrict__ o, const float* __restrict__ gate, int64_t total, int head_dim,
                                     int8_t* __restrict__ xq_out, float* __restrict__ xd_out,
                                     uint16_t* __restrict__ out16) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float v = o[i] * sigmoid_exact(gate[i / head_dim]);
    o[i] = v;
    if (xq_out != nullptr) q8_store_warp(v, (size_t) i, xq_out, xd_out);
    if (out16 != nullptr) ((__half*) out16)[i] = __float2half_rn(v);
}

// ---- Post-SiLU clamped SwiGLU (+ optional fused Q8 or FP16) ----
__global__ void swiglu_clamp_out_kernel(const float* __restrict__ g, const float* __restrict__ u, int64_t total,
                                        float limit, float* __restrict__ h_opt, int8_t* __restrict__ xq_out,
                                        float* __restrict__ xd_out, uint16_t* __restrict__ out16) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float gv = g[i], uv = u[i];
    const float a = gv * sigmoid_exact(gv);
    const float v = limit > 1e-6f ? fminf(a, limit) * fminf(fmaxf(uv, -limit), limit) : a * uv;
    if (h_opt != nullptr) h_opt[i] = v;
    if (xq_out != nullptr) q8_store_warp(v, (size_t) i, xq_out, xd_out);
    if (out16 != nullptr) ((__half*) out16)[i] = __float2half_rn(v);
}

// ---- Batch embedding lookup ----
__global__ void emb_rows_q8_0_kernel(const uint8_t* __restrict__ w, int64_t k, const int* __restrict__ ids,
                                     float* __restrict__ out) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    const int64_t row = (int64_t) ids[t];
    const uint8_t* b = w + row * (k / 32) * 34 + (i >> 5) * 34;
    uint16_t dh;
    memcpy(&dh, b, 2);
    out[(size_t) t * k + i] = (float) ((const int8_t*) (b + 2))[i & 31] * __half2float(*(const __half*) &dh);
}

__global__ void emb_rows_f16_kernel(const uint16_t* __restrict__ w, int64_t k, const int* __restrict__ ids,
                                    float* __restrict__ out) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) out[(size_t) t * k + i] = __half2float(*(const __half*) &w[(int64_t) ids[t] * k + i]);
}

__global__ void emb_rows_bf16_kernel(const uint16_t* __restrict__ w, int64_t k, const int* __restrict__ ids,
                                     float* __restrict__ out) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) {
        const uint32_t bits = (uint32_t) w[(int64_t) ids[t] * k + i] << 16;
        float v;
        memcpy(&v, &bits, 4);
        out[(size_t) t * k + i] = v;
    }
}

__global__ void emb_rows_f32_kernel(const float* __restrict__ w, int64_t k, const int* __restrict__ ids,
                                    float* __restrict__ out) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) out[(size_t) t * k + i] = w[(int64_t) ids[t] * k + i];
}

// ---- MTP concat [rms(emb)*enorm ; rms(h)*hnorm] (grid: 2, block: 256) ----
__global__ void __launch_bounds__(256)
    concat_norm_eh_kernel(const float* __restrict__ emb, const float* __restrict__ enorm, const float* __restrict__ h,
                          const float* __restrict__ hnorm, float eps, int E, float* __restrict__ out_f32,
                          int8_t* __restrict__ xq_out, float* __restrict__ xd_out) {
    __shared__ float sred[8];
    const int part = blockIdx.x;   // 0: emb, 1: h
    const int tid = threadIdx.x;
    const float* x = part == 0 ? emb : h;
    const float* w = part == 0 ? enorm : hnorm;
    float ss = 0.0f;
    for (int i = tid; i < E; i += blockDim.x) ss += x[i] * x[i];
    ss = warp_sum_f(ss);
    if ((tid & 31) == 0) sred[tid >> 5] = ss;
    __syncthreads();
    if (tid < 32) {
        float v = tid < 8 ? sred[tid] : 0.0f;
        v = warp_sum_f(v);
        if (tid == 0) sred[0] = v;
    }
    __syncthreads();
    const float r = rsqrtf(sred[0] / (float) E + eps);
    const size_t off = (size_t) part * E;
    for (int i = tid; i < E; i += blockDim.x) {
        const float y = x[i] * r * w[i];
        if (out_f32 != nullptr) out_f32[off + i] = y;
        if (xq_out != nullptr) q8_store_warp(y, off + i, xq_out, xd_out);
    }
}

__global__ void add2_kernel(float* __restrict__ x, const float* __restrict__ a, const float* __restrict__ b,
                            int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += a[i] + b[i];
}

__global__ void fill_iota_kernel(int* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = i;
}

}  // namespace

void kda_prep_full(const float* q_in, const float* k_in, const float* v_in, const float* f_in, const float* conv_q,
                   const float* conv_k, const float* conv_v, float* conv_state, const float* ssm_a, float lb, float eps,
                   int n_head, int head_dim, int d_conv, float* q_out, float* k_out, float* v_out, float* g1_out,
                   cudaStream_t s) {
    kda_prep_full_kernel<<<dim3((unsigned) n_head, 4u), 128, 0, s>>>(
        q_in, k_in, v_in, f_in, conv_q, conv_k, conv_v, conv_state, ssm_a, lb, eps, n_head, head_dim, d_conv, q_out,
        k_out, v_out, g1_out);
}

void kda_rec_q8(const float* q, const float* k, const float* v, const float* g1, const float* beta_raw, float* state,
                const float* g2, const float* norm_w, float eps, int n_head, int head_dim, int8_t* xq_out,
                float* xd_out, float* raw_o_opt, cudaStream_t s) {
    (void) head_dim;
    kda_rec_q8_kernel<<<(unsigned) n_head, 256, 0, s>>>(q, k, v, g1, beta_raw, state, g2, norm_w, eps, xq_out, xd_out,
                                                        raw_o_opt);
}

void router_grouped(const float* logits, const float* bias, int T, int n_exp, int n_used, int n_group, int n_group_used,
                    bool w_norm, float w_scale, int* ids_out, float* w_out, cudaStream_t s) {
    if (T <= 0) return;
    router_grouped_kernel<<<(unsigned) T, 512, 0, s>>>(logits, bias, n_exp, n_used, n_group, n_group_used,
                                                       w_norm ? 1 : 0, w_scale, ids_out, w_out);
}

void kv_store_split(const float* kv_in, int T, int kv_lora, int n_rot, const float* kv_a_norm, float eps,
                    const int* pos_dev, int p0, float base, void* lat, bool lat_q8, uint16_t* k_pe, cudaStream_t s) {
    if (T <= 0) return;
    kv_store_split_kernel<<<(unsigned) T, 256, 0, s>>>(kv_in, kv_lora, n_rot, kv_a_norm, eps, pos_dev, p0, base, lat,
                                                       lat_q8 ? 1 : 0, k_pe);
}

void mla_q_prep(float* qm, int T, int n_head, int qk_head, int qk_nope, int n_rot, const int* pos_dev, int p0,
                float base, int8_t* q8n, float* d8n, uint16_t* qm16, cudaStream_t s) {
    if (T <= 0) return;
    mla_q_prep_kernel<<<dim3((unsigned) n_head, (unsigned) T), 256, 0, s>>>(qm, T, n_head, qk_head, qk_nope, n_rot,
                                                                            pos_dev, p0, base, q8n, d8n, qm16);
}

void head_gate_out(float* o, const float* gate, int T, int n_head, int head_dim, int8_t* xq_out, float* xd_out,
                   uint16_t* out16, cudaStream_t s) {
    const int64_t total = (int64_t) T * n_head * head_dim;
    if (total <= 0) return;
    head_gate_out_kernel<<<(unsigned) ((total + 255) / 256), 256, 0, s>>>(o, gate, total, head_dim, xq_out, xd_out,
                                                                          out16);
}

void swiglu_clamp_out(const float* g, const float* u, int rows, int n, float limit, float* h_opt, int8_t* xq_out,
                      float* xd_out, uint16_t* out16, cudaStream_t s) {
    const int64_t total = (int64_t) rows * n;
    if (total <= 0) return;
    swiglu_clamp_out_kernel<<<(unsigned) ((total + 255) / 256), 256, 0, s>>>(g, u, total, limit, h_opt, xq_out, xd_out,
                                                                             out16);
}

bool emb_rows(const uint8_t* w, int type, int64_t k, const int* ids_dev, int T, float* out, cudaStream_t s) {
    if (T <= 0) return true;
    const dim3 grid((unsigned) ((k + 255) / 256), (unsigned) T);
    if (type == 8) emb_rows_q8_0_kernel<<<grid, 256, 0, s>>>(w, k, ids_dev, out);
    else if (type == 1) emb_rows_f16_kernel<<<grid, 256, 0, s>>>((const uint16_t*) w, k, ids_dev, out);
    else if (type == 30) emb_rows_bf16_kernel<<<grid, 256, 0, s>>>((const uint16_t*) w, k, ids_dev, out);
    else if (type == 0) emb_rows_f32_kernel<<<grid, 256, 0, s>>>((const float*) w, k, ids_dev, out);
    else return false;
    return true;
}

void concat_norm_eh(const float* emb, const float* enorm, const float* h, const float* hnorm, float eps, int E,
                    float* out_f32, int8_t* xq_out, float* xd_out, cudaStream_t s) {
    concat_norm_eh_kernel<<<2, 256, 0, s>>>(emb, enorm, h, hnorm, eps, E, out_f32, xq_out, xd_out);
}

void add2(float* x, const float* a, const float* b, int64_t n, cudaStream_t s) {
    if (n <= 0) return;
    add2_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(x, a, b, n);
}

void fill_iota(int* dst, int n, cudaStream_t s) {
    if (n <= 0) return;
    fill_iota_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(dst, n);
}

}  // namespace ling::k
