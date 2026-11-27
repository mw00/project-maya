// src/ling/ling_kernels.cu - Ling-3.2-air (bailingmoe3) CUDA kernels.
#include "ling_kernels.cuh"

#include <cuda_fp16.h>
#if !defined(MAYA_USE_HIP)
#include <mma.h>   // (nvcuda::wmma: CUDA only - on AMD the tensor-core MLA kernel's body is empty and never launched)
#endif
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

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

// a xor step among lanes 0..15 only (a block's per-warp sums, the other lanes outside the branch): CUDA's 16-lane
// mask; the HIP compat layer takes full-wave masks only (a partial one traps), so on AMD the plain 16-wide xor - every
// partner of a lane 0..15 is one of them, all active
__device__ __forceinline__ float lanes16_xor_f(float v, int o) {
#if defined(MAYA_USE_HIP)
    return __shfl_xor(v, o, 16);
#else
    return __shfl_xor_sync(0x0000ffffu, v, o);
#endif
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

__device__ __forceinline__ int mrope_pos_at(int t, int kv_p, int rope_delta, int i,
                                            const int* __restrict__ mrope_pos, int mrope_hw, int mrope_stride) {
    if (mrope_pos == nullptr) return kv_p + rope_delta;
    if (i < mrope_hw) return mrope_pos[(size_t) ((i & 1) ? 2 : 1) * (size_t) mrope_stride + (size_t) t];
    return mrope_pos[(size_t) t];
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

    float o_val = 0.0f;
    if (half == 0) {
        o_val = (sacc[0][j] + sacc[1][j]) * rsqrtf((float) HD);
        if (raw_o_opt != nullptr) raw_o_opt[h * HD + j] = o_val;
        const float wsum = warp_sum_f(o_val * o_val);
        if ((j & 31) == 0) sred[j >> 5] = wsum;
    }
    __syncthreads();
    if (half == 0) {
        const float ss = sred[0] + sred[1] + sred[2] + sred[3];
        const float inv = rsqrtf(ss / (float) HD + eps);
        const float gv = o_val * inv * norm_w[j] * sigmoid_exact(g2[h * HD + j]);
        q8_store_warp(gv, (size_t) h * HD + j, xq_out, xd_out);
    }
}

// ---- Fast warp-parallel grouped sigmoid router (grid: T, block: 256) ----
// Preserves exact deterministic tie-breaking (higher score, then lower index).
__device__ __forceinline__ bool r_better(float av, int ai, float bv, int bi) {
    return av > bv || (av == bv && ai < bi);
}

__device__ __forceinline__ void warp_best(float& v, int& id) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, id, o);
        if (r_better(ov, oi, v, id)) {
            v = ov;
            id = oi;
        }
    }
}

__global__ void __launch_bounds__(256)
    router_grouped_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int n_exp, int n_used,
                          int n_group, int n_group_used, int w_norm, float w_scale, int* __restrict__ ids_out,
                          float* __restrict__ w_out, int* __restrict__ err_flag) {
    __shared__ float sp[512];
    __shared__ int s_bad;
    if (threadIdx.x == 0) s_bad = 0;
    __shared__ float sv[512];
    __shared__ float gscore[16];
    __shared__ int gslot[16];
    __shared__ float cand_v[64];
    __shared__ int cand_id[64];

    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int w = tid >> 5;
    const int lane = tid & 31;
    const float* lr = logits + (size_t) t * n_exp;

    // Fast 3-barrier path for Ling (n_group <= 8, per == 64, n_group_used * n_used <= 32)
    if (n_group > 1 && n_group <= 8 && n_exp == n_group * 64 && n_group_used * n_used <= 32) {
        if (w < n_group) {
            const int a_id = w * 64 + lane;
            const int b_id = a_id + 32;
            const float la = lr[a_id];
            const float lb = lr[b_id];
            if (!isfinite(la) || !isfinite(lb)) atomicOr(&s_bad, 1);
            const float pa = isfinite(la) ? sigmoid_exact(la) : 0.0f;
            const float pb = isfinite(lb) ? sigmoid_exact(lb) : 0.0f;
            sp[a_id] = pa;
            sp[b_id] = pb;
            float va = pa + (bias != nullptr ? bias[a_id] : 0.0f);
            float vb = pb + (bias != nullptr ? bias[b_id] : 0.0f);

            // Top-1 in group w
            float bv1 = va;
            int bi1 = a_id;
            if (r_better(vb, b_id, bv1, bi1)) {
                bv1 = vb;
                bi1 = b_id;
            }
            warp_best(bv1, bi1);
            // Top-2 in group w
            float ra = (a_id == bi1) ? -INFINITY : va;
            float rb = (b_id == bi1) ? -INFINITY : vb;
            float bv2 = ra;
            int bi2 = a_id;
            if (r_better(rb, b_id, bv2, bi2)) {
                bv2 = rb;
                bi2 = b_id;
            }
            warp_best(bv2, bi2);
            if (lane == 0) {
                gscore[w] = bv1 + bv2;
                gslot[w] = -1;
            }
            __syncthreads();

            if (tid == 0) {
                for (int k = 0; k < n_group_used; ++k) {
                    int best = -1;
                    float best_v = -INFINITY;
                    for (int g = 0; g < n_group; ++g) {
                        if (gslot[g] >= 0) continue;
                        if (best < 0 || r_better(gscore[g], g, best_v, best)) {
                            best = g;
                            best_v = gscore[g];
                        }
                    }
                    if (best >= 0) gslot[best] = k;
                }
            }
            __syncthreads();

            const int gs = gslot[w];
            if (gs >= 0) {
                // Kept group warp finds its top-n_used purely in registers (zero __syncthreads!)
                for (int k = 0; k < n_used; ++k) {
                    float cv = va;
                    int ci = a_id;
                    if (r_better(vb, b_id, cv, ci)) {
                        cv = vb;
                        ci = b_id;
                    }
                    warp_best(cv, ci);
                    if (lane == 0) {
                        cand_v[gs * n_used + k] = cv;
                        cand_id[gs * n_used + k] = ci;
                    }
                    if (a_id == ci) va = -INFINITY;
                    if (b_id == ci) vb = -INFINITY;
                }
            }
            __syncthreads();

            if (w == 0) {
                const int n_cand = n_group_used * n_used;
                float mv = (lane < n_cand) ? cand_v[lane] : -INFINITY;
                int mi = (lane < n_cand) ? cand_id[lane] : 0x7fffffff;
                int* idr = ids_out + (size_t) t * n_used;
                float* wr = w_out + (size_t) t * n_used;
                int pk_id[16];
                float pk_p[16];
                float sum = 0.0f;
                for (int k = 0; k < n_used; ++k) {
                    float cv = mv;
                    int ci = mi;
                    warp_best(cv, ci);
                    if (lane == 0) {
                        pk_id[k] = ci;
                        const float p = sp[ci];
                        pk_p[k] = p;
                        sum += p;
                    }
                    if (mi == ci) mv = -INFINITY;
                }
                if (lane == 0) {
                    bool bad = (s_bad != 0) || !isfinite(sum);
                    const float inv = (w_norm != 0 && sum > 0.0f) ? (w_scale / sum) : w_scale;
                    if (!isfinite(inv)) bad = true;
                    for (int k = 0; k < n_used; ++k) {
                        if (pk_id[k] < 0 || pk_id[k] >= n_exp || !isfinite(pk_p[k])) bad = true;
                    }
                    if (bad) {
                        if (err_flag != nullptr) atomicExch(err_flag, 1);
                        for (int k = 0; k < n_used; ++k) {
                            idr[k] = 0;
                            wr[k] = 0.0f;
                        }
                    } else {
                        for (int k = 0; k < n_used; ++k) {
                            idr[k] = pk_id[k];
                            wr[k] = pk_p[k] * inv;
                        }
                    }
                }
            }
            return;
        }
        __syncthreads();
        __syncthreads();
        __syncthreads();
        return;
    }

    // Generic fallback for any other (n_exp <= 512, n_group, n_used)
    __shared__ float wv[8];
    __shared__ int wi[8];
    const int e0 = tid;
    const int e1 = tid + 256;
    float p0 = 0.0f, v0 = -INFINITY;
    float p1 = 0.0f, v1 = -INFINITY;
    if (e0 < n_exp) {
        const float l0 = lr[e0];
        if (!isfinite(l0)) atomicOr(&s_bad, 1);
        p0 = isfinite(l0) ? sigmoid_exact(l0) : 0.0f;
        v0 = p0 + (bias != nullptr ? bias[e0] : 0.0f);
    }
    if (e1 < n_exp) {
        const float l1 = lr[e1];
        if (!isfinite(l1)) atomicOr(&s_bad, 1);
        p1 = isfinite(l1) ? sigmoid_exact(l1) : 0.0f;
        v1 = p1 + (bias != nullptr ? bias[e1] : 0.0f);
    }
    sp[e0] = p0;
    sv[e0] = v0;
    sp[e1] = p1;
    sv[e1] = v1;
    __syncthreads();

    if (n_group > 1) {
        const int per = n_exp / n_group;
        if (tid < n_group) {
            float b1 = -INFINITY, b2 = -INFINITY;
            const int g0 = tid * per, g1 = g0 + per;
            for (int e = g0; e < g1; ++e) {
                const float v = sv[e];
                if (v > b1) {
                    b2 = b1;
                    b1 = v;
                } else if (v > b2) {
                    b2 = v;
                }
            }
            gscore[tid] = b1 + b2;
            gslot[tid] = 0;
        }
        __syncthreads();
        if (tid == 0) {
            for (int k = 0; k < n_group_used; ++k) {
                int best = -1;
                float best_v = -INFINITY;
                for (int g = 0; g < n_group; ++g) {
                    if (gslot[g]) continue;
                    if (best < 0 || r_better(gscore[g], g, best_v, best)) {
                        best = g;
                        best_v = gscore[g];
                    }
                }
                if (best >= 0) gslot[best] = 1;
            }
        }
        __syncthreads();
        if (e0 < n_exp && !gslot[e0 / per]) v0 = -INFINITY;
        if (e1 < n_exp && !gslot[e1 / per]) v1 = -INFINITY;
    }

    int* idr = ids_out + (size_t) t * n_used;
    float* wr = w_out + (size_t) t * n_used;
    __shared__ int picked_id[16];
    __shared__ float picked_p[16];
    for (int k = 0; k < n_used; ++k) {
        float bv = v0;
        int bi = e0;
        if (r_better(v1, e1, bv, bi)) {
            bv = v1;
            bi = e1;
        }
        warp_best(bv, bi);
        if (lane == 0) {
            wv[w] = bv;
            wi[w] = bi;
        }
        __syncthreads();
        if (tid == 0) {
            float best_v = wv[0];
            int best_i = wi[0];
            for (int j = 1; j < 8; ++j) {
                if (r_better(wv[j], wi[j], best_v, best_i)) {
                    best_v = wv[j];
                    best_i = wi[j];
                }
            }
            picked_id[k] = best_i;
            picked_p[k] = sp[best_i];
            wi[0] = best_i;
        }
        __syncthreads();
        const int chosen = wi[0];
        if (e0 == chosen) v0 = -INFINITY;
        if (e1 == chosen) v1 = -INFINITY;
    }
    if (tid == 0) {
        float sum = 0.0f;
        bool bad = (s_bad != 0);
        for (int k = 0; k < n_used; ++k) {
            if (picked_id[k] < 0 || picked_id[k] >= n_exp || !isfinite(picked_p[k])) bad = true;
            sum += picked_p[k];
        }
        const float inv = (w_norm != 0 && sum > 0.0f) ? (w_scale / sum) : w_scale;
        if (!isfinite(sum) || !isfinite(inv)) bad = true;
        if (bad) {
            if (err_flag != nullptr) atomicExch(err_flag, 1);
            for (int k = 0; k < n_used; ++k) {
                idr[k] = 0;
                wr[k] = 0.0f;
            }
        } else {
            for (int k = 0; k < n_used; ++k) {
                idr[k] = picked_id[k];
                wr[k] = picked_p[k] * inv;
            }
        }
    }
}

// ---- Split KV cache store (grid: T, block: 256) ----
__global__ void __launch_bounds__(256)
    kv_store_split_kernel(const float* __restrict__ kv_in, int kv_lora, int n_rot, const float* __restrict__ w,
                          float eps, const int* __restrict__ pos_dev, int p0, float base, void* __restrict__ lat,
                          int lat_q8, uint16_t* __restrict__ k_pe, const int* __restrict__ rope_delta_dev,
                          const int* __restrict__ mrope_pos, int mrope_hw, int mrope_stride) {
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

    const int rd = rope_delta_dev != nullptr ? *rope_delta_dev : 0;
    __half* kp = (__half*) k_pe + (size_t) p * n_rot;
    for (int i = tid; i < n_rot / 2; i += blockDim.x) {
        float a = x[kv_lora + 2 * i];
        float b = x[kv_lora + 2 * i + 1];
        const int rp = mrope_pos_at(t, p, rd, i, mrope_pos, mrope_hw, mrope_stride);
        rope_pair_d(a, b, rp, i, n_rot, base);
        kp[2 * i] = __float2half_rn(a);
        kp[2 * i + 1] = __float2half_rn(b);
    }
}

// ---- MLA query prep: Q8/FP16 q_nope + in-place RoPE on q_pe (grid: (H, T), block: 256) ----
__global__ void __launch_bounds__(256)
    mla_q_prep_kernel(float* __restrict__ qm, int T, int n_head, int qk_head, int qk_nope, int n_rot,
                      const int* __restrict__ pos_dev, int p0, float base, int8_t* __restrict__ q8n,
                      float* __restrict__ d8n, uint16_t* __restrict__ qm16, const int* __restrict__ rope_delta_dev,
                      const int* __restrict__ mrope_pos, int mrope_hw, int mrope_stride) {
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
        const int rd = rope_delta_dev != nullptr ? *rope_delta_dev : 0;
        const int rp = mrope_pos_at(t, p, rd, i, mrope_pos, mrope_hw, mrope_stride);
        float a = qh[qk_nope + 2 * i];
        float b = qh[qk_nope + 2 * i + 1];
        rope_pair_d(a, b, rp, i, n_rot, base);
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

__device__ __forceinline__ float emb_h2f(const uint8_t* p) {
    uint16_t h;
    memcpy(&h, p, 2);
    return __half2float(__ushort_as_half(h));
}

__device__ __forceinline__ void emb_q4k_scale(const uint8_t* sc, int j, int& d, int& m) {
    if (j < 4) {
        d = sc[j] & 63;
        m = sc[j + 4] & 63;
    } else {
        d = (sc[j + 4] & 0x0f) | ((sc[j - 4] >> 6) << 4);
        m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
    }
}

__global__ void emb_rows_q4_0_kernel(const uint8_t* __restrict__ w, int64_t k, const int* __restrict__ ids,
                                     float* __restrict__ out) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    const int64_t row_bytes = (k / 32) * 18;
    const uint8_t* blk = w + (int64_t) ids[t] * row_bytes + (i / 32) * 18;
    const float d = emb_h2f(blk);
    const int j = (int) (i & 31);
    const uint8_t byte = blk[2 + (j & 15)];
    const int q = (j < 16) ? (byte & 0x0f) : (byte >> 4);
    out[(size_t) t * k + i] = d * (float) (q - 8);
}

__global__ void emb_rows_q4_k_kernel(const uint8_t* __restrict__ w, int64_t k, const int* __restrict__ ids,
                                     float* __restrict__ out) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    const int64_t row_bytes = (k / 256) * 144;
    const uint8_t* blk = w + (int64_t) ids[t] * row_bytes + (i / 256) * 144;
    const int off = (int) (i & 255);
    const int ib = off >> 5;
    const int r = off & 31;
    int sc, m;
    emb_q4k_scale(blk + 4, ib, sc, m);
    const float d1 = emb_h2f(blk) * (float) sc;
    const float m1 = emb_h2f(blk + 2) * (float) m;
    const uint8_t byte = blk[16 + 32 * (ib >> 1) + r];
    const int q = (ib & 1) ? (byte >> 4) : (byte & 0x0f);
    out[(size_t) t * k + i] = d1 * (float) q - m1;
}

__global__ void emb_rows_q5_k_kernel(const uint8_t* __restrict__ w, int64_t k, const int* __restrict__ ids,
                                     float* __restrict__ out) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    const int64_t row_bytes = (k / 256) * 176;
    const uint8_t* blk = w + (int64_t) ids[t] * row_bytes + (i / 256) * 176;
    const int off = (int) (i & 255);
    const int ib = off >> 5;
    const int r = off & 31;
    int sc, m;
    emb_q4k_scale(blk + 4, ib, sc, m);
    const float d1 = emb_h2f(blk) * (float) sc;
    const float m1 = emb_h2f(blk + 2) * (float) m;
    const uint8_t byte = blk[48 + 32 * (ib >> 1) + r];
    const int lo = (ib & 1) ? (byte >> 4) : (byte & 0x0f);
    const int hi = ((blk[16 + r] >> ib) & 1) << 4;
    out[(size_t) t * k + i] = d1 * (float) (lo | hi) - m1;
}

__global__ void emb_rows_q6_k_kernel(const uint8_t* __restrict__ w, int64_t k, const int* __restrict__ ids,
                                     float* __restrict__ out) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    const int64_t row_bytes = (k / 256) * 210;
    const uint8_t* blk = w + (int64_t) ids[t] * row_bytes + (i / 256) * 210;
    const int off = (int) (i & 255);
    const int ib = off >> 5;
    const int r = off & 31;
    const int hf = ib >> 2, qt = ib & 3;
    const uint8_t ql = blk[64 * hf + 32 * (qt & 1) + r];
    const uint8_t qh = blk[128 + 32 * hf + r];
    const int8_t sc = ((const int8_t*) (blk + 192 + 8 * hf + 2 * qt))[r >> 4];
    const int q = (((ql >> (4 * (qt >> 1))) & 15) | (((qh >> (2 * qt)) & 3) << 4)) - 32;
    out[(size_t) t * k + i] = __fmul_rn(__fmul_rn(emb_h2f(blk + 208), (float) sc), (float) q);
}

// Per-256 Q8_K activation quantization matching maya::ds4::par::quant bit-for-bit.
// Grid: (k / 256, rows), block: 256.
__global__ void __launch_bounds__(256)
    q8k_rows_kernel(const float* __restrict__ x, int k, int8_t* __restrict__ xq, float* __restrict__ xd) {
    __shared__ float s_am[8];
    __shared__ int s_ix[8];
    const int b = blockIdx.x;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const float* xb = x + (size_t) row * k + (size_t) b * 256;
    const float v = xb[tid];
    float am = fabsf(v);
    int ix = tid;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float oa = __shfl_xor_sync(0xffffffffu, am, o);
        const int oi = __shfl_xor_sync(0xffffffffu, ix, o);
        if (oa > am || (oa == am && oi < ix)) {
            am = oa;
            ix = oi;
        }
    }
    if (lane == 0) {
        s_am[warp] = am;
        s_ix[warp] = ix;
    }
    __syncthreads();
    if (warp == 0) {
        float bm = (lane < 8) ? s_am[lane] : -1.0f;
        int bi = (lane < 8) ? s_ix[lane] : 256;
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) {
            const float oa = __shfl_xor_sync(0xffffffffu, bm, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (oa > bm || (oa == bm && oi < bi)) {
                bm = oa;
                bi = oi;
            }
        }
        if (lane == 0) {
            s_am[0] = bm;
            s_ix[0] = bi;
        }
    }
    __syncthreads();
    const float max_a = s_am[0];
    const int max_i = s_ix[0];
    int8_t* qb = xq + (size_t) row * k + (size_t) b * 256;
    float* db = xd + (size_t) row * (k / 32) + (size_t) b * 8;
    float dv = 0.0f;
    int8_t qv = 0;
    if (max_a > 0.0f) {
        const float iscale = __fdiv_rn(-127.0f, xb[max_i]);
        const float val = __fadd_rn(__fmul_rn(iscale, v), 12582912.0f);
        const int32_t bits = __float_as_int(val);
        int qi = (bits & 0x007fffff) - 0x00400000;
        if (qi > 127) qi = 127;
        qv = (int8_t) qi;
        dv = __fdiv_rn(1.0f, iscale);
    }
    qb[tid] = qv;
    if (tid < 8) db[tid] = dv;
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

// ---- Tiled dense MLA prefill kernels ----
__global__ void __launch_bounds__(256)
    mla_q_split_rope_f16_kernel(const float* __restrict__ qm, int T, int n_head, int qk_head, int qk_nope, int n_rot,
                                int pos0, float base, uint16_t* __restrict__ q_nope16, uint16_t* __restrict__ q_pe16,
                                const int* __restrict__ mrope_pos, int mrope_hw, int mrope_stride) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int c = threadIdx.x;
    const float* qh = qm + ((size_t) t * n_head + h) * qk_head;
    if (c < qk_nope) {
        ((__half*) q_nope16)[((size_t) h * T + t) * qk_nope + c] = __float2half_rn(qh[c]);
    } else if (c < qk_nope + (n_rot >> 1)) {
        const int i = c - qk_nope;
        float a = qh[qk_nope + 2 * i];
        float b = qh[qk_nope + 2 * i + 1];
        const int rp = mrope_pos_at(t, pos0 + t, 0, i, mrope_pos, mrope_hw, mrope_stride);
        rope_pair_d(a, b, rp, i, n_rot, base);
        __half* dst = (__half*) q_pe16 + ((size_t) h * T + t) * n_rot;
        dst[2 * i] = __float2half_rn(a);
        dst[2 * i + 1] = __float2half_rn(b);
    }
}

__global__ void __launch_bounds__(256)
    lat_q8_to_f16_kernel(const uint8_t* __restrict__ lat, int k0, int kv_lora, uint16_t* __restrict__ out16) {
    const int r = blockIdx.x;
    const int tid = threadIdx.x;
    const size_t rec_bytes = (size_t) kv_lora + (size_t) (kv_lora / 32) * 2;
    const uint8_t* rec = lat + (size_t) (k0 + r) * rec_bytes;
    const int8_t* q = (const int8_t*) rec;
    const __half* scales = (const __half*) (rec + kv_lora);
    __half* dst = (__half*) out16 + (size_t) r * kv_lora;
    for (int i = tid; i < kv_lora; i += blockDim.x) {
        const float v = (float) q[i] * __half2float(scales[i >> 5]);
        dst[i] = __float2half_rn(v);
    }
}

__global__ void __launch_bounds__(256)
    mla_init_state_kernel(float* __restrict__ m, float* __restrict__ l, float* __restrict__ o_acc, int64_t HT,
                          int v_head) {
    const int64_t idx0 = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    for (int64_t r = idx0; r < HT; r += stride) {
        m[r] = -INFINITY;
        l[r] = 0.0f;
    }
    const int64_t total_o = HT * (int64_t) v_head;
    for (int64_t idx = idx0; idx < total_o; idx += stride) o_acc[idx] = 0.0f;
}

__global__ void __launch_bounds__(256)
    max_rows_from_bounds_kernel(const int* __restrict__ bounds, int n_slots, int* __restrict__ out_dev) {
    __shared__ int sm[8];
    const int tid = threadIdx.x;
    int mx = 0;
    for (int e = tid; e < n_slots; e += blockDim.x) mx = max(mx, bounds[e + 1] - bounds[e]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if ((tid & 31) == 0) sm[tid >> 5] = mx;
    __syncthreads();
    if (tid == 0) {
        int best = sm[0];
        for (int w = 1; w < 8; ++w) best = max(best, sm[w]);
        *out_dev = best;
    }
}

__global__ void __launch_bounds__(256)
    mla_online_softmax_kernel(const float* __restrict__ S, int H, int T, int q0, int BQ, int k0, int BK, int pos0,
                              float* __restrict__ m, float* __restrict__ l, float* __restrict__ o_acc, int v_head,
                              uint16_t* __restrict__ P16) {
    __shared__ float smax[8];
    __shared__ float ssum[8];
    const int q = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int q_pos = pos0 + q0 + q;
    const int valid_k = min(BK, max(0, q_pos - k0 + 1));

    const size_t s_off = ((size_t) h * BQ + q) * BK;
    const float* srow = S + s_off;
    __half* prow = (__half*) P16 + s_off;

    if (valid_k <= 0) {
        for (int j = tid; j < BK; j += blockDim.x) prow[j] = __float2half_rn(0.0f);
        return;
    }

    float rmax = -INFINITY;
    for (int j = tid; j < valid_k; j += blockDim.x) rmax = fmaxf(rmax, srow[j]);
    rmax = warp_max_f(rmax);
    if ((tid & 31) == 0) smax[tid >> 5] = rmax;
    __syncthreads();
    if (tid < 32) {
        float v = tid < 8 ? smax[tid] : -INFINITY;
        v = warp_max_f(v);
        if (tid == 0) smax[0] = v;
    }
    __syncthreads();

    const size_t ht = (size_t) h * T + (size_t) (q0 + q);
    const float m_old = m[ht];
    const float m_tile = smax[0];
    const float m_new = fmaxf(m_old, m_tile);

    float rsum = 0.0f;
    for (int j = tid; j < BK; j += blockDim.x) {
        if (j < valid_k) {
            const float p = expf(srow[j] - m_new);
            prow[j] = __float2half_rn(p);
            rsum += p;
        } else {
            prow[j] = __float2half_rn(0.0f);
        }
    }
    rsum = warp_sum_f(rsum);
    if ((tid & 31) == 0) ssum[tid >> 5] = rsum;
    __syncthreads();
    if (tid < 32) {
        float v = tid < 8 ? ssum[tid] : 0.0f;
        v = warp_sum_f(v);
        if (tid == 0) ssum[0] = v;
    }
    __syncthreads();

    const float alpha = (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);
    if (alpha != 1.0f && m_old != -INFINITY) {
        float* orow = o_acc + ht * (size_t) v_head;
        for (int d = tid; d < v_head; d += blockDim.x) orow[d] *= alpha;
    }
    if (tid == 0) {
        m[ht] = m_new;
        l[ht] = l[ht] * alpha + ssum[0];
    }
}

__global__ void __launch_bounds__(128)
    mla_o_finalize_gate_kernel(const float* __restrict__ o_acc, const float* __restrict__ l,
                               const float* __restrict__ gate, int T, int H, int v_head, uint16_t* __restrict__ out16) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int d = threadIdx.x;
    if (d >= v_head) return;
    const size_t ht = (size_t) h * T + (size_t) t;
    const float inv_l = 1.0f / l[ht];
    const float g = sigmoid_exact(gate[(size_t) t * H + h]);
    const float v = o_acc[ht * (size_t) v_head + d] * inv_l * g;
    ((__half*) out16)[((size_t) t * H + h) * v_head + d] = __float2half_rn(v);
}

// ---- Fast O(N) adaptive expert cache resolver (grid: 1, block: 512) ----
__device__ __forceinline__ bool f_worse(float av, int ai, float bv, int bi) {
    return av < bv || (av == bv && ai < bi);
}

__global__ void __launch_bounds__(512)
    fetch_resolve_fast_kernel(const int* __restrict__ ids, int T, int n_used, hy4::k::FetchLayer fl,
                              hy4::k::FetchScratch fs, int k_gpu, int cache, float decay) {
    __shared__ int ent[hy4::k::kMaxWin * 16];
    __shared__ int cnt[1024];
    __shared__ unsigned char gpu_e[1024];
    __shared__ float sc[1024];
    __shared__ int miss_e[hy4::k::kMaxWin * 16];
    __shared__ float miss_sc[hy4::k::kMaxWin * 16];
    __shared__ float wv[16];
    __shared__ int wi[16];
    __shared__ int n_miss, n_hit, n_lane, n_fetch, n_zc, n_adm, n_stage, s_admit_open;

    const int tid = threadIdx.x;
    const int n_ent = T * n_used;
    const int n_exp = fl.n_exp;
    if (tid == 0) n_miss = n_hit = n_lane = n_fetch = n_zc = n_adm = n_stage = 0;
    for (int i = tid; i < n_ent; i += blockDim.x) ent[i] = ids[i];
    for (int e = tid; e < n_exp; e += blockDim.x) {
        cnt[e] = 0;
        gpu_e[e] = 0;
    }
    __syncthreads();

    if (tid < T) {
        int g = 0, hit = 0, lane = 0;
        for (int j = 0; j < n_used; ++j) {
            const int i = tid * n_used + j;
            const int e = ent[i];
            const bool res = fl.slot[e] >= 0;
            const bool mine = res || k_gpu < 0 || g < k_gpu;
            if (!res && mine) ++g;
            fs.who[i] = mine ? 0 : 1;
            if (mine) gpu_e[e] = 1;
            hit += res;
            lane += !mine;
            atomicAdd(&cnt[e], 1);
        }
        atomicAdd(&n_hit, hit);
        atomicAdd(&n_lane, lane);
    }
    __syncthreads();

    const float dec_pow = cache ? (T == 1 ? decay : powf(decay, (float) T)) : 0.0f;
    for (int e = tid; e < n_exp; e += blockDim.x) {
        const int s = fl.slot[e];
        float v = 0.0f;
        if (cache) {
            v = fl.score[e] * dec_pow + (float) cnt[e];
            fl.score[e] = v;
        }
        sc[e] = v;
        if (gpu_e[e]) {
            if (s >= 0) {
                fs.eptr[e] = (unsigned long long) (fl.vram + (size_t) s * fl.blob);
            } else {
                const int m_idx = atomicAdd(&n_miss, 1);
                miss_e[m_idx] = e;
                miss_sc[m_idx] = v;
            }
        } else {
            fs.eptr[e] = 0ull;
        }
    }
    __syncthreads();

    const int nm = n_miss;
    if (nm == 0) {
        if (tid == 0) {
            *fs.nf = 0u;
            if (fs.stats) {
                atomicAdd(&fs.stats[0], (unsigned long long) n_ent);
                atomicAdd(&fs.stats[1], (unsigned long long) n_hit);
                atomicAdd(&fs.stats[4], (unsigned long long) n_lane);
            }
        }
        return;
    }

    // Sort the tiny miss list (typically 1-4 elements) descending by score on thread 0
    if (tid == 0 && nm > 1) {
        for (int i = 1; i < nm; ++i) {
            const int ke = miss_e[i];
            const float ks = miss_sc[i];
            int j = i - 1;
            while (j >= 0 && (miss_sc[j] < ks || (miss_sc[j] == ks && miss_e[j] > ke))) {
                miss_e[j + 1] = miss_e[j];
                miss_sc[j + 1] = miss_sc[j];
                --j;
            }
            miss_e[j + 1] = ke;
            miss_sc[j + 1] = ks;
        }
    }
    __syncthreads();

    // O(nm * N) parallel warp-min victim selection (nm is typically 1-2!)
    const int w = tid >> 5;
    const int lane = tid & 31;
    const int n_warps = blockDim.x >> 5;
    if (tid == 0) s_admit_open = (cache != 0 && fl.n_res > 0) ? 1 : 0;
    __syncthreads();
    for (int mi = 0; mi < nm; ++mi) {
        int vict = -1;
        float vict_sc = INFINITY;
        const bool ao = (s_admit_open != 0);
        if (ao) {
            float bv = INFINITY;
            int bi = 0x7fffffff;
            for (int e = tid; e < n_exp; e += blockDim.x) {
                if (fl.slot[e] >= 0 && cnt[e] == 0) {
                    const float v = sc[e];
                    if (f_worse(v, e, bv, bi)) {
                        bv = v;
                        bi = e;
                    }
                }
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (f_worse(ov, oi, bv, bi)) {
                    bv = ov;
                    bi = oi;
                }
            }
            if (lane == 0) {
                wv[w] = bv;
                wi[w] = bi;
            }
            __syncthreads();
            if (tid == 0) {
                float best_v = INFINITY;
                int best_i = 0x7fffffff;
                for (int k = 0; k < n_warps; ++k) {
                    if (f_worse(wv[k], wi[k], best_v, best_i)) {
                        best_v = wv[k];
                        best_i = wi[k];
                    }
                }
                wv[0] = best_v;
                wi[0] = best_i;
            }
            __syncthreads();
            if (wi[0] < n_exp) {
                vict = wi[0];
                vict_sc = wv[0];
            }
        }

        if (tid == 0) {
            const int m = miss_e[mi];
            unsigned long long dst = 0ull;
            if (ao && vict >= 0 && miss_sc[mi] > vict_sc) {
                const int sl = fl.slot[vict];
                fl.slot[vict] = -1;
                fl.slot[m] = sl;
                fl.owner[sl] = m;
                dst = (unsigned long long) (fl.vram + (size_t) sl * fl.blob);
                ++n_adm;
            } else {
                s_admit_open = 0;
                if (n_stage < fs.n_stage) {
                    dst = (unsigned long long) (fs.stage + (size_t) n_stage * fl.blob);
                    ++n_stage;
                }
            }
            if (dst != 0ull) {
                const int f = n_fetch++;
                fs.src[f] = fl.hptr[m];
                fs.dst[f] = dst;
                fs.eptr[m] = dst;
            } else {
                fs.eptr[m] = fl.hptr[m];
                ++n_zc;
            }
        }
        if (tid == 0 && ao && wi[0] >= n_exp) s_admit_open = 0;
        __syncthreads();
    }

    if (tid == 0) {
        *fs.nf = (unsigned) n_fetch;
        if (fs.stats) {
            atomicAdd(&fs.stats[0], (unsigned long long) n_ent);
            atomicAdd(&fs.stats[1], (unsigned long long) n_hit);
            atomicAdd(&fs.stats[2], (unsigned long long) n_fetch);
            atomicAdd(&fs.stats[3], (unsigned long long) n_zc);
            atomicAdd(&fs.stats[4], (unsigned long long) n_lane);
            atomicAdd(&fs.stats[5], (unsigned long long) n_adm);
        }
    }
}

// ---- Fast 2-stage parallel argmax (stage 1: 128 blocks x 256, stage 2: 1 block x 128) ----
__global__ void __launch_bounds__(256)
    argmax_stage1_kernel(const float* __restrict__ x, int n, float* __restrict__ pval, int* __restrict__ pidx) {
    __shared__ float sv[8];
    __shared__ int si[8];
    const int tid = threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) {
        const float v = x[i];
        if (v > bv || (v == bv && i < bi)) {
            bv = v;
            bi = i;
        }
    }
    warp_best(bv, bi);
    if ((tid & 31) == 0) {
        sv[tid >> 5] = bv;
        si[tid >> 5] = bi;
    }
    __syncthreads();
    if (tid == 0) {
        float best_v = sv[0];
        int best_i = si[0];
        for (int w = 1; w < 8; ++w) {
            if (r_better(sv[w], si[w], best_v, best_i)) {
                best_v = sv[w];
                best_i = si[w];
            }
        }
        pval[blockIdx.x] = best_v;
        pidx[blockIdx.x] = best_i;
    }
}

__global__ void __launch_bounds__(128)
    argmax_stage2_kernel(const float* __restrict__ pval, const int* __restrict__ pidx, int m, int* __restrict__ out) {
    __shared__ float sv[4];
    __shared__ int si[4];
    const int tid = threadIdx.x;
    float bv = tid < m ? pval[tid] : -INFINITY;
    int bi = tid < m ? pidx[tid] : 0x7fffffff;
    warp_best(bv, bi);
    if ((tid & 31) == 0) {
        sv[tid >> 5] = bv;
        si[tid >> 5] = bi;
    }
    __syncthreads();
    if (tid == 0) {
        float best_v = sv[0];
        int best_i = si[0];
        for (int w = 1; w < 4; ++w) {
            if (r_better(sv[w], si[w], best_v, best_i)) {
                best_v = sv[w];
                best_i = si[w];
            }
        }
        *out = best_i;
    }
}

// ---- Fused add / add2 + RMSNorm + Q8_0 quantization (T >= 1, E <= 4096) ----
template <int NADD>
__global__ void __launch_bounds__(512)
    add_rmsnorm_q8_kernel(float* __restrict__ x, const float* __restrict__ a, const float* __restrict__ b,
                          const float* __restrict__ w, int E, float eps, float* __restrict__ cur_out,
                          int8_t* __restrict__ xq, float* __restrict__ xd) {
    __shared__ float s_sum[16];
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    float* xr = x + (size_t) t * E;
    const float* ar = a + (size_t) t * E;
    const float* br = (NADD == 2) ? (b + (size_t) t * E) : nullptr;
    float* cr = cur_out != nullptr ? (cur_out + (size_t) t * E) : nullptr;
    int8_t* qr = xq + (size_t) t * E;
    float* dr = xd + (size_t) t * (E / 32);

    float xv[8], wv[8];
    float ss = 0.0f;
    const int n_iter = (E + 511) >> 9;
    for (int j = 0; j < n_iter; ++j) {
        const int i = tid + j * 512;
        if (i < E) {
            float v = xr[i] + ar[i];
            if constexpr (NADD == 2) v += br[i];
            xr[i] = v;
            xv[j] = v;
            wv[j] = w[i];
            ss += v * v;
        }
    }
    ss = warp_sum_f(ss);
    if (lane == 0) s_sum[warp] = ss;
    __syncthreads();
    if (tid < 16) {
        float v = s_sum[tid];
        for (int o = 8; o > 0; o >>= 1) v += lanes16_xor_f(v, o);
        if (tid == 0) s_sum[0] = rsqrtf(v / (float) E + eps);
    }
    __syncthreads();
    const float inv = s_sum[0];

    for (int j = 0; j < n_iter; ++j) {
        const int i = tid + j * 512;
        if (i < E) {
            const float o = xv[j] * inv * wv[j];
            if (cr != nullptr) cr[i] = o;
            const float am = warp_max_f(fabsf(o));
            const float sc = am / 127.0f;
            qr[i] = (int8_t) (am > 0.0f ? __float2int_rn(o / sc) : 0);
            if (lane == 0) dr[i >> 5] = sc;
        }
    }
}

__global__ void __launch_bounds__(32)
    pack_route_cpu_kernel(const int* __restrict__ ids, const float* __restrict__ w, const int* __restrict__ who,
                          int n_used, RoutePack* __restrict__ dst) {
    const int i = threadIdx.x;
    if (i < n_used) {
        dst->ids[i] = ids[i];
        dst->w[i] = w[i];
        dst->who[i] = who[i];
    }
}

// ---- Portable Vectorized Register-Accumulated WMMA Split-K MLA Decode (H=32, KL=512, R=64) ----
constexpr int kMlaMaxSplits1 = 128;
constexpr int kMlaMaxSplitsM = 64;
constexpr int kMlaK = 576;
constexpr int kMlaLora = 512;
constexpr int kMlaPart = kMlaLora + 4;

template <int TILE>
constexpr size_t mla_wmma_smem_bytes() {
    return (size_t) 32 * kMlaK * 2 + (size_t) TILE * kMlaK * 2 + (size_t) 8 * 256 * 4 + (size_t) 32 * TILE * 2 + 512;
}

__device__ __forceinline__ int mla_num_splits(int n, int tile, int s_max) {
    const int n_tiles = (n + tile - 1) / tile;
    return min(s_max, max(1, (n_tiles + 1) >> 1));
}

template <int LQ8, int TILE>
__global__ void __launch_bounds__(256, 1)
    mla_dec_wmma_kernel(const float* __restrict__ q_abs, const float* __restrict__ q, int q_stride, int q_pe_off,
                        const void* __restrict__ lat, const uint16_t* __restrict__ k_pe, int n_sel_cap,
                        const int* __restrict__ n_sel_dev, int T, float scale, int s_max, float* __restrict__ part) {
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    extern __shared__ __align__(16) unsigned char smem[];
    __half* sQ = (__half*) smem;                                                   // [32][576]
    __half* sL = sQ + 32 * kMlaK;                                                  // [TILE][576]
    float* sTmp = (float*) (sL + TILE * kMlaK);                                    // [8][256]
    __half* sP = (__half*) (sTmp + 8 * 256);                                       // [32][TILE]
    float* sM = (float*) (sP + 32 * TILE);                                         // [32]
    float* sSum = sM + 32;                                                         // [32]
    float* sAlpha = sSum + 32;                                                     // [32]

    const int b = blockIdx.x;
    const int tok = blockIdx.y;
    const int tid = threadIdx.x;
    const int w = tid >> 5;
    const int lane = tid & 31;
    const int hg = w >> 2;   // 0: heads 0..15, 1: heads 16..31
    const int ks = w & 3;    // 0..3

    const int n_base = n_sel_dev != nullptr ? min(n_sel_cap, *n_sel_dev) : n_sel_cap;
    const int n = n_base - (T - 1 - tok);
    if (n <= 0) return;
    const int S = mla_num_splits(n, TILE, s_max);
    if (b >= S) return;

    const int n_tiles = (n + TILE - 1) / TILE;
    const int t0 = (int) (((int64_t) b * n_tiles) / S);
    const int t1 = (int) (((int64_t) (b + 1) * n_tiles) / S);

    // Load all 32 heads of query token `tok` into sQ[32][576] (FP16) using warp-coalesced rows
    const float* qa_tok = q_abs + (size_t) tok * 32 * kMlaLora;
    const float* qp_tok = q + (size_t) tok * 32 * q_stride + q_pe_off;
    for (int h = w; h < 32; h += 8) {
        __half* dst = sQ + (size_t) h * kMlaK;
        const float* qa_h = qa_tok + (size_t) h * kMlaLora;
        const float* qp_h = qp_tok + (size_t) h * q_stride;
        for (int d = lane * 2; d < kMlaLora; d += 64) {
            *((__half2*) (dst + d)) = __floats2half2_rn(qa_h[d], qa_h[d + 1]);
        }
        *((__half2*) (dst + kMlaLora + lane * 2)) = __floats2half2_rn(qp_h[lane * 2], qp_h[lane * 2 + 1]);
    }
    // Portable WMMA row-index discovery (exact on sm_70..sm_120 without arch-specific lane assumptions!)
    sTmp[tid] = (float) (tid >> 4);
    if (tid < 32) {
        sM[tid] = -INFINITY;
        sSum[tid] = 0.0f;
        sAlpha[tid] = 0.0f;
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> rfrag;
    wmma::load_matrix_sync(rfrag, sTmp, 16, wmma::mem_row_major);
    int my_row[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) my_row[e] = (int) (rfrag.x[e] + 0.5f);
    __syncthreads();

    // Register-resident WMMA accumulators for O[hg*16..+15, ks*128..+127] — zero smem roundtrips in tile loop!
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) wmma::fill_fragment(o_acc[j], 0.0f);

    const size_t rec_bytes = (size_t) kMlaLora + (size_t) (kMlaLora / 32) * 2;  // 544 B (16B aligned)

    for (int tile = t0; tile < t1; ++tile) {
        const int k_base = tile * TILE;

        // Vectorized 16-byte coalesced load of TILE KV rows into sL[TILE][576] (FP16)
        for (int c = w; c < TILE; c += 8) {
            __half* dst = sL + (size_t) c * kMlaK;
            const int pos = k_base + c;
            if (pos < n) {
                if constexpr (LQ8 == 0) {
                    const uint4* src_lat = (const uint4*) ((const __half*) lat + (size_t) pos * kMlaLora);
                    uint4* dst_lat = (uint4*) dst;
                    dst_lat[lane] = src_lat[lane];
                    dst_lat[lane + 32] = src_lat[lane + 32];
                } else {
                    const uint8_t* rec = (const uint8_t*) lat + (size_t) pos * rec_bytes;
                    const uint4 qv = ((const uint4*) rec)[lane];
                    const float sc = __half2float(((const __half*) (rec + kMlaLora))[lane >> 1]);
                    uint4 out0, out1;
                    __half2* p0 = (__half2*) &out0;
                    __half2* p1 = (__half2*) &out1;
                    auto up = [&](uint32_t wd, __half2& h01, __half2& h23) {
                        h01 = __floats2half2_rn((float) (int8_t) (wd & 0xffu) * sc,
                                                (float) (int8_t) ((wd >> 8) & 0xffu) * sc);
                        h23 = __floats2half2_rn((float) (int8_t) ((wd >> 16) & 0xffu) * sc,
                                                (float) (int8_t) ((wd >> 24) & 0xffu) * sc);
                    };
                    up(qv.x, p0[0], p0[1]);
                    up(qv.y, p0[2], p0[3]);
                    up(qv.z, p1[0], p1[1]);
                    up(qv.w, p1[2], p1[3]);
                    uint4* dst_lat = (uint4*) dst;
                    dst_lat[lane * 2] = out0;
                    dst_lat[lane * 2 + 1] = out1;
                }
                if (lane < 8) {
                    ((uint4*) (dst + kMlaLora))[lane] = ((const uint4*) (k_pe + (size_t) pos * 64))[lane];
                }
            } else {
                uint4 z = make_uint4(0, 0, 0, 0);
                uint4* dst_lat = (uint4*) dst;
                dst_lat[lane] = z;
                dst_lat[lane + 32] = z;
                if (lane < 8) ((uint4*) (dst + kMlaLora))[lane] = z;
            }
        }
        __syncthreads();

        // Phase A: Score GEMM S[32, TILE] = sQ[32, 576] @ sL[TILE, 576]^T
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_acc;
        wmma::fill_fragment(s_acc, 0.0f);
        if constexpr (TILE == 32) {
            const int cg = (w >> 1) & 1;
            const int kh = w & 1;
            const int k_start = kh * 18;
            const __half* sQ_row = sQ + (size_t) (hg * 16) * kMlaK;
            const __half* sL_row = sL + (size_t) (cg * 16) * kMlaK;
#pragma unroll
            for (int ki = 0; ki < 18; ++ki) {
                const int k_off = (k_start + ki) * 16;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> qa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> kb;
                wmma::load_matrix_sync(qa, sQ_row + k_off, kMlaK);
                wmma::load_matrix_sync(kb, sL_row + k_off, kMlaK);
                wmma::mma_sync(s_acc, qa, kb, s_acc);
            }
        } else {
            const int k_start = ks * 9;
            const __half* sQ_row = sQ + (size_t) (hg * 16) * kMlaK;
#pragma unroll
            for (int ki = 0; ki < 9; ++ki) {
                const int k_off = (k_start + ki) * 16;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> qa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> kb;
                wmma::load_matrix_sync(qa, sQ_row + k_off, kMlaK);
                wmma::load_matrix_sync(kb, sL + k_off, kMlaK);
                wmma::mma_sync(s_acc, qa, kb, s_acc);
            }
        }
        wmma::store_matrix_sync(sTmp + w * 256, s_acc, 16, wmma::mem_row_major);
        __syncthreads();

        // Phase B: 8-warp parallel online softmax (8 consecutive lanes per head -> 32 heads across 256 threads)
        {
            const int h = tid >> 3;
            const int lh = tid & 7;
            const int h_g = h >> 4;
            const int h_l = h & 15;
            constexpr int C_PER = TILE / 8;  // 4 for TILE=32, 2 for TILE=16
            float sc_loc[C_PER];
            float m_loc = -INFINITY;
#pragma unroll
            for (int u = 0; u < C_PER; ++u) {
                const int c = lh * C_PER + u;
                if (k_base + c < n) {
                    float sum_v;
                    if constexpr (TILE == 32) {
                        const int cg = c >> 4;
                        const int idx = h_l * 16 + (c & 15);
                        sum_v = sTmp[(h_g * 4 + cg * 2 + 0) * 256 + idx] +
                                sTmp[(h_g * 4 + cg * 2 + 1) * 256 + idx];
                    } else {
                        const int idx = h_l * 16 + c;
                        sum_v = sTmp[(h_g * 4 + 0) * 256 + idx] + sTmp[(h_g * 4 + 1) * 256 + idx] +
                                sTmp[(h_g * 4 + 2) * 256 + idx] + sTmp[(h_g * 4 + 3) * 256 + idx];
                    }
                    const float sv = sum_v * scale;
                    sc_loc[u] = sv;
                    m_loc = fmaxf(m_loc, sv);
                } else {
                    sc_loc[u] = -INFINITY;
                }
            }
#pragma unroll
            for (int o = 4; o > 0; o >>= 1) m_loc = fmaxf(m_loc, __shfl_xor_sync(0xffffffffu, m_loc, o));
            const float m_old = sM[h];
            const float m_new = fmaxf(m_old, m_loc);
            const float al = (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);
            float l_loc = 0.0f;
#pragma unroll
            for (int u = 0; u < C_PER; ++u) {
                const int c = lh * C_PER + u;
                const float p = (k_base + c < n) ? expf(sc_loc[u] - m_new) : 0.0f;
                sP[h * TILE + c] = __float2half_rn(p);
                l_loc += p;
            }
#pragma unroll
            for (int o = 4; o > 0; o >>= 1) l_loc += __shfl_xor_sync(0xffffffffu, l_loc, o);
            if (lh == 0) {
                sM[h] = m_new;
                sSum[h] = sSum[h] * al + l_loc;
                sAlpha[h] = al;
            }
        }
        __syncthreads();

        // Phase C: Value GEMM O[32, 512] = alpha * O + sP[32, TILE] @ sL[TILE, 512] (pure register accumulation!)
        float al_e[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) al_e[e] = sAlpha[hg * 16 + my_row[e]];
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) o_acc[j].x[e] *= al_e[e];

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pa0;
        wmma::load_matrix_sync(pa0, sP + (hg * 16) * TILE, TILE);
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pa1;
        if constexpr (TILE == 32) {
            wmma::load_matrix_sync(pa1, sP + (hg * 16) * TILE + 16, TILE);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int col_base = (ks * 8 + j) * 16;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vb0;
            wmma::load_matrix_sync(vb0, sL + col_base, kMlaK);
            wmma::mma_sync(o_acc[j], pa0, vb0, o_acc[j]);
            if constexpr (TILE == 32) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vb1;
                wmma::load_matrix_sync(vb1, sL + 16 * kMlaK + col_base, kMlaK);
                wmma::mma_sync(o_acc[j], pa1, vb1, o_acc[j]);
            }
        }
        __syncthreads();
    }

    // Store final accumulated O once per split to part[tok][b][h][0..513]
    float* p_tok = part + (size_t) tok * (size_t) s_max * 32 * kMlaPart;
    const int o_row = lane >> 1;
    const int o_col = (lane & 1) << 3;
    const int h_out = hg * 16 + o_row;
    float* dst_row = p_tok + ((size_t) b * 32 + h_out) * kMlaPart;
    float* wt = sTmp + w * 256;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        wmma::store_matrix_sync(wt, o_acc[j], 16, wmma::mem_row_major);
        __syncwarp();
        const float4* src4 = (const float4*) (wt + o_row * 16 + o_col);
        float4* dst4 = (float4*) (dst_row + (ks * 8 + j) * 16 + o_col);
        dst4[0] = src4[0];
        dst4[1] = src4[1];
    }
    if (ks == 0 && (lane & 1) == 0) {
        dst_row[kMlaLora] = sM[h_out];
        dst_row[kMlaLora + 1] = sSum[h_out];
    }
#else
    (void) q_abs; (void) q; (void) q_stride; (void) q_pe_off; (void) lat; (void) k_pe;
    (void) n_sel_cap; (void) n_sel_dev; (void) T; (void) scale; (void) s_max; (void) part;
#endif
}

__global__ void __launch_bounds__(256)
    mla_dec_wmma_combine_kernel(const float* __restrict__ part, int n_sel_cap, const int* __restrict__ n_sel_dev, int T,
                                int n_head, int tile_sz, int s_max, float* __restrict__ ctx) {
    __shared__ float s_w[kMlaMaxSplits1];
    __shared__ float s_inv;
    const int h = blockIdx.x;
    const int tok = blockIdx.y;
    const int tid = threadIdx.x;
    const int n_base = n_sel_dev != nullptr ? min(n_sel_cap, *n_sel_dev) : n_sel_cap;
    const int n = n_base - (T - 1 - tok);
    const int S = n > 0 ? mla_num_splits(n, tile_sz, s_max) : 0;
    const float* p_tok = part + (size_t) tok * (size_t) s_max * n_head * kMlaPart;
    float* out_row = ctx + ((size_t) tok * n_head + h) * kMlaLora;
    if (S <= 0) {
        for (int c = tid; c < kMlaLora; c += 256) out_row[c] = 0.0f;
        return;
    }
    if (tid < 32) {
        float mb[4], lb[4], M = -INFINITY;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int b = tid + 32 * k;
            const float* pb = p_tok + ((size_t) b * n_head + h) * kMlaPart;
            mb[k] = b < S ? pb[kMlaLora] : -INFINITY;
            lb[k] = b < S ? pb[kMlaLora + 1] : 0.0f;
            M = fmaxf(M, mb[k]);
        }
        M = warp_max_f(M);
        float L = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float w = mb[k] == -INFINITY ? 0.0f : expf(mb[k] - M);
            s_w[tid + 32 * k] = w;
            L += w * lb[k];
        }
        L = warp_sum_f(L);
        if (tid == 0) s_inv = L > 0.0f ? 1.0f / L : 0.0f;
    }
    __syncthreads();
    for (int c = tid; c < kMlaLora; c += 256) {
        float acc = 0.0f;
        for (int b = 0; b < S; ++b) acc += s_w[b] * p_tok[((size_t) b * n_head + h) * kMlaPart + c];
        out_row[c] = acc * s_inv;
    }
}

static int mla_dec_dev_sms() {
    int dev = 0;
    cudaGetDevice(&dev);
    static int sms[64] = {};
    int& v = sms[dev & 63];
    if (v != 0) return v;
    int mp = 0;
    cudaDeviceGetAttribute(&mp, cudaDevAttrMultiProcessorCount, dev);
    v = mp > 0 ? mp : 64;
    return v;
}

// Returns 32 (TILE=32 supported), 16 (TILE=16 supported), or 0 (unsupported)
template <int LQ8>
static int mla_dec_wmma_tile_ok() {
#if defined(MAYA_USE_HIP)
    // AMD: no nvcuda::wmma (the kernel's body is compiled out, and HIP reports a compute capability such as 11.5 that
    // would pass the checks below) - the decode attention takes the portable split kernel (glmd::mla_dec)
    return 0;
#endif
    int dev = 0;
    cudaGetDevice(&dev);
    static int checked[64] = {};
    int& st = checked[dev & 63];
    if (st != 0) return st < 0 ? 0 : st;
    int cc_maj = 0, optin = 0;
    cudaDeviceGetAttribute(&cc_maj, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (cc_maj >= 8 && (size_t) optin >= mla_wmma_smem_bytes<32>()) {
        if (cudaFuncSetAttribute(mla_dec_wmma_kernel<LQ8, 32>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int) mla_wmma_smem_bytes<32>()) == cudaSuccess) {
            st = 32;
            return 32;
        }
        cudaGetLastError();
    }
    if (cc_maj >= 7 && (size_t) optin >= mla_wmma_smem_bytes<16>()) {
        if (cudaFuncSetAttribute(mla_dec_wmma_kernel<LQ8, 16>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int) mla_wmma_smem_bytes<16>()) == cudaSuccess) {
            st = 16;
            return 16;
        }
        cudaGetLastError();
    }
    st = -1;
    return 0;
}

// ---- Windowed KDA prep + rollback + register-resident recurrence + exact commit (W <= 4) ----
__global__ void __launch_bounds__(128)
    kda_prep_window_kernel(const float* __restrict__ q_in, const float* __restrict__ k_in,
                           const float* __restrict__ v_in, const float* __restrict__ f_in,
                           const float* __restrict__ conv_q, const float* __restrict__ conv_k,
                           const float* __restrict__ conv_v, float* __restrict__ conv_state,
                           float* __restrict__ conv_bak, const float* __restrict__ ssm_a, float lb, float eps, int W,
                           int n_head, int head_dim, int d_conv, float* __restrict__ q_out, float* __restrict__ k_out,
                           float* __restrict__ v_out, float* __restrict__ g1_out) {
    const int h = blockIdx.x;
    const int role = blockIdx.y;
    const int tid = threadIdx.x;
    const int d_inner = n_head * head_dim;
    if (tid >= head_dim) return;
    const int c = h * head_dim + tid;

    if (role == 3) {
        const float sa = ssm_a[h];
        for (int t = 0; t < W; ++t) {
            const float v = f_in[(size_t) t * d_inner + c] * sa;
            g1_out[(size_t) t * d_inner + c] = lb * sigmoid_exact(v);
        }
        return;
    }

    const float* in = role == 0 ? q_in : (role == 1 ? k_in : v_in);
    const float* cw = role == 0 ? conv_q : (role == 1 ? conv_k : conv_v);
    float* out = role == 0 ? q_out : (role == 1 ? k_out : v_out);
    const int hist = d_conv - 1;
    const size_t st_off = (size_t) role * d_inner * hist + (size_t) hist * c;
    float* st = conv_state + st_off;
    const float* w = cw + (size_t) d_conv * c;

    float sh[4];
    for (int tap = 0; tap < hist; ++tap) {
        sh[tap] = st[tap];
        if (conv_bak != nullptr) conv_bak[st_off + tap] = sh[tap];
    }
    float wt[4];
    for (int tap = 0; tap < d_conv; ++tap) wt[tap] = w[tap];

    __shared__ double ssum[4];
    for (int t = 0; t < W; ++t) {
        const float xp = in[(size_t) t * d_inner + c];
        float acc = 0.0f;
        for (int tap = 0; tap < hist; ++tap) acc += wt[tap] * sh[tap];
        acc += wt[hist] * xp;
        float y = acc * sigmoid_exact(acc);
        for (int j = 0; j + 1 < hist; ++j) sh[j] = sh[j + 1];
        sh[hist - 1] = xp;

        if (role < 2) {
            const double wv = warp_sum_d((double) y * (double) y);
            if ((tid & 31) == 0) ssum[tid >> 5] = wv;
            __syncthreads();
            const double total = ssum[0] + ssum[1] + ssum[2] + ssum[3];
            y *= 1.0f / fmaxf(sqrtf((float) total), eps);
            __syncthreads();
        }
        out[(size_t) t * d_inner + c] = y;
    }
    for (int tap = 0; tap < hist; ++tap) st[tap] = sh[tap];
}

__global__ void __launch_bounds__(128)
    kda_conv_rollback_kernel(const float* __restrict__ conv_bak, const float* __restrict__ q_in,
                             const float* __restrict__ k_in, const float* __restrict__ v_in, int a, int n_head,
                             int head_dim, int d_conv, float* __restrict__ conv_state) {
    const int h = blockIdx.x;
    const int role = blockIdx.y;
    const int tid = threadIdx.x;
    const int d_inner = n_head * head_dim;
    if (tid >= head_dim) return;
    const int c = h * head_dim + tid;
    const int hist = d_conv - 1;
    const size_t st_off = (size_t) role * d_inner * hist + (size_t) hist * c;
    const float* in = role == 0 ? q_in : (role == 1 ? k_in : v_in);

    float sh[4];
    for (int tap = 0; tap < hist; ++tap) sh[tap] = conv_bak[st_off + tap];
    for (int t = 0; t < a; ++t) {
        for (int j = 0; j + 1 < hist; ++j) sh[j] = sh[j + 1];
        sh[hist - 1] = in[(size_t) t * d_inner + c];
    }
    for (int tap = 0; tap < hist; ++tap) conv_state[st_off + tap] = sh[tap];
}

__global__ void __launch_bounds__(256)
    kda_rec_window_q8_kernel(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                             const float* __restrict__ g1, const float* __restrict__ beta_raw,
                             const float* __restrict__ state, float* __restrict__ state_tmp,
                             const float* __restrict__ g2, const float* __restrict__ norm_w, float eps, int W,
                             int n_head, int8_t* __restrict__ xq_out, float* __restrict__ xd_out) {
    constexpr int HD = 128;
    const int h = blockIdx.x;
    const int t_idx = threadIdx.x;
    const int j = t_idx & (HD - 1);
    const int half = t_idx >> 7;
    const int DI = n_head * HD;

    __shared__ float sq[HD], sk[HD], sdec[HD], sdel[HD];
    __shared__ float sacc[2][HD];
    __shared__ float sred[4];

    const float* Sh = state + (size_t) h * HD * HD;
    float reg[64];
#pragma unroll
    for (int ii = 0; ii < 64; ++ii) reg[ii] = Sh[(half * 64 + ii) * HD + j];

    for (int step = 0; step < W; ++step) {
        const size_t base = (size_t) step * DI + (size_t) h * HD;
        if (t_idx < HD) {
            sq[t_idx] = q[base + t_idx];
            sk[t_idx] = k[base + t_idx];
            sdec[t_idx] = expf(g1[base + t_idx]);
        }
        const float b = sigmoid_exact(beta_raw[(size_t) step * n_head + h]);
        __syncthreads();

        float A = 0.0f;
#pragma unroll
        for (int ii = 0; ii < 64; ++ii) {
            const int i = half * 64 + ii;
            const float sv = reg[ii] * sdec[i];
            reg[ii] = sv;
            A += sv * sk[i];
        }
        sacc[half][j] = A;
        __syncthreads();

        if (half == 0) sdel[j] = b * (v[base + j] - (sacc[0][j] + sacc[1][j]));
        __syncthreads();

        const float dl = sdel[j];
        float O = 0.0f;
#pragma unroll
        for (int ii = 0; ii < 64; ++ii) {
            const int i = half * 64 + ii;
            const float s2 = reg[ii] + sk[i] * dl;
            reg[ii] = s2;
            O += s2 * sq[i];
        }
        sacc[half][j] = O;
        __syncthreads();

        float o_val = 0.0f;
        if (half == 0) {
            o_val = (sacc[0][j] + sacc[1][j]) * rsqrtf((float) HD);
            const float wsum = warp_sum_f(o_val * o_val);
            if ((j & 31) == 0) sred[j >> 5] = wsum;
        }
        __syncthreads();
        if (half == 0) {
            const float ss = sred[0] + sred[1] + sred[2] + sred[3];
            const float inv = rsqrtf(ss / (float) HD + eps);
            const float gv = o_val * inv * norm_w[j] * sigmoid_exact(g2[base + j]);
            q8_store_warp(gv, base + j, xq_out, xd_out);
        }
        __syncthreads();
    }

    float* Sd = state_tmp + (size_t) h * HD * HD;
#pragma unroll
    for (int ii = 0; ii < 64; ++ii) Sd[(half * 64 + ii) * HD + j] = reg[ii];
}

__global__ void __launch_bounds__(256)
    kda_replay_state_kernel(const float* __restrict__ k, const float* __restrict__ v, const float* __restrict__ g1,
                            const float* __restrict__ beta_raw, float* __restrict__ state, int a, int n_head) {
    constexpr int HD = 128;
    const int h = blockIdx.x;
    const int t_idx = threadIdx.x;
    const int j = t_idx & (HD - 1);
    const int half = t_idx >> 7;
    const int DI = n_head * HD;

    __shared__ float sk[HD], sdec[HD], sdel[HD];
    __shared__ float sacc[2][HD];

    float* Sh = state + (size_t) h * HD * HD;
    float reg[64];
#pragma unroll
    for (int ii = 0; ii < 64; ++ii) reg[ii] = Sh[(half * 64 + ii) * HD + j];

    for (int step = 0; step < a; ++step) {
        const size_t base = (size_t) step * DI + (size_t) h * HD;
        if (t_idx < HD) {
            sk[t_idx] = k[base + t_idx];
            sdec[t_idx] = expf(g1[base + t_idx]);
        }
        const float b = sigmoid_exact(beta_raw[(size_t) step * n_head + h]);
        __syncthreads();

        float A = 0.0f;
#pragma unroll
        for (int ii = 0; ii < 64; ++ii) {
            const int i = half * 64 + ii;
            const float sv = reg[ii] * sdec[i];
            reg[ii] = sv;
            A += sv * sk[i];
        }
        sacc[half][j] = A;
        __syncthreads();

        if (half == 0) sdel[j] = b * (v[base + j] - (sacc[0][j] + sacc[1][j]));
        __syncthreads();

        const float dl = sdel[j];
#pragma unroll
        for (int ii = 0; ii < 64; ++ii) {
            const int i = half * 64 + ii;
            reg[ii] += sk[i] * dl;
        }
        __syncthreads();
    }
#pragma unroll
    for (int ii = 0; ii < 64; ++ii) Sh[(half * 64 + ii) * HD + j] = reg[ii];
}

__global__ void __launch_bounds__(32)
    pack_route_win_cpu_kernel(const int* __restrict__ ids, const float* __restrict__ w, const int* __restrict__ who,
                              int n_ent, RoutePackWin* __restrict__ dst) {
    const int i = threadIdx.x;
    if (i < n_ent) {
        dst->ids[i] = ids[i];
        dst->w[i] = w[i];
        dst->who[i] = who[i];
    }
}

// 2-stage on-device top-1 + softmax probability (grid 128 -> grid 1; scratch_val needs 256 floats: [0..127] max, [128..255] sumexp)
__global__ void __launch_bounds__(256)
    top1_prob_s1_kernel(const float* __restrict__ x, int n, float* __restrict__ pval, int* __restrict__ pidx) {
    __shared__ float sm[8], ss[8];
    __shared__ int si[8];
    const int tid = threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    float bm = -INFINITY;
    int bi = 0;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) {
        const float v = x[i];
        if (v > bm) {
            bm = v;
            bi = i;
        }
    }
    // Online softmax sumexp alongside max inside each thread pass? Two-pass over registers is unnecessary; compute sumexp vs local max
    float bm_w = bm;
    int bi_w = bi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float om = __shfl_xor_sync(0xffffffffu, bm_w, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi_w, o);
        if (om > bm_w || (om == bm_w && oi < bi_w)) {
            bm_w = om;
            bi_w = oi;
        }
    }
    if ((tid & 31) == 0) {
        sm[tid >> 5] = bm_w;
        si[tid >> 5] = bi_w;
    }
    __syncthreads();
    if (tid == 0) {
        float m = sm[0];
        int id = si[0];
        for (int k = 1; k < 8; ++k) {
            if (sm[k] > m || (sm[k] == m && si[k] < id)) {
                m = sm[k];
                id = si[k];
            }
        }
        sm[0] = m;
        si[0] = id;
    }
    __syncthreads();
    const float bmax = sm[0];
    float lsum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) {
        lsum += expf(x[i] - bmax);
    }
    lsum = warp_sum_f(lsum);
    if ((tid & 31) == 0) ss[tid >> 5] = lsum;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int k = 0; k < 8; ++k) tot += ss[k];
        pval[blockIdx.x] = bmax;
        pval[gridDim.x + blockIdx.x] = tot;
        pidx[blockIdx.x] = si[0];
    }
}

__global__ void __launch_bounds__(128)
    top1_prob_s2_kernel(const float* __restrict__ pval, const int* __restrict__ pidx, int nb, float* __restrict__ out) {
    __shared__ float sm[4], ss[4];
    __shared__ int si[4];
    const int tid = threadIdx.x;
    float bm = tid < nb ? pval[tid] : -INFINITY;
    int bi = tid < nb ? pidx[tid] : 0;
    float bs = tid < nb ? pval[nb + tid] : 0.0f;
    float wm = bm;
    int wi = bi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float om = __shfl_xor_sync(0xffffffffu, wm, o);
        const int oi = __shfl_xor_sync(0xffffffffu, wi, o);
        if (om > wm || (om == wm && oi < wi)) {
            wm = om;
            wi = oi;
        }
    }
    if ((tid & 31) == 0) {
        sm[tid >> 5] = wm;
        si[tid >> 5] = wi;
    }
    __syncthreads();
    if (tid == 0) {
        float m = sm[0];
        int id = si[0];
        for (int k = 1; k < 4; ++k) {
            if (sm[k] > m || (sm[k] == m && si[k] < id)) {
                m = sm[k];
                id = si[k];
            }
        }
        sm[0] = m;
        si[0] = id;
    }
    __syncthreads();
    const float gmax = sm[0];
    float adj = tid < nb ? bs * expf(bm - gmax) : 0.0f;
    adj = warp_sum_f(adj);
    if ((tid & 31) == 0) ss[tid >> 5] = adj;
    __syncthreads();
    if (tid == 0) {
        float tot = ss[0] + ss[1] + ss[2] + ss[3];
        const int best_id = si[0];
        out[0] = __int_as_float(best_id);
        out[1] = tot > 0.0f ? (1.0f / tot) : 0.0f;
    }
}

__global__ void set_step_kernel(int* __restrict__ d_step, int tok, int pos, int n_kv, int rope_delta) {
    d_step[0] = tok;
    d_step[1] = pos;
    d_step[2] = n_kv;
    d_step[3] = rope_delta;
}

__global__ void set_step_win_kernel(int* __restrict__ d_step, int* __restrict__ d_tok, int* __restrict__ d_pos,
                                    int t0, int t1, int t2, int t3, int W, int pos0, int rope_delta) {
    d_step[0] = t0;
    d_step[1] = pos0;
    d_step[2] = pos0 + W;
    d_step[3] = rope_delta;
    const int ts[4] = {t0, t1, t2, t3};
    for (int i = 0; i < W; ++i) {
        d_tok[i] = ts[i];
        d_pos[i] = pos0 + i;
#if defined(MAYA_USE_HIP)
        d_step[kStepWinCounts + i] = pos0 + i + 1;   // (AMD: each window row's own cache length - mla_decode_win)
#endif
    }
}

__global__ void __launch_bounds__(256)
    store_hring_kernel(const float* __restrict__ src, int E, float* __restrict__ ring, int R, int pos0) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < E) {
        const int slot = (pos0 + t) % R;
        ring[(size_t) slot * E + i] = src[(size_t) t * E + i];
    }
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
                    bool w_norm, float w_scale, int* ids_out, float* w_out, cudaStream_t s, int* err_flag) {
    if (T <= 0) return;
    router_grouped_kernel<<<(unsigned) T, 256, 0, s>>>(logits, bias, n_exp, n_used, n_group, n_group_used,
                                                       w_norm ? 1 : 0, w_scale, ids_out, w_out, err_flag);
}

void kv_store_split(const float* kv_in, int T, int kv_lora, int n_rot, const float* kv_a_norm, float eps,
                    const int* pos_dev, int p0, float base, void* lat, bool lat_q8, uint16_t* k_pe, cudaStream_t s,
                    const int* rope_delta_dev, const int* mrope_pos, int mrope_hw, int mrope_stride) {
    if (T <= 0) return;
    kv_store_split_kernel<<<(unsigned) T, 256, 0, s>>>(kv_in, kv_lora, n_rot, kv_a_norm, eps, pos_dev, p0, base, lat,
                                                       lat_q8 ? 1 : 0, k_pe, rope_delta_dev, mrope_pos, mrope_hw,
                                                       mrope_stride > 0 ? mrope_stride : T);
}

void mla_q_prep(float* qm, int T, int n_head, int qk_head, int qk_nope, int n_rot, const int* pos_dev, int p0,
                float base, int8_t* q8n, float* d8n, uint16_t* qm16, cudaStream_t s, const int* rope_delta_dev,
                const int* mrope_pos, int mrope_hw, int mrope_stride) {
    if (T <= 0) return;
    mla_q_prep_kernel<<<dim3((unsigned) n_head, (unsigned) T), 256, 0, s>>>(
        qm, T, n_head, qk_head, qk_nope, n_rot, pos_dev, p0, base, q8n, d8n, qm16, rope_delta_dev, mrope_pos, mrope_hw,
        mrope_stride > 0 ? mrope_stride : T);
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
    else if (type == 2) emb_rows_q4_0_kernel<<<grid, 256, 0, s>>>(w, k, ids_dev, out);
    else if (type == 12) emb_rows_q4_k_kernel<<<grid, 256, 0, s>>>(w, k, ids_dev, out);
    else if (type == 13) emb_rows_q5_k_kernel<<<grid, 256, 0, s>>>(w, k, ids_dev, out);
    else if (type == 14) emb_rows_q6_k_kernel<<<grid, 256, 0, s>>>(w, k, ids_dev, out);
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

void mla_q_split_rope_f16(const float* qm, int T, int n_head, int qk_head, int qk_nope, int n_rot, int pos0, float base,
                          uint16_t* q_nope16, uint16_t* q_pe16, cudaStream_t s, const int* mrope_pos, int mrope_hw,
                          int mrope_stride) {
    if (T <= 0) return;
    mla_q_split_rope_f16_kernel<<<dim3((unsigned) n_head, (unsigned) T), 256, 0, s>>>(
        qm, T, n_head, qk_head, qk_nope, n_rot, pos0, base, q_nope16, q_pe16, mrope_pos, mrope_hw,
        mrope_stride > 0 ? mrope_stride : T);
}

void lat_q8_to_f16(const void* lat, int k0, int BK, int kv_lora, uint16_t* out16, cudaStream_t s) {
    if (BK <= 0) return;
    lat_q8_to_f16_kernel<<<(unsigned) BK, 256, 0, s>>>((const uint8_t*) lat, k0, kv_lora, out16);
}

void mla_init_state(float* m, float* l, float* o_acc, int H, int T, int v_head, cudaStream_t s) {
    const int64_t HT = (int64_t) H * T;
    if (HT <= 0) return;
    const int blocks = (int) std::min<int64_t>((HT * v_head + 255) / 256, 4096);
    mla_init_state_kernel<<<(unsigned) blocks, 256, 0, s>>>(m, l, o_acc, HT, v_head);
}

void mla_online_softmax(const float* S, int H, int T, int q0, int BQ, int k0, int BK, int pos0, float* m, float* l,
                        float* o_acc, int v_head, uint16_t* P16, cudaStream_t s) {
    if (BQ <= 0 || BK <= 0) return;
    mla_online_softmax_kernel<<<dim3((unsigned) BQ, (unsigned) H), 256, 0, s>>>(
        S, H, T, q0, BQ, k0, BK, pos0, m, l, o_acc, v_head, P16);
}

void mla_o_finalize_gate(const float* o_acc, const float* l, const float* gate, int T, int H, int v_head,
                         uint16_t* out16, cudaStream_t s) {
    if (T <= 0) return;
    mla_o_finalize_gate_kernel<<<dim3((unsigned) H, (unsigned) T), 128, 0, s>>>(o_acc, l, gate, T, H, v_head, out16);
}

void max_rows_from_bounds(const int* bounds, int n_slots, int* out_dev, cudaStream_t s) {
    max_rows_from_bounds_kernel<<<1, 256, 0, s>>>(bounds, n_slots, out_dev);
}

void fetch_resolve_fast(const int* ids, int T, int n_used, const hy4::k::FetchLayer& fl, const hy4::k::FetchScratch& fs,
                        int k_gpu, bool cache, float decay, cudaStream_t s) {
    fetch_resolve_fast_kernel<<<1, 512, 0, s>>>(ids, T, n_used, fl, fs, k_gpu, cache ? 1 : 0, decay);
}

void argmax_f32(const float* logits, int n, float* scratch_val, int* scratch_idx, int* out_id, cudaStream_t s) {
    constexpr int kBlocks = 128;
    argmax_stage1_kernel<<<kBlocks, 256, 0, s>>>(logits, n, scratch_val, scratch_idx);
    argmax_stage2_kernel<<<1, 128, 0, s>>>(scratch_val, scratch_idx, kBlocks, out_id);
}

void add_rmsnorm_q8(float* x, const float* a, const float* w, int T, int E, float eps, float* cur_out, int8_t* xq,
                    float* xd, cudaStream_t s) {
    if (T <= 0) return;
    add_rmsnorm_q8_kernel<1><<<(unsigned) T, 512, 0, s>>>(x, a, nullptr, w, E, eps, cur_out, xq, xd);
}

void add2_rmsnorm_q8(float* x, const float* a, const float* b, const float* w, int T, int E, float eps, float* cur_out,
                     int8_t* xq, float* xd, cudaStream_t s) {
    if (T <= 0) return;
    add_rmsnorm_q8_kernel<2><<<(unsigned) T, 512, 0, s>>>(x, a, b, w, E, eps, cur_out, xq, xd);
}

void pack_route_cpu(const int* ids, const float* w, const int* who, int n_used, RoutePack* dst, cudaStream_t s) {
    pack_route_cpu_kernel<<<1, 32, 0, s>>>(ids, w, who, n_used, dst);
}

size_t mla_dec_wmma_part_bytes(int n_head, int max_T) {
    const size_t slots = std::max<size_t>((size_t) kMlaMaxSplits1, (size_t) std::max(1, max_T) * (size_t) kMlaMaxSplitsM);
    return slots * (size_t) n_head * (size_t) kMlaPart * sizeof(float);
}

void mla_dec_wmma_init() {
    (void) mla_dec_dev_sms();
    (void) mla_dec_wmma_tile_ok<0>();
    (void) mla_dec_wmma_tile_ok<1>();
}

bool mla_dec_wmma(const float* q_abs, const float* q, int q_stride, int q_pe_off, const void* lat, int lat_q8,
                  const uint16_t* k_pe, int n_sel_cap, const int* n_sel_dev, int T, int n_head, int kv_lora,
                  int qk_rope, float scale, float* part, float* ctx, cudaStream_t s) {
    if (T <= 0 || n_head != 32 || kv_lora != kMlaLora || qk_rope != 64) return false;
    const int tile = lat_q8 != 0 ? mla_dec_wmma_tile_ok<1>() : mla_dec_wmma_tile_ok<0>();
    if (tile == 0) return false;
    const int sms = mla_dec_dev_sms();
    const int s_max = std::min(kMlaMaxSplitsM, std::max(1, sms / 4));
    const dim3 grid((unsigned) s_max, (unsigned) T);
    if (tile == 32) {
        if (lat_q8 != 0)
            mla_dec_wmma_kernel<1, 32><<<grid, 256, mla_wmma_smem_bytes<32>(), s>>>(
                q_abs, q, q_stride, q_pe_off, lat, k_pe, n_sel_cap, n_sel_dev, T, scale, s_max, part);
        else
            mla_dec_wmma_kernel<0, 32><<<grid, 256, mla_wmma_smem_bytes<32>(), s>>>(
                q_abs, q, q_stride, q_pe_off, lat, k_pe, n_sel_cap, n_sel_dev, T, scale, s_max, part);
    } else {
        if (lat_q8 != 0)
            mla_dec_wmma_kernel<1, 16><<<grid, 256, mla_wmma_smem_bytes<16>(), s>>>(
                q_abs, q, q_stride, q_pe_off, lat, k_pe, n_sel_cap, n_sel_dev, T, scale, s_max, part);
        else
            mla_dec_wmma_kernel<0, 16><<<grid, 256, mla_wmma_smem_bytes<16>(), s>>>(
                q_abs, q, q_stride, q_pe_off, lat, k_pe, n_sel_cap, n_sel_dev, T, scale, s_max, part);
    }
    mla_dec_wmma_combine_kernel<<<dim3((unsigned) n_head, (unsigned) T), 256, 0, s>>>(part, n_sel_cap, n_sel_dev, T,
                                                                                      n_head, tile, s_max, ctx);
    return true;
}


void kda_prep_window(const float* q_in, const float* k_in, const float* v_in, const float* f_in, const float* conv_q,
                     const float* conv_k, const float* conv_v, float* conv_state, float* conv_bak, const float* ssm_a,
                     float lb, float eps, int W, int n_head, int head_dim, int d_conv, float* q_out, float* k_out,
                     float* v_out, float* g1_out, cudaStream_t s) {
    if (W <= 0) return;
    kda_prep_window_kernel<<<dim3((unsigned) n_head, 4u), 128, 0, s>>>(
        q_in, k_in, v_in, f_in, conv_q, conv_k, conv_v, conv_state, conv_bak, ssm_a, lb, eps, W, n_head, head_dim,
        d_conv, q_out, k_out, v_out, g1_out);
}

void kda_conv_rollback(const float* conv_bak, const float* q_in, const float* k_in, const float* v_in, int a,
                       int n_head, int head_dim, int d_conv, float* conv_state, cudaStream_t s) {
    kda_conv_rollback_kernel<<<dim3((unsigned) n_head, 3u), 128, 0, s>>>(
        conv_bak, q_in, k_in, v_in, a, n_head, head_dim, d_conv, conv_state);
}

void kda_rec_window_q8(const float* q, const float* k, const float* v, const float* g1, const float* beta_raw,
                       const float* state, float* state_tmp, const float* g2, const float* norm_w, float eps, int W,
                       int n_head, int head_dim, int8_t* xq_out, float* xd_out, cudaStream_t s) {
    (void) head_dim;
    if (W <= 0) return;
    kda_rec_window_q8_kernel<<<(unsigned) n_head, 256, 0, s>>>(
        q, k, v, g1, beta_raw, state, state_tmp, g2, norm_w, eps, W, n_head, xq_out, xd_out);
}

void kda_commit_window(const float* k, const float* v, const float* g1, const float* beta_raw, float* state,
                       const float* state_tmp, int a, int W, int n_head, int head_dim, cudaStream_t s) {
    if (a == W) {
        const size_t bytes = (size_t) n_head * (size_t) head_dim * (size_t) head_dim * sizeof(float);
        cudaMemcpyAsync(state, state_tmp, bytes, cudaMemcpyDeviceToDevice, s);
        return;
    }
    kda_replay_state_kernel<<<(unsigned) n_head, 256, 0, s>>>(k, v, g1, beta_raw, state, a, n_head);
}

void pack_route_win_cpu(const int* ids, const float* w, const int* who, int n_ent, RoutePackWin* dst, cudaStream_t s) {
    pack_route_win_cpu_kernel<<<1, 32, 0, s>>>(ids, w, who, n_ent, dst);
}

void top1_prob_f32(const float* logits, int n, float* scratch_val, int* scratch_idx, float* out_pair, cudaStream_t s) {
    constexpr int kBlocks = 128;
    top1_prob_s1_kernel<<<kBlocks, 256, 0, s>>>(logits, n, scratch_val, scratch_idx);
    top1_prob_s2_kernel<<<1, 128, 0, s>>>(scratch_val, scratch_idx, kBlocks, out_pair);
}

void argmax_rows_f32(const float* logits, int W, int n, float* scratch_val, int* scratch_idx, int* out_ids,
                     cudaStream_t s) {
    constexpr int kBlocks = 128;
    for (int t = 0; t < W; ++t) {
        argmax_stage1_kernel<<<kBlocks, 256, 0, s>>>(logits + (size_t) t * n, n, scratch_val, scratch_idx);
        argmax_stage2_kernel<<<1, 128, 0, s>>>(scratch_val, scratch_idx, kBlocks, out_ids + t);
    }
}

void store_hring(const float* src, int W, int E, float* ring, int R, int pos0, cudaStream_t s) {
    if (W <= 0) return;
    store_hring_kernel<<<dim3((unsigned) ((E + 255) / 256), (unsigned) W), 256, 0, s>>>(src, E, ring, R, pos0);
}

void set_step(int* d_step, int tok, int pos, int n_kv, int rope_delta, cudaStream_t s) {
    set_step_kernel<<<1, 1, 0, s>>>(d_step, tok, pos, n_kv, rope_delta);
}

void set_step_win(int* d_step, int* d_tok, int* d_pos, int t0, int t1, int t2, int t3, int W, int pos0, int rope_delta,
                  cudaStream_t s) {
    set_step_win_kernel<<<1, 1, 0, s>>>(d_step, d_tok, d_pos, t0, t1, t2, t3, W, pos0, rope_delta);
}

void q8k_rows(const float* x, int rows, int k, int8_t* xq, float* xd, cudaStream_t s) {
    if (rows <= 0 || k <= 0) return;
    q8k_rows_kernel<<<dim3((unsigned) (k / 256), (unsigned) rows), 256, 0, s>>>(x, k, xq, xd);
}

namespace {
__global__ void ref_add_kernel(float* x, const float* a, const float* b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] + a[i] + (b ? b[i] : 0.0f);
}
__global__ void ref_rmsnorm_q8_kernel(const float* x, const float* w, int E, float eps, float* cur_out, int8_t* xq,
                                      float* xd) {
    __shared__ float s_sum[16];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    float xv[8], wv[8];
    float ss = 0.0f;
    const int n_iter = (E + 511) >> 9;
    for (int j = 0; j < n_iter; ++j) {
        const int i = tid + j * 512;
        if (i < E) {
            xv[j] = x[i];
            wv[j] = w[i];
            ss += xv[j] * xv[j];
        }
    }
    ss = warp_sum_f(ss);
    if (lane == 0) s_sum[warp] = ss;
    __syncthreads();
    if (tid < 16) {
        float v = s_sum[tid];
        for (int o = 8; o > 0; o >>= 1) v += lanes16_xor_f(v, o);
        if (tid == 0) s_sum[0] = rsqrtf(v / (float) E + eps);
    }
    __syncthreads();
    const float inv = s_sum[0];
    for (int j = 0; j < n_iter; ++j) {
        const int i = tid + j * 512;
        if (i < E) {
            const float o = xv[j] * inv * wv[j];
            if (cur_out) cur_out[i] = o;
            q8_store_warp(o, (size_t) i, xq, xd);
        }
    }
}
__global__ void ref_swiglu_unfused_kernel(const float* g, const float* u, int n, float limit, float* h_out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float act = g[i] * sigmoid_exact(g[i]);
    float uv = u[i];
    if (limit > 0.0f) {
        if (act > limit) act = limit;
        uv = fminf(fmaxf(uv, -limit), limit);
    }
    h_out[i] = act * uv;
}
__global__ void ref_q8_kernel(const float* h, int n, int8_t* xq, float* xd) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) q8_store_warp(h[i], (size_t) i, xq, xd);
}
}  // namespace

bool verify_fused_kernels(cudaStream_t s, std::string& err) {
    constexpr int E = 2560, FE = 768;
    const float eps = 1e-6f;
    float *dx1 = nullptr, *dx2 = nullptr, *da = nullptr, *db = nullptr, *dw = nullptr;
    float *dc1 = nullptr, *dc2 = nullptr, *dd1 = nullptr, *dd2 = nullptr;
    int8_t *dq1 = nullptr, *dq2 = nullptr;
    auto cleanup = [&] {
        cudaFree(dx1); cudaFree(dx2); cudaFree(da); cudaFree(db); cudaFree(dw);
        cudaFree(dc1); cudaFree(dc2); cudaFree(dd1); cudaFree(dd2);
        cudaFree(dq1); cudaFree(dq2);
    };
    auto alloc = [&](void** p, size_t n) -> bool { return cudaMalloc(p, n) == cudaSuccess; };
    if (!alloc((void**) &dx1, E * 4) || !alloc((void**) &dx2, E * 4) || !alloc((void**) &da, E * 4) ||
        !alloc((void**) &db, E * 4) || !alloc((void**) &dw, E * 4) || !alloc((void**) &dc1, E * 4) ||
        !alloc((void**) &dc2, E * 4) || !alloc((void**) &dd1, (E / 32) * 4) || !alloc((void**) &dd2, (E / 32) * 4) ||
        !alloc((void**) &dq1, E) || !alloc((void**) &dq2, E)) {
        cleanup();
        err = "verify_fused_kernels: cudaMalloc failed";
        return false;
    }
    std::vector<float> hx(E), ha(E), hb(E), hw(E);
    for (int i = 0; i < E; ++i) {
        hx[i] = sinf((float) i * 0.13f) * 2.5f;
        ha[i] = cosf((float) i * 0.27f) * 1.7f;
        hb[i] = sinf((float) i * 0.41f + 0.5f) * 0.9f;
        hw[i] = 0.8f + 0.4f * cosf((float) i * 0.09f);
    }
    cudaMemcpyAsync(dx1, hx.data(), E * 4, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(dx2, hx.data(), E * 4, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(da, ha.data(), E * 4, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(db, hb.data(), E * 4, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(dw, hw.data(), E * 4, cudaMemcpyHostToDevice, s);

    // 1. add_rmsnorm_q8 vs unfused add + rmsnorm_q8
    add_rmsnorm_q8(dx1, da, dw, 1, E, eps, dc1, dq1, dd1, s);
    ref_add_kernel<<<(E + 255) / 256, 256, 0, s>>>(dx2, da, nullptr, E);
    ref_rmsnorm_q8_kernel<<<1, 512, 0, s>>>(dx2, dw, E, eps, dc2, dq2, dd2);
    std::vector<float> rx1(E), rx2(E), rc1(E), rc2(E), rd1(E / 32), rd2(E / 32);
    std::vector<int8_t> rq1(E), rq2(E);
    cudaMemcpyAsync(rx1.data(), dx1, E * 4, cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(rx2.data(), dx2, E * 4, cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(rc1.data(), dc1, E * 4, cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(rc2.data(), dc2, E * 4, cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(rd1.data(), dd1, (E / 32) * 4, cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(rd2.data(), dd2, (E / 32) * 4, cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(rq1.data(), dq1, E, cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(rq2.data(), dq2, E, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    if (std::memcmp(rx1.data(), rx2.data(), E * 4) != 0 || std::memcmp(rc1.data(), rc2.data(), E * 4) != 0 ||
        std::memcmp(rq1.data(), rq2.data(), E) != 0 || std::memcmp(rd1.data(), rd2.data(), (E / 32) * 4) != 0) {
        cleanup();
        err = "verify_fused_kernels: add_rmsnorm_q8 bit mismatch vs unfused reference";
        return false;
    }

    // 2. add2_rmsnorm_q8 vs unfused add2 + rmsnorm_q8
    cudaMemcpyAsync(dx1, hx.data(), E * 4, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(dx2, hx.data(), E * 4, cudaMemcpyHostToDevice, s);
    add2_rmsnorm_q8(dx1, da, db, dw, 1, E, eps, dc1, dq1, dd1, s);
    ref_add_kernel<<<(E + 255) / 256, 256, 0, s>>>(dx2, da, db, E);
    ref_rmsnorm_q8_kernel<<<1, 512, 0, s>>>(dx2, dw, E, eps, dc2, dq2, dd2);
    cudaMemcpyAsync(rx1.data(), dx1, E * 4, cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(rx2.data(), dx2, E * 4, cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(rc1.data(), dc1, E * 4, cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(rc2.data(), dc2, E * 4, cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(rd1.data(), dd1, (E / 32) * 4, cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(rd2.data(), dd2, (E / 32) * 4, cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(rq1.data(), dq1, E, cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(rq2.data(), dq2, E, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    if (std::memcmp(rx1.data(), rx2.data(), E * 4) != 0 || std::memcmp(rc1.data(), rc2.data(), E * 4) != 0 ||
        std::memcmp(rq1.data(), rq2.data(), E) != 0 || std::memcmp(rd1.data(), rd2.data(), (E / 32) * 4) != 0) {
        cleanup();
        err = "verify_fused_kernels: add2_rmsnorm_q8 bit mismatch vs unfused reference";
        return false;
    }

    // 3. swiglu_clamp_out vs unfused swiglu + q8_store
    swiglu_clamp_out(da, db, 1, FE, 5.0f, dc1, dq1, dd1, nullptr, s);
    ref_swiglu_unfused_kernel<<<(FE + 255) / 256, 256, 0, s>>>(da, db, FE, 5.0f, dc2);
    ref_q8_kernel<<<(FE + 255) / 256, 256, 0, s>>>(dc2, FE, dq2, dd2);
    cudaMemcpyAsync(rc1.data(), dc1, FE * 4, cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(rc2.data(), dc2, FE * 4, cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(rd1.data(), dd1, (FE / 32) * 4, cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(rd2.data(), dd2, (FE / 32) * 4, cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(rq1.data(), dq1, FE, cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(rq2.data(), dq2, FE, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    if (std::memcmp(rc1.data(), rc2.data(), FE * 4) != 0 || std::memcmp(rq1.data(), rq2.data(), FE) != 0 ||
        std::memcmp(rd1.data(), rd2.data(), (FE / 32) * 4) != 0) {
        cleanup();
        err = "verify_fused_kernels: swiglu_clamp_out bit mismatch vs unfused reference";
        return false;
    }

    cleanup();
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace ling::k


