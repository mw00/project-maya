// src/core/glm_dsa_cpu_fast_twin.cpp - the CPU twins of the glm-dsa decode attention kernels (B4.3-B4.5).
//
// Same contracts and storage formats as glmd::dsa_prep / dsa_rope_q / mla2 / topk (glmd_fast.hpp), in float with
// the GPU kernels' algorithms: the FP16 / INT8 latent records and q8_1 blocks are bit-for-bit the formats the
// kernels write (the reductions run in a different order, so values agree to float rounding), the attention is the
// same chunked online softmax over FP16-staged cells, and the selector is the same radix cut-off plus
// position-ordered compaction, so its output is identical to the GPU's.  Plain C++, no CUDA: the CPU test links
// this file directly.
#include "strata/kernels/glmd_fast.hpp"
#include "strata/kernels/glmd_volta.hpp"

#include "../kernels/cuda/glmd_had.cuh"
#include "../kernels/cuda/glmd_rope.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace glmd::twin {

uint16_t half_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u, ax = x & 0x7fffffffu;
    if (ax >= 0x7f800000u) return (uint16_t) (sign | (ax > 0x7f800000u ? 0x7e00u : 0x7c00u));
    if (ax >= 0x477ff000u) return (uint16_t) (sign | 0x7c00u);   // >= 65520 rounds to infinity
    if (ax < 0x38800000u) {                                       // below 2^-14: subnormal (or zero)
        float v;
        std::memcpy(&v, &ax, 4);
        return (uint16_t) (sign | (uint32_t) std::nearbyint(v * 16777216.0f));   // exact product, RN-even
    }
    uint32_t h = ((ax >> 23) - 112u) << 10 | ((ax & 0x7fffffu) >> 13);
    const uint32_t rem = ax & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) (sign | h);
}

float half_float(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    uint32_t x;
    if (e == 0) {
        const float v = (float) m * (1.0f / 16777216.0f);
        std::memcpy(&x, &v, 4);
        x |= sign;
    } else if (e == 31) {
        x = sign | 0x7f800000u | (m << 13);
    } else {
        x = sign | ((e + 112u) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

uint32_t score_key(float f) {
    if (f == 0.0f) f = 0.0f;
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

namespace {

float bf16_float(uint16_t v) {
    const uint32_t x = (uint32_t) v << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

// glm_fast.cu q8_1_store_warp on 32 values
void q8_1_store(const float* v, Q81* dst) {
    float amax = 0.0f, sum = 0.0f;
    for (int l = 0; l < 32; ++l) {
        amax = std::max(amax, std::fabs(v[l]));
        sum += v[l];
    }
    const float d = std::min(amax / 127.0f, 65504.0f);
    for (int l = 0; l < 32; ++l)
        dst->qs[l] = amax == 0.0f ? 0 : (int8_t) std::max(-127.0f, std::min(127.0f, std::round(v[l] / d)));
    dst->d = half_bits(d);
    dst->s = half_bits(std::max(-65504.0f, std::min(65504.0f, sum)));
}

// Maya's INT8 latent record for one row of kv_lora values
void lat8_store(const float* y, uint8_t* rec, int kv_lora) {
    for (int g = 0; g < kv_lora / 32; ++g) {
        float amax = 0.0f;
        for (int l = 0; l < 32; ++l) amax = std::max(amax, std::fabs(y[32 * g + l]));
        const uint16_t d16 = half_bits(amax / 127.0f);
        const float d = half_float(d16);
        for (int l = 0; l < 32; ++l) {
            const int q = d > 0.0f ? std::max(-127, std::min(127, (int) std::nearbyint(y[32 * g + l] / d))) : 0;
            ((int8_t*) rec)[32 * g + l] = (int8_t) q;
        }
        std::memcpy(rec + kv_lora + 2 * g, &d16, 2);
    }
}

float lat8_value(const uint8_t* rec, int kv_lora, int c) {
    uint16_t d16;
    std::memcpy(&d16, rec + kv_lora + 2 * (c >> 5), 2);
    return (float) ((const int8_t*) rec)[c] * half_float(d16);
}

// latent value c of a cell as the split kernel stages it: FP16 (an INT8 record is decoded, then rounded to FP16)
float lat_staged(const void* lat, int lat_q8, int kv_lora, int cell, int c) {
    if (lat_q8)
        return half_float(half_bits(
            lat8_value((const uint8_t*) lat + (size_t) lat8_rec_bytes(kv_lora) * cell, kv_lora, c)));
    return half_float(((const uint16_t*) lat)[(size_t) kv_lora * cell + c]);
}

}  // namespace

void dsa_prep(const DsaPrep& a) {
    const int p = a.p_dev ? *a.p_dev : a.p;
    {   // q_a
        float ss = 0.0f;
        for (int e = 0; e < a.q_lora; ++e) ss += a.qr_raw[e] * a.qr_raw[e];
        const float inv = 1.0f / std::sqrt(ss / (float) a.q_lora + a.eps);
        std::vector<float> y(a.q_lora);
        for (int e = 0; e < a.q_lora; ++e) y[e] = a.qr_raw[e] * inv * a.q_a_norm[e];
        if (a.qr) std::copy(y.begin(), y.end(), a.qr);
        if (a.qr_q)
            for (int b = 0; b < a.q_lora / 32; ++b) q8_1_store(y.data() + 32 * b, (Q81*) a.qr_q + b);
    }
    {   // c_kv -> latent record, k_pe -> rope -> FP16
        float ss = 0.0f;
        for (int c = 0; c < a.kv_lora; ++c) ss += a.kv_raw[c] * a.kv_raw[c];
        const float inv = 1.0f / std::sqrt(ss / (float) a.kv_lora + a.eps);
        std::vector<float> y(a.kv_lora);
        for (int c = 0; c < a.kv_lora; ++c) y[c] = a.kv_raw[c] * inv * a.kv_norm[c];
        if (a.lat_q8)
            lat8_store(y.data(), (uint8_t*) a.lat + (size_t) lat8_rec_bytes(a.kv_lora) * p, a.kv_lora);
        else
            for (int c = 0; c < a.kv_lora; ++c) ((uint16_t*) a.lat)[(size_t) a.kv_lora * p + c] = half_bits(y[c]);
        std::vector<float> kp(a.kv_raw + a.kv_lora, a.kv_raw + a.kv_lora + a.qk_rope);
        rope_row(kp.data(), a.qk_rope, p, a.inv_freq);
        for (int j = 0; j < a.qk_rope; ++j) a.k_pe[(size_t) a.qk_rope * p + j] = half_bits(kp[j]);
    }
    if (a.ik_raw) {   // index key: two-pass LayerNorm, then rope dims [0, idx_rope)
        const int K = a.idx_dim;
        float mu = 0.0f;
        for (int i = 0; i < K; ++i) mu += a.ik_raw[i];
        mu /= (float) K;
        float var = 0.0f;
        for (int i = 0; i < K; ++i) var += (a.ik_raw[i] - mu) * (a.ik_raw[i] - mu);
        var /= (float) K;
        const float inv = 1.0f / std::sqrt(var + a.idx_ln_eps);
        std::vector<float> y(K);
        for (int i = 0; i < K; ++i) y[i] = (a.ik_raw[i] - mu) * inv * a.ik_ln_w[i] + a.ik_ln_b[i];
        rope_row(y.data(), a.idx_rope, p, a.idx_inv_freq);
        if (a.ik_had) had_row(y.data(), K);   // B14.11: the rotated key
        if (a.ik_f16)   // B10.8: FP16 index keys
            for (int i = 0; i < K; ++i) ((uint16_t*) a.ik_cache)[(size_t) K * p + i] = half_bits(y[i]);
        else
            std::copy(y.begin(), y.end(), a.ik_cache + (size_t) K * p);
    }
}

void dsa_rope_q(const DsaRopeQ& a) {
    const int p = a.p_dev ? *a.p_dev : a.p;
    for (int h = 0; h < a.n_head; ++h) rope_row(a.q + (size_t) a.q_stride * h + a.rot_off, a.n_rot, p, a.inv_freq);
    if (a.iq)
        for (int h = 0; h < a.idx_heads; ++h) {
            rope_row(a.iq + (size_t) a.idx_dim * h, a.idx_rope, p, a.idx_inv_freq);
            if (a.iq_had) had_row(a.iq + (size_t) a.idx_dim * h, a.idx_dim);   // B14.11
        }
}

void hadamard(float* x, int n) { had_row(x, n); }

void mla2(const Mla2& a) {
    const int H = a.n_head, L = a.kv_lora, R = a.qk_rope, CH = kMla2Chunk;
    const int n_chunks = (a.n_sel + CH - 1) / CH;
    const int n_act = a.n_sel_dev ? std::min(a.n_sel, *a.n_sel_dev) : a.n_sel;   // (host pointer here)
    std::vector<float> q_abs((size_t) H * L), ctx((size_t) H * L);
    if (a.w_type == kWTypeQ8_0)   // B10.7: the native Q8_0 headwise GEMV
        vtwin::headwise(a.wk_b, a.w_type, a.q, H, L, a.qk_nope, a.q_stride, q_abs.data(), nullptr);
    else
    for (int h = 0; h < H; ++h)
        for (int r = 0; r < L; ++r) {
            const uint16_t* w = a.wk_b + ((size_t) h * L + r) * a.qk_nope;
            const float* x = a.q + (size_t) a.q_stride * h;
            float acc = 0.0f;
            for (int c = 0; c < a.qk_nope; ++c) acc += bf16_float(w[c]) * x[c];
            q_abs[(size_t) h * L + r] = acc;
        }
    // partials per (chunk, head): exp-weighted context, chunk max, chunk sum
    std::vector<float> part((size_t) n_chunks * H * (L + 2));
    std::vector<float> sL((size_t) CH * L), sP((size_t) CH * std::max(R, 1)), e(CH);
    std::vector<int> cell(CH);
    for (int b = 0; b < n_chunks; ++b) {
        for (int s = 0; s < CH; ++s) {
            const int i = b * CH + s;
            cell[s] = i < n_act ? a.cells[i] : -1;
            for (int c = 0; c < L; ++c) sL[(size_t) s * L + c] = cell[s] < 0 ? 0.0f : lat_staged(a.lat, a.lat_q8, L, cell[s], c);
            for (int j = 0; j < R; ++j)
                sP[(size_t) s * R + j] = cell[s] < 0 ? 0.0f : half_float(a.k_pe[(size_t) R * cell[s] + j]);
        }
        for (int h = 0; h < H; ++h) {
            const float* qa = q_abs.data() + (size_t) h * L;
            const float* qp = a.q + (size_t) a.q_stride * h + a.qk_nope;
            float m = -std::numeric_limits<float>::infinity();
            for (int s = 0; s < CH; ++s) {
                float acc = 0.0f;
                for (int c = 0; c < L; ++c) acc += qa[c] * sL[(size_t) s * L + c];
                for (int j = 0; j < R; ++j) acc += qp[j] * sP[(size_t) s * R + j];
                e[s] = cell[s] >= 0 ? acc * a.scale : -std::numeric_limits<float>::infinity();
                m = std::max(m, e[s]);
            }
            float l = 0.0f;
            for (int s = 0; s < CH; ++s) {
                e[s] = e[s] == -std::numeric_limits<float>::infinity() ? 0.0f : std::exp(e[s] - m);
                l += e[s];
            }
            float* pm = part.data() + ((size_t) b * H + h) * (L + 2);
            pm[L] = m;
            pm[L + 1] = l;
            for (int c = 0; c < L; ++c) {
                float acc = 0.0f;
                for (int s = 0; s < CH; ++s) acc += e[s] * sL[(size_t) s * L + c];
                pm[c] = acc;
            }
        }
    }
    for (int h = 0; h < H; ++h) {   // the combine
        float M = -std::numeric_limits<float>::infinity();
        for (int b = 0; b < n_chunks; ++b) M = std::max(M, part[((size_t) b * H + h) * (L + 2) + L]);
        float Ls = 0.0f;
        for (int b = 0; b < n_chunks; ++b) {
            const float* pb = part.data() + ((size_t) b * H + h) * (L + 2);
            if (pb[L] != -std::numeric_limits<float>::infinity()) Ls += std::exp(pb[L] - M) * pb[L + 1];
        }
        const float invL = Ls > 0.0f ? 1.0f / Ls : 0.0f;
        for (int c = 0; c < L; ++c) {
            float acc = 0.0f;
            for (int b = 0; b < n_chunks; ++b) {
                const float* pb = part.data() + ((size_t) b * H + h) * (L + 2);
                if (pb[L] != -std::numeric_limits<float>::infinity()) acc += std::exp(pb[L] - M) * pb[c];
            }
            ctx[(size_t) h * L + c] = acc * invL;
        }
    }
    std::vector<float> out((size_t) H * a.v_head);
    if (a.w_type == kWTypeQ8_0)
        vtwin::headwise(a.wv_b, a.w_type, ctx.data(), H, a.v_head, L, L, out.data(), nullptr);
    else
    for (int h = 0; h < H; ++h)
        for (int v = 0; v < a.v_head; ++v) {
            const uint16_t* w = a.wv_b + ((size_t) h * a.v_head + v) * L;
            float acc = 0.0f;
            for (int c = 0; c < L; ++c) acc += bf16_float(w[c]) * ctx[(size_t) h * L + c];
            out[(size_t) h * a.v_head + v] = acc;
        }
    if (a.out) std::copy(out.begin(), out.end(), a.out);
    if (a.out_q8_1)
        for (size_t b = 0; b < out.size() / 32; ++b) q8_1_store(out.data() + 32 * b, (Q81*) a.out_q8_1 + b);
}

namespace {
// topk's steps 2-3: every key above cut and the first need_eq equal to it, in cell order
void compact_keys(const std::vector<uint32_t>& key, int n, uint32_t cut, int need_eq, int* out) {
    int o = 0, eq_seen = 0;
    for (int i = 0; i < n; ++i) {
        if (key[i] > cut) {
            out[o++] = i;
        } else if (key[i] == cut) {
            if (eq_seen < need_eq) out[o++] = i;
            ++eq_seen;
        }
    }
}
}  // namespace

void topk_hist(const float* score, int ld, int n_kv, int T, const int* pos, int top_k, int* sel, int* fallbacks) {
    std::vector<uint32_t> key, cand;
    std::vector<int> hist(kTopkHistBuckets);
    for (int t = 0; t < T; ++t) {
        const int n = pos ? std::min(n_kv, pos[t] + 1) : n_kv;
        int* out = sel + (size_t) t * top_k;
        if (n <= top_k) {
            for (int i = 0; i < top_k; ++i) out[i] = i < n ? i : -1;
            continue;
        }
        const float* sc = score + (size_t) t * ld;
        key.resize(n);
        uint32_t kmin = 0xffffffffu, kmax = 0u;
        for (int i = 0; i < n; ++i) {   // A. the key range
            key[i] = score_key(sc[i]);
            kmin = std::min(kmin, key[i]);
            kmax = std::max(kmax, key[i]);
        }
        const uint32_t span = kmax - kmin;
        if (span == 0u) {
            compact_keys(key, n, kmin, top_k, out);
            continue;
        }
        std::fill(hist.begin(), hist.end(), 0);   // B. the histogram
        for (int i = 0; i < n; ++i) ++hist[(size_t) topk_hist_bucket(key[i], kmin, span)];
        int bstar = 0, need_b = 0, cum = 0;
        for (int b = kTopkHistBuckets - 1; b >= 0; --b) {
            if (cum + hist[(size_t) b] >= top_k) {
                bstar = b;
                need_b = top_k - cum;
                break;
            }
            cum += hist[(size_t) b];
        }
        if (hist[(size_t) bstar] > kTopkHistCand) {   // the radix fallback (the same cells)
            if (fallbacks) ++*fallbacks;
            topk(sc, n, n, 1, nullptr, top_k, out);
            continue;
        }
        cand.clear();   // C. the boundary bucket, sorted descending
        for (int i = 0; i < n; ++i)
            if (topk_hist_bucket(key[i], kmin, span) == bstar) cand.push_back(key[i]);
        std::sort(cand.begin(), cand.end(), [](uint32_t a, uint32_t b) { return a > b; });
        const uint32_t cut = cand[(size_t) need_b - 1];
        int first_eq = need_b - 1;
        while (first_eq > 0 && cand[(size_t) first_eq - 1] == cut) --first_eq;
        compact_keys(key, n, cut, need_b - first_eq, out);
    }
}

void topk(const float* score, int ld, int n_kv, int T, const int* pos, int top_k, int* sel) {
    std::vector<uint32_t> key;
    for (int t = 0; t < T; ++t) {
        const int n = pos ? std::min(n_kv, pos[t] + 1) : n_kv;
        int* out = sel + (size_t) t * top_k;
        if (n <= top_k) {
            for (int i = 0; i < top_k; ++i) out[i] = i < n ? i : -1;
            continue;
        }
        const float* sc = score + (size_t) t * ld;
        key.resize(n);
        for (int i = 0; i < n; ++i) key[i] = score_key(sc[i]);
        // 1. the cut-off key, a byte at a time from the top
        uint32_t prefix = 0u, mask = 0u;
        int need = top_k;
        for (int shift = 24; shift >= 0; shift -= 8) {
            int hist[256] = {};
            for (int i = 0; i < n; ++i)
                if ((key[i] & mask) == prefix) ++hist[(key[i] >> shift) & 255u];
            for (int b = 255; b >= 0; --b) {
                if (hist[b] >= need) {
                    prefix |= (uint32_t) b << shift;
                    mask |= 255u << shift;
                    break;
                }
                need -= hist[b];
            }
        }
        // 2-3. every key above the cut-off and the first `need` equal to it, in cell order
        int o = 0, eq_seen = 0;
        for (int i = 0; i < n; ++i) {
            if (key[i] > prefix) {
                out[o++] = i;
            } else if (key[i] == prefix) {
                if (eq_seen < need) out[o++] = i;
                ++eq_seen;
            }
        }
    }
}

}  // namespace glmd::twin

// ---- the mixer's own kernels (glmd_mixer.hpp)
#include "strata/kernels/glmd_mixer.hpp"

namespace glmd::twin {

void idx_score(const IdxScore& a) {
    const int n = std::min(a.n_cap, (a.p_dev ? *a.p_dev : a.p) + 1);
    if (a.n_sel_out) *a.n_sel_out = std::min(n, a.sel_cap);
    if (n <= a.skip_le) return;
    const int K = a.idx_dim;
    std::vector<float> wide;   // B10.8: an FP16 key row widened (the kernel widens each element as it reads it)
    for (int c = 0; c < n; ++c) {
        const float* pk = a.ik_cache + (size_t) K * c;
        if (a.key_f16) {
            wide.resize((size_t) K);
            const uint16_t* h = (const uint16_t*) a.ik_cache + (size_t) K * c;
            for (int e = 0; e < K; ++e) wide[(size_t) e] = half_float(h[e]);
            pk = wide.data();
        }
        float acc = 0.0f;
        for (int h = 0; h < a.idx_heads; ++h) {
            // the kernel's lane-strided partial sums, then its xor butterfly
            float part[32];
            for (int l = 0; l < 32; ++l) {
                part[l] = 0.0f;
                for (int e = l; e < K; e += 32) part[l] += a.iq[(size_t) h * K + e] * pk[e];
            }
            for (int o = 16; o > 0; o >>= 1) {
                float nx[32];
                for (int l = 0; l < 32; ++l) nx[l] = part[l] + part[l ^ o];
                std::copy(nx, nx + 32, part);
            }
            acc += std::max(part[0], 0.0f) * a.w[h];
        }
        a.score[c] = acc;
    }
}

uint16_t bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40u);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

}  // namespace glmd::twin
