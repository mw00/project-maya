// src/core/glm_dsa_cpu_volta_twin.cpp - CPU twins of the Volta glm-dsa decode kernels (glm-dsa B10.2-B10.4, B10.7,
// B10.8); see include/strata/kernels/glmd_volta.hpp.
//
// Each twin runs the kernel's algorithm on host arrays: the same FP16 staging and saturation, the same tiling and
// split ranges, the same k-quarter partial sums and their combination order, the same online softmax with fp16 P
// and the 8-lane butterfly sums, the same radix passes and per-block decomposition of the selector.  Sums inside one
// WMMA k-step run in an order the hardware does not expose; the twin adds the exact fp16 x fp16 products in f32 in
// index order there, so twin and kernel agree to f32 rounding, not bit for bit (the selector is exact).
// Plain C++, no CUDA: the CPU test links this file directly; the default-path helpers (fused_mla_enabled & co.)
// live in the .cu files.
#include "strata/kernels/glmd_volta.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace glmd {

size_t head_row_bytes(int w_type, int n) {
    if (n <= 0) return 0;
    if (w_type == kWTypeBF16) return (size_t) n * 2;
    if (w_type == kWTypeQ8_0) return n % 32 ? 0 : (size_t) (n / 32) * 34;
    return 0;
}

int mla_fused_splits(int n_sel, int req) {
    if (n_sel <= 0) return 0;
    const int cells = mla_fused_split_cells(n_sel, req);
    return (n_sel + cells - 1) / cells;
}

int mla_fused_split_cells(int n_sel, int req) {
    if (n_sel <= 0) return kFusedUnit;
    const int units = (n_sel + kFusedUnit - 1) / kFusedUnit;
    int ns = req > 0 ? req : kFusedSplitDefault;
    ns = std::max(1, std::min(ns, units));
    const int ups = (units + ns - 1) / ns;
    return ups * kFusedUnit;
}

size_t mla_fused_work_bytes(int n_head, int kv_lora, int qk_rope, int max_sel, int req) {
    (void) n_head;
    (void) kv_lora;
    const size_t K = (size_t) kFusedLora + qk_rope;
    const size_t ns = (size_t) std::max(1, mla_fused_splits(max_sel, req));
    auto al = [](size_t b) { return (b + 255) & ~(size_t) 255; };
    return al((size_t) kFusedHeads * K * 2)                         // Q tile FP16
           + al(ns * kFusedHeads * kFusedLora * 4)                  // partial O
           + al(ns * kFusedHeads * 2 * 4)                           // partial (m, l)
           + al((size_t) kFusedHeads * kFusedLora * 4);             // ctx
}

bool mla_fused_supported(int n_head, int qk_nope, int qk_rope, int kv_lora, int v_head, int w_type) {
    if (n_head < 1 || n_head > kFusedHeads || kv_lora != kFusedLora || (qk_rope != 0 && qk_rope != 64)) return false;
    if (v_head <= 0 || v_head % 32 || qk_nope <= 0) return false;
    if (w_type == kWTypeBF16) return qk_nope % 8 == 0;
    if (w_type == kWTypeQ8_0) return qk_nope % 32 == 0;
    return false;
}

size_t topk_mb_work_bytes(int T) {
    // per row: hist[256] u32, state {prefix, mask, need, counter, counter2, pad x3}, per-block {gt, eq, off, eqb}
    return (size_t) std::max(T, 0) * (256 + 8 + 4 * kTopkMbBlocks) * 4;
}

namespace vtwin {
namespace {

using twin::half_bits;
using twin::half_float;

constexpr float kInf = std::numeric_limits<float>::infinity();

float bf16_float(uint16_t v) {
    const uint32_t x = (uint32_t) v << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

// fp16 with saturation (the kernels clamp to +-65504 before __float2half so Q never becomes inf)
uint16_t half_sat(float f) { return half_bits(std::max(-65504.0f, std::min(65504.0f, f))); }

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

float lat_value(const void* lat, int lat_q8, int kv_lora, int cell, int c) {
    if (lat_q8) {
        const uint8_t* rec = (const uint8_t*) lat + (size_t) lat8_rec_bytes(kv_lora) * cell;
        uint16_t d16;
        std::memcpy(&d16, rec + kv_lora + 2 * (c >> 5), 2);
        return (float) ((const int8_t*) rec)[c] * half_float(d16);
    }
    return half_float(((const uint16_t*) lat)[(size_t) kv_lora * cell + c]);
}

// one row dot in the kernel's per-lane grouping: lane l takes groups l, l + 32, ...; the 32 lane sums are then
// added by the butterfly (xor 16, 8, 4, 2, 1)
float lane_tree(float* s) {
    for (int o = 16; o > 0; o >>= 1)
        for (int l = 0; l < 32; ++l)
            if ((l & o) == 0) s[l] = s[l] + s[l + o];
    return s[0];
}

float row_dot(const void* W, int w_type, size_t row, const float* x, int K) {
    float s[32] = {};
    if (w_type == kWTypeBF16) {
        const uint16_t* r = (const uint16_t*) W + row * (size_t) K;
        for (int k = 0; k < K / 8; ++k) {   // 8 values per group (one uint4)
            float acc = 0.0f;
            for (int j = 0; j < 8; ++j) acc += bf16_float(r[8 * k + j]) * x[8 * k + j];
            s[k % 32] += acc;
        }
    } else {
        const uint8_t* r = (const uint8_t*) W + row * head_row_bytes(kWTypeQ8_0, K);
        for (int k = 0; k < K / 4; ++k) {   // 4 values per group (one int of codes)
            const uint8_t* blk = r + (size_t) (k / 8) * 34;
            uint16_t d16;
            std::memcpy(&d16, blk, 2);
            const int8_t* q = (const int8_t*) (blk + 2) + 4 * (k % 8);
            const float* xx = x + 4 * k;
            const float acc = (float) q[0] * xx[0] + (float) q[1] * xx[1] + (float) q[2] * xx[2] + (float) q[3] * xx[3];
            s[k % 32] += half_float(d16) * acc;
        }
    }
    return lane_tree(s);
}

}  // namespace

void q8_0_quantize(const float* x, int n, void* blocks) {
    uint8_t* b = (uint8_t*) blocks;
    for (int i = 0; i < n / 32; ++i, b += 34) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[32 * i + j]));
        const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        const uint16_t d16 = half_bits(d);
        std::memcpy(b, &d16, 2);
        for (int j = 0; j < 32; ++j) ((int8_t*) (b + 2))[j] = (int8_t) std::round(x[32 * i + j] * id);
    }
}

void q8_0_dequantize(const void* blocks, int n, float* y) {
    const uint8_t* b = (const uint8_t*) blocks;
    for (int i = 0; i < n / 32; ++i, b += 34) {
        uint16_t d16;
        std::memcpy(&d16, b, 2);
        const float d = half_float(d16);
        for (int j = 0; j < 32; ++j) y[32 * i + j] = d * (float) ((const int8_t*) (b + 2))[j];
    }
}

void headwise(const void* W, int w_type, const float* in, int n_head, int R, int K, int in_stride, float* out,
              Q81* q8) {
    std::vector<float> o((size_t) n_head * R);
    for (int h = 0; h < n_head; ++h)
        for (int r = 0; r < R; ++r) o[(size_t) h * R + r] = row_dot(W, w_type, (size_t) h * R + r, in + (size_t) h * in_stride, K);
    if (out) std::copy(o.begin(), o.end(), out);
    if (q8)
        for (size_t b = 0; b < o.size() / 32; ++b) q8_1_store(o.data() + 32 * b, q8 + b);
}

void mla_fused(const FusedMla& a) {
    const int H = a.n_head, L = kFusedLora, R = a.qk_rope, K = L + R, KS = K / 16, HP = kFusedHeads;
    std::vector<float> out((size_t) H * a.v_head, 0.0f);
    std::vector<float> ctx((size_t) HP * L, 0.0f);
    if (a.n_sel > 0) {
        // 1. the Q tile: FP16 [q_abs | q_pe], rows >= H zero
        std::vector<float> q_abs((size_t) H * L);
        headwise(a.wk_b, a.w_type, a.q, H, L, a.qk_nope, a.q_stride, q_abs.data(), nullptr);
        std::vector<float> Q((size_t) HP * K, 0.0f);
        for (int h = 0; h < H; ++h) {
            for (int r = 0; r < L; ++r) Q[(size_t) h * K + r] = half_float(half_sat(q_abs[(size_t) h * L + r]));
            for (int j = 0; j < R; ++j)
                Q[(size_t) h * K + L + j] = half_float(half_sat(a.q[(size_t) a.q_stride * h + a.qk_nope + j]));
        }
        // 2. the split attention
        const int cps = mla_fused_split_cells(a.n_sel, a.n_split);
        const int ns = mla_fused_splits(a.n_sel, a.n_split);
        std::vector<float> po((size_t) ns * HP * L), pm((size_t) ns * HP), pl((size_t) ns * HP);
        std::vector<float> Ls((size_t) kFusedSub * K), S((size_t) HP * kFusedSub), P((size_t) HP * kFusedSub);
        std::vector<float> part[4];
        for (auto& v : part) v.assign((size_t) HP * kFusedSub, 0.0f);
        for (int b = 0; b < ns; ++b) {
            const int c0 = b * cps, c1 = std::min(a.n_sel, c0 + cps);
            std::vector<float> O((size_t) HP * L, 0.0f), m(HP, -kInf), l(HP, 0.0f);
            for (int st = c0; st < c1; st += kFusedSub) {
                int cell[kFusedSub];
                for (int s = 0; s < kFusedSub; ++s) {
                    cell[s] = st + s < c1 ? a.cells[st + s] : -1;
                    for (int c = 0; c < L; ++c)
                        Ls[(size_t) s * K + c] =
                            cell[s] < 0 ? 0.0f : half_float(half_bits(lat_value(a.lat, a.lat_q8, L, cell[s], c)));
                    for (int j = 0; j < R; ++j)
                        Ls[(size_t) s * K + L + j] = cell[s] < 0 ? 0.0f : half_float(a.k_pe[(size_t) R * cell[s] + j]);
                }
                // S: 4 k-quarters, each a sum over its 16-wide k-steps
                for (int g = 0; g < 4; ++g) {
                    const int k0 = g * KS / 4 * 16, k1 = (g + 1) * KS / 4 * 16;
                    for (int h = 0; h < HP; ++h)
                        for (int s = 0; s < kFusedSub; ++s) {
                            float acc = 0.0f;
                            for (int k = k0; k < k1; ++k) acc += Q[(size_t) h * K + k] * Ls[(size_t) s * K + k];
                            part[g][(size_t) h * kFusedSub + s] = acc;
                        }
                }
                for (size_t i = 0; i < S.size(); ++i) S[i] = part[0][i] + (part[1][i] + (part[2][i] + part[3][i]));
                // online softmax, 8 lanes per head with 2 cells each
                std::vector<float> alpha(HP);
                for (int h = 0; h < HP; ++h) {
                    float sv[kFusedSub];
                    float tmax = -kInf;
                    for (int s = 0; s < kFusedSub; ++s) {
                        sv[s] = cell[s] >= 0 ? S[(size_t) h * kFusedSub + s] * a.scale : -kInf;
                        tmax = std::max(tmax, sv[s]);
                    }
                    const float mn = std::max(m[h], tmax);
                    float al, ps[kFusedSub];
                    if (mn == -kInf) {
                        al = 1.0f;
                        for (float& p : ps) p = 0.0f;
                    } else {
                        al = m[h] == -kInf ? 0.0f : std::exp(m[h] - mn);
                        for (int s = 0; s < kFusedSub; ++s)
                            ps[s] = sv[s] == -kInf ? 0.0f : half_float(half_bits(std::exp(sv[s] - mn)));
                    }
                    float t[32];
                    for (int j = 0; j < 8; ++j) t[j] = ps[2 * j] + ps[2 * j + 1];
                    for (int o = 1; o < 8; o <<= 1)   // butterfly xor 1, 2, 4: lane 0's value
                        for (int j = 0; j < 8; ++j)
                            if ((j & o) == 0) t[j] = t[j] + t[j + o];
                    l[h] = l[h] * al + t[0];
                    m[h] = mn;
                    alpha[h] = al;
                    for (int s = 0; s < kFusedSub; ++s) P[(size_t) h * kFusedSub + s] = ps[s];
                }
                for (int h = 0; h < HP; ++h)
                    for (int c = 0; c < L; ++c) {
                        float acc = 0.0f;
                        for (int s = 0; s < kFusedSub; ++s) acc += P[(size_t) h * kFusedSub + s] * Ls[(size_t) s * K + c];
                        O[(size_t) h * L + c] = O[(size_t) h * L + c] * alpha[h] + acc;
                    }
            }
            std::copy(O.begin(), O.end(), po.begin() + (size_t) b * HP * L);
            for (int h = 0; h < HP; ++h) {
                pm[(size_t) b * HP + h] = m[h];
                pl[(size_t) b * HP + h] = l[h];
            }
        }
        // 3. combine
        for (int h = 0; h < H; ++h) {
            float M = -kInf;
            for (int b = 0; b < ns; ++b) M = std::max(M, pm[(size_t) b * HP + h]);
            float Lsum = 0.0f;
            for (int b = 0; b < ns; ++b)
                if (pm[(size_t) b * HP + h] != -kInf) Lsum += std::exp(pm[(size_t) b * HP + h] - M) * pl[(size_t) b * HP + h];
            const float inv = Lsum > 0.0f ? 1.0f / Lsum : 0.0f;
            for (int c = 0; c < L; ++c) {
                float acc = 0.0f;
                for (int b = 0; b < ns; ++b)
                    if (pm[(size_t) b * HP + h] != -kInf)
                        acc += std::exp(pm[(size_t) b * HP + h] - M) * po[((size_t) b * HP + h) * L + c];
                ctx[(size_t) h * L + c] = acc * inv;
            }
        }
    }
    // 4. W_vb
    headwise(a.wv_b, a.w_type, ctx.data(), H, a.v_head, L, L, out.data(), nullptr);
    if (a.out) std::copy(out.begin(), out.end(), a.out);
    if (a.out_q8_1)
        for (size_t b = 0; b < out.size() / 32; ++b) q8_1_store(out.data() + 32 * b, (Q81*) a.out_q8_1 + b);
}

namespace {
float key_value(const IdxScoreWmma& a, int s, int e) {
    if (a.key_f16) return half_float(((const uint16_t*) a.keys)[(size_t) a.idx_dim * s + e]);
    return ((const float*) a.keys)[(size_t) a.idx_dim * s + e];
}
}  // namespace

void idx_score_wmma(const IdxScoreWmma& a) {
    const int ld = a.ld > 0 ? a.ld : a.n_kv, D = a.idx_dim;
    std::vector<float> key(D);
    for (int t = 0; t < a.T; ++t) {
        const int pt = a.pos ? a.pos[t] : a.n_kv - 1;
        std::vector<float> q((size_t) a.idx_heads * D);
        for (size_t i = 0; i < q.size(); ++i) q[i] = half_float(half_sat(a.iq[(size_t) t * a.idx_heads * D + i]));
        for (int s = 0; s < a.n_kv; ++s) {
            float* out = a.score + (size_t) t * ld + s;
            if (s > pt) {
                *out = -kInf;
                continue;
            }
            for (int e = 0; e < D; ++e) key[e] = half_float(half_sat(key_value(a, s, e)));
            float acc = 0.0f;
            for (int h = 0; h < a.idx_heads; ++h) {
                float dot = 0.0f;
                for (int e = 0; e < D; ++e) dot += q[(size_t) h * D + e] * key[e];
                acc += std::max(dot, 0.0f) * a.w[(size_t) t * a.idx_heads + h];
            }
            *out = acc;
        }
    }
}

void idx_score_simple(const IdxScoreWmma& a) {
    const int ld = a.ld > 0 ? a.ld : a.n_kv, D = a.idx_dim;
    for (int t = 0; t < a.T; ++t) {
        const int pt = a.pos ? a.pos[t] : a.n_kv - 1;
        const float* q = a.iq + (size_t) t * a.idx_heads * D;
        for (int s = 0; s < a.n_kv; ++s) {
            float* out = a.score + (size_t) t * ld + s;
            if (s > pt) {
                *out = -kInf;
                continue;
            }
            float acc = 0.0f;
            for (int h = 0; h < a.idx_heads; ++h) {
                float lane[32] = {};
                for (int e = 0; e < D; ++e) lane[e % 32] += q[(size_t) h * D + e] * key_value(a, s, e);
                const float dot = lane_tree(lane);
                acc += std::max(dot, 0.0f) * a.w[(size_t) t * a.idx_heads + h];
            }
            *out = acc;
        }
    }
}

void topk_mb(const float* score, int ld, int n_kv, int T, const int* pos, int top_k, int* sel) {
    const int G = kTopkMbBlocks;
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
        for (int i = 0; i < n; ++i) key[i] = twin::score_key(sc[i]);
        const int chunk = (n + G - 1) / G;
        // 1. four radix passes; each block adds its range's histogram into the global one
        uint32_t prefix = 0u, mask = 0u;
        int need = top_k;
        for (int shift = 24; shift >= 0; shift -= 8) {
            uint32_t hist[256] = {};
            for (int b = 0; b < G; ++b)
                for (int i = b * chunk; i < std::min(n, (b + 1) * chunk); ++i)
                    if ((key[i] & mask) == prefix) ++hist[(key[i] >> shift) & 255u];
            for (int d = 255; d >= 0; --d) {
                if ((int) hist[d] >= need) {
                    prefix |= (uint32_t) d << shift;
                    mask |= 255u << shift;
                    break;
                }
                need -= (int) hist[d];
            }
        }
        // 2. per-block counts, the scan, the tie shares (lower blocks first), the offsets
        std::vector<int> gt(G, 0), eq(G, 0), off(G, 0), eqb(G, 0);
        for (int b = 0; b < G; ++b)
            for (int i = b * chunk; i < std::min(n, (b + 1) * chunk); ++i) {
                gt[b] += key[i] > prefix;
                eq[b] += key[i] == prefix;
            }
        int run_eq = 0, run_off = 0;
        for (int b = 0; b < G; ++b) {
            eqb[b] = run_eq;
            off[b] = run_off;
            const int take = std::max(0, std::min(eq[b], need - run_eq));
            run_eq += eq[b];
            run_off += gt[b] + take;
        }
        // 3. each block writes its winners in cell order
        for (int b = 0; b < G; ++b) {
            int o = off[b], seen = eqb[b];
            for (int i = b * chunk; i < std::min(n, (b + 1) * chunk); ++i) {
                if (key[i] > prefix) {
                    out[o++] = i;
                } else if (key[i] == prefix) {
                    if (seen < need) out[o++] = i;
                    ++seen;
                }
            }
        }
    }
}

}  // namespace vtwin

}  // namespace glmd
