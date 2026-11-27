// src/ling/ling_parity.cpp - GPU-identical CPU expert lane (-ffp-contract=off) and ggml row verification.
#include "ling_parity.hpp"

#include "ds4_gguf_cpu.hpp"
#include "ds4_lane_parity.hpp"
#include "ds4_par_exp.h"

#include <ggml-cpu.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <random>
#include <unordered_map>
#include <vector>

namespace ling::par {
namespace {

constexpr int kQ4_0 = 2, kQ8_0 = 8, kQ2_K = 10, kQ3_K = 11, kQ4_K = 12, kQ5_K = 13, kQ6_K = 14;
constexpr int kIQ2_XXS = 16, kIQ2_XS = 17, kIQ3_XXS = 18, kIQ1_S = 19, kIQ4_NL = 20, kIQ3_S = 21, kIQ2_S = 22,
              kIQ4_XS = 23, kIQ1_M = 29, kMXFP4 = 39;

constexpr int8_t kIq4NlTab[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

inline float h2f(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    const uint32_t s = (uint32_t) (h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    uint32_t b;
    if (e == 0) {
        const float f = (float) m * 5.9604644775390625e-8f;
        std::memcpy(&b, &f, 4);
        b |= s;
    } else if (e == 31) {
        b = s | 0x7f800000u | (m << 13);
    } else {
        b = s | ((e + 112u) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

inline void q4k_scale(const uint8_t* sc, int j, int& d, int& m) {
    if (j < 4) {
        d = sc[j] & 63;
        m = sc[j + 4] & 63;
    } else {
        d = (sc[j + 4] & 0x0f) | ((sc[j - 4] >> 6) << 4);
        m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
    }
}

inline int q3k_scale(const uint8_t* sc, int j) {
    const uint32_t a0 = (uint32_t) sc[0] | ((uint32_t) sc[1] << 8) | ((uint32_t) sc[2] << 16) | ((uint32_t) sc[3] << 24);
    const uint32_t a1 = (uint32_t) sc[4] | ((uint32_t) sc[5] << 8) | ((uint32_t) sc[6] << 16) | ((uint32_t) sc[7] << 24);
    const uint32_t a2 = (uint32_t) sc[8] | ((uint32_t) sc[9] << 8) | ((uint32_t) sc[10] << 16) | ((uint32_t) sc[11] << 24);
    uint32_t v;
    if (j < 4) v = ((a0 >> (8 * j)) & 0x0fu) | (((a2 >> (8 * j)) & 0x03u) << 4);
    else if (j < 8) v = ((a1 >> (8 * (j - 4))) & 0x0fu) | (((a2 >> (8 * (j - 4) + 2)) & 0x03u) << 4);
    else if (j < 12) v = ((a0 >> (8 * (j - 8) + 4)) & 0x0fu) | (((a2 >> (8 * (j - 8) + 4)) & 0x03u) << 4);
    else v = ((a1 >> (8 * (j - 12) + 4)) & 0x0fu) | (((a2 >> (8 * (j - 12) + 6)) & 0x03u) << 4);
    return (int) v - 32;
}

// Compute one 32-value sub-block dot matching ds4_iq.cu::sub_dot<T, -1> bit-for-bit.
float sub_dot_exact(int type, const uint8_t* row, int kb, const int8_t* xq) {
    const int ib = kb & 7;
    int8_t wq[32];
    int s0 = 1, s1 = 1;
    float d = 0.0f, m0 = 0.0f, m1 = 0.0f;
    bool has_min = false;

    if (type == kQ4_K) {
        const uint8_t* b = row + (size_t) (kb >> 3) * 144;
        int sc, m;
        q4k_scale(b + 4, ib, sc, m);
        const uint8_t* qs = b + 16 + 32 * (ib >> 1);
        const int sh = 4 * (ib & 1);
        for (int i = 0; i < 32; ++i) wq[i] = (int8_t) ((qs[i] >> sh) & 0x0f);
        s0 = s1 = sc;
        d = h2f(b);
        m0 = m1 = h2f(b + 2) * (float) m;
        has_min = true;
    } else if (type == kQ5_K) {
        const uint8_t* b = row + (size_t) (kb >> 3) * 176;
        int sc, m;
        q4k_scale(b + 4, ib, sc, m);
        const uint8_t* qs = b + 48 + 32 * (ib >> 1);
        const uint8_t* qh = b + 16;
        const int sh = 4 * (ib & 1);
        for (int i = 0; i < 32; ++i) wq[i] = (int8_t) (((qs[i] >> sh) & 0x0f) | (((qh[i] >> ib) & 1) << 4));
        s0 = s1 = sc;
        d = h2f(b);
        m0 = m1 = h2f(b + 2) * (float) m;
        has_min = true;
    } else if (type == kQ6_K) {
        const uint8_t* b = row + (size_t) (kb >> 3) * 210;
        const int hf = ib >> 2, qt = ib & 3;
        const uint8_t* ql = b + 64 * hf + 32 * (qt & 1);
        const uint8_t* qh = b + 128 + 32 * hf;
        const int8_t* sc = (const int8_t*) (b + 192 + 8 * hf + 2 * qt);
        for (int i = 0; i < 32; ++i) {
            const int v = ((ql[i] >> (4 * (qt >> 1))) & 15) | (((qh[i] >> (2 * qt)) & 3) << 4);
            wq[i] = (int8_t) (v - 32);
        }
        s0 = sc[0];
        s1 = sc[1];
        d = h2f(b + 208);
    } else if (type == kQ2_K) {
        const uint8_t* b = row + (size_t) (kb >> 3) * 84;
        const int n = ib >> 2, j = ib & 3;
        const uint8_t* qs = b + 16 + 32 * n;
        for (int i = 0; i < 32; ++i) wq[i] = (int8_t) ((qs[i] >> (2 * j)) & 3);
        const int c0 = b[8 * n + 2 * j], c1 = b[8 * n + 2 * j + 1];
        const float dmin = h2f(b + 82);
        s0 = c0 & 15;
        s1 = c1 & 15;
        d = h2f(b + 80);
        m0 = dmin * (float) (c0 >> 4);
        m1 = dmin * (float) (c1 >> 4);
        has_min = true;
    } else if (type == kQ3_K) {
        const uint8_t* b = row + (size_t) (kb >> 3) * 110;
        const int n = ib >> 2, j = ib & 3;
        const uint8_t* qs = b + 32 + 32 * n;
        const uint8_t* hm = b;
        for (int i = 0; i < 32; ++i) {
            const int q2 = (qs[i] >> (2 * j)) & 3;
            const int hb = (hm[i] >> (4 * n + j)) & 1;
            wq[i] = (int8_t) (q2 - (hb ? 0 : 4));
        }
        s0 = q3k_scale(b + 96, 8 * n + 2 * j);
        s1 = q3k_scale(b + 96, 8 * n + 2 * j + 1);
        d = h2f(b + 108);
    } else if (type == kQ8_0) {
        const uint8_t* b = row + (size_t) kb * 34;
        d = h2f(b);
        std::memcpy(wq, b + 2, 32);
        s0 = s1 = 1;
    } else if (type == kIQ4_NL) {
        const uint8_t* b = row + (size_t) kb * 18;
        d = h2f(b);
        for (int i = 0; i < 16; ++i) {
            wq[i] = kIq4NlTab[b[2 + i] & 15];
            wq[16 + i] = kIq4NlTab[b[2 + i] >> 4];
        }
        s0 = s1 = 1;
    } else if (type == kIQ4_XS) {
        const uint8_t* b = row + (size_t) (kb >> 3) * 136;
        const uint16_t sh = (uint16_t) b[2] | ((uint16_t) b[3] << 8);
        const int ls = ((b[4 + (ib >> 1)] >> (4 * (ib & 1))) & 15) | (((sh >> (2 * ib)) & 3) << 4);
        const uint8_t* qs = b + 8 + 16 * ib;
        for (int i = 0; i < 16; ++i) {
            wq[i] = kIq4NlTab[qs[i] & 15];
            wq[16 + i] = kIq4NlTab[qs[i] >> 4];
        }
        s0 = s1 = ls - 32;
        d = h2f(b);
    } else {
        return 0.0f;
    }

    int lo = 0, hi = 0;
    for (int i = 0; i < 16; ++i) lo += (int) wq[i] * (int) xq[i];
    for (int i = 16; i < 32; ++i) hi += (int) wq[i] * (int) xq[i];
    float v = d * (float) (s0 * lo + s1 * hi);
    if (has_min) {
        int sl = 0, sh = 0;
        for (int i = 0; i < 16; ++i) sl += (int) xq[i];
        for (int i = 16; i < 32; ++i) sh += (int) xq[i];
        v = v - (m0 * (float) sl + m1 * (float) sh);
    }
    return v;
}

float row_parity_exact(int type, const uint8_t* row, int k, const void* act, int lpr) {
    const int nsb = k / 32;
    const int8_t* xq = (const int8_t*) act;
    const float* xd = (const float*) (xq + k);
    float acc[32] = {};
    for (int lane = 0; lane < lpr; ++lane) {
        float a = 0.0f;
        for (int kb = lane; kb < nsb; kb += lpr) {
            const float v = sub_dot_exact(type, row, kb, xq + kb * 32);
            a = std::fmaf(xd[kb], v, a);
        }
        acc[lane] = a;
    }
    // Pairwise XOR-shuffle order matching ds4_iq.cu::lanes_sum<RW>
    for (int step = lpr >> 1; step >= 1; step >>= 1) {
        for (int i = 0; i < step; ++i) acc[i] = acc[i] + acc[i + step];
    }
    return acc[0];
}

size_t row_stride_bytes(int type, int k) {
    return ggml_row_size((ggml_type) type, k);
}

uint64_t fnv1a_f32(const float* p, size_t n, uint64_t h = 1469598103934665603ULL) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n * sizeof(float); ++i) h = (h ^ b[i]) * 1099511628211ULL;
    return h;
}

}  // namespace

bool supported(int type) {
    if (maya::ds4::par::lane_type(type)) return true;
    switch (type) {
        case kQ4_K:
        case kQ5_K:
        case kQ6_K:
        case kQ2_K:
        case kQ3_K:
        case kQ8_0:
        case kIQ4_NL:
        case kIQ4_XS:
            return true;
        default:
            return false;
    }
}

void quant_q8k(const float* x, int k, void* act) {
    maya::ds4::par::quant(x, k, act);
}

void cpu_experts_parity(hy4::Lane& lane, int tg, int td, int E, int F, float limit,
                        const std::vector<hy4::CpuEntry>& entries, const uint8_t* const* gate_ptrs,
                        const uint8_t* const* up_ptrs, const uint8_t* const* down_ptrs, const float* x, int T,
                        float* out) {
    const int n = (int) entries.size();
    std::fill(out, out + (size_t) T * E, 0.0f);
    if (n == 0) return;

    const size_t xab = act_bytes(E), hab = act_bytes(F);
    std::vector<uint8_t> xact((size_t) T * xab);
    for (int t = 0; t < T; ++t) quant_q8k(x + (size_t) t * E, E, xact.data() + (size_t) t * xab);

    std::vector<float> h((size_t) n * F, 0.0f);
    std::vector<uint8_t> hact((size_t) n * hab);
    std::vector<float> y((size_t) n * E, 0.0f);

    const bool use_ds4_gu = maya::ds4::par::lane_type(tg);
    const bool use_ds4_dn = maya::ds4::par::lane_type(td);
    const size_t gu_rs = row_stride_bytes(tg, E);
    const size_t dn_rs = row_stride_bytes(td, F);

    constexpr int kChunk = 32;
    const int gu_chunks = (F + kChunk - 1) / kChunk;
    lane.parallel_for(n * gu_chunks, [&](int it) {
        const int idx = it / gu_chunks;
        const int r0 = (it % gu_chunks) * kChunk;
        const int r1 = std::min(F, r0 + kChunk);
        const hy4::CpuEntry& ce = entries[(size_t) idx];
        const uint8_t* gp = gate_ptrs[(size_t) ce.e];
        const uint8_t* up = up_ptrs[(size_t) ce.e];
        const void* actp = xact.data() + (size_t) ce.t * xab;
        float* hp = h.data() + (size_t) idx * F;
        if (use_ds4_gu) {
            maya::ds4::par::gate_up(tg, gp, up, E, &actp, 1, &hp, r0, r1, limit);
        } else {
            for (int r = r0; r < r1; ++r) {
                const float gv = row_parity_exact(tg, gp + (size_t) r * gu_rs, E, actp, 32);
                const float uv = row_parity_exact(tg, up + (size_t) r * gu_rs, E, actp, 32);
                hp[r] = maya::ds4::par::swiglu_portable(gv, uv, limit);
            }
        }
    });

    for (int i = 0; i < n; ++i) quant_q8k(h.data() + (size_t) i * F, F, hact.data() + (size_t) i * hab);

    const int dn_chunks = (E + kChunk - 1) / kChunk;
    lane.parallel_for(n * dn_chunks, [&](int it) {
        const int idx = it / dn_chunks;
        const int r0 = (it % dn_chunks) * kChunk;
        const int r1 = std::min(E, r0 + kChunk);
        const hy4::CpuEntry& ce = entries[(size_t) idx];
        const uint8_t* dp = down_ptrs[(size_t) ce.e];
        const void* actp = hact.data() + (size_t) idx * hab;
        float* yp = y.data() + (size_t) idx * E;
        if (use_ds4_dn) {
            maya::ds4::par::down(td, dp, F, &actp, 1, &yp, r0, r1);
        } else {
            for (int r = r0; r < r1; ++r) {
                yp[r] = row_parity_exact(td, dp + (size_t) r * dn_rs, F, actp, 16);
            }
        }
    });

    for (int i = 0; i < n; ++i) {
        const hy4::CpuEntry& ce = entries[(size_t) i];
        float* dst = out + (size_t) ce.t * E;
        const float* src = y.data() + (size_t) i * E;
        for (int c = 0; c < E; ++c) dst[c] += ce.w * src[c];
    }
}

bool verify_format_once(int type, std::string& msg) {
    static std::mutex mu;
    static std::unordered_map<int, std::pair<bool, std::string>> cache;
    std::lock_guard<std::mutex> lk(mu);
    auto it = cache.find(type);
    if (it != cache.end()) {
        msg = it->second.second;
        return it->second.first;
    }
    ggml_cpu_init();
    const ggml_type gt = (ggml_type) type;
    const auto* tr = ggml_get_type_traits_cpu(gt);
    const char* tname = ggml_type_name(gt);
    if (tr == nullptr || tr->vec_dot == nullptr) {
        msg = std::string(tname ? tname : "?") + ": unsupported in ggml-cpu";
        cache[type] = {false, msg};
        return false;
    }
    maya::ds4::gguf::IqCpu f;
    std::string err;
    constexpr int K = 512, R = 256;
    if (!maya::ds4::gguf::iq_cpu_fmt(type, type, R, K, f, err)) {
        msg = std::string(tname ? tname : "?") + ": " + err;
        cache[type] = {false, msg};
        return false;
    }
    const size_t rb = ggml_row_size(gt, K);
    const size_t bb = ggml_type_size(gt);
    const size_t ab = ggml_row_size((ggml_type) f.d_act, K);
    std::mt19937 rng(0x4c494e47u ^ (unsigned) type);
    std::vector<uint8_t> w((size_t) R * rb);
    for (auto& b : w) b = (uint8_t) (rng() & 255u);
    // Keep F16 scales finite & small to avoid NaN/Inf in random bit patterns
    const int nb = K / (int) ggml_blck_size(gt);
    const uint16_t sane_d = 0x2400u;  // ~0.00390625 in F16
    for (int r = 0; r < R; ++r) {
        for (int b = 0; b < nb; ++b) {
            uint8_t* blk = w.data() + (size_t) r * rb + (size_t) b * bb;
            if (gt == GGML_TYPE_Q6_K) std::memcpy(blk + 208, &sane_d, 2);
            else if (gt == GGML_TYPE_Q3_K) std::memcpy(blk + 108, &sane_d, 2);
            else if (gt == GGML_TYPE_Q2_K) {
                std::memcpy(blk + 80, &sane_d, 2);
                std::memcpy(blk + 82, &sane_d, 2);
            } else if (gt == GGML_TYPE_Q4_K || gt == GGML_TYPE_Q5_K) {
                std::memcpy(blk, &sane_d, 2);
                std::memcpy(blk + 2, &sane_d, 2);
            } else if (gt == GGML_TYPE_IQ1_M) { for (int q = 0; q < 4; ++q) { uint16_t v = 0x1000u; std::memcpy(blk + 48 + q * 2, &v, 2); } } else {
                std::memcpy(blk, &sane_d, 2);
            }
        }
    }
    std::vector<float> xf(K);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& v : xf) v = dist(rng);
    std::vector<uint8_t> act(ab);
    maya::ds4::gguf::iq_cpu_quant(f.d_act, xf.data(), K, act.data());

    std::vector<float> ref(R, 0.0f), got(R, 0.0f);
    for (int r = 0; r < R; ++r) tr->vec_dot(K, &ref[(size_t) r], 0, w.data() + (size_t) r * rb, 0, act.data(), 0, 1);
    const void* actp = act.data();
    float* gotp = got.data();
    maya::ds4::gguf::iq_cpu_down(f, w.data(), &actp, 1, &gotp, 0, R);

    int diff_bits = 0;
    float max_abs = 0.0f;
    for (int r = 0; r < R; ++r) {
        uint32_t ba = 0, bb2 = 0;
        std::memcpy(&ba, &ref[(size_t) r], 4);
        std::memcpy(&bb2, &got[(size_t) r], 4);
        if (ba != bb2) ++diff_bits;
        max_abs = std::max(max_abs, std::fabs(ref[(size_t) r] - got[(size_t) r]));
    }
    char buf[160];
    std::snprintf(buf, sizeof buf, "%s: native vs ggml diff_bits=%d/%d max_abs=%.3e", tname ? tname : "?", diff_bits, R,
                  max_abs);
    msg = buf;
    const bool ok = (diff_bits == 0);
    cache[type] = {ok, msg};
    return ok;
}

bool bench_lane_verify(int n_calls) {
    ggml_cpu_init();
    const int formats[] = {kQ8_0, kQ6_K, kQ5_K, kQ4_K, kQ3_K, kQ2_K, kIQ4_NL, kIQ4_XS,
                           kIQ3_S, kIQ3_XXS, kIQ2_S, kIQ2_XS, kIQ2_XXS, kIQ1_S, kIQ1_M};
    bool all_ok = true;
    constexpr int K = 512, R = 256;
    std::printf("maya-ling --bench-lane: verifying %zu shipped formats x %d random calls vs ggml-cpu\n",
                sizeof(formats) / sizeof(formats[0]), n_calls);
    for (int t : formats) {
        const ggml_type gt = (ggml_type) t;
        const char* tname = ggml_type_name(gt);
        const auto* tr = ggml_get_type_traits_cpu(gt);
        maya::ds4::gguf::IqCpu f;
        std::string err;
        if (!tr || !tr->vec_dot || !maya::ds4::gguf::iq_cpu_fmt(t, t, R, K, f, err)) {
            std::printf("  %-10s SKIP (%s)\n", tname ? tname : "?", err.c_str());
            continue;
        }
        const size_t rb = ggml_row_size(gt, K);
        const size_t bb = ggml_type_size(gt);
        const size_t ab = ggml_row_size((ggml_type) f.d_act, K);
        const int nb = K / (int) ggml_blck_size(gt);
        std::mt19937 rng(0x13579bdfu ^ (unsigned) t);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<uint8_t> w((size_t) R * rb);
        std::vector<float> xf(K), ref(R), got(R);
        std::vector<uint8_t> act(ab);
        uint64_t h_ref = 1469598103934665603ULL, h_got = 1469598103934665603ULL;
        int bit_diffs = 0;
        float max_diff = 0.0f;
        const uint16_t sane_d = 0x2400u;
        for (int c = 0; c < n_calls; ++c) {
            for (auto& b : w) b = (uint8_t) (rng() & 255u);
            for (int r = 0; r < R; ++r) {
                for (int b = 0; b < nb; ++b) {
                    uint8_t* blk = w.data() + (size_t) r * rb + (size_t) b * bb;
                    if (gt == GGML_TYPE_Q6_K) std::memcpy(blk + 208, &sane_d, 2);
                    else if (gt == GGML_TYPE_Q3_K) std::memcpy(blk + 108, &sane_d, 2);
                    else if (gt == GGML_TYPE_Q2_K) {
                        std::memcpy(blk + 80, &sane_d, 2);
                        std::memcpy(blk + 82, &sane_d, 2);
                    } else if (gt == GGML_TYPE_Q4_K || gt == GGML_TYPE_Q5_K) {
                        std::memcpy(blk, &sane_d, 2);
                        std::memcpy(blk + 2, &sane_d, 2);
                    } else if (gt == GGML_TYPE_IQ1_M) { for (int q = 0; q < 4; ++q) { uint16_t v = 0x1000u; std::memcpy(blk + 48 + q * 2, &v, 2); } } else {
                        std::memcpy(blk, &sane_d, 2);
                    }
                }
            }
            for (float& v : xf) v = dist(rng);
            maya::ds4::gguf::iq_cpu_quant(f.d_act, xf.data(), K, act.data());
            for (int r = 0; r < R; ++r)
                tr->vec_dot(K, &ref[(size_t) r], 0, w.data() + (size_t) r * rb, 0, act.data(), 0, 1);
            const void* actp = act.data();
            float* gotp = got.data();
            maya::ds4::gguf::iq_cpu_down(f, w.data(), &actp, 1, &gotp, 0, R);
            h_ref = fnv1a_f32(ref.data(), (size_t) R, h_ref);
            h_got = fnv1a_f32(got.data(), (size_t) R, h_got);
            for (int r = 0; r < R; ++r) {
                uint32_t ba = 0, bb2 = 0;
                std::memcpy(&ba, &ref[(size_t) r], 4);
                std::memcpy(&bb2, &got[(size_t) r], 4);
                if (ba != bb2) ++bit_diffs;
                max_diff = std::max(max_diff, std::fabs(ref[(size_t) r] - got[(size_t) r]));
            }
        }
        const bool ok = (bit_diffs == 0 && h_ref == h_got);
        if (!ok) all_ok = false;
        std::printf("  %-10s calls=%d ref=%016llx got=%016llx bit_diffs=%d max_diff=%.3e %s\n",
                    tname ? tname : "?", n_calls, (unsigned long long) h_ref, (unsigned long long) h_got, bit_diffs,
                    max_diff, ok ? "PASS" : "DIFF");
    }
    return all_ok;
}

}  // namespace ling::par
