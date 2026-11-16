// tests/glm_dsa/glm_lane_parity_test.cpp - glm-dsa B14.8: CPU-lane parity cases from ik_llama.cpp's AVX2 bug history
// (a8ae4fb3 epi32 vs epi8 add, 310f8b1d the nrc_y = 1 path, 0488a1f9 / 49befdd4).  CPU only, runs everywhere; a
// kernel this CPU cannot run is reported as skipped.
//
// Every lane kernel - the multi-token i-quant kernels (iq256_* AVX-2, iq512_* AVX-512 when the CPU has VNNI + VBMI),
// the Q2_K / Q3_K offset-trick kernel (kq_rows, AVX-512BW), the IQ4_NL down kernel (iq4nl256_down_rows), the
// dispatchers the lane calls (native_gu_rows / native_down_rows) and ggml-cpu's single-token vec_dot - is checked for:
//   1. nt = 1 vs nt > 1 identical per row: each token's row computed alone and inside every batch of 2..8 tokens
//      (and in reversed token order) is bit-identical for the multi-token kernels; the dispatchers' nt = 1 (ggml-cpu)
//      against their nt > 1 (the multi-token kernel) within float summation order;
//   2. row counts that are not a multiple of 4 or 8: row ranges [3, 11), [5, 6), [1, 13), [0, 13) of a 13-row matrix
//      equal the whole run bit for bit and leave the rows outside untouched;
//   3. max-magnitude grid entries: rows made only of the grid entry with the largest |value| (found by dequantizing
//      every grid index), the largest block scales and all-negative or all-positive signs, against Q8_K activations
//      of +-127 (the int16 / int32 overflow cases; -128 is outside the quantizers' range, checked too), checked against a double-precision reference from ggml's own
//      dequantizers (relative 1e-5 of the row's sum of |terms|).
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/kq_avx512.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace kc = strata::kernels::cpu;

namespace {

int g_fail = 0, g_skip = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-104s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
void skip(const std::string& what) {
    std::printf("  %-104s skipped\n", what.c_str());
    ++g_skip;
}

constexpr int QK = 256;
constexpr int kRows = 13;   // not a multiple of 4 or 8

void put_half(uint8_t* p, float v) {
    const ggml_fp16_t h = ggml_fp32_to_fp16(v);
    std::memcpy(p, &h, sizeof h);
}

int grid_size(ggml_type t) {
    switch (t) {
        case GGML_TYPE_IQ2_XXS: return 256;
        case GGML_TYPE_IQ2_XS: return 512;
        case GGML_TYPE_IQ2_S: return 1024;
        case GGML_TYPE_IQ3_XXS: return 256;
        case GGML_TYPE_IQ3_S: return 512;
        default: return 0;
    }
}

// one block whose every grid index is k, at the largest block scales, signs all negative (neg) or all positive
void grid_block(ggml_type t, int k, bool neg, float d, uint8_t* b) {
    std::memset(b, 0, ggml_type_size(t));
    put_half(b, d);
    uint8_t* q = b + 2;
    switch (t) {
        case GGML_TYPE_IQ2_XXS:   // per 32 values: 4 index bytes, then u32 = 4 x 7 sign bits | scale << 28
        case GGML_TYPE_IQ3_XXS: {   // 64 index bytes, then 8 x u32 of signs | scale
            const bool xxs2 = t == GGML_TYPE_IQ2_XXS;
            const uint32_t ss = (15u << 28) | (neg ? 0x0fffffffu : 0u);
            for (int ib = 0; ib < 8; ++ib) {
                uint8_t* idx = q + 8 * ib;   // both: chunk ib's index bytes (IQ2_XXS: 4, then its sign word; IQ3_XXS: 8)
                for (int j = 0; j < (xxs2 ? 4 : 8); ++j) idx[j] = (uint8_t) k;
                std::memcpy(xxs2 ? q + 8 * ib + 4 : q + 64 + 4 * ib, &ss, 4);
            }
            break;
        }
        case GGML_TYPE_IQ2_XS: {   // 32 x u16: 9-bit index | 7 sign bits << 9; scales[8] at 66
            const uint16_t v = (uint16_t) ((k & 511) | (neg ? 127 << 9 : 0));
            for (int j = 0; j < 32; ++j) std::memcpy(q + 2 * j, &v, 2);
            std::memset(b + 66, 0xff, 8);
            break;
        }
        case GGML_TYPE_IQ2_S: {   // qs[32] low index bits, signs[32], qh[8] (2 bits an index), scales[8]
            std::memset(q, k & 255, 32);
            std::memset(q + 32, neg ? 0xff : 0, 32);
            uint8_t qh = 0;
            for (int l = 0; l < 4; ++l) qh |= (uint8_t) (((k >> 8) & 3) << (2 * l));
            std::memset(b + 66, qh, 8);
            std::memset(b + 74, 0xff, 8);
            break;
        }
        case GGML_TYPE_IQ3_S: {   // qs[64] low index bits, qh[8] (1 bit an index), signs[32], scales[4]
            std::memset(q, k & 255, 64);
            std::memset(b + 66, (k >> 8) & 1 ? 0xff : 0, 8);
            std::memset(b + 74, neg ? 0xff : 0, 32);
            std::memset(b + 106, 0xff, 4);
            break;
        }
        default: break;
    }
}

// the grid index whose entries have the largest magnitude (largest max |v|, then largest sum |v|)
int max_grid_index(ggml_type t) {
    const auto* tr = ggml_get_type_traits(t);
    std::vector<uint8_t> b(ggml_type_size(t));
    std::vector<float> v(QK);
    int best = 0;
    double best_max = -1, best_sum = -1;
    for (int k = 0; k < grid_size(t); ++k) {
        grid_block(t, k, false, 1.0f, b.data());
        tr->to_float(b.data(), v.data(), QK);
        double mx = 0, sm = 0;
        for (float x : v) mx = std::max(mx, (double) std::fabs(x)), sm += std::fabs(x);
        if (mx > best_max || (mx == best_max && sm > best_sum)) best = k, best_max = mx, best_sum = sm;
    }
    return best;
}

// rows of n values: r % 3 == 0 the max grid entry with negative signs, 1 positive, 2 random codes
std::vector<uint8_t> make_rows(ggml_type t, int n, int rows, std::mt19937& rng, bool max_cases) {
    const size_t bb = ggml_type_size(t), rb = (size_t) (n / ggml_blck_size(t)) * bb;
    std::vector<uint8_t> w(rb * (size_t) rows);
    for (auto& x : w) x = (uint8_t) rng();
    const int km = grid_size(t) > 0 ? max_grid_index(t) : 0;
    for (int r = 0; r < rows; ++r)
        for (size_t o = 0; o < rb; o += bb) {
            uint8_t* b = w.data() + (size_t) r * rb + o;
            const int kind = max_cases ? r % 3 : 2;
            if (kind < 2 && grid_size(t) > 0) {
                grid_block(t, km, kind == 0, 0.5f, b);
            } else if (kind < 2 && t == GGML_TYPE_Q3_K) {
                std::memset(b, 0, bb);   // q = 0, no high bit, 6-bit scale 0: (0 - 4) * (0 - 32) = 128, the largest
                put_half(b + 108, kind == 0 ? -0.5f : 0.5f);
            } else if (kind < 2 && t == GGML_TYPE_Q2_K) {
                std::memset(b, 0xff, bb);   // q = 3, scale 15, min 15
                put_half(b + 80, 0.5f);
                put_half(b + 82, kind == 0 ? 0.0f : 0.25f);
            } else if (kind < 2 && t == GGML_TYPE_IQ4_NL) {   // block 18 B: d, 16 bytes of nibbles; index 0 = -127
                std::memset(b, 0, bb);
                put_half(b, kind == 0 ? 0.5f : -0.5f);
            } else {
                // random codes at a sane scale (Q2_K / Q3_K keep d after their scales)
                const size_t off = t == GGML_TYPE_Q2_K ? 80 : t == GGML_TYPE_Q3_K ? 108 : 0;
                put_half(b + off, 0.003f + 0.002f * (float) (rng() % 100) / 100.0f);
                if (t == GGML_TYPE_Q2_K) put_half(b + 82, 0.001f);
            }
        }
    return w;
}

// activations: Q8_K (d f32, qs[256], bsums[16]) or Q8_0 (d f16, qs[32]); pattern 0: +127, 1: -127, 2: +-127
// alternating, 3+: random.  The range is the quantizers' [-127, 127] (activation_range_check): ggml's i-quant dots
// (and the lane's kernels, which match them) negate the activation by the weight's sign in int8, so a -128 would wrap -
// a value no lane input can hold.
std::vector<uint8_t> make_act(ggml_type at, int n, int pattern, std::mt19937& rng) {
    const size_t bb = ggml_type_size(at);
    const int bk = (int) ggml_blck_size(at);
    std::vector<uint8_t> a((size_t) (n / bk) * bb, 0);
    for (int b = 0; b < n / bk; ++b) {
        uint8_t* p = a.data() + (size_t) b * bb;
        int8_t q[QK];
        for (int j = 0; j < bk; ++j)
            q[j] = (int8_t) (pattern == 0 ? 127 : pattern == 1 ? -127 : pattern == 2 ? (j & 1 ? 127 : -127)
                                                                                  : (int) (rng() % 255) - 127);
        const float d = 1.0f / 127.0f;
        if (at == GGML_TYPE_Q8_K) {
            std::memcpy(p, &d, 4);
            std::memcpy(p + 4, q, QK);
            for (int s = 0; s < 16; ++s) {
                int16_t sum = 0;
                for (int j = 0; j < 16; ++j) sum = (int16_t) (sum + q[16 * s + j]);
                std::memcpy(p + 4 + QK + 2 * s, &sum, 2);
            }
        } else {
            put_half(p, d);
            std::memcpy(p + 2, q, (size_t) bk);
        }
    }
    return a;
}

std::vector<double> deq_act(ggml_type at, const uint8_t* a, int n) {
    std::vector<double> v((size_t) n);
    const size_t bb = ggml_type_size(at);
    const int bk = (int) ggml_blck_size(at);
    for (int b = 0; b < n / bk; ++b) {
        const uint8_t* p = a + (size_t) b * bb;
        float d;
        const int8_t* q;
        if (at == GGML_TYPE_Q8_K) {
            std::memcpy(&d, p, 4);
            q = (const int8_t*) (p + 4);
        } else {
            ggml_fp16_t h;
            std::memcpy(&h, p, 2);
            d = ggml_fp16_to_fp32(h);
            q = (const int8_t*) (p + 2);
        }
        for (int j = 0; j < bk; ++j) v[(size_t) (b * bk + j)] = (double) d * q[j];
    }
    return v;
}

// the double reference of row r: (sum, sum of |terms|)
std::pair<double, double> ref_dot(ggml_type t, const uint8_t* row, int n, const std::vector<double>& a) {
    std::vector<float> w((size_t) n);
    ggml_get_type_traits(t)->to_float(row, w.data(), n);
    double s = 0, m = 0;
    for (int j = 0; j < n; ++j) s += (double) w[(size_t) j] * a[(size_t) j], m += std::fabs((double) w[(size_t) j] * a[(size_t) j]);
    return {s, m};
}

// a lane kernel in the dot-rows shape: out[t][r] for rows [r0, r1), nt tokens
using RowsFn = std::function<void(const void* const* act, int nt, float* const* out, int r0, int r1)>;

constexpr float kSentinel = -12345.0f;

// 1 + 2 for a multi-token kernel: bitwise per row across nt and token order, row ranges, untouched rows
void kernel_cases(const std::string& name, const RowsFn& fn, const std::vector<std::vector<uint8_t>>& acts) {
    const int NT = (int) acts.size();   // 8
    std::vector<const void*> ap;
    for (const auto& a : acts) ap.push_back(a.data());
    // alone
    std::vector<std::vector<float>> alone((size_t) NT, std::vector<float>(kRows));
    for (int t = 0; t < NT; ++t) {
        float* o[1] = {alone[(size_t) t].data()};
        fn(&ap[(size_t) t], 1, o, 0, kRows);
    }
    bool batch_same = true;
    for (int nt = 2; nt <= NT; ++nt)
        for (int rev = 0; rev < 2; ++rev) {
            std::vector<std::vector<float>> out((size_t) nt, std::vector<float>(kRows));
            std::vector<const void*> a((size_t) nt);
            std::vector<float*> o((size_t) nt);
            for (int t = 0; t < nt; ++t) {
                const int src = rev ? nt - 1 - t : t;
                a[(size_t) t] = ap[(size_t) src];
                o[(size_t) t] = out[(size_t) t].data();
            }
            fn(a.data(), nt, o.data(), 0, kRows);
            for (int t = 0; t < nt; ++t) {
                const int src = rev ? nt - 1 - t : t;
                batch_same = batch_same && std::memcmp(out[(size_t) t].data(), alone[(size_t) src].data(), kRows * 4) == 0;
            }
        }
    check(batch_same, name + ": nt = 1 vs nt = 2..8 (both token orders) bit-identical per row");
    bool ranges_ok = true;
    for (int nt : {1, 3, 8}) {
        for (const auto& rg : std::vector<std::pair<int, int>>{{3, 11}, {5, 6}, {1, 13}, {0, 13}, {7, 13}}) {
            std::vector<std::vector<float>> out((size_t) nt, std::vector<float>(kRows, kSentinel));
            std::vector<float*> o((size_t) nt);
            for (int t = 0; t < nt; ++t) o[(size_t) t] = out[(size_t) t].data();
            fn(ap.data(), nt, o.data(), rg.first, rg.second);
            for (int t = 0; t < nt; ++t)
                for (int r = 0; r < kRows; ++r) {
                    const bool in = r >= rg.first && r < rg.second;
                    const float want = in ? alone[(size_t) t][(size_t) r] : kSentinel;
                    ranges_ok = ranges_ok && std::memcmp(&out[(size_t) t][(size_t) r], &want, 4) == 0;
                }
        }
    }
    check(ranges_ok, name + ": row ranges [3,11) [5,6) [1,13) [0,13) [7,13) at nt 1/3/8: same rows, others untouched");
}

// 3: a kernel's rows vs the double reference, every activation pattern
void reference_case(const std::string& name, const RowsFn& fn, ggml_type t, const uint8_t* w, size_t rb, int n,
                    ggml_type at, const std::vector<std::vector<uint8_t>>& acts, bool gu = false, size_t up_off = 0) {
    double worst = 0;
    for (const auto& a : acts) {
        const auto av = deq_act(at, a.data(), n);
        std::vector<float> out(kRows);
        const void* ap[1] = {a.data()};
        float* o[1] = {out.data()};
        fn(ap, 1, o, 0, kRows);
        for (int r = 0; r < kRows; ++r) {
            double want, scale;
            if (gu) {   // silu(g) * u
                const auto g = ref_dot(t, w + (size_t) r * rb, n, av), u = ref_dot(t, w + up_off + (size_t) r * rb, n, av);
                want = g.first / (1.0 + std::exp(-g.first)) * u.first;
                scale = std::max(1e-30, (std::fabs(g.first) + g.second) * (std::fabs(u.first) + u.second));
            } else {
                const auto s = ref_dot(t, w + (size_t) r * rb, n, av);
                want = s.first;
                scale = std::max(1e-30, s.second);
            }
            worst = std::max(worst, std::fabs((double) out[(size_t) r] - want) / scale);
        }
    }
    char buf[48];
    std::snprintf(buf, sizeof buf, "%.1e", worst);
    check(worst < 1e-5, name + ": max-magnitude grid rows, +-127 activations vs double ref (" + buf + " < 1e-5)");
}

// the lane's activations never hold -128: ggml-cpu's Q8_K / Q8_0 quantizers on adversarial rows (one extreme against
// many values at the opposite extreme, ties, a single nonzero, huge and tiny scales)
void activation_range_check(std::mt19937& rng) {
    bool ok = true;
    for (ggml_type at : {GGML_TYPE_Q8_K, GGML_TYPE_Q8_0}) {
        const auto* tr = ggml_get_type_traits_cpu(at);
        const int n = 2048;
        std::vector<uint8_t> q(ggml_row_size(at, n));
        std::vector<float> x((size_t) n);
        for (int c = 0; c < 64; ++c) {
            std::normal_distribution<float> nd(0.0f, c % 2 ? 1e-3f : 1e3f);
            for (auto& v : x) v = nd(rng);
            if (c % 4 == 0) for (int j = 0; j < n; ++j) x[(size_t) j] = j % 256 == 0 ? 1.0f : -1.0f;
            if (c % 4 == 1) for (int j = 0; j < n; ++j) x[(size_t) j] = j % 256 == 7 ? -2.0f : 1.999f;
            if (c % 4 == 2) for (int j = 0; j < n; ++j) x[(size_t) j] = j % 32 == 3 ? -5.0f : 0.0f;
            tr->from_float(x.data(), q.data(), n);
            const size_t bb = ggml_type_size(at), off = at == GGML_TYPE_Q8_K ? 4 : 2;
            const int bk = (int) ggml_blck_size(at);
            for (int b = 0; b < n / bk; ++b)
                for (int j = 0; j < bk; ++j) ok = ok && (int8_t) q[(size_t) b * bb + off + (size_t) j] != -128;
        }
    }
    check(ok, "ggml-cpu's Q8_K / Q8_0 activation quantizers stay in [-127, 127] (the lane's input domain)");
}

void type_cases(ggml_type t, std::mt19937& rng) {
    const std::string tn = ggml_type_name(t);
    const auto* tc = ggml_get_type_traits_cpu(t);
    const ggml_type at = tc->vec_dot_type;
    const int n = 2048;
    const size_t rb = ggml_row_size(t, n);
    std::vector<uint8_t> w = make_rows(t, n, kRows, rng, true);
    std::vector<std::vector<uint8_t>> acts;
    for (int p = 0; p < 8; ++p) acts.push_back(make_act(at, n, p, rng));
    const std::vector<std::vector<uint8_t>> extremes(acts.begin(), acts.begin() + 4);

    // ggml-cpu's vec_dot: the lane's nt = 1 kernel
    const RowsFn ggml_rows = [&](const void* const* a, int nt, float* const* o, int r0, int r1) {
        for (int r = r0; r < r1; ++r)
            for (int k = 0; k < nt; ++k) tc->vec_dot(n, &o[k][r], 0, w.data() + (size_t) r * rb, 0, a[k], 0, 1);
    };
    kernel_cases(tn + " ggml vec_dot", ggml_rows, acts);
    reference_case(tn + " ggml vec_dot", ggml_rows, t, w.data(), rb, n, at, extremes);

    if (kc::iq256_supported((int) t)) {
        if (kc::cpu_avx2_ok()) {
            const RowsFn f = [&](const void* const* a, int nt, float* const* o, int r0, int r1) {
                kc::iq256_rows((int) t, w.data(), rb, n, a, nt, o, r0, r1);
            };
            kernel_cases(tn + " iq256_rows", f, acts);
            reference_case(tn + " iq256_rows", f, t, w.data(), rb, n, at, extremes);
        } else {
            skip(tn + " iq256_rows (no AVX2)");
        }
    }
    if (kc::iq512_supported((int) t)) {
        if (kc::cpu_avx512_ok()) {
            const RowsFn f = [&](const void* const* a, int nt, float* const* o, int r0, int r1) {
                kc::iq512_rows((int) t, w.data(), rb, n, a, nt, o, r0, r1);
            };
            kernel_cases(tn + " iq512_rows", f, acts);
            reference_case(tn + " iq512_rows", f, t, w.data(), rb, n, at, extremes);
        } else {
            skip(tn + " iq512_rows (needs AVX-512 VNNI + VBMI; the test PC's Skylake-SP takes iq256)");
        }
    }
    if (kc::kq_type_ok((int) t)) {
        if (kc::kq_avx512_ok()) {
            const RowsFn f = [&](const void* const* a, int nt, float* const* o, int r0, int r1) {
                kc::kq_rows((int) t, w.data(), rb, n, a, nt, o, r0, r1);
            };
            kernel_cases(tn + " kq_rows (offset trick)", f, acts);
            reference_case(tn + " kq_rows (offset trick)", f, t, w.data(), rb, n, at, extremes);
        } else {
            skip(tn + " kq_rows (no AVX-512BW)");
        }
    }
    if (t == GGML_TYPE_IQ4_NL) {
        if (kc::cpu_avx2_ok()) {
            const RowsFn f = [&](const void* const* a, int nt, float* const* o, int r0, int r1) {
                kc::iq4nl256_down_rows(w.data(), rb, n, a, nt, o, r0, r1);
            };
            kernel_cases(tn + " iq4nl256_down_rows", f, acts);
            reference_case(tn + " iq4nl256_down_rows", f, t, w.data(), rb, n, at, extremes);
        } else {
            skip(tn + " iq4nl256_down_rows (no AVX2)");
        }
    }
}

// the lane's dispatchers: native_gu_rows (nt = 1 ggml-cpu, nt > 1 the multi-token kernel) and native_down_rows, and
// the multi-token gate/up kernels themselves
void dispatcher_cases(ggml_type gu, ggml_type dn, std::mt19937& rng) {
    const int E = 2048, FF = 512;
    kc::NativeFmt f;
    std::string why;
    const std::string name = std::string(ggml_type_name(gu)) + " / " + ggml_type_name(dn);
    if (!kc::native_fmt(gu, dn, E, FF, f, why)) {
        check(false, name + ": native_fmt (" + why + ")");
        return;
    }
    // a blob of kRows gate rows, kRows up rows (at up_off as the fmt says for FF rows), kRows down rows
    std::vector<uint8_t> blob(f.bytes, 0);
    const auto g = make_rows(gu, E, kRows, rng, true), u = make_rows(gu, E, kRows, rng, true);
    const auto d = make_rows(dn, FF, kRows, rng, true);
    std::memcpy(blob.data(), g.data(), g.size());
    std::memcpy(blob.data() + f.up_off, u.data(), u.size());
    std::memcpy(blob.data() + f.down_off, d.data(), d.size());
    std::vector<std::vector<uint8_t>> acts, hqs;
    for (int p = 0; p < 8; ++p) acts.push_back(make_act((ggml_type) f.gu_act, E, p, rng));
    for (int p = 0; p < 8; ++p) hqs.push_back(make_act((ggml_type) f.d_act, FF, p, rng));
    std::vector<const void*> ap, hp;
    for (auto& a : acts) ap.push_back(a.data());
    for (auto& h : hqs) hp.push_back(h.data());

    const auto close_rows = [&](const std::string& what, const std::function<void(const void* const*, int, float* const*)>& fn,
                                const std::vector<const void*>& in) {
        std::vector<std::vector<float>> one(8, std::vector<float>(kRows)), many(8, std::vector<float>(kRows));
        for (int t = 0; t < 8; ++t) {
            float* o[1] = {one[(size_t) t].data()};
            fn(&in[(size_t) t], 1, o);
        }
        bool ok = true;
        double worst = 0;
        for (int nt = 2; nt <= 8; ++nt) {
            std::vector<float*> o((size_t) nt);
            for (int t = 0; t < nt; ++t) o[(size_t) t] = many[(size_t) t].data();
            fn(in.data(), nt, o.data());
            for (int t = 0; t < nt; ++t)
                for (int r = 0; r < kRows; ++r) {
                    const double a = one[(size_t) t][(size_t) r], b = many[(size_t) t][(size_t) r];
                    const double e = std::fabs(a - b) / std::max(1e-6, std::fabs(a));
                    worst = std::max(worst, e);
                    ok = ok && std::isfinite(b) && e < 2e-5;
                }
        }
        char buf[48];
        std::snprintf(buf, sizeof buf, "%.1e", worst);
        check(ok, name + ": " + what + " nt = 1 (ggml-cpu) vs nt = 2..8 per row within " + buf + " < 2e-5");
    };
    close_rows("native_gu_rows", [&](const void* const* a, int nt, float* const* o) {
        kc::native_gu_rows(f, blob.data(), a, nt, o, 0, kRows);
    }, ap);
    close_rows("native_down_rows", [&](const void* const* h, int nt, float* const* o) {
        kc::native_down_rows(f, blob.data(), h, nt, o, 0, kRows);
    }, hp);
    const auto gu_kernel = [&](const std::string& kn, bool ok_cpu, void (*k)(int, const uint8_t*, size_t, size_t, int,
                                                                           const void* const*, int, float* const*, int, int)) {
        if (!ok_cpu) {
            skip(name + " " + kn + " (CPU lacks it)");
            return;
        }
        const RowsFn fn = [&](const void* const* a, int nt, float* const* o, int r0, int r1) {
            k((int) gu, blob.data(), f.gu_row, f.up_off, E, a, nt, o, r0, r1);
        };
        kernel_cases(name + " " + kn, fn, acts);
        reference_case(name + " " + kn, fn, gu, blob.data(), f.gu_row, E, (ggml_type) f.gu_act,
                       std::vector<std::vector<uint8_t>>(acts.begin(), acts.begin() + 4), true, f.up_off);
    };
    if (kc::iq256_supported((int) gu)) gu_kernel("iq256_gu_rows", kc::cpu_avx2_ok(), kc::iq256_gu_rows);
    if (kc::iq512_supported((int) gu)) gu_kernel("iq512_gu_rows", kc::cpu_avx512_ok(), kc::iq512_gu_rows);
}

}  // namespace

int main() {
    ggml_cpu_init();   // (ggml-cpu's fp16 tables: native_fmt does this in the engine)
    std::mt19937 rng(14008);
    std::printf("B14.8 lane parity (avx2 %d, avx512 vnni+vbmi %d, avx512bw kq %d)\n", (int) kc::cpu_avx2_ok(),
                (int) kc::cpu_avx512_ok(), (int) kc::kq_avx512_ok());
    activation_range_check(rng);
    for (ggml_type t : {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,
                        GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_IQ4_NL})
        type_cases(t, rng);
    dispatcher_cases(GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS, rng);
    dispatcher_cases(GGML_TYPE_IQ2_S, GGML_TYPE_Q2_K, rng);
    dispatcher_cases(GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, rng);
    if (g_fail) {
        std::printf("glm_lane_parity_test: %d FAILED (%d skipped)\n", g_fail, g_skip);
        return 1;
    }
    std::printf("glm_lane_parity_test: all passed (%d skipped)\n", g_skip);
    return 0;
}
