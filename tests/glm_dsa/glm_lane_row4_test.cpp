// tests/glm_dsa/glm_lane_row4_test.cpp - glm-dsa B14.9: the lockstep 4-row lane GEMV (glm_lane_row4) against
// ggml-cpu's vec_dot, bit for bit.  CPU only; on a CPU (or a ggml-cpu build) without AVX2 the row4 cases are
// reported as skipped and the hooks must decline.
//
//   1. row4_rows vs ggml vec_dot per row on random blocks of every type (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, Q2_K,
//      Q3_K, IQ1_S, IQ4_XS), n = 256 / 1024 / 2048 / 6144, row counts 1..9 at row offsets 0, 1, 3 (B14.8 style:
//      counts that are not a multiple of 4 take the per-row tail), rows outside the range untouched; row4_dot on
//      scattered rows;
//   2. extremes: +-1 activations (every Q8 byte +-127 from ggml's quantizer) and raw Q8_K activations of -128 / +127
//      bytes (the quantizer never writes -128), all-0xFF and all-0x00 weight bytes (largest / smallest grid indices
//      and scales, all signs negative / positive), and each block's qh / scale field all-0xFF or all-0x00 under
//      random grid bytes (IQ1_S qh: grid high bits, scale, delta sign; IQ4_XS scales_h + scales_l; IQ2_S qh +
//      scales; IQ2_XS scales; Q2_K scales; Q3_K hmask);
//   3. IQ1_S: with AVX2 the probe found one of the two forms of ggml's scalar delta term;
//   4. the lane hooks: native_gu_rows_split (plain and with GLM's swiglu clamp, limits inf / 10 / 0.05) and
//      native_down_rows_split at nt = 1 with the switch off (Maya's ggml loop) vs on (row4), bit for bit, for each
//      type as gate/up and as down, plus the served pair (IQ1_S gate/up, IQ4_XS down) at GLM-5.3's 6144 x 2048;
//   5. lane_row4_set, the per-request switch: mode, type mask (a masked-out type declines), back to the env default.
//
// `glm_lane_row4_test --bench [seconds]` adds a single-thread microbenchmark: GB/s of weight rows streamed by ggml's
// vec_dot loop vs row4_rows, for an L2-resident (~1 MB) and a DRAM-sized (~192 MB) matrix: 6144-wide rows (GLM-5.3's
// gate/up rows, IQ1_S among them) and IQ4_XS at 2048 (its down rows).
#include "strata/kernels/cpu/glm_lane_row4.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace kc = strata::kernels::cpu;

namespace {

int g_fail = 0, g_ok = 0, g_skip = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("  %-90s FAIL\n", what.c_str());
        ++g_fail;
    } else {
        ++g_ok;
    }
}

struct Ty {
    int t;
    const char* name;
};
const Ty kTypes[] = {{16, "IQ2_XXS"}, {17, "IQ2_XS"}, {22, "IQ2_S"}, {18, "IQ3_XXS"}, {10, "Q2_K"}, {11, "Q3_K"},
                     {19, "IQ1_S"},   {23, "IQ4_XS"}};
constexpr int kNTypes = (int) (sizeof kTypes / sizeof kTypes[0]);

int type_index(int t) {
    for (int i = 0; i < kNTypes; ++i)
        if (kTypes[i].t == t) return i;
    return 0;
}

void put_half(uint8_t* p, float v) {
    const ggml_fp16_t h = ggml_fp32_to_fp16(v);
    std::memcpy(p, &h, sizeof h);
}

// weight bytes: random, all 0xFF, all 0x00, or random with each block's qh / scale field (hi_field) all 0xFF or all
// 0x00; the fp16 super-block scales stay sane in every case
enum Fill { kRandom, kAllFF, kAll00, kHiFF, kHi00 };

// the per-block field of grid high bits / sub-block scales the kHi* fills set (len 0: no contiguous one)
void hi_field(int type, size_t& off, size_t& len) {
    switch (type) {
        case 19: off = 34; len = 16; break;   // IQ1_S qh[8]: 4 x 3 grid high bits | scale << 12 | delta sign << 15
        case 23: off = 2; len = 6; break;     // IQ4_XS scales_h, scales_l[4]
        case 22: off = 66; len = 16; break;   // IQ2_S qh[8], scales[8]
        case 17: off = 66; len = 8; break;    // IQ2_XS scales[8]
        case 10: off = 0; len = 16; break;    // Q2_K scales | mins << 4
        case 11: off = 0; len = 32; break;    // Q3_K hmask
        default: off = 0; len = 0; break;     // IQ2_XXS / IQ3_XXS: sign and scale words between the grid bytes
    }
}

// random bytes (or a Fill pattern) with sane fp16 super-block scales (the fields' offsets in each block)
void fill_rows(int type, uint8_t* p, size_t bytes, std::mt19937& rng, Fill fill) {
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> sc(-0.02f, 0.02f);
    const bool fixed = fill == kAllFF || fill == kAll00;
    for (size_t i = 0; i < bytes; ++i) p[i] = fill == kAllFF ? 0xFF : fill == kAll00 ? 0x00 : (uint8_t) byte(rng);
    const size_t bs = ggml_type_size((ggml_type) type);
    size_t ho = 0, hl = 0;
    hi_field(type, ho, hl);
    for (size_t o = 0; o + bs <= bytes; o += bs) {
        uint8_t* b = p + o;
        if (fill == kHiFF || fill == kHi00) std::memset(b + ho, fill == kHiFF ? 0xFF : 0x00, hl);
        switch (type) {
            case 10: put_half(b + 80, fixed ? 0.02f : sc(rng)); put_half(b + 82, fixed ? -0.02f : sc(rng)); break;
            case 11: put_half(b + 108, fixed ? -0.02f : sc(rng)); break;
            default: put_half(b, fixed ? 0.02f : sc(rng)); break;
        }
    }
}

std::vector<uint8_t> quant_q8k(const std::vector<float>& x) {
    std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_Q8_K, (int64_t) x.size()));
    ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(x.data(), q.data(), (int64_t) x.size());
    return q;
}

std::vector<float> rand_act(int n, std::mt19937& rng, int mode) {
    std::vector<float> x((size_t) n);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> coin(0, 1);
    for (int i = 0; i < n; ++i) {
        if (mode == 0) x[(size_t) i] = nd(rng) * (i % 97 == 0 ? 30.f : 1.f);   // a few outliers
        else x[(size_t) i] = coin(rng) ? 1.f : -1.f;                            // every Q8 byte +-127
    }
    return x;
}

// a raw Q8_K activation (float d, int8 qs[256], int16 bsums[16] per block) whose bytes are -128 or +127 at random:
// ggml's quantizer never writes -128, a Q8_K made elsewhere may; bsums consistent with the bytes
std::vector<uint8_t> q8k_extreme(int n, std::mt19937& rng) {
    std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_Q8_K, n));
    const size_t bs = ggml_type_size(GGML_TYPE_Q8_K);
    std::uniform_int_distribution<int> coin(0, 1);
    for (int b = 0; b < n / 256; ++b) {
        uint8_t* p = q.data() + (size_t) b * bs;
        const float d = 0.0123f;
        std::memcpy(p, &d, sizeof d);
        int8_t qs[256];
        for (int j = 0; j < 256; ++j) qs[j] = coin(rng) ? (int8_t) 127 : (int8_t) -128;
        std::memcpy(p + 4, qs, sizeof qs);
        for (int g = 0; g < 16; ++g) {
            int s = 0;
            for (int j = 0; j < 16; ++j) s += qs[16 * g + j];
            const int16_t s16 = (int16_t) s;
            std::memcpy(p + 260 + 2 * g, &s16, sizeof s16);
        }
    }
    return q;
}

// activation modes: 0 normal with outliers, 1 +-1 (bytes +-127), 2 raw -128 / +127 bytes
std::vector<uint8_t> act_q8k(int n, std::mt19937& rng, int mode) {
    return mode == 2 ? q8k_extreme(n, rng) : quant_q8k(rand_act(n, rng, mode));
}

bool same_bits(const float* a, const float* b, int n) { return std::memcmp(a, b, sizeof(float) * (size_t) n) == 0; }

// data variants: 0, 1 random; 2 +-127 acts; 3 all-0xFF weights + +-127 acts (B14.9's four); 4 all-0x00 weights;
// 5 -128 / +127 acts; 6 all-0xFF weights + -128 / +127 acts; 7 qh / scale field all-0xFF + -128 / +127 acts;
// 8 qh / scale field all-0x00 (7 and 8 only for types with such a field)
struct Variant {
    Fill fill;
    int act;
};
const Variant kVariants[] = {{kRandom, 0}, {kRandom, 0}, {kRandom, 1}, {kAllFF, 1}, {kAll00, 0},
                             {kRandom, 2}, {kAllFF, 2},  {kHiFF, 2},   {kHi00, 0}};
constexpr int kNVariants = (int) (sizeof kVariants / sizeof kVariants[0]);

// ---- 1 + 2: row4_rows vs vec_dot
void test_rows(std::mt19937& rng) {
    std::printf("row4_rows vs ggml vec_dot (bit-exact), rows 1..9 at offsets 0/1/3:\n");
    const int kNs[] = {256, 1024, 2048, 6144};
    for (const Ty& ty : kTypes) {
        if (!kc::row4_supported(ty.t)) {
            std::printf("  %-10s skipped (%s)\n", ty.name,
                        kc::row4_supported(16) ? "declined on this AVX2 path: see the IQ1_S probe check below"
                                               : "no AVX2 here or ggml-cpu built without it");
            ++g_skip;
            continue;
        }
        const ggml_vec_dot_t dot = ggml_get_type_traits_cpu((ggml_type) ty.t)->vec_dot;
        size_t ho = 0, hl = 0;
        hi_field(ty.t, ho, hl);
        int cases = 0;
        for (int n : kNs) {
            const size_t rb = ggml_row_size((ggml_type) ty.t, n);
            constexpr int kRows = 16;
            for (int variant = 0; variant < kNVariants; ++variant) {
                const Variant& v = kVariants[variant];
                if ((v.fill == kHiFF || v.fill == kHi00) && hl == 0) continue;
                std::vector<uint8_t> w(rb * kRows);
                fill_rows(ty.t, w.data(), w.size(), rng, v.fill);
                const auto q = act_q8k(n, rng, v.act);
                std::vector<float> ref(kRows);
                for (int r = 0; r < kRows; ++r) dot(n, &ref[(size_t) r], 0, w.data() + r * rb, 0, q.data(), 0, 1);
                for (int r0 : {0, 1, 3}) {
                    for (int cnt = 1; cnt <= 9; ++cnt) {
                        std::vector<float> out(kRows, -12345.f);
                        kc::row4_rows(ty.t, n, w.data(), rb, q.data(), out.data(), r0, r0 + cnt);
                        bool ok = same_bits(out.data() + r0, ref.data() + r0, cnt);
                        for (int r = 0; r < kRows; ++r)
                            if ((r < r0 || r >= r0 + cnt) && out[(size_t) r] != -12345.f) ok = false;
                        char buf[160];
                        std::snprintf(buf, sizeof buf, "%s n=%d variant=%d rows [%d,%d)", ty.name, n, variant, r0, r0 + cnt);
                        check(ok, buf);
                        if (!ok) {
                            for (int r = r0; r < r0 + cnt; ++r)
                                std::printf("    row %d: row4 %.9g ggml %.9g\n", r, out[(size_t) r], ref[(size_t) r]);
                        }
                        ++cases;
                    }
                }
            }
        }
        // direct row4_dot on four non-adjacent rows (the hook's gate/up pattern uses arbitrary pointers): random
        // activations at n = 2048, -128 / +127 ones at n = 6144
        const int kScatterN[2] = {2048, 6144};
        for (int s = 0; s < 2; ++s) {
            const int n = kScatterN[s];
            const size_t rb = ggml_row_size((ggml_type) ty.t, n);
            std::vector<uint8_t> w(rb * 11);
            fill_rows(ty.t, w.data(), w.size(), rng, kRandom);
            const auto q = act_q8k(n, rng, s == 0 ? 0 : 2);
            const int pick[4] = {9, 2, 7, 0};
            const uint8_t* rows[4] = {w.data() + pick[0] * rb, w.data() + pick[1] * rb, w.data() + pick[2] * rb,
                                      w.data() + pick[3] * rb};
            float out[4], ref[4];
            kc::row4_dot(ty.t, n, rows, q.data(), out);
            for (int k = 0; k < 4; ++k) dot(n, &ref[k], 0, rows[k], 0, q.data(), 0, 1);
            check(same_bits(out, ref, 4), std::string(ty.name) + " row4_dot on scattered rows n=" + std::to_string(n));
            ++cases;
        }
        std::printf("  %-10s %d cases\n", ty.name, cases);
    }
}

// ---- 3: which form of IQ1_S's delta term (accum1 += d * sumi1) the probe found in this ggml-cpu
void test_iq1s_form() {
    const int form = kc::row4_iq1s_form();
    std::printf("IQ1_S delta term in this ggml-cpu: %s\n",
                form == 2   ? "one fmaf (accum1 += d * sumi1 contracted)"
                : form == 1 ? "a multiply and an add"
                : form == 0 ? "NEITHER of row4's forms"
                            : "not probed (no AVX2 path)");
    if (kc::row4_supported(16)) {   // the AVX2 path is there: the probe must have matched ggml in one of the forms
        check(form == 1 || form == 2, "IQ1_S probe matched ggml's delta term");
        check(kc::row4_supported(19) == (form == 1 || form == 2), "IQ1_S supported iff the probe matched");
    } else {
        check(form == -1 && !kc::row4_supported(19), "no AVX2 path: IQ1_S not probed and unsupported");
    }
}

// ---- 4: the lane hooks, switch off vs on
struct HookCase {
    int gu, dn;   // kTypes indices
    int n_embd, n_ff;
};

void test_hooks(std::mt19937& rng) {
    std::printf("lane hooks (native_gu_rows_split / native_down_rows_split, nt = 1), MAYA_GLM_LANE_ROW4 off vs on:\n");
    std::vector<HookCase> hc;
    for (int i = 0; i < kNTypes; ++i) hc.push_back({i, (i + 3) % kNTypes, 1024, 768});   // every type as gu and down
    hc.push_back({type_index(19), type_index(23), 6144, 2048});   // the served pair at GLM-5.3's expert shape
    for (const HookCase& c : hc) {
        const Ty& gu = kTypes[c.gu];
        const Ty& dn = kTypes[c.dn];
        const int n_embd = c.n_embd, n_ff = c.n_ff;
        kc::NativeFmt f;
        std::string err;
        if (!kc::native_fmt(gu.t, dn.t, n_embd, n_ff, f, err)) {
            check(false, std::string("native_fmt ") + gu.name + "/" + dn.name + ": " + err);
            continue;
        }
        std::vector<uint8_t> blob(f.bytes);
        fill_rows(gu.t, blob.data(), f.up_off * 2, rng, kRandom);
        fill_rows(dn.t, blob.data() + f.down_off, f.bytes - f.down_off, rng, kRandom);
        std::vector<uint8_t> act(f.act_bytes);
        kc::native_quant_act(f, rand_act(n_embd, rng, 0).data(), act.data());
        const void* a[1] = {act.data()};
        int cases = 0;
        for (float lim : {std::numeric_limits<float>::infinity(), 10.0f, 0.05f}) {
            const std::vector<std::pair<int, int>> gu_ranges = {{0, n_ff}, {3, 11}, {5, 6}, {1, 14}, {0, 9}};
            for (const auto& rg : gu_ranges) {
                std::vector<float> off((size_t) n_ff, 7.f), on((size_t) n_ff, 7.f);
                float* o0[1] = {off.data()};
                float* o1[1] = {on.data()};
                kc::lane_row4_override(0);
                kc::native_gu_rows_split(f, blob.data(), blob.data() + f.up_off, a, 1, o0, rg.first, rg.second, lim);
                kc::lane_row4_override(1);
                kc::native_gu_rows_split(f, blob.data(), blob.data() + f.up_off, a, 1, o1, rg.first, rg.second, lim);
                char buf[160];
                std::snprintf(buf, sizeof buf, "gu %s n=%d limit %g rows [%d,%d)", gu.name, n_embd, lim, rg.first,
                              rg.second);
                check(same_bits(off.data(), on.data(), n_ff), buf);
                ++cases;
            }
        }
        // down: h from the gu output of the full range
        std::vector<float> ff((size_t) n_ff);
        float* of[1] = {ff.data()};
        kc::lane_row4_override(0);
        kc::native_gu_rows_split(f, blob.data(), blob.data() + f.up_off, a, 1, of, 0, n_ff, 10.0f);
        std::vector<uint8_t> hq(f.h_bytes);
        kc::native_quant_h(f, ff.data(), hq.data());
        const void* h[1] = {hq.data()};
        const std::vector<std::pair<int, int>> dn_ranges = {{0, n_embd}, {2, 9}, {7, 8}, {4, 17}};
        for (const auto& rg : dn_ranges) {
            std::vector<float> off((size_t) n_embd, 3.f), on((size_t) n_embd, 3.f);
            float* o0[1] = {off.data()};
            float* o1[1] = {on.data()};
            kc::lane_row4_override(0);
            kc::native_down_rows_split(f, blob.data() + f.down_off, h, 1, o0, rg.first, rg.second);
            kc::lane_row4_override(1);
            kc::native_down_rows_split(f, blob.data() + f.down_off, h, 1, o1, rg.first, rg.second);
            char buf[160];
            std::snprintf(buf, sizeof buf, "down %s n=%d rows [%d,%d)", dn.name, n_ff, rg.first, rg.second);
            check(same_bits(off.data(), on.data(), n_embd), buf);
            ++cases;
        }
        std::printf("  gu %-8s / down %-8s %4d x %4d  %d cases%s\n", gu.name, dn.name, n_embd, n_ff, cases,
                    kc::row4_supported(gu.t) ? "" : " (row4 unsupported here: both runs are ggml's)");
    }
    kc::lane_row4_override(-1);
}

// ---- 5: lane_row4_set, the per-request switch (mode + type mask), on the served pair
void test_setter(std::mt19937& rng, bool env_on) {
    std::printf("lane_row4_set (per-request switch): mode, type mask, back to the env default:\n");
    const int n_embd = 512, n_ff = 256;
    kc::NativeFmt f;
    std::string err;
    if (!kc::native_fmt(19, 23, n_embd, n_ff, f, err)) {
        check(false, "native_fmt IQ1_S/IQ4_XS: " + err);
        return;
    }
    std::vector<uint8_t> blob(f.bytes);
    fill_rows(19, blob.data(), f.up_off * 2, rng, kRandom);
    fill_rows(23, blob.data() + f.down_off, f.bytes - f.down_off, rng, kRandom);
    std::vector<uint8_t> act(f.act_bytes), hq(f.h_bytes);
    kc::native_quant_act(f, rand_act(n_embd, rng, 0).data(), act.data());
    kc::native_quant_h(f, rand_act(n_ff, rng, 0).data(), hq.data());
    std::vector<float> ff((size_t) n_ff), out((size_t) n_embd);
    const float inf = std::numeric_limits<float>::infinity();
    auto gu = [&] {
        return kc::lane_row4_gu(f, blob.data(), blob.data() + f.up_off, act.data(), ff.data(), 0, n_ff, inf);
    };
    auto dn = [&] { return kc::lane_row4_down(f, blob.data() + f.down_off, hq.data(), out.data(), 0, n_embd); };
    const bool s19 = kc::row4_supported(19), s23 = kc::row4_supported(23);
    const uint32_t b19 = kc::row4_type_bit(19), b23 = kc::row4_type_bit(23);

    kc::lane_row4_set(1);
    check(kc::lane_row4_on() && kc::lane_row4_types() == kc::kRow4AllTypes, "set(1): on, every type");
    check(gu() == s19 && dn() == s23, "set(1): IQ1_S gate/up and IQ4_XS down hooks taken iff supported");
    kc::lane_row4_set(1, kc::kRow4AllTypes & ~b19);
    check(kc::lane_row4_types() == (kc::kRow4AllTypes & ~b19), "set(1, all but IQ1_S): mask read back");
    check(!gu() && dn() == s23, "set(1, all but IQ1_S): gate/up declines, down as before");
    kc::lane_row4_set(1, b19);
    check(gu() == s19 && !dn(), "set(1, IQ1_S only): gate/up taken iff supported, down declines");
    kc::lane_row4_set(1, kc::kRow4AllTypes & ~(b19 | b23));
    check(!gu() && !dn(), "set(1, without IQ1_S / IQ4_XS): both decline");
    kc::lane_row4_set(0);
    check(!kc::lane_row4_on() && !gu() && !dn(), "set(0): off, both decline");
    kc::lane_row4_override(1);   // override keeps the mask
    check(kc::lane_row4_on() && kc::lane_row4_types() == kc::kRow4AllTypes, "override(1) after set(0): on, mask kept");
    kc::lane_row4_set(-1);
    check(kc::lane_row4_on() == env_on && kc::lane_row4_types() == kc::kRow4AllTypes,
          "set(-1): back to MAYA_GLM_LANE_ROW4, every type");
    std::printf("  done\n");
}

// ---- --bench: single-thread GB/s, ggml vec_dot loop vs row4_rows
double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// n = 6144: GLM-5.3's gate/up rows (IQ1_S: the served file's gate/up type); n = 2048: its down rows (IQ4_XS)
struct BenchCase {
    int t;
    const char* name;
    int n;
};
const BenchCase kBench[] = {{16, "IQ2_XXS", 6144}, {17, "IQ2_XS", 6144}, {22, "IQ2_S", 6144}, {18, "IQ3_XXS", 6144},
                            {10, "Q2_K", 6144},    {11, "Q3_K", 6144},   {19, "IQ1_S", 6144}, {23, "IQ4_XS", 2048}};

void bench(double secs) {
    std::mt19937 rng(7);
    std::printf("\nmicrobenchmark: one thread, GB/s of weight rows (best of runs); n = 6144 gate/up rows, n = 2048 down\n");
    std::printf("  NOTE: indicative only unless run on the test PC's Xeon Gold 6138 (numactl -N <node> -m <node>).\n");
    std::printf("  %-8s %6s %-10s %10s %10s %10s %8s\n", "type", "n", "matrix", "rows", "ggml GB/s", "row4 GB/s", "ratio");
    for (const BenchCase& bc : kBench) {
        if (!kc::row4_supported(bc.t)) {
            std::printf("  %-8s skipped\n", bc.name);
            continue;
        }
        const int n = bc.n;
        const auto q = quant_q8k(rand_act(n, rng, 0));
        const ggml_vec_dot_t dot = ggml_get_type_traits_cpu((ggml_type) bc.t)->vec_dot;
        const size_t rb = ggml_row_size((ggml_type) bc.t, n);
        for (size_t target : {(size_t) 1 << 20, (size_t) 192 << 20}) {
            const int rows = (int) std::max<size_t>(4, (target / rb) / 4 * 4);
            std::vector<uint8_t> w(rb * (size_t) rows);
            fill_rows(bc.t, w.data(), w.size(), rng, kRandom);
            std::vector<float> o1((size_t) rows), o2((size_t) rows);
            auto run_ggml = [&] {
                for (int r = 0; r < rows; ++r) dot(n, &o1[(size_t) r], 0, w.data() + (size_t) r * rb, 0, q.data(), 0, 1);
            };
            auto run_row4 = [&] { kc::row4_rows(bc.t, n, w.data(), rb, q.data(), o2.data(), 0, rows); };
            double best1 = 1e30, best2 = 1e30;
            const double t_end = now_s() + secs;
            int reps = 0;
            do {   // interleave the two so drift on the shared machine hits both alike
                double t0 = now_s();
                run_ggml();
                best1 = std::min(best1, now_s() - t0);
                t0 = now_s();
                run_row4();
                best2 = std::min(best2, now_s() - t0);
                ++reps;
            } while (now_s() < t_end || reps < 3);
            const double gb = (double) rb * rows / 1e9;
            check(same_bits(o1.data(), o2.data(), rows), std::string("bench output identical ") + bc.name);
            std::printf("  %-8s %6d %-10s %10d %10.2f %10.2f %7.2fx\n", bc.name, n,
                        target < ((size_t) 8 << 20) ? "~1 MB" : "~192 MB", rows, gb / best1, gb / best2, best1 / best2);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool do_bench = false;
    double secs = 1.0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) {
            do_bench = true;
            if (i + 1 < argc) secs = std::max(0.1, std::atof(argv[++i]));
        }
    }
    ggml_cpu_init();
    const bool env_on = kc::lane_row4_on();
    std::printf("glm_lane_row4_test (B14.9); ggml-cpu avx2=%d fma=%d f16c=%d; MAYA_GLM_LANE_ROW4 now %s\n",
                ggml_cpu_has_avx2(), ggml_cpu_has_fma(), ggml_cpu_has_f16c(), env_on ? "on" : "off");
    std::mt19937 rng(20261012);
    test_rows(rng);
    test_iq1s_form();
    test_hooks(rng);
    test_setter(rng, env_on);
    if (do_bench) bench(secs);
    std::printf("%d checks ok, %d failed, %d types skipped\n", g_ok, g_fail, g_skip);
    return g_fail ? 1 : 0;
}
