// Bit parity with unchanged scalar: all codes/scales, int8 extrema, tail groups, strides, row ranges, 1..17 tokens,
// rounding/FTZ modes and a complete real-shape expert on a worker pool. No model or GPU required.
#include "../src/ds4/ds4_gguf_cpu.hpp"
#include "../src/ds4/ds4_pool.hpp"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
namespace gg = strata::ds4::gguf;
using strata::ds4::Pool;
namespace {
size_t checked_values = 0, checked_cases = 0;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void equal(const std::vector<float>& ref, const std::vector<float>& got, const char* stage) {
    require(ref.size() == got.size(), "output size");
    if (std::memcmp(ref.data(), got.data(), ref.size() * sizeof(float))) {
        for (size_t j = 0; j < ref.size(); ++j)
            if (std::memcmp(&ref[j], &got[j], 4)) {
                uint32_t a, b; std::memcpy(&a, &ref[j], 4); std::memcpy(&b, &got[j], 4);
                std::fprintf(stderr, "%s value %zu: %08x != %08x\n", stage, j, a, b); break;
            }
        throw std::runtime_error(stage);
    }
    checked_values += ref.size(); ++checked_cases;
}
std::vector<uint8_t> activations(int k, int n, int pattern, std::mt19937& rng) {
    const int nb = k / 32; const size_t ab = gg::mx_act_bytes(k), qb = 128 * ((nb + 3) / 4);
    std::vector<uint8_t> a(std::max<size_t>(1, ab * std::max(1, n)), 0);
    for (int i = 0; i < n; ++i) {
        uint8_t* p = a.data() + size_t(i) * ab;
        float* d = (float*) (p + qb);
        int32_t* c = (int32_t*) (p + qb + 32 * ((nb + 7) / 8));
        for (int b = 0; b < nb; ++b) {
            int sum = 0;
            for (int j = 0; j < 32; ++j) {
                const int q = pattern == 0 ? 0 : (pattern == 1 || pattern == 5) ? -128 : (pattern == 2 || pattern == 6) ? 127
                              : pattern == 3 ? ((j & 1) ? -128 : 127) : int(rng() % 256) - 128;
                ((int8_t*) p)[(b >> 2) * 128 + (j >= 16 ? 64 : 0) + 16 * (b & 3) + (j & 15)] = (int8_t) q;
                sum += q;
            }
            const int l = 8 * (b >> 3) + (b & 1) * 4 + ((b & 7) >> 1);
            d[l] = pattern == 0 ? 0.0f : std::ldexp(1.0f + float(rng() % 1024) / 1024.0f, int(rng() % 17) - 12);
            c[l] = -12 * sum;
        }
    }
    return a;
}
void row_case(int k, int n, int pattern, int offset, unsigned seed, int tier) {
    std::mt19937 rng(seed);
    constexpr int R = 9, r0 = 1, r1 = 8, out_stride = (r1 - r0) + 2;
    const size_t rb = size_t(k) / 32 * 17 + (offset ? 3 : 0), ab = gg::mx_act_bytes(k);
    std::vector<uint8_t> w(size_t(R) * rb + offset + 16, 0x55);
    for (int r = 0; r < R; ++r)
        for (int b = 0; b < k / 32; ++b) {
            uint8_t* p = w.data() + offset + size_t(r) * rb + b * 17;
            // Cycle every exponent byte, and every nibble in each block.
            p[0] = uint8_t(r * 37 + b + seed);
            for (int j = 0; j < 16; ++j)
                p[j + 1] = pattern == 5 ? 0x77 : pattern == 6 ? 0xff : pattern == 4 ? uint8_t(rng())
                           : uint8_t(((j + b) & 15) | (((15 - j + r) & 15) << 4));
        }
    auto a = activations(k, n, pattern, rng);
    std::vector<const void*> ap(std::max(1, n));
    std::vector<float> ref(std::max(1, n) * out_stride, -12345.25f), got(ref);
    std::vector<float*> rp(std::max(1, n)), yp(std::max(1, n));
    for (int i = 0; i < n; ++i) { ap[i] = a.data() + size_t(i) * ab; rp[i] = ref.data() + i * out_stride + 1; yp[i] = got.data() + i * out_stride + 1; }
    gg::mx_rows_reference(w.data() + offset, rb, k, r0, r1, ap.data(), n, rp.data());
    gg::mx_rows_tier(tier, w.data() + offset, rb, k, r0, r1, ap.data(), n, yp.data(), offset ? 64 : 0);
    equal(ref, got, "row parity / guards");
    // Empty row range must preserve both guard and output values.
    gg::mx_rows_tier(tier, w.data() + offset, rb, k, r0, r0, ap.data(), n, yp.data());
    equal(ref, got, "empty row range");
}
std::vector<float> expert(int tier, int nt, int threads, const std::vector<uint8_t>& w, const std::vector<float>& x) {
    constexpr int E = 5120, FF = 2304, RB = 64;
    const size_t gr = size_t(E) / 32 * 17, dr = size_t(FF) / 32 * 17, mat = gr * FF;
    const size_t xa = gg::mx_act_bytes(E), ha = gg::mx_act_bytes(FF);
    std::vector<uint8_t> xq(xa * nt), hq(ha * nt);
    std::vector<float> h(size_t(FF) * nt), out(size_t(E) * nt);
    for (int t = 0; t < nt; ++t) gg::mx_quant(x.data() + size_t(t) * E, E, xq.data() + size_t(t) * xa);
    Pool pool(threads - 1);
    std::function<void(int)> gu = [&](int j) {
        const int lo = j * RB, hi = std::min(FF, lo + RB);
        std::vector<float> g(nt * RB), u(nt * RB); std::vector<const void*> ap(nt); std::vector<float*> gp(nt), up(nt);
        for (int t = 0; t < nt; ++t) { ap[t] = xq.data() + size_t(t) * xa; gp[t] = g.data() + t * RB; up[t] = u.data() + t * RB; }
        if (tier == -2) {
            gg::mx_rows_reference(w.data(), gr, E, lo, hi, ap.data(), nt, gp.data());
            gg::mx_rows_reference(w.data() + mat, gr, E, lo, hi, ap.data(), nt, up.data());
        } else {
            gg::mx_rows_tier(tier, w.data(), gr, E, lo, hi, ap.data(), nt, gp.data());
            gg::mx_rows_tier(tier, w.data() + mat, gr, E, lo, hi, ap.data(), nt, up.data());
        }
        for (int t = 0; t < nt; ++t)
            for (int r = lo; r < hi; ++r) { float gv = std::min(gp[t][r-lo], 10.0f), uv = std::clamp(up[t][r-lo], -10.0f, 10.0f); h[size_t(t)*FF+r] = gv / (1.0f + std::exp(-gv)) * uv; }
    };
    pool.run((FF + RB - 1) / RB, gu);
    for (int t = 0; t < nt; ++t) gg::mx_quant(h.data() + size_t(t) * FF, FF, hq.data() + size_t(t) * ha);
    std::function<void(int)> down = [&](int j) {
        const int lo = j * RB, hi = std::min(E, lo + RB);
        std::vector<const void*> ap(nt); std::vector<float*> yp(nt);
        for (int t = 0; t < nt; ++t) { ap[t] = hq.data() + size_t(t) * ha; yp[t] = out.data() + size_t(t)*E + lo; }
        if (tier == -2) gg::mx_rows_reference(w.data() + 2 * mat, dr, FF, lo, hi, ap.data(), nt, yp.data());
        else gg::mx_rows_tier(tier, w.data() + 2 * mat, dr, FF, lo, hi, ap.data(), nt, yp.data());
    };
    pool.run((E + RB - 1) / RB, down);
    return out;
}
void expert_cases() {
    constexpr size_t bytes = size_t(3) * 5120 * 2304 / 32 * 17;
    std::mt19937 rng(991); std::vector<uint8_t> w(bytes);
    for (size_t b = 0; b < bytes; b += 17) { w[b] = uint8_t(114 + rng() % 9); for (int j = 1; j < 17; ++j) w[b+j] = uint8_t(rng()); }
    std::vector<float> x(size_t(9) * 5120); for (auto& v : x) v = float(int(rng()%65536)-32768) / 32768.0f;
    for (int nt : {1, 4, 9}) {
        auto ref = expert(-2, nt, 1, w, x);
        for (int threads : {1, 12, 24}) equal(ref, expert(gg::kMxSsse3, nt, threads, w, x), "complete expert / threads");
    }
}
}  // namespace
int main(int argc, char** argv) {
    try {
        const bool smoke = argc > 1 && std::string(argv[1]) == "--smoke";
        const bool fallback = argc > 1 && std::string(argv[1]) == "--fallback-only";
        if (fallback) {
            require(!gg::mx_tier_ok(gg::kMxSsse3), "fallback-only needs a CPU without SSSE3");
            require(gg::mx_tier() == gg::kMxScalar, "expected scalar fallback selection");
            row_case(256, 3, 4, 1, 777, gg::kMxSsse3);
            std::puts("PASS unsupported SSSE3 falls back to scalar"); return 0;
        }
        if (!gg::mx_tier_ok(gg::kMxSsse3)) { std::puts("SKIP CPU has no SSSE3"); return 77; }
        require(std::string(gg::mx_tier_name(gg::kMxSsse3)) == "ssse3", "tier name");
        const std::vector<int> widths = smoke ? std::vector<int>{32,256,288} : std::vector<int>{0,32,64,96,224,256,288,512,2304,5120};
        const std::vector<int> ns = smoke ? std::vector<int>{1,3,9} : std::vector<int>{0,1,2,3,4,5,6,7,8,9,16,17};
        for (int k : widths) for (int n : ns) for (int pattern = 0; pattern < 7; ++pattern)
            for (int offset : {0,1}) row_case(k,n,pattern,offset,37 + k + n + pattern,gg::kMxSsse3);
        if (!smoke) {
            for (int mode : {FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}) { require(std::fesetround(mode) == 0,"rounding mode"); for(int k:{96,2304,5120}) for(int n:{1,4,9}) row_case(k,n,4,1,123,gg::kMxSsse3); }
            require(std::fesetround(FE_TONEAREST) == 0,"restore rounding");
#if defined(__x86_64__) || defined(_M_X64)
            const unsigned old = _mm_getcsr(); _mm_setcsr(old | 0x8040u);
            for(int k:{96,2304,5120}) row_case(k,3,4,1,0,gg::kMxSsse3);
            _mm_setcsr(old);
#endif
            row_case(256,3,4,1,123,-1);  // unsupported enum -> scalar
            if(!gg::mx_tier_ok(gg::kMxAvx2)) row_case(256,3,4,1,123,gg::kMxAvx2);
            expert_cases();
        }
        std::printf("PASS %zu comparisons, %zu float values bit-identical; selected=%s (%s)\n",checked_cases,checked_values,gg::mx_tier_name(gg::mx_tier()),gg::mx_tier_report());
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
