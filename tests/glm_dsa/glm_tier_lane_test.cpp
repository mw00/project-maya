// tests/glm_dsa/glm_tier_lane_test.cpp - glm-dsa B8.4 on the CPU (ggml-cpu, no GPU): the NUMA node-split expert layout
// against the unified one, consumer by consumer.
//
//   1. The CPU lane (consumer 4): GLM-5.3 experts (6144 x 2048) on IQ2_XXS / IQ2_S / IQ3_XXS gate/up and IQ3_XXS /
//      Q2_K / IQ4_XS down.  Each socket's half (glmtier::half_gu_rows / half_down_rows / half_experts_serial) against
//      Maya's unsplit lane (native_gu_rows_split / native_quant_h / native_down_rows_split, the weighted sum):
//      gate/up rows bit-identical (also when computed in row pieces not a multiple of 4 or 8), the half's down
//      activation bytes equal to its part of the whole one, the two partial y's added (one 6144-float add) within a
//      max relative error of 1e-5 of the unsplit lane; the clamped SwiGLU too.
//   2. The device's zero-copy pull (consumer 1, moe_fetch / moe_prefetch): the split_fetch kernel's word walk
//      (glm_tier_numa.cu unsplit_words) emulated on the host at the layout's word re-creates the unified blob.
//   3. Promotion and demotion (consumer 3): the six 2-D copies of split_ops, applied as cudaMemcpy2DAsync applies
//      them, move a unified blob into halves at a class's half-1 offset D and back.
//   4. Disk reads into the RAM tier (warm-up, misses, lookahead: fast_read_part's 4 KB-rounded chunks per role):
//      scatter_range per chunk equals the whole-blob scatter.
#include "glm_tier_lane.hpp"
#include "glm_tier_plan.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace gt = strata::core::glmtier;
namespace kc = strata::kernels::cpu;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-100s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// the byte offsets of a block's fp16 scales (random bytes elsewhere are valid codes for these types)
std::vector<size_t> scale_offsets(ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q2_K: return {80, 82};   // scales[16] qs[64] d dmin
        case GGML_TYPE_Q3_K: return {108};      // hmask[32] qs[64] scales[12] d
        case GGML_TYPE_Q4_K: return {0, 2};
        default: return {0};                    // IQ*, IQ4_XS, Q8_0: d first
    }
}

void fill(ggml_type t, uint8_t* p, size_t bytes, std::mt19937& rng, float d_lo, float d_hi) {
    for (size_t i = 0; i < bytes; ++i) p[i] = (uint8_t) rng();
    const size_t bb = ggml_type_size(t);
    std::uniform_real_distribution<float> du(d_lo, d_hi);
    for (size_t b = 0; b + bb <= bytes; b += bb)
        for (size_t o : scale_offsets(t)) {
            const ggml_fp16_t h = ggml_fp32_to_fp16(du(rng));
            std::memcpy(p + b + o, &h, sizeof h);
        }
}

// Maya's lane on unified blobs (fast_cpu_experts, serial): y = sum_i w_i * down_i(quant(silu(gate_i a) * up_i a))
void unsplit_lane(const kc::NativeFmt& f, int ne, const uint8_t* const* blob, const float* w, const float* x,
                  float limit, std::vector<float>& y, std::vector<std::vector<float>>& ff_all,
                  std::vector<std::vector<float>>& dn_all) {
    std::vector<uint8_t> act(kc::kNativeActBytes), hq(kc::kNativeHBytes);
    kc::native_quant_act(f, x, act.data());
    y.assign((size_t) f.n_embd, 0.0f);
    ff_all.assign((size_t) ne, std::vector<float>((size_t) f.n_ff));
    dn_all.assign((size_t) ne, std::vector<float>((size_t) f.n_embd));
    for (int i = 0; i < ne; ++i) {
        const void* a[1] = {act.data()};
        float* o[1] = {ff_all[(size_t) i].data()};
        kc::native_gu_rows_split(f, blob[i], blob[i] + f.up_off, a, 1, o, 0, (int) f.n_ff, limit);
        kc::native_quant_h(f, ff_all[(size_t) i].data(), hq.data());
        const void* h[1] = {hq.data()};
        float* d[1] = {dn_all[(size_t) i].data()};
        kc::native_down_rows_split(f, blob[i] + f.down_off, h, 1, d, 0, (int) f.n_embd);
    }
    for (int r = 0; r < (int) f.n_embd; ++r) {
        float m = 0.0f;
        for (int i = 0; i < ne; ++i) m += w[i] * dn_all[(size_t) i][(size_t) r];
        y[(size_t) r] = m;
    }
}

double max_rel(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        num = std::max(num, (double) std::fabs(a[i] - b[i]));
        den = std::max(den, (double) std::fabs(b[i]));
    }
    return den > 0.0 ? num / den : num;
}

// ---------------------------------------------------------------- 1. the lane

void lane_case(ggml_type gu, ggml_type dn, float limit, std::mt19937& rng) {
    const int64_t E = 6144, FF = 2048;
    const int ne = 3;
    char lim[32] = "";
    if (!std::isinf(limit)) std::snprintf(lim, sizeof lim, ", swiglu clamp %.2f", limit);
    const std::string name = std::string(ggml_type_name(gu)) + " / " + ggml_type_name(dn) + lim;
    kc::NativeFmt f, hf;
    std::string why;
    if (!kc::native_fmt(gu, dn, E, FF, f, why)) {
        check(false, name + ": native_fmt (" + why + ")");
        return;
    }
    if (!gt::half_fmt(f, hf, why)) {
        check(false, name + ": half_fmt (" + why + ")");
        return;
    }
    gt::SplitLayout L;
    if (!gt::split_layout(f.gu_row, f.d_row, FF, E, ggml_blck_size(dn), ggml_type_size(dn), L, why)) {
        check(false, name + ": split_layout (" + why + ")");
        return;
    }
    check(L.blob == f.bytes && L.h_gu == hf.up_off && L.h_down_off == hf.down_off && hf.d_row == L.hd_row,
          name + ": the half format is the split layout");
    std::vector<std::vector<uint8_t>> blob((size_t) ne, std::vector<uint8_t>(f.bytes));
    std::vector<std::vector<uint8_t>> h0((size_t) ne, std::vector<uint8_t>(L.half_bytes)),
        h1((size_t) ne, std::vector<uint8_t>(L.half_bytes));
    std::vector<const uint8_t*> bp, p0, p1;
    for (int i = 0; i < ne; ++i) {
        fill(gu, blob[(size_t) i].data(), 2 * f.up_off, rng, 0.002f, 0.004f);
        fill(dn, blob[(size_t) i].data() + f.down_off, f.bytes - f.down_off, rng, 0.002f, 0.004f);
        gt::scatter(L, blob[(size_t) i].data(), h0[(size_t) i].data(), h1[(size_t) i].data());
        bp.push_back(blob[(size_t) i].data());
        p0.push_back(h0[(size_t) i].data());
        p1.push_back(h1[(size_t) i].data());
    }
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) E);
    for (auto& v : x) v = nd(rng);
    const float w[ne] = {0.45f, 0.35f, 0.2f};

    std::vector<float> y_ref;
    std::vector<std::vector<float>> ff_ref, dn_ref;
    unsplit_lane(f, ne, bp.data(), w, x.data(), limit, y_ref, ff_ref, dn_ref);

    std::vector<uint8_t> act(kc::kNativeActBytes);
    kc::native_quant_act(f, x.data(), act.data());
    const int H = (int) L.H;
    // gate/up: each half's rows bit-identical to the unified rows [h*H, (h+1)*H); in pieces of odd sizes too
    bool gu_same = true, pieces_same = true, hq_same = true, dn_close = true;
    double dn_worst = 0.0;
    std::vector<float> ff((size_t) H), ffp((size_t) H), dsum((size_t) E), dh((size_t) E);
    std::vector<uint8_t> hq_full(kc::kNativeHBytes), hq_half(kc::kNativeHBytes);
    for (int i = 0; i < ne; ++i) {
        kc::native_quant_h(f, ff_ref[(size_t) i].data(), hq_full.data());
        std::fill(dsum.begin(), dsum.end(), 0.0f);
        for (int h = 0; h < 2; ++h) {
            const uint8_t* hb = (h ? p1 : p0)[(size_t) i];
            gt::half_gu_rows(hf, L, hb, act.data(), ff.data(), 0, H, limit);
            gu_same = gu_same && std::memcmp(ff.data(), ff_ref[(size_t) i].data() + (size_t) h * H, (size_t) H * 4) == 0;
            for (int r0 = 0; r0 < H;) {   // pieces of 1, 3, 5, 7, 13, 37 rows ...
                static const int sz[] = {1, 3, 5, 7, 13, 37, 101};
                const int r1 = std::min(H, r0 + sz[(r0 / 7) % 7]);
                gt::half_gu_rows(hf, L, hb, act.data(), ffp.data(), r0, r1, limit);
                r0 = r1;
            }
            pieces_same = pieces_same && std::memcmp(ff.data(), ffp.data(), (size_t) H * 4) == 0;
            kc::native_quant_h(hf, ff.data(), hq_half.data());
            hq_same = hq_same && std::memcmp(hq_half.data(), hq_full.data() + (size_t) h * hf.h_bytes, hf.h_bytes) == 0;
            gt::half_down_rows(hf, L, hb, hq_half.data(), dh.data(), 0, (int) E);
            for (int r = 0; r < (int) E; ++r) dsum[(size_t) r] += dh[(size_t) r];
        }
        const double e = max_rel(dsum, dn_ref[(size_t) i]);
        dn_worst = std::max(dn_worst, e);
        dn_close = dn_close && e < 1e-5;
    }
    check(gu_same, name + ": gate/up half rows bit-identical to the unified lane");
    check(pieces_same, name + ": half rows in pieces of 1, 3, 5, 7, 13, 37, 101 = the whole range");
    check(hq_same, name + ": a half's down activation = its part of the whole one");
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.2e", dn_worst);
    check(dn_close, name + ": down, two half-column dots added vs one: max rel " + buf + " < 1e-5");

    // the whole call: each socket's partial y, then one add
    std::vector<float> y0((size_t) E), y1((size_t) E), y((size_t) E), sff((size_t) H);
    std::vector<uint8_t> shq(kc::kNativeHBytes);
    std::vector<float> sdn((size_t) E);
    gt::half_experts_serial(hf, L, ne, p0.data(), w, act.data(), limit, y0.data(), sff.data(), shq.data(), sdn.data());
    gt::half_experts_serial(hf, L, ne, p1.data(), w, act.data(), limit, y1.data(), sff.data(), shq.data(), sdn.data());
    gt::add_partial(y0.data(), y1.data(), y.data(), E);
    const double ey = max_rel(y, y_ref);
    std::snprintf(buf, sizeof buf, "%.2e", ey);
    check(ey < 1e-5, name + ": node-split lane (y_0 + y_1) vs the unsplit lane: max rel " + buf + " < 1e-5");
}

// ---------------------------------------------------------------- 2-4. the layout's consumers

// glm_tier_numa.cu's unsplit_words<W> on the host: every word of both halves (half 1 at h0 + D) to its unified place
template<typename T>
void device_walk(const gt::SplitLayout& L, const uint8_t* h0, size_t D, uint8_t* dst, size_t nth) {
    const size_t W = sizeof(T), nw = L.half_bytes / W;
    for (size_t tid = 0; tid < nth; ++tid)
        for (size_t j = tid; j < 2 * nw; j += nth) {
            const int h = j >= nw ? 1 : 0;
            const size_t off = (j - (size_t) h * nw) * W;
            T v;
            std::memcpy(&v, h0 + (size_t) h * D + off, W);
            std::memcpy(dst + gt::unsplit_off(L, h, off), &v, W);
        }
}

struct W16 {
    uint8_t b[16];
};

void consumers_case(ggml_type gu, ggml_type dn, std::mt19937& rng) {
    const int64_t E = 6144, FF = 2048;
    const std::string name = std::string(ggml_type_name(gu)) + " / " + ggml_type_name(dn);
    gt::SplitLayout L;
    std::string why;
    const size_t gu_row = ggml_row_size(gu, E), d_row = ggml_row_size(dn, FF);
    if (!gt::split_layout(gu_row, d_row, FF, E, ggml_blck_size(dn), ggml_type_size(dn), L, why)) {
        check(false, name + ": split_layout (" + why + ")");
        return;
    }
    check(L.half_bytes % (size_t) gt::split_word(L) == 0 && L.h_gu % (size_t) gt::split_word(L) == 0,
          name + ": the halves are whole device words (" + std::to_string(gt::split_word(L)) + " B)");
    std::vector<uint8_t> u(L.blob);
    for (auto& b : u) b = (uint8_t) rng();
    // a class region of 3 slots a half, the expert in slot 1: half 0 at map + stride, half 1 at that + D
    const size_t stride = gt::half_stride({L}), D = 3 * stride;
    std::vector<uint8_t> map(2 * D, 0);
    uint8_t* slot = map.data() + stride;
    // (3) demotion: the six 2-D copies, D2H side
    for (const auto& op : gt::split_ops(L))
        for (size_t r = 0; r < op.rows; ++r)
            std::memcpy(slot + (size_t) op.half * D + op.h_off + r * op.h_pitch, u.data() + op.u_off + r * op.u_pitch,
                        op.width);
    std::vector<uint8_t> r0(L.half_bytes), r1(L.half_bytes);
    gt::scatter(L, u.data(), r0.data(), r1.data());
    check(std::memcmp(slot, r0.data(), L.half_bytes) == 0 && std::memcmp(slot + D, r1.data(), L.half_bytes) == 0,
          name + ": demotion copy ops = scatter (half 1 at + D)");
    // (3) promotion: the copies back
    std::vector<uint8_t> back(L.blob, 0);
    for (const auto& op : gt::split_ops(L))
        for (size_t r = 0; r < op.rows; ++r)
            std::memcpy(back.data() + op.u_off + r * op.u_pitch, slot + (size_t) op.half * D + op.h_off + r * op.h_pitch,
                        op.width);
    check(back == u, name + ": promotion copy ops re-create the unified blob");
    // (1) zero-copy pull: the device's word walk at its word, and at the narrower ones it could fall back to
    for (int w : {16, 8, 4}) {
        if (w > gt::split_word(L)) continue;
        std::vector<uint8_t> v(L.blob, 0);
        if (w == 16) device_walk<W16>(L, slot, D, v.data(), 80 * 256 / 64);
        else if (w == 8) device_walk<uint64_t>(L, slot, D, v.data(), 37);
        else device_walk<uint32_t>(L, slot, D, v.data(), 1);
        check(v == u, name + ": device pull, " + std::to_string(w) + "-byte words: two half reads re-interleaved");
    }
    // (4) disk reads: fast_read_part's chunks (len / n_chunks rounded up to 4 KB) per role, into the split slot
    for (int n_chunks : {1, 3, 8}) {
        std::vector<uint8_t> m2(2 * D, 0);
        uint8_t* s2 = m2.data() + stride;
        for (int role = 0; role < 3; ++role) {
            const size_t len = role == 2 ? (size_t) E * d_row : (size_t) FF * gu_row;
            const size_t base = role == 0 ? 0 : role == 1 ? L.gu_bytes : 2 * L.gu_bytes;
            const size_t piece = ((len + (size_t) n_chunks - 1) / (size_t) n_chunks + 4095) & ~(size_t) 4095;
            for (int c = 0; c < n_chunks; ++c) {
                const size_t c0 = std::min(len, (size_t) c * piece), c1 = std::min(len, c0 + piece);
                if (c1 > c0) gt::scatter_range(L, base + c0, u.data() + base + c0, c1 - c0, s2, s2 + D);
            }
        }
        check(std::memcmp(s2, r0.data(), L.half_bytes) == 0 && std::memcmp(s2 + D, r1.data(), L.half_bytes) == 0,
              name + ": disk read in " + std::to_string(n_chunks) + " chunk(s) a role = scatter");
    }
}

}  // namespace

int main() {
    std::mt19937 rng(20261012);
    if (!kc::native_experts_available()) {
        std::printf("glm_tier_lane_test: no ggml-cpu in this build\n");
        return 77;
    }
    std::printf("B8.4 node-split CPU lane (GLM-5.3 6144 x 2048, 3 experts)\n");
    const float inf = std::numeric_limits<float>::infinity();
    lane_case(GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS, inf, rng);
    lane_case(GGML_TYPE_IQ2_S, GGML_TYPE_Q2_K, inf, rng);
    lane_case(GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_XS, inf, rng);
    lane_case(GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, 0.05f, rng);
    std::printf("B8.4 layout consumers\n");
    consumers_case(GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS, rng);
    consumers_case(GGML_TYPE_IQ2_S, GGML_TYPE_Q2_K, rng);
    consumers_case(GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_XS, rng);
    if (g_fail) {
        std::printf("glm_tier_lane_test: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("glm_tier_lane_test: all passed\n");
    return 0;
}
