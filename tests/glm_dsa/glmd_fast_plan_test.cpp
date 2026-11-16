// tests/glm_dsa/glmd_fast_plan_test.cpp - the host half of the glm-dsa fast binder (glm-dsa B2.4), CPU only:
//   * weight resolution: pick_mv (native -> BF16 copy -> F32), pick_float (BF16 -> F32, never native rows) and
//     pick_router (F32 first, G23) over synthetic tables that mimic ws_map_ / w16_ / w_ for every upcast type;
//   * the glm-dsa arena at GLM-5.3 geometry: q 16384, kv_raw 576, score max_ctx, cells 2048, 256-byte alignment,
//     no overlap, total at 4K / 64K / 128K ctx;
//   * the q8_1 host twins (FP16 round trip, quantize -> dequantize error bound);
//   * shexp_fused_ok: which shared-down types the fused moe_gate_up dot serves (the rest run apart).
#include "glm_dsa_fast_plan.hpp"

#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace strata::core::glmdsa;

static int g_fail = 0;
#define CHECK(c)                                                                \
    do {                                                                        \
        if (!(c)) {                                                             \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);   \
            ++g_fail;                                                           \
        }                                                                       \
    } while (0)

namespace {
struct Native {
    int type;
    const void* q;
    const void* f32;
};
struct Fake {
    std::map<std::string, Native> ws;
    std::map<std::string, const void*> w16, w32;
    Tables tables() const {
        Tables t;
        t.native = [this](const std::string& n, int& type, const void*& q, const void*& f) {
            const auto it = ws.find(n);
            if (it == ws.end()) return false;
            type = it->second.type;
            q = it->second.q;
            f = it->second.f32;
            return true;
        };
        t.bf16 = [this](const std::string& n) -> const void* {
            const auto it = w16.find(n);
            return it == w16.end() ? nullptr : it->second;
        };
        t.f32 = [this](const std::string& n) -> const void* {
            const auto it = w32.find(n);
            return it == w32.end() ? nullptr : it->second;
        };
        // gf::mv_type_ok: F32, BF16, Q4_K 12, Q5_K 13, Q6_K 14, Q8_0 8, IQ2_XXS 16, IQ2_XS 17, IQ3_XXS 18, IQ1_S 19,
        // IQ3_S 21, IQ2_S 22, IQ4_XS 23, Q2_K 10, Q3_K 11, IQ1_M 29
        t.mv_ok = [](int ty) {
            switch (ty) {
                case 0: case 30: case 12: case 13: case 14: case 8: case 16: case 18: case 19: case 23: case 10:
                case 11: case 17: case 22: case 21: case 29: return true;
                default: return false;
            }
        };
        return t;
    }
};
char kA, kB, kC, kD;   // distinct addresses
}  // namespace

static void test_resolution() {
    Fake f;
    const std::string P = "blk.5.";
    // native types mv serves: Native
    f.ws[P + "attn_q_b.weight"] = {14, &kA, nullptr};
    // the B1.6 upcast types (F16 1, Q4_0 2, Q4_1 3, Q5_0 6, Q5_1 7, IQ4_NL 20): iq_pack writes a BF16 copy, ws_map_ has
    // no slot for them (and when one lingers with its native type, the BF16 copy still wins)
    const int upcast[] = {1, 2, 3, 6, 7, 20};
    for (int ut : upcast) {
        const std::string n = P + "upcast_" + std::to_string(ut);
        f.ws[n] = {ut, &kA, nullptr};
        f.w16[n] = &kB;
        const Pick p = pick_mv(f.tables(), n);
        CHECK(p.src == Src::Bf16 && p.type == kTypeBF16 && p.w == &kB);
    }
    {
        const Pick p = pick_mv(f.tables(), P + "attn_q_b.weight");
        CHECK(p.src == Src::Native && p.type == 14 && p.w == &kA);
    }
    // BF16 copy only (attn_output, shexp, dense ffn, eh_proj in a pack that upcast them)
    for (const char* n : {"attn_output.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                          "ffn_down_shexp.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
                          "nextn.eh_proj.weight"}) {
        f.w16[P + n] = &kB;
        const Pick p = pick_mv(f.tables(), P + n);
        CHECK(p.src == Src::Bf16 && p.type == kTypeBF16 && p.w == &kB);
    }
    // a native type without an mv kernel and no copy: Missing, with the reason
    f.ws[P + "attn_kv_a_mqa.weight"] = {1, &kA, nullptr};
    {
        const Pick p = pick_mv(f.tables(), P + "attn_kv_a_mqa.weight");
        CHECK(p.src == Src::Missing && p.w == nullptr && p.why.find("type 1") != std::string::npos);
    }
    // an F32 slot in ws_map_ (type 0, f32 set) and a small tensor only in w_
    f.ws[P + "attn_q_a.weight"] = {0, nullptr, &kC};
    f.w32[P + "small.weight"] = &kD;
    CHECK(pick_mv(f.tables(), P + "attn_q_a.weight").src == Src::F32);
    CHECK(pick_mv(f.tables(), P + "attn_q_a.weight").w == &kC);
    CHECK(pick_mv(f.tables(), P + "small.weight").src == Src::F32 && pick_mv(f.tables(), P + "small.weight").w == &kD);
    CHECK(pick_mv(f.tables(), P + "nothing.weight").src == Src::Missing);

    // pick_float: BF16 first, then F32; native rows alone are refused
    f.w16[P + "attn_k_b.weight"] = &kB;
    f.w32[P + "attn_k_b.weight"] = &kD;
    CHECK(pick_float(f.tables(), P + "attn_k_b.weight").src == Src::Bf16);
    f.w32[P + "indexer.proj.weight"] = &kD;
    CHECK(pick_float(f.tables(), P + "indexer.proj.weight").src == Src::F32);
    f.ws[P + "indexer.attn_k.weight"] = {8, &kA, nullptr};
    {
        const Pick p = pick_float(f.tables(), P + "indexer.attn_k.weight");
        CHECK(p.src == Src::Missing && p.why.find("native") != std::string::npos);
    }

    // pick_router: F32 (w_) first, then BF16; never native
    f.w32[P + "ffn_gate_inp.weight"] = &kD;
    f.w16[P + "ffn_gate_inp.weight"] = &kB;
    CHECK(pick_router(f.tables(), P + "ffn_gate_inp.weight").src == Src::F32);
    CHECK(pick_router(f.tables(), P + "ffn_gate_inp.weight").type == kTypeF32);
    f.w32.erase(P + "ffn_gate_inp.weight");
    CHECK(pick_router(f.tables(), P + "ffn_gate_inp.weight").src == Src::Bf16);
    f.w16.erase(P + "ffn_gate_inp.weight");
    f.ws[P + "ffn_gate_inp.weight"] = {14, &kA, nullptr};
    CHECK(pick_router(f.tables(), P + "ffn_gate_inp.weight").src == Src::Missing);
    CHECK(std::string(src_name(Src::Bf16)) == "bf16");
}

static ArenaDims glm53(int64_t ctx) {
    ArenaDims d;
    d.n_embd = 6144;
    d.n_head = 64;
    d.q_lora = 2048;
    d.kv_lora = 512;
    d.qk_nope = 192;
    d.qk_rope = 64;
    d.v_head = 256;
    d.idx_heads = 32;
    d.idx_key = 128;
    d.idx_top_k = 2048;
    d.max_ctx = ctx;
    d.n_ff_dense = 12288;
    d.n_ff_sh = 2048;
    return d;
}

static void test_arena() {
    for (const int64_t ctx : {4096LL, 32768LL, 65536LL, 131072LL}) {
        ArenaPlan a;
        std::string err;
        CHECK(arena_plan(glm53(ctx), a, err));
        CHECK(a.bytes[kQ] == 16384u * 4);
        CHECK(a.bytes[kKvRaw] == 576u * 4);
        CHECK(a.bytes[kQAbs] == 64u * 576 * 4);
        CHECK(a.bytes[kScore] == (size_t) ctx * 4);
        CHECK(a.bytes[kCells] == 2048u * 4);
        CHECK(a.bytes[kQrQ] == 2048u / 32 * 36);
        CHECK(a.bytes[kIq] == 32u * 128 * 4);
        CHECK(a.bytes[kAttnOut] == 16384u * 4);
        CHECK(a.bytes[kAttnOutQ] == 16384u / 32 * 36);
        CHECK(a.act_floats == 16384);   // n_head * v_head is the widest q8_1-only input at GLM-5.3
        CHECK(a.bytes[kAct] == (size_t) kMaxMvJobs * 16384 * 4);
        size_t end = 0, sum = 0;
        for (int b = 0; b < kNumBufs; ++b) {
            CHECK(a.off[b] % 256 == 0);
            CHECK(a.off[b] >= end);   // in order, no overlap
            end = a.off[b] + a.bytes[b];
            sum += (a.bytes[b] + 255) / 256 * 256;
            CHECK(std::string(buf_name(b)) != "?");
        }
        CHECK(end <= a.total);
        CHECK(sum == a.total);
        if (ctx == 65536) {
            std::printf("glm-dsa arena at 64K ctx: %zu B (%.2f MB)\n", a.total, a.total / 1048576.0);
            CHECK(a.total < 2u * 1048576u);   // the score row dominates: 256 KB
        }
    }
    // a width the q8_1 buffers cannot hold, and a non-positive size
    {
        ArenaDims d = glm53(4096);
        d.q_lora = 2050;
        ArenaPlan a;
        std::string err;
        CHECK(!arena_plan(d, a, err) && !err.empty());
        d = glm53(0);
        err.clear();
        CHECK(!arena_plan(d, a, err) && !err.empty());
    }
    // the Flash-shaped dims also plan (the helper is arch-neutral)
    {
        ArenaDims d = glm53(8192);
        d.n_embd = 4096;
        d.qk_nope = 256;
        d.qk_rope = 0;
        ArenaPlan a;
        std::string err;
        CHECK(arena_plan(d, a, err));
        CHECK(a.bytes[kKvRaw] == 512u * 4);
    }
}

static void test_q8() {
    // FP16 round trip of representable values, rounding to nearest even
    for (float v : {0.0f, 1.0f, -2.5f, 65504.0f, 6.103515625e-05f, 5.9604644775390625e-08f, 0.333251953125f})
        CHECK(half_to_float(float_to_half(v)) == v);
    CHECK(half_to_float(float_to_half(1.0f + 2.0f / 4096.0f)) == 1.0f);                  // half an ulp: to even
    CHECK(half_to_float(float_to_half(1.0f + 6.0f / 4096.0f)) == 1.0f + 2.0f / 1024.0f);   // 1.5 ulp: to even
    CHECK(half_to_float(float_to_half(1.0f + 3.0f / 4096.0f)) == 1.0f + 1.0f / 1024.0f);
    CHECK(std::isinf(half_to_float(float_to_half(1e6f))));
    // quantize -> dequantize: |err| <= d / 2 + FP16 scale rounding per element
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 3.0f);
    const int n = 6144;
    std::vector<float> x((size_t) n), y((size_t) n);
    for (auto& v : x) v = nd(rng);
    for (int k = 0; k < 32; ++k) x[(size_t) (64 + k)] = 0.0f;   // an all-zero block
    std::vector<uint8_t> q(q8_1_bytes(n));
    q8_1_quantize_host(x.data(), n, q.data());
    q8_1_to_f32_host(q.data(), n, y.data());
    double worst = 0.0;
    for (int b = 0; b < n / 32; ++b) {
        float amax = 0.0f;
        for (int k = 0; k < 32; ++k) amax = std::max(amax, std::fabs(x[(size_t) (b * 32 + k)]));
        for (int k = 0; k < 32; ++k) {
            const double e = std::fabs((double) y[(size_t) (b * 32 + k)] - x[(size_t) (b * 32 + k)]);
            const double bound = amax / 127.0 * (0.5 + 127.0 * 1e-3) + 1e-12;
            worst = std::max(worst, e / (bound > 0 ? bound : 1));
            CHECK(e <= bound);
        }
    }
    for (int k = 0; k < 32; ++k) CHECK(y[(size_t) (64 + k)] == 0.0f);
    std::printf("q8_1 host round trip: worst error / bound %.3f\n", worst);
}

// shexp_fused_ok: moe_gate_up's fused shared down (any_row_dot, q8_1 input) serves Q8_0 and the K / IQ quants; the
// float types and B10.7's six types must run apart (a fused Q4_1 down read 0: the shared expert silently dropped)
static void test_shexp_fused() {
    for (int t : {8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 29}) CHECK(shexp_fused_ok(t));
    for (int t : {kTypeF32, kTypeBF16, 1, 2, 3, 6, 7, 20}) CHECK(!shexp_fused_ok(t));
    CHECK(!shexp_fused_ok(-1) && !shexp_fused_ok(9) && !shexp_fused_ok(15) && !shexp_fused_ok(1000));
}

int main() {
    test_shexp_fused();
    test_resolution();
    test_arena();
    test_q8();
    if (g_fail != 0) {
        std::fprintf(stderr, "glmd_fast_plan_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("glmd_fast_plan_test: OK\n");
    return 0;
}
