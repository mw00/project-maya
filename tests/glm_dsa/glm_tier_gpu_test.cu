// tests/glm_dsa/glm_tier_gpu_test.cu - glm-dsa B8.4 on the device (label gpu; exit 77 without a CUDA device).
//
// A node-split RAM class over mapped pinned memory (half 1 at half 0 + D), at three layouts whose device gather word
// is 16, 8 and 4 bytes:
//   1. consumer 1, the zero-copy pulls: glmtier::fetch (moe_fetch's place) and glmtier::prefetch (moe_prefetch's)
//      re-create every planned expert's unified blob in its VRAM slot bit for bit, a source outside the class (the
//      disk staging) is copied as Maya's kernel does, and an entry without a source is left alone;
//   2. consumer 3, promotion and demotion: glmtier::h2d (six cudaMemcpy2DAsync) gives the unified blob, glmtier::d2h
//      writes the halves as scatter; with no tier both are Maya's single cudaMemcpyAsync;
//   3. consumer 4 against the device: GLM-5.3 experts (6144 x 2048, IQ2_XXS / IQ3_XXS) through the node-split CPU lane
//      (glmtier::cpu_experts), Maya's unsplit lane and the device's moe_gate_up / moe_down (fed by the split fetch).
//      The split lane equals the unsplit one within the plan's per-consumer max rel 1e-5.  The lane rounds x and h to
//      Q8_K (256-value blocks), the device to q8_1 (32-value blocks), so the two sides differ by ~2.5e-2 and are not
//      held to each other: each is held to a double reference in its own activation format (its quantized x and h
//      within their rounding of the floats; the lane's dots within 1e-5, the device's within what llama.cpp's
//      truncating integer dots and their f32 sums can move them).
#include "glm_tier_numa.hpp"

#include "strata/core/glm_arch.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

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
namespace gfk = strata::kernels::glmf;
using strata::core::Glm5Geometry;
using strata::core::Glm5Model;
using strata::core::GlmArch;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-100s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

bool ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::printf("  CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        ++g_fail;
        return false;
    }
    return true;
}

// a split class of `slots` slots over one mapped pinned allocation
struct FakeClass {
    uint8_t* map = nullptr;
    size_t stride = 0, D = 0;
    int slots = 0;
    std::shared_ptr<gt::TierState> T;
    explicit FakeClass(const gt::SplitLayout& L, int n) : slots(n) {
        stride = gt::half_stride({L});
        D = (size_t) n * stride;
        if (cudaHostAlloc((void**) &map, 2 * D, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) map = nullptr;
        T = std::make_shared<gt::TierState>();
        T->dsa = true;
        T->split_req = true;
        T->cls.resize(1);
        auto& c = T->cls[0];
        c.ours = true;
        c.split = true;
        c.map = map;
        c.D = D;
        c.stride = stride;
        c.n = n;
        T->L = {L};
        T->half = {kc::NativeFmt{}};
        T->layer_cls = {0};
    }
    ~FakeClass() {
        if (map != nullptr) cudaFreeHost(map);
    }
    uint8_t* slot(int s) const { return map + (size_t) s * stride; }
};

void layout_case(const std::string& name, const gt::SplitLayout& L, std::mt19937& rng) {
    const int K = 8;
    FakeClass C(L, K);
    if (C.map == nullptr) {
        check(false, name + ": cudaHostAlloc");
        return;
    }
    std::vector<std::vector<uint8_t>> u((size_t) K, std::vector<uint8_t>(L.blob));
    for (int i = 0; i < K; ++i) {
        for (auto& b : u[(size_t) i]) b = (uint8_t) rng();
        gt::scatter(L, u[(size_t) i].data(), C.slot(i), C.slot(i) + C.D);
    }
    // a staging buffer outside the class holding a unified blob (Maya's disk staging)
    uint8_t* stage = nullptr;
    cudaHostAlloc((void**) &stage, L.blob + 16, cudaHostAllocMapped | cudaHostAllocPortable);
    std::vector<uint8_t> us(L.blob);
    for (auto& b : us) b = (uint8_t) rng();
    std::memcpy(stage, us.data(), L.blob);
    const size_t vstride = (L.blob + 255) & ~(size_t) 255;
    uint8_t* vram = nullptr;
    ok(cudaMalloc(&vram, (size_t) K * vstride), "cudaMalloc slots");
    cudaMemset(vram, 0xcd, (size_t) K * vstride);

    gfk::MoeDev d{};
    unsigned long long *fsrc = nullptr, *pptr = nullptr, *psrc = nullptr, *pdst = nullptr;
    int* pn = nullptr;
    cudaMalloc(&fsrc, 8 * 8);
    cudaMalloc(&pptr, 8 * 8);
    cudaMalloc(&psrc, 8 * 8);
    cudaMalloc(&pdst, 8 * 8);
    cudaMalloc(&pn, 4);
    d.fetch_src = fsrc;
    d.plan_ptr = pptr;
    d.pf_src = psrc;
    d.pf_dst = pdst;
    d.pf_n = pn;
    // fetch: entries 0..5 from slots 5,1,0,7,2 and the staging; entry 6 nothing to fetch
    const int from[6] = {5, 1, 0, 7, 2, -1};
    unsigned long long hs[8] = {}, hp[8] = {};
    for (int i = 0; i < 7; ++i) {
        hp[i] = (unsigned long long) (vram + (size_t) i * vstride);
        hs[i] = i < 6 ? (unsigned long long) (from[i] >= 0 ? C.slot(from[i]) : stage) : 0ull;
    }
    cudaMemcpy(fsrc, hs, sizeof hs, cudaMemcpyHostToDevice);
    cudaMemcpy(pptr, hp, sizeof hp, cudaMemcpyHostToDevice);
    cudaStream_t s;
    cudaStreamCreate(&s);
    const bool took = gt::fetch(C.T.get(), d, 0, 7, s);
    ok(cudaStreamSynchronize(s), "fetch");
    std::vector<uint8_t> got(L.blob);
    bool fetched = took;
    for (int i = 0; i < 6; ++i) {
        cudaMemcpy(got.data(), vram + (size_t) i * vstride, L.blob, cudaMemcpyDeviceToHost);
        fetched = fetched && got == (from[i] >= 0 ? u[(size_t) from[i]] : us);
    }
    cudaMemcpy(got.data(), vram + 6 * vstride, L.blob, cudaMemcpyDeviceToHost);
    const bool untouched = std::all_of(got.begin(), got.end(), [](uint8_t b) { return b == 0xcd; });
    check(fetched, name + ": fetch: split slots re-interleaved, the staging copied plain");
    check(untouched, name + ": fetch: an entry with no source is left alone");
    // prefetch: 3 entries
    cudaMemset(vram, 0, (size_t) K * vstride);
    const int pf_from[3] = {3, 6, -1};
    unsigned long long ps[8] = {}, pd[8] = {};
    for (int i = 0; i < 3; ++i) {
        ps[i] = (unsigned long long) (pf_from[i] >= 0 ? C.slot(pf_from[i]) : stage);
        pd[i] = (unsigned long long) (vram + (size_t) (i + 2) * vstride);
    }
    const int three = 3;
    cudaMemcpy(psrc, ps, sizeof ps, cudaMemcpyHostToDevice);
    cudaMemcpy(pdst, pd, sizeof pd, cudaMemcpyHostToDevice);
    cudaMemcpy(pn, &three, 4, cudaMemcpyHostToDevice);
    const bool ptook = gt::prefetch(C.T.get(), d, 0, s);
    ok(cudaStreamSynchronize(s), "prefetch");
    bool pref = ptook;
    for (int i = 0; i < 3; ++i) {
        cudaMemcpy(got.data(), vram + (size_t) (i + 2) * vstride, L.blob, cudaMemcpyDeviceToHost);
        pref = pref && got == (pf_from[i] >= 0 ? u[(size_t) pf_from[i]] : us);
    }
    check(pref, name + ": prefetch: split slots re-interleaved, the staging copied plain");
    // h2d / d2h
    cudaMemset(vram, 0, vstride);
    ok(gt::h2d(C.T.get(), 0, vram, C.slot(4), L.blob, s), "h2d");
    cudaStreamSynchronize(s);
    cudaMemcpy(got.data(), vram, L.blob, cudaMemcpyDeviceToHost);
    check(got == u[4], name + ": promotion (h2d, six 2-D copies) = the unified blob");
    cudaMemcpy(vram, us.data(), L.blob, cudaMemcpyHostToDevice);
    std::memset(C.slot(6), 0, L.half_bytes);
    std::memset(C.slot(6) + C.D, 0, L.half_bytes);
    ok(gt::d2h(C.T.get(), 0, C.slot(6), vram, L.blob, s), "d2h");
    cudaStreamSynchronize(s);
    std::vector<uint8_t> r0(L.half_bytes), r1(L.half_bytes);
    gt::scatter(L, us.data(), r0.data(), r1.data());
    check(std::memcmp(C.slot(6), r0.data(), L.half_bytes) == 0 && std::memcmp(C.slot(6) + C.D, r1.data(), L.half_bytes) == 0,
          name + ": demotion (d2h) writes both halves as scatter");
    // no tier: Maya's single copies
    std::memcpy(stage, u[0].data(), L.blob);
    ok(gt::h2d(nullptr, 0, vram, stage, L.blob, s), "h2d plain");
    cudaStreamSynchronize(s);
    cudaMemcpy(got.data(), vram, L.blob, cudaMemcpyDeviceToHost);
    check(got == u[0] && !gt::fetch(nullptr, d, 0, 7, s) && !gt::prefetch(nullptr, d, 0, s),
          name + ": no tier: Maya's copies and kernels");
    cudaStreamDestroy(s);
    cudaFree(vram);
    cudaFree(fsrc);
    cudaFree(pptr);
    cudaFree(psrc);
    cudaFree(pdst);
    cudaFree(pn);
    cudaFreeHost(stage);
}

void fill(ggml_type t, uint8_t* p, size_t bytes, std::mt19937& rng) {
    for (size_t i = 0; i < bytes; ++i) p[i] = (uint8_t) rng();
    const size_t bb = ggml_type_size(t), off = t == GGML_TYPE_Q2_K ? 80 : 0;
    for (size_t b = 0; b + bb <= bytes; b += bb) {
        const ggml_fp16_t h = ggml_fp32_to_fp16(0.002f + 0.002f * (float) (rng() % 100) / 100.0f);
        std::memcpy(p + b + off, &h, 2);
    }
}

template<typename T>
double rel_l2(const std::vector<float>& a, const std::vector<T>& b) {
    double e = 0, n = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double) a[i] - (double) b[i];
        e += d * d, n += (double) b[i] * b[i];
    }
    return std::sqrt(e / std::max(n, 1e-30));
}

// max |a - b| / max |b| (glm_tier_lane_test's measure, the plan's per-consumer criterion); a NaN makes it NaN
double max_rel(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double e = std::fabs((double) a[i] - (double) b[i]);
        if (e > num || std::isnan(e)) num = e;
        den = std::max(den, (double) std::fabs(b[i]));
    }
    return den > 0.0 ? num / den : num;
}

// the largest error / bound so far (a NaN, or an error against a zero bound, is infinitely far out)
void worst(double& w, double e, double bound) {
    const double t = e <= 0.0 ? 0.0 : bound > 0.0 ? e / bound : std::numeric_limits<double>::infinity();
    if (!(t <= w)) w = std::isnan(t) ? std::numeric_limits<double>::infinity() : t;
}

// An activation as one side quantized it (ggml's block_q8_K: f32 d, 256 int8, 16 int16 sums; block_q8_1: f16 d, f16
// sum, 32 int8): each value dequantized, its int8, and its block's scale.
struct QAct {
    std::vector<double> v, d;
    std::vector<int> q;
};
QAct read_q8(ggml_type t, const uint8_t* p, int n) {
    QAct a;
    const int blk = (int) ggml_blck_size(t);
    const size_t bytes = ggml_type_size(t);
    for (int b = 0; b < n / blk; ++b) {
        const uint8_t* bp = p + (size_t) b * bytes;
        double d;
        if (t == GGML_TYPE_Q8_K) {
            float s;
            std::memcpy(&s, bp, sizeof s);
            d = s;
        } else {
            ggml_fp16_t h;
            std::memcpy(&h, bp, sizeof h);
            d = ggml_fp16_to_fp32(h);
        }
        a.d.push_back(d);
        for (int k = 0; k < blk; ++k) {
            const int q = (int) (int8_t) bp[4 + k];
            a.q.push_back(q);
            a.v.push_back(d * q);
        }
    }
    return a;
}

// |q d16 - v| for a value v the device stored as q8_1 (q8_1_store_warp: d = amax / 127 in f32, q = roundf(v / d), d
// kept as fp16): half a step plus the division's rounding (|v / d| <= 127: under 127 x 2^-24 of a step; 1e-4 allowed),
// plus |q| x the fp16 rounding of d (2^-11 relative, so d <= d16 (1 + 2^-10); 2^-25 absolute below fp16's normal range)
double q8_1_err(const QAct& a, int i) {
    const double q = std::fabs((double) a.q[(size_t) i]), d = a.d[(size_t) i / 32];
    return (0.5 + 1e-4 + q / 2048.0) * d * (1.0 + 1.0 / 1024.0) + (q + 1.0) * std::ldexp(1.0, -25);
}

// A dot of ggml's dequantized weight row (exact for IQ2_XXS / IQ3_XXS: fp16 d x a 4-bit sub-scale x a codebook value)
// with an activation, in double, by 32-value groups; S = the sum of |group dot|.  With `ad` (the q8_1 group scales)
// also T = the sum over the groups of d_w x d_act, d_w the fp16 d at the head of each `bsz`-byte block of the row `q`:
// llama.cpp's CUDA i-quant dots truncate one integer division a group (IQ2_XXS sumi * ls / 8: < 1 unit of d_w x d_act;
// IQ3_XXS (ls * sumi + sumi / 2) / 2: < 1.25 units).
struct RefDot {
    double v = 0, S = 0, T = 0;
};
RefDot ref_dot(const float* w, const double* a, int n, const uint8_t* q = nullptr, size_t bsz = 0,
               const double* ad = nullptr) {
    RefDot r;
    for (int g = 0; g < n / 32; ++g) {
        double p = 0;
        for (int k = 32 * g; k < 32 * g + 32; ++k) p += (double) w[k] * a[k];
        r.v += p;
        r.S += std::fabs(p);
        if (ad != nullptr) {
            ggml_fp16_t h;
            std::memcpy(&h, q + (size_t) (g / 8) * bsz, sizeof h);   // 8 groups a 256-value block
            r.T += (double) ggml_fp16_to_fp32(h) * ad[g];
        }
    }
    return r;
}

void lane_vs_device(std::mt19937& rng) {
    const int E = 6144, FF = 2048;
    const ggml_type gu = GGML_TYPE_IQ2_XXS, dn = GGML_TYPE_IQ3_XXS;
    kc::NativeFmt f, hf;
    std::string why;
    gt::SplitLayout L;
    if (!kc::native_fmt(gu, dn, E, FF, f, why) || !gt::half_fmt(f, hf, why) ||
        !gt::split_layout(f.gu_row, f.d_row, FF, E, ggml_blck_size(dn), ggml_type_size(dn), L, why)) {
        check(false, "lane vs device: setup (" + why + ")");
        return;
    }
    if (f.gu_act != GGML_TYPE_Q8_K || f.d_act != GGML_TYPE_Q8_K) {   // read_q8 below decodes the lane's activations
        check(false, "lane vs device: setup (the lane's activations are not Q8_K)");
        return;
    }
    FakeClass C(L, 2);
    std::vector<uint8_t> u(L.blob);
    fill(gu, u.data(), 2 * f.up_off, rng);
    fill(dn, u.data() + f.down_off, f.bytes - f.down_off, rng);
    gt::scatter(L, u.data(), C.slot(1), C.slot(1) + C.D);
    C.T->half = {hf};
    C.T->lane[0] = std::make_shared<gt::NodeLane>(0, std::vector<int>{}, 2);
    C.T->lane[1] = std::make_shared<gt::NodeLane>(1, std::vector<int>{}, 2);
    C.T->act.assign(kc::kNativeActBytes, 0);
    for (int h = 0; h < 2; ++h) {
        C.T->ff[h].assign((size_t) 8 * L.H, 0.0f);
        C.T->hq[h].assign((size_t) 8 * kc::kNativeHBytes, 0);
        C.T->dn[h].assign((size_t) 8 * E, 0.0f);
        C.T->part[h].assign((size_t) E, 0.0f);
    }
    Glm5Model::FastState F;
    F.tier = C.T;
    F.cpu_fmt = {f};
    Glm5Geometry g;
    g.arch = GlmArch::GlmDsa;
    g.n_embd = E;
    g.n_ff_exp = FF;
    g.swiglu_exp = std::numeric_limits<float>::infinity();
    // the device: x = rms(h) and its q8_1, the expert pulled from the split class by the split fetch, gate/up, down
    cudaStream_t s;
    cudaStreamCreate(&s);
    float *dh, *dw, *dx, *dout, *plan_w, *cpu_part;
    void *dxq, *dhq;
    uint8_t* slot;
    unsigned long long *pptr, *fsrc;
    int* cpu_flag;
    cudaMalloc(&dh, E * 4);
    cudaMalloc(&dw, E * 4);
    cudaMalloc(&dx, E * 4);
    cudaMalloc(&dout, E * 4);
    cudaMalloc(&dxq, (size_t) E / 32 * 36);
    cudaMalloc(&dhq, (size_t) 8 * FF / 32 * 36);
    cudaMalloc(&slot, L.blob);
    cudaMalloc(&pptr, 64);
    cudaMalloc(&fsrc, 64);
    cudaMalloc(&plan_w, 32);
    cudaMalloc(&cpu_part, E * 4);
    cudaMalloc(&cpu_flag, 4);
    std::vector<float> h((size_t) E), ones((size_t) E, 1.0f), xc((size_t) E), og((size_t) E), oc((size_t) E), ou((size_t) E);
    for (int i = 0; i < E; ++i) h[(size_t) i] = std::sin(0.71f * i) + ((i % 97) == 0 ? 6.0f : 0.0f);
    cudaMemcpy(dh, h.data(), E * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, ones.data(), E * 4, cudaMemcpyHostToDevice);
    const unsigned long long sp[8] = {(unsigned long long) slot}, fs[8] = {(unsigned long long) C.slot(1)};
    const float one = 1.0f;
    cudaMemcpy(pptr, sp, 64, cudaMemcpyHostToDevice);
    cudaMemcpy(fsrc, fs, 64, cudaMemcpyHostToDevice);
    cudaMemcpy(plan_w, &one, 4, cudaMemcpyHostToDevice);
    cudaMemset(cpu_flag, 0, 4);
    gfk::MoeDev md{};
    md.plan_ptr = pptr;
    md.fetch_src = fsrc;
    md.plan_w = plan_w;
    md.cpu_part = cpu_part;
    md.cpu_flag = cpu_flag;
    gfk::rms_q8(dh, nullptr, dw, 1e-5f, E, dx, dxq, s);
    gt::fetch(C.T.get(), md, 0, 1, s);
    gfk::moe_gate_up(gu, md, 1, E, FF, g.swiglu_exp, dxq, dhq, nullptr, 0, nullptr, 0, nullptr, s);
    gfk::moe_down(dn, md, 1, E, FF, L.down_off, dhq, nullptr, dout, s);
    cudaMemcpyAsync(xc.data(), dx, E * 4, cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(og.data(), dout, E * 4, cudaMemcpyDeviceToHost, s);
    ok(cudaStreamSynchronize(s), "device expert");
    const uint8_t* half0 = C.slot(1);
    const bool ran = gt::cpu_experts(F, g, 0, 1, &half0, &one, xc.data(), oc.data());
    // Maya's unsplit lane on the unified blob
    std::vector<uint8_t> act(kc::kNativeActBytes), hq(kc::kNativeHBytes);
    std::vector<float> ff((size_t) FF);
    {
        kc::native_quant_act(f, xc.data(), act.data());
        const void* a[1] = {act.data()};
        float* o[1] = {ff.data()};
        kc::native_gu_rows_split(f, u.data(), u.data() + f.up_off, a, 1, o, 0, FF, g.swiglu_exp);
        kc::native_quant_h(f, ff.data(), hq.data());
        const void* hh[1] = {hq.data()};
        float* d[1] = {ou.data()};
        kc::native_down_rows_split(f, u.data() + f.down_off, hh, 1, d, 0, E);
    }
    // the device's q8_1 x and h (expert 0's)
    std::vector<uint8_t> xq1((size_t) E / 32 * 36), hq1((size_t) FF / 32 * 36);
    cudaMemcpy(xq1.data(), dxq, xq1.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(hq1.data(), dhq, hq1.size(), cudaMemcpyDeviceToHost);
    const QAct xK = read_q8(GGML_TYPE_Q8_K, act.data(), E), hK = read_q8(GGML_TYPE_Q8_K, hq.data(), FF);
    const QAct x1 = read_q8(GGML_TYPE_Q8_1, xq1.data(), E), h1 = read_q8(GGML_TYPE_Q8_1, hq1.data(), FF);
    // the references: ggml's dequantized rows of the unified blob dotted in double with the lane's Q8_K activations,
    // with the device's q8_1 ones (and what its dots may differ by), and with the unquantized ones
    const ggml_type_traits* tg = ggml_get_type_traits(gu);
    const ggml_type_traits* td = ggml_get_type_traits(dn);
    const double rnd = std::ldexp(1.0, -24);   // f32's unit roundoff
    const auto silu = [](double v) { return v / (1.0 + std::exp(-v)); };
    std::vector<float> row((size_t) E);
    std::vector<double> xf(xc.begin(), xc.end()), hrK((size_t) FF), hr1((size_t) FF), hrF((size_t) FF);
    std::vector<double> hslack((size_t) FF);
    for (int r = 0; r < FF; ++r) {
        RefDot gr[3], ur[3];   // the gate and up row dotted with the lane's x, the device's x, x
        for (int m = 0; m < 2; ++m) {
            const uint8_t* q = u.data() + (m ? f.up_off : 0) + (size_t) r * f.gu_row;
            RefDot* t = m ? ur : gr;
            tg->to_float(q, row.data(), E);
            t[0] = ref_dot(row.data(), xK.v.data(), E);
            t[1] = ref_dot(row.data(), x1.v.data(), E, q, ggml_type_size(gu), x1.d.data());
            t[2] = ref_dot(row.data(), xf.data(), E);
        }
        hrK[(size_t) r] = silu(gr[0].v) * ur[0].v;
        hr1[(size_t) r] = silu(gr[1].v) * ur[1].v;
        hrF[(size_t) r] = silu(gr[2].v) * ur[2].v;
        // the device's g and u may differ from these by their truncations (< 1 unit a group) and the f32 roundings of
        // E / 32 group terms; that reaches h through SwiGLU (silu is 1.1-Lipschitz), and h's own f32 math adds ~5 ulp
        const double eg = gr[1].T + (E / 32 + 2) * rnd * (gr[1].S + gr[1].T);
        const double eu = ur[1].T + (E / 32 + 2) * rnd * (ur[1].S + ur[1].T);
        hslack[(size_t) r] = 1.1 * eg * (std::fabs(ur[1].v) + eu) + std::fabs(silu(gr[1].v)) * eu +
                             1e-6 * std::fabs(hr1[(size_t) r]);
    }
    std::vector<double> orK((size_t) E), or1((size_t) E), orF((size_t) E), obound((size_t) E);
    for (int r = 0; r < E; ++r) {
        const uint8_t* q = u.data() + f.down_off + (size_t) r * f.d_row;
        td->to_float(q, row.data(), FF);
        orK[(size_t) r] = ref_dot(row.data(), hK.v.data(), FF).v;
        const RefDot o1 = ref_dot(row.data(), h1.v.data(), FF, q, ggml_type_size(dn), h1.d.data());
        or1[(size_t) r] = o1.v;
        // < 1.25 units a group (IQ3_XXS's truncation) and the f32 roundings of FF / 32 group terms
        obound[(size_t) r] = 1.25 * o1.T + (FF / 32 + 2) * rnd * (o1.S + 1.25 * o1.T);
        orF[(size_t) r] = ref_dot(row.data(), hrF.data(), FF).v;
    }
    char buf[128];
    // B8.4 consumer 4: the node-split lane against Maya's unsplit lane on the same expert (the same Q8_K x; gate/up
    // rows bit-identical; h quantized per half on Q8_K block boundaries, so the same bytes; down two half-column dots
    // added, which only reassociates f32 sums): the plan's per-consumer criterion, as in glm_tier_lane_test
    const double es = max_rel(oc, ou);
    std::snprintf(buf, sizeof buf, "max rel %.2e < 1e-5", es);
    check(ran && es < 1e-5, std::string("lane: node-split vs Maya's unsplit lane, ") + buf);
    // The lane (ggml-cpu: exact int32 block sums) against its Q8_K reference: x and h within half a step of their
    // floats (|d| / 2 with d = max / -127; the f32 roundings of iscale * x and 1 / iscale make it under 1 + 3.1e-5),
    // gate/up and down within 1e-5 (only the f32 sums of 24 and 8 block terms and the f32 SwiGLU differ, ~sqrt(n) ulp)
    double wk = 0;
    for (int i = 0; i < E; ++i)
        worst(wk, std::fabs(xK.v[(size_t) i] - xc[(size_t) i]), 0.5 * std::fabs(xK.d[(size_t) i / 256]));
    for (int i = 0; i < FF; ++i)
        worst(wk, std::fabs(hK.v[(size_t) i] - ff[(size_t) i]), 0.5 * std::fabs(hK.d[(size_t) i / 256]));
    std::snprintf(buf, sizeof buf, "worst %.5f half steps <= 1.0001", wk);
    check(wk <= 1.0001, std::string("lane: Q8_K x and h vs their floats, ") + buf);
    const double lg = rel_l2(ff, hrK), ld = rel_l2(ou, orK);
    std::snprintf(buf, sizeof buf, "gate/up %.2e, down %.2e < 1e-5", lg, ld);
    check(lg < 1e-5 && ld < 1e-5, std::string("lane vs its Q8_K double reference: relative L2 ") + buf);
    // The device (llama.cpp's integer dots) against its q8_1 reference, value by value: x within its q8_1 rounding of
    // x; h within its rounding of the reference h plus what the gate/up dots can move h; down within its dots'
    // truncations and f32 roundings
    double wx = 0, wh = 0, wd = 0;
    for (int i = 0; i < E; ++i) worst(wx, std::fabs(x1.v[(size_t) i] - xc[(size_t) i]), q8_1_err(x1, i));
    for (int j = 0; j < FF; ++j)
        worst(wh, std::fabs(h1.v[(size_t) j] - hr1[(size_t) j]), q8_1_err(h1, j) + hslack[(size_t) j]);
    for (int r = 0; r < E; ++r) worst(wd, std::fabs((double) og[(size_t) r] - or1[(size_t) r]), obound[(size_t) r]);
    std::snprintf(buf, sizeof buf, "worst x %.3f, h %.3f of the bound", wx, wh);
    check(wx <= 1.0 && wh <= 1.0, std::string("device: q8_1 x and h vs x and the q8_1 double reference, ") + buf);
    std::snprintf(buf, sizeof buf, "worst %.3f of the bound", wd);
    check(wd <= 1.0, std::string("device: down vs its q8_1 double reference, ") + buf);
    // not a criterion: the two sides round the activations differently (256-value Q8_K blocks with an f32 scale vs
    // 32-value q8_1 blocks), so neither is the other's reference
    std::printf("  lane vs device %.2e; vs the unquantized double reference: lane %.2e, device %.2e\n", rel_l2(ou, og),
                rel_l2(ou, orF), rel_l2(og, orF));
    cudaFree(dh);
    cudaFree(dw);
    cudaFree(dx);
    cudaFree(dout);
    cudaFree(dxq);
    cudaFree(dhq);
    cudaFree(slot);
    cudaFree(pptr);
    cudaFree(fsrc);
    cudaFree(plan_w);
    cudaFree(cpu_part);
    cudaFree(cpu_flag);
    cudaStreamDestroy(s);
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("glm_tier_gpu_test: no CUDA device, skipped\n");
        return 77;
    }
    cudaSetDevice(0);
    std::mt19937 rng(8402);
    std::printf("B8.4 split class on the device\n");
    std::string why;
    gt::SplitLayout a, b, c;
    // GLM-5.3 IQ2_S / Q2_K (16-byte words), IQ2_XXS / IQ3_XXS (8), and a small layout with 4-byte words
    gt::split_layout(6144 / 256 * 82, 2048 / 256 * 84, 2048, 6144, 256, 84, a, why);
    gt::split_layout(6144 / 256 * 66, 2048 / 256 * 98, 2048, 6144, 256, 98, b, why);
    gt::split_layout(20, 16, 512, 33, 256, 8, c, why);
    // a named array, not a braced list: nvcc 12.4's front end cannot deduce the range of a braced list here
    const std::pair<std::string, gt::SplitLayout> cases[] = {
        {"IQ2_S / Q2_K", a}, {"IQ2_XXS / IQ3_XXS", b}, {"small, 4-byte words", c}};
    for (const auto& [name, L] : cases) {
        check(L.blob > 0, name + ": layout (word " + std::to_string(L.blob ? gt::split_word(L) : 0) + ")");
        if (L.blob > 0) layout_case(name, L, rng);
    }
    std::printf("B8.4 node-split CPU lane vs the device's expert kernels\n");
    lane_vs_device(rng);
    if (g_fail) {
        std::printf("glm_tier_gpu_test: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("glm_tier_gpu_test: all passed\n");
    return 0;
}
