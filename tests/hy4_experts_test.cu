// tests/hy4_experts_test.cu - maya-hy4's routed experts on ds4_iq's kernels (no model): random experts in the types
// Hy4-flash's GGUF and its MTP block take (Q4_K gate / up with Q6_K or Q4_K down - the Q4_K_M quant's; Q8_0 for all
// three - the BF16 MTP experts as maya-hy4 rounds them), run through the token form (a plan), the window form (grouped
// entries, two input rows a token) and the prompt form (the sets), each against the same expert computed matrix by
// matrix with the GEMVs (iq::mv) and the clamped SwiGLU on the host - and the three forms against each other; then the
// CPU lane (src/hy4/hy4_lane: the experts that are not in VRAM, from host memory) against the same reference.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../src/ds4/ds4_iq.cuh"
#include "../src/hy4/hy4_kernels.cuh"
#include "../src/hy4/hy4_lane.hpp"

namespace iq = strata::ds4::gguf::iq;

namespace {

int fails = 0;
#define CHECK(c)                                                                  \
    do {                                                                          \
        if (!(c)) {                                                               \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);     \
            ++fails;                                                              \
        }                                                                         \
    } while (0)
#define CU(x) CHECK((x) == cudaSuccess)

uint16_t f2h(float f) {
    __half h = __float2half_rn(f);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}

// ggml's quantize_row_q8_0_ref
void quant_q8_0(const float* x, int k, uint8_t* out) {
    for (int b = 0; b < k / 32; ++b) {
        float am = 0;
        for (int i = 0; i < 32; ++i) am = std::max(am, std::fabs(x[b * 32 + i]));
        const float d = am / 127.0f, id = d ? 1.0f / d : 0.0f;
        const uint16_t dh = f2h(d);
        std::memcpy(out + b * 34, &dh, 2);
        for (int i = 0; i < 32; ++i) out[b * 34 + 2 + i] = (uint8_t) (int8_t) std::lround(x[b * 32 + i] * id);
    }
}
// a Q4_K / Q6_K row: random bytes with small scales (any bit pattern is a valid block; the kernels decode them as
// ggml does - the comparison is between two of Maya's decodes, both checked against ggml elsewhere)
void random_k(int type, int k, uint8_t* out, std::mt19937& rng) {
    const int bb = iq::block_bytes(type);
    std::uniform_int_distribution<int> byte(0, 255);
    for (int b = 0; b < k / 256; ++b) {
        uint8_t* p = out + b * bb;
        for (int i = 0; i < bb; ++i) p[i] = (uint8_t) byte(rng);
        const uint16_t d = f2h(0.002f), dm = f2h(0.001f);
        if (type == iq::kQ4_K || type == iq::kQ5_K) {
            std::memcpy(p, &d, 2);
            std::memcpy(p + 2, &dm, 2);
        } else if (type == iq::kQ6_K) {   // d at the end
            std::memcpy(p + bb - 2, &d, 2);
        } else {   // (the IQ types: the block's scale first, every other byte a valid code)
            std::memcpy(p, &d, 2);
        }
    }
}

struct Expert {
    std::vector<uint8_t> blob;   // gate | up | down, 256-aligned
    size_t tr[3];
};

float silu(float x) { return x / (1.0f + std::exp(-x)); }

// (lane false: the GPU forms alone - the draws added for the forms' bits, the CPU lane's tolerance is not their test)
void run_case(int tg, int td, int n_exp, int T, std::mt19937& rng, int F = 1536, bool lane = true) {
    const int E = 1536, nu = 4;
    const float limit = 10.0f;
    const size_t gb = iq::row_bytes(tg, E) * F, db = iq::row_bytes(td, F) * E;
    auto al = [](size_t b) { return (b + 255) / 256 * 256; };
    const size_t tr[3] = {0, al(gb), 2 * al(gb)}, blob = al(2 * al(gb) + db);
    std::vector<uint8_t> host(blob * n_exp);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    for (int e = 0; e < n_exp; ++e) {
        uint8_t* b = host.data() + (size_t) e * blob;
        for (int role = 0; role < 3; ++role) {
            const int type = role < 2 ? tg : td, k = role < 2 ? E : F, n = role < 2 ? F : E;
            for (int r = 0; r < n; ++r) {
                uint8_t* row = b + tr[role] + (size_t) r * iq::row_bytes(type, k);
                if (type == iq::kQ8_0) {
                    std::vector<float> w(k);
                    for (float& v : w) v = nd(rng);
                    quant_q8_0(w.data(), k, row);
                } else {
                    random_k(type, k, row, rng);
                }
            }
        }
    }
    uint8_t* dw;
    CU(cudaMalloc(&dw, host.size()));
    CU(cudaMemcpy(dw, host.data(), host.size(), cudaMemcpyHostToDevice));
    // the tokens' inputs and picks
    std::vector<float> x((size_t) T * E);
    std::normal_distribution<float> nx(0.0f, 1.0f);
    for (float& v : x) v = nx(rng);
    std::vector<int> ids((size_t) T * nu);
    std::vector<float> w((size_t) T * nu);
    std::uniform_int_distribution<int> pick(0, n_exp - 1);
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < nu; ++j) {
            int e;
            bool dup;
            do {
                e = pick(rng);
                dup = false;
                for (int q = 0; q < j; ++q) dup |= ids[(size_t) t * nu + q] == e;
            } while (dup);
            ids[(size_t) t * nu + j] = e;
            w[(size_t) t * nu + j] = 0.25f + 0.1f * j;
        }
    cudaStream_t s;
    CU(cudaStreamCreate(&s));
    float *dx, *dh, *dpart, *dy, *dw_, *dout, *dref;
    int8_t *dq, *dq2, *dhq;
    float *dd, *dd2, *dhd;
    int* dids;
    CU(cudaMalloc(&dx, sizeof(float) * T * E));
    CU(cudaMalloc(&dh, sizeof(float) * 16 * T * F));
    CU(cudaMalloc(&dpart, sizeof(float) * 16 * T * E));
    CU(cudaMalloc(&dy, sizeof(float) * 16 * T * E));
    CU(cudaMalloc(&dw_, sizeof(float) * T * nu));
    CU(cudaMalloc(&dout, sizeof(float) * T * E));
    CU(cudaMalloc(&dref, sizeof(float) * T * E));
    CU(cudaMalloc(&dq, T * E));
    CU(cudaMalloc(&dq2, 2 * T * E));
    CU(cudaMalloc(&dhq, 16 * T * F));
    CU(cudaMalloc(&dd, sizeof(float) * T * E / 32));
    CU(cudaMalloc(&dd2, sizeof(float) * 2 * T * E / 32));
    CU(cudaMalloc(&dhd, sizeof(float) * 16 * T * F / 32));
    CU(cudaMalloc(&dids, sizeof(int) * T * nu));
    CU(cudaMemcpy(dx, x.data(), sizeof(float) * T * E, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dids, ids.data(), sizeof(int) * T * nu, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dw_, w.data(), sizeof(float) * T * nu, cudaMemcpyHostToDevice));
    hy4::k::q8_rows(dx, E, T, E, dq, dd, s);

    // the reference: every pick's matrices one GEMV at a time, the SwiGLU on the host
    std::vector<float> ref((size_t) T * E, 0.0f);
    {
        float *g1, *u1, *y1;
        int8_t* hq1;
        float* hd1;
        CU(cudaMalloc(&g1, sizeof(float) * F));
        CU(cudaMalloc(&u1, sizeof(float) * F));
        CU(cudaMalloc(&y1, sizeof(float) * E));
        CU(cudaMalloc(&hq1, F));
        CU(cudaMalloc(&hd1, sizeof(float) * F / 32));
        std::vector<float> hg(F), hu(F), hh(F), hy(E);
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < nu; ++j) {
                const uint8_t* b = dw + (size_t) ids[(size_t) t * nu + j] * blob;
                iq::MvJob jg, ju;
                jg.w = b + tr[0];
                ju.w = b + tr[1];
                jg.ld = ju.ld = iq::row_bytes(tg, E);
                jg.type = ju.type = tg;
                jg.k = ju.k = E;
                jg.n = ju.n = F;
                jg.xq = ju.xq = dq + (size_t) t * E;
                jg.xd = ju.xd = dd + (size_t) t * E / 32;
                jg.y = g1;
                ju.y = u1;
                iq::MvJob jj[2] = {jg, ju};
                CHECK(iq::mv(jj, 2, s));
                CU(cudaMemcpyAsync(hg.data(), g1, sizeof(float) * F, cudaMemcpyDeviceToHost, s));
                CU(cudaMemcpyAsync(hu.data(), u1, sizeof(float) * F, cudaMemcpyDeviceToHost, s));
                CU(cudaStreamSynchronize(s));
                for (int i = 0; i < F; ++i) {
                    const float gg = std::min(hg[i], limit), uu = std::max(-limit, std::min(hu[i], limit));
                    hh[i] = silu(gg) * uu;
                }
                CU(cudaMemcpy(g1, hh.data(), sizeof(float) * F, cudaMemcpyHostToDevice));
                hy4::k::q8_rows(g1, F, 1, F, hq1, hd1, s);
                iq::MvJob jd;
                jd.w = b + tr[2];
                jd.ld = iq::row_bytes(td, F);
                jd.type = td;
                jd.k = F;
                jd.n = E;
                jd.xq = hq1;
                jd.xd = hd1;
                jd.y = y1;
                CHECK(iq::mv(&jd, 1, s));
                CU(cudaMemcpyAsync(hy.data(), y1, sizeof(float) * E, cudaMemcpyDeviceToHost, s));
                CU(cudaStreamSynchronize(s));
                for (int i = 0; i < E; ++i) ref[(size_t) t * E + i] += w[(size_t) t * nu + j] * hy[i];
            }
        cudaFree(g1);
        cudaFree(u1);
        cudaFree(y1);
        cudaFree(hq1);
        cudaFree(hd1);
    }
    auto rel = [&](const std::vector<float>& a) {
        double d2 = 0, r2 = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            d2 += (double) (a[i] - ref[i]) * (a[i] - ref[i]);
            r2 += (double) ref[i] * ref[i];
        }
        return std::sqrt(d2 / std::max(r2, 1e-30));
    };
    iq::Args a;
    a.tr[0] = tr[0];
    a.tr[1] = tr[1];
    a.tr[2] = tr[2];
    a.n_embd = E;
    a.n_ff = F;
    a.limit = limit;
    a.h = dh;
    a.part = dpart;
    a.tg = tg;
    a.td = td;
    a.act = iq::kActSwiglu;
    std::vector<float> out((size_t) T * E);
    // the token / window forms (up to 16 tokens)
    if (T <= 16) {
        if (T == 1) {
            unsigned long long* plan;
            CU(cudaMalloc(&plan, sizeof(unsigned long long) * 16));
            hy4::k::plan_experts(dids, nu, dw, blob, plan, s);
            a.plan_ptr = plan;
            a.xq = dq;
            a.xd = dd;
            CHECK(iq::gate_up(a, nu, 1, s));
            hy4::k::q8_rows(dh, F, nu, F, dhq, dhd, s);
            a.hq = dhq;
            a.hd = dhd;
            CHECK(iq::down(a, nu, 1, s));
            hy4::k::moe_sum(dpart, 1, nu, E, dw_, dout, s);
            cudaFree(plan);
        } else {
            unsigned long long* gp;
            int *gs, *ge, *ng;
            CU(cudaMalloc(&gp, sizeof(unsigned long long) * (16 * T + 1)));
            CU(cudaMalloc(&gs, sizeof(int) * (16 * T + 2)));
            CU(cudaMalloc(&ge, sizeof(int) * 16 * T));
            CU(cudaMalloc(&ng, sizeof(int)));
            hy4::k::dup_rows_q8(dq, dd, T, E, dq2, dd2, s);
            hy4::k::group_experts(dids, T, nu, n_exp, dw, blob, gp, gs, ge, ng, s);
            a.grp_ptr = gp;
            a.grp_start = gs;
            a.grp_ent = ge;
            a.n_groups = ng;
            a.xq = dq2;
            a.xd = dd2;
            const int groups = std::min(nu * T, n_exp);
            CHECK(iq::gate_up(a, groups, T, s));
            hy4::k::q8_rows(dh, F, 16 * T, F, dhq, dhd, s);
            a.hq = dhq;
            a.hd = dhd;
            CHECK(iq::down(a, groups, T, s));
            hy4::k::moe_sum(dpart, T, nu, E, dw_, dout, s);
            cudaFree(gp);
            cudaFree(gs);
            cudaFree(ge);
            cudaFree(ng);
            // every token alone through the token form: the same bits as its row of the window's (drafting relies on
            // it: a verify window's rows must be the rows the tokens get one at a time)
            {
                std::vector<float> win((size_t) T * E), one((size_t) E);
                CU(cudaMemcpy(win.data(), dout, sizeof(float) * T * E, cudaMemcpyDeviceToHost));
                unsigned long long* plan;
                float* d1;
                CU(cudaMalloc(&plan, sizeof(unsigned long long) * 16));
                CU(cudaMalloc(&d1, sizeof(float) * E));
                int diff = 0;
                double worst = 0;
                for (int t = 0; t < T; ++t) {
                    iq::Args b = a;
                    b.grp_ptr = nullptr;
                    b.grp_start = nullptr;
                    b.grp_ent = nullptr;
                    b.n_groups = nullptr;
                    hy4::k::plan_experts(dids + (size_t) t * nu, nu, dw, blob, plan, s);
                    b.plan_ptr = plan;
                    b.xq = dq + (size_t) t * E;
                    b.xd = dd + (size_t) t * E / 32;
                    CHECK(iq::gate_up(b, nu, 1, s));
                    hy4::k::q8_rows(dh, F, nu, F, dhq, dhd, s);
                    b.hq = dhq;
                    b.hd = dhd;
                    CHECK(iq::down(b, nu, 1, s));
                    hy4::k::moe_sum(dpart, 1, nu, E, dw_ + (size_t) t * nu, d1, s);
                    CU(cudaMemcpy(one.data(), d1, sizeof(float) * E, cudaMemcpyDeviceToHost));
                    for (int i = 0; i < E; ++i)
                        if (std::memcmp(&win[(size_t) t * E + i], &one[(size_t) i], 4) != 0) {
                            ++diff;
                            worst = std::max(worst, (double) std::fabs(win[(size_t) t * E + i] - one[(size_t) i]) /
                                                        std::max(1e-30, (double) std::fabs(one[(size_t) i])));
                        }
                }
                std::printf("  token form vs window form, types %s/%s, %d tokens: %d of %d values differ (worst rel %.1e)\n",
                            iq::name(tg), iq::name(td), T, diff, T * E, worst);
                CHECK(diff == 0);
                cudaFree(plan);
                cudaFree(d1);
            }
        }
        CU(cudaMemcpy(out.data(), dout, sizeof(float) * T * E, cudaMemcpyDeviceToHost));
        const double r = rel(out);
        uint32_t hsh = 2166136261u;   // (the bits' FNV-1a: two builds' sums side by side)
        for (size_t i = 0; i < out.size() * 4; ++i) hsh = (hsh ^ ((const uint8_t*) out.data())[i]) * 16777619u;
        std::printf("  token/window form, types %s/%s, %d tokens: rel %.2e bits %08x\n", iq::name(tg), iq::name(td), T, r,
                    hsh);
        CHECK(r < 2e-3);
    }
    // the prompt form: the sets over every expert's sorted rows
    {
        int *bounds, *tok, *row_of;
        uint16_t* h16;
        CU(cudaMalloc(&bounds, sizeof(int) * (n_exp + 1)));
        CU(cudaMalloc(&tok, sizeof(int) * T * nu));
        CU(cudaMalloc(&row_of, sizeof(int) * T * nu));
        CU(cudaMalloc(&h16, sizeof(uint16_t) * T * nu * F));
        hy4::k::moe_sort(dids, T, nu, n_exp, bounds, tok, row_of, s);
        std::vector<int> hb((size_t) n_exp + 1);
        CU(cudaMemcpyAsync(hb.data(), bounds, sizeof(int) * (n_exp + 1), cudaMemcpyDeviceToHost, s));
        CU(cudaStreamSynchronize(s));
        int mr = 0;
        for (int e = 0; e < n_exp; ++e) mr = std::max(mr, hb[(size_t) e + 1] - hb[(size_t) e]);
        iq::SetArgs sa;
        sa.wbase = dw;
        sa.stride = blob;
        sa.tr[0] = tr[0];
        sa.tr[1] = tr[1];
        sa.tr[2] = tr[2];
        sa.bounds = bounds;
        sa.max_rows = mr;
        sa.n_exp = n_exp;
        sa.n_embd = E;
        sa.n_ff = F;
        sa.limit = limit;
        sa.xf = dx;
        sa.idx = tok;
        sa.ldx = E;
        sa.h = h16;
        sa.ldo = F;
        sa.tg = tg;
        sa.td = td;
        CHECK(iq::set_gate_up(sa, s));
        iq::SetArgs sd = sa;
        sd.xf = nullptr;
        sd.idx = nullptr;
        sd.xh = h16;
        sd.ldx = F;
        sd.h = nullptr;
        sd.y = dy;
        sd.ldo = E;
        CHECK(iq::set_down(sd, s));
        hy4::k::moe_combine(dy, row_of, dw_, T, nu, E, dout, s);
        CU(cudaMemcpy(out.data(), dout, sizeof(float) * T * E, cudaMemcpyDeviceToHost));
        const double r = rel(out);
        std::printf("  prompt form, types %s/%s, %d tokens (%d rows an expert at most): rel %.2e\n", iq::name(tg),
                    iq::name(td), T, mr, r);
        // (F32 inputs and F16 h on the tiles: not the token form's int8 sums; random IQ4_XS blocks come out at 1.15e-2
        // on an RTX 3050 (sm_86), under 1e-2 on a V100 - IQ4_XS / Q5_K get the room, the other types stay at 1e-2)
        CHECK(r < (tg == iq::kIQ4_XS || tg == iq::kQ5_K ? 2e-2 : 1e-2));
        cudaFree(bounds);
        cudaFree(tok);
        cudaFree(row_of);
        cudaFree(h16);
    }
    // the CPU lane: the same experts from host memory (each role's rows back to back, as a GGUF stacks them)
    if (lane) {
        std::vector<uint8_t> hg((size_t) n_exp * gb), hu((size_t) n_exp * gb), hd((size_t) n_exp * db);
        for (int e = 0; e < n_exp; ++e) {
            const uint8_t* b = host.data() + (size_t) e * blob;
            std::memcpy(hg.data() + (size_t) e * gb, b + tr[0], gb);
            std::memcpy(hu.data() + (size_t) e * gb, b + tr[1], gb);
            std::memcpy(hd.data() + (size_t) e * db, b + tr[2], db);
        }
        hy4::CpuExperts cx;
        std::string err;
        CHECK(cx.init(tg, td, E, F, err));
        cx.gate = hg.data();
        cx.up = hu.data();
        cx.down = hd.data();
        cx.limit = limit;
        std::vector<hy4::CpuEntry> ent;
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < nu; ++j) ent.push_back({t, ids[(size_t) t * nu + j], w[(size_t) t * nu + j]});
        static hy4::Lane lane(7);
        hy4::cpu_experts(lane, cx, ent, x.data(), T, out.data());
        const double r = rel(out);
        std::printf("  CPU lane, types %s/%s, %d tokens: rel %.2e\n", iq::name(tg), iq::name(td), T, r);
        // ggml-cpu's activations for the K-quants are Q8_K - a scale a 256 values, the device's a 32 - rounded twice (x,
        // then h): ~1.5% on these Gaussian rows; Q8_0's are Q8_0 (a scale a 32, as the device's)
        // (the lane's own rounding of the activations: random IQ4_XS / Q5_K blocks come out at 3.1e-2 .. 3.4e-2, the
        // other types under 2.5e-2)
        CHECK(r < (tg == iq::kQ8_0 ? 5e-3 : tg == iq::kIQ4_XS || tg == iq::kQ5_K ? 4e-2 : 3e-2));
    }
    CHECK(hy4::k::errors() == 0);
    cudaFree(dw);
    cudaFree(dx);
    cudaFree(dh);
    cudaFree(dpart);
    cudaFree(dy);
    cudaFree(dw_);
    cudaFree(dout);
    cudaFree(dref);
    cudaFree(dq);
    cudaFree(dq2);
    cudaFree(dhq);
    cudaFree(dd);
    cudaFree(dd2);
    cudaFree(dhd);
    cudaFree(dids);
    cudaStreamDestroy(s);
}

}  // namespace

// ds4_iq's dense GEMV (iq::mv): a column alone and as one of a window's 4 / 16 - the same bits (the verify windows
// of the drafts rely on it)
void mv_columns(int type, std::mt19937& rng) {
    const int k = 3072, n = 1536;
    const size_t rb = iq::row_bytes(type, k);
    std::vector<uint8_t> w(rb * n);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : w) b = (uint8_t) byte(rng);
    // (sane scales: each block's leading half-precision scale small)
    const int bb = iq::block_bytes(type), bv = iq::block_values(type);
    for (size_t r = 0; r < (size_t) n; ++r)
        for (int b = 0; b < k / bv; ++b) {
            uint8_t* p = w.data() + r * rb + (size_t) b * bb;
            if (type == iq::kQ8_0) {
                p[0] = 0x00;
                p[1] = 0x20;
            } else if (type == iq::kQ6_K) {
                p[bb - 2] = 0x00;
                p[bb - 1] = 0x20;
            } else if (type == iq::kQ4_K || type == iq::kQ5_K) {
                // (d and dmin with full mantissas: powers of two would make the minimum term's products exact, and
                //  every rounding order the same bits)
                p[0] = 0x55;
                p[1] = 0x21;
                p[2] = 0x33;
                p[3] = 0x11;
            } else {
                p[0] = 0x00;
                p[1] = 0x20;
                p[2] = 0x00;
                p[3] = 0x10;
            }
        }
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) 16 * k);
    for (int i = 0; i < k; ++i) x[(size_t) i] = nd(rng);
    for (int c = 1; c < 16; ++c)
        for (int i = 0; i < k; ++i) x[(size_t) c * k + i] = c % 2 ? nd(rng) : x[(size_t) i];   // (column 0 again at even c)
    uint8_t* dw;
    float *dx, *dy, *dd;
    int8_t* dq;
    CU(cudaMalloc(&dw, w.size()));
    CU(cudaMalloc(&dx, x.size() * 4));
    CU(cudaMalloc(&dy, (size_t) 16 * n * 4));
    CU(cudaMalloc(&dq, x.size()));
    CU(cudaMalloc(&dd, x.size() / 32 * 4));
    CU(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    cudaStream_t s = nullptr;
    std::vector<float> y1((size_t) n);
    for (int nc : {1, 4, 16}) {
        hy4::k::q8_rows(dx, k, nc, k, dq, dd, s);
        iq::MvJob j;
        j.w = dw;
        j.ld = rb;
        j.type = type;
        j.k = k;
        j.n = n;
        j.ncol = nc;
        j.xq = dq;
        j.xd = dd;
        j.y = dy;
        j.ycs = n;
        CHECK(iq::mv(&j, 1, s));
        std::vector<float> y((size_t) nc * n);
        CU(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
        if (nc == 1) {
            y1 = y;
            uint32_t hsh = 2166136261u;
            for (size_t i = 0; i < y.size() * 4; ++i) hsh = (hsh ^ ((const uint8_t*) y.data())[i]) * 16777619u;
            std::printf("  mv %s: one column, bits %08x\n", iq::name(type), hsh);
            continue;
        }
        int diff = 0;
        for (int c = 0; c < nc; c += 2)
            for (int r = 0; r < n; ++r) diff += std::memcmp(&y[(size_t) c * n + r], &y1[(size_t) r], 4) != 0;
        std::printf("  mv %s: column 0 alone vs in %d columns: %d of %d values differ\n", iq::name(type), nc, diff,
                    n * ((nc + 1) / 2));
        CHECK(diff == 0);
    }
    cudaFree(dw);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dq);
    cudaFree(dd);
}

// The CPU lane's waits by system and environment (no GPU needed): the measured set on every system, HY4_LANE_WAITS=0
// for the plain one, each variable alone for its own.
void lane_waits_cases() {
    using Env = std::map<std::string, std::string>;
    auto waits = [](bool windows, const Env& env) {
        return hy4::lane_waits(windows, [&env](const char* k) -> const char* {
            const auto it = env.find(k);
            return it == env.end() ? nullptr : it->second.c_str();
        });
    };
    hy4::LaneWaits w = waits(true, {});
    CHECK(w.spin_measured && w.spin_us == 3000 && w.ev_poll_us == 2000 && w.pin_caller);
    w = waits(false, {});
    CHECK(w.spin_measured && w.spin_us == 3000 && w.ev_poll_us == 2000 && w.pin_caller);
    w = waits(false, {{"HY4_LANE_WAITS", "0"}});
    CHECK(!w.spin_measured && w.spin_us == 3000 && w.ev_poll_us == 0 && !w.pin_caller);
    w = waits(false, {{"HY4_LANE_WAITS", "1"}});
    CHECK(w.spin_measured && w.spin_us == 3000 && w.ev_poll_us == 2000 && w.pin_caller);
    w = waits(true, {{"HY4_LANE_WAITS", "0"}});
    CHECK(!w.spin_measured && w.spin_us == 3000 && w.ev_poll_us == 0 && !w.pin_caller);
    w = waits(true, {{"HY4_LANE_SPIN_US", "0"}, {"HY4_EV_POLL_US", "0"}, {"HY4_LANE_PIN_CALLER", "0"}});
    CHECK(!w.spin_measured && w.spin_us == 0 && w.ev_poll_us == 0 && !w.pin_caller);
    w = waits(false, {{"HY4_LANE_WAITS", "0"}, {"HY4_LANE_SPIN_US", "500"}, {"HY4_EV_POLL_US", "750"},
                      {"HY4_LANE_PIN_CALLER", "1"}});
    CHECK(!w.spin_measured && w.spin_us == 500 && w.ev_poll_us == 750 && w.pin_caller);
    w = waits(false, {{"HY4_LANE_WAITS", "1"}, {"HY4_EV_POLL_US", "-5"}});
    CHECK(w.spin_measured && w.ev_poll_us == 0 && w.pin_caller);
    std::printf("  CPU lane's waits by system and environment: %s\n", fails ? "FAILED" : "ok");
}

int main() {
    lane_waits_cases();
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        if (fails) return 1;
        std::printf("hy4_experts_test: no GPU, skipped\n");
        return 0;
    }
    std::mt19937 rng(7);
    // HY4_EXPERTS_K=N: the K-quants' forms alone, N draws (two builds' bits side by side: the minimum term's rounding)
    if (const char* e = std::getenv("HY4_EXPERTS_K")) {
        for (int rep = 0; rep < std::atoi(e); ++rep)
            for (int ty : {iq::kQ4_K, iq::kQ5_K}) {
                for (int F : {512, 1536}) run_case(ty, iq::kQ6_K, 24, 1, rng, F, false);
                run_case(ty, ty, 24, 1, rng, 512, false);
                run_case(ty, ty, 24, 3, rng, 512, false);
                std::mt19937 rng2(7 + rep);
                mv_columns(ty, rng2);
            }
        std::printf("hy4_experts_test (K-quants): %s\n", fails ? "FAILED" : "ok");
        return fails ? 1 : 0;
    }
    const int types[3][2] = {{iq::kQ4_K, iq::kQ6_K}, {iq::kQ4_K, iq::kQ4_K}, {iq::kQ8_0, iq::kQ8_0}};
    for (const auto& tt : types) {
        CHECK(iq::expert_supported(tt[0]) && iq::expert_supported(tt[1]));
        for (int T : {1, 5, 16, 80}) run_case(tt[0], tt[1], 24, T, rng);
        {
            std::mt19937 rng2(7);   // (its own draws: the cases after it keep theirs)
            mv_columns(tt[0], rng2);
        }
        // a tensor-parallel slice's widths (a member's one block, a helper's two)
        for (int F : {256, 512})
            if (F % iq::block_values(tt[1]) == 0)
                for (int T : {1, 5}) run_case(tt[0], tt[1], 24, T, rng, F);
    }
    // the lower-bit types (gate = up = down a layer): the same three forms
    for (int ty : {iq::kIQ2_XXS, iq::kIQ2_S, iq::kIQ3_XXS, iq::kIQ4_XS, iq::kQ5_K}) {
        CHECK(iq::expert_supported(ty));
        for (int T : {1, 5, 16, 80}) run_case(ty, ty, 24, T, rng);
        std::mt19937 rng2(7);
        mv_columns(ty, rng2);
    }
    // the verify windows' sizes (a token and 1 .. 3 drafts), the types of Maya's sizes
    for (int ty : {iq::kIQ2_XXS, iq::kIQ3_XXS, iq::kIQ4_XS, iq::kQ4_K, iq::kQ5_K, iq::kQ8_0})
        for (int T : {2, 3, 4}) run_case(ty, ty, 24, T, rng, 512, false);
    // (the K-quants' minimum term in the gate / up alone, the down's type without one: a few draws of the token form)
    for (int rep = 0; rep < 4; ++rep)
        for (int ty : {iq::kQ4_K, iq::kQ5_K}) {
            run_case(ty, iq::kQ6_K, 24, 1, rng, 512, false);
            run_case(ty, iq::kQ6_K, 24, 3, rng, 512, false);
        }
    std::printf("hy4_experts_test: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
