// tests/glm_dsa/glm_dsa_router_test.cpp - glm-dsa B5.1 on the device (label gpu, exit 77 without a CUDA device): the
// decode router gf::moe_route at glm-dsa's 256 experts x top-8 against the F64 HF oracle and the CPU twin.
//
//   1. 10^5 random router rows (F32 logits as the F32 ffn_gate_inp GEMV writes them, F32 exp_probs_b), x2.5, HF
//      normalisation (kRouteNormHf): the published route's ids equal the oracle's outside near-ties (biased-score gap
//      < 1e-6) and its weights' relative error is < 1e-6; every expert resident, so each route is a plain publish;
//   2. an all-tiny row: kRouteNormHf weights sum to 2.5, kRouteNormClamp (Flash, Maya's default) keeps the 2^-14 clamp
//      exactly as the twin computes it;
//   3. the shared expert's SwiGLU (moe_route's extra blocks) with limit +INF equals the unclamped one bit for bit (the
//      same q8_1 as limit FLT_MAX), where Flash's 10 changes it; swiglu_q8 likewise.
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/glmd_route.hpp"

#include "glm_dsa_router_oracle.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace gf = strata::kernels::glmf;
using namespace router_oracle;

static_assert(gf::kRouteNormClamp == glmd::kRouteNormClamp && gf::kRouteNormHf == glmd::kRouteNormHf,
              "the twin's normalisation modes are the kernel's");

#define CHECK(call) do { const cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e_)); return 2; } } while (0)

static int g_fail = 0;
static void expect(bool ok, const char* what) {
    std::printf("  %-96s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

namespace {

struct Dev {
    gf::MoeDev d;
    void* arena = nullptr;
    gf::MoeRequest* ring = nullptr;   // host view
    float* logits = nullptr;          // device: every row
    float* bias = nullptr;            // device: every bias vector
    float* x = nullptr;               // device: the FFN input a route copies into its request (Maya's ROUTE_FAST
                                      // does at every route: a null x was an illegal read on CUDA and HIP alike)
};

int setup(Dev& D, const Rows& R) {
    const size_t abytes = 1 << 20;
    CHECK(cudaMalloc(&D.arena, abytes));
    CHECK(cudaMemset(D.arena, 0, abytes));
    size_t off = 0;
    auto take = [&](size_t bytes) {
        off = (off + 255) & ~size_t(255);
        void* p = (char*) D.arena + off;
        off += bytes;
        return p;
    };
    gf::MoeDev& d = D.d;
    d.n_keys = kE;
    d.tab = (unsigned long long*) take((2 * kE + gf::kSpares) * 8);
    d.scratch = (unsigned long long*) take(8 * 8);
    d.plan_ptr = (unsigned long long*) take(8 * 8);
    d.plan_w = (float*) take(8 * 4);
    d.plan_id = (int*) take(8 * 4);
    d.fetch_src = (unsigned long long*) take(8 * 8);
    d.pf_src = (unsigned long long*) take(8 * 8);
    d.pf_dst = (unsigned long long*) take(8 * 8);
    d.pf_n = (int*) take(4);
    d.seq = (unsigned int*) take(4);
    d.wait_seq = (unsigned int*) take(4);
    d.cpu_flag = (int*) take(4);
    d.cpu_part = (float*) take((size_t) gf::kMaxEmbd * 4);
    if (off > abytes) return 3;
    // every expert resident in VRAM (a fake non-null slot): the route only publishes
    std::vector<unsigned long long> table(2 * kE + gf::kSpares, 0ull);
    for (int e = 0; e < kE; ++e) table[(size_t) e] = 0x200000000ull + 4096ull * (unsigned) e;
    CHECK(cudaMemcpy(d.tab, table.data(), table.size() * 8, cudaMemcpyHostToDevice));
    const size_t ring_bytes = sizeof(gf::MoeRequest) * gf::kRingSize + sizeof(int);
    CHECK(cudaHostAlloc((void**) &D.ring, ring_bytes, cudaHostAllocMapped));
    std::memset(D.ring, 0, ring_bytes);
    void* mapped = nullptr;
    CHECK(cudaHostGetDevicePointer(&mapped, D.ring, 0));
    d.ring = mapped;
    d.route_error = (volatile int*) ((gf::MoeRequest*) d.ring + gf::kRingSize);
    CHECK(cudaMalloc(&D.logits, R.logits.size() * 4));
    CHECK(cudaMalloc(&D.bias, R.bias.size() * 4));
    CHECK(cudaMalloc(&D.x, (size_t) 4096 * 4));
    CHECK(cudaMemset(D.x, 0, (size_t) 4096 * 4));
    CHECK(cudaMemcpy(D.logits, R.logits.data(), R.logits.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(D.bias, R.bias.data(), R.bias.size() * 4, cudaMemcpyHostToDevice));
    return 0;
}

// routes rows [r0, r1) (at most kRingSize - 8 of them), then reads each published request back
int route_batch(Dev& D, const Rows& R, int r0, int r1, int norm_mode, std::vector<gf::MoeRequest>& out,
                cudaStream_t s) {
    unsigned int seq0 = 0;
    CHECK(cudaMemcpy(&seq0, D.d.seq, 4, cudaMemcpyDeviceToHost));
    for (int r = r0; r < r1; ++r)
        gf::moe_route(D.logits + (size_t) r * kE, D.bias + (size_t) (r % R.n_bias) * kE, kE, kK, kScale, true, 0,
                      D.x, 4096, D.d, nullptr, nullptr, 1.0f, 0, nullptr, s, nullptr, nullptr, 0, nullptr, nullptr,
                      0, kK, 0ull, 0, 0, 0, 0, nullptr, 0, 8, 0.0f, /*miss_skip_w*/ 0.0f, /*win_rows*/ 0,
                      /*resident_ok*/ true, norm_mode);   // (norm_mode last: before, it landed in miss_skip_w)
    CHECK(cudaStreamSynchronize(s));
    out.clear();
    for (int r = r0; r < r1; ++r) {
        const unsigned int seq = seq0 + 1 + (unsigned) (r - r0);
        const gf::MoeRequest& rq = D.ring[seq % gf::kRingSize];
        gf::MoeRequest copy;
        std::memcpy(&copy, (const void*) &rq, offsetof(gf::MoeRequest, x));
        if (copy.seq != seq) copy.error = -1;   // not published
        out.push_back(copy);
    }
    return 0;
}

int shared_expert(Dev& D, cudaStream_t s) {
    constexpr int F = 2048;   // glm-dsa n_ff_shexp
    std::vector<float> g(F), u(F);
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0f, 30.0f);   // far past Flash's 10
    for (int i = 0; i < F; ++i) {
        g[(size_t) i] = nd(rng);
        u[(size_t) i] = nd(rng);
    }
    float *dg, *du;
    void *h_inf, *h_max, *h_10, *h_sw;
    const size_t hb = (size_t) F / 32 * 36;
    CHECK(cudaMalloc(&dg, F * 4));
    CHECK(cudaMalloc(&du, F * 4));
    CHECK(cudaMalloc(&h_inf, hb));
    CHECK(cudaMalloc(&h_max, hb));
    CHECK(cudaMalloc(&h_10, hb));
    CHECK(cudaMalloc(&h_sw, hb));
    CHECK(cudaMemcpy(dg, g.data(), F * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(du, u.data(), F * 4, cudaMemcpyHostToDevice));
    const float inf = std::numeric_limits<float>::infinity();
    const auto route = [&](float lim, void* hq) {
        gf::moe_route(D.logits, D.bias, kE, kK, kScale, true, 0, D.x, 4096, D.d, dg, du, lim, F, hq, s, nullptr,
                      nullptr, 0, nullptr, nullptr, 0, kK, 0ull, 0, 0, 0, 0, nullptr, 0, 8, 0.0f, 0.0f, 0, true,
                      gf::kRouteNormHf);
    };
    route(inf, h_inf);
    route(FLT_MAX, h_max);
    route(10.0f, h_10);
    gf::swiglu_q8(dg, du, inf, F, h_sw, s);
    CHECK(cudaStreamSynchronize(s));
    std::vector<uint8_t> a(hb), b(hb), c(hb), sw(hb);
    CHECK(cudaMemcpy(a.data(), h_inf, hb, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(b.data(), h_max, hb, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(c.data(), h_10, hb, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sw.data(), h_sw, hb, cudaMemcpyDeviceToHost));
    expect(a == b, "shared expert SwiGLU: limit +INF == limit FLT_MAX (no clamp), bit for bit");
    expect(a != c, "shared expert SwiGLU: Flash's limit 10 differs on these activations");
    expect(a == sw, "swiglu_q8 at +INF == the shared expert's blocks");
    for (void* p : {(void*) dg, (void*) du, h_inf, h_max, h_10, h_sw}) cudaFree(p);
    return 0;
}

}  // namespace

int main() {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::printf("glm_dsa_router_test: no CUDA device - skipped\n");
        return 77;
    }
    std::printf("glm_dsa_router_test\n");
    cudaStream_t s;
    CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    const Rows R = make_rows(100000, 75, 53u);
    Dev D;
    if (int rc = setup(D, R)) return rc;
    const int e0 = gf::launch_errors();

    // 1. 10^5 rows, the kernel vs the oracle (and vs the twin, for the record)
    {
        Tally t, tw;
        long unpublished = 0, twin_id_diff = 0;
        std::vector<gf::MoeRequest> got;
        const int batch = gf::kRingSize - 8;
        for (int r0 = 0; r0 < R.n; r0 += batch) {
            const int r1 = std::min(R.n, r0 + batch);
            if (int rc = route_batch(D, R, r0, r1, gf::kRouteNormHf, got, s)) return rc;
            for (int r = r0; r < r1; ++r) {
                const gf::MoeRequest& rq = got[(size_t) (r - r0)];
                if (rq.error != 0) {
                    ++unpublished;
                    continue;
                }
                const float* lg = R.logits.data() + (size_t) r * kE;
                const float* bs = R.bias.data() + (size_t) (r % R.n_bias) * kE;
                const Pick o = oracle(lg, bs);
                compare(o, rq.ids, rq.w, t);
                int ids[kK];
                float w[kK];
                glmd::cpu::route_select(lg, bs, kE, kK, kScale, true, glmd::kRouteNormHf, ids, w);
                for (int i = 0; i < kK; ++i)
                    if (ids[i] != rq.ids[i]) {
                        ++twin_id_diff;
                        break;
                    }
            }
        }
        std::printf("  rows %ld, near-ties skipped %ld, id mismatches %ld, weight failures %ld, worst rel err %.3g; "
                    "kernel vs twin id differences %ld\n",
                    t.rows, t.near_ties, t.id_fail, t.w_fail, t.worst_rel, twin_id_diff);
        expect(unpublished == 0, "every route published without error");
        expect(t.id_fail == 0, "kernel top-8 ids == oracle outside near-ties (10^5 rows, E 256)");
        expect(t.w_fail == 0 && t.worst_rel < kRelTol, "kernel weights rel err < 1e-6");
        expect(twin_id_diff <= t.near_ties, "kernel and twin differ only where the oracle sees a near-tie");
    }

    // 2. the all-tiny row: HF vs the clamp on the device
    {
        Rows T;
        T.n = 2;
        T.n_bias = 1;
        T.logits.assign((size_t) 2 * kE, -30.0f);
        for (int e = 0; e < kE; ++e) T.logits[(size_t) e] = T.logits[(size_t) kE + e] = -30.0f + 0.001f * (float) (e % 17);
        T.bias.assign(kE, 0.0f);
        CHECK(cudaMemcpy(D.logits, T.logits.data(), T.logits.size() * 4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(D.bias, T.bias.data(), T.bias.size() * 4, cudaMemcpyHostToDevice));
        std::vector<gf::MoeRequest> hf, cl;
        if (int rc = route_batch(D, T, 0, 1, gf::kRouteNormHf, hf, s)) return rc;
        if (int rc = route_batch(D, T, 1, 2, gf::kRouteNormClamp, cl, s)) return rc;
        int ids[kK];
        float w_cl[kK];
        glmd::cpu::route_select(T.logits.data(), T.bias.data(), kE, kK, kScale, true, glmd::kRouteNormClamp, ids, w_cl);
        double sh = 0, sc = 0, rel = 0;
        for (int i = 0; i < kK; ++i) {
            sh += hf[0].w[i];
            sc += cl[0].w[i];
            rel = std::max(rel, std::fabs((double) cl[0].w[i] - w_cl[i]) / w_cl[i]);
        }
        char what[160];
        std::snprintf(what, sizeof what, "all-tiny row: kRouteNormHf weights sum to 2.5 (sum %.7g, error %d, w0 %.4g)", sh,
                      hf[0].error, (double) hf[0].w[0]);
        expect(hf[0].error == 0 && std::fabs(sh - 2.5) < 1e-5, what);
        expect(cl[0].error == 0 && sc < 1e-6 && rel < 1e-6, "all-tiny row: kRouteNormClamp keeps Maya's 2^-14 clamp");
    }

    // 3. no SwiGLU clamp on the shared expert
    if (int rc = shared_expert(D, s)) return rc;
    expect(gf::launch_errors() == e0, "no launch errors");

    cudaFreeHost(D.ring);
    cudaFree(D.logits);
    cudaFree(D.bias);
    cudaFree(D.x);
    cudaFree(D.arena);
    cudaStreamDestroy(s);
    if (g_fail != 0) {
        std::printf("glm_dsa_router_test: %d check(s) failed\n", g_fail);
        return 1;
    }
    std::printf("glm_dsa_router_test: ok\n");
    return 0;
}
