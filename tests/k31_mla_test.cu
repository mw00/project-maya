// tests/k31_mla_test.cu - the K3.1 MLA attention core and gate (src/k31/k31_mla.hpp) held to ref/kimi_k31.py's
// mla_layer and to a float64 reference, plus the mutations, each of which must fail by a wide margin.
//
//   k31_mla_test --selftest      K3.1's shape: 96 heads, 2600 cached keys, a 2048-pick query, a 1001-key query, a 1-key query
//   k31_mla_test --dumps DIR     every MLA layer of the fixture (python tools/kimi_k31_mla_dump.py --out DIR)
//
// The composed seam is the engine's M1 path: k31 core (ctx) -> attn_v_b per head as a strided-batched SGEMM ->
// k31 gate -> attn_output as an SGEMM (CUBLAS_DEFAULT_MATH, never TF32).  Errors: ctx in float32 ULPs of the
// reference and as max |d| / rms(ref); kqv_out, gated, mla_out as max |d| / rms(ref).  Bounds: the measured gap
// with a margin, each at least 10x below the smallest mutation's effect: the scale 1/sqrt(576) or 1/sqrt(128), the
// 64 un-rotated dims dropped from the score, the value taken from row[64..576) instead of row[0..512), the
// indexer's selection ignored (every visible key), a future key let in, no output gate, a SiLU gate (g*sigmoid(g)),
// the heads transposed in kqv_out.
#include "../src/k31/k31_mla.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

using strata::k31::kMlaLatent;
using strata::k31::kMlaRow;
constexpr double kMinNormal = 1.17549435082228751e-38;
constexpr double kBoundUlp = 2.0, kBoundRelCtx = 1e-6;
constexpr double kBoundRel = 2.5e-5;   // kqv_out / gated / mla_out: a float32 v_b GEMM against the oracle's float64 (measured <= 6.3e-6, x4)
const double kNeed = 10.0 * kBoundRel;
using Vec = std::vector<float>;
using IVec = std::vector<int32_t>;

void cuda_ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
void blas_ok(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "%s: cublas %d\n", what, (int) s); std::exit(2); }
}
double ulp_unit(double r) {
    const double ar = std::fabs(r);
    if (ar < kMinNormal) return kMinNormal;
    const float f = (float) ar;
    return std::max((double) std::nextafter(f, std::numeric_limits<float>::infinity()) - (double) f, kMinNormal);
}
double max_ulp(const Vec& a, const Vec& r) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs((double) a[i] - (double) r[i]) / ulp_unit(r[i]));
    return m;
}
double rel_err(const Vec& a, const Vec& r) {
    double num = 0, ss = 0;
    for (size_t i = 0; i < a.size(); ++i) { num = std::max(num, std::fabs((double) a[i] - (double) r[i])); ss += (double) r[i] * r[i]; }
    return num / std::max(std::sqrt(ss / std::max<size_t>(r.size(), 1)), 1e-30);
}
template <class T>
T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    cuda_ok(cudaMalloc(&d, std::max<size_t>(h.size(), 1) * sizeof(T)), "malloc");
    if (!h.empty()) cuda_ok(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "h2d");
    return d;
}
template <class T>
std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    cuda_ok(cudaDeviceSynchronize(), "sync");
    cuda_ok(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "d2h");
    return h;
}

struct Att {
    int T = 0, H = 0, n_kv = 0, k_sel = 0;
    double scale = 0;
    Vec q, kv;   // [T][H][576], [n_kv][576]
    IVec sel;    // [T][k_sel]
};

Vec run_core(const Att& A) {
    float *dq = dev(A.q), *dkv = dev(A.kv), *dctx = nullptr;
    int32_t* ds = dev(A.sel);
    cuda_ok(cudaMalloc(&dctx, (size_t) A.T * A.H * kMlaLatent * 4), "malloc");
    strata::k31::MlaAttnArgs a;
    a.q = dq; a.kv = dkv; a.sel = ds; a.n_tokens = A.T; a.n_head = A.H; a.n_kv = A.n_kv; a.k_sel = A.k_sel; a.scale = A.scale; a.ctx = dctx;
    if (!strata::k31::mla_attn_launch(a, nullptr)) { std::fprintf(stderr, "mla_attn refused\n"); std::exit(2); }
    Vec ctx = host(dctx, (size_t) A.T * A.H * kMlaLatent);
    cudaFree(dq); cudaFree(dkv); cudaFree(dctx); cudaFree(ds);
    return ctx;
}

// float64 reference of the core, with the misreadings: use_pe = false drops the 64 dims from the score; voff = 64
// takes the value from row[64..576); `sel` may be replaced (selection ignored / a future key)
Vec ref_core(const Att& A, double scale, bool use_pe, int voff, const IVec& sel, int k_sel) {
    Vec ctx((size_t) A.T * A.H * kMlaLatent, 0);
    for (int t = 0; t < A.T; ++t) {
        std::vector<int> keys;
        for (int i = 0; i < k_sel; ++i) if (sel[(size_t) t * k_sel + i] >= 0) keys.push_back(sel[(size_t) t * k_sel + i]);
        std::sort(keys.begin(), keys.end());   // the oracle attends in position order
        if (keys.empty()) continue;
        for (int h = 0; h < A.H; ++h) {
            const float* q = &A.q[((size_t) t * A.H + h) * kMlaRow];
            std::vector<double> s(keys.size());
            double mx = -1e300;
            for (size_t j = 0; j < keys.size(); ++j) {
                const float* row = &A.kv[(size_t) keys[j] * kMlaRow];
                double acc = 0;
                for (int d = 0; d < (use_pe ? kMlaRow : kMlaLatent); ++d) acc += (double) q[d] * (double) row[d];
                s[j] = acc * scale;
                mx = std::max(mx, s[j]);
            }
            double tot = 0;
            for (auto& x : s) { x = std::exp(x - mx); tot += x; }
            for (auto& x : s) x /= tot;
            for (int d = 0; d < kMlaLatent; ++d) {
                double acc = 0;
                for (size_t j = 0; j < keys.size(); ++j) acc += s[j] * (double) A.kv[(size_t) keys[j] * kMlaRow + voff + d];
                ctx[((size_t) t * A.H + h) * kMlaLatent + d] = (float) acc;
            }
        }
    }
    return ctx;
}

// attn_v_b per head on the device: out[t][h*vh + v] = sum_c wv_b[h][v][c] * ctx[t][h][c]
Vec run_vb(cublasHandle_t hb, const Vec& ctx, const Vec& wv_b, int T, int H, int vh) {
    float *dctx = dev(ctx), *dw = dev(wv_b), *dout = nullptr;
    cuda_ok(cudaMalloc(&dout, (size_t) T * H * vh * 4), "malloc");
    const float one = 1.0f, zero = 0.0f;
    blas_ok(cublasSgemmStridedBatched(hb, CUBLAS_OP_T, CUBLAS_OP_N, vh, T, kMlaLatent, &one, dw, kMlaLatent, (long long) vh * kMlaLatent,
                                      dctx, H * kMlaLatent, kMlaLatent, &zero, dout, H * vh, vh, H), "vb");
    Vec out = host(dout, (size_t) T * H * vh);
    cudaFree(dctx); cudaFree(dw); cudaFree(dout);
    return out;
}

Vec run_gate(const Vec& kqv, const Vec& g) {
    float *dk = dev(kqv), *dg = dev(g), *dout = nullptr;
    cuda_ok(cudaMalloc(&dout, kqv.size() * 4), "malloc");
    if (!strata::k31::mla_gate_launch(dk, dg, dout, (int64_t) kqv.size(), nullptr)) { std::fprintf(stderr, "gate refused\n"); std::exit(2); }
    Vec out = host(dout, kqv.size());
    cudaFree(dk); cudaFree(dg); cudaFree(dout);
    return out;
}

Vec run_wo(cublasHandle_t hb, const Vec& gated, const Vec& wo, int T, int in, int n_embd) {
    float *dg = dev(gated), *dw = dev(wo), *dout = nullptr;
    cuda_ok(cudaMalloc(&dout, (size_t) T * n_embd * 4), "malloc");
    const float one = 1.0f, zero = 0.0f;
    blas_ok(cublasSgemm(hb, CUBLAS_OP_N, CUBLAS_OP_N, n_embd, T, in, &one, dw, n_embd, dg, in, &zero, dout, n_embd), "wo");
    Vec out = host(dout, (size_t) T * n_embd);
    cudaFree(dg); cudaFree(dw); cudaFree(dout);
    return out;
}

uint64_t g_s = 0x9E3779B97F4A7C15ull;
double urand() { g_s = g_s * 6364136223846793005ull + 1442695040888963407ull; return (double) (g_s >> 11) / 9007199254740992.0; }
double nrand() { const double u = std::max(urand(), 1e-300), v = urand(); return std::sqrt(-2 * std::log(u)) * std::cos(6.283185307179586 * v); }

struct Eff { const char* name; double e = 1e300; };

int selftest() {
    std::printf("k31_mla_test --selftest (96 heads, 2600 keys)\n");
    Att A;
    A.T = 3; A.H = 96; A.n_kv = 2600; A.k_sel = 2048; A.scale = 1.0 / std::sqrt(192.0);
    A.q.resize((size_t) A.T * A.H * kMlaRow); A.kv.resize((size_t) A.n_kv * kMlaRow); A.sel.assign((size_t) A.T * A.k_sel, -1);
    for (auto& x : A.q) x = (float) (0.15 * nrand());
    for (auto& x : A.kv) x = (float) nrand();
    const int pos[3] = {2599, 1000, 0};
    for (int t = 0; t < A.T; ++t) {
        std::vector<int> vis(pos[t] + 1);
        for (int i = 0; i <= pos[t]; ++i) vis[i] = i;
        for (int i = (int) vis.size() - 1; i > 0; --i) std::swap(vis[i], vis[(int) (urand() * (i + 1))]);   // any order
        const int n = std::min<int>((int) vis.size(), A.k_sel);
        for (int i = 0; i < n; ++i) A.sel[(size_t) t * A.k_sel + i] = vis[i];
    }
    const Vec ctx = run_core(A), ref = ref_core(A, A.scale, true, 0, A.sel, A.k_sel);
    const double eu = max_ulp(ctx, ref), er = rel_err(ctx, ref);
    int fails = !(eu <= kBoundUlp && er <= kBoundRelCtx);
    std::printf("  ctx: %.2f ULP, %.2e (bounds %.0f ULP, %.0e)  %s\n", eu, er, kBoundUlp, kBoundRelCtx, fails ? "FAIL" : "ok");
    // gate against the float64 formula (exact up to the sigmoid's rounding)
    Vec kq(4096), g(4096);
    for (auto& x : kq) x = (float) nrand();
    for (auto& x : g) x = (float) (8 * nrand());
    const Vec gt = run_gate(kq, g);
    long gbad = 0;
    for (size_t i = 0; i < kq.size(); ++i) gbad += gt[i] != kq[i] * (float) (1.0 / (1.0 + std::exp(-(double) g[i])));
    std::printf("  gate: %ld of %zu differ from kqv * float(sigmoid64)  %s\n", gbad, kq.size(), gbad ? "FAIL" : "ok");
    fails += gbad != 0;
    // mutations on the core (the effect on ctx)
    std::vector<Eff> eff = {{"scale 1/sqrt(576)"}, {"scale 1/sqrt(128)"}, {"64 un-rotated dims dropped"}, {"value from row[64..576)"},
                            {"selection ignored"}, {"a future key let in"}};
    Att B = A;
    B.scale = 1.0 / std::sqrt(576.0);
    eff[0].e = rel_err(run_core(B), ref);
    B.scale = 1.0 / std::sqrt(128.0);
    eff[1].e = rel_err(run_core(B), ref);
    eff[2].e = rel_err(ref_core(A, A.scale, false, 0, A.sel, A.k_sel), ref);
    eff[3].e = rel_err(ref_core(A, A.scale, true, 64, A.sel, A.k_sel), ref);
    {   // every visible key instead of the 2048 picks (token 0 only differs)
        IVec all((size_t) A.T * A.n_kv, -1);
        for (int t = 0; t < A.T; ++t) for (int i = 0; i <= pos[t]; ++i) all[(size_t) t * A.n_kv + i] = i;
        eff[4].e = rel_err(ref_core(A, A.scale, true, 0, all, A.n_kv), ref);
    }
    {   // position pos+1 added to the 1001-key query
        Att C = A;
        C.sel[(size_t) 1 * C.k_sel + (pos[1] + 1)] = pos[1] + 1;
        eff[5].e = rel_err(run_core(C), ref);
    }
    for (const auto& e : eff) {
        const bool caught = e.e >= kNeed;
        std::printf("  mutation %-28s effect %.2e (needs >= %.0e)  %s\n", e.name, e.e, kNeed, caught ? "caught" : "NOT CAUGHT");
        fails += !caught;
    }
    // refusals
    strata::k31::MlaAttnArgs bad;
    float* d = nullptr;
    int32_t* di = nullptr;
    cuda_ok(cudaMalloc(&d, 64), "m");
    cuda_ok(cudaMalloc(&di, 64), "m");
    bad.q = d; bad.kv = d; bad.sel = di; bad.ctx = d; bad.n_tokens = 1; bad.n_head = 1; bad.n_kv = 4;
    int refused = 0;
    bad.k_sel = 0; refused += !strata::k31::mla_attn_launch(bad, nullptr);
    bad.k_sel = 2049; refused += !strata::k31::mla_attn_launch(bad, nullptr);
    bad.k_sel = 4; bad.n_head = 0; refused += !strata::k31::mla_attn_launch(bad, nullptr);
    cudaFree(d); cudaFree(di);
    std::printf("  refusals (k_sel 0, k_sel 2049, 0 heads): %d/3  %s\n", refused, refused == 3 ? "ok" : "FAIL");
    fails += refused != 3;
    return fails;
}

bool read_raw(const std::string& path, size_t n, void* out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.read(static_cast<char*>(out), (std::streamsize) (n * 4));
    return (size_t) f.gcount() == n * 4;
}

int dumps(const std::string& dir, cublasHandle_t hb) {
    std::stringstream jss;
    jss << std::ifstream(dir + "/mla.json").rdbuf();
    const std::string js = jss.str();
    if (js.empty()) { std::printf("%s/mla.json: missing (python tools/kimi_k31_mla_dump.py --out %s)  FAIL\n", dir.c_str(), dir.c_str()); return 1; }
    auto num = [&](const char* key) { const size_t p = js.find(std::string("\"") + key + "\""); return p == std::string::npos ? 0.0 : std::strtod(js.c_str() + js.find(':', p) + 1, nullptr); };
    const int H = (int) num("H"), T = (int) num("T"), n_kv = (int) num("n_kv"), vh = (int) num("v_head"), k = (int) num("top_k"), n_embd = (int) num("n_embd");
    const double scale = num("scale");
    std::vector<int> layers;
    {
        const size_t a = js.find('[', js.find("\"layers\"")), b = js.find(']', a);
        std::stringstream ls(js.substr(a + 1, b - a - 1));
        std::string tok;
        while (std::getline(ls, tok, ',')) layers.push_back(std::atoi(tok.c_str()));
    }
    std::printf("k31_mla_test --dumps %s: layers %zu, H %d, T %d, n_kv %d, top_k %d, v_head %d, n_embd %d\n", dir.c_str(), layers.size(), H, T, n_kv, k, vh, n_embd);
    int fails = 0;
    std::vector<Eff> eff = {{"scale 1/sqrt(576)"}, {"scale 1/sqrt(128)"}, {"64 un-rotated dims dropped"}, {"value from row[64..576)"},
                            {"selection ignored"}, {"a future key let in"}, {"no output gate"}, {"SiLU gate"}, {"heads transposed in kqv_out"}};
    double worst[5] = {0, 0, 0, 0, 0};
    for (int il : layers) {
        Att A;
        A.T = T; A.H = H; A.n_kv = n_kv; A.k_sel = k; A.scale = scale;
        A.q.resize((size_t) T * H * kMlaRow); A.kv.resize((size_t) n_kv * kMlaRow); A.sel.resize((size_t) T * k);
        Vec rctx((size_t) T * H * kMlaLatent), wv_b((size_t) H * vh * kMlaLatent), rkqv((size_t) T * H * vh), gpre((size_t) T * H * vh),
            rgated((size_t) T * H * vh), wo((size_t) H * vh * n_embd), rout((size_t) T * n_embd);
        const std::string p = dir + "/mla." + std::to_string(il) + ".";
        const bool ok = read_raw(p + "q.f32", A.q.size(), A.q.data()) && read_raw(p + "kv.f32", A.kv.size(), A.kv.data()) &&
                        read_raw(p + "sel.f32", A.sel.size(), A.sel.data()) && read_raw(p + "ctx.f32", rctx.size(), rctx.data()) &&
                        read_raw(p + "wv_b.f32", wv_b.size(), wv_b.data()) && read_raw(p + "kqv_out.f32", rkqv.size(), rkqv.data()) &&
                        read_raw(p + "gate_pre.f32", gpre.size(), gpre.data()) && read_raw(p + "gated.f32", rgated.size(), rgated.data()) &&
                        read_raw(p + "wo.f32", wo.size(), wo.data()) && read_raw(p + "out.f32", rout.size(), rout.data());
        if (!ok) { std::printf("  layer %d: unreadable  FAIL\n", il); ++fails; continue; }
        const Vec ctx = run_core(A);
        const Vec kqv = run_vb(hb, ctx, wv_b, T, H, vh);
        const Vec gated = run_gate(kqv, gpre);
        const Vec out = run_wo(hb, gated, wo, T, H * vh, n_embd);
        const double e[5] = {max_ulp(ctx, rctx), rel_err(ctx, rctx), rel_err(kqv, rkqv), rel_err(gated, rgated), rel_err(out, rout)};
        for (int i = 0; i < 5; ++i) worst[i] = std::max(worst[i], e[i]);
        const bool lok = e[0] <= kBoundUlp && e[1] <= kBoundRelCtx && e[2] <= kBoundRel && e[3] <= kBoundRel && e[4] <= kBoundRel;
        std::printf("  layer %d: ctx %.2f ULP %.2e  kqv_out %.2e  gated %.2e  mla_out %.2e  %s\n", il, e[0], e[1], e[2], e[3], e[4], lok ? "ok" : "FAIL");
        fails += !lok;
        // mutations: the effect on mla_out
        auto through = [&](const Vec& c) { return run_wo(hb, run_gate(run_vb(hb, c, wv_b, T, H, vh), gpre), wo, T, H * vh, n_embd); };
        Att B = A;
        B.scale = 1.0 / std::sqrt(576.0);
        eff[0].e = std::min(eff[0].e, rel_err(through(run_core(B)), rout));
        B.scale = 1.0 / std::sqrt(128.0);
        eff[1].e = std::min(eff[1].e, rel_err(through(run_core(B)), rout));
        eff[2].e = std::min(eff[2].e, rel_err(through(ref_core(A, scale, false, 0, A.sel, k)), rout));
        eff[3].e = std::min(eff[3].e, rel_err(through(ref_core(A, scale, true, 64, A.sel, k)), rout));
        {
            IVec all((size_t) T * n_kv, -1);
            for (int t = 0; t < T; ++t) for (int i = 0; i <= t; ++i) all[(size_t) t * n_kv + i] = i;   // one prefill call: pos = t
            eff[4].e = std::min(eff[4].e, rel_err(through(ref_core(A, scale, true, 0, all, n_kv)), rout));
        }
        {   // the next position let in for every query that has room
            IVec fut((size_t) T * (k + 1), -1);
            for (int t = 0; t < T; ++t) {
                int c = 0;
                for (int i = 0; i < k; ++i) if (A.sel[(size_t) t * k + i] >= 0) fut[(size_t) t * (k + 1) + c++] = A.sel[(size_t) t * k + i];
                if (t + 1 < n_kv) fut[(size_t) t * (k + 1) + c] = t + 1;
            }
            eff[5].e = std::min(eff[5].e, rel_err(through(ref_core(A, scale, true, 0, fut, k + 1)), rout));
        }
        eff[6].e = std::min(eff[6].e, rel_err(run_wo(hb, kqv, wo, T, H * vh, n_embd), rout));
        {
            Vec silu(kqv.size());
            for (size_t i = 0; i < kqv.size(); ++i) silu[i] = kqv[i] * gpre[i] * (float) (1.0 / (1.0 + std::exp(-(double) gpre[i])));
            eff[7].e = std::min(eff[7].e, rel_err(run_wo(hb, silu, wo, T, H * vh, n_embd), rout));
        }
        {
            Vec tr(kqv.size());
            for (int t = 0; t < T; ++t)
                for (int h = 0; h < H; ++h)
                    for (int v = 0; v < vh; ++v) tr[(size_t) t * H * vh + (size_t) v * H + h] = kqv[(size_t) t * H * vh + (size_t) h * vh + v];
            eff[8].e = std::min(eff[8].e, rel_err(run_wo(hb, run_gate(tr, gpre), wo, T, H * vh, n_embd), rout));
        }
    }
    std::printf("  worst: ctx %.2f ULP %.2e, kqv_out %.2e, gated %.2e, mla_out %.2e (bounds %.0f ULP, %.0e; %.0e)\n", worst[0], worst[1],
                worst[2], worst[3], worst[4], kBoundUlp, kBoundRelCtx, kBoundRel);
    for (const auto& e : eff) {
        const bool caught = e.e >= kNeed && e.e < 1e299;
        std::printf("  mutation %-28s smallest effect on mla_out %.2e (needs >= %.0e)  %s\n", e.name, e.e, kNeed, caught ? "caught" : "NOT CAUGHT");
        fails += !caught;
    }
    return fails;
}

}  // namespace

int main(int argc, char** argv) {
    cublasHandle_t hb;
    blas_ok(cublasCreate(&hb), "cublasCreate");
    blas_ok(cublasSetMathMode(hb, CUBLAS_DEFAULT_MATH), "math mode");
    int fails = 0;
    bool any = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") { fails += selftest(); any = true; }
        else if (a == "--dumps" && i + 1 < argc) { fails += dumps(argv[++i], hb); any = true; }
        else { std::fprintf(stderr, "usage: %s [--selftest] [--dumps DIR]\n", argv[0]); return 2; }
    }
    if (!any) fails += selftest();
    cublasDestroy(hb);
    std::printf("k31_mla_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
