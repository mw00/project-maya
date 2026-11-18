// tests/k31_moe_combine_test.cu - the K3.1 latent combine (src/k31/k31_moe_combine.hpp) held to ref/kimi_k31.py's
// seams and to a float64 reference, plus the structural mutations, each of which must fail by a wide margin.
//
//   k31_moe_combine_test --selftest      K3.1's shape: k 16, latent 3584, n_embd 7168, random parts/weights
//   k31_moe_combine_test --dumps DIR     the oracle's seams on the fixture (python tools/kimi_k31_moe_dump.py --out DIR)
//
// The composed seam is the engine's: k31 combine (sum + routed_norm) -> routed_up as a cuBLAS SGEMM with
// CUBLAS_DEFAULT_MATH (never TF32) -> + shared.  Errors:
//   moe_out, normed : float32 ULPs of the reference (unit floor 2^-126)
//   routed_out, ffn_out : max |kernel - ref| / rms(ref) over the layer (a float32 GEMV's summation order differs)
// The bounds are the measured gaps with a 4x margin, and the test asserts that every bound is at least 10x below the
// smallest structural mutation's effect (the plan's rule): picks truncated to 8, uniform weights, the norm applied to
// each part before the sum, no norm weight, the wrong eps (1e-6).
#include "../src/k31/k31_moe_combine.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr double kMinNormal = 1.17549435082228751e-38;
// bounds (measured gap x 4; see the printout): moe/normed in ULPs, routed/ffn as max-abs / rms
constexpr double kBoundUlp = 4.0;         // exact reference, moe/normed (measured 0)
constexpr double kBoundRel = 1e-5;        // GEMV order, and every oracle seam (measured <= 2.4e-6)

void cuda_ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
void blas_ok(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "%s: cublas status %d\n", what, (int) s); std::exit(2); }
}

double ulp_unit(double r) {
    const double ar = std::fabs(r);
    if (ar < kMinNormal) return kMinNormal;
    const float f = (float) ar;
    return std::max((double) std::nextafter(f, std::numeric_limits<float>::infinity()) - (double) f, kMinNormal);
}
double max_ulp(const std::vector<float>& a, const std::vector<float>& ref) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs((double) a[i] - (double) ref[i]) / ulp_unit(ref[i]));
    return m;
}
double rel_err(const std::vector<float>& a, const std::vector<float>& ref) {
    double num = 0, ss = 0;
    for (size_t i = 0; i < a.size(); ++i) { num = std::max(num, std::fabs((double) a[i] - (double) ref[i])); ss += (double) ref[i] * ref[i]; }
    return num / std::max(std::sqrt(ss / std::max<size_t>(ref.size(), 1)), 1e-30);
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

struct Layer {
    std::string name;
    int T = 0, k = 0, lat = 0, embd = 0;
    float eps = 1e-5f;
    std::vector<float> parts, w, norm_w, up_w, shexp;           // inputs
    std::vector<float> moe_out, normed, routed_out, ffn_out;    // references
};

struct Out { std::vector<float> moe_out, normed, routed_out, ffn_out; };

enum Mut { kNone, kTrunc8, kUniform, kNormEach, kNoNormW, kEps6, kNMut };
const char* const kMutName[kNMut] = {"none", "picks_truncated_to_8", "uniform_weights", "norm_before_sum", "no_norm_weight", "eps_1e-6"};

// the engine's composed seam on the device, optionally mutated
Out run(const Layer& L, Mut mut, cublasHandle_t h) {
    std::vector<float> parts = L.parts, w = L.w, nw = L.norm_w;
    int k = L.k;
    float eps = L.eps;
    if (mut == kTrunc8) {   // the combine reads only the first 8 picks (and renormalises over them, as a router would)
        std::vector<float> p8((size_t) L.T * 8 * L.lat), w8((size_t) L.T * 8);
        for (int t = 0; t < L.T; ++t) {
            double s = 0;
            for (int i = 0; i < 8; ++i) s += L.w[(size_t) t * L.k + i];
            double sall = 0;
            for (int i = 0; i < L.k; ++i) sall += L.w[(size_t) t * L.k + i];
            for (int i = 0; i < 8; ++i) {
                w8[(size_t) t * 8 + i] = (float) (L.w[(size_t) t * L.k + i] / s * sall);
                std::memcpy(&p8[((size_t) t * 8 + i) * L.lat], &L.parts[((size_t) t * L.k + i) * L.lat], L.lat * 4);
            }
        }
        parts = p8; w = w8; k = 8;
    } else if (mut == kUniform) {
        for (int t = 0; t < L.T; ++t) {
            double s = 0;
            for (int i = 0; i < k; ++i) s += w[(size_t) t * k + i];
            for (int i = 0; i < k; ++i) w[(size_t) t * k + i] = (float) (s / k);
        }
    } else if (mut == kNormEach) {   // RMS-normalise every part before the weighted sum
        for (int t = 0; t < L.T; ++t)
            for (int i = 0; i < k; ++i) {
                float* p = &parts[((size_t) t * k + i) * L.lat];
                double ms = 0;
                for (int j = 0; j < L.lat; ++j) ms += (double) p[j] * p[j];
                const double den = std::sqrt(ms / L.lat + eps);
                for (int j = 0; j < L.lat; ++j) p[j] = (float) (p[j] / den);
            }
    } else if (mut == kNoNormW) {
        std::fill(nw.begin(), nw.end(), 1.0f);
    } else if (mut == kEps6) {
        eps = 1e-6f;
    }
    float *dp = dev(parts), *dw = dev(w), *dn = dev(nw), *dup = dev(L.up_w), *dsh = dev(L.shexp);
    float *dmoe = nullptr, *dnorm = nullptr, *dr = nullptr, *dout = nullptr;
    cuda_ok(cudaMalloc(&dmoe, (size_t) L.T * L.lat * 4), "m");
    cuda_ok(cudaMalloc(&dnorm, (size_t) L.T * L.lat * 4), "m");
    cuda_ok(cudaMalloc(&dr, (size_t) L.T * L.embd * 4), "m");
    cuda_ok(cudaMalloc(&dout, (size_t) L.T * L.embd * 4), "m");
    strata::k31::CombineArgs a;
    a.parts = dp; a.w = dw; a.norm_w = dn; a.n_tokens = L.T; a.k = k; a.latent = L.lat; a.eps = eps;
    a.moe_out = dmoe; a.normed = dnorm;
    if (!strata::k31::moe_combine_launch(a, nullptr)) { std::fprintf(stderr, "combine refused\n"); std::exit(2); }
    // routed_up: (n_embd x latent, column-major, ld n_embd) x (latent x T) -> (n_embd x T); F32 means F32
    const float one = 1.0f, zero = 0.0f;
    blas_ok(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, L.embd, L.T, L.lat, &one, dup, L.embd, dnorm, L.lat, &zero, dr, L.embd), "sgemm");
    if (!strata::k31::add_launch(dr, dsh, dout, (int64_t) L.T * L.embd, nullptr)) { std::fprintf(stderr, "add refused\n"); std::exit(2); }
    Out o{host(dmoe, (size_t) L.T * L.lat), host(dnorm, (size_t) L.T * L.lat), host(dr, (size_t) L.T * L.embd), host(dout, (size_t) L.T * L.embd)};
    for (float* p : {dp, dw, dn, dup, dsh, dmoe, dnorm, dr, dout}) cudaFree(p);
    return o;
}

// float64 reference of the same composed seam (selftest): the kernel's formula with the GEMV in float64
void reference(Layer& L) {
    L.moe_out.assign((size_t) L.T * L.lat, 0);
    L.normed.assign((size_t) L.T * L.lat, 0);
    L.routed_out.assign((size_t) L.T * L.embd, 0);
    L.ffn_out.assign((size_t) L.T * L.embd, 0);
    for (int t = 0; t < L.T; ++t) {
        double ss = 0;
        for (int j = 0; j < L.lat; ++j) {
            double acc = 0;
            for (int i = 0; i < L.k; ++i) acc += (double) L.w[(size_t) t * L.k + i] * (double) L.parts[((size_t) t * L.k + i) * L.lat + j];
            const float m = (float) acc;
            L.moe_out[(size_t) t * L.lat + j] = m;
            ss += (double) m * m;
        }
        const double den = std::sqrt(ss / L.lat + (double) L.eps);
        for (int j = 0; j < L.lat; ++j) L.normed[(size_t) t * L.lat + j] = (float) (L.moe_out[(size_t) t * L.lat + j] / den) * L.norm_w[j];
        for (int o = 0; o < L.embd; ++o) {
            double acc = 0;
            for (int j = 0; j < L.lat; ++j) acc += (double) L.up_w[(size_t) j * L.embd + o] * (double) L.normed[(size_t) t * L.lat + j];
            L.routed_out[(size_t) t * L.embd + o] = (float) acc;
            L.ffn_out[(size_t) t * L.embd + o] = (float) acc + L.shexp[(size_t) t * L.embd + o];
        }
    }
}

// returns failures.  L carries the inputs and the exact reference; `oracle` (may be null) ref/kimi_k31.py's seams.
int check_layer(const Layer& L, const Layer* oracle, cublasHandle_t h, double* worst) {
    int fails = 0;
    const Out o = run(L, kNone, h);
    const double e_moe = max_ulp(o.moe_out, L.moe_out), e_norm = max_ulp(o.normed, L.normed);
    const double e_r = rel_err(o.routed_out, L.routed_out), e_f = rel_err(o.ffn_out, L.ffn_out);
    worst[0] = std::max(worst[0], e_moe); worst[1] = std::max(worst[1], e_norm);
    worst[2] = std::max(worst[2], e_r); worst[3] = std::max(worst[3], e_f);
    bool ok = e_moe <= kBoundUlp && e_norm <= kBoundUlp && e_r <= kBoundRel && e_f <= kBoundRel;
    std::printf("  %-8s T %2d k %2d latent %4d n_embd %4d  exact: moe %.2f ULP  normed %.2f ULP  routed %.2e  ffn_out %.2e",
                L.name.c_str(), L.T, L.k, L.lat, L.embd, e_moe, e_norm, e_r, e_f);
    const Layer& R = oracle ? *oracle : L;
    if (oracle) {
        const double q[4] = {rel_err(o.moe_out, R.moe_out), rel_err(o.normed, R.normed), rel_err(o.routed_out, R.routed_out),
                             rel_err(o.ffn_out, R.ffn_out)};
        for (int i = 0; i < 4; ++i) { worst[4 + i] = std::max(worst[4 + i], q[i]); ok = ok && q[i] <= kBoundRel; }
        std::printf("  | oracle: moe %.2e normed %.2e routed %.2e ffn_out %.2e", q[0], q[1], q[2], q[3]);
    }
    std::printf("  %s\n", ok ? "ok" : "FAIL");
    fails += !ok;
    // mutations: the effect on normed and on ffn_out (against the oracle when there is one) must clear 10x the bounds
    for (int m = 1; m < kNMut; ++m) {
        const Out x = run(L, (Mut) m, h);
        const double eff = std::min(rel_err(x.normed, R.normed), rel_err(x.ffn_out, R.ffn_out));
        const double need = 10.0 * std::max(kBoundRel, kBoundUlp * 1.2e-7);
        const bool caught = eff >= need;
        std::printf("    mutation %-22s effect %.2e (needs >= %.1e)  %s\n", kMutName[m], eff, need, caught ? "caught" : "NOT CAUGHT");
        fails += !caught;
    }
    return fails;
}

uint64_t g_s = 0x2545F4914F6CDD1Dull;
double urand() { g_s = g_s * 6364136223846793005ull + 1442695040888963407ull; return (double) (g_s >> 11) / 9007199254740992.0; }

int selftest(cublasHandle_t h) {
    std::printf("k31_moe_combine_test --selftest (K3.1 shape)\n");
    int fails = 0;
    double worst[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int variant = 0; variant < 2; ++variant) {
        Layer L;
        L.name = variant == 0 ? "k31" : "k31-tiny";
        L.T = 4; L.k = 16; L.lat = 3584; L.embd = 7168; L.eps = 1e-5f;
        const double scale = variant == 0 ? 1.0 : 1e-3;   // tiny parts: the eps matters
        L.parts.resize((size_t) L.T * L.k * L.lat); L.w.resize((size_t) L.T * L.k); L.norm_w.resize(L.lat);
        L.up_w.resize((size_t) L.lat * L.embd); L.shexp.resize((size_t) L.T * L.embd);
        for (auto& x : L.parts) x = (float) ((urand() - 0.5) * 2 * scale);
        for (int t = 0; t < L.T; ++t) {
            double s = 0;
            std::vector<double> r(L.k);
            for (auto& x : r) { x = urand() + 0.05; s += x; }
            for (int i = 0; i < L.k; ++i) L.w[(size_t) t * L.k + i] = (float) (r[i] / s);
        }
        for (auto& x : L.norm_w) x = (float) (0.5 + urand());
        for (auto& x : L.up_w) x = (float) ((urand() - 0.5) * 0.05);
        for (auto& x : L.shexp) x = (float) ((urand() - 0.5) * 2);
        reference(L);
        fails += check_layer(L, nullptr, h, worst);
    }
    std::printf("  worst: moe %.2f ULP, normed %.2f ULP, routed %.2e, ffn_out %.2e (bounds %.0f ULP, %.0e)\n", worst[0], worst[1],
                worst[2], worst[3], kBoundUlp, kBoundRel);
    // refusals
    strata::k31::CombineArgs bad;
    float* d = nullptr;
    cuda_ok(cudaMalloc(&d, 64), "m");
    bad.parts = d; bad.w = d; bad.norm_w = d; bad.normed = d; bad.n_tokens = 1; bad.latent = 128;
    int refused = 0;
    bad.k = 0; refused += !strata::k31::moe_combine_launch(bad, nullptr);
    bad.k = 33; refused += !strata::k31::moe_combine_launch(bad, nullptr);
    bad.k = 16; bad.latent = 8193; refused += !strata::k31::moe_combine_launch(bad, nullptr);
    bad.latent = 128; bad.normed = nullptr; refused += !strata::k31::moe_combine_launch(bad, nullptr);
    cudaFree(d);
    std::printf("  refusals (k 0, k 33, latent 8193, no output): %d/4  %s\n", refused, refused == 4 ? "ok" : "FAIL");
    fails += refused != 4;
    return fails;
}

bool read_raw(const std::string& path, size_t n, std::vector<float>& v) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    v.resize(n);
    f.read(reinterpret_cast<char*>(v.data()), (std::streamsize) (n * 4));
    return (size_t) f.gcount() == n * 4;
}

int dumps(const std::string& dir, cublasHandle_t h) {
    std::ifstream mf(dir + "/manifest.txt");
    if (!mf) { std::printf("%s/manifest.txt: missing (python tools/kimi_k31_moe_dump.py --out %s)  FAIL\n", dir.c_str(), dir.c_str()); return 1; }
    std::stringstream jss;
    jss << std::ifstream(dir + "/moe.json").rdbuf();
    const std::string js = jss.str();
    auto num = [&](const char* key) { const size_t p = js.find(std::string("\"") + key + "\""); return p == std::string::npos ? 0.0 : std::strtod(js.c_str() + js.find(':', p) + 1, nullptr); };
    const int T = (int) num("T"), k = (int) num("k"), lat = (int) num("latent"), embd = (int) num("n_embd");
    const float eps = (float) num("eps");
    std::vector<int> layers;
    std::string line;
    while (std::getline(mf, line)) {
        std::istringstream ls(line);
        std::string name;
        ls >> name;
        const std::string suf = ".parts";
        if (name.rfind("moe.", 0) == 0 && name.size() > suf.size() && name.substr(name.size() - suf.size()) == suf)
            layers.push_back(std::atoi(name.c_str() + 4));
    }
    std::printf("k31_moe_combine_test --dumps %s: %zu layers, T %d k %d latent %d n_embd %d eps %.9g\n", dir.c_str(), layers.size(), T, k, lat, embd, eps);
    int fails = 0;
    double worst[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int il : layers) {
        Layer L;
        L.name = "layer " + std::to_string(il);
        L.T = T; L.k = k; L.lat = lat; L.embd = embd; L.eps = eps;
        const std::string p = dir + "/moe." + std::to_string(il) + ".";
        const bool ok = read_raw(p + "parts.f32", (size_t) T * k * lat, L.parts) && read_raw(p + "w.f32", (size_t) T * k, L.w) &&
                        read_raw(p + "norm_w.f32", lat, L.norm_w) && read_raw(p + "up_w.f32", (size_t) lat * embd, L.up_w) &&
                        read_raw(p + "shexp.f32", (size_t) T * embd, L.shexp) && read_raw(p + "moe_out.f32", (size_t) T * lat, L.moe_out) &&
                        read_raw(p + "normed.f32", (size_t) T * lat, L.normed) && read_raw(p + "routed_out.f32", (size_t) T * embd, L.routed_out) &&
                        read_raw(p + "ffn_out.f32", (size_t) T * embd, L.ffn_out);
        if (!ok) { std::printf("  %s: unreadable  FAIL\n", L.name.c_str()); ++fails; continue; }
        Layer X = L;          // the exact reference: the same formula with the same float32 weights
        reference(X);
        fails += check_layer(X, &L, h, worst);
    }
    if (layers.empty()) { std::printf("  no moe.* layers  FAIL\n"); return 1; }
    std::printf("  worst exact: moe %.2f ULP, normed %.2f ULP, routed %.2e, ffn_out %.2e; oracle: moe %.2e normed %.2e routed %.2e "
                "ffn_out %.2e (bounds %.0f ULP, %.0e)\n", worst[0], worst[1], worst[2], worst[3], worst[4], worst[5], worst[6], worst[7],
                kBoundUlp, kBoundRel);
    return fails;
}

}  // namespace

int main(int argc, char** argv) {
    cublasHandle_t h;
    blas_ok(cublasCreate(&h), "cublasCreate");
    blas_ok(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "math mode");   // F32 means F32: no TF32
    int fails = 0;
    bool any = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") { fails += selftest(h); any = true; }
        else if (a == "--dumps" && i + 1 < argc) { fails += dumps(argv[++i], h); any = true; }
        else { std::fprintf(stderr, "usage: %s [--selftest] [--dumps DIR]\n", argv[0]); return 2; }
    }
    if (!any) fails += selftest(h);
    cublasDestroy(h);
    std::printf("k31_moe_combine_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
