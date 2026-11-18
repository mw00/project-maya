// tests/k31_situ_test.cu - k31 SiTU-GLU (src/k31/k31_situ.hpp) on the device and the host, held to the float64
// reference and to tools/kimi_k31_golden.py's vectors; plus the wrong readings, each of which must fail.
//
//   k31_situ_test --selftest            built-in inputs (sweeps, magnitudes 1e-30..1e4, edges, +-inf), no files
//   k31_situ_test --golden DIR          the golden vectors (python tools/kimi_k31_golden.py --out DIR)
//
// The gate (the golden's rule): |out - ref| <= budget float32 ULPs of the float64 reference where g >= -87 (ULP unit
// floored at 2^-126), |out - ref| <= 1e-35 below g = -87 (where any float32 sigmoid leaves the normal range),
// infinities where the float32-rounded reference is infinite, NaN positions equal.  Budget 10 ULPs, measured in
// tools/kimi_k31_golden.py from a plain float32 kernel (worst 4.56); the shipped forms' adversarial worst is 6.9
// (two of those pairs are in the selftest), so the headroom is ~1.3x.  The ULP unit is floored at 2^-126, so for
// |ref| < 2^-103 (g below ~-72 at |u| ~ 1) the gate is in effect absolute (budget x 2^-126).  Checked for four forms
// (device precise, device portable, host precise, host portable) at the released betas 4 / 25 and at (4, 0) (a
// linear up branch), (1, 1) and (3, 7); the two portable forms must also be bit-identical for every non-NaN output.
// Wrong readings (SwiGLU, swapped betas, a linear up branch, no sigmoid, no gate tanh) must fail by >= 10x the
// budget.  The fast-math reading (what nvcc --use_fast_math makes of the precise form) fails by ~5.6x only: it is a
// build setting, guarded in CMake, and reported here as such.
#include "../src/k31/k31_situ.hpp"

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

using strata::k31::kSituBeta;
using strata::k31::kSituLinearBeta;

namespace {

constexpr double kMinNormal = 1.17549435082228751e-38;   // 2^-126
double g_budget_ulp = 10.0, g_under_g = -87.0, g_under_abs = 1e-35;

void cuda_ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

double ref64(float g, float u, double beta = kSituBeta, double lbeta = kSituLinearBeta) {
    const double gd = g, ud = u;
    const double a = beta * std::tanh(gd / beta) / (1.0 + std::exp(-gd));
    return a * (lbeta > 0 ? lbeta * std::tanh(ud / lbeta) : ud);
}

double ulp_unit(double ref) {
    const double ar = std::fabs(ref);
    if (ar < kMinNormal) return kMinNormal;
    const float f = (float) ar;
    const double sp = (double) std::nextafter(f, std::numeric_limits<float>::infinity()) - (double) f;
    return std::max(sp, kMinNormal);
}

struct Stats {
    double worst_ulp = 0, worst_abs_under = 0;
    long bad = 0, n = 0;
};

Stats gate(const std::vector<float>& g, const std::vector<double>& ref, const std::vector<float>& out) {
    Stats s;
    for (size_t i = 0; i < out.size(); ++i) {
        ++s.n;
        const double o = out[i], r = ref[i];
        if (std::isnan(r) || std::isnan(o)) {
            if (std::isnan(r) != std::isnan(o)) ++s.bad;
            continue;
        }
        if (std::isinf((float) r) || std::isinf(o)) {   // past FLT_MAX the right float32 answer is the infinity
            if (o != (double) (float) r) ++s.bad;
            continue;
        }
        if (g[i] < g_under_g) {
            const double a = std::fabs(o - r);
            s.worst_abs_under = std::max(s.worst_abs_under, a);
            if (a > g_under_abs) ++s.bad;
        } else {
            const double e = std::fabs(o - r) / ulp_unit(r);
            s.worst_ulp = std::max(s.worst_ulp, e);
            if (e > g_budget_ulp) ++s.bad;
        }
    }
    return s;
}

// ---- wrong readings (test-only device code)
enum Wrong { kSwiglu, kSwapped, kLinearUp, kNoSigmoid, kNoGateTanh, kFastMath, kNWrong };
const char* const kWrongName[kNWrong] = {"swiglu", "swapped_betas", "linear_up", "no_sigmoid", "no_gate_tanh", "fast_math"};
const bool kBuildRule[kNWrong] = {false, false, false, false, false, true};   // a compiler setting, not a reading

__global__ void wrong_kernel(const float* g, const float* u, float* out, int64_t n, int which) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gv = g[i], uv = u[i];
    const float sg = 1.0f / (1.0f + expf(-gv));
    float r = 0;
    switch (which) {
        case kSwiglu: r = gv * sg * uv; break;
        case kSwapped: r = 25.0f * tanhf(gv / 25.0f) * sg * 4.0f * tanhf(uv / 4.0f); break;
        case kLinearUp: r = 4.0f * tanhf(gv / 4.0f) * sg * uv; break;
        case kNoSigmoid: r = 4.0f * tanhf(gv / 4.0f) * 25.0f * tanhf(uv / 25.0f); break;
        case kNoGateTanh: r = gv * sg * 25.0f * tanhf(uv / 25.0f); break;
        // what nvcc --use_fast_math (CUDA 12.4) makes of situ(): expf -> __expf, approximate divisions, tanhf kept
        case kFastMath: r = 4.0f * tanhf(__fdividef(gv, 4.0f)) * __fdividef(1.0f, 1.0f + __expf(-gv)) * 25.0f * tanhf(__fdividef(uv, 25.0f)); break;
    }
    out[i] = r;
}

struct Dev {
    float *g = nullptr, *u = nullptr, *o = nullptr;
    int64_t n = 0;
    explicit Dev(const std::vector<float>& gv, const std::vector<float>& uv) : n((int64_t) gv.size()) {
        cuda_ok(cudaMalloc(&g, n * 4), "malloc g");
        cuda_ok(cudaMalloc(&u, n * 4), "malloc u");
        cuda_ok(cudaMalloc(&o, n * 4), "malloc o");
        cuda_ok(cudaMemcpy(g, gv.data(), n * 4, cudaMemcpyHostToDevice), "h2d g");
        cuda_ok(cudaMemcpy(u, uv.data(), n * 4, cudaMemcpyHostToDevice), "h2d u");
    }
    ~Dev() { cudaFree(g); cudaFree(u); cudaFree(o); }
    std::vector<float> get() const {
        std::vector<float> h((size_t) n);
        cuda_ok(cudaDeviceSynchronize(), "sync");
        cuda_ok(cudaMemcpy(h.data(), o, n * 4, cudaMemcpyDeviceToHost), "d2h o");
        return h;
    }
};

// returns failures
int run_case(const char* name, const std::vector<float>& g, const std::vector<float>& u, const std::vector<double>& ref,
             float beta = kSituBeta, float lbeta = kSituLinearBeta) {
    int fails = 0;
    Dev d(g, u);
    std::vector<float> dev_p, dev_q, cpu_p((size_t) g.size()), cpu_q((size_t) g.size());
    if (!strata::k31::situ_launch(d.g, d.u, d.o, d.n, beta, lbeta, false, nullptr)) { std::printf("  launch failed\n"); return 1; }
    dev_p = d.get();
    if (!strata::k31::situ_launch(d.g, d.u, d.o, d.n, beta, lbeta, true, nullptr)) { std::printf("  launch failed\n"); return 1; }
    dev_q = d.get();
    strata::k31::situ_cpu(g.data(), u.data(), cpu_p.data(), (int64_t) g.size(), beta, lbeta, false);
    strata::k31::situ_cpu(g.data(), u.data(), cpu_q.data(), (int64_t) g.size(), beta, lbeta, true);
    const struct { const char* form; const std::vector<float>* v; } forms[] = {
        {"device precise", &dev_p}, {"device portable", &dev_q}, {"host precise", &cpu_p}, {"host portable", &cpu_q}};
    for (const auto& f : forms) {
        const Stats s = gate(g, ref, *f.v);
        const bool ok = s.bad == 0;
        fails += !ok;
        std::printf("  %-10s %-16s n %7ld  worst %6.2f ULP  under-abs %.2e  bad %ld  %s\n", name, f.form, s.n, s.worst_ulp,
                    s.worst_abs_under, s.bad, ok ? "ok" : "FAIL");
    }
    long diff = 0;   // bit for bit, except that both-NaN counts as equal (the payloads are not part of the contract)
    for (size_t i = 0; i < dev_q.size(); ++i) diff += std::memcmp(&dev_q[i], &cpu_q[i], 4) != 0 && !(dev_q[i] != dev_q[i] && cpu_q[i] != cpu_q[i]);
    std::printf("  %-10s portable device == host bit for bit (non-NaN): %ld differ  %s\n", name, diff, diff ? "FAIL" : "ok");
    fails += diff != 0;
    return fails;
}

int run_wrong(const std::vector<float>& g, const std::vector<float>& u, const std::vector<double>& ref) {
    int fails = 0;
    Dev d(g, u);
    for (int w = 0; w < kNWrong; ++w) {
        wrong_kernel<<<(unsigned) ((d.n + 255) / 256), 256>>>(d.g, d.u, d.o, d.n, w);
        const Stats s = gate(g, ref, d.get());
        const double margin = s.worst_ulp / g_budget_ulp;
        const bool caught = s.bad > 0 && (margin >= 10.0 || kBuildRule[w]);
        fails += !caught;
        std::printf("  wrong reading %-14s bad %7ld  worst %.3g ULP (%.1fx the budget)  %s\n", kWrongName[w], s.bad, s.worst_ulp, margin,
                    !caught ? (s.bad > 0 ? "UNDER 10x" : "NOT CAUGHT") : kBuildRule[w] ? "fails (a build rule: guarded in CMake, under 10x)" : "caught");
    }
    return fails;
}

void push_sweep(std::vector<float>& g, std::vector<float>& u, bool gate_side) {
    auto add = [&](double x) { (gate_side ? g : u).push_back((float) x); (gate_side ? u : g).push_back(1.0f); };
    for (int i = 0; i <= 8000; ++i) add(gate_side ? -100.0 + 200.0 * i / 8000 : -400.0 + 800.0 * i / 8000);
    for (int i = 0; i <= 4800; ++i) add(gate_side ? -12.0 + 24.0 * i / 4800 : -30.0 + 60.0 * i / 4800);
    for (int i = 0; i < 600; ++i) {
        const double m = std::pow(10.0, -30.0 + 34.0 * i / 599);
        add(m);
        add(-m);
    }
}

int selftest() {
    std::vector<float> g, u;
    push_sweep(g, u, true);
    push_sweep(g, u, false);
    // random pairs over 1e-30 .. 1e4, both signs (a fixed LCG, so the test is reproducible)
    uint64_t s = 0x9E3779B97F4A7C15ull;
    auto rnd = [&]() { s = s * 6364136223846793005ull + 1442695040888963407ull; return (double) (s >> 11) / 9007199254740992.0; };
    for (int i = 0; i < 20000; ++i) {
        const double mg = std::pow(10.0, -30 + 34 * rnd()), mu = std::pow(10.0, -30 + 34 * rnd());
        g.push_back((float) (rnd() < 0.5 ? -mg : mg));
        u.push_back((float) (rnd() < 0.5 ? -mu : mu));
    }
    const float edges[] = {0.0f, -0.0f, 1e-45f, -1e-45f, 1e-39f, -1e-39f, 1.1754944e-38f, 40.0f, -40.0f, 87.0f, -87.0f,
                           88.7f, -88.7f, 89.0f, -89.0f, 103.0f, -103.0f, 250.0f, -250.0f, 1e30f, -1e30f, 3.4e38f, -3.4e38f};
    for (float a : edges)
        for (float b : edges) { g.push_back(a); u.push_back(b); }
    const float inf = std::numeric_limits<float>::infinity();
    const float ig[] = {inf, -inf, inf, -inf, 1.0f, 1.0f, inf, -3.0f}, iu[] = {1.0f, 1.0f, -2.0f, 5.0f, inf, -inf, inf, -inf};
    for (int i = 0; i < 8; ++i) { g.push_back(ig[i]); u.push_back(iu[i]); }
    // the review's adversarial worst pairs (6.9 ULPs, precise and portable) and a u/25 that is denormal
    const float ag[] = {0.00244858535f, 0.00428816676f, 2.86e35f}, au[] = {3.19202375f, 15.8594198f, 2.94e-37f};
    for (int i = 0; i < 3; ++i) { g.push_back(ag[i]); u.push_back(au[i]); }
    std::vector<double> ref(g.size());
    for (size_t i = 0; i < g.size(); ++i) ref[i] = ref64(g[i], u[i]);
    std::printf("k31_situ_test --selftest: %zu inputs, budget %.0f ULP (g >= %.0f), abs %.0e below\n", g.size(), g_budget_ulp,
                g_under_g, g_under_abs);
    int fails = run_case("selftest", g, u, ref);
    fails += run_wrong(g, u, ref);
    // other betas: a linear up branch (lbeta 0, llama.cpp's convention), and two non-released configurations
    const float cfg[3][2] = {{4.0f, 0.0f}, {1.0f, 1.0f}, {3.0f, 7.0f}};
    for (const auto& c : cfg) {
        // a linear up branch leaves |b| = |u| unbounded, and a denormal (or underflowed) gate factor's absolute error
        // times u = 1e30 .. 3.4e38 is one no float32 kernel can keep within the float64 reference (the 20 edge pairs
        // g in {denormal, < -87} x u in {1e30, 3.4e38}); the released lbeta 25 caps |b| at 25, so the linear
        // configuration is gated for |u| <= 1e4
        std::vector<float> gc, uc;
        for (size_t i = 0; i < g.size(); ++i)
            if (c[1] > 0.0f || !(std::fabs(u[i]) > 1e4f)) { gc.push_back(g[i]); uc.push_back(u[i]); }
        std::vector<double> rc(gc.size());
        for (size_t i = 0; i < gc.size(); ++i) rc[i] = ref64(gc[i], uc[i], c[0], c[1]);
        char nm[32];
        std::snprintf(nm, sizeof nm, "b%g/l%g", c[0], c[1]);
        if (gc.size() != g.size()) std::printf("  %-10s %zu of %zu inputs (|u| <= 1e4: the linear up branch)\n", nm, gc.size(), g.size());
        fails += run_case(nm, gc, uc, rc, c[0], c[1]);
    }
    // the activation's own invariants: situ(0, u) == 0, odd in u, saturation 4 * 25 = 100 at g, u -> +inf
    float o[3];
    const float tg[3] = {0.0f, 50.0f, 50.0f}, tu[3] = {3.0f, 400.0f, -400.0f};
    strata::k31::situ_cpu(tg, tu, o, 3, kSituBeta, kSituLinearBeta, false);
    const bool inv = o[0] == 0.0f && o[1] == -o[2] && std::fabs(o[1] - 100.0f) < 1e-4f;
    std::printf("  invariants (situ(0,u)=0, odd in u, saturation 100): %s\n", inv ? "ok" : "FAIL");
    fails += !inv;
    return fails;
}

// ---- golden files
bool read_floats(const std::string& path, size_t n, std::vector<float>& v) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    v.resize(n);
    f.read(reinterpret_cast<char*>(v.data()), (std::streamsize) (n * 4));
    return (size_t) f.gcount() == n * 4;
}

// `key`'s number among the section object's own keys (not a nested object's, not a later section's)
double json_number_after(const std::string& js, const std::string& section, const std::string& key, double dflt) {
    const size_t a0 = js.find("\"" + section + "\"");
    const size_t a = a0 == std::string::npos ? a0 : js.find('{', a0);
    if (a == std::string::npos) return dflt;
    int depth = 0;
    bool str = false;
    size_t key_at = std::string::npos;
    const std::string pat = "\"" + key + "\"";
    for (size_t i = a; i < js.size(); ++i) {
        const char ch = js[i];
        if (str) { if (ch == '\\') ++i; else if (ch == '"') str = false; continue; }
        if (ch == '"') {
            if (depth == 1 && js.compare(i, pat.size(), pat) == 0) { key_at = i; break; }
            str = true;
        } else if (ch == '{') ++depth;
        else if (ch == '}' && --depth == 0) break;
    }
    if (key_at == std::string::npos) return dflt;
    const size_t c = js.find(':', key_at);
    return c == std::string::npos ? dflt : std::strtod(js.c_str() + c + 1, nullptr);
}

int golden(const std::string& dir) {
    std::ifstream mf(dir + "/manifest.txt");
    if (!mf) { std::printf("%s/manifest.txt: missing (python tools/kimi_k31_golden.py --out %s)  FAIL\n", dir.c_str(), dir.c_str()); return 1; }
    std::ifstream jf(dir + "/golden.json");
    std::stringstream jss;
    jss << jf.rdbuf();
    const std::string js = jss.str();
    g_budget_ulp = json_number_after(js, "situ", "budget_ulp", g_budget_ulp);
    g_under_g = json_number_after(js, "situ", "underflow_g", g_under_g);
    g_under_abs = json_number_after(js, "situ", "underflow_abs", g_under_abs);
    std::vector<std::pair<std::string, size_t>> entries;
    std::string line;
    while (std::getline(mf, line)) {
        std::istringstream ls(line);
        std::string name;
        ls >> name;
        size_t n = 1, d;
        while (ls >> d) n *= d;
        entries.push_back({name, n});
    }
    std::printf("k31_situ_test --golden %s: budget %.0f ULP (g >= %.0f), abs %.0e below\n", dir.c_str(), g_budget_ulp, g_under_g, g_under_abs);
    int fails = 0, cases = 0;
    std::vector<float> all_g, all_u;
    std::vector<double> all_ref;
    for (const auto& [name, n] : entries) {
        const std::string suf = ".g";
        if (name.rfind("situ.", 0) != 0 || name.size() < suf.size() || name.compare(name.size() - 2, 2, suf) != 0) continue;
        const std::string base = name.substr(0, name.size() - 2);
        std::vector<float> g, u, hi, lo;
        if (!read_floats(dir + "/" + base + ".g.f32", n, g) || !read_floats(dir + "/" + base + ".u.f32", n, u) ||
            !read_floats(dir + "/" + base + ".ref_hi.f32", n, hi) || !read_floats(dir + "/" + base + ".ref_lo.f32", n, lo)) {
            std::printf("  %s: unreadable  FAIL\n", base.c_str());
            ++fails;
            continue;
        }
        std::vector<double> ref(n);
        for (size_t i = 0; i < n; ++i) ref[i] = std::isfinite(hi[i]) ? (double) hi[i] + (double) lo[i] : (double) hi[i];
        fails += run_case(base.c_str() + 5, g, u, ref);
        all_g.insert(all_g.end(), g.begin(), g.end());
        all_u.insert(all_u.end(), u.begin(), u.end());
        all_ref.insert(all_ref.end(), ref.begin(), ref.end());
        ++cases;
    }
    if (cases == 0) { std::printf("  no situ.* cases in %s  FAIL\n", dir.c_str()); return 1; }
    fails += run_wrong(all_g, all_u, all_ref);
    return fails;
}

}  // namespace

int main(int argc, char** argv) {
    int fails = 0;
    bool any = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") { fails += selftest(); any = true; }
        else if (a == "--golden" && i + 1 < argc) { fails += golden(argv[++i]); any = true; }
        else { std::fprintf(stderr, "usage: %s [--selftest] [--golden DIR]\n", argv[0]); return 2; }
    }
    if (!any) fails += selftest();
    std::printf("k31_situ_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
