// tests/k31_attn_res_test.cu - the Block Attention Residuals mix (src/k31/k31_attn_res.hpp) held to ref/kimi_k31.py
// and to a float64 reference, with the runner's block bookkeeping, plus the mutations, each of which must fail.
//
//   k31_attn_res_test --selftest      n_embd 7168, 1..16 candidates, float64 reference
//   k31_attn_res_test --dumps DIR     every site of the fixture (python tools/kimi_k31_attnres_dump.py --out DIR)
//
// --dumps rebuilds each site's candidates from the per-layer states with the block rule the runner must follow.
// A block is pushed at every layer j with j % B == 0 (the embeddings at j = 0, else l_out of layer j-1), right
// after layer j's attention mix: so the attention site of layer L sees the pushes j < L plus l_out-(L-1), the FFN
// site of L the pushes j <= L plus prefix_sum_attn-L, and the output site every push plus the last l_out.  Errors: the mix in
// float32 ULPs of the reference (unit floor 2^-126) and as max |d| / rms(ref); the probabilities as max |d|.
// Bounds: the measured gap with a margin; every bound at least 10x below the smallest mutation's effect:
// bf16 state, a wrong block boundary (pushes at j % B == 1), a wrong block size (B + 1), a dropped partial sum,
// the mix taken over the NORMALISED candidates, scores without the RMS normalisation.
#include "../src/k31/k31_attn_res.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr double kMinNormal = 1.17549435082228751e-38;
constexpr double kBoundUlp = 2.0;     // the mix (measured: see the printout)
constexpr double kBoundRel = 1e-6;    // the mix, as max |d| / rms
constexpr double kBoundProb = 1e-6;   // the probabilities, absolute

void cuda_ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
double ulp_unit(double r) {
    const double ar = std::fabs(r);
    if (ar < kMinNormal) return kMinNormal;
    const float f = (float) ar;
    return std::max((double) std::nextafter(f, std::numeric_limits<float>::infinity()) - (double) f, kMinNormal);
}
double max_ulp(const std::vector<float>& a, const std::vector<float>& r) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs((double) a[i] - (double) r[i]) / ulp_unit(r[i]));
    return m;
}
double rel_err(const std::vector<float>& a, const std::vector<float>& r) {
    double num = 0, ss = 0;
    for (size_t i = 0; i < a.size(); ++i) { num = std::max(num, std::fabs((double) a[i] - (double) r[i])); ss += (double) r[i] * r[i]; }
    return num / std::max(std::sqrt(ss / std::max<size_t>(r.size(), 1)), 1e-30);
}
double max_abs(const std::vector<float>& a, const std::vector<float>& r) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs((double) a[i] - (double) r[i]));
    return m;
}
float to_bf16(float x) {   // round to nearest even, as a bf16 state store would
    uint32_t b;
    std::memcpy(&b, &x, 4);
    b = (b + 0x7fffu + ((b >> 16) & 1u)) & 0xffff0000u;
    float y;
    std::memcpy(&y, &b, 4);
    return y;
}

using Vec = std::vector<float>;

// the float64 reference (ref/kimi_k31.py res_mix), with the two misreadings the study warns about
enum Read { kRight, kMixNormed, kUnnormedScore };
void reference(const std::vector<const Vec*>& c, const Vec& w, float eps, int T, int n, Read rd, Vec& out, Vec& probs) {
    const int nc = (int) c.size();
    out.assign((size_t) T * n, 0);
    probs.assign((size_t) T * nc, 0);
    for (int t = 0; t < T; ++t) {
        std::vector<double> s(nc), den(nc);
        double mx = -1e300;
        for (int i = 0; i < nc; ++i) {
            double d = 0, q = 0;
            for (int j = 0; j < n; ++j) { const double v = (*c[i])[(size_t) t * n + j]; d += (double) w[j] * v; q += v * v; }
            den[i] = std::sqrt(q / n + (double) eps);
            s[i] = rd == kUnnormedScore ? d : d / den[i];
            mx = std::max(mx, s[i]);
        }
        double tot = 0;
        for (int i = 0; i < nc; ++i) { s[i] = std::exp(s[i] - mx); tot += s[i]; }
        for (int i = 0; i < nc; ++i) { s[i] /= tot; probs[(size_t) t * nc + i] = (float) s[i]; }
        for (int j = 0; j < n; ++j) {
            double acc = 0;
            for (int i = 0; i < nc; ++i) acc += s[i] * (double) (*c[i])[(size_t) t * n + j] / (rd == kMixNormed ? den[i] : 1.0);
            out[(size_t) t * n + j] = (float) acc;
        }
    }
}

void run_kernel(const std::vector<const Vec*>& c, const Vec& w, float eps, int T, int n, Vec& out, Vec& probs) {
    strata::k31::AttnResArgs a;
    std::vector<float*> dc;
    for (const Vec* v : c) {
        float* d = nullptr;
        cuda_ok(cudaMalloc(&d, v->size() * 4), "malloc");
        cuda_ok(cudaMemcpy(d, v->data(), v->size() * 4, cudaMemcpyHostToDevice), "h2d");
        dc.push_back(d);
    }
    float *dw = nullptr, *dout = nullptr, *dp = nullptr;
    cuda_ok(cudaMalloc(&dw, w.size() * 4), "malloc");
    cuda_ok(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice), "h2d");
    cuda_ok(cudaMalloc(&dout, (size_t) T * n * 4), "malloc");
    cuda_ok(cudaMalloc(&dp, (size_t) T * c.size() * 4), "malloc");
    for (size_t i = 0; i < c.size(); ++i) a.cand[i] = dc[i];
    a.n_cand = (int) c.size(); a.n_tokens = T; a.n_embd = n; a.score_w = dw; a.eps = eps; a.out = dout; a.probs = dp;
    if (!strata::k31::attn_res_launch(a, nullptr)) { std::fprintf(stderr, "attn_res refused\n"); std::exit(2); }
    cuda_ok(cudaDeviceSynchronize(), "sync");
    out.resize((size_t) T * n);
    probs.resize((size_t) T * c.size());
    cuda_ok(cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost), "d2h");
    cuda_ok(cudaMemcpy(probs.data(), dp, probs.size() * 4, cudaMemcpyDeviceToHost), "d2h");
    for (float* d : dc) cudaFree(d);
    cudaFree(dw); cudaFree(dout); cudaFree(dp);
}

struct Worst { double ulp = 0, rel = 0, prob = 0; };

bool gate(const char* name, const Vec& out, const Vec& probs, const Vec& rout, const Vec& rprobs, Worst& w) {
    const double u = max_ulp(out, rout), r = rel_err(out, rout), p = rprobs.size() == probs.size() ? max_abs(probs, rprobs) : 1.0;
    w.ulp = std::max(w.ulp, u); w.rel = std::max(w.rel, r); w.prob = std::max(w.prob, p);
    const bool ok = u <= kBoundUlp && r <= kBoundRel && p <= kBoundProb;
    if (!ok) std::printf("    %s: mix %.2f ULP %.2e  probs %.2e  FAIL\n", name, u, r, p);
    return ok;
}

// the mutation's effect on the mix, relative (it must clear 10x the bounds)
double effect(const Vec& out, const Vec& rout) { return rel_err(out, rout); }
const double kNeed = 10.0 * std::max(kBoundRel, kBoundUlp * 1.2e-7);

uint64_t g_s = 0xD1B54A32D192ED03ull;
double urand() { g_s = g_s * 6364136223846793005ull + 1442695040888963407ull; return (double) (g_s >> 11) / 9007199254740992.0; }

int selftest() {
    std::printf("k31_attn_res_test --selftest\n");
    int fails = 0;
    const int T = 4, n = 7168;
    Worst w;
    Vec score(n);
    for (auto& x : score) x = (float) ((urand() - 0.5) * 0.2);
    double min_eff[4] = {1e300, 1e300, 1e300, 1e300};   // bf16, dropped partial, normed mix, unnormed score
    for (int nc : {1, 2, 5, 9, 16}) {
        std::vector<Vec> store(nc, Vec((size_t) T * n));
        for (int i = 0; i < nc; ++i) {
            const double sc = 1.0 + 6.0 * i;   // deeper blocks grow
            for (auto& x : store[i]) x = (float) ((urand() - 0.5) * 2 * sc);
        }
        std::vector<const Vec*> c;
        for (auto& v : store) c.push_back(&v);
        Vec out, probs, rout, rprobs;
        run_kernel(c, score, 1e-5f, T, n, out, probs);
        reference(c, score, 1e-5f, T, n, kRight, rout, rprobs);
        std::string nm = "n_cand " + std::to_string(nc);
        bool ok = gate(nm.c_str(), out, probs, rout, rprobs, w);
        if (nc == 1) ok = ok && std::memcmp(out.data(), store[0].data(), out.size() * 4) == 0;   // one candidate: itself
        fails += !ok;
        if (nc >= 2) {
            std::vector<Vec> bf(store);
            for (auto& v : bf) for (auto& x : v) x = to_bf16(x);
            std::vector<const Vec*> cb;
            for (auto& v : bf) cb.push_back(&v);
            Vec o2, p2;
            run_kernel(cb, score, 1e-5f, T, n, o2, p2);
            min_eff[0] = std::min(min_eff[0], effect(o2, rout));
            std::vector<const Vec*> cd(c.begin(), c.end() - 1);
            run_kernel(cd, score, 1e-5f, T, n, o2, p2);
            min_eff[1] = std::min(min_eff[1], effect(o2, rout));
            reference(c, score, 1e-5f, T, n, kMixNormed, o2, p2);
            min_eff[2] = std::min(min_eff[2], effect(o2, rout));
            reference(c, score, 1e-5f, T, n, kUnnormedScore, o2, p2);
            min_eff[3] = std::min(min_eff[3], effect(o2, rout));
        }
    }
    std::printf("  worst: mix %.2f ULP, %.2e; probs %.2e (bounds %.0f ULP, %.0e, %.0e)  %s\n", w.ulp, w.rel, w.prob, kBoundUlp,
                kBoundRel, kBoundProb, fails ? "FAIL" : "ok");
    const char* names[4] = {"bf16 state", "dropped partial", "mix of normalised candidates", "unnormalised scores"};
    for (int i = 0; i < 4; ++i) {
        const bool caught = min_eff[i] >= kNeed;
        std::printf("  mutation %-30s smallest effect %.2e (needs >= %.1e)  %s\n", names[i], min_eff[i], kNeed, caught ? "caught" : "NOT CAUGHT");
        fails += !caught;
    }
    // refusals
    strata::k31::AttnResArgs bad;
    float* d = nullptr;
    cuda_ok(cudaMalloc(&d, 64), "m");
    bad.cand[0] = d; bad.score_w = d; bad.out = d; bad.n_tokens = 1; bad.n_embd = 4;
    int refused = 0;
    bad.n_cand = 0; refused += !strata::k31::attn_res_launch(bad, nullptr);
    bad.n_cand = 17; refused += !strata::k31::attn_res_launch(bad, nullptr);
    bad.n_cand = 2; refused += !strata::k31::attn_res_launch(bad, nullptr);   // cand[1] missing
    cudaFree(d);
    std::printf("  refusals (0 candidates, 17, a missing candidate): %d/3  %s\n", refused, refused == 3 ? "ok" : "FAIL");
    fails += refused != 3;
    return fails;
}

bool read_raw(const std::string& path, size_t n, Vec& v) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    v.resize(n);
    f.read(reinterpret_cast<char*>(v.data()), (std::streamsize) (n * 4));
    return (size_t) f.gcount() == n * 4;
}

int dumps(const std::string& dir) {
    std::stringstream jss;
    jss << std::ifstream(dir + "/attnres.json").rdbuf();
    const std::string js = jss.str();
    if (js.empty()) { std::printf("%s/attnres.json: missing (python tools/kimi_k31_attnres_dump.py --out %s)  FAIL\n", dir.c_str(), dir.c_str()); return 1; }
    auto num = [&](const char* key) { const size_t p = js.find(std::string("\"") + key + "\""); return p == std::string::npos ? 0.0 : std::strtod(js.c_str() + js.find(':', p) + 1, nullptr); };
    const int L = (int) num("n_layers"), B = (int) num("block_size"), T = (int) num("T"), n = (int) num("n_embd");
    const float eps = (float) num("eps");
    std::printf("k31_attn_res_test --dumps %s: %d layers, block %d, T %d, n_embd %d, eps %.9g\n", dir.c_str(), L, B, T, n, eps);
    std::map<std::string, Vec> st;
    auto get = [&](const std::string& name, size_t cnt) -> const Vec& {
        auto it = st.find(name);
        if (it != st.end()) return it->second;
        Vec v;
        if (!read_raw(dir + "/" + name + ".f32", cnt, v)) { std::printf("  %s: unreadable  FAIL\n", name.c_str()); std::exit(1); }
        return st.emplace(name, std::move(v)).first->second;
    };
    const size_t tn = (size_t) T * n;
    // the runner's block rule, with the mutations' parameters
    auto cands = [&](const std::string& site, int il, int bsize, int boff, bool drop) {
        std::vector<const Vec*> c;
        const int upto = site == "attn" ? il : site == "ffn" ? il + 1 : L;
        for (int j = 0; j < upto; ++j)
            if (j % bsize == boff) c.push_back(j == 0 ? &get("ar.inp_embd", tn) : &get("ar." + std::to_string(j - 1) + ".l_out", tn));
        if (!drop) {
            if (site == "attn") c.push_back(&get("ar." + std::to_string(il - 1) + ".l_out", tn));
            else if (site == "ffn") c.push_back(&get("ar." + std::to_string(il) + ".prefix_attn", tn));
            else c.push_back(&get("ar." + std::to_string(L - 1) + ".l_out", tn));
        }
        return c;
    };
    int fails = 0, sites = 0;
    Worst w;
    double min_eff[6];
    for (double& e : min_eff) e = 1e300;
    const char* mnames[6] = {"bf16 state", "wrong block boundary (j % B == 1)", "wrong block size (B + 1)", "dropped partial",
                             "mix of normalised candidates", "unnormalised scores"};
    std::vector<std::pair<std::string, int>> list;
    for (int il = 0; il < L; ++il) {
        if (il >= 1) list.push_back({"attn", il});
        list.push_back({"ffn", il});
    }
    list.push_back({"out", -1});
    for (const auto& [site, il] : list) {
        const std::string pre = site == "out" ? std::string("ar.") : "ar." + std::to_string(il) + ".";
        const Vec& score = get(site == "out" ? "ar.score_out" : pre + "score_" + site, n);
        const std::vector<const Vec*> c = cands(site, il, B, 0, false);
        const int nc = (int) c.size();
        const Vec& rmix = get(site == "out" ? "ar.mix_out" : pre + "mix_" + site, tn);
        const Vec& rprob = get(site == "out" ? "ar.probs_out" : pre + "probs_" + site, (size_t) T * nc);
        Vec out, probs;
        run_kernel(c, score, eps, T, n, out, probs);
        const std::string nm = site + (il >= 0 ? " " + std::to_string(il) : "") + " (" + std::to_string(nc) + " candidates)";
        fails += !gate(nm.c_str(), out, probs, rmix, rprob, w);
        ++sites;
        // mutations
        Vec o2, p2;
        {
            std::vector<Vec> bf;
            for (const Vec* v : c) { bf.push_back(*v); for (auto& x : bf.back()) x = to_bf16(x); }
            std::vector<const Vec*> cb;
            for (auto& v : bf) cb.push_back(&v);
            run_kernel(cb, score, eps, T, n, o2, p2);
            min_eff[0] = std::min(min_eff[0], effect(o2, rmix));
        }
        if (L > 1) {
            const auto c1 = cands(site, il, B, 1, false);   // boundaries one layer late
            if (c1.size() >= 1) { run_kernel(c1, score, eps, T, n, o2, p2); if (c1 != c) min_eff[1] = std::min(min_eff[1], effect(o2, rmix)); }
            const auto c2 = cands(site, il, B + 1, 0, false);
            if (c2 != c) { run_kernel(c2, score, eps, T, n, o2, p2); min_eff[2] = std::min(min_eff[2], effect(o2, rmix)); }
        }
        if (nc >= 2) {
            const auto c3 = cands(site, il, B, 0, true);
            run_kernel(c3, score, eps, T, n, o2, p2);
            min_eff[3] = std::min(min_eff[3], effect(o2, rmix));
            reference(c, score, eps, T, n, kMixNormed, o2, p2);
            min_eff[4] = std::min(min_eff[4], effect(o2, rmix));
            reference(c, score, eps, T, n, kUnnormedScore, o2, p2);
            min_eff[5] = std::min(min_eff[5], effect(o2, rmix));
        }
    }
    std::printf("  %d sites: worst mix %.2f ULP, %.2e; probs %.2e (bounds %.0f ULP, %.0e, %.0e)  %s\n", sites, w.ulp, w.rel, w.prob, kBoundUlp,
                kBoundRel, kBoundProb, fails ? "FAIL" : "ok");
    for (int i = 0; i < 6; ++i) {
        const bool caught = min_eff[i] >= kNeed && min_eff[i] < 1e299;
        std::printf("  mutation %-36s smallest effect over the sites it changes %.2e (needs >= %.1e)  %s\n", mnames[i], min_eff[i], kNeed,
                    caught ? "caught" : "NOT CAUGHT");
        fails += !caught;
    }
    return fails;
}

}  // namespace

int main(int argc, char** argv) {
    int fails = 0;
    bool any = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") { fails += selftest(); any = true; }
        else if (a == "--dumps" && i + 1 < argc) { fails += dumps(argv[++i]); any = true; }
        else { std::fprintf(stderr, "usage: %s [--selftest] [--dumps DIR]\n", argv[0]); return 2; }
    }
    if (!any) fails += selftest();
    std::printf("k31_attn_res_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
