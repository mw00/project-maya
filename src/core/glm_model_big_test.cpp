// src/core/glm_model_big_test.cpp - the runner against the 45-layer big fixture (seed 777).
//
// The seed-1234 test proves the math; this one proves the SHAPE: 45 layers in the released
// cadence (a DSA layer at il%4==3, the last DSA at 43), 512-token sequences that cross 128
// k-pools and exercise the top-k discard path, per-layer arrays of odd length, sinkhorn
// iterations the small fixture never sees, and magnitudes hot enough that the SwiGLU clamps
// fire inside the runner too.  Four gates:
//
//   A. STREAMING VS BATCHED at three prefixes (100/250/511): the per-token streamed logits of
//      the final position must match one batched forward of the whole prefix.
//   B. STATE DETERMINISM: A, reset, B, reset, A - the second A is bitwise the first.
//   C. VS THE F64 ORACLE over all 512 columns: the acceptance rule calibrated in
//      docs/GLM5-FLASH.md §8.2 - argmax agreement (relaxed: MoE selection near-ties flip on
//      f32-vs-f64 rounding), p99 of the per-column KL below 1e-4, and max KL below 0.1.
//   D. NO NaN/INF anywhere in the streamed logits (the hot scales are the point).
//
// Fixtures: the GGUF is generated into the build tree by glm_model_big_gen; the expectation
// (vocab, T) float32 dump is committed at data/glm5-big-dumps/.
#include "maya/core/glm_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

void require(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "glm_model_big_test: %s\n", what.c_str());
        std::exit(1);
    }
}

size_t argmax(const std::vector<float>& v) {
    size_t am = 0;
    for (size_t i = 1; i < v.size(); ++i)
        if (v[i] > v[am]) am = i;
    return am;
}

double max_abs(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) m = std::max(m, (double) std::fabs(a[i] - b[i]));
    return m;
}

double kl(const std::vector<float>& a, const std::vector<float>& b) {
    double mx = *std::max_element(a.begin(), a.end());
    double mx2 = *std::max_element(b.begin(), b.end());
    double za = 0.0, zb = 0.0;
    for (float x : a) za += std::exp((double) x - mx);
    for (float x : b) zb += std::exp((double) x - mx2);
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double pa = std::exp((double) a[i] - mx) / za;
        const double pb = std::exp((double) b[i] - mx2) / zb;
        if (pa > 1e-300) s += pa * (std::log(pa) - std::log(pb));
    }
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string gguf = argc > 1 ? argv[1] : "glm5-big-fixture.gguf";
    const std::string expect = argc > 2 ? argv[2] : "data/glm5-big-dumps/result_output.f32";
    const char* dump_out = argc > 3 ? argv[3] : nullptr;   // optional: write the streamed logits
    constexpr size_t T = 512;

    maya::core::Glm5Model model;
    std::string err;
    if (!model.load(gguf, 1024, err)) {
        // stage the failure: parse the GGUF and the geometry by hand before giving up
        try {
            maya::GgufFile gg(gguf);
            std::fprintf(stderr, "gguf ok: %zu tensors, version %u\n", gg.tensors().size(), gg.version());
            maya::core::Glm5Geometry geo;
            std::string gerr;
            const bool pok = maya::core::Glm5Geometry::from_gguf(gg, geo, gerr);
            std::fprintf(stderr, "geometry: ok=%d err=[%s] layers=%d n_embd=%d kv_lora=%d kpool=%d topk=%d tail=%d\n",
                         (int) pok, gerr.c_str(), geo.n_layers, geo.n_embd, geo.kv_lora, geo.idx_kpool,
                         geo.idx_top_k, geo.idx_select_tail);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "gguf open threw: %s\n", e.what());
        }
        require(false, "load: " + err);
    }
    const size_t V = (size_t) model.geometry().n_vocab;
    require(model.geometry().n_layers == 45, "the big fixture has 45 layers");

    std::vector<int32_t> tokens;
    for (size_t i = 0; i < T; ++i) tokens.push_back((int32_t) ((i % 511) + 1));

    // ---- stream all 512 tokens, collecting every column
    std::vector<float> streamed((size_t) V * T);
    for (size_t t = 0; t < T; ++t) {
        std::vector<float> lg;
        require(model.forward({tokens[t]}, lg, err), "streamed forward: " + err);
        require(lg.size() == V, "logit width");
        for (size_t v = 0; v < V; ++v) {
            require(std::isfinite(lg[v]), "non-finite logit at token " + std::to_string(t));
            streamed[v + V * t] = lg[v];
        }
    }

    if (dump_out) {
        FILE* f = std::fopen(dump_out, "wb");
        require(f != nullptr, "cannot write the streamed dump");
        std::fwrite(streamed.data(), sizeof(float), streamed.size(), f);
        std::fclose(f);
    }

    // ---- A. streaming vs batched at three prefixes
    double worst_sb = 0.0;
    for (size_t len : {size_t(100), size_t(250), size_t(511)}) {
        model.reset();
        std::vector<float> batched;
        require(model.forward(std::vector<int32_t>(tokens.begin(), tokens.begin() + (long) len), batched, err),
                "batched forward: " + err);
        const double d = max_abs(batched, std::vector<float>(streamed.begin() + (long) V * (len - 1),
                                                             streamed.begin() + (long) V * len));
        worst_sb = std::max(worst_sb, d);
    }
    require(worst_sb < 5e-5, "streaming vs batched diverges (" + std::to_string(worst_sb) + ")");

    // ---- B. state determinism at 128 tokens
    {
        std::vector<int32_t> a(tokens.begin(), tokens.begin() + 128);
        std::vector<int32_t> b(tokens.begin() + 200, tokens.begin() + 328);
        std::vector<float> a1, bb, a2;
        model.reset();   // gate A's batched forwards leave the state mid-sequence
        require(model.forward(a, a1, err), "A1: " + err);
        model.reset();
        require(model.forward(b, bb, err), "B: " + err);
        model.reset();
        require(model.forward(a, a2, err), "A2: " + err);
        require(a1 == a2, "state determinism: A-after-B differs bitwise");
    }

    // ---- C. vs the f64 oracle, with the calibrated acceptance rule
    std::vector<float> oracle((size_t) V * T);
    {
        FILE* f = std::fopen(expect.c_str(), "rb");
        require(f != nullptr, "cannot open " + expect);
        require(std::fread(oracle.data(), sizeof(float), oracle.size(), f) == oracle.size(),
                "short read on " + expect);
        std::fclose(f);
    }
    std::vector<double> kls(T);
    std::vector<double> maxd(T);
    size_t agree = 0;
    for (size_t t = 0; t < T; ++t) {
        std::vector<float> ours(streamed.begin() + (long) V * t, streamed.begin() + (long) V * (t + 1));
        std::vector<float> ref(oracle.begin() + (long) V * t, oracle.begin() + (long) V * (t + 1));
        kls[t] = kl(ours, ref);
        maxd[t] = max_abs(ours, ref);
        agree += argmax(ours) == argmax(ref);
    }
    std::vector<double> sorted_kl(kls), sorted_d(maxd);
    std::sort(sorted_kl.begin(), sorted_kl.end());
    std::sort(sorted_d.begin(), sorted_d.end());
    const double p99_kl = sorted_kl[(size_t)(T * 0.99)];
    const double med_d = sorted_d[T / 2];
    // report the worst columns so a failure localizes immediately
    std::vector<size_t> order(T);
    for (size_t t = 0; t < T; ++t) order[t] = t;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return kls[a] > kls[b]; });
    for (size_t i = 0; i < 8 && i < T; ++i)
        std::fprintf(stderr, "  col %3zu: KL %.3e  max|d| %.3e\n", order[i], kls[order[i]], maxd[order[i]]);
    require(p99_kl < 1e-4, "KL p99 too high (" + std::to_string(p99_kl) + ")");
    require(sorted_kl[T - 1] < 0.1, "max KL too high (" + std::to_string(sorted_kl[T - 1]) + ")");
    require(agree >= 505, "argmax agreement " + std::to_string(agree) + "/512 (calibrated: >= 505)");
    require(med_d < 2e-3, "median column max|diff| too high (" + std::to_string(med_d) + ")");

    size_t flips = 0;
    for (size_t t = 0; t < T; ++t)
        if (kls[t] > 1e-4) ++flips;
    std::printf("glm_model_big_test: PASS (stream-vs-batch %.1e, median|d| %.1e, KL p99 %.1e max %.1e, "
                "argmax %zu/512, %zu columns beyond KL 1e-4)\n",
                worst_sb, med_d, p99_kl, sorted_kl[T - 1], agree, flips);
    return 0;
}
