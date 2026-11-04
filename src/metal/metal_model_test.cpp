// src/metal/metal_model_test.cpp - maya-metal against the GLM oracles, with the gates the CUDA engine's tests hold
// (src/core/glm_model_test.cpp, src/core/glm_model_big_test.cpp):
//
//   metal_model_test small [GGUF] [DUMPS] [DUMPS12]   6-layer fixture vs ref/glm.py: batched logits within 2e-3,
//       every prefix's argmax, streamed = batched within 1e-5, 12-token parity and the greedy chain, the
//       prefill-to-decode continuation, state determinism (bitwise), mutation sensitivity
//   metal_model_test big GGUF [EXPECT]                45 layers, 512 tokens vs the f64 oracle: streamed vs batched
//       within 5e-5, state determinism, KL p99 < 1e-4, max KL < 0.1, argmax >= 505/512, no NaN/Inf
//   metal_model_test mtp GGUF [N]                     MTP drafting (tools/glm_mtp_fixture.py's GGUF): N greedy tokens
//       through NextN drafts and verify windows must equal plain decoding token for token
//   metal_model_test dflash GGUF DRAFTER [N]          DFlash2 drafting (tools/glm_dflash_fixture.py's drafter for the
//       45-layer fixture): N greedy tokens through drafts and verify windows must equal plain decoding
//   metal_model_test prefill GGUF [T]                 a T-token prompt in one slice (quantized experts take ggml-metal's
//       matrix-matrix kernel there) against the same tokens fed one by one: the last logits' argmax equal, KL < 1e-3
//   metal_model_test window GGUF [N]                  verify windows (speculative decoding's): N greedy tokens through
//       windows of 1-8 rows whose drafts are partly wrong must equal plain decoding token for token, every
//       accepted row's logits within 1e-5 of the plain step's
//
// Settings (environment): MAYA_METAL_CPU=1 runs on ggml-cpu, MAYA_METAL_POOL_SLOTS=N shrinks the expert pool (the
// eviction path), MAYA_METAL_UBATCH / MAYA_METAL_ATTN_CHUNK change the slicing, MAYA_METAL_KV_F16=1 / MAYA_METAL_KV_F32=1 the latent cache.
#include "glm_metal.hpp"
#include "spec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using maya::metal::GlmMetal;

void require(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "metal_model_test: %s\n", what.c_str());
        std::exit(1);
    }
}

int env(const char* k) {
    const char* v = std::getenv(k);
    return v && *v ? std::atoi(v) : 0;
}

void load(GlmMetal& m, const std::string& gguf, int64_t ctx, int max_window = 0, bool mtp = false,
          const std::string& dflash = "", bool dsa_matvec = false) {
    maya::metal::GlmOptions o;
    o.max_window = max_window;
    o.mtp = mtp;
    o.dflash = dflash;
    o.use_gpu = env("MAYA_METAL_CPU") == 0;
    o.pool_slots = env("MAYA_METAL_POOL_SLOTS");
    if (env("MAYA_METAL_UBATCH") > 0) o.n_ubatch = env("MAYA_METAL_UBATCH");
    if (env("MAYA_METAL_ATTN_CHUNK") > 0) o.attn_chunk = env("MAYA_METAL_ATTN_CHUNK");
    o.kv_f16 = env("MAYA_METAL_KV_F16") != 0;
    o.kv_f32 = env("MAYA_METAL_KV_F32") != 0;
    if (const char* v = std::getenv("MAYA_METAL_DSA_MM"); v && *v) o.dsa_mm = std::atoi(v) != 0;
    if (const char* v = std::getenv("MAYA_METAL_DSA_KERNEL"); v && *v) o.dsa_kernel = std::atoi(v) != 0;
    if (dsa_matvec) o.dsa_mm = false;
    m.set_options(o);
    std::string err;
    require(m.load(gguf, ctx, err), "load " + gguf + ": " + err);
}

std::vector<float> fwd(GlmMetal& m, const std::vector<int32_t>& toks) {
    std::vector<float> lg;
    std::string err;
    require(m.forward(toks, lg, err), "forward: " + err);
    return lg;
}

std::vector<float> read_f32(const std::string& path, size_t n) {
    std::vector<float> v(n);
    FILE* f = std::fopen(path.c_str(), "rb");
    require(f != nullptr, "cannot open " + path);
    require(std::fread(v.data(), sizeof(float), n, f) == n, "short read on " + path);
    std::fclose(f);
    return v;
}

size_t argmax(const float* v, size_t n) {
    size_t am = 0;
    for (size_t i = 1; i < n; ++i)
        if (v[i] > v[am]) am = i;
    return am;
}

double max_abs(const float* a, const float* b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) m = std::max(m, (double) std::fabs(a[i] - b[i]));
    return m;
}

double kl(const float* a, const float* b, size_t n) {
    const double ma = *std::max_element(a, a + n), mb = *std::max_element(b, b + n);
    double za = 0.0, zb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        za += std::exp((double) a[i] - ma);
        zb += std::exp((double) b[i] - mb);
    }
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double pa = std::exp((double) a[i] - ma) / za, pb = std::exp((double) b[i] - mb) / zb;
        if (pa > 1e-300) s += pa * (std::log(pa) - std::log(pb));
    }
    return s;
}

int small(const std::string& gguf, const std::string& dumps, const std::string& dumps12) {
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6, 7, 8};
    GlmMetal m;
    load(m, gguf, 64);
    const size_t V = (size_t) m.geometry().n_vocab, T = prompt.size();
    const std::vector<float> oracle = read_f32(dumps + "/result_output.f32", V * T);

    // 1. batched vs the oracle (the last token's logits)
    const std::vector<float> batched = fwd(m, prompt);
    const double d_batched = max_abs(batched.data(), oracle.data() + V * (T - 1), V);
    require(d_batched < 2e-3, "batched logits diverge from ref/glm.py (" + std::to_string(d_batched) + ")");

    // 2. every prefix's argmax, each prefix its own sequence
    for (size_t t = 1; t <= T; ++t) {
        m.reset();
        const std::vector<float> lg = fwd(m, std::vector<int32_t>(prompt.begin(), prompt.begin() + (long) t));
        require(argmax(lg.data(), V) == argmax(oracle.data() + V * (t - 1), V),
                "prefix " + std::to_string(t) + " argmax differs from the oracle");
    }

    // 3. streamed vs batched
    m.reset();
    std::vector<float> streamed;
    for (int32_t tok : prompt) streamed = fwd(m, {tok});
    const double d_stream = max_abs(streamed.data(), batched.data(), V);
    require(d_stream < 1e-5, "streamed and batched logits differ (" + std::to_string(d_stream) + ")");

    // 4. 12 tokens (the prompt + the oracle's greedy chain): every prefix and the chain; the chain is derived from
    // the dump, never a constant
    const size_t T12 = 12;
    const std::vector<float> oracle12 = read_f32(dumps12 + "/result_output.f32", V * T12);
    const std::vector<int32_t> tokens12 = {1, 2, 3, 4, 5, 6, 7, 8, 98, 162, 57, 401};
    std::vector<size_t> want, got;
    double worst12 = 0.0;
    for (size_t t = 1; t <= T12; ++t) {
        m.reset();
        const std::vector<float> lg = fwd(m, std::vector<int32_t>(tokens12.begin(), tokens12.begin() + (long) t));
        worst12 = std::max(worst12, max_abs(lg.data(), oracle12.data() + V * (t - 1), V));
        if (t >= T) {
            got.push_back(argmax(lg.data(), V));
            want.push_back(argmax(oracle12.data() + V * (t - 1), V));
        }
    }
    require(worst12 < 2e-3, "12-token logits diverge from the oracle (" + std::to_string(worst12) + ")");
    require(got == want, "the greedy chain diverges from the oracle's");

    // 5. continuation: one 8-token prefill, then the chain one token at a time from the same state
    m.reset();
    fwd(m, prompt);
    double worst_cont = 0.0;
    for (size_t t = T; t < T12; ++t) {
        const std::vector<float> lg = fwd(m, {tokens12[t]});
        worst_cont = std::max(worst_cont, max_abs(lg.data(), oracle12.data() + V * t, V));
    }
    require(worst_cont < 2e-3, "continuation logits diverge from the oracle (" + std::to_string(worst_cont) + ")");

    // 6. state determinism: A, reset, B, reset, A - bitwise
    m.reset();
    const std::vector<float> a1 = fwd(m, prompt);
    m.reset();
    fwd(m, {11, 12, 13, 14, 15, 16, 17, 18});
    m.reset();
    const std::vector<float> a2 = fwd(m, prompt);
    require(a1 == a2, "state determinism: A after B differs bitwise");

    // 7. mutation sensitivity: a corrupted weight must move the logits
    {
        const std::string bad = gguf + ".mutation.tmp";
        FILE* src = std::fopen(gguf.c_str(), "rb");
        require(src != nullptr, "mutation: cannot open " + gguf);
        std::fseek(src, 0, SEEK_END);
        const long size = std::ftell(src);
        std::fseek(src, 0, SEEK_SET);
        std::vector<char> blob((size_t) size);
        require(std::fread(blob.data(), 1, blob.size(), src) == blob.size(), "mutation: short read");
        std::fclose(src);
        float* f = (float*) (blob.data() + size / 2);
        for (int i = 0; i < 1024; ++i) f[i] = f[i] * -4.0f + 3.0f;
        FILE* dst = std::fopen(bad.c_str(), "wb");
        require(dst != nullptr, "mutation: cannot write the copy");
        require(std::fwrite(blob.data(), 1, blob.size(), dst) == blob.size(), "mutation: short write");
        std::fclose(dst);
        GlmMetal c;
        load(c, bad, 64);
        const std::vector<float> mut = fwd(c, prompt);
        std::remove(bad.c_str());
        require(max_abs(mut.data(), batched.data(), V) > 1e-2, "a corrupted weight did not move the logits");
    }
    std::printf("metal_model_test small: PASS (batched-8 %.2e, streamed-8 %.2e, 12-token %.2e, continuation %.2e)\n",
                d_batched, d_stream, worst12, worst_cont);
    return 0;
}

int big(const std::string& gguf, const std::string& expect) {
    constexpr size_t T = 512;
    GlmMetal m;
    load(m, gguf, 1024);
    require(m.geometry().n_layers == 45, "the big fixture has 45 layers");
    // (MTP turns on mid-session once a streaming pool grows to hold every expert)
    require(m.pool_holds_all() == (env("MAYA_METAL_POOL_SLOTS") == 0), "pool_holds_all() disagrees with the pool");
    const size_t V = (size_t) m.geometry().n_vocab;
    std::vector<int32_t> tokens;
    for (size_t i = 0; i < T; ++i) tokens.push_back((int32_t) ((i % 511) + 1));

    std::vector<float> streamed(V * T);
    for (size_t t = 0; t < T; ++t) {
        const std::vector<float> lg = fwd(m, {tokens[t]});
        for (size_t v = 0; v < V; ++v) {
            require(std::isfinite(lg[v]), "non-finite logit at token " + std::to_string(t));
            streamed[v + V * t] = lg[v];
        }
    }
    // A. streamed vs batched at three prefixes
    double worst_sb = 0.0;
    for (size_t len : {size_t(100), size_t(250), size_t(511)}) {
        m.reset();
        const std::vector<float> b = fwd(m, std::vector<int32_t>(tokens.begin(), tokens.begin() + (long) len));
        worst_sb = std::max(worst_sb, max_abs(b.data(), streamed.data() + V * (len - 1), V));
    }
    require(worst_sb < 5e-5, "streamed vs batched diverges (" + std::to_string(worst_sb) + ")");
    // B. state determinism at 128 tokens
    {
        const std::vector<int32_t> a(tokens.begin(), tokens.begin() + 128), b(tokens.begin() + 200, tokens.begin() + 328);
        m.reset();
        const std::vector<float> a1 = fwd(m, a);
        m.reset();
        fwd(m, b);
        m.reset();
        require(fwd(m, a) == a1, "state determinism: A after B differs bitwise");
    }
    // C. vs the f64 oracle, the CUDA test's calibrated rule
    const std::vector<float> oracle = read_f32(expect, V * T);
    std::vector<double> kls(T);
    size_t agree = 0;
    for (size_t t = 0; t < T; ++t) {
        kls[t] = kl(streamed.data() + V * t, oracle.data() + V * t, V);
        agree += argmax(streamed.data() + V * t, V) == argmax(oracle.data() + V * t, V);
    }
    std::vector<double> sorted(kls);
    std::sort(sorted.begin(), sorted.end());
    const double p99 = sorted[(size_t) (T * 0.99)], worst = sorted[T - 1];
    require(p99 < 1e-4, "KL p99 too high (" + std::to_string(p99) + ")");
    require(worst < 0.1, "max KL too high (" + std::to_string(worst) + ")");
    require(agree >= 505, "argmax agreement " + std::to_string(agree) + "/512 (calibrated: >= 505)");
    std::printf("metal_model_test big: PASS (stream-vs-batch %.1e, KL p99 %.1e max %.1e, argmax %zu/512)\n", worst_sb,
                p99, worst, agree);
    return 0;
}

int mtp(const std::string& gguf, int n_gen) {
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6, 7, 8};
    GlmMetal m;
    load(m, gguf, 1024, 5, true);
    require(m.has_mtp(), "the GGUF has no NextN block (tools/glm_mtp_fixture.py writes one)");
    const size_t V = (size_t) m.geometry().n_vocab;
    std::string err;
    std::vector<int32_t> ref;   // plain greedy
    std::vector<float> lg = fwd(m, prompt);
    for (int i = 0; i < n_gen; ++i) {
        ref.push_back((int32_t) argmax(lg.data(), V));
        if (i + 1 < n_gen) lg = fwd(m, {ref.back()});
    }
    for (int fixed : {1, 2, 4}) {   // fixed draft lengths, then the adaptive choice
        m.reset();
        lg = fwd(m, prompt);
        std::vector<int32_t> got;
        maya::metal::DecodeHooks h;
        h.pick = [&](const float* l) { return (int32_t) argmax(l, V); };
        h.emit = [&](int32_t t) {
            got.push_back(t);
            return (int) got.size() < n_gen;
        };
        h.left = [&] { return (int64_t) (n_gen - (int) got.size()); };
        const int32_t y = (int32_t) argmax(lg.data(), V);
        if (!h.emit(y)) continue;
        setenv("MAYA_GLM_MTP_FIXED", std::to_string(fixed).c_str(), 1);
        maya::metal::DraftLength len(4);
        require(maya::metal::decode_mtp(m, y, len, h, err), "decode_mtp: " + err);
        for (size_t i = 0; i < got.size() && i < ref.size(); ++i)
            if (got[i] != ref[i]) {
                std::fprintf(stderr, "draft length %d: first difference at token %zu: %d vs %d\n", fixed, i, got[i],
                             ref[i]);
                break;
            }
        require(got == ref, "MTP decoding differs from plain decoding");
    }
    unsetenv("MAYA_GLM_MTP_FIXED");
    std::printf("metal_model_test mtp: PASS (%d tokens, draft lengths 1, 2, 4)\n", n_gen);
    return 0;
}

int dflash(const std::string& gguf, const std::string& drafter, int n_gen) {
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int32_t> ref;   // plain greedy, without the drafter loaded
    {
        GlmMetal p;
        load(p, gguf, 1024);
        const size_t V = (size_t) p.geometry().n_vocab;
        std::vector<float> lg = fwd(p, prompt);
        for (int i = 0; i < n_gen; ++i) {
            ref.push_back((int32_t) argmax(lg.data(), V));
            if (i + 1 < n_gen) lg = fwd(p, {ref.back()});
        }
    }
    GlmMetal m;
    load(m, gguf, 1024, 16, false, drafter);
    require(m.has_dflash(), "the drafter did not load");
    const size_t V = (size_t) m.geometry().n_vocab;
    std::string err;
    int drafted = 0, accepted = 0;
    for (float p_min : {0.0f, 0.5f}) {   // every draft offered, then the confidence cut
        m.reset();
        std::vector<float> lg = fwd(m, prompt);
        std::vector<int32_t> got;
        maya::metal::DecodeHooks h;
        h.pick = [&](const float* l) { return (int32_t) argmax(l, V); };
        h.emit = [&](int32_t t) {
            got.push_back(t);
            return (int) got.size() < n_gen;
        };
        h.left = [&] { return (int64_t) (n_gen - (int) got.size()); };
        const int32_t y = (int32_t) argmax(lg.data(), V);
        require(h.emit(y), "n_gen < 2");
        const maya::metal::Drafter df = [&](int32_t tok, int64_t, int n, std::vector<int32_t>& d, std::string& e) {
            return m.dflash_draft(tok, n, p_min, 0, 0.0f, d, e);
        };
        require(maya::metal::decode_spec(m, y, m.dflash_block() - 1, df, [&](int n, int acc, double) {
                    drafted += n;
                    accepted += acc;
                }, h, err),
                "decode_spec: " + err);
        for (size_t i = 0; i < got.size() && i < ref.size(); ++i)
            if (got[i] != ref[i]) {
                std::fprintf(stderr, "p_min %.1f: first difference at token %zu: %d vs %d\n", p_min, i, got[i], ref[i]);
                break;
            }
        require(got == ref, "DFlash2 decoding differs from plain decoding");
    }
    require(drafted > 0, "the drafter drafted nothing");
    std::printf("metal_model_test dflash: PASS (%d tokens twice, %d drafted, %d accepted)\n", n_gen, drafted, accepted);
    return 0;
}

int prefill(const std::string& gguf, int T) {
    std::vector<int32_t> toks;
    for (int i = 0; i < T; ++i) toks.push_back((int32_t) ((i * 37 + 11) % 500 + 1));
    GlmMetal m;
    load(m, gguf, 2048);
    const size_t V = (size_t) m.geometry().n_vocab;
    const std::vector<float> batched = fwd(m, toks);
    m.reset();
    std::vector<float> streamed;
    for (int32_t t : toks) streamed = fwd(m, {t});
    for (float x : batched) require(std::isfinite(x), "non-finite logit in the batched prompt");
    const double k = kl(batched.data(), streamed.data(), V);
    require(argmax(batched.data(), V) == argmax(streamed.data(), V), "the prompt's argmax differs from streaming's");
    require(k < 1e-3, "the prompt's logits drift from streaming's (KL " + std::to_string(k) + ")");
    // the DSA products as matrices (dsa_mm: half tiles) against the per-head f32 matvecs they replace
    GlmMetal mv;
    load(mv, gguf, 2048, 0, false, "", true);
    const std::vector<float> matvec = fwd(mv, toks);
    const double kmv = kl(batched.data(), matvec.data(), V);
    require(argmax(batched.data(), V) == argmax(matvec.data(), V), "the prompt's argmax differs from the matvec DSA's");
    require(kmv < 1e-3, "the prompt's logits drift from the matvec DSA's (KL " + std::to_string(kmv) + ")");
    std::printf("metal_model_test prefill: PASS (%d tokens, KL %.1e, max|d| %.1e; matvec DSA KL %.1e)\n", T, k,
                max_abs(batched.data(), streamed.data(), V), kmv);
    return 0;
}

int window(const std::string& gguf, int n_gen) {
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6, 7, 8};
    GlmMetal m;
    load(m, gguf, 1024, 8);
    const size_t V = (size_t) m.geometry().n_vocab;
    std::string err;
    // plain greedy: the reference tokens and each step's logits
    std::vector<int32_t> ref;
    std::vector<std::vector<float>> ref_lg;
    std::vector<float> lg = fwd(m, prompt);
    for (int i = 0; i < n_gen; ++i) {
        ref_lg.push_back(lg);
        ref.push_back((int32_t) argmax(lg.data(), V));
        if (i + 1 < n_gen) lg = fwd(m, {ref.back()});
    }
    // windows: row 0 is the token just picked, rows 1.. are drafts - the reference's next tokens, every third one
    // wrong - and the window grows 1, 2, ... 8 rows and wraps
    m.reset();
    lg = fwd(m, prompt);
    std::vector<int32_t> got;
    double worst = 0.0;
    int t = (int) argmax(lg.data(), V), round = 0;
    while ((int) got.size() < n_gen) {
        const int k = std::min(1 + round++ % 8, n_gen - (int) got.size());
        std::vector<int32_t> win = {t};
        for (int i = 1; i < k; ++i) {
            const size_t at = got.size() + (size_t) i;
            win.push_back(at < ref.size() && (at + round) % 3 != 0 ? ref[at] : (int32_t) ((at * 7 + 3) % V));
        }
        std::vector<float> all;
        require(m.forward_window(win, all, err), "window: " + err);
        int n = 1;   // row i's pick checks draft i + 1
        while (n < k && (int32_t) argmax(all.data() + V * (size_t) (n - 1), V) == win[(size_t) n]) ++n;
        for (int i = 0; i < n; ++i) {
            got.push_back(win[(size_t) i]);
            if (got.size() < ref_lg.size())
                worst = std::max(worst, max_abs(all.data() + V * (size_t) i, ref_lg[got.size()].data(), V));
        }
        t = (int) argmax(all.data() + V * (size_t) (n - 1), V);
        require(m.accept(n, err), "accept: " + err);
    }
    got.resize((size_t) n_gen);
    for (size_t i = 0; i < got.size(); ++i)
        if (got[i] != ref[i]) {
            std::fprintf(stderr, "first difference at token %zu: %d vs %d\n", i, got[i], ref[i]);
            break;
        }
    require(got == ref, "window decoding differs from plain decoding");
    require(worst < 1e-5, "window logits differ from plain decoding's (" + std::to_string(worst) + ")");
    std::printf("metal_model_test window: PASS (%d tokens, %d windows, logits %.1e)\n", n_gen, round, worst);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "small")
        return small(argc > 2 ? argv[2] : "data/glm5-synth.gguf", argc > 3 ? argv[3] : "data/glm5-synth-dumps",
                     argc > 4 ? argv[4] : "data/glm5-synth-dumps12");
    if (mode == "big" && argc > 2) return big(argv[2], argc > 3 ? argv[3] : "data/glm5-big-dumps/result_output.f32");
    if (mode == "prefill" && argc > 2) return prefill(argv[2], argc > 3 ? std::atoi(argv[3]) : 512);
    if (mode == "dflash" && argc > 3) return dflash(argv[2], argv[3], argc > 4 ? std::atoi(argv[4]) : 120);
    if (mode == "mtp" && argc > 2) return mtp(argv[2], argc > 3 ? std::atoi(argv[3]) : 40);
    if (mode == "window" && argc > 2) return window(argv[2], argc > 3 ? std::atoi(argv[3]) : 48);
    std::fprintf(stderr, "usage: metal_model_test small [GGUF DUMPS DUMPS12] | big GGUF [EXPECT] | window GGUF [N] | mtp GGUF [N] | dflash GGUF DRAFTER [N] | prefill GGUF [T]\n");
    return 2;
}
