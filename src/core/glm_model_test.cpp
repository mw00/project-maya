// src/core/glm_model_test.cpp - the glm5-next runner vs the oracle, and streaming vs batched.
//
// Four gates (docs/GLM5-FLASH.md Phase 3):
//
//   1. BATCHED VS THE ORACLE: forward(prompt) must reproduce ref/glm.py's teacher-forced logits
//      (the last column of the result_output dump) - the whole model through the runner.
//   2. STREAMING VS BATCHED: single-token calls on a persistent state must produce the SAME logits
//      as the batched call.  This is the gate that matters for the engine: decode IS streaming, and
//      the runner runs the identical per-token steps either way, so the tolerance is tight.
//   3. EVERY PREFIX ARGMAX: each teacher-forced prefix's argmax must match the oracle's - the
//      sharpest end-to-end check, because it flips on any seam that diverges enough to reorder the
//      top logits (each prefix is its OWN sequence: the runner is reset before it).
//   4. GREEDY CONTINUATION: argmax-stepping the runner (state persisting) must follow the oracle's
//      greedy chain for the CURRENT dump - computed at run time from the dump's own columns,
//      never a constant (a Phase-1 chain hardcoded here went stale when the oracle was repaired).
#include "maya/core/glm_model.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

void require(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "glm_model_test: %s\n", what.c_str());
        std::exit(1);
    }
}

// the oracle's dump: (vocab, T) ggml order - element (v, t) at v + vocab*t
std::vector<float> load_logits(const std::string& path, size_t vocab, size_t T) {
    std::vector<float> v(vocab * T);
    FILE* f = std::fopen(path.c_str(), "rb");
    require(f != nullptr, "cannot open " + path);
    require(std::fread(v.data(), sizeof(float), v.size(), f) == v.size(), "short read on " + path);
    std::fclose(f);
    return v;
}

double max_abs(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) m = std::max(m, (double) std::fabs(a[i] - b[i]));
    return m;
}

double max_abs(const float* a, const float* b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) m = std::max(m, (double) std::fabs(a[i] - b[i]));
    return m;
}

size_t argmax(const std::vector<float>& v) {
    size_t am = 0;
    for (size_t i = 1; i < v.size(); ++i)
        if (v[i] > v[am]) am = i;
    return am;
}

}  // namespace

int main(int argc, char** argv) {
    const bool selftest = argc > 1 && std::string(argv[1]) == "--selftest";
    if (!selftest) {
        std::fprintf(stderr, "glm_model_test: run with --selftest (the CTest form)\n");
        return 2;
    }
    const std::string gguf = argc > 2 ? argv[2] : "data/glm5-synth.gguf";
    const std::string dumps = argc > 3 ? argv[3] : "data/glm5-synth-dumps";
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6, 7, 8};
    // the data paths are relative to the repo root (the CTest form runs there): from the build folder it ended with
    // no message, 0xC0000409 on Windows (#54)
    if (std::FILE* f = std::fopen(gguf.c_str(), "rb")) {
        std::fclose(f);
    } else {
        std::fprintf(stderr, "glm_model_test: %s not found - run it from the repo root, or pass the file\n",
                     gguf.c_str());
        return 2;
    }

    maya::core::Glm5Model model;
    std::string err;
    require(model.load(gguf, 64, err), "load: " + err);
    const size_t V = (size_t) model.geometry().n_vocab;
    const size_t T = prompt.size();
    const std::vector<float> oracle = load_logits(dumps + "/result_output.f32", V, T);

    // ---- 1. batched vs the oracle (last token's logits)
    std::vector<float> batched;
    require(model.forward(prompt, batched, err), "batched forward: " + err);
    const std::vector<float> oracle_last(oracle.begin() + (long) V * (long) (T - 1), oracle.end());
    const double d_batched = max_abs(batched, oracle_last);
    require(d_batched < 2e-3, "batched logits diverge from ref/glm.py (" + std::to_string(d_batched) + ")");

    // ---- 3. every prefix argmax (each prefix its own sequence: reset first)
    for (size_t t = 1; t <= T; ++t) {
        model.reset();
        std::vector<float> pref;
        require(model.forward(std::vector<int32_t>(prompt.begin(), prompt.begin() + (long) t), pref, err),
                "prefix forward: " + err);
        const size_t am = argmax(pref);
        const float* col = oracle.data() + V * (t - 1);
        const size_t om = argmax(std::vector<float>(col, col + V));
        if (am != om)
            std::fprintf(stderr, "dbg prefix %zu: argmax %zu vs %zu, maxdiff %.3e\n", t, am, om,
                         max_abs(pref, std::vector<float>(col, col + V)));
        require(am == om, "prefix " + std::to_string(t) + " argmax differs from the oracle");
    }

    // ---- 2. streaming vs batched: the same per-token steps, so the same values
    model.reset();
    std::vector<float> streamed;
    for (int32_t tok : prompt) {
        std::vector<float> one;
        require(model.forward({tok}, one, err), "streamed forward: " + err);
        streamed = std::move(one);
    }
    const double d_stream = max_abs(streamed, batched);
    require(d_stream < 1e-5, "streamed and batched logits differ (" + std::to_string(d_stream) + ")");

    // ---- 4. full-sequence parity at 12 tokens (prompt 1..8 + the oracle's own greedy chain) and
    // streaming continuation.  The expected chain is DERIVED from the committed 12-token oracle
    // dump, never hardcoded - a hardcoded chain goes stale the moment the oracle gains a fix
    // (this exact bug: the Phase-1 chain [385, 93, 505, 203] predated the conv-einsum repair and
    // silently poisoned this gate).  The runner's forward returns the LAST token's logits, so
    // per-column parity means one forward per prefix (78 cheap steps total), and the streaming
    // check chains one 8-token prefill plus the 4 chain tokens singly from the same state.
    const std::string dumps12 = argc > 4 ? argv[4] : "data/glm5-synth-dumps12";
    const size_t T12 = 12;
    const std::vector<float> oracle12 = load_logits(dumps12 + "/result_output.f32", V, T12);
    const std::vector<int32_t> tokens12 = {1, 2, 3, 4, 5, 6, 7, 8, 98, 162, 57, 401};
    std::vector<int32_t> greedy_expected;
    for (size_t t = T - 1; t < T12; ++t) {
        const float* col = oracle12.data() + V * t;
        greedy_expected.push_back((int32_t) argmax(std::vector<float>(col, col + V)));
    }

    double worst12 = 0.0;
    std::vector<int32_t> greedy;
    for (size_t t = 1; t <= T12; ++t) {
        model.reset();
        std::vector<float> lg;
        require(model.forward(std::vector<int32_t>(tokens12.begin(), tokens12.begin() + (long) t), lg, err),
                "12-token forward: " + err);
        worst12 = std::max(worst12, max_abs(lg.data(), oracle12.data() + V * (t - 1), V));
        if (t >= T)
            greedy.push_back((int32_t) argmax(lg));   // columns 8..11: the greedy chain
    }
    require(worst12 < 2e-3, "12-token logits diverge from the oracle (" + std::to_string(worst12) + ")");
    require(greedy == greedy_expected, "the runner's greedy chain diverges from the oracle's");

    // streaming continuation: one 8-token prefill, then the chain fed singly from the same state;
    // every continuation logit vs the oracle's columns 8..11 - the persistent-state check at the
    // prefill-to-decode boundary (positions 8..11 have never been covered by any other gate)
    model.reset();
    std::vector<float> cont;
    require(model.forward(prompt, cont, err), "continuation prefill: " + err);
    double worst_cont = 0.0;
    for (size_t t = T; t < T12; ++t) {
        require(model.forward({tokens12[t]}, cont, err), "continuation forward: " + err);
        worst_cont = std::max(worst_cont, max_abs(cont.data(), oracle12.data() + V * t, V));
    }
    require(worst_cont < 2e-3, "continuation logits diverge from the oracle (" + std::to_string(worst_cont) + ")");

    // ---- 5. STATE DETERMINISM (second opinion): forward(A), reset, forward(B), reset,
    // forward(A) again - the third run must reproduce the first BITWISE.  Anything stateful
    // leaking across sequences (an unwashed KDA S, a stale DSA pool row) shows up here.
    const std::vector<int32_t> seqB = {11, 12, 13, 14, 15, 16, 17, 18};
    std::vector<float> runA1, runB, runA2;
    model.reset();   // gate 4's greedy loop leaves the state at position 12; start clean
    require(model.forward(prompt, runA1, err), "state A1: " + err);
    model.reset();
    require(model.forward(seqB, runB, err), "state B: " + err);
    model.reset();
    require(model.forward(prompt, runA2, err), "state A2: " + err);
    require(runA1 == runA2, "state determinism: A-after-B differs bitwise from A (max "
                                + std::to_string(max_abs(runA1, runA2)) + ")");

    // ---- 6. MUTATION SENSITIVITY (second opinion): a corrupted weight MUST move the logits,
    // or gate 1 could pass against a wrong model.  Copy the GGUF, overwrite 4 KB inside a
    // routed expert slab, reload, forward - the max delta must be far above f32 noise.
    {
        const std::string corrupt_path = gguf + ".mutation.tmp";
        FILE* src = std::fopen(gguf.c_str(), "rb");
        require(src != nullptr, "mutation: cannot open " + gguf);
        std::fseek(src, 0, SEEK_END);
        long size = std::ftell(src);
        std::fseek(src, 0, SEEK_SET);
        std::vector<char> blob((size_t) size);
        require(std::fread(blob.data(), 1, blob.size(), src) == blob.size(), "mutation: short read");
        std::fclose(src);
        // find the expert slab by scanning for its float payload: the fixture's blk.1
        // ffn_gate_exps starts after the metadata; corrupt a slice in the middle of the file
        const long at = size / 2;                      // any mid-file weight bytes
        float* f = (float*) (blob.data() + at);
        for (int i = 0; i < 1024; ++i) f[i] = f[i] * -4.0f + 3.0f;
        FILE* dst = std::fopen(corrupt_path.c_str(), "wb");
        require(dst != nullptr, "mutation: cannot write copy");
        require(std::fwrite(blob.data(), 1, blob.size(), dst) == blob.size(), "mutation: short write");
        std::fclose(dst);

        maya::core::Glm5Model corrupt;
        require(corrupt.load(corrupt_path, 64, err), "mutation load: " + err);
        std::vector<float> mutated;
        require(corrupt.forward(prompt, mutated, err), "mutation forward: " + err);
        std::remove(corrupt_path.c_str());
        const double d_mut = max_abs(mutated, batched);
        require(d_mut > 1e-2, "mutation was NOT detected by the logits (max delta "
                                  + std::to_string(d_mut) + " - gate 1 cannot catch a wrong model)");
        std::printf("mutation sensitivity: %.3e\n", d_mut);
    }

    std::printf("glm_model_test: PASS (batched-8 %.2e, streamed-8 %.2e, 12-token %.2e, continuation %.2e)\n",
                d_batched, d_stream, worst12, worst_cont);
    return 0;
}
