// src/metal/metal_mm3_test.cpp - the MiniMax-M3.1 Metal engine against its synthetic fixture
// (tools/mm3_metal_fixture.py: a numpy port of tools/mm3/ref.py's forward over the fixture's weights).
//
//   metal_mm3_test [fixture.gguf expected.bin]
// Gates: the prompt's logits against the oracle; the same prompt in two slices (the KV state carries
// across forwards); greedy decoding token for token against the oracle's whole-sequence greedy.
// MAYA_MM3_CPU=1 runs the engine on ggml-cpu (the oracle backend).
#include "mm3_metal.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Expected {
    std::vector<int32_t> ids;
    std::vector<float> logits;
    std::vector<int32_t> greedy;
    int vocab = 0;
};

bool read_expected(const std::string& path, Expected& e) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[8] = {};
    uint32_t hdr[3] = {};
    bool ok = std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, "MM3EXP1", 7) == 0 &&
              std::fread(hdr, 4, 3, f) == 3;
    if (ok) {
        e.ids.resize(hdr[0]);
        e.greedy.resize(hdr[1]);
        e.vocab = (int) hdr[2];
        e.logits.resize((size_t) e.vocab);
        ok = std::fread(e.ids.data(), 4, e.ids.size(), f) == e.ids.size() &&
             std::fread(e.logits.data(), 4, e.logits.size(), f) == e.logits.size() &&
             std::fread(e.greedy.data(), 4, e.greedy.size(), f) == e.greedy.size();
    }
    std::fclose(f);
    return ok;
}

double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, r = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += (double) (a[i] - b[i]) * (a[i] - b[i]);
        r += (double) b[i] * b[i];
    }
    return std::sqrt(d) / std::sqrt(r);
}

int32_t argmax(const std::vector<float>& l) {
    size_t best = 0;
    for (size_t v = 1; v < l.size(); ++v)
        if (l[v] > l[best]) best = v;
    return (int32_t) best;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string gguf = argc > 1 ? argv[1] : "build-metal/minimax-m3-synth.gguf";
    const std::string exp_path = argc > 2 ? argv[2] : gguf + ".expected.bin";
    Expected e;
    if (!read_expected(exp_path, e)) {
        std::fprintf(stderr, "metal_mm3_test: cannot read %s (run tools/mm3_metal_fixture.py)\n",
                     exp_path.c_str());
        return 2;
    }
    maya::metal::Mm3Options o;
    if (const char* v = std::getenv("MAYA_MM3_CPU"); v && *v && *v != '0') o.use_gpu = false;
    if (const char* v = std::getenv("MAYA_MM3_UBATCH"); v && *v) o.n_ubatch = std::atoi(v);
    maya::metal::Mm3Metal model;
    model.set_options(o);
    std::string err;
    const int64_t ctx = (int64_t) e.ids.size() + (int64_t) e.greedy.size() + 16;
    if (!model.load(gguf, ctx, err)) {
        std::fprintf(stderr, "metal_mm3_test: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "metal_mm3_test: %s\n", model.describe().c_str());
    int fails = 0;
    auto check = [&](const char* name, bool ok, double rel) {
        std::printf("%-28s %s  rel L2 %.3e\n", name, ok ? "OK" : "MISMATCH", rel);
        if (!ok) ++fails;
    };

    // gate 1: the whole prompt in one forward, logits against the oracle
    std::vector<float> logits;
    if (!model.forward(e.ids, logits, err)) {
        std::fprintf(stderr, "metal_mm3_test: %s\n", err.c_str());
        return 1;
    }
    const double rel1 = rel_l2(logits, e.logits);
    check("prompt logits vs oracle", rel1 < 1e-3, rel1);

    // gate 2: the same prompt in two slices lands on the same logits (the caches carry)
    model.reset();
    const size_t cut = 1000;
    std::vector<int32_t> head(e.ids.begin(), e.ids.begin() + (long) cut), tail(e.ids.begin() + (long) cut,
                                                                               e.ids.end());
    if (!model.forward(head, logits, err) || !model.forward(tail, logits, err)) {
        std::fprintf(stderr, "metal_mm3_test: %s\n", err.c_str());
        return 1;
    }
    const double rel2 = rel_l2(logits, e.logits);
    check("sliced prompt logits", rel2 < 1e-3, rel2);

    // gate 3: greedy decoding, token for token against the oracle's whole-sequence greedy
    model.reset();
    if (!model.forward(e.ids, logits, err)) {
        std::fprintf(stderr, "metal_mm3_test: %s\n", err.c_str());
        return 1;
    }
    bool greedy_ok = true;
    for (size_t i = 0; i < e.greedy.size(); ++i) {
        const int32_t t = argmax(logits);
        if (t != e.greedy[i]) {
            std::printf("greedy step %zu: engine %d, oracle %d\n", i, t, e.greedy[i]);
            greedy_ok = false;
            break;
        }
        if (i + 1 < e.greedy.size() && !model.forward({t}, logits, err)) {
            std::fprintf(stderr, "metal_mm3_test: %s\n", err.c_str());
            return 1;
        }
    }
    check("greedy continuation", greedy_ok, 0.0);
    std::printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
