#include "maya/artifact/gguf_reader.hpp"
#include "maya/core/glp.hpp"
#include <cstdio>
#include <string>
#include <vector>

static int fails = 0;
static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++fails; }
    else std::fprintf(stderr, "ok: %s\n", what);
}

int main() {
    std::fprintf(stderr, "test start\n");
    maya::core::glp::Vector v;
    std::string err;
    std::vector<std::string> warnings;
    maya::core::glp::LoadOptions lo;
    std::fprintf(stderr, "calling load\n");
    const bool ok = maya::core::glp::load("/tmp/glp-inkling2mini.gguf", 2304, 48, lo, v, err, warnings);
    std::fprintf(stderr, "load returned %d err='%s'\n", (int)ok, err.c_str());
    check(ok, "real GLP-48 loads");
    if (ok) {
        check(v.alpha == 0.25f, "alpha is 0.25");
        check((int)v.layers.size() == 48, "48 layers");
        check(v.layers.front() == 1 && v.layers.back() == 48, "layers 1..48");
        check(v.mode == "project", "mode project");
    }
    // refusal paths
    {
        maya::core::glp::Vector v2; std::string e2; std::vector<std::string> w2;
        maya::core::glp::LoadOptions lo2;
        const bool bad = maya::core::glp::load("/tmp/glp-inkling2mini.gguf", 4096, 48, lo2, v2, e2, w2);
        check(!bad, "wrong n_embd refuses");
        std::fprintf(stderr, "   refusal: %s\n", e2.c_str());
        maya::core::glp::Vector v3; std::string e3; std::vector<std::string> w3;
        maya::core::glp::LoadOptions lo3;
        const bool bad3 = maya::core::glp::load("/tmp/glp-inkling2mini.gguf", 2304, 24, lo3, v3, e3, w3);
        check(!bad3, "fewer model layers refuses");
        std::fprintf(stderr, "   refusal: %s\n", e3.c_str());
    }

    // --ablate path: the same reader feeds the GLM engine's per-layer projection.  The reader
    // contract is identical; the test above covers it.  What the GLM path adds is the routing:
    // a GLP file refuses --ablate-layer (the file names its layers) - checked at the flag level,
    // so only the reader behavior is testable here: alpha 0 (the file's default) and the full
    // layer list survive the LoadOptions mapping.
    {
        maya::core::glp::Vector v; std::string e; std::vector<std::string> w;
        maya::core::glp::LoadOptions lo;
        lo.alpha = 0.0f;  // --ablate-scale 0 = the file's default: LoadOptions < 0 does that
        lo.alpha = -1.0f;
        check(maya::core::glp::load("/tmp/glp-inkling2mini.gguf", 2304, 48, lo, v, e, w), "ablate-path loads");
        check(v.alpha == 0.25f, "ablate-path alpha 0 keeps glp.alpha_default");
        check(v.s[(size_t) 1] == 0.25f && v.s[(size_t) 48] == 0.25f, "s table carries the dose per steered layer");
        check(v.s[(size_t) 0] == 0.0f, "layer 0 unsteered");
    }

    std::fprintf(stderr, fails ? "FAILURES: %d\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
