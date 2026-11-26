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
    std::fprintf(stderr, fails ? "FAILURES: %d\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
