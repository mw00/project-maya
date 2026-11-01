// src/metal/mm3_main.cpp - maya-metal-mm3: MiniMax-M3.1-Flash on Apple Silicon (the CLI half; the
// maya.sh serve wiring drives the same engine through maya.py's server, see docs/METAL.md).
//
//   maya-metal-mm3 --gguf MODEL.gguf --serve [...]                 the serve protocol (mm3_serve.cpp)
//   maya-metal-mm3 --gguf MODEL.gguf --ids 1,2,3 [--n 64]           greedy continuation of token ids, with timings
//   maya-metal-mm3 --gguf MODEL.gguf --ids 1,2,3 --cpu              the same on ggml-cpu (the oracle backend)
#include "mm3_metal.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace maya::metal {
int serve(Mm3Metal& model, const std::vector<std::string>& args);
}

namespace {

void usage() {
    std::fprintf(stderr,
                 "usage: maya-metal-mm3 --gguf MODEL.gguf [--serve | --ids 1,2,3 [--n N]]\n"
                 "  --max-context N      tokens the KV cache holds (default 32768)\n"
                 "  --ubatch N           prompt tokens per slice (default 512)\n"
                 "  --cpu                run on ggml-cpu instead of the GPU (reference runs)\n");
}

std::vector<int32_t> parse_ids(const char* s) {
    std::vector<int32_t> v;
    for (const char* p = s; *p;) {
        char* end = nullptr;
        v.push_back((int32_t) std::strtol(p, &end, 10));
        p = *end == ',' ? end + 1 : end;
        if (end == p && *p) break;
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf;
    int64_t max_ctx = 32768;
    bool serve = false;
    std::vector<int32_t> ids;
    int n_gen = 32;
    maya::metal::Mm3Options o;
    std::vector<std::string> rest;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "maya-metal-mm3: %s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--gguf") gguf = next();
        else if (a == "--max-context") max_ctx = std::atoll(next());
        else if (a == "--serve") serve = true;
        else if (a == "--ids") ids = parse_ids(next());
        else if (a == "--n") n_gen = std::atoi(next());
        else if (a == "--ubatch") o.n_ubatch = std::atoi(next());
        else if (a == "--cpu") o.use_gpu = false;
        else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else {
            rest.push_back(a);   // the serve loop takes the server's other flags
        }
    }
    if (gguf.empty() || (ids.empty() && !serve)) {
        usage();
        return 2;
    }
    maya::metal::Mm3Metal model;
    model.set_options(o);
    std::string err;
    const auto t0 = std::chrono::steady_clock::now();
    if (!model.load(gguf, max_ctx, err)) {
        std::fprintf(stderr, "maya-metal-mm3: %s\n", err.c_str());
        return 1;
    }
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "maya-metal-mm3: loaded in %.1f s: %s\n", load_s, model.describe().c_str());
    if (serve) return maya::metal::serve(model, rest);

    std::vector<float> logits;
    auto t1 = std::chrono::steady_clock::now();
    if (!model.forward(ids, logits, err)) {
        std::fprintf(stderr, "maya-metal-mm3: %s\n", err.c_str());
        return 1;
    }
    const double pp = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    std::printf("prompt %zu tokens in %.2f s (%.1f tok/s)\n", ids.size(), pp, ids.size() / pp);
    const size_t V = (size_t) model.geometry().n_vocab;
    auto argmax = [&](const float* l) {
        int32_t best = 0;
        for (size_t v = 1; v < V; ++v)
            if (l[v] > l[(size_t) best]) best = (int32_t) v;
        return best;
    };
    std::vector<int32_t> out;
    t1 = std::chrono::steady_clock::now();
    for (int32_t t = argmax(logits.data()); (int) out.size() < n_gen;) {
        out.push_back(t);
        if ((int) out.size() >= n_gen) break;
        if (!model.forward({t}, logits, err)) {
            std::fprintf(stderr, "maya-metal-mm3: %s\n", err.c_str());
            return 1;
        }
        t = argmax(logits.data());
    }
    const double tg = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    std::printf("tokens:");
    for (int32_t t : out) std::printf(" %d", t);
    const auto& s = model.stats();
    std::printf("\ndecode %d tokens in %.2f s (%.2f tok/s), %llu graphs, gpu %.2f s, host %.2f s\n",
                (int) out.size(), tg, out.size() / tg, (unsigned long long) s.graphs, s.gpu_seconds,
                s.host_seconds);
    return 0;
}
