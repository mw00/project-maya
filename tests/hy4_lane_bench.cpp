// tests/hy4_lane_bench.cpp - maya-hy4's CPU lane (hy4::cpu_experts) on synthetic experts of Hy4-flash's shape (1536 x
// 1536, no GPU, no model): a call's time and GB/s by format and tokens, and a hash of every call's outputs - two runs
// that print the same hashes gave the same bits (e.g. DS4_IQ_OWN=0, the native lane's rows = ggml-cpu's dot, against
// the default choice).  The experts: random blocks (each block's F16 scales small normal values), more of them than
// the caches hold, a call's 16 experts a token drawn at random, the activations random.
//
//   hy4_lane_bench [--format iq2_xxs|iq2_s|iq3_xxs|iq4_xs|q4_k] [--tokens 1] [--calls 400] [--experts 96]
//                  [--threads -1] [--dump FILE] (one hash a call) [--q4x 1] (iq2_xxs: every expert's 4-bit copy, the
//                  rows the lane then takes)
#include "../src/hy4/hy4_lane.hpp"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {
uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}
int type_of(const std::string& s) {
    if (s == "iq2_xxs") return GGML_TYPE_IQ2_XXS;
    if (s == "iq2_xs") return GGML_TYPE_IQ2_XS;
    if (s == "iq2_s") return GGML_TYPE_IQ2_S;
    if (s == "iq3_xxs") return GGML_TYPE_IQ3_XXS;
    if (s == "iq3_s") return GGML_TYPE_IQ3_S;
    if (s == "iq4_xs") return GGML_TYPE_IQ4_XS;
    if (s == "q4_k") return GGML_TYPE_Q4_K;
    return -1;
}
// random rows of `type`, each block's leading F16 scale (and Q4_K's dmin) small and normal
void fill(std::vector<uint8_t>& w, int type, std::mt19937_64& rng) {
    for (size_t o = 0; o + 8 <= w.size(); o += 8) {
        const uint64_t v = rng();
        std::memcpy(&w[o], &v, 8);
    }
    const size_t bb = ggml_type_size((ggml_type) type);
    for (size_t o = 0; o + bb <= w.size(); o += bb) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(0.0005f + 0.0001f * (float) (rng() % 64));
        std::memcpy(&w[o], &d, 2);
        if (type == GGML_TYPE_Q4_K) {
            const ggml_fp16_t m = ggml_fp32_to_fp16(0.0002f + 0.0001f * (float) (rng() % 16));
            std::memcpy(&w[o + 2], &m, 2);
        }
    }
}
}  // namespace

int main(int argc, char** argv) {
    std::string fmt = "iq2_xxs", dump;
    int tokens = 1, calls = 400, n_exp = 96, threads = -1, q4x = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--format") fmt = argv[i + 1];
        else if (a == "--tokens") tokens = std::atoi(argv[i + 1]);
        else if (a == "--calls") calls = std::atoi(argv[i + 1]);
        else if (a == "--experts") n_exp = std::atoi(argv[i + 1]);
        else if (a == "--threads") threads = std::atoi(argv[i + 1]);
        else if (a == "--dump") dump = argv[i + 1];
        else if (a == "--q4x") q4x = std::atoi(argv[i + 1]);
    }
    const int type = type_of(fmt);
    if (type < 0) {
        std::fprintf(stderr, "unknown format %s\n", fmt.c_str());
        return 2;
    }
    const int E = 1536, F = 1536, K = 16;
    hy4::CpuExperts ex;
    std::string err;
    if (!ex.init(type, type, E, F, err)) {
        std::fprintf(stderr, "format %s: %s\n", fmt.c_str(), err.c_str());
        return 2;
    }
    std::mt19937_64 rng(12345);
    std::vector<uint8_t> gate(ex.g_stride * (size_t) n_exp), up(gate.size()), down(ex.d_stride * (size_t) n_exp);
    fill(gate, type, rng);
    fill(up, type, rng);
    fill(down, type, rng);
    ex.gate = gate.data();
    ex.up = up.data();
    ex.down = down.data();
    ex.limit = 7.0f;
    if (q4x) {
        if (!ex.q4x_can()) {
            std::fprintf(stderr, "format %s: no 4-bit copies here\n", fmt.c_str());
            return 2;
        }
        ex.q4x = std::make_shared<hy4::Q4xSet>(n_exp);
        for (int e = 0; e < n_exp; ++e) ex.q4x_make(e);
    }
    hy4::Lane lane(threads);
    // the calls' entries and activations, fixed by the seed (the same in every run)
    std::vector<std::vector<hy4::CpuEntry>> ent((size_t) calls);
    std::vector<std::vector<float>> xs((size_t) calls);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int c = 0; c < calls; ++c) {
        for (int t = 0; t < tokens; ++t) {
            std::vector<int> pick;
            while ((int) pick.size() < K) {
                const int e = (int) (rng() % (uint64_t) n_exp);
                if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
            }
            for (int e : pick) ent[(size_t) c].push_back({t, e, 0.02f + 0.01f * (float) (rng() % 8)});
        }
        xs[(size_t) c].resize((size_t) tokens * E);
        for (float& v : xs[(size_t) c]) v = nd(rng);
    }
    std::vector<float> out((size_t) tokens * E);
    FILE* fd = dump.empty() ? nullptr : std::fopen(dump.c_str(), "w");
    uint64_t h = 1469598103934665603ull;
    for (int c = 0; c < 8; ++c) hy4::cpu_experts(lane, ex, ent[(size_t) c], xs[(size_t) c].data(), tokens, out.data());   // (warm-up)
    double best = 1e30, sum = 0;
    for (int c = 0; c < calls; ++c) {
        const auto t0 = std::chrono::steady_clock::now();
        hy4::cpu_experts(lane, ex, ent[(size_t) c], xs[(size_t) c].data(), tokens, out.data());
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        best = std::min(best, s);
        sum += s;
        const uint64_t hc = fnv(out.data(), out.size() * sizeof(float));
        h = fnv(&hc, sizeof hc, h);
        if (fd) std::fprintf(fd, "%d %016llx\n", c, (unsigned long long) hc);
    }
    if (fd) std::fclose(fd);
    // bytes a call: each distinct expert of the call once
    double bytes = 0;
    {
        std::vector<int> d;
        for (const auto& e : ent[0]) d.push_back(e.e);
        std::sort(d.begin(), d.end());
        bytes = (double) (std::unique(d.begin(), d.end()) - d.begin()) *
                (double) (q4x ? 2 * ex.q4x_gu + ex.q4x_d : 2 * ex.g_stride + ex.d_stride);
    }
    const double mean = sum / calls;
    std::printf("%s%s tokens %d threads %d: mean %.3f ms (best %.3f), %.1f GB/s mean, hash %016llx\n", fmt.c_str(),
                q4x ? " q4x" : "", tokens,
                lane.threads() + 1, 1e3 * mean, 1e3 * best, bytes / mean / 1e9, (unsigned long long) h);
    return 0;
}
