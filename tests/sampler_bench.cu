// tests/sampler_bench.cu - the sampled chain's kernel alone: us a row for the plain draw (sample_tokens) and the draft
// (sample_draft) on a vocabulary-sized row of synthetic logits (a normal bulk and a
// few likely tokens, as a model's row), and an FNV hash of every result - two builds whose hashes agree drew the same
// tokens and lists.    sampler_bench [iters [vocab [top_k [top_p]]]]   (2000, 129280, 0, 1.0)
#include "maya/kernels/sampler.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
using namespace maya::kernels;
static uint64_t fnv(const void* p, size_t n, uint64_t h) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}
int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 2000, V = argc > 2 ? std::atoi(argv[2]) : 129280;
    const int rows = 8;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 2.5f);
    std::vector<float> lg((size_t) rows * V);
    for (auto& x : lg) x = nd(rng);
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < 40; ++i) lg[(size_t) r * V + rng() % V] = 9.0f + 0.25f * (float) (rng() % 40);
    float* d_lg = nullptr;
    int* d_out = nullptr;
    SpecList* d_q = nullptr;
    cudaMalloc(&d_lg, lg.size() * 4);
    cudaMalloc(&d_out, 64 * 4);
    cudaMalloc(&d_q, 16 * sizeof(SpecList));
    cudaMemcpy(d_lg, lg.data(), lg.size() * 4, cudaMemcpyHostToDevice);
    SamplerParams sp;
    sp.greedy = false;
    sp.temperature = 1.0f;
    sp.top_k = argc > 3 ? std::atoi(argv[3]) : 0;
    sp.top_p = argc > 4 ? (float) std::atof(argv[4]) : 1.0f;
    sp.seed = 12345;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    uint64_t h = 1469598103934665603ull;
    for (int mode = 0; mode < 2; ++mode) {
        std::vector<int> outs;
        std::vector<SpecList> ls(1);
        for (int pass = 0; pass < 2; ++pass) {   // pass 0: results for the hash (and a warm-up); pass 1: timed
            const int n = pass == 0 ? 64 : iters;
            cudaEventRecord(e0, 0);
            for (int i = 0; i < n; ++i) {
                sp.counter = (uint64_t) i;
                const float* row = d_lg + (size_t) (i % rows) * V;
                if (mode == 0) sample_tokens(row, 1, V, nullptr, 0, sp, d_out, (void*) 0);
                else sample_draft(row, 1, V, sp, d_q, d_out, (void*) 0);
                if (pass == 0) {
                    int o = 0;
                    cudaMemcpy(&o, d_out, 4, cudaMemcpyDeviceToHost);
                    h = fnv(&o, 4, h);
                    if (mode > 0) {
                        cudaMemcpy(ls.data(), d_q, sizeof(SpecList), cudaMemcpyDeviceToHost);
                        h = fnv(&ls[0].n, 4, h);
                        h = fnv(ls[0].ids, (size_t) ls[0].n * 4, h);
                        h = fnv(ls[0].p, (size_t) ls[0].n * 8, h);
                    }
                }
            }
            cudaEventRecord(e1, 0);
            cudaEventSynchronize(e1);
            if (pass == 1) {
                float ms = 0;
                cudaEventElapsedTime(&ms, e0, e1);
                std::printf("%-13s %8.1f us a row\n", mode == 0 ? "sample_tokens" : "sample_draft",
                            1000.0 * ms / iters);
            }
        }
    }
    std::printf("hash %016llx\n", (unsigned long long) h);
    return 0;
}
