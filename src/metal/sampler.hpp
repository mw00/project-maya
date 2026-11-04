// src/metal/sampler.hpp - maya-metal's token pick, the CUDA engine's sampler chain on the host (src/kernels/cuda/
// sampler.cu, its reference in src/kernels/sampler_parity.cpp): logit bias, penalties, top_k (ties to the lower id),
// top_p over the top_k list in double, min_p, temperature, one Philox draw at (seed, counter).  Greedy: the argmax of
// the biased, penalized logits.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace maya::metal {

struct Sampling {
    bool greedy = true;             // no sampler key: greedy (generate.cpp o.greedy)
    int top_k = 20;
    float top_p = 0.95f;
    float min_p = 0.0f;
    float temperature = 1.0f;
    int min_keep = 1;
    uint64_t seed = 0;
    uint64_t counter = 0;
    int pen_n = 0;
    float pen_repeat = 1.0f, pen_freq = 0.0f, pen_present = 0.0f;
    std::vector<std::pair<int32_t, float>> bias;
};

float philox_uniform(uint64_t seed, uint64_t counter);
// hist: the tokens the state was fed (the prompt, then what was written); the window is its last pen_n
int32_t pick(const std::vector<float>& logits, const Sampling& p, const std::vector<int32_t>& hist);

}  // namespace maya::metal
