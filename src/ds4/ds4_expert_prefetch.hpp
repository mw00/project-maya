// Expert read-ahead configuration, cached at the first use in each process.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace strata::ds4 {

/// Expert prefetch uses zero for unconditional hints, or a positive number of page probes per range.
struct ExpertPrefetchPolicy {
    int samples;
    const char* source;
};

/// Parse the effective prefetch count; the legacy force-all override takes precedence over sampling.
inline ExpertPrefetchPolicy parse_expert_prefetch_policy(const char* samples, const char* legacy_all) {
    if (legacy_all != nullptr && std::atoi(legacy_all) != 0) return {0, "DS4_PF_ALL"};
#if defined(_WIN32)
    constexpr int default_samples = 8;
#else
    constexpr int default_samples = 0;
#endif
    if (samples == nullptr) return {default_samples, "default"};
    int count = 0;
    for (const char* at = samples; *at != '\0'; ++at) {
        if (*at < '0' || *at > '9')
            throw std::invalid_argument("DS4_PF_SAMPLES: expected an integer from 0 through 16");
        count = count * 10 + (*at - '0');
        if (count > 16)
            throw std::invalid_argument("DS4_PF_SAMPLES: expected an integer from 0 through 16");
    }
    if (*samples == '\0')
        throw std::invalid_argument("DS4_PF_SAMPLES: expected an integer from 0 through 16");
    return {count, "DS4_PF_SAMPLES"};
}

/// Read and log the effective expert prefetch count once, before model loading in maya-ds4.
inline const ExpertPrefetchPolicy& expert_prefetch_policy() {
    static const ExpertPrefetchPolicy policy = [] {
        const auto value = parse_expert_prefetch_policy(std::getenv("DS4_PF_SAMPLES"), std::getenv("DS4_PF_ALL"));
        std::fprintf(stderr, "ds4: expert prefetch samples=%d source=%s (0=always hint)\n", value.samples, value.source);
        return value;
    }();
    return policy;
}

}  // namespace strata::ds4
