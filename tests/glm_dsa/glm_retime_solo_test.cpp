// CPU-only unit tests for include/maya/core/glm_retime_solo.hpp (no CUDA or ggml needed).
#include "maya/core/glm_retime_solo.hpp"

#undef NDEBUG
#include <cassert>
#include <cstdint>
#include <cstdio>

int main() {
    using maya::core::glmretime::is_solo_sample;

    // Non-concurrent pool: always uncontended regardless of counters.
    assert(is_solo_sample(false, 0, 0, 0, 0));
    assert(is_solo_sample(false, 1, 2, 10, 99));

    // Concurrent pool: clean solo calls (0, 1, or 2 pool batches, no other slot busy).
    assert(is_solo_sample(true, 0, 0, 100, 100));
    assert(is_solo_sample(true, 0, 0, 100, 101));
    assert(is_solo_sample(true, 0, 0, 100, 102));

    // Concurrent pool: unsigned wraparound across uint64_t::max still yields delta == 2.
    const uint64_t max_u64 = ~uint64_t{0};
    assert(is_solo_sample(true, 0, 0, max_u64, 1));
    assert(!is_solo_sample(true, 0, 0, max_u64, 2));   // delta == 3 -> contended

    // Concurrent pool: busy before, busy after, or extra batch from another GPU mid-call -> rejected.
    assert(!is_solo_sample(true, 1, 0, 100, 102));
    assert(!is_solo_sample(true, 0, 1, 100, 102));
    assert(!is_solo_sample(true, 2, 1, 100, 102));
    assert(!is_solo_sample(true, 0, 0, 100, 103));
    assert(!is_solo_sample(true, 0, 0, 100, 104));

    std::printf("glm_retime_solo_test: PASSED\n");
    return 0;
}
