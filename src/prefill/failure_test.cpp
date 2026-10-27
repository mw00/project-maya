// src/prefill/failure_test.cpp - the allocation failures the recovery recognizes (no GPU, no model).
#include "strata/prefill/failure.hpp"

#include <cstdio>
#include <string>

int main() {
    int fail = 0;
    // the messages the engine's own failures carry
    const char* alloc[] = {
        "prefill mmq: the scratch pool wanted 2 MB and could not have it: out of memory",
        "prefill: device buffers for a chunk of 2048 tokens do not fit",
        "glm prefill: the layout of a chunk of 2048 tokens does not fit the lendable region",
        "prefill: cudaMalloc(1024) failed: out of memory",
        "cudaErrorMemoryAllocation",
        "prefill mmq: there is not enough memory for the scratch",
    };
    for (const char* m : alloc)
        if (!strata::prefill::is_allocation_failure(m)) {
            std::fprintf(stderr, "not seen as an allocation failure: %s\n", m);
            ++fail;
        }
    // the failures that are not about memory: a smaller chunk does not help with these
    const char* other[] = {
        "prefill: the stream failed",
        "glm router: layer 7 decode invalid expert ID or non-finite score",
        "verify: layer 31 never rang",
        "cancelled",
        "prefill gemm: cuBLAS status 1",
    };
    for (const char* m : other)
        if (strata::prefill::is_allocation_failure(m)) {
            std::fprintf(stderr, "seen as an allocation failure: %s\n", m);
            ++fail;
        }
    // the Failure carries its reason
    try {
        strata::prefill::fail("prefill: out of memory");
        std::fprintf(stderr, "fail() did not throw\n");
        ++fail;
    } catch (const strata::prefill::Failure& e) {
        if (std::string(e.what()) != "prefill: out of memory") {
            std::fprintf(stderr, "the reason was lost: %s\n", e.what());
            ++fail;
        }
    }
    std::printf("prefill_failure_test: %s\n", fail ? "FAILED" : "OK");
    return fail;
}
