// Private, host-only scheduling for GGML Prefill's CPU weight-row blocks.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace strata::core::glmfast {

int prefill_row_node();

// Stored with one live RAM class, set only for a successful row-halves mapping.
struct RowHalvesPlacement {
    std::size_t gate_bytes = 0, down_offset = 0, blob_bytes = 0;
    int low_node = -1, high_node = -1;

    bool matches(std::size_t gate, std::size_t down, std::size_t blob) const {
        return low_node >= 0 && high_node >= 0 && low_node != high_node &&
               gate_bytes == gate && down_offset == down && blob_bytes == blob;
    }
};

inline bool row_halves_has_slot(const void* slot, const void* base, std::size_t stride, std::size_t count) {
    if (slot == nullptr || base == nullptr || stride == 0) return false;
    const auto address = reinterpret_cast<std::uintptr_t>(slot);
    const auto start = reinterpret_cast<std::uintptr_t>(base);
    if (address < start) return false;
    const auto offset = address - start;
    return offset % stride == 0 && offset / stride < count;
}

inline bool row_halves_fit_blocks(int rows, int block) {
    return rows > 0 && rows % (2 * block) == 0;
}

// Like the one-token lane's row-halves queues: one ticket per job, prefer the worker's half,
// then steal from the other half. Keep the original flat job IDs and an unsplit row block.
// Workers::run joins the batch, so both queues live only for this call.
template<class Pool, class Node, class Job>
void run_prefill_row_blocks(Pool& pool, int experts, int rows, int block, int low_node,
                            const Node& node, const Job& job) {
    const int blocks = rows / block;
    if (low_node < 0 || !row_halves_fit_blocks(rows, block)) {
        pool.run(experts * blocks, job);
        return;
    }
    const int jobs_per_half = experts * (blocks / 2);
    alignas(64) std::atomic<int> next[2]{};
    pool.run(2 * jobs_per_half, [&](int) {
        const int local = node() == low_node ? 0 : 1;
        for (int attempt = 0; attempt < 2; ++attempt) {
            const int half = attempt == 0 ? local : 1 - local;
            const int index = next[half].fetch_add(1, std::memory_order_acq_rel);
            if (index >= jobs_per_half) continue;
            job(half * jobs_per_half + index);
            return;
        }
    });
}

}  // namespace strata::core::glmfast
