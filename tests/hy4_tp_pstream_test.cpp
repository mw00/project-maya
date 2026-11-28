// Host CPU-only unit tests for HY4_TP_PSTREAM coalesced pinned-pool DMA planner
// (src/hy4/hy4_tp_pstream.hpp; no GPU).
#include "../src/hy4/hy4_tp_pstream.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(expr)                                                                     \
    do {                                                                                \
        if (!(expr)) {                                                                  \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr);                    \
            return 1;                                                                   \
        }                                                                               \
    } while (0)

namespace {

int verify_coalesced_transfer(const std::vector<int>& req, const std::vector<int>& owned,
                              int stage_base, int capacity, size_t blob, size_t expected_runs) {
    constexpr int kExperts = 32;
    std::array<int, kExperts> ps_of{};
    ps_of.fill(-1);
    const int ps_m = static_cast<int>(owned.size());
    std::vector<std::uint8_t> ps_mem(static_cast<size_t>(ps_m) * blob);
    std::vector<std::uint8_t> reference(static_cast<size_t>(kExperts) * blob);
    for (size_t i = 0; i < reference.size(); ++i) {
        const auto e = i / blob, off = i % blob;
        reference[i] = static_cast<std::uint8_t>((e * 31 + off * 17) ^ (off >> 5));
    }
    for (int q = 0; q < ps_m; ++q) {
        ps_of[owned[q]] = q;
        std::memcpy(ps_mem.data() + static_cast<size_t>(q) * blob,
                    reference.data() + static_cast<size_t>(owned[q]) * blob, blob);
    }

    CHECK(static_cast<size_t>(stage_base) + req.size() <= static_cast<size_t>(capacity));
    std::vector<std::uint8_t> gpu_stage(static_cast<size_t>(capacity) * blob, 0);
    std::vector<hy4::PinnedCopyRun> runs;
    CHECK(hy4::coalesce_pinned_runs(req.data(), static_cast<int>(req.size()), ps_of.data(), ps_m, stage_base, runs));
    if (expected_runs > 0) CHECK(runs.size() == expected_runs);

    int copied = 0;
    for (const hy4::PinnedCopyRun& r : runs) {
        CHECK(r.dst_slot == stage_base + copied);
        CHECK(r.src_slot >= 0 && r.src_slot + r.count <= ps_m);
        std::memcpy(gpu_stage.data() + static_cast<size_t>(r.dst_slot) * blob,
                    ps_mem.data() + static_cast<size_t>(r.src_slot) * blob,
                    static_cast<size_t>(r.count) * blob);
        copied += r.count;
    }
    CHECK(copied == static_cast<int>(req.size()));

    for (size_t i = 0; i < req.size(); ++i) {
        const int e = req[i];
        CHECK(std::memcmp(gpu_stage.data() + static_cast<size_t>(stage_base + static_cast<int>(i)) * blob,
                          reference.data() + static_cast<size_t>(e) * blob, blob) == 0);
    }
    return 0;
}

}  // namespace

int main() {
    for (size_t block : {136, 176}) {
        constexpr size_t cols = 1536;
        const size_t expert_bytes = 3 * cols * cols / 256 * block;
        const std::vector<int> owned = {1, 3, 4, 5, 7, 8, 9, 11};
        // In ps_mem (owned slots 0..7): [1,3,4,5]->slots 0..3 and [8,9]->slots 5..6 => 2 coalesced DMA runs.
        CHECK(verify_coalesced_transfer({1, 3, 4, 5, 8, 9}, owned, 0, 8, expert_bytes, 2) == 0);
        // Skipping owned slot 1 (expert 3) and slot 4 (expert 7): [1], [4,5], [8] => 3 coalesced DMA runs.
        CHECK(verify_coalesced_transfer({1, 4, 5, 8}, owned, 0, 8, expert_bytes, 3) == 0);
        // Non-zero stage_base offset and single merged run: [4,5,7,8]->slots 2..5 => 1 DMA run.
        CHECK(verify_coalesced_transfer({4, 5, 7, 8}, owned, 3, 8, expert_bytes, 1) == 0);
    }

    // Negative / guard checks.
    {
        std::array<int, 8> ps_of{};
        ps_of.fill(-1);
        ps_of[2] = 0;
        ps_of[3] = 2;   // out of range for ps_m == 1
        std::vector<hy4::PinnedCopyRun> runs;
        std::vector<int> missing = {2, 5};
        CHECK(!hy4::coalesce_pinned_runs(missing.data(), 2, ps_of.data(), 1, 0, runs));
        std::vector<int> oob = {3};
        CHECK(!hy4::coalesce_pinned_runs(oob.data(), 1, ps_of.data(), 1, 0, runs));
        std::vector<int> ok = {2};
        CHECK(!hy4::coalesce_pinned_runs(ok.data(), -1, ps_of.data(), 1, 0, runs));
        CHECK(!hy4::coalesce_pinned_runs(ok.data(), 1, ps_of.data(), 1, -1, runs));
    }

    std::mt19937 rng(0x11842026);
    for (int iter = 0; iter < 2048; ++iter) {
        std::vector<int> pool(32);
        for (int i = 0; i < 32; ++i) pool[i] = i;
        std::shuffle(pool.begin(), pool.end(), rng);
        const int n_own = 1 + static_cast<int>(rng() % 16);
        std::vector<int> owned(pool.begin(), pool.begin() + n_own);
        std::sort(owned.begin(), owned.end());
        std::vector<int> req;
        for (int e : owned)
            if (rng() & 1) req.push_back(e);
        if (req.empty()) req.push_back(owned[0]);
        const int stage_base = static_cast<int>(rng() % 5);
        const int capacity = stage_base + static_cast<int>(req.size()) + static_cast<int>(rng() % 4);
        CHECK(verify_coalesced_transfer(req, owned, stage_base, capacity, 113, 0) == 0);
    }

    std::printf("hy4_tp_pstream_test: 4 full payloads + negative + 2048 randomized checks passed (memcmp==0)\n");
    return 0;
}
