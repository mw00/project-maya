// Host CPU-only unit tests for HY4_TP_STAGE_REUSE slot bookkeeping (src/hy4/hy4_stage_reuse.hpp; no GPU).
#include "../src/hy4/hy4_stage_reuse.hpp"

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

int check(std::vector<int> old, std::vector<int> request, int capacity, size_t blob) {
    std::array<int, 32> map{};
    std::vector<std::uint8_t> memory(static_cast<size_t>(capacity) * blob);
    std::vector<std::uint8_t> reference(32 * blob);
    for (size_t i = 0; i < reference.size(); ++i) {
        const auto e = i / blob, offset = i % blob;
        reference[i] = static_cast<std::uint8_t>((e * 17 + offset * 23) ^ (offset >> 7));
    }
    hy4::stage_slots_for_copy(map.data(), static_cast<int>(map.size()), old.data(), static_cast<int>(old.size()), 0);
    for (size_t i = 0; i < old.size(); ++i)
        std::memcpy(memory.data() + i * blob, reference.data() + static_cast<size_t>(old[i]) * blob, blob);
    const auto original = request;
    const bool full = std::all_of(request.begin(), request.end(), [&](int e) { return map[e] >= 0; });
    if (!full) {
        const int base = hy4::stage_tail(request, map.data(), static_cast<int>(old.size()), capacity);
        CHECK(static_cast<size_t>(base) + request.size() <= static_cast<size_t>(capacity));
        hy4::stage_slots_for_copy(map.data(), static_cast<int>(map.size()), request.data(),
                                  static_cast<int>(request.size()), base);
        for (size_t i = 0; i < request.size(); ++i)
            std::memcpy(memory.data() + (static_cast<size_t>(base) + i) * blob,
                        reference.data() + static_cast<size_t>(request[i]) * blob, blob);
    }
    for (int expert : original) {
        CHECK(map[expert] >= 0 && map[expert] < capacity);
        CHECK(std::memcmp(memory.data() + static_cast<size_t>(map[expert]) * blob,
                          reference.data() + static_cast<size_t>(expert) * blob, blob) == 0);
    }
    return 0;
}

}  // namespace

int main() {
    for (size_t block : {136, 176}) {
        constexpr size_t cols = 1536;
        const size_t expert_bytes = 3 * cols * cols / 256 * block;
        CHECK(check({1, 5, 7}, {3, 5, 11}, 5, expert_bytes) == 0);   // exact-fit partial append
        CHECK(check({1, 5, 7}, {3, 5, 11}, 4, expert_bytes) == 0);   // capacity fallback
        CHECK(check({1, 5, 7}, {1, 7}, 4, expert_bytes) == 0);       // original full hit
        CHECK(check({1, 5}, {3, 7}, 5, expert_bytes) == 0);          // disjoint fallback
    }
    std::mt19937 rng(0x051928);
    for (int pass = 0; pass < 2048; ++pass) {
        std::vector<int> pool(32);
        for (int i = 0; i < 32; ++i) pool[i] = i;
        const int capacity = 1 + static_cast<int>(rng() % 16);
        std::shuffle(pool.begin(), pool.end(), rng);
        const std::vector<int> initial(pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>(rng() % (capacity + 1)));
        std::shuffle(pool.begin(), pool.end(), rng);
        const std::vector<int> next(pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>(rng() % (capacity + 1)));
        CHECK(check(initial, next, capacity, 97) == 0);
    }
    std::printf("hy4_stage_reuse_test: 8 full payloads + 2048 deterministic checks passed (memcmp==0)\n");
    return 0;
}
