#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <new>
#include <span>
#include <vector>

namespace strata::core::glm_prefix {

inline constexpr size_t min_tokens = 1024;
inline constexpr size_t default_states = 16;
inline constexpr size_t max_states = 512;
inline constexpr uint64_t default_state_budget = uint64_t{2} << 30;
inline constexpr uint64_t state_budget = uint64_t{64} << 30;  // maximum individual wire state
inline constexpr size_t history_paths = 8;
inline constexpr size_t max_path_tokens = 1u << 20;
inline constexpr size_t max_state_tokens = max_path_tokens;
inline constexpr size_t retained_token_budget = 8u << 20;     // 32 MiB across learned slot keys

inline bool matches(std::span<const int32_t> prefix, std::span<const int32_t> prompt) {
    return !prefix.empty() && prefix.size() < prompt.size() &&
           std::equal(prefix.begin(), prefix.end(), prompt.begin());
}

// A bounded ring of exact witnesses. Only a mismatch with tokens on both sides
// is a fork. Materialized states belong to the slot catalogue, not this ring.
class Admission {
    std::vector<int32_t> paths_[history_paths];
    size_t next_ = 0;

public:
    size_t observe(std::span<const int32_t> prompt, size_t ancestor, size_t minimum = min_tokens) {
        if (prompt.size() <= minimum || prompt.size() > max_path_tokens) return 0;
        size_t best = 0;
        bool seen = false;
        for (const auto& old : paths_) {
            const auto [a, b] = std::mismatch(old.begin(), old.end(), prompt.begin(), prompt.end());
            if (a == old.end() && b == prompt.end()) seen = true;
            if (a == old.end() || b == prompt.end()) continue;
            const auto p = static_cast<size_t>(b - prompt.begin());
            if (p >= minimum && p <= max_state_tokens && p > ancestor && p - ancestor >= min_tokens)
                best = std::max(best, p);
        }
        if (!seen) {
            try {
                auto& path = paths_[next_];
                // Release the previous witness before replacing it so the ring never needs a ninth path.
                if (path.capacity() < prompt.size()) std::vector<int32_t>().swap(path);
                path.assign(prompt.begin(), prompt.end());
                next_ = (next_ + 1) % history_paths;
            } catch (const std::bad_alloc&) {
                return 0;
            }
        }
        return best;
    }
};

}  // namespace strata::core::glm_prefix
