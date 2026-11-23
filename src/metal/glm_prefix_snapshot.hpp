#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace maya::metal {
// GLM-5.3 tokenizer/vocab.json special-token IDs, not text matching. Other
// endings keep the upstream n-1 behavior, including PAUSE continuations.
inline size_t glm_snapshot_target(const std::vector<int32_t>& prompt) {
    constexpr int32_t assistant = 154828, think = 154841;
    const size_t n = prompt.size();
    if (n >= 2 && prompt[n - 2] == assistant && prompt[n - 1] == think) return n - 2;
    return n ? n - 1 : 0;
}
}
