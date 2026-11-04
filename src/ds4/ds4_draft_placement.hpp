#pragma once

#include <algorithm>
#include <climits>
#include <string>
#include <string_view>
#include <vector>

namespace strata::ds4 {

// Device IDs are those printed at startup, not positions in a TP group's member list.
// Return indices into the selected-device vector, so a filtered/noncontiguous selection works too.
inline bool draft_gpu_indices(std::string_view value, int blocks, const std::vector<int>& device_ids,
                              std::vector<int>& indices, std::string& err) {
    indices.clear();
    err.clear();
    auto fail = [&]() {
        indices.clear();
        err = "DS4_DRAFT_GPU requires one selected GPU ID, or one per DSpark block (" +
              std::to_string(blocks) + ")";
        return false;
    };
    if (blocks <= 0 || value.empty()) return fail();
    size_t pos = 0;
    while (pos < value.size()) {
        int id = 0;
        const size_t begin = pos;
        while (pos < value.size() && value[pos] >= '0' && value[pos] <= '9') {
            const int digit = value[pos++] - '0';
            if (id > (INT_MAX - digit) / 10) return fail();
            id = id * 10 + digit;
        }
        if (pos == begin) return fail();
        auto it = std::find(device_ids.begin(), device_ids.end(), id);
        if (it == device_ids.end()) return fail();
        indices.push_back((int) (it - device_ids.begin()));
        if (indices.size() > (size_t) blocks) return fail();
        if (pos == value.size()) break;
        if (value[pos++] != ',' || pos == value.size()) return fail();
    }
    if (indices.size() == 1) indices.resize((size_t) blocks, indices[0]);
    if (indices.size() != (size_t) blocks) return fail();
    return true;
}

}  // namespace strata::ds4
