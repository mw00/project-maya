#pragma once

#include <algorithm>
#include <string>
#include <vector>

namespace strata::core {

// The devices of a GLM layer split, in part order: `named` (STRATA_GLM_DEVS, when given) or every visible device
// in order, without the expert helper's (`helper`, -1: none).  `n_parts` > 0 (pinned boundaries) takes that many;
// 0 (`auto`) takes up to `max_parts`.  False with `err` when the named list repeats a device, names the helper,
// or has fewer devices than the parts need; a split of one device is the caller's single-device case.
inline bool glm_split_devices(const std::vector<int>* named, int n_dev, int helper, int n_parts, int max_parts,
                              std::vector<int>& out, std::string& err) {
    out.clear();
    if (named != nullptr) {
        for (size_t i = 0; i < named->size(); ++i) {
            const int d = (*named)[i];
            if (d == helper) {
                err = "GLM expert helper: CUDA" + std::to_string(d) + " is the helper, not a part of the split (MAYA_GLM_DEVS)";
                return false;
            }
            if (std::find(named->begin(), named->begin() + (long) i, d) != named->begin() + (long) i) {
                err = "glm split: MAYA_GLM_DEVS names CUDA" + std::to_string(d) + " twice";
                return false;
            }
        }
        out = *named;
    } else {
        for (int d = 0; d < n_dev && (int) out.size() < (n_parts > 0 ? n_parts : max_parts); ++d)
            if (d != helper) out.push_back(d);
    }
    if (n_parts > 0 && (int) out.size() < n_parts) {
        err = "glm split: " + std::to_string(n_parts - 1) + " boundaries need " + std::to_string(n_parts) +
              " devices, " + std::to_string(out.size()) + " available" + (helper >= 0 ? " beside the expert helper" : "");
        return false;
    }
    return true;
}

} // namespace strata::core
