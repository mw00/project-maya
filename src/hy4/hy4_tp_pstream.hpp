#pragma once

#include <cstddef>
#include <vector>

namespace hy4 {

struct PinnedCopyRun {
    int dst_slot;   // offset into gp.stage (including stage_base)
    int src_slot;   // offset into l.ps_mem
    int count;      // contiguous expert blobs
};

// Coalesces requested expert indices into contiguous runs in l.ps_mem so each run
// can be transferred with a single async H2D cudaMemcpyAsync without a bounce buffer.
// Returns false if any requested expert is missing from ps_of (slot < 0).
inline bool coalesce_pinned_runs(const int* experts, int count, const int* ps_of, int ps_m,
                                 int stage_base, std::vector<PinnedCopyRun>& runs) {
    runs.clear();
    if (count < 0 || stage_base < 0) return false;
    int i = 0;
    while (i < count) {
        const int q0 = ps_of[experts[i]];
        if (q0 < 0 || q0 >= ps_m) return false;
        int len = 1;
        while (i + len < count) {
            const int qn = ps_of[experts[i + len]];
            if (qn != q0 + len || qn >= ps_m) break;
            ++len;
        }
        runs.push_back(PinnedCopyRun{stage_base + i, q0, len});
        i += len;
    }
    return true;
}

}  // namespace hy4
