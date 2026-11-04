// src/mm3/mm3_slots.hpp - which idle conversation (--parallel) a new request takes: its cache is kept up to the
// request's shared start and the rest dropped.
#pragma once

#include <cstddef>
#include <vector>

namespace mm3 {

/// One conversation as the choice sees it: busy, the request's shared start with its cache (`from`, the tokens not
/// read again), the tokens in its cache (`size`), and when it last ran (`used`, a clock).
struct SlotView {
    bool busy = false;
    std::size_t from = 0;
    std::size_t size = 0;
    unsigned long long used = 0;
};

/// A shared start this long always counts, whatever it drops (an agent's system prompt, a document asked about again)
constexpr std::size_t kSlotKeepReuse = 512;

/// The conversation a request takes, or -1 (all busy): the most of the request it reuses, then one that drops
/// nothing (empty, or continued), then the least recently used.  A short shared start (under kSlotKeepReuse tokens and
/// under half of what it would drop) counts as none: a short text question shares only the chat template's ~186
/// tokens with a long conversation about a picture, and taking that one's cache read the picture again at its next
/// turn - an empty conversation, else the least recently used, is taken instead.  A continued one counts in full.
inline int pick_slot(const std::vector<SlotView>& v) {
    int best = -1;
    std::size_t b_use = 0;
    bool b_keep = false;
    for (int s = 0; s < (int) v.size(); ++s) {
        const SlotView& x = v[(std::size_t) s];
        if (x.busy) continue;
        const std::size_t drop = x.size > x.from ? x.size - x.from : 0;
        const std::size_t use = drop == 0 || x.from >= kSlotKeepReuse || 2 * x.from >= drop ? x.from : 0;
        const bool keep = drop == 0;
        if (best < 0 || use > b_use ||
            (use == b_use && (keep != b_keep ? keep : x.used < v[(std::size_t) best].used))) {
            best = s;
            b_use = use;
            b_keep = keep;
        }
    }
    return best;
}

}  // namespace mm3
