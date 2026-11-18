// include/strata/core/glm_kv_groups.hpp - KV streaming on glm-dsa (GLM-5.3 744B): which residency map each DSA layer
// of a part uses.  Plain C++ (no CUDA): Glm5Model::kv_stream_make and tests/glm_dsa/glm_kv_groups_test.cpp share it.
//
// glm-dsa's token indexer runs on the "full" layers only; a shared layer attends the cells the full layer before it
// selected (same token, same part - a part starts on a full layer).  The layers of such an indexer group therefore
// name the same blocks token after token, their residency would never differ, and one map serves them all: a resolve
// on the full layer makes the selection resident for every cache of the group, and the shared layers read the same
// slot rows.  The NextN block is a group of its own.
#pragma once

#include <cstddef>
#include <functional>
#include <vector>

namespace strata::core {

/// lead[il] for every layer of `ls` (a part's DSA layers in Glm5Model::dsa_layers() order: the NextN block first when
/// the part carries it, then its trunk layers ascending): the layer whose map layer il uses - itself (a map of its
/// own), or the full-indexer layer that leads its group.  Layers not in `ls` get -1.  n_layers: the trunk's layers (a
/// layer index >= n_layers is the NextN block).  grouped = false: every layer leads itself.  per_set: the most layers
/// one map serves (what one copy launch carries); a longer group continues on another map, whose lead is then a
/// shared layer that resolves for itself.  n_maps: the maps to allocate.
inline std::vector<int> glm_kv_leads(const std::vector<int>& ls, int n_layers, const std::function<bool(int)>& full,
                                     bool grouped, int per_set, size_t& n_maps) {
    std::vector<int> lead((size_t) n_layers + 1, -1);
    n_maps = 0;
    int cur = -1, members = 0;
    for (const int il : ls) {
        if (il < 0 || il > n_layers) continue;
        const bool own = !grouped || il >= n_layers || full(il) || cur < 0 || cur >= n_layers || il != cur + members ||
                         members >= per_set;
        if (own) {
            cur = il;
            members = 0;
            ++n_maps;
        }
        lead[(size_t) il] = cur;
        ++members;
    }
    return lead;
}

// ---- the identity window (KvStream::window / ident): after a reset block b sits in slot b.  The three rules that
// decide when a reader may use the slots as an identity cache of `cells` positions (the VRAM positions a layer keeps):

/// A decode attention at positions up to p_hi reads its cells as slot rows, with no resolve.
inline bool glm_kv_window_serves(bool ident, long long p_hi, long long cells) { return ident && p_hi < cells; }
/// A prompt chunk of T tokens at p0 runs on the slots (no stage) and leaves the maps as they are.
inline bool glm_kv_window_chunk(bool ident, long long p0, long long T, long long cells) {
    return ident && p0 >= 0 && p0 + T <= cells;
}
/// A conversation of n positions loaded into the host copies goes into the window (its rows copied to the slots).
inline bool glm_kv_window_load(bool window, long long n, long long cells) { return window && n <= cells; }

}  // namespace strata::core
