#pragma once

namespace strata::ds4 {

// A spare is publishable only after the source read and every member's transfer (including its wait) succeeded.
// Callers own all buffers until this returns; failure never means that the spare contains a complete expert.
template<class Read, class Copy>
bool prepare_cache_swap(int members, Read&& read, Copy&& copy) {
    if (!read()) return false;
    for (int member = 0; member < members; ++member)
        if (!copy(member)) return false;
    return true;
}

// Retire a failed completion at the same between-window point as a successful one. The active mappings and both
// spares remain owned by their original experts; only the pending reservation is released, allowing a later retry.
inline bool discard_failed_cache_swap(bool ready, bool& pending, int& incoming, int& inflight) {
    if (ready) return false;
    pending = false;
    incoming = -1;
    --inflight;
    return true;
}

}  // namespace strata::ds4
