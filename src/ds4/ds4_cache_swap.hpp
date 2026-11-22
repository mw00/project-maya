#pragma once

namespace maya::ds4 {

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
// A device's table inputs between windows (tick_flush): `slots` host-mapped inputs used in turn, each with its own
// event, recorded on the layer stream right after the tab_set that reads it.  The input taken next is overwritten
// only once its own last reader is done: `done(k)` (the event's query: 0 done or never recorded, 1 not yet, -1 an
// error), then, not yet, `wait(k)` (the event's wait: false an error).  A tick after a window finds that tab_set long
// done (the window ran after it on the same stream), so the host waits for nothing - where waiting for the whole
// stream (v1.1.69) put a host wait on a GPU the window's own end never waited for (#1086, #1027).  -1: an error.
template<class Done, class Wait>
int take_table_input(int& next, int slots, Done&& done, Wait&& wait) {
    const int k = next;
    next = (k + 1) % slots;
    const int q = done(k);
    if (q == 0) return k;
    if (q == 1 && wait(k)) return k;
    return -1;
}

inline bool discard_failed_cache_swap(bool ready, bool& pending, int& incoming, int& inflight) {
    if (ready) return false;
    pending = false;
    incoming = -1;
    --inflight;
    return true;
}

}  // namespace maya::ds4
