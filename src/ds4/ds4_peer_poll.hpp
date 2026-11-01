#pragma once

#include <chrono>

namespace strata::ds4 {
enum class PeerPoll { ready, pending, error };

// Sampling may have expired with a collective in flight. Poll that work against
// its own drain deadline, never the sampling deadline. Driver calls themselves
// cannot be interrupted. Query must report every submitted stream as ready
// before the caller may validate, reset flags or release their buffers.
template<class Query, class Now, class Pause>
bool drain_peer_probe(Query query, Now now, Pause pause,
                      std::chrono::steady_clock::duration budget = std::chrono::milliseconds(2500)) {
    const auto deadline = now() + budget;
    for (;;) {
        const auto state = query();
        if (state == PeerPoll::ready) return true;
        if (state == PeerPoll::error || now() >= deadline) return false;
        pause();
    }
}
} // namespace strata::ds4
