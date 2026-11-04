// src/ds4/ds4_watchdog.hpp - maya-ds4 --serve's watchdog rule (#539, #494): a request has hung when its heartbeat - a
// prompt chunk or window read, a token, a step between them - has not moved for `limit` seconds while it runs.  It
// measures the time since the last beat, never the request's length: a long prompt that keeps reading chunks beats
// with each of them, however long it takes as a whole.  Waiting for a request (not busy) never counts.
#pragma once

#include <atomic>
#include <cstdint>

namespace strata::ds4 {

struct StallRule {
    double limit_s = 600;   // (<= 0: off)
    uint64_t last = 0;      // the heartbeat when it last moved
    double since = 0;       // when (seconds, any monotonic clock)
    bool started = false;
    // one look at `now`: true once the request has gone `limit_s` without a beat
    bool stalled(bool busy, uint64_t beats, double now) {
        if (limit_s <= 0) return false;
        if (!started || !busy || beats != last) {
            started = true;
            last = beats;
            since = now;
            return false;
        }
        return now - since >= limit_s;
    }
};

}  // namespace strata::ds4
