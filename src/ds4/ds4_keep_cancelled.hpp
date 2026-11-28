// src/ds4/ds4_keep_cancelled.hpp - maya-ds4: a prompt read that a STOP ends keeps what it read
// (MAYA_DS4_KEEP_CANCELLED, on by default; 0: the state goes back to where the read started, as before).
//
// A client that gives up on a long read (a proxy or an agent with a timeout) and sends the same prompt again found
// the conversation back where the stopped read began, and read the whole prompt again - and again, if it gave up at
// the same point (GLM's engine had the same until MAYA_GLM_KEEP_CANCELLED, #573).  A STOP now ends the run of chunks
// the way a waiting request's yield does (Model::set_yield): at a chunk every device reaches, the chunks read kept.
// The conversation then stands part way through the prompt.
//
// That state is not the one a whole read leaves there: a long prompt's skip (MAYA_DS4_PROMPT_SKIP) runs layers 20-39
// only on the last ~2,700 rows of the stretch the read was planned to (the next restore point, or the prompt's end:
// Model::feed's `after`), so those layers never saw the rows near the stop.  A read that goes on to that stretch's
// end at least gives what the whole read gives (every row those layers need is in it); one that ends sooner would read
// rows they never saw.  So the part-way state remembers the stretch (`need`): only a prompt whose read reaches it goes
// on from it, and it is never a checkpoint, a restore point or a kept conversation (slot).
//
// Pure bookkeeping, no GPU: tests/ds4_keep_cancelled_test.cpp.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace maya::ds4 {

// a conversation's state that a STOP left part way through a prompt: `at` tokens (0: not part way), read with the plan
// of a stretch that ends at `need`
struct PartWay {
    size_t at = 0;
    size_t need = 0;
};

// a prompt of n tokens that continues the part-way state token for token reads on from it when its read reaches
// `need` (its restore points in [at, need) left out: drop_cuts) and it has a token past `at` to read
inline bool part_way_resumes(const PartWay& p, size_t n) { return p.at > 0 && n > p.at && n >= p.need; }

// the restore points of a prompt that reads on from a part-way state: none in [at, need) - the state at `at` is not a
// whole one, and the read's first stretch has to reach `need`
template <class Cut>
void part_way_cuts(std::vector<Cut>& cuts, const PartWay& p) {
    cuts.erase(std::remove_if(cuts.begin(), cuts.end(),
                              [&](const Cut& c) { return c.pos >= (int64_t) p.at && c.pos < (int64_t) p.need; }),
               cuts.end());
}

// a run's pace from its progress calls (the tokens done so far in the run, every 50 ms): the time between the last
// two times it moved - a chunk through every layer, or a layer through every chunk (a run that goes layer by layer,
// which cannot end at a chunk).  Not moved yet: as long as the run has taken (its first chunk takes that long at least)
struct RunPace {
    int64_t last = -1;
    double t0 = -1, moved = -1, gap = 0;
    void see(int64_t done, double now_ms) {
        if (t0 < 0 || done < last) {   // (a new run)
            t0 = moved = now_ms;
            last = done;
            gap = 0;
            return;
        }
        if (done > last) {
            gap = now_ms - moved;
            moved = now_ms;
            last = done;
        }
    }
    double ms(double now_ms) const { return gap > 0 ? gap : t0 >= 0 ? now_ms - t0 : 0; }
};

// a STOP that has waited `waited_ms` for the run to end at a chunk (keeping what it read) cuts the run short now (the
// run lost; the latest whole state before it stays): past 2.5 times the run's pace, 20 s at least - a chunk takes a
// few seconds warm and ~40 s while a cold Engram is read from a network disk, and a run that has not moved yet waits
// for its first chunk
inline bool stop_cuts_now(double waited_ms, double pace_ms) { return waited_ms > std::max(20000.0, 2.5 * pace_ms); }

// the pictures the state holds once a read of a prompt stops at `at` tokens: the ones it held before the prompt
// (`held`) that start before `at`, the prompt's own (`pics`) aside, and the prompt's that start before `at` - one cut
// by `at` too, with its whole span: the read that goes on feeds the rest of it, its rows by position (the same picture
// at the same place).  At a whole state (a restore point) no span is cut.
template <class Pic>
std::vector<Pic> pics_at(const std::vector<Pic>& held, const std::vector<Pic>& pics, size_t at) {
    std::vector<Pic> out;
    for (const Pic& x : held) {
        bool mine = false;
        for (const Pic& q : pics) mine = mine || (q.start == x.start && q.hash == x.hash);
        if (!mine && x.len > 0 && x.start >= 0 && (size_t) x.start < at) out.push_back(x);
    }
    for (const Pic& q : pics)
        if (q.start >= 0 && (size_t) q.start < at) out.push_back(q);
    return out;
}

}  // namespace maya::ds4
