// maya-hy4's plan, the part with no GPU in it: what --kv auto and the pictures' room come to where the layers are
// split over the GPUs (one GPU too).  The model's own parts - its dense matrices and its caches - have no RAM fallback:
// they fit in VRAM or the engine does not start.  A start that fits with F16 cache rows and the room of the tower's
// largest picture (or with F16 rows where pictures are off) is never changed.
#pragma once

namespace hy4 {

struct KvPick {
    bool fits = false;       // the model's own parts fit in VRAM this way (false: nothing fits - the plan's error)
    bool q8 = false;         // the latent cache in 8-bit rows
    bool vision = false;     // a picture's room kept on the main GPU
    long long patches = 0;   // ... for a picture of this many patches at most (the tower's largest, or fewer)
};

/// What is tried, in order, and the first that fits taken - `fits(q8, room)`: the model's own parts fit in VRAM with
/// the cache in 8-bit rows (or F16) and `room` bytes kept free on the main GPU for a picture (0: none).  Pictures go
/// before the rows' precision, a picture of the tower's largest size before a smaller one:
///   1. as asked: F16 rows (8-bit with --kv q8) and the room of the largest picture (`n_max` patches: `room(n_max)`);
///   2. 8-bit rows (--kv auto only: `may_q8`) and that room;
///   3. F16 rows and the room of a smaller picture - the most patches that fit, `n_min` at least (the pictures are
///      then made smaller before they are read);
///   4. 8-bit rows (`may_q8`) and the room of a smaller picture;
///   5. F16 rows and no room: text only - only where pictures were not asked for by name (`vision_auto`);
///   6. 8-bit rows and no room (`may_q8` and `vision_auto`).
/// `q8`: --kv q8 (forced: the F16 steps are then its own, the 8-bit ones fall away).  `vision` false: 5 and 6 alone.
/// Nothing fits: {false, q8, vision && !vision_auto}.
template <class Fits, class Room>
KvPick kv_pick(Fits&& fits, bool q8, bool may_q8, bool vision, bool vision_auto, long long n_max, long long n_min,
               Room&& room) {
    const bool try_q8 = !q8 && may_q8;
    if (vision) {
        if (fits(q8, room(n_max))) return {true, q8, true, n_max};
        if (try_q8 && fits(true, room(n_max))) return {true, true, true, n_max};
        for (int step = 0; step < (try_q8 ? 2 : 1) && n_min > 0 && n_min < n_max; ++step) {
            const bool r = step ? true : q8;
            if (!fits(r, room(n_min))) continue;
            long long lo = n_min, hi = n_max;          // (lo fits, hi does not)
            while (hi - lo > 64) {
                const long long mid = lo + (hi - lo) / 2;
                if (fits(r, room(mid))) lo = mid;
                else hi = mid;
            }
            return {true, r, true, lo};
        }
        if (!vision_auto) return {false, q8, true, n_max};
    }
    if (fits(q8, 0.0)) return {true, q8, false, 0};
    if (try_q8 && fits(true, 0.0)) return {true, true, false, 0};
    return {false, q8, false, 0};
}

}  // namespace hy4
