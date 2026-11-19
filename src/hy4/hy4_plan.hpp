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

/// What a start's plan comes to with one prompt chunk: whether it fits at all, the GPUs in tensor parallel (0: the
/// layer split), the cache's rows, a picture's room and the largest picture it holds.
struct ChunkTry {
    bool ok = false;
    int tp = 0;
    bool q8 = false;
    bool vision = false;
    long long pixels = 0;
};

/// The prompt chunk of a start that names none (--chunk auto): the largest of `cands` (largest first) whose plan
/// `plan_at(chunk)` comes to what the plan at `base` comes to in everything that goes before the prompts' speed - the
/// same layout (tensor parallel over as many GPUs, or the layer split), the same rows of the cache, pictures as large.
/// A larger chunk never costs a picture, a context or tensor parallel: where it would, the next smaller one is tried,
/// and `base` stays where none does (or where the plan at `base` does not fit: that start's own error).
template <class PlanAt, class Cands>
int chunk_pick(PlanAt&& plan_at, const Cands& cands, int base) {
    const ChunkTry ref = plan_at(base);
    if (!ref.ok) return base;
    for (int c : cands) {
        if (c <= base) continue;
        const ChunkTry t = plan_at(c);
        if (t.ok && t.tp == ref.tp && t.q8 == ref.q8 && t.vision == ref.vision && t.pixels >= ref.pixels) return c;
    }
    return base;
}

/// The bytes of the MTP block's own rows for a prompt chunk of `chunk` tokens of `n_embd` values: its input (two
/// states a token), its residual, the hidden states it reads and its output - five F32 rows a token.
inline double mtp_rows_bytes(long long chunk, long long n_embd) { return (double) chunk * (double) n_embd * 5 * 4; }

}  // namespace hy4
