// maya-hy4's plan, the part with no GPU in it: what --kv auto and the pictures' room come to where the layers are
// split over the GPUs (one GPU too).  The model's own parts - its dense matrices and its caches - have no RAM fallback:
// they fit in VRAM or the engine does not start.  A start that fits with F16 cache rows is never changed.
#pragma once

namespace hy4 {

struct KvPick {
    bool fits = false;     // the model's own parts fit in VRAM this way (false: nothing fits - the plan's error)
    bool q8 = false;       // the latent cache in 8-bit rows
    bool vision = false;   // a picture's room kept on the main GPU
};

/// What is tried, in order, and the first that fits taken - `fits(q8, vision)`: the model's own parts fit in VRAM with
/// the cache in 8-bit rows (or F16) and a picture's room kept (or not):
///   1. as asked: F16 rows (8-bit with --kv q8), the picture's room where pictures are on;
///   2. F16 rows without the picture's room - only where pictures were not asked for by name (`vision_auto`);
///   3. 8-bit rows (--kv auto only: `may_q8`) with the picture's room;
///   4. 8-bit rows without it (`vision_auto`).
/// `q8`: --kv q8 (forced: 2 is then the only step left).  Nothing fits: {false, q8, vision && !vision_auto}.
template <class Fits>
KvPick kv_pick(Fits&& fits, bool q8, bool may_q8, bool vision, bool vision_auto) {
    const bool drop = vision && vision_auto;
    if (fits(q8, vision)) return {true, q8, vision};
    if (drop && fits(q8, false)) return {true, q8, false};
    if (!q8 && may_q8) {
        if (fits(true, vision)) return {true, true, vision};
        if (drop && fits(true, false)) return {true, true, false};
    }
    return {false, q8, vision && !vision_auto};
}

}  // namespace hy4
