// maya-hy4's plan, the part with no GPU in it: what --kv auto and the pictures' room come to where the layers are
// split over the GPUs (one GPU too).  The model's own parts - its dense matrices and its caches - have no RAM fallback:
// they fit in VRAM or the engine does not start.  A start that fits with F16 cache rows and the room of the tower's
// largest picture (or with F16 rows where pictures are off) is never changed.
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace hy4 {

/// A MoE layer's routed experts in VRAM (the INFO line's vram_slots, the Monitor's "Where the experts live"): those a GPU
/// holds - whole (the layer split: the layer's own slot map) or a slice of (tensor parallel: each member's and each
/// helper's map; every expert has its blocks on some member, so all of them) - the others the CPU lane's.  `maps`:
/// each holder's slot map for the layer (an expert's slot, -1 or past the map's end: not held there; null: none).
inline int experts_held(int n_expert, const std::vector<const std::vector<int>*>& maps) {
    int n = 0;
    for (int e = 0; e < n_expert; ++e)
        for (const std::vector<int>* m : maps)
            if (m && e < (int) m->size() && (*m)[(size_t) e] >= 0) {
                ++n;
                break;
            }
    return n;
}

struct KvPick {
    bool fits = false;       // the model's own parts fit in VRAM this way (false: nothing fits - the plan's error)
    bool q8 = false;         // the latent cache in 8-bit rows
    bool vision = false;     // a picture's room kept on the main GPU
    long long patches = 0;   // ... for a picture of this many patches at most (the tower's largest, or fewer)
};

/// The VRAM a start gives up on the main GPU for a picture whose encode takes `room` bytes: none where the head
/// GPU's prompt buffers (`lend` bytes of scratch in one block, idle while a picture is read - a picture comes before
/// its prompt) hold it - the tower works in them and nothing of the plan is given up for pictures; where they are
/// smaller, only what is missing (the block is made that much longer).  kv_pick's `room` is this: with every
/// picture's room lent, its steps with a picture are its steps without one.
inline double vision_keep(double room, double lend) { return room <= lend ? 0.0 : room - lend; }

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
    bool own = false;        // tensor parallel: the MTP block's pass on a scratch of its own (dense_pick's alone)
    bool head = false;       // tensor parallel: the output head's rows over the members too (dense_pick's alone)
    bool lane = false;       // tensor parallel: a part of the routed experts on the CPU lane (the group holds the rest)
    long long res = 0;       // the routed experts the plan keeps in VRAM, over the layers (dense_pick's alone)
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
        if (t.ok && t.tp == ref.tp && t.q8 == ref.q8 && t.vision == ref.vision && t.pixels >= ref.pixels &&
            (t.lane == ref.lane || ref.lane))
            return c;
    }
    return base;
}

/// The bytes of the MTP block's own rows for a prompt chunk of `chunk` tokens of `n_embd` values: its input (two
/// states a token), its residual, the hidden states it reads and its output - five F32 rows a token.
inline double mtp_rows_bytes(long long chunk, long long n_embd) { return (double) chunk * (double) n_embd * 5 * 4; }

/// The indexed layers' dense prompt path (S = Q K^T over every key a sub-batch of `sub` tokens may see, on the tensor
/// cores, the indexer's picks kept by a mask): what its buffers come to for a span of `dense` key positions at most.
/// The engine's plan counts them and its scratch allocates them from these same three, so a span the plan took is a
/// span the GPUs hold.
/// - the scores and the probabilities (an F32 and an F16 value each): the sub-batch's rows (a token's `H` heads, 32
///   counted at most as before) times the keys - a window layer's (`n_swa` + the sub-batch) or the span's;
inline size_t dense_att_values(size_t sub, size_t H, size_t n_swa, size_t dense) {
    return std::max(sub * H * (n_swa + sub + 8), sub * std::min<size_t>(H, 32) * dense);
}
/// - the cache's F16 rows a layer's sub-batches read in order (an 8-bit cache's dequantized there): a window's and a
///   chunk's, or the span's and a sub-batch's;
inline size_t dense_kbuf_rows(size_t n_swa, size_t chunk, size_t dense, size_t sub) {
    return std::max(n_swa + chunk, dense + sub);
}
/// - and the mask (an F32 a key a token).  All of it, in bytes, for a latent row of `kv_row` values:
inline size_t dense_bytes(size_t sub, size_t H, size_t n_swa, size_t chunk, size_t dense, size_t kv_row) {
    return dense_att_values(sub, H, n_swa, dense) * 6 + sub * dense * 4 + dense_kbuf_rows(n_swa, chunk, dense, sub) * kv_row * 2;
}

/// The span of the dense prompt path a start takes: the largest of `cands` (largest first) whose plan `plan_at(span)`
/// comes to what the plan at `base` comes to - the same layout, the same rows of the cache, pictures as large, and
/// in tensor parallel the MTP block's pass still on its own scratch and the output head still over the members (each
/// token's head otherwise on the head's GPU alone: slower decoding), and as many routed experts in VRAM (a card that
/// holds a part of the experts would give some up for the span's buffers: other experts on the CPU lane, so slower
/// decoding and other rounding in every answer - there `base` stays).  A longer span only makes prompts between
/// `base` and it faster, so it never costs any of those, and never a start: `base` stays where no candidate keeps
/// them (or where the plan at `base` does not fit: that start's own error).
template <class PlanAt, class Cands>
int dense_pick(PlanAt&& plan_at, const Cands& cands, int base) {
    const ChunkTry ref = plan_at(base);
    if (!ref.ok) return base;
    for (int d : cands) {
        if (d <= base) continue;
        const ChunkTry t = plan_at(d);
        if (t.ok && t.tp == ref.tp && t.q8 == ref.q8 && t.vision == ref.vision && t.pixels >= ref.pixels &&
            t.own == ref.own && t.head == ref.head && t.res >= ref.res)
            return d;
    }
    return base;
}

/// The DFly2 drafter (a block of dense layers that proposes several tokens in one pass over a query block; each of
/// its layers reads one trunk layer's latent cache and that layer's indexer picks - it keeps no cache of its own).
/// The keys of the query block's rows (positions pos .. pos + rows - 1): row j sees the trunk layer's picks for the
/// last accepted position (`picks`, those below pos, in their order) and then the block's own positions pos .. pos + j.
/// keys [rows][ldk], nk [rows]; returns the widest list (0: ldk is too narrow - nothing written).
inline int dfly_keys(const int* picks, int n_picks, long long pos, int rows, int* keys, int ldk, int* nk) {
    int n = 0;
    for (int i = 0; i < n_picks; ++i) n += picks[i] >= 0 && picks[i] < pos;
    if (n + rows > ldk) return 0;
    for (int j = 0; j < rows; ++j) {
        int* k = keys + (size_t) j * (size_t) ldk;
        int c = 0;
        for (int i = 0; i < n_picks; ++i)
            if (picks[i] >= 0 && picks[i] < pos) k[c++] = picks[i];
        for (int q = 0; q <= j; ++q) k[c++] = (int) (pos + q);
        nk[j] = c;
    }
    return n + rows;
}
/// The drafter's kv_b projection [heads (nope + v)][kv_lora] (a head's key rows, then its value rows) as the engine's
/// two matrices: the row of head h's value r, and the engine's key matrix of a head - [kv_lora][nope], the head's key
/// rows transposed (a query's nope part goes through it into the latent space).
inline size_t dfly_kvb_k_row(int h, int r, int nope, int v) { return (size_t) h * (size_t) (nope + v) + (size_t) r; }
inline size_t dfly_kvb_v_row(int h, int r, int nope, int v) {
    return (size_t) h * (size_t) (nope + v) + (size_t) nope + (size_t) r;
}
/// k [nope][kv_lora] (a head's key rows) -> out [kv_lora][nope]
inline void dfly_k_b(const float* k, int nope, int kv_lora, float* out) {
    for (int c = 0; c < kv_lora; ++c)
        for (int i = 0; i < nope; ++i) out[(size_t) c * nope + i] = k[(size_t) i * kv_lora + c];
}
/// The bytes the drafter takes on the head's GPU beside its matrices (`file`: its tensors' bytes in the pack): the
/// drafts' copy of the output head (`head`: its bytes, 0 when the model's own head serves), the picks kept from the
/// trunk's last window and the query block's key lists (`layers` trunk layers, `win` rows, lists of `maxk` + `win`
/// positions), its rows and its FFN's (`win` rows of `n_embd` and of `ff`).
inline double dfly_bytes(double file, double head, long long layers, long long win, long long maxk, long long n_embd,
                         long long ff) {
    return file + head + (double) layers * (double) win * (double) (2 * maxk + win + 2) * 4 +
           (double) win * (double) (8 * n_embd + 3 * ff) * 4 + (double) (8u << 20);
}

}  // namespace hy4
