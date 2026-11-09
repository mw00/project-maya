// include/strata/core/glm_warm_seq.hpp - glm-dsa B8.1 (warm): the load-time expert warm-up in file order, and a
// digest of what it loaded.
//
// fast_warm (Maya-owned src/core/glm_fast_path.cu) first gives every expert its VRAM or RAM-tier slot, then streams
// the experts in.  Maya streams them in slot order - each layer's VRAM slots, then the RAM tier round-robin over the
// layers - 8 experts a batch whose 24 slices 8 threads read at once, and in a parallel split load every part does so
// at the same time.  From a single hard disk that is a seek every slice: the test PC's USB disk gave 39 MB/s, 201 GB in
// 89 min (bench/results/strata/volta3-bringup.md).  STRATA_GLM_WARM_SEQ (on by default for glm-dsa, off for Flash):
//   * the jobs sorted by where each expert starts in the pack (shard, then offset: a layer's experts in id order);
//   * each batch's slices read in file order, one after another, on the warming thread (about 3 contiguous runs a
//     batch: gate, up and down of 8 neighbouring experts);
//   * in a parallel split load the parts warm one after another, in part order (Glm5Model::SplitLoad's warm turn), so
//     the disk sees one stream; the dense load, the RAM planning and the pinning stay parallel.
// Only the order of the reads changes: every expert lands in the slot it had, through the same read (fast_warm's
// per-slice reader).  STRATA_GLM_WARM_SUM=1 prints a digest of the slots and their bytes after each part's warm-up,
// and glm_dsa_pack_fast_test compares the two orders' digests.
//
// The order, the run count and the hash are plain C++ (tests/glm_dsa/glm_warm_seq_test.cpp checks them, no CUDA);
// Turn and the digests read the engine (src/core/glm_warm_seq_model.cu).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace strata::core {

class Glm5Model;

namespace glmwarm {

/// STRATA_GLM_WARM_SEQ (read at each warm-up): the file-order warm-up, default on for glm-dsa, off for Flash.
bool seq_on(bool glm_dsa);
/// STRATA_GLM_WARM_SUM (read at each warm-up, default off): print each part's warm digest.
bool sum_on();

/// Where an expert (or one of its slices) starts in the pack: the shard, then the byte offset in it.
struct At {
    int shard = 0;
    uint64_t off = 0;
};
/// The permutation that puts `at` in file order (shard, then offset); equal places keep their given order.
std::vector<size_t> file_order(const std::vector<At>& at);

/// One read of a warm-up batch: [off, off + len) of shard `shard`, the batch's job index `job` (fast_warm: 3 x the
/// expert's place in the batch + its slice: 0 gate, 1 up, 2 down).
struct Piece {
    int shard = 0;
    uint64_t off = 0;
    size_t len = 0;
    int job = 0;
};
/// Sorts a batch's reads into file order (stable).
void read_order(std::vector<Piece>& pc);
/// The contiguous runs in `pc` as read in its order: a read that starts where the one before it ended (same shard)
/// continues that run.
size_t runs(const std::vector<Piece>& pc);

/// One step of the digest: v folded into h (a bijection of h for each v, so a single changed value always changes it).
inline uint64_t mix(uint64_t h, uint64_t v) {
    h ^= v * 0x9E3779B97F4A7C15ull;
    h = (h << 27 | h >> 37) * 0xC2B2AE3D27D4EB4Full + 0x165667B19E3779F9ull;
    return h;
}
/// The bytes [p, p + n) hashed (8-byte words through mix, the tail, the length, a final avalanche): evidence that two
/// copies are equal, not a cryptographic hash.
inline uint64_t hash_bytes(const void* p, size_t n, uint64_t h = 0x27D4EB2F165667C5ull) {
    const uint8_t* b = (const uint8_t*) p;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w;
        std::memcpy(&w, b + i, 8);
        h = mix(h, w);
    }
    uint64_t t = 0;
    for (size_t k = 0; i + k < n; ++k) t |= (uint64_t) b[i + k] << (8 * k);
    h = mix(mix(h, t), (uint64_t) n);
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDull;
    h ^= h >> 33;
    h *= 0xC4CEB9FE1A85EC53ull;
    h ^= h >> 33;
    return h;
}

/// The tiers' contents: every VRAM slot of every MoE layer and every RAM-tier slot, in slot order, as (tier, layer or
/// class, slot, expert, state, a hash of the expert's bytes when the slot holds one).  Two loads that put the same
/// experts in the same slots with the same bytes have the same sum, whatever order they read them in.
struct Digest {
    uint64_t sum = 0;
    uint64_t vram = 0, ram = 0;   ///< slots holding an expert
    uint64_t bytes = 0;           ///< their expert bytes
    int parts = 0;                ///< the parts digested
    bool ok = true;               ///< false: a device copy failed (sum is not evidence)
};
/// One part right now (this instance's own tiers; parts 0 without the fast path).
Digest part_digest(Glm5Model& m);
/// fast_warm's STRATA_GLM_WARM_SUM line, "glm fast: CUDA<d> warm digest <hex>: ...", at the end of the part's warm-up
/// (before anything else touches the tiers); the digest is kept for warm_digest().
void print_digest(Glm5Model& m);
/// Keeps a part's warm digest; warm_digest(): the ones kept since stats_reset(), folded in part order (a load's warm
/// digest, which a test compares between two loads).
void note_digest(int part, const Digest& d);
Digest warm_digest();

/// What the file-order warm-ups did since stats_reset() (every part, every load of the process).
struct Stats {
    uint64_t warms = 0;     ///< file-order warm-ups
    uint64_t jobs = 0;      ///< experts they streamed
    uint64_t moved = 0;     ///< ... that the file order read at another place than the slot order
    uint64_t batches = 0;   ///< batches of up to 8 experts
    uint64_t reads = 0;     ///< slice reads (an EXL3 expert: one)
    uint64_t runs = 0;      ///< contiguous runs those reads formed
    int max_busy = 0;       ///< the most warm-ups (either order) that streamed at the same time
};
Stats stats();
void stats_reset();
/// fast_warm: one file-order warm-up's counts.
void note(size_t jobs, size_t moved, size_t batches, size_t reads, size_t runs);
/// A warm-up starts / ends streaming (Turn calls them).
void busy_enter();
void busy_leave();

/// fast_warm's turn on the disk, held for the whole warm-up: with `seq` in a parallel split load it waits until the
/// parts before this one have warmed (Glm5Model::SplitLoad::wait_warm); its end lets the next part on (pass_warm,
/// whatever the order, so a part that fails never holds the others).
class Turn {
public:
    Turn(Glm5Model& m, bool seq);
    ~Turn();
    Turn(const Turn&) = delete;
    Turn& operator=(const Turn&) = delete;

private:
    Glm5Model& m_;
};

}  // namespace glmwarm
}  // namespace strata::core
