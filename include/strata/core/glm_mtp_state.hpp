// include/strata/core/glm_mtp_state.hpp - glm-dsa B15.3: the NextN (MTP) block's sequence state.
//
// The draft block reads, for position p, the trunk's post-output-norm hidden h_p (the "carrier": llama.cpp's
// pending_h / t_h_nextn) and the embedding of the token chosen for p + 1.  Maya's fast_mtp read it from the tail
// part's shared scratch row (FastState::head_x), which belongs to whatever forward ran last on that card: another
// sequence (STRATA_GLM_SEQS), an earlier request whose length happens to equal this one's, or nothing at all after a
// slot restore.  llama.cpp hit the same three bugs: #28333 (zero pending_h when a sequence starts at pos 0), #30141
// (save and restore it with the sequence state, magic MTP1) and #28473 (stop drafting at the output budget).
//
// The rules (BUILD-PLAN B15.3):
//   (a) every sequence owns one carrier row and the position it holds (CarrierState).  The trunk head notes it
//       (Glm5Model::mtp_carry_note, right after res_head); the draft block reads it for its own position only
//       (mtp_carry_row): a row noted for another position, or none, reads as zeros - never a stale row.  A sequence
//       that starts at pos 0 (Glm5Model::reset) zeroes it (begin()).
//   (b) the row belongs to the sequence state: the in-memory snapshot keeps a copy (valid only when it holds the
//       snapshot position's last row, n - 1); slot files and the B14.14 prompt-cache entries end with an MTP1
//       trailer (write_trailer / read_trailer) holding it, and the slot fingerprint gains two words: the NextN
//       block's identity (block_id over its tensors' names, types and source; a slot written with another draft
//       block, e.g. STRATA_GLM_MTP_GGUF, is refused because its NextN cache rows came from other weights) and the
//       carrier row bytes.  The NextN block's latent, k_pe and index-key runs were already in the slot
//       (GlmStateLayout::slot_runs, NextN first).
//   (c) draft_budget(n_max, remaining n_predict, remaining ctx): a verify row past the output budget or the context
//       is never issued (decode_spec's speculative head position; B11.4 / B15.9 multi-token drafts cap with it).
//
// Plain C++ (no CUDA): the state machine, the trailer and the budget are shared by the engine
// (src/core/glm_mtp_state_model.cu, the device rows) and the CPU twin test (tests/glm_dsa/glm_mtp_state_test.cpp).
#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace strata::core::glmmtp {

/// (c) How many draft tokens may be verified next: min(n_max, remaining_predict, remaining_ctx), never negative.
/// remaining_predict: the tokens the request may still emit AFTER the one being produced now; remaining_ctx: the
/// positions left after the draft's own position.
int draft_budget(int n_max, int64_t remaining_predict, int64_t remaining_ctx);

/// decode_spec's form: the head would run position spec_pos on a draft while the tail produces the token of
/// spec_pos - 1 (the produced-th).  The draft is verified only if a token after that one is wanted
/// (produced + 1 < max_new) and the loop may still advance (spec_pos + 1 < max_ctx).
inline int spec_budget(int n_max, int64_t produced, int64_t max_new, int64_t spec_pos, int64_t max_ctx) {
    return draft_budget(n_max, max_new - produced - 1, max_ctx - spec_pos - 1);
}

/// (a) One sequence's carrier bookkeeping (the row itself lives beside it: a device row in the engine, a host
/// vector in HostCarrier).
struct CarrierState {
    int64_t pos = -1;        // the position whose trunk hidden the row holds; -1: the row is zero
    int64_t snap_pos = -1;   // the snapshot copy's position; -1: the copy is zero
    uint64_t notes = 0, reads = 0, zero_reads = 0, stale_reads = 0, resets = 0;

    /// A sequence starts (pos 0, a reset, a slot or cache restore without a row): the row is zeroed.
    void begin() {
        pos = -1;
        ++resets;
    }
    void note(int64_t p) {
        pos = p;
        ++notes;
    }
    /// The draft block at position p: true = read the row, false = read zeros (the row is zeroed first when it held
    /// another position's hidden: stale_reads counts those, the cases Maya's shared scratch row got wrong).
    bool use(int64_t p);
    /// The snapshot at model position n (positions 0..n-1 kept): the copy is valid when the row is h_{n-1}.
    void snapshot_save(int64_t n) { snap_pos = pos == n - 1 ? pos : -1; }
    /// Back to the snapshot (the row takes the copy).
    void snapshot_restore() { pos = snap_pos; }
};

/// The carrier bookkeeping plus a host row (the CPU twin and the trailer's in-memory form).
struct HostCarrier {
    CarrierState st;
    std::vector<float> row, snap;
    explicit HostCarrier(int n_embd = 0) : row((size_t) n_embd, 0.0f), snap((size_t) n_embd, 0.0f) {}
    void begin();
    void note(int64_t p, const float* h);
    /// The row the draft block reads at p (zeros unless noted for p).
    const float* read(int64_t p);
    void snapshot_save(int64_t n);
    void snapshot_restore();
};

/// The NextN block's identity: FNV-1a over (name, ggml type, bytes) of each of its tensors in name order and the
/// source tag ("own" or the STRATA_GLM_MTP_GGUF file's basename and size), folded to 31 bits, never 0 (0 = no block).
struct BlockTensor {
    std::string name;
    int type = 0;
    uint64_t bytes = 0;
};
uint32_t block_id(std::vector<BlockTensor> tensors, const std::string& source);

/// (b) The MTP1 trailer after a slot file's runs (glm-dsa with a NextN block only):
///   u32 magic 0x4d545031 ("MTP1"), i32 n_embd, i64 pos (-1 or n - 1), u32 block id, u32 0, n_embd f32, u64 FNV-1a of
///   the row bytes.
constexpr uint32_t kTrailerMagic = 0x4d545031u;
size_t trailer_bytes(int n_embd);
bool write_trailer(std::ostream& f, int n_embd, int64_t pos, uint32_t block, const float* row);
enum class Trailer : int { Ok = 0, Absent = 1, Bad = 2 };
/// Reads a trailer for a slot of n positions.  Ok: pos (-1 or n - 1) and row (n_embd floats) are set.  Absent: end of
/// file (a slot written before B15.3).  Bad: anything else (why says what); pos / row untouched.
Trailer read_trailer(std::istream& f, int n_embd, int64_t n, uint32_t block, int64_t& pos, std::vector<float>& row,
                     std::string& why);

/// The fingerprint words B15.3 adds (GlmStateFingerprint[12], [13]).
inline int32_t fp_block_word(uint32_t block) { return (int32_t) (block & 0x7fffffffu); }
inline int32_t fp_carrier_word(bool glm_dsa, uint32_t block, int n_embd) {
    return glm_dsa && block != 0 ? n_embd * 4 : 0;
}

/// Counters of the engine's carriers (all sequences of a model chain), for logs and tests.
struct CarrierStats {
    uint64_t notes = 0, reads = 0, zero_reads = 0, stale_reads = 0, resets = 0, restored = 0, trailer_bad = 0;
};

}  // namespace strata::core::glmmtp
