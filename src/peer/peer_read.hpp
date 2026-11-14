// The RDMA READ engine: a (byte range -> local buffer) pull, split into ~2 MB chunks issued through
// a bounded window of outstanding WRs.  This is the whole per-row cost on the wire - the responder
// runs no software (its NIC answers the reads), so the window and the completion cadence are the
// only knobs.  Posting faster than completions come back ends in ENOSPC on ibv_post_send (the
// spike's lesson), so every post batch is preceded by a poll that keeps the in-flight count under
// the window.
//
// The row (a pack byte range: one expert's pieces, several MB to tens of MB) arrives whole in the
// caller's registered buffer or not at all: a partial row is worth nothing to the decode path, so a
// failed or timed-out chunk fails the row and the caller falls back to the disk tier.
#pragma once

#include "peer_rdma.hpp"

#include <cstdint>

namespace strata::peer {

/// A chunked RDMA READ of [src_addr, src_addr + bytes) (the peer's registered slab, rkey) into
/// [dst + dst_off, ...) (this side's registered MR).  Returns false on any completion error,
/// timeout (`timeout_ms` for the whole row) or a post failure - the buffer's contents are then
/// undefined and the caller re-reads the row another way.
class Reader {
public:
    /// window: outstanding chunks at once (<= the QP's max_send_wr; 48 default, env MAYA_PEER_WINDOW).
    /// chunk: bytes per WR (default 2 MiB, env MAYA_PEER_CHUNK_MB; the last chunk takes the rest).
    /// cq: one shared completion queue, polled non-blocking between post batches.
    Reader(ibv_qp* qp, ibv_cq* cq, int window, size_t chunk);
    ~Reader() = default;

    bool read(uint64_t src_addr, uint32_t src_rkey, ibv_mr* dst_mr, uint64_t dst_off, size_t bytes,
              int timeout_ms, std::string& err);

    /// Rows read and bytes pulled, cumulative over this Reader (the tiers line's peer counters).
    uint64_t rows() const { return rows_; }
    uint64_t bytes_read() const { return bytes_; }
    uint64_t us() const { return us_; }

private:
    bool poll_once(int64_t deadline_ms, std::string& err);   // one completion or a deadline check

    ibv_qp* qp_;
    ibv_cq* cq_;
    int window_;
    size_t chunk_;

    // the in-flight ring: a chunk's wr_id is its slot; completions land back on their slot
    struct Inflight {
        bool posted = false;
    };
    std::vector<Inflight> inflight_;
    int outstanding_ = 0;

    uint64_t rows_ = 0, bytes_ = 0, us_ = 0;
};

}   // namespace strata::peer
