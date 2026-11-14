// The initiator's peer session: the control channel's client half, the row table it hands over, the
// QP it brings up against the daemon's, and the map of advertised rows (row -> addr/len/rkey) the
// READs go through.  Phase 3's engine tier wraps exactly this class; the test binary drives it
// directly.
//
// One session at a time, and a failed row fails the session: a completion error leaves the QP in an
// undefined state, so the honest recovery is close + reconnect (the daemon keeps its slab, the map
// is re-advertised on hello), never a retry through a half-dead QP.
#pragma once

#include "peer_cache.hpp"     // (Row; a daemon type, but the row table's wire format is shared)
#include "peer_ctl.hpp"
#include "peer_read.hpp"
#include "peer_rdma.hpp"

#include <unordered_map>

namespace strata::peer {

class PeerClient {
public:
    /// window/chunk/timeout as Reader's (env MAYA_PEER_* defaults, see the daemon's flags).
    PeerClient(int window = 48, size_t chunk = 2ULL << 20, int timeout_ms = 400,
               const std::string& device = "");
    ~PeerClient();
    PeerClient(const PeerClient&) = delete;
    PeerClient& operator=(const PeerClient&) = delete;

    /// Full handshake against host:port: hello, manifest (names/sizes plus the fingerprint of the
    /// initiator's own copy under pack_dir; the daemon refuses a mismatch), the row table, the QP
    /// exchange.  On success the map is whatever the daemon advertised at hello (empty until the
    /// first hints).
    bool connect(const std::string& host, int port, const std::string& pack_dir,
                 const std::vector<std::string>& shard_names, const std::vector<uint64_t>& shard_sizes,
                 const std::vector<Row>& rows, std::string& err);
    void close();
    bool is_up() const { return up_; }

    /// Registers a destination buffer (the rows land here; the engine's pinned staging in phase 3).
    bool add_dst(void* buf, size_t bytes, std::string& err);
    /// This session's PD - a dst MR must be registered on IT (a second Device's PD registers an
    /// lkey the QP cannot use: the READ's local sink then fails with local protection error)
    ibv_pd* pd() const;
    void clear_dst();

    /// Hints rows (the daemon faults them in and advertises them); merges the answer into the map
    /// and applies any drops.  Call before reading rows the map does not have yet.
    bool hint(const std::vector<long long>& rows, std::string& err);

    /// RDMA READs row `ri` into dst_mr at dst_off.  False (with the session closed) when the row
    /// is unadvertised - the caller falls back to its disk tier - or the read failed.
    bool read_row(long long ri, Mr& dst, uint64_t dst_off, std::string& err);
    bool advertised(long long ri) const { return map_.count(ri) != 0; }

    /// Liveness: ping the daemon, refresh the map (drops included).  Cheap; the engine calls it
    /// between requests, the test at its own pace.
    bool keepalive(std::string& err);

    uint64_t rows_read() const { return reader_ ? reader_->rows() : 0; }
    uint64_t bytes_read() const { return reader_ ? reader_->bytes_read() : 0; }
    uint64_t us_read() const { return reader_ ? reader_->us() : 0; }
    size_t advertised_rows() const { return map_.size(); }

private:
    bool handshake(const std::string& host, int port, const std::vector<std::string>& shard_names,
                   const std::vector<uint64_t>& sizes,
                   const std::vector<Row>& rows, std::string& err);

    int window_, timeout_ms_;
    size_t chunk_;
    std::string device_;
    Device dev_;
    Ctl ctl_;
    Qp qp_;
    std::unique_ptr<Reader> reader_;
    std::vector<std::unique_ptr<Mr>> dst_;       // registered destination buffers
    std::unordered_map<long long, Cached> map_;  // the daemon's advertisement, as received
    std::vector<Row> rows_;
    std::string pack_dir_;           // the initiator's own copy: fingerprinted for the manifest
    bool up_ = false;
};

}   // namespace strata::peer
