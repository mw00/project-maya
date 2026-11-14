#include "peer_client.hpp"

#include "peer_read.hpp"

#include <cerrno>
#include <cstring>

namespace strata::peer {

PeerClient::PeerClient(int window, size_t chunk, int timeout_ms, const std::string& device)
    : window_(window), timeout_ms_(timeout_ms), chunk_(chunk), device_(device) {}

PeerClient::~PeerClient() {
    close();
}

void PeerClient::close() {
    up_ = false;
    reader_.reset();
    qp_.destroy();
    dev_.close();
    dst_.clear();
    map_.clear();
    ctl_.close();
}

ibv_pd* PeerClient::pd() const { return dev_.pd(); }

bool PeerClient::add_dst(void* buf, size_t bytes, std::string& err) {
    auto mr = std::make_unique<Mr>();
    if (!mr->reg(dev_.pd(), buf, bytes, IBV_ACCESS_LOCAL_WRITE, err)) return false;
    dst_.push_back(std::move(mr));
    return true;
}

void PeerClient::clear_dst() {
    dst_.clear();
}

bool PeerClient::connect(const std::string& host, int port, const std::string& pack_dir,
                         const std::vector<std::string>& shard_names, const std::vector<uint64_t>& shard_sizes,
                         const std::vector<Row>& rows, std::string& err) {
    close();
    pack_dir_ = pack_dir;
    if (!ctl_.connect(host, port, timeout_ms_, err)) return false;
    rows_ = rows;
    return handshake(host, port, shard_names, shard_sizes, rows, err) && (up_ = true);
}

bool PeerClient::handshake(const std::string& host, int port, const std::vector<std::string>& shard_names,
                           const std::vector<uint64_t>& sizes, const std::vector<Row>& rows,
                           std::string& err) {
    if (!dev_.open(device_, err)) return false;
    // hello
    Json hello;
    hello.add("type", "hello").add("proto", 1).add("window", window_)
         .add("chunk_mb", (long long) (chunk_ >> 20)).add("timeout_ms", timeout_ms_);
    if (!ctl_.send(hello.str(), err)) return false;
    const JsonView welcome(ctl_.recv(timeout_ms_, err));
    if (!welcome.ok() || welcome.type() != "welcome") {
        err = "no welcome: " + err;
        return false;
    }
    if (welcome.num("proto") != 1) {
        err = "the daemon speaks protocol " + welcome.str("proto") + ", this client speaks 1";
        return false;
    }
    // manifest: names + sizes plus the fingerprint of this side's own copy - the daemon computes
    // the same string over its copy and refuses a mismatch (a half-copied pack would otherwise
    // serve wrong bytes silently)
    std::string fp = pack_fingerprint(pack_dir_, shard_names, sizes, err);
    if (fp.empty()) return false;
    Json man;
    man.add("type", "manifest").add("shards", (long long) shard_names.size()).add("fp", fp);
    for (size_t i = 0; i < shard_names.size(); ++i) {
        man.add("shard" + std::to_string(i), shard_names[i]);
        man.add("size" + std::to_string(i), (long long) sizes[i]);
    }
    if (!ctl_.send(man.str(), err)) return false;
    const JsonView fp_ack(ctl_.recv(timeout_ms_, err));
    if (!fp_ack.ok() || fp_ack.type() != "fingerprint") {
        err = "the daemon refused the manifest: " + (fp_ack.ok() ? fp_ack.str("msg") : err);
        return false;
    }
    // the row table: one raw binary frame after a one-line header (16 bytes a row: u16 shard,
    // u16 pad, u64 offset, u32 len - the daemon's Row, laid out identically).  The frame rides the
    // socket hex-encoded: arbitrary bytes would carry newlines and break the line framing.
    std::vector<char> blob(rows.size() * 16);
    for (size_t i = 0; i < rows.size(); ++i) {
        const uint16_t s = rows[i].shard;
        const uint64_t off = rows[i].off;
        const uint32_t len = rows[i].len;
        std::memcpy(blob.data() + i * 16, &s, 2);
        std::memcpy(blob.data() + i * 16 + 2, &s, 2);      // the pad keeps the offsets aligned
        std::memcpy(blob.data() + i * 16 + 4, &off, 8);
        std::memcpy(blob.data() + i * 16 + 12, &len, 4);
    }
    static const char* hex = "0123456789abcdef";
    std::string framed;
    framed.reserve(blob.size() * 2 + 5);
    for (char c : blob) {
        framed += hex[(unsigned char) c >> 4];
        framed += hex[(unsigned char) c & 15];
    }
    // one JSON line carries the whole table (a bare "hex:"-prefixed line would not parse as JSON)
    const Json rows_line = Json().add("type", "rows").add("count", (long long) rows.size())
                                .add("hex", framed);
    if (!ctl_.send(rows_line.str(), err)) return false;
    const JsonView rows_ack(ctl_.recv(timeout_ms_, err));
    if (!rows_ack.ok() || rows_ack.type() != "rows") {
        err = "the daemon refused the row table: " + (rows_ack.ok() ? rows_ack.str("msg") : err);
        return false;
    }
    // the QP: create it here, hand {qpn, psn, gid} over the ctl channel (llama.cpp's ggml-rpc
    // recipe), and arm it from the daemon's numbers - both sides then talk RC over the link
    if (!qp_.create(dev_, window_ * 2 + 8, 8, err)) return false;
    if (!ctl_.send(Json().add("type", "qp-up").add("qpn", (long long) qp_.qpn())
                       .add("psn", (long long) qp_.psn()).add("gid", qp_.gid()).str(), err)) {
        return false;
    }
    const JsonView ready(ctl_.recv(timeout_ms_, err));
    if (!ready.ok() || ready.type() != "ready") {
        err = "the daemon never came ready: " + (ready.ok() ? ready.type() : err);
        return false;
    }
    if (!qp_.arm(dev_, (uint32_t) ready.num("qpn"), (uint32_t) ready.num("psn"),
                 ready.str("gid"), err)) {
        return false;
    }
    // the chunked-read engine rides the fresh QP: one shared CQ (the device's), the window and
    // chunk from the hello (the daemon sized its QP against the same window)
    reader_ = std::make_unique<Reader>(qp_.qp_ptr(), dev_.cq(), window_, chunk_);
    say("session up: " + std::to_string(rows.size()) + " rows in the table, window " +
        std::to_string(window_));
    return true;
}

bool PeerClient::hint(const std::vector<long long>& rows, std::string& err) {
    if (!up_ || rows.empty()) return true;
    Json h;
    h.add("type", "hint").add("n", (long long) rows.size());
    h.add("rows", rows);
    if (!ctl_.send(h.str(), err)) {
        up_ = false;
        return false;
    }
    // the first hint makes the daemon allocate and register a fresh arena (GiBs of pinning), so
    // the reply can legitimately outrun the read timeout - hints are rare, reads are the hot path
    const int budget = timeout_ms_ > 30000 ? timeout_ms_ : 30000;
    const JsonView map(ctl_.recv(budget, err));
    if (!map.ok()) {
        say("hint reply lost: up=" + std::to_string(up_) + " closed=" +
            std::to_string(ctl_.closed()) + " after " + std::to_string(budget) + " ms: " + err);
        up_ = false;
        return false;
    }
    say("hint reply: " + std::to_string(map.nums("i").size()) + " entries");
    if (map.type() == "error") {
        err = "the daemon refused the hint: " + map.str("msg");
        return false;
    }
    if (map.type() != "map") {
        err = "expected a map line, got " + map.type();
        up_ = false;
        return false;
    }
    for (long long dropped : map.nums("dropped")) map_.erase(dropped);
    // the entries: the daemon answers with four parallel integer arrays - i, addr, len, rkey
    const auto idx = map.nums("i");
    const auto addr = map.nums("addr");
    const auto len = map.nums("len");
    const auto rkey = map.nums("rkey");
    if (idx.size() != addr.size() || idx.size() != len.size() || idx.size() != rkey.size()) {
        err = "a map line with mismatched arrays";
        up_ = false;
        return false;
    }
    for (size_t k = 0; k < idx.size(); ++k) {
        map_[idx[k]] = Cached{0, (uint64_t) addr[k], (uint32_t) len[k], (uint32_t) rkey[k]};
    }
    return true;
}

bool PeerClient::read_row(long long ri, Mr& dst, uint64_t dst_off, std::string& err) {
    const auto it = map_.find(ri);
    if (it == map_.end()) return false;                  // unadvertised: the caller's disk fallback
    if (!up_) return false;
    const Cached& c = it->second;
    const bool ok = reader_->read(c.addr, c.rkey, dst.mr_ptr(), dst_off, c.len, timeout_ms_, err);
    if (!ok) {
        up_ = false;             // a failed read leaves the QP's state unknown: reconnect, not retry
        err = "row " + std::to_string(ri) + " failed: " + err;
    }
    return ok;
}

bool PeerClient::keepalive(std::string& err) {
    if (!up_) return false;
    if (!ctl_.send(Json().add("type", "ping").str(), err)) {
        up_ = false;
        return false;
    }
    const JsonView pong(ctl_.recv(timeout_ms_, err));
    if (!pong.ok() || pong.type() != "pong") {
        up_ = false;
        err = "no pong: " + (pong.ok() ? pong.type() : err);
        return false;
    }
    return true;
}

}   // namespace strata::peer
