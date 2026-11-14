// The peer control channel: a TCP connection carrying newline-delimited JSON lines - the QP
// endpoints, the pack manifest, the row table's size, the advertisement map and liveness all travel
// here, while the expert bytes themselves go over the RDMA link.
//
// Why NDJSON with an explicit flush (the pair's RDMA spike paid for this lesson): a buffered writer
// that only flushes on its own schedule deadlocks a full-duplex read-then-write handshake - both
// sides sit waiting for a line the other has "sent" into a buffer.  Every line goes out with
// write_all + an explicit flush point, and every read is a poll-bounded getline.
#pragma once

#include <poll.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace strata::peer {

// --------------------------------------------------------------------------- tiny flat JSON
// The peer protocol's messages are flat (no nesting beyond the few arrays we build ourselves), so
// the parser is a scanner, not a DOM: keys are looked up in a written-in-the-same-breath format we
// control on both ends.  Strings escape \\ and \" and control bytes as \u00XX; numbers stay plain.

std::string json_escape(const std::string& s);

/// A flat JSON object being built: add(key, value) in any order, str() renders it.
class Json {
public:
    Json& add(const std::string& key, const std::string& value);
    Json& add(const std::string& key, const char* value);
    Json& add(const std::string& key, long long value);
    Json& add(const std::string& key, int value) { return add(key, (long long) value); }
    Json& add(const std::string& key, uint64_t value) { return add(key, (long long) value); }
    Json& add(const std::string& key, const std::vector<long long>& values);
    /// Raw text as the value (a nested array we built by hand): add_raw("rows","[1,2]")
    Json& add_raw(const std::string& key, const std::string& raw);
    std::string str() const;

private:
    std::vector<std::pair<std::string, std::string>> kv_;   // (key, rendered value)
};

/// A parsed flat object: only the keys asked for, looked up after the line arrived.
class JsonView {
public:
    explicit JsonView(const std::string& line);
    bool ok() const { return ok_; }
    std::string type() const { return str("type"); }
    std::string str(const std::string& key, const std::string& def = "") const;
    long long num(const std::string& key, long long def = -1) const;
    /// A "key":[n,n,...] array of integers (the hint batches); empty when the key is absent.
    std::vector<long long> nums(const std::string& key) const;

private:
    std::map<std::string, std::string> kv_;   // values still JSON-escaped as they came
    bool ok_ = false;
};

// --------------------------------------------------------------------------- the channel itself

class Ctl {
public:
    Ctl() = default;
    ~Ctl();
    Ctl(const Ctl&) = delete;
    Ctl& operator=(const Ctl&) = delete;

    /// Connects to host:port (TCP).  host may be a name (getaddrinfo resolves it).
    bool connect(const std::string& host, int port, int timeout_ms, std::string& err);
    /// Takes an accepted socket over (the daemon's per-connection half).
    bool adopt(int fd, std::string& err);
    void close();
    bool is_open() const { return fd_ >= 0; }
    int fd() const { return fd_; }
    /// This end's / the peer's IPv4 address on the link (the cm setup resolves over it).
    std::string local_ip(std::string& err) const;
    std::string peer_ip(std::string& err) const;

    /// One JSON line out (the newline is added; the flush is the send).
    bool send(const std::string& json_line, std::string& err);
    /// One JSON line in, or "" past `timeout_ms` (then err says so; a closed peer sets closed_).
    std::string recv(int timeout_ms, std::string& err);
    bool closed() const { return closed_; }

private:
    bool write_all(const void* p, size_t bytes, std::string& err);

    int fd_ = -1;
    bool closed_ = false;
    std::string line_;              // recv's over-read: bytes past the last full line, kept here
};

}   // namespace strata::peer
