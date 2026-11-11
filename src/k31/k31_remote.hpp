// src/k31/k31_remote.hpp - maya-k31's remote experts: a layer's routed experts computed by an expert server over TCP.
//
// The client (the engine holding attention and the dense weights) sends, per MoE layer, the picks a server owns and
// the rows those experts read: the latent z exactly as the engine's experts consume it (quantized: per row zin int8
// and zin/32 float32 scales; float32 experts: zin floats).  The server runs gate / up / SiTU / down for each pick (the
// engine's own code) and answers with each pick's float32 output row (latent wide); the client writes them into its
// parts buffer and combines as it would its own, so the result is the same bits as local compute.
//
// Wire format, version 1: every field little-endian and fixed width, written field by field (no struct images), so
// an ARM64 (Grace) server reads what an x86-64 client writes.
//   request : u32 magic 'K31Q', u32 version, i32 layer, i32 n_rows, i32 zin, i32 zfmt (0 f32 rows, 1 q8 rows),
//             i32 n_pick, i32 flags (0)
//             n_pick x (i32 expert, i32 row)
//             n_rows x row: zfmt 1: zin x i8, then zin/32 x f32;  zfmt 0: zin x f32
//   reply   : u32 magic 'K31A', u32 version, i32 layer, i32 n_pick, i32 lat, i32 status (0 ok; else an error, no rows)
//             n_pick x lat x f32, in the request's pick order
// layer -1 is an echo (no compute): the reply carries n_pick rows of `lat` = the request's zin zeros (link timing).
// One persistent connection, TCP_NODELAY; requests are answered in order.
#pragma once

#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "k31_remote: the wire format is little-endian and this build assumes a little-endian host (x86-64, ARM64)"
#endif

namespace k31::remote {

inline constexpr uint32_t kMagicReq = 0x5131334bu;   // "K31Q"
inline constexpr uint32_t kMagicRep = 0x4131334bu;   // "K31A"
inline constexpr uint32_t kVersion = 1;
inline constexpr int kHeadReq = 32, kHeadRep = 24;

struct Pick { int32_t expert, row; };

// ---- bytes
class Buf {
public:
    std::vector<uint8_t> b;
    template <class T> void put(T v) { const size_t o = b.size(); b.resize(o + sizeof v); std::memcpy(b.data() + o, &v, sizeof v); }
    void put_bytes(const void* p, size_t n) { const size_t o = b.size(); b.resize(o + n); std::memcpy(b.data() + o, p, n); }
};
template <class T> inline T get(const uint8_t* p) { T v; std::memcpy(&v, p, sizeof v); return v; }

// ---- sockets
inline void send_all(int fd, const void* p, size_t n) {
    const uint8_t* c = (const uint8_t*) p;
    while (n) {
        const ssize_t k = ::send(fd, c, n, MSG_NOSIGNAL);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) throw std::runtime_error(std::string("k31 remote: send failed: ") + std::strerror(errno));
        c += k;
        n -= (size_t) k;
    }
}
inline bool recv_all(int fd, void* p, size_t n) {   // false: the peer closed before the first byte
    uint8_t* c = (uint8_t*) p;
    size_t got = 0;
    while (got < n) {
        const ssize_t k = ::recv(fd, c + got, n - got, 0);
        if (k < 0 && errno == EINTR) continue;
        if (k == 0 && got == 0) return false;
        if (k <= 0) throw std::runtime_error(std::string("k31 remote: recv failed: ") + (k == 0 ? "peer closed" : std::strerror(errno)));
        got += (size_t) k;
    }
    return true;
}
inline void tune(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    int sz = 8 << 20;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sz, sizeof sz);
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &sz, sizeof sz);
}
inline int connect_to(const std::string& hostport) {
    const size_t c = hostport.rfind(':');
    if (c == std::string::npos) throw std::runtime_error("k31 remote: want HOST:PORT, got " + hostport);
    const std::string host = hostport.substr(0, c), port = hostport.substr(c + 1);
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0 || !res) throw std::runtime_error("k31 remote: cannot resolve " + hostport);
    int fd = -1;
    for (addrinfo* a = res; a; a = a->ai_next) {
        fd = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (fd < 0) continue;
        if (::connect(fd, a->ai_addr, a->ai_addrlen) == 0) break;
        ::close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) throw std::runtime_error("k31 remote: cannot connect to " + hostport);
    tune(fd);
    return fd;
}
/// Listens on HOST:PORT (HOST may be an address or name), or on 127.0.0.1 when only a port is given: the server has no
/// authentication, so it is never on a network unless asked by address.
inline int listen_on(const std::string& where) {
    const size_t c = where.rfind(':');
    const std::string host = c == std::string::npos ? "127.0.0.1" : where.substr(0, c);
    const std::string port = c == std::string::npos ? where : where.substr(c + 1);
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0 || !res) throw std::runtime_error("k31 remote: cannot resolve " + where);
    int fd = -1;
    for (addrinfo* a = res; a; a = a->ai_next) {
        fd = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (fd < 0) continue;
        int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        if (::bind(fd, a->ai_addr, a->ai_addrlen) == 0 && ::listen(fd, 4) == 0) break;
        ::close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) throw std::runtime_error("k31 remote: cannot listen on " + host + ":" + port + ": " + std::strerror(errno));
    return fd;
}

// ---- messages
struct Request {
    int32_t layer = 0, n_rows = 0, zin = 0, zfmt = 1, flags = 0;
    std::vector<Pick> picks;
    std::vector<uint8_t> rows;   // n_rows x row bytes (row_bytes())
    size_t row_bytes() const { return zfmt == 1 ? (size_t) zin + (size_t) (zin / 32) * 4 : (size_t) zin * 4; }
};
inline void write_request(int fd, const Request& r) {
    Buf b;
    b.b.reserve(kHeadReq + r.picks.size() * 8 + r.rows.size());
    b.put(kMagicReq); b.put(kVersion); b.put(r.layer); b.put(r.n_rows); b.put(r.zin); b.put(r.zfmt);
    b.put((int32_t) r.picks.size()); b.put(r.flags);
    for (const Pick& p : r.picks) { b.put(p.expert); b.put(p.row); }
    b.put_bytes(r.rows.data(), r.rows.size());
    send_all(fd, b.b.data(), b.b.size());
}
inline bool read_request(int fd, Request& r) {   // false: the connection closed cleanly
    uint8_t h[kHeadReq];
    if (!recv_all(fd, h, sizeof h)) return false;
    if (get<uint32_t>(h) != kMagicReq || get<uint32_t>(h + 4) != kVersion) throw std::runtime_error("k31 remote: bad request header");
    r.layer = get<int32_t>(h + 8); r.n_rows = get<int32_t>(h + 12); r.zin = get<int32_t>(h + 16); r.zfmt = get<int32_t>(h + 20);
    const int32_t np = get<int32_t>(h + 24);
    r.flags = get<int32_t>(h + 28);
    if (r.n_rows < 0 || np < 0 || r.zin < 0 || (r.zfmt != 0 && r.zfmt != 1) || (r.zfmt == 1 && r.zin % 32))
        throw std::runtime_error("k31 remote: bad request sizes");
    std::vector<uint8_t> pk((size_t) np * 8);
    if (np && !recv_all(fd, pk.data(), pk.size())) throw std::runtime_error("k31 remote: short request");
    r.picks.resize((size_t) np);
    for (int32_t i = 0; i < np; ++i) r.picks[(size_t) i] = {get<int32_t>(pk.data() + 8 * i), get<int32_t>(pk.data() + 8 * i + 4)};
    r.rows.resize((size_t) r.n_rows * r.row_bytes());
    if (!r.rows.empty() && !recv_all(fd, r.rows.data(), r.rows.size())) throw std::runtime_error("k31 remote: short request");
    return true;
}
inline void write_reply(int fd, int32_t layer, int32_t lat, int32_t status, const std::vector<float>& rows, int32_t n_pick) {
    Buf b;
    b.b.reserve(kHeadRep + rows.size() * 4);
    b.put(kMagicRep); b.put(kVersion); b.put(layer); b.put(n_pick); b.put(lat); b.put(status);
    if (status == 0) b.put_bytes(rows.data(), rows.size() * 4);
    send_all(fd, b.b.data(), b.b.size());
}
/// Reads the reply to `layer` into rows ([n_pick][lat]); throws on a mismatch or an error status.
inline void read_reply(int fd, int32_t layer, int32_t n_pick, int32_t lat, std::vector<float>& rows) {
    uint8_t h[kHeadRep];
    if (!recv_all(fd, h, sizeof h)) throw std::runtime_error("k31 remote: the server closed the connection");
    if (get<uint32_t>(h) != kMagicRep || get<uint32_t>(h + 4) != kVersion) throw std::runtime_error("k31 remote: bad reply header");
    const int32_t L = get<int32_t>(h + 8), np = get<int32_t>(h + 12), la = get<int32_t>(h + 16), st = get<int32_t>(h + 20);
    if (st != 0) throw std::runtime_error("k31 remote: the server failed layer " + std::to_string(L) + " (status " + std::to_string(st) + ")");
    if (L != layer || np != n_pick || la != lat) throw std::runtime_error("k31 remote: reply out of step (layer " + std::to_string(L) + ")");
    rows.resize((size_t) np * (size_t) la);
    if (!rows.empty() && !recv_all(fd, rows.data(), rows.size() * 4)) throw std::runtime_error("k31 remote: short reply");
}

}  // namespace k31::remote
