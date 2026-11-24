// glm_link: see maya/core/glm_link.hpp
#include "maya/core/glm_link.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

#if !defined(_WIN32)
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/random.h>
#include <sys/uio.h>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#endif
#include <cerrno>
#include <utility>
#include <vector>

namespace maya::core::glmlink {
namespace {

int g_data = -1, g_ctl = -1, g_listen = -1;
std::atomic<bool> g_up{false};
// what open() was given (reopen / reaccept HELLO with it again)
int g_split = 0, g_my_chunk = 0, g_peer_chunk = -1;
int64_t g_ctx = 0;
// the request in flight: the back saw the front's kEnd (a cancel) - its position and flags
bool g_cancel = false;
int64_t g_end_pos = -1;
int32_t g_end_flags = 0;
bool g_bf16 = true;
bool g_pack = true;   // MAYA_GLM_LINK_PACK: the bf16 rows' high bytes nibble-coded (lossless)
std::mutex g_ctl_mu;
std::string g_model_id;   // set_model: the model this side loaded (the HELLO compares it)
std::string g_sched = "1";   // set_sched: several conversations' scheduler settings ("1": one conversation)
std::atomic<uint64_t> g_gen{0};       // generation(): links made
std::function<bool()> g_wait_hook;    // set_wait_hook
// stats
uint64_t s_hops = 0, s_hop_bytes = 0, s_toks = 0;
double s_send_ms = 0, s_recv_ms = 0;

const char* env1(const char* name) {   // (an empty setting counts as unset)
    const char* v = std::getenv(name);
    return v != nullptr && v[0] ? v : nullptr;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void shut_data();   // (the data connection shut both ways: a peer blocked on it fails too)
// why this thread's last wait on the link ended, when a bound ended it (io_wait) - fail() says it
thread_local const char* t_io_why = nullptr;
std::atomic<bool> g_term{false};      // term(): the back was told to stop (a signal)
std::atomic<bool> g_listening{false};   // the back listens (before that a signal ends the process at once)
bool g_stay = false;                  // set_stay(): the back serves one front after another
std::atomic<int64_t> g_stop_ms{0};    // stop_hint(): when the front's first STOP came (ms, steady clock; 0: none)
std::atomic<int64_t> g_stop_waited{0};   // ... and the time waited on the link since (ms): kStopGraceS in all

void fail(std::string& err, const std::string& what) {
    std::string w = what;
    if (t_io_why != nullptr) w += std::string(" - ") + t_io_why;
    t_io_why = nullptr;
    if (g_up.exchange(false)) std::fprintf(stderr, "glm link: DOWN - %s\n", w.c_str());
    shut_data();
    err = "glm link: " + w;
}

bool split_hostport(const std::string& s, std::string& host, std::string& port) {
    const size_t c = s.rfind(':');
    if (c == std::string::npos || c + 1 >= s.size()) return false;
    host = s.substr(0, c);
    port = s.substr(c + 1);
    if (host.empty()) host = "127.0.0.1";   // (never "every interface" by omission: the back binds loopback)
    return true;
}

// a numeric loopback address (127.0.0.0/8): what a link may use without a key.  A name is never taken for loopback
// (it is not resolved here, and "127.example.com" is not one)
bool is_loopback_v4(const std::string& host) {
    int part[4] = {0, 0, 0, 0}, n = 0, digits = 0;
    for (const char ch : host) {
        if (ch >= '0' && ch <= '9') {
            if (digits > 0 && part[n] == 0) return false;   // (no leading zeros: "127.008.0.1" is not this address)
            if (++digits > 3 || (part[n] = part[n] * 10 + (ch - '0')) > 255) return false;
        } else if (ch == '.' && digits > 0 && n < 3) {
            ++n;
            digits = 0;
        } else {
            return false;
        }
    }
    return n == 3 && digits > 0 && part[0] == 127;
}

#if defined(_WIN32)
void shut_data() {}
#else
void shut_data() {
    if (g_data >= 0) ::shutdown(g_data, SHUT_RDWR);
}
#endif

#if !defined(_WIN32)
// I/O ON THE LINK'S SOCKETS never waits in the kernel without a bound:
//  - a handshake (a connection's tag and proof, HELLO) runs under an absolute deadline: read_by / write_by;
//  - the link in use (read_all / write_all / send_msg) waits for the other side's next bytes for at most
//    MAYA_GLM_LINK_STALL_S seconds (60; 0: no bound).  TCP's own timers (tune) end a peer that is gone.  This ends
//    one that is up but stuck: its kernel still acknowledges and answers the keepalive probes.  It is under the
//    server's 90 s for an engine that does nothing (MAYA_ENGINE_STALL_S) and the serve watchdog's three minutes
//    (MAYA_WATCHDOG_S), so a stuck peer costs the request and the link, not the engine;
//  - after the front's STOP (stop_hint) the link is waited on for kStopGraceS seconds more, all waits together and
//    whatever moves meanwhile: a peer that does not answer, or answers a byte at a time, must not hold a cancel;
//  - a back that was told to stop (term) leaves every wait within a second, and starts no read or write.
using Clock = std::chrono::steady_clock;
using Deadline = Clock::time_point;
constexpr int kStopGraceS = 30;

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count();
}

int stall_s() {
    static const int s = [] {
        const char* v = env1("MAYA_GLM_LINK_STALL_S");
        return v != nullptr ? std::max(0, std::atoi(v)) : 60;
    }();
    return s;
}

// wait until fd can be read (POLLIN) or written (POLLOUT).  by: a handshake's deadline.  Null: the link in use,
// since_ms being when this call began to wait.  false: a bound passed (t_io_why says which), or poll failed
bool io_wait(int fd, short ev, const Deadline* by, int64_t since_ms) {
    for (;;) {
        if (g_term.load()) {
            t_io_why = "this side was told to stop";
            return false;
        }
        int64_t left = 1000;
        if (by != nullptr) {
            left = std::chrono::duration_cast<std::chrono::milliseconds>(*by - Clock::now()).count();
            if (left <= 0) return false;
        } else {
            if (stall_s() > 0 && now_ms() - since_ms >= (int64_t) stall_s() * 1000) {
                t_io_why = "the other side is connected but nothing moved for MAYA_GLM_LINK_STALL_S seconds";
                return false;
            }
            if (g_stop_ms.load() != 0) {
                const int64_t budget = (int64_t) kStopGraceS * 1000 - g_stop_waited.load();
                if (budget <= 0) {
                    t_io_why = "the other side kept this side waiting for 30 s after a STOP";
                    return false;
                }
                left = std::min(left, budget);
            }
        }
        struct pollfd pf {fd, ev, 0};
        const int64_t t_poll = now_ms();
        const int pr = ::poll(&pf, 1, (int) std::min<int64_t>(left, 1000));
        // (what a wait costs after a STOP counts against the STOP's budget, whether it ends with bytes or not)
        if (by == nullptr && g_stop_ms.load() != 0) g_stop_waited.fetch_add(now_ms() - t_poll);
        if (pr > 0) return true;   // (ready, or an error / a hang-up: the read or write that follows reports it)
        if (pr < 0 && errno != EINTR) return false;
    }
}

// before a read or write: not past a handshake's deadline, and not after this side was told to stop (a socket that
// is ready all the time never comes to io_wait)
bool io_may(const Deadline* by) {
    if (g_term.load()) {
        t_io_why = "this side was told to stop";
        return false;
    }
    return by == nullptr || Clock::now() < *by;
}

bool write_io(int fd, const void* p, size_t n, const Deadline* by) {
    const char* c = (const char*) p;
    int64_t since = 0;
    bool waiting = false;
    if (by == nullptr) t_io_why = nullptr;
    while (n > 0) {
        if (!io_may(by)) return false;
        const ssize_t w = ::send(fd, c, n, MSG_NOSIGNAL | MSG_DONTWAIT);
        if (w > 0) {
            c += w;
            n -= (size_t) w;
            waiting = false;
            continue;
        }
        if (w < 0 && errno == EINTR) continue;
        if (w == 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) return false;
        if (!waiting) since = now_ms();
        waiting = true;
        if (!io_wait(fd, POLLOUT, by, since)) return false;
    }
    return true;
}

bool read_io(int fd, void* p, size_t n, const Deadline* by) {
    char* c = (char*) p;
    int64_t since = 0;
    bool waiting = false;
    if (by == nullptr) t_io_why = nullptr;
    while (n > 0) {
        if (!io_may(by)) return false;
        const ssize_t r = ::recv(fd, c, n, MSG_DONTWAIT);
        if (r > 0) {
            c += r;
            n -= (size_t) r;
            waiting = false;
            continue;
        }
        if (r < 0 && errno == EINTR) continue;
        if (r == 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) return false;   // (0: the other side closed)
        if (!waiting) since = now_ms();
        waiting = true;
        if (!io_wait(fd, POLLIN, by, since)) return false;
    }
    return true;
}

bool write_all(int fd, const void* p, size_t n) { return write_io(fd, p, n, nullptr); }
bool read_all(int fd, void* p, size_t n) { return read_io(fd, p, n, nullptr); }
bool write_by(int fd, const void* p, size_t n, Deadline by) { return write_io(fd, p, n, &by); }
bool read_by(int fd, void* p, size_t n, Deadline by) { return read_io(fd, p, n, &by); }

// header and payload in one call (TCP_NODELAY: a lone header would leave as its own segment)
bool send_msg(const Hdr& h, const void* data, size_t bytes) {
    struct iovec iov[2];
    iov[0].iov_base = (void*) &h;
    iov[0].iov_len = sizeof h;
    iov[1].iov_base = (void*) data;
    iov[1].iov_len = bytes;
    size_t total = sizeof h + bytes, done = 0;
    int i0 = 0;
    int64_t since = 0;
    bool waiting = false;
    t_io_why = nullptr;
    while (done < total) {
        if (!io_may(nullptr)) return false;
        struct msghdr m {};
        m.msg_iov = iov + i0;
        m.msg_iovlen = (size_t) (bytes > 0 ? 2 - i0 : 1);
        const ssize_t w = ::sendmsg(g_data, &m, MSG_NOSIGNAL | MSG_DONTWAIT);
        if (w < 0 && errno == EINTR) continue;
        if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            if (!waiting) since = now_ms();
            waiting = true;
            if (!io_wait(g_data, POLLOUT, nullptr, since)) return false;
            continue;
        }
        if (w <= 0) return false;
        waiting = false;
        done += (size_t) w;
        size_t adv = (size_t) w;
        while (adv > 0 && i0 < 2) {
            if (adv >= iov[i0].iov_len) {
                adv -= iov[i0].iov_len;
                iov[i0].iov_len = 0;
                ++i0;
            } else {
                iov[i0].iov_base = (char*) iov[i0].iov_base + adv;
                iov[i0].iov_len -= adv;
                adv = 0;
            }
        }
    }
    return true;
}

void tune(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    int buf = 8 << 20;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof buf);
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof buf);
    // A PEER THAT IS GONE WITHOUT A WORD (its power, a cable, a partition): nothing else would end a blocked read
    // or an unacknowledged write.  Keepalive probes and TCP_USER_TIMEOUT do, after MAYA_GLM_LINK_DEAD_S seconds (30;
    // 0: off).  A peer that is only busy still answers the probes - its kernel does
    static const int dead_s = [] {
        const char* v = std::getenv("MAYA_GLM_LINK_DEAD_S");
        return v != nullptr && v[0] ? std::max(0, std::atoi(v)) : 30;
    }();
    if (dead_s > 0) {
        setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof one);
#if defined(TCP_KEEPIDLE) && defined(TCP_KEEPINTVL) && defined(TCP_KEEPCNT)
        int idle = std::max(1, dead_s / 3), intvl = std::max(1, dead_s / 6), cnt = 4;
        setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof idle);
        setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &intvl, sizeof intvl);
        setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, &cnt, sizeof cnt);
#endif
#if defined(TCP_USER_TIMEOUT)
        unsigned int ut = (unsigned int) dead_s * 1000u;
        setsockopt(fd, IPPROTO_TCP, TCP_USER_TIMEOUT, &ut, sizeof ut);
#endif
    }
}

// THE BACK'S PEERS: MAYA_GLM_LINK_ALLOW = comma-separated IPv4 addresses the
// back accepts a front from; unset: loopback (127.0.0.0/8) only.  Any other source is closed at accept().
bool peer_allowed(int fd, std::string& ip) {
    struct sockaddr_in a {};
    socklen_t al = sizeof a;
    ip = "?";
    if (::getpeername(fd, (struct sockaddr*) &a, &al) != 0 || a.sin_family != AF_INET) return false;
    char buf[INET_ADDRSTRLEN] = {0};
    if (::inet_ntop(AF_INET, &a.sin_addr, buf, sizeof buf) == nullptr) return false;
    ip = buf;
    const char* al_env = std::getenv("MAYA_GLM_LINK_ALLOW");
    if (al_env == nullptr || !al_env[0]) return (ntohl(a.sin_addr.s_addr) >> 24) == 127;
    const std::string list = al_env;
    size_t i = 0;
    while (i <= list.size()) {
        size_t j = list.find(',', i);
        if (j == std::string::npos) j = list.size();
        std::string t = list.substr(i, j - i);
        while (!t.empty() && t.front() == ' ') t.erase(t.begin());
        while (!t.empty() && t.back() == ' ') t.pop_back();
        struct in_addr x {};
        if (!t.empty() && ::inet_pton(AF_INET, t.c_str(), &x) == 1 && x.s_addr == a.sin_addr.s_addr) return true;
        i = j + 1;
    }
    return false;
}

// the back's address, resolved once (the front's start): a reconnect inside a command must not wait on a resolver
struct sockaddr_storage g_peer_sa {};
socklen_t g_peer_len = 0;

bool resolve_peer(const std::string& host, const std::string& port) {
    if (g_peer_len != 0) return true;
    struct addrinfo hints {}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0 || res == nullptr) return false;
    std::memcpy(&g_peer_sa, res->ai_addr, res->ai_addrlen);
    g_peer_len = (socklen_t) res->ai_addrlen;
    freeaddrinfo(res);
    return true;
}

// one attempt to connect, over by `by` and after kConnectS seconds at most (a host that does not answer: seconds
// here, not the kernel's minutes of SYN retries)
int connect_once(Deadline by) {
    constexpr int kConnectS = 5;
    if (g_peer_len == 0 || Clock::now() >= by) return -1;
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    const int fl = ::fcntl(fd, F_GETFL, 0);
    ::fcntl(fd, F_SETFL, fl | O_NONBLOCK);
    bool ok = ::connect(fd, (const struct sockaddr*) &g_peer_sa, g_peer_len) == 0;
    if (!ok && (errno == EINPROGRESS || errno == EINTR)) {
        const Deadline until = std::min(by, Clock::now() + std::chrono::seconds(kConnectS));
        for (;;) {
            const auto left =
                std::chrono::duration_cast<std::chrono::milliseconds>(until - Clock::now()).count();
            if (left <= 0) break;
            struct pollfd pf {fd, POLLOUT, 0};
            const int pr = ::poll(&pf, 1, (int) std::min<long long>(left, 1000));
            if (pr < 0 && errno != EINTR) break;
            if (pr <= 0) continue;
            int soerr = 0;
            socklen_t sl = sizeof soerr;
            ok = ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &sl) == 0 && soerr == 0;
            break;
        }
    }
    if (!ok) {
        ::close(fd);
        return -1;
    }
    ::fcntl(fd, F_SETFL, fl);
    tune(fd);
    return fd;
}

// the socket's own address (peer: the one it is connected to) is IPv4 loopback
bool sock_is_loopback(int fd, bool peer) {
    struct sockaddr_in a {};
    socklen_t al = sizeof a;
    if ((peer ? ::getpeername(fd, (struct sockaddr*) &a, &al) : ::getsockname(fd, (struct sockaddr*) &a, &al)) != 0)
        return false;
    return a.sin_family == AF_INET && (ntohl(a.sin_addr.s_addr) >> 24) == 127;
}
#endif

// LINK AUTHENTICATION: MAYA_GLM_LINK_KEY_FILE names a file holding a shared secret (at least
// 32 bytes; trailing whitespace is dropped).  With it, each connection (DATA, CTRL) proves both ways that the other
// side holds the same key before anything else crosses: the back sends a 32-byte nonce, the front answers with its
// own nonce and HMAC-SHA256(key, "front" label | tag | back nonce | front nonce), the back checks it (constant time;
// a failure is logged and the connection closed, the back keeps listening) and answers with HMAC-SHA256(key, "back"
// label | tag | front nonce | back nonce), which the front checks.  The env set and the file missing or short: the
// process does not start.  Unset: the process does not start either, unless the link is on loopback or
// MAYA_GLM_LINK_NO_AUTH=1 (a trusted network: the allowlist alone).  The link is not encrypted.  Self-contained SHA-256 (FIPS 180-4).
#if !defined(_WIN32)
struct Sha256 {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    unsigned char buf[64];
    uint64_t len = 0;
    size_t fill = 0;
    static uint32_t ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void block(const unsigned char* p) {
        static const uint32_t K[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t) p[4 * i] << 24 | (uint32_t) p[4 * i + 1] << 16 | (uint32_t) p[4 * i + 2] << 8 | p[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = hh + (ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
            const uint32_t t2 = (ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    void update(const void* data, size_t n) {
        const unsigned char* p = (const unsigned char*) data;
        len += n;
        while (n > 0) {
            const size_t k = std::min(n, (size_t) 64 - fill);
            std::memcpy(buf + fill, p, k);
            fill += k; p += k; n -= k;
            if (fill == 64) { block(buf); fill = 0; }
        }
    }
    void final(unsigned char out[32]) {
        const uint64_t bits = len * 8;
        const unsigned char one = 0x80, zero = 0;
        update(&one, 1);
        while (fill != 56) update(&zero, 1);
        unsigned char lb[8];
        for (int i = 0; i < 8; ++i) lb[i] = (unsigned char) (bits >> (56 - 8 * i));
        update(lb, 8);
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 4; ++j) out[4 * i + j] = (unsigned char) (h[i] >> (24 - 8 * j));
    }
};

// HMAC-SHA256 (RFC 2104) over the concatenation of parts
void hmac_sha256(const std::string& key, const std::vector<std::pair<const void*, size_t>>& parts, unsigned char out[32]) {
    unsigned char k0[64] = {0};
    if (key.size() > 64) {
        Sha256 s;
        s.update(key.data(), key.size());
        s.final(k0);
    } else {
        std::memcpy(k0, key.data(), key.size());
    }
    unsigned char ipad[64], opad[64], inner[32];
    for (int i = 0; i < 64; ++i) { ipad[i] = k0[i] ^ 0x36; opad[i] = k0[i] ^ 0x5c; }
    Sha256 si;
    si.update(ipad, 64);
    for (const auto& p : parts) si.update(p.first, p.second);
    si.final(inner);
    Sha256 so;
    so.update(opad, 64);
    so.update(inner, 32);
    so.final(out);
}

bool ct_equal(const unsigned char* a, const unsigned char* b, size_t n) {
    volatile unsigned char d = 0;
    for (size_t i = 0; i < n; ++i) d = d | (unsigned char) (a[i] ^ b[i]);
    return d == 0;
}
#endif

std::string g_key;          // the link's shared key (empty: no authentication)
bool g_loopback_only = false;   // no key and no MAYA_GLM_LINK_NO_AUTH: the sockets themselves must be on loopback
int g_key_state = -1;       // -1 not read yet, 0 failed, 1 read (or no key file set)
std::string g_key_err;

bool load_key() {
    if (g_key_state >= 0) return g_key_state == 1;
    g_key_state = 0;
    const char* path = env1("MAYA_GLM_LINK_KEY_FILE");
    if (path == nullptr || !path[0]) {
        // no key: only a link on a numeric loopback address - the one the front connects to, the one the back
        // listens on (MAYA_GLM_LINK_BIND over the --link host, as open() takes it) - or MAYA_GLM_LINK_NO_AUTH=1
        std::string host, port;
        const char* lk = env1("MAYA_GLM_LINK");
        bool loop = lk != nullptr && split_hostport(lk, host, port);
        if (loop && role() == kBack)
            if (const char* bind = env1("MAYA_GLM_LINK_BIND")) host = bind;
        loop = loop && is_loopback_v4(host);
        const char* na = env1("MAYA_GLM_LINK_NO_AUTH");
        const bool no_auth = na != nullptr && std::atoi(na) != 0;
        if (loop || no_auth) {
            g_loopback_only = !no_auth;   // (open() checks the connected / bound address too)
            g_key_state = 1;
            return true;
        }
        g_key_err = "glm link: set MAYA_GLM_LINK_KEY_FILE (the same key file of at least 32 bytes on both rigs), or "
                    "MAYA_GLM_LINK_NO_AUTH=1 on a trusted network";
        return false;
    }
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) {
        g_key_err = std::string("glm link: MAYA_GLM_LINK_KEY_FILE ") + path + " cannot be read: " + std::strerror(errno);
        return false;
    }
    std::string k;
    char b[256];
    size_t r;
    while ((r = std::fread(b, 1, sizeof b, f)) > 0 && k.size() < 4096) k.append(b, r);
    std::fclose(f);
    while (!k.empty() && (k.back() == '\n' || k.back() == '\r' || k.back() == ' ' || k.back() == '\t')) k.pop_back();
    if (k.size() < 32) {
        g_key_err = std::string("glm link: MAYA_GLM_LINK_KEY_FILE ") + path + " holds " + std::to_string(k.size()) +
                    " bytes, at least 32 are needed";
        return false;
    }
    g_key = std::move(k);
    g_key_state = 1;
    return true;
}

#if !defined(_WIN32)
constexpr size_t kNonce = 32;
// A LINK'S TWO CONNECTIONS BELONG TOGETHER: the front draws one session id per pair and sends it after each
// connection's tag; the proofs cover it, and the back pairs a DATA and a CTRL connection only when theirs are equal
// (two fronts starting at once on one address must not be assembled into one link)
constexpr size_t kSid = 16;
const char kLabelFront[] = "maya-glm-link-auth-v1 front";
const char kLabelBack[] = "maya-glm-link-auth-v1 back";

bool random_bytes(unsigned char* p, size_t n) {
    while (n > 0) {
        const ssize_t r = ::getrandom(p, n, 0);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        p += r;
        n -= (size_t) r;
    }
    return true;
}

// the back, on an accepted connection (its tag read): true when the front proved the key and got the back's proof
bool auth_back(int fd, const char tag[4], const unsigned char sid[kSid], Deadline deadline, std::string& why) {
    if (g_key.empty()) return true;
    unsigned char nb[kNonce], nf[kNonce], mf[32], want[32], mb[32];
    if (!random_bytes(nb, kNonce)) { why = "no random bytes for the nonce"; return false; }
    if (!write_by(fd, nb, kNonce, deadline)) { why = "the nonce did not go out"; return false; }
    if (!read_by(fd, nf, kNonce, deadline) || !read_by(fd, mf, 32, deadline)) {
        why = "no authentication reply in time (no key on the front?)";
        return false;
    }
    hmac_sha256(g_key, {{kLabelFront, sizeof kLabelFront - 1}, {tag, 4}, {sid, kSid}, {nb, kNonce}, {nf, kNonce}}, want);
    if (!ct_equal(want, mf, 32)) { why = "authentication failed (wrong link key)"; return false; }
    hmac_sha256(g_key, {{kLabelBack, sizeof kLabelBack - 1}, {tag, 4}, {sid, kSid}, {nf, kNonce}, {nb, kNonce}}, mb);
    if (!write_by(fd, mb, 32, deadline)) { why = "the proof did not go out"; return false; }
    return true;
}

// the front, on a connection (its tag sent): true when both proofs check out, all of it by `by`
bool auth_front(int fd, const char tag[4], const unsigned char sid[kSid], Deadline by, std::string& why) {
    if (g_key.empty()) return true;
    unsigned char nb[kNonce], nf[kNonce], mf[32], want[32], mb[32];
    if (!read_by(fd, nb, kNonce, by)) {
        why = "the back sent no nonce in time (no key on the back, it refused this address, or it is busy)";
        return false;
    }
    if (!random_bytes(nf, kNonce)) { why = "no random bytes for the nonce"; return false; }
    hmac_sha256(g_key, {{kLabelFront, sizeof kLabelFront - 1}, {tag, 4}, {sid, kSid}, {nb, kNonce}, {nf, kNonce}}, mf);
    if (!write_by(fd, nf, kNonce, by) || !write_by(fd, mf, 32, by)) { why = "the proof did not go out"; return false; }
    if (!read_by(fd, mb, 32, by)) {
        why = "no proof from the back in time: it rejected this side's (wrong link key), or the wait ran out";
        return false;
    }
    hmac_sha256(g_key, {{kLabelBack, sizeof kLabelBack - 1}, {tag, 4}, {sid, kSid}, {nf, kNonce}, {nb, kNonce}}, want);
    if (!ct_equal(want, mb, 32)) { why = "the back's proof is wrong (wrong link key)"; return false; }
    return true;
}
#endif

}  // namespace

Role role() {
    static const Role r = [] {
        const char* v = env1("MAYA_GLM_ROLE");
        if (v == nullptr || !v[0]) return kNone;
        if (std::strcmp(v, "front") == 0) return kFront;
        if (std::strcmp(v, "back") == 0) return kBack;
        std::fprintf(stderr, "glm link: MAYA_GLM_ROLE=%s is not front|back - ignored\n", v);
        return kNone;
    }();
    return r;
}

const char* role_name() { return role() == kFront ? "front" : role() == kBack ? "back" : "none"; }
bool up() { return g_up.load(); }
int ctl_fd() { return role() == kBack ? g_ctl : -1; }
void set_split(int split_at) { g_split = split_at; }
void set_model(const std::string& id) { g_model_id = id; }
void set_sched(const std::string& cfg) { g_sched = cfg.empty() ? "1" : cfg; }
uint64_t generation() { return g_gen.load(); }
void set_wait_hook(std::function<bool()> hook) { g_wait_hook = std::move(hook); }
int split_at() { return g_split; }
void set_stay(bool stay) { g_stay = stay; }
void stop_hint() {   // (the first STOP of a request starts its budget; later ones leave it)
    int64_t none = 0;
    g_stop_ms.compare_exchange_strong(none, std::max<int64_t>(1, std::chrono::duration_cast<std::chrono::milliseconds>(
                                                                  std::chrono::steady_clock::now().time_since_epoch()).count()));
}
void stop_clear() {
    g_stop_ms.store(0);
    g_stop_waited.store(0);
}

bool key_ready(std::string& err) {
    if (load_key()) return true;
    err = g_key_err;
    return false;
}

#if defined(_WIN32)
bool open(int, int64_t, int, int&, std::string& err) {
    err = "glm link: not on Windows";
    return false;
}
bool send_hop(int64_t, int, const void*, size_t, std::string& err) { err = "glm link: not on Windows"; return false; }
bool recv_hop(int64_t, int, void*, size_t, std::string& err) { err = "glm link: not on Windows"; return false; }
bool send_tok(Type, int64_t, int32_t) { return false; }
void abort_link(const std::string&) {}
bool recv_tok(Type, int64_t, int32_t&, std::string& err) { err = "glm link: not on Windows"; return false; }
bool send_pair(Type, int64_t, int32_t, int32_t) { return false; }
bool recv_pair(Type, int64_t, int32_t&, int32_t&, std::string& err) { err = "glm link: not on Windows"; return false; }
void ctl_send_line(const std::string&) {}
void ctl_close() {}
void report() {}
void request_begin() {}
bool cancelled() { return false; }
bool begin_exchange(int64_t, int32_t, int64_t&, int32_t&, std::string& err) { err = "glm link: not on Windows"; return false; }
bool chunk_exchange(int64_t, int32_t, int32_t, int32_t&, std::string& err) { err = "glm link: not on Windows"; return false; }
bool end_front(int64_t, int32_t, bool&, std::string& err) { err = "glm link: not on Windows"; return false; }
bool end_back(int64_t, int32_t&, bool&, std::string& err) { err = "glm link: not on Windows"; return false; }
bool bf16() { return false; }
bool idle_check() { return false; }
bool reopen(std::string& err) { err = "glm link: not on Windows"; return false; }
void wake() {}
void hangup() {}
bool reaccept(std::string& err) { err = "glm link: not on Windows"; return false; }
bool readable() { return false; }
int stall_seconds() { return 0; }
void term() {}
long ctl_read(char*, size_t) { return 0; }
#else
// this side's settings both must share: the hop's format and the prompt's chunking floor (the chunks must match)
// and the model each side loaded
static std::string link_config() {
    const char* mc = std::getenv("MAYA_GLM_PREFILL_MIN_CHUNKS");
    const char* pm = std::getenv("MAYA_GLM_PREFILL_MIN");   // (below it a prompt goes token by token: both or neither)
    return std::string("bf16=") + (g_bf16 ? "1" : "0") + " pack=" + (g_bf16 && g_pack ? "1" : "0") +
           " min_chunks=" + (mc != nullptr ? mc : "") + " prefill_min=" + (pm != nullptr ? pm : "") +
           " parallel=" + g_sched + " model=" + g_model_id;
}

// (all of it by `by`, sent and read: a side that connected but hangs, or sends a byte now and then, must not hold
// the other - the back's accept loop above all)
static bool hello(std::string& err, Deadline by) {
    const std::string cfg = link_config();
    Hdr h{kMagic, kHello, kVersion, g_split, g_my_chunk, (uint64_t) g_ctx}, p{};
    const uint32_t cl = (uint32_t) cfg.size();
    std::string out((const char*) &h, sizeof h);
    out.append((const char*) &cl, 4);
    out.append(cfg);
    bool ok = write_by(g_data, out.data(), out.size(), by) && read_by(g_data, &p, sizeof p, by);
    uint32_t pl = 0;
    std::string pcfg;
    if (ok && p.magic == kMagic && p.type == kHello && p.pos == kVersion) {
        ok = read_by(g_data, &pl, 4, by) && pl < 4096;
        if (ok) {
            pcfg.assign(pl, '\0');
            ok = pl == 0 || read_by(g_data, &pcfg[0], pl, by);
        }
    }
    if (!ok) {
        err = g_term.load() ? "glm link: told to stop" : "glm link: HELLO failed (no answer from the other side in time)";
        return false;
    }
    if (p.magic != kMagic || p.type != kHello || p.pos != kVersion) {
        err = "glm link: the other side speaks another protocol";
        return false;
    }
    if (p.n != g_split || (int64_t) p.bytes != g_ctx) {
        err = "glm link: the other side splits at " + std::to_string(p.n) + " with context " +
              std::to_string((long long) p.bytes) + ", this one at " + std::to_string(g_split) + " with " +
              std::to_string((long long) g_ctx);
        return false;
    }
    if (pcfg != cfg) {
        const size_t a = pcfg.find(" model="), b = cfg.find(" model=");
        const size_t pa = pcfg.find(" parallel="), pb = cfg.find(" parallel=");
        if (a != std::string::npos && b != std::string::npos && pcfg.substr(a) != cfg.substr(b))
            err = "glm link: the other side loaded another model (" + pcfg.substr(a + 7) + ") than this one (" +
                  cfg.substr(b + 7) + ") - both need the same pack";
        else if (pa != std::string::npos && pb != std::string::npos && a != std::string::npos && b != std::string::npos &&
                 pa < a && pb < b && pcfg.substr(pa, a - pa) != cfg.substr(pb, b - pb))
            // (the two schedulers run in lock step: their settings must be the same)
            err = "glm link: the other side runs --parallel with '" + pcfg.substr(pa + 10, a - pa - 10) +
                  "', this one with '" + cfg.substr(pb + 10, b - pb - 10) + "' - both sides need the same --parallel "
                  "and the same MAYA_GLM_SEQ_CTX, MAYA_GLM_SPLIT_ROWS, MAYA_GLM_EVEN_GROUPS, MAYA_GLM_MULTI_BOUNDARY, "
                  "MAYA_GLM_MULTI_SPEC and MAYA_GLM_MULTI_ROOM";
        else
            err = "glm link: the other side runs with '" + pcfg + "', this one with '" + cfg + "'";
        return false;
    }
    // (a front that started again may have sized its prompt path differently.  That is no reason to refuse it: every
    // prompt-path call agrees on its chunk, or on going token by token - Glm5Model::prefill)
    if (g_peer_chunk >= 0 && p.val != g_peer_chunk)
        std::fprintf(stderr, "glm link: the other side's prompt chunk is %d now (%d before)\n", p.val, g_peer_chunk);
    g_peer_chunk = p.val;
    return true;
}

static void close_fds() {
    if (g_data >= 0) ::close(g_data);
    if (g_ctl >= 0) ::close(g_ctl);
    g_data = g_ctl = -1;
}

// the front: both connections (DATA, CTRL) by `by` - connecting (retried while the back is not there yet) and each
// connection's proof (kAuthS seconds of its own at most)
static bool connect_pair(const std::string& host, const std::string& port, Deadline by, double wait_s, std::string& err) {
    constexpr int kAuthS = 10;
    unsigned char sid[kSid] = {0};   // this pair's session id
    const auto wait_more = [&]() -> bool {   // (a quarter second before the next attempt, inside the budget)
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(by - Clock::now()).count();
        if (left <= 0) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(std::min<long long>(left, 250)));
        return Clock::now() < by;
    };
    while (!resolve_peer(host, port))
        if (!wait_more()) {
            err = "glm link: cannot resolve " + host;
            return false;
        }
    for (int which = 0; which < 2; ++which) {
        int fd = -1;
        while ((fd = connect_once(by)) < 0)
            if (!wait_more()) {
                close_fds();
                err = "glm link: no back at " + host + ":" + port + " after " + std::to_string((int) wait_s) + " s";
                return false;
            }
        if (g_loopback_only && !sock_is_loopback(fd, true)) {
            ::close(fd);
            close_fds();
            err = "glm link: without a key the link runs on loopback only, and " + host + " is not (set "
                  "MAYA_GLM_LINK_KEY_FILE, or MAYA_GLM_LINK_NO_AUTH=1 on a trusted network)";
            return false;
        }
        const char* tag = which == 0 ? "DATA" : "CTRL";
        if (which == 0 && !random_bytes(sid, kSid)) {
            ::close(fd);
            close_fds();
            err = "glm link: no random bytes for the session id";
            return false;
        }
        const Deadline auth_by = std::min(by, Clock::now() + std::chrono::seconds(kAuthS));
        if (!write_by(fd, tag, 4, auth_by) || !write_by(fd, sid, kSid, auth_by)) {
            ::close(fd);
            close_fds();
            err = "glm link: the connection's tag did not go out";
            return false;
        }
        std::string why;
        if (!auth_front(fd, tag, sid, auth_by, why)) {
            ::close(fd);
            close_fds();
            err = std::string("glm link: link authentication failed on the ") + tag + " connection: " + why;
            std::fprintf(stderr, "%s\n", err.c_str());
            return false;
        }
        (which == 0 ? g_data : g_ctl) = fd;
    }
    return true;
}

// the back: the front's two connections on the listening socket.  A connection that proved itself waits here for
// its companion - the same session id from the same address, the other kind - for kHandshakeS seconds of its own.
// The first pair that is whole is the link; what else waited is closed.  (Two fronts started at once on one back:
// one gets the link and the other's HELLO fails - neither can take the other's half away.)
static bool accept_pair(std::string& err) {
    constexpr int kHandshakeS = 10;      // a connection's tag and proof, and the wait for a pair's second connection
    constexpr int kTagS = 2;             // ... its tag and session id alone (a front sends them as it connects)
    constexpr size_t kMaxWaiting = 8;    // halves waiting at once (the oldest goes for a ninth)
    struct Half {
        int fd;
        bool is_data;
        std::string ip, sid;
        Deadline by;
    };
    std::vector<Half> waiting;
    const auto drop = [&](size_t i, const char* why) {
        if (why != nullptr)
            std::fprintf(stderr, "glm link: dropped half a connection pair from %s (%s)\n", waiting[i].ip.c_str(), why);
        ::close(waiting[i].fd);
        waiting.erase(waiting.begin() + (long) i);
    };
    const auto drop_all = [&] {
        while (!waiting.empty()) drop(waiting.size() - 1, nullptr);
    };
    for (;;) {
        if (g_term.load()) {
            drop_all();
            err = "glm link: told to stop";
            return false;
        }
        long long left = 1000;
        for (size_t i = 0; i < waiting.size();) {
            const long long l =
                std::chrono::duration_cast<std::chrono::milliseconds>(waiting[i].by - Clock::now()).count();
            if (l <= 0) {
                drop(i, "no second connection in 10 s");
                continue;
            }
            left = std::min(left, l);
            ++i;
        }
        // (the listening socket does not block: a look every second at most, for a half's deadline and for term)
        struct pollfd pf {g_listen, POLLIN, 0};
        const int pr = ::poll(&pf, 1, (int) left);
        if (pr < 0 && errno != EINTR) {
            drop_all();
            err = std::string("glm link: poll: ") + std::strerror(errno);
            return false;
        }
        if (pr <= 0) continue;
        const int fd = ::accept(g_listen, nullptr, nullptr);
        if (fd < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK || errno == ECONNABORTED) continue;
            drop_all();
            err = std::string("glm link: accept: ") + std::strerror(errno);
            return false;
        }
        if (const int fl = ::fcntl(fd, F_GETFL, 0); fl >= 0 && (fl & O_NONBLOCK)) ::fcntl(fd, F_SETFL, fl & ~O_NONBLOCK);
        std::string ip;
        if (!peer_allowed(fd, ip)) {
            std::fprintf(stderr, "glm link: refused a connection from %s (not in MAYA_GLM_LINK_ALLOW)\n", ip.c_str());
            ::close(fd);
            continue;
        }
        tune(fd);
        char tag[4];
        unsigned char sid[kSid];
        // HANDSHAKES RUN ONE AT A TIME, so the time one takes is given back to the halves that wait: their
        // companions may sit in the listen queue meanwhile (a probe that connects and says nothing costs kTagS)
        const Deadline t_in = Clock::now();
        const auto give_back = [&] {
            const auto took = Clock::now() - t_in;
            for (Half& h : waiting) h.by += took;
        };
        const Deadline own = t_in + std::chrono::seconds(kHandshakeS), tag_by = t_in + std::chrono::seconds(kTagS);
        if (!read_by(fd, tag, 4, tag_by) || (std::memcmp(tag, "DATA", 4) != 0 && std::memcmp(tag, "CTRL", 4) != 0) ||
            !read_by(fd, sid, kSid, tag_by)) {
            ::close(fd);
            give_back();
            continue;
        }
        const bool is_data = std::memcmp(tag, "DATA", 4) == 0;
        const std::string sid_s((const char*) sid, kSid);
        const auto find = [&](bool data_kind) -> long {
            for (size_t i = 0; i < waiting.size(); ++i)
                if (waiting[i].is_data == data_kind && waiting[i].sid == sid_s && waiting[i].ip == ip) return (long) i;
            return -1;
        };
        // (a pair's second connection proves itself within what is left of the first one's time)
        const long mate = find(!is_data);
        const Deadline mate_by = mate >= 0 ? waiting[(size_t) mate].by : own;
        std::string why;
        if (!auth_back(fd, tag, sid, std::min(own, mate_by), why)) {
            std::fprintf(stderr, "glm link: refused a connection from %s: %s\n", ip.c_str(), why.c_str());
            ::close(fd);
            give_back();
            continue;
        }
        if (mate >= 0 && Clock::now() >= mate_by) {
            // (too late for its pair, with a key or without: the half itself goes at the top of the loop)
            ::close(fd);
            continue;
        }
        if (g_term.load()) {   // (told to stop meanwhile: no pair is taken up)
            ::close(fd);
            drop_all();
            err = "glm link: told to stop";
            return false;
        }
        if (mate >= 0) {
            (is_data ? g_data : g_ctl) = fd;
            (is_data ? g_ctl : g_data) = waiting[(size_t) mate].fd;
            waiting.erase(waiting.begin() + mate);
            if (!waiting.empty())
                std::fprintf(stderr, "glm link: %zu connection(s) of other sessions closed (one front at a time)\n",
                             waiting.size());
            drop_all();
            return true;
        }
        // the first of its pair.  (The same session's same kind once more replaces the one that waits)
        if (const long same = find(is_data); same >= 0) drop((size_t) same, nullptr);
        if (waiting.size() >= kMaxWaiting) {
            size_t oldest = 0;
            for (size_t i = 1; i < waiting.size(); ++i)
                if (waiting[i].by < waiting[oldest].by) oldest = i;
            drop(oldest, "too many connections waiting for their second one");
        }
        give_back();
        waiting.push_back({fd, is_data, ip, sid_s, Clock::now() + std::chrono::seconds(kHandshakeS)});
    }
}

constexpr int kHelloS = 60;   // HELLO, sent and read, when a link opens (a reconnect: what is left of its budget)

static bool link_hostport(std::string& host, std::string& port, std::string& err) {
    const char* lk = env1("MAYA_GLM_LINK");
    if (lk == nullptr || !split_hostport(lk, host, port)) {
        err = "glm link: MAYA_GLM_LINK=host:port (or --link) is needed with --role";
        return false;
    }
    return true;
}

bool open(int split_at, int64_t max_ctx, int my_chunk, int& peer_chunk, std::string& err) {
    std::string host, port;
    if (!link_hostport(host, port, err)) return false;
    g_split = split_at;
    g_ctx = max_ctx;
    g_my_chunk = my_chunk;
    if (const char* b = env1("MAYA_GLM_LINK_BF16")) g_bf16 = std::atoi(b) != 0;
    if (const char* b = env1("MAYA_GLM_LINK_PACK")) g_pack = std::atoi(b) != 0;
    if (!g_bf16) g_pack = false;
    if (!key_ready(err)) return false;
    std::fprintf(stderr, "glm link: authentication %s\n",
                 g_key.empty() ? "OFF (no MAYA_GLM_LINK_KEY_FILE; the allowlist only)" : "on (HMAC-SHA256 challenge-response, both ways)");
    const auto t0 = std::chrono::steady_clock::now();
    if (role() == kFront) {
        double wait_s = 1800;
        if (const char* w = std::getenv("MAYA_GLM_LINK_WAIT")) wait_s = std::atof(w);
        std::fprintf(stderr, "glm link: front connecting to %s:%s\n", host.c_str(), port.c_str());
        wait_s = std::max(1.0, wait_s);
        const Deadline by = t0 + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(wait_s));
        if (!connect_pair(host, port, by, wait_s, err)) return false;
    } else {
        // the address to listen on: MAYA_GLM_LINK_BIND / MAYA_GLM_LINK_BIND, else the --link host (empty: loopback)
        if (const char* b = env1("MAYA_GLM_LINK_BIND"); b != nullptr && b[0]) host = b;
        struct addrinfo hints {}, *res = nullptr;
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_flags = AI_NUMERICSERV;
        if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0 || res == nullptr) {
            err = "glm link: cannot resolve " + host;
            return false;
        }
        const int ls = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        int one = 1;
        setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        if (ls < 0 || ::bind(ls, res->ai_addr, res->ai_addrlen) != 0 || ::listen(ls, 4) != 0) {
            freeaddrinfo(res);
            err = std::string("glm link: cannot listen on ") + host + ":" + port + ": " + std::strerror(errno);
            return false;
        }
        freeaddrinfo(res);
        if (g_loopback_only && !sock_is_loopback(ls, false)) {
            ::close(ls);
            err = "glm link: without a key the link runs on loopback only, and " + host + " is not (set "
                  "MAYA_GLM_LINK_KEY_FILE, or MAYA_GLM_LINK_NO_AUTH=1 on a trusted network)";
            return false;
        }
        ::fcntl(ls, F_SETFL, ::fcntl(ls, F_GETFL, 0) | O_NONBLOCK);   // (accept_pair polls it)
        g_listen = ls;   // (kept by a back that stays: the next front is accepted on it - reaccept)
        g_listening.store(true);
        {
            const char* al = std::getenv("MAYA_GLM_LINK_ALLOW");
            std::fprintf(stderr, "glm link: back listening on %s:%s, fronts accepted from %s\n", host.c_str(), port.c_str(),
                         al != nullptr && al[0] ? al : "loopback only (set MAYA_GLM_LINK_ALLOW)");
        }
        for (;;) {
            if (!accept_pair(err)) return false;
            if (hello(err, Clock::now() + std::chrono::seconds(kHelloS))) break;
            close_fds();
            // (a back that stays is not ended by one front's wrong settings: the next one may have the right ones)
            if (!g_stay || g_term.load()) return false;
            std::fprintf(stderr, "%s - waiting for another front\n", err.c_str());
        }
        if (g_term.load()) {
            close_fds();
            err = "glm link: told to stop";
            return false;
        }
        if (!g_stay) {
            // one front, once (a single run, MAYA_GLM_BACK_STAY=0): nobody accepts here again, so a later connect
            // is refused at once instead of waiting in a queue (the next run's front finds the next run's back)
            ::close(g_listen);
            g_listen = -1;
        }
    }
    if (role() == kFront && !hello(err, Clock::now() + std::chrono::seconds(kHelloS))) {
        close_fds();
        return false;
    }
    peer_chunk = g_peer_chunk;
    g_gen.fetch_add(1);
    g_up.store(true);
    std::fprintf(stderr, "glm link: %s up in %.1f s (boundary layer %d, context %lld, prompt chunks %d / %d, %s)\n",
                 role_name(), ms_since(t0) / 1000.0, split_at, (long long) max_ctx, my_chunk, peer_chunk,
                 link_config().c_str());
    std::atexit(report);
    return true;
}

bool bf16() { return g_bf16; }

bool idle_check() {
    if (!up() || g_data < 0) return false;
    struct pollfd pf {g_data, POLLIN | POLLRDHUP, 0};
    // between requests nothing is due from the back: anything readable is its end (EOF) or an ERR
    if (::poll(&pf, 1, 0) > 0 && (pf.revents & (POLLIN | POLLRDHUP | POLLHUP | POLLERR))) {
        std::string err;
        fail(err, "the back has gone");
        return false;
    }
    return true;
}

bool reopen(std::string& err) {
    std::string host, port;
    if (!link_hostport(host, port, err)) return false;
    g_up.store(false);
    {
        std::lock_guard<std::mutex> lk(g_ctl_mu);
        close_fds();
        // (MAYA_GLM_RELINK_WAIT seconds for all of it - connecting, both proofs, HELLO: a command does not wait
        // longer than that for a back that is not there, or is there and does not answer)
        double wait_s = 5;
        if (const char* w = std::getenv("MAYA_GLM_RELINK_WAIT")) wait_s = std::atof(w);
        wait_s = std::max(1.0, wait_s);
        const auto t0 = std::chrono::steady_clock::now();
        const Deadline by = t0 + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(wait_s));
        if (!connect_pair(host, port, by, wait_s, err)) return false;
        if (!hello(err, by)) {
            close_fds();
            return false;
        }
        std::fprintf(stderr, "glm link: front connected again in %.1f s\n", ms_since(t0) / 1000.0);
    }
    g_gen.fetch_add(1);
    g_up.store(true);
    return true;
}

void wake() {
    g_up.store(false);
    if (g_data >= 0) ::shutdown(g_data, SHUT_RDWR);
}

// (the descriptors are not closed here: the reader thread, which owns the next accept, closes them in reaccept)
void hangup() {
    g_up.store(false);
    if (g_data >= 0) ::shutdown(g_data, SHUT_RDWR);
    if (g_ctl >= 0) ::shutdown(g_ctl, SHUT_RDWR);
}

// (from a signal handler: atomics and _exit only.  Nothing is woken from here - every wait of this side looks at
// g_term once a second: io_wait, accept_pair, ctl_read - so no descriptor is touched outside its owner's thread)
void term() {
    if (g_term.exchange(true) || !g_listening.load()) ::_exit(130);   // (a second signal, or nothing to end in order yet)
}

long ctl_read(char* p, size_t n) {
    for (;;) {
        if (g_term.load() || g_ctl < 0) return 0;
        struct pollfd pf {g_ctl, POLLIN, 0};
        const int pr = ::poll(&pf, 1, 1000);
        if (pr < 0 && errno != EINTR) return -1;
        if (pr <= 0) continue;
        const ssize_t r = ::read(g_ctl, p, n);
        if (r < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        return (long) r;
    }
}

bool reaccept(std::string& err) {
    g_up.store(false);
    {
        std::lock_guard<std::mutex> lk(g_ctl_mu);
        close_fds();
    }
    if (g_listen < 0) {
        err = "glm link: not listening";
        return false;
    }
    if (g_term.load()) {
        err = "glm link: told to stop";
        return false;
    }
    std::fprintf(stderr, "glm link: the front has gone - waiting for it again\n");
    for (;;) {
        if (!accept_pair(err)) return false;
        if (hello(err, Clock::now() + std::chrono::seconds(kHelloS))) break;
        close_fds();
        if (g_term.load()) return false;
        std::fprintf(stderr, "%s - waiting for another front\n", err.c_str());
    }
    if (g_term.load()) {
        std::lock_guard<std::mutex> lk(g_ctl_mu);
        close_fds();
        err = "glm link: told to stop";
        return false;
    }
    g_gen.fetch_add(1);
    g_up.store(true);
    std::fprintf(stderr, "glm link: back up again\n");
    return true;
}

// THE BF16 HOP: the rows cross as bf16 (the top half of each fp32, rounded to nearest even - ggml's rounding), in
// blocks of kBlk values staged per thread, so the conversion of one block overlaps the wire of the one before
constexpr size_t kBlk = 1 << 20;
static inline uint16_t f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 64);   // NaN stays NaN
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

// THE PACKED HOP (MAYA_GLM_LINK_PACK=1, default with bf16; lossless): a bf16 value's high byte (sign and 7 exponent
// bits) takes few values in a residual row (entropy ~3.2 bits, the 15 commonest ~98.5%), so each block sends its low
// bytes as they are, the high bytes as 4-bit codes into the block's table of its 15 commonest (15: escape, the byte
// follows in the escape list) - ~12.1 bits a value instead of 16.  A block: u32 payload bytes, u32 values, u32 escapes,
// u8 table[16], then low bytes, nibbles, escapes.
struct PackHdr {
    uint32_t len, k, n_esc;
    uint8_t table[16];
};
static size_t pack_block(const float* src, size_t k, std::vector<uint8_t>& out) {
    thread_local std::vector<uint16_t> b;
    b.resize(k);
    uint32_t hist[256] = {0};
    for (size_t j = 0; j < k; ++j) {
        b[j] = f2bf(src[j]);
        ++hist[b[j] >> 8];
    }
    PackHdr h{};
    uint8_t code[256];
    std::memset(code, 15, sizeof code);
    for (int t = 0; t < 15; ++t) {   // the 15 commonest high bytes
        int best = -1;
        for (int v = 0; v < 256; ++v)
            if (code[v] == 15 && hist[v] > 0 && (best < 0 || hist[v] > hist[best])) best = v;
        if (best < 0) break;
        code[best] = (uint8_t) t;
        h.table[t] = (uint8_t) best;
    }
    size_t n_esc = 0;
    for (int v = 0; v < 256; ++v)
        if (code[v] == 15) n_esc += hist[v];
    const size_t nb = (k + 1) / 2, total = sizeof(PackHdr) + k + nb + n_esc;
    out.resize(total);
    uint8_t* lo = out.data() + sizeof(PackHdr);
    uint8_t* nib = lo + k;
    uint8_t* esc = nib + nb;
    std::memset(nib, 0, nb);
    size_t e = 0;
    for (size_t j = 0; j < k; ++j) {
        const uint8_t hi = (uint8_t) (b[j] >> 8), c = code[hi];
        lo[j] = (uint8_t) b[j];
        nib[j >> 1] |= (uint8_t) (c << ((j & 1) * 4));
        if (c == 15) esc[e++] = hi;
    }
    h.len = (uint32_t) (total - 4);
    h.k = (uint32_t) k;
    h.n_esc = (uint32_t) n_esc;
    std::memcpy(out.data(), &h, sizeof h);
    return total;
}
// p: the whole block as read (buf bytes: its u32 length + 4); false on a block that does not fit its own length
static bool unpack_block(const uint8_t* p, size_t buf, size_t k_want, uint32_t* dst) {
    PackHdr h;
    if (buf < sizeof h) return false;
    std::memcpy(&h, p, sizeof h);
    if (h.k != k_want || h.n_esc > h.k || (size_t) h.len + 4 != buf) return false;
    if (sizeof(PackHdr) + (size_t) h.k + ((size_t) h.k + 1) / 2 + (size_t) h.n_esc > buf) return false;
    const uint8_t* lo = p + sizeof(PackHdr);
    const uint8_t* nib = lo + h.k;
    const uint8_t* esc = nib + (h.k + 1) / 2;
    size_t e = 0;
    for (size_t j = 0; j < h.k; ++j) {
        const uint8_t c = (uint8_t) ((nib[j >> 1] >> ((j & 1) * 4)) & 15);
        if (c == 15 && (e >= h.n_esc || esc + e >= p + buf)) return false;   // (every escape inside the block)
        const uint8_t hi = c == 15 ? esc[e++] : h.table[c];
        dst[j] = ((uint32_t) hi << 24) | ((uint32_t) lo[j] << 16);
    }
    return e == h.n_esc;
}

bool send_hop(int64_t pos, int n, const void* data, size_t bytes, std::string& err) {
    if (!up()) { err = "glm link: down"; return false; }
    const auto t0 = std::chrono::steady_clock::now();
    bool ok;
    size_t wire_total = 0;
    if (!g_bf16) {
        Hdr h{kMagic, kHop, pos, n, 0, (uint64_t) bytes};
        ok = send_msg(h, data, bytes);
    } else if (g_pack) {
        // (the header carries the values, not the bytes: the blocks say their own length)
        thread_local std::vector<uint8_t> st;
        const size_t nv = bytes / sizeof(float);
        const float* src = (const float*) data;
        Hdr h{kMagic, kHop, pos, n, 2, (uint64_t) nv};
        ok = true;
        for (size_t i = 0; ok && i < nv; i += kBlk) {
            const size_t k = std::min(kBlk, nv - i);
            const size_t w = pack_block(src + i, k, st);
            wire_total += w;
            ok = i == 0 ? send_msg(h, st.data(), w) : write_all(g_data, st.data(), w);
        }
        if (nv == 0) ok = send_msg(h, nullptr, 0);
    } else {
        thread_local std::vector<uint16_t> st;
        const size_t nv = bytes / sizeof(float);
        const float* src = (const float*) data;
        Hdr h{kMagic, kHop, pos, n, 1, (uint64_t) (nv * 2)};
        st.resize(std::min(nv, kBlk));
        ok = true;
        for (size_t i = 0; ok && i < nv; i += kBlk) {
            const size_t k = std::min(kBlk, nv - i);
            for (size_t j = 0; j < k; ++j) st[j] = f2bf(src[i + j]);
            ok = i == 0 ? send_msg(h, st.data(), k * 2) : write_all(g_data, st.data(), k * 2);
        }
        if (nv == 0) ok = send_msg(h, nullptr, 0);
    }
    if (!ok) {
        fail(err, "HOP send failed");
        return false;
    }
    s_send_ms += ms_since(t0);
    ++s_hops;
    s_hop_bytes += g_pack ? wire_total : g_bf16 ? bytes / 2 : bytes;
    return true;
}

// read one header; an ERR from the other side becomes this side's error
// an ERR's text: at most kMaxErrBytes (a longer one is not read - the caller drops the link)
constexpr uint64_t kMaxErrBytes = 4096;
constexpr uint64_t kMaxHopBytes = 4ull << 30;   // a HOP's payload bound for dropped messages (an 8K chunk is 512 MB)
static bool read_err(const Hdr& h, std::string& why) {
    if (h.bytes > kMaxErrBytes) return false;
    why.assign((size_t) h.bytes, '\0');
    if (h.bytes > 0 && !read_all(g_data, &why[0], (size_t) h.bytes)) why = "?";
    return true;
}

int stall_seconds() { return stall_s(); }

bool readable() {
    if (g_data < 0) return true;
    struct pollfd pf {g_data, POLLIN, 0};
    return ::poll(&pf, 1, 0) != 0;
}

static bool recv_hdr(Hdr& h, uint32_t want, std::string& err) {
    // (several conversations, the back: its picked tokens go out while it waits - the hook says when none is left.
    // Not for longer than a wait on the link may last: then the read below waits, under its own bounds)
    if (g_wait_hook) {
        const int64_t t0 = now_ms();
        while (!readable() && g_wait_hook()) {
            if (g_term.load() || (stall_s() > 0 && now_ms() - t0 >= (int64_t) stall_s() * 1000)) break;
            std::this_thread::yield();
        }
    }
    if (!read_all(g_data, &h, sizeof h) || h.magic != kMagic) {
        fail(err, "the connection is lost");
        return false;
    }
    if (h.type == kEnd && want != kEnd && role() == kBack) {
        if (h.bytes != 0) {
            fail(err, "an END with a payload (a protocol error)");
            return false;
        }
        // THE FRONT ENDED THE REQUEST (a cancel): this side stops where it is, the link stays up
        g_cancel = true;
        g_end_pos = h.pos;
        g_end_flags = h.val;
        err = "cancelled";
        return false;
    }
    if (h.type == kErr) {
        std::string why;
        if (!read_err(h, why)) {
            fail(err, "an ERR of " + std::to_string((unsigned long long) h.bytes) + " bytes (over 4 KiB: a protocol error)");
            return false;
        }
        fail(err, "the other side failed: " + why);
        return false;
    }
    if (h.type != kHop && h.bytes != 0) {   // (only a HOP and an ERR carry a payload)
        fail(err, "message " + std::to_string(h.type) + " with a payload of " +
                      std::to_string((unsigned long long) h.bytes) + " bytes (a protocol error)");
        return false;
    }
    if (h.type != want) {
        fail(err, "expected message " + std::to_string(want) + ", got " + std::to_string(h.type) + " (the sides are out of step)");
        return false;
    }
    return true;
}

bool recv_hop(int64_t pos, int n, void* dst, size_t bytes, std::string& err) {
    if (!up()) { err = "glm link: down"; return false; }
    const auto t0 = std::chrono::steady_clock::now();
    Hdr h{};
    if (!recv_hdr(h, kHop, err)) return false;
    size_t wire = g_pack ? bytes / 4 : g_bf16 ? bytes / 2 : bytes;   // (packed: the header carries the values)
    if (h.pos != pos || h.n != n || h.bytes != wire || h.val != (g_pack ? 2 : g_bf16 ? 1 : 0)) {
        fail(err, "HOP for positions " + std::to_string((long long) h.pos) + "+" + std::to_string(h.n) +
                      ", expected " + std::to_string((long long) pos) + "+" + std::to_string(n) + " (out of step)");
        return false;
    }
    bool ok;
    if (!g_bf16) {
        ok = read_all(g_data, dst, bytes);
    } else if (g_pack) {
        thread_local std::vector<uint8_t> st;
        const size_t nv = bytes / sizeof(float);
        uint32_t* out = (uint32_t*) dst;
        ok = true;
        wire = 0;
        for (size_t i = 0; ok && i < nv; i += kBlk) {
            const size_t k = std::min(kBlk, nv - i);
            uint32_t len = 0;
            ok = read_all(g_data, &len, 4) && len >= sizeof(PackHdr) - 4 && len <= 4 * kBlk;
            if (!ok) break;
            st.resize(len + 4);
            std::memcpy(st.data(), &len, 4);
            if (!(ok = read_all(g_data, st.data() + 4, len))) break;
            if (!unpack_block(st.data(), (size_t) len + 4, k, out + i)) {
                fail(err, "a packed HOP block that does not fit its length (a protocol error)");
                return false;
            }
            wire += len + 4;
        }
    } else {
        thread_local std::vector<uint16_t> st;
        const size_t nv = bytes / sizeof(float);
        uint32_t* out = (uint32_t*) dst;
        st.resize(std::min(nv, kBlk));
        ok = true;
        for (size_t i = 0; ok && i < nv; i += kBlk) {
            const size_t k = std::min(kBlk, nv - i);
            ok = read_all(g_data, st.data(), k * 2);
            for (size_t j = 0; ok && j < k; ++j) out[i + j] = (uint32_t) st[j] << 16;
        }
    }
    if (!ok) {
        fail(err, "the connection is lost in a HOP");
        return false;
    }
    s_recv_ms += ms_since(t0);
    ++s_hops;
    s_hop_bytes += wire;
    return true;
}

bool send_tok(Type t, int64_t pos, int32_t tok) {
    if (!up()) return false;
    Hdr h{kMagic, (uint32_t) t, pos, 0, tok, 0};
    if (!send_msg(h, nullptr, 0)) {
        std::string err;
        fail(err, "TOK send failed");
        return false;
    }
    ++s_toks;
    return true;
}

void abort_link(const std::string& why) {
    if (!up() || g_data < 0) return;
    const size_t wn = std::min<size_t>(why.size(), 4096);   // (the receiver drops a longer ERR as a protocol error)
    Hdr e{kMagic, kErr, 0, 0, 0, (uint64_t) wn};
    send_msg(e, why.data(), wn);
    std::string err;
    fail(err, "aborted: " + why);
    ::shutdown(g_data, SHUT_RDWR);
}

bool recv_tok(Type t, int64_t pos, int32_t& tok, std::string& err) {
    if (!up()) { err = "glm link: down"; return false; }
    const auto t0 = std::chrono::steady_clock::now();
    Hdr h{};
    if (!recv_hdr(h, t, err)) return false;
    if (h.pos != pos) {
        fail(err, "token for position " + std::to_string((long long) h.pos) + ", expected " +
                      std::to_string((long long) pos) + " (out of step)");
        return false;
    }
    tok = h.val;
    s_recv_ms += ms_since(t0);
    ++s_toks;
    return true;
}

bool send_pair(Type t, int64_t pos, int32_t val, int32_t n) {
    if (!up()) return false;
    Hdr h{kMagic, (uint32_t) t, pos, n, val, 0};
    if (!send_msg(h, nullptr, 0)) {
        std::string err;
        fail(err, "message send failed");
        return false;
    }
    ++s_toks;
    return true;
}

bool recv_pair(Type t, int64_t pos, int32_t& val, int32_t& n, std::string& err) {
    if (!up()) { err = "glm link: down"; return false; }
    const auto t0 = std::chrono::steady_clock::now();
    Hdr h{};
    if (!recv_hdr(h, t, err)) return false;
    if (h.pos != pos) {
        fail(err, "message " + std::to_string(t) + " for position " + std::to_string((long long) h.pos) + ", expected " +
                      std::to_string((long long) pos) + " (out of step)");
        return false;
    }
    val = h.val;
    n = h.n;
    s_recv_ms += ms_since(t0);
    ++s_toks;
    return true;
}

void request_begin() {
    g_cancel = false;
    g_end_pos = -1;
    g_end_flags = 0;
}

bool cancelled() { return g_cancel; }

bool begin_exchange(int64_t reuse, int32_t mode, int64_t& peer_reuse, int32_t& peer_mode, std::string& err) {
    if (!up()) { err = "glm link: down"; return false; }
    Hdr h{kMagic, kBegin, reuse, 0, mode, 0}, p{};
    if (role() == kFront) {
        if (!send_msg(h, nullptr, 0)) { fail(err, "BEGIN send failed"); return false; }
        if (!recv_hdr(p, kBegin, err)) return false;
    } else {
        if (!recv_hdr(p, kBegin, err)) return false;
        if (!send_msg(h, nullptr, 0)) { fail(err, "BEGIN send failed"); return false; }
    }
    peer_reuse = p.pos;
    peer_mode = p.val;
    return true;
}

bool chunk_exchange(int64_t pos, int32_t n, int32_t mine, int32_t& peer, std::string& err) {
    if (!up()) { err = "glm link: down"; return false; }
    Hdr h{kMagic, kChunk, pos, n, mine, 0}, p{};
    if (!send_msg(h, nullptr, 0)) { fail(err, "CHUNK send failed"); return false; }
    if (!recv_hdr(p, kChunk, err)) return false;   // (the back: "cancelled" when the front's kEnd came instead)
    if (p.pos != pos || p.n != n) {
        // the sides cut the prompt into different calls: their checkpoint / slot / prefix-cut settings differ
        fail(err, "the other side reads " + std::to_string(p.n) + " prompt tokens at position " +
                      std::to_string((long long) p.pos) + ", this one " + std::to_string(n) + " at " +
                      std::to_string((long long) pos) + " - both sides need the same engine settings (checkpoints, "
                      "slots, MAYA_GLM_PREFIX_CUT)");
        return false;
    }
    peer = p.val;
    return true;
}

// read and drop messages until one of type t (its header in h); an ERR or a lost connection fails
static bool drain_until(uint32_t t, Hdr& h, int& dropped, std::string& err) {
    dropped = 0;
    std::vector<char> junk;
    // (the drain as a whole has the stall bound, and after a STOP its 30 s: a side that keeps sending something else
    // never lets a single read wait that long)
    const int64_t t0 = now_ms();
    for (;;) {
        const int64_t took = now_ms() - t0;
        if ((stall_s() > 0 && took >= (int64_t) stall_s() * 1000) ||
            (g_stop_ms.load() != 0 && took >= (int64_t) kStopGraceS * 1000)) {
            fail(err, "the request's end was not answered in time (the other side keeps sending something else)");
            return false;
        }
        if (!read_all(g_data, &h, sizeof h) || h.magic != kMagic) {
            fail(err, "the connection is lost");
            return false;
        }
        if (h.type == t) {
            if (h.bytes != 0) {   // (END and ACK carry none: it would be read as the next request's first header)
                fail(err, "message " + std::to_string(h.type) + " with a payload (a protocol error)");
                return false;
            }
            return true;
        }
        if (h.type == kErr) {
            std::string why;
            if (!read_err(h, why)) {
                fail(err, "an ERR of " + std::to_string((unsigned long long) h.bytes) + " bytes (over 4 KiB: a protocol error)");
                return false;
            }
            fail(err, "the other side failed: " + why);
            return false;
        }
        // (a dropped message's payload: a HOP's rows at most - the context's whole prompt chunk, fp32)
        if ((h.type != kHop && h.bytes != 0) || h.bytes > kMaxHopBytes) {
            fail(err, "message " + std::to_string(h.type) + " with a payload of " +
                          std::to_string((unsigned long long) h.bytes) + " bytes (a protocol error)");
            return false;
        }
        ++dropped;
        if (h.type == kHop && h.val == 2) {   // a packed hop: its blocks say their length
            for (uint64_t left = h.bytes; left > 0;) {
                uint32_t len = 0, k = 0;
                if (!read_all(g_data, &len, 4) || len < sizeof(PackHdr) - 4 || len > 4 * kBlk) {
                    fail(err, "the connection is lost");
                    return false;
                }
                junk.resize(len);
                if (!read_all(g_data, junk.data(), len)) {
                    fail(err, "the connection is lost");
                    return false;
                }
                std::memcpy(&k, junk.data(), 4);
                left -= std::min<uint64_t>(left, k == 0 ? left : k);
            }
            continue;
        }
        for (uint64_t left = h.bytes; left > 0;) {
            const size_t k = (size_t) std::min<uint64_t>(left, 1 << 20);
            junk.resize(k);
            if (!read_all(g_data, junk.data(), k)) {
                fail(err, "the connection is lost");
                return false;
            }
            left -= k;
        }
    }
}

bool end_front(int64_t pos, int32_t flags, bool& agree, std::string& err) {
    if (!up()) { err = "glm link: down"; return false; }
    Hdr h{kMagic, kEnd, pos, 0, flags, 0}, p{};
    if (!send_msg(h, nullptr, 0)) { fail(err, "END send failed"); return false; }
    int dropped = 0;
    if (!drain_until(kAck, p, dropped, err)) return false;
    agree = p.val != 0 && p.pos == pos;
    if (dropped > 0) std::fprintf(stderr, "glm link: %d stale message(s) from the back dropped at the request's end\n", dropped);
    return true;
}

bool end_back(int64_t pos, int32_t& flags, bool& agree, std::string& err) {
    if (!up()) { err = "glm link: down"; return false; }
    if (!g_cancel) {
        Hdr p{};
        int dropped = 0;
        if (!drain_until(kEnd, p, dropped, err)) return false;
        g_end_pos = p.pos;
        g_end_flags = p.val;
        if (dropped > 0) std::fprintf(stderr, "glm link: %d stale message(s) from the front dropped at the request's end\n", dropped);
    }
    flags = g_end_flags;
    agree = g_end_pos == pos;
    Hdr h{kMagic, kAck, pos, 0, agree ? 1 : 0, 0};
    g_cancel = false;
    if (!send_msg(h, nullptr, 0)) { fail(err, "ACK send failed"); return false; }
    return true;
}

void ctl_send_line(const std::string& line) {
    std::lock_guard<std::mutex> lk(g_ctl_mu);
    if (g_ctl < 0) return;
    std::string l = line;
    l.push_back('\n');
    if (!write_all(g_ctl, l.data(), l.size())) {
        std::string err;
        fail(err, "control line send failed");
    }
}

void ctl_close() {
    std::lock_guard<std::mutex> lk(g_ctl_mu);
    if (g_ctl >= 0) ::shutdown(g_ctl, SHUT_WR);
}

void report() {
    std::fprintf(stderr, "glm link: %s %llu hops (%.1f MB), %llu tokens, %.1f ms sending, %.1f ms receiving\n", role_name(),
                 (unsigned long long) s_hops, (double) s_hop_bytes / 1e6, (unsigned long long) s_toks, s_send_ms,
                 s_recv_ms);
}
#endif

}  // namespace maya::core::glmlink
