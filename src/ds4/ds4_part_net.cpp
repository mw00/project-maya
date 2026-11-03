// src/ds4/ds4_part_net.cpp - the pool's TCP link (ds4_part_net.hpp)
#include "ds4_part_net.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using sock_t = SOCKET;
#define PART_BAD INVALID_SOCKET
static int part_errno() { return WSAGetLastError(); }
static void part_close(sock_t s) { closesocket(s); }
#else
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
using sock_t = int;
#define PART_BAD (-1)
static int part_errno() { return errno; }
static void part_close(sock_t s) { ::close(s); }
#endif

#include <algorithm>
#include <cstdio>

namespace strata::ds4::part {

namespace {
bool net_init() {
#if defined(_WIN32)
    static const bool ok = [] {
        WSADATA w;
        return WSAStartup(MAKEWORD(2, 2), &w) == 0;
    }();
    return ok;
#else
    return true;
#endif
}
void tune(sock_t s) {
    int one = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char*) &one, sizeof one);
    int buf = 8 << 20;
    setsockopt(s, SOL_SOCKET, SO_SNDBUF, (const char*) &buf, sizeof buf);
    setsockopt(s, SOL_SOCKET, SO_RCVBUF, (const char*) &buf, sizeof buf);
    setsockopt(s, SOL_SOCKET, SO_KEEPALIVE, (const char*) &one, sizeof one);
}
std::string addr_str(const sockaddr_in& a) {
    char b[64];
    const unsigned char* q = (const unsigned char*) &a.sin_addr;
    std::snprintf(b, sizeof b, "%u.%u.%u.%u:%u", q[0], q[1], q[2], q[3], (unsigned) ntohs(a.sin_port));
    return b;
}
}  // namespace

Conn::~Conn() {
    close();
    if (lfd_ >= 0) part_close((sock_t) lfd_);
}

void Conn::close() {
#if defined(_WIN32)
    if (fd_ >= 0) shutdown((sock_t) fd_, SD_BOTH);
#else
    if (fd_ >= 0) shutdown((sock_t) fd_, SHUT_RDWR);
#endif
    if (fd_ >= 0) part_close((sock_t) fd_);
    fd_ = -1;
}

bool Conn::connect(const std::string& host, int port, int timeout_ms, std::string& err) {
    if (!net_init()) {
        err = "network start-up failed";
        return false;
    }
    close();
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || res == nullptr) {
        err = "cannot resolve " + host;
        return false;
    }
    sock_t s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == PART_BAD) {
        freeaddrinfo(res);
        err = "socket() failed";
        return false;
    }
    tune(s);
    // non-blocking connect with a timeout
#if defined(_WIN32)
    u_long nb = 1;
    ioctlsocket(s, FIONBIO, &nb);
#else
    fcntl(s, F_SETFL, fcntl(s, F_GETFL, 0) | O_NONBLOCK);
#endif
    int rc = ::connect(s, res->ai_addr, (int) res->ai_addrlen);
    freeaddrinfo(res);
    if (rc != 0) {
        fd_set w;
        FD_ZERO(&w);
        FD_SET(s, &w);
        timeval tv{timeout_ms / 1000, (timeout_ms % 1000) * 1000};
        rc = select((int) s + 1, nullptr, &w, nullptr, &tv);
        int soerr = 0;
        socklen_t sl = sizeof soerr;
        if (rc <= 0 || getsockopt(s, SOL_SOCKET, SO_ERROR, (char*) &soerr, &sl) != 0 || soerr != 0) {
            part_close(s);
            err = "cannot connect to " + host + ":" + std::to_string(port) + (rc == 0 ? " (timeout)" : "");
            return false;
        }
    }
#if defined(_WIN32)
    nb = 0;
    ioctlsocket(s, FIONBIO, &nb);
#else
    fcntl(s, F_SETFL, fcntl(s, F_GETFL, 0) & ~O_NONBLOCK);
#endif
    fd_ = (long long) s;
    peer_ = host + ":" + std::to_string(port);
    return true;
}

bool Conn::listen(int port, std::string& err) {
    if (!net_init()) {
        err = "network start-up failed";
        return false;
    }
    sock_t s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == PART_BAD) {
        err = "socket() failed";
        return false;
    }
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char*) &one, sizeof one);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    a.sin_port = htons((unsigned short) port);
    if (bind(s, (sockaddr*) &a, sizeof a) != 0 || ::listen(s, 4) != 0) {
        part_close(s);
        err = "cannot listen on port " + std::to_string(port);
        return false;
    }
    lfd_ = (long long) s;
    return true;
}

bool Conn::accept(std::string& err) {
    close();
    sockaddr_in a{};
    socklen_t al = sizeof a;
    sock_t s = ::accept((sock_t) lfd_, (sockaddr*) &a, &al);
    if (s == PART_BAD) {
        err = "accept() failed";
        return false;
    }
    tune(s);
    fd_ = (long long) s;
    peer_ = addr_str(a);
    return true;
}

void Conn::set_timeout(int ms) {
    if (fd_ < 0) return;
#if defined(_WIN32)
    DWORD t = (DWORD) ms;
    setsockopt((sock_t) fd_, SOL_SOCKET, SO_RCVTIMEO, (const char*) &t, sizeof t);
    setsockopt((sock_t) fd_, SOL_SOCKET, SO_SNDTIMEO, (const char*) &t, sizeof t);
#else
    timeval tv{ms / 1000, (ms % 1000) * 1000};
    setsockopt((sock_t) fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt((sock_t) fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
#endif
}

bool Conn::send_all(const void* p, size_t n) {
    const char* c = (const char*) p;
    while (n > 0) {
        const int k = (int) std::min<size_t>(n, (size_t) 1 << 24);
        const int r = (int) ::send((sock_t) fd_, c, k, 0);
        if (r <= 0) {
            err_ = "send failed (" + std::to_string(part_errno()) + ")";
            close();
            return false;
        }
        c += r;
        n -= (size_t) r;
    }
    return true;
}

bool Conn::recv_all(void* p, size_t n) {
    char* c = (char*) p;
    while (n > 0) {
        const int k = (int) std::min<size_t>(n, (size_t) 1 << 24);
        const int r = (int) ::recv((sock_t) fd_, c, k, 0);
        if (r <= 0) {
            err_ = r == 0 ? "the link closed" : "receive failed or timed out (" + std::to_string(part_errno()) + ")";
            close();
            return false;
        }
        c += r;
        n -= (size_t) r;
    }
    return true;
}

bool Conn::send_frame(uint32_t op, const void* a, size_t na, const void* b, size_t nb, const void* c, size_t nc) {
    if (fd_ < 0) {
        err_ = "no link";
        return false;
    }
    const uint64_t len = (uint64_t) na + nb + nc;
    // the header and a small first part in one send (no packet of its own for the header)
    std::vector<uint8_t> head(16 + (na <= 65536 ? na : 0));
    const uint32_t h[2] = {op, 0};
    std::memcpy(head.data(), h, 8);
    std::memcpy(head.data() + 8, &len, 8);
    if (na <= 65536 && na) std::memcpy(head.data() + 16, a, na);
    if (!send_all(head.data(), head.size())) return false;
    if (na > 65536 && !send_all(a, na)) return false;
    if (nb && !send_all(b, nb)) return false;
    if (nc && !send_all(c, nc)) return false;
    return true;
}

bool Conn::recv_frame(uint32_t& op, std::vector<uint8_t>& out) {
    if (fd_ < 0) {
        err_ = "no link";
        return false;
    }
    uint8_t h[16];
    if (!recv_all(h, 16)) return false;
    uint64_t len = 0;
    std::memcpy(&op, h, 4);
    std::memcpy(&len, h + 8, 8);
    if (len > ((uint64_t) 8 << 30)) {
        err_ = "a frame of " + std::to_string(len) + " bytes";
        close();
        return false;
    }
    out.resize((size_t) len);
    return len == 0 || recv_all(out.data(), (size_t) len);
}

}  // namespace strata::ds4::part
