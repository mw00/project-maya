#include "peer_ctl.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <time.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>

namespace strata::peer {

// --------------------------------------------------------------------------- tiny flat JSON

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        const unsigned char u = (unsigned char) c;
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (u < 0x20) {
            char b[8];
            std::snprintf(b, sizeof b, "\\u%04x", u);
            out += b;
        } else {
            out += c;
        }
    }
    return out;
}

Json& Json::add(const std::string& key, const std::string& value) {
    kv_.push_back({key, "\"" + json_escape(value) + "\""});
    return *this;
}

Json& Json::add(const std::string& key, const char* value) {
    return add(key, std::string(value));
}

Json& Json::add(const std::string& key, long long value) {
    kv_.push_back({key, std::to_string(value)});
    return *this;
}

Json& Json::add(const std::string& key, const std::vector<long long>& values) {
    std::string raw = "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) raw += ',';
        raw += std::to_string(values[i]);
    }
    return add_raw(key, raw + "]");
}

Json& Json::add_raw(const std::string& key, const std::string& raw) {
    kv_.push_back({key, raw});
    return *this;
}

std::string Json::str() const {
    std::string out = "{";
    for (size_t i = 0; i < kv_.size(); ++i) {
        if (i) out += ',';
        out += '"' + json_escape(kv_[i].first) + "\":" + kv_[i].second;
    }
    return out + "}";
}

// A flat object's scanner: after each key we expect ':' and a value - a "..." string (unescaped by
// the same rules json_escape applies) or a number/[array] taken verbatim.  Anything else ends the
// parse with ok_ false; the caller treats an unparsable line as a protocol error and reconnects.
JsonView::JsonView(const std::string& line) {
    size_t at = line.find('{');
    if (at == std::string::npos) return;
    ++at;
    while (at < line.size()) {
        while (at < line.size() && (std::isspace((unsigned char) line[at]) || line[at] == ',')) ++at;
        if (at < line.size() && line[at] == '}') {
            ok_ = true;
            return;
        }
        const size_t k0 = line.find('"', at);
        if (k0 == std::string::npos) return;
        const size_t k1 = line.find('"', k0 + 1);
        if (k1 == std::string::npos) return;
        const std::string key = line.substr(k0 + 1, k1 - k0 - 1);
        at = line.find(':', k1 + 1);
        if (at == std::string::npos) return;
        ++at;
        while (at < line.size() && std::isspace((unsigned char) line[at])) ++at;
        if (at >= line.size()) return;
        if (line[at] == '"') {
            std::string val;
            size_t p = at + 1;
            while (p < line.size() && line[p] != '"') {
                if (line[p] == '\\' && p + 1 < line.size()) {
                    const char c = line[p + 1];
                    if (c == 'u') {
                        // \u00XX only - what json_escape emits (a raw UTF-8 pass-through keeps the
                        // value's bytes, which is all this protocol needs)
                        val += (char) std::stoi(line.substr(p + 2, 2), nullptr, 16);
                        p += 6;
                        continue;
                    }
                    val += c == 'n' ? '\n' : c;
                    p += 2;
                    continue;
                }
                val += line[p++];
            }
            if (p >= line.size()) return;
            kv_[key] = val;
            at = p + 1;
        } else if (line[at] == '[' || std::isdigit((unsigned char) line[at]) || line[at] == '-') {
            if (line[at] == '[') {
                // a raw array value: run to the MATCHING ']' - find_first_of(",}") would stop
                // at the array's own commas and truncate it ("rows":[0,1] became "[0", the
                // scan died on "1]}"; one-element arrays slipped through, which is why only
                // the deep-window session's [3] hint ever parsed)
                size_t p = at;
                int depth = 0;
                while (p < line.size()) {
                    if (line[p] == '[') ++depth;
                    else if (line[p] == ']' && --depth == 0) break;
                    ++p;
                }
                if (p >= line.size()) return;
                kv_[key] = line.substr(at, p + 1 - at);
                at = p + 1;                  // past ']': the loop top skips the following , or }
            } else {
                const size_t end = line.find_first_of(",}", at);
                if (end == std::string::npos) return;
                kv_[key] = line.substr(at, end - at);
                at = end;                    // ON the ',' or '}': the loop top skips ',' and
                                             // stops on '}' (ok_); +1 here would skip past a
                                             // trailing '}' and leave ok_ false
            }
        } else {
            return;
        }
    }
}

std::string JsonView::str(const std::string& key, const std::string& def) const {
    const auto it = kv_.find(key);
    return it == kv_.end() ? def : it->second;
}

long long JsonView::num(const std::string& key, long long def) const {
    const auto it = kv_.find(key);
    if (it == kv_.end()) return def;
    try {
        return std::stoll(it->second);
    } catch (...) {
        return def;
    }
}

std::vector<long long> JsonView::nums(const std::string& key) const {
    std::vector<long long> out;
    const auto it = kv_.find(key);
    if (it == kv_.end()) return out;
    size_t p = it->second.find('[');
    const size_t end = it->second.rfind(']');
    if (p == std::string::npos || end == std::string::npos) return out;
    ++p;
    while (p < end) {
        try {
            out.push_back(std::stoll(it->second.substr(p)));
        } catch (...) {
            break;
        }
        const size_t comma = it->second.find(',', p);
        if (comma == std::string::npos || comma >= end) break;
        p = comma + 1;
    }
    return out;
}

// --------------------------------------------------------------------------- the channel

Ctl::~Ctl() {
    close();
}

bool Ctl::connect(const std::string& host, int port, int timeout_ms, std::string& err) {
    close();
    addrinfo hint = {}, *res = nullptr;
    hint.ai_family = AF_UNSPEC;
    hint.ai_socktype = SOCK_STREAM;
    const std::string service = std::to_string(port);
    const int rc = getaddrinfo(host.c_str(), service.c_str(), &hint, &res);
    if (rc != 0 || res == nullptr) {
        err = "getaddrinfo " + host + ":" + service + " failed: " + gai_strerror(rc);
        return false;
    }
    for (addrinfo* a = res; a != nullptr; a = a->ai_next) {
        fd_ = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (fd_ < 0) continue;
        // TCP_NODELAY: the protocol is a handful of small lines where a 40 ms Nagle delay per
        // round trip would dwarf everything else the handshake does
        const int one = 1;
        setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        if (::connect(fd_, a->ai_addr, a->ai_addrlen) == 0) break;
        ::close(fd_);
        fd_ = -1;
    }
    freeaddrinfo(res);
    if (fd_ < 0) {
        err = "connect " + host + ":" + service + " failed: " + std::strerror(errno);
        return false;
    }
    (void) timeout_ms;              // the caller's recv/send timeouts cover the slow-peer case
    return true;
}

bool Ctl::adopt(int fd, std::string& err) {
    close();
    fd_ = fd;
    const int one = 1;
    setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    (void) err;
    return true;
}

void Ctl::close() {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
    closed_ = false;
}

bool Ctl::write_all(const void* p, size_t bytes, std::string& err) {
    const char* c = (const char*) p;
    while (bytes > 0) {
        const ssize_t n = ::send(fd_, c, bytes, MSG_NOSIGNAL);
        if (n <= 0) {
            err = std::string("ctl send failed: ") + (n < 0 ? std::strerror(errno) : "peer closed");
            closed_ = true;
            return false;
        }
        c += n;
        bytes -= (size_t) n;
    }
    return true;
}

static std::string ip_of(int fd, bool local, std::string& err) {
    sockaddr_in sa = {};
    socklen_t len = sizeof sa;
    const int rc = local ? getsockname(fd, (sockaddr*) &sa, &len)
                         : getpeername(fd, (sockaddr*) &sa, &len);
    if (rc != 0 || sa.sin_family != AF_INET) {
        err = std::string("the socket's ") + (local ? "local" : "peer") + " address did not resolve";
        return "";
    }
    char buf[INET_ADDRSTRLEN];
    if (!inet_ntop(AF_INET, &sa.sin_addr, buf, sizeof buf)) {
        err = "the address did not render";
        return "";
    }
    return buf;
}

std::string Ctl::local_ip(std::string& err) const { return ip_of(fd_, true, err); }
std::string Ctl::peer_ip(std::string& err) const { return ip_of(fd_, false, err); }

bool Ctl::send(const std::string& json_line, std::string& err) {
    return write_all(json_line.data(), json_line.size(), err) &&
           write_all("\n", 1, err);        // the newline is the flush point: one frame, one line
}

std::string Ctl::recv(int timeout_ms, std::string& err) {
    std::string line;
    size_t scan = 0;
    timespec deadline = {};
    const bool forever = timeout_ms < 0;             // the daemon's serve loop has no deadline
    clock_gettime(CLOCK_MONOTONIC, &deadline);
    deadline.tv_sec += timeout_ms / 1000;
    deadline.tv_nsec += (timeout_ms % 1000) * 1000000L;
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_sec += deadline.tv_nsec / 1000000000L;
        deadline.tv_nsec %= 1000000000L;
    }
    while (true) {
        // bytes already buffered from an earlier over-read first
        const size_t nl = line_.find('\n', scan);
        if (nl != std::string::npos) {
            line = line_.substr(0, nl);
            line_.erase(0, nl + 1);
            return line;
        }
        scan = line_.size();
        timespec now = {};
        clock_gettime(CLOCK_MONOTONIC, &now);
        if (!forever && (now.tv_sec > deadline.tv_sec ||
                         (now.tv_sec == deadline.tv_sec && now.tv_nsec >= deadline.tv_nsec))) {
            err = "ctl recv timed out after " + std::to_string(timeout_ms) + " ms";
            return "";
        }
        int wait = forever ? -1
                 : (int) ((deadline.tv_sec - now.tv_sec) * 1000 +
                          (deadline.tv_nsec - now.tv_nsec) / 1000000);
        if (wait <= 0) wait = 1;
        pollfd p = {fd_, POLLIN, 0};
        if (poll(&p, 1, wait) <= 0) continue;
        char buf[65536];
        const ssize_t n = ::recv(fd_, buf, sizeof buf, MSG_DONTWAIT);
        if (n <= 0) {
            if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) continue;   // spurious wake
            err = std::string("ctl recv failed: ") + (n < 0 ? std::strerror(errno) : "peer closed");
            closed_ = true;
            return "";
        }
        line_.append(buf, (size_t) n);
    }
}

}   // namespace strata::peer
