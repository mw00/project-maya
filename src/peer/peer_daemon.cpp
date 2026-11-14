// strata-peer: the daemon side of the peer expert tier.  It serves one initiator at a time (the
// engine's session): verifies the initiator's pack fingerprint against its own copy, takes the row
// table, brings the RDMA link up (its QP at RTS is all a READ responder needs - no per-row
// software), faults hinted rows from its own pack copy into a budget-capped pinned slab and
// advertises them.  The initiator pulls bytes with RDMA READ; the daemon never touches them.
//
// The daemon is deliberately engine-free (no GPU runtime, no strata_core): it runs beside an engine
// on the peer machine under that machine's own RAM rules, and its slab budget is cgroup-aware for
// exactly the reason the engine's RAM tier is (a container's /proc/meminfo is the host's).
//
// Usage: strata-peer --pack <dir> [--bind <ip>] [--port 9447] [--ram-gb <n>] [--arena-mb <n>]
//                     [--device <ibv device>] (-x: a budget of 0 probes a quarter of what is free)
#include "peer_cache.hpp"
#include "peer_ctl.hpp"
#include "peer_rdma.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>

namespace strata::peer {

namespace {

/// One session's serving state (a borrowed cache and a borrowed cm listener per accepted
/// connection - one initiator at a time, so the listener's child id serves this session alone).
struct Session {
    Ctl ctl;
    Device* dev = nullptr;          // the daemon-wide device: its PD carries both the QPs and
                                    // the cache's arena MRs (one PD, or the RDMA reads break)
    Qp qp;                          // this session's RC QP (armed from the client's qp-up)
    PeerCache* cache;
    std::vector<Row> rows;
    std::string pack_dir;
    uint64_t clock_stamp = 0;       // the session's LRU stamps (one initiator at a time)

    bool serve(std::string& err);
    bool handshake(std::string& err);
    int hello_window = 48;          // the client's window (identifies this session in the log)
    bool on_hint(const JsonView& hint, std::string& err);
};

bool Session::handshake(std::string& err) {
    const JsonView hello(ctl.recv(10000, err));
    if (!hello.ok() || hello.type() != "hello") {
        err = "no hello: " + (hello.ok() ? hello.type() : err);
        return false;
    }
    if (hello.num("proto") != 1) {
        err = "the client speaks protocol " + hello.str("proto") + ", this daemon speaks 1";
        return false;
    }
    const Json welcome = Json().add("type", "welcome").add("proto", 1)
                             .add("budget_gb", (long long) (cache->budget() >> 30));
    if (!ctl.send(welcome.str(), err)) return false;
    // the manifest: names + sizes + the client's own fingerprint of its copy - a mismatch refuses
    const JsonView man(ctl.recv(30000, err));
    if (!man.ok() || man.type() != "manifest") {
        err = "no manifest: " + (man.ok() ? man.type() : err);
        return false;
    }
    const long long n_shards = man.num("shards");
    if (n_shards <= 0 || n_shards > 4096) {
        err = "a manifest of " + man.str("shards") + " shards makes no sense";
        return false;
    }
    std::vector<std::string> names;
    std::vector<uint64_t> sizes;
    for (long long i = 0; i < n_shards; ++i) {
        names.push_back(man.str("shard" + std::to_string(i)));
        sizes.push_back((uint64_t) man.num("size" + std::to_string(i)));
        if (names.back().empty()) {
            err = "manifest shard " + std::to_string(i) + " has no name";
            return false;
        }
    }
    const std::string mine = pack_fingerprint(pack_dir, names, sizes, err);
    if (mine.empty()) return false;
    if (mine != man.str("fp")) {
        err = "pack fingerprint mismatch: the client's copy is not this pack (" + mine.substr(0, 32) +
              "... vs " + man.str("fp").substr(0, 32) + "...)";
        ctl.send(Json().add("type", "error").add("msg", err).str(), err);
        return false;
    }
    if (!cache->open_shards(names, sizes, pack_dir, err)) return false;
    if (!ctl.send(Json().add("type", "fingerprint").add("ok", 1).str(), err)) return false;
    // the row table (hex-framed, 16 bytes a row)
    const JsonView rows_line(ctl.recv(30000, err));
    if (!rows_line.ok() || rows_line.type() != "rows") {
        err = "no row table: " + (rows_line.ok() ? rows_line.type() : err);
        return false;
    }
    const std::string hex = rows_line.str("hex");
    if (hex.size() % 32 != 0 || hex.size() < 32) {
        err = "the row table's " + std::to_string(hex.size()) + " hex digits frame no whole rows";
        return false;
    }
    rows.clear();
    auto nib = [](char c) -> int {
        return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
    };
    for (size_t at = 0; at < hex.size(); at += 32) {
        uint16_t shard = 0, pad = 0;
        uint64_t off = 0;
        uint32_t len = 0;
        uint8_t* rec[4] = {(uint8_t*) &shard, (uint8_t*) &pad, (uint8_t*) &off, (uint8_t*) &len};
        const int widths[4] = {4, 4, 16, 8};             // nibbles per field
        int at_hex = (int) at;
        for (int f = 0; f < 4; ++f) {
            for (int b = 0; b < widths[f] / 2; ++b) {
                const int hi = nib(hex[at_hex++]), lo = nib(hex[at_hex++]);
                if (hi < 0 || lo < 0) {
                    err = "the row table's hex broke at record " + std::to_string(rows.size());
                    return false;
                }
                rec[f][b] = (uint8_t) ((hi << 4) | lo);
            }
        }
        rows.push_back(Row{shard, off, len});
    }
    if (!ctl.send(Json().add("type", "rows").add("count", (long long) rows.size()).str(), err))
        return false;
    say("row table: " + std::to_string(rows.size()) + " rows");
    // the QP comes up by hand (llama.cpp's ggml-rpc recipe): the client's qp-up carries its
    // {qpn, psn, gid}, we create and arm our side, and the ready answers with ours
    hello_window = (int) hello.num("window", 48);
    const int window = hello_window;
    const JsonView qpup(ctl.recv(30000, err));
    if (!qpup.ok() || qpup.type() != "qp-up") {
        err = "no qp-up: " + (qpup.ok() ? qpup.type() : err);
        return false;
    }
    if (!qp.create(*dev, window * 2 + 8, 8, err)) return false;
    if (!qp.arm(*dev, (uint32_t) qpup.num("qpn"), (uint32_t) qpup.num("psn"), qpup.str("gid"), err))
        return false;
    if (!ctl.send(Json().add("type", "ready").add("qpn", (long long) qp.qpn())
                      .add("psn", (long long) qp.psn()).add("gid", qp.gid()).str(), err)) {
        return false;
    }
    say("session up: budget " + std::to_string(cache->budget() >> 20) + " MiB, window " +
        std::to_string(window));
    return true;
}

bool Session::on_hint(const JsonView& hint, std::string& err) {
    const std::vector<long long> want = hint.nums("rows");
    std::vector<Cached> got;
    if (!cache->fill(want, rows, ++clock_stamp, got, err)) {
        ctl.send(Json().add("type", "error").add("msg", err).str(), err);
        return false;
    }
    // fill answers one entry per requested row, in the request's order: the parallel arrays line
    // straight up against it
    Json map;
    map.add("type", "map");
    std::vector<long long> idx, addr, len, rkey;
    for (size_t k = 0; k < got.size(); ++k) {
        idx.push_back(want[k]);
        addr.push_back((long long) got[k].addr);
        len.push_back(got[k].len);
        rkey.push_back(got[k].rkey);
    }
    map.add("i", idx).add("addr", addr).add("len", len).add("rkey", rkey);
    map.add("dropped", cache->dropped());
    if (!ctl.send(map.str(), err)) return false;
    say("hint: " + std::to_string(want.size()) + " rows, " +
        std::to_string(cache->cached_bytes() >> 20) + " MiB cached (" +
        std::to_string(cache->cached_rows()) + " rows, " + std::to_string(cache->arenas()) +
        " arenas)");
    return true;
}

bool Session::serve(std::string& err) {
    if (!handshake(err)) return false;
    while (true) {
        const JsonView line(ctl.recv(-1, err));      // no timeout: the read loop parks in poll
        if (!line.ok()) {
            err = "session idle: " + err;
            return false;
        }
        if (line.type() == "hint") {
            say("hint in: " + std::to_string(line.nums("rows").size()) + " rows (window " +
                std::to_string(hello_window) + ")");
            if (!on_hint(line, err)) return false;
        } else if (line.type() == "ping") {
            if (!ctl.send(Json().add("type", "pong").str(), err)) return false;
        } else if (line.type() == "bye") {
            say("the client said bye");
            return true;
        } else {
            say("an unknown line of type " + line.type() + " ignored");
        }
    }
}

uint64_t clock_stamp = 0;            // (unused here; the session carries its own - kept for symmetry)

}   // namespace

}   // namespace strata::peer

// ------------------------------------------------------------------------------------- main

int main(int argc, char** argv) {
    std::string pack_dir = "pack", bind = "0.0.0.0", device = "";
    int port = 9447;
    long long ram_gb = 0, arena_mb = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            return i + 1 < argc ? argv[++i] : "";
        };
        if (a == "--pack") pack_dir = next();
        else if (a == "--bind") bind = next();
        else if (a == "--port") port = std::atoi(next().c_str());
        else if (a == "--ram-gb") ram_gb = std::atoll(next().c_str());
        else if (a == "--arena-mb") arena_mb = std::atoll(next().c_str());
        else if (a == "--device") device = next();
        else if (a == "-h" || a == "--help") {
            std::printf("usage: strata-peer --pack <dir> [--bind <ip>] [--port 9447] [--ram-gb <n>] "
                        "[--arena-mb <n>] [--device <ibv device>]\n");
            return 0;
        } else {
            std::fprintf(stderr, "strata-peer: unknown flag %s\n", a.c_str());
            return 2;
        }
    }
    std::string err;
    strata::peer::Device dev;       // ONE device for the whole daemon: the cache's arena MRs and
    if (!dev.open(device, err)) {   // every session's QP must share its PD, or the reads break
        strata::peer::say("no IB device: " + err);
        return 1;
    }
    strata::peer::PeerCache cache;
    // the arena allocation happens inside a request: log it (the first one pins GiBs and can take
    // a second - a hung-looking gap here is the arena being born, not a lost reply)
    strata::peer::PeerCache::set_alloc_log([](uint64_t bytes) {
        strata::peer::say("allocating a " + std::to_string(bytes >> 20) + " MiB arena (pinning, be patient)");
    });
    if (!cache.open(&dev, (uint64_t) ram_gb << 30, (uint64_t) arena_mb << 20, err)) {
        strata::peer::say(err);
        return 1;
    }
    // the listener: one session at a time; a session's end keeps the slab for the next one
    addrinfo hint = {}, *res = nullptr;
    hint.ai_family = AF_INET;
    hint.ai_socktype = SOCK_STREAM;
    hint.ai_flags = AI_PASSIVE;
    if (getaddrinfo(bind.c_str(), std::to_string(port).c_str(), &hint, &res) != 0 || res == nullptr) {
        strata::peer::say(std::string("getaddrinfo bind failed: ") + gai_strerror(errno));
        return 1;
    }
    const int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    const int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (::bind(fd, res->ai_addr, res->ai_addrlen) != 0 || listen(fd, 4) != 0) {
        strata::peer::say(std::string("bind/listen failed: ") + strerror(errno));
        return 1;
    }
    freeaddrinfo(res);
    strata::peer::say("strata-peer listening on " + bind + ":" + std::to_string(port) +
                      ", pack " + pack_dir);
    while (true) {
        // the previous session's qp async errors (a NAK'd read, a transport failure) surface here:
        // the device's async queue drains between connections
        pollfd ap = { dev.ctx()->async_fd, POLLIN, 0 };
        while (poll(&ap, 1, 0) > 0) {
            ibv_async_event ae = {};
            if (ibv_get_async_event(dev.ctx(), &ae) != 0) break;
            strata::peer::say(std::string("device async event: ") +
                              ibv_event_type_str(ae.event_type));
            ibv_ack_async_event(&ae);
            ap = { dev.ctx()->async_fd, POLLIN, 0 };
        }
        const int conn = accept(fd, nullptr, nullptr);
        if (conn < 0) continue;
        strata::peer::Session s;
        s.cache = &cache;
        s.pack_dir = pack_dir;
        s.dev = &dev;
        if (!s.ctl.adopt(conn, err)) continue;
        if (s.serve(err)) {
            strata::peer::say("session closed cleanly, " +
                              std::to_string(cache.cached_bytes() >> 20) + " MiB kept warm");
        } else {
            strata::peer::say("session ended (window " + std::to_string(s.hello_window) + "): " + err);
        }
        s.qp.destroy();
        s.ctl.close();
    }
}
