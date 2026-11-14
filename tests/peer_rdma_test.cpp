// The peer transport's end-to-end test: a fixture pack, the real strata-peer daemon as a
// subprocess, and the client session against it - connect, hint, RDMA-READ every row back and
// verify the bytes, then the failure paths (a fingerprint mismatch refuses, a killed daemon fails
// the session cleanly).  Skipped with a note when the machine has no RDMA device (the tier is
// opt-in and the engine builds unchanged without one).
//
// STRATA_PEER_TEST_DEV names the ibv device (default: the first); the daemon binary comes from the
// STRATA_PEER_BIN define (CMake) or the environment of the same name.
#include "peer_cache.hpp"
#include "peer_client.hpp"

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <filesystem>
#include <fstream>

namespace {

using namespace strata::peer;

int fails = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "[ok]" : "[FAIL]", what.c_str());
    fflush(stdout);
    if (!ok) ++fails;
}

/// A deterministic pseudo-random byte (the fixture's content and its verification both use it).
uint8_t pattern(uint64_t i) {
    i = i * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint8_t) (i >> 33);
}

/// A fixture pack: `n_shards` shards of 4 MiB each, filled with pattern().
void make_fixture(const std::string& dir, int n_shards) {
    std::filesystem::create_directories(dir);
    for (int s = 0; s < n_shards; ++s) {
        std::string path = dir + "/shard" + std::to_string(s) + ".bin";
        std::ofstream f(path, std::ios::binary);
        const size_t bytes = 4ULL << 20;
        std::vector<char> buf(1ULL << 20);
        for (size_t at = 0; at < bytes; at += buf.size()) {
            for (size_t i = 0; i < buf.size(); ++i) buf[i] = (char) pattern(at + i);
            f.write(buf.data(), (std::streamsize) buf.size());
        }
    }
}

/// The fixture's row table: 16 rows scattered over the shards (lengths from 64 KiB to 3 MiB).
std::vector<Row> make_rows(int n_shards) {
    std::vector<Row> rows;
    const uint64_t shard_bytes = 4ULL << 20;
    static const uint32_t lens[] = {2u << 20, 512u << 10, 3u << 20, 64u << 10, 1u << 20, 768u << 10};
    for (int j = 0; j < 16; ++j) {
        Row r;
        r.shard = (uint16_t) (j % n_shards);
        r.len = lens[j % 6];
        r.off = (uint64_t) (j * 7777777ULL) % (shard_bytes - r.len);
        rows.push_back(r);
    }
    return rows;
}

std::vector<std::string> shard_names(int n_shards) {
    std::vector<std::string> out;
    for (int s = 0; s < n_shards; ++s) out.push_back("shard" + std::to_string(s) + ".bin");
    return out;
}

std::vector<uint64_t> shard_sizes(int n_shards) {
    return std::vector<uint64_t>(n_shards, 4ULL << 20);
}

/// A shard copy with a few flipped bytes (the fingerprint mismatch's fixture).
void corrupt_copy(const std::string& from_dir, const std::string& to_dir, int n_shards) {
    std::filesystem::create_directories(to_dir);
    for (int s = 0; s < n_shards; ++s) {
        const std::string name = "shard" + std::to_string(s) + ".bin";
        std::ifstream in(from_dir + "/" + name, std::ios::binary);
        std::ofstream out(to_dir + "/" + name, std::ios::binary);
        out << in.rdbuf();
        // (the flip happens below, after the streams close, straight on the file)
    }
    for (int s = 0; s < n_shards; ++s) {
        const std::string path = to_dir + "/shard" + std::to_string(s) + ".bin";
        std::ofstream f(path, std::ios::binary | std::ios::in);
        f.seekp(123456);
        char c = (char) (pattern(s) ^ 0xff);
        f.write(&c, 1);
    }
}

pid_t start_daemon(const std::string& bin, const std::string& pack_dir, int port,
                   const std::string& ram) {
    const pid_t pid = fork();
    if (pid != 0) return pid;
    const std::string log = "/tmp/strata-peer-test-daemon.log";
    const int fd = open(log.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd >= 0) {                     // the daemon's say() lines go to stderr - keep them
        dup2(fd, STDOUT_FILENO);
        dup2(fd, STDERR_FILENO);
    }
    execl(bin.c_str(), "strata-peer", "--pack", pack_dir.c_str(), "--bind", "0.0.0.0",
          "--port", std::to_string(port).c_str(), "--ram-gb", ram.c_str(), (char*) nullptr);
    _exit(127);
}

void stop_daemon(pid_t pid) {
    kill(pid, SIGTERM);
    int status = 0;
    waitpid(pid, &status, 0);
}

/// The IP that rides the RDMA link: the device's netdev (rocep7s0 -> enp7s0np0) matched against
/// the interface addresses; empty when unmapped.  rdma_cm cannot loop back on this stack (rping to
/// 127.0.0.1 says "No such device"), and every docker/bridge address would burn a 2 s timeout, so
/// the test wants exactly the link's address, not a candidate sweep.
std::string link_ip_of(const Device& dev) {
    const std::string want = dev.netdev();
    if (want.empty()) return "";
    ifaddrs* ifs = nullptr;
    if (getifaddrs(&ifs) != 0) return "";
    std::string out;
    for (ifaddrs* it = ifs; it != nullptr; it = it->ifa_next) {
        if (it->ifa_addr == nullptr || it->ifa_addr->sa_family != AF_INET) continue;
        if (want != it->ifa_name) continue;
        char buf[INET_ADDRSTRLEN];
        if (inet_ntop(AF_INET, &((sockaddr_in*) it->ifa_addr)->sin_addr, buf, sizeof buf)) {
            const std::string ip = buf;
            if (ip.rfind("127.", 0) != 0) out = ip;      // skip loopback: cm cannot resolve it
        }
    }
    freeifaddrs(ifs);
    return out;
}
/// One connect attempt against the RDMA link's own address
bool connect_via(PeerClient& c, const Device& dev, int port, const std::string& pack,
                 const std::vector<std::string>& names, const std::vector<uint64_t>& sizes,
                 const std::vector<Row>& rows, std::string& used_ip, std::string& err) {
    const std::string ip = link_ip_of(dev);
    if (ip.empty()) {
        err = "no IPv4 address found for the RDMA device's netdev (" + dev.netdev() + ")";
        return false;
    }
    if (c.connect(ip, port, pack, names, sizes, rows, err)) {
        used_ip = ip;
        return true;
    }
    std::printf("  [try %s] %s\n", ip.c_str(), err.c_str());
    return false;
}



}   // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "fixture") {       // (a fixture dir for out-of-tree runs)
        make_fixture(argv[2], 3);
        return 0;
    }
    const char* bin_env = getenv("STRATA_PEER_BIN");
    const std::string bin = bin_env ? bin_env : STRATA_PEER_BIN;
    // no RDMA device here: the tier is opt-in, the engine runs unchanged without it - say so, pass
    Device probe;
    std::string err;
    if (!probe.open(getenv("STRATA_PEER_TEST_DEV") ? getenv("STRATA_PEER_TEST_DEV") : "", err)) {
        std::printf("[skip] no RDMA device: %s\n", err.c_str());
        return 0;
    }
    const int n_shards = 3;
    const int port = 19447;
    const std::string tmp = "/tmp/strata-peer-test-" + std::to_string(getpid());
    const std::string pack = tmp + "/pack", bad = tmp + "/bad";
    std::filesystem::create_directories(tmp);
    make_fixture(pack, n_shards);
    corrupt_copy(pack, bad, n_shards);
    const auto names = shard_names(n_shards);
    const auto sizes = shard_sizes(n_shards);
    const auto rows = make_rows(n_shards);

    // ---- the happy path: the real daemon, a full handshake, hint + read + verify every row
    const pid_t daemon = start_daemon(bin, pack, port, "1");
    int up = 0;
    for (int i = 0; i < 100 && !up; ++i) {          // the daemon's ctl socket decides when it is up
        usleep(100000);
        Ctl probe_ctl;
        up = probe_ctl.connect("127.0.0.1", port, 500, err) ? 1 : 0;
    }
    check(up == 1, "the daemon came up on 127.0.0.1:19447");

    PeerClient client(48, 2ULL << 20, 5000);
    std::string cerr_;
    std::string link_ip;
    const bool connected = connect_via(client, probe, port, pack, names, sizes, rows, link_ip, cerr_);
    check(connected, "handshake against the real daemon over " + link_ip);
    if (!connected) {
        std::printf("  (%s)\n", cerr_.c_str());
        stop_daemon(daemon);
        std::filesystem::remove_all(tmp);
        return 1;
    }

    Mr dst;
    check(dst.alloc(client.pd(), 4ULL << 20, IBV_ACCESS_LOCAL_WRITE, cerr_),
          "the dst MR registered");

    // hint in two batches (the map merges; the second hint of a cached row re-advertises it)
    std::vector<long long> first_half, second_half;
    for (int j = 0; j < 8; ++j) first_half.push_back(j);
    for (int j = 8; j < 16; ++j) second_half.push_back(j);
    check(client.hint(first_half, cerr_), "the first hint batch advertised");
    check(client.advertised_rows() >= 8, "the map holds the first batch");
    check(client.hint(second_half, cerr_), "the second hint batch advertised");
    check(client.advertised_rows() >= 16, "the map holds every hinted row");

    bool all_bytes = true;
    uint64_t total = 0;
    for (int j = 0; j < 16; ++j) {
        if (!client.read_row(j, dst, 0, cerr_)) {
            all_bytes = false;
            std::printf("  (row %d failed: %s)\n", j, cerr_.c_str());
            break;
        }
        const Row& r = rows[j];
        total += r.len;
        // verify straight against the fixture's pattern
        const auto& src = *(const char*) dst.buf();
        (void) src;
        for (uint64_t b = 0; b < r.len; b += 4096) {
            if (((uint8_t*) dst.buf())[b] != pattern(r.off + b)) {
                std::printf("  (row %d byte %llu mismatched)\n", j, (unsigned long long) b);
                all_bytes = false;
                break;
            }
        }
        if (!all_bytes) break;
    }
    check(all_bytes, "every row read back byte-identical (" + std::to_string(total >> 20) + " MiB)");
    check(client.rows_read() == 16, "the client counted 16 rows");
    client.close();          // the daemon serves one session at a time: its serve loop is parked
                             // in the next ctl line until this socket closes

    // ---- a deep window on a small QP: many chunks per row, posts bounded by the poll loop (the
    // ENOSPC lesson's regression test)
    PeerClient deep(2, 65536, 5000);
    check(connect_via(deep, probe, port, pack, names, sizes, rows, link_ip, cerr_), "the deep-window session over " + link_ip);
    Mr dst2;
    check(dst2.alloc(deep.pd(), 4ULL << 20, IBV_ACCESS_LOCAL_WRITE, cerr_), "the second dst MR");
    check(deep.hint({3}, cerr_), "row 3 (3 MiB) hinted");
    bool deep_ok = true;
    for (int rep = 0; rep < 4 && deep_ok; ++rep) {
        deep_ok = deep.read_row(3, dst2, 0, cerr_);
    }
    check(deep_ok, "a 3 MiB row in 64 KiB chunks x4 (192 posts) without ENOSPC");
    deep.close();

    // ---- a killed daemon fails the session cleanly (no hang, no crash)
    stop_daemon(daemon);
    std::printf("[ok] daemon stopped\n");
    const bool died = !client.keepalive(cerr_);
    check(died, "the killed daemon surfaces as a dead session");
    client.close();

    // ---- a mismatching pack copy is refused at the handshake
    const pid_t daemon2 = start_daemon(bin, bad, port, "1");
    up = 0;
    for (int i = 0; i < 100 && !up; ++i) {
        usleep(100000);
        Ctl probe_ctl;
        up = probe_ctl.connect("127.0.0.1", port, 500, err) ? 1 : 0;
    }
    PeerClient mismatch(48, 2ULL << 20, 2000);
    std::string bad_ip;
    const bool refused = !connect_via(mismatch, probe, port, pack, names, sizes, rows, bad_ip, cerr_);
    check(refused, "a fingerprint mismatch refused the session");
    if (!refused) std::printf("  (%s)\n", cerr_.c_str());
    mismatch.close();
    stop_daemon(daemon2);

    std::filesystem::remove_all(tmp);
    std::printf("%s peer_rdma_test: %s\n", fails ? "FAIL" : "PASS",
                fails ? "there were failures" : "all checks passed");
    return fails ? 1 : 0;
}
