// maya-hy4's conversation slots on a drive (src/hy4/hy4_slot_dir.hpp, SLOTS v2; no GPU, no model; a folder
// in the temp directory).  Written from the header's interface comment and docs/SETTINGS.md's MAYA_HY4_SLOT_DIR rows,
// not from the code:
//  - write_atomic: the bytes land whole, no .tmp left; go() false between two 64 MB pieces stops it - no file, no
//    .tmp, and a file already at the path keeps its old bytes (the rename never happened)
//  - read_file: the bytes back; a size not `expect`, or no file, refused with a reason
//  - remove_files(stem): <stem>.bin, .meta and their .tmp - whatever is there, nothing else
//  - lock_dir: a second holder (another process) does not get the folder (-1, after waiting wait_s); it does once the
//    first holder's process has ended
//  - Writer: on_thread runs first on the writer's own thread; a job writes <stem>.bin and <stem>.meta (the .meta
//    decodes to the job's meta with the .bin's size and checksum; Done says ok with that checksum); abort(id) of a
//    queued job and of the one in flight, abort_idle and a deadline past leave nothing on the drive; front: next;
//    a job into a folder that is not there fails and leaves nothing; the destructor stops what runs - every job's
//    files complete or absent, never a .tmp
// Failures are counted, not fatal (each with the contract line it encodes).  Linux / POSIX (fork for the lock).
#include "../src/hy4/hy4_slot_dir.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace {

int g_fail = 0, g_checks = 0;
#define CHECK(expr, clause)                                                                                          \
    do {                                                                                                             \
        ++g_checks;                                                                                                  \
        if (!(expr)) {                                                                                               \
            ++g_fail;                                                                                                \
            std::fprintf(stderr, "FAIL line %d: %s\n      contract: %s\n", __LINE__, #expr, clause);                \
        }                                                                                                            \
    } while (0)

namespace fs = std::filesystem;
using namespace hy4::slots;
using Bytes = std::vector<uint8_t>;

std::string g_dir;

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
bool exists(const std::string& p) { return fs::exists(p); }
Bytes slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return Bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
void spit(const std::string& p, const Bytes& b) {
    std::ofstream f(p, std::ios::binary);
    f.write((const char*) b.data(), (std::streamsize) b.size());
}
Bytes pattern(size_t n, uint32_t seed) {
    Bytes b(n);
    std::mt19937 r(seed);
    for (size_t i = 0; i < n; i += 4) {
        const uint32_t v = r();
        for (size_t k = 0; k < 4 && i + k < n; ++k) b[i + k] = (uint8_t) (v >> (8 * k));
    }
    return b;
}
// every file in the folder whose name starts with `stem`'s file name
std::vector<std::string> files_of(const std::string& stem) {
    std::vector<std::string> v;
    const std::string base = fs::path(stem).filename().string();
    for (const auto& e : fs::directory_iterator(fs::path(stem).parent_path())) {
        const std::string n = e.path().filename().string();
        if (n.rfind(base, 0) == 0) v.push_back(n);
    }
    return v;
}
bool any_tmp() {
    for (const auto& e : fs::directory_iterator(g_dir))
        if (e.path().extension() == ".tmp") return true;
    return false;
}
Meta meta_of(int64_t n, uint64_t bin_bytes) {
    Meta m{};
    m.identity = "hy4 slot-dir test";
    m.n = n;
    m.lo = 0;
    for (int64_t i = 0; i < n; ++i) m.tokens.push_back((int) (i * 3 + 1));
    m.bin_bytes = bin_bytes;
    return m;
}
Writer::Job job(uint64_t id, const std::string& name, size_t bytes, bool idle = false) {
    Writer::Job j{};
    j.id = id;
    j.stem = g_dir + "/" + name;
    j.data = std::make_shared<const Bytes>(pattern(bytes, (uint32_t) id));
    j.meta = meta_of(600, bytes);
    j.idle = idle;
    return j;
}
// a job's files: both complete (ok) or none at all
bool whole_or_none(const std::string& stem, bool* whole = nullptr) {
    const bool b = exists(stem + ".bin"), m = exists(stem + ".meta");
    const bool t = exists(stem + ".bin.tmp") || exists(stem + ".meta.tmp");
    if (whole) *whole = b && m;
    return !t && b == m;
}
bool none_of(const std::string& stem) { return files_of(stem).empty(); }

// ---- write_atomic / read_file / remove_files ----------------------------------------------------------------------
void test_files() {
    const char* wa = "write_atomic: p[0, n) to path through path.tmp - the file whole, no .tmp left";
    const char* ab = "write_atomic: go() false between two pieces stops it and removes the .tmp (no partial file)";
    const char* rf = "read_file: the file into out; its size must be expect";
    const char* rm = "remove_files(stem): <stem>.bin, .meta and their .tmp - whatever of them is there";
    std::string err;
    const std::string p1 = g_dir + "/slot-a.bin";
    const Bytes small = pattern(100000, 1);
    CHECK(write_atomic(p1, small.data(), small.size(), [] { return true; }, err), wa);
    CHECK(slurp(p1) == small && !exists(p1 + ".tmp"), wa);
    const Bytes zero;
    CHECK(write_atomic(g_dir + "/slot-z.bin", zero.data(), 0, [] { return true; }, err) &&
              exists(g_dir + "/slot-z.bin") && fs::file_size(g_dir + "/slot-z.bin") == 0, wa);
    // overwrite: the new bytes
    const Bytes small2 = pattern(5000, 2);
    CHECK(write_atomic(p1, small2.data(), small2.size(), [] { return true; }, err) && slurp(p1) == small2, wa);

    // an abort part way: 3 pieces of 64 MB, go() false at its second asking
    const Bytes big = pattern((size_t) 150 << 20, 3);
    const std::string p2 = g_dir + "/slot-b.bin";
    int asked = 0;
    err.clear();
    const bool r = write_atomic(p2, big.data(), big.size(), [&] { return ++asked < 2; }, err);
    CHECK(!r, ab);
    CHECK(asked >= 1, ab);   // (it was asked between pieces)
    CHECK(!exists(p2) && !exists(p2 + ".tmp"), ab);
    // an abort over an old file: the old bytes stay (the rename is the only change)
    asked = 0;
    CHECK(!write_atomic(p1, big.data(), big.size(), [&] { return ++asked < 2; }, err), ab);
    CHECK(slurp(p1) == small2 && !exists(p1 + ".tmp"), ab);
    // into a folder that is not there: refused with a reason, nothing made
    err.clear();
    CHECK(!write_atomic(g_dir + "/no-such-dir/slot-c.bin", small.data(), small.size(), [] { return true; }, err) &&
              !err.empty(), wa);

    Bytes out;
    err.clear();
    CHECK(read_file(p1, out, small2.size(), err) && out == small2, rf);
    out.assign(3, 7);
    err.clear();
    CHECK(!read_file(p1, out, small2.size() + 1, err) && !err.empty(), rf);
    err.clear();
    CHECK(!read_file(p1, out, small2.size() - 1, err) && !err.empty(), rf);
    err.clear();
    CHECK(!read_file(g_dir + "/slot-missing.bin", out, 10, err) && !err.empty(), rf);
    // a big one back whole
    const std::string p3 = g_dir + "/slot-big.bin";
    CHECK(write_atomic(p3, big.data(), big.size(), [] { return true; }, err), wa);
    err.clear();
    CHECK(read_file(p3, out, big.size(), err) && out == big, rf);

    const std::string stem = g_dir + "/slot-r-1";
    for (const char* suf : {".bin", ".meta", ".bin.tmp", ".meta.tmp"}) spit(stem + suf, small);
    spit(g_dir + "/slot-r-10.bin", small);   // (another stem that starts with this one's name)
    spit(g_dir + "/slot-r-1.txt", small);    // (not a slot's file)
    remove_files(stem);
    CHECK(!exists(stem + ".bin") && !exists(stem + ".meta") && !exists(stem + ".bin.tmp") &&
              !exists(stem + ".meta.tmp"), rm);
    CHECK(exists(g_dir + "/slot-r-10.bin") && exists(g_dir + "/slot-r-1.txt"), rm);
    remove_files(stem);   // nothing there: no harm
    spit(stem + ".meta", small);
    remove_files(stem);
    CHECK(!exists(stem + ".meta"), rm);
    for (const char* n : {"/slot-a.bin", "/slot-z.bin", "/slot-big.bin", "/slot-r-10.bin", "/slot-r-1.txt"})
        fs::remove(g_dir + n);
}

// ---- lock_dir -----------------------------------------------------------------------------------------------------
void test_lock() {
    const char* lk = "lock_dir: the folder this engine's alone - a second holder gets -1 (after waiting wait_s)";
    const char* ld = "lock_dir: it ends with the process - the next engine gets the folder";
    const std::string d = g_dir + "/lockdir";
    fs::create_directories(d);
    int to_parent[2], to_child[2];
    if (pipe(to_parent) || pipe(to_child)) {
        CHECK(false, "pipe");
        return;
    }
    const pid_t first = fork();
    if (first == 0) {   // the first engine: holds the folder until told to end
        const int fd = lock_dir(d, 0.0);
        const char c = fd >= 0 ? 'y' : 'n';
        if (write(to_parent[1], &c, 1) != 1) _exit(3);
        char x;
        if (read(to_child[0], &x, 1) != 1) _exit(4);
        _exit(fd >= 0 ? 0 : 2);
    }
    char got = 0;
    CHECK(read(to_parent[0], &got, 1) == 1 && got == 'y', lk);
    // a second engine (another process) while the first holds it
    auto try_in_child = [&](double wait_s, double* took) -> int {
        int res[2];
        if (pipe(res)) return -99;
        const double t0 = now_s();
        const pid_t p = fork();
        if (p == 0) {
            const int fd = lock_dir(d, wait_s);
            const char c = fd >= 0 ? 'y' : 'n';
            if (write(res[1], &c, 1) != 1) _exit(3);
            _exit(0);
        }
        char c = 0;
        const bool okr = read(res[0], &c, 1) == 1;
        int st = 0;
        waitpid(p, &st, 0);
        if (took) *took = now_s() - t0;
        close(res[0]);
        close(res[1]);
        return okr ? (c == 'y' ? 1 : 0) : -1;
    };
    double took = 0;
    CHECK(try_in_child(0.0, &took) == 0, lk);
    CHECK(try_in_child(0.6, &took) == 0, lk);
    CHECK(took >= 0.4, "lock_dir: waited for up to wait_s before giving up");
    std::printf("  (lock) a second holder with wait_s 0.6 gave up after %.2f s\n", took);
    // the first ends: the folder is free
    if (write(to_child[1], "x", 1) != 1) CHECK(false, "pipe write");
    int st = 0;
    waitpid(first, &st, 0);
    CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, lk);
    CHECK(try_in_child(0.0, nullptr) == 1, ld);
    // a holder that ends while the next one waits: got within its wait
    const pid_t second = fork();
    if (second == 0) {
        const int fd = lock_dir(d, 0.0);
        if (fd < 0) _exit(2);
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
        _exit(0);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    CHECK(try_in_child(5.0, &took) == 1, "lock_dir: waited for (a swap's last engine still writing at its end)");
    waitpid(second, &st, 0);
    // this process: the lock file is the only thing it leaves
    for (const auto& e : fs::directory_iterator(d))
        CHECK(file_kind(e.path().filename().string()) == kOther, "lock_dir: its lock file is not a slot's file");
    for (int fd : {to_parent[0], to_parent[1], to_child[0], to_child[1]}) close(fd);
}

// ---- Writer -------------------------------------------------------------------------------------------------------
std::vector<Writer::Done> drain(Writer& w, double secs) {
    std::vector<Writer::Done> all;
    w.wait_idle(now_s() + secs);
    for (const auto& d : w.poll()) all.push_back(d);
    return all;
}
const Writer::Done* done_of(const std::vector<Writer::Done>& v, uint64_t id) {
    for (const auto& d : v)
        if (d.id == id) return &d;
    return nullptr;
}
// wait (up to secs) for a file to appear
bool appears(const std::string& p, double secs) {
    const double end = now_s() + secs;
    while (now_s() < end) {
        if (exists(p)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

void test_writer() {
    const char* wj = "Writer: a job writes <stem>.bin and <stem>.meta; Done ok with the .bin's checksum";
    const char* wc = "Writer: abort / abort_idle / a deadline / the destructor - a job never leaves a file of its own "
                     "unless it ended ok";
    {
        Writer w;
        std::thread::id on;
        std::atomic<int> ran{0};
        w.start([&] {
            on = std::this_thread::get_id();
            ++ran;
        });
        Writer::Job j = job(1, "slot-w-1", (size_t) 3 << 20);
        const Bytes want = *j.data;
        const Meta jm = j.meta;
        w.submit(j, false);
        const auto done = drain(w, 30);
        CHECK(ran == 1 && on != std::this_thread::get_id(), "Writer: on_thread runs on the writer's own thread first");
        const Writer::Done* d = done_of(done, 1);
        CHECK(d && d->ok && !d->aborted && d->err.empty(), wj);
        CHECK(d && d->sum == checksum(want.data(), want.size()), wj);
        CHECK(d && d->bytes == want.size(), wj);
        CHECK(slurp(g_dir + "/slot-w-1.bin") == want, wj);
        const Bytes mb = slurp(g_dir + "/slot-w-1.meta");
        Meta m{};
        std::string why;
        CHECK(meta_decode(mb.data(), mb.size(), m, why), wj);
        {   // the written .meta's own bytes: "HY4META2" both ends, the crc of every byte before it (zlib's)
            bool laid = mb.size() >= 32 && std::string(mb.begin(), mb.begin() + 8) == "HY4META2" &&
                        std::string(mb.end() - 8, mb.end()) == "HY4META2";
            if (laid) {
                uint64_t crc = 0;
                for (int k = 7; k >= 0; --k) crc = (crc << 8) | mb[mb.size() - 16 + (size_t) k];
                laid = crc == (uint64_t) crc32(mb.data(), mb.size() - 16, 0);
            }
            CHECK(laid, "Writer: the .meta is HY4META2 with its CRC-32 before the closing magic");
        }
        CHECK(m.bin_bytes == want.size() && m.checksum == checksum(want.data(), want.size()) && m.n == jm.n &&
                  m.tokens == jm.tokens && m.identity == jm.identity, "Writer: the .meta names the .bin (size, checksum) "
                  "and keeps the job's identity and tokens");
        CHECK(!any_tmp(), wj);
        CHECK(w.poll().empty(), "Writer: poll hands a job over once");
        remove_files(g_dir + "/slot-w-1");
    }
    {   // abort: one queued, one in flight
        Writer w;
        w.start([] {});
        w.submit(job(10, "slot-x-10", (size_t) 256 << 20), false);
        w.submit(job(11, "slot-x-11", (size_t) 1 << 20), false);
        w.abort(11);   // queued behind 10: dropped
        const bool seen = appears(g_dir + "/slot-x-10.bin.tmp", 10);
        w.abort(10);   // in flight (or, if the drive outran us, done)
        const auto done = drain(w, 60);
        CHECK(none_of(g_dir + "/slot-x-11"), wc);
        const Writer::Done* d11 = done_of(done, 11);
        CHECK(!d11 || !d11->ok, wc);
        const Writer::Done* d10 = done_of(done, 10);
        bool whole = false;
        CHECK(whole_or_none(g_dir + "/slot-x-10", &whole), wc);
        if (seen) {
            CHECK(d10 && !d10->ok && d10->aborted, "Writer: abort(id) stops the job in flight (Done aborted)");
            CHECK(none_of(g_dir + "/slot-x-10"), wc);
        } else {
            std::printf("  (note) the 256 MB job's .tmp was not seen before abort - the in-flight abort not proven\n");
        }
        CHECK(!d10 || d10->ok == whole, wc);
        remove_files(g_dir + "/slot-x-10");
    }
    {   // abort_idle: the idle jobs go, the others are written
        Writer w;
        w.start([] {});
        w.submit(job(20, "slot-i-20", (size_t) 256 << 20, true), false);
        w.submit(job(21, "slot-i-21", (size_t) 1 << 20, true), false);
        w.submit(job(22, "slot-i-22", (size_t) 1 << 20, false), false);
        appears(g_dir + "/slot-i-20.bin.tmp", 10);
        w.abort_idle();
        const auto done = drain(w, 60);
        const Writer::Done* d22 = done_of(done, 22);
        CHECK(d22 && d22->ok && exists(g_dir + "/slot-i-22.bin") && exists(g_dir + "/slot-i-22.meta"),
              "Writer: abort_idle leaves the jobs not marked idle");
        CHECK(none_of(g_dir + "/slot-i-21"), wc);
        CHECK(whole_or_none(g_dir + "/slot-i-20"), wc);
        const Writer::Done* d20 = done_of(done, 20);
        if (d20 && !d20->ok) CHECK(none_of(g_dir + "/slot-i-20"), wc);
        CHECK(!d20 || d20->idle, "Done: idle as the job was");
        for (const char* s : {"/slot-i-20", "/slot-i-21", "/slot-i-22"}) remove_files(g_dir + s);
    }
    {   // a deadline already past: what runs is stopped, nothing left
        Writer w;
        w.start([] {});
        w.set_deadline(now_s() - 1.0);
        w.submit(job(30, "slot-d-30", (size_t) 200 << 20), false);
        const auto done = drain(w, 60);
        const Writer::Done* d = done_of(done, 30);
        CHECK(d && !d->ok, "Writer: set_deadline(t) stops whatever runs past t");
        CHECK(none_of(g_dir + "/slot-d-30"), wc);
        // deadline off again: a job writes
        w.set_deadline(0);
        w.submit(job(31, "slot-d-31", (size_t) 1 << 20), false);
        const auto done2 = drain(w, 30);
        const Writer::Done* d2 = done_of(done2, 31);
        CHECK(d2 && d2->ok, "Writer: set_deadline(0) - none");
        remove_files(g_dir + "/slot-d-31");
    }
    {   // front: next
        Writer w;
        w.start([] {});
        w.submit(job(40, "slot-f-40", (size_t) 128 << 20), false);
        appears(g_dir + "/slot-f-40.bin.tmp", 10);
        w.submit(job(41, "slot-f-41", (size_t) 1 << 20), false);
        w.submit(job(42, "slot-f-42", (size_t) 1 << 20), true);
        std::vector<uint64_t> order;
        const double end = now_s() + 60;
        while (order.size() < 3 && now_s() < end) {
            for (const auto& d : w.poll()) order.push_back(d.id);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        size_t i41 = 9, i42 = 9;
        for (size_t i = 0; i < order.size(); ++i) {
            if (order[i] == 41) i41 = i;
            if (order[i] == 42) i42 = i;
        }
        CHECK(order.size() == 3 && i42 < i41, "Writer: submit(job, front) - the next job written");
        for (const char* s : {"/slot-f-40", "/slot-f-41", "/slot-f-42"}) remove_files(g_dir + s);
    }
    {   // a job that cannot be written: failed, nothing left
        Writer w;
        w.start([] {});
        Writer::Job j = job(50, "x", (size_t) 1 << 20);
        j.stem = g_dir + "/no-such-dir/slot-50";
        w.submit(j, false);
        const auto done = drain(w, 30);
        const Writer::Done* d = done_of(done, 50);
        CHECK(d && !d->ok && !d->err.empty(), "Writer: a failed job is handed over (not ok, err says why)");
        CHECK(!any_tmp(), wc);
    }
    {   // the destructor stops every job and joins: each job's files whole or absent, no .tmp
        {
            Writer w;
            w.start([] {});
            w.submit(job(60, "slot-e-60", (size_t) 256 << 20), false);
            w.submit(job(61, "slot-e-61", (size_t) 64 << 20), false);
            appears(g_dir + "/slot-e-60.bin.tmp", 10);
        }
        CHECK(whole_or_none(g_dir + "/slot-e-60") && whole_or_none(g_dir + "/slot-e-61") && !any_tmp(), wc);
        for (const char* s : {"/slot-e-60", "/slot-e-61"}) remove_files(g_dir + s);
    }
    {   // wait_idle returns false at its time while a job still runs (not asserted: a fast drive may finish first)
        Writer w;
        w.start([] {});
        w.submit(job(70, "slot-q-70", (size_t) 256 << 20), false);
        const bool idle = w.wait_idle(now_s() + 0.001);
        std::printf("  (note) wait_idle(now + 1 ms) with a 256 MB job queued: %s\n", idle ? "true" : "false");
        w.abort(70);
        drain(w, 60);
        CHECK(whole_or_none(g_dir + "/slot-q-70"), wc);
        remove_files(g_dir + "/slot-q-70");
    }
}

}  // namespace

int main() {
    char tmpl[] = "/tmp/hy4-slot-dir-test-XXXXXX";
    const char* base = std::getenv("HY4_SLOT_DIR_TEST_TMP");   // (another drive: a folder there)
    std::string t = base ? std::string(base) + "/hy4-slot-dir-test-XXXXXX" : std::string(tmpl);
    std::vector<char> buf(t.begin(), t.end());
    buf.push_back(0);
    if (!mkdtemp(buf.data())) {
        std::perror("mkdtemp");
        return 2;
    }
    g_dir = buf.data();
    test_files();
    test_lock();
    test_writer();
    std::set<std::string> left;
    for (const auto& e : fs::directory_iterator(g_dir))
        if (e.path().filename() != "lockdir") left.insert(e.path().filename().string());
    for (const auto& n : left) std::fprintf(stderr, "      left in the folder: %s\n", n.c_str());
    CHECK(left.empty(), "every test's files removed (nothing stray left by the writer)");
    fs::remove_all(g_dir);
    if (g_fail) {
        std::fprintf(stderr, "hy4_slot_dir_test: %d of %d checks FAILED\n", g_fail, g_checks);
        return 1;
    }
    std::printf("hy4_slot_dir_test: ok (%d checks: write_atomic, read_file, remove_files, lock_dir, Writer)\n",
                g_checks);
    return 0;
}
