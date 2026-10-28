// ds4_store_test: the RAM expert store (src/ds4/ds4_store.hpp, --ssd-cache) over files of fake experts, CPU only -
// the reads (unbuffered where the file system takes O_DIRECT, and buffered), where a matrix lands in its slot, which
// expert gives way (lru, lfu), the read queue (demand before guesses, a guess asked for, stale guesses dropped), the
// store giving RAM back below its floor (a fake free-RAM figure), preload (the hottest kept longest), the miss split's
// order (hits at once, each miss as its read lands, a failed read read in place), put() and acquire().
// Usage: ds4_store_test [dir] (the fake model's files are written there, ~1 MB, and removed; default: the current
// directory - a file system that enforces O_DIRECT's alignment, ext4 or xfs, tests the unbuffered reads for real).
// Build: g++ -std=c++17 -O2 -pthread -I src/ds4 tests/ds4_store_test.cpp
#include "ds4_store.hpp"

#if !defined(_WIN32)
#include <fcntl.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using strata::ds4::ExpertStore;
namespace os = strata::ds4::os;

namespace {

int failures = 0;
#define EXPECT(c)                                                               \
    do {                                                                        \
        if (!(c)) {                                                             \
            std::fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++failures;                                                         \
        }                                                                       \
    } while (0)

// the fake model: file f's byte p is byte_at(f, p); layers 0-1 in file 0 after a 777-byte header, layer 2 in file 1
// (its last down ends the file: the aligned read past the end comes back short), layer 3 points past file 1's end (its
// reads fail)
constexpr int kExp = 8;
constexpr uint64_t kBytes[3] = {13001, 13001, 9001};
uint8_t byte_at(int f, uint64_t p) {
    uint64_t x = (p + 1) * 0x9E3779B97F4A7C15ull ^ (uint64_t) (f + 7) * 0xC2B2AE3D27D4EB4Full;
    x ^= x >> 29;
    return (uint8_t) (x * 0xBF58476D1CE4E5B9ull >> 56);
}
struct Model {
    std::vector<std::string> paths;
    std::vector<ExpertStore::Geo> geo;
    ExpertStore::Files files;   // (direct)
    ExpertStore::Files plain;   // (buffered only)
    bool direct_open = false;
    ~Model() {
        for (os::fd_t f : files.buf) os::close_fd(f);
        for (os::fd_t f : files.direct) os::close_fd(f);
        for (const std::string& p : paths) std::remove(p.c_str());
    }
};
bool make_model(const std::string& dir, Model& m) {
    uint64_t size[2] = {777, 0};
    m.geo.resize(4);
    for (int l = 0; l < 3; ++l) {
        ExpertStore::Geo& g = m.geo[(size_t) l];
        g.n_exp = kExp;
        const int f = l < 2 ? 0 : 1;
        for (int r = 0; r < 3; ++r) {
            g.file[r] = f;
            g.off[r] = size[f];
            g.stride[r] = kBytes[r];
            g.bytes[r] = kBytes[r];
            size[f] += kExp * kBytes[r];
        }
    }
    ExpertStore::Geo& bad = m.geo[3];   // (past the end)
    bad = m.geo[2];
    for (int r = 0; r < 3; ++r) bad.off[r] = size[1] + 1000000 + (uint64_t) r * 100000;
    const std::string tag = std::to_string((long long) std::chrono::steady_clock::now().time_since_epoch().count());
    for (int f = 0; f < 2; ++f) {
        const std::string p = dir + "/ds4_store_test." + tag + "." + std::to_string(f) + ".bin";
        std::FILE* o = std::fopen(p.c_str(), "wb");
        if (o == nullptr) return false;
        std::vector<uint8_t> b((size_t) size[f]);
        for (uint64_t i = 0; i < size[f]; ++i) b[(size_t) i] = byte_at(f, i);
        const bool ok = std::fwrite(b.data(), 1, b.size(), o) == b.size();
        std::fclose(o);
        if (!ok) return false;
        m.paths.push_back(p);
        m.files.buf.push_back(os::open_par(p));
        m.files.direct.push_back(os::open_direct(p));
        m.files.paths.push_back(p);
        m.direct_open = m.direct_open || m.files.direct.back() != os::kBadFd;
    }
    m.plain.buf = m.files.buf;
    m.plain.direct.assign(m.files.buf.size(), os::kBadFd);
    m.plain.paths = m.files.paths;
    return m.files.buf[0] != os::kBadFd && m.files.buf[1] != os::kBadFd;
}
bool same(const Model& m, int layer, int e, int r, const uint8_t* p) {
    const ExpertStore::Geo& g = m.geo[(size_t) layer];
    const uint64_t off = g.off[r] + (uint64_t) e * g.stride[r];
    if (((uintptr_t) p) % 4096 != off % 4096) return false;   // (the page offset the file's mapping has it at)
    for (uint64_t i = 0; i < g.bytes[r]; ++i)
        if (p[i] != byte_at(g.file[r], off + i)) return false;
    return true;
}

// reads held back per expert until the test lets them go; every read noted
struct Gate {
    std::mutex mu;
    std::condition_variable cv;
    std::set<std::pair<int, int>> shut;
    std::vector<ExpertStore::ReadNote> notes;
    int misaligned = 0;
    void hook(const ExpertStore::ReadNote& n) {
        std::unique_lock<std::mutex> lk(mu);
        notes.push_back(n);
        if (((uintptr_t) n.dst) % 4096 != 0) ++misaligned;
        cv.wait(lk, [&] { return shut.count({n.layer, n.e}) == 0; });
    }
    void close(int l, int e) {
        std::lock_guard<std::mutex> lk(mu);
        shut.insert({l, e});
    }
    void open(int l, int e) {
        {
            std::lock_guard<std::mutex> lk(mu);
            shut.erase({l, e});
        }
        cv.notify_all();
    }
    size_t count() {
        std::lock_guard<std::mutex> lk(mu);
        return notes.size();
    }
    std::vector<ExpertStore::ReadNote> seen() {
        std::lock_guard<std::mutex> lk(mu);
        return notes;
    }
};
bool wait_for(const std::function<bool()>& f, int ms = 5000) {
    for (int i = 0; i < ms; ++i) {
        if (f()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return f();
}

struct Mk {
    size_t slots = 64, chunk = 4;
    ExpertStore::Policy policy = ExpertStore::kLru;
    int readers = 4;
    bool direct = true;
    Gate* gate = nullptr;
    std::function<uint64_t()> avail;
};
bool make_store(const Model& m, const Mk& k, ExpertStore& st) {
    ExpertStore::Cfg c;
    c.chunk_slots = k.chunk;
    c.policy = k.policy;
    c.readers = k.readers;
    c.ram_avail = k.avail;
    if (k.gate != nullptr) c.before_read = [g = k.gate](const ExpertStore::ReadNote& n) { g->hook(n); };
    // (the slot's size first: a store of one slot says it)
    ExpertStore probe;
    c.budget = (size_t) 1 << 30;
    ExpertStore::Cfg c1 = c;
    c1.readers = 1;
    c1.before_read = nullptr;
    std::string err;
    if (!probe.init(c1, m.geo, k.direct ? m.files : m.plain, err)) return false;
    c.budget = k.slots * probe.slot_bytes();
    return st.init(c, m.geo, k.direct ? m.files : m.plain, err);
}

// a window's ask of experts es of layer l: every one handed out, then released; returns the kinds begin() gave
std::vector<uint8_t> ask(ExpertStore& st, const Model& m, int l, const std::vector<int>& es, bool check = true) {
    ExpertStore::Hold h;
    st.begin(l, es.data(), (int) es.size(), h);
    const std::vector<uint8_t> kinds = h.st;
    std::vector<int> out;
    size_t got = 0;
    while (st.next(h, out, true))
        for (int i : out) {
            ++got;
            if (check && h.st[(size_t) i] == ExpertStore::kHit)
                for (int r = 0; r < 3; ++r) EXPECT(same(m, l, h.e[(size_t) i], r, st.mat(h, i, r)));
        }
    EXPECT(got == es.size());
    st.release(h);
    return kinds;
}

void test_auto_size() {
    const uint64_t G = 1ull << 30;
    EXPECT(strata::ds4::store_auto_bytes(64 * G, 12 * G, 100 * G, 4 * G) == 52 * G);
    EXPECT(strata::ds4::store_auto_bytes(64 * G, 12 * G, 30 * G, 4 * G) == 30 * G);
    EXPECT(strata::ds4::store_auto_bytes(14 * G, 12 * G, 100 * G, 4 * G) == 0);
    EXPECT(strata::ds4::store_auto_bytes(10 * G, 12 * G, 100 * G, 4 * G) == 0);
    EXPECT(strata::ds4::store_auto_bytes(~0ull, 12 * G, 100 * G, 4 * G) == 0);
}

void test_reads(const Model& m, bool direct) {
    Gate gate;   // (none shut: notes and the alignment of each read's destination)
    Mk k;
    k.direct = direct;
    k.gate = &gate;
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    EXPECT(st.slot_bytes() % 4096 == 0);
    std::vector<int> all;
    for (int e = 0; e < kExp; ++e) all.push_back(e);
    for (int l = 0; l < 3; ++l) {
        const std::vector<uint8_t> a = ask(st, m, l, all);
        EXPECT(std::count(a.begin(), a.end(), ExpertStore::kMiss) == kExp);
        const std::vector<uint8_t> b = ask(st, m, l, all);   // (all here now)
        EXPECT(std::count(b.begin(), b.end(), ExpertStore::kHit) == kExp);
    }
    EXPECT(st.counts().misses.load() == 3 * kExp);
    EXPECT(st.counts().hits.load() == 3 * kExp);
    EXPECT(st.counts().failed.load() == 0);
    EXPECT(gate.misaligned == 0);
    EXPECT(gate.count() == (size_t) (3 * 3 * kExp));
    if (!direct) EXPECT(st.counts().direct_bytes.load() == 0);
    std::fprintf(stderr, "  reads (%s): %lld bytes, %lld of them unbuffered\n", direct ? "direct handles" : "buffered only",
                 (long long) st.counts().bytes.load(), (long long) st.counts().direct_bytes.load());
}

// unbuffered handles that open but whose reads fail (a write-only descriptor stands in for a file system or a USB
// bridge that refuses O_DIRECT reads): each file read buffered from its first failure on, the bytes right
void test_direct_refused(const Model& m) {
#if !defined(_WIN32)
    Model bad;
    bad.geo = m.geo;
    bad.plain = m.plain;
    bad.plain.direct.clear();
    for (const std::string& p : m.paths) bad.plain.direct.push_back(os::fd_t(::open(p.c_str(), O_WRONLY)));
    Mk k;
    k.direct = false;   // (bad.plain: the refusing handles as the direct ones)
    ExpertStore st;
    EXPECT(make_store(bad, k, st));
    std::vector<int> all;
    for (int e = 0; e < kExp; ++e) all.push_back(e);
    for (int l = 0; l < 3; ++l) {
        const std::vector<uint8_t> a = ask(st, m, l, all);
        EXPECT(std::count(a.begin(), a.end(), ExpertStore::kMiss) == kExp);
    }
    EXPECT(st.counts().failed.load() == 0);
    EXPECT(st.counts().direct_bytes.load() == 0);
    EXPECT(st.counts().direct_off.load() == 2);   // (once a file)
    for (os::fd_t f : bad.plain.direct) os::close_fd(f);
    bad.paths.clear();   // (m's files: m removes them)
    bad.files = ExpertStore::Files{};
#else
    (void) m;
#endif
}

void test_lru(const Model& m) {
    Mk k;
    k.slots = 4;
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    for (int e : {0, 1, 2, 3}) ask(st, m, 0, {e});   // A B C D
    ask(st, m, 0, {0});                               // A again
    ask(st, m, 0, {4});                               // E: B gives way
    EXPECT(st.has(0, 0) && !st.has(0, 1) && st.has(0, 2) && st.has(0, 3) && st.has(0, 4));
    ask(st, m, 0, {5});   // F: C
    EXPECT(!st.has(0, 2) && st.has(0, 0) && st.has(0, 3));
    EXPECT(st.counts().evicted.load() == 2);
    EXPECT(st.held() == 4);
}

void test_lfu(const Model& m) {
    Mk k;
    k.slots = 4;
    k.policy = ExpertStore::kLfu;
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    for (int e : {0, 1, 2, 3, 0, 0, 2}) ask(st, m, 1, {e});   // heats A 3, B 1, C 2, D 1 (B the older of the ones)
    ask(st, m, 1, {4});                                     // E: B
    EXPECT(!st.has(1, 1) && st.has(1, 0) && st.has(1, 2) && st.has(1, 3) && st.has(1, 4));
    ask(st, m, 1, {5});   // F: D (heat 1, older than E's 1)
    EXPECT(!st.has(1, 3) && st.has(1, 4) && st.has(1, 0) && st.has(1, 2));
    ask(st, m, 1, {6});   // G: E or F, both 1: E the older
    EXPECT(!st.has(1, 4) && st.has(1, 5));
}

void test_pinned(const Model& m) {
    Mk k;
    k.slots = 4;
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    const std::vector<int> four = {0, 1, 2, 3};
    ExpertStore::Hold h;
    st.begin(2, four.data(), 4, h);
    std::vector<int> out;
    while (st.next(h, out, true)) {
    }
    const std::vector<uint8_t> k5 = ask(st, m, 2, {4});   // (every slot pinned: nothing to give up)
    EXPECT(k5.size() == 1 && k5[0] == ExpertStore::kNone);
    EXPECT(st.counts().no_slot.load() == 1);
    EXPECT(st.counts().evicted.load() == 0);
    st.release(h);
    const std::vector<uint8_t> k6 = ask(st, m, 2, {4});
    EXPECT(k6.size() == 1 && k6[0] == ExpertStore::kMiss);
}

void test_queue(const Model& m) {
    {   // demand before guesses
        Gate gate;
        Mk k;
        k.readers = 1;
        k.gate = &gate;
        ExpertStore st;
        EXPECT(make_store(m, k, st));
        gate.close(0, 1);
        EXPECT(st.guess(0, 1));
        EXPECT(wait_for([&] { return gate.count() == 1; }));   // (the reader holds its first read)
        EXPECT(st.guess(2, 2));
        EXPECT(st.guess(0, 3));
        EXPECT(!st.guess(0, 3));   // (on its way already)
        const int d = 4;
        ExpertStore::Hold h;
        st.begin(1, &d, 1, h);
        gate.open(0, 1);
        std::vector<int> out;
        while (st.next(h, out, true)) {
        }
        st.release(h);
        EXPECT(wait_for([&] { return gate.count() == 12; }));
        const std::vector<ExpertStore::ReadNote> n = gate.seen();
        EXPECT(n[0].layer == 0 && n[0].e == 1 && n[0].guess);
        for (int i = 1; i <= 3; ++i) EXPECT(n[(size_t) i].layer == 1 && n[(size_t) i].e == 4 && !n[(size_t) i].guess);
        for (int i = 4; i < 12; ++i) EXPECT(n[(size_t) i].guess);
        // (the guesses in their order: the first one's other two matrices, then 2/2, then 0/3)
        EXPECT(n[4].layer == 0 && n[4].e == 1 && n[5].layer == 0 && n[5].e == 1);
        EXPECT(n[6].layer == 2 && n[8].layer == 2 && n[9].layer == 0 && n[9].e == 3 && n[11].e == 3);
        EXPECT(st.counts().guesses.load() == 3);
        EXPECT(wait_for([&] { return st.has(0, 3); }));
    }
    {   // a guess asked for jumps the other guesses; stale guesses dropped
        Gate gate;
        Mk k;
        k.readers = 1;
        k.gate = &gate;
        ExpertStore st;
        EXPECT(make_store(m, k, st));
        gate.close(0, 5);
        EXPECT(st.guess(0, 5));
        EXPECT(wait_for([&] { return gate.count() == 1; }));
        EXPECT(st.guess(0, 6));   // (layer 0: stale once the token is at layer 1)
        EXPECT(st.guess(2, 6));
        EXPECT(st.guess(1, 2));
        EXPECT(st.guess(2, 7));
        const int d = 7;   // layer 2's guess asked for
        ExpertStore::Hold h;
        st.begin(2, &d, 1, h);
        EXPECT(h.st[0] == ExpertStore::kMiss);
        EXPECT(st.counts().guess_used.load() == 1);
        EXPECT(st.cancel_stale(1) == 2);   // (0/6 and 1/2; 0/5 is being read: it finishes)
        EXPECT(st.counts().cancelled.load() == 2);
        EXPECT(!st.has(0, 6) && !st.has(1, 2));
        gate.open(0, 5);
        std::vector<int> out;
        while (st.next(h, out, true)) {
        }
        EXPECT(h.st[0] == ExpertStore::kHit);
        st.release(h);
        EXPECT(wait_for([&] { return gate.count() == 9; }));   // 0/5, 2/7 (asked), 0/5's other two, 2/6
        const std::vector<ExpertStore::ReadNote> n = gate.seen();
        EXPECT(n[1].layer == 2 && n[1].e == 7 && !n[1].guess && n[3].e == 7);
        EXPECT(n[4].layer == 0 && n[4].e == 5 && n[5].e == 5);
        EXPECT(n[6].layer == 2 && n[6].e == 6 && n[6].guess);
        for (const auto& x : n) EXPECT(!(x.layer == 0 && x.e == 6) && !(x.layer == 1 && x.e == 2));
        EXPECT(wait_for([&] { return st.has(0, 5) && st.has(2, 6); }));
        EXPECT(st.held() == 3);
    }
}

void test_preload(const Model& m) {
    {
        Mk k;
        k.slots = 4;
        ExpertStore st;
        EXPECT(make_store(m, k, st));
        const std::vector<std::pair<int, int>> list = {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}};
        EXPECT(st.preload(list, 0) == 4);   // (the hottest four)
        EXPECT(st.has(0, 0) && st.has(0, 1) && st.has(0, 2) && st.has(0, 3) && !st.has(0, 4) && !st.has(0, 5));
        const std::vector<uint8_t> a = ask(st, m, 0, {0, 1, 2, 3});
        EXPECT(std::count(a.begin(), a.end(), ExpertStore::kHit) == 4);
    }
    {   // the list's last (the spare ones) give way first, the hottest last
        Mk k;
        k.slots = 4;
        ExpertStore st;
        EXPECT(make_store(m, k, st));
        EXPECT(st.preload({{1, 0}, {1, 1}, {2, 6}, {2, 7}}, 0) == 4);
        ask(st, m, 0, {0});
        EXPECT(!st.has(2, 7) && st.has(2, 6));
        ask(st, m, 0, {1});
        EXPECT(!st.has(2, 6) && st.has(1, 0) && st.has(1, 1));
        ask(st, m, 0, {2});
        EXPECT(!st.has(1, 1) && st.has(1, 0));
    }
    {   // leave: slots kept for the demand
        Mk k;
        k.slots = 4;
        ExpertStore st;
        EXPECT(make_store(m, k, st));
        EXPECT(st.preload({{0, 0}, {0, 1}, {0, 2}, {0, 3}}, 1) == 3);
        EXPECT(st.preload({{0, 0}, {0, 1}}, 0) == 0);   // (here already)
        EXPECT(st.held() == 3);
    }
}

void test_shrink(const Model& m) {
    const uint64_t G = 1ull << 30;
    std::atomic<uint64_t> avail{100 * G};   // (watch_every's thread reads it too)
    Mk k;
    k.slots = 8;
    k.chunk = 2;
    k.avail = [&] { return avail.load(); };
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    std::vector<std::pair<int, int>> list;
    for (int e = 0; e < kExp; ++e) list.push_back({0, e});
    EXPECT(st.preload(list, 0) == 8);
    EXPECT(st.slots_now() == 8);
    EXPECT(st.watch() == 0);   // (plenty free)
    const int s = st.acquire(0, 7);   // (the last chunk's: pinned, so nothing past it goes)
    EXPECT(s >= 0);
    if (s >= 0) EXPECT(same(m, 0, 7, 2, st.mat_of(s, 2)));
    avail = 3 * G;   // (below the 4 GB floor)
    size_t gave = st.watch();
    EXPECT(gave == 0);
    EXPECT(st.slots_now() == 8);
    if (s >= 0) st.unpin(s);
    gave = st.watch();
    EXPECT(gave == 4);
    EXPECT(st.slots_now() == 0 && st.held() == 0);
    EXPECT(!st.has(0, 0) && !st.has(0, 7));
    EXPECT(st.counts().shrunk.load() == 4);
    const std::vector<uint8_t> a = ask(st, m, 0, {1});   // (no chunk while RAM is short: read in place)
    EXPECT(a.size() == 1 && a[0] == ExpertStore::kNone);
    avail = 9 * G;   // (above the floor, but not 8 GB beside a chunk yet... 9 GB is)
    EXPECT(st.watch() == 0);
    const std::vector<uint8_t> b = ask(st, m, 0, {1});
    EXPECT(b.size() == 1 && b[0] == ExpertStore::kMiss);
    EXPECT(st.slots_now() == 2);
    avail = 5 * G;   // (above the floor: kept, but no new chunk)
    EXPECT(st.watch() == 0);
    ask(st, m, 0, {2});
    ask(st, m, 0, {3});   // (two slots: 1 gives way)
    EXPECT(st.slots_now() == 2 && !st.has(0, 1));
    st.watch_every(5);   // (on its own thread: the RAM short again, the chunk goes by itself)
    avail = 2 * G;
    EXPECT(wait_for([&] { return st.slots_now() == 0; }));
}

void test_split(const Model& m) {
    Gate gate;
    Mk k;
    k.gate = &gate;
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    EXPECT(st.preload({{1, 0}}, 0) == 1);
    gate.close(1, 1);
    gate.close(1, 2);
    gate.close(1, 3);
    const std::vector<int> es = {3, 0, 2, 1};   // (D, A, C, B: A held)
    ExpertStore::Hold h;
    st.begin(1, es.data(), 4, h);
    EXPECT(h.st[1] == ExpertStore::kHit && h.st[0] == ExpertStore::kMiss && h.st[3] == ExpertStore::kMiss);
    std::vector<int> out;
    EXPECT(st.next(h, out, false) && out.size() == 1 && out[0] == 1);   // the hit at once
    EXPECT(!st.next(h, out, false));                                    // (the misses not in)
    std::vector<int> order;
    for (int e : {2, 3, 1}) {   // C, then D, then B land
        gate.open(1, e);
        EXPECT(st.next(h, out, true));
        EXPECT(out.size() == 1);
        for (int i : out) {
            order.push_back(h.e[(size_t) i]);
            for (int r = 0; r < 3; ++r) EXPECT(same(m, 1, h.e[(size_t) i], r, st.mat(h, i, r)));
        }
    }
    EXPECT((order == std::vector<int>{2, 3, 1}));
    EXPECT(!st.next(h, out, true));
    st.release(h);
    {   // two that landed together come out together
        gate.close(1, 4);
        gate.close(1, 5);
        const std::vector<int> two = {4, 5};
        ExpertStore::Hold h2;
        st.begin(1, two.data(), 2, h2);
        gate.open(1, 4);
        gate.open(1, 5);
        EXPECT(wait_for([&] { return st.has(1, 4) && st.has(1, 5); }));
        EXPECT(st.next(h2, out, true) && out.size() == 2);
        st.release(h2);
    }
    {   // a read that fails: read in place (no slot, no bytes), the slot back
        const std::vector<int> one = {2};
        ExpertStore::Hold h3;
        const size_t held0 = st.held();
        st.begin(3, one.data(), 1, h3);
        EXPECT(h3.st[0] == ExpertStore::kMiss);
        EXPECT(st.next(h3, out, true) && out.size() == 1);
        EXPECT(h3.st[0] == ExpertStore::kNone && st.mat(h3, 0, 0) == nullptr);
        st.release(h3);
        EXPECT(st.counts().failed.load() == 1);
        EXPECT(st.held() == held0 && !st.has(3, 2));
    }
    EXPECT(st.counts().wait_ns.load() > 0);
}

void test_put(const Model& m) {
    Mk k;
    k.slots = 2;
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    std::vector<uint8_t> src[3];   // (layer 1's expert 3, as VRAM would give it back)
    const uint8_t* sp[3];
    const ExpertStore::Geo& g = m.geo[1];
    for (int r = 0; r < 3; ++r) {
        src[r].resize((size_t) g.bytes[r]);
        for (uint64_t i = 0; i < g.bytes[r]; ++i) src[r][(size_t) i] = byte_at(g.file[r], g.off[r] + 3 * g.stride[r] + i);
        sp[r] = src[r].data();
    }
    EXPECT(st.put(1, 3, sp));
    EXPECT(!st.put(1, 3, sp));   // (here already)
    const std::vector<uint8_t> a = ask(st, m, 1, {3});   // (same() checks the bytes and their page offset)
    EXPECT(a.size() == 1 && a[0] == ExpertStore::kHit);
    EXPECT(st.counts().put.load() == 1 && st.counts().misses.load() == 0);
    ask(st, m, 0, {0});
    ask(st, m, 0, {1});   // (full: the put one, the oldest, gives way)
    EXPECT(!st.has(1, 3));
    EXPECT(st.acquire(1, 3) < 0);
    const int s = st.acquire(0, 1);
    EXPECT(s >= 0);
    if (s >= 0) {
        for (int r = 0; r < 3; ++r) EXPECT(same(m, 0, 1, r, st.mat_of(s, r)));
        st.unpin(s);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : ".";
    Model m;
    if (!make_model(dir, m)) {
        std::fprintf(stderr, "ds4_store_test: cannot write the fake model's files in %s\n", dir.c_str());
        return 1;
    }
    std::fprintf(stderr, "ds4_store_test: files in %s (O_DIRECT %s)\n", dir.c_str(),
                 m.direct_open ? "opened" : "refused: buffered reads only");
    test_auto_size();
    test_reads(m, true);
    test_reads(m, false);
    test_direct_refused(m);
    test_lru(m);
    test_lfu(m);
    test_pinned(m);
    test_queue(m);
    test_preload(m);
    test_shrink(m);
    test_split(m);
    test_put(m);
    if (failures == 0) std::fprintf(stderr, "ds4_store_test: ALL OK\n");
    else std::fprintf(stderr, "ds4_store_test: %d FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
