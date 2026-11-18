// ds4_store_test: the RAM expert store (src/ds4/ds4_store.hpp, --ssd-cache) over files of fake experts, CPU only -
// the reads (unbuffered where the file system takes O_DIRECT, and buffered), where a matrix lands in its slot, which
// expert gives way (lru, lfu), the read queue (demand before guesses, a guess asked for, stale guesses dropped), the
// store giving RAM back below its floor (a fake free-RAM figure), preload (the hottest kept longest), the miss split's
// order (hits at once, each miss as its read lands, a failed read read in place), put() and acquire(); and layers of
// different sizes: each expert's slot its own size, the budget spent by those bytes, eviction within a size, RAM
// given back across sizes, two layers of one size whose matrices split it otherwise, the room kept (in the budget and
// in RAM) for a size that holds none.
// Usage: ds4_store_test [dir] (the fake model's files are written there, ~2 MB, and removed; default: the current
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
// reads fail).  Layers 4-6 in file 2, each matrix at a page start + a lead of its own: 4 smaller than layers 0-3, 5
// larger, 6 layer 5's three matrices in another order (a slot of the same pages, split otherwise)
constexpr int kExp = 8;
constexpr uint64_t kBytes[3] = {13001, 13001, 9001};
constexpr uint64_t kSized[3][3][2] = {   // layers 4-6: each matrix's bytes and lead
    {{5001, 3000}, {5001, 100}, {3001, 3900}},
    {{20001, 1000}, {20001, 2500}, {15001, 300}},
    {{15001, 300}, {20001, 1000}, {20001, 2500}}};
constexpr size_t kSlot0 = 14 * 4096;                      // (layers 0-3's slot: 5 + 5 + 4 pages)
constexpr size_t kSlot4 = 8 * 4096, kSlot5 = 17 * 4096;   // (layer 4's slot: 3 + 3 + 2 pages; layers 5 and 6's: 17)
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
    uint64_t size[3] = {777, 0, 0};
    m.geo.resize(7);
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
    for (int l = 4; l < 7; ++l) {
        ExpertStore::Geo& g = m.geo[(size_t) l];
        g.n_exp = kExp;
        for (int r = 0; r < 3; ++r) {
            g.file[r] = 2;
            g.off[r] = (size[2] + 4095) / 4096 * 4096 + kSized[l - 4][r][1];
            g.stride[r] = kSized[l - 4][r][0];
            g.bytes[r] = kSized[l - 4][r][0];
            size[2] = g.off[r] + kExp * g.bytes[r];
        }
    }
    const std::string tag = std::to_string((long long) std::chrono::steady_clock::now().time_since_epoch().count());
    for (int f = 0; f < 3; ++f) {
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
    return m.files.buf[0] != os::kBadFd && m.files.buf[1] != os::kBadFd && m.files.buf[2] != os::kBadFd;
}
bool same(const Model& m, int layer, int e, int r, const uint8_t* p) {
    const ExpertStore::Geo& g = m.geo[(size_t) layer];
    const uint64_t off = g.off[r] + (uint64_t) e * g.stride[r];
    if (((uintptr_t) p) % 4096 != off % 4096) return false;   // (the page offset the file's mapping has it at)
    for (uint64_t i = 0; i < g.bytes[r]; ++i)
        if (p[i] != byte_at(g.file[r], off + i)) return false;
    return true;
}
// an expert's bytes in memory (as the VRAM cache gives one back to the CPU lane): p for put()
struct Mem {
    std::vector<uint8_t> v[3];
    const uint8_t* p[3];
    Mem(const Model& m, int l, int e) {
        const ExpertStore::Geo& g = m.geo[(size_t) l];
        for (int r = 0; r < 3; ++r) {
            for (uint64_t i = 0; i < g.bytes[r]; ++i)
                v[r].push_back(byte_at(g.file[r], g.off[r] + (uint64_t) e * g.stride[r] + i));
            p[r] = v[r].data();
        }
    }
};

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
    size_t budget = 0;      // (bytes; 0: `slots` of the first covered layer's slot)
    int from = 0, to = 4;   // the layers the store covers, [from, to): 0-3 one size (the tests before the sizes)
    int hold = -1;          // (their n_hold)
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
    std::vector<ExpertStore::Geo> geo = m.geo;
    for (int l = 0; l < (int) geo.size(); ++l)
        if (l < k.from || l >= k.to) geo[(size_t) l].n_exp = 0;   // (a layer it does not cover)
        else geo[(size_t) l].n_hold = k.hold;
    c.budget = k.budget != 0 ? k.budget : k.slots * ExpertStore::slot_layout(geo[(size_t) k.from]);
    std::string err;
    return st.init(c, geo, k.direct ? m.files : m.plain, err);
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
    EXPECT(st.slot_bytes(0) % 4096 == 0);
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
    {   // leave: a slot's bytes kept for the demand
        Mk k;
        k.slots = 4;
        ExpertStore st;
        EXPECT(make_store(m, k, st));
        EXPECT(st.preload({{0, 0}, {0, 1}, {0, 2}, {0, 3}}, st.slot_bytes(0)) == 3);
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
    const Mem x(m, 1, 3);   // (layer 1's expert 3, as VRAM would give it back)
    EXPECT(st.put(1, 3, x.p));
    EXPECT(!st.put(1, 3, x.p));   // (here already)
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

// a slot is its layer's own size: each matrix's aligned read at any lead (its bytes and up to a page before it), whole
// pages - layers 0-3 5 + 5 + 4 pages, 4 3 + 3 + 2, 5 6 + 6 + 5, 6 5 + 6 + 6 - where the largest of each matrix over the
// layers made every slot 6 + 6 + 6
void test_sizes(const Model& m) {
    Mk k;
    k.to = 7;
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    for (int l = 0; l < 4; ++l) EXPECT(st.slot_bytes(l) == kSlot0);
    EXPECT(st.slot_bytes(4) == kSlot4);
    EXPECT(st.slot_bytes(5) == kSlot5 && st.slot_bytes(6) == kSlot5);
    EXPECT(st.slot_bytes(7) == 0);
}

// the budget buys experts at their own size: four of layer 4 and four of layer 5 fill it to the byte, where slots of
// layer 5's size would hold five
void test_capacity(const Model& m) {
    Mk k;
    k.from = 4;
    k.to = 7;
    k.budget = 4 * kSlot4 + 4 * kSlot5;
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    EXPECT(st.slots_max() == k.budget / kSlot4);   // (none made yet: what the budget buys at the smallest size)
    std::vector<std::pair<int, int>> list;
    for (int e = 0; e < 6; ++e) {   // (hottest first, the sizes in turn: two of each more than fit)
        list.push_back({4, e});
        list.push_back({5, e});
    }
    EXPECT(st.preload(list, 0) == 8);
    EXPECT(st.held() == 8 && st.held_bytes() == k.budget);
    EXPECT(st.slots_now() == 8);   // (the budget spent)
    for (int e = 0; e < 4; ++e) EXPECT(st.has(4, e) && st.has(5, e));
    EXPECT(!st.has(4, 4) && !st.has(5, 4));
    const std::vector<uint8_t> a = ask(st, m, 4, {0, 1, 2, 3});   // (same(): the bytes, at their page offset)
    EXPECT(std::count(a.begin(), a.end(), ExpertStore::kHit) == 4);
    const std::vector<uint8_t> b = ask(st, m, 5, {0, 1, 2, 3});
    EXPECT(std::count(b.begin(), b.end(), ExpertStore::kHit) == 4);
}

// a miss gives up an expert of its own size - the policy's choice among them (lru the least recently asked for, lfu
// the coldest) - and never one of another size, older or colder as it may be
void test_evict_sizes(const Model& m, ExpertStore::Policy policy) {
    Mk k;
    k.from = 4;
    k.to = 7;
    k.chunk = 2;
    k.policy = policy;
    k.budget = 2 * kSlot4 + 2 * kSlot5;
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    // (5/0 the oldest of all and as cold as any; 4/0 older than 4/1, and hotter)
    for (const auto& x : std::vector<std::pair<int, int>>{{5, 0}, {5, 1}, {4, 0}, {4, 0}, {4, 0}, {4, 1}})
        ask(st, m, x.first, {x.second});
    ask(st, m, 4, {2});   // a small one: a small one gives way - lru 4/0, lfu 4/1 - and 5/0 stays
    EXPECT(st.has(5, 0) && st.has(5, 1) && st.has(4, 2));
    if (policy == ExpertStore::kLru) EXPECT(!st.has(4, 0) && st.has(4, 1));
    else EXPECT(st.has(4, 0) && !st.has(4, 1));
    ask(st, m, 6, {0});   // layer 6, the large size: 5/0 gives way (lfu: as cold as 5/1, and older)
    EXPECT(!st.has(5, 0) && st.has(5, 1) && st.has(6, 0));
    EXPECT(st.counts().evicted.load() == 2 && st.counts().no_slot.load() == 0);
    EXPECT(st.held() == 4 && st.held_bytes() == k.budget);
    const std::vector<uint8_t> b = ask(st, m, 6, {0});   // (where layer 6's regions put it: same() on the hit)
    EXPECT(b.size() == 1 && b[0] == ExpertStore::kHit);
}

// RAM given back across sizes: the newest chunk first, whichever size it holds; the bytes it gave back buy the other
// size; the experts left stay where they are, right
void test_shrink_sizes(const Model& m) {
    const uint64_t G = 1ull << 30;
    std::atomic<uint64_t> avail{100 * G};
    Mk k;
    k.from = 4;
    k.to = 7;
    k.chunk = 2;
    k.budget = 4 * kSlot5 + 2 * kSlot4;
    k.avail = [&] { return avail.load(); };
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    EXPECT(st.preload({{5, 0}, {5, 1}, {4, 0}, {4, 1}, {5, 2}, {5, 3}}, 0) == 6);   // (chunks: 5's, 4's, 5's again)
    EXPECT(st.slots_now() == 6 && st.held_bytes() == k.budget);
    avail = 4 * G - 1;   // (just below the 4 GB floor: one chunk back is enough)
    EXPECT(st.watch() == 1);   // (the newest: 5/2 and 5/3)
    EXPECT(st.slots_now() == 4 && st.held_bytes() == 2 * kSlot5 + 2 * kSlot4);
    EXPECT(!st.has(5, 2) && !st.has(5, 3) && st.has(5, 0) && st.has(4, 1));
    const std::vector<uint8_t> a = ask(st, m, 5, {0, 1});
    EXPECT(std::count(a.begin(), a.end(), ExpertStore::kHit) == 2);
    avail = 100 * G;
    EXPECT(st.watch() == 0);
    const std::vector<uint8_t> b = ask(st, m, 4, {2});   // (a chunk of layer 4's size, from the bytes given back)
    EXPECT(b.size() == 1 && b[0] == ExpertStore::kMiss);
    EXPECT(st.slots_now() == 6 && st.held_bytes() == 2 * kSlot5 + 3 * kSlot4);
    avail = 3 * G;
    EXPECT(st.watch() == 3);
    EXPECT(st.slots_now() == 0 && st.held() == 0 && st.held_bytes() == 0);
}

// layers 5 and 6: one size, their matrices splitting it otherwise, side by side in one chunk - each matrix where its
// own layer's regions put it, at its file page offset, none running into the next slot; put() lands one the same way
void test_landing(const Model& m) {
    Mk k;
    k.from = 5;
    k.to = 7;
    k.chunk = 8;
    k.budget = 8 * kSlot5;
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    std::vector<std::pair<int, int>> list;
    for (int e = 0; e < 4; ++e) {
        list.push_back({5, e});
        list.push_back({6, e});
    }
    EXPECT(st.preload(list, 0) == 8);
    EXPECT(st.slots_now() == 8);   // (one chunk)
    const std::vector<uint8_t> a = ask(st, m, 5, {0, 1, 2, 3});   // (same(): every matrix's bytes and page offset)
    EXPECT(std::count(a.begin(), a.end(), ExpertStore::kHit) == 4);
    const std::vector<uint8_t> b = ask(st, m, 6, {0, 1, 2, 3});
    EXPECT(std::count(b.begin(), b.end(), ExpertStore::kHit) == 4);
    EXPECT(st.put(6, 5, Mem(m, 6, 5).p));   // (from memory: 5/0, the least recently asked for, gives way)
    EXPECT(!st.has(5, 0) && st.has(6, 5) && st.has(5, 1));
    const int s = st.acquire(6, 5);
    EXPECT(s >= 0);
    if (s >= 0) {
        for (int r = 0; r < 3; ++r) EXPECT(same(m, 6, 5, r, st.mat_of(s, r)));
        st.unpin(s);
    }
    const std::vector<uint8_t> c = ask(st, m, 6, {0, 1, 2, 3});   // (its neighbours untouched)
    EXPECT(std::count(c.begin(), c.end(), ExpertStore::kHit) == 4);
}

// a slot follows its matrices' sizes, not where they sit in the files: two layers of one quantization, one at page
// boundaries and one not, take one slot size (each region the aligned read's span at any lead)
void test_slot_follows_bytes() {
    ExpertStore::Geo a;
    a.n_exp = kExp;
    const uint64_t b[3] = {8192, 8192, 4096};   // (whole pages: at a page boundary every read starts there)
    for (int r = 0; r < 3; ++r) {
        a.file[r] = 0;
        a.bytes[r] = a.stride[r] = b[r];
        a.off[r] = (uint64_t) (r + 1) << 20;
    }
    ExpertStore::Geo u = a;
    for (int r = 0; r < 3; ++r) u.off[r] += 100;
    EXPECT(ExpertStore::slot_layout(a) == ExpertStore::slot_layout(u));
    EXPECT(ExpertStore::slot_layout(u) == 8 * 4096);   // (3 + 3 + 2 pages: a lead up to a page before each matrix)
}

// a size that asks late still gets slots: each size's first chunk stays kept for it while it holds none, so the first
// size's greedy fill leaves room for the others (here 2 slots each: what the budget gives every size)
void test_floor(const Model& m) {
    Mk k;
    k.to = 7;
    k.budget = 4 * kSlot0 + kSlot4 + kSlot5;
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    std::vector<std::pair<int, int>> list;
    for (int e = 0; e < kExp; ++e) list.push_back({0, e});
    EXPECT(st.preload(list, 0) == 2);   // (layer 0's size: 2 slots, the others' first ones kept)
    const std::vector<uint8_t> a = ask(st, m, 4, {0});
    const std::vector<uint8_t> b = ask(st, m, 5, {0, 1});
    EXPECT(a.size() == 1 && a[0] == ExpertStore::kMiss);
    EXPECT(b.size() == 2 && b[0] == ExpertStore::kMiss && b[1] == ExpertStore::kMiss);
    EXPECT(st.counts().no_slot.load() == 0 && st.held() == 5);
}

// preload passes over an expert whose size has no room left (the budget spent on another size's chunk) and still
// reads a later one whose size has a free slot
void test_preload_sizes(const Model& m) {
    Mk k;
    k.from = 4;
    k.to = 7;
    k.budget = 4 * kSlot4 + kSlot5;   // (layer 4's chunk of four, then one slot of layer 5's size: the budget spent)
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    EXPECT(st.preload({{4, 0}, {5, 0}, {5, 1}, {4, 1}}, 0) == 3);   // (5/1: no slot of its size; 4/1: a free one)
    EXPECT(st.has(4, 1) && !st.has(5, 1) && st.slots_now() == 5);
}

// a size takes no more slots than its layers have experts (chunks of 3 for layer 4's 8 make 3 + 3 + 2), or than they
// may hold (n_hold: the CPU lane's, the VRAM's never coming in) - so a budget of every holdable expert holds them all
void test_class_cap(const Model& m) {
    Mk k;
    k.from = 4;
    k.to = 5;
    k.chunk = 3;
    k.budget = (size_t) 1 << 20;   // (more than every expert's bytes)
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    std::vector<std::pair<int, int>> list;
    for (int e = 0; e < kExp; ++e) list.push_back({4, e});
    EXPECT(st.preload(list, 0) == 8);
    EXPECT(st.slots_now() == 8 && st.held_bytes() == 8 * kSlot4);
    Mk k2 = k;
    k2.hold = 5;
    ExpertStore s2;
    EXPECT(make_store(m, k2, s2));
    EXPECT(s2.preload(list, 0) == 5 && s2.slots_now() == 5);
    Mk k3;   // (3 of each layer's: layer 4's size 3, layers 5-6's 6; the budget every one of them, chunks of 4)
    k3.from = 4;
    k3.to = 7;
    k3.hold = 3;
    k3.budget = 3 * kSlot4 + 6 * kSlot5;
    ExpertStore s3;
    EXPECT(make_store(m, k3, s3));
    std::vector<std::pair<int, int>> all3;
    for (int l = 4; l < 7; ++l)
        for (int e = 0; e < 3; ++e) all3.push_back({l, e});
    EXPECT(s3.preload(all3, 0) == 9 && s3.held_bytes() == k3.budget);
}

// cancel_stale drops the queued guesses only: a demand read under way, a preloaded expert and a put one stay
void test_cancel_kinds(const Model& m) {
    Gate gate;
    Mk k;
    k.readers = 1;
    k.gate = &gate;
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    EXPECT(st.preload({{0, 0}}, 0) == 1);
    EXPECT(st.put(0, 1, Mem(m, 0, 1).p));
    gate.close(0, 2);
    const int d = 2;
    ExpertStore::Hold h;
    st.begin(0, &d, 1, h);   // (a demand read, held at its first matrix)
    EXPECT(wait_for([&] { return gate.count() == 4; }));
    EXPECT(st.guess(0, 3));   // (its three reads queued behind the demand's)
    EXPECT(st.cancel_stale(0) == 1);
    EXPECT(st.has(0, 0) && st.has(0, 1) && !st.has(0, 3));
    gate.open(0, 2);
    std::vector<int> out;
    while (st.next(h, out, true)) {
    }
    EXPECT(h.st[0] == ExpertStore::kHit);
    st.release(h);
    EXPECT(st.held() == 3);
}

// a guess whose read fails: its slot back on its own size's list, taken there with no new chunk
void test_guess_fail_class(const Model& m) {
    Mk k;
    k.to = 5;
    k.chunk = 1;
    k.budget = kSlot0 + kSlot4;   // (one slot of layer 0's size, one of layer 4's)
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    EXPECT(st.guess(3, 1));   // (layer 3's reads fail)
    EXPECT(wait_for([&] {
        st.watch();
        return st.counts().failed.load() == 1;
    }));
    EXPECT(st.held() == 0 && !st.has(3, 1) && st.slots_now() == 1);
    ask(st, m, 4, {0});
    const std::vector<uint8_t> a = ask(st, m, 0, {0});
    EXPECT(a.size() == 1 && a[0] == ExpertStore::kMiss);
    EXPECT(st.slots_now() == 2 && st.held() == 2 && st.has(4, 0) && st.has(0, 0));
}

// a landed guess never asked for gives way only to a miss of its own size (guess_wasted), never to another size's
void test_guess_evicted(const Model& m) {
    Mk k;
    k.to = 5;
    k.chunk = 1;
    k.budget = kSlot0 + kSlot4;
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    EXPECT(st.guess(4, 0));
    EXPECT(wait_for([&] { return st.has(4, 0); }));   // (landed, never asked: the oldest)
    ask(st, m, 0, {0});
    ask(st, m, 0, {1});   // (layer 0's size: 0/0 gives way, not the older guess of the other size)
    EXPECT(st.has(4, 0) && !st.has(0, 0) && st.has(0, 1) && st.counts().guess_wasted.load() == 0);
    ask(st, m, 4, {1});   // (layer 4's size: the unasked guess gives way)
    EXPECT(!st.has(4, 0) && st.has(4, 1) && st.counts().guess_wasted.load() == 1);
    EXPECT(st.counts().evicted.load() == 2 && st.held() == 2 && st.held_bytes() == k.budget);
}

// a guess in the newest chunk: kept while its read runs, given back with its chunk once landed
void test_guess_shrunk(const Model& m) {
    const uint64_t G = 1ull << 30;
    std::atomic<uint64_t> avail{100 * G};
    Gate gate;
    Mk k;
    k.to = 5;
    k.chunk = 1;
    k.readers = 1;
    k.gate = &gate;
    k.avail = [&] { return avail.load(); };
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    EXPECT(st.guess(0, 0));
    EXPECT(wait_for([&] { return st.has(0, 0); }));
    gate.close(4, 0);
    EXPECT(st.guess(4, 0));   // (the newest chunk: its read held)
    EXPECT(wait_for([&] { return gate.count() == 4; }));
    avail = 3 * G;
    EXPECT(st.watch() == 0 && st.slots_now() == 2);   // (a guess still reading keeps its chunk and the ones before it)
    gate.open(4, 0);
    EXPECT(wait_for([&] { return st.has(4, 0); }));
    EXPECT(st.watch() == 2);
    EXPECT(st.slots_now() == 0 && st.held() == 0 && st.held_bytes() == 0 && !st.has(0, 0) && !st.has(4, 0));
}

// preload of an expert whose read fails, in a store of two sizes: not counted, its slot back on its own size's list
// and taken by that size's next miss with no eviction or new chunk
void test_preload_fail(const Model& m) {
    Mk k;
    k.to = 5;
    k.chunk = 1;
    k.budget = kSlot0 + kSlot4;
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    EXPECT(st.preload({{3, 0}, {4, 0}}, 0) == 1);
    EXPECT(st.counts().failed.load() == 1 && !st.has(3, 0) && st.has(4, 0) && st.held() == 1);
    const std::vector<uint8_t> a = ask(st, m, 0, {0});
    EXPECT(a.size() == 1 && a[0] == ExpertStore::kMiss && st.slots_now() == 2);
    EXPECT(st.counts().evicted.load() == 0 && st.has(4, 0));
}

// an expert put in, in the newest chunk: given back with it (its size's slots and bytes down), read again later
void test_put_shrunk(const Model& m) {
    const uint64_t G = 1ull << 30;
    std::atomic<uint64_t> avail{100 * G};
    Mk k;
    k.to = 5;
    k.chunk = 1;
    k.avail = [&] { return avail.load(); };
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    EXPECT(st.preload({{0, 0}}, 0) == 1);
    EXPECT(st.put(4, 2, Mem(m, 4, 2).p));
    avail = 4 * G - 1;
    EXPECT(st.watch() == 1);
    EXPECT(!st.has(4, 2) && st.has(0, 0) && st.slots_now() == 1 && st.held_bytes() == st.slot_bytes(0));
    avail = 100 * G;
    EXPECT(st.watch() == 0);
    const std::vector<uint8_t> a = ask(st, m, 4, {2});
    EXPECT(a.size() == 1 && a[0] == ExpertStore::kMiss && st.slots_now() == 2);
}

// a size that asks late and may hold fewer than floor_: only its hold is kept for it, so a budget of every expert the
// sizes may hold still holds them all (here the large size asking first)
void test_floor_hold(const Model& m) {
    Mk k;
    k.from = 4;
    k.to = 7;
    k.hold = 3;
    k.budget = 3 * kSlot4 + 6 * kSlot5;
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    std::vector<std::pair<int, int>> list;
    for (int l : {6, 5, 4})
        for (int e = 0; e < 3; ++e) list.push_back({l, e});
    EXPECT(st.preload(list, 0) == 9 && st.held_bytes() == k.budget);
}

// a size whose last chunk was given back holds none again, so its first chunk's room is kept for it again: the other
// size, growing, leaves that room, and the first size gets it when it asks
void test_floor_regiven(const Model& m) {
    const uint64_t G = 1ull << 30;
    std::atomic<uint64_t> avail{100 * G};
    Mk k;
    k.from = 4;
    k.to = 6;
    k.chunk = 1;
    k.budget = 2 * kSlot4 + kSlot5;
    k.avail = [&] { return avail.load(); };
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    ask(st, m, 5, {0});   // (layer 5's chunk: one slot)
    avail = 4 * G - 1;
    EXPECT(st.watch() == 1 && st.slots_now() == 0);
    avail = 100 * G;
    EXPECT(st.watch() == 0);
    ask(st, m, 4, {0});
    ask(st, m, 4, {1});
    ask(st, m, 4, {2});   // (a third slot of 4's would take 5's room: 4/0 gives way)
    EXPECT(st.slots_now() == 2 && !st.has(4, 0) && st.has(4, 1) && st.has(4, 2));
    const std::vector<uint8_t> a = ask(st, m, 5, {1});
    EXPECT(a.size() == 1 && a[0] == ExpertStore::kMiss && st.slots_now() == 3 && st.held_bytes() == k.budget);
}

// the free RAM a chunk needs: a size that holds slots leaves room for the first chunk of one that holds none, and is
// refused alone (the other still grows); a size that holds none takes its first chunk with only `floor` free beside it
// - once watch() sees that much, and not with only what a give-back left
void test_ram_room(const Model& m) {
    const uint64_t G = 1ull << 30;
    std::atomic<uint64_t> avail{100 * G};
    Mk k;
    k.from = 4;
    k.to = 6;
    k.chunk = 2;
    k.budget = 8 * kSlot4 + 8 * kSlot5;
    k.avail = [&] { return avail.load(); };
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    ask(st, m, 4, {0});   // (layer 4's first chunk: 2 slots)
    avail = 8 * G + 2 * kSlot4 + 2 * kSlot5 - 1;   // (grow_room and a chunk of 4's, but not 5's first chunk beside them)
    ask(st, m, 4, {1});
    ask(st, m, 4, {2});   // (no chunk for 4: 4/0 gives way)
    EXPECT(st.slots_now() == 2 && !st.has(4, 0) && st.has(4, 2));
    const std::vector<uint8_t> a = ask(st, m, 5, {0});   // (5's first chunk: 4's refusal is its own)
    EXPECT(a.size() == 1 && a[0] == ExpertStore::kMiss && st.slots_now() == 4);
    avail = 4 * G - 1;
    EXPECT(st.watch() == 1 && !st.has(5, 0));   // (5's chunk, the newest, given back)
    const std::vector<uint8_t> b = ask(st, m, 5, {1});   // (no chunk while it gives back)
    EXPECT(b.size() == 1 && b[0] == ExpertStore::kNone && st.slots_now() == 2);
    avail = 4 * G - 1 + 2 * kSlot5;   // (what that give-back left free: below floor and the chunk)
    EXPECT(st.watch() == 0);
    const std::vector<uint8_t> c = ask(st, m, 5, {2});
    EXPECT(c.size() == 1 && c[0] == ExpertStore::kNone && st.slots_now() == 2);
    avail = 6 * G + 2 * kSlot5 - 1;   // (5's first chunk and just under halfway from floor to grow_room)
    EXPECT(st.watch() == 0);
    const std::vector<uint8_t> c2 = ask(st, m, 5, {4});
    EXPECT(c2.size() == 1 && c2[0] == ExpertStore::kNone && st.slots_now() == 2);
    avail = 6 * G + 2 * kSlot5;   // (halfway: room above floor, so the give-back does not take it again at once)
    EXPECT(st.watch() == 0);
    ask(st, m, 4, {3});   // (4 holds slots: still no chunk for it)
    EXPECT(st.slots_now() == 2);
    const std::vector<uint8_t> d = ask(st, m, 5, {3});
    EXPECT(d.size() == 1 && d[0] == ExpertStore::kMiss && st.slots_now() == 4);
}

// the preload's tail: the lowest count first, experts of equal count across the layers in turn
void test_by_count() {
    std::vector<std::pair<int, int>> out;
    ExpertStore::by_count({{0, {4, 0}}, {0, {4, 1}}, {0, {5, 0}}, {0, {5, 1}}, {1, {4, 2}}, {0, {6, 0}}}, out);
    const std::vector<std::pair<int, int>> want = {{4, 0}, {5, 0}, {6, 0}, {4, 1}, {5, 1}, {4, 2}};
    EXPECT(out == want);
}

// a read that fails and its slot refilled for the same expert (put) before the window looks again: the window reads
// that expert in place - the slot is no longer the one it pinned
void test_hold_refilled(const Model& m) {
    Mk k;
    k.slots = 1;
    k.chunk = 1;
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    const int d = 1;
    ExpertStore::Hold h;
    st.begin(3, &d, 1, h);   // (layer 3's reads fail)
    EXPECT(h.st[0] == ExpertStore::kMiss);
    EXPECT(wait_for([&] {
        st.watch();
        return st.counts().failed.load() == 1;
    }));
    EXPECT(st.put(3, 1, Mem(m, 2, 1).p));   // (layer 2's bytes, layer 3's sizes: into the one slot again)
    std::vector<int> out;
    EXPECT(st.next(h, out, true) && h.st[0] == ExpertStore::kNone && st.mat(h, 0, 0) == nullptr);
    st.release(h);
    const std::vector<uint8_t> a = ask(st, m, 0, {0});   // (3/1 unpinned: it gives way)
    EXPECT(a.size() == 1 && a[0] == ExpertStore::kMiss && !st.has(3, 1));
}

// a window released before it looks again, its failed slot refilled and pinned by another since: the release leaves
// the other's pin alone
void test_release_refilled(const Model& m) {
    Mk k;
    k.slots = 1;
    k.chunk = 1;
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    const int d = 1;
    ExpertStore::Hold h;
    st.begin(3, &d, 1, h);   // (layer 3's reads fail)
    EXPECT(wait_for([&] {
        st.watch();
        return st.counts().failed.load() == 1;
    }));
    EXPECT(st.put(3, 1, Mem(m, 2, 1).p));
    const int s = st.acquire(3, 1);   // (the other's pin on the refilled slot)
    EXPECT(s >= 0);
    st.release(h);
    const std::vector<uint8_t> a = ask(st, m, 0, {0});   // (still pinned: nothing to give up)
    EXPECT(a.size() == 1 && a[0] == ExpertStore::kNone && st.has(3, 1));
    if (s >= 0) st.unpin(s);
}

// a demand read that fails in a store of two sizes: read in place, its slot back on its own size's list and taken by
// that size's next miss with no new chunk, the other size untouched
void test_demand_fail_class(const Model& m) {
    Mk k;
    k.to = 5;
    k.chunk = 1;
    k.budget = kSlot0 + kSlot4;
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    ask(st, m, 4, {0});
    ask(st, m, 3, {2});   // (layer 3's reads fail)
    EXPECT(st.counts().failed.load() == 1 && st.held() == 1 && !st.has(3, 2));
    const std::vector<uint8_t> a = ask(st, m, 0, {0});
    EXPECT(a.size() == 1 && a[0] == ExpertStore::kMiss && st.slots_now() == 2 && st.has(4, 0));
    EXPECT(st.counts().evicted.load() == 0);
}

// a queued guess dropped in a store of two sizes: its slot back on its own size's list and taken there with no new
// chunk; the other size's guess, under way, lands
void test_guess_cancel_class(const Model& m) {
    Gate gate;
    Mk k;
    k.to = 5;
    k.chunk = 1;
    k.readers = 1;
    k.gate = &gate;
    k.budget = kSlot0 + kSlot4;
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    gate.close(4, 0);
    EXPECT(st.guess(4, 0));
    EXPECT(wait_for([&] { return gate.count() == 1; }));   // (its first read held: the one reader busy)
    EXPECT(st.guess(0, 1));
    EXPECT(st.cancel_stale(0) == 1 && !st.has(0, 1));
    gate.open(4, 0);
    EXPECT(wait_for([&] { return st.has(4, 0); }));
    const std::vector<uint8_t> a = ask(st, m, 0, {2});
    EXPECT(a.size() == 1 && a[0] == ExpertStore::kMiss && st.slots_now() == 2 && st.has(4, 0));
    EXPECT(st.counts().evicted.load() == 0);
}

// a guess a window asked for whose read fails: read in place, its slot back on its own size's list, not a wasted guess
void test_asked_guess_fail(const Model& m) {
    Gate gate;
    Mk k;
    k.to = 5;
    k.chunk = 1;
    k.readers = 1;
    k.gate = &gate;
    k.budget = 2 * kSlot0 + kSlot4;
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    gate.close(0, 0);
    EXPECT(st.guess(0, 0));
    EXPECT(wait_for([&] { return gate.count() == 1; }));
    EXPECT(st.guess(3, 1));   // (layer 3's reads fail)
    const int d = 1;
    ExpertStore::Hold h;
    st.begin(3, &d, 1, h);
    EXPECT(h.st[0] == ExpertStore::kMiss && st.counts().guess_used.load() == 1);
    gate.open(0, 0);
    std::vector<int> out;
    EXPECT(st.next(h, out, true) && h.st[0] == ExpertStore::kNone && st.mat(h, 0, 0) == nullptr);
    st.release(h);
    EXPECT(st.counts().failed.load() == 1 && !st.has(3, 1) && st.counts().guess_wasted.load() == 0);
    EXPECT(wait_for([&] { return st.has(0, 0); }));
    ask(st, m, 4, {0});
    const std::vector<uint8_t> a = ask(st, m, 1, {0});
    EXPECT(a.size() == 1 && a[0] == ExpertStore::kMiss && st.slots_now() == 3);
    EXPECT(st.counts().evicted.load() == 0);
}

// cancel_stale(l) right after begin(l) (DS4_STORE_AHEAD's order): a guess of layer l the window asked for keeps its
// reads and lands; one of l not asked for is dropped
void test_asked_guess_cancel(const Model& m) {
    Gate gate;
    Mk k;
    k.readers = 1;
    k.gate = &gate;
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    gate.close(0, 0);
    EXPECT(st.guess(0, 0));
    EXPECT(wait_for([&] { return gate.count() == 1; }));
    EXPECT(st.guess(1, 1) && st.guess(1, 2));
    const int d = 1;
    ExpertStore::Hold h;
    st.begin(1, &d, 1, h);
    EXPECT(st.cancel_stale(1) == 1);
    gate.open(0, 0);
    std::vector<int> out;
    while (st.next(h, out, true)) {
    }
    EXPECT(h.st[0] == ExpertStore::kHit && !st.has(1, 2) && st.counts().cancelled.load() == 1);
    st.release(h);
}

// a guess asked for, landed and released, gives way to a miss of its own size like any read expert (not a wasted
// guess), never to the other size's
void test_asked_guess_evicted(const Model& m) {
    Mk k;
    k.to = 5;
    k.chunk = 1;
    k.budget = kSlot0 + kSlot4;
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    EXPECT(st.guess(4, 0));
    ask(st, m, 4, {0});
    EXPECT(st.counts().guess_used.load() == 1);
    ask(st, m, 0, {0});
    ask(st, m, 0, {1});
    EXPECT(st.has(4, 0));
    ask(st, m, 4, {1});
    EXPECT(!st.has(4, 0) && st.has(4, 1) && st.counts().evicted.load() == 2 && st.counts().guess_wasted.load() == 0);
}

// a guess asked for in the newest chunk: kept while it reads and while the window holds it, given back with its chunk
// once released
void test_asked_guess_shrunk(const Model& m) {
    const uint64_t G = 1ull << 30;
    std::atomic<uint64_t> avail{100 * G};
    Gate gate;
    Mk k;
    k.to = 5;
    k.chunk = 1;
    k.readers = 1;
    k.gate = &gate;
    k.avail = [&] { return avail.load(); };
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    gate.close(4, 0);
    EXPECT(st.guess(4, 0));
    EXPECT(wait_for([&] { return gate.count() == 1; }));
    const int d = 0;
    ExpertStore::Hold h;
    st.begin(4, &d, 1, h);
    avail = 3 * G;
    EXPECT(st.watch() == 0);   // (reading)
    gate.open(4, 0);
    std::vector<int> out;
    while (st.next(h, out, true)) {
    }
    EXPECT(st.watch() == 0 && st.has(4, 0));   // (pinned)
    st.release(h);
    EXPECT(st.watch() == 1 && st.slots_now() == 0 && st.held_bytes() == 0);
}

// preloaded experts give way to a miss of their own size, the list's last first (the coldest), never to the other
// size's
void test_preload_evict_class(const Model& m) {
    Mk k;
    k.to = 5;
    k.chunk = 1;
    k.budget = 2 * kSlot0 + kSlot4;
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    EXPECT(st.preload({{0, 0}, {0, 1}, {4, 0}}, 0) == 3);
    ask(st, m, 0, {2});
    EXPECT(!st.has(0, 1) && st.has(0, 0) && st.has(4, 0) && st.counts().evicted.load() == 1);
}

// put() with no slot of its size to be had (its one slot pinned, the budget left kept for the other size): false,
// nothing taken from the other size or given up; once the slot is released it gives way
void test_put_no_slot(const Model& m) {
    Mk k;
    k.to = 5;
    k.chunk = 1;
    k.budget = kSlot0 + kSlot4;
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    const int d = 0;
    ExpertStore::Hold h;
    st.begin(4, &d, 1, h);
    std::vector<int> out;
    while (st.next(h, out, true)) {
    }
    const Mem x(m, 4, 1);
    EXPECT(!st.put(4, 1, x.p));
    EXPECT(!st.has(4, 1) && st.has(4, 0) && st.slots_now() == 1);
    EXPECT(st.counts().put.load() == 0 && st.counts().evicted.load() == 0);
    st.release(h);
    EXPECT(st.put(4, 1, x.p) && !st.has(4, 0));
}

// an expert put in gives way only to a miss of its own size, never to the other size's, older as it is
void test_put_evicted_class(const Model& m) {
    Mk k;
    k.to = 5;
    k.chunk = 1;
    k.budget = kSlot0 + kSlot4;
    ExpertStore st;
    EXPECT(make_store(m, k, st));
    EXPECT(st.put(4, 0, Mem(m, 4, 0).p));
    ask(st, m, 0, {0});
    ask(st, m, 0, {1});
    EXPECT(st.has(4, 0) && !st.has(0, 0));
    ask(st, m, 4, {1});
    EXPECT(!st.has(4, 0) && st.has(4, 1) && st.counts().evicted.load() == 2);
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
    test_sizes(m);
    test_capacity(m);
    test_evict_sizes(m, ExpertStore::kLru);
    test_evict_sizes(m, ExpertStore::kLfu);
    test_shrink_sizes(m);
    test_landing(m);
    test_slot_follows_bytes();
    test_floor(m);
    test_preload_sizes(m);
    test_class_cap(m);
    test_cancel_kinds(m);
    test_guess_fail_class(m);
    test_guess_evicted(m);
    test_guess_shrunk(m);
    test_preload_fail(m);
    test_put_shrunk(m);
    test_floor_hold(m);
    test_floor_regiven(m);
    test_ram_room(m);
    test_by_count();
    test_hold_refilled(m);
    test_release_refilled(m);
    test_demand_fail_class(m);
    test_guess_cancel_class(m);
    test_asked_guess_fail(m);
    test_asked_guess_cancel(m);
    test_asked_guess_evicted(m);
    test_asked_guess_shrunk(m);
    test_preload_evict_class(m);
    test_put_no_slot(m);
    test_put_evicted_class(m);
    if (failures == 0) std::fprintf(stderr, "ds4_store_test: ALL OK\n");
    else std::fprintf(stderr, "ds4_store_test: %d FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
