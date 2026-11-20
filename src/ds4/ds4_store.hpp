// src/ds4/ds4_store.hpp - the RAM expert store (--ssd-cache; opt-in): the CPU lane's experts that are read in place (a
// GGUF's routed experts the GPUs do not hold, --mmap-experts) kept whole in RAM slots of the store's own instead of the
// OS page cache - which knows nothing of which experts are hot, reads a page at a time as the lane faults, and gives
// them up to any other I/O.
//  - a slot holds one expert: its three matrices, each in a region of whole 4096-byte pages, read from the model's
//    files past the page cache (Linux: O_DIRECT - Pack's dfds; elsewhere, or where the file system refuses it, through
//    the buffered handles), so each read lands its matrix at the region's start + (file offset % 4096): nothing is moved
//    after the read, and a matrix lies at the page offset the file's mapping has it at
//  - a slot is its layer's own size: each region the span of that matrix's aligned read at any page offset (its bytes
//    rounded up to whole pages, and the page its file offset may start into), so an expert takes its own bytes, not
//    the largest layer's (a GGUF quantizes its layers differently), and layers of one quantization take one size.  The
//    layers whose slots are of one size share a class, its slots packed at that size in chunks of its own; the budget
//    counts the bytes of every class's chunks
//  - which expert gives way: lru (the least recently asked for) or lfu (the lowest heat - its asks, decayed layer by
//    layer) among its own class's experts (a slot of another size cannot take it), never one a window is computing
//    (pinned) or one still being read.  A class keeps the share of the budget it took (at load: by the preload's
//    order), until a give-back moves it; while a class holds no slots, the room for its first chunk stays kept for it
//    (floor_ slots: chunk_slots, or what the budget gives every class, at most its hold), in the budget and in the RAM
//    the others grow into, and that chunk needs only halfway from `floor` to `grow_room` of RAM free beside it - so a
//    class that asks late, or whose chunks were given back, still gets slots, and not one given back again at once
//  - reads queue by kind: a demand read (the CPU lane waits for it) always before a guess (the lookahead's); a guess
//    asked for becomes a demand read, and a guess still queued once the token is past its layer is dropped unread
//  - the slots come in chunks (a class's) as they are first needed, while `grow_room` of RAM stays free beside them;
//    watch() gives chunks back from the end (the newest first, of any class) when the free RAM falls below `floor`
//  - put(): an expert whose bytes are in memory already (one the VRAM cache gives back to the CPU lane) copied into a
//    slot - no read
// Host code only (no GPU runtime): tests/ds4_store_test.cpp runs it over a file of fake experts.
#pragma once

#include "ds4_os.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace strata::ds4 {

/// --ssd-cache auto's size: the RAM free now less `room` (the system's and the owner's programs), at most `cap` (every
/// expert it could hold); 0 below `min` (too little to be worth a store)
inline uint64_t store_auto_bytes(uint64_t avail, uint64_t room, uint64_t cap, uint64_t min) {
    if (avail == ~uint64_t{0} || avail <= room) return 0;
    const uint64_t b = std::min(avail - room, cap);
    return b >= min ? b : 0;
}

class ExpertStore {
public:
    static constexpr uint64_t kAlign = os::kDirectAlign;
    /// a layer's experts in the files: expert e's matrix r (0 gate, 1 up, 2 down) is bytes[r] at off[r] + e stride[r]
    /// of file file[r] (n_exp 0: a layer the store does not cover); n_hold: how many of them it holds at once at most
    /// (-1: n_exp) - the CPU lane's, where the VRAM's never come in; a class has no more slots than its layers' sum
    struct Geo {
        int n_exp = 0, n_hold = -1;
        int file[3] = {-1, -1, -1};
        uint64_t off[3] = {0, 0, 0}, stride[3] = {0, 0, 0}, bytes[3] = {0, 0, 0};
    };
    /// the model's files: each one's handle for buffered reads from any thread (os::open_par), its unbuffered one
    /// (os::open_direct; kBadFd: none) and its path (for `unbuffered`)
    struct Files {
        std::vector<os::fd_t> buf, direct;
        std::vector<std::string> paths;
    };
    enum Policy { kLru = 0, kLfu = 1 };
    /// (tests) a read about to start
    struct ReadNote {
        int layer, e, r;
        bool guess;
        const uint8_t* dst;   // (where the aligned read starts: a page boundary)
        uint64_t off, len;
    };
    struct Cfg {
        size_t budget = 0;                     // bytes at most (the chunks of every class)
        size_t floor = (size_t) 4 << 30;       // free RAM below this: watch() gives chunks back
        size_t grow_room = (size_t) 8 << 30;   // a new chunk only while this much RAM stays free beside it (a
                                               // class's first: halfway from `floor` to this)
        size_t chunk_slots = 32;               // slots a chunk (the unit of growing and of giving back)
        Policy policy = kLru;
        int readers = 8;                       // reading threads
        float decay = 0.9f;                    // lfu: a layer's heats x decay every `decay_every` asks of the layer
        int decay_every = 256;
        // (Windows) each reader its own FILE_FLAG_NO_BUFFERING handles (os::open_unbuffered) - Windows' unbuffered
        // reads of a mapped file take a slow path (ds4_os.hpp), so off by default there: the buffered handles
        bool unbuffered = false;
        std::function<uint64_t()> ram_avail;   // the RAM this process can still get (null or ~0: unknown)
        std::function<void(const ReadNote&)> before_read;   // (tests) called by a reader before each read
    };
    /// counters (any thread): asks served from a slot, asks read (or not: no_slot), the lookahead's guesses (asked for
    /// later, given up unasked, dropped unread), slots given up for another expert, reads failed, experts put, bytes
    /// read (of them unbuffered), the CPU lane's waits for reads, chunks given back
    struct Counts {
        std::atomic<int64_t> hits{0}, misses{0}, no_slot{0}, guesses{0}, guess_used{0}, guess_wasted{0}, cancelled{0},
            evicted{0}, failed{0}, put{0}, bytes{0}, direct_bytes{0}, wait_ns{0}, shrunk{0};
        std::atomic<int64_t> direct_off{0};   // files whose unbuffered reads failed where buffered ones worked
    };
    /// a window's experts of one layer from begin() to release(): st[i] kHit (in its slot, pinned - mat() reads it),
    /// kMiss (its read under way, the slot pinned), kNone (no slot to be had, or its read failed: read it in place)
    static constexpr uint8_t kHit = 0, kMiss = 1, kNone = 2;
    struct Hold {
        int layer = -1;
        std::vector<int> e, slot;
        std::vector<uint8_t> st;
        std::vector<char> given;   // (next) handed out already
        std::vector<uint64_t> gen;   // (its slots' gen at begin(): the slot holds its expert while they match)
    };

    ExpertStore() = default;
    ExpertStore(const ExpertStore&) = delete;
    ExpertStore& operator=(const ExpertStore&) = delete;
    ~ExpertStore() {
        {
            std::lock_guard<std::mutex> lk(w_mu_);
            w_quit_ = true;
        }
        w_cv_.notify_all();
        if (w_th_.joinable()) w_th_.join();
        {
            std::lock_guard<std::mutex> lk(q_mu_);
            quit_ = true;
            demand_q_.clear();
            guess_q_.clear();
        }
        q_cv_.notify_all();
        for (std::thread& t : th_) t.join();   // (a read under way ends first: it writes into a chunk)
        for (const Chunk& c : chunks_) os::anon_free(c.p, c.bytes);
    }

    /// a slot's bytes for layer g's experts (0: a layer the store does not cover), and where matrix r's region starts
    /// in it (tr): each region the span of matrix r's aligned read at any lead - the matrix and up to a page before it,
    /// whole pages (so a layer's slot follows its matrices' sizes, not where they sit in the files)
    static size_t slot_layout(const Geo& g, size_t* tr = nullptr) {
        if (g.n_exp <= 0) return 0;
        size_t off = 0;
        for (int r = 0; r < 3; ++r) {
            if (tr != nullptr) tr[r] = off;
            off += (size_t) ((g.bytes[r] + 2 * kAlign - 1) / kAlign * kAlign);
        }
        return off;
    }

    bool init(const Cfg& c, const std::vector<Geo>& geo, const Files& files, std::string& err) {
        cfg_ = c;
        geo_ = geo;
        files_ = files;
        for (const Geo& g : geo_)
            for (int r = 0; r < 3 && g.n_exp > 0; ++r)
                if (g.file[r] < 0 || (size_t) g.file[r] >= files_.buf.size() || g.bytes[r] == 0) {
                    err = "the RAM store: a layer's expert matrix in no file";
                    return false;
                }
        lay_.assign(geo_.size(), Lay{});
        size_t hold = 0;
        for (size_t l = 0; l < geo_.size(); ++l) {
            const Geo& g = geo_[l];
            const size_t b = slot_layout(g, lay_[l].tr);
            if (b == 0) continue;
            size_t k = 0;
            while (k < cls_.size() && cls_[k].bytes != b) ++k;
            if (k == cls_.size()) cls_.emplace_back();
            const size_t h = (size_t) (g.n_hold >= 0 ? std::min(g.n_hold, g.n_exp) : g.n_exp);
            cls_[k].bytes = b;
            cls_[k].hold += h;
            lay_[l].cls = (int) k;
            hold += h;
        }
        if (cls_.empty()) {
            err = "the RAM store: no layer of experts to hold";
            return false;
        }
        size_t one = 0;   // (a slot of every class that holds any)
        for (const Class& k : cls_) {
            if (k.hold == 0) continue;
            least_ = least_ == 0 ? k.bytes : std::min(least_, k.bytes);
            one += k.bytes;
        }
        nmax_ = least_ == 0 ? 0 : std::min(hold, cfg_.budget / least_);   // (the slots it could ever have)
        if (nmax_ == 0) {
            err = "the RAM store: its size holds no expert";
            return false;
        }
        chunk_slots_ = std::max<size_t>(1, cfg_.chunk_slots);
        floor_ = std::min(chunk_slots_, cfg_.budget / one);
        no_direct_.reset(new std::atomic<bool>[files_.buf.size() + 1]);
        for (size_t i = 0; i <= files_.buf.size(); ++i) no_direct_[i].store(false, std::memory_order_relaxed);
        slots_.reset(new Slot[nmax_]);
        where_.resize(geo_.size());
        heat_.resize(geo_.size());
        asks_.assign(geo_.size(), 0);
        for (size_t l = 0; l < geo_.size(); ++l) {
            where_[l].assign((size_t) std::max(0, geo_[l].n_exp), -1);
            heat_[l].assign((size_t) std::max(0, geo_[l].n_exp), 0.0f);
        }
        for (int i = 0; i < std::max(1, cfg_.readers); ++i) th_.emplace_back([this] { reader(); });
        return true;
    }

    /// (the preload's tail) the experts of `in` (count, expert) into `out`, the lowest count first and those of equal
    /// count across the layers in turn: layer by layer (no profile: every count 0) the first layers' slot sizes would
    /// take the budget's tail, and each size keeps the share it takes at load
    static void by_count(const std::vector<std::pair<uint32_t, std::pair<int, int>>>& in,
                         std::vector<std::pair<int, int>>& out) {
        std::map<std::pair<int, uint32_t>, int> seen;   // ((layer, count): its experts so far)
        std::vector<std::pair<std::pair<uint32_t, int>, std::pair<int, int>>> v;   // ((count, rank in its layer), expert)
        for (const auto& x : in) v.push_back({{x.first, seen[{x.second.first, x.first}]++}, x.second});
        std::stable_sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        for (const auto& x : v) out.push_back(x.second);
    }
    /// a slot's bytes for `layer`'s experts (0: a layer it does not cover)
    size_t slot_bytes(int layer) const { return covers(layer) ? cls_[(size_t) lay_[(size_t) layer].cls].bytes : 0; }
    /// the slots it could ever have (the budget at the smallest size, at most every expert it may hold)
    size_t slots_max() const { return nmax_; }
    /// (the log) each class's slot bytes and the experts it holds
    std::vector<std::pair<size_t, size_t>> classes() const {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<std::pair<size_t, size_t>> v;
        for (const Class& k : cls_) v.push_back({k.bytes, k.slots - k.free.size()});
        return v;
    }
    size_t slots_now() const {   // (allocated)
        std::lock_guard<std::mutex> lk(mu_);
        return nalloc_;
    }
    size_t held() const {   // (slots holding an expert, or reading one)
        std::lock_guard<std::mutex> lk(mu_);
        size_t n = 0;
        for (const Class& k : cls_) n += k.slots - k.free.size();
        return n;
    }
    size_t held_bytes() const {   // (their bytes)
        std::lock_guard<std::mutex> lk(mu_);
        return held_bytes_locked();
    }
    const Counts& counts() const { return counts_; }
    bool covers(int layer) const { return layer >= 0 && layer < (int) geo_.size() && geo_[(size_t) layer].n_exp > 0; }
    /// where expert e's matrix r lies in its slot: the region's start + its file offset's lead
    uint64_t lead(int layer, int e, int r) const {
        const Geo& g = geo_[(size_t) layer];
        return (g.off[r] + (uint64_t) e * g.stride[r]) % kAlign;
    }

    /// The window's experts es[0..n) of `layer` (distinct): the ones held pinned, the others' reads started as demand
    /// reads (a guess under way for one becomes a demand read).  next() hands them out as they can be computed;
    /// release() ends the hold.
    void begin(int layer, const int* es, int n, Hold& h) {
        h.layer = layer;
        h.e.assign(es, es + n);
        h.slot.assign((size_t) n, -1);
        h.st.assign((size_t) n, kNone);
        h.given.assign((size_t) n, 0);
        h.gen.assign((size_t) n, 0);
        if (!covers(layer)) return;
        std::lock_guard<std::mutex> lk(mu_);
        for (int i = 0; i < n; ++i) {
            const int e = es[i];
            if (e < 0 || e >= geo_[(size_t) layer].n_exp) continue;
            ask(layer, e);
            int s = where_[(size_t) layer][(size_t) e];
            if (s >= 0) {
                settle(s);   // (a failed read lets its slot go)
                s = where_[(size_t) layer][(size_t) e];
            }
            if (s >= 0) {
                Slot& sp = slots_[(size_t) s];
                if (sp.guess) {
                    counts_.guess_used.fetch_add(1, std::memory_order_relaxed);
                    sp.guess = false;
                    if (sp.state == kLoading) promote(s);
                } else if (sp.state == kReady) {
                    counts_.hits.fetch_add(1, std::memory_order_relaxed);
                }
                ++sp.pin;
                unlink(s);
                link_back(s);
                h.slot[(size_t) i] = s;
                h.gen[(size_t) i] = sp.gen;
                h.st[(size_t) i] = sp.state == kReady ? kHit : kMiss;
                continue;
            }
            counts_.misses.fetch_add(1, std::memory_order_relaxed);
            s = take_slot(layer, true);
            if (s < 0) {
                counts_.no_slot.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            fill(s, layer, e, false, 1, false);
            h.slot[(size_t) i] = s;
            h.gen[(size_t) i] = slots_[(size_t) s].gen;
            h.st[(size_t) i] = kMiss;
        }
    }
    /// The hold's experts not handed out yet that can be computed now, into `out`: the held ones and the ones without
    /// a slot at once, a miss once its read has landed (failed: kNone, read it in place).  wait: until there is at least
    /// one.  False: none left.
    bool next(Hold& h, std::vector<int>& out, bool wait) {
        out.clear();
        std::chrono::steady_clock::time_point t0{};
        bool waited = false;
        for (;;) {
            uint64_t seen;
            {
                std::lock_guard<std::mutex> lk(done_mu_);   // (before the look: a read landing after it wakes the wait)
                seen = done_;
            }
            bool pending = false;
            {
                std::lock_guard<std::mutex> lk(mu_);
                for (size_t i = 0; i < h.st.size(); ++i) {
                    if (h.given[i]) continue;
                    if (h.st[i] == kMiss) {
                        const int s = h.slot[i];
                        Slot& sp = slots_[(size_t) s];
                        if (sp.left.load(std::memory_order_acquire) > 0) {
                            pending = true;
                            continue;
                        }
                        settle(s);
                        if (sp.state == kReady && sp.gen == h.gen[i]) {
                            h.st[i] = kHit;
                        } else {   // (failed: its slot went - taken since by another, maybe for the same expert)
                            h.st[i] = kNone;
                            h.slot[i] = -1;
                        }
                    }
                    h.given[i] = 1;
                    out.push_back((int) i);
                }
            }
            if (!out.empty() || !pending || !wait) break;
            if (!waited) {
                waited = true;
                t0 = std::chrono::steady_clock::now();
            }
            std::unique_lock<std::mutex> lk(done_mu_);
            done_cv_.wait_for(lk, std::chrono::milliseconds(2), [&] { return done_ != seen; });
        }
        if (waited)
            counts_.wait_ns.fetch_add(
                (int64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(),
                std::memory_order_relaxed);
        return !out.empty();
    }
    /// expert i's matrix r in its slot (kHit), else null (read it in place)
    const uint8_t* mat(const Hold& h, int i, int r) const {
        if (h.st[(size_t) i] != kHit) return nullptr;
        return slots_[(size_t) h.slot[(size_t) i]].base + lay_[(size_t) h.layer].tr[r] + lead(h.layer, h.e[(size_t) i], r);
    }
    void release(Hold& h) {
        if (!covers(h.layer)) return;
        std::lock_guard<std::mutex> lk(mu_);
        for (size_t i = 0; i < h.slot.size(); ++i) {
            const int s = h.slot[i];
            if (s < 0) continue;
            Slot& sp = slots_[(size_t) s];
            if (sp.gen == h.gen[i] && sp.pin > 0) --sp.pin;
            h.slot[i] = -1;
        }
    }

    /// The lookahead's guess: (layer, e) read behind every demand read, unpinned, unless it is here or on its way.
    bool guess(int layer, int e) {
        if (!covers(layer) || e < 0 || e >= geo_[(size_t) layer].n_exp) return false;
        std::lock_guard<std::mutex> lk(mu_);
        if (where_[(size_t) layer][(size_t) e] >= 0) return false;
        const int s = take_slot(layer, true);
        if (s < 0) return false;
        fill(s, layer, e, true, 0, false);
        counts_.guesses.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    /// The guesses for layers up to `layer` (the token is past them) still all queued: dropped unread, their slots
    /// free; one being read already finishes.  Returns how many.
    size_t cancel_stale(int layer) {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<std::pair<int, int>> n;   // (slot, its jobs queued)
        {
            std::lock_guard<std::mutex> qk(q_mu_);
            if (guess_q_.empty()) return 0;
            for (const Job& j : guess_q_) {
                if (j.layer > layer) continue;
                auto it = std::find_if(n.begin(), n.end(), [&](const std::pair<int, int>& x) { return x.first == j.slot; });
                if (it == n.end()) n.push_back({j.slot, 1});
                else ++it->second;
            }
            n.erase(std::remove_if(n.begin(), n.end(),
                                   [&](const std::pair<int, int>& x) {
                                       const Slot& sp = slots_[(size_t) x.first];
                                       return x.second < 3 || !sp.guess || sp.pin > 0;
                                   }),
                    n.end());
            if (n.empty()) return 0;
            guess_q_.erase(std::remove_if(guess_q_.begin(), guess_q_.end(),
                                          [&](const Job& j) {
                                              return std::find_if(n.begin(), n.end(), [&](const std::pair<int, int>& x) {
                                                         return x.first == j.slot;
                                                     }) != n.end();
                                          }),
                           guess_q_.end());
        }
        for (const auto& x : n) {
            slots_[(size_t) x.first].left.store(0, std::memory_order_release);
            drop(x.first);
        }
        counts_.cancelled.fetch_add((int64_t) n.size(), std::memory_order_relaxed);
        return n.size();
    }

    /// Reads the experts of `list` (layer, expert; hottest first) into free slots - none given up for them - and
    /// returns once they are in: the first ones that fit, `leave` bytes of the budget left over (one whose class has no
    /// room left is passed over: a later one's may have).  Each later one is put before the earlier ones in the
    /// policy's order, so the coldest is the first to give way.  Returns how many were read.
    size_t preload(const std::vector<std::pair<int, int>>& list, size_t leave) {
        std::vector<int> mine;
        {
            std::lock_guard<std::mutex> lk(mu_);
            const size_t room = cfg_.budget > leave ? cfg_.budget - leave : 0;
            size_t held = held_bytes_locked();
            for (const auto& x : list) {
                const int layer = x.first, e = x.second;
                if (!covers(layer) || e < 0 || e >= geo_[(size_t) layer].n_exp) continue;
                if (where_[(size_t) layer][(size_t) e] >= 0) continue;
                const size_t b = slot_bytes(layer);
                if (held + b > room) continue;
                const int s = take_slot(layer, false);
                if (s < 0) continue;
                fill(s, layer, e, false, 0, true);
                mine.push_back(s);
                held += b;
            }
        }
        for (;;) {
            uint64_t seen;
            {
                std::lock_guard<std::mutex> lk(done_mu_);
                seen = done_;
            }
            bool busy = false;
            for (int s : mine) busy = busy || slots_[(size_t) s].left.load(std::memory_order_acquire) > 0;
            if (!busy) break;
            std::unique_lock<std::mutex> lk(done_mu_);
            done_cv_.wait_for(lk, std::chrono::milliseconds(5), [&] { return done_ != seen; });
        }
        std::lock_guard<std::mutex> lk(mu_);
        size_t n = 0;
        for (int s : mine) {
            settle(s);
            n += slots_[(size_t) s].state == kReady ? 1 : 0;
        }
        return n;
    }

    /// An expert whose bytes are in memory already (src[r]: its matrix r) copied into a slot - one given up for it if
    /// need be - and held from now on.  False: it is here already, or no slot could be had.
    bool put(int layer, int e, const uint8_t* const src[3]) {
        if (!covers(layer) || e < 0 || e >= geo_[(size_t) layer].n_exp) return false;
        int s;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (where_[(size_t) layer][(size_t) e] >= 0) return false;
            s = take_slot(layer, true);
            if (s < 0) return false;
            Slot& sp = slots_[(size_t) s];   // (loading, pinned: nothing takes it while the copy runs)
            sp.layer = layer;
            sp.e = e;
            sp.pin = 1;
            sp.guess = false;
            sp.state = kLoading;
            sp.bad.store(false, std::memory_order_relaxed);
            sp.left.store(1, std::memory_order_relaxed);
            where_[(size_t) layer][(size_t) e] = s;
            link_back(s);
        }
        const Geo& g = geo_[(size_t) layer];
        for (int r = 0; r < 3; ++r)
            std::memcpy(slots_[(size_t) s].base + lay_[(size_t) layer].tr[r] + lead(layer, e, r), src[r], (size_t) g.bytes[r]);
        {
            std::lock_guard<std::mutex> lk(mu_);
            Slot& sp = slots_[(size_t) s];
            sp.left.store(0, std::memory_order_release);
            --sp.pin;
            settle(s);
        }
        counts_.put.fetch_add(1, std::memory_order_relaxed);
        landed();   // (a window waiting for it as a miss)
        return true;
    }

    /// (layer, e) here and read: its slot pinned (an ask - the policy sees it), else -1.  mat_of() reads it until
    /// unpin().  Nothing is read for an expert not here.
    int acquire(int layer, int e) {
        if (!covers(layer) || e < 0 || e >= geo_[(size_t) layer].n_exp) return -1;
        std::lock_guard<std::mutex> lk(mu_);
        const int s = where_[(size_t) layer][(size_t) e];
        if (s < 0) return -1;
        settle(s);
        Slot& sp = slots_[(size_t) s];
        if (sp.state != kReady || sp.layer != layer || sp.e != e) return -1;
        ask(layer, e);
        if (sp.guess) {
            counts_.guess_used.fetch_add(1, std::memory_order_relaxed);
            sp.guess = false;
        } else {
            counts_.hits.fetch_add(1, std::memory_order_relaxed);
        }
        ++sp.pin;
        unlink(s);
        link_back(s);
        return s;
    }
    const uint8_t* mat_of(int s, int r) const {
        const Slot& sp = slots_[(size_t) s];
        return sp.base + lay_[(size_t) sp.layer].tr[r] + lead(sp.layer, sp.e, r);
    }
    void unpin(int s) {
        std::lock_guard<std::mutex> lk(mu_);
        if (slots_[(size_t) s].pin > 0) --slots_[(size_t) s].pin;
    }
    /// (layer, e) here and read now (no pin: the answer may change at once - for estimates)
    bool has(int layer, int e) const {
        if (!covers(layer) || e < 0 || e >= geo_[(size_t) layer].n_exp) return false;
        std::lock_guard<std::mutex> lk(mu_);
        const int s = where_[(size_t) layer][(size_t) e];
        if (s < 0) return false;
        const Slot& sp = slots_[(size_t) s];
        return sp.state == kReady ||
               (sp.state == kLoading && sp.left.load(std::memory_order_acquire) == 0 && !sp.bad.load(std::memory_order_acquire));
    }

    /// The free RAM looked at: below `floor` chunks are given back from the end (stopping at one with a slot in use -
    /// pinned or reading) until it is back above it; a new chunk is allowed again once grow_room beside it is free (a
    /// class's first: halfway from floor to grow_room).
    /// Returns the chunks given back.
    size_t watch() {
        const uint64_t av = cfg_.ram_avail ? cfg_.ram_avail() : ~uint64_t{0};
        std::lock_guard<std::mutex> lk(mu_);
        for (size_t k = 0; k < cls_.size(); ++k)   // (the reads that landed: a failed one's slot back)
            for (int s = cls_[k].head; s >= 0;) {
                const int nx = slots_[(size_t) s].next;
                settle(s);
                s = nx;
            }
        if (av == ~uint64_t{0}) return 0;
        const size_t gave = av < cfg_.floor ? shrink((size_t) (cfg_.floor - av)) : 0;
        for (size_t k = 0; k < cls_.size(); ++k)
            cls_[k].grow_ok = av >= cfg_.floor && av >= ram_need((int) k, chunk_n((int) k) * cls_[k].bytes);
        return gave;
    }

    /// watch() every `ms` on a thread of its own until the store ends (so the free RAM is never read on the CPU lane's
    /// way)
    void watch_every(int ms) {
        if (ms <= 0 || w_th_.joinable()) return;
        w_th_ = std::thread([this, ms] {
            std::unique_lock<std::mutex> lk(w_mu_);
            while (!w_cv_.wait_for(lk, std::chrono::milliseconds(ms), [&] { return w_quit_; })) {
                lk.unlock();
                watch();
                lk.lock();
            }
        });
    }

private:
    static constexpr uint8_t kFree = 0, kLoading = 1, kReady = 2;
    struct Slot {
        uint8_t* base = nullptr;            // (page-aligned; null: its chunk is not allocated)
        int layer = -1, e = -1, pin = 0;
        int cls = -1;                       // its chunk's class
        uint8_t state = kFree;              // (mu_)
        bool guess = false;                 // read on a guess, not asked for since
        int prev = -1, next = -1;           // its class's order (oldest first): every slot not free
        std::atomic<int> left{0};           // its matrices' reads still to land
        std::atomic<bool> bad{false};       // a read failed
        uint64_t gen = 0;                   // (drop()s so far)
    };
    struct Job {
        int slot = -1, layer = -1, e = -1, r = 0, file = -1;
        uint64_t off = 0, len = 0;
        uint8_t* dst = nullptr;
        bool guess = false;
    };
    struct Class {   // the layers whose slots are of one size
        size_t bytes = 0;           // a slot's
        size_t hold = 0;            // the most experts its layers hold (a slot more would hold nothing)
        size_t slots = 0;           // allocated
        std::vector<int> free;      // its slots holding no expert
        int head = -1, tail = -1;   // the others, in the policy's order (oldest first)
        bool grow_ok = true;        // a new chunk allowed (the free RAM watch() last saw)
    };
    struct Lay {
        int cls = -1;
        size_t tr[3] = {0, 0, 0};   // matrix r's region in a slot
    };
    struct Chunk {
        uint8_t* p = nullptr;
        size_t bytes = 0, first = 0;   // (its slots: first up to the next chunk's)
    };

    void link_back(int s) {   // (its class's order)
        Slot& sp = slots_[(size_t) s];
        Class& c = cls_[(size_t) sp.cls];
        sp.prev = c.tail;
        sp.next = -1;
        if (c.tail >= 0) slots_[(size_t) c.tail].next = s;
        else c.head = s;
        c.tail = s;
    }
    void link_front(int s) {
        Slot& sp = slots_[(size_t) s];
        Class& c = cls_[(size_t) sp.cls];
        sp.next = c.head;
        sp.prev = -1;
        if (c.head >= 0) slots_[(size_t) c.head].prev = s;
        else c.tail = s;
        c.head = s;
    }
    void unlink(int s) {
        Slot& sp = slots_[(size_t) s];
        Class& c = cls_[(size_t) sp.cls];
        if (sp.prev >= 0) slots_[(size_t) sp.prev].next = sp.next;
        else if (c.head == s) c.head = sp.next;
        if (sp.next >= 0) slots_[(size_t) sp.next].prev = sp.prev;
        else if (c.tail == s) c.tail = sp.prev;
        sp.prev = sp.next = -1;
    }
    void ask(int layer, int e) {   // (lfu's heat)
        heat_[(size_t) layer][(size_t) e] += 1.0f;
        if (cfg_.policy == kLfu && cfg_.decay_every > 0 && ++asks_[(size_t) layer] % (uint64_t) cfg_.decay_every == 0)
            for (float& x : heat_[(size_t) layer]) x *= cfg_.decay;
    }
    size_t held_bytes_locked() const {   // (mu_ held)
        size_t b = 0;
        for (const Class& k : cls_) b += (k.slots - k.free.size()) * k.bytes;
        return b;
    }
    // the slot empty again (its expert forgotten, its place in the order gone)
    void drop(int s) {
        Slot& sp = slots_[(size_t) s];
        if (sp.layer >= 0 && sp.e >= 0 && where_[(size_t) sp.layer][(size_t) sp.e] == s)
            where_[(size_t) sp.layer][(size_t) sp.e] = -1;
        unlink(s);
        sp.layer = sp.e = -1;
        sp.pin = 0;
        sp.guess = false;
        sp.state = kFree;
        ++sp.gen;
        cls_[(size_t) sp.cls].free.push_back(s);
    }
    // a slot whose reads have all landed: read (whole) or empty again (a read failed)
    void settle(int s) {
        Slot& sp = slots_[(size_t) s];
        if (sp.state != kLoading || sp.left.load(std::memory_order_acquire) > 0) return;
        if (!sp.bad.load(std::memory_order_acquire)) {
            sp.state = kReady;
            return;
        }
        counts_.failed.fetch_add(1, std::memory_order_relaxed);
        drop(s);
    }
    // a guess's queued reads now demand reads (mu_ held)
    void promote(int s) {
        {
            std::lock_guard<std::mutex> qk(q_mu_);
            for (auto it = guess_q_.begin(); it != guess_q_.end();) {
                if (it->slot == s) {
                    Job j = *it;
                    j.guess = false;
                    demand_q_.push_back(j);
                    it = guess_q_.erase(it);
                } else {
                    ++it;
                }
            }
        }
        q_cv_.notify_all();
    }
    // slot s takes (layer, e): its three reads queued (mu_ held).  front: first in the policy's order (preload)
    void fill(int s, int layer, int e, bool guess, int pin, bool front) {
        Slot& sp = slots_[(size_t) s];
        sp.layer = layer;
        sp.e = e;
        sp.pin = pin;
        sp.guess = guess;
        sp.state = kLoading;
        sp.bad.store(false, std::memory_order_relaxed);
        sp.left.store(3, std::memory_order_release);
        where_[(size_t) layer][(size_t) e] = s;
        if (front) link_front(s);
        else link_back(s);
        const Geo& g = geo_[(size_t) layer];
        {
            std::lock_guard<std::mutex> qk(q_mu_);
            for (int r = 0; r < 3; ++r) {
                Job j;
                j.slot = s;
                j.layer = layer;
                j.e = e;
                j.r = r;
                j.file = g.file[r];
                j.off = g.off[r] + (uint64_t) e * g.stride[r];
                j.len = g.bytes[r];
                j.dst = sp.base + lay_[(size_t) layer].tr[r];
                j.guess = guess;
                (guess ? guess_q_ : demand_q_).push_back(j);
            }
        }
        q_cv_.notify_all();
        counts_.bytes.fetch_add((int64_t) (g.bytes[0] + g.bytes[1] + g.bytes[2]), std::memory_order_relaxed);
    }
    // the room kept for the classes other than k that hold no slots: their first chunks (mu_ held)
    size_t kept_for(int k) const {
        size_t b = 0;
        for (size_t o = 0; o < cls_.size(); ++o)
            if ((int) o != k && cls_[o].slots == 0) b += std::min(floor_, cls_[o].hold) * cls_[o].bytes;
        return b;
    }
    // the slots a new chunk of class k takes: as many as the budget has left beside the room kept for the classes that
    // hold none (each one's first chunk, at most its hold) - at most chunk_slots, and no more than it holds (mu_ held)
    size_t chunk_n(int k) const {
        const Class& c = cls_[(size_t) k];
        const size_t kept = kept_for(k), unspent = cfg_.budget - alloc_;
        return std::min({chunk_slots_, unspent > kept ? (unspent - kept) / c.bytes : 0, c.hold - c.slots});
    }
    // the free RAM a new chunk of class k (`bytes`) needs: grow_room beside it and the room kept for the classes that
    // hold none - or, its first chunk, only halfway from `floor` to grow_room beside it: room above floor, so the
    // give-back below floor does not take it again at once (mu_ held)
    uint64_t ram_need(int k, size_t bytes) const {
        const Class& c = cls_[(size_t) k];
        const uint64_t first = std::max<uint64_t>(cfg_.floor, ((uint64_t) cfg_.floor + cfg_.grow_room) / 2);
        return (uint64_t) bytes + (c.slots > 0 ? (uint64_t) cfg_.grow_room + kept_for(k) : first);
    }
    // a chunk more for class k (chunk_n's slots) while the free RAM allows (ram_need) (mu_ held)
    bool grow(int k) {
        Class& c = cls_[(size_t) k];
        if (!c.grow_ok) return false;
        const size_t n = chunk_n(k), bytes = n * c.bytes;
        if (n == 0) return false;
        const uint64_t av = cfg_.ram_avail ? cfg_.ram_avail() : ~uint64_t{0};
        if (av != ~uint64_t{0} && av < ram_need(k, bytes)) {
            c.grow_ok = false;   // (until watch() sees the room)
            return false;
        }
        uint8_t* p = (uint8_t*) os::anon_alloc(bytes, -1, 1);
        if (p == nullptr) {
            c.grow_ok = false;
            return false;
        }
        chunks_.push_back({p, bytes, nalloc_});
        for (size_t i = n; i-- > 0;) {   // (the lowest first out of the free list)
            Slot& sp = slots_[nalloc_ + i];
            sp.base = p + i * c.bytes;
            sp.cls = k;
            c.free.push_back((int) (nalloc_ + i));
        }
        nalloc_ += n;
        c.slots += n;
        alloc_ += bytes;
        return true;
    }
    // the slot of class k to give up: its oldest unpinned read one (lru), or of those the lowest heat, the oldest of
    // equals (lfu).  -1: none - or a landed read of the class failed on the way and freed its slot (mu_ held)
    int victim(int k) {
        int best = -1;
        float bh = 0.0f;
        for (int s = cls_[(size_t) k].head; s >= 0;) {
            Slot& sp = slots_[(size_t) s];
            const int nx = sp.next;
            if (sp.state == kLoading) {
                settle(s);
                if (!cls_[(size_t) k].free.empty()) return -1;
            }
            if (sp.state == kReady && sp.pin == 0) {
                if (cfg_.policy == kLru) return s;
                const float h = heat_[(size_t) sp.layer][(size_t) sp.e];
                if (best < 0 || h < bh) {
                    best = s;
                    bh = h;
                }
            }
            s = nx;
        }
        return best;
    }
    // a slot for a new expert of `layer`: a free one of its class, a new chunk's, or (evict) one of its class given up
    // (mu_ held); -1: none
    int take_slot(int layer, bool evict) {
        const int k = lay_[(size_t) layer].cls;
        std::vector<int>& fr = cls_[(size_t) k].free;
        if (fr.empty()) grow(k);
        if (fr.empty() && evict) {
            const int v = victim(k);
            if (v >= 0) {
                if (slots_[(size_t) v].guess) counts_.guess_wasted.fetch_add(1, std::memory_order_relaxed);
                counts_.evicted.fetch_add(1, std::memory_order_relaxed);
                drop(v);
            }
        }
        if (fr.empty()) return -1;
        const int s = fr.back();
        fr.pop_back();
        return s;
    }
    // chunks given back from the end until `want` bytes are (mu_ held); returns how many
    size_t shrink(size_t want) {
        size_t done = 0, gave = 0;
        while (gave < want && !chunks_.empty()) {
            const Chunk ch = chunks_.back();
            bool busy = false;
            for (size_t i = ch.first; i < nalloc_ && !busy; ++i) {
                settle((int) i);
                busy = slots_[i].state == kLoading || slots_[i].pin > 0;
            }
            if (busy) break;
            for (size_t i = ch.first; i < nalloc_; ++i)
                if (slots_[i].state != kFree) {
                    counts_.evicted.fetch_add(1, std::memory_order_relaxed);
                    drop((int) i);
                }
            Class& c = cls_[(size_t) slots_[ch.first].cls];
            c.free.erase(std::remove_if(c.free.begin(), c.free.end(), [&](int s) { return (size_t) s >= ch.first; }),
                         c.free.end());
            c.slots -= nalloc_ - ch.first;
            for (size_t i = ch.first; i < nalloc_; ++i) slots_[i].base = nullptr;
            os::anon_free(ch.p, ch.bytes);
            chunks_.pop_back();
            nalloc_ = ch.first;
            alloc_ -= ch.bytes;
            gave += ch.bytes;
            ++done;
            counts_.shrunk.fetch_add(1, std::memory_order_relaxed);
        }
        return done;
    }
    void landed() {
        {
            std::lock_guard<std::mutex> lk(done_mu_);
            ++done_;
        }
        done_cv_.notify_all();
    }
    // one matrix into its region: unbuffered (the aligned span, the bytes landing at dst + the lead), else buffered
    // straight to where they belong.  A file whose unbuffered read fails where the buffered one then works (a file
    // system or a USB bridge that opens with O_DIRECT but refuses its reads) is read buffered from then on
    bool read(const Job& j, std::vector<os::fd_t>& own, std::vector<char>& tried) {
        const size_t f = (size_t) j.file;
        bool direct_failed = false;
        if (f < files_.direct.size() && files_.direct[f] != os::kBadFd && !no_direct_[f].load(std::memory_order_relaxed)) {
            if (os::pread_direct(files_.direct[f], j.dst, j.len, j.off)) {
                counts_.direct_bytes.fetch_add((int64_t) j.len, std::memory_order_relaxed);
                return true;
            }
            direct_failed = true;
        }
        if (cfg_.unbuffered && f < files_.paths.size()) {
            if (own.size() <= f) {
                own.resize(f + 1, os::kBadFd);
                tried.resize(f + 1, 0);
            }
            if (!tried[f]) {
                tried[f] = 1;
                own[f] = os::open_unbuffered(files_.paths[f]);
            }
            if (own[f] != os::kBadFd && os::pread_unbuffered(own[f], j.dst, j.len, j.off)) {
                counts_.direct_bytes.fetch_add((int64_t) j.len, std::memory_order_relaxed);
                return true;
            }
        }
        const bool ok = f < files_.buf.size() && files_.buf[f] != os::kBadFd &&
                        os::pread_par(files_.buf[f], j.dst + j.off % kAlign, j.len, j.off);
        if (ok && direct_failed && !no_direct_[f].exchange(true)) {
            counts_.direct_off.fetch_add(1, std::memory_order_relaxed);
            std::fprintf(stderr, "ds4: RAM store: unbuffered reads of file %zu fail - it is read through the page cache\n", f);
        }
        return ok;
    }
    void reader() {
        std::vector<os::fd_t> own;   // (unbuffered: this reader's own handles, opened as first needed)
        std::vector<char> tried;
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> lk(q_mu_);
                q_cv_.wait(lk, [&] { return quit_ || !demand_q_.empty() || !guess_q_.empty(); });
                if (quit_) break;
                std::deque<Job>& q = !demand_q_.empty() ? demand_q_ : guess_q_;
                j = q.front();
                q.pop_front();
            }
            if (cfg_.before_read) cfg_.before_read(ReadNote{j.layer, j.e, j.r, j.guess, j.dst, j.off, j.len});
            const bool ok = read(j, own, tried);
            Slot& sp = slots_[(size_t) j.slot];
            if (!ok) sp.bad.store(true, std::memory_order_release);
            sp.left.fetch_sub(1, std::memory_order_acq_rel);
            landed();
        }
        for (os::fd_t f : own) os::close_fd(f);
    }

    Cfg cfg_;
    std::vector<Geo> geo_;
    Files files_;
    std::vector<Lay> lay_;      // [layer]: its class, its regions
    std::vector<Class> cls_;
    size_t least_ = 0;   // the smallest slot (of the classes that hold any)
    size_t floor_ = 0;   // the room kept for a class that holds no slots: this many, at most its hold
    size_t nmax_ = 0, nalloc_ = 0, alloc_ = 0, chunk_slots_ = 1;   // (alloc_: the chunks' bytes, at most the budget)
    std::unique_ptr<Slot[]> slots_;
    std::vector<Chunk> chunks_;   // (in the order allocated: their slots in that order too)
    std::vector<std::vector<int>> where_;   // [layer][e]: its slot, -1
    std::vector<std::vector<float>> heat_;
    std::vector<uint64_t> asks_;
    mutable std::mutex mu_;   // (the slots' bookkeeping; before q_mu_ when both)
    std::mutex q_mu_;
    std::condition_variable q_cv_;
    std::deque<Job> demand_q_, guess_q_;
    bool quit_ = false;
    std::mutex done_mu_;
    std::condition_variable done_cv_;
    uint64_t done_ = 0;
    std::unique_ptr<std::atomic<bool>[]> no_direct_;   // [file]: unbuffered reads fail there
    std::vector<std::thread> th_;
    std::mutex w_mu_;   // (watch_every's thread)
    std::condition_variable w_cv_;
    bool w_quit_ = false;
    std::thread w_th_;
    Counts counts_;
};

}  // namespace strata::ds4
