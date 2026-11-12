// src/ds4/ds4_store.hpp - the RAM expert store (--ssd-cache; opt-in): the CPU lane's experts that are read in place (a
// GGUF's routed experts the GPUs do not hold, --mmap-experts) kept whole in RAM slots of the store's own instead of the
// OS page cache - which knows nothing of which experts are hot, reads a page at a time as the lane faults, and gives
// them up to any other I/O.
//  - a slot holds one expert: its three matrices, each in a region of whole 4096-byte pages, read from the model's
//    files past the page cache (Linux: O_DIRECT - Pack's dfds; elsewhere, or where the file system refuses it, through
//    the buffered handles), so each read lands its matrix at the region's start + (file offset % 4096): nothing is moved
//    after the read, and a matrix lies at the page offset the file's mapping has it at
//  - which expert gives way: lru (the least recently asked for) or lfu (the lowest heat - its asks, decayed layer by
//    layer), never one a window is computing (pinned) or one still being read
//  - reads queue by kind: a demand read (the CPU lane waits for it) always before a guess (the lookahead's); a guess
//    asked for becomes a demand read, and a guess still queued once the token is past its layer is dropped unread
//  - the slots come in chunks as they are first needed, while `grow_room` of RAM stays free beside them; watch() gives
//    chunks back from the end when the free RAM falls below `floor`
//  - a size class a slot layout: the layers whose three matrices need the same regions share one, so a GGUF that mixes
//    types by layer (unsloth's UD quants: IQ2_XXS beside IQ3_S) holds each expert in a slot of its own size, not the
//    largest layer's.  The budget goes to the classes by their bytes (each holds the same share of its experts); each
//    has its own slots, order and chunks, and an expert gives way to one of its class
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
    /// of file file[r] (n_exp 0: a layer the store does not cover)
    struct Geo {
        int n_exp = 0;
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
        size_t budget = 0;                     // bytes at most (whole slots)
        size_t floor = (size_t) 4 << 30;       // free RAM below this: watch() gives chunks back
        size_t grow_room = (size_t) 8 << 30;   // a new chunk only while this much RAM stays free beside it
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
        for (const Cls& k : cls_)
            for (const auto& c : k.chunks) os::anon_free(c.first, c.second);
    }

    /// a slot's bytes for these layers, and where matrix r's region starts in it (tr): each region holds the aligned
    /// read's span of the layers' largest matrix r - the matrix and its lead, whole pages
    static size_t slot_layout(const std::vector<Geo>& geo, size_t* tr = nullptr) {
        size_t off = 0;
        for (int r = 0; r < 3; ++r) {
            size_t room = 0;
            for (const Geo& g : geo)
                if (g.n_exp > 0) room = std::max<size_t>(room, (size_t) ((g.bytes[r] + 2 * kAlign - 1) / kAlign * kAlign));
            if (tr != nullptr) tr[r] = off;
            off += room;
        }
        return off;
    }
    /// one layer's slot (its size class): the same layout over that layer alone
    static size_t layer_layout(const Geo& g, size_t* tr = nullptr) { return slot_layout(std::vector<Geo>{g}, tr); }
    /// the bytes holding count[l] experts of each layer l in its own class's slots (every expert it could hold: `cap`)
    static uint64_t bytes_for(const std::vector<Geo>& geo, const std::vector<size_t>& count) {
        uint64_t b = 0;
        for (size_t l = 0; l < geo.size() && l < count.size(); ++l)
            if (geo[l].n_exp > 0) b += (uint64_t) layer_layout(geo[l]) * (uint64_t) count[l];
        return b;
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
        // the size classes: the layers of one slot layout together; the budget by each class's bytes (n_exp slots)
        cls_.clear();
        cls_of_.assign(geo_.size(), -1);
        std::vector<double> weight;
        for (size_t l = 0; l < geo_.size(); ++l) {
            if (geo_[l].n_exp <= 0) continue;
            size_t t[3];
            const size_t sb = layer_layout(geo_[l], t);
            size_t c = 0;
            while (c < cls_.size() && !(cls_[c].slot_bytes == sb && cls_[c].tr[0] == t[0] && cls_[c].tr[1] == t[1] &&
                                        cls_[c].tr[2] == t[2]))
                ++c;
            if (c == cls_.size()) {
                cls_.emplace_back();
                cls_[c].slot_bytes = sb;
                for (int r = 0; r < 3; ++r) cls_[c].tr[r] = t[r];
                weight.push_back(0.0);
            }
            cls_of_[l] = (int) c;
            weight[c] += (double) geo_[l].n_exp * (double) sb;
        }
        if (cls_.empty()) {
            err = "the RAM store: no layer of experts to hold";
            return false;
        }
        double wsum = 0.0;
        for (double w : weight) wsum += w;
        nmax_ = 0;
        slot_bytes_ = 0;
        for (size_t c = 0; c < cls_.size(); ++c) {
            Cls& k = cls_[c];
            k.nmax = (size_t) ((double) cfg_.budget * (weight[c] / wsum) / (double) k.slot_bytes);
            k.first = nmax_;
            nmax_ += k.nmax;
            slot_bytes_ = std::max(slot_bytes_, k.slot_bytes);
        }
        if (nmax_ == 0) {
            err = "the RAM store: its size holds no expert";
            return false;
        }
        chunk_slots_ = std::max<size_t>(1, cfg_.chunk_slots);
        no_direct_.reset(new std::atomic<bool>[files_.buf.size() + 1]);
        for (size_t i = 0; i <= files_.buf.size(); ++i) no_direct_[i].store(false, std::memory_order_relaxed);
        slots_.reset(new Slot[nmax_]);
        for (size_t c = 0; c < cls_.size(); ++c)
            for (size_t i = 0; i < cls_[c].nmax; ++i) slots_[cls_[c].first + i].cls = (int) c;
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

    size_t slot_bytes() const { return slot_bytes_; }   // (the largest class's)
    size_t slot_bytes_min() const {
        size_t m = slot_bytes_;
        for (const Cls& k : cls_) m = std::min(m, k.slot_bytes);
        return m;
    }
    size_t classes() const { return cls_.size(); }
    size_t slots_max() const { return nmax_; }
    uint64_t bytes_max() const {
        uint64_t b = 0;
        for (const Cls& k : cls_) b += (uint64_t) k.nmax * k.slot_bytes;
        return b;
    }
    size_t slots_now() const {   // (allocated)
        std::lock_guard<std::mutex> lk(mu_);
        size_t n = 0;
        for (const Cls& k : cls_) n += k.nalloc;
        return n;
    }
    size_t held() const {   // (slots holding an expert, or reading one)
        std::lock_guard<std::mutex> lk(mu_);
        size_t n = 0;
        for (const Cls& k : cls_) n += k.nalloc - k.free.size();
        return n;
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
                h.st[(size_t) i] = sp.state == kReady ? kHit : kMiss;
                continue;
            }
            counts_.misses.fetch_add(1, std::memory_order_relaxed);
            s = take_slot(cls_of_[(size_t) layer], true);
            if (s < 0) {
                counts_.no_slot.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            fill(s, layer, e, false, 1, false);
            h.slot[(size_t) i] = s;
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
                        if (sp.state == kReady && sp.layer == h.layer && sp.e == h.e[i]) {
                            h.st[i] = kHit;
                        } else {   // (failed: its slot went)
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
        const Slot& sp = slots_[(size_t) h.slot[(size_t) i]];
        return sp.base + cls_[(size_t) sp.cls].tr[r] + lead(h.layer, h.e[(size_t) i], r);
    }
    void release(Hold& h) {
        if (!covers(h.layer)) return;
        std::lock_guard<std::mutex> lk(mu_);
        for (size_t i = 0; i < h.slot.size(); ++i) {
            const int s = h.slot[i];
            if (s < 0) continue;
            Slot& sp = slots_[(size_t) s];
            if (sp.layer == h.layer && sp.e == h.e[i] && sp.pin > 0) --sp.pin;
            h.slot[i] = -1;
        }
    }

    /// The lookahead's guess: (layer, e) read behind every demand read, unpinned, unless it is here or on its way.
    bool guess(int layer, int e) {
        if (!covers(layer) || e < 0 || e >= geo_[(size_t) layer].n_exp) return false;
        std::lock_guard<std::mutex> lk(mu_);
        if (where_[(size_t) layer][(size_t) e] >= 0) return false;
        const int s = take_slot(cls_of_[(size_t) layer], true);
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
    /// returns once they are in: the first ones that fit, `leave` slots left over.  Each later one is put before the
    /// earlier ones in the policy's order, so the coldest is the first to give way.  Returns how many were read.
    size_t preload(const std::vector<std::pair<int, int>>& list, size_t leave) {
        std::vector<int> mine;
        {
            std::lock_guard<std::mutex> lk(mu_);
            const size_t room = nmax_ > leave ? nmax_ - leave : 0;
            for (size_t k = 0; k < list.size() && mine.size() < room; ++k) {
                const int layer = list[k].first, e = list[k].second;
                if (!covers(layer) || e < 0 || e >= geo_[(size_t) layer].n_exp) continue;
                if (where_[(size_t) layer][(size_t) e] >= 0) continue;
                const int s = take_slot(cls_of_[(size_t) layer], false);
                if (s < 0) continue;   // (its class is full: another's may not be)
                fill(s, layer, e, false, 0, true);
                mine.push_back(s);
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
            s = take_slot(cls_of_[(size_t) layer], true);
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
        const size_t* tr = cls_[(size_t) slots_[(size_t) s].cls].tr;
        for (int r = 0; r < 3; ++r) std::memcpy(slots_[(size_t) s].base + tr[r] + lead(layer, e, r), src[r], (size_t) g.bytes[r]);
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
        return sp.base + cls_[(size_t) sp.cls].tr[r] + lead(sp.layer, sp.e, r);
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
    /// pinned or reading) until it is back above it; a new chunk is allowed again once grow_room beside it is free.
    /// Returns the chunks given back.
    size_t watch() {
        const uint64_t av = cfg_.ram_avail ? cfg_.ram_avail() : ~uint64_t{0};
        std::lock_guard<std::mutex> lk(mu_);
        for (const Cls& k : cls_)
            for (int s = k.head; s >= 0;) {   // (the reads that landed: a failed one's slot back)
                const int nx = slots_[(size_t) s].next;
                settle(s);
                s = nx;
            }
        if (av == ~uint64_t{0}) return 0;
        const size_t cb = chunk_slots_ * slot_bytes_;
        size_t gave = 0;
        if (av < cfg_.floor) {
            gave = shrink((size_t) ((cfg_.floor - av + cb - 1) / cb));
            grow_ok_ = false;
        } else {
            grow_ok_ = av >= (uint64_t) cfg_.grow_room + cb;
        }
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
        int cls = 0;                        // its size class
        uint8_t state = kFree;              // (mu_)
        bool guess = false;                 // read on a guess, not asked for since
        int prev = -1, next = -1;           // the policy's order (oldest first): every slot not free
        std::atomic<int> left{0};           // its matrices' reads still to land
        std::atomic<bool> bad{false};       // a read failed
    };
    struct Job {
        int slot = -1, layer = -1, e = -1, r = 0, file = -1;
        uint64_t off = 0, len = 0;
        uint8_t* dst = nullptr;
        bool guess = false;
    };

    void link_back(int s) {
        Slot& sp = slots_[(size_t) s];
        Cls& k = cls_[(size_t) sp.cls];
        sp.prev = k.tail;
        sp.next = -1;
        if (k.tail >= 0) slots_[(size_t) k.tail].next = s;
        else k.head = s;
        k.tail = s;
    }
    void link_front(int s) {
        Slot& sp = slots_[(size_t) s];
        Cls& k = cls_[(size_t) sp.cls];
        sp.next = k.head;
        sp.prev = -1;
        if (k.head >= 0) slots_[(size_t) k.head].prev = s;
        else k.tail = s;
        k.head = s;
    }
    void unlink(int s) {
        Slot& sp = slots_[(size_t) s];
        Cls& k = cls_[(size_t) sp.cls];
        if (sp.prev >= 0) slots_[(size_t) sp.prev].next = sp.next;
        else if (k.head == s) k.head = sp.next;
        if (sp.next >= 0) slots_[(size_t) sp.next].prev = sp.prev;
        else if (k.tail == s) k.tail = sp.prev;
        sp.prev = sp.next = -1;
    }
    void ask(int layer, int e) {   // (lfu's heat)
        heat_[(size_t) layer][(size_t) e] += 1.0f;
        if (cfg_.policy == kLfu && cfg_.decay_every > 0 && ++asks_[(size_t) layer] % (uint64_t) cfg_.decay_every == 0)
            for (float& x : heat_[(size_t) layer]) x *= cfg_.decay;
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
                j.dst = sp.base + cls_[(size_t) sp.cls].tr[r];
                j.guess = guess;
                (guess ? guess_q_ : demand_q_).push_back(j);
            }
        }
        q_cv_.notify_all();
        counts_.bytes.fetch_add((int64_t) (g.bytes[0] + g.bytes[1] + g.bytes[2]), std::memory_order_relaxed);
    }
    // a chunk more of class c, while its budget and the free RAM allow (mu_ held)
    bool grow(int c) {
        Cls& k = cls_[(size_t) c];
        if (!grow_ok_ || k.nalloc >= k.nmax) return false;
        const size_t n = std::min(chunk_slots_, k.nmax - k.nalloc), bytes = n * k.slot_bytes;
        const uint64_t av = cfg_.ram_avail ? cfg_.ram_avail() : ~uint64_t{0};
        if (av != ~uint64_t{0} && av < (uint64_t) bytes + cfg_.grow_room) {
            grow_ok_ = false;   // (until watch() sees the room)
            return false;
        }
        uint8_t* p = (uint8_t*) os::anon_alloc(bytes, -1, 1);
        if (p == nullptr) {
            grow_ok_ = false;
            return false;
        }
        k.chunks.push_back({p, bytes});
        for (size_t i = n; i-- > 0;) {   // (the lowest first out of the free list)
            slots_[k.first + k.nalloc + i].base = p + i * k.slot_bytes;
            k.free.push_back((int) (k.first + k.nalloc + i));
        }
        k.nalloc += n;
        return true;
    }
    // the slot of class c to give up: the oldest unpinned read one (lru), or of those the lowest heat, the oldest of
    // equals (lfu).  -1: none - or a landed read failed on the way and freed its slot (mu_ held)
    int victim(int c) {
        Cls& k = cls_[(size_t) c];
        int best = -1;
        float bh = 0.0f;
        for (int s = k.head; s >= 0;) {
            Slot& sp = slots_[(size_t) s];
            const int nx = sp.next;
            if (sp.state == kLoading) {
                settle(s);
                if (!k.free.empty()) return -1;
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
    // a slot of class c for a new expert: a free one, a new chunk's, or (evict) one given up (mu_ held); -1: none
    int take_slot(int c, bool evict) {
        if (c < 0) return -1;
        Cls& k = cls_[(size_t) c];
        if (k.free.empty()) grow(c);
        if (k.free.empty() && evict) {
            const int v = victim(c);
            if (v >= 0) {
                if (slots_[(size_t) v].guess) counts_.guess_wasted.fetch_add(1, std::memory_order_relaxed);
                counts_.evicted.fetch_add(1, std::memory_order_relaxed);
                drop(v);
            }
        }
        if (k.free.empty()) return -1;
        const int s = k.free.back();
        k.free.pop_back();
        return s;
    }
    // class c's last chunk given back (mu_ held); false: it has none, or one of its slots is in use (pinned or reading)
    bool shrink_one(int c) {
        Cls& k = cls_[(size_t) c];
        if (k.chunks.empty()) return false;
        const size_t ch = k.chunks.size() - 1, first = k.first + ch * chunk_slots_, end = k.first + k.nalloc;
        for (size_t i = first; i < end; ++i) {
            settle((int) i);
            if (slots_[i].state == kLoading || slots_[i].pin > 0) return false;
        }
        for (size_t i = first; i < end; ++i)
            if (slots_[i].state != kFree) {
                counts_.evicted.fetch_add(1, std::memory_order_relaxed);
                drop((int) i);
            }
        k.free.erase(std::remove_if(k.free.begin(), k.free.end(), [&](int s) { return (size_t) s >= first; }), k.free.end());
        for (size_t i = first; i < end; ++i) slots_[i].base = nullptr;
        os::anon_free(k.chunks[ch].first, k.chunks[ch].second);
        k.chunks.pop_back();
        k.nalloc = first - k.first;
        counts_.shrunk.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    // up to `want` chunks given back, the class holding the most bytes first each time (mu_ held)
    size_t shrink(size_t want) {
        size_t done = 0;
        std::vector<char> stuck(cls_.size(), 0);
        while (done < want) {
            int c = -1;
            for (size_t i = 0; i < cls_.size(); ++i)
                if (!stuck[i] && !cls_[i].chunks.empty() &&
                    (c < 0 || cls_[i].nalloc * cls_[i].slot_bytes > cls_[(size_t) c].nalloc * cls_[(size_t) c].slot_bytes))
                    c = (int) i;
            if (c < 0) break;
            if (shrink_one(c)) ++done;
            else stuck[(size_t) c] = 1;
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
    // a size class: its slots are slots_[first .. first + nmax), the allocated ones [first, first + nalloc)
    struct Cls {
        size_t tr[3] = {0, 0, 0};   // matrix r's region in a slot
        size_t slot_bytes = 0, nmax = 0, nalloc = 0, first = 0;
        std::vector<std::pair<uint8_t*, size_t>> chunks;
        std::vector<int> free;
        int head = -1, tail = -1;   // the policy's order (oldest first)
    };
    std::vector<Cls> cls_;
    std::vector<int> cls_of_;   // [layer]: its class, -1
    size_t slot_bytes_ = 0, nmax_ = 0, chunk_slots_ = 1;
    std::unique_ptr<Slot[]> slots_;
    std::vector<std::vector<int>> where_;   // [layer][e]: its slot, -1
    std::vector<std::vector<float>> heat_;
    std::vector<uint64_t> asks_;
    bool grow_ok_ = true;
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
