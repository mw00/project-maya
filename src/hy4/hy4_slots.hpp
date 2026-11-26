// src/hy4/hy4_slots.hpp - maya-hy4's conversation slots (SLOTS, MAYA_HY4_SLOT_KEEP=1): the serve loop's table
// of parked conversations and its decisions, with no GPU and no model (hy4_main.cpp's serve() drives it; the bytes
// themselves are Model::conv_save / conv_load's).  Everything here is pure: a test drives it with made-up token lists.
//
// THE INTERFACE (namespace hy4::slots)
//
//   struct Pic {start, hash}            a picture in a state or a request: its first image token's position, its pixels'
//                                       hash (hy4_main's Pic / Held carry the same two fields)
//   struct Knobs                        the MAYA_HY4_SLOT_* settings; knobs_from(env, warn) reads them through any
//                                       `const char* env(const char* name)`, knobs_from_env(warn) through std::getenv.
//                                       on(): keep && cap > 0.  Invalid numbers keep the default (warn gets a line)
//   struct Geometry {rows, back}        the model's ring (Model::ring_rows) and the positions a forward reads before its
//                                       start (Model::ring_back: max(n_swa, ix_kpool) - 1) - hy4_ring.hpp's ring_holds'
//   shared_start(have, have_pics, ids, want)
//                                       the serve loop's reuse of a held state for a request (hy4_main.cpp, before this
//                                       existed: the loop at "the state's shared start"): the longest common prefix,
//                                       at most ids.size() - 1 (a prompt's last token is always read again), then cut
//                                       at any picture before it that is not in both at the same place with the same
//                                       hash (have's first, then want's - the serve loop's order)
//   usable(c, n, lo, geo)               where a forward may resume in a state of n tokens whose rings hold [lo, n):
//                                       c when ring_holds(c, n, lo, rows, back), else 0 (ring_holds is true for 0 and
//                                       for a run [L + back, n]: the largest such p <= c is c or 0)
//   struct Slot                         a parked conversation: its tokens (positions 0 .. n - 1), its pictures, lo (its
//                                       rings' lowest position held, Model::conv_save's), its last use (the table's
//                                       clock), its bytes (Model::conv_save's buffer; a test may leave data null and
//                                       set `size` alone)
//   class Table                         the slots in order of their parking:
//     best(ids, want, geo)              -> Match{index, len}: the slot whose usable() resume for this request is the
//                                       longest (len > 0; ties: the most recently used), index npos / len 0: none
//     put(slot, protect)                -> Put{superseded, evicted}: parks a slot (its `used` set to the newest).
//                                       First every older slot whose tokens are a prefix of the new one's (and whose
//                                       pictures before its end are the new one's) is dropped - an earlier state of
//                                       this conversation (ds4's supersede); then, while more than cap slots, or more
//                                       than ram_bytes over them all with more than one, the least recently used goes
//                                       (ds4's LRU) - never the new one, never `protect` (a slot about to be taken
//                                       back: a data pointer, or null).  So the table can hold the new slot and the
//                                       protected one over the caps when the two alone are over them
//     touch(i)                          slot i used now (a take)
//     drop(i), size(), operator[](i), ram_bytes(), find(data)
//   decide(state_n, from_cur, held, best, knobs)
//                                       -> Plan{park, take}: the switch rule (before the serve loop's own reuse, which
//                                       then runs unchanged):
//                                         take: best.len > from_cur + knobs.gain - the kept conversation saves more
//                                               than the one in the model would;
//                                         park: state_n >= knobs.min, the state not already in a slot (held: the model
//                                               unchanged since a save or a load, Model::conv_held), and either a take
//                                               or an in-place rewind that drops at least knobs.min tokens
//                                               (state_n - from_cur) - but never with a plain continuation's take.
//                                       A plain continuation (from_cur == state_n > 0) never parks: it takes a slot
//                                       that covers more than the state does by the gain (the state is then a prefix
//                                       of that slot's tokens - nothing lost), else does nothing.
//   after a park the serve loop asks best() again (put() may have dropped slots; the table's order changed) and takes
//   only if that one still passes the take test.
//
// THE FOLDER (v2: MAYA_HY4_SLOT_DIR, parked conversations kept across engine starts - the files' I/O and the writer
// thread are hy4_slot_dir.hpp's; what is decided about them stays here, pure)
//
//   Slot::file / disk / file_bytes / sum / id
//                                       a slot's files: file the path stem (<dir>/slot-<stamp>-<seq>; empty: none),
//                                       disk 0 none, 1 being written (the writer's job id), 2 written (.bin and .meta
//                                       complete), file_bytes the .bin's size, sum its checksum.  A slot is in RAM
//                                       (in_ram) when it has its bytes, or has neither bytes nor a file (a test's slot
//                                       of `size` alone); one with a file and no bytes is the folder's alone.  bytes()
//                                       is its RAM bytes (0 for the folder's alone)
//   Knobs::dir, folder_bytes, days, idle_s, quit_s
//                                       MAYA_HY4_SLOT_DIR (dir_set: the variable is there, empty or not), _GB (16),
//                                       _DAYS (7), _IDLE_S (5; 0 no idle save), _QUIT_S (10)
//   Table changes with the folder:
//     put(...)                          the caps count the slots in RAM alone (ram_count, ram_bytes): past them the
//                                       least recently used in RAM that has a file (written or being written) gives
//                                       up its bytes and stays as the folder's (Put::spilled); one without a file is
//                                       dropped as before (Put::evicted).  Every slot put drops (supersede, evict) is
//                                       moved to Put::gone - the caller removes its files / stops its write
//     best(...)                         skips a slot whose bytes are only in a write still running (no bytes, disk 1)
//     add(slot)                         a folder's slot found at the start: appended as the most recently used, no caps
//     trim_folder(cap, keep_id, keep2)  -> the file stems to remove: while the folder's slots (disk 1 and 2) hold more
//                                       than cap bytes, the least recently used written one (disk 2) goes - never
//                                       keep_id or keep2 (the one just written; one about to be taken back), never one
//                                       being written; one in RAM keeps its bytes and loses only its file, one of the
//                                       folder's alone is dropped
//     find_id(id), live(tokens, pics), ram_count(), disk_bytes(), slot(i) (a mutable reference)
//   checksum(p, n)                      the .bin's 64-bit checksum (4 lanes of 64-bit multiply-rotate, xxh64's shape:
//                                       several GB/s; integrity, not security)
//   Meta, meta_encode(meta), meta_decode(p, n, meta, why)
//                                       the .meta's bytes, every integer little-endian (u64 / i64: 8 bytes, i32: 4):
//                                         [0, 8)       "HY4META2" (ASCII; "HY4META1", v2's first format, is refused)
//                                         [8, 16)      u64 L: the identity's bytes
//                                         [16, 16+L)   the identity (Model::conv_identity's text, no terminator)
//                                         then         u64 bin_bytes (the .bin's size), u64 bin_checksum (checksum()
//                                                      of the whole .bin), u64 n, u64 lo, u64 ntok (== n),
//                                                      ntok x i32 tokens, u64 npics, npics x (i64 start, u64 hash)
//                                         then         u64 crc: CRC-32 (crc32() below = Python's zlib.crc32) of every
//                                                      byte before this field, in the low 32 bits, the high 32 zero
//                                         last 8       "HY4META2" again, nothing after it
//                                       So a test can rewrite a field (say the identity: L and its bytes) and recompute
//                                       crc = zlib.crc32(b[:-16]) for the bytes before it.  decode checks every length
//                                       and bound, then the crc, then the closing magic (why: the first that fails)
//   crc32(p, n, crc)                    zlib's CRC-32 (the .meta's own check)
//   file_kind(name)                     a folder entry's name: kBin / kMeta (slot-*.bin / .meta), kTmp (slot-*.tmp:
//                                       an interrupted write), kOther (not ours: left alone)
//   The serve loop's use of them (hy4_main.cpp): every park's slot is queued for the writer at once (write-through:
//   Slot::disk 1, then 2 when the writer reports it); MAYA_HY4_SLOT_IDLE_S idle seconds after a request the
//   conversation in the model is parked if it is in no slot (or its slot written if the folder lacks it) and every
//   slot in RAM without a file written - idle jobs, given up when a request's line comes; at QUIT, SIGTERM / SIGINT or
//   the input's end the same without the idle mark, within MAYA_HY4_SLOT_QUIT_S.  A take of the folder's alone reads
//   its .bin, checks its size, checksum and tokens against the .meta, then Model::conv_load; any failure removes the
//   files (the prompt is read from token 0).  At the start every slot-*.meta is decoded and checked (identity, its .bin's
//   size, MAYA_HY4_SLOT_DAYS by its time) - add() in the files' time order - and the rest of slot-* removed
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "hy4_ring.hpp"

namespace hy4::slots {

struct Pic {
    long long start = 0;
    uint64_t hash = 0;
};

struct Knobs {
    bool keep = false;                          // MAYA_HY4_SLOT_KEEP (off)
    int cap = 4;                                // MAYA_HY4_SLOT_CAP: conversations kept (0: none)
    uint64_t ram_bytes = (uint64_t) 4 << 30;    // MAYA_HY4_SLOT_RAM_GB (4): their bytes at most (0: the count alone)
    size_t min = 512;                           // MAYA_HY4_SLOT_MIN: the shortest conversation parked (tokens)
    int64_t ring = 0;                           // MAYA_HY4_SLOT_RING: ring positions kept a slot (0: all it holds)
    bool verify = false;                        // MAYA_HY4_SLOT_VERIFY: a restore read back and compared (tests)
    size_t gain = 256;                          // (a take: tokens a slot must save beyond the model's own reuse)
    // (v2: the folder)
    std::string dir;                            // MAYA_HY4_SLOT_DIR (unset: <pack>/slots, the caller's; set empty: none)
    bool dir_set = false;                       // (the variable is there)
    uint64_t folder_bytes = (uint64_t) 16 << 30;   // MAYA_HY4_SLOT_GB: the folder's .bin files at most (0: no folder)
    double days = 7;                            // MAYA_HY4_SLOT_DAYS: a file unused this long is removed at the start
    double idle_s = 5;                          // MAYA_HY4_SLOT_IDLE_S: the live conversation written after this idle
    double quit_s = 10;                         // MAYA_HY4_SLOT_QUIT_S: the engine's end's writes at most (s)
    bool on() const { return keep && cap > 0; }
};

// the knobs through env(name) -> value or null; an unreadable value keeps the default and adds a line to *warn
template <class Env>
Knobs knobs_from(Env&& env, std::string* warn = nullptr) {
    Knobs k;
    auto num = [&](const char* name, double lo, double hi, double& out) {
        const char* v = env(name);
        if (v == nullptr || *v == '\0') return false;
        char* end = nullptr;
        const double d = std::strtod(v, &end);
        while (end != nullptr && (*end == ' ' || *end == '\t')) ++end;
        if (end == nullptr || end == v || *end != '\0' || !std::isfinite(d) || d < lo || d > hi) {
            if (warn) *warn += std::string(warn->empty() ? "" : "; ") + "invalid " + name + "=" + v + ", the default kept";
            return false;
        }
        out = d;
        return true;
    };
    double d = 0;
    if (num("MAYA_HY4_SLOT_KEEP", -1e18, 1e18, d)) k.keep = d != 0;
    if (num("MAYA_HY4_SLOT_CAP", 0, 1 << 20, d)) k.cap = (int) d;
    if (num("MAYA_HY4_SLOT_RAM_GB", 0, 1 << 24, d)) k.ram_bytes = (uint64_t) (d * 1073741824.0);
    if (num("MAYA_HY4_SLOT_MIN", 1, 1e12, d)) k.min = (size_t) d;
    if (num("MAYA_HY4_SLOT_RING", 0, 1e12, d)) k.ring = (int64_t) d;
    if (num("MAYA_HY4_SLOT_VERIFY", -1e18, 1e18, d)) k.verify = d != 0;
    if (const char* v = env("MAYA_HY4_SLOT_DIR")) {
        k.dir_set = true;
        k.dir = v;
    }
    if (num("MAYA_HY4_SLOT_GB", 0, 1 << 24, d)) k.folder_bytes = (uint64_t) (d * 1073741824.0);
    if (num("MAYA_HY4_SLOT_DAYS", 0, 1e6, d)) k.days = d;
    if (num("MAYA_HY4_SLOT_IDLE_S", 0, 1e6, d)) k.idle_s = d;
    if (num("MAYA_HY4_SLOT_QUIT_S", 0, 3600, d)) k.quit_s = d;
    return k;
}
inline Knobs knobs_from_env(std::string* warn = nullptr) {
    return knobs_from([](const char* n) { return (const char*) std::getenv(n); }, warn);
}

struct Geometry {
    int64_t rows = 0;   // the rings' rows (Model::ring_rows)
    int64_t back = 0;   // the positions a forward reads before its start (Model::ring_back)
};

inline bool same_pic(const Pic& a, const Pic& b) { return a.start == b.start && a.hash == b.hash; }

inline size_t shared_start(const std::vector<int>& have, const std::vector<Pic>& have_pics, const std::vector<int>& ids,
                           const std::vector<Pic>& want) {
    size_t from = 0;
    while (from < have.size() && from + 1 < ids.size() && have[from] == ids[from]) ++from;
    for (const Pic& q : have_pics)
        if (q.start < (long long) from &&
            std::none_of(want.begin(), want.end(), [&](const Pic& x) { return same_pic(x, q); }))
            from = (size_t) q.start;
    for (const Pic& x : want)
        if (x.start < (long long) from &&
            std::none_of(have_pics.begin(), have_pics.end(), [&](const Pic& q) { return same_pic(x, q); }))
            from = (size_t) x.start;
    return from;
}

inline size_t usable(size_t c, int64_t n, int64_t lo, const Geometry& geo) {
    return c > 0 && ring_holds((int64_t) c, n, lo, geo.rows, geo.back) ? c : 0;
}

struct Slot {
    std::vector<int> tokens;
    std::vector<Pic> pics;
    int64_t lo = 0;
    uint64_t used = 0;
    std::shared_ptr<std::vector<uint8_t>> data;
    uint64_t size = 0;   // (its bytes when data is null: tests)
    // (v2: the folder)
    std::string file;          // its files' path stem (empty: none)
    int disk = 0;              // 0 no file, 1 being written, 2 written
    uint64_t file_bytes = 0;   // its .bin's bytes
    uint64_t sum = 0;          // its .bin's checksum (disk 2)
    uint64_t id = 0;           // (the writer's job)
    bool in_ram() const { return data != nullptr || file.empty(); }
    uint64_t bytes() const { return data ? (uint64_t) data->size() : file.empty() ? size : 0; }
};

struct Match {
    static constexpr size_t npos = (size_t) -1;
    size_t index = npos;
    size_t len = 0;
};

struct Put {
    size_t superseded = 0, evicted = 0;
    size_t spilled = 0;          // (v2: in RAM no longer, the folder's)
    std::vector<Slot> gone;      // (v2: the slots dropped - their files to remove, their writes to stop)
};

class Table {
public:
    Table() = default;
    explicit Table(const Knobs& k) : k_(k) {}
    const Knobs& knobs() const { return k_; }
    size_t size() const { return s_.size(); }
    const Slot& operator[](size_t i) const { return s_[i]; }
    uint64_t ram_bytes() const {
        uint64_t t = 0;
        for (const Slot& s : s_) t += s.bytes();
        return t;
    }
    size_t ram_count() const {
        size_t n = 0;
        for (const Slot& s : s_) n += s.in_ram();
        return n;
    }
    uint64_t disk_bytes() const {
        uint64_t t = 0;
        for (const Slot& s : s_)
            if (s.disk != 0) t += s.file_bytes;
        return t;
    }
    Slot& slot(size_t i) { return s_[i]; }
    size_t find_id(uint64_t id) const {
        for (size_t i = 0; i < s_.size(); ++i)
            if (id != 0 && s_[i].id == id) return i;
        return Match::npos;
    }
    // a slot holding exactly this state (its tokens, its pictures)
    size_t live(const std::vector<int>& tokens, const std::vector<Pic>& pics) const {
        for (size_t i = s_.size(); i-- > 0;) {
            const Slot& s = s_[i];
            if (s.tokens != tokens || s.pics.size() != pics.size()) continue;
            bool same = true;
            for (const Pic& q : pics)
                same = same && std::any_of(s.pics.begin(), s.pics.end(), [&](const Pic& x) { return same_pic(x, q); });
            if (same) return i;
        }
        return Match::npos;
    }
    void add(Slot slot) {
        slot.used = ++clock_;
        s_.push_back(std::move(slot));
    }
    std::vector<std::string> trim_folder(uint64_t cap, uint64_t keep_id, uint64_t keep2 = 0) {
        std::vector<std::string> files;
        while (disk_bytes() > cap) {
            size_t lru = Match::npos;
            for (size_t i = 0; i < s_.size(); ++i)
                if (s_[i].disk == 2 && !(keep_id != 0 && s_[i].id == keep_id) && !(keep2 != 0 && s_[i].id == keep2) &&
                    (lru == Match::npos || s_[i].used < s_[lru].used))
                    lru = i;
            if (lru == Match::npos) break;
            files.push_back(s_[lru].file);
            if (s_[lru].data) {
                s_[lru].file.clear();
                s_[lru].disk = 0;
                s_[lru].file_bytes = 0;
            } else {
                drop(lru);
            }
        }
        return files;
    }
    size_t find(const std::vector<uint8_t>* data) const {
        for (size_t i = 0; i < s_.size(); ++i)
            if (data != nullptr && s_[i].data.get() == data) return i;
        return Match::npos;
    }
    Match best(const std::vector<int>& ids, const std::vector<Pic>& want, const Geometry& geo) const {
        Match m;
        for (size_t i = 0; i < s_.size(); ++i) {
            const Slot& s = s_[i];
            if (!s.data && s.disk == 1) continue;   // (its bytes only in a write still running)
            const size_t c = shared_start(s.tokens, s.pics, ids, want);
            const size_t u = usable(c, (int64_t) s.tokens.size(), s.lo, geo);
            if (u == 0) continue;
            if (u > m.len || (u == m.len && s.used > s_[m.index].used)) {
                m.index = i;
                m.len = u;
            }
        }
        return m;
    }
    void touch(size_t i) { s_[i].used = ++clock_; }
    void drop(size_t i) { s_.erase(s_.begin() + (long) i); }
    Put put(Slot slot, const std::vector<uint8_t>* protect = nullptr) {
        Put r;
        const auto earlier = [&](const Slot& o) {   // (an earlier state of this conversation)
            if (o.tokens.size() > slot.tokens.size() || !std::equal(o.tokens.begin(), o.tokens.end(), slot.tokens.begin()))
                return false;
            for (const Pic& q : o.pics)
                if (std::none_of(slot.pics.begin(), slot.pics.end(), [&](const Pic& x) { return same_pic(x, q); }))
                    return false;
            for (const Pic& x : slot.pics)   // (the new one's pictures inside the old one's tokens: the old one's too)
                if (x.start < (long long) o.tokens.size() &&
                    std::none_of(o.pics.begin(), o.pics.end(), [&](const Pic& q) { return same_pic(x, q); }))
                    return false;
            return true;
        };
        for (size_t i = s_.size(); i-- > 0;)
            if (!(protect != nullptr && s_[i].data.get() == protect) && earlier(s_[i])) {
                r.gone.push_back(std::move(s_[i]));
                drop(i);
                ++r.superseded;
            }
        slot.used = ++clock_;
        s_.push_back(std::move(slot));
        const size_t cap = (size_t) std::max(0, k_.cap);
        for (;;) {   // (the caps over the slots in RAM; v1's tables have no files, so every slot is)
            const size_t n_ram = ram_count();
            const bool over = n_ram > cap || (k_.ram_bytes > 0 && n_ram > 1 && ram_bytes() > k_.ram_bytes);
            if (!over) break;
            size_t lru = Match::npos;
            for (size_t i = 0; i + 1 < s_.size(); ++i)   // (never the new one: the last)
                if (s_[i].in_ram() && !(protect != nullptr && s_[i].data.get() == protect) &&
                    (lru == Match::npos || s_[i].used < s_[lru].used))
                    lru = i;
            if (lru == Match::npos) break;
            if (!s_[lru].file.empty()) {   // (its file written or being written: the folder's from now on)
                s_[lru].data.reset();
                ++r.spilled;
                continue;
            }
            r.gone.push_back(std::move(s_[lru]));
            drop(lru);
            ++r.evicted;
        }
        return r;
    }

private:
    Knobs k_;
    std::vector<Slot> s_;
    uint64_t clock_ = 0;
};

struct Plan {
    bool park = false, take = false;
};

inline bool take_pays(size_t from_cur, const Match& best, const Knobs& k) {
    return best.index != Match::npos && best.len > from_cur + k.gain;
}

inline Plan decide(size_t state_n, size_t from_cur, bool held, const Match& best, const Knobs& k) {
    Plan p;
    if (!k.on()) return p;
    p.take = take_pays(from_cur, best, k);
    // (a plain continuation that takes a longer slot: the state is a prefix of that slot's - nothing to set aside)
    const bool plain = state_n > 0 && from_cur == state_n;
    p.park = state_n >= k.min && !held && from_cur <= state_n && !(p.take && plain) &&
             (p.take || state_n - from_cur >= k.min);
    return p;
}

// ---- v2: the folder's bytes (pure: no file is opened here)

inline uint64_t checksum(const uint8_t* p, size_t n, uint64_t seed = 0) {
    constexpr uint64_t P1 = 0x9E3779B185EBCA87ull, P2 = 0xC2B2AE3D27D4EB4Full, P3 = 0x165667B19E3779F9ull;
    const auto rotl = [](uint64_t x, int r) { return (x << r) | (x >> (64 - r)); };
    const auto rd = [&](size_t k) {
        uint64_t w;
        std::memcpy(&w, p + k, 8);
        return w;
    };
    const auto round = [&](uint64_t h, uint64_t w) { return rotl(h + w * P2, 31) * P1; };
    uint64_t a = seed + P1 + P2, b = seed + P2, c = seed, d = seed - P1;
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        a = round(a, rd(i));
        b = round(b, rd(i + 8));
        c = round(c, rd(i + 16));
        d = round(d, rd(i + 24));
    }
    uint64_t h = rotl(a, 1) + rotl(b, 7) + rotl(c, 12) + rotl(d, 18) + (uint64_t) n;
    for (; i + 8 <= n; i += 8) h = rotl(h ^ round(0, rd(i)), 27) * P1 + P3;
    for (; i < n; ++i) h = rotl(h ^ (p[i] * P3), 11) * P1;
    h ^= h >> 33;
    h *= P2;
    h ^= h >> 29;
    h *= P3;
    h ^= h >> 32;
    return h;
}

struct Meta {
    std::string identity;
    uint64_t bin_bytes = 0, checksum = 0;
    int64_t n = 0, lo = 0;
    std::vector<int> tokens;
    std::vector<Pic> pics;
};
constexpr uint64_t kMetaMagic = 0x324154454d345948ull;   // "HY4META2" (little endian; "HY4META1" files are refused)

// CRC-32 as zlib's crc32 / Python's zlib.crc32 (IEEE 802.3, reflected 0xEDB88320, init and final xor 0xFFFFFFFF)
inline uint32_t crc32(const uint8_t* p, size_t n, uint32_t crc = 0) {
    static const auto table = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[i] = c;
        }
        return t;
    }();
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) crc = table[(crc ^ p[i]) & 0xff] ^ (crc >> 8);
    return ~crc;
}

inline std::vector<uint8_t> meta_encode(const Meta& m) {
    std::vector<uint8_t> b;
    const auto put = [&](const void* p, size_t k) {
        const uint8_t* q = (const uint8_t*) p;
        b.insert(b.end(), q, q + k);
    };
    const auto u64 = [&](uint64_t v) { put(&v, 8); };
    u64(kMetaMagic);
    u64(m.identity.size());
    put(m.identity.data(), m.identity.size());
    u64(m.bin_bytes);
    u64(m.checksum);
    u64((uint64_t) m.n);
    u64((uint64_t) m.lo);
    u64(m.tokens.size());
    for (int t : m.tokens) {
        const int32_t v = (int32_t) t;
        put(&v, 4);
    }
    u64(m.pics.size());
    for (const Pic& q : m.pics) {
        const int64_t st = (int64_t) q.start;
        put(&st, 8);
        u64(q.hash);
    }
    u64((uint64_t) crc32(b.data(), b.size()));   // (the bytes so far: their CRC-32 in the low half)
    u64(kMetaMagic);
    return b;
}

inline bool meta_decode(const uint8_t* p, size_t n, Meta& m, std::string& why) {
    size_t off = 0;
    const auto get = [&](void* d, size_t k) {
        if (k > n || off > n - k) return false;
        std::memcpy(d, p + off, k);
        off += k;
        return true;
    };
    uint64_t v = 0;
    const auto u64 = [&](uint64_t& x) { return get(&x, 8); };
    if (p == nullptr || !u64(v) || v != kMetaMagic) {
        why = "not a slot's .meta";
        return false;
    }
    uint64_t il = 0;
    if (!u64(il) || il > n) {
        why = "the .meta is cut short (identity)";
        return false;
    }
    m.identity.assign(il, '\0');
    uint64_t nn = 0, lo = 0, nt = 0, np = 0;
    if (!get(m.identity.data(), il) || !u64(m.bin_bytes) || !u64(m.checksum) || !u64(nn) || !u64(lo) || !u64(nt)) {
        why = "the .meta is cut short";
        return false;
    }
    if (nt == 0 || nt != nn || nt > (n - off) / 4 || lo > nn) {
        why = "the .meta's positions are not valid";
        return false;
    }
    m.n = (int64_t) nn;
    m.lo = (int64_t) lo;
    m.tokens.resize((size_t) nt);
    for (size_t i = 0; i < nt; ++i) {
        int32_t t = 0;
        get(&t, 4);
        m.tokens[i] = (int) t;
    }
    if (!u64(np) || np > (n - off) / 16) {
        why = "the .meta's pictures are cut short";
        return false;
    }
    m.pics.resize((size_t) np);
    for (Pic& q : m.pics) {
        int64_t st = 0;
        get(&st, 8);
        u64(q.hash);
        q.start = (long long) st;
        if (st < 0 || st >= m.n) {
            why = "a picture of the .meta is outside its tokens";
            return false;
        }
    }
    const size_t crc_at = off;
    uint64_t crc = 0;
    if (!u64(crc)) {
        why = "the .meta is cut short (its checksum)";
        return false;
    }
    if (crc != (uint64_t) crc32(p, crc_at)) {
        why = "the .meta's checksum is not its bytes' (damaged)";
        return false;
    }
    if (!u64(v) || v != kMetaMagic || off != n) {
        why = "the .meta's end is not there";
        return false;
    }
    if (m.bin_bytes == 0) {
        why = "the .meta names an empty .bin";
        return false;
    }
    return true;
}

enum FileKind { kOther = 0, kBin, kMeta, kTmp };
inline FileKind file_kind(const std::string& name) {
    const auto ends = [&](const char* x) {
        const size_t k = std::strlen(x);
        return name.size() > k && name.compare(name.size() - k, k, x) == 0;
    };
    if (name.rfind("slot-", 0) != 0) return kOther;
    if (ends(".tmp")) return kTmp;
    if (ends(".bin")) return kBin;
    if (ends(".meta")) return kMeta;
    return kOther;
}

}  // namespace hy4::slots
