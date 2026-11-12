// include/strata/core/glm_req_knobs.hpp - glm-dsa (speed plan, code change 2): the decode settings one request may
// change without a restart.
//
// Each knob is a switch the engine read once at start.  serve/server.py's `strata_tune` (the keys
// serve/glm_dsa_tune.py lets through) puts them on the GEN line; generate.cpp hands every key of the line to
// Request::take and calls apply() before the request runs.  A knob holds the value the last request named, or kStart:
// each reader keeps its own read-once parse of the switch and passes it to get(), so a request that names no knob runs
// exactly as the engine started, and a request that leaves a key out puts that knob back to its start value (as
// pcie_frac does: Glm5Model::set_pcie_share(-1)).
//
//   GEN key          start value                              a request's values
//   promote_min      MAYA_GLM_PROMOTE_MIN                   0..65535 (0 / 1: keep any fetched expert a spare fits)
//   prefetch_n       MAYA_GLM_PREFETCH_N                    0..2 (kSpares - 1); above 0 the decode graphs stay off
//   prefetch_at      MAYA_GLM_PREFETCH_AT                   route | fetch | cpu
//   prefetch_rank    MAYA_GLM_PREFETCH_RANK                 0..64 (0: any rank of the prediction)
//   prefetch_blocks  MAYA_GLM_PREFETCH_BLOCKS               0..1024 (0: half the SMs)
//   mtp_keep         MAYA_GLM_MTP_KEEP (or _MTP_MISS)       0..8
//   cpu_split        MAYA_GLM_CPU_SPLIT                     0..64 (0: about 48 pieces in all)
//   vram_evict       MAYA_GLM_VRAM_EVICT                    lfu | lru
//   ngram            MAYA_GLM_NGRAM_DRAFT                   0 | 1
//   lane_row4        MAYA_GLM_LANE_ROW4                     0 | 1 (kernels::cpu::lane_row4_override; -1 the switch)
//   graphs           (MAYA_GLM_DSA_GRAPHS as started)       0: this request decodes eagerly (glmdsa::host_off_reason)
//   pcie_frac_head   pcie_frac, else the start plan           0..1: the head parts' CPU-lane share (Glm5Model::
//                                                             set_pcie_share_head; the other parts keep pcie_frac)
//
// promote_min is an argument of the route kernel, so a captured graph holds it: a change bumps the glm-dsa decode
// graphs' generation (apply), and Flash's KDA layer graphs (MAYA_GLM_KDA_GRAPH, keyed without one) stay off while a
// request names it.  The other knobs reach no captured graph: prefetch_n > 0 keeps both kinds off (prefetch_at /
// _rank / _blocks act only then), and the rest act outside them (the boundary, the CPU lane, the NextN draft's route,
// the n-gram drafts).
//
// The values are process-wide relaxed atomics (every part of a split reads the same one), set between requests; with
// several conversations at once (MAYA_GLM_SEQS) a GEN line sets them for every running request, as pcie_frac does.
// The store and the parser are header-only (the readers live in strata_engine, strata_kernels' moe_prefetch and the
// n-gram draft's CPU tests); apply() is in src/core/glm_req_knobs.cpp.
#pragma once

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace strata::core {
class Glm5Model;
}

namespace strata::core::glmknob {

/// The knobs a reader asks for (lane_row4 and pcie_frac_head have their own setters).
enum Id : int {
    PromoteMin,
    PrefetchN,
    PrefetchAt,       // 0 route, 1 fetch, 2 cpu
    PrefetchRank,
    PrefetchBlocks,
    MtpKeep,
    CpuSplit,
    VramLru,          // 0 lfu, 1 lru
    Ngram,
    GraphsOff,        // 1: graphs=0 on the request
    kCount
};

/// No request value: the reader's start value holds.
constexpr int kStart = -1;

struct Slot {
    std::atomic<int> v{kStart};
};
inline Slot* slots() {
    static Slot s[kCount];
    return s;
}

/// The knob's value: the last request's, else `start` (the reader's read-once parse of its switch).
inline int get(Id k, int start) {
    const int v = slots()[k].v.load(std::memory_order_relaxed);
    return v == kStart ? start : v;
}
/// The request's value (kStart: none) - apply() and the tests.
inline int raw(Id k) { return slots()[k].v.load(std::memory_order_relaxed); }
inline void set(Id k, int v) { slots()[k].v.store(v < 0 ? kStart : v, std::memory_order_relaxed); }

/// The GEN key of a knob.
inline const char* key_name(Id k) {
    switch (k) {
    case PromoteMin: return "promote_min";
    case PrefetchN: return "prefetch_n";
    case PrefetchAt: return "prefetch_at";
    case PrefetchRank: return "prefetch_rank";
    case PrefetchBlocks: return "prefetch_blocks";
    case MtpKeep: return "mtp_keep";
    case CpuSplit: return "cpu_split";
    case VramLru: return "vram_evict";
    case Ngram: return "ngram";
    case GraphsOff: return "graphs";
    default: return "?";
    }
}

/// A whole decimal integer in [lo, hi] (false for anything else).
inline bool parse_int(const std::string& s, int lo, int hi, int& out) {
    if (s.empty()) return false;
    char* end = nullptr;
    errno = 0;
    const long x = std::strtol(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0' || errno == ERANGE || x < lo || x > hi) return false;
    out = (int) x;
    return true;
}

/// One GEN line's knobs (kStart / -1: the line does not name it).
struct Request {
    int v[kCount];
    int lane_row4 = -1;
    double pcie_frac_head = -1.0;
    Request() {
        for (int k = 0; k < kCount; ++k) v[k] = kStart;
    }
    int value(Id k) const { return v[k]; }
    bool named() const {
        for (int k = 0; k < kCount; ++k)
            if (v[k] != kStart) return true;
        return lane_row4 >= 0 || pcie_frac_head >= 0.0;
    }
    /// The GEN key `k` with value `val`: true when it is a knob's key (a value out of its range is reported on stderr
    /// and left out, so that knob keeps its start value); false for every other key.
    bool take(const std::string& k, const std::string& val) {
        int x = kStart;
        bool ok = false;
        if (k == "prefetch_at") {
            ok = val == "route" || val == "fetch" || val == "cpu";
            x = val == "fetch" ? 1 : val == "cpu" ? 2 : 0;
            if (ok) v[PrefetchAt] = x;
        } else if (k == "vram_evict") {
            ok = val == "lfu" || val == "lru";
            if (ok) v[VramLru] = val == "lru" ? 1 : 0;
        } else if (k == "graphs") {
            ok = parse_int(val, 0, 1, x);
            if (ok) v[GraphsOff] = x == 0 ? 1 : 0;
        } else if (k == "lane_row4") {
            ok = parse_int(val, 0, 1, x);
            if (ok) lane_row4 = x;
        } else if (k == "pcie_frac_head") {
            char* end = nullptr;
            const double s = std::strtod(val.c_str(), &end);
            ok = !val.empty() && *end == '\0' && s >= 0.0 && s <= 1.0;
            if (ok) pcie_frac_head = s;
        } else {
            // the numeric knobs: key, knob, largest value (the smallest is 0)
            static const char* const keys[] = {"promote_min", "prefetch_n", "prefetch_rank", "prefetch_blocks",
                                               "mtp_keep", "cpu_split", "ngram"};
            static const Id ids[] = {PromoteMin, PrefetchN, PrefetchRank, PrefetchBlocks, MtpKeep, CpuSplit, Ngram};
            static const int his[] = {65535, 2, 64, 1024, 8, 64, 1};
            const int n = (int) (sizeof ids / sizeof ids[0]);
            int i = 0;
            while (i < n && k != keys[i]) ++i;
            if (i == n) return false;
            ok = parse_int(val, 0, his[i], x);
            if (ok) v[ids[i]] = x;
        }
        if (!ok)
            std::fprintf(stderr, "glm knobs: %s=%s is outside the key's values - the start value kept\n", k.c_str(),
                         val.c_str());
        return true;
    }
    /// The knobs it names, as GEN keys ("" when none).
    std::string line() const {
        std::string s;
        char buf[64];
        for (int k = 0; k < kCount; ++k) {
            if (v[k] == kStart) continue;
            const Id id = (Id) k;
            if (id == PrefetchAt)
                std::snprintf(buf, sizeof buf, " prefetch_at=%s", v[k] == 1 ? "fetch" : v[k] == 2 ? "cpu" : "route");
            else if (id == VramLru)
                std::snprintf(buf, sizeof buf, " vram_evict=%s", v[k] == 1 ? "lru" : "lfu");
            else if (id == GraphsOff)
                std::snprintf(buf, sizeof buf, " graphs=%d", v[k] == 1 ? 0 : 1);
            else
                std::snprintf(buf, sizeof buf, " %s=%d", key_name(id), v[k]);
            s += buf;
        }
        if (lane_row4 >= 0) {
            std::snprintf(buf, sizeof buf, " lane_row4=%d", lane_row4);
            s += buf;
        }
        if (pcie_frac_head >= 0.0) {
            std::snprintf(buf, sizeof buf, " pcie_frac_head=%.3f", pcie_frac_head);
            s += buf;
        }
        return s.empty() ? s : s.substr(1);
    }
};

/// Every knob from r (the ones it does not name back to their start values), the lane's row4 switch
/// (kernels::cpu::lane_row4_override, called when the value changes; a build without native experts has no row4 lane
/// and says so) and the head parts' CPU-lane share (Glm5Model::set_pcie_share_head; call it after set_pcie_share), and
/// a stderr line naming what the request changed.  src/core/glm_req_knobs.cpp (strata_engine).
void apply(Glm5Model& model, const Request& r);

}  // namespace strata::core::glmknob
