// src/ds4/ds4_pcie.hpp - the decode path's PCIe lane for maya-ds4 (the host side's arithmetic; ds4_model.cu runs it).
//
// A decoded token's routed experts that VRAM does not hold were all computed by the CPU lane.  As GLM's maya does
// (glm_fast_path.cu's lane_plan), some of them can instead be read into pinned memory, copied over PCIe into a few
// staging slots of the GPU and computed there by the same expert kernels as the VRAM ones - while the CPU lane computes
// the rest - so the two lanes finish together.  Which ones is decided per request from this PC's measured costs: the
// CPU lane's time an expert of the layer's format (at its thread count), the stream lane's (the read into pinned
// memory and the copy), the GPU's time an expert after the wait, and a read from the disk for experts not in RAM.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace strata::ds4::pcie {

/// One format class's measured times on this PC (ms): the CPU lane's an expert and a call; the stream lane's an expert
/// (its read into pinned memory and its copy to the GPU) and a request (the wait for the copies); the GPU's an expert
/// streamed (its gate / up and down after the wait); a read of one expert from the disk (not in RAM).
struct Costs {
    double c = 0, oc = 0, p = 0, op = 0, g = 0, disk = 0;
    bool ok() const { return c > 0 && p > 0; }
};

/// The time a request takes with the picks in `mask` streamed and the rest on the CPU lane: the slower lane, then the
/// streamed experts on the GPU.  Bit i of `cold`: pick i is not in RAM (both lanes read it from the disk first; the
/// stream lane's copy of it overlaps the read but the last part).
inline double split_time(const Costs& k, int n, unsigned cold, unsigned mask) {
    double tc = 0, tp = 0;
    int s = 0;
    for (int i = 0; i < n; ++i) {
        const bool cd = ((cold >> i) & 1u) != 0;
        if ((mask >> i) & 1u) {
            tp += cd ? k.disk + k.p / 3.0 : k.p;
            ++s;
        } else {
            tc += k.c + (cd ? k.disk : 0.0);
        }
    }
    if (s < n) tc += k.oc;
    if (s > 0) tp += k.op;
    return std::fmax(tc, tp) + k.g * s;
}

/// The split of n picks (n <= 8): the set of them the stream lane takes - at most max_s (its staging slots) - that
/// minimises split_time; a tie keeps the fewer streamed (then the lower mask).  Returns the mask; *t its time.
inline unsigned split(const Costs& k, int n, unsigned cold, int max_s, double* t = nullptr) {
    unsigned best = 0;
    double bt = split_time(k, n, cold, 0u);
    int bs = 0;
    if (k.ok() && max_s > 0)
        for (unsigned m = 1; m < (1u << n); ++m) {
            int s = 0;
            for (unsigned v = m; v != 0; v &= v - 1) ++s;
            if (s > max_s) continue;
            const double tm = split_time(k, n, cold, m);
            if (tm < bt - 1e-9 || (tm <= bt + 1e-9 && s < bs)) {
                best = m;
                bt = tm;
                bs = s;
            }
        }
    if (t != nullptr) *t = bt;
    return best;
}

/// How many of n warm picks the split streams, for n = 0..8 ("0 0 1 1 2 2 3 3 4": the log's table)
inline std::string table(const Costs& k, int max_s, int n_max = 8) {
    std::string s;
    for (int n = 0; n <= n_max; ++n) {
        const unsigned m = split(k, n, 0u, max_s);
        int c = 0;
        for (unsigned v = m; v != 0; v &= v - 1) ++c;
        s += (n ? " " : "") + std::to_string(c);
    }
    return s;
}

/// Whether the split streams anything for some n <= n_max picks, all of them warm or all cold
inline bool streams_any(const Costs& k, int max_s, int n_max) {
    for (int n = 1; n <= n_max && n <= 8; ++n)
        if (split(k, n, 0u, max_s) != 0u || split(k, n, (1u << n) - 1u, max_s) != 0u) return true;
    return false;
}

/// The pre-placement allocation always follows the persisted baseline decision; experimental ZC settings cannot
/// override it and thereby change expert placement. Forced routing keeps its existing explicit override.
inline bool baseline_lane_off(const std::string& forced, const std::string& persisted) {
    return forced.empty() && persisted == "off";
}

/// ZC-all becomes usable only after the copies and their pinning actually exist. An empty/all-zero slot table, an
/// explicit forced route, and the hold-for-tests mode are all inert, including for the subordinate no-wait setting.
inline bool zc_all_usable(bool requested, bool forced, bool hold, bool zc, const std::vector<char>& pinned_slots) {
    return requested && !forced && !hold && zc &&
           std::any_of(pinned_slots.begin(), pinned_slots.end(), [](char v) { return v != 0; });
}

/// The times file (DS4_LANE_CAL, by default <pack>/lane-times.txt): a line a key, "key<TAB>values".  get: the values
/// of `key` (empty: none); put: key's line replaced by vals (empty vals: dropped), through a temporary file.
inline std::string cal_get(const std::string& file, const std::string& key) {
    std::string out;
    if (FILE* f = std::fopen(file.c_str(), "r")) {
        char line[4096];
        while (std::fgets(line, sizeof line, f)) {
            const char* tab = std::strrchr(line, '\t');
            if (tab == nullptr || std::string(line, (size_t) (tab - line)) != key) continue;
            out = tab + 1;
            while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
        }
        std::fclose(f);
    }
    return out;
}
template <class Replace>
inline void cal_put(const std::string& file, const std::string& key, const std::string& vals, Replace replace) {
    std::string keep;
    if (FILE* f = std::fopen(file.c_str(), "r")) {
        char line[4096];
        while (std::fgets(line, sizeof line, f)) {
            const char* tab = std::strrchr(line, '\t');
            if (tab == nullptr || std::string(line, (size_t) (tab - line)) != key) keep += line;
        }
        std::fclose(f);
    }
    if (!vals.empty()) keep += key + "\t" + vals + "\n";
    const std::string tmp = file + ".tmp";
    if (FILE* f = std::fopen(tmp.c_str(), "w")) {
        const bool ok = std::fputs(keep.c_str(), f) >= 0;
        if (std::fclose(f) == 0 && ok && replace(tmp, file)) return;
        std::remove(tmp.c_str());
    }
}

}  // namespace strata::ds4::pcie
