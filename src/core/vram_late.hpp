// (#1026) The VRAM reserve held for the whole run - shared by the engines (maya-ds4, maya's GLM path, maya-mm3,
// maya-hy4).
//
// Each engine sizes what it keeps in VRAM (the experts' plan, the expert pool, the residency) once, from the free VRAM
// less the reserve - but the start allocates more after that (the prompt path's workspaces, graphs, the CPU lane's
// scratch, a request's buffers ...), and that came out of the reserve: on a small card the reserve was gone by the
// first request, whatever its size.  So what a PC's start and requests take after the plan is measured (check: the
// least free VRAM once loaded and between requests, against the reserve), kept per GPU, pack and settings in the
// pack's vram-late.txt ("key<TAB>MiB" lines), and kept free beside the reserve from the next start on.  With the
// reserve not asked for, the last release's plan stays wherever the default reserve holds the measured figure with
// 128 MiB to spare (kept_for).  A start found short says so (Tracker::short_mib: the engine's INFO reserve_short=,
// on which the server starts it once more).
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace maya::vramlate {

constexpr int64_t kStep = (int64_t) 32 << 20;      // the figure moves in steps of 32 MiB
constexpr size_t kSpare = (size_t) 128 << 20;      // the default reserve keeps the plan while it holds this beyond it

/// the file's figure for `key` ("": none); the file: a line a key, "key<TAB>value"
inline std::string get(const std::string& file, const std::string& key) {
    std::string out;
    if (FILE* f = std::fopen(file.c_str(), "r")) {
        char line[4096];
        while (std::fgets(line, sizeof line, f)) {
            std::string l(line);
            const size_t tab = l.rfind('\t');
            if (tab == std::string::npos || l.compare(0, tab, key) != 0 || tab != key.size()) continue;
            out = l.substr(tab + 1);
            while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
        }
        std::fclose(f);
    }
    return out;
}
/// key's line replaced by value (through a temporary file renamed over it)
inline bool put(const std::string& file, const std::string& key, const std::string& value) {
    std::string keep;
    if (FILE* f = std::fopen(file.c_str(), "r")) {
        char line[4096];
        while (std::fgets(line, sizeof line, f)) {
            const std::string l(line);
            const size_t tab = l.rfind('\t');
            if (tab == std::string::npos || tab != key.size() || l.compare(0, tab, key) != 0) keep += l;
        }
        std::fclose(f);
    }
    if (!value.empty()) keep += key + "\t" + value + "\n";
    const std::string tmp = file + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "w");
    if (f == nullptr) return false;
    const bool ok = std::fputs(keep.c_str(), f) >= 0;
    if (std::fclose(f) != 0 || !ok) {
        std::remove(tmp.c_str());
        return false;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, file, ec);
    if (ec) std::remove(tmp.c_str());
    return !ec;
}

/// the file a pack keeps the figures in: `env` (a path, or "none": not kept), else <pack>/vram-late.txt
inline std::string file_of(const std::string& pack, const char* env) {
    const char* f = env != nullptr ? std::getenv(env) : nullptr;
    if (f == nullptr) f = std::getenv("MAYA_VRAM_LATE_FILE");
    if (f != nullptr) return std::string(f) == "none" ? std::string() : std::string(f);
    return pack.empty() ? std::string() : pack + "/vram-late.txt";
}
/// a fixed figure in MiB (`env`, else MAYA_VRAM_LATE_MB; 0: the reserve alone), -1: measured
inline int64_t fixed_of(const char* env) {
    const char* v = env != nullptr ? std::getenv(env) : nullptr;
    if (v == nullptr) v = std::getenv("MAYA_VRAM_LATE_MB");
    return v == nullptr ? -1 : std::max<int64_t>(0, std::atoll(v));
}

/// what a start keeps free beside the reserve from the measured figure: all of it with a reserve asked for; with the
/// default one, none (the last release's plan) while the default holds it with kSpare to spare
inline size_t kept_for(size_t measured, size_t reserve_b, bool given) {
    return !given && measured + kSpare <= reserve_b ? 0 : measured;
}

struct Gpu {
    size_t late = 0;              // kept free beside the reserve at this start
    size_t kept = 0;              // the file's figure (what the next start reads)
    size_t fr_min = ~(size_t) 0;  // the least free VRAM this run has seen
};

/// one reading of the free VRAM (`fr` of `tot`) against the reserve `res`: whether the figure moved (then `want`,
/// what the next start keeps beside the reserve: this start's plus what is short, less what is over, at most a
/// quarter of the card - not another program's - in steps of kStep) and whether the reserve was found short
struct Step {
    bool moved = false, short_ = false;
    int64_t want = 0;
};
// `room`: how much more the plan could have kept free at this start (a pool already at its floor: 0) - the figure
// never grows past late + room, and a start that kept all it could is not said short (a restart would not help)
inline Step check(Gpu& g, size_t fr, size_t tot, size_t res, bool given, size_t room = ~(size_t) 0 >> 2) {
    Step s;
    if (fr >= g.fr_min) return s;
    g.fr_min = fr;
    int64_t want = (int64_t) g.late + (int64_t) res - (int64_t) fr;
    want = std::min<int64_t>(std::max<int64_t>(0, want), (int64_t) tot / 4);
    want = (want + kStep - 1) / kStep * kStep;
    const int64_t lim = std::max<int64_t>((int64_t) g.late, ((int64_t) g.late + (int64_t) std::min(room, tot)) / kStep * kStep);
    want = std::min(want, lim);
    if (std::llabs(want - (int64_t) g.kept) < kStep) return s;
    s.moved = true;
    s.want = want;
    s.short_ = (int64_t) fr + 2 * kStep < (int64_t) res && (given || want + 4 * kStep > (int64_t) res) &&
               want > (int64_t) g.late;
    g.kept = (size_t) want;
    return s;
}

/// an engine's GPUs: read at the start (before the plan), checked once loaded and before every request
struct Tracker {
    std::string tag;          // the log lines' prefix ("glm fast", "mm3", "hy4")
    std::string file;         // "": not kept
    int64_t fixed = -1;       // a fixed figure (MiB), -1: measured
    int short_mib = 0;        // > 0: this start came out short of the reserve; the next keeps this many MiB beside it
    struct G {
        Gpu g;
        std::string key;
        size_t res = 0, room = ~(size_t) 0 >> 2;   // (room: what the plan could still have kept free - check)
        bool given = false, read = false;
    };
    std::vector<G> gpus;

    void init(const std::string& tag_, const std::string& pack, const char* file_env, const char* mb_env, size_t n) {
        tag = tag_;
        file = file_of(pack, file_env);
        fixed = fixed_of(mb_env);
        gpus.assign(n, G{});
    }
    /// GPU i's figure (bytes) to keep free beside its reserve `res` (`given`: asked for); `key`: its GPU, pack and
    /// settings; `dev`: its name in the log
    size_t read(size_t i, const std::string& key, size_t res, bool given, int dev) {
        if (i >= gpus.size()) return 0;
        G& x = gpus[i];
        x.key = key, x.res = res, x.given = given, x.read = true;
        if (fixed >= 0) {
            x.g.late = (size_t) fixed << 20;
        } else if (!file.empty()) {
            const std::string v = get(file, key);
            x.g.kept = v.empty() ? 0 : (size_t) std::max<long long>(0, std::atoll(v.c_str())) << 20;
            x.g.late = kept_for(x.g.kept, res, given);
        }
        if (x.g.late > 0)
            std::fprintf(stderr, "%s: CUDA%d keeps %zu MiB free beside the %zu MiB reserve: what this PC's start and "
                                 "requests allocate after the memory plan%s\n", tag.c_str(), dev, x.g.late >> 20,
                         res >> 20, fixed >= 0 ? " (MAYA_VRAM_LATE_MB)" : " (measured before)");
        return x.g.late;
    }
    /// the reserve GPU i's plan ended up keeping (a start that lowered it on a small card)
    void set_reserve(size_t i, size_t res) {
        if (i < gpus.size()) gpus[i].res = res;
    }
    /// how much more GPU i's plan could have kept free at this start (check's `room`)
    void set_room(size_t i, size_t room) {
        if (i < gpus.size()) gpus[i].room = room;
    }
    /// GPU i's free VRAM now (`loaded`: the start's last step, else before a request)
    void check(size_t i, size_t fr, size_t tot, bool loaded, int dev) {
        if (i >= gpus.size() || !gpus[i].read) return;
        G& x = gpus[i];
        if (loaded)
            std::fprintf(stderr, "%s: CUDA%d: %zu MiB of VRAM free with everything loaded (the reserve %zu MiB%s)\n",
                         tag.c_str(), dev, fr >> 20, x.res >> 20,
                         x.g.late > 0 ? (", " + std::to_string(x.g.late >> 20) + " MiB more kept beside it").c_str() : "");
        if (fixed >= 0) return;
        const Step s = vramlate::check(x.g, fr, tot, x.res, x.given, x.room);
        if (!s.moved) return;
        if (s.short_)
            std::fprintf(stderr, "%s: WARNING - CUDA%d: %zu MiB free, short of the %zu MiB reserve - the next start keeps "
                                 "%lld MiB free beside it for what the start and the requests allocate after the memory "
                                 "plan%s\n", tag.c_str(), dev, fr >> 20, x.res >> 20, (long long) (s.want >> 20),
                         file.empty() ? " (not kept: MAYA_VRAM_LATE_FILE=none)" : "");
        if (loaded && s.short_ && !file.empty()) short_mib = std::max(short_mib, (int) (s.want >> 20));
        if (!file.empty()) put(file, x.key, std::to_string(s.want >> 20));
    }
};

}  // namespace maya::vramlate
