// src/ds4/ds4_main.cpp - maya-ds4: DeepSeek-V4.1-Flash on Maya.
//
//   maya-ds4 --pack <model>/maya-ds4 --ids 0,128803,... --n 16          greedy tokens after the ids
//   maya-ds4 --pack ... --ids ... --dump-logits out.f32                every position's logits (the oracle's check)
//   maya-ds4 --pack ... --serve                                       Maya's engine protocol on stdin / stdout
//
// --serve speaks what serve/server.py's StrataEngine expects: INFO lines, then `READY <context> stop`; per request
// `GEN <max_new> [key=value ...] <id,id,...>` answered by `T <id>` lines and `DONE <generated> <prompt> <prompt ms>
// <decode ms> <stop|length|cancel> <accepted> <offered> <reused> <hits> <lookups>` (the drafts; the decode's routed
// experts in VRAM / all); `STOP` ends a request early, `QUIT` the engine.  A prompt that
// continues the state (the conversation so far) reuses it; one that only shares the last prompt's start goes back to
// that prompt's end; anything else starts over.  `GENI <max_new> [keys] <file> <ids>` carries the prompt's pictures
// (the file: see read_pics); the state is shared only where the same picture sits at the same place.  With
// MAYA_DS4_SLOT_KEEP=1 the conversations are kept when another one takes the model (a slot each: the state, the
// compressed KV included), and across runs in MAYA_DS4_SLOT_DIR - so a request that does not continue the state but
// does continue a kept conversation reads only its new tokens.  A kept conversation holds its pictures' marks too (a
// span's start, its pixels' fingerprint, its length): it comes back only for a prompt whose pictures are the same at
// the same places.  `marks=<pos>,...` and `shared=<pos>` on a GEN ask for restore points besides the end of the last
// prompt (the server's MAYA_DS4_RESTORE_POINTS: the start of the last user message, a point near the prompt's tail,
// the end of the system prompt and tools): the read stops at each to keep the state there, and a later prompt that
// leaves this one past one of them goes back to it (ds4_marks.hpp).
#include "ds4_interleave.hpp"
#include "ds4_marks.hpp"
#include "ds4_gguf.cuh"       // (--pack-info: the types the kernels take, asked of them)
#include "ds4_iq.cuh"
#include "ds4_model.hpp"
#include "ds4_packinfo.hpp"
#include "ds4_watchdog.hpp"
#include "strata/platform/env_alias.hpp"
#include "strata/spec/draft_policy.hpp"
#include "strata/spec/suffix_drafter.hpp"

#include <cuda_profiler_api.h>

// the commit and the release this binary was built from (--version, the --verbose "ds4: build" line):
// cmake/build_id.cmake writes them at every build
#if __has_include("maya_build_id.h")
#include "maya_build_id.h"
#endif
#ifndef MAYA_BUILD_ID
#define MAYA_BUILD_ID "unknown"
#endif
#ifndef MAYA_RELEASE
#define MAYA_RELEASE "?"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#if defined(_WIN32)
#include <io.h>
#define NOMINMAX
#include <windows.h>
#include <timeapi.h>
#pragma comment(lib, "winmm.lib")
#else
#include <unistd.h>
#include <cerrno>
#if defined(__linux__)
#include <sys/syscall.h>
#endif
#endif

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// the bytes this process had the storage read for it (Linux's /proc/self/io; -1 where there is none): the Monitor's
// disk reads a token, the experts read in place from the model's files included
long long io_read_bytes() {
#ifdef __linux__
    std::FILE* f = std::fopen("/proc/self/io", "r");
    if (f == nullptr) return -1;
    char line[128];
    long long v = -1;
    while (std::fgets(line, sizeof line, f) != nullptr)
        if (std::sscanf(line, "read_bytes: %lld", &v) == 1) break;
    std::fclose(f);
    return v;
#else
    return -1;
#endif
}

std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> v;
    const char* p = s.c_str();
    while (*p) {
        char* e = nullptr;
        const long x = std::strtol(p, &e, 10);
        if (e == p) break;
        v.push_back((int) x);
        p = e;
        while (*p == ',' || *p == ' ') ++p;
    }
    return v;
}

void usage() {
    std::fprintf(stderr,
                 "maya-ds4 --pack <model>/maya-ds4 [--gpus 0,1,...] [--max-context N] [--reserve-mib N]\n"
                 "           [--vram-experts F] [--cpu-threads N] [--expert-profile FILE|none] [--no-expert-cache] [--verbose]\n"
                 "           [--tp N] [--threads on|off] [--prefill-chunk N] [--dspark on|off|none] [--dspark-conf F]\n"
                 "           [--suffix-draft N]\n"
                 "           [--vision on|off] [--mmap-experts on|off|auto] [--parallel N (--serve: N conversations at once)]\n"
                 "           [--ablate DIRECTION.gguf|ADAPTER.gguf [--ablate-layer N] [--ablate-scale F]]\n"
                 "           (--ids I,J,... [--images FILE] [--n N] [--dump-logits FILE] | --ppl FILE | --serve\n"
                 "           | --encode-image FILE.rgb --size WxH --dump-image OUT)\n"
                 "maya-ds4 --pack <model>/maya-ds4 --pack-info [--expert-profile FILE]   the pack described: no GPU, no "
                 "weights read\n"
                 "maya-ds4 --version   the release and the commit it was built from\n"
                 "maya-ds4 --pack ... --profile-run FILE [--profile-out OUT]   an expert profile from a corpus (a line "
                 "of ids each)\n"
                 "           [--profile-readonly] [--profile-out OUT]   any run: the profile never written back; this "
                 "run's counts alone to OUT\n"
                 "maya-ds4 --pack ... --spec-eval FILE [--n N]   the drafter's acceptance over prompts, every draft "
                 "verified (DS4_ROUND_LOG=<file>: a line a round)\n"
                 "maya-ds4 --pack ... --sampler-test N --ids I,J,...   speculative sampling against the model's own "
                 "draws\n");
}

// --pack-info: the types the kernels take, asked of them (ds4_gguf / ds4_iq: the routed experts' and the dense GEMVs'
// ggml types; the EXL3 kernels' bitrates with the mul1 codebook)
strata::ds4::KernelTypes kernel_types() {
    namespace gg = strata::ds4::gguf;
    strata::ds4::KernelTypes kt;
    kt.name = [](int t) { return gg::type_name(t); };
    kt.expert = [](int t) { return gg::expert_type_ok(t); };
    kt.dense = [](int t) { return t != gg::kTypeMXFP4 && gg::iq::mv_supported(t); };
    kt.exl3_k2 = [](int k2) { return strata::kernels::exl3::k2_supported(k2, 2); };
    return kt;
}

// stdin lines on a thread, so a request can see STOP while it decodes.  read(2), not std::cin - glibc's exit()
// flushes stdio and waits on stdin's lock, which getline holds while it waits for input (the GLM serve's pattern)
struct Lines {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    bool eof = false;
    std::atomic<bool> stop{false};
    // --parallel: `STOP <rid>` ends that request alone (stop_rids), a bare STOP, QUIT or the input's end all of them
    std::vector<long long> stop_rids;
    std::atomic<bool> stop_all{false};
    bool stopped(long long rid) {
        if (stop_all.load()) return true;
        std::lock_guard<std::mutex> lk(mu);
        return std::find(stop_rids.begin(), stop_rids.end(), rid) != stop_rids.end();
    }
    void forget(long long rid) {   // (the request is over: a late STOP of it changes nothing)
        std::lock_guard<std::mutex> lk(mu);
        stop_rids.erase(std::remove(stop_rids.begin(), stop_rids.end(), rid), stop_rids.end());
    }
    // the next line if one is waiting (false: none now) - the several-conversation loop reads between its steps
    bool poll(std::string& l, bool& end) {
        std::lock_guard<std::mutex> lk(mu);
        end = q.empty() && eof;
        if (q.empty()) return false;
        l = q.front();
        q.pop_front();
        return true;
    }
    void start() {
        std::thread([this] {
            std::string l, buf;
            char chunk[4096];
            auto getline_fd = [&](std::string& out) -> bool {
                for (;;) {
                    const size_t nlpos = buf.find('\n');
                    if (nlpos != std::string::npos) {
                        out.assign(buf, 0, nlpos);
                        buf.erase(0, nlpos + 1);
                        return true;
                    }
#if defined(_WIN32)
                    const int n = _read(0, chunk, (unsigned) sizeof chunk);
#else
                    const ssize_t n = ::read(0, chunk, sizeof chunk);
                    if (n < 0 && errno == EINTR) continue;
#endif
                    if (n <= 0) {
                        if (buf.empty()) return false;
                        out.swap(buf);
                        buf.clear();
                        return true;
                    }
                    buf.append(chunk, (size_t) n);
                }
            };
            while (getline_fd(l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                std::lock_guard<std::mutex> lk(mu);
                if (l == "STOP" || l.rfind("STOP ", 0) == 0) {
                    stop.store(true);
                    if (l == "STOP") stop_all.store(true);
                    else stop_rids.push_back(std::atoll(l.c_str() + 5));
                    cv.notify_all();
                    continue;
                }
                if (l == "QUIT") stop.store(true), stop_all.store(true);
                q.push_back(l);
                cv.notify_all();
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            stop.store(true);
            stop_all.store(true);
            cv.notify_all();
        }).detach();
    }
    bool next(std::string& l) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !q.empty() || eof; });
        if (q.empty()) return false;
        l = q.front();
        q.pop_front();
        return true;
    }
};

// DS4_PROF: the GPU's time a decoded token by stage between two snapshots - the CLI's run, or a serve request's
// decode (one line a request whose decode was profiled, so a caller can count them)
void prof_line(const strata::ds4::Stats& s0, const strata::ds4::Stats& s) {
    if (s.prof_tokens <= s0.prof_tokens) return;
    const double pt = (double) (s.prof_tokens - s0.prof_tokens);
    double sum = 0;
    for (int i = 0; i < 6; ++i) sum += (s.prof_ms[i] - s0.prof_ms[i]) / pt;
    std::fprintf(stderr, "GPU per token (DS4_PROF): %.2f ms - between layers %.2f, attention %.2f, routing + shared "
                         "expert %.2f, experts gate/up %.2f, waiting for the CPU lane %.2f, down + post %.2f; %lld tokens\n",
                 sum, (s.prof_ms[0] - s0.prof_ms[0]) / pt, (s.prof_ms[1] - s0.prof_ms[1]) / pt,
                 (s.prof_ms[2] - s0.prof_ms[2]) / pt, (s.prof_ms[3] - s0.prof_ms[3]) / pt,
                 (s.prof_ms[4] - s0.prof_ms[4]) / pt, (s.prof_ms[5] - s0.prof_ms[5]) / pt,
                 (long long) (s.prof_tokens - s0.prof_tokens));
}

// The drafts between two stats: each draft position (1st, 2nd, ..) - how often it was verified with every draft
// before it kept, and how often it was kept then
std::string per_position(const strata::ds4::Stats& s0, const strata::ds4::Stats& s1) {
    std::string pp = "per position (kept / reached):";
    for (int i = 0; i < 8; ++i) {
        const long long r = s1.draft_reached[i] - s0.draft_reached[i], k = s1.draft_kept[i] - s0.draft_kept[i];
        if (r <= 0) break;
        char b[64];
        std::snprintf(b, sizeof b, " %d: %lld/%lld %.0f%%", i + 1, k, r, 100.0 * (double) k / (double) r);
        pp += b;
    }
    return pp;
}

// The drafts' cut on the product of their confidences (DeepInfer's cumulative rule, GLM's spec_cum): a draft is
// verified while the chance that it and every draft before it are kept stays at least this - a long window whose later
// drafts are likely lost is not verified whole.  MAYA_DS4_DSPARK_CUM=F (STRATA_DS4_DSPARK_CUM), a request's dspark_cum
// key; 0: off, only the floor on each draft's own confidence (dspark_conf).  0.5 by default: a Radeon 8065S, Maya-S,
// greedy over code / chat / a list, two runs: decode 12.04 -> 12.82 tok/s (+6.5%), the same tokens
float dspark_cum_default() {
    static const float v = [] {
        const char* e = std::getenv("STRATA_DS4_DSPARK_CUM");
        return e != nullptr ? (float) std::atof(e) : 0.5f;
    }();
    return v;
}

// Prompt lookup beside DSpark (the suffix drafter: the tokens that followed an earlier repeat of the context - code
// edits, quoted text, a list that comes back): a round whose lookup proposal starts with DSpark's first draft verifies
// the lookup's tokens instead where the draft policy (strata/spec/draft_policy.hpp) expects more committed tokens a
// millisecond from them - from this PC's round times by window size and the lookup drafts' acceptance, both measured
// as it decodes - so a lookup window may run past DSpark's block, up to verify_max().  Only which drafts are verified
// changes: verify() keeps a draft the drafter did not draw with probability p(d) (q a point at it), so every token is
// still the model's own draw.  Not beside DS4_DSPARK_SAMPLE=1's drawn drafts (the choice would then hang on a draw).
// A round with no drafter window (--dspark none or off, or DSpark resting after drafts below its confidence) weighs
// the lookup's drafts against a plain step the same way.  --suffix-draft N: the shortest match it proposes from (3; 0:
// off)
struct Lookup {
    strata::spec::SuffixDrafter sfx;
    strata::spec::DraftPolicy policy;   // (over the engine's life: the round times are this PC's)
    std::vector<int32_t> buf;
    long long windows = 0, offered = 0, accepted = 0;   // the lookup's own rounds and drafts
    Lookup(int min_match, int max_t, size_t cap) : sfx(min_match, 64, cap), policy(max_t), buf((size_t) max_t) {}
    void start(const std::vector<int32_t>& ctx) {   // a request's history: what the state has consumed
        sfx.reset();
        sfx.append(ctx.data(), ctx.size());
    }
};

// DS4_ROUND_LOG=<file>: a line a decoding round that drafted (appended) - the tag (ids: the CLI's run, p<n>: a
// --spec-eval prompt, r<n> / rid<n>: a serve request), the position, the drafts' source (dspark / lookup), how many
// were made, verified and kept, and the drafter's confidences for its drafts (- for the lookup's): the drafter's
// acceptance read off afterwards, for any confidence floor.  The rounds of one conversation at a time (a --parallel
// step of several is not logged).  --spec-eval reads the last round's note (record) without a file too
struct RoundLog {
    std::FILE* f = nullptr;
    bool record = false;
    std::string tag = "-";
    // the last round's note: drafts made (-1: no drafter round), verified, kept; the drafter's confidences
    int made = -1, verified = 0, kept = 0;
    bool lookup = false;
    std::vector<float> conf;
    bool on() const { return f != nullptr || record; }
    void note(long long pos, bool lk, int made_, int verified_, int kept_, const std::vector<float>& c) {
        made = made_, verified = verified_, kept = kept_, lookup = lk;
        conf = c;
        if (f == nullptr) return;
        std::fprintf(f, "%s\t%lld\t%s\t%d\t%d\t%d\t", tag.c_str(), pos, lk ? "lookup" : "dspark", made_, verified_, kept_);
        if (lk || c.empty()) std::fputc('-', f);
        for (size_t i = 0; !lk && i < c.size(); ++i) std::fprintf(f, "%s%.4f", i ? "," : "", c[i]);
        std::fputc('\n', f);
        std::fflush(f);
    }
};
RoundLog& round_log() {
    static RoundLog r = [] {
        RoundLog x;
        if (const char* p = std::getenv("DS4_ROUND_LOG")) {
            x.f = std::fopen(p, "a");
            if (x.f == nullptr) std::fprintf(stderr, "maya-ds4: DS4_ROUND_LOG=%s: cannot open it\n", p);
            else std::fprintf(x.f, "# tag\tposition\tsource\tmade\tverified\tkept\tconfidences\n");
        }
        return x;
    }();
    return r;
}

// One round of decoding after `tok` (the token at pos(), not fed yet): with the drafter, its drafts while their
// confidence holds (at most `room` - 1 of them) - or the lookup's (lk, above) - verified together; else a plain step.
// `out` gets the tokens that follow tok (the accepted drafts, then the trunk's next), the last of them the next round's
// tok.
bool round(strata::ds4::Model& m, int tok, int room, bool spec, float conf_min, float temperature, float top_p, int top_k,
           float min_p, unsigned long long seed, unsigned long long& counter, std::vector<int>& out, int& offered,
           int& accepted, std::string& err, float cum_min = -1.0f, Lookup* lk = nullptr) {
    out.clear();
    offered = accepted = 0;
    const double t0 = now_ms();
    int match = 0;
    const bool drafter = spec && room > 1 && m.draft_block() > 0;
    RoundLog& rl = round_log();   // (DS4_ROUND_LOG, --spec-eval)
    const long long pos0 = rl.on() ? (long long) m.pos() : 0;
    if (rl.on()) rl.made = -1;
    if (drafter) {
        std::vector<int> drafts;
        std::vector<float> conf;
        if (!m.draft(tok, drafts, conf, err, temperature, top_p, top_k, min_p, seed, counter)) return false;
        const int made = (int) drafts.size();
        int keep = 0;
        if (cum_min < 0) cum_min = dspark_cum_default();
        float cum = 1.0f;
        while (keep < (int) drafts.size() && keep < room - 1 && conf[(size_t) keep] >= conf_min &&
               (cum *= conf[(size_t) keep]) >= cum_min)
            ++keep;
        bool from_lk = false;
        if (lk != nullptr && !drafts.empty() && !m.drafts_sampled()) {
            const int k = lk->sfx.propose(std::min(room, m.verify_max()) - 1, lk->buf.data());
            match = lk->sfx.last_match();
            if (k > 0 && lk->buf[0] == drafts[0]) {
                const strata::spec::DraftPolicy::Pick pk = lk->policy.choose(keep + 1, k, match);
                if (pk.lookup) {
                    keep = pk.t - 1;
                    drafts.assign(lk->buf.begin(), lk->buf.begin() + keep);
                    from_lk = true;
                }
            }
        }
        if (keep > 0) {
            drafts.resize((size_t) keep);
            int acc = 0, next = -1;
            if (!m.verify(tok, drafts, temperature, top_p, top_k, min_p, seed, counter, acc, next, err)) return false;
            if (rl.on()) rl.note(pos0, from_lk, from_lk ? keep : made, keep, acc, conf);
            counter += (unsigned long long) acc + 1;
            for (int i = 0; i < acc; ++i) out.push_back(drafts[(size_t) i]);
            out.push_back(next);
            offered = keep;
            accepted = acc;
            if (lk != nullptr) {
                lk->policy.observe(from_lk, keep + 1, acc, match, now_ms() - t0);
                if (from_lk) {
                    ++lk->windows;
                    lk->offered += keep;
                    lk->accepted += acc;
                }
            }
            return true;
        }
        if (rl.on() && made > 0) rl.note(pos0, false, made, 0, 0, conf);   // (drafts made, none confident enough)
    }
    // no drafter this round (DSpark not loaded or off, or resting after a round of drafts below its confidence): the
    // lookup's drafts alone, where the policy expects more committed tokens a millisecond from them than from a plain
    // step - each a point to verify(), as beside DSpark
    if (lk != nullptr && !drafter && room > 1) {
        const int k = lk->sfx.propose(std::min(room, m.verify_max()) - 1, lk->buf.data());
        match = lk->sfx.last_match();
        strata::spec::DraftPolicy::Pick pk;
        if (k > 0) pk = lk->policy.choose(1, k, match);
        if (pk.lookup && pk.t > 1) {
            const std::vector<int> drafts(lk->buf.begin(), lk->buf.begin() + (pk.t - 1));
            int acc = 0, next = -1;
            if (!m.verify(tok, drafts, temperature, top_p, top_k, min_p, seed, counter, acc, next, err)) return false;
            if (rl.on()) rl.note(pos0, true, k, (int) drafts.size(), acc, {});
            counter += (unsigned long long) acc + 1;
            for (int i = 0; i < acc; ++i) out.push_back(drafts[(size_t) i]);
            out.push_back(next);
            offered = (int) drafts.size();
            accepted = acc;
            lk->policy.observe(true, pk.t, acc, match, now_ms() - t0);
            ++lk->windows;
            lk->offered += offered;
            lk->accepted += acc;
            return true;
        }
    }
    if (!m.step(tok, true, err)) return false;
    out.push_back(m.sample(temperature, top_p, top_k, min_p, seed, counter++));
    if (lk != nullptr) lk->policy.observe(false, 1, 0, match, now_ms() - t0);
    return true;
}

struct Req {
    long long max_new = 0;
    float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 64;
    unsigned long long seed = 0;
    int spec = -1;          // spec=0|1: this request drafts or not (-1: as the engine was started)
    float conf = -1.0f;     // dspark_conf=F: the drafts' confidence floor for this request (-1: the engine's)
    float cum = -1.0f;      // dspark_cum=F: the cut on the product of the confidences (-1: the engine's, dspark_cum_default)
    int lookup = -1;        // suffix_draft=0|1: this request's prompt lookup off / on (-1: as the engine was started)
    // the penalties (penalty_last_n / _repeat / _freq / _present, llama.cpp's): over the last pen_n tokens of the state
    // (-1: the whole context, which the engine caps at its window; 0: none)
    int pen_n = 0;
    float pen_repeat = 1.0f, pen_freq = 0.0f, pen_present = 0.0f;
    // the thinking budget (think_budget=<n> think_end=<id>, the server's run config `thinking_budget`): a reasoning
    // block still open after n tokens of the answer is closed - its next token is think_end - and the answer follows
    long long think_budget = 0;
    int think_end = -1;
    std::string file;   // GENI: the pictures
    std::vector<int> ids;
    long long rid = -1;   // rid=<id> (--parallel: every request's, its lines tagged with it)
    // marks=<pos>,... and shared=<pos>: restore points the read keeps (the server's: the start of the last user
    // message, a point near the prompt's tail; the end of a start other conversations share, given up last) - a later
    // prompt that leaves this one past one of them goes back there (ds4_marks.hpp); none: as before
    std::vector<int64_t> marks;
    long long shared = -1;
};

// A prompt's picture: where its span starts, its pixels (the server's resize: RGB rows, both sides multiples of the
// patch size), a hash of them; once encoded, its span's rows
struct Pic {
    int64_t start = 0;
    int width = 0, height = 0;
    uint64_t hash = 0;
    size_t len = 0;   // the span (0: not known yet)
    std::vector<uint8_t> rgb;
};

// the same picture at the same place (start and pixel hash): `v` holds `q`
bool pic_same(const std::vector<Pic>& v, const Pic& q) {
    for (const Pic& x : v)
        if (x.start == q.start && x.hash == q.hash) return true;
    return false;
}

// do `have` and `want` agree on every picture that starts before `c` - both ways (no picture in one, none missed in
// the other): the states up to `c` share their pictures
bool pics_match(const std::vector<Pic>& have, const std::vector<Pic>& want, size_t c) {
    for (const Pic& q : want)
        if ((size_t) q.start < c && !pic_same(have, q)) return false;
    for (const Pic& q : have)
        if ((size_t) q.start < c && !pic_same(want, q)) return false;
    return true;
}

// GENI's file: int32 'DSI1', int32 count, then per picture int64 start, int32 width, int32 height and its
// height x width x 3 RGB bytes, in prompt order
bool read_pics(const std::string& path, std::vector<Pic>& v, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "cannot open the pictures' file " + path;
        return false;
    }
    char magic[4] = {};
    int32_t n = 0;
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "DSI1", 4) == 0 && std::fread(&n, 4, 1, f) == 1 &&
              n >= 0 && n <= 4096;
    for (int i = 0; ok && i < n; ++i) {
        Pic p;
        int32_t wh[2] = {};
        ok = std::fread(&p.start, 8, 1, f) == 1 && std::fread(wh, 4, 2, f) == 2 && wh[0] > 0 && wh[1] > 0 &&
             (int64_t) wh[0] * wh[1] <= (64 << 20);
        if (!ok) break;
        p.width = wh[0], p.height = wh[1];
        p.rgb.resize((size_t) p.width * p.height * 3);
        ok = std::fread(p.rgb.data(), 1, p.rgb.size(), f) == p.rgb.size();
        uint64_t h = 1469598103934665603ull;   // FNV-1a over the size and the pixels
        for (int32_t x : wh)
            for (int b = 0; b < 4; ++b) h = (h ^ (uint8_t) (x >> (8 * b))) * 1099511628211ull;
        for (uint8_t b : p.rgb) h = (h ^ b) * 1099511628211ull;
        p.hash = h;
        v.push_back(std::move(p));
    }
    std::fclose(f);
    if (!ok) err = path + ": not a pictures' file (DSI1)";
    return ok;
}

// the encoded spans, the latest few (a conversation sends its pictures with every turn)
struct SpanCache {
    std::deque<std::pair<uint64_t, std::shared_ptr<const std::vector<uint16_t>>>> q;
    std::shared_ptr<const std::vector<uint16_t>> get(strata::ds4::Model& m, const Pic& p, std::string& err) {
        for (size_t i = 0; i < q.size(); ++i)
            if (q[i].first == p.hash) {
                auto r = q[i].second;
                q.erase(q.begin() + (long) i);
                q.emplace_front(p.hash, r);
                return r;
            }
        auto rows = std::make_shared<std::vector<uint16_t>>();
        if (!m.encode_image(p.rgb.data(), p.width, p.height, *rows, err)) return nullptr;
        q.emplace_front(p.hash, rows);
        if (q.size() > 8) q.pop_back();
        return rows;
    }
};

bool parse_gen(const std::string& line, Req& r, bool geni) {
    const char* p = line.c_str() + (geni ? 5 : 4);
    char* e = nullptr;
    r.max_new = std::strtoll(p, &e, 10);
    if (e == p) return false;
    p = e;
    for (;;) {
        while (*p == ' ') ++p;
        const char* s = p;
        while (*p && *p != ' ') ++p;
        if (p == s) break;
        const std::string tok(s, (size_t) (p - s));
        const size_t eq = tok.find('=');
        if (eq == std::string::npos && geni && r.file.empty()) {
            r.file = tok;
            continue;
        }
        if (eq == std::string::npos) {
            r.ids = parse_ids(tok);
            break;
        }
        const std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
        if (k == "temperature") r.temperature = std::strtof(v.c_str(), nullptr);
        else if (k == "top_p") r.top_p = std::strtof(v.c_str(), nullptr);
        else if (k == "top_k") r.top_k = std::atoi(v.c_str());
        else if (k == "min_p") r.min_p = std::strtof(v.c_str(), nullptr);
        else if (k == "seed") r.seed = std::strtoull(v.c_str(), nullptr, 10);
        else if (k == "spec") r.spec = std::atoi(v.c_str()) != 0;
        else if (k == "dspark_conf") r.conf = std::strtof(v.c_str(), nullptr);
        else if (k == "dspark_cum" || k == "spec_cum") r.cum = std::strtof(v.c_str(), nullptr);
        else if (k == "spec_min_p") r.conf = std::strtof(v.c_str(), nullptr);   // (the server's strata_tune names)
        else if (k == "suffix_draft") r.lookup = std::atoi(v.c_str()) != 0;
        else if (k == "penalty_last_n") r.pen_n = std::atoi(v.c_str());
        else if (k == "penalty_repeat") r.pen_repeat = std::strtof(v.c_str(), nullptr);
        else if (k == "penalty_freq") r.pen_freq = std::strtof(v.c_str(), nullptr);
        else if (k == "penalty_present") r.pen_present = std::strtof(v.c_str(), nullptr);
        else if (k == "think_budget") r.think_budget = std::atoll(v.c_str());
        else if (k == "think_end") r.think_end = std::atoi(v.c_str());
        else if (k == "rid") r.rid = std::atoll(v.c_str());
        else if (k == "marks") {
            for (int x : parse_ids(v))
                if (x > 0) r.marks.push_back(x);
        } else if (k == "shared") r.shared = std::atoll(v.c_str());
    }
    return !r.ids.empty() && (!geni || !r.file.empty());
}

// the serve watchdog (#539, #494): a request whose heartbeat - its prompt's progress, its tokens, the steps between -
// stops for MAYA_DS4_WATCHDOG_S seconds (600; 0: off) ends the engine with a line saying where each device's thread
// was, and the server starts it again for the next request (a run that never ends otherwise holds every request behind
// it until someone restarts the engine by hand).  A healthy request beats every few seconds - a prompt every chunk, a
// slow PC's chunk of a long prompt a minute or two
std::atomic<uint64_t> g_beats{0};
std::atomic<bool> g_busy{false};
void beat() { g_beats.fetch_add(1, std::memory_order_relaxed); }
void start_watchdog(const strata::ds4::Model& m) {
    const char* v = std::getenv("STRATA_DS4_WATCHDOG_S");
    if (v == nullptr) v = std::getenv("MAYA_DS4_WATCHDOG_S");
    const int limit = v != nullptr ? std::atoi(v) : 600;
    if (limit <= 0) return;
    std::thread([&m, limit] {
        strata::ds4::StallRule rule;   // (ds4_watchdog.hpp: the time since the last beat, never the request's length)
        rule.limit_s = limit;
        const auto t0 = std::chrono::steady_clock::now();
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            const double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (!rule.stalled(g_busy.load(), g_beats.load(), now)) continue;
            std::fprintf(stderr, "maya ds4 serve: where it stopped: %s\n", m.where().c_str());
            // (the server reads the line with "issue #29" as the engine's own watchdog, not a lack of RAM)
            std::fprintf(stderr, "maya ds4 serve: no progress for %d s during a request - stopping the engine so the "
                                 "server starts it again (issue #29, #539; MAYA_DS4_WATCHDOG_S)\n", limit);
            std::fflush(stderr);
            std::_Exit(3);   // (not abort(): a core dump of the pinned expert tiers takes minutes)
        }
    }).detach();
}

int serve(strata::ds4::Model& m, const std::string& eos_list, bool spec, float conf_min, const std::string& pack,
          Lookup* lk) {
    std::vector<int> eos = parse_ids(eos_list);
    for (const std::string& l : m.info_lines()) std::printf("%s\n", l.c_str());
    if (m.seq_count() > 1)   // (--parallel: n conversations at once, GEN rid=<id>)
        std::printf("READY %lld stop multi=%d\n", (long long) m.max_context(), m.seq_count());
    else
        std::printf("READY %lld stop\n", (long long) m.max_context());
    std::fflush(stdout);
    start_watchdog(m);
    std::fprintf(stderr, "maya ds4 serve: GEN <max_new> [keys] <ids>; GENI <max_new> [keys] <pictures> <ids>; STOP, QUIT\n");
    Lines in;
    in.start();
    std::string line;
    unsigned long long counter = 0;
    long long log_req = 0;   // (DS4_ROUND_LOG's tags)
    std::vector<Pic> held;   // the pictures the state holds (start, hash, span)
    // (DS4_SERVE_DUMP=<prefix>, tests: request i's logits - the prompt's last row, then every decode step's rows - into
    //  <prefix>.<i>.f32, to compare a request's logits byte for byte across processes and histories)
    const char* dump_prefix = std::getenv("DS4_SERVE_DUMP");
    int n_req = 0;
    std::vector<float> dump_lg;
    std::FILE* dump_f = nullptr;
    auto dump_rows = [&]() {
        if (dump_f != nullptr && m.logits(dump_lg)) std::fwrite(dump_lg.data(), 4, dump_lg.size(), dump_f);
    };
    SpanCache spans;
    const int img_tok = m.image_token();
    // the vision tower's room taken for a request's pictures and given back: the VRAM lines again, so the server's
    // Monitor shows the tower only while it reads them
    m.set_vision_note([&m] {
        for (const std::string& l : m.info_lines())
            if (l.rfind("VRAM ", 0) == 0) std::printf("%s\n", l.c_str());
        std::fflush(stdout);
    });
    // CONVERSATION SLOTS: a request that does not continue the conversation in the model first sets that one aside
    // (Model::slot_save: its window rings, the compressors' partial groups and every source's compressed KV and index
    // keys), then takes back the kept conversation this prompt does continue (slot_load) - chats, and an agent's side
    // requests, that interleave no longer re-read each other's prompts (70k and 185k tokens alternating: 75-200 s a
    // switch at ~930 tok/s).  MAYA_DS4_SLOT_KEEP=1 turns them on and keeps them across runs, in MAYA_DS4_SLOT_DIR
    // (default <pack>/slots): a slot is a .bin (the state) and a .meta (this model's identity, the conversation's
    // tokens and its size), both written under a temporary name first; a slot whose .meta is missing, foreign or does
    // not match its file, or unused for MAYA_DS4_SLOT_DAYS (7) days, is removed at start.  MAYA_DS4_SLOT_CAP (4) is
    // the conversations kept in RAM; past it the least recently used is written to the folder and read back from there
    // (RAM first, the folder past the cap), and the engine's end writes the ones in RAM to the folder, so the next run
    // reads on from them.  MAYA_DS4_SLOT_RAM_GB (0, off) additionally caps their bytes in RAM after each save: the
    // incoming state is serialized first (the peak can exceed the cap), then the least recently used spills to the
    // folder, with one RAM conversation kept even when it alone is over the cap.  MAYA_DS4_SLOT_GB
    // (16) caps the folder's .bin files: at start, and after every write, the folder's least recently used conversation
    // is removed while they are over it (never the one just written, so a conversation bigger than the cap stays).
    // MAYA_DS4_SLOT_MIN (1024) is the shortest conversation set aside (a
    // shorter one is read again in about the time its slot would take).  Pictures travel as their marks: where a span
    // starts, the fingerprint of its pixels, its length - not the pixels, not the rows (the KV of the span is what the
    // rows did).  A conversation is taken back for a prompt whose pictures are the same at the same places; another
    // picture, or the same one elsewhere, shares no state past that place.
    const auto env_num = [](const char* k, double d) { const char* v = std::getenv(k); return v ? std::atof(v) : d; };
    const bool slot_keep =
        std::getenv("STRATA_DS4_SLOT_KEEP") != nullptr && std::atoi(std::getenv("STRATA_DS4_SLOT_KEEP")) != 0;
    int slot_max = slot_keep ? std::max(0, (int) env_num("STRATA_DS4_SLOT_CAP", 4)) : 0;
    const size_t slot_min = (size_t) std::max(1.0, env_num("STRATA_DS4_SLOT_MIN", 1024));
    const double slot_days = env_num("STRATA_DS4_SLOT_DAYS", 7);
    const uint64_t slot_budget = (uint64_t) (env_num("STRATA_DS4_SLOT_GB", 16) * 1073741824.0);
    const uint64_t slot_ram_budget = [] {
        const char* k = "STRATA_DS4_SLOT_RAM_GB", *v = std::getenv(k);
        if (v == nullptr) return (uint64_t) 0;
        char* end = nullptr;
        const double gb = std::strtod(v, &end);
        while (end != nullptr && (*end == ' ' || *end == '\t')) ++end;
        const double max_gb = (double) (~(uint64_t) 0) / 1073741824.0;
        if (end == nullptr || end == v || *end != '\0' || !std::isfinite(gb) || gb < 0 || gb >= max_gb) {
            std::fprintf(stderr, "ds4 slots: invalid %s=%s, using 0 (count-limited only)\n", k, v);
            return (uint64_t) 0;
        }
        return (uint64_t) (gb * 1073741824.0);
    }();
    namespace fs = std::filesystem;
    std::filesystem::path slot_dir;
    struct Slot {
        std::vector<int32_t> tokens;
        std::string path;                            // empty: the state is in data (RAM)
        uint64_t bytes = 0, used = 0;
        std::shared_ptr<std::vector<uint8_t>> data;   // the RAM slot's bytes (null: a file slot)
        std::vector<int64_t> cks;                     // the positions short of its end it can be taken back at
        std::vector<Pic> pics;                        // the pictures it holds (start, hash; len once it is loaded)
    };
    std::vector<Slot> slots;
    uint64_t slot_clock = 0, slot_seq = 0;
    const std::string slot_identity = m.slot_identity();
    constexpr uint64_t kMetaMagic = 0x334154454d345344ull;   // "DS4META3" (little endian)
    const auto note = [&](const std::string& msg) {
        std::fprintf(stderr, "ds4 slots: %s\n", msg.c_str());
        std::printf("NOTE sessions: %s\n", msg.c_str());
        std::fflush(stdout);
    };
    const auto gb = [](uint64_t b) {
        char s[32];
        std::snprintf(s, sizeof s, "%.2f GB", (double) b / 1e9);
        return std::string(s);
    };
    const auto meta_of = [](const std::string& bin) { return bin.substr(0, bin.size() - 4) + ".meta"; };
    const auto meta_write = [&](const std::string& path, const std::vector<int32_t>& toks, uint64_t bytes,
                                const std::vector<int64_t>& cks, const std::vector<Pic>& pics) -> bool {
        const std::string tmp = path + ".tmp";
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            const uint64_t il = slot_identity.size(), nt = toks.size();
            f.write((const char*) &kMetaMagic, 8);
            f.write((const char*) &il, 8);
            f.write(slot_identity.data(), (std::streamsize) il);
            f.write((const char*) &bytes, 8);
            f.write((const char*) &nt, 8);
            f.write((const char*) toks.data(), (std::streamsize) (nt * sizeof(int32_t)));
            const uint64_t nc = cks.size();
            f.write((const char*) &nc, 8);
            f.write((const char*) cks.data(), (std::streamsize) (nc * sizeof(int64_t)));
            const uint64_t np = pics.size();   // (the pictures: where they start and their pixels' hash - the match
            f.write((const char*) &np, 8);      //  is judged from here, before the .bin is read)
            for (const Pic& q : pics) {
                f.write((const char*) &q.start, 8);
                f.write((const char*) &q.hash, 8);
            }
            f.write((const char*) &kMetaMagic, 8);
            f.flush();
            if (!f) return false;
        }
        std::error_code ec;
        fs::rename(tmp, path, ec);
        return !ec;
    };
    // a .meta of this model -> its tokens, the slot's size and its pictures (false: foreign, damaged or not this
    // model's)
    const auto meta_read = [&](const std::string& path, std::vector<int32_t>& toks, uint64_t& bytes,
                               std::vector<int64_t>& cks, std::vector<Pic>& pics) -> bool {
        std::ifstream f(path, std::ios::binary);
        uint64_t m0 = 0, il = 0, nt = 0, nc = 0, np = 0, m1 = 0;
        if (!f.read((char*) &m0, 8) || m0 != kMetaMagic || !f.read((char*) &il, 8) || il != slot_identity.size())
            return false;
        std::string id(il, '\0');
        if (!f.read(id.data(), (std::streamsize) il) || id != slot_identity) return false;
        if (!f.read((char*) &bytes, 8) || !f.read((char*) &nt, 8) || nt == 0 || nt > (uint64_t) m.max_context())
            return false;
        toks.resize(nt);
        if (!f.read((char*) toks.data(), (std::streamsize) (nt * sizeof(int32_t))) || !f.read((char*) &nc, 8) ||
            nc > 3 + (uint64_t) strata::ds4::kMarksMax)   // (the three checkpoints and the marks)
            return false;
        cks.resize(nc);
        if (!f.read((char*) cks.data(), (std::streamsize) (nc * sizeof(int64_t))) || !f.read((char*) &np, 8)) return false;
        if (np > 4096) return false;
        pics.clear();
        for (uint64_t k = 0; k < np; ++k) {
            Pic q;
            if (!f.read((char*) &q.start, 8) || !f.read((char*) &q.hash, 8)) return false;
            if (q.start < 0 || (uint64_t) q.start >= nt || (k > 0 && q.start <= pics.back().start)) return false;
            pics.push_back(q);
        }
        if (!f.read((char*) &m1, 8) || m1 != kMetaMagic) return false;
        for (int64_t c : cks)
            if (c <= 0 || (uint64_t) c > nt) return false;
        return true;
    };
    const auto slot_age_days = [](const std::string& p) {
        std::error_code ec;
        const auto t = fs::last_write_time(p, ec);
        if (ec) return 1e9;
        return std::chrono::duration<double>(fs::file_time_type::clock::now() - t).count() / 86400.0;
    };
    const auto slot_drop = [&](size_t i) {
        if (slots[i].data)   // a RAM slot: its buffer simply dies
            slots[i].data.reset();
        else {
            std::error_code ec;
            std::filesystem::remove(slots[i].path, ec);
            std::filesystem::remove(meta_of(slots[i].path), ec);
        }
        slots.erase(slots.begin() + (long) i);
    };
    const auto slot_count = [&](int& ram, int& disk) {
        ram = disk = 0;
        for (const Slot& s : slots) (s.data ? ram : disk) += 1;
    };
    const auto slot_ram_bytes = [&]() {
        uint64_t total = 0;
        for (const Slot& s : slots)
            if (s.data) total += s.data->size();
        return total;
    };
    // the folder's conversations (their .bin; a .meta beside each is a few KB: identity and tokens)
    const auto slot_folder_bytes = [&]() {
        uint64_t total = 0;
        std::error_code ec;
        for (const auto& de : std::filesystem::directory_iterator(slot_dir, ec)) {
            const std::string fn = de.path().filename().string();
            if (fn.rfind("slot-", 0) == 0 && fn.size() > 9 && fn.compare(fn.size() - 4, 4, ".bin") == 0) {
                const uint64_t bytes = (uint64_t) std::filesystem::file_size(de.path(), ec);
                if (!ec) total += bytes;
            }
        }
        return total;
    };
    // the folder over MAYA_DS4_SLOT_GB: its least recently used conversation goes first, and never `keep` (the one just
    // written), so a conversation bigger than the cap stays
    const auto slot_trim = [&](const std::string& keep) {
        const uint64_t before = slot_folder_bytes();
        size_t dropped = 0;
        while (slot_folder_bytes() > slot_budget) {
            size_t lru = SIZE_MAX;
            for (size_t i = 0; i < slots.size(); ++i)
                if (!slots[i].data && slots[i].path != keep && (lru == SIZE_MAX || slots[i].used < slots[lru].used))
                    lru = i;
            if (lru == SIZE_MAX) break;
            slot_drop(lru);
            ++dropped;
        }
        if (dropped > 0) {
            char msg[256];
            std::snprintf(msg, sizeof msg, "the folder held %s (over MAYA_DS4_SLOT_GB, %s): %zu least recently used "
                          "conversation(s) removed, %s left", gb(before).c_str(), gb(slot_budget).c_str(), dropped,
                          gb(slot_folder_bytes()).c_str());
            note(msg);
        }
    };
    // a RAM slot into the folder (its bytes, then its .meta); false: not written (its bytes go, then)
    const auto slot_write_file = [&](Slot& s) -> bool {
        std::error_code ec;
        const auto space = std::filesystem::space(slot_dir, ec);
        if (!ec && space.available < ((uint64_t) 8 << 30)) {   // (8 GB free, the GLM engine's guard)
            std::fprintf(stderr, "ds4 slots: under 8 GB free in %s - the conversation is not written\n",
                         slot_dir.string().c_str());
            return false;
        }
        const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::system_clock::now().time_since_epoch()).count();
        const std::string path =
            (slot_dir / ("slot-" + std::to_string(stamp) + "-" + std::to_string(++slot_seq) + ".bin")).string();
        {
            std::ofstream f(path + ".tmp", std::ios::binary | std::ios::trunc);
            f.write((const char*) s.data->data(), (std::streamsize) s.data->size());
            f.flush();
            if (!f) {
                std::filesystem::remove(path + ".tmp", ec);
                std::fprintf(stderr, "ds4 slots: writing %s.tmp failed (disk full?)\n", path.c_str());
                return false;
            }
        }
        fs::rename(path + ".tmp", path, ec);
        if (ec || !meta_write(meta_of(path), s.tokens, s.data->size(), s.cks, s.pics)) {
            std::filesystem::remove(path + ".tmp", ec);
            std::filesystem::remove(path, ec);
            std::fprintf(stderr, "ds4 slots: writing %s failed\n", path.c_str());
            return false;
        }
        s.path = path;
        s.bytes = s.data->size();
        s.data.reset();
        slot_trim(path);   // (the folder over its cap: the least recently used conversation goes, not this one)
        return true;
    };
    // the conversation in the model into a slot (the state is lost otherwise: another one takes the model)
    const auto slot_put = [&](const char* why) {
        if (slot_max <= 0) return;
        const std::vector<int32_t>& t = m.tokens();
        if (t.size() < slot_min) return;   // (a fresh engine's 0-token state is no conversation: nothing to set aside)
        std::vector<strata::ds4::PicMark> marks;   // (the state's pictures: what the slot keeps of them - no pixels,
                                                   //  no rows)
        for (const Pic& x : held)
            if (x.len > 0 && x.start >= 0 && (size_t) x.start + x.len <= (size_t) m.pos())
                marks.push_back({x.start, x.hash, (uint64_t) x.len});
        const auto t0 = std::chrono::steady_clock::now();
        auto buf = std::make_shared<std::vector<uint8_t>>();
        std::string e;
        const uint64_t bytes = m.slot_save(*buf, e, marks);   // (a state unchanged since it was taken refuses here:
        if (bytes == 0) {                                     // nothing changes, its slot stays)
            if (e != "it is already set aside")
                std::fprintf(stderr, "ds4 slots: setting the conversation aside failed (%s) - it is read from token 0 "
                                     "when it comes back\n", e.c_str());
            return;
        }
        for (size_t i = slots.size(); i-- > 0;)   // an earlier state of this conversation is superseded
            if (slots[i].tokens.size() <= t.size() && std::equal(slots[i].tokens.begin(), slots[i].tokens.end(), t.begin()))
                slot_drop(i);
        slots.push_back({t, "", bytes, ++slot_clock, nullptr, m.check_positions(), {}});
        slots.back().data = buf;
        slots.back().pics.clear();
        for (const auto& k : marks)
            slots.back().pics.push_back({k.start, 0, 0, k.hash, (size_t) k.len, {}});   // (start, hash, span: the
                                                                                       //  .meta keeps start and hash)
        while (true) {   // past either cap: the least recently used conversation in RAM to the folder, or gone
            int ram = 0, disk = 0;
            slot_count(ram, disk);
            const bool over_bytes = slot_ram_budget > 0 && ram > 1 && slot_ram_bytes() > slot_ram_budget;
            if (ram <= slot_max && !over_bytes) break;
            size_t lru = SIZE_MAX;
            for (size_t i = 0; i < slots.size(); ++i)
                if (slots[i].data && (lru == SIZE_MAX || slots[i].used < slots[lru].used)) lru = i;
            if (lru == SIZE_MAX) break;
            if (!slot_write_file(slots[lru])) slot_drop(lru);
        }
        int ram = 0, disk = 0;
        slot_count(ram, disk);
        char msg[256];
        std::snprintf(msg, sizeof msg, "saved a %zu-token conversation%s (%s in RAM in %.1f s); %d in RAM, %d in %s",
                      t.size(), why, gb(bytes).c_str(),
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), ram, disk,
                      slot_dir.string().c_str());
        note(msg);
        if (slot_ram_budget > 0) {
            std::snprintf(msg, sizeof msg, "RAM conversations: %s / %s; %d in RAM, %d in the folder",
                          gb(slot_ram_bytes()).c_str(), gb(slot_ram_budget).c_str(), ram, disk);
            note(msg);
        }
    };
    // the kept conversation this prompt shares the most of - past `floor` tokens, what the model itself can go back
    // to - back in the model: the tokens it covers (0: none).  A conversation the prompt continues whole comes back
    // whole (`whole`); one it leaves part way - a retried prompt, a history the client renders again without the
    // earlier reasoning - comes back at its latest checkpoint the prompt still agrees with (the end of a prompt, the
    // position before it, the anchor where its run of continued prompts began).  A conversation whose pictures (their
    // places, their pixels) the prompt agrees with up to that point comes back too: the KV of a picture's span needs
    // the same picture (the tokens alone are all the image token, whatever the image).
    const auto slot_take = [&](const std::vector<int>& prompt, const std::vector<Pic>& want, size_t floor,
                               bool& whole) -> size_t {
        whole = false;
        if (slot_max <= 0 || prompt.size() < 2) return 0;
        size_t best = SIZE_MAX, len = 0;
        for (size_t i = 0; i < slots.size(); ++i) {
            const auto& t = slots[i].tokens;
            const size_t lim = std::min(t.size(), prompt.size() - 1);   // (a prompt always reads its last token)
            size_t c = 0;
            while (c < lim && t[c] == prompt[c]) ++c;
            size_t u = c == t.size() ? c : 0;
            if (u == 0)
                for (int64_t k : slots[i].cks)
                    if ((size_t) k <= c && (size_t) k > u) u = (size_t) k;
            if (u == 0 || u <= len || u <= floor) continue;
            if (!pics_match(slots[i].pics, want, u)) continue;   // (another picture, another place: not this state)
            best = i;
            len = u;
        }
        if (best == SIZE_MAX) return 0;
        const size_t whole_len = slots[best].tokens.size();
        const auto t0 = std::chrono::steady_clock::now();
        const bool ram = slots[best].data != nullptr;
        std::string e;
        bool back = false;
        std::vector<strata::ds4::PicMark> marks;
        if (ram) back = m.slot_load(slots[best].data->data(), slots[best].data->size(), (int64_t) whole_len, e, &marks);
        else {
            std::ifstream f(slots[best].path, std::ios::binary);
            std::vector<uint8_t> file;
            if (f) {
                f.seekg(0, std::ios::end);
                const std::streamoff n = f.tellg();
                f.seekg(0, std::ios::beg);
                if (n > 0) {
                    file.resize((size_t) n);
                    if (f.read((char*) file.data(), n))
                        back = m.slot_load(file.data(), file.size(), (int64_t) whole_len, e, &marks);
                    else e = "reading " + slots[best].path + " failed";
                } else e = slots[best].path + " is empty";
            } else e = "cannot read " + slots[best].path;
        }
        if (!back) {
            note("taking a conversation back failed (" + (e.empty() ? std::string("it did not load") : e) +
                 ") - reading its prompt again");
            slot_drop(best);
            return 0;
        }
        if (len < whole_len && !(m.restore((int64_t) len) && (size_t) m.pos() == len)) {
            note("going back to token " + std::to_string(len) + " of a kept conversation failed - reading its prompt again");
            m.reset();
            return 0;
        }
        {   // the pictures the state holds now (the ones past `len` went with the tokens the restore dropped)
            std::vector<Pic> hp;
            for (const strata::ds4::PicMark& k : marks)
                if (k.start >= 0 && k.len > 0 && (size_t) k.start + (size_t) k.len <= len)
                    hp.push_back({k.start, 0, 0, k.hash, (size_t) k.len, {}});
            held = std::move(hp);   // (the prompt's pictures that `from` no longer covers are encoded again below)
        }
        whole = len == whole_len;
        slots[best].used = ++slot_clock;
        if (!ram) {   // its age counts from its last use (a file slot's only)
            std::error_code ec;
            fs::last_write_time(meta_of(slots[best].path), fs::file_time_type::clock::now(), ec);
        }
        char msg[256];
        std::string part;
        if (!whole) part = " at token " + std::to_string(len) + " of " + std::to_string(whole_len);
        std::snprintf(msg, sizeof msg, "took back a %zu-token conversation%s from %s (%.1f s); %zu new tokens to read",
                      len, part.c_str(), ram ? "RAM" : "the folder",
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                      prompt.size() - len);
        note(msg);
        return len;
    };
    if (slot_max > 0) {
        slot_dir = std::getenv("STRATA_DS4_SLOT_DIR") ? std::filesystem::path(std::getenv("STRATA_DS4_SLOT_DIR"))
                                                      : std::filesystem::path(pack) / "slots";
        std::error_code ec;
        std::filesystem::create_directories(slot_dir, ec);
        if (!std::filesystem::is_directory(slot_dir, ec)) {
            std::fprintf(stderr, "ds4 slots: %s is not writable - conversations are not set aside\n",
                         slot_dir.string().c_str());
            slot_max = 0;
        } else {
            struct Found {
                std::string bin;
                std::vector<int32_t> toks;
                uint64_t bytes;
                std::vector<int64_t> cks;
                std::vector<Pic> pics;
                fs::file_time_type t;
            };
            std::vector<Found> found;
            size_t refused = 0;
            for (const auto& de : std::filesystem::directory_iterator(slot_dir, ec)) {   // a previous run's slots
                const std::string fn = de.path().filename().string(), p = de.path().string();
                if (fn.rfind("slot-", 0) != 0) continue;
                const bool is_bin = fn.size() > 9 && fn.compare(fn.size() - 4, 4, ".bin") == 0;
                const bool is_meta = fn.size() > 10 && fn.compare(fn.size() - 5, 5, ".meta") == 0;
                if (!is_bin && !is_meta) {   // a .tmp an interrupted write left
                    std::filesystem::remove(de.path(), ec);
                    continue;
                }
                if (!is_meta) continue;      // a .bin is judged by its .meta
                const std::string bin = p.substr(0, p.size() - 5) + ".bin";
                Found fd{bin, {}, 0, {}, {}, fs::last_write_time(de.path(), ec)};
                const uint64_t have = fs::exists(bin, ec) ? (uint64_t) fs::file_size(bin, ec) : 0;
                if (meta_read(p, fd.toks, fd.bytes, fd.cks, fd.pics) && have == fd.bytes && fd.bytes > 0 &&
                    slot_age_days(p) <= slot_days)
                    found.push_back(std::move(fd));
                else {
                    ++refused;
                    std::filesystem::remove(p, ec);
                    std::filesystem::remove(bin, ec);
                }
            }
            for (const auto& de : std::filesystem::directory_iterator(slot_dir, ec)) {   // a .bin without its .meta
                const std::string p = de.path().string(), fn = de.path().filename().string();
                if (fn.rfind("slot-", 0) == 0 && fn.size() > 9 && fn.compare(fn.size() - 4, 4, ".bin") == 0 &&
                    !fs::exists(meta_of(p), ec))
                    std::filesystem::remove(de.path(), ec);
            }
            std::sort(found.begin(), found.end(), [](const Found& a, const Found& b) { return a.t < b.t; });
            uint64_t total = 0;
            for (auto& fd : found) {
                total += fd.bytes;
                slots.push_back({std::move(fd.toks), fd.bin, fd.bytes, ++slot_clock, nullptr, std::move(fd.cks),
                                 std::move(fd.pics)});
            }
            std::string msg = "folder " + slot_dir.string() + ": " + std::to_string(slots.size()) +
                              " conversation(s) kept from earlier runs (" + gb(total) + ")";
            if (refused) msg += ", " + std::to_string(refused) + " not this model's or incomplete - removed";
            note(msg);
            slot_trim("");   // (the folder over MAYA_DS4_SLOT_GB: the least recently used conversation goes first)
        }
    }
    if (slot_max > 0 && slot_ram_budget > 0) {
        char msg[256];
        std::snprintf(msg, sizeof msg, "RAM conversation budget: %s (LRU states spill to the folder; one RAM slot is always kept)",
                      gb(slot_ram_budget).c_str());
        note(msg);
    }
    if (slot_max > 0) {
#if !defined(_WIN32)
        // (the server ends the engine with QUIT and waits for it; a SIGTERM to the whole process group - systemd
        // stopping a service, a supervisor killing the server's group - must not end it mid-write, and a server that
        // is already gone must not end it with SIGPIPE when it writes its last lines)
        std::signal(SIGTERM, SIG_IGN);
        std::signal(SIGINT, SIG_IGN);
        std::signal(SIGPIPE, SIG_IGN);
#else
        // (Windows sends a console's Ctrl+C and Ctrl+Break to every process on it, the engine too; the server stops
        // on them and ends the engine with QUIT)
        std::signal(SIGINT, SIG_IGN);
        std::signal(SIGBREAK, SIG_IGN);
#endif
    }
    // STAT for the Monitor over a decode so far (from t1, the counters s_dec and the storage's io_dec then): the GLM
    // engine's keys (speed, the VRAM hit, promotions a token) and DeepSeek's own - experts a token on the CPU lane, MB
    // a token the storage read (Linux), the n-gram memory's (Engram) read time a token, the drafts' acceptance
    const auto stat_text = [&](double t1, const strata::ds4::Stats& s_dec, long long io_dec, long long produced,
                               long long d_off, long long d_acc) -> std::string {
        const strata::ds4::Stats& s = m.stats();
        const double ms = now_ms() - t1;
        const double fed = (double) std::max<int64_t>(1, s.tokens - s_dec.tokens);
        const double ex = (double) (s.experts - s_dec.experts), cx = (double) (s.cpu_experts - s_dec.cpu_experts),
                     sx = (double) (s.stream_experts - s_dec.stream_experts),
                     hx = (double) (s.helper_experts - s_dec.helper_experts);
        std::string out = "STAT";
        char b[160];
        std::snprintf(b, sizeof b, " tok_s=%.2f ms_tok=%.2f vram_hit=%.4f cpu=%.3f ram=%.3f promo=%.3f engram_ms=%.3f",
                      ms > 0 ? 1000.0 * (double) produced / ms : 0.0, produced > 0 ? ms / (double) produced : 0.0,
                      ex > 0 ? 1.0 - (cx + sx + hx) / ex : 1.0, cx / fed, (double) (s.ram_experts - s_dec.ram_experts) / fed,
                      (double) (s.swaps - s_dec.swaps) / fed, (s.engram_ms - s_dec.engram_ms) / fed);
        out += b;
        if (sx > 0) {   // (the PCIe lane: experts a token streamed to the GPU)
            std::snprintf(b, sizeof b, " pcie=%.3f", sx / fed);
            out += b;
        }
        if (s.helper_held > 0) {   // (DS4_HELPER: experts a token the helper GPU computed, the CPU lane for it after
                                   //  a failure; the service thread's wait for it, ms a token; its pool; failed)
            std::snprintf(b, sizeof b, " helper=%.3f helper_fb=%.3f helper_ms=%.3f helper_held=%d helper_dead=%d",
                          hx / fed, (double) (s.helper_fallback - s_dec.helper_fallback) / fed,
                          (s.helper_ms - s_dec.helper_ms) / fed, s.helper_held, s.helper_dead);
            out += b;
        }
        const long long io = io_read_bytes();
        // the GLM engine's tier keys too: where the experts live (VRAM, RAM - copies or the read-ahead -, the rest on
        // the SSD) and, a token, the CPU lane's experts from the SSD (the storage's reads over an expert's bytes, at
        // most the CPU lane's) and from RAM (the rest of them)
        const strata::ds4::Model::Tiers tr = m.tiers();
        std::snprintf(b, sizeof b, " vram_used=%d ram_used=%d cpu_n=%d vram_gb=%.2f ram_gb=%.2f", tr.vram,
                      tr.ram, tr.cpu, tr.vram_gb, tr.ram_gb);
        out += b;
        std::snprintf(b, sizeof b, " eg_cache_rows=%lld eg_cache_cap=%lld eg_cache_row_bytes=%lld eg_lookup_rows=%lld",
                      (long long) s.eg_cached, (long long) s.eg_cache_cap, (long long) s.eg_cache_row_bytes,
                      (long long) s.eg_rows);
        out += b;
        std::snprintf(b, sizeof b, " eg_lookup_hits=%lld eg_lookup_reads=%lld", (long long) s.eg_hits,
                      (long long) s.eg_reads);
        out += b;
        if (slot_max > 0) {
            int slot_ram = 0, slot_disk = 0;
            slot_count(slot_ram, slot_disk);
            std::snprintf(b, sizeof b, " slot_ram=%d slot_disk=%d slot_ram_bytes=%llu slot_ram_cap=%llu", slot_ram,
                          slot_disk, (unsigned long long) slot_ram_bytes(), (unsigned long long) slot_ram_budget);
            out += b;
        }
        if (s.eg_pf_rows > 0) {
            std::snprintf(b, sizeof b,
                          " eg_prefetch_rows=%lld eg_prefetch_hits=%lld eg_prefetch_reads=%lld "
                          "eg_prefetch_io_ms=%.2f eg_prefetch_wait_ms=%.2f",
                          (long long) s.eg_pf_rows, (long long) s.eg_pf_hits, (long long) s.eg_pf_reads,
                          s.eg_pf_io_ms, s.eg_pf_wait_ms);
            out += b;
        }
        if (io >= 0 && io_dec >= 0) {
            const double disk = tr.cpu_blob > 0 ? std::min(cx / fed, (double) (io - io_dec) / tr.cpu_blob / fed) : 0.0;
            std::snprintf(b, sizeof b, " disk_mb=%.3f disk=%.3f ram_fetch=%.3f", (double) (io - io_dec) / 1e6 / fed,
                          disk, std::max(0.0, cx / fed - disk));
            out += b;
        } else if (cx == 0) {   // (the storage's reads not measured here - but no expert ran on the CPU lane)
            out += " disk=0 ram_fetch=0";
        }
        if (d_off > 0) {
            std::snprintf(b, sizeof b, " drafts=%.3f", (double) d_acc / (double) d_off);
            out += b;
        }
        if (s.skipped > s_dec.skipped) {   // (STRATA_DS4_DISK_SKIP_W: experts left out a token)
            std::snprintf(b, sizeof b, " skipped=%.3f", (double) (s.skipped - s_dec.skipped) / fed);
            out += b;
        }
        if (s.skip_on >= 0) {   // (... _AUTO: skipping now)
            std::snprintf(b, sizeof b, " skip_on=%d", s.skip_on);
            out += b;
        }
        return out;
    };
    // a request's prompt into the bound conversation (one conversation's requests and --parallel's alike): its
    // pictures read, the state it continues reused (in place, a checkpoint, or a kept conversation), the rest read.
    // Its lines carry `tag` (" <rid>" with --parallel), stop_now() is its STOP; false: it ended here (its ERR or
    // DONE written)
    struct Prompt {
        double t0 = 0;
        strata::ds4::Stats s0;   // (the request's own CPU lane share)
        size_t from = 0;         // the tokens the state already held
    };
    // --parallel: a prompt read beside the running requests (ds4_interleave.hpp; null: in one call, as before) -
    // whether to read it in parts of whole chunks (Model::feed_plan), the next part's chunks of those left, and what
    // runs between two parts (the part's ms, the tokens and chunks still to read; it binds this conversation again)
    struct Pace {
        std::function<bool()> parts;
        std::function<int(int)> chunks;
        std::function<void(double, int64_t, int)> between;
    };
    const auto prompt_in = [&](Req& r, bool geni, const std::string& tag, const std::function<bool()>& stop_now,
                               Prompt& P, const Pace* pace) -> bool {
        const double t0 = now_ms();
        P.t0 = t0;
        P.s0 = m.stats();
        std::vector<Pic> pics;
        {
            std::string e;
            if (geni && img_tok < 0) e = "this engine has no vision tower, so it cannot read pictures";
            else if (geni && read_pics(r.file, pics, e))
                for (const Pic& q : pics)
                    if (q.start < 0 || (size_t) q.start >= r.ids.size() || r.ids[(size_t) q.start] != img_tok)
                        e = "a picture's span does not start at an image token of the prompt";
            if (!e.empty()) {
                std::printf("ERR%s %s\n", tag.c_str(), e.c_str());
                std::printf("DONE%s 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", tag.c_str(), r.ids.size());
                std::fflush(stdout);
                return false;
            }
        }
        const auto& st = m.tokens();
        const size_t n_state = st.size();
        size_t c = 0;
        while (c < n_state && c < r.ids.size() && st[c] == r.ids[c]) ++c;
        // the same tokens with another picture (or none) do not share the state from that picture's span on
        for (const Pic& q : pics)
            if ((size_t) q.start < c && !pic_same(held, q)) c = (size_t) q.start;
        for (const Pic& q : held)
            if ((size_t) q.start < c && !pic_same(pics, q)) c = (size_t) q.start;
        size_t from = 0;
        bool continued = false;   // (continued: the conversation goes on - its anchor stays)
        // the conversation continues, reused in place - a conversation has to be in the model first: a fresh engine
        // (the start of a run, so every restart) holds 0 tokens, and every prompt "continues" that to token 0; taken
        // as a continuation, the take below would never run for the folder's conversations, the case they are for
        if (n_state > 0 && c == n_state && n_state < r.ids.size()) {
            from = n_state;
            continued = true;
        }
        else {
            // CONVERSATION SLOTS: this prompt does not continue the conversation in the model - that one is set aside
            // first (another one takes the model, and its state is lost otherwise), then the kept conversation this
            // prompt does continue is taken back - if it goes further back than the checkpoints the state itself holds
            if (slot_max > 0) slot_put("");
            beat();
            const int64_t limit = (int64_t) std::min(c, r.ids.size() - 1);
            const int64_t rs = m.restorable(limit);
            bool whole = false;
            const size_t back = slot_take(r.ids, pics, rs > 0 ? (size_t) rs : 0, whole);
            beat();
            if (back > 0) {
                from = back;
                continued = whole;
            } else if (rs >= 0 && m.restore(limit))
                from = (size_t) m.pos();
            else m.reset();
        }
        if (from == 0) held.clear();   // (a slot that came back set held itself: its pictures' marks, not only start
                                       //  and hash, so the spans already in the KV are not encoded again)
        if ((int64_t) r.ids.size() + std::max(0LL, r.max_new) > m.max_context()) {
            std::printf("ERR%s the prompt (%zu tokens) and max_new %lld are more than the context %lld\n", tag.c_str(),
                        r.ids.size(),
                        r.max_new, (long long) m.max_context());
            std::fflush(stdout);
            return false;
        }
        std::string err;
        bool ok = true;
        std::vector<strata::ds4::Image> imgs_fed;   // (set again after each pause between two parts)
        // the pictures whose spans the prompt reads (the rest the state holds): encoded (or the cache's), checked
        if (!pics.empty()) {
            std::vector<strata::ds4::Image> imgs;
            const double te = now_ms();
            int encoded = 0;
            m.vision_hold();   // (one borrowing of the tower's room for all of them, given back before the prompt)
            for (Pic& q : pics) {
                for (const Pic& x : held)
                    if (x.start == q.start && x.hash == q.hash) q.len = x.len;
                if (q.len > 0 && (size_t) q.start + q.len <= from) continue;
                std::shared_ptr<const std::vector<uint16_t>> rows = spans.get(m, q, err);
                if (!rows) {
                    ok = false;
                    break;
                }
                ++encoded;
                beat();
                q.len = rows->size() / (size_t) m.dim();
                bool fits = (size_t) q.start + q.len <= r.ids.size() - 1;
                for (size_t i = 0; fits && i < q.len; ++i) fits = r.ids[(size_t) q.start + i] == img_tok;
                if (!fits) {
                    err = "a picture's span (" + std::to_string(q.len) + " tokens at " + std::to_string(q.start) +
                          ") is not a run of image tokens in the prompt";
                    ok = false;
                    break;
                }
                imgs.push_back({q.start, rows});
            }
            if (!m.vision_release() && ok) {
                ok = false;
                err = "vision: the expert slots lent for the pictures were not read back as they were";
            }
            if (ok) {
                imgs_fed = imgs;
                m.set_images(std::move(imgs));
                std::fprintf(stderr, "maya ds4 serve: %zu pictures, %d read (%.0f ms)\n", pics.size(), encoded,
                             now_ms() - te);
            }
            for (Pic& q : pics) q.rgb = {};   // (only the start, hash and span stay)
        }
        // the prompt in windows (PP <read> <total> <ms> <tok/s> <reuse> every 64 tokens), a checkpoint before its last
        // token and one after it: the next turn re-renders this one, and shares its start up to one of them
        const size_t total = r.ids.size();
        // the restore points the request asks for (marks= / shared=; ds4_marks.hpp picks the ones this read reaches):
        // the read stops at each and keeps the state there.  Each piece is read as a run of its own, so a long piece's
        // skip (MAYA_DS4_PROMPT_SKIP) runs every layer on its own last ~2,700 rows - the state kept at the point is the
        // one a whole read leaves (the later layers' window rings included), at the cost of up to that many rows
        // through the layers the skip leaves out (V4.1: 20-39) that one run would have skipped - as many as lie
        // between the point and the prompt's end, when that is less
        std::vector<strata::ds4::MarkAt> cuts;
        if ((!r.marks.empty() || r.shared >= 0) && m.marks_cap() > 0) {
            std::vector<std::pair<int64_t, int64_t>> spans;
            for (const Pic& q : pics) spans.push_back({q.start, (int64_t) q.len});
            cuts = strata::ds4::mark_plan(r.marks, r.shared, (int64_t) from, (int64_t) total, m.marks_cap(), 16, spans);
            static bool said = false;
            if (!said)
                std::fprintf(stderr, "maya ds4 serve: restore points a request asks for: up to %d a conversation, "
                                     "%.1f MB of RAM each (MAYA_DS4_RESTORE_POINTS)\n",
                             m.marks_cap(), (double) m.check_bytes() / 1e6);
            said = true;
        }
        size_t next_cut = 0;
        if (!cuts.empty() && cuts[0].pos == (int64_t) from) m.mark(cuts[next_cut++].shared);   // (the state as it is)
        size_t done = from, next_pp = from + 64;
        int64_t beat_at = -1;   // (the run's progress at the last beat)
        // (a run of chunks: each one through every layer)
        const std::function<void(int64_t)> progress = [&](int64_t d) {
            // (a beat only when it moved: the run calls this every 50 ms with what is done so far - a wedged chunk run
            //  repeats its last value for hours, #539, and must not look alive to the watchdog)
            if (d + (int64_t) done != beat_at) {
                beat_at = d + (int64_t) done;
                beat();
            }
            const double ms = now_ms() - t0;
            std::printf("PP%s %zu %zu %.1f %.1f %zu\n", tag.c_str(), done + (size_t) d, r.ids.size(), ms,
                        ms > 0 ? 1000.0 * (double) (done + (size_t) d - from) / ms : 0.0, from);
            std::fflush(stdout);
        };
        m.set_progress(progress);
        // (DS4_LAST_WINDOW=1: the last token in a window of its own after the checkpoint, as before - for A/B runs)
        static const bool last_window = std::getenv("DS4_LAST_WINDOW") != nullptr;
        // a STOP while the prompt is read ends it there (the engine's run of chunks at a layer or chunk, windows between
        // feeds) and goes back to the state before this prompt - kept here first, in checkpoint 1, which the prompt's
        // own end overwrites as before - so the next request still shares the conversation up to it
        if (from > 0 && from < total) m.checkpoint(1);
        m.set_cancel(stop_now);
        bool stopped = false;
        // --parallel (pace): the prompt's run of chunks in parts - the plan of the one call it would be
        // (Model::feed_plan), so the same chunks, the same skip and the same arithmetic - and between two parts the
        // running requests' rounds, or a much shorter request let in first (pace->between); a prompt in windows of 16
        // (no chunked path) pauses between its windows
        strata::ds4::FeedPlan plan;
        size_t plan_from = 0;   // (where the plan's run starts)
        int plan_k = 0;         // (its next chunk)
        int parts = 0, pauses = 0;
        double paused_ms = 0;
        while (ok && done < total) {
            if (stop_now()) {
                stopped = true;
                break;
            }
            size_t n = std::min<size_t>((size_t) m.feed_unit(), total - done);
            if (next_cut < cuts.size() && (size_t) cuts[next_cut].pos < done + n)   // (a restore point: the read
                n = (size_t) cuts[next_cut].pos - done;                             //  stops there)
            // (the prompt's end: the last token in the same run, the checkpoint before it kept on the way)
            const bool fin = done + n == total;
            const double tp = now_ms();
            if (pace != nullptr && plan.ends.empty() && !(fin && last_window) && pace->parts() &&
                m.feed_plan((int) n, fin ? 1 : 0, fin ? 1 : -1, plan)) {
                plan_from = done;
                plan_k = 0;
            }
            if (!plan.ends.empty()) {   // (the plan's next chunks: the last part takes the logits and the checkpoint)
                const int nk = (int) plan.ends.size(), a = plan_k > 0 ? plan.ends[(size_t) plan_k - 1] : 0;
                const int k1 = plan_k + std::max(1, std::min(nk - plan_k, pace->chunks(nk - plan_k)));
                ok = m.feed_part(plan, r.ids.data() + plan_from, plan_k, k1, err);
                n = (size_t) (plan.ends[(size_t) k1 - 1] - a);
                plan_k = k1;
                if (plan_k == nk) plan = strata::ds4::FeedPlan{};
            } else if (fin && last_window) {
                ok = (n < 2 || m.feed(r.ids.data() + done, (int) n - 1, 0, err));
                if (ok) m.checkpoint(1);
                ok = ok && m.feed(r.ids.data() + done + n - 1, 1, 1, err);
            } else
                ok = fin ? m.feed_mark(r.ids.data() + done, (int) n, 1, 1, err)
                         : m.feed(r.ids.data() + done, (int) n, 0, err);
            done += n;
            ++parts;
            beat();
            if (ok && next_cut < cuts.size() && done == (size_t) cuts[next_cut].pos) m.mark(cuts[next_cut++].shared);
            if (ok && (done >= next_pp || done == total)) {
                const double ms = now_ms() - t0;
                std::printf("PP%s %zu %zu %.1f %.1f %zu\n", tag.c_str(), done, r.ids.size(), ms,
                            ms > 0 ? 1000.0 * (double) (done - from) / ms : 0.0, from);
                std::fflush(stdout);
                next_pp = done + 64;
            }
            if (ok && done < total && pace != nullptr && pace->parts()) {
                // (between two parts the prompt's own callbacks and pictures are off: other conversations step and
                //  read; then this conversation is bound again and they are back)
                m.set_cancel(nullptr);
                m.set_images({});
                const double tb = now_ms();
                const int left_chunks = !plan.ends.empty() ? (int) plan.ends.size() - plan_k
                                                           : (int) ((total - done + 15) / 16);
                pace->between(tb - tp, (int64_t) (total - done), left_chunks);
                paused_ms += now_ms() - tb;
                ++pauses;
                m.set_images(imgs_fed);
                m.set_progress(progress);
                m.set_cancel(stop_now);
            }
        }
        m.set_cancel(nullptr);
        m.set_progress(nullptr);   // (its lambda reads this read's locals)
        m.set_images({});
        if (pauses > 0)
            std::fprintf(stderr, "maya ds4 serve: request%s: the prompt was read in %d parts, %.0f ms between them for "
                                 "the others\n", tag.c_str(), parts, paused_ms);
        if (stopped || (!ok && m.cancelled())) {
            const bool back = from > 0 && m.restore((int64_t) from) && (size_t) m.pos() == from;
            if (!back) {
                m.reset();
                held.clear();
            }
            std::fprintf(stderr, "maya ds4 serve: stopped while reading the prompt of %zu tokens, after %.0f ms%s\n",
                         total, now_ms() - t0, back ? "; the conversation before it is kept" : "");
            std::printf("DONE%s 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0\n", tag.c_str(), r.ids.size(), now_ms() - t0, back ? from : (size_t) 0);
            std::fflush(stdout);
            return false;
        }
        if (!ok) {
            std::fprintf(stderr, "ds4: %s\n", err.c_str());
            std::printf("ERR%s %s\n", tag.c_str(), err.c_str());
            std::printf("DONE%s 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", tag.c_str(), r.ids.size());
            std::fflush(stdout);
            m.reset();
            held.clear();
            return false;
        }
        {   // the pictures the state holds: the prompt's, and those the state kept short of `from` (a turn that adds no
            // picture still holds the ones the history above it has)
            std::vector<Pic> nh;
            nh.reserve(held.size() + pics.size());
            for (const Pic& x : held)
                if (x.len > 0 && (size_t) x.start + x.len <= from && !pic_same(pics, x)) nh.push_back(x);
            for (const Pic& x : pics) nh.push_back(x);
            held = std::move(nh);
        }
        m.checkpoint(0);
        if (!continued) m.pin();   // (a new run of prompts: where it may be rendered again from)
        P.from = from;
        return true;
    };
    const bool multi = m.seq_count() > 1;
    while (!multi) {   // one conversation: a request at a time
        g_busy.store(false);   // (waiting for a request: no watchdog)
        if (!in.next(line) || line == "QUIT") break;
        beat();
        g_busy.store(true);
        const bool geni = line.rfind("GENI ", 0) == 0;
        if (!geni && line.rfind("GEN ", 0) != 0) {
            std::printf("ERR unsupported command '%s' (GEN, GENI, STOP and QUIT)\n", line.substr(0, line.find(' ')).c_str());
            std::fflush(stdout);
            continue;
        }
        in.stop.store(false);
        Req r;
        if (!parse_gen(line, r, geni)) {
            std::printf("ERR expected: GEN <max_new> [keys] <id,id,...> or GENI <max_new> [keys] <file> <id,id,...>\n");
            std::fflush(stdout);
            continue;
        }
        m.begin_request();   // (DISK_SKIP_AUTO: between requests only)
        if (round_log().f != nullptr) round_log().tag = "r" + std::to_string(++log_req);   // (DS4_ROUND_LOG)
        // a request's own seed: its draws from the start of its stream, so the same seed gives the same answer (the
        // counter ran on across requests: a seed was not reproducible); without one the stream runs on as before
        if (r.seed != 0) counter = 0;
        Prompt P;
        if (!prompt_in(r, geni, "", [&in] { return in.stop.load(); }, P, nullptr)) continue;
        const double t0 = P.t0;
        const strata::ds4::Stats s0 = P.s0;
        const size_t from = P.from;
        std::string err;
        const double t1 = now_ms();
        const strata::ds4::Stats sd = m.stats();   // (the decode's own counters: STAT lines, DONE's hit fields)
        long long produced = 0;
        const char* finish = "length";
        const long long limit = r.max_new > 0 ? r.max_new : m.max_context() - (long long) r.ids.size();
        long long d_off = 0, d_acc = 0;
        bool cool = false;
        if (dump_prefix != nullptr) {
            if (dump_f != nullptr) std::fclose(dump_f);
            dump_f = std::fopen((std::string(dump_prefix) + "." + std::to_string(++n_req) + ".f32").c_str(), "wb");
            dump_rows();
        }
        // this request's penalties (a repeat penalty of 0 or below, or a value that is not a number: neutral) for every
        // pick of its answer, the drafts' verify included
        m.set_penalties(r.pen_n < 0 ? INT_MAX : r.pen_n,
                        std::isfinite(r.pen_repeat) && r.pen_repeat > 0.0f ? r.pen_repeat : 1.0f,
                        std::isfinite(r.pen_freq) ? r.pen_freq : 0.0f, std::isfinite(r.pen_present) ? r.pen_present : 0.0f);
        int tok = m.sample(r.temperature, r.top_p, r.top_k, r.min_p, r.seed, counter++);
        std::vector<int> got;
        if (lk) lk->start(m.tokens());
        const long long lk_w0 = lk ? lk->windows : 0, lk_o0 = lk ? lk->offered : 0, lk_a0 = lk ? lk->accepted : 0;
        // STAT for the Monitor every 8 tokens and at the end, over this answer so far: the GLM engine's keys (speed, the
        // VRAM hit, promotions a token) and DeepSeek's own - experts a token on the CPU lane, MB a token the storage
        // read (Linux), the n-gram memory's (Engram) read time a token, the drafts' acceptance
        const strata::ds4::Stats s_dec = m.stats();
        const long long io_dec = io_read_bytes();
        auto stat_line = [&]() {
            std::printf("%s\n", stat_text(t1, s_dec, io_dec, produced, d_off, d_acc).c_str());
        };
        // (#363: the budget was the GLM engine's only - DeepSeek reasoned past it to max_tokens.  A round never runs
        //  past the budget, and the round at it is a plain step whose next token, not fed yet, becomes think_end)
        bool in_think = r.think_budget > 0 && r.think_end >= 0;
        for (;;) {
            if (in.stop.load()) {
                finish = "cancel";
                break;
            }
            if (tok < 0) {   // (a sample's CUDA error, printed by the sampler: no T -1 to the client)
                std::fprintf(stderr, "ds4: the sample failed\n");
                finish = "cancel";
                break;
            }
            std::printf("T %d\n", tok);
            beat();
            ++produced;
            if (lk) lk->sfx.append(tok);
            if (in_think && tok == r.think_end) in_think = false;   // (closed: by the model, or by the budget)
            if (produced % 8 == 0) stat_line();
            std::fflush(stdout);
            bool end = false;
            for (int e : eos) end = end || tok == e;
            if (end) {
                finish = "stop";
                break;
            }
            if (produced >= limit || m.pos() >= m.max_context()) break;
            if (got.empty()) {   // the next round: the drafts (if any) and the token after them
                int room = (int) std::min<long long>(limit - produced, m.max_context() - m.pos());
                const bool close_think = in_think && produced >= r.think_budget;
                if (close_think) room = 1;
                else if (in_think) room = (int) std::min<long long>(room, r.think_budget - produced);
                int off = 0, acc = 0;
                // a round whose drafts were all below the confidence floor: the next one does without the drafter
                // (a request's spec / dspark_conf keys over the engine's: the speed calibration measures both ways)
                const bool try_spec = (r.spec < 0 ? spec : r.spec != 0) && !cool;
                if (!round(m, tok, room, try_spec, r.conf >= 0 ? r.conf : conf_min, r.temperature, r.top_p, r.top_k,
                           r.min_p, r.seed, counter, got,
                           off, acc, err, r.cum, r.lookup == 0 ? nullptr : lk)) {
                    std::fprintf(stderr, "ds4: %s\n", err.c_str());
                    finish = "cancel";
                    break;
                }
                d_off += off;
                d_acc += acc;
                cool = try_spec && off == 0;
                if (close_think && !got.empty()) got.back() = r.think_end;   // (room 1: one token, not fed yet)
                std::reverse(got.begin(), got.end());
                dump_rows();
            }
            tok = got.back();
            got.pop_back();
        }
        if (dump_f != nullptr) std::fflush(dump_f);
        // (a stop within accepted drafts leaves the state past it: the next prompt shares less with it, and goes back to
        //  the checkpoint at this prompt's end)
        stat_line();
        const double t2 = now_ms();
        const strata::ds4::Stats& se = m.stats();   // (hits / lookups: the decode's routed experts in VRAM / all)
        std::printf("DONE %lld %zu %.1f %.1f %s %lld %lld %zu %lld %lld\n", produced, r.ids.size(), t1 - t0, t2 - t1,
                    finish, d_acc, d_off, from, (long long) ((se.experts - sd.experts) - (se.cpu_experts - sd.cpu_experts)),
                    (long long) (se.experts - sd.experts));
        std::fflush(stdout);
        const auto& s = m.stats();
        std::fprintf(stderr,
                     "maya ds4 serve: prompt %zu tokens = %zu reused + %zu read in %.0f ms (%.1f tok/s), %lld generated "
                     "in %.0f ms (%.2f tok/s); CPU lane %.1f%% of the experts\n",
                     r.ids.size(), from, r.ids.size() - from, t1 - t0,
                     t1 > t0 ? 1000.0 * (double) (r.ids.size() - from) / (t1 - t0) : 0.0, produced, t2 - t1,
                     t2 > t1 ? 1000.0 * (double) produced / (t2 - t1) : 0.0,
                     s.experts > s0.experts ? 100.0 * (double) (s.cpu_experts - s0.cpu_experts) / (double) (s.experts - s0.experts)
                                            : 0.0);
        if (d_off > 0) std::fprintf(stderr, "maya ds4 serve: drafts %lld of %lld kept; %s\n", d_acc, d_off,
                                    per_position(s_dec, s).c_str());
        // (DS4_DRAFT_VRAM, read as STRATA_DS4_DRAFT_VRAM or DS4_DRAFT_VRAM: the cap's misses - this request's decode,
        //  the drafter's picks)
        if (std::getenv("STRATA_DS4_DRAFT_VRAM") != nullptr || std::getenv("DS4_DRAFT_VRAM") != nullptr)
            std::fprintf(stderr, "maya ds4 serve: drafter experts: %lld picks, %lld of them on the CPU lane (this "
                                 "request's decode; DS4_DRAFT_VRAM)\n",
                         (long long) (s.draft_experts - s_dec.draft_experts),
                         (long long) (s.draft_cpu - s_dec.draft_cpu));
        if (s.hr_maps + s.hr_fails + s.hr_abandoned + s.hr_hits > 0 ||
            std::getenv("STRATA_DS4_HELPER_RECENT") != nullptr || std::getenv("DS4_HELPER_RECENT") != nullptr)
            std::fprintf(stderr, "maya ds4 serve: helper recent slots: %lld picks served, %lld mapped, %lld let go, "
                                 "%lld fills not done, %lld left unmapped (this request)\n",
                         (long long) (s.hr_hits - s_dec.hr_hits), (long long) (s.hr_maps - s_dec.hr_maps),
                         (long long) (s.hr_evicts - s_dec.hr_evicts), (long long) (s.hr_fails - s_dec.hr_fails),
                         (long long) (s.hr_abandoned - s_dec.hr_abandoned));
        if (lk && lk->windows > lk_w0)
            std::fprintf(stderr, "maya ds4 serve: prompt lookup: %lld rounds, %lld of its %lld drafts kept (in the above)\n",
                         lk->windows - lk_w0, lk->accepted - lk_a0, lk->offered - lk_o0);
        prof_line(s_dec, s);   // DS4_PROF: this request's decode (the CLI's line, a request at a time)
        m.save_profile();
    }
    // ---- SEVERAL CONVERSATIONS AT ONCE (--parallel <n>: READY ... multi=<n>).  GEN lines carry rid=<id>, and every
    // line a request causes carries it after its keyword (T <rid> <id>, PP <rid> ..., DONE <rid> ..., ERR <rid> ...);
    // STOP <rid> ends that request, a bare STOP every one.  Each request runs in one of the n conversations the model
    // holds (Model::seq_bind): the idle one whose state its prompt reuses the most, else an empty one, else the least
    // recently used (whose conversation is set aside first with MAYA_DS4_SLOT_KEEP, as one conversation's is).  A new
    // request's prompt - its pictures too: the tower borrows its room and gives it back before the prompt, with every
    // other request between two of its tokens - is read while the others wait (with MAYA_DS4_PROMPT_SHARE they decode
    // between its chunks, and with MAYA_DS4_PROMPT_YIELD a much shorter request goes first: ds4_interleave.hpp); then
    // each running request writes a token a step.  A request alone decodes in its rounds as with one conversation
    // (DSpark's drafts, the prompt lookup); two or more in rounds together - each one's DSpark drafts (its share of a
    // window), every one's token and drafts through the model at once (Model::step_multi_spec, verify_multi), or with
    // no drafts a token of each (Model::step_multi).  With several running, each writes all of its round's tokens at
    // once, so every one needs the model in the same step again.
    if (multi) {
        struct Act {
            Req r;
            Prompt P;
            int seq = 0;
            double t1 = 0;
            strata::ds4::Stats sd;
            long long produced = 0, limit = 0, d_off = 0, d_acc = 0;
            const char* finish = "length";
            bool cool = false, in_think = false, done = false;
            unsigned long long counter = 0;   // (its own: its picks the same alone or beside others)
            int tok = -1;                     // written next, not fed yet
            std::vector<int> got;             // a round's tokens still to write, the last first
        };
        const int nseq = m.seq_count();
        std::vector<long long> seq_rid((size_t) nseq, -1);   // the request in each conversation (-1: idle)
        std::vector<uint64_t> seq_used((size_t) nseq, 0);    // (the least recently used goes first)
        std::vector<std::vector<Pic>> held_of((size_t) nseq);   // the pictures each one holds (`held`: the bound one's)
        uint64_t use_clock = 0;
        std::vector<std::unique_ptr<Act>> acts;
        std::deque<std::string> waiting;   // GEN lines no conversation is free for yet
        long long lk_owner = -1;           // the request the prompt lookup's history follows (its rounds alone)
        // the Monitor's counters: over everything written since the engine was last idle
        double busy_t1 = 0;
        long long busy_n = 0, busy_io = -1;
        strata::ds4::Stats busy_s;
        std::string err;
        std::fprintf(stderr, "maya ds4 serve: %d conversations at once (--parallel): GEN needs rid=<id>; STOP <rid>\n",
                     nseq);
        const auto bind = [&](int s) {
            const int b = m.seq_bound();
            if (s == b) return;
            held_of[(size_t) b] = std::move(held);
            m.seq_bind(s);
            held = std::move(held_of[(size_t) s]);
            held_of[(size_t) s].clear();
        };
        const auto penalties = [&](const Req& r) {
            m.set_penalties(r.pen_n < 0 ? INT_MAX : r.pen_n,
                            std::isfinite(r.pen_repeat) && r.pen_repeat > 0.0f ? r.pen_repeat : 1.0f,
                            std::isfinite(r.pen_freq) ? r.pen_freq : 0.0f,
                            std::isfinite(r.pen_present) ? r.pen_present : 0.0f);
        };
        // a request's next token: the thinking budget closes its reasoning (as one conversation's round at the budget)
        const auto close_think = [](const Act& a) { return a.in_think && a.produced >= a.r.think_budget; };
        const auto finish_act = [&](Act& a) {
            const double t2 = now_ms();
            const strata::ds4::Stats& se = m.stats();   // (the engine's over the request's decode, the others' too)
            std::printf("DONE %lld %lld %zu %.1f %.1f %s %lld %lld %zu %lld %lld\n", a.r.rid, a.produced, a.r.ids.size(),
                        a.t1 - a.P.t0, t2 - a.t1, a.finish, a.d_acc, a.d_off, a.P.from,
                        (long long) ((se.experts - a.sd.experts) - (se.cpu_experts - a.sd.cpu_experts)),
                        (long long) (se.experts - a.sd.experts));
            std::fflush(stdout);
            std::fprintf(stderr,
                         "maya ds4 serve: request %lld: prompt %zu tokens = %zu reused + %zu read in %.0f ms (%.1f tok/s), "
                         "%lld generated in %.0f ms (%.2f tok/s, %zu conversations running; drafts %lld of %lld kept)\n",
                         a.r.rid, a.r.ids.size(), a.P.from, a.r.ids.size() - a.P.from, a.t1 - a.P.t0,
                         a.t1 > a.P.t0 ? 1000.0 * (double) (a.r.ids.size() - a.P.from) / (a.t1 - a.P.t0) : 0.0,
                         a.produced, t2 - a.t1, t2 > a.t1 ? 1000.0 * (double) a.produced / (t2 - a.t1) : 0.0, acts.size(),
                         a.d_acc, a.d_off);
            seq_rid[(size_t) a.seq] = -1;
            seq_used[(size_t) a.seq] = ++use_clock;
            in.forget(a.r.rid);
            if (lk_owner == a.r.rid) lk_owner = -1;
        };
        // the input between steps: requests and QUIT (STOP <rid> is the reader's); a line that is not a request is
        // answered at once
        bool quit = false;
        const auto poll_input = [&] {
            std::string ln;
            bool end = false;
            while (in.poll(ln, end)) {
                if (ln == "QUIT") quit = true;
                else waiting.push_back(ln);
            }
            for (size_t i = 0; i < waiting.size();) {   // (not GEN: answered at once)
                if (waiting[i].rfind("GEN ", 0) == 0 || waiting[i].rfind("GENI ", 0) == 0) {
                    ++i;
                    continue;
                }
                std::printf("ERR unsupported command '%s' (GEN, GENI, STOP and QUIT)\n",
                            waiting[i].substr(0, waiting[i].find(' ')).c_str());
                std::fflush(stdout);
                waiting.erase(waiting.begin() + (long) i);
            }
        };
        // the conversation for a prompt: the idle one it reuses the most of (its whole state continued, else a
        // checkpoint it can go back to: u, the tokens reused), else an empty one, else the least recently used; -1:
        // none idle.  (It binds each idle one to look.)
        const auto pick_seq = [&](const std::vector<int>& ids, int64_t& best_u) -> int {
            int best = -1;
            best_u = -1;
            for (int s = 0; s < nseq; ++s) {
                if (seq_rid[(size_t) s] >= 0) continue;
                bind(s);
                const auto& st = m.tokens();
                size_t c = 0;
                while (c < st.size() && c < ids.size() && st[c] == ids[c]) ++c;
                int64_t u = 0;
                if (!st.empty() && c == st.size() && st.size() < ids.size()) u = (int64_t) st.size();
                else if (c > 0) u = std::max<int64_t>(0, m.restorable((int64_t) std::min(c, ids.size() - 1)));
                const bool better = best < 0 || u > best_u ||
                                    (u == best_u && (st.empty() != m.seq_tokens(best).empty()
                                                         ? st.empty()
                                                         : seq_used[(size_t) s] < seq_used[(size_t) best]));
                if (better) best = s, best_u = u;
            }
            return best;
        };
        // one step of the running requests (the loop's, and between two parts of a prompt read beside them)
        const auto step_all = [&] {
            // each running request writes its next token; those whose next one the model must make go below (with
            // several running, a round's tokens all at once: every one then needs the model in the same step, so
            // their rounds go through it together)
            std::vector<Act*> need;
            const long long n0 = busy_n;
            const bool several = acts.size() > 1;
            for (auto& ap : acts) {
                Act& a = *ap;
                for (;;) {
                    if (in.stopped(a.r.rid)) {
                        a.finish = "cancel";
                        a.done = true;
                        break;
                    }
                    if (a.tok < 0) {   // (a sample's CUDA error, printed by the sampler: no T -1 to the client)
                        std::fprintf(stderr, "ds4: the sample failed\n");
                        a.finish = "cancel";
                        a.done = true;
                        break;
                    }
                    std::printf("T %lld %d\n", a.r.rid, a.tok);
                    beat();
                    ++a.produced;
                    ++busy_n;
                    if (lk && lk_owner == a.r.rid) lk->sfx.append(a.tok);
                    if (a.in_think && a.tok == a.r.think_end) a.in_think = false;
                    bool eos_tok = false;
                    for (int e : eos) eos_tok = eos_tok || a.tok == e;
                    if (eos_tok) {
                        a.finish = "stop";
                        a.done = true;
                        break;
                    }
                    if (a.produced >= a.limit || m.seq_pos(a.seq) >= m.max_context()) {
                        a.done = true;
                        break;
                    }
                    if (!a.got.empty()) {
                        a.tok = a.got.back();
                        a.got.pop_back();
                        if (several) continue;
                        break;
                    }
                    need.push_back(&a);
                    break;
                }
            }
            if (busy_n / 8 != n0 / 8) {   // the Monitor's line, every 8 tokens of all of them
                std::string st = stat_text(busy_t1, busy_s, busy_io, busy_n, 0, 0);
                st += " active=" + std::to_string(acts.size());
                std::printf("%s\n", st.c_str());
            }
            std::fflush(stdout);
            if (need.size() == 1) {   // alone (or the last of several still running): its rounds, as with one conversation
                Act& a = *need[0];
                bind(a.seq);
                if (round_log().f != nullptr) round_log().tag = "rid" + std::to_string(a.r.rid);   // (DS4_ROUND_LOG)
                int room = (int) std::min<long long>(a.limit - a.produced, m.max_context() - m.pos());
                const bool closing = close_think(a);
                if (closing) room = 1;
                else if (a.in_think) room = (int) std::min<long long>(room, a.r.think_budget - a.produced);
                // (the prompt lookup alone only: its history follows one request)
                Lookup* L = a.r.lookup == 0 || acts.size() > 1 ? nullptr : lk;
                if (L != nullptr && lk_owner != a.r.rid) {   // (its history: what it consumed, then the token written)
                    L->start(m.tokens());
                    L->sfx.append(a.tok);
                    lk_owner = a.r.rid;
                }
                penalties(a.r);
                int off = 0, acc = 0;
                const bool try_spec = (a.r.spec < 0 ? spec : a.r.spec != 0) && !a.cool;
                if (!round(m, a.tok, room, try_spec, a.r.conf >= 0 ? a.r.conf : conf_min, a.r.temperature, a.r.top_p,
                           a.r.top_k, a.r.min_p, a.r.seed, a.counter, a.got, off, acc, err, a.r.cum, L)) {
                    std::fprintf(stderr, "ds4: %s\n", err.c_str());
                    a.finish = "cancel";
                    a.done = true;
                } else {
                    a.d_off += off;
                    a.d_acc += acc;
                    a.cool = try_spec && off == 0;
                    if (closing && !a.got.empty()) a.got.back() = a.r.think_end;
                    std::reverse(a.got.begin(), a.got.end());
                    a.tok = a.got.back();
                    a.got.pop_back();
                }
            } else if (need.size() > 1) {   // several: a round of each through the model together
                // each one's drafts (DSpark, a conversation at a time: its own drafter rings) while their confidence
                // holds, its share of a window at most; then every one's token and drafts through the model at once
                // (step_multi_spec) and each its verdict (verify_multi).  No drafts this round: a token of each
                const int nn = (int) need.size(), share = m.verify_max() / nn;
                const auto cancel_all = [&] {
                    std::fprintf(stderr, "ds4: %s\n", err.c_str());
                    for (Act* a : need) {
                        a->finish = "cancel";
                        a->done = true;
                    }
                };
                std::vector<std::vector<int>> drafts((size_t) nn);
                std::vector<std::vector<float>> confs((size_t) nn);
                std::vector<char> tried((size_t) nn, 0), closing((size_t) nn, 0);
                std::vector<int> rooms((size_t) nn, 1), dsv, dys, dix;   // (the ones that draft: conversation, y, index)
                bool any = false, failed = false;
                for (int i = 0; i < nn; ++i) {
                    Act& a = *need[(size_t) i];
                    int room = (int) std::min<long long>(a.limit - a.produced, m.max_context() - m.seq_pos(a.seq));
                    closing[(size_t) i] = close_think(a);
                    if (closing[(size_t) i]) room = 1;
                    else if (a.in_think) room = (int) std::min<long long>(room, a.r.think_budget - a.produced);
                    rooms[(size_t) i] = room = std::min(room, share);
                    if (!((a.r.spec < 0 ? spec : a.r.spec != 0) && !a.cool && room > 1 && m.draft_block() > 0)) continue;
                    tried[(size_t) i] = 1;
                    dsv.push_back(a.seq);
                    dys.push_back(a.tok);
                    dix.push_back(i);
                }
                // their drafts: in one drafter pass where they fit one (Model::draft_multi), else one after another
                // (the drafter's argmax either way: verify_multi takes each draft as a point)
                const int nd = (int) dsv.size();
                const bool one_pass = nd > 1 && m.can_draft_multi(dsv.data(), nd);
                if (nd > 1 && !one_pass) {   // (more drafters than one pass holds - four with DSpark's block: a drafter
                                             //  pass each cost more than their drafts won, so a token of each)
                    for (int j : dix) tried[(size_t) j] = 0;
                } else if (one_pass) {
                    std::vector<std::vector<int>> dd;
                    std::vector<std::vector<float>> cc;
                    failed = !m.draft_multi(dsv.data(), dys.data(), nd, dd, cc, err);
                    for (int j = 0; j < nd && !failed; ++j) {
                        drafts[(size_t) dix[(size_t) j]] = std::move(dd[(size_t) j]);
                        confs[(size_t) dix[(size_t) j]] = std::move(cc[(size_t) j]);
                    }
                } else
                    for (int j = 0; j < nd && !failed; ++j) {
                        bind(dsv[(size_t) j]);
                        failed = !m.draft(dys[(size_t) j], drafts[(size_t) dix[(size_t) j]], confs[(size_t) dix[(size_t) j]], err);
                    }
                for (int i = 0; i < nn && !failed; ++i) {
                    if (!tried[(size_t) i]) continue;
                    Act& a = *need[(size_t) i];
                    const int room = rooms[(size_t) i];
                    std::vector<int>& d = drafts[(size_t) i];
                    const std::vector<float>& conf = confs[(size_t) i];
                    const float cmin = a.r.conf >= 0 ? a.r.conf : conf_min;
                    const float cum_min = a.r.cum >= 0 ? a.r.cum : dspark_cum_default();
                    int keep = 0;
                    float cum = 1.0f;
                    while (keep < (int) d.size() && keep < room - 1 && conf[(size_t) keep] >= cmin &&
                           (cum *= conf[(size_t) keep]) >= cum_min)
                        ++keep;
                    d.resize((size_t) keep);
                    any = any || keep > 0;
                }
                if (failed) cancel_all();
                else if (!any) {
                    std::vector<int> sv, tv;
                    for (Act* a : need) {
                        sv.push_back(a->seq);
                        tv.push_back(a->tok);
                    }
                    if (!m.step_multi(sv.data(), tv.data(), nn, err)) cancel_all();
                    else
                        for (int i = 0; i < nn; ++i) {
                            Act& a = *need[(size_t) i];
                            penalties(a.r);
                            a.tok = m.sample_multi(i, a.r.temperature, a.r.top_p, a.r.top_k, a.r.min_p, a.r.seed,
                                                   a.counter++);
                            if (closing[(size_t) i]) a.tok = a.r.think_end;
                            a.cool = tried[(size_t) i] != 0;   // (its drafter tried and kept none: rests a round)
                        }
                } else {
                    std::vector<std::vector<int>> rows((size_t) nn);
                    std::vector<const int*> tp((size_t) nn);
                    std::vector<int> sv((size_t) nn), lens((size_t) nn);
                    for (int i = 0; i < nn; ++i) {
                        rows[(size_t) i].push_back(need[(size_t) i]->tok);
                        rows[(size_t) i].insert(rows[(size_t) i].end(), drafts[(size_t) i].begin(), drafts[(size_t) i].end());
                        tp[(size_t) i] = rows[(size_t) i].data();
                        lens[(size_t) i] = (int) rows[(size_t) i].size();
                        sv[(size_t) i] = need[(size_t) i]->seq;
                    }
                    if (!m.step_multi_spec(sv.data(), tp.data(), lens.data(), nn, err)) cancel_all();
                    else
                        for (int i = 0; i < nn; ++i) {
                            Act& a = *need[(size_t) i];
                            const std::vector<int>& d = drafts[(size_t) i];
                            penalties(a.r);
                            int acc = 0, next = -1;
                            if (!m.verify_multi(i, d, a.r.temperature, a.r.top_p, a.r.top_k, a.r.min_p, a.r.seed,
                                                a.counter, acc, next, err)) {
                                cancel_all();
                                break;
                            }
                            a.counter += (unsigned long long) acc + 1;
                            a.d_off += (long long) d.size();
                            a.d_acc += acc;
                            a.cool = tried[(size_t) i] && d.empty();
                            a.got.assign(d.begin(), d.begin() + acc);
                            a.got.push_back(next);
                            if (closing[(size_t) i]) a.got.back() = a.r.think_end;
                            std::reverse(a.got.begin(), a.got.end());
                            a.tok = a.got.back();
                            a.got.pop_back();
                        }
                }
            }
            for (size_t i = 0; i < acts.size();) {
                if (!acts[i]->done) {
                    ++i;
                    continue;
                }
                std::unique_ptr<Act> a = std::move(acts[i]);
                acts.erase(acts.begin() + (long) i);
                finish_act(*a);
            }
        };
        // A prompt read beside them (ds4_interleave.hpp): MAYA_DS4_PROMPT_SHARE (STRATA_DS4_ / DS4_PROMPT_SHARE) - the
        // running requests decode between the prompt's chunks for this share of the time each chunk took (0, the
        // default: off - they wait for the whole prompt, as before); MAYA_DS4_PROMPT_YIELD=<n> - a long read gives way
        // at a part's end to a waiting request that needs at most half the tokens it has left (at most twice a read),
        // reading in parts of n chunks where nothing else decodes (0, the default: off).  The parts are whole chunks of
        // the one call's plan (Model::feed_plan), so the prompt's arithmetic is the same.  Opt-in until measured.
        strata::ds4::ReadPace pace_rule;
        {
            const auto env = [](const char* name) {
                const char* v = std::getenv((std::string("STRATA_") + name).c_str());
                return v != nullptr ? v : std::getenv(name);
            };
            if (const char* v = env("DS4_PROMPT_SHARE")) pace_rule.share = std::max(0.0, std::atof(v));
            if (const char* v = env("DS4_PROMPT_YIELD")) pace_rule.yield = std::max(0, std::atoi(v));
            if (pace_rule.on())
                std::fprintf(stderr, "maya ds4 serve: a prompt read beside running requests: they decode %.2f x a "
                                     "part's time between its parts (MAYA_DS4_PROMPT_SHARE); a long read gives way "
                                     "to a much shorter one: %s (MAYA_DS4_PROMPT_YIELD)\n",
                             pace_rule.share, pace_rule.yield > 0 ? ("in parts of " + std::to_string(pace_rule.yield) +
                                                                     " chunks").c_str() : "off");
        }
        int reading = 0;   // prompts being read (a read that gave way, and the one it let in)
        std::function<void(const std::string&, bool)> admit;   // (below: a read in parts lets a request in)
        // between two parts of request rid's prompt (conversation s): new lines taken in; a much shorter waiting
        // request let in first - read, its first token written (MAYA_DS4_PROMPT_YIELD; not from a read that was let in
        // itself); the running requests' steps for the share of the part's time (MAYA_DS4_PROMPT_SHARE); then s bound
        // again for the next part
        const auto between = [&](int s, long long rid, bool nested, int& yields, double part_ms, int64_t left,
                                 int left_chunks) {
            poll_input();
            bool admitted = false;
            if (!quit && pace_rule.yield > 0 && !nested && yields < pace_rule.yields_max && left_chunks > 1 &&
                !in.stopped(rid) && std::find(seq_rid.begin(), seq_rid.end(), -1LL) != seq_rid.end())
                for (size_t i = 0; i < waiting.size(); ++i) {
                    Req w;
                    if (!parse_gen(waiting[i], w, waiting[i].rfind("GENI ", 0) == 0) || w.rid < 0) continue;
                    int64_t u = 0;
                    if (pick_seq(w.ids, u) < 0) break;
                    const int64_t need = (int64_t) w.ids.size() - std::max<int64_t>(0, u);
                    if (!pace_rule.give_way(need, left, left_chunks, yields, true, nested)) continue;
                    const std::string g = waiting[i];
                    waiting.erase(waiting.begin() + (long) i);
                    ++yields;
                    std::fprintf(stderr, "maya ds4 serve: request %lld's prompt gives way with %lld tokens to read: "
                                         "request %lld (%lld to read) goes first (MAYA_DS4_PROMPT_YIELD)\n",
                                 rid, (long long) left, w.rid, (long long) need);
                    const size_t before = acts.size();
                    admit(g, true);
                    admitted = acts.size() > before;
                    break;
                }
            const double budget = pace_rule.budget_ms(part_ms), t0 = now_ms();
            for (int k = 0; pace_rule.decode_more(k, now_ms() - t0, budget, !acts.empty(), admitted); ++k) step_all();
            bind(s);
        };
        // a GEN line into a free conversation: its prompt read (the others wait, or run between its parts), its
        // first token picked; nested: let in between the parts of another read (it does not give way itself)
        admit = [&](const std::string& ln, bool nested) {
            const bool geni = ln.rfind("GENI ", 0) == 0;
            auto a = std::make_unique<Act>();
            Req& r = a->r;
            if (!parse_gen(ln, r, geni)) {
                std::printf("ERR expected: GEN <max_new> [keys] <id,id,...> or GENI <max_new> [keys] <file> <id,id,...>\n");
                std::fflush(stdout);
                return;
            }
            if (r.rid < 0) {
                std::printf("ERR this engine runs several conversations: GEN needs rid=<id>\n");
                std::fflush(stdout);
                return;
            }
            int64_t best_u = -1;
            const int best = pick_seq(r.ids, best_u);
            if (best < 0) {   // (the server gives each conversation one request: a request over that waits)
                waiting.push_front(ln);
                return;
            }
            bind(best);
            seq_used[(size_t) best] = ++use_clock;
            // (DISK_SKIP_AUTO: between requests only - none runs now, and no read waits between its parts)
            if (acts.empty() && reading == 0) m.begin_request();
            const long long rid = r.rid;
            const std::string tag = " " + std::to_string(rid);
            seq_rid[(size_t) best] = rid;   // (its own while the prompt is read: a request let in goes elsewhere)
            int yields = 0;
            Pace pc;
            pc.parts = [&] { return pace_rule.parts(!acts.empty()); };
            pc.chunks = [&](int left) { return pace_rule.part_chunks(left, !acts.empty()); };
            pc.between = [&, best, rid, nested](double part_ms, int64_t left, int left_chunks) {
                between(best, rid, nested, yields, part_ms, left, left_chunks);
            };
            ++reading;
            const bool in_ok = prompt_in(r, geni, tag, [&in, rid] { return in.stopped(rid); }, a->P,
                                         pace_rule.on() ? &pc : nullptr);
            --reading;
            bind(best);
            if (!in_ok) {
                seq_rid[(size_t) best] = -1;
                in.forget(rid);
                return;
            }
            a->seq = best;
            a->t1 = now_ms();
            a->sd = m.stats();
            a->limit = r.max_new > 0 ? r.max_new : m.max_context() - (long long) r.ids.size();
            a->in_think = r.think_budget > 0 && r.think_end >= 0;
            penalties(r);
            a->tok = m.sample(r.temperature, r.top_p, r.top_k, r.min_p, r.seed, a->counter++);
            if (acts.empty()) {
                busy_t1 = now_ms();
                busy_n = 0;
                busy_s = m.stats();
                busy_io = io_read_bytes();
            }
            acts.push_back(std::move(a));
        };
        for (;;) {
            // the input between steps (waited for only when idle), then the waiting requests into free conversations
            if (acts.empty() && waiting.empty()) {
                std::string ln;
                g_busy.store(false);   // (waiting for a request: no watchdog)
                m.save_profile();
                if (quit || !in.next(ln) || ln == "QUIT") break;
                g_busy.store(true);
                beat();
                waiting.push_back(ln);
            }
            poll_input();
            while (!quit && !waiting.empty() &&
                   std::find(seq_rid.begin(), seq_rid.end(), -1LL) != seq_rid.end()) {
                const std::string g = waiting.front();
                waiting.pop_front();
                admit(g, false);
                beat();
            }
            if (quit) in.stop_all.store(true);
            step_all();
            if (acts.empty() && quit) break;
        }
    }
    // the engine ends (QUIT or the end of its input): the conversation in the model is set aside, and every one in
    // RAM is written to the folder, so the next run reads on from them
    if (slot_max > 0) {
        slot_put(" as the engine ended");
        size_t written = 0;
        while (true) {   // (a write can take the folder's least recently used conversation away: the list is asked again)
            size_t i = SIZE_MAX;
            for (size_t k = 0; k < slots.size(); ++k)
                if (slots[k].data) {
                    i = k;
                    break;
                }
            if (i == SIZE_MAX) break;
            if (slot_write_file(slots[i])) ++written;
            else slot_drop(i);   // (its state could not be written: there is nothing more to do at the engine's end)
        }
        if (written > 0) {
            int ram = 0, disk = 0;
            slot_count(ram, disk);
            char msg[256];
            std::snprintf(msg, sizeof msg, "wrote %zu conversation(s) to %s; %d there in all", written,
                          slot_dir.string().c_str(), disk);
            note(msg);
        }
    }
    m.save_profile();
    return 0;
}

// --multi-check (tests, with --parallel N): N conversations - the first len - 5 i tokens of the --ids prompt each -
// decode n greedy tokens alone (a plain step at a time, one conversation after another), then all of them together
// (Model::step_multi).  Each row's logits must be its own step's alone bit for bit, its tokens the same.  (The times
// are the steps' alone and together on tiers the first pass warmed; the server's real questions are the measure.)
// --profile-run FILE: an expert profile from a corpus - each line of FILE a run of ids (BOS + text, as --ppl's) fed
// from a fresh state, no logits but its last row; what the runs route is counted, and saved as it goes (every 10 lines
// or minute, and at the end) to --profile-out: this run's counts alone, while the pack's profile places the experts and
// stays as it was.  A line longer than the context is cut there.  tools/ds4/profile.py make takes such files
int profile_run(strata::ds4::Model& m, const std::string& path, const std::string& out) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "maya-ds4: --profile-run: cannot open %s\n", path.c_str());
        return 1;
    }
    std::string line, err;
    long long lines = 0, tokens = 0, cut = 0;
    const double t0 = now_ms();
    double saved = t0;
    while (std::getline(in, line)) {
        std::vector<int> seq = parse_ids(line);
        if (seq.empty()) continue;
        if (seq.size() > (size_t) m.max_context()) {
            seq.resize((size_t) m.max_context());
            ++cut;
        }
        m.reset();
        for (size_t done = 0; done < seq.size();) {
            const size_t k = std::min<size_t>((size_t) m.feed_unit(), seq.size() - done);
            if (!m.feed(seq.data() + done, (int) k, done + k == seq.size() ? 1 : 0, err)) {
                std::fprintf(stderr, "maya-ds4: --profile-run, line %lld: %s\n", lines + 1, err.c_str());
                m.save_profile();
                return 1;
            }
            done += k;
        }
        ++lines;
        tokens += (long long) seq.size();
        const double now = now_ms();
        std::fprintf(stderr, "profile-run %lld: %zu tokens, %lld in all, %.0f s, %.0f tok/s\n", lines, seq.size(), tokens,
                     (now - t0) / 1000.0, 1000.0 * (double) tokens / std::max(1.0, now - t0));
        if (lines % 10 == 0 || now - saved > 60000.0) {
            m.save_profile();
            saved = now;
        }
    }
    m.save_profile();
    std::fprintf(stderr, "profile-run: %lld lines, %lld tokens (%lld lines cut at the context) in %.0f s; the counts in %s "
                         "(python tools/ds4/profile.py make --pack <pack> --out <profile> %s ...)\n",
                 lines, tokens, cut, (now_ms() - t0) / 1000.0, out.c_str(), out.c_str());
    return lines > 0 ? 0 : 1;
}

// a prompt from a fresh state, its last row's logits kept (mark: the state before its last token kept in checkpoint 1)
bool feed_prompt(strata::ds4::Model& m, const std::vector<int>& ids, bool mark, std::string& err) {
    m.reset();
    for (size_t done = 0; done < ids.size();) {
        const size_t k = std::min<size_t>((size_t) m.feed_unit(), ids.size() - done);
        const bool fin = done + k == ids.size();
        if (!(fin && mark ? m.feed_mark(ids.data() + done, (int) k, 1, 1, err)
                          : m.feed(ids.data() + done, (int) k, fin ? 1 : 0, err)))
            return false;
        done += k;
    }
    return true;
}

// --spec-eval FILE --n N: the drafter's acceptance over prompts (a line of ids each, BOS and text as --ppl's), each from
// a fresh state, N tokens decoded after it (greedy, or --temperature / --seed) with every draft verified - no
// confidence floor or cut, no prompt lookup - so what a floor would have kept can be read off afterwards.  Per prompt
// and in all: rounds, tokens a round, the drafts kept per position and by the drafter's confidence; DS4_ROUND_LOG gets
// a line a round
int spec_eval(strata::ds4::Model& m, const std::string& path, int n, float temp, unsigned long long seed) {
    if (m.draft_block() <= 0) {
        std::fprintf(stderr, "maya-ds4: --spec-eval needs the DSpark drafter (a pack with it, --dspark on)\n");
        return 2;
    }
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "maya-ds4: --spec-eval: cannot open %s\n", path.c_str());
        return 1;
    }
    RoundLog& rl = round_log();
    rl.record = true;
    const int top_k = temp > 0 ? 64 : 1;
    const strata::ds4::Stats s0 = m.stats();
    long long prompts = 0, all_tok = 0, all_rounds = 0, all_off = 0, all_acc = 0;
    long long reach[10] = {}, kept[10] = {};   // drafts by the drafter's confidence (tenths): verified with all before
                                              // them kept, and kept
    std::string line, err;
    const double t_all = now_ms();
    while (std::getline(in, line)) {
        const std::vector<int> ids = parse_ids(line);
        if (ids.empty()) continue;
        if ((long long) ids.size() + n + 16 > m.max_context()) {
            std::fprintf(stderr, "spec-eval: a prompt of %zu tokens and %d more do not fit the context: left out\n",
                         ids.size(), n);
            continue;
        }
        rl.tag = "p" + std::to_string(++prompts);
        if (!feed_prompt(m, ids, false, err)) {
            std::fprintf(stderr, "maya-ds4: --spec-eval, prompt %lld: %s\n", prompts, err.c_str());
            return 1;
        }
        unsigned long long counter = 0;
        int tok = m.sample(temp, 1.0f, top_k, 0.0f, seed, counter++);
        long long produced = 1, rounds = 0, off_p = 0, acc_p = 0;
        std::vector<int> got;
        const double t0 = now_ms();
        bool end = tok == 1;
        while (!end && produced < n) {
            int off = 0, acc = 0;
            if (!round(m, tok, (int) (n - produced), true, 0.0f, temp, 1.0f, top_k, 0.0f, seed, counter, got, off, acc,
                       err, 0.0f, nullptr)) {
                std::fprintf(stderr, "maya-ds4: --spec-eval, prompt %lld: %s\n", prompts, err.c_str());
                return 1;
            }
            ++rounds;
            off_p += off;
            acc_p += acc;
            for (int i = 0; rl.made > 0 && !rl.lookup && i < rl.verified && i < (int) rl.conf.size(); ++i) {
                const int b = std::min(9, std::max(0, (int) (rl.conf[(size_t) i] * 10.0f)));
                reach[b] += i <= rl.kept;
                kept[b] += i < rl.kept;
            }
            for (int t : got) {
                ++produced;
                if (t == 1) {
                    end = true;
                    break;
                }
            }
            tok = got.back();
        }
        const double ms = now_ms() - t0;
        std::fprintf(stderr, "spec-eval %lld: prompt %zu tokens, %lld written in %lld rounds (%.2f a round), %lld of %lld "
                             "drafts kept (%.0f%%), %.1f tok/s\n",
                     prompts, ids.size(), produced, rounds, rounds > 0 ? (double) (produced - 1) / (double) rounds : 0.0,
                     acc_p, off_p, off_p > 0 ? 100.0 * (double) acc_p / (double) off_p : 0.0,
                     ms > 0 ? 1000.0 * (double) (produced - 1) / ms : 0.0);
        all_tok += produced - 1;
        all_rounds += rounds;
        all_off += off_p;
        all_acc += acc_p;
    }
    rl.record = false;
    const strata::ds4::Stats& s1 = m.stats();
    const double nr = (double) std::max<int64_t>(1, s1.drafts_rounds - s0.drafts_rounds);
    std::printf("spec-eval: %lld prompts, %lld tokens after their first in %lld rounds (%.2f a round), %lld of %lld drafts "
                "kept (%.1f%%); drafting %.2f ms a round, verifying %.2f ms a round; %.0f s\n",
                prompts, all_tok, all_rounds, all_rounds > 0 ? (double) all_tok / (double) all_rounds : 0.0, all_acc,
                all_off, all_off > 0 ? 100.0 * (double) all_acc / (double) all_off : 0.0, (s1.draft_ms - s0.draft_ms) / nr,
                (s1.verify_ms - s0.verify_ms) / nr, (now_ms() - t_all) / 1000.0);
    std::printf("spec-eval %s\n", per_position(s0, s1).c_str());
    std::string bc = "spec-eval by the drafter's confidence (kept / reached):";
    for (int b = 0; b < 10; ++b) {
        if (reach[b] == 0) continue;
        char s[96];
        std::snprintf(s, sizeof s, " %.1f-%.1f %lld/%lld %.0f%%", b / 10.0, (b + 1) / 10.0, kept[b], reach[b],
                      100.0 * (double) kept[b] / (double) reach[b]);
        bc += s;
    }
    std::printf("%s\n", bc.c_str());
    return prompts > 0 ? 0 : 1;
}

// the chi-square distribution's upper tail at x with df degrees of freedom: Q(df / 2, x / 2), the regularized upper
// incomplete gamma function - its series below a + 1, its continued fraction (Lentz) above
double chi2_tail(double x, int df) {
    const double a = 0.5 * df, z = 0.5 * x;
    if (df <= 0 || z <= 0) return 1.0;
    const double front = std::exp(-z + a * std::log(z) - std::lgamma(a));
    if (z < a + 1.0) {
        double term = 1.0 / a, sum = term;
        for (int i = 1; i < 10000 && term > sum * 1e-15; ++i) {
            term *= z / (a + i);
            sum += term;
        }
        return std::min(1.0, std::max(0.0, 1.0 - front * sum));
    }
    const double tiny = 1e-300;
    double b = z + 1.0 - a, c = 1.0 / tiny, d = 1.0 / b, h = d;
    for (int i = 1; i < 10000; ++i) {
        const double an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if (std::fabs(d) < tiny) d = tiny;
        c = b + an / c;
        if (std::fabs(c) < tiny) c = tiny;
        d = 1.0 / d;
        h *= d * c;
        if (std::fabs(d * c - 1.0) < 1e-15) break;
    }
    return std::min(1.0, std::max(0.0, front * h));
}

// whether two samples of tokens come from one distribution: the two-sample chi-square over the tokens they drew (those
// whose expected count in the smaller sample is under 5 pooled into one group), its groups and p-value
struct Chi {
    double x = 0.0, p = 1.0;
    int groups = 0;
};
Chi two_sample(const std::vector<int>& a, const std::vector<int>& b) {
    std::map<int, std::pair<double, double>> n;
    for (int t : a) n[t].first += 1.0;
    for (int t : b) n[t].second += 1.0;
    const double na = (double) a.size(), nb = (double) b.size(), small = std::min(na, nb);
    std::vector<std::pair<double, double>> g;
    double ra = 0.0, rb = 0.0;
    for (const auto& kv : n) {
        const auto& c = kv.second;
        if ((c.first + c.second) * small / (na + nb) >= 5.0) {
            g.push_back(c);
        } else {
            ra += c.first;
            rb += c.second;
        }
    }
    if (ra + rb > 0.0) g.emplace_back(ra, rb);
    Chi r;
    if (na <= 0.0 || nb <= 0.0) return r;
    const double ka = std::sqrt(nb / na), kb = std::sqrt(na / nb);
    for (const auto& [x, y] : g) r.x += (ka * x - kb * y) * (ka * x - kb * y) / (x + y);
    r.groups = (int) g.size();
    r.p = chi2_tail(r.x, r.groups - 1);
    return r;
}

// --sampler-test N --ids I,J,...: speculative sampling against the model's own draws.  After the prompt and its greedy
// next token y, the token after y is drawn 20 N times from the model's own row (sample(): a plain step's draw), and N
// times as a drafted round writes it - the drafter's drafts (drawn from its own distribution with DS4_DSPARK_SAMPLE=1,
// set unless given; else its argmax) verified by verify()'s rule, the state taken back to the prompt each time
// (checkpoint 1, its last token fed again).  The two samples are compared with a chi-square test for four sampling
// chains, the last with penalties (its drafts are the argmax: points); a p-value under 0.001 is a mismatch (exit 1).
// The plain draws' two halves are the test's own reference.  A prompt whose next token is nearly certain is said: the
// test then sees little
int sampler_test(strata::ds4::Model& m, const std::vector<int>& ids, int n, unsigned long long seed) {
    if (m.draft_block() <= 0) {
        std::fprintf(stderr, "maya-ds4: --sampler-test needs the DSpark drafter (a pack with it, --dspark on)\n");
        return 2;
    }
    if (ids.size() < 2 || n < 50) {
        std::fprintf(stderr, "maya-ds4: --sampler-test N (50 or more) needs --ids of two tokens or more\n");
        return 2;
    }
    std::string err;
    const int64_t P = (int64_t) ids.size();
    if (!feed_prompt(m, ids, true, err)) {
        std::fprintf(stderr, "maya-ds4: --sampler-test: %s\n", err.c_str());
        return 1;
    }
    const int y = m.sample(0.0f, 1.0f, 1, 0.0f, 0, 0);
    // back to the prompt's end: the state before its last token, that token fed again (y's logits)
    const auto back = [&]() -> bool {
        if (!m.restore(P - 1) || m.pos() != P - 1) {
            err = "the state before the prompt's last token did not come back";
            return false;
        }
        return m.feed(ids.data() + P - 1, 1, 1, err);
    };
    struct Chain {
        float t, top_p, min_p;
        int top_k, pen_n;
        float rep, freq, pres;
    };
    const Chain chains[] = {{1.0f, 1.0f, 0.0f, 64, 0, 1.0f, 0.0f, 0.0f},
                            {0.8f, 0.95f, 0.0f, 20, 0, 1.0f, 0.0f, 0.0f},
                            {0.7f, 0.9f, 0.05f, 40, 0, 1.0f, 0.0f, 0.0f},
                            {0.8f, 0.95f, 0.0f, 20, 64, 1.3f, 0.1f, 0.1f}};
    int bad = 0, cases = 0;
    std::fprintf(stderr, "sampler-test: prompt %lld tokens, y = %d; %d plain draws and %d drafted rounds a chain\n",
                 (long long) P, y, 20 * n, n);
    for (const Chain& c : chains) {
        m.set_penalties(c.pen_n, c.rep, c.freq, c.pres);
        if (!back() || !m.step(y, true, err)) {
            std::fprintf(stderr, "maya-ds4: --sampler-test: %s\n", err.c_str());
            return 1;
        }
        std::vector<int> plain((size_t) 20 * n), spec;
        std::map<int, int> seen;
        for (size_t i = 0; i < plain.size(); ++i) {
            plain[i] = m.sample(c.t, c.top_p, c.top_k, c.min_p, seed, i);
            ++seen[plain[i]];
        }
        int top = 0;
        for (const auto& kv : seen) top = std::max(top, kv.second);
        long long first_kept = 0, no_draft = 0;
        bool drawn = false;
        for (int i = 0; i < n; ++i) {
            const uint64_t ctr = (1ull << 32) + (uint64_t) i * 64;   // (apart from the plain draws' and each other's)
            std::vector<int> drafts;
            std::vector<float> conf;
            int acc = 0, next = -1;
            if (!back() || !m.draft(y, drafts, conf, err, c.t, c.top_p, c.top_k, c.min_p, seed, ctr)) {
                std::fprintf(stderr, "maya-ds4: --sampler-test: %s\n", err.c_str());
                return 1;
            }
            drawn = drawn || m.drafts_sampled();
            if (drafts.empty()) {
                ++no_draft;
                continue;
            }
            if ((int) drafts.size() > m.verify_max() - 1) drafts.resize((size_t) (m.verify_max() - 1));
            if (!m.verify(y, drafts, c.t, c.top_p, c.top_k, c.min_p, seed, ctr, acc, next, err)) {
                std::fprintf(stderr, "maya-ds4: --sampler-test: %s\n", err.c_str());
                return 1;
            }
            spec.push_back(acc > 0 ? drafts[0] : next);
            first_kept += acc > 0;
        }
        const Chi x = two_sample(plain, spec);
        const std::vector<int> h0(plain.begin(), plain.begin() + (long) (plain.size() / 2)),
            h1(plain.begin() + (long) (plain.size() / 2), plain.end());
        const Chi ref = two_sample(h0, h1);
        const bool ok = x.p >= 1e-3;
        ++cases;
        bad += !ok;
        std::printf("sampler-test T %.2f top_k %d top_p %.2f min_p %.2f%s: %zu plain draws of %zu tokens (the most drawn "
                    "%.0f%%%s), %zu drafted rounds (%s drafts, the first kept %lld times%s): chi-square %.1f over %d "
                    "groups, p %.4f (the plain halves: p %.4f) - %s\n",
                    c.t, c.top_k, c.top_p, c.min_p, c.pen_n > 0 ? " penalties" : "", plain.size(), seen.size(),
                    100.0 * top / (double) plain.size(), top > 0.95 * (double) plain.size() ? ": nearly certain" : "",
                    spec.size(), drawn ? "drawn" : "argmax", first_kept,
                    no_draft > 0 ? (", " + std::to_string(no_draft) + " without drafts").c_str() : "", x.x, x.groups,
                    x.p, ref.p, ok ? "the same distribution" : "MISMATCH");
        std::fflush(stdout);
    }
    m.set_penalties(0, 1.0f, 0.0f, 0.0f);
    std::printf("sampler-test: %s\n", bad == 0 ? "speculative sampling gave the model's own draws in every chain"
                                                : (std::to_string(bad) + " of " + std::to_string(cases) +
                                                   " chains MISMATCH").c_str());
    return bad == 0 ? 0 : 1;
}

int check_multi(strata::ds4::Model& m, const std::vector<int>& ids, int n) {
    const int N = m.seq_count(), V = m.vocab();
    std::string err;
    std::vector<std::vector<int>> prompts((size_t) N);
    for (int i = 0; i < N; ++i) {
        const size_t len = ids.size() > (size_t) (5 * i + 2) ? ids.size() - (size_t) (5 * i) : ids.size();
        prompts[(size_t) i].assign(ids.begin(), ids.begin() + (long) len);
    }
    const auto prompt = [&](int i) -> bool {
        m.seq_bind(i);
        m.reset();
        const std::vector<int>& p = prompts[(size_t) i];
        for (size_t done = 0; done < p.size();) {
            const size_t k = std::min<size_t>((size_t) m.feed_unit(), p.size() - done);
            if (!m.feed(p.data() + done, (int) k, done + k == p.size() ? 1 : 0, err)) return false;
            done += k;
        }
        return true;
    };
    std::vector<std::vector<int>> alone((size_t) N), together((size_t) N);
    std::vector<std::vector<std::vector<float>>> lg_alone((size_t) N);
    std::vector<float> lg;
    double ms_alone = 0, ms_together = 0;
    for (int i = 0; i < N; ++i) {
        if (!prompt(i)) {
            std::fprintf(stderr, "multi check: %s\n", err.c_str());
            return 1;
        }
        int tok = m.sample(0.0f, 1.0f, 0, 0.0f, 0, 0);
        alone[(size_t) i].push_back(tok);
        const double t0 = now_ms();
        for (int k = 1; k < n; ++k) {
            if (!m.step(tok, true, err) || !m.logits(lg)) {
                std::fprintf(stderr, "multi check: %s\n", err.c_str());
                return 1;
            }
            lg_alone[(size_t) i].push_back(lg);
            tok = m.sample(0.0f, 1.0f, 0, 0.0f, 0, 0);
            alone[(size_t) i].push_back(tok);
        }
        ms_alone += now_ms() - t0;
    }
    for (int i = 0; i < N; ++i) {
        if (!prompt(i)) {
            std::fprintf(stderr, "multi check: %s\n", err.c_str());
            return 1;
        }
        together[(size_t) i].push_back(m.sample(0.0f, 1.0f, 0, 0.0f, 0, 0));
    }
    long long rows = 0, same_rows = 0;
    double max_d = 0;
    std::vector<int> sv((size_t) N), tv((size_t) N);
    for (int k = 1; k < n; ++k) {
        for (int i = 0; i < N; ++i) {
            sv[(size_t) i] = i;
            tv[(size_t) i] = together[(size_t) i].back();
        }
        const double t0 = now_ms();
        if (!m.step_multi(sv.data(), tv.data(), N, err) || !m.logits(lg) || lg.size() < (size_t) N * V) {
            std::fprintf(stderr, "multi check: %s\n", err.empty() ? "no logits" : err.c_str());
            return 1;
        }
        for (int i = 0; i < N; ++i) together[(size_t) i].push_back(m.sample_multi(i, 0.0f, 1.0f, 0, 0.0f, 0, 0));
        ms_together += now_ms() - t0;
        for (int i = 0; i < N; ++i) {   // (while the tokens agree: the same history, so the same logits are due)
            if (together[(size_t) i][(size_t) k - 1] != alone[(size_t) i][(size_t) k - 1]) continue;
            const float* a = lg_alone[(size_t) i][(size_t) k - 1].data();
            const float* b = lg.data() + (size_t) i * V;
            ++rows;
            if (std::memcmp(a, b, (size_t) V * 4) == 0) ++same_rows;
            else
                for (int v = 0; v < V; ++v) max_d = std::max(max_d, (double) std::fabs(a[v] - b[v]));
        }
    }
    bool tokens_same = true;
    for (int i = 0; i < N; ++i) {
        size_t d = 0;
        while (d < alone[(size_t) i].size() && alone[(size_t) i][d] == together[(size_t) i][d]) ++d;
        const bool same = d == alone[(size_t) i].size();
        tokens_same = tokens_same && same;
        std::printf("multi check: conversation %d (%zu prompt tokens): %d tokens %s", i, prompts[(size_t) i].size(), n,
                    same ? "the same alone and together\n" : "");
        if (!same) std::printf("differ from token %zu on\n", d);
    }
    const double steps = (double) (n - 1);
    std::printf("multi check: %d conversations x %d tokens: logits bit for bit in %lld of %lld rows (max |d| %.3g); "
                "tokens the same: %s; alone %.2f tok/s, together %.2f tok/s in all (x%.2f)\n",
                N, n, same_rows, rows, max_d, tokens_same ? "yes" : "NO",
                ms_alone > 0 ? 1000.0 * steps * N / ms_alone : 0.0, ms_together > 0 ? 1000.0 * steps * N / ms_together : 0.0,
                ms_together > 0 ? ms_alone / ms_together : 0.0);
    // ... then with the drafter, greedy, so the tokens alone are due again - every draft verified, whatever its
    // confidence, so rejected windows are undone often: each conversation alone (draft / verify, one conversation's
    // rounds), then together (step_multi_spec / verify_multi, as the server's rounds, each conversation's drafts its
    // share of a window) with the drafts made one conversation after another, then in one drafter pass (draft_multi)
    bool spec_same = true;
    double ms_spec_alone = 0;
    for (int pass = 0; pass < 3 && m.draft_block() > 0; ++pass) {
        std::vector<std::vector<int>> spec_t((size_t) N);
        long long offered = 0, kept = 0, rounds = 0, one_pass = 0;
        double ms_spec = 0;
        for (int i = 0; i < N; ++i) {
            if (!prompt(i)) {
                std::fprintf(stderr, "multi check: %s\n", err.c_str());
                return 1;
            }
            spec_t[(size_t) i].push_back(m.sample(0.0f, 1.0f, 0, 0.0f, 0, 0));
        }
        if (pass == 0)
            for (int i = 0; i < N; ++i) {
                std::vector<int>& got = spec_t[(size_t) i];
                m.seq_bind(i);
                const double t0 = now_ms();
                while ((int) got.size() < n) {
                    const int room = std::min(m.verify_max(), n - (int) got.size());
                    std::vector<int> drafts;
                    std::vector<float> conf;
                    int acc = 0, next = -1;
                    if (room > 1 && !m.draft(got.back(), drafts, conf, err)) {
                        std::fprintf(stderr, "multi check: %s\n", err.c_str());
                        return 1;
                    }
                    if ((int) drafts.size() > room - 1) drafts.resize((size_t) std::max(0, room - 1));
                    if (drafts.empty()) {
                        if (!m.step(got.back(), true, err)) {
                            std::fprintf(stderr, "multi check: %s\n", err.c_str());
                            return 1;
                        }
                        next = m.sample(0.0f, 1.0f, 0, 0.0f, 0, 0);
                    } else if (!m.verify(got.back(), drafts, 0.0f, 1.0f, 0, 0.0f, 0, 0, acc, next, err)) {
                        std::fprintf(stderr, "multi check: %s\n", err.c_str());
                        return 1;
                    }
                    got.insert(got.end(), drafts.begin(), drafts.begin() + acc);
                    got.push_back(next);
                    offered += (long long) drafts.size();
                    kept += acc;
                    ++rounds;
                }
                ms_spec += now_ms() - t0;
            }
        else {
            const int share = m.verify_max() / N;
            const double t0 = now_ms();
            for (;;) {
                std::vector<int> sv;
                for (int i = 0; i < N; ++i)
                    if ((int) spec_t[(size_t) i].size() < n) sv.push_back(i);
                if (sv.empty()) break;
                const int k = (int) sv.size();
                std::vector<std::vector<int>> rws((size_t) k), drafts((size_t) k);
                std::vector<const int*> tp((size_t) k);
                std::vector<int> lens((size_t) k), dsv, dys, dix;
                for (int j = 0; j < k; ++j) {
                    const std::vector<int>& got = spec_t[(size_t) sv[(size_t) j]];
                    if (std::min(share, n - (int) got.size()) > 1) {
                        dsv.push_back(sv[(size_t) j]);
                        dys.push_back(got.back());
                        dix.push_back(j);
                    }
                }
                const int nd = (int) dsv.size();
                if (pass == 2 && nd > 1 && m.can_draft_multi(dsv.data(), nd)) {
                    std::vector<std::vector<int>> dd;
                    std::vector<std::vector<float>> cc;
                    if (!m.draft_multi(dsv.data(), dys.data(), nd, dd, cc, err)) {
                        std::fprintf(stderr, "multi check: %s\n", err.c_str());
                        return 1;
                    }
                    for (int j = 0; j < nd; ++j) drafts[(size_t) dix[(size_t) j]] = dd[(size_t) j];
                    ++one_pass;
                } else
                    for (int j = 0; j < nd; ++j) {
                        std::vector<float> conf;
                        m.seq_bind(dsv[(size_t) j]);
                        if (!m.draft(dys[(size_t) j], drafts[(size_t) dix[(size_t) j]], conf, err)) {
                            std::fprintf(stderr, "multi check: %s\n", err.c_str());
                            return 1;
                        }
                    }
                for (int j = 0; j < k; ++j) {
                    const std::vector<int>& got = spec_t[(size_t) sv[(size_t) j]];
                    const int room = std::min(share, n - (int) got.size());
                    if ((int) drafts[(size_t) j].size() > room - 1) drafts[(size_t) j].resize((size_t) std::max(0, room - 1));
                    rws[(size_t) j].push_back(got.back());
                    rws[(size_t) j].insert(rws[(size_t) j].end(), drafts[(size_t) j].begin(), drafts[(size_t) j].end());
                    tp[(size_t) j] = rws[(size_t) j].data();
                    lens[(size_t) j] = (int) rws[(size_t) j].size();
                }
                if (!m.step_multi_spec(sv.data(), tp.data(), lens.data(), k, err)) {
                    std::fprintf(stderr, "multi check: %s\n", err.c_str());
                    return 1;
                }
                for (int j = 0; j < k; ++j) {
                    int acc = 0, next = -1;
                    if (!m.verify_multi(j, drafts[(size_t) j], 0.0f, 1.0f, 0, 0.0f, 0, 0, acc, next, err)) {
                        std::fprintf(stderr, "multi check: %s\n", err.c_str());
                        return 1;
                    }
                    std::vector<int>& got = spec_t[(size_t) sv[(size_t) j]];
                    got.insert(got.end(), drafts[(size_t) j].begin(), drafts[(size_t) j].begin() + acc);
                    got.push_back(next);
                    offered += (long long) drafts[(size_t) j].size();
                    kept += acc;
                }
                ++rounds;
            }
            ms_spec = now_ms() - t0;
        }
        if (pass == 0) ms_spec_alone = ms_spec;
        bool same_all = true;
        for (int i = 0; i < N; ++i) {
            size_t d = 0;
            while (d < alone[(size_t) i].size() && alone[(size_t) i][d] == spec_t[(size_t) i][d]) ++d;
            const bool same = d == alone[(size_t) i].size();
            same_all = same_all && same;
            if (!same) std::printf("multi check: conversation %d with the drafter (pass %d): differs from token %zu on\n", i, pass, d);
        }
        spec_same = spec_same && same_all;
        static const char* const what[3] = {"alone with the drafter", "together, drafts one after another",
                                            "together, drafts in one pass"};
        std::printf("multi check: %d conversations x %d tokens %s: tokens the same: %s; %lld rounds%s, drafts %lld of "
                    "%lld kept; %.2f tok/s in all (x%.2f over together without, x%.2f over alone with the drafter)\n",
                    N, n, what[pass], same_all ? "yes" : "NO", rounds,
                    pass == 2 ? (" (" + std::to_string(one_pass) + " drafted in one pass)").c_str() : "", kept, offered,
                    ms_spec > 0 ? 1000.0 * steps * N / ms_spec : 0.0, ms_spec > 0 ? ms_together / ms_spec : 0.0,
                    ms_spec > 0 ? ms_spec_alone / ms_spec : 0.0);
    }
    return tokens_same && spec_same && same_rows == rows ? 0 : 1;
}

// --parts-check (tests): the --ids prompt read as the server reads it - in one call (feed_mark: the last row's logits,
// the checkpoint before the last token) - then again in parts of one chunk (Model::feed_plan / feed_part), with
// --parallel 2 or more another conversation read and stepped between every two parts, as a prompt read beside a
// running request.  The last row's logits must be the same bit for bit, the n greedy tokens after them the same, and
// the state before the last token too (the last token stepped again from that checkpoint: the same logits).
// (--no-expert-cache keeps the experts where the load put them, so the flow sums a chunk's experts in the same order
// in both reads.)
int check_parts(strata::ds4::Model& m, const std::vector<int>& ids, int n) {
    std::string err;
    const int N = (int) ids.size();
    if (N < 2) {
        std::fprintf(stderr, "parts check: --ids needs two tokens at least\n");
        return 2;
    }
    struct Read {
        std::vector<float> last, again;
        std::vector<int> toks;
        int parts = 0;
    };
    // the other conversation: its prompt (the --ids' first 300 tokens at most) read at the first pause, a greedy step
    // at each one after
    const bool beside = m.seq_count() > 1;
    const std::vector<int> other(ids.begin(), ids.begin() + std::min(N - 1, 300));
    int other_tok = -1, other_steps = 0;
    const auto pause = [&]() -> bool {
        if (!beside) return true;
        m.seq_bind(1);
        bool ok = true;
        if (other_tok < 0) {
            m.reset();
            ok = m.feed(other.data(), (int) other.size(), 1, err);
        } else {
            ok = m.step(other_tok, true, err);
            ++other_steps;
        }
        if (ok) other_tok = m.sample(0.0f, 1.0f, 0, 0.0f, 0, 0);
        m.seq_bind(0);
        return ok;
    };
    const auto read = [&](bool in_parts, Read& rd) -> bool {
        m.seq_bind(0);
        m.reset();
        if (!in_parts) {
            if (!m.feed_mark(ids.data(), N, 1, 1, err)) return false;
            rd.parts = 1;
        } else {
            strata::ds4::FeedPlan plan;
            if (!m.feed_plan(N, 1, 1, plan)) {
                err = "the prompt is not a run of two chunks or more here (a longer --ids, or a smaller "
                      "--prefill-chunk)";
                return false;
            }
            const int nk = (int) plan.ends.size();
            for (int k = 0; k < nk; ++k) {
                if (!m.feed_part(plan, ids.data(), k, k + 1, err)) return false;
                if (k + 1 < nk && !pause()) return false;
            }
            rd.parts = nk;
        }
        if (!m.logits(rd.last)) {
            err = "no logits";
            return false;
        }
        int tok = m.sample(0.0f, 1.0f, 0, 0.0f, 0, 0);
        rd.toks.push_back(tok);
        for (int k = 1; k < n; ++k) {
            if (!m.step(tok, true, err)) return false;
            tok = m.sample(0.0f, 1.0f, 0, 0.0f, 0, 0);
            rd.toks.push_back(tok);
        }
        // the checkpoint the prompt's end kept, before its last token: back there, and the last token again
        if (!m.restore(N - 1) || m.pos() != N - 1) {
            err = "the checkpoint before the prompt's last token is not there";
            return false;
        }
        if (!m.step(ids.back(), true, err) || !m.logits(rd.again)) {
            if (err.empty()) err = "no logits";
            return false;
        }
        return true;
    };
    Read one, parts;
    if (!read(false, one) || !read(true, parts)) {
        std::fprintf(stderr, "parts check: %s\n", err.c_str());
        return 1;
    }
    const auto diff = [](const std::vector<float>& a, const std::vector<float>& b, bool& same) {
        same = a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
        double d = 0;
        for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) d = std::max(d, (double) std::fabs(a[i] - b[i]));
        return d;
    };
    bool same_last = false, same_again = false;
    const double d_last = diff(one.last, parts.last, same_last), d_again = diff(one.again, parts.again, same_again);
    size_t t = 0;
    while (t < one.toks.size() && t < parts.toks.size() && one.toks[t] == parts.toks[t]) ++t;
    const bool same_toks = t == one.toks.size() && t == parts.toks.size();
    std::printf("parts check: %d prompt tokens in one call and in %d parts%s: the last row's logits %s (max |d| %.3g); "
                "the state before the last token %s (max |d| %.3g); %d greedy tokens %s\n",
                N, parts.parts, beside ? (", another conversation stepped " + std::to_string(other_steps) +
                                          " times between them").c_str() : "",
                same_last ? "bit for bit" : "DIFFER", d_last, same_again ? "bit for bit" : "DIFFERS", d_again, n,
                same_toks ? "the same" : ("differ from token " + std::to_string(t) + " on").c_str());
    return same_last && same_again && same_toks ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    strata::platform::env_aliases();   // (MAYA_* settings under their older names too)
#if defined(__linux__)
    // the kernel's NUMA balancing (kernel.numa_balancing=1) scans memory whose policy migrates on fault: every thread
    // then takes hinting faults - the launch thread hundreds a second, the CPU lane's workers on their shared scratch,
    // each phase as slow as its straggler (a 2-socket Xeon: 3k faults/s decoding).  MPOL_LOCAL as the task policy,
    // before any thread (they inherit it), allocates as the default does but takes the process out of the scan.
    // DS4_NUMA_BALANCE=1: left to the kernel.  (A container needs SYS_NICE for set_mempolicy / mbind.)
    // A policy the user set (numactl --interleave / --membind around the engine) is left as it is: only the default one
    // is replaced.
    int pol_mode = 0;
    const bool pol_default = syscall(SYS_get_mempolicy, &pol_mode, nullptr, 0, nullptr, 0) != 0 || pol_mode == 0;
    if (const char* nb = std::getenv("DS4_NUMA_BALANCE"); pol_default && (nb == nullptr || std::atoi(nb) == 0))
        if (syscall(SYS_set_mempolicy, 4 /* MPOL_LOCAL */, nullptr, 0) != 0)
            std::fprintf(stderr, "ds4: set_mempolicy(MPOL_LOCAL) failed (%s): NUMA balancing still scans the engine\n",
                         std::strerror(errno));
#endif
    // cuBLAS gives the same bits run after run only with one stream active or with its workspace pinned (its notes on
    // reproducibility): the chunked prompt path's GEMMs run beside the copy and hc streams, and without this each
    // process could pick other kernels - a long prompt's logits differing from one start to the next.  Before any
    // cuBLAS handle; a setting the caller made stays
    if (std::getenv("CUBLAS_WORKSPACE_CONFIG") == nullptr) {
#if defined(_WIN32)
        _putenv_s("CUBLAS_WORKSPACE_CONFIG", ":4096:8");
#else
        setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 0);
#endif
    }
#if defined(_WIN32)
    // Windows sleeps in 15.6 ms ticks unless a process asks for finer ones: the CPU lane's service thread naps
    // (sleep_for 20 us) between tokens, and the device's first wait of the next token then waits out the tick.
    // MAYA_DS4_TIMER_MS=0: leave the system's resolution alone.
    {
        const char* v = std::getenv("MAYA_DS4_TIMER_MS");
        const int ms = v != nullptr ? std::atoi(v) : 1;
        if (ms > 0) timeBeginPeriod((UINT) ms);
    }
#endif
    strata::ds4::Options o;
    std::string ids_s, dump, eos = "1", ppl;
    int n = 0, window = 16, suffix_draft = 3;
    bool spec = true;
    bool dspark_on_asked = false;   // (--dspark on given: a pack without drafter blocks is then told)
    float temp = 0.0f;   // the CLI's sampling (0: greedy)
    unsigned long long cli_seed = 1;
    bool serve_mode = false, multi_check = false, parts_check = false, describe = false, dspark_set = false;
    std::string prof_run, prof_out, spec_eval_path;
    int sampler_n = 0;
    std::string images, enc_rgb, enc_out;
    int enc_w = 0, enc_h = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        // a number that really is one, in [lo, hi]: atof("abc") is 0, and 0 means "the file's own scale" - a typo (or
        // a 1e-50 that rounds to float 0) must not quietly become a default-strength run
        auto num = [&](const char* what, double lo, double hi, bool whole) -> double {
            const std::string v = val(what);
            char* end = nullptr;
            const double d = std::strtod(v.c_str(), &end);
            if (v.empty() || end == v.c_str() || *end != '\0' || !(d >= lo && d <= hi) || (whole && std::trunc(d) != d)) {
                std::fprintf(stderr, "%s needs a %s from %g to %g, not \"%s\"\n", what, whole ? "whole number" : "number",
                             lo, hi, v.c_str());
                std::exit(2);
            }
            return d;
        };
        if (a == "--pack") o.pack = val("--pack");
        else if (a == "--gpus") {
            for (int g : parse_ids(val("--gpus"))) o.gpus.push_back(g);
        } else if (a == "--max-context") o.max_context = std::atoll(val("--max-context").c_str());
        else if (a == "--reserve-mib") o.reserve_mib = std::atoll(val("--reserve-mib").c_str());
        else if (a == "--vram-experts") o.vram_experts = std::atof(val("--vram-experts").c_str());
        else if (a == "--cpu-threads") o.cpu_threads = std::atoi(val("--cpu-threads").c_str());
        else if (a == "--expert-profile") o.profile = val("--expert-profile");
        else if (a == "--eos") eos = val("--eos");
        else if (a == "--no-expert-cache") o.expert_cache = false;
        else if (a == "--dspark") {
            const std::string v = val("--dspark");
            spec = v != "off";
            o.dspark = v != "none";   // none: not even loaded
            dspark_on_asked = v == "on";
            dspark_set = true;
        } else if (a == "--dspark-conf") o.dspark_conf = (float) std::atof(val("--dspark-conf").c_str());
        else if (a == "--suffix-draft") suffix_draft = std::max(0, std::atoi(val("--suffix-draft").c_str()));
        else if (a == "--tp") o.tp = std::max(1, std::atoi(val("--tp").c_str()));
        else if (a == "--prefill-chunk") o.prefill_chunk = std::max(0, std::atoi(val("--prefill-chunk").c_str()));
        else if (a == "--threads") o.threads = val("--threads") != "off";
        else if (a == "--vision") o.vision = val("--vision") != "off";
        else if (a == "--mmap-experts") {
            const std::string v = val("--mmap-experts");
            o.mmap_experts = v == "on" ? 1 : v == "off" ? 0 : -1;
        } else if (a == "--images") images = val("--images");
        else if (a == "--encode-image") enc_rgb = val("--encode-image");
        else if (a == "--dump-image") enc_out = val("--dump-image");
        else if (a == "--size") {
            const std::string v = val("--size");
            enc_w = std::atoi(v.c_str());
            enc_h = v.find('x') != std::string::npos ? std::atoi(v.c_str() + v.find('x') + 1) : 0;
        }
        else if (a == "--temperature") temp = (float) std::atof(val("--temperature").c_str());
        else if (a == "--seed") cli_seed = std::strtoull(val("--seed").c_str(), nullptr, 10);
        else if (a == "--verbose" || a == "-v") o.verbose = true;
        else if (a == "--version") {
            std::printf("maya-ds4 %s (build %s)\n", MAYA_RELEASE, MAYA_BUILD_ID);
            return 0;
        }
        else if (a == "--ids") ids_s = val("--ids");
        else if (a == "--n") n = std::atoi(val("--n").c_str());
        else if (a == "--window") window = std::max(1, std::atoi(val("--window").c_str()));   // (> 16: the chunked prompt path)
        else if (a == "--dump-logits") dump = val("--dump-logits");
        else if (a == "--ppl") ppl = val("--ppl");
        else if (a == "--ablate") o.ablate = val("--ablate");
        else if (a == "--ablate-layer") o.ablate_layer = (int) num("--ablate-layer", 1, 4096, true);
        else if (a == "--ablate-scale") o.ablate_scale = (float) num("--ablate-scale", 1e-6, 1e6, false);
        else if (a == "--serve") serve_mode = true;
        else if (a == "--parallel") o.seqs = (int) num("--parallel", 1, 8, true);   // (--serve: conversations at once)
        else if (a == "--multi-check") multi_check = true;   // (tests: --parallel's rows against each alone)
        else if (a == "--parts-check") parts_check = true;   // (tests: a prompt in parts against one call)
        else if (a == "--pack-info") describe = true;
        else if (a == "--profile-run") prof_run = val("--profile-run");
        else if (a == "--profile-out") prof_out = val("--profile-out");
        else if (a == "--profile-readonly") o.profile_readonly = true;
        else if (a == "--spec-eval") spec_eval_path = val("--spec-eval");
        else if (a == "--sampler-test") sampler_n = (int) num("--sampler-test", 50, 1e6, true);
        else if (a == "--layer-split") (void) val("--layer-split");   // the server's (this engine balances its own split)
        else if (a == "--help" || a == "-h") {
            usage();
            return 0;
        } else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            usage();
            return 2;
        }
    }
    if (o.verbose) std::fprintf(stderr, "ds4: build %s (Maya %s)\n", MAYA_BUILD_ID, MAYA_RELEASE);
    if (describe) {   // the pack's description alone, before anything that would touch a GPU
        if (o.pack.empty()) {
            usage();
            return 2;
        }
        return strata::ds4::pack_info(o.pack, o.profile, kernel_types(), stdout);
    }
    if (parts_check && ids_s.empty()) {
        std::fprintf(stderr, "--parts-check needs --ids (a prompt of two chunks or more)\n");
        return 2;
    }
    if (multi_check && (o.seqs < 2 || ids_s.empty())) {
        std::fprintf(stderr, "--multi-check needs --parallel 2..8 and --ids\n");
        return 2;
    }
    if (sampler_n > 0 && ids_s.empty()) {
        std::fprintf(stderr, "--sampler-test needs --ids (a prompt whose next tokens are open)\n");
        return 2;
    }
    if (o.pack.empty() || (!serve_mode && ids_s.empty() && enc_rgb.empty() && ppl.empty() && prof_run.empty() &&
                           spec_eval_path.empty())) {
        usage();
        return 2;
    }
    if (o.profile.empty()) o.profile = o.pack + "/expert-profile.bin";   // "none": no profile
    if (o.profile == "none") o.profile.clear();
    o.profile_out = prof_out;
    if (!prof_run.empty()) {   // (a profile run: the pack's profile read only, the run's counts to their own file)
        o.profile_readonly = true;
        if (o.profile_out.empty()) o.profile_out = prof_run + ".profile.bin";
        if (!dspark_set) o.dspark = false;   // (it never drafts: the drafter's VRAM to experts)
        suffix_draft = 0;
        // every layer routes every token, as a decode's do: the prompt skip (layers 21-39 of a long run's earlier
        // tokens left out) off unless set
        const bool skip_set = std::getenv("DS4_YOCO") != nullptr || std::getenv("MAYA_DS4_PROMPT_SKIP") != nullptr;
        if (!skip_set) {
#if defined(_WIN32)
            _putenv_s("DS4_YOCO", "0");
#else
            setenv("DS4_YOCO", "0", 1);
#endif
        }
        std::fprintf(stderr, "profile-run: %s's routing counted into %s; the expert profile %s is only read%s\n",
                     prof_run.c_str(), o.profile_out.c_str(), o.profile.empty() ? "(none)" : o.profile.c_str(),
                     skip_set ? "" : "; every layer on every token (the prompt skip off)");
    }
    if (!spec_eval_path.empty() || sampler_n > 0) {   // (the drafter's tools: it loaded, no prompt lookup)
        if (!dspark_set) o.dspark = true;
        suffix_draft = 0;
    }
    if (sampler_n > 0 && std::getenv("DS4_DSPARK_SAMPLE") == nullptr) {   // (drafts drawn from the drafter's q)
#if defined(_WIN32)
        _putenv_s("DS4_DSPARK_SAMPLE", "1");
#else
        setenv("DS4_DSPARK_SAMPLE", "1", 1);
#endif
    }
    o.vision_only = !enc_rgb.empty();
    strata::ds4::Model m;
    std::string err;
    o.lookup = suffix_draft > 0;   // (its verify windows' buffers, with no drafter too)
    if (!m.load(o, err)) {
        std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
        return 1;
    }
    if (!o.vision_only) std::fprintf(stderr, "%s", m.describe().c_str());
    if (dspark_on_asked && !o.vision_only && !m.has_drafter())
        std::fprintf(stderr, "maya-ds4: warning: --dspark on, but this pack has no DSpark drafter blocks (mtp.* tensors): "
                             "decoding with prompt lookup only (no drafter)\n");
    std::unique_ptr<Lookup> lk;   // (prompt lookup beside DSpark, or alone: one policy over every request of this run)
    if (suffix_draft > 0)
        lk = std::make_unique<Lookup>(suffix_draft, m.verify_max(), (size_t) m.max_context() + 4096);
    if (serve_mode) return serve(m, eos, spec, o.dspark_conf, o.pack, lk.get());
    if (!prof_run.empty()) return profile_run(m, prof_run, o.profile_out);
    if (!spec_eval_path.empty()) return spec_eval(m, spec_eval_path, n > 0 ? n : 256, temp, cli_seed);
    if (sampler_n > 0) {
        if (!ids_s.empty() && ids_s[0] == '@') {   // (--ids @FILE, as the CLI's run takes it)
            std::ifstream f(ids_s.substr(1));
            ids_s.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        }
        return sampler_test(m, parse_ids(ids_s), sampler_n, cli_seed);
    }
    if (multi_check) return check_multi(m, parse_ids(ids_s), std::max(2, n));
    if (parts_check) return check_parts(m, parse_ids(ids_s), std::max(2, n));
    if (!ppl.empty()) {   // --ppl FILE: a line a sequence of ids (BOS + text), each in one forward, tokens 2.. scored
        std::ifstream in(ppl);
        std::string line;
        double nll = 0.0;
        long long scored = 0;
        int k = 0;
        std::vector<float> lg;
        const int V = m.vocab();
        while (std::getline(in, line)) {
            const std::vector<int> seq = parse_ids(line);
            if (seq.size() < 2) continue;
            m.reset();
            if (!m.feed(seq.data(), (int) seq.size(), (int) seq.size(), err) || !m.logits(lg) ||
                lg.size() < (seq.size() - 1) * (size_t) V) {
                std::fprintf(stderr, "maya-ds4: sequence %d: %s\n", k, err.empty() ? "no logits" : err.c_str());
                return 1;
            }
            double s_nll = 0.0;
            for (size_t i = 0; i + 1 < seq.size(); ++i) {   // row i: the next token's log-probability
                const float* r = lg.data() + i * (size_t) V;
                float mx = r[0];
                for (int v = 1; v < V; ++v) mx = std::max(mx, r[v]);
                double z = 0.0;
                for (int v = 0; v < V; ++v) z += std::exp((double) (r[v] - mx));
                s_nll += (double) mx + std::log(z) - (double) r[seq[i + 1]];
            }
            nll += s_nll;
            scored += (long long) seq.size() - 1;
            std::fprintf(stderr, "ppl: sequence %d, %zu tokens: %.4f\n", k++, seq.size(), std::exp(s_nll / (double) (seq.size() - 1)));
        }
        std::printf("perplexity %.4f over %lld tokens of %d sequences\n", scored ? std::exp(nll / (double) scored) : 0.0,
                    scored, k);
        return 0;
    }

    if (!enc_rgb.empty()) {   // the vision tower alone: a picture's span rows (BF16 [rows][dim]) into --dump-image
        std::vector<uint8_t> px((size_t) std::max(0, enc_w) * std::max(0, enc_h) * 3);
        std::FILE* f = std::fopen(enc_rgb.c_str(), "rb");
        const bool got = f != nullptr && !px.empty() && std::fread(px.data(), 1, px.size(), f) == px.size();
        if (f != nullptr) std::fclose(f);
        if (!got) {
            std::fprintf(stderr, "maya-ds4: %s: expected %d x %d RGB bytes (--size WxH)\n", enc_rgb.c_str(), enc_w, enc_h);
            return 1;
        }
        std::vector<uint16_t> span;
        for (int k = 0; k < 2; ++k) {   // (the second: warm)
            const double te = now_ms();
            if (!m.encode_image(px.data(), enc_w, enc_h, span, err)) {
                std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
                return 1;
            }
            std::fprintf(stderr, "vision: %d x %d pixels -> a span of %zu rows in %.0f ms\n", enc_w, enc_h,
                         span.size() / (size_t) m.dim(), now_ms() - te);
        }
        std::FILE* o2 = enc_out.empty() ? nullptr : std::fopen(enc_out.c_str(), "wb");
        if (o2 != nullptr) {
            std::fwrite(span.data(), 2, span.size(), o2);
            std::fclose(o2);
        }
        return 0;
    }
    if (!ids_s.empty() && ids_s[0] == '@') {   // --ids @FILE: the ids from a file (a long prompt passes no argv limit)
        std::FILE* f = std::fopen(ids_s.c_str() + 1, "rb");
        if (f == nullptr) {
            std::fprintf(stderr, "maya-ds4: cannot open %s\n", ids_s.c_str() + 1);
            return 1;
        }
        std::string all;
        char buf[1 << 16];
        for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;) all.append(buf, n);
        std::fclose(f);
        ids_s = all;
    }
    const std::vector<int> ids = parse_ids(ids_s);
    if (!images.empty()) {   // --images: a GENI file's pictures, for the prompt's spans
        std::vector<Pic> pics;
        if (!read_pics(images, pics, err)) {
            std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
            return 1;
        }
        std::vector<strata::ds4::Image> imgs;
        for (const Pic& q : pics) {
            auto rows = std::make_shared<std::vector<uint16_t>>();
            const double te = now_ms();
            if (!m.encode_image(q.rgb.data(), q.width, q.height, *rows, err)) {
                std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
                return 1;
            }
            std::fprintf(stderr, "picture at %lld: %d x %d pixels, %zu tokens, %.0f ms\n", (long long) q.start, q.width,
                         q.height, rows->size() / (size_t) m.dim(), now_ms() - te);
            imgs.push_back({q.start, rows});
        }
        m.set_images(std::move(imgs));
    }
    // DS4_BENCH_REPEAT=N: the prompt and the decode N times, the state reset between them and the expert cache kept
    // (the later passes measure a warmed cache, as a server's requests run)
    const int reps = std::max(1, std::getenv("DS4_BENCH_REPEAT") ? std::atoi(std::getenv("DS4_BENCH_REPEAT")) : 1);
    for (int rep = 0; rep < reps; ++rep) {
    if (rep > 0) {
        m.reset();
        std::fprintf(stderr, "pass %d of %d\n", rep + 1, reps);
    }
    // (repeated: each pass's logits to a file of its own, <dump>.<pass> - a run's determinism, one load)
    const std::string dpath = reps > 1 ? dump + "." + std::to_string(rep) : dump;
    std::FILE* df = dump.empty() ? nullptr : std::fopen(dpath.c_str(), "wb");
    std::vector<float> lg;
    m.begin_request();
    // DS4_PROFILE_PROMPT: the prompt alone in the capture (nsys --capture-range=cudaProfilerApi), the last pass
    const bool prof_prompt = std::getenv("DS4_PROFILE_PROMPT") != nullptr && rep == reps - 1;
    const strata::ds4::Stats sp = m.stats();
    if (prof_prompt) cudaProfilerStart();
    const double t0 = now_ms();
    for (size_t a = 0; a < ids.size();) {
        const int T = (int) std::min<size_t>((size_t) window, ids.size() - a);
        const bool last = a + (size_t) T == ids.size();
        if (!m.feed(ids.data() + a, T, df != nullptr ? T : (last ? 1 : 0), err)) {
            std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
            return 1;
        }
        if (df != nullptr && m.logits(lg)) std::fwrite(lg.data(), 4, lg.size(), df);
        a += (size_t) T;
    }
    m.set_images({});
    const double t1 = now_ms();
    if (prof_prompt) cudaProfilerStop();
    std::fprintf(stderr, "prompt: %zu tokens in %.0f ms (%.2f tok/s, windows of %d)\n", ids.size(), t1 - t0,
                 1000.0 * (double) ids.size() / (t1 - t0), window);
    if (const char* dl = std::getenv("DS4_DUMP_LAST")) {   // the prompt's last row of logits, <file>.<pass>
        std::vector<float> ll;
        std::FILE* f = m.logits(ll) ? std::fopen((std::string(dl) + "." + std::to_string(rep)).c_str(), "wb") : nullptr;
        if (f != nullptr) {
            const size_t V = (size_t) m.vocab();
            std::fwrite(ll.data() + ll.size() - V, 4, V, f);
            std::fclose(f);
        }
    }
    const strata::ds4::Stats s0 = m.stats();
    if (s0.eg_rows > 0)
        std::fprintf(stderr, "prompt's Engram rows: %lld, %.1f%% without a read (repeats and the row cache), %lld read\n",
                     (long long) s0.eg_rows, 100.0 * (double) s0.eg_hits / (double) s0.eg_rows, (long long) s0.eg_reads);
    const long long io0 = io_read_bytes();
    std::fprintf(stderr, "prompt's host time: layers %.0f ms, head wait %.0f, Engram waits %.0f, CPU lane %.0f\n",
                 s0.layer_ms - sp.layer_ms, s0.head_ms - sp.head_ms, s0.engram_ms - sp.engram_ms, s0.cpu_ms - sp.cpu_ms);
    const bool prof = std::getenv("DS4_PROFILE") != nullptr;   // (nsys --capture-range=cudaProfilerApi: the decode alone)
    if (prof && rep == reps - 1) cudaProfilerStart();   // (the last pass)
    std::vector<int> out;
    unsigned long long counter = 0;
    long long d_off = 0, d_acc = 0, rounds = 0;
    bool cool = false;
    if (n > 0) {
        int tok = m.sample(temp, 1.0f, temp > 0 ? 64 : 1, 0.0f, cli_seed, counter++);
        std::vector<int> got;
        round_log().tag = "ids";   // (DS4_ROUND_LOG)
        if (lk) lk->start(m.tokens());
        const long long lk_w0 = lk ? lk->windows : 0, lk_o0 = lk ? lk->offered : 0, lk_a0 = lk ? lk->accepted : 0;
        for (;;) {
            if (tok < 0) {   // (a sample's CUDA error, printed by the sampler)
                std::fprintf(stderr, "maya-ds4: the sample failed\n");
                return 1;
            }
            out.push_back(tok);
            std::printf("%d,", tok);
            std::fflush(stdout);
            if (lk) lk->sfx.append(tok);
            if (tok == 1 || (int) out.size() >= n) break;
            if (got.empty()) {
                int off = 0, acc = 0;
                const bool try_spec = spec && !cool;
                if (!round(m, tok, n - (int) out.size(), try_spec, o.dspark_conf, temp, 1.0f, temp > 0 ? 64 : 1, 0.0f,
                           cli_seed, counter, got, off, acc, err, -1.0f, lk.get())) {
                    std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
                    return 1;
                }
                d_off += off;
                d_acc += acc;
                ++rounds;
                cool = try_spec && off == 0;
                std::reverse(got.begin(), got.end());
            }
            tok = got.back();
            got.pop_back();
        }
        std::printf("\n");
        if (df != nullptr && m.logits(lg)) std::fwrite(lg.data(), 4, lg.size(), df);
        if (lk && lk->windows > lk_w0)
            std::fprintf(stderr, "prompt lookup: %lld rounds, %lld of its %lld drafts kept (in the dspark line's)\n",
                         lk->windows - lk_w0, lk->accepted - lk_a0, lk->offered - lk_o0);
    }
    if (prof && rep == reps - 1) cudaProfilerStop();
    if (spec && rounds > 0) {
        const auto& s1 = m.stats();
        const double nv = (double) std::max<int64_t>(1, s1.drafts_rounds - s0.drafts_rounds);
        std::fprintf(stderr, "dspark: %lld rounds, %lld drafts offered, %lld accepted (%.2f tokens a round); drafting "
                             "%.2f ms a round, verifying %.1f ms in all (%lld windows)\n",
                     rounds, d_off, d_acc, (double) out.size() / (double) rounds, (s1.draft_ms - s0.draft_ms) / nv,
                     s1.verify_ms - s0.verify_ms, (long long) (s1.drafts_offered > 0 ? 0 : 0));
        std::fprintf(stderr, "dspark %s\n", per_position(s0, s1).c_str());
    }
    const double t2 = now_ms();
    const auto& s = m.stats();
    const double nt = std::max<double>(1.0, (double) (s.tokens - s0.tokens));
    std::fprintf(stderr, "decode: %zu tokens in %.0f ms (%.2f tok/s); CPU lane: %lld of %lld experts, streamed to the GPU: "
                         "%lld; the cache moved %lld (%lld in all)\n", out.size(), t2 - t1,
                 out.size() > 1 ? 1000.0 * (double) (out.size() - 1) / (t2 - t1) : 0.0,
                 (long long) (s.cpu_experts - s0.cpu_experts), (long long) (s.experts - s0.experts),
                 (long long) (s.stream_experts - s0.stream_experts), (long long) (s.swaps - s0.swaps),
                 (long long) s.swaps);
    std::fprintf(stderr, "decode per token: layers %.2f ms (of it: routing waits %.2f, CPU lane %.2f, the rest host launches), "
                         "head wait %.2f, Engram reads %.2f\n",
                 (s.layer_ms - s0.layer_ms) / nt, (s.gpu_wait_ms - s0.gpu_wait_ms) / nt, (s.cpu_ms - s0.cpu_ms) / nt,
                 (s.head_ms - s0.head_ms) / nt, (s.engram_ms - s0.engram_ms) / nt);
    if (s.helper_held > 0)   // DS4_HELPER
        std::fprintf(stderr, "decode helper: %lld experts on the helper GPU (%d held), %lld on the CPU lane after it failed%s; "
                             "waits for it %.2f ms a token\n",
                     (long long) (s.helper_experts - s0.helper_experts), s.helper_held,
                     (long long) (s.helper_fallback - s0.helper_fallback), s.helper_dead ? " (it failed)" : "",
                     (s.helper_ms - s0.helper_ms) / nt);
    if (s.eg_rows > s0.eg_rows) {   // the Engram rows: how many the row cache served, the reads' time behind the layers
        const double rows = (double) (s.eg_rows - s0.eg_rows);
        std::fprintf(stderr, "Engram per token: %.0f rows, %.1f%% without a read (the row cache holds %lld of %lld), "
                             "%.1f read, the reads %.2f ms from asking to the last one (left to wait: %.2f)\n",
                     rows / nt, 100.0 * (double) (s.eg_hits - s0.eg_hits) / rows, (long long) s.eg_cached,
                     (long long) s.eg_cache_cap, (double) (s.eg_reads - s0.eg_reads) / nt, (s.eg_io_ms - s0.eg_io_ms) / nt,
                     (s.engram_ms - s0.engram_ms) / nt);
    }
    const long long io1 = io_read_bytes();
    if (io0 >= 0 && io1 >= 0)
        std::fprintf(stderr, "decode reads from the disk: %.1f MB a token\n", (double) (io1 - io0) / 1e6 / nt);
    if (s.skipped > s0.skipped)   // STRATA_DS4_DISK_SKIP_W
        std::fprintf(stderr, "decode left out %.1f experts a token not in RAM (%.1f%% of the routed)\n",
                     (double) (s.skipped - s0.skipped) / nt,
                     100.0 * (double) (s.skipped - s0.skipped) / std::max<double>(1.0, (double) (s.experts - s0.experts)));
    if (s.skip_on >= 0)
        std::fprintf(stderr, "DISK_SKIP auto: %s at the end, %lld switches\n", s.skip_on ? "on" : "off",
                     (long long) s.skip_switches);
    if (s.ahead_picks > s0.ahead_picks) {   // lookahead
        const double p = (double) (s.ahead_picks - s0.ahead_picks), c = (double) (s.ahead_cold - s0.ahead_cold);
        std::fprintf(stderr, "lookahead: the guesses a layer before held %.1f%% of the picks (their top k %.1f%%, two layers "
                             "before %.1f%%); file-read picks asked for early %.1f%% (%lld of %lld); %.1f experts asked "
                             "for early a token\n",
                     100.0 * (double) (s.ahead_hit - s0.ahead_hit) / p, 100.0 * (double) (s.ahead_hit_k - s0.ahead_hit_k) / p,
                     100.0 * (double) (s.ahead_hit2 - s0.ahead_hit2) / p,
                     c > 0 ? 100.0 * (double) (s.ahead_cold_hit - s0.ahead_cold_hit) / c : 0.0,
                     (long long) (s.ahead_cold_hit - s0.ahead_cold_hit), (long long) (s.ahead_cold - s0.ahead_cold),
                     (double) (s.ahead_pf - s0.ahead_pf) / nt);
        std::string rk = "lookahead: picked by the guess's rank (1 ..):";
        for (int i = 0; i < 16; ++i) {
            const long long n = (long long) (s.ahead_rank_n[i] - s0.ahead_rank_n[i]);
            if (n <= 0) break;
            char b[24];
            std::snprintf(b, sizeof b, " %.0f%%", 100.0 * (double) (s.ahead_rank_hit[i] - s0.ahead_rank_hit[i]) / (double) n);
            rk += b;
        }
        std::fprintf(stderr, "%s\n", rk.c_str());
    }
    prof_line(s0, s);   // DS4_PROF
    if (df != nullptr) std::fclose(df);
    }
    m.save_profile();
    return 0;
}
