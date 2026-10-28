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
// the same places.
#include "ds4_evt.hpp"
#include "ds4_model.hpp"
#include "ds4_watchdog.hpp"
#include "strata/platform/env_alias.hpp"
#include "strata/spec/draft_policy.hpp"
#include "strata/spec/suffix_drafter.hpp"

#include <cuda_profiler_api.h>

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
                 "           | --encode-image FILE.rgb --size WxH --dump-image OUT)\n");
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

// --parallel with the drafter, MAYA_DS4_MULTI_DRAFT=1 (off by default): every conversation answering drafts its own
// tokens (DSpark, bound in turn) and one window verifies them all (Model::verify_multi) - instead of a token of each
// a step.  Each conversation's drafts are cut as a lone round's (the confidence floor, the cumulative cut), then to
// MAYA_DS4_MULTI_DRAFT_MAX a conversation (the block) and to what fits the window of MAYA_DS4_MULTI_DRAFT_ROWS rows
// (8): (rows - n) / n each with n answering.  8 rows, not 16: a window of more than 8 rows is not bit for bit what
// each conversation gets alone - the experts' kernels take rows in tiles of up to 8 (gu_ne / dn_ne, light_kernel),
// and an expert with 9 rows or more sums in another order (a layer split, --multi-check: windows of 7 and 8 rows
// exact, the first of 10 differing by ~0.8 in every conversation's logits); up to 16 may be set where speed matters
// more than that.
// The round's drafting conversations draft in one pass of the drafter (Model::draft_multi: their blocks one window
// through the drafter's blocks and the head, up to Model::draft_pass() of them a pass - 3 at a block of 5 in 16 rows -
// the rest in another), not one call each: the drafter's time for j drafting is about one call's and a part more,
// not j calls'.
// Which conversations draft in a round, and how many drafts each, a measured cost model decides
// (MAYA_DS4_MULTI_DRAFT_POLICY=1, the default): the round times by window rows (a size not timed yet is read off the
// line through the two nearest timed sizes, or scaled from the one timed as (FIXED + rows)), the drafter's time by how
// many drafted together (an untimed count scaled from a timed one by its passes and their rows: a pass of r costs
// (DFIXED + r) / (DFIXED + 1) of a pass of one, MAYA_DS4_MULTI_DRAFT_DFIXED 2) and each conversation's recent
// acceptance per draft, all averaged as the engine serves; of the plans "the j conversations with the best acceptance
// draft, (rows - n) / j each at most" it takes the one expecting the most tokens a millisecond, if that beats the plain
// step (every row one token) by MAYA_DS4_MULTI_DRAFT_MARGIN (0.03) - so a round whose drafts would not pay is the plain
// step, and fewer conversations with more drafts each are considered (three answering in 8 rows: one of them with 5
// drafts, or two with 2).  Note what the window's rows cap: at 4 answering in 8 rows the drafts of a round are 4
// however they are shared out, so at an acceptance near 1 every plan expects the same 8 tokens and the cheapest
// drafter wins - one conversation drafting 4; a request's line then shows most of its rounds without drafts of its
// own, though every round had drafts.  A conversation the plan leaves out drafts once every
// MAYA_DS4_MULTI_DRAFT_RETRY rounds (8) anyway, the gap doubling (to MAYA_DS4_MULTI_DRAFT_PROBE_CAP, 128) while the
// probes do not get it into the plan - its acceptance and the drafter's time stay measured; a new conversation starts
// from the engine's recent averages.  MAYA_DS4_MULTI_DRAFT_POLICY=0 is the earlier rule instead: a window of R rows
// costs about a + b R, so n conversations' drafts pay only while their acceptance is above n / (a / b + n) - about
// 0.28, 0.44, 0.54, 0.61 at n = 1 .. 4 with a / b = 2.6 rows (MAYA_DS4_MULTI_DRAFT_FIXED; MAYA_DS4_MULTI_DRAFT_MIN=F
// sets one floor instead) - a conversation under the floor takes a plain row for RETRY rounds, then drafts once (the
// drafter's time is not in that rule: it counts the window alone; FIXED is also the shape the measured model scales
// a row count from while one size is timed).
// A window's tokens are written in one turn (MAYA_DS4_MULTI_DRAFT_LOCKSTEP=1, the default), so every answering
// conversation is in every round: written one a turn, the conversations whose drafts were kept would sit out the
// next rounds, and the one left needing a token would take a plain step of one row - the dearest row there is.
// Greedy, each conversation answers token for token as alone (the drafts and the verdicts are the same; only how
// many drafts a round takes differs).  DS4_DSPARK_SAMPLE=1 (drafts drawn from the drafter's distribution) is not
// taken here: the drafts are the drafter's argmax, as verify() takes drafts not its own draws.  A request's spec=0
// keeps it a plain row.  MAYA_DS4_MULTI_DRAFT_LOG=1: a line on stderr for every round of two or more - who drafted,
// the drafter's and the window's time, the tokens, against the plan it beat.  The request's line counts its rounds of
// several three ways: it drafted, others drafted and it took a plain row, nobody drafted (the plain step).
struct MultiDraft {
    bool on = false;
    int max_drafts = 0;    // a conversation's drafts at most (0: the drafter's block)
    int max_rows = 8;      // the window's rows at most (8: exact; up to 16)
    double fixed = 2.6;    // the window's fixed cost in rows (a / b): the floor rule's, the measured model's shape
    double min_rate = -1;  // (floor rule) one acceptance floor for every n (-1: n / (fixed + n))
    int retry = 8;         // (floor rule) rounds without drafts under the floor, then one with; (measured) the first probe gap
    bool measured = true;  // the measured cost model (1), or the floor rule (0)
    double margin = 0.03;  // (measured) a drafting plan must beat the plain step's tokens a millisecond by this
    int probe_cap = 128;   // (measured) the probe gap grows to this (a probe is ~half a plain step more: ~1-2% of the time at the cap)
    double dfixed = 2.0;   // (measured) a draft pass's fixed cost in conversations: a pass of r costs (dfixed + r) / (dfixed + 1) of one
    bool lockstep = true;  // a window's tokens written in one turn
    int log = 0;           // a line a round on stderr
    int pipe_windows = 3;  // (MAYA_DS4_MD_PIPE) windows a group runs in a round, two groups in flight (1: no overlap)
    bool pipe_serial = false;   // (MAYA_DS4_MD_PIPE_SERIAL) the same groups and windows, one window at a time (the
                                // reference a pipelined run must match token for token, whatever the experts' placement)
    // (MAYA_DS4_MD_PIPE_PRICE) what a pipelined window costs the policy (MultiCost::see_round), its windows overlapping:
    // 0 equal - the round's time less the drafter's, an equal share a window; 1 share - the same time shared out in
    // proportion to the cost model's current cost of each window's rows; 2 stage (the default) - each window its own
    // GPU time, the slower of its two stages (Model::pipe_window_ms), plus an equal share of what the round spent
    // beyond the stages and the drafter (the host's work, the pipe's fill and drain)
    int pipe_price = 2;
    // (MAYA_DS4_MD_PIPE_DRAFT_PRICE=gpu) a pipelined round's drafter passes priced by their GPU time (the blocks' and
    // the head's, Stats::draft_block_ms / draft_head_ms) instead of the host's wait for them (wall, the default): a
    // pass drafted while the other window's first stage runs queues behind it, and its wall time includes the rest of
    // that stage
    bool pipe_draft_gpu = false;
};
const char* pipe_price_name(int p) { return p == 0 ? "equal" : p == 1 ? "share" : "stage"; }
MultiDraft multi_draft_settings() {
    const auto env = [](const char* strata, const char* maya) -> const char* {
        const char* v = std::getenv(strata);
        if (v == nullptr || *v == '\0') v = std::getenv(maya);
        return v != nullptr && *v != '\0' ? v : nullptr;
    };
    MultiDraft d;
    if (const char* v = env("STRATA_DS4_MULTI_DRAFT", "MAYA_DS4_MULTI_DRAFT")) d.on = std::atoi(v) != 0;
    if (const char* v = env("STRATA_DS4_MULTI_DRAFT_MAX", "MAYA_DS4_MULTI_DRAFT_MAX")) d.max_drafts = std::max(0, std::atoi(v));
    if (const char* v = env("STRATA_DS4_MULTI_DRAFT_ROWS", "MAYA_DS4_MULTI_DRAFT_ROWS"))
        d.max_rows = std::max(1, std::min(16, std::atoi(v)));
    if (const char* v = env("STRATA_DS4_MULTI_DRAFT_FIXED", "MAYA_DS4_MULTI_DRAFT_FIXED")) d.fixed = std::max(0.0, std::atof(v));
    if (const char* v = env("STRATA_DS4_MULTI_DRAFT_MIN", "MAYA_DS4_MULTI_DRAFT_MIN")) d.min_rate = std::atof(v);
    if (const char* v = env("STRATA_DS4_MULTI_DRAFT_RETRY", "MAYA_DS4_MULTI_DRAFT_RETRY")) d.retry = std::max(0, std::atoi(v));
    if (const char* v = env("STRATA_DS4_MULTI_DRAFT_POLICY", "MAYA_DS4_MULTI_DRAFT_POLICY")) d.measured = std::atoi(v) != 0;
    if (const char* v = env("STRATA_DS4_MULTI_DRAFT_MARGIN", "MAYA_DS4_MULTI_DRAFT_MARGIN")) d.margin = std::max(0.0, std::atof(v));
    if (const char* v = env("STRATA_DS4_MULTI_DRAFT_PROBE_CAP", "MAYA_DS4_MULTI_DRAFT_PROBE_CAP"))
        d.probe_cap = std::max(1, std::atoi(v));
    if (const char* v = env("STRATA_DS4_MULTI_DRAFT_LOCKSTEP", "MAYA_DS4_MULTI_DRAFT_LOCKSTEP")) d.lockstep = std::atoi(v) != 0;
    if (const char* v = env("STRATA_DS4_MULTI_DRAFT_DFIXED", "MAYA_DS4_MULTI_DRAFT_DFIXED")) d.dfixed = std::max(0.0, std::atof(v));
    if (const char* v = env("STRATA_DS4_MULTI_DRAFT_LOG", "MAYA_DS4_MULTI_DRAFT_LOG")) d.log = std::atoi(v);
    if (const char* v = env("STRATA_DS4_MD_PIPE_WINDOWS", "MAYA_DS4_MD_PIPE_WINDOWS"))
        d.pipe_windows = std::max(1, std::min(16, std::atoi(v)));
    if (const char* v = env("STRATA_DS4_MD_PIPE_SERIAL", "MAYA_DS4_MD_PIPE_SERIAL")) d.pipe_serial = std::atoi(v) != 0;
    if (const char* v = env("STRATA_DS4_MD_PIPE_PRICE", "MAYA_DS4_MD_PIPE_PRICE")) {
        const std::string p = v;
        if (p == "equal" || p == "0") d.pipe_price = 0;
        else if (p == "share" || p == "1") d.pipe_price = 1;
        else if (p == "stage" || p == "2") d.pipe_price = 2;
        else std::fprintf(stderr, "maya ds4 serve: MAYA_DS4_MD_PIPE_PRICE=%s: not equal, share or stage - stage\n", v);
    }
    if (const char* v = env("STRATA_DS4_MD_PIPE_DRAFT_PRICE", "MAYA_DS4_MD_PIPE_DRAFT_PRICE"))
        d.pipe_draft_gpu = std::string(v) == "gpu" || std::string(v) == "1";
    return d;
}

// (the measured policy) what the engine has timed as it serves: a round's milliseconds by its window's rows (a plain
// step of n is n rows), the drafter's milliseconds by how many conversations drafted together (one draft_multi call),
// and every conversation's drafts - their acceptance per draft and how many the cut leaves - as a new conversation's
// starting point
struct MultiCost {
    double c_ms[17] = {}, c_n[17] = {};
    double d_ms[17] = {}, d_n[17] = {};
    double gp = 0.7, gk = 3.0;
    double fixed = 2.6;    // (one size timed) a round's cost as (fixed + rows)
    double dfixed = 2.0;   // a draft pass of r conversations as (dfixed + r) of a pass of one
    int pass = 3;          // conversations a draft pass takes (Model::draft_pass)
    // a round of R rows: as timed; else the line through the two nearest sizes timed (the ones either side of R, or
    // the two below / above it), its slope never negative; one size timed: scaled from it as (fixed + R)
    double cost(int R) const {
        R = std::max(1, std::min(16, R));
        if (c_n[R] > 0) return c_ms[R];
        int lo = 0, lo2 = 0, hi = 0, hi2 = 0;
        for (int u = R - 1; u >= 1; --u)
            if (c_n[u] > 0) {
                if (lo == 0) lo = u;
                else if (lo2 == 0) lo2 = u;
            }
        for (int u = R + 1; u <= 16; ++u)
            if (c_n[u] > 0) {
                if (hi == 0) hi = u;
                else if (hi2 == 0) hi2 = u;
            }
        int a = 0, b = 0;   // the two sizes the line goes through
        if (lo > 0 && hi > 0) a = lo, b = hi;
        else if (lo > 0 && lo2 > 0) a = lo2, b = lo;
        else if (hi > 0 && hi2 > 0) a = hi, b = hi2;
        if (a > 0) {
            const double slope = std::max(0.0, (c_ms[b] - c_ms[a]) / (double) (b - a));
            return std::max(0.0, c_ms[a] + slope * (double) (R - a));
        }
        const int u = lo > 0 ? lo : hi;
        if (u > 0) return c_ms[u] * (fixed + (double) R) / (fixed + (double) u);
        return 10.0 * (fixed + (double) R) / (fixed + 1.0);   // (nothing timed yet: a guess)
    }
    // the drafter's time with j conversations drafting (one call): as timed; else from the nearest count timed, by
    // their passes and the rows of each (a pass of r conversations as (dfixed + r) of a pass of one); none: a part of
    // a row's
    double passes(int j) const {
        const int per = std::max(1, pass);
        double t = 0.0;
        for (int r = j; r > 0; r -= per) t += (dfixed + (double) std::min(per, r)) / (dfixed + 1.0);
        return t;
    }
    double draft(int j) const {
        j = std::max(1, std::min(16, j));
        if (d_n[j] > 0) return d_ms[j];
        int u = 0;
        for (int w = 1; w <= 16; ++w)
            if (d_n[w] > 0 && (u == 0 || std::abs(w - j) < std::abs(u - j))) u = w;
        if (u > 0) return d_ms[u] * passes(j) / passes(u);
        return 0.3 * cost(1) * passes(j);
    }
    void see_round(int R, double ms) {
        R = std::max(1, std::min(16, R));
        if (ms <= 0) return;
        c_ms[R] = c_n[R] > 0 ? 0.9 * c_ms[R] + 0.1 * ms : ms;
        c_n[R] += 1.0;
    }
    void see_draft(int j, double ms) {
        j = std::max(1, std::min(16, j));
        if (ms <= 0) return;
        d_ms[j] = d_n[j] > 0 ? 0.9 * d_ms[j] + 0.1 * ms : ms;
        d_n[j] += 1.0;
    }
    void see_drafts(int offered, int accepted) {
        if (offered > 0) gp = 0.9 * gp + 0.1 * ((double) accepted / (double) offered);
    }
};
// the acceptance a conversation's drafts need to pay beside n - 1 others' rows (above)
double multi_draft_floor(const MultiDraft& d, int n) {
    if (d.min_rate >= 0) return d.min_rate;
    return (double) n / (d.fixed + (double) n);
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
    if (drafter) {
        std::vector<int> drafts;
        std::vector<float> conf;
        if (!m.draft(tok, drafts, conf, err, temperature, top_p, top_k, min_p, seed, counter)) return false;
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
        if (!f.read((char*) toks.data(), (std::streamsize) (nt * sizeof(int32_t))) || !f.read((char*) &nc, 8) || nc > 3)
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
    // a request's prompt into the bound conversation (one conversation's requests and --parallel's alike), in three
    // steps: prompt_begin (its pictures read, the state it continues reused - in place, a checkpoint, or a kept
    // conversation), prompt_piece (the next tokens of the rest read: one conversation reads it in one piece, --parallel
    // a piece at a time with the other conversations' tokens between) and prompt_end (the state's checkpoints).  Its
    // lines carry `tag` (" <rid>" with --parallel), stop_now() is its STOP; false: it ended there (its ERR or DONE
    // written)
    struct Prompt {
        double t0 = 0;
        strata::ds4::Stats s0;   // (the request's own CPU lane share)
        size_t from = 0;         // the tokens the state already held
        size_t done = 0;         // the prompt's tokens in the state so far
        size_t next_pp = 0;      // (the next PP line's)
        bool continued = false;  // (the conversation goes on - its anchor stays)
        std::vector<Pic> pics;
    };
    const auto prompt_begin = [&](Req& r, bool geni, const std::string& tag, Prompt& P) -> bool {
        const double t0 = now_ms();
        P.t0 = t0;
        P.s0 = m.stats();
        std::vector<Pic>& pics = P.pics;
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
        bool continued = false;
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
                m.set_images(std::move(imgs));
                std::fprintf(stderr, "maya ds4 serve: %zu pictures, %d read (%.0f ms)\n", pics.size(), encoded,
                             now_ms() - te);
            }
            for (Pic& q : pics) q.rgb = {};   // (only the start, hash and span stay)
        }
        if (!ok) {
            std::fprintf(stderr, "ds4: %s\n", err.c_str());
            std::printf("ERR%s %s\n", tag.c_str(), err.c_str());
            std::printf("DONE%s 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", tag.c_str(), r.ids.size());
            std::fflush(stdout);
            m.set_images({});
            m.reset();
            held.clear();
            return false;
        }
        // a STOP while the prompt is read ends it there (the engine's run of chunks at a layer or chunk, windows between
        // feeds, --parallel's between pieces) and goes back to the state before this prompt - kept here first, in
        // checkpoint 1, which the prompt's own end overwrites as before - so the next request still shares the
        // conversation up to it
        if (from > 0 && from < r.ids.size()) m.checkpoint(1);
        P.from = from;
        P.done = from;
        P.next_pp = from + 64;
        P.continued = continued;
        return true;
    };
    // the prompt's next `piece` tokens (or the rest) in windows (PP <read> <total> <ms> <tok/s> <reuse> every 64
    // tokens), its last with a checkpoint before its last token and one after it: the next turn re-renders this one,
    // and shares its start up to one of them
    const auto prompt_piece = [&](Req& r, const std::string& tag, const std::function<bool()>& stop_now, Prompt& P,
                                  size_t piece) -> bool {
        const size_t total = r.ids.size();
        const size_t from = P.from;
        const double t0 = P.t0;
        const size_t start = P.done;
        int64_t beat_at = -1;   // (the run's progress at the last beat)
        m.set_progress([&](int64_t d) {   // (a run of chunks: each one through every layer)
            // (a beat only when it moved: the run calls this every 50 ms with what is done so far - a wedged chunk run
            //  repeats its last value for hours, #539, and must not look alive to the watchdog)
            if (d + (int64_t) P.done != beat_at) {
                beat_at = d + (int64_t) P.done;
                beat();
            }
            const double ms = now_ms() - t0;
            std::printf("PP%s %zu %zu %.1f %.1f %zu\n", tag.c_str(), P.done + (size_t) d, total, ms,
                        ms > 0 ? 1000.0 * (double) (P.done + (size_t) d - from) / ms : 0.0, from);
            std::fflush(stdout);
        });
        // (DS4_LAST_WINDOW=1: the last token in a window of its own after the checkpoint, as before - for A/B runs)
        static const bool last_window = std::getenv("DS4_LAST_WINDOW") != nullptr;
        m.set_cancel(stop_now);
        std::string err;
        bool ok = true, stopped = false;
        while (ok && P.done < total) {
            if (stop_now()) {
                stopped = true;
                break;
            }
            if (P.done - start >= piece) break;
            const size_t n = std::min<size_t>({(size_t) m.feed_unit(), total - P.done, piece - (P.done - start)});
            // (the prompt's end: the last token in the same run, the checkpoint before it kept on the way)
            const bool fin = P.done + n == total;
            if (fin && last_window) {
                ok = (n < 2 || m.feed(r.ids.data() + P.done, (int) n - 1, 0, err));
                if (ok) m.checkpoint(1);
                ok = ok && m.feed(r.ids.data() + P.done + n - 1, 1, 1, err);
            } else
                ok = fin ? m.feed_mark(r.ids.data() + P.done, (int) n, 1, 1, err)
                         : m.feed(r.ids.data() + P.done, (int) n, 0, err);
            P.done += n;
            beat();
            if (ok && (P.done >= P.next_pp || P.done == total)) {
                const double ms = now_ms() - t0;
                std::printf("PP%s %zu %zu %.1f %.1f %zu\n", tag.c_str(), P.done, total, ms,
                            ms > 0 ? 1000.0 * (double) (P.done - from) / ms : 0.0, from);
                std::fflush(stdout);
                P.next_pp = P.done + 64;
            }
        }
        m.set_cancel(nullptr);
        m.set_progress(nullptr);
        if (stopped || (!ok && m.cancelled())) {
            m.set_images({});
            const bool back = from > 0 && m.restore((int64_t) from) && (size_t) m.pos() == from;
            if (!back) {
                m.reset();
                held.clear();
            }
            std::fprintf(stderr, "maya ds4 serve: stopped while reading the prompt of %zu tokens, after %.0f ms%s\n",
                         total, now_ms() - t0, back ? "; the conversation before it is kept" : "");
            std::printf("DONE%s 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0\n", tag.c_str(), total, now_ms() - t0, back ? from : (size_t) 0);
            std::fflush(stdout);
            return false;
        }
        if (!ok) {
            m.set_images({});
            std::fprintf(stderr, "ds4: %s\n", err.c_str());
            std::printf("ERR%s %s\n", tag.c_str(), err.c_str());
            std::printf("DONE%s 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", tag.c_str(), total);
            std::fflush(stdout);
            m.reset();
            held.clear();
            return false;
        }
        return true;
    };
    const auto prompt_end = [&](Prompt& P) {
        m.set_images({});
        {   // the pictures the state holds: the prompt's, and those the state kept short of `from` (a turn that adds no
            // picture still holds the ones the history above it has)
            std::vector<Pic> nh;
            nh.reserve(held.size() + P.pics.size());
            for (const Pic& x : held)
                if (x.len > 0 && (size_t) x.start + x.len <= P.from && !pic_same(P.pics, x)) nh.push_back(x);
            for (const Pic& x : P.pics) nh.push_back(x);
            held = std::move(nh);
        }
        m.checkpoint(0);
        if (!P.continued) m.pin();   // (a new run of prompts: where it may be rendered again from)
    };
    const auto prompt_in = [&](Req& r, bool geni, const std::string& tag, const std::function<bool()>& stop_now,
                               Prompt& P) -> bool {
        if (!prompt_begin(r, geni, tag, P) || !prompt_piece(r, tag, stop_now, P, r.ids.size())) return false;
        prompt_end(P);
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
        // a request's own seed: its draws from the start of its stream, so the same seed gives the same answer (the
        // counter ran on across requests: a seed was not reproducible); without one the stream runs on as before
        if (r.seed != 0) counter = 0;
        Prompt P;
        if (!prompt_in(r, geni, "", [&in] { return in.stop.load(); }, P)) continue;
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
    // recently used (whose conversation is set aside first with MAYA_DS4_SLOT_KEEP, as one conversation's is).  The
    // prompts being read take turns, a piece each (a chunk per pipeline stage: the split's stages all busy), and every
    // running request writes a token between two pieces - so a long prompt no longer holds every answer, and a short
    // question no longer waits for a long prompt read before it: a prompt whole in one piece goes ahead of the longer
    // reads (MAYA_DS4_READ_ORDER=arrival: each prompt whole as it comes, the others waiting).  A prompt with pictures is
    // read whole (the tower borrows its room and gives it back before the prompt, with every other request between
    // two of its tokens).  A request alone decodes in its rounds as with one conversation (DSpark's drafts, the prompt
    // lookup); two or more go through Model::step_multi - a token of each through the model together, no drafts.
    // The loop runs on events (ds4_evt.hpp, rig's evt): the reader's lines, and the turns - a round of tokens or a
    // prompt's piece - each queueing the next as it ends; a line outranks a turn, and with nothing running it waits.
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
            bool reading = false;   // its prompt is being read, a piece a turn
            bool ended = false;     // it ended while its prompt was read (its DONE written)
            unsigned long long counter = 0;   // (its own: its picks the same alone or beside others)
            int tok = -1;                     // written next, not fed yet
            std::vector<int> got;             // a round's tokens still to write, the last first
            bool flush = false;               // (multi draft) they came from a window of several: written in one turn
            double acc_rate = 1.0;            // (multi draft) its drafts' recent acceptance per draft, an average over its rounds
            double k_hat = 3.0;               // (multi draft) the drafts its cut leaves a round, an average
            int skip = 0;                     // (multi draft, floor rule) rounds left without drafts, under the floor
            int probe_in = 0, probe_gap = 0;  // (multi draft, measured) rounds until it drafts unplanned, the gap now
            long long rounds_d = 0, rounds_p = 0;   // (multi draft) rounds of several it drafted in / took a plain row in
            long long rounds_o = 0;                 // (multi draft) ... of the latter, rounds in which others drafted
            double ms_draft = 0.0;                  // (multi draft) its own draft() calls
            // (multi draft) its rounds of several by its acceptance when planned (< 0.5, 0.5-0.7, 0.7-0.85, >= 0.85):
            // how many, and in how many it drafted
            long long band_n[4] = {}, band_d[4] = {};
        };
        const auto acc_band = [](double p) { return p < 0.5 ? 0 : p < 0.7 ? 1 : p < 0.85 ? 2 : 3; };
        const int nseq = m.seq_count();
        const MultiDraft md = multi_draft_settings();
        MultiCost cost;   // (the measured policy's round and drafter times, over the engine's life)
        cost.fixed = md.fixed;
        cost.dfixed = md.dfixed;
        cost.pass = m.draft_pass();
        if (md.on && m.draft_block() > 0) {
            if (md.measured)
                std::fprintf(stderr, "maya ds4 serve: every conversation drafts (MAYA_DS4_MULTI_DRAFT): %d drafts each at most, "
                                     "a window of %d rows, the drafter %d conversations a pass; who drafts a round the measured "
                                     "round and drafter times decide (a plan must beat the plain step by %.0f%%; a probe every "
                                     "%d .. %d rounds otherwise)%s\n",
                             md.max_drafts > 0 ? std::min(md.max_drafts, m.draft_block()) : m.draft_block(), md.max_rows,
                             m.draft_pass(), 100.0 * md.margin, md.retry, md.probe_cap,
                             md.lockstep ? "" : "; tokens one a turn");
            else
                std::fprintf(stderr, "maya ds4 serve: every conversation drafts (MAYA_DS4_MULTI_DRAFT): %d drafts each at most, "
                                     "a window of %d rows, the floor %.2f .. %.2f at 1 .. %d answering%s\n",
                             md.max_drafts > 0 ? std::min(md.max_drafts, m.draft_block()) : m.draft_block(), md.max_rows,
                             multi_draft_floor(md, 1), multi_draft_floor(md, nseq), nseq, md.lockstep ? "" : "; tokens one a turn");
        }
        std::vector<long long> seq_rid((size_t) nseq, -1);   // the request in each conversation (-1: idle)
        std::vector<uint64_t> seq_used((size_t) nseq, 0);    // (the least recently used goes first)
        std::vector<std::vector<Pic>> held_of((size_t) nseq);   // the pictures each one holds (`held`: the bound one's)
        uint64_t use_clock = 0;
        std::vector<std::unique_ptr<Act>> acts;
        std::deque<std::string> waiting;   // GEN lines no conversation is free for yet
        std::deque<Act*> read_line;        // the prompts being read, in turn: a piece each, then to the back
        const bool read_pieces = [] {
            const char* v = std::getenv("MAYA_DS4_READ_ORDER");
            return v == nullptr || std::string(v) != "arrival";
        }();
        const size_t read_piece = m.prompt_chunk() > 0 ? (size_t) m.prompt_chunk() * (size_t) m.pipeline_stages()
                                                       : (size_t) m.feed_unit();
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
        const auto release = [&](const Act& a) {   // (its conversation free for the next request)
            seq_rid[(size_t) a.seq] = -1;
            seq_used[(size_t) a.seq] = ++use_clock;
            in.forget(a.r.rid);
            if (lk_owner == a.r.rid) lk_owner = -1;
        };
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
                         "%lld generated in %.0f ms (%.2f tok/s, %zu conversations running)\n",
                         a.r.rid, a.r.ids.size(), a.P.from, a.r.ids.size() - a.P.from, a.t1 - a.P.t0,
                         a.t1 > a.P.t0 ? 1000.0 * (double) (a.r.ids.size() - a.P.from) / (a.t1 - a.P.t0) : 0.0,
                         a.produced, t2 - a.t1, t2 > a.t1 ? 1000.0 * (double) a.produced / (t2 - a.t1) : 0.0, acts.size());
            if (md.on && m.draft_block() > 0)   // (its drafts beside others: offered / kept, the rounds of several with and without)
                std::fprintf(stderr,
                             "maya ds4 serve: request %lld: drafts %lld of %lld kept in %lld rounds of several, %lld rounds with "
                             "others' drafts only, %lld without any; its share of the drafter %.0f ms; acceptance %.2f, "
                             "%.1f drafts a round; drafted / rounds by its acceptance <0.5 %lld/%lld, 0.5-0.7 %lld/%lld, "
                             "0.7-0.85 %lld/%lld, >=0.85 %lld/%lld\n",
                             a.r.rid, a.d_acc, a.d_off, a.rounds_d, a.rounds_o, a.rounds_p - a.rounds_o, a.ms_draft,
                             a.acc_rate, a.k_hat, a.band_d[0], a.band_n[0], a.band_d[1], a.band_n[1], a.band_d[2],
                             a.band_n[2], a.band_d[3], a.band_n[3]);
            release(a);
        };
        // its prompt read: its first token picked, and it writes from the next turn on
        const auto start_answer = [&](Act& a) {
            prompt_end(a.P);
            a.reading = false;
            a.t1 = now_ms();
            a.sd = m.stats();
            a.limit = a.r.max_new > 0 ? a.r.max_new : m.max_context() - (long long) a.r.ids.size();
            a.in_think = a.r.think_budget > 0 && a.r.think_end >= 0;
            a.acc_rate = md.measured ? cost.gp : 1.0;   // (the measured policy: it starts from the engine's recent averages,
            a.k_hat = cost.gk;                          //  and drafts once in its first round of several unless planned)
            a.probe_in = 0;
            a.probe_gap = md.retry;
            penalties(a.r);
            a.tok = m.sample(a.r.temperature, a.r.top_p, a.r.top_k, a.r.min_p, a.r.seed, a.counter++);
        };
        // the next `piece` tokens of its prompt into its conversation (a STOP ends it here, its DONE written)
        const auto read_turn = [&](Act& a, size_t piece) {
            bind(a.seq);
            const long long rid = a.r.rid;
            if (!prompt_piece(a.r, " " + std::to_string(rid), [&in, rid] { return in.stopped(rid); }, a.P, piece)) {
                a.ended = a.done = true;
                return;
            }
            if (a.P.done == a.r.ids.size()) start_answer(a);
        };
        // a prompt whole in one piece joins the line ahead of the longer reads, behind the other one-piece prompts: it
        // is through in its turn, so a short question does not wait a round of long reads' pieces
        const auto one_piece = [&](const Act* a) { return a->r.ids.size() - a->P.done <= read_piece; };
        const auto join_read = [&](Act* a) {
            if (!one_piece(a)) {
                read_line.push_back(a);
                return;
            }
            auto at = read_line.begin();
            while (at != read_line.end() && one_piece(*at)) ++at;
            read_line.insert(at, a);
        };
        // a GEN line into a free conversation: its prompt begun, and read in turn (arrival, or with pictures: whole now)
        const auto admit = [&](const std::string& ln) {
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
            // the conversation: the idle one the prompt reuses the most of (its whole state continued, else a
            // checkpoint it can go back to), else an empty one, else the least recently used
            int best = -1;
            int64_t best_u = -1;
            for (int s = 0; s < nseq; ++s) {
                if (seq_rid[(size_t) s] >= 0) continue;
                bind(s);
                const auto& st = m.tokens();
                size_t c = 0;
                while (c < st.size() && c < r.ids.size() && st[c] == r.ids[c]) ++c;
                int64_t u = 0;
                if (!st.empty() && c == st.size() && st.size() < r.ids.size()) u = (int64_t) st.size();
                else if (c > 0) u = std::max<int64_t>(0, m.restorable((int64_t) std::min(c, r.ids.size() - 1)));
                const bool better = best < 0 || u > best_u ||
                                    (u == best_u && (st.empty() != m.seq_tokens(best).empty()
                                                         ? st.empty()
                                                         : seq_used[(size_t) s] < seq_used[(size_t) best]));
                if (better) best = s, best_u = u;
            }
            if (best < 0) {   // (the server gives each conversation one request: a request over that waits)
                waiting.push_front(ln);
                return;
            }
            bind(best);
            seq_used[(size_t) best] = ++use_clock;
            if (acts.empty()) m.begin_request();   // (DISK_SKIP_AUTO: between requests only - none runs now)
            const long long rid = r.rid;
            if (!prompt_begin(r, geni, " " + std::to_string(rid), a->P)) {
                in.forget(rid);
                return;
            }
            a->seq = best;
            a->reading = true;
            if (!read_pieces || !a->P.pics.empty()) {
                read_turn(*a, r.ids.size());
                if (a->ended) {
                    in.forget(rid);
                    return;
                }
            }
            seq_rid[(size_t) best] = rid;
            if (acts.empty()) {
                busy_t1 = now_ms();
                busy_n = 0;
                busy_s = m.stats();
                busy_io = io_read_bytes();
            }
            if (a->reading) join_read(a.get());
            acts.push_back(std::move(a));
        };
        // the loop's events (ds4_evt.hpp, rig's evt), run one at a time on this thread: the reader's lines, then the
        // turns - a token of every answer running (a round), or the next prompt's piece, in turn.  A line outranks a
        // turn, so every request that came is admitted before the next one; nothing polls: with no request running
        // the engine waits for a line
        enum : int { kTurn = 1, kLine = 2 };
        strata::ds4::evt::Engine eng;
        bool quit = false;
        bool turn_queued = false;   // (one turn in the queue at a time: each queues the next as it ends)
        bool read_next = false;     // (a round was the last turn: a prompt's piece goes next, if one is being read)
        const auto admit_waiting = [&] {
            while (!quit && !waiting.empty() && std::find(seq_rid.begin(), seq_rid.end(), -1LL) != seq_rid.end()) {
                const std::string g = waiting.front();
                waiting.pop_front();
                admit(g);
                beat();
            }
        };
        // each running request writes its next token; those whose next one the model must make go through it together
        // (MAYA_DS4_MULTI_DRAFT) a round of several: which of them draft, and how many drafts each (k[i]; 0: a plain
        // row); true when any does.  Each one's room (its tokens left this round) and whether its thinking closes here
        // are filled for the round.  The measured policy: of the plans "the j conversations with the best acceptance
        // draft, (rows - n) / j each at most, each as many as its cut usually leaves", the one expecting the most
        // tokens a millisecond - its tokens n + sum over the drafting of p + p^2 + .. + p^k (p its acceptance per
        // draft), its time the round's for that many rows plus j drafter calls, both as timed so far - against the
        // plain step's n / time(n); none taken unless it beats the step by the margin.  Then at most one of the
        // eligible left out whose probe gap is over drafts anyway, with the rows left; a conversation in the plan
        // gets the first gap back, one probing gets a gap twice as long (to the cap).  est[0]: the plain step's
        // tokens a millisecond, est[1]: the plan's (or the best plan's, when none was taken).
        // The floor rule (POLICY=0): every eligible conversation drafts (rows - n) / n at most, unless its acceptance
        // is under n / (fixed + n) - then a plain row for RETRY rounds and one probe.
        const auto multi_plan = [&](const std::vector<Act*>& need, std::vector<int>& k, std::vector<int>& roomv,
                                    std::vector<char>& closing_v, double* est) -> bool {
            const int n = (int) need.size();
            const int rows = std::min(md.max_rows, 16);
            const int kcap = md.max_drafts > 0 ? std::min(md.max_drafts, m.draft_block()) : m.draft_block();
            k.assign((size_t) n, 0);
            roomv.assign((size_t) n, 0);
            closing_v.assign((size_t) n, 0);
            est[0] = est[1] = 0.0;
            std::vector<int> el;   // the eligible (spec on, not resting, more than one token to go)
            for (int i = 0; i < n; ++i) {
                Act& a = *need[(size_t) i];
                int room = (int) std::min<long long>(a.limit - a.produced, m.max_context() - m.seq_pos(a.seq));
                const bool closing = close_think(a);
                closing_v[(size_t) i] = closing;
                if (closing) room = 1;
                else if (a.in_think) room = (int) std::min<long long>(room, a.r.think_budget - a.produced);
                roomv[(size_t) i] = room;
                if ((a.r.spec < 0 ? spec : a.r.spec != 0) && !a.cool && room > 1) el.push_back(i);
            }
            if (md.measured) est[0] = (double) n / cost.cost(n);
            if (el.empty() || rows - n < 1 || kcap < 1) return false;
            if (!md.measured) {
                const int per = std::min(kcap, (rows - n) / n);
                const double floor = multi_draft_floor(md, n);
                bool any = false;
                for (int i : el) {
                    Act& a = *need[(size_t) i];
                    bool draft_now = per > 0;
                    if (draft_now && a.acc_rate < floor) {   // under the floor: a plain row, every retry-th round a probe
                        if (a.skip > 0) {
                            --a.skip;
                            draft_now = false;
                        } else a.skip = md.retry;
                    }
                    if (!draft_now) continue;
                    k[(size_t) i] = per;
                    any = true;
                }
                return any;
            }
            std::sort(el.begin(), el.end(),
                      [&](int x, int y) { return need[(size_t) x]->acc_rate > need[(size_t) y]->acc_rate; });
            int bj = 0;
            std::vector<int> bk;
            for (int j = 1; j <= (int) el.size() && (rows - n) / j >= 1; ++j) {
                const int cap = std::min(kcap, (rows - n) / j);
                double tok = (double) n;
                int R = n;
                std::vector<int> kk((size_t) j);
                for (int u = 0; u < j; ++u) {
                    const int i = el[(size_t) u];
                    const Act& a = *need[(size_t) i];
                    const int ki = std::max(1, std::min({cap, roomv[(size_t) i] - 1, (int) std::lround(a.k_hat)}));
                    kk[(size_t) u] = ki;
                    R += ki;
                    double pm = 1.0;
                    for (int q = 0; q < ki; ++q) {
                        pm *= a.acc_rate;
                        tok += pm;
                    }
                }
                const double rate = tok / (cost.cost(R) + cost.draft(j));
                if (rate > est[1]) {
                    est[1] = rate;
                    bj = j;
                    bk = kk;
                }
            }
            bool go = bj > 0 && est[1] > est[0] * (1.0 + md.margin);
            if (!go && bj > 0 && est[1] > 0.9 * est[0]) {
                // (a guessed cost can keep the policy from ever timing a size - the plain steps time n rows and the
                //  probes n + a few, never a full plan's: the first three times the best plan's row count is untimed
                //  and the plan is not well behind the step, it is taken, as the lookup's policy tries a size)
                int Rb = n;
                for (int v : bk) Rb += v;
                go = cost.c_n[std::max(1, std::min(16, Rb))] < 3.0;
            }
            int R = n;
            for (int u = 0; go && u < bj; ++u) {
                const int i = el[(size_t) u];
                Act& a = *need[(size_t) i];
                k[(size_t) i] = bk[(size_t) u];
                R += bk[(size_t) u];
                a.probe_gap = md.retry;
                a.probe_in = md.retry;
            }
            bool probed = false;
            for (int u = go ? bj : 0; u < (int) el.size(); ++u) {
                const int i = el[(size_t) u];
                Act& a = *need[(size_t) i];
                if (!probed && a.probe_in <= 0 && R < rows) {
                    const int ki = std::max(1, std::min({kcap, rows - R, roomv[(size_t) i] - 1, (int) std::lround(a.k_hat)}));
                    k[(size_t) i] = ki;
                    R += ki;
                    probed = true;
                    a.probe_gap = std::min(md.probe_cap, std::max(md.retry, 2 * a.probe_gap));
                    a.probe_in = a.probe_gap;
                } else --a.probe_in;
            }
            return R > n;
        };
        // a request's token written (a.tok): its counts, its lookup history, its thinking's end; false once it is done
        // (stopped, an end token, its limit or the context's)
        const auto write_tok = [&](Act& a) -> bool {
            if (in.stopped(a.r.rid)) {
                a.finish = "cancel";
                a.done = true;
                return false;
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
                return false;
            }
            if (a.produced >= a.limit || m.seq_pos(a.seq) >= m.max_context()) {
                a.done = true;
                return false;
            }
            return true;
        };
        // (MAYA_DS4_MD_PIPE) two windows in flight, one on each GPU of a layer split: the model says whether it can
        std::string pipe_why;
        const bool pipe_on = md.on && m.draft_block() > 0 && m.pipe_ok(&pipe_why);
        if (md.on && m.draft_block() > 0 && std::getenv("MAYA_DS4_MD_PIPE") == nullptr &&
            std::getenv("STRATA_DS4_MD_PIPE") == nullptr)
            pipe_why.clear();
        if (pipe_on)
            std::fprintf(stderr, "maya ds4 serve: two windows in flight (MAYA_DS4_MD_PIPE): the answering conversations in "
                                 "two groups, each %d window%s a round, one group's second stage beside the other's first; "
                                 "a window priced by %s (MAYA_DS4_MD_PIPE_PRICE)%s%s\n",
                         md.pipe_windows, md.pipe_windows == 1 ? "" : "s",
                         md.pipe_price == 0   ? "an equal share of the round"
                         : md.pipe_price == 1 ? "a share of the round by its rows' cost"
                                              : "its stages' GPU time",
                         md.pipe_draft_gpu ? ", a drafter pass by its GPU time (MAYA_DS4_MD_PIPE_DRAFT_PRICE=gpu)" : "",
                         md.pipe_serial ? " - MAYA_DS4_MD_PIPE_SERIAL: the same windows one at a time (the reference)" : "");
        else if (!pipe_why.empty())
            std::fprintf(stderr, "maya ds4 serve: MAYA_DS4_MD_PIPE is off: it needs %s\n", pipe_why.c_str());
        const auto round_turn = [&] {
            std::vector<Act*> need;
            std::vector<int> kplan, roomv;   // (multi draft) the round's plan: drafts each, tokens left each
            std::vector<char> closing_v;     // (multi draft) whose thinking closes at this round
            double est[2] = {0.0, 0.0};      // (multi draft) the plain step's and the plan's expected tokens a ms
            const long long n0 = busy_n;
            for (auto& ap : acts) {
                Act& a = *ap;
                if (a.reading) continue;
                // (a token a turn; the tokens a window of several conversations gave it all in this turn - so it is in
                //  the next round with the others, MAYA_DS4_MULTI_DRAFT_LOCKSTEP)
                for (;;) {
                    if (!write_tok(a)) break;
                    if (a.got.empty()) {
                        need.push_back(&a);
                        break;
                    }
                    a.tok = a.got.back();
                    a.got.pop_back();
                    if (!a.flush) break;
                }
            }
            if (busy_n / 8 != n0 / 8) {   // the Monitor's line, every 8 tokens of all of them
                std::string st = stat_text(busy_t1, busy_s, busy_io, busy_n, 0, 0);
                st += " active=" + std::to_string(acts.size());
                std::printf("%s\n", st.c_str());
            }
            std::fflush(stdout);
            // (multi draft) a window's row of a request: its token and sampling, as verify_multi takes them
            const auto spec_of = [&](const Act& a, strata::ds4::Model::MultiSpec& r) {
                r.seq = a.seq;
                r.tok = a.tok;
                r.temperature = a.r.temperature;
                r.top_p = a.r.top_p;
                r.top_k = a.r.top_k;
                r.min_p = a.r.min_p;
                r.seed = a.r.seed;
                r.counter = a.counter;
                r.pen_n = a.r.pen_n < 0 ? INT_MAX : a.r.pen_n;
                r.pen_repeat = std::isfinite(a.r.pen_repeat) && a.r.pen_repeat > 0.0f ? a.r.pen_repeat : 1.0f;
                r.pen_freq = std::isfinite(a.r.pen_freq) ? a.r.pen_freq : 0.0f;
                r.pen_present = std::isfinite(a.r.pen_present) ? a.r.pen_present : 0.0f;
            };
            // (multi draft) a request's drafts cut as the lone round cuts them (the confidence floor, the cumulative
            // cut), then to the plan's count; the cut alone (keep_raw) is what the policy expects of it next time
            const auto cut_for = [&](Act& a, int kmax, int room, std::vector<int>& drafts, const std::vector<float>& conf) {
                int keep_raw = 0;
                float cum_min = a.r.cum >= 0 ? a.r.cum : dspark_cum_default(), cum = 1.0f;
                const float cmin = a.r.conf >= 0 ? a.r.conf : conf_min;
                while (keep_raw < (int) drafts.size() && keep_raw < room - 1 && conf[(size_t) keep_raw] >= cmin &&
                       (cum *= conf[(size_t) keep_raw]) >= cum_min)
                    ++keep_raw;
                const int keep = std::min(keep_raw, kmax);
                drafts.resize((size_t) keep);
                a.k_hat = 0.75 * a.k_hat + 0.25 * (double) keep_raw;
                cost.gk = 0.9 * cost.gk + 0.1 * (double) keep_raw;
                return keep;
            };
            // (multi draft) a request's verdict landed: its counts and averages, its tokens to write (the accepted
            // drafts then the next token, the last first, as a lone round's) - how many it got
            const auto land_verdict = [&](Act& a, const strata::ds4::Model::MultiSpec& r, bool tried, bool closing,
                                          int& off_all, int& acc_all) {
                const int off = (int) r.drafts.size(), acc = r.accepted;
                {   // (its acceptance as the plan saw it: the average is updated below)
                    const int b = acc_band(a.acc_rate);
                    ++a.band_n[b];
                    if (tried) ++a.band_d[b];
                }
                // (every draft under its confidence: the next round without; a round without clears it, as the lone
                //  path's unconditional assignment does)
                a.cool = tried && off == 0;
                if (tried) {
                    ++a.rounds_d;
                    a.d_off += off;
                    a.d_acc += acc;
                    off_all += off;
                    acc_all += acc;
                    if (off > 0) {   // (one draft is a yes or no: it weighs less than three)
                        const double w = std::min(0.4, 0.1 * (double) off);
                        a.acc_rate = (1.0 - w) * a.acc_rate + w * ((double) acc / (double) off);
                        cost.see_drafts(off, acc);
                    }
                } else {
                    ++a.rounds_p;
                    ++a.rounds_o;   // (a plain row beside others' drafts: the branch runs only with some)
                }
                a.counter += (unsigned long long) acc + 1;
                a.got.clear();
                a.got.push_back(closing ? a.r.think_end : r.next);
                for (int j = acc - 1; j >= 0; --j) a.got.push_back(r.drafts[(size_t) j]);
                a.tok = a.got.back();
                a.got.pop_back();
                a.flush = md.lockstep;
                return acc + 1;
            };
            if (need.size() == 1 && acts.size() == 1) {   // alone: its rounds, as with one conversation
                Act& a = *need[0];
                bind(a.seq);
                int room = (int) std::min<long long>(a.limit - a.produced, m.max_context() - m.pos());
                const bool closing = close_think(a);
                if (closing) room = 1;
                else if (a.in_think) room = (int) std::min<long long>(room, a.r.think_budget - a.produced);
                Lookup* L = a.r.lookup == 0 ? nullptr : lk;
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
                    a.flush = false;
                }
            } else if (need.size() == 1) {   // one of several: a plain step of its conversation
                Act& a = *need[0];
                bind(a.seq);
                if (!m.step(a.tok, true, err)) {
                    std::fprintf(stderr, "ds4: %s\n", err.c_str());
                    a.finish = "cancel";
                    a.done = true;
                } else {
                    penalties(a.r);
                    const bool closing = close_think(a);
                    a.tok = m.sample(a.r.temperature, a.r.top_p, a.r.top_k, a.r.min_p, a.r.seed, a.counter++);
                    if (closing) a.tok = a.r.think_end;
                }
            } else if (need.size() > 1 && pipe_on && m.pipe_ok()) {
                // (MAYA_DS4_MD_PIPE) the answering conversations in two groups, each a window of its own (its plan,
                // its drafters in one pass, its verdicts: as the branch below gives one window), the two windows in
                // flight at once - one's second stage on the second GPU while the other's first stage runs on the
                // first (Model::verify_multi_issue / verify_multi_take) - and md.pipe_windows windows a group a round:
                // issue A, issue B, take A, issue A's next, take B, issue B's next, ...  A group's next window takes
                // its verdicts' tokens: written now, all but the last (the trunk's token after the kept drafts), which
                // is the next window's input and is written at the next round's top, as a lone round's is.  A request
                // that ends with it (an end token, its limit) leaves the group.  The policy times each window as its
                // share of the round
                struct Grp {
                    std::vector<Act*> who;
                    std::vector<strata::ds4::Model::MultiSpec> rs;
                    std::vector<char> tried, closing;
                    int R = 0, slot = 0;
                    bool live = false;
                    bool unwritten = false;   // its input tokens are not written yet (a group's later windows: the
                                              // previous window's last token) - written first when it lands
                    double e2[2] = {0.0, 0.0};   // its plan's: the plain step's and the plan's expected tokens a ms
                    double p_used = -1.0;        // the drafting conversations' mean acceptance the plan used (-1: none)
                };
                // each window taken this round: its rows, its stages' GPU times (serial: its own time as both), its plan
                struct WinRec {
                    int R = 0, slot = 0;
                    float s0 = -1.0f, s1 = -1.0f;
                    double e2[2] = {0.0, 0.0}, p_used = -1.0;
                };
                std::vector<WinRec> wrec;
                Grp G[2];
                const int n = (int) need.size(), nA = (n + 1) / 2;
                for (int i = 0; i < n; ++i) G[i < nA ? 0 : 1].who.push_back(need[(size_t) i]);
                G[1].slot = 1;
                const double t_round = now_ms();
                const double dr_b0 = m.stats().draft_block_ms, dr_h0 = m.stats().draft_head_ms;   // (the log line's)
                double ms_d = 0.0;
                double ms_dc = 0.0;   // (the drafter's passes as priced: their wall time, or their GPU time)
                int toks = 0, windows = 0, off_all = 0, acc_all = 0;
                bool failed = false;
                std::vector<int> rows_of;   // each window's rows, taken in order (the policy's cost a window)
                std::string who_s;
                // the token a request writes next ends it (as the round's top would find): no window for it
                const auto ends_next = [&](const Act& a) {
                    if (a.done || in.stopped(a.r.rid)) return true;
                    for (int e : eos)
                        if (a.tok == e) return true;
                    return a.produced + 1 >= a.limit || m.seq_pos(a.seq) >= m.max_context();
                };
                // the group's plan, its drafts, its window's rows (again: a later window of the round, its input
                // tokens - the previous window's last - not written yet)
                const auto prepare = [&](Grp& g, bool again) -> bool {
                    const int ng = (int) g.who.size();
                    g.unwritten = again;
                    std::vector<int> kp, rv;
                    std::vector<char> cl;
                    double e2[2] = {0.0, 0.0};
                    // (its token to write next is not written yet: the plan sees it as written, as the round's top
                    //  would have, so a window near its limit or its thinking budget is cut as the branch below cuts it)
                    for (Act* a : g.who) ++a->produced;
                    const bool planned = multi_plan(g.who, kp, rv, cl, e2);
                    for (Act* a : g.who) --a->produced;
                    if (!planned) kp.assign((size_t) ng, 0);   // (no plan: every row plain, in the window all the same)
                    g.e2[0] = e2[0];
                    g.e2[1] = e2[1];
                    {
                        double ps = 0.0;
                        int pn = 0;
                        for (int i = 0; i < ng; ++i)
                            if (kp[(size_t) i] > 0) {
                                ps += g.who[(size_t) i]->acc_rate;
                                ++pn;
                            }
                        g.p_used = pn > 0 ? ps / (double) pn : -1.0;
                    }
                    g.rs.assign((size_t) ng, strata::ds4::Model::MultiSpec());
                    g.tried.assign((size_t) ng, 0);
                    g.closing = cl;
                    std::vector<int> dseq, dtok, dwho;
                    for (int i = 0; i < ng; ++i)
                        if (kp[(size_t) i] > 0) {
                            dseq.push_back(g.who[(size_t) i]->seq);
                            dtok.push_back(g.who[(size_t) i]->tok);
                            dwho.push_back(i);
                        }
                    std::vector<std::vector<int>> dv;
                    std::vector<std::vector<float>> cv;
                    if (!dseq.empty()) {
                        const double g0 = m.stats().draft_block_ms + m.stats().draft_head_ms;
                        const double td = now_ms();
                        if (!m.draft_multi(dseq.data(), dtok.data(), (int) dseq.size(), dv, cv, err)) return false;
                        const double dt = now_ms() - td;
                        const double dg = m.stats().draft_block_ms + m.stats().draft_head_ms - g0;
                        const double dc = md.pipe_draft_gpu && dg > 0.0 ? dg : dt;   // (untimed: the wall time)
                        ms_d += dt;
                        ms_dc += dc;
                        cost.see_draft((int) dseq.size(), dc);
                        for (int i : dwho) g.who[(size_t) i]->ms_draft += dt / (double) dseq.size();
                    }
                    g.R = ng;
                    for (int i = 0, di = 0; i < ng; ++i) {
                        Act& a = *g.who[(size_t) i];
                        strata::ds4::Model::MultiSpec& r = g.rs[(size_t) i];
                        spec_of(a, r);
                        if (kp[(size_t) i] <= 0) continue;
                        std::vector<int>& drafts = dv[(size_t) di];
                        const std::vector<float>& conf = cv[(size_t) di];
                        ++di;
                        g.R += cut_for(a, kp[(size_t) i], rv[(size_t) i], drafts, conf);
                        r.drafts = std::move(drafts);
                        g.tried[(size_t) i] = 1;
                    }
                    return true;
                };
                const auto issue = [&](Grp& g) -> bool {   // its window issued (serial: run when it lands)
                    if (!md.pipe_serial && !m.verify_multi_issue(g.rs.data(), (int) g.who.size(), g.slot, err))
                        return false;
                    g.live = true;
                    return true;
                };
                const auto launch = [&](Grp& g, bool again) { return prepare(g, again) && issue(g); };
                const auto land = [&](Grp& g) -> bool {   // the window taken: verdicts, tokens written, the ended out
                    const int ng = (int) g.who.size();
                    const double tv = now_ms();
                    if (md.pipe_serial ? !m.verify_multi(g.rs.data(), ng, err) : !m.verify_multi_take(g.rs.data(), ng, g.slot, err))
                        return false;
                    g.live = false;
                    ++windows;
                    rows_of.push_back(g.R);
                    {
                        WinRec wr;
                        wr.R = g.R;
                        wr.slot = g.slot;
                        if (md.pipe_serial) wr.s0 = wr.s1 = (float) (now_ms() - tv);   // (one at a time: its own time)
                        else m.pipe_window_ms(g.slot, &wr.s0, &wr.s1);
                        wr.e2[0] = g.e2[0];
                        wr.e2[1] = g.e2[1];
                        wr.p_used = g.p_used;
                        wrec.push_back(wr);
                    }
                    std::vector<Act*> on;
                    for (int i = 0; i < ng; ++i) {
                        Act& a = *g.who[(size_t) i];
                        const strata::ds4::Model::MultiSpec& r = g.rs[(size_t) i];
                        // (a later window's input token first: the previous window's last, fed but not written)
                        if (g.unwritten && !write_tok(a)) {
                            land_verdict(a, r, g.tried[(size_t) i] != 0, g.closing[(size_t) i] != 0, off_all, acc_all);
                            a.got.clear();   // (it ended at its input: the window's tokens are not written)
                            a.flush = false;
                            continue;
                        }
                        toks += land_verdict(a, r, g.tried[(size_t) i] != 0, g.closing[(size_t) i] != 0, off_all, acc_all);
                        if (md.log > 0) {
                            char b[48];
                            std::snprintf(b, sizeof b, "%s%lld:%d/%d", who_s.empty() ? "" : " ", a.r.rid, r.accepted,
                                          (int) r.drafts.size());
                            who_s += b;
                        }
                        // written now, all but the last (the next window's input, written at the next round's top)
                        bool going = true;
                        while (going && !a.got.empty()) {
                            going = write_tok(a);
                            if (!going) break;
                            a.tok = a.got.back();
                            a.got.pop_back();
                        }
                        a.flush = false;
                        if (going && g.closing[(size_t) i]) a.in_think = false;   // (its end token is the next written)
                        if (going && !ends_next(a)) on.push_back(&a);
                    }
                    g.who = on;
                    return true;
                };
                // the expert cache's tick held for the round (Model::cache_hold): no switch of the expert tables
                // while a window's requests are out, and MAYA_DS4_MD_PIPE_SERIAL ticks where the pipe does (at the
                // round's end), so the reference's experts move at the same points
                m.cache_hold(true);
                // both groups' drafts first, then both windows: a pass drafted with a window in flight queues its
                // head behind that window's stage and head on the head device's stream (the host would wait a stage
                // before issuing the second window); between a take and the next issue it does not (that window's
                // second stage is enqueued at the issue, after the pass)
                for (int gi = 0; gi < 2 && !failed; ++gi)
                    if (!G[gi].who.empty() && !prepare(G[gi], false)) failed = true;
                for (int gi = 0; gi < 2 && !failed; ++gi)
                    if (!G[gi].who.empty() && !issue(G[gi])) failed = true;
                for (int w = 0; w < md.pipe_windows && !failed; ++w)
                    for (int gi = 0; gi < 2 && !failed; ++gi) {
                        if (!G[gi].live) continue;
                        if (!land(G[gi])) {
                            failed = true;
                            break;
                        }
                        if (w + 1 < md.pipe_windows && !G[gi].who.empty() && !launch(G[gi], true)) failed = true;
                    }
                // the hold released, the round's tick now (every window taken; after a failure, every window out of
                // flight first - pipe_reset - so the tick runs now, with nothing in flight, not at some later take)
                if (failed) m.pipe_reset();
                m.cache_hold(false);
                if (failed) {
                    std::fprintf(stderr, "ds4: %s\n", err.c_str());
                    for (Act* a : need) {
                        a->finish = "cancel";
                        a->done = true;
                    }
                } else {
                    const double ms_w = now_ms() - t_round;
                    // each window's price (MAYA_DS4_MD_PIPE_PRICE): the windows overlap, so none has a time of its own
                    // on the wall clock.  equal: the round's time less the drafter's, an equal share each.  share: the
                    // same time in proportion to the cost model's current cost of each window's rows.  stage: each
                    // window's own GPU time - the slower of its two stages, the other stage running beside it the
                    // other window's - plus an equal share of what the round spent beyond the stages and the drafter
                    // (the host's verdicts and issues, the pipe's fill and drain).  A serving window's cost grows far
                    // less with its rows than a lone window's shape says (fixed + rows), so a shape-based share
                    // over-charges the drafted (big) windows and under-charges the plain (small) ones
                    const int nw = (int) wrec.size();
                    const double busy = std::max(0.0, ms_w - ms_dc);
                    std::vector<double> price((size_t) nw, 0.0);
                    int mode = md.pipe_price;
                    if (mode == 2) {
                        double sum = 0.0;
                        for (const WinRec& w : wrec) {
                            if (w.s0 < 0.0f || w.s1 < 0.0f) mode = 1;   // (a stage not timed: the share this round)
                            sum += std::max(w.s0, w.s1);
                        }
                        if (mode == 2) {
                            const double extra = std::max(0.0, busy - sum) / (double) std::max(1, nw);
                            for (int i = 0; i < nw; ++i)
                                price[(size_t) i] = std::max(wrec[(size_t) i].s0, wrec[(size_t) i].s1) + extra;
                        }
                    }
                    if (mode == 1) {
                        std::vector<double> c((size_t) nw);
                        double den = 0.0;
                        for (int i = 0; i < nw; ++i) den += (c[(size_t) i] = cost.cost(wrec[(size_t) i].R));
                        for (int i = 0; i < nw; ++i)
                            price[(size_t) i] = den > 0.0 ? busy * c[(size_t) i] / den : busy / (double) nw;
                    } else if (mode == 0)
                        for (int i = 0; i < nw; ++i) price[(size_t) i] = busy / (double) std::max(1, nw);
                    for (int i = 0; i < nw; ++i) cost.see_round(wrec[(size_t) i].R, price[(size_t) i]);
                    if (md.log > 0)
                        for (int i = 0; i < nw; ++i) {
                            const WinRec& w = wrec[(size_t) i];
                            std::fprintf(stderr,
                                         "maya ds4 multi draft pipe window: rows %d s0 %.2f s1 %.2f priced %.2f mode %s "
                                         "est_plain %.1f est_plan %.1f p_used %.3f\n",
                                         w.R, w.s0, w.s1, price[(size_t) i], pipe_price_name(mode), 1000.0 * w.e2[0],
                                         1000.0 * w.e2[1], w.p_used);
                        }
                    if (md.log > 0) {
                        std::string rows_s;
                        for (int r : rows_of) rows_s += (rows_s.empty() ? "" : "+") + std::to_string(r);
                        const strata::ds4::Stats& s1 = m.stats();
                        std::fprintf(stderr,
                                     "maya ds4 multi draft pipe: %d answering, %d windows of %s rows: drafter %.1f ms (its "
                                     "blocks %.1f, its head %.1f on the GPUs), round %.1f ms, %d tokens (%.0f tok/s); "
                                     "kept/offered %s\n",
                                     n, windows, rows_s.c_str(), ms_d, s1.draft_block_ms - dr_b0, s1.draft_head_ms - dr_h0,
                                     ms_w, toks, 1000.0 * (double) toks / ms_w, who_s.c_str());
                    }
                }
            } else if (need.size() > 1 && md.on && m.draft_block() > 0 && multi_plan(need, kplan, roomv, closing_v, est)) {
                // several, some drafting (MAYA_DS4_MULTI_DRAFT): the plan's conversations draft in one pass of the
                // drafter (each cut as a lone round's and to its plan's count), then one window verifies them all -
                // the others take a plain row in it.  No plan: the plain step below
                const int n = (int) need.size();
                std::vector<strata::ds4::Model::MultiSpec> rs((size_t) n);
                std::vector<char> tried((size_t) n, 0);
                bool failed = false;
                double ms_d = 0.0;
                int R = n;
                std::vector<int> dseq, dtok, dwho;   // the drafting conversations, their tokens, their rows of need
                for (int i = 0; i < n; ++i)
                    if (kplan[(size_t) i] > 0) {
                        dseq.push_back(need[(size_t) i]->seq);
                        dtok.push_back(need[(size_t) i]->tok);
                        dwho.push_back(i);
                    }
                std::vector<std::vector<int>> dv;
                std::vector<std::vector<float>> cv;
                if (!dseq.empty()) {
                    // (the drafter's argmax, as a lone greedy round's and as verify_multi takes them; its time is the
                    //  pass's, shared out over the drafting)
                    const double td = now_ms();
                    if (!m.draft_multi(dseq.data(), dtok.data(), (int) dseq.size(), dv, cv, err)) failed = true;
                    else {
                        const double dt = now_ms() - td;
                        ms_d = dt;
                        cost.see_draft((int) dseq.size(), dt);
                        for (int i : dwho) need[(size_t) i]->ms_draft += dt / (double) dseq.size();
                    }
                }
                for (int i = 0, di = 0; i < n && !failed; ++i) {
                    Act& a = *need[(size_t) i];
                    strata::ds4::Model::MultiSpec& r = rs[(size_t) i];
                    spec_of(a, r);
                    const int kmax = kplan[(size_t) i];
                    if (kmax <= 0) continue;
                    std::vector<int>& drafts = dv[(size_t) di];
                    std::vector<float>& conf = cv[(size_t) di];
                    ++di;
                    R += cut_for(a, kmax, roomv[(size_t) i], drafts, conf);
                    r.drafts = std::move(drafts);
                    tried[(size_t) i] = 1;
                }
                const double tw = now_ms();
                if (!failed && !m.verify_multi(rs.data(), n, err)) failed = true;
                if (failed) {
                    std::fprintf(stderr, "ds4: %s\n", err.c_str());
                    for (Act* a : need) {
                        a->finish = "cancel";
                        a->done = true;
                    }
                } else {
                    int toks = 0, off_all = 0, acc_all = 0;
                    for (int i = 0; i < n; ++i)
                        toks += land_verdict(*need[(size_t) i], rs[(size_t) i], tried[(size_t) i] != 0,
                                             closing_v[(size_t) i] != 0, off_all, acc_all);
                    const double ms_w = now_ms() - tw;
                    cost.see_round(R, ms_w);
                    if (md.log > 0) {
                        std::string who;
                        for (int i = 0; i < n; ++i) {
                            char b[48];
                            std::snprintf(b, sizeof b, "%s%lld:%d/%d", i ? " " : "", need[(size_t) i]->r.rid,
                                          rs[(size_t) i].accepted, (int) rs[(size_t) i].drafts.size());
                            who += b;
                        }
                        std::fprintf(stderr,
                                     "maya ds4 multi draft: %d answering, %d rows: drafter %.1f ms, window %.1f ms, %d tokens "
                                     "(%.0f tok/s; expected %.0f, the plain step %.0f); kept/offered %s\n",
                                     n, R, ms_d, ms_w, toks, 1000.0 * (double) toks / (ms_d + ms_w), 1000.0 * est[1],
                                     1000.0 * est[0], who.c_str());
                    }
                }
            } else if (need.size() > 1) {   // several: a token of each through the model together
                const double t0 = now_ms();
                std::vector<int> sv, tv;
                for (Act* a : need) {
                    sv.push_back(a->seq);
                    tv.push_back(a->tok);
                }
                if (!m.step_multi(sv.data(), tv.data(), (int) sv.size(), err)) {
                    std::fprintf(stderr, "ds4: %s\n", err.c_str());
                    for (Act* a : need) {
                        a->finish = "cancel";
                        a->done = true;
                    }
                } else {
                    for (size_t i = 0; i < need.size(); ++i) {
                        Act& a = *need[i];
                        penalties(a.r);
                        const bool closing = close_think(a);
                        a.tok = m.sample_multi((int) i, a.r.temperature, a.r.top_p, a.r.top_k, a.r.min_p, a.r.seed,
                                               a.counter++);
                        if (closing) a.tok = a.r.think_end;
                        a.cool = false;   // (a round without drafts clears the rest, as the lone path's does)
                        a.flush = false;
                        ++a.rounds_p;
                        if (md.on && m.draft_block() > 0) ++a.band_n[acc_band(a.acc_rate)];
                    }
                    if (md.on && m.draft_block() > 0) {   // (the measured policy: a plain step of n rows is a round of n)
                        const double ms = now_ms() - t0;
                        cost.see_round((int) need.size(), ms);
                        if (md.log > 0)
                            std::fprintf(stderr, "maya ds4 multi draft: %d answering, plain step %.1f ms (%.0f tok/s; "
                                                 "the best plan expected %.0f, the plain step %.0f)\n",
                                         (int) need.size(), ms, 1000.0 * (double) need.size() / ms, 1000.0 * est[1],
                                         1000.0 * est[0]);
                    }
                }
            }
        };
        // the next prompt in line reads a piece (its first token picked once it is all read), then goes to the back
        const auto read_turn_next = [&] {
            Act* a = read_line.front();
            read_line.pop_front();
            if (!a->done) read_turn(*a, read_piece);
            if (a->reading && !a->done) read_line.push_back(a);
        };
        std::function<void()> settle;
        // the next turn: a piece after a round while a prompt is being read, else a round while an answer runs
        const auto queue_turn = [&] {
            if (turn_queued || acts.empty()) return;
            turn_queued = true;
            eng.add([&] {
                turn_queued = false;
                const bool answering = std::any_of(acts.begin(), acts.end(),
                                                   [](const std::unique_ptr<Act>& a) { return !a->reading; });
                if (!read_line.empty() && (read_next || !answering)) {
                    read_turn_next();
                    read_next = false;
                } else if (answering) {
                    round_turn();
                    read_next = true;
                }
                settle();
            }, kTurn);
        };
        // after every event: a STOP of a prompt waiting for its turn (ended now, the conversation before it kept), the
        // requests ended (their conversations to the requests waiting), then the next turn - or, none running, idle
        settle = [&] {
            for (auto& ap : acts)
                if (ap->reading && !ap->done && in.stopped(ap->r.rid)) read_turn(*ap, 0);
            bool freed = false;
            for (size_t i = 0; i < acts.size();) {
                if (!acts[i]->done) {
                    ++i;
                    continue;
                }
                std::unique_ptr<Act> a = std::move(acts[i]);
                acts.erase(acts.begin() + (long) i);
                read_line.erase(std::remove(read_line.begin(), read_line.end(), a.get()), read_line.end());
                if (a->ended) release(*a);   // (its DONE written as its prompt was read)
                else finish_act(*a);
                freed = true;
            }
            if (freed) admit_waiting();
            if (!acts.empty()) {
                queue_turn();
                return;
            }
            if (quit) {
                eng.stop();
                return;
            }
            if (waiting.empty()) {
                g_busy.store(false);   // (waiting for a request: no watchdog)
                m.save_profile();
            }
        };
        // a line: a request (admitted when a conversation is free), QUIT (every request stopped, the engine's end once
        // they are), anything else answered at once (STOP <rid> is the reader's)
        const auto on_line = [&](const std::string& ln) {
            g_busy.store(true);
            beat();
            if (ln == "QUIT") {
                quit = true;
                in.stop_all.store(true);
            } else if (ln.rfind("GEN ", 0) == 0 || ln.rfind("GENI ", 0) == 0) {
                waiting.push_back(ln);
                admit_waiting();
            } else {
                std::printf("ERR unsupported command '%s' (GEN, GENI, STOP and QUIT)\n",
                            ln.substr(0, ln.find(' ')).c_str());
                std::fflush(stdout);
            }
            settle();
        };
        std::thread reader([&] {   // (the reader's lines into the loop; the input's end is a QUIT)
            std::string l;
            bool quit_seen = false;
            while (!quit_seen && in.next(l)) {
                quit_seen = l == "QUIT";
                eng.add([&, l] { on_line(l); }, kLine);
            }
            if (!quit_seen) eng.add([&] { on_line("QUIT"); }, kLine);
        });
        eng.run();
        reader.join();
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
    return tokens_same && same_rows == rows ? 0 : 1;
}

// --multi-check with the drafter loaded and on: the same N conversations answer n greedy tokens alone in DSpark
// rounds (draft, the lone round's cut, verify - one conversation after another), then all together with every one
// drafting (Model::verify_multi, the drafts cut as the serve loop cuts them: to what fits the window).  Each
// conversation's tokens must be the same both ways (the window's rows and verdicts are its own either way; only how
// many drafts a round takes differs).  Reports the drafts offered and kept, and the rounds' speed.
int check_multi_spec(strata::ds4::Model& m, const std::vector<int>& ids, int n, float conf_min) {
    const int N = m.seq_count();
    const int block = m.draft_block();
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
    // the drafts after tok, cut as a round's: the floor, the cumulative cut, at most kmax
    const auto cut_given = [&](int kmax, std::vector<int>& drafts, const std::vector<float>& conf) {
        int keep = 0;
        const float cum_min = dspark_cum_default();
        float cum = 1.0f;
        while (keep < (int) drafts.size() && keep < kmax && conf[(size_t) keep] >= conf_min &&
               (cum *= conf[(size_t) keep]) >= cum_min)
            ++keep;
        drafts.resize((size_t) keep);
    };
    const auto cut = [&](int tok, int kmax, std::vector<int>& drafts) -> bool {
        std::vector<float> conf;
        if (!m.draft(tok, drafts, conf, err, 0.0f, 1.0f, 0, 0.0f, 0, 0)) return false;
        int keep = 0;
        const float cum_min = dspark_cum_default();
        float cum = 1.0f;
        while (keep < (int) drafts.size() && keep < kmax && conf[(size_t) keep] >= conf_min &&
               (cum *= conf[(size_t) keep]) >= cum_min)
            ++keep;
        drafts.resize((size_t) keep);
        return true;
    };
    std::vector<std::vector<int>> alone((size_t) N), together((size_t) N);
    double ms_alone = 0, ms_together = 0;
    long long off_alone = 0, acc_alone = 0, off_together = 0, acc_together = 0, rounds_alone = 0, rounds_together = 0;
    // DS4_MCHECK_ROWS=1: every row's logits both ways, by (position, token) - the first row that differs, where it sat
    // in the window and by how much (a last-bits difference, or not)
    const bool keep_rows = std::getenv("DS4_MCHECK_ROWS") != nullptr;
    // (MAYA_DS4_MD_PIPE at load, and a layout that allows it: the together phase runs two windows in flight; the
    // per-row logits are not kept then - Model::logits gives one window's rows)
    std::string pipe_why;
    const bool pipe = m.pipe_ok(&pipe_why);
    // (asked for and refused: the comparison the setting is checked by did not run - the check fails rather than
    //  passing on what it did not compare)
    bool pipe_refused = false;
    if (pipe) std::printf("multi check (drafting): two windows in flight (MAYA_DS4_MD_PIPE)\n");
    else if (std::getenv("MAYA_DS4_MD_PIPE") != nullptr || std::getenv("STRATA_DS4_MD_PIPE") != nullptr) {
        std::printf("multi check (drafting): MAYA_DS4_MD_PIPE is off: it needs %s - NOT CHECKED\n", pipe_why.c_str());
        pipe_refused = true;
    }
    struct RowInfo { std::vector<float> lg; int R = 0, seg = 0, t = 0, T = 0; };
    std::vector<std::map<std::pair<int64_t, int>, RowInfo>> rows_alone((size_t) N), rows_together((size_t) N);
    std::vector<float> lgw;
    const int V = m.vocab();
    for (int i = 0; i < N; ++i) {
        if (!prompt(i)) {
            std::fprintf(stderr, "multi check (drafting): %s\n", err.c_str());
            return 1;
        }
        std::vector<int>& out = alone[(size_t) i];
        int tok = m.sample(0.0f, 1.0f, 0, 0.0f, 0, 0);
        out.push_back(tok);
        uint64_t counter = 0;
        const double t0 = now_ms();
        while ((int) out.size() < n) {
            std::vector<int> drafts;
            if (!cut(tok, std::min(block, n - (int) out.size() - 1), drafts)) {
                std::fprintf(stderr, "multi check (drafting): %s\n", err.c_str());
                return 1;
            }
            int acc = 0, next = -1;
            const int64_t p0 = m.pos();
            if (drafts.empty()) {
                if (!m.step(tok, true, err)) {
                    std::fprintf(stderr, "multi check (drafting): %s\n", err.c_str());
                    return 1;
                }
                next = m.sample(0.0f, 1.0f, 0, 0.0f, 0, counter++);
            } else {
                if (!m.verify(tok, drafts, 0.0f, 1.0f, 0, 0.0f, 0, counter, acc, next, err)) {
                    std::fprintf(stderr, "multi check (drafting): %s\n", err.c_str());
                    return 1;
                }
                counter += (uint64_t) acc + 1;
                off_alone += (long long) drafts.size();
                acc_alone += acc;
            }
            if (keep_rows && m.logits(lgw)) {
                const int T = (int) drafts.size() + 1;
                for (int t = 0; t < T && (size_t) (t + 1) * V <= lgw.size(); ++t) {
                    const std::pair<int64_t, int> key(p0 + t, t == 0 ? tok : drafts[(size_t) t - 1]);
                    if (rows_alone[(size_t) i].count(key)) continue;
                    RowInfo& ri = rows_alone[(size_t) i][key];
                    ri.lg.assign(lgw.begin() + (long) t * V, lgw.begin() + (long) (t + 1) * V);
                    ri.R = T, ri.seg = 0, ri.t = t, ri.T = T;
                }
            }
            ++rounds_alone;
            for (int j = 0; j < acc; ++j) out.push_back(drafts[(size_t) j]);
            out.push_back(next);
            tok = out.back();
        }
        ms_alone += now_ms() - t0;
    }
    std::vector<uint64_t> counters((size_t) N, 0);
    const auto restart = [&]() -> bool {   // every conversation at its prompt's end again, its first token
        for (int i = 0; i < N; ++i) {
            together[(size_t) i].clear();
            counters[(size_t) i] = 0;
            if (!prompt(i)) {
                std::fprintf(stderr, "multi check (drafting): %s\n", err.c_str());
                return false;
            }
            together[(size_t) i].push_back(m.sample(0.0f, 1.0f, 0, 0.0f, 0, 0));
        }
        return true;
    };
    if (!restart()) return 1;
    // (the serve loop's caps: MAYA_DS4_MULTI_DRAFT_MAX a conversation, what fits MAYA_DS4_MULTI_DRAFT_ROWS rows)
    const MultiDraft md = multi_draft_settings();
    const int kblock = md.max_drafts > 0 ? std::min(md.max_drafts, block) : block, rows_max = std::min(md.max_rows, 16);
    const int kmax = std::min(kblock, std::max(0, (rows_max - N) / N));
    // (two windows in flight: each half of the round's conversations is a window of its own, with its own cap - the
    //  serve loop's groups)
    const auto kmax_half = [&](int nh) { return std::min(kblock, std::max(0, (rows_max - nh) / nh)); };
    // the together phase drafts as the serve loop does - every conversation's block in one pass of the drafter
    // (Model::draft_multi) - and checks each conversation's drafts and confidences against the lone drafter's (the
    // same rings, a window of its own): a pass of several is 10-15 rows - past DS4_ASYNC_T (8), and not under
    // --parallel's async passes, it takes layer()'s host-routed path, other GPU kernels than the lone draft's window
    // path, so with the drafter's experts in VRAM it is not the lone drafter's bit for bit (on the async path, or with
    // the drafter's experts on the CPU lane, it is).  The tokens are the trunk's either way: a difference here costs
    // speed only
    long long fly_calls = 0, fly_diff = 0;   // (two windows in flight) a pass beside a window's stages: not as the same
                                             // pass (the same conversations) with no window in flight
    long long draft_calls = 0, draft_same = 0, draft_diff = 0;   // conversations' drafts in a pass: as alone, not
    std::string draft_first;                                     // the first that differed
    double ms_draft_pass = 0, ms_draft_lone = 0;
    // (two windows in flight) pass 0 runs the rounds with two windows in flight; pass 1 runs the same windows one at
    // a time from the same prompts (Model::verify_multi, the same rows in the same order): the streams must be the
    // same - what certifies the pipe's buffers and request rings.  The check runs with the expert cache held
    // (Model::cache_hold, main): the experts stay where the load put them, so pass 1 meets the placement pass 0 did -
    // with the cache moving experts between the passes the two would differ wherever an expert moved between the
    // GPU and the CPU lane (their kernels round differently), whatever the pipe did.  Only pass 0's counts are kept.
    std::vector<std::vector<int>> together_pipe;
    for (int pass = 0; pass < (pipe ? 2 : 1); ++pass) {
    if (pass == 1) {
        together_pipe = together;
        if (!restart()) return 1;
    }
    for (;;) {
        std::vector<strata::ds4::Model::MultiSpec> rs;
        std::vector<int> who, dseq, dtok;
        for (int i = 0; i < N; ++i) {
            if ((int) together[(size_t) i].size() >= n) continue;
            strata::ds4::Model::MultiSpec r;
            r.seq = i;
            r.tok = together[(size_t) i].back();
            r.counter = counters[(size_t) i];
            rs.push_back(std::move(r));
            who.push_back(i);
            dseq.push_back(i);
            dtok.push_back(together[(size_t) i].back());
        }
        if (rs.empty()) break;
        std::vector<std::vector<int>> dv;
        std::vector<std::vector<float>> cv;
        const double td = now_ms();
        if (!m.draft_multi(dseq.data(), dtok.data(), (int) dseq.size(), dv, cv, err)) {
            std::fprintf(stderr, "multi check (drafting): %s\n", err.c_str());
            return 1;
        }
        if (pass == 0) ms_draft_pass += now_ms() - td;
        for (size_t j = 0; j < rs.size(); ++j) {   // (the lone drafter's, for the comparison - pass 0)
            if (pass == 0) {
            m.seq_bind(who[j]);
            std::vector<int> ld;
            std::vector<float> lc;
            const double tl = now_ms();
            if (!m.draft(rs[j].tok, ld, lc, err, 0.0f, 1.0f, 0, 0.0f, 0, 0)) {
                std::fprintf(stderr, "multi check (drafting): %s\n", err.c_str());
                return 1;
            }
            ms_draft_lone += now_ms() - tl;
            ++draft_calls;
            const bool same = ld == dv[j] && lc == cv[j];
            if (same) ++draft_same;
            else {
                ++draft_diff;
                if (draft_first.empty()) {
                    size_t k = 0;
                    while (k < ld.size() && k < dv[j].size() && ld[k] == dv[j][k] && lc[k] == cv[j][k]) ++k;
                    char b[256];
                    std::snprintf(b, sizeof b, "conversation %d at position %lld, draft %zu: alone %d (%.4f), in a pass of %zu %d (%.4f)",
                                  who[j], (long long) m.seq_pos(who[j]), k + 1, k < ld.size() ? ld[k] : -1,
                                  k < lc.size() ? lc[k] : 0.0f, rs.size(), k < dv[j].size() ? dv[j][k] : -1,
                                  k < cv[j].size() ? cv[j][k] : 0.0f);
                    draft_first = b;
                }
            }
            }
            rs[j].drafts = std::move(dv[j]);
            const int nh = pipe && rs.size() >= 2 ? (j < (rs.size() + 1) / 2 ? (int) (rs.size() + 1) / 2
                                                                            : (int) rs.size() - (int) (rs.size() + 1) / 2)
                                                 : 0;
            cut_given(std::min(nh > 0 ? kmax_half(nh) : kmax, n - (int) together[(size_t) who[j]].size() - 1),
                      rs[j].drafts, cv[j]);
        }
        std::vector<int64_t> p0s(rs.size());
        for (size_t j = 0; j < rs.size(); ++j) p0s[j] = m.seq_pos(rs[j].seq);
        // a window's verdicts landed: each conversation's kept drafts and next token, its counter, the counts
        const auto landed = [&](const strata::ds4::Model::MultiSpec* r0, const int* w0, int cnt) {
            for (int j = 0; j < cnt; ++j) {
                const strata::ds4::Model::MultiSpec& r = r0[j];
                std::vector<int>& out = together[(size_t) w0[j]];
                for (int a = 0; a < r.accepted; ++a) out.push_back(r.drafts[(size_t) a]);
                out.push_back(r.next);
                counters[(size_t) w0[j]] += (uint64_t) r.accepted + 1;
                if (pass == 0) {
                    off_together += (long long) r.drafts.size();
                    acc_together += r.accepted;
                }
            }
        };
        const double t0 = now_ms();
        if (pipe && rs.size() >= 2 && pass == 0) {
            // (MAYA_DS4_MD_PIPE) the round as the serve loop runs it: the conversations in two windows (A, B), both
            // in flight; A taken, its next window drafted while B is in flight (draft_multi alone: the lone drafter's
            // comparison binds, which is refused then) and issued - B's second stage goes with that issue - then B
            // taken, then A's second window.  The tokens must be the same as the one window's
            const int nA = ((int) rs.size() + 1) / 2, nB = (int) rs.size() - nA;
            const auto fail = [&](const char* what) {
                std::fprintf(stderr, "multi check (drafting, two windows in flight, %s): %s\n", what, err.c_str());
                m.pipe_reset();
                return 1;
            };
            // B's drafts as a pass of B's conversations alone, before A is issued - the reference for the same pass
            // in flight below (the round's pass drafted every conversation: other rows, so another path and tile)
            std::vector<int> ds, dt;
            for (int j = nA; j < (int) rs.size(); ++j) {
                ds.push_back(rs[(size_t) j].seq);
                dt.push_back(rs[(size_t) j].tok);
            }
            std::vector<std::vector<int>> dvb_ref;
            std::vector<std::vector<float>> cvb_ref;
            if (!m.draft_multi(ds.data(), dt.data(), (int) ds.size(), dvb_ref, cvb_ref, err)) return fail("draft B");
            if (!m.verify_multi_issue(rs.data(), nA, 0, err)) return fail("issue A");
            {
                // B's drafts again, now that A's stages are in flight: the pass must give what the same pass gave with
                // no window in flight, bit for bit, and A's tokens must stay the one window's - the drafter's head
                // writes its own logits rows, not A's (which its head has written and its take has not sampled yet),
                // and the pass is enqueued only once the stage threads have enqueued A's stages (interleaved, they
                // shared scratch)
                std::vector<std::vector<int>> dvb;
                std::vector<std::vector<float>> cvb;
                if (!m.draft_multi(ds.data(), dt.data(), (int) ds.size(), dvb, cvb, err)) return fail("draft B in flight");
                for (int j = 0; j < nB; ++j) {
                    ++fly_calls;
                    if (dvb[(size_t) j] != dvb_ref[(size_t) j] || cvb[(size_t) j] != cvb_ref[(size_t) j]) ++fly_diff;
                }
            }
            if (!m.verify_multi_issue(rs.data() + nA, nB, 1, err)) return fail("issue B");
            if (!m.verify_multi_take(rs.data(), nA, 0, err)) return fail("take A");
            landed(rs.data(), who.data(), nA);
            std::vector<strata::ds4::Model::MultiSpec> rs2;
            std::vector<int> who2, dseq2, dtok2;
            for (int j = 0; j < nA; ++j) {
                const int i = who[(size_t) j];
                if ((int) together[(size_t) i].size() >= n) continue;
                strata::ds4::Model::MultiSpec r;
                r.seq = i;
                r.tok = together[(size_t) i].back();
                r.counter = counters[(size_t) i];
                rs2.push_back(std::move(r));
                who2.push_back(i);
                dseq2.push_back(i);
                dtok2.push_back(together[(size_t) i].back());
            }
            if (!rs2.empty()) {
                std::vector<std::vector<int>> dv2;
                std::vector<std::vector<float>> cv2;
                const double td2 = now_ms();
                if (!m.draft_multi(dseq2.data(), dtok2.data(), (int) dseq2.size(), dv2, cv2, err)) return fail("draft A'");
                ms_draft_pass += now_ms() - td2;
                for (size_t j = 0; j < rs2.size(); ++j) {
                    rs2[j].drafts = std::move(dv2[j]);
                    cut_given(std::min(kmax_half((int) rs2.size()), n - (int) together[(size_t) who2[j]].size() - 1),
                              rs2[j].drafts, cv2[j]);
                }
                if (!m.verify_multi_issue(rs2.data(), (int) rs2.size(), 0, err)) return fail("issue A'");
            }
            if (!m.verify_multi_take(rs.data() + nA, nB, 1, err)) return fail("take B");
            landed(rs.data() + nA, who.data() + nA, nB);
            if (!rs2.empty()) {
                if (!m.verify_multi_take(rs2.data(), (int) rs2.size(), 0, err)) return fail("take A'");
                landed(rs2.data(), who2.data(), (int) rs2.size());
                ++rounds_together;
            }
        } else if (pipe && rs.size() >= 2) {
            // (pass 1) the same windows one at a time: A, then A's next window drafted as above, B, A's next
            const int nA = ((int) rs.size() + 1) / 2, nB = (int) rs.size() - nA;
            const auto fail1 = [&](const char* what) {
                std::fprintf(stderr, "multi check (drafting, the same windows one at a time, %s): %s\n", what, err.c_str());
                return 1;
            };
            if (!m.verify_multi(rs.data(), nA, err)) return fail1("A");
            landed(rs.data(), who.data(), nA);
            std::vector<strata::ds4::Model::MultiSpec> rs2;
            std::vector<int> who2, dseq2, dtok2;
            for (int j = 0; j < nA; ++j) {
                const int i = who[(size_t) j];
                if ((int) together[(size_t) i].size() >= n) continue;
                strata::ds4::Model::MultiSpec r;
                r.seq = i;
                r.tok = together[(size_t) i].back();
                r.counter = counters[(size_t) i];
                rs2.push_back(std::move(r));
                who2.push_back(i);
                dseq2.push_back(i);
                dtok2.push_back(together[(size_t) i].back());
            }
            if (!rs2.empty()) {
                std::vector<std::vector<int>> dv2;
                std::vector<std::vector<float>> cv2;
                if (!m.draft_multi(dseq2.data(), dtok2.data(), (int) dseq2.size(), dv2, cv2, err)) return fail1("draft A'");
                for (size_t j = 0; j < rs2.size(); ++j) {
                    rs2[j].drafts = std::move(dv2[j]);
                    cut_given(std::min(kmax_half((int) rs2.size()), n - (int) together[(size_t) who2[j]].size() - 1),
                              rs2[j].drafts, cv2[j]);
                }
            }
            if (!m.verify_multi(rs.data() + nA, nB, err)) return fail1("B");
            landed(rs.data() + nA, who.data() + nA, nB);
            if (!rs2.empty()) {
                if (!m.verify_multi(rs2.data(), (int) rs2.size(), err)) return fail1("A'");
                landed(rs2.data(), who2.data(), (int) rs2.size());
            }
        } else if (!m.verify_multi(rs.data(), (int) rs.size(), err)) {
            std::fprintf(stderr, "multi check (drafting): %s\n", err.c_str());
            return 1;
        }
        if (pass == 0) {
            ms_together += now_ms() - t0;
            ++rounds_together;
        }
        if (keep_rows && !pipe && m.logits(lgw)) {
            int R = 0;
            for (const auto& r : rs) R += (int) r.drafts.size() + 1;
            int r0 = 0;
            for (size_t j = 0; j < rs.size(); ++j) {
                const int T = (int) rs[j].drafts.size() + 1;
                for (int t = 0; t < T && (size_t) (r0 + t + 1) * V <= lgw.size(); ++t) {
                    const std::pair<int64_t, int> key(p0s[j] + t, t == 0 ? rs[j].tok : rs[j].drafts[(size_t) t - 1]);
                    if (rows_together[(size_t) who[j]].count(key)) continue;
                    RowInfo& ri = rows_together[(size_t) who[j]][key];
                    ri.lg.assign(lgw.begin() + (long) (r0 + t) * V, lgw.begin() + (long) (r0 + t + 1) * V);
                    ri.R = R, ri.seg = (int) j, ri.t = t, ri.T = T;
                }
                r0 += T;
            }
        }
        if (!(pipe && rs.size() >= 2)) landed(rs.data(), who.data(), (int) rs.size());   // (the pipe landed its windows above)
    }
    }   // (the passes)
    // (two windows in flight) pass 1's streams against pass 0's; the comparisons below are pass 0's (the pipe's)
    bool pipe_same = true;
    std::string pipe_first;
    if (pipe) {
        for (int i = 0; i < N; ++i) {
            const std::vector<int>& a = together[(size_t) i];          // one at a time
            const std::vector<int>& b = together_pipe[(size_t) i];     // two in flight
            size_t d = 0;
            while (d < a.size() && d < b.size() && a[d] == b[d]) ++d;
            if (d == a.size() && a.size() == b.size()) continue;
            pipe_same = false;
            if (pipe_first.empty())
                pipe_first = "conversation " + std::to_string(i) + " from token " + std::to_string(d) + " on (one at a time " +
                             std::to_string(d < a.size() ? a[d] : -1) + ", in flight " + std::to_string(d < b.size() ? b[d] : -1) + ")";
        }
        together = together_pipe;
    }
    if (keep_rows)
        for (int i = 0; i < N; ++i) {
            long long matched = 0, same = 0;
            bool first = true;
            for (const auto& kv : rows_alone[(size_t) i]) {
                const auto it = rows_together[(size_t) i].find(kv.first);
                if (it == rows_together[(size_t) i].end()) continue;
                ++matched;
                const RowInfo& a = kv.second;
                const RowInfo& b = it->second;
                if (std::memcmp(a.lg.data(), b.lg.data(), (size_t) V * 4) == 0) {
                    ++same;
                    continue;
                }
                if (!first) continue;
                first = false;
                double max_d = 0;
                int argmax_a = 0, argmax_b = 0;
                for (int v = 0; v < V; ++v) {
                    max_d = std::max(max_d, (double) std::fabs(a.lg[(size_t) v] - b.lg[(size_t) v]));
                    if (a.lg[(size_t) v] > a.lg[(size_t) argmax_a]) argmax_a = v;
                    if (b.lg[(size_t) v] > b.lg[(size_t) argmax_b]) argmax_b = v;
                }
                std::printf("multi check (drafting) rows: conversation %d: first row that differs: position %lld token %d: "
                            "alone a window of %d (row %d), together a window of %d rows, segment %d row %d of %d; "
                            "max |d| %.3g, argmax %d vs %d\n",
                            i, (long long) kv.first.first, kv.first.second, a.R, a.t, b.R, b.seg, b.t, b.T, max_d,
                            argmax_a, argmax_b);
            }
            std::printf("multi check (drafting) rows: conversation %d: %lld rows both ways, %lld bit for bit\n", i,
                        matched, same);
            int shown = 0;   // the first rows each way, in position order: which rows each phase ran, and which agree
            for (const auto& kv : rows_alone[(size_t) i]) {
                if (shown++ >= 8) break;
                const auto it = rows_together[(size_t) i].find(kv.first);
                std::printf("  alone    %lld:%d window %d row %d  | together %s\n", (long long) kv.first.first,
                            kv.first.second, kv.second.R, kv.second.t,
                            it == rows_together[(size_t) i].end()
                                ? "(no such row)"
                                : std::memcmp(it->second.lg.data(), kv.second.lg.data(), (size_t) V * 4) == 0 ? "same" : "DIFFERS");
            }
            shown = 0;
            for (const auto& kv : rows_together[(size_t) i]) {
                if (shown++ >= 8) break;
                std::printf("  together %lld:%d window %d segment %d row %d of %d\n", (long long) kv.first.first,
                            kv.first.second, kv.second.R, kv.second.seg, kv.second.t, kv.second.T);
            }
        }
    bool tokens_same = true;
    long long toks = 0;
    for (int i = 0; i < N; ++i) {
        std::vector<int>& a = alone[(size_t) i];
        std::vector<int>& b = together[(size_t) i];
        a.resize((size_t) std::min<size_t>(a.size(), (size_t) n));
        b.resize((size_t) std::min<size_t>(b.size(), (size_t) n));
        size_t d = 0;
        while (d < a.size() && d < b.size() && a[d] == b[d]) ++d;
        const bool same = d == a.size() && a.size() == b.size();
        tokens_same = tokens_same && same;
        toks += (long long) a.size();
        std::printf("multi check (drafting): conversation %d (%zu prompt tokens): %zu tokens %s", i,
                    prompts[(size_t) i].size(), a.size(), same ? "the same alone and together\n" : "");
        if (!same) std::printf("differ from token %zu on\n", d);
    }
    if (pipe) {
        std::printf("multi check (drafting) two windows in flight vs the same windows one at a time: the same tokens: %s%s%s\n",
                    pipe_same ? "yes" : "NO", pipe_first.empty() ? "" : "; ", pipe_first.c_str());
        std::printf("multi check (drafting) drafter in flight: %lld conversations' drafts in a pass beside a window's "
                    "stages: %lld as the same pass with no window in flight, %lld not\n",
                    fly_calls, fly_calls - fly_diff, fly_diff);
    }
    std::printf("multi check (drafting) drafter: %lld conversations' drafts in passes of up to %d: %lld as the lone drafter's "
                "(drafts and confidences), %lld not%s%s; the passes %.1f ms, the lone drafts %.1f ms\n",
                draft_calls, m.draft_pass(), draft_same, draft_diff, draft_first.empty() ? "" : "; the first: ",
                draft_first.c_str(), ms_draft_pass, ms_draft_lone);
    std::printf("multi check (drafting): %d conversations x %d tokens: tokens the same: %s; alone %lld rounds, %lld of "
                "%lld drafts kept, %.2f tok/s; together %lld rounds (%d drafts each at most), %lld of %lld kept, "
                "%.2f tok/s in all (x%.2f)\n",
                N, n, tokens_same ? "yes" : "NO", rounds_alone, acc_alone, off_alone,
                ms_alone > 0 ? 1000.0 * (double) (toks - N) / ms_alone : 0.0, rounds_together, kmax, acc_together,
                off_together, ms_together > 0 ? 1000.0 * (double) (toks - N) / ms_together : 0.0,
                ms_together > 0 ? ms_alone / ms_together : 0.0);
    // (experts on the CPU lane, held where the load put them for the check: a row's CPU-lane experts are summed per
    //  token, in expert order, whatever rows share its request - every comparison is expected to hold, as with every
    //  expert in VRAM)
    const int lane = m.lane_experts();
    if (lane > 0)
        std::printf("multi check (drafting): %d of the trunk's experts are on the CPU lane, held where the load put "
                    "them: every comparison counts; two windows in flight vs the same windows one at a time: %s\n",
                    lane, pipe ? (pipe_same ? "yes" : "NO") : pipe_refused ? "NOT CHECKED (the pipe is off)" : "(not asked)");
    if (pipe_refused) return 2;
    return tokens_same && pipe_same ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    strata::platform::env_aliases();   // (MAYA_* settings under their older names too)
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
    float temp = 0.0f;   // the CLI's sampling (0: greedy)
    unsigned long long cli_seed = 1;
    bool serve_mode = false, multi_check = false;
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
    if (multi_check && (o.seqs < 2 || ids_s.empty())) {
        std::fprintf(stderr, "--multi-check needs --parallel 2..8 and --ids\n");
        return 2;
    }
    if (o.pack.empty() || (!serve_mode && ids_s.empty() && enc_rgb.empty() && ppl.empty())) {
        usage();
        return 2;
    }
    if (o.profile.empty()) o.profile = o.pack + "/expert-profile.bin";   // "none": no profile
    if (o.profile == "none") o.profile.clear();
    o.vision_only = !enc_rgb.empty();
    strata::ds4::Model m;
    std::string err;
    o.lookup = suffix_draft > 0;   // (its verify windows' buffers, with no drafter too)
    if (!m.load(o, err)) {
        std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
        return 1;
    }
    if (!o.vision_only) std::fprintf(stderr, "%s", m.describe().c_str());
    std::unique_ptr<Lookup> lk;   // (prompt lookup beside DSpark, or alone: one policy over every request of this run)
    if (suffix_draft > 0)
        lk = std::make_unique<Lookup>(suffix_draft, m.verify_max(), (size_t) m.max_context() + 4096);
    if (serve_mode) return serve(m, eos, spec, o.dspark_conf, o.pack, lk.get());
    if (multi_check) {   // (the plain rows' check; then, with the drafter loaded and on, every conversation drafting)
        // The experts where the load put them for the whole check (Model::cache_hold: the expert cache neither switches
        // nor queues a swap).  With experts on the CPU lane the cache otherwise moves them between VRAM and the lane as
        // the rows use them, and the GPU's kernels and the CPU lane round an expert differently: two runs of the same
        // rows at different times - alone, then together; the pipe, then the same windows one at a time - differ
        // wherever an expert moved in between (upstream's "with the experts where the load put them").  Held, a row's
        // sums depend on its own rows only, on any placement
        m.cache_hold(true);
        const int rc = check_multi(m, parse_ids(ids_s), std::max(2, n));
        const int lane = m.lane_experts();
        if (rc != 0 && lane > 0)
            std::printf("multi check: %d of the trunk's experts are on the CPU lane, held where the load put them: the "
                        "plain comparison was expected to hold - the drafting phase runs anyway\n", lane);
        if (!spec || m.draft_block() <= 0) return rc;
        if (rc != 0 && lane == 0) return rc;
        const int rs = check_multi_spec(m, parse_ids(ids_s), std::max(2, n), o.dspark_conf);
        return rc != 0 ? rc : rs;
    }
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
        if (lk) lk->start(m.tokens());
        const long long lk_w0 = lk ? lk->windows : 0, lk_o0 = lk ? lk->offered : 0, lk_a0 = lk ? lk->accepted : 0;
        for (;;) {
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
