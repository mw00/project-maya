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
#include "ds4_model.hpp"
#include "strata/platform/env_alias.hpp"

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
#include <iostream>
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
                 "           [--vision on|off] [--mmap-experts on|off|auto]\n"
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
                    continue;
                }
                if (l == "QUIT") stop.store(true);
                q.push_back(l);
                cv.notify_all();
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            stop.store(true);
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

// One round of decoding after `tok` (the token at pos(), not fed yet): with the drafter, its drafts while their
// confidence holds (at most `room` - 1 of them), verified together; else a plain step.  `out` gets the tokens that follow
// tok (the accepted drafts, then the trunk's next), the last of them the next round's tok.
bool round(strata::ds4::Model& m, int tok, int room, bool spec, float conf_min, float temperature, float top_p, int top_k,
           float min_p, unsigned long long seed, unsigned long long& counter, std::vector<int>& out, int& offered,
           int& accepted, std::string& err, float cum_min = -1.0f) {
    out.clear();
    offered = accepted = 0;
    if (spec && room > 1 && m.draft_block() > 0) {
        std::vector<int> drafts;
        std::vector<float> conf;
        if (!m.draft(tok, drafts, conf, err, temperature, top_p, top_k, min_p, seed, counter)) return false;
        int keep = 0;
        if (cum_min < 0) cum_min = dspark_cum_default();
        float cum = 1.0f;
        while (keep < (int) drafts.size() && keep < room - 1 && conf[(size_t) keep] >= conf_min &&
               (cum *= conf[(size_t) keep]) >= cum_min)
            ++keep;
        if (keep > 0) {
            drafts.resize((size_t) keep);
            int acc = 0, next = -1;
            if (!m.verify(tok, drafts, temperature, top_p, top_k, min_p, seed, counter, acc, next, err)) return false;
            counter += (unsigned long long) acc + 1;
            for (int i = 0; i < acc; ++i) out.push_back(drafts[(size_t) i]);
            out.push_back(next);
            offered = keep;
            accepted = acc;
            return true;
        }
    }
    if (!m.step(tok, true, err)) return false;
    out.push_back(m.sample(temperature, top_p, top_k, min_p, seed, counter++));
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
        else if (k == "penalty_last_n") r.pen_n = std::atoi(v.c_str());
        else if (k == "penalty_repeat") r.pen_repeat = std::strtof(v.c_str(), nullptr);
        else if (k == "penalty_freq") r.pen_freq = std::strtof(v.c_str(), nullptr);
        else if (k == "penalty_present") r.pen_present = std::strtof(v.c_str(), nullptr);
        else if (k == "think_budget") r.think_budget = std::atoll(v.c_str());
        else if (k == "think_end") r.think_end = std::atoi(v.c_str());
    }
    return !r.ids.empty() && (!geni || !r.file.empty());
}

int serve(strata::ds4::Model& m, const std::string& eos_list, bool spec, float conf_min, const std::string& pack) {
    std::vector<int> eos = parse_ids(eos_list);
    for (const std::string& l : m.info_lines()) std::printf("%s\n", l.c_str());
    std::printf("READY %lld stop\n", (long long) m.max_context());
    std::fflush(stdout);
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
    while (in.next(line)) {
        if (line == "QUIT") break;
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
        const double t0 = now_ms();
        const strata::ds4::Stats s0 = m.stats();   // (the request's own CPU lane share)
        std::vector<Pic> pics;
        {
            std::string e;
            if (geni && img_tok < 0) e = "this engine has no vision tower, so it cannot read pictures";
            else if (geni && read_pics(r.file, pics, e))
                for (const Pic& q : pics)
                    if (q.start < 0 || (size_t) q.start >= r.ids.size() || r.ids[(size_t) q.start] != img_tok)
                        e = "a picture's span does not start at an image token of the prompt";
            if (!e.empty()) {
                std::printf("ERR %s\n", e.c_str());
                std::printf("DONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", r.ids.size());
                std::fflush(stdout);
                continue;
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
            const int64_t limit = (int64_t) std::min(c, r.ids.size() - 1);
            const int64_t rs = m.restorable(limit);
            bool whole = false;
            const size_t back = slot_take(r.ids, pics, rs > 0 ? (size_t) rs : 0, whole);
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
            std::printf("ERR the prompt (%zu tokens) and max_new %lld are more than the context %lld\n", r.ids.size(),
                        r.max_new, (long long) m.max_context());
            std::fflush(stdout);
            continue;
        }
        std::string err;
        bool ok = true;
        // the pictures whose spans the prompt reads (the rest the state holds): encoded (or the cache's), checked
        if (!pics.empty()) {
            std::vector<strata::ds4::Image> imgs;
            const double te = now_ms();
            int encoded = 0;
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
            if (ok) {
                m.set_images(std::move(imgs));
                std::fprintf(stderr, "maya ds4 serve: %zu pictures, %d read (%.0f ms)\n", pics.size(), encoded,
                             now_ms() - te);
            }
            for (Pic& q : pics) q.rgb = {};   // (only the start, hash and span stay)
        }
        // the prompt in windows (PP <read> <total> <ms> <tok/s> <reuse> every 64 tokens), a checkpoint before its last
        // token and one after it: the next turn re-renders this one, and shares its start up to one of them
        const size_t total = r.ids.size();
        size_t done = from, next_pp = from + 64;
        m.set_progress([&](int64_t d) {   // (a run of chunks: each one through every layer)
            const double ms = now_ms() - t0;
            std::printf("PP %zu %zu %.1f %.1f %zu\n", done + (size_t) d, r.ids.size(), ms,
                        ms > 0 ? 1000.0 * (double) (done + (size_t) d - from) / ms : 0.0, from);
            std::fflush(stdout);
        });
        // (DS4_LAST_WINDOW=1: the last token in a window of its own after the checkpoint, as before - for A/B runs)
        static const bool last_window = std::getenv("DS4_LAST_WINDOW") != nullptr;
        // a STOP while the prompt is read ends it there (the engine's run of chunks at a layer or chunk, windows between
        // feeds) and goes back to the state before this prompt - kept here first, in checkpoint 1, which the prompt's
        // own end overwrites as before - so the next request still shares the conversation up to it
        if (from > 0 && from < total) m.checkpoint(1);
        m.set_cancel([&in] { return in.stop.load(); });
        bool stopped = false;
        while (ok && done < total) {
            if (in.stop.load()) {
                stopped = true;
                break;
            }
            const size_t n = std::min<size_t>((size_t) m.feed_unit(), total - done);
            // (the prompt's end: the last token in the same run, the checkpoint before it kept on the way)
            const bool fin = done + n == total;
            if (fin && last_window) {
                ok = (n < 2 || m.feed(r.ids.data() + done, (int) n - 1, 0, err));
                if (ok) m.checkpoint(1);
                ok = ok && m.feed(r.ids.data() + done + n - 1, 1, 1, err);
            } else
                ok = fin ? m.feed_mark(r.ids.data() + done, (int) n, 1, 1, err)
                         : m.feed(r.ids.data() + done, (int) n, 0, err);
            done += n;
            if (ok && (done >= next_pp || done == total)) {
                const double ms = now_ms() - t0;
                std::printf("PP %zu %zu %.1f %.1f %zu\n", done, r.ids.size(), ms,
                            ms > 0 ? 1000.0 * (double) (done - from) / ms : 0.0, from);
                std::fflush(stdout);
                next_pp = done + 64;
            }
        }
        m.set_cancel(nullptr);
        m.set_images({});
        if (stopped || (!ok && m.cancelled())) {
            const bool back = from > 0 && m.restore((int64_t) from) && (size_t) m.pos() == from;
            if (!back) {
                m.reset();
                held.clear();
            }
            std::fprintf(stderr, "maya ds4 serve: stopped while reading the prompt of %zu tokens, after %.0f ms%s\n",
                         total, now_ms() - t0, back ? "; the conversation before it is kept" : "");
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0\n", r.ids.size(), now_ms() - t0, back ? from : (size_t) 0);
            std::fflush(stdout);
            continue;
        }
        if (!ok) {
            std::fprintf(stderr, "ds4: %s\n", err.c_str());
            std::printf("ERR %s\n", err.c_str());
            std::printf("DONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", r.ids.size());
            std::fflush(stdout);
            m.reset();
            held.clear();
            continue;
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
        // STAT for the Monitor every 8 tokens and at the end, over this answer so far: the GLM engine's keys (speed, the
        // VRAM hit, promotions a token) and DeepSeek's own - experts a token on the CPU lane, MB a token the storage
        // read (Linux), the n-gram memory's (Engram) read time a token, the drafts' acceptance
        const strata::ds4::Stats s_dec = m.stats();
        const long long io_dec = io_read_bytes();
        auto stat_line = [&]() {
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
            std::printf("%s\n", out.c_str());
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
            ++produced;
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
                           off, acc, err, r.cum)) {
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
        prof_line(s_dec, s);   // DS4_PROF: this request's decode (the CLI's line, a request at a time)
        m.save_profile();
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
    int n = 0, window = 16;
    bool spec = true;
    float temp = 0.0f;   // the CLI's sampling (0: greedy)
    unsigned long long cli_seed = 1;
    bool serve_mode = false;
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
        else if (a == "--serve") serve_mode = true;
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
    if (o.pack.empty() || (!serve_mode && ids_s.empty() && enc_rgb.empty() && ppl.empty())) {
        usage();
        return 2;
    }
    if (o.profile.empty()) o.profile = o.pack + "/expert-profile.bin";   // "none": no profile
    if (o.profile == "none") o.profile.clear();
    o.vision_only = !enc_rgb.empty();
    strata::ds4::Model m;
    std::string err;
    if (!m.load(o, err)) {
        std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
        return 1;
    }
    if (!o.vision_only) std::fprintf(stderr, "%s", m.describe().c_str());
    if (serve_mode) return serve(m, eos, spec, o.dspark_conf, o.pack);
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
        for (;;) {
            out.push_back(tok);
            std::printf("%d,", tok);
            std::fflush(stdout);
            if (tok == 1 || (int) out.size() >= n) break;
            if (got.empty()) {
                int off = 0, acc = 0;
                const bool try_spec = spec && !cool;
                if (!round(m, tok, n - (int) out.size(), try_spec, o.dspark_conf, temp, 1.0f, temp > 0 ? 64 : 1, 0.0f,
                           cli_seed, counter, got, off, acc, err)) {
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
