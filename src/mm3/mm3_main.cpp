// src/mm3/mm3_main.cpp - maya-mm3: MiniMax-M3.1-Flash (EXL3) on NVIDIA GPUs.
//
//   maya-mm3 --pack <model>/maya-mm3 --serve [options]     Maya's engine protocol on stdin / stdout
//
// --serve speaks what serve/server.py's StrataEngine expects: INFO lines, then `READY <context> stop`; per request
// `GEN <max_new> [key=value ...] <id,id,...>` answered by `PP <read> <total> <ms> <tok/s> <reused>` while the prompt
// is read, `T <id>` lines, `STAT ...` every 8 tokens and `DONE <generated> <prompt> <prompt ms> <decode ms>
// <stop|length|cancel> 0 0 <reused> <hits> <lookups>` (the decode's routed experts in VRAM / all); `STOP` ends a
// request early, `QUIT` the engine.  A prompt that shares a start with the state (the last conversation) reads only
// the rest.  Keys: temperature, top_p, top_k, min_p, seed, penalty_last_n / _repeat / _freq / _present,
// think_budget + think_end (a reasoning block still open after that many tokens is closed with think_end).
// With --vision on, `GENI <max_new> [keys] <file> <ids>` carries the prompt's pictures (the file: see read_pics; each
// picture's image tokens take its rows); the state is shared only where the same picture sits at the same place.
//
//   maya-mm3 --pack <model>/maya-mm3 --encode-image FILE.rgb --size WxH --dump-image OUT [--gpus N]
//
// The vision tower alone (no experts loaded): a picture's rows, F32 [tokens][hidden] (tools/mm3/vision_ref.py's).
//
//   maya-mm3 --pack <model>/maya-mm3 --ids @prompt.txt [--gen N] [--dump-logits out.bin --last N]
//            [--gpus 0,1,...] [--max-context N] [--chunk N] [--staging N] [--resident-cap N] [--reserve-gb X]
//            [--profile FILE] [--profile-out FILE] [--no-pin-pool] [--no-fast-decode]
//
// --ids: comma- or whitespace-separated token ids, or @FILE holding them.  --gen: greedy tokens after the prompt
// (their ids printed).  --dump-logits: the logits of the prompt's last --last positions (default 1), as
//   "MM3L" int32 rows, int32 vocab, int32 n_ids, int32 ids[n_ids], int32 pos[rows], float logits[rows][vocab]
// (tools/mm3/compare.py reads it beside tools/mm3/ref.py's dump).
#include <algorithm>
#include <cmath>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <fstream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include "mm3_cands.hpp"
#include "mm3_cpu.hpp"
#include "mm3_env.hpp"
#include "mm3_kernels.cuh"
#include "mm3_lane_plan.hpp"
#include "mm3_model.hpp"
#include "mm3_os.hpp"
#include "mm3_pack.hpp"
#include "mm3_sample.hpp"
#include "mm3_slots.hpp"
#include "mm3_vision.hpp"
#if defined(_WIN32)
#include <timeapi.h>
#pragma comment(lib, "winmm.lib")
#endif

namespace {

using namespace mm3::sample;

std::vector<int> parse_ids(const std::string& a) {
    std::string src = a;
    if (!a.empty() && a[0] == '@') {
        std::ifstream f(a.substr(1));
        std::stringstream ss;
        ss << f.rdbuf();
        src = ss.str();
    }
    for (char& c : src)
        if (c == ',') c = ' ';
    std::vector<int> v;
    std::istringstream is(src);
    for (long x; is >> x;) v.push_back((int) x);
    return v;
}

std::vector<int> parse_list(const std::string& a) {
    std::vector<int> v;
    std::istringstream is(a);
    for (std::string x; std::getline(is, x, ',');)
        if (!x.empty()) v.push_back(std::atoi(x.c_str()));
    return v;
}

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// GENI's file (serve/mm3_vision.py): int32 'MMI1', int32 count, then per picture int64 start (the position of its
// first image token), int32 width, int32 height, height x width x 3 RGB bytes
struct Pic {
    long long start = 0;
    int width = 0, height = 0;
    std::vector<uint8_t> rgb;
    uint64_t hash = 0;   // (FNV-1a over its size and pixels: the state's pictures told apart)
};

bool read_pics(const std::string& path, std::vector<Pic>& v, std::string& err) {
    v.clear();
    std::ifstream f(path, std::ios::binary);
    char magic[4];
    int32_t n = 0;
    if (!f.read(magic, 4) || std::memcmp(magic, "MMI1", 4) != 0 || !f.read((char*) &n, 4) || n < 0 || n > 4096) {
        err = path + ": not a pictures file (MMI1)";
        return false;
    }
    for (int i = 0; i < n; ++i) {
        Pic q;
        int64_t st = 0;
        int32_t wh[2] = {0, 0};
        if (!f.read((char*) &st, 8) || !f.read((char*) wh, 8) || wh[0] <= 0 || wh[1] <= 0 ||
            (long long) wh[0] * wh[1] > (1LL << 26)) {
            err = path + ": picture " + std::to_string(i) + " is cut short";
            return false;
        }
        q.start = st;
        q.width = wh[0];
        q.height = wh[1];
        q.rgb.resize((size_t) wh[0] * wh[1] * 3);
        if (!f.read((char*) q.rgb.data(), (std::streamsize) q.rgb.size())) {
            err = path + ": picture " + std::to_string(i) + " is cut short";
            return false;
        }
        uint64_t h = 1469598103934665603ull;
        auto mix = [&](const uint8_t* b, size_t k) {
            for (size_t j = 0; j < k; ++j) h = (h ^ b[j]) * 1099511628211ull;
        };
        mix((const uint8_t*) wh, 8);
        mix(q.rgb.data(), q.rgb.size());
        q.hash = h;
        v.push_back(std::move(q));
    }
    return true;
}

// the pictures' spans in ids: ]<]start of image[>[, its image tokens, ]<]end of image[>[ - and no image token outside
// one; their rows for those that end at or after position `from` (the others' positions are the state's already)
bool picture_rows(mm3::Model& m, const std::vector<Pic>& pics, const std::vector<int>& ids, size_t from,
                  std::vector<mm3::Model::Image>& out, std::string& err) {
    out.clear();
    const mm3::VisionGeo* vg = m.vision();
    if (!vg) {
        err = "this engine has no vision tower (start it with --vision on), so it cannot read pictures";
        return false;
    }
    std::vector<char> covered(ids.size(), 0);
    for (size_t i = 0; i < pics.size(); ++i) {
        const Pic& q = pics[i];
        const int f = vg->factor();
        const long long n = q.width % f || q.height % f ? 0 : (long long) (q.height / f) * (q.width / f);
        if (n == 0 || (long long) q.width * q.height > vg->max_pixels) {
            err = "picture " + std::to_string(i) + ": " + std::to_string(q.width) + " x " + std::to_string(q.height) +
                  " pixels (sides multiples of " + std::to_string(f) + ", at most " + std::to_string(vg->max_pixels) +
                  " pixels)";
            return false;
        }
        if (q.start < 1 || q.start + n >= (long long) ids.size() || ids[(size_t) q.start - 1] != vg->image_start ||
            ids[(size_t) (q.start + n)] != vg->image_end) {
            err = "picture " + std::to_string(i) + "'s span (" + std::to_string(n) + " image tokens at " +
                  std::to_string(q.start) + ") is not between the prompt's start and end of image tokens";
            return false;
        }
        for (long long k = q.start; k < q.start + n; ++k) {
            if (ids[(size_t) k] != vg->image_token || covered[(size_t) k]) {
                err = "picture " + std::to_string(i) + "'s span is not a run of image tokens of its own";
                return false;
            }
            covered[(size_t) k] = 1;
        }
        if (q.start + n > (long long) from) {
            mm3::Model::Image im;
            im.start = (int) q.start;
            if (!m.encode_image(q.rgb.data(), q.width, q.height, im.rows, err)) return false;
            out.push_back(std::move(im));
        }
    }
    for (size_t k = 0; k < ids.size(); ++k)
        if (ids[k] == vg->image_token && !covered[k]) {
            err = "an image token at " + std::to_string(k) + " without its picture";
            return false;
        }
    return true;
}

// stdin's lines on a thread of their own: STOP is seen while a request runs
struct Lines {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    bool eof = false;
    std::atomic<bool> stop{false};
    std::atomic<bool> wrap{false};   // (WRAP: the server is stopping - a request still thinking closes it and answers)
    std::vector<int> stops;          // (--parallel: STOP <rid> - that request alone; under mu)
    bool stopped(int rid) {          // (the request's STOP <rid> taken, if one came)
        std::lock_guard<std::mutex> lk(mu);
        const auto it = std::find(stops.begin(), stops.end(), rid);
        if (it == stops.end()) return false;
        stops.erase(it);
        return true;
    }
    void forget(int rid) {
        std::lock_guard<std::mutex> lk(mu);
        stops.erase(std::remove(stops.begin(), stops.end(), rid), stops.end());
    }
    bool try_next(std::string& out) {   // (a line if one is waiting, without waiting)
        std::lock_guard<std::mutex> lk(mu);
        if (q.empty()) return false;
        out = std::move(q.front());
        q.pop_front();
        return true;
    }
    bool ended() {
        std::lock_guard<std::mutex> lk(mu);
        return eof && q.empty();
    }
    void start() {
        std::thread([this] {
            std::string buf;
            char chunk[4096];
            for (;;) {
                const long n = mm3::os::read_stdin(chunk, sizeof chunk);
                if (n <= 0) break;
                buf.append(chunk, (size_t) n);
                for (size_t nl; (nl = buf.find('\n')) != std::string::npos;) {
                    std::string l = buf.substr(0, nl);
                    buf.erase(0, nl + 1);
                    if (!l.empty() && l.back() == '\r') l.pop_back();
                    if (l == "STOP") {
                        stop.store(true);
                        continue;
                    }
                    if (l == "WRAP") {
                        wrap.store(true);
                        continue;
                    }
                    if (l.rfind("STOP ", 0) == 0) {   // (--parallel: one request's)
                        const int rid = std::atoi(l.c_str() + 5);
                        std::lock_guard<std::mutex> lk(mu);
                        if (stops.size() >= 256) stops.erase(stops.begin());   // (stale ones: requests already ended)
                        if (rid >= 0) stops.push_back(rid);
                        continue;
                    }
                    std::lock_guard<std::mutex> lk(mu);
                    q.push_back(std::move(l));
                    cv.notify_one();
                }
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            cv.notify_one();
        }).detach();
    }
    bool next(std::string& out) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !q.empty() || eof; });
        if (q.empty()) return false;
        out = std::move(q.front());
        q.pop_front();
        return true;
    }
};

bool parse_gen(const std::string& line, Req& r) {
    const bool geni = line.rfind("GENI ", 0) == 0;
    std::istringstream is(line.substr(geni ? 5 : 4));
    if (!(is >> r.max_new)) return false;
    for (std::string tok; is >> tok;) {
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) {
            if (geni && r.file.empty()) {
                r.file = tok;
                continue;
            }
            r.ids = parse_ids(tok);
            break;
        }
        const std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
        if (k == "temperature") r.temperature = std::strtof(v.c_str(), nullptr);
        else if (k == "top_p") r.top_p = std::strtof(v.c_str(), nullptr);
        else if (k == "top_k") r.top_k = std::atoi(v.c_str());
        else if (k == "min_p") r.min_p = std::strtof(v.c_str(), nullptr);
        else if (k == "seed") r.seed = std::strtoull(v.c_str(), nullptr, 10);
        else if (k == "penalty_last_n") r.pen_n = std::atoi(v.c_str());
        else if (k == "penalty_repeat") r.pen_repeat = std::strtof(v.c_str(), nullptr);
        else if (k == "penalty_freq") r.pen_freq = std::strtof(v.c_str(), nullptr);
        else if (k == "penalty_present") r.pen_present = std::strtof(v.c_str(), nullptr);
        else if (k == "think_budget") r.think_budget = std::atoll(v.c_str());
        else if (k == "think_end") r.think_end = std::atoi(v.c_str());
        else if (k == "spec") r.spec = std::atoi(v.c_str()) != 0;
        else if (k == "dspark_conf" || k == "spec_min_p") r.conf = std::strtof(v.c_str(), nullptr);
        else if (k == "rid") r.rid = std::atoi(v.c_str());
        // (keys this engine does not use are taken and ignored)
    }
    return !r.ids.empty();
}

// --serve's first lines: the load's notes, the Monitor's GPU map, the INFO facts (READY follows)
void serve_header(mm3::Model& m, bool spec_on, const std::string& kv_fmt) {
    for (const std::string& l : m.info_lines()) std::printf("NOTE %s\n", l.c_str());
    for (const std::string& l : m.vram_lines()) std::printf("%s\n", l.c_str());   // (the Monitor's GPU map)
    const auto& st0 = m.stats();
    std::printf("INFO engine=mm3 engine_kind=mm3 model=MiniMax-M3.1-Flash experts_vram=%ld experts=%ld "
                "expert_slots=%ld ram_slots=%ld dspark=%d kv=%s cpu_threads=%d cpu_experts=%ld ctx=%d", st0.resident,
                st0.experts, st0.resident, st0.pooled, m.draft_block() > 0 && spec_on ? 1 : 0, kv_fmt.c_str(),
                st0.cpu_threads, st0.experts - st0.resident, m.max_ctx());
    if (st0.kv_ring > 0) std::printf(" kv_resident=%d", st0.kv_ring);   // (KV streaming: a layer's positions in VRAM)
    std::printf("\n");
    // the same facts in the engine log (stderr) - the lines above reach the server only - for --report and a look at
    // where the model went, as the run without --serve prints them
    std::fprintf(stderr, "maya-mm3: loaded in %.1f s, %ld of %ld experts in VRAM, %.1f GB pinned pool\n", st0.load_s,
                 st0.resident, st0.experts, st0.pinned_gb);
    for (const std::string& l : m.info_lines()) std::fprintf(stderr, "maya-mm3: %s\n", l.c_str());
    for (const std::string& l : m.vram_lines()) std::fprintf(stderr, "maya-mm3: %s\n", l.c_str());
    if (!m.vision()) std::printf("INFO vision=off\n");   // (no tower: the server answers text only)
    if (const mm3::VisionGeo* vg = m.vision())   // (serve/mm3_vision.py's Geometry.from_info)
        std::printf("INFO vision=mm3 image_token=%d image_start=%d image_end=%d vision_patch=%d vision_merge=%d "
                    "vision_max_pixels=%d vision_min_pixels=%d\n", vg->image_token, vg->image_start, vg->image_end,
                    vg->patch, vg->merge, vg->max_pixels, vg->min_pixels);
}

int serve(mm3::Model& m, const std::vector<int>& eos, int chunk, bool spec_on, float conf_floor, float draft_cost,
          const std::string& kv_fmt, const std::string& profile_out) {
    serve_header(m, spec_on, kv_fmt);
    std::printf("READY %d stop\n", m.max_ctx());
    std::fflush(stdout);
    std::fprintf(stderr, "maya mm3 serve: GEN <max_new> [keys] <ids>; GENI <max_new> [keys] <file> <ids>; STOP, QUIT\n");
#if defined(SIGPIPE)
    std::signal(SIGPIPE, SIG_IGN);
#endif
    Lines in;
    in.start();
    std::vector<int> state;   // the tokens in the cache, positions 0 ..
    struct Held {
        long long start;
        uint64_t hash;
    };
    std::vector<Held> held;   // the pictures in the cache (their first image token's position, their pixels' hash)
    std::vector<float> lg((size_t) m.vocab());
    unsigned long long counter = 0;
    std::string line, err;
    while (in.next(line)) {
        if (line == "QUIT") break;
        if (line.rfind("GEN ", 0) != 0 && line.rfind("GENI ", 0) != 0) {
            std::printf("ERR unsupported command '%s' (GEN, GENI, STOP and QUIT)\n",
                        line.substr(0, line.find(' ')).c_str());
            std::fflush(stdout);
            continue;
        }
        in.stop.store(false);
        Req r;
        if (!parse_gen(line, r) || (line.rfind("GENI ", 0) == 0 && r.file.empty())) {
            std::printf("ERR expected: GEN <max_new> [keys] <id,id,...> (GENI: the pictures' file before the ids)\n");
            std::printf("DONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", r.ids.size());
            std::fflush(stdout);
            continue;
        }
        std::vector<Pic> pics;
        if (!r.file.empty() && !read_pics(r.file, pics, err)) {
            std::printf("ERR %s\n", err.c_str());
            std::printf("DONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", r.ids.size());
            std::fflush(stdout);
            continue;
        }
        if ((long long) r.ids.size() + std::max(0LL, r.max_new) > m.max_ctx()) {
            std::printf("ERR the prompt (%zu tokens) and max_new %lld are more than the context %d\n", r.ids.size(),
                        r.max_new, m.max_ctx());
            std::printf("DONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", r.ids.size());
            std::fflush(stdout);
            continue;
        }
        const double t0 = now() * 1e3;
        // the state's shared start (the last token is always read again: its logits are this request's first)
        size_t from = 0;
        while (from < state.size() && from + 1 < r.ids.size() && state[from] == r.ids[from]) ++from;
        // (a picture before `from` in either the state or the request must be in both, at the same place)
        for (const Held& q : held)
            if (q.start < (long long) from &&
                std::none_of(pics.begin(), pics.end(), [&](const Pic& x) { return x.start == q.start && x.hash == q.hash; }))
                from = (size_t) q.start;
        for (const Pic& x : pics)
            if (x.start < (long long) from &&
                std::none_of(held.begin(), held.end(), [&](const Held& q) { return x.start == q.start && x.hash == q.hash; }))
                from = (size_t) x.start;
        state.resize(from);
        held.clear();
        std::vector<mm3::Model::Image> rows;
        if (!(pics.empty() && std::find(r.ids.begin(), r.ids.end(), m.vision() ? m.vision()->image_token : -1) ==
                                  r.ids.end()) &&
            !picture_rows(m, pics, r.ids, from, rows, err)) {
            std::printf("ERR %s\n", err.c_str());
            std::printf("DONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", r.ids.size());
            std::fflush(stdout);
            continue;
        }
        for (const Pic& x : pics) held.push_back({x.start, x.hash});
        m.set_images(std::move(rows));
        const auto s_pre = m.stats();
        bool ok = true, cancel = false;
        // (the sampler's rows as the head GPU's candidates unless MM3_FULL_LOGITS or the request needs whole rows)
        const bool full_env = std::getenv("MM3_FULL_LOGITS") != nullptr;
        mm3::Cands cd;
        bool use_c = false;
        // (the tensor-parallel path takes the rest of the prompt in one call - its chunks pipelined over the GPU
        //  groups - and reports each chunk done; else calls of `chunk` tokens)
        const bool whole = m.pipelined();
        const int C = std::max(1, m.chunk() > 0 ? m.chunk() : chunk);   // (the model's: prefill-tune.txt applies there)
        auto pp_line = [&](size_t done) {
            const double ms = now() * 1e3 - t0;
            std::printf("PP %zu %zu %.1f %.1f %zu\n", done, r.ids.size(), ms,
                        ms > 0 ? 1000.0 * (double) (done - from) / ms : 0.0, from);
            std::fflush(stdout);
        };
        for (size_t a = from; a < r.ids.size();) {
            const int T = (int) (whole ? r.ids.size() - a : std::min<size_t>((size_t) C, r.ids.size() - a));
            const bool last = a + (size_t) T == r.ids.size();
            if (last) use_c = !full_env && cands_for(r, r.ids, {}, cd);   // (the first token's history: the prompt)
            if (whole)
                m.set_progress([&, a](int d) {
                    pp_line(a + (size_t) d);
                    return !in.stop.load();
                });
            const bool fok = last && use_c ? m.forward_cands(r.ids.data() + a, T, (int) a, 1, cd, err)
                                           : m.forward(r.ids.data() + a, T, (int) a, last ? 1 : 0,
                                                       last ? lg.data() : nullptr, err);
            if (whole) m.set_progress({});
            if (!fok) {
                ok = false;
                break;
            }
            const int done = whole ? m.done() : T;
            state.insert(state.end(), r.ids.begin() + (long) a, r.ids.begin() + (long) a + done);
            a += (size_t) done;
            if (!whole || T <= 16) pp_line(a);   // (a short call: a window, no chunks reported)
            if (done < T || (in.stop.load() && !last)) {
                cancel = true;
                break;
            }
        }
        const double t1 = now() * 1e3;
        m.set_images({});
        if (!ok) {
            std::printf("ERR %s\n", err.c_str());
            std::printf("DONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", r.ids.size());
            std::fflush(stdout);
            state.clear();
            held.clear();
            continue;
        }
        if (cancel) {
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0\n", r.ids.size(), t1 - t0, from);
            std::fflush(stdout);
            continue;
        }
        // (a request's own seed alone: the same seed, the same answer - mixed with the request count it was not
        //  reproducible; without one each request still gets a stream of its own)
        ++counter;
        std::mt19937_64 rng(r.seed != 0 ? r.seed : 0x9E3779B97F4A7C15ull * counter);
        const auto s_dec = m.stats();
        long long produced = 0;
        const char* finish = "length";
        bool in_think = r.think_budget > 0 && r.think_end >= 0;
        int tok = draw(use_c ? dist_c(cd.top[0], cd.extra, cd.ex[0], r, state) : dist(lg.data(), m.vocab(), r, state),
                       rng);
        // (the Monitor's tier keys too: the experts in VRAM, in the pinned pool - the CPU lane computes those from it -
        //  and a token the CPU lane's experts and the promotions; without a pool the misses come from the mapped files)
        long long d_off = 0, d_acc = 0;
        auto stat_line = [&]() {
            const auto& s = m.stats();
            const double per0 = produced > 0 ? 1.0 / (double) produced : 0.0;
            char extra[96] = "";
            int ne = 0;
            if (s.staged > s_dec.staged)
                ne += std::snprintf(extra + ne, sizeof extra - ne, " pcie=%.3f", (double) (s.staged - s_dec.staged) * per0);
            if (d_off > 0) std::snprintf(extra + ne, sizeof extra - ne, " drafts=%.3f", (double) d_acc / (double) d_off);
            const double ms = now() * 1e3 - t1;
            const double ex = (double) ((s.profiled - s_dec.profiled)), cx = (double) (s.cpu_experts - s_dec.cpu_experts);
            const double per = produced > 0 ? 1.0 / (double) produced : 0.0;
            std::printf("STAT tok_s=%.2f ms_tok=%.2f vram_hit=%.4f cpu=%.3f vram_used=%ld ram_used=%ld vram_gb=%.2f "
                        "ram_gb=%.2f %s=%.3f %s=0 promo=%.3f%s\n",
                        ms > 0 ? 1000.0 * (double) produced / ms : 0.0, produced > 0 ? ms / (double) produced : 0.0,
                        ex > 0 ? 1.0 - cx / ex : 1.0, cx * per, s.resident, s.pooled, s.vram_gb, s.pinned_gb,
                        s.pooled ? "ram_fetch" : "disk", cx * per, s.pooled ? "disk" : "ram_fetch",
                        (double) (s.promoted - s_dec.promoted) * per, extra);
        };
        // DSpark: a round drafts after the token, verifies the token and its drafts in one forward and keeps the drafts
        // the target agrees with - greedy: its argmax; sampling: a draft d (the draft's argmax) kept with the target's
        // probability of d, else the token drawn from the target's distribution without d - so every token comes from
        // the target's own distribution p (MM3_SAMPLED_DRAFTS: each draft drawn from the draft's own distribution q,
        // kept with min(1, p(d) / q(d)), else drawn from max(0, p - q) - speculative sampling, exact too; it kept ~3%
        // more drafts at temperature 1 but its draws cost more than that); then the target's next token after the last
        // kept one.  A round verifies the drafts up
        // to the first whose confidence is below the floor, or (draft_cost > 0) whose confidences' product is below
        // draft_cost - each one's from the drafts before it alone, so the cut leaves the sampling exact.
        const int B = m.draft_block();
        const bool spec = B > 0 && (r.spec < 0 ? spec_on : r.spec != 0);
        const float floor = r.conf >= 0.0f ? r.conf : conf_floor;
        const bool sampled = r.temperature > 0.0f && std::getenv("MM3_SAMPLED_DRAFTS") != nullptr;
        mm3::DraftSampling ds;
        ds.temperature = r.temperature;
        ds.top_p = r.top_p;
        ds.min_p = r.min_p;
        ds.top_k = r.top_k > 0 ? std::min(r.top_k, 64) : 64;
        std::vector<float> lgw(spec ? (size_t) (B + 1) * m.vocab() : 0);
        std::vector<int> drafts;
        std::vector<float> confs, draws((size_t) std::max(B, 1));
        std::vector<std::vector<std::pair<int, float>>> qd;
        // row i's distribution of the last forward (candidates or the whole row)
        auto row_dist = [&](int i, const std::vector<int>& hist) {
            return use_c ? dist_c(cd.top[(size_t) i], cd.extra, cd.ex[(size_t) i], r, hist)
                         : dist(lgw.data() + (size_t) i * m.vocab(), m.vocab(), r, hist);
        };
        std::string fail;   // (a forward or a draft that failed while the answer is written)
        // one token out: false when the answer ends with it (an end token, max_new, the context)
        auto emit = [&](int t) -> bool {
            std::printf("T %d\n", t);
            ++produced;
            if (in_think && t == r.think_end) in_think = false;
            if (produced % 8 == 0) stat_line();
            std::fflush(stdout);
            for (int e : eos)
                if (t == e) {
                    finish = "stop";
                    return false;
                }
            return produced < r.max_new && (int) state.size() < m.max_ctx();
        };
        for (;;) {
            if (in.stop.load()) {
                finish = "cancel";
                break;
            }
            if (in_think && (produced >= r.think_budget || in.wrap.load()) && tok != r.think_end)
                tok = r.think_end;   // (the budget, or WRAP)
            if (!emit(tok)) break;
            const int pos = (int) state.size();
            int k = 0;
            if (spec) {   // (the drafts that can still be answer tokens: max_new, the context, the thinking budget)
                long long room = std::min<long long>(r.max_new - produced, (long long) m.max_ctx() - pos - 1);
                if (in_think) room = std::min<long long>(room, r.think_budget - produced);
                if (in_think && in.wrap.load()) room = 0;
                if (room > 0) {
                    if (sampled) {
                        for (float& u : draws) u = (float) std::uniform_real_distribution<double>(0.0, 1.0)(rng);
                        if (!m.draft_sampled(tok, pos, B, ds, draws, drafts, confs, qd, err)) {
                            fail = err;
                            break;
                        }
                    } else if (!m.draft(tok, pos, B, drafts, confs, err)) {
                        fail = err;
                        break;
                    }
                    double surv = 1.0;
                    while (k < (int) drafts.size() && k < room && confs[(size_t) k] >= floor) {
                        surv *= confs[(size_t) k];
                        if (draft_cost > 0.0f && surv < draft_cost) break;
                        ++k;
                    }
                }
            }
            std::vector<int> win(1, tok);
            win.insert(win.end(), drafts.begin(), drafts.begin() + k);
            {   // (the window's candidates: the penalised ids of the history with the window's tokens)
                std::vector<int> more(win);
                use_c = !full_env && cands_for(r, state, more, cd);
            }
            if (k == 0) {
                if (!(use_c ? m.forward_cands(&tok, 1, pos, 1, cd, err)
                            : m.forward(&tok, 1, pos, 1, lgw.empty() ? lg.data() : lgw.data(), err))) {
                    fail = err;
                    break;
                }
                state.push_back(tok);
                tok = draw(use_c ? row_dist(0, state)
                                 : dist(lgw.empty() ? lg.data() : lgw.data(), m.vocab(), r, state), rng);
                continue;
            }
            if (!(use_c ? m.forward_cands(win.data(), k + 1, pos, k + 1, cd, err)
                        : m.forward(win.data(), k + 1, pos, k + 1, lgw.data(), err))) {
                fail = err;
                break;
            }
            state.push_back(tok);
            d_off += k;
            bool more = true;
            int next = -1;
            for (int i = 0; i <= k && more; ++i) {
                const auto d = row_dist(i, state);
                if (i == k) {   // every draft kept: the target's token after them
                    next = draw(d, rng);
                    break;
                }
                const int x = drafts[(size_t) i];
                double px = 0, qx = 1;
                for (const auto& [t, q] : d)
                    if (t == x) px = q;
                if (sampled)
                    for (const auto& [t, q] : qd[(size_t) i])
                        if (t == x) qx = q;
                const bool keep = r.temperature > 0.0f
                                      ? std::uniform_real_distribution<double>(0.0, 1.0)(rng) * qx < px
                                      : d[0].first == x;
                if (!keep) {
                    next = r.temperature > 0.0f ? (sampled ? draw_residual(d, qd[(size_t) i], rng) : draw(d, rng, x))
                                                : d[0].first;
                    break;
                }
                ++d_acc;
                if (in.stop.load()) {
                    finish = "cancel";
                    more = false;
                    break;
                }
                more = emit(x);
                state.push_back(x);
            }
            if (!more) break;
            tok = next;
        }
        if (spec && d_off > 0)
            std::fprintf(stderr, "maya mm3 serve: DSpark kept %lld of %lld drafts\n", d_acc, d_off);
        if (!fail.empty()) {
            std::fprintf(stderr, "mm3: %s\n", fail.c_str());
            std::printf("ERR %s\n", fail.c_str());
            finish = "cancel";
            state.clear();
            held.clear();
        }
        stat_line();
        const double t2 = now() * 1e3;
        const auto& se = m.stats();
        const long long look = se.profiled - s_dec.profiled, cpu = se.cpu_experts - s_dec.cpu_experts;
        std::printf("DONE %lld %zu %.1f %.1f %s %lld %lld %zu %lld %lld\n", produced, r.ids.size(), t1 - t0, t2 - t1,
                    finish, d_acc, d_off, from, look - cpu, look);
        std::fflush(stdout);
        if (!profile_out.empty()) {   // (this PC's routing learned: the next start places its experts by it)
            std::string perr;
            const std::string tmp = profile_out + ".tmp";
            if (!m.save_profile(tmp, perr) || !strata::ds4::os::replace_file(tmp, profile_out))
                std::fprintf(stderr, "mm3: the expert profile %s not saved %s\n", profile_out.c_str(), perr.c_str());
        }
        std::fprintf(stderr,
                     "maya mm3 serve: prompt %zu tokens = %zu reused + %zu read in %.0f ms (%.1f tok/s), %lld generated "
                     "in %.0f ms (%.2f tok/s); CPU lane %.1f%% of the experts\n",
                     r.ids.size(), from, r.ids.size() - from, t1 - t0,
                     t1 > t0 ? 1000.0 * (double) (r.ids.size() - from) / (t1 - t0) : 0.0, produced, t2 - t1,
                     t2 > t1 ? 1000.0 * (double) produced / (t2 - t1) : 0.0, look > 0 ? 100.0 * cpu / look : 0.0);
        (void) s_pre;
    }
    return 0;
}

// --serve with --parallel N (N > 1): several conversations at once, as the GLM engine runs them (serve/server.py's
// multi protocol).  READY ... multi=<n>; GEN / GENI lines carry rid=<id> and every line a request causes carries it
// after its keyword (T <rid> <id>, PP <rid> ..., DONE <rid> ..., ERR <rid> ...); STOP <rid> ends one request, STOP
// alone all of them.  A request takes one of the n sequences (a KV cache each): the idle one whose tokens its prompt
// extends the most (only the rest is read), else the least recently used.  A prompt is read in chunks with a decode
// step of the running requests between two (not a prompt with pictures: the chunks' rows are its pictures'); the
// running requests decode together, a token each a step (Model::forward_rows; a request alone: the token path) -
// greedy, each request's tokens are its tokens alone (the CPU lane's experts aside: their last bits differ from the
// GPU's, and which experts the CPU lane takes depends on what runs beside).  No DSpark drafting.
int serve_multi(mm3::Model& m, const std::vector<int>& eos, int chunk, const std::string& kv_fmt,
                const std::string& profile_out) {
    const int N = m.seqs();
    serve_header(m, false, kv_fmt);
    std::printf("READY %d stop multi=%d\n", m.max_ctx(), N);
    std::fflush(stdout);
    std::fprintf(stderr, "maya mm3 serve: %d conversations at once (--parallel); GEN <max_new> [keys] rid=<id> <ids>; "
                         "STOP <rid>, STOP, QUIT\n", N);
#if defined(SIGPIPE)
    std::signal(SIGPIPE, SIG_IGN);
#endif
    Lines in;
    in.start();
    struct Held {
        long long start;
        uint64_t hash;
    };
    struct Seq {
        std::vector<int> state;   // the tokens in its cache, positions 0 ..
        std::vector<Held> held;   // the pictures in its cache
        bool busy = false;
        unsigned long long used = 0;
    };
    std::vector<Seq> seqs((size_t) N);
    unsigned long long clock = 0, counter = 0;
    struct Gen {
        int rid = -1, seq = 0;
        Req r;
        std::mt19937_64 rng;
        long long produced = 0;
        const char* finish = "length";
        bool in_think = false, done = false, use_c = false;
        bool shown = false;        // (tok written, not read yet)
        bool flight = false;       // (tok in the pipeline)
        int tok = -1;              // the next token: written at the next step, then read
        size_t from = 0;
        double t0 = 0, t1 = 0;
        long prof0 = 0, cpu0 = 0;  // (the engine's routings / CPU lane experts when its decode began)
        mm3::Cands cd;
    };
    std::vector<std::unique_ptr<Gen>> run;
    std::deque<std::string> waiting;   // GEN lines no idle sequence could take yet
    const bool full_env = std::getenv("MM3_FULL_LOGITS") != nullptr;
    const int V = m.vocab();
    std::vector<float> lg, lg1((size_t) V);
    std::vector<std::pair<int, float>> top;
    std::vector<float> ex;
    // (the Monitor's STAT line: everything decoded since the engine was last idle)
    unsigned long long multi_n = 0;
    double busy_t0 = 0;
    mm3::Stats busy_s0;
    auto stat_line = [&]() {
        const auto& s = m.stats();
        const double ms = now() * 1e3 - busy_t0, n = (double) std::max(1ULL, multi_n);
        const double xp = (double) (s.profiled - busy_s0.profiled), cx = (double) (s.cpu_experts - busy_s0.cpu_experts);
        std::printf("STAT tok_s=%.2f ms_tok=%.2f vram_hit=%.4f cpu=%.3f vram_used=%ld ram_used=%ld vram_gb=%.2f "
                    "ram_gb=%.2f %s=%.3f %s=0 promo=%.3f active=%zu\n",
                    ms > 0 ? 1000.0 * (double) multi_n / ms : 0.0, ms / n, xp > 0 ? 1.0 - cx / xp : 1.0, cx / n,
                    s.resident, s.pooled, s.vram_gb, s.pinned_gb, s.pooled ? "ram_fetch" : "disk", cx / n,
                    s.pooled ? "disk" : "ram_fetch", (double) (s.promoted - busy_s0.promoted) / n, run.size());
    };
    auto release = [&](int s) {
        seqs[(size_t) s].busy = false;
        seqs[(size_t) s].used = ++clock;
    };
    // a request's end: its DONE line (the engine's counters over its decode are shared with the others meanwhile)
    auto finish = [&](Gen& g) {
        const double t2 = now() * 1e3;
        const auto& se = m.stats();
        const long long look = se.profiled - g.prof0, cpu = se.cpu_experts - g.cpu0;
        std::printf("DONE %d %lld %zu %.1f %.1f %s 0 0 %zu %lld %lld\n", g.rid, g.produced, g.r.ids.size(), g.t1 - g.t0,
                    t2 - g.t1, g.finish, g.from, look - cpu, look);
        std::fprintf(stderr,
                     "maya mm3 serve: request %d (sequence %d): prompt %zu tokens = %zu reused + %zu read in %.0f ms, "
                     "%lld generated in %.0f ms (%.2f tok/s, %zu running) - %s\n",
                     g.rid, g.seq, g.r.ids.size(), g.from, g.r.ids.size() - g.from, g.t1 - g.t0, g.produced, t2 - g.t1,
                     t2 > g.t1 ? 1000.0 * (double) g.produced / (t2 - g.t1) : 0.0, run.size(), g.finish);
        release(g.seq);
        in.forget(g.rid);
        g.done = true;
    };
    auto emit = [&](Gen& g, int t) -> bool {
        std::printf("T %d %d\n", g.rid, t);
        ++g.produced;
        ++multi_n;
        if (g.in_think && t == g.r.think_end) g.in_think = false;
        if (multi_n % 8 == 0) stat_line();
        for (int e : eos)
            if (t == e) {
                g.finish = "stop";
                return false;
            }
        return g.produced < g.r.max_new && (int) seqs[(size_t) g.seq].state.size() < m.max_ctx();
    };
    auto sweep = [&]() {
        run.erase(std::remove_if(run.begin(), run.end(), [](const std::unique_ptr<Gen>& p) { return p->done; }),
                  run.end());
        if (!run.empty()) return;
        in.stop.store(false);   // (a STOP for all: every request it found has ended)
        if (!profile_out.empty()) {   // (this PC's routing learned: the next start places its experts by it)
            std::string perr;
            const std::string tmp = profile_out + ".tmp";
            if (!m.save_profile(tmp, perr) || !strata::ds4::os::replace_file(tmp, profile_out))
                std::fprintf(stderr, "mm3: the expert profile %s not saved %s\n", profile_out.c_str(), perr.c_str());
        }
    };
    // the tokens drawn and not written yet: written (T lines); a request that ends there finishes
    auto show = [&]() {
        for (auto& gp : run) {
            Gen& g = *gp;
            if (g.done || g.flight || g.shown) continue;
            if (in.stop.load() || in.stopped(g.rid)) {
                g.finish = "cancel";
                finish(g);
                continue;
            }
            if (g.in_think && (g.produced >= g.r.think_budget || in.wrap.load()) && g.tok != g.r.think_end)
                g.tok = g.r.think_end;   // (the budget, or WRAP)
            if (!emit(g, g.tok)) {
                finish(g);
                continue;
            }
            g.shown = true;
        }
        std::fflush(stdout);
    };
    // the requests whose written token is to be read
    auto ready = [&]() {
        std::vector<Gen*> v;
        for (auto& gp : run)
            if (!gp->done && !gp->flight && gp->shown) v.push_back(gp.get());
        return v;
    };
    // rows' tokens, sequences, positions, and each one's candidates (the penalised ids of the history with its token)
    auto rows_of = [&](const std::vector<Gen*>& rows, std::vector<int>& ids, std::vector<int>& sq, std::vector<int>& ps) {
        ids.clear();
        sq.clear();
        ps.clear();
        for (Gen* gp : rows) {
            Gen& g = *gp;
            const std::vector<int>& st = seqs[(size_t) g.seq].state;
            ids.push_back(g.tok);
            sq.push_back(g.seq);
            ps.push_back((int) st.size());
            g.use_c = !full_env && cands_for(g.r, st, std::vector<int>(1, g.tok), g.cd);
        }
    };
    // a request's token read, its logits back: the next token drawn
    auto took = [&](Gen& g, const float* row) {
        std::vector<int>& st = seqs[(size_t) g.seq].state;
        st.push_back(g.tok);
        if (g.use_c) {   // (as Model::forward_cands on the layer split: the row's candidates on the host)
            mm3::host_cands(row, V, g.cd.kc, g.cd.extra, top, ex);
            g.tok = draw(dist_c(top, g.cd.extra, ex, g.r, st), g.rng);
        } else {
            g.tok = draw(dist(row, V, g.r, st), g.rng);
        }
        g.shown = false;
        g.flight = false;
    };
    auto failed = [&](Gen& g, const std::string& err) {   // (its cache may hold part of the step: read again)
        std::printf("ERR %d %s\n", g.rid, err.c_str());
        seqs[(size_t) g.seq].state.clear();
        seqs[(size_t) g.seq].held.clear();
        g.flight = false;
        g.finish = "cancel";
        finish(g);
    };
    // a decode step: the written tokens of every request going on read together (Model::forward_rows)
    std::vector<int> ids_, sq_, ps_;
    auto feed = [&](const std::vector<Gen*>& rows) {
        if (rows.empty()) return;
        rows_of(rows, ids_, sq_, ps_);
        lg.resize(rows.size() * (size_t) V);
        std::string err;
        if (!m.forward_rows(ids_.data(), sq_.data(), ps_.data(), (int) rows.size(), lg.data(), err)) {
            std::fprintf(stderr, "mm3: %s\n", err.c_str());
            for (Gen* gp : rows) failed(*gp, err);
        } else {
            for (size_t i = 0; i < rows.size(); ++i) took(*rows[i], lg.data() + i * (size_t) V);
        }
        std::fflush(stdout);
    };
    auto step = [&]() {
        show();
        feed(ready());
        sweep();
    };
    // a layer split's pipeline (Model::pipe_tick; MAYA_MM3_PIPE=0: off - the requests read together instead): each
    // tick a group of the ready requests enters the first GPU - about 1 / stages of all the requests, so the GPUs each
    // hold one - and the groups past the last GPU draw their next tokens
    const int stages = mm3::env("MM3_PIPE") && std::atoi(mm3::env("MM3_PIPE")) == 0 ? 1 : m.pipe_stages();
    std::map<int, std::vector<Gen*>> groups;
    int ptag = 0;
    auto pipe_fail = [&](const std::string& err, std::vector<Gen*> also) {
        std::fprintf(stderr, "mm3: %s\n", err.c_str());
        for (auto& kv : groups) also.insert(also.end(), kv.second.begin(), kv.second.end());
        groups.clear();
        m.pipe_drop();
        for (Gen* gp : also)
            if (!gp->done) failed(*gp, err);
    };
    auto tick = [&](const std::vector<Gen*>& group) {
        std::string err;
        int tag = -1;
        if (!group.empty()) {
            rows_of(group, ids_, sq_, ps_);
            tag = ++ptag;
            groups[tag] = group;
            for (Gen* gp : group) gp->flight = true;
        }
        if (!m.pipe_tick(ids_.data(), sq_.data(), ps_.data(), (int) group.size(), tag, err)) {
            pipe_fail(err, {});
            return;
        }
        std::vector<float> lgp;
        while (m.pipe_take(tag, lgp)) {
            auto it = groups.find(tag);
            if (it == groups.end()) continue;
            for (size_t i = 0; i < it->second.size(); ++i) took(*it->second[i], lgp.data() + i * (size_t) V);
            groups.erase(it);
        }
    };
    auto pipe_step = [&]() {
        show();
        std::vector<Gen*> rows = ready();
        if (m.pipe_busy() == 0 && rows.size() <= 1) {   // (a request alone: the token path)
            feed(rows);
            sweep();
            return;
        }
        const int free_ = std::max(1, stages - std::min(stages - 1, m.pipe_busy()));
        size_t n = std::min<size_t>(rows.size(), (rows.size() + (size_t) free_ - 1) / (size_t) free_);
        n = std::min<size_t>(n, 16);
        rows.resize(n);
        tick(rows);
        std::fflush(stdout);
        sweep();
    };
    auto drain = [&]() {   // (before a prompt's read: every group out of the pipeline)
        while (m.pipe_busy() > 0) tick({});
    };
    auto idle_seq = [&]() {
        for (const Seq& s : seqs)
            if (!s.busy) return true;
        return false;
    };
    // a GEN / GENI line: its sequence, its prompt read, its first token drawn - then it runs with the others
    auto admit = [&](const std::string& line) {
        Req r;
        std::string err;
        const bool geni = line.rfind("GENI ", 0) == 0;
        const bool parsed = parse_gen(line, r) && !(geni && r.file.empty());
        if (r.rid < 0) {
            std::printf("ERR expected rid=<id> in each GEN line (this engine runs %d conversations at once)\n", N);
            std::fflush(stdout);
            return;
        }
        auto refuse = [&](const std::string& why) {
            std::printf("ERR %d %s\n", r.rid, why.c_str());
            std::printf("DONE %d 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", r.rid, r.ids.size());
            std::fflush(stdout);
            in.forget(r.rid);
        };
        if (!parsed) return refuse("expected: GEN <max_new> [keys] <id,id,...> (GENI: the pictures' file before the ids)");
        if (in.stopped(r.rid)) {   // (stopped while it waited)
            std::printf("DONE %d 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", r.rid, r.ids.size());
            std::fflush(stdout);
            return;
        }
        std::vector<Pic> pics;
        if (!r.file.empty() && !read_pics(r.file, pics, err)) return refuse(err);
        if ((long long) r.ids.size() + std::max(0LL, r.max_new) > m.max_ctx())
            return refuse("the prompt (" + std::to_string(r.ids.size()) + " tokens) and max_new " +
                          std::to_string(r.max_new) + " are more than the context " + std::to_string(m.max_ctx()));
        // the sequence: idle, the longest shared start (the last token always read again: its logits are this
        // request's first; a picture before it in either must be in both, at the same place), else least recently
        // used - a short shared start never takes a longer conversation's cache while one drops nothing (pick_slot)
        std::vector<mm3::SlotView> views((size_t) N);
        for (int s = 0; s < N; ++s) {
            const Seq& S = seqs[(size_t) s];
            views[(size_t) s].busy = S.busy;
            views[(size_t) s].size = S.state.size();
            views[(size_t) s].used = S.used;
            if (S.busy) continue;
            size_t from = 0;
            while (from < S.state.size() && from + 1 < r.ids.size() && S.state[from] == r.ids[from]) ++from;
            for (const Held& q : S.held)
                if (q.start < (long long) from && std::none_of(pics.begin(), pics.end(), [&](const Pic& x) {
                        return x.start == q.start && x.hash == q.hash;
                    }))
                    from = (size_t) q.start;
            for (const Pic& x : pics)
                if (x.start < (long long) from && std::none_of(S.held.begin(), S.held.end(), [&](const Held& q) {
                        return x.start == q.start && x.hash == q.hash;
                    }))
                    from = (size_t) x.start;
            views[(size_t) s].from = from;
        }
        const int best = mm3::pick_slot(views);
        const size_t bfrom = best >= 0 ? views[(size_t) best].from : 0;
        if (best < 0) {   // (every sequence busy - the server sends at most n at once: after one ends)
            waiting.push_back(line);
            return;
        }
        if (stages > 1) drain();   // (the pipeline empty: the prompt's forwards and its pictures use every GPU)
        if (run.empty()) in.stop.store(false);   // (a STOP for all before this request: nothing it could stop)
        Seq& S = seqs[(size_t) best];
        S.busy = true;
        auto gp = std::make_unique<Gen>();
        Gen& g = *gp;
        g.rid = r.rid;
        g.seq = best;
        g.r = std::move(r);
        g.t0 = now() * 1e3;
        g.from = bfrom;
        const std::vector<int>& ids = g.r.ids;
        S.state.resize(bfrom);
        S.held.clear();
        std::vector<mm3::Model::Image> rows;
        if (!(pics.empty() && std::find(ids.begin(), ids.end(), m.vision() ? m.vision()->image_token : -1) ==
                                  ids.end()) &&
            !picture_rows(m, pics, ids, bfrom, rows, err)) {
            release(best);
            return refuse(err);
        }
        for (const Pic& x : pics) S.held.push_back({x.start, x.hash});
        const bool between = rows.empty();   // (decode steps of the others between its chunks)
        m.bind(best);
        m.set_images(std::move(rows));
        const int C = std::max(1, m.chunk() > 0 ? m.chunk() : chunk);
        bool ok = true, cancel = false;
        for (size_t a = bfrom; a < ids.size();) {
            const int T = (int) std::min<size_t>((size_t) C, ids.size() - a);
            const bool last = a + (size_t) T == ids.size();
            if (last) g.use_c = !full_env && cands_for(g.r, ids, {}, g.cd);   // (the first token's history: the prompt)
            m.bind(best);
            if (!m.forward(ids.data() + a, T, (int) a, last ? 1 : 0, last ? lg1.data() : nullptr, err)) {
                ok = false;
                break;
            }
            S.state.insert(S.state.end(), ids.begin() + (long) a, ids.begin() + (long) a + T);
            a += (size_t) T;
            const double ms = now() * 1e3 - g.t0;
            std::printf("PP %d %zu %zu %.1f %.1f %zu\n", g.rid, a, ids.size(), ms,
                        ms > 0 ? 1000.0 * (double) (a - bfrom) / ms : 0.0, bfrom);
            std::fflush(stdout);
            if (last) break;
            if (in.stop.load() || in.stopped(g.rid)) {
                cancel = true;
                break;
            }
            if (between && !run.empty()) step();   // (the running requests' next token)
        }
        m.set_images({});
        g.t1 = now() * 1e3;
        if (!ok) {
            std::printf("ERR %d %s\n", g.rid, err.c_str());
            std::printf("DONE %d 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", g.rid, ids.size());
            std::fflush(stdout);
            S.state.clear();
            S.held.clear();
            release(best);
            in.forget(g.rid);
            return;
        }
        if (cancel) {
            std::printf("DONE %d 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0\n", g.rid, ids.size(), g.t1 - g.t0, bfrom);
            std::fflush(stdout);
            release(best);
            in.forget(g.rid);
            return;
        }
        // (a request's own seed alone, as with one conversation: the same seed, the same answer)
        ++counter;
        g.rng.seed(g.r.seed != 0 ? g.r.seed : 0x9E3779B97F4A7C15ull * counter);
        g.in_think = g.r.think_budget > 0 && g.r.think_end >= 0;
        if (g.use_c) {
            mm3::host_cands(lg1.data(), V, g.cd.kc, g.cd.extra, top, ex);
            g.tok = draw(dist_c(top, g.cd.extra, ex, g.r, S.state), g.rng);
        } else {
            g.tok = draw(dist(lg1.data(), V, g.r, S.state), g.rng);
        }
        if (run.empty()) {   // (the STAT lines' window: from the engine's first busy moment)
            busy_t0 = now() * 1e3;
            busy_s0 = m.stats();
            multi_n = 0;
        }
        const auto& s = m.stats();
        g.prof0 = s.profiled;
        g.cpu0 = s.cpu_experts;
        run.push_back(std::move(gp));
    };
    bool quit = false;
    auto handle = [&](const std::string& l) {
        if (l == "QUIT") {
            quit = true;
        } else if (l.rfind("GEN ", 0) == 0 || l.rfind("GENI ", 0) == 0) {
            admit(l);
        } else {
            std::printf("ERR unsupported command '%s' (GEN, GENI, STOP and QUIT)\n", l.substr(0, l.find(' ')).c_str());
            std::fflush(stdout);
        }
    };
    std::string line;
    while (!quit) {
        if (run.empty() && waiting.empty()) {   // (idle: wait for a line)
            if (!in.next(line)) break;
            handle(line);
        }
        while (!quit && in.try_next(line)) handle(line);
        if (quit) break;
        while (!waiting.empty() && idle_seq()) {
            line = std::move(waiting.front());
            waiting.pop_front();
            admit(line);
        }
        if (!run.empty()) stages > 1 ? pipe_step() : step();
        else if (waiting.empty() && in.ended()) break;
    }
    return 0;
}

// --rows-check G with --parallel N (2 or more) and --ids: forward_rows against each token alone.  Sequence s reads part
// s of the ids (N equal parts); then G steps, each first every sequence's token alone (bind + forward, the token path)
// and then the same tokens together (forward_rows: 2 sequences on the even steps, all N on the odd ones) - the
// logits compared bit for bit - and each sequence's greedy token next.  Run it with the CPU lane off (--cpu-threads -1):
// the lane's experts may come out with other last bits.
int rows_check(mm3::Model& m, const std::vector<int>& ids, int steps) {
    const int N = m.seqs(), V = m.vocab();
    if (N < 2 || (int) ids.size() < 2 * N) {
        std::fprintf(stderr, "--rows-check: --parallel 2 or more, and --ids with at least 2 tokens a sequence\n");
        return 2;
    }
    const size_t part = ids.size() / (size_t) N;
    std::vector<int> tok((size_t) N), pos((size_t) N);
    std::vector<float> one((size_t) N * V), rows((size_t) N * V);
    std::string err;
    auto argmax = [&](const float* r) { return (int) (std::max_element(r, r + V) - r); };
    for (int s = 0; s < N; ++s) {
        m.bind(s);
        if (!m.forward(ids.data() + (size_t) s * part, (int) part, 0, 1, one.data(), err)) {
            std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
            return 1;
        }
        tok[(size_t) s] = argmax(one.data());
        pos[(size_t) s] = (int) part;
    }
    long bad = 0;
    for (int k = 0; k < steps; ++k) {
        const int R = k % 2 == 0 ? 2 : N;
        std::vector<int> sq((size_t) R);
        for (int r = 0; r < R; ++r) {
            sq[(size_t) r] = R == N ? r : (k / 2 + r) % N;   // (the pairs move through the sequences)
            m.bind(sq[(size_t) r]);
            if (!m.forward(&tok[(size_t) sq[(size_t) r]], 1, pos[(size_t) sq[(size_t) r]], 1,
                           one.data() + (size_t) r * V, err)) {
                std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
                return 1;
            }
        }
        std::vector<int> t((size_t) R), p((size_t) R);
        for (int r = 0; r < R; ++r) {
            t[(size_t) r] = tok[(size_t) sq[(size_t) r]];
            p[(size_t) r] = pos[(size_t) sq[(size_t) r]];
        }
        if (!m.forward_rows(t.data(), sq.data(), p.data(), R, rows.data(), err)) {
            std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
            return 1;
        }
        int same = 0;
        float worst = 0.0f;
        for (int r = 0; r < R; ++r) {
            const float* a = one.data() + (size_t) r * V;
            const float* b = rows.data() + (size_t) r * V;
            same += std::memcmp(a, b, (size_t) V * 4) == 0;
            for (int i = 0; i < V; ++i) worst = std::max(worst, std::fabs(a[i] - b[i]));
            tok[(size_t) sq[(size_t) r]] = argmax(a);
            ++pos[(size_t) sq[(size_t) r]];
        }
        bad += R - same;
        std::printf("rows check: step %d, %d rows: %d bit for bit, max |diff| %g\n", k, R, same, worst);
        std::fflush(stdout);
    }
    // a layer split's pipeline (Model::pipe_tick): each step every sequence's token alone first, then the same tokens
    // as groups through the pipeline - the even steps one sequence a group, the odd ones two - entering a tick apart
    for (int k = 0; k < steps && m.pipe_stages() > 1; ++k) {
        for (int s = 0; s < N; ++s) {
            m.bind(s);
            if (!m.forward(&tok[(size_t) s], 1, pos[(size_t) s], 1, one.data() + (size_t) s * V, err)) {
                std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
                return 1;
            }
        }
        const int gs = k % 2 == 0 ? 1 : 2;
        for (int s0 = 0, tag = 0; s0 < N || m.pipe_busy() > 0; s0 += gs, ++tag) {
            const int n = s0 < N ? std::min(gs, N - s0) : 0;
            std::vector<int> sq((size_t) n), t((size_t) n), p((size_t) n);
            for (int i = 0; i < n; ++i) {
                sq[(size_t) i] = s0 + i;
                t[(size_t) i] = tok[(size_t) (s0 + i)];
                p[(size_t) i] = pos[(size_t) (s0 + i)];
            }
            if (!m.pipe_tick(t.data(), sq.data(), p.data(), n, n ? s0 : -1, err)) {
                std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
                return 1;
            }
            int first;
            std::vector<float> lgp;
            while (m.pipe_take(first, lgp))
                for (size_t i = 0; i < lgp.size() / (size_t) V; ++i)
                    std::memcpy(rows.data() + (size_t) (first + (int) i) * V, lgp.data() + i * V, (size_t) V * 4);
        }
        int same = 0;
        float worst = 0.0f;
        for (int s = 0; s < N; ++s) {
            const float* a = one.data() + (size_t) s * V;
            const float* b = rows.data() + (size_t) s * V;
            same += std::memcmp(a, b, (size_t) V * 4) == 0;
            for (int i = 0; i < V; ++i) worst = std::max(worst, std::fabs(a[i] - b[i]));
            tok[(size_t) s] = argmax(a);
            ++pos[(size_t) s];
        }
        bad += N - same;
        std::printf("rows check: pipeline step %d (%d GPUs, groups of %d): %d of %d bit for bit, max |diff| %g\n", k,
                    m.pipe_stages(), gs, same, N, worst);
        std::fflush(stdout);
    }
    std::printf("rows check: %s (%ld rows differ)\n", bad == 0 ? "PASS" : "FAIL", bad);
    return bad == 0 ? 0 : 1;
}

// --score FILE [--write-ref OUT]: the quality check against a reference's top-K log-probs (tools/mm3/kv_gate.py writes
// FILE from ref_topk.npz, or from another run's --write-ref).  FILE: "MM3R", int32 rows, int32 T, int32 K, then per row
// int32 ids[T], int32 ref_ids[T][K], float ref_lp[T][K] (K 0: none), uint8 mask[T - 1] (position t predicts ids[t + 1];
// 1: an assistant token).  Each row runs from position 0; per position t < T - 1: KL(ref || engine) over the
// reference's top K plus the rest of the vocabulary as one bucket (calib_mm31.py's), whether the top tokens agree, and
// the target's NLL - summed over the assistant tokens and over all.  OUT: this engine's own top 64 in FILE's format
// (the reference for another KV format's run).  Prints "SCORE ..." lines (one a row, then all).
int score_main(mm3::Model& m, const std::string& path, const std::string& out_path) {
    std::ifstream f(path, std::ios::binary);
    char magic[4];
    int32_t hdr[3];
    if (!f.read(magic, 4) || std::memcmp(magic, "MM3R", 4) != 0 || !f.read((char*) hdr, 12)) {
        std::fprintf(stderr, "%s: not a reference file (MM3R)\n", path.c_str());
        return 2;
    }
    const int rows = hdr[0], T = hdr[1], K = hdr[2], V = m.vocab(), KO = 64;
    if (T < 2 || T > m.max_ctx()) {
        std::fprintf(stderr, "%s: rows of %d tokens, the context is %d (--max-context)\n", path.c_str(), T, m.max_ctx());
        return 2;
    }
    std::FILE* fo = out_path.empty() ? nullptr : std::fopen(out_path.c_str(), "wb");
    if (!out_path.empty() && !fo) {
        std::fprintf(stderr, "%s: cannot write\n", out_path.c_str());
        return 2;
    }
    if (fo) {
        const int32_t oh[3] = {rows, T, KO};
        std::fwrite("MM3R", 1, 4, fo);
        std::fwrite(oh, 4, 3, fo);
    }
    const int C = 256;   // (the logits of 256 positions at a time: 205 MB)
    std::vector<float> lg((size_t) C * V);
    double kl_a = 0, kl_all = 0, same_a = 0, same_all = 0, nll_a = 0, nll_all = 0;
    long n_a = 0, n_all = 0;
    std::string err;
    for (int r = 0; r < rows; ++r) {
        std::vector<int32_t> ids((size_t) T), rid((size_t) T * K);
        std::vector<float> rlp((size_t) T * K);
        std::vector<uint8_t> mask((size_t) T - 1);
        if (!f.read((char*) ids.data(), 4 * (std::streamsize) T) ||
            !f.read((char*) rid.data(), 4 * (std::streamsize) rid.size()) ||
            !f.read((char*) rlp.data(), 4 * (std::streamsize) rlp.size()) ||
            !f.read((char*) mask.data(), (std::streamsize) mask.size())) {
            std::fprintf(stderr, "%s: row %d cut short\n", path.c_str(), r);
            return 2;
        }
        std::vector<int32_t> oid((size_t) T * KO);
        std::vector<float> olp((size_t) T * KO);
        double rk = 0, rka = 0, rs = 0, rsa = 0, rn = 0, rna = 0;
        long ra = 0;
        const double t0 = now();
        for (int a = 0; a < T; a += C) {
            const int n = std::min(C, T - a);
            if (!m.forward(ids.data() + a, n, a, n, lg.data(), err)) {
                std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
                return 1;
            }
            for (int i = 0; i < n; ++i) {
                const int t = a + i;
                const float* row = lg.data() + (size_t) i * V;
                double mx = row[0];
                int am = 0;
                for (int v = 1; v < V; ++v)
                    if (row[v] > mx) {
                        mx = row[v];
                        am = v;
                    }
                double se = 0;
                for (int v = 0; v < V; ++v) se += std::exp((double) row[v] - mx);
                const double lse = mx + std::log(se);
                if (fo) {   // (this engine's top 64)
                    std::vector<std::pair<float, int>> top((size_t) V);
                    for (int v = 0; v < V; ++v) top[(size_t) v] = {row[v], v};
                    std::partial_sort(top.begin(), top.begin() + KO, top.end(),
                                      [](const auto& x, const auto& y) { return x.first > y.first; });
                    for (int j = 0; j < KO; ++j) {
                        oid[(size_t) t * KO + j] = top[(size_t) j].second;
                        olp[(size_t) t * KO + j] = (float) ((double) top[(size_t) j].first - lse);
                    }
                }
                if (t + 1 >= T) continue;
                const double nll = lse - (double) row[ids[(size_t) t + 1]];
                double kl = 0, same = 0;
                if (K > 0) {
                    double pr_sum = 0, q_sum = 0;
                    for (int j = 0; j < K; ++j) {
                        const double rl = rlp[(size_t) t * K + j];
                        const double ql = (double) row[rid[(size_t) t * K + j]] - lse;
                        const double pr = std::exp(rl);
                        kl += pr * (rl - ql);
                        pr_sum += pr;
                        q_sum += std::exp(ql);
                    }
                    const double rest_r = std::max(1e-12, 1.0 - pr_sum), rest_q = std::max(1e-12, 1.0 - q_sum);
                    kl += rest_r * (std::log(rest_r) - std::log(rest_q));
                    same = am == rid[(size_t) t * K] ? 1.0 : 0.0;
                }
                rk += kl;
                rs += same;
                rn += nll;
                if (mask[(size_t) t]) {
                    rka += kl;
                    rsa += same;
                    rna += nll;
                    ++ra;
                }
            }
        }
        const long rnall = T - 1;
        kl_all += rk;
        same_all += rs;
        nll_all += rn;
        n_all += rnall;
        kl_a += rka;
        same_a += rsa;
        nll_a += rna;
        n_a += ra;
        std::printf("SCORE row=%d tokens=%d asst=%ld asst_kl=%.5f asst_same=%.4f asst_nll=%.4f all_kl=%.5f all_same=%.4f "
                    "all_nll=%.4f s=%.1f\n", r, T, ra, ra ? rka / ra : 0.0, ra ? rsa / ra : 0.0, ra ? rna / ra : 0.0,
                    rk / rnall, rs / rnall, rn / rnall, now() - t0);
        std::fflush(stdout);
        if (fo) {
            std::fwrite(ids.data(), 4, ids.size(), fo);
            std::fwrite(oid.data(), 4, oid.size(), fo);
            std::fwrite(olp.data(), 4, olp.size(), fo);
            std::fwrite(mask.data(), 1, mask.size(), fo);
        }
    }
    if (fo) std::fclose(fo);
    std::printf("SCORE all rows=%d tokens=%ld asst=%ld asst_kl=%.5f asst_same=%.4f asst_nll=%.4f all_kl=%.5f "
                "all_same=%.4f all_nll=%.4f\n", rows, n_all, n_a, n_a ? kl_a / n_a : 0.0, n_a ? same_a / n_a : 0.0,
                n_a ? nll_a / n_a : 0.0, n_all ? kl_all / n_all : 0.0, n_all ? same_all / n_all : 0.0,
                n_all ? nll_all / n_all : 0.0);
    return 0;
}

// --calibrate-prefill OUT: the prompt path's settings measured on THIS PC - the chunk a forward takes (up to the one the
// load sized its buffers for) and the dense GEMMs' backend (the BLAS, or hipBLASLt with each shape's kernel measured,
// on AMD) - over a prompt of `ids` (--ids, a real text's tokens; repeated to the length) from position 0, each setting
// twice and its faster run taken.  The GEMMs first at the largest chunk, then the chunks with the faster GEMMs; a
// chunk above the default only where every routed expert is in VRAM at the largest one (else a larger chunk's
// buffers would take room from the experts).  The fastest setting is kept for this PC in the pack's
// prefill-tune.txt (the next starts take it) unless it beats the defaults by less than kKeepGain; OUT gets the
// measurements (JSON) for the setup.  Where not every expert is in VRAM the larger chunks are measured too (after the
// rest: what is kept does not change) with the decode of this load's placement - each chunk stages the experts VRAM
// does not hold, so there a larger one reads prompts much faster (#552: 4096 2.2x 1024's on one 48 GB GPU) - and
// the setup (tools/calibrate_mm3_prefill.py) keeps one when a start with the default chunk does not decode faster.
// decode_rate: tokens decoded one by one after the prompt's (whose logits are in lg), the faster of two runs
// (-1: an error, 0: no room in the context)
constexpr int kDecodeTokens = 32;
double decode_rate(mm3::Model& m, int N, std::vector<float>& lg) {
    if (N + kDecodeTokens >= m.max_ctx()) return 0.0;
    const auto argmax = [&] { return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin()); };
    const int tok0 = argmax();
    std::string err;
    double best = 0;
    for (int r = 0; r < 2; ++r) {
        int tok = tok0;
        const double t0 = now();
        for (int i = 0; i < kDecodeTokens; ++i) {
            if (!m.forward(&tok, 1, N + i, 1, lg.data(), err)) {
                std::fprintf(stderr, "maya-mm3: calibration: %s\n", err.c_str());
                return -1.0;
            }
            tok = argmax();
        }
        best = std::max(best, (double) kDecodeTokens / (now() - t0));
    }
    std::fprintf(stderr, "calibration: decode %.2f tok/s (%ld of %ld experts in VRAM)\n", best,
                 (long) m.stats().resident, (long) m.stats().experts);
    return best;
}

// the calibration's prompt: `ids` (a real text's tokens) repeated to N - the calibration's 4096 or its largest chunk -
// or, without them, a fixed pseudo-random text
int calib_prompt(const mm3::Model& m, std::vector<int>& ids) {
    const int N = std::max(1, std::min(m.max_ctx() - 1, std::max(4096, m.chunk_max())));
    if (ids.empty() || (ids.size() == 1 && ids[0] == 0)) {   // (no text: a fixed pseudo-random one)
        ids.clear();
        unsigned x = 12345u;
        for (int i = 0; i < N; ++i) {
            x = x * 1664525u + 1013904223u;
            ids.push_back(1000 + (int) ((x >> 8) % (unsigned) std::max(1, m.vocab() - 2000)));
        }
    }
    for (size_t i = 0; ids.size() < (size_t) N; ++i) ids.push_back(ids[i]);
    ids.resize((size_t) N);
    return N;
}

// --calibrate-decode OUT (the setup's decode check, tools/calibrate_mm3_prefill.py): this load's decode - its placement
// beside the buffers of its --chunk - after the calibration's prompt: {"decode_tok_s", "resident", "experts"}
int calibrate_decode(mm3::Model& m, std::vector<int> ids, const std::string& out_path) {
    const int N = calib_prompt(m, ids);
    std::vector<float> lg((size_t) m.vocab());
    std::string err;
    if (!m.forward(ids.data(), N, 0, 1, lg.data(), err)) {
        std::fprintf(stderr, "maya-mm3: calibration: %s\n", err.c_str());
        return 1;
    }
    const double dec = decode_rate(m, N, lg);
    if (dec < 0) return 1;
    char buf[256];
    std::snprintf(buf, sizeof buf, "{\"decode_tok_s\": %.2f, \"chunk\": %d, \"resident\": %ld, \"experts\": %ld}\n", dec,
                  m.chunk_max(), (long) m.stats().resident, (long) m.stats().experts);
    std::ofstream f(out_path, std::ios::binary | std::ios::trunc);
    f << buf;
    if (!f) {
        std::fprintf(stderr, "%s: cannot write\n", out_path.c_str());
        return 1;
    }
    return 0;
}

int calibrate_prefill(mm3::Model& m, std::vector<int> ids, const std::string& out_path) {
    constexpr double kKeepGain = 0.02;
    constexpr int kDefChunk = 1024;
    const double t_start = now();
    const auto& st = m.stats();
    const int cmax = m.chunk_max();
    const bool all_res = st.resident >= st.experts;
    const int N = calib_prompt(m, ids);
    std::vector<float> lg((size_t) m.vocab());
    std::string err;
    struct Trial {
        int chunk, gemm;
        double tok_s;
    };
    std::vector<Trial> trials;
    auto measure = [&](int c, int gm) -> double {
        for (const Trial& t : trials)
            if (t.chunk == c && t.gemm == gm) return t.tok_s;
        m.set_chunk(c);
        m.set_prefill_gemm(gm);
        double best = 0;
        for (int r = 0; r < 2; ++r) {   // (the first also measures hipBLASLt's kernels for its shapes)
            const double t0 = now();
            if (!m.forward(ids.data(), N, 0, 1, lg.data(), err)) {
                std::fprintf(stderr, "maya-mm3: calibration: %s\n", err.c_str());
                return -1;
            }
            best = std::max(best, (double) N / (now() - t0));
        }
        trials.push_back({c, gm, best});
        std::fprintf(stderr, "calibration: chunk %d, %s GEMMs: %.1f tok/s\n", c, gm == 1 ? "hipBLASLt" : "BLAS", best);
        return best;
    };
    std::vector<int> chunks, wide;   // (wide: the larger ones where not every expert is in VRAM - measured last)
    for (int c : {1024, 2048, 4096})
        if (c <= cmax) (all_res || c <= kDefChunk ? chunks : wide).push_back(c);
    if (chunks.empty()) chunks.push_back(cmax);
    const int top = chunks.back();
    std::vector<int> gemms{0};
    if (m.lt_available()) gemms.push_back(1);
    int bg = 0;
    double bt = -1;
    for (int gm : gemms) {
        const double r = measure(top, gm);
        if (r < 0) return 1;
        if (r > bt) bt = r, bg = gm;
    }
    int bc = top;
    for (int c : chunks) {
        const double r = measure(c, bg);
        if (r < 0) return 1;
        if (r > bt) bt = r, bc = c;
    }
    const int dc = std::min(kDefChunk, cmax);
    const double def = measure(dc, 0);
    if (def < 0) return 1;
    if (bt < def * (1.0 + kKeepGain)) bc = dc, bg = 0, bt = def;   // (the defaults, unless the best is clearly faster)
    int wc = 0;
    double wt = 0, dec = 0;
    for (int c : wide) {   // (with the kept GEMMs; the prompt's logits of the last one stay in lg for the decode)
        const double r = measure(c, bg);
        if (r < 0) return 1;
        if (r > wt) wt = r, wc = c;
    }
    if (!wide.empty()) {
        dec = decode_rate(m, N, lg);
        if (dec < 0) return 1;
    }
    char set[256];
    std::snprintf(set, sizeof set, "chunk=%d gemm=%d tok_s=%.1f default_tok_s=%.1f tokens=%d", bc, bg, bt, def, N);
    if (!m.write_tune(set, err)) {
        std::fprintf(stderr, "maya-mm3: calibration: %s\n", err.c_str());
        return 1;
    }
    std::string js = "{\"key\": \"" + m.tune_key() + "\", \"file\": \"";
    for (char ch : m.tune_file()) js += ch == '\\' ? std::string("\\\\") : std::string(1, ch);
    char buf[512], wbuf[256] = "";
    if (wc > 0)   // (the setup's decode check: this load's decode - its buffers the largest chunk's)
        std::snprintf(wbuf, sizeof wbuf, "\"wide_chunk\": %d, \"wide_tok_s\": %.1f, \"decode_tok_s\": %.2f, "
                      "\"resident\": %ld, \"experts\": %ld, ", wc, wt, dec, (long) st.resident, (long) st.experts);
    std::snprintf(buf, sizeof buf, "\", \"chunk\": %d, \"gemm\": \"%s\", \"tok_s\": %.1f, \"default_tok_s\": %.1f, "
                  "\"tokens\": %d, \"all_resident\": %s, \"lt_available\": %s, \"seconds\": %.0f, %s\"trials\": [",
                  bc, bg == 1 ? "hipblaslt" : "blas", bt, def, N, all_res ? "true" : "false",
                  m.lt_available() ? "true" : "false", now() - t_start, wbuf);
    js += buf;
    for (size_t i = 0; i < trials.size(); ++i) {
        std::snprintf(buf, sizeof buf, "%s{\"chunk\": %d, \"gemm\": \"%s\", \"tok_s\": %.1f}", i ? ", " : "",
                      trials[i].chunk, trials[i].gemm == 1 ? "hipblaslt" : "blas", trials[i].tok_s);
        js += buf;
    }
    js += "]}\n";
    std::fprintf(stderr, "calibration: chunk %d, %s GEMMs: %.1f tok/s (the defaults %.1f) - kept for this PC in %s\n", bc,
                 bg == 1 ? "hipBLASLt" : "BLAS", bt, def, m.tune_file().c_str());
    if (!out_path.empty() && out_path != "-") {
        std::ofstream f(out_path, std::ios::binary | std::ios::trunc);
        f << js;
        if (!f) {
            std::fprintf(stderr, "%s: cannot write\n", out_path.c_str());
            return 1;
        }
    } else {
        std::printf("%s", js.c_str());
    }
    return 0;
}

int usage() {
    std::fprintf(stderr,
                 "usage: maya-mm3 --pack DIR --ids IDS|@FILE [--gen N] [--dump-logits FILE --last N] [--gpus LIST]\n"
                 "                [--max-context N] [--chunk N] [--staging N] [--resident-cap N] [--reserve-gb X]\n"
                 "                [--vision on|off] [--vision-gpu I] [--images FILE] [--layer-split auto|L,L,...]\n"
                 "                [--tp 1|2|4] [--cache-gb X] [--kv-cache F|K,V (bf16 int8 fp8 fp4; default int8)]\n"
                 "                [--kv-rot on|off] [--kv-streaming on|off] [--expert-profile FILE]\n"
                 "                [--score REF [--write-ref OUT]] [--cpu-threads N] [--dspark on|off]\n"
                 "                [--kv-resident N] [--tp-chunk N] [--tp-ffn-rows N] [--dspark-cost X]\n"
                 "                [--prefill-gemm blas|hipblaslt|auto] [--prefill-tune FILE] [--no-prefill-tune]\n"
                 "                [--calibrate-prefill OUT.json [--ids @prompt]] [--calibrate-decode OUT.json] [--verbose]\n"
                 "                [--serve [--parallel N (1-8: conversations at once)]] [--parallel N --rows-check G]\n"
                 "       maya-mm3 --pack DIR --encode-image FILE.rgb --size WxH --dump-image OUT [--gpus N]\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    // cuBLAS gives the same bits run after run only with its workspace pinned (its notes on reproducibility): before
    // any handle; a setting the caller made stays
    if (std::getenv("CUBLAS_WORKSPACE_CONFIG") == nullptr) {
#if defined(_WIN32)
        _putenv_s("CUBLAS_WORKSPACE_CONFIG", ":4096:8");
#else
        setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 0);
#endif
    }
#if defined(_WIN32)
    // Windows sleeps in 15.6 ms ticks unless a process asks for finer ones (the CPU lane's idle workers, the waits'
    // naps).  MAYA_MM3_TIMER_MS=0: leave the system's resolution alone
    {
        const char* v = std::getenv("MAYA_MM3_TIMER_MS");
        const int ms = v != nullptr ? std::atoi(v) : 1;
        if (ms > 0) timeBeginPeriod((UINT) ms);
    }
#endif
    mm3::Options o;
    std::string ids_arg, dump, profile_out, batch, eos_arg, images, enc_rgb, enc_size, enc_out, score_path, ref_out, learned;
    std::string calib_out, calib_dec;
    bool serve_mode = false;
    int kv_stream = -1;   // (--kv-streaming: -1 not given)
    int gen = 0, last = 1, ppl = 0, rows_steps = 0;
    float dspark_conf = 0.7f;   // the drafts' confidence floor: a round verifies the drafts before the first below it
    float dspark_cost = 0.0f;   // (> 0) ... and those before the first whose confidences' product is below it
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s: needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--pack") o.pack = val();
        else if (a == "--ids") ids_arg = val();
        else if (a == "--gen") gen = std::atoi(val().c_str());
        else if (a == "--dump-logits") dump = val();
        else if (a == "--last") last = std::atoi(val().c_str());
        else if (a == "--gpus") o.gpus = parse_list(val());
        else if (a == "--max-context") o.max_ctx = std::atoi(val().c_str());
        else if (a == "--chunk") {
            o.chunk = std::atoi(val().c_str());
            o.chunk_given = true;
        }
        else if (a == "--prefill-gemm") {   // (the prefill calibration's choice by default: auto)
            const std::string v = val();
            o.prefill_gemm = v == "auto" ? -1 : v == "hipblaslt" || v == "lt" ? 1 : v == "blas" ? 0 : -2;
            if (o.prefill_gemm == -2) return usage();
        }
        else if (a == "--prefill-tune") o.prefill_tune = val();
        else if (a == "--no-prefill-tune") o.tune = false;
        else if (a == "--calibrate-prefill") calib_out = val();
        else if (a == "--calibrate-decode") calib_dec = val();
        else if (a == "--staging") o.staging = std::atoi(val().c_str());
        else if (a == "--resident-cap") o.resident_cap = std::atoi(val().c_str());
        else if (a == "--reserve-gb") o.reserve_gb = std::atof(val().c_str());
        else if (a == "--profile") o.profile = val();
        else if (a == "--profile-out") profile_out = val();
        else if (a == "--no-pin-pool") o.pin_pool = false;
        else if (a == "--no-fast-decode") o.fast_decode = false;
        else if (a == "--no-promote") o.promote = false;
        else if (a == "--promote-lag") o.promote_lag = std::max(1, std::atoi(val().c_str()));
        else if (a == "--ppl") ppl = std::atoi(val().c_str());
        else if (a == "--batch") batch = val();
        else if (a == "--cpu-threads")   // (the lane's threads, the caller included: its workers one fewer - #552)
            o.cpu_threads = mm3::lane_workers_of_threads(std::atoi(val().c_str()));
        else if (a == "--dspark") {
            const std::string v = val();
            o.dspark = v != "off" && v != "none";
        }
        else if (a == "--expert-profile") learned = val();
        else if (a == "--vision") {   // (on by default when the pack has the tower; --vision on: it must)
            o.vision = val() != "off";
            o.vision_auto = false;
        }
        else if (a == "--tp") o.tp = std::atoi(val().c_str());
        else if (a == "--cache-gb") o.cache_gb = std::atof(val().c_str());
        else if (a == "--tp-chunk") o.tp_chunk = std::atoi(val().c_str());
        else if (a == "--tp-ffn-rows") o.tp_ffn_rows = std::atoi(val().c_str());
        else if (a == "--kv-streaming") {   // (on: --kv-resident 32768 unless that is given; the context above it)
            const std::string v = val();
            if (v != "on" && v != "off") return usage();
            kv_stream = v == "on" ? 1 : 0;
        } else if (a == "--kv-resident") o.kv_resident = std::max(0, std::atoi(val().c_str()));
        else if (a == "--kv-cache") {   // (one format for K and V, "K,V", or kXvY: k8v4, k16v4, ki8vi8)
            std::string v = val(), kf = v, vf = v;
            if (const size_t c = v.find(','); c != std::string::npos) {
                kf = v.substr(0, c);
                vf = v.substr(c + 1);
            } else if (v.size() > 2 && v[0] == 'k' && v.find('v') != std::string::npos) {
                kf = v.substr(1, v.find('v') - 1);
                vf = v.substr(v.find('v') + 1);
            }
            const int K = mm3::k::kv_parse(kf.c_str()), V = mm3::k::kv_parse(vf.c_str());
            if (K < 0 || V < 0) return usage();
            o.kv_k = K;
            o.kv_v = V;
        } else if (a == "--kv-rot" || a == "--kv-rotate") {   // (#490's name too)
            const std::string v = val();
            if (v != "on" && v != "off") return usage();
            o.kv_rot = v == "on";
        }
        else if (a == "--layer-split") {   // (serve/server.py's: "auto", or the first layer of each later GPU's share)
            const std::string v = val();
            o.layer_split = v == "auto" ? std::vector<int>{} : parse_list(v);
        }
        else if (a == "--vision-gpu") o.vision_gpu = std::atoi(val().c_str());
        else if (a == "--images") images = val();
        else if (a == "--encode-image") enc_rgb = val();
        else if (a == "--size") enc_size = val();
        else if (a == "--dump-image") enc_out = val();
        else if (a == "--dspark-conf") dspark_conf = (float) std::atof(val().c_str());
        else if (a == "--dspark-cost") dspark_cost = (float) std::atof(val().c_str());
        else if (a == "--score") score_path = val();
        else if (a == "--write-ref") ref_out = val();
        else if (a == "--serve") serve_mode = true;
        else if (a == "--parallel") o.seqs = std::atoi(val().c_str());   // (--serve: conversations at once, 1-8)
        else if (a == "--rows-check") rows_steps = std::atoi(val().c_str());
        else if (a == "--eos") eos_arg = val();
        else if (a == "--verbose") o.verbose = true;
        else return usage();
    }
    if (!enc_rgb.empty()) {   // the vision tower alone: a picture's rows into --dump-image
        int w = 0, h = 0;
        if (o.pack.empty() || enc_out.empty() || std::sscanf(enc_size.c_str(), "%dx%d", &w, &h) != 2) return usage();
        mm3::Pack pk;
        mm3::Vision v;
        std::string err;
        std::ifstream f(enc_rgb, std::ios::binary);
        std::vector<uint8_t> px((size_t) w * h * 3);
        if (!f.read((char*) px.data(), (std::streamsize) px.size())) {
            std::fprintf(stderr, "%s: not %d x %d x 3 bytes\n", enc_rgb.c_str(), w, h);
            return 1;
        }
        if (!pk.load(o.pack, err) || !mm3::Vision::in_pack(pk) || !v.load(pk, o.gpus.empty() ? 0 : o.gpus[0], err)) {
            std::fprintf(stderr, "maya-mm3: %s\n", err.empty() ? "the pack has no vision tower" : err.c_str());
            return 1;
        }
        void* lent = nullptr;   // (MAYA_MM3_VISION_LEND=1: the lender's path tested - a room lent, given back)
        if (const char* fl = mm3::env("MM3_VISION_LEND"); fl && std::atoi(fl) == 1)
            v.set_lender([&lent](size_t b) { return cudaMalloc(&lent, b) == cudaSuccess ? lent : nullptr; },
                         [&lent]() {
                             cudaFree(lent);
                             lent = nullptr;
                             std::fprintf(stderr, "vision: the lent room given back\n");
                         });
        std::vector<float> rows;
        size_t fr0 = 0, fr1 = 0, tot = 0;
        double first_ms = 0;
        cudaSetDevice(v.device());
        cudaMemGetInfo(&fr0, &tot);
        for (int rep = 0; rep < 2; ++rep) {   // (the second one timed)
            const double t0 = now();
            if (!v.encode(px.data(), w, h, rows, err)) {
                std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
                return 1;
            }
            if (rep == 1)
                std::fprintf(stderr, "vision: %d x %d pixels -> %zu rows in %.0f ms (%zu MB of CUDA%d during it, the "
                             "first %.0f ms)\n", w, h, rows.size() / (size_t) v.geo().out_dim, 1e3 * (now() - t0),
                             v.last_bytes() >> 20, v.device(), 1e3 * first_ms);
            else
                first_ms = now() - t0;
        }
        cudaMemGetInfo(&fr1, &tot);
        std::fprintf(stderr, "vision: CUDA%d free %zu MB before, %zu MB after (the tower in VRAM between pictures: %zu "
                     "MB)\n", v.device(), fr0 >> 20, fr1 >> 20, v.bytes() >> 20);
        std::FILE* fo = std::fopen(enc_out.c_str(), "wb");
        if (!fo || std::fwrite(rows.data(), 4, rows.size(), fo) != rows.size()) {
            std::fprintf(stderr, "%s: cannot write\n", enc_out.c_str());
            return 1;
        }
        std::fclose(fo);
        return 0;
    }
    if (o.pack.empty() || (ids_arg.empty() && batch.empty() && !serve_mode && score_path.empty() && calib_out.empty() &&
                           calib_dec.empty()))
        return usage();
    if (!calib_out.empty()) {   // (the calibration: buffers for its largest chunk, measured from scratch - no cache)
        if (!o.chunk_given) o.chunk = 4096;
        o.chunk_given = true;
        o.tune = false;
        if (o.prefill_gemm < 0) o.prefill_gemm = 0;
    } else if (!calib_dec.empty()) {   // (its decode check: the --chunk given, not a calibration's)
        o.tune = false;
        if (o.prefill_gemm < 0) o.prefill_gemm = 0;
    }
    if (kv_stream == 0) o.kv_resident = 0;
    else if (kv_stream == 1 && o.kv_resident == 0) o.kv_resident = 32768;
    if (!images.empty()) {
        o.vision = true;
        o.vision_auto = false;
        o.vision_must = true;
    }
    if (!learned.empty()) {   // (this PC's own routing: read when it exists, else the shipped --profile; saved)
        if (std::FILE* f = std::fopen(learned.c_str(), "rb")) {
            std::fclose(f);
            o.profile = learned;
        }
        if (profile_out.empty()) profile_out = learned;
    }
    if (o.cpu_threads == -2) o.cpu_threads = mm3::CpuLane::auto_workers();   // (no --cpu-threads: this host's cores)
    std::vector<int> ids = ids_arg.empty() ? std::vector<int>{0} : parse_ids(ids_arg);
    if (ids.empty()) {
        std::fprintf(stderr, "no token ids\n");
        return 2;
    }
    mm3::Model m;
    std::string err;
    if (!m.load(o, err)) {
        std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
        return 1;
    }
    if (!calib_out.empty()) {
        for (const auto& l : m.info_lines()) std::fprintf(stderr, "  %s\n", l.c_str());
        return calibrate_prefill(m, ids_arg.empty() ? std::vector<int>{} : ids, calib_out);
    }
    if (!calib_dec.empty()) {
        for (const auto& l : m.info_lines()) std::fprintf(stderr, "  %s\n", l.c_str());
        return calibrate_decode(m, ids_arg.empty() ? std::vector<int>{} : ids, calib_dec);
    }
    if (rows_steps > 0) return rows_check(m, ids, rows_steps);
    if (!score_path.empty()) {
        for (const auto& l : m.info_lines()) std::fprintf(stderr, "  %s\n", l.c_str());
        return score_main(m, score_path, ref_out);
    }
    if (serve_mode) {
        std::vector<int> eos = eos_arg.empty() ? std::vector<int>{m.eos()} : parse_list(eos_arg);
        const std::string kvf = o.kv_k == o.kv_v ? std::string(mm3::k::kv_name(o.kv_k))
                                                 : std::string(mm3::k::kv_name(o.kv_k)) + "," + mm3::k::kv_name(o.kv_v);
        if (m.seqs() > 1) return serve_multi(m, eos, o.chunk, kvf, profile_out);
        return serve(m, eos, o.chunk, o.dspark, dspark_conf, dspark_cost, kvf, profile_out);
    }
    const auto& st = m.stats();
    std::fprintf(stderr, "maya-mm3: loaded in %.1f s, %ld of %ld experts in VRAM, %.1f GB pinned pool\n", st.load_s,
                 st.resident, st.experts, st.pinned_gb);
    for (const auto& l : m.info_lines()) std::fprintf(stderr, "  %s\n", l.c_str());
    if (!batch.empty()) {   // one prompt a line (ids), each from position 0: the expert profile's corpus
        std::ifstream bf(batch);
        int k = 0;
        long tok = 0;
        const double b0 = now();
        for (std::string line; std::getline(bf, line);) {
            std::vector<int> b = parse_ids(line);
            if (b.empty()) continue;
            if ((int) b.size() > m.max_ctx()) b.resize((size_t) m.max_ctx());
            if (!m.forward(b.data(), (int) b.size(), 0, 0, nullptr, err)) {
                std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
                return 1;
            }
            tok += (long) b.size();
            std::fprintf(stderr, "batch %d: %zu tokens\n", ++k, b.size());
        }
        std::fprintf(stderr, "batch: %d prompts, %ld tokens in %.1f s\n", k, tok, now() - b0);
        if (!profile_out.empty()) {
            if (!m.save_profile(profile_out, err)) {
                std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
                return 1;
            }
            std::fprintf(stderr, "wrote the expert profile %s (%ld routings)\n", profile_out.c_str(),
                         m.stats().profiled);
        }
        if (ids_arg.empty()) return 0;
    }
    const int n = (int) ids.size();
    if (ppl > 0 && o.tp > 0) {   // (the tensor-parallel path: one prompt, the NLL taken chunk by chunk)
        if (!m.forward_ppl(ids.data(), n, err)) {
            std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
            return 1;
        }
        const std::vector<double>& nl = m.ppl_nll();
        double sum = 0, bsum = 0;
        int bn = 0, cnt = 0;
        for (int p = 0; p + 1 < n && p < (int) nl.size(); ++p) {
            sum += nl[(size_t) p];
            bsum += nl[(size_t) p];
            ++cnt;
            if (++bn == ppl) {
                std::printf("positions %6d-%6d: NLL %.4f  ppl %.3f\n", p + 1 - ppl, p, bsum / bn, std::exp(bsum / bn));
                bsum = 0;
                bn = 0;
            }
        }
        if (bn) std::printf("positions %6d-%6d: NLL %.4f  ppl %.3f\n", n - 1 - bn, n - 2, bsum / bn, std::exp(bsum / bn));
        std::printf("all %d positions: NLL %.4f  ppl %.3f\n", cnt, sum / cnt, std::exp(sum / cnt));
        return 0;
    }
    if (ppl > 0) {   // the next-token NLL of every position, by buckets of `ppl` positions
        const int C = o.chunk;
        std::vector<float> lg((size_t) C * m.vocab());
        double sum = 0, bsum = 0;
        int bn = 0, cnt = 0;
        for (int a0 = 0; a0 < n - 1; a0 += C) {
            const int T = std::min(C, n - a0);
            if (!m.forward(ids.data() + a0, T, a0, T, lg.data(), err)) {
                std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
                return 1;
            }
            for (int t = 0; t < T && a0 + t + 1 < n; ++t) {
                const float* r = lg.data() + (size_t) t * m.vocab();
                double mx = r[0];
                for (int i = 1; i < m.vocab(); ++i) mx = std::max(mx, (double) r[i]);
                double se = 0;
                for (int i = 0; i < m.vocab(); ++i) se += std::exp((double) r[i] - mx);
                const double nll = mx + std::log(se) - r[ids[(size_t) a0 + t + 1]];
                sum += nll;
                bsum += nll;
                ++cnt;
                if (++bn == ppl) {
                    std::printf("positions %6d-%6d: NLL %.4f  ppl %.3f\n", a0 + t + 1 - ppl, a0 + t, bsum / bn,
                                std::exp(bsum / bn));
                    bsum = 0;
                    bn = 0;
                }
            }
        }
        if (bn) std::printf("positions %6d-%6d: NLL %.4f  ppl %.3f\n", n - 1 - bn, n - 2, bsum / bn, std::exp(bsum / bn));
        std::printf("all %d positions: NLL %.4f  ppl %.3f\n", cnt, sum / cnt, std::exp(sum / cnt));
        return 0;
    }
    if (!images.empty()) {   // --images: an MMI1 file's pictures, for the prompt's spans
        std::vector<Pic> pics;
        std::vector<mm3::Model::Image> rows;
        for (const std::string& l : m.vram_lines()) std::fprintf(stderr, "before the pictures: %s\n", l.c_str());
        const double t0 = now();
        if (!read_pics(images, pics, err) || !picture_rows(m, pics, ids, 0, rows, err)) {
            std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "pictures: %zu encoded in %.0f ms\n", pics.size(), 1e3 * (now() - t0));
        for (const std::string& l : m.vram_lines()) std::fprintf(stderr, "after the pictures: %s\n", l.c_str());
        m.set_images(std::move(rows));
    }
    last = std::max(1, std::min(last, n));
    std::vector<float> logits((size_t) last * m.vocab());
    double t0 = now();
    if (!m.forward(ids.data(), n, 0, last, logits.data(), err)) {
        std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
        return 1;
    }
    double t1 = now();
    m.set_images({});
    std::fprintf(stderr, "prompt: %d tokens in %.2f s (%.1f tok/s), %ld experts staged\n", n, t1 - t0,
                 n / (t1 - t0), m.stats().staged);
    if (!dump.empty()) {
        std::FILE* f = std::fopen(dump.c_str(), "wb");
        if (!f) {
            std::fprintf(stderr, "%s: cannot write\n", dump.c_str());
            return 1;
        }
        const int hdr[3] = {last, m.vocab(), n};
        std::fwrite("MM3L", 1, 4, f);
        std::fwrite(hdr, 4, 3, f);
        std::fwrite(ids.data(), 4, ids.size(), f);
        for (int r = 0; r < last; ++r) {
            const int p = n - last + r;
            std::fwrite(&p, 4, 1, f);
        }
        std::fwrite(logits.data(), 4, logits.size(), f);
        std::fclose(f);
        std::fprintf(stderr, "wrote %s (%d positions)\n", dump.c_str(), last);
    }
    auto argmax = [&](const float* row) {
        int b = 0;
        for (int i = 1; i < m.vocab(); ++i)
            if (row[i] > row[b]) b = i;
        return b;
    };
    int next = argmax(logits.data() + (size_t) (last - 1) * m.vocab());
    std::printf("next %d\n", next);
    // (debugging, MM3_STRESS=N [MM3_STRESS_T=T]: one verify window's forward N more times - its logits may not change)
    if (const char* sv = ::mm3::env("MM3_STRESS"); sv && m.draft_block() > 0) {
        const int iters = std::atoi(sv);
        const int tw = ::mm3::env("MM3_STRESS_T") ? std::atoi(::mm3::env("MM3_STRESS_T")) : 4;
        std::vector<int> drafts;
        std::vector<float> conf;
        if (!m.draft(next, n, m.draft_block(), drafts, conf, err)) return 1;
        std::vector<int> win(1, next);
        for (int i = 0; i + 1 < tw && i < (int) drafts.size(); ++i) win.push_back(drafts[(size_t) i]);
        const size_t W = win.size() * (size_t) m.vocab();
        std::vector<float> ref(W), cur(W);
        int bad = 0;
        for (int it = 0; it <= iters; ++it) {
            if (!m.forward(win.data(), (int) win.size(), n, (int) win.size(), it == 0 ? ref.data() : cur.data(), err)) {
                std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
                return 1;
            }
            if (it == 0) continue;
            if (std::memcmp(ref.data(), cur.data(), W * 4) != 0) {
                ++bad;
                double mx = 0;
                std::string rows;
                for (size_t r = 0; r < win.size(); ++r) {
                    bool d = false;
                    for (int v = 0; v < m.vocab(); ++v) {
                        const double e = std::fabs((double) ref[r * m.vocab() + v] - cur[r * m.vocab() + v]);
                        if (e > 0) d = true;
                        mx = std::max(mx, e);
                    }
                    if (d) rows += " " + std::to_string(r);
                }
                std::fprintf(stderr, "STRESS iteration %d differs: rows%s, max |diff| %.3g\n", it, rows.c_str(), mx);
            }
        }
        std::fprintf(stderr, "STRESS %d forwards of %zu tokens at %d: %d differ\n", iters, win.size(), n, bad);
        return 0;
    }
    if (gen > 0 && m.draft_block() > 0) {   // greedy decode in DSpark rounds: draft, verify in one forward, keep
        const int B = m.draft_block();
        std::vector<float> lg((size_t) (B + 1) * m.vocab());
        std::vector<int> out, drafts;
        std::vector<float> conf;
        t0 = now();
        int pos = n, rounds = 0, offered = 0, accepted = 0, drafted = 0;
        double t_draft = 0, t_verify = 0;
        const auto s0 = m.stats();
        while ((int) out.size() < gen) {
            out.push_back(next);
            if (next == m.eos()) break;
            if (const char* ns = ::mm3::env("MM3_NSYS")) {   // (profiling: rounds 20 .. 20 + N under nsys's capture)
                if (rounds == 20) cudaProfilerStart();
                if (rounds == 20 + std::atoi(ns)) cudaProfilerStop();
            }
            const double td = now();
            int k = 0;
            if (!m.draft(next, pos, B, drafts, conf, err)) {
                std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
                return 1;
            }
            while (k < (int) drafts.size() && conf[(size_t) k] >= dspark_conf) ++k;
            ++drafted;
            t_draft += now() - td;
            if (::mm3::env("MM3_CONF") && k > 0) {   // (debugging: each round's confidences)
                std::fprintf(stderr, "conf");
                for (float c : conf) std::fprintf(stderr, " %.3f", c);
                std::fprintf(stderr, " -> %d\n", k);
            }
            std::vector<int> win(1, next);
            win.insert(win.end(), drafts.begin(), drafts.begin() + k);
            const double tv = now();
            if (!m.forward(win.data(), (int) win.size(), pos, (int) win.size(), lg.data(), err)) {
                std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
                return 1;
            }
            t_verify += now() - tv;
            int j = 0;
            for (; j < k; ++j) {
                if (drafts[(size_t) j] != argmax(lg.data() + (size_t) j * m.vocab())) break;
                out.push_back(drafts[(size_t) j]);
                if (drafts[(size_t) j] == m.eos()) break;
            }
            ++rounds;
            offered += k;
            accepted += j;
            if (j < k && drafts[(size_t) j] == m.eos()) break;
            if (j > 0 && out.back() == m.eos()) break;
            next = argmax(lg.data() + (size_t) j * m.vocab());
            pos += j + 1;
        }
        t1 = now();
        for (size_t i = 0; i < out.size(); ++i) std::printf("%d%s", out[i], i + 1 < out.size() ? " " : "\n");
        std::fprintf(stderr, "decode (DSpark): %zu tokens in %.2f s (%.2f tok/s), %d rounds, %d of %d drafts kept "
                     "(%.2f a round), drafting %.1f ms a draft (%d drafts)\n", out.size(), t1 - t0, out.size() / (t1 - t0),
                     rounds, accepted, offered, rounds ? (double) accepted / rounds : 0.0,
                     drafted ? 1e3 * t_draft / drafted : 0.0, drafted);
        if (rounds) {
            const auto& s1 = m.stats();
            std::fprintf(stderr, "  experts promoted into staging slots: %ld (their copies waited for %.1f ms a round)\n",
                         s1.promoted - s0.promoted, 1e3 * (s1.t_promo - s0.t_promo) / rounds);
            std::fprintf(stderr, "  a round: verify %.1f ms (routing waits %.1f, CPU lane %.1f, %.1f experts on the CPU, "
                         "%.1f staged)\n", 1e3 * t_verify / rounds, 1e3 * (s1.t_sync - s0.t_sync) / rounds,
                         1e3 * (s1.t_lane - s0.t_lane) / rounds, (double) (s1.cpu_experts - s0.cpu_experts) / rounds,
                         (double) (s1.staged - s0.staged) / rounds);
        }
        gen = 0;
    }
    if (gen > 0) {
        std::vector<float> lg((size_t) m.vocab());
        std::string out;
        t0 = now();
        int pos = n, done = 0;
        for (int g = 0; g < gen; ++g) {
            std::printf("%d%s", next, g + 1 < gen ? " " : "\n");
            std::fflush(stdout);
            if (next == m.eos()) {
                std::printf("\n");
                break;
            }
            if (const char* ns = ::mm3::env("MM3_NSYS")) {   // (profiling: tokens 8 .. 8 + N under nsys's capture)
                if (g == 8) cudaProfilerStart();
                if (g == 8 + std::atoi(ns)) cudaProfilerStop();
            }
            if (!m.forward(&next, 1, pos++, 1, lg.data(), err)) {
                std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
                return 1;
            }
            ++done;
            next = argmax(lg.data());
        }
        t1 = now();
        std::fprintf(stderr, "decode: %d tokens in %.2f s (%.2f tok/s), %ld experts staged in all, %ld on the CPU\n",
                     done, t1 - t0, done / (t1 - t0), m.stats().staged, m.stats().cpu_experts);
        if (::mm3::env("MM3_TIMING") && done > 0) {
            const auto& s2 = m.stats();
            std::fprintf(stderr, "  a token: %.1f ms - routing waits %.1f ms, CPU lane %.1f ms, promotions' copies %.1f ms "
                         "(%ld promoted)\n", 1e3 * s2.t_token / done, 1e3 * s2.t_sync / done, 1e3 * s2.t_lane / done,
                         1e3 * s2.t_promo / done, s2.promoted);
        }
    }
    if (!profile_out.empty()) {
        if (!m.save_profile(profile_out, err)) {
            std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "wrote the expert profile %s (%ld routings)\n", profile_out.c_str(), m.stats().profiled);
    }
    return 0;
}
