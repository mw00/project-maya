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
#include <fstream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <unistd.h>

#include <cuda_profiler_api.h>

#include "mm3_model.hpp"
#include "mm3_pack.hpp"
#include "mm3_vision.hpp"

namespace {

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
    void start() {
        std::thread([this] {
            std::string buf;
            char chunk[4096];
            for (;;) {
                const ssize_t n = ::read(0, chunk, sizeof chunk);
                if (n < 0 && errno == EINTR) continue;
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

struct Req {
    long long max_new = 0;
    float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 64;
    unsigned long long seed = 0;
    int pen_n = 0;
    float pen_repeat = 1.0f, pen_freq = 0.0f, pen_present = 0.0f;
    long long think_budget = 0;
    int think_end = -1;
    int spec = -1;          // spec=0|1: this request drafts or not (-1: as the engine was started)
    float conf = -1.0f;     // dspark_conf=F: the drafts' confidence floor for this request (-1: the engine's)
    std::string file;       // GENI: the pictures
    std::vector<int> ids;
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
        // (keys this engine does not use are taken and ignored)
    }
    return !r.ids.empty();
}

// the next token's distribution from one row of logits: penalties over the history's last pen_n tokens (llama.cpp's),
// then greedy (temperature <= 0: the argmax alone) or temperature, top_k, top_p, min_p - (token, probability) pairs,
// the most likely first, summing to 1
std::vector<std::pair<int, double>> dist(const float* row, int V, const Req& r, const std::vector<int>& hist) {
    std::vector<float> lg(row, row + V);
    if (r.pen_n != 0 && (r.pen_repeat != 1.0f || r.pen_freq != 0.0f || r.pen_present != 0.0f)) {
        std::unordered_map<int, int> cnt;
        const size_t n = r.pen_n < 0 ? hist.size() : std::min(hist.size(), (size_t) r.pen_n);
        for (size_t i = hist.size() - n; i < hist.size(); ++i) ++cnt[hist[i]];
        for (const auto& [t, c] : cnt) {
            if (t < 0 || t >= V) continue;
            float& x = lg[(size_t) t];
            if (r.pen_repeat > 0.0f && r.pen_repeat != 1.0f) x = x > 0 ? x / r.pen_repeat : x * r.pen_repeat;
            x -= r.pen_freq * (float) c + r.pen_present;
        }
    }
    if (!(r.temperature > 0.0f)) {
        int b = 0;
        for (int i = 1; i < V; ++i)
            if (lg[(size_t) i] > lg[(size_t) b]) b = i;
        return {{b, 1.0}};
    }
    std::vector<std::pair<float, int>> c;
    c.reserve((size_t) V);
    for (int i = 0; i < V; ++i) c.push_back({lg[(size_t) i] / r.temperature, i});
    const size_t k = r.top_k > 0 ? std::min((size_t) r.top_k, c.size()) : c.size();
    std::partial_sort(c.begin(), c.begin() + (long) k, c.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    c.resize(k);
    double sum = 0;
    std::vector<double> p(k);
    for (size_t i = 0; i < k; ++i) sum += (p[i] = std::exp((double) (c[i].first - c[0].first)));
    size_t keep = k;
    double cum = 0;
    for (size_t i = 0; i < k; ++i) {
        cum += p[i] / sum;
        if (r.min_p > 0.0f && p[i] < r.min_p * p[0]) {
            keep = std::max<size_t>(1, i);
            break;
        }
        if (r.top_p < 1.0f && cum >= r.top_p) {
            keep = i + 1;
            break;
        }
    }
    double s2 = 0;
    for (size_t i = 0; i < keep; ++i) s2 += p[i];
    std::vector<std::pair<int, double>> out(keep);
    for (size_t i = 0; i < keep; ++i) out[i] = {c[i].second, p[i] / s2};
    return out;
}

// a draw from d, without `skip` (renormalised; -1: none)
int draw(const std::vector<std::pair<int, double>>& d, std::mt19937_64& rng, int skip = -1) {
    double tot = 0;
    for (const auto& [t, q] : d)
        if (t != skip) tot += q;
    if (!(tot > 0)) return d[0].first;
    double u = std::uniform_real_distribution<double>(0.0, tot)(rng);
    int last = d[0].first;
    for (const auto& [t, q] : d) {
        if (t == skip) continue;
        last = t;
        u -= q;
        if (u <= 0) return t;
    }
    return last;
}

int sample(std::vector<float>& lg, const Req& r, const std::vector<int>& hist, std::mt19937_64& rng) {
    return draw(dist(lg.data(), (int) lg.size(), r, hist), rng);
}

int serve(mm3::Model& m, const std::vector<int>& eos, int chunk, bool spec_on, float conf_floor) {
    for (const std::string& l : m.info_lines()) std::printf("NOTE %s\n", l.c_str());
    const auto& st0 = m.stats();
    std::printf("INFO engine=mm3 model=MiniMax-M3.1-Flash experts_vram=%ld experts=%ld dspark=%d\n", st0.resident,
                st0.experts, m.draft_block() > 0 && spec_on ? 1 : 0);
    if (const mm3::VisionGeo* vg = m.vision())   // (serve/mm3_vision.py's Geometry.from_info)
        std::printf("INFO vision=mm3 image_token=%d image_start=%d image_end=%d vision_patch=%d vision_merge=%d "
                    "vision_max_pixels=%d vision_min_pixels=%d\n", vg->image_token, vg->image_start, vg->image_end,
                    vg->patch, vg->merge, vg->max_pixels, vg->min_pixels);
    std::printf("READY %d stop\n", m.max_ctx());
    std::fflush(stdout);
    std::fprintf(stderr, "maya mm3 serve: GEN <max_new> [keys] <ids>; GENI <max_new> [keys] <file> <ids>; STOP, QUIT\n");
    std::signal(SIGPIPE, SIG_IGN);
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
        for (size_t a = from; a < r.ids.size();) {
            const int T = (int) std::min<size_t>((size_t) chunk, r.ids.size() - a);
            const bool last = a + (size_t) T == r.ids.size();
            if (!m.forward(r.ids.data() + a, T, (int) a, last ? 1 : 0, last ? lg.data() : nullptr, err)) {
                ok = false;
                break;
            }
            state.insert(state.end(), r.ids.begin() + (long) a, r.ids.begin() + (long) a + T);
            a += (size_t) T;
            const double ms = now() * 1e3 - t0;
            std::printf("PP %zu %zu %.1f %.1f %zu\n", a, r.ids.size(), ms, ms > 0 ? 1000.0 * (double) (a - from) / ms : 0.0,
                        from);
            std::fflush(stdout);
            if (in.stop.load() && !last) {
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
        std::mt19937_64 rng(r.seed ^ (0x9E3779B97F4A7C15ull * ++counter));
        const auto s_dec = m.stats();
        long long produced = 0;
        const char* finish = "length";
        bool in_think = r.think_budget > 0 && r.think_end >= 0;
        int tok = sample(lg, r, state, rng);
        auto stat_line = [&]() {
            const auto& s = m.stats();
            const double ms = now() * 1e3 - t1;
            const double ex = (double) ((s.profiled - s_dec.profiled)), cx = (double) (s.cpu_experts - s_dec.cpu_experts);
            std::printf("STAT tok_s=%.2f ms_tok=%.2f vram_hit=%.4f cpu=%.3f\n",
                        ms > 0 ? 1000.0 * (double) produced / ms : 0.0, produced > 0 ? ms / (double) produced : 0.0,
                        ex > 0 ? 1.0 - cx / ex : 1.0, produced > 0 ? cx / (double) produced : 0.0);
        };
        // DSpark: a round drafts after the token, verifies the token and its drafts in one forward and keeps the drafts
        // the target agrees with - greedy: its argmax; sampling: a draft d kept with the target's probability of d (the
        // drafts are the draft's argmax), else the token drawn from the target's distribution without d - so every
        // token comes from the target's own distribution; then the target's next token after the last kept one
        const int B = m.draft_block();
        const bool spec = B > 0 && (r.spec < 0 ? spec_on : r.spec != 0);
        const float floor = r.conf >= 0.0f ? r.conf : conf_floor;
        std::vector<float> lgw(spec ? (size_t) (B + 1) * m.vocab() : 0);
        std::vector<int> drafts;
        std::vector<float> confs;
        long long d_off = 0, d_acc = 0;
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
            if (in_think && produced >= r.think_budget && tok != r.think_end) tok = r.think_end;   // (the budget)
            if (!emit(tok)) break;
            const int pos = (int) state.size();
            int k = 0;
            if (spec) {   // (the drafts that can still be answer tokens: max_new, the context, the thinking budget)
                long long room = std::min<long long>(r.max_new - produced, (long long) m.max_ctx() - pos - 1);
                if (in_think) room = std::min<long long>(room, r.think_budget - produced);
                if (room > 0) {
                    if (!m.draft(tok, pos, B, drafts, confs, err)) {
                        std::fprintf(stderr, "mm3: %s\n", err.c_str());
                        finish = "cancel";
                        break;
                    }
                    while (k < (int) drafts.size() && k < room && confs[(size_t) k] >= floor) ++k;
                }
            }
            if (k == 0) {
                if (!m.forward(&tok, 1, pos, 1, lgw.empty() ? lg.data() : lgw.data(), err)) {
                    std::fprintf(stderr, "mm3: %s\n", err.c_str());
                    finish = "cancel";
                    break;
                }
                state.push_back(tok);
                const float* row = lgw.empty() ? lg.data() : lgw.data();
                tok = draw(dist(row, m.vocab(), r, state), rng);
                continue;
            }
            std::vector<int> win(1, tok);
            win.insert(win.end(), drafts.begin(), drafts.begin() + k);
            if (!m.forward(win.data(), k + 1, pos, k + 1, lgw.data(), err)) {
                std::fprintf(stderr, "mm3: %s\n", err.c_str());
                finish = "cancel";
                break;
            }
            state.push_back(tok);
            d_off += k;
            bool more = true;
            int next = -1;
            for (int i = 0; i <= k && more; ++i) {
                const auto d = dist(lgw.data() + (size_t) i * m.vocab(), m.vocab(), r, state);
                if (i == k) {   // every draft kept: the target's token after them
                    next = draw(d, rng);
                    break;
                }
                const int x = drafts[(size_t) i];
                double px = 0;
                for (const auto& [t, q] : d)
                    if (t == x) px = q;
                const bool keep = r.temperature > 0.0f ? std::uniform_real_distribution<double>(0.0, 1.0)(rng) < px
                                                       : d[0].first == x;
                if (!keep) {
                    next = r.temperature > 0.0f ? draw(d, rng, x) : d[0].first;
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
        stat_line();
        const double t2 = now() * 1e3;
        const auto& se = m.stats();
        const long long look = se.profiled - s_dec.profiled, cpu = se.cpu_experts - s_dec.cpu_experts;
        std::printf("DONE %lld %zu %.1f %.1f %s %lld %lld %zu %lld %lld\n", produced, r.ids.size(), t1 - t0, t2 - t1,
                    finish, d_acc, d_off, from, look - cpu, look);
        std::fflush(stdout);
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

int usage() {
    std::fprintf(stderr,
                 "usage: maya-mm3 --pack DIR --ids IDS|@FILE [--gen N] [--dump-logits FILE --last N] [--gpus LIST]\n"
                 "                [--max-context N] [--chunk N] [--staging N] [--resident-cap N] [--reserve-gb X]\n"
                 "                [--vision on|off] [--vision-gpu I] [--images FILE] [--layer-split auto|L,L,...]\n"
                 "                [--verbose]\n"
                 "       maya-mm3 --pack DIR --encode-image FILE.rgb --size WxH --dump-image OUT [--gpus N]\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    mm3::Options o;
    std::string ids_arg, dump, profile_out, batch, eos_arg, images, enc_rgb, enc_size, enc_out;
    bool serve_mode = false;
    int gen = 0, last = 1, ppl = 0;
    float dspark_conf = 0.7f;   // the drafts' confidence floor: a round verifies the drafts before the first below it
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
        else if (a == "--chunk") o.chunk = std::atoi(val().c_str());
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
        else if (a == "--cpu-threads") o.cpu_threads = std::atoi(val().c_str());
        else if (a == "--dspark") o.dspark = val() != "off";
        else if (a == "--vision") o.vision = val() != "off";
        else if (a == "--tp") o.tp = std::atoi(val().c_str());
        else if (a == "--cache-gb") o.cache_gb = std::atof(val().c_str());
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
        else if (a == "--serve") serve_mode = true;
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
        std::vector<float> rows;
        for (int rep = 0; rep < 2; ++rep) {   // (the second one timed)
            const double t0 = now();
            if (!v.encode(px.data(), w, h, rows, err)) {
                std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
                return 1;
            }
            if (rep == 1)
                std::fprintf(stderr, "vision: %d x %d pixels -> %zu rows in %.0f ms (%zu MB on CUDA%d)\n", w, h,
                             rows.size() / (size_t) v.geo().out_dim, 1e3 * (now() - t0), v.bytes() >> 20, v.device());
        }
        std::FILE* fo = std::fopen(enc_out.c_str(), "wb");
        if (!fo || std::fwrite(rows.data(), 4, rows.size(), fo) != rows.size()) {
            std::fprintf(stderr, "%s: cannot write\n", enc_out.c_str());
            return 1;
        }
        std::fclose(fo);
        return 0;
    }
    if (o.pack.empty() || (ids_arg.empty() && batch.empty() && !serve_mode)) return usage();
    if (!images.empty()) o.vision = true;
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
    if (serve_mode) {
        std::vector<int> eos = eos_arg.empty() ? std::vector<int>{m.eos()} : parse_list(eos_arg);
        return serve(m, eos, o.chunk, o.dspark, dspark_conf);
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
        const double t0 = now();
        if (!read_pics(images, pics, err) || !picture_rows(m, pics, ids, 0, rows, err)) {
            std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "pictures: %zu encoded in %.0f ms\n", pics.size(), 1e3 * (now() - t0));
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
    if (const char* sv = std::getenv("MM3_STRESS"); sv && m.draft_block() > 0) {
        const int iters = std::atoi(sv);
        const int tw = std::getenv("MM3_STRESS_T") ? std::atoi(std::getenv("MM3_STRESS_T")) : 4;
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
            if (const char* ns = std::getenv("MM3_NSYS")) {   // (profiling: rounds 20 .. 20 + N under nsys's capture)
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
            if (std::getenv("MM3_CONF") && k > 0) {   // (debugging: each round's confidences)
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
        if (std::getenv("MM3_TIMING") && done > 0) {
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
