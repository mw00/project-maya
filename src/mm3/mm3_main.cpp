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

#include "mm3_model.hpp"

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
    std::vector<int> ids;
};

bool parse_gen(const std::string& line, Req& r) {
    std::istringstream is(line.substr(4));
    if (!(is >> r.max_new)) return false;
    for (std::string tok; is >> tok;) {
        const size_t eq = tok.find('=');
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
        else if (k == "penalty_last_n") r.pen_n = std::atoi(v.c_str());
        else if (k == "penalty_repeat") r.pen_repeat = std::strtof(v.c_str(), nullptr);
        else if (k == "penalty_freq") r.pen_freq = std::strtof(v.c_str(), nullptr);
        else if (k == "penalty_present") r.pen_present = std::strtof(v.c_str(), nullptr);
        else if (k == "think_budget") r.think_budget = std::atoll(v.c_str());
        else if (k == "think_end") r.think_end = std::atoi(v.c_str());
        // (keys this engine does not use - spec, dspark_conf, ... - are taken and ignored)
    }
    return !r.ids.empty();
}

// the next token from one row of logits: penalties over the history's last pen_n tokens (llama.cpp's), then greedy
// (temperature <= 0) or temperature, top_k, top_p, min_p and a draw
int sample(std::vector<float>& lg, const Req& r, const std::vector<int>& hist, std::mt19937_64& rng) {
    const int V = (int) lg.size();
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
        return b;
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
    double u = std::uniform_real_distribution<double>(0.0, s2)(rng);
    for (size_t i = 0; i < keep; ++i) {
        u -= p[i];
        if (u <= 0) return c[i].second;
    }
    return c[keep - 1].second;
}

int serve(mm3::Model& m, const std::vector<int>& eos, int chunk) {
    for (const std::string& l : m.info_lines()) std::printf("NOTE %s\n", l.c_str());
    const auto& st0 = m.stats();
    std::printf("INFO engine=mm3 model=MiniMax-M3.1-Flash experts_vram=%ld experts=%ld\n", st0.resident, st0.experts);
    std::printf("READY %d stop\n", m.max_ctx());
    std::fflush(stdout);
    std::fprintf(stderr, "maya mm3 serve: GEN <max_new> [keys] <ids>; STOP, QUIT\n");
    std::signal(SIGPIPE, SIG_IGN);
    Lines in;
    in.start();
    std::vector<int> state;   // the tokens in the cache, positions 0 ..
    std::vector<float> lg((size_t) m.vocab());
    unsigned long long counter = 0;
    std::string line, err;
    while (in.next(line)) {
        if (line == "QUIT") break;
        if (line.rfind("GEN ", 0) != 0) {
            std::printf("ERR unsupported command '%s' (GEN, STOP and QUIT)\n", line.substr(0, line.find(' ')).c_str());
            std::fflush(stdout);
            continue;
        }
        in.stop.store(false);
        Req r;
        if (!parse_gen(line, r)) {
            std::printf("ERR expected: GEN <max_new> [keys] <id,id,...>\n");
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
        state.resize(from);
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
        if (!ok) {
            std::printf("ERR %s\n", err.c_str());
            std::printf("DONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", r.ids.size());
            std::fflush(stdout);
            state.clear();
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
        for (;;) {
            if (in.stop.load()) {
                finish = "cancel";
                break;
            }
            if (in_think && produced >= r.think_budget && tok != r.think_end) tok = r.think_end;   // (the budget)
            std::printf("T %d\n", tok);
            ++produced;
            if (in_think && tok == r.think_end) in_think = false;
            if (produced % 8 == 0) stat_line();
            std::fflush(stdout);
            bool end = false;
            for (int e : eos) end = end || tok == e;
            if (end) {
                finish = "stop";
                break;
            }
            if (produced >= r.max_new || (int) state.size() >= m.max_ctx()) break;
            if (!m.forward(&tok, 1, (int) state.size(), 1, lg.data(), err)) {
                std::fprintf(stderr, "mm3: %s\n", err.c_str());
                finish = "cancel";
                break;
            }
            state.push_back(tok);
            tok = sample(lg, r, state, rng);
        }
        stat_line();
        const double t2 = now() * 1e3;
        const auto& se = m.stats();
        const long long look = se.profiled - s_dec.profiled, cpu = se.cpu_experts - s_dec.cpu_experts;
        std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu %lld %lld\n", produced, r.ids.size(), t1 - t0, t2 - t1, finish,
                    from, look - cpu, look);
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
                 "                [--verbose]\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    mm3::Options o;
    std::string ids_arg, dump, profile_out, batch, eos_arg;
    bool serve_mode = false;
    int gen = 0, last = 1, ppl = 0;
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
        else if (a == "--ppl") ppl = std::atoi(val().c_str());
        else if (a == "--batch") batch = val();
        else if (a == "--cpu-threads") o.cpu_threads = std::atoi(val().c_str());
        else if (a == "--serve") serve_mode = true;
        else if (a == "--eos") eos_arg = val();
        else if (a == "--verbose") o.verbose = true;
        else return usage();
    }
    if (o.pack.empty() || (ids_arg.empty() && batch.empty() && !serve_mode)) return usage();
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
        return serve(m, eos, o.chunk);
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
    last = std::max(1, std::min(last, n));
    std::vector<float> logits((size_t) last * m.vocab());
    double t0 = now();
    if (!m.forward(ids.data(), n, 0, last, logits.data(), err)) {
        std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
        return 1;
    }
    double t1 = now();
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
            std::fprintf(stderr, "  a token: %.1f ms - routing waits %.1f ms, CPU lane %.1f ms\n", 1e3 * s2.t_token / done,
                         1e3 * s2.t_sync / done, 1e3 * s2.t_lane / done);
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
