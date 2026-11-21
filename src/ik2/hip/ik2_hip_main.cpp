// src/ik2/hip/ik2_hip_main.cpp - maya-ik2's AMD build: Thinking Machines' Inkling 2 Small on the GPUs
// (ik2_hip_model.hpp).  The CUDA build is src/ik2/ik2_main.cu (docs/INKLING2.md); this one runs on maya-ds4's ggml
// block-type kernels (docs/INKLING2_AMD.md).
//
//   maya-ik2 --pack <dir> --tokens 371,151451,... [--n-predict N]   greedy decode from token ids, with timings
//   maya-ik2 --pack <dir> --parity <reference dir>                   the reference's tokens one at a time, every
//                                                                    tensor of its dump compared (llama.cpp's
//                                                                    graph names: docs/INKLING2_AMD.md)
//   maya-ik2 --pack <dir> --serve                                    serve/server.py's line protocol
//   (--model <the set's first .gguf> in place of --pack: the pack beside it, as the CUDA build is given the model)
// Decoding drafts with the model's DSpark drafter unless --dspark off (draft_round below; docs/INKLING2_DSPARK.md).
#include "ik2_hip_model.hpp"

#include "mm3_os.hpp"
#include "mm3_sample.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <random>
#include <thread>
#include <sstream>
#include <string>
#include <vector>

namespace {

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
std::vector<int> int_list(const std::string& s) {
    std::vector<int> v;
    std::string x = s;
    for (char& c : x)
        if (c == ',') c = ' ';
    std::istringstream is(x);
    for (int i; is >> i;) v.push_back(i);
    return v;
}

// One tensor of a reference dump (index.jsonl: step, name, file, type, ne)
struct RefT {
    std::string file;
    bool i32 = false;
    int64_t ne[4] = {1, 1, 1, 1};
    int64_t total() const { return ne[0] * ne[1] * ne[2] * ne[3]; }
};
struct Ref {
    std::string dir;
    std::map<std::string, RefT> t;   // "<step>/<name>"
    std::vector<int> prompt, greedy;
    std::map<std::string, FILE*> open;

    static std::string field(const std::string& line, const std::string& key) {
        const size_t a = line.find("\"" + key + "\":");
        if (a == std::string::npos) return "";
        size_t b = a + key.size() + 3;
        if (line[b] == '"') return line.substr(b + 1, line.find('"', b + 1) - b - 1);
        if (line[b] == '[') return line.substr(b + 1, line.find(']', b) - b - 1);
        return line.substr(b, line.find_first_of(",}", b) - b);
    }
    bool load(const std::string& d, std::string& err) {
        dir = d;
        std::ifstream f(dir + "/index.jsonl"), tk(dir + "/tokens.txt");
        if (!f || !tk) {
            err = dir + ": no index.jsonl / tokens.txt (a reference dump: tools/ik2/ref)";
            return false;
        }
        for (std::string line; std::getline(f, line);) {
            RefT r;
            r.file = field(line, "file");
            r.i32 = field(line, "type") == "i32";
            const std::vector<int> ne = int_list(field(line, "ne"));
            for (size_t i = 0; i < ne.size() && i < 4; ++i) r.ne[i] = ne[i];
            t[field(line, "step") + "/" + field(line, "name")] = r;
        }
        std::string l1, l2;
        std::getline(tk, l1);
        std::getline(tk, l2);
        prompt = int_list(l1);
        greedy = int_list(l2);
        return !prompt.empty();
    }
    // row `row` of `rows` of the step's tensor (n values of 4 bytes) -> dst; false: not in the dump, or another shape
    bool row(int step, const std::string& name, int64_t rw, int64_t rows, size_t n, void* dst) {
        auto it = t.find(std::to_string(step) + "/" + name);
        if (it == t.end() || it->second.total() != (int64_t) n * rows) return false;
        FILE*& fp = open[it->second.file];
        if (!fp) fp = std::fopen((dir + "/" + it->second.file).c_str(), "rb");
        if (!fp) return false;
        return fseeko(fp, (off_t) (rw * (int64_t) n * 4), SEEK_SET) == 0 && std::fread(dst, 4, n, fp) == n;
    }
};

struct Acc {
    double cos_min = 2.0, cos_sum = 0, rel_max = 0;
    int64_t n = 0, hit = 0, all = 0, pos_diff = 0;
};

// ---- --serve: the line protocol serve/server.py's engine speaks (maya-hy4's, docs/INKLING2_AMD.md, "Serving") -------------
// The requests' lines from stdin on a thread of their own: STOP and WRAP are flags the request in hand sees at once
struct Lines {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    bool eof = false;
    std::atomic<bool> stop{false}, wrap{false};
    // (--parallel, maya-ds4's: `STOP <rid>` ends that request alone - stop_rids; a bare STOP every one)
    bool multi = false;
    std::vector<long long> stop_rids;
    bool stopped(long long rid) {
        if (stop.load() || wrap.load()) return true;
        std::lock_guard<std::mutex> lk(mu);
        return std::find(stop_rids.begin(), stop_rids.end(), rid) != stop_rids.end();
    }
    void forget(long long rid) {   // (the request is over: a late STOP of it changes nothing)
        std::lock_guard<std::mutex> lk(mu);
        stop_rids.erase(std::remove(stop_rids.begin(), stop_rids.end(), rid), stop_rids.end());
    }
    bool try_next(std::string& out) {
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
            char chunk[4096];   // (a read's size only: the lines are put together across reads)
            for (;;) {
                const long n = mm3::os::read_stdin(chunk, sizeof chunk);
                if (n <= 0) break;
                buf.append(chunk, (size_t) n);
                for (size_t nl; (nl = buf.find('\n')) != std::string::npos;) {
                    std::string l = buf.substr(0, nl);
                    buf.erase(0, nl + 1);
                    if (!l.empty() && l.back() == '\r') l.pop_back();
                    if (l == "STOP" || l.rfind("STOP ", 0) == 0) {
                        if (multi && l.size() > 5) {
                            std::lock_guard<std::mutex> lk(mu);
                            stop_rids.push_back(std::atoll(l.c_str() + 5));
                        } else {
                            stop.store(true);
                        }
                        continue;
                    }
                    if (l == "WRAP") {
                        wrap.store(true);
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

// GEN <max_new> [key=value ...] <id,id,...>: keys this engine does not use are taken and ignored
// logit_bias=<id>:<bias>,... (OpenAI's logit_bias; the server's ignore_eos bans the end tokens with -100) -> bias, for
// Model::set_logit_bias: added to the logits of every pick of the answer, on the device
bool parse_gen(const std::string& line, mm3::sample::Req& r, float& cum, std::vector<std::pair<int, float>>& bias) {
    bias.clear();
    std::istringstream is(line.substr(4));
    if (!(is >> r.max_new)) return false;
    for (std::string tok; is >> tok;) {
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) {
            r.ids = int_list(tok);
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
        else if (k == "spec") r.spec = std::atoi(v.c_str());
        else if (k == "dspark_cum" || k == "spec_cum") cum = std::strtof(v.c_str(), nullptr);
        else if (k == "logit_bias") {   // (the server checked them: ids in the vocabulary, -100 .. 100)
            size_t at = 0;
            while (at < v.size()) {
                const size_t end = std::min(v.find(',', at), v.size()), colon = v.find(':', at);
                if (colon != std::string::npos && colon < end)
                    bias.emplace_back(std::atoi(v.substr(at, colon - at).c_str()),
                                      std::strtof(v.substr(colon + 1, end - colon - 1).c_str(), nullptr));
                at = end + 1;
            }
        }
    }
    return !r.ids.empty();
}

// The next token from a row of logits (the padded vocabulary rows are not in it): greedy without a temperature; else
// mm3's distribution (temperature, top_k, top_p, min_p, the penalties) over the tokens within 40 temperature-nats of
// the top - the rest hold under 1e-12 of the mass - or over the whole row when a penalty may move any of them
int sample(const float* row, int V, const mm3::sample::Req& r, const std::vector<int>& hist, std::mt19937_64& rng) {
    if (mm3::sample::penalised(r)) return mm3::sample::draw(mm3::sample::dist(row, V, r, hist), rng);
    int b = 0;
    for (int i = 1; i < V; ++i)
        if (row[i] > row[b]) b = i;
    if (!(r.temperature > 0.0f)) return b;
    const float cut = row[b] - 40.0f * r.temperature;
    if (!(r.top_k > 0) && !(r.top_p < 1.0f) && !(r.min_p > 0.0f)) {   // (plain sampling: the softmax itself, no sort)
        static thread_local std::vector<float> w;
        w.resize((size_t) V);
        const float it = 1.0f / r.temperature, mx = row[b];
        double z = 0;
        for (int i = 0; i < V; ++i) z += (w[(size_t) i] = row[i] >= cut ? std::exp((row[i] - mx) * it) : 0.0f);
        double u = std::uniform_real_distribution<double>(0.0, z)(rng);
        for (int i = 0; i < V; ++i) {
            u -= w[(size_t) i];
            if (u <= 0 && w[(size_t) i] > 0) return i;
        }
        return b;
    }
    std::vector<std::pair<int, float>> c;
    for (int i = 0; i < V; ++i)
        if (row[i] >= cut) c.push_back({i, row[i]});
    return mm3::sample::draw(mm3::sample::dist_c(std::move(c), {}, {}, r, hist), rng);
}

// ---- drafting (the DSpark drafter, Model::draft): a round reads the token in hand and k drafts in one window, keeps the
// drafts the model agrees with and gives the token after them ----------------------------------------------------------
// The window: k maximizes the tokens a millisecond the round is expected to commit, (1 + sum_i prod_{j<=i} p_j) /
// (t(k + 1) + t_draft), p the drafter's probabilities (close to the measured acceptance: docs/INKLING2_DSPARK.md), t(T)
// this PC's time of a pass of T rows - a line a + b (T - 1) fitted as it decodes (a: the token steps, b: the windows),
// from the measured table's values at 3k context - and t_draft the draft's measured time.  k 0: a plain token step.
// The drafter's own cut on the running product of the probabilities (Options::dspark_cum, 0.5) is the floor under it.
// Rest: a drafted round gains (kept drafts) x a - (t_draft + k b) milliseconds over plain steps; when that, averaged
// over the last drafted rounds, is under a tenth of a token step (terse thinking text: half of 1.5 drafts kept), the
// drafter is not asked for a while (the tokens' taps still reach it: Model::window feeds them), then tried again -
// 8 rounds, doubling to 32 while it keeps not paying, from 8 again with every request.  (Short on purpose: a rest costs
// little where drafts are break-even and much where the text turns to code or prose - measured through the server:
// rests of up to 128 rounds carried from a thinking block slept through most of the code after it.)
// (a, b and t_draft's starting values, rest_len / its cap 32, the 8 rounds before judging, the 0.1 a / 0.3 a bars, the
//  0.9 / 0.1 averages and b's 0.5 ms floor: each with an IK2_HIP_DRAFT_* override, read when a Drafting is made -
//  docs/INKLING2_AMD.md, "Values and their sources")
double env_d(const char* k, double d) {
    const char* v = std::getenv(k);
    return v != nullptr && *v != 0 ? std::strtod(v, nullptr) : d;
}
struct Drafting {
    bool on = false;
    double a = env_d("IK2_HIP_DRAFT_A0", 30.0), b = env_d("IK2_HIP_DRAFT_B0", 7.0),
           t_draft = env_d("IK2_HIP_DRAFT_T0", 10.0);   // ms: a token's pass, a window row more, a draft
    double gain = 0.0;                          // ms a drafted round gains over plain steps (EMA)
    int rest_len0 = (int) env_d("IK2_HIP_DRAFT_REST", 8), rest_max = (int) env_d("IK2_HIP_DRAFT_REST_MAX", 32),
        warm = (int) env_d("IK2_HIP_DRAFT_WARM", 8);
    double rest_gain = env_d("IK2_HIP_DRAFT_REST_GAIN", 0.1), reset_gain = env_d("IK2_HIP_DRAFT_RESET_GAIN", 0.3),
           ema = env_d("IK2_HIP_DRAFT_EMA", 0.9), b_min = env_d("IK2_HIP_DRAFT_B_MIN", 0.5);
    int n_d = 0, rest = 0, rest_len = rest_len0;
    int rest0 = 0;   // rounds without a draft after one whose drafts were not worth a window (--dspark-rest0; measured:
                     // 1 costs prose a sixth of its drafted rounds and gains thinking text nothing)
    // (counters over the engine's life; a request's are differences)
    long long rounds = 0, drafted = 0, offered = 0, accepted = 0, passes = 0, tokens = 0, khist[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    long long rested = 0;
    double draft_ms = 0, verify_ms = 0;
    bool log = std::getenv("IK2_HIP_ROUND_LOG") != nullptr;
    // IK2_HIP_MARGIN_LOG=1 (tests): a line a greedy token - the gap between the row's two largest logits (a near-tie
    // is where two runs whose experts sit differently may part)
    bool margins = std::getenv("IK2_HIP_MARGIN_LOG") != nullptr;

    double cost(int T) const { return a + b * (T - 1); }
    int choose(const std::vector<float>& p, int n) const {
        int best = 0;
        double cum = 1.0, exp_tok = 1.0, best_rate = 1.0 / (cost(1) + t_draft);
        for (int k = 1; k <= n; ++k) {
            cum *= std::min(1.0, std::max(0.0, (double) p[(size_t) k - 1]));
            exp_tok += cum;
            const double rate = exp_tok / (cost(k + 1) + t_draft);
            if (rate > best_rate) {
                best_rate = rate;
                best = k;
            }
        }
        return best;
    }
    void observe_pass(int T, double ms) {
        if (T == 1) a = ema * a + (1 - ema) * ms;
        else b = ema * b + (1 - ema) * std::max(b_min, (ms - a) / (T - 1));
    }
    void observe_round(bool was_drafted, int k, int kept, double t_draft_ms) {
        if (!was_drafted) return;
        const double g = kept * a - (t_draft_ms + k * b);
        gain = n_d > 0 ? ema * gain + (1 - ema) * g : g;
        if (++n_d < warm) return;
        if (gain < rest_gain * a) {   // drafting does not pay here: rest, longer each time in a row
            rest = rest_len;
            rest_len = std::min(rest_max, rest_len * 2);
            n_d = 0;
            ++rested;
        } else if (gain > reset_gain * a) {
            rest_len = rest_len0;
        }
    }
    void start() {   // (a request's first round: no rest left over from the last one)
        rest = 0;
        rest_len = rest_len0;
        n_d = 0;
    }
};

// The model's verdict on draft d at a row of logits: true: d stands (greedy: it is the row's pick; sampled: with
// probability p(d), p the request's own distribution - ds4's rule for argmax drafts, so the token is the model's own
// draw either way); false: `next` is the row's token instead (greedy: its pick; sampled: a draw from p without d)
bool verdict(const float* row, int V, const mm3::sample::Req& r, const std::vector<int>& hist, std::mt19937_64& rng, int d,
             int& next) {
    const bool pen = mm3::sample::penalised(r);
    if (!(r.temperature > 0.0f)) {
        if (pen) next = mm3::sample::dist(row, V, r, hist)[0].first;
        else {
            next = 0;
            for (int i = 1; i < V; ++i)
                if (row[i] > row[next]) next = i;
        }
        return next == d;
    }
    if (!pen && !(r.top_k > 0) && !(r.top_p < 1.0f) && !(r.min_p > 0.0f)) {   // (plain sampling: the softmax itself)
        int b = 0;
        for (int i = 1; i < V; ++i)
            if (row[i] > row[b]) b = i;
        static thread_local std::vector<float> w;
        w.resize((size_t) V);
        const float it = 1.0f / r.temperature, mx = row[b], cut = row[b] - 40.0f * r.temperature;
        double z = 0;
        for (int i = 0; i < V; ++i) z += (w[(size_t) i] = row[i] >= cut ? std::exp((row[i] - mx) * it) : 0.0f);
        const double pd = d >= 0 && d < V ? w[(size_t) d] / z : 0.0;
        if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < pd) return true;
        double u = std::uniform_real_distribution<double>(0.0, z - (d >= 0 && d < V ? w[(size_t) d] : 0.0))(rng);
        next = b == d ? -1 : b;
        for (int i = 0; i < V; ++i) {
            if (i == d || !(w[(size_t) i] > 0)) continue;
            if (next < 0) next = i;
            u -= w[(size_t) i];
            if (u <= 0) {
                next = i;
                break;
            }
        }
        if (next < 0) next = b;
        return false;
    }
    std::vector<std::pair<int, double>> p;
    if (pen) {
        p = mm3::sample::dist(row, V, r, hist);
    } else {
        int b = 0;
        for (int i = 1; i < V; ++i)
            if (row[i] > row[b]) b = i;
        const float cut = row[b] - 40.0f * r.temperature;
        std::vector<std::pair<int, float>> c;
        for (int i = 0; i < V; ++i)
            if (row[i] >= cut) c.push_back({i, row[i]});
        p = mm3::sample::dist_c(std::move(c), {}, {}, r, hist);
    }
    double pd = 0;
    for (const auto& [t, q] : p)
        if (t == d) pd = q;
    if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < pd) return true;
    next = mm3::sample::draw(p, rng, d);
    return false;
}

// One round after `tok`, the token at m.end() (given out, not read yet).  hist: the tokens before tok and tok (the
// penalties' history).  room: the tokens the request may still give out (>= 1).  out: the tokens after tok - the drafts
// that stood, then the model's next - the last of them the next round's tok; `read`: the rows the context took (tok and
// the kept drafts).  spec: this request drafts.
bool draft_round(ik2hip::Model& m, Drafting& dr, int tok, const mm3::sample::Req& r, const std::vector<int>& hist,
                 std::mt19937_64& rng, long long room, bool spec, std::vector<int>& out, std::vector<float>& lg,
                 std::string& err) {
    out.clear();
    const int V = m.vocab();
    const bool sampled = r.temperature > 0.0f || mm3::sample::penalised(r);
    const bool want_lg = sampled || dr.margins;
    const double t0 = now_s();
    std::vector<int> toks;
    std::vector<float> probs;
    int k = 0;
    bool was_drafted = false;
    const long long ctx_room = m.max_ctx() - m.end();
    if (spec && dr.on && room > 1 && ctx_room > 2) {
        if (dr.rest > 0) {
            --dr.rest;
        } else {
            const int max_n = (int) std::min<long long>({7, room - 1, (long long) m.window_max() - 1, ctx_room - 1});
            if (!m.draft(tok, max_n, toks, probs, err)) return false;
            const double td = (now_s() - t0) * 1e3;
            dr.t_draft = dr.ema * dr.t_draft + (1 - dr.ema) * td;
            dr.draft_ms += td;
            ++dr.drafted;
            was_drafted = true;
            k = dr.choose(probs, (int) toks.size());
            if (k == 0) dr.rest = std::max(dr.rest, dr.rest0);   // (no draft worth a window here: likely none at the next token)
        }
    }
    const int T = k + 1;
    int ids[16], tops[16];   // (at least window_max(), 9 rows)
    ids[0] = tok;
    for (int i = 0; i < k; ++i) ids[i + 1] = toks[(size_t) i];
    if (want_lg) lg.resize((size_t) T * V);
    const double t1 = now_s();
    if (!m.window(ids, T, m.end(), tops, want_lg ? lg.data() : nullptr, err)) return false;
    const double tv = (now_s() - t1) * 1e3;
    dr.verify_ms += tv;
    dr.observe_pass(T, tv);
    int acc = 0, next = -1;
    std::vector<int> h2;
    if (want_lg) h2 = hist;
    for (int i = 0; i < k; ++i) {
        const int d = toks[(size_t) i];
        bool ok;
        if (!sampled) {
            ok = tops[i] == d;
            next = tops[i];
        } else {
            ok = verdict(lg.data() + (size_t) i * V, V, r, h2, rng, d, next);
        }
        if (!ok) break;
        ++acc;
        if (want_lg) h2.push_back(d);
    }
    if (dr.margins && !sampled)
        for (int i = 0; i <= acc; ++i) {   // (the rows that gave a token: its logit over the runner-up's)
            const float* row = lg.data() + (size_t) i * V;
            float b1 = -1e30f, b2 = -1e30f;
            for (int v = 0; v < V; ++v) {
                if (row[v] > b1) {
                    b2 = b1;
                    b1 = row[v];
                } else if (row[v] > b2) {
                    b2 = row[v];
                }
            }
            std::fprintf(stderr, "MARGIN pos %lld token %d gap %.4f\n", (long long) (m.end() - T + i + 1), tops[i], b1 - b2);
        }
    if (acc == k) {
        if (!sampled) next = tops[k];
        else {
            next = sample(lg.data() + (size_t) k * V, V, r, h2, rng);
        }
    }
    if (acc < k && !m.keep(acc + 1, err)) return false;
    for (int i = 0; i < acc; ++i) out.push_back(toks[(size_t) i]);
    out.push_back(next);
    ++dr.rounds;
    ++dr.passes;
    dr.tokens += acc + 1;
    dr.offered += k;
    dr.accepted += acc;
    ++dr.khist[std::min(k, 7)];
    dr.observe_round(was_drafted, k, acc, (t1 - t0) * 1e3);
    if (dr.log) {
        std::fprintf(stderr, "ROUND pos %lld made %zu k %d kept %d draft %.1f ms pass %.1f ms p", (long long) m.end(), toks.size(),
                     k, acc, was_drafted ? (t1 - t0) * 1e3 : 0.0, tv);
        for (float p : probs) std::fprintf(stderr, " %.2f", p);
        std::fprintf(stderr, "\n");
    }
    return true;
}

// short: a prompt's unread tokens up to this many go through the decode path, a token a step (measured: the prompt
// path's chunk costs more than that many steps)
// spec: a request drafts unless it says (GEN spec=0|1), as maya-ds4's --dspark on / off; drafting: it may draft at
// all (--dspark idle: the drafter fed, never asked)
int serve(ik2hip::Model& m, const std::vector<int>& stops, int short_n, bool spec_default, bool drafting, float cum_default,
          int rest0) {
    for (const std::string& l : m.info_lines()) std::printf("NOTE %s\n", l.c_str());
    const ik2hip::Stats& st = m.stats();
    std::printf("INFO engine=ik2 engine_kind=ik2-hip model=Inkling-2-Small ctx=%lld vram_gb=%.2f dspark=%d mtp=0 "
                "experts=%d experts_vram=%d\n", (long long) m.max_ctx(), st.vram_gb + m.dspark_vram_gb(),
                m.dspark() && drafting && spec_default ? 1 : 0, st.experts_all, st.experts_vram);
    std::printf("INFO logit_bias=1\n");   // (GEN logit_bias=<id>:<bias>,...: added to the logits of every pick)
    Drafting dr;
    dr.on = m.dspark() && drafting;
    if (rest0 >= 0) dr.rest0 = rest0;
    std::vector<int> out;
    std::vector<float> lgw;
    std::printf("READY %lld stop\n", (long long) m.max_ctx());
    std::fflush(stdout);
    std::fprintf(stderr, "maya ik2 serve: GEN <max_new> [keys] <ids>; STOP, WRAP, QUIT\n");
    Lines lines;
    lines.start();
    std::vector<int> held;   // the tokens the context holds
    std::vector<float> lg((size_t) m.vocab());
    unsigned long long counter = 0;
    std::string err;
    auto is_stop = [&](int t) { return std::find(stops.begin(), stops.end(), t) != stops.end(); };
    for (std::string line; lines.next(line);) {
        if (line == "QUIT") break;
        if (line.empty()) continue;
        if (line.rfind("GEN ", 0) != 0) {
            std::printf("ERR unsupported command '%s' (GEN, STOP, WRAP and QUIT)\n", line.substr(0, 40).c_str());
            std::fflush(stdout);
            continue;
        }
        lines.stop = false;
        ++counter;
        mm3::sample::Req r;
        float cum = cum_default;
        std::vector<std::pair<int, float>> bias;
        if (!parse_gen(line, r, cum, bias)) {
            std::printf("ERR expected: GEN <max_new> [keys] <id,id,...>\n");
            std::fflush(stdout);
            continue;
        }
        const size_t n = r.ids.size();
        bool bad = false;
        for (int t : r.ids) bad = bad || t < 0 || t >= m.vocab();
        if (bad || (long long) n + std::max(0ll, r.max_new) > m.max_ctx()) {
            if (bad) std::printf("ERR a prompt token is outside the vocabulary (%d)\n", m.vocab());
            else
                std::printf("ERR the prompt (%zu tokens) and max_new %lld are more than the context %lld\n", n, r.max_new,
                            (long long) m.max_ctx());
            std::printf("DONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", n);
            std::fflush(stdout);
            continue;
        }
        // the request's logit bias, before its prompt is read (the prompt's last row gives the answer's first token);
        // none: the last request's is cleared
        if (!m.set_logit_bias(bias, err)) {
            std::printf("ERR the logit bias: %s\n", err.c_str());
            std::fprintf(stderr, "ik2: the logit bias: %s\n", err.c_str());
            std::printf("DONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", n);
            std::fflush(stdout);
            continue;
        }
        // the longest prefix the context holds (one token at least is read: its logits start the answer), then the
        // latest saved position at or before it - the tokens from there are read again
        size_t lcp = 0;
        while (lcp < held.size() && lcp < n - 1 && held[lcp] == r.ids[lcp]) ++lcp;
        const size_t from = (size_t) m.rewind((int64_t) lcp);
        held.resize(from);
        const double t0 = now_s();
        auto pp = [&](size_t read) {
            const double ms = (now_s() - t0) * 1e3;
            std::printf("PP %zu %zu %.1f %.1f %zu\n", read, n, ms, ms > 0 ? 1e3 * (double) (read - from) / ms : 0.0, from);
            std::fflush(stdout);
        };
        pp(from);
        const bool want_lg = r.temperature > 0.0f || mm3::sample::penalised(r);
        bool ok = true, cancel = false;
        if (n - from <= (size_t) short_n) {
            for (size_t i = from; i < n && ok && !cancel; ++i) {
                ok = m.forward(&r.ids[i], 1, (int64_t) i, i + 1 == n && want_lg ? lg.data() : nullptr, err);
                cancel = lines.stop.load() || lines.wrap.load();
                if (ok && (i + 1 - from) % 64 == 0) pp(i + 1);   // (maya-ds4's: a PP line every 64 tokens)
            }
            if (ok && cancel && (size_t) m.end() == n) cancel = false;   // (all read: the answer's first token may still go)
        } else {
            m.set_progress([&](int64_t done) {
                pp(from + (size_t) done);
                return !(lines.stop.load() || lines.wrap.load());
            });
            ok = m.forward(&r.ids[from], (int) (n - from), (int64_t) from, want_lg ? lg.data() : nullptr, err);
            m.set_progress(nullptr);
            cancel = ok && m.cancelled();
        }
        const double t1 = now_s();
        held.assign(r.ids.begin(), r.ids.begin() + (long) m.end());
        if (!ok) {
            std::printf("ERR %s\n", err.c_str());
            std::fprintf(stderr, "ik2: %s\n", err.c_str());
            m.reset();
            held.clear();
        }
        if (!ok || cancel) {
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0\n", n, (t1 - t0) * 1e3, from);
            std::fflush(stdout);
            continue;
        }
        pp(n);
        std::mt19937_64 rng(r.seed != 0 ? r.seed : 0x9E3779B97F4A7C15ull * counter);
        const char* finish = "length";
        long long produced = 0;
        const int64_t cpu0 = st.cpu_experts, ex0 = st.experts;
        dr.start();
        const Drafting d0 = dr;
        const bool spec = dr.on && (r.spec < 0 ? spec_default : r.spec != 0);
        if (dr.on) m.set_draft_cum(cum);
        // the answer's first token from the prompt's last row; then rounds: the token in hand read (with drafts, when
        // the drafter has some worth a window), the tokens after it given out
        int tok = want_lg ? sample(lg.data(), m.vocab(), r, held, rng) : m.top();
        bool more = r.max_new > 0;
        auto emit = [&](int t) {   // false: the answer ends with it
            std::printf("T %d\n", t);
            ++produced;
            if (is_stop(t)) {   // (the end token goes out too: the server's finish reason comes from it)
                finish = "stop";
                std::fflush(stdout);
                return false;
            }
            if (produced % 8 == 0) {   // (maya-ds4's: a STAT line every 8 tokens)
                const double ms = (now_s() - t1) * 1e3;
                const double share = (double) (st.cpu_experts - cpu0) / (double) std::max<int64_t>(1, st.experts - ex0);
                std::printf("STAT tok_s=%.2f ms_tok=%.2f vram_hit=%.4f cpu=%.3f cpu_share=%.4f swaps=%lld", 1e3 * (double) produced / ms,
                            ms / (double) produced, 1.0 - share, 0.0, share, (long long) st.swaps);
                if (dr.on) {
                    std::printf(" draft_rounds=%lld draft_ms=%.3f verify_ms=%.3f", dr.drafted - d0.drafted,
                                dr.draft_ms - d0.draft_ms, dr.verify_ms - d0.verify_ms);
                    if (dr.offered > d0.offered)
                        std::printf(" drafts=%.3f draft_acc=%lld draft_off=%lld",
                                    (double) (dr.accepted - d0.accepted) / (double) (dr.offered - d0.offered),
                                    dr.accepted - d0.accepted, dr.offered - d0.offered);
                }
                std::printf("\n");
            }
            std::fflush(stdout);
            return true;
        };
        if (more) more = emit(tok);
        while (more) {
            if (lines.stop.load() || lines.wrap.load()) {
                held.push_back(tok);
                finish = "cancel";
                break;
            }
            if (produced >= r.max_new || m.end() + 1 > m.max_ctx()) {
                held.push_back(tok);
                break;
            }
            held.push_back(tok);
            const int64_t at = m.end();
            if (!draft_round(m, dr, tok, r, held, rng, r.max_new - produced, spec, out, lgw, err)) {
                std::printf("ERR %s\n", err.c_str());
                std::fprintf(stderr, "ik2: %s\n", err.c_str());
                m.reset();
                held.clear();
                finish = "cancel";
                break;
            }
            for (size_t i = 0; i < out.size() && more; ++i) {
                more = emit(out[i]);
                if (i + 1 < out.size()) {
                    if (more) held.push_back(out[i]);   // (a kept draft: the context read it)
                    else if (!m.keep((int) i + 1, err)) {   // (an end token among the kept drafts: the context ends before it)
                        std::fprintf(stderr, "ik2: %s\n", err.c_str());
                        m.reset();
                        held.clear();
                    }
                }
            }
            (void) at;
            tok = out.back();
        }
        held.resize((size_t) std::min<int64_t>((int64_t) held.size(), m.end()));   // (the last token given out is not read yet)
        const double t2 = now_s();
        std::printf("DONE %lld %zu %.1f %.1f %s %lld %lld %zu 0 0\n", produced, n, (t1 - t0) * 1e3, (t2 - t1) * 1e3, finish,
                    dr.accepted - d0.accepted, dr.offered - d0.offered, from);
        std::fflush(stdout);
        m.vram_notes("after a request");
        std::fprintf(stderr,
                     "maya ik2 serve: prompt %zu tokens = %zu reused (%zu held in common) + %zu read in %.0f ms (%.1f "
                     "tok/s), %lld generated in %.0f ms (%.1f tok/s), %s; VRAM hit %.1f%%, CPU lane %.1f%%, %lld cache moves "
                     "so far\n",
                     n, from, lcp, n - from, (t1 - t0) * 1e3, t1 > t0 ? (double) (n - from) / (t1 - t0) : 0.0, produced,
                     (t2 - t1) * 1e3, t2 > t1 ? (double) produced / (t2 - t1) : 0.0, finish,
                     100.0 - 100.0 * (double) (st.cpu_experts - cpu0) / (double) std::max<int64_t>(1, st.experts - ex0),
                     100.0 * (double) (st.cpu_experts - cpu0) / (double) std::max<int64_t>(1, st.experts - ex0),
                     (long long) st.swaps);
        if (dr.on) {
            std::fprintf(stderr, "maya ik2 serve: drafts %lld of %lld kept in %lld drafted rounds of %lld (%.2f tokens a pass), "
                                 "k histogram", dr.accepted - d0.accepted, dr.offered - d0.offered, dr.drafted - d0.drafted,
                         dr.rounds - d0.rounds, (double) (dr.tokens - d0.tokens) / (double) std::max(1ll, dr.passes - d0.passes));
            for (int k = 0; k < 8; ++k) std::fprintf(stderr, " %lld", dr.khist[k] - d0.khist[k]);
            std::fprintf(stderr, "; draft %.1f ms a round, a pass %.1f + %.1f ms a row, rests %lld\n", dr.t_draft, dr.a, dr.b,
                         dr.rested - d0.rested);
        }
        if (!m.save_profile(err)) std::fprintf(stderr, "ik2: the expert profile: %s\n", err.c_str());
    }
    return 0;
}

// ---- SEVERAL CONVERSATIONS AT ONCE (--parallel <n>: READY ... multi=<n>, maya-ds4's protocol).  GEN lines carry
// rid=<id>, and every line a request causes carries it after its keyword (T <rid> <id>, PP <rid> ..., DONE <rid> ...,
// ERR <rid> ...); STOP <rid> ends that request, a bare STOP every one.  Each request runs in one of the n conversations
// the model holds (Model::bind): the idle one whose tokens its prompt shares the most of, else the least recently used.
// A prompt is read whole when its turn comes (the others' answers wait for it); between two prompts every running
// request writes: one alone in its rounds as with one conversation (the drafter's windows), two or more a token each
// in one step through the layers (Model::step_multi: the weights read once), each request's logit bias and sampler
// on its own row.  STAT lines are the engine's (untagged).
long long gen_rid(const std::string& line) {
    std::istringstream is(line.substr(4));
    for (std::string tok; is >> tok;)
        if (tok.rfind("rid=", 0) == 0) return std::atoll(tok.c_str() + 4);
    return -1;
}
int serve_multi(ik2hip::Model& m, const std::vector<int>& stops, int short_n, bool spec_default, bool drafting,
                float cum_default, int rest0) {
    for (const std::string& l : m.info_lines()) std::printf("NOTE %s\n", l.c_str());
    const ik2hip::Stats& st = m.stats();
    const int N = m.seq_count(), V = m.vocab();
    std::printf("INFO engine=ik2 engine_kind=ik2-hip model=Inkling-2-Small ctx=%lld vram_gb=%.2f dspark=%d mtp=0 "
                "experts=%d experts_vram=%d\n", (long long) m.max_ctx(), st.vram_gb + m.dspark_vram_gb(),
                m.dspark() && drafting && spec_default ? 1 : 0, st.experts_all, st.experts_vram);
    std::printf("INFO logit_bias=1\n");
    std::printf("READY %lld stop multi=%d\n", (long long) m.max_ctx(), N);
    std::fflush(stdout);
    std::fprintf(stderr, "maya ik2 serve: %d conversations at once (--parallel): GEN needs rid=<id>; STOP <rid>\n", N);
    Lines lines;
    lines.multi = true;
    lines.start();
    auto is_stop = [&](int t) { return std::find(stops.begin(), stops.end(), t) != stops.end(); };
    struct Act {
        long long rid = -1;
        mm3::sample::Req r;
        float cum = 0.5f;
        std::vector<std::pair<int, float>> bias;
        int seq = 0;
        bool read = false, done = false;
        size_t from = 0;
        double t0 = 0, t1 = 0;
        long long produced = 0;
        const char* finish = "length";
        int tok = -1;
        bool spec = false, want_lg = false;
        std::mt19937_64 rng;
        Drafting d0;
        int64_t cpu0 = 0, ex0 = 0;
    };
    std::vector<std::vector<int>> held((size_t) N);
    std::vector<long long> seq_rid((size_t) N, -1);
    std::vector<uint64_t> seq_used((size_t) N, 0);
    std::vector<Drafting> dr((size_t) N);
    for (Drafting& d : dr) {
        d.on = m.dspark() && drafting;
        if (rest0 >= 0) d.rest0 = rest0;
    }
    uint64_t clock = 0;
    unsigned long long counter = 0;
    std::vector<std::unique_ptr<Act>> acts;
    std::deque<std::string> waiting;
    std::vector<float> lg((size_t) V), rows, lgw;
    std::vector<int> out;
    std::string err;
    bool quit = false;
    auto done_line = [&](Act& a) {
        const double t2 = now_s();
        const Drafting& d = dr[(size_t) a.seq];
        std::printf("DONE %lld %lld %zu %.1f %.1f %s %lld %lld %zu 0 0\n", a.rid, a.produced, a.r.ids.size(),
                    (a.t1 - a.t0) * 1e3, a.read ? (t2 - a.t1) * 1e3 : 0.0, a.finish, d.accepted - a.d0.accepted,
                    d.offered - a.d0.offered, a.from);
        std::fflush(stdout);
        std::fprintf(stderr, "maya ik2 serve: rid %lld in conversation %d: prompt %zu tokens (%zu reused), %lld "
                             "generated in %.0f ms (%.1f tok/s), %s\n", a.rid, a.seq, a.r.ids.size(), a.from, a.produced,
                     (t2 - a.t1) * 1e3, t2 > a.t1 && a.read ? (double) a.produced / (t2 - a.t1) : 0.0, a.finish);
        held[(size_t) a.seq].resize((size_t) std::min<int64_t>((int64_t) held[(size_t) a.seq].size(), m.end_of(a.seq)));
        seq_rid[(size_t) a.seq] = -1;
        lines.forget(a.rid);
        a.done = true;
        if (!m.save_profile(err)) std::fprintf(stderr, "ik2: the expert profile: %s\n", err.c_str());
    };
    auto emit = [&](Act& a, int t) {   // false: the answer ends with it
        std::printf("T %lld %d\n", a.rid, t);
        ++a.produced;
        if (is_stop(t)) {
            a.finish = "stop";
            std::fflush(stdout);
            return false;
        }
        if (a.produced % 8 == 0) {
            const double ms = (now_s() - a.t1) * 1e3;
            const double share = (double) (st.cpu_experts - a.cpu0) / (double) std::max<int64_t>(1, st.experts - a.ex0);
            std::printf("STAT tok_s=%.2f ms_tok=%.2f vram_hit=%.4f cpu=%.3f cpu_share=%.4f swaps=%lld\n",
                        1e3 * (double) a.produced / ms, ms / (double) a.produced, 1.0 - share, 0.0, share,
                        (long long) st.swaps);
        }
        std::fflush(stdout);
        return true;
    };
    // a GEN line into a free conversation (false: none is free - it waits)
    auto place = [&](const std::string& line) -> bool {
        auto a = std::make_unique<Act>();
        a->rid = gen_rid(line);
        a->cum = cum_default;
        if (a->rid < 0) {
            std::printf("ERR this engine runs several conversations: GEN needs rid=<id>\n");
            std::fflush(stdout);
            return true;
        }
        if (!parse_gen(line, a->r, a->cum, a->bias)) {
            std::printf("ERR %lld expected: GEN <max_new> [keys] <id,id,...>\n", a->rid);
            std::printf("DONE %lld 0 0 0.0 0.0 cancel 0 0 0 0 0\n", a->rid);
            std::fflush(stdout);
            return true;
        }
        const size_t n = a->r.ids.size();
        bool bad = false;
        for (int t : a->r.ids) bad = bad || t < 0 || t >= V;
        if (bad || (long long) n + std::max(0ll, a->r.max_new) > m.max_ctx()) {
            if (bad) std::printf("ERR %lld a prompt token is outside the vocabulary (%d)\n", a->rid, V);
            else
                std::printf("ERR %lld the prompt (%zu tokens) and max_new %lld are more than the context %lld\n", a->rid, n,
                            a->r.max_new, (long long) m.max_ctx());
            std::printf("DONE %lld 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", a->rid, n);
            std::fflush(stdout);
            return true;
        }
        int best = -1;
        size_t best_l = 0;
        for (int s2 = 0; s2 < N; ++s2) {   // (the idle one sharing the most; ties: the least recently used)
            if (seq_rid[(size_t) s2] >= 0) continue;
            size_t l2 = 0;
            const std::vector<int>& h = held[(size_t) s2];
            while (l2 < h.size() && l2 < n - 1 && h[l2] == a->r.ids[l2]) ++l2;
            if (best < 0 || l2 > best_l || (l2 == best_l && seq_used[(size_t) s2] < seq_used[(size_t) best])) {
                best = s2;
                best_l = l2;
            }
        }
        if (best < 0) return false;
        a->seq = best;
        seq_rid[(size_t) best] = a->rid;
        seq_used[(size_t) best] = ++clock;
        ++counter;
        a->rng.seed(a->r.seed != 0 ? a->r.seed : 0x9E3779B97F4A7C15ull * counter);
        acts.push_back(std::move(a));
        return true;
    };
    // a request's prompt read in its conversation, its first token given out
    auto read_prompt = [&](Act& a) {
        a.read = true;
        a.t0 = now_s();
        std::vector<int>& h = held[(size_t) a.seq];
        const size_t n = a.r.ids.size();
        bool ok = m.bind(a.seq, err) && m.set_logit_bias(a.bias, err);
        size_t lcp = 0;
        if (ok) {
            while (lcp < h.size() && lcp < n - 1 && h[lcp] == a.r.ids[lcp]) ++lcp;
            a.from = (size_t) m.rewind((int64_t) lcp);
            h.resize(a.from);
        }
        auto pp = [&](size_t read) {
            const double ms = (now_s() - a.t0) * 1e3;
            std::printf("PP %lld %zu %zu %.1f %.1f %zu\n", a.rid, read, n, ms,
                        ms > 0 ? 1e3 * (double) (read - a.from) / ms : 0.0, a.from);
            std::fflush(stdout);
        };
        a.want_lg = a.r.temperature > 0.0f || mm3::sample::penalised(a.r);
        bool cancel = false;
        if (ok) {
            pp(a.from);
            if (n - a.from <= (size_t) short_n) {
                for (size_t i = a.from; i < n && ok && !cancel; ++i) {
                    ok = m.forward(&a.r.ids[i], 1, (int64_t) i, i + 1 == n && a.want_lg ? lg.data() : nullptr, err);
                    cancel = lines.stopped(a.rid);
                }
                if (ok && cancel && (size_t) m.end() == n) cancel = false;
            } else {
                const long long rid = a.rid;
                m.set_progress([&, rid](int64_t done) {
                    pp(a.from + (size_t) done);
                    return !lines.stopped(rid);
                });
                ok = m.forward(&a.r.ids[a.from], (int) (n - a.from), (int64_t) a.from, a.want_lg ? lg.data() : nullptr, err);
                m.set_progress(nullptr);
                cancel = ok && m.cancelled();
            }
        }
        a.t1 = now_s();
        h.assign(a.r.ids.begin(), a.r.ids.begin() + (long) m.end());
        if (!ok) {
            std::printf("ERR %lld %s\n", a.rid, err.c_str());
            std::fprintf(stderr, "ik2: %s\n", err.c_str());
            m.reset();
            h.clear();
        }
        if (!ok || cancel) {
            a.finish = "cancel";
            a.produced = 0;
            done_line(a);
            return;
        }
        pp(n);
        Drafting& d = dr[(size_t) a.seq];
        d.start();
        a.d0 = d;
        a.spec = d.on && (a.r.spec < 0 ? spec_default : a.r.spec != 0);
        a.cpu0 = st.cpu_experts;
        a.ex0 = st.experts;
        a.tok = a.want_lg ? sample(lg.data(), V, a.r, h, a.rng) : m.top();
        if (a.r.max_new <= 0 || !emit(a, a.tok)) done_line(a);
    };
    // the end of an answer before its token in hand is read: the context holds it as given out
    auto room_left = [&](Act& a) {
        if (lines.stopped(a.rid)) {
            held[(size_t) a.seq].push_back(a.tok);
            a.finish = "cancel";
            done_line(a);
            return false;
        }
        if (a.produced >= a.r.max_new || m.end_of(a.seq) + 1 > m.max_ctx()) {
            held[(size_t) a.seq].push_back(a.tok);
            done_line(a);
            return false;
        }
        return true;
    };
    while (!quit) {
        // the lines that came (a wait for one when nothing runs)
        for (std::string line;;) {
            bool idle = waiting.empty();
            for (const auto& a : acts) idle = idle && a->done;
            if (idle && !lines.next(line)) {
                quit = true;
                break;
            }
            if (!idle && !lines.try_next(line)) {
                if (lines.ended()) quit = true;
                break;
            }
            if (line == "QUIT") {
                quit = true;
                break;
            }
            if (line.empty()) continue;
            if (line.rfind("GEN ", 0) != 0) {
                std::printf("ERR unsupported command '%s' (GEN, STOP, WRAP and QUIT)\n", line.substr(0, 40).c_str());
                std::fflush(stdout);
                continue;
            }
            waiting.push_back(line);
            if (idle) break;
        }
        if (quit) break;
        if (lines.stop.exchange(false) | lines.wrap.exchange(false))   // (a bare STOP: every request running now)
            for (auto& a : acts)
                if (!a->done) {
                    if (a->read) held[(size_t) a->seq].push_back(a->tok);
                    a->finish = "cancel";
                    done_line(*a);
                }
        acts.erase(std::remove_if(acts.begin(), acts.end(), [](const std::unique_ptr<Act>& a) { return a->done; }),
                   acts.end());
        while (!waiting.empty() && place(waiting.front())) waiting.pop_front();
        // a prompt whose turn it is (the first placed)
        for (auto& a : acts)
            if (!a->done && !a->read) {
                read_prompt(*a);
                break;
            }
        // a round of the answers
        std::vector<Act*> run;
        for (auto& a : acts)
            if (!a->done && a->read && room_left(*a)) run.push_back(a.get());
        if (run.empty()) continue;
        if (run.size() == 1) {   // (alone: as one conversation - its rounds with drafts)
            Act& a = *run[0];
            Drafting& d = dr[(size_t) a.seq];
            std::vector<int>& h = held[(size_t) a.seq];
            if (!m.bind(a.seq, err) || !m.set_logit_bias(a.bias, err)) {
                std::printf("ERR %lld %s\n", a.rid, err.c_str());
                a.finish = "cancel";
                done_line(a);
                continue;
            }
            if (d.on) m.set_draft_cum(a.cum);
            h.push_back(a.tok);
            if (!draft_round(m, d, a.tok, a.r, h, a.rng, a.r.max_new - a.produced, a.spec, out, lgw, err)) {
                std::printf("ERR %lld %s\n", a.rid, err.c_str());
                std::fprintf(stderr, "ik2: %s\n", err.c_str());
                m.reset();
                h.clear();
                a.finish = "cancel";
                done_line(a);
                continue;
            }
            bool more = true;
            for (size_t i = 0; i < out.size() && more; ++i) {
                more = emit(a, out[i]);
                if (i + 1 < out.size()) {
                    if (more) h.push_back(out[i]);
                    else if (!m.keep((int) i + 1, err)) {
                        std::fprintf(stderr, "ik2: %s\n", err.c_str());
                        m.reset();
                        h.clear();
                    }
                }
            }
            a.tok = out.back();
            if (!more) done_line(a);
            continue;
        }
        // two or more: a token of each through the layers at once
        const int n = (int) run.size();
        std::vector<int> sq((size_t) n), ids((size_t) n);
        for (int i = 0; i < n; ++i) {
            sq[(size_t) i] = run[(size_t) i]->seq;
            ids[(size_t) i] = run[(size_t) i]->tok;
            held[(size_t) run[(size_t) i]->seq].push_back(run[(size_t) i]->tok);
        }
        rows.resize((size_t) n * V);
        if (!m.step_multi(sq.data(), ids.data(), n, rows.data(), err)) {
            std::fprintf(stderr, "ik2: %s\n", err.c_str());
            for (Act* a : run) {
                std::printf("ERR %lld %s\n", a->rid, err.c_str());
                a->finish = "cancel";
                done_line(*a);
            }
            continue;
        }
        for (int i = 0; i < n; ++i) {
            Act& a = *run[(size_t) i];
            float* row = rows.data() + (size_t) i * V;
            for (const auto& [id, b] : a.bias)
                if (id >= 0 && id < V) row[id] += b;
            a.tok = sample(row, V, a.r, held[(size_t) a.seq], a.rng);
            if (!emit(a, a.tok)) done_line(a);
        }
    }
    for (auto& a : acts)
        if (!a->done) {
            a->finish = "cancel";
            done_line(*a);
        }
    return 0;
}

int usage() {
    std::fprintf(stderr,
                 "maya-ik2 (AMD build) --pack <dir> | --model <first .gguf> (--tokens <ids> [--n-predict N] | --serve\n"
                 "           | --parity <reference dir> [--parity <dir> ...])\n"
                 "  --pack DIR          the pack of the GGUF set (tools/ik2/pack_gguf.py: <the set's folder>/maya-ik2)\n"
                 "  --model F           the GGUF set's first shard, as the CUDA build takes it: its folder's pack\n"
                 "  --gpus N | 0,1,..   the first N GPUs, as the CUDA build's --gpus N, or these GPUs (default: all)\n"
                 "  --max-context N     positions the global layers' caches hold (default 32768)\n"
                 "  --reserve-mib N     VRAM each GPU keeps free, MiB (default 600)\n"
                 "  --vram-experts X    the share of a layer's experts in VRAM at most (default 1.0)\n"
                 "  --layer-split a,b,c the first layer of each GPU after the first (auto: by the layers' bytes, the default)\n"
                 "  --expert-profile F  the expert profile read (the most routed in VRAM) and written back (default: the\n"
                 "                      pack's expert-profile.bin; none: no profile)\n"
                 "  --profile-readonly  the profile is read, never written back\n"
                 "  --profile-out F     this run's routing alone, written to F\n"
                 "  --cpu-threads N     the CPU lane's threads (default 0: a thread a physical core; N: at most N)\n"
                 "  --log-floor N       log_scaling_n_floor in place of the pack's\n"
                 "  --layers N          (tests) the first N layers only\n"
                 "  --parity-max N      (parity) the first N positions only\n"
                 "  --parity-detail     (parity) a line a position and tensor\n"
                 "  --parity-decode     (parity) the prompt token by token (the decode path), not the prompt path\n"
                 "  --prefill-chunk N   a prompt's tokens a chunk (default 512; 1: token by token; --chunk N the same)\n"
                 "  --parity-force-picks  (parity) each reference a second time, every layer taking the reference's\n"
                 "                      picks (what is left is not the router's near-ties)\n"
                 "  --no-expert-cache   the experts stay where the load put them (no cache moves)\n"
                 "  --parallel N        (serve) N conversations at once (1 .. 8, maya-ds4's): each its own caches and\n"
                 "                      drafter state in VRAM; GEN rid=<id>, STOP <rid>\n"
                 "  --multi-check DIR   (tests, with --parallel N) N conversations of a reference's prompt decoded a token\n"
                 "                      each alone and through multi steps: each row the same bits, the step times\n"
                 "  --mmap-experts on|off|auto  the CPU lane's experts (maya-ds4's): read in place from the model's files\n"
                 "                      (on), or the most routed that fit copied into RAM and the rest in place (off;\n"
                 "                      auto, the default, the same here: maya-ds4's copies none unless all fit)\n"
                 "  --tokens-file F     (tests) a prompt a line; with --profile-out: a corpus' routings\n"
                 "  --serve             the server's line protocol on stdin / stdout (docs/INKLING2_AMD.md, \"Serving\")\n"
                 "  --rewind N          positions a rewind may go back without reading the prompt again (default 4096)\n"
                 "  --short-prompt N    (serve) a prompt's unread tokens up to N take the decode path (default 4: measured,\n"
                 "                      the prompt path is faster from 5 tokens on)\n"
                 "  --eos a,b           (serve) the ids that end an answer (default 200006,199999)\n"
                 "  --window-test DIR   (tests) a verify window of 9 rows after a reference's prompt against 9 token steps,\n"
                 "                      a window cut back, a window's time for 1 .. 9 rows\n"
                 "  --rewind-test DIR   (tests) a reference's prompt read, rewound and read on, against a cold read\n"
                 "  --short-bench       (tests, with --tokens) short suffixes on the prompt path and the decode path\n"
                 "  --dspark on|off|none  the DSpark drafter (maya-ds4's): speculative decoding with it (on, the default),\n"
                 "                      loaded but a request drafts only when it asks (off: GEN spec=1), or not loaded\n"
                 "                      (none).  idle (tests; maya-ds4 has none): loaded and fed, never asked for drafts\n"
                 "  --dspark-index DIR  its index (tools/ik2/convert_dspark.py's maya-ik2-dspark-q8_0; none given:\n"
                 "                      decoding without drafts)\n"
                 "  --dspark-cum F      the floor on the running product of the drafts' probabilities (default 0.5; a\n"
                 "                      request's dspark_cum key)\n"
                 "  --temperature F / --seed N   (tests, with --tokens) sampled decoding\n"
                 "  --prompts F         (tests) a prompt a line (ids), each decoded from an empty context\n"
                 "  --ab                (tests) each prompt twice, without drafts and with them: the speeds, the tokens compared\n"
                 "  --sweep VAR=a,b,..  (tests, with --prompts) each prompt's drafted decode with each value of the IK2_HIP_*\n"
                 "                      override VAR (--sweep-rounds N times, 2 by default, the order reversed every other)\n"
                 "  --window-bits       (tests, with --window-test) which part of a window differs from the token steps\n"
                 "  --verbose\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    ik2hip::Options o;
    std::vector<std::string> parities;
    std::string tokens;
    // (short_n: --short-bench after the 2.7k-token prompt, ms, prompt path / decode path: 3 tokens 118.7 / 95.9, 4 tokens
    //  125.9 / 123.2, 5 tokens 129.8 / 148.1 - means of 5, each within 3 ms; docs/INKLING2_AMD.md)
    int n_predict = 0, repeat = 1, short_n = 4;
    bool serving = false, short_bench = false, window_bits = false;
    std::string dspark = "on", dspark_index;   // (the drafter's index: the config names it, the setup writes it)
    float dspark_cum = 0.5f, temperature = 0.0f;
    unsigned long long seed = 0;
    // the ids that end an answer: 200006, <|content_model_end_sampling|> - the model's end of sequence (the GGUF's
    // tokenizer.ggml.eos_token_id, config.json's eos_token_id), which ends its turn - and 199999, <|endoftext|>
    std::vector<int> stops = {200006, 199999};
    std::string rewind_test, tokens_file, window_test, prompts_file, multi_check;
    bool ab = false;
    // (tests) --sweep VAR=v1,v2,..: the drafted decode of each prompt with each value of the IK2_HIP_* override VAR,
    // --sweep-rounds times (ABBA order); several --sweep: one after the other, each VAR unset after its own
    std::vector<std::pair<std::string, std::vector<std::string>>> sweeps;
    int sweep_rounds = 2;
    std::vector<int> rest0_list;   // (tests: prompt i's Drafting::rest0)
    std::vector<int> try_chunks;
    int64_t parity_max = -1;
    bool decode_only = false;   // (parity) the prompt token by token (the decode path) instead of the prompt path
    bool detail = false;   // (parity) a line a position and tensor
    bool force_only = false;   // (parity) that second run alone
    bool force_opt = false;   // (parity) a second run in which every layer takes the reference's picks
    std::string model_gguf;   // (--model: the CUDA build's way to name the model - the pack beside it)
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--pack") o.pack = val();
        else if (a == "--model") model_gguf = val();   // (maya-ik2's CUDA build's flag: the GGUF set's first shard)
        else if (a == "--gpus") {   // (a count, as the CUDA build's --gpus N: the first N visible GPUs; or a list)
            const std::string v = val();
            if (v.find(',') == std::string::npos && !v.empty() && std::all_of(v.begin(), v.end(), ::isdigit)) {
                o.gpus.clear();
                for (int g = 0; g < std::atoi(v.c_str()); ++g) o.gpus.push_back(g);
            } else {
                o.gpus = int_list(v);
            }
        }
        else if (a == "--max-context" || a == "--ctx") o.max_ctx = std::atoll(val().c_str());
        else if (a == "--reserve-mib") o.reserve_mib = std::atoll(val().c_str());
        else if (a == "--vram-experts") o.vram_experts = std::atof(val().c_str());
        else if (a == "--layer-split") {   // (auto: the plan's own split, by the layers' bytes)
            const std::string v = val();
            if (v != "auto") o.layer_split = int_list(v);
        }
        else if (a == "--expert-profile") o.profile = val();
        else if (a == "--profile-out") o.profile_out = val();
        else if (a == "--profile-readonly") o.profile_readonly = true;
        else if (a == "--cpu-threads") o.cpu_threads = std::atoi(val().c_str());
        else if (a == "--log-floor") o.log_floor = std::atof(val().c_str());
        else if (a == "--layers") o.layers = std::atoi(val().c_str());
        else if (a == "--parity") parities.push_back(val());
        else if (a == "--parity-max") parity_max = std::atoll(val().c_str());
        else if (a == "--tokens") tokens = val();
        else if (a == "--n-predict") n_predict = std::atoi(val().c_str());
        else if (a == "--repeat") repeat = std::atoi(val().c_str());
        else if (a == "--try-chunks") try_chunks = int_list(val());
        else if (a == "--verbose") o.verbose = true;
        else if (a == "--serve") serving = true;
        else if (a == "--no-expert-cache") o.expert_cache = false;
        else if (a == "--parallel") {   // (maya-ds4's: --serve, 1 .. 8 conversations at once)
            o.seqs = std::atoi(val().c_str());
            if (o.seqs < 1 || o.seqs > 8) {
                std::fprintf(stderr, "--parallel: 1 .. 8 conversations\n");
                return 2;
            }
        }
        else if (a == "--multi-check") multi_check = val();
        else if (a == "--mmap-experts") {
            const std::string v = val();
            if (v != "on" && v != "off" && v != "auto") {
                std::fprintf(stderr, "--mmap-experts: on, off or auto, not %s\n", v.c_str());
                return 2;
            }
            o.mmap_experts = v == "on" ? 1 : v == "off" ? 0 : -1;
        }
        else if (a == "--tokens-file") tokens_file = val();
        else if (a == "--rewind") o.rewind = std::atoi(val().c_str());
        else if (a == "--short-prompt") short_n = std::atoi(val().c_str());
        else if (a == "--eos") stops = int_list(val());
        else if (a == "--rewind-test") rewind_test = val();
        else if (a == "--window-test") window_test = val();
        else if (a == "--window-bits") window_bits = true;
        else if (a == "--prompts") prompts_file = val();
        else if (a == "--ab") ab = true;
        else if (a == "--sweep") {
            const std::string v = val();
            const size_t eq = v.find('=');
            if (eq == std::string::npos) return usage();
            sweeps.push_back({v.substr(0, eq), {}});
            for (size_t i = eq + 1; i <= v.size();) {
                const size_t c = std::min(v.find(',', i), v.size());
                sweeps.back().second.push_back(v.substr(i, c - i));
                i = c + 1;
            }
        }
        else if (a == "--sweep-rounds") sweep_rounds = std::max(1, std::atoi(val().c_str()));
        else if (a == "--dspark-rest0") rest0_list = int_list(val());
        else if (a == "--short-bench") short_bench = true;
        else if (a == "--parity-detail") detail = true;
        else if (a == "--parity-decode") decode_only = true;
        else if (a == "--prefill-chunk" || a == "--chunk") o.prefill_chunk = std::atoi(val().c_str());
        else if (a == "--parity-force-picks") force_opt = true;
        else if (a == "--parity-force-only") force_opt = force_only = true;
        else if (a == "--dspark") dspark = val();
        else if (a == "--dspark-index") dspark_index = val();
        else if (a == "--dspark-cum") dspark_cum = std::strtof(val().c_str(), nullptr);
        else if (a == "--temperature") temperature = std::strtof(val().c_str(), nullptr);
        else if (a == "--seed") seed = std::strtoull(val().c_str(), nullptr, 10);
        else return usage();
    }
    if (o.pack.empty() && !model_gguf.empty()) {   // (the setup's layout: <the set's folder>/maya-ik2)
        const std::filesystem::path dir = std::filesystem::path(model_gguf).parent_path();
        if (std::filesystem::exists(dir / "maya-ik2" / "inkling2.txt"))
            o.pack = (dir / "maya-ik2").string();
        if (o.pack.empty()) {
            std::fprintf(stderr, "maya-ik2: no pack beside %s - tools/ik2/pack_gguf.py --model %s builds it (or "
                                 "./maya.sh --setup)\n", model_gguf.c_str(), dir.string().c_str());
            return 2;
        }
    }
    if (o.pack.empty() || (parities.empty() && tokens.empty() && !serving && rewind_test.empty() && tokens_file.empty() &&
                           window_test.empty() && prompts_file.empty() && multi_check.empty()))
        return usage();
    std::string err;
    // (--tokens-file, a corpus' routings - maya-ds4's --profile-run: the pack's profile only read, the run's counts
    //  to --profile-out alone)
    if (!tokens_file.empty()) o.profile_readonly = true;
    // the drafter: with --serve and the plain decode (the tests run without it); its index missing: none, said so
    // (--dspark: on drafts, off loads it for the requests that ask, none leaves it out; idle: loaded, never asked)
    if (dspark != "on" && dspark != "off" && dspark != "none" && dspark != "idle") return usage();
    const bool drafting = dspark != "none" && parities.empty() && rewind_test.empty() && !short_bench &&
                          tokens_file.empty();
    if (drafting) {
        if (dspark_index.empty()) {
            std::fprintf(stderr, "maya-ik2: no drafter given (--dspark-index): decoding without drafts\n");
        } else if (std::ifstream(dspark_index + "/dspark.txt").good()) {
            o.dspark_index = dspark_index;
            o.dspark_cum = dspark_cum;
        } else {
            std::fprintf(stderr, "maya-ik2: no drafter at %s: decoding without drafts\n", dspark_index.c_str());
        }
    }
    for (const std::string& d : parities) {   // (a wrong folder: said before the load)
        Ref r;
        if (!r.load(d, err)) {
            std::fprintf(stderr, "maya-ik2: %s\n", err.c_str());
            return 1;
        }
    }
    ik2hip::Model model;
    if (!model.load(o, err)) {
        std::fprintf(stderr, "maya-ik2: %s\n", err.c_str());
        return 1;
    }
    const bool swap_opt = o.expert_cache;
    if (serving && model.seq_count() > 1)
        return serve_multi(model, stops, short_n, dspark == "on", drafting && dspark != "idle", dspark_cum,
                           rest0_list.empty() ? -1 : rest0_list[0]);
    if (serving)
        return serve(model, stops, short_n, dspark == "on", drafting && dspark != "idle", dspark_cum,
                     rest0_list.empty() ? -1 : rest0_list[0]);
    for (const std::string& s : model.info_lines()) std::printf("%s\n", s.c_str());
    std::printf("loaded in %.1f s, %.1f GB of VRAM\n", model.stats().load_s, model.stats().vram_gb);
    if (model.dspark()) std::printf("the DSpark drafter: %.2f GB of VRAM on the last GPU%s\n", model.dspark_vram_gb(),
                                    dspark == "on" ? "" : " (fed, not asked)");
    if (!tokens_file.empty()) {
        // --tokens-file F: a prompt a line (ids), each read from an empty context and --n-predict tokens decoded after
        // it - with --profile-out the routings of a corpus (data/expert-profile-ik2.bin)
        std::ifstream tf(tokens_file);
        int np = 0;
        for (std::string line; std::getline(tf, line);) {
            std::vector<int> ids = int_list(line);
            if (ids.empty()) continue;
            if ((int64_t) ids.size() + n_predict > model.max_ctx()) ids.resize((size_t) (model.max_ctx() - n_predict));
            model.reset();
            const double t0 = now_s();
            if (!model.forward(ids.data(), (int) ids.size(), 0, nullptr, err)) {
                std::fprintf(stderr, "maya-ik2: %s\n", err.c_str());
                return 1;
            }
            const double t1 = now_s();
            int n = 0;
            for (; n < n_predict; ++n) {
                const int tok = model.top();
                if (tok == model.eos() || !model.forward(&tok, 1, model.end(), nullptr, err)) break;
            }
            std::printf("prompt %d: %zu tokens, %.0f tok/s; %d decoded, %.1f tok/s\n", ++np, ids.size(),
                        ids.size() / (t1 - t0), n, n / std::max(1e-9, now_s() - t1));
            std::fflush(stdout);
        }
        if (!model.save_profile(err)) std::fprintf(stderr, "maya-ik2: %s\n", err.c_str());
        return 0;
    }
    if (!window_test.empty()) {
        // --window-test <reference dir>: after its prompt, 9 greedy tokens a step each (their logits kept), then the
        // same 9 as one verify window from the same position: each row's argmax and logits against the token steps';
        // a window cut back to 4 rows and read on; and a window's time for 1 .. 9 rows
        Ref rf;
        if (!rf.load(window_test, err)) return 1;
        const int W = model.window_max(), V = model.vocab();
        if (W < 2) {
            std::fprintf(stderr, "maya-ik2: no verify windows in this configuration\n");
            return 1;
        }
        auto cosine = [&](const float* a, const float* b) {
            double dot = 0, a2 = 0, b2 = 0;
            for (int i = 0; i < V; ++i) {
                dot += (double) a[i] * b[i];
                a2 += (double) a[i] * a[i];
                b2 += (double) b[i] * b[i];
            }
            return dot / std::sqrt(a2 * b2);
        };
        model.reset();
        if (!model.forward(rf.prompt.data(), (int) rf.prompt.size(), 0, nullptr, err)) return 1;
        const int64_t P = model.end();
        // (the experts stay where they are for the comparison: an expert the cache moves between the two passes is
        //  summed on the CPU lane in one and on the GPU in the other, which no window can undo - shown at the end)
        model.set_swap(false);
        std::vector<int> gt, tops((size_t) W);
        std::vector<float> lt((size_t) W * V), lw((size_t) W * V), l1((size_t) V);
        for (int t = 0; t < W; ++t) {
            gt.push_back(model.top());
            if (!model.forward(&gt.back(), 1, P + t, lt.data() + (size_t) t * V, err)) return 1;
        }
        int bad = 0, n_bits = 0;
        if (model.rewind(P) != P || !model.window(gt.data(), W, P, tops.data(), lw.data(), err)) {
            std::fprintf(stderr, "maya-ik2: the window at %lld: %s\n", (long long) P, err.c_str());
            return 1;
        }
        std::printf("window of %d rows at position %lld against the same tokens a step each:\n", W, (long long) P);
        for (int t = 0; t < W; ++t) {
            const double c = cosine(lt.data() + (size_t) t * V, lw.data() + (size_t) t * V);
            int am = 0;
            for (int i = 1; i < V; ++i)
                if (lt[(size_t) t * V + i] > lt[(size_t) t * V + am]) am = i;
            // (with the experts where they are a window's rows are the token steps' to the bit - 9 of 9,
            //  docs/INKLING2_AMD.md: the bound only has to catch a real difference)
            const bool ok = am == tops[(size_t) t] && c >= 0.9995;
            bad += !ok;
            n_bits += std::memcmp(lt.data() + (size_t) t * V, lw.data() + (size_t) t * V, sizeof(float) * (size_t) V) == 0;
            double dmax = 0;
            for (int i = 0; i < V; ++i) dmax = std::max(dmax, (double) std::fabs(lt[(size_t) t * V + i] - lw[(size_t) t * V + i]));
            std::printf("  row %d: next token %d / %d a step, logits cosine %.6f, max difference %.3g%s - %s\n", t,
                        tops[(size_t) t], am, c, dmax, dmax == 0 ? " (the same to the bit)" : "", ok ? "OK" : "DIFFERENT");
        }
        // cut back to 4 rows, then the 5th token a step: its logits against the token steps'
        if (!model.keep(4, err) || !model.forward(&gt[4], 1, P + 4, l1.data(), err)) {
            std::fprintf(stderr, "maya-ik2: keep: %s\n", err.c_str());
            return 1;
        }
        {
            const double c = cosine(lt.data() + (size_t) 4 * V, l1.data());
            const bool ok = c >= 0.9995;
            bad += !ok;
            std::printf("window cut back to 4 rows, the next token a step: logits cosine %.6f - %s\n", c, ok ? "OK" : "DIFFERENT");
        }
        if (window_bits) {
            // --window-bits: which part of a window differs from the token steps - the window again with a part of it
            // run a row at a time through a token's launches (Model::set_debug), each against the token steps' logits;
            // and the token steps themselves a second time (with --no-swap the same to the bit)
            auto cmp = [&](const char* what, const float* x) {
                int same = 0;
                double cmin = 1.0;
                for (int t = 0; t < W; ++t) {
                    same += std::memcmp(lt.data() + (size_t) t * V, x + (size_t) t * V, sizeof(float) * (size_t) V) == 0;
                    cmin = std::min(cmin, cosine(lt.data() + (size_t) t * V, x + (size_t) t * V));
                }
                std::printf("bits: %-52s %d of %d rows the same to the bit, least cosine %.6f\n", what, same, W, cmin);
                std::fflush(stdout);
            };
            if (model.rewind(P) != P) return 1;
            for (int t = 0; t < W; ++t)
                if (!model.forward(&gt[(size_t) t], 1, P + t, lw.data() + (size_t) t * V, err)) return 1;
            cmp("the token steps again", lw.data());
            static const std::pair<int, const char*> masks[] = {
                {0, "the window"}, {1, "... block-type GEMVs a column a launch"}, {2, "... experts a row at a time (plan form)"},
                {4, "... the CPU lane a row at a time"}, {8, "... F32 GEMVs a column a launch"}, {15, "... all four"}};
            for (const auto& [mask, what] : masks) {
                model.set_debug(mask);
                if (model.rewind(P) != P || !model.window(gt.data(), W, P, tops.data(), lw.data(), err)) return 1;
                cmp(what, lw.data());
            }
            model.set_debug(0);
        }
        std::printf("%d of %d rows the same to the bit as the token steps (the experts where they were)\n", n_bits, W);
        if (swap_opt) {   // (not judged: what the cache's moves between two passes do to the same token steps)
            model.set_swap(true);
            double cmin = 1.0;
            int same = 0;
            for (int rep2 = 0; rep2 < 2; ++rep2) {
                if (model.rewind(P) != P) return 1;
                for (int t = 0; t < W; ++t)
                    if (!model.forward(&gt[(size_t) t], 1, P + t, lw.data() + (size_t) t * V, err)) return 1;
            }
            for (int t = 0; t < W; ++t) {
                cmin = std::min(cmin, cosine(lt.data() + (size_t) t * V, lw.data() + (size_t) t * V));
                same += std::memcmp(lt.data() + (size_t) t * V, lw.data() + (size_t) t * V, sizeof(float) * (size_t) V) == 0;
            }
            std::printf("the token steps again with the expert cache moving experts in between: %d of %d rows the same to the "
                        "bit, least logits cosine %.6f (CPU-lane and GPU sums of an expert differ)\n", same, W, cmin);
        }
        std::printf("a window's time at position %lld, ms (5 runs each):", (long long) P);
        for (int T = 1; T <= W; ++T) {
            double best = 1e9, sum = 0;
            for (int r = 0; r < 5; ++r) {
                if (model.rewind(P) != P) return 1;
                const double t0 = now_s();
                if (!model.window(gt.data(), T, P, tops.data(), nullptr, err)) return 1;
                const double dt = (now_s() - t0) * 1e3;
                best = std::min(best, dt);
                sum += dt;
            }
            std::printf(" T=%d %.1f (best %.1f);", T, sum / 5, best);
        }
        std::printf("\n%s\n", bad ? "WINDOW DIFFERENT" : "WINDOW OK");
        return bad ? 3 : 0;
    }
    if (!multi_check.empty()) {
        // --multi-check <reference dir> (with --parallel N): conversation i reads the reference prompt's first
        // len - 37 i tokens, then K greedy tokens are decoded - first each conversation alone (a token's step), then
        // all of them through multi steps.  The expert cache off: each multi row's logits must be its own step's bits
        const int N = model.seq_count(), K = n_predict > 0 ? n_predict : 32, V = model.vocab();
        Ref rf;
        if (!rf.load(multi_check, err)) return 1;
        if (N < 2) {
            std::fprintf(stderr, "maya-ik2: --multi-check needs --parallel 2 or more\n");
            return 2;
        }
        model.set_swap(false);
        std::vector<std::vector<float>> alone((size_t) N * K);
        std::vector<std::vector<int>> toks_a((size_t) N), toks_m((size_t) N);
        std::vector<int> first((size_t) N);
        auto read = [&](int i) {
            const size_t len = rf.prompt.size() - 37 * (size_t) i;
            return model.bind(i, err) && (model.reset(), true) && model.forward(rf.prompt.data(), (int) len, 0, nullptr, err);
        };
        double t_alone = 0, t_multi = 0;
        std::vector<float> lw((size_t) V);
        for (int i = 0; i < N; ++i) {
            if (!read(i)) break;
            first[(size_t) i] = model.top();
            int t = first[(size_t) i];
            for (int k2 = 0; k2 < K && err.empty(); ++k2) {
                const double t0 = now_s();
                if (!model.forward(&t, 1, model.end(), lw.data(), err)) break;
                t_alone += now_s() - t0;
                alone[(size_t) i * K + k2] = lw;
                toks_a[(size_t) i].push_back(t);
                t = model.top();
            }
        }
        for (int i = 0; i < N && err.empty(); ++i) read(i);
        std::vector<int> sq((size_t) N), ids = first;
        for (int i = 0; i < N; ++i) sq[(size_t) i] = i;
        std::vector<float> rows((size_t) N * V);
        long bad_rows = 0;
        double worst = 1.0;
        for (int k2 = 0; k2 < K && err.empty(); ++k2) {
            const double t0 = now_s();
            if (!model.step_multi(sq.data(), ids.data(), N, rows.data(), err)) break;
            t_multi += now_s() - t0;
            for (int i = 0; i < N; ++i) {
                const float* a = alone[(size_t) i * K + k2].data();
                const float* b = rows.data() + (size_t) i * V;
                double dot = 0, a2 = 0, b2 = 0;
                bool same = true;
                int top = 0;
                for (int v = 0; v < V; ++v) {
                    same = same && a[v] == b[v];
                    dot += (double) a[v] * b[v];
                    a2 += (double) a[v] * a[v];
                    b2 += (double) b[v] * b[v];
                    if (b[v] > b[top]) top = v;
                }
                worst = std::min(worst, dot / std::sqrt(a2 * b2));
                bad_rows += !same;
                toks_m[(size_t) i].push_back(ids[(size_t) i]);
                ids[(size_t) i] = top;
            }
        }
        if (!err.empty()) {
            std::fprintf(stderr, "maya-ik2: %s\n", err.c_str());
            return 1;
        }
        int same_toks = 0;
        for (int i = 0; i < N; ++i) same_toks += toks_a[(size_t) i] == toks_m[(size_t) i];
        std::printf("multi check: %d conversations x %d tokens: %ld of %d rows differ from their own step's bits (worst "
                    "logits cosine %.7f), %d of %d conversations the same tokens; alone %.1f ms a token (%.1f tok/s), "
                    "multi %.1f ms a step (%.1f tok/s in all)\n",
                    N, K, bad_rows, N * K, worst, same_toks, N, 1e3 * t_alone / (N * K), N * K / t_alone,
                    1e3 * t_multi / K, N * K / t_multi);
        std::printf("%s\n", bad_rows == 0 ? "MULTI OK" : "MULTI DIFFERENT");
        return bad_rows == 0 ? 0 : 3;
    }
    if (!rewind_test.empty()) {
        // --rewind-test <reference dir>: its prompt as A; B = A's first part and other tokens after it.  B read after
        // A (a rewind to the common prefix) against B read cold: the same next token, the last logits' cosine
        Ref rf;
        if (!rf.load(rewind_test, err)) return 1;
        const std::vector<int>& A = rf.prompt;
        std::vector<float> lw((size_t) model.vocab()), lc((size_t) model.vocab());
        int bad = 0;
        auto run = [&](const char* what, size_t keep, size_t tail, int decode, bool exact) {
            std::vector<int> B(A.begin(), A.begin() + (long) std::min(keep, A.size()));
            model.reset();
            if (!model.forward(A.data(), (int) A.size(), 0, nullptr, err)) return false;
            for (int i = 0; i < decode; ++i) {   // (B keeps the first half of the decoded tokens)
                const int t = model.top();
                if (i < decode / 2) B.push_back(t);
                if (!model.forward(&t, 1, model.end(), nullptr, err)) return false;
            }
            keep = B.size();
            for (size_t i = 0; i < tail; ++i) B.push_back(A[(i * 7 + 13) % A.size()]);
            const double t0 = now_s();
            const int64_t at = model.rewind((int64_t) keep);
            if (!model.forward(&B[(size_t) at], (int) (B.size() - (size_t) at), at, lw.data(), err)) return false;
            const double tw = now_s() - t0;
            const int topw = model.top();
            model.reset();
            const double t1 = now_s();
            if (!model.forward(B.data(), (int) B.size(), 0, lc.data(), err)) return false;
            const double tc = now_s() - t1;
            double dot = 0, a2 = 0, b2 = 0;
            for (size_t i = 0; i < lw.size(); ++i) {
                dot += (double) lw[i] * lc[i];
                a2 += (double) lw[i] * lw[i];
                b2 += (double) lc[i] * lc[i];
            }
            const double c = dot / std::sqrt(a2 * b2);
            // (exact means: the shared prefix was read by the prompt path and B resumes at one of its chunks' starts -
            //  then the same chunks follow as in the cold read, only rounding is left: 0.9999; otherwise the two differ
            //  in their arithmetic's order)
            const bool ok = topw == model.top() && (c >= 0.9999 || !exact);
            bad += !ok;
            std::printf("%s: A %zu tokens (+%d decoded), B %zu tokens sharing %zu: resumed at %lld (%zu read again, %.2f s; "
                        "cold %.2f s), next token %d / cold %d, logits cosine %.6f - %s\n",
                        what, A.size(), decode, B.size(), keep, (long long) at, B.size() - (size_t) at, tw, tc, topw,
                        model.top(), c, ok ? "OK" : "DIFFERENT");
            std::fflush(stdout);
            return true;
        };
        const size_t nA = A.size();
        if (!run("diverges 500 before A's end", nA - 500, 300, 0, true) ||
            !run("diverges 37 before A's end", nA - 37, 64, 0, true) ||
            !run("diverges 2000 before A's end", nA - std::min<size_t>(2000, nA - 1), 300, 0, true) ||
            !run("diverges in the decoded part", nA, 50, 200, false) || !run("B is A and 5 more", nA, 5, 0, false)) {
            std::fprintf(stderr, "maya-ik2: %s\n", err.c_str());
            return 1;
        }
        return bad ? 3 : 0;
    }
    if (short_bench && !tokens.empty()) {
        // --short-bench: n tokens after a held context (the --tokens prompt), through the prompt path and through the
        // decode path - where the two cross is serve's --short-prompt
        std::vector<int> ids = int_list(tokens);
        for (int n : {1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 64}) {
            if ((size_t) n * 2 > ids.size()) break;
            const size_t base = ids.size() - (size_t) n;
            double tp = 0, td = 0, tp_all = 0, td_all = 0;
            const int reps = std::max(3, repeat);
            for (int rep2 = 0; rep2 < reps; ++rep2) {
                model.reset();
                if (!model.forward(ids.data(), (int) base, 0, nullptr, err)) return 1;
                double t0 = now_s();
                if (!model.forward(&ids[base], n, (int64_t) base, nullptr, err)) return 1;
                tp = now_s() - t0;
                model.reset();
                if (!model.forward(ids.data(), (int) base, 0, nullptr, err)) return 1;
                t0 = now_s();
                for (int i = 0; i < n; ++i)
                    if (!model.forward(&ids[base + (size_t) i], 1, (int64_t) base + i, nullptr, err)) return 1;
                td = now_s() - t0;
                if (rep2 > 0) tp_all += tp, td_all += td;   // (the first of each: warm-up)
                if (rep2 > 0) std::printf("short-bench %d tokens, run %d: the prompt path %.1f ms, the decode path %.1f ms\n", n,
                                          rep2, tp * 1e3, td * 1e3);
            }
            std::printf("short-bench %d tokens after %zu: the prompt path %.1f ms, the decode path %.1f ms (means of %d)\n", n,
                        base, tp_all * 1e3 / (reps - 1), td_all * 1e3 / (reps - 1), reps - 1);
            std::fflush(stdout);
        }
        return 0;
    }
    std::fflush(stdout);
    std::vector<float> logits((size_t) model.vocab());

    if (parities.empty()) {
        // --tokens: a prompt; --prompts F: a prompt a line, each from an empty context.  --ab: each prompt twice, without
        // drafts and then with them (the drafter fed both times): the speeds, and whether the tokens are the same
        std::vector<std::vector<int>> plist;
        if (!prompts_file.empty()) {
            std::ifstream pf(prompts_file);
            for (std::string line; std::getline(pf, line);)
                if (!int_list(line).empty()) plist.push_back(int_list(line));
        } else {
            plist.push_back(int_list(tokens));
        }
        if (plist.empty() || plist[0].empty()) return usage();
        std::vector<int> gen, gen_a;
        struct Run {
            size_t pi;
            int mode, sw, sv;
        };
        std::vector<Run> runs;
        if (!sweeps.empty()) {
            for (size_t sw = 0; sw < sweeps.size(); ++sw) {
                const int nv = (int) sweeps[sw].second.size();
                for (int r = 0; r < sweep_rounds; ++r)
                    for (size_t pi = 0; pi < plist.size(); ++pi) {
                        runs.push_back({pi, 1, (int) sw, -2});   // (a warm run first: the cache settled on the prompt)
                        for (int j = 0; j < nv; ++j) runs.push_back({pi, 1, (int) sw, r % 2 ? nv - 1 - j : j});
                    }
            }
        } else {
            for (size_t pi = 0; pi < plist.size(); ++pi)
                for (int mode = ab ? 0 : 1; mode < 2; ++mode) runs.push_back({pi, mode, -1, -1});
        }
        for (const Run& run : runs) {
        const size_t pi = run.pi;
        const int mode = run.mode;
        if (run.sw >= 0) {
            const auto& [var, vals] = sweeps[(size_t) run.sw];
            if (run.sw > 0) unsetenv(sweeps[(size_t) run.sw - 1].first.c_str());
            if (run.sv >= 0) setenv(var.c_str(), vals[(size_t) run.sv].c_str(), 1);
            else unsetenv(var.c_str());
            model.retune();
            std::printf("sweep %s=%s prompt %zu\n", var.c_str(), run.sv >= 0 ? vals[(size_t) run.sv].c_str() : "warm", pi + 1);
        }
        const std::vector<int>& ids = plist[pi];
        gen.clear();
        double t0 = 0, t1 = 0;
        for (int r = 0; r < std::max(1, repeat); ++r) {   // (--repeat: the prompt again from an empty context)
            model.reset();
            if (!try_chunks.empty()) {   // (--try-chunks a,b,..: run r in chunks of at most the r-th value)
                model.set_chunk(try_chunks[(size_t) r % try_chunks.size()]);
                std::printf("chunks of %d: ", try_chunks[(size_t) r % try_chunks.size()]);
            }
            t0 = now_s();
            if (!model.forward(ids.data(), (int) ids.size(), 0, nullptr, err)) {
                std::fprintf(stderr, "maya-ik2: %s\n", err.c_str());
                return 1;
            }
            t1 = now_s();
            std::printf("prompt: %zu tokens, %.2f s, %.2f tok/s (%lld experts streamed from RAM so far)\n", ids.size(), t1 - t0,
                        ids.size() / (t1 - t0), (long long) model.stats().streamed);
            std::fflush(stdout);
            model.vram_notes("after a prompt");
        }
        const int64_t cpu0 = model.stats().cpu_experts, ex0 = model.stats().experts;
        const double cms0 = model.stats().cpu_ms;
        int n = 0;
        Drafting dr;
        dr.on = model.dspark() && dspark == "on" && mode == 1;
        if (!rest0_list.empty()) {   // (tests: prompt i's rest0; 9: without drafts)
            dr.rest0 = rest0_list[pi % rest0_list.size()];
            if (dr.rest0 == 9) dr.on = false;
        }
        {   // rounds, as serve's: the token in hand read (with the drafter's drafts when they are worth a window)
            mm3::sample::Req rq;
            rq.temperature = temperature;
            rq.top_k = 0;
            std::mt19937_64 rng(seed != 0 ? seed : 0x9E3779B97F4A7C15ull);
            std::vector<int> hist = ids, out;
            std::vector<float> lgw;
            if (temperature > 0.0f) {   // (the prompt's last row again for its logits: a token's step back)
                std::fprintf(stderr, "maya-ik2: --temperature: the first token is the greedy one\n");
            }
            int tok = model.top();
            bool more = n_predict > 0;
            auto emit = [&](int t) {
                std::printf("%d ", t);
                std::fflush(stdout);
                gen.push_back(t);
                ++n;
                return t != model.eos() && n < n_predict;
            };
            if (more) more = emit(tok);
            while (more) {
                hist.push_back(tok);
                if (!draft_round(model, dr, tok, rq, hist, rng, n_predict - n, true, out, lgw, err)) {
                    std::fprintf(stderr, "\nmaya-ik2: %s\n", err.c_str());
                    return 1;
                }
                for (size_t i = 0; i < out.size() && more; ++i) {
                    more = emit(out[i]);
                    if (i + 1 < out.size()) hist.push_back(out[i]);
                }
                tok = out.back();
            }
        }
        const double t2 = now_s();
        const ik2hip::Stats& st = model.stats();
        model.vram_notes(dr.on ? "after a drafted decode" : "after a decode");
        if (n > 0) {
            std::printf("\ndecode: %d tokens, %.2f tok/s; %.1f%% of the routed experts on the CPU lane (%.1f ms a token), %lld "
                        "cache moves\n",
                        n, n / (t2 - t1), 100.0 * (double) (st.cpu_experts - cpu0) / (double) std::max<int64_t>(1, st.experts - ex0),
                        (st.cpu_ms - cms0) / n, (long long) st.swaps);
            if (dr.on) {
                std::printf("drafting: %lld passes for %lld tokens (%.2f tokens a pass), %lld of %lld drafts kept, %lld drafted "
                            "rounds (%.1f ms a draft), %lld rests; a pass %.1f ms + %.1f a row; k histogram (0 .. 7):",
                            dr.passes, dr.tokens, (double) dr.tokens / (double) std::max(1ll, dr.passes), dr.accepted, dr.offered,
                            dr.drafted, dr.drafted ? dr.draft_ms / (double) dr.drafted : 0.0, dr.rested, dr.a, dr.b);
                for (int k = 0; k < 8; ++k) std::printf(" %lld", dr.khist[k]);
                std::printf("\n");
            }
        } else {
            std::printf("next: %d\n", model.top());
        }
        if (ab && mode == 0) gen_a = gen;
        if (ab && mode == 1) {
            size_t same = 0;
            while (same < gen.size() && same < gen_a.size() && gen[same] == gen_a[same]) ++same;
            if (same == gen.size() && gen.size() == gen_a.size())
                std::printf("ab: prompt %zu: the %zu tokens with drafts are the ones without\n", pi + 1, gen.size());
            else
                std::printf("ab: prompt %zu: the tokens differ from token %zu on (%d with drafts, %d without)\n", pi + 1, same,
                            same < gen.size() ? gen[same] : -1, same < gen_a.size() ? gen_a[same] : -1);
        }
        std::fflush(stdout);
        }
        if (!model.save_profile(err)) std::fprintf(stderr, "maya-ik2: %s\n", err.c_str());
        return 0;
    }

    // parity: each reference as its own run from position 0 - with --parity-force-picks a second run of it in which
    // every layer takes the reference's picks
    int rc = 0;
    for (const std::string& parity : parities)
    for (int pass = force_only ? 1 : 0; pass < (force_opt ? 2 : 1); ++pass) {
    const bool force = pass == 1;
    Ref ref;
    if (!ref.load(parity, err)) return 1;
    model.reset();
    model.set_pick_hook(nullptr);
    // parity: the reference's prompt, then its greedy tokens, one a step; every tapped tensor against the dump's row
    std::vector<int> ids = ref.prompt;
    ids.insert(ids.end(), ref.greedy.begin(), ref.greedy.end());
    const int64_t np = (int64_t) ref.prompt.size();
    int64_t total = (int64_t) ids.size();
    if (parity_max > 0) total = std::min(total, parity_max);
    std::map<std::string, std::map<int, Acc>> acc;
    int64_t cur_p = 0;
    std::vector<float> rbuf, scores;
    double gap_max = 0;
    int64_t gap_big = 0;
    int64_t scores_p0 = 0;
    int scores_T = 1;
    auto step_of = [&](int64_t p) { return p < np ? 0 : (int) (p - np + 1); };
    if (force)
        model.set_pick_hook([&](int il, int* pick, int n, int64_t pos0, int T) {
            for (int t = 0; t < T; ++t) {
                const int64_t p = pos0 + t;
                const int step = step_of(p);
                if (!ref.row(step, "ffn_moe_topk-" + std::to_string(il), step == 0 ? p : 0, step == 0 ? np : 1, (size_t) n,
                             pick + (size_t) t * n))
                    return false;
            }
            return true;
        });
    // one row (a token's tensor) against the dump's
    auto one = [&](const char* name, int il, const void* data, const void* extra, size_t n, int64_t p) {
        const int step = step_of(p);
        const int64_t rows = step == 0 ? np : 1, rw = step == 0 ? p : 0;
        const std::string nm = std::string(name) + (il >= 0 ? "-" + std::to_string(il) : "");
        Acc& a = acc[name][il];
        rbuf.resize(n);
        if (std::strcmp(name, "ffn_moe_topk") == 0) {
            if (!ref.row(step, nm, rw, rows, n, rbuf.data())) return;
            const int32_t *mine = (const int32_t*) data, *theirs = (const int32_t*) rbuf.data();
            int hit = 0;
            for (size_t i = 0; i < n; ++i) hit += std::count(theirs, theirs + n, mine[i]) > 0;
            a.hit += hit;
            a.all += (int64_t) n;
            a.pos_diff += hit != (int) n;
            ++a.n;
            // how close the differing picks were: my last pick's score over my score of each reference pick I left out
            if (hit != (int) n && !scores.empty()) {
                const float* sc = scores.data() + (size_t) (p - scores_p0) * (scores.size() / (size_t) scores_T);
                float last = 1e30f;
                for (size_t i = 0; i < n; ++i) last = std::min(last, sc[mine[i]]);
                for (size_t i = 0; i < n; ++i)
                    if (std::count(mine, mine + n, theirs[i]) == 0) {
                        const double gap = (double) last - sc[theirs[i]];
                        a.rel_max = std::max(a.rel_max, gap);
                        gap_max = std::max(gap_max, gap);
                        gap_big += gap > 0.01;
                    }
            }
            if (detail && hit != (int) n) {
                std::printf("p %lld %s: mine", (long long) p, nm.c_str());
                for (size_t i = 0; i < n; ++i) std::printf(" %d(%.3f)", mine[i], ((const float*) extra)[i]);
                std::printf(" | reference");
                for (size_t i = 0; i < n; ++i) std::printf(" %d", theirs[i]);
                std::printf("\n");
            }
            return;
        }
        if (std::strncmp(name, "result_", 7) == 0) {   // (the dump has the batch's last row only)
            if (step == 0 && p != np - 1) return;
            if (!ref.row(step, nm, 0, 1, n, rbuf.data())) return;
        } else if (!ref.row(step, nm, rw, rows, n, rbuf.data())) {
            return;
        }
        const float* x = (const float*) data;
        double dot = 0, xx = 0, yy = 0, dmax = 0, ymax = 0;
        for (size_t i = 0; i < n; ++i) {
            dot += (double) x[i] * rbuf[i];
            xx += (double) x[i] * x[i];
            yy += (double) rbuf[i] * rbuf[i];
            dmax = std::max(dmax, (double) std::fabs(x[i] - rbuf[i]));
            ymax = std::max(ymax, (double) std::fabs(rbuf[i]));
        }
        const double c = dot / (std::sqrt(xx * yy) + 1e-30);
        a.cos_min = std::min(a.cos_min, c);
        a.cos_sum += c;
        a.rel_max = std::max(a.rel_max, dmax / (ymax + 1e-30));
        ++a.n;
        if (detail) std::printf("p %lld %s: cos %.6f, max rel err %.4f\n", (long long) p, nm.c_str(), c, dmax / (ymax + 1e-30));
    };
    model.set_tap([&](const char* name, int il, const void* data, size_t n, int64_t pos0, int T) {
        if (std::strcmp(name, "ffn_moe_scores") == 0) {
            scores.assign((const float*) data, (const float*) data + n);
            scores_p0 = pos0;
            scores_T = T;
            return;
        }
        const size_t row = n / (size_t) T;
        for (int t = 0; t < T; ++t)
            one(name, il, (const char*) data + (size_t) t * row * 4, (const char*) data + (n + (size_t) t * row) * 4, row,
                pos0 + t);
    });
    std::vector<int> mine;
    int greedy_same = 0, greedy_all = 0;
    const double t0 = now_s();
    for (int64_t p = 0; p < total;) {
        // the prompt in one call (the prompt path's chunks) unless --parity-decode; the greedy tokens one a step
        const int64_t n = !decode_only && p == 0 ? std::min(total, np) : 1;
        cur_p = p + n - 1;
        if (!model.forward(&ids[(size_t) p], (int) n, p, logits.data(), err)) {
            std::fprintf(stderr, "maya-ik2: position %lld: %s\n", (long long) p, err.c_str());
            return 1;
        }
        p += n;
        if (p >= np) {
            mine.push_back(model.top());
            const size_t gi = (size_t) (p - np);
            if (gi < ref.greedy.size()) {
                ++greedy_all;
                greedy_same += ref.greedy[gi] == model.top();
            }
        }
    }
    std::printf("parity against %s: %lld positions (%lld of the prompt), %.1f s\n", parity.c_str(), (long long) total,
                (long long) std::min(total, np), now_s() - t0);
    auto get = [&](const char* name, int il) -> const Acc* {
        auto it = acc.find(name);
        if (it == acc.end()) return nullptr;
        auto jt = it->second.find(il);
        return jt == it->second.end() || jt->second.n == 0 ? nullptr : &jt->second;
    };
    std::printf("layer  l_out: cos min   cos mean   max rel err | picks same   positions differing, widest gap | attn_out   ffn_inp   "
                "ffn_out (cos min)\n");
    double worst = 2.0;
    int64_t hit = 0, all = 0;
    for (int il = 0; il < model.layers(); ++il) {
        const Acc *lo = get("l_out", il), *pk = get("ffn_moe_topk", il), *ao = get("attn_out", il),
                  *fi = get("ffn_inp", il), *fo = get("ffn_out", il);
        std::printf("%5d  ", il);
        if (lo) {
            std::printf("%.6f     %.6f   %.4f      ", lo->cos_min, lo->cos_sum / (double) lo->n, lo->rel_max);
            worst = std::min(worst, lo->cos_min);
        } else {
            std::printf("%-37s", "-");
        }
        if (pk) {
            std::printf("| %6lld/%-6lld %4lld/%-5lld %.4f ", (long long) pk->hit, (long long) pk->all, (long long) pk->pos_diff,
                        (long long) pk->n, pk->rel_max);
            hit += pk->hit;
            all += pk->all;
        } else {
            std::printf("| %-33s", "-");
        }
        auto c = [](const Acc* a) { return a ? a->cos_min : NAN; };
        std::printf("| %.6f   %.6f  %.6f\n", c(ao), c(fi), c(fo));
    }
    for (const char* name : {"Qcur", "Kcur", "Vcur", "Rcur", "kqv_out"})
        for (int il = 0; il < model.layers(); ++il)
            if (const Acc* a = get(name, il))
                std::printf("%s-%d: cos min %.6f, max rel err %.4f (%lld rows)\n", name, il, a->cos_min, a->rel_max, (long long) a->n);
    for (const char* name : {"result_norm", "result_output"})
        if (const Acc* a = get(name, -1))
            std::printf("%s: cos min %.6f, cos mean %.6f, max rel err %.4f (%lld rows)\n", name, a->cos_min,
                        a->cos_sum / (double) a->n, a->rel_max, (long long) a->n);
    std::printf("worst layer output cosine %.6f; picks %lld of %lld the same (%.3f%%)%s\n", worst, (long long) hit,
                (long long) all, all ? 100.0 * (double) hit / (double) all : 0.0,
                force ? " - every layer ran the reference's picks" : "");
    std::printf("the differing picks' selection scores: at most %.5f under my last pick's, %lld more than 0.01 under\n",
                gap_max, (long long) gap_big);
    std::printf("greedy tokens:");
    for (int tkn : mine) std::printf(" %d", tkn);
    std::printf("\nreference:    ");
    for (int tkn : ref.greedy) std::printf(" %d", tkn);
    std::printf("\ngreedy tokens the same: %d of %d\n", greedy_same, greedy_all);
    const ik2hip::Stats& st = model.stats();
    std::printf("routed experts: %lld, %.1f%% on the CPU lane (%.2f ms a token)\n", (long long) st.experts,
                100.0 * (double) st.cpu_experts / (double) std::max<int64_t>(1, st.experts),
                st.cpu_ms / (double) std::max<int64_t>(1, st.tokens));
    // the spec's test ("Parity, as measured"): the same greedy tokens always; with the reference's picks every layer's
    // mean cosine >= 0.999, and on a short reference (no room for the model's own sensitivity) every row
    const bool full = model.layers() == 48 && total == (int64_t) ids.size();
    bool pass_ok = greedy_same == greedy_all;
    if (force) {
        for (int il = 0; il < model.layers(); ++il)
            if (const Acc* lo = get("l_out", il)) pass_ok = pass_ok && lo->cos_sum / (double) lo->n >= 0.999;
        if (total <= 64) pass_ok = pass_ok && worst >= 0.999;
    }
    std::printf("%s\n", !full ? "(a part of the model or of the reference: not judged)" : pass_ok ? "PARITY OK" : "PARITY FAILED");
    if (full && !pass_ok) rc = 3;
    std::fflush(stdout);
    }
    if (!model.save_profile(err)) std::fprintf(stderr, "maya-ik2: %s\n", err.c_str());
    return rc;
}
