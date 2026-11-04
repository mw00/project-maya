// src/metal/serve.cpp - maya-metal --serve: the GLM engine's stdin/stdout protocol (src/program/generate.cpp's serve
// loop, one conversation at a time), so serve/server.py drives maya-metal exactly as it drives maya.
//
//   in:  GEN <max_new> [key=value ...] <id,id,...>   STOP   PAUSE [keep]   WRAP   QUIT
//   out: INFO ... / READY <ctx> stop, then per request PP ..., T <id> ..., DONE ... (or ERR ...)
//
// Sampling is generate.cpp's / sampler.cu's chain on the host (the logits are already in unified memory): bias,
// penalties, top_k (ties to the lower id), top_p over the top_k list in double, min_p, temperature, one Philox draw
// at (seed, counter).  Same keys, same defaults, same draws, so a seeded request gives the CUDA engine's tokens for the
// same logits.  The thinking budget and the loop guard are generate.cpp's, rule for rule.
#include "glm_metal.hpp"
#include "sampler.hpp"
#include "spec.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace maya::metal {

int mtp_draft_cap();

namespace {

constexpr int32_t kEos[] = {154820, 154827, 154829};
constexpr int32_t kArgValue = 154849, kArgValueEnd = 154850, kUser = 154827;
constexpr int32_t kImageToken = 154854;   // <|image|>: one per image row maya-vision encoded

int env_int(const char* k, int def) {
    const char* v = std::getenv(k);
    return v && *v ? std::atoi(v) : def;
}

// stdin on its own thread: STOP / PAUSE / WRAP act at once (a running request sees them), the rest queue up
struct Input {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    std::atomic<bool> stop{false}, wrap{false}, pause{false};
    bool eof = false;

    void run() {
        std::string line;
        while (std::getline(std::cin, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
            if (line.empty()) continue;
            if (line == "STOP" || line.rfind("STOP ", 0) == 0 || line == "PAUSE" || line.rfind("PAUSE ", 0) == 0) {
                // PAUSE: a STOP that keeps the state with what was written, so the server's follow-up GEN (the
                // prompt + those tokens) reads none of it again (generate.cpp's pause_snapshot)
                if (line.rfind("PAUSE", 0) == 0) pause.store(true);
                stop.store(true);
                continue;
            }
            if (line == "WRAP") {
                wrap.store(true);
                continue;
            }
            if (line == "QUIT") stop.store(true);
            // a request's STOP / PAUSE / WRAP may come before its run() starts (the server sends one GEN at a time
            // and drains it to DONE): the flags clear here, as the GEN arrives - and a WRAP ends one request's thinking
            if (line.rfind("GEN", 0) == 0) {
                stop.store(false);
                pause.store(false);
                wrap.store(false);
            }
            std::lock_guard<std::mutex> lk(mu);
            q.push_back(line);
            cv.notify_all();
        }
        std::lock_guard<std::mutex> lk(mu);
        eof = true;
        cv.notify_all();
    }
    bool next(std::string& line) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !q.empty() || eof; });
        if (q.empty()) return false;
        line = std::move(q.front());
        q.pop_front();
        return true;
    }
};

// a request's drafting keys (generate.cpp's): spec=1 off, spec=T windows of T rows; DFlash2's confidence cut
struct Spec {
    int req = 0;
    float p_min = 0.95f;   // one GPU's default (glm_dflash.cu)
    int n_min = 0;
    float p_cum = 0.0f;
};

struct Server {
    GlmMetal& model;
    Input in;
    std::vector<int32_t> cached;      // the previous prompt less its last token: what the snapshot holds
    uint64_t cached_img = 0;          // the images inside it (positions and rows, hashed)
    const int loop_w_think = env_int("MAYA_GLM_LOOP_THINK", 256);
    const int loop_w_answer = env_int("MAYA_GLM_LOOP_ANSWER", 1024);
    int chunk = 512;                            // prompt tokens a forward() takes: the engine's slice (serve())
    bool mtp = false;                           // the NextN block drafts (has_mtp(), not MAYA_GLM_NO_MTP)
    bool mtp_auto = false;                      // mtp follows the pool (no MAYA_GLM_MTP_DECODE, MTP allowed)
    std::unique_ptr<DraftLength> mtp_len;       // kept across requests: what drafts were worth on this machine

    // the request's images, for the prefix check (generate.cpp's img_hash): FNV-1a over the rows and positions
    // before a length
    std::vector<int64_t> cur_pos;
    std::vector<float> cur_rows;
    uint64_t img_prefix_hash(int64_t len) const {
        uint64_t h = 1469598103934665603ull;
        const size_t E = (size_t) model.geometry().n_embd;
        for (size_t k = 0; k < cur_pos.size() && cur_pos[k] < len; ++k) {
            const unsigned char* b = (const unsigned char*) (cur_rows.data() + k * E);
            for (size_t j = 0; j < E * sizeof(float); ++j) h = (h ^ b[j]) * 1099511628211ull;
            h = (h ^ (uint64_t) cur_pos[k]) * 1099511628211ull;
        }
        return h;
    }

    void err(const std::string& m) {
        std::printf("ERR %s\n", m.c_str());
        std::fflush(stdout);
    }

    void gen(const std::string& line) {
        std::istringstream ss(line);
        std::string cmd, tok;
        long long max_new = 0;
        ss >> cmd >> max_new;
        Sampling sp;
        bool any_key = false, greedy_req = false;
        int64_t think_budget = 0;
        int32_t think_end = -1;
        int spec_req = 0;
        Spec spec;
        int pen_n = 0;
        float pen_repeat = 1.0f, pen_freq = 0.0f, pen_present = 0.0f;
        std::string ids_s, bias_err, emb_path;
        const int n_vocab = model.geometry().n_vocab;
        while (ss >> tok) {
            const size_t eq = tok.find('=');
            if (eq == std::string::npos) {
                if (cmd == "GENI" && emb_path.empty()) emb_path = tok;   // GENI: the embeddings file, then the ids
                else ids_s = tok;
                continue;
            }
            const std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
            if (k == "temperature") {
                sp.temperature = (float) std::atof(v.c_str());
                if (!(sp.temperature > 0.0f) || !std::isfinite(1.0f / sp.temperature)) greedy_req = true;   // NaN, 0, denormal
                else any_key = true;
            } else if (k == "top_p") { sp.top_p = (float) std::atof(v.c_str()); any_key = true; }
            else if (k == "top_k") { sp.top_k = std::atoi(v.c_str()); any_key = true; }
            else if (k == "min_p") { sp.min_p = (float) std::atof(v.c_str()); any_key = true; }
            else if (k == "seed") sp.seed = (uint64_t) std::atoll(v.c_str());
            else if (k == "think_budget") think_budget = std::atoll(v.c_str());
            else if (k == "think_end") think_end = std::atoi(v.c_str());
            else if (k == "spec") spec_req = std::atoi(v.c_str());   // 1: no drafting; T >= 2: windows of T rows
            else if (k == "spec_min_p") spec.p_min = (float) std::atof(v.c_str());
            else if (k == "spec_n_min") spec.n_min = std::atoi(v.c_str());
            else if (k == "spec_cum") spec.p_cum = (float) std::atof(v.c_str());
            else if (k == "penalty_last_n") pen_n = std::atoi(v.c_str());
            else if (k == "penalty_repeat") pen_repeat = std::strtof(v.c_str(), nullptr);
            else if (k == "penalty_freq") pen_freq = std::strtof(v.c_str(), nullptr);
            else if (k == "penalty_present") pen_present = std::strtof(v.c_str(), nullptr);
            else if (k == "logit_bias") {   // <id>:<bias>,... - -100 or below bans the token
                for (size_t at = 0; at < v.size() && bias_err.empty();) {
                    size_t end = v.find(',', at);
                    if (end == std::string::npos) end = v.size();
                    const std::string e1 = v.substr(at, end - at);
                    const size_t colon = e1.find(':');
                    char* p1 = nullptr;
                    char* p2 = nullptr;
                    const long id = colon == std::string::npos ? -1 : std::strtol(e1.c_str(), &p1, 10);
                    const float b = colon == std::string::npos ? 0.0f : std::strtof(e1.c_str() + colon + 1, &p2);
                    if (id < 0 || id >= n_vocab || p1 != e1.c_str() + colon || p2 == nullptr || *p2 != '\0' ||
                        !std::isfinite(b))
                        bias_err = "malformed logit_bias entry '" + e1 + "'";
                    else
                        sp.bias.emplace_back((int32_t) id, b <= -100.0f ? -INFINITY : b);
                    at = end + 1;
                }
            }
        }
        if (any_key) sp.greedy = false;
        if (greedy_req) sp.greedy = true;
        if (!std::isfinite(pen_repeat) || pen_repeat <= 0.0f) pen_repeat = 1.0f;
        if (!std::isfinite(pen_freq)) pen_freq = 0.0f;
        if (!std::isfinite(pen_present)) pen_present = 0.0f;
        if (pen_n < 0) pen_n = INT_MAX;
        if (pen_n > 0 && (pen_repeat != 1.0f || pen_freq != 0.0f || pen_present != 0.0f)) {
            sp.pen_n = pen_n;
            sp.pen_repeat = pen_repeat;
            sp.pen_freq = pen_freq;
            sp.pen_present = pen_present;
        }
        if (!bias_err.empty()) return err(bias_err);
        std::vector<int32_t> prompt;
        bool ok = max_new > 0 && !ids_s.empty();
        for (size_t at = 0; ok && at < ids_s.size();) {
            char* end = nullptr;
            const long long t = std::strtoll(ids_s.c_str() + at, &end, 10);
            if (end == ids_s.c_str() + at) {
                ok = false;
                break;
            }
            if (t < 0 || t >= n_vocab)
                return err("token " + std::to_string(t) + " out of range [0, " + std::to_string(n_vocab) + ")");
            prompt.push_back((int32_t) t);
            at = (size_t) (end - ids_s.c_str());
            if (at < ids_s.size()) {
                if (ids_s[at] != ',') {
                    ok = false;
                    break;
                }
                ++at;
            }
        }
        if (!ok || prompt.empty()) return err("malformed GEN line");
        // GENI: maya-vision's rows ("SVE1" records, n_embd floats each) take the <|image|> tokens' places
        std::vector<int64_t> img_pos;
        std::vector<float> img_rows;
        if (cmd == "GENI") {
            const int64_t E = model.geometry().n_embd;
            std::FILE* f = std::fopen(emb_path.c_str(), "rb");
            std::string ge = f ? "" : "cannot open " + emb_path;
            while (f && ge.empty()) {
                int32_t hdr[5];
                const size_t got = std::fread(hdr, 1, sizeof hdr, f);
                if (got == 0 && std::feof(f)) break;
                if (got != sizeof hdr || hdr[0] != 0x31455653 || hdr[1] <= 0 || hdr[4] != E ||
                    (size_t) hdr[1] > prompt.size() - img_rows.size() / (size_t) E) {
                    ge = "not a maya-vision record for this model";
                    break;
                }
                const size_t at = img_rows.size();
                img_rows.resize(at + (size_t) hdr[1] * (size_t) E);
                if (std::fread(img_rows.data() + at, sizeof(float) * (size_t) E, (size_t) hdr[1], f) != (size_t) hdr[1])
                    ge = "short embeddings file";
            }
            if (f) std::fclose(f);
            for (size_t i = 0; ge.empty() && i < prompt.size(); ++i)
                if (prompt[i] == kImageToken) img_pos.push_back((int64_t) i);
            if (ge.empty() && (int64_t) img_pos.size() * E != (int64_t) img_rows.size())
                ge = std::to_string(img_pos.size()) + " image tokens, " + std::to_string(img_rows.size() / (size_t) E) +
                     " image rows";
            if (!ge.empty()) return err(ge);
        }
        cur_pos = img_pos;
        cur_rows = img_rows;
        model.set_image_rows(std::move(img_pos), std::move(img_rows));
        if (max_new > model.max_context() - (int64_t) prompt.size())
            max_new = model.max_context() - (int64_t) prompt.size();
        if (max_new <= 0)
            return err("the prompt (" + std::to_string(prompt.size()) + " tokens) fills the context (" +
                       std::to_string(model.max_context()) + ")");
        spec.req = spec_req;
        run(prompt, max_new, sp, think_budget, think_end, spec);
    }

    void run(const std::vector<int32_t>& prompt, int64_t max_new, Sampling sp, int64_t think_budget,
             int32_t think_end, const Spec& spec) {
        const int spec_req = spec.req;
        const int n_vocab = model.geometry().n_vocab;
        std::string e;
        const char* finish = "length";
        const auto t0 = std::chrono::steady_clock::now();
        const size_t n = prompt.size();

        // the prefix the snapshot holds, when the new prompt starts with all of it
        size_t reuse = 0;
        const int64_t snap = model.snapshot_position();
        if (snap > 0 && (size_t) snap == cached.size() && (size_t) snap <= n - 1 &&
            std::equal(cached.begin(), cached.end(), prompt.begin()) && cached_img == img_prefix_hash(snap) &&
            model.snapshot_restore())
            reuse = (size_t) snap;
        else
            model.reset();
        cached.clear();

        std::vector<float> logits;
        auto pp = [&](size_t read) {
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("PP %zu %zu %.1f %.1f %zu\n", read, n, ms, ms > 0 ? (read - reuse) * 1000.0 / ms : 0.0, reuse);
            std::fflush(stdout);
        };
        for (size_t at = reuse; at + 1 < n; at += (size_t) chunk) {   // every prompt token but the last
            if (in.stop.load()) {
                finish = "cancel";
                break;
            }
            const size_t k = std::min((size_t) chunk, n - 1 - at);
            if (!model.forward(std::vector<int32_t>(prompt.begin() + (long) at, prompt.begin() + (long) (at + k)),
                               logits, e)) {
                model.reset();
                return err(e);
            }
            pp(at + k);
        }
        int64_t produced = 0;
        auto t1 = std::chrono::steady_clock::now();
        if (std::strcmp(finish, "cancel") != 0) {
            model.snapshot_save();
            cached.assign(prompt.begin(), prompt.end() - 1);
            cached_img = img_prefix_hash((int64_t) cached.size());
            if (!model.forward({prompt.back()}, logits, e)) {
                model.reset();
                cached.clear();
                return err(e);
            }
            pp(n);
            t1 = std::chrono::steady_clock::now();
        } else {
            model.reset();   // a cut prompt leaves no state worth keeping
        }

        std::vector<int32_t> pen_hist;
        if (sp.pen_n > 0) pen_hist = prompt;
        bool in_think = think_budget > 0 && think_end >= 0;
        std::vector<int32_t> gen_hist;
        bool in_arg = false;
        int guard_fired = 0;
        int32_t forced = -1;
        auto looping = [&](int w) -> int {
            const int m = (int) gen_hist.size();
            if (w <= 0 || m < w) return 0;
            for (int per = 1; per <= 32 && per < w / 4; ++per) {
                bool rep = true;
                for (int i = m - w + per; i < m && rep; ++i) rep = gen_hist[(size_t) i] == gen_hist[(size_t) (i - per)];
                if (rep) return per;
            }
            return 0;
        };
        auto note = [&](const char* m) {
            std::fprintf(stderr, "glm loop guard: %s\n", m);
            std::printf("NOTE %s\n", m);
        };
        std::vector<int32_t> emitted;   // what this request wrote (a PAUSE keeps the state they lead to)
        auto on_token = [&](int32_t t) -> bool {
            std::printf("T %d\n", t);
            std::fflush(stdout);
            emitted.push_back(t);
            ++produced;
            if (sp.pen_n > 0) pen_hist.push_back(t);
            gen_hist.push_back(t);
            if (in_think && t != think_end) {
                if (const int per = looping(loop_w_think)) {
                    forced = think_end;
                    in_think = false;
                    char m[200];
                    std::snprintf(m, sizeof m,
                                  "thinking was repeating a %d-token pattern (%d tokens in): closed, answering", per,
                                  (int) produced);
                    note(m);
                    gen_hist.clear();
                }
            } else if (!in_think) {
                if (t == kArgValue) in_arg = true;
                if (t == kArgValueEnd) in_arg = false;
                if (const int per = looping(loop_w_answer)) {
                    char m[240];
                    if (guard_fired == 0) {
                        forced = in_arg ? kArgValueEnd : kUser;
                        std::snprintf(m, sizeof m, "the answer was repeating a %d-token pattern (%d tokens in): %s", per,
                                      (int) produced,
                                      in_arg ? "the tool call's argument was closed there" : "the turn was ended there");
                    } else {
                        std::snprintf(m, sizeof m, "the answer kept repeating a %d-token pattern (%d tokens in): ended",
                                      per, (int) produced);
                    }
                    note(m);
                    gen_hist.clear();
                    if (guard_fired++ > 0) {
                        finish = "length";
                        return false;
                    }
                }
            }
            if (in_think && (t == think_end || produced >= think_budget || in.wrap.load())) {
                if (t != think_end) forced = think_end;
                in_think = false;
                gen_hist.clear();
            }
            if (produced % 8 == 0) {   // the Monitor's live counters
                const auto& st = model.stats();
                const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
                const uint64_t lk = st.expert_hits + st.expert_misses;
                std::printf("STAT tok_s=%.2f ms_tok=%.2f vram_hit=%.4f ram_fetch=0.000 disk=%.3f promo=0.000 "
                            "vram_used=%zu vram_slots=%zu ram_used=0 ram_slots=0 ram_gb=0.0 vram_gb=%.1f\n",
                            produced / std::max(1e-9, el), 1000.0 * el / (double) produced,
                            lk ? (double) st.expert_hits / (double) lk : 1.0,
                            (double) st.expert_misses / std::max<double>(1.0, (double) st.tokens), st.pool_slots,
                            st.pool_slots, st.pool_bytes / 1e9);
            }
            for (const int32_t x : kEos)
                if (x == t) {
                    finish = "stop";
                    return false;
                }
            if (in.stop.load()) {
                finish = "cancel";
                return false;
            }
            return produced < max_new;
        };
        if (std::strcmp(finish, "cancel") != 0) {
            // one token: the plain pick of these logits (a forced token wins), the counter one draw on
            bool bad = false;
            auto pick_one = [&](const float* l) -> int32_t {
                for (size_t v = 0; v < (size_t) n_vocab; ++v)
                    if (!std::isfinite(l[v])) {
                        bad = true;
                        return 0;
                    }
                int32_t t = pick(std::vector<float>(l, l + n_vocab), sp, pen_hist);
                if (forced >= 0) {
                    t = forced;
                    forced = -1;
                }
                sp.counter += 1;
                return t;
            };
            if (!in.stop.load()) {
                const int32_t y = pick_one(logits.data());
                if (bad) {
                    model.reset();
                    cached.clear();
                    return err("non-finite logits");
                }
                if (on_token(y)) {
                    DecodeHooks hk;
                    hk.pick = pick_one;
                    hk.emit = [&](int32_t t) { return !bad && on_token(t); };
                    hk.left = [&] { return max_new - produced; };
                    if (model.has_dflash() && spec_req != 1) {   // DFlash2 drafts, a verify window checks
                        const int n_max = std::min(15, spec_req >= 2 ? spec_req - 1 : model.dflash_block() - 1);
                        const Drafter df = [&](int32_t tok, int64_t, int n, std::vector<int32_t>& d, std::string& er) {
                            return model.dflash_draft(tok, n, spec.p_min, spec.n_min, spec.p_cum, d, er);
                        };
                        const bool ok = decode_spec(model, y, n_max, df, nullptr, hk, e);
                        if (!ok || bad) {
                            model.reset();
                            cached.clear();
                            return err(bad ? "non-finite logits" : e);
                        }
                    } else if (mtp && spec_req != 1) {   // the NextN block drafts, a verify window checks (same tokens)
                        const bool ok = decode_mtp(model, y, *mtp_len, hk, e);
                        if (!ok || bad) {
                            model.reset();
                            cached.clear();
                            return err(bad ? "non-finite logits" : e);
                        }
                    } else {
                        for (int32_t t = y;;) {   // (on_token checked STOP: the last token written is never fed)
                            if (!model.forward({t}, logits, e)) {
                                model.reset();
                                cached.clear();
                                return err(e);
                            }
                            t = pick_one(logits.data());
                            if (bad) {
                                model.reset();
                                cached.clear();
                                return err("non-finite logits");
                            }
                            if (!on_token(t)) break;
                        }
                    }
                }
            } else {
                finish = "cancel";
            }
        }
        const auto t2 = std::chrono::steady_clock::now();
        // PAUSE: every token written but the last was fed, so the state is the follow-up prompt's less its last
        // token - exactly what the prefix reuse restores
        if (in.pause.load() && std::strcmp(finish, "cancel") == 0 && !emitted.empty() &&
            model.position() == (int64_t) (n + emitted.size() - 1)) {
            model.snapshot_save();
            cached.assign(prompt.begin(), prompt.end());
            cached.insert(cached.end(), emitted.begin(), emitted.end() - 1);
            cached_img = img_prefix_hash((int64_t) cached.size());
        }
        std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu 0 0\n", (long long) produced, n,
                    std::chrono::duration<double, std::milli>(t1 - t0).count(),
                    std::chrono::duration<double, std::milli>(t2 - t1).count(), finish, reuse);
        std::fflush(stdout);
        if (const size_t grown = model.grow_pool()) {   // memory freed since the start: more experts stay resident
            std::fprintf(stderr, "maya-metal: the expert pool grew by %.1f GB to %.1f GB\n", grown / 1e9,
                         model.stats().pool_bytes / 1e9);
            // started short (a restart before the old engine's memory came back): MTP's default turns on once the
            // pool holds every expert, as it would have at a start with that memory
            if (mtp_auto && !mtp && model.pool_holds_all()) {
                mtp = true;
                mtp_len = std::make_unique<DraftLength>(mtp_draft_cap());
                std::fprintf(stderr, "maya-metal: every expert has a slot now: MTP drafting is on\n");
            }
        }
        std::string pe;   // this machine's expert usage, for the next start (after DONE: nobody waits on it)
        if (!model.save_profile(pe)) std::fprintf(stderr, "maya-metal: %s\n", pe.c_str());
        if (produced > 0) {
            const auto& st = model.stats();
            const double s = std::chrono::duration<double>(t2 - t1).count();
            std::fprintf(stderr, "maya-metal: %lld tokens, %.2f tok/s; experts %llu hits / %llu misses so far\n",
                         (long long) produced, produced / std::max(1e-9, s), (unsigned long long) st.expert_hits,
                         (unsigned long long) st.expert_misses);
        }
    }
};

}  // namespace

int mtp_draft_cap() {   // MAYA_GLM_MTP_DRAFT (default 3), at most 8 drafts a round
    const char* v = std::getenv("MAYA_GLM_MTP_DRAFT");
    return std::min(8, std::max(0, v && *v ? std::atoi(v) : 3));
}

int serve(GlmMetal& model, const std::vector<std::string>& args) {
    (void) args;   // the server's CUDA-engine flags (--expert-cache, --prefill, --kv-streaming ...) mean nothing here
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    Server s{model, {}, {}};
    // a prompt goes to forward() a slice at a time (STOP and the PP lines land between them): one --ubatch slice, or
    // 512-token pieces cut every --ubatch 2048 slice to 512 and its experts' tiles ran a quarter full
    s.chunk = model.ubatch();
    if (const char* c = std::getenv("MAYA_METAL_PP_CHUNK")) s.chunk = std::max(1, std::atoi(c));
    // MTP drafts by default only when every expert is resident: then a token is GPU work and a verify window writes
    // several for about the price of one.  With experts streamed from the SSD a window needs more of them than one
    // token (measured on Maya-S, 64 GB M4 Max: plain 7.3 tok/s, MTP 6.9 - the same tokens) - as the CUDA engine, whose
    // MTP decode is on by default only where memory is unified and large.  MAYA_GLM_MTP_DECODE=0/1 decides instead.
    const char* md = std::getenv("MAYA_GLM_MTP_DECODE");
    const bool mtp_wanted = md && *md ? std::atoi(md) != 0 : model.stats().experts_all_resident;
    s.mtp = mtp_wanted && model.has_mtp() && std::getenv("MAYA_GLM_NO_MTP") == nullptr &&
            std::getenv("MAYA_GLM_NO_SPEC") == nullptr;
    s.mtp_auto = !(md && *md) && model.has_mtp() && std::getenv("MAYA_GLM_NO_MTP") == nullptr &&
                 std::getenv("MAYA_GLM_NO_SPEC") == nullptr;
    if (model.has_mtp() && !s.mtp)
        std::printf("NOTE MTP drafting is off (the experts stream from the SSD); MAYA_GLM_NO_MTP=1 frees the NextN "
                    "block's memory for the expert pool\n");
    if (s.mtp) s.mtp_len = std::make_unique<DraftLength>(mtp_draft_cap());
    const auto& st = model.stats();
    std::printf("INFO engine=%s context=%lld logit_bias=1 kv_ctx=%lld kv_gb=%.3f vram_gb=%.1f vram_slots=%zu\n",
                MAYA_VERSION, (long long) model.max_context(), (long long) model.max_context(), st.state_bytes / 1e9,
                (st.dense_bytes + st.pool_bytes + st.state_bytes) / 1e9, st.pool_slots);
    std::printf("NOTE %s\n", model.describe().c_str());
    std::printf("READY %lld stop pause\n", (long long) model.max_context());
    std::fflush(stdout);
    std::thread reader([&] { s.in.run(); });
    reader.detach();   // blocked in getline until stdin closes; the process exits around it
    std::string line;
    // QUIT or the end of stdin: out at once - the reader thread stays blocked in getline, so no return through
    // the destructors and stdio's exit-time locks
    auto quit = [] {
        std::fflush(stdout);
        std::fflush(stderr);
        std::_Exit(0);
    };
    while (s.in.next(line)) {
        if (line == "QUIT") quit();
        const std::string head = line.substr(0, line.find(' '));
        if (head == "GEN" || head == "GENI") s.gen(line);
        else if (head == "VLEND" || head == "VRECLAIM" || head == "LOGP")
            s.err("unsupported command '" + head + "' (maya-metal serves text only)");
        else
            s.err("unsupported command '" + head + "'");
    }
    quit();
    return 0;
}

}  // namespace maya::metal
