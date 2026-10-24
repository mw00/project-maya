// src/metal/spec.cpp - see spec.hpp.
#include "spec.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>

namespace maya::metal {

namespace {

int env_int(const char* k, int def) {
    const char* v = std::getenv(k);
    return v && *v ? std::atoi(v) : def;
}

}  // namespace

DraftLength::DraftLength(int n_cap)
    : cap_(std::max(0, n_cap)),
      start_(std::min(std::max(0, env_int("MAYA_GLM_MTP_START", 2)), std::max(0, n_cap))),
      fixed_(env_int("MAYA_GLM_MTP_FIXED", -1)),
      adapt_(env_int("MAYA_GLM_MTP_ADAPT", 1) != 0),
      hit_((size_t) cap_ + 1, 0.0),
      reach_((size_t) cap_ + 1, 0.0),
      ms_((size_t) cap_ + 1, 0.0),
      ms_n_((size_t) cap_ + 1, 0) {
    best_ = start_;
}

int DraftLength::choose() {
    if (fixed_ >= 0) return std::min(fixed_, cap_);
    if (!adapt_) return cap_;
    if (rounds_ < 6) return start_;   // the first rounds at the start length, measured
    if (probing_ > 0) return probe_n_;
    // expected tokens of a round of n drafts: 1 + sum over depths of the product of their acceptance (prior 0.75)
    auto tokens = [&](int n) {
        double e = 1.0, prod = 1.0, prev = 0.75;
        for (int i = 1; i <= n; ++i) {
            const double a = (hit_[(size_t) i] + 4.0 * prev) / (reach_[(size_t) i] + 4.0);
            prod *= a;
            e += prod;
            prev = a;
        }
        return e;
    };
    // a length's time per round: measured, or the nearest measured one scaled by the window's rows
    auto ms = [&](int n) {
        if (ms_n_[(size_t) n] > 0) return ms_[(size_t) n];
        for (int d = 1; d <= cap_; ++d)
            for (const int k : {n - d, n + d})
                if (k >= 0 && k <= cap_ && ms_n_[(size_t) k] > 0) return ms_[(size_t) k] * (1.0 + n) / (1.0 + k);
        return 1.0 + n;
    };
    int best = 0;
    double best_rate = 0.0;
    for (int n = 0; n <= cap_; ++n) {
        const double r = tokens(n) / ms(n);
        if (r > best_rate) {
            best_rate = r;
            best = n;
        }
    }
    best_ = best;
    if (rounds_ >= probe_at_) {   // now and then two rounds at a neighbour, to keep its timing current
        probe_n_ = best_ + ((rounds_ / probe_gap_) % 2 ? 1 : -1);
        if (probe_n_ >= 0 && probe_n_ <= cap_) probing_ = 2;
        probe_gap_ = std::min(probe_gap_ * 2, 1024);
        probe_at_ = rounds_ + probe_gap_;
        if (probing_ > 0) return probe_n_;
    }
    return best_;
}

void DraftLength::record(int n, int accepted, double ms) {
    ++rounds_;
    if (probing_ > 0) --probing_;
    constexpr double kDecay = 1.0 - 1.0 / 64.0;
    for (int i = 1; i <= cap_; ++i) {
        hit_[(size_t) i] *= kDecay;
        reach_[(size_t) i] *= kDecay;
    }
    for (int i = 1; i <= n; ++i) {   // depth i was reached if every earlier draft was accepted
        if (i > accepted + 1) break;
        reach_[(size_t) i] += 1.0;
        if (i <= accepted) hit_[(size_t) i] += 1.0;
    }
    run_ = n == last_n_ ? run_ + 1 : 1;
    last_n_ = n;
    if (run_ >= 3) {   // a length's time only from its third round in a row (the first ones pay for the switch)
        ms_[(size_t) n] = ms_n_[(size_t) n] == 0 ? ms : 0.8 * ms_[(size_t) n] + 0.2 * ms;
        ++ms_n_[(size_t) n];
    }
}

bool decode_spec(GlmMetal& model, int32_t y, int n_max, const Drafter& draft,
                 const std::function<void(int, int, double)>& done_round, const DecodeHooks& hooks, std::string& err) {
    const size_t V = (size_t) model.geometry().n_vocab;
    std::vector<float> logits;
    std::vector<int32_t> drafts;
    for (;;) {
        const auto t0 = std::chrono::steady_clock::now();
        const int64_t p0 = model.position();   // y goes here
        int64_t room = std::min<int64_t>(n_max, hooks.left() - 1);
        room = std::min<int64_t>(room, model.max_context() - 1 - p0);
        drafts.clear();
        if (room > 0 && !draft(y, p0, (int) room, drafts, err)) return false;
        if (drafts.size() > (size_t) std::max<int64_t>(room, 0)) drafts.resize((size_t) std::max<int64_t>(room, 0));
        if (drafts.empty()) {   // a plain step
            if (!model.forward({y}, logits, err)) return false;
            const int32_t t = hooks.pick(logits.data());
            if (done_round) done_round(0, 0, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            if (!hooks.emit(t)) return true;
            y = t;
            continue;
        }
        std::vector<int32_t> win = {y};
        win.insert(win.end(), drafts.begin(), drafts.end());
        if (!model.forward_window(win, logits, err)) return false;
        const int T = (int) win.size();
        int keep = 0;
        bool done = false;
        for (int t = 0; t < T; ++t) {
            const int32_t tok = hooks.pick(logits.data() + V * (size_t) t);
            keep = t + 1;
            if (!hooks.emit(tok)) {
                done = true;
                break;
            }
            y = tok;
            if (t + 1 < T && tok != win[(size_t) t + 1]) break;   // the first draft the model would not write
        }
        if (!model.accept(keep, err)) return false;
        if (done_round)
            done_round(T - 1, keep - 1,
                       std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        if (done) return true;
    }
}

bool decode_mtp(GlmMetal& model, int32_t y, DraftLength& len, const DecodeHooks& hooks, std::string& err) {
    int chosen = 0;
    const Drafter mtp = [&](int32_t tok, int64_t p0, int n_max, std::vector<int32_t>& drafts, std::string& e) {
        chosen = std::min(len.choose(), n_max);
        if (model.mtp_hidden_pos() != p0 - 1) chosen = 0;   // the NextN block reads the trunk's state of p0 - 1
        for (int i = 0; i < chosen; ++i) {   // chained greedy steps: step i at p0 - 1 + i with the token after it
            int32_t d = 0;
            if (!model.mtp_step(i == 0 ? tok : drafts.back(), p0 - 1 + i, i == 0, d, e)) return false;
            drafts.push_back(d);
        }
        return true;
    };
    return decode_spec(model, y, 15, mtp, [&](int n, int acc, double ms) { len.record(chosen, std::min(acc, n), ms); },
                       hooks, err);
}

}  // namespace maya::metal
