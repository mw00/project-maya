// src/core/glm_ngram_draft.cpp - glm-dsa B14.13: the n-gram draft matcher, its bookkeeping and the decode_spec session
// (include/strata/core/glm_ngram_draft.hpp).
#include "strata/core/glm_env.hpp"
#include "strata/core/glm_ngram_draft.hpp"
#include "strata/core/glm_req_knobs.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace strata::core::glmng {

namespace {

int env_int(const char* k, int dflt, int lo, int hi) {
    return (int) std::max<long long>(lo, std::min<long long>(hi, strata::glmenv::env_int(k, dflt)));
}

std::mutex g_mu;
std::unordered_map<const void*, std::vector<int32_t>>& registry() {
    static std::unordered_map<const void*, std::vector<int32_t>> r;
    return r;
}

}  // namespace

NgramConfig ngram_config_env() {
    NgramConfig c;
    // a request's GEN ngram= (the speed plan's knobs, glm_req_knobs.hpp) wins over the switch for that request
    c.on = glmknob::get(glmknob::Ngram, strata::glmenv::env_flag("MAYA_GLM_NGRAM_DRAFT", false) ? 1 : 0) != 0;
    c.n_max = env_int("MAYA_GLM_NGRAM_MAX", 4, 1, kMaxDraft);
    c.min_match = env_int("MAYA_GLM_NGRAM_MIN", 3, 1, 16);
    c.max_match = std::max(c.max_match, c.min_match);
    return c;
}

// ---------------------------------------------------------------- the matcher
NgramMatcher::NgramMatcher(const NgramConfig& c) : cfg_(c) {
    cfg_.n_max = std::max(1, std::min(kMaxDraft, cfg_.n_max));
    cfg_.min_match = std::max(1, cfg_.min_match);
    cfg_.max_match = std::max(cfg_.max_match, cfg_.min_match);
    cfg_.candidates = std::max(1, cfg_.candidates);
}

void NgramMatcher::reset() {
    hist_.clear();
    ends_.clear();
    last_match_ = 0;
    last_src_ = -1;
}

uint64_t NgramMatcher::gram_key(size_t end) const {
    uint64_t h = 1469598103934665603ull;   // FNV-1a over the gram's token ids
    for (size_t i = end - (size_t) cfg_.min_match; i < end; ++i) {
        const uint32_t t = (uint32_t) hist_[i];
        for (int b = 0; b < 4; ++b) h = (h ^ ((t >> (8 * b)) & 0xffu)) * 1099511628211ull;
    }
    return h;
}

void NgramMatcher::append(const int32_t* tokens, size_t n) {
    for (size_t k = 0; k < n; ++k) {
        hist_.push_back(tokens[k]);
        const size_t end = hist_.size();
        if (end >= (size_t) cfg_.min_match) ends_[gram_key(end)].push_back((uint32_t) end);
    }
}

int NgramMatcher::propose(int32_t* out, int max_k) {
    last_match_ = 0;
    last_src_ = -1;
    const size_t E = hist_.size();
    const int want = max_k < 0 ? cfg_.n_max : std::min(max_k, cfg_.n_max);
    if (want <= 0 || E < (size_t) cfg_.min_match + 1) return 0;
    const auto it = ends_.find(gram_key(E));
    if (it == ends_.end()) return 0;
    const std::vector<uint32_t>& ends = it->second;
    int best_len = 0;
    size_t best_end = 0;
    int seen = 0;
    for (size_t k = ends.size(); k-- > 0 && seen < cfg_.candidates;) {
        const size_t e = ends[k];
        if (e >= E) continue;   // the current suffix itself
        ++seen;
        int len = 0;   // backwards agreement of hist_[.. e) with hist_[.. E)
        while (len < cfg_.max_match && (size_t) len < e && hist_[e - 1 - (size_t) len] == hist_[E - 1 - (size_t) len])
            ++len;
        if (len >= cfg_.min_match && len > best_len) {   // strictly longer: the most recent wins equal lengths
            best_len = len;
            best_end = e;
        }
    }
    if (best_len == 0) return 0;
    const int n = (int) std::min<size_t>((size_t) want, E - best_end);
    for (int i = 0; i < n; ++i) out[i] = hist_[best_end + (size_t) i];
    last_match_ = best_len;
    last_src_ = (int64_t) best_end;
    return n;
}

// ---------------------------------------------------------------- the bookkeeping
bool DraftBook::verdict(bool hit) {
    if (pending.empty()) return false;
    const int s = (int) pending.front();
    pending.pop_front();
    ++drafts[s];
    hits[s] += hit ? 1 : 0;
    ema[s] += kAlpha * ((hit ? 1.0 : 0.0) - ema[s]);
    return true;
}

bool DraftBook::prefer_ngram() const {
    if (drafts[(int) DraftSrc::Ngram] < (uint64_t) kWarmup) return true;
    return ema[(int) DraftSrc::Ngram] >= ema[(int) DraftSrc::Mtp] - kMargin;
}

double DraftBook::rate(DraftSrc s) const {
    const int i = (int) s;
    return drafts[i] ? (double) hits[i] / (double) drafts[i] : 0.0;
}

std::string DraftBook::line() const {
    char b[320];
    std::snprintf(b, sizeof b,
                  "n-gram drafts %llu (accepted %.1f%%, %llu equal to the MTP draft), MTP drafts %llu (accepted %.1f%%); "
                  "no match %llu, declined %llu",
                  (unsigned long long) drafts[1], 100.0 * rate(DraftSrc::Ngram), (unsigned long long) agree,
                  (unsigned long long) drafts[0], 100.0 * rate(DraftSrc::Mtp), (unsigned long long) no_match,
                  (unsigned long long) declined);
    return b;
}

// ---------------------------------------------------------------- the decode_spec session
NgramSession::NgramSession(const void* model) : m_(ngram_config_env()) {
    on_ = m_.config().on;
    ad_floor_ = on_ ? draft_adaptive_floor() : 0;   // B15.9
    if (ad_floor_ > 0) ad_.reset(m_.config().n_max, ad_floor_);
    std::vector<int32_t> h = take_history(model);
    if (on_ && !h.empty()) m_.append(h.data(), h.size());
}

NgramSession::~NgramSession() {
    if (on_ && (book_.drafts[0] + book_.drafts[1]) > 0) std::fprintf(stderr, "glm ngram: %s\n", book_.line().c_str());
    if (ad_floor_ > 0 && ad_.rounds > 0)
        std::fprintf(stderr, "glm ngram adaptive depth: %d at the end (floor %d, cap %d), %llu rounds, %llu climbs, %llu "
                             "drops; a depth-k verify would have kept %llu of %llu drafted tokens\n", ad_.depth(),
                     ad_floor_, m_.config().n_max, (unsigned long long) ad_.rounds, (unsigned long long) ad_.climbs,
                     (unsigned long long) ad_.drops, (unsigned long long) ad_acc_, (unsigned long long) ad_rows_);
}

void NgramSession::push(int32_t tok) {
    if (!on_) return;
    m_.append(tok);
    int nd = 0, na = 0;
    if (ad_floor_ > 0 && tr_.push(tok, nd, na)) {   // B15.9: a tracked proposal resolved
        ad_.update(nd, na, m_.config().n_max, ad_floor_);
        ad_rows_ += (uint64_t) nd;
        ad_acc_ += (uint64_t) na;
    }
}

int32_t NgramSession::pick(int32_t mtp_draft) {
    prop_.clear();
    if (!on_) return mtp_draft;
    mtp_.push_back(mtp_draft);
    int32_t buf[kMaxDraft];
    const int n = m_.propose(buf, ad_floor_ > 0 ? ad_.depth() : -1);   // B15.9: the adaptive depth
    if (n <= 0) {
        ++book_.no_match;
        book_.issued(DraftSrc::Mtp);
        return mtp_draft;
    }
    if (!book_.prefer_ngram()) {
        ++book_.declined;
        book_.issued(DraftSrc::Mtp);
        return mtp_draft;
    }
    prop_.assign(buf, buf + n);
    if (ad_floor_ > 0) tr_.issue(buf, n);   // B15.9: one tracked round at a time
    book_.agree += buf[0] == mtp_draft ? 1 : 0;
    book_.issued(DraftSrc::Ngram);
    return buf[0];
}

void NgramSession::verdict(int32_t draft, int32_t truth) {
    if (!on_) return;
    book_.verdict(draft == truth);
    if (!mtp_.empty()) mtp_.pop_front();
}

void set_history(const void* model, const std::vector<int32_t>& tokens) {
    if (!ngram_config_env().on) return;   // off (every default run, Flash too): nothing kept, nothing copied
    std::lock_guard<std::mutex> lk(g_mu);
    if (tokens.empty())
        registry().erase(model);
    else
        registry()[model] = tokens;
}

void set_history(const void* model, const std::vector<int32_t>& prompt, const std::vector<int32_t>& emitted) {
    if (!ngram_config_env().on) return;   // off (every default run, Flash too): nothing kept, nothing copied
    std::vector<int32_t> h;
    h.reserve(prompt.size() + emitted.size());
    h.insert(h.end(), prompt.begin(), prompt.end());
    h.insert(h.end(), emitted.begin(), emitted.end());
    set_history(model, h);
}

std::vector<int32_t> take_history(const void* model) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = registry().find(model);
    if (it == registry().end()) return {};
    std::vector<int32_t> v = std::move(it->second);
    registry().erase(it);
    return v;
}

}  // namespace strata::core::glmng
