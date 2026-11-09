// src/core/glm_heat_evict.cpp - glm-dsa B15.8: heat-decay expert promotion / eviction (include/strata/core/
// glm_heat_evict.hpp): the heat table, the LRU-window victim, upload coalescing, the engine registry and the
// policy simulator.  Plain C++.
#include "strata/core/glm_env.hpp"
#include "strata/core/glm_heat_evict.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <numeric>

namespace strata::core::glmheat {

bool heat_on() {
    static const bool on = [] {
        return strata::glmenv::env_flag("STRATA_GLM_HEAT_EVICT", false);
    }();
    return on;
}

double heat_margin() {
    static const double m = [] {
        const char* v = std::getenv("STRATA_GLM_HEAT_MARGIN");
        const double x = v != nullptr ? std::atof(v) : 0.0;
        return x > 0.0 ? x : kMarginDefault;
    }();
    return m;
}

bool HeatTable::swap_ok(uint32_t hot, uint32_t cold, int il, double margin) {
    if ((double) hot < (double) cold + 16.0 * margin) return false;
    if (il < 0) return true;
    if ((size_t) il >= swaps_.size()) swaps_.resize((size_t) il + 1, 0);
    if (swaps_[(size_t) il] >= kInsertCap) return false;
    ++swaps_[(size_t) il];
    return true;
}

void HeatTable::tick() {
    scale_ /= (double) kDecay;
    ++ticks_;
    if (scale_ > 1e30) {
        for (double& v : h_) v /= scale_;
        scale_ = 1.0;
    }
}

uint32_t HeatTable::quant(size_t key) const {
    const double q = heat(key) * 16.0;
    return q >= 4294967295.0 ? 0xffffffffu : (uint32_t) (q + 0.5);
}

int lru_window_victim(const HeatTable& H, int n, const char* st, char resident, const int* key, const uint64_t* tick,
                      size_t key_base, int window) {
    // the `window` resident slots with the oldest ticks (a partial selection; n is a layer's slot count)
    std::vector<int> res;
    res.reserve((size_t) n);
    for (int s = 0; s < n; ++s)
        if (st[s] == resident && key[s] >= 0) res.push_back(s);
    if (res.empty()) return -1;
    const size_t w = std::min<size_t>((size_t) std::max(1, window), res.size());
    std::partial_sort(res.begin(), res.begin() + (long) w, res.end(),
                      [&](int a, int b) { return tick[a] != tick[b] ? tick[a] < tick[b] : a < b; });
    int best = res[0];
    double bh = H.heat(key_base + (size_t) key[best]);
    for (size_t i = 1; i < w; ++i) {
        const int s = res[i];
        const double h = H.heat(key_base + (size_t) key[s]);
        if (h < bh) bh = h, best = s;   // ties keep the older tick (earlier in the window)
    }
    return best;
}

std::vector<Run> coalesce(std::vector<Up> ups, const std::function<bool(int, const void*)>& plain) {
    std::sort(ups.begin(), ups.end(), [](const Up& a, const Up& b) { return a.il != b.il ? a.il < b.il : a.dst < b.dst; });
    std::vector<Run> out;
    for (const Up& u : ups) {
        if (!out.empty()) {
            Run& r = out.back();
            if (r.il == u.il && r.dst + r.bytes == u.dst && r.src + r.bytes == u.src && plain && plain(u.il, u.src) &&
                plain(r.il, r.src)) {
                r.bytes += u.bytes;
                r.tags.push_back(u.tag);
                continue;
            }
        }
        out.push_back(Run{u.il, u.dst, u.src, u.bytes, {u.tag}});
    }
    return out;
}

namespace {
std::mutex g_mu;
std::map<const void*, HeatTable> g_tab;
std::map<const void*, std::vector<Up>> g_ups;   // deferred promotion uploads, by table
}  // namespace

HeatTable* table(const void* model, size_t n_keys) {
    if (!heat_on()) return nullptr;
    std::lock_guard<std::mutex> lk(g_mu);
    HeatTable& t = g_tab[model];
    if (t.size() != n_keys) t.resize(n_keys);
    return &t;
}

void forget(const void* model) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_tab.find(model);
    if (it == g_tab.end()) return;
    g_ups.erase(&it->second);
    g_tab.erase(it);
}

HeatTable* route(const void* model, size_t n_keys, int il) {
    HeatTable* H = table(model, n_keys);
    if (H != nullptr) H->route_layer(il);
    return H;
}

HeatTable* boundary(const void* model, size_t n_keys) {
    HeatTable* H = table(model, n_keys);
    if (H != nullptr) H->begin_boundary();
    return H;
}

bool defer_up(HeatTable* H, int il, uint8_t* dst, const uint8_t* src, size_t bytes, void* ev) {
    if (H == nullptr) return false;
    std::lock_guard<std::mutex> lk(g_mu);
    g_ups[H].push_back(Up{il, dst, src, bytes, ev});
    return true;
}

std::pair<size_t, size_t> flush_ups(HeatTable* H, const std::function<void(int, uint8_t*, const uint8_t*, size_t)>& copy,
                                    const std::function<void(void*)>& record,
                                    const std::function<bool(int, const void*)>& plain) {
    if (H == nullptr) return {0, 0};
    std::vector<Up> ups;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_ups.find(H);
        if (it == g_ups.end()) return {0, 0};
        ups.swap(it->second);
    }
    const size_t n = ups.size();
    const std::vector<Run> runs = coalesce(std::move(ups), plain);
    for (const Run& r : runs) {
        copy(r.il, r.dst, r.src, r.bytes);
        for (void* ev : r.tags) record(ev);
    }
    return {runs.size(), n};
}

// ---------------------------------------------------------------- the simulator
namespace {

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uni() { return ((double) (next() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
    int below(int n) { return (int) (next() % (uint64_t) n); }
};

std::vector<int> permutation(Rng& r, int n) {
    std::vector<int> p((size_t) n);
    std::iota(p.begin(), p.end(), 0);
    for (int i = n - 1; i > 0; --i) std::swap(p[(size_t) i], p[(size_t) r.below(i + 1)]);
    return p;
}

}  // namespace

Trace synth_trace(Drift d, int n_layers, int n_expert, int top_k, int64_t tokens, int period, double zipf_s,
                  uint64_t seed) {
    Trace t;
    t.n_layers = n_layers;
    t.n_expert = n_expert;
    t.top_k = top_k;
    t.ids.reserve((size_t) (tokens * n_layers * top_k));
    Rng r(seed);
    constexpr int kTopics = 4;
    // rank -> expert per (topic, layer); a topic's popularity w(rank) = 1 / (rank + 1)^s
    std::vector<std::vector<int>> perm((size_t) kTopics * n_layers);
    for (auto& p : perm) p = permutation(r, n_expert);
    std::vector<double> w((size_t) n_expert);
    for (int i = 0; i < n_expert; ++i) w[(size_t) i] = 1.0 / std::pow((double) (i + 1), zipf_s);
    std::vector<std::pair<double, int>> race((size_t) n_expert);
    const int walk = std::max(1, period / 32);
    for (int64_t tok = 0; tok < tokens; ++tok) {
        const int topic = d == Drift::Topics ? (int) ((tok / std::max(1, period)) % kTopics) : 0;
        if (d == Drift::Gradual && tok > 0 && tok % walk == 0)
            for (int il = 0; il < n_layers; ++il) {
                auto& p = perm[(size_t) il];
                for (int k = 0; k < 3; ++k) {   // a hot rank trades places with a random one
                    const int a = r.below(std::max(1, n_expert / 4)), b = r.below(n_expert);
                    std::swap(p[(size_t) a], p[(size_t) b]);
                }
            }
        for (int il = 0; il < n_layers; ++il) {
            const auto& p = perm[(size_t) topic * n_layers + il];
            for (int rk = 0; rk < n_expert; ++rk) race[(size_t) rk] = {-std::log(r.uni()) / w[(size_t) rk], p[(size_t) rk]};
            std::partial_sort(race.begin(), race.begin() + top_k, race.end());
            for (int k = 0; k < top_k; ++k) t.ids.push_back((uint16_t) race[(size_t) k].second);
        }
    }
    return t;
}

std::string SimResult::line() const {
    char b[256];
    std::snprintf(b, sizeof b, "hit %.2f%% (%llu / %llu), promotions %llu, demotions %llu, upload copies %llu -> %llu "
                               "coalesced", 100.0 * hit_rate(), (unsigned long long) hits, (unsigned long long) visits,
                  (unsigned long long) promotions, (unsigned long long) demotions, (unsigned long long) copies,
                  (unsigned long long) runs);
    return b;
}

SimResult simulate(const Trace& t, Policy pol, const SimConfig& c) {
    const int L = t.n_layers, E = t.n_expert, K = t.top_k, P = c.slots;
    enum : char { kFree = 0, kResident = 1, kDraining = 3, kLanding = 5 };
    struct Layer {
        std::vector<int> key;
        std::vector<char> st;
        std::vector<uint64_t> tick;
    };
    std::vector<Layer> lp((size_t) L);
    for (auto& l : lp) {
        l.key.assign((size_t) P, -1);
        l.st.assign((size_t) P, kFree);
        l.tick.assign((size_t) P, 0);
    }
    std::vector<int> slot_of((size_t) L * E, -1);
    std::vector<uint32_t> cnt((size_t) L * E, 0);
    uint64_t cnt_events = 0, clock = 0;
    HeatTable H;
    H.resize((size_t) L * E);
    struct Move {
        int il, slot, e;
        bool up;
    };
    std::vector<Move> inflight;
    SimResult R;
    const bool heat = pol == Policy::Heat;
    const int rule = heat ? c.heat_rule : 2;
    // the ranking and swap score: rule 0 the quantized heat, 1 the heat (margin rule below), 2 Maya's aged count
    const auto score = [&](size_t key) -> uint32_t { return rule == 2 ? cnt[key] : H.quant(key); };
    for (int64_t tok = 0; tok < t.tokens(); ++tok) {
        // the boundary: landed moves go live
        for (const Move& m : inflight) {
            Layer& l = lp[(size_t) m.il];
            if (m.up) {
                l.st[(size_t) m.slot] = kResident;
                l.key[(size_t) m.slot] = m.e;
                l.tick[(size_t) m.slot] = clock;
                slot_of[(size_t) m.il * E + m.e] = m.slot;
            } else {
                l.st[(size_t) m.slot] = kFree;
                l.key[(size_t) m.slot] = -1;
                slot_of[(size_t) m.il * E + m.e] = -1;   // live in VRAM until the demotion's copy landed
            }
        }
        inflight.clear();
        H.begin_boundary();
        // the background moves (glm_fast_path.cu fast_boundary step 5)
        struct Cand {
            uint32_t hot, cold;
            int il, e, free_slot, cold_slot;
        };
        std::vector<Cand> cands;
        for (int il = 0; il < L; ++il) {
            Layer& l = lp[(size_t) il];
            const size_t base = (size_t) il * E;
            int frees[8], nfree = 0, cs = -1;
            uint32_t cold = UINT32_MAX;
            for (int s = 0; s < P; ++s) {
                if (l.st[(size_t) s] == kFree) {
                    if (nfree < 8) frees[nfree++] = s;
                } else if (l.st[(size_t) s] == kResident) {
                    const uint32_t v = score(base + (size_t) l.key[(size_t) s]);
                    if (v < cold) cold = v, cs = s;
                }
            }
            if (heat) {
                cs = lru_window_victim(H, P, l.st.data(), kResident, l.key.data(), l.tick.data(), base);
                cold = cs >= 0 ? score(base + (size_t) l.key[(size_t) cs]) : UINT32_MAX;
            }
            const int want = std::max(1, nfree);
            std::vector<Cand> top;
            for (int e = 0; e < E; ++e) {
                if (slot_of[base + (size_t) e] >= 0) continue;   // resident or draining
                top.push_back(Cand{score(base + (size_t) e), cold, il, e, -1, -1});
            }
            std::partial_sort(top.begin(), top.begin() + std::min<long>(want, (long) top.size()), top.end(),
                              [](const Cand& a, const Cand& b) { return a.hot != b.hot ? a.hot > b.hot : a.e < b.e; });
            top.resize(std::min<size_t>((size_t) want, top.size()));
            if (heat)   // adjacent experts into adjacent free slots: their uploads coalesce
                std::sort(top.begin(), top.end(), [](const Cand& a, const Cand& b) { return a.e < b.e; });
            for (size_t j = 0; j < top.size(); ++j) {
                Cand cd = top[j];
                cd.free_slot = (int) j < nfree ? frees[j] : -1;
                cd.cold_slot = j == 0 ? cs : -1;
                if (cd.free_slot >= 0 || cd.cold_slot >= 0) cands.push_back(cd);
            }
        }
        std::stable_sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.hot > b.hot; });
        int issued = 0, filled = 0;
        const int up_max = std::max(c.bg_n, c.bg_fill);
        std::vector<int> swaps((size_t) L, 0);
        std::vector<std::pair<int, std::pair<int, int>>> ups;   // (il, (slot, expert))
        for (const Cand& cd : cands) {
            if (issued >= c.bg_n && filled >= up_max) break;
            Layer& l = lp[(size_t) cd.il];
            if (cd.free_slot >= 0) {
                if (filled >= up_max) continue;
                l.st[(size_t) cd.free_slot] = kLanding;
                inflight.push_back({cd.il, cd.free_slot, cd.e, true});
                ups.push_back({cd.il, {cd.free_slot, cd.e}});
                ++filled;
                ++R.promotions;
            } else if (cd.cold_slot >= 0 && issued < c.bg_n &&
                       (rule == 1 ? H.swap_ok(cd.hot, cd.cold, cd.il, c.heat_margin)   // the engine's rule
                                  : cd.hot >= 2u * std::max<uint32_t>(cd.cold, 1u) + 4u &&
                                        (!heat || swaps[(size_t) cd.il] < kInsertCap))) {
                const int vkey = l.key[(size_t) cd.cold_slot];
                l.st[(size_t) cd.cold_slot] = kDraining;   // still live until the copy landed
                inflight.push_back({cd.il, cd.cold_slot, vkey, false});
                ++issued;
                ++swaps[(size_t) cd.il];
                ++R.demotions;
            }
        }
        // the uploads: one copy each, or runs of adjacent (slot, expert) pairs when coalescing
        std::sort(ups.begin(), ups.end());
        R.copies += ups.size();
        for (size_t i = 0; i < ups.size(); ++i) {
            const bool cont = heat && i > 0 && ups[i].first == ups[i - 1].first &&
                              ups[i].second.first == ups[i - 1].second.first + 1 &&
                              ups[i].second.second == ups[i - 1].second.second + 1;
            if (!cont) ++R.runs;
        }
        // the token: every layer routes; hits stamp the slot (to the LRU tail) before the next boundary plans
        ++clock;
        const uint16_t* row = t.ids.data() + (size_t) tok * L * K;
        for (int il = 0; il < L; ++il) {
            H.route_layer(il);
            const size_t base = (size_t) il * E;
            for (int k = 0; k < K; ++k) {
                const int e = row[(size_t) il * K + k];
                const size_t key = base + (size_t) e;
                if (cnt[key] < (1u << 24)) ++cnt[key];
                H.hit(key);
                const int s = slot_of[key];
                const bool hit = s >= 0 && (lp[(size_t) il].st[(size_t) s] == kResident ||
                                            lp[(size_t) il].st[(size_t) s] == kDraining);
                if (hit) lp[(size_t) il].tick[(size_t) s] = clock;
                if (tok >= c.warmup) {
                    ++R.visits;
                    R.hits += hit;
                }
            }
            cnt_events += (uint64_t) K;
            if (cnt_events >= 32768) {
                cnt_events = 0;
                for (auto& v : cnt) v >>= 1;
            }
        }
    }
    return R;
}

}  // namespace strata::core::glmheat
