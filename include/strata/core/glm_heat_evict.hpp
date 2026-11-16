// include/strata/core/glm_heat_evict.hpp - glm-dsa B15.8: heat-decay expert promotion / eviction for the VRAM tier,
// behind MAYA_GLM_HEAT_EVICT=1 (default off: Maya's aged-count policy, unchanged).
//
// The policy (llama.cpp #30059 llama-moecache, #27861):
//   * heat per (layer, expert): +1 a route, decayed by 0.90 per token (per forward pass, not per layer, so deep models
//     do not forget faster).  HeatTable keeps it lazily: one global scale grows by 1/0.90 a token and a hit adds the
//     scale, so the stored values keep their order without touching every key every token (renormalised at 1e30);
//   * the victim: the coldest of the first 16 slots from the LRU head (resident slots by their last route tick),
//     instead of the coldest by count over the whole layer.  Hits are moved to the tail before the misses are planned:
//     in the engine the route handler stamps P.tick before the boundary plans its moves, which is that order;
//   * a swap (demote the victim, promote the hottest RAM-only expert) needs the candidate's heat to exceed the
//     victim's by MAYA_GLM_HEAT_MARGIN (default 2.0 routes' worth): Maya's 2x + 4 count rule on a 0.90-decayed heat
//     churned bg_n swaps every token in the simulator and lost 4 points on a stationary trace;
//   * at most 2 swaps (demote + promote) per layer per token (#27861's insert cap), the global bg_n cap unchanged;
//   * coalesced uploads: promotions sorted by (layer, VRAM slot) go in one copy when their VRAM slots and their RAM
//     slots are both adjacent (Up / coalesce; a node-split RAM class never merges: its copies are 2D).
// Maya's policy (the "current" one): aged counts (halved every 32768 route events), the coldest resident of the layer
// by count is demoted when a RAM-only expert has 2x + 4 its count, background refill of free slots.
//
// The simulator (simulate / synth_trace) replays synthetic routing traces through both policies with the engine's
// boundary mechanics (moves land one token later, refill of free slots, bg_n per token); BUILD-PLAN B15.8 adopts the
// heat policy only if the test PC's B8.5 hit-rate curve shows topic drift costing >= 2 points.
//
// Plain C++ (no CUDA).  tests/glm_dsa/glm_heat_evict_test.cpp runs the simulator and the unit checks.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace strata::core::glmheat {

constexpr float kDecay = 0.90f;
constexpr int kVictims = 16;
constexpr int kInsertCap = 2;
constexpr double kMarginDefault = 2.0;

/// MAYA_GLM_HEAT_EVICT set to a non-zero value (read once).
bool heat_on();
/// MAYA_GLM_HEAT_MARGIN (heat units, > 0; default 2.0, read once).
double heat_margin();

/// Lazily decayed heat per key (il * n_expert + e).
class HeatTable {
public:
    void resize(size_t n_keys) { h_.assign(n_keys, 0.0); }
    size_t size() const { return h_.size(); }
    /// A new token (forward pass): every heat decays by kDecay.
    void tick();
    /// A route to key.
    void hit(size_t key, double w = 1.0) {
        if (key < h_.size()) h_[key] += w * scale_;
    }
    /// The heat now (decayed).
    double heat(size_t key) const { return key < h_.size() ? h_[key] / scale_ : 0.0; }
    /// The heat as a count-like integer (x16, saturating), for code written against Maya's uint32 counts.
    uint32_t quant(size_t key) const;
    /// A route for layer il: a layer index not above the previous one starts a new forward pass (tick first).
    void route_layer(int il) {
        if (il <= last_il_) tick();
        last_il_ = il;
    }
    uint64_t ticks() const { return ticks_; }
    /// A boundary (the background moves' planning) starts: the per-layer swap counts restart.
    void begin_boundary() { swaps_.clear(); }
    /// The swap rule (hot / cold are quant()s): hot >= cold + 16 * margin, at most kInsertCap swaps per layer per
    /// boundary (counted when true).
    bool swap_ok(uint32_t hot, uint32_t cold, int il, double margin);

private:
    std::vector<int> swaps_;
    std::vector<double> h_;
    double scale_ = 1.0;
    int last_il_ = 1 << 30;
    uint64_t ticks_ = 0;
};

/// The victim among a layer's slots: of the resident slots (st[s] == resident), the kVictims (window) with the oldest
/// ticks, the one with the least heat (key_base + key[s]); ties to the older tick.  -1 when none is resident.
int lru_window_victim(const HeatTable& H, int n, const char* st, char resident, const int* key, const uint64_t* tick,
                      size_t key_base, int window = kVictims);

/// Coalescing: one promotion's copy.
struct Up {
    int il = 0;
    uint8_t* dst = nullptr;
    const uint8_t* src = nullptr;
    size_t bytes = 0;
    void* tag = nullptr;   // the engine's event (recorded after the run's copy)
};
struct Run {
    int il = 0;
    uint8_t* dst = nullptr;
    const uint8_t* src = nullptr;
    size_t bytes = 0;
    std::vector<void*> tags;
};
/// Sorts by (layer, dst) and merges neighbours whose dst and src both continue the run (plain(il, src) must allow it:
/// false for a node-split class).
std::vector<Run> coalesce(std::vector<Up> ups, const std::function<bool(int, const void*)>& plain);

/// Engine registry: the table of a model part (sized on first use), or null when the switch is off.
HeatTable* table(const void* model, size_t n_keys);
void forget(const void* model);
/// Maya's count, or the quantized heat when the switch is on (H non-null).
inline uint32_t count(const HeatTable* H, size_t key, uint32_t maya_cnt) { return H ? H->quant(key) : maya_cnt; }

// ---- engine hooks (src/core/glm_fast_path.cu; every one a no-op when H is null, i.e. the switch is off)
/// The route handler: the table (null when off) after noting a route of layer il (a new forward pass ticks).
HeatTable* route(const void* model, size_t n_keys, int il);
/// fast_boundary step 5: the table (null when off), its per-layer swap counts restarted.
HeatTable* boundary(const void* model, size_t n_keys);
/// The demotion victim: with H, the coldest of the LRU window replaces Maya's coldest-by-count (cs, cold).
inline void window_victim(const HeatTable* H, int n, const char* st, char resident, const int* key, const uint64_t* tick,
                          size_t key_base, int& cs, uint32_t& cold) {
    if (H == nullptr) return;
    cs = lru_window_victim(*H, n, st, resident, key, tick, key_base);
    cold = cs >= 0 ? H->quant(key_base + (size_t) key[cs]) : 0xffffffffu;
}
/// The swap rule: Maya's (hot >= ratio * max(cold, 1) + 4) without H, heat's (margin, per-layer cap) with it.
inline bool swap_ok(HeatTable* H, uint32_t hot, uint32_t cold, uint32_t ratio, int il) {
    if (H == nullptr) return hot >= ratio * (cold > 1u ? cold : 1u) + 4u;
    return H->swap_ok(hot, cold, il, heat_margin());
}
/// Promotion uploads: with H the copy is deferred (true: the caller issues nothing and records no event) and
/// flush_ups issues the coalesced runs; without H false (the caller copies as before).
bool defer_up(HeatTable* H, int il, uint8_t* dst, const uint8_t* src, size_t bytes, void* ev);
/// Issues the deferred uploads: copy(il, dst, src, bytes) per run, then record(ev) for each of the run's events;
/// plain(il, src) says a RAM slot may be copied with others (not a node-split class).  Returns (copies, uploads).
std::pair<size_t, size_t> flush_ups(HeatTable* H, const std::function<void(int, uint8_t*, const uint8_t*, size_t)>& copy,
                                    const std::function<void(void*)>& record,
                                    const std::function<bool(int, const void*)>& plain);

// ---------------------------------------------------------------- the simulator
struct Trace {
    int n_layers = 0, n_expert = 0, top_k = 0;
    std::vector<uint16_t> ids;   // [token][layer][k]
    int64_t tokens() const { return n_layers > 0 && top_k > 0 ? (int64_t) ids.size() / (n_layers * top_k) : 0; }
};
enum class Drift { Stationary, Topics, Gradual };
/// Synthetic routing: Zipf(s) popularity over a per-(layer, topic) permutation, top_k distinct experts a token
/// (exponential races).  Topics: 4 recurring topics, switching every `period` tokens; Gradual: every `period` / 32
/// tokens a few rank swaps (a slow walk).  Deterministic in seed.
Trace synth_trace(Drift d, int n_layers, int n_expert, int top_k, int64_t tokens, int period, double zipf_s,
                  uint64_t seed);

enum class Policy { Maya, Heat };
struct SimConfig {
    int slots = 96;            // VRAM slots per layer
    int bg_n = 8;              // demotions a token (MAYA_GLM_PROMOTE)
    int bg_fill = 24;          // promotions into free slots a token (MAYA_GLM_PROMOTE_FILL)
    int64_t warmup = 1000;     // tokens before the hit rate counts
    int heat_rule = 1;         // Heat: 0 Maya's 2x + 4 rule on the x16 heat, 1 (the engine's) heat >= victim's +
                               // heat_margin, 2 the LRU-window heat victim with Maya's aged counts deciding the swap
    double heat_margin = kMarginDefault;
};
struct SimResult {
    uint64_t visits = 0, hits = 0, promotions = 0, demotions = 0, copies = 0, runs = 0;
    double hit_rate() const { return visits ? (double) hits / (double) visits : 0.0; }
    std::string line() const;
};
/// Replays the trace through the policy with the engine's boundary mechanics (exclusive tiers, the CPU lane serves
/// every non-resident expert, moves land one token after they are issued).  RAM slots are expert-ordered (B8.2's
/// segmented pinning per layer), so promotions of adjacent experts into adjacent VRAM slots coalesce.
SimResult simulate(const Trace& t, Policy p, const SimConfig& c);

}  // namespace strata::core::glmheat
