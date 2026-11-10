// include/strata/core/glm_dsa_spec.hpp - the glm-dsa port B11.2 / B11.3: the pipelined speculative decode on 3 GPUs.
//
// Maya's pipelined decode (glm_fast_path.cu spec_ready / decode_spec) splits the parts of a layer split into a HEAD
// GROUP (parts 0 .. h-1) and a TAIL GROUP (parts h .. n-1, the last one carrying the NextN draft block).  The head runs
// position q+2 with the draft token while the tail finishes position q+1; a rejected draft makes the head redo q+2
// with the real token.  On the test PC (3x V100) the head group is either {GPU0} or {GPU0, GPU1}.  This header holds the
// host side of that choice and of the acceptance accounting (plain C++, no CUDA; CPU tests in
// tests/glm_dsa/glm_dsa_spec_test.cpp):
//
//   spec_price       the expected time of one emitted token for a head of h parts, from each part's token time and
//                    its CPU-lane share (a CPU-lane pool shared by head and tail serialises their lane work), the
//                    draft acceptance and the hop cost; against the token-at-a-time decode (the parts added).
//   spec_head_parts  STRATA_GLM_SPEC_HEAD: a number keeps Maya's meaning (head = that many parts, clamped to
//                    [1, n-1]); "auto" (and unset, on glm-dsa) takes the priced choice recorded by the split search
//                    (glm_model.cu glm_search_bounds -> spec_record_prices); without a price: Maya's n / 2.
//   AcceptStats      B11.3: accepted / tried per draft position (depth 1 = the token after next, the only depth the
//                    pipelined path drafts; 2..4 kept for chain drafts, B11.4) and per context bucket; a running log
//                    (STRATA_GLM_SPEC_LOG=<steps>) and a per-draft trace file (STRATA_GLM_SPEC_TRACE=<path>).
//   spec_rows        what a rejected draft leaves behind on glm-dsa: the latent, k_pe and index-key rows of every DSA
//                    layer of the head at the speculative position, and nothing else.  A redo at the same position
//                    rewrites exactly these rows (the caches are append-only, cell s = position s), so no copy-back
//                    is needed (glm-dsa has no KDA state); the selection state is reset by glmdsa::mixer_rollback.
//
// Flash (glm5-next) keeps Maya's head rule byte for byte: spec_head_parts(n, env, false) is n / 2 or atoi(env).
#pragma once

#include "strata/core/glm_state_layout.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core::glmspec {

// ---------------------------------------------------------------- B11.2 pricing

/// One part's decode cost for a token (the split search's per-part price, glm_search_bounds Eval).
struct PartCost {
    int dev = 0;
    double ms = 0.0;        ///< the part's whole token time (its layers; the last part: head, and the draft block)
    double lane_ms = 0.0;   ///< of which the CPU lane computes (misses); 0 when unknown
    int pool = 0;           ///< the CPU-lane pool it runs its misses on (equal ids: one pool, serialised)
};

/// The modelled cost of one emitted token with a head group of `head` parts.
struct SpecPrice {
    int head = 0;               ///< parts in the head group
    double head_ms = 0.0;       ///< H: the head group's token (its parts and the hops between them)
    double tail_ms = 0.0;       ///< T: the tail group's token, the draft included (hops included)
    double lane_overlap_ms = 0.0;   ///< lane work of head + tail on a shared pool (0: separate pools)
    double hit_ms = 0.0;        ///< a step whose draft stands: max(T, H, shared lane)
    double miss_ms = 0.0;       ///< a step whose draft is rejected: T + H (the redo waits for the tail's token);
                                ///< redo_waits: hit + H (it also waits for the head's rejected speculation)
    double step_ms = 0.0;       ///< expected: a * hit + (1 - a) * miss
    double base_ms = 0.0;       ///< the token-at-a-time decode: every part in turn
    double speedup = 0.0;       ///< base / step
    bool shared_lane = false;   ///< a head part and a tail part share a CPU-lane pool
};

/// Prices head = `head` parts (1 <= head < parts.size()).  accept: the draft acceptance (0..1); hop_ms: one residual
/// hop between parts (device -> host -> device).  redo_waits (STRATA_GLM_SPEC_DRAFT_EARLY's pricing): a rejected
/// draft costs the hit step plus the head's redo, because decode_spec's redo first waits for the head to finish the
/// rejected speculation (spec_head synchronises the head's stream) - T + H while H <= T as before, 2H once the head
/// is the slower group; false: T + H always (the B11.2 price, the default).
SpecPrice spec_price(const std::vector<PartCost>& parts, int head, double accept, double hop_ms = 0.05,
                     bool redo_waits = false);

/// Every head size 1 .. n-1, and the index of the cheapest (smallest step; ties go to the smaller head).
std::vector<SpecPrice> spec_prices(const std::vector<PartCost>& parts, double accept, double hop_ms, int& best,
                                   bool redo_waits = false);

/// STRATA_GLM_SPEC_DRAFT_EARLY (default off; read at each call): decode_spec runs each step's NextN draft while the
/// tail group's parts before its last compute (glm_fast_path.cu spec_tail_front / spec_tail_back), and the head
/// pricing (spec_record_prices) prices a miss with redo_waits.
bool spec_draft_early();

/// The acceptance the pricing assumes: STRATA_GLM_SPEC_ACCEPT (0.05..1), else 0.8 (GLM-5.2's 0.83, R).
double spec_accept_assumed();

/// The lane share of a part's token time from the split search's terms: hits and misses (routes - hits) of its
/// routed experts, priced at the card's bandwidth and the lane's (expert bytes cancel; dense bytes are left out, so
/// the share is an upper bound).
double lane_share(double hits, double routes, double card_bps, double miss_bps);

/// Records the split search's prices for an n-part split (process-wide; the latest search wins) and logs them.
/// Returns the priced head.
int spec_record_prices(const std::vector<PartCost>& parts, bool log = true);
/// The recorded priced head for n parts, -1 when none.
int spec_priced_head(int n_parts);
/// Forgets the recorded prices (tests).
void spec_clear_prices();

/// The head group's size for an n-part split.  env: STRATA_GLM_SPEC_HEAD (nullptr = unset).  dsa: glm-dsa (Flash
/// keeps Maya's rule: n / 2, or atoi(env)).  glm-dsa unset / "auto": the priced head (else n / 2), then, on 3+ parts
/// with the tuner on, the measured A/B (head_tuner below).  Clamped to [1, n - 1].
int spec_head_parts(int n_parts, const char* env, bool dsa);

/// B11.2 in-situ A/B of the head group (glm-dsa, 3+ parts, STRATA_GLM_SPEC_HEAD unset or "auto"): the speculative
/// decode runs each candidate head for `window` emitted tokens (STRATA_GLM_SPEC_TUNE=<tokens>, default 128; 0 off),
/// the priced (or n / 2) head first, then keeps the one with the lowest measured wall time a token (cumulative, so it
/// can still switch when the averages cross).  The output is the same for every head (the decode is exact); only the
/// speed differs.  Explicit layer bounds (layer_split 26,54) have no split-search price, so this is how the test PC's
/// pinned split picks {GPU0} or {GPU0,GPU1}.
struct HeadTuner {
    int n = 0;                          ///< the split's parts (a new n resets)
    int window = 128;
    std::array<double, 8> ms{};         ///< by head (index 0: no drafting, when `plain`)
    std::array<int64_t, 8> tokens{};
    bool settled_logged = false;
    /// "no drafting" (the token-by-token decode, the same tokens) is a candidate too, as head 0: measured after the
    /// heads, kept only when it is faster than the fastest head by kPlainMargin (drafting stays the default when the
    /// two are within the noise).  STRATA_GLM_SPEC_TUNE_PLAIN=1 (off by default until measured on the GPUs).
    bool plain = false;
    static constexpr double kPlainMargin = 0.02;
    /// The head for the next decode (stable until feed): the first candidate (base, then 1 .. n-1, then 0 when
    /// `plain`) short of the window, else the fastest measured.  0 = decode token by token.
    int pick(int n_parts, int base) const;
    void feed(int head, int64_t n_tokens, double wall_ms);
    double ms_per_token(int head) const;
};
/// STRATA_GLM_SPEC_TUNE_PLAIN (default off; read once): the tuner also measures the token-by-token decode.
bool spec_tune_plain();
/// The tuner's no-drafting turn (glm-dsa, n_parts >= 3, STRATA_GLM_SPEC_HEAD unset / "auto", the tuner and
/// spec_tune_plain on, after the split's first speculative decode): true when the next decode should go token by token
/// (the plain candidate is short of its window, or measured fastest) - its clock then runs as tune_begin's, and
/// tune_end(produced) after the plain loop feeds it as head 0.  false: decode_spec as before.  head_env:
/// STRATA_GLM_SPEC_HEAD (nullptr = unset).
bool tune_plain_turn(int n_parts, const char* head_env);
/// The window from STRATA_GLM_SPEC_TUNE (0: the tuner is off).
int spec_tune_window();
/// B15.7: true when the random-routing bench mode is on (STRATA_GLM_RANDOM_ROUTING=1): Glm5Model::spec_ready() then
/// says no, as STRATA_GLM_NO_SPEC=1 does (random routes make every draft a miss, so the floor would count re-runs).
bool spec_bench_off();
/// decode_spec's start / end (process-wide; the head the call used is the one spec_head_parts returned): feeds the
/// tuner the call's emitted tokens and wall time.  An end without a start, or a call that emitted under 8 tokens,
/// feeds nothing.
void tune_begin();
void tune_end(int64_t produced);
/// The process tuner (a copy) and its reset (tests).
HeadTuner tuner_state();
void tuner_reset();

// ---------------------------------------------------------------- B11.3 acceptance

constexpr int kMaxDepth = 4;     ///< draft positions tracked (the pipelined path drafts depth 1 only)
constexpr int kCtxBuckets = 5;   ///< positions < 1K, < 4K, < 16K, < 64K, more
int ctx_bucket(int64_t pos);
const char* ctx_bucket_name(int b);

struct AcceptStats {
    std::array<uint64_t, kMaxDepth> tried{}, hit{};
    std::array<uint64_t, kCtxBuckets> tried_ctx{}, hit_ctx{};   ///< depth 1 by the speculative position
    uint64_t run = 0;           ///< the current run of consecutive accepted drafts (depth 1)
    uint64_t run_max = 0;
    uint64_t redo = 0;          ///< head redos (a rejected depth-1 draft)
    void note(int depth, int64_t pos, bool accepted);
    double rate(int depth) const;
    double rate_ctx(int b) const;
    /// One line: "depth 1 x/y (r%) | ctx <1K .. | longest accepted run k | redos m".
    std::string line() const;
};

/// The stats of one model (keyed by its address; process-wide, thread-safe).  note() also feeds the running log and
/// the trace file when their env vars are set.
AcceptStats accept_stats(const void* model);
void note(const void* model, int depth, int64_t pos, int draft, int truth);
/// Prints the model's acceptance line to stderr (when it drafted anything).  final: the unload summary.
void report(const void* model, bool final);
/// Forgets a model's stats (unload, tests).
void forget(const void* model);

// ---------------------------------------------------------------- B11.2 rollback (glm-dsa)

/// A cache row the head writes at one position.
struct SpecRow {
    enum Kind : int { Latent = 0, Kpe = 1, IndexKey = 2 };
    int il = 0;
    Kind kind = Latent;
    int64_t off = 0;      ///< floats from the arena start
    int64_t floats = 0;
};
/// The rows of every carried DSA layer of `lay` at position p (the NextN block left out unless with_mtp): latent
/// (lat_pp), k_pe (kpe_pp, when kept apart) and index keys (ik_pp, full layers).  Empty when p is outside the
/// context or the layout is not glm-dsa.
std::vector<SpecRow> spec_rows(const GlmStateLayout& lay, int64_t p, bool with_mtp = false);
/// Checks the rollback invariant of a layout: rows at distinct positions never overlap, every row lies inside the
/// arena past the residual, and each carried DSA layer has its latent (and k_pe / index-key rows where the layout
/// keeps them).  false + why.
bool spec_rows_check(const GlmStateLayout& lay, const std::vector<int64_t>& positions, std::string& why);

}  // namespace strata::core::glmspec
