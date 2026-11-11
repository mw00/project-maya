// src/k31/k31_model.hpp - maya-k31's M1 runner: Kimi K3.1's forward pass, per op (plan M1a).
//
// One GPU or a layer split over several (ModelOptions::gpus), float32 activations, the k31 kernels (k31_route / k31_situ / k31_moe_combine / k31_attn_res / k31_mla /
// k31_kda / k31_index) composed in ref/kimi_k31.py forward's order:
//
//   x = embed(tokens); prefix = x; blocks = []
//   per layer il:  cur = res_mix(blocks, prefix, attn_res_score)        (no blocks: prefix itself)
//                  if il % B == 0: blocks.push(prefix)                   (the raw incoming partial sum)
//                  a = kda(rms_norm(cur)) or mla(rms_norm(cur))          (MLA: q_a/kv_a latents, the indexer's top-k)
//                  prefix = a if il % B == 0 else prefix + a
//                  cur = res_mix(blocks, prefix, ffn_res_score)
//                  prefix += dense_ffn(rms_norm(cur)) (layer 0) or moe_ffn(rms_norm(cur))
//   logits = output @ rms_norm(res_mix(blocks, prefix, output_res_score))
//
// The block states live only inside a call: they are per token. What a sequence carries between calls is the KDA
// conv and recurrent state, the MLA latent cache (576-wide rows) and the indexer's key cache (float32: the parity
// format; BF16 is the production cache, M1c), all at the call's positions n_past ...
//
// Weights: float32 tensors on the device; BF16 ones decoded to float32 at load (their products take float inputs, as the
// oracle's); every iq::mv type (Q8_0, Q4_K, Q6_K, IQ1_S, ...) kept as ggml blocks in mapped pinned host memory and read
// in place by iq::mv with int8 inputs (ds4's q8_quant_block rounding: ref/kimi_k31.py --act-q8 32); attn_k_b / attn_v_b
// decoded to float32 (the oracle gives them float inputs); a quantized embedding decoded per token row.
// M1a scope: a float32 GGUF (the synthetic fixtures), every dense product the runner's own float32 GEMV (one warp a
// token and row, a fixed reduction: a token's bits do not depend on n_tokens, so prefill = chunked = token by
// token bitwise; cuBLAS's do), the routed experts grouped by expert on the host after the router (a device -> host sync per MoE
// layer).  Quantized weights (iq::mv for gate and up, iq::down) are M1b; a file that has them is refused at load.
// Seams are dumped under ref/kimi_k31.py's cb() names, in the layout tools/kimi_k31_gate.py check-dumps reads.
#pragma once

#include "k31_geo.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace k31 {

/// Seam dumps over one run (any number of forward calls): each seam's per-token rows, concatenated in call order,
/// written as ref/glm.py's raw layout (manifest `name d0 d1 ... T` in ggml order, features fastest, int32 for the
/// selections).  Rows of a seam whose width is the key count (indexer_score, kq_mask_top_k) are padded to the run's
/// final key count with -inf.
class Dumps {
public:
    void add(const std::string& name, const std::vector<int64_t>& feat, const float* rows, int T);
    void add_int(const std::string& name, const std::vector<int64_t>& feat, const int32_t* rows, int T);
    void add_keyed(const std::string& name, int width, const float* rows, int T);   // rows of `width` keys
    /// Keep only seams matching one of these globs (fnmatch; empty: every seam).  The runner asks before copying.
    void set_filter(std::vector<std::string> globs) { filter_ = std::move(globs); }
    bool wants(const std::string& name) const;
    bool write(const std::string& dir, std::string& err) const;
    size_t size() const { return order_.size(); }

private:
    struct Seam {
        std::vector<int64_t> feat;
        std::vector<float> f;
        std::vector<int32_t> i;
        std::vector<int> widths;   // keyed seams: each row's width
        bool is_int = false, keyed = false;
        int rows = 0;
    };
    Seam& get(const std::string& name, const std::vector<int64_t>& feat, bool is_int, bool keyed, std::string& bad);
    std::map<std::string, Seam> seams_;
    std::vector<std::string> order_;
    std::string bad_;
    std::vector<std::string> filter_;
};

struct ModelOptions {
    int n_ctx = 4096;          // positions the caches hold (--n-ctx)
    int n_layers = 0;          // layers to run (0: all); fewer: a truncated run (the cut's layers, then the head)
    bool fast_mix = false;     // reserved: the M3 attn_res / moe_combine forms (k31/p3-w4 145cef7) once merged
    // ---- placement (milestone S): a layer split over cards, the experts by usage rank in VRAM slots, the rest in RAM
    std::vector<int> gpus;     // CUDA ids, one per layer range (empty: the current device).  A repeated id splits on one
                               // card: the same bits, the cross-card moves as plain copies (how the split is tested)
    std::vector<int> split;    // the first layer of each card after the first (empty: even ranges, the first the larger)
    std::vector<int> vram_cap_mb;   // per card in gpus order: at most this much of its VRAM is used (0 or missing: all
                                    // that is free).  A card shared with another service (receipts on GPU3) gets a cap
    int expert_slots = -1;     // VRAM experts a MoE layer (-1: as many as fit, by usage rank; 0: every expert in RAM)
    int vram_reserve_mb = -1;  // per card, left for a call's scratch and the CUDA context; -1: sized from
                               // max_call_tokens and the geometry (measured: ~2.3 MB of scratch a token of a call)
    int max_call_tokens = 0;   // the most tokens one forward() call holds (the chunk, or the longest prompt in one
                               // call); 0: 512 is assumed.  Only the auto reserve reads it
    std::string usage_path;    // expert usage counts per layer (tools/k31_usage.py from a REAP report); else file order
    std::string placement_in;  // take the placement from this file (written by placement_out, maybe edited)
    std::string placement_out; // write the placement used to this file
    // ---- the CPU lane (src/k31/k31_cpu_lane): RAM experts' rows on the host, the device's bits, while the device runs
    // the others.  (The routed experts' SiTU is the portable form on both sides, so it gives the same bits too.)
    int cpu_threads = 0;       // 0: off
    double cpu_frac = 0.5;     // the share of a call's RAM experts (by count) the CPU takes
    std::vector<double> cpu_frac_cards;   // per card in gpus order (empty or < 0: cpu_frac): e.g. the CPU lane for a card
                                          // on a narrow link (GPU3 at PCIe x4), the device's zero copy on the x16 ones
    int cpu_max_tokens = 8;    // calls of more tokens leave every expert to the device (a prefill reads each once)
    bool profile = false;      // time each part of a call (waits for the device at each boundary)
    // ---- loading: the tensors are read in file order with large pread()s (a sequential stream), or (load_mmap)
    // touched through the file's mapping.  The bytes are the same; a busy ZFS pool served the mapping's page faults
    // at ~1 MB/s, a stream at its full rate.
    bool load_mmap = false;
    // ---- paged experts: the experts without a VRAM slot stay in the file's mapping (no pinned copy), so the page
    // cache is the RAM tier and the NVMe the backstop: a model larger than RAM runs.  The CPU lane reads them in place;
    // the device copies each one it runs into a staging buffer (pageable memory: no zero-copy).  The mapping stays
    // open for the model's life.  page_stats: before each call, mincore() on the routed experts' pages counts the
    // ones not resident (misses), and the process's major faults are counted per call (page_report()).
    bool page_experts = false;
    // prefetch (threads > 0): after a layer's routing, the picked paged experts' ranges that mincore() finds not
    // resident are read into the page cache by `prefetch` threads (k31_prefetch.hpp: many large reads in flight),
    // and the layer runs its resident experts first.  pregate: before layer L's experts, layer L+1's norm and router
    // are run on L's FFN input and L+1's predicted picks are prefetched (low priority) while L computes; the hit rate
    // of the prediction is counted.  Neither changes a computed bit (0 / false: today's path).
    int prefetch = 0;
    bool pregate = false;
    int pregate_rank = 0;          // pregate reads only the predicted picks of rank < this (0: all K)
    bool pregate_decode_only = false;   // no pregate on prompt calls (a prompt reads every pick it routes anyway)
    bool page_stats = false;
    // cache-aware routing (car_rank r > 0; R&D's R13): on a decode call's rows (forward(..., decode = true)) a routed
    // pick that is not resident (a paged expert with a page out of the page cache; VRAM and pinned experts always are)
    // is read only if its router rank < r.  When a pick is skipped, the token routes among the experts resident now
    // plus its read picks: the router's masked top-k (k31_route's order and weights, on the host).  Prompt calls route
    // as today.  r = n_expert_used never skips (today's bits); 0: off.
    int car_rank = 0;
    // prompt CAR (car_prompt_rank R > 0; R&D's --car-prompt-rank / --car-prompt-rows): the last car_prompt_rows rows
    // of a prompt (forward(..., car_tail)) are one batch under cache-aware routing.  An expert is available to the
    // batch when it is resident now, picked by an earlier row of the call (read in full anyway), or ranked < R by
    // some batch row (read if missing); a batch row with a pick not available routes among the available experts
    // (car()'s masked top-k on the host).  Rows before the tail route as today.  0: off (today's bits).
    int car_prompt_rank = 0;
    int car_prompt_rows = 50;
    // the RAM expert tier (k31_ramtier.hpp; with page_experts): paged experts are read with O_DIRECT into pinned,
    // device-mapped slots (an LRU keyed by layer and expert) instead of through the page cache.  ram_slots: slots
    // (-1: as many as fit in MemAvailable less 16 GB; 0: off, today's path), never more than the paged experts;
    // pinned in chunks, and the pool is what the driver granted.  ram_readers: O_DIRECT reader threads.  pregate
    // fills slots ahead (low priority); CAR counts an expert with a slot (ready or loading) as resident.
    int ram_slots = 0;
    int ram_readers = 16;
    // size classes (RAM tier and dynamic VRAM pool): tier_classes 0 (one): every slot sized for the largest gate, up
    // and down over all layers (467ff87's pool); 1 (layout): a class per distinct (gate, up, down) expert byte layout,
    // a layer's experts only in slots of its class.  The budget is bytes: ram_gb > 0 pins that many GB (1e9), else
    // ram_slots x the smallest class's slot (one class: ram_slots slots as before); MemAvailable less 16 GB caps it.
    // tier_split: the classes' shares of the budget in class order (empty: by routing-weighted paged bytes, from
    // usage_path's pick counts, else uniform per paged expert); --vram-dyn N is N x a card's smallest class slot.
    int tier_classes = 0;
    double ram_gb = 0;
    std::vector<double> tier_split;
    // the dynamic VRAM tier (with the RAM tier): per card a pool of VRAM expert slots, an LRU by layer and expert.  On
    // a decode call a RAM tier hit is copied to its layer's card on a copy stream while the call runs (it computes from
    // its RAM slot this time); once the copy is done the expert runs from VRAM and its RAM slot is freed.  A VRAM
    // victim goes back to a RAM slot (device to host on the same stream), or to the mapping when none is free.  The
    // bytes are the file's, so no computed bit changes.  vram_dyn: slots a card (-1: all VRAM free after the load less
    // the call reserve; 0: off); vram_dyn_up: at most this many uploads a layer call.  Static slots (--expert-slots,
    // --placement) stay as they are: --expert-slots 0 gives all of a card's expert VRAM to the pool.
    // dense_override (a GGUF from tools/k31_dense_requant.py): its tensors replace the model's of the same name and
    // shape (dense weights in another type, e.g. Q4_K for Q6_K); routed experts are refused.  Empty: off.
    std::string dense_override;
    // tier_prompt_cold (the RAM tier): a prompt call's reads go in at the LRU's cold end and its hits keep their age, so
    // a long prompt (a reuse prefill can touch twice the pool) evicts its own reads before the decode working set
    bool tier_prompt_cold = false;
    int vram_dyn = 0;
    int vram_dyn_up = 64;
    // remote experts (k31_remote.hpp): the picks of experts an expert server owns are computed there.  remote_addr:
    // HOST:PORT (empty: off); the owner map: remote_own_path (lines "layer e e ...") or remote_frac P (a share of each
    // MoE layer's experts without a VRAM slot: the most used by usage_path if given, else a fixed hash of the ids)
    std::string remote_addr;
    std::string remote_own_path;
    double remote_frac = 0.0;
};

class Model {
public:
    Model();
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    /// Reads the GGUF's geometry (k31::read) and uploads its tensors.  false + err on anything M1a cannot run.
    bool load(const std::string& gguf_path, const ModelOptions& opt, std::string& err, EpsPreset eps = EpsPreset::moonshot);
    /// A fresh sequence: n_past 0, the KDA states and caches cleared.
    void reset();
    /// The sequence's state as bytes (n_past; the MLA and indexer cache rows 0..n_past of each attention layer; each
    /// KDA layer's conv and recurrent state; a geometry header), and back: a sequence restored from it continues as the
    /// saved one would (the same bits in every cache).  load_state replaces the sequence; a state from another geometry,
    /// or longer than n_ctx, is refused with err (the sequence is then reset).
    bool save_state(std::vector<uint8_t>& out, std::string& err);
    bool load_state(const std::vector<uint8_t>& in, std::string& err);
    /// Runs `tokens` as the continuation of the sequence.  logits (optional): the call's last R rows [R][n_vocab],
    /// R = T - (T-1)/16*16 (1..16; read the last row), or all T rows when dumps is given; dumps (optional): every
    /// seam of the call appended.  false + err on a refusal (context full, a bad token id) or a CUDA/cuBLAS failure.
    /// decode: the call's rows are generated tokens (cache-aware routing applies to them; ModelOptions::car_rank).
    /// car_tail: on a prompt call with ModelOptions::car_prompt_rank, its last car_tail rows are the prompt CAR batch.
    bool forward(const std::vector<int>& tokens, std::vector<float>* logits, Dumps* dumps, std::string& err,
                 bool decode = false, int car_tail = 0);
    const Geo& geo() const;
    int n_past() const;
    /// The placement load() used, in a few lines (per card: layers, dense weights, expert slots, VRAM and RAM bytes).
    const std::string& placement_summary() const;
    /// Routed expert runs so far: from VRAM slots, from RAM by the device (zero copy), by the CPU lane.
    std::string expert_stats() const;
    /// With ModelOptions::page_stats: the paged experts' misses and major faults, the last call's and the totals since
    /// load (or page_reset).
    struct PageCount {
        long long calls = 0, tokens = 0;     // forward() calls counted, and their tokens
        long long used = 0, missed = 0;      // paged expert matrices a call used, and those with a page not resident
        long long miss_bytes = 0;            // their non-resident bytes (whole pages)
        long long majflt = 0;                // the process's major faults during the calls
        long long pf_ranges = 0, pf_bytes = 0;   // prefetch: ranges queued and their bytes (the real picks)
        long long pg_ranges = 0, pg_bytes = 0;   // pregate: predicted ranges queued and their bytes
        long long pg_picks = 0, pg_hits = 0;     // pregate: the real picks of predicted layers, and those predicted
        long long car_rows = 0;                  // car: decode token-layers routed under it
        long long car_missed = 0, car_fetched = 0, car_skipped = 0;   // their picks not resident; read (rank < r); skipped
        long long car_checked = 0;               // residency checks of experts beyond the top k (the re-routes)
        long long cp_rows = 0, cp_missed = 0;    // prompt CAR: batch token-layers; their picks not resident (nor read
                                                 // by the call's earlier rows)
        long long cp_read = 0, cp_full = 0;      // distinct experts a layer's batch reads (rank < R); a full read's
        long long cp_skipped = 0;                // batch picks replaced (routed among the available experts)
        long long wall_us = 0;                   // the calls' wall time
        long long utime_us = 0, stime_us = 0;    // the process's user and kernel CPU time during the calls
        long long tier_hits = 0, tier_misses = 0;    // RAM tier: used paged experts found in a slot / read into one
        long long tier_evicts = 0, tier_fallback = 0;   // slots taken from another expert; experts left in the mapping
        long long tier_wait_us = 0;              // time the calls blocked on the tier's reads
        long long dio_reads = 0, dio_bytes = 0;  // the tier's reads completed during the calls (O_DIRECT)
        long long dio_fails = 0;                 // of them, reads that came short (their experts fell back to the mapping)
        long long remote_calls = 0, remote_picks = 0;   // remote experts: layer requests sent, picks in them
        long long remote_tx = 0, remote_rx = 0;  // their bytes sent and received
        long long remote_wait_us = 0;            // time blocked on the replies after the local experts ran
        long long dyn_hits = 0;                  // dynamic VRAM tier: used experts that ran from a pool slot
        long long dyn_up = 0, dyn_down = 0;      // uploads started (RAM -> VRAM); victims copied back to a RAM slot
        long long dyn_drop = 0;                  // victims left to the mapping (no RAM slot free)
        long long dyn_up_bytes = 0;              // the uploads' bytes
    };
    PageCount page_last() const;
    /// --expert-server: answers remote-expert requests (k31_remote.hpp) on TCP, one connection at a time, until
    /// killed.  Computes on the device, or on the CPU lane when the model was loaded with one (quantized experts).
    int serve(const std::string& where, std::string& err);   // where: PORT (on 127.0.0.1) or HOST:PORT
    PageCount page_total() const;
    void page_reset();
    /// With ModelOptions::profile: wall time by part since load (or profile_reset).
    std::string profile() const;
    void profile_reset();

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace k31
