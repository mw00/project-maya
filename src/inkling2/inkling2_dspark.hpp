// src/inkling2/inkling2_dspark.hpp - maya-inkling2's drafter: Inkling 2 Small's DSpark block (Thinking Machines'
// mtp.safetensors as tools/inkling2/convert_dspark.py's Q8_0 GGUF and its index) on one GPU
// (docs/INKLING2_DSPARK.md: the equations, the measured acceptance).
//
// The drafter is fed the target's taps of every position the context keeps (feed: 8 vectors of n_embd a position -
// the residual stream entering layers 5, 11, 17, 23, 29, 35, 41, i.e. the engine's "l_out" of layers 4, 10, ..., 40,
// and the head's input after its mix and norm, "result_norm") and drafts the tokens after the next one (draft).  Its
// state is position-keyed K / V rows and one context row a position, each a function of that position's taps alone:
// rollback(pos) only moves the end.
//
//   Dspark d;  d.load(o, err);
//   d.feed(taps, n, pos0, err);            // the prompt's rows, then each kept row of a decode step / verify window
//   d.draft(y, 7, toks, probs, err);       // y: the token at end(), sampled and not read yet -> the tokens after it
//   ... the engine verifies [y, toks], keeps m of them ...
//   d.feed(taps of rows 0 .. m, m + 1, pos, err);
//   d.rollback(pos);                       // a rewind: the context ends at pos again
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace inkling2 {

struct DsparkOptions {
    std::string index;         // the drafter's index folder (maya-inkling2-dspark-q8_0: dspark.txt, tensors.txt)
    std::string target_pack;   // the target's pack (maya-inkling2: token_embd, token_embd_norm, output)
    std::string joint_index;   // (optional) another index of the same drafter to take joint_proj.weight from (the
                               // BF16 one: Q8_0 rounds its small weights coarsely); empty: the index's own
    int gpu = 0;               // the device (the engine's: the GPU that holds the target's head)
    int64_t max_ctx = 32768;   // positions the global layer's cache holds (the engine's Options::max_ctx, its default)
    int rewind = 4096;         // rows the sliding-window rings and the context ring keep beyond the window
    float cum_min = 0.5f;      // draft() keeps a draft while the product of the drafts' top probabilities so far is
                               // at least this (0: every draft)
    // the target's output matrix when the caller already holds it on this GPU (ds4_iq's rows: Q6_K / Q8_0, ld bytes a
    // row, n_vocab rows of n_embd) - else the drafter uploads a copy of its own (760 MB)
    const uint8_t* output_dev = nullptr;
    int output_type = 0;
    size_t output_ld = 0;
    int attn_tile = 8;         // keys a tile of the block's attention (4 .. 64).  An R9700 at 300 / 2.7K / 16K
                               // positions: 8 keys 0.44 / 1.60 / 4.54 ms a block, 16 keys 0.57 / 2.25 / 6.02, 24 keys
                               // 0.77 / 3.13 / 8.08; the scratch is 50 KB a tile (max_ctx / attn_tile tiles)
    int attn_gemm_min = 1024;  // a layer's attention as BLAS GEMMs from this many keys on (0: always, < 0: never, and
                               // no BLAS handle).  An R9700: the GEMMs cost ~0.2-0.3 ms a layer at any length, the
                               // tiles 0.09 ms at 300 keys, 0.24 at 1,032 (a full window), 3.6 at 16K
    bool own_gemv = false;     // the block's Q8_0 GEMVs through this file's 8-column kernel instead of ds4_iq's:
                               // measured the same within 3% (FFN 3.87 against 4.01 ms, Q/K/V 0.98 against 0.94)
    bool profile = false;      // a synchronisation after each phase of draft(): Stats' split (slows a draft a little)
    bool verbose = false;
};

struct DsparkStats {
    double load_s = 0;
    double vram_gb = 0;                      // what the drafter allocated on its GPU
    int64_t fed = 0, drafts = 0;             // positions fed, draft() calls
    double feed_ms = 0, draft_ms = 0;        // host time in feed() / draft() (draft: to its tokens on the host)
    double embed_ms = 0, layers_ms = 0, base_ms = 0, head_ms = 0;   // (profile) draft()'s phases
    double qkv_ms = 0, attn_ms = 0, o_ms = 0, ffn_ms = 0;           // (profile) layers_ms' parts: Q / K / V, the
                                                                    // attention, its output matrix, the FFN
};

class Dspark {
public:
    static constexpr int kTaps = 8;     // taps a position
    static constexpr int kBlock = 8;    // a block's rows: the anchor and 7 drafts
    /// The VRAM a position of DsparkOptions::max_ctx costs (Inkling 2 Small's drafter: one global layer's K and V rows,
    /// F16 [8 kv heads][128], and the GEMM attention's scores, F32 and F16 a (block row, head) pair: 8 x 48)
    static constexpr size_t ctx_bytes() { return 2 * 8 * 128 * 2 + (size_t) kBlock * 48 * (4 + 2); }

    Dspark();
    ~Dspark();
    Dspark(const Dspark&) = delete;
    Dspark& operator=(const Dspark&) = delete;

    bool load(const DsparkOptions& o, std::string& err);

    /// The taps of n consecutive positions pos0 .. pos0 + n - 1 (pos0 must be end()): taps [n][kTaps][n_embd] F32 on
    /// the host, tap k of a position in target_layers()' order.  Their K / V rows go into every layer's cache and the
    /// markov head's context row is kept; end() is then pos0 + n.
    bool feed(const float* taps, int n, int64_t pos0, std::string& err);
    /// The same from the device (this GPU), rows ld floats apart.  It queues the reads on the drafter's own stream and
    /// returns: read_done (a cudaEvent_t / hipEvent_t of this GPU, or null) is recorded there after the last of them,
    /// so the caller's stream waits on it before it writes d_taps again
    bool feed_dev(const float* d_taps, size_t ld, int n, int64_t pos0, std::string& err, void* read_done = nullptr);

    /// A block after `anchor`, the token at position end() (sampled, not read by the target yet): up to max_n (<= 7)
    /// drafts for positions end() + 1 .., each the head's argmax (ties: the lowest id) with its probability under the
    /// head's distribution (the confidence stand-in: close to the measured acceptance).  The drafts are cut where the
    /// running product of the probabilities falls below cum_min (toks may come back empty).  Needs end() >= 1 and the
    /// context row of end() - 1 still in its ring (else no drafts, true).  The state does not change.
    bool draft(int anchor, int max_n, std::vector<int>& toks, std::vector<float>& probs, std::string& err);

    /// The context ends at pos (<= end()): the rows past it are dead and written again when those positions are fed
    void rollback(int64_t pos);
    void reset();
    int64_t end() const;

    void set_cum_min(float c);
    float cum_min() const;
    const std::vector<int>& target_layers() const;   // 5, 11, 17, 23, 29, 35, 41, 48
    int n_embd() const;
    int vocab() const;   // the unpadded vocabulary
    const DsparkStats& stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m;
};

}  // namespace inkling2
