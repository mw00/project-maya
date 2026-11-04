// src/mm3/mm3_model.hpp - maya-mm3: MiniMax-M3.1-Flash (EXL3) on the GPUs.
//
// The layers split over the GPUs by their memory, each GPU holding its layers' dense matrices, KV cache (BF16
// [ctx][4][128] K and V) and index-key cache (BF16 [ctx][128]), then as many of its layers' routed experts as fit
// (in VRAM by expert, a blob each); the others are read from the model's files into a GPU's staging slots when a
// token routes to them.  One forward (the prompt path) runs any number of tokens at once, a decode step included.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "mm3_pack.hpp"

namespace mm3 {

struct VisionGeo;
class TpEngine;

struct Options {
    std::string pack;
    std::vector<int> gpus;        // empty: every visible GPU
    std::vector<int> layer_split; // the first layer of each GPU's share after the first (empty: by their free memory)
    int max_ctx = 32768;
    int chunk = 1024;             // prompt tokens a forward takes at most
    bool chunk_given = false;     // (--chunk: chosen - the prefill calibration's chunk is not taken)
    // the prompt path's dense GEMMs (mm3_gguf.cuh's set_gemm): -1 the prefill calibration's for this PC (else the
    // BLAS), 0 the BLAS, 1 hipBLASLt (AMD); and the calibration's cache (prefill_tune: its file - empty: the pack's
    // prefill-tune.txt; tune: read it at all)
    int prefill_gemm = -1;
    std::string prefill_tune;
    bool tune = true;
    int staging = 48;             // expert staging slots a GPU (each one expert's blob)
    double reserve_gb = -1;       // VRAM a GPU keeps free beyond the plan (< 0: 1.0, the tensor-parallel path 0.3 -
                                  // it allocates all it needs at the load)
    int resident_cap = -1;        // routed experts kept in VRAM a layer at most (-1: as many as fit)
    std::string profile;          // an expert profile (uint32 [layers][experts] routings): the most used stay in VRAM
    bool pin_pool = true;         // the experts not in VRAM assembled once in pinned host memory (a miss: one DMA)
    bool fast_decode = true;      // one token: the decode GEMVs and the token's fused experts (else the prompt path)
    int cpu_threads = -2;         // the CPU lane's workers besides the caller (-1: no CPU lane, misses are staged;
                                  // -2: this host's, CpuLane::auto_workers - one a physical core, the first the caller's,
                                  // 2 cores left free where there are no SMT siblings); --cpu-threads N: N - 1 of them
    bool dspark = false;          // load the DSpark draft (the pack's dspark/) on the head's GPU
    bool vision = true;           // load the vision tower (the pack's vision.*) on gpus[vision_gpu], before the experts
    bool vision_auto = true;      // (vision not asked for by name: a pack without a tower runs text only, no error)
    bool vision_must = false;     // (--images: a tower that does not load is an error; else vision off with a NOTE)
    int vision_gpu = 0;
    bool promote = true;          // the staging slots a cache: an expert the CPU lane keeps computing copied into one
                                  // (from the pinned pool, on a copy stream of its own) while the model runs
    int promote_lag = 2;          // a promoted expert's slot is its own this many forwards after its copy began, the
                                  // copy waited for then (1: the next forward - plain decode then waits ~5 ms a token)
    int tp = 0;                   // > 0: the tensor-parallel path (mm3_tp.hpp) in groups of tp GPUs (1, 2 or 4)
    double cache_gb = 1.5;        // (tp) each GPU's expert cache (promote's slots), taken from its experts in VRAM -
                                  // at least this, more where a layer's experts outside VRAM need more (tight memory)
    int tp_chunk = 1024;          // (tp) a prompt chunk's tokens: its experts not in VRAM are copied in once a chunk
    int tp_ffn_rows = 0;          // (tp) ... and its FFN half run in slices of this many tokens (0: the whole chunk) -
                                  // the same bits; 256 keeps ~37 more experts in VRAM, a prompt reads ~13% slower
    int kv_resident = 0;          // KV streaming (--kv-streaming on: 32768): the positions of each layer's K / V kept in
                                  // VRAM (a ring), all of them in pinned host memory (0, or the context or less: off)
    // the K / V caches' formats (--kv-cache, mm3_kernels.cuh's kKv*: 0 BF16 - the model's dtype, exact - 1 INT8, 2 FP8
    // E4M3 with a power-of-two scale a 32, 3 FP4 E2M1 with an E4M3 scale a 16) and whether their rows are rotated before
    // they are rounded (--kv-rot); the index keys stay BF16 (the blocks each query keeps do not move)
    int kv_k = 1, kv_v = 1;   // (INT8, rows rotated: tools/mm3/kv_gate.py decides whether FP4 joins it)
    bool kv_rot = true;
    // --parallel: conversations at once - a KV cache (and index-key cache) each, the context's room for every one of
    // them (more than 1: no KV streaming, no DSpark draft, the layer split)
    int seqs = 1;
    bool verbose = false;
};

struct Stats {
    double load_s = 0;
    long staged = 0;              // expert blobs copied into staging slots
    long resident = 0, experts = 0;
    double vram_gb = 0;           // the resident experts' bytes (every member's slice)
    long pooled = 0;              // experts in the pinned pool (0: no pool - the misses read from the mapped files)
    double pinned_gb = 0;         // the pinned pool of the experts not in VRAM
    double all_gb = 0;            // every routed expert's bytes (the Monitor: what neither VRAM nor the pool holds is on the SSD)
    int kv_ring = 0;              // KV streaming: the positions a layer's VRAM cache holds (0: all - off)
    double kv_host_gb = 0;        // ... and the host memory holding every position
    long profiled = 0;            // routings counted (save_profile writes them)
    long cpu_experts = 0;         // routed experts the CPU lane computed
    int cpu_threads = 0;          // the CPU lane's threads (0: no CPU lane)
    long promoted = 0;            // experts copied into a staging slot because they kept coming back
    double t_promo = 0;           // a forward's start waiting for those copies
    double t_sync = 0, t_lane = 0, t_token = 0;   // decode: waiting on the routing, the CPU lane, all (seconds)
};

/// A forward's logits rows as the sampler's candidates (Model::forward_cands): per row the kc largest logits and every
/// one equal to the kc-th ((id, logit) pairs by logit desc, then id asc) and the logits of the extra ids - all a
/// sampler with top_k <= kc - extra.size() needs of the row when its penalties touch the extra ids alone.
struct Cands {
    int kc = 1;                                            // in: the largest a row
    std::vector<int> extra;                                // in: ids every row reports too (the penalised ones)
    std::vector<std::vector<std::pair<int, float>>> top;   // out: [row]
    std::vector<std::vector<float>> ex;                    // out: [row][extra]
};

/// A sampled DSpark proposal's distribution (Model::draft_sampled): the draft's biased logits through these, as the
/// request samples the target.
struct DraftSampling {
    float temperature = 1.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 64;
};

class Model {
public:
    Model();
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    bool load(const Options& o, std::string& err);
    /// Runs ids at positions pos0 .. pos0 + n - 1 (the cache holds every position before pos0), in chunks of at most
    /// Options::chunk; logits_out (if given) gets the logits of the last `want` positions of this call, [want][vocab]
    /// F32, row-major.
    bool forward(const int* ids, int n, int pos0, int want, float* logits_out, std::string& err);
    /// The same with the last `want` rows as candidates instead of logits (the tensor-parallel path picks them on the
    /// head's GPU: a few KB a row back instead of the vocabulary's 800).
    bool forward_cands(const int* ids, int n, int pos0, int want, Cands& c, std::string& err);
    /// --parallel (Options::seqs): the sequences, and the one forward / forward_cands read and write (0 at the load).
    int seqs() const;
    void bind(int s);
    /// One token of each of R conversations together (R <= 16, distinct sequences): ids[r] at position pos[r] of
    /// sequence seq[r], whose cache holds its positions before it; logits_out [R][vocab].  Each row is the same bits as
    /// that token's forward alone: the dense matrices on the token kernels' several columns, the routed experts on the
    /// verify window's (each distinct expert read once for its rows), each row's attention on the token path's calls
    /// over its own cache.
    bool forward_rows(const int* ids, const int* seq, const int* pos, int R, float* logits_out, std::string& err);
    /// --parallel on a layer split: the GPUs as a pipeline.  pipe_stages(): its GPUs (1: none - one GPU, the
    /// tensor-parallel path, no fast decode, one sequence).  pipe_tick: a group of R rows (forward_rows' rows, R 0: none)
    /// enters the first GPU - free after every tick - and every group in flight runs its GPU's layers, then moves one
    /// GPU on; a group past the last GPU: its logits [R][vocab] for pipe_take (oldest first, with its tag).  Each row
    /// the same bits as forward_rows'.  A sequence is in one group at most; forward / forward_rows only with none in
    /// flight (pipe_busy() 0); pipe_drop() forgets them (after an error).
    int pipe_stages() const;
    int pipe_busy() const;
    bool pipe_tick(const int* ids, const int* seq, const int* pos, int R, int tag, std::string& err);
    bool pipe_take(int& tag, std::vector<float>& logits);
    void pipe_drop();
    /// (the tensor-parallel path, whose prompt chunks go through its GPU groups as a pipeline: a whole prompt is one
    /// call) f(tokens of the call done) after each chunk, on the calling thread; false: no further chunk starts - the
    /// call returns true with done() of its tokens in the cache (the logits only when all are).  set_progress({}) ends
    /// it.  pipelined(): whether a prompt in one call gains from it (else calls of Options::chunk are as good).
    void set_progress(std::function<bool(int)> f);
    int done() const;
    /// (the tensor-parallel path) ids as one prompt from position 0, the next-token NLL of every position kept
    /// (ppl_nll()[p] = -log p(ids[p + 1])) - its chunks one at a time, the head on each one's rows.
    bool forward_ppl(const int* ids, int n, std::string& err);
    const std::vector<double>& ppl_nll() const;
    bool pipelined() const;
    int vocab() const;
    int max_ctx() const;
    int bos() const;
    int eos() const;
    const Stats& stats() const;
    std::vector<std::string> info_lines() const;
    /// The Monitor's map of each GPU (serve/gpu_map.py's VRAM lines): what each part holds there, measured at the load.
    std::vector<std::string> vram_lines() const;
    /// The prefill calibration (maya-mm3 --calibrate-prefill): the prompt chunk a forward takes (at most the chunk the
    /// load sized its buffers for: chunk_max) and the prompt path's GEMM backend, changed between forwards.
    int chunk() const;
    int chunk_max() const;
    void set_chunk(int c);
    void set_prefill_gemm(int backend);
    int prefill_gemm() const;
    bool lt_available() const;
    /// The calibration's key of this PC (its GPUs, their architecture, the GPU runtime and driver) and the file its
    /// results are kept in (one line a PC; the load takes the line of its own PC); write_tune keeps `settings` for it.
    std::string tune_key() const;
    std::string tune_file() const;
    bool write_tune(const std::string& settings, std::string& err) const;
    /// The routings counted so far (with the loaded profile's, if any) as a profile file.
    bool save_profile(const std::string& path, std::string& err) const;
    /// The DSpark draft (Options::dspark): its block (the most tokens a proposal has; 0 without the draft), and up to
    /// n tokens proposed after `anchor` at position pos (the cache holds the positions before it) with their
    /// confidences.
    int draft_block() const;
    bool draft(int anchor, int pos, int n, std::vector<int>& drafts, std::vector<float>& conf, std::string& err);
    /// ... each draft drawn from the draft's distribution (ds; u[i] the draw of position i, uniform in [0, 1)) instead
    /// of its argmax, q[i] that distribution ((id, probability), the most likely first) - the target keeps a draft d
    /// with min(1, p(d) / q(d)) and else draws from max(0, p - q), which leaves every token the target's own sample.
    bool draft_sampled(int anchor, int pos, int n, const DraftSampling& ds, const std::vector<float>& u,
                       std::vector<int>& drafts, std::vector<float>& conf,
                       std::vector<std::vector<std::pair<int, float>>>& q, std::string& err);
    /// The vision tower (Options::vision): its geometry (null without it), a picture's rows (rgb height x width x 3,
    /// sides multiples of 28 -> F32 [tokens][hidden]), and the pictures the next forwards' image tokens take their
    /// rows from (position start .. start + rows / hidden - 1; set_images({}) after the prompt).
    struct Image {
        int start = 0;
        std::vector<float> rows;
    };
    const VisionGeo* vision() const;
    bool encode_image(const uint8_t* rgb, int width, int height, std::vector<float>& rows, std::string& err);
    void set_images(std::vector<Image> images);

private:
    struct Impl;
    Impl* p;
    TpEngine* tpe = nullptr;   // (Options::tp > 0: every call goes to it)
};

}  // namespace mm3
