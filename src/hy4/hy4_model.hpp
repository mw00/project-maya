// src/hy4/hy4_model.hpp - maya-hy4: Tencent's Hy4-flash (a GGUF set, tools/hy4/pack_gguf.py) on the GPUs.
//
// The trunk's 42 layers split over the GPUs by their memory, each holding its layers' dense matrices, routed experts
// (in VRAM, an expert's gate | up | down a blob) and caches: the indexed layers' latent cache (F16, a row a position:
// the normed kv_cmpr, then the rotated k_pe), their indexer's pooled keys (F32, a row a group of kpool positions) and
// a ring of each position's pooling value | gate; the sliding-window layers' latent rows in a ring.  The Engram table
// stays in the model's file (read in place, the page cache keeping what it can), its rows gathered on the host.
//
// One forward runs a chunk of tokens through every layer: up to k::kMaxWin of them on the token path (GEMVs over int8
// inputs, ds4_iq's token / window experts - a decode step is a chunk of one), more on the prompt path (cuBLAS GEMMs
// over F16 weight bands, ds4_iq's expert sets); each token sees what the GGUF's reference graph (hy_v4_flash) gives it.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "hy4_vision.hpp"

namespace hy4 {

struct Options {
    std::string pack;
    std::vector<int> gpus;        // empty: every visible GPU
    int64_t max_ctx = 32768;
    int ring_slack = 16384;       // positions a sliding-window ring keeps beyond its window (how far a rewind may go
                                  // without starting over)
    double reserve_gb = 0.75;     // VRAM each GPU keeps free beyond the plan
    int prefill_chunk = 0;        // a prompt's tokens a forward takes at once (more than k::kMaxWin: the GEMMs); 0: auto -
                                  // 4096, 2048 or 1024, the largest the VRAM holds beside the model and the pictures
    double vram_experts = 1.0;    // the share of each layer's routed experts kept in VRAM at most (the plan takes less
                                  // when they do not fit; the rest run on the CPU lane)
    int cpu_threads = -1;         // the CPU lane's workers besides the caller (-1: one a physical core but the first)
    int tp = -1;                  // tensor parallel over tp GPUs of one kind (0 / 1: the layer split; -1: auto -
                                  // the largest group of one kind when it holds everything; hy4_tp.hpp)
    int main_gpu = -1;            // the head's and the drafts' GPU, an index into gpus (-1: the fastest, measured)
    std::string profile;          // an expert profile (uint32 [layers][experts] routings): the most routed in VRAM
    std::string profile_out;      // where save_profile writes it (the read one's counts with this run's)
    bool mtp = false;             // load the MTP block (the GGUF's nextn layer, its BF16 rounded to Q8_0) for drafts
    bool dfly2 = false;           // load the DFly2 drafter (the pack's model.dfly_layers.* tensors) for drafts: a few
                                  // dense layers over the trunk's own caches, several tokens a pass (not with mtp)
    std::vector<int> layer_split; // the first layer of each GPU after the first (empty: fill each GPU in turn)
    int kv = -1;                  // the latent cache's rows: 0 F16, 1 8-bit (the latent int8 in blocks of 32 with F16
                                  // scales, the rotated part F16; the indexer's rows F16: about 58% of the VRAM), -1
                                  // auto - F16 where the context fits in VRAM with it, 8-bit rows where it fits
                                  // only with them (any layout: one GPU, the layer split, tensor parallel)
    int kv_streaming = -1;        // KV streaming: 0 off, 1 on, -1 auto (the default) - on where the context does not
                                  // fit in the VRAM without it (after 8-bit rows and no picture's room: a start that
                                  // failed before), or, with HY4_KV_MORE=1 (opt-in), where it gives every layer more
                                  // routed experts in VRAM (the cache fits, the experts do not - nothing read from the
                                  // host until a conversation passes the ring), and the RAM holds the rows;
                                  // kv_resident then the ring tried first
    int64_t kv_resident = 0;      // KV streaming: the indexed layers' cache rows kept in VRAM - a ring of the newest
                                  // positions (a chunk's at least); every position's row in pinned host memory, the
                                  // picked rows outside the ring read from there (the same bits: the same answers).
                                  // 0: off - every row in VRAM.  Off as well where the context is no larger
                                  // than the ring.
    int kv_spill = -1;            // KV spill (with KV streaming): 0 off, 1 on, -1 auto (the default) - where the RAM does not
                                  // hold every position's row beside its headroom: the newest positions' rows in RAM,
                                  // the others in a file on a drive (kv_spill_dir; the pack's directory by default),
                                  // read for the positions a token picks only - the same bits, the same answers.  Auto
                                  // only where nothing else starts this context, and not on a drive that reads slower
                                  // than HY4_KV_SPILL_MIN_MBS (default 10 MB/s in 4 KB reads one at a time, measured once at the start)
    std::string kv_spill_dir;
    bool vision = true;           // the vision tower (the pack's mmproj file) on the main GPU: a picture's room is lent by
                                  // the prompt buffers while it is read (made longer by what they lack of it, if anything)
    bool vision_auto = true;      // (not asked for by name: a pack without one runs text only, no error)
    bool room_report = false;     // --room-report: say how much less free VRAM each GPU could have had with this start's
                                  // choices unchanged (Model::room_report; the plan only - the start is the same)
    std::string gpu_fetch;        // --gpu-fetch off|auto|gpu|lane|K (empty: HY4_FETCH, else off) - see Impl::Fetch
    bool trial = false;           // a drafting trial's load (--drafting auto / on, hy4_main.cpp): the plan, the weights
                                  // and the caches as a start with this drafter has them - no warm-up, no measuring,
                                  // no 4-bit copies, no GPU fetch, nothing written (memory files, profile, room report)
    bool verbose = false;
};

/// --drafting auto / on: the drafting a PC measured once is remembered by draft_file() in the line draft_key() names -
/// this PC (CPU, RAM, the lane's threads and kernels), its GPUs and driver, the model's pack and the start's options
/// that shape the plan (context, chunk, experts, VRAM kept free, layout, cache, pictures, HY4_HOT).  The file:
/// HY4_DRAFT_FILE, else beside HY4_Q4X_FILE (the server's data folder), else beside --profile; empty: none (then not
/// measured: the pack's drafter is taken, the MTP block first)
std::string draft_key(const Options& o);
std::string draft_file(const Options& o);
/// the GPUs a start with these options runs on (--gpus, else every visible one)
int gpus_of(const Options& o);

struct Stats {
    double load_s = 0;
    double vram_gb = 0;           // all the model's VRAM
    double engram_ms = 0;         // the host's Engram gathers
    int64_t tokens = 0;
    int experts_vram = 0, experts_all = 0;   // the routed experts in VRAM, all of them (the trunk's)
    int64_t experts = 0, cpu_experts = 0;    // routed experts run, and how many on the CPU lane
    double cpu_ms = 0;                       // the CPU lane's time
    int64_t streamed = 0;                    // experts a prompt's chunks copied to the GPUs (not in VRAM)
    double stream_ms = 0;                    // ... their copies into the pinned bounce
};

class Model {
public:
    Model();
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    bool load(const Options& o, std::string& err);
    /// ids at positions pos0 .. pos0 + n - 1 (the caches hold the positions before pos0: resumable(pos0)); logits_out
    /// (if given) gets the logits of the last `want` positions, [want][vocab] F32
    bool forward(const int* ids, int n, int64_t pos0, int want, float* logits_out, std::string& err);
    /// The MTP block's drafts (Options::mtp): up to n tokens after `tok` at position pos (the caches hold the positions
    /// before it), each MTP step's top token and its probability, stopping after one below `floor`.  A forward keeps
    /// the block's caches in step with the trunk's (every forward runs it over its tokens), so the drafts after an
    /// accepted prefix need nothing else.  has_mtp(): the block is loaded.
    /// (base >= 0: the drafts after the first `base` only while every draft so far has a probability of at least `sure`)
    bool draft(int tok, int64_t pos, int n, float floor, std::vector<int>& drafts, std::vector<float>& conf,
               std::string& err, int base = -1, float sure = 0.0f);
    bool has_mtp() const;
    /// The DFly2 drafter is loaded (Options::dfly2): draft() is then its pass - up to n tokens after `tok` from one
    /// pass over the trunk's caches (nothing to keep in step: a forward leaves it what it needs).  has_draft(): either
    /// drafter; draft_default(): the drafts a round the loaded one is made for (3: the MTP block; the drafter's own
    /// count), draft_max() the most a round takes.
    bool has_dfly2() const;
    bool has_draft() const { return has_mtp() || has_dfly2(); }
    /// The drafters the pack holds, loaded or not: 1 the MTP block, 2 the DFly2 drafter (bits); gpu_count(): the GPUs
    /// this start runs on
    int pack_drafters() const;
    int gpu_count() const;
    int draft_default() const;
    int draft_max() const;
    /// The routings counted (with the profile read at the load) to Options::profile_out; no file named: nothing.
    /// HY4_HOT, in a start that learns (its file missing, or HY4_HOT_LEARN=1): its experts' scores to its file as well
    bool save_profile(std::string& err) const;
    /// An answer's end: HY4_HOT counts the routes of its tokens fed before position `end` (the prompt and the answer's
    /// tokens but its last - the rows drafting and a plain decode both read; a token window's rows count once kept),
    /// so the next placement is the same with drafting on or off; -1: every row fed.  Before save_profile and idle()
    void answered(int64_t end);
    /// Between requests: HY4_HOT moves the experts in VRAM to what the scores call for (the answers' routing) - the
    /// only place it moves them, so an answer runs on one placement; nothing where it is off
    void idle();
    /// Page-lock a host buffer the logits go to (the copies then run at the link's speed, not through the driver's
    /// staging); unpin it before it is freed.  false: left as it was
    static bool pin(void* p, size_t bytes);
    static void unpin(void* p);
    /// Whether a forward may start at pos (the sliding-window rings still hold the window before it)
    bool resumable(int64_t pos) const;
    /// The positions processed so far (the next forward's pos0 when continuing)
    int64_t end() const;

    int vocab() const;
    /// Row r of the last forward's logits: its largest value's index (of equals the lowest - the host's argmax), found
    /// by the GPUs that computed the row's parts (tensor parallel with the head over the members); -1: not known, the
    /// caller's own walk over the row
    int row_top(int r) const;
    int64_t max_ctx() const;
    /// the most tokens a forward reads at once (the prompt path's chunk): a prompt in calls of this many reports its
    /// progress between them
    int chunk() const;
    /// the GPUs a layer is split over (tensor parallel: 2 or more; 1: the layer split or one GPU)
    int tensor_parallel() const;
    int eos() const;
    const Stats& stats() const;
    std::vector<std::string> info_lines() const;
    /// --room-report's measurement, the server's INFO keys (empty: not asked, or the start did not fit):
    /// "room_rows=f16|q8 room_drafter=0|1 room_streamed=0|1 room_gpus=<PCI bus id>,<free>,<slack>;..." - each GPU's
    /// free VRAM as the plan read it at the start (bytes) and how much less of it the start's choices hold with: the
    /// latent cache's rows, KV streaming, the drafter's room, the pictures, the GPUs' layout, that it starts at all
    std::string room_report() const;

    /// Pictures (Options::vision): the tower's geometry (nullptr: none loaded); a picture's rows, F32 [tokens][n_embd]
    /// (VisionGeo::tokens of them); and the request's pictures - each one's image tokens, the positions start ..
    /// start + rows - 1, take its rows instead of their embedding (the MTP block's too) and have no Engram lookups.
    /// Kept until set_images({}) at the request's end (the drafts' first step may read rows of the prompt).
    struct Image {
        int64_t start = 0;
        std::vector<float> rows;
    };
    const VisionGeo* vision() const;
    /// Where a picture's room comes from: "chunk" - lent by the head GPU's prompt buffers (idle while a picture is
    /// read; no VRAM kept for pictures), "free" - kept free between pictures (HY4_VISION_LEND=0);
    /// nullptr: no tower.  Its bytes for the largest picture this start reads; the pictures read so far each way.
    const char* vision_room() const;
    size_t vision_room_bytes() const;
    void vision_reads(long long& lent, long long& own) const;
    bool encode_image(const uint8_t* rgb, int width, int height, std::vector<float>& rows, std::string& err);
    void set_images(std::vector<Image> images);

private:
    struct Impl;
    std::unique_ptr<Impl> m;
};

}  // namespace hy4
