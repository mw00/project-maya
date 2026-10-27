// src/ds4/ds4_model.hpp - DeepSeek-V4.1-Flash on Maya: an EXL3 model (tools/ds4/pack.py's pack beside its files) over
// the machine's GPUs, layer-split, with the routed experts that do not fit in VRAM on the CPU lane.
//
// Windows of up to 16 tokens through every layer (a prompt in windows, a decode step a window of one): each query of a
// window sees exactly what the reference's decode path gives it (DeepSeek's inference/model.py at start_pos > 0) -
// its own 128-token window, the compressed groups completed by its position.  The state a token leaves is the
// 128-token window ring of every layer, the compressors' partial groups, the compressed KV and index keys of the KV
// source layers (append-only), and the Engram n-gram history; checkpoint() / restore() keep the first two (the rest is
// truncated by position) so a conversation can go back to the end of its last prompt.  A conversation that leaves the
// model altogether (another chat answered in between) is a slot: slot_save / slot_load take the same state - the
// compressed KV and index keys included, which a checkpoint leaves in place - through a file or a host buffer.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "ds4_prefill_gaps.hpp"

namespace strata::ds4 {

struct Options {
    std::string pack;                 // <model>/maya-ds4
    int64_t max_context = 32768;
    std::vector<int> gpus;            // empty: every visible GPU
    int64_t reserve_mib = 600;        // VRAM left free on each GPU
    double vram_experts = 1.0;        // the share of the expert budget to use (tests: less, so the CPU lane works)
    // the CPU lane's threads: 0 a thread a physical core, or where the host offers it the performance cores or one
    // NUMA node's cores when measured faster at load; n: at most n, a thread a core
    int cpu_threads = 0;
    std::string profile;              // expert hit counts (read to place the hottest experts in VRAM, written back)
    bool expert_cache = true;         // experts move between VRAM and the CPU lane as they are used
    // the CPU lane's experts read in place from the model's files, the page cache keeping what fits, instead of
    // copied into RAM: 1 on, 0 off, -1 (auto) where the RAM this process can get does not hold them
    int mmap_experts = -1;
    bool dspark = true;              // load the DSpark drafter (the checkpoint's mtp.* blocks) for speculative decoding
    float dspark_conf = 0.5f;         // a draft is verified while its confidence (sigmoid) is at least this
    // prompt lookup (ds4_main's --suffix-draft): its verify windows run without the drafter too - their buffers (the
    // rows and states a rejected draft undoes) are allocated with the drafter, or with this
    bool lookup = false;
    // tensor parallelism: runs of tp GPUs of one kind (the same memory) take the same layers, each a share of the
    // heads, of the output projection and of every expert's FFN, summed through host-mapped memory (a GPU unlike the
    // next stays alone); tp > 1 runs with threads (--tp 1 --threads off: one host thread, layer split only)
    int tp = 2;
    bool threads = false;             // a host thread per GPU enqueues its layers (the hand-offs host-mapped)
    // conversations at once (--parallel, 1-8): N conversation states, one bound at a time (seq_bind), their decode
    // windows pipelined over the split's stages (multi_issue / multi_take) so a card works on one conversation while
    // the cards after it work on another.  1: the engine as before, one conversation (nothing of this allocates);
    // the first state for the whole context, the others for STRATA_DS4_SEQ_CTX tokens (the context, default).  A split
    // with threads (a lane's state and its hand-offs are per conversation, so the serial path has none), and no
    // tensor parallel: load() refuses both, and N > 1 decodes without the DSpark drafter (the log says so)
    int parallel = 1;
    // prompts in chunks of this many tokens through each layer at once (threads): the GEMMs on the tensor cores, the
    // CPU lane's experts streamed to the GPUs a chunk at a time instead of computed on the host; 0: windows of 16;
    // -1: 2048 on one GPU, 4096 on several, or 8192 with the experts read in place (every chunk streams them all from
    // the files)
    int prefill_chunk = -1;
    bool vision = true;               // the vision tower (the checkpoint's vision.* / aligner.*), when it has one
    bool vision_only = false;         // (tests) load the vision tower alone: encode_image() and nothing else
    bool verbose = false;
    // abliteration (ds4_ablate.hpp): a refusal direction projected out of every residual write.  The direction file
    // (a llama-cvector-generator GGUF, or a rank-1 abliteration adapter GGUF), which of its directions (1-based; 0: the
    // adapter's own), the scale (0: the adapter's own, else 1)
    std::string ablate;
    int ablate_layer = 0;
    float ablate_scale = 0.0f;
};

/// A picture in the prompt: its span's rows (every position of the span carries the image token), BF16 [span][dim]
struct Image {
    int64_t start = 0;
    std::shared_ptr<const std::vector<uint16_t>> rows;
};

/// A picture as a slot keeps it: where its span starts, the fingerprint of its pixels, the span's length in tokens.
/// No pixels, no rows - the rows' work is in the KV (they are read only while the span itself is fed, and a restored
/// state with the same picture at the same place has that KV).  Two states whose pictures agree on all three are the
/// same state; another picture, another place or another span: not.
struct PicMark {
    int64_t start = 0;
    uint64_t hash = 0;
    uint64_t len = 0;
};

struct Stats {
    int64_t tokens = 0;
    int64_t experts = 0, cpu_experts = 0;   // routed experts run, and how many of them on the CPU lane
    int64_t draft_experts = 0, draft_cpu = 0;   // ... of them the DSpark blocks', and theirs on the CPU lane
    int64_t stream_experts = 0;             // ... and how many of the others streamed to the GPU (the PCIe lane)
    int64_t helper_experts = 0;             // ... and how many of the others the helper GPU computed (DS4_HELPER)
    int64_t helper_fallback = 0;            // its picks the CPU lane computed instead (after a helper failure)
    int helper_held = 0, helper_dead = 0;   // experts in its pool; 1: it failed (or answered late), the CPU lane took over
    double helper_ms = 0.0;                 // the service thread's waits for the helper's answers, after its own part
    int64_t hr_hits = 0;                    // DS4_HELPER_RECENT: picks its recent slots served; fills mapped, slots'
    int64_t hr_maps = 0, hr_evicts = 0;     //   experts let go, fills that failed, fills left unmapped (VRAM took the
    int64_t hr_fails = 0, hr_abandoned = 0; //   expert meanwhile)
    double cpu_ms = 0.0, gpu_wait_ms = 0.0;   // the CPU lane's time; the host's waits for the routing
    double layer_ms = 0.0, head_ms = 0.0, engram_ms = 0.0;   // the layers' host time in all; the head's wait; Engram reads
    int64_t eg_rows = 0, eg_hits = 0, eg_reads = 0;   // consumer Engram lookups: cache hits, on-demand reads
    int64_t eg_pf_rows = 0, eg_pf_hits = 0, eg_pf_reads = 0;   // the same accounting for the prompt prefetcher
    double eg_io_ms = 0.0;                            // the windows' Engram reads, issue to last done (mostly hidden)
    double eg_pf_io_ms = 0.0, eg_pf_wait_ms = 0.0;    // prefetch read span and worker wait, cumulative
    int64_t eg_cached = 0, eg_cache_cap = 0;          // the row cache's rows held, its capacity
    int64_t eg_cache_row_bytes = 0;                   // each cached row's payload bytes
    int64_t swaps = 0;                // experts the cache moved into VRAM
    int64_t skipped = 0;              // routed experts STRATA_DS4_DISK_SKIP_W left out (light, not in RAM)
    int skip_on = -1;                 // DISK_SKIP_AUTO: 1 skipping, 0 off (warm), -1 not in use
    int64_t skip_switches = 0;        // ... the times it switched
    double skip_cold = 0.0;           // ... the cold share over its window (while skipping)
    int64_t ram_experts = 0, ram_moves = 0;   // CPU-lane experts read from the RAM tier; experts moved into it
    double prof_ms[6] = {};          // DS4_PROF: the GPU's time in decode by stage (between layers, attention, routing +
    int64_t prof_tokens = 0;         // shared expert, experts' gate / up, the wait for the CPU lane, down + post)
    // lookahead (decode windows of one): the picks of layers guessed a layer before (the next layer's gate on this
    // layer's FFN input), of them in the guess's top k / in all its guesses / in the guess two layers before; the picks
    // read from the files, of them prefetched early; experts prefetched early
    int64_t ahead_picks = 0, ahead_hit_k = 0, ahead_hit = 0, ahead_hit2 = 0;
    int64_t ahead_cold = 0, ahead_cold_hit = 0, ahead_pf = 0;
    int64_t ahead_rank_n[16] = {}, ahead_rank_hit[16] = {};   // the next-layer guesses by rank, and how many were picked
    int64_t drafts_offered = 0, drafts_accepted = 0, drafts_rounds = 0;
    // per draft position (the 1st .. 8th of a round): verified with all the drafts before it accepted, and accepted
    int64_t draft_reached[8] = {}, draft_kept[8] = {};
    double draft_ms = 0.0, verify_ms = 0.0;   // the drafter's time, the verification windows' time
};

class Model {
public:
    Model();
    ~Model();
    bool load(const Options& o, std::string& err);

    /// Feeds `token` at position pos(); with logits, the next token's logits are left on the head's device.
    bool step(int token, bool logits, std::string& err);
    /// Feeds n tokens (windows of up to 16, or a long run in chunks of prefill_chunk); the last `logits_rows` of them get
    /// logits (logits_rows > 16: only a run of one chunk or window).
    bool feed(const int* tokens, int n, int logits_rows, std::string& err);
    /// feed() of a prompt's end that also keeps the state before its last token in checkpoint `slot` (a prompt's
    /// last token then rides along in the run of chunks instead of a window of its own); the same as
    /// feed(n - 1), checkpoint(slot), feed(1)
    bool feed_mark(const int* tokens, int n, int logits_rows, int slot, std::string& err);
    /// n tokens through windows of 16, with the bound conversation's own lane buffers (`lane < 0`: the one
    /// conversation's shared ones).  Used while a prompt's read is paused at a chunk boundary: a short prompt
    /// admitted in the gap is read in its own lane, so it does not touch the read's staging slots or hand-offs.
    bool feed_windows(int lane, const int* tokens, int n, int logits_rows, std::string& err);
    /// The chunked prompt path's chunk (0: windows), for the multi loop's short-prompt read in a gap.
    int prompt_chunk() const;
    /// A prompt window's tokens, and what a full one took here on average (s; 0: none ran yet) - what reading a prompt
    /// in windows costs, measured (the multi loop's gap reads a short new prompt in windows only when that is quicker
    /// than its turn in chunks).
    int window_rows() const;
    /// The split's pipeline stages (its groups of GPUs, one after another): a read of that many chunks keeps them all
    /// busy - the multi loop's read pieces (MAYA_DS4_READ_ORDER's default) are that many chunks.
    int pipeline_stages() const;
    double window_s() const;
    /// The decode gaps of a prompt read (MAYA_DS4_PREFILL_GAPS), and the gap the serve loop runs at a boundary:
    /// `wanted` says whether this boundary is due, `gap` runs it (0 go on, 1 the read was stopped, 2 failed).
    PrefillGaps prefill_gaps() const;
    void set_prefill_gap(std::function<int()> f);
    void set_prefill_gap_wanted(std::function<bool()> f);
    /// The next token from the last feed's last logit row (greedy when temperature <= 0).
    int sample(float temperature, float top_p, int top_k, float min_p, uint64_t seed, uint64_t counter);
    /// The request's repetition / frequency / presence penalties (llama.cpp's) for every pick after this - sample() and
    /// verify() alike: over the last `last_n` tokens the state has consumed (at most 4096; a verify window's row counts
    /// the drafts before it).  last_n 0 (or every penalty neutral): none, the picks exactly as without.
    void set_penalties(int last_n, float repeat, float freq, float present);
    /// The last feed's logit rows on the host (rows x vocab floats).
    bool logits(std::vector<float>& out);

    // ---- several conversations at once (options.parallel, N > 1): N states, one bound at a time.  Every other call
    // about the state (feed, step, sample, checkpoint, slot_save, tokens, penalties, set_images) is the bound one's,
    // as before - so a conversation is read, saved and taken back exactly as it was alone.
    int seq_count() const;
    /// Makes conversation `s` the model's: its state, position, tokens, checkpoints, penalties and pictures.  (The
    /// states' device memory is theirs alone: an in-flight window of another conversation is unaffected.)
    void seq_bind(int s);
    /// The tokens conversation `s` can reach: the context, or STRATA_DS4_SEQ_CTX for the states after the first.
    int64_t seq_context(int s) const;
    /// One token of conversation `s` through every layer of the split without waiting for it (threads mode): the
    /// conversation consumes it (pos + 1), its sample follows on the head's stream with the request's chain, and the
    /// token lands in the conversation's own slot, whose event multi_take waits for.  The conversation is bound.
    bool multi_issue(int s, int token, float temperature, float top_p, int top_k, float min_p, uint64_t seed,
                     uint64_t counter, std::string& err);
    /// The token multi_issue sent for conversation `s`, waited for (its window through every layer done).  A token < 0
    /// and err: the window failed (a kernel's error), and the conversation's state is not to be trusted past it.
    int multi_take(int s, std::string& err);
    /// Called with s once conversation s's issued token is ready (multi_take returns it at once): on a CUDA thread,
    /// after the window's last work on the head's stream, so it only posts - it must not call the model.  Null: none.
    void multi_on_done(std::function<void(int)> f);
    /// The boundary between rounds of several conversations (GLM's multi_boundary): every device's stream finished, then
    /// the expert cache's look at the tiers - between windows, where nothing in flight is moved out from under it.  The
    /// engine calls it once every STRATA_DS4_MULTI_BOUNDARY windows (32, times the conversations running) of an
    /// answer, every issued token taken back first; `tokens` is what they decoded since the last one (its decay).
    bool multi_boundary(int tokens, std::string& err);

    /// The DSpark drafter (speculative decoding): its block size (0: not loaded), drafts for the positions after `y`
    /// (y: the token at pos(), not fed yet) with their confidences, and the verification of [y, drafts]: how many of the
    /// drafts stand and the token after them (the state then holds y and the accepted drafts).  The drafts are the
    /// drafter's argmax; a sampled request passes its chain (the same temperature / top_p / top_k / min_p / seed /
    /// counter as its verify()) so that with DS4_DSPARK_SAMPLE=1 they are drawn from the drafter's distribution, as
    /// DeepSeek's reference does.  verify() keeps a draft d with probability min(1, p(d) / q(d)) over the request's
    /// chain - the speculative-sampling rule, so every token is still distributed as the model's own draw.
    int draft_block() const;
    bool draft(int y, std::vector<int>& drafts, std::vector<float>& conf, std::string& err, float temperature = 0.0f,
               float top_p = 1.0f, int top_k = 0, float min_p = 0.0f, uint64_t seed = 0, uint64_t counter = 0);
    bool verify(int y, const std::vector<int>& drafts, float temperature, float top_p, int top_k, float min_p,
                uint64_t seed, uint64_t counter, int& accepted, int& next, std::string& err);
    /// A verify window's tokens at most (y and the drafts after it); verify() takes drafts from any source - those not
    /// the drafter's own draws as a point each (q 1)
    int verify_max() const;
    /// The last draft()'s drafts were drawn from the drafter's distribution (DS4_DSPARK_SAMPLE=1), not its argmax
    bool drafts_sampled() const;

    /// Tokens a feed call takes at a time to use the chunked prompt path (all of a prompt: its chunks run as a
    /// pipeline; 16 without it).
    int feed_unit() const;
    /// Called from feed() while a run of chunks goes through the layers, with the tokens done so far.
    void set_progress(std::function<void(int64_t)> f);
    /// Asked while a run of chunks goes through the layers: true (a Stop) ends it early - one GPU after the layer it is
    /// on, several at a chunk all of them reach - and feed() returns false with cancelled() true.  The state past the
    /// prompt's start is then not usable: restore() a checkpoint kept before the feed, or reset().
    void set_cancel(std::function<bool()> f);
    bool cancelled() const;   // the last feed() / feed_mark() ended early on a Stop
    /// A request starts (before its prompt): DISK_SKIP_AUTO may switch back on here, never mid-answer.
    void begin_request();
    int64_t pos() const;
    int64_t max_context() const;
    int vocab() const;
    int dim() const;
    const std::vector<int32_t>& tokens() const;   // what the state has consumed, in order

    void reset();
    /// Keep the state at the current position in checkpoint slot 0 or 1 (the end of a prompt, the position before it);
    /// restore(limit) goes back to the latest one at or before limit (-1: none).
    void checkpoint(int slot = 0);
    bool restore(int64_t limit);
    int64_t restorable(int64_t limit) const;
    /// Checkpoint 1 also becomes the anchor (kept until a restore goes before it): the position before the end of the
    /// first prompt of a run that continued in place, where a re-rendered history (earlier reasoning dropped) splits.
    void pin();
    /// The positions restore() can go back to, latest first (a slot keeps them: it can be taken back at any of them).
    std::vector<int64_t> check_positions() const;

    /// The conversation in the model into `out` (slot_save): its position, its tokens, its pictures' marks (`pics`:
    /// what the caller's state holds), every window ring (the DSpark blocks' included), the compressors' partial groups
    /// and each KV source's compressed KV and index keys up to it, the pieces the state a prompt continues is made of.
    /// 0 and err: nothing was set aside (there is no conversation, a picture's mark is not the state's, or the
    /// conversation is already set aside - unchanged since).
    uint64_t slot_save(std::vector<uint8_t>& out, std::string& err, const std::vector<PicMark>& pics = {});
    /// `data` (a slot's bytes) back into the model: the state at the slot's position n_pos, as slot_save took it -
    /// what a prompt that continues the slot's tokens reads on from (the caller checked the tokens are a prefix and the
    /// pictures agree).  The slot's pictures' marks go to `pics` (`pics` null: a slot that holds pictures is refused -
    /// a caller that cannot check them must not serve it).
    bool slot_load(const uint8_t* data, size_t bytes, int64_t n_pos, std::string& err,
                   std::vector<PicMark>* pics = nullptr);
    /// What a saved conversation must match: this engine's version, this model's geometry and files and the drafter's
    /// blocks - a slot naming anything else is not this model's state (a file's .meta says so; slot_load refuses).
    std::string slot_identity() const;

    /// The vision tower: the token every position of an image's span carries (-1: no tower).
    int image_token() const;
    /// An image's pixels (RGB rows, width x height, both multiples of the patch size: the server's resize) -> the
    /// rows of its span, BF16 [n_h * (n_w + 1) + 2][dim]: image_start, n_h rows of (n_w aligner rows, image_newline),
    /// image_end.
    bool encode_image(const uint8_t* rgb, int width, int height, std::vector<uint16_t>& span, std::string& err);
    /// The tower holds no GPU memory between pictures: an encode_image borrows its room (the prompt buffers, expert
    /// slots, or free memory) and gives it back.  vision_hold(): a request's pictures share one borrowing - the room,
    /// taken at the first encode_image, stays till vision_release() (before the prompt is fed: its buffers may be the
    /// lender).
    void vision_hold();
    /// false: expert slots lent for the pictures did not come back as they were (the request must fail).
    bool vision_release();
    /// Called after the tower's room is taken and after it is given back (the server's Monitor follows the VRAM lines).
    void set_vision_note(std::function<void()> note);
    /// The pictures of the prompt about to be fed ({}: none): positions inside a span read its rows (and route with
    /// the image bias, and take no part in the Engram n-grams); the rest the embedding table's.
    void set_images(std::vector<Image> images);

    void save_profile() const;
    const Stats& stats() const;
    std::string describe() const;
    /// INFO / VRAM lines for the server's Monitor.
    std::vector<std::string> info_lines() const;
    /// Where the routed experts live (the drafter's apart), for the Monitor's tiers: in VRAM; in RAM - copied there, or
    /// read ahead into the page cache (in place); the rest read from the SSD when routed.  cpu_blob: a CPU-lane expert's
    /// bytes on average (the SSD's reads a token -> experts a token)
    struct Tiers {
        int vram = 0, ram = 0, cpu = 0;
        double vram_gb = 0, ram_gb = 0, cpu_blob = 0;
    };
    Tiers tiers() const;

    struct Impl;

private:
    std::unique_ptr<Impl> p_;
};

}  // namespace strata::ds4
