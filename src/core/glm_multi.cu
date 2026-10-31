// src/core/glm_multi.cu - several conversations at once (STRATA_GLM_SEQS=<n>); see glm_model.hpp (SeqCtx).
//
// Every per-sequence thing a part keeps lives in its state arena (the residual, the KDA states and conv histories, the
// DSA latent / indexer caches) plus a handful of members (the position, the snapshot, the image rows).  A part holds n
// arenas and binds one: the token path, the prompt path, snapshots and slots all read `state_`, so they work on the
// bound sequence unchanged.  Activations and the tiers' machinery stay shared: a part runs one token at a time on its
// stream, whichever sequence it belongs to.
//
// THE PIPELINED DECODE.  multi_issue enqueues one token of one sequence through every part (each part's layers on its
// stream, the residual crossing over through that sequence's pinned hop slot, ordered by an event) and its head and
// sampler on the tail, and returns at once.  With several sequences in flight, the cards of a layer split work on
// different sequences at the same time: the first card on sequence C while the last one is on A.  The host only waits
// for the oldest sequence's token (multi_take), and issues that sequence's next one right away.
#include "glm_fast_state.hpp"
#include "strata/core/glm_prefix.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core {

// Task 2.1: the dynamic lifecycle.  seq_ stays pre-sized to the cap at load with slot 0 as the
// load-time alias and every later slot parked (all pointers null), so glm_model.cu's destructor
// sweep (the per-pointer guards over {state, snap, snap_pool, kda_bak}, glm_model.cu:589-592)
// needs no edit: it frees nothing on parked slots.  seq_acquire allocates the sequence's own
// buffers on demand (free-list first) and builds the SeqCtx through the UNCHANGED seq_layout();
// seq_release frees them back (the bytes-outstanding counter returns to 0) and parks the slot.
// The expert pool sizing subtracts the guaranteed states' floor (seq_bytes_floor_, the
// STRATA_GLM_SEQ_FLOOR first states; default the full cap = seq_bytes_reserved_): the cap's
// tail is the pressure ladder's to arbitrate (generate.cpp's evict_idle_for_admission).
// The pinned pipeline buffers (seq_pipe_ready's cudaHostAlloc per sequence)
// live in ONE cap-sized reusable pool carved per slot.

size_t Glm5Model::seq_arena_bytes() const {
    // one parked slot's own arena bytes: captured at load from the STRATA_GLM_SEQ_CTX layout
    // (seq_alloc_arenas), the sizing anchor every acquire re-runs
    return seq_state_bytes_;
}

// the shared allocation a parked slot's acquire runs: the UNCHANGED seq_layout fills the slot's
// offsets, then the incumbent's own alloc pattern (cudaMalloc + cudaMemset) takes the arena and
// the snapshot pair; bytes outstanding grows by exactly what was allocated.  Returns false on
// OOM (the slot stays parked, the caller decides).
bool Glm5Model::seq_alloc_slot(int s) {
    if (s <= 0 || s >= (int) seq_.size()) return s == 0;   // slot 0 is the load-time alias
    SeqCtx& c = seq_[(size_t) s];
    if (c.state) return true;                              // already live
    cudaSetDevice(dev_);
    seq_layout(seq_ctx_, c);                               // the UNCHANGED layout arithmetic
    float* a = nullptr;
    if (cudaMalloc(&a, c.bytes) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    cudaMemset(a, 0, c.bytes);
    c.state = a;
    seq_bytes_outstanding_ += c.bytes;
    if (seq_snap_n_rec_bytes_ > 0) {
        float* sn = nullptr;
        if (cudaMalloc(&sn, seq_snap_n_rec_bytes_) != cudaSuccess) {
            cudaGetLastError();
            cudaFree(c.state);
            c.state = nullptr;
            seq_bytes_outstanding_ -= c.bytes;
            return false;
        }
        cudaMemset(sn, 0, seq_snap_n_rec_bytes_);
        c.snap = sn;
        seq_bytes_outstanding_ += seq_snap_n_rec_bytes_;
    }
    if (seq_snap_pool_bytes_ > 0) {
        float* sp = nullptr;
        if (cudaMalloc(&sp, seq_snap_pool_bytes_) != cudaSuccess) {
            cudaGetLastError();
            cudaFree(c.state);
            c.state = nullptr;
            seq_bytes_outstanding_ -= c.bytes;
            if (c.snap) { cudaFree(c.snap); c.snap = nullptr; seq_bytes_outstanding_ -= seq_snap_n_rec_bytes_; }
            return false;
        }
        cudaMemset(sp, 0, seq_snap_pool_bytes_);
        c.snap_pool = sp;
        seq_bytes_outstanding_ += seq_snap_pool_bytes_;
    }
    // KV streaming, composed for ours' lazy lifecycle (upstream v1.1.29 allocates every sequence's
    // stream at load inside seq_alloc_arenas; ours parks the slots at load, so the stream joins at
    // first acquire, mirroring the snapshot pair): the arena's latent runs are the VRAM slots (the
    // UNCHANGED layout arithmetic above already sizes them from kv_resident_cells), the host copy
    // is this slot's own.  Parked slots hold none (kv_prep_targets routes them to the incumbent
    // path), the destructor still sweeps sequence 0's from kv_ itself.  No DSA layer here or a
    // non-streamed context: kv_stream_make returns null with err left empty and the slot decodes
    // incumbent-style (no bytes leg).
    const int64_t kv_cells = kv_resident_cells(g_.idx_kpool, seq_ctx_);
    if (kv_cells > 0) {
        std::string kv_err;
        c.kv = kv_stream_make(kv_cells, seq_ctx_, kv_err);
        if (!kv_err.empty()) {
            cudaGetLastError();
            cudaFree(c.state);
            c.state = nullptr;
            seq_bytes_outstanding_ -= c.bytes;
            if (c.snap) { cudaFree(c.snap); c.snap = nullptr; seq_bytes_outstanding_ -= seq_snap_n_rec_bytes_; }
            if (c.snap_pool) { cudaFree(c.snap_pool); c.snap_pool = nullptr; seq_bytes_outstanding_ -= seq_snap_pool_bytes_; }
            std::fprintf(stderr, "glm multi: sequence %d's KV stream did not allocate: %s\n", s, kv_err.c_str());
            return false;
        }
    }
    // A parked sequence does not inherit slot 0's pre-pool speculative backup.
    // Allocate it with the rest of this slot, before admission can run a draft:
    // allocating on the first spec step can fail after the request is already live.
    const size_t bak_bytes = getenv("STRATA_GLM_NO_MTP") == nullptr && getenv("STRATA_GLM_NO_SPEC") == nullptr
                                 ? spec_kda_bytes() : 0;
    if (bak_bytes > 0 && cudaMalloc(&c.kda_bak, bak_bytes) != cudaSuccess) {
        cudaGetLastError();
        c.kda_bak = nullptr;
        if (c.kv) { kv_stream_free(c.kv); c.kv = nullptr; }
        cudaFree(c.state); c.state = nullptr; seq_bytes_outstanding_ -= c.bytes;
        if (c.snap) { cudaFree(c.snap); c.snap = nullptr; seq_bytes_outstanding_ -= seq_snap_n_rec_bytes_; }
        if (c.snap_pool) { cudaFree(c.snap_pool); c.snap_pool = nullptr; seq_bytes_outstanding_ -= seq_snap_pool_bytes_; }
        return false;
    }
    seq_bytes_outstanding_ += bak_bytes;
    return true;
}

int Glm5Model::seq_acquire() {
    // the lazy acquire: a free-list slot first; the cap is a cap, past it fails cleanly
    int s = -1;
    if (!seq_free_.empty()) { s = seq_free_.back(); seq_free_.pop_back(); }
    if (s < 0 || s >= (int) seq_.size()) return -1;
    if (!seq_alloc_slot(s)) { seq_free_.push_back(s); return -1; }
    return s;
}

bool Glm5Model::seq_release(int s) {
    if (s < 0 || s >= (int) seq_.size()) return false;
    SeqCtx& c = seq_[(size_t) s];
    // Task 4.1: the park guard - an already-parked slot (not bound, nothing live) releases
    // nothing: no double subtract, no second free-list entry.  The scheduler's DONE/STOP/ERR
    // path may release a slot more than once across a conversation's turns.
    if (s != cur_seq_ && !c.state && !c.snap && !c.snap_pool && !c.kda_bak) return false;
    if (s == cur_seq_) {
        // bind another live slot first so the live members never dangle; slot 0 is always live
        // (the load-time alias), so the bound slot always has somewhere to go
        if (s != 0) seq_bind(0);
    }
    // the accounting reverses exactly what this release frees (the pointer truth): a slot whose
    // bind failed to allocate releases 0 bytes (seq_layout's capture set c.bytes, but the
    // acquire added nothing), and a second release's null pointers free nothing
    size_t freed = 0;
    if (s != 0 && c.state) {
        cudaFree(c.state);
        freed += c.bytes;
    }
    // the pair's bytes are the parked slots' own accounting (seq_alloc_slot counted them);
    // slot 0's pair is the members' lazy allocation (never counted; fast_destroy frees the
    // member copies at shutdown)
    if (c.snap) {
        cudaFree(c.snap);
        if (s != 0) freed += seq_snap_n_rec_bytes_;
    }
    if (c.snap_pool) {
        cudaFree(c.snap_pool);
        if (s != 0) freed += seq_snap_pool_bytes_;
    }
    if (c.kda_bak) {
        cudaFree(c.kda_bak);
        if (s != 0) freed += spec_kda_bytes();
    }
    // KV streaming, ours' lazy lifecycle: a freed slot's stream goes with it (the arena's
    // latent runs are the slots, so a stale slot pointer would hand a parked slot's slot
    // map to a new sequence); the alias keeps its members' stream with its arena
    if (s != 0 && c.kv) {
        kv_stream_free(c.kv);
        c.kv = nullptr;
    }
    seq_bytes_outstanding_ -= freed;
    if (s != 0) c.state = nullptr;   // slot 0's arena is the load-time state_ (the alias stays)
    c.snap = nullptr;
    c.snap_pool = nullptr;
    c.kda_bak = nullptr;
    // a parked slot advertises no snapshot: its pair is gone, so snapshot_restore's gate
    // (snap_pos_ >= 0) must not pass on a freshly-zeroed buffer at re-acquire
    c.snap_pos = -1;
    if (s != 0) {
        // the free list holds each parked slot exactly once (the guard above + the bind's pop
        // make a second entry impossible, but the check costs nothing and keeps the invariant
        // local)
        if (std::find(seq_free_.begin(), seq_free_.end(), s) == seq_free_.end())
            seq_free_.push_back(s);
    }
    if (s == 0 && cur_seq_ == 0) {   // slot 0's snapshot pointers were the members' (seq_bind's alias)
        snap_ = nullptr;
        snap_pool_ = nullptr;
        kda_bak_ = nullptr;
        snap_pos_ = -1;              // Task 4.1: the same no-snapshot park for the alias
    }
    return true;
}

// Task 4.1: the slot's pointer truth over every part of the split (the admission's live check,
// the pressure ladder's idle test).  Slot 0 is the load-time alias: its arena is the members'.
bool Glm5Model::seq_live(int s) const {
    if (s < 0 || s >= (int) seq_.size()) return false;
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        if (s >= (int) m->seq_.size()) return false;
        const SeqCtx& c = m->seq_[(size_t) s];
        if (!c.state && !c.snap && !c.snap_pool && !c.kda_bak) return false;
    }
    return true;
}

// Task 4.1: one more sequence fits if a free-list slot exists, its parked context holds the
// want, and the slot's own bytes (the arena + the snapshot pair, the load-time sizing anchors -
// every parked slot's are the same size on this part) fit the device's free VRAM on every part
// of the split.  STRATA_GLM_SEQ_VRAM_FIT forces the VRAM leg (the unit test pins the admission
// logic; the driver's real gate is cudaMemGetInfo's answer).
bool Glm5Model::seq_ready(int64_t want) const {
    if (seq_.empty()) return max_ctx_ >= want;
    const char* vf = getenv("STRATA_GLM_SEQ_VRAM_FIT");
    const bool vram_fit_forced = vf && std::atof(vf) != 0.0;
    bool visited = false;
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        if (m->seq_.empty()) continue;
        const size_t bak_bytes = getenv("STRATA_GLM_NO_MTP") == nullptr && getenv("STRATA_GLM_NO_SPEC") == nullptr
                                     ? m->spec_kda_bytes() : 0;
        const size_t need = m->seq_arena_bytes() + m->seq_snap_bytes(m->seq_[0]) + bak_bytes;
        bool slot = false;
        for (const int f : m->seq_free_)
            if (f > 0 && f < (int) m->seq_.size() && m->seq_[(size_t) f].max_ctx >= want) {
                slot = true;
                break;
            }
        if (!slot) return false;
        if (!vram_fit_forced) {
            size_t free_b = 0, total_b = 0;
            cudaSetDevice(m->dev_);
            if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) cudaGetLastError();
            if (free_b < need) { cudaSetDevice(dev_); return false; }
        }
        visited = true;
    }
    cudaSetDevice(dev_);
    return visited;
}

// Task 4.1: every part releases the slot (the pressure ladder's idle-state eviction); the
// answer is whether at least one part freed it.
int Glm5Model::seq_release_all(int s) {
    int released = 0;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        released += m->seq_release(s) ? 1 : 0;
    return released;
}

bool Glm5Model::seq_layout_matches() const {
    // the acquired layout must match load_pack's check: the same arena size + per-layer offsets
    // the loader verified (seq_alloc_arenas' own comparison, re-run against the first sequence)
    SeqCtx check;
    if (seq_.empty()) return seq_layout(max_ctx_, check) == state_bytes_;
    return seq_layout(max_ctx_, check) == state_bytes_ && check.dsa_pool == dsa_pool_ &&
           check.kda_conv == kda_conv_ && check.dsa_lat == dsa_lat_ && check.ik_ring == ik_ring_;
}

bool Glm5Model::seq_acquire_past_cap() {
    // acquire every free slot, then one more: the past-cap attempt must fail cleanly (-1, no
    // allocation, bytes outstanding unchanged) and the method undoes itself
    size_t before = seq_bytes_outstanding_;
    std::vector<int> taken;
    for (size_t i = 0; i < seq_.size() + 1; ++i) {
        int s = seq_acquire();
        if (s < 0) break;
        taken.push_back(s);
    }
    bool ok = (int) taken.size() == (int) seq_.size() - 1;  // every parked slot taken, slot 0 stays the alias
    for (int s : taken) seq_release(s);
    ok = ok && seq_bytes_outstanding_ == before;
    return ok;
}

bool Glm5Model::seq_parked_slots_null() const {
    // the destructor-sweep invariant: every slot on the free list has null pointers, so the
    // per-pointer guards free nothing on parked slots; slot 0 is the always-live alias
    for (size_t i = 0; i < seq_.size(); ++i) {
        const SeqCtx& c = seq_[i];
        bool live = (c.state || c.snap || c.snap_pool || c.kda_bak);
        bool on_free = std::find(seq_free_.begin(), seq_free_.end(), (int) i) != seq_free_.end();
        if (on_free && live) return false;      // parked slots' pointers are null
        if (!on_free && !live && (int) i != 0 && cur_seq_ != (int) i) return false;  // unacquired slots too
    }
    return true;
}

size_t Glm5Model::seq_snap_bytes(const SeqCtx& c) const {
    // one slot's snapshot pair bytes: the load-time capture (both sizes are per-slot constants
    // of this part's geometry, so every slot's pair is the same size)
    (void) c;
    return seq_snap_n_rec_bytes_ + seq_snap_pool_bytes_;
}

void* Glm5Model::seq_hop_slot(int s) {
    // ONE cap-sized reusable pool replaces seq_pipe_ready's cudaHostAlloc per sequence: the
    // pool is allocated once, lazily, and its slots are carved by sequence index (the hop row,
    // the embedding row, the sampled token), so the decode's call sites read the same pointers
    // unchanged.  The pool is portable pinned RAM: a layer split's parts cross devices through
    // it (part A's card writes the hop, part B's card reads it).
    if (!seq_hop_pool_) {
        const Glm5Geometry& g = g_;
        seq_hop_slot_bytes_ = (size_t) g.hc * g.n_embd * sizeof(float) + (size_t) g.n_embd * sizeof(float)
                              + sizeof(int);
        void* pool = nullptr;
        if (cudaHostAlloc(&pool, seq_hop_slot_bytes_ * seq_.size(), cudaHostAllocPortable) != cudaSuccess) {
            cudaGetLastError();
            return nullptr;
        }
        seq_hop_pool_ = pool;
    }
    char* base = (char*) seq_hop_pool_ + seq_hop_slot_bytes_ * (size_t) s;
    SeqCtx& c = seq_[(size_t) s];
    if (c.hop_h == nullptr) {
        const Glm5Geometry& g = g_;
        c.hop_h = (float*) base;
        c.emb_h = (float*) (base + (size_t) g.hc * g.n_embd * sizeof(float));
        c.tok_h = (int*) (base + ((size_t) g.hc + g.n_embd) * sizeof(float));
    }
    return base;
}

void Glm5Model::seq_bind(int s) {
    ckpt_join();   // (ckpt_save_snapshot reads snap_ / snap_pool_; upstream v1.1.29's fix, replayed)
    if (seq_.empty() || s == cur_seq_ || s < 0 || s >= (int) seq_.size()) return;
    SeqCtx& target = seq_[(size_t) s];
    // Task 4.1: the bind IS the scheduler's acquire (the driver binds without acquire,
    // generate.cpp's bind_ctx): when the bind allocates a parked slot, the slot's free-list
    // entry goes - a release pushes it back exactly once, so a second entry would double-count
    // the cap's bytes and hand the slot out twice
    const bool parked = target.state == nullptr;
    if (!seq_alloc_slot(s))
        std::fprintf(stderr, "glm multi: sequence %d's state did not fit on CUDA%d - it stays parked\n", s, dev_);
    else if (parked) {
        for (size_t i = seq_free_.size(); i-- > 0;)
            if (seq_free_[i] == s) {
                seq_free_.erase(seq_free_.begin() + (long) i);
                break;
            }
    }
    SeqCtx& o = seq_[(size_t) cur_seq_];
    o.state = state_;
    o.pos = pos_;
    o.snap = snap_;
    o.snap_pool = snap_pool_;
    o.snap_pos = snap_pos_;
    o.kda_bak = kda_bak_;
    o.last_tok = last_tok_;
    o.img_pos = std::move(img_pos_);
    o.img_rows = std::move(img_rows_);
    o.max_ctx = max_ctx_;
    o.bytes = state_bytes_;
    o.ik_ring = ik_ring_;
    std::swap(o.kda_S, kda_S_);
    std::swap(o.kda_conv, kda_conv_);
    std::swap(o.dsa_lat, dsa_lat_);
    std::swap(o.dsa_ik, dsa_ik_);
    std::swap(o.dsa_ig, dsa_ig_);
    std::swap(o.dsa_pool, dsa_pool_);
    o.kv = kv_;   // (KV streaming: the bound sequence's stream lives in kv_, its SeqCtx holds none; upstream v1.1.29)
    SeqCtx& n = seq_[(size_t) s];
    state_ = n.state;
    pos_ = n.pos;
    snap_ = n.snap;
    snap_pool_ = n.snap_pool;
    snap_pos_ = n.snap_pos;
    kda_bak_ = n.kda_bak;
    last_tok_ = n.last_tok;
    img_pos_ = std::move(n.img_pos);
    img_rows_ = std::move(n.img_rows);
    n.img_pos.clear();
    n.img_rows.clear();
    max_ctx_ = n.max_ctx;
    state_bytes_ = n.bytes;
    ik_ring_ = n.ik_ring;
    std::swap(n.kda_S, kda_S_);
    std::swap(n.kda_conv, kda_conv_);
    std::swap(n.dsa_lat, dsa_lat_);
    std::swap(n.dsa_ik, dsa_ik_);
    std::swap(n.dsa_ig, dsa_ig_);
    std::swap(n.dsa_pool, dsa_pool_);
    std::swap(n.lat_table, lat_table_);   // Task 1.3: the latent page table follows the sequence (identity until P3)
    kv_ = n.kv;   // (KV streaming, upstream v1.1.29: the bound sequence's stream; the parked one holds none)
    n.kv = nullptr;
    // Task 3.2: the acquire path adopts the registry pages the intake lookup
    // refcounted, so a re-acquired sequence re-enters with its shared-prefix
    // pages still resident and refcounted (an evicted page never breaks a
    // live request: the registry's refcount held it while the slot was parked).
    cur_seq_ = s;
}

void Glm5Model::seq_bind_all(int s) {
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) m->seq_bind(s);
}

// load_pack's state layout for a context of ctx tokens: the residual pair, then the NextN block's DSA
// caches, then this part's layers - the offsets into c's vectors, the bytes returned.  The same
// path-dependent arithmetic load_pack runs (the FP16 latent + the ik ring on the fast path, the F32
// latent + the whole context on the reference path), so acquire reproduces either load's layout.
uint64_t Glm5Model::seq_layout(int64_t ctx, SeqCtx& c) const {
    const Glm5Geometry& g = g_;
    const int64_t hc_dim = (int64_t) g.hc * g.n_embd;
    const int64_t max_pools = ctx / g.idx_kpool;
    int64_t floats = 2 * hc_dim;
    // (KV streaming, upstream v1.1.29 replayed: the arena holds the VRAM slots of kv_resident_cells
    // positions, exactly as load_pack lays it out; the setting off returns 0 and this is the
    // incumbent full-context arithmetic)
    const int64_t kv_cells = kv_resident_cells(g.idx_kpool, ctx);
    const int64_t lat_rows = kv_cells > 0 ? kv_cells : ctx;
    const int64_t lat_floats = lat_q8_ ? (int64_t) gf::lat8_rec_bytes(g.kv_lora) / 4 * lat_rows
                                       : fast_mode_ ? (int64_t) g.kv_lora * lat_rows / 2
                                                    : (int64_t) g.kv_lora * lat_rows;
    for (auto* v : {&c.kda_S, &c.kda_conv, &c.dsa_lat, &c.dsa_ik, &c.dsa_ig, &c.dsa_pool})
        v->assign((size_t) g.n_layers + 1, 0);
    c.ik_ring = fast_mode_ ? (int) std::min<int64_t>(ctx, 8192 + 64) : (int) ctx;
    const auto dsa = [&](int il) {
        c.dsa_lat[(size_t) il] = floats;
        floats += lat_floats;
        c.dsa_ik[(size_t) il] = floats;
        floats += (int64_t) g.idx_key * c.ik_ring;
        c.dsa_ig[(size_t) il] = floats;
        floats += (int64_t) g.idx_key * c.ik_ring;
        c.dsa_pool[(size_t) il] = floats;
        floats += (int64_t) g.idx_key * max_pools;
    };
    if (mtp_il_ >= 0) dsa(mtp_il_);
    for (int il = l0_; il < l1_; ++il) {
        if (g.is_recr(il)) {
            c.kda_S[(size_t) il] = floats;
            floats += (int64_t) g.d_inner() * g.kda_head_dim;
            c.kda_conv[(size_t) il] = floats;
            floats += (int64_t) 3 * g.d_inner() * (g.d_conv - 1);
        } else {
            dsa(il);
        }
    }
    c.max_ctx = ctx;
    c.bytes = (uint64_t) floats * sizeof(float);
    return c.bytes;
}

// STRATA_GLM_SEQS=<n> (1..8): n - 1 more state arenas on this part, zeroed - each for STRATA_GLM_SEQ_CTX tokens
// (default: the engine's context; at least 8320, at most the context).  Called by load_pack right after the first
// arena, so the expert pool is sized from what is left.
bool Glm5Model::seq_alloc_arenas(std::string& err) {
    const char* v = getenv("STRATA_GLM_SEQS");
    const int n = v ? std::max(1, std::min(8, std::atoi(v))) : 1;
    if (n <= 1) return true;
    int64_t ctx = max_ctx_;
    if (const char* c = getenv("STRATA_GLM_SEQ_CTX")) ctx = std::max<int64_t>(8320, std::min<int64_t>(max_ctx_, std::atoll(c)));
    cudaSetDevice(dev_);
    seq_.assign((size_t) n, SeqCtx{});   // the cap: every slot parked at load (Task 2.1), seq_free_ = all but 0
    // the first sequence's layout recomputed must be the one load_pack made, or the others' would be wrong
    SeqCtx check;
    if (seq_layout(max_ctx_, check) != state_bytes_ || check.dsa_pool != dsa_pool_ || check.kda_conv != kda_conv_ ||
        check.dsa_lat != dsa_lat_ || check.ik_ring != ik_ring_) {
        err = "glm multi: the state layout does not match load_pack's (this build cannot run STRATA_GLM_SEQS)";
        return false;
    }
    // slot 0 stays the load-time alias (the members hold its arena), the rest park: only their
    // sizes are captured at load, the buffers come at seq_acquire (Task 2.1: the cap is a cap,
    // not a reservation - the parked slots' bytes must stay accounted against the expert pool)
    seq_[0].state = state_;
    seq_[0].kv = kv_;   // KV streaming: the alias holds the members' stream (binding back to slot 0 restores it)
    for (int s = 1; s < n; ++s) seq_free_.push_back(s);
    seq_ctx_ = ctx;                                        // the parked slots' own context
    // Task 4.1: the parked slots' own context is known at load (the admission's fits() reads
    // seq_max_ctx(c) for every never-acquired slot; 0 would hide them from the cap entirely)
    for (int s = 1; s < n; ++s) seq_[(size_t) s].max_ctx = seq_ctx_;
    SeqCtx parked;
    seq_layout(seq_ctx_, parked);                          // the capture every acquire re-runs
    const size_t one_arena = (size_t) parked.bytes;
    size_t one_snaps = 0;
    {
        const Glm5Geometry& g = g_;
        const size_t run = (size_t) g.d_inner() * g.kda_head_dim + (size_t) 3 * g.d_inner() * (g.d_conv - 1);
        int n_rec = 0;
        for (int il = l0_; il < l1_; ++il) n_rec += g.is_recr(il) ? 1 : 0;
        const size_t pool_floats = dsa_layers().size() * 2 * (size_t) ((g.idx_kpool - 1) * g.idx_key);
        seq_snap_n_rec_bytes_ = (size_t) n_rec * run * sizeof(float);
        seq_snap_pool_bytes_ = pool_floats * sizeof(float);
        one_snaps = seq_snap_n_rec_bytes_ + seq_snap_pool_bytes_;
    }
    seq_state_bytes_ = one_arena;
    // slot 0's arena is the load-time state_ (the load-time alias, never parked); its snapshot
    // pair joins on demand (snapshot_save / snap_pool_copy) but belongs to the cap too, so the
    // reservation carries it - the pool sizing must not spend the bytes slot 0 will need
    // Slot 0's backup was allocated before the pool; only parked slots need
    // their speculative backups charged against the sequence reservation.
    const size_t one_bak = getenv("STRATA_GLM_NO_MTP") == nullptr && getenv("STRATA_GLM_NO_SPEC") == nullptr
                               ? spec_kda_bytes() : 0;
    seq_bytes_reserved_ = ((size_t) n - 1) * (one_arena + one_snaps + one_bak) + one_snaps;
    // Task 4.1 amendment: the full-cap reservation starves the
    // expert pool while the ladder cannot reach pressure at the production context (the 2-GPU
    // split halves the per-state cost). The pool sizing subtracts the FLOOR instead: the first
    // STRATA_GLM_SEQ_FLOOR states' bytes stay guaranteed (the common case keeps Task 2.1's
    // promise), and the cap's tail is the ladder's to arbitrate (each parked acquire the device
    // cannot honor evicts live-idle states LRU-first, or the request waits - never a failed
    // bind). Default: the full cap (today's sizing, unchanged). seq_bytes_reserved_ stays the
    // cap's true accounting.
    const char* fl = getenv("STRATA_GLM_SEQ_FLOOR");
    int f = fl ? std::atoi(fl) : n;
    f = std::max(1, std::min(n, f));
    seq_bytes_floor_ = ((size_t) f - 1) * (one_arena + one_snaps + one_bak) + one_snaps;
    std::fprintf(stderr, "glm multi: CUDA%d the state floor: %d guaranteed state(s), %.2f GB "
                 "the pool sizing sees\n", dev_, f, (double) seq_bytes_floor_ / 1e9);
    // the pinned pipeline buffers: ONE cap-sized reusable pool, carved per slot at first acquire
    // (seq_hop_slot); its bytes are host RAM, not VRAM, so the pool sizing ignores them
    uint64_t more = (uint64_t) seq_bytes_reserved_;
    std::fprintf(stderr, "glm multi: CUDA%d %d sequence states: the first for %lld tokens (%.2f GB), %d more parked "
                 "for %lld (reserved, %.2f GB in all with every sequence's snapshot)\n", dev_, n, (long long) max_ctx_,
                 (double) state_bytes_ / 1e9, n - 1, (long long) ctx, (double) more / 1e9);
    // several conversations drafting at once (STRATA_GLM_MULTI_MTP): their windows' buffers, before the pool too
    if (multi_mtp_setting() > 0 && !mtpw_alloc(err)) return false;
    // one GPU: the sequences decode together (glm_rows.cu), their activations allocated here, before the pool;
    // ours keeps the fast-mode gate (the rows arena is fast-path-bound through rows_ok/split_rows_ok, so slow mode
    // would allocate VRAM nothing reads; upstream v1.1.29 calls it unconditionally, diverged deliberately)
    if (n_parts_ == 1 && fast_mode_) return rows_alloc(err);
    return true;
}

bool Glm5Model::seq_pipe_ready(std::string& err) {
    const Glm5Geometry& g = g_;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        if (m->seq_.empty()) {
            err = "glm multi: one sequence state only (STRATA_GLM_SEQS)";
            return false;
        }
        cudaSetDevice(m->dev_);
        for (size_t s = 0; s < m->seq_.size(); ++s) {
            // the pinned buffers come from the cap-sized reusable pool (Task 2.1: one
            // cudaHostAlloc instead of one per sequence), carved into the slot on first use
            SeqCtx& c = m->seq_[s];
            if (c.hop_h == nullptr && m->seq_hop_slot((int) s) == nullptr) {
                cudaGetLastError();
                err = "glm multi: the pipeline buffer pool did not allocate";
                return false;
            }
            if (c.ev_hop == nullptr && cudaEventCreateWithFlags(&c.ev_hop, cudaEventDisableTiming) != cudaSuccess) {
                cudaGetLastError();
                err = "glm multi: the pipeline events did not allocate";
                return false;
            }
            if (c.ev_done == nullptr && cudaEventCreateWithFlags(&c.ev_done, cudaEventDisableTiming) != cudaSuccess) {
                cudaGetLastError();
                err = "glm multi: the pipeline events did not allocate";
                return false;
            }
        }
    }
    cudaSetDevice(dev_);
    (void) g;
    return true;
}

bool Glm5Model::multi_issue(int s, int32_t token, const strata::kernels::SamplerParams& sp, std::string& err) {
    if (s < 0 || s >= seq_count() || seq_.empty()) {
        err = "glm multi: no sequence " + std::to_string(s);
        return false;
    }
    if (seq_[0].hop_h == nullptr && !seq_pipe_ready(err)) return false;
    const Glm5Geometry& g = g_;
    seq_bind_all(s);
    FastState* F = fast_;
    SeqCtx& c = seq_[(size_t) s];
    const int64_t p = pos_++;
    if (p + 1 >= max_ctx_) {
        err = "glm multi: the context is full";
        return false;
    }
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
        err = "glm multi: no embedding dequantizer";
        return false;
    }
    // this sequence's own pinned row: its previous token has completed (multi_take), so no copy still reads it
    tt->to_float(pack_emb_src_ + (size_t) token * ggml_row_size((ggml_type) pack_emb_type_, g.n_embd), c.emb_h,
                 g.n_embd);
    if (const float* img = image_row(p)) std::memcpy(c.emb_h, img, (size_t) g.n_embd * sizeof(float));
    cudaSetDevice(dev_);
    cudaMemcpyAsync(F->emb, c.emb_h, (size_t) g.n_embd * sizeof(float), cudaMemcpyHostToDevice, F->cs);
    gf::embed_streams(F->emb, state_, g.n_embd, F->cs);
    if (!fast_layers(p, false, err)) return false;
    Glm5Model* tail = this;
    const size_t hop = (size_t) g.hc * g.n_embd * sizeof(float);
    for (Glm5Model* B = split_next_.get(); B != nullptr; B = B->split_next_.get()) {
        SeqCtx& ca = tail->seq_[(size_t) s];
        cudaSetDevice(tail->dev_);
        cudaMemcpyAsync(ca.hop_h, tail->state_, hop, cudaMemcpyDeviceToHost, tail->fast_->cs);
        cudaEventRecord(ca.ev_hop, tail->fast_->cs);
        cudaSetDevice(B->dev_);
        B->pos_ = pos_;
        cudaStreamWaitEvent(B->fast_->cs, ca.ev_hop, 0);
        cudaMemcpyAsync(B->state_, ca.hop_h, hop, cudaMemcpyHostToDevice, B->fast_->cs);
        if (!B->fast_layers(p, true, err)) return false;
        tail = B;
    }
    FastState* FT = tail->fast_;
    SeqCtx& ct = tail->seq_[(size_t) s];
    cudaSetDevice(tail->dev_);
    gf::head_prep(tail->state_, tail->w_.at("output_norm.weight"), g.norm_eps, g.n_embd, FT->head_x, FT->head_xq,
                  FT->cs);
    const WSlot& ow = tail->ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, FT->head_xq, FT->head_x, tail->sc_ + tail->sc_logits, nullptr, 1.0f, ow.type, g.n_embd,
                   g.n_vocab};
    if (ow.type == 0) o.w = ow.f32;
    if (!gf::mv(&o, 1, FT->cs)) {
        err = "glm multi: output head";
        return false;
    }
    // the logits are the tail's shared scratch: the draw goes on the same stream right behind them, before the next
    // sequence's head can write them again
    if ((sp.greedy || sp.temperature <= 0.0f) && sp.penalty_last_n <= 0)   // (penalties: the sampler's greedy pick)
        gf::argmax(tail->sc_ + tail->sc_logits, g.n_vocab, FT->tok, FT->cs);
    else
        tail->pick_tokens(tail->sc_ + tail->sc_logits, 1, sp, nullptr, 0, FT->tok, FT->cs);
    cudaMemcpyAsync(ct.tok_h, FT->tok, sizeof(int), cudaMemcpyDeviceToHost, FT->cs);
    cudaEventRecord(ct.ev_done, FT->cs);
    cudaSetDevice(dev_);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("glm multi: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

int Glm5Model::multi_take(int s, std::string& err) {
    Glm5Model* tail = this;
    while (tail->split_next_) tail = tail->split_next_.get();
    SeqCtx& ct = tail->seq_[(size_t) s];
    cudaSetDevice(tail->dev_);
    if (!glmfast::wait_event(ct.ev_done, "glm multi token", err)) return -1;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (!m->fast_->route_ok(err)) return -1;
    if (gf::launch_errors() > 0) {
        err = "glm multi: " + std::to_string(gf::launch_errors()) + " kernel launch(es) failed (see stderr)";
        return -1;
    }
    cudaSetDevice(dev_);
    const int tok = ct.tok_h[0];
    if (s == cur_seq_) last_tok_ = tok;
    else seq_[(size_t) s].last_tok = tok;
    return tok;
}

bool Glm5Model::multi_boundary(std::string& err) {
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        if (!m->fast_boundary(err)) return false;
    }
    cudaSetDevice(dev_);
    return true;
}
// ---------------------------------------------------------------------------
// Task 3.2 Step 1: the prefix cache (the intake lookup + LRU eviction), gated
// on STRATA_GLM_PAGED (the pool exists) - the incumbent path is untouched.
//
// A block is one TILE of the prompt: up to STRATA_GLM_PAGE_SIZE positions,
// tiled contiguously from the prompt's position 0 (so a skip is always a
// whole chain of whole tiles).  The canonical rows live in the pool's slab
// pages (per part, one page per DSA layer; the identity table keeps the
// incumbent addresses, so the apply copies the rows back byte for byte).
// At each CHUNK end the chunk's last tile is STATEFUL: its boundary state -
// the KDA runs + the open key pool, exactly snapshot_save/snap_pool_copy's
// regions - is captured to host, so a continuation from that boundary is
// byte-identical to the registrar's.  The intake walks the chain by boundary
// order, verifies each tile's full-content ID (hash_block over the tokens +
// every layer's canonical bytes), refcounts the hits and reports the skip;
// the prefill then runs only the tail.  evict_lru runs under pool pressure
// only (page_alloc's sweep); the store page_refs its tiles' pages, drops its
// least-recently-hit refcount-0 tiles, and the pages return to the pool.
// (glm_pages.cu carries the unit-level LRU + pressure selftests; this file
// the store itself.)
//
// Spaces: a tile's START/BOUNDARY are PROMPT-relative (the intake's walk
// space; the adopting sequence's absolute position == the prompt index after
// the apply, since the sequence restarts at 0).  The capture walks in
// ABSOLUTE space (the arena's rows) and converts with
// prompt_base = pos_start - prefix_base_ (prefix_base_ = the intake's skip:
// 0 for a fresh or continuing request, the skip for a reused one).
// ---------------------------------------------------------------------------

namespace {

// One tile's per-part share: the canonical pages (one per DSA layer, in
// dsa_layers() order) + the boundary state's host copies (the stateful tiles)
// or the D6 carriage's slab pages (Task 3.3, the same flat stream).
struct PrefixTileData {
    int64_t start = 0, end = 0;     // the tile's prompt-relative span
    int64_t vec_start = 0;          // ... and its index into the capturing request's token vector
    bool stateful = false;          // the boundary state captured (a chunk's last tile)
    std::vector<int32_t> pages;     // the slab page per DSA layer
    std::vector<char> snap;         // the KDA runs at the boundary (stateful)
    std::vector<char> snap_pool;    // ... and the open key pool
    strata::prefix::SnapPages snap_gpu;       // Task 3.3: the D6 carriage (the runs stream)
    strata::prefix::SnapPages snap_pool_gpu;  // ... and the open key pool's stream
};
struct PrefixPartData {
    std::vector<PrefixTileData> tiles;   // this chunk's tiles, in span order
};
// One registered tile (the store's entry): the registry's slot indexes it.
struct PrefixBlock {
    strata::prefix::BlockId id;
    uint64_t key = 0;               // the tokens' key (the intake's index)
    int64_t start = 0, boundary = 0;   // the tile's prompt-relative span
    bool stateful = false;
    std::vector<PrefixPartData> parts;   // one per part, in split order
    uint64_t last_use = 0;          // the store's LRU (the last hit's clock)
};
// A chunk end's pending capture: the parts' shares land as their prefill_half
// returns; the finalize turns complete entries into store entries.
struct PrefixPendingChunk {
    int64_t chunk_start = 0;
    std::vector<PrefixPartData> parts;   // indexed by the part index
    std::vector<bool> present;
};

// the process-wide store (one engine per process) + its registry + the pending
// captures + the shared clock + the mutex (the pipelined path's parts capture
// on their own threads).
std::vector<PrefixBlock> g_prefix_blocks;
strata::prefix::BlockRegistry g_prefix_registry_;
std::unordered_map<int64_t, PrefixPendingChunk> g_prefix_pending;
uint64_t g_prefix_clock = 0;
std::mutex g_prefix_mu;

// the store's capacity (the least-recently-hit refcount-0 tiles get dropped).
int prefix_block_cap() {
    const char* v = getenv("STRATA_GLM_PREFIX_BLOCKS");
    return v ? std::max(4, std::atoi(v)) : 64;
}

// Task 3.3 (the D6 carriage): one boundary snapshot's flat stream packed
// straight into the pool's slab pages, D2D, region by region in stream order
// (the runs in recr-layer order, then the pool rows per DSA layer - the same
// order the host staging walks, so the apply reads either medium with
// identical offsets).  Each page is refcounted on allocation: the store's
// entry holds the refs, the pool's sweep cannot take a refcounted page.  The
// packer drops its own partial pages when anything fails (the caller falls
// back to the host staging) and leaves the empty list it started with.
struct SnapPacker {
    GlmPages& pool;
    strata::prefix::SnapPages& out;
    uint8_t* base = nullptr;   // the current page's first byte (nullptr: allocate)
    int64_t filled = 0;        // the bytes already in the current page
    bool put(const void* src, int64_t bytes) {
        const int32_t pb = pool.page_bytes();
        const char* s = (const char*) src;
        while (bytes > 0) {
            if (base == nullptr) {
                const int32_t pg = pool.page_alloc();
                if (pg < 0) return false;
                pool.page_ref(pg);
                out.pages.push_back(pg);
                base = (uint8_t*) pool.slab_ptr() + (size_t) pg * (size_t) pb;
                filled = 0;
            }
            const int64_t room = pb - filled;
            const int64_t chunk = bytes < room ? bytes : room;
            if (cudaMemcpy(base + filled, s, (size_t) chunk,
                           cudaMemcpyDeviceToDevice) != cudaSuccess) return false;
            filled += chunk; bytes -= chunk; s += chunk;
            if (filled == pb) { base = nullptr; filled = 0; }
        }
        return true;
    }
    void drop() {
        for (int32_t pg : out.pages) pool.page_unref(pg);
        out.pages.clear();
        out.bytes = 0;
        base = nullptr;
        filled = 0;
    }
};

// Task 3.3: a tile's pages back to the pool: the canonical rows' one page per
// DSA layer + the D6 carriage's snapshot pages (the store releases them only
// when the block leaves the store or the capture fails entirely).
static void drop_tile_pages(GlmPages& pool, const PrefixTileData& td) {
    for (int32_t pg : td.pages)
        if (pg >= 0) pool.page_unref(pg);
    for (int32_t pg : td.snap_gpu.pages)
        if (pg >= 0) pool.page_unref(pg);
    for (int32_t pg : td.snap_pool_gpu.pages)
        if (pg >= 0) pool.page_unref(pg);
}

// Task 3.3: the apply's read side: the flat stream's bytes at [off, off+bytes)
// back out of the carriage's pages (segmented async D2D; the region may span
// page boundaries).  Returns false when the pages ran short (the caller
// surfaces the error; the store holds the refs, so the pages cannot be
// reclaimed under a live request).
static bool snap_unpack(const GlmPages& pool, const std::vector<int32_t>& pages,
                        int64_t off, int64_t bytes, void* dst, cudaStream_t cs) {
    const int32_t pb = pool.page_bytes();
    int64_t done = 0;
    while (done < bytes) {
        const int64_t at = off + done;
        const size_t pi = (size_t) at / (size_t) pb;
        const int64_t in_off = (int64_t) (at % (int64_t) pb);
        if (pi >= pages.size()) return false;
        const int64_t room = pb - in_off;
        const int64_t chunk = std::min<int64_t>(bytes - done, room);
        if (cudaMemcpyAsync((uint8_t*) dst + done,
                            (const uint8_t*) pool.slab_ptr()
                                + (size_t) pages[pi] * (size_t) pb + (size_t) in_off,
                            (size_t) chunk, cudaMemcpyDeviceToDevice, cs) != cudaSuccess)
            return false;
        done += chunk;
    }
    return true;
}

// Task 4.2: the PAUSE path's registry-owned snapshots (one engine per process).
// A paused request's state (the KDA runs + the open ik/ig pool + the pooled sums
// + the latent rows [0, pos) through the bound table + the table's mode) packed
// into refcounted pool pages, one SnapPages per part of the split, keyed by the
// conversation's token span: the pages survive every pressure sweep and the
// slot's park (the incumbent snap_/snap_pool_ pair dies at the park).  The
// restore walks the entries for the longest span that is a strict prefix of the
// resuming prompt, unpacks into the freshly-acquired arena, consumes the entry.
// The registry is off = the pool off or STRATA_GLM_NO_PREFIX: the incumbent pair
// is the only path (unchanged).
struct PauseSnap {
    std::vector<int32_t> span;                 // the capture's token span (the restore's match)
    int64_t pos = 0;                           // the state's position (== span.size())
    std::vector<strata::prefix::SnapPages> gpu;   // the flat stream's pages, one per part
    std::vector<int> ring;                     // each part's ik_ring_ (the layout guard)
    const Glm5Model* owner = nullptr;          // the pages belong to this engine's slab
                                               // (the process store serves only matching
                                               // owners: the pool's off = the incumbent path)
    uint64_t used = 0;                         // the LRU cap's clock
};
std::vector<PauseSnap> g_pause_snaps;
// the entries' cap (a conversation that never resumes must not hold pages
// forever): the least-recently-used entry drops at capture time over this.
int pause_snap_cap() {
    const char* v = getenv("STRATA_GLM_PAUSE_BLOCKS");
    return v ? std::max(2, std::atoi(v)) : 8;
}

} // namespace

// a tile's full-content ID: the tokens then every part's layers' canonical
// bytes in split order (the same framing at registration and verify, so a hit
// verifies against the canonical pages as they are NOW - the D2H staging).
strata::prefix::BlockId Glm5Model::prefix_content_hash(const std::vector<int32_t>& toks,
                                            const std::vector<std::vector<int32_t>>& pages,
                                            const std::vector<Glm5Model*>& chain,
                                            int64_t tile_rows,
                                            const std::vector<int64_t>& row_bytes) {
    std::vector<std::vector<char>> buf;
    std::vector<const void*> ptrs;
    size_t page_bytes_hash = 0;
    cudaSetDevice(chain[0]->dev_);
    for (size_t p = 0; p < pages.size() && p < chain.size(); ++p) {
        Glm5Model* m = chain[p];
        cudaSetDevice(m->dev_);
        const int32_t page_stride = (int32_t) m->lat_pages_.page_bytes();
        for (size_t li = 0; li < pages[p].size(); ++li) {
            const int32_t pg = pages[p][li];
            const size_t span = (size_t) tile_rows * (size_t) row_bytes[p];
            buf.emplace_back(span);
            if (cudaMemcpy(buf.back().data(), (const char*) m->lat_pages_.slab_ptr()
                                + (size_t) pg * page_stride, span,
                            cudaMemcpyDeviceToHost) != cudaSuccess)
                return strata::prefix::BlockId{0, 0};
            ptrs.push_back(buf.back().data());
            if (page_bytes_hash == 0) page_bytes_hash = span;
        }
    }
    cudaSetDevice(chain[0]->dev_);
    return strata::prefix::hash_block(toks.data(), toks.size(), ptrs.data(), ptrs.size(), page_bytes_hash);
}
// the per-part row bytes (the latent rows' span per position) + the arena's
// row stride, in floats - the slot_dsa_runs arithmetic (the FP16 latent takes
// half a float each, INT8 records lat8_rec_bytes / 4).
int64_t Glm5Model::prefix_row_bytes(const Glm5Model* m) {
    return m->lat_q8_ ? (int64_t) gf::lat8_rec_bytes(m->g_.kv_lora)
                      : (int64_t) m->g_.kv_lora * 2;
}


// ---- the capture: the part's share of the tiles that became final at this
// chunk end.  Called on each part as its prefill_half returns (the part's own
// thread on the pipelined path); the head's finalize turns the pending shares
// into store entries.  boundary = the chunk's end (= pos_ after the chunk),
// chunk_start = the chunk's start, pos_start = the prompt's absolute base for
// this prefill (the first chunk's chunk_start; the intake's skip is 0 for a
// fresh request, so the prompt-aligned arena keeps every tile space absolute).
void Glm5Model::prefix_capture_part(int64_t boundary, int64_t chunk_start, int64_t pos_start) {
    if (lat_pages_.n_pages() <= 0) return;   // the pool is off: the incumbent path
    if (fast_ == nullptr) return;            // the reference path: the incumbent prefill
    if (getenv("STRATA_GLM_NO_PREFIX") != nullptr) return;   // the reference window: the cache is off
    const Glm5Geometry& g = g_;
    const int32_t page_rows = lat_pages_.page_size();
    const int64_t row_bytes = prefix_row_bytes(this);
    const int64_t prompt_base = 0;   // the prompt-aligned arena: the absolute and the
                                     // prompt-relative tile spaces are identical
    const std::vector<int> ls = dsa_layers();
    // the compute stream's writes (the state + the chunk's rows) settle first
    if (cudaStreamSynchronize(fast_->cs) != cudaSuccess) return;
    // one lock for the whole body: the pipelined path captures from TWO threads (the head's
    // inline call + the tail thread's), and GlmPages' allocator has no mutex of its own -
    // two concurrent page_allocs would race the bump allocator and double-assign pages.
    std::lock_guard<std::mutex> lk(g_prefix_mu);
    PrefixPartData pd;
    // the chunk's tiles: page-aligned spans from the chunk's start, the last
    // one partial to the chunk end (the stateful one)
    for (int64_t t0 = chunk_start; t0 < boundary; ) {
        const int64_t t1 = std::min<int64_t>(t0 + page_rows, boundary);
        const int64_t rel0 = t0 - prompt_base, rel1 = t1 - prompt_base;   // the tile's prompt-relative span
        PrefixTileData td;
        td.start = rel0; td.end = rel1;
        td.vec_start = t0 - pos_start;   // the index into this request's token vector
        td.stateful = (t1 == boundary);
        td.pages.assign(ls.size(), -1);
        bool ok = true;
        // the canonical rows: one slab page per DSA layer (the pressure sweep's
        // one place; the store's ref holds the page against the sweep)
        for (size_t li = 0; li < ls.size() && ok; ++li) {
            const int32_t pg = lat_pages_.page_alloc();
            if (pg < 0) { ok = false; break; }
            lat_pages_.page_ref(pg);
            td.pages[li] = pg;
            uint8_t* dst = (uint8_t*) lat_pages_.slab_ptr() + (size_t) pg * (size_t) lat_pages_.page_bytes();
            const uint8_t* src = (const uint8_t*) (state_ + dsa_lat_[(size_t) ls[li]]) + (size_t) t0 * row_bytes;
            const size_t bytes = (size_t)(t1 - t0) * row_bytes;
            ok = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice) == cudaSuccess;
        }
        if (ok && td.stateful) {
            // the boundary state: the KDA runs (S + the conv histories, one run
            // per recr layer) + the open key pool (snapshot_save's regions), in
            // the SAME flat order both mediums use.  Task 3.3: the D6 carriage
            // first (the pool's slab pages, the store's entry holds the refs);
            // the host staging below is the fallback when the pool cannot
            // serve.  All-or-nothing per tile: both streams must carry or
            // neither does (the apply reads both from one medium).
            const size_t run = (size_t) g.d_inner() * g.kda_head_dim + (size_t) 3 * g.d_inner() * (g.d_conv - 1);
            SnapPacker pk{lat_pages_, td.snap_gpu};
            SnapPacker pkp{lat_pages_, td.snap_pool_gpu};
            bool carried = true;
            for (int il = l0_; il < l1_ && carried; ++il) {
                if (!g.is_recr(il)) continue;
                carried = pk.put(state_ + kda_S_[(size_t) il], run * sizeof(float));
            }
            const int64_t rows = (int64_t) (g.idx_kpool - 1) * g.idx_key;
            if (carried && !ls.empty() && rows > 0) {
                const int64_t at = (int64_t) g.idx_key * ((boundary - boundary % g.idx_kpool) % ik_ring_);
                for (size_t i = 0; i < ls.size() && carried; ++i) {
                    float* ring[2] = {state_ + dsa_ik_[(size_t) ls[i]] + at, state_ + dsa_ig_[(size_t) ls[i]] + at};
                    for (int c = 0; c < 2 && carried; ++c)
                        carried = pkp.put(ring[c], rows * sizeof(float));
                }
            }
            if (carried) {
                // both streams carried: the runs' bytes = the staging's would-be
                // total (one run per recr layer), the pool rows' the same total
                // the staging computes below; the host vectors stay empty (the
                // pages are the only storage; the apply reads them first)
                int64_t n_recr = 0;
                for (int il = l0_; il < l1_; ++il) n_recr += g.is_recr(il) ? 1 : 0;
                td.snap_gpu.bytes = (int64_t) n_recr * (int64_t) run * sizeof(float);
                td.snap_pool_gpu.bytes = ls.empty() || rows <= 0
                    ? 0 : (int64_t) ls.size() * 2 * rows * sizeof(float);
                // Step 2's window diagnostic: which medium the boundary rode
                std::fprintf(stderr, "glm prefix: boundary %lld carried D2D (%d+%d pages, %d recr runs) (Task 3.3)\n",
                             (long long) boundary, (int) td.snap_gpu.pages.size(),
                             (int) td.snap_pool_gpu.pages.size(), (int) n_recr);
            } else {
                // the pool could not serve (exhausted or a copy failed): the
                // partial pages back, the host staging below covers the tile
                pk.drop();
                pkp.drop();
                // Step 2's window diagnostic: the fallback's medium, counted
                std::fprintf(stderr, "glm prefix: boundary %lld to host staging (the pool could not serve) (Task 3.3)\n",
                             (long long) boundary);
            }
        }
        if (ok && td.stateful && td.snap_gpu.pages.empty() && td.snap_pool_gpu.pages.empty()) {
            // the D6 fallback: the boundary state as host copies (the runs then
            // the pool rows, the same stream order the pages carry)
            const size_t run = (size_t) g.d_inner() * g.kda_head_dim + (size_t) 3 * g.d_inner() * (g.d_conv - 1);
            for (int il = l0_; il < l1_ && ok; ++il) {
                if (!g.is_recr(il)) continue;
                const size_t at = td.snap.size();
                td.snap.resize(at + run * sizeof(float));
                ok = cudaMemcpy(td.snap.data() + at, state_ + kda_S_[(size_t) il], run * sizeof(float),
                                cudaMemcpyDeviceToHost) == cudaSuccess;
            }
            const int64_t rows = (int64_t) (g.idx_kpool - 1) * g.idx_key;
            if (ok && !ls.empty() && rows > 0) {
                const int64_t at = (int64_t) g.idx_key * ((boundary - boundary % g.idx_kpool) % ik_ring_);
                for (size_t i = 0; i < ls.size() && ok; ++i) {
                    float* ring[2] = {state_ + dsa_ik_[(size_t) ls[i]] + at, state_ + dsa_ig_[(size_t) ls[i]] + at};
                    for (int c = 0; c < 2 && ok; ++c) {
                        const size_t off = td.snap_pool.size();
                        td.snap_pool.resize(off + rows * sizeof(float));
                        ok = cudaMemcpy(td.snap_pool.data() + off, ring[c], rows * sizeof(float),
                                        cudaMemcpyDeviceToHost) == cudaSuccess;
                    }
                }
            }
        }
        if (ok) {
            pd.tiles.push_back(std::move(td));   // the body already holds g_prefix_mu
        } else {
            for (int32_t pg : td.pages)
                if (pg >= 0) lat_pages_.page_unref(pg);   // the failed tile's pages back
        }
        t0 = t1;
    }
    // the pending chunk entry (the part index keys the share)
    PrefixPendingChunk& pc = g_prefix_pending[boundary];
    if (pc.parts.size() <= (size_t) part_) pc.parts.resize((size_t) part_ + 1);
    if (pc.present.size() <= (size_t) part_) pc.present.resize((size_t) part_ + 1, false);
    pc.parts[(size_t) part_] = std::move(pd);
    pc.present[(size_t) part_] = true;
    pc.chunk_start = chunk_start;
}

// ---- the finalize: the complete pending chunk entries become store entries
// (the hash + the registry).  Runs on the head, single-threaded; final = the
// prefill's last call (the incomplete stragglers get dropped there).
void Glm5Model::prefix_finalize(const std::vector<int32_t>& tokens, int64_t pos_start,
                                int64_t prompt_end, bool final) {
    (void) prompt_end;
    if (getenv("STRATA_GLM_NO_PREFIX") != nullptr) return;   // the reference window: nothing captures, nothing stores
    // the chain's parts, in split order
    std::vector<Glm5Model*> chain;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) chain.push_back(m);
    const int np = (int) chain.size();
    std::lock_guard<std::mutex> lk(g_prefix_mu);
    for (auto it = g_prefix_pending.begin(); it != g_prefix_pending.end(); ) {
        PrefixPendingChunk& pc = it->second;
        const int64_t chunk_end = it->first;
        bool complete = (int) pc.parts.size() >= np;
        for (int p = 0; complete && p < np; ++p) complete = pc.present[(size_t) p];
        if (!complete) {
            if (!final) { ++it; continue; }          // a lagging part's share may still land
            for (int p = 0; p < np && p < (int) pc.parts.size(); ++p)   // the failed part's pages back
                if (pc.present[(size_t) p])
                    for (const PrefixTileData& td : pc.parts[(size_t) p].tiles)
                        drop_tile_pages(chain[(size_t) p]->lat_pages_, td);
            it = g_prefix_pending.erase(it);
            continue;
        }
        // the tiles: the parts' lists agree (the same deterministic tiling)
        const size_t nt = pc.parts[0].tiles.size();
        for (size_t ti = 0; ti < nt; ++ti) {
            const PrefixTileData& t0t = pc.parts[0].tiles[ti];
            bool agree = true;
            for (int p = 1; p < np && agree; ++p) {
                const std::vector<PrefixTileData>& pts = pc.parts[(size_t) p].tiles;
                if (pts.size() != nt) { agree = false; break; }
                if (pts[ti].start != t0t.start || pts[ti].end != t0t.end ||
                    pts[ti].stateful != t0t.stateful) agree = false;
            }
            if (!agree) {   // a tiling disagreement: the tile's pages go back, the store skips it
                for (int p = 0; p < np; ++p)
                    if (ti < pc.parts[(size_t) p].tiles.size())
                        drop_tile_pages(chain[(size_t) p]->lat_pages_, pc.parts[(size_t) p].tiles[ti]);
                continue;
            }
            // the tile's tokens (this request's vector span)
            std::vector<int32_t> toks((size_t)(t0t.end - t0t.start));
            for (int64_t i = 0; i < t0t.end - t0t.start; ++i)
                toks[(size_t) i] = tokens[(size_t)(t0t.vec_start + i)];
            // the per-part row bytes (the hash's per-layer span)
            std::vector<int64_t> row_bytes;
            for (Glm5Model* m : chain) row_bytes.push_back(prefix_row_bytes(m));
            // the pages per part, in split order
            std::vector<std::vector<int32_t>> pages(np);
            for (int p = 0; p < np; ++p) {
                if (ti < pc.parts[(size_t) p].tiles.size())
                    pages[(size_t) p] = pc.parts[(size_t) p].tiles[ti].pages;
            }
            // the ID: the tokens + every layer's canonical bytes (the D2H staging)
            const strata::prefix::BlockId id = [&] {
                std::vector<std::vector<char>> buf;
                std::vector<const void*> ptrs;
                size_t page_bytes_hash = 0;
                cudaSetDevice(dev_);
                for (int p = 0; p < np; ++p) {
                    Glm5Model* m = chain[(size_t) p];
                    cudaSetDevice(m->dev_);
                    const int32_t page_stride = (int32_t) m->lat_pages_.page_bytes();
                    for (size_t li = 0; li < pages[(size_t) p].size(); ++li) {
                        const int32_t pg = pages[(size_t) p][li];
                        const size_t span = (size_t)(t0t.end - t0t.start) * (size_t) row_bytes[(size_t) p];
                        buf.emplace_back(span);
                        if (cudaMemcpy(buf.back().data(), (const char*) m->lat_pages_.slab_ptr()
                                            + (size_t) pg * page_stride, span,
                                        cudaMemcpyDeviceToHost) != cudaSuccess)
                            return strata::prefix::BlockId{0, 0};
                        ptrs.push_back(buf.back().data());
                        if (page_bytes_hash == 0) page_bytes_hash = span;
                    }
                }
                cudaSetDevice(dev_);
                return strata::prefix::hash_block(toks.data(), toks.size(), ptrs.data(), ptrs.size(), page_bytes_hash);
            }();
            if (id.hi == 0 && id.lo == 0) {   // the staging copy failed: the tile's pages go back
                for (int p = 0; p < np; ++p)
                    if (ti < pc.parts[(size_t) p].tiles.size())
                        drop_tile_pages(chain[(size_t) p]->lat_pages_, pc.parts[(size_t) p].tiles[ti]);
                continue;
            }
            // a duplicate tile (the same span + the same content): drop the new pages
            bool have = false;
            for (const PrefixBlock& b : g_prefix_blocks)
                if (b.start == t0t.start && b.boundary == t0t.end && b.id == id) { have = true; break; }
            if (have) {
                for (int p = 0; p < np; ++p)
                    if (ti < pc.parts[(size_t) p].tiles.size())
                        drop_tile_pages(chain[(size_t) p]->lat_pages_, pc.parts[(size_t) p].tiles[ti]);
                continue;
            }
            // the store's entry
            PrefixBlock blk;
            blk.id = id;
            blk.key = strata::prefix::BlockRegistry::token_key(toks.data(), toks.size());
            blk.start = t0t.start;
            blk.boundary = t0t.end;
            blk.stateful = t0t.stateful;
            blk.last_use = ++g_prefix_clock;
            blk.parts.resize((size_t) np);
            for (int p = 0; p < np; ++p) {
                PrefixTileData td;
                if (ti < pc.parts[(size_t) p].tiles.size()) {
                    const PrefixTileData& pt = pc.parts[(size_t) p].tiles[ti];
                    td.start = pt.start; td.end = pt.end; td.vec_start = pt.vec_start;
                    td.stateful = pt.stateful;
                    td.pages = pt.pages;
                    td.snap = pt.snap;
                    td.snap_pool = pt.snap_pool;
                    td.snap_gpu = pt.snap_gpu;          // Task 3.3: the D6 carriage's
                    td.snap_pool_gpu = pt.snap_pool_gpu;  // pages ride with the tile
                }
                blk.parts[(size_t) p].tiles.push_back(std::move(td));
            }
            // the store's pin, NOT a request's hold: the registration lands
            // refs 0 so the Step 3 eviction + the cap-drop can reclaim the
            // tile when no request holds it (an earlier run's
            // 0 evictions among 19 tiles showed the registration's own ref pinned
            // every tile, the capture's eviction helper found none evictable
            // and the capture refused).  A request's hold comes only from the
            // intake's acquire (prefix_refs_), released with the request.
            g_prefix_registry_.register_block(blk.id, blk.key);
            g_prefix_registry_.release(blk.id);   // refs 1 -> 0 (the store's pin)
            g_prefix_blocks.push_back(std::move(blk));
        }
        it = g_prefix_pending.erase(it);
    }
    // the store's cap: drop the least-recently-hit refcount-0 tiles
    const int cap = prefix_block_cap();
    std::fprintf(stderr, "glm prefix: the store holds %d tile(s) after this prefill (Task 3.2)\n",
                 (int) g_prefix_blocks.size());   // Step 2's window diagnostic
    while ((int) g_prefix_blocks.size() > cap) {
        int lru = -1;
        for (int i = 0; i < (int) g_prefix_blocks.size(); ++i) {
            const PrefixBlock& b = g_prefix_blocks[(size_t) i];
            if (g_prefix_registry_.refcount(b.id) != 0) continue;
            if (lru < 0 || b.last_use < g_prefix_blocks[(size_t) lru].last_use) lru = i;
        }
        if (lru < 0) break;   // every tile is live: the cap is soft
        const PrefixBlock& b = g_prefix_blocks[(size_t) lru];
        for (size_t p = 0; p < b.parts.size() && p < chain.size(); ++p) {   // the pages back to the pool
            Glm5Model* m = chain[p];
            cudaSetDevice(m->dev_);
            for (const PrefixTileData& td : b.parts[p].tiles)
                drop_tile_pages(m->lat_pages_, td);   // the canonical pages + the D6 carriage's
        }
        cudaSetDevice(dev_);
        g_prefix_registry_.forget(b.id);
        g_prefix_blocks.erase(g_prefix_blocks.begin() + (long) lru);
    }
}

// ---- the intake: the tile chain from 0, verified against the prompt's
// tokens; the hits refcounted.  Returns the positions the best stateful hit
// lets the prompt skip, 0 on a miss.  The verified chain lands in
// prefix_chain_ (the apply's walk) and the refs in prefix_refs_ (the
// release); the previous request's stragglers (a failed prefill's pending
// captures) drop their pages here.
size_t Glm5Model::prefix_intake(const std::vector<int32_t>& prompt) {
    // the pool off, or the reference path (no fast state): the incumbent path
    if (lat_pages_.n_pages() <= 0 || fast_ == nullptr) return 0;
    std::vector<Glm5Model*> chain;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) chain.push_back(m);
    size_t skip = 0;
    int64_t expect = 0;   // the next tile's prompt-relative start
    std::lock_guard<std::mutex> lk(g_prefix_mu);
    for (const auto& id : prefix_refs_) g_prefix_registry_.release(id);   // a stale request's refs
    prefix_refs_.clear();
    prefix_chain_.clear();
    // the previous request's stragglers: the present shares' pages back, the map cleared
    for (auto& kv : g_prefix_pending) {
        PrefixPendingChunk& pc = kv.second;
        for (size_t p = 0; p < pc.present.size() && p < chain.size(); ++p)
            if (pc.present[p])
                for (const PrefixTileData& td : pc.parts[p].tiles)
                    drop_tile_pages(chain[p]->lat_pages_, td);
    }
    g_prefix_pending.clear();
    for (;;) {
        // the tile at the expected boundary: the store may hold several blocks
        // with the same span (different conversations), so scan them all and
        // take the first whose tokens AND content match
        PrefixBlock* b = nullptr;
        for (PrefixBlock& c : g_prefix_blocks) {
            if (c.start != expect) continue;
            if (c.boundary > (int64_t) prompt.size()) continue;
            std::vector<int32_t> toks((size_t)(c.boundary - c.start));
            for (int64_t i = 0; i < c.boundary - c.start; ++i)
                toks[(size_t) i] = prompt[(size_t)(c.start + i)];
            if (strata::prefix::BlockRegistry::token_key(toks.data(), toks.size()) != c.key) continue;
            // the full-content verify: the canonical pages as they are now
            std::vector<int64_t> row_bytes;
            for (Glm5Model* m : chain) row_bytes.push_back(prefix_row_bytes(m));
            std::vector<std::vector<int32_t>> pages(chain.size());
            for (size_t p = 0; p < chain.size(); ++p)
                if (!c.parts[p].tiles.empty()) pages[p] = c.parts[p].tiles[0].pages;
            const strata::prefix::BlockId id = prefix_content_hash(toks, pages, chain,
                                                                   c.boundary - c.start, row_bytes);
            if (!(id == c.id)) continue;       // the bytes moved: the next candidate
            b = &c;
            break;
        }
        if (b == nullptr) break;               // no tile matches: the chain ends here
        // the hit: refcount the tile for the request
        g_prefix_registry_.acquire(b->id);
        prefix_refs_.push_back(b->id);
        prefix_chain_.push_back(b->id);
        b->last_use = ++g_prefix_clock;
        if (b->stateful) skip = (size_t) b->boundary;
        expect = b->boundary;
    }
    if (skip == 0) {   // no stateful tile in the chain: drop the refs again
        for (const auto& id : prefix_refs_) g_prefix_registry_.release(id);
        prefix_refs_.clear();
        prefix_chain_.clear();
    }
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) m->prefix_base_ = (int64_t) skip;
    return skip;
}

// ---- the apply: the hit's canonical state into the bound sequence's arena.
// The skipped span's rows copy back to their incumbent addresses (the
// identity table) and the stateful tile at the skip restores the registrar's
// boundary state (the KDA runs + the open key pool); the position becomes the
// skip.  The copies ride the part's compute stream (the next prefill's embed
// syncs it), exactly snapshot_restore's semantics for the boundary state.
// The walk is the intake's VERIFIED chain (prefix_chain_), so a foreign
// block's rows can never leak into the request.
bool Glm5Model::prefix_apply(const std::vector<int32_t>& prompt, size_t skip, std::string& err) {
    (void) prompt;
    if (lat_pages_.n_pages() <= 0) return true;   // the pool is off: nothing to apply
    std::vector<Glm5Model*> chain;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) chain.push_back(m);
    const int64_t end = (int64_t) skip;
    std::lock_guard<std::mutex> lk(g_prefix_mu);
    PrefixBlock* last = nullptr;
    bool failed = false;
    std::string fail_err;
    for (const auto& bid : prefix_chain_) {
        PrefixBlock* b = nullptr;   // the block by ID (the intake's verified chain)
        for (PrefixBlock& c : g_prefix_blocks)
            if (c.id == bid) { b = &c; break; }
        if (b == nullptr) {
            failed = true; fail_err = "glm prefix: the chain's tile left the store";
            break;
        }
        if (b->boundary > end) break;          // beyond the skip: the tail's prefill writes those
        // the rows: the block's canonical pages back into the arena rows
        for (size_t p = 0; p < chain.size() && p < b->parts.size() && !failed; ++p) {
            Glm5Model* m = chain[p];
            cudaSetDevice(m->dev_);
            const int64_t row_bytes = prefix_row_bytes(m);
            const int64_t lat_pp = row_bytes / 4;
            const std::vector<int> ls = m->dsa_layers();
            const PrefixTileData& td = b->parts[p].tiles[0];
            if (td.pages.size() != ls.size()) {
                failed = true; fail_err = "glm prefix: the tile's pages ran short";
                break;
            }
            for (size_t li = 0; li < ls.size(); ++li) {
                const int32_t pg = td.pages[li];
                if (pg < 0 || !m->lat_pages_.page_allocated(pg)) {   // the canonical rows died
                    failed = true; fail_err = "glm prefix: the tile's rows were evicted";
                    break;
                }
                float* dst = m->state_ + m->dsa_lat_[(size_t) ls[li]] + (size_t) b->start * lat_pp;
                const uint8_t* src = (const uint8_t*) m->lat_pages_.slab_ptr()
                                     + (size_t) pg * (size_t) m->lat_pages_.page_bytes();
                const size_t bytes = (size_t)(td.end - td.start) * row_bytes;
                if (cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, m->fast_->cs) != cudaSuccess) {
                    failed = true; fail_err = "glm prefix: the rows' copy failed";
                    break;
                }
            }
        }
        if (failed) break;
        if (b->boundary == end && b->stateful) last = b;   // the stateful tile at the skip
        if (b->boundary >= end) break;         // the chain reached the skip
    }
    if (failed) { err = fail_err; return false; }
    if (last == nullptr) { err = "glm prefix: no stateful tile at the skip"; return false; }
    // the boundary state: the stateful tile at the skip (the registrar's own
    // state there; the identity of the arena layout makes the restore exact)
    for (size_t p = 0; p < chain.size() && p < last->parts.size(); ++p) {
        Glm5Model* m = chain[p];
        cudaSetDevice(m->dev_);
        const Glm5Geometry& g = m->g_;
        const PrefixTileData& td = last->parts[p].tiles[0];
        const size_t run = (size_t) g.d_inner() * g.kda_head_dim + (size_t) 3 * g.d_inner() * (g.d_conv - 1);
        // Task 3.3: the D6 carriage's pages first (D2D on the same stream, the
        // store's refs hold the pages against the sweep); the host staging is
        // the fallback.  The offsets are identical on both mediums.
        size_t at = 0;
        for (int il = m->l0_; il < m->l1_; ++il) {
            if (!g.is_recr(il)) continue;
            if (td.snap_gpu.live()) {
                if (at + (int64_t) run * sizeof(float) > td.snap_gpu.bytes ||
                    !snap_unpack(m->lat_pages_, td.snap_gpu.pages, (int64_t) at,
                                 (int64_t) run * sizeof(float),
                                 m->state_ + m->kda_S_[(size_t) il], m->fast_->cs)) {
                    err = "glm prefix: the snapshot ran short";
                    return false;
                }
            } else {
                if (at + run * sizeof(float) > td.snap.size()) { err = "glm prefix: the snapshot ran short"; return false; }
                if (cudaMemcpyAsync(m->state_ + m->kda_S_[(size_t) il], td.snap.data() + at, run * sizeof(float),
                                    cudaMemcpyHostToDevice, m->fast_->cs) != cudaSuccess) {
                    err = "glm prefix: the KDA restore failed";
                    return false;
                }
            }
            at += run * sizeof(float);
        }
        const int64_t rows = (int64_t) (g.idx_kpool - 1) * g.idx_key;
        const std::vector<int> ls = m->dsa_layers();
        if (!ls.empty() && rows > 0) {
            const int64_t at2 = (int64_t) g.idx_key * (((int64_t) skip - ((int64_t) skip % g.idx_kpool)) % m->ik_ring_);
            for (size_t i = 0; i < ls.size(); ++i) {
                float* ring[2] = {m->state_ + m->dsa_ik_[(size_t) ls[i]] + at2,
                                  m->state_ + m->dsa_ig_[(size_t) ls[i]] + at2};
                for (int c = 0; c < 2; ++c) {
                    const int64_t off = ((i * 2 + (size_t) c) * (size_t) rows) * sizeof(float);
                    if (td.snap_pool_gpu.live()) {
                        if (off + (int64_t) rows * sizeof(float) > td.snap_pool_gpu.bytes ||
                            !snap_unpack(m->lat_pages_, td.snap_pool_gpu.pages, off,
                                         (int64_t) rows * sizeof(float), ring[c], m->fast_->cs)) {
                            err = "glm prefix: the pool rows' restore failed";
                            return false;
                        }
                    } else if (td.snap_pool.size() >= 2 * ls.size() * (size_t) rows * sizeof(float)) {
                        if (cudaMemcpyAsync(ring[c], td.snap_pool.data() + off, rows * sizeof(float),
                                            cudaMemcpyHostToDevice, m->fast_->cs) != cudaSuccess) {
                            err = "glm prefix: the pool rows' restore failed";
                            return false;
                        }
                    }
                }
            }
        }
    }
    // the boundary position on every part (snapshot_restore's semantics) + the
    // drafter's positions from here on belong to the previous continuation
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) m->pos_ = end;
    cudaSetDevice(dev_);
    if (cap_d_ != nullptr) cap_d_->reset(end);
    return true;
}

// ---- the register: the tiles' shares land in the pending entries as the
// prefill chunks end (prefix_capture_part); the finalize turns them into
// store entries.  Kept for the hpp's declaration symmetry - the prefill's
// chunk spots call the capture + the finalize directly.
bool Glm5Model::prefix_register(int64_t start_pos, const std::vector<int32_t>& tail, std::string& err) {
    (void) start_pos; (void) tail; (void) err;
    return true;
}

// ---- the release: the request's refcounts (the parked slot's pages stay
// canonical until the last holder lets go).
void Glm5Model::prefix_release() {
    std::lock_guard<std::mutex> lk(g_prefix_mu);
    for (const auto& id : prefix_refs_) g_prefix_registry_.release(id);
    prefix_refs_.clear();
}

// Task 4.1: the ladder's pool-pages rung - every refcount-0 page back under the capture's
// allocator lock (the same invariant page_alloc's pressure sweep runs under: the lock guards
// both the refcounts and the capture's page allocations).  The answer is the pages swept; the
// pool off (STRATA_GLM_PAGED unset or init failed) is a clean 0.
int Glm5Model::pages_sweep_idle() {
    if (lat_pages_.n_pages() <= 0) return 0;
    std::lock_guard<std::mutex> lk(g_prefix_mu);
    const int32_t before = lat_pages_.counters().live;
    const int32_t swept = lat_pages_.evict_lru(lat_pages_.n_pages());
    if (swept > 0)
        std::fprintf(stderr, "glm pages: swept %d idle page(s) back (%d live -> %d)\n", swept,
                     before, lat_pages_.counters().live);
    return swept;
}

// ================= Task 4.2: the PAUSE path's registry-owned snapshots =================
//
// PAUSE captures the bound slot's full decode state into the pool's refcounted
// pages (the Task 3.3 carriage's framing), so a paused sequence's pages survive
// every pressure sweep and the slot's park; RESUME unpacks the capture into the
// fresh arena when the new prompt extends the capture's token span.  The
// incumbent snap_/snap_pool_ pair stays the registry-off fallback (the pool off
// or STRATA_GLM_NO_PREFIX: the capture is refused, the pair carries).  Under
// pressure the tile store's unheld (registry refcount 0) LRU tiles evict first
// (Task 4.2 step 3); a still-short pool refuses cleanly, with numbers.
//
// THE FLAT STREAM (one framing, mirrored by the capture, the restore and the
// selftest's byte-compare lambda), per part in chain order:
//   1. the KDA runs in recr-layer order: run floats from state_ + kda_S_[il]
//   2. the open ik/ig: per DSA layer, 2 rings x rows floats at the capture's
//      ring offset (the same snap_pool_copy/slot_save arithmetic)
//   3. the pooled sums: per DSA layer, pools x idx_key floats from dsa_pool_
//   4. the latent rows [0, pos): per DSA layer, logical token order, read
//      through the bound table at the layout's row width (lat_row_offset)
//   5. the table mode: 1 int32 (0 = NULL rows, 1 = the identity table)
//
// THE ROW WIDTH is the seq layout's, not prefix_row_bytes' (the fast path's
// FP16 tile width): lat_q8 -> lat8_rec_bytes(kv_lora), the fast mode kv_lora*2
// (FP16), the slow path kv_lora*4 (FP32) - slot_dsa_runs' lat_pp * 4.
//
// THE RESTORE'S LAYOUT GUARD: an entry restores only into an arena whose
// ik_ring and table mode match the capture's (the mode rides the stream's
// tail); a mismatch skips the entry (the incumbent pair carries the state).

// ---- Task 4.2 step 3: the capture's room answer (the vLLM/SGLang shape):
// the tile store's refcount-0 entries are reclaimable cache exactly like
// vLLM's cached-block free list (an allocation pops a cached block the
// moment it takes it: _maybe_evict_cached_block, vllm/v1/core/block_pool.py)
// and SGLang's evict_to_free_tokens (the fit check evicts BEFORE it answers,
// sglang/srt/mem_cache/allocator/base.py).  The caller holds g_prefix_mu and
// has synced the compute stream, so the tile pages' pending copies settle
// before the packer rewrites the freed pages (the packer's cudaMemcpy rides
// the legacy NULL stream, which serializes with the blocking compute
// stream).  Whole tiles drop, LRU-first, until the part's shortfall is
// covered; a tile with no pages in this part's pool cannot help and stays;
// a tile a request holds (registry refcount != 0) never drops.  The answer
// is the pages freed (the drops stand even when the capture still refuses:
// the freed pages serve the next capture).
int64_t Glm5Model::pause_evict_tiles_for(const std::vector<Glm5Model*>& chain,
                                         size_t part, int64_t shortfall) {
    if (shortfall <= 0 || part >= chain.size()) return 0;
    int64_t freed = 0;
    for (;;) {
        // the least-recently-used evictable candidate (the cap-drop's order
        // and predicate: last_use ascending, registry refcount 0)
        size_t best = (size_t) -1;
        int64_t best_pages = 0;
        for (size_t i = 0; i < g_prefix_blocks.size(); ++i) {
            const PrefixBlock& b = g_prefix_blocks[i];
            if (g_prefix_registry_.refcount(b.id) != 0) continue;  // a request holds it
            if (part >= b.parts.size()) continue;
            int64_t pages_here = 0;
            for (const PrefixTileData& td : b.parts[part].tiles)
                pages_here += (int64_t) td.pages.size()
                              + (int64_t) td.snap_gpu.pages.size()
                              + (int64_t) td.snap_pool_gpu.pages.size();
            if (pages_here <= 0) continue;   // no pages in this part's pool
            if (best == (size_t) -1 || b.last_use < g_prefix_blocks[best].last_use) {
                best = i;
                best_pages = pages_here;
            }
        }
        if (best == (size_t) -1) break;   // nothing evictable left: the caller refuses
        const PrefixBlock& b = g_prefix_blocks[best];
        for (size_t p = 0; p < b.parts.size() && p < chain.size(); ++p) {
            cudaSetDevice(chain[p]->dev_);
            for (const PrefixTileData& td : b.parts[p].tiles)
                drop_tile_pages(chain[p]->lat_pages_, td);   // the canonical pages + the D6 carriage's
        }
        cudaSetDevice(chain[part]->dev_);
        g_prefix_registry_.forget(b.id);
        g_prefix_blocks.erase(g_prefix_blocks.begin() + (long) best);
        std::fprintf(stderr, "glm pause: evicted %lld tile-store page(s) for the capture (LRU; the store holds %zu tile(s))\n",
                     (long long) best_pages, g_prefix_blocks.size());
        freed += best_pages;
        if (freed >= shortfall) break;   // the shortfall is covered: no further drops
    }
    return freed;
}

bool Glm5Model::pause_capture(const int32_t* tokens, int n) {
    // the registry off: the pool off or the reference window (the incumbent
    // snap_/snap_pool_ pair carries; the capture is refused, never fails PAUSE)
    if (lat_pages_.n_pages() <= 0 || getenv("STRATA_GLM_NO_PREFIX") != nullptr) return false;
    std::vector<Glm5Model*> chain;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) chain.push_back(m);
    const int64_t pos = pos_;
    const int64_t row_bytes = lat_q8_ ? (int64_t) gf::lat8_rec_bytes(g_.kv_lora)
                              : fast_mode_ ? (int64_t) g_.kv_lora * 2
                                           : (int64_t) g_.kv_lora * 4;
    const int64_t ls0_n = (int64_t) dsa_layers().size();
    (void) ls0_n;   // (the per-part lambda computes its own count; kept for the comment's shape)
    // the stream's size per part (the pool's need): 1 + the KDA runs + the open
    // rings + the pooled sums + the latent rows, in bytes
    const auto part_bytes = [&](const Glm5Model* m) -> int64_t {
        const Glm5Geometry& g = m->g_;
        const int64_t run = (int64_t) g.d_inner() * g.kda_head_dim
                            + 3 * (int64_t) g.d_inner() * (g.d_conv - 1);
        int n_rec = 0;
        for (int il = m->l0_; il < m->l1_; ++il) n_rec += g.is_recr(il);
        const int64_t ls_n = (int64_t) m->dsa_layers().size();
        const int64_t rows = (int64_t)(g.idx_kpool - 1) * g.idx_key;
        const int64_t pools = (pos + g.idx_kpool - 1) / g.idx_kpool;
        return (int64_t) sizeof(int32_t) + (int64_t) n_rec * run * sizeof(float)
               + ls_n * 2 * rows * sizeof(float)
               + ls_n * pools * g.idx_key * sizeof(float)
               + ls_n * pos * row_bytes;
    };
    // the pages' need per part (the packer allocates from that part's own
    // pool): the unwinnable cut first - every pool drained + every tile
    // dropped still short - then the room check under the lock (Task 4.2
    // step 3).  The capture is best-effort: the pair carries when the pool
    // cannot serve.
    std::vector<int64_t> need_p(chain.size());
    int64_t need = 0, total = 0;
    for (size_t pc = 0; pc < chain.size(); ++pc) {
        need_p[pc] = (part_bytes(chain[pc]) + chain[pc]->lat_pages_.page_bytes() - 1)
                     / chain[pc]->lat_pages_.page_bytes();
        need += need_p[pc];
        total += chain[pc]->lat_pages_.n_pages();
    }
    if (need > total) {
        std::fprintf(stderr, "glm pause: the capture needs %lld pool page(s), the pools hold %lld in all - the pair carries\n",
                     (long long) need, (long long) total);
        return false;
    }
    std::lock_guard<std::mutex> lk(g_prefix_mu);
    // the compute stream's writes settle first (snapshot_save already synced it
    // on the PAUSE path; the slow path's work is synchronous)
    if (fast_ != nullptr && cudaStreamSynchronize(fast_->cs) != cudaSuccess) return false;
    cudaSetDevice(dev_);
    // the room check + the evict-then-retry (Task 4.2 step 3): the packer's
    // serveable pages are the free tail plus the idle pages its own pressure
    // sweep converts; the tile store's refcount-0 entries (cached prefixes no
    // request holds) are reclaimable cache (vLLM's cached-block free list,
    // SGLang's evict_to_free_tokens) and cover the rest, LRU-first.  A
    // still-short pool refuses cleanly, with numbers, BEFORE the packer
    // starts (the window's mid-pack abort was this check's absence); the
    // drops stand - the freed pages serve the next capture.
    for (size_t pc = 0; pc < chain.size(); ++pc) {
        const GlmPages& pool = chain[pc]->lat_pages_;
        const int32_t np = pool.n_pages();
        int64_t have = (int64_t) np - pool.counters().live;   // the free tail
        for (int32_t i = 0; i < np; ++i)
            if (pool.page_allocated(i) && pool.refcount(i) == 0) ++have;  // the packer's sweep serves these
        if (have >= need_p[pc]) continue;
        const int64_t freed = pause_evict_tiles_for(chain, pc, need_p[pc] - have);
        have += freed;
        if (have >= need_p[pc]) continue;
        std::fprintf(stderr, "glm pause: the capture needs %lld page(s) on part %zu, the pool serves %lld after %lld evicted - the pair carries\n",
                     (long long) need_p[pc], pc, (long long) have, (long long) freed);
        cudaSetDevice(dev_);
        return false;
    }
    PauseSnap snap;
    snap.pos = pos;
    snap.owner = this;   // the pages belong to this engine's slab
    snap.span.assign(tokens, tokens + n);
    snap.ring.resize(chain.size());
    snap.gpu.resize(chain.size());
    bool ok = true;
    const char* fail_why = "";
    for (size_t p = 0; p < chain.size() && ok; ++p) {
        Glm5Model* m = chain[p];
        cudaSetDevice(m->dev_);
        snap.ring[p] = (int) m->ik_ring_;
        const Glm5Geometry& g = m->g_;
        const int64_t run = (int64_t) g.d_inner() * g.kda_head_dim
                            + 3 * (int64_t) g.d_inner() * (g.d_conv - 1);
        const std::vector<int> ls = m->dsa_layers();
        const int64_t rows = (int64_t)(g.idx_kpool - 1) * g.idx_key;
        const int64_t pools = (pos + g.idx_kpool - 1) / g.idx_kpool;
        const int64_t at_ring = (int64_t) g.idx_key * ((pos - pos % g.idx_kpool) % m->ik_ring_);
        const int64_t m_row_bytes = m->lat_q8_ ? (int64_t) gf::lat8_rec_bytes(m->g_.kv_lora)
                                    : m->fast_mode_ ? (int64_t) m->g_.kv_lora * 2
                                                    : (int64_t) m->g_.kv_lora * 4;
        // the rows' addresses: the host computes them ONLY for a null or identity
        // table (lat_row_offset's identity math); a shuffled table (P3's reorders,
        // not landed) is the capture's refusal - the incumbent pair carries
        if (m->lat_table_ != nullptr && !m->lat_pages_.table_identity()) { ok = false; fail_why = "the table is shuffled"; }
        SnapPacker packer{m->lat_pages_, snap.gpu[p]};
        // 1. the KDA runs
        for (int il = m->l0_; il < m->l1_ && ok; ++il)
            if (g.is_recr(il))
                ok = packer.put(m->state_ + m->kda_S_[(size_t) il], run * sizeof(float));
        // 2. the open ik/ig rings
        for (size_t i = 0; i < ls.size() && ok; ++i)
            for (int c = 0; c < 2 && ok; ++c)
                ok = packer.put(m->state_ + (c ? m->dsa_ig_[(size_t) ls[i]] : m->dsa_ik_[(size_t) ls[i]]) + at_ring,
                                rows * sizeof(float));
        // 3. the pooled sums
        for (size_t i = 0; i < ls.size() && ok; ++i)
            ok = packer.put(m->state_ + m->dsa_pool_[(size_t) ls[i]], pools * g.idx_key * sizeof(float));
        // 4. the latent rows [0, pos): per DSA layer, logical token order - the
        //    identity table's addresses are the absolute rows (the same ones the
        //    batched kernels wrote)
        for (size_t i = 0; i < ls.size() && ok; ++i)
            for (int64_t t = 0; t < pos && ok; ++t)
                ok = packer.put((const uint8_t*)(m->state_ + m->dsa_lat_[(size_t) ls[i]])
                                    + (size_t) t * (size_t) m_row_bytes, m_row_bytes);
        // 5. the table mode
        if (ok) {
            const int32_t mode = m->lat_table_ == nullptr ? 0
                                 : (m->lat_pages_.table_identity() ? 1 : 2);
            ok = packer.put(&mode, sizeof(mode));
        }
        if (ok) snap.gpu[p].bytes = part_bytes(m);   // the stream's total (the restore's bounds check)
        if (!ok) { fail_why = "the pack ran short"; break; }
    }
    if (!ok) {
        std::fprintf(stderr, "glm pause: the capture failed (%s) - the pair carries\n", fail_why);
        for (size_t p = 0; p < snap.gpu.size(); ++p)
            for (int32_t pg : snap.gpu[p].pages)
                if (pg >= 0) chain[p]->lat_pages_.page_unref(pg);
        cudaSetDevice(dev_);
        return false;
    }
    cudaSetDevice(dev_);
    // (the pages' pins are the packer's page_ref: the sweep cannot take them.
    //  The PREFIX registry stays out of this: its slot indexes g_prefix_blocks,
    //  which the pause store never fills - the intake's lookup would read a
    //  wrong tile.)
    // the duplicate capture of the same conversation: the older entry's pages
    // drop, the newer carries (same span, same content; this engine's entries
    // only - the store is process-wide)
    for (size_t e = g_pause_snaps.size(); e-- > 0;) {
        const PauseSnap& old = g_pause_snaps[e];
        if (old.owner != this) continue;
        if (old.span.size() == snap.span.size() && old.span == snap.span) {
            for (size_t p = 0; p < old.gpu.size(); ++p)
                for (int32_t pg : old.gpu[p].pages)
                    if (pg >= 0) chain[p]->lat_pages_.page_unref(pg);
            g_pause_snaps.erase(g_pause_snaps.begin() + (long) e);
        }
    }
    snap.used = ++g_prefix_clock;
    g_pause_snaps.push_back(std::move(snap));
    // the LRU cap: the least-recently-used refcount-0 entries drop (a
    // conversation that never resumes must not hold pages forever; a live
    // refcount pins its entry)
    while ((int) g_pause_snaps.size() > pause_snap_cap()) {
        // every stored entry is idle (the resume consumes its entry; the
        // registry's refs are the prefix store's bookkeeping, not this store's);
        // this engine's entries only (the store is process-wide)
        int lru = -1;
        for (int e = 0; e < (int) g_pause_snaps.size(); ++e) {
            if (g_pause_snaps[(size_t) e].owner != this) continue;
            if (lru < 0 || g_pause_snaps[(size_t) e].used < g_pause_snaps[(size_t) lru].used) lru = e;
        }
        if (lru < 0) break;
        PauseSnap& old = g_pause_snaps[(size_t) lru];
        for (size_t p = 0; p < old.gpu.size(); ++p)
            for (int32_t pg : old.gpu[p].pages)
                if (pg >= 0) chain[p]->lat_pages_.page_unref(pg);
        g_pause_snaps.erase(g_pause_snaps.begin() + (long) lru);
    }
    {
        size_t pages_total = 0;
        for (const auto& sp : g_pause_snaps.back().gpu) pages_total += sp.pages.size();
        std::fprintf(stderr, "glm pause: the store holds %d capture(s) (%zu page(s), pos %lld)\n",
                     (int) g_pause_snaps.size(), pages_total, (long long) pos);
    }
    return true;
}

int64_t Glm5Model::pause_resume(const std::vector<int32_t>& prompt, std::vector<int32_t>& span_out) {
    span_out.clear();
    if (lat_pages_.n_pages() <= 0 || getenv("STRATA_GLM_NO_PREFIX") != nullptr) return 0;
    std::vector<Glm5Model*> chain;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) chain.push_back(m);
    std::lock_guard<std::mutex> lk(g_prefix_mu);
    // the longest span that is a strict prefix of the prompt (the incumbent's
    // reuse gate: a resumed prompt reads only what is new)
    int best_idx = -1;
    for (int e = 0; e < (int) g_pause_snaps.size(); ++e) {
        const PauseSnap& en = g_pause_snaps[(size_t) e];
        if (en.owner != this) continue;                    // the pages belong to this engine
        if (en.pos != (int64_t) en.span.size()) continue;           // the invariant
        if (en.span.size() >= prompt.size()) continue;              // strict
        if (!std::equal(en.span.begin(), en.span.end(), prompt.begin())) continue;
        // the layout guard: every part's ring matches this engine's
        bool fits = en.ring.size() == chain.size();
        for (size_t p = 0; fits && p < chain.size(); ++p)
            if (en.ring[p] != (int) chain[p]->ik_ring_) fits = false;
        if (!fits) continue;
        if (best_idx < 0 || en.span.size() > g_pause_snaps[(size_t) best_idx].span.size()) best_idx = e;
    }
    if (best_idx < 0) return 0;
    // the entry leaves the store: its pages' pins move to the restore (the
    // unpack reads the pages, then every page unrefs)
    PauseSnap snap = std::move(g_pause_snaps[(size_t) best_idx]);
    g_pause_snaps.erase(g_pause_snaps.begin() + (long) best_idx);
    cudaSetDevice(dev_);
    bool ok = true;
    const char* fail_why = "";
    for (size_t p = 0; p < chain.size() && ok; ++p) {
        Glm5Model* m = chain[p];
        cudaSetDevice(m->dev_);
        const Glm5Geometry& g = m->g_;
        const int64_t run = (int64_t) g.d_inner() * g.kda_head_dim
                            + 3 * (int64_t) g.d_inner() * (g.d_conv - 1);
        const std::vector<int> ls = m->dsa_layers();
        const int64_t rows = (int64_t)(g.idx_kpool - 1) * g.idx_key;
        const int64_t pools = (snap.pos + g.idx_kpool - 1) / g.idx_kpool;
        const int64_t at_ring = (int64_t) g.idx_key * ((snap.pos - snap.pos % g.idx_kpool) % m->ik_ring_);
        const int64_t m_row_bytes = m->lat_q8_ ? (int64_t) gf::lat8_rec_bytes(m->g_.kv_lora)
                                    : m->fast_mode_ ? (int64_t) m->g_.kv_lora * 2
                                                    : (int64_t) m->g_.kv_lora * 4;
        const cudaStream_t cs = m->fast_ ? m->fast_->cs : (cudaStream_t) nullptr;
        strata::prefix::SnapPages& sp = snap.gpu[p];
        int64_t off = 0;
        auto unpack = [&](void* dst, int64_t bytes) -> bool {
            if (off + bytes > sp.bytes || !snap_unpack(m->lat_pages_, sp.pages, off, bytes, dst, cs))
                return false;
            off += bytes;
            return true;
        };
        // the rows' addresses: identity-only here too (the mode guard below pins
        // the table's shape; a shuffled table skips the entry, the pair carries)
        if (m->lat_table_ != nullptr && !m->lat_pages_.table_identity()) { ok = false; fail_why = "the table is shuffled"; }
        // 1. the KDA runs into the state (the S states + the conv histories, one span)
        for (int il = m->l0_; il < m->l1_ && ok; ++il)
            if (g.is_recr(il)) ok = unpack(m->state_ + m->kda_S_[(size_t) il], run * sizeof(float));
        // 2. the open ik/ig rings
        for (size_t i = 0; i < ls.size() && ok; ++i)
            for (int c = 0; c < 2 && ok; ++c)
                ok = unpack(m->state_ + (c ? m->dsa_ig_[(size_t) ls[i]] : m->dsa_ik_[(size_t) ls[i]]) + at_ring,
                            rows * sizeof(float));
        // 3. the pooled sums
        for (size_t i = 0; i < ls.size() && ok; ++i)
            ok = unpack(m->state_ + m->dsa_pool_[(size_t) ls[i]], pools * g.idx_key * sizeof(float));
        // 4. the latent rows: the identity table's addresses are the absolute
        //    rows (the capture's logical order lands at the same physical rows)
        for (size_t i = 0; i < ls.size() && ok; ++i)
            for (int64_t t = 0; t < snap.pos && ok; ++t)
                ok = unpack((uint8_t*)(m->state_ + m->dsa_lat_[(size_t) ls[i]])
                                + (size_t) t * (size_t) m_row_bytes, m_row_bytes);
        // 5. the table mode: it must match the current arena's (a NULL-rows
        //    restore into a tabled arena would write the rows to the wrong
        //    addresses; the mismatch skips the entry, the incumbent carries)
        if (ok) {
            int32_t mode = 0;
            if (unpack(&mode, sizeof(mode))) {
                const int32_t cur = m->lat_table_ == nullptr ? 0
                                    : (m->lat_pages_.table_identity() ? 1 : 2);
                if (mode != cur) { ok = false; fail_why = "the table mode differs"; }
            } else { ok = false; fail_why = "the stream ran short"; }
        }
        if (ok && m->fast_) {   // the copies settle before anything reads them
            if (cudaStreamSynchronize(m->fast_->cs) != cudaSuccess) { ok = false; fail_why = "the restore did not sync"; }
        }
        if (!ok) break;
    }
    if (!ok) {
        std::fprintf(stderr, "glm pause: the restore failed (%s) - the incumbent path carries\n", fail_why);
        for (size_t p = 0; p < snap.gpu.size(); ++p)
            for (int32_t pg : snap.gpu[p].pages)
                if (pg >= 0) chain[p]->lat_pages_.page_unref(pg);
        cudaSetDevice(dev_);
        return 0;
    }
    // the restore is done: the pages' pins drop (the pool takes them back; the
    // unpack's copies are ordered on the bound slot's stream before anything
    // re-allocates the page, and the next capture syncs the stream first)
    for (size_t p = 0; p < snap.gpu.size(); ++p)
        for (int32_t pg : snap.gpu[p].pages)
            if (pg >= 0) chain[p]->lat_pages_.page_unref(pg);
    // the restore's semantics = snapshot_restore's: the position on every part
    // + the drafter's positions from here on belong to the previous continuation
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) m->pos_ = snap.pos;
    cudaSetDevice(dev_);
    if (cap_d_ != nullptr) cap_d_->reset(snap.pos);
    span_out = std::move(snap.span);
    std::fprintf(stderr, "glm pause: resumed %zu of %zu prompt tokens from the registry capture\n",
                 span_out.size(), prompt.size());
    return (int64_t) span_out.size();
}

void Glm5Model::pause_release(const std::vector<int32_t>& span) {
    if (span.empty()) return;
    std::lock_guard<std::mutex> lk(g_prefix_mu);
    for (size_t e = g_pause_snaps.size(); e-- > 0;) {
        PauseSnap& s = g_pause_snaps[e];
        if (s.owner != this || s.span != span) continue;   // this engine's entries only
        std::vector<Glm5Model*> chain;
        for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) chain.push_back(m);
        for (size_t p = 0; p < s.gpu.size(); ++p)
            for (int32_t pg : s.gpu[p].pages)
                if (pg >= 0) chain[p]->lat_pages_.page_unref(pg);
        g_pause_snaps.erase(g_pause_snaps.begin() + (long) e);
        std::fprintf(stderr, "glm pause: released a %zu-token capture\n", span.size());
        return;   // one entry per conversation
    }
}

// Task 4.2: the destructor's cleanup - every entry this engine owns drops (the
// pages are this instance's slab; a destroyed model's pages must never be
// reached).  The driver holds one engine per process, so production drops at
// most its own entries; the selftest's harnesses drop between scopes.  Each
// part's pages live in that part's own lat_pages_ (the capture walked the
// split), so the chain's pools unref them.
void Glm5Model::pause_release_all() {
    if (g_pause_snaps.empty()) return;   // the common case: no lock
    std::lock_guard<std::mutex> lk(g_prefix_mu);
    std::vector<Glm5Model*> chain;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) chain.push_back(m);
    for (size_t e = g_pause_snaps.size(); e-- > 0;) {
        const PauseSnap& s = g_pause_snaps[e];
        if (s.owner != this) continue;
        for (size_t p = 0; p < s.gpu.size() && p < chain.size(); ++p)
            for (int32_t pg : s.gpu[p].pages)
                if (pg >= 0) chain[p]->lat_pages_.page_unref(pg);
        g_pause_snaps.erase(g_pause_snaps.begin() + (long) e);
    }
    cudaSetDevice(dev_);
}

// the selftest's byte-compare helper: the restore-owned regions D2H, in the
// capture's framing order, walking the split's parts (each part holds its own
// regions).  Full private access (the glm_pages.cu selftest pattern).
std::vector<char> Glm5Model::pause_state_bytes(const Glm5Model& m) {
    std::vector<char> out;
    auto take = [&](const void* src, int64_t bytes) {
        if (bytes <= 0) return;                     // a negative/nonsense size never resizes
        const size_t at = out.size();
        out.resize(at + (size_t) bytes);
        if (cudaMemcpy(out.data() + at, src, (size_t) bytes,
                       cudaMemcpyDeviceToHost) != cudaSuccess)
            std::fprintf(stderr, "glm pause: the selftest's state copy failed\n");
    };
    for (const Glm5Model* p = &m; p != nullptr; p = p->split_next_.get()) {
        cudaSetDevice(p->dev_);
        const Glm5Geometry& g = p->g_;
        const int64_t run = (int64_t) g.d_inner() * g.kda_head_dim
                            + 3 * (int64_t) g.d_inner() * (g.d_conv - 1);
        const std::vector<int> ls = p->dsa_layers();
        const int64_t rows = (int64_t)(g.idx_kpool - 1) * g.idx_key;
        const int64_t pos = p->position();
        const int64_t pools = (pos + g.idx_kpool - 1) / g.idx_kpool;
        const int64_t at_ring = (int64_t) g.idx_key * ((pos - pos % g.idx_kpool) % p->ik_ring_);
        const int64_t row_bytes = p->lat_q8_ ? (int64_t) gf::lat8_rec_bytes(p->g_.kv_lora)
                                  : p->fast_mode_ ? (int64_t) p->g_.kv_lora * 2
                                                  : (int64_t) p->g_.kv_lora * 4;
        // 1. the KDA runs (the S states + the conv histories, contiguous spans)
        for (int il = p->l0_; il < p->l1_; ++il)
            if (g.is_recr(il)) take(p->state_ + p->kda_S_[(size_t) il], run * sizeof(float));
        // 2. the open ik/ig rings at the position's ring offset
        for (size_t i = 0; i < ls.size(); ++i)
            for (int c = 0; c < 2; ++c)
                take(p->state_ + (c ? p->dsa_ig_[(size_t) ls[i]] : p->dsa_ik_[(size_t) ls[i]]) + at_ring,
                     rows * sizeof(float));
        // 3. the pooled sums
        for (size_t i = 0; i < ls.size(); ++i)
            take(p->state_ + p->dsa_pool_[(size_t) ls[i]], pools * g.idx_key * sizeof(float));
        // 4. the latent rows [0, pos): the identity table's absolute rows (the
        //    slow path's FP32 rows; the harness runs STRATA_GLM_SLOW)
        for (size_t i = 0; i < ls.size(); ++i)
            for (int64_t t = 0; t < pos; ++t)
                take((const uint8_t*)(p->state_ + p->dsa_lat_[(size_t) ls[i]])
                         + (size_t) t * (size_t) row_bytes, row_bytes);
    }
    cudaSetDevice(m.dev_);
    return out;
}

// the selftest (glm_seq_dynamic_test's Task 4.2 legs; the pack path comes from
// the test's harness): the reference drive, the registry capture + pressure +
// park + byte-identical resume, the dedupe + the cap, and the registry-off
// fallback legs (the capture refuses; the incumbent pair carries).
bool Glm5Model::glm_pause_snapshots_selftest(const std::string& pack, int* checks, int* failures) {
    auto req = [&](const char* name, bool ok) {
        ++*checks;
        printf(ok ? "ok: %s\n" : "FAIL: %s\n", name);
        if (!ok) ++*failures;
    };
    const char* seqs_env = getenv("STRATA_GLM_SEQS");
    const char* seqctx_env = getenv("STRATA_GLM_SEQ_CTX");
    auto restore_envs = [&]() {
        if (seqs_env) setenv("STRATA_GLM_SEQS", seqs_env, 1); else unsetenv("STRATA_GLM_SEQS");
        if (seqctx_env) setenv("STRATA_GLM_SEQ_CTX", seqctx_env, 1); else unsetenv("STRATA_GLM_SEQ_CTX");
        unsetenv("STRATA_GLM_PAGED");
        unsetenv("STRATA_GLM_PAGE_SIZE");
        unsetenv("STRATA_GLM_PAGE_POOL_GB");
        unsetenv("STRATA_GLM_PAUSE_BLOCKS");
        unsetenv("STRATA_GLM_NO_PREFIX");
        unsetenv("STRATA_GLM_SEQ_FLOOR");
    };
    setenv("STRATA_GLM_SLOW", "1", 1);          // the harness's path (the caller's main set it)
    const std::vector<int32_t> prompt = {11, 13, 17, 19, 23, 29};
    const std::string pack_path = pack.empty() ? "data/glm5-synth-pack" : pack;
    std::string err;
    // the greedy drive: forward the prompt, then sample -> forward pairs; the
    // span collects the tokens FED (the prompt + the sampled ones behind it)
    auto drive = [&](Glm5Model& m, const std::vector<int32_t>& pr, int samples,
                     std::vector<int32_t>& span_out) -> bool {
        std::vector<float> lg;
        if (!m.forward(pr, lg, err)) {
            printf("FAIL: the drive's forward: %s\n", err.c_str());
            return false;
        }
        span_out = pr;
        strata::kernels::SamplerParams sp;
        sp.temperature = 0.0f;                  // greedy
        for (int i = 0; i < samples; ++i) {
            const int tok = m.sample_token(sp, err);
            if (tok < 0) { printf("FAIL: the drive's sample: %s\n", err.c_str()); return false; }
            const std::vector<int32_t> one{(int32_t) tok};
            std::vector<float> lg2;
            if (!m.forward(one, lg2, err)) { printf("FAIL: the drive's decode: %s\n", err.c_str()); return false; }
            span_out.push_back((int32_t) tok);
        }
        return true;
    };
    auto more = [&](Glm5Model& m, std::vector<int32_t>& span, int n) -> bool {
        strata::kernels::SamplerParams sp;
        sp.temperature = 0.0f;
        for (int i = 0; i < n; ++i) {
            const int tok = m.sample_token(sp, err);
            if (tok < 0) return false;
            const std::vector<int32_t> one{(int32_t) tok};
            std::vector<float> lg;
            if (!m.forward(one, lg, err)) return false;
            span.push_back((int32_t) tok);
        }
        return true;
    };
    strata::kernels::SamplerParams sp;
    sp.temperature = 0.0f;

    // ---- leg A: the reference drive (the pool off) + the registry-off refusal
    unsetenv("STRATA_GLM_PAGED");
    unsetenv("STRATA_GLM_NO_PREFIX");
    int ref_a3 = -1, ref_s4 = -1;
    std::vector<char> ref_bytes;
    {
        Glm5Model a;
        if (!a.load_pack(pack_path, 4096, err)) {
            req("leg A loaded the pack", false);
            printf("FAIL: leg A's load_pack: %s\n", err.c_str());
            restore_envs();
            return false;
        }
        req("leg A loaded the pack", true);
        a.seq_bind(1);
        std::vector<int32_t> span_a;
        if (!drive(a, prompt, 2, span_a) || a.position() != (int64_t) span_a.size()) {
            req("leg A drove to the pause position", false);
            restore_envs();
            return false;
        }
        req("leg A drove to the pause position", true);
        ref_bytes = pause_state_bytes(a);
        req("the reference copied its state regions", !ref_bytes.empty());
        req("the capture refuses with the pool off",
            !a.pause_capture(span_a.data(), (int) span_a.size()));
        ref_a3 = a.sample_token(sp, err);
        const std::vector<int32_t> one{(int32_t) ref_a3};
        std::vector<float> lg;
        if (ref_a3 < 0 || !a.forward(one, lg, err)) {
            req("the reference continued the decode", false);
            restore_envs();
            return false;
        }
        ref_s4 = a.sample_token(sp, err);
        req("the reference sampled the continuation", ref_a3 >= 0 && ref_s4 >= 0);
    }   // a's destructor: the pool off, nothing to release

    // ---- leg B: the registry on (the pool + the identity table): capture ->
    // pressure -> park -> resume -> byte-identical continuation + state
    setenv("STRATA_GLM_SEQS", "2", 1);
    setenv("STRATA_GLM_SEQ_CTX", "2048", 1);
    setenv("STRATA_GLM_SEQ_VRAM_FIT", "1", 1);
    setenv("STRATA_GLM_PAGED", "1", 1);
    setenv("STRATA_GLM_PAGE_SIZE", "4096", 1);     // one page covers the whole ctx: identity
    // (STRATA_GLM_PAGE_POOL_GB stays unset: atoll parses whole GBs; the pool
    //  shrinks by halves beside production, the 4-page floor always fits)
    setenv("STRATA_GLM_PAUSE_BLOCKS", "2", 1);     // the cap the legs below exercise
    bool leg_b_ok = true;
    {
        Glm5Model b;
        if (!b.load_pack(pack_path, 4096, err)) {
            req("leg B loaded the pack", false);
            printf("FAIL: leg B's load_pack: %s\n", err.c_str());
            restore_envs();
            return false;
        }
        req("leg B loaded the pack", true);
        req("leg B's pool is on with the identity table",
            b.lat_pages_.n_pages() > 0 && b.lat_pages_.table_identity());
        if (!leg_b_ok) { restore_envs(); return false; }
        b.seq_bind(1);
        std::vector<int32_t> span_b;
        if (!drive(b, prompt, 2, span_b) || b.position() != (int64_t) span_b.size()) {
            req("leg B drove to the pause position", false);
            restore_envs();
            return false;
        }
        req("leg B drove to the pause position", true);
        // the reference window env: the capture refuses, the incumbent carries
        setenv("STRATA_GLM_NO_PREFIX", "1", 1);
        req("the capture refuses under STRATA_GLM_NO_PREFIX",
            !b.pause_capture(span_b.data(), (int) span_b.size()));
        unsetenv("STRATA_GLM_NO_PREFIX");
        // the capture lands
        const bool captured = b.pause_capture(span_b.data(), (int) span_b.size());
        req("the capture captured the paused state", captured);
        if (!captured || g_pause_snaps.size() != 1) {
            req("the store holds one capture", false);
            restore_envs();
            return false;
        }
        req("the store holds one capture", true);
        size_t cap_pages = 0, pinned = 0;
        for (int32_t pg : g_pause_snaps[0].gpu[0].pages) {
            ++cap_pages;
            if (b.lat_pages_.refcount(pg) > 0) ++pinned;
        }
        req("the capture's pages are pinned", pinned == cap_pages && cap_pages > 0);
        // the pressure: fill the pool with raw pages, sweep, the capture's pages hold
        std::vector<int32_t> raw;
        for (;;) {
            const int32_t pg = b.lat_pages_.page_alloc();
            if (pg < 0) break;
            raw.push_back(pg);
            if (b.lat_pages_.counters().live >= b.lat_pages_.n_pages()) break;
        }
        req("the pool filled beside the capture's pages",
            b.lat_pages_.counters().live == b.lat_pages_.n_pages());
        const int32_t swept = b.pages_sweep_idle();
        req("the sweep fired under pressure", swept > 0);
        req("the sweep took only the free pages",
            b.lat_pages_.counters().live == (int64_t) cap_pages);
        size_t still = 0;
        for (int32_t pg : g_pause_snaps[0].gpu[0].pages)
            if (b.lat_pages_.refcount(pg) > 0) ++still;
        req("the capture's pages survived the sweep", still == cap_pages);
        // the park: the slot's arena frees (the incumbent pair dies with it)
        b.seq_release(1);
        req("the park freed the slot", !b.seq_live(1));
        b.seq_bind(1);   // the fresh zeroed arena
        req("the re-bind allocated a fresh arena", b.seq_live(1) && b.seq_bound() == 1);
        // the resume: the new request's prompt extends the capture's span
        std::vector<int32_t> prompt2 = span_b;
        prompt2.push_back((int32_t) ref_a3);
        std::vector<int32_t> span_out;
        const int64_t r = b.pause_resume(prompt2, span_out);
        req("the resume restored the capture", r == (int64_t) span_b.size());
        req("the resume reported the span", span_out == span_b);
        req("the resume consumed its entry", g_pause_snaps.size() == 0);
        const std::vector<char> got_bytes = pause_state_bytes(b);
        req("the restored state is byte-identical", got_bytes == ref_bytes);
        const int got_a3 = b.sample_token(sp, err);
        req("the restored state samples the same greedy token", got_a3 == ref_a3);
        const std::vector<int32_t> one{(int32_t) got_a3};
        std::vector<float> lg;
        if (got_a3 < 0 || !b.forward(one, lg, err)) leg_b_ok = false;
        const int got_s4 = b.sample_token(sp, err);
        req("the continuation is byte-identical", leg_b_ok && got_s4 == ref_s4);
        if (!leg_b_ok) { restore_envs(); return false; }
        // the dedupe + the cap: the second capture (the state moved on), the
        // third fills the cap 2, and the FOURTH pushes past it - the least
        // recently captured entry drops, its page comes back
        std::vector<int32_t> span3 = span_b;
        span3.push_back((int32_t) ref_a3);           // the state now stands at span3
        req("the second capture lands", b.pause_capture(span3.data(), (int) span3.size()));
        req("the store holds the second capture", g_pause_snaps.size() == 1);
        if (!more(b, span3, 2) || !b.pause_capture(span3.data(), (int) span3.size())
            || g_pause_snaps.size() != 2) {
            req("the third capture fills the cap", false);
            restore_envs();
            return false;
        }
        req("the third capture fills the cap", true);
        if (!more(b, span3, 2) || !b.pause_capture(span3.data(), (int) span3.size())
            || g_pause_snaps.size() != 2) {
            req("the cap dropped the least recent capture", false);
            restore_envs();
            return false;
        }
        req("the cap dropped the least recent capture", true);
        // the drop's page_unref frees the page at once (refcount 0 -> the free
        // list), so exactly the two surviving captures' pages stay live
        req("the dropped capture's pages came back",
            b.lat_pages_.counters().live == 2);
        // the resume of the latest capture: the state restores into a fresh arena
        // (release -> re-bind -> resume over the span + one more token)
        std::vector<int32_t> last_span;
        int64_t r2 = 0;
        {
            // the latest entry's span: the store's back (the P+7 capture)
            std::vector<int32_t> s7 = g_pause_snaps.back().span;
            std::vector<int32_t> prompt3 = s7;
            prompt3.push_back(101);   // one more token than the capture holds
            b.seq_release(1);
            b.seq_bind(1);
            std::vector<int32_t> span_out2;
            r2 = b.pause_resume(prompt3, span_out2);
            req("the second resume restored the latest capture",
                r2 == (int64_t) s7.size() && span_out2 == s7);
            last_span = s7;
            (void) r2;
        }
        (void) last_span;
    }   // b's destructor: the entries drop with their pages
    req("the destructor dropped the entries", g_pause_snaps.empty());

    // ---- leg C: the registry off (the pool off): the incumbent path carries
    unsetenv("STRATA_GLM_PAGED");
    unsetenv("STRATA_GLM_PAGE_SIZE");
    unsetenv("STRATA_GLM_PAUSE_BLOCKS");
    unsetenv("STRATA_GLM_NO_PREFIX");
    {
        Glm5Model c;
        if (!c.load_pack(pack_path, 4096, err)) {
            req("leg C loaded the pack", false);
            printf("FAIL: leg C's load_pack: %s\n", err.c_str());
            restore_envs();
            return false;
        }
        req("leg C loaded the pack", true);
        c.seq_bind(1);
        std::vector<int32_t> span_c;
        const std::vector<char> before_c = [&] { std::vector<char> v; if (drive(c, prompt, 2, span_c)) v = pause_state_bytes(c); return v; }();
        if (before_c.empty()) {
            req("leg C drove to the pause position", false);
            restore_envs();
            return false;
        }
        req("leg C drove to the pause position", true);
        req("the capture refuses without the pool (leg C)",
            !c.pause_capture(span_c.data(), (int) span_c.size()));
        req("the store stays empty", g_pause_snaps.empty());
        // the resume answers 0: the incumbent path carries (the live state was
        // never touched - the driver's own snapshot pair stays where it was)
        std::vector<int32_t> span_out_c;
        req("the resume answers 0 without the pool",
            c.pause_resume(span_c, span_out_c) == 0 && span_out_c.empty());
        req("the release is a clean no-op", (c.pause_release(span_c), true));
        // the live state survived the refused capture and the answered-0 resume
        // byte for byte: the decode continues where the drive left it
        const std::vector<char> c_bytes = pause_state_bytes(c);
        req("the state survived the fallback legs", c_bytes == before_c);
        const int got_a3c = c.sample_token(sp, err);
        req("the fallback continuation matches", got_a3c == ref_a3);
        const std::vector<int32_t> one{(int32_t) got_a3c};
        std::vector<float> lg;
        if (got_a3c < 0 || !c.forward(one, lg, err)) { restore_envs(); return false; }
        const int got_s4c = c.sample_token(sp, err);
        req("the fallback continuation is byte-identical", got_s4c == ref_s4);
    }

    // ---- leg D: the evict-then-retry (Task 4.2 step 3): a pool whose pages
    // are all pinned by an unheld store tile; the capture cannot pack without
    // evicting it, evicts LRU-first, lands byte-identically.  The held-tile
    // variant refuses cleanly and drops nothing.  Leg E finalizes a REAL
    // pending chunk (the capture_part shape) and proves the store's
    // registration lands refs 0 (the store's pin, not a request's hold):
    // the eviction reclaims the tile when the capture needs the pages.
    setenv("STRATA_GLM_PAGED", "1", 1);
    setenv("STRATA_GLM_PAGE_SIZE", "4096", 1);
    setenv("STRATA_GLM_PAUSE_BLOCKS", "2", 1);
    {
        Glm5Model d;
        if (!d.load_pack(pack_path, 4096, err)) {
            req("leg D loaded the pack", false);
            printf("FAIL: leg D's load_pack: %s\n", err.c_str());
            restore_envs();
            return false;
        }
        req("leg D loaded the pack", true);
        req("leg D's pool is on with the identity table",
            d.lat_pages_.n_pages() > 0 && d.lat_pages_.table_identity());
        req("the store starts empty", g_prefix_blocks.empty());
        d.seq_bind(1);
        std::vector<int32_t> span_d;
        if (!drive(d, prompt, 2, span_d) || d.position() != (int64_t) span_d.size()) {
            req("leg D drove to the pause position", false);
            restore_envs();
            return false;
        }
        req("leg D drove to the pause position", true);
        const int32_t n_pages_d = d.lat_pages_.n_pages();
        // D1: the unheld tile pins EVERY page: the packer's page_alloc finds
        // no free page, so the capture must evict the tile to land
        const strata::prefix::BlockId tid{0xD001, 0xD002};
        PrefixBlock blk;
        blk.id = tid;
        blk.key = 0xDEADBEEFull;
        blk.start = 0;
        blk.boundary = 1;
        blk.stateful = true;
        blk.last_use = ++g_prefix_clock;
        blk.parts.resize(1);
        PrefixTileData td;
        for (int32_t i = 0; i < n_pages_d; ++i) {
            const int32_t pg = d.lat_pages_.page_alloc();
            if (pg < 0) break;
            d.lat_pages_.page_ref(pg);   // the store's pin
            td.pages.push_back(pg);
        }
        blk.parts[0].tiles.push_back(std::move(td));
        g_prefix_registry_.register_block(tid, blk.key);
        g_prefix_registry_.release(tid);   // refs 0: unheld, evictable
        g_prefix_blocks.push_back(std::move(blk));
        req("the tile pins the whole pool", d.lat_pages_.counters().live == n_pages_d);
        const bool captured_d = d.pause_capture(span_d.data(), (int) span_d.size());
        req("the capture evicted the tile and landed", captured_d);
        req("the tile left the store", g_prefix_blocks.empty());
        req("the registry forgot the tile", g_prefix_registry_.refcount(tid) == 0);
        size_t pages_d = 0;
        if (!g_pause_snaps.empty())
            for (int32_t pg : g_pause_snaps[0].gpu[0].pages) ++pages_d;
        req("the capture's pages are the pool's live pages",
            pages_d > 0 && d.lat_pages_.counters().live == (int64_t) pages_d);
        const std::vector<char> got_d = pause_state_bytes(d);
        req("the evicted-capture's state is byte-identical", got_d == ref_bytes);
        const int got_a3d = d.sample_token(sp, err);
        req("the evicted-capture samples the reference token", got_a3d == ref_a3);
        if (got_a3d >= 0) {
            const std::vector<int32_t> one{(int32_t) got_a3d};
            std::vector<float> lg;
            if (!d.forward(one, lg, err)) { restore_envs(); return false; }
        }
        const std::vector<char> bytes_d2 = pause_state_bytes(d);   // the post-continuation baseline
        // D2: the held tile: the capture refuses cleanly, nothing drops
        d.pause_release(span_d);
        req("the release freed the capture's pages", d.lat_pages_.counters().live == 0);
        const strata::prefix::BlockId hid{0xD003, 0xD004};
        PrefixBlock hblk;
        hblk.id = hid;
        hblk.key = 0xDEADBEEF1ull;
        hblk.start = 0;
        hblk.boundary = 1;
        hblk.stateful = true;
        hblk.last_use = ++g_prefix_clock;
        hblk.parts.resize(1);
        PrefixTileData htd;
        for (int32_t i = 0; i < n_pages_d; ++i) {
            const int32_t pg = d.lat_pages_.page_alloc();
            if (pg < 0) break;
            d.lat_pages_.page_ref(pg);
            htd.pages.push_back(pg);
        }
        hblk.parts[0].tiles.push_back(std::move(htd));
        g_prefix_registry_.register_block(hid, hblk.key);   // refs 1: a request holds it
        g_prefix_blocks.push_back(std::move(hblk));
        req("the held tile pins the whole pool", d.lat_pages_.counters().live == n_pages_d);
        req("the capture refuses with a held tile",
            !d.pause_capture(span_d.data(), (int) span_d.size()));
        req("the held tile stayed in the store", g_prefix_blocks.size() == 1);
        req("the held tile kept its pages", d.lat_pages_.counters().live == n_pages_d);
        const std::vector<char> got_d2 = pause_state_bytes(d);
        req("the state survived the refused capture", got_d2 == bytes_d2);
        // the held tile drops by hand (the destructor's cleanup covers the
        // pause store only): the same path the cap-drop uses
        if (!g_prefix_blocks.empty()) {
            PrefixBlock& b = g_prefix_blocks.back();
            for (size_t p = 0; p < b.parts.size(); ++p)
                for (const PrefixTileData& tdd : b.parts[p].tiles)
                    drop_tile_pages(d.lat_pages_, tdd);
            g_prefix_registry_.forget(b.id);
            g_prefix_blocks.erase(g_prefix_blocks.end() - 1);
        }
        req("the held tile dropped clean",
            g_prefix_blocks.empty() && d.lat_pages_.counters().live == 0);
        // E: the REAL finalize lifecycle (an earlier run's 0 evictions showed why):
        // the store's registration is the tile's pin, NOT a request's hold -
        // refs must land 0 so the Step 3 eviction + the cap-drop can reclaim
        // the tile when no request holds it.  The pending chunk is built in
        // the capture_part's shape (the pages pinned, a non-stateful tile)
        // and prefix_finalize runs for real (the hash staging + the registry).
        req("leg E's pool is free", d.lat_pages_.counters().live == 0);
        PrefixPendingChunk pchunk;
        pchunk.chunk_start = 0;
        pchunk.parts.resize(1);
        pchunk.present.resize(1);
        PrefixTileData etd;
        etd.start = 0;
        etd.end = 4;
        etd.vec_start = 0;
        etd.stateful = false;
        for (int32_t i = 0; i < n_pages_d; ++i) {
            const int32_t pg = d.lat_pages_.page_alloc();
            if (pg < 0) break;
            d.lat_pages_.page_ref(pg);   // the tile's pin (capture_part's)
            etd.pages.push_back(pg);
        }
        pchunk.parts[0].tiles.push_back(std::move(etd));
        pchunk.present[0] = true;
        g_prefix_pending[4] = std::move(pchunk);
        req("the tile pins the whole pool", d.lat_pages_.counters().live == n_pages_d);
        std::vector<int32_t> span_e = span_d;
        span_e.push_back(got_a3d);   // the model sits at pos 9 (D1's continuation)
        d.prefix_finalize(span_e, 0, 4, true);
        req("the finalize stored the tile", g_prefix_blocks.size() == 1);
        req("the finalized tile is unheld (refs 0: reclaimable)",
            !g_prefix_blocks.empty() && g_prefix_registry_.refcount(g_prefix_blocks[0].id) == 0);
        req("the pending map drained", g_prefix_pending.empty());
        const bool captured_e = d.pause_capture(span_e.data(), (int) span_e.size());
        req("the capture evicted the finalized tile and landed", captured_e);
        req("the store drained after the capture", g_prefix_blocks.empty());
        size_t pages_e = 0;
        if (!g_pause_snaps.empty())
            for (int32_t pg : g_pause_snaps.back().gpu[0].pages) ++pages_e;
        req("the capture's pages are the pool's live pages",
            pages_e > 0 && d.lat_pages_.counters().live == (int64_t) pages_e);
        // the leg's cleanup runs whatever the capture answered (the red run
        // leaves the tile held): the same hand-drop the cap-drop uses
        while (!g_prefix_blocks.empty()) {
            PrefixBlock& b = g_prefix_blocks.back();
            for (size_t p = 0; p < b.parts.size(); ++p)
                for (const PrefixTileData& tdd : b.parts[p].tiles)
                    drop_tile_pages(d.lat_pages_, tdd);
            g_prefix_registry_.forget(b.id);
            g_prefix_blocks.erase(g_prefix_blocks.end() - 1);
        }
        req("the store drained at the leg's end",
            g_prefix_blocks.empty() && d.lat_pages_.counters().live == (int64_t) pages_e);
    }   // d's destructor: the pause store's entries drop with their pages

    restore_envs();
    req("the envs restored", getenv("STRATA_GLM_PAGED") == nullptr
        && getenv("STRATA_GLM_PAGE_SIZE") == nullptr && getenv("STRATA_GLM_PAGE_POOL_GB") == nullptr
        && getenv("STRATA_GLM_PAUSE_BLOCKS") == nullptr && getenv("STRATA_GLM_NO_PREFIX") == nullptr);
    printf("glm_pause_snapshots_selftest: %d checks, %d failures\n", *checks, *failures);
    return *failures == 0;
}

}  // namespace strata::core