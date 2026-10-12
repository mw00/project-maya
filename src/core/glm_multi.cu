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

#if !defined(STRATA_USE_HIP)
#include <cuda.h>
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core {

// ---- ELASTIC CONTEXTS: the driver's virtual memory calls, taken from the runtime (no link to libcuda)
#if !defined(STRATA_USE_HIP)
namespace {
struct Vmm {
    decltype(&::cuMemAddressReserve) reserve = nullptr;
    decltype(&::cuMemAddressFree) addr_free = nullptr;
    decltype(&::cuMemCreate) create = nullptr;
    decltype(&::cuMemRelease) release = nullptr;
    decltype(&::cuMemMap) map = nullptr;
    decltype(&::cuMemUnmap) unmap = nullptr;
    decltype(&::cuMemSetAccess) access = nullptr;
    decltype(&::cuMemGetAllocationGranularity) granularity = nullptr;
    decltype(&::cuDeviceGetAttribute) attribute = nullptr;
    bool ok = false;
};
const Vmm& vmm() {
    static const Vmm v = [] {
        Vmm r;
        const auto get = [](const char* name, void** fn) {
            cudaDriverEntryPointQueryResult q{};
#if CUDART_VERSION >= 12050
            return cudaGetDriverEntryPointByVersion(name, fn, 12000, cudaEnableDefault, &q) == cudaSuccess &&
                   q == cudaDriverEntryPointSuccess && *fn != nullptr;
#else
            return cudaGetDriverEntryPoint(name, fn, cudaEnableDefault, &q) == cudaSuccess &&
                   q == cudaDriverEntryPointSuccess && *fn != nullptr;
#endif
        };
        r.ok = get("cuMemAddressReserve", (void**) &r.reserve) && get("cuMemAddressFree", (void**) &r.addr_free) &&
               get("cuMemCreate", (void**) &r.create) && get("cuMemRelease", (void**) &r.release) &&
               get("cuMemMap", (void**) &r.map) && get("cuMemUnmap", (void**) &r.unmap) &&
               get("cuMemSetAccess", (void**) &r.access) &&
               get("cuMemGetAllocationGranularity", (void**) &r.granularity) &&
               get("cuDeviceGetAttribute", (void**) &r.attribute);
        return r;
    }();
    return v;
}
CUmemAllocationProp vmm_prop(int dev) {
    CUmemAllocationProp p{};
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    p.location.id = dev;
    return p;
}
size_t vmm_granule(int dev) {
    const Vmm& v = vmm();
    int sup = 0;
    if (!v.ok || v.attribute(&sup, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev) != CUDA_SUCCESS || !sup)
        return 0;
    const CUmemAllocationProp p = vmm_prop(dev);
    size_t gr = 0;
    if (v.granularity(&gr, &p, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS) return 0;
    return gr;
}
}  // namespace

struct Glm5Model::Elastic {
    CUdeviceptr base = 0;
    size_t va = 0, fixed = 0, gran = 0;
    int dev = 0;
    std::vector<CUmemGenericAllocationHandle> hfixed;
    struct Grow {
        size_t off = 0, cap = 0, backed = 0;   // bytes: where in the reservation, how many it may take, backed now
        double bpt = 0;                         // bytes a position
        std::vector<CUmemGenericAllocationHandle> h;
    };
    std::vector<Grow> grow;
    int64_t tokens = 0, max_ctx = 0;            // positions every growing region backs; the context
    bool fixed_on = false;                      // the fixed part (state, rings, snapshots) backed
    // back [off, off + bytes) of the reservation (a granule multiple) with one new handle
    bool back(size_t off, size_t bytes, std::vector<CUmemGenericAllocationHandle>& hs) {
        const Vmm& v = vmm();
        const CUmemAllocationProp p = vmm_prop(dev);
        CUmemGenericAllocationHandle h = 0;
        if (v.create(&h, bytes, &p, 0) != CUDA_SUCCESS) return false;
        if (v.map(base + off, bytes, 0, h, 0) != CUDA_SUCCESS) {
            v.release(h);
            return false;
        }
        CUmemAccessDesc ad{};
        ad.location = p.location;
        ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        if (v.access(base + off, bytes, &ad, 1) != CUDA_SUCCESS) {
            v.unmap(base + off, bytes);
            v.release(h);
            return false;
        }
        hs.push_back(h);
        return true;
    }
};
#else
struct Glm5Model::Elastic {};
#endif

// the elastic expert pool: one reservation, every layer's main slots on a granule boundary, backed one granule a
// handle so a layer's top can be given up a granule at a time (its last, partly used granule stays)
#if !defined(STRATA_USE_HIP)
struct Glm5Model::ElPool {
    CUdeviceptr base = 0;
    size_t va = 0, gran = 0;
    int dev = 0;
    struct Layer {
        size_t off = 0, stride = 0, mapped = 0;
        int max_slots = 0, live = 0;
        std::vector<CUmemGenericAllocationHandle> h;   // granule i of the layer -> its handle (0: unbacked)
    };
    std::vector<Layer> L;   // by absolute layer
    // back the layer up to `slots` slots / unback it down to them; the bytes that changed
    size_t grow(int il, int slots) {
        Layer& y = L[(size_t) il];
        const size_t end = std::min(y.h.size() * gran, ((size_t) slots * y.stride + gran - 1) / gran * gran);
        size_t added = 0;
        const Vmm& v = vmm();
        const CUmemAllocationProp p = vmm_prop(dev);
        CUmemAccessDesc ad{};
        ad.location = p.location;
        ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        while (y.mapped < end) {
            const size_t i = y.mapped / gran;
            CUmemGenericAllocationHandle hd = 0;
            if (v.create(&hd, gran, &p, 0) != CUDA_SUCCESS) break;
            if (v.map(base + y.off + y.mapped, gran, 0, hd, 0) != CUDA_SUCCESS) {
                v.release(hd);
                break;
            }
            if (v.access(base + y.off + y.mapped, gran, &ad, 1) != CUDA_SUCCESS) {
                v.unmap(base + y.off + y.mapped, gran);
                v.release(hd);
                break;
            }
            y.h[i] = hd;
            y.mapped += gran;
            added += gran;
        }
        y.live = std::min(slots, (int) (y.mapped / y.stride));
        return added;
    }
    size_t shrink(int il, int slots) {
        Layer& y = L[(size_t) il];
        const size_t end = ((size_t) slots * y.stride + gran - 1) / gran * gran;
        size_t freed = 0;
        const Vmm& v = vmm();
        while (y.mapped > end) {
            y.mapped -= gran;
            const size_t i = y.mapped / gran;
            v.unmap(base + y.off + y.mapped, gran);
            v.release(y.h[i]);
            y.h[i] = 0;
            freed += gran;
        }
        y.live = slots;
        return freed;
    }
};

uint8_t* Glm5Model::el_pool_alloc(const std::vector<std::array<size_t, 3>>& layers, std::vector<size_t>& offs,
                                  std::string& err) {
    const size_t gran = vmm_granule(dev_);
    if (gran == 0) return nullptr;
    auto* P = new ElPool();
    P->dev = dev_;
    P->gran = gran;
    P->L.assign((size_t) g_.n_layers + 1, ElPool::Layer{});
    offs.assign((size_t) g_.n_layers + 1, 0);
    size_t off = 0;
    for (const auto& ly : layers) {   // {layer, slots, stride}
        auto& y = P->L[ly[0]];
        y.off = off;
        y.stride = ly[2];
        y.max_slots = (int) ly[1];
        const size_t bytes = (ly[1] * ly[2] + gran - 1) / gran * gran;
        y.h.assign(bytes / gran, 0);
        offs[ly[0]] = off;
        off += bytes;
    }
    P->va = off;
    if (vmm().reserve(&P->base, P->va, gran, 0, 0) != CUDA_SUCCESS) {
        delete P;
        err = "glm elastic pool: no address space";
        return nullptr;
    }
    elp_ = P;
    for (const auto& ly : layers)
        if (P->grow((int) ly[0], (int) ly[1]) < P->L[ly[0]].h.size() * gran) {
            el_pool_free();
            err = "glm elastic pool: CUDA" + std::to_string(dev_) + " could not back the expert pool";
            return nullptr;
        }
    return (uint8_t*) P->base;
}

void Glm5Model::el_pool_free() {
    if (elp_ == nullptr) return;
    cudaSetDevice(dev_);
    cudaDeviceSynchronize();
    for (size_t il = 0; il < elp_->L.size(); ++il) elp_->shrink((int) il, 0);
    vmm().addr_free(elp_->base, elp_->va);
    delete elp_;
    elp_ = nullptr;
}

// this part: at least `bytes` of expert slots given up for the contexts' budget - the layer whose top slots hold the
// coldest experts first, a few slots at a time, never below a route's experts and the spares (the caller has nothing
// in flight on this part; a prompt is not being read beside it)
bool Glm5Model::pool_give(size_t bytes, std::string& err) {
    FastState* F = fast_;
    if (elp_ == nullptr || F == nullptr) {
        err = "glm elastic: the contexts' budget is used up (no elastic expert pool)";
        return false;
    }
    const int floor_slots = g_.n_exp_used + gf::kSpares + 2;
    const int NE = g_.n_expert;
    size_t got = 0;
    cudaSetDevice(dev_);
    while (got < bytes) {
        int best = -1, bj = 0;
        double bc = 0;
        for (int il = l0_; il < lt_; ++il) {
            if (!F->L[(size_t) il].moe) continue;
            const auto& y = elp_->L[(size_t) il];
            const auto& P = F->lp[(size_t) il];
            const int room = y.live - floor_slots;
            if (room <= 0) continue;
            const int j = std::min(room, std::max(1, (int) std::min<size_t>(16, (bytes - got + y.stride - 1) / y.stride)));
            // the share of this layer's own traffic those slots serve, per byte: a layer used only part of the time (the
            // NextN block drafts only alone) is not the cheapest just because its counts grow slower
            double c = 0, tot = 0;
            for (int s = y.live - j; s < y.live; ++s)
                if (P.st[(size_t) s] == FastState::kResident && P.key[(size_t) s] >= 0)
                    c += (double) F->cnt[(size_t) il * NE + P.key[(size_t) s]];
            for (int e = 0; e < NE; ++e) tot += (double) F->cnt[(size_t) il * NE + e];
            c /= std::max(1.0, tot) * (double) j * (double) y.stride;
            if (best < 0 || c < bc) {
                best = il;
                bj = j;
                bc = c;
            }
        }
        if (best < 0) {
            err = "glm elastic: CUDA" + std::to_string(dev_) + " no expert slots left to give a context";
            return false;
        }
        const int lo = elp_->L[(size_t) best].live - bj, hi = elp_->L[(size_t) best].live;
        uint64_t moved = 0, dropped = 0;
        if (!lend_slots([best, lo, hi](int il, int s) { return il == best && s >= lo && s < hi; }, moved, dropped, err))
            return false;
        cudaStreamSynchronize(F->cs);
        const size_t freed = elp_->shrink(best, lo);
        el_budget_ += freed;
        got += freed;
        std::fprintf(stderr, "glm elastic: CUDA%d layer %d gives %d expert slots (%.0f MB) to the contexts: %d left\n",
                     dev_, best, bj, (double) freed / 1048576.0, lo);
    }
    return true;
}

// this part: the budget beyond the base one goes back to the experts (the layers that gave most, first)
void Glm5Model::pool_take_back() {
    FastState* F = fast_;
    if (elp_ == nullptr || F == nullptr) return;
    cudaSetDevice(dev_);
    for (;;) {
        const size_t slack = el_budget_ > el_backed_ + el_budget0_ ? el_budget_ - el_backed_ - el_budget0_ : 0;
        if (slack < elp_->gran) return;
        int best = -1;
        double bf = 2.0;
        for (int il = l0_; il < lt_; ++il) {
            const auto& y = elp_->L[(size_t) il];
            if (!F->L[(size_t) il].moe || y.live >= y.max_slots) continue;
            const double f = (double) y.live / y.max_slots;
            if (f < bf) {
                bf = f;
                best = il;
            }
        }
        if (best < 0) return;
        auto& y = elp_->L[(size_t) best];
        const int from = y.live;
        int to = from;
        while (to < y.max_slots && ((size_t) (to + 1) * y.stride + elp_->gran - 1) / elp_->gran * elp_->gran <= y.mapped + slack)
            ++to;
        if (to == from) return;
        const size_t used = elp_->grow(best, to);
        el_budget_ -= std::min(el_budget_, used);
        if (used > 0)
            std::fprintf(stderr, "glm elastic: CUDA%d layer %d takes back %d expert slots (%.0f MB): %d\n", dev_, best,
                         y.live - from, (double) used / 1048576.0, y.live);
        std::lock_guard<std::mutex> lk(F->mu);
        auto& P = F->lp[(size_t) best];
        for (int s = from; s < y.live; ++s)
            if (P.st[(size_t) s] == FastState::kLent) P.st[(size_t) s] = FastState::kFree;
        if (used == 0) return;
    }
}

void Glm5Model::pool_take_back_all() {
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) m->pool_take_back();
    cudaSetDevice(dev_);
}

// every part: s backs `tokens` positions; where the budget is short, expert slots give way first (the caller has
// nothing in flight and no prompt being read beside it)
bool Glm5Model::seq_reserve_all(int s, int64_t tokens, std::string& err) {
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        if (m->el_.empty() || s < 0 || s >= (int) m->el_.size()) continue;
        const Elastic& e = *m->el_[(size_t) s];
        size_t more = e.fixed_on ? 0 : e.fixed;
        for (const auto& r : e.grow) {
            const size_t need = std::min(r.cap, (size_t) std::ceil((double) tokens * r.bpt));
            const size_t need_g = (need + e.gran - 1) / e.gran * e.gran;
            if (need_g > r.backed) more += need_g - r.backed;
        }
        if (m->el_backed_ + more > m->el_budget_ && !m->pool_give(m->el_backed_ + more - m->el_budget_, err)) {
            cudaSetDevice(dev_);
            return false;
        }
        if (!m->seq_grow(s, tokens, err)) {
            cudaSetDevice(dev_);
            return false;
        }
    }
    cudaSetDevice(dev_);
    return true;
}
#else
struct Glm5Model::ElPool {};
uint8_t* Glm5Model::el_pool_alloc(const std::vector<std::array<size_t, 3>>&, std::vector<size_t>&, std::string&) {
    return nullptr;
}
void Glm5Model::el_pool_free() {}
bool Glm5Model::pool_give(size_t, std::string& err) {
    err = "glm elastic: not on HIP";
    return false;
}
void Glm5Model::pool_take_back() {}
void Glm5Model::pool_take_back_all() {}
bool Glm5Model::seq_reserve_all(int s, int64_t tokens, std::string& err) { return seq_grow_all(s, tokens, err); }
#endif

// the state layout of an ELASTIC arena (ctx tokens): the fixed part first - the residual pair, then per layer (the
// NextN block's first) the KDA state + conv history or the DSA index rings - then every DSA layer's latent cache and
// pooled keys, each on a granule boundary so it can be backed from its start as positions arrive.  c's offsets and
// fixed bytes; `grow` gets each growing region's {offset, capacity, bytes a position} in bytes.
static uint64_t el_layout_impl(const Glm5Geometry& g, bool lat_q8, int mtp_il, int l0, int l1, int64_t ctx,
                               size_t gran, Glm5Model::SeqCtx& c, std::vector<std::array<double, 3>>& grow,
                               size_t& fixed, int64_t& snap_off, int64_t& pool_off) {
    const int64_t hc_dim = (int64_t) g.hc * g.n_embd;
    const int64_t max_pools = ctx / g.idx_kpool;
    const int64_t gfl = (int64_t) gran / 4;
    const auto up = [&](int64_t f) { return (f + gfl - 1) / gfl * gfl; };
    const double lat_bpt = lat_q8 ? (double) gf::lat8_rec_bytes(g.kv_lora) : (double) g.kv_lora * 2.0;
    const int64_t lat_floats = lat_q8 ? (int64_t) gf::lat8_rec_bytes(g.kv_lora) / 4 * ctx : (int64_t) g.kv_lora * ctx / 2;
    for (auto* v : {&c.kda_S, &c.kda_conv, &c.dsa_lat, &c.dsa_ik, &c.dsa_ig, &c.dsa_pool})
        v->assign((size_t) g.n_layers + 1, 0);
    c.ik_ring = (int) std::min<int64_t>(ctx, 8192 + 64);
    int64_t floats = 2 * hc_dim;
    std::vector<int> dsa;
    if (mtp_il >= 0) dsa.push_back(mtp_il);
    for (int il = l0; il < l1; ++il) if (!g.is_recr(il)) dsa.push_back(il);
    for (const int il : dsa) {
        c.dsa_ik[(size_t) il] = floats;
        floats += (int64_t) g.idx_key * c.ik_ring;
        c.dsa_ig[(size_t) il] = floats;
        floats += (int64_t) g.idx_key * c.ik_ring;
    }
    int64_t n_rec = 0;
    for (int il = l0; il < l1; ++il)
        if (g.is_recr(il)) {
            c.kda_S[(size_t) il] = floats;
            floats += (int64_t) g.d_inner() * g.kda_head_dim;
            c.kda_conv[(size_t) il] = floats;
            floats += (int64_t) 3 * g.d_inner() * (g.d_conv - 1);
            ++n_rec;
        }
    // the sequence's snapshot (its recurrent states, its DSA layers' open index pools: snapshot_save's buffers)
    snap_off = floats;
    floats += n_rec * ((int64_t) g.d_inner() * g.kda_head_dim + (int64_t) 3 * g.d_inner() * (g.d_conv - 1));
    pool_off = floats;
    floats += (int64_t) dsa.size() * 2 * (g.idx_kpool - 1) * g.idx_key;
    floats = up(floats);
    fixed = (size_t) floats * sizeof(float);
    grow.clear();
    for (const int il : dsa) {
        c.dsa_lat[(size_t) il] = floats;
        grow.push_back({(double) floats * 4.0, (double) lat_floats * 4.0, lat_bpt});
        floats = up(floats + lat_floats);
        c.dsa_pool[(size_t) il] = floats;
        grow.push_back({(double) floats * 4.0, (double) g.idx_key * max_pools * 4.0, (double) g.idx_key * 4.0 / g.idx_kpool});
        floats = up(floats + (int64_t) g.idx_key * max_pools);
    }
    c.max_ctx = ctx;
    c.bytes = fixed;
    return (uint64_t) floats * sizeof(float);
}

// STRATA_GLM_ELASTIC=1 (the fast path, CUDA): a reservation for ctx tokens with its fixed part backed and zeroed
bool Glm5Model::el_create(int64_t ctx, SeqCtx& c, std::string& err, bool back_fixed) {
#if defined(STRATA_USE_HIP)
    (void) ctx;
    (void) c;
    err = "glm elastic: not on HIP";
    return false;
#else
    const size_t gran = vmm_granule(dev_);
    if (gran == 0) {
        err = "glm elastic: no virtual memory management on CUDA" + std::to_string(dev_);
        return false;
    }
    std::vector<std::array<double, 3>> grow;
    size_t fixed = 0;
    int64_t snap_off = 0, pool_off = 0;
    const uint64_t va = el_layout_impl(g_, lat_q8_, mtp_il_, l0_, l1_, ctx, gran, c, grow, fixed, snap_off, pool_off);
    auto* e = new Elastic();
    e->dev = dev_;
    e->gran = gran;
    e->va = (size_t) va;
    e->fixed = fixed;
    e->max_ctx = ctx;
    if (vmm().reserve(&e->base, e->va, gran, 0, 0) != CUDA_SUCCESS || (back_fixed && !e->back(0, fixed, e->hfixed))) {
        if (e->base) vmm().addr_free(e->base, e->va);
        delete e;
        err = "glm elastic: CUDA" + std::to_string(dev_) + " could not reserve the context's address space or back " +
              std::to_string(fixed >> 20) + " MB of fixed state";
        return false;
    }
    for (const auto& gr : grow) {
        Elastic::Grow r;
        r.off = (size_t) gr[0];
        r.cap = (size_t) gr[1];
        r.bpt = gr[2];
        e->grow.push_back(r);
    }
    if (back_fixed) {
        cudaMemset((void*) e->base, 0, fixed);
        e->fixed_on = true;
    }
    c.state = (float*) e->base;
    c.snap = (float*) e->base + snap_off;
    c.snap_pool = (float*) e->base + pool_off;
    el_.push_back(e);
    return true;
#endif
}

// this part: sequence s's fixed part backed (zeroed) out of the budget - a slot costs nothing until a conversation is in it
bool Glm5Model::seq_activate(int s, std::string& err) {
#if defined(STRATA_USE_HIP)
    (void) s;
    (void) err;
    return true;
#else
    if (el_.empty() || s < 0 || s >= (int) el_.size()) return true;
    Elastic& e = *el_[(size_t) s];
    std::lock_guard<std::mutex> lk(el_mu_);
    if (e.fixed_on) return true;
    if (el_backed_ + e.fixed > el_budget_) {
        err = "glm elastic: CUDA" + std::to_string(dev_) + " context budget (" + std::to_string(el_budget_ >> 20) +
              " MB) has no room for another conversation's state";
        return false;
    }
    cudaSetDevice(dev_);
    if (!e.back(0, e.fixed, e.hfixed)) {
        err = "glm elastic: CUDA" + std::to_string(dev_) + " could not back a conversation's state";
        return false;
    }
    cudaMemset((void*) e.base, 0, e.fixed);
    e.fixed_on = true;
    el_backed_ += e.fixed;
    return true;
#endif
}

bool Glm5Model::seq_active(int s) const {
#if !defined(STRATA_USE_HIP)
    if (s >= 0 && s < (int) el_.size()) return el_[(size_t) s]->fixed_on;
#else
    (void) s;
#endif
    return true;
}

bool Glm5Model::seq_holds(int s) const {
#if !defined(STRATA_USE_HIP)
    if (s >= 0 && s < (int) el_.size()) return el_[(size_t) s]->tokens > 0 || (s > 0 && el_[(size_t) s]->fixed_on);
#else
    (void) s;
#endif
    return false;
}

bool Glm5Model::seq_grow(int s, int64_t tokens, std::string& err) {
#if defined(STRATA_USE_HIP)
    (void) s;
    (void) tokens;
    (void) err;
    return true;
#else
    if (el_.empty() || s < 0 || s >= (int) el_.size()) return true;
    Elastic& e = *el_[(size_t) s];
    if (tokens <= e.tokens) return true;
    if (!e.fixed_on && !seq_activate(s, err)) return false;
    static const int64_t step = [] {
        const char* v = getenv("STRATA_GLM_ELASTIC_STEP");
        return std::max<int64_t>(256, v ? std::atoll(v) : 4096);
    }();
    std::lock_guard<std::mutex> lk(el_mu_);
    if (tokens <= e.tokens) return true;
    if (tokens > e.max_ctx) {
        err = "glm elastic: " + std::to_string(tokens) + " positions exceed the context (" + std::to_string(e.max_ctx) + ")";
        return false;
    }
    // a step ahead, or just what is asked when the budget has no room for the step
    int64_t want = std::min<int64_t>(e.max_ctx, (tokens + step - 1) / step * step);
    size_t more = 0;
    for (const auto& r : e.grow) {
        const size_t need = (std::min(r.cap, (size_t) std::ceil((double) want * r.bpt)) + e.gran - 1) / e.gran * e.gran;
        if (need > r.backed) more += need - r.backed;
    }
    if (el_backed_ + more > el_budget_) want = tokens;
    cudaSetDevice(dev_);
    for (auto& r : e.grow) {
        const size_t need = std::min(r.cap, (size_t) std::ceil((double) want * r.bpt));
        const size_t need_g = (need + e.gran - 1) / e.gran * e.gran;
        while (r.backed < need_g) {
            if (el_backed_ + e.gran > el_budget_) {
                err = "glm elastic: CUDA" + std::to_string(dev_) + " context budget (" +
                      std::to_string(el_budget_ >> 20) + " MB) is used up";
                return false;
            }
            if (!e.back(r.off + r.backed, e.gran, r.h)) {
                err = "glm elastic: CUDA" + std::to_string(dev_) + " could not back context memory";
                return false;
            }
            r.backed += e.gran;
            el_backed_ += e.gran;
        }
    }
    e.tokens = want;
    return true;
#endif
}

bool Glm5Model::seq_grow_all(int s, int64_t tokens, std::string& err) {
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (!m->seq_grow(s, tokens, err)) return false;
    cudaSetDevice(dev_);
    return true;
}

bool Glm5Model::seq_can_grow(int s, int64_t tokens) const {
#if defined(STRATA_USE_HIP)
    (void) s;
    (void) tokens;
    return true;
#else
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        if (m->el_.empty()) continue;
        if (s < 0 || s >= (int) m->el_.size() || tokens > m->el_[(size_t) s]->max_ctx) return false;
        const Elastic& e = *m->el_[(size_t) s];
        size_t more = e.fixed_on ? 0 : e.fixed;
        for (const auto& r : e.grow) {
            const size_t need = std::min(r.cap, (size_t) std::ceil((double) tokens * r.bpt));
            const size_t need_g = (need + e.gran - 1) / e.gran * e.gran;
            if (need_g > r.backed) more += need_g - r.backed;
        }
        if (m->el_backed_ + more > m->el_budget_) return false;
    }
    return true;
#endif
}

// the caller has nothing of s in flight; its state from the context-sized caches on is gone (reset or reload it)
void Glm5Model::seq_release(int s, bool all) {
#if !defined(STRATA_USE_HIP)
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        if (m->el_.empty() || s < 0 || s >= (int) m->el_.size()) continue;
        std::lock_guard<std::mutex> lk(m->el_mu_);
        Elastic& e = *m->el_[(size_t) s];
        cudaSetDevice(m->dev_);
        cudaDeviceSynchronize();
        for (auto& r : e.grow) {
            if (r.backed > 0) vmm().unmap(e.base + r.off, r.backed);
            for (auto h : r.h) vmm().release(h);
            r.h.clear();
            m->el_backed_ -= r.backed;
            r.backed = 0;
        }
        e.tokens = 0;
        if (all && s > 0 && e.fixed_on) {   // the whole slot: free until a conversation is in it again
            vmm().unmap(e.base, e.fixed);
            for (auto h : e.hfixed) vmm().release(h);
            e.hfixed.clear();
            m->el_backed_ -= e.fixed;
            e.fixed_on = false;
        }
    }
    cudaSetDevice(dev_);
#else
    (void) s;
    (void) all;
#endif
}

int64_t Glm5Model::seq_backed(int s) const {
#if !defined(STRATA_USE_HIP)
    if (s >= 0 && s < (int) el_.size()) return el_[(size_t) s]->tokens;
#else
    (void) s;
#endif
    return max_ctx_;
}

void Glm5Model::el_destroy() {
#if !defined(STRATA_USE_HIP)
    if (el_.empty()) return;
    cudaSetDevice(dev_);
    cudaDeviceSynchronize();
    for (Elastic* e : el_) {
        for (auto& r : e->grow) {
            if (r.backed > 0) vmm().unmap(e->base + r.off, r.backed);
            for (auto h : r.h) vmm().release(h);
        }
        if (e->fixed_on) vmm().unmap(e->base, e->fixed);
        for (auto h : e->hfixed) vmm().release(h);
        vmm().addr_free(e->base, e->va);
        delete e;
    }
    el_.clear();
    el_backed_ = 0;
#endif
}

void Glm5Model::seq_bind(int s) {
    if (seq_.empty() || s == cur_seq_ || s < 0 || s >= (int) seq_.size()) return;
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
    cur_seq_ = s;
}

void Glm5Model::seq_bind_all(int s) {
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) m->seq_bind(s);
}

// load_pack's state layout (fast path) for a context of ctx tokens: the residual pair, then the NextN block's DSA
// caches, then this part's layers - the offsets into c's vectors, the bytes returned
uint64_t Glm5Model::seq_layout(int64_t ctx, SeqCtx& c) const {
    const Glm5Geometry& g = g_;
    const int64_t hc_dim = (int64_t) g.hc * g.n_embd;
    const int64_t max_pools = ctx / g.idx_kpool;
    int64_t floats = 2 * hc_dim;
    const int64_t lat_floats = lat_q8_ ? (int64_t) gf::lat8_rec_bytes(g.kv_lora) / 4 * ctx
                                       : (int64_t) g.kv_lora * ctx / 2;
    for (auto* v : {&c.kda_S, &c.kda_conv, &c.dsa_lat, &c.dsa_ik, &c.dsa_ig, &c.dsa_pool})
        v->assign((size_t) g.n_layers + 1, 0);
    c.ik_ring = (int) std::min<int64_t>(ctx, 8192 + 64);
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
    seq_.assign((size_t) n, SeqCtx{});
    // the first sequence's layout recomputed must be the one load_pack made, or the others' would be wrong
    SeqCtx check;
    if (el_.empty() && (seq_layout(max_ctx_, check) != state_bytes_ || check.dsa_pool != dsa_pool_ ||
                        check.kda_conv != kda_conv_ || check.dsa_lat != dsa_lat_ || check.ik_ring != ik_ring_)) {
        err = "glm multi: the state layout does not match load_pack's (this build cannot run STRATA_GLM_SEQS)";
        return false;
    }
    seq_[0].state = state_;
    uint64_t more = 0;
    if (!el_.empty()) {   // elastic: every sequence reserves the whole context, backs what it writes
        for (int s = 1; s < n; ++s) {
            SeqCtx& c = seq_[(size_t) s];
            if (!el_create(max_ctx_, c, err, false)) return false;   // backed when a conversation is admitted to it
            more += c.bytes;
        }
        std::fprintf(stderr, "glm elastic: CUDA%d %d sequences of %lld tokens each; the first's state (%.2f GB) backed, the "
                     "others' (%.2f GB each) when a conversation is in them, context caches as written, %.2f GB "
                     "budget\n", dev_, n, (long long) max_ctx_, (double) state_bytes_ / 1e9,
                     (double) (n > 1 ? more / (n - 1) : 0) / 1e9, (double) el_budget_ / 1e9);
    }
    for (int s = 1; s < n && el_.empty(); ++s) {
        SeqCtx& c = seq_[(size_t) s];
        seq_layout(ctx, c);
        float* a = nullptr;
        if (cudaMalloc(&a, c.bytes) != cudaSuccess) {
            cudaGetLastError();
            err = "pack: sequence state " + std::to_string(s) + " of " + std::to_string(n) + " (" +
                  std::to_string(c.bytes >> 20) + " MB) does not fit on CUDA" + std::to_string(dev_) +
                  " - fewer STRATA_GLM_SEQS or a smaller STRATA_GLM_SEQ_CTX";
            return false;
        }
        cudaMemset(a, 0, c.bytes);
        c.state = a;
        more += c.bytes;
    }
    // every sequence's snapshot (its recurrent states and the open key pool: the prompt cache, PAUSE) now as well:
    // made on demand, after the expert pool had taken the card, only the first found room
    const Glm5Geometry& g = g_;
    const size_t run = (size_t) g.d_inner() * g.kda_head_dim + (size_t) 3 * g.d_inner() * (g.d_conv - 1);
    int n_rec = 0;
    for (int il = l0_; il < l1_; ++il) n_rec += g.is_recr(il) ? 1 : 0;
    const size_t pool_floats = dsa_layers().size() * 2 * (size_t) ((g.idx_kpool - 1) * g.idx_key);
    for (int s = 0; s < n; ++s) {
        float*& snap = s == 0 ? snap_ : seq_[(size_t) s].snap;
        float*& snap_pool = s == 0 ? snap_pool_ : seq_[(size_t) s].snap_pool;
        if ((snap == nullptr && n_rec > 0 && cudaMalloc(&snap, (size_t) n_rec * run * sizeof(float)) != cudaSuccess) ||
            (snap_pool == nullptr && pool_floats > 0 &&
             cudaMalloc(&snap_pool, pool_floats * sizeof(float)) != cudaSuccess)) {
            cudaGetLastError();
            err = "pack: the snapshot of sequence " + std::to_string(s) + " of " + std::to_string(n) +
                  " does not fit on CUDA" + std::to_string(dev_) + " - fewer conversations at once (--parallel)";
            return false;
        }
        more += (uint64_t) ((n_rec > 0 ? (size_t) n_rec * run : 0) + pool_floats) * sizeof(float);
    }
    if (el_.empty())
        std::fprintf(stderr, "glm multi: CUDA%d %d sequence states: the first for %lld tokens (%.2f GB), %d more for %lld "
                     "(%.2f GB in all, with every sequence's snapshot)\n", dev_, n, (long long) max_ctx_,
                     (double) state_bytes_ / 1e9, n - 1, (long long) ctx, (double) more / 1e9);
    // one GPU: the sequences decode together (glm_rows.cu), their activations allocated here, before the pool
    if (n_parts_ == 1) return rows_alloc(err);
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
        for (auto& c : m->seq_) {
            if (c.hop_h != nullptr) continue;
            if (cudaHostAlloc((void**) &c.hop_h, (size_t) g.hc * g.n_embd * sizeof(float), cudaHostAllocPortable) !=
                    cudaSuccess ||
                cudaHostAlloc((void**) &c.emb_h, (size_t) g.n_embd * sizeof(float), cudaHostAllocPortable) !=
                    cudaSuccess ||
                cudaHostAlloc((void**) &c.tok_h, sizeof(int), cudaHostAllocPortable) != cudaSuccess ||
                cudaEventCreateWithFlags(&c.ev_hop, cudaEventDisableTiming) != cudaSuccess ||
                cudaEventCreateWithFlags(&c.ev_done, cudaEventDisableTiming) != cudaSuccess) {
                cudaGetLastError();
                err = "glm multi: the pipeline buffers did not allocate";
                return false;
            }
        }
    }
    cudaSetDevice(dev_);
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
    if (sp.greedy || sp.temperature <= 0.0f)
        gf::argmax(tail->sc_ + tail->sc_logits, g.n_vocab, FT->tok, FT->cs);
    else
        strata::kernels::sample_tokens(tail->sc_ + tail->sc_logits, 1, (int) g.n_vocab, nullptr, 0, sp, FT->tok,
                                       FT->cs);
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

}  // namespace strata::core
