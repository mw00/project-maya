// src/core/glm_fast_path.cu - the glm5-next FAST decode path (Glm5Model members).
//
// The same model as Glm5Model::step_layers (the correctness-first reference, STRATA_GLM_SLOW=1), run
// as ~11 fused launches per layer (strata/kernels/glm_fast.hpp) on a private non-blocking stream per
// device, with NO host synchronisation inside a token except where an expert is missing:
//
//   * every weight pointer is resolved once at load (no string lookups per layer per token);
//   * the router runs on the device and looks its experts up in a DEVICE table (layer x expert ->
//     VRAM slot pointer); an all-resident layer never waits for the host;
//   * every route is published to a host-mapped ring; a per-device SERVICE thread keeps the LFU
//     counts and, when a route has misses, reads the missing experts from the shards (pread into
//     pinned staging, in parallel), DMAs them into victim slots OF THAT LAYER on a copy stream and
//     answers through host-mapped memory - the device spins in a one-block wait kernel meanwhile.
//
// Victims come only from the requesting layer's own slot partition, so no kernel can be reading a
// slot while it is overwritten: the device is parked in that layer's wait kernel, the layer's
// resident experts are excluded, and every other layer's slots are untouched.
#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#if !defined(STRATA_USE_HIP)
#include <nvtx3/nvToolsExt.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <mutex>
#include <thread>
#include <vector>
#include <bit>
#ifndef _WIN32
#include <unistd.h>
#else
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#ifdef __linux__
#include <sched.h>
#endif

namespace gf = strata::kernels::glmf;

namespace strata::core {

using glmfast::cpu_relax;
using glmfast::Workers;

namespace glmfast {

// [off, off + len) of a shard into dst: O_DIRECT through an aligned per-thread bounce buffer when the shard has a
// direct fd (no page cache), a buffered pread otherwise, the mapping as the last resort
void read_slice(const Glm5Model::Shard& sh, uint64_t off, size_t len, uint8_t* dst) {
#ifndef _WIN32
    if (sh.fd_direct >= 0) {
        static thread_local uint8_t* bounce = nullptr;
        static thread_local size_t cap = 0;
        const uint64_t a0 = off & ~(uint64_t) 4095, a1 = (off + len + 4095) & ~(uint64_t) 4095;
        const size_t need = (size_t) (a1 - a0);
        if (cap < need) {
            std::free(bounce);
            bounce = (uint8_t*) std::aligned_alloc(4096, need);
            cap = bounce ? need : 0;
        }
        if (bounce != nullptr) {
            size_t got = 0;
            while (got < need) {
                const ssize_t r = pread(sh.fd_direct, bounce + got, need - got, (off_t) (a0 + got));
                if (r <= 0) break;   // EOF: the last aligned read is short where the file ends
                got += (size_t) r;
            }
            if (got >= (size_t) (off - a0) + len) {
                std::memcpy(dst, bounce + (off - a0), len);
                return;
            }
        }
    }
    size_t done = 0;
    while (done < len) {
        const ssize_t r = pread(sh.fd, dst + done, len - done, (off_t) (off + done));
        if (r <= 0) {
            std::memcpy(dst + done, sh.base + off + done, len - done);
            return;
        }
        done += (size_t) r;
    }
#else
    // Windows: the unbuffered handle the same way (sector-aligned offset, size and buffer), the mapping otherwise.
    // The handle is overlapped (reads from many threads run at once): each read waits on this thread's own event
    if (sh.h_direct != nullptr) {
        static thread_local uint8_t* bounce = nullptr;
        static thread_local size_t cap = 0;
        static thread_local HANDLE done_ev = CreateEventA(nullptr, TRUE, FALSE, nullptr);
        const uint64_t a0 = off & ~(uint64_t) 4095, a1 = (off + len + 4095) & ~(uint64_t) 4095;
        const size_t need = (size_t) (a1 - a0);
        if (cap < need) {
            _aligned_free(bounce);
            bounce = (uint8_t*) _aligned_malloc(need, 4096);
            cap = bounce ? need : 0;
        }
        if (bounce != nullptr && done_ev != nullptr) {
            size_t got = 0;
            while (got < need) {
                OVERLAPPED ov{};
                const uint64_t at = a0 + got;
                ov.Offset = (DWORD) at;
                ov.OffsetHigh = (DWORD) (at >> 32);
                ov.hEvent = done_ev;
                const DWORD want = (DWORD) std::min<size_t>(need - got, (size_t) 1 << 30);
                DWORD r = 0;
                if (!ReadFile((HANDLE) sh.h_direct, bounce + got, want, nullptr, &ov) &&
                    GetLastError() != ERROR_IO_PENDING) break;   // EOF: the last aligned read is short
                if (!GetOverlappedResult((HANDLE) sh.h_direct, &ov, &r, TRUE) || r == 0) break;
                got += (size_t) r;
            }
            if (got >= (size_t) (off - a0) + len) {
                std::memcpy(dst, bounce + (off - a0), len);
                return;
            }
        }
    }
    std::memcpy(dst, sh.base + off, len);
#endif
}

}  // namespace glmfast

using glmfast::read_slice;

static float* fa_take(uint8_t*& at, size_t floats) {
    float* p = (float*) at;
    at += (floats * sizeof(float) + 255u) & ~(size_t) 255u;
    return p;
}
static void* fa_take_b(uint8_t*& at, size_t bytes) {
    void* p = at;
    at += (bytes + 255u) & ~(size_t) 255u;
    return p;
}

bool Glm5Model::fast_setup(std::string& err) {
    cudaSetDevice(dev_);
    auto* F = new FastState();
    fast_ = F;
    const Glm5Geometry& g = g_;
    if (const char* rs = getenv("STRATA_GLM_RAM_SHADOW")) F->ram_shadow = std::atoi(rs) != 0;
    F->timing = getenv("STRATA_GLM_TIMING") != nullptr;
    F->prof_on = getenv("STRATA_GLM_PROF") != nullptr;
    if (F->prof_on) F->pskip = (uint64_t) std::max(0, std::atoi(getenv("STRATA_GLM_PROF")));
    if (const char* pn = getenv("STRATA_GLM_PREFETCH_N")) F->max_pf = std::max(0, std::min(gf::kSpares - 1, std::atoi(pn)));
    if (cudaStreamCreateWithFlags(&F->cs, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&F->copy, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&F->ps, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&F->ev_hop, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&F->ev_pred, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&F->ev_pf, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&F->ev_pf_prev, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&F->ev_done, cudaEventDisableTiming) != cudaSuccess) {
        err = "glm fast: streams/events did not create";
        return false;
    }

    // ---- the weights, resolved once
    const int NL = g.n_layers + 1;   // every per-layer table: the trunk + the NextN block's slot
    F->L.assign((size_t) NL, FastState::Layer{});
    std::string missing;
    for (int il = l0_; il < lt_; ++il) {
        const bool is_mtp = il == mtp_il_;
        auto& Ly = F->L[(size_t) il];
        const std::string P = "blk." + std::to_string(il) + ".";
        const auto f32 = [&](const char* n) -> const float* {
            auto it = w_.find(P + n);
            if (it == w_.end()) { missing += P + n + " (f32) "; return nullptr; }
            return it->second;
        };
        const auto b16 = [&](const char* n) -> const uint16_t* {
            auto it = w16_.find(P + n);
            if (it == w16_.end()) { missing += P + n + " (bf16) "; return nullptr; }
            return it->second;
        };
        const auto q = [&](const char* n) -> WSlot {
            auto it = ws_map_.find(P + n);
            if (it == ws_map_.end() || it->second.type == 0 || !gf::mv_supported(it->second.type)) {
                missing += P + n + " (quant) ";
                return WSlot{};
            }
            return it->second;
        };
        Ly.recr = is_mtp ? false : g.is_recr(il);
        Ly.moe = il >= g.dense_lead;
        Ly.mtp = is_mtp;
        if (is_mtp) {
            // no hyper-connections: eh_proj in, plain residuals, the shared head's norm out
            Ly.eh = q("nextn.eh_proj.weight");
            Ly.enorm = f32("nextn.enorm.weight");
            Ly.hnorm = f32("nextn.hnorm.weight");
            Ly.shnorm = f32("nextn.shared_head_norm.weight");
        } else {
            Ly.hc_attn_fn = b16("hc_attn_fn.weight");
            Ly.hc_ffn_fn = b16("hc_ffn_fn.weight");
            Ly.hc_attn_scale = f32("hc_attn_scale.weight");
            Ly.hc_attn_base = f32("hc_attn_base.weight");
            Ly.hc_ffn_scale = f32("hc_ffn_scale.weight");
            Ly.hc_ffn_base = f32("hc_ffn_base.weight");
        }
        Ly.attn_norm = f32("attn_norm.weight");
        Ly.ffn_norm = f32("ffn_norm.weight");
        if (Ly.recr) {
            Ly.q = q("attn_q.weight");
            Ly.k = q("attn_k.weight");
            Ly.v = q("attn_v.weight");
            Ly.f_a = b16("ssm_f_a.weight");
            Ly.g_a = b16("ssm_g_a.weight");
            Ly.f_b = b16("ssm_f_b.weight");
            Ly.g_b = b16("ssm_g_b.weight");
            Ly.beta = b16("ssm_beta.weight");
            Ly.conv[0] = f32("ssm_conv1d_q.weight");
            Ly.conv[1] = f32("ssm_conv1d_k.weight");
            Ly.conv[2] = f32("ssm_conv1d_v.weight");
            Ly.dt_bias = f32("ssm_dt.bias");
            Ly.ssm_a = f32("ssm_a");
            Ly.ssm_norm = f32("ssm_norm.weight");
        } else {
            Ly.q_a = q("attn_q_a.weight");
            Ly.q_b = q("attn_q_b.weight");
            Ly.kv_a = q("attn_kv_a_mqa.weight");
            Ly.q_a_norm = f32("attn_q_a_norm.weight");
            Ly.kv_a_norm = f32("attn_kv_a_norm.weight");
            Ly.k_norm_w = f32("indexer.k_norm.weight");
            Ly.k_norm_b = f32("indexer.k_norm.bias");
            Ly.ape = f32("indexer_compressor_ape.weight");
            Ly.idx_k = b16("indexer.attn_k.weight");
            Ly.idx_gate = b16("indexer_compressor_gate.weight");
            Ly.idx_q_b = b16("indexer.attn_q_b.weight");
            Ly.idx_proj = b16("indexer.proj.weight");
            Ly.k_b = b16("attn_k_b.weight");
            Ly.v_b = b16("attn_v_b.weight");
        }
        Ly.out = q("attn_output.weight");
        if (Ly.moe) {
            Ly.router = b16("ffn_gate_inp.weight");
            Ly.router_bias = f32("exp_probs_b.bias");
            Ly.sh_gate = q("ffn_gate_shexp.weight");
            Ly.sh_up = q("ffn_up_shexp.weight");
            Ly.sh_down = q("ffn_down_shexp.weight");
            const auto& nl = pack_layers_[(size_t) il];
            if (nl.layer < 0) {
                missing += P + "native experts ";
            } else {
                Ly.gu_type = nl.fmt.gu_type;
                Ly.d_type = nl.fmt.d_type;
                Ly.gu_bytes = (size_t) nl.fmt.gu_row * (size_t) g.n_ff_exp;
                Ly.dn_bytes = (size_t) nl.fmt.d_row * (size_t) g.n_embd;
                Ly.down_off = 2 * Ly.gu_bytes;
                Ly.blob = 2 * Ly.gu_bytes + Ly.dn_bytes;
                if (gf::row_bytes(Ly.gu_type, g.n_embd) != nl.fmt.gu_row ||
                    gf::row_bytes(Ly.d_type, g.n_ff_exp) != nl.fmt.d_row)
                    missing += P + "expert types " + std::to_string(Ly.gu_type) + "/" + std::to_string(Ly.d_type) + " ";
            }
        } else {
            Ly.ffn_gate = q("ffn_gate.weight");
            Ly.ffn_up = q("ffn_up.weight");
            Ly.ffn_down = q("ffn_down.weight");
        }
    }
    if (!missing.empty()) {
        err = "glm fast: unresolved weights: " + missing.substr(0, 600);
        return false;
    }

    // ---- the activation arena
    const int64_t E = g.n_embd, DI = g.d_inner(), FF = g.n_ff_exp * g.n_shared;
    const int64_t max_pools = std::max<int64_t>(1, max_ctx_ / g.idx_kpool);
    const size_t q8 = 36;   // bytes per q8_1 block
    size_t need = 0;
    {
        // generous upper bound, carved below
        need = (size_t) (4 * E + 64 + 32 * 26 + 128 + 6 * DI + 4 * 128 + 2 * DI + 2 * g.q_lora + 2 * g.kv_lora +
                         4 * g.idx_key + 64 + (int64_t) g.n_head * g.qk_nope + (int64_t) g.idx_heads * g.idx_key +
                         max_pools + g.n_sel_max() + 2 * g.n_expert + 2 * FF + 2 * g.n_ff_dense + 2 * E + 64) *
                   sizeof(float);
        need += (size_t) (E + DI + g.q_lora + (int64_t) g.n_head * g.v_head + FF + 8 * g.n_ff_exp + g.n_ff_dense + E) /
                32 * q8;
        need += 64 * 256 + (size_t) (2 * NL * g.n_expert + NL * gf::kSpares) * sizeof(unsigned long long) +
                8 * 16 + 4096 * 4 + 4 * 256 + (size_t) g.n_expert * 4 + 256 + 4 * 256;
        need += (size_t) gf::kAhead * g.n_expert * sizeof(float) + 256 + (size_t) E * sizeof(float) + 256;
    }
    if (cudaMalloc(&F->arena, need) != cudaSuccess) {
        err = "glm fast: the activation arena did not allocate";
        return false;
    }
    cudaMemset(F->arena, 0, need);
    F->arena_bytes = need;
    uint8_t* at = (uint8_t*) F->arena;
    F->x = fa_take(at, E);
    F->mixer = fa_take(at, E);
    F->ffn = fa_take(at, E);
    F->pre = fa_take(at, 8);
    F->post = fa_take(at, 8);
    F->comb = fa_take(at, 16);
    F->part = fa_take(at, (size_t) (4 * E / 512) * 26 + 64);   // 25 partials per block + the norm's per-block sums
    F->counter = (unsigned int*) fa_take_b(at, 64);
    F->xq = fa_take_b(at, (size_t) E / 32 * q8);
    for (int i = 0; i < 3; ++i) F->proj[i] = fa_take(at, DI);
    for (int i = 0; i < 3; ++i) F->conv[i] = fa_take(at, DI);
    F->fa = fa_take(at, g.kda_head_dim);
    F->ga = fa_take(at, g.kda_head_dim);
    F->beta = fa_take(at, g.n_head);
    F->g1 = fa_take(at, DI);
    F->g2 = fa_take(at, DI);
    F->gated_q = fa_take_b(at, (size_t) DI / 32 * q8);
    F->qr_raw = fa_take(at, g.q_lora);
    F->qr = fa_take(at, g.q_lora);
    F->qr_q = fa_take_b(at, (size_t) g.q_lora / 32 * q8);
    F->kv_raw = fa_take(at, g.kv_lora);
    F->ik_raw = fa_take(at, g.idx_key);
    F->ig_raw = fa_take(at, g.idx_key);
    F->iw = fa_take(at, g.idx_heads);
    F->q = fa_take(at, (size_t) g.n_head * g.qk_nope);
    F->iq = fa_take(at, (size_t) g.idx_heads * g.idx_key);
    F->score = fa_take(at, (size_t) max_pools);
    F->cells = (int*) fa_take(at, (size_t) g.n_sel_max());
    F->attn_q = fa_take_b(at, (size_t) g.n_head * g.v_head / 32 * q8);
    F->rlog = fa_take(at, g.n_expert);
    F->plog = fa_take(at, g.n_expert);
    F->alog = fa_take(at, (size_t) gf::kAhead * g.n_expert);
    F->sh_out = fa_take(at, E);
    F->sh_g = fa_take(at, FF);
    F->sh_u = fa_take(at, FF);
    F->sh_hq = fa_take_b(at, (size_t) FF / 32 * q8);
    F->hq = fa_take_b(at, (size_t) 8 * g.n_ff_exp / 32 * q8);
    F->dg = fa_take(at, g.n_ff_dense);
    F->du = fa_take(at, g.n_ff_dense);
    F->dhq = fa_take_b(at, (size_t) g.n_ff_dense / 32 * q8);
    F->head_x = fa_take(at, E);
    F->head_xq = fa_take_b(at, (size_t) E / 32 * q8);
    F->emb = fa_take(at, E);
    F->tok = (int*) fa_take_b(at, 64);
    F->n_keys = NL * g.n_expert;
    F->tab = (unsigned long long*) fa_take_b(
        at, (size_t) (2 * F->n_keys + NL * gf::kSpares) * sizeof(unsigned long long));
    F->md.tab = F->tab;
    F->md.n_keys = F->n_keys;
    F->md.scratch = (unsigned long long*) fa_take_b(at, 8 * sizeof(unsigned long long));
    F->md.fetch_src = (unsigned long long*) fa_take_b(at, 8 * sizeof(unsigned long long));
    for (int b2 = 0; b2 < 2; ++b2) {
        F->pf_buf[b2] = (unsigned long long*) fa_take_b(at, 16 * sizeof(unsigned long long));
        F->pf_n_buf[b2] = (int*) fa_take_b(at, 64);
    }
    F->md.plan_ptr = (unsigned long long*) fa_take_b(at, 8 * sizeof(unsigned long long));
    F->md.plan_w = fa_take(at, 8);
    F->md.plan_id = (int*) fa_take(at, 8);
    F->md.seq = (unsigned int*) fa_take_b(at, 64);
    F->md.wait_seq = (unsigned int*) fa_take_b(at, 64);
    F->md.cpu_part = fa_take(at, E);
    F->md.cpu_flag = (int*) fa_take_b(at, 64);
    if ((size_t) (at - (uint8_t*) F->arena) > need) {
        err = "glm fast: the activation arena overflowed its estimate";
        return false;
    }
    if (mtp_il_ >= 0) {
        if (cudaMalloc(&F->mtp_h, (size_t) g.n_embd * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&F->mtp_catq, (size_t) 2 * g.n_embd / 32 * 36) != cudaSuccess ||
            cudaMalloc(&F->mtp_logits, (size_t) g.n_vocab * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&F->mtp_tok, 64) != cudaSuccess ||
            cudaHostAlloc((void**) &F->mtp_tok_h, 64, cudaHostAllocDefault) != cudaSuccess ||
            cudaEventCreateWithFlags(&F->ev_mtp, cudaEventDisableTiming) != cudaSuccess) {
            err = "glm fast: the NextN block's buffers did not allocate";
            return false;
        }
    }

    // ---- host-mapped routing ring + response, pinned staging for the embedding / hop / token
    void* hp = nullptr;
    if (cudaHostAlloc(&hp, sizeof(gf::MoeRequest) * gf::kRingSize, cudaHostAllocMapped) != cudaSuccess) {
        err = "glm fast: the routing ring did not allocate";
        return false;
    }
    std::memset(hp, 0, sizeof(gf::MoeRequest) * gf::kRingSize);
    F->ring_h = (gf::MoeRequest*) hp;
    void* dp = nullptr;
    cudaHostGetDevicePointer(&dp, hp, 0);
    F->md.ring = dp;
    if (cudaHostAlloc(&hp, sizeof(gf::MoeResponse), cudaHostAllocMapped) != cudaSuccess) {
        err = "glm fast: the routing response did not allocate";
        return false;
    }
    std::memset(hp, 0, sizeof(gf::MoeResponse));
    F->resp_h = (gf::MoeResponse*) hp;
    cudaHostGetDevicePointer(&dp, hp, 0);
    F->md.resp = dp;
    if (cudaHostAlloc((void**) &F->emb_h, (size_t) E * sizeof(float), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**) &F->hop_h, (size_t) g.hc * E * sizeof(float), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**) &F->tok_h, 64, cudaHostAllocDefault) != cudaSuccess) {
        err = "glm fast: pinned staging did not allocate";
        return false;
    }

    // ---- the expert pool: per-layer partitions (equal quota per layer within its size class)
    size_t blob_max = 0;
    int n_moe = 0;
    for (int il = l0_; il < lt_; ++il)
        if (F->L[(size_t) il].moe) {
            blob_max = std::max(blob_max, F->L[(size_t) il].blob);
            ++n_moe;
        }
    F->lp.assign((size_t) NL, FastState::LayerPool{});
    F->slot_of.assign((size_t) NL * g.n_expert, -1);
    if (n_moe > 0) {
        // the scratch slots: a fetched expert that found no spare lands here (8 = one per routed entry)
        const size_t sstride = (blob_max + 255u) & ~(size_t) 255u;
        if (cudaMalloc(&F->scratch, 8 * sstride) != cudaSuccess) {
            err = "glm fast: the scratch slots did not allocate";
            return false;
        }
        {
            unsigned long long sp[8];
            for (int i = 0; i < 8; ++i) sp[i] = (unsigned long long) (F->scratch + (size_t) i * sstride);
            cudaMemcpy(F->md.scratch, sp, sizeof sp, cudaMemcpyHostToDevice);
        }
        // the batched prompt path: its sizes (it borrows the pool's tail, see below)
        if (!prefill_setup(err)) return false;
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        {
            const double G = 1073741824.0, used = (double) (total_b - free_b);
            const double known = (double) (w_arena_bytes_ + state_bytes_ + sc_bytes_ + F->arena_bytes + 8 * sstride);
            std::fprintf(stderr, "glm fast: CUDA%d VRAM before the expert pool: %.2f of %.2f GB in use - dense weights %.2f, "
                                 "state/KV (%lld ctx) %.2f, activations %.2f, other (context, MTP, scratch, prompt path) "
                                 "%.2f\n", dev_, used / G, (double) total_b / G, (double) w_arena_bytes_ / G,
                         (long long) max_ctx_, (double) state_bytes_ / G,
                         (double) (sc_bytes_ + F->arena_bytes + 8 * sstride) / G, (used - known) / G);
        }
        // headroom: cuBLAS-free path; the context's own growth is already allocated (state arena);
        // keep ~700 MB for the driver, the sampler and kernel launches' local memory
        size_t reserve = (size_t) 700 << 20;
        if (const char* r = getenv("STRATA_GLM_RESERVE_MB")) reserve = (size_t) std::atoll(r) << 20;
        size_t avail = free_b > reserve ? free_b - reserve : 0;
        if (const char* cap = getenv("STRATA_GLM_POOL_GB"))
            avail = std::min(avail, (size_t) (std::atof(cap) * 1073741824.0));
        // STRATA_GLM_VRAM_GB=<n>: behave like a card with n GB - the cap counts everything this process already
        // holds on the device (context, dense weights, state, arenas), so the pool gets what such a card would have
        if (const char* vc = getenv("STRATA_GLM_VRAM_GB")) {
            const size_t cap = (size_t) (std::atof(vc) * 1073741824.0);
            const size_t used = total_b - free_b;
            const size_t room = cap > used + reserve ? cap - used - reserve : 0;
            std::fprintf(stderr, "glm fast: CUDA%d VRAM cap %.1f GB: %.2f GB already in use, %.2f GB left for experts\n",
                         dev_, (double) cap / 1073741824.0, (double) used / 1073741824.0, (double) room / 1073741824.0);
            avail = std::min(avail, room);
        }
        // a layer's slot stride: MMQ-addressable (glmfast::expert_stride), so the prompt path multiplies the pool
        // partition in place
        const auto slot_stride = [&](int il) {
            const auto& Ly = F->L[(size_t) il];
            return glmfast::expert_stride(Ly.blob, Ly.gu_type, Ly.d_type);
        };
        // every layer gets the same number of slots: avail / sum(blob over layers)
        size_t blob_sum = 0;
        for (int il = l0_; il < lt_; ++il)
            if (F->L[(size_t) il].moe) blob_sum += F->L[(size_t) il].blob;
        int per = (int) std::min<size_t>((size_t) g.n_expert, avail / std::max<size_t>(1, blob_sum));
        // keep a whole number of 256-byte-aligned blobs
        while (per > 0) {
            size_t tot = 0;
            for (int il = l0_; il < lt_; ++il)
                if (F->L[(size_t) il].moe) tot += (size_t) per * slot_stride(il);
            if (tot <= avail) break;
            --per;
        }
        if (per < g.n_exp_used) {
            err = "glm fast: not enough VRAM for the expert pool (" + std::to_string(per) + " slots per layer)";
            return false;
        }
        size_t stride_sum = 0;
        for (int il = l0_; il < lt_; ++il)
            if (F->L[(size_t) il].moe) stride_sum += slot_stride(il);
        // the prompt path borrows the pool's TAIL: the last k slots of every layer, laid out as one region after all
        // the main segments - the prompt path's buffers while a prompt runs, expert slots the rest of the time
        int k_extra = 0;
        const auto tail_slots = [&](size_t bytes) {
            return (int) ((bytes + stride_sum - 1) / std::max<size_t>(1, stride_sum));
        };
        if (pf_ != nullptr) {
            k_extra = tail_slots(prefill_borrow_bytes());
            if (per - k_extra < g.n_exp_used + gf::kSpares + 8) {
                std::fprintf(stderr, "glm prefill: CUDA%d the pool (%d slots/layer) cannot lend %zu MB - token by token\n",
                             dev_, per, prefill_borrow_bytes() >> 20);
                prefill_destroy();
                k_extra = 0;
            }
        }
        // an on-demand vision encoder on the first GPU (the server sets STRATA_GLM_VISION_LEND_MB) borrows the same
        // tail while it encodes, so the tail there is the larger of the two needs
        vis_lend_ok_ = false;
        if (const char* v = getenv("STRATA_GLM_VISION_LEND_MB"); v != nullptr && dev_ == 0) {
            const size_t vb = (size_t) std::max(0LL, std::atoll(v)) << 20;
            if (vb > 0 && per - tail_slots(vb) >= g.n_exp_used + gf::kSpares + 8) {
                k_extra = std::max(k_extra, tail_slots(vb));
                vis_lend_ok_ = true;
            } else if (vb > 0) {
                std::fprintf(stderr, "glm fast: CUDA%d the pool (%d slots/layer) cannot lend %zu MB to the vision "
                                     "encoder\n", dev_, per, vb >> 20);
            }
        }
        // the slots PER LAYER: uniform, or - with the pack's expert_counts.txt (tools/glm_expert_prior.py) - by routing
        // share: every layer gets a floor, then each further slot goes to the layer whose next-most-routed expert serves
        // the largest share of its lookups (early layers spread their routing wide and gain the most).
        // STRATA_GLM_UNIFORM_SLOTS=1 keeps the uniform split (A/B).
        std::vector<int> nsl((size_t) NL, 0);
        for (int il = l0_; il < lt_; ++il)
            if (F->L[(size_t) il].moe) nsl[(size_t) il] = per;
        if (getenv("STRATA_GLM_UNIFORM_SLOTS") == nullptr) {
            std::vector<std::vector<double>> share((size_t) NL);
            std::ifstream cf(pack_dir_ + "/expert_counts.txt");
            std::string line;
            while (std::getline(cf, line)) {
                std::istringstream ss(line);
                int il = -1;
                ss >> il;
                if (il < l0_ || il >= lt_ || !F->L[(size_t) il].moe) continue;
                std::vector<double> c;
                double v = 0, tot_c = 0;
                while (ss >> v) {
                    c.push_back(v);
                    tot_c += v;
                }
                if (tot_c <= 0) continue;
                std::sort(c.rbegin(), c.rend());
                for (double& x : c) x /= tot_c;
                share[(size_t) il] = std::move(c);
            }
            // a layer the counts do not cover (the NextN block) takes the mean profile of the others
            {
                std::vector<double> mean;
                int nm = 0;
                for (int il = l0_; il < lt_; ++il) {
                    const auto& sh = share[(size_t) il];
                    if (sh.empty()) continue;
                    if (mean.size() < sh.size()) mean.resize(sh.size(), 0.0);
                    for (size_t j = 0; j < sh.size(); ++j) mean[j] += sh[j];
                    ++nm;
                }
                if (nm > 0) {
                    for (double& x : mean) x /= nm;
                    for (int il = l0_; il < lt_; ++il)
                        if (F->L[(size_t) il].moe && share[(size_t) il].empty()) share[(size_t) il] = mean;
                }
            }
            bool all = true;
            for (int il = l0_; il < lt_; ++il)
                if (F->L[(size_t) il].moe && share[(size_t) il].empty()) all = false;
            if (all) {
                const int cap = g.n_expert + gf::kSpares;
                const int fl = std::min(cap, g.n_exp_used + gf::kSpares + k_extra + 16);
                size_t used = 0;
                for (int il = l0_; il < lt_; ++il)
                    if (F->L[(size_t) il].moe) {
                        nsl[(size_t) il] = fl;
                        used += (size_t) fl * slot_stride(il);
                    }
                // the value of layer l's next slot: the routing share of its next expert (spares past the 288th serve none)
                const auto gain = [&](int il) {
                    const auto& sh = share[(size_t) il];
                    const int j = nsl[(size_t) il] - gf::kSpares;
                    return (j >= 0 && j < (int) sh.size()) ? sh[(size_t) j] : 0.0;
                };
                for (;;) {
                    int best = -1;
                    double bg = -1.0;
                    for (int il = l0_; il < lt_; ++il) {
                        if (!F->L[(size_t) il].moe || nsl[(size_t) il] >= cap) continue;
                        if (used + slot_stride(il) > avail) continue;
                        const double gv = gain(il);
                        if (gv > bg) {
                            bg = gv;
                            best = il;
                        }
                    }
                    if (best < 0) break;
                    ++nsl[(size_t) best];
                    used += slot_stride(best);
                }
            }
        }
        size_t tot = 0, xtot = 0;
        int nmin = INT32_MAX, nmax = 0;
        for (int il = l0_; il < lt_; ++il)
            if (F->L[(size_t) il].moe) {
                tot += (size_t) nsl[(size_t) il] * slot_stride(il);
                xtot += (size_t) k_extra * slot_stride(il);
                nmin = std::min(nmin, nsl[(size_t) il]);
                nmax = std::max(nmax, nsl[(size_t) il]);
            }
        if (cudaMalloc(&F->pool, tot - xtot) != cudaSuccess ||
            (xtot > 0 && cudaMalloc(&F->xpool, xtot) != cudaSuccess)) {
            err = "glm fast: the expert pool (" + std::to_string(tot >> 20) + " MB) did not allocate";
            return false;
        }
        F->pool_bytes = tot;
        F->xpool_bytes = xtot;
        uint8_t* b = F->pool;
        for (int il = l0_; il < lt_; ++il) {
            if (!F->L[(size_t) il].moe) continue;
            auto& P = F->lp[(size_t) il];
            const int nl = nsl[(size_t) il];
            P.stride = slot_stride(il);
            P.n = nl;
            P.n_main = nl - k_extra;
            P.base = b;
            P.key.assign((size_t) nl, -1);
            P.tick.assign((size_t) nl, 0);
            P.st.assign((size_t) nl, FastState::kFree);
            b += (size_t) P.n_main * P.stride;
            F->pool_slots += nl;
        }
        uint8_t* xregion = F->xpool;
        b = xregion;
        for (int il = l0_; il < lt_; ++il) {
            if (!F->L[(size_t) il].moe) continue;
            auto& P = F->lp[(size_t) il];
            P.xbase = b;
            b += (size_t) k_extra * P.stride;
        }
        if (pf_ != nullptr && !prefill_bind(xregion, xtot, err)) return false;
        if (vis_lend_ok_)
            std::fprintf(stderr, "glm fast: CUDA%d the vision encoder borrows the pool's tail (%.2f GB) while it "
                                 "encodes\n", dev_, (double) xtot / 1073741824.0);
        F->cnt.assign((size_t) NL * g.n_expert, 0);
        F->usage.assign((size_t) NL * g.n_expert, 0);
        F->pred_of.assign((size_t) NL, std::array<int, 8>{-1, -1, -1, -1, -1, -1, -1, -1});
        std::fprintf(stderr, "glm fast: CUDA%d layers [%d,%d) expert pool %.2f GB, %d-%d slots/layer (%lld total)\n",
                     dev_, l0_, l1_, (double) tot / 1073741824.0, nmin, nmax, (long long) F->pool_slots);
        if (getenv("STRATA_GLM_TIMING") != nullptr) {
            std::string sl;
            for (int il = l0_; il < lt_; ++il)
                if (F->L[(size_t) il].moe) sl += std::to_string(il) + ":" + std::to_string(nsl[(size_t) il]) + " ";
            std::fprintf(stderr, "glm fast: CUDA%d slots per layer %s\n", dev_, sl.c_str());
        }

        // ---- the RAM tier: one pinned arena per blob size class, slots in proportion to the class's
        //      share of this half's expert bytes NOT already held by the VRAM pool
        F->layer_rc.assign((size_t) NL, -1);
        F->ram_of.assign((size_t) NL * g.n_expert, -1);
        F->left.assign((size_t) NL * g.n_expert, 0);
        std::vector<size_t> cls_stride;
        std::vector<double> cls_weight;
        for (int il = l0_; il < lt_; ++il) {
            if (!F->L[(size_t) il].moe) continue;
            const size_t st = (F->L[(size_t) il].blob + 4095u) & ~(size_t) 4095u;
            int c = -1;
            for (size_t k = 0; k < cls_stride.size(); ++k)
                if (cls_stride[k] == st) c = (int) k;
            if (c < 0) {
                c = (int) cls_stride.size();
                cls_stride.push_back(st);
                cls_weight.push_back(0.0);
            }
            F->layer_rc[(size_t) il] = c;
            cls_weight[(size_t) c] +=
                (double) std::max(2, g.n_expert - F->lp[(size_t) il].n + gf::kSpares + 2) * (double) st;
        }
        double wsum = 0;
        for (double w : cls_weight) wsum += w;
        int64_t budget = ram_budget_;
        // the whole machine's budget is measured ONCE (the first half's pinning would shrink what the second
        // half sees): MemAvailable minus headroom for the OS, the server and the page cache the disk reads go
        // through; STRATA_GLM_RAM_GB pins it.  Each half takes its share of the MoE layers still unserved,
        // and whatever a half cannot use (its experts already fit) passes on to the next half.
        static int64_t total = -1, remaining = -1;
        static int remaining_layers = 0;
        if (budget < 0) {
            if (total < 0) {
                if (const char* rg = getenv("STRATA_GLM_RAM_GB")) {
                    total = (int64_t) (std::atof(rg) * 1073741824.0);
                } else {
                    int64_t avail_kb = 0;
#ifdef _WIN32
                    MEMORYSTATUSEX ms{};
                    ms.dwLength = sizeof ms;
                    // pinned RAM is charged to the commit (RAM + page file), and under WDDM so is every allocation
                    // on the card - the pool above already took its share: the smaller of the two is what can pin
                    if (GlobalMemoryStatusEx(&ms))
                        avail_kb = (int64_t) (std::min(ms.ullAvailPhys, ms.ullAvailPageFile) >> 10);
#else
                    if (FILE* mf = std::fopen("/proc/meminfo", "r")) {
                        char line[256];
                        while (std::fgets(line, sizeof line, mf))
                            if (std::sscanf(line, "MemAvailable: %lld kB", (long long*) &avail_kb) == 1) break;
                        std::fclose(mf);
                    }
#endif
                    double head_gb = 6.0;
                    if (const char* h = getenv("STRATA_GLM_RAM_HEADROOM_GB")) head_gb = std::atof(h);
                    total = std::max<int64_t>(0, avail_kb * 1024 - (int64_t) (head_gb * 1073741824.0));
                }
                remaining = total;
                remaining_layers = g.n_layers - g.dense_lead + (getenv("STRATA_GLM_NO_MTP") ? 0 : g.nextn);
            }
            // the last half takes everything left (one device: the NextN block, counted above, is not loaded)
            budget = l1_ >= g.n_layers ? remaining
                                       : std::min<int64_t>(remaining, (int64_t) ((double) remaining * (double) n_moe /
                                                                                 (double) std::max(1, remaining_layers)));
        }
        F->rc.assign(cls_stride.size(), FastState::RamClass{});
        for (size_t c = 0; c < cls_stride.size() && wsum > 0; ++c) {
            auto& R = F->rc[c];
            R.stride = cls_stride[c];
            int64_t n = (int64_t) ((double) budget * cls_weight[c] / wsum / (double) R.stride);
            n = std::max<int64_t>(n, 16);   // a floor so a miss always has somewhere to land
            // never more than the experts of the class that the VRAM pool does not hold.  When the pool holds every
            // expert (a big card: 288 of 288 slots a layer) that is a few spares' worth - below the floor for a class
            // of one layer (the NextN block), which then never allocated and stopped the start ("did not allocate")
            const int64_t cap = std::max<int64_t>(1, (int64_t) (cls_weight[c] / (double) R.stride));
            n = std::min<int64_t>(n, cap);
            const int64_t least = std::min<int64_t>(16, cap);   // the floor, or the whole cap when that is smaller
            void* p = nullptr;
            while (n >= least && cudaHostAlloc(&p, (size_t) n * R.stride, cudaHostAllocPortable) != cudaSuccess) {
                cudaGetLastError();
                p = nullptr;
                n = (n > least && n * 7 / 8 < least) ? least : n * 7 / 8;
            }
            if (p == nullptr) {
                err = "glm fast: the pinned RAM tier did not allocate";
                return false;
            }
            R.base = (uint8_t*) p;
            R.n = (int) n;
            R.key.assign((size_t) n, -1);
            R.st.assign((size_t) n, FastState::kRFree);
            R.tick.assign((size_t) n, 0);
            F->ram_bytes += (size_t) n * R.stride;
        }
        if (ram_budget_ < 0 && remaining >= 0) {
            remaining = std::max<int64_t>(0, remaining - (int64_t) F->ram_bytes);
            remaining_layers -= n_moe;
        }
        int64_t ram_slots = 0;
        for (auto& R : F->rc) ram_slots += R.n;
        std::fprintf(stderr, "glm fast: CUDA%d RAM tier %.2f GB pinned, %lld slots\n", dev_,
                     (double) F->ram_bytes / 1073741824.0, (long long) ram_slots);
        if (cudaHostAlloc((void**) &F->upd_key_h, FastState::kMaxUpd * sizeof(int), cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc((void**) &F->upd_val_h, FastState::kMaxUpd * sizeof(unsigned long long),
                          cudaHostAllocDefault) != cudaSuccess ||
            cudaMalloc((void**) &F->upd_key_d, FastState::kMaxUpd * sizeof(int)) != cudaSuccess ||
            cudaMalloc((void**) &F->upd_val_d, FastState::kMaxUpd * sizeof(unsigned long long)) != cudaSuccess) {
            err = "glm fast: the table update buffers did not allocate";
            return false;
        }
        F->upd_at.assign((size_t) (2 * F->n_keys + NL * gf::kSpares), 0);
        F->upd_gen.assign(F->upd_at.size(), 0u);
        // disk reads: three slices per expert in parallel; pinned staging for when the RAM tier has no free slot
        F->workers.reset(new Workers(8));
        F->stage.assign((size_t) g.n_exp_used, nullptr);
        for (auto& sbuf : F->stage)
            if (cudaHostAlloc((void**) &sbuf, sstride, cudaHostAllocPortable) != cudaSuccess) {
                err = "glm fast: pinned disk staging did not allocate";
                return false;
            }
        // LOOKAHEAD: the predictions per layer, a load state per RAM slot, and the reader threads
        if (const char* ah = getenv("STRATA_GLM_AHEAD")) F->n_ahead = std::max(0, std::min(gf::kAhead, std::atoi(ah)));
        if (g.n_expert > 512) F->n_ahead = 0;
        const char* ahr = getenv("STRATA_GLM_AHEAD_READ");
        F->ahead_read = F->n_ahead > 0 && (ahr == nullptr || std::atoi(ahr) != 0);
        std::array<std::array<short, 8>, gf::kAhead> none;
        for (auto& r : none) r.fill(-1);
        F->ah_pred.assign((size_t) NL, none);
        for (auto& R : F->rc) {
            F->rload.emplace_back(new std::atomic<uint64_t>[(size_t) R.n]);
            for (int s = 0; s < R.n; ++s) F->rload.back()[(size_t) s].store(0);
        }
        if (F->ahead_read)
            for (int i = 0; i < 2; ++i) F->ah_th.emplace_back([this] { fast_ahead_reader(); });
        const char* wv = getenv("STRATA_GLM_WARM");
        if ((wv == nullptr || std::atoi(wv) != 0) && !fast_warm(err)) return false;
        if (!fast_cpu_lane_setup(err)) return false;
        F->svc = std::thread([this] { fast_service(); });
    }
    return true;
}

// ---------------------------------------------------------------- the CPU lane
// ne RAM-tier experts of layer il (blob[i]: [gate | up | down] as in the tiers) on the CPU pool: out = sum_i w[i] *
// expert_i(x), in i order.  Gate/up rows then down rows, each split in row chunks across the pool and the caller.
void Glm5Model::fast_cpu_experts(int il, int ne, const uint8_t* const* blob, const float* w, const float* x, float* out) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const auto& Ly = F->L[(size_t) il];
    const auto& nf = F->cpu_fmt[(size_t) il];
    namespace kc = strata::kernels::cpu;
    const int n_ff = g.n_ff_exp, n_embd = g.n_embd;
    constexpr int kGuRows = 64, kDnRows = 128;
    kc::native_quant_act(nf, x, F->cpu_act.data());
    const int gu_chunks = n_ff / kGuRows;
    F->cpu_pool->run(ne * gu_chunks, [&](int job) {
        const int i = job / gu_chunks, r0 = (job % gu_chunks) * kGuRows;
        const void* a[1] = {F->cpu_act.data()};
        float* o[1] = {F->cpu_ff.data() + (size_t) i * n_ff};
        kc::native_gu_rows_split(nf, blob[i], blob[i] + Ly.gu_bytes, a, 1, o, r0, r0 + kGuRows, g.swiglu_exp);
    });
    for (int i = 0; i < ne; ++i)
        kc::native_quant_h(nf, F->cpu_ff.data() + (size_t) i * n_ff, F->cpu_hq.data() + (size_t) i * kc::kNativeHBytes);
    const int dn_chunks = n_embd / kDnRows;
    F->cpu_pool->run(dn_chunks, [&](int job) {
        const int r0 = job * kDnRows, r1 = r0 + kDnRows;
        for (int i = 0; i < ne; ++i) {
            const void* h[1] = {F->cpu_hq.data() + (size_t) i * kc::kNativeHBytes};
            float* o[1] = {F->cpu_dn.data() + (size_t) i * n_embd};
            kc::native_down_rows_split(nf, blob[i] + Ly.down_off, h, 1, o, r0, r1);
        }
        for (int r = r0; r < r1; ++r) {
            float m = 0.0f;
            for (int i = 0; i < ne; ++i) m += w[i] * F->cpu_dn[(size_t) i * n_embd + r];
            out[r] = m;
        }
    });
    // STRATA_GLM_CPU_LANE_VERIFY=1 (debug, slow): the same experts in double precision from ggml's dequantised rows;
    // the relative L2 distance of the lane's sum, accumulated and printed every 16 calls
    static const bool verify = getenv("STRATA_GLM_CPU_LANE_VERIFY") != nullptr;
    if (verify) {
        static double e2 = 0, n2 = 0, worst = 0;
        static int calls = 0;
        const ggml_type_traits* tg = ggml_get_type_traits((ggml_type) Ly.gu_type);
        const ggml_type_traits* td = ggml_get_type_traits((ggml_type) Ly.d_type);
        std::vector<float> row((size_t) std::max(n_embd, n_ff));
        std::vector<double> hr((size_t) n_ff), ref((size_t) n_embd, 0.0);
        const double lim = g.swiglu_exp;
        for (int i = 0; i < ne; ++i) {
            for (int r = 0; r < n_ff; ++r) {
                double gs = 0.0, us = 0.0;
                tg->to_float(blob[i] + (size_t) r * nf.gu_row, row.data(), n_embd);
                for (int c = 0; c < n_embd; ++c) gs += (double) row[(size_t) c] * x[c];
                tg->to_float(blob[i] + Ly.gu_bytes + (size_t) r * nf.gu_row, row.data(), n_embd);
                for (int c = 0; c < n_embd; ++c) us += (double) row[(size_t) c] * x[c];
                gs = std::min(gs, lim);
                us = std::min(std::max(us, -lim), lim);
                hr[(size_t) r] = gs / (1.0 + std::exp(-gs)) * us;
            }
            for (int r = 0; r < n_embd; ++r) {
                double s = 0.0;
                td->to_float(blob[i] + Ly.down_off + (size_t) r * nf.d_row, row.data(), n_ff);
                for (int j = 0; j < n_ff; ++j) s += (double) row[(size_t) j] * hr[(size_t) j];
                ref[(size_t) r] += w[i] * s;
            }
        }
        double a = 0, b = 0;
        for (int r = 0; r < n_embd; ++r) {
            a += (out[r] - ref[(size_t) r]) * (out[r] - ref[(size_t) r]);
            b += ref[(size_t) r] * ref[(size_t) r];
        }
        e2 += a;
        n2 += b;
        worst = std::max(worst, std::sqrt(a / std::max(b, 1e-30)));
        if (++calls % 16 == 0)
            std::fprintf(stderr, "cpu lane verify (%d calls, layer %d): relative L2 error %.5f, worst call %.5f\n", calls, il,
                         std::sqrt(e2 / n2), worst);
    }
}

// This process's physical cores (distinct package/core pairs among the CPUs it may run on); 0 when unknown.
static int physical_cores() {
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) != 0) return 0;
    std::vector<std::pair<int, int>> seen;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &set)) continue;
        const std::string b = "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/";
        int pk = -1, co = -1;
        std::ifstream(b + "physical_package_id") >> pk;
        std::ifstream(b + "core_id") >> co;
        if (co < 0) return 0;
        if (std::find(seen.begin(), seen.end(), std::make_pair(pk, co)) == seen.end()) seen.push_back({pk, co});
    }
    return (int) seen.size();
#elif defined(_WIN32)
    // Windows: the processor cores (each one entry, whatever its SMT threads)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) return 0;
    std::vector<uint8_t> buf(len);
    auto* info = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) buf.data();
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len)) return 0;
    int cores = 0;
    for (DWORD at = 0; at < len;) {
        auto* e = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) (buf.data() + at);
        if (e->Relationship == RelationProcessorCore) ++cores;
        at += e->Size;
    }
    return cores;
#else
    return 0;
#endif
}

// The CPU LANE, on by default: one thread per physical core (split across the halves of a two-GPU split);
// STRATA_GLM_CPU_LANE=<threads> sets the count, 0 turns it off.  Measures this machine once - one expert on the CPU
// pool vs one over PCIe - and derives the split: of f RAM-tier experts in a route, the k the host computes so that
// the slower of the two lanes finishes first (STRATA_GLM_CPU_PLAN=<digits for f = 0..8> overrides).  A CPU slower
// than the PCIe link at every f leaves the lane off.
bool Glm5Model::fast_cpu_lane_setup(std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const char* lv = getenv("STRATA_GLM_CPU_LANE");
    int threads = 0;
    if (lv != nullptr) {
        threads = std::max(0, std::min(64, std::atoi(lv)));
    } else {
        const bool split = l0_ > 0 || l1_ < g.n_layers;   // this half's pool shares the CPU with the other's
        threads = physical_cores();
        if (threads <= 0) threads = (int) std::thread::hardware_concurrency() / 2;
        if (split) threads /= 2;
        if (threads < 2) threads = 0;   // one core is the service thread's
    }
    if (threads <= 0 || g.n_embd > 4096 || g.n_exp_used > 8) return true;
    namespace kc = strata::kernels::cpu;
    F->cpu_fmt.assign(F->L.size(), kc::NativeFmt{});
    int il_cal = -1;
    for (int il = l0_; il < lt_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        if (!Ly.moe || Ly.mtp) continue;
        std::string e2;
        kc::NativeFmt f;
        if (kc::native_fmt(Ly.gu_type, Ly.d_type, g.n_embd, g.n_ff_exp, f, e2) && f.h_bytes <= kc::kNativeHBytes)
            F->cpu_fmt[(size_t) il] = f;
        if (il_cal < 0 && F->cpu_fmt[(size_t) il].n_ff > 0) il_cal = il;
    }
    if (il_cal < 0) return true;
    if (const char* ck = getenv("STRATA_GLM_CPU_LANE_CHECK"))   // the check's layer (below)
        if (std::atoi(ck) > 0 && std::atoi(ck) < (int) F->cpu_fmt.size() && F->cpu_fmt[(size_t) std::atoi(ck)].n_ff > 0)
            il_cal = std::atoi(ck);
    // a calibration blob: a RAM-tier expert of that layer (the warm-up filled the tier; a class holds every layer of
    // its blob size)
    const auto& R = F->rc[(size_t) F->layer_rc[(size_t) il_cal]];
    const uint8_t* cal = nullptr;
    for (int s = 0; s < R.n && cal == nullptr; ++s)
        if (R.st[(size_t) s] == FastState::kRHold && R.key[(size_t) s] / g.n_expert == il_cal)
            cal = R.base + (size_t) s * R.stride;
    if (cal == nullptr) return true;   // nothing in RAM: the lane would never run
    void* hp = nullptr;
    if (cudaHostAlloc(&hp, sizeof(gf::CpuAnswer), cudaHostAllocMapped) != cudaSuccess ||
        cudaMalloc((void**) &F->cpu_seq_d, 64) != cudaSuccess) {
        err = "glm fast: the CPU lane's buffers did not allocate";
        return false;
    }
    std::memset(hp, 0, sizeof(gf::CpuAnswer));
    cudaMemset(F->cpu_seq_d, 0, 64);
    F->cpu_ans_h = (gf::CpuAnswer*) hp;
    void* dp = nullptr;
    cudaHostGetDevicePointer(&dp, hp, 0);
    F->cpu_act.assign(kc::kNativeActBytes, 0);
    F->cpu_hq.assign((size_t) 8 * kc::kNativeHBytes, 0);
    F->cpu_ff.assign((size_t) 8 * g.n_ff_exp, 0.0f);
    F->cpu_dn.assign((size_t) 8 * g.n_embd, 0.0f);
    // the service thread is the pool's last worker; idle workers spin 20 ms (a decode's routes come ~1 ms apart)
    F->cpu_pool.reset(new glmfast::Workers(threads - 1, 20000));
    // ---- calibration, each lane alone: the CPU after 100 ms of the same work (an idle CPU's clocks take tens of ms to
    //      ramp up - a decode keeps them up), then the mean of 16 runs
    std::vector<float> x((size_t) g.n_embd), out((size_t) g.n_embd);
    for (int i = 0; i < g.n_embd; ++i) x[(size_t) i] = 0.01f * (float) ((i * 37) % 101 - 50);
    const float w1 = 1.0f;
    for (const auto tw = std::chrono::steady_clock::now();
         std::chrono::steady_clock::now() - tw < std::chrono::milliseconds(100);)
        fast_cpu_experts(il_cal, 1, &cal, &w1, x.data(), out.data());
    double c_ms = 0.0;
    for (int rep = 0; rep < 16; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        fast_cpu_experts(il_cal, 1, &cal, &w1, x.data(), out.data());
        c_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    c_ms /= 16.0;
    const size_t blob = F->L[(size_t) il_cal].blob;
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    double p_ms = 0.0;
    for (int rep = 0; rep < 12; ++rep) {
        cudaEventRecord(e0, F->cs);
        cudaMemcpyAsync(F->scratch, cal, blob, cudaMemcpyHostToDevice, F->cs);
        cudaEventRecord(e1, F->cs);
        cudaEventSynchronize(e1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        if (rep >= 4) p_ms += ms;
    }
    p_ms /= 8.0;
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    // the split: k of f to the host minimises max(PCIe (f - k) p, lane overhead + k c); ties keep the PCIe lane
    const double over_ms = 0.04;
    unsigned long long plan = 0;
    std::string tab;
    const char* pv = getenv("STRATA_GLM_CPU_PLAN");
    for (int f = 0; f <= 8; ++f) {
        int k = 0;
        if (pv != nullptr) {
            k = f < (int) std::strlen(pv) ? std::max(0, std::min(f, pv[f] - '0')) : 0;
        } else {
            double best = p_ms * f;
            for (int kk = 1; kk <= f; ++kk) {
                const double t = std::max(p_ms * (f - kk), over_ms + c_ms * kk);
                if (t < best - 1e-9) {
                    best = t;
                    k = kk;
                }
            }
        }
        plan |= (unsigned long long) k << (4 * f);
        tab += " " + std::to_string(k);
    }
    if (plan == 0ull) {
        std::fprintf(stderr, "glm fast: CUDA%d CPU lane off: an expert %.3f ms on the CPU vs %.3f ms over PCIe\n", dev_,
                     c_ms, p_ms);
        F->cpu_pool.reset();
        return true;
    }
    F->cpu_plan = plan;
    F->md.cpu_seq = F->cpu_seq_d;
    F->md.cpu_ans = dp;
    // the route counts that pick the coldest RAM-tier experts for the host, seeded with the tiers' LFU counts
    // (STRATA_GLM_CPU_COLD=0: the last ones in route order instead)
    const char* cv = getenv("STRATA_GLM_CPU_COLD");
    if ((cv == nullptr || std::atoi(cv) != 0) && F->cnt.size() == (size_t) F->n_keys) {
        if (cudaMalloc((void**) &F->dcnt_d, (size_t) F->n_keys * sizeof(unsigned int)) != cudaSuccess) {
            err = "glm fast: the CPU lane's route counts did not allocate";
            return false;
        }
        cudaMemcpy(F->dcnt_d, F->cnt.data(), (size_t) F->n_keys * sizeof(unsigned int), cudaMemcpyHostToDevice);
        F->md.dcnt = F->dcnt_d;
    }
    std::fprintf(stderr, "glm fast: CUDA%d CPU lane: %d threads, an expert %.3f ms on the CPU vs %.3f ms over PCIe -> "
                         "of 0..8 RAM-tier experts the CPU takes%s\n", dev_, threads, c_ms, p_ms, tab.c_str());
    // STRATA_GLM_CPU_LANE_CHECK=1: the calibration expert on one normalised input three ways - the device's decode
    // kernels, the CPU lane, and a double-precision reference from ggml's dequantised rows - and their distances
    if (getenv("STRATA_GLM_CPU_LANE_CHECK") != nullptr) {
        const int E = g.n_embd, FFE = g.n_ff_exp;
        const auto& Ly = F->L[(size_t) il_cal];
        const auto& nf = F->cpu_fmt[(size_t) il_cal];
        float *dh = nullptr, *dw = nullptr, *dx = nullptr, *dout = nullptr;
        void *dxq = nullptr, *dhq = nullptr;
        cudaMalloc(&dh, (size_t) E * 4);
        cudaMalloc(&dw, (size_t) E * 4);
        cudaMalloc(&dx, (size_t) E * 4);
        cudaMalloc(&dout, (size_t) E * 4);
        cudaMalloc(&dxq, (size_t) E / 32 * 36);
        cudaMalloc(&dhq, (size_t) 8 * FFE / 32 * 36);
        std::vector<float> h((size_t) E), ones((size_t) E, 1.0f), xc((size_t) E), og((size_t) E), oc((size_t) E);
        for (int i = 0; i < E; ++i) h[(size_t) i] = std::sin(0.71f * i) + ((i % 97) == 0 ? 6.0f : 0.0f);
        cudaMemcpy(dh, h.data(), (size_t) E * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dw, ones.data(), (size_t) E * 4, cudaMemcpyHostToDevice);
        gf::rms_q8(dh, nullptr, dw, g.norm_eps, E, dx, dxq, F->cs);
        cudaMemcpyAsync(F->scratch, cal, blob, cudaMemcpyHostToDevice, F->cs);
        const unsigned long long sp = (unsigned long long) F->scratch;
        const float one = 1.0f;
        cudaMemcpyAsync(F->md.plan_ptr, &sp, 8, cudaMemcpyHostToDevice, F->cs);
        cudaMemcpyAsync(F->md.plan_w, &one, 4, cudaMemcpyHostToDevice, F->cs);
        cudaMemsetAsync(F->md.cpu_flag, 0, 4, F->cs);
        gf::moe_gate_up(Ly.gu_type, F->md, 1, E, FFE, g.swiglu_exp, dxq, dhq, nullptr, 0, nullptr, 0, nullptr, F->cs);
        gf::moe_down(Ly.d_type, F->md, 1, E, FFE, Ly.down_off, dhq, nullptr, dout, F->cs);
        cudaMemcpyAsync(xc.data(), dx, (size_t) E * 4, cudaMemcpyDeviceToHost, F->cs);
        cudaMemcpyAsync(og.data(), dout, (size_t) E * 4, cudaMemcpyDeviceToHost, F->cs);
        cudaStreamSynchronize(F->cs);
        fast_cpu_experts(il_cal, 1, &cal, &one, xc.data(), oc.data());
        // the reference: rows dequantised by ggml, dots in double
        const ggml_type_traits* tg = ggml_get_type_traits((ggml_type) Ly.gu_type);
        const ggml_type_traits* td = ggml_get_type_traits((ggml_type) Ly.d_type);
        std::vector<float> row((size_t) std::max(E, FFE));
        std::vector<double> hr((size_t) FFE), orf((size_t) E);
        for (int r = 0; r < FFE; ++r) {
            double gs = 0.0, us = 0.0;
            tg->to_float(cal + (size_t) r * nf.gu_row, row.data(), E);
            for (int i = 0; i < E; ++i) gs += (double) row[(size_t) i] * xc[(size_t) i];
            tg->to_float(cal + Ly.gu_bytes + (size_t) r * nf.gu_row, row.data(), E);
            for (int i = 0; i < E; ++i) us += (double) row[(size_t) i] * xc[(size_t) i];
            const double lim = g.swiglu_exp;
            gs = std::min(gs, lim);
            us = std::min(std::max(us, -lim), lim);
            hr[(size_t) r] = gs / (1.0 + std::exp(-gs)) * us;
        }
        for (int r = 0; r < E; ++r) {
            double s = 0.0;
            td->to_float(cal + Ly.down_off + (size_t) r * nf.d_row, row.data(), FFE);
            for (int j = 0; j < FFE; ++j) s += (double) row[(size_t) j] * hr[(size_t) j];
            orf[(size_t) r] = s;
        }
        double nr = 0, eg = 0, ec = 0, egc = 0;
        for (int r = 0; r < E; ++r) {
            nr += orf[(size_t) r] * orf[(size_t) r];
            eg += (og[(size_t) r] - orf[(size_t) r]) * (og[(size_t) r] - orf[(size_t) r]);
            ec += (oc[(size_t) r] - orf[(size_t) r]) * (oc[(size_t) r] - orf[(size_t) r]);
            egc += (double) (og[(size_t) r] - oc[(size_t) r]) * (og[(size_t) r] - oc[(size_t) r]);
        }
        std::fprintf(stderr, "glm fast: CPU lane check (layer %d, types %d/%d): relative L2 error vs the double "
                             "reference: device %.5f, CPU lane %.5f; device vs CPU lane %.5f\n", il_cal, Ly.gu_type,
                     Ly.d_type, std::sqrt(eg / nr), std::sqrt(ec / nr), std::sqrt(egc / nr));
        cudaFree(dh);
        cudaFree(dw);
        cudaFree(dx);
        cudaFree(dout);
        cudaFree(dxq);
        cudaFree(dhq);
    }
    return true;
}

// ---------------------------------------------------------------- the usage profile
// Where this machine's expert usage is kept between sessions: STRATA_GLM_USAGE=<file> (0: none), else
// expert_usage.txt in the pack's folder.
std::string Glm5Model::usage_path() const {
    const char* u = getenv("STRATA_GLM_USAGE");
    if (u != nullptr) return std::string(u) == "0" ? std::string() : std::string(u);
    return pack_dir_.empty() ? std::string() : pack_dir_ + "/expert_usage.txt";
}

// Every half's long-memory routes ("layer e:count ..."), one file, written whole and renamed into place.
bool Glm5Model::save_usage() {
    const std::string up = usage_path();
    if (up.empty() || fast_ == nullptr) return false;
    std::string out = "# strata expert usage: routes per expert across sessions (the warm-up fills the tiers in this "
                      "order)\n";
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        FastState* F = m->fast_;
        if (F == nullptr || F->usage.empty()) continue;
        std::vector<uint32_t> u;
        {
            std::lock_guard<std::mutex> lk(F->mu);
            u = F->usage;
        }
        const int NE = m->g_.n_expert;
        for (int il = m->l0_; il < m->lt_; ++il) {
            std::string line = std::to_string(il);
            bool any = false;
            for (int e = 0; e < NE; ++e) {
                const uint32_t n = u[(size_t) il * NE + e];
                if (n == 0) continue;
                line += " " + std::to_string(e) + ":" + std::to_string(n);
                any = true;
            }
            if (any) out += line + "\n";
        }
    }
    const std::string tmp = up + ".tmp";
    {
        std::ofstream f(tmp, std::ios::trunc);
        if (!f) return false;
        f << out;
        if (!f) return false;
    }
    return std::rename(tmp.c_str(), up.c_str()) == 0;
}

// ---------------------------------------------------------------- load-time warm-up
// VRAM and the pinned RAM tier together hold (nearly) every routed expert, so the whole model is streamed from the
// shards ONCE at load: each layer's most frequently routed experts (the pack's expert_prior.txt, written by
// tools/glm_expert_prior.py from routing traces; id order without it) fill its VRAM slots - all but the spares - and
// the rest fill the RAM tier (a few slots per class stay free for disk reads).  The first request then runs warm and
// the disk is touched again only for experts that fit nowhere.  STRATA_GLM_WARM=0 skips it (the tiers fill on demand).
bool Glm5Model::fast_warm(std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const auto t0 = std::chrono::steady_clock::now();
    const int NL = g.n_layers + 1;
    std::vector<std::vector<int>> order((size_t) NL);
    {
        std::ifstream pf(pack_dir_ + "/expert_prior.txt");
        std::string line;
        while (std::getline(pf, line)) {
            std::istringstream ss(line);
            int il = -1, e = 0;
            ss >> il;
            if (il < 0 || il >= NL) continue;
            while (ss >> e)
                if (e >= 0 && e < g.n_expert) order[(size_t) il].push_back(e);
        }
    }
    // THIS user's experts first: the usage profile of earlier sessions (save_usage) leads each layer's order, its
    // counts become the long memory again, and the LFU counts start from them scaled to at most 32 (enough to keep
    // the profile's experts against one-off routes, little enough that a new task takes over within ~100 tokens;
    // a restart then hits ~75% instead of ~67% over the first 50 tokens on one V100, simulated)
    const std::string up = usage_path();
    int n_prof = 0;
    if (!up.empty()) {
        std::ifstream uf(up);
        std::string line;
        while (std::getline(uf, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            int il = -1;
            ss >> il;
            if (il < l0_ || il >= lt_ || !F->L[(size_t) il].moe) continue;
            std::vector<std::pair<uint32_t, int>> v;
            std::string tok;
            while (ss >> tok) {
                const size_t c = tok.find(':');
                if (c == std::string::npos) continue;
                const int e = std::atoi(tok.substr(0, c).c_str());
                const uint32_t n = (uint32_t) std::strtoul(tok.substr(c + 1).c_str(), nullptr, 10);
                if (e >= 0 && e < g.n_expert && n > 0) v.emplace_back(n, e);
            }
            if (v.empty()) continue;
            std::sort(v.rbegin(), v.rend());
            std::vector<int> o;
            for (auto& p : v) {
                o.push_back(p.second);
                const size_t key = (size_t) il * g.n_expert + p.second;
                F->usage[key] = p.first;
                F->cnt[key] = (uint32_t) std::max<uint64_t>(1, (uint64_t) 32 * p.first / v.front().first);
            }
            o.insert(o.end(), order[(size_t) il].begin(), order[(size_t) il].end());
            order[(size_t) il].swap(o);
            ++n_prof;
        }
        if (n_prof > 0)
            std::fprintf(stderr, "glm fast: CUDA%d the tiers follow this machine's usage profile (%s, %d layers)\n", dev_,
                         up.c_str(), n_prof);
    }
    for (int il = 0; il < NL; ++il) {
        auto& o = order[(size_t) il];
        std::vector<char> seen((size_t) g.n_expert, 0);
        std::vector<int> clean;
        for (int e : o)
            if (!seen[(size_t) e]) {
                seen[(size_t) e] = 1;
                clean.push_back(e);
            }
        for (int e = 0; e < g.n_expert; ++e)
            if (!seen[(size_t) e]) clean.push_back(e);
        o.swap(clean);
    }
    struct Job {
        int il, e;
        uint8_t* dst;    // the RAM slot, or the VRAM slot (staged)
        bool vram;
    };
    std::vector<Job> jobs;
    const size_t N = (size_t) F->n_keys;
    std::vector<unsigned long long> th(2 * N + (size_t) NL * gf::kSpares, 0ull);
    std::vector<int> next((size_t) NL, 0);
    for (int il = l0_; il < lt_; ++il) {
        if (!F->L[(size_t) il].moe) continue;
        auto& P = F->lp[(size_t) il];
        const int nv = std::max(0, P.n - gf::kSpares);
        for (int j = 0; j < nv; ++j) {
            const int e = order[(size_t) il][(size_t) j];
            const int key = il * g.n_expert + e;
            P.key[(size_t) j] = e;
            P.st[(size_t) j] = FastState::kResident;
            F->slot_of[(size_t) key] = j;
            th[(size_t) key] = (unsigned long long) P.slot_ptr(j);
            jobs.push_back(Job{il, e, P.slot_ptr(j), true});
        }
        next[(size_t) il] = nv;
    }
    // the RAM tier, round-robin over the layers so every layer gets its share of a class that cannot hold all
    std::vector<int> rfree(F->rc.size(), 0), rcur(F->rc.size(), 0);
    for (size_t c = 0; c < F->rc.size(); ++c) rfree[c] = F->rc[c].n;
    for (bool progress = true; progress;) {
        progress = false;
        for (int il = l0_; il < lt_; ++il) {
            if (!F->L[(size_t) il].moe || next[(size_t) il] >= g.n_expert) continue;
            const int c = F->layer_rc[(size_t) il];
            auto& R = F->rc[(size_t) c];
            if (rfree[(size_t) c] <= 4) continue;
            while (rcur[(size_t) c] < R.n && R.st[(size_t) rcur[(size_t) c]] != FastState::kRFree) ++rcur[(size_t) c];
            if (rcur[(size_t) c] >= R.n) continue;
            const int s = rcur[(size_t) c]++;
            const int e = order[(size_t) il][(size_t) next[(size_t) il]++];
            const int key = il * g.n_expert + e;
            R.key[(size_t) s] = key;
            R.st[(size_t) s] = FastState::kRHold;
            F->ram_of[(size_t) key] = s;
            th[N + (size_t) key] = (unsigned long long) (R.base + (size_t) s * R.stride);
            jobs.push_back(Job{il, e, R.base + (size_t) s * R.stride, false});
            --rfree[(size_t) c];
            progress = true;
        }
    }
    // stream them: 8 experts per batch, 3 slices each in parallel; VRAM ones through the pinned staging
    size_t done_bytes = 0, total_bytes = 0;
    for (const auto& j : jobs) total_bytes += F->L[(size_t) j.il].blob;
    int last_pct = -1;
    for (size_t b = 0; b < jobs.size(); b += 8) {
        const int nb = (int) std::min<size_t>(8, jobs.size() - b);
        F->workers->run(nb * 3, [&](int job) {
            const Job& J = jobs[b + (size_t) (job / 3)];
            const int role = job % 3;
            const auto& Ly = F->L[(size_t) J.il];
            const auto& nl = pack_layers_[(size_t) J.il];
            const Shard& sh = pack_shards_[(size_t) (role == 0 ? nl.gate_shard : role == 1 ? nl.up_shard : nl.down_shard)];
            const uint64_t off = role == 0   ? nl.gate_off + (uint64_t) J.e * Ly.gu_bytes
                                 : role == 1 ? nl.up_off + (uint64_t) J.e * Ly.gu_bytes
                                             : nl.down_off + (uint64_t) J.e * Ly.dn_bytes;
            const size_t len = role == 2 ? Ly.dn_bytes : Ly.gu_bytes;
            uint8_t* base = J.vram ? F->stage[(size_t) (job / 3)] : J.dst;
            uint8_t* d = base + (role == 0 ? 0 : role == 1 ? Ly.gu_bytes : 2 * Ly.gu_bytes);
            read_slice(sh, off, len, d);
        });
        for (int k = 0; k < nb; ++k) {
            const Job& J = jobs[b + (size_t) k];
            if (J.vram)
                cudaMemcpyAsync(J.dst, F->stage[(size_t) k], F->L[(size_t) J.il].blob, cudaMemcpyHostToDevice, F->copy);
            done_bytes += F->L[(size_t) J.il].blob;
        }
        cudaStreamSynchronize(F->copy);
        const int pct = (int) (100.0 * (double) done_bytes / (double) std::max<size_t>(1, total_bytes));
        if (pct / 10 != last_pct / 10) {
            last_pct = pct;
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "glm fast: CUDA%d warming the expert tiers %d%% (%.1f of %.1f GB, %.0f s)\n", dev_, pct,
                         (double) done_bytes / 1e9, (double) total_bytes / 1e9, s);
        }
    }
    if (cudaMemcpy(F->tab, th.data(), th.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "glm fast: the warmed expert tables did not upload";
        return false;
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "glm fast: CUDA%d tiers warm: %zu experts (%.1f GB) in %.0f s (%.2f GB/s)\n", dev_, jobs.size(),
                 (double) total_bytes / 1e9, s, (double) total_bytes / 1e9 / std::max(1e-9, s));
    return true;
}

void Glm5Model::fast_destroy() {
    FastState* F = fast_;
    if (F == nullptr) return;
    cudaSetDevice(dev_);
    prefill_destroy();
    F->quit.store(true);
    if (F->svc.joinable()) F->svc.join();
    {
        std::lock_guard<std::mutex> lk(F->ah_mu);
        F->ah_quit = true;
    }
    F->ah_cv.notify_all();
    for (auto& t : F->ah_th) t.join();
    F->workers.reset();
    F->cpu_pool.reset();
    if (F->cpu_ans_h) cudaFreeHost(F->cpu_ans_h);
    if (F->cpu_seq_d) cudaFree(F->cpu_seq_d);
    if (F->dcnt_d) cudaFree(F->dcnt_d);
    if (F->timing && F->ah_routes > 0) {
        std::string ov;
        for (int d = 0; d < F->n_ahead; ++d)
            ov += " " + std::to_string(d + 1) + ":" +
                  std::to_string((double) F->ah_ov[d] / (double) std::max<uint64_t>(1, F->ah_npred[d])).substr(0, 4);
        std::string cov;
        for (int d = 0; d < F->n_ahead; ++d)
            cov += " " + std::to_string(d + 1) + ":" +
                   std::to_string(100.0 * (double) F->ah_disk_cov[d] / (double) std::max<uint64_t>(1, F->ah_disk))
                       .substr(0, 4) + "%";
        std::fprintf(stderr,
                     "glm fast ahead CUDA%d: of 8 experts predicted d layers early%s | disk misses %llu, named d early%s, "
                     "by any %.1f%% | reads issued %llu, used %llu (waited %llu, %.1f ms; taken over %llu), queue full %llu, "
                     "no free slot %llu\n",
                     dev_, ov.c_str(), (unsigned long long) F->ah_disk, cov.c_str(),
                     100.0 * (double) F->ah_disk_any / (double) std::max<uint64_t>(1, F->ah_disk),
                     (unsigned long long) F->ah_issued, (unsigned long long) F->ah_used,
                     (unsigned long long) F->ah_waited, (double) F->ah_wait_us.load() / 1000.0,
                     (unsigned long long) F->ah_stolen, (unsigned long long) F->ah_full,
                     (unsigned long long) F->ah_nofree);
    }
    if (F->timing && F->tokens > 0)
        std::fprintf(stderr, "glm fast timing: %llu tokens, %.2f ms/token\n", (unsigned long long) F->tokens,
                     F->ms / (double) F->tokens);
    if (F->timing && F->hits.load() + F->misses.load() > 0)
        std::fprintf(stderr,
                     "glm fast tiers CUDA%d: vram hits %llu, ram fetches %llu, disk reads %llu (%.2f%% vram hit) | "
                     "promotions %llu (scratch %llu) demotions %llu drops %llu prefetches %llu | disk waits %llu, %.1f ms\n",
                     dev_, (unsigned long long) F->hits.load(), (unsigned long long) F->ram_hits.load(),
                     (unsigned long long) F->disk_reads.load(),
                     100.0 * (double) F->hits.load() / (double) std::max<uint64_t>(1, F->hits.load() + F->misses.load()),
                     (unsigned long long) F->promotions.load(), (unsigned long long) F->scratch_uses.load(),
                     (unsigned long long) F->demotions.load(), (unsigned long long) F->drops.load(),
                     (unsigned long long) F->prefetches.load(), (unsigned long long) F->miss_layers.load(),
                     (double) F->miss_us.load() / 1000.0);
    if (F->timing && F->pred_n.load() > 0)
        std::fprintf(stderr, "glm fast predict CUDA%d: %.2f of 8 experts predicted one layer early; %.1f%% of the "
                             "non-resident ones (%llu)\n",
                     dev_, (double) F->pred_overlap.load() / (double) F->pred_n.load(),
                     100.0 * (double) F->pred_miss_hit.load() / (double) std::max<uint64_t>(1, F->pred_miss.load()),
                     (unsigned long long) F->pred_miss.load());
    if (F->prof_on && F->ptokens > 0) {
        std::vector<std::pair<double, std::string>> v;
        double tot = 0;
        for (auto& kv : F->pacc) {
            v.push_back({kv.second, kv.first});
            tot += kv.second;
        }
        std::sort(v.rbegin(), v.rend());
        std::fprintf(stderr, "glm fast prof CUDA%d (%llu tokens, %.2f ms/token of GPU stream time):\n", dev_,
                     (unsigned long long) F->ptokens, tot / (double) F->ptokens);
        for (auto& e : v)
            std::fprintf(stderr, "  %-18s %7.3f ms/token  %5.1f%%\n", e.second.c_str(), e.first / (double) F->ptokens,
                         100.0 * e.first / tot);
    }
    for (auto e : F->pev) cudaEventDestroy(e);
    if (snap_) {
        cudaFree(snap_);
        snap_ = nullptr;
    }
    if (kda_bak_) {
        cudaFree(kda_bak_);
        kda_bak_ = nullptr;
    }
    for (int i = 0; i < 2; ++i) {
        if (spec_hop_h_[i]) cudaFreeHost(spec_hop_h_[i]);
        if (spec_ev_hop_[i]) cudaEventDestroy(spec_ev_hop_[i]);
        spec_hop_h_[i] = nullptr;
        spec_ev_hop_[i] = nullptr;
    }
    if (spec_steps_ > 0)
        std::fprintf(stderr, "glm spec: %llu speculative positions, %.1f%% accepted\n", (unsigned long long) spec_steps_,
                     100.0 * (double) spec_hits_ / (double) spec_steps_);
    if (F->cs) cudaStreamSynchronize(F->cs);
    if (F->copy) cudaStreamSynchronize(F->copy);
    for (auto& R : F->rc)
        if (R.base) cudaFreeHost(R.base);
    for (auto& d : F->draining) cudaEventDestroy(d.ev);
    for (auto e : F->ev_free) cudaEventDestroy(e);
    for (auto* sb : F->stage)
        if (sb) cudaFreeHost(sb);
    if (F->upd_key_h) cudaFreeHost(F->upd_key_h);
    if (F->upd_val_h) cudaFreeHost(F->upd_val_h);
    if (F->upd_key_d) cudaFree(F->upd_key_d);
    if (F->upd_val_d) cudaFree(F->upd_val_d);
    if (F->ring_h) cudaFreeHost(F->ring_h);
    if (F->resp_h) cudaFreeHost(F->resp_h);
    if (F->emb_h) cudaFreeHost(F->emb_h);
    if (F->hop_h) cudaFreeHost(F->hop_h);
    if (F->tok_h) cudaFreeHost(F->tok_h);
    if (F->scratch) cudaFree(F->scratch);
    if (F->mtp_h) cudaFree(F->mtp_h);
    if (F->mtp_catq) cudaFree(F->mtp_catq);
    if (F->mtp_logits) cudaFree(F->mtp_logits);
    if (F->mtp_tok) cudaFree(F->mtp_tok);
    if (F->mtp_tok_h) cudaFreeHost(F->mtp_tok_h);
    if (F->ev_mtp) cudaEventDestroy(F->ev_mtp);
    if (F->pool) cudaFree(F->pool);
    if (F->xpool) cudaFree(F->xpool);
    if (F->arena) cudaFree(F->arena);
    if (F->ps) cudaStreamSynchronize(F->ps);
    if (F->ev_hop) cudaEventDestroy(F->ev_hop);
    if (F->ev_done) cudaEventDestroy(F->ev_done);
    if (F->ev_pred) cudaEventDestroy(F->ev_pred);
    if (F->ev_pf) cudaEventDestroy(F->ev_pf);
    if (F->ev_pf_prev) cudaEventDestroy(F->ev_pf_prev);
    if (F->cs) cudaStreamDestroy(F->cs);
    if (F->copy) cudaStreamDestroy(F->copy);
    if (F->ps) cudaStreamDestroy(F->ps);
    delete F;
    fast_ = nullptr;
}

Glm5Model::FastStats Glm5Model::fast_stats() const {
    FastStats s;
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        const FastState* F = m->fast_;
        if (F == nullptr) continue;
        s.hits += F->hits.load();
        s.misses += F->misses.load();
        s.miss_layers += F->miss_layers.load();
        s.ram_hits += F->ram_hits.load();
        s.disk_reads += F->disk_reads.load();
        s.promotions += F->promotions.load();
        s.cpu_experts += F->cpu_experts.load();
        s.cpu_ms += (double) F->cpu_us.load() / 1000.0;
        s.miss_ms += (double) F->miss_us.load() / 1000.0;
        s.disk_ms += (double) F->disk_us.load() / 1000.0;
        s.pool_slots += F->pool_slots;
        s.pool_gb += (double) F->pool_bytes / 1073741824.0;
        s.ram_gb += (double) F->ram_bytes / 1073741824.0;
        for (const auto& P : F->lp)
            for (char c : P.st) s.pool_used += c == FastState::kResident;
        for (const auto& R : F->rc) {
            s.ram_slots += R.n;
            for (char c : R.st) s.ram_used += c == FastState::kRHold || c == FastState::kRNew;
        }
    }
    if (fast_ != nullptr) {
        s.tokens = fast_->tokens;
        s.ms = fast_->ms;
    }
    return s;
}

// ---------------------------------------------------------------- the service thread
//
// Consumes every route the device publishes, in order: LFU counts, the promotions the device made (an expert
// landed in a spare slot), and - the only case the device waits for - the experts that are on disk only: read
// into a free RAM-tier slot (or pinned staging) and answered with that source for the fetch kernel.
void Glm5Model::fast_service() {
    FastState* F = fast_;
    cudaSetDevice(dev_);
    const Glm5Geometry& g = g_;
    const int K = g.n_exp_used;
    unsigned int next = 1;
    while (!F->quit.load(std::memory_order_relaxed)) {
        gf::MoeRequest* rq = F->ring_h + (next % gf::kRingSize);
        const unsigned int sq = rq->seq;
        if (sq != next) {
            // an entry AHEAD of us means the device lapped the ring (only non-waiting routes can be lapped: a
            // disk miss parks the device until it is answered) - resync instead of waiting forever
            if ((int) (sq - next) > 0 && sq % gf::kRingSize == next % gf::kRingSize) {
                F->processed.fetch_add((uint64_t) (sq - next), std::memory_order_release);
                next = sq;
            } else {
                cpu_relax();
                continue;
            }
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        {
            std::lock_guard<std::mutex> lk(F->mu);
            const int il = rq->layer;
            const unsigned int miss = rq->miss_mask, fetch = rq->fetch_mask, promo = rq->promo_mask;
            const unsigned int cpu = rq->cpu_mask;   // the CPU lane's experts: not in VRAM, computed below
            auto& P = F->lp[(size_t) il];
            auto& R = F->rc[(size_t) F->layer_rc[(size_t) il]];
            ++F->clock;
            int ids[8];
            for (int i = 0; i < K; ++i) ids[i] = rq->ids[i];
            // STRATA_GLM_ROUTE_LOG=<prefix>: "layer e0 .. e7 tier-mask" per route into <prefix>.<device> (cache studies)
            static const char* rlog = getenv("STRATA_GLM_ROUTE_LOG");
            if (rlog != nullptr) {
                static thread_local FILE* rf = std::fopen((std::string(rlog) + "." + std::to_string(dev_)).c_str(), "w");
                if (rf != nullptr) {
                    std::fprintf(rf, "%d", il);
                    for (int i = 0; i < K; ++i) std::fprintf(rf, " %d", ids[i]);
                    std::fprintf(rf, " %u %u %u\n", fetch, miss, cpu);
                }
            }
            uint64_t nh = 0;
            for (int i = 0; i < K; ++i) {
                const int key = il * g.n_expert + ids[i];
                if (F->cnt[(size_t) key] < (1u << 24)) ++F->cnt[(size_t) key];
                if (++F->usage[(size_t) key] >= (1u << 30))
                    for (auto& u : F->usage) u >>= 1;
                const bool resident = (((miss | fetch | cpu) >> i) & 1u) == 0;
                if (resident) {
                    const int s = F->slot_of[(size_t) key];
                    if (s >= 0) P.tick[(size_t) s] = F->clock;
                    ++nh;
                } else {
                    const int rs0 = F->ram_of[(size_t) key];   // used from the RAM tier: recent there
                    if (rs0 >= 0) R.tick[(size_t) rs0] = F->clock;
                }
                if ((promo >> i) & 1u) {
                    // the device filled one of this layer's spares with it: resident from now on
                    const int s = P.slot_index(rq->promo_ptr[i]);
                    if (s >= 0 && s < P.n) {
                        P.st[(size_t) s] = FastState::kResident;
                        P.key[(size_t) s] = ids[i];
                        P.tick[(size_t) s] = F->clock;
                        F->slot_of[(size_t) key] = s;
                        for (int j = 0; j < gf::kSpares; ++j)
                            if (P.spare[j] == s) P.spare[j] = -1;
                        F->promotions.fetch_add(1, std::memory_order_relaxed);
                    }
                    // exclusive tiers: its RAM copy is released at the next boundary (the fetch reads it now)
                    const int rs = F->ram_of[(size_t) key];
                    if (!F->ram_shadow && rs >= 0 && R.st[(size_t) rs] == FastState::kRHold)
                        R.st[(size_t) rs] = FastState::kRRelease;
                } else if (((fetch | miss) >> i) & 1u) {
                    F->scratch_uses.fetch_add(1, std::memory_order_relaxed);
                }
            }
            F->cnt_events += (uint64_t) K;
            if (F->cnt_events >= 32768) {
                F->cnt_events = 0;
                for (auto& c : F->cnt) c >>= 1;
            }
            F->hits.fetch_add(nh, std::memory_order_relaxed);
            {
                // prediction quality: how many of this route's experts (and of its non-resident ones) the previous
                // layer's prediction named
                const auto& pv = F->pred_of[(size_t) il];
                if (pv[0] >= 0) {
                    uint64_t ov = 0, mh = 0;
                    for (int i = 0; i < K; ++i) {
                        bool in = false;
                        for (int q2 = 0; q2 < K; ++q2) in |= pv[(size_t) q2] == ids[i];
                        ov += in;
                        if ((((fetch | miss) >> i) & 1u) && in) ++mh;
                    }
                    F->pred_n.fetch_add(1, std::memory_order_relaxed);
                    F->pred_overlap.fetch_add(ov, std::memory_order_relaxed);
                    F->pred_miss.fetch_add((uint64_t) std::popcount((unsigned) (fetch | miss)), std::memory_order_relaxed);
                    F->pred_miss_hit.fetch_add(mh, std::memory_order_relaxed);
                }
                if (il + 1 < (int) F->pred_of.size())
                    for (int i = 0; i < K; ++i) F->pred_of[(size_t) il + 1][(size_t) i] = rq->pred[i];
            }
            // the next layer's experts this route prefetched into that layer's spares: resident from now on
            const int npf = rq->pf_n;
            if (npf > 0 && il + 1 < (int) F->lp.size()) {
                auto& P1 = F->lp[(size_t) il + 1];
                auto& R1 = F->rc[(size_t) F->layer_rc[(size_t) il + 1]];
                for (int q2 = 0; q2 < npf && q2 < 8; ++q2) {
                    const int e = rq->pf_ids[q2];
                    const int key = (il + 1) * g.n_expert + e;
                    const int s = P1.slot_index(rq->pf_ptr[q2]);
                    if (s < 0 || s >= P1.n) continue;
                    P1.st[(size_t) s] = FastState::kResident;
                    P1.key[(size_t) s] = e;
                    P1.tick[(size_t) s] = F->clock;
                    F->slot_of[(size_t) key] = s;
                    for (int j = 0; j < gf::kSpares; ++j)
                        if (P1.spare[j] == s) P1.spare[j] = -1;
                    const int rs = F->ram_of[(size_t) key];
                    if (!F->ram_shadow && rs >= 0 && R1.st[(size_t) rs] == FastState::kRHold)
                        R1.st[(size_t) rs] = FastState::kRRelease;
                    F->prefetches.fetch_add(1, std::memory_order_relaxed);
                }
            }
            F->ram_hits.fetch_add((uint64_t) std::popcount((unsigned) (fetch)), std::memory_order_relaxed);
            F->misses.fetch_add((uint64_t) std::popcount((unsigned) (fetch | miss | cpu)), std::memory_order_relaxed);
            if (F->n_ahead > 0) fast_ahead_route(il, ids, miss, rq->ahead);
            if (miss != 0) {
                // ---- disk only: read each into a free RAM slot (it stays there unless it was promoted) or staging;
                //      one the LOOKAHEAD already read (or is reading) is answered from its RAM slot
                const auto t0 = std::chrono::steady_clock::now();
                const int cls = F->layer_rc[(size_t) il];
                int nm = 0, mi[8], nr = 0, rd[8], nw = 0, wt[8], ns = 0, sl[8];
                uint64_t stv[8];
                uint8_t* dst[8];
                for (int i = 0; i < K; ++i)
                    if ((miss >> i) & 1u) mi[nm++] = i;
                for (int m = 0; m < nm; ++m) {
                    const int key = il * g.n_expert + ids[mi[m]];
                    int rs = F->ram_of[(size_t) key];
                    if (rs >= 0 && (R.st[(size_t) rs] == FastState::kRLoad || R.st[(size_t) rs] == FastState::kRNew)) {
                        dst[m] = R.base + (size_t) rs * R.stride;
                        if (R.st[(size_t) rs] == FastState::kRLoad) {
                            auto& ld = F->rload[(size_t) cls][(size_t) rs];
                            uint64_t v = ld.load(std::memory_order_acquire);
                            if ((v & 3u) == 0u && ld.compare_exchange_strong(v, v | 1u, std::memory_order_acq_rel)) {
                                rd[nr++] = m;   // still queued: read it here (the reader skips it)
                                sl[ns] = rs;
                                stv[ns++] = (v & ~(uint64_t) 3u) | 2u;
                                ++F->ah_stolen;
                            } else if ((v & 3u) != 2u) {
                                wt[nw++] = rs;   // a reader has it in hand
                            }
                            ++F->ah_used;
                        }
                        if ((promo >> mi[m]) & 1u && !F->ram_shadow) {
                            // Exclusive tiers: a promoted disk read is only staging and frees at the boundary.
                            R.st[(size_t) rs] = FastState::kRRelease;
                            R.key[(size_t) rs] = -1;
                            F->ram_of[(size_t) key] = -1;
                        } else {
                            R.st[(size_t) rs] = FastState::kRNew;
                        }
                        continue;
                    }
                    rd[nr++] = m;
                    rs = -1;
                    for (int s2 = 0; s2 < R.n; ++s2)
                        if (R.st[(size_t) s2] == FastState::kRFree) {
                            rs = s2;
                            break;
                        }
                    if (rs >= 0) {
                        R.key[(size_t) rs] = key;
                        R.tick[(size_t) rs] = F->clock;
                        const bool staging = ((promo >> mi[m]) & 1u) && !F->ram_shadow;
                        R.st[(size_t) rs] = staging ? FastState::kRRelease : FastState::kRNew;
                        F->ram_of[(size_t) key] = staging ? -1 : rs;
                        dst[m] = R.base + (size_t) rs * R.stride;
                        if (staging) R.key[(size_t) rs] = -1;   // a pure staging use: freed at the boundary
                    } else {
                        dst[m] = F->stage[(size_t) m];
                    }
                }
                // each part in chunks across the workers: the O_DIRECT bounce copy of a whole part was ~0.4 ms of the
                // wait (Mercury: 3.6 -> 3.1 ms a disk wait at 8 chunks; STRATA_GLM_READ_CHUNKS overrides)
                static const int rch = [] {
                    const char* v = getenv("STRATA_GLM_READ_CHUNKS");
                    return std::max(1, std::min(16, v ? std::atoi(v) : 8));
                }();
                F->workers->run(nr * 3 * rch, [&](int job) {
                    const int m = rd[job / (3 * rch)], part = job % (3 * rch);
                    fast_read_part(il, ids[mi[m]], part / rch, dst[m], part % rch, rch);
                });
                for (int i = 0; i < ns; ++i) F->rload[(size_t) cls][(size_t) sl[i]].store(stv[i], std::memory_order_release);
                if (nw > 0) {
                    const auto tw = std::chrono::steady_clock::now();
                    for (int i = 0; i < nw; ++i)
                        while ((F->rload[(size_t) cls][(size_t) wt[i]].load(std::memory_order_acquire) & 3u) != 2u)
                            cpu_relax();
                    F->ah_waited += (uint64_t) nw;
                    F->ah_wait_us.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                                                std::chrono::steady_clock::now() - tw).count(),
                                            std::memory_order_relaxed);
                }
                gf::MoeResponse* Rp = F->resp_h;
                Rp->ptr_mask = 0;
                Rp->n_upd = 0;
                Rp->cpu = 0;
                for (int m = 0; m < nm; ++m) {
                    Rp->ptr[mi[m]] = (unsigned long long) dst[m];
                    Rp->ptr_mask |= 1u << mi[m];
                }
                std::atomic_thread_fence(std::memory_order_release);
                Rp->seq = next;
                const uint64_t us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::steady_clock::now() - t0).count();
                F->disk_reads.fetch_add((uint64_t) nr, std::memory_order_relaxed);   // read here (not ahead)
                for (int r2 = 0; r2 < nr; ++r2) ++F->diag_disk[F->left[(size_t) il * g.n_expert + ids[mi[rd[r2]]]] & 3];
                F->disk_us.fetch_add(us, std::memory_order_relaxed);
                F->miss_us.fetch_add(us, std::memory_order_relaxed);
                F->miss_layers.fetch_add(1, std::memory_order_relaxed);
                static const bool trace = getenv("STRATA_GLM_SVC_TRACE") != nullptr;
                if (trace)
                    std::fprintf(stderr, "svc CUDA%d seq %u layer %d disk %d %.2f ms\n", dev_, next, il, nm,
                                 (double) us / 1000.0);
            }
        }
        // ---- the CPU LANE: this route's host experts from their RAM-tier blobs (they stay in RAM until the next
        //      boundary: the device waits for this answer before its down combine, so the token cannot end first)
        if (rq->cpu_mask != 0u && F->cpu_ans_h != nullptr) {
            const auto t0 = std::chrono::steady_clock::now();
            const unsigned int cm = rq->cpu_mask;
            int ne = 0;
            const uint8_t* blob[8];
            float w[8];
            for (int i = 0; i < g.n_exp_used; ++i)
                if ((cm >> i) & 1u) {
                    blob[ne] = (const uint8_t*) rq->cpu_src[i];
                    w[ne++] = rq->w[i];
                }
            fast_cpu_experts(rq->layer, ne, blob, w, rq->x, F->cpu_ans_h->part);
            std::atomic_thread_fence(std::memory_order_release);
            F->cpu_ans_h->seq = next;
            F->cpu_experts.fetch_add((uint64_t) ne, std::memory_order_relaxed);
            F->cpu_routes.fetch_add(1, std::memory_order_relaxed);
            F->cpu_us.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now() - t0).count(),
                                std::memory_order_relaxed);
        }
        F->processed.fetch_add(1, std::memory_order_release);
        ++next;
    }
}

// one part (0 gate, 1 up, 2 down) of expert e of layer il - or its chunk-th of n_chunks equal pieces - from the shards
// into its place in the blob at `blob`
void Glm5Model::fast_read_part(int il, int e, int role, uint8_t* blob, int chunk, int n_chunks) {
    const auto& Ly = fast_->L[(size_t) il];
    const auto& nl = pack_layers_[(size_t) il];
    const Shard& sh = pack_shards_[(size_t) (role == 0 ? nl.gate_shard : role == 1 ? nl.up_shard : nl.down_shard)];
    const uint64_t off = role == 0   ? nl.gate_off + (uint64_t) e * Ly.gu_bytes
                         : role == 1 ? nl.up_off + (uint64_t) e * Ly.gu_bytes
                                     : nl.down_off + (uint64_t) e * Ly.dn_bytes;
    const size_t len = role == 2 ? Ly.dn_bytes : Ly.gu_bytes;
    const size_t piece = ((len + (size_t) n_chunks - 1) / (size_t) n_chunks + 4095) & ~(size_t) 4095;
    const size_t c0 = std::min(len, (size_t) chunk * piece), c1 = std::min(len, c0 + piece);
    if (c1 > c0)
        read_slice(sh, off + c0, c1 - c0, blob + (role == 0 ? 0 : role == 1 ? Ly.gu_bytes : 2 * Ly.gu_bytes) + c0);
}

// ---- LOOKAHEAD, for the route of layer il (service thread, F->mu held): score the predictions made for this layer
//      d+1 layers earlier, keep this route's predictions for the layers after it, and queue a read of every
//      predicted expert that is on disk only (VRAM and RAM tiers both lack it) into a free RAM slot.  The slot is
//      kRLoad until it lands; the layer's own route then finds it through ram_of (the device still calls it a disk
//      miss: rtab learns of it at the next boundary).
void Glm5Model::fast_ahead_route(int il, const int* ids, unsigned int miss, const short (*ahead)[8]) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const int K = g.n_exp_used;
    ++F->ah_routes;
    auto& pr = F->ah_pred[(size_t) il];
    bool cov[8] = {false, false, false, false, false, false, false, false};
    for (int d = 0; d < F->n_ahead; ++d) {
        if (pr[(size_t) d][0] < 0) continue;
        ++F->ah_npred[d];
        for (int i = 0; i < K; ++i) {
            bool in = false;
            for (int q = 0; q < 8; ++q) in |= pr[(size_t) d][(size_t) q] == ids[i];
            F->ah_ov[d] += in;
            if (in && ((miss >> i) & 1u)) {
                ++F->ah_disk_cov[d];
                cov[i] = true;
            }
        }
        pr[(size_t) d].fill(-1);
    }
    for (int i = 0; i < K; ++i)
        if ((miss >> i) & 1u) {
            ++F->ah_disk;
            F->ah_disk_any += cov[i];
        }
    for (int d = 0; d < F->n_ahead; ++d) {
        const int la = il + 1 + d;
        if (ahead[d][0] < 0 || la >= (int) F->ah_pred.size()) continue;
        for (int i = 0; i < 8; ++i) F->ah_pred[(size_t) la][(size_t) d][(size_t) i] = ahead[d][i];
    }
    if (!F->ahead_read) return;
    constexpr int kMaxInflight = 4;
    bool queued = false;
    for (int d = 0; d < F->n_ahead; ++d) {   // the nearest layer first: its reads are the most urgent
        const int la = il + 1 + d;
        if (ahead[d][0] < 0 || la >= (int) F->layer_rc.size() || F->layer_rc[(size_t) la] < 0) continue;
        const int cls = F->layer_rc[(size_t) la];
        auto& R = F->rc[(size_t) cls];
        for (int i = 0; i < K; ++i) {
            const int e = ahead[d][i];
            if (e < 0) continue;
            const int key = la * g.n_expert + e;
            if (F->slot_of[(size_t) key] >= 0 || F->ram_of[(size_t) key] >= 0) continue;
            if (F->ah_inflight.load(std::memory_order_relaxed) >= kMaxInflight) {
                ++F->ah_full;
                continue;
            }
            int rs = -1;
            for (int s = 0; s < R.n; ++s)
                if (R.st[(size_t) s] == FastState::kRFree) {
                    rs = s;
                    break;
                }
            if (rs < 0) {
                ++F->ah_nofree;
                continue;
            }
            R.key[(size_t) rs] = key;
            R.st[(size_t) rs] = FastState::kRLoad;
            F->ram_of[(size_t) key] = rs;
            const uint64_t tag = ++F->ah_tag;
            F->rload[(size_t) cls][(size_t) rs].store(tag << 2, std::memory_order_release);
            F->ah_inflight.fetch_add(1, std::memory_order_relaxed);
            ++F->ah_issued;
            {
                std::lock_guard<std::mutex> lk(F->ah_mu);
                F->ah_q.push_back(FastState::AheadJob{cls, rs, la, e, tag});
            }
            queued = true;
        }
    }
    if (queued) F->ah_cv.notify_all();
}

// the LOOKAHEAD reader threads: one queued expert at a time into the RAM slot the service thread claimed for it - unless
// the route got there first and read it itself (the tag's phase moved on), or the slot was reused under a newer tag
void Glm5Model::fast_ahead_reader() {
    FastState* F = fast_;
    for (;;) {
        FastState::AheadJob j;
        {
            std::unique_lock<std::mutex> lk(F->ah_mu);
            F->ah_cv.wait(lk, [&] { return F->ah_quit || !F->ah_q.empty(); });
            if (F->ah_quit) return;
            j = F->ah_q.front();
            F->ah_q.erase(F->ah_q.begin());
        }
        auto& ld = F->rload[(size_t) j.rclass][(size_t) j.rslot];
        uint64_t want = j.tag << 2;
        if (ld.compare_exchange_strong(want, (j.tag << 2) | 1u, std::memory_order_acq_rel)) {
            uint8_t* blob = F->rc[(size_t) j.rclass].base + (size_t) j.rslot * F->rc[(size_t) j.rclass].stride;
            for (int role = 0; role < 3; ++role) fast_read_part(j.il, j.e, role, blob);
            ld.store((j.tag << 2) | 2u, std::memory_order_release);
        }
        F->ah_inflight.fetch_sub(1, std::memory_order_relaxed);
    }
}

// ---------------------------------------------------------------- between tokens
//
// Every device idle, the service thread caught up: (1) RAM slots of promoted experts free up, disk reads of the
// last token become RAM residents; (2) demotions that landed make their RAM copy live and their VRAM slot free;
// (3) every layer's spares are refilled - from free slots, else by evicting the layer's least-used resident
// (demoted into the RAM tier when it beats the RAM tier's least-used, dropped otherwise); (4) the RAM tier keeps
// a few free slots for disk reads.  The table edits ride one small update kernel at the head of the next token.
void Glm5Model::fast_boundary() {
    FastState* F = fast_;
    if (F == nullptr || F->lp.empty() || F->rc.empty()) return;
    const Glm5Geometry& g = g_;
    while (F->processed.load(std::memory_order_acquire) < F->expected) cpu_relax();
    std::lock_guard<std::mutex> lk(F->mu);
    int nu = 0;
    const auto flush = [&] {
        if (nu > 0) {
            cudaMemcpyAsync(F->upd_key_d, F->upd_key_h, (size_t) nu * sizeof(int), cudaMemcpyHostToDevice, F->cs);
            cudaMemcpyAsync(F->upd_val_d, F->upd_val_h, (size_t) nu * sizeof(unsigned long long), cudaMemcpyHostToDevice,
                            F->cs);
            gf::tab_update(F->tab, F->upd_key_d, F->upd_val_d, nu, F->cs);
        }
        nu = 0;
        ++F->upd_g;
    };
    ++F->upd_g;
    const auto upd = [&](size_t k, unsigned long long v) {
        if (F->upd_gen[k] == F->upd_g) {   // edited earlier in this batch: the last value is the one that counts
            F->upd_val_h[F->upd_at[k]] = v;
            return;
        }
        if (nu == FastState::kMaxUpd) {   // a full batch goes out first (the host buffers are reused after it ran)
            flush();
            cudaStreamSynchronize(F->cs);
        }
        F->upd_gen[k] = F->upd_g;
        F->upd_at[k] = nu;
        F->upd_key_h[nu] = (int) k;
        F->upd_val_h[nu] = v;
        ++nu;
    };
    // the RAM tier's victim among its held experts: the least-used by the aged counts among those no route used in the
    // last `protect` routes (~64 tokens) - a newly hot expert keeps its place while its count catches up, instead of
    // going back to disk as the lowest count and being read again a moment later (real chats measured 7-11 disk reads
    // a token that way); all of them recent: the least recently used.  STRATA_GLM_RAM_EVICT=lfu: the plain least-used
    // (the old rule), =lru: the least recently used.  -1: none held.
    static const int ram_policy = [] {
        const char* v = getenv("STRATA_GLM_RAM_EVICT");
        return v == nullptr ? 0 : std::strcmp(v, "lfu") == 0 ? 1 : std::strcmp(v, "lru") == 0 ? 2 : 0;
    }();
    static const uint64_t protect = [] {
        const char* v = getenv("STRATA_GLM_RAM_PROTECT");
        return (uint64_t) (v ? std::max(0, std::atoi(v)) : 64) * 42u;
    }();
    const auto ram_victim = [&](const FastState::RamClass& R) -> int {
        int best = -1, oldest = -1;
        uint32_t bc = UINT32_MAX;
        uint64_t bt = UINT64_MAX;
        for (int s = 0; s < R.n; ++s) {
            if (R.st[(size_t) s] != FastState::kRHold) continue;
            const uint64_t t = R.tick[(size_t) s];
            if (t < bt) {
                bt = t;
                oldest = s;
            }
            if (ram_policy == 2) continue;
            if (ram_policy == 0 && t + protect > F->clock) continue;
            const uint32_t c = F->cnt[(size_t) R.key[(size_t) s]];
            if (c < bc || (c == bc && t < R.tick[(size_t) best])) {
                bc = c;
                best = s;
            }
        }
        return best >= 0 ? best : oldest;
    };
    static const bool tier_gc = [] {
        const char* v = getenv("STRATA_GLM_TIER_GC");
        return v == nullptr || std::atoi(v) != 0;
    }();
    // (1)
    for (size_t c = 0; c < F->rc.size(); ++c) {
        auto& R = F->rc[c];
        for (int s = 0; s < R.n; ++s) {
            if (R.st[(size_t) s] == FastState::kRLoad) {
                // a LOOKAHEAD read nobody asked for yet: in the RAM tier once it landed
                if ((F->rload[c][(size_t) s].load(std::memory_order_acquire) & 3u) == 2u) {
                    R.st[(size_t) s] = FastState::kRHold;
                    upd(F->rtab_key(R.key[(size_t) s]), (unsigned long long) (R.base + (size_t) s * R.stride));
                }
            } else if (R.st[(size_t) s] == FastState::kRRelease) {
                const int key = R.key[(size_t) s];
                if (key >= 0) {
                    upd(F->rtab_key(key), 0ull);
                    if (F->ram_of[(size_t) key] == s) F->ram_of[(size_t) key] = -1;
                }
                R.key[(size_t) s] = -1;
                R.st[(size_t) s] = FastState::kRFree;
            } else if (R.st[(size_t) s] == FastState::kRNew) {
                R.st[(size_t) s] = FastState::kRHold;
                upd(F->rtab_key(R.key[(size_t) s]), (unsigned long long) (R.base + (size_t) s * R.stride));
            } else if (R.st[(size_t) s] == FastState::kRHold && tier_gc) {
                // exclusive tiers, kept (STRATA_GLM_TIER_GC=0: not): a held copy the tables lost (a route promoted or re-read the expert while its
                // demotion was still landing, which then held a second copy nothing points at) or of an expert VRAM
                // holds is freed; one that is the expert's only copy is adopted back.  Such copies piled up to ~12%
                // of the experts in no tier at all, the RAM tier full of duplicates (chats: ~5 disk reads a token)
                const int key = R.key[(size_t) s];
                const int ro = key >= 0 ? F->ram_of[(size_t) key] : -1;
                if (key >= 0 && ro == s && F->slot_of[(size_t) key] < 0) continue;   // the normal case
                if (F->ram_shadow && key >= 0 && ro == s && F->slot_of[(size_t) key] >= 0)
                    continue;   // an intentional redundant copy; step (4) reclaims shadows before unique copies
                if (key >= 0 && ro < 0 && F->slot_of[(size_t) key] < 0) {
                    F->ram_of[(size_t) key] = s;
                    upd(F->rtab_key(key), (unsigned long long) (R.base + (size_t) s * R.stride));
                    ++F->diag_adopt;
                    continue;
                }
                if (key >= 0 && (ro == s || ro < 0)) {
                    upd(F->rtab_key(key), 0ull);
                    if (ro == s) F->ram_of[(size_t) key] = -1;
                }
                R.key[(size_t) s] = -1;
                R.st[(size_t) s] = FastState::kRFree;
                ++F->diag_dup;
            }
        }
    }
    // (2)
    std::vector<FastState::Drain> still;
    for (auto& d : F->draining) {
        if (cudaEventQuery(d.ev) != cudaSuccess) {
            still.push_back(d);
            continue;
        }
        auto& R = F->rc[(size_t) d.rclass];
        if (R.st[(size_t) d.rslot] == FastState::kRDemote && R.key[(size_t) d.rslot] == d.key) {
            R.st[(size_t) d.rslot] = FastState::kRHold;
            upd(F->rtab_key(d.key), (unsigned long long) (R.base + (size_t) d.rslot * R.stride));
        }
        char& vst = F->lp[(size_t) d.il].st[(size_t) d.vslot];
        if (vst == FastState::kDraining) vst = FastState::kFree;   // (a lent slot stays lent)
        F->ev_free.push_back(d.ev);
    }
    F->draining.swap(still);
    // (3)
    for (int il = l0_; il < lt_; ++il) {
        auto& P = F->lp[(size_t) il];
        if (P.n == 0) continue;
        auto& R = F->rc[(size_t) F->layer_rc[(size_t) il]];
        for (int j = 0; j < gf::kSpares; ++j) {
            if (P.spare[j] >= 0) continue;
            int s = -1;
            for (int s2 = 0; s2 < P.n; ++s2)
                if (P.st[(size_t) s2] == FastState::kFree) {
                    s = s2;
                    break;
                }
            if (s >= 0) {
                P.st[(size_t) s] = FastState::kSpare;
                P.key[(size_t) s] = -1;
                P.spare[j] = s;
                upd(F->spare_key(il, j), (unsigned long long) P.slot_ptr(s));
                continue;
            }
            // evict the least-used resident of this layer (recency breaks ties)
            int v = -1;
            uint32_t bc = UINT32_MAX;
            uint64_t bt = UINT64_MAX;
            for (int s2 = 0; s2 < P.n; ++s2) {
                if (P.st[(size_t) s2] != FastState::kResident) continue;
                const uint32_t c = F->cnt[(size_t) il * g.n_expert + P.key[(size_t) s2]];
                if (c < bc || (c == bc && P.tick[(size_t) s2] < bt)) {
                    bc = c;
                    bt = P.tick[(size_t) s2];
                    v = s2;
                }
            }
            if (v < 0) break;
            const int vkey = il * g.n_expert + P.key[(size_t) v];
            upd(F->tab_key(vkey), 0ull);
            F->slot_of[(size_t) vkey] = -1;
            ++P.evictions;
            const int kept = F->ram_of[(size_t) vkey];
            if (F->ram_shadow && kept >= 0 && R.st[(size_t) kept] == FastState::kRHold) {
                // The bytes already live in RAM: reuse this VRAM slot without a device-to-host copy.
                P.st[(size_t) v] = FastState::kSpare;
                P.key[(size_t) v] = -1;
                P.spare[j] = v;
                upd(F->spare_key(il, j), (unsigned long long) P.slot_ptr(v));
                continue;
            }
            // demote it into the RAM tier: a free slot, else the RAM tier's victim (if colder than it)
            int rs = -1;
            uint32_t rc_min = UINT32_MAX;
            bool redundant = false;
            for (int s2 = 0; s2 < R.n; ++s2)
                if (R.st[(size_t) s2] == FastState::kRFree) {
                    rs = s2;
                    rc_min = 0;
                    break;
                }
            if (rs < 0 && F->ram_shadow) {
                // Replace a redundant shadow before considering an expert's only RAM copy.  This preserves the
                // combined VRAM+RAM working set while the hot set moves between the tiers.
                uint64_t bt2 = UINT64_MAX;
                for (int s2 = 0; s2 < R.n; ++s2) {
                    if (R.st[(size_t) s2] != FastState::kRHold) continue;
                    const int old = R.key[(size_t) s2];
                    if (old >= 0 && F->slot_of[(size_t) old] >= 0 && R.tick[(size_t) s2] < bt2) {
                        rs = s2;
                        bt2 = R.tick[(size_t) s2];
                        redundant = true;
                    }
                }
            }
            if (rs < 0) {
                rs = ram_victim(R);
                if (rs >= 0) rc_min = F->cnt[(size_t) R.key[(size_t) rs]];
            }
            if (rs >= 0 && (redundant || R.st[(size_t) rs] == FastState::kRFree || rc_min < bc)) {
                if (R.st[(size_t) rs] == FastState::kRHold) {
                    const int old = R.key[(size_t) rs];
                    F->left[(size_t) old] = 2;
                    ++F->diag_ram_evict;
                    upd(F->rtab_key(old), 0ull);
                    F->ram_of[(size_t) old] = -1;
                }
                R.key[(size_t) rs] = vkey;
                R.tick[(size_t) rs] = F->clock;
                R.st[(size_t) rs] = FastState::kRDemote;
                F->ram_of[(size_t) vkey] = rs;
                cudaMemcpyAsync(R.base + (size_t) rs * R.stride, P.slot_ptr(v),
                                F->L[(size_t) il].blob, cudaMemcpyDeviceToHost, F->copy);
                cudaEvent_t ev = F->get_event();
                cudaEventRecord(ev, F->copy);
                F->draining.push_back(FastState::Drain{il, v, vkey, F->layer_rc[(size_t) il], rs, ev});
                P.st[(size_t) v] = FastState::kDraining;
                P.key[(size_t) v] = -1;
                F->demotions.fetch_add(1, std::memory_order_relaxed);
            } else {
                // colder than everything in RAM: dropped (disk only); the slot can be a spare right away - the
                // table edit that forgets it runs before any route of the next token
                P.st[(size_t) v] = FastState::kSpare;
                P.key[(size_t) v] = -1;
                P.spare[j] = v;
                upd(F->spare_key(il, j), (unsigned long long) P.slot_ptr(v));
                F->drops.fetch_add(1, std::memory_order_relaxed);
                F->left[(size_t) vkey] = 1;
                ++F->diag_drop;
            }
        }
    }
    // (4) a few free RAM slots per class for the next token's disk reads (and its reads ahead)
    static const char* kf = getenv("STRATA_GLM_RAM_FREE");
    const int keep_free = kf ? std::max(1, std::atoi(kf)) : F->ahead_read ? 12 : 4;
    for (auto& R : F->rc) {
        int nfree = 0;
        for (char c : R.st) nfree += c == FastState::kRFree;
        while (nfree < keep_free) {
            int best = -1;
            if (F->ram_shadow) {
                // A shadow is redundant with VRAM, so reclaim it before removing an expert's only RAM copy.
                uint64_t bt = UINT64_MAX;
                for (int s = 0; s < R.n; ++s) {
                    if (R.st[(size_t) s] != FastState::kRHold) continue;
                    const int key = R.key[(size_t) s];
                    if (key >= 0 && F->slot_of[(size_t) key] >= 0 && R.tick[(size_t) s] < bt) {
                        best = s;
                        bt = R.tick[(size_t) s];
                    }
                }
            }
            if (best < 0) best = ram_victim(R);
            if (best < 0) break;
            upd(F->rtab_key(R.key[(size_t) best]), 0ull);
            F->ram_of[(size_t) R.key[(size_t) best]] = -1;
            F->left[(size_t) R.key[(size_t) best]] = 2;
            ++F->diag_ram_evict;
            R.key[(size_t) best] = -1;
            R.st[(size_t) best] = FastState::kRFree;
            ++nfree;
        }
    }
    flush();
    static const bool diag = getenv("STRATA_GLM_TIER_DIAG") != nullptr;
    if (diag && ++F->diag_b % 64 == 0)
        std::fprintf(stderr, "glm tier diag CUDA%d (%llu boundaries): VRAM drops %llu, RAM evictions %llu, lend drops %llu | "
                             "disk reads of: never held %llu, dropped from VRAM %llu, evicted from RAM %llu, lend-dropped %llu\n",
                     dev_, (unsigned long long) F->diag_b, (unsigned long long) F->diag_drop,
                     (unsigned long long) F->diag_ram_evict, (unsigned long long) F->diag_lend,
                     (unsigned long long) F->diag_disk[0], (unsigned long long) F->diag_disk[1],
                     (unsigned long long) F->diag_disk[2], (unsigned long long) F->diag_disk[3]);
    if (diag && F->diag_b % 64 == 0) {
        // per RAM class: slots, held, free, and how many of its layers' experts are in no tier at all
        for (size_t c = 0; c < F->rc.size(); ++c) {
            const auto& R = F->rc[c];
            int held = 0, nfree = 0, out = 0, layers = 0;
            for (char st : R.st) {
                held += st == FastState::kRHold;
                nfree += st == FastState::kRFree;
            }
            for (int il = l0_; il < lt_; ++il) {
                if (!F->L[(size_t) il].moe || F->layer_rc[(size_t) il] != (int) c) continue;
                ++layers;
                for (int e = 0; e < g.n_expert; ++e)
                    out += F->slot_of[(size_t) il * g.n_expert + e] < 0 && F->ram_of[(size_t) il * g.n_expert + e] < 0;
            }
            std::fprintf(stderr, "glm tier diag class %zu (%d layers, %.2f MB): %d slots, %d held, %d free, %d experts in "
                                 "no tier\n", c, layers, (double) R.stride / 1e6, R.n, held, nfree, out);
        }
        std::fprintf(stderr, "glm tier diag clean-up: %llu duplicate RAM copies freed, %llu lost copies adopted\n",
                     (unsigned long long) F->diag_dup, (unsigned long long) F->diag_adopt);
    }
}

// GLM_CB_DIR seam dumps (the reference path's names, so the same diff script bisects both): synchronous,
// debug only - every token overwrites, so a run of n tokens leaves position n-1's seams
static void fast_dump(cudaStream_t s, const std::string& name, const float* dev, int64_t n) {
    static const char* dir = getenv("GLM_CB_DIR");
    if (!dir || !dir[0]) return;
    cudaStreamSynchronize(s);
    std::vector<float> buf((size_t) n);
    cudaMemcpy(buf.data(), dev, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost);
    if (FILE* f = std::fopen((std::string(dir) + "/" + name + ".f32").c_str(), "wb")) {
        std::fwrite(buf.data(), 4, (size_t) n, f);
        std::fclose(f);
    }
}

// ---------------------------------------------------------------- the forward
// ---- one DSA (MLA + indexer) mixer: F->x / F->xq -> F->mixer at position p (the trunk's DSA layers and the NextN block)
bool Glm5Model::fast_dsa(int il, int64_t p, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    static const bool dumps = getenv("GLM_CB_DIR") != nullptr;
    const auto& Ly = F->L[(size_t) il];
    const float prescale = 1.0f / std::sqrt((float) (g.idx_key * g.idx_heads));
    gf::MvJob j[5];
    j[0] = {Ly.q_a.q, F->xq, nullptr, F->qr_raw, nullptr, 1.0f, Ly.q_a.type, g.n_embd, g.q_lora};
    j[1] = {Ly.kv_a.q, F->xq, nullptr, F->kv_raw, nullptr, 1.0f, Ly.kv_a.type, g.n_embd, g.kv_lora};
    j[2] = {Ly.idx_k, nullptr, F->x, F->ik_raw, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.idx_key};
    j[3] = {Ly.idx_gate, nullptr, F->x, F->ig_raw, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.idx_key};
    j[4] = {Ly.idx_proj, nullptr, F->x, F->iw, nullptr, prescale, gf::kTypeBF16, g.n_embd, g.idx_heads};
    if (!gf::mv(j, 5, s)) { err = "glm fast: dsa projections"; return false; }
    if (F->prof_on) F->mark("dsa_proj_mv");
    gf::DsaPrepArgs d;
    d.qr_raw = F->qr_raw;
    d.q_a_norm = Ly.q_a_norm;
    d.qr = F->qr;
    d.qr_q = F->qr_q;
    d.q_lora = g.q_lora;
    d.kv_raw = F->kv_raw;
    d.kv_norm = Ly.kv_a_norm;
    d.lat = (uint16_t*) (state_ + dsa_lat_[(size_t) il]);
    d.kv_lora = g.kv_lora;
    d.ik_raw = F->ik_raw;
    d.k_norm_w = Ly.k_norm_w;
    d.k_norm_b = Ly.k_norm_b;
    d.ik_cache = state_ + dsa_ik_[(size_t) il];
    d.ig_raw = F->ig_raw;
    d.ig_cache = state_ + dsa_ig_[(size_t) il];
    d.ape = Ly.ape;
    d.pooled = state_ + dsa_pool_[(size_t) il];
    d.idx_key = g.idx_key;
    d.kpool = g.idx_kpool;
    d.p = (int) p;
    d.eps = g.norm_eps;
    gf::dsa_prep(d, s);
    if (F->prof_on) F->mark("dsa_prep");
    gf::MvJob j2[2];
    j2[0] = {Ly.q_b.q, F->qr_q, nullptr, F->q, nullptr, 1.0f, Ly.q_b.type, g.q_lora, g.n_head * g.qk_nope};
    j2[1] = {Ly.idx_q_b, nullptr, F->qr, F->iq, nullptr, 1.0f, gf::kTypeBF16, g.q_lora, g.idx_heads * g.idx_key};
    if (!gf::mv(j2, 2, s)) { err = "glm fast: dsa q"; return false; }
    if (F->prof_on) F->mark("dsa_q_mv");
    if (dumps) {
        fast_dump(s, "dsa_qr-" + std::to_string(il), F->qr, g.q_lora);
        fast_dump(s, "dsa_q-" + std::to_string(il), F->q, (int64_t) g.n_head * g.qk_nope);
        fast_dump(s, "dsa_iq-" + std::to_string(il), F->iq, (int64_t) g.idx_heads * g.idx_key);
    }
    const int pool_done = (int) ((p + 1) / g.idx_kpool);
    if (pool_done > 0)
        gf::dsa_score(F->iq, state_ + dsa_pool_[(size_t) il], F->iw, g.idx_key, g.idx_heads, pool_done,
                      F->score, s);
    const int top_pools = std::min(g.top_pools_max(), pool_done);
    const int n_sel = g.idx_kpool * top_pools + (g.idx_select_tail ? g.idx_kpool - 1 : 0);
    gf::dsa_select(F->score, pool_done, g.idx_kpool, top_pools, n_sel, (int) p, F->cells, s);
    if (F->prof_on) F->mark("dsa_score_select");
    gf::mla(F->q, Ly.k_b, Ly.v_b, (const uint16_t*) (state_ + dsa_lat_[(size_t) il]), F->cells, n_sel, g.n_head, g.qk_nope,
            g.kv_lora, g.v_head, F->attn_q, s);
    if (F->prof_on) F->mark("mla");
    gf::MvJob o = {Ly.out.q, F->attn_q, nullptr, F->mixer, nullptr, 1.0f, Ly.out.type,
                   g.n_head * g.v_head, g.n_embd};
    if (!gf::mv(&o, 1, s)) { err = "glm fast: dsa output"; return false; }
    if (F->prof_on) F->mark("dsa_out_mv");
    return true;
}

// ---- one MoE FFN: F->x / F->xq -> F->ffn (routes, tiers, routed + shared experts)
// The NextN draft block leaves its experts that are not in VRAM out (no RAM pull, no disk wait): its draft only has
// to be a good guess - every token is the trunk's own - and its misses sat on the tail's critical path.
// STRATA_GLM_MTP_MISS=1 fetches them like the trunk.
static bool mtp_skip_miss() {
    static const bool v = getenv("STRATA_GLM_MTP_MISS") == nullptr;
    return v;
}

bool Glm5Model::fast_moe(int il, bool& pf_pending, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const auto& Ly = F->L[(size_t) il];
    const int FFs = g.n_ff_exp * g.n_shared;
    gf::MvJob j[gf::kMaxMvJobs];
    int nj = 3;
    j[0] = {Ly.router, nullptr, F->x, F->rlog, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.n_expert};
    j[1] = {Ly.sh_gate.q, F->xq, nullptr, F->sh_g, nullptr, 1.0f, Ly.sh_gate.type, g.n_embd, FFs};
    j[2] = {Ly.sh_up.q, F->xq, nullptr, F->sh_u, nullptr, 1.0f, Ly.sh_up.type, g.n_embd, FFs};
    // the next layer's router on THIS layer's FFN input: its experts, one layer early
    // (only with prefetch on: the prediction costs a router GEMV and a second top-k per layer)
    const bool pred = F->max_pf > 0 && il + 1 < l1_ && F->L[(size_t) il + 1].moe;
    if (pred)
        j[nj++] = {F->L[(size_t) il + 1].router, nullptr, F->x, F->plog, nullptr, 1.0f, gf::kTypeBF16, g.n_embd,
                   g.n_expert};
    // LOOKAHEAD: the next layers' routers on this input too (their disk-only experts are read ahead)
    int n_ah = 0;
    const float* ah_bias[gf::kAhead] = {nullptr, nullptr, nullptr, nullptr};
    for (int d = 0; d < F->n_ahead && il + 1 + d < l1_ && F->L[(size_t) il + 1 + d].moe && nj < gf::kMaxMvJobs; ++d) {
        const auto& La = F->L[(size_t) il + 1 + d];
        j[nj++] = {La.router, nullptr, F->x, F->alog + (size_t) d * g.n_expert, nullptr, 1.0f, gf::kTypeBF16, g.n_embd,
                   g.n_expert};
        ah_bias[d] = La.router_bias;
        ++n_ah;
    }
    if (!gf::mv(j, nj, s)) { err = "glm fast: router / shared expert"; return false; }
    if (F->prof_on) F->mark("router_shexp_mv");
    // the prefetch list of THIS route lives in the buffer of this layer's parity (the side stream may still
    // read the previous layer's list while this route writes)
    gf::MoeDev md = F->md;
    md.pf_src = F->pf_buf[il & 1];
    md.pf_dst = F->pf_buf[il & 1] + 8;
    md.pf_n = F->pf_n_buf[il & 1];
    // the CPU lane splits this layer's RAM-tier experts when the CPU has a dot product for its types
    const bool lane = F->cpu_plan != 0ull && F->cpu_fmt[(size_t) il].n_ff > 0;
    gf::moe_route(F->rlog, Ly.router_bias, g.n_expert, g.n_exp_used, g.w_scale, g.norm_w != 0, il, F->x,
                  g.n_embd, md, F->sh_g, F->sh_u, g.swiglu_shexp, FFs, F->sh_hq, s,
                  pred ? F->plog : nullptr, pred ? F->L[(size_t) il + 1].router_bias : nullptr,
                  pred ? F->max_pf : 0, n_ah > 0 ? F->alog : nullptr, ah_bias, n_ah, il == mtp_il_ && mtp_skip_miss(),
                  lane ? F->cpu_plan : 0ull);
    if (F->prof_on) F->mark("moe_route");
    ++F->expected;
    if (pred && F->max_pf > 0) {
        // the side stream copies the next layer's predicted experts while this layer computes
        cudaEventRecord(F->ev_pred, s);
        cudaStreamWaitEvent(F->ps, F->ev_pred, 0);
        gf::moe_prefetch(md, F->L[(size_t) il + 1].blob, F->ps);
        cudaEventRecord(F->ev_pf, F->ps);
    }
    // disk-only experts (rare once warm) park the device until the host has read them; everything not
    // in VRAM is then pulled over PCIe into its slot, and all 8 run from VRAM
    gf::moe_wait(md, g.n_embd, s);
    if (F->prof_on) F->mark("moe_wait");
    gf::moe_fetch(md, g.n_exp_used, Ly.blob, s);
    if (F->prof_on) F->mark("moe_fetch");
    // this layer's own prefetched experts (issued one layer ago) must have landed before they are read
    if (pf_pending) cudaStreamWaitEvent(s, F->ev_pf_prev, 0);
    pf_pending = false;
    // (the shared expert's down rides the gate/up launch: sh_out)
    gf::moe_gate_up(Ly.gu_type, md, g.n_exp_used, g.n_embd, g.n_ff_exp, g.swiglu_exp, F->xq, F->hq, Ly.sh_down.q,
                    Ly.sh_down.type, F->sh_hq, FFs, F->sh_out, s);
    if (F->prof_on) F->mark("moe_gate_up");
    gf::moe_down(Ly.d_type, md, g.n_exp_used, g.n_embd, g.n_ff_exp, Ly.down_off, F->hq,
                 Ly.sh_down.q != nullptr ? F->sh_out : nullptr, F->ffn, s);
    if (F->prof_on) F->mark("moe_down");
    // the CPU lane's experts last: the device's own down rows ran while the CPU worked
    if (lane) {
        gf::moe_cpu_wait(md, g.n_embd, F->ffn, s);
        if (F->prof_on) F->mark("moe_cpu_wait");
    }
    if (pred && F->max_pf > 0) {
        std::swap(F->ev_pf, F->ev_pf_prev);   // the next layer waits on THIS layer's prefetch
        pf_pending = true;
    }
    return true;
}

bool Glm5Model::fast_layers(int64_t p, bool hop_in, std::string& err) {
    (void) hop_in;
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    static const bool dumps = getenv("GLM_CB_DIR") != nullptr;
    bool pf_pending = false;
    float* Rc = state_;
    float* Ro = state_ + (int64_t) g.hc * g.n_embd;
    if (F->prof_on) F->mark("start");
    for (int il = l0_; il < l1_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        // ---- the attention-side read (fused with the previous FFN's write)
        gf::HcArgs h;
        h.block_out = il == l0_ ? nullptr : F->ffn;
        h.R_old = Rc;
        h.R_new = Ro;
        h.post_in = F->post;
        h.comb_in = F->comb;
        h.pre = F->pre;
        h.post = F->post;
        h.comb = F->comb;
        h.w_fn = Ly.hc_attn_fn;
        h.w_scale = Ly.hc_attn_scale;
        h.w_base = Ly.hc_attn_base;
        h.norm_w = Ly.attn_norm;
        h.norm_eps = g.norm_eps;
        h.hc_eps = g.hc_eps;
        h.iters = g.sinkhorn_iters;
        h.n_embd = g.n_embd;
        h.x = F->x;
        h.xq = F->xq;
        h.part = F->part;
        h.counter = F->counter;
        gf::hc(h, s);
        if (F->prof_on) F->mark("hc_attn");
        if (il != l0_) std::swap(Rc, Ro);
        if (dumps) {
            const std::string Ls = std::to_string(il);
            if (il != l0_) fast_dump(s, "l_out-" + std::to_string(il - 1), Rc, (int64_t) g.hc * g.n_embd);
            fast_dump(s, "attn_norm-" + Ls, F->x, g.n_embd);
        }

        if (Ly.recr) {
            const int DI = g.d_inner();
            gf::MvJob j[6];
            j[0] = {Ly.q.q, F->xq, nullptr, F->proj[0], nullptr, 1.0f, Ly.q.type, g.n_embd, DI};
            j[1] = {Ly.k.q, F->xq, nullptr, F->proj[1], nullptr, 1.0f, Ly.k.type, g.n_embd, DI};
            j[2] = {Ly.v.q, F->xq, nullptr, F->proj[2], nullptr, 1.0f, Ly.v.type, g.n_embd, DI};
            j[3] = {Ly.f_a, nullptr, F->x, F->fa, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.kda_head_dim};
            j[4] = {Ly.g_a, nullptr, F->x, F->ga, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.kda_head_dim};
            j[5] = {Ly.beta, nullptr, F->x, F->beta, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.n_head};
            if (!gf::mv(j, 6, s)) { err = "glm fast: kda projections"; return false; }
            if (F->prof_on) F->mark("kda_proj_mv");
            gf::KdaPrepArgs k;
            for (int i = 0; i < 3; ++i) {
                k.proj[i] = F->proj[i];
                k.conv_w[i] = Ly.conv[i];
                k.out[i] = F->conv[i];
            }
            k.conv_state = state_ + kda_conv_[(size_t) il];
            k.fa = F->fa;
            k.ga = F->ga;
            k.f_b = Ly.f_b;
            k.g_b = Ly.g_b;
            k.dt_bias = Ly.dt_bias;
            k.ssm_a = Ly.ssm_a;
            k.lower_bound = g.kda_lb;
            k.g1 = F->g1;
            k.g2 = F->g2;
            k.n_head = g.n_head;
            k.head_dim = g.kda_head_dim;
            k.d_conv = g.d_conv;
            gf::kda_prep(k, s);
            if (F->prof_on) F->mark("kda_prep");
            gf::kda_rec(F->conv[0], F->conv[1], F->conv[2], F->g1, F->beta, state_ + kda_S_[(size_t) il], F->g2,
                        Ly.ssm_norm, g.norm_eps, g.n_head, g.kda_head_dim, F->gated_q, s);
            if (F->prof_on) F->mark("kda_rec");
            gf::MvJob o = {Ly.out.q, F->gated_q, nullptr, F->mixer, nullptr, 1.0f, Ly.out.type, DI, g.n_embd};
            if (!gf::mv(&o, 1, s)) { err = "glm fast: kda output"; return false; }
            if (F->prof_on) F->mark("kda_out_mv");
        } else {
            if (!fast_dsa(il, p, err)) return false;
        }

        // ---- the FFN-side read (fused with the mixer's write)
        h.block_out = F->mixer;
        h.R_old = Rc;
        h.R_new = Ro;
        h.w_fn = Ly.hc_ffn_fn;
        h.w_scale = Ly.hc_ffn_scale;
        h.w_base = Ly.hc_ffn_base;
        h.norm_w = Ly.ffn_norm;
        gf::hc(h, s);
        if (F->prof_on) F->mark("hc_ffn");
        std::swap(Rc, Ro);
        if (dumps) {
            const std::string Ls = std::to_string(il);
            fast_dump(s, "hc_attn_post-" + Ls, Rc, (int64_t) g.hc * g.n_embd);
            fast_dump(s, "ffn_norm-" + Ls, F->x, g.n_embd);
            fast_dump(s, "mixer-" + Ls, F->mixer, g.n_embd);
        }

        if (!Ly.moe) {
            gf::MvJob j[2];
            j[0] = {Ly.ffn_gate.q, F->xq, nullptr, F->dg, nullptr, 1.0f, Ly.ffn_gate.type, g.n_embd, g.n_ff_dense};
            j[1] = {Ly.ffn_up.q, F->xq, nullptr, F->du, nullptr, 1.0f, Ly.ffn_up.type, g.n_embd, g.n_ff_dense};
            if (!gf::mv(j, 2, s)) { err = "glm fast: dense ffn"; return false; }
            if (F->prof_on) F->mark("dense_gu_mv");
            gf::swiglu_q8(F->dg, F->du, g.swiglu_shexp, g.n_ff_dense, F->dhq, s);
            if (F->prof_on) F->mark("dense_swiglu");
            gf::MvJob dn = {Ly.ffn_down.q, F->dhq, nullptr, F->ffn, nullptr, 1.0f, Ly.ffn_down.type, g.n_ff_dense,
                            g.n_embd};
            if (!gf::mv(&dn, 1, s)) { err = "glm fast: dense ffn down"; return false; }
            if (F->prof_on) F->mark("dense_down_mv");
        } else {
            if (!fast_moe(il, pf_pending, err)) return false;
        }
        if (dumps) fast_dump(s, "ffn_out-" + std::to_string(il), F->ffn, g.n_embd);
    }
    // the last layer's write half: R (in the OTHER buffer) = post x ffn + comb . R
    gf::hc_post(F->ffn, Rc, F->post, F->comb, g.n_embd, Ro, s);
    if (F->prof_on) F->mark("hc_post_end");
    // keep the convention "the residual lives at state_" for the hop and the head: copy back
    if (Ro != state_)
        cudaMemcpyAsync(state_, Ro, (size_t) g.hc * g.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, s);
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { err = std::string("glm fast: ") + cudaGetErrorString(e); return false; } }
    return true;
}

bool Glm5Model::fast_token(int32_t token, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const int64_t p = pos_++;
    // between tokens (every device idle): finished promotions go live in the device tables
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        m->fast_boundary();
    }
    cudaSetDevice(dev_);
    // ---- the embedding row (host-dequantized from the shard mapping) -> the 4 streams
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
        err = "glm fast: no embedding dequantizer";
        return false;
    }
    tt->to_float(pack_emb_src_ + (size_t) token * strata::kernels::iq_row_bytes(pack_emb_type_, g.n_embd), F->emb_h,
                 g.n_embd);
    if (const float* img = image_row(p)) std::memcpy(F->emb_h, img, (size_t) g.n_embd * sizeof(float));   // an image
    cudaMemcpyAsync(F->emb, F->emb_h, (size_t) g.n_embd * sizeof(float), cudaMemcpyHostToDevice, F->cs);
    gf::embed_streams(F->emb, state_, g.n_embd, F->cs);
    if (!fast_layers(p, false, err)) return false;
    Glm5Model* tail = this;
    if (split_next_) {
        Glm5Model* B = split_next_.get();
        FastState* FB = B->fast_;
        const size_t hop = (size_t) g.hc * g.n_embd * sizeof(float);
        cudaMemcpyAsync(F->hop_h, state_, hop, cudaMemcpyDeviceToHost, F->cs);
        cudaEventRecord(F->ev_hop, F->cs);
        cudaSetDevice(B->dev_);
        B->pos_ = pos_;
        cudaStreamWaitEvent(FB->cs, F->ev_hop, 0);
        cudaMemcpyAsync(B->state_, F->hop_h, hop, cudaMemcpyHostToDevice, FB->cs);
        if (!B->fast_layers(p, true, err)) return false;
        tail = B;
    }
    FastState* FT = tail->fast_;
    cudaSetDevice(tail->dev_);
    gf::head_prep(tail->state_, tail->w_.at("output_norm.weight"), g.norm_eps, g.n_embd, FT->head_x, FT->head_xq,
                  FT->cs);
    // GLM_DUMP_H=<file> (debug): append every position's final hidden state (output_norm of the stream mean - what the
    // NextN/MTP block reads) as n_embd floats
    static const char* dh = getenv("GLM_DUMP_H");
    if (dh != nullptr && dh[0]) {
        cudaStreamSynchronize(FT->cs);
        std::vector<float> hb((size_t) g.n_embd);
        cudaMemcpy(hb.data(), FT->head_x, hb.size() * sizeof(float), cudaMemcpyDeviceToHost);
        if (FILE* f = std::fopen(dh, "ab")) {
            std::fwrite(hb.data(), sizeof(float), hb.size(), f);
            std::fclose(f);
        }
    }
    const WSlot& ow = tail->ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, FT->head_xq, FT->head_x, tail->sc_ + tail->sc_logits, nullptr, 1.0f, ow.type, g.n_embd,
                   g.n_vocab};
    if (ow.type == 0) o.w = ow.f32;
    if (!gf::mv(&o, 1, FT->cs)) {
        err = "glm fast: output head (type " + std::to_string(ow.type) + ")";
        return false;
    }
    gf::argmax(tail->sc_ + tail->sc_logits, g.n_vocab, FT->tok, FT->cs);
    cudaMemcpyAsync(FT->tok_h, FT->tok, sizeof(int), cudaMemcpyDeviceToHost, FT->cs);
    cudaEventRecord(FT->ev_done, FT->cs);
    // wait for the token (the service threads answer misses meanwhile)
    const auto tw = std::chrono::steady_clock::now();
    bool warned = false;
    while (cudaEventQuery(FT->ev_done) == cudaErrorNotReady) {
        std::this_thread::yield();
        if (!warned && std::chrono::steady_clock::now() - tw > std::chrono::seconds(10)) {
            warned = true;
            for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
                const FastState* Fm = m->fast_;
                unsigned int last = 0;
                for (int i = 0; i < gf::kRingSize; ++i) last = std::max(last, (unsigned int) Fm->ring_h[i].seq);
                std::fprintf(stderr, "glm fast: token %lld waiting > 10 s - CUDA%d last route seq %u, last answer %u\n",
                             (long long) p, m->dev_, last, (unsigned int) Fm->resp_h->seq);
            }
        }
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("glm fast: ") + cudaGetErrorString(e);
        return false;
    }
    if (gf::launch_errors() > 0) {
        err = "glm fast: " + std::to_string(gf::launch_errors()) + " kernel launch(es) failed (see stderr)";
        return false;
    }
    last_tok_ = FT->tok_h[0];
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (m->fast_ && m->fast_->prof_on) {
            cudaSetDevice(m->dev_);
            m->fast_->collect();
        }
    cudaSetDevice(dev_);
    return true;
}

// ---------------------------------------------------------------- the NextN draft block
// At position p (whose final hidden state the head left in head_x): eh_proj([enorm(emb(next)), hnorm(h_p)]) -> a DSA
// mixer and a MoE FFN with plain pre-norm residuals (the trunk's own kernels: fast_dsa / fast_moe on the block's layer
// index; its caches at p) -> shared_head_norm -> the output head -> argmax into mtp_tok (mtp_tok_h after ev_mtp).
bool Glm5Model::fast_mtp(int64_t p, int32_t next_tok, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const int il = mtp_il_;
    if (F == nullptr || il < 0) {
        err = "glm mtp: no draft block on this half";
        return false;
    }
    const auto& Ly = F->L[(size_t) il];
    const int E = g.n_embd;
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
        err = "glm mtp: no embedding dequantizer";
        return false;
    }
    cudaStreamSynchronize(s);   // emb_h may still feed an earlier copy
    tt->to_float(pack_emb_src_ + (size_t) next_tok * strata::kernels::iq_row_bytes(pack_emb_type_, E), F->emb_h, E);
    cudaMemcpyAsync(F->emb, F->emb_h, (size_t) E * sizeof(float), cudaMemcpyHostToDevice, s);
    gf::mtp_in(F->emb, F->head_x, Ly.enorm, Ly.hnorm, g.norm_eps, E, F->mtp_catq, s);
    gf::MvJob eh = {Ly.eh.q, F->mtp_catq, nullptr, F->mtp_h, nullptr, 1.0f, Ly.eh.type, 2 * E, E};
    if (!gf::mv(&eh, 1, s)) {
        err = "glm mtp: eh_proj";
        return false;
    }
    gf::rms_q8(F->mtp_h, nullptr, Ly.attn_norm, g.norm_eps, E, F->x, F->xq, s);
    if (!fast_dsa(il, p, err)) return false;
    gf::rms_q8(F->mtp_h, F->mixer, Ly.ffn_norm, g.norm_eps, E, F->x, F->xq, s);
    bool pf_pending = false;
    if (!fast_moe(il, pf_pending, err)) return false;
    gf::rms_q8(F->mtp_h, F->ffn, Ly.shnorm, g.norm_eps, E, F->head_x, F->head_xq, s);
    const WSlot& ow = ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, F->head_xq, F->head_x, F->mtp_logits, nullptr, 1.0f, ow.type, E, g.n_vocab};
    if (ow.type == 0) o.w = ow.f32;
    if (!gf::mv(&o, 1, s)) {
        err = "glm mtp: the head";
        return false;
    }
    gf::argmax(F->mtp_logits, g.n_vocab, F->mtp_tok, s);
    cudaMemcpyAsync(F->mtp_tok_h, F->mtp_tok, sizeof(int), cudaMemcpyDeviceToHost, s);
    cudaEventRecord(F->ev_mtp, s);
    return true;
}

bool Glm5Model::has_mtp() const {
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (m->fast_ != nullptr && m->mtp_il_ >= 0) return true;
    return false;
}

int Glm5Model::mtp_draft(int32_t next_tok, std::string& err) {
    Glm5Model* t = nullptr;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (m->fast_ != nullptr && m->mtp_il_ >= 0) t = m;
    if (t == nullptr) return -1;
    cudaSetDevice(t->dev_);
    // between tokens: the draft block's routes go through the same tiers, so the boundary runs first
    t->fast_boundary();
    if (!t->fast_mtp(t->pos_ - 1, next_tok, err)) {
        cudaSetDevice(dev_);
        return -1;
    }
    while (cudaEventQuery(t->fast_->ev_mtp) == cudaErrorNotReady) std::this_thread::yield();
    const int d = t->fast_->mtp_tok_h[0];
    cudaSetDevice(dev_);
    if (gf::launch_errors() > 0) {
        err = "glm mtp: a kernel launch failed (see stderr)";
        return -1;
    }
    return d;
}

// ---------------------------------------------------------------- pipelined speculative decode (two halves + NextN)
// The two halves of a split take turns on one token, so each GPU idles half the time.  With the NextN block's draft
// for the token after next, the HEAD half runs that draft while the TAIL half finishes the current token: when the
// tail's token equals the draft, the head's work stands and both GPUs stay busy (a token costs the slower half);
// when it differs, the head restores its recurrent states (saved before the speculative token) and reruns the
// position with the real token.  Only the head speculates - the tail, the NextN block and the sampler see confirmed
// tokens only - so the output is exactly what the token-at-a-time decode would produce for the same samples.
bool Glm5Model::spec_ready() const {
    const Glm5Model* B = split_next_.get();
    return fast_ != nullptr && B != nullptr && B->split_next_ == nullptr && B->fast_ != nullptr && B->mtp_il_ >= 0 &&
           getenv("STRATA_GLM_NO_SPEC") == nullptr;
}

// the head's KDA states (S + conv history of every recurrent layer here: one contiguous run each) <-> kda_bak_
bool Glm5Model::spec_kda_copy(bool restore) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const size_t run = (size_t) g.d_inner() * g.kda_head_dim + (size_t) 3 * g.d_inner() * (g.d_conv - 1);
    int n_rec = 0;
    for (int il = l0_; il < l1_; ++il) n_rec += g.is_recr(il);
    if (n_rec == 0) return true;
    if (kda_bak_ == nullptr && cudaMalloc(&kda_bak_, (size_t) n_rec * run * sizeof(float)) != cudaSuccess) {
        cudaGetLastError();
        kda_bak_ = nullptr;
        return false;
    }
    int k = 0;
    for (int il = l0_; il < l1_; ++il)
        if (g.is_recr(il)) {
            float* live = state_ + kda_S_[(size_t) il];
            float* bak = kda_bak_ + (size_t) k++ * run;
            cudaMemcpyAsync(restore ? live : bak, restore ? bak : live, run * sizeof(float), cudaMemcpyDeviceToDevice,
                            F->cs);
        }
    return true;
}

// the head half's layers for (p, token); the residual rows go to hop slot p & 1 (spec_ev_hop_ marks them)
bool Glm5Model::spec_head(int64_t p, int32_t token, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaSetDevice(dev_);
    static const bool sprof = getenv("STRATA_GLM_SPEC_PROF") != nullptr;
    auto t0 = std::chrono::steady_clock::now();
    const auto lap = [&](int i) {
        if (!sprof) return;
        const auto t = std::chrono::steady_clock::now();
        spec_t_[i] += std::chrono::duration<double, std::milli>(t - t0).count();
        t0 = t;
    };
    cudaStreamSynchronize(F->cs);   // the boundary needs this device idle (and emb_h free)
    lap(0);
    if (sprof) {
        if (spec_ev_[0] == nullptr)
            for (auto& e : spec_ev_) cudaEventCreate(&e);
        else if (spec_ev_live_) {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, spec_ev_[0], spec_ev_[1]);
            spec_dev_ms_ += ms;
            ++spec_dev_n_;
        }
        if (spec_ev_live_) {
            float gap = 0.0f;
            cudaEventRecord(spec_ev_[2], F->cs);   // now: the gap since the last token ended
            cudaEventSynchronize(spec_ev_[2]);
            cudaEventElapsedTime(&gap, spec_ev_[1], spec_ev_[2]);
            spec_gap_ms_ += gap;
        }
        cudaEventRecord(spec_ev_[0], F->cs);
    }
    if (F->prof_on) F->collect();
    fast_boundary();
    lap(1);
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    tt->to_float(pack_emb_src_ + (size_t) token * strata::kernels::iq_row_bytes(pack_emb_type_, g.n_embd), F->emb_h,
                 g.n_embd);
    if (const float* img = image_row(p)) std::memcpy(F->emb_h, img, (size_t) g.n_embd * sizeof(float));   // an image
    cudaMemcpyAsync(F->emb, F->emb_h, (size_t) g.n_embd * sizeof(float), cudaMemcpyHostToDevice, F->cs);
    gf::embed_streams(F->emb, state_, g.n_embd, F->cs);
    lap(2);
    if (!fast_layers(p, false, err)) return false;
    lap(3);
    ++spec_n_;
    const int sl = (int) (p & 1);
    cudaMemcpyAsync(spec_hop_h_[sl], state_, (size_t) g.hc * g.n_embd * sizeof(float), cudaMemcpyDeviceToHost, F->cs);
    cudaEventRecord(spec_ev_hop_[sl], F->cs);
    if (sprof) {
        cudaEventRecord(spec_ev_[1], F->cs);
        spec_ev_live_ = true;
    }
    return true;
}

// the tail half (this) for position p: waits on the head's hop slot ON THE DEVICE, then its layers, the head and the
// argmax (tok_h after ev_done)
bool Glm5Model::spec_tail(Glm5Model* head, int64_t p, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaSetDevice(dev_);
    static const bool sprof = getenv("STRATA_GLM_SPEC_PROF") != nullptr;
    auto t0 = std::chrono::steady_clock::now();
    const auto lap = [&](int i) {
        if (!sprof) return;
        const auto t = std::chrono::steady_clock::now();
        spec_t_[i] += std::chrono::duration<double, std::milli>(t - t0).count();
        t0 = t;
    };
    cudaStreamSynchronize(F->cs);
    lap(0);
    if (F->prof_on) F->collect();
    fast_boundary();
    lap(1);
    ++spec_n_;
    pos_ = p + 1;
    const int sl = (int) (p & 1);
    if (sprof) {
        if (spec_ev_[0] == nullptr)
            for (auto& e : spec_ev_) cudaEventCreate(&e);
        else if (spec_ev_live_) {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, spec_ev_[0], spec_ev_[1]);
            spec_dev_ms_ += ms;
            ++spec_dev_n_;
        }
    }
    cudaStreamWaitEvent(F->cs, head->spec_ev_hop_[sl], 0);
    if (sprof) cudaEventRecord(spec_ev_[0], F->cs);
    cudaMemcpyAsync(state_, head->spec_hop_h_[sl], (size_t) g.hc * g.n_embd * sizeof(float), cudaMemcpyHostToDevice,
                    F->cs);
    if (!fast_layers(p, true, err)) return false;
    gf::head_prep(state_, w_.at("output_norm.weight"), g.norm_eps, g.n_embd, F->head_x, F->head_xq, F->cs);
    const WSlot& ow = ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, F->head_xq, F->head_x, sc_ + sc_logits, nullptr, 1.0f, ow.type, g.n_embd, g.n_vocab};
    if (ow.type == 0) o.w = ow.f32;
    if (!gf::mv(&o, 1, F->cs)) {
        err = "glm spec: output head";
        return false;
    }
    gf::argmax(sc_ + sc_logits, g.n_vocab, F->tok, F->cs);
    cudaMemcpyAsync(F->tok_h, F->tok, sizeof(int), cudaMemcpyDeviceToHost, F->cs);
    cudaEventRecord(F->ev_done, F->cs);
    if (sprof) {
        cudaEventRecord(spec_ev_[1], F->cs);
        spec_ev_live_ = true;
    }
    return true;
}

bool Glm5Model::decode_spec(strata::kernels::SamplerParams& sp, int64_t max_new, const std::function<bool(int)>& emit,
                            int64_t& produced, std::string& err) {
    produced = 0;
    Glm5Model* B = split_next_.get();
    if (!spec_ready()) {
        err = "glm spec: needs a two-half split with the NextN block";
        return false;
    }
    const Glm5Geometry& g = g_;
    if (spec_hop_h_[0] == nullptr) {
        cudaSetDevice(dev_);
        for (int i = 0; i < 2; ++i)
            if (cudaHostAlloc((void**) &spec_hop_h_[i], (size_t) g.hc * g.n_embd * sizeof(float), cudaHostAllocPortable) !=
                    cudaSuccess ||
                cudaEventCreateWithFlags(&spec_ev_hop_[i], cudaEventDisableTiming) != cudaSuccess) {
                err = "glm spec: the hop slots did not allocate";
                return false;
            }
    }
    // the tail's token for its last position: the argmax it left, or a sample from its logits
    const auto take = [&](int& y) -> bool {
        if (sp.greedy) {
            cudaSetDevice(B->dev_);
            while (cudaEventQuery(B->fast_->ev_done) == cudaErrorNotReady) std::this_thread::yield();
            y = B->fast_->tok_h[0];
        } else {
            cudaSetDevice(B->dev_);
            cudaEventSynchronize(B->fast_->ev_done);
            strata::kernels::sample_tokens(B->sc_ + B->sc_logits, 1, (int) g.n_vocab, nullptr, 0, sp, B->d_tok_, nullptr);
            if (cudaMemcpy(&y, B->d_tok_, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
                err = "glm spec: sampler copy";
                return false;
            }
        }
        y = forced(y);
        sp.counter += 1;
        last_tok_ = y;
        return true;
    };
    const auto draft = [&](int64_t p, int32_t next, int& d) -> bool {
        cudaSetDevice(B->dev_);
        if (!B->fast_mtp(p, next, err)) return false;
        while (cudaEventQuery(B->fast_->ev_mtp) == cudaErrorNotReady) std::this_thread::yield();
        d = B->fast_->mtp_tok_h[0];
        return true;
    };
    const auto check = [&]() -> bool {
        if (gf::launch_errors() > 0) {
            err = "glm spec: a kernel launch failed (see stderr)";
            return false;
        }
        return true;
    };
    // the prompt's last forward left position q = pos_ - 1 processed by both halves and its logits on the tail
    int64_t q = pos_ - 1;
    int y = -1;
    if (sp.greedy && last_tok_ >= 0) {
        y = forced(last_tok_);
        sp.counter += 1;
    } else {
        cudaSetDevice(B->dev_);
        cudaEventRecord(B->fast_->ev_done, B->fast_->cs);
        if (!take(y)) return false;
    }
    ++produced;
    if (!emit(y) || produced >= max_new || q + 2 >= max_ctx_) return true;
    int d = -1;
    if (!draft(q, y, d)) return false;
    // the head: position q+1 with the confirmed token, then q+2 with the draft (states saved first)
    if (!spec_head(q + 1, y, err)) return false;
    if (!B->spec_tail(this, q + 1, err)) return false;
    cudaSetDevice(dev_);
    if (!spec_kda_copy(false)) {
        err = "glm spec: the recurrent-state backup did not allocate";
        return false;
    }
    if (!spec_head(q + 2, d, err)) return false;
    bool ok = true;
    // STRATA_GLM_SPEC_PROF=1: host time per phase of a step (debug)
    static const bool sprof = getenv("STRATA_GLM_SPEC_PROF") != nullptr;
    double tp[6] = {0, 0, 0, 0, 0, 0};
    int64_t np = 0;
    auto tnow = std::chrono::steady_clock::now();
    static const char* const lap_names[6] = {"wait tail", "draft", "redo", "enqueue tail", "enqueue head", "emit"};
    const auto lap = [&](int i) {
        if (!sprof) return;
        const auto t = std::chrono::steady_clock::now();
        tp[i] += std::chrono::duration<double, std::milli>(t - tnow).count();
        tnow = t;
#if !defined(STRATA_USE_HIP)
        nvtxRangePop();
        nvtxRangePushA(lap_names[(i + 1) % 6]);
#endif
    };
    // STRATA_GLM_NSYS=<from>,<to>: a profiler capture window over those steps (nsys --capture-range=cudaProfilerApi)
    int64_t nsys_from = -1, nsys_to = -1;
    if (const char* ns = getenv("STRATA_GLM_NSYS")) std::sscanf(ns, "%lld,%lld", (long long*) &nsys_from, (long long*) &nsys_to);
#if !defined(STRATA_USE_HIP)
    if (sprof) nvtxRangePushA("emit");
#endif
    for (;;) {
        if (np == nsys_from) cudaProfilerStart();
        if (np == nsys_to) cudaProfilerStop();
        // (1) the tail's token for position q+1 -> y (the truth for position q+2)
        int y2 = -1;
        lap(5);
        if (!take(y2)) { ok = false; break; }
        lap(0);
        ++produced;
        const bool more = emit(y2) && produced < max_new && q + 3 < max_ctx_;
        if (!more) break;
        // (2) the draft for q+3 from (h_{q+1}, y2)
        int d2 = -1;
        if (!draft(q + 1, y2, d2)) { ok = false; break; }
        lap(1);
        // (3) the head's speculative position q+2 stands iff its draft was y2; else restore and redo it
        ++spec_steps_;
        if (d == y2) {
            ++spec_hits_;
        } else {
            cudaSetDevice(dev_);
            spec_kda_copy(true);
            if (!spec_head(q + 2, y2, err)) { ok = false; break; }
        }
        lap(2);
        // (4) the tail's next position goes in first (it waits for the head's hop on the device), then the head's
        //     next speculative position (its boundary waits for the head's current one)
        if (!B->spec_tail(this, q + 2, err)) { ok = false; break; }
        lap(3);
        cudaSetDevice(dev_);
        spec_kda_copy(false);
        if (!spec_head(q + 3, d2, err)) { ok = false; break; }
        lap(4);
        ++np;
        if (!check()) { ok = false; break; }
        ++q;
        d = d2;
    }
    // drain: both devices idle; the head may stand one or two positions past the tail - a request after this one
    // restores its snapshot or resets, so nothing else reads that state
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        cudaStreamSynchronize(m->fast_->cs);
    }
    cudaSetDevice(dev_);
    pos_ = B->pos_;
    if (sprof && np > 0) {
        std::fprintf(stderr, "glm spec prof (%lld steps, ms/step): wait tail token %.2f | draft %.2f | redo %.2f | "
                             "enqueue tail %.2f | enqueue head %.2f | emit %.2f\n", (long long) np, tp[0] / np, tp[1] / np,
                     tp[2] / np, tp[3] / np, tp[4] / np, tp[5] / np);
        for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
            if (m->spec_n_ > 0)
                std::fprintf(stderr, "glm spec prof CUDA%d (%lld calls, ms/call): sync %.3f | boundary %.3f | embed %.3f | "
                                     "enqueue layers %.3f | DEVICE token %.2f ms, gap before %.2f ms\n", m->dev_,
                             (long long) m->spec_n_, m->spec_t_[0] / m->spec_n_, m->spec_t_[1] / m->spec_n_,
                             m->spec_t_[2] / m->spec_n_, m->spec_t_[3] / m->spec_n_,
                             m->spec_dev_ms_ / (double) std::max<int64_t>(1, m->spec_dev_n_),
                             m->spec_gap_ms_ / (double) std::max<int64_t>(1, m->spec_dev_n_));
    }
    if (ok && !check()) ok = false;
    return ok;
}

// ---------------------------------------------------------------- conversation reuse
// The sequence state that cannot be rebuilt from the position alone is the KDA layers' recurrent state and conv
// history (one contiguous run per KDA layer in state_); the DSA caches are append-only, so restoring the position
// is enough for them (entries past it are rewritten as the new tokens arrive).
bool Glm5Model::snapshot_save() {
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        if (m->fast_ == nullptr) return false;
        const Glm5Geometry& g = m->g_;
        const size_t run = (size_t) g.d_inner() * g.kda_head_dim + (size_t) 3 * g.d_inner() * (g.d_conv - 1);
        int n_rec = 0;
        for (int il = m->l0_; il < m->l1_; ++il) n_rec += g.is_recr(il);
        cudaSetDevice(m->dev_);
        if (m->snap_ == nullptr && n_rec > 0 &&
            cudaMalloc(&m->snap_, (size_t) n_rec * run * sizeof(float)) != cudaSuccess) {
            cudaGetLastError();
            m->snap_ = nullptr;
            cudaSetDevice(dev_);
            return false;
        }
        int k = 0;
        for (int il = m->l0_; il < m->l1_; ++il)
            if (g.is_recr(il))
                cudaMemcpyAsync(m->snap_ + (size_t) k++ * run, m->state_ + m->kda_S_[(size_t) il], run * sizeof(float),
                                cudaMemcpyDeviceToDevice, m->fast_->cs);
        cudaStreamSynchronize(m->fast_->cs);
        m->snap_pos_ = pos_;
    }
    cudaSetDevice(dev_);
    return true;
}

bool Glm5Model::snapshot_restore() {
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (m->fast_ == nullptr || m->snap_pos_ < 0) return false;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        const Glm5Geometry& g = m->g_;
        const size_t run = (size_t) g.d_inner() * g.kda_head_dim + (size_t) 3 * g.d_inner() * (g.d_conv - 1);
        cudaSetDevice(m->dev_);
        int k = 0;
        for (int il = m->l0_; il < m->l1_; ++il)
            if (g.is_recr(il))
                cudaMemcpyAsync(m->state_ + m->kda_S_[(size_t) il], m->snap_ + (size_t) k++ * run, run * sizeof(float),
                                cudaMemcpyDeviceToDevice, m->fast_->cs);
        cudaStreamSynchronize(m->fast_->cs);
        m->pos_ = m->snap_pos_;
    }
    cudaSetDevice(dev_);
    return true;
}

// ---------------------------------------------------------------- conversation slots
// A conversation set aside while others run: per half, the snapshot's KDA states (snap_, taken just before its prompt's
// last token) and every DSA cache's rows up to that position - the latents, the indexer's two key rows and the pools,
// the NextN block's too.  The file: a header (magic, halves, positions, per half its layer range and run sizes), then
// per half the KDA snapshot and the DSA runs in that order.
namespace {
constexpr uint64_t kSlotMagic = 0x3154534C4159414Dull;   // "MAYALST1"

// (state_ offset, floats) of each DSA cache run a slot of n positions holds on one half, in file order
std::vector<std::pair<int64_t, int64_t>> slot_dsa_runs(const Glm5Geometry& g, bool fast_mode, int l0, int l1, int mtp_il,
                                                       const std::vector<int64_t>& lat, const std::vector<int64_t>& ik,
                                                       const std::vector<int64_t>& ig, const std::vector<int64_t>& pool,
                                                       int64_t n) {
    std::vector<std::pair<int64_t, int64_t>> r;
    const int64_t lat_pp = fast_mode ? g.kv_lora / 2 : g.kv_lora;   // FP16 latents take half a float each
    const int64_t pools = (n + g.idx_kpool - 1) / g.idx_kpool;
    const auto dsa = [&](size_t il) {
        r.push_back({lat[il], n * lat_pp});
        r.push_back({ik[il], n * g.idx_key});
        r.push_back({ig[il], n * g.idx_key});
        r.push_back({pool[il], pools * g.idx_key});
    };
    if (mtp_il >= 0) dsa((size_t) mtp_il);
    for (int il = l0; il < l1; ++il)
        if (!g.is_recr(il)) dsa((size_t) il);
    return r;
}
}  // namespace

uint64_t Glm5Model::slot_save(const std::string& path, std::string& err) {
    std::vector<Glm5Model*> halves;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        if (m->fast_ == nullptr || m->snap_pos_ < 0 || m->snap_pos_ != snap_pos_) {
            err = "no snapshot to save";
            return 0;
        }
        halves.push_back(m);
    }
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        err = "cannot write " + path;
        return 0;
    }
    const int64_t n = snap_pos_;
    const auto put = [&](const void* p, size_t bytes) { f.write((const char*) p, (std::streamsize) bytes); };
    const uint32_t nh = (uint32_t) halves.size();
    put(&kSlotMagic, 8);
    put(&nh, 4);
    put(&n, 8);
    std::vector<float> buf;
    double t_sync = 0, t_copy = 0, t_write = 0;
    for (Glm5Model* m : halves) {
        const Glm5Geometry& g = m->g_;
        const int64_t run = (int64_t) g.d_inner() * g.kda_head_dim + (int64_t) 3 * g.d_inner() * (g.d_conv - 1);
        int32_t n_rec = 0;
        for (int il = m->l0_; il < m->l1_; ++il) n_rec += g.is_recr(il);
        const auto runs = slot_dsa_runs(g, m->fast_mode_, m->l0_, m->l1_, m->mtp_il_, m->dsa_lat_, m->dsa_ik_,
                                        m->dsa_ig_, m->dsa_pool_, n);
        const int32_t hdr[4] = {m->l0_, m->l1_, n_rec, (int32_t) runs.size()};
        put(hdr, sizeof(hdr));
        put(&run, 8);
        for (const auto& r : runs) put(&r.second, 8);
        cudaSetDevice(m->dev_);
        auto tq = std::chrono::steady_clock::now();
        cudaStreamSynchronize(m->fast_->cs);
        t_sync += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tq).count();
        std::vector<std::pair<const float*, int64_t>> src;
        if (n_rec > 0) src.push_back({m->snap_, (int64_t) n_rec * run});
        for (const auto& r : runs) src.push_back({m->state_ + r.first, r.second});
        for (const auto& s : src) {
            constexpr int64_t kChunk = (int64_t) 16 << 20;   // floats
            for (int64_t at = 0; at < s.second; at += kChunk) {
                const int64_t c = std::min(kChunk, s.second - at);
                buf.resize((size_t) c);
                tq = std::chrono::steady_clock::now();
                if (cudaMemcpy(buf.data(), s.first + at, (size_t) c * sizeof(float), cudaMemcpyDeviceToHost) !=
                    cudaSuccess) {
                    cudaGetLastError();
                    cudaSetDevice(dev_);
                    err = "the state did not copy back";
                    return 0;
                }
                const auto tw = std::chrono::steady_clock::now();
                t_copy += std::chrono::duration<double, std::milli>(tw - tq).count();
                put(buf.data(), (size_t) c * sizeof(float));
                t_write += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw).count();
            }
        }
    }
    cudaSetDevice(dev_);
    if (getenv("STRATA_GLM_SLOT_TIMING"))
        std::fprintf(stderr, "glm slots: save %lld positions - sync %.0f ms, copy %.0f ms, write %.0f ms\n",
                     (long long) n, t_sync, t_copy, t_write);
    f.flush();
    if (!f) {
        err = "writing " + path + " failed (disk full?)";
        return 0;
    }
    return (uint64_t) f.tellp();
}

bool Glm5Model::slot_load(const std::string& path, int64_t n_pos, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        err = "cannot read " + path;
        return false;
    }
    const auto get = [&](void* p, size_t bytes) { return (bool) f.read((char*) p, (std::streamsize) bytes); };
    uint64_t magic = 0;
    uint32_t nh = 0;
    int64_t n = 0;
    std::vector<Glm5Model*> halves;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) halves.push_back(m);
    if (!get(&magic, 8) || !get(&nh, 4) || !get(&n, 8) || magic != kSlotMagic || nh != halves.size() || n != n_pos ||
        n <= 0 || n > max_ctx_) {
        err = "not a slot of this model";
        return false;
    }
    std::vector<float> buf;
    for (Glm5Model* m : halves) {
        if (m->fast_ == nullptr) {
            err = "no fast path";
            return false;
        }
        const Glm5Geometry& g = m->g_;
        const int64_t run = (int64_t) g.d_inner() * g.kda_head_dim + (int64_t) 3 * g.d_inner() * (g.d_conv - 1);
        int32_t n_rec = 0;
        for (int il = m->l0_; il < m->l1_; ++il) n_rec += g.is_recr(il);
        const auto runs = slot_dsa_runs(g, m->fast_mode_, m->l0_, m->l1_, m->mtp_il_, m->dsa_lat_, m->dsa_ik_,
                                        m->dsa_ig_, m->dsa_pool_, n);
        int32_t hdr[4] = {};
        int64_t frun = 0;
        if (!get(hdr, sizeof(hdr)) || !get(&frun, 8) || hdr[0] != m->l0_ || hdr[1] != m->l1_ || hdr[2] != n_rec ||
            hdr[3] != (int32_t) runs.size() || frun != run) {
            err = "the slot's layout differs from this model's";
            return false;
        }
        for (const auto& r : runs) {
            int64_t fl = 0;
            if (!get(&fl, 8) || fl != r.second) {
                err = "the slot's layout differs from this model's";
                return false;
            }
        }
        cudaSetDevice(m->dev_);
        cudaStreamSynchronize(m->fast_->cs);
        if (m->snap_ == nullptr && n_rec > 0 &&
            cudaMalloc(&m->snap_, (size_t) n_rec * run * sizeof(float)) != cudaSuccess) {
            cudaGetLastError();
            m->snap_ = nullptr;
            cudaSetDevice(dev_);
            err = "the snapshot did not allocate";
            return false;
        }
        std::vector<std::pair<float*, int64_t>> dst;
        if (n_rec > 0) dst.push_back({m->snap_, (int64_t) n_rec * run});
        for (const auto& r : runs) dst.push_back({m->state_ + r.first, r.second});
        for (const auto& d : dst) {
            constexpr int64_t kChunk = (int64_t) 16 << 20;
            for (int64_t at = 0; at < d.second; at += kChunk) {
                const int64_t c = std::min(kChunk, d.second - at);
                buf.resize((size_t) c);
                if (!get(buf.data(), (size_t) c * sizeof(float)) ||
                    cudaMemcpy(d.first + at, buf.data(), (size_t) c * sizeof(float), cudaMemcpyHostToDevice) !=
                        cudaSuccess) {
                    cudaGetLastError();
                    cudaSetDevice(dev_);
                    err = "the slot did not read back";
                    return false;
                }
            }
        }
        m->snap_pos_ = n;
    }
    cudaSetDevice(dev_);
    // the KDA states go from the snapshot into the live state, as snapshot_restore does (it also sets the position)
    if (!snapshot_restore()) {
        err = "the snapshot did not restore";
        return false;
    }
    return true;
}

bool Glm5Model::forward_fast(const std::vector<int32_t>& tokens, std::vector<float>& logits_out, std::string& err) {
    FastState* F = fast_;
    for (int32_t t : tokens) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!fast_token(t, err)) return false;
        F->ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ++F->tokens;
    }
    if (host_logits_) {
        Glm5Model* tail = this;
        while (tail->split_next_) tail = tail->split_next_.get();
        cudaSetDevice(tail->dev_);
        logits_out.resize((size_t) g_.n_vocab);
        if (cudaMemcpy(logits_out.data(), tail->sc_ + tail->sc_logits, (size_t) g_.n_vocab * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            err = "glm fast: logits copy";
            return false;
        }
        cudaSetDevice(dev_);
    }
    return true;
}

}  // namespace strata::core
