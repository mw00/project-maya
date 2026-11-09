// src/core/glm_warm_seq_model.cu - glm-dsa B8.1 (warm): the engine side of the file-order warm-up
// (include/strata/core/glm_warm_seq.hpp): the parts' turn on the disk and the digest of the warmed tiers.
//
//   Turn            fast_warm holds one for the whole warm-up: in a parallel split load with STRATA_GLM_WARM_SEQ the
//                   part waits for the parts before it (SplitLoad::wait_warm) and lets the next one on when it ends
//                   (pass_warm; load_pack_split passes again after load_pack for a part that never warmed).
//   part_digest     every VRAM slot of the part's MoE layers (copied to the host 8 at a time on the copy stream) and
//                   every RAM-tier slot (a node-split slot through glmtier::unified), hashed by the part's disk-read
//                   workers, then folded in slot order with each slot's layer / class, index, expert and state.
//   print_digest    STRATA_GLM_WARM_SUM's line at the end of a part's warm-up (the tiers as the warm-up left them,
//                   before the rest of the load or any token), kept for glmwarm::warm_digest().
// The digest reads the tiers while nothing moves them; it changes nothing.
#include "strata/core/glm_model.hpp"
#include "strata/core/glm_warm_seq.hpp"

#include "glm_fast_state.hpp"
#include "glm_tier_numa.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <vector>

namespace strata::core::glmwarm {

Turn::Turn(Glm5Model& m, bool seq) : m_(m) {
    if (seq && m_.split_load_ != nullptr) {
        if (m_.part_ > 0)
            std::fprintf(stderr, "glm fast: CUDA%d warms its expert tiers after the parts before it "
                                 "(STRATA_GLM_WARM_SEQ: one disk stream)\n", m_.dev_);
        m_.split_load_->wait_warm(m_.part_);
    }
    busy_enter();
}

Turn::~Turn() {
    busy_leave();
    if (m_.split_load_ != nullptr) m_.split_load_->pass_warm(m_.part_);
}

namespace {

// a slot's record: (tier, layer or class, slot, expert key, state, content hash)
uint64_t record(uint64_t h, uint64_t tier, uint64_t where, uint64_t slot, int64_t key, int st, uint64_t content) {
    h = mix(h, tier);
    h = mix(h, where);
    h = mix(h, slot);
    h = mix(h, (uint64_t) key);
    h = mix(h, (uint64_t) (int64_t) st);
    return mix(h, content);
}

}  // namespace

Digest part_digest(Glm5Model& m) {
    Digest d;
    Glm5Model::FastState* F = m.fast_;
    if (F == nullptr || F->workers == nullptr) return d;
    const Glm5Geometry& g = m.geometry();
    int prev = 0;
    cudaGetDevice(&prev);
    cudaSetDevice(m.dev_);
    // the tables, as one snapshot (the service thread edits them under F->mu)
    std::vector<std::vector<int>> vkey((size_t) m.lt_);
    std::vector<std::vector<char>> vst((size_t) m.lt_);
    std::vector<std::vector<int>> rkey(F->rc.size());
    std::vector<std::vector<char>> rst(F->rc.size());
    {
        std::lock_guard<std::mutex> lk(F->mu);
        for (int il = m.l0_; il < m.lt_; ++il)
            if (F->L[(size_t) il].moe) {
                vkey[(size_t) il] = F->lp[(size_t) il].key;
                vst[(size_t) il] = F->lp[(size_t) il].st;
            }
        for (size_t c = 0; c < F->rc.size(); ++c) {
            rkey[c] = F->rc[c].key;
            rst[c] = F->rc[c].st;
        }
    }
    uint64_t h = mix(0x5741524D44494731ull, (uint64_t) m.part_);
    // VRAM: 8 slots a round through a host buffer, hashed in parallel
    std::vector<uint8_t> hb;
    for (int il = m.l0_; il < m.lt_ && d.ok; ++il) {
        if (!F->L[(size_t) il].moe) continue;
        const auto& P = F->lp[(size_t) il];
        const size_t blob = F->L[(size_t) il].blob;
        const std::vector<int>& key = vkey[(size_t) il];
        std::vector<uint64_t> hv(key.size(), 0);
        hb.resize(8 * blob);
        for (int j0 = 0; j0 < (int) key.size() && d.ok; j0 += 8) {
            const int nj = std::min(8, (int) key.size() - j0);
            for (int k = 0; k < nj; ++k)
                if (key[(size_t) (j0 + k)] >= 0 &&
                    cudaMemcpyAsync(hb.data() + (size_t) k * blob, P.slot_ptr(j0 + k), blob, cudaMemcpyDeviceToHost,
                                    F->copy) != cudaSuccess)
                    d.ok = false;
            if (cudaStreamSynchronize(F->copy) != cudaSuccess) d.ok = false;
            if (!d.ok) {
                std::fprintf(stderr, "glm fast: CUDA%d warm digest: a VRAM slot of layer %d did not copy (%s)\n",
                             m.dev_, il, cudaGetErrorString(cudaGetLastError()));
                break;
            }
            F->workers->run(nj, [&](int k) {
                const size_t j = (size_t) (j0 + k);
                if (key[j] >= 0) hv[j] = hash_bytes(hb.data() + (size_t) k * blob, blob);
            });
        }
        const std::vector<char>& st = vst[(size_t) il];
        for (size_t j = 0; j < key.size(); ++j) {
            h = record(h, 1, (uint64_t) il, j, key[j], j < st.size() ? st[j] : -1, hv[j]);
            if (key[j] >= 0) {
                ++d.vram;
                d.bytes += blob;
            }
        }
    }
    // the RAM tier, class by class: each held slot's expert bytes (the unified blob of a node-split slot)
    for (size_t c = 0; c < F->rc.size() && d.ok; ++c) {
        const auto& R = F->rc[c];
        const std::vector<int>& key = rkey[c];
        std::vector<uint64_t> hv(key.size(), 0);
        if (R.base != nullptr)
            F->workers->run((int) key.size(), [&](int s) {
                const int k = key[(size_t) s];
                if (k < 0) return;
                const int il = k / (int) g.n_expert;
                std::vector<uint8_t> tmp;
                const uint8_t* p = glmtier::unified(F->tier.get(), il, R.ptr(s), tmp);
                hv[(size_t) s] = hash_bytes(p, F->L[(size_t) il].blob);
            });
        const std::vector<char>& st = rst[c];
        for (size_t s = 0; s < key.size(); ++s) {
            h = record(h, 2, c, s, key[s], s < st.size() ? st[s] : -1, hv[s]);
            if (key[s] >= 0 && R.base != nullptr) {
                ++d.ram;
                d.bytes += F->L[(size_t) (key[s] / (int) g.n_expert)].blob;
            }
        }
    }
    d.sum = h;
    d.parts = 1;
    cudaSetDevice(prev);
    return d;
}

void print_digest(Glm5Model& m) {
    const Digest d = part_digest(m);
    note_digest(m.part_, d);
    std::fprintf(stderr, "glm fast: CUDA%d warm digest %016llx: %llu VRAM + %llu RAM slots, %.2f GB%s\n", m.dev_,
                 (unsigned long long) d.sum, (unsigned long long) d.vram, (unsigned long long) d.ram,
                 (double) d.bytes / 1e9, d.ok ? "" : " (INCOMPLETE: a device copy failed)");
}

}  // namespace strata::core::glmwarm
