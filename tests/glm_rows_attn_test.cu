// tests/glm_rows_attn_test.cu - the batched rows' DSA attention and selection (glm_rows.cu, STRATA_GLM_ROWS_ATTN) against
// the per-row path the rows take without it, on random inputs (needs an NVIDIA GPU of compute capability 8.0+, no model), compared
// bitwise:
//   per row (the reference): gf::mla(q_i, ...) over that row's own latent cache - resident, or a streamed layer's slot
//     rows after glm_kv_resolve - into its q8_1 output;
//   batched: gf::mla_q_abs for every row at once, each row's selection resolved into its own slot-cell buffer when its
//     cache is streamed, gf::mla_attend_rows over all the rows (each its own cache and cells), gf::mla_out for every
//     row at once.
// 2..16 rows, each its own conversation with its own context length (from a few positions, where the selection is
// every pool, to ~60K), resident and streamed caches mixed, some rows later positions of the row before's conversation
// (a verify window's rows; streamed: one resolve over the run's cells, as the engine), FP16 rows and INT8 records, and launches of fewer tokens than rows (part_tokens).  And the
// rows' selections in one launch (gf::dsa_select_rows) against dsa_select a row, cell for cell; and STRATA_GLM_ROWS_SMALL's
// pieces: the rows' scores (dsa_score_rows), KV resolves (glm_kv_resolve_rows: each selected cell reads its position's
// bytes, as glm_kv_resolve's) and the experts' grouping with a thread an entry, against the one-a-row / one-thread
// kernels.
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/glm_kv_stream.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace gf = strata::kernels::glmf;
namespace k = strata::kernels;

namespace {

int g_fail = 0, g_checks = 0;

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}
template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}
uint16_t bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) (u >> 16);
}
uint16_t f16(float f) {   // round to nearest of the FP16 bits (normal range only, as the test's values are)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    int e = (int) ((u >> 23) & 0xff) - 127 + 15;
    uint32_t m = u & 0x7fffffu;
    if (e <= 0) return (uint16_t) sign;
    if (e >= 31) return (uint16_t) (sign | 0x7bffu);
    uint32_t h = sign | ((uint32_t) e << 10) | (m >> 13);
    if ((m & 0x1fffu) > 0x1000u || ((m & 0x1fffu) == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) h;
}

// the model's geometry (GLM-5.x: 64 heads, qk_nope 256, kv_lora 512, v_head 256; 512 pools of 4 cells selected)
constexpr int kHead = 64, kNope = 256, kLora = 512, kVHead = 256, kPage = 4, kTopPools = 512;
constexpr int kNSelMax = kTopPools * kPage + kPage - 1;
// a streamed layer's VRAM pages: 36,864 positions, so long contexts evict, and a run of 16 rows' selections (one
// resolve, up to 16 x 515 blocks) still fits, as a run of a verify window's rows fits the engine's (>= 5,120 pages)
constexpr int kSlots = 9216;

struct Seq {
    int ctx = 0;
    bool streamed = false;
    uint8_t* full = nullptr;     // resident: the identity cache
    uint8_t* host_h = nullptr;   // streamed: the host copy (mapped), its device pointer, the slots and the map
    uint8_t* host = nullptr;
    uint8_t* slots = nullptr;
    k::KvStreamMap m;
};

void free_seq(Seq& q) {
    if (q.full) cudaFree(q.full);
    if (q.host_h) cudaFreeHost(q.host_h);
    if (q.slots) cudaFree(q.slots);
    for (int32_t* p : {q.m.page_table, q.m.slot_block, q.m.slot_stamp, q.m.slot_ref, q.m.miss_block, q.m.miss_slot,
                       q.m.ctl})
        if (p) cudaFree(p);
}

void run(bool q8, int n, int part_tokens, double p_streamed, double p_dup, uint32_t seed, cudaStream_t s,
         const uint16_t* wk_b, const uint16_t* wv_b) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    const int row = q8 ? gf::lat8_rec_bytes(kLora) : kLora * 2;
    // the rows' conversations: mostly one each, now and then the row before's (a resident one) again
    std::vector<Seq> seqs;
    std::vector<int> row_seq((size_t) n), row_pos((size_t) n);
    for (int i = 0; i < n; ++i) {
        if (i > 0 && u01(rng) < p_dup) {
            row_seq[(size_t) i] = row_seq[(size_t) i - 1];
            continue;
        }
        Seq q;
        const double r = u01(rng);
        q.ctx = r < 0.15 ? 8 + (int) (u01(rng) * 400) : r < 0.3 ? 400 + (int) (u01(rng) * 3000)
                                                               : 2000 + (int) (u01(rng) * 58000);
        q.streamed = u01(rng) < p_streamed;
        const int64_t n_blocks = (q.ctx + kPage - 1) / kPage;
        const size_t bytes = (size_t) n_blocks * kPage * row;
        // the latents: FP16 rows, or INT8 records (kLora codes, then an FP16 scale per 32 values)
        std::vector<uint8_t> h(bytes);
        for (int64_t p = 0; p < n_blocks * kPage; ++p) {
            uint8_t* rp = h.data() + (size_t) p * row;
            if (!q8) {
                for (int c = 0; c < kLora; ++c) {
                    const uint16_t v = f16(nd(rng) * 0.5f);
                    std::memcpy(rp + 2 * c, &v, 2);
                }
            } else {
                for (int c = 0; c < kLora; ++c) rp[c] = (uint8_t) (int8_t) std::clamp((int) std::lround(nd(rng) * 40.0f), -127, 127);
                for (int b = 0; b < kLora / 32; ++b) {
                    const uint16_t sc = f16(0.004f + 0.01f * (float) u01(rng));
                    std::memcpy(rp + kLora + 2 * b, &sc, 2);
                }
            }
        }
        if (!q.streamed) {
            q.full = dalloc<uint8_t>(bytes);
            ck(cudaMemcpy(q.full, h.data(), bytes, cudaMemcpyHostToDevice), "cache");
        } else {
            ck(cudaHostAlloc((void**) &q.host_h, bytes, cudaHostAllocMapped | cudaHostAllocPortable), "host copy");
            ck(cudaHostGetDevicePointer((void**) &q.host, q.host_h, 0), "host copy pointer");
            std::memcpy(q.host_h, h.data(), bytes);
            q.slots = dalloc<uint8_t>((size_t) kSlots * kPage * row);
            q.m.page_table = dalloc<int32_t>((size_t) n_blocks);
            q.m.slot_block = dalloc<int32_t>(kSlots);
            q.m.slot_stamp = dalloc<int32_t>(kSlots);
            q.m.slot_ref = dalloc<int32_t>(kSlots);
            q.m.miss_block = dalloc<int32_t>(kSlots);
            q.m.miss_slot = dalloc<int32_t>(kSlots);
            q.m.ctl = dalloc<int32_t>(k::kKvCtlInts);
            q.m.n_blocks = n_blocks;
            q.m.n_slots = kSlots;
            k::kv_stream_reset(q.m, s);
        }
        row_seq[(size_t) i] = (int) seqs.size();
        seqs.push_back(q);
    }
    // each row's position (a conversation's later rows: the positions after its first one's, as in a window)
    for (int i = 0; i < n; ++i) {
        const Seq& q = seqs[(size_t) row_seq[(size_t) i]];
        int before = 0;
        for (int j = 0; j < i; ++j) before += row_seq[(size_t) j] == row_seq[(size_t) i];
        row_pos[(size_t) i] = std::max(0, q.ctx - 1 - 3 + before);
        row_pos[(size_t) i] = std::min(row_pos[(size_t) i], q.ctx - 1);
    }
    // each row's selection shaped like the indexer's: up to 512 whole pools (half recent, half anywhere), the tail
    // (one list of n rows of kNSelMax cells, -1 past each row's own count, as the engine's S->acells: a run of one
    // conversation's rows on a streamed layer is resolved as one list)
    int* cells_all = dalloc<int>((size_t) n * kNSelMax);
    int* slot_ref_all = dalloc<int>((size_t) n * kNSelMax);
    int* slot_bat_all = dalloc<int>((size_t) n * kNSelMax);
    ck(cudaMemset(cells_all, 0xff, (size_t) n * kNSelMax * sizeof(int)), "padding");
    std::vector<int*> cells((size_t) n), slot_ref((size_t) n), slot_bat((size_t) n);
    std::vector<int> n_sel((size_t) n);
    for (int i = 0; i < n; ++i) {
        const int p = row_pos[(size_t) i];
        const int n_vis = (p + 1) / kPage;
        const int top = std::min(kTopPools, n_vis);
        std::vector<int> pools;
        std::set<int> seen;
        std::uniform_int_distribution<int> any(0, std::max(0, n_vis - 1));
        std::uniform_int_distribution<int> recent(std::max(0, n_vis - 700), std::max(0, n_vis - 1));
        if (top == n_vis) {
            for (int pi = 0; pi < n_vis; ++pi) pools.push_back(pi);
        } else {
            while ((int) pools.size() < top) {
                const int pi = (pools.size() % 2 == 0) ? recent(rng) : any(rng);
                if (seen.insert(pi).second) pools.push_back(pi);
            }
        }
        const int ns = top * kPage + kPage - 1;
        std::vector<int> c((size_t) ns, -1);
        for (int a = 0; a < top; ++a)
            for (int j = 0; j < kPage; ++j) c[(size_t) a * kPage + j] = pools[(size_t) a] * kPage + j;
        for (int j = 0; j < kPage - 1; ++j)
            if (n_vis * kPage + j <= p) c[(size_t) top * kPage + j] = n_vis * kPage + j;
        cells[(size_t) i] = cells_all + (size_t) i * kNSelMax;
        slot_ref[(size_t) i] = slot_ref_all + (size_t) i * kNSelMax;
        slot_bat[(size_t) i] = slot_bat_all + (size_t) i * kNSelMax;
        ck(cudaMemcpy(cells[(size_t) i], c.data(), c.size() * 4, cudaMemcpyHostToDevice), "cells");
        n_sel[(size_t) i] = ns;
    }
    // the rows' queries, one after the other (as the rows' S->q)
    const int64_t qd = (int64_t) kHead * kNope, ad = (int64_t) kHead * kLora;
    const size_t out_row = (size_t) kHead * kVHead / 32 * 36;
    float* q = dalloc<float>((size_t) n * qd);
    {
        std::vector<float> h((size_t) n * qd);
        for (auto& x : h) x = nd(rng);
        ck(cudaMemcpy(q, h.data(), h.size() * 4, cudaMemcpyHostToDevice), "q");
    }
    uint8_t* out_ref = dalloc<uint8_t>((size_t) n * out_row);
    uint8_t* out_bat = dalloc<uint8_t>((size_t) n * out_row);
    ck(cudaMemset(out_bat, 0xA5, (size_t) n * out_row), "poison");
    float* aq = dalloc<float>((size_t) n * ad);
    float* actx = dalloc<float>((size_t) n * ad);
    float* part = dalloc<float>(gf::mla_rows_part_floats(kHead, kLora) * (size_t) part_tokens);

    // the reference: each row through mla() alone (a streamed row resolves its selection first)
    for (int i = 0; i < n; ++i) {
        Seq& sq = seqs[(size_t) row_seq[(size_t) i]];
        const uint16_t* lat = (const uint16_t*) (sq.streamed ? sq.slots : sq.full);
        const int* c = cells[(size_t) i];
        if (sq.streamed) {
            gf::glm_kv_resolve(sq.m, sq.host, sq.slots, row, kPage, c, n_sel[(size_t) i], slot_ref[(size_t) i], s);
            c = slot_ref[(size_t) i];
        }
        gf::mla(q + (size_t) i * qd, wk_b, wv_b, lat, c, n_sel[(size_t) i], kHead, kNope, kLora, kVHead,
                out_ref + (size_t) i * out_row, s, q8);
    }
    // batched: q_abs of every row, every row's cells (resolved into its own buffer when streamed), one attention
    gf::mla_q_abs(q, qd, wk_b, kHead, kNope, kLora, n, aq, ad, s);
    std::vector<const uint16_t*> lats((size_t) n);
    std::vector<const int*> cl((size_t) n);
    for (int i = 0; i < n;) {   // (a run of one conversation's rows: one resolve over their cells, as the engine)
        int len = 1;
        while (i + len < n && row_seq[(size_t) (i + len)] == row_seq[(size_t) i]) ++len;
        Seq& sq = seqs[(size_t) row_seq[(size_t) i]];
        if (sq.streamed)
            gf::glm_kv_resolve(sq.m, sq.host, sq.slots, row, kPage, cells[(size_t) i],
                               (len - 1) * kNSelMax + n_sel[(size_t) (i + len - 1)], slot_bat[(size_t) i], s);
        for (int k2 = 0; k2 < len; ++k2) {
            lats[(size_t) (i + k2)] = (const uint16_t*) (sq.streamed ? sq.slots : sq.full);
            cl[(size_t) (i + k2)] = sq.streamed ? slot_bat[(size_t) (i + k2)] : cells[(size_t) (i + k2)];
        }
        i += len;
    }
    if (!gf::mla_attend_rows(aq, ad, lats.data(), cl.data(), n_sel.data(), n, kHead, kNope, kLora, actx, ad, part,
                             part_tokens, s, q8)) {
        std::fprintf(stderr, "FAIL: mla_attend_rows is not available (mla_rows_ok said it was)\n");
        ++g_fail;
        return;
    }
    gf::mla_out(actx, ad, wv_b, kHead, kLora, kVHead, n, out_bat, (int64_t) out_row, s);
    ck(cudaStreamSynchronize(s), "run");
    ck(cudaGetLastError(), "kernels");

    std::vector<uint8_t> a((size_t) n * out_row), b((size_t) n * out_row);
    ck(cudaMemcpy(a.data(), out_ref, a.size(), cudaMemcpyDeviceToHost), "out ref");
    ck(cudaMemcpy(b.data(), out_bat, b.size(), cudaMemcpyDeviceToHost), "out bat");
    int streamed_rows = 0, dup_rows = 0, streamed_dup = 0;
    for (int i = 0; i < n; ++i) {
        streamed_rows += seqs[(size_t) row_seq[(size_t) i]].streamed;
        dup_rows += i > 0 && row_seq[(size_t) i] == row_seq[(size_t) i - 1];
        streamed_dup += i > 0 && row_seq[(size_t) i] == row_seq[(size_t) i - 1] && seqs[(size_t) row_seq[(size_t) i]].streamed;
        ++g_checks;
        if (std::memcmp(a.data() + (size_t) i * out_row, b.data() + (size_t) i * out_row, out_row) != 0) {
            size_t at = 0;
            while (a[(size_t) i * out_row + at] == b[(size_t) i * out_row + at]) ++at;
            std::fprintf(stderr, "FAIL: %s, %d rows (%d a launch), row %d (context %d, %d cells, %s): differs from "
                                 "mla() at byte %zu of %zu\n",
                         q8 ? "int8" : "fp16", n, part_tokens, i, seqs[(size_t) row_seq[(size_t) i]].ctx,
                         n_sel[(size_t) i], seqs[(size_t) row_seq[(size_t) i]].streamed ? "streamed" : "resident", at,
                         out_row);
            ++g_fail;
        }
    }
    std::printf("  %s %2d rows, %d a launch: %d streamed, %d window rows (%d streamed), cells %d..%d - %s\n",
                q8 ? "int8" : "fp16", n, part_tokens, streamed_rows, dup_rows, streamed_dup, *std::min_element(n_sel.begin(), n_sel.end()),
                *std::max_element(n_sel.begin(), n_sel.end()), g_fail ? "(failures so far)" : "same");
    for (void* p : {(void*) cells_all, (void*) slot_ref_all, (void*) slot_bat_all, (void*) q, (void*) out_ref, (void*) out_bat, (void*) aq, (void*) actx, (void*) part}) cudaFree(p);
    for (Seq& sq : seqs) free_seq(sq);
}

// the rows' selection in one launch (gf::dsa_select_rows) against dsa_select a row: n rows of random scores (ties
// included), each its own pool count (none, fewer than the top pools, many) and position
void run_select(int n, uint32_t seed, cudaStream_t s) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    gf::DsaSelectRows a;
    std::vector<float*> score((size_t) n);
    std::vector<int*> ref((size_t) n), bat((size_t) n);
    for (int i = 0; i < n; ++i) {
        const double r = u01(rng);
        const int n_vis = r < 0.1 ? 0 : r < 0.25 ? 1 + (int) (u01(rng) * 600) : 600 + (int) (u01(rng) * 30000);
        const int top = std::min(kTopPools, n_vis);
        const int pos = n_vis * kPage + (int) (u01(rng) * kPage);
        std::vector<float> h((size_t) std::max(1, n_vis));
        for (auto& x : h) x = u01(rng) < 0.05 ? 0.5f : (float) u01(rng) * 8.0f - 4.0f;   // (some ties)
        score[(size_t) i] = dalloc<float>(h.size());
        ck(cudaMemcpy(score[(size_t) i], h.data(), h.size() * 4, cudaMemcpyHostToDevice), "score");
        ref[(size_t) i] = dalloc<int>(kNSelMax);
        bat[(size_t) i] = dalloc<int>(kNSelMax);
        const int n_sel = top * kPage + kPage - 1;
        gf::dsa_select(score[(size_t) i], n_vis, kPage, top, n_sel, pos, ref[(size_t) i], s);
        a.score[i] = score[(size_t) i];
        a.cells[i] = bat[(size_t) i];
        a.n_vis[i] = n_vis;
        a.top_pools[i] = top;
        a.n_sel[i] = n_sel;
        a.pos[i] = pos;
    }
    gf::dsa_select_rows(a, n, kPage, s);
    ck(cudaStreamSynchronize(s), "select");
    for (int i = 0; i < n; ++i) {
        const std::vector<int> x = [&] {
            std::vector<int> v(kNSelMax);
            ck(cudaMemcpy(v.data(), ref[(size_t) i], v.size() * 4, cudaMemcpyDeviceToHost), "ref");
            return v;
        }();
        std::vector<int> y(kNSelMax);
        ck(cudaMemcpy(y.data(), bat[(size_t) i], y.size() * 4, cudaMemcpyDeviceToHost), "bat");
        ++g_checks;
        if (std::memcmp(x.data(), y.data(), (size_t) a.n_sel[i] * 4) != 0) {
            std::fprintf(stderr, "FAIL: selection, %d rows: row %d (%d pools) differs from dsa_select's\n", n, i,
                         a.n_vis[i]);
            ++g_fail;
        }
        cudaFree(score[(size_t) i]);
        cudaFree(ref[(size_t) i]);
        cudaFree(bat[(size_t) i]);
    }
}

// STRATA_GLM_ROWS_SMALL: the rows' scores in one launch (gf::dsa_score_rows) against dsa_score a row
void run_score(int n, uint32_t seed, cudaStream_t s) {
    constexpr int kKey = 128, kIdxHeads = 32;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    gf::DsaScoreRows a;
    std::vector<float*> ref((size_t) n), bufs;
    for (int i = 0; i < n; ++i) {
        const double r = u01(rng);
        const int np = r < 0.2 ? 1 + (int) (u01(rng) * 20) : r < 0.4 ? 20 + (int) (u01(rng) * 2000)
                                                              : 2000 + (int) (u01(rng) * 30000);
        const auto up = [&](size_t cnt, float sc) {
            std::vector<float> h(cnt);
            for (auto& x : h) x = nd(rng) * sc;
            float* d = dalloc<float>(cnt);
            ck(cudaMemcpy(d, h.data(), cnt * 4, cudaMemcpyHostToDevice), "up");
            bufs.push_back(d);
            return d;
        };
        a.iq[i] = up((size_t) kIdxHeads * kKey, 1.0f);
        a.pooled[i] = up((size_t) np * kKey, 1.0f);
        a.iw[i] = up(kIdxHeads, 0.2f);
        a.score[i] = dalloc<float>((size_t) np);
        bufs.push_back(a.score[i]);
        ref[(size_t) i] = dalloc<float>((size_t) np);
        bufs.push_back(ref[(size_t) i]);
        a.n_pools[i] = np;
        gf::dsa_score(a.iq[i], a.pooled[i], a.iw[i], kKey, kIdxHeads, np, ref[(size_t) i], s);
    }
    gf::dsa_score_rows(a, n, kKey, kIdxHeads, s);
    ck(cudaStreamSynchronize(s), "score");
    for (int i = 0; i < n; ++i) {
        std::vector<float> x((size_t) a.n_pools[i]), y((size_t) a.n_pools[i]);
        ck(cudaMemcpy(x.data(), ref[(size_t) i], x.size() * 4, cudaMemcpyDeviceToHost), "ref");
        ck(cudaMemcpy(y.data(), a.score[i], y.size() * 4, cudaMemcpyDeviceToHost), "bat");
        ++g_checks;
        if (std::memcmp(x.data(), y.data(), x.size() * 4) != 0) {
            std::fprintf(stderr, "FAIL: scores, %d rows: row %d (%d pools) differs from dsa_score's\n", n, i, a.n_pools[i]);
            ++g_fail;
        }
    }
    for (float* b : bufs) cudaFree(b);
}

// STRATA_GLM_ROWS_SMALL: n sequences' resolves in one launch (gf::glm_kv_resolve_rows) against glm_kv_resolve a row, on
// twin copies of every sequence's map and slots, over several steps (so slots get evicted).  A resolve places its misses
// in the order its threads claim them (atomics), so which slot a block lands in - and from there the clock's later
// victims - differs from run to run with either kernel; what the attention reads may not: for every selected cell, the
// slot row named in its slot cell holds that position's host bytes, padding stays padding, and no call overflows (the
// streaming parity test's rule, glm_kv_stream_parity) - checked for both twins
void run_resolve(int n, bool q8, uint32_t seed, cudaStream_t s) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    const int row = q8 ? gf::lat8_rec_bytes(kLora) : kLora * 2;
    constexpr int kSl = 1024;   // slots: small, so the steps evict
    struct Twin {
        int ctx = 0;
        int64_t n_blocks = 0;
        uint8_t *host_h = nullptr, *host = nullptr, *slots[2] = {nullptr, nullptr};
        k::KvStreamMap m[2];
        int *cells = nullptr, *sc[2] = {nullptr, nullptr};
    };
    std::vector<Twin> tw((size_t) n);
    for (Twin& t : tw) {
        t.ctx = 4000 + (int) (u01(rng) * 60000);
        t.n_blocks = (t.ctx + kPage - 1) / kPage;
        const size_t bytes = (size_t) t.n_blocks * kPage * row;
        ck(cudaHostAlloc((void**) &t.host_h, bytes, cudaHostAllocMapped | cudaHostAllocPortable), "host");
        ck(cudaHostGetDevicePointer((void**) &t.host, t.host_h, 0), "host ptr");
        for (size_t b = 0; b < bytes; ++b) t.host_h[b] = (uint8_t) (rng() & 0xff);
        for (int c = 0; c < 2; ++c) {
            t.slots[c] = dalloc<uint8_t>((size_t) kSl * kPage * row);
            k::KvStreamMap& m = t.m[c];
            m.page_table = dalloc<int32_t>((size_t) t.n_blocks);
            m.slot_block = dalloc<int32_t>(kSl);
            m.slot_stamp = dalloc<int32_t>(kSl);
            m.slot_ref = dalloc<int32_t>(kSl);
            m.miss_block = dalloc<int32_t>(kSl);
            m.miss_slot = dalloc<int32_t>(kSl);
            m.ctl = dalloc<int32_t>(k::kKvCtlInts);
            m.n_blocks = t.n_blocks;
            m.n_slots = kSl;
            k::kv_stream_reset(m, s);
            t.sc[c] = dalloc<int>(kNSelMax);
        }
        t.cells = dalloc<int>(kNSelMax);
    }
    for (int step = 0; step < 6; ++step) {
        gf::KvResolveRows a;
        std::vector<std::vector<int>> sel((size_t) n);
        std::vector<int> ns((size_t) n);
        for (int i = 0; i < n; ++i) {
            Twin& t = tw[(size_t) i];
            const int n_vis = t.ctx / kPage;
            const int top = std::min(kTopPools, n_vis);
            std::set<int> seen;
            std::vector<int>& c = sel[(size_t) i];
            std::uniform_int_distribution<int> any(0, n_vis - 1);
            while ((int) seen.size() < top) {
                const int pi = any(rng);
                if (seen.insert(pi).second)
                    for (int j = 0; j < kPage; ++j) c.push_back(pi * kPage + j);
            }
            if (step % 3 == 2) c.resize(c.size() / 3);   // (and a short one now and then)
            c.push_back(-1);                             // (and padding)
            ns[(size_t) i] = step == 5 && i == 0 ? 0 : (int) c.size();   // (a row that resolves nothing)
            ck(cudaMemcpy(t.cells, c.data(), c.size() * 4, cudaMemcpyHostToDevice), "cells");
            if (ns[(size_t) i] > 0)
                gf::glm_kv_resolve(t.m[0], t.host, t.slots[0], row, kPage, t.cells, ns[(size_t) i], t.sc[0], s);
            a.m[i] = t.m[1];
            a.host[i] = t.host;
            a.slots[i] = t.slots[1];
            a.cells[i] = t.cells;
            a.slot_cells[i] = t.sc[1];
            a.n_sel[i] = ns[(size_t) i];
        }
        gf::glm_kv_resolve_rows(a, n, row, kPage, s);
        ck(cudaStreamSynchronize(s), "resolve");
        for (int i = 0; i < n; ++i) {
            Twin& t = tw[(size_t) i];
            ++g_checks;
            for (int c = 0; c < 2; ++c) {
                const char* who = c == 0 ? "glm_kv_resolve" : "glm_kv_resolve_rows";
                int ctl[k::kKvCtlInts];
                ck(cudaMemcpy(ctl, t.m[c].ctl, sizeof ctl, cudaMemcpyDeviceToHost), "ctl");
                if (ctl[3] != 0) {
                    std::fprintf(stderr, "FAIL: %s, %d rows, step %d, row %d: overflowed\n", who, n, step, i);
                    ++g_fail;
                }
                if (ns[(size_t) i] <= 0) continue;
                std::vector<int> sc((size_t) ns[(size_t) i]);
                ck(cudaMemcpy(sc.data(), t.sc[c], sc.size() * 4, cudaMemcpyDeviceToHost), "slot cells");
                std::vector<uint8_t> sl((size_t) kSl * kPage * row);
                ck(cudaMemcpy(sl.data(), t.slots[c], sl.size(), cudaMemcpyDeviceToHost), "slots");
                int bad = 0;
                for (int j = 0; j < ns[(size_t) i]; ++j) {
                    const int cell = sel[(size_t) i][(size_t) j], slc = sc[(size_t) j];
                    if (cell < 0 || slc < 0) {
                        bad += (cell < 0) != (slc < 0);
                        continue;
                    }
                    bad += slc >= kSl * kPage ||
                           std::memcmp(sl.data() + (size_t) slc * row, t.host_h + (size_t) cell * row, (size_t) row) != 0;
                }
                if (bad) {
                    std::fprintf(stderr, "FAIL: %s, %d rows, step %d, row %d: %d of %d cells do not read their "
                                         "position's bytes\n", who, n, step, i, bad, ns[(size_t) i]);
                    ++g_fail;
                }
            }
        }
    }
    for (Twin& t : tw) {
        cudaFreeHost(t.host_h);
        cudaFree(t.cells);
        for (int c = 0; c < 2; ++c) {
            cudaFree(t.slots[c]);
            cudaFree(t.sc[c]);
            for (int32_t* p : {t.m[c].page_table, t.m[c].slot_block, t.m[c].slot_stamp, t.m[c].slot_ref,
                               t.m[c].miss_block, t.m[c].miss_slot, t.m[c].ctl})
                cudaFree(p);
        }
    }
}

// STRATA_GLM_ROWS_SMALL: the experts' grouping with a thread an entry against the one-thread kernel, value for value
void run_group(int T, uint32_t seed, cudaStream_t s) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pick(0, 3 * T + 4);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    const int ne = T * 8;
    std::vector<unsigned long long> plan((size_t) ne);
    for (auto& p : plan) p = u01(rng) < 0.1 ? 0ull : 0x10000ull + 0x1000ull * (unsigned long long) pick(rng);
    unsigned long long* d_plan = dalloc<unsigned long long>((size_t) ne);
    ck(cudaMemcpy(d_plan, plan.data(), plan.size() * 8, cudaMemcpyHostToDevice), "plan");
    gf::MoeWin w[2];
    for (int c = 0; c < 2; ++c) {
        w[c].plan_ptr = d_plan;
        w[c].grp_ptr = dalloc<unsigned long long>(128);
        w[c].grp_start = dalloc<int>(129);
        w[c].grp_ent = dalloc<int>(128);
        w[c].n_groups = dalloc<int>(1);
        w[c].T = T;
        w[c].k = 8;
        gf::moe_win_group(w[c], s, c == 1);
    }
    ck(cudaStreamSynchronize(s), "group");
    int ng[2];
    for (int c = 0; c < 2; ++c) ck(cudaMemcpy(&ng[c], w[c].n_groups, 4, cudaMemcpyDeviceToHost), "ng");
    ++g_checks;
    const auto same_bytes = [&](const void* x, const void* y, size_t bytes) {
        std::vector<uint8_t> hx(bytes), hy(bytes);
        ck(cudaMemcpy(hx.data(), x, bytes, cudaMemcpyDeviceToHost), "x");
        ck(cudaMemcpy(hy.data(), y, bytes, cudaMemcpyDeviceToHost), "y");
        return hx == hy;
    };
    int tot = 0;
    if (ng[0] == ng[1]) ck(cudaMemcpy(&tot, w[0].grp_start + ng[0], 4, cudaMemcpyDeviceToHost), "tot");
    if (ng[0] != ng[1] || !same_bytes(w[0].grp_ptr, w[1].grp_ptr, (size_t) ng[0] * 8) ||
        !same_bytes(w[0].grp_start, w[1].grp_start, (size_t) (ng[0] + 1) * 4) ||
        !same_bytes(w[0].grp_ent, w[1].grp_ent, (size_t) tot * 4)) {
        std::fprintf(stderr, "FAIL: grouping, %d tokens: %d / %d groups, arrays differ\n", T, ng[0], ng[1]);
        ++g_fail;
    }
    cudaFree(d_plan);
    for (int c = 0; c < 2; ++c) {
        cudaFree(w[c].grp_ptr);
        cudaFree(w[c].grp_start);
        cudaFree(w[c].grp_ent);
        cudaFree(w[c].n_groups);
    }
}

}  // namespace

int main() {
    int dev = 0;
    ck(cudaGetDevice(&dev), "device");
    cudaDeviceProp pr{};
    ck(cudaGetDeviceProperties(&pr, dev), "properties");
    if (!gf::mla_rows_ok(kLora)) {
        std::printf("glm_rows_attn_test: SKIP - no tensor-core attention on %s (sm_%d%d)\n", pr.name, pr.major, pr.minor);
        return 0;
    }
    cudaStream_t s = nullptr;
    ck(cudaStreamCreate(&s), "stream");
    std::mt19937 rng(777);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    uint16_t* wk_b = dalloc<uint16_t>((size_t) kHead * kLora * kNope);
    uint16_t* wv_b = dalloc<uint16_t>((size_t) kHead * kVHead * kLora);
    {
        std::vector<uint16_t> w((size_t) kHead * kLora * kNope);
        for (auto& x : w) x = bf16(nd(rng) * 0.05f);
        ck(cudaMemcpy(wk_b, w.data(), w.size() * 2, cudaMemcpyHostToDevice), "wk_b");
        for (auto& x : w) x = bf16(nd(rng) * 0.05f);
        ck(cudaMemcpy(wv_b, w.data(), w.size() * 2, cudaMemcpyHostToDevice), "wv_b");
    }
    std::printf("glm_rows_attn_test on %s: the rows' attention in one launch against mla() a row, bitwise\n", pr.name);
    uint32_t seed = 1;
    for (const bool q8 : {false, true})
        for (const int n : {2, 3, 4, 5, 7, 8, 9, 12, 16})
            for (const int pt : {std::min(n, gf::kMlaTokMax), 3}) {
                if (pt == 3 && n <= 3) continue;
                run(q8, n, pt, 0.5, 0.25, seed++, s, wk_b, wv_b);
            }
    // all resident / all streamed
    for (const bool q8 : {false, true}) {
        run(q8, 8, 8, 0.0, 0.0, seed++, s, wk_b, wv_b);
        run(q8, 8, 8, 1.0, 0.0, seed++, s, wk_b, wv_b);
        run(q8, 16, 8, 0.0, 0.6, seed++, s, wk_b, wv_b);
        run(q8, 8, 8, 1.0, 0.6, seed++, s, wk_b, wv_b);    // verify windows on streamed layers
        run(q8, 16, 8, 1.0, 0.5, seed++, s, wk_b, wv_b);
    }
    int f0 = 0;   // (the attention runs above came first)
    const auto verdict = [&](const char* what) {
        std::printf("  %s: %s\n", what, g_fail == f0 ? "same" : "FAILED");
        f0 = g_fail;
    };
    verdict("the attention of 2..16 rows in one launch");
    for (const int n : {1, 2, 4, 7, 8, 16}) run_select(n, 900 + (uint32_t) n, s);
    verdict("selections of 1..16 rows in one launch");
    for (const int n : {1, 2, 4, 7, 16}) run_score(n, 1200 + (uint32_t) n, s);
    verdict("scores of 1..16 rows in one launch");
    for (const bool q8 : {false, true})
        for (const int n : {1, 3, 8, 16}) run_resolve(n, q8, 1300 + (uint32_t) n, s);
    verdict("KV resolves of 1..16 rows in one launch (6 steps, evicting: every cell reads its position's bytes)");
    for (int T = 1; T <= 16; ++T)
        for (int r = 0; r < 4; ++r) run_group(T, 1400 + (uint32_t) (T * 4 + r), s);
    verdict("experts' grouping, a thread an entry, 1..16 tokens");
    cudaFree(wk_b);
    cudaFree(wv_b);
    if (g_fail) {
        std::printf("glm_rows_attn_test: FAILED (%d failed checks, %d cases)\n", g_fail, g_checks);
        return 1;
    }
    std::printf("glm_rows_attn_test: OK (%d rows, each mla()'s bytes)\n", g_checks);
    return 0;
}
