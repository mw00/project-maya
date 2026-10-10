// src/kernels/glm_kv_stream_parity.cpp - KV streaming of the GLM latent cache (glm_kv_stream.hpp) against a fully
// resident cache (GPU, synthetic, no model).
//
// Writes one synthetic sequence twice, with the engine's own writers: into a fully resident latent cache, and into a
// streamed layer (host copy + a VRAM slot pool small enough to evict constantly).  Positions 0..P1 one at a time
// (decode dsa_prep), P1..P2 as a prompt in sub-batches (the batched dsa_prep, which also writes the staging copy),
// then P2..N one at a time again.  Every few decode positions a selection shaped like the indexer's (up to 512 whole
// pools in score order - half of them recent, half anywhere - then the tail) is attended in both; the streamed side
// runs glm_kv_resolve first and reads the slot rows it writes.  Checks:
//   1. the attention outputs are BITWISE equal (the streamed reader sees exactly the resident values);
//   2. the residency map is consistent after every call (slot_block and page_table invert each other, no block is
//      claimed but unplaced), and every resident slot holds its block's bytes of the host copy;
//   3. after the prompt, the host copy and the staging copy equal the resident cache row for row;
//   4. no call overflowed, and the counters add up.
// FP16 rows and INT8 records.
#include "strata/kernels/glm_batch.hpp"
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
namespace gb = strata::kernels::glmb;
namespace k = strata::kernels;

namespace {
int g_fail = 0;
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
void fail(const std::string& what) {
    if (g_fail++ < 12) std::fprintf(stderr, "FAIL %s\n", what.c_str());
}
uint16_t bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) (u >> 16);
}

constexpr int kLora = 512, kQLora = 512, kKey = 128, kPage = 4, kHead = 16, kNope = 256, kVHead = 256;
constexpr int kTopPools = 512, kNSelMax = kTopPools * kPage + kPage - 1;

void run(bool q8, int N, int P1, int P2, int n_slots, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    cudaStream_t s = nullptr;
    ck(cudaStreamCreate(&s), "stream");
    const int row = q8 ? gf::lat8_rec_bytes(kLora) : kLora * 2;
    const int64_t n_blocks = (N + kPage - 1) / kPage;
    const size_t cache_bytes = (size_t) n_blocks * kPage * row;

    // the two caches: resident (identity) and streamed (host copy + slots + map), and the prompt's staging copy
    uint8_t* full = dalloc<uint8_t>(cache_bytes);
    uint8_t* stage = dalloc<uint8_t>(cache_bytes);
    uint8_t* slots = dalloc<uint8_t>((size_t) n_slots * kPage * row);
    uint8_t* host_h = nullptr;
    uint8_t* host = nullptr;
    ck(cudaHostAlloc((void**) &host_h, cache_bytes, cudaHostAllocMapped | cudaHostAllocPortable), "host copy");
    ck(cudaHostGetDevicePointer((void**) &host, host_h, 0), "host copy pointer");
    std::memset(host_h, 0, cache_bytes);
    k::KvStreamMap m;
    m.page_table = dalloc<int32_t>((size_t) n_blocks);
    m.slot_block = dalloc<int32_t>((size_t) n_slots);
    m.slot_stamp = dalloc<int32_t>((size_t) n_slots);
    m.slot_ref = dalloc<int32_t>((size_t) n_slots);
    m.miss_block = dalloc<int32_t>((size_t) n_slots);
    m.miss_slot = dalloc<int32_t>((size_t) n_slots);
    m.ctl = dalloc<int32_t>((size_t) k::kKvCtlInts);
    m.n_blocks = n_blocks;
    m.n_slots = n_slots;
    k::kv_stream_reset(m, s);

    // the writers' other inputs and outputs (shared by both sides: the same values land in them twice)
    const int sub = 512;
    float* kv_raw = dalloc<float>((size_t) sub * kLora);
    float* qr_raw = dalloc<float>(kQLora);
    float* ik_raw = dalloc<float>((size_t) sub * kKey);
    float* ig_raw = dalloc<float>((size_t) sub * kKey);
    float* ones = dalloc<float>(std::max(kLora, kQLora));
    float* zeros = dalloc<float>(kKey * kPage);
    float* qr = dalloc<float>(kQLora);
    uint8_t* qr_q = dalloc<uint8_t>((size_t) kQLora / 32 * 36);
    const int ring = 1024;
    float* ik_cache = dalloc<float>((size_t) ring * kKey);
    float* ig_cache = dalloc<float>((size_t) ring * kKey);
    float* pooled = dalloc<float>((size_t) (n_blocks + 1) * kKey);
    {
        std::vector<float> one((size_t) std::max(kLora, kQLora), 1.0f);
        ck(cudaMemcpy(ones, one.data(), one.size() * 4, cudaMemcpyHostToDevice), "ones");
    }
    // the attention's inputs
    float* q = dalloc<float>((size_t) kHead * kNope);
    uint16_t* wk_b = dalloc<uint16_t>((size_t) kHead * kLora * kNope);
    uint16_t* wv_b = dalloc<uint16_t>((size_t) kHead * kVHead * kLora);
    {
        std::vector<uint16_t> w((size_t) kHead * kLora * kNope);
        for (auto& x : w) x = bf16(nd(rng) * 0.05f);
        ck(cudaMemcpy(wk_b, w.data(), w.size() * 2, cudaMemcpyHostToDevice), "wk_b");
        for (auto& x : w) x = bf16(nd(rng) * 0.05f);
        ck(cudaMemcpy(wv_b, w.data(), w.size() * 2, cudaMemcpyHostToDevice), "wv_b");
    }
    int* cells = dalloc<int>(kNSelMax);
    int* slot_cells = dalloc<int>(kNSelMax);
    const size_t out_bytes = (size_t) kHead * kVHead / 32 * 36;
    uint8_t* out_a = dalloc<uint8_t>(out_bytes);
    uint8_t* out_b = dalloc<uint8_t>(out_bytes);

    const auto upload_rand = [&](float* d, size_t n, float scale) {
        std::vector<float> h(n);
        for (auto& x : h) x = nd(rng) * scale;
        ck(cudaMemcpyAsync(d, h.data(), n * 4, cudaMemcpyHostToDevice, s), "upload");
        ck(cudaStreamSynchronize(s), "upload sync");
    };

    // the map's invariants, and every resident slot against the host copy
    const auto check_map = [&](const std::string& at) {
        ck(cudaStreamSynchronize(s), "sync");
        std::vector<int32_t> pt((size_t) n_blocks), sb((size_t) n_slots);
        ck(cudaMemcpy(pt.data(), m.page_table, pt.size() * 4, cudaMemcpyDeviceToHost), "page table");
        ck(cudaMemcpy(sb.data(), m.slot_block, sb.size() * 4, cudaMemcpyDeviceToHost), "slot block");
        std::vector<uint8_t> sl((size_t) n_slots * kPage * row);
        ck(cudaMemcpy(sl.data(), slots, sl.size(), cudaMemcpyDeviceToHost), "slots");
        for (int64_t b = 0; b < n_blocks; ++b) {
            const int v = pt[(size_t) b];
            if (v < -1 || v >= n_slots) { fail(at + ": page_table[" + std::to_string(b) + "] = " + std::to_string(v)); continue; }
            if (v >= 0 && sb[(size_t) v] != b) fail(at + ": slot " + std::to_string(v) + " does not hold block " + std::to_string(b));
            if (v >= 0 && std::memcmp(sl.data() + (size_t) v * kPage * row, host_h + (size_t) b * kPage * row,
                                      (size_t) kPage * row) != 0)
                fail(at + ": slot " + std::to_string(v) + " differs from block " + std::to_string(b) + "'s host copy");
        }
        for (int j = 0; j < n_slots; ++j)
            if (sb[(size_t) j] >= 0 && pt[(size_t) sb[(size_t) j]] != j)
                fail(at + ": slot " + std::to_string(j) + " names block " + std::to_string(sb[(size_t) j]) + " whose entry differs");
    };

    // one decode position p through both writers
    const auto decode_write = [&](int p) {
        upload_rand(kv_raw, kLora, 1.0f);
        upload_rand(qr_raw, kQLora, 1.0f);
        upload_rand(ik_raw, kKey, 1.0f);
        upload_rand(ig_raw, kKey, 1.0f);
        gf::DsaPrepArgs d;
        d.qr_raw = qr_raw; d.q_a_norm = ones; d.qr = qr; d.qr_q = qr_q; d.q_lora = kQLora;
        d.kv_raw = kv_raw; d.kv_norm = ones; d.kv_lora = kLora; d.lat_q8 = q8;
        d.ik_raw = ik_raw; d.k_norm_w = ones; d.k_norm_b = zeros; d.ik_cache = ik_cache;
        d.ig_raw = ig_raw; d.ig_cache = ig_cache; d.ape = zeros; d.pooled = pooled;
        d.idx_key = kKey; d.kpool = kPage; d.ring = ring; d.p = p;
        d.lat = (uint16_t*) full;
        gf::dsa_prep(d, s);
        d.lat = nullptr;
        d.lat_host = (uint16_t*) host;
        d.lat_slots = (uint16_t*) slots;
        d.lat_table = m.page_table;
        d.lat_page = kPage;
        gf::dsa_prep(d, s);
    };

    // a selection at position p shaped like the indexer's, attended both ways
    int compares = 0;
    const auto attend = [&](int p) {
        const int n_vis = (p + 1) / kPage;
        const int top = std::min(kTopPools, n_vis);
        std::vector<int> pools;
        std::set<int> seen;
        std::uniform_int_distribution<int> any(0, std::max(0, n_vis - 1));
        std::uniform_int_distribution<int> recent(std::max(0, n_vis - 700), std::max(0, n_vis - 1));
        while ((int) pools.size() < top) {
            const int pi = (pools.size() % 2 == 0) ? recent(rng) : any(rng);
            if (seen.insert(pi).second) pools.push_back(pi);
        }
        const int n_sel = top * kPage + kPage - 1;
        std::vector<int> c((size_t) n_sel, -1);
        for (int i = 0; i < top; ++i)
            for (int j = 0; j < kPage; ++j) c[(size_t) i * kPage + j] = pools[(size_t) i] * kPage + j;
        for (int j = 0; j < kPage - 1; ++j)
            if (n_vis * kPage + j <= p) c[(size_t) top * kPage + j] = n_vis * kPage + j;
        ck(cudaMemcpyAsync(cells, c.data(), c.size() * 4, cudaMemcpyHostToDevice, s), "cells");
        upload_rand(q, (size_t) kHead * kNope, 1.0f);
        gf::mla(q, wk_b, wv_b, (const uint16_t*) full, cells, n_sel, kHead, kNope, kLora, kVHead, out_a, s, q8);
        gf::glm_kv_resolve(m, host, slots, row, kPage, cells, n_sel, slot_cells, s);
        gf::mla(q, wk_b, wv_b, (const uint16_t*) slots, slot_cells, n_sel, kHead, kNope, kLora, kVHead, out_b, s, q8);
        ck(cudaStreamSynchronize(s), "attend");
        std::vector<uint8_t> a(out_bytes), b(out_bytes);
        ck(cudaMemcpy(a.data(), out_a, out_bytes, cudaMemcpyDeviceToHost), "out a");
        ck(cudaMemcpy(b.data(), out_b, out_bytes, cudaMemcpyDeviceToHost), "out b");
        if (a != b) fail(std::string(q8 ? "int8" : "fp16") + ": attention differs at position " + std::to_string(p));
        ++compares;
        if (compares % 16 == 0) check_map(std::string(q8 ? "int8" : "fp16") + " at position " + std::to_string(p));
    };

    for (int p = 0; p < P1; ++p) {
        decode_write(p);
        if (p % 7 == 3) attend(p);
    }
    // the prompt: the batched writer, in sub-batches, into the resident cache and into staging + host copy + slots
    for (int p0 = P1; p0 < P2; p0 += sub) {
        const int tn = std::min(sub, P2 - p0);
        upload_rand(kv_raw, (size_t) tn * kLora, 1.0f);
        upload_rand(ik_raw, (size_t) tn * kKey, 1.0f);
        upload_rand(ig_raw, (size_t) tn * kKey, 1.0f);
        gb::DsaPrepArgs d;
        d.kv_raw = kv_raw; d.kv_norm = ones; d.kv_lora = kLora; d.lat_q8 = q8;
        d.ik_raw = ik_raw; d.k_norm_w = ones; d.k_norm_b = zeros; d.ik_cache = ik_cache;
        d.ig_raw = ig_raw; d.ig_cache = ig_cache; d.idx_key = kKey; d.ring = ring; d.p0 = p0; d.T = tn;
        d.lat = (uint16_t*) full;
        gb::dsa_prep(d, s);
        d.lat = (uint16_t*) stage;
        d.lat_host = (uint16_t*) host;
        d.lat_slots = (uint16_t*) slots;
        d.lat_table = m.page_table;
        d.lat_page = kPage;
        gb::dsa_prep(d, s);
    }
    ck(cudaStreamSynchronize(s), "prompt");
    {
        std::vector<uint8_t> f(cache_bytes), st(cache_bytes);
        ck(cudaMemcpy(f.data(), full, cache_bytes, cudaMemcpyDeviceToHost), "full");
        ck(cudaMemcpy(st.data(), stage, cache_bytes, cudaMemcpyDeviceToHost), "stage");
        for (int p = 0; p < P2; ++p) {
            if (std::memcmp(host_h + (size_t) p * row, f.data() + (size_t) p * row, (size_t) row) != 0)
                fail(std::string(q8 ? "int8" : "fp16") + ": host copy row " + std::to_string(p) + " differs");
            if (p >= P1 && std::memcmp(st.data() + (size_t) p * row, f.data() + (size_t) p * row, (size_t) row) != 0)
                fail(std::string(q8 ? "int8" : "fp16") + ": staging row " + std::to_string(p) + " differs");
        }
    }
    check_map(std::string(q8 ? "int8" : "fp16") + " after the prompt");
    for (int p = P2; p < N; ++p) {
        decode_write(p);
        if (p % 5 == 1) attend(p);
    }
    check_map(std::string(q8 ? "int8" : "fp16") + " at the end");
    const k::KvStreamCounters c = k::kv_stream_counters(m);
    if (c.overflow) fail(std::string(q8 ? "int8" : "fp16") + ": a resolve overflowed");
    if ((int) c.calls != compares) fail("calls counted " + std::to_string(c.calls) + ", made " + std::to_string(compares));
    if (c.misses > c.lookups || c.lookups == 0) fail("counters: misses > lookups or no lookups");
    std::printf("glm_kv_stream_parity %s: %d positions, %d compares, %llu of %llu block reads hit VRAM (%d slots)\n",
                q8 ? "int8" : "fp16", N, compares, (unsigned long long) (c.lookups - c.misses),
                (unsigned long long) c.lookups, n_slots);
    cudaFreeHost(host_h);
    cudaStreamDestroy(s);
}
}  // namespace

int main(int argc, char** argv) {
    const bool selftest = argc > 1 && std::string(argv[1]) == "--selftest";
    if (!selftest) {
        std::fprintf(stderr, "glm_kv_stream_parity: run with --selftest (the CTest form)\n");
        return 2;
    }
    // 1,100 slots (the fewest a resolve allows is 1,024) against selections of up to 513 blocks: constant evictions
    run(false, 6002, 2001, 4003, 1100, 7u);
    run(true, 6002, 2001, 4003, 1100, 8u);
    if (g_fail) {
        std::printf("glm_kv_stream_parity: FAIL (%d)\n", g_fail);
        return 1;
    }
    std::printf("glm_kv_stream_parity: PASS\n");
    return 0;
}
