// tests/glm_kv_group_test.cu - KV streaming on glm-dsa: the kernels behind a residency map shared by an indexer group
// (GPU, synthetic, no model).
//
// One map for kLayers caches pairs (a latent of 544 B a position and a k_pe of 128 B, page 1 - GLM-5.3's INT8 rows),
// each with its own host copy (pinned, device-mapped, filled with random bytes) and its own slot rows.  Checked:
//   1. kv_stream_reset_ident: block b in slot b, the rest evicted, counters zero;
//   2. glm_kv_resolve_set on random selections (recent-heavy, enough to evict constantly): after every call each
//      selected cell's slot row holds that position's bytes of the host copy in EVERY cache of the set, the map is a
//      bijection, nothing overflowed - and the map and the slot rows equal those of a second map driven by
//      glm_kv_resolve + glm_kv_copy_misses a cache (v1.1.67's launches) on the same selections;
//   3. glm_kv_resolve_set with the cell count on the device (n_sel_dev) reads only that many cells;
//   4. glm_kv_check: zero differing rows on the resolved selections (slot rows) and on an identity window (cells as
//      rows); one flipped byte in a slot row is counted, with its position and tag.
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
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}
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

constexpr int kLayers = 4, kCaches = 2 * kLayers, kLat = 544, kKpe = 128, kTop = 2048;
constexpr int kCtx = 24000, kSlots = 4096;

struct MapMem {
    k::KvStreamMap m;
    std::vector<int32_t> table, block;
    void make() {
        m.page_table = dalloc<int32_t>(kCtx);
        m.slot_block = dalloc<int32_t>(kSlots);
        m.slot_stamp = dalloc<int32_t>(kSlots);
        m.slot_ref = dalloc<int32_t>(kSlots);
        m.miss_block = dalloc<int32_t>(kSlots);
        m.miss_slot = dalloc<int32_t>(kSlots);
        m.ctl = dalloc<int32_t>(k::kKvCtlInts);
        m.n_blocks = kCtx;
        m.n_slots = kSlots;
    }
    void pull() {
        table.resize(kCtx);
        block.resize(kSlots);
        ck(cudaMemcpy(table.data(), m.page_table, kCtx * 4, cudaMemcpyDeviceToHost), "table");
        ck(cudaMemcpy(block.data(), m.slot_block, kSlots * 4, cudaMemcpyDeviceToHost), "block");
    }
    bool bijection() const {
        for (int b = 0; b < kCtx; ++b)
            if (table[(size_t) b] < -1 || table[(size_t) b] >= kSlots ||
                (table[(size_t) b] >= 0 && block[(size_t) table[(size_t) b]] != b))
                return false;
        for (int s = 0; s < kSlots; ++s)
            if (block[(size_t) s] >= kCtx || (block[(size_t) s] >= 0 && table[(size_t) block[(size_t) s]] != s)) return false;
        return true;
    }
};

std::vector<int> selection(std::mt19937& rng, int p) {
    std::set<int> s;
    s.insert(p);
    while ((int) s.size() < std::min(kTop, p + 1)) {
        const int back = (rng() & 3) ? (int) (rng() % (uint32_t) std::min(p + 1, 3 * kTop)) : (int) (rng() % (uint32_t) (p + 1));
        s.insert(p - back);
    }
    std::vector<int> c(s.begin(), s.end());
    c.resize(kTop, -1);
    return c;
}
}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("glm_kv_group_test: no CUDA device - SKIP\n");
        return 77;
    }
    std::printf("glm_kv_group_test: one residency map for %d caches (%d layers' latent + k_pe)\n", kCaches, kLayers);
    cudaStream_t s = nullptr;
    ck(cudaStreamCreate(&s), "stream");
    std::mt19937 rng(20261118);
    // the host copies and both sets of slots (A: the group's map; B: the per-cache launches)
    uint8_t* host_h[kCaches];
    uint8_t* host_d[kCaches];
    uint8_t* slots_a[kCaches];
    uint8_t* slots_b[kCaches];
    int len[kCaches];
    for (int j = 0; j < kCaches; ++j) {
        len[j] = j % 2 == 0 ? kLat : kKpe;
        ck(cudaHostAlloc((void**) &host_h[j], (size_t) kCtx * len[j], cudaHostAllocMapped | cudaHostAllocPortable), "host copy");
        ck(cudaHostGetDevicePointer((void**) &host_d[j], host_h[j], 0), "host copy pointer");
        uint32_t* w = (uint32_t*) host_h[j];
        for (size_t i = 0; i < (size_t) kCtx * len[j] / 4; ++i) w[i] = rng();
        slots_a[j] = dalloc<uint8_t>((size_t) kSlots * len[j]);
        slots_b[j] = dalloc<uint8_t>((size_t) kSlots * len[j]);
    }
    MapMem A, B;
    A.make();
    B.make();
    int* cells_d = dalloc<int>(kTop);
    int* sc_a = dalloc<int>(kTop);
    int* sc_b = dalloc<int>(kTop);
    int* n_dev_d = dalloc<int>(1);
    int32_t* dbg = dalloc<int32_t>(4);
    gf::KvCopySet set;
    for (int j = 0; j < kCaches; ++j) {
        set.host[j] = host_d[j];
        set.slots[j] = slots_a[j];
        set.len[j] = len[j];
    }
    set.n = kCaches;

    // 1. the identity window
    k::kv_stream_reset_ident(A.m, s);
    ck(cudaStreamSynchronize(s), "sync");
    A.pull();
    {
        bool ok = A.bijection();
        for (int b = 0; b < kCtx; ++b) ok = ok && A.table[(size_t) b] == (b < kSlots ? b : -1);
        const k::KvStreamCounters c = k::kv_stream_counters(A.m);
        check(ok && c.lookups == 0 && c.misses == 0 && !c.overflow, "reset_ident: block b in slot b, the rest evicted");
    }
    // (the slots of the window hold their positions' rows, as a conversation that grew inside it leaves them)
    for (int j = 0; j < kCaches; ++j) {
        ck(cudaMemcpyAsync(slots_a[j], host_h[j], (size_t) kSlots * len[j], cudaMemcpyHostToDevice, s), "prime");
        ck(cudaMemcpyAsync(slots_b[j], host_h[j], (size_t) kSlots * len[j], cudaMemcpyHostToDevice, s), "prime");
    }
    k::kv_stream_reset_ident(B.m, s);
    {   // 4a. the check on an identity window: the cells as rows
        const std::vector<int> c = selection(rng, kSlots - 1);
        ck(cudaMemcpyAsync(cells_d, c.data(), kTop * 4, cudaMemcpyHostToDevice, s), "cells");
        ck(cudaStreamSynchronize(s), "sync");
        for (int j = 0; j < kCaches; ++j) gf::glm_kv_check(host_d[j], slots_a[j], len[j], cells_d, nullptr, kTop, nullptr, j, dbg, s);
        int32_t d[4];
        ck(cudaStreamSynchronize(s), "sync");
        ck(cudaMemcpy(d, dbg, sizeof d, cudaMemcpyDeviceToHost), "dbg");
        check(d[0] == kCaches * kTop && d[1] == 0, "check, identity window: " + std::to_string(d[0]) + " rows, " +
                                                      std::to_string(d[1]) + " differ");
    }
    // 2. the group's resolve against the per-cache launches
    int bad_rows = 0, bad_map = 0, bad_eq = 0, calls = 0;
    std::vector<int> sca(kTop), scb(kTop);
    std::vector<uint8_t> row(kLat);
    for (int p = kSlots; p < kCtx; p += 97) {
        const std::vector<int> c = selection(rng, p);
        ck(cudaMemcpyAsync(cells_d, c.data(), kTop * 4, cudaMemcpyHostToDevice, s), "cells");
        gf::glm_kv_resolve_set(A.m, set, 1, cells_d, kTop, nullptr, sc_a, s);
        gf::glm_kv_resolve(B.m, host_d[0], slots_b[0], len[0], 1, cells_d, kTop, sc_b, s);
        for (int j = 1; j < kCaches; ++j) gf::glm_kv_copy_misses(B.m, host_d[j], slots_b[j], len[j], 1, s);
        ck(cudaStreamSynchronize(s), "sync");
        ck(cudaMemcpy(sca.data(), sc_a, kTop * 4, cudaMemcpyDeviceToHost), "slot cells");
        ck(cudaMemcpy(scb.data(), sc_b, kTop * 4, cudaMemcpyDeviceToHost), "slot cells");
        A.pull();
        B.pull();
        bad_map += !A.bijection();
        bad_eq += sca != scb || A.table != B.table;
        ++calls;
        if (calls % 8 != 1) continue;   // (the rows: every 8th call, 64 cells of it, every cache)
        for (int i = 0; i < kTop; i += kTop / 64) {
            if (c[(size_t) i] < 0) continue;
            for (int j = 0; j < kCaches; ++j) {
                if (sca[(size_t) i] < 0) {
                    ++bad_rows;
                    continue;
                }
                ck(cudaMemcpy(row.data(), slots_a[j] + (size_t) sca[(size_t) i] * len[j], (size_t) len[j], cudaMemcpyDeviceToHost), "row");
                bad_rows += std::memcmp(row.data(), host_h[j] + (size_t) c[(size_t) i] * len[j], (size_t) len[j]) != 0;
                ck(cudaMemcpy(row.data(), slots_b[j] + (size_t) scb[(size_t) i] * len[j], (size_t) len[j], cudaMemcpyDeviceToHost), "row");
                bad_eq += std::memcmp(row.data(), host_h[j] + (size_t) c[(size_t) i] * len[j], (size_t) len[j]) != 0;
            }
        }
        // 4b. the check kernel on the resolved selection, every cache
        for (int j = 0; j < kCaches; ++j) gf::glm_kv_check(host_d[j], slots_a[j], len[j], cells_d, sc_a, kTop, nullptr, j, dbg, s);
    }
    const k::KvStreamCounters ca = k::kv_stream_counters(A.m), cb = k::kv_stream_counters(B.m);
    check(bad_rows == 0 && bad_map == 0 && !ca.overflow,
          std::to_string(calls) + " group resolves: every sampled row of every cache holds its position's bytes (" +
              std::to_string(bad_rows) + " wrong), the map a bijection (" + std::to_string(bad_map) + " broken), " +
              std::to_string(ca.misses) + " of " + std::to_string(ca.lookups) + " block reads from RAM");
    check(bad_eq == 0 && ca.misses == cb.misses && ca.lookups == cb.lookups,
          "the same map, slot rows and counters as a resolve + a copy launch a cache (" + std::to_string(bad_eq) + " differ)");
    {
        int32_t d[4];
        ck(cudaStreamSynchronize(s), "sync");
        ck(cudaMemcpy(d, dbg, sizeof d, cudaMemcpyDeviceToHost), "dbg");
        check(d[0] > kCaches * kTop && d[1] == 0, "check, resolved selections: " + std::to_string(d[0]) + " rows, " +
                                                     std::to_string(d[1]) + " differ");
    }
    // 3. the device count: only the first n cells are read (the rest would name blocks never made resident)
    {
        const int p = kCtx - 1, n = 700;
        std::vector<int> c = selection(rng, p);
        for (int i = n; i < kTop; ++i) c[(size_t) i] = 5;   // (a live block number past the count)
        std::fill(sca.begin(), sca.end(), -7);
        ck(cudaMemcpyAsync(cells_d, c.data(), kTop * 4, cudaMemcpyHostToDevice, s), "cells");
        ck(cudaMemcpyAsync(sc_a, sca.data(), kTop * 4, cudaMemcpyHostToDevice, s), "slot cells");
        ck(cudaMemcpyAsync(n_dev_d, &n, 4, cudaMemcpyHostToDevice, s), "count");
        gf::glm_kv_resolve_set(A.m, set, 1, cells_d, kTop, n_dev_d, sc_a, s);
        ck(cudaStreamSynchronize(s), "sync");
        ck(cudaMemcpy(sca.data(), sc_a, kTop * 4, cudaMemcpyDeviceToHost), "slot cells");
        bool ok = true;
        for (int i = 0; i < kTop; ++i) ok = ok && (i < n ? sca[(size_t) i] >= 0 : sca[(size_t) i] == -7);
        check(ok, "the device cell count: " + std::to_string(n) + " cells resolved, the rest untouched");
        // 4c. one flipped byte in a slot row is found, with its position and tag
        const int victim = 123, cache = 3;
        uint8_t b = 0;
        uint8_t* at = slots_a[cache] + (size_t) sca[(size_t) victim] * len[cache] + 77;
        ck(cudaMemcpy(&b, at, 1, cudaMemcpyDeviceToHost), "byte");
        b ^= 0x40;
        ck(cudaMemcpy(at, &b, 1, cudaMemcpyHostToDevice), "byte");
        ck(cudaMemset(dbg, 0, 16), "dbg");
        for (int j = 0; j < kCaches; ++j) gf::glm_kv_check(host_d[j], slots_a[j], len[j], cells_d, sc_a, kTop, n_dev_d, j, dbg, s);
        int32_t d[4];
        ck(cudaStreamSynchronize(s), "sync");
        ck(cudaMemcpy(d, dbg, sizeof d, cudaMemcpyDeviceToHost), "dbg");
        check(d[0] == kCaches * n && d[1] == 1 && d[2] - 1 == c[(size_t) victim] && d[3] - 1 == cache,
              "check: a flipped byte counted once (" + std::to_string(d[1]) + " of " + std::to_string(d[0]) +
                  " rows; position " + std::to_string(d[2] - 1) + ", tag " + std::to_string(d[3] - 1) + ")");
    }
    if (g_fail) {
        std::printf("glm_kv_group_test: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("glm_kv_group_test: ok\n");
    return 0;
}
