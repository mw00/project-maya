// tests/glm_dsa/glm_state_slot_gpu_test.cpp - glm-dsa B4.2: a state arena's slot round trip on the device (GPU).
//
// For a glm-dsa layout (GLM-5.3's per-layer shape over a short trunk, FP16 and INT8) and a Flash one, the test fills
// a device arena laid out by the descriptor with a pattern, saves the slot runs of the first n positions to the host
// the way Glm5Model::slot_save does (chunked device-to-host copies, runs in file order), overwrites the arena, loads
// the runs back the way slot_load does and checks every byte the slot covers is restored and nothing else is touched.
// A real-model save/restore (next 32 greedy tokens identical) is B6.4's PACK fixture.  Exit 77 without a device.
#include "strata/core/glm_model.hpp"
#include "strata/core/glm_state_layout.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using strata::core::GlmArch;
using strata::core::GlmIdxKind;
using strata::core::GlmStateLayout;
using strata::core::GlmStateMode;
using strata::core::Glm5Geometry;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

Glm5Geometry small_dsa() {
    Glm5Geometry g;
    g.arch = GlmArch::GlmDsa;
    g.n_embd = 6144;
    g.n_layers = 10;
    g.n_head_kv.assign(10, 1);
    g.kv_lora = 512;
    g.qk_nope = 192;
    g.qk_rope = 64;
    g.idx_key = 128;
    g.idx_kpool = 1;
    g.idx_select_tail = 0;
    g.idx_kind = GlmIdxKind::Token;
    g.hc = 1;
    g.indexer_full.assign(11, 0);
    for (int il = 0; il < 11; ++il) g.indexer_full[(size_t) il] = il < 2 || (il - 2) % 4 == 0 || il == 10;
    return g;
}

Glm5Geometry small_flash() {
    Glm5Geometry g;
    g.n_layers = 6;
    g.n_head = 4;   // d_inner 512: a small KDA state
    g.n_head_kv = {0, 0, 0, 1, 0, 1};
    g.indexer_full.assign(6, 1);
    return g;
}

// fills, saves n positions, scrambles, restores, compares
bool round_trip(const GlmStateLayout& L, int64_t n, const std::string& what) {
    float* d = nullptr;
    if (cudaMalloc(&d, L.bytes) != cudaSuccess) {
        check(false, what + ": cudaMalloc");
        return false;
    }
    std::vector<uint32_t> a(L.bytes / 4), b(a.size()), back(a.size());
    for (size_t i = 0; i < a.size(); ++i) a[i] = (uint32_t) (i * 2654435761u) ^ 0x5a5a5a5au;
    for (size_t i = 0; i < b.size(); ++i) b[i] = ~a[i];
    cudaMemcpy(d, a.data(), L.bytes, cudaMemcpyHostToDevice);
    // save (slot_save: chunks of 16M floats)
    const auto runs = L.slot_runs(n);
    std::vector<float> file, buf;
    for (const auto& r : runs) {
        constexpr int64_t kChunk = (int64_t) 16 << 20;
        for (int64_t at = 0; at < r.second; at += kChunk) {
            const int64_t c = std::min(kChunk, r.second - at);
            buf.resize((size_t) c);
            cudaMemcpy(buf.data(), d + r.first + at, (size_t) c * 4, cudaMemcpyDeviceToHost);
            file.insert(file.end(), buf.begin(), buf.end());
        }
    }
    // another conversation overwrites the arena, then the slot comes back (slot_load)
    cudaMemcpy(d, b.data(), L.bytes, cudaMemcpyHostToDevice);
    size_t at = 0;
    std::vector<uint8_t> covered(a.size(), 0);
    for (const auto& r : runs) {
        cudaMemcpy(d + r.first, file.data() + at, (size_t) r.second * 4, cudaMemcpyHostToDevice);
        at += (size_t) r.second;
        for (int64_t i = 0; i < r.second; ++i) covered[(size_t) (r.first + i)] = 1;
    }
    const bool copied = cudaMemcpy(back.data(), d, L.bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(d);
    bool ok = copied;
    size_t n_cov = 0;
    for (size_t i = 0; i < a.size() && ok; ++i) {
        ok = back[i] == (covered[i] ? a[i] : b[i]);
        n_cov += covered[i];
    }
    // the slot covers exactly n positions of each cache the layout says it keeps
    uint64_t expect = 0;
    for (int il : L.dsa_layers()) {
        expect += (uint64_t) n * L.lat_pp + (uint64_t) (L.has_kpe(il) ? n * L.kpe_pp : 0);
        if (L.glm_dsa() && L.has_index_cache(il)) expect += (uint64_t) n * L.ik_pp;
        if (!L.glm_dsa()) expect += (uint64_t) ((n + L.idx_kpool - 1) / L.idx_kpool) * L.idx_key;
    }
    check(ok && n_cov == expect && at == file.size(),
          what + ": " + std::to_string(runs.size()) + " runs, " + std::to_string(file.size() * 4) + " bytes round-trip");
    return ok;
}

}  // namespace

int main() {
    int devs = 0;
    if (cudaGetDeviceCount(&devs) != cudaSuccess || devs == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    std::string err;
    const int64_t ctx = 4096;
    {
        const Glm5Geometry g = small_dsa();
        for (GlmStateMode m : {GlmStateMode::Fp16, GlmStateMode::Int8}) {
            GlmStateLayout L;
            const bool ok = strata::core::glm_state_carve(g, 0, 10, 10, ctx, m, 0, L, err);
            check(ok && L.snap_pool_floats() == 0, "glm-dsa carve, pool-less snapshot");
            for (int64_t n : {(int64_t) 1, (int64_t) 2049, ctx - 1})
                round_trip(L, n, std::string("glm-dsa ") + (m == GlmStateMode::Int8 ? "INT8" : "FP16") + " n " +
                                     std::to_string(n));
        }
    }
    {
        const Glm5Geometry g = small_flash();
        GlmStateLayout L;
        const bool ok = strata::core::glm_state_carve(g, 0, 6, 6, ctx, GlmStateMode::Fp16,
                                                      strata::core::glm_state_fast_ik_ring(ctx), L, err);
        check(ok, "Flash carve");
        for (int64_t n : {(int64_t) 1, (int64_t) 1023, ctx - 1}) round_trip(L, n, "Flash FP16 n " + std::to_string(n));
    }
    std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
