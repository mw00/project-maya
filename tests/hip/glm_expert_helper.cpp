// Helper ownership/replacement and weighted sums against the primary GPU.
#include "strata/core/remote_experts.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(call) do { const auto e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e)); return 2; } } while (0)

static int helper_routing() {
    namespace gf = strata::kernels::glmf;
    constexpr int NE = 64, N = 4096, K = 8, keys = 2 * NE;
    void* arena = nullptr;
    CHECK(hipMalloc(&arena, 256 << 10));
    CHECK(hipMemset(arena, 0, 256 << 10));
    size_t at = 0;
    const auto take = [&](size_t bytes) {
        at = (at + 63) & ~size_t(63);
        void* p = (uint8_t*) arena + at; at += bytes; return p;
    };
    gf::MoeDev d;
    d.n_keys = keys;
    d.tab = (unsigned long long*) take((2 * keys + 2 * gf::kSpares) * 8);
    d.helper_owned = (uint8_t*) take(keys);
    d.scratch = (unsigned long long*) take(K * 8);
    d.plan_ptr = (unsigned long long*) take(K * 8);
    d.plan_w = (float*) take(K * 4);
    d.plan_id = (int*) take(K * 4);
    d.fetch_src = (unsigned long long*) take(K * 8);
    d.pf_src = (unsigned long long*) take(K * 8);
    d.pf_dst = (unsigned long long*) take(K * 8);
    d.pf_n = (int*) take(4);
    d.seq = (unsigned int*) take(4);
    d.wait_seq = (unsigned int*) take(4);
    d.cpu_seq = (unsigned int*) take(4);
    d.cpu_flag = (int*) take(4);
    auto* logits = (float*) take(NE * 4);
    auto* x = (float*) take(N * 4);
    gf::MoeRequest* ring = nullptr;
    CHECK(hipHostMalloc((void**) &ring, sizeof(*ring) * gf::kRingSize, hipHostMallocMapped));
    CHECK(hipHostGetDevicePointer(&d.ring, ring, 0));
    CHECK(hipMemset((void*) d.helper_owned, 1, keys));
    std::vector<unsigned long long> table(2 * keys + 2 * gf::kSpares);
    // Both RAM hits and available primary spares must lose to helper ownership.
    for (int i = keys; i < (int) table.size(); ++i) table[(size_t) i] = 0x100000000ull + i * 256;
    for (unsigned long long plan : {0ull, 8ull << 32}) {
        CHECK(hipMemset(d.seq, 0, 4));
        CHECK(hipMemcpy(d.tab, table.data(), table.size() * 8, hipMemcpyHostToDevice));
        gf::moe_route(logits, nullptr, NE, K, 1.0f, true, 0, x, N, d, nullptr, nullptr, 1.0f, 0, nullptr,
                      nullptr, logits, nullptr, 2, nullptr, nullptr, 0, K, plan, 0, K);
        CHECK(hipDeviceSynchronize());
        if (ring[1].seq != 1 || ring[1].cpu_mask != 255u || ring[1].miss_mask || ring[1].fetch_mask ||
            ring[1].promo_mask || ring[1].pf_n) return 16;
        for (int i = 0; i < K; ++i) if (ring[1].cpu_src[i] != 0) return 17;
        std::vector<unsigned long long> after(table.size());
        CHECK(hipMemcpy(after.data(), d.tab, after.size() * 8, hipMemcpyDeviceToHost));
        if (after != table) return 18;
    }
    CHECK(hipHostFree(ring)); CHECK(hipFree(arena));
    std::puts("GLM helper ownership precedes PCIe quota, admission and prefetch");
    return 0;
}

int main() {
    namespace gf = strata::kernels::glmf;
    constexpr int E = 4096, FF = 2048, K = 8;
    int devices = 0, helper_device = -1;
    CHECK(hipGetDeviceCount(&devices));
    if (devices > 1) helper_device = 1;
    if (helper_device < 0) {
        std::puts("SKIP: requires a second GPU");
        return 77;
    }
    CHECK(hipSetDevice(0));
    if (int status = helper_routing()) return status;
    std::mt19937 rng(47);
    std::vector<float> x(E), reference(E), actual(E);
    float weights[K];
    for (int i = 0; i < E; ++i) x[i] = float(int(rng() % 101) - 50) * 0.01f;
    for (int i = 0; i < K; ++i) weights[i] = float(i + 1) / 36.0f;
    std::string err;
    const auto block_bytes = [](int type) -> size_t {
        switch (type) {
            case 10: return sizeof(block_q2_K);
            case 11: return sizeof(block_q3_K);
            case 16: return sizeof(block_iq2_xxs);
            case 18: return sizeof(block_iq3_xxs);
            default: return sizeof(block_iq2_s);
        }
    };
    for (const auto [gu_type, d_type] : {std::pair{16, 22}, std::pair{18, 22}, std::pair{16, 18}, std::pair{10, 11}}) {
        const size_t gu_block = block_bytes(gu_type), d_block = block_bytes(d_type);
        const size_t gu_bytes = size_t(2) * FF * E / 256 * gu_block;
        const size_t down_bytes = size_t(E) * FF / 256 * d_block;
        const size_t blob_bytes = gu_bytes + down_bytes;
        uint8_t* host = nullptr;
        CHECK(hipSetDevice(0));
        CHECK(hipHostMalloc((void**) &host, K * blob_bytes, hipHostMallocMapped | hipHostMallocPortable));
        for (int i = 0; i < K; ++i) {
            for (int role = 0; role < 2; ++role) {
                const size_t block = role ? d_block : gu_block;
                const size_t bytes = role ? down_bytes : gu_bytes;
                uint8_t* p = host + size_t(i) * blob_bytes + (role ? gu_bytes : 0);
                for (size_t b = 0; b < bytes; b += block) {
                    for (size_t j = 0; j < block; ++j) p[b + j] = rng() & 255;
                    const uint16_t scale = std::bit_cast<uint16_t>(__float2half(0.0001f * float(1 + rng() % 4)));
                    const int type = role ? d_type : gu_type;
                    const size_t scale_at = type == 10 ? block - 4 : type == 11 ? block - 2 : 0;
                    std::memcpy(p + b + scale_at, &scale, 2);
                    if (type == 10) std::memcpy(p + b + block - 2, &scale, 2);   // finite Q2_K dmin too
                }
            }
        }
        strata::core::GlmExpertHelper helper;
        if (!helper.open(helper_device, E, FF, {{host, K * blob_bytes}}, err)) {
            std::fprintf(stderr, "%s\n", err.c_str()); return 1;
        }
        // Seven residents plus an unowned candidate. No duplicate admission;
        // bytes, not expert count, cap the cache on a discrete helper.
        if (!helper.reserve(32, (K - 1) * blob_bytes, err)) return 8;
        for (int i = 0; i < K - 1; ++i)
            if (!helper.insert(i, host + size_t(i) * blob_bytes, blob_bytes, blob_bytes, err)) return 9;
        if (helper.available() != 0 || helper.insert(K - 1, host, blob_bytes, blob_bytes, err)) return 10;
        std::vector<uint32_t> heat{1, 4, 5, 6, 7, 8, 9, 20};
        heat.resize(32);
        heat[31] = 100;   // hotter, but in another layer: must not steal this layer's slot
        std::vector<int> primary(32, -1);
        auto swaps = helper.swaps(heat, primary, 16, 1);
        if (swaps.size() != 1 || swaps[0].out != 0 || swaps[0].in != 7) return 11;
        primary[7] = 0;   // also represents a pending primary admission
        if (!helper.swaps(heat, primary, 16, 8).empty()) return 12;
        primary[7] = -1; primary[8] = 0;   // helper victim hotter than primary's coldest (zero)
        if (!helper.swaps(heat, primary, 16, 8).empty()) return 13;
        primary[8] = -1; heat[7] = 2;   // not 1.5 hotter than helper victim
        if (!helper.swaps(heat, primary, 16, 8).empty()) return 14;
        // A shared-RAM candidate changes ownership without destroying the old blob.
        heat[7] = 20;
        const std::vector<uint8_t> old_blob(host, host + blob_bytes);
        if (!helper.replace(0, 7, host + 7 * blob_bytes, err) || helper.owns(0) || !helper.owns(7)) return 15;
        if (helper.integrated() && std::memcmp(host, old_blob.data(), blob_bytes) != 0) return 19;
        int current = -1;
        CHECK(hipGetDevice(&current));
        if (current != 0) return 3;   // helper calls must restore the caller's device
        // Reference: all experts on the primary GPU, with the same GLM kernels.
        void *dev_blobs = nullptr, *dxq = nullptr, *dhq = nullptr;
        float *dx = nullptr, *dout = nullptr;
        gf::MoeDev plan;
        CHECK(hipMalloc(&dev_blobs, K * blob_bytes));
        CHECK(hipMemcpy(dev_blobs, host, K * blob_bytes, hipMemcpyHostToDevice));
        CHECK(hipMalloc((void**) &dx, E * 4));
        CHECK(hipMalloc((void**) &dout, E * 4));
        CHECK(hipMalloc(&dxq, E / 32 * sizeof(block_q8_1)));
        CHECK(hipMalloc(&dhq, K * FF / 32 * sizeof(block_q8_1)));
        CHECK(hipMalloc((void**) &plan.plan_ptr, K * 8));
        CHECK(hipMalloc((void**) &plan.plan_w, K * 4));
        CHECK(hipMalloc((void**) &plan.cpu_flag, 4));
        CHECK(hipMemset(plan.cpu_flag, 0, 4));
        CHECK(hipMemcpy(dx, x.data(), E * 4, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(plan.plan_w, weights, K * 4, hipMemcpyHostToDevice));
        int keys[K];
        unsigned long long pointers[K];
        // Repeated pointers and noncontiguous ordering must preserve the weights.
        for (int i = 0; i < K; ++i) {
            const int e = (i * 3) % 5 == 0 ? 7 : (i * 3) % 5;
            keys[i] = e;
            pointers[i] = (unsigned long long) ((uint8_t*) dev_blobs + size_t(e) * blob_bytes);
        }
        CHECK(hipMemcpy(plan.plan_ptr, pointers, K * 8, hipMemcpyHostToDevice));
        gf::quantize_q8_1(dx, dxq, E, nullptr);
        for (int ne : {1, 3, 8}) {
            gf::moe_gate_up(gu_type, plan, ne, E, FF, 8.0f, dxq, dhq, nullptr, 0, nullptr, 0, nullptr, nullptr);
            gf::moe_down(d_type, plan, ne, E, FF, gu_bytes, dhq, nullptr, dout, nullptr);
            CHECK(hipMemcpy(reference.data(), dout, E * 4, hipMemcpyDeviceToHost));
            if (!helper.compute(gu_type, d_type, gu_bytes, 8.0f, ne, keys, weights, x.data(), actual.data(), err)) {
                std::fprintf(stderr, "%s\n", err.c_str()); return 1;
            }
            double diff2 = 0, ref2 = 0, max_abs = 0;
            for (int i = 0; i < E; ++i) {
                if (!std::isfinite(actual[i]) || !std::isfinite(reference[i])) return 4;
                const double diff = double(actual[i]) - reference[i];
                diff2 += diff * diff; ref2 += double(reference[i]) * reference[i];
                max_abs = std::max(max_abs, std::abs(diff));
            }
            const double rel = std::sqrt(diff2 / std::max(ref2, 1e-30));
            std::printf("GLM %s helper: types %d/%d, experts %d, max_abs %.3g rel_L2 %.3g\n",
                        helper.integrated() ? "UMA" : "discrete",
                        gu_type, d_type, ne, max_abs, rel);
            if (max_abs > 1e-4 || rel > 1e-6) return 5;
            CHECK(hipGetDevice(&current));
            if (current != 0) return 6;
        }
        // An unpinned staging source models a disk read: now replace the bytes
        // in the helper's slot and compare against expert 1 on the primary.
        const std::vector<uint8_t> staging(host + blob_bytes, host + 2 * blob_bytes);
        if (!helper.replace(7, 0, staging.data(), err)) return 20;
        pointers[0] = (unsigned long long) ((uint8_t*) dev_blobs + blob_bytes);
        CHECK(hipMemcpy(plan.plan_ptr, pointers, 8, hipMemcpyHostToDevice));
        gf::moe_gate_up(gu_type, plan, 1, E, FF, 8.0f, dxq, dhq, nullptr, 0, nullptr, 0, nullptr, nullptr);
        gf::moe_down(d_type, plan, 1, E, FF, gu_bytes, dhq, nullptr, dout, nullptr);
        CHECK(hipMemcpy(reference.data(), dout, E * 4, hipMemcpyDeviceToHost));
        keys[0] = 0;
        if (!helper.compute(gu_type, d_type, gu_bytes, 8.0f, 1, keys, weights, x.data(), actual.data(), err)) return 21;
        for (int i = 0; i < E; ++i)
            if (!std::isfinite(actual[i]) || std::fabs(actual[i] - reference[i]) > 1e-4f) return 22;
        const int evicted = 7;
        if (helper.compute(gu_type, d_type, gu_bytes, 8.0f, 1, &evicted, weights, x.data(), actual.data(), err)) return 7;
        helper.close();
        CHECK(hipFree(plan.cpu_flag)); CHECK(hipFree(plan.plan_w)); CHECK(hipFree(plan.plan_ptr));
        CHECK(hipFree(dhq)); CHECK(hipFree(dxq)); CHECK(hipFree(dout)); CHECK(hipFree(dx)); CHECK(hipFree(dev_blobs));
        CHECK(hipHostFree(host));
    }
    std::puts("GLM expert helper ownership, adaptation and parity passed");
}
