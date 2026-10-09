// Exercise Maya's actual fused routing request/response without loading weights.
// CPU polling must see the signal and payload without a driver query or sync.
#include <hip/hip_runtime.h>
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/glm_batch.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

#define CHECK(call) do { const auto e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e)); return 2; } } while (0)

int main() {
    namespace gf = strata::kernels::glmf;
    constexpr int E = 64, N = 4096, K = 8, rounds = 100;
    hipDeviceProp_t props{};
    CHECK(hipGetDeviceProperties(&props, 0));
    std::printf("GLM handoff device: %s (%s)\n", props.name, props.gcnArchName);

    // One zeroed device arena; all buffers used by the real routing kernel.
    void* arena = nullptr;
    CHECK(hipMalloc(&arena, 256 << 10));
    CHECK(hipMemset(arena, 0, 256 << 10));
    size_t offset = 0;
    auto take = [&](size_t bytes) {
        offset = (offset + 63) & ~size_t(63);
        void* ptr = (char*) arena + offset;
        offset += bytes;
        return ptr;
    };
    gf::MoeDev d;
    d.n_keys = E;
    d.tab = (unsigned long long*) take((2 * E + gf::kSpares) * 8);
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
    d.cpu_part = (float*) take(N * 4);
    float* logits = (float*) take(E * 4);
    float* bias = (float*) take(E * 4);
    float* input = (float*) take(N * 4);
    float* output = (float*) take(N * 4);
    if (offset > (256 << 10)) return 3;

    gf::MoeRequest* ring = nullptr;
    gf::MoeResponse* response = nullptr;
    gf::CpuAnswer* answer = nullptr;
    CHECK(hipHostMalloc((void**) &ring, sizeof(*ring) * gf::kRingSize + sizeof(int), hipHostMallocMapped));
    CHECK(hipHostMalloc((void**) &response, sizeof(*response), hipHostMallocMapped));
    CHECK(hipHostMalloc((void**) &answer, sizeof(*answer), hipHostMallocMapped));
    CHECK(hipHostGetDevicePointer(&d.ring, ring, 0));
    auto* route_error = (volatile int*) (ring + gf::kRingSize);
    d.route_error = (volatile int*) ((gf::MoeRequest*) d.ring + gf::kRingSize);
    *route_error = 0;
    void* mapped = nullptr;
    CHECK(hipHostGetDevicePointer(&mapped, response, 0)); d.resp = mapped;
    CHECK(hipHostGetDevicePointer(&mapped, answer, 0)); d.cpu_ans = mapped;

    hipStream_t stream;
    CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    std::vector<float> biases(E), values(N), got(N);
    std::vector<unsigned long long> table(2 * E + gf::kSpares);
    for (int cpu = 0; cpu < 2; ++cpu) {
        std::memset(ring, 0, sizeof(*ring) * gf::kRingSize);
        std::memset(response, 0, sizeof(*response));
        std::memset(answer, 0, sizeof(*answer));
        CHECK(hipMemset(d.seq, 0, 4));
        // Nonzero RAM-tier addresses are identifiers only; this test never fetches
        // or dereferences expert blobs. All eight RAM hits go to the CPU lane.
        for (int e = 0; e < E; ++e) table[E + e] = cpu ? 0x100000000ull + e : 0;
        CHECK(hipMemcpy(d.tab, table.data(), table.size() * 8, hipMemcpyHostToDevice));
        const unsigned long long plan = cpu ? 8ull << 32 : 0;
        hipGraph_t graph;
        hipGraphExec_t exec;
        CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
        gf::moe_route(logits, bias, E, K, 1.0f, true, 0, input, N, d, nullptr, nullptr, 1.0f, 0, nullptr,
                      stream, nullptr, nullptr, 0, nullptr, nullptr, 0, K, plan);   // skip_from K: none left out
        gf::moe_wait(d, N, stream);
        gf::moe_cpu_wait(d, N, output, stream);
        CHECK(hipStreamEndCapture(stream, &graph));
        CHECK(hipGraphInstantiateWithFlags(&exec, graph, 0));
        for (int r = 1; r <= rounds; ++r) {
            std::fill(biases.begin(), biases.end(), -10.0f);
            for (int i = 0; i < K; ++i) biases[(r + i) % E] = 8.0f - i;
            for (int e = 0; e < N; ++e) values[e] = (float) (r * 10000 + e);
            CHECK(hipMemcpyAsync(bias, biases.data(), E * 4, hipMemcpyHostToDevice, stream));
            CHECK(hipMemcpyAsync(input, values.data(), N * 4, hipMemcpyHostToDevice, stream));
            CHECK(hipMemsetAsync(output, 0, N * 4, stream));
            CHECK(hipGraphLaunch(exec, stream));
            // submit without waiting: Windows' runtime holds launches until the host waits on the GPU (the engine's
            // service thread does the same when no route comes)
            (void) hipStreamQuery(stream);
            auto* request = ring + (r % gf::kRingSize);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (__atomic_load_n(&request->seq, __ATOMIC_ACQUIRE) != (unsigned int) r) {
                if (std::chrono::steady_clock::now() > deadline) {
                    std::fprintf(stderr, "GLM request visibility timeout: cpu=%d round=%d\n", cpu, r);
                    return 4;
                }
                std::this_thread::yield();
            }
            if (request->layer != 0 || request->miss_mask != (cpu ? 0u : 255u) ||
                request->cpu_mask != (cpu ? 255u : 0u)) return 5;
            for (int i = 0; i < K; ++i) {
                if (request->ids[i] != (r + i) % E || !std::isfinite(request->w[i]) ||
                    std::fabs(request->w[i] - 0.125f) > 1e-6f) return 6;
                if (cpu && request->cpu_src[i] != 0x100000000ull + (r + i) % E) return 7;
            }
            if (cpu) {
                for (int e = 0; e < N; ++e) if (request->x[e] != values[e]) return 8;
                for (int e = 0; e < N; ++e) answer->part[e] = values[e];
                __atomic_store_n(&answer->seq, (unsigned int) r, __ATOMIC_RELEASE);
            } else {
                response->cpu = 1;
                for (int e = 0; e < N; ++e) response->cpu_part[e] = values[e];
                __atomic_store_n(&response->seq, (unsigned int) r, __ATOMIC_RELEASE);
            }
            CHECK(hipStreamSynchronize(stream)); // only after publishing the host answer
            CHECK(hipMemcpy(got.data(), cpu ? output : d.cpu_part, N * 4, hipMemcpyDeviceToHost));
            for (int e = 0; e < N; ++e) if (got[e] != values[e]) return 9;
        }
        CHECK(hipGraphExecDestroy(exec));
        CHECK(hipGraphDestroy(graph));
    }
    // Non-finite primary, predicted and lookahead scores must publish an error with an empty plan.
    const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
    for (int test = 0; test < 8; ++test) {
        *route_error = 0;
        CHECK(hipMemset(d.seq, 0, 4));
        std::vector<float> scores(E, 0.0f), bad(E, 0.0f);
        if (test == 0 || test >= 6) std::fill(bad.begin(), bad.end(), nan);
        else bad[7] = test == 2 || test == 4 ? inf : nan;
        CHECK(hipMemcpy(logits, (test < 3 ? bad : scores).data(), E * 4, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(bias, bad.data(), E * 4, hipMemcpyHostToDevice));
        const float* ahead_bias[] = {nullptr};
        const bool predicted = test == 3 || test == 4 || test == 6;
        const bool ahead = test == 5 || test == 7;
        // Layer 7 has no table storage in this fixture: rejection must precede every table access.
        gf::moe_route(logits, nullptr, E, K, 1.0f, true, 7, input, N, d, nullptr, nullptr, 1.0f, 0, nullptr,
                      stream, predicted ? bias : nullptr, nullptr, 3,
                      ahead ? bias : nullptr, ahead_bias, ahead ? 1 : 0);
        gf::moe_wait(d, N, stream);
        gf::moe_cpu_wait(d, N, output, stream);
        gf::moe_fetch(d, K, 16, stream);
        CHECK(hipStreamSynchronize(stream));
        const int kind = test < 3 ? 1 : predicted ? 2 : 3;
        if (*route_error != 4 * 7 + kind || ring[1].seq != 1 || ring[1].layer != 7 || ring[1].error != kind) return 11;
        std::vector<unsigned long long> pointers(K);
        CHECK(hipMemcpy(pointers.data(), d.plan_ptr, K * 8, hipMemcpyDeviceToHost));
        for (auto ptr : pointers) if (ptr != 0) return 12;
        int n = -1;
        CHECK(hipMemcpy(&n, d.pf_n, 4, hipMemcpyDeviceToHost));
        if (n != 0) return 13;
    }
    // A failed request suppresses later routes, and clearing its error permits a fresh valid request.
    CHECK(hipMemset(logits, 0, E * 4));
    for (int reset = 0; reset < 2; ++reset) {
        if (reset) *route_error = 0;
        gf::moe_route(logits, nullptr, E, K, 1.0f, true, reset ? 0 : 7, input, N, d,
                      nullptr, nullptr, 1.0f, 0, nullptr, stream, nullptr, nullptr, 0, nullptr, nullptr, 0, 0);
        gf::moe_wait(d, N, stream);
        CHECK(hipStreamSynchronize(stream));
        if (ring[2 + reset].seq != (unsigned) (2 + reset) ||
            (reset ? ring[3].error != 0 || *route_error != 0 : ring[2].error == 0)) return 16;
    }
    // Prompt rows reject even one NaN/Inf, leaving -1 IDs that count as no routes.
    namespace gb = strata::kernels::glmb;
    float* batch_logits = (float*) take(2 * E * 4);
    int* ids = (int*) take(2 * K * 4);
    float* weights = (float*) take(2 * K * 4);
    int* counts = (int*) take(E * 4);
    int* ranks = (int*) take(2 * K * 4);
    for (int test = 0; test < 4; ++test) {
        std::vector<float> scores(2 * E, 0.0f);
        if (test == 1) std::fill(scores.begin() + E, scores.end(), nan);
        if (test >= 2) scores[E + 7] = test == 2 ? nan : inf;
        CHECK(hipMemcpy(batch_logits, scores.data(), scores.size() * 4, hipMemcpyHostToDevice));
        gb::route(batch_logits, nullptr, E, K, 1.0f, true, 2, ids, weights, stream);
        gb::expert_count(ids, 2 * K, E, counts, ranks, stream);
        CHECK(hipStreamSynchronize(stream));
        std::vector<int> hids(2 * K), hc(E);
        CHECK(hipMemcpy(hids.data(), ids, hids.size() * 4, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(hc.data(), counts, hc.size() * 4, hipMemcpyDeviceToHost));
        int total = 0;
        for (auto count : hc) total += count;
        if (total != (test == 0 ? 2 * K : K)) return 14;
        for (int i = 0; i < 2 * K; ++i)
            if (hids[i] != (test != 0 && i >= K ? -1 : i % K)) return 15;
    }
    if (gf::launch_errors()) return 10;
    CHECK(hipStreamDestroy(stream));
    CHECK(hipHostFree(answer)); CHECK(hipHostFree(response)); CHECK(hipHostFree(ring)); CHECK(hipFree(arena));
    std::puts("PASS GLM routing graph handoff: 100 disk requests + 100 CPU-lane requests; changing IDs, weights, inputs and answers");
    std::puts("PASS invalid decode/predicted/lookahead and prompt routing: NaN/Inf rejected, no unsafe fetch or lookup");
    return 0;
}
