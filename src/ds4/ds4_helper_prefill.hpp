#pragma once

#include "ds4_kernels.cuh"
#include "strata/kernels/exl3.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace strata::ds4::helper_prefill {
namespace ex = strata::kernels::exl3;

static __global__ void add_result(float* acc, const float* contribution, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += contribution[i];
}

struct Layer {
    ex::ExpertLayout layout;
    std::vector<uint8_t*> pointers;
};

struct Counts {
    uint64_t rows = 0, uses = 0;
};

// One lead and one in-flight layer. Cache placement is unchanged.
struct Workspace {
    static constexpr int kBounds = 2048;
    int helper = -1, lead = -1, C = 0, D = 0, FF = 0, K = 0;
    cudaStream_t stream = nullptr, lead_stream = nullptr;
    cudaEvent_t done = nullptr;
    bool inflight = false;
    float *x = nullptr, *y = nullptr, *acc = nullptr, *receive = nullptr, *weights = nullptr;
    uint16_t* hidden = nullptr;
    int *bounds = nullptr, *tokens = nullptr, *positions = nullptr;
    const uint8_t* pool = nullptr;
    size_t pool_bytes = 0;
    std::vector<void*> allocations;
    std::vector<ex::ExpertLayout> layouts;
    std::vector<unsigned long long*> tables;
    std::vector<int> sizes;
    Counts counts;

    // A failed dispatch can leave another rank in the reduction and the helper still accessing the lead's buffers.
    [[noreturn]] static void fatal(const char* stage, cudaError_t rc = cudaSuccess) {
        std::fprintf(stderr, "ds4: helper prefill failed at %s (%s); engine reload required\n",
                     stage, cudaGetErrorString(rc));
        std::fflush(stderr);
        std::_Exit(3);   // (ds4's unknown device state)
    }

    static void require(cudaError_t rc, const char* stage) {
        if (rc != cudaSuccess) fatal(stage, rc);
    }

    void wait() {
        if (!inflight) return;
        int previous = 0;
        require(cudaGetDevice(&previous), "wait device");
        require(cudaSetDevice(helper), "wait helper");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        for (;;) {
            const auto rc = cudaEventQuery(done);
            if (rc == cudaSuccess) break;
            if (rc != cudaErrorNotReady) fatal("completion", rc);
            if (std::chrono::steady_clock::now() >= deadline) fatal("completion timeout");
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        inflight = false;
        require(cudaSetDevice(previous), "restore wait device");
    }

    ~Workspace() {
        if (helper < 0) return;
        wait();
        int previous = 0;
        (void) cudaGetDevice(&previous);
        if (receive) {
            // The receiver may still be read by the lead's queued addition.
            (void) cudaSetDevice(lead);
            (void) cudaStreamSynchronize(lead_stream);
            (void) cudaFree(receive);
        }
        (void) cudaSetDevice(helper);
        for (void* p : allocations) (void) cudaFree(p);
        if (done) (void) cudaEventDestroy(done);
        if (stream) (void) cudaStreamDestroy(stream);
        (void) cudaSetDevice(previous);
    }

    bool open(int hd, int ld, cudaStream_t ls, int cap, int dim, int ff, int k,
              const uint8_t* base, size_t bytes, const std::vector<Layer>& layers, std::string& err) {
        helper = hd;
        lead = ld;
        lead_stream = ls;
        C = cap;
        D = dim;
        FF = ff;
        K = k;
        pool = base;
        pool_bytes = bytes;

        auto check = [&](cudaError_t rc) {
            if (rc == cudaSuccess) return true;
            err = std::string("DS4_HELPER_PREFILL: ") + cudaGetErrorString(rc);
            return false;
        };
        int previous = 0;
        if (!check(cudaGetDevice(&previous))) return false;
        struct Restore { int device; ~Restore() { (void) cudaSetDevice(device); } } restore{previous};

        for (int device : {lead, helper}) {
            int accessible = 0;
            const int peer = device == lead ? helper : lead;
            if (!check(cudaDeviceCanAccessPeer(&accessible, device, peer))) return false;
            if (!accessible) {
                err = "DS4_HELPER_PREFILL requires bidirectional peer access";
                return false;
            }
            if (!check(cudaSetDevice(device))) return false;
            const auto rc = cudaDeviceEnablePeerAccess(peer, 0);
            if (rc == cudaErrorPeerAccessAlreadyEnabled) (void) cudaGetLastError();
            else if (!check(rc)) return false;
        }

        if (!check(cudaSetDevice(helper)) || !check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)) ||
            !check(cudaEventCreateWithFlags(&done, cudaEventDisableTiming))) return false;
        auto alloc = [&](auto& p, size_t n) {
            void* v = nullptr;
            if (!check(cudaMalloc(&v, n))) return false;
            allocations.push_back(v);
            p = static_cast<std::remove_reference_t<decltype(p)>>(v);
            return true;
        };
        if (!alloc(x, (size_t) C * D * sizeof(float)) || !alloc(acc, (size_t) C * D * sizeof(float)) ||
            !alloc(hidden, (size_t) C * K * FF * sizeof(uint16_t)) || !alloc(y, (size_t) C * K * D * sizeof(float)) ||
            !alloc(weights, (size_t) C * kPicks * sizeof(float)) || !alloc(positions, (size_t) C * kPicks * sizeof(int)) ||
            !alloc(tokens, (size_t) C * K * sizeof(int)) || !alloc(bounds, kBounds * sizeof(int))) return false;

        for (const auto& l : layers) {
            layouts.push_back(l.layout);
            sizes.push_back((int) l.pointers.size());
            unsigned long long* table = nullptr;
            if (!l.pointers.empty()) {
                if (!ex::mixed_supported(l.layout)) {
                    err = "DS4_HELPER_PREFILL: unsupported indirect expert layout";
                    return false;
                }
                std::vector<unsigned long long> ptrs;
                for (auto* p : l.pointers) ptrs.push_back((unsigned long long) p);
                const size_t size = ptrs.size() * sizeof(ptrs[0]);
                if (!alloc(table, size) || !check(cudaMemcpy(table, ptrs.data(), size, cudaMemcpyHostToDevice)))
                    return false;
            }
            tables.push_back(table);
        }

        if (!check(cudaSetDevice(lead)) ||
            !check(cudaMalloc((void**) &receive, (size_t) C * D * sizeof(float)))) return false;
        std::fprintf(stderr, "ds4: helper prefill enabled: lead CUDA%d, helper CUDA%d, up to %d rows\n", lead, helper, C);
        return true;
    }

    bool submit(int il, int T, int k, int first, int nvs, const int* host_bounds,
                const float* source_x, const int* source_bounds, const int* source_tokens,
                const int* source_positions, const float* source_weights, float limit) {
        if (inflight) fatal("workspace reuse");
        const int ne = sizes[(size_t) il];
        if (ne == 0) return false;
        if (T < 1 || T > C || k < 1 || k > K || first < 0 || nvs >= kBounds || first + ne != nvs)
            fatal("routing bounds");
        const int r0 = host_bounds[first], r1 = host_bounds[nvs];
        if (r0 < 0 || r1 < r0 || r1 > T * k) fatal("routing rows");
        if (r0 == r1) return false;

        int max_rows = 0;
        uint64_t uses = 0;
        for (int i = first; i < nvs; ++i) {
            const int rows = host_bounds[i + 1] - host_bounds[i];
            if (rows < 0) fatal("nonmonotonic bounds");
            max_rows = std::max(max_rows, rows);
            uses += rows != 0;
        }
        require(cudaSetDevice(helper), "submit helper");
        inflight = true;
        const size_t activation = (size_t) T * D * sizeof(float);
        require(cudaMemcpyPeerAsync(x, helper, source_x, lead, activation, stream), "input copy");
        require(cudaMemcpyPeerAsync(bounds, helper, source_bounds, lead, (size_t) (nvs + 1) * sizeof(int), stream),
                "bounds copy");
        require(cudaMemcpyPeerAsync(tokens, helper, source_tokens, lead, (size_t) T * k * sizeof(int), stream),
                "token copy");
        require(cudaMemcpyPeerAsync(positions, helper, source_positions, lead, (size_t) T * kPicks * sizeof(int), stream),
                "position copy");
        require(cudaMemcpyPeerAsync(weights, helper, source_weights, lead, (size_t) T * kPicks * sizeof(float), stream),
                "weight copy");
        require(cudaMemsetAsync(acc, 0, activation, stream), "zero helper sum");

        const int launch_errors = ex::launch_errors();
        ex::SetArgs a;
        a.mix.n = 1;
        a.mix.lo[0] = (unsigned long long) pool;
        a.mix.hi[0] = (unsigned long long) pool + pool_bytes;
        a.mix.lay[0] = layouts[(size_t) il];
        a.eptr = tables[(size_t) il];
        a.n_exp = ne;
        a.bounds = bounds + first;
        a.row0 = r0;
        a.max_rows = max_rows;
        a.n_embd = D;
        a.n_ff = FF;
        a.limit = limit;
        ex::set_gate_up(a, x, D, tokens + r0, hidden, FF, stream);
        ex::set_down(a, hidden, FF, y, D, stream);
        moe_combine(acc, y, positions, weights, T, k, r0, r1, D, stream);
        if (ex::launch_errors() != launch_errors) fatal("expert library launch");
        require(cudaGetLastError(), "expert kernels");

        // Return on the source stream so completion covers both the kernels and the copy.
        require(cudaMemcpyPeerAsync(receive, lead, acc, helper, activation, stream), "return copy");
        require(cudaEventRecord(done, stream), "completion record");
        require(cudaSetDevice(lead), "restore lead");
        counts.rows += (uint64_t) (r1 - r0);
        counts.uses += uses;
        return true;
    }

    // The next submit follows the lead's routing synchronization, which also completes this read of receive.
    void finish(float* primary, int T) {
        wait();
        require(cudaStreamWaitEvent(lead_stream, done, 0), "receiver dependency");
        const size_t n = (size_t) T * D;
        add_result<<<(unsigned) ((n + 255) / 256), 256, 0, lead_stream>>>(primary, receive, n);
        require(cudaGetLastError(), "single lead contribution");
    }
};

struct Drain {
    Workspace* p;
    ~Drain() { if (p) p->wait(); }
};
}  // namespace strata::ds4::helper_prefill
