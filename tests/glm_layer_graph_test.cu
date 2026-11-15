// Capture/direct repeated-step parity, dynamic tables, plans and request counts.
#include "../src/core/glm_layer_graphs.hpp"
#include "strata/kernels/glm_fast.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace gf = strata::kernels::glmf;
using strata::core::glmfast::LayerGraphs;

static void check(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "glm_layer_graph_test: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}
static void require(bool ok, const std::string& msg) {
    if (!ok) { std::fprintf(stderr, "glm_layer_graph_test: %s\n", msg.c_str()); std::exit(1); }
}
template<class T> static T* alloc(size_t n) {
    T* p = nullptr;
    check(cudaMalloc(&p, n * sizeof(T)));
    check(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}
__global__ void planned(const unsigned long long* table, const int* index, unsigned* counter,
                        int* output, unsigned long long plan, int layer) {
    *output = *(const int*) table[*index] + (int) plan * 17 + layer * 23 + (int) *counter;
    ++*counter;
}

int main() {
    constexpr int H = 2, HD = 128, N = H * HD, S = H * HD * HD, Q = N / 32 * 36, STEPS = 96;
    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream));
    float *q = alloc<float>(N), *k = alloc<float>(N), *v = alloc<float>(N), *g1 = alloc<float>(N);
    float *g2 = alloc<float>(N), *beta = alloc<float>(H), *norm = alloc<float>(HD);
    float *sa = alloc<float>(S), *sb = alloc<float>(S);
    unsigned char *oa = alloc<unsigned char>(Q), *ob = alloc<unsigned char>(Q);
    int *values = alloc<int>(2), *index = alloc<int>(1), *pa = alloc<int>(1), *pb = alloc<int>(1);
    unsigned *ca = alloc<unsigned>(1), *cb = alloc<unsigned>(1);
    auto* table = alloc<unsigned long long>(2);
    std::vector<float> h(N), a(S), b(S);
    std::vector<unsigned char> qa(Q), qb(Q);
    for (int i = 0; i < N; ++i) h[i] = 0.03f * std::cos((float) i);
    check(cudaMemcpy(k, h.data(), N * 4, cudaMemcpyHostToDevice));
    std::fill(h.begin(), h.end(), -0.2f);
    check(cudaMemcpy(g1, h.data(), N * 4, cudaMemcpyHostToDevice));
    std::fill(h.begin(), h.end(), 0.5f);
    check(cudaMemcpy(g2, h.data(), N * 4, cudaMemcpyHostToDevice));
    std::fill(h.begin(), h.end(), 1.0f);
    check(cudaMemcpy(norm, h.data(), HD * 4, cudaMemcpyHostToDevice));
    LayerGraphs graphs;
    uint64_t expected = 0;
    std::string err;
    for (int step = 0; step < STEPS; ++step) {
        const int layer = step % 2, selected = step % 2;
        const unsigned long long plan = (unsigned long long) ((step / 9) % 3);
        for (int i = 0; i < N; ++i) h[i] = 0.025f * std::sin((float) (i + step));
        check(cudaMemcpyAsync(q, h.data(), N * 4, cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(v, h.data(), N * 4, cudaMemcpyHostToDevice, stream));
        int vals[2] = {step + 19, 137 - step};
        unsigned long long pointers[2] = {(unsigned long long) (values + (step % 2)),
                                         (unsigned long long) (values + (1 - step % 2))};
        check(cudaMemcpyAsync(values, vals, sizeof vals, cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(index, &selected, sizeof selected, cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(table, pointers, sizeof pointers, cudaMemcpyHostToDevice, stream));
        gf::kda_rec(q, k, v, g1, beta, sa, g2, norm, 1e-5f, H, HD, oa, stream);
        planned<<<1, 1, 0, stream>>>(table, index, ca, pa, plan, layer);
        bool replayed = false;
        require(graphs.enqueue(layer, {plan, sb, nullptr}, stream, expected, 1, [&]() {
            gf::kda_rec(q, k, v, g1, beta, sb, g2, norm, 1e-5f, H, HD, ob, stream);
            planned<<<1, 1, 0, stream>>>(table, index, cb, pb, plan, layer);
            ++expected;
            return true;
        }, replayed, err), err);
        check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(a.data(), sa, S * 4, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(b.data(), sb, S * 4, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(qa.data(), oa, Q, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(qb.data(), ob, Q, cudaMemcpyDeviceToHost));
        int x = 0, y = 0;
        unsigned count = 0;
        check(cudaMemcpy(&x, pa, sizeof x, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(&y, pb, sizeof y, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(&count, cb, sizeof count, cudaMemcpyDeviceToHost));
        require(a == b && qa == qb, "KDA state/quant bytes differ at step " + std::to_string(step));
        require(x == y, "dynamic table or captured plan differs");
        require(count == expected && expected == (uint64_t) step + 1, "replayed request count differs");
    }
    // (a graph per layer and key is kept: a changed plan captures another graph and invalidates none)
    require(graphs.replays > 0 && graphs.invalidations == 0 && graphs.captures < STEPS, "capture cache not exercised");
    const uint64_t before = expected;
    bool replayed = false;
    err.clear();
    require(!graphs.enqueue(99, {}, stream, expected, 1, [&]() {
        planned<<<1, 1, 0, stream>>>(table, index, cb, pb, 0, 99);
        return true;   // deliberately missing host request accounting
    }, replayed, err), "bad capture request accounting was accepted");
    require(expected == before, "failed capture changed expected requests");
    check(cudaStreamSynchronize(stream));
    unsigned count = 0;
    check(cudaMemcpy(&count, cb, sizeof count, cudaMemcpyDeviceToHost));
    require(count == expected, "abandoned capture executed");
    std::printf("glm_layer_graph_test: PASS (%d repeated steps, %llu captures, %llu replays, %llu invalidations; dynamic tables/plans, request accounting, abandoned capture)\n",
                STEPS, (unsigned long long) graphs.captures, (unsigned long long) graphs.replays,
                (unsigned long long) graphs.invalidations);
    graphs.clear();
    for (void* p : {q, k, v, g1, g2, beta, norm, sa, sb}) check(cudaFree(p));
    for (void* p : {oa, ob}) check(cudaFree(p));
    for (void* p : {values, index, pa, pb}) check(cudaFree(p));
    check(cudaFree(ca)); check(cudaFree(cb)); check(cudaFree(table));
    check(cudaStreamDestroy(stream));
    return 0;
}
