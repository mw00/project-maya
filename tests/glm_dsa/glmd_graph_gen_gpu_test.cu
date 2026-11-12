// tests/glm_dsa/glmd_graph_gen_gpu_test.cu - glm-dsa B15.6 on a V100: a decode-graph cache over an arena that is
// resized mid-session.
//
// glmdsa::DsaGraphs (the per-part cache the glm-dsa decode uses) replays a captured body that adds 1 to every float
// of an arena.  Mid-session the arena is freed and allocated again at twice the size (a context change), the
// allocation bumps the graph generation as glm_dsa_fast.cu / glm_fast_path.cu do, and the key takes the new
// generation: the next enqueue must drop the old graph and capture against the new arena (one invalidation), and the
// new arena must see exactly the increments issued after the resize.  The same check runs with the new arena forced
// onto the old base address (the case the address fields of the key cannot see): there the generation alone forces
// the re-capture.
// Needs a GPU; exits 77 (ctest SKIP_RETURN_CODE) when there is none.
#include "glm_dsa_graphs.hpp"
#include "strata/kernels/glmd_graph.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_fail = 0;
void expect(bool ok, const char* what) {
    std::printf("%-100s %s\n", what, ok ? "ok" : "FAIL");
    g_fail += !ok;
}

__global__ void add_one(float* a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += 1.0f;
}

glmd::GraphKey key_for(const void* arena) {
    glmd::GraphKey k;
    k.part = 0;
    k.l0 = 0;
    k.l1 = 26;
    k.arena = arena;
    k.n_cap = 4096;
    k.gen = glmd::graph_generation();
    return k;
}

bool all_equal(const float* d, int n, float v) {
    std::vector<float> h((size_t) n);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), d, (size_t) n * 4, cudaMemcpyDeviceToHost);
    for (float x : h)
        if (x != v) return false;
    return true;
}

void run(bool same_base) {
    using strata::core::glmdsa::DsaGraphs;
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    int n = 1 << 16;
    // a pool of two blocks so the "same base" case can hand the arena's old address back deterministically
    float* pool = nullptr;
    cudaMalloc(&pool, (size_t) 4 * n * 4);
    float* arena = pool;
    // zeroed on the stream that adds (s is non-blocking: nothing orders a legacy-stream memset before its work, and
    // Windows' HIP runtime submits legacy-stream work only when the host waits - #133)
    cudaMemsetAsync(arena, 0, (size_t) n * 4, s);
    glmd::graph_generation_bump("test arena");
    DsaGraphs G;
    uint64_t expected = 0;
    std::string err;
    for (int i = 0; i < 3; ++i) {
        bool replayed = false;
        float* a = arena;
        const int na = n;
        G.enqueue(key_for(arena), s, expected, 0, [&] {
            add_one<<<(na + 255) / 256, 256, 0, s>>>(a, na);
            return true;
        }, replayed, err);
    }
    char what0[200];
    std::snprintf(what0, sizeof what0, "%s: captured once, replayed twice (%llu %llu%s%s)",
                  same_base ? "same base" : "moved", (unsigned long long) G.captures, (unsigned long long) G.replays,
                  err.empty() ? "" : ", ", err.c_str());
    expect(G.captures == 1 && G.replays == 2 && all_equal(arena, n, 3.0f), what0);
    // the resize: free + allocate again (here: carve again from the pool) at twice the size
    float* fresh = same_base ? pool : pool + 2 * n;
    n *= 2;
    cudaMemsetAsync(fresh, 0, (size_t) n * 4, s);
    arena = fresh;
    glmd::graph_generation_bump("test arena (resized)");   // what every arena allocation in the engine does
    for (int i = 0; i < 2; ++i) {
        bool replayed = false;
        float* a = arena;
        const int na = n;
        G.enqueue(key_for(arena), s, expected, 0, [&] {
            add_one<<<(na + 255) / 256, 256, 0, s>>>(a, na);
            return true;
        }, replayed, err);
    }
    char what[200];
    std::snprintf(what, sizeof what, "%s: after the resize 1 invalidation, 1 new capture, the new arena = 2 everywhere (%llu %llu %llu)",
                  same_base ? "same base" : "moved", (unsigned long long) G.invalidations,
                  (unsigned long long) G.captures, (unsigned long long) G.replays);
    expect(G.invalidations == 1 && G.captures == 2 && G.replays == 3 && all_equal(arena, n, 2.0f) && err.empty(), what);
    G.clear();
    cudaFree(pool);
    cudaStreamDestroy(s);
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    run(false);
    run(true);
    std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
