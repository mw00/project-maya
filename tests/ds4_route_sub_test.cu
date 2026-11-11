// Real routing kernels, no model: native/off parity, primary/helper residency, reference mass and mailbox publication.
#include "../src/ds4/ds4_cache_sub.hpp"
#include "../src/ds4/ds4_kernels.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

using namespace strata::ds4;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); std::exit(1); } } while (0)
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)

template<class T> T* device(size_t n) {
    T* p = nullptr;
    CK(cudaMalloc((void**) &p, n*sizeof(T)));
    return p;
}
template<class T> T* mapped(size_t n) {
    T* p = nullptr;
    CK(cudaHostAlloc((void**) &p, n*sizeof(T), cudaHostAllocMapped | cudaHostAllocPortable));
    return p;
}

static void run(int gpu) {
    CK(cudaSetDevice(gpu));
    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));
    constexpr int D = 16, N = 1024;
    float* logits = device<float>(kWin*N);
    float* bias = device<float>(N);
    float* picture_bias = device<float>(N);
    float* input = device<float>(kWin*D);
    auto table = device<unsigned long long>(N);
    auto helper = device<uint8_t>(N);
    auto idx = device<int>(kWin*kPicks);
    auto weights = device<float>(kWin*kPicks);
    auto plan = device<unsigned long long>(kWin*kPicks);
    auto cpu_any = device<int>(1);
    Req* req = mapped<Req>(1);
    Resp* resp = mapped<Resp>(1);
    SubRoute* original = mapped<SubRoute>(1);
    unsigned* host_done = mapped<unsigned>(1);
    std::vector<float> gl(kWin*N), bs(N), pbs(N), x(kWin*D);
    std::vector<unsigned long long> vram(N);
    std::vector<uint8_t> hr(N);
    std::vector<int> actual(kWin*kPicks), native(kWin*kPicks);
    std::vector<float> w(kWin*kPicks), native_w(kWin*kPicks);
    std::vector<unsigned long long> ptr(kWin*kPicks), native_ptr(kWin*kPicks);
    for (int i = 0; i < kWin*D; ++i) x[i] = (float)(i+1);
    for (int t = 0; t < kWin; ++t)
        for (int i = 0; i < N; ++i) {
            // Perfect positive squares >20: CPU/GPU sqrt is exact and softplus's exponential branch is avoided.
            const float root = (float)(5+(i*37+t*11)%N);
            gl[t*N+i] = root*root;
        }
    for (int i = 0; i < N; ++i) {
        bs[i] = (float)((i*13)%17-8)*128;
        pbs[i] = -bs[i];
        vram[i] = i%4 == 0 ? 4096ull*(i+1) : 0ull;
        hr[i] = i%5 == 1;
    }
    CK(cudaMemcpy(logits, gl.data(), gl.size()*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(bias, bs.data(), bs.size()*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(picture_bias, pbs.data(), pbs.size()*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(input, x.data(), x.size()*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(table, vram.data(), vram.size()*8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(helper, hr.data(), hr.size(), cudaMemcpyHostToDevice));
    int cases = 0;
    for (int n : {16, 384, 1024}) for (int k : {1, 6, 8}) for (int T : {1, 4, 16}) {
        auto route_once = [&](const CacheSub* sub) {
            std::memset(req, 0, sizeof(*req));
            std::memset(resp, 0, sizeof(*resp));
            std::memset(original, 0x7f, sizeof(*original));
            *host_done = 0;
            CK(cudaMemsetAsync(cpu_any, 0, sizeof(int), stream));
            route_publish(logits, bias, n, k, T, 1.5f, table, input, D, idx, weights, plan, req, 7, stream,
                          picture_bias, 0xaaaau, 0, 0, resp, cpu_any, host_done, sub);
            CK(cudaStreamSynchronize(stream));
            CHECK(req->seq == 7 && req->T == T);
            CK(cudaMemcpy(actual.data(), idx, actual.size()*4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(w.data(), weights, w.size()*4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(ptr.data(), plan, ptr.size()*8, cudaMemcpyDeviceToHost));
        };
        route_once(nullptr);
        native = actual; native_w = w; native_ptr = ptr;
        CacheSub sub;
        sub.ratio = 0;
        sub.original = original;
        route_once(&sub);
        for (int t = 0; t < T; ++t) for (int i = 0; i < k; ++i) {
            const int a = t*kPicks+i;
            CHECK(actual[a] == native[a] && ptr[a] == native_ptr[a]);
            CHECK(std::memcmp(&w[a], &native_w[a], sizeof(float)) == 0);
        }
        CHECK(original->ids[0][0] == 0x7f7f7f7f); // disabled path never touches the sideband
        for (int cap : {12, 16, 32}) for (float budget : {0.0f, 0.6f, 2.0f}) for (float ratio : {.1f, .5f, 1.0f}) for (bool healthy : {false, true}) {
            sub.ratio = ratio; sub.candidates = cap; sub.budget = budget;
            sub.helper = healthy ? helper : nullptr;
            route_once(&sub);
            for (int t = 0; t < T; ++t) {
                std::vector<int> order(n);
                std::iota(order.begin(), order.end(), 0);
                auto score = [&](int i) { return std::sqrt(gl[t*N+i]); };
                const auto& correction = (0xaaaau >> t)&1 ? pbs : bs;
                std::sort(order.begin(), order.end(), [&](int a, int b) {
                    const float x = score(a)+correction[a], y = score(b)+correction[b];
                    return x > y || (x == y && a < b);
                });
                const int count = std::min(n, cap);
                float scores[32], mass = 0;
                uint8_t resident[32]; int selected[kPicks];
                for (int i = 0; i < count; ++i) {
                    scores[i] = score(order[i]);
                    resident[i] = vram[order[i]] != 0 || (healthy && hr[order[i]]);
                    if (i < k) mass += scores[i];
                }
                CHECK(cache_sub::choose(order.data(), scores, resident, k, count, ratio, true, budget, selected));
                for (int i = 0; i < k; ++i) {
                    const int at = t*kPicks+i, expected = order[selected[i]];
                    CHECK(original->ids[t][i] == order[i]);
                    CHECK(std::fabs(original->weights[t][i] - scores[i]/mass*1.5f) < 1e-6f);
                    CHECK(actual[at] == expected && req->ids[t][i] == expected);
                    CHECK(std::fabs(w[at]-score(expected)/mass*1.5f) < 1e-6f);
                    CHECK(req->w[t][i] == w[at] && ptr[at] == vram[expected]);
                    if (!vram[expected]) for (int d = 0; d < D; ++d) CHECK(req->x[t][d] == x[t*D+d]);
                }
            }
            ++cases;
        }
    }
    for (void* p : {(void*)logits, (void*)bias, (void*)picture_bias, (void*)input, (void*)table, (void*)helper,
                   (void*)idx, (void*)weights, (void*)plan, (void*)cpu_any}) CK(cudaFree(p));
    for (void* p : {(void*)req, (void*)resp, (void*)original, (void*)host_done}) CK(cudaFreeHost(p));
    CK(cudaStreamDestroy(stream));
    std::printf("GPU%d: %d substitution cases and native/off bit parity passed\n", gpu, cases);
}

int main() {
#ifdef _WIN32
    _putenv_s("STRATA_DS4_RESIDENT_SKIP_W", ""); _putenv_s("STRATA_DS4_RESIDENT_TOPK", "");
#else
    unsetenv("STRATA_DS4_RESIDENT_SKIP_W"); unsetenv("STRATA_DS4_RESIDENT_TOPK");
#endif
    int count = 0;
    CK(cudaGetDeviceCount(&count));
    CHECK(count > 0);
    for (int i = 0; i < std::min(count, 2); ++i) run(i);
}
