// src/ds4/ds4_ernie_rx.cu - maya-ds4's helper-GPU expert executor for ERNIE-5.5 (see ds4_ernie_rx.h).
//
// Built apart from the engine with the helper GPU's runtime (HIP for an AMD card: tools/ds4/build_ernie_rx.sh) into a
// shared library the engine loads with dlopen (DS4_ERNIE_RX).  A decoded token's experts of one layer are computed here
// with the engine's own kernels (ds4_gguf's moe_gate_up / moe_down: ds4_iq's IQ kernels with SiTU-GLU, the weighted
// sum in plan order), asynchronously: submit() launches and returns, wait() takes the sum.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "ds4_ernie_rx.h"
#include "ds4_gguf.cuh"
#include "ds4_iq.cuh"

#if defined(_WIN32)
#define ERNX_EXPORT __declspec(dllexport)
#else
#define ERNX_EXPORT __attribute__((visibility("default")))
#endif

namespace {

namespace gg = maya::ds4::gguf;
namespace x3 = maya::kernels::exl3;

constexpr int kMax = 16, kCall = 8;   // experts a submit; a kernel call's (the token path's plan width)

struct GpuError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
void ck(cudaError_t e) {
    if (e != cudaSuccess) throw GpuError(cudaGetErrorString(e));
}
void require(bool b, const char* msg) {
    if (!b) throw std::runtime_error(msg);
}
int message(const char* s, char* err, size_t n) {
    if (err && n) std::snprintf(err, n, "%s", s);
    return 1;
}
struct DevScope {
    int old = -1;
    explicit DevScope(int dev) {
        ck(cudaGetDevice(&old));
        ck(cudaSetDevice(dev));
    }
    ~DevScope() {
        if (old >= 0) cudaSetDevice(old);
    }
};

struct Ctx {
    ernx_config cfg{};
    std::mutex mu;
    bool poisoned = false, pending = false;
    int parts = 0;
    cudaStream_t s = nullptr;
    unsigned char* pool = nullptr;
    size_t stride = 0;
    float *x = nullptr, *h = nullptr, *out = nullptr;   // device: x [dim], h [8][ff], out [2][dim]
    unsigned long long* plan_ptr = nullptr;             // device [2][8]
    float* plan_w = nullptr;                            // device [2][8]
    float *hx = nullptr, *hout = nullptr;               // pinned: x, out [2][dim]
    unsigned long long* hplan = nullptr;
    float* hw = nullptr;
    x3::ExpertLayout lay;
    std::vector<char> filled;
    ernx_stats stats{};
};

int create(const ernx_config* cfg, void** opaque, char* err, size_t n) {
    if (!opaque) return message("null output context", err, n);
    *opaque = nullptr;
    Ctx* c = nullptr;
    try {
        require(cfg && cfg->size == sizeof(*cfg) && cfg->version == 1, "invalid config ABI");
        require(cfg->dim > 0 && cfg->ff > 0 && cfg->dim % 256 == 0 && cfg->ff % 256 == 0, "unsupported shape");
        require(cfg->slots > 0 && cfg->blob_bytes > 0, "no slots");
        require(cfg->type[0] == cfg->type[1], "gate and up of different types");
        for (int r = 0; r < 3; ++r) {
            require(gg::expert_type_ok(cfg->type[r]), "unsupported quantization");
            require(cfg->off[r] < cfg->blob_bytes, "bad blob layout");
        }
        c = new Ctx;
        c->cfg = *cfg;
        c->stride = (size_t(cfg->blob_bytes) + 255) & ~size_t(255);
        c->lay.fmt = gg::kFmtMXFP4;   // (a GGUF expert layout; the roles' types below)
        for (int r = 0; r < 3; ++r) {
            c->lay.trellis[r] = size_t(cfg->off[r]);
            c->lay.gg_type[r] = cfg->type[r];
        }
        DevScope dev(cfg->device);
        gg::set_expert_act(gg::iq::kActSitu, cfg->beta, cfg->lin);
        ck(cudaStreamCreateWithFlags(&c->s, cudaStreamNonBlocking));
        require(gg::prepare(c->s), "scratch");
        ck(cudaMalloc(reinterpret_cast<void**>(&c->pool), c->stride * cfg->slots));
        ck(cudaMalloc(reinterpret_cast<void**>(&c->x), size_t(cfg->dim) * 4));
        ck(cudaMalloc(reinterpret_cast<void**>(&c->h), size_t(kCall) * cfg->ff * 4));
        ck(cudaMalloc(reinterpret_cast<void**>(&c->out), 2 * size_t(cfg->dim) * 4));
        ck(cudaMalloc(reinterpret_cast<void**>(&c->plan_ptr), 2 * kCall * sizeof(unsigned long long)));
        ck(cudaMalloc(reinterpret_cast<void**>(&c->plan_w), 2 * kCall * sizeof(float)));
        ck(cudaHostAlloc(reinterpret_cast<void**>(&c->hx), size_t(cfg->dim) * 4, cudaHostAllocDefault));
        ck(cudaHostAlloc(reinterpret_cast<void**>(&c->hout), 2 * size_t(cfg->dim) * 4, cudaHostAllocDefault));
        ck(cudaHostAlloc(reinterpret_cast<void**>(&c->hplan), 2 * kCall * sizeof(unsigned long long), cudaHostAllocDefault));
        ck(cudaHostAlloc(reinterpret_cast<void**>(&c->hw), 2 * kCall * sizeof(float), cudaHostAllocDefault));
        c->filled.assign(cfg->slots, 0);
        c->stats.pool_bytes = c->stride * cfg->slots;
        *opaque = c;
        return 0;
    } catch (const std::exception& e) {
        if (c) {
            cudaFree(c->pool);
            delete c;
        }
        return message(e.what(), err, n);
    }
}

template <class F>
int invoke(void* opaque, char* err, size_t n, F fn) {
    if (!opaque) return message("null context", err, n);
    Ctx* c = static_cast<Ctx*>(opaque);
    std::lock_guard<std::mutex> lk(c->mu);
    try {
        require(!c->poisoned, "context poisoned by a runtime failure");
        DevScope dev(c->cfg.device);
        fn(*c);
        return 0;
    } catch (const GpuError& e) {
        c->poisoned = true;
        return message(e.what(), err, n);
    } catch (const std::exception& e) {
        return message(e.what(), err, n);
    }
}

int upload(void* opaque, uint32_t slot, const void* blob, size_t bytes, char* err, size_t n) {
    return invoke(opaque, err, n, [&](Ctx& c) {
        require(slot < c.cfg.slots && blob != nullptr && bytes == c.cfg.blob_bytes, "bad upload");
        ck(cudaMemcpy(c.pool + size_t(slot) * c.stride, blob, bytes, cudaMemcpyHostToDevice));
        c.filled[slot] = 1;
        ++c.stats.uploads;
    });
}

int submit(void* opaque, uint32_t count, const uint32_t* slots, const float* w, const float* x, char* err, size_t n) {
    return invoke(opaque, err, n, [&](Ctx& c) {
        require(!c.pending, "a submit is already in flight");
        require(count >= 1 && count <= kMax && slots && w && x, "bad submit arguments");
        const int D = c.cfg.dim;
        std::memcpy(c.hx, x, size_t(D) * 4);
        ck(cudaMemcpyAsync(c.x, c.hx, size_t(D) * 4, cudaMemcpyHostToDevice, c.s));
        c.parts = (int) ((count + kCall - 1) / kCall);
        for (int p = 0; p < c.parts; ++p) {
            const int i0 = p * kCall, k = std::min<int>(kCall, (int) count - i0);
            for (int i = 0; i < kCall; ++i) {
                c.hplan[p * kCall + i] = 0;
                c.hw[p * kCall + i] = 0.0f;
            }
            for (int i = 0; i < k; ++i) {
                const uint32_t sl = slots[i0 + i];
                require(sl < c.cfg.slots && c.filled[sl], "slot not filled");
                c.hplan[p * kCall + i] = (unsigned long long) (c.pool + size_t(sl) * c.stride);
                c.hw[p * kCall + i] = w[i0 + i];
            }
            ck(cudaMemcpyAsync(c.plan_ptr + p * kCall, c.hplan + p * kCall, kCall * sizeof(unsigned long long),
                               cudaMemcpyHostToDevice, c.s));
            ck(cudaMemcpyAsync(c.plan_w + p * kCall, c.hw + p * kCall, kCall * sizeof(float), cudaMemcpyHostToDevice,
                               c.s));
            x3::MoeArgs a;
            a.plan_ptr = c.plan_ptr + p * kCall;
            a.plan_w = c.plan_w + p * kCall;
            a.lay = c.lay;
            a.k = k;
            a.n_embd = D;
            a.n_ff = c.cfg.ff;
            a.limit = c.cfg.beta;
            const int e0 = gg::launch_errors();
            gg::moe_gate_up(a, c.x, c.h, c.s);
            gg::moe_down(a, c.h, nullptr, c.out + size_t(p) * D, c.s);
            ck(cudaGetLastError());
            require(gg::launch_errors() == e0, "a kernel launch failed");
        }
        ck(cudaMemcpyAsync(c.hout, c.out, size_t(c.parts) * D * 4, cudaMemcpyDeviceToHost, c.s));
        c.pending = true;
        ++c.stats.submits;
        c.stats.experts += count;
    });
}

int wait(void* opaque, float* out, char* err, size_t n) {
    return invoke(opaque, err, n, [&](Ctx& c) {
        require(c.pending && out != nullptr, "nothing submitted");
        c.pending = false;
        ck(cudaStreamSynchronize(c.s));
        const int D = c.cfg.dim;
        std::memcpy(out, c.hout, size_t(D) * 4);
        for (int p = 1; p < c.parts; ++p)
            for (int e = 0; e < D; ++e) out[e] += c.hout[size_t(p) * D + e];
    });
}

int stats(void* opaque, ernx_stats* out, char* err, size_t n) {
    return invoke(opaque, err, n, [&](Ctx& c) {
        require(out != nullptr, "null stats");
        *out = c.stats;
    });
}

int destroy(void** opaque, char* err, size_t n) {
    if (!opaque || !*opaque) return 0;
    Ctx* c = static_cast<Ctx*>(*opaque);
    *opaque = nullptr;
    int rc = 0;
    try {
        std::lock_guard<std::mutex> lk(c->mu);
        DevScope dev(c->cfg.device);
        if (c->s) cudaStreamSynchronize(c->s);
        cudaFree(c->pool);
        cudaFree(c->x);
        cudaFree(c->h);
        cudaFree(c->out);
        cudaFree(c->plan_ptr);
        cudaFree(c->plan_w);
        cudaFreeHost(c->hx);
        cudaFreeHost(c->hout);
        cudaFreeHost(c->hplan);
        cudaFreeHost(c->hw);
    } catch (const std::exception& e) {
        rc = message(e.what(), err, n);
    }
    delete c;
    return rc;
}

const ernx_api kApi = {sizeof(ernx_api), 1, create, upload, submit, wait, stats, destroy};

}  // namespace

extern "C" ERNX_EXPORT const ernx_api* ernx_get_api(uint32_t version) { return version == 1 ? &kApi : nullptr; }
