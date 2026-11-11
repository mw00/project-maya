// src/core/glm_dsa_prefill_stream.cpp - the glm-dsa port B9.9: the prompt's expert streaming driver, run-time side
// (the next parts' early prestage, the NUMA-bound landing ring, the node-split host share).  The design and the env
// knobs are in glm_dsa_prefill_stream.hpp; the pure schedule in glm_dsa_prefill_stream_plan.cpp; one node's share
// arithmetic in glm_dsa_prefill_share.cpp.  Maya-owned glm_prefill.cu calls in through one-line hooks
// (STRATA-EDITS.txt, B9.9).
#include "strata/core/glm_env.hpp"
#include "glm_dsa_prefill_stream.hpp"

#include "glm_fast_state.hpp"
#include "glm_tier_numa.hpp"
#include "strata/core/glm_model.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <string>

#ifdef __linux__
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace strata::core {

namespace glmpfs {

namespace {

// the NUMA node a GPU hangs off (sysfs, by its PCI address), -1 when unknown
int gpu_node(int dev) {
    int node = -1;
#ifdef __linux__
    char bus[32] = {0};
    if (cudaDeviceGetPCIBusId(bus, (int) sizeof bus, dev) != cudaSuccess) {
        cudaGetLastError();
        return -1;
    }
    std::string b(bus);
    for (char& ch : b) ch = (char) std::tolower((unsigned char) ch);
    std::ifstream("/sys/bus/pci/devices/" + b + "/numa_node") >> node;
#else
    (void) dev;
#endif
    return node;
}

// the host's NUMA nodes (the highest online node + 1; 1 without NUMA)
int n_nodes() {
#ifdef __linux__
    std::ifstream f("/sys/devices/system/node/online");
    std::string s;
    if (!(f >> s)) return 1;
    int hi = 0;
    for (size_t i = 0; i < s.size();) {
        if (!std::isdigit((unsigned char) s[i])) {
            ++i;
            continue;
        }
        size_t j = i;
        while (j < s.size() && std::isdigit((unsigned char) s[j])) ++j;
        hi = std::max(hi, std::atoi(s.substr(i, j - i).c_str()));
        i = j;
    }
    return hi + 1;
#else
    return 1;
#endif
}

std::mutex g_mu;
std::map<void*, size_t> g_bound;   // landing rings this file mapped: base -> length

}  // namespace

int host_alloc_node(void** p, size_t bytes, int dev, bool dsa) {
    *p = nullptr;
    const auto plain = [&]() { return (int) cudaHostAlloc(p, bytes, cudaHostAllocDefault); };
    if (!dsa || bytes == 0) return plain();
    const int node = staging_node(gpu_node(dev), n_nodes(), getenv("MAYA_GLM_STAGE_NUMA"));
    if (node < 0 || node >= 64) return plain();
#ifdef __linux__
    const size_t huge = (size_t) 2 << 20;
    const size_t len = (bytes + huge - 1) / huge * huge;
    void* m = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) return plain();
    unsigned long mask[2] = {0, 0};
    mask[node / 64] |= 1ul << (node % 64);
    constexpr int kMpolBind = 2;
    if (syscall(SYS_mbind, m, len, kMpolBind, mask, (unsigned long) (sizeof mask * 8), 0u) != 0) {
        munmap(m, len);
        return plain();
    }
#ifdef MADV_HUGEPAGE
    madvise(m, len, MADV_HUGEPAGE);   // (a hint: the ring is streamed through by DMA)
#endif
    std::memset(m, 0, len);           // fault in on the bound node before the driver pins it
    if (cudaHostRegister(m, len, cudaHostRegisterPortable) != cudaSuccess) {
        cudaGetLastError();
        munmap(m, len);
        return plain();
    }
    {
        std::lock_guard<std::mutex> lk(g_mu);
        g_bound[m] = len;
    }
    *p = m;
    std::fprintf(stderr, "glm prefill: CUDA%d disk landing ring %.0f MB bound to NUMA node %d (B9.9)\n", dev,
                 (double) len / 1048576.0, node);
    return 0;
#else
    return plain();
#endif
}

void host_free_node(void* p) {
    if (p == nullptr) return;
    size_t len = 0;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        const auto it = g_bound.find(p);
        if (it != g_bound.end()) {
            len = it->second;
            g_bound.erase(it);
        }
    }
    if (len == 0) {
        cudaFreeHost(p);
        return;
    }
#ifdef __linux__
    cudaHostUnregister(p);
    munmap(p, len);
#endif
}

bool share_ok(const glmtier::TierState* T, int il) {
    if (!glmtier::split_layer(T, il)) return true;
    static const bool on = [] {
        return strata::glmenv::env_flag("MAYA_GLM_PREFILL_SPLIT_SHARE", true);
    }();
    return on && T->lane[0] != nullptr && T->lane[1] != nullptr && il < (int) T->half.size() &&
           T->half[(size_t) il].n_ff > 0 && il < (int) T->L.size() && T->L[(size_t) il].blob > 0;
}

bool split_share(glmtier::TierState* T, const kc::NativeFmt& nf, const Glm5Geometry& g, int il, int n_exp,
                 const std::function<ShareX(int)>& ex, const float* x_h, const int* rt_h, int rows, float* out_h) {
    if (!glmtier::split_layer(T, il)) return false;
    if (!share_ok(T, il)) {
        // (the plan admits a split layer only when share_ok holds)
        std::fprintf(stderr, "glm prefill: layer %d is node-split but has no node-split host share\n", il);
        std::abort();
    }
    if (n_exp <= 0 || rows <= 0) return true;
    std::vector<ShareX> xs((size_t) n_exp);
    for (int j = 0; j < n_exp; ++j) xs[(size_t) j] = ex(j);
    const kc::NativeFmt& hf = T->half[(size_t) il];
    const glmtier::SplitLayout& L = T->L[(size_t) il];
    const size_t D = glmtier::split_offset(T, il);
    const int E = (int) g.n_embd;
    constexpr size_t kA = kc::kNativeActBytes;
    std::lock_guard<std::mutex> one(glmtier::lane_mutex());
    static std::vector<uint8_t> act;   // (guarded by the lane mutex)
    static ShareScratch sc[2];
    if (act.size() < (size_t) rows * kA) act.resize((size_t) rows * kA);
    const auto run_on = [T](int h) -> RunFn {
        glmfast::Workers* pool = T->lane[h]->pool.get();
        return [pool](int n, const std::function<void(int)>& fn) { pool->run(n, fn); };
    };
    // the rows' activations once (both halves dot the same gate / up rows' format)
    run_on(0)(rows, [&](int r) { kc::native_quant_act(nf, x_h + (size_t) rt_h[r] * E, act.data() + (size_t) r * kA); });
    std::function<void()> task[2];
    for (int h = 0; h < 2; ++h)
        task[h] = [&, h] {
            share_node(hf, L, n_exp, xs.data(), h == 0 ? 0 : D, act.data(), rows, g.swiglu_exp, sc[h], run_on(h));
        };
    T->lane[1]->start(&task[1]);
    T->lane[0]->start(&task[0]);
    T->lane[0]->wait();
    T->lane[1]->wait();
    share_add(sc[0].part.data(), sc[1].part.data(), out_h, (size_t) rows * E, run_on(0), rows);
    return true;
}

}  // namespace glmpfs

// Before part 0 starts chunk `chunk`, the later parts that are idle run their prefill_half up to their first MoE
// layer's prestage (the copies land on their own links while the parts before them compute); their real pass skips
// that prestage (glmpfs::early_take).
bool Glm5Model::prefill_stream_early(int64_t p0, int T, int chunk, bool pipelined, std::string& err) {
    if (split_next_ == nullptr || !glmpfs::stream_on(g_.arch == GlmArch::GlmDsa)) return true;
    int n = 0;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) ++n;
    if (chunk == 0)   // (a record a cancelled or failed prompt left unclaimed must not skip a later prestage)
        for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) glmpfs::early_forget(m);
    const std::vector<int> ks = glmpfs::early_parts(n, chunk, pipelined);
    if (ks.empty()) return true;
    int cur = 0;
    cudaGetDevice(&cur);
    int k = 0;
    bool ok = true;
    for (Glm5Model* m = this; m != nullptr && ok; m = m->split_next_.get(), ++k) {
        if (std::find(ks.begin(), ks.end(), k) == ks.end() || m->pf_ == nullptr || m->fast_ == nullptr) continue;
        cudaSetDevice(m->dev_);
        glmpfs::set_prestage_only(m);
        ok = m->prefill_half(p0, T, err, nullptr);
        glmpfs::set_prestage_only(nullptr);
        if (ok) glmpfs::early_record(m, p0, T);
    }
    cudaSetDevice(cur);
    return ok;
}

}  // namespace strata::core
