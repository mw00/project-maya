// src/prefill/ggml_cuda_host.cu - prompt-speed plan step 2b: the host-side symbols of llama.cpp's ggml-cuda that its MMQ
// and quantize code reference, for the MMQ kernels compiled into strata_mmq without the rest of ggml-cuda.cu.
#include "common.cuh"
#include "strata/prefill/failure.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

[[noreturn]] void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    std::fprintf(stderr, "ggml-cuda (strata mmq): %s: %s\n  in %s at %s:%d\n", msg, stmt, func, file, line);
    std::abort();
}

int ggml_cuda_get_device() {
    int id = 0;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

#if defined(GGML_USE_HIP)
namespace {

unsigned parse_hex(const std::string& value) {
    if (value.empty()) return 0;
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value.c_str(), &end, 16);
    return end == value.c_str() || *end != '\0' ? 0 : static_cast<unsigned>(parsed);
}

// Match GGML's AMD cc encoding: OFFSET_AMD + gfx major * 0x100 + minor/stepping.
// For example, gfx1100 maps to GGML_CUDA_CC_RDNA3 (0x01001100).
int ggml_cuda_parse_amd_id(const char* device_name) {
    if (device_name == nullptr) return GGML_CUDA_CC_OFFSET_AMD;

    std::string arch(device_name);
    if (arch.compare(0, 3, "gfx") == 0) arch.erase(0, 3);
    const size_t suffix = arch.find(':');
    if (suffix != std::string::npos) arch.resize(suffix);

    const size_t generic = arch.rfind("-generic");
    if (generic != std::string::npos && generic + 8 == arch.size()) arch.resize(generic);

    unsigned major = 0;
    unsigned minor = 0;
    const size_t separator = arch.find('-');
    if (separator != std::string::npos) {
        major = parse_hex(arch.substr(0, separator));
        minor = parse_hex(arch.substr(separator + 1)) * 0x10;
    } else if (arch.size() >= 3) {
        minor = parse_hex(arch.substr(arch.size() - 2));
        major = parse_hex(arch.substr(0, arch.size() - 2));
    }
    return GGML_CUDA_CC_OFFSET_AMD + static_cast<int>(major * 0x100 + minor);
}

}  // namespace
#endif

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = [] {
        ggml_cuda_device_info in = {};
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess) n = 0;
        n = n > GGML_CUDA_MAX_DEVICES ? GGML_CUDA_MAX_DEVICES : n;
        in.device_count = n;
        in.physical_device_count = n;
        for (int id = 0; id < n; ++id) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
            auto & d = in.devices[id];
#if defined(GGML_USE_HIP)
            d.cc = ggml_cuda_parse_amd_id(prop.gcnArchName);
            if ((d.cc & 0xff00) == 0) {
                // Fall back to HIP's device version fields if its architecture name is unavailable.
                d.cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100 + prop.minor * 0x10;
            }
            d.smpbo = prop.sharedMemPerBlock;
            d.integrated = false;
            d.supports_cooperative_launch = false;
#else
            d.cc = 100 * prop.major + 10 * prop.minor;
            d.smpbo = prop.sharedMemPerBlockOptin;
            d.integrated = prop.integrated != 0;
            d.supports_cooperative_launch = prop.cooperativeLaunch != 0;
#endif
            d.nsm = prop.multiProcessorCount;
            d.smpb = prop.sharedMemPerBlock;
            d.vmm = false;
            d.total_vram = prop.totalGlobalMem;
            d.warp_size = prop.warpSize;
            d.physical_device = id;
            d.physical_share_count = 1;
            d.virtual_index = 0;
        }
        return in;
    }();
    return info;
}

namespace {
// The prompt path's MMQ scratch, where the pool has no business taking VRAM: the part of the borrowed region
// (prefill_bind's, the expert pool's tail) that this chunk's layout did not use.  It is the prompt's memory for as
// long as the prompt holds the loan, so the fixup tiles cost the card nothing - and on a card whose free VRAM is a
// handful of MB at prompt time (CUDA1: 3.4 MB of 7,707 MB after the pool, the decode path and cuBLAS had taken
// theirs) a cudaMalloc there is what used to abort the engine mid-prompt.  Set per chunk (the layout's need moves
// with T), so the buffers cached from the previous chunk are dropped when it moves.
struct Slab {
    uint8_t * base = nullptr;
    size_t bytes = 0, used = 0;
};
Slab g_slab[GGML_CUDA_MAX_DEVICES];
std::mutex g_slab_mu;

// MMQ's stream-k fixup asks for nsm*J*I floats, and its J search (8..128 by 8) picks a different J per launch, so a
// cache keyed on the exact size holds one buffer per J: 32 buffers, 31.5 MB, measured on a 579-token chunk - and the
// same at 16k, since J does not grow with the chunk.  Rounding a request up to a size class (a power of two, 128 KB
// floor - 128 KB, 256 KB, 512 KB, 1 MB, 2 MB cover the whole J range on a 30-SM card) bounds the pool to a few MB
// whatever J wins.  Handing back more than was asked is what every caching allocator does; the kernels only touch
// the first nsm*J*I floats.
size_t size_class(size_t n) {
    size_t c = 128u << 10;
    while (c < n) c <<= 1;
    return c;
}

// Buffers are kept and reused: MMQ asks for the same few sizes every launch (its stream-k fixup tiles).
struct CachingPool : ggml_cuda_pool {
    struct Buf { void * p; size_t size; bool used; bool slab; };
    std::vector<Buf> bufs;
    std::mutex mu;
    int dev;
    void * slab_seen = nullptr;   // the slab base the cached buffers came from; a move makes them stale
    explicit CachingPool(int device) : dev(device) {}
    size_t held() const {   // the bytes this pool holds on the card (never given back)
        size_t n = 0;
        for (const auto & b : bufs) n += b.size;
        return n;
    }
    void * alloc(size_t size, size_t * actual_size) override {
        std::lock_guard<std::mutex> lk(mu);
        const size_t cls = size_class(size);
        // A test hook: STRATA_MMQ_POOL_FAIL=<n> makes the first n allocations here fail, so the engine's recovery can be
        // checked (the request must end with an error and the engine must live).
        static std::atomic<int> fail_left = [] {
            const char* v = std::getenv("STRATA_MMQ_POOL_FAIL");
            return std::atomic<int>(v ? std::atoi(v) : 0);
        }();
        if (fail_left.load() > 0 && fail_left.fetch_sub(1) > 0) {
            std::fprintf(stderr, "ggml-cuda (strata mmq): CUDA%d wanted %.1f MB of scratch and cannot have it "
                                 "(STRATA_MMQ_POOL_FAIL)\n", dev, (double) cls / 1048576.0);
            strata::prefill::fail("prefill mmq: the scratch pool wanted " +
                                  std::to_string((unsigned long long) (cls >> 20)) + " MB and could not have it: "
                                  "out of memory");
        }
        {
            std::lock_guard<std::mutex> sl(g_slab_mu);
            Slab & s = g_slab[dev];
            if (s.base != slab_seen) {
                slab_seen = s.base;
                // the layout moved under the cached buffers: nothing is in flight between chunks, so drop them
                bufs.erase(std::remove_if(bufs.begin(), bufs.end(),
                                          [](const Buf & b) { return b.slab; }), bufs.end());
            }
            for (auto & b : bufs)
                if (!b.used && b.size >= cls) { b.used = true; *actual_size = b.size; return b.p; }
            if (s.base != nullptr && s.bytes >= s.used && s.bytes - s.used >= cls) {
                void * p = s.base + s.used;
                s.used += cls;
                *actual_size = cls;
                bufs.push_back({p, cls, true, true});
                return p;
            }
        }
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        void * p = nullptr;
        const cudaError_t e = cudaMalloc(&p, cls);
        if (e != cudaSuccess) {
            // The whole reason the engine used to abort mid-prompt with "out of memory" and nothing else.  The
            // request ends with an error now (the reason is thrown and the caller reports it); the engine lives and
            // the next request runs.
            std::fprintf(stderr,
                         "ggml-cuda (strata mmq): CUDA%d wanted %.1f MB of scratch (the prompt path's pool held %.1f MB "
                         "of its %zu buffers) and only %.1f MB of %.1f MB was free: %s\n",
                         dev, (double) cls / 1048576.0, (double) held() / 1048576.0, bufs.size(),
                         (double) free_b / 1048576.0, (double) total_b / 1048576.0, cudaGetErrorString(e));
            (void) cudaGetLastError();   // the error is reported here, not left for something else to find
            strata::prefill::fail(std::string("prefill mmq: the scratch pool wanted ") +
                                  std::to_string((unsigned long long) (cls >> 20)) + " MB and the card had " +
                                  std::to_string((unsigned long long) (free_b >> 20)) + " MB free: " +
                                  cudaGetErrorString(e));
        }
        if (cls >= ((size_t) 4 << 20))
            std::fprintf(stderr, "glm prefill mmq: CUDA%d the scratch pool takes a new %.1f MB buffer (%.1f MB free of "
                                 "%.1f MB, %.1f MB held)\n",
                         dev, (double) cls / 1048576.0, (double) free_b / 1048576.0, (double) total_b / 1048576.0,
                         (double) held() / 1048576.0);
        bufs.push_back({p, cls, true, false});
        *actual_size = cls;
        return p;
    }
    void free(void * ptr, size_t) override {
        std::lock_guard<std::mutex> lk(mu);
        for (auto & b : bufs)
            if (b.p == ptr) { b.used = false; return; }
    }
    ~CachingPool() override {
        for (auto & b : bufs)
            if (!b.slab) cudaFree(b.p);   // slab buffers are the borrowed region's tail, not ours to free
    }
};
}  // namespace

// Called by the prompt path once per chunk, before that chunk's MMQ launches: `base`/`bytes` are the tail of the
// borrowed expert pool region this chunk's layout left over (glm_prefill.cu's prefill_carve).  The bump restarts
// there; a chunk whose layout puts the tail somewhere else invalidates the buffers cached from the old one.
extern "C" void strata_mmq_pool_slab(int dev, void * base, size_t bytes) {
    std::lock_guard<std::mutex> lk(g_slab_mu);
    if (dev < 0 || dev >= GGML_CUDA_MAX_DEVICES) return;
    g_slab[dev] = {(uint8_t *) base, base != nullptr ? bytes : 0, 0};
}

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int device, int) {
    return std::unique_ptr<ggml_cuda_pool>(new CachingPool(device));
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {}
