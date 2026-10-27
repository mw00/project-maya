// src/ds4/ds4_lt.cu - see ds4_lt.hpp.
#include "ds4_lt.hpp"

#include <string>

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
// (the HIP shim's CUDA shuffle spellings would expand inside hipBLASLt's headers - as src/prefill/gemm.cu)
#undef __shfl_xor_sync
#undef __shfl_down_sync
#undef __shfl_up_sync
#undef __shfl_sync
#undef __ballot_sync
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <tuple>
#include <vector>
#endif

namespace strata::ds4::lt {

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
namespace {

constexpr size_t kWorkspace = (size_t) 64 << 20;
constexpr int kCandidates = 16;

struct Key {
    int n, k, ldx, ldy, tb;
    bool operator<(const Key& o) const { return std::tie(n, k, ldx, ldy, tb) < std::tie(o.n, o.k, o.ldx, o.ldy, o.tb); }
};
struct Pick {
    bool ok = false;
    hipblasLtMatmulAlgo_t algo{};
};
struct Dev {
    std::mutex mu;                 // this device's handle, workspace and picks
    bool tried = false, live = false;
    hipblasLtHandle_t h = nullptr;
    void* ws = nullptr;
    std::string tag;                 // "<arch> <hipBLASLt version>": the picks of this GPU kind in the store
    std::map<Key, Pick> picks;
    std::map<Key, int> saved;        // the store's solution ids for this tag
};
std::mutex g_mu;                   // store path and file appends only; never held while waiting on a GPU
Dev g_dev[64];
std::string g_store;                 // set_store's file ("": the picks this process makes only)

struct Desc {   // y (n x t, FP32, ldy) = W^T (W: k x n column-major = [n][k] row-major) . x (k x t, ldx)
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    ~Desc() {
        if (op) hipblasLtMatmulDescDestroy(op);
        if (a) hipblasLtMatrixLayoutDestroy(a);
        if (b) hipblasLtMatrixLayoutDestroy(b);
        if (c) hipblasLtMatrixLayoutDestroy(c);
    }
    bool init(int n, int k, int ldx, int t, int ldy) {
        const hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
        return hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F) == HIPBLAS_STATUS_SUCCESS &&
               hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta) == HIPBLAS_STATUS_SUCCESS &&
               hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb) == HIPBLAS_STATUS_SUCCESS &&
               hipblasLtMatrixLayoutCreate(&a, HIP_R_16F, k, n, k) == HIPBLAS_STATUS_SUCCESS &&
               hipblasLtMatrixLayoutCreate(&b, HIP_R_16F, k, t, ldx) == HIPBLAS_STATUS_SUCCESS &&
               hipblasLtMatrixLayoutCreate(&c, HIP_R_32F, n, t, ldy) == HIPBLAS_STATUS_SUCCESS;
    }
};

int bucket(int t) {   // (a kernel picked for a power-of-two bucket of rows serves every count in it)
    int b = kMinRows;
    while (b < t) b *= 2;
    return b;
}

bool off() {
    static const bool o = std::getenv("DS4_LT") != nullptr && std::atoi(std::getenv("DS4_LT")) == 0;
    return o;
}

// the store's lines for this device's tag: "<arch> <version> n k ldx ldy rows solution"
void load_saved(Dev& d, const std::string& store) {
    if (store.empty()) return;
    std::ifstream f(store);
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream is(line);
        std::string arch;
        int ver = 0, sol = -1;
        Key k{};
        if (is >> arch >> ver >> k.n >> k.k >> k.ldx >> k.ldy >> k.tb >> sol && arch + " " + std::to_string(ver) == d.tag)
            d.saved[k] = sol;
    }
}

}  // namespace

void set_store(const char* path) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_store = path != nullptr ? path : "";
}
bool available() { return !off(); }

bool gemm(void* stream, const uint16_t* W, int n, int k, const uint16_t* x, int ldx, int t, float* y, int ldy) {
    if (off() || t < kMinRows || n <= 0 || k <= 0 || ldx < k || ldy < n) return false;
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) return false;
    Dev& d = g_dev[dev];
    // Another GPU's stream may wait for this device's layer hand-off while its first Lt shape is timed.
    // A process-wide lock across hipEventSynchronize would prevent this device from submitting that work.
    std::lock_guard<std::mutex> lk(d.mu);
    std::string store;
    {
        std::lock_guard<std::mutex> store_lk(g_mu);
        store = g_store;
    }
    if (!d.tried) {
        d.tried = true;
        d.live = hipblasLtCreate(&d.h) == HIPBLAS_STATUS_SUCCESS && hipMalloc(&d.ws, kWorkspace) == hipSuccess;
        if (!d.live) std::fprintf(stderr, "ds4 gemm: hipBLASLt unavailable on device %d - hipBLAS instead\n", dev);
        hipDeviceProp_t p{};
        int ver = 0;
        if (d.live && hipGetDeviceProperties(&p, dev) == hipSuccess &&
            hipblasLtGetVersion(d.h, &ver) == HIPBLAS_STATUS_SUCCESS) {
            std::string arch(p.gcnArchName);
            const size_t colon = arch.find(':');
            if (colon != std::string::npos) arch.resize(colon);
            d.tag = arch + " " + std::to_string(ver);
            load_saved(d, store);
        }
    }
    if (!d.live) return false;
    Desc desc;
    if (!desc.init(n, k, ldx, t, ldy)) return false;
    const float alpha = 1.0f, beta = 0.0f;
    hipStream_t s = (hipStream_t) stream;
    const Key key{n, k, ldx, ldy, bucket(t)};
    auto it = d.picks.find(key);
    if (it == d.picks.end()) {
        Pick pick;
        const bool verbose = std::getenv("DS4_LT_VERBOSE") != nullptr;
        const auto sv = d.saved.find(key);
        if (sv != d.saved.end()) {   // the store's pick for this GPU kind and hipBLASLt, if it still takes the shape
            std::vector<int> ids{sv->second};
            std::vector<hipblasLtMatmulHeuristicResult_t> res;
            size_t need = 0;
            if (hipblaslt_ext::getAlgosFromIndex(d.h, ids, res) == HIPBLAS_STATUS_SUCCESS && !res.empty() &&
                res.front().state == HIPBLAS_STATUS_SUCCESS &&
                hipblaslt_ext::matmulIsAlgoSupported(d.h, desc.op, &alpha, desc.a, desc.b, &beta, desc.c, desc.c,
                                                     res.front().algo, need) == HIPBLAS_STATUS_SUCCESS &&
                need <= kWorkspace) {
                pick.ok = true;
                pick.algo = res.front().algo;
            }
            if (verbose)
                std::fprintf(stderr, "ds4 gemm: hipBLASLt n %d k %d ldx %d ldy %d rows <= %d: stored solution %d %s\n",
                             n, k, ldx, ldy, key.tb, sv->second, pick.ok ? "taken" : "refused - measuring again");
        }
        if (!pick.ok) {   // the shape's first product here: every candidate timed on this GPU, the fastest kept
            hipblasLtMatmulPreference_t pref = nullptr;
            std::vector<hipblasLtMatmulHeuristicResult_t> res(kCandidates);
            int got = 0;
            size_t wsb = kWorkspace;
            if (hipblasLtMatmulPreferenceCreate(&pref) == HIPBLAS_STATUS_SUCCESS &&
                hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof wsb) ==
                    HIPBLAS_STATUS_SUCCESS)
                (void) hipblasLtMatmulAlgoGetHeuristic(d.h, desc.op, desc.a, desc.b, desc.c, desc.c, pref, kCandidates,
                                                      res.data(), &got);
            if (pref) hipblasLtMatmulPreferenceDestroy(pref);
            hipEvent_t e0 = nullptr, e1 = nullptr;
            float best = 1e30f;
            if (got > 0 && hipEventCreate(&e0) == hipSuccess && hipEventCreate(&e1) == hipSuccess) {
                for (int i = 0; i < got; ++i) {
                    if (res[(size_t) i].state != HIPBLAS_STATUS_SUCCESS || res[(size_t) i].workspaceSize > kWorkspace)
                        continue;
                    const hipblasLtMatmulAlgo_t* algo = &res[(size_t) i].algo;
                    if (hipblasLtMatmul(d.h, desc.op, &alpha, W, desc.a, x, desc.b, &beta, y, desc.c, y, desc.c, algo,
                                        d.ws, kWorkspace, s) != HIPBLAS_STATUS_SUCCESS)   // (a warm-up run)
                        continue;
                    (void) hipEventRecord(e0, s);
                    bool ran = true;
                    for (int r = 0; r < 2 && ran; ++r)
                        ran = hipblasLtMatmul(d.h, desc.op, &alpha, W, desc.a, x, desc.b, &beta, y, desc.c, y, desc.c,
                                              algo, d.ws, kWorkspace, s) == HIPBLAS_STATUS_SUCCESS;
                    (void) hipEventRecord(e1, s);
                    float ms = 0.0f;
                    if (!ran || hipEventSynchronize(e1) != hipSuccess || hipEventElapsedTime(&ms, e0, e1) != hipSuccess)
                        continue;
                    if (ms < best) {
                        best = ms;
                        pick.ok = true;
                        pick.algo = *algo;
                    }
                }
            }
            if (e0) (void) hipEventDestroy(e0);
            if (e1) (void) hipEventDestroy(e1);
            (void) hipGetLastError();
            if (verbose)
                std::fprintf(stderr, "ds4 gemm: hipBLASLt n %d k %d ldx %d ldy %d rows <= %d: %d candidates, %s %.3f ms\n",
                             n, k, ldx, ldy, key.tb, got, pick.ok ? "the fastest" : "none ran", pick.ok ? best / 2 : 0.0f);
            if (pick.ok && !store.empty() && !d.tag.empty()) {   // (kept for the next start on this GPU kind)
                std::lock_guard<std::mutex> store_lk(g_mu);
                std::ofstream f(store, std::ios::app);
                f << d.tag << ' ' << n << ' ' << k << ' ' << ldx << ' ' << ldy << ' ' << key.tb << ' '
                  << hipblaslt_ext::getIndexFromAlgo(pick.algo) << '\n';
            }
        }
        it = d.picks.emplace(key, pick).first;
    }
    if (!it->second.ok) return false;
    if (hipblasLtMatmul(d.h, desc.op, &alpha, W, desc.a, x, desc.b, &beta, y, desc.c, y, desc.c, &it->second.algo, d.ws,
                        kWorkspace, s) == HIPBLAS_STATUS_SUCCESS)
        return true;
    it->second.ok = false;   // (this shape's kernel refused: hipBLAS from now on)
    (void) hipGetLastError();
    return false;
}
#else
void set_store(const char*) {}
bool available() { return false; }
bool gemm(void*, const uint16_t*, int, int, const uint16_t*, int, int, float*, int) { return false; }
#endif

}  // namespace strata::ds4::lt
