// src/ds4/ds4_lt.cu - see ds4_lt.hpp.
#include "ds4_lt.hpp"

#include <string>

#if defined(__HIPCC__) && defined(MAYA_HIPBLASLT_AVAILABLE)
// (the HIP shim's CUDA shuffle spellings would expand inside hipBLASLt's headers - as src/prefill/gemm.cu)
#undef __shfl_xor_sync
#undef __shfl_down_sync
#undef __shfl_up_sync
#undef __shfl_sync
#undef __ballot_sync
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <tuple>
#include <vector>
#endif

namespace maya::ds4::lt {

#if defined(__HIPCC__) && defined(MAYA_HIPBLASLT_AVAILABLE)
namespace {

constexpr size_t kWorkspace = (size_t) 64 << 20;
constexpr int kCandidates = 16;

// kind: 0 gemm's shapes (W^T x), 1 gemm_n's (A as it lies, la its row stride), + 2 a loose pick (n and k rounded up
// to powers of two, the strides left out: one measurement serves every shape of the bucket that takes its kernel)
struct Key {
    int n, k, ldx, ldy, tb, la = 0, kind = 0;
    bool operator<(const Key& o) const {
        return std::tie(n, k, ldx, ldy, tb, la, kind) < std::tie(o.n, o.k, o.ldx, o.ldy, o.tb, o.la, o.kind);
    }
};
const char* const kKindTag[4] = {"", "/n", "/b", "/nb"};   // (after the arch in the store: an older build skips them)
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
    std::map<Key, Pick> exact;       // a loose pick, checked against one exact shape
    std::string arch;                // the tag's two parts
    int ver = 0;
    std::map<Key, int> saved;        // the store's solution ids for this tag
};
std::mutex g_mu;                   // store path and file appends only; never held while waiting on a GPU
Dev g_dev[64];
std::string g_store;                 // set_store's file ("": the picks this process makes only)
std::string g_store_setting;         // the setting that names another file (the warning below names it)
bool g_store_warned = false;         // a store that does not open for writing: said once

// y (n x t, FP32, ldy) = W^T (W: k x n column-major = [n][k] row-major) . x (k x t, ldx)
// as_lies: y = A . x, A n x k column-major with stride la (= [k][la] row-major, its first n columns)
struct Desc {
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    ~Desc() {
        if (op) hipblasLtMatmulDescDestroy(op);
        if (a) hipblasLtMatrixLayoutDestroy(a);
        if (b) hipblasLtMatrixLayoutDestroy(b);
        if (c) hipblasLtMatrixLayoutDestroy(c);
    }
    bool init(int n, int k, int ldx, int t, int ldy, bool as_lies = false, int la = 0) {
        const hipblasOperation_t ta = as_lies ? HIPBLAS_OP_N : HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
        const int ar = as_lies ? n : k, ac = as_lies ? k : n, ald = as_lies ? la : k;
        return hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F) == HIPBLAS_STATUS_SUCCESS &&
               hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta) == HIPBLAS_STATUS_SUCCESS &&
               hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb) == HIPBLAS_STATUS_SUCCESS &&
               hipblasLtMatrixLayoutCreate(&a, HIP_R_16F, ar, ac, ald) == HIPBLAS_STATUS_SUCCESS &&
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
        if (!(is >> arch >> ver >> k.n >> k.k >> k.ldx >> k.ldy >> k.tb >> sol) || ver != d.ver) continue;
        for (int kind = 0; kind < 4; ++kind)
            if (arch == d.arch + kKindTag[kind]) {
                k.kind = kind;
                if (kind == 1) is >> k.la;   // (gemm_n's: the row stride after the solution; a loose pick has none)
                d.saved[k] = sol;
            }
    }
}

}  // namespace

void set_store(const char* path, const char* setting) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_store = path != nullptr ? path : "";
    g_store_setting = setting != nullptr ? setting : "";
}
bool available() { return !off(); }

namespace {
int pow2_up(int v) {
    int b = 64;
    while (b < v) b *= 2;
    return b;
}
// as_lies: gemm_n's operand (la its row stride); loose: the pick by bucket, each exact shape checked once
bool run(void* stream, bool as_lies, bool loose, const uint16_t* W, int la, int n, int k, const uint16_t* x, int ldx,
         int t, float* y, int ldy) {
    if (off() || t < kMinRows || n <= 0 || k <= 0 || ldx < k || ldy < n || (as_lies && la < n)) return false;
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
            d.arch = arch;
            d.ver = ver;
            load_saved(d, store);
        }
    }
    if (!d.live) return false;
    Desc desc;
    if (!desc.init(n, k, ldx, t, ldy, as_lies, la)) return false;
    const float alpha = 1.0f, beta = 0.0f;
    hipStream_t s = (hipStream_t) stream;
    const int kind = (as_lies ? 1 : 0) + (loose ? 2 : 0);
    const Key shape{n, k, ldx, ldy, bucket(t), as_lies ? la : 0, kind};
    const Key key = loose ? Key{pow2_up(n), pow2_up(k), 0, 0, bucket(t), 0, kind} : shape;
    if (loose) {   // (this exact shape met before: its checked kernel, or none)
        const auto ex = d.exact.find(shape);
        if (ex != d.exact.end()) {
            if (!ex->second.ok) return false;
            if (hipblasLtMatmul(d.h, desc.op, &alpha, W, desc.a, x, desc.b, &beta, y, desc.c, y, desc.c, &ex->second.algo,
                                d.ws, kWorkspace, s) == HIPBLAS_STATUS_SUCCESS)
                return true;
            ex->second.ok = false;
            (void) hipGetLastError();
            return false;
        }
    }
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
                if (!f && !g_store_warned) {   // (a read-only pack: each start measured the shapes again, silently)
                    const int err = errno;
                    g_store_warned = true;
                    const std::string hint =
                        g_store_setting.empty() ? "" : "; " + g_store_setting + "=<a writable file> keeps them elsewhere";
                    std::fprintf(stderr,
                                 "ds4 gemm: the measured hipBLASLt picks are not kept - %s does not open for writing "
                                 "(errno %d: %s), so each start measures the prompt's shapes again%s\n",
                                 store.c_str(), err, std::strerror(err), hint.c_str());
                }
                f << d.arch << kKindTag[kind] << ' ' << d.ver << ' ' << key.n << ' ' << key.k << ' ' << key.ldx << ' '
                  << key.ldy << ' ' << key.tb << ' ' << hipblaslt_ext::getIndexFromAlgo(pick.algo);
                if (kind == 1) f << ' ' << key.la;
                f << '\n';
            }
        }
        it = d.picks.emplace(key, pick).first;
    }
    if (loose) {   // (the bucket's kernel for this exact shape, if it takes it)
        Pick ex = it->second;
        size_t need = 0;
        ex.ok = ex.ok && hipblaslt_ext::matmulIsAlgoSupported(d.h, desc.op, &alpha, desc.a, desc.b, &beta, desc.c, desc.c,
                                                              ex.algo, need) == HIPBLAS_STATUS_SUCCESS &&
                need <= kWorkspace;
        (void) hipGetLastError();
        const auto e2 = d.exact.emplace(shape, ex).first;
        if (!e2->second.ok) return false;
        if (hipblasLtMatmul(d.h, desc.op, &alpha, W, desc.a, x, desc.b, &beta, y, desc.c, y, desc.c, &e2->second.algo,
                            d.ws, kWorkspace, s) == HIPBLAS_STATUS_SUCCESS)
            return true;
        e2->second.ok = false;
        (void) hipGetLastError();
        return false;
    }
    if (!it->second.ok) return false;
    if (hipblasLtMatmul(d.h, desc.op, &alpha, W, desc.a, x, desc.b, &beta, y, desc.c, y, desc.c, &it->second.algo, d.ws,
                        kWorkspace, s) == HIPBLAS_STATUS_SUCCESS)
        return true;
    it->second.ok = false;   // (this shape's kernel refused: hipBLAS from now on)
    (void) hipGetLastError();
    return false;
}
}  // namespace

bool gemm(void* stream, const uint16_t* W, int n, int k, const uint16_t* x, int ldx, int t, float* y, int ldy) {
    return run(stream, false, false, W, 0, n, k, x, ldx, t, y, ldy);
}
bool gemm_loose(void* stream, const uint16_t* W, int n, int k, const uint16_t* x, int ldx, int t, float* y, int ldy) {
    return run(stream, false, true, W, 0, n, k, x, ldx, t, y, ldy);
}
bool gemm_n(void* stream, const uint16_t* A, int la, int n, int k, const uint16_t* x, int ldx, int t, float* y, int ldy,
            bool loose) {
    return run(stream, true, loose, A, la, n, k, x, ldx, t, y, ldy);
}
#else
void set_store(const char*, const char*) {}
bool available() { return false; }
bool gemm(void*, const uint16_t*, int, int, const uint16_t*, int, int, float*, int) { return false; }
bool gemm_loose(void*, const uint16_t*, int, int, const uint16_t*, int, int, float*, int) { return false; }
bool gemm_n(void*, const uint16_t*, int, int, int, const uint16_t*, int, int, float*, int, bool) { return false; }
#endif

}  // namespace maya::ds4::lt
