// Decode-only, model-free benchmark of the GLM decode attention's two largest kernels at Maya-M's geometry (090 C10:
// at the near-full 128K point the indexer's scores and MLA's split attention are its largest attention stages): the
// original kernels against the take-two shapes (glm_fast.cu dsa_score2 / mla_split2 + mla_combine2), every shape
// checked bit for bit - the scores, the selected cells and the attention's context - before it is timed.
// Run from the build directory:
//   glm_attn_bench                 parity of the default shapes, then their timings
//   glm_attn_bench --parity-only   parity only (a ctest)
//   glm_attn_bench --sweep         every take-two shape
// Geometry (docs/GLM5-FLASH.md): the indexer's 32 heads x 128, pools of 4 positions, the top 512 pools (index_topk
// 2048) and the 3-position tail; MLA's 64 heads, kv_lora 512, qk_nope 256, FP16 latents (the engine's default cache).
// The points are the 090 fixtures' native prompt counts: 28,517 (32K), 114,093 (114K), 130,397 (nf128k).
// Timing scope: in the engine a layer's pooled keys and selected latent rows come from VRAM (a whole token's weights
// pass between two calls of a layer), its query, weights and partials straight from the kernels before.  So every
// timed call reads pooled keys / latent rows last read >= 256 MiB of such reads earlier - past the RX 7900's 96 MB
// Infinity Cache: the indexer rotates over >= 256 MiB of distinct pooled copies; MLA over 128 DISJOINT cell sets (one
// shuffled permutation of a 512 MiB latent pool's rows, cut into sets: no row in two sets), each round all of them, the
// rotation never restarting (warm-up and rounds continue it); the query, the head weights and the partials stay hot
// as in the engine.  The copies and sets are bounded (<= 128, <= 512 MiB); the bench prints the bytes each round reads.
#include "strata/kernels/glm_expert_bench.hpp"
#include "strata/kernels/glm_fast.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace gf = strata::kernels::glmf;
namespace hb = strata::kernels::glmf::hip_expert_bench;
namespace {
constexpr int kH = 32, kK = 128, kPool = 4, kTop = 512, kTail = kPool - 1;
constexpr int kHeads = 64, kLora = 512, kNope = 256, kSel = kTop * kPool + kTail;
constexpr int kRows = 32768, kRowsCold = 524288;   // parity's latent rows; timing's (512 MB of FP16 rows)
constexpr int kPoints[] = {28517, 114093, 130397};

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
        std::exit(2);
    }
}
template<class T> T* dev_copy(const std::vector<T>& h) {
    T* d = nullptr;
    check(hipMalloc(&d, std::max<size_t>(1, h.size()) * sizeof(T)), "hipMalloc");
    if (!h.empty()) check(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "H2D");
    return d;
}
template<class T> std::vector<T> host_copy(const T* d, size_t n) {
    std::vector<T> h(n);
    check(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost), "D2H");
    return h;
}
// differing words (bitwise) and the reference's / the shape's non-finite values
struct Diff {
    size_t differ = 0, nonfinite = 0;
};
Diff compare(const std::vector<float>& a, const std::vector<float>& b) {
    Diff d;
    for (size_t i = 0; i < a.size(); ++i) {
        d.differ += std::memcmp(&a[i], &b[i], 4) != 0;
        d.nonfinite += !std::isfinite(a[i]) || !std::isfinite(b[i]);
    }
    return d;
}

// the median of 7 rounds of `calls` launches; launch(seq) gets one running sequence number over the warm-up and every
// round (a rotation by seq % n never restarts, so a round of n calls reads all n copies)
double time_us(hipStream_t s, int calls, const std::function<void(int)>& launch) {
    hipEvent_t a, b;
    check(hipEventCreate(&a), "event");
    check(hipEventCreate(&b), "event");
    int seq = 0;
    for (int i = 0; i < calls; ++i) launch(seq++);
    check(hipStreamSynchronize(s), "warmup");
    std::vector<double> v;
    for (int rnd = 0; rnd < 7; ++rnd) {
        check(hipEventRecord(a, s), "record");
        for (int i = 0; i < calls; ++i) launch(seq++);
        check(hipEventRecord(b, s), "record");
        check(hipEventSynchronize(b), "sync");
        float ms = 0;
        check(hipEventElapsedTime(&ms, a, b), "elapsed");
        v.push_back(ms * 1000.0 / calls);
    }
    std::sort(v.begin(), v.end());
    check(hipEventDestroy(a), "event");
    check(hipEventDestroy(b), "event");
    return v[v.size() / 2];
}

std::vector<std::string> dsa_shapes(bool sweep) {
    if (!sweep) return {"rl2b4", "rl1b4", "sl2b4", "sx2b4"};
    std::vector<std::string> v;
    for (char q : {'r', 's'})
        for (char l : {'l', 'x'})
            for (char p : {'1', '2'})
                for (int b : {2, 4, 8, 16}) v.push_back(std::string(1, q) + l + p + "b" + std::to_string(b));
    return v;
}
std::vector<std::string> mla_shapes(bool sweep) {
    if (!sweep) return {"h64lvc", "h32lvc", "h16lvc", "h16xeo"};
    std::vector<std::string> v;
    for (const char* h : {"16", "32", "64"})
        for (char l : {'l', 'x'})
            for (char st : {'v', 'e'})
                for (char c : {'c', 'o'}) v.push_back(std::string("h") + h + l + st + c);
    return v;
}

// the butterflies: every lane's sum and max, across the lanes against the shuffles, over 4096 warps of mixed values
// (normal, wide exponents, signed zeros, subnormals)
bool reduce_parity(hipStream_t s) {
    const int warps = 4096, n = warps * 32;
    std::mt19937 rng(0x7ed5);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> ex(-60, 60), kind(0, 15);
    std::vector<float> in(n);
    for (float& f : in) {
        const int k = kind(rng);
        f = k == 0 ? 0.0f : k == 1 ? -0.0f : k == 2 ? std::ldexp(nd(rng), -140) : std::ldexp(nd(rng), k < 8 ? 0 : ex(rng));
    }
    float* d_in = dev_copy(in);
    float *a, *b, *c, *d;
    check(hipMalloc(&a, n * 4), "hipMalloc");
    check(hipMalloc(&b, n * 4), "hipMalloc");
    check(hipMalloc(&c, n * 4), "hipMalloc");
    check(hipMalloc(&d, n * 4), "hipMalloc");
    hb::warp_reduce_check(d_in, a, b, c, d, warps, s);
    check(hipStreamSynchronize(s), "reduce");
    const auto sa = host_copy(a, n), sb = host_copy(b, n), ma = host_copy(c, n), mb = host_copy(d, n);
    const Diff ds = compare(sa, sb), dm = compare(ma, mb);
    // (finite inputs: a non-finite sum or max fails the row even where both butterflies agree on it)
    const bool ok = ds.differ == 0 && dm.differ == 0 && ds.nonfinite == 0 && dm.nonfinite == 0;
    std::printf("parity reduce warps=%d outputs=%d sum_differ=%zu max_differ=%zu nonfinite=%zu %s\n", warps, n, ds.differ,
                dm.differ, ds.nonfinite + dm.nonfinite, ok ? "PASS" : "FAIL");
    for (float* p : {d_in, a, b, c, d}) check(hipFree(p), "free");
    return ok;
}

// ---- the indexer's scores: one point's inputs, the original's scores and cells as the reference
struct DsaPoint {
    int ctx, n_vis, top, n_sel;
    std::vector<float*> pooled;   // copies (timing cycles them; parity reads copy 0)
    float *iq, *iw, *score, *ref;
    int *cells, *ref_cells;
};
DsaPoint dsa_point(int ctx, int n_pools_override, uint32_t seed, int copies) {
    DsaPoint P;
    P.ctx = ctx;
    P.n_vis = n_pools_override > 0 ? n_pools_override : (ctx + 1) / kPool;
    P.top = std::min(kTop, P.n_vis);
    P.n_sel = kPool * P.top + kTail;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f), wd(0.f, 0.1f);
    std::vector<float> iq(kH * kK), iw(kH), pk((size_t) P.n_vis * kK);
    for (float& f : iq) f = nd(rng);
    for (float& f : iw) f = wd(rng);
    for (int c = 0; c < copies; ++c) {
        for (float& f : pk) f = nd(rng);
        if (c == 0 && P.n_vis > 64)   // exact ties: pools 1..8 copy pool 0 (the tie rule: the lower index first)
            for (int t = 1; t <= 8; ++t) std::copy(pk.begin(), pk.begin() + kK, pk.begin() + (size_t) t * kK);
        P.pooled.push_back(dev_copy(pk));
    }
    P.iq = dev_copy(iq);
    P.iw = dev_copy(iw);
    check(hipMalloc(&P.score, (size_t) P.n_vis * 4), "hipMalloc");
    check(hipMalloc(&P.ref, (size_t) P.n_vis * 4), "hipMalloc");
    check(hipMalloc(&P.cells, (size_t) P.n_sel * 4), "hipMalloc");
    check(hipMalloc(&P.ref_cells, (size_t) P.n_sel * 4), "hipMalloc");
    return P;
}
void dsa_free(DsaPoint& P) {
    for (float* p : P.pooled) check(hipFree(p), "free");
    for (void* p : {(void*) P.iq, (void*) P.iw, (void*) P.score, (void*) P.ref, (void*) P.cells, (void*) P.ref_cells})
        check(hipFree(p), "free");
}
void dsa_run(const DsaPoint& P, int copy, float* score, int* cells, hipStream_t s) {
    gf::dsa_score(P.iq, P.pooled[copy], P.iw, kK, kH, P.n_vis, score, s);
    if (cells) gf::dsa_select(score, P.n_vis, kPool, P.top, P.n_sel, P.ctx, cells, s);
}
bool dsa_parity(DsaPoint& P, const std::vector<std::string>& shapes, uint32_t seed, std::vector<bool>& exact, hipStream_t s) {
    hb::dsa_score_config("original");
    dsa_run(P, 0, P.ref, P.ref_cells, s);
    check(hipStreamSynchronize(s), "ref");
    const auto ref = host_copy(P.ref, P.n_vis);
    const auto ref_cells = host_copy(P.ref_cells, P.n_sel);
    bool ok = true;
    for (size_t k = 0; k < shapes.size(); ++k) {
        const std::string& sh = shapes[k];
        if (!hb::dsa_score_config(sh.c_str())) { std::fprintf(stderr, "no shape %s\n", sh.c_str()); return false; }
        check(hipMemset(P.score, 0xff, (size_t) P.n_vis * 4), "memset");
        check(hipMemset(P.cells, 0x7f, (size_t) P.n_sel * 4), "memset");
        dsa_run(P, 0, P.score, P.cells, s);
        check(hipStreamSynchronize(s), "shape");
        const Diff d = compare(ref, host_copy(P.score, P.n_vis));
        const auto cells = host_copy(P.cells, P.n_sel);
        size_t cd = 0;
        for (int i = 0; i < P.n_sel; ++i) cd += cells[i] != ref_cells[i];
        const bool pass = d.differ == 0 && d.nonfinite == 0 && cd == 0;
        ok &= pass;
        exact[k] = exact[k] && pass;
        std::printf("parity dsa_score ctx=%d n_vis=%d seed=%u %-7s outputs=%d differ=%zu nonfinite=%zu cells=%d cells_differ=%zu %s\n",
                    P.ctx, P.n_vis, seed, sh.c_str(), P.n_vis, d.differ, d.nonfinite, P.n_sel, cd, pass ? "PASS" : "FAIL");
    }
    hb::dsa_score_config(nullptr);
    return ok;
}
void dsa_timing(DsaPoint& P, const std::vector<std::string>& shapes, const std::vector<bool>& exact, hipStream_t s) {
    const int copies = (int) P.pooled.size();
    hb::dsa_score_config("original");
    const double t_ref = time_us(s, copies, [&](int i) { dsa_run(P, i % copies, P.ref, nullptr, s); });
    const double bytes = (double) P.n_vis * kK * 4;
    for (size_t k = 0; k < shapes.size(); ++k) {
        hb::dsa_score_config(shapes[k].c_str());
        const double t = time_us(s, copies, [&](int i) { dsa_run(P, i % copies, P.score, nullptr, s); });
        std::printf("bench dsa_score ctx=%d n_vis=%d %-7s %8.1f us %6.1f GB/s | original %8.1f us %6.1f GB/s | %.3fx %s\n",
                    P.ctx, P.n_vis, shapes[k].c_str(), t, bytes / t / 1000.0, t_ref, bytes / t_ref / 1000.0, t_ref / t,
                    exact[k] ? "bit-exact" : "DIFFERS");
    }
    hb::dsa_score_config("original");
    const double t_sel = time_us(s, copies, [&](int i) {
        (void) i;
        gf::dsa_select(P.ref, P.n_vis, kPool, P.top, P.n_sel, P.ctx, P.ref_cells, s);
    });
    std::printf("bench dsa_select ctx=%d n_vis=%d original %8.1f us\n", P.ctx, P.n_vis, t_sel);
    hb::dsa_score_config(nullptr);
}

// ---- MLA's split attention: q_abs, FP16 latent rows and the selected cells (a pattern of masked ones)
struct MlaInputs {
    float *q_abs, *ctx, *ref;
    uint16_t* lat;
    std::vector<int*> cells;   // disjoint sets (timing cycles them; parity reads set 0)
    size_t touched;            // the latent bytes one pass over every set reads (masked cells read nothing)
};
uint16_t half_bits(float f) { return std::bit_cast<uint16_t>(__float2half(f)); }
// pattern: "all" every cell a row; "tail" the select's last 3 positions masked (-1, a context that ends mid-pool);
// "chunk" also a whole 32-cell chunk masked (its max -inf, its weights 0, the merge skipping it)
// set k of a pool's rows: the k-th kSel rows of one shuffled permutation (the sets are disjoint)
std::vector<int> cell_set(const std::vector<int>& perm, int k, const char* pattern) {
    std::vector<int> c(perm.begin() + (size_t) k * kSel, perm.begin() + (size_t) (k + 1) * kSel);
    if (std::strcmp(pattern, "all") != 0)
        for (int m = 0; m < kTail; ++m) c[kSel - 1 - m] = -1;
    if (std::strcmp(pattern, "chunk") == 0)
        for (int i = 32 * 5; i < 32 * 6; ++i) c[i] = -1;
    return c;
}
MlaInputs mla_inputs(const char* pattern, int sets, uint32_t seed, int n_rows) {
    MlaInputs M;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> q((size_t) kHeads * kLora);
    for (float& f : q) f = nd(rng);
    std::vector<uint16_t> lat((size_t) n_rows * kLora);
    for (uint16_t& h : lat) h = half_bits(nd(rng));
    M.q_abs = dev_copy(q);
    M.lat = dev_copy(lat);
    if ((size_t) sets * kSel > (size_t) n_rows) {
        std::fprintf(stderr, "%d disjoint sets of %d rows need more than %d rows\n", sets, kSel, n_rows);
        std::exit(2);
    }
    std::vector<int> perm(n_rows);
    for (int i = 0; i < n_rows; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    M.touched = 0;
    for (int k = 0; k < sets; ++k) {
        const std::vector<int> c = cell_set(perm, k, pattern);
        for (int x : c) M.touched += x >= 0 ? (size_t) kLora * sizeof(uint16_t) : 0;
        M.cells.push_back(dev_copy(c));
    }
    check(hipMalloc(&M.ctx, (size_t) kHeads * kLora * 4), "hipMalloc");
    check(hipMalloc(&M.ref, (size_t) kHeads * kLora * 4), "hipMalloc");
    return M;
}
void mla_free(MlaInputs& M) {
    for (int* p : M.cells) check(hipFree(p), "free");
    for (void* p : {(void*) M.q_abs, (void*) M.ctx, (void*) M.ref, (void*) M.lat}) check(hipFree(p), "free");
}
bool mla_run(const MlaInputs& M, int set, float* ctx, hipStream_t s) {
    return gf::mla_attend(M.q_abs, M.lat, M.cells[set], kSel, kHeads, kNope, kLora, ctx, s, false);
}
bool mla_parity(MlaInputs& M, const char* pattern, const std::vector<std::string>& shapes, std::vector<bool>& exact, hipStream_t s) {
    hb::mla_attn_config("original");
    if (!mla_run(M, 0, M.ref, s)) { std::fprintf(stderr, "mla_attend not available\n"); return false; }
    check(hipStreamSynchronize(s), "ref");
    const auto ref = host_copy(M.ref, (size_t) kHeads * kLora);
    bool ok = true;
    for (size_t k = 0; k < shapes.size(); ++k) {
        const std::string& sh = shapes[k];
        if (!hb::mla_attn_config(sh.c_str())) { std::fprintf(stderr, "no shape %s\n", sh.c_str()); return false; }
        check(hipMemset(M.ctx, 0xff, (size_t) kHeads * kLora * 4), "memset");
        mla_run(M, 0, M.ctx, s);
        check(hipStreamSynchronize(s), "shape");
        const Diff d = compare(ref, host_copy(M.ctx, (size_t) kHeads * kLora));
        const bool pass = d.differ == 0 && d.nonfinite == 0;
        ok &= pass;
        exact[k] = exact[k] && pass;
        std::printf("parity mla_attend n_sel=%d cells=%-5s %-7s outputs=%d differ=%zu nonfinite=%zu %s\n", kSel, pattern,
                    sh.c_str(), kHeads * kLora, d.differ, d.nonfinite, pass ? "PASS" : "FAIL");
    }
    hb::mla_attn_config(nullptr);
    return ok;
}
void mla_timing(MlaInputs& M, const std::vector<std::string>& shapes, const std::vector<bool>& exact, hipStream_t s) {
    const int sets = (int) M.cells.size();
    hb::mla_attn_config("original");
    const double t_ref = time_us(s, sets, [&](int i) { mla_run(M, i % sets, M.ref, s); });
    for (size_t k = 0; k < shapes.size(); ++k) {
        hb::mla_attn_config(shapes[k].c_str());
        const double t = time_us(s, sets, [&](int i) { mla_run(M, i % sets, M.ctx, s); });
        std::printf("bench mla_attend n_sel=%d %-7s %8.1f us | original %8.1f us | %.3fx %s\n", kSel, shapes[k].c_str(), t,
                    t_ref, t_ref / t, exact[k] ? "bit-exact" : "DIFFERS");
    }
    hb::mla_attn_config(nullptr);
}
}  // namespace

int main(int argc, char** argv) {
    // whole lines: the engine's records go to stderr (unbuffered) while these rows go to stdout - block-buffered into a
    // file, a record landed inside a row (0400Z raw line 354); line-buffered, each row is one write between them
#if defined(_WIN32)
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // (MSVC's CRT has no line buffering, and _IOLBF with size 0 aborts)
#else
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
#endif
    bool parity_only = false, sweep = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--parity-only") parity_only = true;
        else if (a == "--sweep") sweep = true;
        else {
            std::fprintf(stderr, "usage: %s [--parity-only] [--sweep]\n", argv[0]);
            return 2;
        }
    }
    check(hipSetDevice(0), "GPU 0");
    hipDeviceProp_t prop;
    check(hipGetDeviceProperties(&prop, 0), "device properties");
    std::printf("GPU 0: %s (%s), %d CUs\n", prop.name, prop.gcnArchName, prop.multiProcessorCount);
    hipStream_t s;
    check(hipStreamCreate(&s), "stream");
    const int lanes = hb::lanes_kind(s);
    std::printf("lanes: %s\n", lanes == 1 ? "dpp" : lanes == 0 ? "shuffles" : "unknown");
    const std::vector<std::string> ds = dsa_shapes(sweep), ms = mla_shapes(sweep);
    bool ok = lanes >= 0 && reduce_parity(s);
    // parity: the three points with two seeds each, then pool counts that end a warp's or a block's run early
    std::vector<bool> d_exact(ds.size(), true), m_exact(ms.size(), true);
    for (int ctx : kPoints)
        for (uint32_t seed : {1u, 2u}) {
            DsaPoint P = dsa_point(ctx, 0, seed * 7919u + (uint32_t) ctx, 1);
            ok &= dsa_parity(P, ds, seed, d_exact, s);
            dsa_free(P);
        }
    for (int n : {1, 7, 9, 513, 1001}) {
        DsaPoint P = dsa_point(4 * n, n, 31u + (uint32_t) n, 1);
        ok &= dsa_parity(P, ds, 31, d_exact, s);
        dsa_free(P);
    }
    for (const char* pattern : {"all", "tail", "chunk"}) {
        MlaInputs M = mla_inputs(pattern, 1, 0x3a11u + (uint32_t) std::strlen(pattern), kRows);
        ok &= mla_parity(M, pattern, ms, m_exact, s);
        mla_free(M);
    }
    if (!parity_only) {
        // timed after every parity row; a shape with any differing or non-finite row is printed DIFFERS (and the run FAILs)
        for (int ctx : kPoints) {
            const size_t bytes = (size_t) ((ctx + 1) / kPool) * kK * 4;
            const int copies = (int) std::min<size_t>(128, ((size_t) 256 << 20) / bytes + 1);   // >= 256 MB a round (header)
            DsaPoint P = dsa_point(ctx, 0, 0xc0ffeeu + (uint32_t) ctx, copies);
            std::printf("rotation dsa_score ctx=%d copies=%d bytes=%zu\n", ctx, copies, (size_t) copies * bytes);
            dsa_timing(P, ds, d_exact, s);
            dsa_free(P);
        }
        MlaInputs M = mla_inputs("tail", 128, 0x77aau, kRowsCold);
        std::printf("rotation mla_attend sets=128 rows=%d touched=%zu\n", kRowsCold, M.touched);
        mla_timing(M, ms, m_exact, s);
        mla_free(M);
    }
    check(hipStreamDestroy(s), "stream");
    std::printf("glm_attn_bench: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
