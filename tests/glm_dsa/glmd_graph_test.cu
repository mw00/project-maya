// tests/glm_dsa/glmd_graph_test.cu - glm-dsa B10.1 / B14.4 (label gpu): one captured glm-dsa decode part replayed for
// several tokens equals the same launches issued directly, and every cache is written at the token's position.
//
// The part has two DSA layers (a full-indexer layer and a shared one reusing its selection), GLM-5.3 sizes:
//   dsa_prep (p_dev) -> q / iq copies (memcpy nodes) -> dsa_rope_q (p_dev) -> idx_score_simple (pos = &dev->p,
//   n_kv = n_cap) -> topk (pos, n_cap) -> mla2 (n_sel = 2048, -1 padding); the shared layer: dsa_prep, rope, mla2.
// The caches are pre-filled at positions 0..2999.  The part is captured at p = 3000 and replayed at 3001 and 3002
// (same key), then the key's n_cap changes at p = 4096 and the part is re-captured.  Checked after every token, against
// a twin set of buffers driven by the same calls without a graph: the latent, k_pe and index-key caches (bit
// identical, the rows past p untouched), the selection and both layers' outputs (bit identical); the capture count,
// replays, invalidations; the launches per token (graph nodes vs 2 = set_dev_pos + the graph launch).
// Needs a GPU; exits 77 without one.
#include "strata/kernels/glmd_fast.hpp"
#include "strata/kernels/glmd_graph.hpp"
#include "strata/kernels/glmd_ref.hpp"
#include "strata/kernels/glmd_volta.hpp"

#include "../../src/core/glm_dsa_graphs.hpp"
#include "../../src/kernels/cuda/glmd_rope.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

int g_fail = 0;
#define CK(x)                                                                                       \
    do {                                                                                            \
        cudaError_t e_ = (x);                                                                       \
        if (e_ != cudaSuccess) {                                                                    \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(1);                                                                           \
        }                                                                                           \
    } while (0)
void expect(bool ok, const char* what) {
    std::printf("%-72s %s\n", what, ok ? "ok" : "FAIL");
    g_fail += !ok;
}

constexpr int kQLora = 2048, kKv = 512, kRope = 64, kIdx = 128, kIdxH = 32, kH = 64, kNope = 192, kV = 256;
constexpr int kMaxCtx = 8192, kTop = 2048;

template <typename T>
T* dalloc(size_t n) {
    T* p;
    CK(cudaMalloc(&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}
template <typename T>
T* dcopy(const std::vector<T>& h) {
    T* p = dalloc<T>(h.size());
    CK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
template <typename T>
std::vector<T> get(const T* p, size_t n) {
    std::vector<T> h(n);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}
std::vector<float> randv(std::mt19937& rng, size_t n, float s = 1.0f) {
    std::normal_distribution<float> d(0.0f, s);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}
uint16_t bf16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    x += 0x7fffu + ((x >> 16) & 1u);
    return (uint16_t) (x >> 16);
}

// one set of the part's buffers (the graph's and the direct run's are separate)
struct Part {
    uint16_t *lat[2], *kpe[2];
    float* ik;
    float *qr, *q[2], *iq, *score, *out[2], *work;
    void* qq;
    int* sel;
    size_t work_bytes;
};
struct Inputs {   // shared, read-only
    float *qr_raw, *one, *zero, *kv_raw[2], *ik_raw, *q_src[2], *iq_src, *w, *inv;
    uint16_t *wk_b, *wv_b;
};

Part make_part() {
    Part P;
    for (int l = 0; l < 2; ++l) {
        P.lat[l] = dalloc<uint16_t>((size_t) kMaxCtx * kKv);
        P.kpe[l] = dalloc<uint16_t>((size_t) kMaxCtx * kRope);
        P.q[l] = dalloc<float>((size_t) kH * (kNope + kRope));
        P.out[l] = dalloc<float>((size_t) kH * kV);
    }
    P.ik = dalloc<float>((size_t) kMaxCtx * kIdx);
    P.qr = dalloc<float>(kQLora);
    P.qq = dalloc<glmd::Q81>(kQLora / 32);
    P.iq = dalloc<float>((size_t) kIdxH * kIdx);
    P.score = dalloc<float>(kMaxCtx);
    P.sel = dalloc<int>(kTop);
    P.work_bytes = glmd::mla2_work_bytes(kH, kKv, kTop);
    P.work = (float*) dalloc<uint8_t>(P.work_bytes);
    return P;
}

glmd::DsaPrep prep(const Part& P, const Inputs& I, int l, int p, const int* p_dev) {
    glmd::DsaPrep a;
    a.qr_raw = I.qr_raw; a.q_a_norm = I.one; a.qr = P.qr; a.qr_q = P.qq; a.q_lora = kQLora;
    a.kv_raw = I.kv_raw[l]; a.kv_norm = I.one; a.lat = P.lat[l]; a.kv_lora = kKv; a.k_pe = P.kpe[l]; a.qk_rope = kRope;
    a.inv_freq = I.inv;
    if (l == 0) {
        a.ik_raw = I.ik_raw; a.ik_ln_w = I.one; a.ik_ln_b = I.zero; a.ik_cache = P.ik; a.idx_dim = kIdx;
        a.idx_rope = 64; a.idx_inv_freq = I.inv;
    }
    a.p = p;
    a.p_dev = p_dev;
    return a;
}

// the part's launches for one token (identical in the captured and the direct run)
bool part_body(const Part& P, const Inputs& I, const glmd::DevPos* dev, int n_cap, cudaStream_t s) {
    const int* pd = &dev->p;
    for (int l = 0; l < 2; ++l) {
        if (!glmd::dsa_prep(prep(P, I, l, -1, pd), s)) return false;
        CK(cudaMemcpyAsync(P.q[l], I.q_src[l], sizeof(float) * kH * (kNope + kRope), cudaMemcpyDeviceToDevice, s));
        if (l == 0) CK(cudaMemcpyAsync(P.iq, I.iq_src, sizeof(float) * kIdxH * kIdx, cudaMemcpyDeviceToDevice, s));
        glmd::DsaRopeQ r;
        r.q = P.q[l]; r.n_head = kH; r.q_stride = kNope + kRope; r.rot_off = kNope; r.n_rot = kRope; r.inv_freq = I.inv;
        if (l == 0) {
            r.iq = P.iq; r.idx_heads = kIdxH; r.idx_dim = kIdx; r.idx_rope = 64; r.idx_inv_freq = I.inv;
        }
        r.p = -1;
        r.p_dev = pd;
        if (!glmd::dsa_rope_q(r, s)) return false;
        if (l == 0) {
            glmd::IdxScoreWmma a;
            a.iq = P.iq; a.w = I.w; a.keys = P.ik; a.idx_heads = kIdxH; a.idx_dim = kIdx; a.n_kv = n_cap; a.T = 1;
            a.pos = pd; a.score = P.score; a.ld = n_cap;
            if (!glmd::idx_score_simple(a, s)) return false;
            if (!glmd::topk(P.score, n_cap, n_cap, 1, pd, kTop, P.sel, s)) return false;
        }
        glmd::Mla2 m;
        m.q = P.q[l]; m.q_stride = kNope + kRope; m.wk_b = I.wk_b; m.wv_b = I.wv_b; m.lat = P.lat[l]; m.k_pe = P.kpe[l];
        m.cells = P.sel; m.n_sel = kTop; m.scale = glmd::kGlm53KqScale; m.work = P.work; m.work_bytes = P.work_bytes;
        m.out = P.out[l];
        if (!glmd::mla2(m, s)) return false;
    }
    return true;
}

bool same_u16(const uint16_t* a, const uint16_t* b, size_t n) {
    return get(a, n) == get(b, n);
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("glmd_graph_test: no CUDA device, skipped\n");
        return 77;
    }
    std::mt19937 rng(14);
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    std::vector<float> inv(kRope / 2);
    glmd::rope_inv_freq(inv.data(), kRope, 8e6);
    Inputs I;
    I.inv = dcopy(inv);
    I.qr_raw = dcopy(randv(rng, kQLora));
    I.one = dcopy(std::vector<float>(kQLora, 1.0f));
    I.zero = dcopy(std::vector<float>(kIdx, 0.0f));
    for (int l = 0; l < 2; ++l) {
        I.kv_raw[l] = dcopy(randv(rng, kKv + kRope));
        I.q_src[l] = dcopy(randv(rng, (size_t) kH * (kNope + kRope)));
    }
    I.ik_raw = dcopy(randv(rng, kIdx));
    I.iq_src = dcopy(randv(rng, (size_t) kIdxH * kIdx));
    I.w = dcopy(randv(rng, kIdxH, 0.02f));
    {
        std::vector<uint16_t> wk((size_t) kH * kKv * kNope), wv((size_t) kH * kV * kKv);
        std::normal_distribution<float> d(0.0f, 0.05f);
        for (auto& v : wk) v = bf16(d(rng));
        for (auto& v : wv) v = bf16(d(rng));
        I.wk_b = dcopy(wk);
        I.wv_b = dcopy(wv);
    }
    Part G = make_part(), D = make_part();
    glmd::DevPos* dev_g = dalloc<glmd::DevPos>(1);
    glmd::DevPos* dev_d = dalloc<glmd::DevPos>(1);
    // pre-fill positions 0..2999 of both sets with distinct keys / latents (random raw inputs per position).  Every
    // position's raw rows get their own device row, uploaded and synchronised before the first dsa_prep: s is
    // non-blocking (as the engine's streams are) and never waits for the legacy stream, so a single scratch row
    // refilled per position by a synchronous cudaMemcpy raced the dsa_prep still queued on s, both ways (the next
    // copy overwrote it under them, and a pageable copy returns before its DMA lands), and G and D cached different
    // rows (the test PC's first GPU run, docs/strata/volta3/ctest-gpu-2026-10-17.txt).
    {
        constexpr int kFill = 3000, kRow = kKv + kRope;
        std::vector<float> kv_h, ik_h;
        kv_h.reserve((size_t) kFill * kRow);
        ik_h.reserve((size_t) kFill * kIdx);
        for (int p = 0; p < kFill; ++p) {
            const auto kv = randv(rng, kRow), ik = randv(rng, kIdx);
            kv_h.insert(kv_h.end(), kv.begin(), kv.end());
            ik_h.insert(ik_h.end(), ik.begin(), ik.end());
        }
        float* kv_all = dcopy(kv_h);
        float* ik_all = dcopy(ik_h);
        CK(cudaDeviceSynchronize());   // the uploads and every buffer's zero fill (legacy stream) land before s runs
        for (int p = 0; p < kFill; ++p)
            for (Part* P : {&G, &D})
                for (int l = 0; l < 2; ++l) {
                    glmd::DsaPrep a = prep(*P, I, l, p, nullptr);
                    a.kv_raw = kv_all + (size_t) p * kRow;
                    if (l == 0) a.ik_raw = ik_all + (size_t) p * kIdx;
                    glmd::dsa_prep(a, s);
                }
        CK(cudaStreamSynchronize(s));
        cudaFree(kv_all);
        cudaFree(ik_all);
    }
    strata::core::glmdsa::DsaGraphs graphs;
    uint64_t expected = 0;
    int dummy_arena = 0;
    for (int p : {3000, 3001, 3002, 4096, 4097}) {
        const int n_cap = (int) glmd::graph_n_cap(p, kMaxCtx);
        glmd::GraphKey key;
        key.part = 0;
        key.l0 = 0;
        key.l1 = 2;
        key.arena = &dummy_arena;
        key.state = G.lat[0];
        key.bits = glmd::graph_kernel_bits(false, false, false, false, false, false, false);
        key.regime = glmd::graph_regime(p, kTop, false, glmd::kIdxWmmaMinDecode, false, glmd::kTopkMbMin);
        key.n_cap = n_cap;
        expect(glmd::set_dev_pos(dev_g, p, kTop, s), "set_dev_pos (graph)");
        bool replayed = false;
        std::string err;
        const bool ok = graphs.enqueue(key, s, expected, 0, [&] { return part_body(G, I, dev_g, n_cap, s); }, replayed, err);
        if (!ok) std::printf("enqueue: %s\n", err.c_str());
        expect(ok, "graph enqueue");
        expect(glmd::set_dev_pos(dev_d, p, kTop, s), "set_dev_pos (direct)");
        expect(part_body(D, I, dev_d, n_cap, s), "direct launches");
        CK(cudaStreamSynchronize(s));
        char what[96];
        std::snprintf(what, sizeof what, "p %d (%s): caches equal the direct run (all rows)", p, replayed ? "replay" : "capture");
        bool caches = true;
        for (int l = 0; l < 2; ++l) {
            caches &= same_u16(G.lat[l], D.lat[l], (size_t) kMaxCtx * kKv);
            caches &= same_u16(G.kpe[l], D.kpe[l], (size_t) kMaxCtx * kRope);
        }
        caches &= get(G.ik, (size_t) kMaxCtx * kIdx) == get(D.ik, (size_t) kMaxCtx * kIdx);
        expect(caches, what);
        // the rows written this token are at p, and nothing past p
        const auto lat = get(G.lat[1], (size_t) kMaxCtx * kKv);
        bool at_p = false, past = true;
        for (int c = 0; c < kKv; ++c) at_p |= lat[(size_t) p * kKv + c] != 0;
        for (size_t i = (size_t) (p + 1) * kKv; i < lat.size(); ++i) past &= lat[i] == 0;
        std::snprintf(what, sizeof what, "p %d: latent row p written, rows past p untouched", p);
        expect(at_p && past, what);
        std::snprintf(what, sizeof what, "p %d: selection and both layer outputs bit identical", p);
        expect(get(G.sel, kTop) == get(D.sel, kTop) && get(G.out[0], (size_t) kH * kV) == get(D.out[0], (size_t) kH * kV) &&
                   get(G.out[1], (size_t) kH * kV) == get(D.out[1], (size_t) kH * kV),
               what);
    }
    std::printf("  captures %llu, replays %llu, invalidations %llu, graph nodes %zu (launches per token: 2 instead of %zu)\n",
                (unsigned long long) graphs.captures, (unsigned long long) graphs.replays,
                (unsigned long long) graphs.invalidations, graphs.nodes(0), graphs.nodes(0) + 1);
    expect(graphs.captures == 2 && graphs.replays == 3 && graphs.invalidations == 1,
           "2 captures (p 3000, n_cap 8192 at 4096), 3 replays, 1 invalidation");
    expect(glmd::launch_errors() == 0 && glmd::volta_launch_errors() == 0, "no launch errors");
    graphs.clear();
    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "PASSED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
