// tests/ik2_hip_fuse_test.cu - the window step's fused launches (ik2_hip_kernels.cuh: mix_rms_rows_fused, route_publish,
// kv_clip_rows) against the launches each replaces, bit for bit, at Inkling 2 Small's shapes on random inputs; and
// each form's time a call (the launches back to back on one stream, as in the step).  No model needed.
//   ik2_hip_fuse_test [reps]
#include "ik2_hip_kernels.cuh"
#include "hy4_kernels.cuh"
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)
namespace k = ik2hip::k;
namespace hk = hy4::k;
static std::mt19937 rng(7);
template <typename T> T* dev(size_t n) { T* p; CK(cudaMalloc(&p, n * sizeof(T))); CK(cudaMemset(p, 0, n * sizeof(T))); return p; }
static float* rnd(size_t n, float sc, float off = 0.0f) {
    std::normal_distribution<float> nd(0.0f, 1.0f); std::vector<float> h(n); for (auto& v : h) v = off + sc * nd(rng);
    float* p = dev<float>(n); CK(cudaMemcpy(p, h.data(), n * 4, cudaMemcpyHostToDevice)); return p;
}
template <typename T> static std::vector<T> get(const T* p, size_t n) { std::vector<T> h(n); CK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDefault)); return h; }
static void cp(void* d, const void* s, size_t b) { CK(cudaMemcpy(d, s, b, cudaMemcpyDefault)); }
static size_t ndiff(const void* a, const void* b, size_t bytes) { size_t n = 0; for (size_t i = 0; i < bytes; ++i) n += ((const unsigned char*) a)[i] != ((const unsigned char*) b)[i]; return n; }
int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 300;
    const int E = 4608, W = 16, NB = 16, NE = 384, NU = 8, H = 48, Hkv = 8, D = 128, NR = 16, RING = 1024;
    cudaStream_t s; CK(cudaStreamCreate(&s)); cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto timeit = [&](auto&& f) { for (int i = 0; i < 20; ++i) f(); CK(cudaEventRecord(e0, s)); for (int i = 0; i < reps; ++i) f(); CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); return 1e3f * ms / reps; };
    int bad = 0;
    // ---- 1. conv_add_rows + mix_rms_rows + q8_rows (+ dup_rows_q8) against mix_rms_rows_fused
    {
        float* bl0 = rnd((size_t) (NB + 1) * W * E, 1.0f); float* cur0 = rnd((size_t) W * E, 1.0f);
        float* y = rnd((size_t) W * E, 0.5f); float* h0 = rnd(3 * (size_t) E, 0.5f); float* cw = rnd(4 * (size_t) E, 0.3f);
        float* qy = rnd(3 * (size_t) E, 0.02f); float* g = rnd(3 * (size_t) E, 0.1f, 1.0f);
        float *bl[2], *cur[2], *h1[2], *out[2], *d[2]; int8_t* q[2];
        for (int f = 0; f < 2; ++f) { bl[f] = dev<float>((size_t) (NB + 1) * W * E); cur[f] = dev<float>((size_t) W * E); h1[f] = dev<float>(3 * (size_t) E);
            out[f] = dev<float>(3 * (size_t) W * E); q[f] = dev<int8_t>(3 * (size_t) W * E); d[f] = dev<float>(3 * (size_t) W * E / 32); }
        int8_t* xq = dev<int8_t>(3 * (size_t) W * E); float* xd = dev<float>(3 * (size_t) W * E / 32);
        std::printf("mix: %3s %3s %3s %4s %4s | %8s %8s | %s\n", "nb", "nq", "T", "conv", "dup", "old us", "new us", "bytes differing (out cur blk h1 q d)");
        for (int nb : {1, 4, 8, 15}) for (int nq : {1, 3}) for (int T : {1, 2, 4, 8}) for (int cv = 0; cv < 3; ++cv) for (int dup = 0; dup < 2; ++dup) {
            if (nb == 15 && cv == 2) continue;
            if (dup && (nq != 1 || T == 1)) continue;
            if (cv == 0 && dup) continue;
            const size_t ldb = (size_t) T * E, TE = ldb;
            const bool withblk = cv == 2; const int nbm = nb + (withblk ? 1 : 0);   // (cv 2: a block's first layer - the conv saves the stream)
            auto reset = [&](int f) { cp(bl[f], bl0, (size_t) (NB + 1) * W * E * 4); cp(cur[f], cur0, (size_t) W * E * 4); };
            auto old = [&]() {
                if (cv) k::conv_add_rows(y, T, h0, h1[0], cw, E, cur[0], withblk ? bl[0] + (size_t) nb * ldb : nullptr, s);
                k::mix_rms_rows(bl[0], ldb, nbm, cur[0], T, E, qy, nq, g, g + E, g + 2 * E, 1e-6f, out[0], s);
                if (!dup) for (int qi = 0; qi < nq; ++qi) hk::q8_rows(out[0] + qi * TE, E, T, E, q[0] + qi * TE, d[0] + qi * TE / 32, s);
                else { hk::q8_rows(out[0], E, T, E, xq, xd, s); hk::dup_rows_q8(xq, xd, T, E, q[0], d[0], s); }
            };
            auto nw = [&]() {
                k::mix_rms_rows_fused(bl[1], ldb, nbm, cur[1], T, E, qy, nq, g, g + E, g + 2 * E, 1e-6f, out[1], cv ? y : nullptr, h0, h1[1], cw,
                                      withblk ? bl[1] + (size_t) nb * ldb : nullptr, q[1], d[1], dup != 0, s);
            };
            // (timed on the inputs as they drift - each call adds to the stream; the checked run from fresh copies)
            reset(0); reset(1); const float uo = timeit(old), un = timeit(nw);
            reset(0); reset(1); old(); nw(); CK(cudaStreamSynchronize(s));
            const size_t no = (size_t) nq * TE, nqb = (dup ? 2 : nq) * TE;
            auto a1 = get(out[0], no), b1 = get(out[1], no); auto a2 = get(cur[0], TE), b2 = get(cur[1], TE);
            auto a3 = get(bl[0], (size_t) (NB + 1) * W * E), b3 = get(bl[1], (size_t) (NB + 1) * W * E);
            auto a4 = get(h1[0], 3 * (size_t) E), b4 = get(h1[1], 3 * (size_t) E);
            auto a5 = get(q[0], nqb), b5 = get(q[1], nqb); auto a6 = get(d[0], nqb / 32), b6 = get(d[1], nqb / 32);
            const size_t n1 = ndiff(a1.data(), b1.data(), no * 4), n2 = ndiff(a2.data(), b2.data(), TE * 4), n3 = ndiff(a3.data(), b3.data(), a3.size() * 4),
                         n4 = cv ? ndiff(a4.data(), b4.data(), a4.size() * 4) : 0, n5 = ndiff(a5.data(), b5.data(), nqb), n6 = ndiff(a6.data(), b6.data(), nqb / 32 * 4);
            const size_t tot = n1 + n2 + n3 + n4 + n5 + n6; bad += tot != 0;
            std::printf("     %3d %3d %3d %4d %4d | %8.2f %8.2f | %zu %zu %zu %zu %zu %zu%s\n", nb, nq, T, cv, dup, uo, un, n1, n2, n3, n4, n5, n6, tot ? "  MISMATCH" : "");
        }
    }
    // ---- 2. router + pack3 + publish against route_publish
    {
        float* xf = rnd((size_t) W * E, 1.0f); float* bias = rnd(NE, 0.01f);
        int* ids[2]; float *w[2], *dst[2]; unsigned *flag[2]; unsigned* cnt = dev<unsigned>(4);
        for (int f = 0; f < 2; ++f) { ids[f] = dev<int>(W * 16); w[f] = dev<float>(W * 16);
            CK(cudaHostAlloc((void**) &dst[f], sizeof(float) * (E + 16) * W, cudaHostAllocMapped | cudaHostAllocPortable));
            CK(cudaHostAlloc((void**) &flag[f], 64, cudaHostAllocMapped | cudaHostAllocPortable)); std::memset(dst[f], 0, sizeof(float) * (E + 16) * W); *flag[f] = 0; }
        float* pub = dev<float>((size_t) W * (E + 16));
        std::printf("route: %3s | %8s %8s | %s\n", "T", "old us", "new us", "differing (ids w dst) flags");
        for (int T : {1, 2, 4, 8, 16}) {
            float* lg = rnd((size_t) T * NE, 2.0f);
            const size_t TE = (size_t) T * E, n = TE + 2 * (size_t) T * NU;
            unsigned v = 0;
            auto old = [&]() { ++v; hk::router(lg, T, NE, bias, NU, true, 8.0f, ids[0], w[0], s); hk::pack3(xf, TE, ids[0], (size_t) T * NU, w[0], (size_t) T * NU, pub, s);
                               hk::publish(pub, dst[0], n, cnt, flag[0], v, s); };
            auto nw = [&]() { ++v; k::route_publish(lg, T, NE, bias, NU, true, 8.0f, ids[1], w[1], xf, TE, dst[1], cnt, flag[1], v, s); };
            const float uo = timeit(old), un = timeit(nw);
            std::memset(dst[0], 0, sizeof(float) * n); std::memset(dst[1], 0, sizeof(float) * n);
            old(); const unsigned vo = v; nw(); const unsigned vn = v; CK(cudaStreamSynchronize(s));
            auto a1 = get(ids[0], T * NU), b1 = get(ids[1], T * NU); auto a2 = get(w[0], T * NU), b2 = get(w[1], T * NU);
            const size_t n1 = ndiff(a1.data(), b1.data(), T * NU * 4), n2 = ndiff(a2.data(), b2.data(), T * NU * 4), n3 = ndiff(dst[0], dst[1], n * 4);
            const bool fl = *(volatile unsigned*) flag[0] == vo && *(volatile unsigned*) flag[1] == vn;
            const unsigned c = get(cnt, 1)[0];
            const size_t tot = n1 + n2 + n3 + (fl ? 0 : 1) + (c ? 1 : 0); bad += tot != 0;
            std::printf("       %3d | %8.2f %8.2f | %zu %zu %zu %s%s\n", T, uo, un, n1, n2, n3, fl && !c ? "set" : "WRONG", tot ? "  MISMATCH" : "");
        }
    }
    // ---- 3. kv_finish_rows + clip2_rows against kv_clip_rows
    {
        const int C = Hkv * D;
        float* ink = rnd((size_t) W * C, 1.0f); float* inv = rnd((size_t) W * C, 1.0f); float* hk0 = rnd(3 * (size_t) C, 1.0f); float* hv0 = rnd(3 * (size_t) C, 1.0f);
        float* kc = rnd(4 * (size_t) C, 0.3f); float* vc = rnd(4 * (size_t) C, 0.3f); float* dk = rnd((size_t) W * 4 * Hkv, 0.2f); float* dv = rnd((size_t) W * 4 * Hkv, 0.2f);
        float* q0 = rnd((size_t) W * H * D, 3.0f); float* r0 = rnd((size_t) W * H * NR, 3.0f); float* qs = rnd((size_t) H * D, 0.5f, 1.0f); float* rs = rnd((size_t) H * NR, 0.5f, 1.0f);
        float *hk1[2], *hv1[2], *q[2], *r[2]; uint16_t *K[2], *V[2];
        for (int f = 0; f < 2; ++f) { hk1[f] = dev<float>(3 * C); hv1[f] = dev<float>(3 * C); q[f] = dev<float>((size_t) W * H * D); r[f] = dev<float>((size_t) W * H * NR);
            K[f] = dev<uint16_t>((size_t) RING * C); V[f] = dev<uint16_t>((size_t) RING * C); }
        float tau[16]; for (int t = 0; t < 16; ++t) tau[t] = 1.0f + 0.01f * t;
        float* sw = rnd((size_t) H * E, 0.02f); float* kpw = rnd((size_t) 4 * Hkv * E, 0.02f); float* vpw = rnd((size_t) 4 * Hkv * E, 0.02f);
        float* xs = rnd((size_t) W * E, 1.0f); float* xk = rnd((size_t) W * E, 1.0f); float* xv = rnd((size_t) W * E, 1.0f);
        float* sy[2] = {dev<float>((size_t) W * H), dev<float>((size_t) W * H)}; float* dko[2] = {dev<float>((size_t) W * 32), dev<float>((size_t) W * 32)};
        float* dvo[2] = {dev<float>((size_t) W * 32), dev<float>((size_t) W * 32)};
        std::printf("kv+clip: %3s %4s %4s | %8s %8s | %s\n", "T", "dyn", "gemv", "old us", "new us", "bytes differing (K V hk hv q rel sink dk dv)");
        for (int T : {1, 2, 4, 8}) for (int dyn = 0; dyn < 2; ++dyn) for (int gm = 0; gm < 2; ++gm) {
            const long long p0 = 1020;   // (the ring wraps inside the window)
            auto reset = [&](int f) { cp(q[f], q0, (size_t) W * H * D * 4); cp(r[f], r0, (size_t) W * H * NR * 4); };
            const bool pj = gm && dyn;   // (gemv: the sinks' GEMV in the launch; with dyn, the conv projections' too)
            float* dkk = pj ? dko[0] : dk; float* dvv = pj ? dvo[0] : dv;
            auto old = [&]() { if (gm) hk::gemv_f32w(sw, E, E, H, xs, E, T, sy[0], H, 1.0f, s);
                               if (pj) { hk::gemv_f32w(kpw, E, E, 32, xk, E, T, dko[0], 32, 1.0f, s); hk::gemv_f32w(vpw, E, E, 32, xv, E, T, dvo[0], 32, 1.0f, s); }
                               k::kv_finish_rows(ink, inv, T, hk0, hv0, hk1[0], hv1[0], kc, vc, dyn ? dkk : nullptr, dyn ? dvv : nullptr, Hkv, D, 22.6f, 1e-6f, K[0], V[0], RING, p0, s);
                               k::clip2_rows(q[0], T, H, D, qs, 22.6f, r[0], NR, rs, 8.0f, tau, s); };
            auto nw = [&]() { k::kv_clip_rows(ink, inv, T, hk0, hv0, hk1[1], hv1[1], kc, vc, dyn ? (pj ? dko[1] : dk) : nullptr, dyn ? (pj ? dvo[1] : dv) : nullptr, Hkv, D, 22.6f, 1e-6f, K[1], V[1], RING, p0,
                                              q[1], H, qs, 22.6f, r[1], NR, rs, 8.0f, tau, s, gm ? sw : nullptr, xs, sy[1], H, E, pj ? kpw : nullptr, pj ? vpw : nullptr, xk, xv, dko[1], dvo[1]); };
            reset(0); reset(1); const float uo = timeit(old), un = timeit(nw);
            reset(0); reset(1); old(); nw(); CK(cudaStreamSynchronize(s));
            size_t nd[6];
            { auto a = get(K[0], (size_t) RING * C), b = get(K[1], (size_t) RING * C); nd[0] = ndiff(a.data(), b.data(), a.size() * 2); }
            { auto a = get(V[0], (size_t) RING * C), b = get(V[1], (size_t) RING * C); nd[1] = ndiff(a.data(), b.data(), a.size() * 2); }
            { auto a = get(hk1[0], 3 * C), b = get(hk1[1], 3 * C); nd[2] = ndiff(a.data(), b.data(), a.size() * 4); }
            { auto a = get(hv1[0], 3 * C), b = get(hv1[1], 3 * C); nd[3] = ndiff(a.data(), b.data(), a.size() * 4); }
            { auto a = get(q[0], (size_t) T * H * D), b = get(q[1], (size_t) T * H * D); nd[4] = ndiff(a.data(), b.data(), a.size() * 4); }
            { auto a = get(r[0], (size_t) T * H * NR), b = get(r[1], (size_t) T * H * NR); nd[5] = ndiff(a.data(), b.data(), a.size() * 4); }
            size_t ne[3] = {0, 0, 0};
            if (gm) { auto a = get(sy[0], (size_t) T * H), b = get(sy[1], (size_t) T * H); ne[0] = ndiff(a.data(), b.data(), a.size() * 4); }
            if (pj) { auto a = get(dko[0], (size_t) T * 32), b = get(dko[1], (size_t) T * 32); ne[1] = ndiff(a.data(), b.data(), a.size() * 4);
                      auto c = get(dvo[0], (size_t) T * 32), d2 = get(dvo[1], (size_t) T * 32); ne[2] = ndiff(c.data(), d2.data(), c.size() * 4); }
            size_t tot = ne[0] + ne[1] + ne[2]; for (size_t x : nd) tot += x; bad += tot != 0;
            std::printf("         %3d %4d %4d | %8.2f %8.2f | %zu %zu %zu %zu %zu %zu %zu %zu %zu%s\n", T, dyn, gm, uo, un, nd[0], nd[1], nd[2], nd[3], nd[4], nd[5], ne[0], ne[1], ne[2], tot ? "  MISMATCH" : "");
        }
    }
    // ---- 4. attention_chunk_rows + q8_rows of its output against attention_chunk_rows making them (1.6K and 13K keys)
    {
        const int HD = H * D, CKs = k::attention_chunk_keys();
        std::printf("attn+q8: %3s %6s | %8s %8s | %s\n", "T", "keys", "old us", "new us", "bytes differing (out q d)");
        for (int nk0 : {1600, 13000}) for (int T : {1, 2, 4, 8}) {
            const int64_t p0 = nk0; const int nk = nk0 + T, ntile = (nk + CKs - 1) / CKs;
            float* q = rnd((size_t) T * HD, 1.0f); float* rel = rnd((size_t) T * H * NR, 1.0f); float* sink = rnd((size_t) T * H, 1.0f);
            float* rb = rnd((size_t) NR * 1024, 0.1f); float* gate = rnd((size_t) T * HD, 1.0f);
            std::vector<uint16_t> hk((size_t) nk * Hkv * D), hv(hk.size()); std::normal_distribution<float> nd(0.0f, 1.0f);
            for (auto& x : hk) { __half h = __float2half(nd(rng)); std::memcpy(&x, &h, 2); } for (auto& x : hv) { __half h = __float2half(nd(rng)); std::memcpy(&x, &h, 2); }
            uint16_t* Kc = dev<uint16_t>(hk.size()); uint16_t* Vc = dev<uint16_t>(hv.size()); cp(Kc, hk.data(), hk.size() * 2); cp(Vc, hv.data(), hv.size() * 2);
            const size_t Sf = (size_t) T * H * ntile * (D + 2) + 1024; float* S = dev<float>(Sf);
            float *o[2], *dd[2]; int8_t* qq[2];
            for (int f = 0; f < 2; ++f) { o[f] = dev<float>((size_t) T * HD); qq[f] = dev<int8_t>((size_t) T * HD); dd[f] = dev<float>((size_t) T * HD / 32); }
            const float sc = 1.0f / std::sqrt((float) D);
            bool okk = true;
            auto old = [&]() { okk &= k::attention_chunk_rows(q, rel, sink, T, H, Hkv, D, NR, rb, 1024, Kc, Vc, 0, p0, 0, 0, sc, S, Sf, gate, o[0], s);
                               hk::q8_rows(o[0], HD, T, HD, qq[0], dd[0], s); };
            auto nw = [&]() { okk &= k::attention_chunk_rows(q, rel, sink, T, H, Hkv, D, NR, rb, 1024, Kc, Vc, 0, p0, 0, 0, sc, S, Sf, gate, o[1], s, qq[1], dd[1]); };
            const float uo = timeit(old), un = timeit(nw); CK(cudaStreamSynchronize(s));
            auto a1 = get(o[0], (size_t) T * HD), b1 = get(o[1], (size_t) T * HD); auto a2 = get(qq[0], (size_t) T * HD), b2 = get(qq[1], (size_t) T * HD);
            auto a3 = get(dd[0], (size_t) T * HD / 32), b3 = get(dd[1], (size_t) T * HD / 32);
            const size_t n1 = ndiff(a1.data(), b1.data(), a1.size() * 4), n2 = ndiff(a2.data(), b2.data(), a2.size()), n3 = ndiff(a3.data(), b3.data(), a3.size() * 4);
            const size_t tot = n1 + n2 + n3 + (okk ? 0 : 1); bad += tot != 0;
            std::printf("         %3d %6d | %8.2f %8.2f | %zu %zu %zu%s\n", T, nk, uo, un, n1, n2, n3, tot ? "  MISMATCH" : "");
            CK(cudaFree(q)); CK(cudaFree(rel)); CK(cudaFree(sink)); CK(cudaFree(rb)); CK(cudaFree(gate)); CK(cudaFree(Kc)); CK(cudaFree(Vc)); CK(cudaFree(S));
            for (int f = 0; f < 2; ++f) { CK(cudaFree(o[f])); CK(cudaFree(qq[f])); CK(cudaFree(dd[f])); }
        }
    }
    // ---- 5. route_publish with a token's plan against router + pack3 + publish + plan_experts
    {
        float* xf = rnd((size_t) E, 1.0f); float* bias = rnd(NE, 0.01f); float* lg = rnd(NE, 2.0f);
        std::vector<int> sl(NE); for (int e = 0; e < NE; ++e) sl[e] = (e * 7) % 3 == 0 ? -1 : e / 2; int* slot = dev<int>(NE); cp(slot, sl.data(), NE * 4);
        const uint8_t* base = (const uint8_t*) 0x100000000ull; const size_t stride = 123456;
        int* ids[2]; float *w[2], *dst[2]; unsigned *flag[2]; unsigned long long* plan[2]; unsigned* cnt = dev<unsigned>(4);
        for (int f = 0; f < 2; ++f) { ids[f] = dev<int>(16); w[f] = dev<float>(16); plan[f] = dev<unsigned long long>(16);
            CK(cudaHostAlloc((void**) &dst[f], sizeof(float) * (E + 16), cudaHostAllocMapped)); CK(cudaHostAlloc((void**) &flag[f], 64, cudaHostAllocMapped)); *flag[f] = 0; }
        float* pub = dev<float>(E + 16); unsigned v = 0; const size_t n = E + 2 * NU;
        auto old = [&]() { ++v; hk::router(lg, 1, NE, bias, NU, true, 8.0f, ids[0], w[0], s); hk::pack3(xf, E, ids[0], NU, w[0], NU, pub, s);
                           hk::publish(pub, dst[0], n, cnt, flag[0], v, s); hk::plan_experts(ids[0], NU, base, stride, plan[0], s, slot, w[0]); };
        auto nw = [&]() { ++v; k::route_publish(lg, 1, NE, bias, NU, true, 8.0f, ids[1], w[1], xf, E, dst[1], cnt, flag[1], v, s, base, stride, plan[1], slot); };
        const float uo = timeit(old), un = timeit(nw); CK(cudaStreamSynchronize(s));
        auto a1 = get(ids[0], NU), b1 = get(ids[1], NU); auto a2 = get(w[0], NU), b2 = get(w[1], NU); auto a3 = get(plan[0], NU), b3 = get(plan[1], NU);
        int ncpu = 0; for (int j = 0; j < NU; ++j) ncpu += sl[a1[j]] < 0;
        const size_t tot = ndiff(a1.data(), b1.data(), NU * 4) + ndiff(a2.data(), b2.data(), NU * 4) + ndiff(a3.data(), b3.data(), NU * 8) + ndiff(dst[0], dst[1], n * 4);
        bad += tot != 0;
        std::printf("route+plan T 1 (%d of %d picks the lane's) | %8.2f %8.2f | %zu bytes differing%s\n", ncpu, NU, uo, un, tot, tot ? "  MISMATCH" : "");
    }
    // ---- 6. moe_lane_rows against moe_sum / wsum + lane_add_rows (the lane's flag already set: no wait)
    {
        float* part = rnd((size_t) 16 * W * E, 1.0f); float* src; CK(cudaHostAlloc((void**) &src, sizeof(float) * W * E, cudaHostAllocMapped));
        for (int i = 0; i < W * E; ++i) src[i] = std::sin(0.001f * i);
        unsigned* flag; CK(cudaHostAlloc((void**) &flag, 64, cudaHostAllocMapped)); *flag = 5;
        std::vector<int> sl(NE); for (int e = 0; e < NE; ++e) sl[e] = e % 5 == 0 ? -1 : e; int* slot = dev<int>(NE); cp(slot, sl.data(), NE * 4);
        std::printf("moe+lane: %3s %5s | %8s %8s | %s\n", "T", "lane", "old us", "new us", "bytes differing");
        float* out[2] = {dev<float>((size_t) W * E), dev<float>((size_t) W * E)};
        for (int T : {1, 2, 4, 8}) for (int lane = 0; lane < 2; ++lane) {
            std::vector<int> id((size_t) T * NU); std::vector<float> wv(id.size());
            for (size_t j = 0; j < id.size(); ++j) { id[j] = lane ? (int) (j * 37 % NE) : (int) (j * 37 % NE) | 1; wv[j] = (j % 3 == 0) ? 0.0f : 0.1f * (float) (j % 7); }
            if (!lane) for (auto& x : id) if (sl[x] < 0) x += 1;
            int* ids = dev<int>(id.size()); cp(ids, id.data(), id.size() * 4); float* w = dev<float>(wv.size()); cp(w, wv.data(), wv.size() * 4);
            auto old = [&]() { if (T == 1) k::wsum(part, NU, E, w, out[0], s); else hk::moe_sum(part, T, NU, E, w, out[0], s);
                               k::lane_add_rows(flag, 5, ids, slot, NU, src, out[0], E, T, s); };
            auto nw = [&]() { k::moe_lane_rows(part, T, NU, E, w, out[1], 16, flag, 5, ids, slot, src, s); };
            const float uo = timeit(old), un = timeit(nw); CK(cudaStreamSynchronize(s));
            auto a = get(out[0], (size_t) T * E), b = get(out[1], (size_t) T * E);
            const size_t tot = ndiff(a.data(), b.data(), a.size() * 4); bad += tot != 0;
            std::printf("          %3d %5d | %8.2f %8.2f | %zu%s\n", T, lane, uo, un, tot, tot ? "  MISMATCH" : "");
            CK(cudaFree(ids)); CK(cudaFree(w));
        }
    }
    // ---- 7. the router's GEMV (F32 [384][4608], T columns): hy4's gemv_f32w against gemv_f32_rows with 1 / 2 / 4 / 8 rows
    //         a block; 24 copies of the weights in turn, so each call reads them from memory as in the step
    {
        const int K = E, N = NE, NCP = 24;
        float* Wm = rnd((size_t) NCP * N * K, 0.02f); float* x = rnd((size_t) 16 * K, 1.0f);
        float* yo = dev<float>(16 * N); float* yn = dev<float>(16 * N);
        std::printf("router gemv: %3s | %8s | %8s %8s %8s %8s | %s\n", "T", "hy4 us", "rpb 1", "rpb 2", "rpb 4", "rpb 8", "bytes differing (1 2 4 8)");
        for (int T : {1, 2, 3, 4, 6, 8}) {
            int ci = 0;
            const float uo = timeit([&]() { hk::gemv_f32w(Wm + (size_t) (ci++ % NCP) * N * K, K, K, N, x, K, T, yo, N, 1.0f, s); });
            float un[4]; size_t nd[4]; const int rp[4] = {1, 2, 4, 8};
            for (int r = 0; r < 4; ++r) {
                ci = 0;
                un[r] = timeit([&]() { k::gemv_f32_rows(rp[r], Wm + (size_t) (ci++ % NCP) * N * K, K, K, N, x, K, T, yn, N, s); });
                hk::gemv_f32w(Wm, K, K, N, x, K, T, yo, N, 1.0f, s); k::gemv_f32_rows(rp[r], Wm, K, K, N, x, K, T, yn, N, s); CK(cudaStreamSynchronize(s));
                auto a = get(yo, (size_t) T * N), b = get(yn, (size_t) T * N); nd[r] = ndiff(a.data(), b.data(), a.size() * 4); bad += nd[r] != 0;
            }
            std::printf("             %3d | %8.2f | %8.2f %8.2f %8.2f %8.2f | %zu %zu %zu %zu%s\n", T, uo, un[0], un[1], un[2], un[3], nd[0], nd[1], nd[2], nd[3],
                        nd[0] + nd[1] + nd[2] + nd[3] ? "  MISMATCH" : "");
        }
        CK(cudaFree(Wm));
    }
    // ---- 8. route_publish with a window's groups against router + pack3 + publish + group_experts
    {
        float* xf = rnd((size_t) W * E, 1.0f); float* bias = rnd(NE, 0.01f);
        std::vector<int> sl(NE); for (int e = 0; e < NE; ++e) sl[e] = (e * 7) % 3 == 0 ? -1 : e / 2; int* slot = dev<int>(NE); cp(slot, sl.data(), NE * 4);
        const uint8_t* base = (const uint8_t*) 0x100000000ull; const size_t stride = 123456;
        int *ids[2], *gs[2], *ge[2], *ng[2]; float *w[2], *dst[2]; unsigned *flag[2]; unsigned long long* gp[2]; unsigned* cnt = dev<unsigned>(4);
        for (int f = 0; f < 2; ++f) { ids[f] = dev<int>(W * 16); w[f] = dev<float>(W * 16); gp[f] = dev<unsigned long long>(NE + 16); gs[f] = dev<int>(NE + 16);
            ge[f] = dev<int>(16 * W + 16); ng[f] = dev<int>(16);
            CK(cudaHostAlloc((void**) &dst[f], sizeof(float) * (E + 16) * W, cudaHostAllocMapped)); CK(cudaHostAlloc((void**) &flag[f], 64, cudaHostAllocMapped)); *flag[f] = 0; }
        float* pub = dev<float>((size_t) W * (E + 16)); unsigned v = 0;
        std::printf("route+groups: %3s | %8s %8s | %s\n", "T", "old us", "new us", "bytes differing (ids w dst groups)");
        for (int T : {2, 3, 4, 8, 16}) {
            float* lg = rnd((size_t) T * NE, 2.0f); const size_t TE = (size_t) T * E, n = TE + 2 * (size_t) T * NU;
            auto old = [&]() { ++v; hk::router(lg, T, NE, bias, NU, true, 8.0f, ids[0], w[0], s); hk::pack3(xf, TE, ids[0], (size_t) T * NU, w[0], (size_t) T * NU, pub, s);
                               hk::publish(pub, dst[0], n, cnt, flag[0], v, s);
                               hk::group_experts(ids[0], T, NU, NE, base, stride, gp[0], gs[0], ge[0], ng[0], s, slot, w[0]); };
            auto nw = [&]() { ++v; k::route_publish(lg, T, NE, bias, NU, true, 8.0f, ids[1], w[1], xf, TE, dst[1], cnt, flag[1], v, s, base, stride, nullptr, slot,
                                                    gp[1], gs[1], ge[1], ng[1]); };
            const float uo = timeit(old), un = timeit(nw);
            for (int f = 0; f < 2; ++f) { CK(cudaMemset(gp[f], 0, (NE + 16) * 8)); CK(cudaMemset(gs[f], 0, (NE + 16) * 4)); CK(cudaMemset(ge[f], 0, (16 * W + 16) * 4)); }
            old(); nw(); CK(cudaStreamSynchronize(s));
            const int G0 = get(ng[0], 1)[0];
            size_t nd = ndiff(get(ids[0], T * NU).data(), get(ids[1], T * NU).data(), T * NU * 4);
            const size_t nw2 = ndiff(get(w[0], T * NU).data(), get(w[1], T * NU).data(), T * NU * 4), nds = ndiff(dst[0], dst[1], n * 4);
            const size_t ngd = ndiff(get(ng[0], 1).data(), get(ng[1], 1).data(), 4) + ndiff(get(gp[0], NE + 16).data(), get(gp[1], NE + 16).data(), (NE + 16) * 8) +
                               ndiff(get(gs[0], NE + 16).data(), get(gs[1], NE + 16).data(), (NE + 16) * 4) + ndiff(get(ge[0], 16 * W + 16).data(), get(ge[1], 16 * W + 16).data(), (16 * W + 16) * 4);
            const size_t tot = nd + nw2 + nds + ngd; bad += tot != 0;
            std::printf("              %3d | %8.2f %8.2f | %zu %zu %zu %zu (%d groups)%s\n", T, uo, un, nd, nw2, nds, ngd, G0, tot ? "  MISMATCH" : "");
            CK(cudaFree(lg));
        }
    }
    if (k::errors() || hk::errors()) { std::printf("a launch failed\n"); ++bad; }
    std::printf("%s\n", bad ? "MISMATCH" : "all bit-identical");
    return bad ? 1 : 0;
}
