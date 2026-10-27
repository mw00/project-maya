// tests/mm3_kernels_test.cu - maya-mm3's MSA, candidate and sampling kernels on one GPU against references:
//   block_scores / select_blocks vs a CPU reference (the same bits), a NaN query's kept list (no crash);
//   sparse_attn split 16 ways vs one block vs an F64 reference, and INT8 K / V with and without the rotation vs BF16;
//   topk_rows vs host_cands (the same lists); markov_sample's distribution and draw; wait_pull.
//     nvcc -O2 -std=c++17 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120 -Isrc/mm3 \
//          tests/mm3_kernels_test.cu src/mm3/mm3_kernels.cu -o build/mm3_kernels_test && build/mm3_kernels_test
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "mm3_cands.hpp"
#include "mm3_kernels.cuh"

namespace k = mm3::k;

static uint16_t f2bf(float f) {   // (round to nearest even)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
static float bf2f(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
template <class T>
T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc(&d, std::max<size_t>(h.size(), 1) * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
template <class T>
std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

int fails = 0;
void check(bool ok, const char* what) {
    std::printf("  %-70s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) ++fails;
}

int main() {
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int blk = 128, topk = 16;
    // ---- block scores and the kept blocks
    std::printf("MSA block scores / selection (CPU reference)\n");
    for (int Hi : {1, 4})
        for (int T : {1, 8, 300}) {
            const int ctx = 70000, nblk = ctx / blk + 1, p0 = ctx - T;
            std::vector<float> iq((size_t) T * Hi * 128);
            std::vector<uint16_t> ic((size_t) ctx * 128);
            for (float& x : iq) x = nd(rng);
            for (uint16_t& x : ic) x = f2bf(nd(rng));
            if (T == 8) iq[3 * Hi * 128] = NAN;   // (query 3: NaN scores)
            std::vector<int> pos((size_t) T);
            for (int t = 0; t < T; ++t) pos[(size_t) t] = p0 + t;
            float *d_iq = dev(iq), *d_bs = nullptr;
            uint16_t* d_ic = dev(ic);
            int *d_pos = dev(pos), *d_kept = nullptr;
            cudaMalloc(&d_bs, (size_t) Hi * T * nblk * 4);
            cudaMalloc(&d_kept, (size_t) Hi * T * topk * 4);
            k::block_scores(d_iq, d_ic, d_pos, T, Hi, nblk, blk, d_bs, 0);
            k::select_blocks(d_bs, d_pos, T, Hi, nblk, blk, topk, d_kept, 0);
            const cudaError_t ce = cudaDeviceSynchronize();
            auto bs = host(d_bs, (size_t) Hi * T * nblk);
            auto kept = host(d_kept, (size_t) Hi * T * topk);
            bool same = ce == cudaSuccess, kept_ok = true;
            for (int h = 0; h < Hi; ++h)
                for (int t = 0; t < T; ++t) {
                    const int qb = pos[(size_t) t] / blk;
                    std::vector<std::pair<float, int>> sc;
                    for (int b = 0; b <= qb; ++b) {
                        float best = -INFINITY;
                        for (int j = 0; j < blk; ++j) {
                            const int kp = b * blk + j;
                            if (kp > pos[(size_t) t]) continue;
                            float d = 0.0f;
                            for (int e = 0; e < 128; ++e)
                                d = std::fmaf(iq[((size_t) t * Hi + h) * 128 + e], bf2f(ic[(size_t) kp * 128 + e]), d);
                            best = std::fmax(best, d);
                        }
                        const float g = bs[((size_t) h * T + t) * nblk + b];
                        if (b < qb && !(g == best || (std::isnan(g) && std::isnan(best)))) same = false;
                        if (b < qb) sc.push_back({best, b});
                    }
                    // the kept set: own block, then the topk - 1 best (score desc, block asc); NaN never kept
                    std::vector<int> want(1, qb);
                    std::vector<std::pair<float, int>> ok;
                    for (auto& x : sc)
                        if (!std::isnan(x.first)) ok.push_back(x);
                    std::sort(ok.begin(), ok.end(), [](auto& a, auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
                    for (int i = 0; i + 1 < topk; ++i) want.push_back(i < (int) ok.size() ? ok[(size_t) i].second : -1);
                    if (qb + 1 <= topk)
                        for (int i = 0; i < topk; ++i) want[(size_t) i] = i <= qb ? i : -1;
                    for (int i = 0; i < topk; ++i)
                        if (kept[((size_t) h * T + t) * topk + i] != want[(size_t) i]) kept_ok = false;
                }
            char buf[128];
            std::snprintf(buf, sizeof buf, "Hi %d, T %3d at 70K: scores the reference's bits", Hi, T);
            check(same, buf);
            std::snprintf(buf, sizeof buf, "Hi %d, T %3d at 70K: kept blocks the reference's%s", Hi, T,
                          T == 8 ? " (a NaN query: -1s)" : "");
            check(kept_ok, buf);
            cudaFree(d_iq); cudaFree(d_bs); cudaFree(d_ic); cudaFree(d_pos); cudaFree(d_kept);
        }
    // ---- attention: split vs one block vs F64; INT8 (rotated or not) vs BF16
    std::printf("sparse attention\n");
    {
        const int T = 8, Hq = 16, Hkv = 1, ctx = 40000, nblk = ctx / blk + 1;
        std::vector<float> q((size_t) T * Hq * 128), kf((size_t) ctx * 128), vf((size_t) ctx * 128);
        for (float& x : q) x = nd(rng);
        for (int p = 0; p < ctx; ++p)
            for (int d = 0; d < 128; ++d) {
                kf[(size_t) p * 128 + d] = nd(rng) * (d == 5 || d == 77 ? 12.0f : 1.0f);   // (outlier dims)
                vf[(size_t) p * 128 + d] = nd(rng) * (d == 9 ? 10.0f : 1.0f);
            }
        std::vector<int> pos((size_t) T), kept((size_t) T * topk);
        for (int t = 0; t < T; ++t) {
            pos[(size_t) t] = ctx - T + t;
            const int qb = pos[(size_t) t] / blk;
            kept[(size_t) t * topk] = qb;
            for (int i = 1; i < topk; ++i) kept[(size_t) t * topk + i] = (int) (rng() % (unsigned) qb);
        }
        // the caches through the engine's own write (qk_norm_rope: norm weights 1 - x normed - no rotary)
        std::vector<float> one(128, 1.0f), inv(32, 0.0f);
        float *d_q = dev(q), *d_one = dev(one), *d_inv = dev(inv);
        int* d_pos_all = nullptr;
        std::vector<int> pa((size_t) ctx);
        for (int p = 0; p < ctx; ++p) pa[(size_t) p] = p;
        d_pos_all = dev(pa);
        float *d_k = dev(kf), *d_v = dev(vf);
        uint16_t *kc = nullptr, *vc = nullptr;
        cudaMalloc(&kc, (size_t) ctx * 128 * 2);
        cudaMalloc(&vc, (size_t) ctx * 128 * 2);
        k::KvQ i8, i8r;   // (the INT8 caches; i8r's rows rotated - the queries are rotated inside sparse_attn)
        for (k::KvQ* c : {&i8, &i8r}) {
            c->kf = c->vf = k::kKvI8;
            cudaMalloc(&c->kc, (size_t) ctx * 128);
            cudaMalloc(&c->vc, (size_t) ctx * 128);
            cudaMalloc(&c->ks, (size_t) ctx * 4);
            cudaMalloc(&c->vs, (size_t) ctx * 4);
        }
        i8r.rot = 3;
        k::qk_norm_rope(nullptr, d_k, d_v, nullptr, d_one, d_pos_all, ctx, 0, Hkv, 128, 0, d_inv, 1e-6f, kc, vc, 0);
        k::qk_norm_rope(nullptr, d_k, d_v, nullptr, d_one, d_pos_all, ctx, 0, Hkv, 128, 0, d_inv, 1e-6f, nullptr, nullptr, 0, 0, i8);
        k::qk_norm_rope(nullptr, d_k, d_v, nullptr, d_one, d_pos_all, ctx, 0, Hkv, 128, 0, d_inv, 1e-6f, nullptr, nullptr, 0, 0, i8r);
        // the queries: normed (as the engine's), and a rotated copy for the rotated cache
        int* d_pos = dev(pos);
        std::vector<float> zk((size_t) T * 128, 0.0f);
        float* d_zk = dev(zk);
        float* d_qr = dev(q);
        k::qk_norm_rope(d_q, d_zk, d_zk, d_one, d_one, d_pos, T, Hq, 0, 128, 0, d_inv, 1e-6f, nullptr, nullptr, 0);
        // (the rotated cache takes the same normed queries: sparse_attn rotates them as the keys)
        cudaMemcpy(d_qr, d_q, q.size() * 4, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        int* d_kept = dev(kept);
        float *o1, *o16, *o8, *o8r, *part;
        for (float** o : {&o1, &o16, &o8, &o8r}) cudaMalloc(o, (size_t) T * Hq * 128 * 4);
        cudaMalloc(&part, k::attn_part_floats(T, Hkv, 16) * 4);
        const float scale = 1.0f / std::sqrt(128.0f);
        k::sparse_attn(d_q, kc, vc, d_kept, d_pos, T, Hq, Hkv, topk, blk, scale, o1, 0);
        k::sparse_attn(d_q, kc, vc, d_kept, d_pos, T, Hq, Hkv, topk, blk, scale, o16, 0, k::KvQ{}, k::KvHost{}, 16, part);
        k::sparse_attn(d_q, kc, vc, d_kept, d_pos, T, Hq, Hkv, topk, blk, scale, o8, 0, i8);
        k::sparse_attn(d_qr, kc, vc, d_kept, d_pos, T, Hq, Hkv, topk, blk, scale, o8r, 0, i8r, k::KvHost{}, 16, part);
        const cudaError_t ce = cudaDeviceSynchronize();
        auto a1 = host(o1, (size_t) T * Hq * 128), a16 = host(o16, a1.size()), a8 = host(o8, a1.size()),
             a8r = host(o8r, a1.size());
        auto qn = host(d_q, q.size());
        auto kb = host(kc, (size_t) ctx * 128), vb = host(vc, (size_t) ctx * 128);
        {   // the wide kernel (T <= 16: a decode token's or window's queries) against the 128-thread one (T 17), bit
            // for bit: the same 8 queries first among 17, BF16 / INT8 / INT8 rotated, one block and split 16
            const int T2 = 17;
            std::vector<int> pos2(pos), kept2(kept);
            std::vector<float> q2(qn);
            for (int t = T; t < T2; ++t) {
                pos2.push_back(pos[(size_t) (t % T)]);
                for (int i = 0; i < topk; ++i) kept2.push_back(kept[(size_t) (t % T) * topk + i]);
                q2.insert(q2.end(), qn.begin() + (size_t) (t % T) * Hq * 128, qn.begin() + (size_t) (t % T + 1) * Hq * 128);
            }
            int *d_pos2 = dev(pos2), *d_kept2 = dev(kept2);
            float* d_q2 = dev(q2);
            float *w = nullptr, *n2 = nullptr, *part2 = nullptr;
            cudaMalloc(&w, (size_t) T * Hq * 128 * 4);
            cudaMalloc(&n2, (size_t) T2 * Hq * 128 * 4);
            cudaMalloc(&part2, k::attn_part_floats(T2, Hkv, 16) * 4);
            bool same = true;
            const k::KvQ none{};
            for (const k::KvQ* c : {&none, (const k::KvQ*) &i8, (const k::KvQ*) &i8r})
                for (int sp : {1, 16}) {
                    k::sparse_attn(d_q2, kc, vc, d_kept2, d_pos2, T, Hq, Hkv, topk, blk, scale, w, 0, *c, k::KvHost{}, sp,
                                   part2);
                    k::sparse_attn(d_q2, kc, vc, d_kept2, d_pos2, T2, Hq, Hkv, topk, blk, scale, n2, 0, *c, k::KvHost{},
                                   sp, part2);
                    cudaDeviceSynchronize();
                    const auto hw = host(w, (size_t) T * Hq * 128), hn = host(n2, (size_t) T * Hq * 128);
                    same = same && std::memcmp(hw.data(), hn.data(), hw.size() * 4) == 0;
                }
            check(same, "the wide kernel (a token's few queries): the 128-thread kernel's bits");
            for (void* x : {(void*) d_pos2, (void*) d_kept2, (void*) d_q2, (void*) w, (void*) n2, (void*) part2})
                cudaFree(x);
        }
        // the F64 reference over the BF16 caches
        double e1 = 0, e16 = 0, e8 = 0, e8r = 0, mag = 0;
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < Hq; ++h) {
                std::vector<double> s;
                std::vector<int> keys;
                for (int i = 0; i < topk; ++i) {
                    const int b = kept[(size_t) t * topk + i];
                    for (int j = 0; j < blk; ++j) {
                        const int kp = b * blk + j;
                        if (kp > pos[(size_t) t]) continue;
                        double d = 0;
                        for (int e = 0; e < 128; ++e) d += (double) qn[((size_t) t * Hq + h) * 128 + e] * bf2f(kb[(size_t) kp * 128 + e]);
                        s.push_back(d * scale);
                        keys.push_back(kp);
                    }
                }
                double mx = -1e300, l = 0;
                for (double x : s) mx = std::max(mx, x);
                for (double& x : s) l += (x = std::exp(x - mx));
                for (int d = 0; d < 128; ++d) {
                    double o = 0;
                    for (size_t i = 0; i < s.size(); ++i) o += s[i] * bf2f(vb[(size_t) keys[i] * 128 + d]);
                    o /= l;
                    const size_t at = ((size_t) t * Hq + h) * 128 + d;
                    mag = std::max(mag, std::fabs(o));
                    e1 = std::max(e1, std::fabs(a1[at] - o));
                    e16 = std::max(e16, std::fabs(a16[at] - o));
                    e8 = std::max(e8, std::fabs(a8[at] - o));
                    e8r = std::max(e8r, std::fabs(a8r[at] - o));
                }
            }
        std::printf("  max |error| vs F64 (outputs up to %.2f): one block %.2e, split 16 %.2e, INT8 %.2e, INT8 rotated %.2e\n",
                    mag, e1, e16, e8, e8r);
        check(ce == cudaSuccess && e1 < 1e-4 && e16 < 1e-4, "BF16: one block and split 16 within 1e-4 of F64");
        check(e8r < e8, "INT8 rotated closer to the BF16 attention than INT8");
    }
    // ---- candidates
    std::printf("candidates (topk_rows vs host_cands)\n");
    {
        const int V = 200064, rows = 16;
        std::vector<float> lg((size_t) rows * V);
        for (float& x : lg) x = nd(rng) * 3.0f;
        for (int r = 0; r < rows; ++r) {   // (ties, zeros, a row of a big tie at the top)
            float* row = lg.data() + (size_t) r * V;
            for (int i = 0; i < 50; ++i) row[rng() % V] = row[rng() % V];
            row[rng() % V] = 0.0f;
            row[rng() % V] = -0.0f;
            if (r == 5) {
                const float top = *std::max_element(row, row + V);
                for (int i = 0; i < 70; ++i) row[rng() % V] = top;
            }
            if (r == 9)
                for (int i = 0; i < 600; ++i) row[i] = 99.0f;   // (more ties than room: n -1)
        }
        std::vector<int> extra;
        for (int i = 0; i < 40; ++i) extra.push_back((int) (rng() % V));
        float* d_lg = dev(lg);
        int* ex = nullptr;
        k::CandRow* out = nullptr;
        cudaHostAlloc((void**) &ex, extra.size() * 4, cudaHostAllocMapped);
        cudaHostAlloc((void**) &out, rows * sizeof(k::CandRow), cudaHostAllocMapped);
        std::memcpy(ex, extra.data(), extra.size() * 4);
        bool all = true;
        for (int kc : {1, 64, 104}) {
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            cudaEventRecord(e0);
            k::topk_rows(d_lg, rows, V, kc, ex, (int) extra.size(), out, 0);
            cudaEventRecord(e1);
            cudaDeviceSynchronize();
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            for (int r = 0; r < rows; ++r) {
                std::vector<std::pair<int, float>> top;
                std::vector<float> exv;
                mm3::host_cands(lg.data() + (size_t) r * V, V, kc, extra, top, exv);
                if (out[r].n < 0) {
                    if ((int) top.size() <= k::kCandCap) all = false;
                    continue;
                }
                bool ok = out[r].n == (int) top.size();
                for (int i = 0; ok && i < out[r].n; ++i)
                    ok = out[r].id[i] == top[(size_t) i].first && out[r].v[i] == top[(size_t) i].second;
                for (size_t i = 0; ok && i < extra.size(); ++i) ok = out[r].ex[i] == exv[i];
                if (!ok) {
                    all = false;
                    std::printf("  row %d kc %d: device %d vs host %zu\n", r, kc, out[r].n, top.size());
                }
            }
            std::printf("  kc %3d: 16 rows of 200064 in %.3f ms\n", kc, ms);
        }
        check(all, "every row's list the host's (and -1 where its ties overflow)");
    }
    // ---- a sampled Markov step
    std::printf("sampled Markov step\n");
    {
        const int V = 200064, D = 64, R = 8;
        std::vector<float> lg((size_t) V), bias((size_t) V), w((size_t) D + R, 0.01f), hid((size_t) D, 1.0f),
            embf((size_t) R, 0.5f), b(1, 0.0f);
        for (float& x : lg) x = nd(rng) * 3.0f;
        for (float& x : bias) x = nd(rng);
        float *d_lg = dev(lg), *d_bias = dev(bias), *d_w = dev(w), *d_hid = dev(hid), *d_embf = dev(embf), *d_b = dev(b),
              *d_conf = nullptr;
        cudaMalloc(&d_conf, 4);
        int *d_prev = nullptr, *d_out = nullptr;
        cudaMalloc(&d_prev, 4);
        cudaMalloc(&d_out, 4);
        k::QRow* qo = nullptr;
        cudaHostAlloc((void**) &qo, sizeof(k::QRow), cudaHostAllocMapped);
        bool ok = true;
        for (int it = 0; it < 20; ++it) {
            k::SampleArgs sa;
            sa.temperature = it % 2 ? 1.0f : 0.7f;
            sa.top_p = it % 3 ? 0.95f : 1.0f;
            sa.top_k = it % 4 ? 64 : 20;
            sa.u = (float) (it + 0.5) / 20.0f;
            k::markov_sample(d_lg, d_bias, V, d_prev, d_out, d_w, d_b, d_hid, D, d_embf, R, d_conf, sa, qo, 0);
            cudaDeviceSynchronize();
            const int tok = host(d_out, 1)[0];
            // the host: the biased row's distribution by the same steps (dist()'s)
            std::vector<std::pair<int, float>> top;
            std::vector<float> exv;
            std::vector<float> v((size_t) V);
            for (int i = 0; i < V; ++i) v[(size_t) i] = lg[(size_t) i] + bias[(size_t) i];
            mm3::host_cands(v.data(), V, sa.top_k, {}, top, exv);
            const int n = std::min((int) top.size(), sa.top_k);
            std::vector<double> p((size_t) n);
            double sum = 0;
            for (int i = 0; i < n; ++i) sum += (p[(size_t) i] = std::exp((double) (top[(size_t) i].second / sa.temperature - top[0].second / sa.temperature)));
            int keep = n;
            double cum = 0;
            for (int i = 0; i < n; ++i) {
                cum += p[(size_t) i] / sum;
                if (sa.top_p < 1.0f && cum >= sa.top_p) {
                    keep = i + 1;
                    break;
                }
            }
            double s2 = 0;
            for (int i = 0; i < keep; ++i) s2 += p[(size_t) i];
            if (qo->n != keep) ok = false;
            double c2 = 0;
            int want = top[(size_t) keep - 1].first;
            bool got = false;
            for (int i = 0; i < keep && i < qo->n; ++i) {
                if (qo->id[i] != top[(size_t) i].first || std::fabs(qo->q[i] - p[(size_t) i] / s2) > 1e-6) ok = false;
                c2 += qo->q[i];
                if (!got && sa.u < c2) {
                    want = qo->id[i];
                    got = true;
                }
            }
            if (tok != want) ok = false;
        }
        check(ok, "q the host's distribution (top_k, top_p, temperature), the draw its inverse CDF at u");
    }
    // ---- the host-free decode's plan on the device vs the host's (mm3_model.cu's layer_token / layer_window)
    std::printf("plan_dev (the routed experts' plan and a window's grouping)\n");
    {
        const int E = 128;
        bool ok = true;
        std::uniform_int_distribution<int> pick(0, E - 1), few(0, 11);
        std::uniform_real_distribution<float> uw(0.01f, 1.0f);
        std::vector<unsigned long long> tab((size_t) E);
        for (int e = 0; e < E; ++e) tab[(size_t) e] = 0x7f0000000000ull + (unsigned long long) ((e * 37) % E) * 0x2a00000ull;
        unsigned long long* d_tab = dev(tab);
        for (int K : {4, 8})
            for (int T : {1, 2, 5, 16}) {
                // each token's K distinct picks, from a few experts often (a window's tokens share them)
                std::vector<int> top((size_t) T * K);
                std::vector<float> w((size_t) T * K);
                for (int t = 0; t < T; ++t)
                    for (int i = 0; i < K; ++i) {
                        int x;
                        bool dup;
                        do {
                            x = t % 2 ? pick(rng) : few(rng);
                            dup = false;
                            for (int j = 0; j < i; ++j) dup |= top[(size_t) t * K + j] == x;
                        } while (dup);
                        top[(size_t) t * K + i] = x;
                        w[(size_t) t * K + i] = uw(rng);
                    }
                const float scale = 2.0f;
                // the host's plan and grouping, as layer_window builds them
                std::vector<unsigned long long> pp((size_t) T * 8), gp;
                std::vector<float> pw((size_t) T * 8);
                std::vector<std::vector<int>> ent;
                std::vector<unsigned> cnt((size_t) E, 0);
                for (int t = 0; t < T; ++t)
                    for (int i = 0; i < 8; ++i) {
                        const int e = t * 8 + i;
                        pp[(size_t) e] = 0;
                        pw[(size_t) e] = 0.0f;
                        if (i >= K) continue;
                        const int x = top[(size_t) t * K + i];
                        pw[(size_t) e] = w[(size_t) t * K + i] * scale;
                        pp[(size_t) e] = tab[(size_t) x];
                        ++cnt[(size_t) x];
                        size_t gi = 0;
                        while (gi < gp.size() && gp[gi] != pp[(size_t) e]) ++gi;
                        if (gi == gp.size()) {
                            gp.push_back(pp[(size_t) e]);
                            ent.emplace_back();
                        }
                        ent[gi].push_back(e);
                    }
                std::vector<int> gs, ge;
                for (const auto& v : ent) {
                    gs.push_back((int) ge.size());
                    ge.insert(ge.end(), v.begin(), v.end());
                }
                gs.push_back((int) ge.size());
                // the device's
                int *d_top = dev(top), *d_gs = nullptr, *d_ge = nullptr, *d_ng = nullptr;
                float *d_w = dev(w), *d_pw = nullptr;
                unsigned long long *d_pp = nullptr, *d_gp = nullptr;
                unsigned* d_cnt = nullptr;
                cudaMalloc(&d_pp, 128 * 8);
                cudaMalloc(&d_pw, 128 * 4);
                cudaMalloc(&d_gp, 128 * 8);
                cudaMalloc(&d_gs, 129 * 4);
                cudaMalloc(&d_ge, 128 * 4);
                cudaMalloc(&d_ng, 4);
                cudaMalloc(&d_cnt, (size_t) E * 4);
                cudaMemset(d_cnt, 0, (size_t) E * 4);
                ok &= k::plan_dev(d_top, d_w, T, K, d_tab, scale, d_pp, d_pw, d_cnt, d_gp, d_gs, d_ge, d_ng, 0);
                cudaDeviceSynchronize();
                const auto hpp = host(d_pp, (size_t) T * 8);
                const auto hpw = host(d_pw, (size_t) T * 8);
                const auto hng = host(d_ng, 1);
                const auto hcnt = host(d_cnt, (size_t) E);
                ok &= hpp == pp && std::memcmp(hpw.data(), pw.data(), pw.size() * 4) == 0 && hcnt == cnt;
                ok &= hng[0] == (int) gp.size();
                if (hng[0] == (int) gp.size()) {
                    ok &= host(d_gp, gp.size()) == gp;
                    ok &= host(d_gs, gs.size()) == gs;
                    ok &= host(d_ge, ge.size()) == ge;
                }
                // the token path's form: no grouping, no counts
                ok &= k::plan_dev(d_top, d_w, T, K, d_tab, scale, d_pp, d_pw, nullptr, nullptr, nullptr, nullptr,
                                  nullptr, 0);
                cudaDeviceSynchronize();
                ok &= host(d_pp, (size_t) T * 8) == pp;
                for (void* p : {(void*) d_top, (void*) d_w, (void*) d_pw, (void*) d_pp, (void*) d_gp, (void*) d_gs,
                                (void*) d_ge, (void*) d_ng, (void*) d_cnt})
                    cudaFree(p);
            }
        ok &= !k::plan_dev(nullptr, nullptr, 17, 8, d_tab, 1.0f, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                           nullptr, 0);   // (more than a window's entries: refused)
        cudaFree(d_tab);
        check(ok, "the host's plan, weights, profile counts and window grouping, value for value");
    }
    // ---- wait_pull
    std::printf("wait_pull\n");
    {
        unsigned* flag = nullptr;
        float* src = nullptr;
        cudaHostAlloc((void**) &flag, 64, cudaHostAllocMapped);
        cudaHostAlloc((void**) &src, 16 * 6144 * 4, cudaHostAllocMapped);
        *flag = 0;
        for (int i = 0; i < 16 * 6144; ++i) src[i] = (float) i;
        float* dst = nullptr;
        cudaMalloc(&dst, 16 * 6144 * 4);
        cudaMemset(dst, 0, 16 * 6144 * 4);
        k::wait_pull(flag, 1, dst, src, 16 * 6144, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const bool waited = cudaStreamQuery(0) == cudaErrorNotReady;
        *(volatile unsigned*) flag = 1;
        cudaDeviceSynchronize();
        auto h = host(dst, 16 * 6144);
        bool same = true;
        for (int i = 0; i < 16 * 6144; ++i) same &= h[(size_t) i] == (float) i;
        check(waited && same, "waits for the counter, then the rows are the host's");
    }
    std::printf("%s\n", fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
}
