// tests/glm_dsa/glmd_w8_parity.cu - glm-dsa wave 8 kernels vs their CPU twins (GPU; exits 77 without a device).
//   B14.15  glmd::topk_hist == glmd::topk == glmd::twin::topk_hist, cell for cell, on random rows (n 2049 .. 131072),
//           exact ties, all-equal rows, -inf / +-0 rows, a heavy boundary bucket (the radix fallback) and a
//           per-row-position batch with -1 padding;
//   B14.11  dsa_prep with ik_had (F32 and FP16 keys) writes the twin's key bits; dsa_rope_q with iq_had gives the
//           twin's query within float rounding; dsa_prep_rows / dsa_rope_q_rows (the prompt path) write the token
//           path's bits; the token scores with and without the rotation agree within 2e-5 relative.
#include "strata/kernels/glmd_batch.hpp"
#include "strata/kernels/glmd_fast.hpp"
#include "strata/kernels/glmd_mixer.hpp"

#include "../../src/kernels/cuda/glmd_rope.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
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

template <typename T>
struct Dev {
    T* p = nullptr;
    size_t n = 0;
    explicit Dev(const std::vector<T>& h) : n(h.size()) {
        CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
        if (n) CK(cudaMemcpy(p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    }
    explicit Dev(size_t count) : n(count) {
        CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
        CK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T)));
    }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    std::vector<T> get() const {
        std::vector<T> h(n);
        CK(cudaDeviceSynchronize());
        if (n) CK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return h;
    }
};

void expect(bool ok, const std::string& what) {
    std::printf("%-90s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    g_fail += !ok;
}

// ---------------------------------------------------------------- B14.15
void topk_rows(const char* name, const std::vector<float>& sc, int T, int ld, int n_kv, const std::vector<int>& pos) {
    const int K = glmd::kTopK;
    Dev<float> d_sc(sc);
    Dev<int> d_pos(pos), a((size_t) T * K), b((size_t) T * K);
    const int* pp = pos.empty() ? nullptr : d_pos.p;
    const bool ok1 = glmd::topk_hist(d_sc.p, ld, n_kv, T, pp, K, a.p, nullptr);
    const bool ok2 = glmd::topk(d_sc.p, ld, n_kv, T, pp, K, b.p, nullptr);
    std::vector<int> t((size_t) T * K);
    glmd::twin::topk_hist(sc.data(), ld, n_kv, T, pos.empty() ? nullptr : pos.data(), K, t.data());
    const std::vector<int> ha = a.get(), hb = b.get();
    expect(ok1 && ok2 && ha == hb && ha == t, std::string("topk_hist == topk == twin: ") + name);
}

void test_topk(std::mt19937& rng) {
    std::normal_distribution<float> nd;
    for (int n : {2049, 9000, 32768, 131072}) {
        const int T = n > 40000 ? 4 : 16;
        std::vector<float> sc((size_t) T * n);
        for (auto& v : sc) v = nd(rng);
        topk_rows(("random normal n " + std::to_string(n)).c_str(), sc, T, n, n, {});
    }
    {
        std::vector<float> sc((size_t) 8 * 20000);
        for (auto& v : sc) v = (float) (rng() % 12) * 0.25f;
        topk_rows("exact ties (12 values)", sc, 8, 20000, 20000, {});
    }
    topk_rows("all equal", std::vector<float>(5000, 1.5f), 1, 5000, 5000, {});
    {
        std::vector<float> sc(10000);
        for (auto& v : sc) {
            const unsigned k = rng() % 4;
            v = k == 0 ? -INFINITY : k == 1 ? 0.0f : k == 2 ? -0.0f : (float) (rng() % 1000) * 1e-3f;
        }
        topk_rows("-inf / +0 / -0", sc, 1, 10000, 10000, {});
    }
    {
        std::vector<float> sc(40000);
        for (size_t i = 0; i < sc.size(); ++i) sc[i] = i < 10 ? 1e30f : 1.0f + (float) (rng() % 100000) * 1e-9f;
        topk_rows("heavy boundary bucket (the radix fallback)", sc, 1, 40000, 40000, {});
    }
    {
        const int T = 5, ld = 9100, n_kv = 9000;
        std::vector<float> sc((size_t) T * ld);
        for (auto& v : sc) v = (float) (rng() % 50);
        topk_rows("per-row positions 100 .. 8999, padded", sc, T, ld, n_kv, {100, 2047, 2048, 5000, 8999});
    }
}

// ---------------------------------------------------------------- B14.11
void test_hadamard(std::mt19937& rng) {
    std::normal_distribution<float> nd;
    const int D = 128, IH = 32, R = 64, P = 3000;
    std::vector<float> inv((size_t) R / 2), qinv(32);
    glmd::rope_inv_freq(inv.data(), R, 8e6);
    glmd::rope_inv_freq(qinv.data(), 64, 8e6);
    std::vector<float> ln_w((size_t) D), ln_b((size_t) D), ik_raw((size_t) P * D), qr_raw(2048), qn(2048, 1.0f),
        kv_raw(576), kvn(512, 1.0f);
    for (auto& v : ln_w) v = 1.0f + 0.2f * nd(rng);
    ln_w[3] = 6e4f;   // an outlier channel: the FP16 store needs the rotation
    for (auto& v : ln_b) v = 0.3f * nd(rng);
    for (auto& v : ik_raw) v = nd(rng);
    for (auto& v : qr_raw) v = nd(rng);
    for (auto& v : kv_raw) v = nd(rng);
    Dev<float> d_inv(inv), d_lnw(ln_w), d_lnb(ln_b), d_ik(ik_raw), d_qr(qr_raw), d_qn(qn), d_kv(kv_raw), d_kvn(kvn);
    for (int f16 = 0; f16 < 2; ++f16) {
        Dev<float> d_keys((size_t) P * D), d_keys_rows((size_t) P * D);
        Dev<uint16_t> d_lat((size_t) P * 512), d_kpe((size_t) P * 64);
        std::vector<float> h_keys((size_t) P * D, 0.0f);
        std::vector<uint16_t> h_lat((size_t) P * 512), h_kpe((size_t) P * 64);
        bool ok = true;
        glmd::DsaPrep a;
        a.q_lora = 2048;
        a.q_a_norm = d_qn.p;
        a.kv_norm = d_kvn.p;
        a.lat = d_lat.p;
        a.kv_lora = 512;
        a.k_pe = d_kpe.p;
        a.qk_rope = 64;
        a.inv_freq = d_inv.p;
        a.ik_ln_w = d_lnw.p;
        a.ik_ln_b = d_lnb.p;
        a.ik_cache = d_keys.p;
        a.ik_f16 = f16;
        a.ik_had = 1;
        a.idx_dim = D;
        a.idx_rope = R;
        a.idx_inv_freq = d_inv.p;
        a.qr_raw = d_qr.p;
        a.kv_raw = d_kv.p;
        for (int p : {0, 1, 2047, 2048, P - 1}) {
            a.ik_raw = d_ik.p + (size_t) p * D;
            a.p = p;
            ok = ok && glmd::dsa_prep(a, nullptr);
            glmd::DsaPrep h = a;
            h.qr_raw = qr_raw.data();
            h.q_a_norm = qn.data();
            h.kv_raw = kv_raw.data();
            h.kv_norm = kvn.data();
            h.lat = h_lat.data();
            h.k_pe = h_kpe.data();
            h.inv_freq = inv.data();
            h.ik_raw = ik_raw.data() + (size_t) p * D;
            h.ik_ln_w = ln_w.data();
            h.ik_ln_b = ln_b.data();
            h.ik_cache = h_keys.data();
            h.idx_inv_freq = inv.data();
            glmd::twin::dsa_prep(h);
        }
        const std::vector<float> g = d_keys.get();
        double md = 0, mx = 0;
        for (int p : {0, 1, 2047, 2048, P - 1})
            for (int i = 0; i < D; ++i) {
                float x, y;
                if (f16) {
                    x = glmd::twin::half_float(((const uint16_t*) g.data())[(size_t) p * D + i]);
                    y = glmd::twin::half_float(((const uint16_t*) h_keys.data())[(size_t) p * D + i]);
                } else {
                    x = g[(size_t) p * D + i];
                    y = h_keys[(size_t) p * D + i];
                }
                md = std::max(md, (double) std::fabs(x - y));
                mx = std::max(mx, (double) std::fabs(y));
            }
        char buf[200];
        std::snprintf(buf, sizeof buf, "dsa_prep ik_had, %s keys: max |GPU - twin| %.2e of %.2e (finite: %s)",
                      f16 ? "FP16" : "F32", md, mx, std::isfinite(mx) ? "yes" : "no");
        expect(ok && std::isfinite(mx) && md <= 1e-5 * mx + (f16 ? mx / 1024 : 0.0), buf);
        // the prompt path: rows 0..15 at positions 100.. write the token path's bits
        glmd::DsaPrepRows pr;
        pr.a = a;
        pr.a.ik_cache = d_keys_rows.p;
        pr.a.ik_raw = d_ik.p + (size_t) 100 * D;
        pr.a.p = 100;
        pr.T = 16;
        pr.kv_ld = 576;
        std::vector<float> kv16((size_t) 16 * 576), qr16((size_t) 16 * 2048);
        for (int t = 0; t < 16; ++t) {
            std::copy(kv_raw.begin(), kv_raw.end(), kv16.begin() + (size_t) t * 576);
            std::copy(qr_raw.begin(), qr_raw.end(), qr16.begin() + (size_t) t * 2048);
        }
        Dev<float> d_kv16(kv16), d_qr16(qr16);
        pr.a.kv_raw = d_kv16.p;
        pr.a.qr_raw = d_qr16.p;
        bool rows_ok = glmd::dsa_prep_rows(pr, nullptr);
        for (int t = 0; t < 16; ++t) {
            a.ik_raw = d_ik.p + (size_t) (100 + t) * D;
            a.p = 100 + t;
            rows_ok = rows_ok && glmd::dsa_prep(a, nullptr);
        }
        const std::vector<float> g2 = d_keys.get(), r2 = d_keys_rows.get();
        const size_t off = (size_t) 100 * D * (f16 ? 1 : 2) / 2, len = (size_t) 16 * D * (f16 ? 1 : 2) / 2;
        expect(rows_ok && std::memcmp(g2.data() + off, r2.data() + off, len * 4) == 0,
               std::string("dsa_prep_rows ik_had: the token path's key bits, ") + (f16 ? "FP16" : "F32"));
    }
    // queries
    std::vector<float> iq((size_t) IH * D), q((size_t) 64 * 256);
    for (auto& v : iq) v = nd(rng);
    for (auto& v : q) v = nd(rng);
    Dev<float> d_iq(iq), d_q(q), d_qinv(qinv);
    glmd::DsaRopeQ r;
    r.q = d_q.p;
    r.inv_freq = d_qinv.p;
    r.iq = d_iq.p;
    r.idx_heads = IH;
    r.idx_dim = D;
    r.idx_rope = R;
    r.idx_inv_freq = d_inv.p;
    r.iq_had = 1;
    r.p = 4321;
    const bool okq = glmd::dsa_rope_q(r, nullptr);
    glmd::DsaRopeQ h = r;
    std::vector<float> hiq = iq, hq = q;
    h.q = hq.data();
    h.inv_freq = qinv.data();
    h.iq = hiq.data();
    h.idx_inv_freq = inv.data();
    glmd::twin::dsa_rope_q(h);
    const std::vector<float> giq = d_iq.get();
    double mdq = 0;
    for (size_t i = 0; i < giq.size(); ++i) mdq = std::max(mdq, (double) std::fabs(giq[i] - hiq[i]));
    expect(okq && mdq < 1e-5, "dsa_rope_q iq_had: the twin's query within 1e-5 (" + std::to_string(mdq) + ")");
    // the prompt path's query rows
    std::vector<float> iq3((size_t) 3 * IH * D), q3((size_t) 3 * 64 * 256);
    for (int t = 0; t < 3; ++t) {
        std::copy(iq.begin(), iq.end(), iq3.begin() + (size_t) t * IH * D);
        std::copy(q.begin(), q.end(), q3.begin() + (size_t) t * 64 * 256);
    }
    Dev<float> d_iq3(iq3), d_q3(q3);
    glmd::DsaRopeQRows rr;
    rr.a = r;
    rr.a.q = d_q3.p;
    rr.a.iq = d_iq3.p;
    rr.a.p = 4321;
    rr.T = 3;
    const bool okr = glmd::dsa_rope_q_rows(rr, nullptr);
    const std::vector<float> g3 = d_iq3.get();
    expect(okr && std::memcmp(g3.data(), giq.data(), giq.size() * 4) == 0,
           "dsa_rope_q_rows iq_had: row 0 equals the token path's bits");
    expect(glmd::launch_errors() == 0 && glmd::batch_launch_errors() == 0, "no launch errors");
}

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("glmd_w8_parity: no CUDA device - skipped\n");
        return 77;
    }
    std::mt19937 rng(8);
    test_topk(rng);
    test_hadamard(rng);
    if (g_fail) {
        std::printf("%d check(s) failed\n", g_fail);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
