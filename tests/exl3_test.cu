// tests/exl3_test.cu - the EXL3 device kernels against the scalar CPU reference (no model needed: any trellis bit
// pattern is a valid encoding, so the matrices are random words).  Covers K 1..8 and the three codebooks, the input
// split (k > 4096 and the occupancy split), column views, q8_1 input, the routed experts with a skipped plan entry,
// the CPU lane's part and the shared expert, and the full reconstruction.
//
//   exl3_test            run everything, exit 0 when every case is within tolerance
#include "strata/kernels/cpu/exl3_cpu.hpp"
#include "strata/kernels/cpu/exl3_ref.hpp"
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace ex = strata::kernels::exl3;
namespace cpu = strata::kernels::cpu;
using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;

namespace {

int g_fail = 0;
std::mt19937_64 g_rng(1234);

#define CK(x)                                                                                    \
    do {                                                                                         \
        const cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                                 \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(2);                                                                        \
        }                                                                                        \
    } while (0)

template <typename T>
T* dev_copy(const std::vector<T>& v) {
    T* d = nullptr;
    CK(cudaMalloc(&d, v.size() * sizeof(T) + 16));
    CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <typename T>
std::vector<T> host_copy(const T* d, size_t n) {
    std::vector<T> v(n);
    CK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}

std::vector<uint32_t> rand_words(size_t n) {
    std::vector<uint32_t> w(n);
    for (auto& x : w) x = (uint32_t) g_rng();
    return w;
}
// channel scales: random sign, magnitude in [0.25, 2]
std::vector<uint16_t> rand_scales(int n) {
    std::uniform_real_distribution<float> u(0.25f, 2.0f);
    std::vector<uint16_t> v((size_t) n);
    for (auto& x : v) x = f16_from_f32((g_rng() & 1) ? u(g_rng) : -u(g_rng));
    return v;
}
std::vector<float> rand_x(int n, float scale = 1.0f) {
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<float> v((size_t) n);
    for (auto& x : v) x = d(g_rng);
    return v;
}

// relative RMS error and the worst element relative to the reference's RMS
bool close(const char* what, const std::vector<float>& got, const std::vector<float>& ref, double tol) {
    double se = 0.0, sr = 0.0, worst = 0.0;
    bool finite = true;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i])) finite = false;
        const double d = (double) got[i] - ref[i];
        se += d * d;
        sr += (double) ref[i] * ref[i];
        worst = std::max(worst, std::fabs(d));
    }
    const double rms = std::sqrt(sr / (double) ref.size());
    const double rel = std::sqrt(se / std::max(sr, 1e-30));
    const bool ok = finite && rel <= tol && worst <= 8 * tol * rms + 1e-6;
    std::printf("  %-44s rel %.2e  worst/rms %.2e  %s\n", what, rel, worst / std::max(rms, 1e-30), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
    return ok;
}

struct HostMat {
    std::vector<uint32_t> tr;
    std::vector<uint16_t> suh, svh;
    int k, n, K, cb;
};
HostMat make_mat(int k, int n, int K, int cb) {
    HostMat m{rand_words((size_t) (k / 16) * (n / 16) * 8 * K), rand_scales(k), rand_scales(n), k, n, K, cb};
    return m;
}

// the engine's q8_1 of x (quantize_act.cu's arithmetic) and its dequantized values
void q8_1(const std::vector<float>& x, std::vector<uint8_t>& q, std::vector<float>& deq) {
    const size_t nb = x.size() / 32;
    q.assign(nb * 36, 0);
    deq.assign(x.size(), 0.0f);
    for (size_t b = 0; b < nb; ++b) {
        float amax = 0.0f, sum = 0.0f;
        for (int i = 0; i < 32; ++i) {
            amax = std::max(amax, std::fabs(x[b * 32 + i]));
            sum += x[b * 32 + i];
        }
        const float d = amax / 127.0f;
        const uint16_t dh = f16_from_f32(d), sh = f16_from_f32(sum);
        std::memcpy(&q[b * 36], &dh, 2);
        std::memcpy(&q[b * 36 + 2], &sh, 2);
        const float dd = f32_from_f16(dh);
        for (int i = 0; i < 32; ++i) {
            const int8_t v = amax == 0.0f ? 0 : (int8_t) std::lround(std::max(-127.0f, std::min(127.0f, x[b * 32 + i] / d)));
            q[b * 36 + 4 + i] = (uint8_t) v;
            deq[b * 32 + i] = dd * v;
        }
    }
}

void test_decode_table() {
    // the CPU decode against a few hand-checked mul1 values: state 0 -> bytesum 0 -> fp16(1024 * k_inv + k_bias)
    const float v0 = cpu::exl3_decode(0, 2);
    const float want = f32_from_f16(f16_from_f32(std::fma(1024.0f, f32_from_f16(0x1eee), f32_from_f16(0xc931))));
    if (v0 != want) {
        std::printf("  decode(0, mul1) = %g, want %g  FAIL\n", v0, want);
        ++g_fail;
    }
}

void test_mv(int k, int n, int K, int cb, bool q8, int view_c0 = -1, int view_n = 0) {
    HostMat h = make_mat(k, n, K, cb);
    uint32_t* d_tr = dev_copy(h.tr);
    uint16_t* d_suh = dev_copy(h.suh);
    uint16_t* d_svh = dev_copy(h.svh);
    ex::Mat m;
    m.trellis = d_tr;
    m.suh = d_suh;
    m.svh = d_svh;
    m.k = k;
    m.n = n;
    m.K = K;
    m.cb = cb;
    int c0 = 0, nc = n;
    if (view_c0 >= 0) {
        m = ex::view(m, view_c0, view_n);
        c0 = view_c0;
        nc = view_n;
    }
    std::vector<float> x = rand_x(k, 2.0f), xr = x;
    std::vector<uint8_t> xq;
    if (q8) q8_1(x, xq, xr);
    std::vector<float> w((size_t) k * h.n);
    cpu::exl3_inner(h.tr.data(), h.n / 16, k, h.n, K, cb, w.data());
    std::vector<float> wv((size_t) k * nc);
    for (int r = 0; r < k; ++r)
        for (int c = 0; c < nc; ++c) wv[(size_t) r * nc + c] = w[(size_t) r * h.n + c0 + c];
    std::vector<float> ref((size_t) nc), bias = rand_x(nc);
    cpu::exl3_mv(wv.data(), h.suh.data(), h.svh.data() + c0, k, nc, xr.data(), ref.data());
    const float alpha = 0.75f;
    for (int c = 0; c < nc; ++c) ref[(size_t) c] = alpha * ref[(size_t) c] + bias[(size_t) c];
    float* d_x = dev_copy(x);
    uint8_t* d_xq = q8 ? dev_copy(xq) : nullptr;
    float* d_bias = dev_copy(bias);
    float* d_y = nullptr;
    CK(cudaMalloc(&d_y, (size_t) nc * sizeof(float)));
    ex::MvJob j;
    j.m = &m;
    j.xf = q8 ? nullptr : d_x;
    j.xq = d_xq;
    j.y = d_y;
    j.bias = d_bias;
    j.alpha = alpha;
    if (!ex::mv(&j, 1, 0)) {
        std::printf("  mv refused  FAIL\n");
        ++g_fail;
    }
    CK(cudaDeviceSynchronize());
    // twice: the split's counters must come back to zero
    if (!ex::mv(&j, 1, 0)) ++g_fail;
    CK(cudaDeviceSynchronize());
    const std::vector<float> got = host_copy(d_y, (size_t) nc);
    const std::string what = "mv k" + std::to_string(k) + " n" + std::to_string(nc) + " K" + std::to_string(K) +
                             " cb" + std::to_string(cb) + (q8 ? " q8_1" : " f32") +
                             (view_c0 >= 0 ? " view@" + std::to_string(view_c0) : "");
    close(what.c_str(), got, ref, 2e-3);
    cudaFree(d_tr);
    cudaFree(d_suh);
    cudaFree(d_svh);
    cudaFree(d_x);
    if (d_xq) cudaFree(d_xq);
    cudaFree(d_bias);
    cudaFree(d_y);
}

void test_batch() {
    // three jobs, two bit widths: the launch per (K, cb) and the per-job scratch ranges
    const int k = 8192;
    HostMat a = make_mat(k, 512, 4, 2), b = make_mat(k, 256, 4, 2), c = make_mat(1024, 384, 3, 2);
    HostMat* hm[3] = {&a, &b, &c};
    ex::Mat m[3];
    ex::MvJob j[3];
    std::vector<float> x = rand_x(k);
    float* d_x = dev_copy(x);
    std::vector<float*> d_y(3);
    for (int i = 0; i < 3; ++i) {
        m[i].trellis = dev_copy(hm[i]->tr);
        m[i].suh = dev_copy(hm[i]->suh);
        m[i].svh = dev_copy(hm[i]->svh);
        m[i].k = hm[i]->k;
        m[i].n = hm[i]->n;
        m[i].K = hm[i]->K;
        m[i].cb = 2;
        CK(cudaMalloc(&d_y[(size_t) i], (size_t) hm[i]->n * sizeof(float)));
        j[i].m = &m[i];
        j[i].xf = d_x;
        j[i].y = d_y[(size_t) i];
    }
    if (!ex::mv(j, 3, 0)) ++g_fail;
    CK(cudaDeviceSynchronize());
    for (int i = 0; i < 3; ++i) {
        std::vector<float> w((size_t) hm[i]->k * hm[i]->n), ref((size_t) hm[i]->n);
        cpu::exl3_inner(hm[i]->tr.data(), hm[i]->n / 16, hm[i]->k, hm[i]->n, hm[i]->K, 2, w.data());
        cpu::exl3_mv(w.data(), hm[i]->suh.data(), hm[i]->svh.data(), hm[i]->k, hm[i]->n, x.data(), ref.data());
        close(("batch job " + std::to_string(i)).c_str(), host_copy(d_y[(size_t) i], (size_t) hm[i]->n), ref, 2e-3);
    }
}

void test_moe(int K, int cb) {
    const int E = 512, F = 256, k = 3;
    const float limit = 1.5f;
    // blob: gate [suh | svh | trellis], up [...], down [...] (4-byte offsets, like the safetensors ranges)
    HostMat parts[3][3];
    ex::ExpertLayout lay;
    lay.cb = cb;
    size_t at = 0;
    std::vector<std::vector<uint8_t>> blobs(3);
    for (int e = 0; e < 3; ++e) {
        parts[e][0] = make_mat(E, F, K, cb);
        parts[e][1] = make_mat(E, F, K, cb);
        parts[e][2] = make_mat(F, E, K, cb);
        auto& b = blobs[(size_t) e];
        at = 0;
        for (int r = 0; r < 3; ++r) {
            const HostMat& p = parts[e][r];
            const size_t o_suh = at, o_svh = o_suh + p.suh.size() * 2, o_tr = o_svh + p.svh.size() * 2 + 4;
            at = o_tr + p.tr.size() * 4;
            b.resize(at);
            std::memcpy(&b[o_suh], p.suh.data(), p.suh.size() * 2);
            std::memcpy(&b[o_svh], p.svh.data(), p.svh.size() * 2);
            std::memcpy(&b[o_tr], p.tr.data(), p.tr.size() * 4);
            lay.suh[r] = o_suh;
            lay.svh[r] = o_svh;
            lay.trellis[r] = o_tr;
            lay.K[r] = K;
        }
    }
    std::vector<uint8_t*> d_blob(3);
    for (int e = 0; e < 3; ++e) d_blob[(size_t) e] = dev_copy(blobs[(size_t) e]);
    // plan: experts 0, (skipped), 2
    std::vector<unsigned long long> plan = {(unsigned long long) d_blob[0], 0ull, (unsigned long long) d_blob[2]};
    std::vector<float> pw = {0.6f, 0.3f, 0.1f};
    std::vector<float> x = rand_x(E), cpu_part = rand_x(E), sh = rand_x(E);
    std::vector<int> flag = {1};
    auto* d_plan = dev_copy(plan);
    float* d_pw = dev_copy(pw);
    float* d_x = dev_copy(x);
    float* d_cpu = dev_copy(cpu_part);
    float* d_sh = dev_copy(sh);
    int* d_flag = dev_copy(flag);
    float* d_out = nullptr;
    CK(cudaMalloc(&d_out, (size_t) E * sizeof(float)));
    ex::MoeArgs a;
    a.plan_ptr = d_plan;
    a.plan_w = d_pw;
    a.cpu_part = d_cpu;
    a.cpu_flag = d_flag;
    a.lay = lay;
    a.k = k;
    a.n_embd = E;
    a.n_ff = F;
    a.limit = limit;
    for (int rep = 0; rep < 2; ++rep) {
        ex::moe_gate_up(a, d_x, nullptr, 0);
        ex::moe_down(a, nullptr, d_sh, d_out, 0);
    }
    CK(cudaDeviceSynchronize());
    std::vector<float> ref((size_t) E, 0.0f);
    for (int e = 0; e < 3; ++e) {
        if (plan[(size_t) e] == 0ull) continue;
        std::vector<float> g((size_t) F), u((size_t) F), hh((size_t) F), dn((size_t) E);
        for (int r = 0; r < 2; ++r) {
            const HostMat& p = parts[e][r];
            std::vector<float> w((size_t) E * F);
            cpu::exl3_inner(p.tr.data(), F / 16, E, F, K, cb, w.data());
            cpu::exl3_mv(w.data(), p.suh.data(), p.svh.data(), E, F, x.data(), r == 0 ? g.data() : u.data());
        }
        for (int i = 0; i < F; ++i) {
            const float gv = std::min(g[(size_t) i], limit), uv = std::min(std::max(u[(size_t) i], -limit), limit);
            hh[(size_t) i] = gv / (1.0f + std::exp(-gv)) * uv;
        }
        const HostMat& p = parts[e][2];
        std::vector<float> w((size_t) F * E);
        cpu::exl3_inner(p.tr.data(), E / 16, F, E, K, cb, w.data());
        cpu::exl3_mv(w.data(), p.suh.data(), p.svh.data(), F, E, hh.data(), dn.data());
        for (int i = 0; i < E; ++i) ref[(size_t) i] += pw[(size_t) e] * dn[(size_t) i];
    }
    for (int i = 0; i < E; ++i) ref[(size_t) i] += cpu_part[(size_t) i] + sh[(size_t) i];
    close(("moe K" + std::to_string(K) + " cb" + std::to_string(cb)).c_str(), host_copy(d_out, (size_t) E), ref, 3e-3);
}

void test_recon(int k, int n, int K, int cb) {
    HostMat h = make_mat(k, n, K, cb);
    ex::Mat m;
    m.trellis = dev_copy(h.tr);
    m.suh = dev_copy(h.suh);
    m.svh = dev_copy(h.svh);
    m.k = k;
    m.n = n;
    m.K = K;
    m.cb = cb;
    std::vector<float> w((size_t) k * n);
    cpu::exl3_inner(h.tr.data(), n / 16, k, n, K, cb, w.data());
    // inner: exact FP16 values
    uint16_t* d_in = nullptr;
    CK(cudaMalloc(&d_in, (size_t) k * n * 2));
    ex::reconstruct_inner(m, d_in, 0);
    CK(cudaDeviceSynchronize());
    const std::vector<uint16_t> in = host_copy(d_in, (size_t) k * n);
    size_t bad = 0;
    for (size_t i = 0; i < in.size(); ++i) bad += f32_from_f16(in[i]) != w[i];
    std::printf("  %-44s %zu of %zu differ  %s\n",
                ("inner k" + std::to_string(k) + " n" + std::to_string(n) + " K" + std::to_string(K) + " cb" +
                 std::to_string(cb)).c_str(), bad, in.size(), bad ? "FAIL" : "ok");
    if (bad) ++g_fail;
    // rows: W^T of the full weight, from the reference's columns of unit inputs
    std::vector<float> full((size_t) n * k);
    {
        std::vector<float> t = w;   // H along n per row, then along k per column, then the scales
        for (int r = 0; r < k; ++r) cpu::exl3_had128(t.data() + (size_t) r * n, (size_t) n);
        std::vector<float> col((size_t) k);
        for (int c = 0; c < n; ++c) {
            for (int r = 0; r < k; ++r) col[(size_t) r] = t[(size_t) r * n + c];
            cpu::exl3_had128(col.data(), (size_t) k);
            for (int r = 0; r < k; ++r)
                full[(size_t) c * k + r] = col[(size_t) r] * f32_from_f16(h.suh[(size_t) r]) * f32_from_f16(h.svh[(size_t) c]);
        }
    }
    uint16_t* d_rows = nullptr;
    CK(cudaMalloc(&d_rows, (size_t) k * n * 2));
    for (int bf = 0; bf < 2; ++bf) {
        ex::reconstruct_rows(m, d_rows, bf == 1, 0);
        CK(cudaDeviceSynchronize());
        const std::vector<uint16_t> rows = host_copy(d_rows, (size_t) k * n);
        std::vector<float> got(rows.size());
        for (size_t i = 0; i < rows.size(); ++i) {
            if (bf) {
                const uint32_t u = (uint32_t) rows[i] << 16;
                std::memcpy(&got[i], &u, 4);
            } else {
                got[i] = f32_from_f16(rows[i]);
            }
        }
        close(("rows " + std::string(bf ? "bf16" : "f16") + " k" + std::to_string(k) + " n" + std::to_string(n) +
               " K" + std::to_string(K)).c_str(), got, full, bf ? 6e-3 : 2e-3);
    }
}

// the prompt path's rotated product: W' alone in the row layout (exact), the inputs rotated (rows_rot_f16, gathered
// by index), the GEMM on the host, the outputs rotated (rows_rot_out) - against the reference's x W
void test_rot_rows(int k, int n, int K, int cb) {
    HostMat h = make_mat(k, n, K, cb);
    ex::Mat m;
    m.trellis = dev_copy(h.tr);
    m.suh = dev_copy(h.suh);
    m.svh = dev_copy(h.svh);
    m.k = k;
    m.n = n;
    m.K = K;
    m.cb = cb;
    std::vector<float> w((size_t) k * n);
    cpu::exl3_inner(h.tr.data(), n / 16, k, n, K, cb, w.data());
    uint16_t* d_w = nullptr;
    CK(cudaMalloc(&d_w, (size_t) k * n * 2));
    ex::reconstruct_inner_rows(m, d_w, 0);
    CK(cudaDeviceSynchronize());
    const std::vector<uint16_t> wt = host_copy(d_w, (size_t) k * n);   // [n][k]
    size_t bad = 0;
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < k; ++r) bad += f32_from_f16(wt[(size_t) c * k + r]) != w[(size_t) r * n + c];
    const std::string sh = " k" + std::to_string(k) + " n" + std::to_string(n) + " K" + std::to_string(K) + " cb" +
                           std::to_string(cb);
    std::printf("  %-44s %zu of %zu differ  %s\n", ("inner rows" + sh).c_str(), bad, wt.size(), bad ? "FAIL" : "ok");
    if (bad) ++g_fail;
    // 3 rows gathered from 5 by index
    const int rows = 3;
    const std::vector<int> idx = {4, 0, 2};
    std::vector<float> x = rand_x(5 * k, 2.0f);
    float* d_x = dev_copy(x);
    int* d_idx = dev_copy(idx);
    uint16_t* d_xr = nullptr;
    CK(cudaMalloc(&d_xr, (size_t) rows * k * 2));
    ex::rows_rot_f16(d_x, k, d_idx, rows, k, m.suh, d_xr, 0);
    CK(cudaDeviceSynchronize());
    const std::vector<uint16_t> xr = host_copy(d_xr, (size_t) rows * k);
    std::vector<float> y((size_t) rows * n), ref((size_t) rows * n);
    for (int i = 0; i < rows; ++i) {
        for (int c = 0; c < n; ++c) {
            double s = 0.0;
            for (int r = 0; r < k; ++r) s += (double) f32_from_f16(xr[(size_t) i * k + r]) * f32_from_f16(wt[(size_t) c * k + r]);
            y[(size_t) i * n + c] = (float) s;
        }
        cpu::exl3_mv(w.data(), h.suh.data(), h.svh.data(), k, n, x.data() + (size_t) idx[(size_t) i] * k,
                     ref.data() + (size_t) i * n);
    }
    float* d_y = dev_copy(y);
    ex::rows_rot_out(d_y, n, rows, n, m.svh, 0);
    CK(cudaDeviceSynchronize());
    close(("rotated product" + sh).c_str(), host_copy(d_y, (size_t) rows * n), ref, 2e-3);
    cudaFree((void*) m.trellis);
    cudaFree((void*) m.suh);
    cudaFree((void*) m.svh);
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_idx);
    cudaFree(d_xr);
    cudaFree(d_y);
}

// the prompt path's tensor-core GEMM over FP16 rows against the reference, beta = 0 and 1
void test_gemm_rows(int k, int n, int K, int cb, int rows) {
    if (!ex::mma_supported()) {
        std::printf("  gemm_rows: no tensor-core kernels on this device (skipped)\n");
        return;
    }
    HostMat h = make_mat(k, n, K, cb);
    ex::Mat m;
    m.trellis = dev_copy(h.tr);
    m.suh = dev_copy(h.suh);
    m.svh = dev_copy(h.svh);
    m.k = k;
    m.n = n;
    m.K = K;
    m.cb = cb;
    std::vector<float> w((size_t) k * n);
    cpu::exl3_inner(h.tr.data(), n / 16, k, n, K, cb, w.data());
    std::vector<uint16_t> x16((size_t) rows * k);
    std::vector<float> xr((size_t) rows * k), y0 = rand_x(rows * n), ref((size_t) rows * n);
    for (size_t i = 0; i < x16.size(); ++i) {
        x16[i] = f16_from_f32(rand_x(1, 2.0f)[0]);
        xr[i] = f32_from_f16(x16[i]);
    }
    for (int r = 0; r < rows; ++r)
        cpu::exl3_mv(w.data(), h.suh.data(), h.svh.data(), k, n, xr.data() + (size_t) r * k, ref.data() + (size_t) r * n);
    uint16_t* d_x = dev_copy(x16);
    float* d_y = dev_copy(y0);
    for (int beta = 0; beta < 2; ++beta) {
        CK(cudaMemcpy(d_y, y0.data(), y0.size() * 4, cudaMemcpyHostToDevice));
        ex::gemm_rows(m, d_x, k, rows, d_y, n, (float) beta, 0);
        CK(cudaDeviceSynchronize());
        std::vector<float> want = ref;
        if (beta)
            for (size_t i = 0; i < want.size(); ++i) want[i] += y0[i];
        close(("gemm_rows k" + std::to_string(k) + " n" + std::to_string(n) + " K" + std::to_string(K) + " cb" +
               std::to_string(cb) + " rows" + std::to_string(rows) + " beta" + std::to_string(beta)).c_str(),
              host_copy(d_y, want.size()), want, 3e-3);
    }
}

// the prompt path's expert set: 3 experts (rows 37, 0, 70) through set_gate_up and set_down against the reference
void test_set(int K, int cb) {
    if (!ex::mma_supported()) {
        std::printf("  set_*: no tensor-core kernels on this device (skipped)\n");
        return;
    }
    const int E = 512, F = 256, T = 64;
    const float limit = 1.5f;
    const int counts[3] = {37, 0, 70};
    // the set's blobs back to back at a fixed stride, each [gate | up | down], a part [trellis | suh | svh]
    HostMat parts[3][3];
    ex::ExpertLayout lay;
    lay.cb = cb;
    size_t at = 0;
    for (int r = 0; r < 3; ++r) {
        const int k = r < 2 ? E : F, n = r < 2 ? F : E;
        lay.trellis[r] = at;
        at += ex::trellis_bytes(k, n, K);
        lay.suh[r] = at;
        at += 2 * (size_t) k;
        lay.svh[r] = at;
        at += 2 * (size_t) n;
        lay.K[r] = K;
    }
    const size_t stride = (at + 255) / 256 * 256;
    std::vector<uint8_t> blobs(3 * stride);
    for (int e = 0; e < 3; ++e)
        for (int r = 0; r < 3; ++r) {
            parts[e][r] = make_mat(r < 2 ? E : F, r < 2 ? F : E, K, cb);
            const HostMat& p = parts[e][r];
            std::memcpy(&blobs[e * stride + lay.trellis[r]], p.tr.data(), p.tr.size() * 4);
            std::memcpy(&blobs[e * stride + lay.suh[r]], p.suh.data(), p.suh.size() * 2);
            std::memcpy(&blobs[e * stride + lay.svh[r]], p.svh.data(), p.svh.size() * 2);
        }
    std::vector<float> x = rand_x(T * E);
    // the sorted rows: expert e's rows are tokens (e * 7 + j) % T
    std::vector<int> bounds = {0}, row_tok;
    for (int e = 0; e < 3; ++e) {
        for (int j = 0; j < counts[e]; ++j) row_tok.push_back((e * 7 + j) % T);
        bounds.push_back((int) row_tok.size());
    }
    const int rows = (int) row_tok.size();
    uint8_t* d_blob = dev_copy(blobs);
    float* d_x = dev_copy(x);
    int* d_bounds = dev_copy(bounds);
    int* d_rows = dev_copy(row_tok);
    uint16_t* d_h = nullptr;
    float* d_out = nullptr;
    CK(cudaMalloc(&d_h, (size_t) rows * F * 2));
    CK(cudaMalloc(&d_out, (size_t) rows * E * 4));
    ex::SetArgs sa;
    sa.wbase = d_blob;
    sa.stride = stride;
    sa.lay = lay;
    sa.n_exp = 3;
    sa.bounds = d_bounds;
    sa.row0 = 0;
    sa.max_rows = 70;
    sa.n_embd = E;
    sa.n_ff = F;
    sa.limit = limit;
    ex::set_gate_up(sa, d_x, E, d_rows, d_h, F, 0);
    ex::set_down(sa, d_h, F, d_out, E, 0);
    CK(cudaDeviceSynchronize());
    std::vector<float> ref((size_t) rows * E);
    for (int e = 0, r = 0; e < 3; ++e) {
        std::vector<float> wg((size_t) E * F), wu((size_t) E * F), wd((size_t) F * E);
        cpu::exl3_inner(parts[e][0].tr.data(), F / 16, E, F, K, cb, wg.data());
        cpu::exl3_inner(parts[e][1].tr.data(), F / 16, E, F, K, cb, wu.data());
        cpu::exl3_inner(parts[e][2].tr.data(), E / 16, F, E, K, cb, wd.data());
        for (int j = 0; j < counts[e]; ++j, ++r) {
            std::vector<float> gg((size_t) F), uu((size_t) F), hh((size_t) F);
            const float* xr = x.data() + (size_t) row_tok[(size_t) r] * E;
            cpu::exl3_mv(wg.data(), parts[e][0].suh.data(), parts[e][0].svh.data(), E, F, xr, gg.data());
            cpu::exl3_mv(wu.data(), parts[e][1].suh.data(), parts[e][1].svh.data(), E, F, xr, uu.data());
            for (int i = 0; i < F; ++i) {
                const float gv = std::min(gg[(size_t) i], limit), uv = std::min(std::max(uu[(size_t) i], -limit), limit);
                hh[(size_t) i] = f32_from_f16(f16_from_f32(gv / (1.0f + std::exp(-gv)) * uv));   // h is FP16 there
            }
            cpu::exl3_mv(wd.data(), parts[e][2].suh.data(), parts[e][2].svh.data(), F, E, hh.data(),
                         ref.data() + (size_t) r * E);
        }
    }
    close(("set K" + std::to_string(K) + " cb" + std::to_string(cb) + " (3 experts, 107 rows)").c_str(),
          host_copy(d_out, ref.size()), ref, 3e-3);
}

// mv with several activations (a verify window's tokens) against one mv job per activation: the same bits
void test_mv_n(int k, int n, int K, int cb, bool q8, int ncol) {
    HostMat h = make_mat(k, n, K, cb);
    ex::Mat m;
    m.trellis = dev_copy(h.tr);
    m.suh = dev_copy(h.suh);
    m.svh = dev_copy(h.svh);
    m.k = k;
    m.n = n;
    m.K = K;
    m.cb = cb;
    // activations k + 128 floats (and q8_1 blocks) apart, outputs n + 64 apart: the strides are the job's
    const size_t xs = (size_t) k + 128, ys = (size_t) n + 64, qs = xs / 32 * 36;
    std::vector<float> x((size_t) ncol * xs), bias = rand_x(n);
    std::vector<uint8_t> xq((size_t) ncol * qs);
    for (int c = 0; c < ncol; ++c) {
        std::vector<float> xc = rand_x(k, 2.0f), deq;
        std::vector<uint8_t> qc;
        std::copy(xc.begin(), xc.end(), x.begin() + (std::ptrdiff_t) (c * xs));
        q8_1(xc, qc, deq);
        std::copy(qc.begin(), qc.end(), xq.begin() + (std::ptrdiff_t) (c * qs));
    }
    float* d_x = dev_copy(x);
    uint8_t* d_xq = dev_copy(xq);
    float* d_bias = dev_copy(bias);
    float *d_y1 = nullptr, *d_yn = nullptr;
    CK(cudaMalloc(&d_y1, (size_t) ncol * ys * sizeof(float)));
    CK(cudaMalloc(&d_yn, (size_t) ncol * ys * sizeof(float)));
    CK(cudaMemset(d_y1, 0, (size_t) ncol * ys * sizeof(float)));
    CK(cudaMemset(d_yn, 0, (size_t) ncol * ys * sizeof(float)));
    for (int c = 0; c < ncol; ++c) {
        ex::MvJob j;
        j.m = &m;
        j.xf = q8 ? nullptr : d_x + c * xs;
        j.xq = q8 ? (const void*) (d_xq + c * qs) : nullptr;
        j.y = d_y1 + c * ys;
        j.bias = d_bias;
        j.alpha = 0.75f;
        if (!ex::mv(&j, 1, 0)) ++g_fail;
    }
    ex::MvJob j;
    j.m = &m;
    j.xf = q8 ? nullptr : d_x;
    j.xq = q8 ? (const void*) d_xq : nullptr;
    j.y = d_yn;
    j.bias = d_bias;
    j.alpha = 0.75f;
    j.ncol = ncol;
    j.xf_cs = xs;
    j.xq_cs = qs;
    j.y_cs = ys;
    for (int rep = 0; rep < 2; ++rep)   // twice: the split's counters must come back to zero
        if (!ex::mv(&j, 1, 0)) ++g_fail;
    CK(cudaDeviceSynchronize());
    const std::vector<float> a = host_copy(d_y1, (size_t) ncol * ys), b = host_copy(d_yn, (size_t) ncol * ys);
    size_t diff = 0;
    for (size_t i = 0; i < a.size(); ++i) diff += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
    const std::string what = "mv x" + std::to_string(ncol) + " k" + std::to_string(k) + " n" + std::to_string(n) +
                             " K" + std::to_string(K) + " cb" + std::to_string(cb) + (q8 ? " q8_1" : " f32");
    std::printf("  %-44s %zu of %zu differ from one job each  %s\n", what.c_str(), diff, a.size(),
                diff == 0 ? "ok" : "FAIL");
    if (diff != 0) ++g_fail;
    cudaFree((void*) m.trellis);
    cudaFree((void*) m.suh);
    cudaFree((void*) m.svh);
    cudaFree(d_x);
    cudaFree(d_xq);
    cudaFree(d_bias);
    cudaFree(d_y1);
    cudaFree(d_yn);
}

// a verify window's routed experts (win_experts: each distinct expert once for its tokens) against each token's
// moe_gate_up + moe_down: the same bits.  T tokens, k plan entries each over a few experts (shared ones, a skipped
// entry, a token whose CPU-lane flag is off)
void test_win(int K, int cb, int T) {
    const int E = 512, F = 384, k = 4, NX = 5;
    const float limit = 1.5f;
    ex::ExpertLayout lay;
    lay.cb = cb;
    std::vector<uint8_t*> d_blob((size_t) NX);
    for (int e = 0; e < NX; ++e) {
        std::vector<uint8_t> b;
        size_t at = 0;
        for (int r = 0; r < 3; ++r) {
            const HostMat p = r < 2 ? make_mat(E, F, K, cb) : make_mat(F, E, K, cb);
            const size_t o_suh = at, o_svh = o_suh + p.suh.size() * 2, o_tr = o_svh + p.svh.size() * 2 + 4;
            at = o_tr + p.tr.size() * 4;
            b.resize(at);
            std::memcpy(&b[o_suh], p.suh.data(), p.suh.size() * 2);
            std::memcpy(&b[o_svh], p.svh.data(), p.svh.size() * 2);
            std::memcpy(&b[o_tr], p.tr.data(), p.tr.size() * 4);
            lay.suh[r] = o_suh;
            lay.svh[r] = o_svh;
            lay.trellis[r] = o_tr;
            lay.K[r] = K;
        }
        d_blob[(size_t) e] = dev_copy(b);
    }
    // plans [T][8]: entry i of token t on expert (t + 2 i) % NX, entry 1 of every third token skipped
    std::vector<unsigned long long> plan((size_t) T * 8, 0ull);
    std::vector<float> pw((size_t) T * 8, 0.0f);
    std::uniform_real_distribution<float> uw(0.05f, 0.5f);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < k; ++i) {
            if (i == 1 && t % 3 == 0) continue;
            plan[(size_t) t * 8 + i] = (unsigned long long) d_blob[(size_t) ((t + 2 * i) % NX)];
            pw[(size_t) t * 8 + i] = uw(g_rng);
        }
    // the groups, as glmf::moe_win_group makes them (expert order of first appearance, entries in order)
    std::vector<unsigned long long> gp;
    std::vector<std::vector<int>> members;
    for (int e = 0; e < T * 8; ++e) {
        if (e % 8 >= k || plan[(size_t) e] == 0ull) continue;
        size_t g = 0;
        while (g < gp.size() && gp[g] != plan[(size_t) e]) ++g;
        if (g == gp.size()) {
            gp.push_back(plan[(size_t) e]);
            members.emplace_back();
        }
        members[g].push_back(e);
    }
    std::vector<int> gstart(1, 0), gent;
    for (auto& mm : members) {
        for (int e : mm) gent.push_back(e);
        gstart.push_back((int) gent.size());
    }
    std::vector<int> ng = {(int) gp.size()}, flag((size_t) T);
    for (int t = 0; t < T; ++t) flag[(size_t) t] = t % 2;
    std::vector<float> x = rand_x(T * E), cpu = rand_x(T * E), sh = rand_x(T * E);
    auto* d_plan = dev_copy(plan);
    float* d_pw = dev_copy(pw);
    float* d_x = dev_copy(x);
    float* d_cpu = dev_copy(cpu);
    float* d_sh = dev_copy(sh);
    int* d_flag = dev_copy(flag);
    auto* d_gp = dev_copy(gp);
    int* d_gs = dev_copy(gstart);
    int* d_ge = dev_copy(gent);
    int* d_ng = dev_copy(ng);
    float *d_h = nullptr, *d_part = nullptr, *d_o1 = nullptr, *d_on = nullptr;
    CK(cudaMalloc(&d_h, (size_t) T * 8 * F * sizeof(float)));
    CK(cudaMalloc(&d_part, (size_t) T * 8 * E * sizeof(float)));
    CK(cudaMalloc(&d_o1, (size_t) T * E * sizeof(float)));
    CK(cudaMalloc(&d_on, (size_t) T * E * sizeof(float)));
    for (int t = 0; t < T; ++t) {
        ex::MoeArgs a;
        a.plan_ptr = d_plan + t * 8;
        a.plan_w = d_pw + t * 8;
        a.cpu_part = d_cpu + (size_t) t * E;
        a.cpu_flag = d_flag + t;
        a.lay = lay;
        a.k = k;
        a.n_embd = E;
        a.n_ff = F;
        a.limit = limit;
        ex::moe_gate_up(a, d_x + (size_t) t * E, nullptr, 0);
        ex::moe_down(a, nullptr, d_sh + (size_t) t * E, d_o1 + (size_t) t * E, 0);
    }
    ex::WinArgs w;
    w.grp_ptr = d_gp;
    w.grp_start = d_gs;
    w.grp_ent = d_ge;
    w.n_groups = d_ng;
    w.plan_ptr = d_plan;
    w.plan_w = d_pw;
    w.cpu_part = d_cpu;
    w.cpu_flag = d_flag;
    w.x = d_x;
    w.x_stride = E;
    w.h = d_h;
    w.part = d_part;
    w.sh_out = d_sh;
    w.out = d_on;
    w.out_stride = E;
    w.lay = lay;
    w.T = T;
    w.k = k;
    w.n_embd = E;
    w.n_ff = F;
    w.limit = limit;
    if (!ex::win_experts(w, 0)) ++g_fail;
    CK(cudaDeviceSynchronize());
    const std::vector<float> a = host_copy(d_o1, (size_t) T * E), b = host_copy(d_on, (size_t) T * E);
    size_t diff = 0;
    for (size_t i = 0; i < a.size(); ++i) diff += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
    const std::string what = "win K" + std::to_string(K) + " cb" + std::to_string(cb) + " " + std::to_string(T) +
                             " tokens, " + std::to_string(gp.size()) + " experts";
    std::printf("  %-44s %zu of %zu differ from the token path  %s\n", what.c_str(), diff, a.size(),
                diff == 0 ? "ok" : "FAIL");
    if (diff != 0) ++g_fail;
    for (auto* p : d_blob) cudaFree(p);
    for (void* p : {(void*) d_plan, (void*) d_pw, (void*) d_x, (void*) d_cpu, (void*) d_sh, (void*) d_flag,
                    (void*) d_gp, (void*) d_gs, (void*) d_ge, (void*) d_ng, (void*) d_h, (void*) d_part, (void*) d_o1,
                    (void*) d_on})
        cudaFree(p);
}

// the CPU lane's kernel for several activations at once against one call each: the same bits
void test_cpu_n(int k, int n, int K) {
    HostMat h = make_mat(k, n, K, 2);
    cpu::Exl3Part m;
    m.trellis = h.tr.data();
    m.suh = h.suh.data();
    m.svh = h.svh.data();
    m.k = k;
    m.n = n;
    m.K = K;
    constexpr int NT = 11;   // past kExl3CpuMaxN: two runs
    std::vector<std::vector<float>> xh;
    const float* xp[NT];
    for (int i = 0; i < NT; ++i) xh.push_back(rand_x(k));
    for (int i = 0; i < NT; ++i) xp[i] = xh[(size_t) i].data();
    std::vector<float> one((size_t) NT * 128), all((size_t) NT * 128);
    float* ap[NT];
    for (int i = 0; i < NT; ++i) ap[i] = all.data() + (size_t) i * 128;
    size_t diff = 0;
    for (int c0 = 0; c0 < n; c0 += 128)
        for (int nt = 2; nt <= NT; ++nt) {
            const int ks0 = c0 == 0 ? 0 : 3, ks1 = k / 16 - (nt % 2);
            for (int i = 0; i < nt; ++i) cpu::exl3_cpu_band_acc(m, xp[i], c0, ks0, ks1, one.data() + (size_t) i * 128);
            cpu::exl3_cpu_band_acc_n(m, xp, nt, c0, ks0, ks1, ap);
            for (size_t i = 0; i < (size_t) nt * 128; ++i) diff += std::memcmp(&one[i], &all[i], sizeof(float)) != 0;
        }
    const std::string what = "cpu x2..11 k" + std::to_string(k) + " n" + std::to_string(n) + " K" + std::to_string(K);
    std::printf("  %-44s %zu differ from one call each  %s\n", what.c_str(), diff, diff == 0 ? "ok" : "FAIL");
    if (diff != 0) ++g_fail;
}

// the CPU lane's split-K pieces (every column, slices [ks0, ks1)) against the bands over the same slices, and for
// several activations at once against one call each: the same bits
void test_cpu_rows(int k, int n, int K) {
    HostMat h = make_mat(k, n, K, 2);
    cpu::Exl3Part m;
    m.trellis = h.tr.data();
    m.suh = h.suh.data();
    m.svh = h.svh.data();
    m.k = k;
    m.n = n;
    m.K = K;
    constexpr int NT = 11;
    std::vector<std::vector<float>> xh;
    const float* xp[NT];
    for (int i = 0; i < NT; ++i) xh.push_back(rand_x(k));
    for (int i = 0; i < NT; ++i) xp[i] = xh[(size_t) i].data();
    std::vector<float> band(128), one((size_t) NT * n), all((size_t) NT * n);
    float* ap[NT];
    for (int i = 0; i < NT; ++i) ap[i] = all.data() + (size_t) i * n;
    size_t diff = 0;
    for (int nt = 1; nt <= NT; ++nt) {
        const int ks0 = nt % 3, ks1 = k / 16 - (nt % 2);
        for (int i = 0; i < nt; ++i) {
            cpu::exl3_cpu_rows_acc(m, xp[i], ks0, ks1, one.data() + (size_t) i * n);
            for (int c0 = 0; c0 < n; c0 += 128) {
                cpu::exl3_cpu_band_acc(m, xp[i], c0, ks0, ks1, band.data());
                for (int j = 0; j < 128; ++j)
                    diff += std::memcmp(&band[(size_t) j], &one[(size_t) i * n + c0 + j], sizeof(float)) != 0;
            }
        }
        if (nt > 1) {
            cpu::exl3_cpu_rows_acc_n(m, xp, nt, ks0, ks1, ap);
            for (size_t i = 0; i < (size_t) nt * n; ++i) diff += std::memcmp(&one[i], &all[i], sizeof(float)) != 0;
        }
    }
    const std::string what = "cpu rows x1..11 k" + std::to_string(k) + " n" + std::to_string(n) + " K" + std::to_string(K);
    std::printf("  %-44s %zu differ from the bands / one call each  %s\n", what.c_str(), diff, diff == 0 ? "ok" : "FAIL");
    if (diff != 0) ++g_fail;
}

// the CPU lane's input block (x * suh, the 128-point transform) against the scalar transform: the same bits
void test_cpu_had() {
    size_t diff = 0;
    for (int rep = 0; rep < 64; ++rep) {
        std::vector<float> x = rand_x(128, 3.0f), want(128), got(128);
        std::vector<uint16_t> suh(128);
        for (int i = 0; i < 128; ++i) suh[(size_t) i] = (uint16_t) (0x3800 + (rep * 131 + i * 37) % 0x800);   // [0.5, 1)
        for (int i = 0; i < 128; ++i) want[(size_t) i] = x[(size_t) i] * strata::kernels::f32_from_f16(suh[(size_t) i]);
        for (int h = 1; h < 128; h <<= 1)
            for (int i = 0; i < 128; i += 2 * h)
                for (int j = i; j < i + h; ++j) {
                    const float a = want[(size_t) j], b = want[(size_t) (j + h)];
                    want[(size_t) j] = a + b;
                    want[(size_t) (j + h)] = a - b;
                }
        for (int i = 0; i < 128; ++i) want[(size_t) i] *= 0.088388347648f;
        cpu::exl3_cpu_prepare_block(suh.data(), x.data(), got.data());
        for (int i = 0; i < 128; ++i) diff += std::memcmp(&want[(size_t) i], &got[(size_t) i], sizeof(float)) != 0;
    }
    std::printf("  %-44s %zu of 8192 differ from the scalar transform  %s\n", "cpu input blocks (128-point transform)",
                diff, diff == 0 ? "ok" : "FAIL");
    if (diff != 0) ++g_fail;
}

// the prompt path's CPU GEMM (a band decoded once for many activations) against the band kernel: the same sums up to
// F32 rounding (another order)
void test_cpu_gemm(int k, int n, int K) {
    HostMat h = make_mat(k, n, K, 2);
    cpu::Exl3Part m;
    m.trellis = h.tr.data();
    m.suh = h.suh.data();
    m.svh = h.svh.data();
    m.k = k;
    m.n = n;
    m.K = K;
    constexpr int NT = 21;   // 3 tiles of 6 and 3 left over
    std::vector<std::vector<float>> xh;
    const float* xp[NT];
    for (int i = 0; i < NT; ++i) xh.push_back(rand_x(k));
    for (int i = 0; i < NT; ++i) xp[i] = xh[(size_t) i].data();
    std::vector<float> one(128), all((size_t) NT * 128), wc(cpu::exl3_cpu_gemm_scratch() + 16);
    float* const w64 = (float*) (((uintptr_t) wc.data() + 63) & ~(uintptr_t) 63);
    float* ap[NT];
    for (int i = 0; i < NT; ++i) ap[i] = all.data() + (size_t) i * 128;
    double worst = 0.0;
    for (int nb : {1, 2}) {
        all.assign((size_t) NT * 128 * nb, 0.0f);
        for (int i = 0; i < NT; ++i) ap[i] = all.data() + (size_t) i * 128 * nb;
        for (int c0 = 0; c0 + 128 * nb <= n; c0 += 128 * nb) {
            cpu::exl3_cpu_band_gemm(m, xp, NT, c0, nb, ap, w64);
            for (int i = 0; i < NT; ++i)
                for (int b = 0; b < nb; ++b) {
                    cpu::exl3_cpu_band_acc(m, xp[i], c0 + 128 * b, 0, k / 16, one.data());
                    double mx = 0.0, d = 0.0;
                    for (int j = 0; j < 128; ++j) {
                        mx = std::max(mx, (double) std::fabs(one[(size_t) j]));
                        d = std::max(d, (double) std::fabs(one[(size_t) j] - ap[i][128 * b + j]));
                    }
                    worst = std::max(worst, d / std::max(mx, 1e-30));
                }
        }
    }
    const std::string what = "cpu gemm x21 nb1,2 k" + std::to_string(k) + " n" + std::to_string(n) + " K" + std::to_string(K);
    const bool ok = worst < 1e-5;
    std::printf("  %-44s worst %.2e of the band's largest  %s\n", what.c_str(), worst, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

const char* cpu_tier_name(int tier) { return tier == 2 ? "avx512" : tier == 1 ? "avx2" : "scalar"; }

// The vector tiers' arithmetic, written out on its own: column c's state at row r is the 16-bit window that ends at
// bit (p + 1 + 256) K of the tile's MSB-first bit ring (p: the tile's step at (r, c) - row (p/8 % 4) 2 + (p & 1) +
// 8 (p/2 & 1), column p/32 + 8 (p/4 & 1)), read bit by bit; its mul1 byte sum s; and one fma(s, xh, acc) a weight in
// (slice, row) order
float exact_band_col(const cpu::Exl3Part& m, const float* xh, int col, int ks0, int ks1) {
    static int step[16][16];
    static bool init = false;
    if (!init) {
        for (int p = 0; p < 256; ++p) step[(p / 8 % 4) * 2 + (p & 1) + 8 * (p / 2 & 1)][p / 32 + 8 * (p / 4 & 1)] = p;
        init = true;
    }
    const int bits = 256 * m.K, t = col / 16, c = col % 16;
    float a = 0.0f;
    for (int ks = ks0; ks < ks1; ++ks) {
        const uint32_t* tile = m.trellis + ((size_t) ks * (m.n / 16) + t) * 8 * m.K;
        for (int r = 0; r < 16; ++r) {
            const int end = (step[r][c] + 1 + 256) * m.K;
            uint32_t st = 0;
            for (int q = end - 16; q < end; ++q) {
                const int b = q % bits;
                st = (st << 1) | ((tile[b >> 5] >> (31 - (b & 31))) & 1u);
            }
            const uint32_t x = st * 0x83DCD12Du;
            const int s = (int) ((x & 0xffu) + ((x >> 8) & 0xffu) + ((x >> 16) & 0xffu) + (x >> 24));
            a = std::fma((float) s, xh[ks * 16 + r], a);
        }
    }
    return a;
}

// every tier this CPU runs against that model: the vector tiers bit for bit (so on an AVX-512 host AVX2 and AVX-512
// against each other too), the scalar tier (no FMA) within rounding; every band, the whole slice range and partial
// ones, 1..11 activations at once (past kExl3CpuMaxN)
void test_cpu_exact(int k, int n, int K) {
    HostMat h = make_mat(k, n, K, 2);
    cpu::Exl3Part m;
    m.trellis = h.tr.data();
    m.suh = h.suh.data();
    m.svh = h.svh.data();
    m.k = k;
    m.n = n;
    m.K = K;
    constexpr int NT = 11;
    std::vector<std::vector<float>> xh;
    const float* xp[NT];
    for (int i = 0; i < NT; ++i) xh.push_back(rand_x(k));
    for (int i = 0; i < NT; ++i) xp[i] = xh[(size_t) i].data();
    std::vector<float> want((size_t) NT * 128), got((size_t) NT * 128);
    float* gp[NT];
    for (int i = 0; i < NT; ++i) gp[i] = got.data() + (size_t) i * 128;
    const int S = k / 16;
    const int ranges[3][2] = {{0, S}, {3, S - 1}, {S / 2, S / 2 + 1}};
    size_t diff[3] = {0, 0, 0}, cmp[3] = {0, 0, 0};
    double se = 0.0, sr = 0.0;
    for (int c0 = 0; c0 < n; c0 += 128)
        for (const auto& rg : ranges) {
            for (int i = 0; i < NT; ++i)
                for (int j = 0; j < 128; ++j) want[(size_t) i * 128 + j] = exact_band_col(m, xp[i], c0 + j, rg[0], rg[1]);
            for (int tier = 0; tier <= 2; ++tier)
                for (int nt : {1, 2, 3, 5, 8, 11}) {
                    if (!cpu::exl3_cpu_band_acc_on(tier, m, xp, nt, c0, rg[0], rg[1], gp)) break;
                    for (size_t i = 0; i < (size_t) nt * 128; ++i) {
                        ++cmp[tier];
                        diff[tier] += std::memcmp(&got[i], &want[i], sizeof(float)) != 0;
                        if (tier == 0) {
                            se += ((double) got[i] - want[i]) * ((double) got[i] - want[i]);
                            sr += (double) want[i] * want[i];
                        }
                    }
                }
        }
    for (int tier = 1; tier <= 2; ++tier) {
        if (cmp[tier] == 0) continue;
        const std::string what = std::string("cpu exact ") + cpu_tier_name(tier) + " k" + std::to_string(k) + " n" +
                                 std::to_string(n) + " K" + std::to_string(K);
        std::printf("  %-44s %zu of %zu differ from the model  %s\n", what.c_str(), diff[tier], cmp[tier],
                    diff[tier] == 0 ? "ok" : "FAIL");
        if (diff[tier] != 0) ++g_fail;
    }
    const double rel = std::sqrt(se / std::max(sr, 1e-30));
    const std::string what = "cpu exact scalar k" + std::to_string(k) + " n" + std::to_string(n) + " K" + std::to_string(K);
    const bool ok = cmp[0] > 0 && rel <= 1e-5;
    std::printf("  %-44s rel %.2e to the model  %s\n", what.c_str(), rel, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// the CPU lane's kernel (mul1) against the reference: every band of a k x n matrix
void test_cpu(int k, int n, int K) {
    HostMat h = make_mat(k, n, K, 2);
    std::vector<float> x = rand_x(k, 2.0f), w((size_t) k * n), ref((size_t) n), got((size_t) n), xh((size_t) k);
    cpu::exl3_inner(h.tr.data(), n / 16, k, n, K, 2, w.data());
    cpu::exl3_mv(w.data(), h.suh.data(), h.svh.data(), k, n, x.data(), ref.data());
    cpu::Exl3Part m;
    m.trellis = h.tr.data();
    m.suh = h.suh.data();
    m.svh = h.svh.data();
    m.k = k;
    m.n = n;
    m.K = K;
    const float xs = cpu::exl3_cpu_prepare(m, x.data(), xh.data());
    for (int c0 = 0; c0 < n; c0 += 128) cpu::exl3_cpu_band(m, xh.data(), xs, c0, got.data() + c0);
    close(("cpu " + std::string(cpu_tier_name(cpu::exl3_cpu_tier())) + " k" + std::to_string(k) + " n" +
           std::to_string(n) + " K" + std::to_string(K)).c_str(), got, ref, 2e-3);
}

}  // namespace

int main() {
    int dev = 0;
    cudaDeviceProp p{};
    CK(cudaGetDevice(&dev));
    CK(cudaGetDeviceProperties(&p, dev));
    std::printf("exl3_test on %s (%.1f GB)\n", p.name, (double) p.totalGlobalMem / 1e9);
    test_decode_table();
    for (int K = 1; K <= 8; ++K)
        for (int cb = 0; cb <= 2; ++cb) test_mv(256, 256, K, cb, false);
    test_mv(4096, 512, 3, 2, false);        // occupancy split
    test_mv(12288, 256, 4, 2, false);       // k > 4096
    test_mv(2048, 4096, 5, 2, true);        // q8_1 input
    test_mv(4096, 1024, 2, 2, false, 384, 256);   // a column view
    test_batch();
    for (int K : {2, 3, 4}) test_moe(K, 2);
    test_moe(3, 1);
    for (int K : {2, 5}) test_recon(256, 384, K, 2);
    test_recon(256, 256, 3, 0);
    for (int K = 1; K <= 8; ++K) test_rot_rows(256, 384, K, 2);
    test_rot_rows(512, 256, 4, 1);
    for (int K : {2, 3, 5, 8}) test_gemm_rows(512, 384, K, 2, 45);
    test_gemm_rows(4096, 256, 3, 1, 100);
    test_gemm_rows(1024, 128, 4, 0, 7);
    for (int K : {2, 3, 4}) test_set(K, 2);
    for (int K = 1; K <= 8; ++K) test_cpu(512, 256, K);
    test_cpu(4096, 2048, 3);
    test_cpu(2048, 4096, 3);
    // several activations at once (a verify window): the same bits as one at a time
    for (int K = 1; K <= 8; ++K) test_mv_n(512, 256, K, 2, false, 3);
    for (int cb = 0; cb <= 1; ++cb) test_mv_n(256, 384, 4, cb, false, 5);
    for (int nc : {2, 4, 7, 16}) test_mv_n(4096, 512, 3, 2, false, nc);   // occupancy split
    test_mv_n(12288, 256, 6, 2, false, 6);                                  // k > 4096
    test_mv_n(2048, 4096, 5, 2, true, 8);                                   // q8_1 input
    for (int K : {2, 3, 4, 8}) test_win(K, 2, 7);
    test_win(3, 1, 2);
    test_win(3, 2, 16);
    for (int K : {1, 3, 6, 8}) test_cpu_n(512, 256, K);
    for (int K : {2, 3, 4, 5, 8}) test_cpu_rows(512, 384, K);
    for (int K : {3, 4, 6}) test_cpu_gemm(768, 256, K);
    test_cpu_had();
    // the CPU tiers against an exact model of their arithmetic (odd slice counts too: 37 slices)
    for (int K = 1; K <= 8; ++K) test_cpu_exact(512, 256, K);
    test_cpu_exact(592, 384, 3);
    test_cpu_exact(592, 256, 4);
    std::printf("exl3_test: %s (%d failures, %d launch errors)\n", g_fail ? "FAIL" : "ok", g_fail,
                ex::launch_errors());
    return g_fail || ex::launch_errors() ? 1 : 0;
}
