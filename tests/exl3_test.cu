// tests/exl3_test.cu - the EXL3 device kernels against the scalar CPU reference (no model needed: any trellis bit
// pattern is a valid encoding, so the matrices are random words).  Covers K 1..8 and the three codebooks, the input
// split (k > 4096 and the occupancy split), column views, q8_1 input, the routed experts with a skipped plan entry,
// the CPU lane's part and the shared expert, and the full reconstruction.
//
//   exl3_test                     run everything, exit 0 when every case is within tolerance
//   exl3_test --exllamav3 <dir>   the decoders against exllamav3 itself: tools/exl3_exllamav3_fixture.py's random
//                                 tiles (data/exl3-exllamav3) - the same states and W' bit for bit
//   exl3_test --expect <dir>      a model's matrices against exllamav3's decode of them (the tool's --model files)
#include "strata/kernels/cpu/exl3_cpu.hpp"
#include "strata/kernels/cpu/exl3_ref.hpp"
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
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

// the bitrate k2 = 2 K as a name: "3", "3.5"
std::string kname(int k2) { return (k2 & 1) ? std::to_string(k2 / 2) + ".5" : std::to_string(k2 / 2); }

struct HostMat {
    std::vector<uint32_t> tr;
    std::vector<uint16_t> suh, svh;
    int k, n, k2, cb;
};
// random words a tile (4 k2 of them: any bit pattern is a valid encoding, a half-integer K too)
HostMat make_mat(int k, int n, int k2, int cb) {
    HostMat m{rand_words((size_t) (k / 16) * (n / 16) * 4 * k2), rand_scales(k), rand_scales(n), k, n, k2, cb};
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

void test_mv(int k, int n, int k2, int cb, bool q8, int view_c0 = -1, int view_n = 0) {
    HostMat h = make_mat(k, n, k2, cb);
    uint32_t* d_tr = dev_copy(h.tr);
    uint16_t* d_suh = dev_copy(h.suh);
    uint16_t* d_svh = dev_copy(h.svh);
    ex::Mat m;
    m.trellis = d_tr;
    m.suh = d_suh;
    m.svh = d_svh;
    m.k = k;
    m.n = n;
    m.k2 = k2;
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
    cpu::exl3_inner_k2(h.tr.data(), h.n / 16, k, h.n, k2, cb, w.data());
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
    const std::string what = "mv k" + std::to_string(k) + " n" + std::to_string(nc) + " K" + kname(k2) +
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

void test_batch(int k2a, int k2c) {
    // three jobs, two bit widths: the launch per (K, cb) and the per-job scratch ranges
    const int k = 8192;
    HostMat a = make_mat(k, 512, k2a, 2), b = make_mat(k, 256, k2a, 2), c = make_mat(1024, 384, k2c, 2);
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
        m[i].k2 = hm[i]->k2;
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
        cpu::exl3_inner_k2(hm[i]->tr.data(), hm[i]->n / 16, hm[i]->k, hm[i]->n, hm[i]->k2, 2, w.data());
        cpu::exl3_mv(w.data(), hm[i]->suh.data(), hm[i]->svh.data(), hm[i]->k, hm[i]->n, x.data(), ref.data());
        close(("batch job " + std::to_string(i)).c_str(), host_copy(d_y[(size_t) i], (size_t) hm[i]->n), ref, 2e-3);
    }
}

// kd: the down matrix's bitrate when it differs from gate / up's (an SC model's layer, e.g. 3.5 / 3.5 / 5)
std::string kname3(int k2, int kd) { return kname(k2) + (kd != k2 ? "/" + kname(kd) : ""); }

void test_moe(int k2, int cb, int kd = 0) {
    if (kd <= 0) kd = k2;
    const int E = 512, F = 256, k = 3;
    const float limit = 1.5f;
    // blob: gate [suh | svh | trellis], up [...], down [...] (4-byte offsets, like the safetensors ranges)
    HostMat parts[3][3];
    ex::ExpertLayout lay;
    lay.cb = cb;
    size_t at = 0;
    std::vector<std::vector<uint8_t>> blobs(3);
    for (int e = 0; e < 3; ++e) {
        parts[e][0] = make_mat(E, F, k2, cb);
        parts[e][1] = make_mat(E, F, k2, cb);
        parts[e][2] = make_mat(F, E, kd, cb);
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
            lay.k2[r] = r < 2 ? k2 : kd;
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
            cpu::exl3_inner_k2(p.tr.data(), F / 16, E, F, k2, cb, w.data());
            cpu::exl3_mv(w.data(), p.suh.data(), p.svh.data(), E, F, x.data(), r == 0 ? g.data() : u.data());
        }
        for (int i = 0; i < F; ++i) {
            const float gv = std::min(g[(size_t) i], limit), uv = std::min(std::max(u[(size_t) i], -limit), limit);
            hh[(size_t) i] = gv / (1.0f + std::exp(-gv)) * uv;
        }
        const HostMat& p = parts[e][2];
        std::vector<float> w((size_t) F * E);
        cpu::exl3_inner_k2(p.tr.data(), E / 16, F, E, kd, cb, w.data());
        cpu::exl3_mv(w.data(), p.suh.data(), p.svh.data(), F, E, hh.data(), dn.data());
        for (int i = 0; i < E; ++i) ref[(size_t) i] += pw[(size_t) e] * dn[(size_t) i];
    }
    for (int i = 0; i < E; ++i) ref[(size_t) i] += cpu_part[(size_t) i] + sh[(size_t) i];
    close(("moe K" + kname3(k2, kd) + " cb" + std::to_string(cb)).c_str(), host_copy(d_out, (size_t) E), ref, 3e-3);
}

void test_recon(int k, int n, int k2, int cb) {
    HostMat h = make_mat(k, n, k2, cb);
    ex::Mat m;
    m.trellis = dev_copy(h.tr);
    m.suh = dev_copy(h.suh);
    m.svh = dev_copy(h.svh);
    m.k = k;
    m.n = n;
    m.k2 = k2;
    m.cb = cb;
    std::vector<float> w((size_t) k * n);
    cpu::exl3_inner_k2(h.tr.data(), n / 16, k, n, k2, cb, w.data());
    // inner: exact FP16 values
    uint16_t* d_in = nullptr;
    CK(cudaMalloc(&d_in, (size_t) k * n * 2));
    ex::reconstruct_inner(m, d_in, 0);
    CK(cudaDeviceSynchronize());
    const std::vector<uint16_t> in = host_copy(d_in, (size_t) k * n);
    size_t bad = 0;
    for (size_t i = 0; i < in.size(); ++i) bad += f32_from_f16(in[i]) != w[i];
    std::printf("  %-44s %zu of %zu differ  %s\n",
                ("inner k" + std::to_string(k) + " n" + std::to_string(n) + " K" + kname(k2) + " cb" +
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
               " K" + kname(k2)).c_str(), got, full, bf ? 6e-3 : 2e-3);
    }
}

// the prompt path's rotated product: W' alone in the row layout (exact), the inputs rotated (rows_rot_f16, gathered
// by index), the GEMM on the host, the outputs rotated (rows_rot_out) - against the reference's x W
void test_rot_rows(int k, int n, int k2, int cb) {
    HostMat h = make_mat(k, n, k2, cb);
    ex::Mat m;
    m.trellis = dev_copy(h.tr);
    m.suh = dev_copy(h.suh);
    m.svh = dev_copy(h.svh);
    m.k = k;
    m.n = n;
    m.k2 = k2;
    m.cb = cb;
    std::vector<float> w((size_t) k * n);
    cpu::exl3_inner_k2(h.tr.data(), n / 16, k, n, k2, cb, w.data());
    uint16_t* d_w = nullptr;
    CK(cudaMalloc(&d_w, (size_t) k * n * 2));
    ex::reconstruct_inner_rows(m, d_w, 0);
    CK(cudaDeviceSynchronize());
    const std::vector<uint16_t> wt = host_copy(d_w, (size_t) k * n);   // [n][k]
    size_t bad = 0;
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < k; ++r) bad += f32_from_f16(wt[(size_t) c * k + r]) != w[(size_t) r * n + c];
    const std::string sh = " k" + std::to_string(k) + " n" + std::to_string(n) + " K" + kname(k2) + " cb" +
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
void test_gemm_rows(int k, int n, int k2, int cb, int rows) {   // (every GPU: before sm_80 the portable kernels)
    HostMat h = make_mat(k, n, k2, cb);
    ex::Mat m;
    m.trellis = dev_copy(h.tr);
    m.suh = dev_copy(h.suh);
    m.svh = dev_copy(h.svh);
    m.k = k;
    m.n = n;
    m.k2 = k2;
    m.cb = cb;
    std::vector<float> w((size_t) k * n);
    cpu::exl3_inner_k2(h.tr.data(), n / 16, k, n, k2, cb, w.data());
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
        close(("gemm_rows k" + std::to_string(k) + " n" + std::to_string(n) + " K" + kname(k2) + " cb" +
               std::to_string(cb) + " rows" + std::to_string(rows) + " beta" + std::to_string(beta)).c_str(),
              host_copy(d_y, want.size()), want, 3e-3);
    }
}

// the prompt path's expert set: 3 experts (rows 37, 0, 70) through set_gate_up and set_down against the reference
void test_set(int k2, int cb, int kd = 0) {
    if (kd <= 0) kd = k2;   // (every GPU: before sm_80 the portable kernels)
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
        at += ex::trellis_bytes_k2(k, n, r < 2 ? k2 : kd);
        lay.suh[r] = at;
        at += 2 * (size_t) k;
        lay.svh[r] = at;
        at += 2 * (size_t) n;
        lay.k2[r] = r < 2 ? k2 : kd;
    }
    const size_t stride = (at + 255) / 256 * 256;
    std::vector<uint8_t> blobs(3 * stride);
    for (int e = 0; e < 3; ++e)
        for (int r = 0; r < 3; ++r) {
            parts[e][r] = make_mat(r < 2 ? E : F, r < 2 ? F : E, r < 2 ? k2 : kd, cb);
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
        cpu::exl3_inner_k2(parts[e][0].tr.data(), F / 16, E, F, k2, cb, wg.data());
        cpu::exl3_inner_k2(parts[e][1].tr.data(), F / 16, E, F, k2, cb, wu.data());
        cpu::exl3_inner_k2(parts[e][2].tr.data(), E / 16, F, E, kd, cb, wd.data());
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
    close(("set K" + kname3(k2, kd) + " cb" + std::to_string(cb) + " (3 experts, 107 rows)").c_str(),
          host_copy(d_out, ref.size()), ref, 3e-3);
}

// a mixed-precision layer: 3 experts each at its own K for gate / up / down, each in its own allocation
// (its class), through the token path (moe_gate_up + moe_down, a plan of the 3) and the prompt path (set_gate_up +
// set_down with an address table) against the reference
void test_mixed(int cb) {
    const int E = 512, F = 256, T = 64;
    const float limit = 1.5f;
    const int ks[3][3] = {{2, 2, 4}, {4, 2, 6}, {6, 8, 2}};   // (gate, up, down) K2 of each expert
    const int counts[3] = {37, 5, 70};
    HostMat parts[3][3];
    ex::MixedLayouts mix;
    mix.n = 3;
    std::vector<uint8_t*> d_blob(3);
    std::vector<unsigned long long> eptr(3);
    for (int e = 0; e < 3; ++e) {
        ex::ExpertLayout& lay = mix.lay[e];
        lay.cb = cb;
        size_t at = 0;
        for (int r = 0; r < 3; ++r) {
            const int k = r < 2 ? E : F, n = r < 2 ? F : E;
            lay.k2[r] = ks[e][r];
            lay.trellis[r] = at;
            at += (ex::trellis_bytes_k2(k, n, ks[e][r]) + 255) / 256 * 256;
            lay.suh[r] = at;
            at += (2 * (size_t) k + 255) / 256 * 256;
            lay.svh[r] = at;
            at += (2 * (size_t) n + 255) / 256 * 256;
        }
        std::vector<uint8_t> blob(at);
        for (int r = 0; r < 3; ++r) {
            parts[e][r] = make_mat(r < 2 ? E : F, r < 2 ? F : E, ks[e][r], cb);
            const HostMat& p = parts[e][r];
            std::memcpy(&blob[lay.trellis[r]], p.tr.data(), p.tr.size() * 4);
            std::memcpy(&blob[lay.suh[r]], p.suh.data(), p.suh.size() * 2);
            std::memcpy(&blob[lay.svh[r]], p.svh.data(), p.svh.size() * 2);
        }
        d_blob[(size_t) e] = dev_copy(blob);
        eptr[(size_t) e] = (unsigned long long) d_blob[(size_t) e];
        mix.lo[e] = eptr[(size_t) e];
        mix.hi[e] = eptr[(size_t) e] + at;
    }
    // the reference's weights
    std::vector<float> w[3][3];
    for (int e = 0; e < 3; ++e)
        for (int r = 0; r < 3; ++r) {
            const int k = r < 2 ? E : F, n = r < 2 ? F : E;
            w[e][r].resize((size_t) k * n);
            cpu::exl3_inner_k2(parts[e][r].tr.data(), n / 16, k, n, ks[e][r], cb, w[e][r].data());
        }
    auto expert = [&](int e, const float* xr, bool h16, float* out) {   // one expert's output for one row
        std::vector<float> gg((size_t) F), uu((size_t) F), hh((size_t) F);
        cpu::exl3_mv(w[e][0].data(), parts[e][0].suh.data(), parts[e][0].svh.data(), E, F, xr, gg.data());
        cpu::exl3_mv(w[e][1].data(), parts[e][1].suh.data(), parts[e][1].svh.data(), E, F, xr, uu.data());
        for (int i = 0; i < F; ++i) {
            const float gv = std::min(gg[(size_t) i], limit), uv = std::min(std::max(uu[(size_t) i], -limit), limit);
            const float v = gv / (1.0f + std::exp(-gv)) * uv;
            hh[(size_t) i] = h16 ? f32_from_f16(f16_from_f32(v)) : v;
        }
        cpu::exl3_mv(w[e][2].data(), parts[e][2].suh.data(), parts[e][2].svh.data(), F, E, hh.data(), out);
    };
    // the token path: one token, a plan of the 3 experts
    {
        std::vector<float> x = rand_x(E), pw = {0.5f, 0.3f, 0.2f}, ref((size_t) E, 0.0f), o((size_t) E);
        for (int e = 0; e < 3; ++e) {
            expert(e, x.data(), false, o.data());
            for (int i = 0; i < E; ++i) ref[(size_t) i] += pw[(size_t) e] * o[(size_t) i];
        }
        unsigned long long* d_plan = dev_copy(eptr);
        float* d_pw = dev_copy(pw);
        float* d_x = dev_copy(x);
        float* d_out = nullptr;
        CK(cudaMalloc(&d_out, (size_t) E * 4));
        ex::MoeArgs ma;
        ma.plan_ptr = d_plan;
        ma.plan_w = d_pw;
        ma.mix = mix;
        ma.k = 3;
        ma.n_embd = E;
        ma.n_ff = F;
        ma.limit = limit;
        ex::moe_gate_up(ma, d_x, nullptr, 0);
        ex::moe_down(ma, nullptr, nullptr, d_out, 0);
        CK(cudaDeviceSynchronize());
        close(("moe mixed cb" + std::to_string(cb) + " (3 experts, K 1/1/2, 2/1/3, 3/4/1)").c_str(),
              host_copy(d_out, (size_t) E), ref, 3e-3);
        cudaFree(d_plan);
        cudaFree(d_pw);
        cudaFree(d_x);
        cudaFree(d_out);
    }
    // the prompt path: the sorted rows of the 3 experts, the set's addresses in a table
    {
        std::vector<float> x = rand_x(T * E);
        std::vector<int> bounds = {0}, row_tok;
        for (int e = 0; e < 3; ++e) {
            for (int j = 0; j < counts[e]; ++j) row_tok.push_back((e * 7 + j) % T);
            bounds.push_back((int) row_tok.size());
        }
        const int rows = (int) row_tok.size();
        unsigned long long* d_eptr = dev_copy(eptr);
        float* d_x = dev_copy(x);
        int* d_bounds = dev_copy(bounds);
        int* d_rows = dev_copy(row_tok);
        uint16_t* d_h = nullptr;
        float* d_out = nullptr;
        CK(cudaMalloc(&d_h, (size_t) rows * F * 2));
        CK(cudaMalloc(&d_out, (size_t) rows * E * 4));
        ex::SetArgs sa;
        sa.mix = mix;
        sa.eptr = d_eptr;
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
        for (int e = 0, r = 0; e < 3; ++e)
            for (int j = 0; j < counts[e]; ++j, ++r)
                expert(e, x.data() + (size_t) row_tok[(size_t) r] * E, true, ref.data() + (size_t) r * E);
        const std::vector<float> got = host_copy(d_out, ref.size());
        close(("set mixed cb" + std::to_string(cb) + " (3 experts, 112 rows)").c_str(), got, ref, 3e-3);
        {   // (the outputs' bits, FNV-1a: a build before and after a change to set_mix_kernel compare them - #479)
            uint64_t hb = 1469598103934665603ull;
            const std::vector<uint16_t> h16 = host_copy(d_h, (size_t) rows * F);
            const uint8_t* ob = (const uint8_t*) got.data();
            const uint8_t* hbp = (const uint8_t*) h16.data();
            for (size_t i = 0; i < got.size() * 4; ++i) hb = (hb ^ ob[i]) * 1099511628211ull;
            for (size_t i = 0; i < h16.size() * 2; ++i) hb = (hb ^ hbp[i]) * 1099511628211ull;
            std::printf("  set mixed cb%d bits %016llx\n", cb, (unsigned long long) hb);
        }
        cudaFree(d_eptr);
        cudaFree(d_x);
        cudaFree(d_bounds);
        cudaFree(d_rows);
        cudaFree(d_h);
        cudaFree(d_out);
    }
    for (uint8_t* b : d_blob) cudaFree(b);
}

// mv with several activations (a verify window's tokens) against one mv job per activation: the same bits
void test_mv_n(int k, int n, int k2, int cb, bool q8, int ncol) {
    HostMat h = make_mat(k, n, k2, cb);
    ex::Mat m;
    m.trellis = dev_copy(h.tr);
    m.suh = dev_copy(h.suh);
    m.svh = dev_copy(h.svh);
    m.k = k;
    m.n = n;
    m.k2 = k2;
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
                             " K" + kname(k2) + " cb" + std::to_string(cb) + (q8 ? " q8_1" : " f32");
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
// mixed: a mixed-precision layer instead - each expert its own class, its gate / up / down at K 1-4 by expert and part
void test_win(int k2, int cb, int T, int kd = 0, bool mixed = false) {
    if (kd <= 0) kd = k2;
    const int E = 512, F = 384, k = 4, NX = 5;
    const float limit = 1.5f;
    ex::ExpertLayout lay;
    lay.cb = cb;
    ex::MixedLayouts mix;
    if (mixed) mix.n = NX;
    std::vector<uint8_t*> d_blob((size_t) NX);
    for (int e = 0; e < NX; ++e) {
        std::vector<uint8_t> b;
        size_t at = 0;
        for (int r = 0; r < 3; ++r) {
            const int kr = mixed ? 2 * (1 + (e + 2 * r) % 4) : r < 2 ? k2 : kd;
            const HostMat p = r < 2 ? make_mat(E, F, kr, cb) : make_mat(F, E, kr, cb);
            const size_t o_suh = at, o_svh = o_suh + p.suh.size() * 2, o_tr = o_svh + p.svh.size() * 2 + 4;
            at = o_tr + p.tr.size() * 4;
            b.resize(at);
            std::memcpy(&b[o_suh], p.suh.data(), p.suh.size() * 2);
            std::memcpy(&b[o_svh], p.svh.data(), p.svh.size() * 2);
            std::memcpy(&b[o_tr], p.tr.data(), p.tr.size() * 4);
            lay.suh[r] = o_suh;
            lay.svh[r] = o_svh;
            lay.trellis[r] = o_tr;
            lay.k2[r] = kr;
        }
        d_blob[(size_t) e] = dev_copy(b);
        if (mixed) {
            mix.lay[e] = lay;
            mix.lo[e] = (unsigned long long) d_blob[(size_t) e];
            mix.hi[e] = mix.lo[e] + b.size();
        }
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
        a.mix = mix;
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
    w.mix = mix;
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
    const std::string what = (mixed ? std::string("win mixed K1-4") : "win K" + kname3(k2, kd)) + " cb" +
                             std::to_string(cb) + " " + std::to_string(T) + " tokens, " + std::to_string(gp.size()) +
                             " experts";
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
void test_cpu_n(int k, int n, int k2) {
    HostMat h = make_mat(k, n, k2, 2);
    cpu::Exl3Part m;
    m.trellis = h.tr.data();
    m.suh = h.suh.data();
    m.svh = h.svh.data();
    m.k = k;
    m.n = n;
    m.k2 = k2;
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
    const std::string what = "cpu x2..11 k" + std::to_string(k) + " n" + std::to_string(n) + " K" + kname(k2);
    std::printf("  %-44s %zu differ from one call each  %s\n", what.c_str(), diff, diff == 0 ? "ok" : "FAIL");
    if (diff != 0) ++g_fail;
}

// the CPU lane's split-K pieces (every column, slices [ks0, ks1)) against the bands over the same slices, and for
// several activations at once against one call each: the same bits
void test_cpu_rows(int k, int n, int k2) {
    HostMat h = make_mat(k, n, k2, 2);
    cpu::Exl3Part m;
    m.trellis = h.tr.data();
    m.suh = h.suh.data();
    m.svh = h.svh.data();
    m.k = k;
    m.n = n;
    m.k2 = k2;
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
        // the same piece cut into runs of columns (uneven: 1..3 bands a run, the lane's cuts): the uncut call's bits
        std::fill(all.begin(), all.end(), -1.0f);
        for (int c0 = 0; c0 < n;) {
            const int c1 = std::min(n, c0 + 128 * (1 + (c0 / 128 + nt) % 3));
            cpu::exl3_cpu_rows_acc_cols_n(m, xp, nt, ks0, ks1, c0, c1, ap);
            c0 = c1;
        }
        for (size_t i = 0; i < (size_t) nt * n; ++i) diff += std::memcmp(&one[i], &all[i], sizeof(float)) != 0;
    }
    const std::string what = "cpu rows x1..11 k" + std::to_string(k) + " n" + std::to_string(n) + " K" + kname(k2);
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
void test_cpu_gemm(int k, int n, int k2) {
    HostMat h = make_mat(k, n, k2, 2);
    cpu::Exl3Part m;
    m.trellis = h.tr.data();
    m.suh = h.suh.data();
    m.svh = h.svh.data();
    m.k = k;
    m.n = n;
    m.k2 = k2;
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
    const std::string what = "cpu gemm x21 nb1,2 k" + std::to_string(k) + " n" + std::to_string(n) + " K" + kname(k2);
    const bool ok = worst < 1e-5;
    std::printf("  %-44s worst %.2e of the band's largest  %s\n", what.c_str(), worst, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

const char* cpu_tier_name(int tier) { return tier == 2 ? "avx512" : tier == 1 ? "avx2" : "scalar"; }

// The vector tiers' arithmetic, written out on its own: column c's state at row r is the 16-bit window that ends at
// bit (p + 1 + 256) K of the tile's MSB-first bit ring (p: the tile's step at (r, c) - row (p/8 % 4) 2 + (p & 1) +
// 8 (p/2 & 1), column p/32 + 8 (p/4 & 1)), read bit by bit; its mul1 byte sum s; and one fma(s, xh, acc) a weight in
// (slice, row) order
// step p's state, read bit by bit from the tile's MSB-first bit ring: the window that ends at bit floor((p + 1) k2 / 2)
// ((p + 1) K for an integer K) of the ring's 128 k2 bits
uint32_t model_state(const uint32_t* tile, int k2, int p) {
    const int bits = 128 * k2, end = ((p + 1) * k2) / 2 + bits;
    uint32_t st = 0;
    for (int q = end - 16; q < end; ++q) {
        const int b = q % bits;
        st = (st << 1) | ((tile[b >> 5] >> (31 - (b & 31))) & 1u);
    }
    return st;
}
float exact_band_col(const cpu::Exl3Part& m, const float* xh, int col, int ks0, int ks1) {
    static int step[16][16];
    static bool init = false;
    if (!init) {
        for (int p = 0; p < 256; ++p) step[(p / 8 % 4) * 2 + (p & 1) + 8 * (p / 2 & 1)][p / 32 + 8 * (p / 4 & 1)] = p;
        init = true;
    }
    const int t = col / 16, c = col % 16;
    float a = 0.0f;
    for (int ks = ks0; ks < ks1; ++ks) {
        const uint32_t* tile = m.trellis + ((size_t) ks * (m.n / 16) + t) * 4 * m.k2;
        for (int r = 0; r < 16; ++r) {
            const uint32_t st = model_state(tile, m.k2, step[r][c]);
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
void test_cpu_exact(int k, int n, int k2) {
    HostMat h = make_mat(k, n, k2, 2);
    cpu::Exl3Part m;
    m.trellis = h.tr.data();
    m.suh = h.suh.data();
    m.svh = h.svh.data();
    m.k = k;
    m.n = n;
    m.k2 = k2;
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
                                 std::to_string(n) + " K" + kname(k2);
        std::printf("  %-44s %zu of %zu differ from the model  %s\n", what.c_str(), diff[tier], cmp[tier],
                    diff[tier] == 0 ? "ok" : "FAIL");
        if (diff[tier] != 0) ++g_fail;
    }
    const double rel = std::sqrt(se / std::max(sr, 1e-30));
    const std::string what = "cpu exact scalar k" + std::to_string(k) + " n" + std::to_string(n) + " K" + kname(k2);
    const bool ok = cmp[0] > 0 && rel <= 1e-5;
    std::printf("  %-44s rel %.2e to the model  %s\n", what.c_str(), rel, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// the CPU lane's kernel (mul1) against the reference: every band of a k x n matrix
void test_cpu(int k, int n, int k2) {
    HostMat h = make_mat(k, n, k2, 2);
    std::vector<float> x = rand_x(k, 2.0f), w((size_t) k * n), ref((size_t) n), got((size_t) n), xh((size_t) k);
    cpu::exl3_inner_k2(h.tr.data(), n / 16, k, n, k2, 2, w.data());
    cpu::exl3_mv(w.data(), h.suh.data(), h.svh.data(), k, n, x.data(), ref.data());
    cpu::Exl3Part m;
    m.trellis = h.tr.data();
    m.suh = h.suh.data();
    m.svh = h.svh.data();
    m.k = k;
    m.n = n;
    m.k2 = k2;
    const float xs = cpu::exl3_cpu_prepare(m, x.data(), xh.data());
    for (int c0 = 0; c0 < n; c0 += 128) cpu::exl3_cpu_band(m, xh.data(), xs, c0, got.data() + c0);
    close(("cpu " + std::string(cpu_tier_name(cpu::exl3_cpu_tier())) + " k" + std::to_string(k) + " n" +
           std::to_string(n) + " K" + kname(k2)).c_str(), got, ref, 2e-3);
}


// exllamav3 writes a half-integer K with mul1 only: every other codebook is refused by name and nothing is launched
void test_refuse_half_non_mul1() {
    const HostMat h = make_mat(256, 256, 7, 2);
    ex::Mat m;
    m.trellis = dev_copy(h.tr);
    m.suh = dev_copy(h.suh);
    m.svh = dev_copy(h.svh);
    m.k = 256;
    m.n = 256;
    m.k2 = 7;
    int bad = 0;
    for (int cb = 0; cb <= 2; ++cb) {
        m.cb = cb;
        const bool want = cb == 2;
        bad += ex::k2_supported(7, cb) != want || ex::supported(m) != want || cpu::exl3_k2_valid(7, cb) != want;
    }
    bad += ex::k2_supported(9, 2) || ex::k2_supported(1, 2) || ex::k2_supported(18, 2) || cpu::exl3_cpu_k2_ok(9) ||
           !cpu::exl3_cpu_k2_ok(7);
    std::vector<float> x = rand_x(256);
    float* d_x = dev_copy(x);
    float* d_y = nullptr;
    CK(cudaMalloc(&d_y, 256 * sizeof(float)));
    m.cb = 1;   // K 3.5 with mcg: mv refuses the job
    ex::MvJob j;
    j.m = &m;
    j.xf = d_x;
    j.y = d_y;
    bad += ex::mv(&j, 1, 0);
    CK(cudaDeviceSynchronize());
    std::printf("  %-44s %s\n", "K 3.5 refused with 3inst / mcg, taken with mul1", bad ? "FAIL" : "ok");
    if (bad) ++g_fail;
    cudaFree(d_x);
    cudaFree(d_y);
}

// ---------------------------------------------------------------- against exllamav3 itself
// tools/exl3_exllamav3_fixture.py writes what exllamav3 makes of EXL3 tiles - random states packed by its packer, each
// step's state as its unpacker reads it, W' as its reconstruct decodes it (data/exl3-exllamav3) - and of a model's
// matrices (--model).  Every decoder here is held to the same bits: the CPU reference (states, W'), the exact model
// the CPU tiers are tested against (its bit-by-bit window), the CPU lane (bit for bit that model on exllamav3's
// tiles), the device's W' in both layouts.

bool read_bytes(const std::string& path, std::vector<uint8_t>& b) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    b.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return !f.bad();
}

void report(const std::string& what, size_t bad, size_t n) {
    std::printf("  %-44s %zu of %zu differ  %s\n", what.c_str(), bad, n, bad || n == 0 ? "FAIL" : "ok");
    if (bad || n == 0) ++g_fail;
}

struct Fixture {
    int k2 = 0, ki = 0, nj = 0, cb = 0, words = 0;
    std::vector<uint32_t> tr;   // [ki][nj][words / 2]: the stored int16 pairs as the engine reads them
    std::vector<uint16_t> st;   // [ki][nj][256]
    std::vector<uint16_t> wh;   // [16 ki][16 nj] FP16 bits
};

bool read_fixture(const std::string& path, Fixture& f, std::string& err) {
    std::vector<uint8_t> b;
    if (!read_bytes(path, b)) {
        err = "cannot read it";
        return false;
    }
    if (b.size() < 24 || std::memcmp(b.data(), "EXH1", 4) != 0) {
        err = "not an EXH1 fixture";
        return false;
    }
    int32_t h[5];
    std::memcpy(h, b.data() + 4, sizeof(h));
    f.k2 = h[0], f.ki = h[1], f.nj = h[2], f.cb = h[3], f.words = h[4];
    if (!cpu::exl3_k2_valid(f.k2, f.cb) || f.words != 8 * f.k2 || f.ki < 1 || f.nj < 1 || f.nj % 8 != 0) {
        err = "bad header (k2 " + std::to_string(f.k2) + ", " + std::to_string(f.ki) + " x " + std::to_string(f.nj) +
              " tiles, cb " + std::to_string(f.cb) + ", " + std::to_string(f.words) + " words a tile)";
        return false;
    }
    const size_t tiles = (size_t) f.ki * f.nj, n_tr = tiles * (size_t) f.words * 2, n_st = tiles * 512;
    if (b.size() != 24 + n_tr + 2 * n_st) {
        err = std::to_string(b.size()) + " bytes, want " + std::to_string(24 + n_tr + 2 * n_st);
        return false;
    }
    f.tr.resize(n_tr / 4);
    f.st.resize(tiles * 256);
    f.wh.resize(tiles * 256);
    std::memcpy(f.tr.data(), b.data() + 24, n_tr);
    std::memcpy(f.st.data(), b.data() + 24 + n_tr, n_st);
    std::memcpy(f.wh.data(), b.data() + 24 + n_tr + n_st, n_st);
    return true;
}

// W' of m on the device in both layouts against want ([k][n] FP16 bits): reconstruct_inner (row-major) and
// reconstruct_inner_rows (dst[c][r])
void device_w_hat(const ex::Mat& m, const std::vector<uint16_t>& want, const std::string& tag) {
    const size_t kn = (size_t) m.k * m.n;
    uint16_t* d = nullptr;
    CK(cudaMalloc(&d, kn * 2));
    ex::reconstruct_inner(m, d, 0);
    CK(cudaDeviceSynchronize());
    std::vector<uint16_t> got = host_copy(d, kn);
    size_t bad = 0;
    for (size_t i = 0; i < kn; ++i) bad += got[i] != want[i];
    report("device W' (inner) " + tag, bad, kn);
    ex::reconstruct_inner_rows(m, d, 0);
    CK(cudaDeviceSynchronize());
    got = host_copy(d, kn);
    bad = 0;
    for (int r = 0; r < m.k; ++r)
        for (int c = 0; c < m.n; ++c) bad += got[(size_t) c * m.k + r] != want[(size_t) r * m.n + c];
    report("device W' (inner rows) " + tag, bad, kn);
    CK(cudaFree(d));
}

// the CPU tiers this CPU runs against the exact model on bands [c0, c0 + 128) of m, 1 and 3 activations at once
void lane_vs_model(const cpu::Exl3Part& m, const std::vector<int>& bands, const std::string& tag) {
    constexpr int NT = 3;
    std::vector<std::vector<float>> xh;
    const float* xp[NT];
    for (int i = 0; i < NT; ++i) xh.push_back(rand_x(m.k));
    for (int i = 0; i < NT; ++i) xp[i] = xh[(size_t) i].data();
    std::vector<float> want((size_t) NT * 128), got((size_t) NT * 128);
    float* gp[NT];
    for (int i = 0; i < NT; ++i) gp[i] = got.data() + (size_t) i * 128;
    for (int tier = 1; tier <= 2; ++tier) {
        size_t bad = 0, cmp = 0;
        for (int c0 : bands) {
            for (int i = 0; i < NT; ++i)
                for (int j = 0; j < 128; ++j) want[(size_t) i * 128 + j] = exact_band_col(m, xp[i], c0 + j, 0, m.k / 16);
            for (int nt : {1, NT}) {
                if (!cpu::exl3_cpu_band_acc_on(tier, m, xp, nt, c0, 0, m.k / 16, gp)) break;
                for (size_t i = 0; i < (size_t) nt * 128; ++i, ++cmp) bad += std::memcmp(&got[i], &want[i], 4) != 0;
            }
        }
        if (cmp) report(std::string("cpu ") + cpu_tier_name(tier) + " vs the model " + tag, bad, cmp);
    }
}

void test_exllamav3_fixture(const std::string& path) {
    Fixture f;
    std::string err;
    const std::string name = std::filesystem::path(path).filename().string();
    if (!read_fixture(path, f, err)) {
        std::printf("  %-44s %s  FAIL\n", name.c_str(), err.c_str());
        ++g_fail;
        return;
    }
    const int k2 = f.k2, W = 4 * k2, k = 16 * f.ki, n = 16 * f.nj;
    const std::string tag = "K" + kname(k2) + " cb" + std::to_string(f.cb);
    // the states: the CPU reference's and the exact model's
    size_t bad_ref = 0, bad_model = 0;
    uint16_t st[256];
    for (size_t t = 0; t < (size_t) f.ki * f.nj; ++t) {
        cpu::exl3_tile_states_k2(f.tr.data() + t * W, k2, st);
        for (int p = 0; p < 256; ++p) {
            bad_ref += st[p] != f.st[t * 256 + p];
            bad_model += model_state(f.tr.data() + t * W, k2, p) != f.st[t * 256 + p];
        }
    }
    report("exllamav3 states: CPU reference " + tag, bad_ref, f.st.size());
    report("exllamav3 states: the CPU tiers' model " + tag, bad_model, f.st.size());
    // the CPU reference's W'
    std::vector<float> w((size_t) k * n);
    cpu::exl3_inner_k2(f.tr.data(), f.nj, k, n, k2, f.cb, w.data());
    size_t bad = 0;
    for (size_t i = 0; i < w.size(); ++i) bad += f16_from_f32(w[i]) != f.wh[i];
    report("exllamav3 W': CPU reference " + tag, bad, w.size());
    // the device and the CPU lane take k a multiple of 128: the tile rows repeated
    const int reps = 128 / std::gcd(k, 128), kk = reps * k;
    std::vector<uint32_t> big;
    std::vector<uint16_t> want;
    for (int r = 0; r < reps; ++r) {
        big.insert(big.end(), f.tr.begin(), f.tr.end());
        want.insert(want.end(), f.wh.begin(), f.wh.end());
    }
    const std::vector<uint16_t> one_k((size_t) kk, (uint16_t) 0x3c00), one_n((size_t) n, (uint16_t) 0x3c00);
    ex::Mat m;
    m.trellis = dev_copy(big);
    m.suh = dev_copy(one_k);
    m.svh = dev_copy(one_n);
    m.k = kk;
    m.n = n;
    m.k2 = k2;
    m.cb = f.cb;
    device_w_hat(m, want, tag + " k" + std::to_string(kk));
    if (f.cb == 2) {   // the CPU lane decodes mul1
        cpu::Exl3Part c;
        c.trellis = big.data();
        c.suh = one_k.data();
        c.svh = one_n.data();
        c.k = kk;
        c.n = n;
        c.k2 = k2;
        std::vector<int> bands;
        for (int c0 = 0; c0 < n; c0 += 128) bands.push_back(c0);
        lane_vs_model(c, bands, tag + " k" + std::to_string(kk));
    }
}

void test_exllamav3(const std::string& dir) {
    std::vector<std::string> files;
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
        const std::string fn = e.path().filename().string();
        if (fn.rfind("exl3_k2_", 0) == 0 && e.path().extension() == ".bin") files.push_back(e.path().string());
    }
    const auto key = [](const std::string& path) {   // exl3_k2_<k2>_cb<cb>.bin: by bitrate, then codebook
        int k2 = 0, cb = 0;
        std::sscanf(std::filesystem::path(path).filename().string().c_str(), "exl3_k2_%d_cb%d", &k2, &cb);
        return std::make_pair(k2, cb);
    };
    std::sort(files.begin(), files.end(), [&](const std::string& a, const std::string& b) { return key(a) < key(b); });
    std::printf("exllamav3 fixtures in %s: %zu\n", dir.c_str(), files.size());
    if (files.empty()) {
        std::printf("  no exl3_k2_*.bin fixtures there (tools/exl3_exllamav3_fixture.py writes them)  FAIL\n");
        ++g_fail;
    }
    for (const auto& f : files) test_exllamav3_fixture(f);
}

// a model's matrices against exllamav3's decode (the fixture tool's --model files: <base>.trellis.i16 / .suh.f16 /
// .svh.f16 / .what.f16 / .w.f16, manifest lines "<base> <k2> <cb> <k> <n>"): W' bit for bit from the CPU reference and
// the device (both layouts); the full weight against exllamav3's reconstruct_had_slice (its fused kernel rounds
// otherwise, so within FP16 rounding); the CPU lane against the reference's product and bit for bit its model
template <typename T>
bool read_raw(const std::string& path, size_t n, std::vector<T>& v) {
    std::vector<uint8_t> b;
    if (!read_bytes(path, b) || b.size() != n * sizeof(T)) {
        std::printf("  %-44s missing or %zu bytes, want %zu  FAIL\n", path.c_str(), b.size(), n * sizeof(T));
        ++g_fail;
        return false;
    }
    v.resize(n);
    std::memcpy(v.data(), b.data(), b.size());
    return true;
}

// W = diag(suh) H W' H diag(svh) [k][n] in F64 (H: the 128-point blocks over sqrt(128) along each axis)
std::vector<double> full_weight_f64(const std::vector<float>& w, const std::vector<uint16_t>& suh,
                                    const std::vector<uint16_t>& svh, int k, int n) {
    std::vector<double> t(w.begin(), w.end());
    const double s = 1.0 / std::sqrt(128.0);
    auto had = [s](double* v, size_t stride) {   // one 128-value block, values stride apart
        for (int h = 1; h < 128; h <<= 1)
            for (int i = 0; i < 128; i += 2 * h)
                for (int j = i; j < i + h; ++j) {
                    const double a = v[(size_t) j * stride], c = v[(size_t) (j + h) * stride];
                    v[(size_t) j * stride] = a + c;
                    v[(size_t) (j + h) * stride] = a - c;
                }
        for (int i = 0; i < 128; ++i) v[(size_t) i * stride] *= s;
    };
    for (int r = 0; r < k; ++r)
        for (int c0 = 0; c0 < n; c0 += 128) had(t.data() + (size_t) r * n + c0, 1);
    for (int r0 = 0; r0 < k; r0 += 128)
        for (int c = 0; c < n; ++c) had(t.data() + (size_t) r0 * n + c, (size_t) n);
    for (int r = 0; r < k; ++r)
        for (int c = 0; c < n; ++c)
            t[(size_t) r * n + c] *= (double) f32_from_f16(suh[(size_t) r]) * (double) f32_from_f16(svh[(size_t) c]);
    return t;
}

// FP16 values against an exact reference: the relative RMS error, and the worst error over the reference's largest
// magnitude (what FP16 rounding of the values and of intermediates of their size is bounded by: 2^-11 of it each)
struct ErrStats {
    double rel = 0.0, worst = 0.0;
};
ErrStats err_stats(const std::vector<uint16_t>& got, const std::vector<double>& ref) {
    ErrStats u;
    double se = 0.0, sr = 0.0, mx = 0.0, amax = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double g = (double) f32_from_f16(got[i]), e = std::isfinite(g) ? std::fabs(g - ref[i]) : 1e300;
        mx = std::max(mx, e);
        amax = std::max(amax, std::fabs(ref[i]));
        se += e * e;
        sr += ref[i] * ref[i];
    }
    u.rel = std::sqrt(se / std::max(sr, 1e-300));
    u.worst = mx / std::max(amax, 1e-300);
    return u;
}

void test_expect_one(const std::string& dir, const std::string& base, int k2, int cb, int k, int n) {
    const std::string tag = base + " K" + kname(k2) + " cb" + std::to_string(cb);
    if (!cpu::exl3_k2_valid(k2, cb) || k % 128 || n % 128) {
        std::printf("  %-44s not a matrix these kernels take  FAIL\n", tag.c_str());
        ++g_fail;
        return;
    }
    const size_t kn = (size_t) k * n;
    std::vector<uint32_t> tr;
    std::vector<uint16_t> suh, svh, what, wfull;
    const std::string p = dir + "/" + base;
    if (!read_raw(p + ".trellis.i16", kn * (size_t) k2 / 64, tr) || !read_raw(p + ".suh.f16", (size_t) k, suh) ||
        !read_raw(p + ".svh.f16", (size_t) n, svh) || !read_raw(p + ".what.f16", kn, what) ||
        !read_raw(p + ".w.f16", kn, wfull))
        return;
    std::printf("%s (%d x %d)\n", tag.c_str(), k, n);
    std::vector<float> w(kn);
    cpu::exl3_inner_k2(tr.data(), n / 16, k, n, k2, cb, w.data());
    size_t bad = 0;
    for (size_t i = 0; i < kn; ++i) bad += f16_from_f32(w[i]) != what[i];
    report("W': CPU reference vs exllamav3", bad, kn);
    ex::Mat m;
    m.trellis = dev_copy(tr);
    m.suh = dev_copy(suh);
    m.svh = dev_copy(svh);
    m.k = k;
    m.n = n;
    m.k2 = k2;
    m.cb = cb;
    device_w_hat(m, what, "vs exllamav3");
    // the full weight W = diag(suh) H W' H diag(svh): exactly (F64) from W' - exllamav3's own, above - then the
    // device's rows (W^T, F16; one FP16 rounding of the tile between the two transforms and one of the result) and
    // exllamav3's reconstruct_had_slice (its fused kernel rounds its FP16 intermediates) against it: both rounding
    // noise, ~3e-4 and ~1e-3 relative - a wrong scale, sign or layout would be ~1
    const std::vector<double> exact = full_weight_f64(w, suh, svh, k, n);
    uint16_t* d = nullptr;
    CK(cudaMalloc(&d, kn * 2));
    ex::reconstruct_rows(m, d, false, 0);
    CK(cudaDeviceSynchronize());
    const std::vector<uint16_t> rows = host_copy(d, kn);
    CK(cudaFree(d));
    std::vector<uint16_t> ours(kn);
    for (int r = 0; r < k; ++r)
        for (int c = 0; c < n; ++c) ours[(size_t) r * n + c] = rows[(size_t) c * k + r];
    const ErrStats a = err_stats(ours, exact), b = err_stats(wfull, exact);
    const bool ok_a = a.rel < 5e-4 && a.worst < 2e-3, ok_b = b.rel < 3e-3 && b.worst < 1e-2;
    std::printf("  %-44s rel %.2e  worst/max %.2e  %s\n", "W: device rows vs the exact W", a.rel, a.worst,
                ok_a ? "ok" : "FAIL");
    std::printf("  %-44s rel %.2e  worst/max %.2e  %s\n", "W: exllamav3's had reconstruct vs the exact W", b.rel,
                b.worst, ok_b ? "ok" : "FAIL");
    if (!ok_a) ++g_fail;
    if (!ok_b) ++g_fail;
    if (cb == 2) {
        cpu::Exl3Part c;
        c.trellis = tr.data();
        c.suh = suh.data();
        c.svh = svh.data();
        c.k = k;
        c.n = n;
        c.k2 = k2;
        std::vector<float> x = rand_x(k), xh((size_t) k), y((size_t) n), yref((size_t) n);
        cpu::exl3_mv(w.data(), suh.data(), svh.data(), k, n, x.data(), yref.data());
        const float xs = cpu::exl3_cpu_prepare(c, x.data(), xh.data());
        for (int c0 = 0; c0 < n; c0 += 128) cpu::exl3_cpu_band(c, xh.data(), xs, c0, y.data() + c0);
        close((std::string("cpu ") + cpu_tier_name(cpu::exl3_cpu_tier()) + " x W vs the reference").c_str(), y, yref,
              2e-3);
        lane_vs_model(c, {0, n - 128}, "(first and last band)");
    }
}

void test_expect(const std::string& dir) {
    std::ifstream f(dir + "/manifest.txt");
    if (!f) {
        std::printf("  %s/manifest.txt: missing (tools/exl3_exllamav3_fixture.py --model writes it)  FAIL\n", dir.c_str());
        ++g_fail;
        return;
    }
    std::string line;
    int n_mat = 0;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string base;
        int k2 = 0, cb = 0, k = 0, n = 0;
        if (!(ss >> base >> k2 >> cb >> k >> n)) {
            std::printf("  manifest line \"%s\": not <base> <k2> <cb> <k> <n>  FAIL\n", line.c_str());
            ++g_fail;
            continue;
        }
        test_expect_one(dir, base, k2, cb, k, n);
        ++n_mat;
    }
    std::printf("expect: %d matrices from %s\n", n_mat, dir.c_str());
    if (n_mat == 0) ++g_fail;
}

}  // namespace

// --bench: one decode token's routed experts at DeepSeek-V4.1's geometry (6 of them, 5120 -> 2304 -> 5120), the
// one-layout kernels against the mixed ones (the same experts as one class each; then K 1 / 2 / 3 mixed) - ms a token
// for one layer (x40 for the model)
void bench_moe() {
    const int E = 5120, F = 2304, k = 6, iters = 200;
    auto build = [&](const int (*ks)[3], ex::MixedLayouts& mix, std::vector<unsigned long long>& ptrs,
                     std::vector<uint8_t*>& blobs) {
        mix = ex::MixedLayouts{};
        mix.n = k;
        for (int e = 0; e < k; ++e) {
            ex::ExpertLayout& lay = mix.lay[e];
            lay.cb = 2;
            size_t at = 0;
            for (int r = 0; r < 3; ++r) {
                const int kk = r < 2 ? E : F, n = r < 2 ? F : E;
                lay.k2[r] = ks[e][r];
                lay.trellis[r] = at;
                at += (ex::trellis_bytes_k2(kk, n, ks[e][r]) + 255) / 256 * 256;
                lay.suh[r] = at;
                at += (2 * (size_t) kk + 255) / 256 * 256;
                lay.svh[r] = at;
                at += (2 * (size_t) n + 255) / 256 * 256;
            }
            std::vector<uint8_t> blob(at);
            for (int r = 0; r < 3; ++r) {
                const HostMat h = make_mat(r < 2 ? E : F, r < 2 ? F : E, ks[e][r], 2);
                std::memcpy(&blob[lay.trellis[r]], h.tr.data(), h.tr.size() * 4);
                std::memcpy(&blob[lay.suh[r]], h.suh.data(), h.suh.size() * 2);
                std::memcpy(&blob[lay.svh[r]], h.svh.data(), h.svh.size() * 2);
            }
            blobs.push_back(dev_copy(blob));
            ptrs.push_back((unsigned long long) blobs.back());
            mix.lo[e] = ptrs.back();
            mix.hi[e] = ptrs.back() + at;
        }
    };
    std::vector<float> x = rand_x(E), pw(k, 1.0f / k);
    float* d_x = dev_copy(x);
    float* d_pw = dev_copy(pw);
    float* d_out = nullptr;
    CK(cudaMalloc(&d_out, (size_t) E * 4));
    auto time = [&](const ex::MoeArgs& ma) {
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        for (int i = 0; i < 5; ++i) {
            ex::moe_gate_up(ma, d_x, nullptr, 0);
            ex::moe_down(ma, nullptr, nullptr, d_out, 0);
        }
        cudaEventRecord(a, 0);
        for (int i = 0; i < iters; ++i) {
            ex::moe_gate_up(ma, d_x, nullptr, 0);
            ex::moe_down(ma, nullptr, nullptr, d_out, 0);
        }
        cudaEventRecord(b, 0);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        return ms / iters;
    };
    const int k1[6][3] = {{2, 2, 2}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}};
    const int kx[6][3] = {{2, 2, 2}, {2, 2, 2}, {4, 4, 4}, {2, 2, 4}, {6, 6, 6}, {4, 2, 4}};
    for (const auto* ks : {k1, kx}) {
        ex::MixedLayouts mix;
        std::vector<unsigned long long> ptrs;
        std::vector<uint8_t*> blobs;
        build(ks, mix, ptrs, blobs);
        unsigned long long* d_plan = dev_copy(ptrs);
        ex::MoeArgs ma;
        ma.plan_ptr = d_plan;
        ma.plan_w = d_pw;
        ma.k = k;
        ma.n_embd = E;
        ma.n_ff = F;
        ma.limit = 10.0f;
        const bool uniform = ks == k1;
        float one = -1;
        if (uniform) {   // (all K 1: also through the one-layout kernels, every expert at its own address)
            ma.lay = mix.lay[0];
            one = time(ma);
        }
        ma.mix = mix;
        const float mixed = time(ma);
        std::printf("  bench %s: one layout %.3f ms, mixed kernels %.3f ms a layer's token\n",
                    uniform ? "6 experts at K 1" : "6 experts at K 1-3", one, mixed);
        cudaFree(d_plan);
        for (uint8_t* p : blobs) cudaFree(p);
    }
    cudaFree(d_x);
    cudaFree(d_pw);
    cudaFree(d_out);
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--bench") {
        bench_moe();
        return 0;
    }
    int dev = 0;
    cudaDeviceProp p{};
    CK(cudaGetDevice(&dev));
    CK(cudaGetDeviceProperties(&p, dev));
    std::printf("exl3_test on %s (%.1f GB)\n", p.name, (double) p.totalGlobalMem / 1e9);
    std::string fx_dir, expect_dir;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--exllamav3" && i + 1 < argc) {
            fx_dir = argv[++i];
        } else if (a == "--expect" && i + 1 < argc) {
            expect_dir = argv[++i];
        } else {
            std::fprintf(stderr, "usage: exl3_test [--exllamav3 <fixture dir>] [--expect <model fixture dir>]\n");
            return 2;
        }
    }
    if (!fx_dir.empty() || !expect_dir.empty()) {
        if (!fx_dir.empty()) test_exllamav3(fx_dir);
        if (!expect_dir.empty()) test_expect(expect_dir);
        std::printf("exl3_test: %s (%d failures, %d launch errors)\n", g_fail ? "FAIL" : "ok", g_fail,
                    ex::launch_errors());
        return g_fail || ex::launch_errors() ? 1 : 0;
    }
    test_decode_table();
    // the bitrate as k2 = 2 K: the integer K 1..8 first (in the order they always ran: the same random matrices)
    for (int k2 = 2; k2 <= 16; k2 += 2)
        for (int cb = 0; cb <= 2; ++cb) test_mv(256, 256, k2, cb, false);
    test_mv(4096, 512, 6, 2, false);        // occupancy split
    test_mv(12288, 256, 8, 2, false);       // k > 4096
    test_mv(2048, 4096, 10, 2, true);       // q8_1 input
    test_mv(4096, 1024, 4, 2, false, 384, 256);   // a column view
    test_batch(8, 6);
    test_batch(16, 12);                     // 8-bit attention / shared experts (EXL3-sc_4.10bpw)
    // routed experts above 4 bits: EXL3-sc_4.10bpw's first MoE layer is 5-bit
    for (int k2 : {4, 6, 8, 10, 12, 16}) test_moe(k2, 2);
    test_moe(6, 1);
    for (int k2 : {4, 10}) test_recon(256, 384, k2, 2);
    test_recon(256, 256, 6, 0);
    for (int k2 = 2; k2 <= 16; k2 += 2) test_rot_rows(256, 384, k2, 2);
    test_rot_rows(512, 256, 8, 1);
    for (int k2 : {4, 6, 10, 16}) test_gemm_rows(512, 384, k2, 2, 45);
    test_gemm_rows(4096, 256, 6, 1, 100);
    test_gemm_rows(1024, 128, 8, 0, 7);
    for (int k2 : {4, 6, 8, 10, 16}) test_set(k2, 2);
    for (int k2 = 2; k2 <= 16; k2 += 2) test_cpu(512, 256, k2);
    test_cpu(4096, 2048, 6);
    test_cpu(2048, 4096, 6);
    // several activations at once (a verify window): the same bits as one at a time
    for (int k2 = 2; k2 <= 16; k2 += 2) test_mv_n(512, 256, k2, 2, false, 3);
    for (int cb = 0; cb <= 1; ++cb) test_mv_n(256, 384, 8, cb, false, 5);
    for (int nc : {2, 4, 7, 16}) test_mv_n(4096, 512, 6, 2, false, nc);   // occupancy split
    test_mv_n(12288, 256, 12, 2, false, 6);                                 // k > 4096
    test_mv_n(2048, 4096, 10, 2, true, 8);                                  // q8_1 input
    for (int k2 : {4, 6, 8, 10, 16}) test_win(k2, 2, 7);
    test_win(6, 1, 2);
    test_win(6, 2, 16);
    for (int k2 : {2, 6, 10, 12, 16}) test_cpu_n(512, 256, k2);
    for (int k2 : {4, 6, 8, 10, 16}) test_cpu_rows(512, 384, k2);
    for (int k2 : {6, 8, 10, 12}) test_cpu_gemm(768, 256, k2);
    test_cpu_had();
    // the CPU tiers against an exact model of their arithmetic (odd slice counts too: 37 slices)
    for (int k2 = 2; k2 <= 16; k2 += 2) test_cpu_exact(512, 256, k2);
    test_cpu_exact(592, 384, 6);
    test_cpu_exact(592, 256, 8);
    // exllamav3 1.6's half-integer K 1.5 / 2.5 / 3.5 (k2 3, 5, 7; mul1 only) through every path, and an SC model's
    // layers (gate / up 3.5 with down 5, 1.5 with 3, 2.5 with 4)
    for (int k2 : {3, 5, 7}) test_mv(256, 256, k2, 2, false);
    test_mv(4096, 512, 7, 2, false);        // occupancy split
    test_mv(12288, 256, 5, 2, false);       // k > 4096
    test_mv(2048, 4096, 7, 2, true);        // q8_1 input
    test_mv(4096, 1024, 3, 2, false, 384, 256);   // a column view
    for (int k2 : {3, 5, 7}) test_moe(k2, 2);
    test_moe(7, 2, 10);
    test_moe(3, 2, 6);
    for (int k2 : {3, 5, 7}) test_recon(256, 384, k2, 2);
    for (int k2 : {3, 5, 7}) test_rot_rows(256, 384, k2, 2);
    for (int k2 : {3, 5, 7}) test_gemm_rows(512, 384, k2, 2, 45);
    for (int k2 : {3, 5, 7}) test_set(k2, 2);
    test_set(7, 2, 10);
    test_set(5, 2, 8);
    for (int cb : {0, 2}) test_mixed(cb);   // (a mixed-precision layer)
    for (int T : {2, 7, 16}) test_win(4, 2, T, 0, true);   // (a mixed layer's window: the token path's bits too)
    test_win(4, 0, 7, 0, true);
    for (int k2 : {3, 5, 7}) test_cpu(512, 256, k2);
    test_cpu(4096, 2048, 7);
    test_cpu(2048, 4096, 7);
    for (int k2 : {3, 5, 7}) test_mv_n(512, 256, k2, 2, false, 3);
    for (int nc : {2, 7, 16}) test_mv_n(4096, 512, 7, 2, false, nc);
    test_mv_n(2048, 4096, 5, 2, true, 8);
    for (int k2 : {3, 5, 7}) test_win(k2, 2, 7);
    test_win(7, 2, 16, 10);
    test_win(3, 2, 7, 6);
    for (int k2 : {3, 5, 7}) test_cpu_n(512, 256, k2);
    for (int k2 : {3, 5, 7}) test_cpu_rows(512, 384, k2);
    for (int k2 : {3, 5, 7}) test_cpu_gemm(768, 256, k2);
    for (int k2 : {3, 5, 7}) test_cpu_exact(512, 256, k2);
    test_cpu_exact(592, 384, 7);
    test_cpu_exact(592, 256, 5);
    // a half-integer K with another codebook is no bitrate exllamav3 writes: refused, nothing launched
    test_refuse_half_non_mul1();
    std::printf("exl3_test: %s (%d failures, %d launch errors)\n", g_fail ? "FAIL" : "ok", g_fail,
                ex::launch_errors());
    return g_fail || ex::launch_errors() ? 1 : 0;
}
