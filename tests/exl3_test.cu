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
    close(("cpu " + std::string(cpu::exl3_cpu_vector() ? "avx512" : "scalar") + " k" + std::to_string(k) + " n" +
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
    for (int K : {2, 3, 5, 8}) test_gemm_rows(512, 384, K, 2, 45);
    test_gemm_rows(4096, 256, 3, 1, 100);
    test_gemm_rows(1024, 128, 4, 0, 7);
    for (int K : {2, 3, 4}) test_set(K, 2);
    for (int K = 1; K <= 8; ++K) test_cpu(512, 256, K);
    test_cpu(4096, 2048, 3);
    test_cpu(2048, 4096, 3);
    std::printf("exl3_test: %s (%d failures, %d launch errors)\n", g_fail ? "FAIL" : "ok", g_fail,
                ex::launch_errors());
    return g_fail || ex::launch_errors() ? 1 : 0;
}
