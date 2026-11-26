#include "hy4_mixed_quant.cuh"
#include "hy4_kernels.cuh"
#include "ggml.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#define CU(x)                                                                                                          \
    do {                                                                                                               \
        auto e = (x);                                                                                                  \
        if (e != cudaSuccess) {                                                                                        \
            fprintf(stderr, "CUDA %s line %d\n", cudaGetErrorString(e), __LINE__);                                     \
            exit(2);                                                                                                   \
        }                                                                                                              \
    } while (0)
template <class T> T* upload(const std::vector<T>& v) {
    T* p = nullptr;
    CU(cudaMalloc(&p, v.size() * sizeof(T)));
    CU(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
__global__ void decode(const uint8_t* p, int type, int n, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = hy4::mixed_quant::value(p, type, i);
}
std::mt19937 rng(20261124);
void fill(std::vector<uint8_t>& v, int ty) {
    for (auto& x : v)
        x = rng();
    int bs = ggml_type_size((ggml_type)ty);
    for (size_t b = 0; b < v.size(); b += bs) {
        uint16_t h = ggml_fp32_to_fp16((int(rng() % 127) - 63) * .0003f);
        size_t off = ty == 14 ? bs - 2 : 0;
        memcpy(v.data() + b + off, &h, 2);
        if (ty == 13) {
            h = ggml_fp32_to_fp16((rng() % 64) * .0002f);
            memcpy(v.data() + b + 2, &h, 2);
        }
        if (ty == 30) {
            h = uint16_t(0x3b00 + (rng() % 768));
            memcpy(v.data() + b, &h, 2);
        }
    }
}
void deq(const std::vector<uint8_t>& b, int ty, std::vector<float>& f) {
    ggml_get_type_traits((ggml_type)ty)->to_float(b.data(), f.data(), f.size());
}
int test_scalar() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int fails = 0;
    for (int ty : {8, 30, 13, 14, 21, 23}) {
        constexpr int n = 256 * 127;
        std::vector<uint8_t> b(hy4::mixed_quant::row_bytes(ty, n));
        fill(b, ty);
        std::vector<float> ref(n), got(n);
        deq(b, ty, ref);
        auto* db = upload(b);
        auto* dy = upload(got);
        decode<<<(n + 127) / 128, 128>>>(db, ty, n, dy);
        CU(cudaDeviceSynchronize());
        CU(cudaMemcpy(got.data(), dy, n * 4, cudaMemcpyDeviceToHost));
        double er = 0, nr = 0;
        for (int i = 0; i < n; ++i) {
            er += pow(got[i] - ref[i], 2);
            nr += pow(ref[i], 2);
        }
        double rel = sqrt(er / std::max(nr, 1e-30));
        printf("decode type=%d relative=%.3e\n", ty, rel);
        if (!(rel < 1e-7))
            ++fails;
        cudaFree(db);
        cudaFree(dy);
    }
    const int formats[][3] = {{21, 23, 23}, {13, 14, 23}, {14, 13, 14}, {23, 13, 13}, {14, 23, 23},
                              {23, 21, 23}, {23, 21, 21}, {8, 30, 8},   {30, 8, 30}};
    for (auto& f : formats)
        for (int T : {1, 3, 33})
            for (bool half : {false, true})
                for (int U : {3, 16}) {
                    const int E = 512, F = 256, N = 3;
                    size_t gb = hy4::mixed_quant::row_bytes(f[0], E) * F, ub = hy4::mixed_quant::row_bytes(f[1], E) * F,
                           db = hy4::mixed_quant::row_bytes(f[2], F) * E;
                    auto align = [](size_t n) { return (n + 255) / 256 * 256; };
                    size_t up = align(gb), dn = up + align(ub), stride = align(dn + db);
                    std::vector<uint8_t> g(N * gb), u(N * ub), d(N * db);
                    fill(g, f[0]);
                    fill(u, f[1]);
                    fill(d, f[2]);
                    std::vector<float> gf(N * F * E), uf(N * F * E), df(N * E * F);
                    deq(g, f[0], gf);
                    deq(u, f[1], uf);
                    deq(d, f[2], df);
                    const std::vector<int> slots = {1, -1, 0};
                    std::vector<uint8_t> blob(2 * stride);
                    for (int e = 0; e < N; ++e)
                        if (slots[e] >= 0) {
                            uint8_t* p = blob.data() + slots[e] * stride;
                            memcpy(p, g.data() + e * gb, gb);
                            memcpy(p + up, u.data() + e * ub, ub);
                            memcpy(p + dn, d.data() + e * db, db);
                        }
                    std::vector<float> x(T * E), weights(T * U), got(T * E), ref(T * E, 0);
                    std::normal_distribution<float> normal(0, .15f);
                    for (auto& v : x)
                        v = normal(rng);
                    std::vector<int> ids(T * U);
                    for (int t = 0; t < T; ++t)
                        for (int j = 0; j < U; ++j) {
                            ids[t * U + j] = (j + t) % N;
                            weights[t * U + j] = (j + 1) / float(U * (U + 1) / 2);
                        }
                    float limit = T == 3 ? 0.f : .2f;
                    for (int entry = 0; entry < T * U; ++entry) {
                        int e = ids[entry], t = entry / U;
                        if (slots[e] < 0)
                            continue;
                        std::vector<float> h(F);
                        for (int r = 0; r < F; ++r) {
                            double a = 0, b = 0;
                            for (int k = 0; k < E; ++k) {
                                a += (double)gf[(e * F + r) * E + k] * x[t * E + k];
                                b += (double)uf[(e * F + r) * E + k] * x[t * E + k];
                            }
                            float ga = a, un = b;
                            if (limit > 0) {
                                ga = std::min(ga, limit);
                                un = std::clamp(un, -limit, limit);
                            }
                            float v = ga / (1.f + expf(-ga)) * un;
                            h[r] = half ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(v)) : v;
                        }
                        for (int r = 0; r < E; ++r) {
                            double sum = 0;
                            for (int k = 0; k < F; ++k)
                                sum += (double)df[(e * E + r) * F + k] * h[k];
                            ref[t * E + r] += weights[entry] * (float)sum;
                        }
                    }
                    auto* dx = upload(x);
                    auto* di = upload(ids);
                    auto* dw = upload(weights);
                    auto* ds = upload(slots);
                    auto* bb = upload(blob);
                    auto* dh = upload(std::vector<float>(T * U * F));
                    auto* dp = upload(std::vector<float>(T * U * E));
                    auto* dy = upload(got);
                    hy4::k::mixed_experts(dx, di, dw, ds, bb, stride, up, dn, f[0], f[1], f[2], E, F, T, U, limit, dh,
                                          half, dp, dy, 0);
                    CU(cudaDeviceSynchronize());
                    CU(cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost));
                    double er = 0, nr = 0;
                    for (size_t i = 0; i < got.size(); ++i) {
                        er += pow(got[i] - ref[i], 2);
                        nr += pow(ref[i], 2);
                    }
                    double rel = sqrt(er / std::max(nr, 1e-30));
                    printf("experts %d/%d/%d T=%d half=%d used=%d relative=%.3e\n", f[0], f[1], f[2], T, half, U, rel);
                    if (!(rel < (half ? .003 : 1e-5)))
                        ++fails;
                    cudaFree(dx);
                    cudaFree(di);
                    cudaFree(dw);
                    cudaFree(ds);
                    cudaFree(bb);
                    cudaFree(dh);
                    cudaFree(dp);
                    cudaFree(dy);
                }
    printf("MIXED_GPU_PARITY_%s failures=%d\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}

#include "../src/ds4/ds4_iq.cuh"
namespace iq = maya::ds4::gguf::iq;
void q8ref(std::vector<float>& x) {
    for (size_t b = 0; b < x.size(); b += 32) {
        float am = 0;
        for (int i = 0; i < 32; ++i)
            am = std::max(am, std::abs(x[b + i]));
        float d = am / 127.f;
        for (int i = 0; i < 32; ++i)
            x[b + i] = d ? std::nearbyint(x[b + i] / d) * d : 0;
    }
}
int test_native() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int fails = 0;
    const int formats[][3] = {{21, 23, 23}, {13, 14, 23}, {14, 13, 14}, {23, 13, 13},
                              {14, 23, 23}, {23, 21, 23}, {23, 21, 21}, {8, 13, 14}};
    for (auto& f : formats)
        for (int T : {1, 3, 4}) {
            const int E = 256, F = 256, U = 16, N = 17;
            size_t gb = hy4::mixed_quant::row_bytes(f[0], E) * F, ub = hy4::mixed_quant::row_bytes(f[1], E) * F,
                   db = hy4::mixed_quant::row_bytes(f[2], F) * E;
            auto align = [](size_t n) { return (n + 255) / 256 * 256; };
            size_t up = align(gb), dn = up + align(ub), stride = align(dn + db);
            std::vector<uint8_t> g(N * gb), u(N * ub), d(N * db);
            fill(g, f[0]);
            fill(u, f[1]);
            fill(d, f[2]);
            std::vector<float> gf(N * F * E), uf(N * F * E), df(N * E * F);
            deq(g, f[0], gf);
            deq(u, f[1], uf);
            deq(d, f[2], df);
            std::vector<int> slots(N);
            int resident = 0;
            for (int e = 0; e < N; ++e)
                slots[e] = e % 3 == 1 ? -1 : resident++;
            std::vector<uint8_t> blob(resident * stride);
            for (int e = 0; e < N; ++e)
                if (slots[e] >= 0) {
                    uint8_t* p = blob.data() + slots[e] * stride;
                    memcpy(p, g.data() + e * gb, gb);
                    memcpy(p + up, u.data() + e * ub, ub);
                    memcpy(p + dn, d.data() + e * db, db);
                }
            std::vector<float> x(T * E), weights(T * U), got(T * E), ref(T * E, 0);
            std::normal_distribution<float> normal(0, .15f);
            for (auto& v : x)
                v = normal(rng);
            std::vector<float> xref = x;
            q8ref(xref);
            std::vector<int> ids(T * U);
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < U; ++j) {
                    ids[t * U + j] = (j + t * 3) % N;
                    weights[t * U + j] = (j + 1) / 136.f;
                }
            const float limit = T == 3 ? 0.f : .2f;
            for (int entry = 0; entry < T * U; ++entry) {
                int e = ids[entry], t = entry / U;
                if (slots[e] < 0)
                    continue;
                std::vector<float> h(F);
                for (int r = 0; r < F; ++r) {
                    double a = 0, b = 0;
                    for (int k = 0; k < E; ++k) {
                        a += (double)gf[(e * F + r) * E + k] * xref[t * E + k];
                        b += (double)uf[(e * F + r) * E + k] * xref[t * E + k];
                    }
                    float ga = a, un = b;
                    if (limit > 0) {
                        ga = std::min(ga, limit);
                        un = std::clamp(un, -limit, limit);
                    }
                    h[r] = ga / (1.f + expf(-ga)) * un;
                }
                q8ref(h);
                for (int r = 0; r < E; ++r) {
                    double sum = 0;
                    for (int k = 0; k < F; ++k)
                        sum += (double)df[(e * E + r) * F + k] * h[k];
                    ref[t * E + r] += weights[entry] * (float)sum;
                }
            }
            auto* dx = upload(x);
            auto* di = upload(ids);
            auto* dw = upload(weights);
            auto* ds = upload(slots);
            auto* bb = upload(blob);
            auto* dh = upload(std::vector<float>(T * U * F));
            auto* dp = upload(std::vector<float>(T * U * E));
            auto* dy = upload(got);
            auto* xq = upload(std::vector<int8_t>(2 * T * E));
            auto* xd = upload(std::vector<float>(2 * T * E / 32));
            auto* hq = upload(std::vector<int8_t>(T * U * F));
            auto* hd = upload(std::vector<float>(T * U * F / 32));
            auto* plan = upload(std::vector<unsigned long long>(T * U));
            auto* starts = upload(std::vector<int>(N + 1));
            auto* ent = upload(std::vector<int>(T * U));
            auto* ng = upload(std::vector<int>(1));
            iq::Args a;
            a.tr[0] = 0;
            a.tr[1] = up;
            a.tr[2] = dn;
            a.n_embd = E;
            a.n_ff = F;
            a.limit = limit;
            a.h = dh;
            a.part = dp;
            a.tg = f[0];
            a.tu = f[1];
            a.td = f[2];
            a.act = iq::kActSwiglu;
            a.xq = xq;
            a.xd = xd;
            if (T == 1) {
                hy4::k::q8_rows(dx, E, T, E, xq, xd, 0);
                hy4::k::plan_experts(di, U, bb, stride, plan, 0, ds, dw);
                a.plan_ptr = plan;
            } else {
                hy4::k::q8_rows_dup(dx, E, T, E, xq, xd, 0);
                hy4::k::group_experts(di, T, U, N, bb, stride, plan, starts, ent, ng, 0, ds, dw);
                a.grp_ptr = plan;
                a.grp_start = starts;
                a.grp_ent = ent;
                a.n_groups = ng;
            }
            if (!iq::gate_up(a, T == 1 ? U : N, T, 0)) {
                puts("native gate rejected");
                return 2;
            }
            hy4::k::q8_rows(dh, F, T * U, F, hq, hd, 0);
            a.hq = hq;
            a.hd = hd;
            if (!iq::down(a, T == 1 ? U : N, T, 0)) {
                puts("native down rejected");
                return 2;
            }
            hy4::k::moe_sum(dp, T, U, E, dw, dy, 0);
            CU(cudaDeviceSynchronize());
            CU(cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost));
            double er = 0, nr = 0;
            for (size_t i = 0; i < got.size(); ++i) {
                er += pow(got[i] - ref[i], 2);
                nr += pow(ref[i], 2);
            }
            double rel = sqrt(er / std::max(nr, 1e-30));
            printf("native %d/%d/%d T=%d relative=%.3e\n", f[0], f[1], f[2], T, rel);
            if (!(rel < .01))
                ++fails;
            cudaFree(dx);
            cudaFree(di);
            cudaFree(dw);
            cudaFree(ds);
            cudaFree(bb);
            cudaFree(dh);
            cudaFree(dp);
            cudaFree(dy);
            cudaFree(xq);
            cudaFree(xd);
            cudaFree(hq);
            cudaFree(hd);
            cudaFree(plan);
            cudaFree(starts);
            cudaFree(ent);
            cudaFree(ng);
        }
    printf("NATIVE_MIXED_PARITY_%s failures=%d\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}

int test_engram() {
    int fails = 0;
    std::mt19937 rng(1282026);
    constexpr int T = 3, C = 16, D = 96;
    for (int type : {6, 7}) {
        const int bb = type == 6 ? 22 : 24;
        std::vector<uint8_t> rows(T * C * D, 0);
        std::vector<float> valid = {1, 0, 1}, ln(C * D), scale(T * C);
        for (int i = 0; i < C * D; ++i)
            ln[i] = .8f + (rng() % 1000) * .0004f;
        for (int i = 0; i < T * C; ++i) {
            scale[i] = .7f + (rng() % 1000) * .0005f;
            for (int b = 0; b < 3; ++b) {
                uint8_t* p = rows.data() + i * D + b * bb;
                for (int j = 0; j < bb; ++j)
                    p[j] = rng();
                uint16_t d = ggml_fp32_to_fp16((i % 2 ? -1.f : 1.f) * (i + 1) * .001f);
                std::memcpy(p, &d, 2);
                if (type == 7) {
                    uint16_t m = ggml_fp32_to_fp16((i - 24) * .01f);
                    std::memcpy(p + 2, &m, 2);
                }
            }
        }
        auto* dr = upload(rows);
        auto* dv = upload(valid);
        auto* dl = upload(ln);
        auto* ds = upload(scale);
        float* out = nullptr;
        CU(cudaMalloc(&out, T * C * D * sizeof(float)));
        for (int scaled : {0, 1})
            for (float eps : {1.1920928955078125e-7f, .0078125f}) {
                hy4::k::engram_rows(dr, dv, T, C, D, dl, eps, out, 0, scaled ? ds : nullptr, type == 6 ? 2 : 1);
                CU(cudaDeviceSynchronize());
                std::vector<float> got(T * C * D);
                CU(cudaMemcpy(got.data(), out, got.size() * 4, cudaMemcpyDeviceToHost));
                double err = 0, norm = 0;
                for (int r = 0; r < T * C; ++r) {
                    float row[D];
                    ggml_get_type_traits((ggml_type)type)->to_float(rows.data() + r * D, row, D);
                    double sum = 0;
                    for (auto& v : row) {
                        if (scaled)
                            v *= scale[r];
                        sum += (double)v * v;
                    }
                    double inv = 1 / std::sqrt(sum / D + eps);
                    for (int i = 0; i < D; ++i) {
                        double ref = row[i] * inv * ln[(r % C) * D + i] * valid[r / C];
                        double d = got[r * D + i] - ref;
                        err += d * d;
                        norm += ref * ref;
                    }
                }
                double rel = std::sqrt(err / norm);
                std::printf("Engram Q5_%d scaled=%d eps=%g relative %.3e\n", type == 6 ? 0 : 1, scaled, eps, rel);
                if (!(rel < 1e-6))
                    ++fails;
            }
        cudaFree(dr);
        cudaFree(dv);
        cudaFree(dl);
        cudaFree(ds);
        cudaFree(out);
    }
    return fails ? 1 : 0;
}

int main() { return test_scalar() | test_native() | test_engram(); }
