// tests/ling_kernels_test.cu - unit tests for maya-ling CUDA kernels (KDA window + rollback, 3D M-RoPE,
// WMMA MLA decode FP16/INT8 + causal window, grouped router, argmax / top1_prob) against double-precision CPU refs.
#include "ling_kernels.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#define CHECK_CUDA(x)                                                                             \
    do {                                                                                          \
        cudaError_t e_ = (x);                                                                     \
        if (e_ != cudaSuccess) {                                                                  \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,                    \
                         cudaGetErrorString(e_));                                                 \
            std::exit(1);                                                                         \
        }                                                                                         \
    } while (0)

static int g_fail = 0;
#define EXPECT(cond, msg, ...)                                                                    \
    do {                                                                                          \
        if (!(cond)) {                                                                            \
            std::fprintf(stderr, "FAIL: " msg "\n", ##__VA_ARGS__);                               \
            ++g_fail;                                                                             \
        }                                                                                         \
    } while (0)

static double sigmoid_d(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// Test 1: Windowed KDA recurrence + rollback (W=4, commit a=1..4) vs sequential single-step KDA
static void test_kda_window_and_rollback() {
    constexpr int H = 4, HD = 128, DI = H * HD, DC = 4, W = 4;
    constexpr float LB = -6.93f, EPS = 1e-6f;
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> u(-0.4f, 0.4f);

    auto rand_vec = [&](size_t n, float sc = 1.0f) {
        std::vector<float> v(n);
        for (auto& x : v) x = u(rng) * sc;
        return v;
    };
    const auto q_in = rand_vec(W * DI), k_in = rand_vec(W * DI), v_in = rand_vec(W * DI);
    const auto f_in = rand_vec(W * DI), g2_in = rand_vec(W * DI), beta_in = rand_vec(W * H);
    const auto cq = rand_vec(DI * DC), ck = rand_vec(DI * DC), cv = rand_vec(DI * DC);
    const auto ssm_a = rand_vec(H, 0.5f), norm_w = rand_vec(HD, 1.0f);
    const auto init_conv = rand_vec(3 * DI * (DC - 1));
    const auto init_state = rand_vec((size_t) H * HD * HD, 0.2f);

    auto dev_put = [](const std::vector<float>& v) {
        float* p = nullptr;
        CHECK_CUDA(cudaMalloc(&p, v.size() * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(p, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
        return p;
    };

    float *dq = dev_put(q_in), *dk = dev_put(k_in), *dv = dev_put(v_in), *df = dev_put(f_in);
    float *dg2 = dev_put(g2_in), *dbeta = dev_put(beta_in);
    float *dcq = dev_put(cq), *dck = dev_put(ck), *dcv = dev_put(cv), *da = dev_put(ssm_a), *dnw = dev_put(norm_w);
    float *d_qo, *d_ko, *d_vo, *d_g1;
    CHECK_CUDA(cudaMalloc(&d_qo, W * DI * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ko, W * DI * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_vo, W * DI * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_g1, W * DI * sizeof(float)));

    // Reference: step 1..W sequentially with single-step kda_prep_full + kda_rec_q8
    std::vector<std::vector<float>> ref_states(W + 1), ref_convs(W + 1);
    std::vector<std::vector<float>> ref_out(W);
    ref_states[0] = init_state;
    ref_convs[0] = init_conv;
    {
        float* d_cs = dev_put(init_conv);
        float* d_st = dev_put(init_state);
        int8_t* d_xq = nullptr;
        float* d_xd = nullptr;
        CHECK_CUDA(cudaMalloc(&d_xq, DI));
        CHECK_CUDA(cudaMalloc(&d_xd, (DI / 32) * sizeof(float)));
        for (int t = 0; t < W; ++t) {
            ling::k::kda_prep_full(dq + t * DI, dk + t * DI, dv + t * DI, df + t * DI, dcq, dck, dcv, d_cs, da, LB, EPS,
                                   H, HD, DC, d_qo, d_ko, d_vo, d_g1, 0);
            ling::k::kda_rec_q8(d_qo, d_ko, d_vo, d_g1, dbeta + t * H, d_st, dg2 + t * DI, dnw, EPS, H, HD, d_xq, d_xd,
                                nullptr, 0);
            CHECK_CUDA(cudaDeviceSynchronize());
            std::vector<int8_t> hq(DI);
            std::vector<float> hd(DI / 32);
            CHECK_CUDA(cudaMemcpy(hq.data(), d_xq, DI, cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(hd.data(), d_xd, (DI / 32) * sizeof(float), cudaMemcpyDeviceToHost));
            ref_out[t].resize(DI);
            for (int i = 0; i < DI; ++i) ref_out[t][i] = (float) hq[i] * hd[i >> 5];
            ref_states[t + 1].resize(init_state.size());
            ref_convs[t + 1].resize(init_conv.size());
            CHECK_CUDA(cudaMemcpy(ref_states[t + 1].data(), d_st, init_state.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(ref_convs[t + 1].data(), d_cs, init_conv.size() * sizeof(float), cudaMemcpyDeviceToHost));
        }
        cudaFree(d_cs); cudaFree(d_st); cudaFree(d_xq); cudaFree(d_xd);
    }

    // Windowed test + commit for every accept count a in 1..W
    for (int a = 1; a <= W; ++a) {
        float* d_cs = dev_put(init_conv);
        float* d_cb = dev_put(init_conv);
        float* d_st = dev_put(init_state);
        float* d_st_tmp = dev_put(init_state);
        int8_t* d_xq = nullptr;
        float* d_xd = nullptr;
        CHECK_CUDA(cudaMalloc(&d_xq, W * DI));
        CHECK_CUDA(cudaMalloc(&d_xd, W * (DI / 32) * sizeof(float)));

        ling::k::kda_prep_window(dq, dk, dv, df, dcq, dck, dcv, d_cs, d_cb, da, LB, EPS, W, H, HD, DC, d_qo, d_ko, d_vo,
                                 d_g1, 0);
        ling::k::kda_rec_window_q8(d_qo, d_ko, d_vo, d_g1, dbeta, d_st, d_st_tmp, dg2, dnw, EPS, W, H, HD, d_xq, d_xd, 0);
        if (a < W) ling::k::kda_conv_rollback(d_cb, dq, dk, dv, a, H, HD, DC, d_cs, 0);
        ling::k::kda_commit_window(d_ko, d_vo, d_g1, dbeta, d_st, d_st_tmp, a, W, H, HD, 0);
        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<int8_t> hq(W * DI);
        std::vector<float> hd(W * (DI / 32));
        std::vector<float> got_st(init_state.size()), got_cs(init_conv.size());
        CHECK_CUDA(cudaMemcpy(hq.data(), d_xq, W * DI, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hd.data(), d_xd, W * (DI / 32) * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(got_st.data(), d_st, init_state.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(got_cs.data(), d_cs, init_conv.size() * sizeof(float), cudaMemcpyDeviceToHost));

        float max_o = 0.0f, max_s = 0.0f, max_c = 0.0f;
        for (int t = 0; t < W; ++t)
            for (int i = 0; i < DI; ++i)
                max_o = std::max(max_o, std::fabs((float) hq[t * DI + i] * hd[(t * DI + i) >> 5] - ref_out[t][i]));
        for (size_t i = 0; i < got_st.size(); ++i) max_s = std::max(max_s, std::fabs(got_st[i] - ref_states[a][i]));
        for (size_t i = 0; i < got_cs.size(); ++i) max_c = std::max(max_c, std::fabs(got_cs[i] - ref_convs[a][i]));
        EXPECT(max_o < 1e-5f && max_s < 1e-5f && max_c < 1e-6f,
               "kda_window a=%d: max_o=%g max_s=%g max_c=%g", a, max_o, max_s, max_c);
        cudaFree(d_cs); cudaFree(d_cb); cudaFree(d_st); cudaFree(d_st_tmp); cudaFree(d_xq); cudaFree(d_xd);
    }

    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(df); cudaFree(dg2); cudaFree(dbeta);
    cudaFree(dcq); cudaFree(dck); cudaFree(dcv); cudaFree(da); cudaFree(dnw);
    cudaFree(d_qo); cudaFree(d_ko); cudaFree(d_vo); cudaFree(d_g1);
    std::printf("PASS test_kda_window_and_rollback (W=%d, a=1..%d)\n", W, W);
}

// Test 2: 3D M-RoPE (kv_store_split & mla_q_split_rope_f16 with interleaved H/W pairs 0..23 + T pairs 24..31)
static void test_mrope_3d() {
    constexpr int T = 6, H = 2, QN = 128, R = 64, QH = QN + R, KL = 512, MHW = 24;
    constexpr float BASE = 1000000.0f, EPS = 1e-6f;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);

    std::vector<float> h_qm(T * H * QH), h_kv(T * (KL + R)), h_wn(KL, 1.0f);
    for (auto& x : h_qm) x = u(rng);
    for (auto& x : h_kv) x = u(rng);
    std::vector<int> h_mrope(3 * T);
    for (int t = 0; t < T; ++t) {
        h_mrope[0 * T + t] = 10 + t;      // T
        h_mrope[1 * T + t] = 10 + (t / 2) - 1;  // H
        h_mrope[2 * T + t] = 10 + (t % 2) - 1;  // W
    }

    float *d_qm, *d_kv, *d_wn;
    int* d_mrope;
    uint16_t *d_qn16, *d_qp16, *d_lat16, *d_kpe16;
    CHECK_CUDA(cudaMalloc(&d_qm, h_qm.size() * 4));
    CHECK_CUDA(cudaMalloc(&d_kv, h_kv.size() * 4));
    CHECK_CUDA(cudaMalloc(&d_wn, KL * 4));
    CHECK_CUDA(cudaMalloc(&d_mrope, h_mrope.size() * 4));
    CHECK_CUDA(cudaMalloc(&d_qn16, T * H * QN * 2));
    CHECK_CUDA(cudaMalloc(&d_qp16, T * H * R * 2));
    CHECK_CUDA(cudaMalloc(&d_lat16, T * KL * 2));
    CHECK_CUDA(cudaMalloc(&d_kpe16, T * R * 2));
    CHECK_CUDA(cudaMemcpy(d_qm, h_qm.data(), h_qm.size() * 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kv, h_kv.data(), h_kv.size() * 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wn, h_wn.data(), KL * 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mrope, h_mrope.data(), h_mrope.size() * 4, cudaMemcpyHostToDevice));

    ling::k::mla_q_split_rope_f16(d_qm, T, H, QH, QN, R, 0, BASE, d_qn16, d_qp16, 0, d_mrope, MHW, T);
    ling::k::kv_store_split(d_kv, T, KL, R, d_wn, EPS, nullptr, 0, BASE, d_lat16, false, d_kpe16, 0, nullptr, d_mrope, MHW, T);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<__half> got_qp(T * H * R), got_kp(T * R);
    CHECK_CUDA(cudaMemcpy(got_qp.data(), d_qp16, got_qp.size() * 2, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(got_kp.data(), d_kpe16, got_kp.size() * 2, cudaMemcpyDeviceToHost));

    double max_err = 0.0;
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < R / 2; ++i) {
            const int pos = (i < MHW) ? h_mrope[((i & 1) ? 2 : 1) * T + t] : h_mrope[t];
            const double th = (double) pos * std::pow((double) BASE, -2.0 * (double) i / (double) R);
            const double c = std::cos(th), s = std::sin(th);
            // Check K_pe
            const double ka = h_kv[t * (KL + R) + KL + 2 * i], kb = h_kv[t * (KL + R) + KL + 2 * i + 1];
            const double rk0 = ka * c - kb * s, rk1 = ka * s + kb * c;
            max_err = std::max(max_err, std::fabs((double) __half2float(got_kp[t * R + 2 * i]) - rk0));
            max_err = std::max(max_err, std::fabs((double) __half2float(got_kp[t * R + 2 * i + 1]) - rk1));
            // Check Q_pe
            for (int h = 0; h < H; ++h) {
                const double qa = h_qm[(t * H + h) * QH + QN + 2 * i];
                const double qb = h_qm[(t * H + h) * QH + QN + 2 * i + 1];
                const double rq0 = qa * c - qb * s, rq1 = qa * s + qb * c;
                const size_t off = ((size_t) h * T + t) * R;
                max_err = std::max(max_err, std::fabs((double) __half2float(got_qp[off + 2 * i]) - rq0));
                max_err = std::max(max_err, std::fabs((double) __half2float(got_qp[off + 2 * i + 1]) - rq1));
            }
        }
    }
    EXPECT(max_err < 2e-3, "mrope_3d max_err=%g", max_err);
    cudaFree(d_qm); cudaFree(d_kv); cudaFree(d_wn); cudaFree(d_mrope);
    cudaFree(d_qn16); cudaFree(d_qp16); cudaFree(d_lat16); cudaFree(d_kpe16);
    std::printf("PASS test_mrope_3d (max_err=%.2e)\n", max_err);
}

// Test 3: WMMA MLA decode (FP16 & INT8 KV, T=4 causal window) vs double-precision CPU reference
static void test_mla_dec_wmma() {
    constexpr int H = 32, KL = 512, R = 64, QH = 192, QN = 128, NKV = 96, T = 4;
    const float scale = 1.0f / std::sqrt((float) QH);
    std::mt19937 rng(777);
    std::uniform_real_distribution<float> u(-0.3f, 0.3f);

    std::vector<float> h_qabs(T * H * KL), h_qm(T * H * QH);
    std::vector<__half> h_lat16(NKV * KL), h_kpe16(NKV * R);
    for (auto& x : h_qabs) x = u(rng);
    for (auto& x : h_qm) x = u(rng);
    for (auto& x : h_lat16) x = __float2half_rn(u(rng));
    for (auto& x : h_kpe16) x = __float2half_rn(u(rng));

    // Double-precision causal reference
    std::vector<double> ref_ctx(T * H * KL, 0.0);
    for (int t = 0; t < T; ++t) {
        const int n_t = NKV - (T - 1 - t);
        for (int h = 0; h < H; ++h) {
            std::vector<double> sc(n_t);
            double mx = -1e30;
            for (int k = 0; k < n_t; ++k) {
                double d = 0.0;
                for (int c = 0; c < KL; ++c)
                    d += (double) h_qabs[(t * H + h) * KL + c] * (double) __half2float(h_lat16[k * KL + c]);
                for (int c = 0; c < R; ++c)
                    d += (double) h_qm[(t * H + h) * QH + QN + c] * (double) __half2float(h_kpe16[k * R + c]);
                sc[k] = d * scale;
                mx = std::max(mx, sc[k]);
            }
            double sum = 0.0;
            for (int k = 0; k < n_t; ++k) { sc[k] = std::exp(sc[k] - mx); sum += sc[k]; }
            for (int c = 0; c < KL; ++c) {
                double acc = 0.0;
                for (int k = 0; k < n_t; ++k) acc += (sc[k] / sum) * (double) __half2float(h_lat16[k * KL + c]);
                ref_ctx[(t * H + h) * KL + c] = acc;
            }
        }
    }

    ling::k::mla_dec_wmma_init();
    float *d_qabs, *d_qm, *d_part, *d_ctx;
    __half *d_lat, *d_kpe;
    int* d_nkv;
    CHECK_CUDA(cudaMalloc(&d_qabs, h_qabs.size() * 4));
    CHECK_CUDA(cudaMalloc(&d_qm, h_qm.size() * 4));
    CHECK_CUDA(cudaMalloc(&d_lat, h_lat16.size() * 2));
    CHECK_CUDA(cudaMalloc(&d_kpe, h_kpe16.size() * 2));
    CHECK_CUDA(cudaMalloc(&d_part, ling::k::mla_dec_wmma_part_bytes(H, T)));
    CHECK_CUDA(cudaMalloc(&d_ctx, T * H * KL * 4));
    CHECK_CUDA(cudaMalloc(&d_nkv, 4));
    const int nkv_v = NKV;
    CHECK_CUDA(cudaMemcpy(d_qabs, h_qabs.data(), h_qabs.size() * 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_qm, h_qm.data(), h_qm.size() * 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_lat, h_lat16.data(), h_lat16.size() * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kpe, h_kpe16.data(), h_kpe16.size() * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_nkv, &nkv_v, 4, cudaMemcpyHostToDevice));

    const bool ok = ling::k::mla_dec_wmma(d_qabs, d_qm, QH, QN, d_lat, 0, (const uint16_t*) d_kpe, NKV, d_nkv, T, H, KL, R, scale, d_part, d_ctx, 0);
#if defined(MAYA_USE_HIP)
    // AMD: no tensor-core kernel - it says so (nothing launched) and the engine takes glmd::mla_dec row by row
    EXPECT(!ok, "mla_dec_wmma launched on HIP");
    cudaFree(d_qabs); cudaFree(d_qm); cudaFree(d_lat); cudaFree(d_kpe); cudaFree(d_part); cudaFree(d_ctx); cudaFree(d_nkv);
    std::printf("PASS test_mla_dec_wmma (HIP: not offered, as expected)\n");
    return;
#endif
    EXPECT(ok, "mla_dec_wmma returned false");
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> got(T * H * KL);
    CHECK_CUDA(cudaMemcpy(got.data(), d_ctx, got.size() * 4, cudaMemcpyDeviceToHost));
    double max_err = 0.0;
    for (size_t i = 0; i < got.size(); ++i) max_err = std::max(max_err, std::fabs((double) got[i] - ref_ctx[i]));
    EXPECT(max_err < 3e-3, "mla_dec_wmma max_err=%g", max_err);

    cudaFree(d_qabs); cudaFree(d_qm); cudaFree(d_lat); cudaFree(d_kpe); cudaFree(d_part); cudaFree(d_ctx); cudaFree(d_nkv);
    std::printf("PASS test_mla_dec_wmma (T=%d causal, max_err=%.2e)\n", T, max_err);
}

// Test 4: Grouped router + argmax_rows_f32 + top1_prob_f32
static void test_router_and_argmax() {
    constexpr int T = 4, NE = 512, NU = 8, NG = 8, NGU = 4;
    constexpr float WSCALE = 2.5f;
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> u(-2.0f, 2.0f);
    std::vector<float> h_log(T * NE), h_bias(NE);
    for (auto& x : h_log) x = u(rng);
    for (auto& x : h_bias) x = u(rng) * 0.2f;

    float *d_log, *d_bias, *d_w, *d_sv, *d_tp;
    int *d_ids, *d_si, *d_ar;
    CHECK_CUDA(cudaMalloc(&d_log, h_log.size() * 4));
    CHECK_CUDA(cudaMalloc(&d_bias, h_bias.size() * 4));
    CHECK_CUDA(cudaMalloc(&d_w, T * NU * 4));
    CHECK_CUDA(cudaMalloc(&d_ids, T * NU * 4));
    CHECK_CUDA(cudaMalloc(&d_sv, T * 256 * 4));
    CHECK_CUDA(cudaMalloc(&d_si, T * 256 * 4));
    CHECK_CUDA(cudaMalloc(&d_ar, T * 4));
    CHECK_CUDA(cudaMalloc(&d_tp, 2 * 4));
    CHECK_CUDA(cudaMemcpy(d_log, h_log.data(), h_log.size() * 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * 4, cudaMemcpyHostToDevice));

    ling::k::router_grouped(d_log, d_bias, T, NE, NU, NG, NGU, true, WSCALE, d_ids, d_w, 0);
    ling::k::argmax_rows_f32(d_log, T, NE, d_sv, d_si, d_ar, 0);
    ling::k::top1_prob_f32(d_log, NE, d_sv, d_si, d_tp, 0);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<int> got_ids(T * NU), got_ar(T);
    std::vector<float> got_w(T * NU), got_tp(2);
    CHECK_CUDA(cudaMemcpy(got_ids.data(), d_ids, got_ids.size() * 4, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(got_w.data(), d_w, got_w.size() * 4, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(got_ar.data(), d_ar, got_ar.size() * 4, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(got_tp.data(), d_tp, 8, cudaMemcpyDeviceToHost));

    for (int t = 0; t < T; ++t) {
        const float* row = h_log.data() + t * NE;
        int ref_am = (int) (std::max_element(row, row + NE) - row);
        EXPECT(got_ar[t] == ref_am, "argmax_rows t=%d: got=%d ref=%d", t, got_ar[t], ref_am);
        if (t == 0) {
            int tp_id = 0;
            std::memcpy(&tp_id, &got_tp[0], 4);
            double sum = 0.0;
            for (int i = 0; i < NE; ++i) sum += std::exp((double) row[i] - (double) row[ref_am]);
            const double ref_p = 1.0 / sum;
            EXPECT(tp_id == ref_am && std::fabs((double) got_tp[1] - ref_p) < 1e-5,
                   "top1_prob: id=%d/%d p=%g/%g", tp_id, ref_am, got_tp[1], ref_p);
        }
        // CPU grouped router reference
        std::vector<double> sp(NE), sv(NE);
        for (int e = 0; e < NE; ++e) { sp[e] = sigmoid_d(row[e]); sv[e] = sp[e] + h_bias[e]; }
        std::vector<std::pair<double, int>> gsc;
        for (int g = 0; g < NG; ++g) {
            std::vector<double> v(sv.begin() + g * 64, sv.begin() + (g + 1) * 64);
            std::sort(v.begin(), v.end(), std::greater<double>());
            gsc.push_back({v[0] + v[1], g});
        }
        std::sort(gsc.begin(), gsc.end(), [](auto a, auto b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
        std::vector<std::pair<double, int>> cands;
        for (int k = 0; k < NGU; ++k) {
            int g = gsc[k].second;
            for (int i = 0; i < 64; ++i) cands.push_back({sv[g * 64 + i], g * 64 + i});
        }
        std::sort(cands.begin(), cands.end(), [](auto a, auto b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
        double wsum = 0.0;
        for (int j = 0; j < NU; ++j) wsum += sp[cands[j].second];
        for (int j = 0; j < NU; ++j) {
            const int re = cands[j].second;
            const double rw = sp[re] / wsum * WSCALE;
            EXPECT(got_ids[t * NU + j] == re && std::fabs((double) got_w[t * NU + j] - rw) < 1e-5,
                   "router t=%d j=%d: id=%d/%d w=%g/%g", t, j, got_ids[t * NU + j], re, got_w[t * NU + j], rw);
        }
    }
    cudaFree(d_log); cudaFree(d_bias); cudaFree(d_w); cudaFree(d_ids);
    cudaFree(d_sv); cudaFree(d_si); cudaFree(d_ar); cudaFree(d_tp);
    std::printf("PASS test_router_and_argmax (T=%d, 512x8)\n", T);
}

static void test_router_nan_guard_and_fused_selfcheck() {
    constexpr int NE = 512, NU = 8;
    std::vector<float> h_log(NE, 0.1f), h_bias(NE, 0.0f);
    h_log[137] = std::nanf("");
    float *d_log = nullptr, *d_bias = nullptr, *d_w = nullptr;
    int *d_ids = nullptr, *d_err = nullptr;
    CHECK_CUDA(cudaMalloc(&d_log, NE * 4));
    CHECK_CUDA(cudaMalloc(&d_bias, NE * 4));
    CHECK_CUDA(cudaMalloc(&d_w, NU * 4));
    CHECK_CUDA(cudaMalloc(&d_ids, NU * 4));
    CHECK_CUDA(cudaMalloc(&d_err, 4));
    CHECK_CUDA(cudaMemcpy(d_log, h_log.data(), NE * 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, h_bias.data(), NE * 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_err, 0, 4));
    ling::k::router_grouped(d_log, d_bias, 1, NE, NU, 8, 4, true, 2.5f, d_ids, d_w, 0, d_err);
    CHECK_CUDA(cudaDeviceSynchronize());
    int h_err = 0;
    std::vector<int> ids(NU, -1);
    std::vector<float> w(NU, -1.0f);
    CHECK_CUDA(cudaMemcpy(&h_err, d_err, 4, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ids.data(), d_ids, NU * 4, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(w.data(), d_w, NU * 4, cudaMemcpyDeviceToHost));
    EXPECT(h_err == 1, "NaN router logit was not caught (err=%d)", h_err);
    for (int k = 0; k < NU; ++k) EXPECT(ids[k] == 0 && w[k] == 0.0f, "NaN router did not zero fallback slot %d", k);
    cudaFree(d_log); cudaFree(d_bias); cudaFree(d_w); cudaFree(d_ids); cudaFree(d_err);

    std::string ferr;
    cudaStream_t st = nullptr;
    CHECK_CUDA(cudaStreamCreate(&st));
    const bool fok = ling::k::verify_fused_kernels(st, ferr);
    EXPECT(fok, "verify_fused_kernels failed: %s", ferr.c_str());

    // Verify q8k_rows bit-exact against CPU Q8_K rounding
    constexpr int K = 512;
    std::vector<float> hx(K), hd(K / 32);
    std::vector<int8_t> hq(K);
    for (int i = 0; i < K; ++i) hx[i] = std::sin((float) i * 0.31f) * 3.0f;
    float *dx = nullptr, *dd = nullptr;
    int8_t *dq = nullptr;
    CHECK_CUDA(cudaMalloc(&dx, K * 4));
    CHECK_CUDA(cudaMalloc(&dq, K));
    CHECK_CUDA(cudaMalloc(&dd, (K / 32) * 4));
    CHECK_CUDA(cudaMemcpyAsync(dx, hx.data(), K * 4, cudaMemcpyHostToDevice, st));
    ling::k::q8k_rows(dx, 1, K, dq, dd, st);
    CHECK_CUDA(cudaMemcpyAsync(hq.data(), dq, K, cudaMemcpyDeviceToHost, st));
    CHECK_CUDA(cudaMemcpyAsync(hd.data(), dd, (K / 32) * 4, cudaMemcpyDeviceToHost, st));
    CHECK_CUDA(cudaStreamSynchronize(st));
    for (int b = 0; b < K / 256; ++b) {
        float am = 0.0f; int ix = 0;
        for (int j = 0; j < 256; ++j) if (std::fabs(hx[b * 256 + j]) > am) { am = std::fabs(hx[b * 256 + j]); ix = j; }
        const float iscale = -127.0f / hx[b * 256 + ix];
        const float ref_d = 1.0f / iscale;
        EXPECT(std::fabs(hd[b * 8] - ref_d) <= 1e-7f, "q8k_rows d mismatch blk %d: %g vs %g", b, hd[b * 8], ref_d);
        for (int j = 0; j < 256; ++j) {
            const float val = iscale * hx[b * 256 + j] + 12582912.0f;
            int32_t bits; std::memcpy(&bits, &val, 4);
            const int8_t ref_q = (int8_t) std::min(127, (bits & 0x007fffff) - 0x00400000);
            EXPECT(hq[b * 256 + j] == ref_q, "q8k_rows q mismatch at %d: %d vs %d", b * 256 + j, (int) hq[b * 256 + j], (int) ref_q);
        }
    }
    cudaFree(dx); cudaFree(dq); cudaFree(dd);
    cudaStreamDestroy(st);
    std::printf("PASS test_router_nan_guard_and_fused_selfcheck\n");
}

int main() {
    test_kda_window_and_rollback();
    test_mrope_3d();
    test_mla_dec_wmma();
    test_router_and_argmax();
    test_router_nan_guard_and_fused_selfcheck();
    if (g_fail > 0) {
        std::fprintf(stderr, "ling_kernels_test: %d failures\n", g_fail);
        return 1;
    }
    std::printf("ALL ling_kernels_test PASSED\n");
    return 0;
}
