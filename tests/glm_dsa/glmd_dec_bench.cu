// tests/glm_dsa/glmd_dec_bench.cu - L1-4 (label gpu): the glm-dsa decode mixer's GEMVs and attention at GLM-5.3's
// shapes, timed with the weights rotated over copies (no L2 reuse between calls, as in a decode where 78 layers each
// read their own); the L1-4 kernels against a double-precision reference (the gate) and against the kernels they
// replace (time).  A HIP build runs it all but the fused attention (nvcuda::wmma, CUDA's): the GEMV part (mv_dsa
// against gf::mv: the bits, and the time that picks the device's default) and the attention part (mla_dec against
// mla2's own kernels where the device takes it - glmd_dec.cu's policy; STRATA_GLMD_MLA_DEC=1 forces it to measure an
// AMD arch it is off on).
//
//   glmd_dec_bench [reps]      (default 40; exits 77 without a GPU)
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/glmd_dec.hpp"
#include "strata/kernels/glmd_fast.hpp"
#include "strata/kernels/glmd_volta.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace gf = strata::kernels::glmf;

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

// one weight matrix in a ggml type (or BF16 / F32), n_copy copies one after another on the device
struct Mat {
    int type = 0, n_in = 0, n_out = 0, n_copy = 1;
    size_t bytes = 0;   // one copy
    uint8_t* d = nullptr;
    std::vector<uint8_t> h;   // the host bytes (one copy)
    const void* at(int c) const { return d + (size_t) (c % n_copy) * bytes; }
    // row r dequantized (BF16 / Q8_0 / F32)
    std::vector<double> row(int r) const {
        std::vector<double> v((size_t) n_in);
        if (type == gf::kTypeBF16) {
            const uint16_t* p = (const uint16_t*) h.data() + (size_t) r * n_in;
            for (int i = 0; i < n_in; ++i) {
                const uint32_t u = (uint32_t) p[i] << 16;
                float f;
                std::memcpy(&f, &u, 4);
                v[(size_t) i] = f;
            }
        } else if (type == 8) {
            const uint8_t* p = h.data() + (size_t) r * (n_in / 32) * 34;
            for (int b = 0; b < n_in / 32; ++b) {
                uint16_t dh;
                std::memcpy(&dh, p + 34 * b, 2);
                const double d = ggml_fp16_to_fp32(dh);
                for (int j = 0; j < 32; ++j) v[(size_t) (32 * b + j)] = d * (double) (int8_t) p[34 * b + 2 + j];
            }
        } else if (type == 0) {
            const float* p = (const float*) h.data() + (size_t) r * n_in;
            for (int i = 0; i < n_in; ++i) v[(size_t) i] = p[i];
        }
        return v;
    }
};

Mat make_mat(int type, int n_in, int n_out, uint32_t seed) {
    Mat m;
    m.type = type;
    m.n_in = n_in;
    m.n_out = n_out;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    std::vector<float> w((size_t) n_in * n_out);
    for (float& v : w) v = nd(rng);
    std::vector<uint8_t> h;
    if (type == gf::kTypeBF16) {
        h.resize(w.size() * 2);
        ggml_fp32_to_bf16_row(w.data(), (ggml_bf16_t*) h.data(), (int64_t) w.size());
    } else if (type == 0) {
        h.resize(w.size() * 4);
        std::memcpy(h.data(), w.data(), h.size());
    } else {
        const ggml_type t = (ggml_type) type;
        h.resize(ggml_row_size(t, n_in) * (size_t) n_out);
        ggml_quantize_chunk(t, w.data(), h.data(), 0, n_out, n_in, nullptr);
    }
    m.bytes = h.size();
    m.n_copy = (int) std::max<size_t>(1, (size_t) (48u << 20) / m.bytes + 1);   // > 8x the V100's 6 MB L2
    CK(cudaMalloc(&m.d, m.bytes * (size_t) m.n_copy));
    for (int c = 0; c < m.n_copy; ++c) CK(cudaMemcpy(m.d + m.bytes * (size_t) c, h.data(), m.bytes, cudaMemcpyHostToDevice));
    m.h = std::move(h);
    return m;
}

// max |gpu - ref| / max |ref| of a headwise result: out[h * R + r] = W[h * R + r] . in[h * stride, + K)
double headwise_err(const Mat& m, const std::vector<float>& in, int n_head, int R, int K, int stride,
                    const std::vector<float>& got) {
    double md = 0, mx = 0;
    for (int h = 0; h < n_head; ++h)
        for (int r = 0; r < R; ++r) {
            const std::vector<double> w = m.row(h * R + r);
            double s = 0;
            for (int k = 0; k < K; ++k) s += w[(size_t) k] * (double) in[(size_t) h * stride + k];
            md = std::max(md, std::fabs(s - (double) got[(size_t) h * R + r]));
            mx = std::max(mx, std::fabs(s));
        }
    return md / std::max(mx, 1e-30);
}

struct Timer {
    cudaEvent_t a, b;
    Timer() {
        CK(cudaEventCreate(&a));
        CK(cudaEventCreate(&b));
    }
};

const char* tname(int t) {
    switch (t) {
        case 0: return "F32";
        case 8: return "Q8_0";
        case 12: return "Q4_K";
        case 13: return "Q5_K";
        case 14: return "Q6_K";
        case gf::kTypeBF16: return "BF16";
        default: return "?";
    }
}

}  // namespace

int main(int argc, char** argv) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("no GPU: skipped\n");
        return 77;
    }
    const int reps = argc > 1 ? std::max(4, std::atoi(argv[1])) : 40;
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
#if defined(STRATA_USE_HIP)
    std::printf("%s (%s), %d reps a timing, weights rotated over copies\n", prop.name, prop.gcnArchName, reps);
#else
    std::printf("%s (sm_%d%d), %d reps a timing, weights rotated over copies\n", prop.name, prop.major, prop.minor, reps);
#endif
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    Timer T;

    // GLM-5.3's decode mixer (n_embd 6144, q_lora 2048, kv_lora 512 + rope 64, 64 heads x (192 + 64), v 256, indexer
    // 32 heads x 128): the UD-IQ2_M GGUF's types, and the pack's (k_b / v_b / indexer upcast to BF16, compat-bf16)
    const int E = 6144, QL = 2048, KVR = 576, NH = 64, QH = 256, VH = 256, IH = 32, IK = 128, KVL = 512;
    struct Job {
        const char* name;
        int type, n_in, n_out;
        float alpha;
    };
    const Job jobs[] = {
        {"attn_q_a", 13, E, QL, 1.0f},          {"attn_kv_a_mqa", 8, E, KVR, 1.0f},
        {"indexer.attn_k (pack)", gf::kTypeBF16, E, IK, 1.0f}, {"indexer.attn_k (native)", 8, E, IK, 1.0f},
        {"indexer.proj (pack)", gf::kTypeBF16, E, IH, 0.5f},   {"indexer.proj (native)", 0, E, IH, 0.5f},
        {"attn_q_b", 8, QL, NH * QH, 1.0f},
        {"indexer.attn_q_b (pack)", gf::kTypeBF16, QL, IH * IK, 1.0f},
        {"indexer.attn_q_b (native)", 8, QL, IH * IK, 1.0f},
        {"attn_output", 13, NH * VH, E, 1.0f},
        // (the review's gap: Q4_K and Q6_K one column against gf::mv's one-column kernels - a Q4_K projection as other
        // quants of the model have it, the Q6_K shared / dense FFN downs; kMvDsaOn leaves a batch of K-quants only to
        // gf::mv and takes them in a mixed one, the fixed shapes 1-3 and 5 take them always)
        {"attn_q_a (Q4_K)", 12, E, QL, 1.0f},
        {"ffn_down_shexp (Q6_K)", 14, 2048, E, 1.0f},
        {"ffn_down dense (Q6_K)", 14, 12288, E, 1.0f},
        // the other one-column GEMVs glmdsa::mv runs on UD-IQ2_M: the dense layers' gate / up, the NextN eh_proj
        {"ffn_gate dense (Q5_K)", 13, E, 12288, 1.0f},
        {"ffn_up dense (Q5_K)", 13, E, 12288, 1.0f},
        {"nextn.eh_proj (Q8_0)", 8, 2 * E, E, 1.0f},
    };
    const int NJ = (int) (sizeof(jobs) / sizeof(jobs[0]));
    std::vector<Mat> mats;
    for (int i = 0; i < NJ; ++i) mats.push_back(make_mat(jobs[i].type, jobs[i].n_in, jobs[i].n_out, 1234u + i));

    // activations: x (n_embd) and qr (q_lora) and the attention output (64 x 256), f32 + q8_1
    const int XMAX = NH * VH;
    std::vector<float> xh(XMAX);
    {
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        for (float& v : xh) v = u(rng);
    }
    float* xf = nullptr;
    void* xq = nullptr;
    float *y = nullptr, *y2 = nullptr;
    CK(cudaMalloc(&xf, XMAX * 4));
    CK(cudaMalloc(&xq, (XMAX / 32) * 36));
    CK(cudaMalloc(&y, (size_t) gf::kMaxMvJobs * 16384 * 4));
    CK(cudaMalloc(&y2, (size_t) gf::kMaxMvJobs * 16384 * 4));
    CK(cudaMemcpy(xf, xh.data(), XMAX * 4, cudaMemcpyHostToDevice));
    gf::quantize_q8_1(xf, xq, XMAX, s);
    CK(cudaStreamSynchronize(s));

    const auto mk = [&](int i, int copy, float* out) {
        const Job& J = jobs[i];
        gf::MvJob j{mats[(size_t) i].at(copy), xq, xf, out, nullptr, J.alpha, J.type, J.n_in, J.n_out};
        return j;
    };
    using Fn = bool (*)(const gf::MvJob*, int, cudaStream_t);
    const auto time_jobs = [&](Fn fn, const std::vector<int>& idx) {
        // reps calls, each on the next copy of every weight; ms per call
        for (int w = 0; w < 3; ++w) {
            gf::MvJob j[gf::kMaxMvJobs];
            for (size_t k = 0; k < idx.size(); ++k) j[k] = mk(idx[k], w, y + (size_t) k * 16384);
            fn(j, (int) idx.size(), s);
        }
        CK(cudaEventRecord(T.a, s));
        for (int r = 0; r < reps; ++r) {
            gf::MvJob j[gf::kMaxMvJobs];
            for (size_t k = 0; k < idx.size(); ++k) j[k] = mk(idx[k], r, y + (size_t) k * 16384);
            fn(j, (int) idx.size(), s);
        }
        CK(cudaEventRecord(T.b, s));
        CK(cudaEventSynchronize(T.b));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, T.a, T.b));
        return (double) ms / reps;
    };
    const auto bytes_of = [&](const std::vector<int>& idx) {
        double b = 0;
        for (int i : idx) b += (double) mats[(size_t) i].bytes;
        return b;
    };
    std::printf("\n%-28s %-5s %6s x %-6s %8s | %9s %7s\n", "GEMV (one column)", "type", "out", "in", "MB", "us", "GB/s");
    for (int i = 0; i < NJ; ++i) {
        const double ms = time_jobs(&gf::mv, {i});
        std::printf("%-28s %-5s %6d x %-6d %8.2f | %9.1f %7.0f\n", jobs[i].name, tname(jobs[i].type), jobs[i].n_out,
                    jobs[i].n_in, mats[(size_t) i].bytes / 1048576.0, ms * 1e3, bytes_of({i}) / (ms * 1e-3) / 1e9);
    }
    // the mixer's launches (glm_dsa_mixer.hpp mixer_layer): a full layer's four input projections, a shared layer's
    // two, the query projections, attn_output - as the pack has them (BF16 indexer) and natively (B10.7)
    struct Set {
        const char* name;
        std::vector<int> idx;
    };
    const Set sets[] = {
        {"in-proj full (pack)", {0, 1, 2, 4}}, {"in-proj full (native)", {0, 1, 3, 5}}, {"in-proj shared", {0, 1}},
        {"q-proj full (pack)", {6, 7}},        {"q-proj full (native)", {6, 8}},        {"q-proj shared", {6}},
        {"attn_output", {9}},
        {"Q4_K one column", {10}},             {"Q6_K one column", {11}},               {"Q6_K long K", {12}},
        {"Q4_K + Q6_K + Q8_0", {10, 11, 1}},   {"attn_q_a alone (Q5_K)", {0}},          {"dense gate + up (Q5_K)", {13, 14}},
        {"nextn.eh_proj (Q8_0)", {15}},
    };
    // gf::mv (as before) against gf::mv_dsa (L1-4) in each shape: the outputs bit for bit, and the time.  auto = the
    // device's default (kMvDsaAuto: gf::mv alone where it is off), 1 = one launch of every job it takes, NR4 = 4-row
    // warps, typed = a launch a type, on = kMvDsaOn (the default's shape on any device: STRATA_GLMD_MV_DSA=1)
    static int s_cfg = 0;
    const Fn dsa = [](const gf::MvJob* j, int n, cudaStream_t st) { return gf::mv_dsa(j, n, st, s_cfg); };
    const int cfgs[] = {gf::kMvDsaAuto, 1, 2, 3, gf::kMvDsaOn, 5};
    std::printf("\n%-28s %8s | %9s %7s | mv_dsa us (GB/s): auto / 1 / NR4 / typed / on / NR1\n", "mixer launch", "MB",
                "mv us", "GB/s");
    for (const Set& st : sets) {
        const double ms = time_jobs(&gf::mv, st.idx);
        const double mb = bytes_of(st.idx);
        std::printf("%-28s %8.2f | %9.1f %7.0f |", st.name, mb / 1048576.0, ms * 1e3, mb / (ms * 1e-3) / 1e9);
        // the reference outputs (copy 0) through gf::mv
        std::vector<float> ref((size_t) gf::kMaxMvJobs * 16384), got(ref.size());
        {
            gf::MvJob j[gf::kMaxMvJobs];
            CK(cudaMemset(y, 0, ref.size() * 4));
            for (size_t k = 0; k < st.idx.size(); ++k) j[k] = mk(st.idx[k], 0, y + (size_t) k * 16384);
            gf::mv(j, (int) st.idx.size(), s);
            CK(cudaMemcpy(ref.data(), y, ref.size() * 4, cudaMemcpyDeviceToHost));
        }
        for (const int cfg : cfgs) {
            s_cfg = cfg;
            gf::MvJob j[gf::kMaxMvJobs];
            CK(cudaMemset(y, 0, got.size() * 4));
            for (size_t k = 0; k < st.idx.size(); ++k) j[k] = mk(st.idx[k], 0, y + (size_t) k * 16384);
            gf::mv_dsa(j, (int) st.idx.size(), s, cfg);
            CK(cudaMemcpy(got.data(), y, got.size() * 4, cudaMemcpyDeviceToHost));
            const bool same = std::memcmp(ref.data(), got.data(), ref.size() * 4) == 0;
#if defined(STRATA_USE_HIP)
            const bool gated = cfg == gf::kMvDsaAuto;   // (off by default on HIP: the forced shapes only inform)
#else
            const bool gated = true;
#endif
            g_fail += !same && gated;
            const double md = time_jobs(dsa, st.idx);
            std::printf(" %6.1f (%3.0f)%s", md * 1e3, mb / (md * 1e-3) / 1e9, same ? "" : gated ? " DIFFERS" : " differs");
        }
        std::printf("\n");
    }

    // the verify window (the one-GPU MTP decode) runs these GEMVs with a column a row: every column must be the one-column
    // GEMV's bit for bit (the token path's), or the window's picks can part from the token path's.  2, 3 and 4 columns
    // of the mixer's jobs and of the other GEMVs a glm-dsa window runs (dense FFN, router, shared expert, the head's
    // rows), against one column at a time (gf::mv and gf::mv_dsa, the token path's)
    {
        std::vector<Mat> extra;
        const Job xj[] = {
            {"ffn_gate (dense)", 13, E, 12288, 1.0f}, {"ffn_down (dense)", 14, 12288, E, 1.0f},
            {"ffn_gate_inp (router)", 0, E, 256, 1.0f}, {"ffn_gate_shexp", 13, E, 2048, 1.0f},
            {"ffn_down_shexp", 14, 2048, E, 1.0f},     {"output (16384 rows)", 8, E, 16384, 1.0f},
        };
        for (int i = 0; i < 6; ++i) extra.push_back(make_mat(xj[i].type, xj[i].n_in, xj[i].n_out, 777u + i));
        const int CMAX = 4, NIN = 16384;
        std::vector<float> xh4((size_t) CMAX * NIN);
        std::mt19937 rng(21);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        for (float& v : xh4) v = u(rng);
        float *xf4 = nullptr, *yr = nullptr, *yc = nullptr;
        void* xq4 = nullptr;
        CK(cudaMalloc(&xf4, xh4.size() * 4));
        CK(cudaMalloc(&xq4, (size_t) CMAX * (NIN / 32) * 36));
        CK(cudaMalloc(&yr, (size_t) CMAX * 16384 * 4));
        CK(cudaMalloc(&yc, (size_t) CMAX * 16384 * 4));
        CK(cudaMemcpy(xf4, xh4.data(), xh4.size() * 4, cudaMemcpyHostToDevice));
        std::printf("\n%-28s | window columns vs one column: 2 / 3 / 4 columns\n", "GEMV");
        const auto check = [&](const char* name, const Mat& m, float alpha) {
            const int n_in = m.n_in, n_out = m.n_out;
            const size_t xq_s = (size_t) n_in / 32 * 36;
            for (int c = 0; c < CMAX; ++c) gf::quantize_q8_1(xf4 + (size_t) c * n_in, (uint8_t*) xq4 + c * xq_s, n_in, s);
            std::printf("%-28s |", name);
            for (const int nc : {2, 3, 4}) {
                CK(cudaMemset(yr, 0, (size_t) CMAX * 16384 * 4));
                CK(cudaMemset(yc, 0xff, (size_t) CMAX * 16384 * 4));
                for (int c = 0; c < nc; ++c) {
                    gf::MvJob j{m.at(0), (uint8_t*) xq4 + c * xq_s, xf4 + (size_t) c * n_in, yr + (size_t) c * n_out,
                                nullptr, alpha, m.type, n_in, n_out};
                    gf::mv_dsa(&j, 1, s, 0);
                }
                gf::MvJob j{m.at(0), xq4, xf4, yc, nullptr, alpha, m.type, n_in, n_out};
                j.ncol = nc;
                j.xq_cstride = (int64_t) xq_s;
                j.xf_cstride = n_in;
                j.y_cstride = n_out;
                gf::mv(&j, 1, s);
                CK(cudaStreamSynchronize(s));
                std::vector<float> a((size_t) nc * n_out), b(a.size());
                CK(cudaMemcpy(a.data(), yr, a.size() * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(b.data(), yc, b.size() * 4, cudaMemcpyDeviceToHost));
                const bool same = std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
                double md = 0, mx = 0;
                for (size_t i = 0; i < a.size(); ++i) {
                    md = std::max(md, (double) std::fabs(a[i] - b[i]));
                    mx = std::max(mx, (double) std::fabs(a[i]));
                }
                g_fail += !same;
                if (same) std::printf("  same");
                else std::printf("  DIFFERS (%.1e)", md / std::max(mx, 1e-30));
            }
            std::printf("\n");
        };
        for (int i = 0; i < NJ; ++i) check(jobs[i].name, mats[(size_t) i], jobs[i].alpha);
        for (int i = 0; i < 6; ++i) check(xj[i].name, extra[(size_t) i], xj[i].alpha);
        CK(cudaFree(xf4));
        CK(cudaFree(xq4));
        CK(cudaFree(yr));
        CK(cudaFree(yc));
        for (auto& m : extra) CK(cudaFree(m.d));
    }

    // mla2 (k_b absorb, the attention over n_sel cells, v_b) - BF16 (pack) and Q8_0 (native) W_kb / W_vb
    std::printf("\nmla_dec on this device: %s\n",
                glmd::mla_dec_ok(NH, KVL, 64) ? "yes (new = mla_dec)"
                                              : "no (new = old: mla2's kernels; STRATA_GLMD_MLA_DEC=1 forces it)");
    {
        const int max_ctx = 4096;
        Mat kb16 = make_mat(gf::kTypeBF16, 192, NH * KVL, 99), vb16 = make_mat(gf::kTypeBF16, KVL, NH * VH, 98);
        Mat kb8 = make_mat(8, 192, NH * KVL, 97), vb8 = make_mat(8, KVL, NH * VH, 96);
        float* q = nullptr;
        uint16_t *lat = nullptr, *kpe = nullptr;
        int* cells = nullptr;
        void* work = nullptr;
        float* out = nullptr;
        void* outq = nullptr;
        CK(cudaMalloc(&q, (size_t) NH * QH * 4));
        CK(cudaMalloc(&lat, (size_t) max_ctx * KVL * 2));
        CK(cudaMalloc(&kpe, (size_t) max_ctx * 64 * 2));
        CK(cudaMalloc(&cells, max_ctx * 4));
        const size_t wb = glmd::mla2_work_bytes(NH, KVL, max_ctx);
        CK(cudaMalloc(&work, wb));
        CK(cudaMalloc(&out, (size_t) NH * VH * 4));
        CK(cudaMalloc(&outq, (size_t) NH * VH / 32 * 36));
        {
            std::vector<float> qh((size_t) NH * QH);
            std::mt19937 rng(5);
            std::uniform_real_distribution<float> u(-1.0f, 1.0f);
            for (float& v : qh) v = u(rng);
            CK(cudaMemcpy(q, qh.data(), qh.size() * 4, cudaMemcpyHostToDevice));
            std::vector<uint16_t> lh((size_t) max_ctx * KVL), ph((size_t) max_ctx * 64);
            for (auto& v : lh) v = ggml_fp32_to_fp16(u(rng) * 0.5f);
            for (auto& v : ph) v = ggml_fp32_to_fp16(u(rng) * 0.5f);
            CK(cudaMemcpy(lat, lh.data(), lh.size() * 2, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(kpe, ph.data(), ph.size() * 2, cudaMemcpyHostToDevice));
            std::vector<int> ch(max_ctx);
            for (int i = 0; i < max_ctx; ++i) ch[(size_t) i] = i;
            CK(cudaMemcpy(cells, ch.data(), ch.size() * 4, cudaMemcpyHostToDevice));
        }
        // INT8 latent records (Maya's: 512 codes + 16 FP16 scales) of the same latents, and a scattered selection
        uint8_t* lat8 = nullptr;
        CK(cudaMalloc(&lat8, (size_t) max_ctx * 544));
        {
            std::vector<uint16_t> lh((size_t) max_ctx * KVL);
            CK(cudaMemcpy(lh.data(), lat, lh.size() * 2, cudaMemcpyDeviceToHost));
            std::vector<uint8_t> rh((size_t) max_ctx * 544);
            for (int c = 0; c < max_ctx; ++c)
                for (int g = 0; g < KVL / 32; ++g) {
                    float amax = 0;
                    for (int j = 0; j < 32; ++j)
                        amax = std::max(amax, std::fabs(ggml_fp16_to_fp32(lh[(size_t) c * KVL + 32 * g + j])));
                    const float d = amax / 127.0f;
                    const uint16_t dh = ggml_fp32_to_fp16(d);
                    std::memcpy(&rh[(size_t) c * 544 + 512 + 2 * g], &dh, 2);
                    for (int j = 0; j < 32; ++j) {
                        const float v = ggml_fp16_to_fp32(lh[(size_t) c * KVL + 32 * g + j]);
                        rh[(size_t) c * 544 + 32 * g + j] = (uint8_t) (int8_t) (d > 0 ? std::lround(v / d) : 0);
                    }
                }
            CK(cudaMemcpy(lat8, rh.data(), rh.size(), cudaMemcpyHostToDevice));
            std::vector<int> ch(max_ctx);
            for (int i = 0; i < max_ctx; ++i) ch[(size_t) i] = i;
            std::mt19937 rng(3);
            std::shuffle(ch.begin(), ch.end(), rng);
            CK(cudaMemcpy(cells, ch.data(), ch.size() * 4, cudaMemcpyHostToDevice));
        }
        int* n_dev = nullptr;
        CK(cudaMalloc(&n_dev, 4));
        // mla2 as the decode mixer calls it (capacity 2048, the live count on the device) and with the host count: the
        // L1-4 attention (mla_dec) against mla2's own kernels (legacy_attn) - time and max |new - old| / max |old|
        std::printf("\n%-34s %6s | %9s %9s | %9s\n", "mla2 (W_kb + attention + W_vb)", "n_sel", "old us", "new us",
                    "rel diff");
        float* out2 = nullptr;
        CK(cudaMalloc(&out2, (size_t) NH * VH * 4));
        for (const int wt : {gf::kTypeBF16, 8})
            for (const int l8 : {1, 0})
                for (const int dev_count : {1, 0})
                    for (const int ns : {1, 64, 150, 256, 1000, 2048}) {
                        if (!dev_count && ns != 150 && ns != 2048) continue;
                        const Mat& kb = wt == 8 ? kb8 : kb16;
                        const Mat& vb = wt == 8 ? vb8 : vb16;
                        CK(cudaMemcpy(n_dev, &ns, 4, cudaMemcpyHostToDevice));
                        double us[2] = {0, 0};
                        for (int legacy = 1; legacy >= 0; --legacy) {
                            const auto run = [&](int c) {
                                glmd::Mla2 a;
                                a.q = q;
                                a.q_stride = QH;
                                a.wk_b = (const uint16_t*) kb.at(c);
                                a.wv_b = (const uint16_t*) vb.at(c);
                                a.lat = l8 ? (const void*) lat8 : (const void*) lat;
                                a.lat_q8 = l8;
                                a.k_pe = kpe;
                                a.cells = cells;
                                a.n_sel = dev_count ? 2048 : ns;
                                a.n_sel_dev = dev_count ? n_dev : nullptr;
                                a.work = work;
                                a.work_bytes = wb;
                                a.out = legacy ? out : out2;
                                a.out_q8_1 = outq;
                                a.w_type = wt;
                                a.legacy_attn = legacy;
                                if (!glmd::mla2(a, s)) {
                                    std::printf("mla2 refused\n");
                                    std::exit(1);
                                }
                            };
                            run(0);
                            for (int w = 0; w < 3; ++w) run(w);
                            CK(cudaEventRecord(T.a, s));
                            for (int r = 0; r < reps; ++r) run(r);
                            CK(cudaEventRecord(T.b, s));
                            CK(cudaEventSynchronize(T.b));
                            float ms = 0;
                            CK(cudaEventElapsedTime(&ms, T.a, T.b));
                            us[legacy] = 1e3 * (double) ms / reps;
                            run(0);   // (the compared outputs: copy 0's weights)
                        }
                        CK(cudaStreamSynchronize(s));
                        std::vector<float> o1((size_t) NH * VH), o2(o1.size());
                        CK(cudaMemcpy(o1.data(), out, o1.size() * 4, cudaMemcpyDeviceToHost));
                        CK(cudaMemcpy(o2.data(), out2, o2.size() * 4, cudaMemcpyDeviceToHost));
                        double md = 0, mx = 0;
                        for (size_t i = 0; i < o1.size(); ++i) {
                            md = std::max(md, (double) std::fabs(o1[i] - o2[i]));
                            mx = std::max(mx, (double) std::fabs(o1[i]));
                        }
                        const double rd = md / std::max(mx, 1e-30);
                        const bool ok = rd < 1e-4 && std::isfinite(rd);
                        g_fail += !ok;
                        char nm[64];
                        std::snprintf(nm, sizeof nm, "%s, %s lat%s", wt == 8 ? "Q8_0" : "BF16", l8 ? "INT8" : "FP16",
                                      dev_count ? ", dev count" : "");
                        std::printf("%-34s %6d | %9.1f %9.1f | %9.2e %s\n", nm, ns, us[1], us[0], rd, ok ? "ok" : "FAIL");
                    }
        CK(cudaFree(out2));
        CK(cudaFree(n_dev));
        CK(cudaFree(lat8));
#if !defined(STRATA_USE_HIP)   // the fused attention (STRATA_GLMD_FUSED_MLA, off by default; nvcuda::wmma) for comparison
        {
            const size_t fwb = glmd::mla_fused_work_bytes(NH, KVL, 64, max_ctx);
            void* fwork = nullptr;
            CK(cudaMalloc(&fwork, fwb));
            for (const int wt : {gf::kTypeBF16, 8})
                for (const int ns : {64, 256, 2048}) {
                    const Mat& kb = wt == 8 ? kb8 : kb16;
                    const Mat& vb = wt == 8 ? vb8 : vb16;
                    const auto run = [&](int c) {
                        glmd::FusedMla a;
                        a.q = q;
                        a.q_stride = QH;
                        a.wk_b = kb.at(c);
                        a.wv_b = vb.at(c);
                        a.w_type = wt;
                        a.lat = lat;
                        a.k_pe = kpe;
                        a.cells = cells;
                        a.n_sel = ns;
                        a.work = fwork;
                        a.work_bytes = fwb;
                        a.out = out;
                        a.out_q8_1 = outq;
                        if (!glmd::mla_fused(a, s)) {
                            std::printf("mla_fused refused\n");
                            std::exit(1);
                        }
                    };
                    for (int w = 0; w < 3; ++w) run(w);
                    CK(cudaEventRecord(T.a, s));
                    for (int r = 0; r < reps; ++r) run(r);
                    CK(cudaEventRecord(T.b, s));
                    CK(cudaEventSynchronize(T.b));
                    float ms = 0;
                    CK(cudaEventElapsedTime(&ms, T.a, T.b));
                    const double per = (double) ms / reps;
                    std::printf("%-28s %6d | %9.1f %7.0f\n", wt == 8 ? "mla_fused Q8_0" : "mla_fused BF16", ns,
                                per * 1e3, (double) (kb.bytes + vb.bytes) / (per * 1e-3) / 1e9);
                }
            CK(cudaFree(fwork));
        }
#endif
        // the two headwise GEMVs alone: W_kb (64 x 512 rows of 192, input q_nope at stride 256) and W_vb (64 x 256
        // rows of 512, input the context at stride 512) - glmd::headwise (B10.7); the error is
        // max |gpu - ref| / max |ref| against a double-precision dot of the same (dequantized) rows
        std::printf("\n%-28s %8s | %9s %7s | %9s\n", "headwise", "MB", "us", "GB/s", "err");
        float* hout = nullptr;
        CK(cudaMalloc(&hout, (size_t) NH * KVL * 4));
        std::vector<float> hin_h((size_t) NH * KVL);
        {
            std::mt19937 rng(11);
            std::uniform_real_distribution<float> u(-1.0f, 1.0f);
            for (float& v : hin_h) v = u(rng);
        }
        float* hin = nullptr;
        CK(cudaMalloc(&hin, hin_h.size() * 4));
        CK(cudaMemcpy(hin, hin_h.data(), hin_h.size() * 4, cudaMemcpyHostToDevice));
        for (const int wt : {gf::kTypeBF16, 8})
            for (const int which : {0, 1}) {
                const Mat& m = which == 0 ? (wt == 8 ? kb8 : kb16) : (wt == 8 ? vb8 : vb16);
                const int R = which == 0 ? KVL : VH, K = which == 0 ? 192 : KVL;
                const int stride = which == 0 ? QH : KVL;   // (in: n_head x stride floats, the first K of each used)
                const auto run = [&](int c) {
                    if (!glmd::headwise(m.at(c), wt, hin, NH, R, K, stride, hout, which == 1 ? outq : nullptr, s)) {
                        std::printf("headwise refused\n");
                        std::exit(1);
                    }
                };
                run(0);
                CK(cudaStreamSynchronize(s));
                std::vector<float> got((size_t) NH * R);
                CK(cudaMemcpy(got.data(), hout, got.size() * 4, cudaMemcpyDeviceToHost));
                const double er = headwise_err(m, hin_h, NH, R, K, stride, got);
                for (int w = 0; w < 3; ++w) run(w);
                CK(cudaEventRecord(T.a, s));
                for (int r = 0; r < reps; ++r) run(r);
                CK(cudaEventRecord(T.b, s));
                CK(cudaEventSynchronize(T.b));
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, T.a, T.b));
                const double us = 1e3 * (double) ms / reps;
                const bool ok = er < 1e-5;
                g_fail += !ok;
                std::printf("%-28s %8.2f | %9.1f %7.0f | %9.2e %s\n",
                            which == 0 ? (wt == 8 ? "W_kb Q8_0" : "W_kb BF16") : (wt == 8 ? "W_vb Q8_0" : "W_vb BF16"),
                            m.bytes / 1048576.0, us, (double) m.bytes / (us * 1e-6) / 1e9, er, ok ? "ok" : "FAIL");
            }
        CK(cudaFree(hout));
        CK(cudaFree(hin));
    }
    CK(cudaGetLastError());
    std::printf("%s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
