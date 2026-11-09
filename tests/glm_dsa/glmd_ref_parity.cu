// tests/glm_dsa/glmd_ref_parity.cu - glm-dsa F32 reference kernels (glmd::ref_*) vs the CPU reference (B3.4).
//
// GLM-5.3 sizes, query positions {0, 1, 2047, 2048, 131071, 1000000}:
//   rope      MLA q_pe (dims 192..255 of 64 heads x 256), k_pe (512..575 of 576), indexer q/k (0..63 of 128):
//             max abs err < 2e-6 (the B3.4 gate);
//   indexer   key, query, head weights, score over 3000 cached keys: max err < 1e-4 of the largest magnitude;
//   top-k     the GPU top-2048 of the GPU scores equals the CPU top-2048 of the same scores, exactly;
//   MLA       absorb + attention over the selected cells (scale 1/16): max err < 1e-4 of the largest magnitude.
// Needs a GPU; exits 77 (ctest SKIP_RETURN_CODE) when there is none.
#include "strata/kernels/glmd_ref.hpp"

#include "../../src/kernels/cuda/glmd_rope.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

template <typename T>
struct Dev {
    T* p = nullptr;
    size_t n = 0;
    explicit Dev(const std::vector<T>& h) : n(h.size()) {
        CK(cudaMalloc(&p, n * sizeof(T)));
        CK(cudaMemcpy(p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    }
    explicit Dev(size_t count) : n(count) { CK(cudaMalloc(&p, n * sizeof(T))); }
    ~Dev() { cudaFree(p); }
    std::vector<T> get() const {
        std::vector<T> h(n);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return h;
    }
};

std::vector<float> randv(std::mt19937& rng, size_t n, float scale = 1.0f) {
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

void gate_abs(const char* what, const std::vector<float>& gpu, const std::vector<float>& cpu, double lim) {
    double e = 0.0;
    for (size_t i = 0; i < gpu.size(); ++i) e = std::max(e, (double) std::fabs(gpu[i] - cpu[i]));
    const bool ok = e < lim;
    std::printf("%-44s max abs err %.3e (gate %.1e) %s\n", what, e, lim, ok ? "ok" : "FAIL");
    g_fail += !ok;
}

// error relative to the largest |cpu| value; -inf entries must match exactly
void gate_rel(const char* what, const std::vector<float>& gpu, const std::vector<float>& cpu, double lim) {
    double e = 0.0, den = 1e-30;
    bool inf_ok = true;
    for (size_t i = 0; i < gpu.size(); ++i) {
        if (std::isinf(cpu[i]) || std::isinf(gpu[i])) {
            inf_ok &= gpu[i] == cpu[i];
            continue;
        }
        e = std::max(e, (double) std::fabs(gpu[i] - cpu[i]));
        den = std::max(den, (double) std::fabs(cpu[i]));
    }
    const bool ok = inf_ok && e / den < lim;
    std::printf("%-44s max rel err %.3e (gate %.1e)%s %s\n", what, e / den, lim, inf_ok ? "" : " -inf mismatch",
                ok ? "ok" : "FAIL");
    g_fail += !ok;
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("glmd_ref_parity: no CUDA device, skipped\n");
        return 77;
    }

    // GLM-5.3
    const int n_embd = 6144, q_lora = 2048, n_head = 64, nope = 192, rope = 64, kv_lora = 512, v_head = 256;
    const int idx_heads = 32, idx_dim = 128, top_k = 2048, n_kv = 3000;
    const int qk_head = nope + rope, q_dim = kv_lora + rope;
    const std::vector<int> pos = {0, 1, 2047, 2048, 131071, 1000000};
    const int T = (int) pos.size();

    std::vector<float> inv(rope / 2);
    glmd::rope_inv_freq(inv.data(), rope, 8e6);
    Dev<float> d_inv(inv);
    Dev<int> d_pos(pos);
    std::mt19937 rng(53);

    // ---- rope on the three row shapes ----
    {
        struct Shape { const char* name; int row_len, rows, off; };
        const Shape shapes[] = {{"rope MLA q_pe (64 x 256, dims 192..255)", qk_head, n_head, nope},
                                {"rope MLA k_pe (576, dims 512..575)", q_dim, 1, kv_lora},
                                {"rope indexer q (32 x 128, dims 0..63)", idx_dim, idx_heads, 0},
                                {"rope indexer k (128, dims 0..63)", idx_dim, 1, 0}};
        for (const Shape& sh : shapes) {
            std::vector<float> x = randv(rng, (size_t) sh.row_len * sh.rows * T);
            Dev<float> d_x(x);
            glmd::ref_rope_norm(d_x.p, sh.row_len, sh.rows, T, sh.off, rope, d_pos.p, d_inv.p, nullptr);
            CK(cudaGetLastError());
            glmd::cpu::rope_norm(x.data(), sh.row_len, sh.rows, T, sh.off, rope, pos.data(), inv.data());
            gate_abs(sh.name, d_x.get(), x, 2e-6);
        }
    }

    // ---- indexer ----
    const auto x = randv(rng, (size_t) n_embd * T), ql = randv(rng, (size_t) q_lora * T);
    const auto wk = randv(rng, (size_t) idx_dim * n_embd, 0.02f), lnw = randv(rng, idx_dim);
    const auto lnb = randv(rng, idx_dim, 0.1f), wqb = randv(rng, (size_t) idx_heads * idx_dim * q_lora, 0.02f);
    const auto wp = randv(rng, (size_t) idx_heads * n_embd, 0.02f);
    Dev<float> d_x(x), d_ql(ql), d_wk(wk), d_lnw(lnw), d_lnb(lnb), d_wqb(wqb), d_wp(wp);
    Dev<float> d_ik((size_t) idx_dim * T), d_iq((size_t) idx_dim * idx_heads * T), d_w((size_t) idx_heads * T);
    glmd::ref_indexer_key(d_x.p, d_wk.p, d_lnw.p, d_lnb.p, 1e-6f, n_embd, idx_dim, T, rope, d_pos.p, d_inv.p, d_ik.p,
                          nullptr);
    glmd::ref_indexer_query(d_ql.p, d_wqb.p, q_lora, idx_heads, idx_dim, T, rope, d_pos.p, d_inv.p, d_iq.p, nullptr);
    glmd::ref_indexer_weights(d_x.p, d_wp.p, n_embd, idx_heads, idx_dim, T, d_w.p, nullptr);
    CK(cudaGetLastError());
    std::vector<float> ik((size_t) idx_dim * T), iq((size_t) idx_dim * idx_heads * T), w((size_t) idx_heads * T);
    glmd::cpu::indexer_key(x.data(), wk.data(), lnw.data(), lnb.data(), 1e-6f, n_embd, idx_dim, T, rope, pos.data(),
                           inv.data(), ik.data());
    glmd::cpu::indexer_query(ql.data(), wqb.data(), q_lora, idx_heads, idx_dim, T, rope, pos.data(), inv.data(),
                             iq.data());
    glmd::cpu::indexer_weights(x.data(), wp.data(), n_embd, idx_heads, idx_dim, T, w.data());
    gate_rel("indexer key (LayerNorm eps 1e-6 + rope)", d_ik.get(), ik, 1e-4);
    gate_rel("indexer query (32 heads + rope)", d_iq.get(), iq, 1e-4);
    gate_rel("indexer head weights (/ 64)", d_w.get(), w, 1e-4);

    // score over a 3000-key cache (keys are random unit-scale rows: the cache contents are not position-checked
    // here), from the GPU query and weights so the score kernel is compared on identical inputs
    const auto ik_cache = randv(rng, (size_t) idx_dim * n_kv);
    Dev<float> d_ikc(ik_cache), d_score((size_t) n_kv * T);
    const auto iq_g = d_iq.get(), w_g = d_w.get();
    glmd::ref_indexer_score(d_iq.p, d_ikc.p, d_w.p, idx_heads, idx_dim, T, n_kv, d_pos.p, d_score.p, nullptr);
    CK(cudaGetLastError());
    std::vector<float> score((size_t) n_kv * T);
    glmd::cpu::indexer_score(iq_g.data(), ik_cache.data(), w_g.data(), idx_heads, idx_dim, T, n_kv, pos.data(),
                             score.data());
    const auto score_g = d_score.get();
    gate_rel("indexer score (n_kv 3000, causal mask)", score_g, score, 1e-4);

    // ---- top-k: same scores in, same cells out ----
    Dev<int> d_sel((size_t) top_k * T);
    glmd::ref_topk(d_score.p, n_kv, T, top_k, d_sel.p, nullptr);
    CK(cudaGetLastError());
    std::vector<int> sel((size_t) top_k * T);
    glmd::cpu::topk(score_g.data(), n_kv, T, top_k, sel.data());
    const auto sel_g = d_sel.get();
    const bool topk_ok = sel_g == sel;
    std::printf("%-44s %s\n", "top-k 2048 (ties by the lower cell, -1 pad)", topk_ok ? "ok" : "FAIL");
    g_fail += !topk_ok;

    // ---- MLA over the selected cells ----
    auto q = randv(rng, (size_t) qk_head * n_head * T), kv = randv(rng, (size_t) q_dim * n_kv);
    const auto wkb = randv(rng, (size_t) nope * kv_lora * n_head, 0.05f);
    const auto wvb = randv(rng, (size_t) kv_lora * v_head * n_head, 0.05f);
    std::vector<int> kv_pos(n_kv);
    for (int s = 0; s < n_kv; ++s) kv_pos[s] = s;
    glmd::cpu::rope_norm(q.data(), qk_head, n_head, T, nope, rope, pos.data(), inv.data());
    glmd::cpu::rope_norm(kv.data(), q_dim, 1, n_kv, kv_lora, rope, kv_pos.data(), inv.data());
    Dev<float> d_q(q), d_kv(kv), d_wkb(wkb), d_wvb(wvb);
    Dev<float> d_Q((size_t) q_dim * n_head * T), d_out((size_t) v_head * n_head * T);
    glmd::ref_mla_absorb_q(d_q.p, d_wkb.p, n_head, nope, rope, kv_lora, T, d_Q.p, nullptr);
    CK(cudaGetLastError());
    std::vector<float> Q((size_t) q_dim * n_head * T), out((size_t) v_head * n_head * T);
    glmd::cpu::mla_absorb_q(q.data(), wkb.data(), n_head, nope, rope, kv_lora, T, Q.data());
    const auto Q_g = d_Q.get();
    gate_rel("MLA absorb Q = [W_kb^T q_nope | q_pe]", Q_g, Q, 1e-4);

    glmd::ref_mla_attention(d_Q.p, d_kv.p, d_sel.p, d_wvb.p, n_head, kv_lora, rope, v_head, T, top_k,
                            glmd::kGlm53KqScale, d_out.p, nullptr);
    CK(cudaGetLastError());
    glmd::cpu::mla_attention(Q_g.data(), kv.data(), sel.data(), wvb.data(), n_head, kv_lora, rope, v_head, T, top_k,
                             glmd::kGlm53KqScale, out.data());
    gate_rel("MLA attention (top-2048 cells, scale 1/16)", d_out.get(), out, 1e-4);

    if (g_fail) {
        std::fprintf(stderr, "glmd_ref_parity: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("glmd_ref_parity OK\n");
    return 0;
}
