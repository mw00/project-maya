// tests/glm_dsa/glm_dsa_ref_cpu_test.cpp - the glm-dsa reference forward's layer math on the CPU (glm-dsa B3.5).
//
// src/core/glm_dsa_ref.cu runs GLM-5.3 one token at a time with the glmd::ref_* kernels.  This test runs the SAME
// sequence of operations on the CPU (the glmd::cpu twins: double accumulation, float storage; the host router and
// policies of glmd_forward.hpp, which the GPU file calls too) over a tiny random glm-dsa model built in memory:
//
//   4 layers (full, shared, full, shared), n_embd 128, 4 heads of 32 + 16 rope, kv_lora 32, q_lora 64,
//   indexer 4 x 32 (16 roped), top-k 8, 8 experts top-2 + 1 shared, one dense lead layer, vocab 64.
//
// and checks the invariants the reference forward relies on:
//   1. the dense -> sparse transition at top_k: while p + 1 <= top_k every cell is attended and the indexer's
//      top-k (which then holds every cell) gives the same output; after, exactly top_k cells are attended and the
//      logits move away from an all-cells forward;
//   2. a shared layer's reused selection gives the same output as recomputing the selection when the two cell sets
//      agree (always in the dense regime; counted in the sparse one);
//   3. the absorbed attention equals the explicit (non-absorbed) per-head K/V attention on the forward's own data;
//   4. the router: ids distinct, weights sum to the scale (2.5), the bias selects but never weighs;
//   5. a layer split at a full layer reproduces the one-part logits bit for bit; a part that starts on a shared
//      layer is refused (no selection to reuse), as step_layers_dsa refuses it;
//   6. the host helpers: attn_scale(256) == 1/16, kv_carve, tile_rows.
#include "strata/kernels/glmd_forward.hpp"
#include "strata/kernels/glmd_ref.hpp"

#include "../../src/kernels/cuda/glmd_rope.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

int g_fail = 0;

void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

struct Geo {
    int n_layers = 4, n_embd = 128, n_head = 4, q_lora = 64, kv_lora = 32, qk_nope = 32, qk_rope = 16, v_head = 32;
    int idx_heads = 4, idx_key = 32, idx_rope = 16, top_k = 8;
    int n_expert = 8, n_used = 2, n_ff_exp = 32, n_shared = 1, n_ff_dense = 96, dense_lead = 1, n_vocab = 64;
    std::vector<int> full = {1, 0, 1, 0};
    float norm_eps = 1e-5f, idx_ln_eps = 1e-6f, w_scale = 2.5f;
    double theta = 8e6;
    int qk_head() const { return qk_nope + qk_rope; }
    int kv_row() const { return kv_lora + qk_rope; }
};

struct Layer {
    std::vector<float> attn_norm, q_a, q_a_norm, q_b, kv_a, kv_a_norm, k_b, v_b, wo, ffn_norm;
    std::vector<float> idx_k, idx_k_norm_w, idx_k_norm_b, idx_q_b, idx_proj;   // full layers
    std::vector<float> gate, up, down;                                          // dense lead
    std::vector<float> router, bias, gate_exps, up_exps, down_exps, gate_sh, up_sh, down_sh;
};

struct Model {
    Geo g;
    std::vector<float> tok_embd, out_norm, out;
    std::vector<Layer> L;
};

std::vector<float> rnd(std::mt19937& rng, size_t n, float scale, float mean = 0.0f) {
    std::normal_distribution<float> d(0.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = mean + scale * d(rng);
    return v;
}

Model make_model(uint32_t seed) {
    Model m;
    const Geo& g = m.g;
    std::mt19937 rng(seed);
    const size_t E = g.n_embd, H = g.n_head;
    const float sE = 1.0f / std::sqrt((float) E);
    m.tok_embd = rnd(rng, (size_t) g.n_vocab * E, 1.0f);
    m.out_norm = rnd(rng, E, 0.1f, 1.0f);
    m.out = rnd(rng, (size_t) g.n_vocab * E, sE);
    m.L.resize((size_t) g.n_layers);
    for (int il = 0; il < g.n_layers; ++il) {
        Layer& l = m.L[(size_t) il];
        l.attn_norm = rnd(rng, E, 0.1f, 1.0f);
        l.q_a = rnd(rng, (size_t) g.q_lora * E, sE);
        l.q_a_norm = rnd(rng, (size_t) g.q_lora, 0.1f, 1.0f);
        l.q_b = rnd(rng, H * g.qk_head() * (size_t) g.q_lora, 1.0f / std::sqrt((float) g.q_lora));
        l.kv_a = rnd(rng, (size_t) g.kv_row() * E, sE);
        l.kv_a_norm = rnd(rng, (size_t) g.kv_lora, 0.1f, 1.0f);
        l.k_b = rnd(rng, H * g.kv_lora * (size_t) g.qk_nope, 1.0f / std::sqrt((float) g.qk_nope));
        l.v_b = rnd(rng, H * g.v_head * (size_t) g.kv_lora, 1.0f / std::sqrt((float) g.kv_lora));
        l.wo = rnd(rng, E * H * g.v_head, 0.5f / std::sqrt((float) (H * g.v_head)));
        l.ffn_norm = rnd(rng, E, 0.1f, 1.0f);
        if (g.full[(size_t) il]) {
            l.idx_k = rnd(rng, (size_t) g.idx_key * E, sE);
            l.idx_k_norm_w = rnd(rng, (size_t) g.idx_key, 0.1f, 1.0f);
            l.idx_k_norm_b = rnd(rng, (size_t) g.idx_key, 0.1f);
            l.idx_q_b = rnd(rng, (size_t) g.idx_heads * g.idx_key * g.q_lora, 1.0f / std::sqrt((float) g.q_lora));
            l.idx_proj = rnd(rng, (size_t) g.idx_heads * E, sE);
        }
        if (il < g.dense_lead) {
            l.gate = rnd(rng, (size_t) g.n_ff_dense * E, sE);
            l.up = rnd(rng, (size_t) g.n_ff_dense * E, sE);
            l.down = rnd(rng, E * g.n_ff_dense, 0.5f / std::sqrt((float) g.n_ff_dense));
        } else {
            const size_t F = g.n_ff_exp, FS = (size_t) g.n_ff_exp * g.n_shared;
            l.router = rnd(rng, (size_t) g.n_expert * E, sE);
            l.bias = rnd(rng, (size_t) g.n_expert, 0.05f);
            l.gate_exps = rnd(rng, (size_t) g.n_expert * F * E, sE);
            l.up_exps = rnd(rng, (size_t) g.n_expert * F * E, sE);
            l.down_exps = rnd(rng, (size_t) g.n_expert * E * F, 0.5f / std::sqrt((float) F));
            l.gate_sh = rnd(rng, FS * E, sE);
            l.up_sh = rnd(rng, FS * E, sE);
            l.down_sh = rnd(rng, E * FS, 0.5f / std::sqrt((float) FS));
        }
    }
    return m;
}

void rms(const float* x, const float* w, float* y, int n, float eps) {
    double acc = 0.0;
    for (int i = 0; i < n; ++i) acc += (double) x[i] * (double) x[i];
    const float inv = (float) (1.0 / std::sqrt(acc / n + (double) eps));
    for (int i = 0; i < n; ++i) y[i] = x[i] * inv * w[i];
}

void silu_mul(const float* g, const float* u, float* h, int n) {
    for (int i = 0; i < n; ++i) h[i] = (float) ((double) g[i] / (1.0 + std::exp(-(double) g[i])) * (double) u[i]);
}

// the explicit (non-absorbed) attention for one token: k_h(s) = [W_kb_h c_kv(s) | k_pe(s)], v_h(s) = W_vb_h c_kv(s)
void attention_explicit(const Geo& g, const float* q, const float* kv, const int* cells, int n_sel, const float* k_b,
                        const float* v_b, float scale, float* out) {
    const int H = g.n_head, QH = g.qk_head(), KL = g.kv_lora, NOPE = g.qk_nope, R = g.qk_rope, VH = g.v_head;
    for (int h = 0; h < H; ++h) {
        const float* qh = q + (size_t) h * QH;
        std::vector<double> logit, vals;
        std::vector<int> used;
        double mx = -1e300;
        for (int s = 0; s < n_sel; ++s) {
            if (cells[s] < 0) continue;
            const float* row = kv + (size_t) cells[s] * g.kv_row();
            double d = 0.0;
            for (int c = 0; c < NOPE; ++c) {
                double kc = 0.0;   // k_nope[c] = sum_r k_b[h][r][c] c_kv[r]
                for (int r = 0; r < KL; ++r) kc += (double) k_b[((size_t) h * KL + r) * NOPE + c] * row[r];
                d += (double) qh[c] * kc;
            }
            for (int i = 0; i < R; ++i) d += (double) qh[NOPE + i] * row[KL + i];
            logit.push_back(d * scale);
            used.push_back(cells[s]);
            mx = std::max(mx, logit.back());
        }
        double sum = 0.0;
        std::vector<double> acc((size_t) VH, 0.0);
        for (size_t j = 0; j < used.size(); ++j) {
            const double p = std::exp(logit[j] - mx);
            sum += p;
            const float* row = kv + (size_t) used[j] * g.kv_row();
            for (int v = 0; v < VH; ++v) {
                double vv = 0.0;
                for (int r = 0; r < KL; ++r) vv += (double) v_b[((size_t) h * VH + v) * KL + r] * row[r];
                acc[(size_t) v] += p * vv;
            }
        }
        for (int v = 0; v < VH; ++v) out[(size_t) h * VH + v] = sum > 0 ? (float) (acc[(size_t) v] / sum) : 0.0f;
    }
}

struct Stats {
    double absorb_max_rel = 0.0;
    int router_checks = 0, router_bad = 0;
    int shared_same = 0, shared_same_sparse = 0, shared_diff = 0;
    double shared_same_max = 0.0;
    int transition_bad = 0, sel_count_bad = 0;
    double dense_vs_topk_max = 0.0;
};

struct Opts {
    bool force_dense = false;     // attend every cell regardless of top_k (an all-cells forward)
    bool shared_check = false;    // shared layers: also recompute a selection and compare
    bool checks = false;          // absorbed vs explicit, router invariants, dense vs top-k in the dense regime
};

// One part of the model ([l0, l1)) on the CPU, mirroring Glm5Model::step_layers_dsa line for line.
struct Part {
    const Model* m;
    int l0, l1;
    std::vector<std::vector<float>> kv, ik, ik_shared;   // per layer: [c_kv | k_pe] rows, index keys
    std::vector<float> inv_q, inv_idx;
    int max_ctx;

    Part(const Model& model, int a, int b, int ctx) : m(&model), l0(a), l1(b), max_ctx(ctx) {
        const Geo& g = m->g;
        kv.assign((size_t) g.n_layers, std::vector<float>((size_t) ctx * g.kv_row()));
        ik.assign((size_t) g.n_layers, std::vector<float>((size_t) ctx * g.idx_key));
        ik_shared.assign((size_t) g.n_layers, std::vector<float>((size_t) ctx * g.idx_key));
        inv_q.resize((size_t) g.qk_rope / 2);
        inv_idx.resize((size_t) g.idx_rope / 2);
        glmd::rope_inv_freq(inv_q.data(), g.qk_rope, g.theta);
        glmd::rope_inv_freq(inv_idx.data(), g.idx_rope, g.theta);
    }

    // indexer for layer il with layer `wl`'s weights: key at p into `cache`, selection into sel
    int select(int wl, const float* x, const float* qr, int p, std::vector<float>& cache, std::vector<int>& sel,
               bool force_dense) const {
        const Geo& g = m->g;
        const Layer& w = m->L[(size_t) wl];
        const int n_kv = p + 1;
        glmd::cpu::indexer_key(x, w.idx_k.data(), w.idx_k_norm_w.data(), w.idx_k_norm_b.data(), g.idx_ln_eps,
                               g.n_embd, g.idx_key, 1, g.idx_rope, &p, inv_idx.data(), cache.data() + (size_t) p * g.idx_key);
        if (force_dense || glmd::attend_dense(n_kv, g.top_k)) {
            sel.resize((size_t) n_kv);
            for (int s = 0; s < n_kv; ++s) sel[(size_t) s] = s;
            return n_kv;
        }
        std::vector<float> iq((size_t) g.idx_heads * g.idx_key), iw((size_t) g.idx_heads), score((size_t) n_kv);
        glmd::cpu::indexer_query(qr, w.idx_q_b.data(), g.q_lora, g.idx_heads, g.idx_key, 1, g.idx_rope, &p,
                                 inv_idx.data(), iq.data());
        glmd::cpu::indexer_weights(x, w.idx_proj.data(), g.n_embd, g.idx_heads, g.idx_key, 1, iw.data());
        glmd::cpu::indexer_score(iq.data(), cache.data(), iw.data(), g.idx_heads, g.idx_key, 1, n_kv, &p, score.data());
        sel.resize((size_t) g.top_k);
        glmd::cpu::topk(score.data(), n_kv, 1, g.top_k, sel.data());
        return g.top_k;
    }

    // the same selection through the top-k path even in the dense regime (for the transition check)
    std::vector<int> topk_all(int wl, const float* x, const float* qr, int p) const {
        const Geo& g = m->g;
        const Layer& w = m->L[(size_t) wl];
        const int n_kv = p + 1;
        std::vector<float> iq((size_t) g.idx_heads * g.idx_key), iw((size_t) g.idx_heads), score((size_t) n_kv);
        glmd::cpu::indexer_query(qr, w.idx_q_b.data(), g.q_lora, g.idx_heads, g.idx_key, 1, g.idx_rope, &p,
                                 inv_idx.data(), iq.data());
        glmd::cpu::indexer_weights(x, w.idx_proj.data(), g.n_embd, g.idx_heads, g.idx_key, 1, iw.data());
        glmd::cpu::indexer_score(iq.data(), ik[(size_t) wl].data(), iw.data(), g.idx_heads, g.idx_key, 1, n_kv, &p,
                                 score.data());
        std::vector<int> sel((size_t) g.top_k);
        glmd::cpu::topk(score.data(), n_kv, 1, g.top_k, sel.data());
        return sel;
    }

    // R: the residual (n_embd), updated in place; returns false (with why) like step_layers_dsa
    bool step(std::vector<float>& R, int p, const Opts& o, Stats& st, std::string& why) {
        const Geo& g = m->g;
        const int E = g.n_embd, QL = g.q_lora, KL = g.kv_lora, H = g.n_head, QH = g.qk_head(), KVR = g.kv_row();
        const int VH = g.v_head, NOPE = g.qk_nope, ROPE = g.qk_rope;
        const float scale = glmd::attn_scale(QH);
        std::vector<float> x((size_t) E), qr((size_t) QL), q((size_t) H * QH), Q((size_t) H * KVR),
            attn((size_t) H * VH), mix((size_t) E), ffn((size_t) E);
        std::vector<int> sel;
        int n_sel = -1, last_full = -1;
        for (int il = l0; il < l1; ++il) {
            const Layer& w = m->L[(size_t) il];
            rms(R.data(), w.attn_norm.data(), x.data(), E, g.norm_eps);
            glmd::cpu::gemv(w.q_a.data(), x.data(), qr.data(), E, QL, 1, 1.0f);
            rms(qr.data(), w.q_a_norm.data(), qr.data(), QL, g.norm_eps);
            std::vector<int> sel_re;
            int n_re = -1;
            if (g.full[(size_t) il]) {
                n_sel = select(il, x.data(), qr.data(), p, ik[(size_t) il], sel, o.force_dense);
                last_full = il;
                if (n_sel != std::min(p + 1, o.force_dense ? p + 1 : g.top_k)) ++st.sel_count_bad;
                if (o.checks && !o.force_dense && glmd::attend_dense(p + 1, g.top_k)) {
                    // the dense regime: the top-k path selects exactly the cells 0..p
                    const std::vector<int> tk = topk_all(il, x.data(), qr.data(), p);
                    std::set<int> a(tk.begin(), tk.end());
                    a.erase(-1);
                    std::set<int> b(sel.begin(), sel.begin() + n_sel);
                    if (a != b) ++st.transition_bad;
                }
            } else {
                if (n_sel < 0) {
                    why = "shared-indexer layer " + std::to_string(il) + " has no full layer before it in this part";
                    return false;
                }
                if (o.shared_check)
                    n_re = select(last_full, x.data(), qr.data(), p, ik_shared[(size_t) il], sel_re, o.force_dense);
            }
            glmd::cpu::gemv(w.q_b.data(), qr.data(), q.data(), QL, H * QH, 1, 1.0f);
            glmd::cpu::rope_norm(q.data(), QH, H, 1, NOPE, ROPE, &p, inv_q.data());
            float* kv_p = kv[(size_t) il].data() + (size_t) p * KVR;
            glmd::cpu::gemv(w.kv_a.data(), x.data(), kv_p, E, KVR, 1, 1.0f);
            glmd::cpu::rope_norm(kv_p, KVR, 1, 1, KL, ROPE, &p, inv_q.data());
            rms(kv_p, w.kv_a_norm.data(), kv_p, KL, g.norm_eps);
            glmd::cpu::mla_absorb_q(q.data(), w.k_b.data(), H, NOPE, ROPE, KL, 1, Q.data());
            glmd::cpu::mla_attention(Q.data(), kv[(size_t) il].data(), sel.data(), w.v_b.data(), H, KL, ROPE, VH, 1,
                                     n_sel, scale, attn.data());
            if (o.checks) {
                std::vector<float> ex((size_t) H * VH);
                attention_explicit(g, q.data(), kv[(size_t) il].data(), sel.data(), n_sel, w.k_b.data(), w.v_b.data(),
                                   scale, ex.data());
                double num = 0, den = 0;
                for (size_t i = 0; i < ex.size(); ++i) {
                    num = std::max(num, (double) std::fabs(ex[i] - attn[i]));
                    den = std::max(den, (double) std::fabs(ex[i]));
                }
                st.absorb_max_rel = std::max(st.absorb_max_rel, num / std::max(den, 1e-30));
                if (g.full[(size_t) il] && glmd::attend_dense(p + 1, g.top_k) && !o.force_dense) {
                    // top-k order instead of 0..p: the same cells, the same output up to summation order
                    const std::vector<int> tk = topk_all(il, x.data(), qr.data(), p);
                    std::vector<float> a2((size_t) H * VH);
                    glmd::cpu::mla_attention(Q.data(), kv[(size_t) il].data(), tk.data(), w.v_b.data(), H, KL, ROPE, VH,
                                             1, g.top_k, scale, a2.data());
                    for (size_t i = 0; i < a2.size(); ++i)
                        st.dense_vs_topk_max = std::max(st.dense_vs_topk_max, (double) std::fabs(a2[i] - attn[i]));
                }
            }
            if (n_re >= 0) {
                std::set<int> a(sel.begin(), sel.begin() + n_sel), b(sel_re.begin(), sel_re.begin() + n_re);
                a.erase(-1);
                b.erase(-1);
                if (a == b) {
                    std::vector<float> a2((size_t) H * VH);
                    glmd::cpu::mla_attention(Q.data(), kv[(size_t) il].data(), sel_re.data(), w.v_b.data(), H, KL, ROPE,
                                             VH, 1, n_re, scale, a2.data());
                    double d = 0;
                    for (size_t i = 0; i < a2.size(); ++i) d = std::max(d, (double) std::fabs(a2[i] - attn[i]));
                    st.shared_same_max = std::max(st.shared_same_max, d);
                    ++st.shared_same;
                    if (!glmd::attend_dense(p + 1, g.top_k)) ++st.shared_same_sparse;
                } else {
                    ++st.shared_diff;
                }
            }
            glmd::cpu::gemv(w.wo.data(), attn.data(), mix.data(), H * VH, E, 1, 1.0f);
            for (int i = 0; i < E; ++i) R[(size_t) i] += mix[(size_t) i];

            rms(R.data(), w.ffn_norm.data(), x.data(), E, g.norm_eps);
            if (il < g.dense_lead) {
                const int FD = g.n_ff_dense;
                std::vector<float> ga((size_t) FD), up((size_t) FD), h((size_t) FD);
                glmd::cpu::gemv(w.gate.data(), x.data(), ga.data(), E, FD, 1, 1.0f);
                glmd::cpu::gemv(w.up.data(), x.data(), up.data(), E, FD, 1, 1.0f);
                silu_mul(ga.data(), up.data(), h.data(), FD);
                glmd::cpu::gemv(w.down.data(), h.data(), ffn.data(), FD, E, 1, 1.0f);
            } else {
                const int NX = g.n_expert, K = g.n_used, F = g.n_ff_exp, FS = g.n_ff_exp * g.n_shared;
                std::vector<float> lg((size_t) NX), rw((size_t) K);
                std::vector<int> ids((size_t) K);
                glmd::cpu::gemv(w.router.data(), x.data(), lg.data(), E, NX, 1, 1.0f);
                glmd::route_sigmoid(lg.data(), w.bias.data(), NX, K, true, g.w_scale, ids.data(), rw.data());
                if (o.checks) {
                    ++st.router_checks;
                    double sum = 0;
                    for (float v : rw) sum += v;
                    std::set<int> u(ids.begin(), ids.end());
                    bool ok = std::fabs(sum - g.w_scale) < 1e-5 && (int) u.size() == K;
                    // the weights are sigmoid(logit) (never + bias), normalized
                    double ps = 0;
                    for (int i = 0; i < K; ++i) ps += 1.0 / (1.0 + std::exp(-(double) lg[(size_t) ids[(size_t) i]]));
                    for (int i = 0; i < K; ++i) {
                        const double want = 1.0 / (1.0 + std::exp(-(double) lg[(size_t) ids[(size_t) i]])) / ps * g.w_scale;
                        ok = ok && std::fabs(want - rw[(size_t) i]) < 1e-6;
                    }
                    if (!ok) ++st.router_bad;
                }
                std::fill(ffn.begin(), ffn.end(), 0.0f);
                std::vector<float> ga((size_t) std::max(F, FS)), up(ga.size()), h(ga.size()), dn((size_t) E);
                for (int i = 0; i < K; ++i) {
                    const size_t e = (size_t) ids[(size_t) i];
                    glmd::cpu::gemv(w.gate_exps.data() + e * F * E, x.data(), ga.data(), E, F, 1, 1.0f);
                    glmd::cpu::gemv(w.up_exps.data() + e * F * E, x.data(), up.data(), E, F, 1, 1.0f);
                    silu_mul(ga.data(), up.data(), h.data(), F);
                    glmd::cpu::gemv(w.down_exps.data() + e * E * F, h.data(), dn.data(), F, E, 1, 1.0f);
                    for (int j = 0; j < E; ++j) ffn[(size_t) j] += rw[(size_t) i] * dn[(size_t) j];
                }
                glmd::cpu::gemv(w.gate_sh.data(), x.data(), ga.data(), E, FS, 1, 1.0f);
                glmd::cpu::gemv(w.up_sh.data(), x.data(), up.data(), E, FS, 1, 1.0f);
                silu_mul(ga.data(), up.data(), h.data(), FS);
                glmd::cpu::gemv(w.down_sh.data(), h.data(), dn.data(), FS, E, 1, 1.0f);
                for (int j = 0; j < E; ++j) ffn[(size_t) j] += dn[(size_t) j];   // the shared expert, unscaled
            }
            for (int i = 0; i < E; ++i) R[(size_t) i] += ffn[(size_t) i];
        }
        return true;
    }
};

std::vector<float> head(const Model& m, const std::vector<float>& R) {
    const Geo& g = m.g;
    std::vector<float> h((size_t) g.n_embd), lg((size_t) g.n_vocab);
    rms(R.data(), m.out_norm.data(), h.data(), g.n_embd, g.norm_eps);
    glmd::cpu::gemv(m.out.data(), h.data(), lg.data(), g.n_embd, g.n_vocab, 1, 1.0f);
    return lg;
}

// a whole sequence through parts split at `bounds`; logits per position
bool run(const Model& m, const std::vector<int>& toks, const std::vector<int>& bounds, const Opts& o, Stats& st,
         std::vector<std::vector<float>>& logits, std::string& why) {
    const Geo& g = m.g;
    std::vector<Part> parts;
    int a = 0;
    for (int b : bounds) {
        parts.emplace_back(m, a, b, (int) toks.size());
        a = b;
    }
    parts.emplace_back(m, a, g.n_layers, (int) toks.size());
    logits.clear();
    for (size_t t = 0; t < toks.size(); ++t) {
        std::vector<float> R(m.tok_embd.begin() + (size_t) toks[t] * g.n_embd,
                             m.tok_embd.begin() + (size_t) (toks[t] + 1) * g.n_embd);
        for (auto& pt : parts)
            if (!pt.step(R, (int) t, o, st, why)) return false;
        logits.push_back(head(m, R));
    }
    return true;
}

double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0;
    for (size_t i = 0; i < a.size(); ++i) d = std::max(d, (double) std::fabs(a[i] - b[i]));
    return d;
}

}  // namespace

int main() {
    std::printf("glm_dsa_ref_cpu_test: the glm-dsa reference forward's layer math on a tiny random model\n");
    const Model m = make_model(20261011u);
    const Geo& g = m.g;
    std::mt19937 rng(7);
    std::vector<int> toks(24);
    for (auto& t : toks) t = (int) (rng() % (uint32_t) g.n_vocab);

    Stats st;
    Opts o;
    o.checks = true;
    o.shared_check = true;
    std::vector<std::vector<float>> lg, lg2, lg_dense, lg_split;
    std::string why;
    check(run(m, toks, {}, o, st, lg, why), "the 4-layer model runs 24 tokens in one part " + why);
    bool finite = true;
    for (const auto& v : lg)
        for (float x : v) finite = finite && std::isfinite(x);
    check(finite, "every logit is finite");

    std::printf("[1] dense -> sparse transition at top_k = %d\n", g.top_k);
    check(st.sel_count_bad == 0, "full layers attend min(p + 1, top_k) cells at every position");
    check(st.transition_bad == 0, "p + 1 <= top_k: the indexer's top-k holds exactly the cells 0..p");
    check(st.dense_vs_topk_max < 1e-6,
          "p + 1 <= top_k: attention over 0..p == over the top-k order (max " + std::to_string(st.dense_vs_topk_max) + ")");
    {
        Stats s2;
        Opts od;
        od.force_dense = true;
        check(run(m, toks, {}, od, s2, lg_dense, why), "the all-cells forward runs");
        double before = 0, after = 0;
        for (int t = 0; t < (int) toks.size(); ++t) {
            const double d = max_diff(lg[(size_t) t], lg_dense[(size_t) t]);
            (t + 1 <= g.top_k ? before : after) = std::max(t + 1 <= g.top_k ? before : after, d);
        }
        check(before == 0.0, "positions 0..top_k-1: bit-identical to the all-cells forward");
        check(after > 1e-4, "positions >= top_k: the selection changes the logits (max diff " + std::to_string(after) + ")");
    }

    std::printf("[2] shared layers reuse the previous full layer's selection\n");
    check(st.shared_same > 0 && st.shared_same_max < 1e-6,
          "same cell set -> reused == recomputed attention (" + std::to_string(st.shared_same) + " cases, " +
              std::to_string(st.shared_same_sparse) + " sparse; max " + std::to_string(st.shared_same_max) + ")");
    std::printf("    (%d sparse cases where a recomputed selection differs: the reuse is a real approximation)\n",
                st.shared_diff);

    std::printf("[3] absorbed MLA == explicit per-head K/V attention\n");
    check(st.absorb_max_rel < 1e-5, "max rel diff " + std::to_string(st.absorb_max_rel));

    std::printf("[4] router\n");
    check(st.router_checks > 0 && st.router_bad == 0,
          "ids distinct, weights = sigmoid / sum * 2.5 (sum 2.5), bias never weighs (" +
              std::to_string(st.router_checks) + " routes)");
    {
        // a bias that forces expert 5 in: selected, but weighted by its sigmoid only
        std::vector<float> lg8 = {2.0f, 1.5f, 1.0f, 0.5f, 0.0f, -3.0f, -0.5f, -1.0f}, bias(8, 0.0f);
        bias[5] = 10.0f;
        int ids[2];
        float w[2];
        glmd::route_sigmoid(lg8.data(), bias.data(), 8, 2, true, 2.5f, ids, w);
        const double s5 = 1.0 / (1.0 + std::exp(3.0)), s0 = 1.0 / (1.0 + std::exp(-2.0));
        check(ids[0] == 5 && ids[1] == 0 && std::fabs(w[0] - s5 / (s5 + s0) * 2.5) < 1e-6,
              "bias picks expert 5, its weight is sigmoid(-3) / sum * 2.5");
        std::vector<float> tie(8, 0.0f);
        glmd::route_sigmoid(tie.data(), nullptr, 8, 2, false, 1.0f, ids, w);
        check(ids[0] == 0 && ids[1] == 1 && std::fabs(w[0] - 0.5f) < 1e-7, "ties by the lower id; unnormalized = sigmoid");
    }

    std::printf("[5] layer split\n");
    {
        Stats s3;
        Opts op;
        check(run(m, toks, {2}, op, s3, lg_split, why), "split at layer 2 (full) runs");
        double d = 0;
        for (size_t t = 0; t < lg.size(); ++t) d = std::max(d, max_diff(lg[t], lg_split[t]));
        check(d == 0.0, "the split logits equal the one-part logits bit for bit");
        Stats s4;
        why.clear();
        const bool ok = run(m, toks, {1}, op, s4, lg_split, why);
        check(!ok && why.find("shared-indexer layer 1") != std::string::npos,
              "a part starting on shared layer 1 is refused (" + why + ")");
        Stats s5;
        check(run(m, toks, {}, op, s5, lg2, why), "a second run");
        double d2 = 0;
        for (size_t t = 0; t < lg.size(); ++t) d2 = std::max(d2, max_diff(lg[t], lg2[t]));
        check(d2 == 0.0, "deterministic");
    }

    std::printf("[6] host helpers\n");
    check(glmd::attn_scale(256) == glmd::kGlm53KqScale && glmd::attn_scale(256) == 0.0625f,
          "attn_scale(192 + 64) == kGlm53KqScale == 1/16");
    check(std::fabs(glmd::attn_scale(48) - 1.0 / std::sqrt(48.0)) < 1e-7, "attn_scale(48) == 1/sqrt(48)");
    {
        // Maya's slow-path arena span per layer: [lat 512 C | ik 128 C | ig 128 C | pool 128 C], C = 4096
        const int64_t C = 4096, lat = 1000;
        glmd::KvCarve c;
        bool ok = glmd::kv_carve(lat, lat + 512 * C, lat + 640 * C, lat + 768 * C, 512 * C, C, C, 128, 576, C, true, c);
        check(ok && c.kv == lat && c.ik == lat + 576 * C, "kv_carve: [576 | 128] x C fits the 896 x C span (full layer)");
        ok = glmd::kv_carve(lat, lat + 512 * C, lat + 640 * C, lat + 768 * C, 512 * C, C, C, 128, 576, C, false, c);
        check(ok && c.kv == lat && c.ik == -1, "kv_carve: shared layer, no index keys");
        ok = glmd::kv_carve(lat, lat + 512 * C + 1, lat + 640 * C, lat + 768 * C, 512 * C, C, C, 128, 576, C, true, c);
        check(!ok, "kv_carve: a gap in the span is refused");
        ok = glmd::kv_carve(lat, lat + 256 * C, lat + 256 * C + 128 * 64, lat + 256 * C + 2 * 128 * 64, 256 * C, 64,
                            64, 128, 576, C, true, c);
        check(!ok, "kv_carve: a span too small (ring 64) is refused");
    }
    check(glmd::tile_rows(6144, (int64_t) 128 << 20) == 21845 && glmd::tile_rows(6144, 6143) == 0,
          "tile_rows: 0.5 GB of F32 holds 21845 rows of 6144; less than a row holds none");

    std::printf(g_fail ? "glm_dsa_ref_cpu_test: %d FAILED\n" : "glm_dsa_ref_cpu_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
