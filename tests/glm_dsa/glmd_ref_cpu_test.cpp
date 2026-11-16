// tests/glm_dsa/glmd_ref_cpu_test.cpp - the glm-dsa CPU reference (B3.4) against first principles.
//
// No GPU: checks the shared NORM rotation (glmd_rope.cuh) and glmd::cpu::* against naive double loops written
// here, independently of the library:
//   rope    inv_freq values, position 0 is the identity, each pair keeps its norm, adjacent pairing with the
//           (c, -s; s, c) sign convention, and q.k depends only on the position difference;
//   indexer key LayerNorm + rope, query rope per head, head-weight scale, ReLU-weighted score with the causal mask;
//   top-k   descending score, ties by the lower cell, -inf excluded, -1 padding, n <= k returns every key;
//   MLA     absorbed attention equals standard attention on K/V decompressed with W_kb / W_vb.
#include "strata/kernels/glmd_ref.hpp"

#include "../../src/kernels/cuda/glmd_rope.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

namespace {

int g_fail = 0;

void check(bool ok, const char* what, double got = 0.0, double lim = 0.0) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s (%.3g vs limit %.3g)\n", what, got, lim);
        ++g_fail;
    }
}

std::vector<float> randv(std::mt19937& rng, size_t n, float scale = 1.0f) {
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

double max_rel(const std::vector<float>& a, const std::vector<double>& b) {
    double num = 0.0, den = 1e-30;
    for (size_t i = 0; i < a.size(); ++i) {
        num = std::max(num, std::fabs((double) a[i] - b[i]));
        den = std::max(den, std::fabs(b[i]));
    }
    return num / den;
}

constexpr int kRot = 64;
constexpr double kTheta = 8e6;
const int kPositions[] = {0, 1, 2047, 2048, 131071, 1000000};

void test_rope() {
    float inv[kRot / 2];
    glmd::rope_inv_freq(inv, kRot, kTheta);
    check(inv[0] == 1.0f, "inv_freq[0] == 1");
    for (int i = 0; i < kRot / 2; ++i)
        check(inv[i] == (float) std::pow(kTheta, -2.0 * i / kRot), "inv_freq rounded once from double");
    check(inv[kRot / 2 - 1] < inv[1] && inv[1] < 1.0f, "inv_freq decreasing");

    std::mt19937 rng(11);
    const std::vector<float> x0 = randv(rng, kRot);

    // position 0: cos 0 = 1, sin 0 = 0, so the rotation is exactly the identity
    std::vector<float> x = x0;
    glmd::rope_row(x.data(), kRot, 0, inv);
    check(x == x0, "rope at position 0 is the identity");

    for (int p : kPositions) {
        x = x0;
        glmd::rope_row(x.data(), kRot, p, inv);
        double err_norm = 0.0, err_formula = 0.0;
        for (int i = 0; i < kRot / 2; ++i) {
            const double a = x0[2 * i], b = x0[2 * i + 1];
            const double n0 = std::hypot(a, b), n1 = std::hypot((double) x[2 * i], (double) x[2 * i + 1]);
            err_norm = std::max(err_norm, std::fabs(n1 - n0) / std::max(n0, 1e-30));
            // adjacent pair (2i, 2i+1), angle = f32(pos) * inv_freq in f32, cos/sin evaluated in double here
            const double ang = (double) ((float) p * inv[i]);
            const double y0 = a * std::cos(ang) - b * std::sin(ang), y1 = a * std::sin(ang) + b * std::cos(ang);
            err_formula = std::max({err_formula, std::fabs(x[2 * i] - y0), std::fabs(x[2 * i + 1] - y1)});
        }
        check(err_norm < 1e-6, "rope pair rotation preserves the norm", err_norm, 1e-6);
        check(err_formula < 1e-6, "rope matches the NORM pair formula", err_formula, 1e-6);
    }

    // NOT NEOX: rotating a vector whose only nonzero entry is dim 0 must touch dim 1 and nothing else
    std::vector<float> e(kRot, 0.0f);
    e[0] = 1.0f;
    glmd::rope_row(e.data(), kRot, 3, inv);
    check(e[1] != 0.0f && e[kRot / 2] == 0.0f, "rope pairs (0, 1), not (0, n_rot/2)");
    check(e[1] == std::sin(3.0f), "rope sign: dim 1 of rotated e0 is +sin");

    // relative position: <R_m q, R_n k> depends only on m - n
    const std::vector<float> q0 = randv(rng, kRot), k0 = randv(rng, kRot);
    auto dot_at = [&](int m, int n) {
        std::vector<float> q = q0, k = k0;
        glmd::rope_row(q.data(), kRot, m, inv);
        glmd::rope_row(k.data(), kRot, n, inv);
        double d = 0.0;
        for (int i = 0; i < kRot; ++i) d += (double) q[i] * k[i];
        return d;
    };
    double qn = 0.0, kn = 0.0;
    for (int i = 0; i < kRot; ++i) {
        qn += (double) q0[i] * q0[i];
        kn += (double) k0[i] * k0[i];
    }
    const double scale = std::sqrt(qn) * std::sqrt(kn);   // the dot's natural size
    // the angles are f32 (HF semantics), so the identity holds to the angle rounding: |err| <~ pos * 2^-24 per pair
    const int pairs[][2] = {{5, 2}, {100, 37}, {200, 0}};
    for (auto& mn : pairs) {
        const double ref = dot_at(mn[0] - mn[1], 0);
        for (int shift : {1, 17, 64, 2048}) {
            const double lim = shift < 2048 ? 1e-4 : 2e-3;
            const double err = std::fabs(dot_at(mn[0] + shift, mn[1] + shift) - ref) / scale;
            check(err < lim, "rope q.k depends only on the position difference", err, lim);
        }
    }
}

// ---- indexer -------------------------------------------------------------------------------------------------

void test_indexer() {
    const int n_embd = 96, q_lora = 64, H = 4, D = 128, T = 3, n_kv = 9;
    const float eps = 1e-6f;
    std::mt19937 rng(21);
    float inv[kRot / 2];
    glmd::rope_inv_freq(inv, kRot, kTheta);

    const auto x = randv(rng, (size_t) n_embd * n_kv), wk = randv(rng, (size_t) D * n_embd, 0.1f);
    const auto lnw = randv(rng, D), lnb = randv(rng, D, 0.1f);
    const auto ql = randv(rng, (size_t) q_lora * T), wqb = randv(rng, (size_t) H * D * q_lora, 0.1f);
    const auto wp = randv(rng, (size_t) H * n_embd, 0.1f);
    std::vector<int> kpos(n_kv);
    for (int s = 0; s < n_kv; ++s) kpos[s] = s;
    const int qpos[T] = {0, 4, 8};

    // keys for every cached position, from the library
    std::vector<float> ik((size_t) D * n_kv);
    glmd::cpu::indexer_key(x.data(), wk.data(), lnw.data(), lnb.data(), eps, n_embd, D, n_kv, kRot, kpos.data(), inv,
                           ik.data());
    // naive: matmul, LayerNorm, rope in double with the shared f32 cos/sin
    std::vector<double> ik_ref((size_t) D * n_kv);
    for (int s = 0; s < n_kv; ++s) {
        std::vector<double> k(D);
        for (int o = 0; o < D; ++o) {
            double a = 0.0;
            for (int i = 0; i < n_embd; ++i) a += (double) wk[(size_t) o * n_embd + i] * x[(size_t) s * n_embd + i];
            k[o] = a;
        }
        double mu = 0.0, var = 0.0;
        for (double v : k) mu += v;
        mu /= D;
        for (double v : k) var += (v - mu) * (v - mu);
        var /= D;
        for (int o = 0; o < D; ++o) k[o] = (k[o] - mu) / std::sqrt(var + eps) * lnw[o] + lnb[o];
        for (int i = 0; i < kRot / 2; ++i) {
            float c, sn;
            glmd::rope_cs(s, inv[i], c, sn);
            const double a = k[2 * i], b = k[2 * i + 1];
            k[2 * i] = a * c - b * sn;
            k[2 * i + 1] = a * sn + b * c;
        }
        for (int o = 0; o < D; ++o) ik_ref[(size_t) s * D + o] = k[o];
    }
    double e = max_rel(ik, ik_ref);
    check(e < 1e-6, "indexer key = rope(LayerNorm(W_k x)) on dims 0..63", e, 1e-6);
    // dims 64..127 are LayerNorm output only: unrotated, so position must not change them
    std::vector<float> ik_shift((size_t) D * n_kv);
    std::vector<int> kpos2(n_kv, 123456);
    glmd::cpu::indexer_key(x.data(), wk.data(), lnw.data(), lnb.data(), eps, n_embd, D, n_kv, kRot, kpos2.data(), inv,
                           ik_shift.data());
    bool tail_same = true;
    for (int s = 0; s < n_kv; ++s)
        for (int o = kRot; o < D; ++o) tail_same &= ik[(size_t) s * D + o] == ik_shift[(size_t) s * D + o];
    check(tail_same, "indexer key dims 64..127 are not rotated ([rope | nope] layout)");

    std::vector<float> iq((size_t) D * H * T), w((size_t) H * T), score((size_t) n_kv * T);
    glmd::cpu::indexer_query(ql.data(), wqb.data(), q_lora, H, D, T, kRot, qpos, inv, iq.data());
    std::vector<float> xq((size_t) n_embd * T);
    for (int t = 0; t < T; ++t)
        std::copy_n(x.begin() + (size_t) qpos[t] * n_embd, n_embd, xq.begin() + (size_t) t * n_embd);
    glmd::cpu::indexer_weights(xq.data(), wp.data(), n_embd, H, D, T, w.data());
    glmd::cpu::indexer_score(iq.data(), ik.data(), w.data(), H, D, T, n_kv, qpos, score.data());

    std::vector<double> iq_ref((size_t) D * H * T), w_ref((size_t) H * T), sc_ref((size_t) n_kv * T);
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < H; ++h) {
            std::vector<double> q(D);
            for (int o = 0; o < D; ++o) {
                double a = 0.0;
                for (int i = 0; i < q_lora; ++i)
                    a += (double) wqb[((size_t) h * D + o) * q_lora + i] * ql[(size_t) t * q_lora + i];
                q[o] = a;
            }
            for (int i = 0; i < kRot / 2; ++i) {
                float c, sn;
                glmd::rope_cs(qpos[t], inv[i], c, sn);
                const double a = q[2 * i], b = q[2 * i + 1];
                q[2 * i] = a * c - b * sn;
                q[2 * i + 1] = a * sn + b * c;
            }
            for (int o = 0; o < D; ++o) iq_ref[((size_t) t * H + h) * D + o] = q[o];
            double a = 0.0;
            for (int i = 0; i < n_embd; ++i) a += (double) wp[(size_t) h * n_embd + i] * xq[(size_t) t * n_embd + i];
            w_ref[(size_t) t * H + h] = a / std::sqrt((double) (H * D));
        }
        for (int s = 0; s < n_kv; ++s) {
            if (s > qpos[t]) {
                sc_ref[(size_t) t * n_kv + s] = -std::numeric_limits<double>::infinity();
                continue;
            }
            double acc = 0.0;
            for (int h = 0; h < H; ++h) {
                double d = 0.0;
                for (int o = 0; o < D; ++o) d += (double) iq[((size_t) t * H + h) * D + o] * ik[(size_t) s * D + o];
                acc += (double) w[(size_t) t * H + h] * (d > 0.0 ? d : 0.0);
            }
            sc_ref[(size_t) t * n_kv + s] = acc;
        }
    }
    e = max_rel(iq, iq_ref);
    check(e < 1e-6, "indexer query = rope(W_qb q_lat) per head", e, 1e-6);
    e = max_rel(w, w_ref);
    check(e < 1e-6, "indexer head weights scaled by 1/sqrt(heads * dim)", e, 1e-6);
    double se = 0.0, sden = 1e-30;
    bool mask_ok = true;
    int positive = 0;
    for (size_t i = 0; i < score.size(); ++i) {
        if (std::isinf(sc_ref[i])) {
            mask_ok &= score[i] == -std::numeric_limits<float>::infinity();
            continue;
        }
        se = std::max(se, std::fabs(score[i] - sc_ref[i]));
        sden = std::max(sden, std::fabs(sc_ref[i]));
        positive += score[i] != 0.0f;
    }
    check(mask_ok, "indexer score is -inf exactly for s > pos[t]");
    check(se / sden < 1e-6, "indexer score = sum_h w_h relu(q_h . k_s)", se / sden, 1e-6);
    check(positive > 0, "indexer score is not all zero (the ReLU is not killing everything)");
}

// ---- top-k ---------------------------------------------------------------------------------------------------

void test_topk() {
    const float ninf = -std::numeric_limits<float>::infinity();
    const std::vector<float> sc = {1.0f, 3.0f, 3.0f, 2.0f, ninf, 3.0f};
    std::vector<int> sel(4);
    glmd::cpu::topk(sc.data(), 6, 1, 4, sel.data());
    check(sel == std::vector<int>({1, 2, 5, 3}), "top-k: score descending, ties by the lower cell");
    sel.assign(8, 0);
    glmd::cpu::topk(sc.data(), 6, 1, 8, sel.data());
    check(sel == std::vector<int>({1, 2, 5, 3, 0, -1, -1, -1}), "top-k: -inf excluded, -1 padding");

    // n <= k: every key, all 2048 at k = 2048; constant scores come back in cell order
    const int n = 2048;
    std::vector<float> flat(n, 0.5f);
    std::vector<int> all(n);
    glmd::cpu::topk(flat.data(), n, 1, 2048, all.data());
    bool in_order = true;
    for (int i = 0; i < n; ++i) in_order &= all[i] == i;
    check(in_order, "top-k: n <= k returns every key; full tie is cell order");

    // random with many ties, vs the rank definition
    std::mt19937 rng(31);
    std::uniform_int_distribution<int> d(0, 40);
    const int m = 3000, k = 2048;
    std::vector<float> r(m);
    for (auto& v : r) v = (float) d(rng) * 0.25f;
    std::vector<int> got(k), want(k, -1);
    glmd::cpu::topk(r.data(), m, 1, k, got.data());
    for (int s = 0; s < m; ++s) {
        int rank = 0;
        for (int j = 0; j < m; ++j) rank += r[j] > r[s] || (r[j] == r[s] && j < s);
        if (rank < k) want[rank] = s;
    }
    check(got == want, "top-k: matches the rank definition with ties (n 3000 > k 2048)");
}

// ---- MLA -----------------------------------------------------------------------------------------------------

void test_mla() {
    check(glmd::kGlm53KqScale == 1.0f / std::sqrt(192.0f + 64.0f), "GLM-5.3 kq_scale = 1/sqrt(key_length_mla 256)");

    const int H = 3, nope = 24, rope = 16, L = 32, V = 20, T = 2, n_cache = 11, n_sel = 7;
    const int qk_head = nope + rope, q_dim = L + rope;
    const float scale = 1.0f / std::sqrt((float) qk_head);
    std::mt19937 rng(41);
    const auto q = randv(rng, (size_t) qk_head * H * T), kv = randv(rng, (size_t) q_dim * n_cache);
    const auto wkb = randv(rng, (size_t) nope * L * H, 0.3f), wvb = randv(rng, (size_t) L * V * H, 0.3f);
    const std::vector<int> cells = {3, 0, 10, -1, 7, 5, 1,     // token 0: unordered, one padding slot
                                    -1, 9, -1, 2, 4, -1, 8};   // token 1: padding in the middle
    std::vector<float> Q((size_t) q_dim * H * T), out((size_t) V * H * T);
    glmd::cpu::mla_absorb_q(q.data(), wkb.data(), H, nope, rope, L, T, Q.data());
    glmd::cpu::mla_attention(Q.data(), kv.data(), cells.data(), wvb.data(), H, L, rope, V, T, n_sel, scale, out.data());

    // standard attention on decompressed per-head K = [k_nope | k_pe] and V = W_vb c_kv
    std::vector<double> ref((size_t) V * H * T);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h) {
            const float* qh = q.data() + ((size_t) t * H + h) * qk_head;
            std::vector<double> lg, pv;
            std::vector<std::vector<double>> vs;
            for (int s = 0; s < n_sel; ++s) {
                const int cell = cells[(size_t) t * n_sel + s];
                if (cell < 0) continue;
                const float* row = kv.data() + (size_t) cell * q_dim;
                double d = 0.0;
                for (int c = 0; c < nope; ++c) {
                    double kn = 0.0;   // k_nope[c] = sum_r W_kb[h][r][c] c_kv[r]
                    for (int r = 0; r < L; ++r) kn += (double) wkb[((size_t) h * L + r) * nope + c] * row[r];
                    d += (double) qh[c] * kn;
                }
                for (int i = 0; i < rope; ++i) d += (double) qh[nope + i] * row[L + i];
                lg.push_back(d * scale);
                std::vector<double> v(V);
                for (int j = 0; j < V; ++j) {
                    double a = 0.0;
                    for (int c = 0; c < L; ++c) a += (double) wvb[((size_t) h * V + j) * L + c] * row[c];
                    v[j] = a;
                }
                vs.push_back(v);
            }
            const double mx = *std::max_element(lg.begin(), lg.end());
            double sum = 0.0;
            for (double l : lg) sum += std::exp(l - mx);
            for (int j = 0; j < V; ++j) {
                double a = 0.0;
                for (size_t s = 0; s < lg.size(); ++s) a += std::exp(lg[s] - mx) / sum * vs[s][j];
                ref[((size_t) t * H + h) * V + j] = a;
            }
        }
    const double e = max_rel(out, ref);
    check(e < 1e-5, "absorbed MLA == attention on K/V decompressed with W_kb / W_vb", e, 1e-5);

    // a row of padding only writes zeros
    const std::vector<int> pad(n_sel * T, -1);
    glmd::cpu::mla_attention(Q.data(), kv.data(), pad.data(), wvb.data(), H, L, rope, V, T, n_sel, scale, out.data());
    check(std::all_of(out.begin(), out.end(), [](float v) { return v == 0.0f; }), "MLA with no valid cell is zero");
}

}  // namespace

int main() {
    test_rope();
    test_indexer();
    test_topk();
    test_mla();
    if (g_fail) {
        std::fprintf(stderr, "glmd_ref_cpu_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("glmd_ref_cpu_test OK\n");
    return 0;
}
