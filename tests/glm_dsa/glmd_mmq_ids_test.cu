// tests/glm_dsa/glmd_mmq_ids_test.cu - B9.8 / B14.2 on a V100: MMQ with expert bounds and ids (one launch over a
// group of experts, the prompt path's call) equals one launch per expert, and both match a host reference that does
// MMQ's arithmetic (rel L2 < 1e-4), per type the prompt path uses, at 80 SMs' stream-k shapes (rows per expert 1..300,
// 8 experts, n_embd 6144 -> 2 x 2048 and 2048 -> 6144).  With whole tiles (mmq::set_whole_tiles, the GLM prompt
// path's setting) the grouped launch equals the per-expert ones BIT FOR BIT on CUDA: the same prompt read with other
// experts resident gives the same numbers.
// The reference rounds the activations to q8_1 in the layout MMQ picks for the type (for Q2_K, Q4_K, Q5_K and IQ1_S
// fp16 scales and fp16 sums of the unquantized values, which their vec dots take for the weights' constant term) and
// takes the weights as MMQ's tiles hold them (fp16 products for Q2_K and IQ1_S). The first V100 run's reference
// (32-value float scales and the rounded values' sums for every type) put those four at 4e-3 to 8e-3, the others at
// about 1e-5.
// Sources cited: the llama.cpp pin's (3cf03257) ggml/src: ggml-common.h, ggml-quants.c and ggml-cuda/ (mmq.cuh,
// mmq-load-tiles.cuh, mmq-vec-dot.cuh, vecdotq.cuh, quantize.cu, common.cuh). On sm_70 MMQ takes its dp4a path (the
// mma layout needs TURING_MMA_AVAILABLE: mmq.cuh:196-202, common.cuh:288-290) with FAST_FP16_AVAILABLE
// (common.cuh:267-273). An AMD build models the form its device takes (g_mma): RDNA3 / RDNA4 (gfx1151 too) and CDNA
// take the MMA path, whose Q4_K / Q5_K tiles hold fp16 group scale products and whose vec dot forms IQ1_S's group
// products in float (the dp4a model put those three at ~5e-4 on the 8065S).
// The host side is most of the run (per type ~3e8 weights to quantize and ~4.5e10 reference multiply-adds), so it runs
// on every hardware thread: on one it went past ctest's 900 s on the test PC's 2.0 GHz Xeons. The weights, activations and
// reference are the same at any thread count.
// Needs a GPU; exits 77 (ctest SKIP_RETURN_CODE) when there is none.
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace mmq = strata::prefill::mmq;

namespace {

int g_fail = 0;
// MMQ's two arithmetic forms (mmq-load-tiles.cuh, mmq-vec-dot.cuh): the dp4a path (V100; AMD RDNA1 / RDNA2) the model
// below was written for, and the MMA path (TURING_MMA_AVAILABLE: sm_75 and newer; AMD_WMMA_AVAILABLE: RDNA3 / RDNA4,
// gfx1151 included; AMD_MFMA_AVAILABLE: CDNA). The MMA tiles of Q4_K / Q5_K hold fp16(d sc) and fp16(dmin m) per 32
// values (dm * make_half2(sc, m)) where the dp4a ones keep the 6-bit scales apart, and its q8_1 vec dot forms every
// group product in float (dmA.x dsB.x sumi + dmA.y dsB.y) where the dp4a one takes IQ1_S's in fp16 (__hmul2).
bool g_mma = false;

// get_scale_min_k4 (ggml-quants.c): group j's 6-bit scale and min of a Q4_K / Q5_K super-block's 12 scale bytes
void scale_min_k4(int j, const uint8_t* q, int& sc, int& m) {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// f(i) for every i in [0, n) on all hardware threads, each thread taking the next i
template <class F>
void parallel_for(int n, const F& f) {
    const int nt = std::min(n, (int) std::max(1u, std::thread::hardware_concurrency()));
    std::atomic<int> next{0};
    std::vector<std::thread> th;
    for (int k = 0; k < nt; ++k)
        th.emplace_back([&] {
            for (int i = next++; i < n; i = next++) f(i);
        });
    for (auto& t : th) t.join();
}

double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

// v rounded to fp16 and back, to nearest even: CUDA's float -> half conversions and half products round so
float f16r(float v) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(v)); }

float fp16_at(const uint8_t* p) {
    ggml_fp16_t h;
    std::memcpy(&h, p, sizeof h);
    return ggml_fp16_to_fp32(h);
}

// The q8_1 activation layout MMQ quantizes to for a weight type (mmq_get_q8_1_ds_layout, mmq.cuh:60-101;
// block_q8_1_mmq, mmq.cuh:27-46): d4 a float scale per 32 values; ds4 an fp16 scale and the fp16 sum of the unquantized
// values per 32; d2s6 an fp16 scale per 64 and fp16 sums of the unquantized values per 16 for the first 96 of each 128.
enum class Layout { d4, ds4, d2s6 };

Layout layout_of(ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_IQ1_S:
            return Layout::ds4;
        case GGML_TYPE_Q2_K:
            return Layout::d2s6;
        default:
            return Layout::d4;
    }
}

// How the reference forms an output from groups b of gs values, with xr the q8_1 activations and w the weights MMQ
// multiplies them by. plain: sum_k w_k xr_k (the d4 types, one group). offset: sum_b [sum_{k in b} w_k xr_k + c_b
// (S_b - sum_{k in b} xr_k)], c_b the weights' constant term and S_b the activation sum MMQ's vec dot takes for it
// (Q2_K, Q4_K, Q5_K). iq1: sum_b [fp16(dl_b d8_b) sum_{k in b} w_k u_k + fp16(ml_b S_b)] (IQ1_S's fp16 vec dot).
// none: a layout with sums for a type this test has no model of.
enum class Model { plain, offset, iq1, none };

Model model_of(ggml_type t) {
    if (layout_of(t) == Layout::d4) return Model::plain;
    if (t == GGML_TYPE_Q2_K || t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K) return Model::offset;
    return t == GGML_TYPE_IQ1_S ? Model::iq1 : Model::none;
}

// The sum of n = 32 (ds4) or 16 (d2s6) unquantized values as quantize_mmq_q8_1 forms it (quantize.cu:504-513): each
// thread adds its 4 values left to right, then __shfl_xor_sync over the n/4 threads at offsets n/8 .. 1. Lane 0, which
// stores the sum (quantize.cu:540-553), adds lane `off`'s partial at each step (float addition commutes, so the lanes
// it reads hold what lanes 0 .. off-1 hold here).
float mmq_sum(const float* v, int n) {
    float p[8];
    for (int t = 0; t < n / 4; ++t) p[t] = v[4 * t] + v[4 * t + 1] + v[4 * t + 2] + v[4 * t + 3];
    for (int off = n / 8; off > 0; off >>= 1)
        for (int t = 0; t < off; ++t) p[t] += p[t + off];
    return p[0];
}

// One weight row (K values: ggml's bytes q and to_float's values w) as MMQ's tiles hold it: w becomes what MMQ
// multiplies the activations by and c, m its per-group terms (Model). Returns how many values disagree with to_float
// (0 unless an offset or formula below is off; K for a block size other than the one cited).
long long mmq_weights(ggml_type t, const uint8_t* q, int K, float* w, float* c, float* m) {
    const size_t bs = ggml_type_size(t);
    const int nsb = K / 256;   // (QK_K super-blocks)
    long long bad = 0;
    if (t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K) {
        // block_q4_K (144 B) / block_q5_K (176 B), ggml-common.h:327-355: fp16 d at byte 0, dmin at 2. to_float gives
        // d1 q - m1 per 32 values, d1 = d sc and m1 = dmin m (dequantize_row_q4_K / _q5_K, ggml-quants.c:1529-1551,
        // 1731-1756), so with every d zeroed it gives c = -m1. MMQ's dp4a tiles keep dm and the 6-bit sc, m apart
        // (mmq-load-tiles.cuh:789-818, 912-942) and its vec dot forms d sum_b sc_b d8_b sumi_b - dmin sum_b m_b S_b in
        // float (vecdotq.cuh:532-558, 595-621; DS4's fp16 d8, S): w stays to_float's.
        if (bs != (t == GGML_TYPE_Q4_K ? 144u : 176u)) return K;
        if (g_mma) {
            // the MMA tiles: per 32 values fp16(d1) and fp16(m1), d1 = d sc, m1 = dmin m (fp16 d at byte 0, dmin at 2,
            // the 12 scale bytes at 4 in both blocks); to_float's value d1 v - m1 gives v, so w = fp16(d1) v - fp16(m1)
            // and c = -fp16(m1)
            for (int i = 0; i < nsb; ++i) {
                const uint8_t* bq = q + i * bs;
                const float d = fp16_at(bq), dmin = fp16_at(bq + 2);
                for (int j = 0; j < 8; ++j) {
                    int sc = 0, mn = 0;
                    scale_min_k4(j, bq + 4, sc, mn);
                    const float d1 = d * (float) sc, m1 = dmin * (float) mn;
                    const int b = i * 8 + j;
                    c[b] = -f16r(m1);
                    for (int k = b * 32; k < b * 32 + 32; ++k) {
                        const float v = d1 != 0.0f ? std::round((w[k] + m1) / d1) : 0.0f;
                        bad += d1 != 0.0f && w[k] != d1 * v - m1;
                        w[k] = f16r(d1) * v - f16r(m1);
                    }
                }
            }
            return bad;
        }
        std::vector<uint8_t> z(q, q + nsb * bs);
        std::vector<float> zf(K);
        for (int i = 0; i < nsb; ++i) z[i * bs] = z[i * bs + 1] = 0;
        ggml_get_type_traits(t)->to_float(z.data(), zf.data(), K);
        for (int b = 0; b < K / 32; ++b) {
            c[b] = zf[b * 32];
            for (int k = b * 32; k < b * 32 + 32; ++k) bad += zf[k] != c[b];
        }
    } else if (t == GGML_TYPE_Q2_K) {
        // block_q2_K (84 B), ggml-common.h:298-308: scales[16] at byte 0, qs[64] at 16, fp16 d at 80, dmin at 82. Value
        // k of a super-block is dl v - ml, v = (qs[32 (k / 128) + k % 32] >> 2 ((k % 128) / 32)) & 3, sc = scales[k /
        // 16], dl = d (sc & 15), ml = dmin (sc >> 4) (dequantize_row_q2_K, ggml-quants.c:961-991). MMQ's tile holds
        // fp16(dl) and fp16(ml) (__hmul2 of dm and {sc & 15, sc >> 4}, mmq-load-tiles.cuh:582-588) and its vec dot
        // forms d8 fp16(dl) sumi - fp16(ml) S per 16 (vecdotq.cuh:394-444): w = fp16(dl) v - fp16(ml), c = -fp16(ml).
        if (bs != 84) return K;
        for (int i = 0; i < nsb; ++i) {
            const uint8_t* bq = q + i * bs;
            const float d = fp16_at(bq + 80), dmin = fp16_at(bq + 82);
            for (int k = 0; k < 256; ++k) {
                const int sc = bq[k / 16], v = (bq[16 + 32 * (k / 128) + k % 32] >> (2 * (k % 128 / 32))) & 3;
                const float dl = d * (sc & 15), ml = dmin * (sc >> 4);
                float& wk = w[i * 256 + k];
                bad += wk != dl * v - ml;
                wk = f16r(dl) * v - f16r(ml);
                if (k % 16 == 0) c[(i * 256 + k) / 16] = -f16r(ml);
            }
        }
    } else if (t == GGML_TYPE_IQ1_S) {
        // block_iq1_s (50 B), ggml-common.h:425-429: fp16 d at byte 0, qs[32] at 2, uint16 qh[8] at 34. Group ib of 32
        // values is dl (g + delta), g in {-1, 0, 1}, dl = d (2 ((qh >> 12) & 7) + 1), delta = qh bit 15 ? -IQ1S_DELTA :
        // IQ1S_DELTA (0.125, ggml-common.h:1132; dequantize_row_iq1_s, ggml-quants.c:2650-2673). MMQ's tile holds the
        // codes g + 1 (iq1s_grid_gpu) and fp16 (dl, dl (delta - 1)) (mmq-load-tiles.cuh:1069-1093): w = g + 1 (read off
        // to_float's value), c = fp16(dl), m = fp16(dl (delta - 1)).
        if (bs != 50) return K;
        for (int i = 0; i < nsb; ++i) {
            const uint8_t* bq = q + i * bs;
            const float d = fp16_at(bq);
            for (int ib = 0; ib < 8; ++ib) {
                uint16_t qh;
                std::memcpy(&qh, bq + 34 + 2 * ib, sizeof qh);
                const float dl = d * (2 * ((qh >> 12) & 7) + 1), delta = qh & 0x8000 ? -0.125f : 0.125f;
                const int b = i * 8 + ib;
                c[b] = f16r(dl);
                m[b] = f16r(dl * (delta - 1.0f));
                for (int k = b * 32; k < b * 32 + 32; ++k) {
                    const float g = dl != 0.0f ? std::round(w[k] / dl - delta) : 0.0f;
                    bad += std::fabs(g) > 1.0f || w[k] != dl * (g + delta);
                    w[k] = g + 1.0f;
                }
            }
        }
    } else {
        return K;
    }
    return bad;
}

void run_type(int type, int N, int K) {
    const ggml_type t = (ggml_type) type;
    const Layout lay = layout_of(t);
    const Model model = model_of(t);
    if (model == Model::none) {
        std::printf("type %2d [%d x %d]: no host model of its MMQ activation layout  FAIL\n", type, N, K);
        ++g_fail;
        return;
    }
    // the reference's groups (Model): ds4's 32 values, d2s6's 16 (Q2_K's scales), one for the d4 types
    const int gs = model == Model::plain ? K : lay == Layout::d2s6 ? 16 : 32, nb = K / gs;
    const int n_exp = 8;
    std::mt19937 rng(type * 31 + N);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> rows_d(1, 300);
    // weights: n_exp experts of [N][K], quantized by ggml (a unit importance matrix where one is required). The draws
    // stay on the one generator, in order; ggml quantizes row by row, so 16-row chunks on threads give the same bytes.
    // wdq: to_float's values, made MMQ's by mmq_weights for the types with group terms (wc, wm per group)
    const size_t ebytes = mmq::matrix_bytes(type, N, K), rbytes = ggml_row_size(t, K);
    std::vector<uint8_t> wq(ebytes * n_exp);
    std::vector<float> wf((size_t) N * K), imat((size_t) K, 1.0f), wdq((size_t) n_exp * N * K);
    std::vector<float> wc(model == Model::plain ? 0 : (size_t) n_exp * N * nb);
    std::vector<float> wm(model == Model::iq1 ? (size_t) n_exp * N * nb : 0);
    std::atomic<long long> off{0};   // values whose mmq_weights decode disagrees with to_float
    ggml_quantize_init(t);   // (before the threads: it builds the i-quant grids)
    const float* qw = ggml_quantize_requires_imatrix(t) ? imat.data() : nullptr;
    const auto* tr = ggml_get_type_traits(t);
    for (int e = 0; e < n_exp; ++e) {
        for (auto& v : wf) v = nd(rng) * 0.05f;
        uint8_t* we = wq.data() + e * ebytes;
        parallel_for((N + 15) / 16, [&](int c) {
            const int r0 = c * 16, nr = std::min(16, N - r0);
            ggml_quantize_chunk(t, wf.data(), we, (int64_t) r0 * K, nr, K, qw);
            for (int r = r0; r < r0 + nr; ++r) {
                const size_t wr = (size_t) e * N + r;
                tr->to_float(we + (size_t) r * rbytes, wdq.data() + wr * K, K);
                if (model != Model::plain)
                    off += mmq_weights(t, we + (size_t) r * rbytes, K, wdq.data() + wr * K, wc.data() + wr * nb,
                                       model == Model::iq1 ? wm.data() + wr * nb : nullptr);
            }
        });
    }
    // activations: rows grouped by expert
    std::vector<int> bounds(n_exp + 1, 0);
    for (int e = 0; e < n_exp; ++e) bounds[e + 1] = bounds[e] + rows_d(rng);
    const int rows = bounds[n_exp];
    int max_rows = 0;
    for (int e = 0; e < n_exp; ++e) max_rows = std::max(max_rows, bounds[e + 1] - bounds[e]);
    std::vector<float> x((size_t) rows * K);
    for (auto& v : x) v = nd(rng);
    x[5] = 3000.0f;   // an outlier (B14.1: the q8_1 sums go large)
    // host q8_1 activations as quantize_mmq_q8_1 (quantize.cu:457-556) writes them in this type's layout: per sb
    // values (64 for d2s6, else 32) d_inv = 127 / amax, u = round(x d_inv), scale d = 1 / d_inv, float in d4 and fp16
    // otherwise (the GPU's divisions are -use_fast_math ones, a few ulp off: the rare u or fp16 d that flips on that
    // stays far under the gate). xr = u d (IQ1_S: u, its d goes into the group product). Per group, S = the fp16 sum of
    // the unquantized x (mmq_sum), except for d2s6's last 2 groups of 16 in each 128, which have none stored: Q2_K's
    // vec dot takes the quantized values' own sum d sum u there (vec_dot_q2_K_q8_1_impl_mmq<ns8>, vecdotq.cuh:394-444,
    // called with ns8 2 for values 0..63 and 1 for 64..127 of each 128: mmq-vec-dot.cuh:630-672). Then xa = S - sum xr
    // (offset) or xa = d, xb = S (iq1).
    const int sb = lay == Layout::d2s6 ? 64 : 32;
    std::vector<float> xr(x.size());
    std::vector<double> xa(model == Model::plain ? 0 : (size_t) rows * nb);
    std::vector<double> xb(model == Model::iq1 ? (size_t) rows * nb : 0);
    parallel_for(rows, [&](int r) {
        const float* xrow = x.data() + (size_t) r * K;
        float* q = xr.data() + (size_t) r * K;
        for (int k0 = 0; k0 < K; k0 += sb) {
            float am = 0.0f;
            for (int k = k0; k < k0 + sb; ++k) am = std::max(am, std::fabs(xrow[k]));
            const float d_inv = 127.0f / am, d = 1.0f / d_inv, ds = lay == Layout::d4 ? d : f16r(d);
            for (int k = k0; k < k0 + sb; ++k) {
                const float u = am > 0.0f ? std::round(xrow[k] * d_inv) : 0.0f;
                q[k] = model == Model::iq1 ? u : u * ds;
            }
            if (model == Model::iq1) xa[(size_t) r * nb + k0 / gs] = ds;   // (one 32-value group per scale)
        }
        for (int b = 0; model != Model::plain && b < nb; ++b) {
            const int k0 = b * gs;
            const double S = f16r(mmq_sum(xrow + k0, gs));
            double sxr = 0.0;
            for (int k = k0; k < k0 + gs; ++k) sxr += q[k];
            if (model == Model::iq1) xb[(size_t) r * nb + b] = S;
            else xa[(size_t) r * nb + b] = lay == Layout::d2s6 && k0 % 128 >= 96 ? 0.0 : S - sxr;
        }
    });
    // every output is a double sum over the groups (one for the d4 types) of each group's double sum in k order, as
    // the type's Model combines them; a task is up to 4 rows of one expert, so each weight row is read once for 4 dot
    // products (4 independent sums)
    std::vector<float> ref((size_t) rows * N);
    std::vector<std::pair<int, int>> tasks;   // (expert, first row)
    for (int e = 0; e < n_exp; ++e)
        for (int r = bounds[e]; r < bounds[e + 1]; r += 4) tasks.emplace_back(e, r);
    parallel_for((int) tasks.size(), [&](int i) {
        const int e = tasks[i].first, r0 = tasks[i].second, nr = std::min(4, bounds[e + 1] - r0);
        int rj[4];   // (a task of fewer rows repeats its last one)
        const float* xs[4];
        for (int j = 0; j < 4; ++j) {
            rj[j] = r0 + std::min(j, nr - 1);
            xs[j] = xr.data() + (size_t) rj[j] * K;
        }
        for (int o = 0; o < N; ++o) {
            const size_t wo = (size_t) e * N + o;
            const float* w = wdq.data() + wo * K;
            double acc[4] = {0.0, 0.0, 0.0, 0.0};
            for (int b = 0; b < nb; ++b) {
                double s[4] = {0.0, 0.0, 0.0, 0.0};
                for (int k = b * gs; k < (b + 1) * gs; ++k)
                    for (int j = 0; j < 4; ++j) s[j] += (double) w[k] * xs[j][k];
                for (int j = 0; j < 4; ++j) {
                    const size_t g = (size_t) rj[j] * nb + b;
                    if (model == Model::plain) acc[j] += s[j];
                    else if (model == Model::offset) acc[j] += s[j] + (double) wc[wo * nb + b] * xa[g];
                    else if (g_mma)   // the MMA vec dot: dl d8 sumi + ml S in float
                        acc[j] += (double) wc[wo * nb + b] * xa[g] * s[j] + (double) wm[wo * nb + b] * xb[g];
                    else   // sumi fp16(dl d8) + fp16(ml S): vec_dot_q8_1_q8_1_impl's __hmul2 (vecdotq.cuh:260-284)
                        acc[j] += (double) f16r(wc[wo * nb + b] * (float) xa[g]) * s[j] +
                                  f16r(wm[wo * nb + b] * (float) xb[g]);
                }
            }
            for (int j = 0; j < nr; ++j) ref[(size_t) (r0 + j) * N + o] = (float) acc[j];
        }
    });
    // device
    void *d_w, *d_xq;
    float *d_x, *d_y1, *d_y2;
    int32_t *d_b, *d_ids;
    cudaMalloc(&d_w, wq.size());
    cudaMalloc(&d_x, x.size() * 4);
    cudaMalloc(&d_xq, mmq::q8_bytes(rows, K));
    cudaMalloc(&d_y1, ref.size() * 4);
    cudaMalloc(&d_y2, ref.size() * 4);
    cudaMalloc(&d_b, bounds.size() * 4);
    cudaMalloc(&d_ids, (size_t) rows * 4);
    cudaMemcpy(d_w, wq.data(), wq.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice);
    mmq::iota(d_ids, rows, nullptr);
    mmq::quantize(d_x, nullptr, d_xq, type, K, K, rows, nullptr);
    mmq::Context ctx;
    mmq::Product p;
    p.w = d_w;
    p.type = type;
    p.w_rows = N;
    p.w_cols = K;
    p.expert_bytes = ebytes;
    p.n = n_exp;
    p.xq = d_xq;
    p.bounds = d_b;
    p.ids = d_ids;
    p.total_rows = rows;
    p.max_rows = max_rows;
    p.dst = d_y1;
    p.ld_dst = N;
    ctx.run(p, nullptr);
    for (int e = 0; e < n_exp; ++e) {   // one launch per expert
        mmq::Product q = p;
        q.w = (const uint8_t*) d_w + e * ebytes;
        q.n = 1;
        q.bounds = d_b + e;
        q.max_rows = bounds[e + 1] - bounds[e];
        q.dst = d_y2;
        ctx.run(q, nullptr);
    }
    cudaDeviceSynchronize();
    std::vector<float> y1(ref.size()), y2(ref.size());
    cudaMemcpy(y1.data(), d_y1, y1.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(y2.data(), d_y2, y2.size() * 4, cudaMemcpyDeviceToHost);
    // whole tiles (mmq::set_whole_tiles, the GLM prompt path's setting): an expert's rows get the same bits whatever
    // else its launch holds - grouped == per-expert bit for bit.  Without it stream-k splits k by the launch's tile
    // count (the count of differing values is printed; NVIDIA's configs split, AMD's RDNA ones tile anyway).
    size_t split_diff = 0, whole_diff = 0;
    for (size_t i = 0; i < y1.size(); ++i) split_diff += std::memcmp(&y1[i], &y2[i], 4) != 0;
    const bool was = mmq::set_whole_tiles(true);
    ctx.run(p, nullptr);
    for (int e = 0; e < n_exp; ++e) {
        mmq::Product q = p;
        q.w = (const uint8_t*) d_w + e * ebytes;
        q.n = 1;
        q.bounds = d_b + e;
        q.max_rows = bounds[e + 1] - bounds[e];
        q.dst = d_y2;
        ctx.run(q, nullptr);
    }
    cudaDeviceSynchronize();
    mmq::set_whole_tiles(was);
    std::vector<float> y3(ref.size()), y4(ref.size());
    cudaMemcpy(y3.data(), d_y1, y3.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(y4.data(), d_y2, y4.size() * 4, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < y3.size(); ++i) whole_diff += std::memcmp(&y3[i], &y4[i], 4) != 0;
#if defined(STRATA_USE_HIP)
    const bool whole_ok = true;   // (HIP: no switch - reported only)
#else
    const bool whole_ok = whole_diff == 0;
#endif
    std::printf("type %2d [%d x %d]: values that differ grouped vs per-expert: stream-k %zu, whole tiles %zu  %s\n", type,
                N, K, split_diff, whole_diff, whole_ok ? "ok" : "FAIL");
    g_fail += !whole_ok;
    const double e_ids = rel_l2(y1, y2), e_ref = rel_l2(y1, ref);
    bool finite = true;
    for (float v : y1) finite = finite && std::isfinite(v);
    const long long n_off = off.load();
    char note[64] = "";
    if (n_off) std::snprintf(note, sizeof note, " (host decode: %lld values off to_float)", n_off);
    // e_ref: MMQ's float sums and the rare u or fp16 scale its fast-math divisions flip leave ~1e-5 (the d4 types in
    // the first V100 run); a missed fp16 rounding of a scale or product costs ~2e-4, a missed stored sum 4e-3 to 8e-3
    const bool ok = finite && !n_off && e_ids < 1e-5 && e_ref < 1e-4 && cudaGetLastError() == cudaSuccess;
    std::printf("type %2d [%d x %d] x %d rows: grouped vs per-expert %.2e, vs host q8_1 reference %.2e%s%s  %s\n", type,
                N, K, rows, e_ids, e_ref, finite ? "" : " (non-finite)", note, ok ? "ok" : "FAIL");
    g_fail += !ok;
    for (void* q : {d_w, (void*) d_x, d_xq, (void*) d_y1, (void*) d_y2, (void*) d_b, (void*) d_ids}) cudaFree(q);
}

}  // namespace

int main() {
#ifdef _WIN32
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // (MSVC's CRT refuses _IOLBF with size 0)
#else
    std::setvbuf(stdout, nullptr, _IOLBF, 0);   // each type's line reaches ctest's log, also when a run is cut off
#endif
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
#if defined(STRATA_USE_HIP)
    {   // which of MMQ's two forms this AMD device's instances take (g_mma); CUDA builds keep the V100 (dp4a) model
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, dev);
        const std::string a = prop.gcnArchName;
        g_mma = a.rfind("gfx11", 0) == 0 || a.rfind("gfx12", 0) == 0 || a.rfind("gfx908", 0) == 0 ||
                a.rfind("gfx90a", 0) == 0 || a.rfind("gfx94", 0) == 0 || a.rfind("gfx95", 0) == 0;
        std::printf("device %s: MMQ's %s form\n", a.c_str(), g_mma ? "MMA (WMMA / MFMA)" : "dp4a");
    }
#endif
    for (int t = 0; t < 64; ++t) {
        if (!mmq::supported(t) || t == GGML_TYPE_Q2_0) continue;   // (Q2_0: the glm-dsa pack format, no ggml quantizer)
        run_type(t, 2 * 2048, 6144);
        run_type(t, 6144, 2048);
    }
    std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
