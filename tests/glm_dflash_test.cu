// tests/glm_dflash_test.cu - the DFlash2 drafter (src/core/glm_dflash.cu) against a CPU float forward of the same GGUF.
//
//   glm_dflash_test <dflash2.gguf> [n_past]
//
// Synthetic trunk features (a fixed pseudo-random sequence) for n_past positions go through the drafter's staging and
// injection; one draft runs on the GPU with a pseudo-random BF16 output head and embedding table of the model's
// vocabulary.  The CPU reference dequantizes every weight with ggml and runs the same forward in float (double sums):
// the context K / V, the block's 5 layers with their grouped convolutions, the non-causal windowed attention, the
// final norm, the head's top-k, the selector's gate and greedy walk.  It checks the mask rows' final hidden states
// (cosine), the candidate sets and the picks, which the GPU's q8_1 activations may only perturb where two scores tie.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/glm_dflash.hpp"
#include "strata/core/glm_model.hpp"

#include "ggml.h"

#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

using strata::core::GlmDflash;
using strata::core::WSlot;

namespace {

uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
float unit(uint64_t k) { return (float) ((mix(k) >> 40) * (1.0 / 16777216.0)) * 2.0f - 1.0f; }   // [-1, 1)

uint16_t bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float from_bf16(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct Ref {
    const strata::GgufFile& G;
    std::map<std::string, std::vector<float>> w;
    explicit Ref(const strata::GgufFile& g) : G(g) {}
    const std::vector<float>& t(const std::string& n) {
        auto it = w.find(n);
        if (it != w.end()) return it->second;
        const strata::TensorInfo* ti = G.find(n);
        if (ti == nullptr) {
            std::fprintf(stderr, "missing %s\n", n.c_str());
            std::exit(2);
        }
        std::vector<float> v((size_t) ti->elements());
        if (ti->type == 0) {
            std::memcpy(v.data(), G.tensor_data(*ti), v.size() * 4);
        } else {
            const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) ti->type);
            tt->to_float(G.tensor_data(*ti), v.data(), (int64_t) v.size());
        }
        return w[n] = std::move(v);
    }
};

// the GPU's q8_1 activations (glm_fast quantize_kernel: per 32 values d = max|x| / 127, q = round(x / d)), applied in
// place: with them the reference sees what the GPU's quantized-weight GEMVs multiply (F32 / BF16 weights - a BF16
// drafter's - read the float rows, as the GPU's GEMVs do: main's mm)
bool g_q8 = true;
std::vector<float> q8(const float* x, size_t n) {
    std::vector<float> y(x, x + n);
    if (!g_q8) return y;
    for (size_t b = 0; b + 32 <= n; b += 32) {
        float am = 0;
        for (int i = 0; i < 32; ++i) am = std::max(am, std::fabs(y[b + i]));
        const float d = am / 127.0f;
        for (int i = 0; i < 32; ++i) y[b + i] = d > 0 ? std::round(y[b + i] / d) * d : 0.0f;
    }
    return y;
}

// y[r][o] = sum_i W[o][i] x[r][i]  (W: n_out rows of n_in, ggml order)
void matmul(const std::vector<float>& W, int n_in, int n_out, const float* x0, int rows, float* y, bool quant = true) {
    const std::vector<float> xq = quant ? q8(x0, (size_t) rows * n_in) : std::vector<float>(x0, x0 + (size_t) rows * n_in);
    const float* x = xq.data();
    const int nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> th;
    for (int k = 0; k < nt; ++k)
        th.emplace_back([&, k] {
            for (int o = k; o < n_out; o += nt) {
                const float* wr = W.data() + (size_t) o * n_in;
                for (int r = 0; r < rows; ++r) {
                    double a = 0;
                    const float* xr = x + (size_t) r * n_in;
                    for (int i = 0; i < n_in; ++i) a += (double) wr[i] * xr[i];
                    y[(size_t) r * n_out + o] = (float) a;
                }
            }
        });
    for (auto& t : th) t.join();
}

void rms(const float* x, const float* w, float eps, int n, float* y) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const float r = (float) (1.0 / std::sqrt(ss / n + eps));
    for (int i = 0; i < n; ++i) y[i] = x[i] * r * w[i];
}

void head_norm_rope(float* v, int heads, const float* w, float eps, int64_t pos, float base) {
    for (int h = 0; h < heads; ++h) {
        float* p = v + (size_t) h * 128;
        float t[128];
        rms(p, w, eps, 128, t);
        for (int j = 0; j < 64; ++j) {
            const float th = (float) pos * std::pow(base, -2.0f * (float) j / 128.0f);
            const float c = std::cos(th), s = std::sin(th);
            p[j] = t[j] * c - t[j + 64] * s;
            p[j + 64] = t[j] * s + t[j + 64] * c;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: glm_dflash_test <dflash2.gguf> [n_past]\n");
        return 2;
    }
    const std::string path = argv[1];
    const int64_t n_past = argc > 2 ? std::atoll(argv[2]) : 2100;
    g_q8 = getenv("DFT_FLOAT") == nullptr;   // DFT_FLOAT=1: the reference without the q8_1 rounding
    const int E = 4096, V = 154880, NL_T = 45;
    std::string err;
    GlmDflash D;
    if (!D.load(path, 0, err) || !D.bind(E, V, NL_T, err)) {
        std::fprintf(stderr, "load: %s\n", err.c_str());
        return 1;
    }
    std::printf("%s\n", D.describe().c_str());
    strata::GgufFile G(path);
    const auto num = [&](const char* k) { return G.get(std::string("dflash.") + k)->num(); };
    const int n_layer = (int) num("block_count"), n_head = (int) num("attention.head_count"),
              n_kv = (int) num("attention.head_count_kv"), n_ff = (int) num("feed_forward_length"),
              taps = (int) num("conv_kernel_size"), group = (int) num("conv_group_size"),
              rank = (int) num("selector_rank"), top_k = (int) num("selector_top_k"),
              window = (int) num("attention.sliding_window");
    const float eps = (float) num("attention.layer_norm_rms_epsilon"), base = (float) num("rope.freq_base");
    const int mask = (int) G.get("tokenizer.ggml.mask_token_id")->u;
    const auto& tl = D.target_layers();
    const int K = (int) tl.size(), qd = n_head * 128, kvd = n_kv * 128, ng = E / group;
    // ---- the synthetic trunk features, staged and injected
    const auto feat = [&](int64_t p, int k, int e) { return 2.0f * unit(((uint64_t) p * 64 + (uint64_t) k) * 8192 + e); };
    for (int64_t p = 0; p < n_past; ++p)
        for (int k = 0; k < K; ++k) {
            float* r = D.stage_row(p, k);
            for (int e = 0; e < E; ++e) r[e] = feat(p, k, e);
            D.staged(p, k);
        }
    cudaStream_t s;
    cudaSetDevice(0);
    cudaStreamCreate(&s);
    if (!D.inject(0, n_past, s, err)) {
        std::fprintf(stderr, "inject: %s\n", err.c_str());
        return 1;
    }
    // ---- a BF16 head and embedding table
    std::vector<uint16_t> head_h((size_t) V * E), emb_h((size_t) V * E);
    for (size_t i = 0; i < head_h.size(); ++i) head_h[i] = bf16(0.02f * unit(0x1000000000ull + i));
    for (size_t i = 0; i < emb_h.size(); ++i) emb_h[i] = bf16(0.05f * unit(0x2000000000ull + i));
    void* head_d = nullptr;
    cudaMalloc(&head_d, head_h.size() * 2);
    cudaMemcpy(head_d, head_h.data(), head_h.size() * 2, cudaMemcpyHostToDevice);
    WSlot head;
    head.type = 30;
    head.q = head_d;
    head.n_in = E;
    head.n_out = V;
    const int anchor = 1234, B = D.block(), n = B - 1;
    std::vector<int32_t> out;
    if (!D.draft(anchor, n_past, n, 0.0f, head, (const uint8_t*) emb_h.data(), 30, s, out, err)) {
        std::fprintf(stderr, "draft: %s\n", err.c_str());
        return 1;
    }
    std::vector<float> g_h, g_lg, g_conf;
    std::vector<int> g_cand, g_pick;
    D.last_rows(n, g_h, g_cand, g_lg, g_pick, g_conf);
    {
        // the draft's time at this context (the same block again: a draft leaves the ring as it was)
        std::vector<double> ms;
        std::vector<int32_t> again;
        for (int r = 0; r < 20; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            if (!D.draft(anchor, n_past, n, 0.0f, head, (const uint8_t*) emb_h.data(), 30, s, again, err)) {
                std::fprintf(stderr, "draft: %s\n", err.c_str());
                return 1;
            }
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(ms.begin(), ms.end());
        std::printf("draft: %.2f ms (median of 20), %s\n", ms[10], again == out ? "same drafts" : "DRAFTS DIFFER");
    }
    if (getenv("DFT_HEAD_Q8") != nullptr) {
        // the draft's time with a Q8_0 head (a GGUF model's usual head type), drafts of 1..n; timing only
        const size_t row_q = ggml_row_size(GGML_TYPE_Q8_0, E);
        std::vector<uint8_t> hq((size_t) V * row_q);
        std::vector<float> f((size_t) 1024 * E);
        for (int r0 = 0; r0 < V; r0 += 1024) {
            const int nr = std::min(1024, V - r0);
            for (size_t i = 0; i < (size_t) nr * E; ++i) f[i] = 0.02f * unit(0x1000000000ull + (size_t) r0 * E + i);
            ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data(), hq.data() + (size_t) r0 * row_q, 0, nr, E, nullptr);
        }
        void* hq_d = nullptr;
        cudaMalloc(&hq_d, hq.size());
        cudaMemcpy(hq_d, hq.data(), hq.size(), cudaMemcpyHostToDevice);
        WSlot hq8 = head;
        hq8.type = GGML_TYPE_Q8_0;
        hq8.q = hq_d;
        const int kq = std::atoi(getenv("DFT_HEAD_Q8"));   // 1..n: only that many drafts (under a profiler)
        const bool one = kq >= 1 && kq <= n;
        if (one) cudaProfilerStart();
        for (int k = one ? kq : 1; k <= (one ? kq : n); ++k) {
            std::vector<double> ms;
            std::vector<int32_t> d;
            for (int r = 0; r < 20; ++r) {
                const auto t0 = std::chrono::steady_clock::now();
                if (!D.draft(anchor, n_past, k, 0.0f, hq8, (const uint8_t*) emb_h.data(), 30, s, d, err)) {
                    std::fprintf(stderr, "draft: %s\n", err.c_str());
                    return 1;
                }
                ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            std::sort(ms.begin(), ms.end());
            std::printf("draft, Q8_0 head, %d drafts: %.2f ms (median of 20)\n", k, ms[10]);
        }
        if (one) cudaProfilerStop();
        cudaFree(hq_d);
    }
    // ---- the CPU reference
    Ref R(G);
    // (the GPU rounds a GEMV's activations to q8_1 only for quantized weights: F32 / BF16 ones read the floats)
    const auto mm = [&](const std::string& n, int n_in, int n_out, const float* x, int rows, float* y) {
        const int ty = (int) G.find(n)->type;
        matmul(R.t(n), n_in, n_out, x, rows, y, ty != 0 && ty != 1 && ty != 30);
    };
    const int64_t ctx_lo = std::max<int64_t>(0, n_past - window + 1);
    const int nc = (int) (n_past - ctx_lo);
    std::vector<std::vector<float>> Kc((size_t) n_layer, std::vector<float>((size_t) nc * kvd)),
        Vc((size_t) n_layer, std::vector<float>((size_t) nc * kvd));
    {
        std::vector<float> f((size_t) nc * K * E), g((size_t) nc * E), gn((size_t) nc * E);
        for (int r = 0; r < nc; ++r)
            for (int k = 0; k < K; ++k)
                for (int e = 0; e < E; ++e) f[((size_t) r * K + k) * E + e] = feat(ctx_lo + r, k, e);
        mm("fc.weight", K * E, E, f.data(), nc, g.data());
        for (int r = 0; r < nc; ++r) rms(&g[(size_t) r * E], R.t("enc.output_norm.weight").data(), eps, E, &gn[(size_t) r * E]);
        for (int il = 0; il < n_layer; ++il) {
            const std::string P = "blk." + std::to_string(il) + ".";
            mm(P + "attn_k.weight", E, kvd, gn.data(), nc, Kc[(size_t) il].data());
            mm(P + "attn_v.weight", E, kvd, gn.data(), nc, Vc[(size_t) il].data());
            for (int r = 0; r < nc; ++r)
                head_norm_rope(&Kc[(size_t) il][(size_t) r * kvd], n_kv, R.t(P + "attn_k_norm.weight").data(), eps,
                               ctx_lo + r, base);
            if (g_q8)   // the ring holds them in FP16
                for (auto* v : {&Kc[(size_t) il], &Vc[(size_t) il]})
                    for (float& f : *v) f = __half2float(__float2half(f));
        }
    }
    std::vector<float> h((size_t) B * E), x((size_t) B * E), xc((size_t) B * E), c((size_t) B * 2 * taps * ng),
        q((size_t) B * qd), kb((size_t) B * kvd), vb((size_t) B * kvd), a((size_t) B * qd), o((size_t) B * E),
        gt((size_t) B * n_ff), up((size_t) B * n_ff);
    for (int t = 0; t < B; ++t)
        for (int e = 0; e < E; ++e) h[(size_t) t * E + e] = from_bf16(emb_h[(size_t) (t == 0 ? anchor : mask) * E + e]);
    const auto conv = [&](const float* in, const float* cc, const float* bs, int side, float* outp) {
        for (int t = 0; t < B; ++t)
            for (int e = 0; e < E; ++e) {
                double acc = 0;
                for (int tap = 0; tap < taps && tap <= t; ++tap)
                    acc += (double) (cc[(size_t) t * 2 * taps * ng + (size_t) side * taps * ng + tap * ng + e / group] +
                                     bs[(size_t) side * taps * E + (size_t) tap * E + e]) *
                           in[(size_t) (t - tap) * E + e];
                outp[(size_t) t * E + e] = (float) acc;
            }
    };
    const char* dd = getenv("STRATA_GLM_DFLASH_DUMP");
    const auto cmp = [&](const char* name, const std::vector<float>& ref, size_t n) {
        if (dd == nullptr) return;
        std::vector<float> g(n);
        FILE* f = std::fopen((std::string(dd) + "/" + name + ".f32").c_str(), "rb");
        if (f == nullptr || std::fread(g.data(), 4, n, f) != n) {
            std::printf("  %-4s: no dump\n", name);
            if (f) std::fclose(f);
            return;
        }
        std::fclose(f);
        double d = 0, na = 0, nb = 0, mx = 0;
        for (size_t i = 0; i < n; ++i) {
            d += (double) ref[i] * g[i];
            na += (double) ref[i] * ref[i];
            nb += (double) g[i] * g[i];
            mx = std::max(mx, (double) std::fabs(ref[i] - g[i]));
        }
        std::printf("  %-4s: cos %.7f  max |diff| %.4g  rms ref %.4g gpu %.4g\n", name, d / std::sqrt(na * nb + 1e-30), mx,
                    std::sqrt(na / n), std::sqrt(nb / n));
    };
    for (int il = 0; il < n_layer; ++il) {
        const std::string P = "blk." + std::to_string(il) + ".";
        if (il == 0) cmp("h0", h, h.size());
        for (int t = 0; t < B; ++t) rms(&h[(size_t) t * E], R.t(P + "attn_norm.weight").data(), eps, E, &x[(size_t) t * E]);
        if (il == 0) cmp("x0", x, x.size());
        mm(P + "attn_conv_proj.weight", E, 2 * taps * ng, x.data(), B, c.data());
        conv(x.data(), c.data(), R.t(P + "attn_conv_base").data(), 0, xc.data());
        if (il == 0) cmp("c0", c, c.size()), cmp("xc0", xc, xc.size());
        mm(P + "attn_q.weight", E, qd, xc.data(), B, q.data());
        mm(P + "attn_k.weight", E, kvd, xc.data(), B, kb.data());
        mm(P + "attn_v.weight", E, kvd, xc.data(), B, vb.data());
        for (int t = 0; t < B; ++t) {
            head_norm_rope(&q[(size_t) t * qd], n_head, R.t(P + "attn_q_norm.weight").data(), eps, n_past + t, base);
            head_norm_rope(&kb[(size_t) t * kvd], n_kv, R.t(P + "attn_k_norm.weight").data(), eps, n_past + t, base);
        }
        if (il == 0) cmp("q0", q, q.size()), cmp("k0", kb, kb.size()), cmp("v0", vb, vb.size());
        for (int t = 0; t < B; ++t)
            for (int hh = 0; hh < n_head; ++hh) {
                const int kh = hh / (n_head / n_kv);
                const float* qv = &q[(size_t) t * qd + (size_t) hh * 128];
                const int64_t lo = std::max<int64_t>(0, n_past + t - window + 1);
                std::vector<double> sc;
                std::vector<const float*> vv;
                for (int64_t pp = lo; pp < n_past; ++pp) {
                    const float* kv = &Kc[(size_t) il][(size_t) (pp - ctx_lo) * kvd + (size_t) kh * 128];
                    double d = 0;
                    for (int e = 0; e < 128; ++e) d += (double) qv[e] * kv[e];
                    sc.push_back(d / std::sqrt(128.0));
                    vv.push_back(&Vc[(size_t) il][(size_t) (pp - ctx_lo) * kvd + (size_t) kh * 128]);
                }
                for (int j = 0; j < B; ++j) {
                    const float* kv = &kb[(size_t) j * kvd + (size_t) kh * 128];
                    double d = 0;
                    for (int e = 0; e < 128; ++e) d += (double) qv[e] * kv[e];
                    sc.push_back(d / std::sqrt(128.0));
                    vv.push_back(&vb[(size_t) j * kvd + (size_t) kh * 128]);
                }
                double m = -1e300, z = 0;
                for (double v : sc) m = std::max(m, v);
                for (double& v : sc) z += (v = std::exp(v - m));
                for (int e = 0; e < 128; ++e) {
                    double acc = 0;
                    for (size_t i = 0; i < sc.size(); ++i) acc += sc[i] * vv[i][e];
                    a[(size_t) t * qd + (size_t) hh * 128 + e] = (float) (acc / z);
                }
            }
        if (il == 0) cmp("a0", a, a.size());
        mm(P + "attn_output.weight", qd, E, a.data(), B, o.data());
        conv(o.data(), c.data(), R.t(P + "attn_conv_base").data(), 1, xc.data());
        if (il == 0) cmp("o0", o, o.size()), cmp("oc0", xc, xc.size());
        for (size_t i = 0; i < h.size(); ++i) h[i] += xc[i];
        for (int t = 0; t < B; ++t) rms(&h[(size_t) t * E], R.t(P + "ffn_norm.weight").data(), eps, E, &x[(size_t) t * E]);
        mm(P + "ffn_conv_proj.weight", E, 2 * taps * ng, x.data(), B, c.data());
        conv(x.data(), c.data(), R.t(P + "ffn_conv_base").data(), 0, xc.data());
        mm(P + "ffn_gate.weight", E, n_ff, xc.data(), B, gt.data());
        mm(P + "ffn_up.weight", E, n_ff, xc.data(), B, up.data());
        for (size_t i = 0; i < gt.size(); ++i) gt[i] = gt[i] / (1.0f + std::exp(-gt[i])) * up[i];
        mm(P + "ffn_down.weight", n_ff, E, gt.data(), B, o.data());
        conv(o.data(), c.data(), R.t(P + "ffn_conv_base").data(), 1, xc.data());
        for (size_t i = 0; i < h.size(); ++i) h[i] += xc[i];
        if (il == 0) cmp("l0", h, h.size());
    }
    std::vector<float> hn((size_t) n * E), lg((size_t) n * V), gate((size_t) n * rank);
    for (int t = 0; t < n; ++t) rms(&h[(size_t) (t + 1) * E], R.t("output_norm.weight").data(), eps, E, &hn[(size_t) t * E]);
    {
        std::vector<float> hw((size_t) V * E);
        for (size_t i = 0; i < hw.size(); ++i) hw[i] = from_bf16(head_h[i]);
        matmul(hw, E, V, hn.data(), n, lg.data(), false);   // (a BF16 head reads the float rows)
    }
    mm("selector_hidden.weight", E, rank, hn.data(), n, gate.data());
    const auto& prev = R.t("selector_predecessor.weight");
    const auto& next = R.t("selector_successor.weight");
    int bad = 0, pred = anchor, cand_miss = 0;
    double cos_min = 1.0;
    for (int t = 0; t < n; ++t) {
        double d = 0, na = 0, nb = 0;
        for (int e = 0; e < E; ++e) {
            d += (double) hn[(size_t) t * E + e] * g_h[(size_t) t * E + e];
            na += (double) hn[(size_t) t * E + e] * hn[(size_t) t * E + e];
            nb += (double) g_h[(size_t) t * E + e] * g_h[(size_t) t * E + e];
        }
        const double cs = d / std::sqrt(na * nb + 1e-30);
        cos_min = std::min(cos_min, cs);
        std::vector<int> ids((size_t) V);
        for (int i = 0; i < V; ++i) ids[(size_t) i] = i;
        const float* lr = &lg[(size_t) t * V];
        std::partial_sort(ids.begin(), ids.begin() + top_k, ids.end(),
                          [&](int a1, int b1) { return lr[a1] > lr[b1] || (lr[a1] == lr[b1] && a1 < b1); });
        for (int j = 0; j < top_k; ++j)
            if (std::find(g_cand.begin() + (size_t) t * top_k, g_cand.begin() + (size_t) (t + 1) * top_k, ids[(size_t) j]) ==
                g_cand.begin() + (size_t) (t + 1) * top_k)
                ++cand_miss;
        // the walk on the GPU's candidates (so a near-tie in the top-k does not count twice)
        std::vector<double> sc((size_t) top_k);
        for (int j = 0; j < top_k; ++j) {
            const int cj = g_cand[(size_t) t * top_k + j];
            double acc = lr[cj];
            for (int r = 0; r < rank; ++r)
                acc += (double) prev[(size_t) pred * rank + r] * gate[(size_t) t * rank + r] * next[(size_t) cj * rank + r];
            sc[(size_t) j] = acc;
        }
        const int b = (int) (std::max_element(sc.begin(), sc.end()) - sc.begin());
        const int ref_pick = g_cand[(size_t) t * top_k + b];
        double z = 0;
        for (double v : sc) z += std::exp(v - sc[(size_t) b]);
        std::printf("row %d: cos %.6f | pick gpu %6d ref %6d | conf gpu %.4f ref %.4f\n", t + 1, cs, g_pick[(size_t) t],
                    ref_pick, g_conf[(size_t) t], 1.0 / z);
        if (ref_pick != g_pick[(size_t) t]) ++bad;
        pred = g_pick[(size_t) t];
    }
    std::printf("n_past %lld: hidden cosine >= %.6f, %d of %d candidates missing, %d of %d picks differ\n",
                (long long) n_past, cos_min, cand_miss, n * top_k, bad, n);
    // (with the q8 emulation the five layers' FFNs still amplify the GPU's rounding on these synthetic rows: the drafter
    // as reviewed - either attention kernel - ends at cosine ~0.995 with 11-15 of 112 candidates and 0-2 near-tied
    // picks moved; the thresholds sit just outside that)
    const bool ok = cos_min > (g_q8 ? 0.99 : 0.98) && cand_miss <= n * top_k / (g_q8 ? 6 : 4) && bad <= (g_q8 ? 2 : 1);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    cudaFree(head_d);
    return ok ? 0 : 1;
}
