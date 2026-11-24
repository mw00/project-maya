// src/kernels/glm_layer_parity.cpp - WHOLE glm5-next layers through the kernel family, vs the oracle.
//
// Phase 2's composition proof: the individual kernels are parity-green (glm_hc, glm_kda, glm_dsa,
// glm_ffn, router_sigmoid), and this test runs two COMPLETE layers of the synthetic glm5-next model
// (tools/glm_synth_gguf.py) through them - layer 0 (KDA + dense FFN) and layer 3 (DSA + MoE) - and
// holds every seam to `ref/glm.py`'s dumps of the same model (data/glm5-synth-dumps/, written by
// `python3 ref/glm.py <gguf> --dump-raw-dir ...` in ggml order; regenerate both fixtures with
// `python3 tools/glm_synth_gguf.py --out data/glm5-synth.gguf` and the dump command).
//
// What runs on the DEVICE here is the new kernel family; what runs on the HOST is the plumbing the
// engine already has machinery for elsewhere (GEMVs, rms/layer norms, the l2 norm, expert matvecs) -
// accumulated in double so the comparison measures the KERNELS, not the host math.  A divergence
// here means the KERNEL FAMILY does not compose into the layer, which no single-kernel parity can
// catch: the layouts agree per kernel while the seams disagree between them.
#include "maya/artifact/gguf_reader.hpp"
#include "maya/kernels/glm_dsa.hpp"
#include "maya/kernels/glm_ffn.hpp"
#include "maya/kernels/glm_hc.hpp"
#include "maya/kernels/glm_kda.hpp"
#include "maya/kernels/router_sigmoid.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// ---- the oracle's dumps: name -> ggml-order values (the selection is int32, everything else f32)
std::map<std::string, std::vector<float>> g_dumps;
std::map<std::string, std::vector<int32_t>> g_idumps;

void load_dumps(const std::string& dir) {
    FILE* mf = std::fopen((dir + "/manifest.txt").c_str(), "r");
    if (!mf) {
        std::fprintf(stderr, "glm_layer_parity: no manifest at %s\n", dir.c_str());
        std::exit(1);
    }
    char name[256];
    long long v;
    int c;
    while (std::fscanf(mf, "%255s", name) == 1) {
        size_t n = 1;
        while ((c = std::fgetc(mf)) == ' ') {
            if (std::fscanf(mf, "%lld", &v) == 1) n *= (size_t) v;
        }
        if (c != '\n' && c != EOF) {
            std::fprintf(stderr, "glm_layer_parity: bad manifest line for %s\n", name);
            std::exit(1);
        }
        const bool is_int = std::strncmp(name, "ffn_moe_topk-", 13) == 0;
        if (is_int) {
            std::vector<int32_t> data(n);
            FILE* f = std::fopen((dir + "/" + name + ".f32").c_str(), "rb");
            if (!f || std::fread(data.data(), sizeof(int32_t), n, f) != n) {
                std::fprintf(stderr, "glm_layer_parity: cannot read dump %s\n", name);
                std::exit(1);
            }
            std::fclose(f);
            g_idumps[name] = std::move(data);
        } else {
            std::vector<float> data(n);
            FILE* f = std::fopen((dir + "/" + name + ".f32").c_str(), "rb");
            if (!f || std::fread(data.data(), sizeof(float), n, f) != n) {
                std::fprintf(stderr, "glm_layer_parity: cannot read dump %s\n", name);
                std::exit(1);
            }
            std::fclose(f);
            g_dumps[name] = std::move(data);
        }
    }
    std::fclose(mf);
}

const std::vector<float>& dump(const std::string& name) {
    auto it = g_dumps.find(name);
    if (it == g_dumps.end()) {
        std::fprintf(stderr, "glm_layer_parity: missing dump %s\n", name.c_str());
        std::exit(1);
    }
    return it->second;
}

double max_abs(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) m = std::max(m, (double) std::fabs(a[i] - b[i]));
    return m;
}

void require(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "glm_layer_parity: %s\n", what.c_str());
        std::exit(1);
    }
}

void compare(const std::vector<float>& got, const std::string& dump_name, double tol) {
    const double d = max_abs(got, dump(dump_name));
    char buf[128];
    std::snprintf(buf, sizeof buf, "%.3e", d);
    require(d < tol, dump_name + " diverges (" + buf + ")");
}

// ---- host math, f64 accumulations

std::vector<float> gemv(const std::vector<float>& w, size_t out, size_t in, const std::vector<float>& x, size_t T) {
    std::vector<float> y(out * T, 0.0f);
    for (size_t t = 0; t < T; ++t)
        for (size_t e = 0; e < out; ++e) {
            double acc = 0.0;
            for (size_t k = 0; k < in; ++k) acc += (double) w[e * in + k] * x[k + in * t];
            y[e + out * t] = (float) acc;
        }
    return y;
}

std::vector<float> rms_norm(const std::vector<float>& x, size_t dim, const std::vector<float>& w, size_t T,
                            double eps) {
    std::vector<float> y(x.size(), 0.0f);
    for (size_t t = 0; t < T; ++t) {
        double ms = 0.0;
        for (size_t e = 0; e < dim; ++e) ms += (double) x[e + dim * t] * x[e + dim * t];
        const double inv = 1.0 / std::sqrt(ms / dim + eps);
        for (size_t e = 0; e < dim; ++e) y[e + dim * t] = (float) ((double) x[e + dim * t] * inv * (double) w[e]);
    }
    return y;
}

std::vector<float> layer_norm(const std::vector<float>& x, const std::vector<float>& w,
                              const std::vector<float>& b, size_t dim, size_t T, double eps) {
    std::vector<float> y(x.size(), 0.0f);
    for (size_t t = 0; t < T; ++t) {
        double mu = 0.0;
        for (size_t e = 0; e < dim; ++e) mu += x[e + dim * t];
        mu /= dim;
        double var = 0.0;
        for (size_t e = 0; e < dim; ++e) var += ((double) x[e + dim * t] - mu) * ((double) x[e + dim * t] - mu);
        var /= dim;
        for (size_t e = 0; e < dim; ++e)
            y[e + dim * t] = (float) (((double) x[e + dim * t] - mu) / std::sqrt(var + eps) * (double) w[e] +
                                      (double) b[e]);
    }
    return y;
}

std::vector<float> gdn_l2(const std::vector<float>& x, size_t head_dim, size_t n_head, size_t T) {
    std::vector<float> y(x.size(), 0.0f);
    for (size_t t = 0; t < T; ++t)
        for (size_t h = 0; h < n_head; ++h) {
            double ss = 0.0;
            for (size_t e = 0; e < head_dim; ++e) {
                const double v = x[e + head_dim * h + head_dim * n_head * t];
                ss += v * v;
            }
            const double inv = 1.0 / std::sqrt(ss + 1e-6);
            for (size_t e = 0; e < head_dim; ++e)
                y[e + head_dim * h + head_dim * n_head * t] =
                    (float) ((double) x[e + head_dim * h + head_dim * n_head * t] * inv);
        }
    return y;
}

std::vector<float> sigmoid_v(const std::vector<float>& x) {
    std::vector<float> y(x.size(), 0.0f);
    for (size_t i = 0; i < x.size(); ++i) y[i] = (float) (1.0 / (1.0 + std::exp((double) -x[i])));
    return y;
}

std::vector<float> swiglu_clamp(const std::vector<float>& gate, const std::vector<float>& up, float limit) {
    std::vector<float> out(gate.size(), 0.0f);
    for (size_t i = 0; i < gate.size(); ++i) {
        const double g = std::min((double) gate[i], (double) limit);
        const double u = std::min(std::max((double) up[i], (double) -limit), (double) limit);
        out[i] = (float) ((g / (1.0 + std::exp(-g))) * u);
    }
    return out;
}

// ---- device staging: stage, upload once, run kernels, read back
struct Dev {
    std::vector<float> host;
    float* d = nullptr;
    std::vector<size_t> at;
    size_t stage(const std::vector<float>& v) {
        at.push_back(host.size());
        host.insert(host.end(), v.begin(), v.end());
        return at.size() - 1;
    }
    size_t stage_zero(size_t n) {
        at.push_back(host.size());
        host.resize(host.size() + n, 0.0f);
        return at.size() - 1;
    }
    void upload() {
        check(cudaMalloc(&d, host.size() * sizeof(float)), "cudaMalloc");
        check(cudaMemcpy(d, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice),
              "staging H2D");
    }
    std::vector<float> read(size_t slot, size_t n) const {
        std::vector<float> v(n);
        check(cudaMemcpy(v.data(), d + at[slot], n * sizeof(float), cudaMemcpyDeviceToHost), "D2H");
        return v;
    }
    float* dev(size_t slot) const { return d + at[slot]; }
    ~Dev() { cudaFree(d); }
};

}  // namespace

int main(int argc, char** argv) {
    const bool selftest = argc > 1 && std::string(argv[1]) == "--selftest";
    if (!selftest) {
        std::fprintf(stderr, "glm_layer_parity: run with --selftest (the CTest form)\n");
        return 2;
    }
    const std::string gguf_path = argc > 2 ? argv[2] : "data/glm5-synth.gguf";
    const std::string dumps_path = argc > 3 ? argv[3] : "data/glm5-synth-dumps";
    load_dumps(dumps_path);

    maya::GgufFile gguf(gguf_path);
    const auto tensor = [&](const std::string& name) -> std::vector<float> {
        const auto* t = gguf.find(name);
        if (!t) {
            std::fprintf(stderr, "glm_layer_parity: missing tensor %s\n", name.c_str());
            std::exit(1);
        }
        const uint8_t* p = gguf.tensor_data(*t);
        return std::vector<float>((const float*) p, (const float*) p + t->elements());
    };
    // scalar KVs read .u; ARRAY KVs carry their values in .items (with .u == 0), so the array form
    // reads its first element - expert_feed_forward_length is the one array this test consumes
    const auto meta_u = [&](const char* key, int def) -> int {
        const auto* v = gguf.get(key);
        if (!v) return def;
        if (!v->items.empty()) return v->items[0].is_num() ? (int) v->items[0].u : def;
        return v->is_num() ? (int) v->u : def;
    };

    const int n_embd = meta_u("glm5-next.embedding_length", 128);
    const int n_head = meta_u("glm5-next.attention.head_count", 8);
    const int hc = meta_u("glm5-next.hyper_connection.count", 4);
    const int sinkhorn = meta_u("glm5-next.hyper_connection.sinkhorn_iterations", 3);
    const int kda_head_dim = meta_u("glm5-next.kda.head_dim", 32);
    const int d_inner = n_head * kda_head_dim;
    const int d_conv = meta_u("glm5-next.ssm.conv_kernel", 4);
    const int q_lora = meta_u("glm5-next.attention.q_lora_rank", 96);
    const int kv_lora = meta_u("glm5-next.attention.kv_lora_rank", 64);
    const int qk_nope = meta_u("glm5-next.attention.key_length_mla", 48);
    const int v_head = meta_u("glm5-next.attention.value_length_mla", 48);
    const int idx_heads = meta_u("glm5-next.attention.indexer.head_count", 4);
    const int idx_key = meta_u("glm5-next.attention.indexer.key_length", 16);
    const int kpool = meta_u("glm5-next.attention.indexer.kpool", 2);
    const int top_pools = meta_u("glm5-next.attention.indexer.top_k", 8) / kpool;
    const int n_expert = meta_u("glm5-next.expert_count", 12);
    const int n_exp_used = meta_u("glm5-next.expert_used_count", 3);
    const int n_ff_exp = meta_u("glm5-next.expert_feed_forward_length", 64);
    const double norm_eps = 1e-5, hc_eps = 1e-6, kda_lb = -5.0;
    const float swiglu_limit = 10.0f;
    const size_t T = dump("hc_init").size() / ((size_t) hc * n_embd);
    const size_t hc_dim = (size_t) hc * n_embd;

    maya::kernels::GlmHcShapes hcs;
    hcs.n_embd = n_embd;
    hcs.hc = hc;
    hcs.sinkhorn_iters = sinkhorn;

    auto hc_half = [&](const std::vector<float>& R, const std::string& half, int il, std::vector<float>& mixed,
                       std::vector<float>& post, std::vector<float>& comb) {
        const std::string L = std::to_string(il);
        Dev dev;
        dev.stage(R);                                                        // 0
        dev.stage(tensor("blk." + L + ".hc_" + half + "_fn.weight"));        // 1
        dev.stage(tensor("blk." + L + ".hc_" + half + "_scale.weight"));     // 2
        dev.stage(tensor("blk." + L + ".hc_" + half + "_base.weight"));      // 3
        dev.stage_zero(hc_dim * T);                                          // 4 mixed
        dev.stage_zero((size_t) hc * T);                                     // 5 pre
        dev.stage_zero((size_t) hc * T);                                     // 6 post
        dev.stage_zero((size_t) hc * hc * T);                                // 7 comb
        dev.stage_zero(1);                                                   // 8 inv_rms workspace
        dev.upload();
        maya::kernels::GlmHcWorkspace ws;
        maya::kernels::glm_hc_workspace_init(hcs, dev.dev(8), ws);
        for (size_t t = 0; t < T; ++t)
            maya::kernels::glm_hc_pre(dev.dev(0) + t * hc_dim, dev.dev(1), dev.dev(2), dev.dev(3),
                                        (float) norm_eps, (float) hc_eps, hcs, ws, dev.dev(4) + t * n_embd,
                                        dev.dev(5) + t * hc, dev.dev(6) + t * hc, dev.dev(7) + t * hc * hc,
                                        nullptr);
        check(cudaGetLastError(), "hc_pre launch");
        check(cudaDeviceSynchronize(), "hc_pre sync");
        mixed = dev.read(4, hc_dim * T);
        post = dev.read(6, (size_t) hc * T);
        comb = dev.read(7, (size_t) hc * hc * T);
        compare(post, "hc_post_" + half + "-" + L, 1e-5);
        // the dump mirrors ggml's comb (ne0 = dst fastest), so its flat index is t*hc*hc + s*hc + d
        // while the kernel writes d*hc + s - compare through the transpose
        std::vector<float> comb_t(comb.size());
        for (size_t t = 0; t < T; ++t)
            for (int d = 0; d < hc; ++d)
                for (int s = 0; s < hc; ++s)
                    comb_t[t * hc * hc + (size_t) d * hc + s] = comb[t * hc * hc + (size_t) s * hc + d];
        compare(comb_t, "hc_comb_" + half + "-" + L, 1e-5);
    };
    auto hc_write = [&](const std::vector<float>& block_out, const std::vector<float>& R,
                        const std::vector<float>& post, const std::vector<float>& comb) {
        Dev dev;
        dev.stage(block_out);
        dev.stage(R);
        dev.stage(post);
        dev.stage(comb);
        dev.stage_zero(hc_dim * T);
        dev.upload();
        for (size_t t = 0; t < T; ++t)
            maya::kernels::glm_hc_post(dev.dev(0) + t * n_embd, dev.dev(1) + t * hc_dim, dev.dev(2) + t * hc,
                                         dev.dev(3) + t * hc * hc, hcs, dev.dev(4) + t * hc_dim, nullptr);
        check(cudaGetLastError(), "hc_post launch");
        check(cudaDeviceSynchronize(), "hc_post sync");
        return dev.read(4, hc_dim * T);
    };

    // ================= layer 0: KDA + dense FFN =================
    {
        std::vector<float> R = dump("hc_init");
        std::vector<float> mixed, post, comb;
        hc_half(R, "attn", 0, mixed, post, comb);
        const std::vector<float> attn_w0 = tensor("blk.0.attn_norm.weight");
        const std::vector<float> x = rms_norm(mixed, n_embd, attn_w0, T, norm_eps);
        compare(x, "attn_norm-0", 1e-4);

        std::vector<float> q_proj0;
        auto conv_one = [&](const char* s) {
            const std::vector<float> proj =
                gemv(tensor("blk.0.attn_" + std::string(s) + ".weight"), d_inner, n_embd, x, T);
            if (std::string(s) == "q") q_proj0 = proj;
            Dev d;
            d.stage(proj);
            d.stage(tensor("blk.0.ssm_conv1d_" + std::string(s) + ".weight"));
            d.stage_zero((size_t) d_inner * (d_conv - 1));
            d.stage_zero((size_t) d_inner * T);
            d.upload();
            maya::kernels::glm_kda_conv(d.dev(0), d.dev(1), d.dev(2), d_inner, d_conv, (int) T, d.dev(3), nullptr);
            check(cudaGetLastError(), "kda conv launch");
            check(cudaDeviceSynchronize(), "kda conv sync");
            return d.read(3, (size_t) d_inner * T);
        };
        const std::vector<float> q_c = conv_one("q");
        const std::vector<float> k_c = conv_one("k");
        const std::vector<float> v_c = conv_one("v");
        compare(q_c, "kda_q_conv-0", 1e-4);
        compare(k_c, "kda_k_conv-0", 1e-4);
        compare(v_c, "kda_v_conv-0", 1e-4);

        std::vector<float> g = gemv(tensor("blk.0.ssm_f_b.weight"), d_inner, kda_head_dim,
                                    gemv(tensor("blk.0.ssm_f_a.weight"), kda_head_dim, n_embd, x, T), T);
        const std::vector<float> dtb = tensor("blk.0.ssm_dt.bias");
        for (size_t t = 0; t < T; ++t)
            for (int c = 0; c < d_inner; ++c) g[c + (size_t) d_inner * t] += dtb[c];

        Dev d3;
        d3.stage(g);                                                             // 0
        d3.stage(tensor("blk.0.ssm_a"));                                  // 1
        d3.stage_zero((size_t) d_inner * T);                                     // 2 g1
        d3.stage(gdn_l2(q_c, kda_head_dim, n_head, T));                          // 3 q
        d3.stage(gdn_l2(k_c, kda_head_dim, n_head, T));                          // 4 k
        d3.stage(v_c);                                                           // 5 v
        d3.stage(sigmoid_v(gemv(tensor("blk.0.ssm_beta.weight"), n_head, n_embd, x, T)));  // 6 beta
        d3.stage_zero((size_t) kda_head_dim * kda_head_dim * n_head);            // 7 state
        d3.stage_zero((size_t) d_inner * T);                                     // 8 out
        d3.upload();
        maya::kernels::glm_kda_gate(d3.dev(0), d3.dev(1), (float) kda_lb, kda_head_dim, n_head, (int) T,
                                      d3.dev(2), nullptr);
        maya::kernels::glm_kda_recurrence(d3.dev(3), d3.dev(4), d3.dev(5), d3.dev(2), d3.dev(6), d3.dev(7),
                                            n_head, kda_head_dim, (int) T, d3.dev(8), nullptr);
        check(cudaGetLastError(), "kda kernels launch");
        check(cudaDeviceSynchronize(), "kda sync");
        compare(d3.read(2, (size_t) d_inner * T), "kda_g1-0", 1e-5);
        const std::vector<float> scan = d3.read(8, (size_t) d_inner * T);
        compare(scan, "kda_scan_out-0", 2e-4);

        const std::vector<float> g2 = gemv(tensor("blk.0.ssm_g_b.weight"), d_inner, kda_head_dim,
                                           gemv(tensor("blk.0.ssm_g_a.weight"), kda_head_dim, n_embd, x, T), T);
        Dev d4;
        d4.stage(scan);
        d4.stage(tensor("blk.0.ssm_norm.weight"));
        d4.stage(g2);
        d4.stage_zero((size_t) d_inner * T);
        d4.upload();
        maya::kernels::glm_kda_out_gate(d4.dev(0), d4.dev(1), d4.dev(2), kda_head_dim, n_head, (int) T,
                                          (float) norm_eps, d4.dev(3), nullptr);
        check(cudaGetLastError(), "out_gate launch");
        check(cudaDeviceSynchronize(), "out_gate sync");
        const std::vector<float> kda_out =
            gemv(tensor("blk.0.attn_output.weight"), n_embd, d_inner, d4.read(3, (size_t) d_inner * T), T);
        compare(kda_out, "kda_out-0", 1e-4);

        R = hc_write(kda_out, R, post, comb);
        compare(R, "hc_attn_post-0", 1e-4);

        std::vector<float> mixed2, post2, comb2;
        hc_half(R, "ffn", 0, mixed2, post2, comb2);
        const std::vector<float> ffn_w0 = tensor("blk.0.ffn_norm.weight");
        const std::vector<float> xf = rms_norm(mixed2, n_embd, ffn_w0, T, norm_eps);
        compare(xf, "ffn_norm-0", 1e-4);
        const size_t n_ff_dense = tensor("blk.0.ffn_down.weight").size() / n_embd;
        const std::vector<float> ffn_out =
            gemv(tensor("blk.0.ffn_down.weight"), n_embd, n_ff_dense,
                 swiglu_clamp(gemv(tensor("blk.0.ffn_gate.weight"), n_ff_dense, n_embd, xf, T),
                              gemv(tensor("blk.0.ffn_up.weight"), n_ff_dense, n_embd, xf, T), swiglu_limit),
                 T);
        compare(ffn_out, "ffn_out-0", 1e-4);
        R = hc_write(ffn_out, R, post2, comb2);
        compare(R, "l_out-0", 1e-4);
    }

    // ================= layer 3: DSA + MoE =================
    {
        std::vector<float> R = dump("l_out-2");
        std::vector<float> mixed, post, comb;
        hc_half(R, "attn", 3, mixed, post, comb);
        const std::vector<float> attn_w3 = tensor("blk.3.attn_norm.weight");
        const std::vector<float> x = rms_norm(mixed, n_embd, attn_w3, T, norm_eps);
        compare(x, "attn_norm-3", 1e-4);

        const std::vector<float> q_a_norm3 = tensor("blk.3.attn_q_a_norm.weight");
        const std::vector<float> qr = rms_norm(gemv(tensor("blk.3.attn_q_a.weight"), q_lora, n_embd, x, T), q_lora,
                                               q_a_norm3, T, norm_eps);
        compare(qr, "q_resid-3", 1e-4);
        const std::vector<float> kv_a_norm3 = tensor("blk.3.attn_kv_a_norm.weight");
        const std::vector<float> kv = rms_norm(gemv(tensor("blk.3.attn_kv_a_mqa.weight"), kv_lora, n_embd, x, T),
                                               kv_lora, kv_a_norm3, T, norm_eps);

        compare(kv, "kv_cmpr-3", 1e-4);

        const std::vector<float> q = gemv(tensor("blk.3.attn_q_b.weight"), (size_t) n_head * qk_nope, q_lora, qr, T);
        const std::vector<float> wk_b = tensor("blk.3.attn_k_b.weight");
        std::vector<float> q_abs((size_t) kv_lora * n_head * T, 0.0f);
        for (size_t t = 0; t < T; ++t)
            for (int h = 0; h < n_head; ++h)
                for (int c = 0; c < kv_lora; ++c) {
                    double acc = 0.0;
                    for (int e = 0; e < qk_nope; ++e)
                        acc += (double) wk_b[(size_t) e + qk_nope * c + (size_t) qk_nope * kv_lora * h] *
                               q[e + (size_t) qk_nope * h + (size_t) qk_nope * n_head * t];
                    q_abs[c + (size_t) kv_lora * h + (size_t) kv_lora * n_head * t] = (float) acc;
                }
        compare(q_abs, "q_absorbed-3", 1e-4);

        const std::vector<float> ik = layer_norm(gemv(tensor("blk.3.indexer.attn_k.weight"), idx_key, n_embd, x, T),
                                                 tensor("blk.3.indexer.k_norm.weight"),
                                                 tensor("blk.3.indexer.k_norm.bias"), idx_key, T, norm_eps);
        compare(ik, "indexer_k-3", 1e-4);
        const std::vector<float> ig = gemv(tensor("blk.3.indexer_compressor_gate.weight"), idx_key, n_embd, x, T);
        compare(ig, "indexer_gate-3", 1e-4);
        const std::vector<float> iq = gemv(tensor("blk.3.indexer.attn_q_b.weight"), (size_t) idx_heads * idx_key,
                                           q_lora, qr, T);
        compare(iq, "indexer_q-3", 1e-4);

        const int n_pools = (int) T / kpool;
        const float prescale = (float) (1.0 / std::sqrt((double) idx_key * idx_heads));
        const std::vector<float> idx_w_scaled = [&] {
            const std::vector<float> idx_w = gemv(tensor("blk.3.indexer.proj.weight"), idx_heads, n_embd, x, T);
            std::vector<float> scaled(idx_w.size());
            for (size_t i = 0; i < idx_w.size(); ++i) scaled[i] = idx_w[i] * prescale;
            return scaled;
        }();
        compare(idx_w_scaled, "indexer_weights-3", 1e-5);

        const int n_sel = kpool * top_pools + (kpool - 1);
        Dev d5;
        d5.stage(ik);                                                 // 0
        d5.stage(ig);                                                 // 1
        d5.stage(tensor("blk.3.indexer_compressor_ape.weight"));           // 2
        d5.stage_zero((size_t) idx_key * n_pools);                    // 3 pooled
        d5.stage(iq);                                                 // 4
        d5.stage(idx_w_scaled);                                       // 5
        d5.stage_zero((size_t) n_pools * T);                          // 6 score
        d5.stage(q_abs);                                              // 7
        d5.stage(kv);                                                 // 8 the latent cache
        d5.stage(tensor("blk.3.attn_v_b.weight"));                    // 9
        d5.stage_zero((size_t) n_sel * T);                            // 10 cells (int)
        d5.stage_zero((size_t) v_head * n_head * T);                  // 11 out
        d5.upload();
        maya::kernels::glm_dsa_pool(d5.dev(0), d5.dev(1), d5.dev(2), idx_key, kpool, n_pools, d5.dev(3), nullptr);
        maya::kernels::glm_dsa_score(d5.dev(4), d5.dev(3), d5.dev(5), idx_key, idx_heads, (int) T, n_pools,
                                       d5.dev(6), nullptr);
        std::vector<int> ident((size_t) T);
        for (int t = 0; t < (int) T; ++t) ident[(size_t) t] = t;
        int* d_ident = nullptr;
        check(cudaMalloc(&d_ident, ident.size() * sizeof(int)), "ident");
        check(cudaMemcpy(d_ident, ident.data(), ident.size() * sizeof(int), cudaMemcpyHostToDevice), "ident H2D");
        maya::kernels::glm_dsa_select(d5.dev(6), n_pools, kpool, top_pools, 1, (int) T, n_sel, d_ident,
                                        (int*) d5.dev(10), nullptr);
        cudaFree(d_ident);
        maya::kernels::glm_mla_attention(d5.dev(7), d5.dev(8), (int*) d5.dev(10), d5.dev(9), kv_lora, n_head,
                                           v_head, qk_nope, (int) T, n_sel, d5.dev(11), nullptr);
        check(cudaGetLastError(), "dsa kernels launch");
        check(cudaDeviceSynchronize(), "dsa sync");
        compare(d5.read(3, (size_t) idx_key * n_pools), "indexer_pool_k-3", 1e-4);
        compare(d5.read(6, (size_t) n_pools * T), "indexer_score-3", 1e-4);
        const std::vector<float> kqv = d5.read(11, (size_t) v_head * n_head * T);
        compare(kqv, "kqv_out-3", 1e-4);
        const std::vector<float> attn =
            gemv(tensor("blk.3.attn_output.weight"), n_embd, (size_t) n_head * v_head, kqv, T);
        compare(attn, "attn_out-3", 1e-4);

        R = hc_write(attn, R, post, comb);
        compare(R, "hc_attn_post-3", 1e-4);

        // the FFN half: MoE + shared expert
        std::vector<float> mixed2, post2, comb2;
        hc_half(R, "ffn", 3, mixed2, post2, comb2);
        const std::vector<float> ffn_w3 = tensor("blk.3.ffn_norm.weight");
        const std::vector<float> xf = rms_norm(mixed2, n_embd, ffn_w3, T, norm_eps);
        compare(xf, "ffn_norm-3", 1e-4);

        const std::vector<float> logits = gemv(tensor("blk.3.ffn_gate_inp.weight"), n_expert, n_embd, xf, T);
        Dev d7;
        d7.stage(logits);                                 // 0
        d7.stage(tensor("blk.3.exp_probs_b.bias"));       // 1
        d7.stage_zero((size_t) n_exp_used * T);           // 2 ids (int32)
        d7.stage_zero((size_t) n_exp_used * T);           // 3 weights
        d7.upload();
        maya::kernels::router_sigmoid(d7.dev(0), d7.dev(1), (int) T, n_expert, n_exp_used, 2.5f, true,
                                        (int*) d7.dev(2), d7.dev(3), nullptr);
        check(cudaGetLastError(), "router launch");
        check(cudaDeviceSynchronize(), "router sync");
        std::vector<int> ids((size_t) n_exp_used * T);
        check(cudaMemcpy(ids.data(), d7.dev(2), ids.size() * sizeof(int), cudaMemcpyDeviceToHost), "ids");
        const std::vector<float> router_w = d7.read(3, (size_t) n_exp_used * T);
        compare(router_w, "ffn_moe_weights-3", 1e-4);
        {
            const std::vector<int32_t>& ref_ids = g_idumps.at("ffn_moe_topk-3");
            bool ok = ref_ids.size() == ids.size();
            for (size_t i = 0; ok && i < ids.size(); ++i) ok = ref_ids[i] == ids[i];
            require(ok, "ffn_moe_topk-3 diverges");
        }

        const std::vector<float> gate_exps = tensor("blk.3.ffn_gate_exps.weight");
        const std::vector<float> up_exps = tensor("blk.3.ffn_up_exps.weight");
        // ffn_down_exps GGUF is {n_ff, n_embd, n_expert}: ne0 is the INPUT dim, so the flat
        // index of down_e[o, v] is v + n_ff*o + n_ff*n_embd*e - the transpose of gate/up
        const std::vector<float> down_exps = tensor("blk.3.ffn_down_exps.weight");
        std::vector<float> moe((size_t) n_embd * T, 0.0f);
        for (size_t t = 0; t < T; ++t) {
            std::vector<double> acc(n_embd, 0.0);
            for (int i = 0; i < n_exp_used; ++i) {
                const int e = ids[i + (size_t) n_exp_used * t];
                const float w = router_w[i + (size_t) n_exp_used * t];
                std::vector<float> ge(n_ff_exp), ue(n_ff_exp);
                for (int v = 0; v < n_ff_exp; ++v) {
                    double a1 = 0.0, a2 = 0.0;
                    for (int k = 0; k < n_embd; ++k) {
                        const double xv = xf[k + n_embd * t];
                        a1 += (double) gate_exps[(size_t) k + (size_t) n_embd * v + (size_t) n_embd * n_ff_exp * e] *
                              xv;
                        a2 += (double) up_exps[(size_t) k + (size_t) n_embd * v + (size_t) n_embd * n_ff_exp * e] * xv;
                    }
                    ge[v] = (float) a1;
                    ue[v] = (float) a2;
                }
                const std::vector<float> hh = swiglu_clamp(ge, ue, swiglu_limit);
                for (int o = 0; o < n_embd; ++o) {
                    double a = 0.0;
                    for (int v = 0; v < n_ff_exp; ++v)
                        a += (double) down_exps[(size_t) v + (size_t) n_ff_exp * o + (size_t) n_embd * n_ff_exp * e] *
                             hh[v];
                    acc[o] += (double) w * a;
                }
            }
            for (int o = 0; o < n_embd; ++o) moe[o + n_embd * t] = (float) acc[o];
        }
        compare(moe, "ffn_moe_out-3", 2e-4);
        const std::vector<float> shexp =
            gemv(tensor("blk.3.ffn_down_shexp.weight"), n_embd, n_ff_exp,
                 swiglu_clamp(gemv(tensor("blk.3.ffn_gate_shexp.weight"), n_ff_exp, n_embd, xf, T),
                              gemv(tensor("blk.3.ffn_up_shexp.weight"), n_ff_exp, n_embd, xf, T), swiglu_limit),
                 T);
        std::vector<float> ffn_out((size_t) n_embd * T);
        for (size_t i = 0; i < ffn_out.size(); ++i) ffn_out[i] = moe[i] + shexp[i];
        compare(ffn_out, "ffn_out-3", 2e-4);
        R = hc_write(ffn_out, R, post2, comb2);
        compare(R, "l_out-3", 1e-4);
    }

    std::printf("glm_layer_parity: PASS\n");
    return 0;
}
