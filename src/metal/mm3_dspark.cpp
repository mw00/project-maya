// src/metal/mm3_dspark.cpp - Mm3Draft (mm3_dspark.hpp).  ggml graphs on the Metal backend,
// the same idioms as mm3_metal.cpp: weights in a persistent backend context, one scheduler,
// per-call graphs in a scratch context.
#include "mm3_dspark.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

namespace maya::metal {
namespace {

constexpr int64_t kChunk = 1024;      // context rows a graph (CUDA's max_rows shape)
constexpr size_t kGraphCtxBytes = 256 * 1024 * 1024;
constexpr int kGraphNodes = 8192;

// ---------------------------------------------------------------- the safetensors file
struct StTensor {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t begin = 0, end = 0;   // absolute file offsets of the data
};

struct StFile {
    std::vector<uint8_t> bytes;   // the whole file (a draft is 2.3 GB; it is read once)
    std::map<std::string, StTensor> tensors;

    bool open(const std::string& path, std::string& err) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) {
            err = "dspark: cannot open " + path;
            return false;
        }
        const std::streamoff size = f.tellg();
        bytes.resize((size_t) size);
        f.seekg(0);
        if (!f.read((char*) bytes.data(), size)) {
            err = "dspark: cannot read " + path;
            return false;
        }
        if (bytes.size() < 8) {
            err = "dspark: " + path + " is truncated";
            return false;
        }
        uint64_t hlen = 0;
        std::memcpy(&hlen, bytes.data(), 8);
        if (8 + hlen > bytes.size()) {
            err = "dspark: " + path + " has a bad header";
            return false;
        }
        return parse_header(std::string((const char*) bytes.data() + 8, (size_t) hlen),
                            8 + hlen, err);
    }

    // a small scanner for the header's flat {"name": {"dtype": .., "shape": [..],
    // "data_offsets": [a, b]}} objects - enough for safetensors headers, nothing more
    bool parse_header(const std::string& h, uint64_t data0, std::string& err) {
        size_t i = 0;
        auto ws = [&] { while (i < h.size() && (h[i] == ' ' || h[i] == '\n' || h[i] == '\t' || h[i] == '\r')) ++i; };
        auto str = [&](std::string& out) -> bool {
            if (i >= h.size() || h[i] != '"') return false;
            ++i;
            out.clear();
            while (i < h.size() && h[i] != '"') {
                if (h[i] == '\\' && i + 1 < h.size()) ++i;
                out += h[i++];
            }
            ++i;
            return true;
        };
        auto num = [&](int64_t& out) -> bool {
            ws();
            const size_t s = i;
            if (i < h.size() && h[i] == '-') ++i;
            while (i < h.size() && h[i] >= '0' && h[i] <= '9') ++i;
            if (s == i) return false;
            out = std::stoll(h.substr(s, i - s));
            return true;
        };
        ws();
        if (i >= h.size() || h[i] != '{') {
            err = "dspark: bad safetensors header";
            return false;
        }
        ++i;
        for (;;) {
            ws();
            if (i < h.size() && h[i] == '}') break;
            std::string name;
            if (!str(name)) {
                err = "dspark: bad safetensors header (name)";
                return false;
            }
            ws();
            if (i >= h.size() || h[i] != ':') return false;
            ++i;
            ws();
            if (name == "__metadata__") {   // skip its object
                int depth = 0;
                do {
                    if (h[i] == '{') ++depth;
                    else if (h[i] == '}') --depth;
                    ++i;
                } while (i < h.size() && depth > 0);
            } else {
                if (i >= h.size() || h[i] != '{') return false;
                ++i;
                StTensor t;
                uint64_t a = 0, b = 0;
                for (;;) {
                    ws();
                    if (i < h.size() && h[i] == '}') {
                        ++i;
                        break;
                    }
                    std::string key;
                    if (!str(key)) return false;
                    ws();
                    if (h[i] != ':') return false;
                    ++i;
                    ws();
                    if (key == "dtype") {
                        if (!str(t.dtype)) return false;
                    } else if (key == "shape" || key == "data_offsets") {
                        if (h[i] != '[') return false;
                        ++i;
                        std::vector<int64_t> vals;
                        for (;;) {
                            ws();
                            if (i < h.size() && h[i] == ']') {
                                ++i;
                                break;
                            }
                            int64_t v = 0;
                            if (!num(v)) return false;
                            vals.push_back(v);
                            ws();
                            if (i < h.size() && h[i] == ',') ++i;
                        }
                        if (key == "shape") t.shape = vals;
                        else if (vals.size() == 2) {
                            a = (uint64_t) vals[0];
                            b = (uint64_t) vals[1];
                        }
                    } else {   // an unknown field: skip a scalar
                        while (i < h.size() && h[i] != ',' && h[i] != '}') ++i;
                    }
                    ws();
                    if (i < h.size() && h[i] == ',') ++i;
                }
                t.begin = data0 + a;
                t.end = data0 + b;
                tensors[name] = t;
            }
            ws();
            if (i < h.size() && h[i] == ',') ++i;
        }
        return true;
    }

    const uint8_t* data(const StTensor& t) const { return bytes.data() + t.begin; }
    size_t size(const StTensor& t) const { return (size_t) (t.end - t.begin); }
};

// a config.json value hunt: the file is small and flat enough that "key": value scans are
// unambiguous for the keys read here (text_config repeats the top level with the same values)
bool cfg_num(const std::string& text, const std::string& key, double& out) {
    const std::string pat = "\"" + key + "\"";
    const size_t at = text.find(pat);
    if (at == std::string::npos) return false;
    const size_t colon = text.find(':', at + pat.size());
    if (colon == std::string::npos) return false;
    size_t i = colon + 1;
    while (i < text.size() && (text[i] == ' ' || text[i] == '\t' || text[i] == '\n')) ++i;
    const size_t s = i;
    if (i < text.size() && (text[i] == '-' || text[i] == '+')) ++i;
    while (i < text.size() && ((text[i] >= '0' && text[i] <= '9') || text[i] == '.' || text[i] == 'e' || text[i] == 'E' || text[i] == '-' || text[i] == '+')) ++i;
    if (s == i) return false;
    out = std::stod(text.substr(s, i - s));
    return true;
}

bool cfg_ints(const std::string& text, const std::string& key, std::vector<int>& out) {
    const std::string pat = "\"" + key + "\"";
    const size_t at = text.find(pat);
    if (at == std::string::npos) return false;
    const size_t lb = text.find('[', at);
    const size_t rb = text.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos) return false;
    out.clear();
    std::string cur;
    for (size_t i = lb + 1; i <= rb; ++i) {
        const char c = i < rb ? text[i] : ',';
        if (c == ',' || c == ' ' || c == '\n') {
            if (!cur.empty()) {
                out.push_back(std::stoi(cur));
                cur.clear();
            }
        } else cur += c;
    }
    return !out.empty();
}

float e4m3(uint8_t b) {
    static float tab[256];
    static bool init = [] {
        for (int i = 0; i < 256; ++i) {
            const int e = (i >> 3) & 0xF, m = i & 0x7;
            float v = e == 0 ? (float) m / 8.0f * 0.015625f
                             : (1.0f + (float) m / 8.0f) * std::ldexp(1.0f, e - 7);
            tab[i] = i & 0x80 ? -v : v;
        }
        return true;
    }();
    (void) init;
    return tab[b];
}

uint16_t f32_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t lsb = (u >> 16) & 1;
    u += 0x7FFF + lsb;
    return (uint16_t) (u >> 16);
}

float bf16_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

const std::string DP = "language_model.model.dspark.";

}  // namespace

struct Mm3Draft::Impl {
    Mm3DraftGeo g;
    ggml_backend_t backend = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_context* wctx = nullptr;         // weights + caches (backend-owned)
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_tensor *emb = nullptr, *head = nullptr;   // the target's (borrowed)
    int64_t max_ctx = 0, hi = 0;          // hi: one past the highest context position fed

    // weights (ggml layout: a safetensors [N][K] matrix is ne {K, N}; mul_mat contracts K)
    ggml_tensor* fc = nullptr;                       // F32 [NT*D][D]
    std::vector<float> hid1, fin1, cw;               // F32 host copies (norms at +1, conf row)
    float cb = 0.0f;
    ggml_tensor *hid1_t = nullptr, *fin1_t = nullptr;
    ggml_tensor *w1 = nullptr, *w2 = nullptr;        // F32 [R][V]
    struct L {
        ggml_tensor *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr;
        ggml_tensor *gt = nullptr, *up = nullptr, *dn = nullptr;
        ggml_tensor *in1 = nullptr, *post1 = nullptr, *qn1 = nullptr, *kn1 = nullptr;   // F32, +1 applied
        ggml_tensor *kc = nullptr, *vc = nullptr;   // BF16 [KV][max_ctx]
    };
    std::vector<L> layers;
    ggml_tensor* ctx_acc = nullptr;   // F32 [D][kChunk]

    std::vector<float> last_hid, last_logits;

    ~Impl() {
        if (sched) ggml_backend_sched_free(sched);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
        if (backend) ggml_backend_free(backend);
    }

    ggml_tensor* new_wt(ggml_type type, int64_t ne0, int64_t ne1) {
        ggml_tensor* t = ggml_new_tensor_2d(wctx, type, ne0, ne1);
        ggml_set_name(t, "dspark_w");
        return t;
    }

    // one graph: build in a scratch context, allocate on the scheduler, upload inputs, run
    bool compute(ggml_context* gctx, ggml_cgraph* gf,
                 const std::vector<std::pair<ggml_tensor*, std::vector<uint8_t>>>& up,
                 std::string& err) {
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            err = "dspark: cannot allocate the compute graph";
            return false;
        }
        for (const auto& [t, d] : up) ggml_backend_tensor_set(t, d.data(), 0, d.size());
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
            err = "dspark: the compute graph failed";
            return false;
        }
        (void) gctx;
        return true;
    }

    struct Graph {
        ggml_context* ctx = nullptr;
        ggml_cgraph* gf = nullptr;
        std::vector<std::pair<ggml_tensor*, std::vector<uint8_t>>> up;
        explicit Graph(size_t bytes = kGraphCtxBytes) {
            ggml_init_params ip{(size_t) bytes, nullptr, true};
            ctx = ggml_init(ip);
            gf = ggml_new_graph_custom(ctx, kGraphNodes, false);
        }
        ~Graph() { if (ctx) ggml_free(ctx); }
        ggml_tensor* in(ggml_type type, int64_t ne0, int64_t ne1, std::vector<uint8_t> data) {
            ggml_tensor* t = ggml_new_tensor_2d(ctx, type, ne0, ne1);
            ggml_set_input(t);
            up.emplace_back(t, std::move(data));
            return t;
        }
        ggml_tensor* in1(ggml_type type, int64_t ne0, std::vector<uint8_t> data) {
            ggml_tensor* t = ggml_new_tensor_1d(ctx, type, ne0);
            ggml_set_input(t);
            up.emplace_back(t, std::move(data));
            return t;
        }
    };

    ggml_tensor* rms(ggml_context* c, ggml_tensor* x, ggml_tensor* w) {
        x = ggml_rms_norm(c, x, g.eps);
        return w ? ggml_mul(c, x, w) : x;
    }

    // per-head Gemma norm over the head dim of a [hd][heads][rows] tensor
    ggml_tensor* head_norm(ggml_context* c, ggml_tensor* x, ggml_tensor* w) {
        x = ggml_rms_norm(c, x, g.eps);
        return ggml_mul(c, x, ggml_view_2d(c, w, w->ne[0], 1, w->nb[1], 0));
    }

    ggml_tensor* rope(ggml_context* c, ggml_tensor* x, ggml_tensor* pos) {
        return ggml_rope_ext(c, x, pos, nullptr, g.n_rot, GGML_ROPE_TYPE_NEOX, 1048576,
                             g.theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    }
};

bool Mm3Draft::load(const std::string& dir, ggml_tensor* emb, ggml_tensor* head, int64_t max_ctx,
                    std::string& err) {
    std::string base = dir;
    {
        std::ifstream probe(base + "/config.json");
        if (!probe) base = dir + "/dspark";
    }
    std::ifstream cf(base + "/config.json");
    if (!cf) {
        err = "dspark: no config.json in " + dir;
        return false;
    }
    const std::string cfg((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
    auto p = std::make_unique<Impl>();
    Mm3DraftGeo& g = p->g;
    double v = 0;
    auto num = [&](const char* key, double dflt) {
        return cfg_num(cfg, key, v) ? v : dflt;
    };
    g.dim = (int) num("hidden_size", 0);
    g.vocab = (int) num("vocab_size", 0);
    g.n_layers = (int) num("num_hidden_layers", 0);
    g.n_head = (int) num("num_attention_heads", 0);
    g.n_kv = (int) num("num_key_value_heads", 0);
    g.hd = (int) num("head_dim", 0);
    g.n_rot = (int) num("rotary_dim", g.hd * num("partial_rotary_factor", 1.0));
    g.eps = (float) num("rms_norm_eps", 1e-6);
    g.theta = (float) num("rope_theta", 0);
    g.alpha = (float) num("swiglu_alpha", 0);
    g.limit = (float) num("swiglu_limit", 0);
    g.ff = (int) num("dspark_ffn_hidden_size", 0);
    g.block = (int) num("dspark_block_size", 0);
    g.rank = (int) num("dspark_markov_rank", 0);
    g.mask_id = (int) num("dspark_noise_token_id", 0);
    g.window = (int) num("sliding_window", 4096);
    cfg_ints(cfg, "dspark_target_layer_ids", g.taps);
    if (g.dim <= 0 || g.vocab <= 0 || g.n_layers <= 0 || g.hd <= 0 || g.ff <= 0 || g.block <= 0 ||
        g.block > 16 || g.rank <= 0 || g.taps.empty() || g.n_head % std::max(g.n_kv, 1) != 0) {
        err = "dspark: the config in " + base + " has no DSpark draft this engine covers";
        return false;
    }
    const int D = g.dim, V = g.vocab, R = g.rank, NT = (int) g.taps.size();
    const int Q = g.n_head * g.hd, KV = g.n_kv * g.hd, F = g.ff;

    StFile st;
    if (!st.open(base + "/model-dspark-00001.safetensors", err)) return false;
    auto get = [&](const std::string& name) -> const StTensor* {
        const auto it = st.tensors.find(name);
        return it == st.tensors.end() ? nullptr : &it->second;
    };

    p->backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!p->backend) {
        err = "dspark: no GPU backend";
        return false;
    }
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend* bufs[2] = {p->backend, cpu};
    p->sched = ggml_backend_sched_new(bufs, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
    ggml_init_params ip{16 * 1024 * 1024, nullptr, true};
    p->wctx = ggml_init(ip);
    p->emb = emb;
    p->head = head;
    p->max_ctx = max_ctx;

    auto bf16 = [&](const std::string& name, int64_t ne0, int64_t ne1, ggml_tensor*& out) -> bool {
        const StTensor* t = get(name);
        if (!t || t->dtype != "BF16" || st.size(*t) != (size_t) ne0 * ne1 * 2) {
            err = "dspark: " + name + ": not the BF16 tensor expected";
            return false;
        }
        out = p->new_wt(GGML_TYPE_F32, ne0, ne1);   // F32 storage: this ggml's Metal library has
        return true;                                // no BF16 x BF16 mul_mat kernel (it aborts)
    };
    auto mx = [&](const std::string& name, int64_t N, int64_t K, ggml_tensor*& out) -> bool {
        const StTensor* w = get(name + ".weight");
        const StTensor* s = get(name + ".weight_scale_inv");
        if (!w || !s || w->dtype != "F8_E4M3" || st.size(*w) != (size_t) N * K ||
            st.size(*s) != (size_t) N * (K / 32)) {
            err = "dspark: " + name + ": not the MXFP8 matrix expected";
            return false;
        }
        out = p->new_wt(GGML_TYPE_F32, K, N);
        return true;
    };
    // norm vectors: F32 tensors holding (stored + 1); also returned as host floats when asked
    auto norm = [&](const std::string& name, int64_t n, ggml_tensor*& out, std::vector<float>* host) -> bool {
        const StTensor* t = get(name);
        if (!t || t->dtype != "BF16" || st.size(*t) != (size_t) n * 2) {
            err = "dspark: " + name + ": not the BF16 vector expected";
            return false;
        }
        out = p->new_wt(GGML_TYPE_F32, n, 1);
        if (host) host->assign((size_t) n, 0.0f);
        return true;
    };

    bool ok = bf16(DP + "fc.weight", (int64_t) NT * D, D, p->fc) &&
              norm(DP + "hidden_norm.weight", D, p->hid1_t, &p->hid1) &&
              norm(DP + "final_norm.weight", D, p->fin1_t, &p->fin1) &&
              bf16(DP + "markov_head.markov_w1.weight", R, V, p->w1) &&
              bf16(DP + "markov_head.markov_w2.weight", R, V, p->w2);
    p->layers.resize((size_t) g.n_layers);
    for (int l = 0; ok && l < g.n_layers; ++l) {
        const std::string P = DP + "layers." + std::to_string(l) + ".decoder_layer.";
        Impl::L& L = p->layers[(size_t) l];
        ok = mx(P + "self_attn.q_proj", Q, D, L.q) && mx(P + "self_attn.k_proj", KV, D, L.k) &&
             mx(P + "self_attn.v_proj", KV, D, L.v) && mx(P + "self_attn.o_proj", D, Q, L.o) &&
             mx(P + "mlp.gate_proj", F, D, L.gt) && mx(P + "mlp.up_proj", F, D, L.up) &&
             mx(P + "mlp.down_proj", D, F, L.dn) &&
             norm(P + "input_layernorm.weight", D, L.in1, nullptr) &&
             norm(P + "post_attention_layernorm.weight", D, L.post1, nullptr) &&
             norm(P + "self_attn.q_norm.weight", g.hd, L.qn1, nullptr) &&
             norm(P + "self_attn.k_norm.weight", g.hd, L.kn1, nullptr);
        if (ok) {
            L.kc = p->new_wt(GGML_TYPE_BF16, KV, max_ctx);
            L.vc = p->new_wt(GGML_TYPE_BF16, KV, max_ctx);
        }
    }
    if (ok) p->ctx_acc = p->new_wt(GGML_TYPE_F32, D, kChunk);
    if (!ok) return false;

    p->wbuf = ggml_backend_alloc_ctx_tensors(p->wctx, p->backend);
    if (!p->wbuf) {
        err = "dspark: cannot allocate the weights";
        return false;
    }
    // uploads: BF16 widened to F32; MXFP8 expanded (exact in F32); norms F32 with the +1 applied
    auto up_bf16 = [&](const std::string& name, ggml_tensor* t) {
        const uint16_t* b = (const uint16_t*) st.data(*get(name));
        std::vector<float> f((size_t) t->ne[0] * t->ne[1]);
        for (size_t i = 0; i < f.size(); ++i) f[i] = bf16_f32(b[i]);
        ggml_backend_tensor_set(t, f.data(), 0, f.size() * 4);
    };
    auto up_mx = [&](const std::string& name, ggml_tensor* t) {
        const StTensor* w = get(name + ".weight");
        const StTensor* s = get(name + ".weight_scale_inv");
        const uint8_t* wd = st.data(*w);
        const uint8_t* sd = st.data(*s);
        const int64_t N = t->ne[1], K = t->ne[0];
        std::vector<float> out((size_t) N * K);
        for (int64_t n = 0; n < N; ++n)
            for (int64_t k = 0; k < K; ++k) {
                const float sc = std::ldexp(1.0f, (int) sd[(size_t) n * (K / 32) + k / 32] - 127);
                out[(size_t) n * K + k] = e4m3(wd[(size_t) n * K + k]) * sc;
            }
        ggml_backend_tensor_set(t, out.data(), 0, out.size() * 4);
    };
    auto up_norm = [&](const std::string& name, ggml_tensor* t, std::vector<float>* host) {
        const StTensor* s = get(name);
        const uint16_t* b = (const uint16_t*) st.data(*s);
        std::vector<float> f((size_t) t->ne[0]);
        for (size_t i = 0; i < f.size(); ++i) f[i] = bf16_f32(b[i]) + 1.0f;
        ggml_backend_tensor_set(t, f.data(), 0, f.size() * 4);
        if (host) *host = f;
    };
    up_bf16(DP + "fc.weight", p->fc);
    up_bf16(DP + "markov_head.markov_w1.weight", p->w1);
    up_bf16(DP + "markov_head.markov_w2.weight", p->w2);
    up_norm(DP + "hidden_norm.weight", p->hid1_t, &p->hid1);
    up_norm(DP + "final_norm.weight", p->fin1_t, &p->fin1);
    for (int l = 0; l < g.n_layers; ++l) {
        const std::string P = DP + "layers." + std::to_string(l) + ".decoder_layer.";
        Impl::L& L = p->layers[(size_t) l];
        up_mx(P + "self_attn.q_proj", L.q);
        up_mx(P + "self_attn.k_proj", L.k);
        up_mx(P + "self_attn.v_proj", L.v);
        up_mx(P + "self_attn.o_proj", L.o);
        up_mx(P + "mlp.gate_proj", L.gt);
        up_mx(P + "mlp.up_proj", L.up);
        up_mx(P + "mlp.down_proj", L.dn);
        up_norm(P + "input_layernorm.weight", L.in1, nullptr);
        up_norm(P + "post_attention_layernorm.weight", L.post1, nullptr);
        up_norm(P + "self_attn.q_norm.weight", L.qn1, nullptr);
        up_norm(P + "self_attn.k_norm.weight", L.kn1, nullptr);
    }
    {   // the confidence head: a host-side dot per draft (D + R floats) + its bias
        const StTensor* w = get(DP + "confidence_head.proj.weight");
        const StTensor* b = get(DP + "confidence_head.proj.bias");
        if (!w || w->dtype != "BF16" || st.size(*w) != (size_t) (D + R) * 2 || !b ||
            st.size(*b) != 2) {
            err = "dspark: the confidence head is not the BF16 row expected";
            return false;
        }
        const uint16_t* wb = (const uint16_t*) st.data(*w);
        p->cw.assign((size_t) D + R, 0.0f);
        for (size_t i = 0; i < p->cw.size(); ++i) p->cw[i] = bf16_f32(wb[i]);
        p->cb = bf16_f32(*(const uint16_t*) st.data(*b));
    }
    this->p = std::move(p);
    return true;
}

bool Mm3Draft::context(const float* taps, int T, int64_t pos0, std::string& err) {
    Impl& p = *this->p;
    const Mm3DraftGeo& g = p.g;
    const int D = g.dim, KV = g.n_kv * g.hd, NT = (int) g.taps.size();
    if (pos0 + T > p.max_ctx) {
        err = "dspark: the context runs past the cache";
        return false;
    }
    for (int a = 0; a < T; a += (int) kChunk) {
        const int n = (int) std::min<int64_t>(kChunk, T - a);
        {   // ctx_acc[:, :n] = sum_j fc_j . taps_j (the CUDA engine rounds the taps to BF16
            // first; here the whole draft runs F32 - see the loader's note on kernels)
            Impl::Graph gr;
            ggml_tensor* acc = nullptr;
            for (int j = 0; j < NT; ++j) {
                const float* rows = taps + ((size_t) j * T + a) * D;
                ggml_tensor* t = gr.in(GGML_TYPE_F32, D, n,
                                       std::vector<uint8_t>((const uint8_t*) rows,
                                                            (const uint8_t*) rows + (size_t) n * D * 4));
                ggml_tensor* fcj = ggml_view_2d(gr.ctx, p.fc, D, D, p.fc->nb[1],
                                                (size_t) j * D * ggml_type_size(GGML_TYPE_F32));
                ggml_tensor* part = ggml_mul_mat(gr.ctx, fcj, t);
                acc = acc ? ggml_add(gr.ctx, acc, part) : part;
            }
            ggml_tensor* dest = ggml_view_2d(gr.ctx, p.ctx_acc, D, n, p.ctx_acc->nb[1], 0);
            ggml_build_forward_expand(gr.gf, ggml_cpy(gr.ctx, acc, dest));
            if (!p.compute(gr.ctx, gr.gf, gr.up, err)) return false;
        }
        {   // the chunk's K/V into every layer's cache at pos0 + a
            Impl::Graph gr;
            ggml_tensor* accv = ggml_view_2d(gr.ctx, p.ctx_acc, D, n, p.ctx_acc->nb[1], 0);
            ggml_tensor* xn = p.rms(gr.ctx, accv, p.hid1_t);
            std::vector<int32_t> pv((size_t) n);
            for (int i = 0; i < n; ++i) pv[(size_t) i] = (int32_t) (pos0 + a + i);
            ggml_tensor* pos = gr.in1(GGML_TYPE_I32, n,
                                      std::vector<uint8_t>((const uint8_t*) pv.data(),
                                                           (const uint8_t*) pv.data() + (size_t) n * 4));
            for (const Impl::L& L : p.layers) {
                ggml_tensor* K = ggml_mul_mat(gr.ctx, L.k, xn);   // [KV][n]
                ggml_tensor* V = ggml_mul_mat(gr.ctx, L.v, xn);
                ggml_tensor* k3 = ggml_reshape_3d(gr.ctx, K, g.hd, g.n_kv, n);
                k3 = p.rope(gr.ctx, p.head_norm(gr.ctx, k3, L.kn1), pos);
                ggml_tensor* kd = ggml_view_2d(gr.ctx, L.kc, KV, n, L.kc->nb[1],
                                               (size_t) (pos0 + a) * KV * ggml_type_size(GGML_TYPE_BF16));
                ggml_build_forward_expand(gr.gf, ggml_cpy(gr.ctx, ggml_reshape_2d(gr.ctx, k3, KV, n), kd));
                ggml_tensor* vd = ggml_view_2d(gr.ctx, L.vc, KV, n, L.vc->nb[1],
                                               (size_t) (pos0 + a) * KV * ggml_type_size(GGML_TYPE_BF16));
                ggml_build_forward_expand(gr.gf, ggml_cpy(gr.ctx, V, vd));
            }
            if (!p.compute(gr.ctx, gr.gf, gr.up, err)) return false;
        }
    }
    p.hi = std::max(p.hi, pos0 + T);
    return true;
}

bool Mm3Draft::propose(int32_t anchor, int64_t pos, int n, std::vector<int32_t>& drafts,
                       std::vector<float>& confs, std::string& err) {
    Impl& p = *this->p;
    const Mm3DraftGeo& g = p.g;
    const int D = g.dim, V = g.vocab, R = g.rank, B = g.block;
    const int Q = g.n_head * g.hd, KV = g.n_kv * g.hd, G = g.n_head / g.n_kv;
    drafts.clear();
    confs.clear();
    if (pos + B > p.max_ctx || pos < 0) return true;   // no room: no drafts, no error
    n = std::max(1, std::min(n, B));
    const int64_t hi = pos + B - 1, lo = std::max<int64_t>(0, hi - g.window + 1);
    const int64_t Twin = hi - lo + 1;
    ggml_tensor* logits_t = nullptr;
    ggml_tensor* hid_t = nullptr;
    {
        Impl::Graph gr;
        std::vector<int32_t> ids((size_t) B, g.mask_id);
        ids[0] = anchor;
        ggml_tensor* idt = gr.in1(GGML_TYPE_I32, B,
                                  std::vector<uint8_t>((const uint8_t*) ids.data(),
                                                       (const uint8_t*) ids.data() + (size_t) B * 4));
        std::vector<int32_t> pv((size_t) B);
        for (int i = 0; i < B; ++i) pv[(size_t) i] = (int32_t) (pos + i);
        ggml_tensor* post = gr.in1(GGML_TYPE_I32, B,
                                   std::vector<uint8_t>((const uint8_t*) pv.data(),
                                                        (const uint8_t*) pv.data() + (size_t) B * 4));
        ggml_tensor* h = ggml_cpy(gr.ctx, ggml_get_rows(gr.ctx, p.emb, idt),
                                  ggml_new_tensor_2d(gr.ctx, GGML_TYPE_F32, D, B));
        const float scale = 1.0f / std::sqrt((float) g.hd);
        for (const Impl::L& L : p.layers) {
            ggml_tensor* xn = p.rms(gr.ctx, h, L.in1);
            ggml_tensor* q = ggml_mul_mat(gr.ctx, L.q, xn);
            ggml_tensor* k = ggml_mul_mat(gr.ctx, L.k, xn);
            ggml_tensor* v = ggml_mul_mat(gr.ctx, L.v, xn);
            ggml_tensor* q3 = ggml_reshape_3d(gr.ctx, q, g.hd, g.n_head, B);
            q3 = p.rope(gr.ctx, p.head_norm(gr.ctx, q3, L.qn1), post);
            ggml_tensor* k3 = ggml_reshape_3d(gr.ctx, k, g.hd, g.n_kv, B);
            k3 = p.rope(gr.ctx, p.head_norm(gr.ctx, k3, L.kn1), post);
            ggml_tensor* kd = ggml_view_2d(gr.ctx, L.kc, KV, B, L.kc->nb[1],
                                           (size_t) pos * KV * ggml_type_size(GGML_TYPE_BF16));
            // the attention below must read the caches THROUGH these cpy results: a view of
            // the base tensor carries no dependency on the write (llama.cpp's KV pattern)
            ggml_tensor* kwr = ggml_cpy(gr.ctx, ggml_reshape_2d(gr.ctx, k3, KV, B), kd);
            ggml_build_forward_expand(gr.gf, kwr);
            ggml_tensor* vd = ggml_view_2d(gr.ctx, L.vc, KV, B, L.vc->nb[1],
                                           (size_t) pos * KV * ggml_type_size(GGML_TYPE_BF16));
            ggml_tensor* vwr = ggml_cpy(gr.ctx, v, vd);
            ggml_build_forward_expand(gr.gf, vwr);
            std::vector<ggml_tensor*> groups;
            for (int gi = 0; gi < g.n_kv; ++gi) {
                ggml_tensor* kw = ggml_view_2d(gr.ctx, kwr, g.hd, Twin, L.kc->nb[1],
                                               (size_t) ((lo - pos) * KV + gi * g.hd) * ggml_type_size(GGML_TYPE_BF16));
                ggml_tensor* kf = ggml_cpy(gr.ctx, kw, ggml_new_tensor_2d(gr.ctx, GGML_TYPE_F32, g.hd, Twin));
                ggml_tensor* vw = ggml_view_2d(gr.ctx, vwr, g.hd, Twin, L.vc->nb[1],
                                               (size_t) ((lo - pos) * KV + gi * g.hd) * ggml_type_size(GGML_TYPE_BF16));
                ggml_tensor* vf = ggml_cpy(gr.ctx, vw, ggml_new_tensor_2d(gr.ctx, GGML_TYPE_F32, g.hd, Twin));
                ggml_tensor* vt = ggml_cont(gr.ctx, ggml_transpose(gr.ctx, vf));
                ggml_tensor* qg = ggml_view_3d(gr.ctx, q3, g.hd, G, B, q3->nb[1], q3->nb[2],
                                               (size_t) gi * G * q3->nb[1]);
                ggml_tensor* sc = ggml_mul_mat(gr.ctx, kf, qg);   // [Twin][G][B]
                sc = ggml_soft_max(gr.ctx, ggml_scale(gr.ctx, sc, scale));
                groups.push_back(ggml_mul_mat(gr.ctx, vt, sc));   // [hd][G][B]
            }
            ggml_tensor* att = groups[0];
            for (size_t i = 1; i < groups.size(); ++i) att = ggml_concat(gr.ctx, att, groups[i], 1);
            att = ggml_cont(gr.ctx, ggml_reshape_2d(gr.ctx, att, Q, B));
            h = ggml_add(gr.ctx, h, ggml_mul_mat(gr.ctx, L.o, att));
            ggml_tensor* xn2 = p.rms(gr.ctx, h, L.post1);
            ggml_tensor* act = ggml_swiglu_oai(gr.ctx, ggml_mul_mat(gr.ctx, L.gt, xn2),
                                               ggml_mul_mat(gr.ctx, L.up, xn2), g.alpha, g.limit);
            h = ggml_add(gr.ctx, h, ggml_mul_mat(gr.ctx, L.dn, act));
        }
        hid_t = p.rms(gr.ctx, h, p.fin1_t);
        ggml_set_output(hid_t);
        ggml_build_forward_expand(gr.gf, hid_t);
        ggml_tensor* hidb = ggml_cpy(gr.ctx, hid_t, ggml_new_tensor_2d(gr.ctx, GGML_TYPE_BF16, D, B));
        ggml_tensor* hidf = ggml_cpy(gr.ctx, hidb, ggml_new_tensor_2d(gr.ctx, GGML_TYPE_F32, D, B));
        logits_t = ggml_mul_mat(gr.ctx, p.head, hidf);
        ggml_set_output(logits_t);
        ggml_build_forward_expand(gr.gf, logits_t);
        if (!p.compute(gr.ctx, gr.gf, gr.up, err)) return false;
        p.last_hid.assign((size_t) B * D, 0.0f);
        p.last_logits.assign((size_t) B * V, 0.0f);
        std::vector<float> hb((size_t) B * D), lb((size_t) B * V);
        ggml_backend_tensor_get(hid_t, hb.data(), 0, hb.size() * 4);
        ggml_backend_tensor_get(logits_t, lb.data(), 0, lb.size() * 4);
        for (int i = 0; i < B; ++i)
            for (int d = 0; d < D; ++d) p.last_hid[(size_t) i * D + d] = hb[(size_t) i * D + d];
        for (int i = 0; i < B; ++i)
            for (int vv = 0; vv < V; ++vv) p.last_logits[(size_t) i * V + vv] = lb[(size_t) i * V + vv];
    }
    // the Markov chain on the host; the bias of a row is one small GEMV graph
    int32_t prev = anchor;
    for (int i = 0; i < n; ++i) {
        Impl::Graph gm;
        ggml_tensor* pt = gm.in1(GGML_TYPE_I32, 1, std::vector<uint8_t>((const uint8_t*) &prev,
                                                                        (const uint8_t*) &prev + 4));
        ggml_tensor* e = ggml_get_rows(gm.ctx, p.w1, pt);          // [R][1] BF16
        ggml_tensor* ef = ggml_cpy(gm.ctx, e, ggml_new_tensor_2d(gm.ctx, GGML_TYPE_F32, R, 1));
        ggml_set_output(ef);
        ggml_build_forward_expand(gm.gf, ef);
        ggml_tensor* bias = ggml_mul_mat(gm.ctx, p.w2, ef);        // [V][1]
        ggml_set_output(bias);
        ggml_build_forward_expand(gm.gf, bias);
        if (!p.compute(gm.ctx, gm.gf, gm.up, err)) return false;
        std::vector<float> bh((size_t) V), eh((size_t) R);
        ggml_backend_tensor_get(bias, bh.data(), 0, bh.size() * 4);
        ggml_backend_tensor_get(ef, eh.data(), 0, eh.size() * 4);
        const float* lg = &p.last_logits[(size_t) i * V];
        int32_t pick = 0;
        float best = -INFINITY;
        for (int vv = 0; vv < V; ++vv) {
            const float s = lg[vv] + bh[(size_t) vv];
            if (s > best) {
                best = s;
                pick = vv;
            }
        }
        double dot = p.cb;
        const float* hd = &p.last_hid[(size_t) i * D];
        for (int d = 0; d < D; ++d) dot += (double) p.cw[(size_t) d] * hd[d];
        for (int r = 0; r < R; ++r) dot += (double) p.cw[(size_t) D + r] * eh[(size_t) r];
        drafts.push_back(pick);
        confs.push_back((float) (1.0 / (1.0 + std::exp(-dot))));
        prev = pick;
    }
    return true;
}

Mm3Draft::Mm3Draft() = default;
Mm3Draft::~Mm3Draft() = default;
const Mm3DraftGeo& Mm3Draft::geo() const { return p->g; }
int64_t Mm3Draft::max_ctx() const { return p->max_ctx; }
void Mm3Draft::reset_context() { p->hi = 0; }
int64_t Mm3Draft::context_hi() const { return p->hi; }
const std::vector<float>& Mm3Draft::last_hid() const { return p->last_hid; }
const std::vector<float>& Mm3Draft::last_logits() const { return p->last_logits; }

}  // namespace maya::metal
