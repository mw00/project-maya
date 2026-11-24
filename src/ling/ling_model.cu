// src/ling/ling_model.cu - Ling-3.2-air (bailingmoe3) loader, decode (CUDA graph / GPU cache / CPU lane),
// double-buffered batched prefill (KDA + MLA WMMA + grouped MoE), multi-GPU layer split, and optional MTP block.
#include "ling_model.hpp"

#include "ling_kernels.cuh"
#include "ds4_iq.cuh"
#include "hy4_kernels.cuh"
#include "maya/kernels/glm_batch.hpp"
#include "maya/kernels/glmd_batch.hpp"
#include "maya/kernels/glmd_dec.hpp"
#include "maya/artifact/gguf_reader.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess)                                                                                         \
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_) + " at " #x);                     \
    } while (0)

#define CKBLAS(x)                                                                                                      \
    do {                                                                                                               \
        cublasStatus_t s_ = (x);                                                                                       \
        if (s_ != CUBLAS_STATUS_SUCCESS)                                                                               \
            throw std::runtime_error("cuBLAS error " + std::to_string((int) s_) + " at " #x);                         \
    } while (0)

namespace ling {

namespace iq = maya::ds4::gguf::iq;
namespace hy = hy4::k;
namespace glmb = maya::kernels::glmb;

static inline size_t align256(size_t n) { return (n + 255u) & ~size_t(255u); }

// ------------------------------ metadata ------------------------------

static double req_num(const maya::GgufFile& g, const std::string& key) {
    const maya::MetaValue* v = g.get(key);
    if (v == nullptr || !v->is_num()) throw std::runtime_error("GGUF: missing key " + key);
    return v->num();
}

static double opt_num(const maya::GgufFile& g, const std::string& key, double dflt) {
    const maya::MetaValue* v = g.get(key);
    return v != nullptr && v->is_num() ? v->num() : dflt;
}

static std::vector<double> get_array(const maya::GgufFile& g, const std::string& key, bool required) {
    maya::MetaValue v;
    if (!g.get_full(key, v)) {
        if (required) throw std::runtime_error("GGUF: missing key " + key);
        return {};
    }
    std::vector<double> out;
    out.reserve((size_t) v.count);
    for (const auto& e : v.items) out.push_back(e.num());
    if ((int64_t) out.size() != (int64_t) v.count) throw std::runtime_error("GGUF: could not read all of " + key);
    return out;
}

static Config parse_config(const maya::GgufFile& g) {
    const std::string a = "bailingmoe3.";
    Config c;
    c.n_layer_all = (int) req_num(g, a + "block_count");
    c.n_nextn = (int) opt_num(g, a + "nextn_predict_layers", 0);
    c.n_layers = c.n_layer_all - c.n_nextn;
    if (c.n_layers < 1 || c.n_layers > 64) throw std::runtime_error("GGUF: implausible block_count");
    c.n_embd = (int) req_num(g, a + "embedding_length");
    c.n_head = (int) req_num(g, a + "attention.head_count");
    const std::vector<double> kv = get_array(g, a + "attention.head_count_kv", true);
    if ((int) kv.size() != c.n_layer_all) throw std::runtime_error("GGUF: head_count_kv not per block");
    for (int il = 0; il < c.n_layers; ++il) c.kda[il] = kv[(size_t) il] == 0.0;
    c.eps = (float) req_num(g, a + "attention.layer_norm_rms_epsilon");
    c.kv_lora = (int) req_num(g, a + "attention.kv_lora_rank");
    c.qk_head = (int) req_num(g, a + "attention.key_length_mla");
    c.v_head = (int) req_num(g, a + "attention.value_length_mla");
    c.n_rot = (int) req_num(g, a + "rope.dimension_count");
    c.qk_nope = c.qk_head - c.n_rot;
    c.rope_base = (float) req_num(g, a + "rope.freq_base");
    if ((int) req_num(g, a + "attention.key_length") != c.kv_lora + c.n_rot)
        throw std::runtime_error("GGUF: attention.key_length != kv_lora_rank + rope dims");
    c.d_conv = (int) req_num(g, a + "ssm.conv_kernel");
    c.kda_head = (int) req_num(g, a + "kda.head_dim");
    c.kda_lb = (float) req_num(g, a + "kda.gate_lower_bound");
    if (opt_num(g, a + "kda.safe_gate", 1) == 0.0 || c.kda_lb >= 0.0f)
        throw std::runtime_error("GGUF: bailingmoe3 needs the safe KDA gate with a negative lower bound");
    c.n_expert = (int) req_num(g, a + "expert_count");
    c.n_used = (int) req_num(g, a + "expert_used_count");
    c.n_group = (int) opt_num(g, a + "expert_group_count", 0);
    c.n_group_used = (int) opt_num(g, a + "expert_group_used_count", 0);
    if ((int) req_num(g, a + "expert_gating_func") != 2)
        throw std::runtime_error("GGUF: only sigmoid gating (expert_gating_func 2) is implemented");
    c.n_ff = (int) req_num(g, a + "feed_forward_length");
    c.n_ff_exp = (int) req_num(g, a + "expert_feed_forward_length");
    c.n_ff_shexp = (int) opt_num(g, a + "expert_shared_feed_forward_length", 0);
    if (c.n_ff_shexp == 0)
        c.n_ff_shexp = c.n_ff_exp * std::max(1, (int) req_num(g, a + "expert_shared_count"));
    c.n_dense_lead = (int) req_num(g, a + "leading_dense_block_count");
    c.w_scale = (float) opt_num(g, a + "expert_weights_scale", 0.0);
    c.w_norm = opt_num(g, a + "expert_weights_norm", 0) != 0.0;
    c.eos_id = (int) opt_num(g, "tokenizer.ggml.eos_token_id", 156895);
    const std::vector<double> ce = get_array(g, a + "swiglu_clamp_exp", false);
    const std::vector<double> cs = get_array(g, a + "swiglu_clamp_shexp", false);
    if ((!ce.empty() && (int) ce.size() != c.n_layer_all) || (!cs.empty() && (int) cs.size() != c.n_layer_all))
        throw std::runtime_error("GGUF: swiglu_clamp_* not per block");
    for (int il = 0; il < c.n_layer_all && il < 64; ++il) {
        c.clamp_exp[il] = ce.empty() ? 0.0f : (float) ce[(size_t) il];
        c.clamp_shexp[il] = cs.empty() ? 0.0f : (float) cs[(size_t) il];
    }
    return c;
}

static size_t tensor_bytes(const maya::TensorInfo& t) {
    int be = 0, bb = 0;
    if (!maya::block_geometry(t.type, be, bb)) throw std::runtime_error("GGUF: unknown type in " + t.name);
    const uint64_t n = t.elements();
    if (n % (uint64_t) be != 0) throw std::runtime_error("GGUF: ragged tensor " + t.name);
    return (size_t) (n / (uint64_t) be) * (size_t) bb;
}

static void expect_shape(const maya::TensorInfo& t, std::initializer_list<int64_t> ne) {
    size_t i = 0;
    for (int64_t d : ne) {
        const int64_t have = i < t.shape.size() ? (int64_t) t.shape[i] : 1;
        if (d >= 0 && have != d) throw std::runtime_error("GGUF: " + t.name + ": unexpected shape");
        ++i;
    }
}

static const std::unordered_map<uint32_t, uint8_t>& gpt2_unicode_to_byte() {
    static const std::unordered_map<uint32_t, uint8_t> map = [] {
        std::unordered_map<uint32_t, uint8_t> m;
        std::vector<int> bs;
        for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
        for (int b = 0xa1; b <= 0xac; ++b) bs.push_back(b);
        for (int b = 0xae; b <= 0xff; ++b) bs.push_back(b);
        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                ++n;
            }
        }
        for (size_t i = 0; i < bs.size(); ++i) m[(uint32_t) cs[i]] = (uint8_t) bs[i];
        return m;
    }();
    return map;
}

static std::string decode_gpt2_token(const std::string& tok) {
    const auto& u2b = gpt2_unicode_to_byte();
    std::string out;
    out.reserve(tok.size());
    const unsigned char* s = (const unsigned char*) tok.data();
    size_t i = 0, n = tok.size();
    while (i < n) {
        uint32_t cp = 0;
        size_t len = 1;
        if ((s[i] & 0x80u) == 0) {
            cp = s[i];
        } else if ((s[i] & 0xe0u) == 0xc0u && i + 1 < n) {
            cp = ((s[i] & 0x1fu) << 6) | (s[i + 1] & 0x3fu);
            len = 2;
        } else if ((s[i] & 0xf0u) == 0xe0u && i + 2 < n) {
            cp = ((s[i] & 0x0fu) << 12) | ((s[i + 1] & 0x3fu) << 6) | (s[i + 2] & 0x3fu);
            len = 3;
        } else if ((s[i] & 0xf8u) == 0xf0u && i + 3 < n) {
            cp = ((s[i] & 0x07u) << 18) | ((s[i + 1] & 0x3fu) << 12) | ((s[i + 2] & 0x3fu) << 6) | (s[i + 3] & 0x3fu);
            len = 4;
        }
        auto it = u2b.find(cp);
        if (it != u2b.end()) out.push_back((char) it->second);
        else out.append(tok, i, len);
        i += len;
    }
    return out;
}

// ------------------------------ helpers ------------------------------

void Model::set_dev(int dev) { CK(cudaSetDevice(dev)); }

void* Model::dev_alloc(Gpu& g, size_t bytes) {
    set_dev(g.dev);
    void* p = nullptr;
    CK(cudaMalloc(&p, std::max<size_t>(bytes, 16)));
    g.dev_allocs.push_back(p);
    return p;
}

void* Model::host_alloc(Gpu& g, size_t bytes) {
    set_dev(g.dev);
    void* p = nullptr;
    CK(cudaMallocHost(&p, std::max<size_t>(bytes, 16)));
    g.host_allocs.push_back(p);
    return p;
}

void Model::load_layer_weights(Layer& l, int il, Gpu& g, const maya::GgufFile& gf,
                               const std::map<std::string, const maya::TensorInfo*>& want) {
    set_dev(g.dev);
    auto at = [&](const std::string& n) -> const maya::TensorInfo& { return *want.at(n); };
    auto dev_copy = [&](const maya::TensorInfo& t) -> const uint8_t* {
        const size_t bytes = tensor_bytes(t);
        void* p = dev_alloc(g, bytes);
        CK(cudaMemcpy(p, gf.tensor_data(t), bytes, cudaMemcpyHostToDevice));
        return (const uint8_t*) p;
    };
    auto dev_float = [&](const maya::TensorInfo& t) -> const float* {
        if (t.type != 0) throw std::runtime_error("GGUF: " + t.name + ": F32 expected");
        return (const float*) dev_copy(t);
    };
    auto dev_mat = [&](const maya::TensorInfo& t, int k, int n) -> Mat {
        expect_shape(t, {k, n});
        if (!iq::mv_supported((int) t.type))
            throw std::runtime_error("GGUF: " + t.name + ": type " + std::string(t.type_name()) + " not supported");
        Mat m;
        m.p = dev_copy(t);
        m.type = (int) t.type;
        m.k = k;
        m.n = n;
        m.ld = iq::row_bytes(m.type, k);
        return m;
    };

    const Config& c = c_;
    const int E = c.n_embd, H = c.n_head, DI = c.d_inner(), KL = c.kv_lora, R = c.n_rot;
    const std::string b = "blk." + std::to_string(il) + ".";

    l.attn_norm = dev_float(at(b + "attn_norm.weight"));
    l.ffn_norm = dev_float(at(b + "ffn_norm.weight"));

    if (l.kda) {
        l.wq = dev_mat(at(b + "attn_q.weight"), E, DI);
        l.wk = dev_mat(at(b + "attn_k.weight"), E, DI);
        l.wv = dev_mat(at(b + "attn_v.weight"), E, DI);
        l.wo = dev_mat(at(b + "attn_output.weight"), DI, E);
        l.wf = dev_mat(at(b + "ssm_f_a.weight"), E, DI);
        l.wbeta = dev_mat(at(b + "ssm_beta.weight"), E, H);
        l.wg = dev_mat(at(b + "ssm_g_a.weight"), E, DI);
        l.conv_q = dev_float(at(b + "ssm_conv1d_q.weight"));
        l.conv_k = dev_float(at(b + "ssm_conv1d_k.weight"));
        l.conv_v = dev_float(at(b + "ssm_conv1d_v.weight"));
        l.dt_bias = dev_float(at(b + "ssm_dt.bias"));
        l.ssm_norm = dev_float(at(b + "ssm_norm.weight"));
        const maya::TensorInfo& ta = at(b + "ssm_a");
        l.ssm_a = dev_float(ta);
        std::vector<float> a_neg(ta.elements());
        std::memcpy(a_neg.data(), gf.tensor_data(ta), a_neg.size() * sizeof(float));
        for (float& v : a_neg) v = -v;
        void* pa = dev_alloc(g, a_neg.size() * sizeof(float));
        CK(cudaMemcpy(pa, a_neg.data(), a_neg.size() * sizeof(float), cudaMemcpyHostToDevice));
        l.ssm_a_neg = (const float*) pa;

        const size_t cs = (size_t) 3 * (size_t) (c.d_conv - 1) * (size_t) DI * sizeof(float);
        l.conv_state = (float*) dev_alloc(g, cs);
        CK(cudaMemset(l.conv_state, 0, cs));
        const size_t st = (size_t) H * (size_t) c.kda_head * (size_t) c.kda_head * sizeof(float);
        l.state = (float*) dev_alloc(g, st);
        CK(cudaMemset(l.state, 0, st));
    } else {
        l.wqm = dev_mat(at(b + "attn_q.weight"), E, H * c.qk_head);
        l.wkva = dev_mat(at(b + "attn_kv_a_mqa.weight"), E, KL + R);
        l.kv_a_norm = dev_float(at(b + "attn_kv_a_norm.weight"));
        const maya::TensorInfo& tk = at(b + "attn_k_b.weight");
        const maya::TensorInfo& tv = at(b + "attn_v_b.weight");
        expect_shape(tk, {c.qk_nope, KL, H});
        expect_shape(tv, {KL, c.v_head, H});
        l.wkb = dev_mat(tk, c.qk_nope, KL);
        l.wvb = dev_mat(tv, KL, c.v_head);
        l.wgate = dev_mat(at(b + "attn_gate.weight"), E, H);
        l.wom = dev_mat(at(b + "attn_output.weight"), H * c.v_head, E);
        const size_t lat_row_bytes = o_.kv_q8 ? (size_t) KL + (size_t) (KL / 32) * 2 : (size_t) KL * sizeof(uint16_t);
        l.lat = dev_alloc(g, (size_t) o_.ctx * lat_row_bytes);
        l.k_pe = (uint16_t*) dev_alloc(g, (size_t) o_.ctx * (size_t) R * sizeof(uint16_t));
    }

    if (il < c.n_dense_lead) {
        l.fg = dev_mat(at(b + "ffn_gate.weight"), E, c.n_ff);
        l.fu = dev_mat(at(b + "ffn_up.weight"), E, c.n_ff);
        l.fd = dev_mat(at(b + "ffn_down.weight"), c.n_ff, E);
    } else {
        const maya::TensorInfo& tr = at(b + "ffn_gate_inp.weight");
        expect_shape(tr, {E, c.n_expert});
        if (tr.type != 0 && tr.type != 30) throw std::runtime_error("GGUF: " + tr.name + ": F32 or BF16 expected");
        l.router_w = dev_copy(tr);
        l.router_type = (int) tr.type;
        const maya::TensorInfo& tb = at(b + "exp_probs_b.bias");
        expect_shape(tb, {c.n_expert});
        l.rb_dev = dev_float(tb);
        l.fg = dev_mat(at(b + "ffn_gate_shexp.weight"), E, c.n_ff_shexp);
        l.fu = dev_mat(at(b + "ffn_up_shexp.weight"), E, c.n_ff_shexp);
        l.fd = dev_mat(at(b + "ffn_down_shexp.weight"), c.n_ff_shexp, E);
    }
}

// ------------------------------ constructor ------------------------------

Model::Model(const std::string& path, const Options& opt) : o_(opt) {
    if (o_.gpus.empty()) o_.gpus = {0};
    maya::GgufFile gf(path);   // unmapped on return once tensors are in VRAM / host_blobs
    const maya::MetaValue* arch = gf.get("general.architecture");
    if (arch == nullptr || arch->s != "bailingmoe3")
        throw std::runtime_error("GGUF: not a bailingmoe3 file (" + path + ")");
    c_ = parse_config(gf);
    const Config& c = c_;

    {
        maya::MetaValue vt;
        if (gf.get_full("tokenizer.ggml.tokens", vt) && !vt.items.empty()) {
            vocab_tokens_.reserve(vt.items.size());
            for (const auto& it : vt.items) vocab_tokens_.push_back(it.s);
        }
    }

    std::map<std::string, const maya::TensorInfo*> want;
    auto add = [&](const std::string& n) { want[n] = nullptr; };
    add("token_embd.weight");
    add("output_norm.weight");
    add("output.weight");
    for (int il = 0; il < c.n_layers; ++il) {
        const std::string b = "blk." + std::to_string(il) + ".";
        add(b + "attn_norm.weight");
        add(b + "ffn_norm.weight");
        if (c.kda[il]) {
            for (const char* n : {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                                  "ssm_conv1d_q.weight", "ssm_conv1d_k.weight", "ssm_conv1d_v.weight",
                                  "ssm_f_a.weight", "ssm_dt.bias", "ssm_a", "ssm_beta.weight", "ssm_g_a.weight",
                                  "ssm_norm.weight"})
                add(b + n);
        } else {
            for (const char* n : {"attn_q.weight", "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight",
                                  "attn_k_b.weight", "attn_v_b.weight", "attn_gate.weight", "attn_output.weight"})
                add(b + n);
        }
        if (il < c.n_dense_lead) {
            for (const char* n : {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}) add(b + n);
        } else {
            for (const char* n : {"ffn_gate_inp.weight", "exp_probs_b.bias", "ffn_gate_exps.weight",
                                  "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_shexp.weight",
                                  "ffn_up_shexp.weight", "ffn_down_shexp.weight"})
                add(b + n);
        }
    }

    const std::string mtp_prefix = "blk." + std::to_string(c.n_layers) + ".";
    std::map<std::string, const maya::TensorInfo*> mtp_want;
    if (o_.load_mtp && c.n_nextn >= 1) {
        for (const char* n : {"attn_norm.weight", "ffn_norm.weight", "attn_q.weight", "attn_kv_a_mqa.weight",
                              "attn_kv_a_norm.weight", "attn_k_b.weight", "attn_v_b.weight", "attn_gate.weight",
                              "attn_output.weight", "ffn_gate_inp.weight", "exp_probs_b.bias", "ffn_gate_exps.weight",
                              "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_shexp.weight",
                              "ffn_up_shexp.weight", "ffn_down_shexp.weight", "nextn.enorm.weight",
                              "nextn.hnorm.weight", "nextn.eh_proj.weight", "layer_output_norm.weight"})
            mtp_want[mtp_prefix + n] = nullptr;
    }

    int mtp_count = 0;
    for (const auto& t : gf.tensors()) {
        auto it = want.find(t.name);
        if (it != want.end()) {
            it->second = &t;
            continue;
        }
        if (t.name.compare(0, mtp_prefix.size(), mtp_prefix) == 0) {
            ++mtp_count;
            auto mit = mtp_want.find(t.name);
            if (mit != mtp_want.end()) mit->second = &t;
            continue;
        }
        throw std::runtime_error("GGUF: unexpected tensor " + t.name);
    }
    for (const auto& kv : want)
        if (kv.second == nullptr) throw std::runtime_error("GGUF: missing tensor " + kv.first);
    if (o_.load_mtp && c.n_nextn >= 1) {
        for (const auto& kv : mtp_want)
            if (kv.second == nullptr) throw std::runtime_error("GGUF: missing MTP tensor " + kv.first);
    }

    const int ngpu = (int) o_.gpus.size();
    gpus_.resize((size_t) ngpu);
    const int n_moe = std::max(0, c.n_layers - c.n_dense_lead);
    for (int gi = 0; gi < ngpu; ++gi) {
        Gpu& gp = gpus_[(size_t) gi];
        gp.dev = o_.gpus[(size_t) gi];
        set_dev(gp.dev);
        cudaSetDeviceFlags(cudaDeviceMapHost);
        cudaGetLastError();
        CK(cudaStreamCreateWithFlags(&gp.s, cudaStreamNonBlocking));
        CK(cudaStreamCreateWithFlags(&gp.s_copy, cudaStreamNonBlocking));
        CKBLAS(cublasCreate(&gp.blas));
        CKBLAS(cublasSetStream(gp.blas, gp.s));
        CK(cudaEventCreateWithFlags(&gp.ev_xfer, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&gp.ev_route, cudaEventDisableTiming));
        for (int b = 0; b < 2; ++b) {
            CK(cudaEventCreateWithFlags(&gp.ev_copy[b], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&gp.ev_comp[b], cudaEventDisableTiming));
        }
        const int m0 = (n_moe * gi) / ngpu;
        const int m1 = (n_moe * (gi + 1)) / ngpu;
        gp.l0 = gi == 0 ? 0 : (c.n_dense_lead + m0);
        gp.l1 = c.n_dense_lead + m1;
    }

    p2p_ok_.assign((size_t) std::max(0, ngpu - 1), false);
    for (int gi = 0; gi + 1 < ngpu; ++gi) {
        const int d0 = gpus_[(size_t) gi].dev, d1 = gpus_[(size_t) (gi + 1)].dev;
        int can01 = 0, can10 = 0;
        cudaDeviceCanAccessPeer(&can01, d0, d1);
        cudaDeviceCanAccessPeer(&can10, d1, d0);
        if (can01 && can10) {
            set_dev(d0);
            cudaError_t e0 = cudaDeviceEnablePeerAccess(d1, 0);
            if (e0 == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
            set_dev(d1);
            cudaError_t e1 = cudaDeviceEnablePeerAccess(d0, 0);
            if (e1 == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
            if ((e0 == cudaSuccess || e0 == cudaErrorPeerAccessAlreadyEnabled) &&
                (e1 == cudaSuccess || e1 == cudaErrorPeerAccessAlreadyEnabled))
                p2p_ok_[(size_t) gi] = true;
        }
    }

    const int E = c.n_embd, H = c.n_head, DI = c.d_inner(), KL = c.kv_lora, R = c.n_rot, FE = c.n_ff_exp;

    {
        const maya::TensorInfo& te = *want.at("token_embd.weight");
        expect_shape(te, {E, -1});
        if (te.type != 0 && te.type != 1 && te.type != 8 && te.type != 30)
            throw std::runtime_error("GGUF: token_embd type not supported");
        c_.vocab = (int) te.shape[1];
        set_dev(gpus_[0].dev);
        void* pe = dev_alloc(gpus_[0], tensor_bytes(te));
        CK(cudaMemcpy(pe, gf.tensor_data(te), tensor_bytes(te), cudaMemcpyHostToDevice));
        emb_ = {(const uint8_t*) pe, (int) te.type, E, c_.vocab, 0};

        Gpu& glast = gpus_.back();
        set_dev(glast.dev);
        const maya::TensorInfo& ton = *want.at("output_norm.weight");
        if (ton.type != 0) throw std::runtime_error("GGUF: output_norm F32 expected");
        void* pon = dev_alloc(glast, tensor_bytes(ton));
        CK(cudaMemcpy(pon, gf.tensor_data(ton), tensor_bytes(ton), cudaMemcpyHostToDevice));
        out_norm_ = (const float*) pon;

        const maya::TensorInfo& tow = *want.at("output.weight");
        expect_shape(tow, {E, c_.vocab});
        if (!iq::mv_supported((int) tow.type))
            throw std::runtime_error("GGUF: output.weight type not supported");
        void* pow = dev_alloc(glast, tensor_bytes(tow));
        CK(cudaMemcpy(pow, gf.tensor_data(tow), tensor_bytes(tow), cudaMemcpyHostToDevice));
        out_w_ = {(const uint8_t*) pow, (int) tow.type, E, c_.vocab, iq::row_bytes((int) tow.type, E)};
    }

    layers_.resize((size_t) c.n_layers);
    for (int gi = 0; gi < ngpu; ++gi) {
        Gpu& gp = gpus_[(size_t) gi];
        for (int il = gp.l0; il < gp.l1; ++il) {
            Layer& l = layers_[(size_t) il];
            l.dev_idx = gi;
            l.kda = c.kda[il];
            load_layer_weights(l, il, gp, gf, want);
        }
    }

    max_blob_ = 0;
    for (int il = c.n_dense_lead; il < c.n_layers; ++il) {
        Layer& l = layers_[(size_t) il];
        const std::string b = "blk." + std::to_string(il) + ".";
        const maya::TensorInfo& tg = *want.at(b + "ffn_gate_exps.weight");
        const maya::TensorInfo& tu = *want.at(b + "ffn_up_exps.weight");
        const maya::TensorInfo& td = *want.at(b + "ffn_down_exps.weight");
        expect_shape(tg, {E, FE, c.n_expert});
        expect_shape(tu, {E, FE, c.n_expert});
        expect_shape(td, {FE, E, c.n_expert});
        if (tg.type != tu.type)
            throw std::runtime_error("GGUF: " + b + "gate and up expert types differ");
        if (!iq::expert_supported((int) tg.type) || !iq::down_supported((int) td.type))
            throw std::runtime_error("GGUF: " + b + "expert type not supported");
        l.etg = (int) tg.type;
        l.etu = (int) tu.type;
        l.etd = (int) td.type;
        l.eg_ld = iq::row_bytes(l.etg, E);
        l.eu_ld = iq::row_bytes(l.etu, E);
        l.ed_ld = iq::row_bytes(l.etd, FE);
        const size_t eg_st = l.eg_ld * (size_t) FE;
        const size_t eu_st = l.eu_ld * (size_t) FE;
        const size_t ed_st = l.ed_ld * (size_t) E;
        l.tr[0] = 0;
        l.tr[1] = align256(eg_st);
        l.tr[2] = align256(l.tr[1] + eu_st);
        l.blob = align256(l.tr[2] + ed_st);
        max_blob_ = std::max(max_blob_, l.blob);
    }

    if (o_.load_mtp && c.n_nextn >= 1 && !mtp_want.empty()) {
        Gpu& glast = gpus_.back();
        set_dev(glast.dev);
        auto at = [&](const std::string& n) -> const maya::TensorInfo& { return *mtp_want.at(n); };
        auto dev_copy = [&](const maya::TensorInfo& t) -> const uint8_t* {
            void* p = dev_alloc(glast, tensor_bytes(t));
            CK(cudaMemcpy(p, gf.tensor_data(t), tensor_bytes(t), cudaMemcpyHostToDevice));
            return (const uint8_t*) p;
        };
        mtp_.enorm = (const float*) dev_copy(at(mtp_prefix + "nextn.enorm.weight"));
        mtp_.hnorm = (const float*) dev_copy(at(mtp_prefix + "nextn.hnorm.weight"));
        mtp_.out_norm = (const float*) dev_copy(at(mtp_prefix + "layer_output_norm.weight"));
        const maya::TensorInfo& teh = at(mtp_prefix + "nextn.eh_proj.weight");
        expect_shape(teh, {2 * E, E});
        mtp_.eh_proj = {dev_copy(teh), (int) teh.type, 2 * E, E, iq::row_bytes((int) teh.type, 2 * E)};
        mtp_.layer.dev_idx = ngpu - 1;
        mtp_.layer.kda = false;
        load_layer_weights(mtp_.layer, c.n_layers, glast, gf, mtp_want);
        const maya::TensorInfo& tg = at(mtp_prefix + "ffn_gate_exps.weight");
        const maya::TensorInfo& tu = at(mtp_prefix + "ffn_up_exps.weight");
        const maya::TensorInfo& td = at(mtp_prefix + "ffn_down_exps.weight");
        mtp_.layer.etg = (int) tg.type;
        mtp_.layer.etu = (int) tu.type;
        mtp_.layer.etd = (int) td.type;
        mtp_.layer.eg_ld = iq::row_bytes(mtp_.layer.etg, E);
        mtp_.layer.eu_ld = iq::row_bytes(mtp_.layer.etu, E);
        mtp_.layer.ed_ld = iq::row_bytes(mtp_.layer.etd, FE);
        const size_t eg_st = mtp_.layer.eg_ld * (size_t) FE;
        const size_t eu_st = mtp_.layer.eu_ld * (size_t) FE;
        const size_t ed_st = mtp_.layer.ed_ld * (size_t) E;
        mtp_.layer.tr[0] = 0;
        mtp_.layer.tr[1] = align256(eg_st);
        mtp_.layer.tr[2] = align256(mtp_.layer.tr[1] + eu_st);
        mtp_.layer.blob = align256(mtp_.layer.tr[2] + ed_st);
        max_blob_ = std::max(max_blob_, mtp_.layer.blob);
        mtp_.emb_buf = (float*) dev_alloc(glast, (size_t) E * 4);
        mtp_.eh_q = (int8_t*) dev_alloc(glast, (size_t) 2 * E);
        mtp_.eh_d = (float*) dev_alloc(glast, (size_t) (2 * E / 32) * 4);
        mtp_loaded_ = true;
    }

    // Allocate decode + double-buffered prefill scratch
    const int P = std::max(1, std::min(o_.prefill_chunk, o_.ctx));
    const size_t mla_part_b = glmd::mla_dec_part_bytes(H);
    const size_t max_w16_elems = std::max<size_t>(
        {(size_t) H * c.qk_head * (size_t) E, (size_t) c.n_ff * (size_t) E, (size_t) H * (size_t) KL * c.qk_nope,
         (size_t) H * (size_t) c.v_head * (size_t) KL, (size_t) DI * (size_t) E});
    for (int gi = 0; gi < ngpu; ++gi) {
        Gpu& gp = gpus_[(size_t) gi];
        set_dev(gp.dev);
        gp.x = (float*) dev_alloc(gp, (size_t) E * 4);
        gp.cur = (float*) dev_alloc(gp, (size_t) E * 4);
        gp.out = (float*) dev_alloc(gp, (size_t) E * 4);
        gp.hn = (float*) dev_alloc(gp, (size_t) E * 4);
        gp.xq = (int8_t*) dev_alloc(gp, (size_t) E);
        gp.xd = (float*) dev_alloc(gp, (size_t) (E / 32) * 4);
        gp.xq2 = (int8_t*) dev_alloc(gp, (size_t) std::max(DI, H * c.v_head));
        gp.xd2 = (float*) dev_alloc(gp, (size_t) (std::max(DI, H * c.v_head) / 32) * 4);
        gp.q = (float*) dev_alloc(gp, (size_t) DI * 4);
        gp.k = (float*) dev_alloc(gp, (size_t) DI * 4);
        gp.v = (float*) dev_alloc(gp, (size_t) DI * 4);
        gp.qc = (float*) dev_alloc(gp, (size_t) DI * 4);
        gp.kc = (float*) dev_alloc(gp, (size_t) DI * 4);
        gp.vc = (float*) dev_alloc(gp, (size_t) DI * 4);
        gp.f = (float*) dev_alloc(gp, (size_t) DI * 4);
        gp.g2 = (float*) dev_alloc(gp, (size_t) DI * 4);
        gp.g1 = (float*) dev_alloc(gp, (size_t) DI * 4);
        gp.beta = (float*) dev_alloc(gp, (size_t) H * 4);
        gp.raw_o = (float*) dev_alloc(gp, (size_t) DI * 4);
        gp.qm = (float*) dev_alloc(gp, (size_t) H * c.qk_head * 4);
        gp.kvall = (float*) dev_alloc(gp, (size_t) (KL + R) * 4);
        gp.q_abs = (float*) dev_alloc(gp, (size_t) H * KL * 4);
        gp.ctxo = (float*) dev_alloc(gp, (size_t) H * KL * 4);
        gp.o128 = (float*) dev_alloc(gp, (size_t) H * c.v_head * 4);
        gp.gate = (float*) dev_alloc(gp, (size_t) H * 4);
        gp.q8n = (int8_t*) dev_alloc(gp, (size_t) H * c.qk_nope);
        gp.d8n = (float*) dev_alloc(gp, (size_t) H * (c.qk_nope / 32) * 4);
        gp.q8c = (int8_t*) dev_alloc(gp, (size_t) H * KL);
        gp.d8c = (float*) dev_alloc(gp, (size_t) H * (KL / 32) * 4);
        gp.mla_part = (float*) dev_alloc(gp, mla_part_b);
        gp.gu = (float*) dev_alloc(gp, (size_t) std::max(2 * c.n_ff, 2 * c.n_ff_shexp) * 4);
        gp.eh = (float*) dev_alloc(gp, (size_t) std::max(c.n_used * FE, c.n_ff) * 4);
        gp.shact = (float*) dev_alloc(gp, (size_t) c.n_ff_shexp * 4);
        gp.part = (float*) dev_alloc(gp, (size_t) c.n_used * E * 4);
        gp.shout = (float*) dev_alloc(gp, (size_t) E * 4);
        gp.moe = (float*) dev_alloc(gp, (size_t) E * 4);
        gp.rlog = (float*) dev_alloc(gp, (size_t) c.n_expert * 4);
        gp.d_w = (float*) dev_alloc(gp, (size_t) 16 * 4);
        gp.d_ids = (int*) dev_alloc(gp, (size_t) 16 * 4);
        gp.d_plan = (unsigned long long*) dev_alloc(gp, (size_t) 16 * sizeof(unsigned long long));
        gp.d_cpu_out = (float*) dev_alloc(gp, (size_t) E * 4);
        gp.ehq = (int8_t*) dev_alloc(gp, (size_t) std::max(c.n_used * FE, c.n_ff));
        gp.ehd = (float*) dev_alloc(gp, (size_t) (std::max(c.n_used * FE, c.n_ff) / 32) * 4);
        gp.shq = (int8_t*) dev_alloc(gp, (size_t) c.n_ff_shexp);
        gp.shd = (float*) dev_alloc(gp, (size_t) (c.n_ff_shexp / 32) * 4);
        if (gi == ngpu - 1) {
            gp.logits = (float*) dev_alloc(gp, (size_t) c_.vocab * 4);
            gp.d_top = (float*) dev_alloc(gp, 2 * sizeof(float));
        }
        gp.d_step = (int*) dev_alloc(gp, 3 * sizeof(int));
        gp.d_iota = (int*) dev_alloc(gp, (size_t) o_.ctx * sizeof(int));
        k::fill_iota(gp.d_iota, o_.ctx, gp.s);

        gp.fxs.eptr = (unsigned long long*) dev_alloc(gp, (size_t) c.n_expert * sizeof(unsigned long long));
        gp.fxs.who = (int*) dev_alloc(gp, 16 * sizeof(int));
        gp.fxs.src = (unsigned long long*) dev_alloc(gp, 16 * sizeof(unsigned long long));
        gp.fxs.dst = (unsigned long long*) dev_alloc(gp, 16 * sizeof(unsigned long long));
        gp.fxs.nf = (unsigned*) dev_alloc(gp, sizeof(unsigned));
        gp.fxs.n_stage = c.n_used;
        gp.fxs.stage = (uint8_t*) dev_alloc(gp, (size_t) gp.fxs.n_stage * max_blob_);
        gp.fxs.stats = (unsigned long long*) dev_alloc(gp, 8 * sizeof(unsigned long long));
        CK(cudaMemset(gp.fxs.stats, 0, 8 * sizeof(unsigned long long)));

        gp.h_step = (int*) host_alloc(gp, 3 * sizeof(int));
        gp.h_ids = (int*) host_alloc(gp, 16 * sizeof(int));
        gp.h_w = (float*) host_alloc(gp, 16 * sizeof(float));
        gp.h_who = (int*) host_alloc(gp, 16 * sizeof(int));
        gp.h_cur = (float*) host_alloc(gp, (size_t) E * 4);
        gp.h_cpu_out = (float*) host_alloc(gp, (size_t) E * 4);
        gp.h_top = (float*) host_alloc(gp, 2 * sizeof(float));
        if (ngpu > 1) gp.h_xfer = (float*) host_alloc(gp, (size_t) P * (size_t) E * 4);

        if (P > 1) {
            gp.pf_cap = P;
            gp.pf_tok_ids = (int*) dev_alloc(gp, (size_t) P * sizeof(int));
            gp.pf_x = (float*) dev_alloc(gp, (size_t) P * E * 4);
            gp.pf_cur = (float*) dev_alloc(gp, (size_t) P * E * 4);
            gp.pf_out = (float*) dev_alloc(gp, (size_t) P * E * 4);
            gp.pf_moe = (float*) dev_alloc(gp, (size_t) P * E * 4);
            gp.pf_x16 = (uint16_t*) dev_alloc(gp, (size_t) P * std::max({DI, c.n_ff, H * c.v_head}) * sizeof(uint16_t));
            gp.pf_w16 = (uint16_t*) dev_alloc(gp, max_w16_elems * sizeof(uint16_t));
            const size_t kda_b = (size_t) P * ((size_t) 8 * DI + (size_t) H) * 4 + 4096;
            const size_t mla_b = (size_t) P * ((size_t) H * c.qk_head * 6 + (size_t) (KL + R) * 4 +
                                               (size_t) H * KL * 10 + (size_t) H * c.v_head * 4 + (size_t) H * 4) +
                                 4096;
            const size_t ffn_b = (size_t) P * (size_t) 2 * std::max(c.n_ff, c.n_ff_shexp) * 4 + 4096;
            const size_t moe_b = (size_t) P * ((size_t) c.n_expert * 4 + (size_t) c.n_used * FE * 2 +
                                               (size_t) c.n_used * E * 4) +
                                 4096;
            gp.pf_tmp_bytes = std::max({kda_b, mla_b, ffn_b, moe_b});
            gp.pf_tmp = (float*) dev_alloc(gp, gp.pf_tmp_bytes);
            gp.pf_ids = (int*) dev_alloc(gp, (size_t) P * c.n_used * sizeof(int));
            gp.pf_w = (float*) dev_alloc(gp, (size_t) P * c.n_used * sizeof(float));
            gp.pf_bounds = (int*) dev_alloc(gp, (size_t) (c.n_expert + 1) * sizeof(int));
            gp.pf_tok = (int*) dev_alloc(gp, (size_t) P * c.n_used * sizeof(int));
            gp.pf_row_of = (int*) dev_alloc(gp, (size_t) P * c.n_used * sizeof(int));
            gp.h_pf_ids = (int*) host_alloc(gp, (size_t) P * c.n_used * sizeof(int));
            gp.h_pf_w = (float*) host_alloc(gp, (size_t) P * c.n_used * sizeof(float));
            for (int b = 0; b < 2; ++b) {
                gp.pf_pmap[b] = (int*) dev_alloc(gp, (size_t) c.n_expert * sizeof(int));
                gp.pf_stage[b] = (uint8_t*) dev_alloc(gp, (size_t) kPfBatch * max_blob_);
                gp.h_pf_pmap[b] = (int*) host_alloc(gp, (size_t) c.n_expert * sizeof(int));
                gp.h_pf_stage[b] = (uint8_t*) host_alloc(gp, (size_t) kPfBatch * max_blob_);
            }
        }
        CK(cudaStreamSynchronize(gp.s));
    }

    // ---- Plan per-expert VRAM tier on each GPU ----
    std::vector<size_t> gpu_vram_used((size_t) ngpu, 0);
    size_t total_vram_experts = 0, total_experts = 0;
    for (int gi = 0; gi < ngpu; ++gi) {
        Gpu& gp = gpus_[(size_t) gi];
        set_dev(gp.dev);
        size_t free_b = 0, total_b = 0;
        CK(cudaMemGetInfo(&free_b, &total_b));
        const double headroom = 1.25 * (double) (1ULL << 30);
        double budget = o_.expert_vram_gib >= 0.0 ? o_.expert_vram_gib * (double) (1ULL << 30)
                                                  : std::max(0.0, (double) free_b - headroom);
        size_t gpu_moe_bytes = 0;
        for (int il = std::max(gp.l0, c.n_dense_lead); il < gp.l1; ++il)
            gpu_moe_bytes += layers_[(size_t) il].blob * (size_t) c.n_expert;
        const double frac = gpu_moe_bytes > 0 ? std::min(1.0, budget / (double) gpu_moe_bytes) : 0.0;
        for (int il = std::max(gp.l0, c.n_dense_lead); il < gp.l1; ++il) {
            Layer& l = layers_[(size_t) il];
            int nr = std::min(c.n_expert, (int) std::floor(frac * c.n_expert));
            while (nr > 0 && (double) nr * (double) l.blob > budget) --nr;
            l.n_res = nr;
            budget -= (double) nr * (double) l.blob;
            gpu_vram_used[(size_t) gi] += (size_t) nr * l.blob;
            total_vram_experts += (size_t) nr;
            total_experts += (size_t) c.n_expert;
        }
    }
    if (mtp_loaded_) mtp_.layer.n_res = 0;

    // ---- Allocate & pack experts in host RAM (THP ram_alloc; register pinned only when needed) ----
    struct PackTask {
        Layer* l;
        const maya::TensorInfo *tg, *tu, *td;
        int il;
    };
    std::vector<PackTask> tasks;
    for (int il = c.n_dense_lead; il < c.n_layers; ++il) {
        const std::string b = "blk." + std::to_string(il) + ".";
        tasks.push_back({&layers_[(size_t) il], want.at(b + "ffn_gate_exps.weight"), want.at(b + "ffn_up_exps.weight"),
                         want.at(b + "ffn_down_exps.weight"), il});
    }
    if (mtp_loaded_) {
        tasks.push_back({&mtp_.layer, mtp_want.at(mtp_prefix + "ffn_gate_exps.weight"),
                         mtp_want.at(mtp_prefix + "ffn_up_exps.weight"),
                         mtp_want.at(mtp_prefix + "ffn_down_exps.weight"), c.n_layers});
    }

    const bool want_pin = (o_.offload != OffloadMode::kCpu) && (o_.pin_ram_gib > 0.0);
    double pin_budget = want_pin ? o_.pin_ram_gib * (double) (1ULL << 30) : 0.0;
    size_t pin_bytes = 0, page_bytes = 0;

    // Allocate THP host memory first (instant via mmap + MADV_HUGEPAGE)
    for (auto& tk : tasks) {
        Layer& l = *tk.l;
        if (l.n_res == c.n_expert) continue;
        const size_t bytes = l.blob * (size_t) c.n_expert;
        l.host_bytes = bytes;
        l.host_blobs = (uint8_t*) hy4::ram_alloc(bytes);
        if (l.host_blobs == nullptr) throw std::runtime_error("out of host RAM packing experts");
    }

    // Pack GGUF gate/up/down into contiguous blobs across CPU threads (also faults THP pages)
    {
        const int nt = std::min<int>((int) tasks.size(), std::max(1, std::min(16, (int) std::thread::hardware_concurrency())));
        std::atomic<size_t> next{0};
        auto worker = [&] {
            while (true) {
                const size_t idx = next.fetch_add(1);
                if (idx >= tasks.size()) break;
                const PackTask& tk = tasks[idx];
                Layer& l = *tk.l;
                if (l.host_blobs == nullptr) continue;
                const uint8_t* sg = gf.tensor_data(*tk.tg);
                const uint8_t* su = gf.tensor_data(*tk.tu);
                const uint8_t* sd = gf.tensor_data(*tk.td);
                const size_t eg_st = l.eg_ld * (size_t) FE;
                const size_t eu_st = l.eu_ld * (size_t) FE;
                const size_t ed_st = l.ed_ld * (size_t) E;
                for (int e = 0; e < c.n_expert; ++e) {
                    uint8_t* dst = l.host_blobs + (size_t) e * l.blob;
                    std::memcpy(dst + l.tr[0], sg + (size_t) e * eg_st, eg_st);
                    std::memcpy(dst + l.tr[1], su + (size_t) e * eu_st, eu_st);
                    std::memcpy(dst + l.tr[2], sd + (size_t) e * ed_st, ed_st);
                }
            }
        };
        std::vector<std::thread> threads;
        for (int i = 0; i < nt; ++i) threads.emplace_back(worker);
        for (auto& th : threads) th.join();
    }

    // Pin packed THP buffers with cudaHostRegister (fast on 2 MiB huge pages) when GPU/mix offload wants UVA DMA
    for (auto& tk : tasks) {
        Layer& l = *tk.l;
        if (l.host_blobs == nullptr) continue;
        if (want_pin && (double) l.host_bytes <= pin_budget) {
            set_dev(gpus_[(size_t) l.dev_idx].dev);
            if (cudaHostRegister(l.host_blobs, l.host_bytes, cudaHostRegisterMapped | cudaHostRegisterPortable) == cudaSuccess) {
                l.host_pinned = true;
                l.host_registered = true;
                pin_budget -= (double) l.host_bytes;
                pin_bytes += l.host_bytes;
            } else {
                cudaGetLastError();
                pin_budget = 0.0;
                any_host_unpinned_ = true;
                page_bytes += l.host_bytes;
            }
        } else {
            any_host_unpinned_ = true;
            page_bytes += l.host_bytes;
        }
    }

    // Populate initial VRAM expert slots + FetchLayer tables + CpuExperts
    for (auto& tk : tasks) {
        Layer& l = *tk.l;
        Gpu& gp = gpus_[(size_t) l.dev_idx];
        set_dev(gp.dev);
        l.h_slot.assign((size_t) c.n_expert, -1);
        l.pf_counts.assign((size_t) c.n_expert, 0);
        std::vector<int> h_owner((size_t) std::max(1, l.n_res), -1);

        l.fl.n_res = l.n_res;
        l.fl.n_exp = c.n_expert;
        l.fl.blob = l.blob;
        l.fl.slot = (int*) dev_alloc(gp, (size_t) c.n_expert * sizeof(int));
        l.fl.owner = (int*) dev_alloc(gp, (size_t) std::max(1, l.n_res) * sizeof(int));
        l.fl.score = (float*) dev_alloc(gp, (size_t) c.n_expert * sizeof(float));
        CK(cudaMemset(l.fl.score, 0, (size_t) c.n_expert * sizeof(float)));

        if (l.n_res > 0) {
            l.fl.vram = (uint8_t*) dev_alloc(gp, (size_t) l.n_res * l.blob);
            if (l.host_blobs != nullptr) {
                CK(cudaMemcpyAsync(l.fl.vram, l.host_blobs, (size_t) l.n_res * l.blob, cudaMemcpyHostToDevice, gp.s));
            } else {
                const uint8_t* sg = gf.tensor_data(*tk.tg);
                const uint8_t* su = gf.tensor_data(*tk.tu);
                const uint8_t* sd = gf.tensor_data(*tk.td);
                const size_t eg_st = l.eg_ld * (size_t) FE;
                const size_t eu_st = l.eu_ld * (size_t) FE;
                const size_t ed_st = l.ed_ld * (size_t) E;
                std::vector<uint8_t> tmp(l.blob);
                for (int e = 0; e < l.n_res; ++e) {
                    std::memcpy(tmp.data() + l.tr[0], sg + (size_t) e * eg_st, eg_st);
                    std::memcpy(tmp.data() + l.tr[1], su + (size_t) e * eu_st, eu_st);
                    std::memcpy(tmp.data() + l.tr[2], sd + (size_t) e * ed_st, ed_st);
                    CK(cudaMemcpy(l.fl.vram + (size_t) e * l.blob, tmp.data(), l.blob, cudaMemcpyHostToDevice));
                }
            }
            for (int s = 0; s < l.n_res; ++s) {
                l.h_slot[(size_t) s] = s;
                h_owner[(size_t) s] = s;
            }
        }
        CK(cudaMemcpyAsync(l.fl.slot, l.h_slot.data(), (size_t) c.n_expert * sizeof(int), cudaMemcpyHostToDevice, gp.s));
        CK(cudaMemcpyAsync(l.fl.owner, h_owner.data(), (size_t) std::max(1, l.n_res) * sizeof(int),
                           cudaMemcpyHostToDevice, gp.s));

        std::vector<unsigned long long> hptr((size_t) c.n_expert, 0ull);
        if (l.host_blobs != nullptr && l.host_pinned) {
            void* dev_p = nullptr;
            CK(cudaHostGetDevicePointer(&dev_p, l.host_blobs, 0));
            const unsigned long long base = (unsigned long long) (uintptr_t) dev_p;
            for (int e = 0; e < c.n_expert; ++e) hptr[(size_t) e] = base + (unsigned long long) e * l.blob;
        }
        unsigned long long* d_hptr =
            (unsigned long long*) dev_alloc(gp, (size_t) c.n_expert * sizeof(unsigned long long));
        CK(cudaMemcpyAsync(d_hptr, hptr.data(), (size_t) c.n_expert * sizeof(unsigned long long),
                           cudaMemcpyHostToDevice, gp.s));
        l.fl.hptr = d_hptr;

        if (l.host_blobs != nullptr) {
            std::string err;
            if (!l.cex.init(l.etg, l.etd, E, FE, err))
                throw std::runtime_error("cpu lane init failed on layer " + std::to_string(tk.il) + ": " + err);
            l.cex.gate = l.host_blobs + l.tr[0];
            l.cex.up = l.host_blobs + l.tr[1];
            l.cex.down = l.host_blobs + l.tr[2];
            l.cex.step = l.blob;
            l.cex.limit = c.clamp_exp[tk.il] > 1e-6f ? -c.clamp_exp[tk.il] : 0.0f;
        }
    }
    for (auto& gp : gpus_) {
        set_dev(gp.dev);
        CK(cudaStreamSynchronize(gp.s));
    }

    if (o_.offload == OffloadMode::kAuto) {
        active_offload_ = any_host_unpinned_ ? OffloadMode::kCpu : OffloadMode::kGpu;
    } else if (o_.offload == OffloadMode::kGpu && any_host_unpinned_) {
        std::fprintf(stderr, "maya-ling: host experts exceed --pin-ram-gib, falling back to CPU lane offload\n");
        active_offload_ = OffloadMode::kCpu;
    } else {
        active_offload_ = o_.offload;
    }

    if (active_offload_ == OffloadMode::kCpu || active_offload_ == OffloadMode::kMix || any_host_unpinned_) {
        maya::ds4::gguf::iq_cpu_row_hint(E);
        lane_ = std::make_unique<hy4::Lane>(o_.cpu_threads);
    }

    const int n_kda = (int) std::count(c.kda, c.kda + c.n_layers, true);
    const char* mode_name = active_offload_ == OffloadMode::kGpu   ? "gpu-cache+graph"
                            : active_offload_ == OffloadMode::kMix ? "gpu-cache+cpu-lane"
                                                                   : "vram+cpu-lane";
    size_t vram_sum = std::accumulate(gpu_vram_used.begin(), gpu_vram_used.end(), size_t(0));
    char buf[1024];
    std::snprintf(buf, sizeof buf,
                  "bailingmoe3: n_embd %d, layers %d (+%d MTP %s; %d KDA / %d MLA), vocab %d, ctx %d (%s KV)\n"
                  "experts %d x top%d in %d groups (keep %d), dense lead %d, mla %d+%d/kv %d/v %d, kda %dx%d\n"
                  "gpus %d (mode: %s, graph %s, prefill_chunk %d): %zu/%zu expert slots in VRAM (%.1f%%, %.2f GiB), "
                  "host RAM %.2f GiB pinned + %.2f GiB paged",
                  c.n_embd, c.n_layers, mtp_count, mtp_loaded_ ? "loaded" : "parked", n_kda, c.n_layers - n_kda,
                  c_.vocab, o_.ctx, o_.kv_q8 ? "INT8" : "FP16", c.n_expert, c.n_used, c.n_group, c.n_group_used,
                  c.n_dense_lead, c.qk_nope, c.n_rot, c.kv_lora, c.v_head, H, c.kda_head, ngpu, mode_name,
                  (o_.use_graph && active_offload_ == OffloadMode::kGpu && o_.seam_dir.empty()) ? "on" : "off", P,
                  total_vram_experts, total_experts,
                  total_experts > 0 ? 100.0 * (double) total_vram_experts / (double) total_experts : 0.0,
                  (double) vram_sum / (double) (1ULL << 30), (double) pin_bytes / (double) (1ULL << 30),
                  (double) page_bytes / (double) (1ULL << 30));
    info_ = buf;
}

Model::~Model() {
    for (auto& gp : gpus_) {
        set_dev(gp.dev);
        cudaStreamSynchronize(gp.s);
        cudaStreamSynchronize(gp.s_copy);
        if (gp.graph_exec != nullptr) cudaGraphExecDestroy(gp.graph_exec);
        if (gp.graph != nullptr) cudaGraphDestroy(gp.graph);
    }
    auto free_layer_host = [&](Layer& l) {
        if (l.host_blobs == nullptr) return;
        if (l.host_registered) {
            set_dev(gpus_[(size_t) l.dev_idx].dev);
            cudaHostUnregister(l.host_blobs);
            hy4::ram_free(l.host_blobs, l.host_bytes);
        } else if (l.host_pinned) {
            cudaFreeHost(l.host_blobs);
        } else {
            hy4::ram_free(l.host_blobs, l.host_bytes);
        }
    };
    for (auto& l : layers_) free_layer_host(l);
    if (mtp_loaded_) free_layer_host(mtp_.layer);

    for (auto& gp : gpus_) {
        set_dev(gp.dev);
        for (void* p : gp.dev_allocs) cudaFree(p);
        for (void* p : gp.host_allocs) cudaFreeHost(p);
        for (int b = 0; b < 2; ++b) {
            if (gp.ev_copy[b] != nullptr) cudaEventDestroy(gp.ev_copy[b]);
            if (gp.ev_comp[b] != nullptr) cudaEventDestroy(gp.ev_comp[b]);
        }
        if (gp.ev_xfer != nullptr) cudaEventDestroy(gp.ev_xfer);
        if (gp.ev_route != nullptr) cudaEventDestroy(gp.ev_route);
        if (gp.blas != nullptr) cublasDestroy(gp.blas);
        if (gp.s_copy != nullptr) cudaStreamDestroy(gp.s_copy);
        if (gp.s != nullptr) cudaStreamDestroy(gp.s);
    }
}

void Model::reset() {
    const Config& c = c_;
    const int DI = c.d_inner(), H = c.n_head;
    const size_t cs = (size_t) 3 * (size_t) (c.d_conv - 1) * (size_t) DI * sizeof(float);
    const size_t st = (size_t) H * (size_t) c.kda_head * (size_t) c.kda_head * sizeof(float);
    for (auto& l : layers_) {
        Gpu& gp = gpus_[(size_t) l.dev_idx];
        set_dev(gp.dev);
        if (l.kda) {
            CK(cudaMemsetAsync(l.conv_state, 0, cs, gp.s));
            CK(cudaMemsetAsync(l.state, 0, st, gp.s));
        }
        std::fill(l.pf_counts.begin(), l.pf_counts.end(), 0);
    }
    for (auto& gp : gpus_) {
        set_dev(gp.dev);
        CK(cudaStreamSynchronize(gp.s));
    }
}

// ------------------------------ seam dumps ------------------------------

void Model::dump_seam_f32(Gpu& gp, const std::string& name, const float* dev, size_t n) {
    if (o_.seam_dir.empty()) return;
    set_dev(gp.dev);
    std::vector<float> tmp(n);
    CK(cudaMemcpyAsync(tmp.data(), dev, n * 4, cudaMemcpyDeviceToHost, gp.s));
    CK(cudaStreamSynchronize(gp.s));
    const std::string path = o_.seam_dir + "/" + name + ".f32";
    FILE* f = std::fopen(path.c_str(), "ab");
    if (f == nullptr) throw std::runtime_error("cannot open seam file " + path);
    std::fwrite(tmp.data(), 4, n, f);
    std::fclose(f);
}

void Model::dump_seam_i32(Gpu& gp, const std::string& name, const int* dev, size_t n) {
    if (o_.seam_dir.empty()) return;
    set_dev(gp.dev);
    std::vector<int> tmp(n);
    CK(cudaMemcpyAsync(tmp.data(), dev, n * 4, cudaMemcpyDeviceToHost, gp.s));
    CK(cudaStreamSynchronize(gp.s));
    const std::string path = o_.seam_dir + "/" + name + ".i32";
    FILE* f = std::fopen(path.c_str(), "ab");
    if (f == nullptr) throw std::runtime_error("cannot open seam file " + path);
    std::fwrite(tmp.data(), 4, n, f);
    std::fclose(f);
}

// ------------------------------ multi-GPU transfer ------------------------------

void Model::transfer_x(int g_from, int g_to, int T, bool prefill_buf) {
    Gpu& src = gpus_[(size_t) g_from];
    Gpu& dst = gpus_[(size_t) g_to];
    const size_t bytes = (size_t) T * (size_t) c_.n_embd * sizeof(float);
    float* p_src = prefill_buf ? src.pf_x : src.x;
    float* p_dst = prefill_buf ? dst.pf_x : dst.x;

    set_dev(src.dev);
    CK(cudaEventRecord(src.ev_xfer, src.s));
    set_dev(dst.dev);
    CK(cudaStreamWaitEvent(dst.s, src.ev_xfer, 0));
    if (p2p_ok_[(size_t) g_from]) {
        CK(cudaMemcpyPeerAsync(p_dst, dst.dev, p_src, src.dev, bytes, dst.s));
    } else {
        set_dev(src.dev);
        CK(cudaMemcpyAsync(src.h_xfer, p_src, bytes, cudaMemcpyDeviceToHost, src.s));
        CK(cudaEventRecord(src.ev_xfer, src.s));
        set_dev(dst.dev);
        CK(cudaStreamWaitEvent(dst.s, src.ev_xfer, 0));
        CK(cudaMemcpyAsync(p_dst, src.h_xfer, bytes, cudaMemcpyHostToDevice, dst.s));
    }
}

// ------------------------------ single-token decode ------------------------------

void Model::kda_decode(Gpu& gp, Layer& l, int il, bool seam) {
    const Config& c = c_;
    const int DI = c.d_inner(), H = c.n_head;
    cudaStream_t s = gp.s;
    iq::MvJob j[6];
    auto mk = [&](const Mat& m, float* y, const float* bias = nullptr) {
        return iq::MvJob{m.p, m.ld, m.type, m.k, m.n, 1, gp.xq, gp.xd, y, (size_t) m.n, bias, 1.0f};
    };
    j[0] = mk(l.wq, gp.q);
    j[1] = mk(l.wk, gp.k);
    j[2] = mk(l.wv, gp.v);
    j[3] = mk(l.wf, gp.f, l.dt_bias);
    j[4] = mk(l.wbeta, gp.beta);
    j[5] = mk(l.wg, gp.g2);
    if (!iq::mv(j, 6, s)) throw std::runtime_error("kda: type not supported");

    k::kda_prep_full(gp.q, gp.k, gp.v, gp.f, l.conv_q, l.conv_k, l.conv_v, l.conv_state, l.ssm_a, c.kda_lb, c.eps, H,
                     c.kda_head, c.d_conv, gp.qc, gp.kc, gp.vc, gp.g1, s);
    if (seam) dump_seam_f32(gp, "kda_gate-" + std::to_string(il), gp.g1, (size_t) DI);

    k::kda_rec_q8(gp.qc, gp.kc, gp.vc, gp.g1, gp.beta, l.state, gp.g2, l.ssm_norm, c.eps, H, c.kda_head, gp.xq2, gp.xd2,
                  seam ? gp.raw_o : nullptr, s);
    if (seam) dump_seam_f32(gp, "attn_output-" + std::to_string(il), gp.raw_o, (size_t) DI);

    iq::MvJob o{l.wo.p, l.wo.ld, l.wo.type, l.wo.k, l.wo.n, 1, gp.xq2, gp.xd2, gp.out, (size_t) l.wo.n, nullptr, 1.0f};
    if (!iq::mv(&o, 1, s)) throw std::runtime_error("kda: output type not supported");
    if (seam) dump_seam_f32(gp, "kda_out-" + std::to_string(il), gp.out, (size_t) c.n_embd);
    hy::add(gp.x, gp.out, (size_t) c.n_embd, s);
}

void Model::mla_decode(Gpu& gp, Layer& l, int il, bool seam) {
    const Config& c = c_;
    const int H = c.n_head, KL = c.kv_lora, R = c.n_rot;
    cudaStream_t s = gp.s;
    iq::MvJob j[3];
    auto mk = [&](const Mat& m, float* y) {
        return iq::MvJob{m.p, m.ld, m.type, m.k, m.n, 1, gp.xq, gp.xd, y, (size_t) m.n, nullptr, 1.0f};
    };
    j[0] = mk(l.wqm, gp.qm);
    j[1] = mk(l.wkva, gp.kvall);
    j[2] = mk(l.wgate, gp.gate);
    if (!iq::mv(j, 3, s)) throw std::runtime_error("mla: type not supported");

    k::mla_q_prep(gp.qm, 1, H, c.qk_head, c.qk_nope, R, gp.d_step + 1, 0, c.rope_base, gp.q8n, gp.d8n, nullptr, s);
    iq::MvJob kb{l.wkb.p, l.wkb.ld, l.wkb.type, c.qk_nope, KL, 1, gp.q8n, gp.d8n, gp.q_abs, (size_t) KL, nullptr, 1.0f};
    if (!iq::mv_heads(kb, H, (size_t) KL * l.wkb.ld, (size_t) c.qk_nope, (size_t) (c.qk_nope / 32), (size_t) KL, s))
        throw std::runtime_error("mla: k_b type not supported");

    k::kv_store_split(gp.kvall, 1, KL, R, l.kv_a_norm, c.eps, gp.d_step + 1, 0, c.rope_base, l.lat, o_.kv_q8, l.k_pe, s);

    if (!glmd::mla_dec(gp.q_abs, gp.qm, c.qk_head, c.qk_nope, l.lat, o_.kv_q8 ? 1 : 0, l.k_pe, gp.d_iota, o_.ctx,
                       gp.d_step + 2, H, KL, R, 1.0f / sqrtf((float) c.qk_head), gp.mla_part, gp.ctxo, s))
        throw std::runtime_error("mla: glmd::mla_dec refused shape");

    hy::q8_heads(gp.ctxo, (size_t) H * KL, (size_t) KL, 1, H, KL, gp.q8c, gp.d8c, s);
    iq::MvJob vb{l.wvb.p, l.wvb.ld, l.wvb.type, KL, c.v_head, 1, gp.q8c, gp.d8c, gp.o128, (size_t) c.v_head, nullptr,
                 1.0f};
    if (!iq::mv_heads(vb, H, (size_t) c.v_head * l.wvb.ld, (size_t) KL, (size_t) (KL / 32), (size_t) c.v_head, s))
        throw std::runtime_error("mla: v_b type not supported");
    if (seam) dump_seam_f32(gp, "kqv_out-" + std::to_string(il), gp.o128, (size_t) H * c.v_head);

    k::head_gate_out(gp.o128, gp.gate, 1, H, c.v_head, gp.xq2, gp.xd2, nullptr, s);
    iq::MvJob o{l.wom.p, l.wom.ld, l.wom.type, l.wom.k, l.wom.n, 1, gp.xq2, gp.xd2, gp.out, (size_t) l.wom.n, nullptr,
                1.0f};
    if (!iq::mv(&o, 1, s)) throw std::runtime_error("mla: output type not supported");
    if (seam) dump_seam_f32(gp, "mla_out-" + std::to_string(il), gp.out, (size_t) c.n_embd);
    hy::add(gp.x, gp.out, (size_t) c.n_embd, s);
}

void Model::ffn_dense_decode(Gpu& gp, Layer& l, int il, bool seam) {
    const Config& c = c_;
    cudaStream_t s = gp.s;
    iq::MvJob j[2] = {
        {l.fg.p, l.fg.ld, l.fg.type, l.fg.k, l.fg.n, 1, gp.xq, gp.xd, gp.gu, (size_t) l.fg.n, nullptr, 1.0f},
        {l.fu.p, l.fu.ld, l.fu.type, l.fu.k, l.fu.n, 1, gp.xq, gp.xd, gp.gu + c.n_ff, (size_t) l.fu.n, nullptr, 1.0f},
    };
    if (!iq::mv(j, 2, s)) throw std::runtime_error("ffn: type not supported");
    k::swiglu_clamp_out(gp.gu, gp.gu + c.n_ff, 1, c.n_ff, c.clamp_shexp[il], seam ? gp.eh : nullptr, gp.ehq, gp.ehd,
                        nullptr, s);
    iq::MvJob d{l.fd.p, l.fd.ld, l.fd.type, l.fd.k, l.fd.n, 1, gp.ehq, gp.ehd, gp.out, (size_t) l.fd.n, nullptr, 1.0f};
    if (!iq::mv(&d, 1, s)) throw std::runtime_error("ffn: down type not supported");
    hy::add(gp.x, gp.out, (size_t) c.n_embd, s);
}

void Model::ffn_moe_decode(Gpu& gp, Layer& l, int il, bool seam) {
    const Config& c = c_;
    const int E = c.n_embd, FE = c.n_ff_exp, FS = c.n_ff_shexp, NU = c.n_used;
    cudaStream_t s = gp.s;

    if (l.router_type == 0)
        hy::gemv_f32w((const float*) l.router_w, (size_t) E, E, c.n_expert, gp.cur, (size_t) E, 1, gp.rlog,
                      (size_t) c.n_expert, 1.0f, s);
    else
        hy::gemv_bf16w((const uint16_t*) l.router_w, (size_t) E, E, c.n_expert, gp.cur, (size_t) E, 1, gp.rlog,
                       (size_t) c.n_expert, 1.0f, s);

    k::router_grouped(gp.rlog, l.rb_dev, 1, c.n_expert, NU, c.n_group, c.n_group_used, c.w_norm, c.w_scale, gp.d_ids,
                      gp.d_w, s);
    if (seam) dump_seam_i32(gp, "ffn_moe_topk-" + std::to_string(il), gp.d_ids, (size_t) NU);

    auto run_shared = [&] {
        iq::MvJob sj[2] = {
            {l.fg.p, l.fg.ld, l.fg.type, E, FS, 1, gp.xq, gp.xd, gp.gu, (size_t) FS, nullptr, 1.0f},
            {l.fu.p, l.fu.ld, l.fu.type, E, FS, 1, gp.xq, gp.xd, gp.gu + FS, (size_t) FS, nullptr, 1.0f},
        };
        if (!iq::mv(sj, 2, s)) throw std::runtime_error("moe: shared gate/up type not supported");
        k::swiglu_clamp_out(gp.gu, gp.gu + FS, 1, FS, c.clamp_shexp[il], seam ? gp.shact : nullptr, gp.shq, gp.shd,
                            nullptr, s);
        if (seam && c.clamp_shexp[il] > 1e-6f)
            dump_seam_f32(gp, "ffn_swiglu_limited-" + std::to_string(il), gp.shact, (size_t) FS);
        iq::MvJob sd{l.fd.p, l.fd.ld, l.fd.type, FS, E, 1, gp.shq, gp.shd, gp.shout, (size_t) E, nullptr, 1.0f};
        if (!iq::mv(&sd, 1, s)) throw std::runtime_error("moe: shared down type not supported");
    };

    iq::Args a;
    a.plan_ptr = gp.d_plan;
    a.tr[0] = l.tr[0];
    a.tr[1] = l.tr[1];
    a.tr[2] = l.tr[2];
    a.n_embd = E;
    a.n_ff = FE;
    a.limit = c.clamp_exp[il] > 1e-6f ? -c.clamp_exp[il] : 0.0f;
    a.xq = gp.xq;
    a.xd = gp.xd;
    a.h = gp.eh;
    a.hq = gp.ehq;
    a.hd = gp.ehd;
    a.part = gp.part;
    a.tg = l.etg;
    a.td = l.etd;
    a.act = iq::kActSwiglu;

    if (l.n_res == c.n_expert) {
        hy::plan_experts(gp.d_ids, NU, l.fl.vram, l.blob, gp.d_plan, s);
        if (!iq::gate_up(a, NU, 1, s)) throw std::runtime_error("moe: gate_up failed");
        if (seam && c.clamp_exp[il] > 1e-6f)
            dump_seam_f32(gp, "ffn_moe_swiglu_limited-" + std::to_string(il), gp.eh, (size_t) NU * FE);
        hy::q8_rows(gp.eh, (size_t) FE, NU, FE, gp.ehq, gp.ehd, s);
        if (!iq::down(a, NU, 1, s)) throw std::runtime_error("moe: down failed");
        hy::moe_sum(gp.part, 1, NU, E, gp.d_w, gp.moe, s);
        run_shared();
        if (seam) {
            dump_seam_f32(gp, "ffn_moe_out-" + std::to_string(il), gp.moe, (size_t) E);
            dump_seam_f32(gp, "ffn_shexp-" + std::to_string(il), gp.shout, (size_t) E);
        }
        k::add2(gp.x, gp.moe, gp.shout, E, s);
        return;
    }

    const bool use_lane = (active_offload_ == OffloadMode::kCpu || active_offload_ == OffloadMode::kMix || !l.host_pinned);
    if (!use_lane) {
        hy::fetch_resolve(gp.d_ids, 1, NU, l.fl, gp.fxs, -1, l.n_res > 0, 0.98f, s);
        hy::fetch_copy(gp.fxs, l.blob, 32, s);
        hy::plan_experts(gp.d_ids, NU, nullptr, 0, gp.d_plan, s, nullptr, gp.d_w, gp.fxs.eptr, gp.fxs.who);
        if (!iq::gate_up(a, NU, 1, s)) throw std::runtime_error("moe: gate_up failed");
        if (seam && c.clamp_exp[il] > 1e-6f)
            dump_seam_f32(gp, "ffn_moe_swiglu_limited-" + std::to_string(il), gp.eh, (size_t) NU * FE);
        hy::q8_rows(gp.eh, (size_t) FE, NU, FE, gp.ehq, gp.ehd, s);
        if (!iq::down(a, NU, 1, s)) throw std::runtime_error("moe: down failed");
        hy::moe_sum(gp.part, 1, NU, E, gp.d_w, gp.moe, s);
        run_shared();
        if (seam) {
            dump_seam_f32(gp, "ffn_moe_out-" + std::to_string(il), gp.moe, (size_t) E);
            dump_seam_f32(gp, "ffn_shexp-" + std::to_string(il), gp.shout, (size_t) E);
        }
        k::add2(gp.x, gp.moe, gp.shout, E, s);
        return;
    }

    const int k_gpu = (active_offload_ == OffloadMode::kMix && l.host_pinned) ? o_.mix_k_gpu : 0;
    const bool adapt = (k_gpu > 0 && l.n_res > 0);
    hy::fetch_resolve(gp.d_ids, 1, NU, l.fl, gp.fxs, k_gpu, adapt, 0.98f, s);
    CK(cudaMemcpyAsync(gp.h_ids, gp.d_ids, (size_t) NU * sizeof(int), cudaMemcpyDeviceToHost, s));
    CK(cudaMemcpyAsync(gp.h_w, gp.d_w, (size_t) NU * sizeof(float), cudaMemcpyDeviceToHost, s));
    CK(cudaMemcpyAsync(gp.h_who, gp.fxs.who, (size_t) NU * sizeof(int), cudaMemcpyDeviceToHost, s));
    CK(cudaMemcpyAsync(gp.h_cur, gp.cur, (size_t) E * sizeof(float), cudaMemcpyDeviceToHost, s));
    CK(cudaEventRecord(gp.ev_route, s));

    if (k_gpu > 0) hy::fetch_copy(gp.fxs, l.blob, 32, s);
    hy::plan_experts(gp.d_ids, NU, nullptr, 0, gp.d_plan, s, nullptr, gp.d_w, gp.fxs.eptr, gp.fxs.who);
    if (!iq::gate_up(a, NU, 1, s)) throw std::runtime_error("moe: gate_up failed");
    hy::q8_rows(gp.eh, (size_t) FE, NU, FE, gp.ehq, gp.ehd, s);
    if (!iq::down(a, NU, 1, s)) throw std::runtime_error("moe: down failed");
    hy::moe_sum(gp.part, 1, NU, E, gp.d_w, gp.moe, s);
    run_shared();

    CK(cudaEventSynchronize(gp.ev_route));
    std::vector<hy4::CpuEntry> entries;
    entries.reserve((size_t) NU);
    for (int j2 = 0; j2 < NU; ++j2) {
        if (gp.h_who[j2] != 0) entries.push_back({0, gp.h_ids[j2], gp.h_w[j2]});
    }
    if (!entries.empty()) {
        hy4::cpu_experts(*lane_, l.cex, entries, gp.h_cur, 1, gp.h_cpu_out);
        cpu_lane_picks_ += (unsigned long long) entries.size();
        CK(cudaMemcpyAsync(gp.d_cpu_out, gp.h_cpu_out, (size_t) E * sizeof(float), cudaMemcpyHostToDevice, s));
        hy::add(gp.moe, gp.d_cpu_out, (size_t) E, s);
    }
    if (seam) {
        dump_seam_f32(gp, "ffn_moe_out-" + std::to_string(il), gp.moe, (size_t) E);
        dump_seam_f32(gp, "ffn_shexp-" + std::to_string(il), gp.shout, (size_t) E);
    }
    k::add2(gp.x, gp.moe, gp.shout, E, s);
}

void Model::decode_segment(int gi, bool capture_safe) {
    Gpu& gp = gpus_[(size_t) gi];
    const Config& c = c_;
    const int E = c.n_embd;
    const bool seam = !capture_safe && !o_.seam_dir.empty();
    cudaStream_t s = gp.s;

    if (gi == 0) {
        if (!k::emb_rows(emb_.p, emb_.type, emb_.k, gp.d_step, 1, gp.x, s))
            throw std::runtime_error("emb_rows: type not supported");
        if (seam) dump_seam_f32(gp, "model.input_embed", gp.x, (size_t) E);
    }

    for (int il = gp.l0; il < gp.l1; ++il) {
        Layer& l = layers_[(size_t) il];
        hy::rmsnorm(gp.x, (size_t) E, 1, E, l.attn_norm, c.eps, gp.cur, (size_t) E, s, {gp.xq, gp.xd});
        if (seam) dump_seam_f32(gp, "attn_norm-" + std::to_string(il), gp.cur, (size_t) E);
        if (l.kda) kda_decode(gp, l, il, seam);
        else mla_decode(gp, l, il, seam);
        if (seam) dump_seam_f32(gp, "ffn_inp-" + std::to_string(il), gp.x, (size_t) E);
        hy::rmsnorm(gp.x, (size_t) E, 1, E, l.ffn_norm, c.eps, gp.cur, (size_t) E, s, {gp.xq, gp.xd});
        if (il < c.n_dense_lead) ffn_dense_decode(gp, l, il, seam);
        else ffn_moe_decode(gp, l, il, seam);
        if (seam) dump_seam_f32(gp, "l_out-" + std::to_string(il), gp.x, (size_t) E);
    }

    if (gi == (int) gpus_.size() - 1) {
        hy::rmsnorm(gp.x, (size_t) E, 1, E, out_norm_, c.eps, gp.hn, (size_t) E, s, {gp.xq, gp.xd});
        if (seam) dump_seam_f32(gp, "result_norm", gp.hn, (size_t) E);
        iq::MvJob j{out_w_.p, out_w_.ld, out_w_.type, E, c.vocab, 1, gp.xq, gp.xd, gp.logits, (size_t) c.vocab, nullptr,
                    1.0f};
        if (!iq::mv(&j, 1, s)) throw std::runtime_error("output: type not supported");
    }
}

const float* Model::step(int token, int pos) {
    const int ngpu = (int) gpus_.size();
    const bool graph_ok = o_.use_graph && active_offload_ == OffloadMode::kGpu && !any_host_unpinned_ &&
                          o_.seam_dir.empty();

    for (int gi = 0; gi < ngpu; ++gi) {
        Gpu& gp = gpus_[(size_t) gi];
        set_dev(gp.dev);
        gp.h_step[0] = token;
        gp.h_step[1] = pos;
        gp.h_step[2] = pos + 1;
        CK(cudaMemcpyAsync(gp.d_step, gp.h_step, 3 * sizeof(int), cudaMemcpyHostToDevice, gp.s));
        if (gi > 0) transfer_x(gi - 1, gi, 1, false);

        if (graph_ok) {
            if (!gp.graph_ready) {
                if (pos == 0) {
                    decode_segment(gi, true);
                } else {
                    CK(cudaStreamBeginCapture(gp.s, cudaStreamCaptureModeRelaxed));
                    decode_segment(gi, true);
                    CK(cudaStreamEndCapture(gp.s, &gp.graph));
                    CK(cudaGraphInstantiate(&gp.graph_exec, gp.graph, nullptr, nullptr, 0));
                    gp.graph_ready = true;
                    CK(cudaGraphLaunch(gp.graph_exec, gp.s));
                }
            } else {
                CK(cudaGraphLaunch(gp.graph_exec, gp.s));
            }
        } else {
            decode_segment(gi, false);
        }
    }
    return gpus_.back().logits;
}

// ------------------------------ batched prefill ------------------------------

void Model::gemm_mat(Gpu& gp, const Mat& m, const uint16_t* a16, int T, float* out_f32) {
    if (!iq::f16_rows(m.type, m.p, m.k, m.n, gp.pf_w16, m.k, gp.s))
        throw std::runtime_error("prefill: f16_rows failed for type " + std::to_string(m.type));
    const float alpha = 1.0f, beta = 0.0f;
    CKBLAS(cublasGemmEx(gp.blas, CUBLAS_OP_T, CUBLAS_OP_N, m.n, T, m.k, &alpha, gp.pf_w16, CUDA_R_16F, m.k, a16,
                        CUDA_R_16F, m.k, &beta, out_f32, CUDA_R_32F, m.n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

void Model::kda_prefill(Gpu& gp, Layer& l, int T) {
    const Config& c = c_;
    const int E = c.n_embd, DI = c.d_inner(), H = c.n_head;
    cudaStream_t s = gp.s;

    float* pf_q = gp.pf_tmp;
    float* pf_k = pf_q + (size_t) T * DI;
    float* pf_v = pf_k + (size_t) T * DI;
    float* pf_qc = pf_v + (size_t) T * DI;
    float* pf_kc = pf_qc + (size_t) T * DI;
    float* pf_vc = pf_kc + (size_t) T * DI;
    float* pf_f = pf_vc + (size_t) T * DI;
    float* pf_g2 = pf_f + (size_t) T * DI;
    float* pf_beta = pf_g2 + (size_t) T * DI;

    hy::to_f16(gp.pf_cur, (size_t) E, T, E, gp.pf_x16, (size_t) E, s);
    gemm_mat(gp, l.wq, gp.pf_x16, T, pf_q);
    gemm_mat(gp, l.wk, gp.pf_x16, T, pf_k);
    gemm_mat(gp, l.wv, gp.pf_x16, T, pf_v);
    gemm_mat(gp, l.wf, gp.pf_x16, T, pf_f);
    gemm_mat(gp, l.wg, gp.pf_x16, T, pf_g2);
    gemm_mat(gp, l.wbeta, gp.pf_x16, T, pf_beta);

    const float* proj3[3] = {pf_q, pf_k, pf_v};
    const float* cw3[3] = {l.conv_q, l.conv_k, l.conv_v};
    float* out3[3] = {pf_qc, pf_kc, pf_vc};
    glmb::kda_conv(proj3, cw3, l.conv_state, out3, T, DI, c.d_conv, s);
    glmb::kda_conv_state(proj3, l.conv_state, T, DI, c.d_conv, s);

    if (T >= 16) {
        glmb::kda_rec_split(pf_qc, pf_kc, pf_vc, pf_f, l.dt_bias, l.ssm_a_neg, c.kda_lb, pf_beta, l.state, pf_g2,
                            l.ssm_norm, c.eps, H, T, gp.pf_x16, s, 4);
    } else {
        glmb::kda_rec(pf_qc, pf_kc, pf_vc, pf_f, l.dt_bias, l.ssm_a_neg, c.kda_lb, pf_beta, l.state, pf_g2, l.ssm_norm,
                      c.eps, H, T, gp.pf_x16, s);
    }
    gemm_mat(gp, l.wo, gp.pf_x16, T, gp.pf_out);
    hy::add(gp.pf_x, gp.pf_out, (size_t) T * E, s);
}

void Model::mla_prefill(Gpu& gp, Layer& l, int T, int pos0) {
    const Config& c = c_;
    const int E = c.n_embd, H = c.n_head, KL = c.kv_lora, R = c.n_rot;
    cudaStream_t s = gp.s;

    auto align_ptr = [](uint8_t* p) -> uint8_t* {
        return (uint8_t*) (((uintptr_t) p + 255u) & ~uintptr_t(255u));
    };
    uint8_t* curp = (uint8_t*) gp.pf_tmp;
    float* pf_qm = (float*) curp;
    curp = align_ptr(curp + (size_t) T * H * c.qk_head * 4);
    uint16_t* pf_qm16 = (uint16_t*) curp;
    curp = align_ptr(curp + (size_t) T * H * c.qk_head * 2);
    float* pf_kvall = (float*) curp;
    curp = align_ptr(curp + (size_t) T * (KL + R) * 4);
    float* pf_qabs = (float*) curp;
    curp = align_ptr(curp + (size_t) T * H * KL * 4);
    float* pf_ctx = (float*) curp;
    curp = align_ptr(curp + (size_t) T * H * KL * 4);
    uint16_t* pf_ctx16 = (uint16_t*) curp;
    curp = align_ptr(curp + (size_t) T * H * KL * 2);
    float* pf_o128 = (float*) curp;
    curp = align_ptr(curp + (size_t) T * H * c.v_head * 4);
    float* pf_gate = (float*) curp;

    hy::to_f16(gp.pf_cur, (size_t) E, T, E, gp.pf_x16, (size_t) E, s);
    gemm_mat(gp, l.wqm, gp.pf_x16, T, pf_qm);
    gemm_mat(gp, l.wkva, gp.pf_x16, T, pf_kvall);
    gemm_mat(gp, l.wgate, gp.pf_x16, T, pf_gate);

    k::mla_q_prep(pf_qm, T, H, c.qk_head, c.qk_nope, R, nullptr, pos0, c.rope_base, nullptr, nullptr, pf_qm16, s);

    if (!iq::f16_rows(l.wkb.type, l.wkb.p, c.qk_nope, KL * H, gp.pf_w16, c.qk_nope, s))
        throw std::runtime_error("mla prefill: k_b f16_rows failed");
    const float alpha = 1.0f, beta = 0.0f;
    CKBLAS(cublasGemmStridedBatchedEx(gp.blas, CUBLAS_OP_T, CUBLAS_OP_N, KL, T, c.qk_nope, &alpha, gp.pf_w16,
                                      CUDA_R_16F, c.qk_nope, (long long) KL * c.qk_nope, pf_qm16, CUDA_R_16F,
                                      H * c.qk_head, (long long) c.qk_head, &beta, pf_qabs, CUDA_R_32F, H * KL,
                                      (long long) KL, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

    k::kv_store_split(pf_kvall, T, KL, R, l.kv_a_norm, c.eps, nullptr, pos0, c.rope_base, l.lat, o_.kv_q8, l.k_pe, s);

    glmd::MlaPrompt mp;
    mp.q_abs = pf_qabs;
    mp.q = pf_qm;
    mp.q_stride = c.qk_head;
    mp.q_pe_off = c.qk_nope;
    mp.lat = l.lat;
    mp.lat_q8 = o_.kv_q8 ? 1 : 0;
    mp.k_pe = l.k_pe;
    mp.walk = glmd::AttnWalk::Dense;
    mp.p0 = pos0;
    mp.T = T;
    mp.n_head = H;
    mp.kv_lora = KL;
    mp.qk_rope = R;
    mp.scale = 1.0f / sqrtf((float) c.qk_head);
    mp.ctx = pf_ctx;
    if (!glmd::mla_prompt_wmma(mp, s)) {
        if (!glmd::mla_prompt_f32(mp, s)) throw std::runtime_error("mla prefill: attention failed");
    }

    hy::to_f16(pf_ctx, (size_t) H * KL, T, H * KL, pf_ctx16, (size_t) H * KL, s);
    if (!iq::f16_rows(l.wvb.type, l.wvb.p, KL, c.v_head * H, gp.pf_w16, KL, s))
        throw std::runtime_error("mla prefill: v_b f16_rows failed");
    CKBLAS(cublasGemmStridedBatchedEx(gp.blas, CUBLAS_OP_T, CUBLAS_OP_N, c.v_head, T, KL, &alpha, gp.pf_w16, CUDA_R_16F,
                                      KL, (long long) c.v_head * KL, pf_ctx16, CUDA_R_16F, H * KL, (long long) KL,
                                      &beta, pf_o128, CUDA_R_32F, H * c.v_head, (long long) c.v_head, H,
                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

    k::head_gate_out(pf_o128, pf_gate, T, H, c.v_head, nullptr, nullptr, gp.pf_x16, s);
    gemm_mat(gp, l.wom, gp.pf_x16, T, gp.pf_out);
    hy::add(gp.pf_x, gp.pf_out, (size_t) T * E, s);
}

void Model::ffn_dense_prefill(Gpu& gp, Layer& l, int il, int T) {
    const Config& c = c_;
    const int E = c.n_embd, FF = c.n_ff;
    cudaStream_t s = gp.s;
    float* pf_g = gp.pf_tmp;
    float* pf_u = pf_g + (size_t) T * FF;
    hy::to_f16(gp.pf_cur, (size_t) E, T, E, gp.pf_x16, (size_t) E, s);
    gemm_mat(gp, l.fg, gp.pf_x16, T, pf_g);
    gemm_mat(gp, l.fu, gp.pf_x16, T, pf_u);
    k::swiglu_clamp_out(pf_g, pf_u, T, FF, c.clamp_shexp[il], nullptr, nullptr, nullptr, gp.pf_x16, s);
    gemm_mat(gp, l.fd, gp.pf_x16, T, gp.pf_out);
    hy::add(gp.pf_x, gp.pf_out, (size_t) T * E, s);
}

void Model::ffn_moe_prefill(Gpu& gp, Layer& l, int il, int T) {
    const Config& c = c_;
    const int E = c.n_embd, FE = c.n_ff_exp, FS = c.n_ff_shexp, NU = c.n_used, NE = c.n_expert;
    cudaStream_t s = gp.s;
    cudaStream_t sc = gp.s_copy;

    auto align_ptr = [](uint8_t* p) -> uint8_t* {
        return (uint8_t*) (((uintptr_t) p + 255u) & ~uintptr_t(255u));
    };
    uint8_t* curp = (uint8_t*) gp.pf_tmp;
    float* pf_rlog = (float*) curp;
    curp = align_ptr(curp + (size_t) T * NE * 4);
    uint16_t* pf_h16 = (uint16_t*) curp;
    curp = align_ptr(curp + (size_t) T * NU * FE * 2);
    float* pf_y = (float*) curp;

    if (l.router_type == 0) {
        const float alpha = 1.0f, beta = 0.0f;
        CKBLAS(cublasSgemm(gp.blas, CUBLAS_OP_T, CUBLAS_OP_N, NE, T, E, &alpha, (const float*) l.router_w, E, gp.pf_cur,
                           E, &beta, pf_rlog, NE));
    } else {
        hy::gemv_bf16w((const uint16_t*) l.router_w, (size_t) E, E, NE, gp.pf_cur, (size_t) E, T, pf_rlog, (size_t) NE,
                       1.0f, s);
    }

    k::router_grouped(pf_rlog, l.rb_dev, T, NE, NU, c.n_group, c.n_group_used, c.w_norm, c.w_scale, gp.pf_ids, gp.pf_w,
                      s);

    std::vector<int> miss_list;
    int vram_hits = 0;
    if (l.n_res < NE) {
        CK(cudaMemcpyAsync(l.h_slot.data(), l.fl.slot, (size_t) NE * sizeof(int), cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(gp.h_pf_ids, gp.pf_ids, (size_t) T * NU * sizeof(int), cudaMemcpyDeviceToHost, s));
        if (lane_ && T <= 48) {
            CK(cudaMemcpyAsync(gp.h_pf_w, gp.pf_w, (size_t) T * NU * sizeof(float), cudaMemcpyDeviceToHost, s));
        }
        CK(cudaStreamSynchronize(s));
        std::vector<uint8_t> seen((size_t) NE, 0);
        for (int i = 0; i < T * NU; ++i) {
            const int e = gp.h_pf_ids[i];
            l.pf_counts[(size_t) e]++;
            if (l.h_slot[(size_t) e] >= 0) {
                ++vram_hits;
            } else if (!seen[(size_t) e]) {
                seen[(size_t) e] = 1;
                miss_list.push_back(e);
            }
        }
    } else {
        vram_hits = T * NU;
    }

    auto run_set = [&](const uint8_t* wbase, const int* slot_map, int n_slots, bool accumulate) {
        hy::moe_sort(gp.pf_ids, T, NU, NE, gp.pf_bounds, gp.pf_tok, gp.pf_row_of, s, slot_map, nullptr, n_slots);
        iq::SetArgs a;
        a.wbase = wbase;
        a.stride = l.blob;
        a.tr[0] = l.tr[0];
        a.tr[1] = l.tr[1];
        a.tr[2] = l.tr[2];
        a.bounds = gp.pf_bounds;
        a.row0 = 0;
        a.max_rows = T * NU;
        a.n_exp = n_slots;
        a.n_embd = E;
        a.n_ff = FE;
        a.limit = c.clamp_exp[il] > 1e-6f ? -c.clamp_exp[il] : 0.0f;
        a.xf = gp.pf_cur;
        a.idx = gp.pf_tok;
        a.ldx = E;
        a.h = pf_h16;
        a.ldo = FE;
        a.tg = l.etg;
        a.td = l.etd;
        a.act = iq::kActSwiglu;
        if (!iq::set_gate_up(a, s)) throw std::runtime_error("moe prefill: set_gate_up failed");
        iq::SetArgs d = a;
        d.xf = nullptr;
        d.idx = nullptr;
        d.xh = pf_h16;
        d.ldx = FE;
        d.h = nullptr;
        d.y = pf_y;
        d.ldo = E;
        if (!iq::set_down(d, s)) throw std::runtime_error("moe prefill: set_down failed");
        hy::moe_combine(pf_y, gp.pf_row_of, gp.pf_w, T, NU, E, gp.pf_moe, s, accumulate);
    };

    const bool cpu_miss = !miss_list.empty() && lane_ && T <= 48;
    const size_t n_batches = cpu_miss ? 0 : (miss_list.size() + (size_t) kPfBatch - 1) / (size_t) kPfBatch;
    auto stage_batch = [&](size_t bi) {
        const int buf = (int) (bi & 1u);
        const size_t off = bi * (size_t) kPfBatch;
        const int B = (int) std::min<size_t>((size_t) kPfBatch, miss_list.size() - off);
        if (bi >= 2) {
            // Wait until stream s finished using buf from batch bi-2 before overwriting h_pf_pmap/pf_stage[buf]
            CK(cudaEventSynchronize(gp.ev_comp[buf]));
        }
        int* pmap_h = gp.h_pf_pmap[buf];
        std::fill(pmap_h, pmap_h + NE, -1);
        for (int b = 0; b < B; ++b) pmap_h[miss_list[off + (size_t) b]] = b;
        CK(cudaMemcpyAsync(gp.pf_pmap[buf], pmap_h, (size_t) NE * sizeof(int), cudaMemcpyHostToDevice, sc));
        if (l.host_pinned) {
            for (int b = 0; b < B; ++b) {
                const int e = miss_list[off + (size_t) b];
                CK(cudaMemcpyAsync(gp.pf_stage[buf] + (size_t) b * l.blob, l.host_blobs + (size_t) e * l.blob, l.blob,
                                   cudaMemcpyHostToDevice, sc));
            }
        } else {
            auto cp = [&](int b) {
                const int e = miss_list[off + (size_t) b];
                std::memcpy(gp.h_pf_stage[buf] + (size_t) b * l.blob, l.host_blobs + (size_t) e * l.blob, l.blob);
            };
            if (lane_ && B >= 4) lane_->parallel_for(B, cp);
            else for (int b = 0; b < B; ++b) cp(b);
            CK(cudaMemcpyAsync(gp.pf_stage[buf], gp.h_pf_stage[buf], (size_t) B * l.blob, cudaMemcpyHostToDevice, sc));
        }
        CK(cudaEventRecord(gp.ev_copy[buf], sc));
    };

    // Start first DMA batch on s_copy immediately so it overlaps with VRAM-resident experts on stream s
    if (n_batches > 0) stage_batch(0);

    bool accum = false;
    if (vram_hits > 0) {
        run_set(l.fl.vram, l.n_res == NE ? nullptr : l.fl.slot, l.n_res, false);
        accum = true;
    }

    // Fast path for tiny prompts when CPU lane is active: compute misses on CPU in parallel with VRAM hits
    if (cpu_miss) {
        std::vector<float> h_x((size_t) T * E);
        std::vector<float> h_o((size_t) T * E);
        CK(cudaMemcpyAsync(h_x.data(), gp.pf_cur, (size_t) T * E * 4, cudaMemcpyDeviceToHost, sc));
        CK(cudaStreamSynchronize(sc));
        std::vector<hy4::CpuEntry> entries;
        entries.reserve((size_t) T * NU);
        for (int t = 0; t < T; ++t) {
            for (int j = 0; j < NU; ++j) {
                const int e = gp.h_pf_ids[t * NU + j];
                if (l.h_slot[(size_t) e] < 0) entries.push_back({t, e, gp.h_pf_w[t * NU + j]});
            }
        }
        hy4::cpu_experts(*lane_, l.cex, entries, h_x.data(), T, h_o.data());
        CK(cudaMemcpyAsync(gp.pf_out, h_o.data(), (size_t) T * E * 4, cudaMemcpyHostToDevice, s));
        if (accum) hy::add(gp.pf_moe, gp.pf_out, (size_t) T * E, s);
        else CK(cudaMemcpyAsync(gp.pf_moe, gp.pf_out, (size_t) T * E * 4, cudaMemcpyDeviceToDevice, s));
        accum = true;
    }

    if (n_batches > 0) {
        for (size_t bi = 0; bi < n_batches; ++bi) {
            const int buf = (int) (bi & 1u);
            const size_t off = bi * (size_t) kPfBatch;
            const int B = (int) std::min<size_t>((size_t) kPfBatch, miss_list.size() - off);
            CK(cudaStreamWaitEvent(s, gp.ev_copy[buf], 0));
            run_set(gp.pf_stage[buf], gp.pf_pmap[buf], B, accum);
            CK(cudaEventRecord(gp.ev_comp[buf], s));
            accum = true;
            if (bi + 1 < n_batches) stage_batch(bi + 1);
        }
        // Ensure final batch completed before host returns or reuses buffers next layer
        CK(cudaEventSynchronize(gp.ev_comp[(n_batches - 1) & 1u]));
    }
    if (!accum) CK(cudaMemsetAsync(gp.pf_moe, 0, (size_t) T * E * 4, s));

    // Shared expert
    float* sh_g = gp.pf_tmp;
    float* sh_u = sh_g + (size_t) T * FS;
    hy::to_f16(gp.pf_cur, (size_t) E, T, E, gp.pf_x16, (size_t) E, s);
    gemm_mat(gp, l.fg, gp.pf_x16, T, sh_g);
    gemm_mat(gp, l.fu, gp.pf_x16, T, sh_u);
    k::swiglu_clamp_out(sh_g, sh_u, T, FS, c.clamp_shexp[il], nullptr, nullptr, nullptr, gp.pf_x16, s);
    gemm_mat(gp, l.fd, gp.pf_x16, T, gp.pf_out);
    k::add2(gp.pf_x, gp.pf_moe, gp.pf_out, (int64_t) T * E, s);
}

void Model::warm_vram_cache() {
    last_warm_s_ = 0.0;
    if (!o_.cache_warm) return;
    const auto t0 = std::chrono::steady_clock::now();
    const Config& c = c_;
    const int NE = c.n_expert;

    // Step 1: read back current slot maps asynchronously across all GPUs
    for (int il = c.n_dense_lead; il < c.n_layers; ++il) {
        Layer& l = layers_[(size_t) il];
        if (l.n_res <= 0 || l.n_res >= NE || l.host_blobs == nullptr) continue;
        Gpu& gp = gpus_[(size_t) l.dev_idx];
        set_dev(gp.dev);
        CK(cudaMemcpyAsync(l.h_slot.data(), l.fl.slot, (size_t) NE * sizeof(int), cudaMemcpyDeviceToHost, gp.s_copy));
    }
    for (auto& gp : gpus_) {
        set_dev(gp.dev);
        CK(cudaStreamSynchronize(gp.s_copy));
    }

    // Step 2: compute updates on host and queue H2D warm copies concurrently across GPUs
    std::vector<std::vector<int>> owners((size_t) c.n_layers);
    std::vector<std::vector<float>> scores((size_t) c.n_layers);
    for (int il = c.n_dense_lead; il < c.n_layers; ++il) {
        Layer& l = layers_[(size_t) il];
        if (l.n_res <= 0 || l.n_res >= NE || l.host_blobs == nullptr) continue;
        uint32_t max_c = 0;
        for (uint32_t v : l.pf_counts) max_c = std::max(max_c, v);
        if (max_c == 0) continue;

        Gpu& gp = gpus_[(size_t) l.dev_idx];
        set_dev(gp.dev);

        std::vector<int> order((size_t) NE);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            if (l.pf_counts[(size_t) a] != l.pf_counts[(size_t) b])
                return l.pf_counts[(size_t) a] > l.pf_counts[(size_t) b];
            return (l.h_slot[(size_t) a] >= 0) > (l.h_slot[(size_t) b] >= 0);
        });

        std::vector<uint8_t> want_e((size_t) NE, 0);
        for (int i = 0; i < l.n_res; ++i) want_e[(size_t) order[(size_t) i]] = 1;

        std::vector<int> free_slots;
        auto& h_owner = owners[(size_t) il];
        h_owner.assign((size_t) l.n_res, -1);
        for (int e = 0; e < NE; ++e) {
            const int sl = l.h_slot[(size_t) e];
            if (sl >= 0) {
                if (want_e[(size_t) e]) {
                    h_owner[(size_t) sl] = e;
                } else {
                    free_slots.push_back(sl);
                    l.h_slot[(size_t) e] = -1;
                }
            }
        }
        size_t fi = 0;
        for (int i = 0; i < l.n_res; ++i) {
            const int e = order[(size_t) i];
            if (l.h_slot[(size_t) e] >= 0) continue;
            const int sl = free_slots[fi++];
            l.h_slot[(size_t) e] = sl;
            h_owner[(size_t) sl] = e;
            CK(cudaMemcpyAsync(l.fl.vram + (size_t) sl * l.blob, l.host_blobs + (size_t) e * l.blob, l.blob,
                               cudaMemcpyHostToDevice, gp.s_copy));
        }

        auto& sc = scores[(size_t) il];
        sc.resize((size_t) NE);
        const float norm = 4.0f / (float) max_c;
        for (int e = 0; e < NE; ++e) sc[(size_t) e] = (float) l.pf_counts[(size_t) e] * norm;
        CK(cudaMemcpyAsync(l.fl.slot, l.h_slot.data(), (size_t) NE * sizeof(int), cudaMemcpyHostToDevice, gp.s_copy));
        CK(cudaMemcpyAsync(l.fl.owner, h_owner.data(), (size_t) l.n_res * sizeof(int), cudaMemcpyHostToDevice,
                           gp.s_copy));
        CK(cudaMemcpyAsync(l.fl.score, sc.data(), (size_t) NE * sizeof(float), cudaMemcpyHostToDevice, gp.s_copy));
        std::fill(l.pf_counts.begin(), l.pf_counts.end(), 0);
    }
    for (auto& gp : gpus_) {
        set_dev(gp.dev);
        CK(cudaStreamSynchronize(gp.s_copy));
    }
    last_warm_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

const float* Model::prefill(const int* tokens, int n, int pos0, std::vector<float>* all_logits_host) {
    if (n <= 0) throw std::runtime_error("prefill: empty prompt");
    last_warm_s_ = 0.0;
    if (n == 1 || o_.prefill_chunk <= 1 || !o_.seam_dir.empty()) {
        const float* lg = nullptr;
        std::vector<float> row;
        if (all_logits_host != nullptr) row.resize((size_t) c_.vocab);
        for (int i = 0; i < n; ++i) {
            lg = step(tokens[i], pos0 + i);
            if (all_logits_host != nullptr) {
                set_dev(output_dev());
                CK(cudaMemcpyAsync(row.data(), lg, (size_t) c_.vocab * 4, cudaMemcpyDeviceToHost, output_stream()));
                CK(cudaStreamSynchronize(output_stream()));
                all_logits_host->insert(all_logits_host->end(), row.begin(), row.end());
            }
        }
        return lg;
    }

    const Config& c = c_;
    const int E = c.n_embd, ngpu = (int) gpus_.size();
    const int P = gpus_[0].pf_cap;
    std::vector<float> row_tmp;
    if (all_logits_host != nullptr) row_tmp.resize((size_t) c_.vocab);

    for (int off = 0; off < n; off += P) {
        const int T = std::min(P, n - off);
        const int cp0 = pos0 + off;
        for (int gi = 0; gi < ngpu; ++gi) {
            Gpu& gp = gpus_[(size_t) gi];
            set_dev(gp.dev);
            if (gi == 0) {
                CK(cudaMemcpyAsync(gp.pf_tok_ids, tokens + off, (size_t) T * sizeof(int), cudaMemcpyHostToDevice, gp.s));
                if (!k::emb_rows(emb_.p, emb_.type, emb_.k, gp.pf_tok_ids, T, gp.pf_x, gp.s))
                    throw std::runtime_error("prefill: emb_rows failed");
            } else {
                transfer_x(gi - 1, gi, T, true);
            }
            for (int il = gp.l0; il < gp.l1; ++il) {
                Layer& l = layers_[(size_t) il];
                hy::rmsnorm(gp.pf_x, (size_t) E, T, E, l.attn_norm, c.eps, gp.pf_cur, (size_t) E, gp.s);
                if (l.kda) kda_prefill(gp, l, T);
                else mla_prefill(gp, l, T, cp0);
                hy::rmsnorm(gp.pf_x, (size_t) E, T, E, l.ffn_norm, c.eps, gp.pf_cur, (size_t) E, gp.s);
                if (il < c.n_dense_lead) ffn_dense_prefill(gp, l, il, T);
                else ffn_moe_prefill(gp, l, il, T);
            }
            if (gi == ngpu - 1) {
                if (all_logits_host != nullptr) {
                    for (int t = 0; t < T; ++t) {
                        hy::rmsnorm(gp.pf_x + (size_t) t * E, (size_t) E, 1, E, out_norm_, c.eps, gp.hn, (size_t) E,
                                    gp.s, {gp.xq, gp.xd});
                        iq::MvJob j{out_w_.p, out_w_.ld, out_w_.type, E, c.vocab, 1, gp.xq, gp.xd, gp.logits,
                                    (size_t) c.vocab, nullptr, 1.0f};
                        if (!iq::mv(&j, 1, gp.s)) throw std::runtime_error("prefill: output failed");
                        CK(cudaMemcpyAsync(row_tmp.data(), gp.logits, (size_t) c.vocab * 4, cudaMemcpyDeviceToHost,
                                           gp.s));
                        CK(cudaStreamSynchronize(gp.s));
                        all_logits_host->insert(all_logits_host->end(), row_tmp.begin(), row_tmp.end());
                    }
                } else if (off + T == n) {
                    hy::rmsnorm(gp.pf_x + (size_t) (T - 1) * E, (size_t) E, 1, E, out_norm_, c.eps, gp.hn, (size_t) E,
                                gp.s, {gp.xq, gp.xd});
                    iq::MvJob j{out_w_.p, out_w_.ld, out_w_.type, E, c.vocab, 1, gp.xq, gp.xd, gp.logits,
                                (size_t) c.vocab, nullptr, 1.0f};
                    if (!iq::mv(&j, 1, gp.s)) throw std::runtime_error("prefill: output failed");
                }
            }
        }
    }

    for (auto& gp : gpus_) {
        set_dev(gp.dev);
        CK(cudaStreamSynchronize(gp.s));
    }
    warm_vram_cache();
    return gpus_.back().logits;
}

// ------------------------------ MTP block ------------------------------

const float* Model::mtp_step(const float* h_nextn_dev, int next_token, int pos) {
    if (!mtp_loaded_) throw std::runtime_error("mtp_step called without --mtp");
    const Config& c = c_;
    const int E = c.n_embd;
    Gpu& g0 = gpus_[0];
    Gpu& gl = gpus_.back();

    set_dev(g0.dev);
    g0.h_step[0] = next_token;
    CK(cudaMemcpyAsync(g0.d_step, g0.h_step, sizeof(int), cudaMemcpyHostToDevice, g0.s));
    if (!k::emb_rows(emb_.p, emb_.type, emb_.k, g0.d_step, 1, g0.dev == gl.dev ? mtp_.emb_buf : g0.out, g0.s))
        throw std::runtime_error("mtp: emb_rows failed");
    if (g0.dev != gl.dev) {
        CK(cudaEventRecord(g0.ev_xfer, g0.s));
        set_dev(gl.dev);
        CK(cudaStreamWaitEvent(gl.s, g0.ev_xfer, 0));
        CK(cudaMemcpyPeerAsync(mtp_.emb_buf, gl.dev, g0.out, g0.dev, (size_t) E * 4, gl.s));
    } else {
        set_dev(gl.dev);
    }

    gl.h_step[0] = next_token;
    gl.h_step[1] = pos;
    gl.h_step[2] = pos + 1;
    CK(cudaMemcpyAsync(gl.d_step, gl.h_step, 3 * sizeof(int), cudaMemcpyHostToDevice, gl.s));

    k::concat_norm_eh(mtp_.emb_buf, mtp_.enorm, h_nextn_dev, mtp_.hnorm, c.eps, E, nullptr, mtp_.eh_q, mtp_.eh_d, gl.s);
    iq::MvJob eh{mtp_.eh_proj.p, mtp_.eh_proj.ld, mtp_.eh_proj.type, 2 * E, E, 1, mtp_.eh_q, mtp_.eh_d, gl.x,
                 (size_t) E, nullptr, 1.0f};
    if (!iq::mv(&eh, 1, gl.s)) throw std::runtime_error("mtp: eh_proj failed");

    hy::rmsnorm(gl.x, (size_t) E, 1, E, mtp_.layer.attn_norm, c.eps, gl.cur, (size_t) E, gl.s, {gl.xq, gl.xd});
    mla_decode(gl, mtp_.layer, c.n_layers, false);
    hy::rmsnorm(gl.x, (size_t) E, 1, E, mtp_.layer.ffn_norm, c.eps, gl.cur, (size_t) E, gl.s, {gl.xq, gl.xd});
    ffn_moe_decode(gl, mtp_.layer, c.n_layers, false);

    hy::rmsnorm(gl.x, (size_t) E, 1, E, mtp_.out_norm, c.eps, gl.hn, (size_t) E, gl.s, {gl.xq, gl.xd});
    iq::MvJob outj{out_w_.p, out_w_.ld, out_w_.type, E, c.vocab, 1, gl.xq, gl.xd, gl.logits, (size_t) c.vocab, nullptr,
                   1.0f};
    if (!iq::mv(&outj, 1, gl.s)) throw std::runtime_error("mtp: output failed");
    return gl.logits;
}

// ------------------------------ utilities ------------------------------

int Model::argmax_logits(const float* logits_dev) {
    Gpu& gl = gpus_.back();
    set_dev(gl.dev);
    hy::row_top(logits_dev, 1, c_.vocab, (size_t) c_.vocab, 0, gl.d_top, gl.s);
    CK(cudaMemcpyAsync(gl.h_top, gl.d_top, 2 * sizeof(float), cudaMemcpyDeviceToHost, gl.s));
    CK(cudaStreamSynchronize(gl.s));
    int id = 0;
    std::memcpy(&id, gl.h_top, 4);
    return id;
}

TierStats Model::stats() const {
    TierStats st;
    for (const auto& gp : gpus_) {
        cudaSetDevice(gp.dev);
        unsigned long long s[8] = {};
        cudaMemcpy(s, gp.fxs.stats, sizeof s, cudaMemcpyDeviceToHost);
        st.picks += s[0];
        st.vram_hits += s[1];
        st.copied += s[2];
        st.zero_copy += s[3];
        st.cpu_lane += s[4];
        st.admitted += s[5];
    }
    if (st.cpu_lane == 0) st.cpu_lane = cpu_lane_picks_;
    return st;
}

std::string Model::detokenize_one(int id) const {
    if (id < 0 || (size_t) id >= vocab_tokens_.size()) return "";
    return decode_gpt2_token(vocab_tokens_[(size_t) id]);
}

std::string Model::detokenize(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) out += detokenize_one(id);
    return out;
}

}  // namespace ling
