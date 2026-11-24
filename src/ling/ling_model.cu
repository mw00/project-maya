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
#include <condition_variable>
#include <mutex>
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
        if (o_.load_mtp) {
            l.conv_bak = (float*) dev_alloc(g, cs);
            l.state_tmp = (float*) dev_alloc(g, st);
            // [W_max] buffers: q_in, k_in, v_in, kc, vc, g1 (6 * W * DI) + beta (W * H)
            const size_t wb = ((size_t) 6 * kMaxWin * DI + (size_t) kMaxWin * H) * sizeof(float);
            l.w_kda_buf = (float*) dev_alloc(g, wb);
        }
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
        CK(cudaStreamCreateWithFlags(&gp.s_d2h, cudaStreamNonBlocking));
        CKBLAS(cublasCreate(&gp.blas));
        CKBLAS(cublasSetStream(gp.blas, gp.s));
        CK(cudaEventCreateWithFlags(&gp.ev_xfer, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&gp.ev_route, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&gp.ev_d2h, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&gp.ev_norm, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&gp.ev_sh, cudaEventDisableTiming));
        for (int b = 0; b < 2; ++b) {
            CK(cudaEventCreateWithFlags(&gp.ev_copy[b], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&gp.ev_comp[b], cudaEventDisableTiming));
        }
        for (int r = 0; r < kPfPmapRing; ++r)
            if (gp.ev_pmap[r]) cudaEventDestroy(gp.ev_pmap[r]);
        for (int b = 0; b < 2; ++b) {
            CK(cudaEventRecord(gp.ev_copy[b], gp.s_copy));
            CK(cudaEventRecord(gp.ev_comp[b], gp.s));
        }
        for (int r = 0; r < kPfPmapRing; ++r) {
            CK(cudaEventCreateWithFlags(&gp.ev_pmap[r], cudaEventDisableTiming));
            CK(cudaEventRecord(gp.ev_pmap[r], gp.s_copy));
        }
        const int m0 = (n_moe * gi) / ngpu;
        const int m1 = (n_moe * (gi + 1)) / ngpu;
        gp.l0 = gi == 0 ? 0 : (c.n_dense_lead + m0);
        gp.l1 = c.n_dense_lead + m1;
    }

    p2p_mat_.assign((size_t) ngpu, std::vector<bool>((size_t) ngpu, false));
    for (int gi = 0; gi < ngpu; ++gi) {
        p2p_mat_[(size_t) gi][(size_t) gi] = true;
        for (int gj = gi + 1; gj < ngpu; ++gj) {
            const int d0 = gpus_[(size_t) gi].dev, d1 = gpus_[(size_t) gj].dev;
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
                    (e1 == cudaSuccess || e1 == cudaErrorPeerAccessAlreadyEnabled)) {
                    p2p_mat_[(size_t) gi][(size_t) gj] = true;
                    p2p_mat_[(size_t) gj][(size_t) gi] = true;
                }
            }
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
        mtp_.hring = (float*) dev_alloc(glast, (size_t) MtpWeights::kRing * (size_t) E * sizeof(float));
        CK(cudaMemset(mtp_.hring, 0, (size_t) MtpWeights::kRing * (size_t) E * sizeof(float)));
        mtp_.h_tmp = (float*) dev_alloc(glast, (size_t) E * sizeof(float));
        mtp_loaded_ = true;
    }

    // Allocate decode + double-buffered prefill scratch
    const int P = std::max(1, std::min(o_.prefill_chunk, o_.ctx));
    const size_t mla_part_b = std::max(glmd::mla_dec_part_bytes(H), k::mla_dec_wmma_part_bytes(H, 4));
    const size_t max_w16_elems = std::max<size_t>(
        {(size_t) H * c.qk_head * (size_t) E, (size_t) c.n_ff * (size_t) E, (size_t) H * (size_t) KL * c.qk_nope,
         (size_t) H * (size_t) c.v_head * (size_t) KL, (size_t) DI * (size_t) E});
    const int Wcap = o_.load_mtp ? kMaxWin : 1;
    for (int gi = 0; gi < ngpu; ++gi) {
        Gpu& gp = gpus_[(size_t) gi];
        set_dev(gp.dev);
        gp.x = (float*) dev_alloc(gp, (size_t) Wcap * E * 4);
        gp.cur = (float*) dev_alloc(gp, (size_t) Wcap * E * 4);
        gp.out = (float*) dev_alloc(gp, (size_t) Wcap * E * 4);
        gp.hn = (float*) dev_alloc(gp, (size_t) E * 4);
        gp.xq = (int8_t*) dev_alloc(gp, (size_t) Wcap * E);
        gp.xd = (float*) dev_alloc(gp, (size_t) Wcap * (E / 32) * 4);
        gp.xq2 = (int8_t*) dev_alloc(gp, (size_t) Wcap * std::max(DI, H * c.v_head));
        gp.xd2 = (float*) dev_alloc(gp, (size_t) Wcap * (std::max(DI, H * c.v_head) / 32) * 4);
        gp.q = (float*) dev_alloc(gp, (size_t) Wcap * DI * 4);
        gp.k = (float*) dev_alloc(gp, (size_t) Wcap * DI * 4);
        gp.v = (float*) dev_alloc(gp, (size_t) Wcap * DI * 4);
        gp.qc = (float*) dev_alloc(gp, (size_t) Wcap * DI * 4);
        gp.kc = (float*) dev_alloc(gp, (size_t) Wcap * DI * 4);
        gp.vc = (float*) dev_alloc(gp, (size_t) Wcap * DI * 4);
        gp.f = (float*) dev_alloc(gp, (size_t) Wcap * DI * 4);
        gp.g2 = (float*) dev_alloc(gp, (size_t) Wcap * DI * 4);
        gp.g1 = (float*) dev_alloc(gp, (size_t) Wcap * DI * 4);
        gp.beta = (float*) dev_alloc(gp, (size_t) Wcap * H * 4);
        gp.raw_o = (float*) dev_alloc(gp, (size_t) DI * 4);
        gp.qm = (float*) dev_alloc(gp, (size_t) Wcap * H * c.qk_head * 4);
        gp.kvall = (float*) dev_alloc(gp, (size_t) Wcap * (KL + R) * 4);
        gp.q_abs = (float*) dev_alloc(gp, (size_t) Wcap * H * KL * 4);
        gp.ctxo = (float*) dev_alloc(gp, (size_t) Wcap * H * KL * 4);
        gp.o128 = (float*) dev_alloc(gp, (size_t) Wcap * H * c.v_head * 4);
        gp.gate = (float*) dev_alloc(gp, (size_t) Wcap * H * 4);
        gp.q8n = (int8_t*) dev_alloc(gp, (size_t) Wcap * H * c.qk_nope);
        gp.d8n = (float*) dev_alloc(gp, (size_t) Wcap * H * (c.qk_nope / 32) * 4);
        gp.q8c = (int8_t*) dev_alloc(gp, (size_t) Wcap * H * KL);
        gp.d8c = (float*) dev_alloc(gp, (size_t) Wcap * H * (KL / 32) * 4);
        gp.mla_part = (float*) dev_alloc(gp, mla_part_b);
        gp.gu = (float*) dev_alloc(gp, (size_t) Wcap * std::max(2 * c.n_ff, 2 * c.n_ff_shexp) * 4);
        const size_t eh_rows = (size_t) Wcap * (Wcap > 1 ? 16 : c.n_used);
        gp.eh = (float*) dev_alloc(gp, std::max(eh_rows * (size_t) FE, (size_t) Wcap * c.n_ff) * 4);
        gp.shact = (float*) dev_alloc(gp, (size_t) Wcap * c.n_ff_shexp * 4);
        gp.part = (float*) dev_alloc(gp, eh_rows * (size_t) E * 4);
        gp.shout = (float*) dev_alloc(gp, (size_t) Wcap * E * 4);
        gp.moe = (float*) dev_alloc(gp, (size_t) Wcap * E * 4);
        gp.rlog = (float*) dev_alloc(gp, (size_t) Wcap * c.n_expert * 4);
        gp.d_w = (float*) dev_alloc(gp, (size_t) Wcap * 16 * 4);
        gp.d_ids = (int*) dev_alloc(gp, (size_t) Wcap * 16 * 4);
        gp.d_plan = (unsigned long long*) dev_alloc(gp, (size_t) 16 * sizeof(unsigned long long));
        gp.d_cpu_out = (float*) dev_alloc(gp, (size_t) Wcap * E * 4);
        gp.ehq = (int8_t*) dev_alloc(gp, std::max(eh_rows * (size_t) FE, (size_t) Wcap * c.n_ff));
        gp.ehd = (float*) dev_alloc(gp, (std::max(eh_rows * (size_t) FE, (size_t) Wcap * c.n_ff) / 32) * 4);
        gp.shq = (int8_t*) dev_alloc(gp, (size_t) Wcap * c.n_ff_shexp);
        gp.shd = (float*) dev_alloc(gp, (size_t) Wcap * (c.n_ff_shexp / 32) * 4);
        if (Wcap > 1) {
            gp.x2q = (int8_t*) dev_alloc(gp, (size_t) 2 * Wcap * E);
            gp.x2d = (float*) dev_alloc(gp, (size_t) 2 * Wcap * (E / 32) * 4);
            gp.grp_ptr = (unsigned long long*) dev_alloc(gp, (size_t) Wcap * 16 * sizeof(unsigned long long));
            gp.grp_start = (int*) dev_alloc(gp, (size_t) (Wcap * 16 + 1) * sizeof(int));
            gp.grp_ent = (int*) dev_alloc(gp, (size_t) Wcap * 16 * sizeof(int));
            gp.n_groups = (int*) dev_alloc(gp, sizeof(int));
            gp.d_pos_win = (int*) dev_alloc(gp, (size_t) Wcap * sizeof(int));
            gp.h_pos_win = (int*) host_alloc(gp, (size_t) Wcap * sizeof(int));
            gp.d_tok_win = (int*) dev_alloc(gp, (size_t) Wcap * sizeof(int));
            gp.h_tok_win = (int*) host_alloc(gp, (size_t) Wcap * sizeof(int));
            gp.d_route_win = (k::RoutePackWin*) dev_alloc(gp, sizeof(k::RoutePackWin));
            gp.h_route_win = (k::RoutePackWin*) host_alloc(gp, sizeof(k::RoutePackWin));
        }
        if (gi == ngpu - 1) {
            gp.logits = (float*) dev_alloc(gp, (size_t) Wcap * c_.vocab * 4);
            gp.d_top = (float*) dev_alloc(gp, 2 * sizeof(float));
            gp.d_arg_val = (float*) dev_alloc(gp, 256 * sizeof(float));
            gp.d_arg_idx = (int*) dev_alloc(gp, 128 * sizeof(int));
            gp.d_arg_out = (int*) dev_alloc(gp, sizeof(int));
            gp.h_arg_out = (int*) host_alloc(gp, sizeof(int));
            gp.d_arg_rows = (int*) dev_alloc(gp, (size_t) Wcap * sizeof(int));
            gp.h_arg_rows = (int*) host_alloc(gp, (size_t) Wcap * sizeof(int));
            gp.d_top_pair = (float*) dev_alloc(gp, 2 * sizeof(float));
            gp.h_top_pair = (float*) host_alloc(gp, 2 * sizeof(float));
        }
        gp.d_step = (int*) dev_alloc(gp, 4 * sizeof(int));
        gp.d_iota = (int*) dev_alloc(gp, (size_t) o_.ctx * sizeof(int));
        k::fill_iota(gp.d_iota, o_.ctx, gp.s);
        k::mla_dec_wmma_init();

        gp.fxs.eptr = (unsigned long long*) dev_alloc(gp, (size_t) c.n_expert * sizeof(unsigned long long));
        gp.fxs.who = (int*) dev_alloc(gp, (size_t) Wcap * 16 * sizeof(int));
        gp.fxs.src = (unsigned long long*) dev_alloc(gp, (size_t) Wcap * 16 * sizeof(unsigned long long));
        gp.fxs.dst = (unsigned long long*) dev_alloc(gp, (size_t) Wcap * 16 * sizeof(unsigned long long));
        gp.fxs.nf = (unsigned*) dev_alloc(gp, sizeof(unsigned));
        gp.fxs.n_stage = Wcap > 1 ? std::max(c.n_used * 2, 16) : c.n_used;
        gp.fxs.stage = (uint8_t*) dev_alloc(gp, (size_t) gp.fxs.n_stage * max_blob_);
        gp.fxs.stats = (unsigned long long*) dev_alloc(gp, 8 * sizeof(unsigned long long));
        CK(cudaMemset(gp.fxs.stats, 0, 8 * sizeof(unsigned long long)));

        gp.d_route_pack = (k::RoutePack*) dev_alloc(gp, sizeof(k::RoutePack));
        gp.h_route_pack = (k::RoutePack*) host_alloc(gp, sizeof(k::RoutePack));
        gp.h_neg1 = (int*) host_alloc(gp, sizeof(int));
        *gp.h_neg1 = -1;
        gp.h_step = (int*) host_alloc(gp, 3 * sizeof(int));
        gp.h_ids = (int*) host_alloc(gp, (size_t) Wcap * 16 * sizeof(int));
        gp.h_w = (float*) host_alloc(gp, (size_t) Wcap * 16 * sizeof(float));
        gp.h_who = (int*) host_alloc(gp, (size_t) Wcap * 16 * sizeof(int));
        gp.h_cur = (float*) host_alloc(gp, (size_t) Wcap * E * 4);
        gp.h_cpu_out = (float*) host_alloc(gp, (size_t) Wcap * E * 4);
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
            const size_t mla_b_old = (size_t) P * ((size_t) H * c.qk_head * 6 + (size_t) (KL + R) * 4 +
                                                   (size_t) H * KL * 10 + (size_t) H * c.v_head * 4 + (size_t) H * 4) +
                                     4096;
            const size_t bq_max = (size_t) std::min(P, 256);
            const size_t bk_max = 1024;
            const size_t mla_b_tiled =
                (size_t) P * ((size_t) H * c.qk_head * 4 + (size_t) (KL + R) * 4 + (size_t) H * 4 +
                              (size_t) H * (size_t) (c.qk_nope + R) * 2 + (size_t) H * c.v_head * 4 + (size_t) H * 8) +
                (size_t) H * KL * (size_t) (c.qk_nope + c.v_head) * 2 +
                bk_max * ((size_t) KL * 2 + (size_t) H * (size_t) (c.qk_nope + c.v_head) * 2) +
                (size_t) H * bq_max * bk_max * 6 + 65536;
            const size_t mla_b = std::max(mla_b_old, mla_b_tiled);
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
            gp.h_pf_cap = std::max(P, std::min(o_.prefill_block > 0 ? o_.prefill_block : P, 32768));
            gp.h_pf_ids = (int*) host_alloc(gp, (size_t) gp.h_pf_cap * c.n_used * sizeof(int));
            gp.h_pf_w = (float*) host_alloc(gp, (size_t) gp.h_pf_cap * c.n_used * sizeof(float));
            gp.d_max_rows = (int*) dev_alloc(gp, sizeof(int));
            gp.h_max_rows = (int*) host_alloc(gp, sizeof(int));
            for (int b = 0; b < 2; ++b) {
                gp.pf_pmap[b] = (int*) dev_alloc(gp, (size_t) c.n_expert * sizeof(int));
                gp.pf_stage[b] = (uint8_t*) dev_alloc(gp, (size_t) kPfBatch * max_blob_);
                gp.h_pf_stage[b] = (uint8_t*) host_alloc(gp, (size_t) kPfBatch * max_blob_);
            }
            for (int r = 0; r < kPfPmapRing; ++r) {
                gp.h_pf_pmap[r] = (int*) host_alloc(gp, (size_t) c.n_expert * sizeof(int));
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
        if (mtp_loaded_ && gi == ngpu - 1) gpu_moe_bytes += mtp_.layer.blob * (size_t) c.n_expert;
        const double frac = gpu_moe_bytes > 0 ? std::min(1.0, budget / (double) gpu_moe_bytes) : 0.0;
        for (int il = std::max(gp.l0, c.n_dense_lead); il < gp.l1; ++il) {
            Layer& l = layers_[(size_t) il];
            int nr = std::min(c.n_expert, (int) std::floor(frac * c.n_expert));
            while (nr > 0 && (double) nr * (double) l.blob > budget) --nr;
            l.n_res = nr;
            budget -= (double) nr * (double) l.blob;
            l.vram_pool_off = gpu_vram_used[(size_t) gi];
            gpu_vram_used[(size_t) gi] += (size_t) nr * l.blob;
            total_vram_experts += (size_t) nr;
            total_experts += (size_t) c.n_expert;
        }
        if (mtp_loaded_ && gi == ngpu - 1) {
            Layer& ml = mtp_.layer;
            int nr = std::min(c.n_expert, (int) std::floor(frac * c.n_expert));
            while (nr > 0 && (double) nr * (double) ml.blob > budget) --nr;
            ml.n_res = nr;
            budget -= (double) nr * (double) ml.blob;
            ml.vram_pool_off = gpu_vram_used[(size_t) gi];
            gpu_vram_used[(size_t) gi] += (size_t) nr * ml.blob;
        }
        gp.vram_pool_bytes = gpu_vram_used[(size_t) gi];
        if (gp.vram_pool_bytes > 0) {
            gp.vram_pool = (uint8_t*) dev_alloc(gp, gp.vram_pool_bytes);
        }
    }

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

    const bool want_pin = (o_.pin_ram_gib > 0.0);
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
        CK(cudaEventCreateWithFlags(&l.ev_promo, cudaEventDisableTiming));
        l.h_owner.assign((size_t) std::max(1, l.n_res), -1);
        l.h_score.assign((size_t) c.n_expert, 0.0f);

        if (l.n_res > 0) {
            l.fl.vram = gp.vram_pool + l.vram_pool_off;
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

    const double vram_res_frac =
        total_experts > 0 ? (double) total_vram_experts / (double) total_experts : 1.0;
    vram_res_frac_ = vram_res_frac;
    if (o_.offload == OffloadMode::kAuto) {
        // When >= 65% of experts fit in VRAM, hit rate is ~97-99% so CUDA-graphed GPU fetch wins;
        // below 65%, async-adapting CPU lane serves misses ~3x faster than PCIe 3.0 while overlapping GPU hits.
        active_offload_ = (any_host_unpinned_ || vram_res_frac < 0.65) ? OffloadMode::kCpu : OffloadMode::kGpu;
    } else if (o_.offload == OffloadMode::kGpu && any_host_unpinned_) {
        std::fprintf(stderr, "maya-ling: host experts exceed --pin-ram-gib, falling back to CPU lane offload\n");
        active_offload_ = OffloadMode::kCpu;
    } else {
        active_offload_ = o_.offload;
    }

    if (active_offload_ == OffloadMode::kCpu || active_offload_ == OffloadMode::kMix || any_host_unpinned_) {
        maya::ds4::gguf::iq_cpu_row_hint(E);
        // Cap auto threads at 24 on large-core CPUs to avoid NUMA/barrier overhead on DDR4/DDR5 decode
        int req_th = o_.cpu_threads;
        if (req_th <= 0) {
            const int hc = (int) std::thread::hardware_concurrency();
            const int est_phys = hc >= 4 ? hc / 2 : std::max(1, hc - 1);
            req_th = std::min(est_phys, 24);
        }
        lane_ = std::make_unique<hy4::Lane>(req_th);
    }

    const int n_kda = (int) std::count(c.kda, c.kda + c.n_layers, true);
    const char* mode_name = active_offload_ == OffloadMode::kGpu   ? "gpu-cache+graph"
                            : active_offload_ == OffloadMode::kMix ? "gpu-cache+cpu-lane"
                                                                   : "vram+cpu-lane";
    size_t vram_sum = std::accumulate(gpu_vram_used.begin(), gpu_vram_used.end(), size_t(0));
    if (!o_.mmproj_path.empty()) {
        vision_ = std::make_unique<VisionTower>();
        std::string verr;
        if (!vision_->load(o_.mmproj_path, gpus_[0].dev, verr)) {
            throw std::runtime_error(verr);
        }
    }

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
        for (int wi = 0; wi < kMaxWin; ++wi) {
            if (gp.wgraph_exec[wi] != nullptr) cudaGraphExecDestroy(gp.wgraph_exec[wi]);
            if (gp.wgraph[wi] != nullptr) cudaGraphDestroy(gp.wgraph[wi]);
        }
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
    for (auto& l : layers_) {
        if (l.ev_promo != nullptr) {
            set_dev(gpus_[(size_t) l.dev_idx].dev);
            cudaEventDestroy(l.ev_promo);
        }
        free_layer_host(l);
    }
    if (mtp_loaded_) free_layer_host(mtp_.layer);

    for (auto& gp : gpus_) {
        set_dev(gp.dev);
        if (gp.d_mrope_pos != nullptr) cudaFree(gp.d_mrope_pos);
        for (void* p : gp.dev_allocs) cudaFree(p);
        for (void* p : gp.host_allocs) cudaFreeHost(p);
        for (int b = 0; b < 2; ++b) {
            if (gp.ev_copy[b] != nullptr) cudaEventDestroy(gp.ev_copy[b]);
            if (gp.ev_comp[b] != nullptr) cudaEventDestroy(gp.ev_comp[b]);
        }
        if (gp.ev_xfer != nullptr) cudaEventDestroy(gp.ev_xfer);
        if (gp.ev_route != nullptr) cudaEventDestroy(gp.ev_route);
        if (gp.ev_d2h != nullptr) cudaEventDestroy(gp.ev_d2h);
        if (gp.ev_norm != nullptr) cudaEventDestroy(gp.ev_norm);
        if (gp.ev_sh != nullptr) cudaEventDestroy(gp.ev_sh);
        if (gp.blas != nullptr) cublasDestroy(gp.blas);
        if (gp.s_d2h != nullptr) cudaStreamDestroy(gp.s_d2h);
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
    mtp_done_ = 0;
    mtp_tokens_.clear();
    rope_delta_ = 0;
    images_.clear();
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

void Model::transfer_bytes(int g_from, int g_to, const void* p_src, void* p_dst, size_t bytes) {
    if (bytes == 0) return;
    Gpu& src = gpus_[(size_t) g_from];
    Gpu& dst = gpus_[(size_t) g_to];
    if (p2p_mat_[(size_t) g_from][(size_t) g_to]) {
        set_dev(src.dev);
        CK(cudaEventRecord(src.ev_xfer, src.s));
        set_dev(dst.dev);
        CK(cudaStreamWaitEvent(dst.s, src.ev_xfer, 0));
        CK(cudaMemcpyPeerAsync(p_dst, dst.dev, p_src, src.dev, bytes, dst.s));
        CK(cudaEventRecord(dst.ev_xfer, dst.s));
        set_dev(src.dev);
        CK(cudaStreamWaitEvent(src.s, dst.ev_xfer, 0));
        set_dev(dst.dev);
        return;
    }
    const size_t chunk = (size_t) std::max(1, src.pf_cap) * (size_t) c_.n_embd * sizeof(float);
    const uint8_t* sp = (const uint8_t*) p_src;
    uint8_t* dp = (uint8_t*) p_dst;
    for (size_t off = 0; off < bytes; off += chunk) {
        const size_t n = std::min(chunk, bytes - off);
        set_dev(src.dev);
        CK(cudaMemcpyAsync(src.h_xfer, sp + off, n, cudaMemcpyDeviceToHost, src.s));
        CK(cudaEventRecord(src.ev_xfer, src.s));
        set_dev(dst.dev);
        CK(cudaStreamWaitEvent(dst.s, src.ev_xfer, 0));
        CK(cudaMemcpyAsync(dp + off, src.h_xfer, n, cudaMemcpyHostToDevice, dst.s));
        CK(cudaEventRecord(dst.ev_xfer, dst.s));
        set_dev(src.dev);
        CK(cudaStreamWaitEvent(src.s, dst.ev_xfer, 0));
        set_dev(dst.dev);
    }
}

void Model::transfer_buf(int g_from, int g_to, const float* p_src, float* p_dst, size_t floats) {
    transfer_bytes(g_from, g_to, p_src, p_dst, floats * sizeof(float));
}

void Model::transfer_x(int g_from, int g_to, int T, bool prefill_buf) {
    Gpu& src = gpus_[(size_t) g_from];
    Gpu& dst = gpus_[(size_t) g_to];
    transfer_buf(g_from, g_to, prefill_buf ? src.pf_x : src.x, prefill_buf ? dst.pf_x : dst.x,
                 (size_t) T * (size_t) c_.n_embd);
}

// ------------------------------ single-token decode ------------------------------

void Model::kda_decode(Gpu& gp, Layer& l, int il, bool seam, bool add_to_x) {
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
    if (add_to_x) hy::add(gp.x, gp.out, (size_t) c.n_embd, s);
}

void Model::mla_decode(Gpu& gp, Layer& l, int il, bool seam, bool add_to_x) {
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

    k::mla_q_prep(gp.qm, 1, H, c.qk_head, c.qk_nope, R, gp.d_step + 1, 0, c.rope_base, gp.q8n, gp.d8n, nullptr, s, gp.d_step + 3);
    iq::MvJob kb{l.wkb.p, l.wkb.ld, l.wkb.type, c.qk_nope, KL, 1, gp.q8n, gp.d8n, gp.q_abs, (size_t) KL, nullptr, 1.0f};
    if (!iq::mv_heads(kb, H, (size_t) KL * l.wkb.ld, (size_t) c.qk_nope, (size_t) (c.qk_nope / 32), (size_t) KL, s))
        throw std::runtime_error("mla: k_b type not supported");

    k::kv_store_split(gp.kvall, 1, KL, R, l.kv_a_norm, c.eps, gp.d_step + 1, 0, c.rope_base, l.lat, o_.kv_q8, l.k_pe, s, gp.d_step + 3);

    if (!k::mla_dec_wmma(gp.q_abs, gp.qm, c.qk_head, c.qk_nope, l.lat, o_.kv_q8 ? 1 : 0, l.k_pe, o_.ctx,
                         gp.d_step + 2, 1, H, KL, R, 1.0f / sqrtf((float) c.qk_head), gp.mla_part, gp.ctxo, s)) {
        if (!glmd::mla_dec(gp.q_abs, gp.qm, c.qk_head, c.qk_nope, l.lat, o_.kv_q8 ? 1 : 0, l.k_pe, gp.d_iota, o_.ctx,
                           gp.d_step + 2, H, KL, R, 1.0f / sqrtf((float) c.qk_head), gp.mla_part, gp.ctxo, s))
            throw std::runtime_error("mla: mla_dec refused shape");
    }

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
    if (add_to_x) hy::add(gp.x, gp.out, (size_t) c.n_embd, s);
}

void Model::ffn_dense_decode(Gpu& gp, Layer& l, int il, bool seam, const float* next_norm) {
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
    if (next_norm != nullptr && !seam) {
        k::add_rmsnorm_q8(gp.x, gp.out, next_norm, 1, c.n_embd, c.eps, gp.cur, gp.xq, gp.xd, s);
    } else {
        hy::add(gp.x, gp.out, (size_t) c.n_embd, s);
    }
}

void Model::ffn_moe_decode(Gpu& gp, Layer& l, int il, bool seam, const float* next_norm) {
    const Config& c = c_;
    const int E = c.n_embd, FE = c.n_ff_exp, FS = c.n_ff_shexp, NU = c.n_used;
    cudaStream_t s = gp.s;

    const bool use_lane = (l.n_res < c.n_expert) &&
                          (active_offload_ == OffloadMode::kCpu || active_offload_ == OffloadMode::kMix || !l.host_pinned);

    auto run_shared = [&](cudaStream_t st) {
        iq::MvJob sj[2] = {
            {l.fg.p, l.fg.ld, l.fg.type, E, FS, 1, gp.xq, gp.xd, gp.gu, (size_t) FS, nullptr, 1.0f},
            {l.fu.p, l.fu.ld, l.fu.type, E, FS, 1, gp.xq, gp.xd, gp.gu + FS, (size_t) FS, nullptr, 1.0f},
        };
        if (!iq::mv(sj, 2, st)) throw std::runtime_error("moe: shared gate/up type not supported");
        k::swiglu_clamp_out(gp.gu, gp.gu + FS, 1, FS, c.clamp_shexp[il], seam ? gp.shact : nullptr, gp.shq, gp.shd,
                            nullptr, st);
        if (seam && c.clamp_shexp[il] > 1e-6f)
            dump_seam_f32(gp, "ffn_swiglu_limited-" + std::to_string(il), gp.shact, (size_t) FS);
        iq::MvJob sd{l.fd.p, l.fd.ld, l.fd.type, FS, E, 1, gp.shq, gp.shd, gp.shout, (size_t) E, nullptr, 1.0f};
        if (!iq::mv(&sd, 1, st)) throw std::runtime_error("moe: shared down type not supported");
    };

    auto finish_add = [&] {
        if (seam) {
            dump_seam_f32(gp, "ffn_moe_out-" + std::to_string(il), gp.moe, (size_t) E);
            dump_seam_f32(gp, "ffn_shexp-" + std::to_string(il), gp.shout, (size_t) E);
        }
        if (next_norm != nullptr && !seam) {
            k::add2_rmsnorm_q8(gp.x, gp.moe, gp.shout, next_norm, 1, E, c.eps, gp.cur, gp.xq, gp.xd, s);
        } else {
            k::add2(gp.x, gp.moe, gp.shout, E, s);
        }
    };

    // In kCpu mode, start copying gp.cur (10 KB) to host on dedicated s_d2h IMMEDIATELY (overlaps router + fetch_resolve!)
    if (use_lane) {
        CK(cudaEventRecord(gp.ev_norm, s));
        CK(cudaStreamWaitEvent(gp.s_d2h, gp.ev_norm, 0));
        CK(cudaMemcpyAsync(gp.h_cur, gp.cur, (size_t) E * sizeof(float), cudaMemcpyDeviceToHost, gp.s_d2h));
    } else if (!seam) {
        // In kGpu / full-VRAM mode, overlap shared expert on s_copy with router + fetch_resolve + fetch_copy!
        CK(cudaEventRecord(gp.ev_norm, s));
        CK(cudaStreamWaitEvent(gp.s_copy, gp.ev_norm, 0));
        run_shared(gp.s_copy);
        CK(cudaEventRecord(gp.ev_sh, gp.s_copy));
    }

    if (l.router_type == 0)
        hy::gemv_f32w((const float*) l.router_w, (size_t) E, E, c.n_expert, gp.cur, (size_t) E, 1, gp.rlog,
                      (size_t) c.n_expert, 1.0f, s);
    else
        hy::gemv_bf16w((const uint16_t*) l.router_w, (size_t) E, E, c.n_expert, gp.cur, (size_t) E, 1, gp.rlog,
                       (size_t) c.n_expert, 1.0f, s);

    k::router_grouped(gp.rlog, l.rb_dev, 1, c.n_expert, NU, c.n_group, c.n_group_used, c.w_norm, c.w_scale, gp.d_ids,
                      gp.d_w, s);
    if (seam) dump_seam_i32(gp, "ffn_moe_topk-" + std::to_string(il), gp.d_ids, (size_t) NU);

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
        if (seam) run_shared(s);
        else CK(cudaStreamWaitEvent(s, gp.ev_sh, 0));
        finish_add();
        return;
    }

    if (!use_lane) {
        k::fetch_resolve_fast(gp.d_ids, 1, NU, l.fl, gp.fxs, -1, l.n_res > 0, 0.98f, s);
        hy::fetch_copy(gp.fxs, l.blob, 32, s);
        hy::plan_experts(gp.d_ids, NU, nullptr, 0, gp.d_plan, s, nullptr, gp.d_w, gp.fxs.eptr, gp.fxs.who);
        if (!iq::gate_up(a, NU, 1, s)) throw std::runtime_error("moe: gate_up failed");
        if (seam && c.clamp_exp[il] > 1e-6f)
            dump_seam_f32(gp, "ffn_moe_swiglu_limited-" + std::to_string(il), gp.eh, (size_t) NU * FE);
        hy::q8_rows(gp.eh, (size_t) FE, NU, FE, gp.ehq, gp.ehd, s);
        if (!iq::down(a, NU, 1, s)) throw std::runtime_error("moe: down failed");
        hy::moe_sum(gp.part, 1, NU, E, gp.d_w, gp.moe, s);
        if (seam) run_shared(s);
        else CK(cudaStreamWaitEvent(s, gp.ev_sh, 0));
        finish_add();
        return;
    }

    // Complete any finished race-free async background VRAM promotion from an earlier token (zero stall!)
    if (l.promo_pending_e >= 0 && cudaEventQuery(l.ev_promo) == cudaSuccess) {
        const int pe = l.promo_pending_e, psl = l.promo_pending_sl;
        l.h_slot[(size_t) pe] = psl;
        l.h_owner[(size_t) psl] = pe;
        CK(cudaMemcpyAsync(l.fl.slot + pe, &l.h_slot[(size_t) pe], sizeof(int), cudaMemcpyHostToDevice, s));
        l.promo_pending_e = -1;
        l.promo_pending_v = -1;
        l.promo_pending_sl = -1;
        ++cpu_admitted_;
    }

    const int k_gpu = (active_offload_ == OffloadMode::kMix && l.host_pinned) ? o_.mix_k_gpu : 0;
    const bool adapt = (k_gpu > 0 && l.n_res > 0);
    k::fetch_resolve_fast(gp.d_ids, 1, NU, l.fl, gp.fxs, k_gpu, adapt, 0.98f, s);
    k::pack_route_cpu(gp.d_ids, gp.d_w, gp.fxs.who, NU, gp.d_route_pack, s);
    CK(cudaEventRecord(gp.ev_route, s));

    // Single 96-byte D2H transfer on dedicated s_d2h (NEVER blocked by s_copy background H2D promos!)
    CK(cudaStreamWaitEvent(gp.s_d2h, gp.ev_route, 0));
    CK(cudaMemcpyAsync(gp.h_route_pack, gp.d_route_pack, sizeof(k::RoutePack), cudaMemcpyDeviceToHost, gp.s_d2h));
    CK(cudaEventRecord(gp.ev_d2h, gp.s_d2h));

    if (k_gpu > 0) hy::fetch_copy(gp.fxs, l.blob, 32, s);
    hy::plan_experts(gp.d_ids, NU, nullptr, 0, gp.d_plan, s, nullptr, gp.d_w, gp.fxs.eptr, gp.fxs.who);
    if (!iq::gate_up(a, NU, 1, s)) throw std::runtime_error("moe: gate_up failed");
    if (seam && c.clamp_exp[il] > 1e-6f)
        dump_seam_f32(gp, "ffn_moe_swiglu_limited-" + std::to_string(il), gp.eh, (size_t) NU * FE);
    hy::q8_rows(gp.eh, (size_t) FE, NU, FE, gp.ehq, gp.ehd, s);
    if (!iq::down(a, NU, 1, s)) throw std::runtime_error("moe: down failed");
    hy::moe_sum(gp.part, 1, NU, E, gp.d_w, gp.moe, s);
    run_shared(s);

    CK(cudaEventSynchronize(gp.ev_d2h));
    const k::RoutePack& rp = *gp.h_route_pack;
    std::vector<hy4::CpuEntry> entries;
    entries.reserve((size_t) NU);
    for (int j2 = 0; j2 < NU; ++j2) {
        if (rp.who[j2] == 1) entries.push_back({0, rp.ids[j2], rp.w[j2]});
    }
    if (!entries.empty()) {
        cpu_lane_picks_ += (unsigned long long) entries.size();
        hy4::cpu_experts(*lane_, l.cex, entries, gp.h_cur, 1, gp.h_cpu_out);
        CK(cudaMemcpyAsync(gp.d_cpu_out, gp.h_cpu_out, (size_t) E * sizeof(float), cudaMemcpyHostToDevice, s));
        hy::add(gp.moe, gp.d_cpu_out, (size_t) E, s);
    }

    // Race-free async background VRAM cache admission across all layers (round-robin, dedicated s_copy!)
    if (k_gpu == 0 && l.n_res > 0 && l.n_res < c.n_expert && l.host_blobs != nullptr && l.host_pinned) {
        if (!l.cpu_cache_ready) {
            if ((int) l.h_owner.size() != l.n_res) l.h_owner.assign((size_t) l.n_res, -1);
            if ((int) l.h_score.size() != c.n_expert) l.h_score.assign((size_t) c.n_expert, 0.0f);
            for (int e = 0; e < c.n_expert; ++e) {
                const int sl = l.h_slot[(size_t) e];
                if (sl >= 0 && sl < l.n_res) l.h_owner[(size_t) sl] = e;
            }
            l.cpu_cache_ready = true;
        }
        const int NE = c.n_expert;
        for (int e = 0; e < NE; ++e) l.h_score[(size_t) e] *= 0.98f;
        uint8_t active_mask[512] = {};
        for (int j2 = 0; j2 < NU; ++j2) {
            const int e = rp.ids[j2];
            l.h_score[(size_t) e] += 1.0f;
            if (e >= 0 && e < 512) active_mask[e] = 1;
        }
        const int layer_dist = (il - promo_rr_ + c.n_layers) % c.n_layers;
        const bool rr_eligible = (tok_promo_budget_ > 0) && (layer_dist < 40);
        if (!entries.empty() && l.promo_pending_e < 0 && rr_eligible) {
            int best_m = -1;
            float best_ms = -1.0f;
            for (const auto& ce : entries) {
                const float sc_e = l.h_score[(size_t) ce.e];
                if (sc_e > best_ms) {
                    best_ms = sc_e;
                    best_m = ce.e;
                }
            }
            int worst_v = -1, worst_sl = -1;
            float worst_vs = 1e30f;
            for (int sl = 0; sl < l.n_res; ++sl) {
                const int ve = l.h_owner[(size_t) sl];
                if (ve < 0 || (ve < 512 && active_mask[ve])) continue;
                const float sv = l.h_score[(size_t) ve];
                if (sv < worst_vs) {
                    worst_vs = sv;
                    worst_v = ve;
                    worst_sl = sl;
                }
            }
            if (best_m >= 0 && worst_v >= 0 && best_ms > worst_vs) {
                // Race-free eviction: mark victim slot=-1 on s BEFORE s_copy overwrites worst_sl!
                l.h_slot[(size_t) worst_v] = -1;
                l.h_owner[(size_t) worst_sl] = -1;
                CK(cudaMemcpyAsync(l.fl.slot + worst_v, gp.h_neg1, sizeof(int), cudaMemcpyHostToDevice, s));
                CK(cudaEventRecord(gp.ev_comp[0], s));
                CK(cudaStreamWaitEvent(gp.s_copy, gp.ev_comp[0], 0));
                CK(cudaMemcpyAsync(l.fl.vram + (size_t) worst_sl * l.blob, l.host_blobs + (size_t) best_m * l.blob,
                                   l.blob, cudaMemcpyHostToDevice, gp.s_copy));
                CK(cudaEventRecord(l.ev_promo, gp.s_copy));
                l.promo_pending_e = best_m;
                l.promo_pending_v = worst_v;
                l.promo_pending_sl = worst_sl;
                --tok_promo_budget_;
            }
        }
    }

    finish_add();
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

    bool normed = false;
    for (int il = gp.l0; il < gp.l1; ++il) {
        Layer& l = layers_[(size_t) il];
        if (!normed) {
            hy::rmsnorm(gp.x, (size_t) E, 1, E, l.attn_norm, c.eps, gp.cur, (size_t) E, s, {gp.xq, gp.xd});
        }
        if (seam) dump_seam_f32(gp, "attn_norm-" + std::to_string(il), gp.cur, (size_t) E);
        const bool fuse_attn = !seam;
        if (l.kda) kda_decode(gp, l, il, seam, !fuse_attn);
        else mla_decode(gp, l, il, seam, !fuse_attn);
        if (fuse_attn) {
            k::add_rmsnorm_q8(gp.x, gp.out, l.ffn_norm, 1, E, c.eps, gp.cur, gp.xq, gp.xd, s);
        } else {
            if (seam) dump_seam_f32(gp, "ffn_inp-" + std::to_string(il), gp.x, (size_t) E);
            hy::rmsnorm(gp.x, (size_t) E, 1, E, l.ffn_norm, c.eps, gp.cur, (size_t) E, s, {gp.xq, gp.xd});
        }
        const float* next_norm = nullptr;
        if (!seam) {
            if (il + 1 < gp.l1) next_norm = layers_[(size_t) (il + 1)].attn_norm;
            else if (gi == (int) gpus_.size() - 1) next_norm = out_norm_;
        }
        if (il < c.n_dense_lead) ffn_dense_decode(gp, l, il, seam, next_norm);
        else ffn_moe_decode(gp, l, il, seam, next_norm);
        normed = (next_norm != nullptr);
        if (seam) dump_seam_f32(gp, "l_out-" + std::to_string(il), gp.x, (size_t) E);
    }

    if (gi == (int) gpus_.size() - 1) {
        if (!normed) {
            hy::rmsnorm(gp.x, (size_t) E, 1, E, out_norm_, c.eps, gp.cur, (size_t) E, s, {gp.xq, gp.xd});
        }
        if (mtp_loaded_) CK(cudaMemcpyAsync(gp.hn, gp.cur, (size_t) E * 4, cudaMemcpyDeviceToDevice, s));
        if (seam) dump_seam_f32(gp, "result_norm", gp.cur, (size_t) E);
        iq::MvJob j{out_w_.p, out_w_.ld, out_w_.type, out_w_.k, out_w_.n, 1, gp.xq, gp.xd, gp.logits, (size_t) out_w_.n,
                    nullptr, 1.0f};
        if (!iq::mv(&j, 1, s)) throw std::runtime_error("output: type not supported");
    }
}

const float* Model::step(int token, int pos) {
    tok_promo_budget_ = 32;
    promo_rr_ = c_.n_dense_lead + ((pos * 7) % std::max(1, c_.n_layers - c_.n_dense_lead));
    const int ngpu = (int) gpus_.size();
    const bool graph_ok = o_.use_graph && active_offload_ == OffloadMode::kGpu && !any_host_unpinned_ &&
                          o_.seam_dir.empty();

    for (int gi = 0; gi < ngpu; ++gi) {
        Gpu& gp = gpus_[(size_t) gi];
        set_dev(gp.dev);
        k::set_step(gp.d_step, token, pos, pos + 1, rope_delta_, gp.s);
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
    if (mtp_loaded_) {
        Gpu& gl = gpus_.back();
        set_dev(gl.dev);
        k::store_hring(gl.hn, 1, c_.n_embd, mtp_.hring, MtpWeights::kRing, pos, gl.s);
        if ((int) mtp_tokens_.size() <= pos) mtp_tokens_.resize((size_t) (pos + 1), 0);
        mtp_tokens_[(size_t) pos] = token;
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

void Model::kda_prefill(Gpu& gp, Layer& l, int T, float* x_io) {
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
    hy::add(x_io, gp.pf_out, (size_t) T * E, s);
}

void Model::mla_prefill(Gpu& gp, Layer& l, int T, int pos0, float* x_io, const int* mrope_pos, int mrope_stride) {
    const int mhw = vision_ ? (vision_->geo().mrope_section[1] + vision_->geo().mrope_section[2]) : 24;
    const Config& c = c_;
    const int E = c.n_embd, H = c.n_head, KL = c.kv_lora, R = c.n_rot;
    const int QN = c.qk_nope, VH = c.v_head;
    cudaStream_t s = gp.s;

    auto align_ptr = [](uint8_t* p) -> uint8_t* {
        return (uint8_t*) (((uintptr_t) p + 255u) & ~uintptr_t(255u));
    };

    const int S_total = pos0 + T;
    // Unabsorbed tiled cuBLAS dense MLA wins massively whenever T >= 64 or total context <= 2048;
    // for tiny agent re-reads (T < 64) at deep context (> 2048), absorbed avoids expanding full KV.
    const bool use_tiled = (T >= 64 || S_total <= 2048);

    if (!use_tiled) {
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
        curp = align_ptr(curp + (size_t) T * H * VH * 4);
        float* pf_gate = (float*) curp;

        hy::to_f16(gp.pf_cur, (size_t) E, T, E, gp.pf_x16, (size_t) E, s);
        gemm_mat(gp, l.wqm, gp.pf_x16, T, pf_qm);
        gemm_mat(gp, l.wkva, gp.pf_x16, T, pf_kvall);
        gemm_mat(gp, l.wgate, gp.pf_x16, T, pf_gate);

        k::mla_q_prep(pf_qm, T, H, c.qk_head, QN, R, nullptr, pos0, c.rope_base, nullptr, nullptr, pf_qm16, s, nullptr, mrope_pos, mhw, mrope_stride);

        if (!iq::f16_rows(l.wkb.type, l.wkb.p, QN, KL * H, gp.pf_w16, QN, s))
            throw std::runtime_error("mla prefill: k_b f16_rows failed");
        const float alpha = 1.0f, beta = 0.0f;
        CKBLAS(cublasGemmStridedBatchedEx(gp.blas, CUBLAS_OP_T, CUBLAS_OP_N, KL, T, QN, &alpha, gp.pf_w16, CUDA_R_16F,
                                          QN, (long long) KL * QN, pf_qm16, CUDA_R_16F, H * c.qk_head,
                                          (long long) c.qk_head, &beta, pf_qabs, CUDA_R_32F, H * KL, (long long) KL, H,
                                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

        k::kv_store_split(pf_kvall, T, KL, R, l.kv_a_norm, c.eps, nullptr, pos0, c.rope_base, l.lat, o_.kv_q8, l.k_pe,
                          s, nullptr, mrope_pos, mhw, mrope_stride);

        glmd::MlaPrompt mp;
        mp.q_abs = pf_qabs;
        mp.q = pf_qm;
        mp.q_stride = c.qk_head;
        mp.q_pe_off = QN;
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
        if (!iq::f16_rows(l.wvb.type, l.wvb.p, KL, VH * H, gp.pf_w16, KL, s))
            throw std::runtime_error("mla prefill: v_b f16_rows failed");
        CKBLAS(cublasGemmStridedBatchedEx(gp.blas, CUBLAS_OP_T, CUBLAS_OP_N, VH, T, KL, &alpha, gp.pf_w16, CUDA_R_16F,
                                          KL, (long long) VH * KL, pf_ctx16, CUDA_R_16F, H * KL, (long long) KL, &beta,
                                          pf_o128, CUDA_R_32F, H * VH, (long long) VH, H, CUBLAS_COMPUTE_32F,
                                          CUBLAS_GEMM_DEFAULT));

        k::head_gate_out(pf_o128, pf_gate, T, H, VH, nullptr, nullptr, gp.pf_x16, s);
        gemm_mat(gp, l.wom, gp.pf_x16, T, gp.pf_out);
        hy::add(x_io, gp.pf_out, (size_t) T * E, s);
        return;
    }

    // ---- Unabsorbed Tiled cuBLAS Dense MLA ----
    const int BQ_tile = std::min(T, 256);
    const int BK_tile = std::min(S_total, 1024);

    uint8_t* curp = (uint8_t*) gp.pf_tmp;
    float* pf_qm = (float*) curp;
    curp = align_ptr(curp + (size_t) T * H * c.qk_head * 4);
    float* pf_kvall = (float*) curp;
    curp = align_ptr(curp + (size_t) T * (KL + R) * 4);
    float* pf_gate = (float*) curp;
    curp = align_ptr(curp + (size_t) T * H * 4);
    uint16_t* q_nope16 = (uint16_t*) curp;
    curp = align_ptr(curp + (size_t) H * T * QN * 2);
    uint16_t* q_pe16 = (uint16_t*) curp;
    curp = align_ptr(curp + (size_t) H * T * R * 2);
    uint16_t* wkb16 = (uint16_t*) curp;
    curp = align_ptr(curp + (size_t) H * KL * QN * 2);
    uint16_t* wvb16 = (uint16_t*) curp;
    curp = align_ptr(curp + (size_t) H * VH * KL * 2);
    uint16_t* lat16_buf = (uint16_t*) curp;
    curp = align_ptr(curp + (size_t) BK_tile * KL * 2);
    uint16_t* K_nope16 = (uint16_t*) curp;
    curp = align_ptr(curp + (size_t) H * BK_tile * QN * 2);
    uint16_t* V16 = (uint16_t*) curp;
    curp = align_ptr(curp + (size_t) H * BK_tile * VH * 2);
    float* S_buf = (float*) curp;
    curp = align_ptr(curp + (size_t) H * BQ_tile * BK_tile * 4);
    uint16_t* P16_buf = (uint16_t*) curp;
    curp = align_ptr(curp + (size_t) H * BQ_tile * BK_tile * 2);
    float* O_acc = (float*) curp;
    curp = align_ptr(curp + (size_t) H * T * VH * 4);
    float* m_buf = (float*) curp;
    curp = align_ptr(curp + (size_t) H * T * 4);
    float* l_buf = (float*) curp;

    hy::to_f16(gp.pf_cur, (size_t) E, T, E, gp.pf_x16, (size_t) E, s);
    gemm_mat(gp, l.wqm, gp.pf_x16, T, pf_qm);
    gemm_mat(gp, l.wkva, gp.pf_x16, T, pf_kvall);
    gemm_mat(gp, l.wgate, gp.pf_x16, T, pf_gate);

    k::mla_q_split_rope_f16(pf_qm, T, H, c.qk_head, QN, R, pos0, c.rope_base, q_nope16, q_pe16, s, mrope_pos, mhw, mrope_stride);
    k::kv_store_split(pf_kvall, T, KL, R, l.kv_a_norm, c.eps, nullptr, pos0, c.rope_base, l.lat, o_.kv_q8, l.k_pe, s, nullptr, mrope_pos, mhw, mrope_stride);

    if (!iq::f16_rows(l.wkb.type, l.wkb.p, QN, KL * H, wkb16, QN, s))
        throw std::runtime_error("mla prefill: k_b f16_rows failed");
    if (!iq::f16_rows(l.wvb.type, l.wvb.p, KL, VH * H, wvb16, KL, s))
        throw std::runtime_error("mla prefill: v_b f16_rows failed");

    k::mla_init_state(m_buf, l_buf, O_acc, H, T, VH, s);

    const float one = 1.0f, zero = 0.0f;
    const float scale = 1.0f / sqrtf((float) c.qk_head);

    for (int k0 = 0; k0 < S_total; k0 += BK_tile) {
        const int BK = std::min(BK_tile, S_total - k0);
        const uint16_t* lat16_cur = nullptr;
        if (o_.kv_q8) {
            k::lat_q8_to_f16(l.lat, k0, BK, KL, lat16_buf, s);
            lat16_cur = lat16_buf;
        } else {
            lat16_cur = (const uint16_t*) l.lat + (size_t) k0 * KL;
        }

        // Expand K_nope[H, BK, QN] = wkb[H, KL, QN]^T_col * lat[BK, KL]
        CKBLAS(cublasGemmStridedBatchedEx(gp.blas, CUBLAS_OP_N, CUBLAS_OP_N, QN, BK, KL, &one, wkb16, CUDA_R_16F, QN,
                                          (long long) KL * QN, lat16_cur, CUDA_R_16F, KL, 0LL, &zero, K_nope16,
                                          CUDA_R_16F, QN, (long long) BK * QN, H, CUBLAS_COMPUTE_32F,
                                          CUBLAS_GEMM_DEFAULT));
        // Expand V[H, BK, VH] = wvb[H, VH, KL] * lat[BK, KL]
        CKBLAS(cublasGemmStridedBatchedEx(gp.blas, CUBLAS_OP_T, CUBLAS_OP_N, VH, BK, KL, &one, wvb16, CUDA_R_16F, KL,
                                          (long long) VH * KL, lat16_cur, CUDA_R_16F, KL, 0LL, &zero, V16, CUDA_R_16F,
                                          VH, (long long) BK * VH, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

        const uint16_t* kpe_cur = l.k_pe + (size_t) k0 * R;

        for (int q0 = 0; q0 < T; q0 += BQ_tile) {
            const int BQ = std::min(BQ_tile, T - q0);
            if (pos0 + q0 + BQ - 1 < k0) continue;   // strictly upper-triangular causal skip

            // S[H, BQ, BK] = scale * (K_nope @ Q_nope^T + K_pe @ Q_pe^T)
            CKBLAS(cublasGemmStridedBatchedEx(gp.blas, CUBLAS_OP_T, CUBLAS_OP_N, BK, BQ, QN, &scale, K_nope16,
                                              CUDA_R_16F, QN, (long long) BK * QN, q_nope16 + (size_t) q0 * QN,
                                              CUDA_R_16F, QN, (long long) T * QN, &zero, S_buf, CUDA_R_32F, BK,
                                              (long long) BQ * BK, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            CKBLAS(cublasGemmStridedBatchedEx(gp.blas, CUBLAS_OP_T, CUBLAS_OP_N, BK, BQ, R, &scale, kpe_cur, CUDA_R_16F,
                                              R, 0LL, q_pe16 + (size_t) q0 * R, CUDA_R_16F, R, (long long) T * R, &one,
                                              S_buf, CUDA_R_32F, BK, (long long) BQ * BK, H, CUBLAS_COMPUTE_32F,
                                              CUBLAS_GEMM_DEFAULT));

            k::mla_online_softmax(S_buf, H, T, q0, BQ, k0, BK, pos0, m_buf, l_buf, O_acc, VH, P16_buf, s);

            // O_acc[H, q0..q0+BQ, VH] += V[H, BK, VH] * P16[H, BQ, BK]
            CKBLAS(cublasGemmStridedBatchedEx(gp.blas, CUBLAS_OP_N, CUBLAS_OP_N, VH, BQ, BK, &one, V16, CUDA_R_16F, VH,
                                              (long long) BK * VH, P16_buf, CUDA_R_16F, BK, (long long) BQ * BK, &one,
                                              O_acc + (size_t) q0 * VH, CUDA_R_32F, VH, (long long) T * VH, H,
                                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }
    }

    k::mla_o_finalize_gate(O_acc, l_buf, pf_gate, T, H, VH, gp.pf_x16, s);
    gemm_mat(gp, l.wom, gp.pf_x16, T, gp.pf_out);
    hy::add(x_io, gp.pf_out, (size_t) T * E, s);
}

void Model::ffn_dense_prefill(Gpu& gp, Layer& l, int il, int T, float* x_io) {
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
    hy::add(x_io, gp.pf_out, (size_t) T * E, s);
}

void Model::ffn_moe_prefill_macro(int gi, Layer& l, int il, int B, bool is_last_block, bool layer_borrowed,
                                  const std::vector<MacroBufs>& mbufs) {
    Gpu& gp = gpus_[(size_t) gi];
    const MacroBufs& mb = mbufs[(size_t) gi];
    const Config& c = c_;
    const int E = c.n_embd, FE = c.n_ff_exp, FS = c.n_ff_shexp, NU = c.n_used, NE = c.n_expert;
    const int P = gp.pf_cap;
    const int ngpu = (int) gpus_.size();
    const int max_set_rows = P * NU;
    const int n_res_eff = layer_borrowed ? 0 : l.n_res;
    cudaStream_t s = gp.s;
    cudaStream_t sc = gp.s_copy;

    auto align_ptr = [](uint8_t* p) -> uint8_t* {
        return (uint8_t*) (((uintptr_t) p + 255u) & ~uintptr_t(255u));
    };
    auto scratch_ptrs = [&](Gpu& g, uint16_t*& h16, float*& y) {
        uint8_t* curp = (uint8_t*) g.pf_tmp;
        curp = align_ptr(curp + (size_t) P * NE * 4);
        h16 = (uint16_t*) curp;
        curp = align_ptr(curp + (size_t) max_set_rows * FE * 2);
        y = (float*) curp;
    };

    set_dev(gp.dev);

    // 1. RMSNorm + Router across all B tokens (tiled by P to reuse pf_tmp for router logits)
    for (int t0 = 0; t0 < B; t0 += P) {
        const int T = std::min(P, B - t0);
        float* x_t = mb.x + (size_t) t0 * E;
        float* cur_t = mb.cur + (size_t) t0 * E;
        int* ids_t = mb.ids + (size_t) t0 * NU;
        float* w_t = mb.w + (size_t) t0 * NU;
        float* pf_rlog = gp.pf_tmp;
        hy::rmsnorm(x_t, (size_t) E, T, E, l.ffn_norm, c.eps, cur_t, (size_t) E, s);
        if (l.router_type == 0) {
            const float alpha = 1.0f, beta = 0.0f;
            CKBLAS(cublasSgemm(gp.blas, CUBLAS_OP_T, CUBLAS_OP_N, NE, T, E, &alpha, (const float*) l.router_w, E, cur_t,
                               E, &beta, pf_rlog, NE));
        } else {
            hy::gemv_bf16w((const uint16_t*) l.router_w, (size_t) E, E, NE, cur_t, (size_t) E, T, pf_rlog, (size_t) NE,
                           1.0f, s);
        }
        k::router_grouped(pf_rlog, l.rb_dev, T, NE, NU, c.n_group, c.n_group_used, c.w_norm, c.w_scale, ids_t, w_t, s);
    }

    // 2. Read back routing IDs and identify hits vs misses
    std::vector<int> miss_list;
    std::vector<int> chunk_counts((size_t) NE, 0);
    int vram_hits = 0;
    int vram_max_rows = 0;
    if (n_res_eff < NE) {
        if (n_res_eff > 0) {
            CK(cudaMemcpyAsync(l.h_slot.data(), l.fl.slot, (size_t) NE * sizeof(int), cudaMemcpyDeviceToHost, s));
        }
        CK(cudaMemcpyAsync(gp.h_pf_ids, mb.ids, (size_t) B * NU * sizeof(int), cudaMemcpyDeviceToHost, s));
        if (lane_ && B <= 48) {
            CK(cudaMemcpyAsync(gp.h_pf_w, mb.w, (size_t) B * NU * sizeof(float), cudaMemcpyDeviceToHost, s));
        }
        CK(cudaStreamSynchronize(s));
        // Recency-weight prompt expert counts across macro-blocks so tail document warms decode cache
        for (int e = 0; e < NE; ++e) l.pf_counts[(size_t) e] >>= 1;
        for (int i = 0; i < B * NU; ++i) {
            const int e = gp.h_pf_ids[i];
            l.pf_counts[(size_t) e]++;
            const int cnt = ++chunk_counts[(size_t) e];
            if (n_res_eff > 0 && l.h_slot[(size_t) e] >= 0) {
                ++vram_hits;
                if (cnt > vram_max_rows) vram_max_rows = cnt;
            } else if (cnt == 1) {
                miss_list.push_back(e);
            }
        }
    } else {
        vram_hits = B * NU;
    }

    // 3. Inline D2D cache warming plan on last macro-block (skipped for borrowed layers, restored post-prefill)
    const bool warm_inline =
        is_last_block && !layer_borrowed && o_.cache_warm && n_res_eff > 0 && n_res_eff < NE && l.host_blobs != nullptr;
    std::vector<int> promote_slot;
    std::vector<int> new_slot;
    std::vector<uint8_t> promoted_done;
    bool do_warm_commit = false;
    if (warm_inline) {
        uint32_t max_c = 0;
        for (uint32_t v : l.pf_counts) max_c = std::max(max_c, v);
        if (max_c > 0) {
            do_warm_commit = true;
            std::vector<int> order((size_t) NE);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
                if (l.pf_counts[(size_t) a] != l.pf_counts[(size_t) b])
                    return l.pf_counts[(size_t) a] > l.pf_counts[(size_t) b];
                return (l.h_slot[(size_t) a] >= 0) > (l.h_slot[(size_t) b] >= 0);
            });
            std::vector<uint8_t> want_e((size_t) NE, 0);
            for (int i = 0; i < n_res_eff; ++i) want_e[(size_t) order[(size_t) i]] = 1;

            new_slot = l.h_slot;
            promote_slot.assign((size_t) NE, -1);
            promoted_done.assign((size_t) NE, 0);
            std::vector<int> free_slots;
            l.h_owner.assign((size_t) n_res_eff, -1);
            for (int e = 0; e < NE; ++e) {
                const int sl = new_slot[(size_t) e];
                if (sl >= 0) {
                    if (want_e[(size_t) e]) {
                        l.h_owner[(size_t) sl] = e;
                    } else {
                        free_slots.push_back(sl);
                        new_slot[(size_t) e] = -1;
                    }
                }
            }
            size_t fi = 0;
            for (int i = 0; i < n_res_eff; ++i) {
                const int e = order[(size_t) i];
                if (new_slot[(size_t) e] >= 0) continue;
                const int sl = free_slots[fi++];
                new_slot[(size_t) e] = sl;
                l.h_owner[(size_t) sl] = e;
                promote_slot[(size_t) e] = sl;
            }
            l.h_score.resize((size_t) NE);
            const float ref_c = std::max(1.0f, (float) l.pf_counts[(size_t) order[(size_t) (n_res_eff - 1)]]);
            for (int e = 0; e < NE; ++e)
                l.h_score[(size_t) e] = std::min(8.0f, 1.5f * (float) l.pf_counts[(size_t) e] / ref_c);
        }
    }

    auto run_set_on = [&](Gpu& g, const MacroBufs& gb, int t0, int T_sub, const uint8_t* wbase, const int* slot_map,
                          int n_slots, int max_rows, bool accumulate) {
        if (max_rows <= 0 && n_res_eff < NE) return;
        uint16_t* h16 = nullptr;
        float* y = nullptr;
        scratch_ptrs(g, h16, y);
        const int* ids_p = gb.ids + (size_t) t0 * NU;
        const float* w_p = gb.w + (size_t) t0 * NU;
        const float* cur_p = gb.cur + (size_t) t0 * E;
        float* moe_p = gb.moe + (size_t) t0 * E;
        hy::moe_sort(ids_p, T_sub, NU, NE, g.pf_bounds, gb.tok, gb.row_of, g.s, slot_map, nullptr, n_slots);
        if (max_rows <= 0) {
            k::max_rows_from_bounds(g.pf_bounds, n_slots, g.d_max_rows, g.s);
            CK(cudaMemcpyAsync(g.h_max_rows, g.d_max_rows, sizeof(int), cudaMemcpyDeviceToHost, g.s));
            CK(cudaStreamSynchronize(g.s));
            max_rows = *g.h_max_rows;
            if (max_rows <= 0) return;
        }
        iq::SetArgs a;
        a.wbase = wbase;
        a.stride = l.blob;
        a.tr[0] = l.tr[0];
        a.tr[1] = l.tr[1];
        a.tr[2] = l.tr[2];
        a.bounds = g.pf_bounds;
        a.row0 = 0;
        a.max_rows = max_rows;
        a.n_exp = n_slots;
        a.n_embd = E;
        a.n_ff = FE;
        a.limit = c.clamp_exp[il] > 1e-6f ? -c.clamp_exp[il] : 0.0f;
        a.xf = cur_p;
        a.idx = gb.tok;
        a.ldx = E;
        a.h = h16;
        a.ldo = FE;
        a.tg = l.etg;
        a.td = l.etd;
        a.act = iq::kActSwiglu;
        if (!iq::set_gate_up(a, g.s)) throw std::runtime_error("moe prefill: set_gate_up failed");
        iq::SetArgs d = a;
        d.xf = nullptr;
        d.idx = nullptr;
        d.xh = h16;
        d.ldx = FE;
        d.h = nullptr;
        d.y = y;
        d.ldo = E;
        if (!iq::set_down(d, g.s)) throw std::runtime_error("moe prefill: set_down failed");
        hy::moe_combine(y, gb.row_of, w_p, T_sub, NU, E, moe_p, g.s, accumulate);
    };

    // Pack an expert list into row-bounded batches (each <= kPfBatch experts AND <= max_set_rows total rows)
    auto make_batches = [&](const std::vector<int>& elist) {
        std::vector<std::vector<int>> out;
        std::vector<int> cur;
        int cur_rows = 0;
        for (int e : elist) {
            const int rc = chunk_counts[(size_t) e];
            if (!cur.empty() && ((int) cur.size() >= kPfBatch || cur_rows + rc > max_set_rows)) {
                out.push_back(std::move(cur));
                cur.clear();
                cur_rows = 0;
            }
            cur.push_back(e);
            cur_rows += rc;
        }
        if (!cur.empty()) out.push_back(std::move(cur));
        return out;
    };

    auto stage_batch_on = [&](Gpu& g, const std::vector<std::vector<int>>& batches, size_t bi) {
        const int buf = (int) (bi & 1u);
        const uint64_t seq = g.pf_pmap_seq++;
        const int ring = (int) (seq % (uint64_t) kPfPmapRing);
        const auto& blist = batches[bi];
        const int Bn = (int) blist.size();
        set_dev(g.dev);
        // Unconditionally wait on ev_comp[buf] so layer il+1 never overwrites pf_stage[buf]/pf_pmap[buf]
        // while layer il is still executing run_set_on on g.s.
        CK(cudaStreamWaitEvent(g.s_copy, g.ev_comp[buf], 0));
        if (seq >= (uint64_t) kPfPmapRing) CK(cudaEventSynchronize(g.ev_pmap[ring]));
        int* pmap_h = g.h_pf_pmap[ring];
        std::fill(pmap_h, pmap_h + NE, -1);
        for (int b = 0; b < Bn; ++b) pmap_h[blist[(size_t) b]] = b;
        CK(cudaMemcpyAsync(g.pf_pmap[buf], pmap_h, (size_t) NE * sizeof(int), cudaMemcpyHostToDevice, g.s_copy));
        CK(cudaEventRecord(g.ev_pmap[ring], g.s_copy));
        if (l.host_pinned) {
            for (int b = 0; b < Bn; ++b) {
                const int e = blist[(size_t) b];
                CK(cudaMemcpyAsync(g.pf_stage[buf] + (size_t) b * l.blob, l.host_blobs + (size_t) e * l.blob, l.blob,
                                   cudaMemcpyHostToDevice, g.s_copy));
            }
        } else {
            CK(cudaEventSynchronize(g.ev_copy[buf]));
            auto cp = [&](int b) {
                const int e = blist[(size_t) b];
                std::memcpy(g.h_pf_stage[buf] + (size_t) b * l.blob, l.host_blobs + (size_t) e * l.blob, l.blob);
            };
            if (lane_ && Bn >= 4) lane_->parallel_for(Bn, cp);
            else for (int b = 0; b < Bn; ++b) cp(b);
            CK(cudaMemcpyAsync(g.pf_stage[buf], g.h_pf_stage[buf], (size_t) Bn * l.blob, cudaMemcpyHostToDevice,
                               g.s_copy));
        }
        CK(cudaEventRecord(g.ev_copy[buf], g.s_copy));
    };

    const bool cpu_miss = !miss_list.empty() && lane_ && B <= 48;
    const bool use_ep = o_.ep_prefill && !cpu_miss && ngpu >= 2 && B >= 256 && (int) miss_list.size() >= 32 &&
                        l.host_pinned;

    // 4. Partition misses across owner GPU + idle helper GPUs (EP Prefill Relay)
    std::vector<std::vector<int>> gpu_misses((size_t) ngpu);
    if (!cpu_miss && !miss_list.empty()) {
        if (!use_ep) {
            gpu_misses[(size_t) gi] = miss_list;
        } else {
            std::vector<int> free_misses;
            free_misses.reserve(miss_list.size());
            for (int e : miss_list) {
                if (do_warm_commit && promote_slot[(size_t) e] >= 0) gpu_misses[(size_t) gi].push_back(e);
                else free_misses.push_back(e);
            }
            const size_t total_m = miss_list.size();
            const size_t owner_target = (n_res_eff > NE / 2) ? (total_m * 42) / (size_t) (100 * (ngpu - 1) + 42)
                                                             : total_m / (size_t) ngpu;
            std::vector<int> helpers;
            for (int hj = 0; hj < ngpu; ++hj)
                if (hj != gi) helpers.push_back(hj);
            size_t h_rr = 0;
            for (int e : free_misses) {
                if (gpu_misses[(size_t) gi].size() < owner_target) {
                    gpu_misses[(size_t) gi].push_back(e);
                } else {
                    gpu_misses[(size_t) helpers[h_rr % helpers.size()]].push_back(e);
                    ++h_rr;
                }
            }
        }
    }

    std::vector<std::vector<std::vector<int>>> gpu_batches((size_t) ngpu);
    for (int gj = 0; gj < ngpu; ++gj) {
        if (!gpu_misses[(size_t) gj].empty()) gpu_batches[(size_t) gj] = make_batches(gpu_misses[(size_t) gj]);
    }

    // 5. Kick off DMA batch 0 on ALL active GPUs immediately + send cur/ids/w to helpers
    for (int gj = 0; gj < ngpu; ++gj) {
        if (!gpu_batches[(size_t) gj].empty()) {
            stage_batch_on(gpus_[(size_t) gj], gpu_batches[(size_t) gj], 0);
        }
    }
    for (int hj = 0; hj < ngpu; ++hj) {
        if (hj == gi || gpu_batches[(size_t) hj].empty()) continue;
        const MacroBufs& hb = mbufs[(size_t) hj];
        transfer_bytes(gi, hj, mb.cur, hb.cur, (size_t) B * E * sizeof(float));
        transfer_bytes(gi, hj, mb.ids, hb.ids, (size_t) B * NU * sizeof(int));
        transfer_bytes(gi, hj, mb.w, hb.w, (size_t) B * NU * sizeof(float));
    }

    // Queue all helper GPU batches asynchronously before owner runs so all GPUs and PCIe links overlap
    for (int hj = 0; hj < ngpu; ++hj) {
        if (hj == gi || gpu_batches[(size_t) hj].empty()) continue;
        Gpu& hg = gpus_[(size_t) hj];
        const MacroBufs& hb = mbufs[(size_t) hj];
        const auto& hbatches = gpu_batches[(size_t) hj];
        set_dev(hg.dev);
        for (size_t bi = 0; bi < hbatches.size(); ++bi) {
            const int buf = (int) (bi & 1u);
            const auto& blist = hbatches[bi];
            int bmax = 0;
            for (int e : blist) bmax = std::max(bmax, chunk_counts[(size_t) e]);
            CK(cudaStreamWaitEvent(hg.s, hg.ev_copy[buf], 0));
            run_set_on(hg, hb, 0, B, hg.pf_stage[buf], hg.pf_pmap[buf], (int) blist.size(), bmax, bi > 0);
            CK(cudaEventRecord(hg.ev_comp[buf], hg.s));
            if (bi + 1 < hbatches.size()) stage_batch_on(hg, hbatches, bi + 1);
        }
    }

    // 6. Owner GPU: Shared expert (tiled by P, accumulated directly into mb.x) overlapped with DMA batch 0
    set_dev(gp.dev);
    for (int t0 = 0; t0 < B; t0 += P) {
        const int T = std::min(P, B - t0);
        float* sh_g = gp.pf_tmp;
        float* sh_u = sh_g + (size_t) T * FS;
        hy::to_f16(mb.cur + (size_t) t0 * E, (size_t) E, T, E, gp.pf_x16, (size_t) E, s);
        gemm_mat(gp, l.fg, gp.pf_x16, T, sh_g);
        gemm_mat(gp, l.fu, gp.pf_x16, T, sh_u);
        k::swiglu_clamp_out(sh_g, sh_u, T, FS, c.clamp_shexp[il], nullptr, nullptr, nullptr, gp.pf_x16, s);
        gemm_mat(gp, l.fd, gp.pf_x16, T, gp.pf_out);
        hy::add(mb.x + (size_t) t0 * E, gp.pf_out, (size_t) T * E, s);
    }

    // 7. Owner GPU: VRAM-resident hits (greedily pack max tokens per pass up to max_set_rows to cut redundant dequant)
    bool accum = false;
    if (vram_hits > 0) {
        const int* vslot = (n_res_eff == NE) ? nullptr : l.fl.slot;
        if (vram_hits <= max_set_rows) {
            run_set_on(gp, mb, 0, B, l.fl.vram, vslot, n_res_eff, n_res_eff == NE ? 0 : vram_max_rows, false);
        } else if (n_res_eff == NE) {
            for (int t0 = 0; t0 < B; t0 += P) {
                const int T = std::min(P, B - t0);
                run_set_on(gp, mb, t0, T, l.fl.vram, nullptr, NE, 0, false);
            }
        } else {
            std::vector<int> scnt((size_t) NE, 0);
            int t0 = 0;
            while (t0 < B) {
                std::fill(scnt.begin(), scnt.end(), 0);
                int sub_rows = 0, sub_max = 0, t1 = t0;
                while (t1 < B) {
                    int tok_hits = 0;
                    for (int j = 0; j < NU; ++j)
                        if (l.h_slot[(size_t) gp.h_pf_ids[t1 * NU + j]] >= 0) ++tok_hits;
                    if (t1 > t0 && sub_rows + tok_hits > max_set_rows) break;
                    sub_rows += tok_hits;
                    for (int j = 0; j < NU; ++j) {
                        const int e = gp.h_pf_ids[t1 * NU + j];
                        if (l.h_slot[(size_t) e] >= 0) sub_max = std::max(sub_max, ++scnt[(size_t) e]);
                    }
                    ++t1;
                }
                const int T = t1 - t0;
                if (sub_max == 0) {
                    CK(cudaMemsetAsync(mb.moe + (size_t) t0 * E, 0, (size_t) T * E * sizeof(float), s));
                } else {
                    run_set_on(gp, mb, t0, T, l.fl.vram, vslot, n_res_eff, sub_max, false);
                }
                t0 = t1;
            }
        }
        accum = true;
    }

    // 8. Tiny-prompt CPU lane fast path
    if (cpu_miss) {
        std::vector<float> h_x((size_t) B * E);
        std::vector<float> h_o((size_t) B * E);
        CK(cudaMemcpyAsync(h_x.data(), mb.cur, (size_t) B * E * 4, cudaMemcpyDeviceToHost, sc));
        CK(cudaStreamSynchronize(sc));
        std::vector<hy4::CpuEntry> entries;
        entries.reserve((size_t) B * NU);
        for (int t = 0; t < B; ++t) {
            for (int j = 0; j < NU; ++j) {
                const int e = gp.h_pf_ids[t * NU + j];
                if (n_res_eff == 0 || l.h_slot[(size_t) e] < 0) entries.push_back({t, e, gp.h_pf_w[t * NU + j]});
            }
        }
        hy4::cpu_experts(*lane_, l.cex, entries, h_x.data(), B, h_o.data());
        CK(cudaMemcpyAsync(gp.pf_out, h_o.data(), (size_t) B * E * 4, cudaMemcpyHostToDevice, s));
        if (accum) hy::add(mb.moe, gp.pf_out, (size_t) B * E, s);
        else CK(cudaMemcpyAsync(mb.moe, gp.pf_out, (size_t) B * E * 4, cudaMemcpyDeviceToDevice, s));
        accum = true;
    }

    // 9. Owner GPU: cold staging batches + inline D2D cache promotion
    const auto& obatches = gpu_batches[(size_t) gi];
    if (!obatches.empty()) {
        for (size_t bi = 0; bi < obatches.size(); ++bi) {
            const int buf = (int) (bi & 1u);
            const auto& blist = obatches[bi];
            int bmax = 0;
            for (int e : blist) bmax = std::max(bmax, chunk_counts[(size_t) e]);
            CK(cudaStreamWaitEvent(s, gp.ev_copy[buf], 0));
            run_set_on(gp, mb, 0, B, gp.pf_stage[buf], gp.pf_pmap[buf], (int) blist.size(), bmax, accum);
            if (do_warm_commit) {
                for (size_t b = 0; b < blist.size(); ++b) {
                    const int e = blist[b];
                    const int sl = promote_slot[(size_t) e];
                    if (sl >= 0) {
                        CK(cudaMemcpyAsync(l.fl.vram + (size_t) sl * l.blob, gp.pf_stage[buf] + b * l.blob, l.blob,
                                           cudaMemcpyDeviceToDevice, s));
                        promoted_done[(size_t) e] = 1;
                    }
                }
            }
            CK(cudaEventRecord(gp.ev_comp[buf], s));
            accum = true;
            if (bi + 1 < obatches.size()) stage_batch_on(gp, obatches, bi + 1);
        }
        if (!l.host_pinned) CK(cudaEventSynchronize(gp.ev_comp[(obatches.size() - 1) & 1u]));
    }
    if (!accum) CK(cudaMemsetAsync(mb.moe, 0, (size_t) B * E * 4, s));

    // 10. Gather helper GPU partial MoE outputs back into owner mb.moe (tiled by P through gp.pf_out)
    for (int hj = 0; hj < ngpu; ++hj) {
        if (hj == gi || gpu_batches[(size_t) hj].empty()) continue;
        const MacroBufs& hb = mbufs[(size_t) hj];
        for (int t0 = 0; t0 < B; t0 += P) {
            const int T = std::min(P, B - t0);
            transfer_buf(hj, gi, hb.moe + (size_t) t0 * E, gp.pf_out, (size_t) T * E);
            set_dev(gp.dev);
            hy::add(mb.moe + (size_t) t0 * E, gp.pf_out, (size_t) T * E, s);
        }
        set_dev(gp.dev);
        CK(cudaEventRecord(gp.ev_xfer, s));
        set_dev(gpus_[(size_t) hj].dev);
        CK(cudaStreamWaitEvent(gpus_[(size_t) hj].s, gp.ev_xfer, 0));
    }

    set_dev(gp.dev);
    if (do_warm_commit) {
        for (int e = 0; e < NE; ++e) {
            const int sl = promote_slot[(size_t) e];
            if (sl >= 0 && !promoted_done[(size_t) e]) {
                CK(cudaMemcpyAsync(l.fl.vram + (size_t) sl * l.blob, l.host_blobs + (size_t) e * l.blob, l.blob,
                                   cudaMemcpyHostToDevice, s));
            }
        }
        l.h_slot = std::move(new_slot);
        CK(cudaMemcpyAsync(l.fl.slot, l.h_slot.data(), (size_t) NE * sizeof(int), cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(l.fl.owner, l.h_owner.data(), (size_t) n_res_eff * sizeof(int), cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(l.fl.score, l.h_score.data(), (size_t) NE * sizeof(float), cudaMemcpyHostToDevice, s));
        std::fill(l.pf_counts.begin(), l.pf_counts.end(), 0);
    }

    hy::add(mb.x, mb.moe, (size_t) B * E, s);
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
    for (auto& l : layers_) {
        if (l.promo_pending_e >= 0) {
            Gpu& gp = gpus_[(size_t) l.dev_idx];
            set_dev(gp.dev);
            CK(cudaEventSynchronize(l.ev_promo));
            const int pe = l.promo_pending_e, psl = l.promo_pending_sl;
            l.h_slot[(size_t) pe] = psl;
            l.h_owner[(size_t) psl] = pe;
            CK(cudaMemcpyAsync(l.fl.slot + pe, &l.h_slot[(size_t) pe], sizeof(int), cudaMemcpyHostToDevice, gp.s));
            l.promo_pending_e = -1;
            l.promo_pending_v = -1;
            l.promo_pending_sl = -1;
            ++cpu_admitted_;
        }
        l.cpu_cache_ready = false;
    }
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
    const int E = c.n_embd, NU = c.n_used, NE = c.n_expert, ngpu = (int) gpus_.size();
    const int P = gpus_[0].pf_cap;

    // Build 3D M-RoPE position table [3][n] if this prompt has multimodal images (body-diagonal symmetry).
    bool use_mrope = false;
    if (!images_.empty() || rope_delta_ != 0) {
        use_mrope = true;
        std::vector<int> h_mrope((size_t) 3 * (size_t) n, 0);
        int st_idx = pos0 + rope_delta_;
        const int merge_f = vision_ ? vision_->geo().factor() : 32;
        int i = 0;
        while (i < n) {
            const int64_t gpos = (int64_t) pos0 + i;
            const Image* hit = nullptr;
            for (const auto& im : images_) {
                if (im.start == gpos && !im.rows.empty()) {
                    hit = &im;
                    break;
                }
            }
            if (hit != nullptr) {
                const int gh = hit->height / merge_f;
                const int gw = hit->width / merge_f;
                const int ntok = gh * gw;
                const int t_idx = st_idx;
                for (int r = 0; r < gh; ++r) {
                    for (int col = 0; col < gw; ++col) {
                        const int k = r * gw + col;
                        if (i + k < n) {
                            h_mrope[(size_t) (i + k)] = t_idx;
                            h_mrope[(size_t) n + (size_t) (i + k)] = r - (gh - 1) / 2 + t_idx;
                            h_mrope[(size_t) 2 * (size_t) n + (size_t) (i + k)] = col - (gw - 1) / 2 + t_idx;
                        }
                    }
                }
                st_idx = t_idx + 1;
                i += std::max(1, ntok);
            } else {
                h_mrope[(size_t) i] = st_idx;
                h_mrope[(size_t) n + (size_t) i] = st_idx;
                h_mrope[(size_t) 2 * (size_t) n + (size_t) i] = st_idx;
                ++st_idx;
                ++i;
            }
        }
        rope_delta_ = st_idx - (pos0 + n);
        for (int gi = 0; gi < ngpu; ++gi) {
            Gpu& gp = gpus_[(size_t) gi];
            set_dev(gp.dev);
            if (gp.mrope_cap < n) {
                if (gp.d_mrope_pos != nullptr) CK(cudaFree(gp.d_mrope_pos));
                CK(cudaMalloc(&gp.d_mrope_pos, (size_t) 3 * (size_t) n * sizeof(int)));
                gp.mrope_cap = n;
            }
            CK(cudaMemcpyAsync(gp.d_mrope_pos, h_mrope.data(), (size_t) 3 * (size_t) n * sizeof(int),
                               cudaMemcpyHostToDevice, gp.s));
        }
    }

    bool any_offloaded = false;
    for (int il = c.n_dense_lead; il < c.n_layers; ++il) {
        if (layers_[(size_t) il].n_res < NE && layers_[(size_t) il].host_blobs != nullptr) {
            any_offloaded = true;
            break;
        }
    }

    // Use stage pipelining across GPUs when ngpu >= 2 and we have >= 2 blocks (no all_logits_host sync)
    // Choose optimal macro-block size:
    // - 1 GPU: maximize B up to 16384 to minimize repeated PCIe cold expert loads.
    // - 2+ GPUs: on n >= 8192, prefer B such that n / B >= 2*ngpu (e.g. B=8192 for 32K -> 4 pipelined stages = 80%
    //   parallel overlap; B=4096 for 8K -> 2 stages = 67% overlap; B=16384 for >=64K -> 90%+ overlap!).
    const bool want_pipe_buf = (ngpu >= 2 && all_logits_host == nullptr && !o_.ep_prefill);
    auto macro_bytes_for = [&](int B, bool with_pipe) -> size_t {
        const size_t x_cnt = with_pipe ? 4u : 3u;
        return align256((size_t) B * E * sizeof(float)) * x_cnt +
               align256((size_t) B * NU * sizeof(int)) * 3 +
               align256((size_t) B * NU * sizeof(float)) + 2048;
    };

    int B_max = P;
    size_t borrow_bytes = 0;
    bool has_pipe_x = false;
    std::vector<bool> layer_borrowed((size_t) c.n_layers, false);
    if ((any_offloaded || ngpu >= 2) && n > P && o_.prefill_block > P) {
        int want_cap = std::min(std::min(o_.prefill_block, gpus_[0].h_pf_cap), ((n + P - 1) / P) * P);
        if (ngpu >= 2 && !o_.ep_prefill && all_logits_host == nullptr) {
            // Keep at least 2*ngpu blocks when n >= 16K (B=8192 on 32K, B=16384 on 64K+), and 2 blocks on 8K (B=4096)
            if (n <= 8192) want_cap = std::min(want_cap, 4096);
            else if (n <= 49152) want_cap = std::min(want_cap, 8192);
        }
        for (int cand : {16384, 8192, 4096}) {
            if (cand > want_cap || cand <= P) continue;
            const bool need_pipe = want_pipe_buf && (n > cand);
            const size_t need = macro_bytes_for(cand, need_pipe);
            bool fits = true;
            for (int gi = 0; gi < ngpu; ++gi) {
                const Gpu& gp = gpus_[(size_t) gi];
                if (gp.vram_pool == nullptr || need * 4 > gp.vram_pool_bytes) {
                    fits = false;
                    break;
                }
                const size_t cut = gp.vram_pool_bytes - need;
                for (int il = std::max(gp.l0, c.n_dense_lead); il < gp.l1; ++il) {
                    const Layer& l = layers_[(size_t) il];
                    if (l.n_res > 0 && l.vram_pool_off + (size_t) l.n_res * l.blob > cut && l.host_blobs == nullptr) {
                        fits = false;
                        break;
                    }
                }
                if (!fits) break;
            }
            if (fits) {
                B_max = cand;
                borrow_bytes = need;
                has_pipe_x = need_pipe;
                break;
            }
        }
    }

    std::vector<MacroBufs> mbufs((size_t) ngpu);
    if (borrow_bytes > 0) {
        for (int gi = 0; gi < ngpu; ++gi) {
            Gpu& gp = gpus_[(size_t) gi];
            const size_t cut = gp.vram_pool_bytes - borrow_bytes;
            for (int il = std::max(gp.l0, c.n_dense_lead); il < gp.l1; ++il) {
                const Layer& l = layers_[(size_t) il];
                if (l.n_res > 0 && l.vram_pool_off + (size_t) l.n_res * l.blob > cut)
                    layer_borrowed[(size_t) il] = true;
            }
            uint8_t* p = gp.vram_pool + cut;
            auto take = [&](size_t bytes) -> uint8_t* {
                uint8_t* r = p;
                p += align256(bytes);
                return r;
            };
            mbufs[(size_t) gi].x = (float*) take((size_t) B_max * E * sizeof(float));
            mbufs[(size_t) gi].pipe_x = has_pipe_x ? (float*) take((size_t) B_max * E * sizeof(float)) : nullptr;
            mbufs[(size_t) gi].cur = (float*) take((size_t) B_max * E * sizeof(float));
            mbufs[(size_t) gi].moe = (float*) take((size_t) B_max * E * sizeof(float));
            mbufs[(size_t) gi].ids = (int*) take((size_t) B_max * NU * sizeof(int));
            mbufs[(size_t) gi].w = (float*) take((size_t) B_max * NU * sizeof(float));
            mbufs[(size_t) gi].tok = (int*) take((size_t) B_max * NU * sizeof(int));
            mbufs[(size_t) gi].row_of = (int*) take((size_t) B_max * NU * sizeof(int));
        }
    } else {
        for (int gi = 0; gi < ngpu; ++gi) {
            Gpu& gp = gpus_[(size_t) gi];
            mbufs[(size_t) gi] = {gp.pf_x, gp.pf_cur, gp.pf_moe, gp.pf_w, nullptr, gp.pf_ids, gp.pf_tok, gp.pf_row_of};
        }
    }

    const int n_blocks = (n + B_max - 1) / B_max;
    const bool use_stage_pipe = (ngpu >= 2 && n_blocks >= 2 && has_pipe_x && all_logits_host == nullptr && !o_.ep_prefill);

    auto run_gpu_stage = [&](int gi, int off, int B, bool is_last, float* x_buf) {
        Gpu& gp = gpus_[(size_t) gi];
        std::vector<MacroBufs> local_mb = mbufs;
        local_mb[(size_t) gi].x = x_buf;
        const int cp0 = pos0 + off;
        set_dev(gp.dev);
        if (gi == 0) {
            for (int t0 = 0; t0 < B; t0 += P) {
                const int T = std::min(P, B - t0);
                CK(cudaMemcpyAsync(gp.pf_tok_ids, tokens + off + t0, (size_t) T * sizeof(int),
                                   cudaMemcpyHostToDevice, gp.s));
                if (!k::emb_rows(emb_.p, emb_.type, emb_.k, gp.pf_tok_ids, T, x_buf + (size_t) t0 * E, gp.s))
                    throw std::runtime_error("prefill: emb_rows failed");
            }
            if (!images_.empty()) {
                for (const auto& im : images_) {
                    const int64_t ntok = (int64_t) (im.rows.size() / (size_t) E);
                    const int64_t a0 = std::max<int64_t>(im.start, (int64_t) cp0);
                    const int64_t a1 = std::min<int64_t>(im.start + ntok, (int64_t) (cp0 + B));
                    if (a1 > a0) {
                        const int64_t dst_off = a0 - (int64_t) cp0;
                        const int64_t src_off = a0 - im.start;
                        CK(cudaMemcpyAsync(x_buf + (size_t) dst_off * E, im.rows.data() + (size_t) src_off * E,
                                           (size_t) (a1 - a0) * (size_t) E * sizeof(float), cudaMemcpyHostToDevice,
                                           gp.s));
                    }
                }
            }
        }
        for (int il = gp.l0; il < gp.l1; ++il) {
            Layer& l = layers_[(size_t) il];
            set_dev(gp.dev);
            for (int t0 = 0; t0 < B; t0 += P) {
                const int T = std::min(P, B - t0);
                float* x_t = x_buf + (size_t) t0 * E;
                hy::rmsnorm(x_t, (size_t) E, T, E, l.attn_norm, c.eps, gp.pf_cur, (size_t) E, gp.s);
                if (l.kda) kda_prefill(gp, l, T, x_t);
                else mla_prefill(gp, l, T, cp0 + t0, x_t, use_mrope ? (gp.d_mrope_pos + (cp0 - pos0) + t0) : nullptr, use_mrope ? n : 0);
            }
            if (il < c.n_dense_lead) {
                for (int t0 = 0; t0 < B; t0 += P) {
                    const int T = std::min(P, B - t0);
                    float* x_t = x_buf + (size_t) t0 * E;
                    hy::rmsnorm(x_t, (size_t) E, T, E, l.ffn_norm, c.eps, gp.pf_cur, (size_t) E, gp.s);
                    ffn_dense_prefill(gp, l, il, T, x_t);
                }
            } else {
                ffn_moe_prefill_macro(gi, l, il, B, is_last, layer_borrowed[(size_t) il], local_mb);
            }
        }
        if (gi == ngpu - 1 && is_last && all_logits_host == nullptr) {
            set_dev(gp.dev);
            if (mtp_loaded_) {
                const int k_save = std::min(B, MtpWeights::kRing);
                const int s0 = B - k_save;
                hy::rmsnorm(x_buf + (size_t) s0 * E, (size_t) E, k_save, E, out_norm_, c.eps, local_mb[(size_t) gi].cur,
                            (size_t) E, gp.s);
                k::store_hring(local_mb[(size_t) gi].cur, k_save, E, mtp_.hring, MtpWeights::kRing, cp0 + s0, gp.s);
            }
            hy::rmsnorm(x_buf + (size_t) (B - 1) * E, (size_t) E, 1, E, out_norm_, c.eps, gp.hn, (size_t) E, gp.s,
                        {gp.xq, gp.xd});
            iq::MvJob j{out_w_.p, out_w_.ld, out_w_.type, E, c.vocab, 1, gp.xq, gp.xd, gp.logits, (size_t) c.vocab,
                        nullptr, 1.0f};
            if (!iq::mv(&j, 1, gp.s)) throw std::runtime_error("prefill: output failed");
        }
    };

    if (use_stage_pipe) {
        // Multi-GPU macro-block stage pipeline: 1 worker thread per GPU stage, ping-pong [0/1] x buffers.
        // Stage gi produces block m into slot (m & 1), transfers it into stage gi+1's slot (m & 1), and signals
        // stage gi+1 while immediately starting block m+1 into slot ((m+1) & 1)!
        struct Link {
            std::mutex mu;
            std::condition_variable cv;
            int ready[2] = {-1, -1};      // block index ready for consumer in slot s
            int consumed[2] = {-1, -1};   // last block index consumer finished reading from slot s
        };
        std::vector<Link> links((size_t) (ngpu - 1));
        std::exception_ptr err_ptr;
        std::mutex err_mu;

        auto stage_worker = [&](int gi) {
            try {
                Gpu& gp = gpus_[(size_t) gi];
                float* x_slots[2] = {mbufs[(size_t) gi].x, mbufs[(size_t) gi].pipe_x};
                for (int blk = 0; blk < n_blocks; ++blk) {
                    const int slot = blk & 1;
                    const int off = blk * B_max;
                    const int B = std::min(B_max, n - off);
                    const bool is_last = (blk + 1 == n_blocks);

                    if (gi > 0) {
                        Link& in_link = links[(size_t) (gi - 1)];
                        std::unique_lock<std::mutex> lk(in_link.mu);
                        in_link.cv.wait(lk, [&] { return in_link.ready[slot] == blk; });
                    }

                    run_gpu_stage(gi, off, B, is_last, x_slots[slot]);

                    if (gi + 1 < ngpu) {
                        // Wait until next stage finished using x_slots_next[slot] from blk - 2
                        Link& out_link = links[(size_t) gi];
                        if (blk >= 2) {
                            std::unique_lock<std::mutex> lk(out_link.mu);
                            out_link.cv.wait(lk, [&] { return out_link.consumed[slot] >= blk - 2; });
                        }
                        Gpu& gn = gpus_[(size_t) (gi + 1)];
                        float* dst_p = (slot == 0) ? mbufs[(size_t) (gi + 1)].x : mbufs[(size_t) (gi + 1)].pipe_x;
                        const size_t bytes = (size_t) B * E * sizeof(float);
                        set_dev(gp.dev);
                        CK(cudaStreamSynchronize(gp.s));
                        if (p2p_mat_[(size_t) gi][(size_t) (gi + 1)]) {
                            CK(cudaMemcpyPeerAsync(dst_p, gn.dev, x_slots[slot], gp.dev, bytes, gp.s));
                            CK(cudaStreamSynchronize(gp.s));
                        } else {
                            const size_t chunk = (size_t) std::max(1, gp.pf_cap) * (size_t) E * sizeof(float);
                            const uint8_t* sp = (const uint8_t*) x_slots[slot];
                            uint8_t* dp = (uint8_t*) dst_p;
                            for (size_t bo = 0; bo < bytes; bo += chunk) {
                                const size_t nb = std::min(chunk, bytes - bo);
                                set_dev(gp.dev);
                                CK(cudaMemcpyAsync(gp.h_xfer, sp + bo, nb, cudaMemcpyDeviceToHost, gp.s));
                                CK(cudaStreamSynchronize(gp.s));
                                set_dev(gn.dev);
                                CK(cudaMemcpyAsync(dp + bo, gp.h_xfer, nb, cudaMemcpyHostToDevice, gn.s_copy));
                                CK(cudaStreamSynchronize(gn.s_copy));
                            }
                        }
                        {
                            std::lock_guard<std::mutex> lk(out_link.mu);
                            out_link.ready[slot] = blk;
                        }
                        out_link.cv.notify_all();
                    }

                    if (gi > 0) {
                        set_dev(gp.dev);
                        CK(cudaStreamSynchronize(gp.s));
                        Link& in_link = links[(size_t) (gi - 1)];
                        {
                            std::lock_guard<std::mutex> lk(in_link.mu);
                            in_link.consumed[slot] = blk;
                        }
                        in_link.cv.notify_all();
                    }
                }
                set_dev(gp.dev);
                CK(cudaStreamSynchronize(gp.s));
            } catch (...) {
                std::lock_guard<std::mutex> lk(err_mu);
                if (!err_ptr) err_ptr = std::current_exception();
            }
        };

        std::vector<std::thread> threads;
        threads.reserve((size_t) ngpu);
        for (int gi = 0; gi < ngpu; ++gi) threads.emplace_back(stage_worker, gi);
        for (auto& th : threads) th.join();
        if (err_ptr) std::rethrow_exception(err_ptr);
    } else {
        std::vector<float> row_tmp;
        if (all_logits_host != nullptr) row_tmp.resize((size_t) c_.vocab);

        for (int off = 0; off < n; off += B_max) {
            const int B = std::min(B_max, n - off);
            const bool is_last = (off + B == n);
            for (int gi = 0; gi < ngpu; ++gi) {
                if (gi > 0) {
                    transfer_buf(gi - 1, gi, mbufs[(size_t) (gi - 1)].x, mbufs[(size_t) gi].x, (size_t) B * E);
                }
                run_gpu_stage(gi, off, B, is_last, mbufs[(size_t) gi].x);
                if (gi == ngpu - 1 && all_logits_host != nullptr) {
                    Gpu& gp = gpus_[(size_t) gi];
                    set_dev(gp.dev);
                    for (int t = 0; t < B; ++t) {
                        hy::rmsnorm(mbufs[(size_t) gi].x + (size_t) t * E, (size_t) E, 1, E, out_norm_, c.eps, gp.hn,
                                    (size_t) E, gp.s, {gp.xq, gp.xd});
                        iq::MvJob j{out_w_.p, out_w_.ld, out_w_.type, E, c.vocab, 1, gp.xq, gp.xd, gp.logits,
                                    (size_t) c.vocab, nullptr, 1.0f};
                        if (!iq::mv(&j, 1, gp.s)) throw std::runtime_error("prefill: output failed");
                        CK(cudaMemcpyAsync(row_tmp.data(), gp.logits, (size_t) c.vocab * 4, cudaMemcpyDeviceToHost,
                                           gp.s));
                        CK(cudaStreamSynchronize(gp.s));
                        all_logits_host->insert(all_logits_host->end(), row_tmp.begin(), row_tmp.end());
                    }
                }
            }
        }
        for (auto& gp : gpus_) {
            set_dev(gp.dev);
            CK(cudaStreamSynchronize(gp.s));
        }
    }

    // Restore the 1-2 borrowed tail layers (warming them directly from pf_counts if cache_warm is enabled)
    if (borrow_bytes > 0) {
        const auto t_r0 = std::chrono::steady_clock::now();
        for (int il = c.n_dense_lead; il < c.n_layers; ++il) {
            if (!layer_borrowed[(size_t) il]) continue;
            Layer& l = layers_[(size_t) il];
            if (l.n_res <= 0 || l.host_blobs == nullptr) continue;
            Gpu& gp = gpus_[(size_t) l.dev_idx];
            set_dev(gp.dev);

            uint32_t max_c = 0;
            for (uint32_t v : l.pf_counts) max_c = std::max(max_c, v);
            std::vector<int> pick;
            pick.reserve((size_t) l.n_res);
            if (o_.cache_warm && max_c > 0) {
                std::vector<int> order((size_t) NE);
                std::iota(order.begin(), order.end(), 0);
                std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
                    if (l.pf_counts[(size_t) a] != l.pf_counts[(size_t) b])
                        return l.pf_counts[(size_t) a] > l.pf_counts[(size_t) b];
                    return (l.h_slot[(size_t) a] >= 0) > (l.h_slot[(size_t) b] >= 0);
                });
                for (int i = 0; i < l.n_res; ++i) pick.push_back(order[(size_t) i]);
            } else {
                for (int e = 0; e < NE && (int) pick.size() < l.n_res; ++e)
                    if (l.h_slot[(size_t) e] >= 0) pick.push_back(e);
                for (int e = 0; e < NE && (int) pick.size() < l.n_res; ++e)
                    if (l.h_slot[(size_t) e] < 0) pick.push_back(e);
            }
            l.h_slot.assign((size_t) NE, -1);
            l.h_owner.assign((size_t) l.n_res, -1);
            for (int sl = 0; sl < l.n_res; ++sl) {
                const int e = pick[(size_t) sl];
                l.h_slot[(size_t) e] = sl;
                l.h_owner[(size_t) sl] = e;
                CK(cudaMemcpyAsync(l.fl.vram + (size_t) sl * l.blob, l.host_blobs + (size_t) e * l.blob, l.blob,
                                   cudaMemcpyHostToDevice, gp.s_copy));
            }
            l.h_score.assign((size_t) NE, 0.0f);
            if (max_c > 0) {
                const float norm = 4.0f / (float) max_c;
                for (int e = 0; e < NE; ++e) l.h_score[(size_t) e] = (float) l.pf_counts[(size_t) e] * norm;
            }
            CK(cudaMemcpyAsync(l.fl.slot, l.h_slot.data(), (size_t) NE * sizeof(int), cudaMemcpyHostToDevice,
                               gp.s_copy));
            CK(cudaMemcpyAsync(l.fl.owner, l.h_owner.data(), (size_t) l.n_res * sizeof(int), cudaMemcpyHostToDevice,
                               gp.s_copy));
            CK(cudaMemcpyAsync(l.fl.score, l.h_score.data(), (size_t) NE * sizeof(float), cudaMemcpyHostToDevice,
                               gp.s_copy));
            std::fill(l.pf_counts.begin(), l.pf_counts.end(), 0);
        }
        for (auto& gp : gpus_) {
            set_dev(gp.dev);
            CK(cudaStreamSynchronize(gp.s_copy));
        }
        last_warm_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_r0).count();
    }
    if (mtp_loaded_) {
        if ((int) mtp_tokens_.size() < pos0 + n) mtp_tokens_.resize((size_t) (pos0 + n), 0);
        for (int i = 0; i < n; ++i) mtp_tokens_[(size_t) (pos0 + i)] = tokens[i];
        // Start MTP KV catch-up from at most the last 32 tokens of the prompt (tiny ~15 ms one-time warm)
        mtp_done_ = std::max(mtp_done_, std::max(0, pos0 + n - 32));
    }
    return gpus_.back().logits;
}

// ------------------------------ windowed decode & MTP speculative loop ------------------------------

void Model::kda_decode_win(Gpu& gp, Layer& l, int il, int W, bool add_to_x) {
    (void) il;
    const Config& c = c_;
    const int DI = c.d_inner(), H = c.n_head;
    cudaStream_t s = gp.s;
    float* q_buf = l.w_kda_buf;
    float* k_buf = q_buf + (size_t) kMaxWin * DI;
    float* v_buf = k_buf + (size_t) kMaxWin * DI;
    float* kc_buf = v_buf + (size_t) kMaxWin * DI;
    float* vc_buf = kc_buf + (size_t) kMaxWin * DI;
    float* g1_buf = vc_buf + (size_t) kMaxWin * DI;
    float* beta_buf = g1_buf + (size_t) kMaxWin * DI;

    iq::MvJob j[6];
    auto mk = [&](const Mat& m, float* y, const float* bias = nullptr) {
        return iq::MvJob{m.p, m.ld, m.type, m.k, m.n, W, gp.xq, gp.xd, y, (size_t) m.n, bias, 1.0f};
    };
    j[0] = mk(l.wq, q_buf);
    j[1] = mk(l.wk, k_buf);
    j[2] = mk(l.wv, v_buf);
    j[3] = mk(l.wf, gp.f, l.dt_bias);
    j[4] = mk(l.wbeta, beta_buf);
    j[5] = mk(l.wg, gp.g2);
    if (!iq::mv(j, 6, s)) throw std::runtime_error("kda_win: type not supported");

    k::kda_prep_window(q_buf, k_buf, v_buf, gp.f, l.conv_q, l.conv_k, l.conv_v, l.conv_state, l.conv_bak, l.ssm_a,
                       c.kda_lb, c.eps, W, H, c.kda_head, c.d_conv, gp.qc, kc_buf, vc_buf, g1_buf, s);
    k::kda_rec_window_q8(gp.qc, kc_buf, vc_buf, g1_buf, beta_buf, l.state, l.state_tmp, gp.g2, l.ssm_norm, c.eps, W, H,
                         c.kda_head, gp.xq2, gp.xd2, s);

    iq::MvJob o{l.wo.p, l.wo.ld, l.wo.type, l.wo.k, l.wo.n, W, gp.xq2, gp.xd2, gp.out, (size_t) l.wo.n, nullptr, 1.0f};
    if (!iq::mv(&o, 1, s)) throw std::runtime_error("kda_win: output type not supported");
    if (add_to_x) hy::add(gp.x, gp.out, (size_t) W * c.n_embd, s);
}

void Model::mla_decode_win(Gpu& gp, Layer& l, int il, int W, bool add_to_x) {
    (void) il;
    const Config& c = c_;
    const int H = c.n_head, KL = c.kv_lora, R = c.n_rot;
    cudaStream_t s = gp.s;

    iq::MvJob j[3] = {
        {l.wqm.p, l.wqm.ld, l.wqm.type, l.wqm.k, l.wqm.n, W, gp.xq, gp.xd, gp.qm, (size_t) l.wqm.n, nullptr, 1.0f},
        {l.wkva.p, l.wkva.ld, l.wkva.type, l.wkva.k, l.wkva.n, W, gp.xq, gp.xd, gp.kvall, (size_t) l.wkva.n, nullptr, 1.0f},
        {l.wgate.p, l.wgate.ld, l.wgate.type, l.wgate.k, l.wgate.n, W, gp.xq, gp.xd, gp.gate, (size_t) l.wgate.n, nullptr, 1.0f},
    };
    if (!iq::mv(j, 3, s)) throw std::runtime_error("mla_win: type not supported");

    k::mla_q_prep(gp.qm, W, H, c.qk_head, c.qk_nope, R, gp.d_pos_win, 0, c.rope_base, gp.q8n, gp.d8n, nullptr, s, gp.d_step + 3);
    iq::MvJob kb{l.wkb.p, l.wkb.ld, l.wkb.type, c.qk_nope, KL, W, gp.q8n, gp.d8n, gp.q_abs, (size_t) H * KL, nullptr, 1.0f};
    if (!iq::mv_heads(kb, H, (size_t) KL * l.wkb.ld, (size_t) W * c.qk_nope, (size_t) W * (c.qk_nope / 32), (size_t) KL, s))
        throw std::runtime_error("mla_win: k_b type not supported");

    k::kv_store_split(gp.kvall, W, KL, R, l.kv_a_norm, c.eps, gp.d_pos_win, 0, c.rope_base, l.lat, o_.kv_q8, l.k_pe, s, gp.d_step + 3);

    if (!k::mla_dec_wmma(gp.q_abs, gp.qm, c.qk_head, c.qk_nope, l.lat, o_.kv_q8 ? 1 : 0, l.k_pe, o_.ctx,
                         gp.d_step + 2, W, H, KL, R, 1.0f / sqrtf((float) c.qk_head), gp.mla_part, gp.ctxo, s))
        throw std::runtime_error("mla_win: mla_dec_wmma failed");

    hy::q8_heads(gp.ctxo, (size_t) H * KL, (size_t) KL, W, H, KL, gp.q8c, gp.d8c, s);
    iq::MvJob vb{l.wvb.p, l.wvb.ld, l.wvb.type, KL, c.v_head, W, gp.q8c, gp.d8c, gp.o128, (size_t) H * c.v_head, nullptr, 1.0f};
    if (!iq::mv_heads(vb, H, (size_t) c.v_head * l.wvb.ld, (size_t) W * KL, (size_t) W * (KL / 32), (size_t) c.v_head, s))
        throw std::runtime_error("mla_win: v_b type not supported");

    k::head_gate_out(gp.o128, gp.gate, W, H, c.v_head, gp.xq2, gp.xd2, nullptr, s);
    iq::MvJob o{l.wom.p, l.wom.ld, l.wom.type, l.wom.k, l.wom.n, W, gp.xq2, gp.xd2, gp.out, (size_t) l.wom.n, nullptr, 1.0f};
    if (!iq::mv(&o, 1, s)) throw std::runtime_error("mla_win: output type not supported");
    if (add_to_x) hy::add(gp.x, gp.out, (size_t) W * c.n_embd, s);
}

void Model::ffn_dense_decode_win(Gpu& gp, Layer& l, int il, int W, const float* next_norm) {
    const Config& c = c_;
    cudaStream_t s = gp.s;
    iq::MvJob j[2] = {
        {l.fg.p, l.fg.ld, l.fg.type, l.fg.k, l.fg.n, W, gp.xq, gp.xd, gp.gu, (size_t) l.fg.n, nullptr, 1.0f},
        {l.fu.p, l.fu.ld, l.fu.type, l.fu.k, l.fu.n, W, gp.xq, gp.xd, gp.gu + (size_t) W * c.n_ff, (size_t) l.fu.n, nullptr, 1.0f},
    };
    if (!iq::mv(j, 2, s)) throw std::runtime_error("ffn_win: type not supported");
    k::swiglu_clamp_out(gp.gu, gp.gu + (size_t) W * c.n_ff, W, c.n_ff, c.clamp_shexp[il], nullptr, gp.ehq, gp.ehd, nullptr, s);
    iq::MvJob d{l.fd.p, l.fd.ld, l.fd.type, l.fd.k, l.fd.n, W, gp.ehq, gp.ehd, gp.out, (size_t) l.fd.n, nullptr, 1.0f};
    if (!iq::mv(&d, 1, s)) throw std::runtime_error("ffn_win: down type not supported");
    if (next_norm != nullptr) k::add_rmsnorm_q8(gp.x, gp.out, next_norm, W, c.n_embd, c.eps, gp.cur, gp.xq, gp.xd, s);
    else hy::add(gp.x, gp.out, (size_t) W * c.n_embd, s);
}

void Model::ffn_moe_decode_win(Gpu& gp, Layer& l, int il, int W, const float* next_norm) {
    const Config& c = c_;
    const int E = c.n_embd, FE = c.n_ff_exp, FS = c.n_ff_shexp, NU = c.n_used;
    cudaStream_t s = gp.s;
    const bool use_lane = (l.n_res < c.n_expert) &&
                          (active_offload_ == OffloadMode::kCpu || active_offload_ == OffloadMode::kMix || !l.host_pinned);

    auto run_shared = [&](cudaStream_t st) {
        iq::MvJob sj[2] = {
            {l.fg.p, l.fg.ld, l.fg.type, E, FS, W, gp.xq, gp.xd, gp.gu, (size_t) FS, nullptr, 1.0f},
            {l.fu.p, l.fu.ld, l.fu.type, E, FS, W, gp.xq, gp.xd, gp.gu + (size_t) W * FS, (size_t) FS, nullptr, 1.0f},
        };
        if (!iq::mv(sj, 2, st)) throw std::runtime_error("moe_win: shared gate/up failed");
        k::swiglu_clamp_out(gp.gu, gp.gu + (size_t) W * FS, W, FS, c.clamp_shexp[il], nullptr, gp.shq, gp.shd, nullptr, st);
        iq::MvJob sd{l.fd.p, l.fd.ld, l.fd.type, FS, E, W, gp.shq, gp.shd, gp.shout, (size_t) E, nullptr, 1.0f};
        if (!iq::mv(&sd, 1, st)) throw std::runtime_error("moe_win: shared down failed");
    };

    auto finish_add = [&] {
        if (next_norm != nullptr) k::add2_rmsnorm_q8(gp.x, gp.moe, gp.shout, next_norm, W, E, c.eps, gp.cur, gp.xq, gp.xd, s);
        else k::add2(gp.x, gp.moe, gp.shout, (int64_t) W * E, s);
    };

    if (use_lane) {
        CK(cudaEventRecord(gp.ev_norm, s));
        CK(cudaStreamWaitEvent(gp.s_d2h, gp.ev_norm, 0));
        CK(cudaMemcpyAsync(gp.h_cur, gp.cur, (size_t) W * E * sizeof(float), cudaMemcpyDeviceToHost, gp.s_d2h));
    } else {
        CK(cudaEventRecord(gp.ev_norm, s));
        CK(cudaStreamWaitEvent(gp.s_copy, gp.ev_norm, 0));
        run_shared(gp.s_copy);
        CK(cudaEventRecord(gp.ev_sh, gp.s_copy));
    }

    if (l.router_type == 0)
        hy::gemv_f32w((const float*) l.router_w, (size_t) E, E, c.n_expert, gp.cur, (size_t) E, W, gp.rlog, (size_t) c.n_expert, 1.0f, s);
    else
        hy::gemv_bf16w((const uint16_t*) l.router_w, (size_t) E, E, c.n_expert, gp.cur, (size_t) E, W, gp.rlog, (size_t) c.n_expert, 1.0f, s);
    k::router_grouped(gp.rlog, l.rb_dev, W, c.n_expert, NU, c.n_group, c.n_group_used, c.w_norm, c.w_scale, gp.d_ids, gp.d_w, s);

    hy::dup_rows_q8(gp.xq, gp.xd, W, E, gp.x2q, gp.x2d, s);
    iq::Args a;
    a.grp_ptr = gp.grp_ptr;
    a.grp_start = gp.grp_start;
    a.grp_ent = gp.grp_ent;
    a.n_groups = gp.n_groups;
    a.tr[0] = l.tr[0];
    a.tr[1] = l.tr[1];
    a.tr[2] = l.tr[2];
    a.n_embd = E;
    a.n_ff = FE;
    a.limit = c.clamp_exp[il] > 1e-6f ? -c.clamp_exp[il] : 0.0f;
    a.xq = gp.x2q;
    a.xd = gp.x2d;
    a.h = gp.eh;
    a.hq = gp.ehq;
    a.hd = gp.ehd;
    a.part = gp.part;
    a.tg = l.etg;
    a.td = l.etd;
    a.act = iq::kActSwiglu;
    const int groups = std::min(W * NU, c.n_expert);

    if (l.n_res == c.n_expert) {
        hy::group_experts(gp.d_ids, W, NU, c.n_expert, l.fl.vram, l.blob, gp.grp_ptr, gp.grp_start, gp.grp_ent, gp.n_groups, s);
        if (!iq::gate_up(a, groups, W, s)) throw std::runtime_error("moe_win: gate_up failed");
        hy::q8_rows(gp.eh, (size_t) FE, W * 16, FE, gp.ehq, gp.ehd, s);
        if (!iq::down(a, groups, W, s)) throw std::runtime_error("moe_win: down failed");
        hy::moe_sum(gp.part, W, NU, E, gp.d_w, gp.moe, s);
        CK(cudaStreamWaitEvent(s, gp.ev_sh, 0));
        finish_add();
        return;
    }

    if (!use_lane) {
        k::fetch_resolve_fast(gp.d_ids, W, NU, l.fl, gp.fxs, -1, l.n_res > 0, 0.98f, s);
        hy::fetch_copy(gp.fxs, l.blob, 32, s);
        hy::group_experts(gp.d_ids, W, NU, c.n_expert, nullptr, 0, gp.grp_ptr, gp.grp_start, gp.grp_ent, gp.n_groups,
                          s, nullptr, gp.d_w, gp.fxs.eptr, gp.fxs.who);
        if (!iq::gate_up(a, groups, W, s)) throw std::runtime_error("moe_win: gate_up failed");
        hy::q8_rows(gp.eh, (size_t) FE, W * 16, FE, gp.ehq, gp.ehd, s);
        if (!iq::down(a, groups, W, s)) throw std::runtime_error("moe_win: down failed");
        hy::moe_sum(gp.part, W, NU, E, gp.d_w, gp.moe, s);
        CK(cudaStreamWaitEvent(s, gp.ev_sh, 0));
        finish_add();
        return;
    }

    // Complete any finished background promotion from an earlier step
    if (l.promo_pending_e >= 0 && cudaEventQuery(l.ev_promo) == cudaSuccess) {
        const int pe = l.promo_pending_e, psl = l.promo_pending_sl;
        l.h_slot[(size_t) pe] = psl;
        l.h_owner[(size_t) psl] = pe;
        CK(cudaMemcpyAsync(l.fl.slot + pe, &l.h_slot[(size_t) pe], sizeof(int), cudaMemcpyHostToDevice, s));
        l.promo_pending_e = -1;
        l.promo_pending_v = -1;
        l.promo_pending_sl = -1;
        ++cpu_admitted_;
    }

    const int k_gpu = (active_offload_ == OffloadMode::kMix && l.host_pinned) ? o_.mix_k_gpu : 0;
    const bool adapt = (k_gpu > 0 && l.n_res > 0);
    k::fetch_resolve_fast(gp.d_ids, W, NU, l.fl, gp.fxs, k_gpu, adapt, 0.98f, s);
    k::pack_route_win_cpu(gp.d_ids, gp.d_w, gp.fxs.who, W * NU, gp.d_route_win, s);
    CK(cudaEventRecord(gp.ev_route, s));

    CK(cudaStreamWaitEvent(gp.s_d2h, gp.ev_route, 0));
    CK(cudaMemcpyAsync(gp.h_route_win, gp.d_route_win, sizeof(k::RoutePackWin), cudaMemcpyDeviceToHost, gp.s_d2h));
    CK(cudaEventRecord(gp.ev_d2h, gp.s_d2h));

    if (k_gpu > 0) hy::fetch_copy(gp.fxs, l.blob, 32, s);
    hy::group_experts(gp.d_ids, W, NU, c.n_expert, nullptr, 0, gp.grp_ptr, gp.grp_start, gp.grp_ent, gp.n_groups,
                      s, nullptr, gp.d_w, gp.fxs.eptr, gp.fxs.who);
    if (!iq::gate_up(a, groups, W, s)) throw std::runtime_error("moe_win: gate_up failed");
    hy::q8_rows(gp.eh, (size_t) FE, W * 16, FE, gp.ehq, gp.ehd, s);
    if (!iq::down(a, groups, W, s)) throw std::runtime_error("moe_win: down failed");
    hy::moe_sum(gp.part, W, NU, E, gp.d_w, gp.moe, s);
    run_shared(s);

    CK(cudaEventSynchronize(gp.ev_d2h));
    const k::RoutePackWin& rp = *gp.h_route_win;
    std::vector<hy4::CpuEntry> entries;
    entries.reserve((size_t) (W * NU));
    for (int t = 0; t < W; ++t) {
        for (int j2 = 0; j2 < NU; ++j2) {
            const int idx = t * NU + j2;
            if (rp.who[idx] == 1) entries.push_back({t, rp.ids[idx], rp.w[idx]});
        }
    }
    if (!entries.empty()) {
        cpu_lane_picks_ += (unsigned long long) entries.size();
        hy4::cpu_experts(*lane_, l.cex, entries, gp.h_cur, W, gp.h_cpu_out);
        CK(cudaMemcpyAsync(gp.d_cpu_out, gp.h_cpu_out, (size_t) W * E * sizeof(float), cudaMemcpyHostToDevice, s));
        hy::add(gp.moe, gp.d_cpu_out, (size_t) W * E, s);
    }

    // Async background promotion (same race-free policy as T=1)
    if (k_gpu == 0 && l.n_res > 0 && l.n_res < c.n_expert && l.host_blobs != nullptr && l.host_pinned) {
        if (!l.cpu_cache_ready) {
            if ((int) l.h_owner.size() != l.n_res) l.h_owner.assign((size_t) l.n_res, -1);
            if ((int) l.h_score.size() != c.n_expert) l.h_score.assign((size_t) c.n_expert, 0.0f);
            for (int e = 0; e < c.n_expert; ++e) {
                const int sl = l.h_slot[(size_t) e];
                if (sl >= 0 && sl < l.n_res) l.h_owner[(size_t) sl] = e;
            }
            l.cpu_cache_ready = true;
        }
        const float dec_w = powf(0.98f, (float) W);
        for (int e = 0; e < c.n_expert; ++e) l.h_score[(size_t) e] *= dec_w;
        uint8_t active_mask[512] = {};
        for (int i = 0; i < W * NU; ++i) {
            const int e = rp.ids[i];
            l.h_score[(size_t) e] += 1.0f;
            if (e >= 0 && e < 512) active_mask[e] = 1;
        }
        if (!entries.empty() && l.promo_pending_e < 0 && tok_promo_budget_ > 0) {
            int best_m = -1;
            float best_ms = -1.0f;
            for (const auto& ce : entries) {
                if (l.h_score[(size_t) ce.e] > best_ms) {
                    best_ms = l.h_score[(size_t) ce.e];
                    best_m = ce.e;
                }
            }
            int worst_v = -1, worst_sl = -1;
            float worst_vs = 1e30f;
            for (int sl = 0; sl < l.n_res; ++sl) {
                const int ve = l.h_owner[(size_t) sl];
                if (ve < 0 || (ve < 512 && active_mask[ve])) continue;
                if (l.h_score[(size_t) ve] < worst_vs) {
                    worst_vs = l.h_score[(size_t) ve];
                    worst_v = ve;
                    worst_sl = sl;
                }
            }
            if (best_m >= 0 && worst_v >= 0 && best_ms > worst_vs) {
                l.h_slot[(size_t) worst_v] = -1;
                l.h_owner[(size_t) worst_sl] = -1;
                CK(cudaMemcpyAsync(l.fl.slot + worst_v, gp.h_neg1, sizeof(int), cudaMemcpyHostToDevice, s));
                CK(cudaEventRecord(gp.ev_comp[0], s));
                CK(cudaStreamWaitEvent(gp.s_copy, gp.ev_comp[0], 0));
                CK(cudaMemcpyAsync(l.fl.vram + (size_t) worst_sl * l.blob, l.host_blobs + (size_t) best_m * l.blob,
                                   l.blob, cudaMemcpyHostToDevice, gp.s_copy));
                CK(cudaEventRecord(l.ev_promo, gp.s_copy));
                l.promo_pending_e = best_m;
                l.promo_pending_v = worst_v;
                l.promo_pending_sl = worst_sl;
                --tok_promo_budget_;
            }
        }
    }

    finish_add();
}

void Model::decode_segment_win(int gi, int W, bool capture_safe) {
    (void) capture_safe;
    Gpu& gp = gpus_[(size_t) gi];
    const Config& c = c_;
    const int E = c.n_embd;
    cudaStream_t s = gp.s;

    if (gi == 0) {
        if (!k::emb_rows(emb_.p, emb_.type, emb_.k, gp.d_tok_win, W, gp.x, s))
            throw std::runtime_error("emb_rows_win: type not supported");
    }

    bool normed = false;
    for (int il = gp.l0; il < gp.l1; ++il) {
        Layer& l = layers_[(size_t) il];
        if (!normed) hy::rmsnorm(gp.x, (size_t) E, W, E, l.attn_norm, c.eps, gp.cur, (size_t) E, s, {gp.xq, gp.xd});
        if (l.kda) kda_decode_win(gp, l, il, W, false);
        else mla_decode_win(gp, l, il, W, false);
        k::add_rmsnorm_q8(gp.x, gp.out, l.ffn_norm, W, E, c.eps, gp.cur, gp.xq, gp.xd, s);
        const float* next_norm = nullptr;
        if (il + 1 < gp.l1) next_norm = layers_[(size_t) (il + 1)].attn_norm;
        else if (gi == (int) gpus_.size() - 1) next_norm = out_norm_;
        if (il < c.n_dense_lead) ffn_dense_decode_win(gp, l, il, W, next_norm);
        else ffn_moe_decode_win(gp, l, il, W, next_norm);
        normed = (next_norm != nullptr);
    }

    if (gi == (int) gpus_.size() - 1) {
        if (!normed) hy::rmsnorm(gp.x, (size_t) E, W, E, out_norm_, c.eps, gp.cur, (size_t) E, s, {gp.xq, gp.xd});
        iq::MvJob j{out_w_.p, out_w_.ld, out_w_.type, out_w_.k, out_w_.n, W, gp.xq, gp.xd, gp.logits, (size_t) out_w_.n,
                    nullptr, 1.0f};
        if (!iq::mv(&j, 1, s)) throw std::runtime_error("output_win: type not supported");
    }
}

const float* Model::step_window(const int* tokens, int W, int pos0) {
    if (W <= 1) return step(tokens[0], pos0);
    if (!mtp_loaded_ || W > kMaxWin) throw std::runtime_error("step_window requires --mtp and W <= 4");
    tok_promo_budget_ = 32;
    promo_rr_ = c_.n_dense_lead + ((pos0 * 7) % std::max(1, c_.n_layers - c_.n_dense_lead));
    const int ngpu = (int) gpus_.size();
    const bool graph_ok = o_.use_graph && active_offload_ == OffloadMode::kGpu && !any_host_unpinned_ && o_.seam_dir.empty();
    const int wi = W - 1;

    for (int gi = 0; gi < ngpu; ++gi) {
        Gpu& gp = gpus_[(size_t) gi];
        set_dev(gp.dev);
        k::set_step_win(gp.d_step, gp.d_tok_win, gp.d_pos_win, tokens[0], W > 1 ? tokens[1] : 0,
                        W > 2 ? tokens[2] : 0, W > 3 ? tokens[3] : 0, W, pos0, rope_delta_, gp.s);
        if (gi > 0) transfer_x(gi - 1, gi, W, false);

        if (graph_ok) {
            if (!gp.wgraph_ready[wi]) {
                CK(cudaStreamBeginCapture(gp.s, cudaStreamCaptureModeRelaxed));
                decode_segment_win(gi, W, true);
                CK(cudaStreamEndCapture(gp.s, &gp.wgraph[wi]));
                CK(cudaGraphInstantiate(&gp.wgraph_exec[wi], gp.wgraph[wi], nullptr, nullptr, 0));
                gp.wgraph_ready[wi] = true;
                CK(cudaGraphLaunch(gp.wgraph_exec[wi], gp.s));
            } else {
                CK(cudaGraphLaunch(gp.wgraph_exec[wi], gp.s));
            }
        } else {
            decode_segment_win(gi, W, false);
        }
    }
    Gpu& gl = gpus_.back();
    set_dev(gl.dev);
    k::store_hring(gl.cur, W, c_.n_embd, mtp_.hring, MtpWeights::kRing, pos0, gl.s);
    return gl.logits;
}

void Model::commit_window(int n_accept, int W, int pos0, const int* tokens) {
    if (W <= 1) return;
    const Config& c = c_;
    const int DI = c.d_inner(), H = c.n_head;
    for (int il = 0; il < c.n_layers; ++il) {
        Layer& l = layers_[(size_t) il];
        if (!l.kda) continue;
        Gpu& gp = gpus_[(size_t) l.dev_idx];
        set_dev(gp.dev);
        float* q_buf = l.w_kda_buf;
        float* k_buf = q_buf + (size_t) kMaxWin * DI;
        float* v_buf = k_buf + (size_t) kMaxWin * DI;
        float* kc_buf = v_buf + (size_t) kMaxWin * DI;
        float* vc_buf = kc_buf + (size_t) kMaxWin * DI;
        float* g1_buf = vc_buf + (size_t) kMaxWin * DI;
        float* beta_buf = g1_buf + (size_t) kMaxWin * DI;
        if (n_accept < W) {
            k::kda_conv_rollback(l.conv_bak, q_buf, k_buf, v_buf, n_accept, H, c.kda_head, c.d_conv, l.conv_state, gp.s);
        }
        k::kda_commit_window(kc_buf, vc_buf, g1_buf, beta_buf, l.state, l.state_tmp, n_accept, W, H, c.kda_head, gp.s);
    }
    if (mtp_loaded_) {
        if ((int) mtp_tokens_.size() < pos0 + n_accept) mtp_tokens_.resize((size_t) (pos0 + n_accept), 0);
        for (int t = 0; t < n_accept; ++t) mtp_tokens_[(size_t) (pos0 + t)] = tokens[t];
    }
}

void Model::argmax_rows(const float* logits_dev, int W, int* out_ids) {
    Gpu& gl = gpus_.back();
    set_dev(gl.dev);
    k::argmax_rows_f32(logits_dev, W, c_.vocab, gl.d_arg_val, gl.d_arg_idx, gl.d_arg_rows, gl.s);
    CK(cudaMemcpyAsync(gl.h_arg_rows, gl.d_arg_rows, (size_t) W * sizeof(int), cudaMemcpyDeviceToHost, gl.s));
    CK(cudaStreamSynchronize(gl.s));
    for (int t = 0; t < W; ++t) out_ids[t] = gl.h_arg_rows[t];
}

void Model::mtp_draft(int base_tok, int base_pos, int max_k, float floor, std::vector<int>& drafts,
                      std::vector<float>& confs, float floor0) {
    drafts.clear();
    confs.clear();
    if (!mtp_loaded_ || max_k <= 0 || base_pos <= 0) return;
    Gpu& gl = gpus_.back();
    const int E = c_.n_embd;

    // Catch up MTP MLA KV cache for [mtp_done_ .. base_pos - 2] using exact trunk hidden states from hring.
    // RoPE/KV position matches mtp_check: (h[p], tok[p+1]) sits at pos = p + 1.
    const int lo = std::max(mtp_done_, std::max(0, base_pos - MtpWeights::kRing + 1));
    for (int p = lo; p < base_pos - 1; ++p) {
        if (p + 1 >= (int) mtp_tokens_.size()) break;
        const float* h_p = mtp_.hring + (size_t) (p % MtpWeights::kRing) * E;
        const int next_t = mtp_tokens_[(size_t) (p + 1)];
        (void) mtp_step(h_p, next_t, p + 1);
    }

    int cur_tok = base_tok;
    const float* cur_h = mtp_.hring + (size_t) ((base_pos - 1) % MtpWeights::kRing) * E;
    for (int i = 0; i < max_k; ++i) {
        const int p_mtp = base_pos + i;
        if (p_mtp >= o_.ctx - 1) break;
        const float* lg = mtp_step(cur_h, cur_tok, p_mtp);
        set_dev(gl.dev);
        k::top1_prob_f32(lg, c_.vocab, gl.d_arg_val, gl.d_arg_idx, gl.d_top_pair, gl.s);
        CK(cudaMemcpyAsync(gl.h_top_pair, gl.d_top_pair, 2 * sizeof(float), cudaMemcpyDeviceToHost, gl.s));
        CK(cudaStreamSynchronize(gl.s));
        int id = 0;
        std::memcpy(&id, &gl.h_top_pair[0], sizeof(int));
        const float prob = gl.h_top_pair[1];
        if (id < 0 || id >= c_.vocab || !std::isfinite(prob)) break;
        if (i == 0 && floor0 > 0.0f && prob < floor0) break;
        drafts.push_back(id);
        confs.push_back(prob);
        if (prob < floor || i + 1 == max_k) break;
        // Autoregressive chain: feed blk.42 post-out_norm hidden state (gl.hn) as next input h
        CK(cudaMemcpyAsync(mtp_.h_tmp, gl.hn, (size_t) E * sizeof(float), cudaMemcpyDeviceToDevice, gl.s));
        cur_h = mtp_.h_tmp;
        cur_tok = id;
    }
    mtp_done_ = base_pos;
}

// ------------------------------ MTP block ------------------------------

const float* Model::mtp_step(const float* h_nextn_dev, int next_token, int pos) {
    if (!mtp_loaded_) throw std::runtime_error("mtp_step called without --mtp");
    const Config& c = c_;
    const int E = c.n_embd;
    Gpu& g0 = gpus_[0];
    Gpu& gl = gpus_.back();

    set_dev(g0.dev);
    k::set_step(g0.d_step, next_token, pos, pos + 1, rope_delta_, g0.s);
    if (!k::emb_rows(emb_.p, emb_.type, emb_.k, g0.d_step, 1, g0.dev == gl.dev ? mtp_.emb_buf : g0.out, g0.s))
        throw std::runtime_error("mtp: emb_rows failed");
    if (g0.dev != gl.dev) {
        transfer_buf(0, (int) gpus_.size() - 1, g0.out, mtp_.emb_buf, (size_t) E);
    } else {
        set_dev(gl.dev);
    }

    k::set_step(gl.d_step, next_token, pos, pos + 1, rope_delta_, gl.s);

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
    k::argmax_f32(logits_dev, c_.vocab, gl.d_arg_val, gl.d_arg_idx, gl.d_arg_out, gl.s);
    CK(cudaMemcpyAsync(gl.h_arg_out, gl.d_arg_out, sizeof(int), cudaMemcpyDeviceToHost, gl.s));
    CK(cudaStreamSynchronize(gl.s));
    return *gl.h_arg_out;
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
    st.admitted += cpu_admitted_;
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


// ------------------------------ Vision (lent-VRAM arena, zero permanent VRAM) ------------------------------

const char* Model::vision_room() const {
    if (!vision_ || !vision_->loaded()) return "none";
    const Gpu& g0 = gpus_[0];
    if (g0.vram_pool != nullptr && g0.vram_pool_bytes >= vision_->max_room_bytes()) return "tail";
    return "free";
}

void Model::set_images(std::vector<Image> imgs) {
    images_ = std::move(imgs);
}

bool Model::encode_image(const uint8_t* rgb, int width, int height, std::vector<float>& out_rows, std::string& err) {
    if (!vision_ || !vision_->loaded()) {
        err = "vision tower not loaded (pass --mmproj PATH)";
        return false;
    }
    Gpu& g0 = gpus_[0];
    set_dev(g0.dev);
    CK(cudaStreamSynchronize(g0.s));
    CK(cudaStreamSynchronize(g0.s_copy));

    const size_t need = vision_->room_of(width, height);
    uint8_t* lent = nullptr;
    size_t lent_bytes = 0;
    size_t cut = g0.vram_pool_bytes;
    struct RestoreLayer {
        Layer* l;
        std::vector<int> owners;
        int sl_first;
    };
    std::vector<RestoreLayer> to_restore;

    if (g0.vram_pool != nullptr && g0.vram_pool_bytes >= need) {
        cut = g0.vram_pool_bytes - need;
        bool can_lend = true;
        for (int il = std::max(g0.l0, c_.n_dense_lead); il < g0.l1; ++il) {
            Layer& l = layers_[(size_t) il];
            if (l.n_res <= 0) continue;
            const size_t end_off = l.vram_pool_off + (size_t) l.n_res * l.blob;
            if (end_off <= cut) continue;
            if (l.host_blobs == nullptr) {
                can_lend = false;
                break;
            }
            const int sl_first = l.vram_pool_off >= cut ? 0 : (int) ((cut - l.vram_pool_off) / l.blob);
            RestoreLayer rl{&l, std::vector<int>((size_t) l.n_res, -1), sl_first};
            CK(cudaMemcpy(rl.owners.data(), l.fl.owner, (size_t) l.n_res * sizeof(int), cudaMemcpyDeviceToHost));
            to_restore.push_back(std::move(rl));
        }
        if (can_lend && mtp_loaded_ && gpus_.size() == 1 && mtp_.layer.n_res > 0) {
            Layer& ml = mtp_.layer;
            const size_t end_off = ml.vram_pool_off + (size_t) ml.n_res * ml.blob;
            if (end_off > cut) {
                if (ml.host_blobs == nullptr) {
                    can_lend = false;
                } else {
                    const int sl_first = ml.vram_pool_off >= cut ? 0 : (int) ((cut - ml.vram_pool_off) / ml.blob);
                    RestoreLayer rl{&ml, std::vector<int>((size_t) ml.n_res, -1), sl_first};
                    CK(cudaMemcpy(rl.owners.data(), ml.fl.owner, (size_t) ml.n_res * sizeof(int), cudaMemcpyDeviceToHost));
                    to_restore.push_back(std::move(rl));
                }
            }
        }
        if (can_lend) {
            lent = g0.vram_pool + cut;
            lent_bytes = need;
        } else {
            to_restore.clear();
        }
    }

    const bool ok = vision_->encode(rgb, width, height, out_rows, err, lent, lent_bytes);

    // Restore every borrowed resident expert slot from host_blobs before any subsequent prefill/decode
    if (lent != nullptr && !to_restore.empty()) {
        set_dev(g0.dev);
        for (const auto& rl : to_restore) {
            Layer& l = *rl.l;
            for (int sl = rl.sl_first; sl < l.n_res; ++sl) {
                const int e = rl.owners[(size_t) sl];
                if (e >= 0 && e < c_.n_expert) {
                    CK(cudaMemcpyAsync(l.fl.vram + (size_t) sl * l.blob, l.host_blobs + (size_t) e * l.blob, l.blob,
                                       cudaMemcpyHostToDevice, g0.s));
                }
            }
        }
        CK(cudaStreamSynchronize(g0.s));
    }
    return ok;
}

}  // namespace ling
