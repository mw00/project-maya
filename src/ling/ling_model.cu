// src/ling/ling_model.cu - Ling-3.2-air (bailingmoe3) loader, decode (CUDA graph / GPU cache / CPU lane),
// double-buffered batched prefill (KDA + MLA WMMA + grouped MoE), multi-GPU layer split, and optional MTP block.
#include "ling_model.hpp"

#include "ling_dense.hpp"
#include "ling_kernels.cuh"
#include "ling_moe_plan.hpp"
#include "ling_parity.hpp"
#include "ling_sys.hpp"
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
#include <regex>
#include <stdexcept>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <filesystem>
#include <fstream>
#include <sstream>
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

// A GGUF whole or in llama.cpp's shards (NAME-00001-of-0000N.gguf ...): the tensors of every shard in one list, each
// read from the shard that holds it; metadata from the first shard.  Any shard's name opens the whole set; each shard's
// split.no / split.count must agree with its name, and the tensor names must be unique across the set.
class GgufSet {
public:
    explicit GgufSet(const std::string& path) {
        namespace fs = std::filesystem;
        const fs::path p(path);
        const std::string base = p.filename().string();
        static const std::regex shard_re(R"(^(.*)-([0-9]{5})-of-([0-9]{5})\.gguf$)");
        std::smatch m;
        int n = 1;
        if (std::regex_match(base, m, shard_re)) {
            const int no = std::stoi(m[2].str());
            n = std::stoi(m[3].str());
            if (n < 1 || no < 1 || no > n)
                throw std::runtime_error("GGUF: " + base + ": shard " + m[2].str() + " of " + m[3].str() + " is out of range");
            for (int i = 1; i <= n; ++i) {
                char tail[32];
                std::snprintf(tail, sizeof tail, "-%05d-of-%05d.gguf", i, n);
                paths_.push_back((p.parent_path() / (m[1].str() + tail)).string());
            }
        } else {
            paths_.push_back(path);
        }
        size_t total = 0;
        for (size_t f = 0; f < paths_.size(); ++f) {
            if (n > 1 && !fs::exists(paths_[f]))
                throw std::runtime_error("GGUF: shard " + std::to_string(f + 1) + " of " + std::to_string(n) +
                                         " is missing: " + paths_[f]);
            files_.push_back(std::make_unique<maya::GgufFile>(paths_[f]));
            const maya::MetaValue* sc = files_.back()->get("split.count");
            const maya::MetaValue* sn = files_.back()->get("split.no");
            const uint64_t count = (sc != nullptr && sc->is_num()) ? sc->u : 1;
            const uint64_t no = (sn != nullptr && sn->is_num()) ? sn->u : 0;
            if (count != (uint64_t) n || no != (uint64_t) f)
                throw std::runtime_error("GGUF: " + paths_[f] + ": its split metadata says shard " + std::to_string(no + 1) +
                                         " of " + std::to_string(count) + ", its name shard " + std::to_string(f + 1) +
                                         " of " + std::to_string(n) +
                                         (n == 1 && count > 1 ? " (open a split model by its -NNNNN-of-NNNNN.gguf name)" : ""));
            total += files_.back()->tensors().size();
        }
        all_.reserve(total);
        owner_.reserve(total);
        for (size_t f = 0; f < files_.size(); ++f)
            for (const auto& t : files_[f]->tensors()) {
                if (!index_.emplace(t.name, all_.size()).second)
                    throw std::runtime_error("GGUF: tensor " + t.name + " appears twice (" + paths_[f] + ")");
                all_.push_back(t);
                owner_.push_back((int) f);
            }
        const maya::MetaValue* tc = files_[0]->get("split.tensors.count");
        if (tc != nullptr && tc->is_num() && tc->u != (uint64_t) all_.size())
            throw std::runtime_error("GGUF: the shards hold " + std::to_string(all_.size()) + " tensors, split.tensors.count says " +
                                     std::to_string(tc->u));
    }
    const std::vector<maya::TensorInfo>& tensors() const { return all_; }
    const uint8_t* tensor_data(const maya::TensorInfo& t) const { return files_[(size_t) file_of(t)]->tensor_data(t); }
    uint64_t tensor_file_offset(const maya::TensorInfo& t) const {
        return files_[(size_t) file_of(t)]->data_start() + t.offset;
    }
    int file_of(const maya::TensorInfo& t) const {
        const auto it = index_.find(t.name);
        if (it == index_.end()) throw std::runtime_error("GGUF: " + t.name + ": not a tensor of this file set");
        return owner_[it->second];
    }
    const std::vector<std::string>& paths() const { return paths_; }
    const maya::MetaValue* get(const std::string& key) const { return files_[0]->get(key); }
    bool get_full(const std::string& key, maya::MetaValue& out) const { return files_[0]->get_full(key, out); }

private:
    std::vector<std::unique_ptr<maya::GgufFile>> files_;
    std::vector<std::string> paths_;
    std::vector<maya::TensorInfo> all_;
    std::vector<int> owner_;
    std::unordered_map<std::string, size_t> index_;
};

// ------------------------------ metadata ------------------------------

static double req_num(const GgufSet& g, const std::string& key) {
    const maya::MetaValue* v = g.get(key);
    if (v == nullptr || !v->is_num()) throw std::runtime_error("GGUF: missing key " + key);
    return v->num();
}

static double opt_num(const GgufSet& g, const std::string& key, double dflt) {
    const maya::MetaValue* v = g.get(key);
    return v != nullptr && v->is_num() ? v->num() : dflt;
}

static std::vector<double> get_array(const GgufSet& g, const std::string& key, bool required) {
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

static Config parse_config(const GgufSet& g) {
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
    c.ctx_train = (int) opt_num(g, a + "context_length", 262144);
    c.eos_id = (int) opt_num(g, "tokenizer.ggml.eos_token_id", -1);
    auto add_stop = [&](int id) {
        if (id >= 0 && std::find(c.stop_ids.begin(), c.stop_ids.end(), id) == c.stop_ids.end())
            c.stop_ids.push_back(id);
    };
    add_stop(c.eos_id);
    add_stop((int) opt_num(g, "tokenizer.ggml.eot_token_id", -1));
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

void Model::check_token_id(int tok, const char* where) const {
    if (tok < 0 || tok >= c_.vocab) {
        throw std::runtime_error(std::string(where) + ": token id " + std::to_string(tok) +
                                 " out of range [0, " + std::to_string(c_.vocab) + ")");
    }
}

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

// One line, the first time a full-precision matrix is turned into Q8_0 (not one for each of the layers' matrices).
static void note_dense_converted(const maya::TensorInfo& t) {
    static std::atomic<bool> said{false};
    if (said.exchange(true)) return;
    std::fprintf(stderr, "ling load: %s is %s - full-precision matrices are quantised to Q8_0 as they load "
                 "(the format the kernels read)\n", t.name.c_str(), std::string(t.type_name()).c_str());
}

void Model::load_layer_weights(Layer& l, int il, Gpu& g, const GgufSet& gf,
                               const std::map<std::string, const maya::TensorInfo*>& want) {
    set_dev(g.dev);
    auto at = [&](const std::string& n) -> const maya::TensorInfo& { return *want.at(n); };
    auto dev_copy = [&](const maya::TensorInfo& t) -> const uint8_t* {
        const size_t bytes = tensor_bytes(t);
        void* p = dev_alloc(g, bytes);
        CK(cudaMemcpyAsync(p, gf.tensor_data(t), bytes, cudaMemcpyHostToDevice, g.s));
        return (const uint8_t*) p;
    };
    auto dev_float = [&](const maya::TensorInfo& t) -> const float* {
        if (t.type != 0) throw std::runtime_error("GGUF: " + t.name + ": F32 expected");
        return (const float*) dev_copy(t);
    };
    // an F32 / F16 / BF16 matrix: Q8_0 rows from the load on (ling_dense.hpp), the type the kernels read
    auto dev_q8 = [&](const maya::TensorInfo& t, int k) -> const uint8_t* {
        note_dense_converted(t);
        const std::vector<uint8_t> q = ling::dense::to_q8_0((int) t.type, gf.tensor_data(t), t.elements(), (uint64_t) k, t.name);
        void* p = dev_alloc(g, q.size());
        CK(cudaMemcpyAsync(p, q.data(), q.size(), cudaMemcpyHostToDevice, g.s));
        CK(cudaStreamSynchronize(g.s));   // (q ends here)
        return (const uint8_t*) p;
    };
    auto dev_mat = [&](const maya::TensorInfo& t, int k, int n) -> Mat {
        expect_shape(t, {k, n});
        Mat m;
        if (ling::dense::is_float_type((int) t.type)) {
            m.p = dev_q8(t, k);
            m.type = ling::dense::kQ8_0;
        } else {
            if (!iq::mv_supported((int) t.type))
                throw std::runtime_error("GGUF: " + t.name + ": type " + std::string(t.type_name()) + " not supported");
            m.p = dev_copy(t);
            m.type = (int) t.type;
        }
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
        CK(cudaMemcpyAsync(pa, a_neg.data(), a_neg.size() * sizeof(float), cudaMemcpyHostToDevice, g.s));
        CK(cudaStreamSynchronize(g.s));
        l.ssm_a_neg = (const float*) pa;

        const size_t cs = (size_t) 3 * (size_t) (c.d_conv - 1) * (size_t) DI * sizeof(float);
        l.conv_state = (float*) dev_alloc(g, cs);
        CK(cudaMemsetAsync(l.conv_state, 0, cs, g.s));
        const size_t st = (size_t) H * (size_t) c.kda_head * (size_t) c.kda_head * sizeof(float);
        l.state = (float*) dev_alloc(g, st);
        CK(cudaMemsetAsync(l.state, 0, st, g.s));
        // Always allocate window KDA scratch so prompt lookup works without --mtp and trunk VRAM plan is MTP-invariant
        l.conv_bak = (float*) dev_alloc(g, cs);
        l.state_tmp = (float*) dev_alloc(g, st);
        const size_t wb = ((size_t) 6 * kMaxWin * DI + (size_t) kMaxWin * H) * sizeof(float);
        l.w_kda_buf = (float*) dev_alloc(g, wb);
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
        if (tr.type != 0 && tr.type != 30 && tr.type != 1)
            throw std::runtime_error("GGUF: " + tr.name + ": F32, F16 or BF16 expected");
        if (tr.type == 1) {                      // F16 -> F32 is exact; the kernels read the router in F32 or BF16
            const std::vector<float> rf = ling::dense::to_f32(1, gf.tensor_data(tr), tr.elements());
            void* rp = dev_alloc(g, rf.size() * sizeof(float));
            CK(cudaMemcpyAsync(rp, rf.data(), rf.size() * sizeof(float), cudaMemcpyHostToDevice, g.s));
            CK(cudaStreamSynchronize(g.s));
            l.router_w = (const uint8_t*) rp;
            l.router_type = 0;
        } else {
            l.router_w = dev_copy(tr);
            l.router_type = (int) tr.type;
        }
        const maya::TensorInfo& tb = at(b + "exp_probs_b.bias");
        expect_shape(tb, {c.n_expert});
        l.rb_dev = dev_float(tb);
        l.fg = dev_mat(at(b + "ffn_gate_shexp.weight"), E, c.n_ff_shexp);
        l.fu = dev_mat(at(b + "ffn_up_shexp.weight"), E, c.n_ff_shexp);
        l.fd = dev_mat(at(b + "ffn_down_shexp.weight"), c.n_ff_shexp, E);
    }
}

// ------------------------------ constructor ------------------------------

Model::Model(const std::string& path, const Options& opt) : o_(opt), gguf_path_(path) {
    if (o_.gpus.empty()) o_.gpus = {0};
    if (o_.repeatable) {
        o_.lane_parity = true;
        o_.use_graph = false;
        o_.cache_warm = false;
        o_.lookahead = 0;
    }
    if (sys::systemd_oomd_active()) {
        std::fprintf(stderr, "maya-ling: WARNING systemd-oomd is active; large host expert arenas may trigger early OOM kills\n");
    }
    {
        std::string hdd_dev;
        if (sys::is_rotational_disk(path, hdd_dev)) {
            std::fprintf(stderr, "maya-ling: WARNING model file %s sits on rotational disk (%s); expect slow expert reads\n",
                         path.c_str(), hdd_dev.c_str());
        }
    }
    cpu_topo_ = sys::detect_cpu_topology();
    if (o_.cpu_threads < 0) o_.cpu_threads = sys::default_cpu_threads(cpu_topo_);

    // Default GPU auto-sort (disable with --no-auto-sort-gpus): headless cards first, then larger VRAM first
    // (bucketed to 512 MiB so identical cards keep their original index order).
    if (o_.auto_sort_gpus && o_.gpus.size() > 1) {
        struct GpuRank { int dev; int display; uint64_t vram_bucket; };
        std::vector<GpuRank> rk;
        for (int d : o_.gpus) {
            cudaSetDevice(d);
            size_t fb = 0, tb = 0;
            cudaMemGetInfo(&fb, &tb);
            int tmo = 0;   // (cudaDeviceProp lost kernelExecTimeoutEnabled in CUDA 13; the attribute is in 12 and 13)
            int disp = (cudaDeviceGetAttribute(&tmo, cudaDevAttrKernelExecTimeout, d) == cudaSuccess && tmo) ? 1 : 0;
            const uint64_t bucket = (tb > 0 ? (uint64_t) tb : (uint64_t) fb) >> 29;
            rk.push_back({d, disp, bucket});
        }
        std::stable_sort(rk.begin(), rk.end(), [](const GpuRank& a, const GpuRank& b) {
            if (a.display != b.display) return a.display < b.display;
            return a.vram_bucket > b.vram_bucket;
        });
        o_.gpus.clear();
        for (const auto& r : rk) o_.gpus.push_back(r.dev);
    }
    auto gf_u = std::make_unique<GgufSet>(path);
    GgufSet& gf = *gf_u;
    const maya::MetaValue* arch = gf.get("general.architecture");
    if (arch == nullptr || arch->s != "bailingmoe3")
        throw std::runtime_error("GGUF: not a bailingmoe3 file (" + path + ")");
    c_ = parse_config(gf);
    const Config& c = c_;

    if (o_.ctx < 1) throw std::runtime_error("maya-ling: --ctx must be >= 1");
    {
        maya::MetaValue vt;
        if (gf.get_full("tokenizer.ggml.tokens", vt) && !vt.items.empty()) {
            vocab_tokens_.reserve(vt.items.size());
            for (size_t i = 0; i < vt.items.size(); ++i) {
                const std::string& ts = vt.items[i].s;
                vocab_tokens_.push_back(ts);
                if (ts == "<|role_end|>" || ts == "<|endoftext|>" || ts == "<|im_end|>" || ts == "<|eot_id|>") {
                    const int id = (int) i;
                    if (std::find(c_.stop_ids.begin(), c_.stop_ids.end(), id) == c_.stop_ids.end())
                        c_.stop_ids.push_back(id);
                }
            }
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
#ifndef MAYA_USE_HIP
        cudaSetDeviceFlags(cudaDeviceMapHost);
#endif
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
        char bus_id[64] = {};
        if (cudaDeviceGetPCIBusId(bus_id, sizeof bus_id, gp.dev) == cudaSuccess) {
            gp.numa_node = sys::gpu_numa_node(bus_id);
        }
        if (gp.numa_node < 0 && !cpu_topo_.node_ids.empty()) gp.numa_node = cpu_topo_.node_ids[0];
        const int m0 = (n_moe * gi) / ngpu;
        const int m1 = (n_moe * (gi + 1)) / ngpu;
        gp.l0 = gi == 0 ? 0 : (c.n_dense_lead + m0);
        gp.l1 = c.n_dense_lead + m1;
    }
    if (ngpu > 1) {
        std::ostringstream go;
        for (int gi = 0; gi < ngpu; ++gi) {
            if (gi > 0) go << ",";
            go << gpus_[(size_t) gi].dev;
        }
        std::fprintf(stderr, "ling gpus: order=[%s] auto_sort=%s\n", go.str().c_str(),
                     o_.auto_sort_gpus ? "on" : "off");
    }
    if (cpu_topo_.numa_nodes > 1) {
        std::ostringstream nm;
        for (int gi = 0; gi < ngpu; ++gi) {
            if (gi > 0) nm << ",";
            nm << "gpu" << gpus_[(size_t) gi].dev << "->n" << gpus_[(size_t) gi].numa_node;
        }
        std::fprintf(stderr, "ling numa: %s; pinning GPU workers & host expert arenas (%s)\n",
                     cpu_topo_.summary.c_str(), nm.str().c_str());
    } else {
        std::fprintf(stderr,
                     "ling numa: %s; single-node host (hy4::Lane pins CPU workers to physical cores; cross-node pinning inactive)\n",
                     cpu_topo_.summary.c_str());
    }

    // Startup bit-for-bit self-check of fused kernels (refuses start on any mismatch)
    {
        set_dev(gpus_[0].dev);
        std::string ferr;
        if (!k::verify_fused_kernels(gpus_[0].s, ferr)) {
            throw std::runtime_error("maya-ling: fused kernel self-check FAILED: " + ferr);
        }
    }

    p2p_mat_.assign((size_t) ngpu, std::vector<bool>((size_t) ngpu, false));
    for (int gi = 0; gi < ngpu; ++gi) p2p_mat_[(size_t) gi][(size_t) gi] = true;
    if (ngpu > 1) peer_setup_verified();

    const int E = c.n_embd, H = c.n_head, DI = c.d_inner(), KL = c.kv_lora, R = c.n_rot, FE = c.n_ff_exp;

    // Embeddings & output head (async on GPU streams - zero NULL-stream copies)
    {
        const maya::TensorInfo& te = *want.at("token_embd.weight");
        if (te.shape.size() != 2 || (int) te.shape[0] != E)
            throw std::runtime_error("GGUF: unexpected token_embd shape");
        c_.vocab = (int) te.shape[1];
        c_.stop_ids.erase(
            std::remove_if(c_.stop_ids.begin(), c_.stop_ids.end(), [&](int id) { return id < 0 || id >= c_.vocab; }),
            c_.stop_ids.end());
        if ((c_.eos_id < 0 || c_.eos_id >= c_.vocab) && !c_.stop_ids.empty()) c_.eos_id = c_.stop_ids[0];
        set_dev(gpus_[0].dev);
        void* pe = dev_alloc(gpus_[0], tensor_bytes(te));
        CK(cudaMemcpyAsync(pe, gf.tensor_data(te), tensor_bytes(te), cudaMemcpyHostToDevice, gpus_[0].s));
        emb_ = {(const uint8_t*) pe, (int) te.type, E, c_.vocab, 0};

        Gpu& glast = gpus_.back();
        set_dev(glast.dev);
        const maya::TensorInfo& ton = *want.at("output_norm.weight");
        if (ton.type != 0) throw std::runtime_error("GGUF: output_norm F32 expected");
        void* pon = dev_alloc(glast, tensor_bytes(ton));
        CK(cudaMemcpyAsync(pon, gf.tensor_data(ton), tensor_bytes(ton), cudaMemcpyHostToDevice, glast.s));
        out_norm_ = (const float*) pon;

        const maya::TensorInfo& tow = *want.at("output.weight");
        expect_shape(tow, {E, c_.vocab});
        if (ling::dense::is_float_type((int) tow.type)) {   // F32 / F16 / BF16 head: Q8_0 rows (ling_dense.hpp)
            note_dense_converted(tow);
            const std::vector<uint8_t> q = ling::dense::to_q8_0((int) tow.type, gf.tensor_data(tow), tow.elements(), (uint64_t) E, tow.name);
            void* pow = dev_alloc(glast, q.size());
            CK(cudaMemcpyAsync(pow, q.data(), q.size(), cudaMemcpyHostToDevice, glast.s));
            CK(cudaStreamSynchronize(glast.s));
            out_w_ = {(const uint8_t*) pow, ling::dense::kQ8_0, E, c_.vocab, iq::row_bytes(ling::dense::kQ8_0, E)};
        } else {
            if (!iq::mv_supported((int) tow.type))
                throw std::runtime_error("GGUF: output.weight type " + std::string(tow.type_name()) + " not supported");
            void* pow = dev_alloc(glast, tensor_bytes(tow));
            CK(cudaMemcpyAsync(pow, gf.tensor_data(tow), tensor_bytes(tow), cudaMemcpyHostToDevice, glast.s));
            out_w_ = {(const uint8_t*) pow, (int) tow.type, E, c_.vocab, iq::row_bytes((int) tow.type, E)};
        }
    }

    // Load trunk layers in parallel across GPUs (1 thread per GPU)
    layers_.resize((size_t) c.n_layers);
    if (ngpu > 1) {
        std::vector<std::thread> lthreads;
        std::mutex err_mu;
        std::string load_err;
        for (int gi = 0; gi < ngpu; ++gi) {
            lthreads.emplace_back([&, gi] {
                try {
                    Gpu& gp = gpus_[(size_t) gi];
                    (void) sys::pin_thread_to_numa_node(cpu_topo_, gp.numa_node);
                    for (int il = gp.l0; il < gp.l1; ++il) {
                        Layer& l = layers_[(size_t) il];
                        l.dev_idx = gi;
                        l.kda = c.kda[il];
                        load_layer_weights(l, il, gp, gf, want);
                    }
                    set_dev(gp.dev);
                    CK(cudaStreamSynchronize(gp.s));
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lk(err_mu);
                    if (load_err.empty()) load_err = e.what();
                }
            });
        }
        for (auto& th : lthreads) th.join();
        if (!load_err.empty()) throw std::runtime_error(load_err);
    } else {
        Gpu& gp = gpus_[0];
        for (int il = gp.l0; il < gp.l1; ++il) {
            Layer& l = layers_[(size_t) il];
            l.dev_idx = 0;
            l.kda = c.kda[il];
            load_layer_weights(l, il, gp, gf, want);
        }
        set_dev(gp.dev);
        CK(cudaStreamSynchronize(gp.s));
    }

    max_blob_ = 0;
    int quant_tg = -1, quant_td = -1;
    std::string quant_mix;   // (the layers whose up's type is not the gate's, and that type: restore / calibration keys)
    for (int il = c.n_dense_lead; il < c.n_layers; ++il) {
        Layer& l = layers_[(size_t) il];
        const std::string b = "blk." + std::to_string(il) + ".";
        const maya::TensorInfo& tg = *want.at(b + "ffn_gate_exps.weight");
        const maya::TensorInfo& tu = *want.at(b + "ffn_up_exps.weight");
        const maya::TensorInfo& td = *want.at(b + "ffn_down_exps.weight");
        expect_shape(tg, {E, FE, c.n_expert});
        expect_shape(tu, {E, FE, c.n_expert});
        expect_shape(td, {FE, E, c.n_expert});
        // (gate and up of different types - Maya-M / Maya-L give a few layers a larger up: the kernels' two passes,
        //  the gate's sums then the up's; such a layer's experts never take the CPU lane, which has one type for both)
        if (!iq::expert_supported((int) tg.type) || !iq::expert_supported((int) tu.type) ||
            !iq::down_supported((int) td.type))
            throw std::runtime_error("GGUF: " + b + "expert type not supported");
        if (tg.type != tu.type) {
            if (FE > E)   // (the prompt sets keep the gate's F32 sums in their down output's rows: n_ff <= n_embd)
                throw std::runtime_error("GGUF: " + b + "gate and up expert types differ (n_ff > n_embd)");
            quant_mix += b + std::to_string((int) tu.type) + ";";
        }
        l.etg = (int) tg.type;
        l.etu = (int) tu.type;
        l.etd = (int) td.type;
        quant_tg = l.etg;
        quant_td = l.etd;
        l.eg_ld = iq::row_bytes(l.etg, E);
        l.eu_ld = iq::row_bytes(l.etu, E);
        l.ed_ld = iq::row_bytes(l.etd, FE);
        l.eg_stride = l.eg_ld * (size_t) FE;
        l.eu_stride = l.eu_ld * (size_t) FE;
        l.ed_stride = l.ed_ld * (size_t) E;
        l.gate_file_off = gf.tensor_file_offset(tg);
        l.up_file_off = gf.tensor_file_offset(tu);
        l.down_file_off = gf.tensor_file_offset(td);
        l.gate_file = gf.file_of(tg);
        l.up_file = gf.file_of(tu);
        l.down_file = gf.file_of(td);
        l.tr[0] = 0;
        l.tr[1] = align256(l.eg_stride);
        l.tr[2] = align256(l.tr[1] + l.eu_stride);
        l.blob = align256(l.tr[2] + l.ed_stride);
        max_blob_ = std::max(max_blob_, l.blob);
    }

    // Verify loaded expert quant formats against ggml-cpu at first use
    if (quant_tg >= 0) {
        std::string vmsg;
        bool ok_g = par::verify_format_once(quant_tg, vmsg);
        std::fprintf(stderr, "ling cpu-check: %s (%s)\n", vmsg.c_str(), ok_g ? "exact" : "ggml-fallback");
        if (quant_td != quant_tg) {
            bool ok_d = par::verify_format_once(quant_td, vmsg);
            std::fprintf(stderr, "ling cpu-check: %s (%s)\n", vmsg.c_str(), ok_d ? "exact" : "ggml-fallback");
        }
    }

    // Allocate trunk decode + window + prefill scratch BEFORE MTP load so trunk plan is 100% invariant to --mtp!
    const size_t reserve_b = o_.reserve_gb > 0.0 ? (size_t) (o_.reserve_gb * (double) (1ULL << 30)) : 0u;
    int P = std::max(1, std::min(o_.prefill_chunk, o_.ctx));
    if (o_.repeatable) {
        P = std::min(P, 512);
    } else {
        size_t min_free_b = (size_t) -1;
        for (const auto& gp : gpus_) {
            set_dev(gp.dev);
            size_t fb = 0, tb = 0;
            if (cudaMemGetInfo(&fb, &tb) == cudaSuccess) {
                const size_t eff = fb > reserve_b ? fb - reserve_b : 0u;
                min_free_b = std::min(min_free_b, eff);
            }
        }
        if (min_free_b < (size_t) 3ULL << 30) P = std::min(P, 128);
        else if (min_free_b < (size_t) 6ULL << 30) P = std::min(P, 256);
        else if (min_free_b < (size_t) 10ULL << 30) P = std::min(P, 512);
        else if (min_free_b < (size_t) 14ULL << 30) P = std::min(P, 1024);
    }
    const size_t mla_part_b = std::max(glmd::mla_dec_part_bytes(H), k::mla_dec_wmma_part_bytes(H, kMaxWin));
    const size_t max_w16_elems = std::max<size_t>(
        {(size_t) H * c.qk_head * (size_t) E, (size_t) c.n_ff * (size_t) E, (size_t) H * (size_t) KL * c.qk_nope,
         (size_t) H * (size_t) c.v_head * (size_t) KL, (size_t) DI * (size_t) E});
    // Always allocate Wcap = kMaxWin (~18 MiB) so prompt lookup works without MTP and trunk VRAM plan is MTP-invariant!
    const int Wcap = kMaxWin;
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
        const size_t eh_rows = (size_t) Wcap * 16;
        gp.eh = (float*) dev_alloc(gp, std::max(eh_rows * (size_t) FE, (size_t) Wcap * c.n_ff) * 4);
        gp.shact = (float*) dev_alloc(gp, (size_t) Wcap * c.n_ff_shexp * 4);
        gp.part = (float*) dev_alloc(gp, eh_rows * (size_t) E * 4);
        gp.shout = (float*) dev_alloc(gp, (size_t) Wcap * E * 4);
        gp.moe = (float*) dev_alloc(gp, (size_t) Wcap * E * 4);
        gp.rlog = (float*) dev_alloc(gp, (size_t) Wcap * c.n_expert * 4);
        gp.d_w = (float*) dev_alloc(gp, (size_t) Wcap * 16 * 4);
        gp.d_ids = (int*) dev_alloc(gp, (size_t) Wcap * 16 * 4);
        gp.d_route_err = (int*) dev_alloc(gp, sizeof(int));
        gp.h_route_err = (int*) host_alloc(gp, sizeof(int));
        *gp.h_route_err = 0;
        CK(cudaMemsetAsync(gp.d_route_err, 0, sizeof(int), gp.s));
        gp.h_stats = (unsigned long long*) host_alloc(gp, 8 * sizeof(unsigned long long));
        gp.d_plan = (unsigned long long*) dev_alloc(gp, (size_t) 16 * sizeof(unsigned long long));
        gp.d_cpu_out = (float*) dev_alloc(gp, (size_t) Wcap * E * 4);
        gp.ehq = (int8_t*) dev_alloc(gp, std::max(eh_rows * (size_t) FE, (size_t) Wcap * c.n_ff));
        gp.ehd = (float*) dev_alloc(gp, (std::max(eh_rows * (size_t) FE, (size_t) Wcap * c.n_ff) / 32) * 4);
        gp.shq = (int8_t*) dev_alloc(gp, (size_t) Wcap * c.n_ff_shexp);
        gp.shd = (float*) dev_alloc(gp, (size_t) Wcap * (c.n_ff_shexp / 32) * 4);
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
        gp.d_step = (int*) dev_alloc(gp, k::kStepInts * sizeof(int));
        gp.d_iota = (int*) dev_alloc(gp, (size_t) o_.ctx * sizeof(int));
        k::fill_iota(gp.d_iota, o_.ctx, gp.s);
        k::mla_dec_wmma_init();

        gp.fxs.eptr = (unsigned long long*) dev_alloc(gp, (size_t) c.n_expert * sizeof(unsigned long long));
        gp.fxs.who = (int*) dev_alloc(gp, (size_t) Wcap * 16 * sizeof(int));
        gp.fxs.src = (unsigned long long*) dev_alloc(gp, (size_t) Wcap * 16 * sizeof(unsigned long long));
        gp.fxs.dst = (unsigned long long*) dev_alloc(gp, (size_t) Wcap * 16 * sizeof(unsigned long long));
        gp.fxs.nf = (unsigned*) dev_alloc(gp, sizeof(unsigned));
        gp.fxs.n_stage = std::max(c.n_used * 2, 16);
        gp.fxs.stage = (uint8_t*) dev_alloc(gp, (size_t) gp.fxs.n_stage * max_blob_);
        gp.fxs.stats = (unsigned long long*) dev_alloc(gp, 8 * sizeof(unsigned long long));
        CK(cudaMemsetAsync(gp.fxs.stats, 0, 8 * sizeof(unsigned long long), gp.s));

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
            if (ngpu >= 2) gp.pf_pipe_x = (float*) dev_alloc(gp, (size_t) P * E * 4);
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

    // ---- Plan per-expert VRAM tier for TRUNK layers FIRST (100% invariant to --mtp) ----
    const size_t lat_row_bytes = o_.kv_q8 ? (size_t) KL + (size_t) (KL / 32) * 2 : (size_t) KL * sizeof(uint16_t);
    const size_t kv_per_mla = (size_t) o_.ctx * (lat_row_bytes + (size_t) R * sizeof(uint16_t));
    const size_t kda_per_layer = ((size_t) 3 * (size_t) (c.d_conv - 1) * (size_t) DI +
                                  (size_t) H * (size_t) c.kda_head * (size_t) c.kda_head) *
                                 sizeof(float);
    std::vector<size_t> gpu_vram_used((size_t) ngpu, 0);
    size_t total_vram_experts = 0, total_experts = 0;
    double total_kv_b = 0.0;
    for (int gi = 0; gi < ngpu; ++gi) {
        Gpu& gp = gpus_[(size_t) gi];
        set_dev(gp.dev);
        size_t free_b = 0, total_b = 0;
        CK(cudaMemGetInfo(&free_b, &total_b));
        const size_t eff_free_b = free_b > reserve_b ? free_b - reserve_b : 0u;
        size_t gpu_kv_b = 0;
        for (int il = gp.l0; il < gp.l1; ++il) gpu_kv_b += c.kda[il] ? kda_per_layer : kv_per_mla;
        total_kv_b += (double) gpu_kv_b;
        const double used_before_gib = (double) (total_b - free_b) / (double) (1ULL << 30);
        const double kv_gib = (double) gpu_kv_b / (double) (1ULL << 30);
        const double eff_free_gib = (double) eff_free_b / (double) (1ULL << 30);
        // Exact format matched by serve/server.py StrataEngine._context_cost and serve/test_engine_contract.py
        std::fprintf(stderr,
                     "CUDA%d VRAM before the expert pool: used %.2f GiB, free %.2f GiB (reserve %.2f GiB), "
                     "state/KV (%d ctx) %.2f GiB\n",
                     gp.dev, used_before_gib, eff_free_gib, o_.reserve_gb, o_.ctx, kv_gib);

        // Leave 1.25 GiB headroom (includes optional MTP dense+KV ~0.35 GiB on last GPU so MTP never changes trunk plan!)
        const double headroom = 1.25 * (double) (1ULL << 30);
        double budget = o_.expert_vram_gib >= 0.0 ? o_.expert_vram_gib * (double) (1ULL << 30)
                                                  : std::max(0.0, (double) eff_free_b - headroom);
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
            l.vram_pool_off = gpu_vram_used[(size_t) gi];
            gpu_vram_used[(size_t) gi] += (size_t) nr * l.blob;
            total_vram_experts += (size_t) nr;
            total_experts += (size_t) c.n_expert;
        }
        gp.vram_pool_bytes = gpu_vram_used[(size_t) gi];
        if (gp.vram_pool_bytes > 0) {
            gp.vram_pool = (uint8_t*) dev_alloc(gp, gp.vram_pool_bytes);
        }
    }
    total_kv_gb_ = total_kv_b / (double) (1ULL << 30);

    {
        char spb[256];
        std::snprintf(spb, sizeof spb, "ling start-plan: ctx=%d kv=%s trunk_vram_slots=%zu/%zu reserve_gb=%.2f",
                      o_.ctx, o_.kv_q8 ? "q8" : "f16", total_vram_experts, total_experts, o_.reserve_gb);
        start_plan_line_ = spb;
        std::fprintf(stderr, "%s\n", start_plan_line_.c_str());
    }

    // ---- Load optional MTP block AFTER trunk pool planning (NEVER alters trunk kv or trunk n_res!) ----
    if (o_.load_mtp && c.n_nextn >= 1 && !mtp_want.empty()) {
        Gpu& glast = gpus_.back();
        set_dev(glast.dev);
        auto at = [&](const std::string& n) -> const maya::TensorInfo& { return *mtp_want.at(n); };
        auto dev_copy = [&](const maya::TensorInfo& t) -> const uint8_t* {
            void* p = dev_alloc(glast, tensor_bytes(t));
            CK(cudaMemcpyAsync(p, gf.tensor_data(t), tensor_bytes(t), cudaMemcpyHostToDevice, glast.s));
            return (const uint8_t*) p;
        };
        mtp_.enorm = (const float*) dev_copy(at(mtp_prefix + "nextn.enorm.weight"));
        mtp_.hnorm = (const float*) dev_copy(at(mtp_prefix + "nextn.hnorm.weight"));
        mtp_.out_norm = (const float*) dev_copy(at(mtp_prefix + "layer_output_norm.weight"));
        const maya::TensorInfo& teh = at(mtp_prefix + "nextn.eh_proj.weight");
        expect_shape(teh, {2 * E, E});
        if (ling::dense::is_float_type((int) teh.type)) {    // F32 / F16 / BF16: Q8_0 rows (ling_dense.hpp)
            note_dense_converted(teh);
            const std::vector<uint8_t> q = ling::dense::to_q8_0((int) teh.type, gf.tensor_data(teh), teh.elements(), (uint64_t) (2 * E), teh.name);
            void* pq = dev_alloc(glast, q.size());
            CK(cudaMemcpyAsync(pq, q.data(), q.size(), cudaMemcpyHostToDevice, glast.s));
            CK(cudaStreamSynchronize(glast.s));
            mtp_.eh_proj = {(const uint8_t*) pq, ling::dense::kQ8_0, 2 * E, E, iq::row_bytes(ling::dense::kQ8_0, 2 * E)};
        } else {
            if (!iq::mv_supported((int) teh.type))
                throw std::runtime_error("GGUF: " + teh.name + ": type " + std::string(teh.type_name()) + " not supported");
            mtp_.eh_proj = {dev_copy(teh), (int) teh.type, 2 * E, E, iq::row_bytes((int) teh.type, 2 * E)};
        }
        mtp_.layer.dev_idx = ngpu - 1;
        mtp_.layer.kda = false;
        load_layer_weights(mtp_.layer, c.n_layers, glast, gf, mtp_want);
        const maya::TensorInfo& tg = at(mtp_prefix + "ffn_gate_exps.weight");
        const maya::TensorInfo& tu = at(mtp_prefix + "ffn_up_exps.weight");
        const maya::TensorInfo& td = at(mtp_prefix + "ffn_down_exps.weight");
        if (tg.type != tu.type) {   // (as the trunk's layers: two passes on the GPU, never the CPU lane)
            if (!iq::expert_supported((int) tg.type) || !iq::expert_supported((int) tu.type) || FE > E)
                throw std::runtime_error("GGUF: " + mtp_prefix + "gate and up expert types differ (not supported)");
            quant_mix += mtp_prefix + std::to_string((int) tu.type) + ";";
        }
        mtp_.layer.etg = (int) tg.type;
        mtp_.layer.etu = (int) tu.type;
        mtp_.layer.etd = (int) td.type;
        mtp_.layer.eg_ld = iq::row_bytes(mtp_.layer.etg, E);
        mtp_.layer.eu_ld = iq::row_bytes(mtp_.layer.etu, E);
        mtp_.layer.ed_ld = iq::row_bytes(mtp_.layer.etd, FE);
        mtp_.layer.eg_stride = mtp_.layer.eg_ld * (size_t) FE;
        mtp_.layer.eu_stride = mtp_.layer.eu_ld * (size_t) FE;
        mtp_.layer.ed_stride = mtp_.layer.ed_ld * (size_t) E;
        mtp_.layer.gate_file_off = gf.tensor_file_offset(tg);
        mtp_.layer.up_file_off = gf.tensor_file_offset(tu);
        mtp_.layer.down_file_off = gf.tensor_file_offset(td);
        mtp_.layer.gate_file = gf.file_of(tg);
        mtp_.layer.up_file = gf.file_of(tu);
        mtp_.layer.down_file = gf.file_of(td);
        mtp_.layer.tr[0] = 0;
        mtp_.layer.tr[1] = align256(mtp_.layer.eg_stride);
        mtp_.layer.tr[2] = align256(mtp_.layer.tr[1] + mtp_.layer.eu_stride);
        mtp_.layer.blob = align256(mtp_.layer.tr[2] + mtp_.layer.ed_stride);
        max_blob_ = std::max(max_blob_, mtp_.layer.blob);
        mtp_.emb_buf = (float*) dev_alloc(glast, (size_t) E * 4);
        mtp_.eh_q = (int8_t*) dev_alloc(glast, (size_t) 2 * E);
        mtp_.eh_d = (float*) dev_alloc(glast, (size_t) (2 * E / 32) * 4);
        mtp_.hring = (float*) dev_alloc(glast, (size_t) MtpWeights::kRing * (size_t) E * sizeof(float));
        CK(cudaMemsetAsync(mtp_.hring, 0, (size_t) MtpWeights::kRing * (size_t) E * sizeof(float), glast.s));
        mtp_.h_tmp = (float*) dev_alloc(glast, (size_t) E * sizeof(float));
        size_t fb = 0, tb = 0;
        CK(cudaMemGetInfo(&fb, &tb));
        const size_t eff_fb = fb > reserve_b ? fb - reserve_b : 0u;
        const double mtp_headroom = 0.75 * (double) (1ULL << 30);
        const double mtp_budget = std::max(0.0, (double) eff_fb - mtp_headroom);
        int mnr = std::min(c.n_expert, (int) std::floor(mtp_budget / (double) mtp_.layer.blob));
        mtp_.layer.n_res = std::max(0, mnr);
        if (mtp_.layer.n_res > 0) {
            mtp_.layer.fl.vram = (uint8_t*) dev_alloc(glast, (size_t) mtp_.layer.n_res * mtp_.layer.blob);
            mtp_.layer.vram_pool_off = 0;
        }
        CK(cudaStreamSynchronize(glast.s));
        mtp_loaded_ = true;
    }

    // ---- Derive prefill macro-block size from GPUs' VRAM pool, host RAM, and GGUF expert count ----
    {
        auto macro_bytes_est = [&](int B) -> size_t {
            const size_t x_cnt = (ngpu >= 2 && !o_.ep_prefill) ? 4u : 3u;
            return align256((size_t) B * E * sizeof(float)) * x_cnt +
                   align256((size_t) B * c.n_used * sizeof(int)) * 3 +
                   align256((size_t) B * c.n_used * sizeof(float)) + 2048;
        };
        size_t min_pool = gpus_[0].vram_pool_bytes;
        for (int gi = 1; gi < ngpu; ++gi) min_pool = std::min(min_pool, gpus_[(size_t) gi].vram_pool_bytes);
        const int exp_sat = std::max(P, 256 * c.n_expert / std::max(1, c.n_used));
        const int auto_cap = std::min(o_.ctx, exp_sat);
        int derived = P;
        if (o_.repeatable) {
            derived = P;
        } else if (o_.prefill_block > 0) {
            derived = std::max(P, std::min(o_.prefill_block, std::max(P, o_.ctx)));
        } else if (min_pool > 0) {
            const uint64_t avail_r = hy4::ram_available();
            const size_t host_cap = avail_r > 0 ? (size_t) std::min<uint64_t>(avail_r / 256, 64ULL << 20) : (8ULL << 20);
            int b = P;
            while (b <= auto_cap / 2) {
                const int next_b = b * 2;
                const size_t need_v = macro_bytes_est(next_b);
                const size_t need_h = (size_t) next_b * (size_t) c.n_used * (sizeof(int) + sizeof(float)) * (size_t) ngpu;
                if (need_v * 4 <= min_pool && need_h <= host_cap) {
                    b = next_b;
                } else {
                    break;
                }
            }
            derived = b;
        }
        prefill_block_ = derived;
        for (int gi = 0; gi < ngpu; ++gi) {
            Gpu& gp = gpus_[(size_t) gi];
            set_dev(gp.dev);
            gp.h_pf_cap = std::max(P, prefill_block_);
            gp.h_pf_ids = (int*) host_alloc(gp, (size_t) gp.h_pf_cap * c.n_used * sizeof(int));
            gp.h_pf_w = (float*) host_alloc(gp, (size_t) gp.h_pf_cap * c.n_used * sizeof(float));
        }
    }

    // ---- Load optional binary expert placement profile ("LNGP", refuses mismatched expert count) ----
    sys::ExpertProfile profile;
    bool has_profile = false;
    std::string prof_path = o_.profile_path;
    if (!o_.no_profile && prof_path.empty()) {
        if (const char* ep = std::getenv("LING_PROFILE")) {
            prof_path = ep;
        } else if (std::filesystem::exists("data/expert-profile-ling32.bin")) {
            prof_path = "data/expert-profile-ling32.bin";
        }
    }
    if (!o_.no_profile && !prof_path.empty()) {
        std::string perr;
        if (!profile.load(prof_path, c.n_layers - c.n_dense_lead, c.n_expert, perr)) {
            throw std::runtime_error("maya-ling: " + perr);
        }
        has_profile = true;
        std::fprintf(stderr, "ling profile: loaded %s (%u moe layers x %u experts)\n",
                     prof_path.c_str(), profile.n_moe_layers, profile.n_expert);
    }

    // ---- Plan RAM vs NVMe SSD expert tiers (MADV_NOHUGEPAGE, 8 GiB RAM spare, O_DIRECT fallback) ----
    struct PackTask {
        Layer* l;
        const maya::TensorInfo *tg, *tu, *td;
        int il;
        int moe_idx;
        uint64_t min_off;
        int file_idx;   // the shard of tg (split GGUF)
    };
    std::vector<PackTask> tasks;
    for (int il = c.n_dense_lead; il < c.n_layers; ++il) {
        const std::string b = "blk." + std::to_string(il) + ".";
        const maya::TensorInfo* tg = want.at(b + "ffn_gate_exps.weight");
        const maya::TensorInfo* tu = want.at(b + "ffn_up_exps.weight");
        const maya::TensorInfo* td = want.at(b + "ffn_down_exps.weight");
        tasks.push_back({&layers_[(size_t) il], tg, tu, td, il, il - c.n_dense_lead, gf.tensor_file_offset(*tg),
                         gf.file_of(*tg)});
    }
    if (mtp_loaded_) {
        const maya::TensorInfo* tg = mtp_want.at(mtp_prefix + "ffn_gate_exps.weight");
        const maya::TensorInfo* tu = mtp_want.at(mtp_prefix + "ffn_up_exps.weight");
        const maya::TensorInfo* td = mtp_want.at(mtp_prefix + "ffn_down_exps.weight");
        tasks.push_back({&mtp_.layer, tg, tu, td, c.n_layers, c.n_layers - c.n_dense_lead, gf.tensor_file_offset(*tg),
                         gf.file_of(*tg)});
    }

    // Sort tasks in ascending file-offset order for sequential file-order warm-up!
    std::vector<size_t> file_order(tasks.size());
    std::iota(file_order.begin(), file_order.end(), 0);
    std::sort(file_order.begin(), file_order.end(),
              [&](size_t a, size_t b) {
                  return std::make_pair(tasks[a].file_idx, tasks[a].min_off) <
                         std::make_pair(tasks[b].file_idx, tasks[b].min_off);   // (per shard, then offset)
              });

    const uint64_t avail_ram = hy4::ram_available();
    const uint64_t total_ram = hy4::ram_total();
    const uint64_t ram_spare = 8ULL << 30;
    uint64_t ram_budget_bytes = (avail_ram > ram_spare) ? (avail_ram - ram_spare) : 0ULL;
    if (o_.ram_cap_gib >= 0.0) {
        const uint64_t cap_b = (uint64_t) (o_.ram_cap_gib * (double) (1ULL << 30));
        ram_budget_bytes = (avail_ram > 0) ? std::min(ram_budget_bytes, cap_b) : cap_b;
    }

    size_t planned_ram_bytes = 0, planned_ssd_bytes = 0;
    size_t ram_slots_cnt = 0, ssd_slots_cnt = 0;
    bool any_ssd = false;

    for (size_t idx : file_order) {
        Layer& l = *tasks[idx].l;
        if (l.n_res == c.n_expert) continue;
        const size_t bytes = l.blob * (size_t) c.n_expert;
        const size_t non_vram = (size_t) (c.n_expert - l.n_res);
        if (planned_ram_bytes + bytes <= ram_budget_bytes) {
            l.host_bytes = bytes;
            l.host_blobs = (uint8_t*) sys::alloc_nohuge(bytes);
            if (l.host_blobs != nullptr) {
                if (cpu_topo_.numa_nodes > 1 && gpus_[(size_t) l.dev_idx].numa_node >= 0) {
                    (void) hy4::Lane::move_to_node(l.host_blobs, bytes, gpus_[(size_t) l.dev_idx].numa_node);
                }
                l.host_nohuge = true;
                l.ssd_backed = false;
                planned_ram_bytes += bytes;
                ram_slots_cnt += non_vram;
                continue;
            }
        }
        l.host_blobs = nullptr;
        l.host_bytes = 0;
        l.ssd_backed = true;
        any_ssd = true;
        planned_ssd_bytes += non_vram * l.blob;
        ssd_slots_cnt += non_vram;
    }

    if (any_ssd) {
        std::string derr;
        ssd_readers_.clear();
        for (const std::string& p : gf.paths()) {
            ssd_readers_.push_back(std::make_unique<sys::DirectGgufReader>());
            if (!ssd_readers_.back()->open(p, o_.direct_io, derr)) throw std::runtime_error("maya-ling: " + derr);
        }
        if (!tasks.empty()) {
            auto& r0 = *ssd_readers_[(size_t) tasks[file_order[0]].file_idx];
            const size_t cal_cb = r0.calibrate_chunk(tasks[file_order[0]].min_off, std::min<size_t>(max_blob_, 2u << 20));
            for (auto& r : ssd_readers_) r->set_chunk_bytes(cal_cb);
            const char* io_tag = direct_io_active() ? "on" : (o_.direct_io ? "fallback-buffered" : "off (buffered)");
            std::fprintf(stderr, "ling ssd: O_DIRECT=%s chunk=%zu KiB ssd_slots=%zu (%.2f GiB) ram_slots=%zu (%.2f GiB)\n",
                         io_tag, cal_cb >> 10, ssd_slots_cnt,
                         (double) planned_ssd_bytes / (double) (1ULL << 30), ram_slots_cnt,
                         (double) planned_ram_bytes / (double) (1ULL << 30));
        }
    }

    // Pack RAM-tier layers in file-offset order
    {
        const int nt = std::min<int>((int) tasks.size(), std::max(1, std::min(16, o_.cpu_threads)));
        std::atomic<size_t> next{0};
        auto worker = [&] {
            while (true) {
                const size_t pos_i = next.fetch_add(1);
                if (pos_i >= file_order.size()) break;
                const PackTask& tk = tasks[file_order[pos_i]];
                Layer& l = *tk.l;
                if (l.host_blobs == nullptr) continue;
                const uint8_t* sg = gf.tensor_data(*tk.tg);
                const uint8_t* su = gf.tensor_data(*tk.tu);
                const uint8_t* sd = gf.tensor_data(*tk.td);
                for (int e = 0; e < c.n_expert; ++e) {
                    uint8_t* dst = l.host_blobs + (size_t) e * l.blob;
                    std::memcpy(dst + l.tr[0], sg + (size_t) e * l.eg_stride, l.eg_stride);
                    std::memcpy(dst + l.tr[1], su + (size_t) e * l.eu_stride, l.eu_stride);
                    std::memcpy(dst + l.tr[2], sd + (size_t) e * l.ed_stride, l.ed_stride);
                }
                // (the copied source pages out of RAM: Linux MADV_DONTNEED, Windows VirtualUnlock - ling_sys.hpp)
                (void) sys::drop_copied_pages(sg, (size_t) c.n_expert * l.eg_stride);
                (void) sys::drop_copied_pages(su, (size_t) c.n_expert * l.eu_stride);
                (void) sys::drop_copied_pages(sd, (size_t) c.n_expert * l.ed_stride);
            }
        };
        std::vector<std::thread> threads;
        for (int i = 0; i < nt; ++i) threads.emplace_back(worker);
        for (auto& th : threads) th.join();
    }

    // Pin packed host buffers when GPU/mix/auto wants UVA DMA (and within safe host_pin_cap)
    const double est_vram_frac =
        total_experts > 0 ? (double) total_vram_experts / (double) total_experts : 1.0;
    const uint64_t ram_headroom = std::max<uint64_t>(8ULL << 30, total_ram / 10);
    const uint64_t host_pin_cap =
        (avail_ram > ram_headroom)
            ? std::min<uint64_t>(avail_ram - ram_headroom, (uint64_t) (0.75 * (double) avail_ram))
            : 0ULL;
    const bool mode_uses_pin = !any_ssd && (o_.offload == OffloadMode::kGpu || o_.offload == OffloadMode::kMix ||
                                            o_.offload == OffloadMode::kAuto);
    bool want_pin = false;
    double pin_budget = 0.0;
    if (o_.pin_ram_gib > 0.0) {
        want_pin = true;
        pin_budget = o_.pin_ram_gib * (double) (1ULL << 30);
        if (avail_ram > 0) pin_budget = std::min(pin_budget, 0.75 * (double) avail_ram);
    } else if (o_.pin_ram_gib < 0.0 && mode_uses_pin) {
        want_pin = (host_pin_cap > 0);
        pin_budget = (double) host_pin_cap;
    }
    size_t pin_bytes = 0, page_bytes = 0;
    for (size_t idx : file_order) {
        Layer& l = *tasks[idx].l;
        if (l.host_blobs == nullptr) {
            if (l.n_res < c.n_expert) any_host_unpinned_ = true;
            continue;
        }
        // (a layer whose gate and up are of different types is pinned whatever the mode and the budget: its experts
        //  outside VRAM run on the GPU from pinned RAM, never on the CPU lane - a few layers of a file)
        if ((want_pin && (double) l.host_bytes <= pin_budget) || (l.mixed_gu() && l.n_res < c.n_expert)) {
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

    // Populate initial VRAM slots by profile rank (in ascending file-offset order!), async on gp.s
    for (size_t idx : file_order) {
        const PackTask& tk = tasks[idx];
        Layer& l = *tk.l;
        Gpu& gp = gpus_[(size_t) l.dev_idx];
        set_dev(gp.dev);
        l.pf_counts.assign((size_t) c.n_expert, 0);
        l.profile_counts.assign((size_t) c.n_expert, 0);
        l.la_next.assign((size_t) c.n_expert, -1);

        std::vector<int> ranked;
        const uint32_t* pc = has_profile ? profile.layer_counts(tk.moe_idx) : nullptr;
        sys::plan_expert_slots(c.n_expert, l.n_res, pc, l.h_slot, l.h_owner, l.h_score, ranked);

        l.fl.n_res = l.n_res;
        l.fl.n_exp = c.n_expert;
        l.fl.blob = l.blob;
        l.fl.slot = (int*) dev_alloc(gp, (size_t) c.n_expert * sizeof(int));
        l.fl.owner = (int*) dev_alloc(gp, (size_t) std::max(1, l.n_res) * sizeof(int));
        l.fl.score = (float*) dev_alloc(gp, (size_t) c.n_expert * sizeof(float));
        CK(cudaMemcpyAsync(l.fl.score, l.h_score.data(), (size_t) c.n_expert * sizeof(float), cudaMemcpyHostToDevice, gp.s));
        CK(cudaEventCreateWithFlags(&l.ev_promo, cudaEventDisableTiming));

        if (l.n_res > 0) {
            if (tk.il < c.n_layers) l.fl.vram = gp.vram_pool + l.vram_pool_off;
            std::vector<std::pair<int, int>> copies;
            copies.reserve((size_t) l.n_res);
            for (int sl = 0; sl < l.n_res; ++sl) {
                const int e = ranked[(size_t) sl];
                l.h_slot[(size_t) e] = sl;
                l.h_owner[(size_t) sl] = e;
                copies.push_back({e, sl});
            }
            std::sort(copies.begin(), copies.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            if (l.host_blobs != nullptr) {
                for (const auto& cp : copies) {
                    CK(cudaMemcpyAsync(l.fl.vram + (size_t) cp.second * l.blob,
                                       l.host_blobs + (size_t) cp.first * l.blob, l.blob, cudaMemcpyHostToDevice, gp.s));
                }
            } else {
                const uint8_t* sg = gf.tensor_data(*tk.tg);
                const uint8_t* su = gf.tensor_data(*tk.tu);
                const uint8_t* sd = gf.tensor_data(*tk.td);
                std::vector<uint8_t> tmp(l.blob);
                for (const auto& cp : copies) {
                    const int e = cp.first;
                    std::memcpy(tmp.data() + l.tr[0], sg + (size_t) e * l.eg_stride, l.eg_stride);
                    std::memcpy(tmp.data() + l.tr[1], su + (size_t) e * l.eu_stride, l.eu_stride);
                    std::memcpy(tmp.data() + l.tr[2], sd + (size_t) e * l.ed_stride, l.ed_stride);
                    CK(cudaMemcpyAsync(l.fl.vram + (size_t) cp.second * l.blob, tmp.data(), l.blob,
                                       cudaMemcpyHostToDevice, gp.s));
                    CK(cudaStreamSynchronize(gp.s));
                }
            }
        }
        CK(cudaMemcpyAsync(l.fl.slot, l.h_slot.data(), (size_t) c.n_expert * sizeof(int), cudaMemcpyHostToDevice, gp.s));
        CK(cudaMemcpyAsync(l.fl.owner, l.h_owner.data(), (size_t) std::max(1, l.n_res) * sizeof(int),
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
        CK(cudaStreamSynchronize(gp.s));
        l.fl.hptr = d_hptr;

        if (l.n_res < c.n_expert && l.mixed_gu()) {
            // (the CPU lane has one type for the gate and the up: this layer's other experts run on the GPU, fetched
            //  from pinned RAM - never computed with the gate's type for the up)
            if (l.host_blobs == nullptr || !l.host_pinned)
                throw std::runtime_error("layer " + std::to_string(tk.il) +
                                         ": its gate and up experts are of different types, and the experts that are "
                                         "not in VRAM must be in pinned RAM for this (more VRAM or --pin-ram-gib helps)");
        } else if (l.n_res < c.n_expert) {
            std::string err;
            if (!l.cex.init(l.etg, l.etd, E, FE, err))
                throw std::runtime_error("cpu lane init failed on layer " + std::to_string(tk.il) + ": " + err);
            if (l.host_blobs != nullptr) {
                l.cex.gate = l.host_blobs + l.tr[0];
                l.cex.up = l.host_blobs + l.tr[1];
                l.cex.down = l.host_blobs + l.tr[2];
                l.cex.step = l.blob;
            }
            l.cex.limit = c.clamp_exp[tk.il] > 1e-6f ? -c.clamp_exp[tk.il] : 0.0f;
        }
    }
    for (auto& gp : gpus_) {
        set_dev(gp.dev);
        CK(cudaStreamSynchronize(gp.s));
    }
    const std::vector<std::string> gf_paths = gf.paths();
    std::vector<sys::FingerprintTensor> fp_tensors;   // (the restore points' file fingerprint reads the tensor table)
    fp_tensors.reserve(gf.tensors().size());
    for (const auto& t : gf.tensors())
        fp_tensors.push_back({t.name, t.type, t.offset, t.elements(), (uint32_t) gf.file_of(t)});
    gf_u.reset();  // unmap GGUF now that VRAM and RAM tensors are copied (RAM counted once)

    // Build keyed calibration ID: host|model|quant|ctx|gpus|driver|kernel_hash
    {
        std::ostringstream gstr;
        for (int gi = 0; gi < ngpu; ++gi) {
            if (gi > 0) gstr << ",";
            cudaDeviceProp dp{};
            cudaGetDeviceProperties(&dp, gpus_[(size_t) gi].dev);
            gstr << gpus_[(size_t) gi].dev << ":" << dp.major << dp.minor;
        }
        int drv = 0;
        cudaDriverGetVersion(&drv);
        // (a file whose layers' up types differ from their gate's: those layers and types in the keys, as a short
        //  hash - the keys of every other file stay as they were, their restore points and measurements kept)
        std::string mix_tag;
        if (!quant_mix.empty()) {
            uint64_t hsh = 1469598103934665603ull;
            for (unsigned char ch : quant_mix) hsh = (hsh ^ ch) * 1099511628211ull;
            char hx[24];
            std::snprintf(hx, sizeof hx, "%016llx", (unsigned long long) hsh);
            mix_tag = std::string("mix") + hx;
        }
        const std::string qtag =
            std::to_string(quant_tg) + "_" + std::to_string(quant_td) + (mix_tag.empty() ? "" : "_" + mix_tag);
        const std::string khash = "ling-v2-" + std::to_string(o_.kv_q8 ? 1 : 0);
        const std::string canon_name =
            std::filesystem::path(gf_paths.empty() ? path : gf_paths[0]).filename().string();
        cal_key_ = sys::cal_key(sys::host_name(), canon_name, qtag, o_.ctx,
                                gstr.str(), std::to_string(drv), khash);
        {
            uint32_t n_kda = 0, n_mla = 0;
            for (int il = 0; il < c.n_layers; ++il) {
                if (c.kda[il]) ++n_kda;
                else ++n_mla;
            }
            restore_expect_.key.model_id = canon_name;
            restore_expect_.key.quant_tag = "etg=" + std::to_string(quant_tg) + ",etd=" + std::to_string(quant_td) +
                                            ",emb=" + std::to_string(emb_.type) + ",out=" + std::to_string(out_w_.type) +
                                            (mix_tag.empty() ? "" : ",etu=" + mix_tag);
            restore_expect_.key.engine_version = sys::kRestoreEngineVersion;
            // (the files' sampled bytes, then the tensor table: names, types, offsets, sizes)
            restore_expect_.key.gguf_fingerprint =
                sys::mix_tensor_table(sys::compute_gguf_set_fingerprint(gf_paths), std::move(fp_tensors));
            restore_expect_.key.ctx = (uint32_t) o_.ctx;
            restore_expect_.key.kv_q8 = o_.kv_q8 ? 1u : 0u;
            restore_expect_.n_kda_layers = n_kda;
            restore_expect_.n_mla_layers = n_mla;
            restore_expect_.vocab = (uint32_t) c_.vocab;
            restore_expect_.kda_state_bytes = (uint64_t) c.n_head * (uint64_t) c.kda_head * (uint64_t) c.kda_head * sizeof(float);
            restore_expect_.kda_conv_bytes = (uint64_t) 3 * (uint64_t) c.d_inner() * (uint64_t) (c.d_conv - 1) * sizeof(float);
            restore_expect_.mla_lat_row_bytes =
                o_.kv_q8 ? ((uint64_t) c.kv_lora + (uint64_t) (c.kv_lora / 32) * 2ULL)
                         : ((uint64_t) c.kv_lora * sizeof(uint16_t));
            restore_expect_.mla_kpe_row_bytes = (uint64_t) c.n_rot * sizeof(uint16_t);
        }
        cal_file_ = o_.cal_file.empty() ? (std::getenv("LING_CAL_FILE") ? std::string(std::getenv("LING_CAL_FILE"))
                                                                        : std::string("data/ling-calibration.txt"))
                                        : o_.cal_file;
    }

    vram_res_frac_ = est_vram_frac;
    total_experts_ = total_experts;
    total_vram_experts_ = total_vram_experts;
    total_ram_experts_ = ram_slots_cnt;
    total_ssd_experts_ = ssd_slots_cnt;
    total_ram_gb_ = (double) planned_ram_bytes / (double) (1ULL << 30);
    total_ssd_gb_ = (double) planned_ssd_bytes / (double) (1ULL << 30);

    size_t vram_sum = 0;
    for (int gi = 0; gi < ngpu; ++gi) {
        set_dev(gpus_[(size_t) gi].dev);
        size_t fb = 0, tb = 0;
        if (cudaMemGetInfo(&fb, &tb) == cudaSuccess) vram_sum += (tb - fb);
    }
    total_vram_gb_ = (double) vram_sum / (double) (1ULL << 30);

    // Initialize CPU lane (threads = affinity physical cores - 2)
    if (total_vram_experts < total_experts || any_ssd || o_.offload == OffloadMode::kCpu || o_.offload == OffloadMode::kMix) {
        const int workers = std::max(0, o_.cpu_threads - 1);
        lane_ = std::make_unique<hy4::Lane>(workers);
    }

    // Choose / calibrate GPU-fetch vs CPU-lane split (saved per PC in cal_file_)
    if (o_.offload == OffloadMode::kAuto) {
        if (total_vram_experts == total_experts) {
            active_offload_ = OffloadMode::kGpu;
            pcie_share_ = 1.0;
        } else if (any_host_unpinned_ || any_ssd) {
            active_offload_ = OffloadMode::kCpu;
            pcie_share_ = 0.0;
        } else {
            calibrate_split_and_threads();
        }
    } else {
        active_offload_ = o_.offload;
        if (active_offload_ == OffloadMode::kGpu && (any_host_unpinned_ || any_ssd)) active_offload_ = OffloadMode::kCpu;
        pcie_share_ = (active_offload_ == OffloadMode::kGpu) ? 1.0 : (active_offload_ == OffloadMode::kMix ? 0.35 : 0.0);
    }
    if (o_.lookahead == 1) lookahead_on_ = true;
    else if (o_.lookahead == 0) lookahead_on_ = false;

    if (!o_.mmproj_path.empty()) {
        vision_ = std::make_unique<VisionTower>();
        std::string verr;
        if (!vision_->load(o_.mmproj_path, gpus_[0].dev, verr)) {
            throw std::runtime_error("maya-ling: vision load failed: " + verr);
        }
    }

    // Launch background cuBLAS + kernel warm-up (first request waits cleanly for it)
    start_warmup_async();

    const char* mode_name = active_offload_ == OffloadMode::kGpu ? "gpu"
                            : active_offload_ == OffloadMode::kMix ? "mix"
                                                                   : "cpu";
    const uint64_t rss_now = sys::process_rss_bytes();
    std::fprintf(stderr, "ling ram: planned=%.2f GiB (pinned=%.2f pageable=%.2f) rss=%.2f GiB ssd=%.2f GiB\n",
                 total_ram_gb_, (double) pin_bytes / (double) (1ULL << 30), (double) page_bytes / (double) (1ULL << 30),
                 (double) rss_now / (double) (1ULL << 30), total_ssd_gb_);

    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "maya-ling: %dL (%d KDA + %d MLA), E=%d H=%d, %d GPU, offload=%s graph=%s pf=%d/%d, "
                  "VRAM=%zu/%zu (%.1f%%, %.2f GiB), RAM=%zu (%.2f GiB), SSD=%zu (%.2f GiB), kv=%s, repeatable=%d",
                  c.n_layers, c.n_layers - 7, 7, E, H, ngpu, mode_name,
                  (o_.use_graph && active_offload_ == OffloadMode::kGpu && o_.seam_dir.empty()) ? "on" : "off", P,
                  prefill_block_, total_vram_experts, total_experts,
                  total_experts > 0 ? 100.0 * (double) total_vram_experts / (double) total_experts : 0.0,
                  total_vram_gb_, ram_slots_cnt, total_ram_gb_, ssd_slots_cnt, total_ssd_gb_,
                  o_.kv_q8 ? "q8" : "f16", o_.repeatable ? 1 : 0);
    info_ = buf;
}

Model::~Model() {
    wait_warmup();
    if (!o_.profile_out.empty()) {
        std::string perr;
        (void) save_profile_now(o_.profile_out, perr);
    }
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
        }
        if (l.host_nohuge) {
            sys::free_nohuge(l.host_blobs, l.host_bytes);
        } else if (l.host_pinned && !l.host_registered) {
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

void Model::wait_warmup() {
    if (warmup_thread_.joinable()) warmup_thread_.join();
}

void Model::start_warmup_async() {
    warmup_done_.store(false);
    warmup_thread_ = std::thread([this] {
        try {
            for (auto& gp : gpus_) {
                set_dev(gp.dev);
                const float alpha = 1.0f, beta = 0.0f;
                if (gp.pf_x16 && gp.pf_w16 && gp.pf_out) {
                    (void) cublasGemmEx(gp.blas, CUBLAS_OP_T, CUBLAS_OP_N, 64, 16, 64, &alpha, gp.pf_w16, CUDA_R_16F,
                                        64, gp.pf_x16, CUDA_R_16F, 64, &beta, gp.pf_out, CUDA_R_32F, 64,
                                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                }
                CK(cudaStreamSynchronize(gp.s));
            }
        } catch (...) {
        }
        warmup_done_.store(true);
    });
}

void Model::peer_setup_verified() {
    const int ngpu = (int) gpus_.size();
    constexpr size_t kProbeBytes = 4ULL << 20;
    constexpr size_t kWords = kProbeBytes / 4;
    std::vector<uint32_t> pat(kWords), back(kWords);
    std::vector<void*> pbuf((size_t) ngpu, nullptr);
    for (int gi = 0; gi < ngpu; ++gi) {
        set_dev(gpus_[(size_t) gi].dev);
        CK(cudaMalloc(&pbuf[(size_t) gi], kProbeBytes));
    }
    std::string summary;
    for (int gi = 0; gi < ngpu; ++gi) {
        for (int gj = 0; gj < ngpu; ++gj) {
            if (gi == gj) continue;
            const bool adjacent = (std::abs(gi - gj) == 1) || (gi == 0 && gj == ngpu - 1) || (gi == ngpu - 1 && gj == 0);
            if (!adjacent) continue;
            Gpu& ga = gpus_[(size_t) gi];
            Gpu& gb = gpus_[(size_t) gj];
            int can = 0;
            if (cudaDeviceCanAccessPeer(&can, gb.dev, ga.dev) != cudaSuccess) can = 0;
            (void) cudaGetLastError();
            bool enabled = false;
            if (can) {
                set_dev(gb.dev);
                const cudaError_t pe = cudaDeviceEnablePeerAccess(ga.dev, 0);
                (void) cudaGetLastError();
                enabled = (pe == cudaSuccess || pe == cudaErrorPeerAccessAlreadyEnabled);
            }
            for (size_t i = 0; i < kWords; ++i)
                pat[i] = (uint32_t) (i * 2654435761u) ^ (0x9e3779b9u + (uint32_t) (gi * 64 + gj));
            set_dev(ga.dev);
            CK(cudaMemcpyAsync(pbuf[(size_t) gi], pat.data(), kProbeBytes, cudaMemcpyHostToDevice, ga.s));
            CK(cudaStreamSynchronize(ga.s));
            set_dev(gb.dev);
            CK(cudaMemsetAsync(pbuf[(size_t) gj], 0, kProbeBytes, gb.s));
            CK(cudaStreamSynchronize(gb.s));
            bool ok = false;
            if (enabled) {
                const cudaError_t ce = cudaMemcpyPeerAsync(pbuf[(size_t) gj], gb.dev, pbuf[(size_t) gi], ga.dev,
                                                           kProbeBytes, gb.s);
                cudaError_t se = cudaSuccess;
                if (ce == cudaSuccess) {
                    const auto t0 = std::chrono::steady_clock::now();
                    while ((se = cudaStreamQuery(gb.s)) == cudaErrorNotReady) {
                        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > 20.0) break;
                        std::this_thread::yield();
                    }
                }
                (void) cudaGetLastError();
                if (ce == cudaSuccess && se == cudaSuccess) {
                    CK(cudaMemcpyAsync(back.data(), pbuf[(size_t) gj], kProbeBytes, cudaMemcpyDeviceToHost, gb.s));
                    CK(cudaStreamSynchronize(gb.s));
                    size_t wrong = 0;
                    for (size_t i = 0; i < kWords; ++i) wrong += (back[i] != pat[i]);
                    ok = (wrong == 0);
                }
                if (!ok) {
                    set_dev(gb.dev);
                    (void) cudaDeviceDisablePeerAccess(ga.dev);
                    (void) cudaGetLastError();
                }
            }
            p2p_mat_[(size_t) gi][(size_t) gj] = ok;
            if (!summary.empty()) summary += ", ";
            summary += "CUDA" + std::to_string(ga.dev) + "->CUDA" + std::to_string(gb.dev) +
                       (ok ? " peer(verified)" : " host-stage");
        }
    }
    for (int gi = 0; gi < ngpu; ++gi) {
        set_dev(gpus_[(size_t) gi].dev);
        cudaFree(pbuf[(size_t) gi]);
    }
    std::fprintf(stderr, "ling peer: %s\n", summary.c_str());
}

void Model::calibrate_split_and_threads() {
    if (!o_.recalibrate && !cal_file_.empty()) {
        auto m = sys::cal_load(cal_file_, cal_key_);
        if (m.count("mode") && m.count("gpu_ms") && m.count("cpu_ms")) {
            const std::string& md = m["mode"];
            active_offload_ = (md == "gpu") ? OffloadMode::kGpu : (md == "mix" ? OffloadMode::kMix : OffloadMode::kCpu);
            if (m.count("mix_k")) o_.mix_k_gpu = std::max(1, std::atoi(m["mix_k"].c_str()));
            if (m.count("cpu_ms")) base_cpu_ms_per_exp_ = std::atof(m["cpu_ms"].c_str());
            if (m.count("lookahead") && o_.lookahead < 0) lookahead_on_ = (m["lookahead"] == "1");
            pcie_share_ = (active_offload_ == OffloadMode::kGpu) ? 1.0 : (active_offload_ == OffloadMode::kMix ? 0.35 : 0.0);
            std::fprintf(stderr, "ling split-mode: cached gpu_fetch=%s ms/exp cpu_lane=%s ms/exp -> mode=%s (mix_k=%d, key=%s)\n",
                         m["gpu_ms"].c_str(), m["cpu_ms"].c_str(), md.c_str(), o_.mix_k_gpu, cal_key_.c_str());
            return;
        }
    }
    Layer* probe = nullptr;
    for (int il = c_.n_dense_lead; il < c_.n_layers; ++il) {
        if (layers_[(size_t) il].n_res < c_.n_expert && layers_[(size_t) il].host_blobs != nullptr &&
            layers_[(size_t) il].host_pinned && !layers_[(size_t) il].mixed_gu()) {   // (a layer with a CPU lane)
            probe = &layers_[(size_t) il];
            break;
        }
    }
    if (!probe || !lane_) {
        active_offload_ = (vram_res_frac_ >= 0.65 && !any_host_unpinned_) ? OffloadMode::kGpu : OffloadMode::kCpu;
        pcie_share_ = (active_offload_ == OffloadMode::kGpu) ? 1.0 : 0.0;
        return;
    }
    Gpu& gp = gpus_[(size_t) probe->dev_idx];
    set_dev(gp.dev);
    constexpr int kProbeK = 4;
    for (int r = 0; r < 3; ++r)
        for (int k = 0; k < kProbeK; ++k)
            CK(cudaMemcpyAsync(gp.fxs.stage + (size_t) k * probe->blob, probe->host_blobs + (size_t) k * probe->blob,
                               probe->blob, cudaMemcpyHostToDevice, gp.s_copy));
    CK(cudaStreamSynchronize(gp.s_copy));
    const auto tg0 = std::chrono::steady_clock::now();
    for (int r = 0; r < 16; ++r)
        for (int k = 0; k < kProbeK; ++k)
            CK(cudaMemcpyAsync(gp.fxs.stage + (size_t) k * probe->blob, probe->host_blobs + (size_t) k * probe->blob,
                               probe->blob, cudaMemcpyHostToDevice, gp.s_copy));
    CK(cudaStreamSynchronize(gp.s_copy));
    const double gpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tg0).count() /
                          (16.0 * (double) kProbeK);

    std::vector<float> hx((size_t) c_.n_embd, 0.05f), ho((size_t) c_.n_embd, 0.0f);
    std::vector<hy4::CpuEntry> ents;
    for (int k = 0; k < kProbeK; ++k) ents.push_back({0, k, 0.25f});
    for (int r = 0; r < 3; ++r) hy4::cpu_experts(*lane_, probe->cex, ents, hx.data(), 1, ho.data());
    const auto tc0 = std::chrono::steady_clock::now();
    for (int r = 0; r < 16; ++r) hy4::cpu_experts(*lane_, probe->cex, ents, hx.data(), 1, ho.data());
    const double cpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc0).count() /
                          (16.0 * (double) kProbeK);
    base_cpu_ms_per_exp_ = cpu_ms;
    cpu_ema_ms_per_exp_ = cpu_ms;

    const double est_miss = std::max(0.05, (double) c_.n_used * (1.0 - std::min(0.99, vram_res_frac_ * 1.35)));
    const double cost_gpu = est_miss * gpu_ms;
    const double cost_cpu = est_miss * cpu_ms + 0.12;
    const char* md = "cpu";
    if (cost_gpu <= cost_cpu) {
        active_offload_ = OffloadMode::kGpu;
        md = "gpu";
        pcie_share_ = 1.0;
    } else if (gpu_ms < cpu_ms * 1.5 && est_miss >= 2.0) {
        active_offload_ = OffloadMode::kMix;
        o_.mix_k_gpu = 1;
        md = "mix";
        pcie_share_ = 0.25;
    } else {
        active_offload_ = OffloadMode::kCpu;
        md = "cpu";
        pcie_share_ = 0.0;
    }
    char gbuf[32], cbuf[32];
    std::snprintf(gbuf, sizeof gbuf, "%.4f", gpu_ms);
    std::snprintf(cbuf, sizeof cbuf, "%.4f", cpu_ms);
    std::map<std::string, std::string> rec = {
        {"mode", md}, {"gpu_ms", gbuf}, {"cpu_ms", cbuf},
        {"mix_k", std::to_string(o_.mix_k_gpu)}, {"threads", std::to_string(o_.cpu_threads)},
        {"lookahead", lookahead_on_ ? "1" : "0"},
    };
    if (!cal_file_.empty()) sys::cal_save(cal_file_, cal_key_, rec);
    std::fprintf(stderr, "ling split-mode: measured gpu_fetch=%.4f ms/exp cpu_lane=%.4f ms/exp (est_miss=%.2f) -> mode=%s (mix_k=%d)\n",
                 gpu_ms, cpu_ms, est_miss, md, o_.mix_k_gpu);
}

void Model::check_route_and_fault(const char* stage) {
    if (int ke = hy4::k::errors(); ke > 0) {
        throw GpuFaultError(std::string("GPU kernel launch fault in ") + stage + " (first fault: " +
                            hy4::k::first_fault() + ")");
    }
    for (auto& gp : gpus_) {
        set_dev(gp.dev);
        const cudaError_t le = cudaGetLastError();
        if (le != cudaSuccess) {
            throw GpuFaultError(std::string("CUDA fault in ") + stage + " (last kernel: " +
                                (last_kernel_.empty() ? stage : last_kernel_) + "): " + cudaGetErrorString(le));
        }
        CK(cudaMemcpyAsync(gp.h_route_err, gp.d_route_err, sizeof(int), cudaMemcpyDeviceToHost, gp.s));
        CK(cudaStreamSynchronize(gp.s));
        if (*gp.h_route_err != 0) {
            *gp.h_route_err = 0;
            CK(cudaMemsetAsync(gp.d_route_err, 0, sizeof(int), gp.s));
            CK(cudaStreamSynchronize(gp.s));
            throw RouteError(std::string("router produced NaN/Inf or out-of-range expert id during ") + stage);
        }
    }
}

bool Model::read_expert_blob(const Layer& l, int e, uint8_t* dst) const {
    if (l.host_blobs != nullptr) {
        std::memcpy(dst, l.host_blobs + (size_t) e * l.blob, l.blob);
        ++ram_fetch_picks_;
        return true;
    }
    const size_t nr = ssd_readers_.size();
    if (!l.ssd_backed || (size_t) l.gate_file >= nr || (size_t) l.up_file >= nr || (size_t) l.down_file >= nr) return false;
    const bool ok =
        ssd_readers_[(size_t) l.gate_file]->read_span(l.gate_file_off + (uint64_t) e * l.eg_stride, l.eg_stride, dst + l.tr[0]) &&
        ssd_readers_[(size_t) l.up_file]->read_span(l.up_file_off + (uint64_t) e * l.eu_stride, l.eu_stride, dst + l.tr[1]) &&
        ssd_readers_[(size_t) l.down_file]->read_span(l.down_file_off + (uint64_t) e * l.ed_stride, l.ed_stride, dst + l.tr[2]);
    if (ok) ++ssd_fetch_picks_;
    return ok;
}

void Model::run_cpu_entries(Layer& l, const std::vector<hy4::CpuEntry>& entries, const float* x, int T, float* out) {
    if (l.mixed_gu())   // (its callers never send such a layer here: the lane would read the up with the gate's type)
        throw std::runtime_error("cpu lane: a layer whose gate and up experts are of different types");
    if (entries.empty()) {
        std::fill(out, out + (size_t) T * c_.n_embd, 0.0f);
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const int E = c_.n_embd, FE = c_.n_ff_exp;
    if (l.host_blobs != nullptr) {
        ram_fetch_picks_ += (unsigned long long) entries.size();
        if (o_.lane_parity && par::supported(l.etg) && par::supported(l.etd)) {
            thread_local std::vector<const uint8_t*> gp_v, up_v, dp_v;
            gp_v.resize((size_t) c_.n_expert);
            up_v.resize((size_t) c_.n_expert);
            dp_v.resize((size_t) c_.n_expert);
            for (const auto& ce : entries) {
                const uint8_t* b = l.host_blobs + (size_t) ce.e * l.blob;
                gp_v[(size_t) ce.e] = b + l.tr[0];
                up_v[(size_t) ce.e] = b + l.tr[1];
                dp_v[(size_t) ce.e] = b + l.tr[2];
            }
            par::cpu_experts_parity(*lane_, l.etg, l.etd, E, FE, l.cex.limit, entries, gp_v.data(), up_v.data(),
                                    dp_v.data(), x, T, out);
        } else {
            hy4::cpu_experts(*lane_, l.cex, entries, x, T, out);
        }
    } else {
        std::vector<int> uniq;
        uniq.reserve(entries.size());
        for (const auto& ce : entries) {
            if (std::find(uniq.begin(), uniq.end(), ce.e) == uniq.end()) uniq.push_back(ce.e);
        }
        thread_local std::vector<uint8_t> ssd_buf;
        ssd_buf.resize(uniq.size() * l.blob);
        uint8_t* ssd_p = ssd_buf.data();
        std::atomic<bool> io_ok{true};
        auto rd = [&, ssd_p](int idx) {
            if (!read_expert_blob(l, uniq[(size_t) idx], ssd_p + (size_t) idx * l.blob)) io_ok.store(false);
        };
        if (lane_ && uniq.size() >= 2) lane_->parallel_for((int) uniq.size(), rd);
        else for (int i = 0; i < (int) uniq.size(); ++i) rd(i);
        if (!io_ok.load()) throw std::runtime_error("SSD expert read failed");

        std::vector<hy4::CpuEntry> remapped;
        remapped.reserve(entries.size());
        for (const auto& ce : entries) {
            const int local = (int) (std::find(uniq.begin(), uniq.end(), ce.e) - uniq.begin());
            remapped.push_back({ce.t, local, ce.w});
        }
        if (o_.lane_parity && par::supported(l.etg) && par::supported(l.etd)) {
            std::vector<const uint8_t*> gp_v(uniq.size()), up_v(uniq.size()), dp_v(uniq.size());
            for (size_t i = 0; i < uniq.size(); ++i) {
                const uint8_t* b = ssd_buf.data() + i * l.blob;
                gp_v[i] = b + l.tr[0];
                up_v[i] = b + l.tr[1];
                dp_v[i] = b + l.tr[2];
            }
            par::cpu_experts_parity(*lane_, l.etg, l.etd, E, FE, l.cex.limit, remapped, gp_v.data(), up_v.data(),
                                    dp_v.data(), x, T, out);
        } else {
            hy4::CpuExperts tmp = l.cex;
            tmp.gate = ssd_buf.data() + l.tr[0];
            tmp.up = ssd_buf.data() + l.tr[1];
            tmp.down = ssd_buf.data() + l.tr[2];
            tmp.step = l.blob;
            hy4::cpu_experts(*lane_, tmp, remapped, x, T, out);
        }
    }
    if (T == 1 && !o_.repeatable && l.host_blobs != nullptr) {
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() /
                          (double) entries.size();
        if (cpu_ema_ms_per_exp_ <= 0.0) cpu_ema_ms_per_exp_ = ms;
        else cpu_ema_ms_per_exp_ = 0.92 * cpu_ema_ms_per_exp_ + 0.08 * ms;
        if (base_cpu_ms_per_exp_ > 0.0 && l.host_pinned) {
            if (cpu_ema_ms_per_exp_ > base_cpu_ms_per_exp_ * 2.0) busy_pc_shift_ = 2;
            else if (cpu_ema_ms_per_exp_ < base_cpu_ms_per_exp_ * 1.35) busy_pc_shift_ = 0;
        }
    }
}

bool Model::save_profile_now(const std::string& path, std::string& err) const {
    const int n_moe = (c_.n_layers - c_.n_dense_lead) + (mtp_loaded_ ? 1 : 0);
    sys::ExpertProfile pr;
    pr.n_moe_layers = (uint32_t) n_moe;
    pr.n_expert = (uint32_t) c_.n_expert;
    pr.counts.assign((size_t) n_moe * (size_t) c_.n_expert, 0);
    for (int il = c_.n_dense_lead; il < c_.n_layers; ++il) {
        const auto& pc = layers_[(size_t) il].profile_counts;
        if (pc.size() == (size_t) c_.n_expert)
            std::copy(pc.begin(), pc.end(), pr.counts.begin() + (size_t) (il - c_.n_dense_lead) * c_.n_expert);
    }
    if (mtp_loaded_ && mtp_.layer.profile_counts.size() == (size_t) c_.n_expert) {
        std::copy(mtp_.layer.profile_counts.begin(), mtp_.layer.profile_counts.end(),
                  pr.counts.begin() + (size_t) (c_.n_layers - c_.n_dense_lead) * c_.n_expert);
    }
    return pr.save(path, err);
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

    const bool use_lane = (l.n_res < c.n_expert) && !l.mixed_gu() &&   // (mixed gate / up: the GPU fetch, pinned RAM)
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

    // Lane parity (--repeatable): re-quantise gp.cur to Q8_K into xq/xd BEFORE ev_norm. The shared expert below
    // reads xq/xd on s_copy; doing this later on s (as before) raced with it, so the shared expert saw the norm's
    // quants on some steps and Q8_K ones on others (#1224 repeatable). Every path now gives it the Q8_K quants.
    if (o_.lane_parity) k::q8k_rows(gp.cur, 1, E, gp.xq, gp.xd, s);

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

    last_kernel_ = "router_grouped";
    k::router_grouped(gp.rlog, l.rb_dev, 1, c.n_expert, NU, c.n_group, c.n_group_used, c.w_norm, c.w_scale, gp.d_ids,
                      gp.d_w, s, gp.d_route_err);
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
    a.tu = l.tu_arg();   // (0: the gate's type; else gate and up in two passes)
    a.td = l.etd;
    a.act = iq::kActSwiglu;
    a.pexp = o_.lane_parity ? 1 : 0;

    if (l.n_res == c.n_expert) {
        hy::plan_experts(gp.d_ids, NU, l.fl.vram, l.blob, gp.d_plan, s);
        if (!iq::gate_up(a, NU, 1, s)) throw std::runtime_error("moe: gate_up failed");
        if (seam && c.clamp_exp[il] > 1e-6f)
            dump_seam_f32(gp, "ffn_moe_swiglu_limited-" + std::to_string(il), gp.eh, (size_t) NU * FE);
        if (o_.lane_parity) k::q8k_rows(gp.eh, NU, FE, gp.ehq, gp.ehd, s);
        else hy::q8_rows(gp.eh, (size_t) FE, NU, FE, gp.ehq, gp.ehd, s);
        if (!iq::down(a, NU, 1, s)) throw std::runtime_error("moe: down failed");
        hy::moe_sum(gp.part, 1, NU, E, gp.d_w, gp.moe, s);
        if (seam) run_shared(s);
        else CK(cudaStreamWaitEvent(s, gp.ev_sh, 0));
        finish_add();
        return;
    }

    if (!use_lane) {
        k::fetch_resolve_fast(gp.d_ids, 1, NU, l.fl, gp.fxs, -1, l.n_res > 0 && !o_.repeatable && !o_.no_promo, 0.98f, s);
        hy::fetch_copy(gp.fxs, l.blob, 32, s);
        hy::plan_experts(gp.d_ids, NU, nullptr, 0, gp.d_plan, s, nullptr, gp.d_w, gp.fxs.eptr, gp.fxs.who);
        if (!iq::gate_up(a, NU, 1, s)) throw std::runtime_error("moe: gate_up failed");
        if (seam && c.clamp_exp[il] > 1e-6f)
            dump_seam_f32(gp, "ffn_moe_swiglu_limited-" + std::to_string(il), gp.eh, (size_t) NU * FE);
        if (o_.lane_parity) k::q8k_rows(gp.eh, NU, FE, gp.ehq, gp.ehd, s);
        else hy::q8_rows(gp.eh, (size_t) FE, NU, FE, gp.ehq, gp.ehd, s);
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

    const int k_gpu = (!o_.repeatable && l.host_pinned && (active_offload_ == OffloadMode::kMix || busy_pc_shift_ > 0))
                          ? std::max(o_.mix_k_gpu, busy_pc_shift_)
                          : 0;
    const bool adapt = (k_gpu > 0 && l.n_res > 0 && !o_.repeatable && !o_.no_promo);
    k::fetch_resolve_fast(gp.d_ids, 1, NU, l.fl, gp.fxs, k_gpu, adapt, 0.98f, s);
    k::pack_route_cpu(gp.d_ids, gp.d_w, gp.fxs.who, NU, gp.d_route_pack, s);
    CK(cudaEventRecord(gp.ev_route, s));

    // Single 96-byte D2H transfer on dedicated s_d2h (NEVER blocked by s_copy background H2D promos!)
    CK(cudaStreamWaitEvent(gp.s_d2h, gp.ev_route, 0));
    CK(cudaMemcpyAsync(gp.h_route_pack, gp.d_route_pack, sizeof(k::RoutePack), cudaMemcpyDeviceToHost, gp.s_d2h));
    CK(cudaEventRecord(gp.ev_d2h, gp.s_d2h));

    if (k_gpu > 0) hy::fetch_copy(gp.fxs, l.blob, 32, s);
    if (l.n_res > 0 || k_gpu > 0) {
        hy::plan_experts(gp.d_ids, NU, nullptr, 0, gp.d_plan, s, nullptr, o_.lane_parity ? gp.rlog : gp.d_w, gp.fxs.eptr, gp.fxs.who);
        if (!iq::gate_up(a, NU, 1, s)) throw std::runtime_error("moe: gate_up failed");
        if (seam && c.clamp_exp[il] > 1e-6f)
            dump_seam_f32(gp, "ffn_moe_swiglu_limited-" + std::to_string(il), gp.eh, (size_t) NU * FE);
        if (o_.lane_parity) k::q8k_rows(gp.eh, NU, FE, gp.ehq, gp.ehd, s);
        else hy::q8_rows(gp.eh, (size_t) FE, NU, FE, gp.ehq, gp.ehd, s);
        if (!iq::down(a, NU, 1, s)) throw std::runtime_error("moe: down failed");
        if (!o_.lane_parity) hy::moe_sum(gp.part, 1, NU, E, gp.d_w, gp.moe, s);
    } else if (!o_.lane_parity) {
        CK(cudaMemsetAsync(gp.moe, 0, (size_t) E * sizeof(float), s));
    }
    run_shared(s);

    CK(cudaEventSynchronize(gp.ev_d2h));
    const k::RoutePack& rp = *gp.h_route_pack;
    std::vector<hy4::CpuEntry> entries;
    std::vector<int> entry_slots;
    entries.reserve((size_t) NU);
    entry_slots.reserve((size_t) NU);
    for (int j2 = 0; j2 < NU; ++j2) {
        const int eid = rp.ids[j2];
        if (eid >= 0 && eid < c.n_expert && !l.profile_counts.empty()) ++l.profile_counts[(size_t) eid];
        if (rp.who[j2] == 1) {
            entries.push_back({o_.lane_parity ? (int) entry_slots.size() : 0, eid, o_.lane_parity ? 1.0f : rp.w[j2]});
            entry_slots.push_back(j2);
        }
    }
    if (lookahead_on_ && il + 1 < gp.l1 && !o_.repeatable && rp.ids[0] >= 0 && rp.ids[0] < c.n_expert) {
        Layer& nxt = layers_[(size_t) (il + 1)];
        const int cur_top = rp.ids[0];
        if (l.last_top_e >= 0 && l.last_top_e < c.n_expert && !l.la_next.empty()) {
            const int pred = l.la_next[(size_t) cur_top];
#if defined(__GNUC__) || defined(__clang__)
            if (pred >= 0 && pred < c.n_expert && nxt.h_slot[(size_t) pred] < 0 && nxt.host_blobs != nullptr) {
                __builtin_prefetch(nxt.host_blobs + (size_t) pred * nxt.blob, 0, 3);
                __builtin_prefetch(nxt.host_blobs + (size_t) pred * nxt.blob + nxt.tr[2], 0, 3);
            }
#endif
        }
        if (il > gp.l0 && il > c.n_dense_lead) {
            Layer& prev = layers_[(size_t) (il - 1)];
            if (prev.last_top_e >= 0 && prev.last_top_e < c.n_expert && !prev.la_next.empty())
                prev.la_next[(size_t) prev.last_top_e] = (int16_t) cur_top;
        }
        l.last_top_e = cur_top;
    }
    if (!entries.empty()) {
        cpu_lane_picks_ += (unsigned long long) entries.size();
        if (o_.lane_parity) {
            // Compute each CPU expert unweighted into its own row, copy into gp.part[j2], then sum all 0..NU-1 in order
            std::vector<float> h_x_rep(entry_slots.size() * (size_t) E);
            for (size_t k = 0; k < entry_slots.size(); ++k)
                std::memcpy(h_x_rep.data() + k * (size_t) E, gp.h_cur, (size_t) E * sizeof(float));
            std::vector<float> h_part_cpu(entry_slots.size() * (size_t) E, 0.0f);
            run_cpu_entries(l, entries, h_x_rep.data(), (int) entry_slots.size(), h_part_cpu.data());
            for (size_t k = 0; k < entry_slots.size(); ++k) {
                CK(cudaMemcpyAsync(gp.part + (size_t) entry_slots[k] * E, h_part_cpu.data() + k * (size_t) E,
                                   (size_t) E * sizeof(float), cudaMemcpyHostToDevice, s));
            }
            CK(cudaStreamSynchronize(s));
            hy::moe_sum(gp.part, 1, NU, E, gp.d_w, gp.moe, s);
        } else {
            run_cpu_entries(l, entries, gp.h_cur, 1, gp.h_cpu_out);
            CK(cudaMemcpyAsync(gp.d_cpu_out, gp.h_cpu_out, (size_t) E * sizeof(float), cudaMemcpyHostToDevice, s));
            hy::add(gp.moe, gp.d_cpu_out, (size_t) E, s);
        }
    } else if (o_.lane_parity) {
        hy::moe_sum(gp.part, 1, NU, E, gp.d_w, gp.moe, s);
    }

    if (!o_.repeatable && k_gpu == 0 && l.n_res > 0 && l.n_res < c.n_expert && l.host_blobs != nullptr && l.host_pinned) {
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
        if (!seam && !o_.repeatable && !o_.lane_parity) {
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
    wait_warmup();
    check_token_id(token, "step");
    tok_promo_budget_ = (o_.no_promo || o_.repeatable) ? 0 : 32;
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
                    CK(cudaStreamBeginCapture(gp.s, cudaStreamCaptureModeThreadLocal));
                    decode_segment(gi, true);
                    CK(cudaStreamEndCapture(gp.s, &gp.graph));
                    CK(cudaGraphInstantiate(&gp.graph_exec, gp.graph, nullptr, nullptr, 0));
                    gp.graph_ready = true;
                    ++graph_captures_;
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
    launches_per_tok_ = graph_ok ? (unsigned long long) ngpu * 2ULL : (unsigned long long) c_.n_layers * 14ULL;
    check_route_and_fault("decode");
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
    const int n_res_eff = (layer_borrowed || ((o_.repeatable || o_.lane_parity) && l.n_res < NE)) ? 0 : l.n_res;
    // Repeatable / lane parity, all experts in VRAM: run them in the same sorted sets of <= kPfBatch experts a
    // partly resident layer stages (weights read in place), so a token's expert outputs are added in the same groups
    // and the answer does not depend on whether every expert fit in VRAM (#1224 repeatable)
    const bool rep_sets = (o_.repeatable || o_.lane_parity) && !layer_borrowed && n_res_eff == NE;
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
        last_kernel_ = "router_grouped_prefill";
        k::router_grouped(pf_rlog, l.rb_dev, T, NE, NU, c.n_group, c.n_group_used, c.w_norm, c.w_scale, ids_t, w_t, s, gp.d_route_err);
    }

    // 2. Read back routing IDs and identify hits vs misses
    std::vector<int> miss_list;
    std::vector<int> chunk_counts((size_t) NE, 0);
    int vram_hits = 0;
    int vram_max_rows = 0;
    if (n_res_eff < NE || rep_sets) {
        if (n_res_eff > 0 && !rep_sets) {
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
            if (n_res_eff > 0 && !rep_sets && l.h_slot[(size_t) e] >= 0) {
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
    const bool warm_inline = is_last_block && !layer_borrowed && o_.cache_warm && !o_.no_promo &&
                             n_res_eff > 0 && n_res_eff < NE && l.host_blobs != nullptr;
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
        a.tu = l.tu_arg();   // (0: the gate's type; else gate and up in two passes)
        // (two passes: the gate's F32 sums [rows][n_ff] between them, in the down output's rows - [rows][n_embd],
        //  n_ff <= n_embd, the load's check - which the down writes only after the up's pass has read them)
        if (l.mixed_gu()) a.graw = y;
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

    // A staged set over the block: one pass when its rows fit the set scratch (max_set_rows), else P tokens a tile
    // (one expert alone can have up to B rows - ling_moe_plan.hpp)
    auto run_staged = [&](Gpu& g, const MacroBufs& gb, int buf, const std::vector<int>& blist, int bmax, bool accumulate) {
        long long rows = 0;
        for (int e : blist) rows += chunk_counts[(size_t) e];
        if (ling::moe_plan::one_pass(rows, max_set_rows)) {
            run_set_on(g, gb, 0, B, rep_sets ? l.fl.vram : g.pf_stage[buf], g.pf_pmap[buf], rep_sets ? NE : (int) blist.size(), bmax, accumulate);
            return;
        }
        std::vector<int> in_set((size_t) NE, -1);
        for (size_t b = 0; b < blist.size(); ++b) in_set[(size_t) blist[b]] = (int) b;
        for (const auto& tl : ling::moe_plan::tiles(gp.h_pf_ids, B, NU, in_set, P)) {
            if (tl.max_rows > 0) {
                run_set_on(g, gb, tl.t0, tl.T, rep_sets ? l.fl.vram : g.pf_stage[buf], g.pf_pmap[buf], rep_sets ? NE : (int) blist.size(), tl.max_rows,
                           accumulate);
            } else if (!accumulate) {
                CK(cudaMemsetAsync(gb.moe + (size_t) tl.t0 * E, 0, (size_t) tl.T * E * sizeof(float), g.s));
            }
        }
    };

    // Pack an expert list into row-bounded batches (each <= kPfBatch experts AND <= max_set_rows total rows; an
    // expert with more rows than that is a batch of its own, run in tiles by run_staged)
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
        for (int b = 0; b < Bn; ++b) pmap_h[blist[(size_t) b]] = rep_sets ? blist[(size_t) b] : b;   // (VRAM slot e)
        CK(cudaMemcpyAsync(g.pf_pmap[buf], pmap_h, (size_t) NE * sizeof(int), cudaMemcpyHostToDevice, g.s_copy));
        CK(cudaEventRecord(g.ev_pmap[ring], g.s_copy));
        if (rep_sets) {
            // weights stay where they are in VRAM: nothing to stage
        } else if (l.host_pinned) {
            for (int b = 0; b < Bn; ++b) {
                const int e = blist[(size_t) b];
                CK(cudaMemcpyAsync(g.pf_stage[buf] + (size_t) b * l.blob, l.host_blobs + (size_t) e * l.blob, l.blob,
                                   cudaMemcpyHostToDevice, g.s_copy));
            }
        } else {
            CK(cudaEventSynchronize(g.ev_copy[buf]));
            std::atomic<bool> rd_ok{true};
            auto cp = [&](int b) {
                const int e = blist[(size_t) b];
                uint8_t* dst = g.h_pf_stage[buf] + (size_t) b * l.blob;
                if (l.host_blobs != nullptr) {
                    std::memcpy(dst, l.host_blobs + (size_t) e * l.blob, l.blob);
                    ++ram_fetch_picks_;
                } else {
                    if (!read_expert_blob(l, e, dst)) rd_ok.store(false);
                }
            };
            if (lane_ && Bn >= 2) lane_->parallel_for(Bn, cp);
            else for (int b = 0; b < Bn; ++b) cp(b);
            if (!rd_ok.load()) throw std::runtime_error("prefill SSD expert read failed");
            CK(cudaMemcpyAsync(g.pf_stage[buf], g.h_pf_stage[buf], (size_t) Bn * l.blob, cudaMemcpyHostToDevice,
                               g.s_copy));
        }
        CK(cudaEventRecord(g.ev_copy[buf], g.s_copy));
    };

    if (o_.repeatable || o_.lane_parity) std::sort(miss_list.begin(), miss_list.end());
    const bool cpu_miss = !o_.repeatable && !o_.lane_parity && !miss_list.empty() && lane_ && B <= 48 && !l.mixed_gu();
    const bool use_ep = !o_.repeatable && !o_.lane_parity && o_.ep_prefill && !cpu_miss && ngpu >= 2 && B >= 256 &&
                        (int) miss_list.size() >= 32 && l.host_pinned;

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
            run_staged(hg, hb, buf, blist, bmax, bi > 0);
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
        run_cpu_entries(l, entries, h_x.data(), B, h_o.data());
        CK(cudaMemcpyAsync(gp.pf_out, h_o.data(), (size_t) B * E * 4, cudaMemcpyHostToDevice, s));
        CK(cudaStreamSynchronize(s));
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
            run_staged(gp, mb, buf, blist, bmax, accum);
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
    if (!o_.cache_warm || o_.repeatable || o_.no_promo) return;
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

const float* Model::prefill(const int* tokens, int n, int pos0, std::vector<float>* all_logits_host,
                            const std::function<bool(int, int)>& on_progress,
                            const std::atomic<bool>* stop_flag) {
    wait_warmup();
    if (n <= 0) throw std::runtime_error("prefill: empty prompt");
    for (int i = 0; i < n; ++i) check_token_id(tokens[i], "prefill");
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
            if (on_progress && ((i + 1) == n || ((i + 1) % 64) == 0)) {
                set_dev(output_dev());
                CK(cudaStreamSynchronize(output_stream()));
                if (!on_progress(i + 1, n)) return nullptr;
            }
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
    const auto t_pf_start = std::chrono::steady_clock::now();
    std::vector<double> free_before_gib((size_t) ngpu, 0.0);
    for (int gi = 0; gi < ngpu; ++gi) {
        set_dev(gpus_[(size_t) gi].dev);
        size_t fb = 0, tb = 0;
        if (cudaMemGetInfo(&fb, &tb) == cudaSuccess) free_before_gib[(size_t) gi] = (double) fb / (double) (1ULL << 30);
    }
    if (mtp_loaded_) {
        mtp_pf_h_host_.assign((size_t) n * (size_t) E, 0.0f);
        mtp_pf_pos0_ = pos0;
    }

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
    if (!o_.repeatable && !o_.lane_parity && (any_offloaded || ngpu >= 2) && n > P && prefill_block_ > P) {
        int want_cap = std::min(std::min(prefill_block_, gpus_[0].h_pf_cap), ((n + P - 1) / P) * P);
        if (ngpu >= 2 && !o_.ep_prefill && all_logits_host == nullptr) {
            // Keep at least 2 blocks per stage-pipeline depth so multi-GPU stages overlap cleanly
            const int pipe_cap = std::max(P, ((n / std::max(2, ngpu) + P - 1) / P) * P);
            if (n > P * 2) want_cap = std::min(want_cap, pipe_cap);
        }
        int top_cand = P;
        while (top_cand <= want_cap / 2) top_cand *= 2;
        for (int cand = top_cand; cand > P; cand >>= 1) {
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
                    if (l.n_res > 0 && l.vram_pool_off + (size_t) l.n_res * l.blob > cut && l.host_blobs == nullptr && !l.ssd_backed) {
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
        const bool can_resident_pipe = (want_pipe_buf && !o_.repeatable && !o_.lane_parity && n > P &&
                                        gpus_[0].pf_pipe_x != nullptr);
        if (can_resident_pipe) has_pipe_x = true;
        for (int gi = 0; gi < ngpu; ++gi) {
            Gpu& gp = gpus_[(size_t) gi];
            mbufs[(size_t) gi] = {gp.pf_x, gp.pf_cur, gp.pf_moe, gp.pf_w,
                                  can_resident_pipe ? gp.pf_pipe_x : nullptr,
                                  gp.pf_ids, gp.pf_tok, gp.pf_row_of};
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
        if (gi == ngpu - 1 && mtp_loaded_ && mtp_pf_h_host_.size() >= (size_t) (off + B) * (size_t) E) {
            set_dev(gp.dev);
            for (int t0 = 0; t0 < B; t0 += P) {
                const int T = std::min(P, B - t0);
                hy::rmsnorm(x_buf + (size_t) t0 * E, (size_t) E, T, E, out_norm_, c.eps, gp.pf_cur, (size_t) E, gp.s);
                CK(cudaMemcpyAsync(mtp_pf_h_host_.data() + (size_t) (off + t0) * E, gp.pf_cur,
                                   (size_t) T * (size_t) E * sizeof(float), cudaMemcpyDeviceToHost, gp.s));
            }
            CK(cudaStreamSynchronize(gp.s));
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
        std::atomic<bool> pipe_abort{false};
        auto wake_all = [&] {
            pipe_abort.store(true, std::memory_order_release);
            for (auto& lnk : links) {
                std::lock_guard<std::mutex> lk(lnk.mu);
                lnk.cv.notify_all();
            }
        };
        auto check_stop = [&]() -> bool {
            if (pipe_abort.load(std::memory_order_acquire) ||
                (stop_flag != nullptr && stop_flag->load(std::memory_order_relaxed))) {
                wake_all();
                return true;
            }
            return false;
        };

        auto stage_worker = [&](int gi) {
            try {
                Gpu& gp = gpus_[(size_t) gi];
                (void) sys::pin_thread_to_numa_node(cpu_topo_, gp.numa_node);
                float* x_slots[2] = {mbufs[(size_t) gi].x, mbufs[(size_t) gi].pipe_x};
                for (int blk = 0; blk < n_blocks; ++blk) {
                    if (check_stop()) break;
                    const int slot = blk & 1;
                    const int off = blk * B_max;
                    const int B = std::min(B_max, n - off);
                    const bool is_last = (blk + 1 == n_blocks);

                    if (gi > 0) {
                        Link& in_link = links[(size_t) (gi - 1)];
                        {
                            std::unique_lock<std::mutex> lk(in_link.mu);
                            while (!(in_link.ready[slot] == blk || pipe_abort.load(std::memory_order_acquire) ||
                                     (stop_flag != nullptr && stop_flag->load(std::memory_order_relaxed)))) {
                                in_link.cv.wait_for(lk, std::chrono::milliseconds(15));
                            }
                        }
                        if (check_stop()) break;
                    }

                    run_gpu_stage(gi, off, B, is_last, x_slots[slot]);
                    if (check_stop()) break;
                    if (gi == ngpu - 1 && on_progress) {
                        set_dev(gpus_.back().dev);
                        CK(cudaStreamSynchronize(gpus_.back().s));
                        if (!on_progress(off + B, n)) {
                            wake_all();
                            break;
                        }
                    }

                    if (gi + 1 < ngpu) {
                        // Wait until next stage finished using x_slots_next[slot] from blk - 2
                        Link& out_link = links[(size_t) gi];
                        if (blk >= 2) {
                            {
                                std::unique_lock<std::mutex> lk(out_link.mu);
                                while (!(out_link.consumed[slot] >= blk - 2 || pipe_abort.load(std::memory_order_acquire) ||
                                         (stop_flag != nullptr && stop_flag->load(std::memory_order_relaxed)))) {
                                    out_link.cv.wait_for(lk, std::chrono::milliseconds(15));
                                }
                            }
                            if (check_stop()) break;
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
                        if (gi + 1 == ngpu) {
                            set_dev(gp.dev);
                            CK(cudaStreamSynchronize(gp.s));
                        }
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
                wake_all();
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

        bool nonpipe_abort = false;
        for (int off = 0; off < n && !nonpipe_abort; off += B_max) {
            const int B = std::min(B_max, n - off);
            const bool is_last = (off + B == n);
            for (int gi = 0; gi < ngpu; ++gi) {
                if (stop_flag != nullptr && stop_flag->load(std::memory_order_relaxed)) {
                    nonpipe_abort = true;
                    break;
                }
                if (gi > 0) {
                    transfer_buf(gi - 1, gi, mbufs[(size_t) (gi - 1)].x, mbufs[(size_t) gi].x, (size_t) B * E);
                }
                run_gpu_stage(gi, off, B, is_last, mbufs[(size_t) gi].x);
                if (stop_flag != nullptr && stop_flag->load(std::memory_order_relaxed)) {
                    nonpipe_abort = true;
                    break;
                }
                if (gi == ngpu - 1 && on_progress) {
                    set_dev(gpus_.back().dev);
                    CK(cudaStreamSynchronize(gpus_.back().s));
                    if (!on_progress(off + B, n)) {
                        nonpipe_abort = true;
                        break;
                    }
                }
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
            if (l.n_res <= 0 || (l.host_blobs == nullptr && !l.ssd_backed)) continue;
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
                if (l.host_blobs != nullptr) {
                    CK(cudaMemcpyAsync(l.fl.vram + (size_t) sl * l.blob, l.host_blobs + (size_t) e * l.blob, l.blob,
                                       cudaMemcpyHostToDevice, gp.s_copy));
                } else {
                    std::vector<uint8_t> tmp(l.blob);
                    if (read_expert_blob(l, e, tmp.data())) {
                        CK(cudaMemcpyAsync(l.fl.vram + (size_t) sl * l.blob, tmp.data(), l.blob,
                                           cudaMemcpyHostToDevice, gp.s_copy));
                        CK(cudaStreamSynchronize(gp.s_copy));
                    }
                }
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
    if (stop_flag != nullptr && stop_flag->load(std::memory_order_relaxed)) {
        return nullptr;
    }
    check_route_and_fault("prefill");
    if (mtp_loaded_) {
        if ((int) mtp_tokens_.size() < pos0 + n) mtp_tokens_.resize((size_t) (pos0 + n), 0);
        for (int i = 0; i < n; ++i) mtp_tokens_[(size_t) (pos0 + i)] = tokens[i];
        mtp_prefill_prompt(tokens, n, pos0);
    }
    {
        const double pf_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pf_start).count();
        std::ostringstream vbefore, vafter;
        vbefore.setf(std::ios::fixed); vbefore.precision(2);
        vafter.setf(std::ios::fixed); vafter.precision(2);
        for (int gi = 0; gi < ngpu; ++gi) {
            set_dev(gpus_[(size_t) gi].dev);
            size_t fb = 0, tb = 0;
            double fa = 0.0;
            if (cudaMemGetInfo(&fb, &tb) == cudaSuccess) fa = (double) fb / (double) (1ULL << 30);
            if (gi > 0) { vbefore << ","; vafter << ","; }
            vbefore << free_before_gib[(size_t) gi];
            vafter << fa;
        }
        const double warm_ms = last_warm_s_ * 1000.0;
        const double idle_pct = pf_ms > 0.0 ? std::min(100.0, 100.0 * warm_ms / pf_ms) : 0.0;
        std::fprintf(stderr,
                     "ling prompt: tok=%d pos0=%d B=%d P=%d total=%.1fms (compute=%.1fms warm=%.1fms, %.1f t/s) "
                     "free_vram_before=[%s] free_vram_after=[%s] idle=%.1f%%\n",
                     n, pos0, B_max, P, pf_ms, std::max(0.0, pf_ms - warm_ms), warm_ms,
                     pf_ms > 0.0 ? 1000.0 * (double) n / pf_ms : 0.0, vbefore.str().c_str(), vafter.str().c_str(), idle_pct);
        if (n >= 1024 && B_max < 512) {
            std::fprintf(stderr, "ling prompt: WARNING tiny prefill window (B=%d for n=%d tokens)\n", B_max, n);
        }
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
                         gp.d_step + 2, W, H, KL, R, 1.0f / sqrtf((float) c.qk_head), gp.mla_part, gp.ctxo, s)) {
#if defined(MAYA_USE_HIP)
        // AMD (no tensor-core window kernel): each row on the token path's kernel with its own cache length
        // (set_step_win's counts) - the rows a verify window gets are the ones the tokens get one at a time
        for (int t = 0; t < W; ++t)
            if (!glmd::mla_dec(gp.q_abs + (size_t) t * H * KL, gp.qm + (size_t) t * H * c.qk_head, c.qk_head, c.qk_nope,
                               l.lat, o_.kv_q8 ? 1 : 0, l.k_pe, gp.d_iota, o_.ctx, gp.d_step + k::kStepWinCounts + t, H,
                               KL, R, 1.0f / sqrtf((float) c.qk_head), gp.mla_part, gp.ctxo + (size_t) t * H * KL, s))
                throw std::runtime_error("mla_win: mla_dec refused shape");
#else
        throw std::runtime_error("mla_win: mla_dec_wmma failed");
#endif
    }

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
    const bool use_lane = (l.n_res < c.n_expert) && !l.mixed_gu() &&   // (mixed gate / up: the GPU fetch, pinned RAM)
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

    // Lane parity: Q8_K quants into xq/xd before the shared expert starts on s_copy (see ffn_moe_decode).
    if (o_.lane_parity) k::q8k_rows(gp.cur, W, E, gp.xq, gp.xd, s);

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
    last_kernel_ = "router_grouped_win";
    k::router_grouped(gp.rlog, l.rb_dev, W, c.n_expert, NU, c.n_group, c.n_group_used, c.w_norm, c.w_scale, gp.d_ids, gp.d_w, s, gp.d_route_err);

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
    a.tu = l.tu_arg();   // (0: the gate's type; else gate and up in two passes)
    a.td = l.etd;
    a.act = iq::kActSwiglu;
    a.pexp = o_.lane_parity ? 1 : 0;
    const int groups = std::min(W * NU, c.n_expert);

    if (l.n_res == c.n_expert) {
        hy::group_experts(gp.d_ids, W, NU, c.n_expert, l.fl.vram, l.blob, gp.grp_ptr, gp.grp_start, gp.grp_ent, gp.n_groups, s);
        if (!iq::gate_up(a, groups, W, s)) throw std::runtime_error("moe_win: gate_up failed");
        if (o_.lane_parity) k::q8k_rows(gp.eh, W * 16, FE, gp.ehq, gp.ehd, s);
        else hy::q8_rows(gp.eh, (size_t) FE, W * 16, FE, gp.ehq, gp.ehd, s);
        if (!iq::down(a, groups, W, s)) throw std::runtime_error("moe_win: down failed");
        hy::moe_sum(gp.part, W, NU, E, gp.d_w, gp.moe, s);
        CK(cudaStreamWaitEvent(s, gp.ev_sh, 0));
        finish_add();
        return;
    }

    if (!use_lane) {
        k::fetch_resolve_fast(gp.d_ids, W, NU, l.fl, gp.fxs, -1, l.n_res > 0 && !o_.repeatable, 0.98f, s);
        hy::fetch_copy(gp.fxs, l.blob, 32, s);
        hy::group_experts(gp.d_ids, W, NU, c.n_expert, nullptr, 0, gp.grp_ptr, gp.grp_start, gp.grp_ent, gp.n_groups,
                          s, nullptr, gp.d_w, gp.fxs.eptr, gp.fxs.who);
        if (!iq::gate_up(a, groups, W, s)) throw std::runtime_error("moe_win: gate_up failed");
        if (o_.lane_parity) k::q8k_rows(gp.eh, W * 16, FE, gp.ehq, gp.ehd, s);
        else hy::q8_rows(gp.eh, (size_t) FE, W * 16, FE, gp.ehq, gp.ehd, s);
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

    const int k_gpu = (!o_.repeatable && l.host_pinned && (active_offload_ == OffloadMode::kMix || busy_pc_shift_ > 0))
                          ? std::max(o_.mix_k_gpu, busy_pc_shift_)
                          : 0;
    const bool adapt = (k_gpu > 0 && l.n_res > 0 && !o_.repeatable && !o_.no_promo);
    k::fetch_resolve_fast(gp.d_ids, W, NU, l.fl, gp.fxs, k_gpu, adapt, 0.98f, s);
    k::pack_route_win_cpu(gp.d_ids, gp.d_w, gp.fxs.who, W * NU, gp.d_route_win, s);
    CK(cudaEventRecord(gp.ev_route, s));

    CK(cudaStreamWaitEvent(gp.s_d2h, gp.ev_route, 0));
    CK(cudaMemcpyAsync(gp.h_route_win, gp.d_route_win, sizeof(k::RoutePackWin), cudaMemcpyDeviceToHost, gp.s_d2h));
    CK(cudaEventRecord(gp.ev_d2h, gp.s_d2h));

    if (k_gpu > 0) hy::fetch_copy(gp.fxs, l.blob, 32, s);
    if (l.n_res > 0 || k_gpu > 0) {
        hy::group_experts(gp.d_ids, W, NU, c.n_expert, nullptr, 0, gp.grp_ptr, gp.grp_start, gp.grp_ent, gp.n_groups,
                          s, nullptr, o_.lane_parity ? gp.rlog : gp.d_w, gp.fxs.eptr, gp.fxs.who);
        if (!iq::gate_up(a, groups, W, s)) throw std::runtime_error("moe_win: gate_up failed");
        if (o_.lane_parity) k::q8k_rows(gp.eh, W * 16, FE, gp.ehq, gp.ehd, s);
        else hy::q8_rows(gp.eh, (size_t) FE, W * 16, FE, gp.ehq, gp.ehd, s);
        if (!iq::down(a, groups, W, s)) throw std::runtime_error("moe_win: down failed");
        if (!o_.lane_parity) hy::moe_sum(gp.part, W, NU, E, gp.d_w, gp.moe, s);
    } else if (!o_.lane_parity) {
        CK(cudaMemsetAsync(gp.moe, 0, (size_t) W * E * sizeof(float), s));
    }
    run_shared(s);

    CK(cudaEventSynchronize(gp.ev_d2h));
    const k::RoutePackWin& rp = *gp.h_route_win;
    std::vector<hy4::CpuEntry> entries;
    std::vector<int> entry_slots;
    entries.reserve((size_t) (W * NU));
    entry_slots.reserve((size_t) (W * NU));
    for (int t = 0; t < W; ++t) {
        for (int j2 = 0; j2 < NU; ++j2) {
            const int idx = t * NU + j2;
            const int eid = rp.ids[idx];
            if (eid >= 0 && eid < c.n_expert && !l.profile_counts.empty()) ++l.profile_counts[(size_t) eid];
            if (rp.who[idx] == 1) {
                entries.push_back({o_.lane_parity ? (int) entry_slots.size() : t, eid, o_.lane_parity ? 1.0f : rp.w[idx]});
                entry_slots.push_back(idx);
            }
        }
    }
    if (!entries.empty()) {
        cpu_lane_picks_ += (unsigned long long) entries.size();
        if (o_.lane_parity) {
            std::vector<float> h_x_rep(entry_slots.size() * (size_t) E);
            for (size_t k = 0; k < entry_slots.size(); ++k) {
                const int t = entry_slots[k] / NU;
                std::memcpy(h_x_rep.data() + k * (size_t) E, gp.h_cur + (size_t) t * E, (size_t) E * sizeof(float));
            }
            std::vector<float> h_part_cpu(entry_slots.size() * (size_t) E, 0.0f);
            run_cpu_entries(l, entries, h_x_rep.data(), (int) entry_slots.size(), h_part_cpu.data());
            for (size_t k = 0; k < entry_slots.size(); ++k) {
                CK(cudaMemcpyAsync(gp.part + (size_t) entry_slots[k] * E, h_part_cpu.data() + k * (size_t) E,
                                   (size_t) E * sizeof(float), cudaMemcpyHostToDevice, s));
            }
            CK(cudaStreamSynchronize(s));
            hy::moe_sum(gp.part, W, NU, E, gp.d_w, gp.moe, s);
        } else {
            run_cpu_entries(l, entries, gp.h_cur, W, gp.h_cpu_out);
            CK(cudaMemcpyAsync(gp.d_cpu_out, gp.h_cpu_out, (size_t) W * E * sizeof(float), cudaMemcpyHostToDevice, s));
            hy::add(gp.moe, gp.d_cpu_out, (size_t) W * E, s);
        }
    } else if (o_.lane_parity) {
        hy::moe_sum(gp.part, W, NU, E, gp.d_w, gp.moe, s);
    }

    // Async background promotion (same race-free policy as T=1)
    if (!o_.repeatable && k_gpu == 0 && l.n_res > 0 && l.n_res < c.n_expert && l.host_blobs != nullptr && l.host_pinned) {
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
        if (!o_.repeatable && !o_.lane_parity) {
            if (il + 1 < gp.l1) next_norm = layers_[(size_t) (il + 1)].attn_norm;
            else if (gi == (int) gpus_.size() - 1) next_norm = out_norm_;
        }
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
    wait_warmup();
    if (W <= 1) return step(tokens[0], pos0);
    if (W > kMaxWin) throw std::runtime_error("step_window requires W <= 4");
    for (int t = 0; t < W; ++t) check_token_id(tokens[t], "step_window");
    tok_promo_budget_ = (o_.no_promo || o_.repeatable) ? 0 : 32;
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
                CK(cudaStreamBeginCapture(gp.s, cudaStreamCaptureModeThreadLocal));
                decode_segment_win(gi, W, true);
                CK(cudaStreamEndCapture(gp.s, &gp.wgraph[wi]));
                CK(cudaGraphInstantiate(&gp.wgraph_exec[wi], gp.wgraph[wi], nullptr, nullptr, 0));
                gp.wgraph_ready[wi] = true;
                ++graph_captures_;
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
    if (mtp_loaded_) k::store_hring(gl.cur, W, c_.n_embd, mtp_.hring, MtpWeights::kRing, pos0, gl.s);
    check_route_and_fault("step_window");
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

void Model::mtp_prefill_prompt(const int* tokens, int n, int pos0) {
    if (!mtp_loaded_ || n <= 1 || mtp_pf_h_host_.size() < (size_t) (n - 1) * (size_t) c_.n_embd) {
        mtp_done_ = pos0 + std::max(0, n - 1);
        return;
    }
    const Config& c = c_;
    const int E = c.n_embd, KL = c.kv_lora, R = c.n_rot;
    Gpu& g0 = gpus_[0];
    Gpu& gl = gpus_.back();
    const int P = std::max(1, gl.pf_cap);
    const int pairs = n - 1;
    float *tmp_eh = nullptr, *tmp_h = nullptr, *tmp_emb = nullptr;
    uint16_t* tmp_eh16 = nullptr;
    set_dev(gl.dev);
    CK(cudaMalloc(&tmp_eh, (size_t) P * (size_t) (2 * E) * sizeof(float)));
    CK(cudaMalloc(&tmp_eh16, (size_t) P * (size_t) (2 * E) * sizeof(uint16_t)));
    CK(cudaMalloc(&tmp_h, (size_t) P * (size_t) E * sizeof(float)));
    CK(cudaMalloc(&tmp_emb, (size_t) P * (size_t) E * sizeof(float)));

    for (int off = 0; off < pairs; off += P) {
        const int T = std::min(P, pairs - off);
        set_dev(g0.dev);
        CK(cudaMemcpyAsync(g0.pf_tok_ids, tokens + off + 1, (size_t) T * sizeof(int), cudaMemcpyHostToDevice, g0.s));
        if (!k::emb_rows(emb_.p, emb_.type, emb_.k, g0.pf_tok_ids, T, g0.dev == gl.dev ? tmp_emb : g0.pf_out, g0.s))
            throw std::runtime_error("mtp_prefill: emb_rows failed");
        if (g0.dev != gl.dev) {
            transfer_buf(0, (int) gpus_.size() - 1, g0.pf_out, tmp_emb, (size_t) T * (size_t) E);
        } else {
            set_dev(gl.dev);
        }
        CK(cudaMemcpyAsync(tmp_h, mtp_pf_h_host_.data() + (size_t) off * (size_t) E,
                           (size_t) T * (size_t) E * sizeof(float), cudaMemcpyHostToDevice, gl.s));
        hy::rmsnorm(tmp_emb, (size_t) E, T, E, mtp_.enorm, c.eps, tmp_eh, (size_t) (2 * E), gl.s);
        hy::rmsnorm(tmp_h, (size_t) E, T, E, mtp_.hnorm, c.eps, tmp_eh + E, (size_t) (2 * E), gl.s);
        hy::to_f16(tmp_eh, (size_t) (2 * E), T, 2 * E, tmp_eh16, (size_t) (2 * E), gl.s);
        gemm_mat(gl, mtp_.eh_proj, tmp_eh16, T, gl.pf_x);
        hy::rmsnorm(gl.pf_x, (size_t) E, T, E, mtp_.layer.attn_norm, c.eps, gl.pf_cur, (size_t) E, gl.s);
        hy::to_f16(gl.pf_cur, (size_t) E, T, E, gl.pf_x16, (size_t) E, gl.s);
        gemm_mat(gl, mtp_.layer.wkva, gl.pf_x16, T, gl.pf_tmp);
        k::kv_store_split(gl.pf_tmp, T, KL, R, mtp_.layer.kv_a_norm, c.eps, nullptr, pos0 + off + 1, c.rope_base,
                          mtp_.layer.lat, o_.kv_q8, mtp_.layer.k_pe, gl.s);
    }
    CK(cudaStreamSynchronize(gl.s));
    cudaFree(tmp_eh);
    cudaFree(tmp_eh16);
    cudaFree(tmp_h);
    cudaFree(tmp_emb);
    mtp_pf_h_host_.clear();
    mtp_pf_h_host_.shrink_to_fit();
    mtp_done_ = pos0 + pairs;
}

void Model::mtp_draft(int base_tok, int base_pos, int max_k, float floor, std::vector<int>& drafts,
                      std::vector<float>& confs, float floor0, float cum_min,
                      std::vector<std::vector<float>>* draft_logits_out) {
    drafts.clear();
    confs.clear();
    if (draft_logits_out) draft_logits_out->clear();
    if (!mtp_loaded_ || max_k <= 0 || base_pos <= 0) return;
    wait_warmup();
    check_token_id(base_tok, "mtp_draft");
    Gpu& gl = gpus_.back();
    const int E = c_.n_embd;

    const int lo = std::max(mtp_done_, std::max(0, base_pos - MtpWeights::kRing + 1));
    for (int p = lo; p < base_pos - 1; ++p) {
        if (p + 1 >= (int) mtp_tokens_.size()) break;
        const float* h_p = mtp_.hring + (size_t) (p % MtpWeights::kRing) * E;
        const int next_t = mtp_tokens_[(size_t) (p + 1)];
        (void) mtp_step(h_p, next_t, p + 1);
    }

    int cur_tok = base_tok;
    const float* cur_h = mtp_.hring + (size_t) ((base_pos - 1) % MtpWeights::kRing) * E;
    float cum_p = 1.0f;
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
        cum_p *= prob;
        if (i > 0 && cum_min > 0.0f && cum_p < cum_min) break;
        drafts.push_back(id);
        confs.push_back(prob);
        if (draft_logits_out) {
            std::vector<float> row((size_t) c_.vocab);
            CK(cudaMemcpyAsync(row.data(), lg, (size_t) c_.vocab * sizeof(float), cudaMemcpyDeviceToHost, gl.s));
            CK(cudaStreamSynchronize(gl.s));
            draft_logits_out->push_back(std::move(row));
        }
        if (prob < floor || (cum_min > 0.0f && cum_p < cum_min) || i + 1 == max_k) break;
        CK(cudaMemcpyAsync(mtp_.h_tmp, gl.hn, (size_t) E * sizeof(float), cudaMemcpyDeviceToDevice, gl.s));
        cur_h = mtp_.h_tmp;
        cur_tok = id;
    }
    mtp_done_ = base_pos;
}

// ------------------------------ MTP block ------------------------------

const float* Model::mtp_step(const float* h_nextn_dev, int next_token, int pos) {
    if (!mtp_loaded_) throw std::runtime_error("mtp_step called without --mtp");
    check_token_id(next_token, "mtp_step");
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
        CK(cudaMemcpyAsync(gp.h_stats, gp.fxs.stats, 8 * sizeof(unsigned long long), cudaMemcpyDeviceToHost, gp.s));
        CK(cudaStreamSynchronize(gp.s));
        st.picks += gp.h_stats[0];
        st.vram_hits += gp.h_stats[1];
        st.copied += gp.h_stats[2];
        st.zero_copy += gp.h_stats[3];
        st.cpu_lane += gp.h_stats[4];
        st.admitted += gp.h_stats[5];
    }
    if (st.cpu_lane == 0) st.cpu_lane = cpu_lane_picks_;
    st.admitted += cpu_admitted_;
    st.ram_fetch = ram_fetch_picks_ + st.copied;
    st.ssd_fetch = ssd_fetch_picks_;
    st.graph_captures = graph_captures_;
    st.launches_per_tok = launches_per_tok_;
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
            if (l.host_blobs == nullptr && !l.ssd_backed) {
                can_lend = false;
                break;
            }
            const int sl_first = l.vram_pool_off >= cut ? 0 : (int) ((cut - l.vram_pool_off) / l.blob);
            RestoreLayer rl{&l, std::vector<int>((size_t) l.n_res, -1), sl_first};
            CK(cudaMemcpyAsync(rl.owners.data(), l.fl.owner, (size_t) l.n_res * sizeof(int), cudaMemcpyDeviceToHost, g0.s));
            to_restore.push_back(std::move(rl));
        }
        CK(cudaStreamSynchronize(g0.s));
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
                    if (l.host_blobs != nullptr) {
                        CK(cudaMemcpyAsync(l.fl.vram + (size_t) sl * l.blob, l.host_blobs + (size_t) e * l.blob, l.blob,
                                           cudaMemcpyHostToDevice, g0.s));
                    } else {
                        std::vector<uint8_t> tmp(l.blob);
                        if (read_expert_blob(l, e, tmp.data())) {
                            CK(cudaMemcpyAsync(l.fl.vram + (size_t) sl * l.blob, tmp.data(), l.blob,
                                               cudaMemcpyHostToDevice, g0.s));
                            CK(cudaStreamSynchronize(g0.s));
                        }
                    }
                }
            }
        }
        CK(cudaStreamSynchronize(g0.s));
    }
    return ok;
}


// ------------------------------ VLEND / VRECLAIM (Serve Contract) ------------------------------

bool Model::vlend(size_t& lent_mib, std::string& err) {
    lent_mib = 0;
    if (!vision_ || !vision_->loaded()) {
        err = "no vision tower loaded (text-only start)";
        return false;
    }
    if (vlent_) {
        lent_mib = vision_->max_room_bytes() >> 20;
        return true;
    }
    Gpu& g0 = gpus_[0];
    set_dev(g0.dev);
    CK(cudaStreamSynchronize(g0.s));
    CK(cudaStreamSynchronize(g0.s_copy));
    const size_t need = vision_->max_room_bytes();
    vlend_saved_.clear();
    if (g0.vram_pool != nullptr && g0.vram_pool_bytes >= need) {
        const size_t cut = g0.vram_pool_bytes - need;
        for (int il = std::max(g0.l0, c_.n_dense_lead); il < g0.l1; ++il) {
            Layer& l = layers_[(size_t) il];
            if (l.n_res <= 0) continue;
            const size_t end_off = l.vram_pool_off + (size_t) l.n_res * l.blob;
            if (end_off <= cut) continue;
            const int sl_first = l.vram_pool_off >= cut ? 0 : (int) ((cut - l.vram_pool_off) / l.blob);
            VlendSaved vs{&l, std::vector<int>((size_t) l.n_res, -1), sl_first};
            CK(cudaMemcpyAsync(vs.owners.data(), l.fl.owner, (size_t) l.n_res * sizeof(int), cudaMemcpyDeviceToHost, g0.s));
            vlend_saved_.push_back(std::move(vs));
        }
        CK(cudaStreamSynchronize(g0.s));
    }
    vlent_ = true;
    lent_mib = need >> 20;
    return true;
}

bool Model::vreclaim(std::string& err) {
    (void) err;
    if (!vlent_) return true;
    Gpu& g0 = gpus_[0];
    set_dev(g0.dev);
    for (const auto& vs : vlend_saved_) {
        Layer& l = *vs.l;
        for (int sl = vs.sl_first; sl < l.n_res; ++sl) {
            const int e = vs.owners[(size_t) sl];
            if (e >= 0 && e < c_.n_expert) {
                if (l.host_blobs != nullptr) {
                    CK(cudaMemcpyAsync(l.fl.vram + (size_t) sl * l.blob, l.host_blobs + (size_t) e * l.blob, l.blob,
                                       cudaMemcpyHostToDevice, g0.s));
                } else {
                    std::vector<uint8_t> tmp(l.blob);
                    if (read_expert_blob(l, e, tmp.data())) {
                        CK(cudaMemcpyAsync(l.fl.vram + (size_t) sl * l.blob, tmp.data(), l.blob,
                                           cudaMemcpyHostToDevice, g0.s));
                        CK(cudaStreamSynchronize(g0.s));
                    }
                }
            }
        }
    }
    CK(cudaStreamSynchronize(g0.s));
    vlend_saved_.clear();
    vlent_ = false;
    return true;
}

// ------------------------------ Persistent KDA + MLA Restore Points ------------------------------

size_t Model::kda_snapshot_bytes() const {
    const Config& c = c_;
    size_t n_kda = 0;
    for (int il = 0; il < c.n_layers; ++il)
        if (c.kda[il]) ++n_kda;
    const size_t st_elems = (size_t) c.n_head * (size_t) c.kda_head * (size_t) c.kda_head;
    const size_t cs_elems = (size_t) 3 * (size_t) c.d_inner() * (size_t) (c.d_conv - 1);
    return n_kda * (st_elems + cs_elems) * sizeof(float);
}

void Model::save_kda_snapshot(KdaSnapshot& snap, const std::vector<int>& tokens,
                              const std::vector<std::pair<int64_t, uint64_t>>& pics, bool include_mla) const {
    const Config& c = c_;
    const size_t st_elems = (size_t) c.n_head * (size_t) c.kda_head * (size_t) c.kda_head;
    const size_t cs_elems = (size_t) 3 * (size_t) c.d_inner() * (size_t) (c.d_conv - 1);
    const size_t lat_row_b = o_.kv_q8 ? (size_t) c.kv_lora + (size_t) (c.kv_lora / 32) * 2
                                      : (size_t) c.kv_lora * sizeof(uint16_t);
    const size_t n_tok = tokens.size();
    snap.tokens = tokens;
    snap.pics = pics;
    snap.rope_delta = rope_delta_;
    snap.kv_q8 = o_.kv_q8;
    snap.states.clear();
    snap.convs.clear();
    snap.mla_lat.clear();
    snap.mla_kpe.clear();
    for (int il = 0; il < c.n_layers; ++il) {
        const Layer& l = layers_[(size_t) il];
        const Gpu& gp = gpus_[(size_t) l.dev_idx];
        cudaSetDevice(gp.dev);
        if (l.kda) {
            std::vector<float> sv(st_elems), cv(cs_elems);
            CK(cudaMemcpyAsync(sv.data(), l.state, st_elems * sizeof(float), cudaMemcpyDeviceToHost, gp.s));
            CK(cudaMemcpyAsync(cv.data(), l.conv_state, cs_elems * sizeof(float), cudaMemcpyDeviceToHost, gp.s));
            snap.states.push_back(std::move(sv));
            snap.convs.push_back(std::move(cv));
        } else if (include_mla && n_tok > 0) {
            std::vector<uint8_t> lv(n_tok * lat_row_b);
            std::vector<uint16_t> kv(n_tok * (size_t) c.n_rot);
            CK(cudaMemcpyAsync(lv.data(), l.lat, lv.size(), cudaMemcpyDeviceToHost, gp.s));
            CK(cudaMemcpyAsync(kv.data(), l.k_pe, kv.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost, gp.s));
            snap.mla_lat.push_back(std::move(lv));
            snap.mla_kpe.push_back(std::move(kv));
        }
    }
    for (const auto& gp : gpus_) {
        cudaSetDevice(gp.dev);
        CK(cudaStreamSynchronize(gp.s));
    }
}

void Model::restore_kda_snapshot(const KdaSnapshot& snap) {
    const Config& c = c_;
    const size_t st_elems = (size_t) c.n_head * (size_t) c.kda_head * (size_t) c.kda_head;
    const size_t cs_elems = (size_t) 3 * (size_t) c.d_inner() * (size_t) (c.d_conv - 1);
    const size_t max_lat_bytes = (size_t) o_.ctx * (size_t) restore_expect_.mla_lat_row_bytes;
    const size_t max_kpe_elems = (size_t) o_.ctx * (size_t) c.n_rot;
    if (snap.kv_q8 != o_.kv_q8 || snap.states.size() != restore_expect_.n_kda_layers ||
        snap.convs.size() != restore_expect_.n_kda_layers) {
        throw std::runtime_error("restore_kda_snapshot: layer or KV format mismatch");
    }
    for (size_t i = 0; i < snap.states.size(); ++i) {
        if (snap.states[i].size() != st_elems || snap.convs[i].size() != cs_elems)
            throw std::runtime_error("restore_kda_snapshot: KDA buffer size mismatch");
    }
    if (!snap.mla_lat.empty()) {
        if (snap.mla_lat.size() != restore_expect_.n_mla_layers || snap.mla_kpe.size() != restore_expect_.n_mla_layers)
            throw std::runtime_error("restore_kda_snapshot: MLA layer count mismatch");
        for (size_t i = 0; i < snap.mla_lat.size(); ++i) {
            if (snap.mla_lat[i].size() > max_lat_bytes || snap.mla_kpe[i].size() > max_kpe_elems)
                throw std::runtime_error("restore_kda_snapshot: MLA buffer exceeds GPU KV allocation");
        }
    }
    size_t ki = 0, mi = 0;
    for (int il = 0; il < c.n_layers; ++il) {
        Layer& l = layers_[(size_t) il];
        Gpu& gp = gpus_[(size_t) l.dev_idx];
        set_dev(gp.dev);
        if (l.kda) {
            CK(cudaMemcpyAsync(l.state, snap.states[ki].data(), st_elems * sizeof(float), cudaMemcpyHostToDevice, gp.s));
            CK(cudaMemcpyAsync(l.conv_state, snap.convs[ki].data(), cs_elems * sizeof(float), cudaMemcpyHostToDevice, gp.s));
            ++ki;
        } else if (mi < snap.mla_lat.size() && !snap.mla_lat[mi].empty()) {
            CK(cudaMemcpyAsync(l.lat, snap.mla_lat[mi].data(), snap.mla_lat[mi].size(), cudaMemcpyHostToDevice, gp.s));
            CK(cudaMemcpyAsync(l.k_pe, snap.mla_kpe[mi].data(), snap.mla_kpe[mi].size() * sizeof(uint16_t),
                               cudaMemcpyHostToDevice, gp.s));
            ++mi;
        }
    }
    for (auto& gp : gpus_) {
        set_dev(gp.dev);
        CK(cudaStreamSynchronize(gp.s));
    }
    rope_delta_ = snap.rope_delta;
    images_.clear();
    mtp_done_ = 0;
    mtp_tokens_ = snap.tokens;
}

bool Model::save_restore_disk(const KdaSnapshot& snap, const std::string& dir, std::string& err) const {
    if (dir.empty() || snap.tokens.empty() || snap.mla_lat.empty()) return false;
    const uint64_t cap_b = (uint64_t) o_.restore_dir_max_mb * (1ULL << 20);
    return sys::save_restore_file(snap, restore_expect_, dir, cap_b, err);
}

bool Model::load_restore_disk(const std::string& dir, std::vector<KdaSnapshot>& out,
                              std::vector<std::string>* skipped_logs) const {
    if (dir.empty()) return false;
    const uint64_t cap_b = (uint64_t) o_.restore_dir_max_mb * (1ULL << 20);
    return sys::load_restore_dir(dir, restore_expect_, cap_b, out, skipped_logs) > 0;
}

}  // namespace ling
