// src/metal/mm3_metal_load.cpp - Mm3Metal: the GGUF set (shards + the indexers' side file), geometry,
// backends, resident weights and the caches.  The file/arena mechanics mirror glm_metal_load.cpp; the
// geometry keys and tensor names are the minimax-m3 GGUF layout (tools/mm3/pack_gguf.py writes them).
#include "mm3_metal_impl.hpp"

#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <regex>
#include <thread>

namespace maya::metal {

namespace {

// split GGUFs: "<stem>-00001-of-00003.gguf" -> every shard; anything else is one file
std::vector<std::string> shard_paths(const std::string& path) {
    static const std::regex re("^(.*)-(\\d{5})-of-(\\d{5})\\.gguf$");
    std::smatch m;
    if (!std::regex_match(path, m, re)) return {path};
    const int n = std::stoi(m[3].str());
    if (n < 1) return {path};
    std::vector<std::string> out;
    for (int i = 1; i <= n; ++i) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "-%05d-of-%05d.gguf", i, n);
        out.push_back(m[1].str() + buf);
    }
    return out;
}

// the set's indexer side file, found the way the real sets name it: <set name>-maya-indexers.bin beside
// the shards, where the set name is a prefix of the shard name but not the whole stem (the shards carry
// a quant tag the side file does not: ...-Maya-XS-IQ1_S-00001-of-00002.gguf next to
// ...-Maya-XS-maya-indexers.bin).  The longest matching prefix in the shards' directory wins.
std::string side_file(const std::string& shard) {
    const size_t slash = shard.find_last_of('/');
    const std::string dir = slash == std::string::npos ? "." : shard.substr(0, slash);
    const std::string base = slash == std::string::npos ? shard : shard.substr(slash + 1);
    for (const char* suf : {"-maya-indexers.bin", "-maya-indexers.gguf"}) {
        std::string best;
        size_t best_stem = 0;
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
            if (!e.is_regular_file()) continue;
            const std::string name = e.path().filename().string();
            const size_t sl = std::strlen(suf);
            if (name.size() <= sl || name.compare(name.size() - sl, std::string::npos, suf) != 0) continue;
            const std::string stem = name.substr(0, name.size() - sl);
            if (base.compare(0, stem.size(), stem) == 0 && stem.size() > best_stem) {
                best = dir + "/" + name;
                best_stem = stem.size();
            }
        }
        if (!best.empty()) return best;
    }
    return {};
}

bool file_exists(const std::string& p) {
    struct stat st {};
    return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

struct Meta {
    const gguf_context* c;

    int64_t key(const std::string& k) const { return gguf_find_key(c, ("minimax-m3." + k).c_str()); }
    static size_t type_size(enum gguf_type t) {
        switch (t) {
            case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: return 1;
            case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: return 2;
            case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: return 8;
            default: return 4;
        }
    }
    static double scalar(const gguf_context* c, int64_t id, size_t i) {
        enum gguf_type t = gguf_get_kv_type(c, id);
        const void* p = nullptr;
        if (t == GGUF_TYPE_ARRAY) {
            t = gguf_get_arr_type(c, id);
            const size_t n = gguf_get_arr_n(c, id);
            if (n == 0 || t == GGUF_TYPE_STRING || t == GGUF_TYPE_ARRAY) return 0;
            if (i >= n) i = n - 1;
            p = (const char*) gguf_get_arr_data(c, id) + i * type_size(t);
        } else {
            p = gguf_get_val_data(c, id);
        }
        switch (t) {
            case GGUF_TYPE_UINT8: return *(const uint8_t*) p;
            case GGUF_TYPE_INT8: return *(const int8_t*) p;
            case GGUF_TYPE_UINT16: return *(const uint16_t*) p;
            case GGUF_TYPE_INT16: return *(const int16_t*) p;
            case GGUF_TYPE_UINT32: return *(const uint32_t*) p;
            case GGUF_TYPE_INT32: return *(const int32_t*) p;
            case GGUF_TYPE_UINT64: return (double) *(const uint64_t*) p;
            case GGUF_TYPE_INT64: return (double) *(const int64_t*) p;
            case GGUF_TYPE_FLOAT32: return *(const float*) p;
            case GGUF_TYPE_FLOAT64: return *(const double*) p;
            case GGUF_TYPE_BOOL: return *(const int8_t*) p ? 1.0 : 0.0;
            default: return 0.0;
        }
    }
    double num(const std::string& k, double def, size_t i = 0) const {
        const int64_t id = key(k);
        return id < 0 ? def : scalar(c, id, i);
    }
};

}  // namespace

Mm3Metal::Impl::~Impl() {
    if (sched) ggml_backend_sched_free(sched);
    if (sbuf) ggml_backend_buffer_free(sbuf);
    if (sctx) ggml_free(sctx);
    if (gpu) ggml_backend_free(gpu);
    if (cpu) ggml_backend_free(cpu);
    for (auto* c : ggufs) gguf_free(c);
    for (auto* m : metas) ggml_free(m);
    for (int fd : fds) close(fd);
}

bool Mm3Metal::Impl::open_files(const std::string& path, std::string& err) {
    std::vector<std::string> paths = shard_paths(path);
    // the dense layers' indexers ride in a side file beside the shards (a GGUF under a .bin name in the
    // current sets; older sets used -maya-indexers.gguf)
    const std::string side = side_file(path);
    if (!side.empty() && std::find(paths.begin(), paths.end(), side) == paths.end()) paths.push_back(side);
    for (const std::string& p : paths) {
        ggml_context* meta = nullptr;
        gguf_init_params ip{true, &meta};
        gguf_context* c = gguf_init_from_file(p.c_str(), ip);
        if (!c) {
            err = "cannot read GGUF " + p;
            return false;
        }
        const int fd = open(p.c_str(), O_RDONLY);
        if (fd < 0) {
            err = "cannot open " + p;
            return false;
        }
        fcntl(fd, F_NOCACHE, 1);
        ggufs.push_back(c);
        metas.push_back(meta);
        fds.push_back(fd);
        const size_t base = gguf_get_data_offset(c);
        for (int64_t i = 0; i < gguf_get_n_tensors(c); ++i) {
            const char* name = gguf_get_tensor_name(c, i);
            loc[name] = {fd, base + gguf_get_tensor_offset(c, i), ggml_get_tensor(meta, name)};
        }
    }
    return true;
}

bool Mm3Metal::Impl::read_geometry(std::string& err) {
    const gguf_context* c = ggufs[0];
    const int64_t a = gguf_find_key(c, "general.architecture");
    const std::string arch = a >= 0 && gguf_get_kv_type(c, a) == GGUF_TYPE_STRING ? gguf_get_val_str(c, a) : "";
    if (arch != "minimax-m3") {
        err = "not a MiniMax-M3 GGUF (architecture '" + arch + "')";
        return false;
    }
    Meta m{c};
    g.n_layer = (int) m.num("block_count", 0);
    g.n_embd = (int) m.num("embedding_length", 0);
    g.n_head = (int) m.num("attention.head_count", 0);
    g.n_kv = (int) m.num("attention.head_count_kv", 0);
    g.head_dim = (int) m.num("attention.key_length", 0);
    g.n_rot = (int) m.num("rope.dimension_count", 0);
    g.rope_theta = (float) m.num("rope.freq_base", 10000000.0);
    g.rms_eps = (float) m.num("attention.layer_norm_rms_epsilon", 1e-6);
    g.n_ff_dense = (int) m.num("feed_forward_length", 0);
    g.n_ff_exp = (int) m.num("expert_feed_forward_length", 0);
    g.n_expert = (int) m.num("expert_count", 0);
    g.n_expert_used = (int) m.num("expert_used_count", 0);
    g.routed_scale = (float) m.num("expert_weights_scale", 1.0);
    g.swiglu_alpha = (float) m.num("swiglu_alpha", 1.702);
    g.swiglu_limit = (float) m.num("swiglu_limit", 7.0);
    g.ctx_train = (int64_t) m.num("context_length", 0);
    g.n_dense_lead = (int) m.num("leading_dense_block_count", 0);
    g.ix_heads = (int) m.num("attention.indexer.head_count", 0);
    g.ix_dim = (int) m.num("attention.indexer.key_length", 0);
    g.ix_topk = (int) m.num("attention.indexer.top_k", 16);
    g.ix_blk = (int) m.num("attention.indexer.block_size", 128);
    g.ix_local = (int) m.num("attention.indexer.local_blocks", 1);
    const auto emb = loc.find("token_embd.weight");
    if (emb == loc.end() || !emb->second.meta) {
        err = "token_embd.weight missing";
        return false;
    }
    g.n_vocab = (int) emb->second.meta->ne[1];
    if (g.n_layer <= 0 || g.n_embd <= 0 || emb->second.meta->ne[0] != g.n_embd || g.n_vocab <= 0) {
        err = "inconsistent geometry (token_embd width, block count or vocabulary)";
        return false;
    }
    if (g.n_kv <= 0 || g.n_head % g.n_kv != 0 || g.head_dim <= 0 || (g.head_dim & 1) || g.n_rot <= 0 ||
        (g.n_rot & 1) || g.n_rot > g.head_dim || g.n_rot > g.ix_dim) {
        err = "unsupported attention geometry (heads, head_dim or rotary width)";
        return false;
    }
    if (g.ix_heads != g.n_kv) {
        err = "the indexer's heads (" + std::to_string(g.ix_heads) + ") are not the KV heads (" +
              std::to_string(g.n_kv) + "): MiniMax's indexer scores one block set per KV group";
        return false;
    }
    if (g.ix_blk < 1 || g.ix_topk < 1 || g.ix_local < 0 || g.ix_dim <= 0) {
        err = "unsupported sparse-attention geometry (block size, top-k, local blocks or index dim)";
        return false;
    }
    if (g.n_dense_lead < 0 || g.n_dense_lead > g.n_layer) {
        err = "leading_dense_block_count outside the layer count";
        return false;
    }
    if (g.n_dense_lead < g.n_layer &&
        (g.n_expert < g.n_expert_used || g.n_expert_used < 1 || g.n_ff_exp <= 0)) {
        err = "inconsistent MoE geometry (expert counts or expert width)";
        return false;
    }
    if (g.n_dense_lead > 0) {
        const auto sh = loc.find("blk." + std::to_string(g.n_dense_lead) + ".ffn_gate_shexp.weight");
        if (g.n_dense_lead < g.n_layer && sh != loc.end() && sh->second.meta)
            g.n_ff_shared = (int) sh->second.meta->ne[1];
    }
    // the serve loop's stop tokens (a scalar or an array in the GGUF's tokenizer keys)
    const int64_t ek = gguf_find_key(c, "tokenizer.ggml.eos_token_id");
    if (ek >= 0) {
        if (gguf_get_kv_type(c, ek) == GGUF_TYPE_ARRAY && gguf_get_arr_type(c, ek) == GGUF_TYPE_UINT32) {
            const uint32_t* v = (const uint32_t*) gguf_get_arr_data(c, ek);
            for (size_t i = 0; i < gguf_get_arr_n(c, ek); ++i) g.eos.push_back((int32_t) v[i]);
        } else if (gguf_get_kv_type(c, ek) == GGUF_TYPE_UINT32) {
            g.eos.push_back((int32_t) gguf_get_val_u32(c, ek));
        }
    }
    return true;
}

bool Mm3Metal::Impl::init_backends(std::string& err) {
    if (o.use_gpu) {
        gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!gpu_dev) gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
        if (!gpu_dev) {
            err = "no Metal GPU found";
            return false;
        }
        const char* any = std::getenv("MAYA_METAL_ALLOW_ANY_GPU");
        if (!(any && *any && *any != '0')) {
            ggml_init_params ip = {3 * ggml_tensor_overhead(), nullptr, true};
            ggml_context* c = ggml_init(ip);
            ggml_tensor* a = ggml_new_tensor_2d(c, GGML_TYPE_F32, 64, 64);
            ggml_tensor* b = ggml_new_tensor_2d(c, GGML_TYPE_F32, 64, 8);
            const bool mm = ggml_backend_dev_supports_op(gpu_dev, ggml_mul_mat(c, a, b));
            ggml_free(c);
            if (!mm) {
                err = std::string("this Mac's GPU (") + ggml_backend_dev_description(gpu_dev) +
                      ") cannot run maya-metal's matrix products: Maya on a Mac needs Apple silicon (M1 or newer)";
                return false;
            }
        }
        gpu = ggml_backend_dev_init(gpu_dev, nullptr);
        if (!gpu) {
            err = "cannot start the Metal backend";
            return false;
        }
    }
    cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) {
        err = "cannot start the CPU backend";
        return false;
    }
    int threads = o.cpu_threads;
    if (threads <= 0) {
        int perf = 0;
        size_t sz = sizeof perf;
        if (sysctlbyname("hw.perflevel0.physicalcpu", &perf, &sz, nullptr, 0) != 0 || perf <= 0)
            perf = (int) std::max(1u, std::thread::hardware_concurrency());
        threads = perf;
    }
    ggml_backend_cpu_set_n_threads(cpu, threads);
    std::vector<ggml_backend_t> be;
    if (gpu) be.push_back(gpu);
    be.push_back(cpu);
    sched = ggml_backend_sched_new(be.data(), nullptr, (int) be.size(), 32768, false, false);
    return sched != nullptr;
}

ggml_tensor* Mm3Metal::Impl::w(const std::string& name) const { return wts.get(name); }

ggml_tensor* Mm3Metal::Impl::wl(int il, const char* suffix) const {
    return w("blk." + std::to_string(il) + "." + suffix);
}

bool Mm3Metal::Impl::load_weights(std::string& err) {
    // everything resident: the dense weights, the embedding and every routed expert (the Maya quants are
    // made to fit a 128 GB Mac whole, so there is no pool and no graph cut at the router)
    std::vector<std::pair<std::string, TensorLoc>> all;
    all.reserve(loc.size());
    for (const auto& [name, l] : loc) all.push_back({name, l});
    std::sort(all.begin(), all.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
    if (!wts.load(all, gpu_dev, 16, err)) return false;
    st.weight_bytes = wts.bytes;

    // every tensor the graphs read must be there: fail at load, not mid-forward
    std::vector<std::string> need = {"token_embd.weight", "output_norm.weight", "output.weight"};
    for (int l = 0; l < g.n_layer; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        for (const char* s : {"attn_norm.weight", "ffn_norm.weight", "attn_q.weight", "attn_k.weight",
                              "attn_v.weight", "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight",
                              "indexer.q_proj.weight", "indexer.k_proj.weight", "indexer.q_norm.weight",
                              "indexer.k_norm.weight"})
            need.push_back(b + s);
        if (l < g.n_dense_lead) {
            for (const char* s : {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"})
                need.push_back(b + s);
        } else {
            for (const char* s : {"ffn_gate_inp.weight", "exp_probs_b.bias", "ffn_gate_shexp.weight",
                                  "ffn_up_shexp.weight", "ffn_down_shexp.weight", "ffn_gate_exps.weight",
                                  "ffn_up_exps.weight", "ffn_down_exps.weight"})
                need.push_back(b + s);
            for (const char* s : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
                const ggml_tensor* t = w(b + s);
                if (t && t->ne[2] != g.n_expert) {
                    err = b + s + " does not hold " + std::to_string(g.n_expert) + " experts";
                    return false;
                }
            }
        }
    }
    for (const std::string& n : need)
        if (!w(n)) {
            err = "the GGUF has no " + n +
                  (n.find("indexer.") != std::string::npos
                       ? " (a sparse layer's indexer: the set's -maya-indexers.bin side file carries the "
                         "dense layers' indexers)"
                       : "");
            return false;
        }
    return true;
}

bool Mm3Metal::Impl::init_state(std::string& err) {
    ggml_init_params ip{ggml_tensor_overhead() * (size_t) (g.n_layer * 3 + 8), nullptr, true};
    sctx = ggml_init(ip);
    ls.assign((size_t) g.n_layer, Mm3LayerState{});
    for (int l = 0; l < g.n_layer; ++l) {
        Mm3LayerState& s = ls[(size_t) l];
        s.kc = ggml_new_tensor_3d(sctx, GGML_TYPE_BF16, g.head_dim, g.n_kv, n_ctx);
        s.vc = ggml_new_tensor_3d(sctx, GGML_TYPE_BF16, g.head_dim, g.n_kv, n_ctx);
        s.ikc = ggml_new_tensor_2d(sctx, GGML_TYPE_BF16, g.ix_dim, n_ctx);
    }
    const int64_t U = o.n_ubatch;
    c_h = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, g.n_embd, U);
    c_q = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, g.head_dim, g.n_head, U);
    c_iq = ggml_new_tensor_3d(sctx, GGML_TYPE_BF16, g.ix_dim, g.ix_heads, U);
    ggml_backend_buffer_type_t bt = gpu_dev ? ggml_backend_dev_buffer_type(gpu_dev) : ggml_backend_cpu_buffer_type();
    sbuf = ggml_backend_alloc_ctx_tensors_from_buft(sctx, bt);
    if (!sbuf) {
        err = "cannot allocate the KV cache and state for " + std::to_string(n_ctx) + " tokens";
        return false;
    }
    st.state_bytes = ggml_backend_buffer_get_size(sbuf);
    return true;
}

}  // namespace maya::metal
