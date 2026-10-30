// src/metal/glm_metal_load.cpp - GlmMetal: GGUF shards, geometry, backends, weights, expert pool, state.
#include "glm_metal_impl.hpp"

#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <fcntl.h>
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <regex>
#include <thread>

namespace maya::metal {

namespace {

constexpr size_t kAlign = 256;               // tensor alignment inside the dense arena
constexpr size_t kArenaChunk = 8ull << 30;   // one GPU buffer per 8 GiB of dense weights

size_t align_up(size_t n, size_t a) { return (n + a - 1) / a * a; }

// split GGUFs: "<stem>-00001-of-00003.gguf" -> every shard; anything else is one file
std::vector<std::string> shard_paths(const std::string& path) {
    static const std::regex re("^(.*)-(\\d{5})-of-(\\d{5})\\.gguf$");
    std::smatch m;
    if (!std::regex_match(path, m, re)) return {path};
    const int n = std::stoi(m[3].str());
    if (n < 1) return {path};   // "-of-00000": not a split set (open() then names the file)
    std::vector<std::string> out;
    for (int i = 1; i <= n; ++i) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "-%05d-of-%05d.gguf", i, n);
        out.push_back(m[1].str() + buf);
    }
    return out;
}

struct Meta {
    const gguf_context* c;
    std::string arch;

    int64_t key(const std::string& k) const {
        for (const char* pre : {"glm5-next.", "glm5next."}) {
            const int64_t id = gguf_find_key(c, (pre + k).c_str());
            if (id >= 0) return id;
        }
        return -1;
    }
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
    bool has(const std::string& k) const { return key(k) >= 0; }
};

}  // namespace

// what the Mac could give this process now: free, inactive, purgeable and speculative pages (the dense weights and the
// state are already allocated, so they are not in it)
size_t free_memory() {
    vm_statistics64_data_t vs{};
    mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vs, &n) != KERN_SUCCESS) return SIZE_MAX;
    const uint64_t pages = (uint64_t) vs.free_count + vs.inactive_count + vs.purgeable_count + vs.speculative_count;
    return (size_t) (pages * (uint64_t) getpagesize());
}

GlmMetal::Impl::~Impl() {
    if (sched) ggml_backend_sched_free(sched);
    if (sbuf) ggml_backend_buffer_free(sbuf);
    if (sctx) ggml_free(sctx);
    if (gpu) ggml_backend_free(gpu);
    if (cpu) ggml_backend_free(cpu);
    for (auto* c : ggufs) gguf_free(c);
    for (auto* m : metas) ggml_free(m);
    for (int fd : fds) close(fd);
}

bool GlmMetal::Impl::open_files(const std::string& path, std::string& err) {
    for (const std::string& p : shard_paths(path)) {
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
        fcntl(fd, F_NOCACHE, 1);   // the pool is the cache; the page cache would hold a second copy
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

bool GlmMetal::Impl::read_geometry(std::string& err) {
    const gguf_context* c = ggufs[0];
    const int64_t a = gguf_find_key(c, "general.architecture");
    const std::string arch = a >= 0 && gguf_get_kv_type(c, a) == GGUF_TYPE_STRING ? gguf_get_val_str(c, a) : "";
    if (arch != "glm5-next" && arch != "glm5next") {
        err = "not a GLM-5.3-Flash GGUF (architecture '" + arch + "')";
        return false;
    }
    Meta m{c, arch};
    const int blocks = (int) m.num("block_count", 45);
    g.n_layers = blocks - (int) m.num("nextn_predict_layers", 0);
    g.n_embd = (int) m.num("embedding_length", 4096);
    g.n_head = (int) m.num("attention.head_count", 64);
    g.hc = (int) m.num("hyper_connection.count", 4);
    g.hc_iters = (int) m.num("hyper_connection.sinkhorn_iterations", 20);
    g.hc_eps = (float) m.num("hyper_connection.epsilon", 1e-6);
    g.rms_eps = (float) m.num("attention.layer_norm_rms_epsilon", 1e-5);
    g.norm_eps = (float) m.num("attention.layer_norm_epsilon", 1e-5);
    g.kda_head_dim = (int) m.num("kda.head_dim", 128);
    g.d_conv = (int) m.num("ssm.conv_kernel", 4);
    g.kda_lower_bound = (float) m.num("kda.gate_lower_bound", -5.0);
    g.q_lora = (int) m.num("attention.q_lora_rank", 1536);
    g.kv_lora = (int) m.num("attention.kv_lora_rank", 512);
    g.head_k_mla = (int) m.num("attention.key_length_mla", 256);
    g.head_v_mla = (int) m.num("attention.value_length_mla", 256);
    g.idx_heads = (int) m.num("attention.indexer.head_count", 32);
    g.idx_key = (int) m.num("attention.indexer.key_length", 128);
    g.idx_top_k = (int) m.num("attention.indexer.top_k", 2048);
    g.idx_kpool = (int) m.num("attention.indexer.kpool", 4);
    g.idx_select_tail = m.num("attention.indexer.kpool_select_tail", 0) != 0;
    g.n_ff = (int) m.num("feed_forward_length", 12288);
    g.n_expert = (int) m.num("expert_count", 288);
    g.n_expert_used = (int) m.num("expert_used_count", 8);
    g.n_ff_exp = (int) m.num("expert_feed_forward_length", 2048);
    g.n_dense_lead = (int) m.num("leading_dense_block_count", 0);
    g.expert_weights_scale = (float) m.num("expert_weights_scale", 1.0);
    g.expert_weights_norm = m.num("expert_weights_norm", 1) != 0;
    if (m.num("rope.dimension_count", 0) != 0) {
        err = "GLM-5.3-Flash's MLA is nope-only; this GGUF has rope dimensions";
        return false;
    }
    if (m.num("expert_gating_func", 2) != 2) {
        err = "only sigmoid expert gating is implemented";
        return false;
    }
    if (g.hc != 4 || g.idx_kpool < 2 || g.idx_top_k % g.idx_kpool != 0) {
        err = "unsupported mHC / indexer geometry";
        return false;
    }
    const auto emb = loc.find("token_embd.weight");
    if (emb == loc.end() || !emb->second.meta || emb->second.meta->ne[0] != g.n_embd || g.n_layers <= 0 ||
        g.n_dense_lead > g.n_layers || g.n_expert_used <= 0 || g.n_expert_used > g.n_expert) {
        err = "inconsistent geometry (token_embd width, block, leading dense or expert counts)";
        return false;
    }
    g.mtp = o.mtp && blocks > g.n_layers;
    const int held = g.n_layers + (g.mtp ? 1 : 0);
    g.recurrent.assign((size_t) held, 0);
    g.clamp_exp.assign((size_t) held, 0.0f);
    g.clamp_shexp.assign((size_t) held, 0.0f);
    for (int l = 0; l < held; ++l) {
        g.recurrent[(size_t) l] = m.num("attention.head_count_kv", 0, (size_t) l) == 0 ? 1 : 0;
        if (m.has("attention.indexer.types") && m.num("attention.indexer.types", 1, (size_t) l) == 0) {
            err = "shared-indexer (non-full) DSA layers are not used by GLM-5.3-Flash and not implemented";
            return false;
        }
        if (m.num("expert_feed_forward_length", g.n_ff_exp, (size_t) l) != g.n_ff_exp) {
            err = "per-layer expert widths are not implemented";
            return false;
        }
        g.clamp_exp[(size_t) l] = (float) m.num("swiglu_clamp_exp", 0.0, (size_t) l);
        g.clamp_shexp[(size_t) l] =
            (float) (m.has("swiglu_clamp_shexp") ? m.num("swiglu_clamp_shexp", 0.0, (size_t) l)
                                                 : g.clamp_exp[(size_t) l]);
    }
    const auto ev = loc.find("token_embd.weight");
    if (ev == loc.end()) {
        err = "token_embd.weight missing";
        return false;
    }
    embd = ev->second;
    g.n_vocab = (int) embd.meta->ne[1];
    return true;
}

bool GlmMetal::Impl::init_backends(std::string& err) {
    if (o.use_gpu) {
        gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!gpu_dev) gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
        if (!gpu_dev) {
            err = "no Metal GPU found";
            return false;
        }
        // a GPU without Apple7-class simdgroup support (an Intel Mac's GPU, a virtual Mac's) runs no matrix product in
        // Metal: the graphs would fall back to the CPU and stop at the first one whose weights sit in GPU-only memory
        // (a ggml assert) - say so up front instead.  MAYA_METAL_ALLOW_ANY_GPU=1 starts anyway (CI's virtual Macs).
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
    sched = ggml_backend_sched_new(be.data(), nullptr, (int) be.size(),
                                   graph_size(std::max(o.n_ubatch, o.max_window)), false, false);
    return sched != nullptr;
}

ggml_tensor* GlmMetal::Impl::w(const std::string& name) const {
    return wts.get(name);
}

ggml_tensor* GlmMetal::Impl::wl(int il, const char* suffix) const {
    return w("blk." + std::to_string(il) + "." + suffix);
}

Weights::~Weights() {
    for (auto* b : bufs) ggml_backend_buffer_free(b);
    for (auto& m : mem) host_free(m.first, m.second);
    if (ctx) ggml_free(ctx);
}

ggml_tensor* Weights::get(const std::string& name) const {
    const auto it = by_name.find(name);
    return it == by_name.end() ? nullptr : it->second;
}

bool Weights::load(const std::vector<std::pair<std::string, TensorLoc>>& tensors, ggml_backend_dev_t dev,
                   int io_threads, std::string& err) {
    ggml_init_params ip{ggml_tensor_overhead() * (tensors.size() + 8), nullptr, true};
    ctx = ggml_init(ip);
    ParallelReader reader(io_threads);
    size_t i = 0;
    while (i < tensors.size()) {
        size_t chunk = 0, j = i;
        while (j < tensors.size() &&
               (chunk == 0 || chunk + align_up(ggml_nbytes(tensors[j].second.meta), kAlign) <= kArenaChunk))
            chunk += align_up(ggml_nbytes(tensors[j++].second.meta), kAlign);
        chunk = align_up(chunk, (size_t) getpagesize());
        void* m = host_alloc(chunk);
        if (!m) {
            err = "cannot allocate " + std::to_string(chunk >> 20) + " MiB for weights";
            return false;
        }
        mem.push_back({m, chunk});
        host_pin(m, chunk);
        ggml_backend_buffer_t buf = dev ? ggml_backend_dev_buffer_from_host_ptr(dev, m, chunk, chunk)
                                        : ggml_backend_cpu_buffer_from_ptr(m, chunk);
        if (!buf) {
            err = "the GPU could not map the weights";
            return false;
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        bufs.push_back(buf);
        std::vector<ReadJob> jobs;
        size_t at = 0;
        for (; i < j; ++i) {
            const auto& [name, l] = tensors[i];
            ggml_tensor* t = ggml_dup_tensor(ctx, l.meta);
            ggml_set_name(t, name.c_str());
            if (ggml_backend_tensor_alloc(buf, t, (char*) m + at) != GGML_STATUS_SUCCESS) {
                err = "cannot place " + name;
                return false;
            }
            jobs.push_back({l.fd, l.offset, (char*) m + at, ggml_nbytes(t)});
            by_name[name] = t;
            at += align_up(ggml_nbytes(t), kAlign);
        }
        if (!reader.read(jobs, err)) return false;
        bytes += chunk;
    }
    return true;
}

bool GlmMetal::Impl::load_weights(std::string& err) {
    // every trunk tensor except the token embedding (CPU rows) and the routed experts (the pool)
    std::vector<std::pair<std::string, TensorLoc>> dense;
    for (const auto& [name, l] : loc) {
        if (name == "token_embd.weight" || name.find("_exps.") != std::string::npos) continue;
        if (name.rfind("blk.", 0) == 0 && std::strtol(name.c_str() + 4, nullptr, 10) >= g.n_layers + (g.mtp ? 1 : 0))
            continue;
        dense.push_back({name, l});
    }
    std::sort(dense.begin(), dense.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
    if (!wts.load(dense, gpu_dev, o.io_threads, err)) return false;
    st.dense_bytes += wts.bytes;

    // every tensor the graph reads must be there: fail at load, not mid-forward
    std::vector<std::string> need = {"output_norm.weight", "output.weight"};
    for (int l = 0; l < g.n_layers; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        for (const char* s : {"attn_norm.weight", "ffn_norm.weight", "hc_attn_fn.weight", "hc_attn_base.weight",
                              "hc_attn_scale.weight", "hc_ffn_fn.weight", "hc_ffn_base.weight", "hc_ffn_scale.weight"})
            need.push_back(b + s);
        if (g.recurrent[(size_t) l]) {
            for (const char* s : {"attn_q.weight", "attn_k.weight", "attn_v.weight", "ssm_conv1d_q.weight",
                                  "ssm_conv1d_k.weight", "ssm_conv1d_v.weight", "ssm_f_a.weight", "ssm_f_b.weight",
                                  "ssm_beta.weight", "ssm_a", "ssm_dt.bias", "ssm_g_a.weight", "ssm_g_b.weight",
                                  "ssm_norm.weight", "attn_output.weight"})
                need.push_back(b + s);
        } else {
            for (const char* s : {"attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight",
                                  "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight", "attn_k_b.weight",
                                  "attn_v_b.weight", "attn_output.weight", "indexer.attn_k.weight",
                                  "indexer.k_norm.weight", "indexer.k_norm.bias", "indexer.attn_q_b.weight",
                                  "indexer.proj.weight", "indexer_compressor_gate.weight",
                                  "indexer_compressor_ape.weight"})
                need.push_back(b + s);
        }
        if (l < g.n_dense_lead) {
            for (const char* s : {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}) need.push_back(b + s);
        } else {
            for (const char* s : {"ffn_gate_inp.weight", "exp_probs_b.bias", "ffn_gate_shexp.weight",
                                  "ffn_up_shexp.weight", "ffn_down_shexp.weight"})
                need.push_back(b + s);
        }
    }
    if (g.mtp) {   // the NextN block: a DSA layer with its own MoE, fed by the trunk's hidden state
        const std::string b = "blk." + std::to_string(g.n_layers) + ".";
        for (const char* s : {"nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight", "attn_norm.weight",
                              "ffn_norm.weight", "attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight",
                              "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight", "attn_k_b.weight", "attn_v_b.weight",
                              "attn_output.weight", "indexer.attn_k.weight", "indexer.k_norm.weight",
                              "indexer.k_norm.bias", "indexer.attn_q_b.weight", "indexer.proj.weight",
                              "indexer_compressor_gate.weight", "indexer_compressor_ape.weight", "ffn_gate_inp.weight",
                              "exp_probs_b.bias", "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"})
            need.push_back(b + s);
        if (g.recurrent[(size_t) g.n_layers]) {
            err = "the NextN block is not an attention layer";
            return false;
        }
    }
    for (const std::string& n : need)
        if (!w(n)) {
            err = "the GGUF has no " + n;
            return false;
        }
    return true;
}

bool GlmMetal::Impl::init_experts(std::string& err) {
    const int held = g.n_layers + (g.mtp ? 1 : 0);
    std::vector<std::vector<ExpertMatrix>> layers((size_t) held);
    size_t total = 0;
    for (int l = g.n_dense_lead; l < held; ++l) {
        for (const char* role : {"gate", "up", "down"}) {
            const std::string name = "blk." + std::to_string(l) + ".ffn_" + role + "_exps.weight";
            const auto it = loc.find(name);
            if (it == loc.end()) {
                err = "the GGUF has no " + name;
                return false;
            }
            const ggml_tensor* t = it->second.meta;
            if (t->ne[2] != g.n_expert) {
                err = name + " does not hold " + std::to_string(g.n_expert) + " experts";
                return false;
            }
            layers[(size_t) l].push_back({it->second.fd, it->second.offset, (size_t) t->nb[2], (int) t->type,
                                          t->ne[0], t->ne[1]});
            total += ggml_nbytes(t);
        }
    }
    size_t budget = o.expert_budget, cap = budget;
    if (budget == 0) {
        // what the GPU may keep resident, less the dense weights, the state and a margin for activations
        size_t working = gpu_dev ? 0 : (size_t) 48 << 30;
        if (gpu_dev) {
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(gpu_dev, &props);
            working = props.memory_total;
        }
        const size_t reserve = st.dense_bytes + st.state_bytes + ((size_t) 6 << 30);
        budget = working > reserve ? working - reserve : (size_t) 1 << 30;
        cap = budget;   // what the pool may grow to as memory frees up (GlmMetal::grow_pool)
        // and no more than this Mac has free now, less a headroom (MAYA_GLM_RAM_HEADROOM_GB, default 6, as for the CUDA engine's RAM tier): unified memory
        // is shared with every other program, and a pool past what is free gets compressed or swapped - a graph that
        // reads it then stalls for seconds (measured: 20-57 s) instead of 10 ms
        const char* hv = std::getenv("MAYA_GLM_RAM_HEADROOM_GB");
        const double head_gb = hv && *hv ? std::atof(hv) : 6.0;
        const size_t avail = free_memory();
        const size_t head = (size_t) (head_gb * (double) (1ull << 30));
        const size_t fits = avail > head ? avail - head : 0;
        if (fits < budget) {
            std::fprintf(stderr, "maya-metal: %.1f GB free now (headroom %.0f GB): the expert pool takes %.1f GB, not %.1f\n",
                         avail / 1e9, head_gb, fits / 1e9, budget / 1e9);
            budget = std::max(fits, (size_t) 1 << 30);
        }
    }
    const int min_slots = g.n_expert + g.n_expert_used;   // one slice may touch every expert of a layer
    if (!pool.init(layers, g.n_expert, min_slots, std::min(budget, total), std::min(cap, total), o.pool_slots, gpu_dev,
                   o.io_threads,
                   err))
        return false;
    // the learned profile when there is one, else the shipped one: the pool starts with their first experts
    for (const std::string& path : {o.expert_profile_save, o.expert_profile}) {
        if (path.empty() || !ranked.empty()) continue;
        // STRP v1 (tools/make_profile.py): magic, version, layers, experts, slots, n, then n (u16 layer, u16 expert)
        if (FILE* f = std::fopen(path.c_str(), "rb")) {
            char magic[4] = {};
            uint32_t h[5] = {};
            if (std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "STRP", 4) == 0 && std::fread(h, 4, 5, f) == 5 &&
                h[0] == 1 && (int) h[2] == g.n_expert && h[4] <= (uint32_t) (h[1] * h[2])) {
                std::vector<uint16_t> pairs((size_t) h[4] * 2);
                if (std::fread(pairs.data(), 2, pairs.size(), f) == pairs.size())
                    for (size_t i = 0; i < pairs.size(); i += 2) ranked.push_back({pairs[i], pairs[i + 1]});
            }
            std::fclose(f);
            if (ranked.empty())
                std::fprintf(stderr, "maya-metal: %s is not a GLM expert profile, skipped\n", path.c_str());
        }
    }
    if (!ranked.empty() && !pool.all_resident() && !pool.warm(ranked, err)) return false;
    st.pool_slots = pool.slots_total();
    st.pool_bytes = pool.bytes();
    st.experts_all_resident = pool.all_resident();
    return true;
}

bool GlmMetal::Impl::init_state(std::string& err) {
    const int kp = g.idx_kpool;
    const int64_t n_pools = n_ctx / kp + 1;
    ggml_init_params ip{ggml_tensor_overhead() * (size_t) ((g.n_layers + 1) * 9 + 16), nullptr, true};
    sctx = ggml_init(ip);
    const int held = g.n_layers + (g.mtp ? 1 : 0);
    ls.assign((size_t) held, LayerState{});
    const int64_t d_inner = (int64_t) g.kda_head_dim * g.n_head;
    for (int l = 0; l < held; ++l) {
        LayerState& s = ls[(size_t) l];
        if (g.recurrent[(size_t) l]) {
            s.conv = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, g.d_conv - 1, d_inner, 3);
            s.ssm = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, g.kda_head_dim, g.kda_head_dim, g.n_head);
            s.conv_snap = ggml_dup_tensor(sctx, s.conv);
            s.ssm_snap = ggml_dup_tensor(sctx, s.ssm);
            if (o.max_window > 0) {
                s.conv_win = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, g.d_conv - 1 + o.max_window, d_inner, 3);
                s.ssm_win = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, (int64_t) g.kda_head_dim * g.kda_head_dim * g.n_head,
                                               o.max_window);
            }
        } else {
            // dsa_mm on the GPU rounds the gathered latent to half in mul_mm's tiles anyway: keeping it in F16 loses
            // nothing more and halves the gather and its transpose (the CPU oracle keeps F32)
            const bool lat16 = o.kv_f16 || (o.dsa_mm && gpu_dev && !o.kv_f32);
            s.lat = ggml_new_tensor_2d(sctx, lat16 ? GGML_TYPE_F16 : GGML_TYPE_F32, g.kv_lora, n_ctx + 1);
            s.ikg = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, 2 * g.idx_key, n_ctx);
            s.pooled = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, g.idx_key, n_pools);
        }
    }
    pool_tab = ggml_new_tensor_2d(sctx, GGML_TYPE_I32, kp, n_pools);
    const int64_t U = o.n_ubatch;
    c_res = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, g.n_embd, g.hc, U);
    c_post = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, g.hc, U);
    c_comb = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, g.hc, g.hc, U);
    c_x = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, g.n_embd, U);
    c_w = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, 1, g.n_expert_used, U);
    c_shexp = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, g.n_embd, U);
    hid = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, g.n_embd, std::max(o.max_window, 1));
    c_mres = ggml_new_tensor_1d(sctx, GGML_TYPE_F32, g.n_embd);
    ggml_backend_buffer_type_t bt =
        gpu_dev ? ggml_backend_dev_buffer_type(gpu_dev) : ggml_backend_cpu_buffer_type();
    sbuf = ggml_backend_alloc_ctx_tensors_from_buft(sctx, bt);
    if (!sbuf) {
        err = "cannot allocate the KV cache and state for " + std::to_string(n_ctx) + " tokens";
        return false;
    }
    st.state_bytes = ggml_backend_buffer_get_size(sbuf);
    std::vector<int32_t> tab((size_t) (kp * n_pools));
    for (size_t i = 0; i < tab.size(); ++i) tab[i] = (int32_t) i;   // pool p = cells p*kp .. p*kp+kp-1
    ggml_backend_tensor_set(pool_tab, tab.data(), 0, tab.size() * sizeof(int32_t));
    return true;
}

bool GlmMetal::Impl::embed(const int32_t* tok, int n, std::vector<float>& out, std::string& err) const {
    if (!embed_raw(tok, n, out, err)) return false;
    for (size_t k = 0; k < img_pos.size(); ++k)   // an image's rows in place of its <|image|> tokens' embeddings
        if (img_pos[k] >= pos && img_pos[k] < pos + n)
            std::memcpy(out.data() + (size_t) (img_pos[k] - pos) * g.n_embd, img_rows.data() + k * (size_t) g.n_embd,
                        sizeof(float) * (size_t) g.n_embd);
    return true;
}

bool GlmMetal::Impl::embed_raw(const int32_t* tok, int n, std::vector<float>& out, std::string& err) const {
    const ggml_tensor* t = embd.meta;
    const size_t row = (size_t) t->nb[1];
    const auto* tr = ggml_get_type_traits(t->type);
    std::vector<uint8_t> buf(row);
    out.resize((size_t) n * g.n_embd);
    for (int i = 0; i < n; ++i) {
        if (tok[i] < 0 || tok[i] >= g.n_vocab) {
            err = "token id " + std::to_string(tok[i]) + " is outside the vocabulary";
            return false;
        }
        if (pread(embd.fd, buf.data(), row, (off_t) (embd.offset + (uint64_t) tok[i] * row)) != (ssize_t) row) {
            err = "cannot read the token embedding";
            return false;
        }
        if (t->type == GGML_TYPE_F32)
            std::memcpy(out.data() + (size_t) i * g.n_embd, buf.data(), row);
        else
            tr->to_float(buf.data(), out.data() + (size_t) i * g.n_embd, g.n_embd);
    }
    return true;
}

}  // namespace maya::metal
