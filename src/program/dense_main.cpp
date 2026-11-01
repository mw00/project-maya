// src/program/dense_main.cpp - `strata-dense --serve --native <model.gguf> --context N`
//
// The dense-model twin of `strata --serve`.  It speaks the SAME line protocol on stdin/stdout, so serve/server.py
// drives it through its unchanged StrataEngine (tokenizer, chat template, thinking levels, tools, OpenAI and
// Anthropic endpoints, web app, Monitor):
//
//   out   INFO key=value ...                      (optional facts for the Monitor)
//   out   READY <context> stop                    ("stop": this engine honours STOP)
//   in    GEN <max_new> [key=value ...] id,id,... (keys: temperature top_p top_k min_p penalty_last_n penalty_repeat
//                                                  penalty_freq penalty_present seed; unknown keys are skipped)
//   out   PP <position> <prompt tokens> <ms> <tok/s>          every few dozen prompt tokens
//   out   T <id>                                              one per generated token
//   out   DONE <generated> <prompt> <prompt ms> <decode ms> <stop|length|cancel> 0 0 <reused>
//   in    STOP | QUIT
//
// Conversation cache: the recurrent GDN state cannot be rolled back, so a request reuses the state only when
// EVERYTHING the state has consumed is a prefix of the new prompt (a chat that only appends).  Otherwise the
// state is reset and the prompt is read again.
//
// The prompt is read one token at a time (the same step as decoding).  Correct, and the simplest thing that can
// be checked against llama.cpp; a batched prefill is the next optimisation, not a prerequisite.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/safetensors_dir.hpp"
#include "strata/core/dense_model.hpp"
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/dense_kernels.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/q35_kernels.hpp"
#include "strata/kernels/q35_dflash.hpp"
#include "strata/spec/suffix_drafter.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <climits>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
// The sampler's chain (sampler.cu: top_k -> top_p on the untempered softmax -> min_p -> temperature) over a row's top
// k (ids / logits, descending): the kept ids and their probabilities - the distribution sample_tokens draws from.
struct SparseDist {
    int n = 0;
    int ids[32];
    double p[32];
    double of(int id) const { for (int i = 0; i < n; ++i) if (ids[i] == id) return p[i]; return 0.0; }
};
SparseDist sparse_dist(const int* ids, const float* lg, int k, const strata::kernels::SamplerParams& sp) {
    SparseDist d;
    int n_keep = k;
    if (sp.top_p < 1.0f) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) sum += std::exp((double) lg[i] - (double) lg[0]);
        double cum = 0.0;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cum += std::exp((double) lg[i] - (double) lg[0]) / sum;
            if (cum >= (double) sp.top_p) { cut = i + 1; break; }
        }
        if (cut < sp.min_keep) cut = std::min(sp.min_keep, k);
        n_keep = cut;
    }
    if (sp.min_p > 0.0f) {
        const float thresh = lg[0] + std::log(sp.min_p);
        for (int i = 0; i < n_keep; ++i) if (lg[i] < thresh) { n_keep = i; break; }
    }
    const double inv_t = 1.0 / sp.temperature;
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i) sum += std::exp(((double) lg[i] - (double) lg[0]) * inv_t);
    d.n = n_keep;
    for (int i = 0; i < n_keep; ++i) {
        d.ids[i] = ids[i];
        d.p[i] = std::exp(((double) lg[i] - (double) lg[0]) * inv_t) / sum;
    }
    return d;
}
int draw(const SparseDist& d, std::mt19937_64& rng) {
    double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng), cum = 0.0;
    for (int i = 0; i < d.n; ++i) { cum += d.p[i]; if (u < cum) return d.ids[i]; }
    return d.ids[d.n - 1];
}

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct Lines {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::string> q;
    std::atomic<bool> stop{false};
    bool eof = false;
    void push(std::string s) {
        std::lock_guard<std::mutex> l(m);
        q.push_back(std::move(s));
        cv.notify_one();
    }
    bool pop(std::string& out) {
        std::unique_lock<std::mutex> l(m);
        cv.wait(l, [&] { return !q.empty() || eof; });
        if (q.empty()) return false;
        out = std::move(q.front());
        q.pop_front();
        return true;
    }
};

bool parse_ids(const char* p, std::vector<int32_t>& ids) {
    ids.clear();
    while (*p == ' ') ++p;
    while (*p) {
        char* end = nullptr;
        const long long v = std::strtoll(p, &end, 10);
        if (end == p || v < 0 || v > INT32_MAX) return false;
        ids.push_back((int32_t) v);
        p = end;
        if (*p == ',') ++p;
        else if (*p != '\0' && *p != '\r' && *p != '\n') return false;
        else break;
    }
    return !ids.empty();
}

void usage() {
    std::fprintf(stderr,
                 "usage: strata-dense --serve --native <model.gguf> [--context N]\n"
                 "  a dense qwen35 GGUF (Qwen3.8-27B) behind Strata's server protocol; the server runs it with\n"
                 "  `serve/server.py --engine strata --config <run config>` whose `exe` is this program.\n");
}

// ---- `strata-dense --selftest`: the new CUDA kernels against a CPU reference, no model needed.
uint16_t f32_to_f16_bits(float f) {          // round to nearest even; the test inputs stay well inside the range
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) return (uint16_t) sign;     // flush subnormals: the test data never reaches them
    if (e >= 31) return (uint16_t) (sign | 0x7bffu);
    uint32_t h = sign | ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rest = m & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) h;
}

struct Diff { double abs = 0, rel = 0; };
Diff compare(const std::vector<float>& a, const std::vector<float>& ref) {
    Diff d;
    for (size_t i = 0; i < a.size(); ++i) {
        const double e = std::fabs((double) a[i] - (double) ref[i]);
        d.abs = std::max(d.abs, e);
        d.rel = std::max(d.rel, e / (std::fabs((double) ref[i]) + 1e-3));
    }
    return d;
}

template <class T> T* to_dev(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc((void**) &d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
std::vector<float> from_dev(const float* d, size_t n) {
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost);
    return h;
}

int selftest() {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto rnd = [&](size_t n, float sc = 1.0f) { std::vector<float> v(n); for (auto& x : v) x = nd(rng) * sc; return v; };
    int bad = 0;
    auto report = [&](const char* what, const Diff& d, double tol) {
        const bool ok = d.abs < tol;
        std::printf("%-34s max abs err %.3g  max rel %.3g  %s\n", what, d.abs, d.rel, ok ? "ok" : "FAIL");
        if (!ok) ++bad;
    };
    try {
        {   // RMSNorm
            const int rows = 3, cols = 5120;
            auto x = rnd((size_t) rows * cols, 3.0f), w = rnd(cols, 1.0f);
            std::vector<float> ref(x.size());
            for (int r = 0; r < rows; ++r) {
                double ss = 0; for (int i = 0; i < cols; ++i) ss += (double) x[(size_t) r * cols + i] * x[(size_t) r * cols + i];
                const double sc = 1.0 / std::sqrt(ss / cols + 1e-6);
                for (int i = 0; i < cols; ++i) ref[(size_t) r * cols + i] = (float) (x[(size_t) r * cols + i] * sc * w[i]);
            }
            float *dx = to_dev(x), *dw = to_dev(w), *dout = nullptr;
            cudaMalloc((void**) &dout, x.size() * 4);
            strata::kernels::dense_rms_norm(dx, dw, dout, rows, cols, 1e-6f, nullptr);
            report("rms_norm 3 x 5120", compare(from_dev(dout, x.size()), ref), 1e-4);
        }
        {   // SwiGLU
            const int n = 17408;
            auto g = rnd(n, 2.0f), u = rnd(n, 2.0f);
            std::vector<float> ref(n);
            for (int i = 0; i < n; ++i) ref[(size_t) i] = (float) (g[(size_t) i] / (1.0 + std::exp(-(double) g[(size_t) i])) * u[(size_t) i]);
            float *dg = to_dev(g), *du = to_dev(u), *dout = nullptr;
            cudaMalloc((void**) &dout, n * 4);
            strata::kernels::dense_swiglu(dg, du, dout, n, nullptr);
            report("swiglu 17408", compare(from_dev(dout, n), ref), 1e-4);
        }
        {   // F32 GEMV
            const int n_in = 5120, n_out = 48;
            auto W = rnd((size_t) n_in * n_out, 0.05f), x = rnd(n_in);
            std::vector<float> ref(n_out);
            for (int r = 0; r < n_out; ++r) { double a = 0; for (int i = 0; i < n_in; ++i) a += (double) W[(size_t) r * n_in + i] * x[(size_t) i]; ref[(size_t) r] = (float) a; }
            float *dW = to_dev(W), *dx = to_dev(x), *dy = nullptr;
            cudaMalloc((void**) &dy, n_out * 4);
            strata::kernels::dense_gemv_f32(dW, dx, dy, n_in, n_out, nullptr);
            report("gemv_f32 5120 -> 48", compare(from_dev(dy, n_out), ref), 1e-3);
        }
        for (int n_ctx : {1, 40, 256, 257, 700}) {   // attention, including several splits and a partial last one
            const int H = 24, HK = 4, D = 256, max_ctx = 1024;
            auto q = rnd((size_t) H * D), k = rnd((size_t) HK * max_ctx * D), v = rnd((size_t) HK * max_ctx * D);
            std::vector<uint16_t> kh(k.size()), vh(v.size());
            std::vector<float> kr(k.size()), vr(v.size());
            for (size_t i = 0; i < k.size(); ++i) {
                kh[i] = f32_to_f16_bits(k[i]); vh[i] = f32_to_f16_bits(v[i]);
                kr[i] = strata::fp16_to_fp32(kh[i]); vr[i] = strata::fp16_to_fp32(vh[i]);
            }
            const float scale = 1.0f / std::sqrt((float) D);
            std::vector<float> ref((size_t) H * D);
            for (int h = 0; h < H; ++h) {
                const int kvh = h / (H / HK);
                std::vector<double> sc((size_t) n_ctx);
                double mx = -1e300;
                for (int t = 0; t < n_ctx; ++t) {
                    double d = 0;
                    for (int i = 0; i < D; ++i) d += (double) q[(size_t) h * D + i] * kr[((size_t) kvh * max_ctx + t) * D + i];
                    sc[(size_t) t] = d * scale; mx = std::max(mx, sc[(size_t) t]);
                }
                double den = 0; for (auto& x : sc) { x = std::exp(x - mx); den += x; }
                for (int i = 0; i < D; ++i) {
                    double a = 0;
                    for (int t = 0; t < n_ctx; ++t) a += sc[(size_t) t] * vr[((size_t) kvh * max_ctx + t) * D + i];
                    ref[(size_t) h * D + i] = (float) (a / den);
                }
            }
            float* dq = to_dev(q);
            uint16_t *dk = to_dev(kh), *dv = to_dev(vh);
            float *dout = nullptr, *dscr = nullptr;
            cudaMalloc((void**) &dout, ref.size() * 4);
            cudaMalloc((void**) &dscr, strata::kernels::dense_attn_scratch_bytes(H, D, max_ctx));
            strata::kernels::dense_attn_decode(dq, dk, dv, dout, dscr, H, HK, D, n_ctx, max_ctx, scale, nullptr);
            char name[64];
            std::snprintf(name, sizeof name, "attention 24/4 x 256, n_ctx %d", n_ctx);
            report(name, compare(from_dev(dout, ref.size()), ref), 2e-3);
        }
        {   // KV append: the cell the attention will read
            const int HK = 4, D = 256, max_ctx = 64, pos = 9;
            auto k = rnd((size_t) HK * D), v = rnd((size_t) HK * D);
            std::vector<uint16_t> zero((size_t) HK * max_ctx * D, 0);
            uint16_t *dk = to_dev(zero), *dv = to_dev(zero);
            float *dkc = to_dev(k), *dvc = to_dev(v);
            strata::kernels::dense_kv_append(dk, dv, dkc, dvc, pos, HK, D, max_ctx, nullptr);
            std::vector<uint16_t> back(zero.size());
            cudaMemcpy(back.data(), dk, back.size() * 2, cudaMemcpyDeviceToHost);
            std::vector<float> got((size_t) HK * D), ref((size_t) HK * D);
            for (int h = 0; h < HK; ++h) for (int i = 0; i < D; ++i) {
                got[(size_t) h * D + i] = strata::fp16_to_fp32(back[((size_t) h * max_ctx + pos) * D + i]);
                ref[(size_t) h * D + i] = strata::fp16_to_fp32(f32_to_f16_bits(k[(size_t) h * D + i]));
            }
            report("kv_append (K cell)", compare(got, ref), 1e-6);
        }
    } catch (const std::exception& e) {
        std::printf("selftest: exception: %s\n", e.what());
        return 1;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("selftest: a kernel failed\n"); return 1; }
    std::printf(bad ? "selftest: %d FAILED\n" : "selftest: all passed\n", bad);
    return bad ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--selftest") return selftest();
    std::string gguf;
    long long context = 32768;
    bool serve = false;
    int draft = -1, chunk = 0;              // drafts a round (-1: STRATA_DENSE_DRAFT, else 3 / DFlash2 7; 0: off), prompt chunk (0: the model picks)
    std::string dflash;                     // a DFlash2 drafter GGUF (else STRATA_DENSE_DFLASH): it drafts, not the MTP block
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "strata-dense: %s needs a value\n", what); std::exit(2); }
            return argv[++i];
        };
        if (a == "--serve") serve = true;
        else if (a == "--native" || a == "--model" || a == "--gguf") gguf = next("--native");
        else if (a == "--context" || a == "-c") context = std::atoll(next("--context"));
        else if (a == "--draft" || a == "--mtp-draft") draft = std::atoi(next("--draft"));
        else if (a == "--chunk") chunk = std::atoi(next("--chunk"));
        else if (a == "--dflash") dflash = next("--dflash");
        else if (a == "--kv") setenv("STRATA_DENSE_KV", next("--kv"), 1);
        else if (a == "--mtp") setenv("STRATA_DENSE_MTP", next("--mtp"), 1);   // (Gemma 4: the assistant GGUF)
        else if (a == "--help" || a == "-h") { usage(); return 0; }
        // the flags setup passes to `strata` (--gpu-layers, --cache, ...) mean nothing here and are not an error
    }
    if (!serve || gguf.empty()) { usage(); return 2; }

    strata::core::DenseModel model;
    std::string err;
    std::fprintf(stderr, "strata-dense: loading %s ...\n", gguf.c_str());
    if (const char* e = std::getenv("STRATA_DENSE_CHUNK")) chunk = std::atoi(e);
    if (!model.load(gguf, context, err, chunk)) {
        std::fprintf(stderr, "strata-dense: %s\n", err.c_str());
        std::printf("ERR %s\n", err.c_str());
        return 1;
    }
    const auto& cfg = model.config();
    if (dflash.empty() && std::getenv("STRATA_DENSE_DFLASH")) dflash = std::getenv("STRATA_DENSE_DFLASH");
    if (!dflash.empty() && !model.dflash_load(dflash, err)) {
        std::fprintf(stderr, "strata-dense: %s\n", err.c_str());
        std::printf("ERR %s\n", err.c_str());
        return 1;
    }
    const bool use_df = model.has_dflash();
    // (a MoE model's MTP drafts adaptively from 5 - measured on Qwen3.6-35B-A3B: 141 -> 168 tok/s at 5, 125 -> 172 at 7;
    // the dense Qwen3.8-27B a fixed 4 (with the 32K draft vocabulary, two cooled rounds: K3 65.7 / 65.7, K4 68.0 / 68.2,
    // K5 66.2 / 66.2, adaptive from 4 63.3 / 63.5 tok/s); the Gemma 4 assistants by size - 31B 1 (short: K1 41.1, K2 38.8,
    // K3 33.0, K4 29.2, plain 33.2; 28K K1 29.6 vs K3 27.2 tok/s), E4B 2 (K1 128.4, K2 157.4, K3 146.7); Muse keeps 3)
    const bool qwen_dense = cfg.n_expert == 0 && cfg.arch.rfind("qwen3", 0) == 0;
    const int gemma_k = cfg.arch == "gemma4" ? (cfg.n_embd >= 4096 ? 1 : 2) : 0;
    if (draft < 0) draft = std::getenv("STRATA_DENSE_DRAFT") ? std::atoi(std::getenv("STRATA_DENSE_DRAFT"))
                                                             : (use_df ? 7 : cfg.n_expert > 0 ? 5 : qwen_dense ? 4 : gemma_k > 0 ? gemma_k : 3);
    draft = std::max(0, std::min(draft, use_df ? model.dflash_max() : 7));
    if (!use_df && !model.has_mtp()) draft = 0;
    cudaStream_t cs = (cudaStream_t) model.stream();

    // the ids that end an answer, from the model's own vocabulary
    std::vector<int32_t> stop_ids;
    if (cfg.eos_id >= 0) stop_ids.push_back(cfg.eos_id);
    try {
        strata::GgufFile g(gguf);
        strata::MetaValue tv;   // (every token: metadata() keeps a sample of each array)
        if (const auto* toks = g.get_full("tokenizer.ggml.tokens", tv) ? &tv : nullptr)
            for (size_t i = 0; i < toks->items.size(); ++i)
                if (toks->items[i].s == "<|im_end|>" || toks->items[i].s == "<|endoftext|>" ||   // Qwen
                    toks->items[i].s == "<|eot|>" ||                                                // Muse Glimmer's turn end
                    toks->items[i].s == "<turn|>")                                                  // Gemma 4's
                    stop_ids.push_back((int32_t) i);
    } catch (const std::exception&) {}
    int32_t im_start = -1;   // the chat template's message header: prompts are checkpointed at its last occurrence
    try {
        strata::GgufFile g(gguf);
        strata::MetaValue tv;   // (every token: metadata() keeps a sample of each array)
        if (const auto* toks = g.get_full("tokenizer.ggml.tokens", tv) ? &tv : nullptr)
            for (size_t i = 0; i < toks->items.size(); ++i)
                if (toks->items[i].s == "<|im_start|>" || toks->items[i].s == "<|start|>" || toks->items[i].s == "<|turn>") {
                    im_start = (int32_t) i;   // (Qwen's, Muse Glimmer's, Gemma 4's message header)
                    break;
                }
    } catch (const std::exception&) {}
    if (std::filesystem::is_directory(gguf)) {   // (a checkpoint directory: its tokenizer_config.json's added tokens)
        try {
            const strata::Json tc = strata::Json::parse(strata::read_text(gguf + "/tokenizer_config.json"));
            if (const strata::Json* add = tc.get("added_tokens_decoder"))
                for (const auto& kv : add->obj) {
                    const strata::Json* c = kv.second.get("content");
                    if (!c) continue;
                    const int32_t id = (int32_t) std::atoi(kv.first.c_str());
                    if (c->str == "<|im_end|>" || c->str == "<|endoftext|>") stop_ids.push_back(id);
                    if (c->str == "<|im_start|>") im_start = id;
                }
        } catch (const std::exception&) {}
    }
    std::sort(stop_ids.begin(), stop_ids.end());
    stop_ids.erase(std::unique(stop_ids.begin(), stop_ids.end()), stop_ids.end());

    int* d_next = nullptr;
    int* d_hist = nullptr;
    float* d_conf = nullptr;   // (a draft's top-1 probability: STRATA_DENSE_P_CUM)
    float* d_hrows = nullptr;                      // the trunk's hidden rows the MTP block has yet to read
    int* d_tki = nullptr;                          // sampled drafts: rows' top-k ids / logits and the selection scratch
    float* d_tkv = nullptr;
    void* d_tks = nullptr;
    constexpr int kMaxHist = 4096;
    if (cudaMalloc((void**) &d_next, 16 * sizeof(int)) != cudaSuccess || cudaMalloc((void**) &d_conf, sizeof(float)) != cudaSuccess ||
        cudaMalloc((void**) &d_hist, kMaxHist * sizeof(int)) != cudaSuccess ||
        cudaMalloc((void**) &d_hrows, (size_t) 16 * cfg.n_embd * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void**) &d_tki, 16 * 32 * sizeof(int)) != cudaSuccess || cudaMalloc((void**) &d_tkv, 16 * 32 * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_tks, (size_t) strata::kernels::q35_df_topk_scratch(16, 32)) != cudaSuccess) {
        std::printf("ERR cannot allocate the sampler buffers\n");
        return 1;
    }

    std::printf("INFO engine=dense-0.2 kv=%s model=%s layers=%d n_embd=%d weights_mib=%llu host_mib=%llu kv_mib=%llu context=%lld mtp=%d\n",
                std::getenv("STRATA_DENSE_KV") ? std::getenv("STRATA_DENSE_KV") : "fp16",
                cfg.arch.c_str(), cfg.n_layer, cfg.n_embd, (unsigned long long) (model.weight_bytes() >> 20),
                (unsigned long long) (model.host_weight_bytes() >> 20), (unsigned long long) (model.kv_bytes() >> 20),
                (long long) model.max_context(), draft);
    std::printf("READY %lld stop\n", (long long) model.max_context());
    std::fflush(stdout);
    std::fprintf(stderr, "strata-dense: ready - %.2f GiB of weights, %.2f GiB of KV cache, context %lld, MTP drafts %d\n",
                 model.weight_bytes() / 1073741824.0, model.kv_bytes() / 1073741824.0, (long long) model.max_context(),
                 draft);

    if (const char* e = std::getenv("STRATA_DENSE_BENCH")) {
        // the cost of a forward pass by width: a prompt of N tokens, then verify windows of 1..8 (each committed 1)
        const int n0 = std::max(16, std::atoi(e));
        std::vector<int32_t> toks((size_t) n0 + 64, 1000);
        for (size_t i = 0; i < toks.size(); ++i) toks[i] = (int32_t) (1000 + (i * 7919) % 50000);
        for (int i = 0; i < n0; i += model.chunk()) {
            const int T = std::min(model.chunk(), n0 - i);
            if (!model.forward(toks.data() + i, T, false, 0, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        }
        model.sync(err);
        const int wmax = std::getenv("STRATA_DENSE_BENCH_W") ? std::atoi(std::getenv("STRATA_DENSE_BENCH_W")) : 8;
        const int reps = std::getenv("STRATA_DENSE_BENCH_REPS") ? std::atoi(std::getenv("STRATA_DENSE_BENCH_REPS")) : 12;
        for (int T = std::getenv("STRATA_DENSE_BENCH_W") ? wmax : 1; T <= wmax; ++T) {
            double tot = 0;
            for (int r = 0; r < reps; ++r) {
                const auto t0 = Clock::now();
                if (!model.forward(toks.data() + n0, T, true, strata::core::DenseModel::kAll, err) || !model.sync(err)) {
                    std::fprintf(stderr, "%s\n", err.c_str()); return 1;
                }
                if (r >= 2) tot += ms_since(t0);
                model.commit(1, err);
            }
            std::fprintf(stderr, "bench ctx %lld: verify window %d: %.2f ms\n", (long long) model.position(), T, tot / (reps - 2));
        }
        if (std::getenv("STRATA_DENSE_BENCH_GRAPH")) {   // the same pass captured once and replayed: launch overhead
            const int T = wmax;
            cudaStream_t gs = (cudaStream_t) model.stream();
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            cudaStreamBeginCapture(gs, cudaStreamCaptureModeGlobal);
            const bool ok = model.forward(toks.data() + n0, T, true, strata::core::DenseModel::kAll, err);
            cudaStreamEndCapture(gs, &graph);
            model.commit(1, err);
            size_t nodes = 0;
            cudaGraphGetNodes(graph, nullptr, &nodes);
            if (!ok || cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != cudaSuccess) {
                std::fprintf(stderr, "bench graph: capture failed: %s %s\n", err.c_str(), cudaGetErrorString(cudaGetLastError()));
                return 1;
            }
            double tot = 0;
            for (int r = 0; r < reps; ++r) {
                const auto t0 = Clock::now();
                cudaGraphLaunch(exec, gs);
                cudaStreamSynchronize(gs);
                if (r >= 2) tot += ms_since(t0);
            }
            std::fprintf(stderr, "bench graph: verify window %d replayed: %.2f ms (%zu nodes)\n", T, tot / (reps - 2), nodes);
        }
        return 0;
    }

    Lines in;
    std::thread reader([&] {
        std::string line;
        while (std::getline(std::cin, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
            if (line == "STOP") { in.stop.store(true); continue; }
            in.push(line);
        }
        std::lock_guard<std::mutex> l(in.m);
        in.eof = true;
        in.cv.notify_all();
    });
    reader.detach();

    // context lookup (NInfer's, Strata's suffix drafter): when the text's last tokens occurred before - code being
    // edited, a tool's output quoted - the tokens that followed are proposed instead of the learned drafter's
    const int lookup_min = std::getenv("STRATA_DENSE_LOOKUP_MIN") ? std::atoi(std::getenv("STRATA_DENSE_LOOKUP_MIN")) : 12;
    strata::spec::SuffixDrafter lookup(std::max(3, lookup_min), 64);
    // prefix checkpoints (FreeToken's anchors, lithos-metal's prefix checkpoints): the recurrent state at the start of
    // a prompt's last message header, in pinned host memory; a later prompt that shares that prefix but not the rest
    // of the history (an agent dropped the reasoning, rewrote a tool call) restores it instead of reading from 0
    const int n_ckpt = std::getenv("STRATA_DENSE_CKPTS") ? std::max(0, std::atoi(std::getenv("STRATA_DENSE_CKPTS"))) : 8;
    struct Ckpt { int64_t len = -1; uint64_t hash = 0, used = 0; };
    std::vector<Ckpt> ckpts((size_t) n_ckpt);
    uint64_t ckpt_clock = 0;
    auto prefix_hash = [](const std::vector<int32_t>& v, int64_t n) {
        uint64_t h = 1469598103934665603ull;
        for (int64_t i = 0; i < n; ++i) { h ^= (uint32_t) v[(size_t) i]; h *= 1099511628211ull; }
        return h;
    };
    if (n_ckpt > 0)
        std::fprintf(stderr, "strata-dense: %d prefix checkpoints of %.0f MiB (pinned host memory, on first use), "
                     "at the last header %d\n", n_ckpt, model.ckpt_bytes() / 1048576.0, im_start);
    std::vector<int32_t> fed;                      // every token the state has consumed, in order
    uint64_t draw_counter = 0;
    std::string line;
    while (in.pop(line)) {
        if (line == "QUIT") break;
        in.stop.store(false);                       // a STOP between requests is stale
        if (line.rfind("GEN ", 0) != 0) { std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n"); std::fflush(stdout); continue; }

        char* endp = nullptr;
        const long long max_new = std::strtoll(line.c_str() + 4, &endp, 10);
        strata::kernels::SamplerParams sp;
        sp.top_k = 20; sp.top_p = 1.0f; sp.temperature = 0.0f; sp.greedy = true;
        unsigned long long seed = 0;
        bool ignore_eos = false;
        int req_draft = draft;
        long long think_budget = 0;               // think_budget=<n> think_end=<id>: the server's reasoning budget
        int32_t think_end = -1;
        for (;;) {
            while (*endp == ' ') ++endp;
            const char* start = endp;
            while (*endp != '\0' && *endp != ' ') ++endp;
            if (endp == start) break;
            const std::string tok(start, (size_t) (endp - start));
            const size_t eq = tok.find('=');
            if (eq == std::string::npos) { endp = const_cast<char*>(start); break; }
            const std::string key = tok.substr(0, eq);
            const float fv = std::strtof(tok.c_str() + eq + 1, nullptr);
            if (key == "temperature") { sp.temperature = fv; sp.greedy = fv <= 0.0f; }
            else if (key == "top_p") sp.top_p = fv;
            else if (key == "top_k") sp.top_k = std::atoi(tok.c_str() + eq + 1);
            else if (key == "min_p") sp.min_p = fv;
            else if (key == "penalty_last_n") sp.penalty_last_n = std::min(std::atoi(tok.c_str() + eq + 1), kMaxHist);
            else if (key == "penalty_repeat") sp.penalty_repeat = fv;
            else if (key == "penalty_freq") sp.penalty_freq = fv;
            else if (key == "penalty_present") sp.penalty_present = fv;
            else if (key == "seed") seed = std::strtoull(tok.c_str() + eq + 1, nullptr, 10);
            else if (key == "ignore_eos") ignore_eos = fv != 0.0f;
            else if (key == "think_budget") think_budget = std::atoll(tok.c_str() + eq + 1);
            else if (key == "think_end") think_end = (int32_t) std::atol(tok.c_str() + eq + 1);
            else if (key == "draft") req_draft = std::max(0, std::min(draft, std::atoi(tok.c_str() + eq + 1)));
        }
        std::vector<int32_t> ids;
        if (max_new < 1 || !parse_ids(endp, ids)) { std::printf("ERR bad request\n"); std::fflush(stdout); continue; }
        if ((long long) ids.size() + 1 > model.max_context()) {
            std::printf("ERR the prompt (%zu tokens) does not fit the context (%lld)\n", ids.size(), (long long) model.max_context());
            std::fflush(stdout);
            continue;
        }
        sp.seed = seed;
        auto is_stop = [&](int32_t t) { return !ignore_eos && std::binary_search(stop_ids.begin(), stop_ids.end(), t); };
        // penalties look at a history the verify rows would each need their own of: those requests decode plainly
        const bool penalties = sp.penalty_last_n > 0 &&
                               (sp.penalty_repeat != 1.0f || sp.penalty_freq != 0.0f || sp.penalty_present != 0.0f);
        const int K = penalties ? 0 : req_draft;

        // ---- what of the state can be kept (the MTP cache follows the trunk: the same positions are valid)
        size_t resume = 0;
        if (!fed.empty() && fed.size() < ids.size() && std::equal(fed.begin(), fed.end(), ids.begin())) resume = fed.size();
        bool restored = false;
        if (resume == 0 && !fed.empty() && n_ckpt > 0) {
            // the K/V cache holds fed's tokens: a checkpoint inside the common prefix of fed and the prompt is valid
            size_t cp = 0;
            while (cp < fed.size() && cp + 1 < ids.size() && fed[cp] == ids[cp]) ++cp;
            int best = -1;
            for (int c = 0; c < n_ckpt; ++c)
                if (ckpts[(size_t) c].len > 0 && (size_t) ckpts[(size_t) c].len <= cp &&
                    (best < 0 || ckpts[(size_t) c].len > ckpts[(size_t) best].len) &&
                    ckpts[(size_t) c].hash == prefix_hash(ids, ckpts[(size_t) c].len))
                    best = c;
            std::fprintf(stderr, "strata-dense: history diverges at %zu of %zu; checkpoint %d\n", cp, fed.size(), best);
            if (best >= 0 && model.ckpt_restore(best, ckpts[(size_t) best].len, err)) {
                resume = (size_t) ckpts[(size_t) best].len;
                fed.resize(resume);
                ckpts[(size_t) best].used = ++ckpt_clock;
                restored = true;
            }
        }
        if (resume == 0) { model.reset(); fed.clear(); }
        if (resume == 0 || lookup.size() != resume) {   // (the lookup history is the prompt and the answer so far)
            lookup.reset();
            lookup.append(ids.data(), ids.size());
            // the MTP draft head's vocabulary for this conversation (its first ids + the prompt's frequent later ones)
            if (K > 0 && !model.mtp_vocab_select(ids.data(), ids.size(), err)) {
                std::printf("ERR %s\n", err.c_str());
                std::fflush(stdout);
                continue;
            }
        } else {
            lookup.append(ids.data() + resume, ids.size() - resume);
        }
        const int64_t n = (int64_t) ids.size();
        const long long budget = std::min<long long>(max_new, model.max_context() - n);

        // ---- read the prompt, a chunk at a time; the MTP block reads every row whose next token is known
        bool cancelled = false, failed = false;
        const auto t_prompt = Clock::now();
        const int ch = model.chunk();
        int64_t anchor_at = -1;   // the last message header (a checkpoint), when it is still to be read
        if (n_ckpt > 0 && im_start >= 0)
            for (int64_t j = n - 1; j > (int64_t) resume; --j)
                if (ids[(size_t) j] == im_start) { anchor_at = j; break; }
        for (int64_t i = (int64_t) resume; i < n && !failed;) {
            int T = (int) std::min<int64_t>(ch, n - i);
            if (anchor_at > i && anchor_at < i + T) T = (int) (anchor_at - i);   // a chunk ends at the header
            const bool last = i + T == n;
            if (!model.forward(ids.data() + i, T, false, last ? strata::core::DenseModel::kLast : strata::core::DenseModel::kNone, err)) {
                std::printf("ERR %s\n", err.c_str()); failed = true; break;
            }
            if (K > 0 && use_df) {
                if (!model.dflash_inject(T, i, err)) { std::printf("ERR %s\n", err.c_str()); failed = true; break; }
            } else if (K > 0) {
                const int rows = last ? T - 1 : T;   // row t pairs with token i + t + 1
                if (rows > 0 && !model.mtp_fill(ids.data() + i + 1, model.hidden_norm(), i, rows, err)) {
                    std::printf("ERR %s\n", err.c_str()); failed = true; break;
                }
                if (last)   // the prompt's last hidden row: the first draft step reads it
                    cudaMemcpyAsync(d_hrows, model.hidden_norm() + (size_t) (T - 1) * cfg.n_embd, cfg.n_embd * 4,
                                    cudaMemcpyDeviceToDevice, cs);
            }
            fed.insert(fed.end(), ids.begin() + i, ids.begin() + i + T);
            i += T;
            if (i == anchor_at) {
                const uint64_t h = prefix_hash(ids, i);
                int slot = -1;
                for (int c = 0; c < n_ckpt && slot < 0; ++c)
                    if (ckpts[(size_t) c].len == i && ckpts[(size_t) c].hash == h) slot = -2;   // (already there)
                if (slot == -1) {
                    slot = 0;
                    for (int c = 1; c < n_ckpt; ++c)
                        if (ckpts[(size_t) c].used < ckpts[(size_t) slot].used) slot = c;
                    if (model.ckpt_save(slot, err)) {
                        ckpts[(size_t) slot] = {i, h, ++ckpt_clock};
                        std::fprintf(stderr, "strata-dense: checkpoint %d at position %lld\n", slot, (long long) i);
                    }
                    else { std::fprintf(stderr, "strata-dense: %s\n", err.c_str()); ckpts[(size_t) slot].len = -1; }
                }
            }
            if (!model.sync(err)) { std::printf("ERR %s\n", err.c_str()); failed = true; break; }
            const double ms = ms_since(t_prompt);
            std::printf("PP %lld %lld %.0f %.1f\n", (long long) i, (long long) n, ms,
                        ms > 0 ? 1000.0 * (i - (int64_t) resume) / ms : 0.0);
            std::fflush(stdout);
            if (in.stop.load()) { cancelled = true; break; }
        }
        if (failed) { model.reset(); fed.clear(); std::fflush(stdout); continue; }
        const double prompt_ms = ms_since(t_prompt);

        // ---- decode
        long long produced = 0;
        long long rounds = 0, drafted = 0, accepted = 0;
        const char* finish = "length";
        const auto t_dec = Clock::now();
        std::vector<int32_t> hist;                  // penalty window, most recent last
        if (sp.penalty_last_n > 0) {
            const size_t take = std::min<size_t>((size_t) sp.penalty_last_n, fed.size());
            hist.assign(fed.end() - (std::ptrdiff_t) take, fed.end());
        }
        // the reasoning budget: past it, the next token not yet fed (a round's last pick, the next anchor) becomes the
        // end of the reasoning - as the GLM engine forces it
        bool in_think = think_budget > 0 && think_end >= 0;
        auto budget_force = [&](int32_t& t) {
            if (in_think && produced >= think_budget && t != think_end) t = think_end;
        };
        long long lookup_rounds = 0;
        // the answer so far, for the MTP draft head's vocabulary (re-picked every 64 tokens with the prompt's)
        // (STRATA_DENSE_MTP_VOCAB_EVERY tokens, default 64; 0: the prompt's pick only)
        static const long long vsel_every = std::getenv("STRATA_DENSE_MTP_VOCAB_EVERY") ? std::atoll(std::getenv("STRATA_DENSE_MTP_VOCAB_EVERY")) : 64;
        std::vector<int32_t> vgen;
        long long vsel_at = vsel_every > 0 ? vsel_every : LLONG_MAX;
        auto emit = [&](int32_t t) -> bool {        // false: the answer ends here
            if (in_think && t == think_end) in_think = false;
            lookup.append(t);
            if (K > 0) vgen.push_back(t);
            ++produced;
            std::printf("T %d\n", t);
            if (sp.penalty_last_n > 0) { hist.push_back(t); if ((int) hist.size() > sp.penalty_last_n) hist.erase(hist.begin()); }
            if (is_stop(t)) { finish = "stop"; return false; }
            return produced < budget;
        };
        auto sample_rows = [&](const float* logits, int rows, int* out_host) -> bool {
            sp.counter = draw_counter;
            const int* hp = nullptr;
            int hl = 0;
            if (sp.penalty_last_n > 0) {
                std::vector<int32_t> row((size_t) sp.penalty_last_n, -1);
                std::copy(hist.begin(), hist.end(), row.end() - (std::ptrdiff_t) hist.size());
                cudaMemcpyAsync(d_hist, row.data(), row.size() * sizeof(int), cudaMemcpyHostToDevice, cs);
                hp = d_hist;
                hl = sp.penalty_last_n;
            }
            try {
                if (sp.greedy) strata::kernels::q35_argmax_rows(logits, cfg.n_vocab, cfg.n_vocab, rows, d_next, cs);
                else strata::kernels::sample_tokens(logits, rows, cfg.n_vocab, hp, hl, sp, d_next, cs);
            } catch (const std::exception& e) { std::printf("ERR sampler: %s\n", e.what()); return false; }
            cudaMemcpyAsync(out_host, d_next, rows * sizeof(int), cudaMemcpyDeviceToHost, cs);
            if (!model.sync(err)) { std::printf("ERR %s\n", err.c_str()); return false; }
            draw_counter += rows;
            return true;
        };
        int anchor = -1;
        double draft_ms = 0;
        // STRATA_DENSE_ADAPTIVE=1: the MTP draft count follows the acceptance (from K down to 1 and back)
        // STRATA_DENSE_SPEC_SAMPLE=1: MTP drafts SAMPLED from the draft head's distribution q and kept with min(1, p / q),
        // a rejection resampled from max(0, p - q) - the output is the target's distribution exactly (lithos-metal's
        // speculative decoding; greedy drafts keep only a match with the target's own draw).  Sampled requests with a
        // top_k of 1..32 (both distributions live on the rows' top k).
        // (default on for a MoE model - Qwen3.6-35B-A3B MTP: T 0.6 148.7 -> 155.5, T 1.0 130.1 -> 147.7 tok/s; the dense
        // 27B lost 2-3%: off there)
        static const bool spec_env = std::getenv("STRATA_DENSE_SPEC_SAMPLE") ? std::atoi(std::getenv("STRATA_DENSE_SPEC_SAMPLE")) != 0
                                                                              : cfg.n_expert > 0;
        // (DFlash2 too: its walk then draws the drafts and reports their proposal distributions - ideas.md 36)
        const bool spec_s = spec_env && !sp.greedy && (!use_df || model.dflash_samples()) && K > 0 && sp.top_k >= 1 &&
                            sp.top_k <= 32;
        // STRATA_DENSE_BLOCK_VERIFY=0: sampled drafts tested token by token instead of as a block
        static const bool block_verify = !(std::getenv("STRATA_DENSE_BLOCK_VERIFY") && std::atoi(std::getenv("STRATA_DENSE_BLOCK_VERIFY")) == 0);
        static const double p_cum_env = std::getenv("STRATA_DENSE_P_CUM") ? std::atof(std::getenv("STRATA_DENSE_P_CUM")) : -1.0;
        static const bool spec_pad = std::getenv("STRATA_DENSE_SPEC_PAD") && std::atoi(std::getenv("STRATA_DENSE_SPEC_PAD")) != 0;
        long long conf_stops = 0;
        std::mt19937_64 srng(sp.seed * 0x9E3779B97F4A7C15ull + draw_counter + 1);
        std::vector<SparseDist> qd;
        auto topk_rows = [&](const float* logits, int ld, int rows, std::vector<SparseDist>& out) -> bool {
            strata::kernels::q35_df_topk(logits, ld, ld, rows, sp.top_k, d_tki, d_tkv, d_tks, cs);
            std::vector<int> ids((size_t) rows * sp.top_k);
            std::vector<float> lg((size_t) rows * sp.top_k);
            cudaMemcpyAsync(ids.data(), d_tki, ids.size() * 4, cudaMemcpyDeviceToHost, cs);
            cudaMemcpyAsync(lg.data(), d_tkv, lg.size() * 4, cudaMemcpyDeviceToHost, cs);
            if (!model.sync(err)) { std::printf("ERR %s\n", err.c_str()); return false; }
            for (int r = 0; r < rows; ++r)
                out.push_back(sparse_dist(ids.data() + (size_t) r * sp.top_k, lg.data() + (size_t) r * sp.top_k, sp.top_k, sp));
            return true;
        };
        // STRATA_DENSE_DEV_CHAIN=1: the greedy MTP drafts chained on the device (one sync a round, not one a draft) - measured
        // even (27B MTP4 68.0 / 68.0 vs 68.0 / 68.4, E4B assistant 156.5 vs 156.6 tok/s, the same tokens): off by default
        static const bool dev_chain_env = std::getenv("STRATA_DENSE_DEV_CHAIN") && std::atoi(std::getenv("STRATA_DENSE_DEV_CHAIN")) != 0;
        const bool dev_chain = dev_chain_env && !spec_s && model.mtp_dev_ok();
        static const bool adaptive = std::getenv("STRATA_DENSE_ADAPTIVE") ? std::atoi(std::getenv("STRATA_DENSE_ADAPTIVE")) != 0
                                                                          : cfg.n_expert > 0;
        int k_cur = K;
        // STRATA_DENSE_KSEL=1: the draft count from measured throughput - per count k (1..K) a running mean of the tokens a
        // round commits and the round's ms (next round's start - this one's); every count tried first, the stalest one
        // again every 64 rounds, else the best tokens / ms (a smaller k only when >= 3 % better).  Lookup rounds not counted
        static const bool ksel = std::getenv("STRATA_DENSE_KSEL") && std::atoi(std::getenv("STRATA_DENSE_KSEL")) != 0;
        double ks_tok[17] = {}, ks_ms[17] = {};
        long long ks_n[17] = {}, ks_last[17] = {}, ks_round = 0;
        int ks_k = -1, ks_commit = 0;
        Clock::time_point ks_t0 = Clock::now();
        if (!cancelled && budget > 0) {
            if (!sample_rows(model.logits(), 1, &anchor)) failed = true;
            else { budget_force(anchor); emit(anchor); }                         // (a stop token sets finish)
        }
        bool going = !failed && !cancelled && budget > 0 && produced < budget && finish != std::string("stop");
        // the MTP rows still to read: tokens mtp_tok at positions mtp_p0 .. with the trunk's hidden rows in d_hrows
        std::vector<int32_t> mtp_tok = {anchor};
        int64_t mtp_p0 = model.position() - 1;
        // STRATA_DENSE_ROUND_PROF=1: each round's phases timed on the host after a sync (draft, verify, accept,
        // commit) - the sum at DONE (debug: the syncs cost a little)
        static const bool rprof = std::getenv("STRATA_DENSE_ROUND_PROF") && std::atoi(std::getenv("STRATA_DENSE_ROUND_PROF")) != 0;
        double rp[4] = {0, 0, 0, 0};
        // STRATA_DENSE_ACC_HIST=1: rounds by drafts kept (0 .. k) and window size, at DONE - per-position acceptance
        static const bool acc_hist = std::getenv("STRATA_DENSE_ACC_HIST") && std::atoi(std::getenv("STRATA_DENSE_ACC_HIST")) != 0;
        long long ah[17][17] = {};   // [drafts offered][drafts kept]
        auto rp_mark = [&](int i, Clock::time_point& t) {
            if (!rprof) return;
            std::string e2;
            model.sync(e2);
            rp[i] += ms_since(t);
            t = Clock::now();
        };
        while (going) {
            if (in.stop.load()) { cancelled = true; break; }
            if (K > 0 && produced >= vsel_at) {
                vsel_at = produced + vsel_every;
                if (!model.mtp_vocab_select(ids.data(), ids.size(), vgen.data(), vgen.size(), err)) {
                    std::printf("ERR %s\n", err.c_str()); failed = true; break;
                }
            }
            const int64_t pos = model.position();
            Clock::time_point t_rp = Clock::now();
            int k = (int) std::min<long long>(K, std::min<long long>(budget - produced, model.max_context() - pos - 1));
            if (adaptive && !use_df) k = std::min(k, k_cur);   // (MTP: each draft is a step of its own)
            if (ksel && K >= 1 && K <= 16) {
                if (ks_k > 0) {   // the last round's tokens and time
                    const double ms = std::chrono::duration<double, std::milli>(t_rp - ks_t0).count();
                    const double a = ks_n[ks_k] == 0 ? 1.0 : 0.15;
                    ks_tok[ks_k] += a * ((double) ks_commit - ks_tok[ks_k]);
                    ks_ms[ks_k] += a * (ms - ks_ms[ks_k]);
                    ++ks_n[ks_k];
                    ks_last[ks_k] = ks_round;
                }
                ++ks_round;
                int pick = -1;
                for (int c = K; c >= 1 && pick < 0; --c)
                    if (ks_n[c] < 3) pick = c;
                if (pick < 0 && ks_round % 64 == 0) {
                    pick = 1;
                    for (int c = 2; c <= K; ++c)
                        if (ks_last[c] < ks_last[pick]) pick = c;
                }
                if (pick < 0) {
                    pick = K;
                    double best = ks_tok[K] / std::max(1e-6, ks_ms[K]);
                    for (int c = K - 1; c >= 1; --c) {
                        const double r = ks_tok[c] / std::max(1e-6, ks_ms[c]);
                        if (r > best * 1.03) { best = r; pick = c; }
                    }
                }
                k = std::min(k, pick);
                ks_k = k > 0 ? k : -1;
                ks_t0 = t_rp;
            }
            k = std::max(0, k);
            std::vector<int32_t> win = {anchor};
            const long long lookup_rounds_before = lookup_rounds;   // (a lookup round's drafts are not q's draws)
            int32_t lk[16];
            const int nl = k > 0 && lookup_min > 0 ? lookup.propose(k, lk) : 0;
            if (nl > 0 && lookup.last_match() >= lookup_min) {
                win.insert(win.end(), lk, lk + nl);
                ++lookup_rounds;
                // the MTP block still reads its pending rows (its cache follows the trunk)
                if (!use_df && !model.mtp_fill(mtp_tok.data(), d_hrows, mtp_p0, (int) mtp_tok.size(), err)) {
                    std::printf("ERR %s\n", err.c_str()); failed = true; break;
                }
            } else if (k > 0 && use_df) {
                std::vector<int32_t> d;
                const auto t_d = Clock::now();
                if (spec_s) {   // a uniform a draft from the request's generator; the walk draws with them
                    std::vector<float> u((size_t) k);
                    for (float& x : u) x = (float) std::uniform_real_distribution<double>(0.0, 1.0)(srng);
                    model.dflash_sample(u, sp.temperature);
                }
                if (!model.dflash_draft(anchor, k, d, err)) { std::printf("ERR %s\n", err.c_str()); failed = true; break; }
                draft_ms += ms_since(t_d);
                // the confidence floor (STRATA_DENSE_P_CUM, as for MTP below): the drafts after the one whose running
                // product of confidences fell under it are not verified
                const double p_cut_df = p_cum_env >= 0.0 ? p_cum_env
                                        : pos < 16384 ? 0.0 : pos >= 49152 ? 0.3 : 0.3 * (double) (pos - 16384) / 32768.0;
                const std::vector<float>& cf = model.dflash_conf();
                if (p_cut_df > 0.0 && cf.size() == d.size()) {
                    double pr = 1.0;
                    for (size_t j = 0; j < d.size(); ++j) {
                        pr *= cf[j];
                        if (pr < p_cut_df && j + 1 < d.size()) { d.resize(j + 1); ++conf_stops; break; }
                    }
                }
                if (spec_s) {   // the drafts' proposal distributions, for the exact test
                    std::vector<float> q;
                    std::vector<int32_t> cand;
                    int kq = 0;
                    model.dflash_proposals(q, cand, kq);
                    qd.clear();
                    for (size_t j = 0; j < d.size() && (j + 1) * (size_t) kq <= q.size(); ++j) {
                        SparseDist sd;
                        sd.n = std::min(kq, 32);
                        for (int i = 0; i < sd.n; ++i) { sd.ids[i] = cand[j * kq + i]; sd.p[i] = q[j * kq + i]; }
                        qd.push_back(sd);
                    }
                }
                win.insert(win.end(), d.begin(), d.end());
            } else if (k > 0) {
                // the drafts: the MTP block over the pending rows (the last one's logits), then one row a draft
                if (!model.mtp_forward(mtp_tok.data(), d_hrows, mtp_p0, (int) mtp_tok.size(), false, err)) {
                    std::printf("ERR %s\n", err.c_str()); failed = true; break;
                }
                qd.clear();
                // STRATA_DENSE_P_CUM: greedy MTP drafts stop once the product of their top-1 probabilities falls below
                // this (the draft head is well calibrated: a cheap verify loses more by a short draft than it saves, a
                // long context's verify gains - Kmic-68/llama.cpp's P100 fork, +10-16% at 260K); unset: 0 below 16K of
                // context, 0.3 from 48K, a ramp between; 0: off
                const double p_cut = p_cum_env >= 0.0 ? p_cum_env
                                     : pos < 16384 ? 0.0 : pos >= 49152 ? 0.3 : 0.3 * (double) (pos - 16384) / 32768.0;
                double p_prod = 1.0;
                for (int j = 0; j < k; ++j) {
                    int d = 0;
                    if (spec_s) {   // a draw from the draft head's own (filtered) distribution, kept for the accept test
                        if (!topk_rows(model.mtp_logits(), model.draft_vocab(), 1, qd)) { failed = true; break; }
                        if (model.draft_ids())   // (the head's slots to token ids)
                            for (int i = 0; i < qd.back().n; ++i) qd.back().ids[i] = model.draft_id(qd.back().ids[i]);
                        d = draw(qd.back(), srng);
                    } else if (dev_chain) {
                        // (device-resident chain: draft j's argmax stays in d_next[j] and feeds the next step; the
                        // drafts reach the host once, after the last)
                        strata::kernels::q35_argmax_rows(model.mtp_logits(), model.draft_vocab(), model.draft_vocab(), 1, d_next + j, cs);
                        if (model.draft_ids()) strata::kernels::q35_map_ids(d_next + j, model.draft_ids(), 1, cs);
                        if (j + 1 < k && !model.mtp_forward_dev(d_next + j, model.mtp_hidden(), pos + j, err)) {
                            std::printf("ERR %s\n", err.c_str()); failed = true; break;
                        }
                        if (j + 1 == k) {
                            int dh[16];
                            cudaMemcpyAsync(dh, d_next, (size_t) k * sizeof(int), cudaMemcpyDeviceToHost, cs);
                            if (!model.sync(err)) { std::printf("ERR %s\n", err.c_str()); failed = true; break; }
                            win.insert(win.end(), dh, dh + k);
                        }
                        continue;
                    } else {
                        strata::kernels::q35_argmax_rows(model.mtp_logits(), model.draft_vocab(), model.draft_vocab(), 1, d_next, cs);
                        cudaMemcpyAsync(&d, d_next, sizeof(int), cudaMemcpyDeviceToHost, cs);
                        float pc = 1.0f;
                        if (p_cut > 0.0) {
                            strata::kernels::q35_top1_prob(model.mtp_logits(), model.draft_vocab(), d_next, d_conf, cs);
                            cudaMemcpyAsync(&pc, d_conf, sizeof(float), cudaMemcpyDeviceToHost, cs);
                        }
                        if (!model.sync(err)) { std::printf("ERR %s\n", err.c_str()); failed = true; break; }
                        d = model.draft_id(d);   // (the head's slot to its token id)
                        p_prod *= pc;
                    }
                    win.push_back(d);
                    if (p_cut > 0.0 && p_prod < p_cut) { ++conf_stops; break; }   // (the next drafts would rarely be kept)
                    if (j + 1 < k && !model.mtp_forward(&d, model.mtp_hidden(), pos + j, 1, false, err)) {
                        std::printf("ERR %s\n", err.c_str()); failed = true; break;
                    }
                }
                if (failed) break;
            }
            // STRATA_DENSE_SPEC_PAD=1: a short draft padded to K drafts (the last token repeated) so every verify pass has
            // one width; only the real drafts can be kept (Kmic-68/llama.cpp's fixed-width verify: there a width change
            // rebuilt the graph - here graphs are kept per width, so this is opt-in, for measuring)
            const int W_real = (int) win.size();
            if (spec_pad && K > 0 && W_real > 1 && W_real < K + 1 && lookup_rounds_before == lookup_rounds)
                while ((int) win.size() < K + 1) win.push_back(win.back());
            const int W = (int) win.size();
            rp_mark(0, t_rp);
            if (!model.forward(win.data(), W, K > 0, K > 0 ? strata::core::DenseModel::kAll : strata::core::DenseModel::kLast, err)) {
                std::printf("ERR %s\n", err.c_str()); failed = true; break;
            }
            rp_mark(1, t_rp);
            int picks[16];
            int acc = 0;
            if (spec_s && block_verify && (int) qd.size() == W - 1 && lookup_rounds_before == lookup_rounds) {
                // block verification (Sun et al., arXiv:2403.10444, Algorithm 2): every prefix of the drafts tested, the
                // longest one accepted kept, the token after it from max(0, p_tau P - Q) - the target's distribution
                // exactly, at least as many drafts kept as the token test (Kmic-68/llama.cpp: +1.8%)
                std::vector<SparseDist> pd;
                if (!topk_rows(model.logits(), cfg.n_vocab, W, pd)) { failed = true; break; }
                const int G = W - 1;
                std::vector<double> pw((size_t) G + 1, 1.0);
                int tau = 0;
                for (int i = 1; i <= G; ++i) {
                    const int x = win[i];
                    const double q = qd[(size_t) i - 1].of(x), pb = pd[(size_t) i - 1].of(x);
                    pw[(size_t) i] = q > 0.0 ? std::min(pw[(size_t) i - 1] * pb / q, 1.0) : 0.0;
                    double h = pw[(size_t) i];
                    if (i < G) {   // the residual mass at the next position over its total with the rest of the block
                        double res = 0.0;
                        const SparseDist& P = pd[(size_t) i];
                        const SparseDist& Q = qd[(size_t) i];
                        for (int a = 0; a < P.n; ++a) res += std::max(pw[(size_t) i] * P.p[a] - Q.of(P.ids[a]), 0.0);
                        h = res + 1.0 - pw[(size_t) i] > 0.0 ? res / (res + 1.0 - pw[(size_t) i]) : 1.0;
                    }
                    if (std::uniform_real_distribution<double>(0.0, 1.0)(srng) <= h) tau = i;
                }
                if (tau == G) {
                    picks[tau] = draw(pd[(size_t) tau], srng);
                } else {
                    SparseDist r = pd[(size_t) tau];
                    const SparseDist& Q = qd[(size_t) tau];
                    double tot = 0.0;
                    for (int a = 0; a < r.n; ++a) { r.p[a] = std::max(0.0, pw[(size_t) tau] * r.p[a] - Q.of(r.ids[a])); tot += r.p[a]; }
                    if (tot > 0.0) { for (int a = 0; a < r.n; ++a) r.p[a] /= tot; picks[tau] = draw(r, srng); }
                    else picks[tau] = draw(pd[(size_t) tau], srng);
                }
                acc = tau;
                draw_counter += W;
            } else if (spec_s && (int) qd.size() == W - 1 && lookup_rounds_before == lookup_rounds) {
                // the exact test: draft j kept with min(1, p_j / q_j); at the first rejection a draw from
                // max(0, p_j - q_j); all kept: a draw from p after the last
                std::vector<SparseDist> pd;
                if (!topk_rows(model.logits(), cfg.n_vocab, W, pd)) { failed = true; break; }
                int j = 0;
                for (; j + 1 < W; ++j) {
                    const int d = win[j + 1];
                    const double pj = pd[(size_t) j].of(d), qj = qd[(size_t) j].of(d);
                    const double u = std::uniform_real_distribution<double>(0.0, 1.0)(srng);
                    if (qj > 0.0 && u < std::min(1.0, pj / qj)) continue;
                    SparseDist r = pd[(size_t) j];
                    double tot = 0.0;
                    for (int i = 0; i < r.n; ++i) { r.p[i] = std::max(0.0, r.p[i] - qd[(size_t) j].of(r.ids[i])); tot += r.p[i]; }
                    if (tot > 0.0) { for (int i = 0; i < r.n; ++i) r.p[i] /= tot; picks[j] = draw(r, srng); }
                    else picks[j] = draw(pd[(size_t) j], srng);
                    break;
                }
                if (j + 1 == W) picks[j] = draw(pd[(size_t) j], srng);
                acc = j;
                draw_counter += W;
            } else {
                if (!sample_rows(model.logits(), W, picks)) { failed = true; break; }
                while (acc + 1 < W_real && win[acc + 1] == picks[acc]) ++acc;
                if (acc_hist && lookup_rounds_before == lookup_rounds && W - 1 <= 16) ++ah[W - 1][acc];
            }
            if (ksel) {   // (this round's commits; a lookup round's not counted)
                ks_commit = acc + 1;
                if (lookup_rounds_before != lookup_rounds) ks_k = -1;
            }
            if (adaptive && !use_df && W > 1) {   // NInfer-style: one more after a full round, back to what was kept + 1
                if (acc == W - 1) k_cur = std::min(K, k_cur + 1);
                else k_cur = std::max(1, std::min(k_cur, acc + 1));
            }
            if (in_think) {   // the budget runs out inside the kept drafts: keep them up to it, then force the end
                const long long room = std::max(0LL, think_budget - produced);
                for (int j = 0; j < acc; ++j)
                    if (win[j + 1] == think_end) { acc = j + 1; break; }
                if (acc > room) { acc = (int) room; }
                const long long before = produced;
                produced += acc;   // (budget_force reads the count after the kept drafts)
                budget_force(picks[acc]);
                produced = before;
            }
            ++rounds; drafted += W - 1; accepted += acc;
            rp_mark(2, t_rp);
            if (K > 0 && use_df) {
                if (!model.commit(acc + 1, err) || !model.dflash_inject(acc + 1, pos, err)) {
                    std::printf("ERR %s\n", err.c_str()); failed = true; break;
                }
            } else if (K > 0) {
                if (!model.commit(acc + 1, err)) { std::printf("ERR %s\n", err.c_str()); failed = true; break; }
                // the MTP block's next rows: the kept window's tokens after the anchor and the new anchor, with the
                // trunk's hidden rows of the anchor .. the last kept draft
                cudaMemcpyAsync(d_hrows, model.hidden_norm(), (size_t) (acc + 1) * cfg.n_embd * 4, cudaMemcpyDeviceToDevice, cs);
                mtp_tok.assign(win.begin() + 1, win.begin() + 1 + acc);
                mtp_tok.push_back(picks[acc]);
                mtp_p0 = pos;
            }
            rp_mark(3, t_rp);
            fed.insert(fed.end(), win.begin(), win.begin() + acc + 1);
            bool more = true;
            for (int j = 0; j < acc && more; ++j) more = emit(win[j + 1]);
            if (more) more = emit(picks[acc]);
            std::fflush(stdout);
            anchor = picks[acc];
            going = more;
        }
        if (failed) { model.reset(); fed.clear(); std::fflush(stdout); continue; }
        if (acc_hist) {   // P(draft j kept | drafts 0 .. j-1 kept), per offered window size
            for (int o = 1; o <= 16; ++o) {
                long long n = 0;
                for (int a = 0; a <= o; ++a) n += ah[o][a];
                if (n == 0) continue;
                std::string line = "strata-dense: acceptance, " + std::to_string(o) + " drafts, " + std::to_string(n) + " rounds:";
                for (int j = 0; j < o; ++j) {
                    long long reach = 0, kept = 0;
                    for (int a = j; a <= o; ++a) reach += ah[o][a];
                    for (int a = j + 1; a <= o; ++a) kept += ah[o][a];
                    char b[48];
                    std::snprintf(b, sizeof b, " p%d %.2f", j + 1, reach ? (double) kept / reach : 0.0);
                    line += b;
                }
                std::fprintf(stderr, "%s\n", line.c_str());
            }
        }
        if (conf_stops > 0)
            std::fprintf(stderr, "strata-dense: %lld rounds drafted fewer by the drafts' confidence (STRATA_DENSE_P_CUM)\n", conf_stops);
        if (rprof && rounds > 0)
            std::fprintf(stderr, "strata-dense: round phases (ms a round): draft %.2f, verify %.2f, accept %.2f, commit %.2f\n",
                         rp[0] / rounds, rp[1] / rounds, rp[2] / rounds, rp[3] / rounds);
        if (cancelled) finish = "cancel";
        const double decode_ms = ms_since(t_dec);
        std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld\n", produced, (long long) n, prompt_ms, decode_ms, finish,
                    accepted, drafted, (long long) resume);
        std::fflush(stdout);
        std::fprintf(stderr, "strata-dense: prompt %lld tokens = %zu reused%s + %lld read in %.0f ms (%.0f tok/s), %lld generated "
                     "in %.0f ms (%.1f tok/s)%s; %s %lld rounds (%lld by lookup), %lld of %lld drafts kept (%.2f tokens a round, drafter %.1f ms a draft)\n",
                     (long long) n, resume, restored ? " (checkpoint)" : "", (long long) (n - (long long) resume), prompt_ms,
                     prompt_ms > 0 ? 1000.0 * (n - (long long) resume) / prompt_ms : 0.0, produced, decode_ms,
                     decode_ms > 0 ? 1000.0 * produced / decode_ms : 0.0, cancelled ? " (cancelled)" : "",
                     use_df ? model.dflash_kind() : "MTP", rounds, lookup_rounds, accepted, drafted, rounds ? (double) (accepted + rounds) / rounds : 0.0,
                     rounds > lookup_rounds ? draft_ms / (rounds - lookup_rounds) : 0.0);
    }
    cudaFree(d_next);
    cudaFree(d_conf);
    cudaFree(d_hist);
    // (exit()'s teardown waits on std::cin, which the reader thread still holds in getline while the server keeps
    // the pipe open: the engine sat there until the server killed it 50 s later.  The OS frees the GPU memory.)
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit(0);
}
