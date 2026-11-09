// tools/llm_logits_dump.cpp - teacher-forced logits from upstream llama.cpp, for the second-oracle
// diff (REPORT-GLM5-FLASH.md §12.1/§12.2): load ANY llama.cpp GGUF, decode explicit token ids with
// logits enabled at every position, and dump (n_tokens, n_vocab) float32 row-major plus the
// per-position argmax to stdout.  CPU-only by default (n_gpu_layers=0).
//
//   llm_logits_dump [options] <model.gguf> <out.bin> <token> [<token>...]
//   llm_logits_dump [options] <model.gguf> <out.bin> @<tokens.txt>     (whitespace-separated ids)
//
// Options (Project glm-dsa B7.3: the GLM-5.3 744B oracle on 3x V100 + CPU experts):
//   --ngl N             layers stored in VRAM (default 0: the CPU backend is the second oracle; -1 = all)
//   --ot PATTERN=BUFT   tensor buffer override, repeatable or comma-separated, as llama.cpp's -ot (exps=CPU)
//   --ubatch N          tokens per llama_decode (default 1: one token per call, the original behaviour)
//   --threads N         CPU threads (default: llama.cpp's)
//   --ctx N             context size (default n_tokens + 8)
//   --fa on|off|auto    flash attention (default auto)
//   --split-mode layer|none   (default layer)    --ts a,b,c  tensor split    --main-gpu N
// LLM_CB_DIR=<dir> dumps every named graph node (the seams) of the LAST decode call; LLM_CB_APPEND=1
// appends each call's data instead, so a per-token run (--ubatch 1) keeps every position.
// LLM_CB_ONLY=<list> (glm-dsa B7.6) keeps only these seams: comma-separated names ("top_k") or exact node names
// ("top_k-26"), the engine's GLM_CB_ONLY syntax (src/core/glm_cb_dump.hpp); an appended 8K run of every seam is
// hundreds of GB.  A node name cb'd twice in one graph (indexer_q, ...) appends twice per call: filter it out.
// LLM_CB_PAD=<n> (with LLM_CB_APPEND) pads every int32 row (top_k: min(n_kv, 2048) cells per token) with -1 to n
// cells, so the rows of calls with a smaller KV view stay aligned (one row per position).
//
// Built out-of-tree against a llama.cpp build:
//   g++ -O2 -std=c++17 -I <llama.cpp>/include -I <llama.cpp>/ggml/include llm_logits_dump.cpp -o llm_logits_dump \
//       -L <build>/bin -L <build>/src -llama -L <build>/ggml/src -lggml -lggml-base -lggml-cpu -lm -pthread
// (tools/strata/llama_to_score_ref.py turns <out.bin> into the engine's STRATA_GLM_SCORE reference files.)
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <list>
#include <map>
#include <string>
#include <vector>

// cb_eval: dump every NAMED node's data (the graph's cb() seams) to <dir>/<name>.bin so the
// per-layer diff against ref/glm.py's dumps can bisect to the first divergent seam.  Non-contiguous
// tensors are gathered element-wise by their strides (a raw memcpy of t->data would be wrong); a
// tensor in device memory is first copied to the host whole (ggml_backend_tensor_get).
static std::string g_cb_dir;
static bool g_cb_append = false;
static std::string g_cb_only;   // LLM_CB_ONLY (glm-dsa B7.6)
static int64_t g_cb_pad = 0;     // LLM_CB_PAD=<n> (glm-dsa B7.6): int32 rows (ne[0]) padded with -1 to n when appending
static bool cb_wanted(const char* name) {
    if (g_cb_only.empty()) return true;
    const std::string n = name;
    const size_t dash = n.rfind('-');
    const bool layered = dash != std::string::npos && dash + 1 < n.size() &&
                         n.find_first_not_of("0123456789", dash + 1) == std::string::npos;
    const std::string seam = layered ? n.substr(0, dash) : n;
    size_t s = 0;
    while (s <= g_cb_only.size()) {
        size_t e = g_cb_only.find(',', s);
        if (e == std::string::npos) e = g_cb_only.size();
        const std::string item = g_cb_only.substr(s, e - s);
        if (!item.empty() && (item == n || item == seam)) return true;
        s = e + 1;
    }
    return false;
}
static void gather_strided(const char* base, const struct ggml_tensor* t, float* dst) {
    const int64_t ne[4] = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
    const size_t  nb[4] = {t->nb[0], t->nb[1], t->nb[2], t->nb[3]};
    for (int64_t i3 = 0; i3 < ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < ne[1]; ++i1) {
                const char* row = base + i3 * nb[3] + i2 * nb[2] + i1 * nb[1];
                for (int64_t i0 = 0; i0 < ne[0]; ++i0) {
                    float v;
                    memcpy(&v, row + i0 * nb[0], sizeof(float));
                    *dst++ = v;
                }
            }
}
static bool eval_cb(struct ggml_tensor* t, bool ask, void*) {
    if (ask) return t->name && t->name[0] && g_cb_dir[0] && cb_wanted(t->name);
    if (!t->data || !t->name || !t->name[0]) return true;
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_I32) return true;
    char path[512];
    std::string name = t->name;
    for (auto& c : name) if (c == '/') c = '_';
    std::snprintf(path, sizeof(path), "%s/%s.%s", g_cb_dir.c_str(), name.c_str(),
                  t->type == GGML_TYPE_I32 ? "i32" : "f32");
    const int64_t n = t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];
    const size_t bytes = (size_t) n * 4;
    // a GPU-resident node (n_gpu_layers > 0) is read through the backend, never through t->data
    std::vector<char> host;
    const char* base = (const char*) t->data;
    if (t->buffer && !ggml_backend_buffer_is_host(t->buffer)) {
        host.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, host.data(), 0, host.size());
        base = host.data();
    }
    std::FILE* f = std::fopen(path, g_cb_append ? "ab" : "wb");
    if (!f) return true;
    if (g_cb_append && g_cb_pad > t->ne[0] && t->type == GGML_TYPE_I32) {   // glm-dsa B7.6: fixed-width id rows
        std::vector<int32_t> ids((size_t) n);
        gather_strided(base, t, (float*) ids.data());   // 4-byte copies: the int32 bits pass through unchanged
        std::vector<int32_t> row((size_t) g_cb_pad, -1);
        for (int64_t r = 0; r < n / t->ne[0]; ++r) {
            std::fill(row.begin(), row.end(), -1);
            std::copy(ids.begin() + r * t->ne[0], ids.begin() + (r + 1) * t->ne[0], row.begin());
            std::fwrite(row.data(), 4, row.size(), f);
        }
        std::fclose(f);
        return true;
    }
    if (!ggml_is_contiguous(t)) {
        std::vector<float> buf((size_t) n);
        gather_strided(base, t, buf.data());
        std::fwrite(buf.data(), 1, bytes, f);
    } else {
        std::fwrite(base, 1, bytes, f);
    }
    std::fclose(f);
    return true;
}

static void usage(const char* a0) {
    std::fprintf(stderr,
                 "usage: %s [--ngl N] [--ot PATTERN=BUFT[,...]] [--ubatch N] [--threads N] [--ctx N] [--fa on|off|auto]\n"
                 "          [--split-mode layer|none] [--ts a,b,c] [--main-gpu N] <model.gguf> <out.bin> <token>... | @tokens.txt\n",
                 a0);
}

// -ot PATTERN=BUFT[,PATTERN=BUFT...] against the loaded backends' buffer types (as common/arg.cpp does)
static bool add_overrides(const std::string& value, std::vector<llama_model_tensor_buft_override>& out) {
    static std::list<std::string> keep;   // the patterns must outlive the model load
    std::map<std::string, ggml_backend_buffer_type_t> bufts;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_buffer_type_t b = ggml_backend_dev_buffer_type(ggml_backend_dev_get(i));
        if (b) bufts[ggml_backend_buft_name(b)] = b;
    }
    size_t start = 0;
    while (start <= value.size()) {
        const size_t comma = value.find(',', start);
        const std::string one = value.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        const size_t eq = one.find('=');
        if (eq == std::string::npos) {
            std::fprintf(stderr, "llm_logits_dump: --ot %s: expected PATTERN=BUFFER_TYPE\n", one.c_str());
            return false;
        }
        const std::string bt = one.substr(eq + 1);
        if (!bufts.count(bt)) {
            std::fprintf(stderr, "llm_logits_dump: unknown buffer type %s; available:", bt.c_str());
            for (const auto& kv : bufts) std::fprintf(stderr, " %s", kv.first.c_str());
            std::fprintf(stderr, "\n");
            return false;
        }
        keep.push_back(one.substr(0, eq));
        out.push_back({keep.back().c_str(), bufts[bt]});
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return true;
}

int main(int argc, char** argv) {
    int ngl = 0, ubatch = 1, threads = 0, n_ctx = 0, main_gpu = 0;
    std::string fa = "auto", split = "layer";
    std::vector<std::string> ot_args;
    std::vector<float> ts;
    int i = 1;
    for (; i < argc && argv[i][0] == '-' && argv[i][1] == '-'; ++i) {
        const std::string a = argv[i];
        if (i + 1 >= argc) {
            usage(argv[0]);
            return 2;
        }
        const char* v = argv[++i];
        if (a == "--ngl") ngl = std::atoi(v);
        else if (a == "--ot") ot_args.push_back(v);
        else if (a == "--ubatch") ubatch = std::max(1, std::atoi(v));
        else if (a == "--threads") threads = std::atoi(v);
        else if (a == "--ctx") n_ctx = std::atoi(v);
        else if (a == "--fa") fa = v;
        else if (a == "--split-mode") split = v;
        else if (a == "--main-gpu") main_gpu = std::atoi(v);
        else if (a == "--ts") {
            for (const char* p = v; *p;) {
                ts.push_back((float) std::atof(p));
                const char* c = std::strchr(p, ',');
                if (!c) break;
                p = c + 1;
            }
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (argc - i < 3) {
        usage(argv[0]);
        return 2;
    }
    const char* model_path = argv[i];
    const char* out_path = argv[i + 1];
    std::vector<llama_token> toks;
    for (int k = i + 2; k < argc; ++k) {
        if (argv[k][0] == '@') {
            std::ifstream tf(argv[k] + 1);
            if (!tf) {
                std::fprintf(stderr, "llm_logits_dump: cannot read %s\n", argv[k] + 1);
                return 1;
            }
            long id;
            while (tf >> id) toks.push_back((llama_token) id);
        } else {
            toks.push_back((llama_token) std::atoi(argv[k]));
        }
    }
    const size_t n_tok = toks.size();
    if (n_tok == 0) {
        std::fprintf(stderr, "llm_logits_dump: no tokens\n");
        return 2;
    }

    llama_backend_init();
    ggml_backend_load_all();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;                     // 0 (default): the CPU backend IS the second oracle
    mp.main_gpu = main_gpu;
    mp.split_mode = split == "none" ? LLAMA_SPLIT_MODE_NONE : LLAMA_SPLIT_MODE_LAYER;
    std::vector<float> tsplit;
    if (!ts.empty()) {
        tsplit.assign(llama_max_devices(), 0.0f);
        for (size_t d = 0; d < ts.size() && d < tsplit.size(); ++d) tsplit[d] = ts[d];
        mp.tensor_split = tsplit.data();
    }
    std::vector<llama_model_tensor_buft_override> ovr;
    for (const auto& s : ot_args)
        if (!add_overrides(s, ovr)) return 2;
    if (!ovr.empty()) {
        ovr.push_back({nullptr, nullptr});
        mp.tensor_buft_overrides = ovr.data();
    }
    llama_model* model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        std::fprintf(stderr, "llm_logits_dump: model load failed\n");
        return 1;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = (uint32_t) (n_ctx > 0 ? n_ctx : (int) n_tok + 8);
    cp.n_batch  = (uint32_t) ubatch;
    cp.n_ubatch = (uint32_t) ubatch;           // every position's logits: one output per token of the ubatch
    cp.no_perf  = true;
    if (threads > 0) {
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
    }
    cp.flash_attn_type = fa == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                       : fa == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
    g_cb_dir = std::getenv("LLM_CB_DIR") ? std::getenv("LLM_CB_DIR") : "";
    g_cb_append = std::getenv("LLM_CB_APPEND") != nullptr;
    g_cb_only = std::getenv("LLM_CB_ONLY") ? std::getenv("LLM_CB_ONLY") : "";
    g_cb_pad = std::getenv("LLM_CB_PAD") ? std::atoll(std::getenv("LLM_CB_PAD")) : 0;
    if (g_cb_dir[0]) cp.cb_eval = eval_cb;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        std::fprintf(stderr, "llm_logits_dump: context init failed\n");
        return 1;
    }

    // --ubatch 1 (default): ONE token per llama_decode call.  The single-ubatch variant aborts inside
    // llama.cpp's graph_reserve (ggml_set_rows shape assert under resolve_fused_ops) at 272aad8b for the
    // Flash model; streaming is also the mode the glm-dsa runner is verified in, so per-token is the
    // comparison that matters.  --ubatch N > 1 runs llama.cpp's batched prompt path (N tokens per call,
    // logits at every position), which is what a fast 2K-token GLM-5.3 reference dump on CUDA needs.
    std::FILE* out = std::fopen(out_path, "wb");
    if (!out) {
        std::fprintf(stderr, "llm_logits_dump: cannot write %s\n", out_path);
        return 1;
    }
    llama_batch batch = llama_batch_init(ubatch, 0, 1);
    for (size_t t0 = 0; t0 < n_tok; t0 += (size_t) ubatch) {
        const int n = (int) std::min<size_t>((size_t) ubatch, n_tok - t0);
        batch.n_tokens = n;
        for (int j = 0; j < n; ++j) {
            batch.token[j] = toks[t0 + j];
            batch.pos[j] = (llama_pos) (t0 + j);
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = 1;
        }
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "llm_logits_dump: decode failed at positions %zu..%zu\n", t0, t0 + n - 1);
            llama_batch_free(batch);
            return 1;
        }
        for (int j = 0; j < n; ++j) {
            const float* lg = llama_get_logits_ith(ctx, j);
            std::fwrite(lg, sizeof(float), (size_t) n_vocab, out);
            int am = 0;
            for (int v = 1; v < n_vocab; ++v)
                if (lg[v] > lg[am]) am = v;
            std::printf("pos %zu tok %d argmax %d (%.6f)\n", t0 + j, toks[t0 + j], am, lg[am]);
        }
    }
    llama_batch_free(batch);
    std::fclose(out);
    llama_free(ctx);
    llama_model_free(model);
    std::fprintf(stderr, "llm_logits_dump: wrote %zu x %d floats to %s\n", n_tok, n_vocab, out_path);
    return 0;
}
