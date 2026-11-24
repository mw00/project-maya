// tools/strata/ik_dsa_logits.cpp - glm-dsa B14.10: teacher-forced logits and top-k from ik_llama.cpp, the THIRD oracle
// for glm-dsa triage (docs/strata/engine/oracle.md section "ik_llama.cpp as a third oracle").  Built and run by
// tools/strata/ik_oracle.sh; the mainline counterpart is tools/strata/glm_dsa_llama_logits.cpp.
//
//   ik_dsa_logits <model.gguf> <out.f32> <token> [<token>...]
//
// Decodes one token per llama_decode call and writes (n_tokens, n_vocab) float32 row-major logits, plus
// <out.f32>.topk.txt: per position the IK_TOPK (default 20) best ids and logits ("pos id:logit id:logit ...").
// Settings (environment; the script sets them from its flags):
//   IK_DSA=1          DSA sparse attention on (ik's --dsa; default 1).  0: dense MLA (the identity selection).
//   IK_NGL=<n>        layers on the GPU (default 0: CPU only, so the indexer scores stay f32, see the deviations).
//   IK_MLA=<n>        ik's -mla mode (default 1; 1, 2 and 3 are independently written paths).
//   IK_CTK=f32|f16|q8_0  the MLA cache type (default f32: as the mainline oracle, f16 moves logits by about 1e-2).
//   IK_IDX_CTK=f16|f32   the indexer key cache type (default f16, ik's default; keep it off q8_0).
//   IK_TOPK=<k>       top-k written per position (default 20).
//   IK_THREADS=<n>    CPU threads (default: ik's).
// ik's own debug env knobs pass through: DSA_SINK=0 turns the sink boost off, DSA_HADAMARD_DISABLE=1 the indexer
// Walsh-Hadamard rotation.  Flash attention is always off (on V100 ik's FA kernel refuses 576/512 and moves attention
// to the CPU).  Build: see tools/strata/ik_oracle.sh (links ik's libllama / libggml from a CPU or CUDA build).
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

static int env_int(const char* k, int d) {
    const char* v = std::getenv(k);
    return v && v[0] ? std::atoi(v) : d;
}
static ggml_type env_type(const char* k, ggml_type d) {
    const char* v = std::getenv(k);
    if (!v || !v[0]) return d;
    if (!std::strcmp(v, "f32")) return GGML_TYPE_F32;
    if (!std::strcmp(v, "f16")) return GGML_TYPE_F16;
    if (!std::strcmp(v, "q8_0")) return GGML_TYPE_Q8_0;
    std::fprintf(stderr, "ik_dsa_logits: %s=%s is not f32, f16 or q8_0\n", k, v);
    std::exit(2);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <out.f32> <token> [<token>...]\n", argv[0]);
        return 2;
    }
    std::vector<llama_token> toks;
    for (int i = 3; i < argc; ++i) toks.push_back((llama_token) std::atoi(argv[i]));
    const size_t n_tok = toks.size();

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = env_int("IK_NGL", 0);
    mp.split_mode = LLAMA_SPLIT_MODE_LAYER;   // never -sm graph: it ignores --dsa
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        std::fprintf(stderr, "ik_dsa_logits: model load failed\n");
        return 1;
    }
    const int n_vocab = llama_n_vocab(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) (n_tok + 8);
    cp.n_batch = cp.n_ubatch = 512;
    cp.flash_attn = false;
    cp.mla_attn = env_int("IK_MLA", 1);
    cp.dsa = env_int("IK_DSA", 1) != 0;
    cp.type_k = env_type("IK_CTK", GGML_TYPE_F32);
    cp.type_v = cp.type_k;
    cp.idx_type_k = env_type("IK_IDX_CTK", GGML_TYPE_F16);
    if (const int th = env_int("IK_THREADS", 0); th > 0) cp.n_threads = cp.n_threads_batch = (uint32_t) th;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        std::fprintf(stderr, "ik_dsa_logits: context init failed\n");
        return 1;
    }
    std::FILE* out = std::fopen(argv[2], "wb");
    const std::string topk_path = std::string(argv[2]) + ".topk.txt";
    std::FILE* tk = std::fopen(topk_path.c_str(), "w");
    if (!out || !tk) {
        std::fprintf(stderr, "ik_dsa_logits: cannot write %s\n", argv[2]);
        return 1;
    }
    const int K = std::max(1, std::min(env_int("IK_TOPK", 20), n_vocab));
    std::vector<int> idx((size_t) n_vocab);
    for (size_t t = 0; t < n_tok; ++t) {
        llama_token one = toks[t];
        if (llama_decode(ctx, llama_batch_get_one(&one, 1, (llama_pos) t, 0)) != 0) {
            std::fprintf(stderr, "ik_dsa_logits: decode failed at position %zu\n", t);
            return 1;
        }
        const float* lg = llama_get_logits_ith(ctx, 0);
        std::fwrite(lg, sizeof(float), (size_t) n_vocab, out);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int b) {
            return lg[a] > lg[b] || (lg[a] == lg[b] && a < b);
        });
        std::fprintf(tk, "%zu", t);
        for (int k = 0; k < K; ++k) std::fprintf(tk, " %d:%.6g", idx[(size_t) k], (double) lg[idx[(size_t) k]]);
        std::fprintf(tk, "\n");
    }
    std::fclose(out);
    std::fclose(tk);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    std::fprintf(stderr, "ik_dsa_logits: wrote %zu x %d floats to %s (top-%d in %s)\n", n_tok, n_vocab, argv[2], K,
                 topk_path.c_str());
    return 0;
}
