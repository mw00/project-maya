// tools/strata/glm_dsa_llama_logits.cpp - teacher-forced logits from upstream llama.cpp (CPU) for
// the glm-dsa oracle cross-check (BUILD-PLAN B3.1: "llama mode vs llama.cpp CPU on SMALL").
//
//   glm_dsa_llama_logits <model.gguf> <out.f32> <token> [<token>...]
//
// Decodes one token per llama_decode call (the decode path the engine is verified in) and writes
// (n_tokens, n_vocab) float32 row-major logits.  Three settings make the comparison tight enough
// for the 1e-4 logit gate; each is overridable through the environment:
//   * F32 K cache (both the 576-wide MLA latent cache and the 128-wide index-key cache);
//     f16 caches move the logits by about 1e-2.            GLM_DSA_REF_KV=f16 to keep f16.
//   * flash attention off: the CPU flash-attention kernel works at f16 precision internally and
//     moves the logits by about 1e-2 even with an F32 cache. GLM_DSA_REF_FA=1 to enable it.
//   * no extra (repacking) buffer types: on AMX hosts llama.cpp repacks the 3-D attn_k_b/attn_v_b
//     into the AMX buffer, no backend can then run the batched MLA matmul and the scheduler aborts
//     (ggml-backend.cpp GGML_ASSERT(*cur_backend_id != -1), seen at aa94f208 on a Sapphire Rapids
//     container; the test PC's Skylake has no AMX).          GLM_DSA_REF_EXTRA_BUFTS=1 to allow them.
//
// Build against a CPU-only llama.cpp build (aa94f208 for the committed references):
//   cmake -S <llama.cpp> -B <build> -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
//         -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF
//         -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
//   ninja -C <build> llama
//   g++ -O2 -std=c++17 -I <llama.cpp>/include -I <llama.cpp>/ggml/include glm_dsa_llama_logits.cpp
//       -o glm_dsa_llama_logits <build>/src/libllama.a <build>/ggml/src/libggml.a
//       <build>/ggml/src/libggml-cpu.a <build>/ggml/src/libggml-base.a -fopenmp -lm -pthread
// (each command is one line; tools/strata/glm_dsa_refs.py runs all of this)
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool env_is(const char* name, const char* value) {
    const char* v = std::getenv(name);
    return v && std::strcmp(v, value) == 0;
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
    mp.n_gpu_layers = 0;
    mp.use_extra_bufts = env_is("GLM_DSA_REF_EXTRA_BUFTS", "1");
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        std::fprintf(stderr, "glm_dsa_llama_logits: model load failed\n");
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) (n_tok + 8);
    cp.n_batch = cp.n_ubatch = 512;
    cp.no_perf = true;
    if (!env_is("GLM_DSA_REF_KV", "f16")) {
        cp.type_k = GGML_TYPE_F32;
        cp.type_v = GGML_TYPE_F32;
    }
    cp.flash_attn_type = env_is("GLM_DSA_REF_FA", "1") ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                                       : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        std::fprintf(stderr, "glm_dsa_llama_logits: context init failed\n");
        return 1;
    }
    std::FILE* out = std::fopen(argv[2], "wb");
    if (!out) {
        std::fprintf(stderr, "glm_dsa_llama_logits: cannot write %s\n", argv[2]);
        return 1;
    }
    for (size_t t = 0; t < n_tok; ++t) {
        llama_token one = toks[t];
        if (llama_decode(ctx, llama_batch_get_one(&one, 1)) != 0) {
            std::fprintf(stderr, "glm_dsa_llama_logits: decode failed at position %zu\n", t);
            return 1;
        }
        std::fwrite(llama_get_logits_ith(ctx, 0), sizeof(float), (size_t) n_vocab, out);
    }
    std::fclose(out);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    std::fprintf(stderr, "glm_dsa_llama_logits: wrote %zu x %d floats to %s\n", n_tok, n_vocab, argv[2]);
    return 0;
}
