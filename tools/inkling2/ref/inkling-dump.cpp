// Reference dump for the maya-inkling2 port: runs a prompt (one batch) and then N greedy
// decode steps, saving every graph tensor whose name matches INK_DUMP_FILTER (a regex,
// default: per-layer outputs, router picks, final norm and logits) as raw F32/I32 files.
//
//   INK_DUMP_DIR=/out INK_DUMP_STEPS=4 llama-inkling-dump -m model.gguf -p "prompt" [-cmoe ...]
//
// Layout: <dir>/tokens.txt (prompt ids, then the greedy ids), <dir>/index.jsonl (one line a
// tensor: step, name, type, ne[4]), <dir>/s<step>/<name>.bin (contiguous, ne0 fastest).
// Step 0 is the prompt batch, step k>0 is the k-th single-token decode.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

struct dump_state {
    std::string   dir;
    std::regex    filter;
    int           step = 0;
    std::ofstream index;
    std::vector<uint8_t> raw;
};

static float to_f32(ggml_type type, const uint8_t * p) {
    switch (type) {
        case GGML_TYPE_F32:  return *(const float *) p;
        case GGML_TYPE_F16:  return ggml_fp16_to_fp32(*(const ggml_fp16_t *) p);
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(*(const ggml_bf16_t *) p);
        default:             return 0.0f;
    }
}

static bool dump_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (dump_state *) user_data;
    const bool want = std::regex_search(t->name, st->filter) &&
        (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16 || t->type == GGML_TYPE_I32);
    if (ask) {
        return want;
    }
    if (!want) {
        return true;
    }

    // fetch the tensor's whole byte range (views included), then gather it contiguous on the host
    const size_t nbytes = ggml_nbytes(t);
    st->raw.resize(nbytes);
    if (ggml_backend_buffer_is_host(t->buffer)) {
        memcpy(st->raw.data(), t->data, nbytes);
    } else {
        ggml_backend_tensor_get(t, st->raw.data(), 0, nbytes);
    }

    const bool is_i32 = t->type == GGML_TYPE_I32;
    const int64_t n = ggml_nelements(t);
    std::vector<uint8_t> out((size_t) n * 4);
    int64_t k = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
    for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
    for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
    for (int64_t i0 = 0; i0 < t->ne[0]; ++i0, ++k) {
        const uint8_t * p = st->raw.data() + i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3];
        if (is_i32) {
            memcpy(out.data() + 4*k, p, 4);
        } else {
            const float v = to_f32(t->type, p);
            memcpy(out.data() + 4*k, &v, 4);
        }
    }

    const std::string sdir = st->dir + "/s" + std::to_string(st->step);
    std::filesystem::create_directories(sdir);
    std::string fname = t->name;
    for (char & c : fname) {
        if (c == '/' || c == ' ' || c == '(' || c == ')') c = '_';
    }
    std::ofstream f(sdir + "/" + fname + ".bin", std::ios::binary);
    f.write((const char *) out.data(), out.size());

    st->index << "{\"step\":" << st->step << ",\"name\":\"" << t->name << "\",\"file\":\"s" << st->step << "/" << fname
              << ".bin\",\"type\":\"" << (is_i32 ? "i32" : "f32") << "\",\"op\":\"" << ggml_op_desc(t)
              << "\",\"ne\":[" << t->ne[0] << "," << t->ne[1] << "," << t->ne[2] << "," << t->ne[3] << "]}\n";
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    const char * env_dir    = getenv("INK_DUMP_DIR");
    const char * env_filter = getenv("INK_DUMP_FILTER");
    const char * env_steps  = getenv("INK_DUMP_STEPS");

    dump_state st;
    st.dir    = env_dir ? env_dir : "inkling-dump";
    st.filter = std::regex(env_filter ? env_filter :
        "^(inp_embd|norm-0|attn_out-|ffn_inp-|ffn_out-|l_out-|ffn_moe_topk-|ffn_moe_weights_norm-|ffn_moe_weights_scaled-|result_norm|result_output)");
    const int n_steps = env_steps ? atoi(env_steps) : 4;
    std::filesystem::create_directories(st.dir);
    st.index.open(st.dir + "/index.jsonl");

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = dump_cb;
    params.cb_eval_user_data = &st;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    std::ofstream ftok(st.dir + "/tokens.txt");
    for (auto id : tokens) ftok << id << " ";
    ftok << "\n";
    LOG_INF("prompt tokens: %zu\n", tokens.size());

    {
        common_batch batch = common_batch_get_one(ctx, tokens);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            LOG_ERR("prompt eval failed\n");
            return 1;
        }
    }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    for (int s = 1; s <= n_steps; ++s) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        llama_token best = 0;
        for (int i = 1; i < n_vocab; ++i) {
            if (logits[i] > logits[best]) best = i;
        }
        const std::string piece = common_token_to_piece(ctx, best);
        LOG_INF("step %d: token %d '%s'\n", s, best, piece.c_str());
        ftok << best << " ";
        ftok.flush();

        st.step = s;
        std::vector<llama_token> one = { best };
        common_batch batch = common_batch_get_one(ctx, one);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            LOG_ERR("decode %d failed\n", s);
            return 1;
        }
    }
    ftok << "\n";
    st.index.flush();

    llama_backend_free();
    return 0;
}
