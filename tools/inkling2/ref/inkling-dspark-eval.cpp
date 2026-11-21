// inkling-dspark-eval: the acceptance rate of Inkling 2 Small's DSpark drafter, measured offline against the target's
// own greedy continuation (maya-inkling2, docs/INKLING2_DSPARK.md).  Built in a worktree of llama.cpp PR #32218.
//
// The target reads each prompt and decodes INK_N greedy tokens once, keeping every position's tap candidates
// (llama_get_embeddings_layer_inp: the PR's slots 5..41 and 48, plus - INK_TAP_EXTRA=1, this worktree's patch - slot 0
// = l_out-47 raw, 1 = the head's mix before its norm, id-3 = layer id's mixed Q input).  A greedy target's tokens do
// not depend on the drafter, so each drafter variant (INK_VARIANTS) is then run alone over the same tokens: the
// contexts injected position by position and a block drafted after EVERY generated position n, its drafts compared
// with the target's tokens at n+1 .. n+7.  Draft i is "reached" when drafts 1..i-1 matched and "kept" when it matches
// too - what a greedy verify window keeps.  The real rounds (the next block starting after the kept drafts) are
// replayed from the same data.
//
// env: INK_PROMPTS "name=file,name=file"; INK_N tokens generated a prompt; INK_OUT a folder;
//      INK_VARIANTS "name|gguf|slot,slot,..(8)|ENV=V;ENV=V" separated by newlines
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "speculative.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<std::string> split(const std::string & s, char c) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, c)) out.push_back(item);
    return out;
}

struct Run {
    std::string name;
    std::vector<llama_token> tok;   // the prompt, then the greedy tokens
    int n_prompt = 0;
    std::vector<std::vector<float>> feat;   // [slot index][pos * n_embd]
    double prompt_s = 0, decode_s = 0;
};

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) return 1;
    const char * e_prompts = getenv("INK_PROMPTS");
    const char * e_var = getenv("INK_VARIANTS");
    const std::string out_dir = getenv("INK_OUT") ? getenv("INK_OUT") : ".";
    const int n_gen = getenv("INK_N") ? atoi(getenv("INK_N")) : 128;
    if (!e_prompts || !e_var) { fprintf(stderr, "INK_PROMPTS and INK_VARIANTS\n"); return 1; }

    llama_backend_init();
    llama_numa_init(params.numa);
    auto init_tgt = common_init_from_params(params);
    llama_model * model_tgt = init_tgt->model();
    llama_context * ctx_tgt = init_tgt->context();
    if (!model_tgt || !ctx_tgt) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int E = llama_model_n_embd(model_tgt);
    const int n_layer = llama_model_n_layer(model_tgt);

    const bool extra = getenv("INK_TAP_EXTRA") != nullptr;
    std::vector<int> slots;
    for (int id = 5; id < n_layer; id += 6) slots.push_back(id);
    slots.push_back(n_layer);
    if (extra) {
        slots.push_back(0);
        slots.push_back(1);
        for (int id = 5; id < n_layer; id += 6) slots.push_back(id - 3);
    }
    for (int s : slots) llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) s, true);
    auto slot_index = [&](int s) { for (size_t i = 0; i < slots.size(); ++i) if (slots[i] == s) return (int) i; return -1; };

    // ---- the target: each prompt and its greedy continuation, the taps of every position
    std::vector<Run> runs;
    for (const std::string & pr : split(e_prompts, ',')) {
        const size_t eq = pr.find('=');
        Run r;
        r.name = pr.substr(0, eq);
        std::ifstream f(pr.substr(eq + 1), std::ios::binary);
        std::stringstream ss;
        ss << f.rdbuf();
        std::string text = ss.str();
        while (!text.empty() && text.back() == '\n') text.pop_back();
        r.tok = common_tokenize(ctx_tgt, text, false, true);
        r.n_prompt = (int) r.tok.size();
        r.feat.assign(slots.size(), {});
        {   // a run already on disk (the same prompt): its tokens and taps, the target not run again
            const std::string d = out_dir + "/" + r.name;
            std::ifstream tf(d + "/tokens.txt");
            int np = 0;
            if (tf >> np && np == r.n_prompt) {
                std::vector<llama_token> tk;
                llama_token t;
                while (tf >> t) tk.push_back(t);
                bool same = (int) tk.size() > np && std::equal(r.tok.begin(), r.tok.end(), tk.begin());
                for (size_t i = 0; same && i < slots.size(); ++i) {
                    std::ifstream bf(d + "/tap-slot" + std::to_string(slots[i]) + ".bin", std::ios::binary);
                    r.feat[i].resize(tk.size() * (size_t) E);
                    bf.read((char *) r.feat[i].data(), (std::streamsize) (r.feat[i].size() * sizeof(float)));
                    same = bf.gcount() == (std::streamsize) (r.feat[i].size() * sizeof(float));
                }
                if (same) {
                    r.tok = tk;
                    fprintf(stderr, "TARGET %s: %d prompt + %d greedy tokens from %s\n", r.name.c_str(), np, (int) tk.size() - np, d.c_str());
                    runs.push_back(std::move(r));
                    continue;
                }
                r.feat.assign(slots.size(), {});
            }
        }
        llama_memory_clear(llama_get_memory(ctx_tgt), true);
        auto grab = [&](int rows) {
            for (size_t i = 0; i < slots.size(); ++i) {
                const float * p = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) slots[i]);
                if (!p) { fprintf(stderr, "slot %d not extracted\n", slots[i]); exit(1); }
                r.feat[i].insert(r.feat[i].end(), p, p + (size_t) rows * E);
            }
        };
        int64_t t0 = ggml_time_us();
        {
            common_batch b(ctx_tgt);
            for (int i = 0; i < r.n_prompt; ++i) b.add(r.tok[i], i, 0, i == r.n_prompt - 1);
            if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, b.get()) != 0) { fprintf(stderr, "prompt decode failed\n"); return 1; }
            grab(r.n_prompt);
        }
        r.prompt_s = (ggml_time_us() - t0) / 1e6;
        t0 = ggml_time_us();
        std::string textout;
        for (int g = 0; g < n_gen; ++g) {
            const float * lg = llama_get_logits_ith(ctx_tgt, -1);
            int best = 0;
            for (int v = 1; v < n_vocab; ++v) if (lg[v] > lg[best]) best = v;
            r.tok.push_back(best);
            textout += common_token_to_piece(ctx_tgt, best, true);
            // (the token is read too: the drafter needs its position's taps to draft after it)
            common_batch b(ctx_tgt);
            b.add(best, (int) r.tok.size() - 1, 0, true);
            if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, b.get()) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
            grab(1);
            if (llama_vocab_is_eog(vocab, best)) break;
        }
        r.decode_s = (ggml_time_us() - t0) / 1e6;
        const int ng = (int) r.tok.size() - r.n_prompt;
        fprintf(stderr, "TARGET %s: prompt %d tokens %.1f s (%.1f tok/s), %d greedy tokens %.1f s (%.2f tok/s)\n", r.name.c_str(),
                r.n_prompt, r.prompt_s, r.n_prompt / r.prompt_s, ng, r.decode_s, ng / r.decode_s);
        fprintf(stderr, "TEXT %s: %s\n", r.name.c_str(), textout.c_str());
        {   // the reference for the engine's own drafter tests
            const std::string d = out_dir + "/" + r.name;
            (void) !system(("mkdir -p '" + d + "'").c_str());
            std::ofstream tf(d + "/tokens.txt");
            tf << r.n_prompt << "\n";
            for (auto t : r.tok) tf << t << " ";
            tf << "\n";
            for (size_t i = 0; i < slots.size(); ++i) {
                std::ofstream bf(d + "/tap-slot" + std::to_string(slots[i]) + ".bin", std::ios::binary);
                bf.write((const char *) r.feat[i].data(), (std::streamsize) (r.feat[i].size() * sizeof(float)));
            }
            std::ofstream xf(d + "/text.txt");
            xf << textout;
        }
        runs.push_back(std::move(r));
        fflush(stderr);
    }

    // ---- the drafter variants
    std::ofstream res(out_dir + "/results.jsonl", std::ios::app);
    for (const std::string & vline : split(e_var, '\n')) {
        if (vline.empty()) continue;
        const auto vp = split(vline, '|');
        if (vp.size() < 3) { fprintf(stderr, "bad variant: %s\n", vline.c_str()); continue; }
        std::vector<int> vs;
        for (const auto & s : split(vp[2], ',')) vs.push_back(slot_index(atoi(s.c_str())));
        bool ok = vs.size() == 8;
        for (int s : vs) ok = ok && s >= 0;
        if (!ok) { fprintf(stderr, "variant %s: slots not extracted\n", vp[0].c_str()); continue; }
        std::vector<std::string> envs = vp.size() > 3 ? split(vp[3], ';') : std::vector<std::string>{};
        for (const auto & kv : envs) { const size_t q = kv.find('='); if (q != std::string::npos) setenv(kv.substr(0, q).c_str(), kv.substr(q + 1).c_str(), 1); }

        common_params pv = params;
        pv.speculative.draft.mparams.path = vp[1];
        common_params pd = common_base_params_to_speculative(pv);
        auto spec_init = common_speculative_init_from_params(pd, model_tgt, ctx_tgt);
        llama_context * ctx_dft = spec_init ? spec_init->context() : nullptr;
        if (!ctx_dft) { fprintf(stderr, "variant %s: the drafter did not load\n", vp[0].c_str()); return 1; }
        llama_set_embeddings_nextn(ctx_dft, true, false);
        llama_set_causal_attn(ctx_dft, false);
        const int n_ub = (int) llama_n_ubatch(ctx_dft);
        const int NT = 8, EN = NT * E;
        const llama_token mask = llama_vocab_mask(llama_model_get_vocab(llama_get_model(ctx_dft)));
        const int nv_d = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx_dft)));

        for (const Run & r : runs) {
            llama_memory_clear(llama_get_memory(ctx_dft), true);
            const int n_all = (int) r.tok.size();
            std::vector<float> fbuf, frnn((size_t) n_all * E);
            common_batch bi(ctx_dft), bd(ctx_dft);
            double inj_ms = 0, dr_ms = 0;
            auto inject = [&](int p0, int n) {
                const int64_t t0 = ggml_time_us();
                for (int off = 0; off < n; off += n_ub) {
                    const int nc = std::min(n_ub, n - off);
                    fbuf.resize((size_t) nc * EN);
                    for (int k = 0; k < NT; ++k)
                        for (int i = 0; i < nc; ++i)
                            memcpy(fbuf.data() + (size_t) i * EN + (size_t) k * E, r.feat[vs[k]].data() + (size_t) (p0 + off + i) * E, E * sizeof(float));
                    bi.clear();
                    for (int i = 0; i < nc; ++i) {
                        const llama_pos p = p0 + off + i;
                        const llama_pos pa[4] = {p, p, p, 0};
                        bi.add_embd({fbuf.data() + (size_t) i * EN, 1, (size_t) EN}, pa, 0, false);
                    }
                    if (llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, bi.get()) != 0) { fprintf(stderr, "inject failed\n"); exit(1); }
                    const float * pj = llama_get_embeddings_nextn(ctx_dft);
                    memcpy(frnn.data() + (size_t) (p0 + off) * E, pj, (size_t) nc * E * sizeof(float));
                }
                inj_ms += (ggml_time_us() - t0) / 1e3;
            };
            inject(0, r.n_prompt);
            long long reached[7] = {0}, kept[7] = {0}, mh[8] = {0};
            long long rounds = 0, racc = 0, rreach[7] = {0}, rkept[7] = {0};
            long long cal_n[7][10] = {{0}}, cal_k[7][10] = {{0}};
            int next_round = r.n_prompt, blocks = 0;
            std::ofstream df(out_dir + "/" + r.name + "/drafts-" + vp[0] + ".txt");
            for (int n = r.n_prompt; n + 7 < n_all; ++n) {
                if (n > r.n_prompt) inject(n - 1, 1);
                const int64_t t0 = ggml_time_us();
                bd.clear();
                for (int i = 0; i < 8; ++i) {
                    const int idx = bd.add(i == 0 ? r.tok[n] : mask, n + i, 0, true);
                    bd.set_embd_state(idx, {frnn.data() + (size_t) (n - 1) * E, 1, (size_t) E});
                }
                if (llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, bd.get()) != 0) { fprintf(stderr, "draft failed\n"); exit(1); }
                int d[7];
                float pm[7];
                for (int i = 1; i < 8; ++i) {
                    const float * lg = llama_get_logits_ith(ctx_dft, i);
                    int best = 0;
                    for (int v = 1; v < nv_d; ++v) if (lg[v] > lg[best]) best = v;
                    double sum = 0;
                    for (int v = 0; v < nv_d; ++v) sum += std::exp((double) lg[v] - lg[best]);
                    d[i - 1] = best;
                    pm[i - 1] = (float) (1.0 / sum);
                }
                dr_ms += (ggml_time_us() - t0) / 1e3;
                if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, n, -1)) { fprintf(stderr, "seq_rm of the block failed at %d\n", n); exit(1); }
                int m = 0;
                while (m < 7 && d[m] == r.tok[n + 1 + m]) ++m;
                ++mh[m];
                ++blocks;
                for (int i = 0; i < 7 && i <= m; ++i) {
                    ++reached[i];
                    const int b = std::min(9, (int) (pm[i] * 10));
                    ++cal_n[i][b];
                    if (i < m) { ++kept[i]; ++cal_k[i][b]; }
                }
                if (n == next_round) {
                    ++rounds;
                    racc += m;
                    for (int i = 0; i < 7 && i <= m; ++i) { ++rreach[i]; if (i < m) ++rkept[i]; }
                    next_round = n + m + 1;
                }
                df << n << " " << m;
                for (int i = 0; i < 7; ++i) df << " " << d[i] << ":" << pm[i];
                df << "\n";
            }
            std::ostringstream o;
            o << "{\"variant\":\"" << vp[0] << "\",\"prompt\":\"" << r.name << "\",\"n_prompt\":" << r.n_prompt << ",\"blocks\":" << blocks;
            auto arr = [&](const char * k, const long long * a, int n) { o << ",\"" << k << "\":["; for (int i = 0; i < n; ++i) o << (i ? "," : "") << a[i]; o << "]"; };
            arr("reached", reached, 7); arr("kept", kept, 7); arr("m_hist", mh, 8);
            o << ",\"rounds\":" << rounds << ",\"round_accepted\":" << racc;
            arr("round_reached", rreach, 7); arr("round_kept", rkept, 7);
            o << ",\"cal_n\":["; for (int i = 0; i < 7; ++i) { o << (i ? "," : "") << "["; for (int b = 0; b < 10; ++b) o << (b ? "," : "") << cal_n[i][b]; o << "]"; } o << "]";
            o << ",\"cal_k\":["; for (int i = 0; i < 7; ++i) { o << (i ? "," : "") << "["; for (int b = 0; b < 10; ++b) o << (b ? "," : "") << cal_k[i][b]; o << "]"; } o << "]";
            o << ",\"inject_ms\":" << inj_ms << ",\"draft_ms_per_block\":" << (blocks ? dr_ms / blocks : 0) << "}";
            res << o.str() << std::endl;
            double mean = 0;
            for (int i = 0; i < 8; ++i) mean += (double) i * mh[i];
            fprintf(stderr, "RESULT %s / %s: %d blocks, mean kept %.2f of 7; by position", vp[0].c_str(), r.name.c_str(), blocks, blocks ? mean / blocks : 0);
            for (int i = 0; i < 7; ++i) fprintf(stderr, " %lld/%lld", kept[i], reached[i]);
            fprintf(stderr, "; rounds %lld accepted %lld; draft %.1f ms a block\n", rounds, racc, blocks ? dr_ms / blocks : 0);
            fflush(stderr);
        }
        spec_init.reset();
        for (const auto & kv : envs) { const size_t q = kv.find('='); if (q != std::string::npos) unsetenv(kv.substr(0, q).c_str()); }
    }
    llama_backend_free();
    return 0;
}
