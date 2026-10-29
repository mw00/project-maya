// src/metal/main.cpp - maya-metal: GLM-5.3-Flash on Apple Silicon.
//
//   maya-metal --gguf MODEL.gguf --serve [--max-context N] ...   the engine `maya.py` / serve/server.py drive
//   maya-metal --gguf MODEL.gguf --ids 1,2,3 [--n 64]            greedy continuation of token ids, with timings
#include "glm_metal.hpp"
#include "spec.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace maya::metal {
int serve(GlmMetal& model, const std::vector<std::string>& args);
int mtp_draft_cap();
}

namespace {

void usage() {
    std::fprintf(stderr,
                 "usage: maya-metal --gguf MODEL.gguf [--serve | --ids 1,2,3 [--n N]]\n"
                 "  --max-context N      tokens the KV cache holds (default 32768)\n"
                 "  --ubatch N           prompt tokens per slice (default 2048)\n"
                 "  --expert-gb X        unified memory for the routed-expert pool (default: what the GPU's\n"
                 "                       working set leaves after the dense weights and the cache)\n"
                 "  --io-threads N       parallel SSD reads for expert misses (default 16)\n"
                 "  --expert-profile F   ranked expert profile that fills the pool at load (data/expert-profile-glm.bin)\n"
                 "  --expert-profile-save F  this Mac's learned profile: read instead when it exists, rewritten after\n"
                 "                       each request\n"
                 "  --kv-f16             MLA latent cache in F16 (half the bytes; default F32, F16 on the GPU)\n"
                 "  --kv-f32             keep the MLA latent cache F32 on the GPU too (MAYA_METAL_KV_F32=1)\n"
                 "  --no-mtp             no MTP drafting (the NextN block is not loaded)\n"
                 "  --dflash F           draft with this DFlash2 drafter GGUF instead of the NextN block\n"
                 "  --cpu                run on ggml-cpu instead of the GPU (reference runs)\n");
}

std::vector<int32_t> parse_ids(const char* s) {
    std::vector<int32_t> v;
    for (const char* p = s; *p;) {
        char* end = nullptr;
        v.push_back((int32_t) std::strtol(p, &end, 10));
        p = *end == ',' ? end + 1 : end;
        if (end == p && *p) break;
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf;
    int64_t max_ctx = 32768;
    bool serve = false;
    std::vector<int32_t> ids;
    int n_gen = 32;
    maya::metal::GlmOptions o;
    o.mtp = std::getenv("MAYA_GLM_NO_MTP") == nullptr && std::getenv("MAYA_GLM_NO_SPEC") == nullptr;
    if (const char* v = std::getenv("MAYA_METAL_DSA_MM"); v && *v) o.dsa_mm = std::atoi(v) != 0;
    if (const char* v = std::getenv("MAYA_METAL_DSA_KERNEL"); v && *v) o.dsa_kernel = std::atoi(v) != 0;
    if (const char* v = std::getenv("MAYA_METAL_KV_F32"); v && *v) o.kv_f32 = std::atoi(v) != 0;
    std::vector<std::string> rest;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "maya-metal: %s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--gguf" || a == "--native") gguf = next();
        else if (a == "--max-context") max_ctx = std::atoll(next());
        else if (a == "--serve") serve = true;
        else if (a == "--ids") ids = parse_ids(next());
        else if (a == "--n") n_gen = std::atoi(next());
        else if (a == "--ubatch") o.n_ubatch = std::atoi(next());
        else if (a == "--attn-chunk") o.attn_chunk = std::atoi(next());
        else if (a == "--expert-gb") o.expert_budget = (size_t) (std::atof(next()) * 1e9);
        else if (a == "--pool-slots") o.pool_slots = std::atoi(next());
        else if (a == "--io-threads") o.io_threads = std::atoi(next());
        else if (a == "--expert-profile") o.expert_profile = next();
        else if (a == "--expert-profile-save") o.expert_profile_save = next();
        else if (a == "--threads") o.cpu_threads = std::atoi(next());
        else if (a == "--kv-f16") o.kv_f16 = true;
        else if (a == "--kv-f32") o.kv_f32 = true;
        else if (a == "--cpu") o.use_gpu = false;
        else if (a == "--no-mtp") o.mtp = false;
        else if (a == "--dflash") o.dflash = next();
        else if (a == "--dump") o.dump_dir = next();
        else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else {
            rest.push_back(a);   // the serve loop takes the server's other flags
        }
    }
    if (gguf.empty()) {
        usage();
        return 2;
    }
    if (!o.dflash.empty()) o.mtp = false;   // as the CUDA engine: a DFlash2 drafter replaces the NextN block
    if (o.mtp) o.max_window = std::max(o.max_window, maya::metal::mtp_draft_cap() + 1);
    if (!o.dflash.empty()) o.max_window = std::max(o.max_window, 16);
    maya::metal::GlmMetal model;
    model.set_options(o);
    std::string err;
    const auto t0 = std::chrono::steady_clock::now();
    if (!model.load(gguf, max_ctx, err)) {
        std::fprintf(stderr, "maya-metal: %s\n", err.c_str());
        return 1;
    }
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "maya-metal: loaded in %.1f s: %s\n", load_s, model.describe().c_str());
    if (serve) return maya::metal::serve(model, rest);

    if (ids.empty()) ids = {1};
    std::vector<float> logits;
    auto t1 = std::chrono::steady_clock::now();
    if (!model.forward(ids, logits, err)) {
        std::fprintf(stderr, "maya-metal: %s\n", err.c_str());
        return 1;
    }
    const double pp = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    std::printf("prompt %zu tokens in %.2f s (%.1f tok/s)\n", ids.size(), pp, ids.size() / pp);
    std::vector<int32_t> out;
    t1 = std::chrono::steady_clock::now();
    const auto s0 = model.stats();
    const size_t V = (size_t) model.geometry().n_vocab;
    auto argmax = [&](const float* l) {
        int32_t best = 0;
        for (size_t v = 1; v < V; ++v)
            if (l[v] > l[(size_t) best]) best = (int32_t) v;
        return best;
    };
    maya::metal::DecodeHooks hk;   // greedy, n_gen tokens
    hk.pick = argmax;
    hk.emit = [&](int32_t t) {
        out.push_back(t);
        return (int) out.size() < n_gen;
    };
    hk.left = [&] { return (int64_t) (n_gen - (int) out.size()); };
    int drafted = 0, accepted = 0, rounds = 0;
    auto tally = [&](int n, int acc, double) {
        drafted += n;
        accepted += acc;
        rounds += n > 0;
    };
    bool ok = true;
    const int32_t y = argmax(logits.data());
    if (hk.emit(y)) {
        if (model.has_dflash()) {
            const maya::metal::Drafter df = [&](int32_t tok, int64_t, int n, std::vector<int32_t>& d, std::string& e) {
                return model.dflash_draft(tok, n, 0.95f, 0, 0.0f, d, e);
            };
            ok = maya::metal::decode_spec(model, y, model.dflash_block() - 1, df, tally, hk, err);
        } else if (model.has_mtp()) {
            maya::metal::DraftLength len(maya::metal::mtp_draft_cap());
            int chosen = 0;
            const maya::metal::Drafter mtp = [&](int32_t tok, int64_t p0, int n, std::vector<int32_t>& d, std::string& e) {
                chosen = model.mtp_hidden_pos() == p0 - 1 ? std::min(len.choose(), n) : 0;
                for (int i = 0; i < chosen; ++i) {
                    int32_t x = 0;
                    if (!model.mtp_step(i == 0 ? tok : d.back(), p0 - 1 + i, i == 0, x, e)) return false;
                    d.push_back(x);
                }
                return true;
            };
            ok = maya::metal::decode_spec(model, y, 15, mtp, [&](int n, int acc, double ms) {
                tally(n, acc, ms);
                len.record(chosen, std::min(acc, n), ms);
            }, hk, err);
        } else {
            for (int32_t t = y;;) {
                if (!(ok = model.forward({t}, logits, err))) break;
                t = argmax(logits.data());
                if (!hk.emit(t)) break;
            }
        }
    }
    if (!ok) {
        std::fprintf(stderr, "maya-metal: %s\n", err.c_str());
        return 1;
    }
    const double tg = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    const auto& s = model.stats();
    std::printf("tokens:");
    for (int32_t t : out) std::printf(" %d", t);
    const uint64_t hits = s.expert_hits - s0.expert_hits, misses = s.expert_misses - s0.expert_misses;
    if (rounds > 0)
        std::printf("\ndrafts: %d rounds, %d drafted, %d accepted (%.2f tokens a pass)", rounds, drafted, accepted,
                    (double) (accepted + rounds) / rounds);
    std::printf("\ndecode %d tokens in %.2f s (%.2f tok/s); experts %.1f%% hit, %.2f GB read, io %.2f s, gpu %.2f s, "
                "host %.2f s (%llu graphs)\n",
                n_gen, tg, (n_gen - 1) / tg, hits + misses ? 100.0 * hits / (hits + misses) : 100.0,
                (s.bytes_read - s0.bytes_read) / 1e9, s.io_seconds - s0.io_seconds, s.gpu_seconds - s0.gpu_seconds,
                s.host_seconds - s0.host_seconds, (unsigned long long) (s.graphs - s0.graphs));
    return 0;
}
