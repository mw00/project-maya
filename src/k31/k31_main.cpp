// src/k31/k31_main.cpp - maya-k31, the Kimi K3.1 engine (docs/K31-ENGINE.md).  Phase 3 skeleton: what it can already
// say about a file, and nothing it cannot yet do.
//
//   maya-k31 --version                                       release and commit (cmake/build_id.cmake)
//   maya-k31 --pack-info FILE.gguf [--eps moonshot|llama.cpp]
//       the file's K3.1 geometry, read from its header and tensor directory alone (k31::read: no tensor data, no GPU):
//       the layer split (KDA / MLA), the experts (a pruned file's kept count and source ids, from its
//       kimi-k31.expert_prune keys or else its <file>.prune.json sidecar), the widths, the epsilons, the tensor types
//       and which kernels take them, then k31::dump's lines (ref/kimi_k31.py's KimiK31Config names, the form
//       tools/k31/geo_ref.py writes for the oracle).  The defaults taken for absent keys go to stderr.
//       A file the engine cannot run is refused with k31::read's message ("maya-k31: geometry not covered: ...").
//
// --version and --help stand alone (with any other argument, before or after, the command line is refused).
// Exit codes: 0 done, 1 the file was refused or could not be read, 2 the command line was.  The engine's modes
// (--serve, --score, --dump-raw-dir, --bench, --experts) are refused until they are built (plan §3.1, M1 onwards).
#include "k31_geo.hpp"
#include "k31_model.hpp"
#include "k31_remote.hpp"
#include "k31_shards.hpp"

#include "../ds4/ds4_iq.cuh"   // (the types the device GEMVs take, asked of them: iq::mv_supported)

// the commit and the release this binary was built from: cmake/build_id.cmake writes them at every build
#if __has_include("maya_build_id.h")
#include "maya_build_id.h"
#endif
#ifndef MAYA_BUILD_ID
#define MAYA_BUILD_ID "unknown"
#endif
#ifndef MAYA_RELEASE
#define MAYA_RELEASE "?"
#endif

#include <algorithm>
#include <chrono>
#include <random>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <numeric>
#include <sstream>
#include <exception>
#include <filesystem>
#include <map>
#include <regex>
#include <string>
#include <system_error>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace iq = strata::ds4::gguf::iq;

const char* const kUsage =
    "usage: maya-k31 --pack-info FILE.gguf [--eps moonshot|llama.cpp]\n"
    "           the file's Kimi K3.1 geometry, from its header and tensor directory (no tensor data, no GPU)\n"
    "       maya-k31 --dump-raw-dir DIR --golden-seqs GOLDEN.json FILE.gguf [--eps ...]\n"
    "           M1a: run the golden's sequences (each as one prefill, and as decode: decode_from tokens, then one\n"
    "           token a call) and write every seam under ref/kimi_k31.py's names, one directory a run\n"
    "           (<seq>.prefill, <seq>.decode), for tools/kimi_k31_gate.py check-dumps.  --layers N: a truncated run (the\n"
    "           first N layers, then the output mix, norm and head), the M1b slices of a real file.  --chunk N: a\n"
    "           prefill as calls of N tokens (the same bits; the device holds one chunk's scratch).\n"
    "           Placement: --gpus 0,1 (a layer split; a repeated id splits on one card), --split L (each later card's\n"
    "           first layer), --expert-slots N (VRAM experts a layer, by usage rank; default as many as fit),\n"
    "           --vram-cap-mb A,B,..: at most this much VRAM on each card in --gpus order (0: all free; for a shared card)\n"
    "           --vram-reserve-mb N (default: sized from the largest call), --usage FILE (tools/k31_usage.py), --placement FILE /\n"
    "           --write-placement FILE (read / write the placement as text).  CPU lane: --cpu-lane THREADS (RAM\n"
    "           experts' rows on the host, the device's bits), --cpu-frac F (its share of a call's RAM experts, 0.5),\n"
    "           --cpu-frac-cards A,B,..: that share per card in --gpus order (a card on a narrow link: 1),\n"
    "           --cpu-max-tokens N (calls of more tokens leave it idle, 8).  --load read|mmap|page (read: the tensors in\n"
    "           file order by large sequential reads, the default; mmap: through the file's mapping; page: the experts\n"
    "           without a VRAM slot stay in the mapping, the page cache their RAM tier, for a model larger than RAM).\n"
    "           --page-stats: count the paged experts' pages not resident, and major faults, per call.\n"
    "           --prefetch THREADS: read a layer's picked, not-resident paged experts with THREADS readers (large\n"
    "           reads, many in flight) and run its resident experts first.  --pregate (with --prefetch): predict layer\n"
    "           L+1's picks from L's FFN input and prefetch them while L computes; --page-stats counts the hit rate.\n"
    "           --car-rank R: cache-aware routing on decode tokens: a pick not resident is read only if its router rank\n"
    "           < R, else the token routes among the resident experts plus the picks read (R = 16: today's routing).\n"
    "           --ram-slots N|auto (with --load page): the RAM expert tier, paged experts read with O_DIRECT into\n"
    "           pinned slots (auto: MemAvailable less 16 GB), an LRU by layer and expert; --ram-readers N (16).\n"
    "           --tier-prompt-cold: a prompt call's RAM tier reads go in at the LRU's cold end (decode's set stays).\n"
    "           --dense-override FILE: dense tensors of the same name and shape from FILE replace the model's (another\n"
    "           type: tools/k31_dense_requant.py); routed experts are refused.\n"
    "           --vram-dyn N|auto (with the RAM tier): a dynamic VRAM expert pool on each card (auto: its free VRAM less\n"
    "           the call reserve); a decode call's RAM hits are uploaded to it on a copy stream, its LRU victims go back\n"
    "           to RAM; --vram-dyn-up N: at most N uploads a layer call (64).  --expert-slots 0 frees the static slots.\n"
    "           --pregate-rank R: pregate reads only the predicted ranks < R; --pregate-decode-only: none on prompts.\n"
    "           --remote-experts HOST:PORT with --remote-own FILE (lines \"layer e e ...\") and/or --remote-frac P (the\n"
    "           most used share P of each layer's non-VRAM experts by --usage, else a fixed hash): those picks are\n"
    "           computed by an expert server.\n"
    "       maya-k31 --expert-server PORT|HOST:PORT [FILE.gguf] [placement, --load page, --cpu-lane N]: serve remote\n"
    "           experts (no FILE: echo only; a bare PORT listens on 127.0.0.1, no authentication).\n"
    "       maya-k31 --remote-ping HOST:PORT N ROWS PICKS ZIN LAT: N echo round trips (q8 rows of ZIN, PICKS x LAT replies).\n"
    "           FILE may be the first shard of a split GGUF (<name>-00001-of-000NN.gguf): the set is read as one file.\n"
    "       maya-k31 --gen FILE.gguf [--chunk N] [placement, CPU lane, --load page, --page-stats]\n"
    "           token ids in, sampled ids out on stdin/stdout, the KV cache kept across turns (k31chat.py drives it)\n"
    "       maya-k31 --version\n"
    "       maya-k31 --bench P,D FILE.gguf [--layers N] [--chunk N] [placement, CPU lane]\n"
    "           a P-token prompt (fixed seed), then D greedy decode tokens one a call, timed: prefill tok/s, decode ms\n"
    "           (--profile: wall time by part, each boundary waiting for the device)\n"
    "Not built yet: --serve, --score, --experts.\n";

// ggml's block sizes (values, bytes) a type, for the tensor directory's byte counts (ggml-common.h); 0 = unknown
struct Block {
    uint64_t values, bytes;
};
Block block_of(uint32_t t) {
    switch (t) {
    case 0: return {1, 4};      // F32
    case 1: return {1, 2};      // F16
    case 2: return {32, 18};    // Q4_0
    case 3: return {32, 20};    // Q4_1
    case 6: return {32, 22};    // Q5_0
    case 7: return {32, 24};    // Q5_1
    case 8: return {32, 34};    // Q8_0
    case 9: return {32, 36};    // Q8_1
    case 10: return {256, 84};  // Q2_K
    case 11: return {256, 110}; // Q3_K
    case 12: return {256, 144}; // Q4_K
    case 13: return {256, 176}; // Q5_K
    case 14: return {256, 210}; // Q6_K
    case 15: return {256, 292}; // Q8_K
    case 16: return {256, 66};  // IQ2_XXS
    case 17: return {256, 74};  // IQ2_XS
    case 18: return {256, 98};  // IQ3_XXS
    case 19: return {256, 50};  // IQ1_S
    case 20: return {32, 18};   // IQ4_NL
    case 21: return {256, 110}; // IQ3_S
    case 22: return {256, 82};  // IQ2_S
    case 23: return {256, 136}; // IQ4_XS
    case 24: return {1, 1};     // I8
    case 25: return {1, 2};     // I16
    case 26: return {1, 4};     // I32
    case 27: return {1, 8};     // I64
    case 28: return {1, 8};     // F64
    case 29: return {256, 56};  // IQ1_M
    case 30: return {1, 2};     // BF16
    case 34: return {256, 54};  // TQ1_0
    case 35: return {256, 66};  // TQ2_0
    case 39: return {32, 17};   // MXFP4
    default: return {0, 0};
    }
}

// bytes of a tensor (0: a type this table does not know, or a row that is not whole blocks)
uint64_t tensor_bytes(const strata::TensorInfo& t) {
    const Block b = block_of(t.type);
    if (b.values == 0 || t.shape.empty() || t.shape[0] % b.values) return 0;
    return t.elements() / b.values * b.bytes;
}

// which of the engine's kernels take a matrix of this type: the device GEMVs (ds4_iq's iq::mv, asked of it), cuBLAS
// for the float types, or none built in yet (MXFP4 is host-exact only: plan §3.3, k31_fmt)
const char* kernel_for(uint32_t t) {
    if (t == 0 || t == 1 || t == 30) return "cuBLAS";
    if (iq::mv_supported((int) t)) return "iq::mv";
    return "not built in";
}

// a float as %g (the summary; k31::dump's lines keep every digit)
std::string g6(double x) {
    char b[32];
    std::snprintf(b, sizeof b, "%g", x);
    return b;
}

std::string gb(uint64_t bytes) {
    char b[64];
    std::snprintf(b, sizeof b, "%.2f GB", (double) bytes / 1e9);
    return b;
}

std::string join(const std::vector<int>& xs, const char* sep = " ") {
    std::string s;
    for (size_t i = 0; i < xs.size(); ++i) s += (i ? sep : "") + std::to_string(xs[i]);
    return s;
}

// "1-3 7 9-12": layer ranges
std::string ranges(const std::vector<int>& xs) {
    std::string s;
    for (size_t i = 0; i < xs.size();) {
        size_t j = i;
        while (j + 1 < xs.size() && xs[j + 1] == xs[j] + 1) ++j;
        if (!s.empty()) s += " ";
        s += std::to_string(xs[i]);
        if (j > i) s += "-" + std::to_string(xs[j]);
        i = j + 1;
    }
    return s;
}

int pack_info(const std::string& path, k31::EpsPreset eps) {
    std::error_code ec;
    const fs::file_status st = fs::status(path, ec);
    if (!fs::exists(st)) {
        std::fprintf(stderr, "maya-k31: cannot read %s: no such file\n", path.c_str());
        return 1;
    }
    if (!fs::is_regular_file(st)) {
        std::fprintf(stderr, "maya-k31: cannot read %s: not a regular file\n", path.c_str());
        return 1;
    }
    try {
        const k31::Shards g(path, /*allow_missing*/ true);
        if (g.n_expected() > 1)
            std::printf("shards      %zu of %d on disk, %zu tensors in them%s\n", g.n_open(), g.n_expected(),
                        g.tensors().size(),
                        g.tensors_count() >= 0 ? (" (split.tensors.count " + std::to_string(g.tensors_count()) + ")").c_str() : "");
        for (const auto& p : g.missing()) std::printf("missing     %s\n", p.c_str());
        k31::Geo geo;
        std::string err;
        if (!k31::read(g, path, geo, err, eps)) {
            if (!g.complete() && err.find(" is missing") != std::string::npos) {
                // a set still downloading: the header's keys passed; the tensor check stops at the first absent one
                std::printf("geometry    the header's keys read; the tensor check waits for the missing shards (%s)\n",
                            err.c_str());
                std::map<uint32_t, std::pair<uint64_t, uint64_t>> part;
                for (const auto& t : g.tensors()) { part[t.type].first += 1; part[t.type].second += tensor_bytes(t); }
                for (const auto& [t, e] : part)
                    std::printf("type        %-8s %5llu tensors %12s  (%s) [shards on disk]\n", strata::ggml_type_name(t),
                                (unsigned long long) e.first, gb(e.second).c_str(), kernel_for(t));
                for (const auto& [k, v] : g.metadata())
                    if (k.rfind(std::string(k31::kArch) + ".", 0) == 0 || k.rfind("general.", 0) == 0 || k.rfind("split.", 0) == 0)
                        std::printf("key         %s = %s\n", k.c_str(),
                                    v.type == strata::MetaType::STRING ? v.s.substr(0, 80).c_str()
                                    : v.type == strata::MetaType::ARRAY ? ("[" + std::to_string(v.count) + " items]").c_str()
                                                                         : g6(v.num()).c_str());
                return 0;
            }
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        for (const auto& n : geo.notes) std::fprintf(stderr, "maya-k31: note: %s\n", n.c_str());

        // the tensor directory: bytes a type, the data the directory describes against what the file holds
        std::map<uint32_t, std::pair<uint64_t, uint64_t>> by_type;   // type -> (tensors, bytes)
        uint64_t data_end = 0, present = 0, file_bytes = 0;   // summed over the shards
        bool sized = true;
        std::vector<uint64_t> end_of(g.n_open(), 0);
        for (const auto& t : g.tensors()) {
            const uint64_t nb = tensor_bytes(t);
            if (nb == 0) sized = false;
            auto& e = by_type[t.type];
            e.first += 1;
            e.second += nb;
            uint64_t& de = end_of[(size_t) g.shard_of(t)];
            de = std::max(de, t.offset + nb);
        }
        bool short_shard = false;
        for (size_t s = 0; s < g.n_open(); ++s) {
            const strata::GgufFile& f = g.file(s);
            const uint64_t have = f.file_size() > f.data_start() ? f.file_size() - f.data_start() : 0;
            data_end += end_of[s];
            present += std::min(have, end_of[s]);
            file_bytes += f.file_size();
            if (have < end_of[s]) {
                short_shard = true;
                std::printf("short       %s: %llu of %llu data bytes\n", g.path(s).c_str(), (unsigned long long) have,
                            (unsigned long long) end_of[s]);
            }
        }
        if (!short_shard) present = data_end;

        // the routed experts' types, per MoE layer: gate / up / down
        const std::regex exps(R"(^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$)");
        std::map<int, std::array<int, 3>> roles;   // layer -> type of gate, up, down (-1: absent)
        for (const auto& t : g.tensors()) {
            std::smatch m;
            if (!std::regex_match(t.name, m, exps)) continue;
            auto it = roles.try_emplace(std::stoi(m[1].str()), std::array<int, 3>{-1, -1, -1}).first;
            const std::string r = m[2].str();
            it->second[r == "gate" ? 0 : r == "up" ? 1 : 2] = (int) t.type;
        }

        const std::vector<int> mla = geo.mla_layers();
        std::vector<int> moe;
        for (int il = geo.n_layer_dense_lead; il < geo.n_layers; ++il) moe.push_back(il);
        std::vector<int> pushes;
        for (int j = 0; j < geo.n_layers; j += std::max(1, geo.attn_res_block_size)) pushes.push_back(j);
        // the expert width per MoE layer: one value, or the range when it differs (thin experts)
        int ff_lo = 0, ff_hi = 0;
        for (int il : moe) {
            const int f = geo.n_ff_exp[(size_t) il];
            ff_lo = ff_lo ? std::min(ff_lo, f) : f;
            ff_hi = std::max(ff_hi, f);
        }

        std::printf("maya-k31 pack-info: %s\n", path.c_str());
        std::printf("file        %llu bytes, GGUF v%u, %zu tensors, %zu metadata keys\n",
                    (unsigned long long) file_bytes, g.version(), g.tensors().size(), g.metadata().size());
        if (sized)
            std::printf("data        %llu bytes in the tensor directory (%s): %s\n", (unsigned long long) data_end,
                        gb(data_end).c_str(),
                        present >= data_end ? "all present"
                                            : (present == 0 ? "header only (no tensor data in this file)"
                                                            : ("truncated: " + std::to_string(present) + " bytes present")
                                                                  .c_str()));
        else
            std::printf("data        (a tensor of a type this build cannot size)\n");
        std::printf("model       %s, %d layers, n_embd %d, vocab %d, %d heads, context %d\n", k31::kArch, geo.n_layers,
                    geo.n_embd, geo.n_vocab, geo.n_head, geo.n_ctx_train);
        std::printf("attention   KDA %d layers, MLA %zu layers (%s)\n", geo.n_layers - (int) mla.size(), mla.size(),
                    join(mla).c_str());
        std::printf("  MLA       q_lora %d, kv_lora %d + rope %d (key row %d, value row %d), head k %d (nope %d) / v %d\n",
                    geo.q_lora_rank, geo.kv_lora_rank, geo.n_rot, geo.kv_lora_rank + geo.n_rot, geo.kv_lora_rank,
                    geo.n_head_k_mla, geo.qk_nope, geo.n_head_v_mla);
        std::printf("  KDA       %d heads x %d (d_inner %d), conv %d, gate lower bound %s\n", geo.n_kda_head,
                    geo.kda_head_dim, geo.d_inner(), geo.ssm_d_conv, g6(geo.kda_gate_lower_bound).c_str());
        std::printf("  indexer   %d heads x %d, top %d\n", geo.idx_n_head, geo.idx_key_dim, geo.idx_top_k);
        std::printf("ffn         dense layers %d (ff %d), MoE layers %zu (%s)\n", geo.n_layer_dense_lead, geo.n_ff,
                    moe.size(), ranges(moe).c_str());
        std::printf("  experts   %d a MoE layer (of %d in the source), %d a token, latent %d, expert ff %s, shared %d\n",
                    geo.n_expert, geo.expert_source_count, geo.n_expert_used, geo.n_expert_latent,
                    ff_lo == ff_hi ? std::to_string(ff_lo).c_str() : (std::to_string(ff_lo) + ".." + std::to_string(ff_hi)).c_str(),
                    geo.n_expert_shared);
        const std::string keep_src = geo.keep_from == "file"      ? "the file's kimi-k31.expert_prune.keep"
                                     : geo.keep_from == "sidecar" ? path + ".prune.json"
                                                                  : "nowhere (no expert_prune keys, no sidecar): "
                                                                    "the file's own ids 0.." +
                                                                        std::to_string(geo.n_expert - 1);
        std::printf("  kept ids  from %s%s\n", keep_src.c_str(),
                    geo.prune_method.empty() ? "" : (" (method " + geo.prune_method + ")").c_str());
        std::printf("  routing   weights scale %s, norm %s, SiTU beta %s / linear %s\n",
                    g6(geo.expert_weights_scale).c_str(), geo.expert_weights_norm ? "true" : "false",
                    g6(geo.situ_beta).c_str(), g6(geo.situ_linear_beta).c_str());
        std::printf("attn_res    block %d: %zu pushes (%s)\n", geo.attn_res_block_size, pushes.size(), join(pushes).c_str());
        std::printf("eps         %s: rms %s, indexer LN %s, MLA latent %s, KDA l2 %s, router renorm %s\n",
                    geo.eps_preset_name(), g6(geo.eps_rms).c_str(), g6(geo.eps_indexer_ln).c_str(),
                    g6(geo.eps_mla_latent).c_str(), g6(geo.eps_kda_l2).c_str(), g6(geo.eps_router_renorm).c_str());
        for (const auto& [t, e] : by_type)
            std::printf("type        %-8s %5llu tensors %12s  (%s)\n", strata::ggml_type_name(t),
                        (unsigned long long) e.first, e.second ? gb(e.second).c_str() : "?", kernel_for(t));
        // routed experts: one line per run of layers with the same gate / up / down types
        std::vector<std::pair<std::array<int, 3>, std::vector<int>>> runs;
        for (const auto& [il, r] : roles) {
            if (!runs.empty() && runs.back().first == r && runs.back().second.back() == il - 1)
                runs.back().second.push_back(il);
            else
                runs.push_back({r, {il}});
        }
        const auto tn = [](int t) { return t < 0 ? "absent" : strata::ggml_type_name((uint32_t) t); };
        for (const auto& [r, ls] : runs)
            std::printf("experts     layers %s: gate %s, up %s, down %s (%s)\n", ranges(ls).c_str(), tn(r[0]), tn(r[1]),
                        tn(r[2]),
                        r[0] >= 0 && r[1] >= 0 && r[2] >= 0
                            ? (std::string(kernel_for((uint32_t) r[0])) + " / " + kernel_for((uint32_t) r[1]) + " / " +
                               kernel_for((uint32_t) r[2]))
                                  .c_str()
                            : "a role missing");
        std::printf("--- geometry (ref/kimi_k31.py KimiK31Config names; tools/k31/geo_ref.py writes the oracle's)\n");
        std::fputs(k31::dump(geo).c_str(), stdout);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "maya-k31: cannot read %s: %s\n", path.c_str(), e.what());
        return 1;
    }
}

// --dump-raw-dir: the golden's sequences through the M1 runner, each run's seams to DIR/<seq>.<mode>
int dump_runs(const std::string& file, const std::string& seqs_path, const std::string& dir, k31::EpsPreset eps,
              const k31::ModelOptions& opt, int chunk, const std::vector<std::string>& seam_globs) {
    std::string text, err;
    if (!k31::detail::read_text(seqs_path, text)) {
        std::fprintf(stderr, "maya-k31: cannot read %s\n", seqs_path.c_str());
        return 1;
    }
    k31::json::Value root;
    if (!k31::json::Parser(text).parse(root, err)) {
        std::fprintf(stderr, "maya-k31: %s: %s\n", seqs_path.c_str(), err.c_str());
        return 1;
    }
    const k31::json::Value* seqs = root.get("sequences");
    if (!seqs || seqs->kind != k31::json::Value::array || seqs->items.empty()) {
        std::fprintf(stderr, "maya-k31: %s has no \"sequences\" list\n", seqs_path.c_str());
        return 1;
    }
    // the auto VRAM reserve sizes a call's scratch: the longest prefill call (a chunk, or a whole sequence)
    k31::ModelOptions o = opt;
    for (const auto& sq : seqs->items) {
        const k31::json::Value* tv = sq.get("tokens");
        const int n = tv && tv->kind == k31::json::Value::array ? (int) tv->items.size() : 0;
        o.max_call_tokens = std::max(o.max_call_tokens, chunk > 0 ? std::min(chunk, n) : n);
    }
    k31::Model m;
    if (!m.load(file, o, err, eps)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "%s", m.placement_summary().c_str());
    for (const auto& sq : seqs->items) {
        const k31::json::Value *nv = sq.get("name"), *tv = sq.get("tokens"), *dv = sq.get("decode_from");
        if (!nv || nv->kind != k31::json::Value::string || !tv || tv->kind != k31::json::Value::array) {
            std::fprintf(stderr, "maya-k31: %s: a sequence without a name or tokens\n", seqs_path.c_str());
            return 1;
        }
        std::vector<int> toks;
        for (const auto& x : tv->items) toks.push_back((int) x.i);
        const int from = dv && dv->kind == k31::json::Value::number ? (int) dv->i : (int) toks.size();
        for (const char* mode : {"prefill", "decode"}) {
            if (std::string(mode) == "decode" && from >= (int) toks.size()) continue;   // a prefill-only sequence
            k31::Dumps d;
            d.set_filter(seam_globs);
            m.reset();
            bool ok;
            // a long prefill as chunks of `chunk` tokens (0: one call): the runner's bits do not depend on the split
            auto prefill = [&](size_t a, size_t b) {
                const size_t step = chunk > 0 ? (size_t) chunk : b - a;
                bool r = true;
                for (size_t c = a; r && c < b; c += step)
                    r = m.forward(std::vector<int>(toks.begin() + c, toks.begin() + std::min(b, c + step)), nullptr, &d, err);
                return r;
            };
            if (std::string(mode) == "prefill") {
                ok = prefill(0, toks.size());
            } else {
                ok = prefill(0, (size_t) from);
                for (size_t i = (size_t) from; ok && i < toks.size(); ++i) ok = m.forward({toks[i]}, nullptr, &d, err);
            }
            const std::string out = dir + "/" + nv->s + "." + mode;
            if (!ok || !d.write(out, err)) {
                std::fprintf(stderr, "%s\n", err.c_str());
                return 1;
            }
            std::printf("maya-k31: %s: %zu tokens, %zu seams\n", out.c_str(), toks.size(), d.size());
        }
    }
    std::fprintf(stderr, "maya-k31: %s\n", m.expert_stats().c_str());
    return 0;
}

// --bench P,D: a prompt of P tokens (fixed seed) as one prefill (--chunk splits it), then D greedy decode tokens, one
// a call, each timed (no dumps).  Prints the build, the placement, prefill tok/s and decode ms a token.
int bench(const std::string& file, k31::EpsPreset eps, const k31::ModelOptions& opt, int chunk, int P, int D) {
    using clk = std::chrono::steady_clock;
    auto secs = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); };
    std::string err;
    k31::ModelOptions o = opt;   // the auto VRAM reserve: the prefill's calls of `chunk` tokens (0: one call of P)
    o.max_call_tokens = chunk > 0 ? std::min(chunk, P) : P;
    k31::Model m;
    const auto t0 = clk::now();
    if (!m.load(file, o, err, eps)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    const auto t1 = clk::now();
    std::printf("maya-k31 %s (build %s) --bench %d,%d: %s, %d layers run\n%s", MAYA_RELEASE, MAYA_BUILD_ID, P, D, file.c_str(),
                opt.n_layers > 0 ? std::min(opt.n_layers, m.geo().n_layers) : m.geo().n_layers, m.placement_summary().c_str());
    std::printf("load %.1f s\n", secs(t0, t1));
    std::mt19937 rng(1);
    std::vector<int> toks((size_t) P);
    for (int& t : toks) t = (int) (rng() % (uint32_t) std::min(m.geo().n_vocab, 150000));
    std::vector<float> lg;
    const auto t2 = clk::now();
    const size_t step = chunk > 0 ? (size_t) chunk : toks.size();
    for (size_t c = 0; c < toks.size(); c += step) {
        const std::vector<int> part(toks.begin() + c, toks.begin() + std::min(toks.size(), c + step));
        if (!m.forward(part, c + step >= toks.size() ? &lg : nullptr, nullptr, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
    }
    const auto t3 = clk::now();
    std::printf("prefill %d tokens: %.2f s, %.1f tok/s\n", P, secs(t2, t3), P / std::max(secs(t2, t3), 1e-9));
    auto pages = [&](const char* what) {
        if (!opt.page_stats) return;
        const k31::Model::PageCount c = m.page_total();
        std::printf("%s pages: %lld calls, %lld tokens; paged expert matrices used %lld, missed %lld (%.1f%%), %.2f GB not "
                    "resident; %lld major faults; a token: %.1f misses, %.1f MB, %.0f faults\n",
                    what, c.calls, c.tokens, c.used, c.missed, c.used ? 100.0 * c.missed / c.used : 0.0, c.miss_bytes / 1e9,
                    c.majflt, (double) c.missed / std::max(1LL, c.tokens), c.miss_bytes / 1e6 / std::max(1LL, c.tokens),
                    (double) c.majflt / std::max(1LL, c.tokens));
        if (opt.prefetch > 0)
            std::printf("%s prefetch: %lld ranges, %.2f GB queued; pregate: %lld ranges, %.2f GB queued, hit %lld of %lld "
                        "picks (%.1f%%)\n", what, c.pf_ranges, c.pf_bytes / 1e9, c.pg_ranges, c.pg_bytes / 1e9, c.pg_hits,
                        c.pg_picks, c.pg_picks ? 100.0 * c.pg_hits / c.pg_picks : 0.0);
        std::printf("%s cpu: wall %.1f s, user %.1f s, kernel %.1f s (%.0f%% of the CPU time)\n", what, c.wall_us / 1e6,
                    c.utime_us / 1e6, c.stime_us / 1e6, 100.0 * c.stime_us / std::max(1LL, c.utime_us + c.stime_us));
        if (c.tier_hits + c.tier_misses + c.dio_reads > 0)
            std::printf("%s ram tier: hits %lld, misses %lld, evictions %lld, left in the mapping %lld; read %.2f GB in %lld "
                        "reads, %lld failed (%.0f MB/s over the calls); blocked on reads %.1f s (%.1f%% of the wall), %.1f ms a token\n",
                        what, c.tier_hits, c.tier_misses, c.tier_evicts, c.tier_fallback, c.dio_bytes / 1e9, c.dio_reads, c.dio_fails,
                        c.dio_bytes / 1e6 / std::max(1e-9, c.wall_us / 1e6), c.tier_wait_us / 1e6,
                        100.0 * c.tier_wait_us / std::max(1LL, c.wall_us), c.tier_wait_us / 1e3 / std::max(1LL, c.tokens));
        if (c.remote_calls > 0)
            std::printf("%s remote experts: %lld layer calls, %lld picks (%.1f a token); sent %.2f MB, received %.2f MB; blocked on "
                        "replies %.1f s (%.1f%% of the wall), %.2f ms a token, %.1f us a layer call\n", what, c.remote_calls, c.remote_picks,
                        (double) c.remote_picks / std::max(1LL, c.tokens), c.remote_tx / 1e6, c.remote_rx / 1e6, c.remote_wait_us / 1e6,
                        100.0 * c.remote_wait_us / std::max(1LL, c.wall_us), c.remote_wait_us / 1e3 / std::max(1LL, c.tokens),
                        (double) c.remote_wait_us / std::max(1LL, c.remote_calls));
        if (c.dyn_hits + c.dyn_up > 0)
            std::printf("%s vram dyn: hits %lld, uploads %lld (%.2f GB, %.2f GB a token), victims back to RAM %lld, to the mapping %lld\n",
                        what, c.dyn_hits, c.dyn_up, c.dyn_up_bytes / 1e9, c.dyn_up_bytes / 1e9 / std::max(1LL, c.tokens), c.dyn_down, c.dyn_drop);
        if (opt.car_rank > 0)
            std::printf("%s car (rank < %d read): %lld token-layers, picks not resident %lld, read %lld, skipped %lld (%.2f a "
                        "token, %.1f%% of picks); %lld residency checks past the top k\n", what, opt.car_rank, c.car_rows,
                        c.car_missed, c.car_fetched, c.car_skipped, (double) c.car_skipped / std::max(1LL, c.tokens),
                        c.car_rows ? 100.0 * c.car_skipped / (16.0 * c.car_rows) : 0.0, c.car_checked);
        m.page_reset();
    };
    pages("prefill");
    const int V = m.geo().n_vocab;
    if (opt.profile) std::printf("prefill %s\n", m.profile().c_str());
    m.profile_reset();
    std::vector<double> ms;
    for (int i = 0; i < D; ++i) {
        const float* last = lg.data() + lg.size() - (size_t) V;
        const int next = (int) (std::max_element(last, last + V) - last);
        const auto a = clk::now();
        if (!m.forward({next}, &lg, nullptr, err, /*decode*/ true)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        ms.push_back(1e3 * secs(a, clk::now()));
    }
    if (!ms.empty()) {
        std::vector<double> s2 = ms;
        std::sort(s2.begin(), s2.end());
        double sum = 0;
        for (double v : ms) sum += v;
        std::printf("decode %d tokens at %d..%d: mean %.1f ms, median %.1f ms, p90 %.1f ms a token: %.2f tok/s\n", D, P, P + D,
                    sum / D, s2[s2.size() / 2], s2[(size_t) (0.9 * (s2.size() - 1))], 1e3 * D / sum);
    }
    pages("decode");
    std::printf("%s\n", m.expert_stats().c_str());
    if (opt.profile) std::printf("decode %s\n", m.profile().c_str());
    return 0;
}


// --gen: token ids in, token ids out, for a chat driver that holds the tokenizer (k31tok).  Commands on stdin, one a
// line; replies on stdout, flushed:
//   R                                  a fresh sequence                      -> OK R
//   P id id ...                        the ids as the sequence's continuation -> OK P <n_past> <seconds>
//   G max temp top_p seed stop,stop..  sample up to max ids from the last logits, each fed back unless it is a stop
//                                      -> T <id> per id, then E <stop|max> <n> <seconds>
// With --page-stats, every P and G reply is followed by "S calls tokens used missed miss_bytes majflt pf_ranges
// pf_bytes pg_ranges pg_bytes pg_picks pg_hits car_rows car_missed car_fetched car_skipped car_checked wall_us utime_us stime_us tier_hits tier_misses
// tier_evicts tier_fallback tier_wait_us dio_reads dio_bytes dio_fails remote_calls remote_picks remote_tx remote_rx
// remote_wait_us dyn_hits dyn_up dyn_down dyn_drop dyn_up_bytes cp_rows cp_missed cp_read cp_full cp_skipped" for it
// (43 fields; cp_*: prompt CAR, --car-prompt-rank).  Prompt CAR applies to P's last --car-prompt-rows ids.
// A failure answers ERR <why> (the sequence is then reset).
int gen_loop(const std::string& file, k31::EpsPreset eps, const k31::ModelOptions& opt, int chunk) {
    using clk = std::chrono::steady_clock;
    std::string err;
    k31::ModelOptions o = opt;
    o.max_call_tokens = chunk > 0 ? chunk : 512;
    if (chunk <= 0) chunk = 512;
    if (opt.car_prompt_rank > 0 && opt.car_prompt_rows > chunk) {
        std::fprintf(stderr, "maya-k31: --car-prompt-rows %d is more than a call's %d rows (--chunk)\n", opt.car_prompt_rows, chunk);
        return 2;
    }
    k31::Model m;
    if (!m.load(file, o, err, eps)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "maya-k31 %s (build %s) --gen: %s\n%s", MAYA_RELEASE, MAYA_BUILD_ID, file.c_str(),
                 m.placement_summary().c_str());
    std::printf("READY %d\n", m.geo().n_vocab);
    std::fflush(stdout);
    const int V = m.geo().n_vocab;
    std::vector<float> lg, last;
    int n_past = 0;
    auto prof = [&](const char* what) {   // --profile: the parts of the call(s) and the experts' split, on stderr
        if (!opt.profile) return;
        std::fprintf(stderr, "PROFILE %s %s; %s\n", what, m.profile().c_str(), m.expert_stats().c_str());
        m.profile_reset();
    };
    auto stats = [&] {
        if (!opt.page_stats) return;
        const k31::Model::PageCount c = m.page_total();
        std::printf("S %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld "
                    "%lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld\n", c.calls, c.tokens,
                    c.used, c.missed, c.miss_bytes, c.majflt, c.pf_ranges, c.pf_bytes, c.pg_ranges, c.pg_bytes, c.pg_picks,
                    c.pg_hits, c.car_rows, c.car_missed, c.car_fetched, c.car_skipped, c.car_checked, c.wall_us, c.utime_us,
                    c.stime_us, c.tier_hits, c.tier_misses, c.tier_evicts, c.tier_fallback, c.tier_wait_us, c.dio_reads, c.dio_bytes, c.dio_fails, c.remote_calls, c.remote_picks, c.remote_tx, c.remote_rx,
                    c.remote_wait_us, c.dyn_hits, c.dyn_up, c.dyn_down, c.dyn_drop, c.dyn_up_bytes, c.cp_rows, c.cp_missed,
                    c.cp_read, c.cp_full, c.cp_skipped);
        m.page_reset();
    };
    auto fail = [&](const std::string& why) {
        std::printf("ERR %s\n", why.c_str());
        std::fflush(stdout);
        m.reset();
        n_past = 0;
        last.clear();
    };
    // prompt CAR (--car-prompt-rank): a prompt's last call holds its last min(n, chunk) ids, so the CAR batch (its
    // last --car-prompt-rows ids, at most a chunk) is one call; without it the chunks run from the start as always
    auto feed = [&](const std::vector<int>& ids, bool decode) {
        const bool pc = !decode && opt.car_prompt_rank > 0;
        const size_t head = pc && ids.size() > (size_t) chunk ? ids.size() - (size_t) chunk : 0;
        std::vector<std::pair<size_t, size_t>> parts;
        for (size_t c = 0; c < head; c += (size_t) chunk) parts.push_back({c, std::min(head, c + (size_t) chunk)});
        for (size_t c = head; c < ids.size(); c += (size_t) chunk) parts.push_back({c, std::min(ids.size(), c + (size_t) chunk)});
        for (size_t k = 0; k < parts.size(); ++k) {
            const std::vector<int> part(ids.begin() + parts[k].first, ids.begin() + parts[k].second);
            const bool end = k + 1 == parts.size();
            const int tail = pc && end ? std::min((int) part.size(), opt.car_prompt_rows) : 0;
            if (!m.forward(part, end ? &lg : nullptr, nullptr, err, decode, tail)) return false;
            n_past += (int) part.size();
        }
        last.assign(lg.end() - V, lg.end());
        return true;
    };
    std::string line;
    char buf[1 << 16];
    while (std::fgets(buf, sizeof buf, stdin)) {
        line = buf;
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        // (a line longer than the buffer: the rest arrives as the next reads)
        while (std::strlen(buf) == sizeof buf - 1 && buf[sizeof buf - 2] != '\n' && std::fgets(buf, sizeof buf, stdin)) {
            line += buf;
            while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        }
        std::istringstream in(line);
        std::string cmd;
        if (!(in >> cmd)) continue;
        if (cmd == "R") {
            m.reset();
            n_past = 0;
            last.clear();
            std::printf("OK R\n");
        } else if (cmd == "P") {
            std::vector<int> ids;
            int x;
            while (in >> x) ids.push_back(x);
            if (ids.empty()) { fail("P without ids"); continue; }
            const auto a = clk::now();
            if (!feed(ids, false)) { fail(err); continue; }
            std::printf("OK P %d %.3f\n", n_past, std::chrono::duration<double>(clk::now() - a).count());
            stats();
            prof("P");
        } else if (cmd == "G") {
            int max_new = 0;
            double temp = 0, top_p = 1;
            unsigned long long seed = 0;
            std::string stops_s;
            if (!(in >> max_new >> temp >> top_p >> seed)) { fail("G max temp top_p seed [stops]"); continue; }
            in >> stops_s;
            std::vector<int> stops;
            for (size_t a = 0; a < stops_s.size();) {
                const size_t b = std::min(stops_s.find(',', a), stops_s.size());
                if (b > a) stops.push_back(std::atoi(stops_s.substr(a, b - a).c_str()));
                a = b + 1;
            }
            if (last.empty()) { fail("G before any P"); continue; }
            std::mt19937_64 rng(seed);
            std::vector<int> idx((size_t) V);
            std::vector<double> pr;
            const auto a = clk::now();
            int n = 0;
            const char* why = "max";
            bool ok = true;
            while (n < max_new) {
                int next;
                if (temp <= 0) {
                    next = (int) (std::max_element(last.begin(), last.end()) - last.begin());
                } else {   // softmax at temp, then the smallest set whose mass reaches top_p, sampled
                    const float mx = *std::max_element(last.begin(), last.end());
                    std::iota(idx.begin(), idx.end(), 0);
                    // candidates: within 40 nats of the best (the rest have no mass at any useful temp)
                    auto mid = std::partition(idx.begin(), idx.end(), [&](int i) { return (mx - last[(size_t) i]) / temp < 40.0; });
                    std::sort(idx.begin(), mid, [&](int x, int y) { return last[(size_t) x] > last[(size_t) y]; });
                    const size_t nc = (size_t) (mid - idx.begin());
                    pr.resize(nc);
                    double sum = 0;
                    for (size_t i = 0; i < nc; ++i) sum += pr[i] = std::exp((last[(size_t) idx[i]] - mx) / temp);
                    size_t keep = nc;
                    double acc = 0;
                    for (size_t i = 0; i < nc; ++i) {
                        acc += pr[i] / sum;
                        if (acc >= top_p) { keep = i + 1; break; }
                    }
                    double tot = 0;
                    for (size_t i = 0; i < keep; ++i) tot += pr[i];
                    double u = std::uniform_real_distribution<double>(0.0, tot)(rng);
                    next = idx[keep - 1];
                    for (size_t i = 0; i < keep; ++i) {
                        if ((u -= pr[i]) <= 0) { next = idx[i]; break; }
                    }
                }
                ++n;
                std::printf("T %d\n", next);
                std::fflush(stdout);
                if (std::find(stops.begin(), stops.end(), next) != stops.end()) { why = "stop"; break; }
                if (!feed({next}, true)) { ok = false; break; }
            }
            if (!ok) { fail(err); continue; }
            std::printf("E %s %d %.3f\n", why, n, std::chrono::duration<double>(clk::now() - a).count());
            stats();
            prof("G");
        } else {
            fail("unknown command " + cmd);
            continue;
        }
        std::fflush(stdout);
    }
    return 0;
}

// --remote-ping HOST:PORT N ROWS PICKS ZIN: N echo round trips (layer -1: q8 rows of ZIN out, PICKS x ZIN f32 back)
int remote_ping(const std::string& addr, int n, int rows, int picks, int zin) {
    namespace R = k31::remote;
    using clk = std::chrono::steady_clock;
    const int fd = R::connect_to(addr);
    R::Request rq;
    rq.layer = -1; rq.n_rows = rows; rq.zin = zin; rq.zfmt = 1;
    for (int i = 0; i < picks; ++i) rq.picks.push_back({i, rows ? i % rows : 0});
    rq.rows.assign((size_t) rows * rq.row_bytes(), 1);
    std::vector<double> us;
    std::vector<float> back;
    for (int i = 0; i < n + 3; ++i) {   // (3 warm-up trips)
        const auto a = clk::now();
        R::write_request(fd, rq);
        R::read_reply(fd, -1, picks, zin, back);
        if (i >= 3) us.push_back(std::chrono::duration<double, std::micro>(clk::now() - a).count());
    }
    ::close(fd);
    std::sort(us.begin(), us.end());
    double sum = 0;
    for (double v : us) sum += v;
    const double tx = R::kHeadReq + picks * 8.0 + rq.rows.size(), rx = R::kHeadRep + (double) picks * zin * 4;
    std::printf("remote-ping %s: %d trips, %d rows (%.1f KB out), %d picks (%.1f KB back): median %.1f us, p10 %.1f, p90 %.1f, "
                "mean %.1f us; %.0f MB/s each way at the median\n", addr.c_str(), n, rows, tx / 1024, picks, rx / 1024,
                us[us.size() / 2], us[us.size() / 10], us[(size_t) (0.9 * (us.size() - 1))], sum / us.size(),
                (tx + rx) / us[us.size() / 2]);
    return 0;
}

// --expert-server PORT with no model: echo only (link timing on a machine without the weights)
int echo_server(const std::string& where) {
    namespace R = k31::remote;
    const int lfd = R::listen_on(where);
    std::fprintf(stderr, "maya-k31 expert server: echo only, on %s\n", where.c_str());
    for (;;) {
        const int fd = ::accept(lfd, nullptr, nullptr);
        if (fd < 0) continue;
        R::tune(fd);
        try {
            R::Request rq;
            std::vector<float> out;
            while (R::read_request(fd, rq)) {
                const int np = (int) rq.picks.size();
                out.assign((size_t) np * (size_t) rq.zin, 0.0f);
                R::write_reply(fd, rq.layer, rq.zin, rq.layer == -1 ? 0 : 2, out, np);
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "maya-k31 expert server: %s\n", e.what());
        }
        ::close(fd);
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string file, dump_dir, seqs;
    int layers = 0, chunk = 0, bench_p = 0, bench_d = 0;
    bool gen = false;
    std::string serve_at;
    std::vector<std::string> ping;
    k31::ModelOptions opt;
    auto ints = [](const std::string& v, std::vector<int>& out) {   // "0,1"
        out.clear();
        size_t a = 0;
        while (a <= v.size()) {
            const size_t b = std::min(v.find(',', a), v.size());
            const std::string x = v.substr(a, b - a);
            if (x.empty() || x.find_first_not_of("0123456789") != std::string::npos) return false;
            out.push_back(std::atoi(x.c_str()));
            a = b + 1;
        }
        return !out.empty();
    };
    std::vector<std::string> seam_globs;
    bool info = false;
    k31::EpsPreset eps = k31::EpsPreset::moonshot;
    if (argc < 2) {
        std::fputs(kUsage, stderr);
        return 2;
    }
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--version" || a == "--help" || a == "-h") {
            // each stands alone: with anything else the command line is refused, wherever it comes
            if (argc != 2) {
                std::fprintf(stderr, "maya-k31: %s takes no other arguments\n", a.c_str());
                return 2;
            }
            if (a == "--version") std::printf("maya-k31 %s (build %s)\n", MAYA_RELEASE, MAYA_BUILD_ID);
            else std::fputs(kUsage, stdout);
            return 0;
        } else if (a == "--pack-info") {
            info = true;
        } else if (a == "--eps") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "maya-k31: --eps needs a value\n");
                return 2;
            }
            const std::string v = argv[++i];
            if (v == "moonshot") eps = k31::EpsPreset::moonshot;
            else if (v == "llama.cpp" || v == "llama_cpp") eps = k31::EpsPreset::llama_cpp;
            else {
                std::fprintf(stderr, "maya-k31: --eps takes moonshot or llama.cpp, not '%s'\n", v.c_str());
                return 2;
            }
        } else if (a == "--seams" && i + 1 < argc) {   // comma-separated globs of the seams to keep
            std::string v = argv[++i];
            size_t at = 0;
            while (at <= v.size()) {
                const size_t c = v.find(',', at);
                const std::string g = v.substr(at, c == std::string::npos ? std::string::npos : c - at);
                if (!g.empty()) seam_globs.push_back(g);
                if (c == std::string::npos) break;
                at = c + 1;
            }
        } else if (a == "--chunk" && i + 1 < argc) {
            chunk = std::atoi(argv[++i]);
            if (chunk < 1) {
                std::fprintf(stderr, "maya-k31: --chunk takes a token count >= 1\n");
                return 2;
            }
        } else if (a == "--vram-cap-mb" && i + 1 < argc) {   // per card, in --gpus order ("0,0,0,6500": 0 = no cap)
            if (!ints(argv[++i], opt.vram_cap_mb)) {
                std::fprintf(stderr, "maya-k31: --vram-cap-mb takes MB per card, comma separated (0: no cap)\n");
                return 2;
            }
        } else if ((a == "--gpus" || a == "--split") && i + 1 < argc) {
            if (!ints(argv[++i], a == "--gpus" ? opt.gpus : opt.split)) {
                std::fprintf(stderr, "maya-k31: %s takes a comma list of numbers\n", a.c_str());
                return 2;
            }
        } else if ((a == "--expert-slots" || a == "--vram-reserve-mb") && i + 1 < argc) {
            const std::string v = argv[++i];
            if (v.empty() || v.find_first_not_of("0123456789") != std::string::npos) {
                std::fprintf(stderr, "maya-k31: %s takes a count >= 0\n", a.c_str());
                return 2;
            }
            (a == "--expert-slots" ? opt.expert_slots : opt.vram_reserve_mb) = std::atoi(v.c_str());
        } else if ((a == "--cpu-lane" || a == "--cpu-max-tokens") && i + 1 < argc) {
            const std::string v = argv[++i];
            if (v.empty() || v.find_first_not_of("0123456789") != std::string::npos) {
                std::fprintf(stderr, "maya-k31: %s takes a count >= 0\n", a.c_str());
                return 2;
            }
            (a == "--cpu-lane" ? opt.cpu_threads : opt.cpu_max_tokens) = std::atoi(v.c_str());
        } else if (a == "--profile") {
            opt.profile = true;
        } else if (a == "--load" && i + 1 < argc) {
            const std::string v = argv[++i];
            if (v != "read" && v != "mmap" && v != "page") {
                std::fprintf(stderr, "maya-k31: --load takes read, mmap or page\n");
                return 2;
            }
            opt.load_mmap = v == "mmap";
            opt.page_experts = v == "page";
        } else if (a == "--prefetch" && i + 1 < argc) {
            opt.prefetch = std::atoi(argv[++i]);
            if (opt.prefetch < 1 || opt.prefetch > 256) {
                std::fprintf(stderr, "maya-k31: --prefetch takes 1..256 reader threads\n");
                return 2;
            }
        } else if (a == "--ram-slots" && i + 1 < argc) {
            const std::string v = argv[++i];
            char* end = nullptr;
            const long n = v == "auto" ? -1 : std::strtol(v.c_str(), &end, 10);
            opt.ram_slots = (int) n;
            if (v != "auto" && (v.empty() || *end || n < 0 || n > 10000000)) {
                std::fprintf(stderr, "maya-k31: --ram-slots takes a slot count or auto\n");
                return 2;
            }
        } else if (a == "--tier-prompt-cold") {
            opt.tier_prompt_cold = true;
        } else if (a == "--dense-override" && i + 1 < argc) {
            opt.dense_override = argv[++i];
        } else if (a == "--vram-dyn" && i + 1 < argc) {
            const std::string v = argv[++i];
            char* end = nullptr;
            const long n = v == "auto" ? -1 : std::strtol(v.c_str(), &end, 10);
            opt.vram_dyn = (int) n;
            if (v != "auto" && (v.empty() || *end || n < 0 || n > 1000000)) {
                std::fprintf(stderr, "maya-k31: --vram-dyn takes a slot count a card or auto\n");
                return 2;
            }
        } else if (a == "--vram-dyn-up" && i + 1 < argc) {
            opt.vram_dyn_up = std::atoi(argv[++i]);
            if (opt.vram_dyn_up < 1 || opt.vram_dyn_up > 100000) {
                std::fprintf(stderr, "maya-k31: --vram-dyn-up takes 1..100000\n");
                return 2;
            }
        } else if (a == "--remote-experts" && i + 1 < argc) {
            opt.remote_addr = argv[++i];
        } else if (a == "--remote-own" && i + 1 < argc) {
            opt.remote_own_path = argv[++i];
        } else if (a == "--remote-frac" && i + 1 < argc) {
            opt.remote_frac = std::atof(argv[++i]);
            if (!(opt.remote_frac > 0 && opt.remote_frac <= 1)) {
                std::fprintf(stderr, "maya-k31: --remote-frac takes a share in (0, 1]\n");
                return 2;
            }
        } else if (a == "--ram-readers" && i + 1 < argc) {
            opt.ram_readers = std::atoi(argv[++i]);
            if (opt.ram_readers < 1 || opt.ram_readers > 256) {
                std::fprintf(stderr, "maya-k31: --ram-readers takes 1..256\n");
                return 2;
            }
        } else if (a == "--car-rank" && i + 1 < argc) {
            opt.car_rank = std::atoi(argv[++i]);
            if (opt.car_rank < 1 || opt.car_rank > 16) {
                std::fprintf(stderr, "maya-k31: --car-rank takes 1..16 (the router's picks a token)\n");
                return 2;
            }
        } else if (a == "--car-prompt-rank" && i + 1 < argc) {
            opt.car_prompt_rank = std::atoi(argv[++i]);
            if (opt.car_prompt_rank < 1 || opt.car_prompt_rank > 16) {
                std::fprintf(stderr, "maya-k31: --car-prompt-rank takes 1..16 (the router's picks a token)\n");
                return 2;
            }
        } else if (a == "--car-prompt-rows" && i + 1 < argc) {
            opt.car_prompt_rows = std::atoi(argv[++i]);
            if (opt.car_prompt_rows < 1 || opt.car_prompt_rows > 1000000) {
                std::fprintf(stderr, "maya-k31: --car-prompt-rows takes a row count\n");
                return 2;
            }
        } else if (a == "--pregate") {
            opt.pregate = true;
        } else if (a == "--pregate-rank" && i + 1 < argc) {
            opt.pregate_rank = std::atoi(argv[++i]);
            if (opt.pregate_rank < 1 || opt.pregate_rank > 16) {
                std::fprintf(stderr, "maya-k31: --pregate-rank takes 1..16\n");
                return 2;
            }
        } else if (a == "--pregate-decode-only") {
            opt.pregate_decode_only = true;
        } else if (a == "--page-stats") {
            opt.page_stats = true;
        } else if (a == "--cpu-frac" && i + 1 < argc) {
            char* end = nullptr;
            opt.cpu_frac = std::strtod(argv[++i], &end);
            if (!end || *end || !(opt.cpu_frac >= 0.0 && opt.cpu_frac <= 1.0)) {
                std::fprintf(stderr, "maya-k31: --cpu-frac takes a share in [0, 1]\n");
                return 2;
            }
        } else if (a == "--cpu-frac-cards" && i + 1 < argc) {   // "0,0,0,1": per card in --gpus order
            std::string v = argv[++i];
            opt.cpu_frac_cards.clear();
            for (size_t b = 0; b <= v.size();) {
                const size_t c = std::min(v.find(',', b), v.size());
                char* end = nullptr;
                const std::string f = v.substr(b, c - b);
                const double x = std::strtod(f.c_str(), &end);
                if (f.empty() || !end || *end || !(x >= 0.0 && x <= 1.0)) {
                    std::fprintf(stderr, "maya-k31: --cpu-frac-cards takes shares in [0, 1], comma separated (one a card)\n");
                    return 2;
                }
                opt.cpu_frac_cards.push_back(x);
                b = c + 1;
            }
        } else if ((a == "--usage" || a == "--placement" || a == "--write-placement") && i + 1 < argc) {
            (a == "--usage" ? opt.usage_path : a == "--placement" ? opt.placement_in : opt.placement_out) = argv[++i];
        } else if (a == "--layers" && i + 1 < argc) {
            layers = std::atoi(argv[++i]);
            if (layers < 1) {
                std::fprintf(stderr, "maya-k31: --layers takes a count >= 1\n");
                return 2;
            }
        } else if ((a == "--dump-raw-dir" || a == "--golden-seqs") && i + 1 < argc) {
            (a == "--dump-raw-dir" ? dump_dir : seqs) = argv[++i];
        } else if (a == "--bench" && i + 1 < argc) {
            std::vector<int> pd;
            if (!ints(argv[++i], pd) || pd.size() != 2 || pd[0] < 1) {
                std::fprintf(stderr, "maya-k31: --bench takes PROMPT,DECODE token counts (prompt >= 1)\n");
                return 2;
            }
            bench_p = pd[0];
            bench_d = pd[1];
        } else if (a == "--gen") {
            gen = true;
        } else if (a == "--expert-server" && i + 1 < argc) {
            serve_at = argv[++i];   // PORT (127.0.0.1) or HOST:PORT
        } else if (a == "--remote-ping" && i + 5 < argc) {
            for (int j = 0; j < 5; ++j) ping.push_back(argv[++i]);
        } else if (a == "--serve" || a == "--score" || a == "--experts") {
            std::fprintf(stderr, "maya-k31: %s is not built yet (this build has --version, --pack-info, --dump-raw-dir)\n", a.c_str());
            return 2;
        } else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "maya-k31: unknown option %s\n", a.c_str());
            return 2;
        } else if (!file.empty()) {
            std::fprintf(stderr, "maya-k31: one GGUF file at a time (also given %s)\n", a.c_str());
            return 2;
        } else {
            file = a;
        }
    }
    if (!dump_dir.empty() || !seqs.empty()) {
        if (dump_dir.empty() || seqs.empty() || file.empty()) {
            std::fprintf(stderr, "maya-k31: --dump-raw-dir needs DIR, --golden-seqs JSON and a GGUF file\n");
            return 2;
        }
        opt.n_layers = layers;
        return dump_runs(file, seqs, dump_dir, eps, opt, chunk, seam_globs);
    }
    if (!ping.empty())
        return remote_ping(ping[0], std::atoi(ping[1].c_str()), std::atoi(ping[2].c_str()), std::atoi(ping[3].c_str()),
                           std::atoi(ping[4].c_str()));
    if (!serve_at.empty()) {
        if (file.empty()) return echo_server(serve_at);
        opt.n_layers = layers;
        k31::Model m;
        std::string err;
        if (!m.load(file, opt, err, eps)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "maya-k31 %s (build %s) --expert-server: %s\n%s", MAYA_RELEASE, MAYA_BUILD_ID, file.c_str(),
                     m.placement_summary().c_str());
        const int rc = m.serve(serve_at, err);
        if (rc) std::fprintf(stderr, "%s\n", err.c_str());
        return rc;
    }
    if (gen) {
        if (file.empty()) {
            std::fprintf(stderr, "maya-k31: --gen needs a GGUF file\n");
            return 2;
        }
        opt.n_layers = layers;
        return gen_loop(file, eps, opt, chunk);
    }
    if (bench_p > 0) {
        if (file.empty()) {
            std::fprintf(stderr, "maya-k31: --bench needs a GGUF file\n");
            return 2;
        }
        opt.n_layers = layers;
        return bench(file, eps, opt, chunk, bench_p, bench_d);
    }
    if (!info) {
        std::fprintf(stderr, "maya-k31: nothing to do: give --pack-info FILE, --dump-raw-dir or --version\n");
        return 2;
    }
    if (file.empty()) {
        std::fprintf(stderr, "maya-k31: --pack-info needs a GGUF file\n");
        return 2;
    }
    return pack_info(file, eps);
}
