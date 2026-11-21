// src/hy4/hy4_main.cpp - maya-hy4: Tencent's Hy4-flash (a GGUF set) on NVIDIA GPUs (CUDA) and AMD GPUs (HIP, Linux).
//
//   maya-hy4 --pack <model>/maya-hy4 --serve [--gpus 0,1,...] [--max-context N] [--chunk N] [--eos id,id,...]
//            [--drafting auto|on|off|mtp|dfly2 | --mtp | --dfly2 [--draft N] [--draft-floor P]] [--vram-experts F]
//            [--cpu-threads N] [--profile F] [--profile-out F]
//            [--tp N|auto] [--main-gpu I] [--vision on|off] [--kv auto|f16|q8]
//            [--kv-streaming on|off|auto] [--kv-resident N] [--kv-spill on|off|auto] [--kv-spill-dir D]
//            [--room-report] [--gpu-fetch off|auto|gpu|lane|K]
//
// --gpu-fetch (HY4_FETCH; off by default): the routed experts VRAM does not hold copied into pinned RAM, and the GPU
// fetching the ones a token routes to it over the bus itself - gpu: all of them on the GPU (the experts in VRAM follow
// the routing), K: each token's first K of them a layer on the GPU and the rest on the CPU lane beside it, lane: the
// CPU lane alone (as off), auto: whichever of them a short decode at the first start measures fastest on this PC
// (remembered; HY4_FETCH_MEASURE=1 or maya --speed-calibration measures again).  gpu / K change the answers in the last
// bits against the CPU lane's (each choice's own are the same from run to run, drafting or not).
// --room-report (the server's, at a first start on a PC whose cards it has no figure for): an INFO line says, for
// each GPU, its free VRAM as the plan read it and how much less of it this start's choices would hold with (the
// cache's rows, KV streaming, the drafter's room, the pictures, the layout) - worked out on the plan alone.
//
// --kv: the latent cache's rows - F16, or q8: the latent's 512 values int8 in blocks of 32 with an F16 scale, the
// rotated part's 64 F16 (672 bytes a row against 1152), the indexer's per-position and pooled rows F16 (the reference's
// own cache type) - about 58% of the VRAM a position; auto (the default) takes F16 wherever the context fits in VRAM
// with it and the 8-bit rows where it fits only with them - on one GPU, the layer split and tensor parallel (every
// member holding the whole cache) alike; an INFO line says so.
// The routed experts that do not fit in VRAM (or beyond --vram-experts, the share of each layer's kept there) run on
// the CPU lane from the model's file, --cpu-threads workers (default: a physical core each but the first); a prompt's
// chunk sends them to the GPU instead where this PC's PCIe is measured faster than its CPU.  --profile: an expert
// profile (uint32 [layers][experts] routings) - each layer's most routed experts are the ones kept in VRAM;
// --profile-out: where the routings this run counts (with the profile's) are written, after each request when serving
// and at the end otherwise.  HY4_HOT (on; HY4_HOT=0 off - the text differs from without it): where a layer's experts
// are partly in VRAM, the ones there follow the answers' kept tokens' routing - between requests only (an answer runs
// on one placement, so drafting on or off gives the same answers) cold ones trade places with hotter CPU-lane ones
// (HY4_HOT_SWAPS: at most that many a time; all by default); a start places each
// layer's highest scored experts in VRAM by HY4_HOT_FILE (<pack>/expert-hot.bin; the server's: its data folder),
// written only by a start that found none (or HY4_HOT_LEARN=1), so every start begins the same; HY4_HOT_LOG=1 logs.
//
// --draft-more N (1) / --draft-sure P (0.8) / --draft-streak S (1): up to N drafts more in a step - one for every S
// steps in a row before it whose drafts were all kept - while every draft of the step so far, and the added one, has a
// probability of at least P.  A longer window pays where the drafts hold (code), and costs where they do not: what
// was just kept says which it is.
// --mtp: the GGUF's MTP block drafts up to --draft tokens (3) a step, each verified by the model in one window with the
// token before it; a draft below --draft-floor (0.55: the block's probability of its own pick) ends the drafts.  Greedy:
// the drafts the model's argmax agrees with are kept; sampling: a draft d is kept with the model's probability of d,
// else the token is drawn from the model's distribution without d - every token is still the model's own sample.  A
// request's spec=0|1 turns the drafts off / on for it.
// --dfly2: the pack's DFly2 drafter instead (a few dense layers over the trunk's own caches: about a quarter of the MTP
// block's VRAM and nothing added to a prompt's read) - up to --draft tokens (the drafter's own count, 5; its floor 0.8)
// from one pass, verified and kept by the same rules.  One drafter at a time; one GPU's layout for now.
// The VRAM plan counts the room of the drafter the start loads, none without one (every byte to the routed experts):
// within a start the drafts on and off (a request's spec) give the same answers; a start with another drafter, or
// none, plans otherwise and may word an answer differently.  A drafter whose room would change more than where
// experts sit (the cache's rows, the dense prompt span, KV streaming, the GPUs' layout) is not loaded (said); an auto
// prompt chunk is the one beside its room.  --drafting auto (the server's): no drafting, the MTP block or the DFly2
// drafter - whichever this PC decodes fastest, each loaded once with its own plan and timed at the first start (before
// READY), remembered per PC, GPUs, model and options (HY4_DRAFT_FILE; pick_drafting); on: the faster drafter of the
// two; off / mtp / dfly2 (--mtp, --dfly2): by hand, not measured.
//
// --serve speaks what serve/server.py's StrataEngine expects (maya-mm3's protocol, one conversation): INFO lines, then
// `READY <context> stop`; per request `GEN <max_new> [key=value ...] <id,id,...>` answered by `PP <read> <total> <ms>
// <tok/s> <reused>` while the prompt is read, `T <id>` lines, `STAT ...` every 8 tokens and `DONE <generated> <prompt>
// <prompt ms> <decode ms> <stop|length|cancel> 0 0 <reused> 0 0`; `STOP` ends a request early, `WRAP` closes a
// thinking block still open, `QUIT` the engine.  A prompt that shares a start with the state (the last conversation)
// reads only the rest, when the sliding-window caches still hold the window before it (else all of it).  Keys:
// temperature, top_p, top_k, min_p, seed, penalty_last_n / _repeat / _freq / _present, think_budget + think_end.
//
// Pictures (the pack's vision tower: on unless --vision off, `INFO vision=hy4 ...` then): `GENI <max_new> [keys] <file>
// <ids>` - the file holds the prompt's pictures (see read_pics), each one's image tokens between its start and end of
// image tokens taking its rows; the state is shared only where the same picture sits at the same place.
//
//   maya-hy4 --pack <model>/maya-hy4 --ids @prompt.txt [--gen N] [--gpus 0,1,...] [--max-context N] [--chunk N]
//
// --ids: comma- or whitespace-separated token ids, or @FILE holding them; --gen: greedy tokens after the prompt (their
// ids printed, and the top 5 of each step's logits with --top); --images FILE: the prompt's pictures (GENI's file).
//
//   maya-hy4 --pack <model>/maya-hy4 --encode-image FILE.rgb --size WxH --dump-image OUT [--gpus I]
//
// The vision tower alone (no model loaded): a picture's raw RGB bytes -> its rows, F32 [tokens][3072], into OUT.
//
//   maya-hy4 --pack <model>/maya-hy4 --kld ref.kld [--chunks N] [--kld-save OUT]
//
// The model against a reference's logits: ref.kld is a perplexity tool's --kl-divergence-base file (AngelSlim's
// hy4-flash build); each chunk's tokens are fed from position 0 and every position the file scores compared - the mean
// KL divergence (reference || this), how often the two pick the same top token, and both perplexities.  --kld-save
// OUT: this run's own log-probabilities in the same format, a base for another setting's --kld.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <atomic>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <mutex>
#include <random>
#include <thread>

#include "hy4_kernels.cuh"
#include "hy4_lane.hpp"
#include "hy4_model.hpp"
#include "hy4_pack.hpp"
#include "mm3_os.hpp"
#include "mm3_sample.hpp"

namespace {

using mm3::sample::Req;

std::vector<int> parse_ids(const std::string& a) {
    std::string src = a;
    if (!a.empty() && a[0] == '@') {
        std::ifstream f(a.substr(1));
        std::stringstream ss;
        ss << f.rdbuf();
        src = ss.str();
    }
    for (char& c : src)
        if (c == ',' || c == '[' || c == ']') c = ' ';
    std::vector<int> v;
    std::istringstream is(src);
    for (long x; is >> x;) v.push_back((int) x);
    return v;
}
std::vector<int> parse_list(const std::string& a) {
    std::vector<int> v;
    std::istringstream is(a);
    for (std::string x; std::getline(is, x, ',');)
        if (!x.empty()) v.push_back(std::atoi(x.c_str()));
    return v;
}
double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// The largest value's index (of equals the lowest): the maximum first, in 16 lanes - a loop the compiler vectorizes,
// where one that carries the index goes a value at a time (a row of logits is 120 thousand, a decode round reads a few)
// - then the first index that holds it
int argmax(const float* r, int n) {
    int i = 0;
    float mx = r[0];
    if (n >= 32) {
        float m[16];
        for (int j = 0; j < 16; ++j) m[j] = r[j];
        for (i = 16; i + 16 <= n; i += 16)
            for (int j = 0; j < 16; ++j) m[j] = r[i + j] > m[j] ? r[i + j] : m[j];
        for (int j = 0; j < 16; ++j) mx = m[j] > mx ? m[j] : mx;
    }
    for (; i < n; ++i) mx = r[i] > mx ? r[i] : mx;
    for (i = 0; i + 16 <= n; i += 16) {
        int any = 0;
        for (int j = 0; j < 16; ++j) any |= r[i + j] == mx;
        if (any) break;
    }
    for (; i < n; ++i)
        if (r[i] == mx) return i;
    int b = 0;   // (a NaN among them: the plain walk)
    for (i = 1; i < n; ++i)
        if (r[i] > r[b]) b = i;
    return b;
}
// log_softmax of a row into lp (double sums)
void log_softmax(const float* r, int n, std::vector<double>& lp) {
    double mx = r[0];
    for (int i = 1; i < n; ++i) mx = std::max(mx, (double) r[i]);
    double s = 0;
    for (int i = 0; i < n; ++i) s += std::exp((double) r[i] - mx);
    const double l = mx + std::log(s);
    lp.resize((size_t) n);
    for (int i = 0; i < n; ++i) lp[(size_t) i] = (double) r[i] - l;
}

// --trace <from> (tests): the model's 5 likeliest next tokens after each position >= from of the prompt, one line a
// position on stdout: `<position> <id>:<logprob> x5` - a quant's own reference compares its top tokens with them.
// The prompt up to `from` in one call (the prompt path), then windows with every row's logits (as run_kld)
int run_trace(hy4::Model& m, const std::vector<int>& ids, int from) {
    const int n = (int) ids.size(), nv = m.vocab();
    from = std::max(1, std::min(from, n));
    std::vector<float> logits((size_t) hy4::k::kMaxWin * nv);
    std::vector<double> lp;
    std::vector<int> ix((size_t) nv);
    std::string err;
    const double t0 = now();
    if (from > 1 && !m.forward(ids.data(), from - 1, 0, 0, nullptr, err)) {
        std::fprintf(stderr, "forward: %s\n", err.c_str());
        return 1;
    }
    const double t1 = now();
    for (int p = from - 1; p < n; p += hy4::k::kMaxWin) {
        const int T = std::min(hy4::k::kMaxWin, n - p);
        if (!m.forward(ids.data() + p, T, p, T, logits.data(), err)) {
            std::fprintf(stderr, "forward at %d: %s\n", p, err.c_str());
            return 1;
        }
        for (int t = 0; t < T; ++t) {
            const float* me = logits.data() + (size_t) t * nv;
            log_softmax(me, nv, lp);
            for (int j = 0; j < nv; ++j) ix[(size_t) j] = j;
            std::partial_sort(ix.begin(), ix.begin() + 5, ix.end(), [&](int a, int b) { return me[a] > me[b]; });
            std::printf("%d", p + t);
            for (int j = 0; j < 5; ++j) std::printf(" %d:%.4f", ix[(size_t) j], lp[(size_t) ix[(size_t) j]]);
            std::printf("\n");
        }
    }
    const double t2 = now();
    std::fprintf(stderr, "trace: prompt %d tokens in %.2f s (%.1f tok/s), %d positions in windows in %.2f s (%.1f tok/s)\n",
                 from - 1, t1 - t0, from > 1 ? (from - 1) / (t1 - t0) : 0.0, n - from + 1, t2 - t1,
                 (n - from + 1) / (t2 - t1));
    return 0;
}

// (save: this run's own log-probabilities written in the same format - a base for another run's --kld, so two
//  settings of this engine are compared with each other, the reference's own noise left out)
int run_kld(hy4::Model& m, const std::string& path, int chunks, const std::string& save = "") {
    std::ifstream f(path, std::ios::binary);
    char magic[8];
    int32_t n_ctx = 0, n_vocab = 0, n_chunk = 0;
    if (!f.read(magic, 8) || std::memcmp(magic, "_logits_", 8) != 0 || !f.read((char*) &n_ctx, 4) ||
        !f.read((char*) &n_vocab, 4) || !f.read((char*) &n_chunk, 4)) {
        std::fprintf(stderr, "%s: not a --kl-divergence-base logits file\n", path.c_str());
        return 1;
    }
    if (n_vocab != m.vocab()) {
        std::fprintf(stderr, "%s: vocabulary %d, the model's %d\n", path.c_str(), n_vocab, m.vocab());
        return 1;
    }
    std::vector<int32_t> tokens((size_t) n_chunk * n_ctx);
    f.read((char*) tokens.data(), (std::streamsize) (tokens.size() * 4));
    const int first = n_ctx / 2, nrow = n_ctx - 1 - first, nv = 2 * ((n_vocab + 1) / 2) + 4;
    chunks = chunks > 0 ? std::min(chunks, n_chunk) : n_chunk;
    std::vector<uint16_t> ref((size_t) nrow * nv);
    std::vector<float> logits((size_t) hy4::k::kMaxWin * n_vocab);
    std::ofstream out;
    if (!save.empty()) {
        out.open(save, std::ios::binary);
        const int32_t hd[3] = {n_ctx, n_vocab, chunks};
        out.write("_logits_", 8);
        out.write((const char*) hd, sizeof hd);
        out.write((const char*) tokens.data(), (std::streamsize) ((size_t) chunks * n_ctx * 4));
        if (!out) {
            std::fprintf(stderr, "%s: not written\n", save.c_str());
            return 1;
        }
    }
    std::vector<uint16_t> mine(save.empty() ? 0 : (size_t) nv);
    std::vector<double> lp;
    double kl = 0, nll_me = 0, nll_ref = 0;
    long n = 0, same = 0;
    std::vector<double> kls;
    for (int c = 0; c < chunks; ++c) {
        if (!f.read((char*) ref.data(), (std::streamsize) (ref.size() * 2))) {
            std::fprintf(stderr, "%s: chunk %d cut short\n", path.c_str(), c);
            return 1;
        }
        const int* ids = tokens.data() + (size_t) c * n_ctx;
        const double t0 = now();
        // the prompt up to `first` in one call, then windows of kMaxWin with every row's logits
        std::string err;
        if (!m.forward(ids, first, 0, 0, nullptr, err)) {
            std::fprintf(stderr, "forward: %s\n", err.c_str());
            return 1;
        }
        const double t1 = now();
        for (int p = first; p < n_ctx - 1; p += hy4::k::kMaxWin) {
            const int T = std::min(hy4::k::kMaxWin, n_ctx - 1 - p);
            if (!m.forward(ids + p, T, p, T, logits.data(), err)) {
                std::fprintf(stderr, "forward at %d: %s\n", p, err.c_str());
                return 1;
            }
            for (int t = 0; t < T; ++t) {
                const int row = p + t - first;
                const float* me = logits.data() + (size_t) t * n_vocab;
                const uint16_t* rr = ref.data() + (size_t) row * nv;
                float scale, minlp;
                std::memcpy(&scale, rr, 4);
                std::memcpy(&minlp, rr + 2, 4);
                const uint16_t* q = rr + 4;
                log_softmax(me, n_vocab, lp);
                if (out.is_open()) {   // (the row: scale, min, then 65535 steps from min to max)
                    double lo = lp[0], hi = lp[0];
                    for (int i = 1; i < n_vocab; ++i) {
                        lo = std::min(lo, lp[(size_t) i]);
                        hi = std::max(hi, lp[(size_t) i]);
                    }
                    const float sc = (float) ((hi - lo) / 65535.0), mn = (float) lo;
                    std::memcpy(mine.data(), &sc, 4);
                    std::memcpy(mine.data() + 2, &mn, 4);
                    for (int i = 0; i < n_vocab; ++i)
                        mine[(size_t) 4 + i] = (uint16_t) std::lround(std::min(65535.0, std::max(0.0, (lp[(size_t) i] - mn) / sc)));
                    for (int i = n_vocab + 4; i < nv; ++i) mine[(size_t) i] = 0;
                    out.write((const char*) mine.data(), (std::streamsize) (mine.size() * 2));
                }
                double k = 0;
                int ra = 0;
                for (int i = 0; i < n_vocab; ++i) {
                    const double lr = (double) minlp + (double) scale * q[i];
                    const double pr = std::exp(lr);
                    k += pr * (lr - lp[(size_t) i]);
                    if (q[i] > q[ra]) ra = i;
                }
                const int tok = ids[p + t + 1];
                nll_me -= lp[(size_t) tok];
                nll_ref -= (double) minlp + (double) scale * q[tok];
                kl += k;
                kls.push_back(k);
                same += argmax(me, n_vocab) == ra;
                ++n;
            }
        }
        const double t2 = now();
        std::printf("chunk %d: prompt %d tokens %.1f s (%.0f tok/s), %d scored in %.1f s; running KL %.5f, top-1 same "
                    "%.2f%%\n",
                    c, first, t1 - t0, first / (t1 - t0), nrow, t2 - t1, kl / n, 100.0 * same / n);
        std::fflush(stdout);
    }
    std::sort(kls.begin(), kls.end());
    std::printf("KLD over %ld positions: mean %.6f, median %.6f, p99 %.6f, max %.6f\n", n, kl / n, kls[kls.size() / 2],
                kls[(size_t) (kls.size() * 0.99)], kls.back());
    std::printf("top-1 same: %.3f%%;  PPL this %.4f, reference %.4f\n", 100.0 * same / n, std::exp(nll_me / n),
                std::exp(nll_ref / n));
    return 0;
}

// --kld-topk FILE (tests): the model against the quants' own oracle - a top-k reference ("_hy4top_": n_ctx, k, rows;
// tokens [rows][n_ctx] int32, mask [rows][n_ctx-1] u8, logp [rows][n_ctx][k] f32, idx [rows][n_ctx][k] int32), scored
// as the quantizer scores (top-k KL with the tails lumped, the same top token), the masked positions of each row's
// second half: the first half in one call (the prompt path), then windows of kMaxWin (the token path).  save: a line
// a scored position "row pos kl same" (paired comparisons)
int run_kld_topk(hy4::Model& m, const std::string& path, int chunks, const std::string& save) {
    std::ifstream f(path, std::ios::binary);
    char magic[8];
    int32_t hd[3] = {0, 0, 0};
    if (!f.read(magic, 8) || std::memcmp(magic, "_hy4top_", 8) != 0 || !f.read((char*) hd, sizeof hd)) {
        std::fprintf(stderr, "%s: not a top-k reference\n", path.c_str());
        return 1;
    }
    const int n_ctx = hd[0], K = hd[1], rows = hd[2], nv = m.vocab();
    std::vector<int32_t> tok((size_t) rows * n_ctx);
    std::vector<uint8_t> mask((size_t) rows * (n_ctx - 1));
    std::vector<float> rlp((size_t) rows * n_ctx * K);
    std::vector<int32_t> rix((size_t) rows * n_ctx * K);
    f.read((char*) tok.data(), (std::streamsize) (tok.size() * 4));
    f.read((char*) mask.data(), (std::streamsize) mask.size());
    f.read((char*) rlp.data(), (std::streamsize) (rlp.size() * 4));
    if (!f.read((char*) rix.data(), (std::streamsize) (rix.size() * 4))) {
        std::fprintf(stderr, "%s: cut short\n", path.c_str());
        return 1;
    }
    std::FILE* out = save.empty() ? nullptr : std::fopen(save.c_str(), "w");
    const int use = chunks > 0 ? std::min(chunks, rows) : rows, first = n_ctx / 2;
    std::vector<float> logits((size_t) hy4::k::kMaxWin * nv);
    std::vector<double> lp, kls;
    double kl = 0;
    long n = 0, same = 0;
    for (int r = 0; r < use; ++r) {
        const int* ids = tok.data() + (size_t) r * n_ctx;
        std::string err;
        const double t0 = now();
        if (!m.forward(ids, first, 0, 0, nullptr, err)) {
            std::fprintf(stderr, "forward: %s\n", err.c_str());
            return 1;
        }
        for (int p = first; p < n_ctx - 1; p += hy4::k::kMaxWin) {
            const int T = std::min(hy4::k::kMaxWin, n_ctx - 1 - p);
            if (!m.forward(ids + p, T, p, T, logits.data(), err)) {
                std::fprintf(stderr, "forward at %d: %s\n", p, err.c_str());
                return 1;
            }
            for (int t = 0; t < T; ++t) {
                const int pos = p + t;
                if (!mask[(size_t) r * (n_ctx - 1) + pos]) continue;
                const float* me = logits.data() + (size_t) t * nv;
                log_softmax(me, nv, lp);
                const float* rl = rlp.data() + ((size_t) r * n_ctx + pos) * K;
                const int32_t* ri = rix.data() + ((size_t) r * n_ctx + pos) * K;
                double k = 0, pr_s = 0, q_s = 0;
                for (int j = 0; j < K; ++j) {
                    const double pr = std::exp((double) rl[j]), q = lp[(size_t) ri[j]];
                    k += pr * ((double) rl[j] - q);
                    pr_s += pr;
                    q_s += std::exp(q);
                }
                const double pt = std::max(1e-9, 1.0 - pr_s), qt = std::max(1e-9, 1.0 - q_s);
                k += pt * (std::log(pt) - std::log(qt));
                const int s = argmax(me, nv) == ri[0];
                kl += k;
                same += s;
                ++n;
                kls.push_back(k);
                if (out) std::fprintf(out, "%d %d %.6g %d\n", r, pos, k, s);
            }
        }
        std::printf("row %d: %.1f s; running KL %.5f, top-1 same %.2f%% (%ld positions)\n", r, now() - t0,
                    n ? kl / n : 0.0, n ? 100.0 * same / n : 0.0, n);
        std::fflush(stdout);
    }
    if (out) std::fclose(out);
    if (kls.empty()) return 1;
    std::sort(kls.begin(), kls.end());
    std::printf("KLD-topk over %ld positions: mean %.6f, median %.6f, p99 %.6f; top-1 same %.3f%%\n", n, kl / n,
                kls[kls.size() / 2], kls[(size_t) (kls.size() * 0.99)], 100.0 * same / n);
    return 0;
}

// stdin's lines on a thread of their own: STOP / WRAP are seen while a request runs
struct Lines {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    bool eof = false;
    std::atomic<bool> stop{false}, wrap{false};
    void start() {
        std::thread([this] {
            std::string buf;
            char chunk[4096];
            for (;;) {
                const long n = mm3::os::read_stdin(chunk, sizeof chunk);
                if (n <= 0) break;
                buf.append(chunk, (size_t) n);
                for (size_t nl; (nl = buf.find('\n')) != std::string::npos;) {
                    std::string l = buf.substr(0, nl);
                    buf.erase(0, nl + 1);
                    if (!l.empty() && l.back() == '\r') l.pop_back();
                    if (l == "STOP" || l.rfind("STOP ", 0) == 0) {
                        stop.store(true);
                        continue;
                    }
                    if (l == "WRAP") {
                        wrap.store(true);
                        continue;
                    }
                    std::lock_guard<std::mutex> lk(mu);
                    q.push_back(std::move(l));
                    cv.notify_one();
                }
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            cv.notify_one();
        }).detach();
    }
    bool next(std::string& out) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !q.empty() || eof; });
        if (q.empty()) return false;
        out = std::move(q.front());
        q.pop_front();
        return true;
    }
};

bool parse_gen(const std::string& line, Req& r) {
    const bool geni = line.rfind("GENI ", 0) == 0;
    std::istringstream is(line.substr(geni ? 5 : 4));
    if (!(is >> r.max_new)) return false;
    for (std::string tok; is >> tok;) {
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) {
            if (geni && r.file.empty()) {
                r.file = tok;
                continue;
            }
            r.ids = parse_ids(tok);
            break;
        }
        const std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
        if (k == "temperature") r.temperature = std::strtof(v.c_str(), nullptr);
        else if (k == "top_p") r.top_p = std::strtof(v.c_str(), nullptr);
        else if (k == "top_k") r.top_k = std::atoi(v.c_str());
        else if (k == "min_p") r.min_p = std::strtof(v.c_str(), nullptr);
        else if (k == "seed") r.seed = std::strtoull(v.c_str(), nullptr, 10);
        else if (k == "penalty_last_n") r.pen_n = std::atoi(v.c_str());
        else if (k == "penalty_repeat") r.pen_repeat = std::strtof(v.c_str(), nullptr);
        else if (k == "penalty_freq") r.pen_freq = std::strtof(v.c_str(), nullptr);
        else if (k == "penalty_present") r.pen_present = std::strtof(v.c_str(), nullptr);
        else if (k == "think_budget") r.think_budget = std::atoll(v.c_str());
        else if (k == "think_end") r.think_end = std::atoi(v.c_str());
        else if (k == "spec") r.spec = std::atoi(v.c_str()) != 0;
        else if (k == "dspark_conf" || k == "spec_min_p") r.conf = std::strtof(v.c_str(), nullptr);
        // (keys this engine does not use are taken and ignored)
    }
    return !r.ids.empty();
}

// GENI's file (serve/hy4_vision.py, maya-mm3's format): int32 'MMI1', int32 count, then per picture int64 start (the
// position of its first image token), int32 width, int32 height, height x width x 3 RGB bytes
struct Pic {
    long long start = 0;
    int width = 0, height = 0;
    std::vector<uint8_t> rgb;
    uint64_t hash = 0;   // (FNV-1a over its size and pixels: the state's pictures told apart)
};

bool read_pics(const std::string& path, std::vector<Pic>& v, std::string& err) {
    v.clear();
    std::ifstream f(path, std::ios::binary);
    char magic[4];
    int32_t n = 0;
    if (!f.read(magic, 4) || std::memcmp(magic, "MMI1", 4) != 0 || !f.read((char*) &n, 4) || n < 0 || n > 4096) {
        err = path + ": not a pictures file (MMI1)";
        return false;
    }
    for (int i = 0; i < n; ++i) {
        Pic q;
        int64_t st = 0;
        int32_t wh[2] = {0, 0};
        if (!f.read((char*) &st, 8) || !f.read((char*) wh, 8) || wh[0] <= 0 || wh[1] <= 0 ||
            (long long) wh[0] * wh[1] > (1LL << 26)) {
            err = path + ": picture " + std::to_string(i) + " is cut short";
            return false;
        }
        q.start = st;
        q.width = wh[0];
        q.height = wh[1];
        q.rgb.resize((size_t) wh[0] * wh[1] * 3);
        if (!f.read((char*) q.rgb.data(), (std::streamsize) q.rgb.size())) {
            err = path + ": picture " + std::to_string(i) + " is cut short";
            return false;
        }
        uint64_t h = 1469598103934665603ull;
        auto mix = [&](const uint8_t* b, size_t k) {
            for (size_t j = 0; j < k; ++j) h = (h ^ b[j]) * 1099511628211ull;
        };
        mix((const uint8_t*) wh, 8);
        mix(q.rgb.data(), q.rgb.size());
        q.hash = h;
        v.push_back(std::move(q));
    }
    return true;
}

// the pictures' spans in ids: the start of image token, its image tokens ((height / 32) * (width / 32 + 1) of them),
// the end of image token - and no image token outside one; their rows for those that end at or after position `from`
// (the others' positions are the state's already)
bool picture_rows(hy4::Model& m, const std::vector<Pic>& pics, const std::vector<int>& ids, size_t from,
                  std::vector<hy4::Model::Image>& out, std::string& err) {
    out.clear();
    const hy4::VisionGeo* vg = m.vision();
    if (!vg) {
        err = "this engine has no vision tower (a pack with the mmproj file, --vision on), so it cannot read pictures";
        return false;
    }
    std::vector<char> covered(ids.size(), 0);
    for (size_t i = 0; i < pics.size(); ++i) {
        const Pic& q = pics[i];
        const long long n = vg->tokens(q.width, q.height);
        if (n == 0) {
            err = "picture " + std::to_string(i) + ": " + std::to_string(q.width) + " x " + std::to_string(q.height) +
                  " pixels (sides multiples of " + std::to_string(vg->factor()) + ", at most " +
                  std::to_string(vg->max_pixels) + " pixels)";
            return false;
        }
        if (q.start < 1 || q.start + n >= (long long) ids.size() || ids[(size_t) q.start - 1] != vg->image_start ||
            ids[(size_t) (q.start + n)] != vg->image_end) {
            err = "picture " + std::to_string(i) + "'s span (" + std::to_string(n) + " image tokens at " +
                  std::to_string(q.start) + ") is not between the prompt's start and end of image tokens";
            return false;
        }
        for (long long k = q.start; k < q.start + n; ++k) {
            if (ids[(size_t) k] != vg->image_token || covered[(size_t) k]) {
                err = "picture " + std::to_string(i) + "'s span is not a run of image tokens of its own";
                return false;
            }
            covered[(size_t) k] = 1;
        }
        if (q.start + n > (long long) from) {
            hy4::Model::Image im;
            im.start = q.start;
            if (!m.encode_image(q.rgb.data(), q.width, q.height, im.rows, err)) return false;
            out.push_back(std::move(im));
        }
    }
    for (size_t k = 0; k < ids.size(); ++k)
        if (ids[k] == vg->image_token && !covered[k]) {
            err = "an image token at " + std::to_string(k) + " without its picture";
            return false;
        }
    return true;
}

// A row's next-token distribution, as mm3::sample::dist gives it, without a copy of the whole row where it is not
// needed: greedy without penalties is the row's argmax (ties: the lower id); sampling without penalties takes the
// tokens within 40 nats of the top (temperature-scaled) - the rest hold under 1e-12 of the mass - to mm3's candidate
// form.  Penalties: the whole row, as before.
// (top: the row's largest value's index when the engine has it - Model::row_top - else -1, the walk)
std::vector<std::pair<int, double>> row_dist(const float* row, int V, const mm3::sample::Req& r,
                                             const std::vector<int>& hist, int top = -1) {
    if (mm3::sample::penalised(r)) return mm3::sample::dist(row, V, r, hist);
    int b = top;
    if (b < 0) {
        b = 0;
        for (int i = 1; i < V; ++i)
            if (row[i] > row[b]) b = i;
    }
    if (!(r.temperature > 0.0f)) return {{b, 1.0}};
    if (r.top_k > 0 && r.top_k < V) {
        // top_k: the k largest (ties: the lower id, as dist_c's sort) without sorting the row - every block of 1024's
        // largest value (a pass), the k-th largest of those a floor the k largest are all at or above (k distinct
        // values reach it), the values at or above it (a second pass: a few hundred) to dist_c
        const int bs = 1024, nb = (V + bs - 1) / bs;
        const size_t k = (size_t) r.top_k;
        if ((size_t) nb >= k) {
            std::vector<float> bm((size_t) nb);
            for (int q = 0; q < nb; ++q) {
                const int a0 = q * bs, a1 = std::min(V, a0 + bs);
                float mx = row[a0];
                for (int i = a0 + 1; i < a1; ++i) mx = row[i] > mx ? row[i] : mx;
                bm[(size_t) q] = mx;
            }
            std::nth_element(bm.begin(), bm.begin() + (long) (k - 1), bm.end(), std::greater<float>());
            const float floor_v = bm[k - 1];
            std::vector<std::pair<int, float>> c;
            for (int i = 0; i < V; ++i)
                if (row[i] >= floor_v) c.push_back({i, row[i]});
            return mm3::sample::dist_c(std::move(c), {}, {}, r, hist);
        }
    }
    const float cut = row[b] - 40.0f * r.temperature;
    if (!(r.top_k > 0) && !(r.top_p < 1.0f) && !(r.min_p > 0.0f)) {
        // plain sampling: the softmax itself, nothing to cut or sort - the top first, the others by id (a draw from
        // them is a draw from the distribution, whatever their order)
        std::vector<std::pair<int, double>> d;
        d.push_back({b, 1.0});
        double z = 1.0;
        const double inv_t = 1.0 / r.temperature;
        for (int i = 0; i < V; ++i)
            if (i != b && row[i] >= cut) {
                const double q = std::exp((double) (row[i] - row[b]) * inv_t);
                d.push_back({i, q});
                z += q;
            }
        for (auto& e : d) e.second /= z;
        return d;
    }
    std::vector<std::pair<int, float>> c;
    for (int i = 0; i < V; ++i)
        if (row[i] >= cut) c.push_back({i, row[i]});
    return mm3::sample::dist_c(std::move(c), {}, {}, r, hist);
}

// Plain sampling (a temperature, no penalties, no top_k / top_p / min_p): a row's weights e[i] = exp((x_i - max) / T)
// in float (a polynomial on [-0.5, 0.5] after the power of two, ~1e-7 relative - a loop the compiler vectorizes), their
// sum, and a walk over them only for a draw: a round's few rows cost a pass each, not a sorted copy of the vocabulary
inline float exp_neg(float x) {   // x <= 0
    x = x < -87.0f ? -87.0f : x;
    const float t = x * 1.44269504f;
    const float n = (t + 12582912.0f) - 12582912.0f;   // (round to nearest)
    const float f = t - n;
    const float p =
        1.0f + f * (0.693147182f + f * (0.240226507f + f * (0.0555041087f + f * (0.00961812911f +
                                                                                   f * (0.00133335581f + f * 0.000154035304f)))));
    int32_t bits;
    std::memcpy(&bits, &p, 4);
    bits += (int32_t) n << 23;
    float r;
    std::memcpy(&r, &bits, 4);
    return r;
}
struct Weights {
    int top = 0;
    double z = 0;
    std::vector<float> e;
    void of(const float* row, int V, float temperature, int top_known = -1) {
        top = top_known >= 0 ? top_known : argmax(row, V);
        e.resize((size_t) V);
        const float m = row[top], it = 1.0f / temperature;
        float* w = e.data();
        for (int i = 0; i < V; ++i) w[i] = exp_neg((row[i] - m) * it);
        float part[16] = {0};
        int i = 0;
        for (; i + 16 <= V; i += 16)
            for (int j = 0; j < 16; ++j) part[j] += w[i + j];
        double sz = 0;
        for (int j = 0; j < 16; ++j) sz += part[j];
        for (; i < V; ++i) sz += w[i];
        z = sz;
    }
    double p(int t) const { return e[(size_t) t] / z; }
    // a draw (u in [0, 1)), without `skip` (renormalised; -1: none)
    int draw(double u, int skip = -1) const {
        double tot = z - (skip >= 0 ? e[(size_t) skip] : 0.0);
        if (!(tot > 0)) return top;
        double target = u * tot, acc = 0;
        int last = top;
        for (int i = 0; i < (int) e.size(); ++i) {
            if (i == skip) continue;
            acc += e[(size_t) i];
            last = i;
            if (acc > target) return i;
        }
        return last;
    }
};
inline bool plain_sampling(const mm3::sample::Req& r) {
    return r.temperature > 0.0f && !mm3::sample::penalised(r) && !(r.top_k > 0) && !(r.top_p < 1.0f) &&
           !(r.min_p > 0.0f);
}

// The drafts a step may add to its usual ones: one more for every `need` steps in a row whose drafts were all kept, up
// to n_more (need 0: n_more always) - and none for a while once the added drafts are not being kept: an added draft
// costs a draft step and a token more in the window, about half of what a kept token is worth, so while fewer than
// half of the recent ones held (the first one counts: nothing is assumed of a text) they are left out for a number of
// steps that doubles each time, until one is kept again.
struct Extra {
    int n_more = 0, need = 1;
    int streak = 0, pause = 0, wait = 16;
    double rate = 0.55;   // (the added drafts kept, lately)
    int now() {
        if (pause > 0) {
            --pause;
            return 0;
        }
        return need > 0 ? std::min(n_more, streak / need) : n_more;
    }
    // a step's k drafts in the window, `kept` of them kept; base: its usual drafts
    void seen(int k, int kept, int base) {
        streak = k > 0 && kept == k ? streak + 1 : 0;
        if (k <= base) return;
        const bool held = kept > base;
        rate = 0.875 * rate + (held ? 0.125 : 0.0);
        if (held) wait = 16;
        if (rate < 0.5) {
            pause = wait;
            wait = std::min(wait * 2, 512);
            rate = 0.55;
        }
    }
};

int serve(hy4::Model& m, const std::vector<int>& eos, int n_draft, float floor0, int n_more, float sure, int need) {
    for (const auto& l : m.info_lines()) std::printf("NOTE %s\n", l.c_str());
    const auto& st0 = m.stats();
    std::printf("INFO engine=hy4 engine_kind=hy4 model=Hy4-flash ctx=%lld vram_gb=%.2f dspark=0 mtp=%d experts=%d "
                "vram_slots=%d\n",
                (long long) m.max_ctx(), st0.vram_gb, m.has_mtp() && n_draft > 0 ? 1 : 0, st0.experts_all,
                st0.experts_vram);
    if (m.has_dfly2()) std::printf("INFO dfly2=%d draft=%d\n", n_draft > 0 ? 1 : 0, n_draft);
    if (!m.room_report().empty()) std::printf("INFO %s\n", m.room_report().c_str());   // (--room-report)
    if (const hy4::VisionGeo* vg = m.vision())   // (serve/hy4_vision.py's Geometry.from_info)
        std::printf("INFO vision=hy4 image_token=%d image_start=%d image_end=%d vision_patch=%d vision_merge=%d "
                    "vision_max_pixels=%d vision_min_pixels=%d vision_room=%s vision_room_mib=%zu\n", vg->image_token,
                    vg->image_start, vg->image_end, vg->patch, vg->merge, vg->max_pixels, vg->min_pixels,
                    m.vision_room(), m.vision_room_bytes() >> 20);
    else
        for (const auto& l : m.info_lines())
            if (l.rfind("NOTE vision off", 0) == 0) {   // (a tower that did not load: the dashboard says why)
                std::printf("INFO vision=off\n");
                break;
            }
    std::printf("READY %lld stop\n", (long long) m.max_ctx());
    std::fflush(stdout);
    std::fprintf(stderr, "maya hy4 serve: GEN <max_new> [keys] <ids>; GENI <max_new> [keys] <file> <ids>; STOP, WRAP, "
                         "QUIT\n");
#if defined(SIGPIPE)
    std::signal(SIGPIPE, SIG_IGN);
#endif
    // (never freed: its reader thread is detached and wakes at stdin's end, which can come after QUIT while the model
    //  is still being torn down - a Lines on this stack would be gone by then and its mutex with it)
    Lines& in = *new Lines;
    in.start();
    std::vector<int> state;   // the tokens in the caches, positions 0 ..
    struct Held {
        long long start;
        uint64_t hash;
    };
    std::vector<Held> held;   // the pictures in the caches (their first image token's position, their pixels' hash)
    struct Unpin {   // (a pinned buffer unpinned before its vector frees it)
        void* p;
        bool on;
        ~Unpin() {
            if (on) hy4::Model::unpin(p);
        }
    };
    std::vector<float> lg((size_t) m.vocab());
    Unpin lg_unpin{lg.data(), hy4::Model::pin(lg.data(), lg.size() * sizeof(float))};
    unsigned long long counter = 0;
    std::string line, err;
    auto done_cancel = [&](size_t n_ids) {
        std::printf("DONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", n_ids);
        std::fflush(stdout);
    };
    while (in.next(line)) {
        if (line == "QUIT") break;
        if (line.rfind("GEN ", 0) != 0 && line.rfind("GENI ", 0) != 0) {
            std::printf("ERR unsupported command '%s' (GEN, GENI, STOP, WRAP and QUIT)\n",
                        line.substr(0, line.find(' ')).c_str());
            std::fflush(stdout);
            continue;
        }
        in.stop.store(false);
        in.wrap.store(false);
        Req r;
        if (!parse_gen(line, r) || (line.rfind("GENI ", 0) == 0 && r.file.empty())) {
            std::printf("ERR expected: GEN <max_new> [keys] <id,id,...> (GENI: the pictures' file before the ids)\n");
            done_cancel(r.ids.size());
            continue;
        }
        std::vector<Pic> pics;
        if (!r.file.empty() && !read_pics(r.file, pics, err)) {
            std::printf("ERR %s\n", err.c_str());
            done_cancel(r.ids.size());
            continue;
        }
        if ((long long) r.ids.size() + std::max(0LL, r.max_new) > m.max_ctx()) {
            std::printf("ERR the prompt (%zu tokens) and max_new %lld are more than the context %lld\n", r.ids.size(),
                        r.max_new, (long long) m.max_ctx());
            done_cancel(r.ids.size());
            continue;
        }
        const double t0 = now() * 1e3;
        // the state's shared start (the prompt's last token is always read again: its logits are the first token's)
        size_t from = 0;
        while (from < state.size() && from + 1 < r.ids.size() && state[from] == r.ids[from]) ++from;
        // (a picture before `from` in either the state or the request must be in both, at the same place)
        for (const Held& q : held)
            if (q.start < (long long) from &&
                std::none_of(pics.begin(), pics.end(), [&](const Pic& x) { return x.start == q.start && x.hash == q.hash; }))
                from = (size_t) q.start;
        for (const Pic& x : pics)
            if (x.start < (long long) from &&
                std::none_of(held.begin(), held.end(), [&](const Held& q) { return x.start == q.start && x.hash == q.hash; }))
                from = (size_t) x.start;
        if (from > 0 && !m.resumable((int64_t) from)) from = 0;   // (the windows before it gone: read it all again)
        state.resize(from);
        held.clear();
        {
            std::vector<hy4::Model::Image> rows;
            const int itok = m.vision() ? m.vision()->image_token : -1;
            const bool any = !pics.empty() || (itok >= 0 && std::find(r.ids.begin(), r.ids.end(), itok) != r.ids.end());
            if (any && !picture_rows(m, pics, r.ids, from, rows, err)) {
                std::printf("ERR %s\n", err.c_str());
                done_cancel(r.ids.size());
                state.clear();   // (the caches' pictures are no longer known)
                continue;
            }
            for (const Pic& x : pics) held.push_back({x.start, x.hash});
            m.set_images(std::move(rows));
        }
        bool ok = true, cancel = false;
        const int C = std::max(1, m.chunk());
        // (the prompt's last token read alone, on the token path, after the rest: a repeat of this request finds every
        //  token before it in the caches and reads that one alone too - the same row, the same first logits, so the
        //  same answer as the first time.  A chunk's last row is the prompt path's - other kernels and, with tensor
        //  parallel, other sums than a token's - so the two differ in the last bits and a greedy answer can part.
        //  Every layout - one GPU, the layer split, tensor parallel; HY4_LAST_ALONE=0: none)
        static const bool last_alone = [] {
            const char* e = std::getenv("HY4_LAST_ALONE");
            return !(e && *e && std::atoi(e) == 0);
        }();
        const bool alone = r.ids.size() - from > 1 && last_alone;
        const size_t body = alone ? r.ids.size() - 1 : r.ids.size();
        for (size_t a = from; a < r.ids.size();) {
            const int T = (int) std::min<size_t>((size_t) C, (a < body ? body : r.ids.size()) - a);
            const bool last = a + (size_t) T == r.ids.size();
            if (!m.forward(r.ids.data() + a, T, (int64_t) a, last ? 1 : 0, last ? lg.data() : nullptr, err)) {
                ok = false;
                break;
            }
            state.insert(state.end(), r.ids.begin() + (long) a, r.ids.begin() + (long) a + T);
            a += (size_t) T;
            const double ms = now() * 1e3 - t0;
            std::printf("PP %zu %zu %.1f %.1f %zu\n", a, r.ids.size(), ms, ms > 0 ? 1000.0 * (double) (a - from) / ms : 0.0,
                        from);
            std::fflush(stdout);
            if (in.stop.load() && !last) {
                cancel = true;
                break;
            }
        }
        const double t1 = now() * 1e3;
        if (!ok) {
            std::printf("ERR %s\n", err.c_str());
            done_cancel(r.ids.size());
            state.clear();
            held.clear();
            m.set_images({});
            continue;
        }
        if (cancel) {
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0\n", r.ids.size(), t1 - t0, from);
            std::fflush(stdout);
            m.set_images({});
            continue;
        }
        ++counter;
        std::mt19937_64 rng(r.seed != 0 ? r.seed : 0x9E3779B97F4A7C15ull * counter);
        long long produced = 0;
        const char* finish = "length";
        bool in_think = r.think_budget > 0 && r.think_end >= 0;
        int tok = mm3::sample::draw(row_dist(lg.data(), m.vocab(), r, state, m.row_top(0)), rng);
        long long d_off = 0, d_acc = 0;
        // the decode's own counts (the model's totals less what they were once the prompt was read): the routed
        // experts run, those of them on the CPU lane, the host's n-gram gathers
        const long long ex0 = (long long) m.stats().experts, cpu0 = (long long) m.stats().cpu_experts;
        const double eg0 = m.stats().engram_ms;
        Extra extra;      // (when a step tries a draft more than the usual ones)
        extra.n_more = n_more;
        extra.need = need;
        auto stat_line = [&]() {
            const double ms = now() * 1e3 - t1;
            const long long d_ex = (long long) m.stats().experts - ex0;
            const long long d_cpu = (long long) m.stats().cpu_experts - cpu0;
            const double d_eg = m.stats().engram_ms - eg0;
            std::printf("STAT tok_s=%.2f ms_tok=%.2f vram_hit=%.4f cpu=%.3f", ms > 0 ? 1000.0 * (double) produced / ms : 0.0,
                        produced > 0 ? ms / (double) produced : 0.0,
                        d_ex > 0 ? 1.0 - (double) d_cpu / (double) d_ex : 1.0,
                        produced > 0 ? (double) d_cpu / (double) produced : 0.0);
            if (d_off > 0) std::printf(" drafts=%.3f draft_acc=%lld draft_off=%lld", (double) d_acc / (double) d_off, d_acc,
                                       d_off);
            std::printf(" engram_ms=%.3f\n", produced > 0 ? d_eg / (double) produced : 0.0);
        };
        const bool spec = m.has_draft() && n_draft > 0 && (r.spec < 0 || r.spec != 0);
        const float floor = r.conf >= 0.0f ? r.conf : floor0;
        std::vector<int> drafts;
        Weights wts;   // (plain sampling's rows)
        std::vector<float> confs;
        std::vector<float> lgw(spec ? (size_t) (n_draft + n_more + 1) * m.vocab() : 0);
        Unpin lgw_unpin{lgw.data(), hy4::Model::pin(lgw.data(), lgw.size() * sizeof(float))};
        std::string fail;
        // one token out: false when the answer ends with it (an end token, max_new, the context)
        auto emit = [&](int t) -> bool {
            std::printf("T %d\n", t);
            ++produced;
            if (in_think && t == r.think_end) in_think = false;
            if (produced % 8 == 0) stat_line();
            std::fflush(stdout);
            if (std::find(eos.begin(), eos.end(), t) != eos.end()) {
                finish = "stop";
                return false;
            }
            return produced < r.max_new && (int64_t) state.size() + 2 < m.max_ctx();
        };
        for (;;) {
            if (in.stop.load()) {
                finish = "cancel";
                break;
            }
            if (in_think && (produced >= r.think_budget || in.wrap.load()) && tok != r.think_end) tok = r.think_end;
            if (!emit(tok)) break;
            const int64_t pos = (int64_t) state.size();
            // the drafts that can still be answer tokens (max_new, the context, the thinking budget)
            long long room = std::min<long long>(r.max_new - produced, m.max_ctx() - pos - 2);
            if (in_think) room = std::min<long long>(room, r.think_budget - produced);
            if (in_think && in.wrap.load()) room = 0;
            int k = 0;
            if (spec && room > 0) {
                if (!m.draft(tok, pos, (int) std::min<long long>(room, n_draft + extra.now()), floor,
                             drafts, confs, err, n_draft, sure)) {
                    fail = err;
                    break;
                }
                while (k < (int) drafts.size() && confs[(size_t) k] >= floor) ++k;
            }
            if (k == 0) {
                extra.seen(0, 0, n_draft);
                if (!m.forward(&tok, 1, pos, 1, lg.data(), err)) {
                    fail = err;
                    break;
                }
                state.push_back(tok);
                if (plain_sampling(r)) {
                    wts.of(lg.data(), m.vocab(), r.temperature, m.row_top(0));
                    tok = wts.draw(std::uniform_real_distribution<double>(0.0, 1.0)(rng));
                } else {
                    tok = mm3::sample::draw(row_dist(lg.data(), m.vocab(), r, state, m.row_top(0)), rng);
                }
                continue;
            }
            std::vector<int> win(1, tok);
            win.insert(win.end(), drafts.begin(), drafts.begin() + k);
            if (!m.forward(win.data(), k + 1, pos, k + 1, lgw.data(), err)) {
                fail = err;
                break;
            }
            state.push_back(tok);
            d_off += k;
            const long long acc0 = d_acc;
            bool more = true;
            int next = -1;
            const bool plain = plain_sampling(r);
            for (int i = 0; i <= k && more && plain; ++i) {   // (plain sampling: the rows' weights, no copies)
                wts.of(lgw.data() + (size_t) i * m.vocab(), m.vocab(), r.temperature, m.row_top(i));
                std::uniform_real_distribution<double> u01(0.0, 1.0);
                if (i == k) {
                    next = wts.draw(u01(rng));
                    break;
                }
                const int x = drafts[(size_t) i];
                if (!(u01(rng) < wts.p(x))) {
                    next = wts.draw(u01(rng), x);
                    break;
                }
                ++d_acc;
                if (in.stop.load()) {
                    finish = "cancel";
                    more = false;
                    break;
                }
                if (in_think && x == r.think_end) in_think = false;
                more = emit(x);
                state.push_back(x);
            }
            for (int i = 0; i <= k && more && !plain; ++i) {
                const auto d = row_dist(lgw.data() + (size_t) i * m.vocab(), m.vocab(), r, state, m.row_top(i));
                if (i == k) {   // every draft kept: the model's token after them
                    next = mm3::sample::draw(d, rng);
                    break;
                }
                const int x = drafts[(size_t) i];
                double px = 0;
                for (const auto& [t, q] : d)
                    if (t == x) px = q;
                const bool keep = r.temperature > 0.0f ? std::uniform_real_distribution<double>(0.0, 1.0)(rng) < px
                                                       : d[0].first == x;
                if (!keep) {
                    next = r.temperature > 0.0f ? mm3::sample::draw(d, rng, x) : d[0].first;
                    break;
                }
                ++d_acc;
                if (in.stop.load()) {
                    finish = "cancel";
                    more = false;
                    break;
                }
                if (in_think && x == r.think_end) in_think = false;
                more = emit(x);
                state.push_back(x);
            }
            extra.seen(k, (int) (d_acc - acc0), n_draft);
            if (!more) break;
            tok = next;
        }
        m.set_images({});
        if (!fail.empty()) {
            std::printf("ERR %s\n", fail.c_str());
            std::fprintf(stderr, "hy4: %s\n", fail.c_str());
            finish = "cancel";
            state.clear();
            held.clear();
        }
        if (spec && d_off > 0) std::fprintf(stderr, "maya hy4 serve: %s kept %lld of %lld drafts\n", m.has_dfly2() ? "DFly2" : "MTP", d_acc, d_off);
        stat_line();
        const double t2 = now() * 1e3;
        std::printf("DONE %lld %zu %.1f %.1f %s %lld %lld %zu 0 0\n", produced, r.ids.size(), t1 - t0, t2 - t1, finish,
                    d_acc, d_off, from);
        std::fflush(stdout);
        {
            // (HY4_HOT: the routes of the rows a plain decode reads - the prompt's and the answer's tokens but its
            //  last - drafting on or off the same; then the experts it moves, while the answer is read)
            m.answered(std::max<int64_t>(0, (int64_t) r.ids.size() + produced - 1));
            std::string perr;
            if (!m.save_profile(perr)) std::fprintf(stderr, "hy4: the expert profile: %s\n", perr.c_str());
            m.idle();
        }
        std::fprintf(stderr,
                     "maya hy4 serve: prompt %zu tokens = %zu reused + %zu read in %.0f ms (%.1f tok/s), %lld generated "
                     "in %.0f ms (%.2f tok/s)\n",
                     r.ids.size(), from, r.ids.size() - from, t1 - t0,
                     t1 > t0 ? 1000.0 * (double) (r.ids.size() - from) / (t1 - t0) : 0.0, produced, t2 - t1,
                     t2 > t1 ? 1000.0 * (double) produced / (t2 - t1) : 0.0);
    }
    return 0;
}

// --bench-lane (no GPU): the CPU lane's routed experts of one layer, n experts a call for one token, at several thread
// counts - each call's experts picked at random, every expert read once first (the page cache)
int bench_lane(const std::string& pack_dir, int il, int n_exp_call, int n_tok) {
    hy4::Pack pk;
    std::string err;
    if (!pk.load(pack_dir, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    const std::string p = "blk." + std::to_string(il) + ".";
    const hy4::Tensor *tg = pk.get(p + "ffn_gate_exps.weight"), *tu = pk.get(p + "ffn_up_exps.weight"),
                      *td = pk.get(p + "ffn_down_exps.weight");
    if (!tg || !tu || !td) {
        std::fprintf(stderr, "layer %d has no routed experts\n", il);
        return 1;
    }
    auto tid = [](const std::string& t) {   // (ggml's type numbers: every block type the lane's rows take)
        static const std::pair<const char*, int> names[] = {
            {"Q8_0", 8},    {"Q2_K", 10},   {"Q3_K", 11},  {"Q4_K", 12},   {"Q5_K", 13},  {"Q6_K", 14},  {"IQ2_XXS", 16},
            {"IQ2_XS", 17}, {"IQ3_XXS", 18}, {"IQ1_S", 19}, {"IQ4_NL", 20}, {"IQ3_S", 21}, {"IQ2_S", 22}, {"IQ4_XS", 23},
            {"IQ1_M", 29}};
        for (const auto& [n, i] : names)
            if (t == n) return i;
        return -1;
    };
    hy4::CpuExperts ex;
    if (!ex.init(tid(tg->type), tid(td->type), pk.g.n_lat, pk.g.n_ff_exp, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    ex.gate = pk.data(*tg);
    ex.up = pk.data(*tu);
    ex.down = pk.data(*td);
    ex.limit = 10.0f;
    const int E = pk.g.n_lat, NE = pk.g.n_expert;
    std::vector<float> x((size_t) E * n_tok), out((size_t) E * n_tok);
    std::mt19937 rng(1);
    for (float& v : x) v = std::normal_distribution<float>(0.0f, 1.0f)(rng);
    volatile uint8_t sink = 0;   // (every expert's bytes touched once: the page cache)
    for (int e = 0; e < NE; ++e) {
        for (size_t b = 0; b < ex.g_stride; b += 4096) sink ^= ex.gate[(size_t) e * ex.g_stride + b] ^ ex.up[(size_t) e * ex.g_stride + b];
        for (size_t b = 0; b < ex.d_stride; b += 4096) sink ^= ex.down[(size_t) e * ex.d_stride + b];
    }
    {
        hy4::Lane probe(0);
        int hist[4] = {0, 0, 0, 0};
        for (int e = 0; e < NE; ++e) {
            const int q = hy4::Lane::node_of_page(ex.gate + (size_t) e * ex.g_stride);
            ++hist[q < 0 ? 3 : std::min(q, 2)];
        }
        std::printf("lane nodes %d; experts' gate pages: node0 %d, node1 %d, other %d, unknown %d\n", probe.nodes(),
                    hist[0], hist[1], hist[2], hist[3]);
    }
    if (const char* q4 = std::getenv("HY4_BENCH_Q4X"); q4 && std::atoi(q4) != 0) {   // (every expert's 4-bit copy)
        if (!ex.q4x_can()) {
            std::fprintf(stderr, "HY4_BENCH_Q4X: this layer's experts have no 4-bit form here\n");
            return 1;
        }
        ex.q4x = std::make_shared<hy4::Q4xSet>(NE);
        const auto t0 = std::chrono::steady_clock::now();
        for (int e = 0; e < NE; ++e) ex.q4x_make(e);
        std::printf("4-bit copies of the %d experts: %.2f GB, %.2f s\n", NE,
                    (double) NE * (2.0 * ex.q4x_gu + ex.q4x_d) / 1e9,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    const double mb = (2.0 * ex.g_stride + ex.d_stride) / 1e6;
    std::printf("layer %d: %s / %s experts, %.2f MB each; %d a token, %d tokens a call\n", il, tg->type.c_str(),
                td->type.c_str(), mb, n_exp_call, n_tok);
    // the thread counts: this host's (1, 2, 4 ... its physical cores, its logical CPUs).  Two rates each: the experts
    // picked at random (the weights from memory), and one small set every call (the weights in the CPU's caches) -
    // the gap is what the memory costs.  The sum printed is the first call's output bits (two builds compared by it)
    const int logical = std::max(1u, std::thread::hardware_concurrency());
    const int gap_us = std::getenv("HY4_BENCH_GAP_US") ? std::atoi(std::getenv("HY4_BENCH_GAP_US")) : 0;   // (a pause before each call)
    const int phys = (int) hy4::Lane(-1).threads() + 1;
    std::vector<int> counts;
    for (int c = 1; c < phys; c *= 2) counts.push_back(c);
    for (int c : {phys - 2, phys, phys + phys / 2, logical}) counts.push_back(c);
    if (const char* hc = std::getenv("HY4_BENCH_THREADS")) {   // (these thread counts only: "1,6")
        counts.clear();
        std::stringstream ss(hc);
        for (std::string v; std::getline(ss, v, ',');) counts.push_back(std::atoi(v.c_str()));
    }
    std::sort(counts.begin(), counts.end());
    counts.erase(std::unique(counts.begin(), counts.end()), counts.end());
    // (the fastest call besides the mean: a busy host's other work only ever adds time, so two builds compare by it)
    for (int c : counts) {
        if (c < 1 || c > logical) continue;
        hy4::Lane lane(c - 1);
        std::vector<hy4::CpuEntry> ent;
        const int reps = n_tok > 1 ? 4 : 60;
        double ms2[2] = {0, 0}, best = 1e30;
        uint64_t bits = 1469598103934665603ull;
        for (int same = 0; same < 2; ++same) {
            std::mt19937 r2(7);
            double tsum = 0;
            for (int r = 0; r < reps + 3; ++r) {
                if (r == 0 || !same) {
                    ent.clear();
                    std::vector<int> pick(NE);
                    for (int e = 0; e < NE; ++e) pick[(size_t) e] = e;
                    for (int t = 0; t < n_tok; ++t) {
                        std::shuffle(pick.begin(), pick.end(), r2);
                        for (int j = 0; j < n_exp_call; ++j) ent.push_back({t, pick[(size_t) j], 0.25f});
                    }
                }
                if (gap_us > 0) std::this_thread::sleep_for(std::chrono::microseconds(gap_us));
                const double t0 = now();
                hy4::cpu_experts(lane, ex, ent, x.data(), n_tok, out.data());
                if (r >= 3) {
                    tsum += now() - t0;
                    if (!same) best = std::min(best, now() - t0);
                }
                if (r == 0 && !same)
                    for (float v : out) {
                        uint32_t u;
                        std::memcpy(&u, &v, 4);
                        bits = (bits ^ u) * 1099511628211ull;
                    }
            }
            ms2[same] = 1e3 * tsum / reps;
        }
        std::printf("  %2d threads: %.3f ms a call (fastest %.3f), %.3f ms an expert-token, %.1f tokens/s; the same experts "
                    "every call %.3f ms; sum %016llx\n", c, ms2[0], 1e3 * best, ms2[0] / ((double) n_exp_call * n_tok),
                    1000.0 * n_tok / ms2[0], ms2[1], (unsigned long long) bits);
    }
    return 0;
}

}  // namespace

// --encode-image: the vision tower alone - the pack's mmproj tensors on one GPU (--gpus' first, else 0), a picture's raw
// RGB bytes (width x height x 3) -> its rows (F32 [tokens][3072]) into --dump-image, twice (the second time warm)
int encode_alone(const hy4::Options& o, const std::string& rgb_path, const std::string& size, const std::string& out) {
    int w = 0, h = 0;
    if (std::sscanf(size.c_str(), "%dx%d", &w, &h) != 2 || w <= 0 || h <= 0 || out.empty()) {
        std::fprintf(stderr, "--encode-image needs --size WxH and --dump-image OUT\n");
        return 2;
    }
    std::ifstream f(rgb_path, std::ios::binary);
    std::vector<uint8_t> px((size_t) w * h * 3);
    if (!f.read((char*) px.data(), (std::streamsize) px.size())) {
        std::fprintf(stderr, "%s: not %d x %d x 3 bytes\n", rgb_path.c_str(), w, h);
        return 1;
    }
    hy4::Pack pk;
    std::string err;
    if (!pk.load(o.pack, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    if (!hy4::Vision::in_pack(pk)) {
        std::fprintf(stderr, "maya-hy4: the pack has no vision tower (pack the model again with its mmproj file)\n");
        return 1;
    }
    const int dev = o.gpus.empty() ? 0 : o.gpus[0];
    hy4::Vision v;
    if (!v.load(pk, dev, err)) {
        std::fprintf(stderr, "maya-hy4: %s\n", err.c_str());
        return 1;
    }
    std::vector<float> rows;
    for (int pass = 0; pass < 2; ++pass) {
        const double t0 = now();
        if (!v.encode(px.data(), w, h, rows, err)) {
            std::fprintf(stderr, "maya-hy4: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "vision: %d x %d pixels -> %zu rows in %.0f ms (%zu MB of CUDA%d during it; the largest "
                     "picture's room %zu MB)\n", w, h, rows.size() / (size_t) v.geo().out_dim, 1e3 * (now() - t0),
                     v.last_bytes() >> 20, dev, v.room_bytes() >> 20);
    }
    std::ofstream of(out, std::ios::binary);
    if (!of.write((const char*) rows.data(), (std::streamsize) (rows.size() * 4))) {
        std::fprintf(stderr, "%s: not written\n", out.c_str());
        return 1;
    }
    return 0;
}

// ---- --drafting auto / on: the drafting this PC runs fastest, measured once and remembered -------------------------
// Each choice - no drafting (auto only), the MTP block, the DFly2 drafter (one GPU; on more only by hand, --dfly2) -
// is loaded on its own with its own plan (only the drafter it runs has room: without one the routed experts get all
// of it) and times a greedy decode: HY4_DRAFT_ROUNDS (3) prompts of HY4_DRAFT_PROMPT (512) tokens from the measured
// text (HY4_MEASURE_IDS - the server's: this install's docs in the model's tokens), each answered for HY4_DRAFT_TOKENS
// (192) / rounds tokens as a request is answered (its drafts as served).  The first load reads the text untimed once
// before (the Engram rows it looks up come from the disk the first time - a cost no later load pays); the others warm
// up on a short answer.  Each choice in a process of its own (this program again, --draft-trial): a load leaves
// per-stream scratch behind in the kernels' caches, so loads in one process would plan on VRAM the next one does not
// have - the start's own load is then the one a later start makes from the remembered choice.  The fastest is kept -
// within HY4_DRAFT_TIE (2%) of it the one with the least VRAM of its own (none, the DFly2 drafter, the MTP block).
// Remembered per PC, GPUs, model and start options (hy4::draft_key) in hy4::draft_file (the server's data folder);
// HY4_DRAFT_MEASURE=1 (maya --speed-calibration) measures again.  No text or no file: not measured - the pack's
// drafter, the MTP block first.  --drafting on: the faster drafter of the same measurement (none only where neither
// loads); --mtp / --dfly2 / --drafting off by hand: that one, nothing measured.
struct DraftArgs {
    int n_draft;
    float floor;
    int n_more;
    float sure;
    int need;
};

void draft_params(const hy4::Model& m, const DraftArgs& a, int& n_draft, float& floor, int& n_more) {
    n_draft = a.n_draft < 0 ? m.draft_default() : a.n_draft;
    if (m.has_dfly2()) n_draft = std::min(n_draft, m.draft_max());
    floor = a.floor < 0 ? (m.has_dfly2() ? 0.8f : 0.55f) : a.floor;
    n_more = m.has_dfly2() ? 0 : std::min(a.n_more, std::max(0, hy4::k::kMaxWin - 1 - n_draft));
}

// one answer: the prompt read from position 0, then `gen` greedy tokens - drafted as a request is where a drafter is
// loaded; the decode's tokens and seconds
bool trial_answer(hy4::Model& m, const std::vector<int>& p, int gen, const DraftArgs& a, std::vector<float>& lg,
                  int& made, double& secs, std::string& err) {
    if (!m.forward(p.data(), (int) p.size(), 0, 1, lg.data(), err)) return false;
    int n_draft = 0, n_more = 0;
    float floor = 0.0f;
    draft_params(m, a, n_draft, floor, n_more);
    auto top_of = [&](const float* row, int r) {
        const int t = m.row_top(r);
        return t >= 0 ? t : argmax(row, m.vocab());
    };
    const double t0 = now();
    int64_t pos = (int64_t) p.size();
    std::vector<int> out;
    if (m.has_draft() && n_draft > 0) {
        std::vector<int> drafts;
        std::vector<float> confs;
        Extra extra;
        extra.n_more = n_more;
        extra.need = a.need;
        int tok = top_of(lg.data(), 0);
        while ((int) out.size() < gen) {
            out.push_back(tok);
            if (tok == m.eos() || (int) out.size() >= gen) break;
            if (!m.draft(tok, pos, std::min(n_draft + extra.now(), gen - (int) out.size()), floor, drafts, confs, err,
                         n_draft, a.sure))
                return false;
            int k = 0;
            while (k < (int) drafts.size() && confs[(size_t) k] >= floor) ++k;
            std::vector<int> win(1, tok);
            win.insert(win.end(), drafts.begin(), drafts.begin() + k);
            if (!m.forward(win.data(), k + 1, pos, k + 1, lg.data(), err)) return false;
            pos += 1;
            int i = 0;
            for (; i < k; ++i) {
                if (top_of(lg.data() + (size_t) i * m.vocab(), i) != drafts[(size_t) i]) break;
                out.push_back(drafts[(size_t) i]);
                ++pos;
                if (drafts[(size_t) i] == m.eos() || (int) out.size() >= gen) break;
            }
            extra.seen(k, i, n_draft);
            if (out.back() == m.eos() || (int) out.size() >= gen) break;
            tok = top_of(lg.data() + (size_t) i * m.vocab(), i);
        }
    } else {
        for (int i = 0; i < gen; ++i) {
            const int next = top_of(lg.data(), 0);
            out.push_back(next);
            if (next == m.eos() || i + 1 >= gen) break;
            if (!m.forward(&next, 1, pos, 1, lg.data(), err)) return false;
            ++pos;
        }
    }
    secs = now() - t0;
    made = (int) out.size();
    return true;
}

std::string line_get(const std::string& file, const std::string& key) {
    std::ifstream f(file);
    std::string out;
    for (std::string line; f && std::getline(f, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const size_t tab = line.rfind('\t');
        if (tab != std::string::npos && line.compare(0, tab, key) == 0 && tab == key.size()) out = line.substr(tab + 1);
    }
    return out;
}

bool line_put(const std::string& file, const std::string& key, const std::string& vals) {
    std::string keep;
    {
        std::ifstream f(file);
        for (std::string line; f && std::getline(f, line);) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            const size_t tab = line.rfind('\t');
            if (line.empty() || (tab == key.size() && line.compare(0, tab, key) == 0)) continue;
            keep += line + "\n";
        }
    }
    keep += key + "\t" + vals + "\n";
    const std::string tmp = file + ".new";
    {
        std::ofstream f(tmp, std::ios::binary);
        if (!(f << keep)) return false;
    }
    std::remove(file.c_str());   // (Windows: rename does not replace)
    return std::rename(tmp.c_str(), file.c_str()) == 0;
}

// the drafters a pack names in its hy4.txt (1 the MTP block, 2 the DFly2 drafter) - before any load
int pack_has_drafters(const std::string& pack) {
    std::ifstream f(pack + "/hy4.txt");
    int bits = 0;
    for (std::string line; f && std::getline(f, line);) {
        if (line.rfind("n_mtp ", 0) == 0 && std::atoi(line.c_str() + 6) > 0) bits |= 1;
        if (line.rfind("dfly_layers ", 0) == 0 && std::atoi(line.c_str() + 12) > 0) bits |= 2;
    }
    return bits;
}

// the measuring text's prompts: HY4_DRAFT_ROUNDS (3) of HY4_DRAFT_PROMPT (512) tokens from its thirds' middles, each
// answered for HY4_DRAFT_TOKENS (192) / rounds tokens.  false: no text (HY4_MEASURE_IDS) or the context is too short
struct TrialText {
    std::vector<std::vector<int>> prompts;
    int gen = 0;
    size_t n = 0;
};
bool trial_text(const hy4::Options& o, TrialText& tx) {
    auto envi = [](const char* k, int d) {
        const char* v = std::getenv(k);
        return v && *v ? std::max(1, std::atoi(v)) : d;
    };
    std::vector<int> text;
    if (const char* path = std::getenv("HY4_MEASURE_IDS")) {   // (its '#' lines: a comment)
        std::ifstream f(path);
        for (std::string line; f && std::getline(f, line);)
            if (!line.empty() && line[0] != '#')
                for (int v : parse_ids(line))
                    if (v >= 0) text.push_back(v);
    }
    tx.n = text.size();
    // (HY4_DRAFT_IDS - the server's: requests in the model's chat template, a line each - answered in place of the
    //  text's slices: the drafts hold as in real use, not as in a document's continuation)
    if (const char* dp = std::getenv("HY4_DRAFT_IDS")) {
        std::ifstream f(dp);
        for (std::string line; f && std::getline(f, line);)
            if (!line.empty() && line[0] != '#') {
                std::vector<int> p = parse_ids(line);
                if (p.size() >= 8) tx.prompts.push_back(std::move(p));
            }
        if (!tx.prompts.empty()) {   // (192 tokens in all, as the text's: the same choice as 384 on every PC tried)
            tx.gen = std::max(8, envi("HY4_DRAFT_TOKENS", 192) / (int) tx.prompts.size());
            for (const auto& p : tx.prompts)
                if (o.max_ctx < (int64_t) p.size() + tx.gen + 64) tx.prompts.clear();
            if (!tx.prompts.empty()) return true;
        }
    }
    const int rounds = envi("HY4_DRAFT_ROUNDS", 3), P = envi("HY4_DRAFT_PROMPT", 512);
    tx.gen = std::max(8, envi("HY4_DRAFT_TOKENS", 192) / rounds);
    if ((int64_t) text.size() < 4LL * P || o.max_ctx < P + tx.gen + 64) return false;
    for (int r = 0; r < rounds; ++r) {
        const size_t at = (size_t) ((double) text.size() * (2 * r + 1) / (2.0 * rounds)) - (size_t) P / 2;
        tx.prompts.emplace_back(text.begin() + (long) at, text.begin() + (long) at + P);
    }
    return true;
}

// --draft-trial none|mtp|dfly2[:full|:short] (pick_drafting's child): that choice loaded as a start with it plans,
// the text answered (full: read once untimed first - the first child; short: a 16-token warm-up), and one line
// "TRIAL <tok/s> <tokens> <experts in VRAM> <experts> <chunk> <load s>" on stdout - 0 tok/s: it does not load beside
// this start (its room would change it).  A loaded child ends at once once it has said it (trial_exit): the system
// frees its memory as with any process; nothing of it is written anywhere
[[noreturn]] void trial_exit() {
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit(0);
}
int run_draft_trial(hy4::Options o, const std::string& spec, const DraftArgs& a) {
    static const char* said[3] = {"no drafting", "the MTP block", "the DFly2 drafter"};
    const std::string what = spec.substr(0, spec.find(':'));
    const int c = what == "mtp" ? 1 : what == "dfly2" ? 2 : 0;
    const bool full = spec.find(":full") != std::string::npos;
    o.mtp = c == 1;
    o.dfly2 = c == 2;
    o.trial = true;
    TrialText tx;
    if (!trial_text(o, tx)) {
        std::printf("TRIAL 0\n");
        return 0;
    }
    const double tl = now();
    std::string err;
    hy4::Model t;
    if (!t.load(o, err)) {
        std::fprintf(stderr, "maya-hy4: drafting trial: %s does not load beside this start (%s)\n", said[c], err.c_str());
        std::printf("TRIAL 0\n");
        return 0;
    }
    if (c > 0 && !t.has_draft()) {   // (its room would change the start: the plan leaves it out, said)
        for (const auto& l : t.info_lines())
            if (l.find("is not loaded") != std::string::npos)
                std::fprintf(stderr, "maya-hy4: drafting trial: %s\n", l.c_str() + 5);
        std::printf("TRIAL 0\n");
        trial_exit();
    }
    const double ls = now() - tl;
    std::vector<float> lg((size_t) hy4::k::kMaxWin * t.vocab());
    const bool pinned = hy4::Model::pin(lg.data(), lg.size() * sizeof(float));
    int made = 0, n = 0;
    double secs = 0.0, s = 0.0;
    bool ok = true;
    if (full)   // (the text's Engram rows read once, untimed: the disk's first reads are no choice's cost)
        for (size_t r = 0; ok && r < tx.prompts.size(); ++r) ok = trial_answer(t, tx.prompts[r], tx.gen, a, lg, made, s, err);
    else
        ok = trial_answer(t, tx.prompts[0], 16, a, lg, made, s, err);
    for (size_t r = 0; ok && r < tx.prompts.size(); ++r) {
        ok = trial_answer(t, tx.prompts[r], tx.gen, a, lg, made, s, err);
        n += made;
        secs += s;
    }
    if (pinned) hy4::Model::unpin(lg.data());
    if (!ok) {
        std::fprintf(stderr, "maya-hy4: drafting trial: %s - %s\n", said[c], err.c_str());
        std::printf("TRIAL 0\n");
        trial_exit();
    }
    const double tps = secs > 0 ? n / secs : 0.0;
    const auto& st = t.stats();
    std::fprintf(stderr, "maya-hy4: drafting trial: %s %.2f tokens/s (%d tokens; load %.0f s; %d of %d routed experts in "
                 "VRAM, prompt chunk %d)\n", said[c], tps, n, ls, st.experts_vram, st.experts_all, t.chunk());
    std::printf("TRIAL %.3f %d %d %d %d %.1f\n", tps, n, st.experts_vram, st.experts_all, t.chunk(), ls);
    trial_exit();
}

// this program's own path (its trials run it again)
std::string self_exe(const std::string& argv0) {
#if defined(_WIN32)
    char* p = nullptr;
    if (_get_pgmptr(&p) == 0 && p && *p) return p;
#else
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
    if (n > 0) return std::string(buf, (size_t) n);
#endif
    return argv0;
}
std::string quote(const std::string& s) {
    std::string q = "\"";
    for (char ch : s) {
        if (ch == '"') q += '\\';
        q += ch;
    }
    return q + "\"";
}
FILE* open_pipe(const std::string& cmd) {
#if defined(_WIN32)
    return _popen(("\"" + cmd + "\"").c_str(), "r");   // (cmd /c drops the outer quotes)
#else
    return popen(cmd.c_str(), "r");
#endif
}
void close_pipe(FILE* p) {
#if defined(_WIN32)
    _pclose(p);
#else
    pclose(p);
#endif
}

// how: "auto" or "on".  o.mtp / o.dfly2 set to the choice; child_args: this start's arguments without its drafting
// and its mode (a trial's)
bool pick_drafting(hy4::Options& o, const std::string& how, const std::string& argv0,
                   const std::vector<std::string>& child_args, std::string& err) {
    (void) err;
    static const char* id[3] = {"none", "mtp", "dfly2"};
    static const char* said[3] = {"no drafting", "the MTP block", "the DFly2 drafter"};
    const double tie = [] {
        const char* v = std::getenv("HY4_DRAFT_TIE");
        return v && *v ? std::max(0.0, std::atof(v)) : 0.02;
    }();
    const bool on = how == "on";
    auto choose = [&](const double* tps) {   // (the order of the least VRAM of its own: none, DFly2, the block)
        static const int order[3] = {0, 2, 1};
        double best = 0.0;
        for (int c = on ? 1 : 0; c < 3; ++c) best = std::max(best, tps[c]);
        if (best <= 0.0) return 0;
        for (int c : order)
            if (!(on && c == 0) && tps[c] > 0.0 && tps[c] >= best * (1.0 - tie)) return c;
        return 0;
    };
    auto apply = [&](int c) {
        o.mtp = c == 1;
        o.dfly2 = c == 2;
    };
    // (the choices: no drafting - not with --drafting on -, the MTP block, the DFly2 drafter on one GPU)
    const int bits0 = pack_has_drafters(o.pack), ngpu0 = hy4::gpus_of(o);
    bool cand[3] = {!on, (bits0 & 1) != 0, (bits0 & 2) != 0 && ngpu0 <= 1};
    const int n_cand = (int) cand[0] + (int) cand[1] + (int) cand[2];
    if (n_cand <= 1) {   // (nothing to choose between: the one there is)
        const int c = cand[1] ? 1 : cand[2] ? 2 : 0;
        apply(c);
        std::fprintf(stderr, "INFO drafting: %s - the only choice here (--drafting %s)\n", said[c], how.c_str());
        return true;
    }
    const std::string file = hy4::draft_file(o), key = hy4::draft_key(o);
    const char* again = std::getenv("HY4_DRAFT_MEASURE");
    if (!file.empty() && !(again && std::atoi(again) != 0)) {
        const std::string v = line_get(file, key);   // "<none> <mtp> <dfly2> tok/s <date>" (0: not loadable, -1: not
        double tps[3] = {0, 0, 0};                   //  measured)
        char date[64] = "";
        if (!v.empty() && std::sscanf(v.c_str(), "%lf %lf %lf %63s", &tps[0], &tps[1], &tps[2], date) == 4 &&
            !(cand[0] && tps[0] < 0) && !(cand[1] && tps[1] < 0) && !(cand[2] && tps[2] < 0)) {
            const int c = choose(tps);
            apply(c);
            std::fprintf(stderr, "INFO drafting: %s - the fastest on this PC (measured %s: no drafting %.2f, the MTP "
                         "block %.2f, the DFly2 drafter %.2f tokens/s - 0: not loadable here; %s)\n", said[c], date,
                         tps[0], tps[1], tps[2], file.c_str());
            return true;
        }
    }
    // the text (the server's measuring text): none, or no file to remember the choice in - the pack's drafter
    TrialText tx;
    if (file.empty() || !trial_text(o, tx)) {
        const int c = cand[1] ? 1 : cand[2] ? 2 : 0;
        apply(c);
        std::fprintf(stderr, "INFO drafting: %s - not measured (%s); --drafting off|mtp|dfly2 picks by hand\n", said[c],
                     file.empty() ? "no file to remember it in: HY4_DRAFT_FILE" : "no text to measure it on: HY4_MEASURE_IDS");
        return true;
    }
    const double t0 = now();
    std::string which;
    for (int c = 0; c < 3; ++c)
        if (cand[c]) which += std::string(which.empty() ? "" : ", ") + said[c];
    std::fprintf(stderr, "maya-hy4: measuring this PC once, before the first answer: drafting (%s: a load and %zu short "
                 "answers each) - on %zu tokens of text; remembered, the next starts skip it\n", which.c_str(),
                 tx.prompts.size(), tx.n);
    std::fflush(stderr);
    const std::string self = self_exe(argv0);
    double tps[3] = {-1, -1, -1};
    bool first = true;
    int done = 0, timed = 0;
    double timed_s = 0.0;   // (the timed choices' seconds: this PC's own pace for the "about ... more" line)
    for (int c : {1, 2, 0}) {   // (the drafters first: one its room leaves out stops after the plan)
        if (!cand[c]) continue;
        tps[c] = 0.0;
        if (c == 0 && tps[1] <= 0.0 && tps[2] <= 0.0) break;   // (no drafter loads here: nothing to time it against)
        if (done > 0 && timed > 0) {   // (what is left, at the pace of the choices timed so far here)
            std::fprintf(stderr, "maya-hy4: measuring this PC: %d of %d choices done in %.0f s - about %.0f s more\n",
                         done, n_cand, now() - t0, timed_s / timed * (n_cand - done));
            std::fflush(stderr);
        }
        const double tc = now();
        const bool full = first;
        std::string cmd = quote(self);
        for (const std::string& x : child_args) cmd += " " + quote(x);
        cmd += std::string(" --draft-trial ") + id[c] + (full ? ":full" : ":short");
        if (FILE* p = open_pipe(cmd)) {   // (its stderr is this one's: its lines in the start's log)
            char line[512];
            int toks = 0;
            while (std::fgets(line, sizeof line, p))
                if (std::strncmp(line, "TRIAL ", 6) == 0) {
                    double v = 0.0;
                    if (std::sscanf(line + 6, "%lf %d", &v, &toks) < 1) v = 0.0;
                    tps[c] = std::max(0.0, v);
                }
            close_pipe(p);
            if (tps[c] > 0.0) {   // (the text read once: by the first trial that timed)
                first = false;
                ++timed;
                // (the first one read the text once more, untimed: a later one is about that much shorter)
                timed_s += std::max(1.0, now() - tc - (full && toks > 0 ? toks / tps[c] : 0.0));
            }
            ++done;
        } else {
            std::fprintf(stderr, "maya-hy4: drafting trial: %s - this program could not be started again (%s)\n",
                         said[c], self.c_str());
        }
    }
    const int c = choose(tps);
    apply(c);
    char day[32];
    const std::time_t tt = std::time(nullptr);
    std::strftime(day, sizeof day, "%Y-%m-%d", std::localtime(&tt));
    char vals[160];
    std::snprintf(vals, sizeof vals, "%.3f %.3f %.3f %s", tps[0], tps[1], tps[2], day);
    const bool kept = line_put(file, key, vals);
    std::fprintf(stderr, "maya-hy4: measured this PC once in %.0f s, before the first answer: drafting - %s (no "
                 "drafting %.2f, the MTP block %.2f, the DFly2 drafter %.2f tokens/s)%s\n", now() - t0, said[c], tps[0],
                 tps[1], tps[2], kept ? "" : " - not remembered (the file does not open for writing)");
    std::fprintf(stderr, "INFO drafting: %s - the fastest on this PC (measured now; %s)\n", said[c], file.c_str());
    (void) id;
    return true;
}

int main(int argc, char** argv) {
    hy4::Options o;
    std::string ids_arg, kld;
    int gen = 0, chunks = 0;
    bool top = false, serving = false;
    int trace = 0;
    int bench_layer = -1, bench_n = 16, bench_t = 1;
    std::vector<int> eos;
    std::string drafting, draft_trial;   // (--drafting; --mtp / --dfly2 by name; --draft-trial: a trial's child)
    int n_draft = -1;   // (the loaded drafter's own: 3 with the MTP block, the DFly2 drafter's count)
    float floor = -1.0f;   // (the loaded drafter's own: 0.55 with the MTP block, 0.8 with the DFly2 drafter)
    int n_more = 1, need = 1;   // (--draft-more / --draft-streak: the MTP block's drafts)
    float sure = 0.8f;
    std::string images, enc_rgb, enc_size, enc_out, kld_save;
    int kv_stream = -1;  // --kv-streaming on: the indexed layers' rows beyond --kv-resident positions in host memory
                         // (auto, the default: where the context or more experts need it - see Options::kv_streaming)
    long long kv_res_arg = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--pack") o.pack = next();
        else if (a == "--ids") ids_arg = next();
        else if (a == "--gen") gen = std::atoi(next().c_str());
        else if (a == "--gpus") o.gpus = parse_list(next());
        else if (a == "--max-context") o.max_ctx = std::atoll(next().c_str());
        else if (a == "--layer-split") {   // (the server's "auto": each GPU filled in turn)
            const std::string v = next();
            o.layer_split = v == "auto" ? std::vector<int>{} : parse_list(v);
        }
        else if (a == "--reserve-gb") o.reserve_gb = std::atof(next().c_str());
        else if (a == "--room-report") o.room_report = true;
        else if (a == "--chunk") {   // (auto, the default: the engine's plan picks it)
            const std::string v = next();
            o.prefill_chunk = v == "auto" ? 0 : std::atoi(v.c_str());
        }
        else if (a == "--vram-experts") o.vram_experts = std::atof(next().c_str());
        else if (a == "--cpu-threads") o.cpu_threads = std::atoi(next().c_str());
        else if (a == "--gpu-fetch") o.gpu_fetch = next();
        else if (a == "--profile") o.profile = next();
        else if (a == "--tp") {
            const std::string v = next();
            o.tp = v == "auto" ? -1 : std::atoi(v.c_str());
        }
        else if (a == "--main-gpu") o.main_gpu = std::atoi(next().c_str());
        else if (a == "--profile-out") o.profile_out = next();
        else if (a == "--kld") kld = next();
        else if (a == "--kld-topk") kld = "topk:" + next();
        else if (a == "--chunks") chunks = std::atoi(next().c_str());
        else if (a == "--top") top = true;
        else if (a == "--trace") trace = std::atoi(next().c_str());
        else if (a == "--serve") serving = true;
        else if (a == "--bench-lane") bench_layer = std::atoi(next().c_str());
        else if (a == "--bench-experts") bench_n = std::atoi(next().c_str());
        else if (a == "--bench-tokens") bench_t = std::atoi(next().c_str());
        else if (a == "--mtp") o.mtp = true, drafting = "mtp";
        else if (a == "--dfly2") o.dfly2 = true, drafting = "dfly2";
        else if (a == "--draft-trial") draft_trial = next();   // (pick_drafting's child: run_draft_trial)
        else if (a == "--drafting") {   // (auto / on: measured once on this PC - pick_drafting; the rest by hand)
            drafting = next();
            if (drafting != "auto" && drafting != "on" && drafting != "off" && drafting != "mtp" && drafting != "dfly2") {
                std::fprintf(stderr, "--drafting: auto, on, off, mtp or dfly2\n");
                return 2;
            }
            o.mtp = drafting == "mtp";
            o.dfly2 = drafting == "dfly2";
        }
        else if (a == "--draft") n_draft = std::atoi(next().c_str());
        else if (a == "--draft-floor") floor = (float) std::atof(next().c_str());
        else if (a == "--draft-more") n_more = std::max(0, std::atoi(next().c_str()));
        else if (a == "--draft-sure") sure = (float) std::atof(next().c_str());
        else if (a == "--draft-streak") need = std::max(0, std::atoi(next().c_str()));
        else if (a == "--eos") eos = parse_list(next());
        else if (a == "--vision") {   // (on by default when the pack has the tower; --vision on: it must)
            const std::string v = next();
            o.vision = v != "off";
            o.vision_auto = false;
        }
        else if (a == "--images") images = next();
        else if (a == "--kld-save") kld_save = next();
        else if (a == "--kv-streaming") {
            const std::string v = next();
            if (v != "on" && v != "off" && v != "auto") {
                std::fprintf(stderr, "--kv-streaming: on, off or auto\n");
                return 2;
            }
            kv_stream = v == "on" ? 1 : v == "auto" ? -1 : 0;
        }
        else if (a == "--kv-resident") kv_res_arg = std::atoll(next().c_str());
        else if (a == "--kv-spill") {
            const std::string v = next();
            if (v != "on" && v != "off" && v != "auto") {
                std::fprintf(stderr, "--kv-spill: on, off or auto\n");
                return 2;
            }
            o.kv_spill = v == "on" ? 1 : v == "auto" ? -1 : 0;
        }
        else if (a == "--kv-spill-dir") o.kv_spill_dir = next();
        else if (a == "--kv") {
            const std::string v = next();
            if (v != "auto" && v != "f16" && v != "q8") {
                std::fprintf(stderr, "--kv: auto, f16 or q8\n");
                return 2;
            }
            o.kv = v == "auto" ? -1 : v == "q8" ? 1 : 0;
        }
        else if (a == "--encode-image") enc_rgb = next();
        else if (a == "--size") enc_size = next();
        else if (a == "--dump-image") enc_out = next();
        else if (a == "-v" || a == "--verbose") o.verbose = true;
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (o.pack.empty()) {
        std::fprintf(stderr, "usage: maya-hy4 --pack <model>/maya-hy4 (--ids @file [--gen N] | --kld ref.kld)\n");
        return 2;
    }
    if (bench_layer >= 0) return bench_lane(o.pack, bench_layer, bench_n, bench_t);
    if (!enc_rgb.empty()) return encode_alone(o, enc_rgb, enc_size, enc_out);
    if (!images.empty()) {
        o.vision = true;
        o.vision_auto = false;
    }
#if defined(_WIN32) && defined(STRATA_USE_HIP)
    // (first, so a log of a start that fails has it too)
    std::fprintf(stderr, "INFO experimental platform: an AMD GPU under Windows (HIP) - Hy4-Flash has not been tested "
                         "there by Maya's team; please include this log when reporting a problem\n");
#endif
    hy4::Model m;
    std::string err;
    o.kv_streaming = kv_stream;
    o.kv_resident = kv_stream != 0 ? (kv_res_arg > 0 ? kv_res_arg : 32768) : 0;
    if (!draft_trial.empty()) return run_draft_trial(o, draft_trial, DraftArgs{n_draft, floor, n_more, sure, need});
    std::vector<std::string> child_args;   // (a drafting trial's: this start's arguments but its drafting and its mode)
    {
        static const char* drop1[] = {"--mtp", "--dfly2", "--serve", "--room-report", "--top", "-v", "--verbose"};
        static const char* drop2[] = {"--drafting", "--draft-trial", "--ids", "--gen", "--kld", "--kld-topk", "--chunks",
                                      "--trace", "--images", "--kld-save", "--profile-out"};
        for (int i = 1; i < argc; ++i) {
            const std::string x = argv[i];
            if (std::find(std::begin(drop2), std::end(drop2), x) != std::end(drop2)) {
                ++i;
                continue;
            }
            if (std::find(std::begin(drop1), std::end(drop1), x) == std::end(drop1)) child_args.push_back(x);
        }
    }
    if ((drafting == "auto" || drafting == "on") && kld.empty() &&
        !pick_drafting(o, drafting, argv[0], child_args, err)) {
        std::fprintf(stderr, "maya-hy4: %s\n", err.c_str());
        return 1;
    }
    if (!m.load(o, err)) {
        std::fprintf(stderr, "maya-hy4: %s\n", err.c_str());
        return 1;
    }
    for (const auto& l : m.info_lines()) std::fprintf(stderr, "%s\n", l.c_str());
    if (eos.empty()) eos.push_back(m.eos());
    if (n_draft < 0) n_draft = m.draft_default();
    if (m.has_dfly2()) n_draft = std::min(n_draft, m.draft_max());
    // (DFly2: 0.8 - its later drafts are kept less often than the block's and each costs a verified row; measured on a V100
    //  from 0 to all routed experts in VRAM: as fast as 0.55 with most experts in VRAM, 4-30% faster with few)
    if (floor < 0) floor = m.has_dfly2() ? 0.8f : 0.55f;
    // (--draft-more: the MTP block's drafts - the DFly2 drafter's round is one pass of its own count; a step's window,
    //  the token and its drafts, is kMaxWin positions at most: the added drafts within it)
    n_more = m.has_dfly2() ? 0 : std::min(n_more, std::max(0, hy4::k::kMaxWin - 1 - n_draft));
    if (serving) return serve(m, eos, n_draft, floor, n_more, sure, need);
    if (kld.rfind("topk:", 0) == 0) return run_kld_topk(m, kld.substr(5), chunks, kld_save);
    if (!kld.empty()) return run_kld(m, kld, chunks, kld_save);
    std::vector<int> ids = parse_ids(ids_arg);
    if (ids.empty()) {
        std::fprintf(stderr, "maya-hy4: no prompt (--ids)\n");
        return 2;
    }
    if (!images.empty()) {   // --images: a GENI file's pictures, for the prompt's spans
        std::vector<Pic> pics;
        std::vector<hy4::Model::Image> rows;
        const double ti = now();
        if (!read_pics(images, pics, err) || !picture_rows(m, pics, ids, 0, rows, err)) {
            std::fprintf(stderr, "maya-hy4: %s\n", err.c_str());
            return 1;
        }
        long long lent = 0, own = 0;
        m.vision_reads(lent, own);   // (the warm-up's picture is one of them)
        std::fprintf(stderr, "pictures: %zu encoded in %.0f ms (so far %lld in the prompt buffers, %lld in room of "
                     "their own)\n", pics.size(), 1e3 * (now() - ti), lent, own);
        m.set_images(std::move(rows));
    }
    if (trace > 0) return run_trace(m, ids, trace);
    std::vector<float> logits((size_t) m.vocab());
    // (page-locked: with the output head over several GPUs each one copies its rows' logits straight into it)
    const bool lg_pinned = hy4::Model::pin(logits.data(), logits.size() * sizeof(float));
    (void) lg_pinned;
    const double t0 = now();
    if (!m.forward(ids.data(), (int) ids.size(), 0, 1, logits.data(), err)) {
        std::fprintf(stderr, "maya-hy4: %s\n", err.c_str());
        return 1;
    }
    const double t1 = now();
    std::fprintf(stderr, "prompt: %zu tokens in %.2f s (%.1f tok/s)", ids.size(), t1 - t0, ids.size() / (t1 - t0));
    if (m.stats().streamed > 0)
        std::fprintf(stderr, "; %lld experts streamed, %.0f ms of copies into the bounce", (long long) m.stats().streamed,
                     m.stats().stream_ms);
    std::fprintf(stderr, "\n");
    std::vector<int> out;
    int64_t pos = (int64_t) ids.size();
    if (m.has_draft() && n_draft > 0 && gen > 0) {   // greedy with the drafts (the MTP block's, or the DFly2 drafter's)
        std::vector<float> lw((size_t) (n_draft + n_more + 1) * m.vocab());
        const bool lw_pinned = hy4::Model::pin(lw.data(), lw.size() * sizeof(float));
        (void) lw_pinned;
        std::vector<int> drafts;
        std::vector<float> confs;
        long long off = 0, acc = 0, rounds = 0;
        Extra extra;
        extra.n_more = n_more;
        extra.need = need;
        double t_draft = 0, t_verify = 0;
        // (a row's top: the GPUs' where they found it - Model::row_top - else the walk over its logits)
        auto top_of = [&](const float* row, int r) {
            const int t = m.row_top(r);
            return t >= 0 ? t : argmax(row, m.vocab());
        };
        int tok = top_of(logits.data(), 0);
        while ((int) out.size() < gen) {
            out.push_back(tok);
            if (tok == m.eos() || (int) out.size() >= gen) break;
            const double ta = now();
            if (!m.draft(tok, pos, std::min(n_draft + extra.now(), gen - (int) out.size()), floor,
                         drafts, confs, err, n_draft, sure)) {
                std::fprintf(stderr, "maya-hy4: %s\n", err.c_str());
                return 1;
            }
            t_draft += now() - ta;
            int k = 0;
            while (k < (int) drafts.size() && confs[(size_t) k] >= floor) ++k;
            std::vector<int> win(1, tok);
            win.insert(win.end(), drafts.begin(), drafts.begin() + k);
            const double tb = now();
            if (!m.forward(win.data(), k + 1, pos, k + 1, lw.data(), err)) {
                std::fprintf(stderr, "maya-hy4: %s\n", err.c_str());
                return 1;
            }
            t_verify += now() - tb;
            ++rounds;
            off += k;
            pos += 1;
            int i = 0;
            for (; i < k; ++i) {
                if (top_of(lw.data() + (size_t) i * m.vocab(), i) != drafts[(size_t) i]) break;
                out.push_back(drafts[(size_t) i]);
                ++acc;
                ++pos;
                if (drafts[(size_t) i] == m.eos() || (int) out.size() >= gen) break;
            }
            extra.seen(k, i, n_draft);
            if (out.back() == m.eos() || (int) out.size() >= gen) break;
            tok = top_of(lw.data() + (size_t) i * m.vocab(), i);
        }
        const double t2 = now();
        for (size_t i = 0; i < out.size(); ++i) std::printf("%s%d", i ? "," : "", out[i]);
        std::printf("\n");
        {   // (HY4_HOT: the rows a plain decode reads counted, as there - its learning start's file the same)
            m.answered((int64_t) (ids.size() + out.size()) - 1);
            std::string perr;
            if (!m.save_profile(perr)) std::fprintf(stderr, "maya-hy4: the expert profile: %s\n", perr.c_str());
        }
        std::fprintf(stderr, "decode: %zu tokens in %.2f s (%.2f tok/s); %lld rounds, %s kept %lld of %lld drafts "
                     "(%.1f tokens a round); a round: drafts %.1f ms, verify %.1f ms\n", out.size(), t2 - t1,
                     out.size() / (t2 - t1), rounds, m.has_dfly2() ? "DFly2" : "MTP", acc, off, rounds ? (double) out.size() / rounds : 0.0,
                     rounds ? 1e3 * t_draft / rounds : 0.0, rounds ? 1e3 * t_verify / rounds : 0.0);
        return 0;
    }
    for (int i = 0; i < gen; ++i) {
        const int next = argmax(logits.data(), m.vocab());
        if (top) {
            std::vector<int> ix((size_t) m.vocab());
            for (int j = 0; j < m.vocab(); ++j) ix[(size_t) j] = j;
            std::partial_sort(ix.begin(), ix.begin() + 5, ix.end(),
                              [&](int a, int b) { return logits[(size_t) a] > logits[(size_t) b]; });
            std::fprintf(stderr, "step %d:", i);
            for (int j = 0; j < 5; ++j) std::fprintf(stderr, " %d(%.3f)", ix[(size_t) j], logits[(size_t) ix[(size_t) j]]);
            std::fprintf(stderr, "\n");
        }
        out.push_back(next);
        if (next == m.eos()) break;
        if (!m.forward(&next, 1, pos, 1, logits.data(), err)) {
            std::fprintf(stderr, "maya-hy4: %s\n", err.c_str());
            return 1;
        }
        ++pos;
    }
    const double t2 = now();
    for (size_t i = 0; i < out.size(); ++i) std::printf("%s%d", i ? "," : "", out[i]);
    std::printf("\n");
    {
        m.answered((int64_t) (ids.size() + out.size()) - 1);   // (HY4_HOT: the last token's row, if read, not counted)
        std::string perr;
        if (!m.save_profile(perr)) std::fprintf(stderr, "maya-hy4: the expert profile: %s\n", perr.c_str());
    }
    if (!out.empty()) {
        const auto& st = m.stats();
        std::fprintf(stderr, "decode: %zu tokens in %.2f s (%.2f tok/s)", out.size(), t2 - t1, out.size() / (t2 - t1));
        if (st.cpu_experts > 0)
            std::fprintf(stderr, "; CPU lane %.1f%% of the experts, %.1f ms of the run", 100.0 * st.cpu_experts / st.experts,
                         st.cpu_ms);
        std::fprintf(stderr, "\n");
    }
    return 0;
}
