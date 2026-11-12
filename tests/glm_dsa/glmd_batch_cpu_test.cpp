// tests/glm_dsa/glmd_batch_cpu_test.cpp - the glm-dsa batched prompt path's host logic on the CPU (glm-dsa B9.1, B9.3,
// B9.9, B14.6, B14.7, B14.12).  Plain C++: links glm_dsa_prompt_plan.cpp and glm_dsa_cpu_fast_twin.cpp only.
//   select    B9.3 gate: the tiled multi-row selection (running candidate sets merged tile by tile, query sub-batches
//             of 2048) equals the token path's per-token selection (glmd::twin::topk over the whole causal row, the
//             algorithm the GPU selector runs) for EVERY row of prompts of 2,049 and 9,000 tokens, a second chunk
//             (p0 > 0), key blocks 16384 / 4096 / 1000 / 2048, random scores and heavy ties;
//   walks     B14.12: the dense, dense-masked and gather attention walks attend the same cells: identical ctx;
//   prep      B9.2: T-row prep equals T single-token preps at a strided kv_raw (the row mapping);
//   tiling    B9.1 / B14.6: an 8K chunk at a 64K context stays within the 2.2 GB lend; sizes past 2^31 bytes;
//   lane      B14.7: a 200-token turn computes RAM-tier experts on the lane, an 8K chunk uploads them;
//   stream    B9.9: three links concurrently take a third of the serial time.
#include "strata/kernels/glmd_batch.hpp"

#include "../../src/kernels/cuda/glmd_rope.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "glm_test_env.hpp"   // setenv / unsetenv / getpid (<unistd.h> on POSIX; mapped on Windows)

namespace {

int g_fail = 0;

void expect(bool ok, const char* what) {
    std::printf("%-78s %s\n", what, ok ? "ok" : "FAIL");
    g_fail += !ok;
}

uint64_t mix(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

struct ScoreGen {
    uint64_t seed = 1;
    int levels = 0;   // > 0: scores quantized to this many values (heavy ties)
};

float score_of(void* user, int64_t pos, int64_t cell) {
    const ScoreGen* g = (const ScoreGen*) user;
    const uint64_t h = mix(g->seed * 0x9e3779b97f4a7c15ULL ^ ((uint64_t) pos << 32) ^ (uint64_t) cell);
    if (g->levels > 0) return (float) (h % (uint64_t) g->levels) * 0.5f;
    // mostly positive (the indexer's sum of ReLUs), some zeros, a few negative weights' sums
    const double u = (double) (h >> 11) * (1.0 / 9007199254740992.0);
    if (u < 0.05) return 0.0f;
    return (float) (std::log(u) * -3.0 - 0.4);
}

// rows [p0, p0 + T) of a chunk: token path vs the tiled sub-batched path
bool select_case(int64_t p0, int T, int key_block, int q_sub, ScoreGen gen, const char* label) {
    const int top = glmd::kTopK;
    std::vector<int> tok((size_t) T * top), bat((size_t) T * top);
    std::vector<float> row;
    for (int t = 0; t < T; ++t) {
        const int64_t pos = p0 + t, n = pos + 1;
        row.resize((size_t) n);
        for (int64_t c = 0; c < n; ++c) row[(size_t) c] = score_of(&gen, pos, c);
        glmd::twin::topk(row.data(), (int) n, (int) n, 1, nullptr, top, tok.data() + (size_t) t * top);
    }
    for (int s0 = 0; s0 < T; s0 += q_sub) {
        const int tn = std::min(q_sub, T - s0);
        glmd::twin::topk_tiled(score_of, &gen, tn, p0 + s0, key_block, top, bat.data() + (size_t) s0 * top);
    }
    int64_t bad_rows = 0;
    for (int t = 0; t < T; ++t)
        if (std::memcmp(tok.data() + (size_t) t * top, bat.data() + (size_t) t * top, sizeof(int) * top) != 0) ++bad_rows;
    char what[200];
    std::snprintf(what, sizeof what, "select %s p0 %lld T %d key_block %d: %lld rows differ", label, (long long) p0, T,
                  key_block, (long long) bad_rows);
    expect(bad_rows == 0, what);
    return bad_rows == 0;
}

void test_select() {
    ScoreGen rnd;
    rnd.seed = 7;
    ScoreGen ties;
    ties.seed = 11;
    ties.levels = 16;
    select_case(0, 2049, 16384, 2048, rnd, "random");
    select_case(0, 2049, 1000, 2048, rnd, "random");
    select_case(0, 2049, 2048, 2048, ties, "ties");
    select_case(0, 9000, 16384, 2048, rnd, "random");
    select_case(0, 9000, 4096, 2048, rnd, "random");
    select_case(0, 9000, 1000, 2048, ties, "ties");
    select_case(5000, 4000, 4096, 2048, rnd, "random, second chunk");
    select_case(0, 300, 128, 64, rnd, "short prompt (identity)");
    // the merged result is position ordered, -1 padded, and has min(n, 2048) entries per row
    {
        const int T = 40, top = glmd::kTopK;
        const int64_t p0 = 2030;
        std::vector<int> sel((size_t) T * top);
        glmd::twin::topk_tiled(score_of, &rnd, T, p0, 512, top, sel.data());
        bool ok = true;
        for (int t = 0; t < T; ++t) {
            const int64_t n = std::min<int64_t>(p0 + t + 1, top);
            const int* r = sel.data() + (size_t) t * top;
            for (int i = 0; i < top; ++i) {
                if (i < n) ok = ok && r[i] >= 0 && r[i] <= p0 + t && (i == 0 || r[i] > r[i - 1]);
                else ok = ok && r[i] == -1;
            }
        }
        expect(ok, "select rows ascending, causal, min(n, 2048) entries, -1 padded");
    }
}

void test_walks() {
    const int T = 6, H = 16, K = 64, R = 64, top = 8;
    const int64_t p0 = 30, n_kv = p0 + T;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint16_t> lat((size_t) n_kv * K), kpe((size_t) n_kv * R);
    for (auto& v : lat) v = glmd::twin::half_bits(nd(rng));
    for (auto& v : kpe) v = glmd::twin::half_bits(nd(rng));
    std::vector<float> qa((size_t) T * H * K), q((size_t) T * H * 256);
    for (auto& v : qa) v = nd(rng);
    for (auto& v : q) v = nd(rng);
    ScoreGen gen;
    gen.seed = 5;
    std::vector<int> sel((size_t) T * top);
    glmd::twin::topk_tiled(score_of, &gen, T, p0, 16, top, sel.data());
    const int64_t words = (n_kv + 31) / 32;
    std::vector<uint32_t> mask((size_t) T * words, 0u);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < top; ++i)
            if (const int c = sel[(size_t) t * top + i]; c >= 0) mask[(size_t) t * words + c / 32] |= 1u << (c % 32);
    glmd::MlaPrompt a;
    a.q_abs = qa.data();
    a.q = q.data();
    a.lat = lat.data();
    a.k_pe = kpe.data();
    a.sel = sel.data();
    a.mask = mask.data();
    a.mask_words = words;
    a.top_k = top;
    a.p0 = p0;
    a.T = T;
    a.n_head = H;
    a.kv_lora = K;
    a.qk_rope = R;
    std::vector<float> c1((size_t) T * H * K), c2(c1.size());
    a.walk = glmd::AttnWalk::Gather;
    a.ctx = c1.data();
    glmd::twin::mla_prompt(a);
    a.walk = glmd::AttnWalk::DenseMasked;
    a.ctx = c2.data();
    glmd::twin::mla_prompt(a);
    expect(c1 == c2, "walks: gather == dense-masked (same cells, same order)");
    // below the crossover every causal cell is selected: dense == gather of the identity
    std::vector<int> id((size_t) T * 64);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < 64; ++i) id[(size_t) t * 64 + i] = i <= p0 + t ? i : -1;
    a.sel = id.data();
    a.top_k = 64;
    a.walk = glmd::AttnWalk::Gather;
    a.ctx = c1.data();
    glmd::twin::mla_prompt(a);
    a.walk = glmd::AttnWalk::Dense;
    a.ctx = c2.data();
    glmd::twin::mla_prompt(a);
    expect(c1 == c2, "walks: dense == gather of the identity selection");
    expect(glmd::attn_walk(2048, 2048, 4.0) == glmd::AttnWalk::Dense &&
               glmd::attn_walk(2049, 2048, 4.0) == glmd::AttnWalk::DenseMasked &&
               glmd::attn_walk(8191, 2048, 4.0) == glmd::AttnWalk::DenseMasked &&
               glmd::attn_walk(8192, 2048, 4.0) == glmd::AttnWalk::Gather &&
               glmd::attn_walk(2049, 2048, 1.0) == glmd::AttnWalk::Gather,
           "walks: crossover at ratio x top_k (B14.12), dense at <= top_k");
}

void test_prep_rows() {
    const int T = 5, QL = 64, KV = 64, R = 16, ID = 32, IR = 16, LD = KV + R + 8;
    std::mt19937 rng(9);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> qr_raw((size_t) T * QL), kv((size_t) T * LD), ik((size_t) T * ID), qn(QL), kn(KV), lw(ID), lb(ID);
    for (auto* v : {&qr_raw, &kv, &ik, &qn, &kn, &lw, &lb})
        for (auto& x : *v) x = nd(rng);
    std::vector<float> inv(R / 2), iinv(IR / 2);
    glmd::rope_inv_freq(inv.data(), R, 8.0e6);
    glmd::rope_inv_freq(iinv.data(), IR, 8.0e6);
    const int p0 = 100, ctx = p0 + T;
    const auto run = [&](bool rows, std::vector<float>& qr, std::vector<uint16_t>& qr16, std::vector<uint16_t>& lat,
                         std::vector<uint16_t>& kpe, std::vector<float>& ikc) {
        qr.assign((size_t) T * QL, 0.0f);
        qr16.assign((size_t) T * QL, 0);
        lat.assign((size_t) ctx * KV, 0);
        kpe.assign((size_t) ctx * R, 0);
        ikc.assign((size_t) ctx * ID, 0.0f);
        glmd::DsaPrep a;
        a.q_a_norm = qn.data();
        a.q_lora = QL;
        a.kv_norm = kn.data();
        a.lat = lat.data();
        a.kv_lora = KV;
        a.k_pe = kpe.data();
        a.qk_rope = R;
        a.inv_freq = inv.data();
        a.ik_ln_w = lw.data();
        a.ik_ln_b = lb.data();
        a.ik_cache = ikc.data();
        a.idx_dim = ID;
        a.idx_rope = IR;
        a.idx_inv_freq = iinv.data();
        if (rows) {
            glmd::DsaPrepRows r;
            r.a = a;
            r.a.qr_raw = qr_raw.data();
            r.a.qr = qr.data();
            r.a.kv_raw = kv.data();
            r.a.ik_raw = ik.data();
            r.a.p = p0;
            r.T = T;
            r.kv_ld = LD;
            r.qr16 = qr16.data();
            glmd::twin::dsa_prep_rows(r);
            return;
        }
        for (int t = 0; t < T; ++t) {
            glmd::DsaPrep b = a;
            b.qr_raw = qr_raw.data() + (size_t) t * QL;
            b.qr = qr.data() + (size_t) t * QL;
            b.kv_raw = kv.data() + (size_t) t * LD;
            b.ik_raw = ik.data() + (size_t) t * ID;
            b.p = p0 + t;
            glmd::twin::dsa_prep(b);
            for (int e = 0; e < QL; ++e) qr16[(size_t) t * QL + e] = glmd::twin::half_bits(b.qr[e]);
        }
    };
    std::vector<float> qa, qb, ia, ib;
    std::vector<uint16_t> ha, hb, la, lb2, ka, kb;
    run(true, qa, ha, la, ka, ia);
    run(false, qb, hb, lb2, kb, ib);
    expect(qa == qb && ha == hb && la == lb2 && ka == kb && ia == ib,
           "prep: T rows at a strided kv_raw == T token preps (qr, qr16, latent, k_pe, keys)");
}

void test_tiling() {
    glmd::PromptDims d;   // GLM-5.3, 8K chunk, 64K context
    d.T = 8192;
    d.max_ctx = 65536;
    const glmd::PromptTiling t = glmd::prompt_tiling(d);
    std::printf("  8K chunk @64K: q_sub %d key_block %d tiles %d sub-batch %.1f MB (score %.1f, q_abs %.1f, ctx %.1f, "
                "run %.1f, mask %.1f) keep %.1f MB\n",
                t.q_sub, t.key_block, t.n_tiles_max, t.sub_bytes / 1048576.0, t.score / 1048576.0, t.q_abs / 1048576.0,
                t.ctx / 1048576.0, t.run / 1048576.0, t.mask / 1048576.0, t.keep_bytes / 1048576.0);
    expect(t.q_sub == 2048 && t.key_block == 16384 && t.n_sub == 4 && t.n_tiles_max == 4,
           "tiling: 2048-query sub-batches, 16K-key blocks (B9.1)");
    expect(t.score == (size_t) 2048 * 16384 * 4, "tiling: score scratch q_sub x key_block floats, context-free");
    expect(t.sub_bytes + t.keep_bytes <= (size_t) 2200 * 1000 * 1000, "tiling: an 8K chunk at 64K within 2.2 GB");
    glmd::PromptDims big = d;
    big.T = 32768;
    big.max_ctx = 1 << 20;
    const glmd::PromptTiling tb = glmd::prompt_tiling(big);
    expect(tb.sel_bytes == (size_t) 32768 * 2048 * 4 && tb.mask == (size_t) 2048 * (1 << 15) * 4,
           "tiling: 64-bit sizes (32K x 2048 selection, 1M-key mask rows) (B14.6)");
    const glmd::PromptTiling tc = glmd::prompt_tiling(d, (size_t) 256 << 20);
    expect(tc.sub_bytes <= ((size_t) 256 << 20) && tc.q_sub < 2048, "tiling: a 256 MB budget shrinks the sub-batch");
}

void test_lane() {
    glmd::LaneLink l;
    const int rows = glmd::upload_rows_min(l);
    std::printf("  upload threshold %d rows per expert (IQ2_M, 10 GB/s link, 60 GB/s lane)\n", rows);
    expect(rows > 1 && rows < 2048, "lane: threshold in (1, 2048) rows");
    expect(!glmd::upload_expert(200 * 8 / 256, l), "lane: a 200-token turn keeps RAM-tier experts on the CPU lane");
    expect(glmd::upload_expert(8192 * 8 / 256, l), "lane: an 8K chunk uploads them");
    glmd::LaneLink q8 = l;
    q8.bytes_per_weight = 1.0625;
    expect(glmd::upload_rows_min(q8) >= rows, "lane: more bytes per weight never lowers the threshold");
    // measured form: 2.1 ms PCIe per expert, 0.3 ms lane fixed, 0.056 ms a row (the prompt path's learned numbers)
    const int rm = glmd::upload_rows_min_measured(2.1, 0.3, 0.056);
    expect(rm == 33 && glmd::upload_rows_min_measured(0.2, 0.3, 0.056) == 0,
           "lane: measured threshold ceil((2.1 - 0.3) / 0.056) = 33 rows; a lane slower than the link forces none");
    setenv("MAYA_GLM_UPLOAD_MIN_ROWS", "5", 1);
    expect(glmd::upload_rows_min(l) == 5 && glmd::upload_rows_min_measured(2.1, 0.3, 0.056) == 5,
           "lane: MAYA_GLM_UPLOAD_MIN_ROWS overrides both forms");
    unsetenv("MAYA_GLM_UPLOAD_MIN_ROWS");
}

void test_stream() {
    glmd::StreamPart p[3];
    const int bounds[4] = {0, 26, 54, 79};
    for (int i = 0; i < 3; ++i) {
        p[i].dev = i;
        p[i].numa_node = i == 0 ? 0 : 1;
        p[i].l0 = bounds[i];
        p[i].l1 = bounds[i + 1];
        p[i].link_bps = 10.0e9;
        p[i].ram_bytes_per_layer = (int64_t) 1800 << 20;
    }
    const glmd::StreamPlan s = glmd::stream_plan(p, 3, (int64_t) 2 << 30);
    std::printf("  stream: serial %.2f s, three links %.2f s, prefetch %d/%d/%d layers\n", s.serial_s, s.overlap_s,
                s.parts[0].prefetch_layers, s.parts[1].prefetch_layers, s.parts[2].prefetch_layers);
    expect(s.n_parts == 3 && s.parts[1].numa_node == 1 && s.parts[0].buffers == 2, "stream: per part, double buffered");
    expect(s.overlap_s <= s.serial_s / 2.5 && s.overlap_s > 0.0, "stream: three links overlap (>= 2.5x)");
}

}  // namespace

int main() {
    test_select();
    test_walks();
    test_prep_rows();
    test_tiling();
    test_lane();
    test_stream();
    std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
