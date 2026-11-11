// src/core/glm_dsa_prompt_plan.cpp - host side of the glm-dsa batched prompt path (glm-dsa B9.1, B9.3, B9.9, B14.6,
// B14.7, B14.12): the scratch tiling, the attention walk crossover, the upload-or-lane rule, the per-part upload
// schedule, and the CPU twins of the batched kernels in src/kernels/cuda/glmd_batch.cu.  Plain C++: the CPU tests
// (tests/glm_dsa/glmd_batch_cpu_test.cpp) link it with glm_dsa_cpu_fast_twin.cpp and nothing else.
#include "strata/kernels/glmd_batch.hpp"

#include "../kernels/cuda/glmd_rope.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace glmd {
namespace {

size_t a256(size_t b) { return (b + 255) / 256 * 256; }

long long env_ll(const char* name, long long dflt) {
    const char* v = std::getenv(name);
    if (v == nullptr || !v[0]) return dflt;
    return std::atoll(v);
}

void sub_sizes(const PromptDims& d, PromptTiling& t) {
    const size_t Q = (size_t) t.q_sub, H = (size_t) d.n_head;
    const size_t kv_row = (size_t) d.kv_lora + d.qk_rope, qk_head = (size_t) d.qk_nope + d.qk_rope;
    t.proj = a256(Q * d.q_lora * 4) * 2 + a256(Q * d.q_lora * 2) + a256(Q * kv_row * 4) + a256(Q * d.idx_dim * 4) +
             a256(Q * d.idx_heads * 4) + a256(Q * H * qk_head * 4) + a256(Q * (size_t) d.idx_heads * d.idx_dim * 4);
    t.score = a256(Q * (size_t) t.key_block * 4);
    t.run = 2 * (a256(Q * (size_t) d.top_k * 4) * 2 + a256(Q * 4));
    const size_t words = (size_t) ((d.max_ctx + 31) / 32);
    t.mask = a256(Q * words * 4);
    t.q16 = a256(Q * H * std::max<size_t>((size_t) d.qk_nope, (size_t) d.kv_lora) * 2);
    t.q_abs = a256(Q * H * d.kv_lora * 4);
    t.ctx = a256(Q * H * d.kv_lora * 4);
    t.attn = a256(Q * H * d.v_head * 4) + a256(Q * H * d.v_head * 2);
    t.sub_bytes = t.proj + t.score + t.run + t.mask + t.q16 + t.q_abs + t.ctx + t.attn;
}

}  // namespace

PromptTiling prompt_tiling(const PromptDims& d, size_t budget) {
    PromptTiling t;
    t.key_block = (int) std::clamp<long long>(env_ll("MAYA_GLMD_KEY_BLOCK", 16384), 256, 1 << 20) / 256 * 256;
    t.q_sub = (int) std::clamp<long long>(env_ll("MAYA_GLMD_Q_SUB", 2048), 16, 8192);
    t.q_sub = (int) std::min<int64_t>(t.q_sub, std::max<int64_t>(1, d.T));
    sub_sizes(d, t);
    while (budget > 0 && t.sub_bytes > budget && t.q_sub > 64) {
        t.q_sub = std::max(64, t.q_sub / 2);
        sub_sizes(d, t);
    }
    while (budget > 0 && t.sub_bytes > budget && t.key_block > 1024) {
        t.key_block /= 2;
        sub_sizes(d, t);
    }
    t.n_sub = (int) ((d.T + t.q_sub - 1) / t.q_sub);
    t.n_tiles_max = (int) ((d.max_ctx + t.key_block - 1) / t.key_block);
    t.sel_bytes = a256((size_t) d.T * (size_t) d.top_k * 4);
    t.keep_bytes = t.sel_bytes + a256((size_t) (d.qk_rope / 2 + d.idx_dim / 2) * 4);
    return t;
}

double gather_ratio(double dflt) {
    const char* v = std::getenv("MAYA_GLMD_GATHER_RATIO");
    if (v != nullptr && v[0]) {
        const double r = std::atof(v);
        if (r >= 1.0) return r;
    }
    return dflt;
}

AttnWalk attn_walk(int64_t n_kv, int top_k, double ratio) {
    if (n_kv <= top_k) return AttnWalk::Dense;
    if ((double) n_kv < ratio * (double) top_k) return AttnWalk::DenseMasked;
    return AttnWalk::Gather;
}

int upload_rows_min(const LaneLink& l) {
    const long long env = env_ll("MAYA_GLM_UPLOAD_MIN_ROWS", -1);
    if (env >= 0) return (int) env;
    // upload: bytes / link once.  lane: per row, the expert's bytes stream through the lane once per row block of
    // kLaneRows rows (the kernels block rows) plus the row compute.  rows* solves upload = lane.
    constexpr double kLaneRows = 8.0;
    const double bytes = l.bytes_per_weight * (double) l.weights;
    const double up = bytes / std::max(1.0, l.link_bps);
    const double lane_per_row = bytes / std::max(1.0, l.lane_bps) / kLaneRows + l.lane_row_s;
    if (lane_per_row <= 0.0) return 1;
    return std::max(1, (int) std::ceil(up / lane_per_row));
}

bool upload_expert(int rows, const LaneLink& l) { return rows >= upload_rows_min(l); }

int upload_rows_min_measured(double upload_ms, double lane_expert_ms, double lane_row_ms) {
    const long long env = env_ll("MAYA_GLM_UPLOAD_MIN_ROWS", -1);
    if (env >= 0) return (int) env;
    if (!(upload_ms > lane_expert_ms) || !(lane_row_ms > 0.0)) return 0;
    return std::max(1, (int) std::ceil((upload_ms - lane_expert_ms) / lane_row_ms));
}

StreamPlan stream_plan(const StreamPart* parts, int n, int64_t staging_bytes_per_part) {
    StreamPlan p;
    p.n_parts = std::min(n, 8);
    for (int i = 0; i < p.n_parts; ++i) {
        const StreamPart& s = parts[i];
        StreamPlan::Part& o = p.parts[i];
        o.dev = s.dev;
        o.numa_node = s.numa_node;
        o.buffers = 2;
        o.slot_bytes = staging_bytes_per_part / 2;
        const int layers = std::max(0, s.l1 - s.l0);
        o.upload_s = (double) s.ram_bytes_per_layer * layers / std::max(1.0, s.link_bps);
        // the next part's first layers ride its link while this part computes: as many as one slot holds
        o.prefetch_layers = s.ram_bytes_per_layer > 0
                                ? (int) std::min<int64_t>(layers, std::max<int64_t>(1, o.slot_bytes / s.ram_bytes_per_layer))
                                : 0;
        p.serial_s += o.upload_s;
        p.overlap_s = std::max(p.overlap_s, o.upload_s);
    }
    return p;
}

// ---------------------------------------------------------------- CPU twins
namespace twin {
namespace {

// latent value c of a cell as both prompt kernels stage it (lat_h in glmd_batch.cu): FP16 (an INT8 record is decoded,
// then rounded to FP16; code x scale is exact in float and half_bits rounds as __float2half: the kernels' bits)
float lat_value(const void* lat, int lat_q8, int kv_lora, int64_t cell, int c) {
    if (!lat_q8) return half_float(((const uint16_t*) lat)[(size_t) kv_lora * cell + c]);
    const uint8_t* rec = (const uint8_t*) lat + (size_t) lat8_rec_bytes(kv_lora) * cell;
    uint16_t d;
    std::memcpy(&d, rec + kv_lora + 2 * (c / 32), 2);
    return half_float(half_bits((float) ((const int8_t*) rec)[c] * half_float(d)));
}

}  // namespace

void dsa_prep_rows(const DsaPrepRows& r) {
    for (int t = 0; t < r.T; ++t) {
        DsaPrep a = r.a;
        a.p = r.a.p_dev ? r.a.p_dev[t] : r.a.p + t;
        a.p_dev = nullptr;
        a.qr_raw = r.a.qr_raw + (size_t) t * r.a.q_lora;
        if (a.qr) a.qr = r.a.qr + (size_t) t * r.a.q_lora;
        if (a.qr_q) a.qr_q = (Q81*) r.a.qr_q + (size_t) t * (r.a.q_lora / 32);
        a.kv_raw = r.a.kv_raw + (size_t) t * r.kv_ld;
        if (a.ik_raw) a.ik_raw = r.a.ik_raw + (size_t) t * r.a.idx_dim;
        // the FP16 qr comes from the f32 one: keep a row of it when the caller asked for none
        std::vector<float> qr_tmp;
        if (r.qr16 && !a.qr) {
            qr_tmp.resize((size_t) r.a.q_lora);
            a.qr = qr_tmp.data();
        }
        glmd::twin::dsa_prep(a);
        if (r.qr16)
            for (int e = 0; e < r.a.q_lora; ++e) r.qr16[(size_t) t * r.a.q_lora + e] = half_bits(a.qr[e]);
    }
}

void dsa_rope_q_rows(const DsaRopeQRows& r) {
    for (int t = 0; t < r.T; ++t) {
        DsaRopeQ a = r.a;
        a.p = r.a.p_dev ? r.a.p_dev[t] : r.a.p + t;
        a.p_dev = nullptr;
        a.q = r.a.q + (size_t) t * r.q_row;
        if (a.iq) a.iq = r.a.iq + (size_t) t * r.iq_row;
        glmd::twin::dsa_rope_q(a);
    }
}

void index_score_tile(const float* iq, const float* w, const float* ik_cache, int idx_heads, int idx_dim, int T,
                      int64_t k0, int nk, float* score, int64_t ld) {
    index_score_tile_keys(iq, w, ik_cache, 0, idx_heads, idx_dim, T, k0, nk, score, ld);
}

void index_score_tile_keys(const float* iq, const float* w, const void* keys, int key_f16, int idx_heads, int idx_dim,
                           int T, int64_t k0, int nk, float* score, int64_t ld) {
    std::vector<float> wide((size_t) idx_dim);
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < nk; ++j) {
            const float* k = (const float*) keys + (size_t) (k0 + j) * idx_dim;
            if (key_f16) {   // B10.8: FP16 rows widened as the kernel stages them
                const uint16_t* h = (const uint16_t*) keys + (size_t) (k0 + j) * idx_dim;
                for (int e = 0; e < idx_dim; ++e) wide[(size_t) e] = half_float(h[e]);
                k = wide.data();
            }
            float acc = 0.0f;
            for (int h = 0; h < idx_heads; ++h) {
                const float* q = iq + ((size_t) t * idx_heads + h) * idx_dim;
                float d = 0.0f;
                for (int e = 0; e < idx_dim; ++e) d += q[e] * k[e];
                acc += w[(size_t) t * idx_heads + h] * std::max(d, 0.0f);
            }
            score[(size_t) t * ld + j] = acc;
        }
}

void topk_merge_tile(const TopkMerge& m) {
    std::vector<uint32_t> key;
    std::vector<float> sc;
    std::vector<int> cell;
    for (int t = 0; t < m.T; ++t) {
        const int64_t pos = m.p0 + t;
        const int nt = (int) std::clamp<int64_t>(pos + 1 - m.k0, 0, m.nk);
        const int r = m.in_n ? m.in_n[t] : 0;
        const int n = r + nt;
        sc.resize((size_t) n);
        cell.resize((size_t) n);
        for (int i = 0; i < r; ++i) {
            sc[(size_t) i] = m.in_score[(size_t) t * m.top_k + i];
            cell[(size_t) i] = m.in_cell[(size_t) t * m.top_k + i];
        }
        for (int j = 0; j < nt; ++j) {
            sc[(size_t) (r + j)] = m.score[(size_t) t * m.ld + j];
            cell[(size_t) (r + j)] = (int) (m.k0 + j);
        }
        int* oc = m.out_cell + (size_t) t * m.top_k;
        float* os = m.out_score + (size_t) t * m.top_k;
        int o = 0;
        if (n <= m.top_k) {
            for (int i = 0; i < n; ++i) {
                oc[o] = cell[(size_t) i];
                os[o++] = sc[(size_t) i];
            }
        } else {
            key.resize((size_t) n);
            for (int i = 0; i < n; ++i) key[(size_t) i] = score_key(sc[(size_t) i]);
            uint32_t prefix = 0u, mask = 0u;
            int need = m.top_k;
            for (int shift = 24; shift >= 0; shift -= 8) {
                int hist[256] = {};
                for (int i = 0; i < n; ++i)
                    if ((key[(size_t) i] & mask) == prefix) ++hist[(key[(size_t) i] >> shift) & 255u];
                for (int b = 255; b >= 0; --b) {
                    if (hist[b] >= need) {
                        prefix |= (uint32_t) b << shift;
                        mask |= 255u << shift;
                        break;
                    }
                    need -= hist[b];
                }
            }
            int eq_seen = 0;
            for (int i = 0; i < n; ++i) {
                const bool take = key[(size_t) i] > prefix || (key[(size_t) i] == prefix && eq_seen++ < need);
                if (take) {
                    oc[o] = cell[(size_t) i];
                    os[o++] = sc[(size_t) i];
                }
            }
        }
        m.out_n[t] = o;
        for (int i = o; i < m.top_k; ++i) {
            oc[i] = -1;
            os[i] = 0.0f;
        }
    }
}

void topk_tiled(ScoreFn fn, void* user, int T, int64_t p0, int key_block, int top_k, int* sel) {
    const int64_t n_kv = p0 + T;
    const int n_tiles = (int) ((n_kv + key_block - 1) / key_block);
    std::vector<float> tile((size_t) T * key_block);
    std::vector<int> cell_a((size_t) T * top_k), cell_b((size_t) T * top_k), n_a((size_t) T), n_b((size_t) T);
    std::vector<float> sc_a((size_t) T * top_k), sc_b((size_t) T * top_k);
    int* cin = nullptr;
    float* sin = nullptr;
    int* nin = nullptr;
    for (int k = 0; k < n_tiles; ++k) {
        const int64_t k0 = (int64_t) k * key_block;
        const int nk = (int) std::min<int64_t>(key_block, n_kv - k0);
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < nk; ++j)
                tile[(size_t) t * key_block + j] = k0 + j <= p0 + t ? fn(user, p0 + t, k0 + j) : 0.0f;
        const bool last = k == n_tiles - 1;
        int* cout = last ? sel : (k & 1 ? cell_b.data() : cell_a.data());
        float* sout = k & 1 ? sc_b.data() : sc_a.data();
        int* nout = k & 1 ? n_b.data() : n_a.data();
        TopkMerge m;
        m.score = tile.data();
        m.ld = key_block;
        m.T = T;
        m.p0 = p0;
        m.k0 = k0;
        m.nk = nk;
        m.top_k = top_k;
        m.in_cell = cin;
        m.in_score = sin;
        m.in_n = nin;
        m.out_cell = cout;
        m.out_score = sout;
        m.out_n = nout;
        topk_merge_tile(m);
        cin = cout;
        sin = sout;
        nin = nout;
    }
    if (n_tiles == 0)
        for (size_t i = 0; i < (size_t) T * top_k; ++i) sel[i] = -1;
}

void mla_prompt(const MlaPrompt& a) {
    const int R = a.qk_rope, K = a.kv_lora;
    std::vector<int64_t> cells;
    std::vector<float> l, lv, kv;
    for (int t = 0; t < a.T; ++t) {
        const int64_t pos = a.p0 + t;
        cells.clear();
        if (a.walk == AttnWalk::Gather) {
            for (int i = 0; i < a.top_k; ++i) {
                const int c = a.sel[(size_t) t * a.top_k + i];
                if (c >= 0 && c <= pos) cells.push_back(c);
            }
        } else {
            for (int64_t c = 0; c <= pos; ++c)
                if (a.walk == AttnWalk::Dense || ((a.mask[(size_t) t * a.mask_words + c / 32] >> (c % 32)) & 1u))
                    cells.push_back(c);
        }
        // every cell's values once per row (the same floats the per-head loops read)
        lv.resize(cells.size() * (size_t) K);
        kv.resize(cells.size() * (size_t) R);
        for (size_t i = 0; i < cells.size(); ++i) {
            for (int c = 0; c < K; ++c) lv[i * K + c] = lat_value(a.lat, a.lat_q8, K, cells[i], c);
            for (int j = 0; j < R; ++j) kv[i * R + j] = half_float(a.k_pe[(size_t) R * cells[i] + j]);
        }
        for (int h = 0; h < a.n_head; ++h) {
            const float* qa = a.q_abs + ((size_t) t * a.n_head + h) * K;
            const float* qp = a.q + ((size_t) t * a.n_head + h) * a.q_stride + a.q_pe_off;
            l.resize(cells.size());
            float mx = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < cells.size(); ++i) {
                float acc = 0.0f;
                for (int c = 0; c < K; ++c) acc += qa[c] * lv[i * K + c];
                for (int j = 0; j < R; ++j) acc += qp[j] * kv[i * R + j];
                l[i] = acc * a.scale;
                mx = std::max(mx, l[i]);
            }
            float sum = 0.0f;
            for (float& v : l) {
                v = std::exp(v - mx);
                sum += v;
            }
            float* out = a.ctx + ((size_t) t * a.n_head + h) * K;
            for (int c = 0; c < K; ++c) {
                float acc = 0.0f;
                for (size_t i = 0; i < cells.size(); ++i) acc += l[i] * lv[i * K + c];
                out[c] = sum > 0.0f ? acc / sum : 0.0f;
            }
        }
    }
}

}  // namespace twin
}  // namespace glmd
