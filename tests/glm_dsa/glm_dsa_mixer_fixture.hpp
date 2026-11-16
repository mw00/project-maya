// tests/glm_dsa/glm_dsa_mixer_fixture.hpp - the glm-dsa decode mixer's test fixture (glm-dsa B4.6, B4.7, B14.3).
//
// Shared by tests/glm_dsa/glm_dsa_mixer_cpu_test.cpp (the orchestration on the CPU twins; runs anywhere) and
// tests/glm_dsa/glmd_kernel_parity.cu (the same orchestration on the GPU kernels).  It holds:
//   * a random layer stack of a given geometry and full/shared pattern (indexer weights on the shared layers too, so
//     the shared-as-full misreading has something to read; tiny-variance index keys with a LayerNorm bias, so the
//     LayerNorm eps changes the selection; k_b / v_b rounded to BF16, the values the kernels read);
//   * the reference layer: the B3.4 CPU reference functions (glmd::cpu, double accumulation, F32 caches), with the
//     selection policy of the F32 reference forward (glm_dsa_ref.cu: every cell while p + 1 <= top_k, the indexer's
//     top-k after; shared layers reuse the previous full layer's selection of the same token);
//   * Harness<B>: glmdsa::mixer_layer over a backend B (host twins or device kernels) with the named wrong readings
//     of B4.7 injected (NEOX pairing, the rope at offset 0, the indexer LayerNorm eps 0, shared layers run as full,
//     the scale 1/sqrt(qk_nope)), and the per-position comparison against the reference.
#pragma once

#include "glm_dsa_mixer.hpp"
#include "strata/kernels/glmd_fast.hpp"
#include "strata/kernels/glmd_mixer.hpp"
#include "strata/kernels/glmd_ref.hpp"
#include "strata/kernels/glmd_volta.hpp"

#include "../../src/kernels/cuda/glmd_rope.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace mixfx {

using strata::core::glmdsa::Gemv;
using strata::core::glmdsa::MixerBufs;
using strata::core::glmdsa::MixerCache;
using strata::core::glmdsa::MixerDims;
using strata::core::glmdsa::MixerLayer;
using strata::core::glmdsa::Proj;
using strata::core::glmdsa::SelState;

// ---------------------------------------------------------------- geometry presets
/// SMALL of tools/glm_dsa_synth_gguf.py (n_embd 128, 4 heads, kv_lora 32 + rope 16, indexer 16 x 64, top-k 8).
inline MixerDims small_dims(int64_t ctx) {
    MixerDims d;
    d.n_embd = 128, d.n_head = 4, d.q_lora = 64, d.kv_lora = 32, d.qk_nope = 48, d.qk_rope = 16, d.v_head = 32;
    d.idx_heads = 16, d.idx_key = 64, d.idx_rope = 16, d.top_k = 8;
    d.max_ctx = ctx;
    d.eps = 1e-5f, d.idx_ln_eps = 1e-6f;
    d.kq_scale = 1.0f / std::sqrt((float) d.qk_head());
    return d;
}
/// A GPU-shaped geometry (mla2 needs qk_rope 64): PACK's MLA head (kv_lora 512, nope 192 + rope 64, v_head 256) with
/// 4 heads, indexer 8 x 128, top-k 16.
inline MixerDims mid_dims(int64_t ctx) {
    MixerDims d;
    d.n_embd = 256, d.n_head = 4, d.q_lora = 256, d.kv_lora = 512, d.qk_nope = 192, d.qk_rope = 64, d.v_head = 256;
    d.idx_heads = 8, d.idx_key = 128, d.idx_rope = 64, d.top_k = 16;
    d.max_ctx = ctx;
    d.eps = 1e-5f, d.idx_ln_eps = 1e-6f;
    d.kq_scale = 0.0625f;   // glmd::kGlm53KqScale at qk_head 256
    return d;
}

// ---------------------------------------------------------------- data
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uni() { return (double) (next() >> 11) * (1.0 / 9007199254740992.0); }
    float normal() {
        const double u = std::max(uni(), 1e-300), v = uni();
        return (float) (std::sqrt(-2.0 * std::log(u)) * std::cos(6.283185307179586 * v));
    }
};

inline uint16_t bf16_rn(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
inline float bf16_f(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct LayerData {
    bool full = false;
    int idx_src = -1;
    std::vector<float> qa, qa_norm, qb, kva, kva_norm, wkb, wvb, out;   // F32 (wkb / wvb hold BF16-exact values)
    std::vector<uint16_t> wkb16, wvb16;
    std::vector<float> idx_k, ln_w, ln_b, idx_q, idx_p;                 // on every layer (shared-as-full reads them)
};

struct Model {
    MixerDims d;
    std::vector<LayerData> L;
    int T = 0;
    std::vector<float> X;                   // attention inputs: x(il, p) at ((p * n_layers + il) * n_embd)
    std::vector<float> inv_q, inv_idx;      // rope tables (theta 8e6)
    const float* x(int il, int p) const { return X.data() + ((size_t) p * L.size() + il) * d.n_embd; }
};

inline std::vector<float> rnd(Rng& r, size_t n, float sd, float mean = 0.0f) {
    std::vector<float> v(n);
    for (auto& e : v) e = mean + sd * r.normal();
    return v;
}

/// A layer stack with the given full (true) / shared (false) pattern; layer 0 must be full.
inline Model make_model(const MixerDims& d, const std::vector<bool>& pattern, int T, uint64_t seed) {
    Model m;
    m.d = d;
    m.T = T;
    Rng r(seed);
    const int E = d.n_embd, QL = d.q_lora, KL = d.kv_lora, H = d.n_head, QH = d.qk_head(), NOPE = d.qk_nope;
    int last_full = -1;
    for (size_t il = 0; il < pattern.size(); ++il) {
        LayerData Ld;
        Ld.full = pattern[il];
        Ld.idx_src = Ld.full ? (int) il : last_full;
        if (Ld.full) last_full = (int) il;
        const float se = 1.0f / std::sqrt((float) E), sq = 1.0f / std::sqrt((float) QL);
        Ld.qa = rnd(r, (size_t) QL * E, se);
        Ld.qa_norm = rnd(r, QL, 0.1f, 1.0f);
        Ld.qb = rnd(r, (size_t) H * QH * QL, sq);
        Ld.kva = rnd(r, (size_t) (KL + d.qk_rope) * E, se);
        Ld.kva_norm = rnd(r, KL, 0.1f, 1.5f);
        Ld.wkb = rnd(r, (size_t) H * KL * NOPE, 1.0f / std::sqrt((float) NOPE));
        Ld.wvb = rnd(r, (size_t) H * d.v_head * KL, 1.0f / std::sqrt((float) KL));
        for (auto* w : {&Ld.wkb, &Ld.wvb}) {
            auto& b16 = w == &Ld.wkb ? Ld.wkb16 : Ld.wvb16;
            b16.resize(w->size());
            for (size_t i = 0; i < w->size(); ++i) {
                b16[i] = bf16_rn((*w)[i]);
                (*w)[i] = bf16_f(b16[i]);
            }
        }
        Ld.out = rnd(r, (size_t) E * H * d.v_head, 1.0f / std::sqrt((float) (H * d.v_head)));
        // tiny-variance index keys (std ~1e-3, as SMALL's idx_k_gain): the LayerNorm eps 1e-6 vs 0 matters
        Ld.idx_k = rnd(r, (size_t) d.idx_key * E, 1e-3f * se);
        Ld.ln_w = rnd(r, d.idx_key, 0.2f, 1.0f);
        Ld.ln_b = rnd(r, d.idx_key, 0.5f);
        Ld.idx_q = rnd(r, (size_t) d.idx_heads * d.idx_key * QL, sq);
        Ld.idx_p = rnd(r, (size_t) d.idx_heads * E, se);
        m.L.push_back(std::move(Ld));
    }
    m.X = rnd(r, (size_t) T * pattern.size() * E, 1.0f);
    m.inv_q.resize((size_t) std::max(1, d.qk_rope / 2));
    m.inv_idx.resize((size_t) std::max(1, d.idx_rope / 2));
    glmd::rope_inv_freq(m.inv_q.data(), d.qk_rope, 8e6);
    glmd::rope_inv_freq(m.inv_idx.data(), d.idx_rope, 8e6);
    return m;
}

// ---------------------------------------------------------------- the reference layer
struct RefState {
    std::vector<std::vector<float>> kv, ik;   // per layer: F32 [ctx][kv_row], [ctx][idx_key]
    std::vector<int> sel;                     // the current token's selection (cells, unordered)
    int sel_layer = -1;
};

inline void ref_rms(float* x, const float* w, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const double inv = 1.0 / std::sqrt(ss / n + eps);
    for (int i = 0; i < n; ++i) x[i] = (float) (x[i] * inv * w[i]);
}

/// The reference mixer output of layer il at position p (n_embd floats); sel_out: the selection it used.
inline void ref_layer(const Model& m, RefState& R, int il, int p, bool dense, std::vector<float>& out,
                      std::vector<int>* sel_out = nullptr) {
    const MixerDims& d = m.d;
    const LayerData& Ld = m.L[(size_t) il];
    const int E = d.n_embd, QL = d.q_lora, KL = d.kv_lora, H = d.n_head, QH = d.qk_head(), KVR = d.kv_row();
    if (R.kv.size() < m.L.size()) {
        R.kv.assign(m.L.size(), {});
        R.ik.assign(m.L.size(), {});
    }
    auto& kvc = R.kv[(size_t) il];
    auto& ikc = R.ik[(size_t) il];
    if (kvc.empty()) kvc.assign((size_t) d.max_ctx * KVR, 0.0f);
    if (ikc.empty()) ikc.assign((size_t) d.max_ctx * d.idx_key, 0.0f);
    const float* x = m.x(il, p);
    std::vector<float> qr(QL);
    glmd::cpu::gemv(Ld.qa.data(), x, qr.data(), E, QL, 1, 1.0f);
    ref_rms(qr.data(), Ld.qa_norm.data(), QL, d.eps);
    const int n_kv = p + 1;
    if (Ld.full) {
        glmd::cpu::indexer_key(x, Ld.idx_k.data(), Ld.ln_w.data(), Ld.ln_b.data(), d.idx_ln_eps, E, d.idx_key, 1,
                               d.idx_rope, &p, m.inv_idx.data(), ikc.data() + (size_t) p * d.idx_key);
        std::vector<float> iq((size_t) d.idx_heads * d.idx_key), w(d.idx_heads), score(n_kv);
        glmd::cpu::indexer_query(qr.data(), Ld.idx_q.data(), QL, d.idx_heads, d.idx_key, 1, d.idx_rope, &p,
                                 m.inv_idx.data(), iq.data());
        glmd::cpu::indexer_weights(x, Ld.idx_p.data(), E, d.idx_heads, d.idx_key, 1, w.data());
        R.sel.clear();
        if (dense || n_kv <= d.top_k) {
            for (int s = 0; s < n_kv; ++s) R.sel.push_back(s);
        } else {
            glmd::cpu::indexer_score(iq.data(), ikc.data(), w.data(), d.idx_heads, d.idx_key, 1, n_kv, &p, score.data());
            R.sel.resize(d.top_k);
            glmd::cpu::topk(score.data(), n_kv, 1, d.top_k, R.sel.data());
        }
        R.sel_layer = il;
    }
    if (sel_out) *sel_out = R.sel;
    std::vector<float> q((size_t) H * QH);
    glmd::cpu::gemv(Ld.qb.data(), qr.data(), q.data(), QL, H * QH, 1, 1.0f);
    glmd::cpu::rope_norm(q.data(), QH, H, 1, d.qk_nope, d.qk_rope, &p, m.inv_q.data());
    float* kv = kvc.data() + (size_t) p * KVR;
    glmd::cpu::gemv(Ld.kva.data(), x, kv, E, KVR, 1, 1.0f);
    glmd::cpu::rope_norm(kv, KVR, 1, 1, KL, d.qk_rope, &p, m.inv_q.data());
    ref_rms(kv, Ld.kva_norm.data(), KL, d.eps);
    std::vector<float> Q((size_t) H * KVR), o((size_t) H * d.v_head);
    glmd::cpu::mla_absorb_q(q.data(), Ld.wkb.data(), H, d.qk_nope, d.qk_rope, KL, 1, Q.data());
    glmd::cpu::mla_attention(Q.data(), kvc.data(), R.sel.data(), Ld.wvb.data(), H, KL, d.qk_rope, d.v_head, 1,
                             (int) R.sel.size(), d.kq_scale, o.data());
    out.assign(E, 0.0f);
    glmd::cpu::gemv(Ld.out.data(), o.data(), out.data(), H * d.v_head, E, 1, 1.0f);
}

inline double rel_l2(const float* a, const std::vector<float>& ref) {
    double e = 0, n = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        e += ((double) a[i] - ref[i]) * ((double) a[i] - ref[i]);
        n += (double) ref[i] * ref[i];
    }
    return std::sqrt(e / std::max(n, 1e-300));
}

// ---------------------------------------------------------------- the harness
enum class Mut { None = 0, Neox, RopeOffset, IdxEps, SharedAsFull, Scale };
inline const char* mut_name(Mut m) {
    switch (m) {
    case Mut::None: return "correct";
    case Mut::Neox: return "NEOX pairing";
    case Mut::RopeOffset: return "q_pe rope at offset 0";
    case Mut::IdxEps: return "indexer LayerNorm eps 0";
    case Mut::SharedAsFull: return "shared layers run as full";
    case Mut::Scale: return "scale 1/sqrt(qk_nope)";
    }
    return "?";
}

/// Undo a NORM rotation at p and apply the NEOX one (pairs (i, i + n/2)) instead: what a NEOX reading would store.
inline void norm_to_neox(float* x, int n_rot, int p, const float* inv) {
    std::vector<float> u(n_rot);
    for (int i = 0; i < n_rot / 2; ++i) {
        float c, s;
        glmd::rope_cs(p, inv[i], c, s);
        u[2 * i] = x[2 * i] * c + x[2 * i + 1] * s;
        u[2 * i + 1] = -x[2 * i] * s + x[2 * i + 1] * c;
    }
    for (int i = 0; i < n_rot / 2; ++i) {
        float c, s;
        glmd::rope_cs(p, inv[i], c, s);
        const float a = u[i], b = u[i + n_rot / 2];
        x[i] = a * c - b * s;
        x[i + n_rot / 2] = a * s + b * c;
    }
}

/// Backend B provides (all pointers are B's own memory, device or host):
///   void* alloc(size_t bytes);  void put(void* dst, const void* host, size_t bytes);
///   void get(void* host, const void* src, size_t bytes);  void sync();
///   bool gemv(const float* W, const float* x, float* y, int in, int out, float alpha);
///   bool prep / rope_q / score / mla2 (glmd structs), bool topk(score, ld, n_kv, pos, top_k, sel)
template <class B>
struct Harness {
    B& be;
    const Model& m;
    Mut mut = Mut::None;
    bool dense = false;
    int lat_q8 = 0;
    int w_type = glmd::kWTypeBF16;   // glm-dsa wave 5 (B10.7): glmd::kWTypeQ8_0 uploads W_kb / W_vb as Q8_0 rows
    MixerDims d;
    MixerBufs b;
    struct LW {
        float *qa, *qa_norm, *qb, *kva, *kva_norm, *out, *idx_k, *ln_w, *ln_b, *idx_q, *idx_p;
        uint16_t *wkb, *wvb;
    };
    std::vector<LW> lw;
    std::vector<MixerCache> cache;
    int* p_word = nullptr;   // B memory
    SelState st;
    float* x_buf = nullptr;
    std::vector<int> cells_h;

    template <class T>
    T* up(const std::vector<T>& v) {
        T* p = (T*) be.alloc(v.size() * sizeof(T));
        be.put(p, v.data(), v.size() * sizeof(T));
        return p;
    }

    Harness(B& be_, const Model& m_, Mut mu, bool dense_, int lat_q8_, int w_type_ = glmd::kWTypeBF16)
        : be(be_), m(m_), mut(mu), dense(dense_), lat_q8(lat_q8_), w_type(w_type_) {
        d = m.d;
        if (mut == Mut::Scale) d.kq_scale = 1.0f / std::sqrt((float) d.qk_nope);
        if (mut == Mut::IdxEps) d.idx_ln_eps = 0.0f;
        const int E = d.n_embd, H = d.n_head;
        const auto f = [&](size_t n) { return (float*) be.alloc(n * 4); };
        x_buf = f(E);
        b.x = x_buf;
        b.xq = be.alloc((size_t) E / 32 * 36);
        b.qr_raw = f(d.q_lora);
        b.qr = f(d.q_lora);
        b.qr_q = be.alloc((size_t) d.q_lora / 32 * 36);
        b.kv_raw = f(d.kv_row());
        b.ik_raw = f(d.idx_key);
        b.iw = f(d.idx_heads);
        b.q = f((size_t) H * d.qk_head());
        b.iq = f((size_t) d.idx_heads * d.idx_key);
        b.score = f(d.max_ctx);
        b.cells = (int*) be.alloc((size_t) d.top_k * 4);
        std::vector<int> iota((size_t) d.max_ctx);
        for (size_t i = 0; i < iota.size(); ++i) iota[i] = (int) i;
        b.cells_dense = up(iota);
        b.n_sel_dev = (int*) be.alloc(4);
        b.attn_out = f((size_t) H * d.v_head);
        b.attn_out_q = be.alloc((size_t) H * d.v_head / 32 * 36);
        b.mixer = f(E);
        b.work_bytes = strata::core::glmdsa::mixer_work_bytes(d, dense);
        b.work = be.alloc(b.work_bytes);
        b.inv_freq = up(m.inv_q);
        b.idx_inv_freq = up(m.inv_idx);
        p_word = (int*) be.alloc(4);
        b.p_dev = p_word;
        for (const LayerData& Ld : m.L) {
            LW w;
            w.qa = up(Ld.qa), w.qa_norm = up(Ld.qa_norm), w.qb = up(Ld.qb), w.kva = up(Ld.kva);
            w.kva_norm = up(Ld.kva_norm), w.out = up(Ld.out), w.idx_k = up(Ld.idx_k), w.ln_w = up(Ld.ln_w);
            w.ln_b = up(Ld.ln_b), w.idx_q = up(Ld.idx_q), w.idx_p = up(Ld.idx_p);
            if (w_type == glmd::kWTypeQ8_0) {   // ggml's quantize_row_q8_0 of the BF16-exact values, row by row
                const auto q8 = [&](const std::vector<float>& f, int row) {
                    std::vector<uint8_t> q(f.size() / 32 * 34);
                    for (size_t r = 0; r < f.size() / (size_t) row; ++r)
                        glmd::vtwin::q8_0_quantize(f.data() + r * row, row, q.data() + r * (size_t) row / 32 * 34);
                    return (uint16_t*) up(q);
                };
                w.wkb = q8(Ld.wkb, d.qk_nope);
                w.wvb = q8(Ld.wvb, d.kv_lora);
            } else {
                w.wkb = up(Ld.wkb16), w.wvb = up(Ld.wvb16);
            }
            lw.push_back(w);
            MixerCache c;
            const size_t lat_b = lat_q8 ? (size_t) glmd::lat8_rec_bytes(d.kv_lora) : (size_t) d.kv_lora * 2;
            c.lat = be.alloc(lat_b * (size_t) d.max_ctx);
            c.lat_q8 = lat_q8;
            c.k_pe = (uint16_t*) be.alloc((size_t) d.qk_rope * 2 * (size_t) d.max_ctx);
            c.ik = (float*) be.alloc((size_t) d.idx_key * 4 * (size_t) d.max_ctx);   // every layer (shared-as-full)
            cache.push_back(c);
        }
    }

    // ---- the Ops mixer_layer calls
    int cur_p = 0;
    bool gemv(int il, const Gemv* j, int n) {
        const LW& w = lw[(size_t) il];
        for (int i = 0; i < n; ++i) {
            const float* W = nullptr;
            switch (j[i].w) {
            case Proj::QA: W = w.qa; break;
            case Proj::KVA: W = w.kva; break;
            case Proj::IdxK: W = w.idx_k; break;
            case Proj::IdxProj: W = w.idx_p; break;
            case Proj::QB: W = w.qb; break;
            case Proj::IdxQB: W = w.idx_q; break;
            case Proj::Out: W = w.out; break;
            }
            if (!be.gemv(W, j[i].xf, j[i].y, j[i].n_in, j[i].n_out, j[i].alpha)) return false;
        }
        return true;
    }
    bool prep(const glmd::DsaPrep& a0) {
        glmd::DsaPrep a = a0;
        if (!be.prep(a)) return false;
        if (mut == Mut::Neox) {   // what a NEOX reading would have stored at p
            const int p = cur_p;
            std::vector<uint16_t> kp((size_t) d.qk_rope);
            be.get(kp.data(), a.k_pe + (size_t) d.qk_rope * p, kp.size() * 2);
            std::vector<float> f(kp.size());
            for (size_t i = 0; i < kp.size(); ++i) f[i] = glmd::twin::half_float(kp[i]);
            norm_to_neox(f.data(), d.qk_rope, p, m.inv_q.data());
            for (size_t i = 0; i < kp.size(); ++i) kp[i] = glmd::twin::half_bits(f[i]);
            be.put(a.k_pe + (size_t) d.qk_rope * p, kp.data(), kp.size() * 2);
            if (a.ik_raw) {
                std::vector<float> k((size_t) d.idx_key);
                be.get(k.data(), a.ik_cache + (size_t) d.idx_key * p, k.size() * 4);
                norm_to_neox(k.data(), d.idx_rope, p, m.inv_idx.data());
                be.put(a.ik_cache + (size_t) d.idx_key * p, k.data(), k.size() * 4);
            }
        }
        return true;
    }
    bool rope_q(const glmd::DsaRopeQ& a0) {
        glmd::DsaRopeQ a = a0;
        if (mut == Mut::RopeOffset) a.rot_off = 0;
        if (!be.rope_q(a)) return false;
        if (mut == Mut::Neox) {
            const int p = cur_p;
            std::vector<float> q((size_t) d.n_head * d.qk_head());
            be.get(q.data(), a.q, q.size() * 4);
            for (int h = 0; h < d.n_head; ++h) norm_to_neox(q.data() + (size_t) h * d.qk_head() + d.qk_nope, d.qk_rope, p, m.inv_q.data());
            be.put(a.q, q.data(), q.size() * 4);
            if (a.iq) {
                std::vector<float> iq((size_t) d.idx_heads * d.idx_key);
                be.get(iq.data(), a.iq, iq.size() * 4);
                for (int h = 0; h < d.idx_heads; ++h) norm_to_neox(iq.data() + (size_t) h * d.idx_key, d.idx_rope, p, m.inv_idx.data());
                be.put(a.iq, iq.data(), iq.size() * 4);
            }
        }
        return true;
    }
    bool score(const glmd::IdxScore& a) { return be.score(a); }
    bool topk(const float* s, int ld, int n_kv, const int* pos, int top_k, int* sel) { return be.topk(s, ld, n_kv, pos, top_k, sel); }
    bool mla2(const glmd::Mla2& a) { return be.mla2(a); }

    /// One layer at position p: x(il, p) in, the mixer output and (full layers) the selected cells out.
    bool step(int il, int p, std::vector<float>& out, std::string& err) {
        const LayerData& Ld = m.L[(size_t) il];
        cur_p = p;
        be.put(p_word, &p, 4);
        be.put(x_buf, m.x(il, p), (size_t) d.n_embd * 4);
        MixerLayer w;
        w.full = Ld.full || mut == Mut::SharedAsFull;
        w.idx_src = mut == Mut::SharedAsFull ? il : Ld.idx_src;
        const LW& lwv = lw[(size_t) il];
        w.q_a_norm = lwv.qa_norm;
        w.kv_a_norm = lwv.kva_norm;
        w.ik_ln_w = lwv.ln_w;
        w.ik_ln_b = lwv.ln_b;
        w.wk_b = lwv.wkb;
        w.wv_b = lwv.wvb;
        w.w_type = w_type;
        MixerCache c = cache[(size_t) il];
        if (!w.full) c.ik = nullptr;
        if (!strata::core::glmdsa::mixer_layer(*this, d, b, c, w, il, p, dense, st, err)) return false;
        be.sync();
        out.resize((size_t) d.n_embd);
        be.get(out.data(), b.mixer, out.size() * 4);
        if (w.full && !dense) {
            cells_h.resize((size_t) d.top_k);
            be.get(cells_h.data(), b.cells, cells_h.size() * 4);
        }
        return true;
    }
};

/// What one run of the stack measured against the reference.
struct RunStats {
    double max_rel = 0, mean_rel = 0;
    int n = 0, sel_checked = 0, sel_equal = 0, sel_near_tie = 0;
    int sel_count_bad = 0;    // st.n_sel_last disagreeing with min(p + 1, top_k)
    std::string err;
};

/// Run every position through every layer (layers in order per token) with harness h; the reference uses the
/// correct reading.  Selection sets are compared on full layers once p + 1 > top_k.
template <class H>
RunStats run_stack(H& h, const Model& m, bool dense_ref) {
    RunStats rs;
    RefState R;
    std::vector<float> got, want;
    std::vector<int> sel_ref;
    double sum = 0;
    for (int p = 0; p < m.T; ++p)
        for (int il = 0; il < (int) m.L.size(); ++il) {
            if (!h.step(il, p, got, rs.err)) return rs;
            ref_layer(m, R, il, p, dense_ref, want, &sel_ref);
            const double e = rel_l2(got.data(), want);
            rs.max_rel = std::max(rs.max_rel, e);
            sum += e;
            ++rs.n;
            const bool full_here = m.L[(size_t) il].full;
            if (full_here && h.st.n_sel_last != strata::core::glmdsa::sel_count(p, m.d.top_k, h.dense)) ++rs.sel_count_bad;
            if (full_here && !h.dense && p + 1 > m.d.top_k) {
                ++rs.sel_checked;
                std::set<int> a(h.cells_h.begin(), h.cells_h.end()), b(sel_ref.begin(), sel_ref.end());
                if (a == b) ++rs.sel_equal;
            }
        }
    rs.mean_rel = sum / std::max(1, rs.n);
    return rs;
}

}  // namespace mixfx
