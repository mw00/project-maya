// src/core/glm_dsa_fast.cu - the glm-dsa fast binder and arena (glm-dsa B2.4).
//
// fast_setup's weight loop hands each glm-dsa layer to Glm5Model::fast_bind_dsa (one hook line), which fills
//   * FastState::Layer, the contract the shared MoE / dense / NextN code reads (norms, attn_output, router, shared
//     expert, dense FFN, expert formats, eh_proj and its norms), and
//   * glmdsa::GlmDsaLayerW, what the glm-dsa attention mixer reads (q_a, q_a_norm, q_b, kv_a with its 576 rows,
//     kv_a_norm, BF16 (or native Q8_0, B10.7) k_b / v_b, the indexer on full layers only; no compressor).
// Every projection in the B1.6 upcast set resolves q_or_b16-style (glm_dsa_fast_plan.cpp pick_mv: the native slot,
// else the BF16 copy in w16_, else F32), the router ffn_gate_inp from w_ (F32, G23) or w16_, and the router GEMV jobs
// take their type from Layer::router_type.  fast_dsa_arena then carves the glm-dsa arena (q 16384, kv_raw 576, score
// max_ctx, cells 2048, ...) and logs what was bound.
#include "glm_dsa_fast.hpp"
#include "glm_fast_state.hpp"

#include "strata/core/glm_env.hpp"
#include "strata/kernels/glmd_graph.hpp"

#include <cstdio>
#include <string>

namespace strata::core {

namespace gf = strata::kernels::glmf;
static_assert(glmdsa::kTypeF32 == gf::kTypeF32 && glmdsa::kTypeBF16 == gf::kTypeBF16, "glmdsa type ids mirror gf's");
static_assert(glmdsa::kMaxMvJobs == gf::kMaxMvJobs, "one fallback row per mv job");

namespace glmdsa {

GlmDsaFast::~GlmDsaFast() {
    if (arena != nullptr) {
        int cur = 0;
        cudaGetDevice(&cur);
        if (cur != dev) cudaSetDevice(dev);
        cudaFree(arena);
        glmd::graph_generation_bump("glm-dsa arena (freed)");   // B15.6: no graph may replay over it
        if (cur != dev) cudaSetDevice(cur);
    }
}

namespace {
struct Q8Block {
    uint16_t d, sum;   // FP16 bits (ggml's half2 ds)
    int8_t qs[32];
};
static_assert(sizeof(Q8Block) == kQ8Bytes, "q8_1 block is 36 bytes");

// FP16 bits -> float without cuda_fp16.h (the file also builds as HIP)
__device__ __forceinline__ float h2f(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu, man = h & 0x3ffu;
    if (exp == 0) {
        const float v = (float) man * 5.9604644775390625e-08f;   // 2^-24
        return sign ? -v : v;
    }
    const uint32_t bits = exp == 31 ? (sign | 0x7f800000u | (man << 13)) : (sign | ((exp + 112u) << 23) | (man << 13));
    return __uint_as_float(bits);
}

__global__ void q8_1_to_f32_kernel(const Q8Block* __restrict__ xq, int n, float* __restrict__ y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Q8Block& b = xq[i / 32];
    y[i] = h2f(b.d) * (float) b.qs[i % 32];
}
}  // namespace

void q8_1_to_f32(const void* xq, int n, float* y, cudaStream_t s) {
    if (n <= 0 || n % 32 != 0) {
        std::fprintf(stderr, "glm-dsa q8_1_to_f32: n %d is not a positive multiple of 32\n", n);
        return;
    }
    q8_1_to_f32_kernel<<<(n + 255) / 256, 256, 0, s>>>((const Q8Block*) xq, n, y);
}

// L1-4: glm-dsa's GEMVs through gf::mv_dsa (bit for bit gf::mv's outputs, more loads in flight) where the device's
// default has it (glm_fast.hpp: kMvDsaAuto); STRATA_GLMD_MV_DSA=0: gf::mv itself (A/Bs), =1: on any device.  Only a
// glm-dsa engine has a GlmDsaFast (fast_bind_dsa / fast_dsa_arena): Flash (ext == nullptr: its NextN eh_proj in
// fast_mtp and mtp_cache_rows, its dense FFN) calls gf::mv exactly as v1.1.56 did - the same kernels on every device.
static bool run_mv(const GlmDsaFast* ext, const gf::MvJob* jobs, int n, cudaStream_t s) {
    if (ext == nullptr) return gf::mv(jobs, n, s);
    static const int cfg = [] {
        const int v = strata::glmenv::env_switch("STRATA_GLMD_MV_DSA", strata::glmenv::kAuto, true);
        return v == strata::glmenv::kOff ? gf::kMvDsaOff : v == strata::glmenv::kOn ? gf::kMvDsaOn : gf::kMvDsaAuto;
    }();
    return gf::mv_dsa(jobs, n, s, cfg);
}

bool mv(GlmDsaFast* ext, const gf::MvJob* jobs, int n, cudaStream_t s) {
    if (n <= 0 || n > gf::kMaxMvJobs) return false;
    bool need = false;
    for (int i = 0; i < n; ++i)
        if (float_type(jobs[i].type) && jobs[i].xf == nullptr) need = true;
    if (!need) return run_mv(ext, jobs, n, s);
    if (ext == nullptr || ext->act == nullptr) {
        std::fprintf(stderr, "glm-dsa mv: a BF16/F32 weight with only a q8_1 input and no fallback scratch\n");
        return false;
    }
    gf::MvJob j[gf::kMaxMvJobs];
    const void* seen[gf::kMaxMvJobs] = {};
    int n_seen = 0;
    for (int i = 0; i < n; ++i) {
        j[i] = jobs[i];
        if (!float_type(j[i].type) || j[i].xf != nullptr) continue;
        if (j[i].xq == nullptr || j[i].n_in <= 0 || j[i].n_in > ext->plan.act_floats ||
            j[i].n_in % 32 != 0) {
            std::fprintf(stderr, "glm-dsa mv: job %d (type %d, n_in %d) has no usable input for the fallback\n", i,
                         j[i].type, j[i].n_in);
            return false;
        }
        int r = 0;
        while (r < n_seen && seen[r] != j[i].xq) ++r;
        float* row = ext->act + (size_t) r * (size_t) ext->plan.act_floats;
        if (r == n_seen) {
            seen[n_seen++] = j[i].xq;
            q8_1_to_f32(j[i].xq, j[i].n_in, row, s);
        }
        j[i].xf = row;
    }
    return run_mv(ext, j, n, s);
}

bool shexp_down_apart(GlmDsaFast* ext, const WSlot& sh_down, const void* sh_hq, int n_ff_sh, int n_embd, float* sh_out,
                      cudaStream_t s, bool& apart) {
    apart = false;
    if (sh_down.q == nullptr || shexp_fused_ok(sh_down.type)) return true;
    gf::MvJob d = {sh_down.q, sh_hq, nullptr, sh_out, nullptr, 1.0f, sh_down.type, n_ff_sh, n_embd};
    if (!mv(ext, &d, 1, s)) return false;
    apart = true;
    return true;
}

}  // namespace glmdsa

// ---------------------------------------------------------------- the binder (one call per glm-dsa layer)
void Glm5Model::fast_bind_dsa(int il, std::string& missing) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    if (!F->dsa) {
        F->dsa = std::make_shared<glmdsa::GlmDsaFast>();
        F->dsa->dev = dev_;
        F->dsa->L.assign((size_t) g.n_layers + 1, glmdsa::GlmDsaLayerW{});
    }
    glmdsa::GlmDsaFast& X = *F->dsa;
    auto& Ly = F->L[(size_t) il];
    auto& W = X.L[(size_t) il];
    const bool is_mtp = il == mtp_il_;
    const std::string P = "blk." + std::to_string(il) + ".";

    glmdsa::Tables T;
    T.native = [&](const std::string& n, int& type, const void*& q, const void*& f) {
        const auto it = ws_map_.find(n);
        if (it == ws_map_.end()) return false;
        type = it->second.type;
        q = it->second.q;
        f = it->second.f32;
        return true;
    };
    T.bf16 = [&](const std::string& n) -> const void* {
        const auto it = w16_.find(n);
        return it == w16_.end() ? nullptr : (const void*) it->second;
    };
    T.f32 = [&](const std::string& n) -> const void* {
        const auto it = w_.find(n);
        return it == w_.end() ? nullptr : (const void*) it->second;
    };
    T.mv_ok = [](int t) { return gf::mv_supported_ext(t); };

    const auto slot = [&](const glmdsa::Pick& p, const char* n, int64_t n_in, int64_t n_out) -> WSlot {
        if (p.src == glmdsa::Src::Missing) {
            missing += P + n + " (" + p.why + ") ";
            return WSlot{};
        }
        if (p.src == glmdsa::Src::Bf16) ++X.n_upcast_bf16;
        if (p.src == glmdsa::Src::F32) ++X.n_upcast_f32;
        WSlot w;
        w.type = p.type;
        w.q = p.w;
        w.f32 = p.type == gf::kTypeF32 ? (const float*) p.w : nullptr;
        w.n_in = n_in;
        w.n_out = n_out;
        return w;
    };
    const auto mvw = [&](const char* n, int64_t n_in, int64_t n_out) { return slot(glmdsa::pick_mv(T, P + n), n, n_in, n_out); };
    const auto flt = [&](const char* n, int64_t n_in, int64_t n_out) -> WSlot {
        const glmdsa::Pick p = glmdsa::pick_float(T, P + n);
        WSlot w = slot(p, n, n_in, n_out);
        // these tensors are float by design: not an upcast
        if (p.src == glmdsa::Src::Bf16) --X.n_upcast_bf16;
        if (p.src == glmdsa::Src::F32) --X.n_upcast_f32;
        return w;
    };
    const auto f32 = [&](const char* n) -> const float* {
        const auto it = w_.find(P + n);
        if (it == w_.end()) { missing += P + n + " (f32) "; return nullptr; }
        return it->second;
    };
    // B10.7: W_kb / W_vb through pick_headwise: a native Q8_0 slot (an STRATA_PACK_NATIVE32 pack) is served as it is
    // (the mixer passes the slot type as Mla2 / FusedMla w_type); a default pack holds only the BF16 copy, unchanged
    const auto hw = [&](const char* n, int64_t n_in, int64_t n_out) -> WSlot {
        const glmdsa::Pick p = glmdsa::pick_headwise(T, P + n);
        WSlot w = slot(p, n, n_in, n_out);
        if (p.src == glmdsa::Src::Bf16) --X.n_upcast_bf16;   // float by design: not an upcast
        if (p.src == glmdsa::Src::F32) --X.n_upcast_f32;
        return w;
    };
    const auto b16_of = [](const WSlot& w) { return w.type == gf::kTypeBF16 ? (const uint16_t*) w.q : nullptr; };

    const int64_t E = g.n_embd, QL = g.q_lora, H = g.n_head;
    W.bound = true;
    W.mtp = is_mtp;
    W.full = g.idx_full(il);
    W.idx_src = il;
    if (!W.full) {
        W.idx_src = -1;
        for (int k = il - 1; k >= 0; --k)
            if (g.idx_full(k)) { W.idx_src = k; break; }
        if (W.idx_src < l0_) missing += P + "indexer (no full layer before it on this part) ";
    }

    // ---- the FastState::Layer contract (the shared MoE / dense / NextN code)
    Ly.recr = false;
    Ly.moe = il >= g.dense_lead;
    Ly.mtp = is_mtp;
    if (is_mtp) {
        Ly.eh = mvw("nextn.eh_proj.weight", 2 * E, E);
        Ly.enorm = f32("nextn.enorm.weight");
        Ly.hnorm = f32("nextn.hnorm.weight");
        Ly.shnorm = f32("nextn.shared_head_norm.weight");   // B1.8: output_norm uploaded under this name when absent
    }
    // no hyper-connections: hc_* stay null (the plain residual path never reads them)
    Ly.attn_norm = f32("attn_norm.weight");
    Ly.ffn_norm = f32("ffn_norm.weight");

    // ---- the attention (GlmDsaLayerW, mirrored into Layer where the Flash field types fit)
    W.attn_norm = Ly.attn_norm;
    W.q_a = mvw("attn_q_a.weight", E, QL);
    W.q_a_norm = f32("attn_q_a_norm.weight");
    W.q_b = mvw("attn_q_b.weight", QL, H * g.qk_head());
    W.kv_a = mvw("attn_kv_a_mqa.weight", E, g.kv_row());
    W.kv_a_norm = f32("attn_kv_a_norm.weight");
    W.k_b = hw("attn_k_b.weight", g.qk_nope, (int64_t) g.kv_lora * H);
    W.v_b = hw("attn_v_b.weight", g.kv_lora, (int64_t) g.v_head * H);
    W.out = mvw("attn_output.weight", H * g.v_head, E);
    if (W.full) {
        W.idx_k = flt("indexer.attn_k.weight", E, g.idx_key);
        W.idx_q_b = flt("indexer.attn_q_b.weight", QL, (int64_t) g.idx_heads * g.idx_key);
        W.idx_proj = flt("indexer.proj.weight", E, g.idx_heads);
        W.idx_k_norm_w = f32("indexer.k_norm.weight");
        W.idx_k_norm_b = f32("indexer.k_norm.bias");
    }
    Ly.q_a = W.q_a;
    Ly.q_b = W.q_b;
    Ly.kv_a = W.kv_a;
    Ly.q_a_norm = W.q_a_norm;
    Ly.kv_a_norm = W.kv_a_norm;
    Ly.k_b = b16_of(W.k_b);
    Ly.v_b = b16_of(W.v_b);
    Ly.idx_k = b16_of(W.idx_k);
    Ly.idx_q_b = b16_of(W.idx_q_b);
    Ly.idx_proj = b16_of(W.idx_proj);
    Ly.k_norm_w = W.idx_k_norm_w;
    Ly.k_norm_b = W.idx_k_norm_b;
    Ly.out = W.out;

    // ---- the FFN
    if (Ly.moe) {
        const glmdsa::Pick r = glmdsa::pick_router(T, P + "ffn_gate_inp.weight");
        if (r.src == glmdsa::Src::Missing) {
            missing += P + "ffn_gate_inp.weight (" + r.why + ") ";
        } else {
            Ly.router_w = r.w;
            Ly.router_type = r.type;
            Ly.router = r.type == gf::kTypeBF16 ? (const uint16_t*) r.w : nullptr;   // the BF16-only prompt path
            if (r.type == gf::kTypeF32) ++X.n_router_f32;
            else ++X.n_router_bf16;
        }
        Ly.router_bias = f32("exp_probs_b.bias");
        const int64_t FFs = (int64_t) g.n_ff_exp * g.n_shared;
        if (g.n_shared > 0) {
            Ly.sh_gate = mvw("ffn_gate_shexp.weight", E, FFs);
            Ly.sh_up = mvw("ffn_up_shexp.weight", E, FFs);
            Ly.sh_down = mvw("ffn_down_shexp.weight", FFs, E);
        }
        const auto& nl = pack_layers_[(size_t) il];
        if (nl.layer < 0) {
            missing += P + "native experts ";
        } else {
            Ly.gu_type = nl.fmt.gu_type;
            Ly.d_type = nl.fmt.d_type;
            Ly.gu_bytes = (size_t) nl.fmt.gu_row * (size_t) g.n_ff_exp;
            Ly.dn_bytes = (size_t) nl.fmt.d_row * (size_t) g.n_embd;
            Ly.down_off = 2 * Ly.gu_bytes;
            Ly.blob = 2 * Ly.gu_bytes + Ly.dn_bytes;
            if (!gf::moe_supported(Ly.gu_type) || !gf::moe_supported(Ly.d_type) ||
                gf::row_bytes(Ly.gu_type, g.n_embd) != nl.fmt.gu_row ||
                gf::row_bytes(Ly.d_type, g.n_ff_exp) != nl.fmt.d_row)
                missing += P + "expert types " + std::to_string(Ly.gu_type) + "/" + std::to_string(Ly.d_type) + " ";
        }
    } else {
        Ly.ffn_gate = mvw("ffn_gate.weight", E, g.n_ff_dense);
        Ly.ffn_up = mvw("ffn_up.weight", E, g.n_ff_dense);
        Ly.ffn_down = mvw("ffn_down.weight", g.n_ff_dense, E);
    }
}

// ---------------------------------------------------------------- the glm-dsa arena (after the shared one)
bool Glm5Model::fast_dsa_arena(std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    if (!F->dsa) {   // a part with no layers of its own still gets the (empty) side state
        F->dsa = std::make_shared<glmdsa::GlmDsaFast>();
        F->dsa->dev = dev_;
        F->dsa->L.assign((size_t) g.n_layers + 1, glmdsa::GlmDsaLayerW{});
    }
    glmdsa::GlmDsaFast& X = *F->dsa;
    glmdsa::ArenaDims d;
    d.n_embd = g.n_embd;
    d.n_head = g.n_head;
    d.q_lora = g.q_lora;
    d.kv_lora = g.kv_lora;
    d.qk_nope = g.qk_nope;
    d.qk_rope = g.qk_rope;
    d.v_head = g.v_head;
    d.idx_heads = g.idx_heads;
    d.idx_key = g.idx_key;
    d.idx_top_k = g.idx_top_k;
    d.max_ctx = max_ctx_;
    d.n_ff_dense = g.n_ff_dense;
    d.n_ff_sh = (int64_t) g.n_ff_exp * g.n_shared;
    if (!glmdsa::arena_plan(d, X.plan, err)) return false;
    if (cudaMalloc(&X.arena, X.plan.total) != cudaSuccess) {
        cudaGetLastError();
        X.arena = nullptr;
        err = "glm-dsa fast: the glm-dsa arena (" + std::to_string(X.plan.total) + " B) did not allocate";
        return false;
    }
    cudaMemset(X.arena, 0, X.plan.total);
    glmd::graph_generation_bump("glm-dsa arena (DSA scores, mixer rows)");   // B15.6: re-capture every decode graph
    uint8_t* base = (uint8_t*) X.arena;
    const auto at = [&](int b) { return (void*) (base + X.plan.off[b]); };
    X.q = (float*) at(glmdsa::kQ);
    X.kv_raw = (float*) at(glmdsa::kKvRaw);
    X.q_abs = (float*) at(glmdsa::kQAbs);
    X.score = (float*) at(glmdsa::kScore);
    X.cells = (int*) at(glmdsa::kCells);
    X.qr_raw = (float*) at(glmdsa::kQrRaw);
    X.qr = (float*) at(glmdsa::kQr);
    X.qr_q = at(glmdsa::kQrQ);
    X.ik_raw = (float*) at(glmdsa::kIkRaw);
    X.iq = (float*) at(glmdsa::kIq);
    X.iw = (float*) at(glmdsa::kIw);
    X.attn_out = (float*) at(glmdsa::kAttnOut);
    X.attn_out_q = at(glmdsa::kAttnOutQ);
    X.act = (float*) at(glmdsa::kAct);
    if (!glmdsa::mixer_setup(X, g, max_ctx_, err)) return false;   // B4.6: the attention mixer's state
    std::fprintf(stderr,
                 "glm-dsa fast: CUDA%d arena %.2f MB (score %lld positions); router ffn_gate_inp F32 on %d layers, BF16 "
                 "on %d; upcast weights served from BF16 copies %d, F32 %d\n",
                 dev_, X.plan.total / 1048576.0, (long long) max_ctx_, X.n_router_f32, X.n_router_bf16, X.n_upcast_bf16,
                 X.n_upcast_f32);
    return true;
}

}  // namespace strata::core
