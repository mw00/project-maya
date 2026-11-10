// src/core/glm_residual.cu - the residual seams of the fast decode path (glm-dsa B2.3); see glm_residual.hpp.
//
// Also Glm5Model::fast_layers_plain: the per-part layer loop for a plain (hc = 1) residual, which Maya's fast_layers
// hands over to on its first line when !mhc().  Per layer:
//   res_read(R, previous FFN out, attn_norm) -> the mixer (fast_dsa) -> res_read(R, mixer, ffn_norm) -> the FFN
// and once at the end of the part res_close(R, last FFN out), so the residual at state_ is complete for the hop or the
// head (the same convention as the mHC path: "the residual lives at state_").  R is updated in place (no ping-pong).
// glm-dsa B10.1: with STRATA_GLM_DSA_GRAPHS=1 the whole loop (and the close) is one captured CUDA graph per part, replayed
// per token (glmdsa::graph_prepare / graph_run, src/core/glm_dsa_mixer.cu); the MoE CPU-lane waits stay device spins
// inside it, as Maya's KDA layer graphs keep them.
#include "glm_residual.hpp"
#include "glm_dsa_fast.hpp"
#include "glm_fast_state.hpp"
#include "glm_memory.hpp"
#include "strata/core/glm_req_knobs.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace strata::core {

namespace gf = strata::kernels::glmf;

namespace glmres {

namespace {
__global__ void add_kernel(float* __restrict__ y, const float* __restrict__ x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}
}  // namespace

void add(float* y, const float* x, int n, cudaStream_t s) {
    if (n <= 0) return;
    add_kernel<<<(n + 255) / 256, 256, 0, s>>>(y, x, n);
}

void res_embed(const Glm5Geometry& g, const float* emb, float* R, cudaStream_t s) {
    if (g.mhc()) {
        gf::embed_streams(emb, R, g.n_embd, s);
        return;
    }
    cudaMemcpyAsync(R, emb, (size_t) g.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, s);
}

void res_read(float* R, const float* block_out, const float* norm_w, float eps, int n_embd, float* x, void* xq,
              cudaStream_t s) {
    gf::rms_q8(R, block_out, norm_w, eps, n_embd, x, xq, s);
}

void res_close(float* R, const float* block_out, int n_embd, cudaStream_t s) { add(R, block_out, n_embd, s); }

void res_head(const Glm5Geometry& g, const float* R, const float* norm_w, float* x, void* xq, cudaStream_t s) {
    if (g.mhc()) {
        gf::head_prep(R, norm_w, g.norm_eps, g.n_embd, x, xq, s);
        return;
    }
    // add == nullptr: rms_q8 only reads h
    gf::rms_q8(const_cast<float*>(R), nullptr, norm_w, g.norm_eps, g.n_embd, x, xq, s);
}

}  // namespace glmres

// GLM_CB_DIR seam dumps, as fast_dump in glm_fast_path.cu (the reference path's names; debug only, synchronous)
static void plain_dump(cudaStream_t s, const std::string& name, const float* dev, int64_t n) {
    static const char* dir = getenv("GLM_CB_DIR");
    if (!dir || !dir[0]) return;
    // (debug: GLM_CB_TOK_ALL="<seam-layer>,..." appends every token's row of those seams to tokall.<name>.f32 - the
    //  prompt path's GLM_CB_PF_ALL counterpart)
    static const char* tok_all = getenv("GLM_CB_TOK_ALL");
    if (tok_all != nullptr && ("," + std::string(tok_all) + ",").find("," + name + ",") == std::string::npos) return;
    cudaStreamSynchronize(s);
    std::vector<float> buf((size_t) n);
    cudaMemcpy(buf.data(), dev, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost);
    if (FILE* f = std::fopen((std::string(dir) + "/" + (tok_all ? "tokall." : "") + name + ".f32").c_str(),
                             tok_all ? "ab" : "wb")) {
        std::fwrite(buf.data(), 4, (size_t) n, f);
        std::fclose(f);
    }
}

bool Glm5Model::fast_layers_plain(int64_t p, bool hop_in, std::string& err) {
    (void) hop_in;   // a hop delivered a closed R at state_: the first layer reads it with no add
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    static const bool dumps = getenv("GLM_CB_DIR") != nullptr;
    bool pf_pending = false;
    float* R = state_;
    const int E = g.n_embd;
    F->kda_graph_on = false;
    // B10.1: the part as one graph?  Only where the body is pure device work: no profiling marks, no dumps (host
    // syncs), no expert prefetch / lookahead (events on the side stream), no hidden-state dump; and not when the
    // request asks for none (the speed plan's knobs: GEN prefetch_n= and graphs=, glm_req_knobs.hpp)
    glmdsa::PartGraph pg;
    if (F->dsa) {
        const char* host_off = glmdsa::host_off_reason(F->prof_on, dumps, glmknob::get(glmknob::PrefetchN, F->max_pf),
                                                        F->n_ahead, getenv("GLM_DUMP_H") != nullptr,
                                                        glmknob::get(glmknob::GraphsOff, 0) != 0);
        uint64_t requests = 0;
        for (int il = l0_; il < l1_; ++il) requests += F->L[(size_t) il].moe ? 1 : 0;
        if (!glmdsa::graph_prepare(*F->dsa, lay_, l0_, l1_, p, F->cpu_plan, R, state_, host_off, requests, s, pg, err,
                                   seq_count()))
            return false;
        pg.key.abl = ablating();   // (--ablate: a projecting conversation's part is another graph)
    }
    if (F->prof_on) F->mark("start");
    const auto body = [&]() -> bool {
    for (int il = l0_; il < l1_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        if (Ly.recr) {
            err = "glm fast: layer " + std::to_string(il) + " is recurrent (KDA) - the plain residual path has none";
            return false;
        }
        // ---- the attention-side read (fused with the previous FFN's residual add)
        if (il != l0_) ablate(F->ffn, 1, s);   // (--ablate: the previous FFN's write loses the direction)
        glmres::res_read(R, il == l0_ ? nullptr : F->ffn, Ly.attn_norm, g.norm_eps, E, F->x, F->xq, s);
        if (F->prof_on) F->mark("res_attn");
        if (dumps) {
            if (il != l0_) plain_dump(s, "l_out-" + std::to_string(il - 1), R, E);
            plain_dump(s, "attn_norm-" + std::to_string(il), F->x, E);
        }
        // the mixer: fast_dsa dispatches glm-dsa to fast_dsa_glm (src/core/glm_dsa_mixer.cu, B4.6), which reads
        // F->dsa->L[il] (glmdsa::GlmDsaLayerW), the glm-dsa arena and this layer's caches through lay_
        if (!fast_dsa(il, p, err)) return false;

        // ---- the FFN-side read (fused with the mixer's residual add)
        ablate(F->mixer, 1, s);   // (--ablate: the mixer's write)
        glmres::res_read(R, F->mixer, Ly.ffn_norm, g.norm_eps, E, F->x, F->xq, s);
        if (F->prof_on) F->mark("res_ffn");
        if (dumps) {
            const std::string Ls = std::to_string(il);
            plain_dump(s, "ffn_inp-" + Ls, R, E);
            plain_dump(s, "ffn_norm-" + Ls, F->x, E);
            plain_dump(s, "mixer-" + Ls, F->mixer, E);
        }

        if (!Ly.moe) {
            // the dense FFN: a BF16 / F32 gate/up (upcast set) reads F->x; a BF16 / F32 down gets its q8_1 input
            // dequantized by glmdsa::mv
            gf::MvJob j[2];
            j[0] = {Ly.ffn_gate.q, F->xq, F->x, F->dg, nullptr, 1.0f, Ly.ffn_gate.type, E, g.n_ff_dense};
            j[1] = {Ly.ffn_up.q, F->xq, F->x, F->du, nullptr, 1.0f, Ly.ffn_up.type, E, g.n_ff_dense};
            if (!glmdsa::mv(F->dsa.get(), j, 2, s)) { err = "glm fast: dense ffn"; return false; }
            if (F->prof_on) F->mark("dense_gu_mv");
            gf::swiglu_q8(F->dg, F->du, g.swiglu_shexp, g.n_ff_dense, F->dhq, s);
            if (F->prof_on) F->mark("dense_swiglu");
            gf::MvJob dn = {Ly.ffn_down.q, F->dhq, nullptr, F->ffn, nullptr, 1.0f, Ly.ffn_down.type, g.n_ff_dense, E};
            if (!glmdsa::mv(F->dsa.get(), &dn, 1, s)) { err = "glm fast: dense ffn down"; return false; }
            if (F->prof_on) F->mark("dense_down_mv");
        } else {
            if (!fast_moe(il, pf_pending, err)) return false;
        }
        if (dumps) plain_dump(s, "ffn_out-" + std::to_string(il), F->ffn, E);
    }
    // the part's last FFN output closes the residual (the hop and the head read R complete)
    if (l1_ > l0_) {
        ablate(F->ffn, 1, s);
        glmres::res_close(R, F->ffn, E, s);
        if (F->prof_on) F->mark("res_close");
        if (dumps) plain_dump(s, "l_out-" + std::to_string(l1_ - 1), R, E);
    }
    return true;
    };
    if (pg.on) {
        if (!glmdsa::graph_run(*F->dsa, pg, p, s, F->expected, body, err)) return false;
    } else if (!body()) {
        return false;
    }
    {
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            err = std::string("glm fast: ") + cudaGetErrorString(e);
            return false;
        }
    }
    return true;
}

// L1-4: the verify window (the one-GPU MTP decode, glm_window.cu's window()) on the plain residual - fast_window_layers
// hands a !mhc() model over on its first line.  Rows 0 .. T - 1 (positions p0 + t) through this part's layers, each
// row's arithmetic the token path's (fast_layers_plain): res_read / res_close per row on the rows' residual W.R[0]
// (T x n_embd), the mixer through fast_dsa_glm_window (its GEMVs batched over the rows, each column the one-column
// GEMV's bit for bit, the per-token steps row by row), the dense FFN's GEMVs batched, the MoE through fast_moe_window
// (each distinct expert once for the rows, the window's CPU lane) - in the two staggered row groups fast_window_layers
// uses, so the CPU lane answers one group while the device runs the other.
bool Glm5Model::fast_window_layers_plain(int64_t p0, int T, std::string& err) {
    FastState* F = fast_;
    auto& W = F->win;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const int E = g.n_embd;
    float* R = W.R[0];
    if (!F->dsa || !glmdsa::win_rows_begin(*F->dsa, p0, T, kWinMax, s, err)) {
        if (err.empty()) err = "glm window: the glm-dsa mixer is not set up on this part";
        return false;
    }
    static const bool prof = getenv("STRATA_GLM_WIN_PROF") != nullptr;
    const auto mark = [&](const char* nm) {
        if (prof) W.mark(nm, s);
    };
    const auto cols = [&](gf::MvJob& j, int n, int64_t xq_s, int64_t xf_s, int64_t y_s) {
        j.ncol = n;
        j.xq_cstride = xq_s;
        j.xf_cstride = xf_s;
        j.y_cstride = y_s;
    };
    // a layer's first half for rows t0 .. t0 + n - 1: the attention-side read, the mixer, the FFN-side read and the FFN
    // up to its routes (moe part 1), or all of it (part 0 / a dense FFN); the second half (part 2): the experts
    const auto pre = [&](int il, int t0, int n, int moe_part, int gid) -> bool {
        const auto& Ly = F->L[(size_t) il];
        if (il != l0_) ablate(W.ffn + (size_t) t0 * E, n, s);   // (--ablate: the window's rows, the bound conversation's)
        for (int t = t0; t < t0 + n; ++t)
            glmres::res_read(R + (size_t) t * E, il == l0_ ? nullptr : W.ffn + (size_t) t * E, Ly.attn_norm, g.norm_eps,
                             E, W.x + (size_t) t * E, (uint8_t*) W.xq + (size_t) t * W.xq_s, s);
        mark("res_attn");
        if (!fast_dsa_glm_window(il, p0, t0, n, W.x, W.xq, W.xq_s, W.mixer, err)) return false;
        mark("dsa");
        ablate(W.mixer + (size_t) t0 * E, n, s);
        for (int t = t0; t < t0 + n; ++t)
            glmres::res_read(R + (size_t) t * E, W.mixer + (size_t) t * E, Ly.ffn_norm, g.norm_eps, E,
                             W.x + (size_t) t * E, (uint8_t*) W.xq + (size_t) t * W.xq_s, s);
        mark("res_ffn");
        if (!Ly.moe) {
            const void* xq0 = (const uint8_t*) W.xq + (size_t) t0 * W.xq_s;
            const float* x0 = W.x + (size_t) t0 * E;
            gf::MvJob j[2];
            j[0] = {Ly.ffn_gate.q, xq0, x0, W.dg + (size_t) t0 * g.n_ff_dense, nullptr, 1.0f, Ly.ffn_gate.type, E,
                    g.n_ff_dense};
            j[1] = {Ly.ffn_up.q, xq0, x0, W.du + (size_t) t0 * g.n_ff_dense, nullptr, 1.0f, Ly.ffn_up.type, E,
                    g.n_ff_dense};
            for (auto& jj : j) cols(jj, n, (int64_t) W.xq_s, E, g.n_ff_dense);
            if (!glmdsa::mv(F->dsa.get(), j, 2, s)) {
                err = "glm window: dense ffn";
                return false;
            }
            for (int t = t0; t < t0 + n; ++t)
                gf::swiglu_q8(W.dg + (size_t) t * g.n_ff_dense, W.du + (size_t) t * g.n_ff_dense, g.swiglu_shexp,
                              g.n_ff_dense, (uint8_t*) W.dhq + (size_t) t * W.dhq_s, s);
            // (a BF16 / F32 down dequantizes its q8_1 input one row at a time: glmdsa::mv's fallback takes one column)
            const bool by_row = glmdsa::float_type(Ly.ffn_down.type);
            for (int t = by_row ? t0 : 0; t < (by_row ? t0 + n : 1); ++t) {
                const int r0 = by_row ? t : t0;
                gf::MvJob dn = {Ly.ffn_down.q, (const uint8_t*) W.dhq + (size_t) r0 * W.dhq_s, nullptr,
                                W.ffn + (size_t) r0 * E, nullptr, 1.0f, Ly.ffn_down.type, g.n_ff_dense, E};
                cols(dn, by_row ? 1 : n, (int64_t) W.dhq_s, 0, E);
                if (!glmdsa::mv(F->dsa.get(), &dn, 1, s)) {
                    err = "glm window: dense ffn down";
                    return false;
                }
            }
            mark("dense_ffn");
            return true;
        }
        return fast_moe_window(il, t0, n, moe_part, gid, err);
    };
    const auto post = [&](int il, int t0, int n, int gid) -> bool {
        if (!F->L[(size_t) il].moe) return true;
        return fast_moe_window(il, t0, n, 2, gid, err);
    };
    mark("start");
    // (fast_window_layers' staggered row groups and their switches)
    static const bool stagger_env = [] {
        const char* v = getenv("STRATA_GLM_WIN_STAGGER");
        return v == nullptr || std::atoi(v) != 0;
    }();
    static const bool batch_env = glmfast::off_by_default_on_amd("STRATA_GLM_WIN_BATCH");
    const bool stagger = stagger_env && batch_env && T >= 2 && !W.per_token && W.mplan_ptr != nullptr;
    if (stagger) {
        const int ka = (T + 1) / 2, kb = T - ka;
        if (!pre(l0_, 0, ka, 1, 0) || !pre(l0_, ka, kb, 1, 1)) return false;
        for (int il = l0_; il < l1_; ++il) {
            if (!post(il, 0, ka, 0)) return false;
            if (il + 1 < l1_ && !pre(il + 1, 0, ka, 1, 0)) return false;
            if (!post(il, ka, kb, 1)) return false;
            if (il + 1 < l1_ && !pre(il + 1, ka, kb, 1, 1)) return false;
        }
    } else {
        for (int il = l0_; il < l1_; ++il)
            if (!pre(il, 0, T, 0, 0)) return false;
    }
    // each row's last FFN output closes its residual (the hop and the head read the rows complete)
    ablate(W.ffn, T, s);
    for (int t = 0; t < T; ++t) glmres::res_close(R + (size_t) t * E, W.ffn + (size_t) t * E, E, s);
    W.out = 0;
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("glm window: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

}  // namespace strata::core
