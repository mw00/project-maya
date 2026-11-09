// src/kernels/cuda/glmd_mv_ext.cuh - glm-dsa B10.6 / B10.7 extensions of glm_fast.cu's dense GEMV (gf::mv).
//
// Included by glm_fast.cu INSIDE namespace strata::kernels::glmf::{anonymous}, right after mv_kernel_t (one-line hook,
// STRATA-EDITS.txt), so it uses glm_fast.cu's F<T>, row_dot<T>, rbytes<T>, MvBatch, MV_ROWS and warp_sum.  No
// #include of system headers here.  More one-line hooks call it: mv_type_ok (default: ext_mv_type_ok), row_bytes
// (default: ext_row_bytes), mv() (ext_mv_shape_ok after the type check, ext_mv right after `done` is declared),
// mv_nt() (ext_mv_shape_ok) and moe_gate_up() (ext_shared_down_ok).
//
//   B10.7  new mv types, always on (Flash never meets them: they were refused before):
//          Q4_0 (2), Q4_1 (3), Q5_0 (6), Q5_1 (7), IQ4_NL (20) through glmd_mv_dots.cuh's llama.cpp dots, one
//          warp per row (the typed kernel mv_kernel_t<T>), and F16 (1) with the f32 input or the q8_1 one.
//   B10.6  behind switches (read once; default off until measured on the test PC):
//          STRATA_GLMD_MV_NW2=1    K-quant rows (Q2_K..Q6_K) at batch 1 split over TWO warps (llama.cpp 42d95816: the
//                                  sm_70 nwarps table); changes the summation order (the two warps' sums are added
//                                  last), so it is a numerics change, not only a schedule;
//          STRATA_GLMD_MV_TYPED=1  typed launches for IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / Q2_K / Q3_K (the generic
//                                  kernel's switch costs registers); same row_dot<T>, so the same arithmetic.
#pragma once

#include "glmd_mv_dots.cuh"

static_assert(sizeof(glmd_mvq::BQ4_0) == sizeof(block_q4_0) && sizeof(glmd_mvq::BQ4_1) == sizeof(block_q4_1) &&
                  sizeof(glmd_mvq::BQ5_0) == sizeof(block_q5_0) && sizeof(glmd_mvq::BQ5_1) == sizeof(block_q5_1) &&
                  sizeof(glmd_mvq::BIQ4_NL) == sizeof(block_iq4_nl) && sizeof(glmd_mvq::BQ8_1) == sizeof(block_q8_1),
              "glmd_mv_dots blocks are ggml's");

#define GLMD_MV_F(T, BLOCK, FN)                                                                                   \
    template<> struct F<T> { static constexpr int qk = 32, ipb = 2, step = 2, bytes = sizeof(BLOCK);             \
        __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) {                      \
            return glmd_mvq::FN(v, (const glmd_mvq::BQ8_1*) y, kbx, iqs); } };
GLMD_MV_F(2, block_q4_0, dot_q4_0)
GLMD_MV_F(3, block_q4_1, dot_q4_1)
GLMD_MV_F(6, block_q5_0, dot_q5_0)
GLMD_MV_F(7, block_q5_1, dot_q5_1)
GLMD_MV_F(20, block_iq4_nl, dot_iq4_nl)
#undef GLMD_MV_F

int ext_mv_type_ok(int t) { return glmd_mvq::type_ok(t) ? 1 : 0; }
size_t ext_row_bytes(int t, int64_t n_in) { return glmd_mvq::row_bytes(t, (long long) n_in); }

// glm_fast.cu's launch checks (defined further down in this anonymous namespace)
void launch_check(const char* what);
void shape_refused(const char* what, int n);

// Fail closed where Maya's kernels meet the six types above: only ext_mv's one-column, one-activation kernels serve
// them.  Maya's several-column kernels (mv() jobs with ncol > 1: a DFlash verify window, a drafter's block; the last
// resort mv_kernel_cols -> any_row_dot) and mv_nt's (--parallel rows: mv_kernel_nt -> any_row_dot_nt) have no case
// for them and would write zeros, so mv() (nt 1) and mv_nt() (nt > 1) refuse such a job (one hook line each).
bool ext_mv_shape_ok(const MvJob* jobs, int n, int nt) {
    for (int i = 0; i < n; ++i)
        if (glmd_mvq::type_ok(jobs[i].type) && (nt > 1 || jobs[i].ncol != 1)) {
            std::fprintf(stderr, "glm_fast mv: type %d with %d columns x %d activations unsupported (one only)\n",
                         jobs[i].type, jobs[i].ncol, nt);
            return false;
        }
    return true;
}

// moe_gate_up's fused shared-expert down dots its weight with any_row_dot against the q8_1 shared activation: these
// are the types it has a case for (F32 / BF16 would read a null float input, the six above a 0).  Any other type is
// refused before the launch (counted like a failed launch).  glm-dsa runs every other type apart first
// (glmdsa::shexp_down_apart, whose glmdsa::shexp_fused_ok mirrors this list).
bool ext_shared_down_ok(int t) {
    switch (t) {
        case 8: case 10: case 11: case 12: case 13: case 14: case 16: case 17: case 18: case 19: case 21: case 22:
        case 23: case 29: return true;
        default: shape_refused("moe_gate_up shared down type", t); return false;
    }
}

// F16 rows: one warp per row, the activation in f32 (xf) or q8_1 (xq); STRATA_GRID_CONSTANT is glm_fast.cu's (Pascal)
__global__ void __launch_bounds__(256) mv_f16_kernel(const STRATA_GRID_CONSTANT MvBatch b) {
    if (b.skip != nullptr && *b.skip) return;   // set_skip (Maya v1.1.3), as mv_kernel_t
    const int bid = blockIdx.x;
    int ji = 0;
    while (ji < b.n - 1 && bid >= b.blk_end[ji]) ++ji;
    const MvJob& J = b.j[ji];
    const int blk0 = ji ? b.blk_end[ji - 1] : 0;
    const int row = (bid - blk0) * MV_ROWS + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= J.n_out) return;
    const float s = warp_sum(glmd_mvq::f16_lane((const uint16_t*) J.w + (size_t) row * J.n_in, J.xf,
                                                (const glmd_mvq::BQ8_1*) J.xq, J.n_in, lane));
    if (lane == 0) J.y[row] = mv_out(J.alpha, s, J.bias ? J.bias[row] : 0.0f);
}

// B10.6: a row over two warps (4 rows per 256-thread block); the warps' sums are added in order (warp 0 + warp 1)
constexpr int MV_NW2_ROWS = 4;
template<int T>
__global__ void __launch_bounds__(256) mv_kernel_nw2(const STRATA_GRID_CONSTANT MvBatch b) {
    if (b.skip != nullptr && *b.skip) return;   // (uniform per launch: no thread reaches the barrier)
    __shared__ float s_p[8];
    using Fm = F<T>;
    const int bid = blockIdx.x;
    int ji = 0;
    while (ji < b.n - 1 && bid >= b.blk_end[ji]) ++ji;
    const MvJob& J = b.j[ji];
    const int blk0 = ji ? b.blk_end[ji - 1] : 0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, part = warp & 1;
    const int row = (bid - blk0) * MV_NW2_ROWS + (warp >> 1);
    float s = 0.0f;
    if (row < J.n_out) {
        const uint8_t* w = (const uint8_t*) J.w + (size_t) row * rbytes<T>(J.n_in);
        const block_q8_1* x = (const block_q8_1*) J.xq;
        const int nb = J.n_in / Fm::qk;
        for (int k = 32 * part + lane; k < nb * Fm::ipb; k += 64) {
            const int kbx = k / Fm::ipb, iqs = Fm::step * (k % Fm::ipb);
            s += Fm::dot(w, x + kbx * (Fm::qk / 32), kbx, iqs);
        }
    }
    s = warp_sum(s);
    if (lane == 0) s_p[warp] = s;
    __syncthreads();
    if (part == 0 && lane == 0 && row < J.n_out) J.y[row] = mv_out(J.alpha, s_p[warp] + s_p[warp + 1], J.bias ? J.bias[row] : 0.0f);
}

// the B10.6 switches: bit 0 typed launches, bit 1 two warps per K-quant row
int ext_mv_cfg() {
    static const int cfg = [] {
        // (strata/core/glm_env.hpp: included at glm_fast.cu's top through glmd_random_route.hpp)
        return (::strata::glmenv::env_flag("STRATA_GLMD_MV_TYPED", false) ? 1 : 0) |
               (::strata::glmenv::env_flag("STRATA_GLMD_MV_NW2", false) ? 2 : 0);
    }();
    return cfg;
}

template<int T, int ROWS, typename Launch>
void ext_batch(const MvJob* jobs, int n, bool* done, Launch launch) {
    MvBatch b{};
    int acc = 0, m = 0;
    for (int i = 0; i < n; ++i)
        if (!done[i] && jobs[i].type == T && jobs[i].ncol == 1) {   // (several columns: Maya's own kernels)
            b.j[m] = jobs[i];
            acc += (jobs[i].n_out + ROWS - 1) / ROWS;
            b.blk_end[m++] = acc;
            done[i] = true;
        }
    if (m == 0) return;
    b.n = m;
    b.skip = g_skip;
    launch(b, acc);
}

// Runs (and marks done) the jobs this file serves; the rest stay for gf::mv's own kernels.
void ext_mv(const MvJob* jobs, int n, bool* done, cudaStream_t s) {
    int before = 0;
    for (int i = 0; i < n; ++i) before += done[i] ? 1 : 0;
#define GLMD_T(T) ext_batch<T, MV_ROWS>(jobs, n, done, [&](const MvBatch& b, int g) { mv_kernel_t<T><<<g, 256, 0, s>>>(b); });
    GLMD_T(2) GLMD_T(3) GLMD_T(6) GLMD_T(7) GLMD_T(20)
    ext_batch<1, MV_ROWS>(jobs, n, done, [&](const MvBatch& b, int g) { mv_f16_kernel<<<g, 256, 0, s>>>(b); });
    const int cfg = ext_mv_cfg();
    if (cfg & 2) {
#define GLMD_NW2(T) ext_batch<T, MV_NW2_ROWS>(jobs, n, done, [&](const MvBatch& b, int g) { mv_kernel_nw2<T><<<g, 256, 0, s>>>(b); });
        GLMD_NW2(10) GLMD_NW2(11) GLMD_NW2(12) GLMD_NW2(13) GLMD_NW2(14)
#undef GLMD_NW2
    }
    if (cfg & 1) {
        GLMD_T(16) GLMD_T(17) GLMD_T(22) GLMD_T(18) GLMD_T(10) GLMD_T(11)
    }
#undef GLMD_T
    // checked here: when these were the batch's only jobs, mv() returns before its own launch_check("mv")
    int after = 0;
    for (int i = 0; i < n; ++i) after += done[i] ? 1 : 0;
    if (after != before) launch_check("mv (glmd ext)");
}

// ---------------------------------------------------------------- glm-dsa L1-4: the decode mixer's GEMVs (mv_dsa)
// The one-column jobs of the types GLM-5.3's decode reads (Q5_K, Q8_0, Q4_K, Q6_K, BF16, F32) - each row's sum bit for
// bit mv_kernel_t<T>'s / mv_kernel's: every lane adds its k steps in row_dot's order (DecW is row_dot's dot split in
// two, bit for bit; the BF16 / F32 steps are row_dot_bf16's / row_dot_f32's expressions), then warp_sum.  What changes
// is only how many loads are in flight: a warp takes NR rows (a k step's loads for all of them issued before the first
// dot, its q8_1 values read once for them), and one launch takes every job of the batch (a layer's input projections:
// Q5_K q_a, Q8_0 kv_a, the indexer's BF16 / F32 rows - the small ones no longer wait behind the big one's launch).
// Which jobs and NR: ext_mv_dsa's policy below.  Measured on a V100 (glmd_dec_bench): see HANDOFF "L1-4".
template<> struct DecW<13> {   // dot_q5_K split in two (DecW<12> with the fifth bit)
    using X = DecW<12>::X;
    int v0[2], v1[2];
    int sc[2], m[2];
    float2 dm;
    __device__ __forceinline__ void load(const void* vbq, int kbx, int iqs) {
        const block_q5_K* bq5 = (const block_q5_K*) vbq + kbx;
        const int bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = (const int*) (bq5->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
        const int* qh = (const int*) (bq5->qh + 4 * ((iqs / 2) % 4));
        const int vl0 = ql[0], vl1 = ql[4];
        const int vh0 = qh[0] >> bq8_offset, vh1 = qh[4] >> bq8_offset;
        const uint16_t* scales = (const uint16_t*) bq5->scales;
        const int j = bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm], s2 = scales[jm + 2], s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        uint16_t aux[2];
        aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        const uint8_t* scb = (const uint8_t*) aux;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            v0[i] = ((vl0 >> (4 * i)) & 0x0f0f0f0f) | (((vh0 >> i) << 4) & 0x10101010);
            v1[i] = ((vl1 >> (4 * i)) & 0x0f0f0f0f) | (((vh1 >> i) << 4) & 0x10101010);
            sc[i] = scb[i];
            m[i] = scb[2 + i];
        }
        dm = __half22float2(bq5->dm);
    }
    __device__ __forceinline__ static X load_x(const block_q8_1* bq8_1, int iqs) { return DecW<12>::load_x(bq8_1, iqs); }
    __device__ __forceinline__ float dot(const X& x) const {
        float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int dot1 = __dp4a(v0[i], x.u[2 * i], __dp4a(v1[i], x.u[2 * i + 1], 0));
            sumf_d += x.d8[i] * (dot1 * sc[i]);
            sumf_m += x.d8[i] * (x.su[i] * m[i]);
        }
        return dm.x * sumf_d - dm.y * sumf_m;
    }
};

// rows row0 .. row0 + nr - 1 of a quantized job: s[r] = lane's partial of row r (row_dot's order)
template<int T, int NR>
__device__ __forceinline__ void dsa_rows_q(const MvJob& J, int row0, int nr, int lane, float (&s)[NR]) {
    using Fm = F<T>;
    using D = DecW<T>;
    const size_t rb = rbytes<T>(J.n_in);
    const uint8_t* w0 = (const uint8_t*) J.w + (size_t) row0 * rb;
    const block_q8_1* x = (const block_q8_1*) J.xq;
    const int nk = (J.n_in / Fm::qk) * Fm::ipb;
    for (int k = lane; k < nk; k += 32) {
        const int kbx = k / Fm::ipb, iqs = Fm::step * (k % Fm::ipb);
        const typename D::X xv = D::load_x(x + kbx * (Fm::qk / 32), iqs);
        D w[NR];   // (every row's loads issued before the first dot)
#pragma unroll
        for (int r = 0; r < NR; ++r)
            if (r < nr) w[r].load(w0 + (size_t) r * rb, kbx, iqs);
#pragma unroll
        for (int r = 0; r < NR; ++r)
            if (r < nr) s[r] += w[r].dot(xv);
    }
}

// rows of a BF16 / F32 job against its f32 activation (row_dot_bf16's / row_dot_f32's sums)
template<int NR>
__device__ __forceinline__ void dsa_rows_f(const MvJob& J, int row0, int nr, int lane, float (&s)[NR]) {
    if (J.type == kTypeBF16) {
        const int n8 = J.n_in / 8;
        const uint4* r8 = (const uint4*) ((const uint16_t*) J.w + (size_t) row0 * J.n_in);
        const float4* x4 = (const float4*) J.xf;
        for (int k = lane; k < n8; k += 32) {
            const float4 a = x4[2 * k], b = x4[2 * k + 1];
            uint4 w[NR];
#pragma unroll
            for (int r = 0; r < NR; ++r)
                if (r < nr) w[r] = r8[(size_t) r * n8 + k];
#pragma unroll
            for (int r = 0; r < NR; ++r)
                if (r < nr) {
                    const uint4 v = w[r];
                    s[r] += __uint_as_float(v.x << 16) * a.x + __uint_as_float(v.x & 0xffff0000u) * a.y +
                            __uint_as_float(v.y << 16) * a.z + __uint_as_float(v.y & 0xffff0000u) * a.w +
                            __uint_as_float(v.z << 16) * b.x + __uint_as_float(v.z & 0xffff0000u) * b.y +
                            __uint_as_float(v.w << 16) * b.z + __uint_as_float(v.w & 0xffff0000u) * b.w;
                }
        }
    } else {
        const int n4 = J.n_in / 4;
        const float4* r4 = (const float4*) ((const float*) J.w + (size_t) row0 * J.n_in);
        const float4* x4 = (const float4*) J.xf;
        for (int k = lane; k < n4; k += 32) {
            const float4 b = x4[k];
            float4 w[NR];
#pragma unroll
            for (int r = 0; r < NR; ++r)
                if (r < nr) w[r] = r4[(size_t) r * n4 + k];
#pragma unroll
            for (int r = 0; r < NR; ++r)
                if (r < nr) {
                    const float4 a = w[r];
                    s[r] += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
                }
        }
    }
}

// T > 0: one type's jobs (a launch per type); T == 0: any of them (one launch for the batch, a switch per block)
template<int T, int NR>
__global__ void __launch_bounds__(256) mv_dsa_kernel(const STRATA_GRID_CONSTANT MvBatch b) {
    if (b.skip != nullptr && *b.skip) return;   // (set_skip, as mv_kernel_t)
    const int bid = blockIdx.x;
    int ji = 0;
    while (ji < b.n - 1 && bid >= b.blk_end[ji]) ++ji;
    const MvJob& J = b.j[ji];
    const int blk0 = ji ? b.blk_end[ji - 1] : 0;
    const int lane = threadIdx.x & 31;
    const int row0 = ((bid - blk0) * MV_ROWS + (threadIdx.x >> 5)) * NR;
    if (row0 >= J.n_out) return;
    const int nr = min(NR, J.n_out - row0);
    float s[NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) s[r] = 0.0f;
    if constexpr (T > 0) {
        dsa_rows_q<T, NR>(J, row0, nr, lane, s);
    } else {
        switch (J.type) {
            case 13: dsa_rows_q<13, NR>(J, row0, nr, lane, s); break;
            case 8: dsa_rows_q<8, NR>(J, row0, nr, lane, s); break;
            case 12: dsa_rows_q<12, NR>(J, row0, nr, lane, s); break;
            case 14: dsa_rows_q<14, NR>(J, row0, nr, lane, s); break;
            default: dsa_rows_f<NR>(J, row0, nr, lane, s); break;   // kTypeBF16 / kTypeF32 (dsa_job_ok's filter)
        }
    }
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        const float v = warp_sum(s[r]);
        if (lane == 0 && r < nr) J.y[row0 + r] = mv_out(J.alpha, v, J.bias ? J.bias[row0 + r] : 0.0f);
    }
}

// the jobs mv_dsa_kernel takes: one column, a type above, its activation present, whole k steps
bool dsa_job_ok(const MvJob& j) {
    if (j.ncol != 1 || j.y == nullptr || j.w == nullptr) return false;
    switch (j.type) {
        case 13: case 12: case 14: return j.xq != nullptr && j.n_in % 256 == 0;
        case 8: return j.xq != nullptr && j.n_in % 32 == 0;
        case kTypeBF16: return j.xf != nullptr && j.n_in % 8 == 0;
        case kTypeF32: return j.xf != nullptr && j.n_in % 4 == 0;
        default: return false;
    }
}

// the jobs of `jobs` that pass dsa_job_ok (and, T > 0, are of type T) in one MvBatch of NR-row warps
template<int T, int NR>
bool dsa_launch(const MvJob* jobs, int n, cudaStream_t s) {
    MvBatch b{};
    b.skip = g_skip;
    int acc = 0, m = 0;
    for (int i = 0; i < n; ++i)
        if (dsa_job_ok(jobs[i]) && (T == 0 || jobs[i].type == T)) {
            b.j[m] = jobs[i];
            acc += (jobs[i].n_out + MV_ROWS * NR - 1) / (MV_ROWS * NR);
            b.blk_end[m++] = acc;
        }
    if (m == 0) return false;
    b.n = m;
    mv_dsa_kernel<T, NR><<<acc, 256, 0, s>>>(b);
    return true;
}

// The device policy (L1-4 review).  mv_dsa_kernel reproduces mv_kernel_t's / mv_kernel's sums, so it stands in only
// for jobs mv() itself gives those kernels, and only where the bits were seen to match: a device-tuned one-column
// kernel of mv() keeps its types (its own sums, its own tuning) - the P100 Q6_K kernel on cc 6.0 (p100_mv_mode),
// mv_kernel_nw2's K-quants when STRATA_GLMD_MV_NW2=1 asks for them, everything when STRATA_GLM_MV_GENERIC asks for the
// generic kernel - and a HIP build takes Q4_K / Q5_K / Q8_0 only: the RDNA GEMV serves Q6_K and BF16 on gfx115x /
// gfx12, and the float rows' sums are not gf::mv's bit for bit under HIP's contraction (gfx1151, glmd_dec_bench: the
// F32 indexer row differs; Q4_K / Q5_K / Q8_0 match).
struct DsaLeft {
    bool all = false, k_quants = false, q6_k = false, floats = false;
    bool operator()(int type) const {
        return all || (k_quants && (type == 12 || type == 13 || type == 14)) || (q6_k && type == 14) ||
               (floats && (type == kTypeBF16 || type == kTypeF32));
    }
};
DsaLeft dsa_left() {
    static const bool generic = getenv("STRATA_GLM_MV_GENERIC") != nullptr;   // (mv()'s own reading of it)
    DsaLeft l;
    l.all = generic;
    l.k_quants = (ext_mv_cfg() & 2) != 0;
#if defined(STRATA_USE_HIP)
    l.q6_k = l.floats = true;
#else
    l.q6_k = p100_mv_mode() > 0;
#endif
    return l;
}

// kMvDsaAuto's devices: where mv_dsa was measured faster than mv() - NVIDIA cc >= 7.0 (a V100: glmd_dec_bench, HANDOFF
// "L1-4").  AMD keeps mv(): on gfx1151 (the Gorgon Halo) the mixer's GEMVs are bound by the LPDDR5X (220-240 GB/s
// either way) and mv_dsa's launches came within 0-3% of mv()'s; Pascal is not measured.  STRATA_GLMD_MV_DSA=1 runs
// mv_dsa on any device.  Once per device.
bool dsa_auto_device() {
#if defined(STRATA_USE_HIP)
    return false;
#else
    static int cache[64];   // 0 unknown, 1 on, 2 off
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    if (cache[dev] == 0) {
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cache[dev] = major >= 7 ? 1 : 2;
    }
    return cache[dev] == 1;
#endif
}

// cfg (glm_fast.hpp): kMvDsaAuto = kMvDsaOn on dsa_auto_device()'s devices, else kMvDsaOff.  kMvDsaOn = the jobs it
// takes in one launch, by what the batch holds (V100, glmd_dec_bench, us against mv()):
//   - K-quants (Q4_K / Q5_K / Q6_K) with Q8_0 / BF16 / F32 jobs (a mixer's input projections): one row a warp for
//     all - in-proj full 43.6 -> 28.1, shared 32.8 -> 26.8, Q4_K + Q6_K + Q8_0 56.5 -> 45.0;
//   - Q8_0 / BF16 / F32 only (the query projections, eh_proj): two rows a warp, the q8_1 / f32 activation read once
//     for both - q_b + the indexer's q_b 76.6 -> 68.9, q_b 51.0 -> 48.5, eh_proj 120.8 -> 114.5;
//   - K-quants only: mv() itself - mv_kernel_t is as fast or faster on them alone (Q5_K 2048 x 6144 19.0 against
//     one-row warps' 20.1 and two-row warps' 24.6, the dense gate + up 134.2 against 142.5 / 169.9), and mv() launches
//     a type's jobs together anyway.
// 1 = one launch, NR 2, every job it takes; 2 = one launch, NR 4; 3 = a launch per type, NR 2; 5 = one launch, NR 1;
// kMvDsaOff = mv() itself (the old path, for A/Bs).  Every cfg leaves dsa_left()'s types to mv().
bool ext_mv_dsa(const MvJob* jobs, int n, cudaStream_t s, int cfg) {
    if (n <= 0 || n > kMaxMvJobs) return false;
    if (cfg == kMvDsaAuto) cfg = dsa_auto_device() ? kMvDsaOn : kMvDsaOff;
    if (cfg < 0) return mv(jobs, n, s);
    const DsaLeft left = dsa_left();
    int n_kq = 0, n_other = 0;   // the jobs it could take: K-quants, the rest
    for (int i = 0; i < n; ++i)
        if (dsa_job_ok(jobs[i]) && !left(jobs[i].type))
            ++(jobs[i].type == 12 || jobs[i].type == 13 || jobs[i].type == 14 ? n_kq : n_other);
    if (cfg == kMvDsaOn && n_other == 0) return mv(jobs, n, s);   // (K-quants only, or nothing it takes)
    MvJob take[kMaxMvJobs], rest[kMaxMvJobs];
    int n_take = 0, n_rest = 0;
    for (int i = 0; i < n; ++i) {
        const MvJob& j = jobs[i];
        if (dsa_job_ok(j) && !left(j.type)) take[n_take++] = j;
        else rest[n_rest++] = j;
    }
    if (n_take > 0) {
        if (cfg == kMvDsaOn) {
            if (n_kq > 0) dsa_launch<0, 1>(take, n_take, s);
            else dsa_launch<0, 2>(take, n_take, s);
        } else if (cfg == 2) {
            dsa_launch<0, 4>(take, n_take, s);
        } else if (cfg == 5) {
            dsa_launch<0, 1>(take, n_take, s);
        } else if (cfg == 3) {
            dsa_launch<13, 2>(take, n_take, s);
            dsa_launch<8, 2>(take, n_take, s);
            dsa_launch<12, 2>(take, n_take, s);
            dsa_launch<14, 2>(take, n_take, s);
            MvJob fl[kMaxMvJobs];   // (the float rows: the mixed kernel)
            int nf = 0;
            for (int i = 0; i < n_take; ++i)
                if (take[i].type == kTypeBF16 || take[i].type == kTypeF32) fl[nf++] = take[i];
            if (nf > 0) dsa_launch<0, 2>(fl, nf, s);
        } else {
            dsa_launch<0, 2>(take, n_take, s);
        }
        launch_check("mv_dsa");
    }
    return n_rest == 0 || mv(rest, n_rest, s);
}
