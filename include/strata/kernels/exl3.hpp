// include/strata/kernels/exl3.hpp - EXL3 (exllamav3) quantized weights on the device: the decode GEMV, the routed
// experts, and the full reconstruction for the prompt path.
//
// The format (exllamav3, MIT, turboderp).  A weight with k inputs and n outputs (y = x W) is stored as
//   trellis  int16 [k/16][n/16][16 K]: each 16x16 tile is a tail-biting trellis of 256 K-bit steps read MSB-first
//            from its 8 K uint32 words (little-endian pairs of the int16s).  Step p's state is the 16-bit window that
//            ends at bit (p + 1) K, modulo the tile's 256 K bits; the codebook turns the state into a value:
//              mul1 (cb 2):  fp16(1024 + bytesum(state * 0x83DCD12D)) * fp16(0x1eee) + fp16(0xc931), one FP16 FMA
//              mcg (cb 1):   x = state * 0xCBAC1FED, 3inst (cb 0): x = state * 89226354 + 64248484; then
//                            x = (x & 0x8fff8fff) ^ 0x3b603b60 and the value is the FP16 sum of x's two halves.
//            Step p sits at tile row (p/8 % 4) * 2 + (p & 1) + 8 (p/2 & 1), column p/32 + 8 (p/4 & 1) (the tensor-
//            core fragment order).
//   suh, svh FP16 [k], [n]: the input and output channel scales (signs and magnitudes).
// With H the 128-point Sylvester Hadamard over sqrt(128), applied to each 128 consecutive values:
//   y = H(H(x * suh) W') * svh,   W' the decoded k x n matrix.
//
// Activations enter as F32 or as the engine's q8_1 blocks; the kernels rotate them on the fly (each block rotates
// the slice it reads), so callers keep their existing buffers.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace strata::kernels::exl3 {

/// The weight type the engine's MvJob / WSlot carry for EXL3: their weight pointer is a (host) const Mat*.
constexpr int kTypeEXL3 = 200;

/// One EXL3 matrix, or a column range of one (a view: trellis/svh advanced to its first column, ld the row stride).
struct Mat {
    const uint32_t* trellis = nullptr;   // device; 4-byte aligned
    const uint16_t* suh = nullptr;       // device FP16 bits [k]
    const uint16_t* svh = nullptr;       // device FP16 bits [n]
    int k = 0, n = 0;
    int K = 0;                           // bits per weight, 1..8
    int cb = 2;                          // codebook: 0 3inst, 1 mcg, 2 mul1
    int ld = 0;                          // tiles per 16-row slice in memory (the full matrix's n / 16); 0: n / 16
};

/// Trellis bytes of a k x n matrix at K bits.
inline size_t trellis_bytes(int64_t k, int64_t n, int K) { return (size_t) (k / 16) * (size_t) (n / 16) * 32u * K; }
/// The columns [c0, c0 + nc) of m (c0 and nc multiples of 128: the output rotation's blocks stay whole).
Mat view(const Mat& m, int c0, int nc);
/// Whether the kernels take this matrix (k, n multiples of 128, K 1..8, cb 0..2).
bool supported(const Mat& m);

/// y[r] = alpha * (x W)[r] + (bias ? bias[r] : 0).  x: xf (F32) when given, else xq (q8_1 blocks of k values).
struct MvJob {
    const Mat* m = nullptr;
    const float* xf = nullptr;
    const void* xq = nullptr;
    float* y = nullptr;
    const float* bias = nullptr;
    float alpha = 1.0f;
};
constexpr int kMaxMvJobs = 8;
bool mv(const MvJob* jobs, int n, cudaStream_t s);

/// Allocates the per-stream scratch of mv / moe_down on the current device (call once per stream before a graph
/// capture; mv allocates it on first use otherwise).
bool prepare(cudaStream_t s);

/// One routed expert's blob: three parts (gate, up, down) at byte offsets from the blob base, each holding the FP16
/// suh, the FP16 svh and the trellis of that matrix.
struct ExpertLayout {
    size_t suh[3] = {0, 0, 0}, svh[3] = {0, 0, 0}, trellis[3] = {0, 0, 0};
    int K[3] = {0, 0, 0};
    int cb = 2;
};
/// The routed experts of one token (the engine's device plan: plan_ptr[i] = the blob in VRAM, 0 = not on the device;
/// plan_w[i] its weight).  gate/up over x (n_embd F32) -> clamped swiglu -> h (F32 [k][n_ff], scratch the caller
/// owns), then down over h, the weighted sum in plan order, + cpu_part when *cpu_flag, + sh_out when given -> out.
struct MoeArgs {
    const unsigned long long* plan_ptr = nullptr;
    const float* plan_w = nullptr;
    const float* cpu_part = nullptr;
    const int* cpu_flag = nullptr;
    ExpertLayout lay;
    int k = 8, n_embd = 4096, n_ff = 2048;
    float limit = 0.0f;   // 0: no clamp
};
void moe_gate_up(const MoeArgs& a, const float* x, float* h, cudaStream_t s);
void moe_down(const MoeArgs& a, const float* h, const float* sh_out, float* out, cudaStream_t s);

/// The prompt path's tensor-core GEMMs (sm_80 and later; mma_supported() says whether this device runs them).
bool mma_supported();
/// A set of n_exp routed experts in VRAM, expert e's blob at wbase + e * stride (the layout in lay), owning the sorted
/// rows [bounds[e], bounds[e + 1]) (device, n_exp + 1 entries); a sorted row's LOCAL row is bounds[e] - row0 + r.
struct SetArgs {
    const uint8_t* wbase = nullptr;
    size_t stride = 0;
    ExpertLayout lay;
    int n_exp = 0;
    const int* bounds = nullptr;
    int row0 = 0;
    int max_rows = 0;   // the most rows one expert of the set has
    int n_embd = 4096, n_ff = 2048;
    float limit = 0.0f;
};
/// h[local row] = FP16 clamped swiglu of gate and up over x[row_idx[local row]] (F32 rows, ldx floats apart).
void set_gate_up(const SetArgs& a, const float* x, int ldx, const int* row_idx, uint16_t* h16, int ldh, cudaStream_t s);
/// y[local row] = down(h[local row]) (F32 rows, ldy floats apart).
void set_down(const SetArgs& a, const uint16_t* h16, int ldh, float* y, int ldy, cudaStream_t s);
/// Y[r] = beta Y[r] + (X[r] W) for r < rows: one matrix over FP16 rows (ldx), F32 output rows (ldy).
void gemm_rows(const Mat& m, const uint16_t* x16, int ldx, int rows, float* y, int ldy, float beta, cudaStream_t s);

/// The full weight in the engine's row layout: dst[r][c] = W[c][r] (n rows of k values), BF16 or F16 bits.
void reconstruct_rows(const Mat& m, uint16_t* dst, bool bf16, cudaStream_t s);
/// The decoded W' alone, row-major [k][n] F16 bits (tests).
void reconstruct_inner(const Mat& m, uint16_t* dst, cudaStream_t s);

/// out[r][c] = FP16(x[idx ? idx[r] : r][c]) for r < nrows, c < ncols (x: ldx floats a row): the prompt path's rows of
/// one expert, gathered for its GEMMs.
void rows_f16(const float* x, int ldx, const int* idx, int nrows, int ncols, uint16_t* out, cudaStream_t s);

/// Launch failures of this family since the process started (each is also printed).
int launch_errors();

}  // namespace strata::kernels::exl3
