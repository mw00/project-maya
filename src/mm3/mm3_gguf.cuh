// src/mm3/mm3_gguf.cuh - maya-mm3's GGUF weights on the device (MiniMax-M3.1-Flash's Maya quants, tools/mm3/pack_gguf.py):
// the dense matrices in ggml's block types (Q4_K, Q5_K, Q6_K, Q8_0 ...) and the routed experts in its low-bit ones (IQ1_S,
// IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS ...), run by maya-ds4's kernels (src/ds4/ds4_iq.cuh) with MiniMax's swigluoai.
//
// They stand in for the EXL3 matrices behind the same interface (mm3_exl3.hpp: an x3::Mat / x3::ExpertLayout with fmt
// kFmtGG; mm3_exl3.cu's entry points hand them here), so the engine's paths - the token, a verify window, the chunked
// prompt - take either format:
//   * a dense GEMV (mv): the activations rounded to int8 per 32 values (llama.cpp's q8_1 scale, ds4_gguf.cu's
//     q8_quant), each sub-block's dot exact in integers, F32 sums;
//   * the prompt path's GEMMs (gemm_rows): the weight decoded to FP16 rows (ds4_iq's f16_rows: ggml's values) and a
//     cuBLAS FP16 GEMM with F32 accumulation - a few column bands at a time when the matrix is large; the output head
//     of a short call through the GEMV instead;
//   * the routed experts: the activations rounded as the CPU lane rounds them (ggml's Q8_K, a scale a 256 - so an
//     expert gives the same sums on either side; MAYA_MM3_IQ_Q8K=0: a scale a 32), gate and up then swigluoai fused
//     (two passes where the layer's gate and up differ in type), down, and the weighted sum in plan order with the
//     CPU lane's part and the shared expert's - as the EXL3 kernels combine them.
// An expert's blob: gate | up | down, each [rows][blocks] as the GGUF holds it, 256-byte aligned
// (ExpertLayout::trellis[r] = role r's offset, gg_type[r] its ggml type).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "mm3_exl3.hpp"

namespace mm3::gg {

namespace x3 = strata::mm3::exl3;

/// x3::Mat::fmt / x3::ExpertLayout::fmt of a GGUF's weights (Mat: raw = [n][k] blocks of gg_type, ld bytes a row).
constexpr int kFmtGG = 4;

/// The ggml type id of a dtype name the pack writes ("Q6_K", "IQ1_S", ...; -1: not a block type these kernels read).
int type_of(const std::string& dtype);
/// Whether the routed experts' kernels take the type (the dense GEMVs take Q6_K and Q8_0 besides).
bool expert_type(int type);
bool dense_type(int type);
/// The type's name ("?" unknown), the bytes of a row of k values (0: not whole blocks), its values a block.
const char* type_name(int type);
size_t row_bytes(int type, int64_t k);
int block_values(int type);

/// mm3_exl3.cu's entry points for GGUF weights (the same contracts: mm3_exl3.hpp).
bool mv(const x3::MvJob* jobs, int n, cudaStream_t s);
void gemm_rows(const x3::Mat& m, const uint16_t* x16, int ldx, int rows, float* y, int ldy, float beta, cudaStream_t s);
void moe_gate_up(const x3::MoeArgs& a, const float* x, float* h, cudaStream_t s);
void moe_down(const x3::MoeArgs& a, const float* h, const float* sh_out, float* out, cudaStream_t s);
bool win_experts(const x3::WinArgs& a, cudaStream_t s);
void set_gate_up(const x3::SetArgs& a, const float* x, int ldx, const int* row_idx, uint16_t* h16, int ldh,
                 cudaStream_t s);
void set_down(const x3::SetArgs& a, const uint16_t* h16, int ldh, float* y, int ldy, cudaStream_t s);
/// The per-stream scratch (the activations' int8 rows, a token's down outputs, the GEMMs' FP16 weight bands and their
/// cuBLAS handle): allocated here at the load, so the plan counts it.  scratch_bytes(): what prepare() takes a stream.
bool prepare(cudaStream_t s);
size_t scratch_bytes();
/// The prompt path's dense GEMMs (gemm_rows with beta 0): 0 the BLAS (cuBLAS / hipBLAS's FP16 GEMM), 1 hipBLASLt with
/// each shape's fastest kernel measured on this GPU at its first product (src/ds4/ds4_lt; the picks kept in
/// set_lt_store's file) - an AMD build with hipBLASLt only, else the BLAS.  The prefill calibration chooses.
void set_gemm(int backend);
int gemm_backend();
bool gemm_lt_available();
void set_lt_store(const char* path);
/// Launch failures since the process started (each printed); the device waits on the host that gave up after 120 s.
int launch_errors();
unsigned wait_timeouts();

/// (tests) rows of k values (k % 256 == 0) rounded as the routed experts' activations are (Q8_K, or a scale a 32 with
/// MAYA_MM3_IQ_Q8K=0): q[r][k] int8, d[r][k / 32] (a 256's scale in each of its eight 32-blocks).
void act_quant(const float* x, int rows, int k, int8_t* q, float* d, cudaStream_t s);
/// ... and the dense GEMVs' (a scale a 32).
void q8_quant(const float* x, size_t ldx, int rows, int k, int8_t* q, float* d, cudaStream_t s);

}  // namespace mm3::gg
