// include/strata/prefill/moe_mmq.hpp - prompt-speed plan step 2b: the prompt path's experts through llama.cpp's MMQ
// kernels (ggml-cuda mmq.cuh, MIT): the weights stay quantized and the activations are rounded to q8_1, the
// products run on int8 tensor cores.  The dequantize-to-FP16 + cuBLAS path wrote ~10 MB of FP16 per expert and
// multiplied in FP16; this reads the ~1.4-2 MB expert once.  A group of experts is gathered into one buffer
// (`gather_*`, one launch per expert as its blob arrives) and multiplied in one launch per product.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::prefill::mmq {

/// This build has the MMQ path (the ggml sources were available to the build).
bool built();
/// MMQ covers this ggml type (the i-quants and Q2_0 the packs use; IQ1_M is not covered).
bool supported(int ggml_type);
/// Bytes of one expert's gate+up ([2*n_ff, n_embd]) or down ([n_embd, n_ff]) weights in `ggml_type`.
size_t matrix_bytes(int ggml_type, int64_t rows, int64_t cols);
/// Bytes of `rows` activation rows of `cols` values quantized for MMQ (the row padded to 512 values).
size_t q8_bytes(int64_t rows, int64_t cols);

/// Lend the MMQ kernels' device scratch (their stream-k fixup tiles, allocated lazily through ggml_cuda_pool) the
/// part of the borrowed expert-pool region a chunk's layout did not use, so it costs VRAM the prompt has already
/// taken rather than a cudaMalloc on a card with a few MB free.  Call per chunk with the chunk's leftover tail;
/// `base` null or a moved base drops the buffers cached from the previous chunk.
void pool_slab(int dev, void* base, size_t bytes);

/// One q8_1 MMQ block's bytes: the activations are laid out column block by column block, so within a block row
/// r + 1 follows row r by this much (a product over rows [a, b) of a buffer quantized from row a points a blocks back).
size_t q8_block_bytes();
/// q8_1 activations for MMQ against weights of `ggml_type`: row i of the output is row ids[i] of x (or row i when
/// ids is null); `x` has `ld` floats per row.
void quantize(const float* x, const int32_t* ids, void* xq, int ggml_type, int64_t cols, int64_t ld, int64_t rows,
              void* stream);

/// One launch over n experts whose weights lie `expert_bytes` apart from `w`: for expert e, the activation rows
/// [bounds[e], bounds[e+1]) of `xq` (bounds on the device, n+1 entries) times its [w_rows, w_cols] matrix into
/// dst rows of the same indices (`ld_dst` floats apart, via `ids`: dst row = ids[row], an identity table works).
/// `total_rows`: the rows of xq; `max_rows`: the most rows one expert has (the launch grid).
struct Product {
    const void* w = nullptr;
    int type = -1;
    int64_t w_rows = 0, w_cols = 0;
    size_t expert_bytes = 0;
    int n = 0;
    const void* xq = nullptr;
    const int32_t* bounds = nullptr;
    const int32_t* ids = nullptr;
    int64_t total_rows = 0, max_rows = 0;
    float* dst = nullptr;
    int64_t ld_dst = 0;
};

/// This build and the current device have the fused gate/up + SwiGLU product (Context::run_glu) for `ggml_type` with
/// `n_ff` output rows: CUDA on an NVIDIA GPU with int8 tensor cores (Turing or later), n_ff a multiple of 128.
bool glu_supported(int ggml_type, int64_t n_ff);

/// The launch context (llama.cpp's MMQ keeps a small scratch pool for its stream-k fixup).  One per prompt path.
class Context {
public:
    Context();
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    void run(const Product& p, void* stream);
    /// The gate/up product with the SwiGLU in its write-back: `p` as for the plain product
    /// of an expert's [2 n_ff, n_embd] gate-then-up rows (w_rows = 2 n_ff), but dst row r gets only
    /// h = silu(min(gate, limit)) * clamp(up, -limit, limit) in its first n_ff floats (limit = INFINITY: plain SwiGLU) -
    /// the plain product's values through a SwiGLU kernel built without fast math, bit for bit.  Only where
    /// glu_supported(p.type, n_ff).
    void run_glu(const Product& p, float limit, void* stream);

private:
    void* ctx_ = nullptr;
};

/// A GGUF-native expert (gate at `gate`, up at `up`, down at `down`, each its GGUF rows) into a group buffer's
/// slot: gate rows then up rows at `gu_dst`, down at `d_dst`.
void gather_native(const void* gate, const void* up, size_t gu_half_bytes, const void* down, size_t d_bytes,
                   void* gu_dst, void* d_dst, void* stream);
/// A Strata-pack Q2_0 expert blob (codes and fp16 scales in separate planes, gate/up rows interleaved) into GGUF
/// Q2_0 blocks: gate/up [1280, 2560] at `gu_dst` (rows stay interleaved), down [2560, 640] at `d_dst`.  Same values.
void gather_strata_q2(const uint8_t* blob, void* gu_dst, void* d_dst, void* stream);

/// h[r, k] = silu(gate) * up of GU rows [2 n_ff wide]: interleaved (gate 2k, up 2k+1: the Strata pack) or split
/// (gate k, up n_ff + k: GGUF).  FP32 out (the down product's quantizer reads floats).
void swiglu(const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved, void* stream);

/// dst[i] = i for i < n (the identity row map MMQ's MoE mode writes through).
void iota(int32_t* dst, int64_t n, void* stream);

/// The readable entries an `ids` table needs after its last row.  MMQ reads whole column tiles of row ids (up to 128
/// each) without checking them against the group's last row: the product past the chunk's last row when the last
/// group's tile is not full, its stream-k fixup up to about two tiles (issue #645: 2 GPUs, 2048-token chunks,
/// compute-sanitizer flagged mul_mat_q reading up to 80 ids and the fixup ~190 past the table's end, which a card where
/// the next page is not mapped turns into "an illegal memory access").  The values there are never used.
constexpr int64_t IDS_PAD = 512;

/// llama.cpp's dequantizer to FP16 (ggml-cuda convert.cu) has `ggml_type`.
bool to_f16_supported(int ggml_type);
/// The `n` weights at `w` (whole blocks of `ggml_type`, row-major) as FP16 at `dst` (IEEE half bits); false when the
/// type is not covered.
bool to_f16(int ggml_type, const void* w, uint16_t* dst, int64_t n, void* stream);

}  // namespace strata::prefill::mmq
