"""tools/vision/ggml_vision_precision.py - float32 accumulation for the vision encoder's two kinds of sums on NVIDIA and
AMD GPUs, in the pinned llama.cpp's ggml-cuda (also ggml's HIP backend).

The encoder (24 ViT blocks, then the merger) amplifies small rounding errors on a few high-norm patches, and on the
GPU two of ggml's choices round far more than the CPU does: those patches' rows came out wrong (cosine 0.3-0.8
against a float32 reference on Strix Halo and on an RTX 3050, where the CPU is at 0.80-0.99):
  - the F16 weight matmuls go to cuBLAS / hipBLAS with an F16 compute type and an F16 result.  ggml takes the F32
    result for Volta, RDNA4 and CDNA only; maya-vision takes it on every GPU (F16 inputs, F32 sums and result);
  - flash attention runs a kernel that sums softmax(KQ)*V in half precision on GPUs with fast FP16 (the MMA kernel on
    NVIDIA Volta and newer, the tile kernel on AMD), although the encoder's graph asks for GGML_PREC_F32, which the
    CUDA backend does not look at.  maya-vision keeps those sums (VKQ) in float: in the MMA kernel on Turing and newer
    (tensor cores with float accumulators, as its KQ already has) and in the tile kernel everywhere else (Volta, AMD,
    attention of 8 or fewer rows); K/V tiles and KQ values stay FP16 as before.
Every change is compiled only where MAYA_VISION_F32_SUMS is defined, which tools/vision/CMakeLists.txt sets on
maya-vision's own ggml-cuda / ggml-hip targets: the engines build the same files from the same tree and get exactly
the code they had before.  Every edit is anchored on the pinned source and idempotent, as glm5next_patch.py's (which
runs this one).
    python ggml_vision_precision.py <llama.cpp dir>
"""
from __future__ import annotations

import pathlib
import sys

MARK = "MAYA_VISION_F32_SUMS"


def edit(path: pathlib.Path, pairs: list[tuple[str, str, int]]):
    s = path.read_text(encoding="utf-8")
    for old, new, count in pairs:
        if new in s:
            continue                                   # already applied
        n = s.count(old)
        if n != count:
            raise SystemExit(f"ggml vision precision patch: {path}: anchor found {n} times (expected {count}):\n{old}")
        s = s.replace(old, new)
    path.write_text(s, encoding="utf-8", newline="\n")


FP16 = "#ifdef FAST_FP16_AVAILABLE\n"
FP16_HALF_VKQ = "#if defined(FAST_FP16_AVAILABLE) && !defined(GGML_FATTN_TILE_VKQ_F32)\n"


def main(root: pathlib.Path):
    cu = root / "ggml" / "src" / "ggml-cuda"
    edit(cu / "ggml-cuda.cu", [(
        "        prefer_f32_output = cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);\n",
        "        prefer_f32_output = cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);\n"
        "#ifdef MAYA_VISION_F32_SUMS\n"
        "        // (Maya, tools/vision/ggml_vision_precision.py: maya-vision only) the F32 result on every GPU - the F16\n"
        "        // compute type rounds the vision encoder's matmuls too coarsely\n"
        "        prefer_f32_output = true;\n"
        "#endif // MAYA_VISION_F32_SUMS\n", 1)])
    edit(cu / "fattn.cu", [(
        "    if (mask && mask->ne[2] != 1) {\n"
        "        return BEST_FATTN_KERNEL_NONE;\n"
        "    }\n\n",
        "    if (mask && mask->ne[2] != 1) {\n"
        "        return BEST_FATTN_KERNEL_NONE;\n"
        "    }\n\n"
        "#ifdef MAYA_VISION_F32_SUMS\n"
        "    // (Maya, tools/vision/ggml_vision_precision.py: maya-vision only) GGML_PREC_F32 is honoured with kernels whose\n"
        "    // softmax(KQ)*V sums are kept in float there: the MMA kernel on Turing and newer (its 16+ column tiles; the\n"
        "    // 8-column one sums in half), the tile kernel otherwise (the vector kernel sums in half)\n"
        "    if (ggml_flash_attn_ext_get_prec(KQV) == GGML_PREC_F32 && !ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {\n"
        "        if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72 && !gqa_opt_applies && Q->ne[1] > 8) {\n"
        "            return BEST_FATTN_KERNEL_MMA_F16;\n"
        "        }\n"
        "        return BEST_FATTN_KERNEL_TILE;\n"
        "    }\n"
        "#endif // MAYA_VISION_F32_SUMS\n\n", 1)])
    # the MMA kernel on Turing and newer: float VKQ accumulators for its 16-column tiles (the 8-column specialization
    # keeps half2: its result path converts VKQ to a B tile)
    edit(cu / "fattn-mma-f16.cuh", [
        ("#if defined(TURING_MMA_AVAILABLE)\n"
         "template<int DV, int ncols> struct mma_tile_sizes {\n"
         "    using T_A_KQ  = tile<16,  8, half2>; // row-major\n"
         "    using T_B_KQ  = tile<16,  8, half2>; // column-major\n"
         "    using T_C_KQ  = tile<16, 16, float>; // column-major\n"
         "    using T_A_VKQ = tile<16,  8, half2>; // row-major\n"
         "    using T_B_VKQ = tile<16,  8, half2>; // column-major\n"
         "    using T_C_VKQ = tile<16,  8, half2>; // column-major\n"
         "};\n",
         "#if defined(TURING_MMA_AVAILABLE)\n"
         "template<int DV, int ncols> struct mma_tile_sizes {\n"
         "    using T_A_KQ  = tile<16,  8, half2>; // row-major\n"
         "    using T_B_KQ  = tile<16,  8, half2>; // column-major\n"
         "    using T_C_KQ  = tile<16, 16, float>; // column-major\n"
         "    using T_A_VKQ = tile<16,  8, half2>; // row-major\n"
         "    using T_B_VKQ = tile<16,  8, half2>; // column-major\n"
         "#ifdef MAYA_VISION_F32_SUMS\n"
         "    using T_C_VKQ = tile<16, 16, float>; // column-major (Maya, ggml_vision_precision.py: float VKQ sums)\n"
         "#else\n"
         "    using T_C_VKQ = tile<16,  8, half2>; // column-major\n"
         "#endif // MAYA_VISION_F32_SUMS\n"
         "};\n", 1),
        ("#if defined(TURING_MMA_AVAILABLE)\n"
         "    T_C_VKQ VKQ_C[cols_per_warp == 8 ? DV/T_C_VKQ::I : DV/(2*T_C_VKQ::J)];\n",
         "#if defined(TURING_MMA_AVAILABLE) && defined(MAYA_VISION_F32_SUMS)\n"
         "    T_C_VKQ VKQ_C[cols_per_warp == 8 ? DV/T_C_VKQ::I : DV/T_C_VKQ::J];\n"
         "#elif defined(TURING_MMA_AVAILABLE)\n"
         "    T_C_VKQ VKQ_C[cols_per_warp == 8 ? DV/T_C_VKQ::I : DV/(2*T_C_VKQ::J)];\n", 1),
        # the VKQ rescales (softmax max update, attention sinks) of the 16-column tiles: this thread's rows are
        # get_i(l)/8 = 0 or 1 (lane/4 and lane/4 + 8)
        ("        } else {\n"
         "#pragma unroll\n"
         "            for (int col = 0; col < cols_per_thread; ++col) {\n"
         "                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);\n",
         "        } else\n"
         "#ifdef MAYA_VISION_F32_SUMS\n"
         "        if constexpr (std::is_same_v<decltype(T_C_VKQ::x), float[T_C_VKQ::ne]>) {\n"
         "#pragma unroll\n"
         "            for (int i = 0; i < DV/T_C_VKQ::J; ++i) {\n"
         "#pragma unroll\n"
         "                for (int l = 0; l < T_C_VKQ::ne; ++l) {\n"
         "                    VKQ_C[i].x[l] *= KQ_max_scale[T_C_VKQ::get_i(l) / 8];\n"
         "                }\n"
         "            }\n"
         "        } else\n"
         "#endif // MAYA_VISION_F32_SUMS\n"
         "        {\n"
         "#pragma unroll\n"
         "            for (int col = 0; col < cols_per_thread; ++col) {\n"
         "                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);\n", 2),
        # the combine's meta rows: the rows of the half2 tile, as before
        ("        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(threadIdx.x % 4);\n",
         "#ifdef MAYA_VISION_F32_SUMS\n"
         "        const int jc_cwm = threadIdx.y*cols_per_warp + tile<16, 8, half2>::get_i(threadIdx.x % 4);\n"
         "#else\n"
         "        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(threadIdx.x % 4);\n"
         "#endif // MAYA_VISION_F32_SUMS\n", 1),
    ])
    edit(cu / "fattn-tile.cuh", [
        ('#include "common.cuh"\n#include "fattn-common.cuh"\n',
         '#include "common.cuh"\n#include "fattn-common.cuh"\n\n'
         "// (Maya, tools/vision/ggml_vision_precision.py: maya-vision only) the softmax(KQ)*V sums (VKQ) are kept in\n"
         "// float: in half precision they lose the vision encoder's attention\n"
         "#if defined(MAYA_VISION_F32_SUMS) && defined(FAST_FP16_AVAILABLE)\n"
         "#define GGML_FATTN_TILE_VKQ_F32\n"
         "#endif\n", 1),
        # the VKQ rescales (softmax max update, attention sinks): the float path's
        (FP16 + "            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);\n",
         FP16_HALF_VKQ + "            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);\n", 2),
        # VKQ += V*KQ: FP16 V and KQ, float sums
        ("                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size] += V_k[i0/warp_size]*KQ_k[jc_VKQ_0];\n",
         "#ifdef GGML_FATTN_TILE_VKQ_F32\n"
         "                    const float2 v_f = __half22float2(V_k[i0/warp_size]);\n"
         "                    const float  p_f = __low2float(KQ_k[jc_VKQ_0]);\n"
         "                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size].x += v_f.x*p_f;\n"
         "                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size].y += v_f.y*p_f;\n"
         "#else\n"
         "                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size] += V_k[i0/warp_size]*KQ_k[jc_VKQ_0];\n"
         "#endif // GGML_FATTN_TILE_VKQ_F32\n", 1),
        # the accumulators themselves
        ("    __align__(16) half2 VKQ[cpw * ((DVp/2)/warp_size)] = {{0.0f, 0.0f}};\n",
         "#ifdef GGML_FATTN_TILE_VKQ_F32\n"
         "    __align__(16) float2 VKQ[cpw * ((DVp/2)/warp_size)] = {{0.0f, 0.0f}};\n"
         "#else\n"
         "    __align__(16) half2 VKQ[cpw * ((DVp/2)/warp_size)] = {{0.0f, 0.0f}};\n"
         "#endif // GGML_FATTN_TILE_VKQ_F32\n", 1),
        # combining the parallel warps' VKQ (in KV_tmp, as floats) and writing the result: the float path's
        ("        static_assert(nbatch_fa*nbatch_K >= nwarps*DVp, \"KV_tmp too small\");\n",
         "        static_assert(nbatch_fa*nbatch_K >= nwarps*DVp, \"KV_tmp too small\");\n"
         "#ifdef GGML_FATTN_TILE_VKQ_F32\n"
         "        static_assert(nbatch_fa*(nbatch_K/2 + cpy_ne) + DVp-DV >= nwarps*DVp, \"KV_tmp too small for float VKQ\");\n"
         "#endif // GGML_FATTN_TILE_VKQ_F32\n", 1),
        (FP16 + "        half2 * VKQ_combine    = (half2 *) KV_tmp;\n",
         FP16_HALF_VKQ + "        half2 * VKQ_combine    = (half2 *) KV_tmp;\n", 1),
        (FP16 + "            constexpr int cpy_ne_D = cpy_ne < (DVp/2)/warp_size ? cpy_ne : (DVp/2)/warp_size;\n",
         FP16_HALF_VKQ + "            constexpr int cpy_ne_D = cpy_ne < (DVp/2)/warp_size ? cpy_ne : (DVp/2)/warp_size;\n",
         2),
        (FP16 + "        constexpr int cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_size ? cpy_ne/2 : (DVp/2)/warp_size;\n",
         FP16_HALF_VKQ + "        constexpr int cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_size ? cpy_ne/2 : (DVp/2)/warp_size;\n",
         1),
    ])
    print(f"llama.cpp at {root}: float32 sums for the vision encoder (maya-vision's {MARK})")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
