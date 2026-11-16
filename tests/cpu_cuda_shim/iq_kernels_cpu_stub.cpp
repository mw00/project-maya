// tests/cpu_cuda_shim/iq_kernels_cpu_stub.cpp - link-time stand-ins for the i-quant kernels.
//
// WHY: native_mmvq.cu and dequant_bf16.cu reference iq_row_bytes / iq_mmvq / iq_dequant_f16 /
// iq_dequant_f32, which live in iq_kernels.cu.  That unit is not part of the CPU build (it is a
// CUDA kernel file with its own device tables), so the CPU test needs these four symbols to
// link.  The checks never call the IQ launch paths, and the only IQ byte math the test can reach
// is native_mmvq_weight_bytes for the IQ type ids - which is answered here from the repo's OWN
// block_geometry table rather than a second hand-written size list, so this stub cannot drift from
// the reader's idea of a block.
//
// Scratch for the CPU test only.  The repo's real builds link iq_kernels.cu.

#include "strata/kernels/iq_kernels.hpp"
#include "strata/artifact/gguf_reader.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

namespace {
[[noreturn]] void refuse_iq_launch(const char* function_name) {
    std::fprintf(stderr,
                 "%s: the i-quant GPU kernels are not built on this CPU-only host; the Q5_1 test "
                 "must not reach this path\n",
                 function_name);
    std::exit(2);
}
}  // namespace

size_t iq_row_bytes(int ggml_type, int64_t n) noexcept {
    int block_elems = 0;
    int block_bytes = 0;
    if (!strata::block_geometry((uint32_t) ggml_type, block_elems, block_bytes) || block_elems <= 0) return 0;
    return (size_t) (n / block_elems) * (size_t) block_bytes;
}

void iq_mmvq(int, const void*, const void*, float*, int, int, int, void*) {
    refuse_iq_launch("iq_mmvq");
}

void iq_dequant_f16(int, const void*, int64_t, uint16_t*, void*) { refuse_iq_launch("iq_dequant_f16"); }

void iq_dequant_f32(int, const void*, int64_t, float*, void*) { refuse_iq_launch("iq_dequant_f32"); }

}  // namespace strata::kernels
