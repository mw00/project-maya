#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace ik2::cpu {
struct PackedLayer;
struct TransferBuffers {
    int* picks = nullptr;
    float *input = nullptr, *gate_up = nullptr, *output = nullptr;
    uint16_t* hidden = nullptr;
};

#ifdef IK2_HOST_AVX2
std::shared_ptr<PackedLayer> pack_layer(int width, int feed_forward, int picks, int experts,
                                        const uint8_t* const* parts, const size_t* strides, int threads, bool q8,
                                        TransferBuffers buffers);
void gate_up_rows(void* layer);
void down_rows(void* layer);
#endif
} // namespace ik2::cpu
