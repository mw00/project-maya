// src/ik2/ik2_off.cuh - maya-ik2's offload token path: a token's routed experts that the VRAM tier does not hold,
// fetched by the GPU itself from pinned host memory into a staging cache in VRAM, with no host step between the
// router and the experts (so a token stays one CUDA graph, as on the resident path).
//
// The resident path's expert kernels run unchanged on what this leaves in the routed slots' table (blob): a VRAM
// tier slot, or the staging slot a miss was copied into - the same bytes, so the same bits as with every expert
// resident.  The staging slots are a small cache of their own (tags on the device, a clock hand), so an expert a
// looped core's later pass (or the next token) picks again is not copied twice.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace ik2::k {

constexpr int kOffMaxK = 16;   // routed experts a token (Mini: 8)

/// A staging pool (one a slot size): n slots of sb bytes, a tag a slot (layer * E + expert, -1 free), the visit that
/// last used it (a slot of the running visit is never the victim), and the clock hand.  Device memory throughout.
struct OffStage {
    uint8_t* data = nullptr;
    size_t sb = 0;
    int n = 0;
    int* tag = nullptr;
    uint32_t* used = nullptr;
    int* hand = nullptr;
};

/// What one visit copies: the plan writes it, the copy reads it (device memory).
struct OffFetch {
    int n;
    int pad[3];
    const uint8_t* src[kOffMaxK];
    uint8_t* dst[kOffMaxK];
};

/// The offload's counters (device memory, read by the host between requests): a pick a token, by expert
/// (heat [n_layer * E]), and the visit counter and the totals - picks, tier hits, staging hits, copies.
struct OffCount {
    uint32_t seq;
    uint32_t pad;
    unsigned long long picks, tier, stage, fetched, bytes;
};

/// The misses of one visit (blob[i] null: not in the tier) into staging: a slot holding the expert already (tag) is
/// reused, else the clock's next slot that the visit does not use is taken and the expert listed for the copy.
/// host: the layer's pinned experts as the device sees them (expert e at host + e * hstride), key0 = layer * E.
/// heat (null: not counted): heat[key0 + top[i]] += 1.  One launch of one warp.
void off_fetch_plan(const uint8_t** blob, const int* top, int K, int key0, const uint8_t* host, size_t hstride,
                    OffStage st, OffFetch* fl, OffCount* cnt, uint32_t* heat, cudaStream_t s);
/// The listed copies (an expert's `bytes` each, bytes % 16 == 0), host-mapped memory to VRAM, 16 bytes a thread at
/// a time; blocks: the grid.
void off_fetch_copy(const OffFetch* fl, size_t bytes, int blocks, cudaStream_t s);

}  // namespace ik2::k
