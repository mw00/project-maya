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
    unsigned long long picks, tier, stage, fetched, bytes, cpu;
};

// ---- the lane-parity mode (IK2_LANE): every expert in ik2_exact.h's arithmetic, so the CPU lane can take misses too

/// A visit's CPU work, written by the GPU into host memory mapped for it (the lane polls seq); x [C] at kLaneJobX.
struct LaneJob {
    uint32_t seq;
    uint32_t n;
    int layer;
    int pad;
    int slot[kOffMaxK], e[kOffMaxK];
    float sg[kOffMaxK], su[kOffMaxK];
};
constexpr size_t kLaneJobX = 512;   // (bytes: the job's x starts here)
/// The lane's answer, in mapped host memory (the GPU polls seq): y [n][C] raw down sums at kLaneDoneY.
struct LaneDone {
    uint32_t seq;
    uint32_t pad[15];
};
constexpr size_t kLaneDoneY = 64;
/// What the split weighs (device memory, the host's measured figures): microseconds an expert by format (0 NVFP4, 1
/// Q8_0) - on the CPU lane, copied over the bus, computed on the GPU - and the lane's round trip; force 1: every miss
/// fetched, 2: every miss on the CPU (tests)
struct LaneCost {
    float cpu[2], fetch[2], gpu[2], lat;
    int force;
};
/// The visit's CPU share on the device (the plan writes it, the wait reads it)
struct LaneState {
    uint32_t seq;
    int n;
    uint32_t err;
    int pad;
    int slot[kOffMaxK];
};

/// off_fetch_plan with the CPU lane beside the fetch: each miss not in staging fetched or given to the lane, the split
/// the one the costs (fmt: the layer's) say finishes first; the lane's picks get cls 1 (the GPU skips them), the
/// others cls 0.  x [C] (the experts' input), sg / su [K] (route_token's) go into the job.  One warp.
void off_lane_plan(const uint8_t** blob, const int* top, int K, int key0, const uint8_t* host, size_t hstride,
                   OffStage st, OffFetch* fl, OffCount* cnt, uint32_t* heat, const LaneCost* cost, int fmt,
                   const float* x, int C, const float* sg, const float* su, int layer, LaneJob* job, LaneState* ls,
                   int* cls, cudaStream_t s);
/// h [K][F] = swiglu(gate . x sg, up . x su) for the slots of cls 0 (ik2_exact.h's sums; gate and up of type ty, up at
/// eb0 after gate)
void off_gate_up_x(const uint8_t* const* blob, const int* cls, int K, int ty, size_t eb0, const float* x, int C, int F,
                   const float* sg, const float* su, float gcap, float ucap, float* h, cudaStream_t s);
/// y [K][C] = down . h (raw sums) for the slots of cls 0 (down of type ty at off_down)
void off_down_x(const uint8_t* const* blob, const int* cls, int K, int ty, size_t off_down, const float* h, int C,
                int F, float* y, cudaStream_t s);
/// The lane's rows of this visit into y (none: nothing waited for); a lane silent for 20 s sets ls->err and goes on.
void off_lane_wait(LaneState* ls, const LaneDone* done, int C, float* y, cudaStream_t s);

/// The misses of one visit (blob[i] null: not in the tier) into staging: a slot holding the expert already (tag) is
/// reused, else the clock's next slot that the visit does not use is taken and the expert listed for the copy.
/// host: the layer's pinned experts as the device sees them (expert e at host + e * hstride), key0 = layer * E.
/// heat (null: not counted): heat[key0 + top[i]] += 1.  One launch of one warp.
void off_fetch_plan(const uint8_t** blob, const int* top, int K, int key0, const uint8_t* host, size_t hstride,
                    OffStage st, OffFetch* fl, OffCount* cnt, uint32_t* heat, cudaStream_t s);
/// off_fetch_plan for a verify window (drafting on the offload path): the window's picks grouped by expert
/// (group_route's blob / n / row [S][GM], top [S] the picks) - a group the tier does not hold found in staging or
/// listed for one copy, its blob filled.  fl [n_fl]: the copies in runs of kOffMaxK (an off_fetch_copy each);
/// st.n >= S.  The experts' bytes are the tier's, so the window's kernels give the token path's results.
void off_fetch_plan_win(const uint8_t** blob, const int* n, const int* row, int GM, const int* top, int S, int key0,
                        const uint8_t* host, size_t hstride, OffStage st, OffFetch* fl, int n_fl, OffCount* cnt,
                        uint32_t* heat, cudaStream_t s);
/// The listed copies (an expert's `bytes` each, bytes % 16 == 0), host-mapped memory to VRAM, 16 bytes a thread at
/// a time; blocks: the grid.
void off_fetch_copy(const OffFetch* fl, size_t bytes, int blocks, cudaStream_t s);

}  // namespace ik2::k
