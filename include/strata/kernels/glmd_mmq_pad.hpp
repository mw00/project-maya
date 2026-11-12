// include/strata/kernels/glmd_mmq_pad.hpp - glm-dsa B15.2: the MMQ read-past guard of the expert banks.
//
// llama.cpp's MMQ kernels (strata_mmq) can read a little past the last byte of an expert (llama.cpp #30059: "one
// extra plane at the end, CUDA MMQ can read past the last expert"; its copies fill min(expert_size, 512) bytes of the
// next expert).  Bytes it reads there must be defined and finite: a NaN / inf pattern in memory nobody wrote poisons
// the product.  The rules every device buffer holding experts for strata_mmq follows:
//
//   1. every ALLOCATION that holds expert slots ends with kMmqTailPad (512) zeroed bytes, and is zeroed whole when it
//      is allocated (mmq_malloc): the expert pool's segments and its lendable tail (xpool, which the prompt path
//      carves its staging groups and its prestage buffer from);
//   2. a staging ring of groups (the prompt path's NG groups of GE slots, gbuf) gives each GROUP its own 512-byte
//      tail (mmq_group_offset / mmq_group_ring_bytes), so a full group never reads into the next group, which the
//      copy stream may be refilling;
//   3. after the copies into a group or the prestage buffer, the gap of each written slot (blob .. stride) and up to
//      512 bytes after the last written slot (inside the group's or buffer's room) are zeroed on the copy stream
//      before the product's event (mmq_zero_gaps): bytes a larger expert of another layer left there are not this
//      layer's type, and their FP16 scale fields may decode to NaN;
//   4. the prestage buffer keeps 512 bytes free at its end (mmq_slots_fit).
//
// The pool's main slots hold one layer each in slot order and are only ever written blob bytes at slot starts: their
// gaps stay as rule 1 left them (zero), and a layer's last slot is followed by the next layer's first slot (an expert
// of a finite type) or the segment's zero tail.  The disk landing ring of the B9.9 stream driver is pinned HOST memory
// that only feeds copies of `blob` bytes into the groups above; MMQ never reads it.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace glmd {

constexpr size_t kMmqTailPad = 512;
/// The pad is a glm-dsa measure: Glm5Model::load_pack switches it on for a glm-dsa model only.  Off (the default,
/// every other model) every function below gives Maya v1.1.55's sizes and offsets, mmq_malloc is a plain cudaMalloc
/// and mmq_zero_gaps does nothing - the Flash prompt and decode paths stay bit-identical.
inline std::atomic<bool>& mmq_pad_flag() {
    static std::atomic<bool> f{false};
    return f;
}
inline bool mmq_pad_on() { return mmq_pad_flag().load(std::memory_order_relaxed); }
inline void mmq_pad_enable(bool on) { mmq_pad_flag().store(on, std::memory_order_relaxed); }
/// The tail in force: kMmqTailPad on glm-dsa, 0 otherwise.
inline size_t mmq_tail() { return mmq_pad_on() ? kMmqTailPad : 0; }

/// The bytes to allocate for `bytes` of expert slots: the slots and the zeroed tail.
inline size_t mmq_padded(size_t bytes) { return bytes + mmq_tail(); }
/// The bytes of a buffer of `bytes` that slots may use (the tail kept free): 0 when it is smaller than the tail.
inline size_t mmq_usable(size_t bytes) { return bytes > mmq_tail() ? bytes - mmq_tail() : 0; }
/// Slots of `stride` bytes that fit a buffer of `bytes` with its tail kept free.
inline int64_t mmq_slots_fit(size_t bytes, size_t stride) {
    return stride == 0 ? 0 : (int64_t) (mmq_usable(bytes) / stride);
}
/// One staging group of `ge` slots of `stride` bytes: its slots and its own tail.
inline size_t mmq_group_room(int ge, size_t stride) { return (size_t) ge * stride + mmq_tail(); }
/// The byte offset of group b in a ring of groups whose slots are `stride` apart (the layer's stride; the ring was
/// sized with the largest stride, mmq_group_ring_bytes).
inline size_t mmq_group_offset(int b, int ge, size_t stride) { return (size_t) b * mmq_group_room(ge, stride); }
/// A ring of ng groups of ge slots of at most `max_stride` bytes.
inline size_t mmq_group_ring_bytes(int ng, int ge, size_t max_stride) {
    return (size_t) ng * mmq_group_room(ge, max_stride);
}

/// A byte range [off, off + len) relative to a slot run's base.
struct MmqRun {
    size_t off = 0, len = 0;
};
/// The ranges mmq_zero_gaps zeroes after n slots of `blob` bytes were written `stride` apart from a base with `room`
/// bytes: each slot's gap [j * stride + blob, (j + 1) * stride) and the tail [n * stride, min(room, n * stride +
/// kMmqTailPad)).  Empty ranges are left out; nothing reaches past room.
std::vector<MmqRun> mmq_gap_runs(int n, size_t stride, size_t blob, size_t room);

// ---- GPU (src/kernels/cuda/glmd_guards.cu; HIP: plain cudaMalloc / no-ops) ---------------------------------------

/// cudaMalloc of mmq_padded(bytes), the whole allocation zeroed (synchronously).  Returns the cudaMalloc result (as
/// int: 0 = cudaSuccess); *p is nullptr on failure.
int mmq_malloc(void** p, size_t bytes);
/// Zeroes mmq_gap_runs(n, stride, blob, room) below base on `stream` (one launch).  false on a launch error.
bool mmq_zero_gaps(void* base, int n, size_t stride, size_t blob, size_t room, void* stream);

}  // namespace glmd
