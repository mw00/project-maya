// src/ds4/ds4_prompt_mem.hpp - the DeepSeek engine's small rules for its prompt path and its memory: which way a
// prompt goes, the chunk its buffers fall back to, the staging ring, the RAM copies of the CPU lane's experts - pure
// functions, CPU only, shared by ds4_model.cu and tests/ds4_prompt_mem_test.cpp
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace strata::ds4 {

// a 0 / 1 switch: -1 when it is not set (or empty), else whether it is non-zero
inline int tri_of(const char* v) { return v == nullptr || *v == 0 ? -1 : std::atoi(v) != 0; }

// ---- which way a prompt goes
// The timing pick (pick_path) predicts each path's time from this run's measured speeds, earlier prompts' times and
// the page cache's state, so the same prompt can take the chunked pass in one run and windows of 16 in the next - and
// the two round differently, so its logits differ.  The length rule: the chunked pass from kPromptFlowMin tokens,
// windows below - the same prompt always the same path.
constexpr int kPromptFlowMin = 96;

// DS4_PROMPT_PICK: "length" (from kPromptFlowMin tokens) or "length:N" (from N, at least 1) -> the length from which
// the chunked pass is taken; anything else, or unset: 0 (the timing pick)
inline int prompt_pick_length(const char* v) {
    if (v == nullptr || std::strncmp(v, "length", 6) != 0) return 0;
    if (v[6] == 0) return kPromptFlowMin;
    if (v[6] != ':' || v[7] == 0) return 0;
    return std::max(1, std::atoi(v + 7));
}
inline bool prompt_by_length(int tokens, int flow_min) { return tokens >= flow_min; }

// ---- the chunk buffers that do not fit: the next chunk to try - half (whole 16s), and 0 (windows of 16) below 256
constexpr int kChunkFloor = 256;
inline int chunk_fallback(int chunk) {
    const int h = chunk / 2 / 16 * 16;
    return h >= kChunkFloor ? h : 0;
}

// ---- the staging ring (MB) from a device's VRAM room for experts (bytes): 64 MB a GB of it, 400 to 1200 MB
inline int staging_ring_mb(double room_bytes) {
    const double mb = std::max(0.0, room_bytes) / 1e9 * 64.0;
    return (int) std::min(1200.0, std::max(400.0, mb));
}

// ---- the staging ring's slots by default (#609): the prompt path streams a layer's CPU-lane experts through the ring
// in `parts` batches, so a batch is what a layer waits for before it computes on it.  A batch of about kRingBatchS of
// this device's measured link (link: bytes a second, 0 = not measured) - a slow link waits less for each, a fast one
// keeps the batch it had - at least one slot, never more than kRingMaxMB (the fixed default before) and never more
// than 1/16 of the device's VRAM room for experts (room: bytes, 0 = not known).  Measured: 2x CMP 170HX on PCIe Gen2
// x4 / x16 read prompts fastest at 1-3 slots a batch, an L40S at 5 of 18.8 MB (the most tried), a V100 on PCIe 3 x16
// (6.9 MB experts) at 15, the 400 MB ring - smaller rings were slower there.  A link that takes kRingSlowS or more
// for one expert takes one a batch: the CMP 170HX on Gen2 x16 (9 MB experts, 6.2 GB/s: 1.5 ms each) read 4K-32K
// prompts 2-3.5% faster at one a batch than at the 6 of ~10 ms (#609).  -> the slots (a multiple of parts)
constexpr double kRingBatchS = 0.010;
constexpr double kRingSlowS = 0.001;
constexpr int kRingMaxMB = 400;
inline int ring_slots(double link, size_t stride, double room, int parts) {
    if (stride == 0 || parts <= 0) return parts;
    const double cap_b = std::min((double) ((size_t) kRingMaxMB << 20), room > 0 ? room / 16.0 : 1e300);
    const int cap = std::max(1, (int) (cap_b / (double) stride) / parts);   // (batches of the cap)
    const int want = link <= 0 ? cap
                   : (double) stride / link >= kRingSlowS ? 1
                   : (int) (link * kRingBatchS / (double) stride);
    return std::max(1, std::min(cap, want)) * parts;
}

// ---- --mmap-experts auto: every CPU-lane expert copied into RAM (all bytes) when they fit what the process can get
// (can) with `spare` more - never beside an integrated GPU, whose own allocations come out of the same RAM (force:
// DS4_COPY_ALL, -1 unset, 0 never, 1 the fit alone decides, integrated or not)
inline bool copy_all_auto(double all, double can, double spare, bool integrated, int force) {
    if (force == 0 || (integrated && force < 0)) return false;
    return all + spare <= can;
}

// ---- the copies on explicit huge pages: env DS4_COPY_HUGETLB (-1 unset, 0 off, 1 on); pool: the free bytes of the
// kernel's reserved pool; need: the copies' bytes, each allocation rounded up to whole pages.  Unset: on only when the
// pool already holds all of them (no pool reserved: the ordinary arena, as before)
inline bool copy_hugetlb_on(int env, uint64_t pool, uint64_t need) {
    if (env >= 0) return env != 0;
    return pool > 0 && need > 0 && pool >= need;
}
inline uint64_t round_up(uint64_t bytes, uint64_t page) { return page == 0 ? bytes : (bytes + page - 1) / page * page; }

// ---- an integrated GPU's room (Linux, NVIDIA): the driver's free figure counts only MemFree there, while the page
// cache (the model's files just read) is what the kernel can still give back - MemAvailable (~0: not known) less
// headroom when that is larger, at most the device's total
inline uint64_t uma_free(uint64_t driver_free, uint64_t total, uint64_t mem_available, uint64_t headroom) {
    if (mem_available == ~uint64_t{0} || mem_available <= headroom) return driver_free;
    return std::min(std::max(driver_free, mem_available - headroom), std::max(total, driver_free));
}

}  // namespace strata::ds4
