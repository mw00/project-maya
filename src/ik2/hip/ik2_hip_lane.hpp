// src/ik2/hip/ik2_hip_lane.hpp - maya-ik2's CPU lane: the routed experts that are not in VRAM, computed on the
// CPU from their copies in RAM or, with --mmap-experts, in place in the model's files (docs/INKLING2_AMD.md) while the
// GPU computes the others.  The arithmetic is maya-ds4's CPU rows for a GGUF's experts (src/ds4/ds4_iq_cpu.cpp:
// ggml-cpu's dots, an activation as Q8_K) with the SITU activation between the fused gate | up rows and down.  The
// thread pool is maya-hy4's (hy4::Lane).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "hy4_lane.hpp"

namespace ik2hip {

/// A layer's experts outside VRAM, by RAM slot r: below n_copy copied into RAM, at base + r blob - its fused gate | up
/// matrix (2 n_ff rows of n_embd) at 0, its down matrix (n_embd rows of n_ff) at tr_down; from n_copy on in place in
/// the model's files (--mmap-experts: the page cache, or the disk), the two matrices at fgu / fdn[r - n_copy]
struct CpuExperts {
    strata::ds4::gguf::IqCpu f;
    const uint8_t* base = nullptr;
    size_t blob = 0, tr_down = 0;
    int n_copy = 1 << 30;
    std::vector<const uint8_t*> fgu, fdn;
    float beta = 4.0f, lb = 25.0f;
    bool init(int tg, int td, int n_embd, int n_ff, std::string& err);
    const uint8_t* gu(int r) const { return r < n_copy ? base + (size_t) r * blob : fgu[(size_t) (r - n_copy)]; }
    const uint8_t* dn(int r) const { return r < n_copy ? base + (size_t) r * blob + tr_down : fdn[(size_t) (r - n_copy)]; }
};

struct CpuEntry {
    int slot = 0;   // the expert's RAM slot
    float w = 0.0f;
};

/// out [n_embd] = sum over the entries of w down(situ(gate x, up x)) (no entries: zeros); x [n_embd]
void cpu_experts(hy4::Lane& lane, const CpuExperts& ex, const std::vector<CpuEntry>& entries, const float* x, float* out);

/// A window's rows at once: entry (row, slot, w) - the rows' picks outside VRAM, each row's in its picks' order.  An
/// expert several rows picked is read once for all of them (ds4_iq_cpu's multi-token rows: each row's sums the ones it
/// gets alone).  x [T][n_embd], out [T][n_embd] (a row without entries: zeros).
struct CpuRowEntry {
    int row = 0, slot = 0;
    float w = 0.0f;
};
void cpu_experts_rows(hy4::Lane& lane, const CpuExperts& ex, const std::vector<CpuRowEntry>& entries, const float* x, int T,
                      float* out);

}  // namespace ik2hip
