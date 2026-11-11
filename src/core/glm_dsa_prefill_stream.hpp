// src/core/glm_dsa_prefill_stream.hpp - the glm-dsa port B9.9: the prompt's expert streaming over three PCIe links.
//
// Maya's prompt path (glm_prefill.cu) already streams each part's own layers' RAM-tier experts over its own link on
// its own copy stream: the staged groups (a ring of NG = 3 groups of GE = 4 slots) and the next MoE layer's
// prestage buffer (the copy stream fills it while a layer's mixer runs) are its double buffer, and a split's parts
// pipeline chunks.  What a 3-GPU glm-dsa split lacked, and this driver adds (all gated on glm-dsa; Flash is Maya's
// path byte for byte):
//
//   1. a prestage buffer on every part of a glm-dsa split (Maya gives one to a single GPU only: a Flash split holds
//      most of its experts; GLM-5.3 on the test PC holds about a third) - split_prestage();
//   2. the next parts pre-stage their first MoE layer while the parts before them compute: before part 0 starts a
//      chunk, every later part that is idle (the serial schedule: every chunk; the chunk pipeline: the first chunk,
//      before the later parts' threads start) runs its prefill_half up to its first prestage and returns
//      (prestage_only); its real pass then skips that prestage (early_take) - the same copies, issued a part's
//      compute earlier, so part i+1's link works while part i computes;
//   3. the disk landing ring (the pinned staging the reader lands experts in) bound to the GPU's NUMA node
//      (host_alloc_node: mmap + mbind(MPOL_BIND) + fault-in + cudaHostRegister), so the DMA never crosses UPI;
//   4. the prompt's CPU host share on node-split layers (MAYA_GLM_NUMA_SPLIT=1, where Maya's share was turned off):
//      both sockets compute their half of each expert for every routed row, partial outputs added (split_share).
//   The B14.7 lane-first rule (glmpf::lane_first) applies unchanged to the share's plan, split layers included.
//
// Env: MAYA_GLM_PREFILL_STREAM=0 turns 1 and 2 off; MAYA_GLM_STAGE_NUMA=off / <node> (the landing ring's
// node, default the GPU's); MAYA_GLM_PREFILL_SPLIT_SHARE=0 keeps split layers off the host share (the B8.4 behaviour).
//
// The pure parts (schedules, the registry, the node choice, the share's job grid and row math) are in
// glm_dsa_prefill_stream_plan.cpp (CPU tests: tests/glm_dsa/glm_dsa_prefill_stream_test.cpp); the run-time side
// (CUDA runtime, NUMA, the node lanes, the Glm5Model member) in glm_dsa_prefill_stream.cpp.
#pragma once

#include "glm_tier_plan.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace strata::core {
struct Glm5Geometry;
}

namespace strata::core::glmtier {
struct TierState;
}

namespace strata::core::glmpfs {

namespace kc = strata::kernels::cpu;

// ---------------------------------------------------------------- the schedule (pure)

/// The driver is on: glm-dsa and MAYA_GLM_PREFILL_STREAM unset or non-zero.
bool stream_on(bool dsa);
/// A part of an n-part glm-dsa split gets the prestage buffer (n > 1 and stream_on).
bool split_prestage(bool dsa, int n_parts);
/// The parts (1 .. n-1) that pre-stage their first MoE layer before part 0 starts chunk c: every later part on the
/// serial schedule, only on chunk 0 under the chunk pipeline (later, each part is busy with the chunk before).
std::vector<int> early_parts(int n_parts, int chunk, bool pipelined);
/// The NUMA node the landing ring binds to: env "off" none (-1); a number: that node when it exists (else -1); unset
/// (or anything else): the GPU's node when it is known and the host has more than one node; else -1.
int staging_node(int gpu_node, int n_nodes, const char* env);

/// The early prestage registry (process-wide, thread-safe): the driver records (model, p0, T) after a part's
/// prestage-only pass; that part's real pass takes the record once and skips its first prestage.
void early_record(const void* model, int64_t p0, int T);
bool early_take(const void* model, int64_t p0, int T);
void early_clear();
/// Drops a model's unclaimed records (a prompt that ended between a part's early pass and its real one).
void early_forget(const void* model);
/// The prestage-only pass of `model` on this thread (prefill_half returns after its first prestage).
void set_prestage_only(const void* model);
bool prestage_only(const void* model);

/// B14.7 in the prestage: a predicted expert the prompt plan would leave on the CPU lane (glmpf::lane_first: fewer
/// predicted rows than the measured upload threshold) is not prestaged, so the link carries only what the plan
/// uploads.  copy_ms / ps_ms / p_ms: the plan's PCIe estimate in its order (the prestage copies' measured time, the
/// calibration's streamed copy, its single copy); c_ms: the lane's one-token expert; row_ms: the lane's cost a row
/// (MAYA_GLM_PREFILL_CPU_ROW_MS overrides it, as it does in the plan);
/// lane_ok: the layer's experts may go to the lane (the CPU share is on for it, split layers included).  Always false
/// off glm-dsa, without a lane, or with MAYA_GLM_PREFILL_CPU=0.
bool prestage_skip(bool dsa, double pred_rows, double copy_ms, double ps_ms, double p_ms, double c_ms, double row_ms,
                   bool lane_ok);

// ---------------------------------------------------------------- B11.1: the NextN rows on the token path (pure)

/// prefill()'s batched-path minimum: MAYA_GLM_PREFILL_MIN (at least 1) when set, else 32.
int64_t prefill_min_tokens(const char* env);
/// Whether Glm5Model::prefill hands a prompt to prefill_tokens_dsa (glm_dsa_mixer.cu): a glm-dsa model with a NextN
/// block whose prompt the batched path does not take - some part has no batched state (--prefill 0,
/// MAYA_GLM_NO_PREFILL, a weight type the prompt path does not cover, a tail lent to the vision encoder) or fewer
/// than min_n tokens.  prefill_tokens_dsa runs the NextN block at every prompt position, so its latent, k_pe and
/// index-key rows cover the prompt as the batched path's mtp_fill does.  False on Flash and without a NextN block.
bool mtp_token_fill(bool dsa, bool has_mtp, bool batched_ready, int64_t n_tokens, int64_t min_n);
/// Whether prefill_chunk() reports 1 (so generate hands the prompt to prefill) where Maya's would report 0: the head
/// part (layer 0) of a glm-dsa model with a fast path and a NextN block but no batched state of its own.
bool mtp_token_chunk(bool dsa, bool head, bool has_fast, bool has_mtp, bool has_pf);

// ---------------------------------------------------------------- the node-split host share (pure)

/// One expert of a layer's host share: its slot (half 0), its rows and the first of them in the CPU segment.
struct ShareX {
    const uint8_t* half0 = nullptr;
    int rows = 0;
    int r0 = 0;
};
/// The job grid Maya's share uses: job -> (expert j = job % n_exp, rows [blk * block, +block) of `total`).
inline int share_jobs(int n_exp, int total, int block) { return n_exp * ((total + block - 1) / block); }
inline void share_job(int job, int n_exp, int block, int total, int& j, int& r0, int& r1) {
    j = job % n_exp;
    r0 = (job / n_exp) * block;
    r1 = r0 + block < total ? r0 + block : total;
}
/// A pool's run(n, fn) (the node lanes' Workers, or a serial loop in tests).
using RunFn = std::function<void(int, const std::function<void(int)>&)>;
/// Scratch of one node's share (grown on demand).
struct ShareScratch {
    std::vector<float> ff;      ///< rows x H
    std::vector<uint8_t> hq;    ///< rows x kNativeHBytes
    std::vector<float> part;    ///< rows x n_embd
};
/// Node h's share of a split layer: for every routed row of every expert, the half's gate/up rows (H values), its down
/// activation, and the half-columns down product into sc.part (rows x n_embd, row r of the CPU segment).  half_off: 0
/// for node 0, the class's half-1 offset for node 1.  act: rows x kNativeActBytes (the full format's activation; the
/// gate/up rows are the same).  Gate/up rows equal the unsplit lane's bit for bit; the two partials add to the
/// unsplit down within one float reassociation (glm_tier_lane.hpp).
void share_node(const kc::NativeFmt& hf, const glmtier::SplitLayout& L, int n_exp, const ShareX* ex, size_t half_off,
                const uint8_t* act, int rows, float swiglu_limit, ShareScratch& sc, const RunFn& run);
/// out[r] = p0[r] + p1[r] for rows x n_embd floats (the order add_partial uses).
void share_add(const float* p0, const float* p1, float* out, size_t n, const RunFn& run, int rows);

// ---------------------------------------------------------------- run time (glm_dsa_prefill_stream.cpp)

/// cudaHostAlloc for the prompt's disk landing ring; on glm-dsa bound to dev's NUMA node (staging_node) through
/// mmap + mbind + cudaHostRegister.  Returns a cudaError_t value (0 = cudaSuccess), so Maya's halving loop stays.
int host_alloc_node(void** p, size_t bytes, int dev, bool dsa);
/// The matching free (cudaFreeHost for a cudaHostAlloc-ed block).
void host_free_node(void* p);
/// Whether layer il may give experts to the host share: not node-split, or split with both node lanes and the half
/// format ready (MAYA_GLM_PREFILL_SPLIT_SHARE != 0).
bool share_ok(const glmtier::TierState* T, int il);
/// The host share of a node-split layer (false: layer il is not split; the caller runs Maya's share).  ex(j) gives
/// expert j of n_exp; x_h + rt_h[r] * n_embd is row r's input; out_h + r * n_embd its output.  Holds
/// glmtier::lane_mutex() and runs both node lanes at once.
bool split_share(glmtier::TierState* T, const kc::NativeFmt& nf, const Glm5Geometry& g, int il, int n_exp,
                 const std::function<ShareX(int)>& ex, const float* x_h, const int* rt_h, int rows, float* out_h);

}  // namespace strata::core::glmpfs
