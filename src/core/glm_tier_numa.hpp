// src/core/glm_tier_numa.hpp - the glm-dsa port B8.2 / B8.3 / B8.4 / B8.7: the memory tiers at run time.
//
// Owned by Glm5Model::FastState::tier (a shared_ptr made by the first pin_class of a load; null until then).  Every
// entry point does nothing (returns false, or falls through to Maya's call) when its feature is off, so glm5-next
// keeps Maya's path byte for byte:
//
//   pin_class      B8.2: a RAM class as one virtual range in segments of <= 16 GiB, mbind-ed (MPOL_INTERLEAVE as
//                  Maya, or the half's node with MPOL_BIND when split), madvise(MADV_HUGEPAGE) per segment while free
//                  2 MB blocks cover it (MAYA_GLM_THP), faulted in by node-pinned threads (2 MB strides in parallel
//                  with huge pages, else one thread per segment) and cudaHostRegister-ed - one thread per NUMA node, so
//                  segments register concurrently.  A refusal shrinks only the last segment.  A bound class (split,
//                  MAYA_GLM_NUMA_SEG=bind) is first capped by each node's free memory less MAYA_GLM_NODE_RESERVE_GB
//                  (MPOL_BIND has no fallback node): shrunk, or interleaved when < 3/4 fits; bound units' THP budget is
//                  their own node's free 2 MB blocks (W6 runtime fix).  Logs the AnonHugePages
//                  share of the tier (B8.7).  On glm-dsa by default; MAYA_GLM_RAM_SEG_GB=<GiB> (0 off) anywhere.
//   unpin_class    the matching free.
//   fetch/prefetch B8.4: the device's zero-copy pulls of a split expert (two half reads, re-interleaved into the VRAM
//                  slot); a source outside the split ranges (the disk staging) is copied as Maya does.
//   h2d / d2h      B8.4: promotions, the calibration's timing copies and demotions: 6 cudaMemcpy2DAsync for a split
//                  slot (glmtier::split_ops), Maya's one cudaMemcpyAsync otherwise.
//   read_split     B8.4: a disk read (warm-up, a miss, a lookahead) into a split slot goes through a bounce buffer and
//                  scatter_range; false (the caller reads as before) for any other destination.
//   lane_pools     B8.3 (MAYA_GLM_CPU_SHARED=socket) and B8.4 (node-split): the CPU lane's per-node pinned pools.
//   cpu_experts    B8.4: the lane on split experts, both sockets at once, partial sums added.
#pragma once

#include "glm_fast_state.hpp"
#include "glm_tier_lane.hpp"
#include "glm_tier_plan.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::core::glmtier {

/// One RAM class pinned by pin_class.
struct TierClass {
    bool ours = false;           ///< pinned here (else Maya's numa_pinned / cudaHostAlloc, untouched)
    bool split = false;          ///< B8.4 layout: half 1 of a slot at its half 0 + D
    uint8_t* map = nullptr;      ///< the reservation (half 0 region; half 1 at map + D)
    size_t D = 0;                ///< planned slots x stride (the half-1 offset; 0 unsplit)
    size_t stride = 0;           ///< bytes a slot takes in one region (split: one half)
    int n = 0;                   ///< slots pinned
    std::vector<std::pair<size_t, size_t>> regs;    ///< registered [offset, bytes) from map
    std::vector<std::pair<size_t, size_t>> mapped;  ///< still-mapped [offset, bytes) from map (munmap on free)
};

/// The CPU lane's pool on one NUMA node, with a driver thread that runs a task on it while the caller does other work
/// (the node-split lane drives both sockets at once).  Shared by every part of a load (registry by node and size).
struct NodeLane {
    NodeLane(int node, std::vector<int> cpus, int threads);
    ~NodeLane();
    void start(const std::function<void()>* task);   ///< run *task on the driver thread
    void wait();                                     ///< until it returned
    int node;
    std::vector<int> cpus;
    std::shared_ptr<glmfast::Workers> pool;          ///< threads - 1 workers (the driver is the last)
  private:
    void loop();
    std::thread th_;
    std::atomic<uint64_t> gen_{0}, done_{0};
    std::atomic<bool> quit_{false};
    const std::function<void()>* task_ = nullptr;
};

struct TierState {
    bool ready = false;         ///< filled by tier_of (pin_class or lane_pools, whichever came first)
    bool dsa = false;
    bool split_req = false;     ///< MAYA_GLM_NUMA_SPLIT=1 on glm-dsa
    size_t seg_cap = 0;
    int dev = 0;
    std::vector<int> nodes;
    std::vector<TierClass> cls;               ///< by RAM class
    std::vector<SplitLayout> L;               ///< by absolute layer (blob == 0: not a split layer)
    std::vector<kc::NativeFmt> half;          ///< by absolute layer: the lane's half format (n_ff == 0: none)
    std::vector<int> layer_cls;               ///< layer -> RAM class (copy of FastState::layer_rc)
    LanePool mode = LanePool::One;
    std::shared_ptr<NodeLane> lane[2];        ///< node-split: node 0 / node 1 pools
    std::shared_ptr<NodeLane> socket_lane;    ///< Socket mode: this part's node pool, held for the load (shared)
    // node-split lane scratch, one set per node (written by that node's threads)
    std::vector<uint8_t> act;
    std::vector<float> ff[2], dn[2], part[2];
    std::vector<uint8_t> hq[2];
    std::atomic<uint64_t> split_calls{0};
    bool split_any() const;
};

/// B8.2: pin RAM class c of F (R = F.rc[c]) with n slots (shrunk on a refusal, never below least).  Returns the base
/// pointer (and sets R.registered, R.stride, R.split_d) or nullptr: then Maya's loop runs as before.
void* pin_class(Glm5Model::FastState& F, const Glm5Geometry& g, int c, int64_t& n, int64_t least, int dev);
/// Frees a class pin_class pinned; false when it was not one of ours.
bool unpin_class(TierState* T, int c);

/// B8.4 device pulls.  fetch: moe_fetch's pulls of layer il (true: done here).  prefetch: moe_prefetch's for layer il1.
bool fetch(TierState* T, const gf::MoeDev& d, int il, int k, cudaStream_t s);
bool prefetch(TierState* T, const gf::MoeDev& d, int il1, cudaStream_t s);
/// B8.4 host-issued copies between a RAM slot (its half 0 pointer) and a unified VRAM blob of layer il.
cudaError_t h2d(TierState* T, int il, void* dst, const void* slot, size_t blob, cudaStream_t s);
cudaError_t d2h(TierState* T, int il, void* slot, const void* src, size_t blob, cudaStream_t s);
/// B15.8: h2d of `slot` is one plain cudaMemcpyAsync (not a node-split class), so adjacent slots may share a copy.
bool h2d_plain(const TierState* T, int il, const void* slot);
/// B8.4: a disk read of unified bytes [u0, u0 + len) of layer il's expert whose destination slot (half 0) is `slot`:
/// read(buf) fills buf with those bytes; scattered into the halves.  False: not a split slot (the caller reads).
bool read_split(TierState* T, int il, uint8_t* slot, size_t u0, size_t len, const std::function<void(uint8_t*)>& read);
/// B8.4: a RAM slot's unified bytes: `slot` itself when it is not split, else both halves gathered into buf.
const uint8_t* unified(const TierState* T, int il, const uint8_t* slot, std::vector<uint8_t>& buf);
/// Whether layer il's RAM tier is split (the lane check and other unified-only debug paths step aside).
bool split_layer(const TierState* T, int il);
/// B9.9: a split layer's half-1 offset from a slot's half-0 pointer (0 when layer il is not split).
size_t split_offset(const TierState* T, int il);
/// B9.9: the lock every user of the node-split lanes holds for a whole call (the decode's cpu_experts and the
/// prompt's node-split host share; a load's parts share the lanes, and they can run at once).
std::mutex& lane_mutex();

/// B8.3 / B8.4: the CPU lane's pools.  Picks the mode, logs it, and for Socket / NodeSplit makes (or finds) the
/// per-node pools: sets F.cpu_pool, F.cpu_node, F.cpu_pin and returns true; false leaves Maya's code to make its pool.
/// threads: the count Maya derived (used as the per-node count when explicit, lane_env != nullptr); gpu_node: this
/// part's GPU's NUMA node (-1 unknown).
bool lane_pools(Glm5Model::FastState& F, const Glm5Geometry& g, int n_parts, int part, bool own_count,
                bool lane_env, int threads, int gpu_node, int dev);
/// B8.3 (W6 runtime fix): the per-socket lane of NUMA node `node` with `threads` threads, kept alive by F's tier state
/// (made here when no class was pinned) so every part of the load on that node gets the same NodeLane and pool.
std::shared_ptr<NodeLane> socket_lane(Glm5Model::FastState& F, const Glm5Geometry& g, int node, int threads, int dev);
/// B8.4 (W6 runtime fix): the per-request cpu_threads knob on a node-split lane: node 0's pool gets n active threads
/// (0: all) and node 1's the same share of its own (lane_threads_scaled).  False when the lane is not node-split
/// (Maya's set_active on F.cpu_pool runs).
bool set_lane_threads(Glm5Model::FastState& F, int n);
/// B8.4: the lane on split experts (blob[i]: half-0 pointers).  False: not split (Maya's lane runs).
bool cpu_experts(Glm5Model::FastState& F, const Glm5Geometry& g, int il, int ne, const uint8_t* const* blob,
                 const float* w, const float* x, float* out);

}  // namespace strata::core::glmtier
