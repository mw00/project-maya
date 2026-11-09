// src/core/glm_tier_plan.hpp - the glm-dsa port B8.2 / B8.3 / B8.4 / B8.7: the memory tiers' host-side rules.
//
// Plain C++ (no CUDA, no ggml): the arithmetic that src/core/glm_tier_numa.cu (the pinning, the transfers, the split
// CPU lane) runs on, kept apart so the CPU tests (tests/glm_dsa/glm_tier_plan_test.cpp) check it without a device.
//
//   B8.2  segmented pinning: a RAM class is one virtual range cut in segments of at most 16 GiB (a whole number of
//         slots each), each mbind-ed and cudaHostRegister-ed on its own and in parallel per NUMA node.  A refused
//         registration shrinks only the LAST segment (Maya's n*31/32 x4 then n*7/8 steps, applied to the class total).
//   B8.3  the CPU lane's pool mode: Maya's shared / per-part A/B (STRATA_GLM_CPU_SHARED=1/0) plus per-socket
//         (STRATA_GLM_CPU_SHARED=socket): one pinned pool per NUMA node, serving the parts whose GPU hangs off it.
//   B8.4  the NUMA node-split expert layout (STRATA_GLM_NUMA_SPLIT=1, glm-dsa only, default off): a RAM-tier expert is
//         two halves, half 0 on node A and half 1 on node B.  Half h holds gate rows [h*H, (h+1)*H), up rows
//         [h*H, (h+1)*H) and, for every down row, the inner columns [h*H, (h+1)*H) (H = n_ff / 2; 1024 at GLM-5.3,
//         so a down half row is 4 of its 8 256-blocks).  Half 1 sits at half 0 + D, one constant per class.
//   B8.7  THP: the knob (STRATA_GLM_THP), the free 2 MB block count from /proc/buddyinfo, and the AnonHugePages share
//         of the tier read back from /proc/self/smaps.
#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GLMTIER_HD __host__ __device__
#else
#define GLMTIER_HD
#endif

namespace strata::core::glmtier {

// ---------------------------------------------------------------- B8.2 segments

/// The default segment cap: 16 GiB (a cudaHostRegister of 120-140 GB in one call took minutes and stalls the others).
constexpr size_t kSegCapDefault = (size_t) 16 << 30;

/// The segment cap in bytes for this arch: STRATA_GLM_RAM_SEG_GB (0 = off: Maya's single allocation) or, unset,
/// 16 GiB on glm-dsa and off on glm5-next (Flash stays on Maya's path).
size_t seg_cap_bytes(bool glm_dsa, const char* env);

/// A class cut in segments: seg_slots slots a segment (the last one may hold fewer); slots[k] per segment.
struct SegPlan {
    size_t slot_bytes = 0;      ///< the bytes one slot takes in ONE segment (a split class: one half)
    int seg_slots = 0;          ///< slots per full segment (>= 1)
    std::vector<int> slots;     ///< per segment; sum = n
    int64_t total() const;
};

/// n slots of slot_bytes in segments of at most cap bytes (at least one slot a segment).
SegPlan seg_plan(int64_t n, size_t slot_bytes, size_t cap);

/// The shrink of a refused class: Maya's steps on the class total (tries < 4: n*31/32, then n*7/8).
int64_t shrink_step(int64_t n, int tries);

/// Pin a planned class.  first_round(ok) pins every segment at its planned size (the caller runs them in parallel,
/// one node-pinned thread per NUMA node) and sets ok[k]; then, at the first segment that failed, every later one is
/// released (release(k, slots)) and that one alone is retried by pin(k, slots) with the class total shrinking by
/// shrink_step - only the LAST segment ever shrinks.  Returns the total pinned, its per-segment slots in `pinned` (a
/// prefix of whole segments plus the shrunk last one), or -1 (everything released) when it fell below `least`.
int64_t seg_pin(const SegPlan& plan, int64_t least, const std::function<void(std::vector<char>&)>& first_round,
                const std::function<bool(int, int)>& pin, const std::function<void(int, int)>& release,
                std::vector<int>& pinned);

/// The NUMA node a segment's work runs on (its node-pinned thread): round-robin over the nodes for an interleaved
/// class, the half's node for a split class (half 0 -> nodes[0], half 1 -> nodes[1]).
int seg_node(int k, int half, bool split, const std::vector<int>& nodes);

/// W6 runtime fix (B8.2): a NUMA node's free memory in bytes from its /sys/devices/system/node/nodeN/meminfo text:
/// MemFree + Inactive(file) + SReclaimable (what a bound allocation can take without swapping or an OOM kill);
/// -1 when MemFree was not found.
int64_t node_mem_avail(const std::string& node_meminfo);
/// The bytes kept free on every node a bound unit lands on (STRATA_GLM_NODE_RESERVE_GB, default 4 GiB: the OS, the
/// server, the driver's pinned buffers and the B9.9 landing rings bound to the GPUs' node).
int64_t node_reserve_bytes(const char* env);
/// The most slots (<= n) of a class whose units are bound to nodes (MPOL_BIND: no fallback) that fit every node's
/// budget: split, n x stride on each of nodes[0] / nodes[1]; else (STRATA_GLM_NUMA_SEG=bind) seg_plan(n, stride, cap)'s
/// segments round-robin over the nodes (seg_node).  budget[j] is nodes[j]'s free bytes less the reserve (< 0 = unknown:
/// no limit).  0 when not one slot fits.
int64_t node_fit_slots(int64_t n, size_t stride, size_t cap, bool split, const std::vector<int>& nodes,
                       const std::vector<int64_t>& budget);
/// What pin_class does with a bound class that does not fit its nodes: 0 keep n (it fits), 1 shrink to fit (the fit
/// keeps at least 3/4 of n and at least `least`), 2 drop the binding (interleave, which falls back across nodes).
int node_fit_action(int64_t n, int64_t fit, int64_t least);

/// Whether the budget statics (fast_setup's total / remaining / remaining_layers / unpinned) start over: at the first
/// part of every load (a reload reused the first load's measurement, its remaining budget already spent).
inline bool budget_reset_due(int part) { return part == 0; }

// ---------------------------------------------------------------- B8.7 huge pages

/// STRATA_GLM_THP: 0 never madvise(MADV_HUGEPAGE), 1 always, unset/other: auto (Maya's rule: only while the free
/// 2 MB blocks cover the segment plus 1 GiB).  Returns 0 / 1 / -1 (auto).
int thp_mode(const char* env);
/// The free bytes in blocks of order >= 9 (2 MB on 4 KB pages) in a /proc/buddyinfo text; -1 when none parsed.
int64_t buddy_huge_free(const std::string& buddyinfo);
/// Per segment (bytes each, in order): madvise huge pages or not.  Auto: the free huge bytes are handed out in order,
/// a segment taking them only while they cover it plus 1 GiB.
std::vector<char> thp_plan(const std::vector<size_t>& seg_bytes, int64_t huge_free, int mode);
/// W6 runtime fix (B8.7): the free bytes in blocks of order >= 9 of each node in `nodes` ("Node N, zone ..." lines);
/// -1 for a node with no line (unknown).
std::vector<int64_t> buddy_huge_free_nodes(const std::string& buddyinfo, const std::vector<int>& nodes);
/// thp_plan for units bound to nodes: unit_node[k] is the index into node_free of the node unit k is bound to (its
/// huge pages can come only from there), or -1 for an interleaved unit (the machine-wide `all_free`).  Each node's
/// free huge bytes are handed out in order to its own units, as thp_plan does.
std::vector<char> thp_plan_nodes(const std::vector<size_t>& unit_bytes, const std::vector<int>& unit_node,
                                 const std::vector<int64_t>& node_free, int64_t all_free, int mode);
/// AnonHugePages and Rss (bytes) of the mappings of a /proc/<pid>/smaps text that lie inside any of `ranges`
/// ([start, start + len)).  A mapping counts when it starts inside a range.
struct SmapsSum {
    int64_t anon_huge = 0, rss = 0, size = 0;
};
SmapsSum smaps_sum(const std::string& smaps, const std::vector<std::pair<uintptr_t, size_t>>& ranges);

/// Fault-in plan of a segment: with huge pages, `threads` node-pinned threads touch it in 2 MB strides (thread t
/// takes strides t, t + threads, ...); without, one thread touches every 4 KB page.  Returns the [begin, end) byte
/// ranges thread t touches, in 2 MB pieces (or one piece for the single thread).
std::vector<std::pair<size_t, size_t>> fault_plan(size_t bytes, bool huge, int threads, int t);

// ---------------------------------------------------------------- B8.3 pool modes

enum class LanePool : int {
    One = 0,        ///< one GPU: one pool (Maya)
    PerPart = 1,    ///< STRATA_GLM_CPU_SHARED=0 (or a 2-part split by default): one pool per part (Maya)
    Shared = 2,     ///< STRATA_GLM_CPU_SHARED=1 (default with 3+ parts): one unpinned pool for every part (Maya)
    Socket = 3,     ///< STRATA_GLM_CPU_SHARED=socket: one pinned pool per NUMA node, shared by that node's parts
    NodeSplit = 4,  ///< STRATA_GLM_NUMA_SPLIT=1: one pinned pool per node, each computing its half of every expert
};
/// The mode a part's lane uses.  own_count: STRATA_GLM_CPU_LANE<dev> set (Maya: a pool of its own).  Maya's rules are
/// unchanged for unset / 0 / 1; on glm-dsa (`dsa`) "socket" (or "2") selects Socket; a node-split tier forces
/// NodeSplit.  Flash (dsa false) keeps Maya's reading of every value: "2" is one shared pool (any non-zero number),
/// "socket" a pool per part (atoi 0).  Inline: the split planner (glm_dsa_split.cpp glm_cpu_pool_ids) prices the same
/// rule without linking this file.
inline LanePool lane_pool_mode(int n_parts, bool own_count, const char* shared_env, bool numa_split, int n_nodes,
                               bool dsa = true) {
    if (numa_split && n_nodes >= 2) return LanePool::NodeSplit;
    if (n_parts < 2) return LanePool::One;
    if (own_count) return LanePool::PerPart;
    if (shared_env != nullptr && shared_env[0] != '\0') {
        if (dsa && (std::strcmp(shared_env, "socket") == 0 || std::strcmp(shared_env, "2") == 0))
            return n_nodes >= 2 ? LanePool::Socket : LanePool::Shared;
        return std::atoi(shared_env) != 0 ? LanePool::Shared : LanePool::PerPart;
    }
    return n_parts >= 3 ? LanePool::Shared : LanePool::PerPart;
}
const char* lane_pool_name(LanePool m);
/// The default thread count of a node's pinned lane pool (Socket / NodeSplit, STRATA_GLM_NUMA_LANE_THREADS unset) from
/// the node's usable CPUs: a thread per PHYSICAL core less 2 (Maya's lane is a thread per physical core, and it measured
/// SMT oversubscription as a loss: 5900X 10 threads on 6 cores 10.88 tok/s, 6 threads 11.45); the cores left free hold
/// the service, main and disk-reader threads.  test-PC node (20C/40T a node): 18.  Physical cores unknown (0): logical - 4, the
/// earlier rule.  Within [2, 64].
inline int numa_lane_threads(int logical_cpus, int physical_cores) {
    const int t = physical_cores > 0 ? physical_cores - 2 : logical_cpus - 4;
    return t < 2 ? 2 : t > 64 ? 64 : t;
}
/// W6 runtime fix (B8.4): the active thread count of a node-split lane pool of `size` threads for the per-request
/// cpu_threads knob n (0: all) given as a count for the reference pool (node 0) of `ref_size` threads: n scaled by
/// size / ref_size, rounded, within [1, size].  Both halves of every expert then shrink alike.
int lane_threads_scaled(int n, int ref_size, int size);
/// Maya's `shared` flag for a mode (Shared only): the Maya code paths keyed on it stay as they were.
inline bool lane_pool_maya_shared(LanePool m) { return m == LanePool::Shared; }

// ---------------------------------------------------------------- B8.4 node-split layout

/// STRATA_GLM_NUMA_SPLIT: true only for "1"-like values; never on glm5-next.
bool numa_split_requested(bool glm_dsa, const char* env);

/// One layer's split geometry (bytes).  Unified blob: [gate n_ff x gu_row | up n_ff x gu_row | down n_embd x d_row].
/// Half h: [gate rows h*H.. (H x gu_row) | up rows h*H.. (H x gu_row) | down n_embd x hd_row (columns h*H..)].
struct SplitLayout {
    size_t gu_row = 0, d_row = 0;   ///< unified row bytes
    int64_t n_ff = 0, n_embd = 0, H = 0;
    size_t gu_bytes = 0;            ///< n_ff * gu_row
    size_t down_off = 0;            ///< 2 * gu_bytes
    size_t blob = 0;                ///< unified bytes
    size_t h_gu = 0;                ///< H * gu_row
    size_t h_down_off = 0;          ///< 2 * h_gu
    size_t hd_row = 0;              ///< d_row / 2
    size_t half_bytes = 0;          ///< h_down_off + n_embd * hd_row
};
/// d_blck / d_blck_bytes: the down type's block (elements, bytes).  False (with why) unless n_ff is even, H a whole
/// number of down blocks, d_row = (n_ff / d_blck) * d_blck_bytes, and gu_row and hd_row multiples of 4 (the device
/// gathers 4-byte words).
bool split_layout(size_t gu_row, size_t d_row, int64_t n_ff, int64_t n_embd, int64_t d_blck, size_t d_blck_bytes,
                  SplitLayout& L, std::string& why);

/// Where unified byte u lives: half and offset in that half.
struct SplitLoc {
    int half;
    size_t off;
};
GLMTIER_HD inline SplitLoc split_loc(const SplitLayout& L, size_t u) {
    if (u < L.down_off) {
        const size_t sec = u < L.gu_bytes ? 0 : 1;   // gate / up
        const size_t v = u - sec * L.gu_bytes;
        const size_t r = v / L.gu_row, c = v - r * L.gu_row;
        const int h = r >= (size_t) L.H ? 1 : 0;
        return SplitLoc{h, sec * L.h_gu + (r - (size_t) h * (size_t) L.H) * L.gu_row + c};
    }
    const size_t v = u - L.down_off;
    const size_t r = v / L.d_row, c = v - r * L.d_row;
    const int h = c >= L.hd_row ? 1 : 0;
    return SplitLoc{h, L.h_down_off + r * L.hd_row + (c - (size_t) h * L.hd_row)};
}

/// The inverse: unified offset of byte `off` of half h.
GLMTIER_HD inline size_t unsplit_off(const SplitLayout& L, int h, size_t off) {
    if (off < L.h_down_off) {
        const size_t sec = off >= L.h_gu ? 1 : 0;
        return sec * L.gu_bytes + (size_t) h * L.h_gu + (off - sec * L.h_gu);
    }
    const size_t v = off - L.h_down_off, r = v / L.hd_row, c = v - r * L.hd_row;
    return L.down_off + r * L.d_row + (size_t) h * L.hd_row + c;
}
/// The widest word (16, 8 or 4 bytes) that every run of the layout is a whole number of: the device gathers in it.
inline int split_word(const SplitLayout& L) {
    for (int w = 16; w > 4; w /= 2)
        if (L.gu_row % (size_t) w == 0 && L.hd_row % (size_t) w == 0) return w;
    return 4;
}

/// The copies that move a whole expert between the unified and the split layout: 2-D (rows of width bytes at the given
/// pitches; a 1-D run is rows = 1).  Six of them: gate half 0/1, up half 0/1 (contiguous runs) and down half 0/1
/// (n_embd rows of hd_row bytes, pitch d_row on the unified side and hd_row on the half side).  The same list drives
/// the host memcpy (scatter / gather) and the device's cudaMemcpy2DAsync promotions and demotions.
struct CopyOp {
    int half;
    size_t u_off, h_off, width, rows, u_pitch, h_pitch;
};
std::vector<CopyOp> split_ops(const SplitLayout& L);
/// unified -> halves (h0, h1), and back.
void scatter(const SplitLayout& L, const uint8_t* u, uint8_t* h0, uint8_t* h1);
void gather(const SplitLayout& L, const uint8_t* h0, const uint8_t* h1, uint8_t* u);
/// [u0, u0 + len) of a unified expert (src holds those bytes) into its halves: a chunk of a disk read.
void scatter_range(const SplitLayout& L, size_t u0, const uint8_t* src, size_t len, uint8_t* h0, uint8_t* h1);

/// The per-slot stride of one half of a class whose layers have these layouts: the largest half rounded to 4 KB.
size_t half_stride(const std::vector<SplitLayout>& layers);

/// The partial sums' combine of the split lane: y[r] = y0[r] + y1[r] (one n_embd add per layer).
void add_partial(const float* y0, const float* y1, float* y, int64_t n);

}  // namespace strata::core::glmtier
