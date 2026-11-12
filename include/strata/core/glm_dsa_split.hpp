// include/strata/core/glm_dsa_split.hpp - the glm-dsa port B6.1-B6.3: the glm-dsa layer split's pure host-side rules.
//
// Maya's split search (glm_model.cu glm_search_bounds), its default bounds (load_pack_env) and the fast path's CPU-lane
// placement (glm_fast_path.cu part_cpus) were written for Flash: any layer can start a part, the NextN block is priced
// only on two GPUs, a miss is priced at host_bps / n, and three parts on two NUMA nodes go round-robin.  glm-dsa needs:
//
//   * B6.1: parts start only on full-indexer layers (Glm5Geometry::boundary_ok): the auto search skips the rest, the
//     default bounds snap to the nearest full layers, and a refusal names the nearest valid layers;
//   * B6.2: the per-layer decode state (glm_dsa_state_bytes; the engine prices it through the B4.1 descriptor,
//     GlmStateLayout::bytes_per_layer, and glm_dsa_split_test checks the two agree on every layer), the
//     NextN block priced on the last part for any n >= 2, and a miss priced over the ONE shared CPU pool that a 3+ GPU
//     split runs by default (the parts take turns on a token, so the pool's whole bandwidth serves each part's misses);
//   * B6.3: with MAYA_GLM_CPU_SHARED=0, three parts on two NUMA nodes grouped by their GPU's node.
//
// Every rule leaves a Flash (glm5-next) geometry exactly as Maya has it.  Nothing here touches CUDA: the functions are
// checked on the CPU by tests/glm_dsa/glm_dsa_split_test.cpp.
#pragma once

#include "strata/core/glm_model.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

// ---------------------------------------------------------------- B6.2: the decode state of one layer

/// Bytes ONE position of layer `layer` keeps in the glm-dsa decode state:
///   * the latent row of every attention layer (the trunk's 78 and the NextN block): Maya's kv_lora record (FP16
///     2 x 512 = 1024 B, or the INT8 record lat8_rec_bytes(512) = 544 B with MAYA_GLM_KV_INT8) plus a separate FP16
///     k_pe of qk_rope (2 x 64 = 128 B) - 1152 B (INT8: 672 B, 16-B aligned); `f32_latent` (the slow reference path)
///     keeps the whole 576-float row in F32 (2304 B);
///   * the F32 index keys idx_key x 4 = 512 B on the full-indexer layers (Glm5Geometry::idx_full: 21 trunk layers + the
///     NextN block); the shared layers reuse the previous full layer's top-k and keep none.
/// No ring, ig or pool rows (those are Flash's k-pool indexer).  0 for a layer outside [0, n_layers + nextn) or a
/// Flash geometry (Maya's own formula prices Flash).  BUILD-PLAN section 2: 79 x 1152 + 22 x 512 = 102,272 B/token.
int64_t glm_dsa_state_row_bytes(const Glm5Geometry& g, int layer, bool int8, bool f32_latent = false);

/// glm_dsa_state_row_bytes x ctx: the bytes layer `layer` holds for a context of `ctx` positions (32K FP16: 79 x 1152
/// x 32768 + 22 x 512 x 32768 = 3.35 GB in all).  The engine prices the state through the B4.1 descriptor
/// (GlmStateLayout::bytes_per_layer, src/core/glm_state_layout.*); this formula is the tests' independent statement of
/// it (glm_dsa_split_test asserts the two agree for FP16 and INT8 carves).
double glm_dsa_state_bytes(const Glm5Geometry& g, int layer, int64_t ctx, bool int8, bool f32_latent = false);

// ---------------------------------------------------------------- B6.1: full-indexer boundaries

/// Every boundary is a layer a part may start on (boundary_ok); silent, for the search's inner loop.  Flash: true.
bool glm_split_bounds_full(const Glm5Geometry& g, const std::vector<int>& bounds);

/// The nearest layers a part may start on below and above `il` (each -1 when there is none); `il` itself when it is
/// valid.  glm-dsa: 27 -> 26 and 30.
void glm_nearest_boundaries(const Glm5Geometry& g, int il, int& below, int& above);

/// The nearest valid boundaries of `il` as text: "26 or 30", "26" (none above), "" (none at all).
std::string glm_nearest_boundaries_text(const Glm5Geometry& g, int il);

/// `bounds` (rising, one per later part) moved to the closest rising set of valid boundaries (least total distance,
/// ties to the lower layers); unchanged for Flash or when every bound is already valid, and returned unchanged when
/// there are fewer valid layers than bounds (glm_split_bounds_ok then refuses it with its message).
std::vector<int> glm_snap_bounds(const Glm5Geometry& g, const std::vector<int>& bounds);

// ---------------------------------------------------------------- B6.2: planner terms

/// `--layer-split auto`'s search (glm_model.cu glm_search_bounds) on the pack in `pack_dir` for the devices `devs`:
/// the later parts' first layers, empty when it has no answer (then load_pack splits by free VRAM).  On a glm-dsa pack
/// MAYA_GLM_SPLIT_CARDS="<free GiB>:<GB/s>[:<name>],..." (one entry per device) replaces the CUDA card query, so
/// the plan can be checked without GPUs; Flash always asks CUDA.
std::vector<int> glm_split_search(const std::string& pack_dir, int64_t max_ctx, const std::vector<int>& devs);

/// The split search prices the NextN draft block (its dense VRAM, its experts' share of the pool, its state, a draft
/// once a token) on the last part: Maya prices it for two parts only although load_pack loads it on the last part of
/// any split; glm-dsa prices it for every n >= 2 (Flash keeps n == 2 so its placements stay as Maya picks them).
bool glm_split_prices_mtp(const Glm5Geometry& g, int n_parts);

/// The CPU-lane pool each part runs its misses on, as the runtime builds them (glm_tier_numa.cu lane_pools, through
/// the same glmtier::lane_pool_mode, without MAYA_GLM_CPU_LANE<dev> or a node-split tier): equal ids are one pool.
/// MAYA_GLM_CPU_SHARED (`shared_env`, nullptr/empty = unset): one shared pool for 3+ parts by default or with 1 (all
/// 0), a pool per part for two by default or with 0 (i); on glm-dsa "socket" or "2" with 2+ NUMA nodes a pinned pool per
/// node, the id being the part's node (its GPU's node from part_node when online, else nodes[i % nodes.size()]).
/// Flash reads every value as Maya does ("2": shared).  nodes: the online NUMA nodes.
std::vector<int> glm_cpu_pool_ids(const Glm5Geometry& g, int n_parts, const char* shared_env,
                                  const std::vector<int>& nodes, const std::vector<int>& part_node);
/// The number of distinct pools in glm_cpu_pool_ids' result.
int glm_cpu_pool_count(const std::vector<int>& pool_ids);
/// The engine's call (src/core/glm_tier_numa.cu: needs CUDA and sysfs): glm_cpu_pool_ids for the split on `devs`
/// with MAYA_GLM_CPU_SHARED, the online NUMA nodes and each GPU's node.
std::vector<int> glm_split_pool_ids(const Glm5Geometry& g, const std::vector<int>& devs);

/// The bytes/s at which one part's misses (its routed experts outside VRAM) are computed by the CPU lane, the price
/// of a miss.  Maya: host_bps / n for every split ("each part's lane, all at once").  glm-dsa without the pipelined
/// decode: the parts take turns on a token, so the n_pools pools (glm_cpu_pool_count) split the host's read speed:
/// one shared pool serves each part's misses at host_bps, a pinned pool per socket at host_bps / sockets; a pool per
/// part (n_pools >= n) or the pipelined decode keep host_bps / n.  host_bps <= 0 (not measured) is 20 GB/s.  At least
/// 1 GB/s.  Flash: Maya's term.
double glm_split_miss_bps(const Glm5Geometry& g, double host_bps, int n_parts, int n_pools, bool pipelined);

// ---------------------------------------------------------------- B6.3: per-part CPU-lane groups

/// The CPU group (index into `group_node`, the NUMA node of each group of this process's CPUs, -1 = no NUMA) each of
/// the parts runs its CPU-lane pool on, given each part's GPU's node `gpu_node` (-1 = unknown).  Maya's rule while
/// there are at least as many groups as parts: each part on its GPU's node when those are distinct and known, else
/// part i on group i.  More parts than groups (3 parts on a 2-socket host): each part on its GPU's node when every GPU's
/// node is known and every group gets a part (the test PC's 2+1 topology: parts 0 and 1 share node 0's cores, part 2 has
/// node 1); else round robin i % G (Maya's rule, which also keeps the 3+0 topology from leaving a socket idle).
/// Empty when there is no group or no part.
std::vector<int> glm_part_groups(const std::vector<int>& group_node, const std::vector<int>& gpu_node, int n_parts,
                                 bool by_node_over = true);

}  // namespace strata::core
