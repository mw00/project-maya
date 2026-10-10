// include/strata/kernels/glm_kv_stream.hpp - KV streaming for the GLM DSA layers' latent cache (kv_stream.hpp's
// residency map and CLOCK resolve, for the absorbed-MLA latents).
//
// A streamed DSA layer keeps its AUTHORITATIVE latent cache - one row a position (an FP16 row of kv_lora, or an INT8
// record of lat8_rec_bytes) - in pinned, device-mapped host memory, in position order, as the fully resident cache
// would hold it.  VRAM holds a pool of `n_slots` pages of `page` positions (page = idx_kpool = one indexer pool, 4
// positions: 4 KB in FP16), and a RESIDENCY MAP: `page_table[block]` is the slot holding positions
// [block * page, block * page + page), or -1.
//
// Every attention kernel is unchanged.  After the selection and before the attention, `glm_kv_resolve` makes every
// block the selection names resident - on device, no host round trip:
//   1. one block of 1,024 threads walks the selected cells: a resident block gets this call's epoch and its clock
//      reference bit, a missing block is claimed once (page_table -1 -> -2) into a miss list;
//   2. the same block picks one victim slot per miss with a CLOCK sweep (second chance) that never takes a slot this
//      call uses, re-points the page table (old block -> -1, new block -> slot), and writes the selection again in
//      SLOT ROWS (cell -> slot * page + cell % page), which the attention reads instead of the cells;
//   3. a copy kernel reads the missed blocks from the host copy (zero-copy, over PCIe) into their slots.
// Writers (both dsa_prep kernels) write the host copy always, and the slot only if the block is resident, so the slots
// never go stale and nothing needs invalidating - not after a rejected draft, not on a snapshot restore.  The prompt
// path does not resolve: it stages a layer's whole prefix from the host copy into an identity-layout buffer and reads
// that (glm_prefill.cu).
#pragma once

#include "strata/kernels/kv_stream.hpp"   // KvStreamMap, kv_stream_map_bytes, kv_stream_reset, kv_stream_counters

#include <cuda_runtime.h>

#include <cstdint>

namespace strata::kernels::glmf {

/// The fewest positions a streamed layer keeps in VRAM: Strata's qsa_kv_resident_min, ten of the indexer's 2,048-cell
/// selections (and at least kKvResolveBlock slots: one clock sweep step must not see a slot twice).
inline constexpr int64_t kKvResidentMin = 20480;
inline constexpr int kKvResolveBlock = 1024;

/// Make every block named by `cells` (n_sel of them, -1 = padding) resident, and write the selection in slot rows to
/// `slot_cells` (-1 stays -1).  `host` and `slots` hold `row_bytes` a position; a page of them must be a multiple of
/// 16 bytes, and the map must have at least kKvResolveBlock slots (the caller checks both once, at load).
void glm_kv_resolve(const KvStreamMap& m, const uint8_t* host, uint8_t* slots, int row_bytes, int page,
                    const int* cells, int n_sel, int* slot_cells, cudaStream_t s);

}  // namespace strata::kernels::glmf
