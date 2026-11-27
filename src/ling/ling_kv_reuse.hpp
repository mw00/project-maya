// src/ling/ling_kv_reuse.hpp - keeping a long prompt's expanded MLA keys / values between its chunks.
//
// The prompt's MLA attention expands the latent cache into per-head keys (K_nope = W_kb lat) and values (V = W_vb lat)
// one tile of kTile positions at a time, and did so again for every chunk of the prompt: a 200K prompt in 2048-token
// chunks expanded its first tiles about a hundred times.  A full tile's expansion depends only on the latent rows of its
// positions (written once, never changed while the prompt is read) and the layer's weights, and is the same GEMM (same
// shape, same aligned output) whether it is written to the chunk's scratch or to this store, so a tile kept here and
// read by later chunks is the same bits as one expanded again.
//
// The store lives only while one prompt is read (allocated at its start, freed at its end), in VRAM that the start plan
// left free: free VRAM less the plan's reserve, the prompt's own transient allocations and a share of the card kept
// for the driver and the BLAS library.  Where that is too small for one tile a layer, nothing is kept and every chunk
// expands as before.  The earliest tiles are kept first: they are read by the most chunks.
//
// Host code only, no CUDA: tests/ling_unit_test.cpp runs it.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ling::kv_reuse {

constexpr int kTile = 1024;   // the prompt attention's key tile (BK) - only full tiles are kept

/// Bytes of one kept tile: K_nope [H][kTile][qk_nope] + V [H][kTile][v_head], FP16.
inline size_t tile_bytes(int n_head, int qk_nope, int v_head) {
    return (size_t) n_head * (size_t) kTile * ((size_t) qk_nope + (size_t) v_head) * 2u;
}

struct Plan {
    int tiles_per_layer = 0;   // tiles kept for each MLA layer of this GPU (0 = nothing kept)
    size_t bytes = 0;          // the store's size
};

/// How many tiles to keep a layer on one GPU for a prompt of n tokens from pos0, read in chunks of `chunk`.
/// free_b: the GPU's free VRAM now; keep_b: VRAM that must stay free (plan reserve + the prompt's own allocations);
/// total_b: the card's memory (1/32 of it is left for the driver / BLAS workspaces).
inline Plan plan(int n_mla_layers, int n_head, int qk_nope, int v_head, int pos0, int n, int chunk, size_t free_b,
                 size_t keep_b, size_t total_b) {
    Plan p;
    if (n_mla_layers <= 0 || n <= 0 || chunk <= 0) return p;
    if (n <= chunk) return p;   // one chunk: no tile is read twice
    // A kept tile is read again by a later chunk only if it was full before the last chunk started
    const int64_t last_chunk_start = (int64_t) pos0 + (int64_t) ((n - 1) / chunk) * (int64_t) chunk;
    const int64_t useful = last_chunk_start / kTile;
    if (useful <= 0) return p;
    const size_t headroom = total_b / 32u;
    const size_t need_keep = keep_b + headroom;
    if (free_b <= need_keep) return p;
    const size_t tb = tile_bytes(n_head, qk_nope, v_head);
    const size_t per_layer = (free_b - need_keep) / (size_t) n_mla_layers;
    const int64_t fit = (int64_t) (per_layer / tb);
    p.tiles_per_layer = (int) std::min<int64_t>(useful, fit);
    if (p.tiles_per_layer <= 0) {
        p.tiles_per_layer = 0;
        return p;
    }
    p.bytes = (size_t) p.tiles_per_layer * tb * (size_t) n_mla_layers;
    return p;
}

}  // namespace ling::kv_reuse
