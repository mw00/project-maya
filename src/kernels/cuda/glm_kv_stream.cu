// src/kernels/cuda/glm_kv_stream.cu - see include/strata/kernels/glm_kv_stream.hpp.
#include "strata/kernels/glm_kv_stream.hpp"

#include <cuda_runtime.h>

#include <cstdio>

namespace strata::kernels::glmf {
namespace {

constexpr int RT = kKvResolveBlock;

// Block-wide exclusive prefix sum of one int per thread (RT threads); `total` is the sum over the block.
__device__ int block_scan(int v, int* warp_sums, int& total) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int x = v;
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) warp_sums[w] = x;
    __syncthreads();
    if (w == 0) {
        int t = warp_sums[lane];
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, t, o);
            if (lane >= o) t += y;
        }
        warp_sums[lane] = t;
    }
    __syncthreads();
    total = warp_sums[31];
    const int excl = x - v + (w > 0 ? warp_sums[w - 1] : 0);
    __syncthreads();
    return excl;
}

__global__ void __launch_bounds__(RT) resolve_kernel(KvStreamMap m, const int* __restrict__ cells, int n_sel, int page,
                                                     int* __restrict__ slot_cells) {
    __shared__ int s_nmiss, s_lookups, s_cut;
    __shared__ int warp_sums[32];
    const int epoch = m.ctl[0] + 1;
    if (threadIdx.x == 0) { s_nmiss = 0; s_lookups = 0; }
    __syncthreads();
    // 1. hits take this epoch and their reference bit; a missing block is claimed exactly once (-1 -> -2)
    int lookups = 0;
    for (int i = threadIdx.x; i < n_sel; i += RT) {
        const int c = cells[i];
        if (c < 0) continue;
        const int b = c / page;
        if (i > 0 && cells[i - 1] >= 0 && cells[i - 1] / page == b) continue;   // a pool's cells are adjacent: one lookup
        ++lookups;
        const int sl = m.page_table[b];
        if (sl >= 0) {
            m.slot_stamp[sl] = epoch;
            m.slot_ref[sl] = 1;
        } else if (sl == -1 && atomicCAS(&m.page_table[b], -1, -2) == -1) {
            m.miss_block[atomicAdd(&s_nmiss, 1)] = b;
        }
    }
    if (lookups > 0) atomicAdd(&s_lookups, lookups);   // most threads see none: skip the shared atomic
    __syncthreads();
    // 2. one victim per miss: a clock sweep from the hand. A slot this call uses (stamp == epoch) is never taken;
    //    a referenced one loses its bit as the hand passes it and is taken on the next pass.
    const int need = s_nmiss, n = (int) m.n_slots;
    int hand = m.ctl[1], got = 0;
    for (int scanned = 0; got < need && scanned < 3 * n; scanned += RT) {
        const int j = (int) (((long long) hand + threadIdx.x) % n);
        const bool mine = m.slot_stamp[j] == epoch;
        const bool cand = !mine && (m.slot_block[j] < 0 || m.slot_ref[j] == 0);
        int total = 0;
        const int rank = block_scan(cand ? 1 : 0, warp_sums, total);
        const int want = need - got;
        if (threadIdx.x == 0) s_cut = RT;
        __syncthreads();
        if (cand && rank == want - 1) s_cut = threadIdx.x + 1;   // the hand stops just past the last slot taken
        __syncthreads();
        const int cut = s_cut;
        if (cand && rank < want) {
            m.miss_slot[got + rank] = j;
            m.slot_stamp[j] = epoch;   // taken: a sweep that wraps around must not take it twice
        } else if (threadIdx.x < cut && !mine) {
            m.slot_ref[j] = 0;
        }
        got += total < want ? total : want;
        hand = (int) (((long long) hand + cut) % n);
        __syncthreads();
    }
    // 3. re-point the table; the copy kernel fills the slots
    const int placed = got < need ? got : need;
    for (int k = threadIdx.x; k < need; k += RT) {
        const int b = m.miss_block[k];
        if (k >= placed) { m.page_table[b] = -1; continue; }   // overflow: never happens with a legal n_slots
        const int sl = m.miss_slot[k];
        const int old = m.slot_block[sl];
        if (old >= 0) m.page_table[old] = -1;
        m.slot_block[sl] = b;
        m.slot_stamp[sl] = epoch;
        m.slot_ref[sl] = 1;
        m.page_table[b] = sl;
    }
    __syncthreads();
    // 4. the selection in slot rows (a block an overflow left out reads as padding)
    for (int i = threadIdx.x; i < n_sel; i += RT) {
        const int c = cells[i];
        const int sl = c < 0 ? -1 : m.page_table[c / page];
        slot_cells[i] = sl < 0 ? -1 : sl * page + c % page;
    }
    if (threadIdx.x == 0) {
        m.ctl[0] = epoch;
        m.ctl[1] = hand;
        m.ctl[2] = placed;
        if (placed < need) m.ctl[3] = 1;
        unsigned long long* c = reinterpret_cast<unsigned long long*>(m.ctl + 4);
        c[0] += (unsigned long long) placed;
        c[1] += (unsigned long long) s_lookups;
        c[2] += 1ull;
    }
}

// One block per missed block (grid-stride): its rows from the host copy into its slot, 16 B per thread.
__global__ void copy_kernel(KvStreamMap m, const uint8_t* __restrict__ host, uint8_t* __restrict__ slots, int len) {
    const int need = m.ctl[2];
    for (int k = blockIdx.x; k < need; k += gridDim.x) {
        const long long b = m.miss_block[k], sl = m.miss_slot[k];
        const uint4* src = reinterpret_cast<const uint4*>(host + b * len);
        uint4* dst = reinterpret_cast<uint4*>(slots + sl * len);
        for (int i = threadIdx.x; i < len / 16; i += blockDim.x) dst[i] = src[i];
    }
}

}  // namespace

void glm_kv_resolve(const KvStreamMap& m, const uint8_t* host, uint8_t* slots, int row_bytes, int page,
                    const int* cells, int n_sel, int* slot_cells, cudaStream_t s) {
    if (n_sel <= 0) return;
    resolve_kernel<<<1, RT, 0, s>>>(m, cells, n_sel, page, slot_cells);
    copy_kernel<<<96, 128, 0, s>>>(m, host, slots, page * row_bytes);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) std::fprintf(stderr, "glm kv stream: resolve: %s\n", cudaGetErrorString(e));
}

}  // namespace strata::kernels::glmf
