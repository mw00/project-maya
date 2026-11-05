# GLM-5.3-Flash (UD-IQ1_S pack) on Project Maya - 2026-10-06

2x V100 PC: 2x V100-32GB PCIe3, Xeon E5-2690 v4, 30 GB RAM (the RAM tier is ~22 GB; ~1-10% of experts stay disk-only).
1x V100 PC: 1x V100-32GB, i5-12600T, 45 GB RAM.  All decode numbers greedy, after a prompt, several hundred tokens.

## What changed (branch glm5-flash, 161aecf .. this commit)

| change | effect |
|---|---|
| batched layer-major prompt path (`src/core/glm_prefill.cu`): cuBLAS FP16 projections, llama.cpp MMQ for the experts **in place on the VRAM pool**, RAM/disk experts staged through a ring, two GPUs pipelined | prompt 31-55 -> **4.7 ms/token** (2.6k tokens: 91 s -> 12 s) |
| prompt buffers **borrowed from the expert pool's tail** while a prompt runs (only what its chunk needs) | decode keeps every VRAM slot |
| **FIX**: the split-K MLA kernel asked for a 96 KB shared-memory opt-in, which fails on V100 - every launch errored, DSA attention was silently zero since e7570df | correct attention; kernel launch errors are now sticky (the forward errors out) |
| VRAM slots per layer by routing share (`expert_counts.txt`) | 35.5 -> 32.9 ms/token |
| hc3 (mHC read half: weights first, warp-parallel partial sums) | 0.60 -> 0.42 ms per GPU per call kind |
| NextN (MTP) draft block (blk.45) loaded from the shards on the tail half; its caches filled by the prompt path | draft == next greedy token 64-68% on teacher-forced doc/code text, 83-99% in free generation |
| **pipelined speculative decode**: the head half runs the draft for the token after next while the tail finishes the current one; recurrent states saved/restored on a miss | **26 -> 36-40 tok/s**, token-identical to the plain loop |
| auto split +2 layers for the head when the draft runs | split 22/23/24/25/26 -> 36.3/36.7/40.0/37.3/39.4 tok/s |

### Later the same day (06d07ef, 1afdec9)

| change | effect (the 2x V100 PC, 500-token prompt, 200 tokens, RAM tier pinned at 20 GB) |
|---|---|
| **FIX**: one between-token table batch could edit a key twice (made live, then freed); the parallel update kernel let either value win, so the device could later fetch another expert's bytes | exact; found by an A/B that should have been token-identical (decode is now verified identical across tier states) |
| MLA split kernel grid (chunks x head groups) - 19 blocks left most SMs idle | 208 -> ~50 us a DSA layer |
| expert kernels: 2+2 rows a warp (occupancy), the shared expert's down as extra blocks of the gate/up launch | 167 -> 138 us a MoE layer, bit-identical |
| argmax float4 loads; disk reads in 8 pieces across the workers | disk wait 3.6 -> 3.1 ms |
| decode total | **26.4 -> 24.2 ms/token (37.8 -> 41.4 tok/s)**; ~45.7 tok/s averaged over 1000 tokens (the tiers adapt) |
| prompt path: disk experts read by a background thread from the plan on | 2.6k-token prompt 5.3 -> 4.5 ms/token |
| prompt path: coalesced Q5_K/Q6_K/Q8_0 -> f16 dequant (the projections' weights, every 256-token sub-batch) | **4.5 -> 3.8 ms/token** (~265 tok/s), bit-identical |
| live server (multi-turn, 60-150 token answers) | 30 -> 39 -> 42 tok/s as the tiers warm (was 31-34) |

Tried and measured NOT faster on the 2x V100 PC (kept, off by default): reading predicted disk-only experts ahead in decode
(STRATA_GLM_AHEAD - only ~40% of disk misses are predictable within 4 layers, the wasted reads share the one NVMe),
and in the prompt path reading the next layer's whole disk-only set ahead (STRATA_GLM_PREFILL_PRED_T - a 2k chunk
leaves ~40% of them unrouted).  A CPU expert path is slower than the PCIe pull on this Xeon (1.4 vs 0.56 ms an expert).
Cache policy: replaying 1000 tokens of routes, LRU = LFU (97.7 / 96.4% per GPU) and even Belady's optimum only
reaches 98.6 / 98.0% - the misses are capacity (30 GB RAM), not policy.

Where a token's time goes now (per half, median): ~11 ms of compute, plus ~8 ms of misses (RAM->VRAM pulls over
PCIe ~0.56 ms an expert, disk reads ~3.1 ms) - the two halves wait on each other's slow tokens.

## The 2x V100 PC: decode after a 500-token prompt, 300 tokens

| mode | ms/token | tok/s | drafts accepted |
|---|---|---|---|
| plain (STRATA_GLM_NO_SPEC=1) | 38.1 | 26.2 | - |
| speculative, split 22 | 27.6 | 36.3 | 99% |
| speculative, split 24 (default) | 25.0 | 40.0 | 97% |

Live server (first three API requests after a restart, 120-token answers): 28.3 -> 31.0 -> 35.8 tok/s as the tiers
warm; 83-86% of drafts accepted; disk waits 10-15 ms/token (the 30 GB host is the limit).

Prompt path, 2.6k-token prompt: 12.2 s (4.7 ms/token, 213 tok/s) vs ~91 s token by token.

## The 1x V100 PC: live server

Decode 6.2-7.2 tok/s (VRAM hit ~70%, ~75 RAM fetches and ~20 disk reads per token: disk-bound - the single-GPU work
is Part 2).  TTFT 13 s for ~880 prompt tokens, 33 s for ~3.5k.

## Reproduce (the 2x V100 PC)

    STRATA_GLM_RAM_GB=20 STRATA_GLM_SPLIT=24 build/strata --glm-pack <pack> --max-context 8192 --greedy --max-new 300 --tokens <ids>
    (STRATA_GLM_NO_SPEC=1 for the plain loop; STRATA_GLM_SPEC_PROF=1 prints the per-step host phases;
     STRATA_GLM_PREFILL_VERBOSE=1 / STRATA_GLM_PREFILL_PROF=1 for the prompt path)

Pinning `STRATA_GLM_RAM_GB` makes runs bit-reproducible: the RAM tier's size otherwise follows MemAvailable, which moves
experts between staging groups and changes MMQ's stream-k summation order in the prompt path.
