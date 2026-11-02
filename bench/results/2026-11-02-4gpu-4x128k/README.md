# GLM-5.3-Flash Maya-S v2 on 3x V100 + RTX 3060: four conversations at 128K, 20+ tok/s each

2026-11-01 / 02. Host: 2x Xeon E5-2690 v4 (28 cores / 56 threads, two NUMA nodes), RTX 3060 12 GB + 3x Tesla
V100-PCIE-16GB (PCIe 3.0 x16, two GPUs a socket), 80 GB DDR4-2400 in 5 DIMMs (socket 0: 2 channels, socket 1: 3),
one NVMe. Maya-S-v2-IQ2_XXS. Engine: v1.1.49 plus this branch's changes (the code measured is the branch's).

## Four conversations (`--parallel 4`, 128K each): 20.1-21.5 tok/s a conversation, 80-84 in all

Client: four concurrent streamed `/v1/chat/completions` requests (short thinking-mode prompts - CPU caches,
a Python interval merge, a Kyoto trip, the primes below 100 - 512 tokens, temperature 1.0, top_p 0.95), warm-up
rounds of the same prompts first, then 2 measured rounds; tok/s a conversation from its first token to its last.

| config (each row adds to the one above) | tok/s a conversation | in all |
| --- | ---: | ---: |
| the single-conversation config + `--parallel 4`, fresh server | 14.1-17.1 | 52-67 |
| `MAYA_GLM_CPU_LANE=48` (was 22) | 18.4-19.6 | 72-77 |
| `MAYA_GLM_SPLIT=10,21,33` (was 8,22,35) | 19.5-20.5 | 77-80 |
| **`MAYA_GLM_CPU_PLAN=012334567` (was 012234556), two server starts** | **20.1-21.5** | **80-84** |

The final config's settings (with `MAYA_GLM_ROW_HALVES=1`, `MAYA_GLM_PROMOTE_MIN=2`, `MAYA_GLM_RAM_RESIDENT=1`,
`MAYA_GLM_VRAM_EVICT=least`, `MAYA_GLM_KDA_GRAPH=1`, `MAYA_GLM_SEQ_CTX=131072`, `MAYA_GLM_CKPTS=0`):

    "args": ["--max-context", "131072", "--parallel", "4", "--kv-streaming", "on", "--kv-resident", "12288",
             "--expert-cache", "auto", "--prefill", "auto", ...],
    "env":  {"MAYA_GLM_SPLIT": "10,21,33", "MAYA_GLM_CPU_LANE": "48", "MAYA_GLM_CPU_PLAN": "012334567",
             "MAYA_GLM_ROW_HALVES": "1", ...}

Why these three: with several conversations the split's GPUs are a pipeline - each works on a different
conversation's token - so the slowest GPU sets the pace. `MAYA_GLM_PROF` on the old split (8,22,35), share of each
GPU's stream time:

| GPU (layers) | waiting for its input | PCIe expert fetches | CPU-lane wait |
| --- | ---: | ---: | ---: |
| CUDA0 RTX 3060 (0-7) | 49% | 1% | 4% |
| CUDA1 V100 (8-21) | 2% | 38% | 19% |
| CUDA2 V100 (22-34) | 21% | 20% | 20% |
| CUDA3 V100 (35-44) | 41% | 10% | 18% |

CUDA1 never waited; the 3060 and the last V100 idled. Moving layers to them (10,21,33), letting the one shared
CPU-lane pool use 48 of the 56 threads (the four service threads and the server keep the rest), and taking one more
of a layer's RAM-tier experts on the CPU instead of over the four GPUs' busy PCIe links did it.

Tried and slower (same client, `tok/s a conversation`):

| change | tok/s |
| --- | ---: |
| `MAYA_GLM_SPLIT_ROWS=2` (+ `EVEN_GROUPS`) | 14.0-14.3 |
| `SPLIT_ROWS=2` + `GROUP_EXPERTS`, `ROWS_BATCH`, `SKIP_EMPTY`, `ROWS_SMALL` | 13.5-14.4 |
| ... + `MAYA_GLM_MULTI_MTP=1` / `2` / `3` | 12.3-14.4 / 10.3-12.5 / 8.4-11.4 |
| `SPLIT_ROWS=4` (/ + grouped + `MULTI_MTP=2`) | 8.3-8.6 (/ 5.7-6.8) |
| a CPU-lane pool a GPU, 4 x 12 / 4 x 13 (each pinned to one socket) | 11.2-12.3 / 12.8-13.1 |
| 4 x 12 on both sockets (`MAYA_GLM_LANE_CPUS<n>`) | 12.7-13.3 |
| split 11,21,33 / 11,22,34 (too many layers for the 3060's VRAM) | 17.3-17.6 / 17.7-18.1 |
| `MAYA_GLM_CPU_PLAN=012345677` / `012345678` | 14.6-14.9 / 17.1-19.1 |
| `MAYA_IQ2XXS_1T=1` (split 10,21,33) | 19.5-19.8 vs 20.2-20.8 without |

Batching the conversations a GPU holds left the other GPUs waiting, and the drafts' extra rows add routed experts
to a CPU lane and PCIe links that four conversations already keep busy. A fresh server varies ~5-10% with one
warm-up round; the decisions above are from three warm-up rounds and the final config from two separate starts.

## One conversation: ~46 tok/s (42.5 before)

Short thinking-mode prompts, 512 tokens, sampled, 3 runs each, through the engine's `--serve` protocol: the
fast512k config 42.3-43.3 -> `MAYA_GLM_ROW_HALVES=1`, `MAYA_GLM_PROMOTE_MIN=2`, one-token drafting (no
`MAYA_GLM_SPEC=3`) 45.6-46.8; easy prompts 51-54. A token's ~45 RAM-tier experts on the CPU lane (85-89% VRAM
hits) are most of it: ~330 us an expert with 22 threads, compute-bound (~2 weights a cycle a core with ggml's
IQ2_XXS / IQ3_XXS dots; a pure read of the same rows takes ~130 us). Next-layer prefetch (`PREFETCH_N`) raised
the VRAM hits to 92% but its PCIe wait cost more: 33-38 tok/s with the copy kernel, 42-44 with `MAYA_GLM_PF_CE=1`.

## Limits on this host
- Socket 0 has 2 of its 4 memory channels filled: ~10% of its row halves spill to socket 1, and the busiest GPUs
  fetch from it. A third DIMM there is the cheapest next step.
- The cores already run ~2.9 GHz under the lane's AVX2 load (the `performance` governor adds 1-3% at most).
