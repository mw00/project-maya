# Project Maya - GLM-5.3-Flash progress log

Model: GLM-5.3-Flash (288 experts x 45 layers, ~330B parameters), Unsloth UD-IQ1_S GGUF (~93 GB), on our own engine.
Every number: greedy, same prompt file, RAM tier pinned (runs are bit-reproducible); "decode" = generated tokens/s,
"prompt" = prompt processing per token.  Short-answer and text-dependent variance is +-20%, so only same-setup numbers
are compared.

Machines
- **Mercury**: 2x Tesla V100 32 GB (PCIe 3), Xeon E5-2690 v4, 30 GB RAM, one NVMe.
- **Uranus**: 1x Tesla V100 32 GB (PCIe 3), i5-12600T, 45 GB RAM (64 GB DDR5 coming), one NVMe.

## Mercury (2 GPUs)

| date | change | decode | prompt |
|---|---|---|---|
| 2026-10-05 | token-by-token engine, experts cached in VRAM + RAM | ~20 tok/s | 31-55 ms/token |
| 2026-10-06 | batched layer-major prompt path, both GPUs pipelined | 26 tok/s | 4.7 ms/token (2.6k tokens) |
| 2026-10-06 | speculative decode with the model's own MTP draft block | 36-40 tok/s | 4.7 ms/token |
| 2026-10-06 | kernel work (attention, experts), tier-table race fixed, faster disk reads | 41.4 tok/s | 4.5 ms/token |
| 2026-10-06 | prompt path: weight dequant 6x faster, background disk reads | 41.4 tok/s | 3.8 ms/token |
| 2026-10-06 | prompt path: light kernels for sparsely routed experts; draft skips non-resident experts | **44.4 tok/s** (200 tok), **47.7 tok/s** (1000 tok) | **3.5 ms/token** (2.6k); 300-token prompt answered in 2.5 s |
| 2026-10-06 | CPU lane (see Uranus): 7 cores per GPU half compute some RAM-tier experts beside the PCIe pulls | 45.9 tok/s (200 tok), **50.5 tok/s** (1000 tok) | 3.5 ms/token |

| 2026-10-07 | Maya-S; prompt path: a deep disk landing ring (64 experts instead of 12 - the reader was the NVMe's bottleneck, ~0.7 of ~2.9 GB/s), the next layer's disk experts read ahead, chunks sized from the card (4.6k tokens on 32 GB) | - | prompts of 2k / 8k / 16k / 30k tokens: 240 / 331 / 345 / - -> **273 / 428 / 487 / 506 tok/s** |
| 2026-10-08 | prompt attention on the tensor cores (FP16 operands, F32 accumulation; the next cells loaded while the current ones compute), the DSA latent cache in FP16 | 28.0 tok/s (5 topics, 300 tokens each) | prompts of 2k / 8k / 16k / 30k tokens: **286 / 469 / 538 / 561 tok/s** |

Live chat server (Mercury, cold start to warm): ~30 -> ~40 tok/s on 60-150 token answers.

One GPU of Mercury (32 GB, 30 GB RAM), the same 2026-10-07 prompt-path change: 2k / 8k-token prompts 120 / 115 ->
134 / **226 tok/s**; with the 2026-10-08 attention work 139 / **243 tok/s**.

## Uranus (1 GPU) - single-GPU work starts 2026-10-06

| date | change | decode | prompt |
|---|---|---|---|
| 2026-10-06 | baseline (RAM tier 36 GB, VRAM pool 24 GB / 3725 experts, 77.7% VRAM hit) | **9.0 tok/s** | 18.4 ms/token |

| 2026-10-06 | prompt-path disk reader keeps 2 experts in flight | 9.0 tok/s | 300-token prompt 25.9 -> 25.0 ms/token (7.5 s) |
| 2026-10-06 | per-user expert usage profile: the warm-up loads your own experts first | 9.0 tok/s | first prompt after a restart ~10% faster (same task) |

| 2026-10-06 | **RAM 45 -> 64 GB** (RAM tier 36 -> 49 GB; disk reads 13.9 -> 2.5 per token) | **16.9 tok/s** | 500-token prompt 18.4 -> 9.9 ms/token |
| 2026-10-06 | **model on a Samsung PM9A1 PCIe 4.0 NVMe** (expert read 3.5 -> 1.2 ms), V100 power limit 150 -> 250 W | **18.0 tok/s** | 300-token prompt 7.5 s -> **4.1 s**; 2.6k-token prompt 5.0 ms/token (13 s) |
| 2026-10-06 | **CPU lane**: when a layer needs experts that sit in RAM, the CPU (6 cores, 0.36 ms an expert) computes the coldest of them while the GPU pulls the rest over PCIe (0.62 ms an expert); the engine measures both at start and picks the split | **21.8 tok/s** (200 tok), 22.5-23.5 on other prompts | unchanged |
| 2026-10-06 | RAM tier 50 -> 55 GB (a one-GPU machine no longer reserves RAM for the draft block it never loads; Uranus keeps 4 GB free instead of 6 - a dedicated server); disk reads 2.0 -> 0.3 a token | **24.4 tok/s** | unchanged |

**Real chat (what the dashboard gives), measured from 2026-10-06 on:** five different topics in a row on one server,
greedy, 400 tokens each, end to end (scratchpad chat_bench.py):

| date | change | real chat (Uranus) |
|---|---|---|
| 2026-10-06 | the above (a server that had been running a while) | 13.2 tok/s; 7-11 disk reads a token |
| 2026-10-06 | duplicate RAM copies freed at every token boundary (a promotion racing a demotion left copies nothing pointed at - 12% of the experts ended up in no tier) | **17.0 tok/s**; 1.3-3.3 disk reads a token |

The greedy numbers above are one text, one topic: a real chat moves between topics, uses a wider spread of experts and
reads more of them from RAM and disk.  Both are reported from here on.

CPU lane quality: perplexity on three texts +1.0%, +1.6%, -0.9% and top-1 accuracy slightly higher on all three -
the same as the GPU-only path within noise (each CPU expert sum is within ~2% of an exact double-precision one; the
GPU's own is ~1.5%).

Uranus before the CPU lane: 55 ms a token = ~22 ms compute, ~28 ms pulling ~50 RAM experts a token over PCIe 3.0, ~5 ms disk.
The power limit barely matters (decode peaks ~140 W; a 2.6k prompt 5.1 -> 5.0 ms/token at 250 W, max 58 C with the
case fan script running).  Next-layer prefetch is still a loss (55 -> 66 ms/token).

Before the upgrades, a single-GPU token (110 ms) went: ~22 ms compute, ~34 ms pulling 61 RAM experts over PCIe, ~48 ms waiting
for 14 disk reads.  A single-GPU prompt is 92% disk: a 300-token prompt reads ~1850 experts (12 GB) at ~2 GB/s.
The RAM upgrade (45 -> 64 GB) should take most of the disk reads away from both.

## Tried and measured, not kept
- Reading predicted disk experts ahead (Mercury and Uranus): exact but slower - only ~45% of disk misses are
  predictable a layer or two ahead, and wrong reads compete for the one NVMe.
- Prefetching the next layer's predicted experts into VRAM: exact, 7% slower on Uranus.
- Computing RAM-tier experts on the CPU instead of copying them: 1.3-1.4 ms an expert vs 0.56 ms over PCIe (both CPUs).
  (Superseded: that was thread start-up cost.  A persistent spinning pool does one in 0.31-0.36 ms on Uranus - the CPU
  lane above.  All of them on the CPU is still slower: nothing gets promoted into VRAM, the hit rate falls to 60%.)
- MTP speculative decoding on one GPU (draft a token, verify two at once): consecutive tokens share only ~1 of 8
  experts per layer and none of their RAM pulls (traced), so a two-token verify costs nearly two tokens of PCIe time -
  no gain on paper at 75-90% acceptance.  On two GPUs the draft pays because it keeps the second GPU busy.
- Twice as many CPU-lane threads (hyperthreads): 21.8 -> 13.3 tok/s - they starve the disk readers.
- Next-layer prefetch again, now that the CPU lane leaves PCIe idle time: still a loss (52.4 -> 65.7 ms/token on a fixed
  text) - the VRAM hit rises 72 -> 77% but the wrong guesses cost more than the right ones save.
- RAM-tier eviction by recency-protected counts, and the prompt path moving its borrowed slots' experts over the
  coldest instead of dropping them: no measurable change on their own (kept: they are the sounder rules).
- Smarter cache eviction: LRU 67%, LFU (ours) 77.5%, decayed-frequency 78%, a small learned model 77%; the theoretical
  optimum (knowing the future) is 87% - the gap needs the future text, not a better history model.
- Keeping a prompt's disk reads in RAM for the answer: more disk reads, not fewer (a long prompt touches every expert).
