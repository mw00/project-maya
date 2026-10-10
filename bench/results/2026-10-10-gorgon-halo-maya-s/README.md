# Maya-S on Gorgon Halo (Ryzen AI Max+ PRO 495 / Radeon 8065S) - 2026-10-10

Machine: Ryzen AI Max+ PRO 495, Radeon 8065S (gfx1151, 40 CUs), 192 GB LPDDR5X-8533 with a 160 GB GPU carve-out
(HIP sees 172 GB, Windows 32 GB), Windows 11, HIP SDK 7.2, stock clocks and power, Balanced power plan. Maya-S
(Maya-S-v2-IQ2_XXS), 32K context, the setup's Windows-APU config. Every expert of Maya-S fits in the carve-out.

Every number below was taken with nothing else running on the PC (no compiles, one engine at a time): on an APU the
CPU and GPU share one power budget and one memory bus, and a build running beside a benchmark cost up to ~25%.

## What changed

This branch is `mtp-rework` (the MTP decode: drafts in a chain, one batched verify, any GPU count) plus:

| change | effect on this PC |
|---|---|
| **a complete expert pool** where the budget holds every expert, its spares and the prompt path's tail (`STRATA_GLM_COMPLETE_POOL`, on): Windows sized the APU's carve-out like a discrete card - 285 of 288 experts a layer warm, 0.1-0.7 SSD reads a token in decode, ~880 experts moved out and ~1900 staged from the SSD by every prompt | vram hit 100%, no SSD reads; MTP sustained 22.65 -> 23.23 tok/s |
| while every expert is in VRAM, no per-route `moe_wait` / `moe_fetch` / `moe_cpu_wait` launches (`STRATA_GLM_RESIDENT_SKIP`, on) | 126 fewer launches a token |
| MTP drafts: a **draft-vocabulary head** (the full head's own rows for ids < 98304 plus digits and control tokens: 64% of its 0.52 GB; `STRATA_GLM_MTP_DRAFT_VOCAB`), **chained drafts** (token_embd on the device, one read-back a round; `STRATA_GLM_MTP_CHAIN`), a start-and-climb length search (`STRATA_GLM_MTP_START`) and per-position acceptance over the recent text (`STRATA_GLM_MTP_ACC_WINDOW`, 64) | first answer no longer slow while the length is searched; 1536-token essay 21.34 -> 21.96 |
| **RDNA3 dense GEMV** (`mv_rdna_kernel`: Q6_K and BF16 jobs in one launch, two super-blocks in flight, Q6_K's -32 as a second sudot4; `STRATA_GLM_MV_RDNA`, on for gfx11) - bit for bit the old kernels' outputs (`glm_mv_bench --parity-only`) | per layer: kda_proj 384 -> 366 us, router+shexp 90 -> 78, dsa_out 258 -> 244, dense_down 197 -> 182 |

## Decode

Sustained: two 4096-token answers (an essay, a program) and 2048 tokens after a 16K-token document prompt, greedy,
total tokens / total time. Short: five 256-token answers (the `--bench` style).

| build / setting | sustained tok/s | essay / code / after 16K | short answers |
|---|---:|---|---:|
| main v1.0.30 | 17.92 | 18.06 / 18.01 / 17.46 | 18.93 |
| this branch, drafting off (`STRATA_GLM_MTP_FIXED=0`) | 18.71 | 18.81 / 18.81 / 18.33 | 19.79 |
| `mtp-rework` alone, adaptive | 22.91 | 21.32 / 24.33 / 23.67 | 26.84 |
| **this branch, adaptive (default)** | **23.81** | 22.22 / 25.67 / 23.79 | **28.55** |
| this branch, adaptive up to 7 drafts (`STRATA_GLM_MTP_DRAFT=7`) | 23.93 | 22.36 / 25.82 / 23.81 | 28.90 |
| this branch, 1 draft a round | 23.70 | 22.79 / 24.85 / 23.39 | 26.75 |
| this branch, 2 drafts a round | 23.67 | 21.57 / 26.12 / 23.84 | 29.03 |
| this branch, 3 drafts a round | 22.08 | 19.02 / 25.54 / 23.26 | 28.75 |
| this branch, 4 drafts a round | 19.64 | 16.25 / 23.65 / 21.30 | 26.65 |
| this branch, 5 drafts a round | 16.90 | 13.68 / 20.82 / 18.66 | 23.63 |
| this branch, 7 drafts a round | 12.81 | 10.18 / 16.14 / 14.30 | 18.58 |

- **+33% sustained, +51% on short answers** against v1.0.30. Long fixed lengths lose on prose (the drafts beyond
  the second are rarely kept there); the adaptive length is the best or within noise of it everywhere.
- Sampled decoding (temperature 1.0, top_p 0.95, the dashboard's default), 3x2048 + 1024 after an 8K prompt:
  v1.0.30 17.51, `mtp-rework` adaptive 21.43 tok/s (+22%).
- **Exact:** with drafting on and off (`STRATA_GLM_MTP_FIXED=0`, the same build and memory layout) the 10,240 tokens of
  the sustained runs are identical; the five short answers' md5s are identical to v1.0.30's. Long prompts differ from
  v1.0.30 after a few dozen tokens as `mtp-rework` already does: the prompt path's arithmetic depends on where the
  experts are (docs/GLM-MTP.md); both outputs read alike.

## Prompt (prefill)

2048 / 8192 / 16384 tokens: v1.0.30 282 / 339 / 341, this branch 290 / 331 / 335 tok/s - unchanged within noise.

Environment settings measured and not better here: `STRATA_GLM_PREFILL_SUB=2048` (+2-3%, kept out pending a quiet
re-measure), `STRATA_GLM_PREFILL_ATTN=wmma2` (-10%), `STRATA_GLM_PREFILL_WINDOW=0` (-13%), `ROCBLAS_USE_HIPBLASLT=1`
(even), `STRATA_GLM_HC2=1` (-8% decode), `STRATA_GLM_MLA_ATTN1=1` (-3% decode), `STRATA_GLM_CPU_LANE=0` (even).

## Where a token's time goes (v1.0.30, single token, ~58 ms of GPU stream time)

KDA q/k/v projections 13.1 ms, routed gate/up 9.4, router + shared gate/up 6.3, KDA out 5.0, routed down 4.6, MLA
3.3, DSA out 3.0, mHC 2.2, DSA q 2.0 - about 10 GB read a token (Q6_K matvecs 6.4 GB, routed experts 2.4 GB) against
the 273 GB/s of the LPDDR5X. An MTP verify of 3 rows: 33 ms of its ~73 are the rows' routed experts (each row reads its
own 8) - reading each expert once for all the rows that route to it is the next step.
