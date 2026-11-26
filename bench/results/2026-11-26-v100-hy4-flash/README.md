# Hy4-Flash Maya-S on one Tesla V100 32 GB: prompts 397 -> ~930 tok/s at 22K, decode 33 -> ~50 tok/s, step by step

2026-11-26. Host: Ryzen 9 7950X (16 cores / 32 threads, AVX-512), 62 GB DDR5 (dual channel), one Tesla
V100-PCIE-32GB (PCIe 3.0 x16, 250 W, ~80 C under load: the SM clock 1380 -> 1200-1330 MHz), the model and its
n-gram table on one NVMe; Ubuntu 24.04, driver 535.288, CUDA 12.0, sm_70. Hy4-Flash Maya-S (`hy4-flash-maya-s`,
experts IQ2_XXS on layers 1-23 and IQ3_XXS on 24-41), set up by `./maya.sh --setup --model hy4-flash-maya-s`
on main 53885cef (v1.1.81), then the changes below one by one. `--max-context 131072`, drafting auto (DFly2 chosen
by the start's own trial: no drafting 32.5, the MTP block 37.6, DFly2 42.3 tok/s). ~40% of the routed experts
fit in VRAM; the rest run on the CPU lane, or are streamed to the GPU for a prompt chunk.

The client is [hy4bench.py](hy4bench.py): over the OpenAI API, temperature 0 - decode (TG) on a short code
prompt (43 tokens, 512 out; greedy, so the same tokens every run of a build), prompt reading (PP) on a ~22K-token
prompt behind a nonce (nothing reused), decode after that prompt (256 out). The numbers are the engine's own
`maya hy4 serve: prompt ...` lines, the third round of three (HY4_HOT settles in the first; the first prompt after
a start also pays the n-gram table's cold pages: ~-20%). Expect +-5% between runs from the card's heat alone.

## Served, step by step (22K prompt = 21,940 tokens)

| step | PP 22K | TG short | TG at 22K |
| --- | ---: | ---: | ---: |
| the setup's config (F16 cache, prompt chunk 4096) | 397 | 34.8 | 33.4 |
| the n-gram table's mapping `MADV_RANDOM` (#1291) | 486 | 46.2 | 38.9 |
| `--kv q8` (+533 experts in VRAM) | 497 | 48.6 | 39.5 |
| `DS4_IQ_TC_RM=2` on Volta (#1294) | 508 | | |
| GEMVs of 2+ columns with 2 rows a warp (#1292) | 510 | 49.2 | 45.9 |
| Q6_K's `ql` / `qh` as aligned words (#1293); token-window attention on the tensor cores | 511 | 55.3 | 50.4 |
| the latent cache's unused room lent to experts (+210-390 in VRAM) | 520 | 54-57 | 50.8 |
| the prompt's streamed experts double-buffered | 641 | 50-54 | 48.7-54.0 |
| attention in 1024-row sub-chunks, the FFN over 8192-row chunks; busy experts as FP16 GEMMs | 772-800 | 50-51 | 47-51 |
| F16 weights once a layer; a warp a row softmax; keys padded to 8; tile k-steps double-buffered (#1296) | 837-859 | 50-55 | 47-51 |
| the prompt's attention in one pass on the tensor cores; #1251 / #1252 / #1175 ported | 915-948 | 51-54 | 47 |

#1291-#1294 and #1296 are this host's upstream PRs; the rest is on merbanan/project-maya, branch `hy4-v100`.
Not faster here: `HY4_Q4X` (the 4-bit lane copies: a call of 16 experts 0.49 ms from the file, 1.02 ms as copies -
this CPU's lane is bound by its memory), more lane threads, the `HY4_*_FORK` second streams on one card,
`HY4_PSTREAM` (#1222: 26.5 GB of copies pinned in ~60 s, a 22K prompt 955 -> 753 tok/s with every layer's
experts outside VRAM streamed a chunk), `HY4_LAT16` (#1175: within noise).

## The engine alone (the same build, the service stopped)

| prompt | before (main) | after |
| --- | ---: | ---: |
| 16K tokens, PP | ~500 | 915-951 |
| 22K tokens, PP | ~450 | 904-973 |
| 4096 + 512 tokens out, TG (DFly2) | 40.8 | 58.4 |

## Kernels alone

`tests/iq_mv_bench.cu` (the GEMVs on Hy4's dense shapes) and `tests/iq_set_bench.cu` (the prompt's expert sets on
its 1536 x 1536 experts, 32 experts a set, against FP16 GEMMs of the same work) are in this PR; their headers give
the nvcc lines. On this V100:

| GEMV | 1 column | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: |
| Q6_K 3072 -> 8192, main | 48.7 us | 67.5 | 99.4 | |
| ... with #1292 + #1293 | 36.1 | 36.8 | 50.1 | 71.5 |
| Q8_0 8192 -> 3072, main | 40.0 | 48.5 | 78.5 | 141.4 |
| ... with #1292 | 38.5 | 39.8 | 43.2 | 71.3 |

| expert sets, IQ2_XXS | 64 rows | 128 | 256 | 512 |
| --- | ---: | ---: | ---: | ---: |
| the tiles (`DS4_IQ_TC_RM=1`) | 19.4 TFLOP/s | 21.2 | 21.8 | 22.8 |
| the tiles, RM 2 (#1294) | 20.4 | 26.2 | 27.2 | 27.7 |
| ... double-buffered (#1296) | | 26.5 | 27.6 | 28.2 |
| decode to FP16 + cuBLAS | (3.06 ms) | (4.02 ms) | 3.96 ms | 5.75 ms (vs 8.37) |

cuBLAS FP16 on Hy4's attention projections reaches 47-98 TFLOP/s here (`q_b`, `attn_gate`, `attn_output` at
1024-8192 rows); the expert tiles' ~27 TFLOP/s is their own ceiling (the weights' decode and the fragments' loads),
so the busiest experts of a big chunk go to cuBLAS on the branch.

## Where the time goes now (a 16K prompt, nvprof)

Expert sets ~5.3 s, cuBLAS (projections, the busy experts) ~9 s, the one-pass attention ~5 s, the hyper-connection
kernels ~1.2 s (memory bound: a chunk's 4 streams), F16 conversions ~1 s. Decode is GPU-bound: ~1760 kernel
launches a decode round, the GPU busy ~84% of it.
