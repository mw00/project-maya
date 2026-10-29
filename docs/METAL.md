# Maya on Apple Silicon (Metal)

Back to the [README](../README.md).

Experimental: GLM-5.3-Flash on a Mac with Apple Silicon, on its GPU through Metal, with the same dashboard and the same
OpenAI- and Anthropic-compatible API as on Linux and Windows (streamed answers, tool calls, thinking levels). It reads
pictures (`maya-vision`, the image encoder, runs on the GPU too), and it drafts
with the model's MTP block or the DFlash2 drafter (`--draft dflash2`) when that is faster on the Mac. One conversation at a time;
DeepSeek-V4.1-Flash, MiniMax-M3.1-Flash and Qwen-3.8-Flash-Next are NVIDIA / AMD only for now.

**Status: experimental.** Mac support is a community contribution. Every change is built and unit-tested on GitHub's
Apple Silicon runners (the `macos` workflow: the engine, its sampler and the small synthetic-model tests), but the
Maya team has no Mac and has not run it with the real model. If you try it on a Mac, please tell us how it went in a
[GitHub issue](https://github.com/mw00/project-maya/issues): the chip and its unified memory, the model, the speed
`./maya.sh --bench` prints, and anything that went wrong (`./maya.sh --report` collects the details).

```sh
./maya.sh --backend metal --check            # this Mac: chip, unified memory, Xcode's compiler, CMake
./maya.sh --backend metal --setup            # build maya-metal, download the model, write the config
./maya.sh                                    # start it: dashboard at http://127.0.0.1:8080
```

You need macOS on Apple Silicon (M1 or newer), the Xcode Command Line Tools (`xcode-select --install`), CMake 3.24+
(`brew install cmake`) and Python 3.10+. On a Mac `./maya.sh` picks the Metal backend by itself when nothing is set
up yet.

## How it runs

`maya-metal` (src/metal) is its own engine. It runs the whole forward pass as ggml graphs on ggml's Metal backend,
from the pinned llama.cpp that Maya already builds against, with the GGUF read directly: no pack. Its graph follows
llama.cpp's `glm5-next` model op for op (the mHC streams, the KDA linear-attention layers, the k-pool sparse indexer
and the nope MLA, the MoE), and the numbers are ggml's, the same kernels Maya's CUDA engine was validated against.

What is Maya's own is the memory plan. A Mac has one pool of unified memory, usually smaller than the model:

- **Dense weights, the KV cache and the recurrent state stay resident** (about 14 GB for Maya-S, plus the context).
- **The routed experts (95 % of the model) live in a slot pool.** If every expert fits (128 GB Macs and up for the
  smaller quants), they are all loaded at start and a token runs as one graph. Otherwise the pool takes what the
  GPU's working set leaves: the graph is cut after each MoE router, the engine reads that layer's top 8, reads the
  experts that are not resident from the SSD with parallel reads, and runs them from their slots. The least
  recently used experts make room. At start the pool is filled from this Mac's own expert profile, which the engine
  learns from every request (`expert-profile.bin` beside the model), or from the shipped one
  (`data/expert-profile-glm.bin`) the first time.
- The GPU never pages model data in from the file: that is far slower than the engine's parallel reads into the
  pool.

On a Mac with less unified memory than the model, decode speed depends on how many experts a token needs that are
not resident, and so on the SSD.

## Speed

Not measured by the Maya team yet (no Mac): your `./maya.sh --bench` numbers are welcome in an issue. What to expect:
with less unified memory than the model, part of the experts come from the SSD as a conversation needs them, and decode
is bound by those reads rather than by the GPU - so MTP drafting is off then (`MAYA_GLM_MTP_DECODE=1` turns it on). A
Mac that holds every expert (128 GB and more for the smaller quants) decodes from memory, where drafting pays.

The pool takes no more than the Mac has free when Maya starts, less `MAYA_GLM_RAM_HEADROOM_GB` (6 GB): unified memory
is shared with every other program, and a pool past it would be swapped (a slice that reads swapped experts stalls
for seconds). Close other large programs before starting Maya for the most experts in memory.

## Drafting

Decoding drafts a few tokens ahead and checks them in one pass of the model (a verify window), so each pass over the
weights can write several tokens. The drafts never change the answer: every row of the window is picked exactly as
plain decoding picks it (the same sampler, the same seed counter, the same penalties), and the tokens after the first
draft the model would not have written are dropped, the recurrent state rolled back to that row.
A window's rows are not the same bits as single tokens, though (within 4.9e-6 of them, below): ggml-metal picks its
matrix kernel by the column count, so where two tokens nearly tie, a window of another length can pick the other one.
The MTP draft length adapts to measured timing and is kept across requests, so the same greedy request can take
another path the second time (a prefix-cache hit after a cold prompt, say). `MAYA_GLM_MTP_FIXED=N` fixes the length;
`spec=1` (or `MAYA_GLM_MTP_DECODE=0`) decodes one token at a time.

- **MTP**: the model's own NextN block drafts with chained greedy steps; the draft length adapts to how many drafts are
  accepted and what a round costs on this Mac (as on CUDA: `MAYA_GLM_MTP_DRAFT`, `MAYA_GLM_MTP_FIXED`,
  `MAYA_GLM_MTP_ADAPT`). It is on when every expert fits in memory, where a token is GPU work and a window writes
  several for about the price of one. With experts streamed from the SSD a window needs more of them than one token
  does, so it is off there (`MAYA_GLM_MTP_DECODE=1` turns it on, `0` off).
- **DFlash2** (`--draft dflash2` at setup): a 0.7 GB block-diffusion drafter reads five of the model's layers and
  drafts up to 7 tokens in one pass, kept while the drafter is confident (`spec_min_p`, `spec_n_min`, `spec_cum`).
- A request's `spec=1` turns drafting off; `MAYA_GLM_NO_MTP=1` for the engine.

## Accuracy

The engine holds the CUDA engine's GLM parity gates (`src/core/glm_model_test.cpp`, `glm_model_big_test.cpp`) on the
Metal GPU (`ctest` in a Metal build; the figures below are from the contributor's Mac - GitHub's runners repeat the
small fixture's tests on every change):

| Test | Gate | maya-metal |
| --- | --- | --- |
| `metal_model_test` (6 layers, against `ref/glm.py`) | logits within 2e-3; streamed = batched within 1e-5; greedy chain equal | max 2.7e-6; 3.6e-7; equal |
| `metal_model_big_test` (45 layers, 512 tokens, f64 oracle) | KL p99 < 1e-4, max KL < 0.1, argmax >= 505/512, streamed vs batched < 5e-5 | KL p99 2.1e-8, max 2.3e-2, 511/512, 2.3e-6 |
| the same, the expert pool at its smallest and 64-token slices | the same gates | the same numbers |
| `metal_model_window` / `metal_model_big_window` (verify windows of 1-8 rows, every third draft wrong) | tokens = plain decoding's | equal; logits within 4.9e-6 |
| `metal_model_mtp` / `metal_model_big_mtp` (MTP drafting, fixed lengths 1 / 2 / 4 and adaptive) | tokens = plain decoding's | equal |

Two choices keep it at f32 accuracy where Metal would otherwise round:

- ggml-metal's matrix-matrix kernel rounds its tiles to half precision. Weights stored in float (the router, the mHC
  projections) therefore run as f32 matrix-vector products, one per token, so a prompt and a decode step give the
  same bits. Quantized weights use ggml-metal's matrix kernels as llama.cpp does.
- The MLA latent cache is F32 by default (about 22 KB per token; `--kv-f16` halves it, as the CUDA engine stores it).
  On the GPU the DSA layers' matrix products round it to half anyway, so it is F16 there; `--kv-f32`
  (`MAYA_METAL_KV_F32=1`) keeps it F32, and the fused sparse-attention kernel then reads F32 rows.

## Options

`maya-metal --help` lists them. The ones that matter:

| Option | What it does |
| --- | --- |
| `--max-context N` | context length (set by the setup) |
| `--expert-gb X` | unified memory for the expert pool, in GB (default: the GPU's working set less the dense weights, the cache and 6 GB) |
| `--expert-profile F` | the ranked expert profile that fills the pool at start (set by the setup) |
| `--io-threads N` | parallel SSD reads for expert misses (default 16) |
| `--kv-f16` | the MLA latent cache in F16 |
| `--kv-f32` | keep the MLA latent cache F32 on the GPU too (twice the cache memory; for accuracy A/B runs) |
| `--cpu` | run on the CPU through ggml-cpu (reference runs) |

`maya-metal --gguf MODEL.gguf --ids 1,2,3 --n 64` decodes 64 tokens greedily from those token ids and prints the
speed, the expert hit rate, the bytes read and where the time went.

## Building it by hand

```sh
cmake -S . -B build-metal -DMAYA_ENABLE_METAL=ON
cmake --build build-metal -j
ctest --test-dir build-metal          # the parity tests above (the 45-layer fixture needs Python with numpy)
```
