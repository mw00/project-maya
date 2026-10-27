# Maya on an NVIDIA DGX Spark (GB10) and Grace

A single DGX Spark runs Maya-S (GLM-5.3-Flash) at its full context, with every expert in the GPU's own pool. The
catch is that a DGX Spark has no VRAM: the GB10 draws on the system's RAM, so the expert pool, the prompt buffers and
the desktop all share the same 120 GB. This page is what a bring-up needs to know about that, what the aarch64 port
had to change, and what is still missing.

Measured on one DGX Spark (see [Measured](#measured)): **16.8 tokens/s** decoding, **730 tokens/s** reading an
8K-token prompt, every expert resident, a 99.97% cache hit. The config this setup installs also drafts with a
DFlash2 drafter, which takes the decode to **21.1 tokens/s** (greedy; see
[Drafting and several requests](#drafting-dflash2-and-several-requests-at-once)), and a second config answers four
requests at once for 1.80x the tokens in all. [BENCHMARK.md](../BENCHMARK.md) has the numbers at every token size,
and the same model on the same GPU with upstream llama.cpp.

What the aarch64 port changed, where each component of the build comes from, and what other people's work it stands
on are in [DGX_SPARK_BUILD.md](DGX_SPARK_BUILD.md).

## The machine

| | |
| --- | --- |
| GPU | NVIDIA GB10, one device, compute capability 12.1 (sm_121) |
| Memory | 120 GB, shared with the system (no VRAM of its own) |
| CPU | 20 ARM cores (Cortex-X925, CPU part 0xd87) |
| Driver | 580 or newer (CUDA 13 needs 580) |
| Toolkit | CUDA 13.0 or newer, `nvcc` on the path |
| OS | Ubuntu 24.04, aarch64 |

Grace (NVIDIA's aarch64 CPU, as in a Grace Superchip or a GH200) has the same shape: an aarch64 host whose GPU draws
on memory the CPU shares with it. A discrete NVIDIA card on an aarch64 host (a card in a Grace server, for example)
takes the ordinary discrete path - the unified-memory rules below apply only where the GPU's memory is the system's.

## Bringing it up

The installer is the same one every platform uses; nothing about the setup is DGX-specific.

```sh
./maya.sh --check                 # the PC checks only
./maya.sh --setup --yes --gpu 0 --context 131072 --no-vision --download-model
```

`--check` prints what the engine will do with the hardware:

```
  NVIDIA GPUs:
    GPU 0 (NVIDIA GB10, 120 GB shared with the system) - can be used
  [ok] using GPU 0 (NVIDIA GB10, 120 GB shared with the system)
  the GPU's memory is the system's RAM: the engine fills it with the most-used experts and leaves the rest to the RAM
  tier and the SSD, so the free RAM at the start is what the pool can take
  [ok] CUDA toolkit 13.0: /usr/local/cuda/bin/nvcc
  [ok] NVIDIA driver 580.95.05
  [ok] RAM: 120 GB, 16 GB available now
  [ok] CPU: ARM (CPU part 0xd87) (ARM64, 20 threads; the expert kernels use ggml-cpu's NEON paths)
```

The setup downloads about 96 GB of GGUF shards (Maya-S-v2-IQ2_XXS) and writes the pack next to them, so the models
folder wants a fast NVMe SSD with ~100 GB free. `--no-vision` skips the image encoder (see
[Still missing](#still-missing)); without it the setup also fetches 1.1 GB of vision files and builds the encoder.

The context costs VRAM that the experts want, so 131072 leaves a pool of about 80 GB - enough for every expert of
Maya-S. A smaller context (32768, the default) leaves more room for the pool, but at 131072 every expert already
fits, so there is nothing to gain from it here: the context only costs state and KV room.

## Running it

The engine answers on an OpenAI-compatible API and serves a dashboard:

```sh
./serve-maya.sh            # start (waits until the API answers)
./serve-maya.sh status     # is it up, and where
./serve-maya.sh logs       # the server's and the engine's logs
./serve-maya.sh stop
```

- API: `http://127.0.0.1:8080/v1` (`/v1/chat/completions`, `/v1/responses`, `/v1/messages` for Anthropic's shape)
- Dashboard: `http://127.0.0.1:8080/`

`serve-maya.sh` finds a server it did not start (one a supervisor or a hand started), so `status` and `stop` work
whatever started it. The first start reads ~86 GB of experts: about 40 s to load them and warm the tiers, and the
first answers wait for that.

The default config drafts with DFlash2; the parallel-4 config (four requests at once, no drafting) is started with
`MAYA_CONFIG=$PWD/maya-maya-s-v2-iq2_xxs-parallel4.json ./serve-maya.sh` - stop the running one first, they are the
same model ([Drafting and several requests](#drafting-dflash2-and-several-requests-at-once)).

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Hello"}],"max_tokens":200}'
```

## What the aarch64 port needed

The engine's fast expert kernels are x86 (AVX-512 and AVX2). On aarch64 the port leaves them out and routes the
work through ggml-cpu's own NEON paths:

| | x86-64 | aarch64 |
| --- | --- | --- |
| The Q2_0 pack's expert kernel | AVX-512 VNNI/VBMI | not built: the native packs use ggml-cpu instead |
| The EXL3 expert kernels | AVX-512 / AVX2 / scalar | not built: the EXL3 quants do not run (the setup refuses them) |
| The native packs (IQ2, IQ3, Q4, ...) | this repository's kernels | ggml-cpu's NEON kernels |
| The spin wait | `_mm_pause` | `yield` |
| The store fence | `_mm_sfence` | `dmb oshst` |
| The CPU feature checks | AVX-512 present or not | always "no AVX-512", so no illegal instruction |

`src/kernels/cpu/portable.cpp` replaces the x86 kernels in the build. Its stubs do not silently compute something
else: they print what is missing and stop, because a wrong answer is worse than no answer. The canonical Q2_0 pack
therefore refuses to start on an ARM CPU with a clear message (`cpu_require_expert_support()`), and the IQ2_XXS pack
this setup uses is a native pack - it goes through ggml-cpu.

ggml-cpu is built with the CPU's own features, taken from `/proc/cpuinfo`:

```
-march=armv8.6-a+dotprod+i8mm+fp16
```

The GB10 has `asimddp` and `i8mm`, so the dot-product and the 8-bit integer matrix-multiply paths are on (its `bf16`
is not in the string: ggml's ARM backend does not use it). `setup.py` picks the string from the kernel's feature list,
so the same setup on an older ARM CPU picks a smaller one.

## The expert pool on a unified-memory device

A discrete card reports its own free VRAM, and the pool takes it. A unified-memory device cannot: the runtime's free
figure counts the model's file-backed page cache as used (on the test machine it read about 2 GB free with the 90 GB
pack cached), so the pool would come out far too small - the start fails with

```
maya generate: glm fast: CUDA0 has no room for the expert pool: 5 slots a layer, at least 8 needed (about 604 MiB
more). To make room: a smaller --max-context, --kv int8, or close other programs that use the GPU
```

The pool is therefore sized from `/proc/meminfo`'s `MemAvailable`, which leaves the reclaimable cache out - the kernel
evicts it as the experts are read back in:

```
glm fast: CUDA0 unified memory: MemAvailable 99.17 GB, headroom 16.00 GB + 1024 MB reserve, experts need 80.17 GB,
           pool budget 80.17 GB
glm fast: CUDA0 layers [0,45) expert pool 80.17 GB, 288-288 slots/layer (12096 total)
glm fast: CUDA0 all experts fit in the GPU pool; RAM tier is staging only
```

- `headroom 16.00 GB` is `MAYA_GLM_RAM_HEADROOM_GB` - the RAM left free for the system (the desktop, the file
  cache, everything else that is not Maya). Raise it on a busy machine; lower it when the pool is short.
- `1024 MB reserve` is the VRAM kept free beside the pool for the driver, the sampler and what is allocated later (a
  unified device uses 1024 MiB, a discrete card 700).
- Nothing is checked after the pool is written: the free figure moves with the page cache, so a second check would
  give the memory back and then take it away. The start fails only if the pool cannot hold one route's experts in
  every layer, and it then says how many MiB more it needs.

Because the pool is RAM, **other programs on the machine take from it**: the pool is sized from what is free at the
start, so a desktop with 40 GB of its own applications running leaves the engine that much less. The RAM tier (the
pinned tier below the pool) is staging only when every expert fits, as it does here.

The CPU lane - the threads that compute the experts that do not fit in the pool - measures itself against the copy
over the link at the start and takes fewer experts when the link is faster:

```
glm fast: CUDA0 CPU lane: 13 threads, an expert 0.171 ms on the CPU vs 0.117 ms over PCIe -> of 0..8 RAM-tier experts
          the CPU takes 0 0 1 1 1 2 2 3 3
```

On this machine the link is the faster way, so the CPU lane takes little, and when every expert fits in the pool it
takes none at all (`glm fast: CUDA0 CPU lane off: the RAM tier holds no expert`).

## Measured

`./maya.sh --bench` with Maya stopped (the standard measurement: three greedy answers of 256 tokens with the
thinking off, and a prompt read at 2048 and 8192 tokens). This is the baseline without drafting - the config this
setup installs drafts with DFlash2, which takes the decode to 21.1 tokens/s
([Drafting and several requests](#drafting-dflash2-and-several-requests-at-once)):

```
Project Maya v1.1.26 benchmark, 2026-10-24 11:36
GPUs: NVIDIA GB10 120 GB; RAM 120 GB; CPU ARM (CPU part 0xd87)
setup: maya-maya-s-v2-iq2_xxs.json, context 131072, GPUs [0]
decode (writing the answer): mean 16.8 tokens/s over 3 answers of 256 tokens (greedy, thinking off)
prefill (reading the prompt): 597 tokens/s, 2048 tokens
prefill (reading the prompt): 730 tokens/s, 8192 tokens
```

The engine's own lines for the same run:

```
glm stat: decode 59.34 ms/tok (16.85 tok/s) | vram hit 99.97% | ram fetches 0.00/tok | disk 0.09 reads/tok
```

Every expert sits in the pool (12 096 of 12 096), so a token needs no disk and no RAM: the experts are read into the
pool at 8.7 GB/s at the start (86 GB in 10 s), and the decode is bound by the kernels.

These numbers reproduce the earlier DGX Spark row in the README's table (16.5 tokens/s, ~730 tokens/s on a 6K
prompt, reported by @ericoh88 on v1.1.15); the row above it is this build, at 21.1 tokens/s with the drafter.

## Drafting (DFlash2) and several requests at once

The pack this setup uses (Maya-S-v2-IQ2_XXS) has no MTP draft block, so a plain install decodes token by token. Two
ways to get more out of the machine were measured here. **They are mutually exclusive**: with `--parallel` above 1 the
several-conversation scheduler decodes without the drafter (`src/program/generate.cpp`: `glm dflash: not used with
MAYA_GLM_SEQS=%d`), so each lives in its own config and only one runs at a time.

### DFlash2 - the default config

`./maya.sh --setup --draft dflash2 --no-vision` downloads the drafter (0.7 GB,
[on Hugging Face](https://huggingface.co/Anbeeld/GLM-5.3-Flash-DFlash2-GGUF); license CC BY-NC-ND 4.0, non-commercial
use only) and writes `--dflash <GGUF>` into the config's `"args"`. A GB10 (sm_121) is eligible - DFlash2 is offered on
NVIDIA from sm_80 (RTX 30 / A100) on. The setup then runs the speed calibration, which measured both sides here (the
model loaded twice, 1536 tokens a side after a warm-up, the config's own sampling):

| | sustained (tokens/s) | |
| --- | ---: | --- |
| no drafting | 16.1 | |
| DFlash2 | 17.4 | +8%, 92% of the drafts accepted |

`./maya.sh --bench` (greedy, the engine alone, thinking off): **decode mean 21.1 tokens/s** over three 256-token
answers against 16.8 without it (+26%); prefill is unchanged (614 / 732 tokens/s at 2048 / 8192 against 597 / 730) -
the drafter only changes decode. The greedy gain is the larger one because every draft is then accepted. The drafter
costs nothing here: the pool is sized from `MemAvailable`, and every expert still fits beside it (12 096 of 12 096,
the RAM tier staging only):

```
glm dflash: 5 layers, hidden 4096, 32 heads (8 kv), block 8, window 2048, selector top-16 rank 256,
            reads trunk layers 6,15,25,34,43; 717 MB on CUDA0
glm fast: CUDA0 all experts fit in the GPU pool; RAM tier is staging only
```

The result is kept in the config (`installer.speed_calibration`: the two speeds, the kept choice), so a start does not
measure again by itself; `./maya.sh --speed-calibration` measures again at any time (the GPUs to itself, 5-20
minutes).

### Several requests at once - the parallel-4 config

`--parallel 4` answers up to four requests at once: a token of each goes through the model together, so every dense
weight is read once for all of them. It is a second config, `maya-maya-s-v2-iq2_xxs-parallel4.json` (the same
arguments, `--parallel 4`, no `--dflash` - the drafter is not used with it - and its own engine log), started with:

```sh
MAYA_CONFIG=$PWD/maya-maya-s-v2-iq2_xxs-parallel4.json ./serve-maya.sh
```

Measured with the engine's own multi bench (`STRATA_GLM_MULTI_BENCH=4,512,256`: four synthetic 512-token prompts,
256 greedy tokens each, sequence 0 alone first as the reference):

| | tokens/s | |
| --- | ---: | --- |
| one request alone | 17.2 | |
| four at once, together | 30.9 | 1.80x in all, 7.7 each |
| the tokens | 256 of 256 the same | each request gets what it would get alone |

Each extra request keeps its own context state, taken from the expert pool (the start log's `glm multi:` line gives
the size): with four running, the VRAM hit was 99.8% and a token saw 0.67 disk reads, against 99.97% and 0.09 for one
- a few more experts live outside the pool, which is what the aggregate gain is paid from.

### Which one to run

- One request at a time (a chat, an agent, `opencode`): the default config - DFlash2 drafts, 21.1 tokens/s greedy.
- Several at once (agents with workers, several people): the parallel-4 config - 30.9 tokens/s in all for four, no
  drafting (a request alone still runs at 17.2).
- `./serve-maya.sh stop` before switching: the two configs are the same model, so only one can hold the GPU.

## What was tried on this machine (the enhancement log)

| | | |
| --- | --- | --- |
| DFlash2 drafter | **done** | +8% sampled / +26% greedy decode, the same answers; the default config drafts now |
| `--parallel 4` | **done** | 1.80x for four at once, identical tokens; the second config |
| Vision | not used | `--no-vision`: the encoder builds for sm_121, but its 1.1 GB of files and ~320 MB on the GPU buy nothing here |
| The 27-29 s load | skipped | agreed not worth the work: only the first start is slower, the steady state is not |
| Prefill at short prompts (0.96x at 256-512 tokens) | skipped | the advantage was uncertain, and the work is engine-level |
| The Q2_0 and EXL3 kernels on NEON | skipped | they unlock other packs on ARM, not this setup's |

## Still missing

- **The canonical Q2_0 pack** is x86-only. Maya-S-v2-IQ2_XXS (a native pack) works; a Q2_0 pack does not, and the
  engine says so at startup instead of failing later.
- **The EXL3 CPU lane** is x86-only, so the DeepSeek EXL3 quants have no CPU lane here: their expert kernels are
  AVX2/AVX-512. The setup refuses them on an aarch64 host and points at the GGUF quant instead:

  ```
  [X]  DeepSeek-V4.1-Flash-1017's EXL3 quant needs the x86 expert kernels (AVX2/AVX-512), which this ARM CPU does not
       have
       set up its GGUF instead: ./maya.sh --setup --model DeepSeek-V4.1-Flash-1017 --ds4-quant gguf-q8
  ```
- **The image encoder** builds for aarch64 (`tools/vision`, CUDA sm_121) and its files exist, but this setup
  deliberately leaves it out (`--no-vision`): 1.1 GB of files and ~320 MB on the GPU for pictures this box does not
  need. Drop the flag and re-run the setup to have it.
- **Allocation failures no longer end the engine** (the CHANGELOG's Unreleased section has the details): a failure
  in the prompt path ends that request with an error (503) and the next one runs, and where the layout does not fit
  the prompt is read again with a smaller chunk (down to 256 tokens). On a unified device that matters most - an
  out-of-memory moment no longer costs the ~40 s reload and the expert cache rebuild.
- **`ple_parity`** (a C++ test) needs the maintainer's fixtures - the Qwen3.8-Flash-Next GGUF and
  `bench/micro/ple_*.bin` - and fails without them on any machine, not only here.

## Troubleshooting

**"has no room for the expert pool"** - the pool came out too small for one route's experts in every layer. On a
unified device that means the RAM was not free: close what holds it, or lower `MAYA_GLM_RAM_HEADROOM_GB` (16 GB by
default here) or `--max-context`.

**The pool is smaller than expected** - it is sized from the free RAM at the start, so anything else on the machine
is subtracted. The start log says what it took (`pool budget ... GB`); compare it with `MemAvailable` in
`/proc/meminfo`.

**The decode is slower than the table says** - check the engine's `glm stat:` line. `vram hit` below 99% means the
experts are not all resident (the pool was too small), and `cpu lane` above zero means the CPU is computing experts
it should not.

**The build fails in ggml** - check the `-march=` line in the configure output. It comes from `/proc/cpuinfo`'s
feature list; a CPU without `i8mm` gets a smaller one.
