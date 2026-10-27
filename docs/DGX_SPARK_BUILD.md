# The DGX Spark build: what we changed, where it comes from, what others did

*The aarch64 port for the NVIDIA DGX Spark (GB10) and Grace, by carlos patriawan
([@xcpatr9231975](https://github.com/xcpatr9231975)), 2026-10-24.*

This page documents that work: **what every component is** and **where it comes from**, **what other people did**
that it builds on, and **what the diff actually is** - file by file, with the reason for each change.

The base is Project Maya, and upstream's **v1.1.27** is merged in (commit `3ed58e6`), so this branch is a
fast-forward of upstream's `main` plus the additions below. **43 files changed, +1960 / -132** over the v1.1.26
base; the merge itself brought their 26 files and was re-tested (68/69 C++ tests, 369 server, 395 tools, and the
benchmark re-measured at 21.2 tokens/s against 21.1 before it).

This pull request carries the two code changes and their tests and docs:

| | |
| --- | --- |
| **The aarch64 port** | the engine builds and runs on aarch64 (GB10, Grace); x86-64 is unchanged |
| **Allocation-failure recovery** | an out-of-memory moment in the prompt path ends the request, not the process |

---

## 1. What the machine is, and what the model is

| | |
| --- | --- |
| Machine | one NVIDIA DGX Spark: GB10 GPU (compute capability 12.1, `sm_121`), 20 ARM cores, **120 GB of RAM shared with the GPU** |
| Software | Ubuntu 24.04 aarch64, driver 580.95.05, CUDA 13.0 (nvcc 13.0.88) |
| Model | `GLM-5.3-Flash-Maya-S-v2-IQ2_XXS`, 89.83 GiB, 320.76 B parameters (18 B active), 46 layers, 288 experts (8 used), 1M-token context, run at 131072 |
| Result | 21.1 tokens/s decoding with the DFlash2 drafter (16.8 without it), 769 tokens/s reading a 16K-token prompt, every expert resident, 99.97% cache hit |

The one thing that makes a DGX Spark different from every machine Maya ran on before: **the GPU has no VRAM of its
own**. Its memory is the system's RAM, so the expert pool, the prompt buffers, the pinned tiers and the desktop all
draw on the same 120 GB. That single fact drives most of the port.

---

## 2. Where every component comes from

Maya's engine is a C++/CUDA program built from several independent code bases. This is what each one is and what it
provides; the licence notices stay with every copy (`third_party/*/LICENSE`).

### What Maya's own code is (`src/`, `serve/`, `tools/`, `ref/`)

| Component | What it is |
| --- | --- |
| `src/core/`, `src/kernels/` | **Maya's own engine**: the GLM-5.3-Flash (llama.cpp's `glm5-next` architecture) implementation, the expert pool with its VRAM / RAM / SSD tiers, the layer split, MTP and speculative decoding, the verify windows, the prompt (prefill) path |
| `src/program/generate.cpp` | the engine's command line and its `--serve` protocol (`GEN` / `GENI` / `DONE` / `ERR` over stdin/stdout) |
| `serve/server.py` | **Maya's own HTTP server**: the OpenAI- and Anthropic-compatible API, the dashboard, the streaming, the request queue |
| `tools/` | the packer (`iq_pack.py`), the installer (`maya.py`, `setup.py`), the benchmark, the parity tests |
| `ref/glm.py`, `ref/load.py` | **Maya's own Python reference forward pass** for the architecture, one op at a time - the specification the C++ kernels are parity-tested against (it is written from the upstream snapshot below) |

Maya's engine started from **Strata** ([Niko1221/Strata](https://github.com/Niko1221/Strata), MIT) and was rewritten
for GLM-5.3-Flash; its server and dashboard grew from Strata's too.

### What is taken from other projects

| Component | Where from | What it provides here |
| --- | --- | --- |
| `third_party/llama.cpp` | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), MIT, **pinned at `3cf03257`** (setup.py's `LLAMA_CPP_COMMIT`) | the quantization formats (IQ2_XXS, IQ4_NL, Q2_0, ...), **ggml-cpu's dot products**, the **prompt path's MMQ kernels** (`src/prefill/` compiles a subset of ggml-cuda with them), `mtmd` for the vision encoder, and ggml's build |
| `third_party/ggml` | the same project, MIT | a **patched copy** of llama.cpp's ggml-cuda sources, with `mmq-fused-swiglu.patch` (the fused gate/up + SwiGLU product) applied at configure time |
| `third_party/exllamav3` | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3), MIT | the EXL3 format, its codebooks, the CPU expert kernel and the tensor-core decoding of a tile |
| `third_party/beellama.cpp` | [Anbeeld/beellama.cpp](https://github.com/Anbeeld/beellama.cpp), MIT | the DFlash2 drafter's forward pass |
| `ref/upstream-llama.cpp/` | **llama.cpp master @ `272aad8b9`** (fetched 2026-09-30), MIT | a read-only snapshot of upstream's `glm5-next.cpp` and friends - the specification `ref/glm.py` transcribes. Documentation only: it does not compile outside llama.cpp and must not drift into `third_party/ggml` |
| `third_party/wheels/` | PyPI packages | the setup screen's Python packages, installed offline |

### The model files

- **GLM-5.3-Flash** by zai-org (MIT, Hugging Face) - the released weights Maya's quant and the image encoder are made
  from.
- **Maya-S-v2-IQ2_XXS** - the quant used here: 89.83 GiB across three GGUF shards, packed by `tools/iq_pack.py` into
  `pack/` (the dense weights, the native expert blobs and the tokenizer).
- The optional **DFlash2 drafter** ([Anbeeld's GGUF](https://huggingface.co/Anbeeld/GLM-5.3-Flash-DFlash2-GGUF), CC
  BY-NC-ND 4.0, non-commercial) - downloaded only when chosen; it is what the server running on this machine uses
  (`--dflash`).

---

## 3. What other people did

This build stands on work by many people. The order below is "what this build needed, and whose it was".

**The project itself.** Project Maya is created and led by **Peasant Smith** ([@mw00](https://github.com/mw00)), who
owns the project, the name and the Maya models. **@needmorevram** is the co-maintainer and contributed the parts this
build leans on hardest: **the expert tiers across VRAM, RAM and SSD** (the pool our port sizes differently), **the
CPU expert kernels**, the expert profile, the layer split, KV streaming, conversation checkpoints, MTP decoding and
DFlash2 drafting.

**The AMD (HIP) port, and the unified-memory policy.** Before this build, Maya already ran on a machine whose GPU
memory is the system's RAM: **AMD's Ryzen AI Max / Strix Halo APUs on Linux** (the experimental HIP path, measured and
documented in `docs/AMD_MAYA.md`). That port established the policy this build reuses - size the expert pool from
`/proc/meminfo`'s `MemAvailable` rather than the runtime's free figure, because on a unified device the runtime
counts the model's page cache as used. Our change lifts that policy from "AMD only" to "any device whose memory is
the system's, including a GB10". Contributors to the AMD path are credited across the CHANGELOG (@rkarp, @izabera,
@Gazelle221B, @pepuscz and others).

**The DGX Spark measurement.** The README's benchmark table already carried a DGX Spark row - **16.5 tokens/s,
~730 tokens/s on a 6K prompt, reported by @ericoh88 on v1.1.15 (#282)**. Our measurement (16.8 tokens/s, 730 tokens/s
on an 8K prompt) reproduces it on the same hardware, which is what makes the port's numbers trustworthy.

**llama.cpp's own implementation of this architecture.** Upstream llama.cpp master carries `glm5-next` - the same
GLM-5.3-Flash architecture - so it can run the same GGUF. We built it for `sm_121a` on this machine and measured it:
it is the baseline in [BENCHMARK.md](../BENCHMARK.md) (Maya reads a prompt up to 3.2x faster there; the decode is a
tie at a short context and 3-7% faster at a long one). The snapshot in `ref/upstream-llama.cpp` is that
implementation, and it is also the specification `ref/glm.py` follows.

**The community's model work.** The GGUF this runs is a community quant of zai-org's weights; the speed reports in
the README's table come from dozens of people on every kind of GPU.

---

## 4. What the diff is, and why

### 4.1 The aarch64 port (`fde5e98`, 28 files)

The engine's fast expert kernels are x86 (AVX-512 and AVX2). On an aarch64 CPU they cannot run, and code compiled for
them would execute an illegal instruction. The port leaves them out and routes the work through ggml-cpu's own NEON
paths instead.

| File | Change | Why |
| --- | --- | --- |
| `include/strata/platform/cpu_relax.hpp` | **new**: `strata_cpu_pause()` and `strata_store_fence()` | the doorbell spin used `_mm_pause` and the store fence `_mm_sfence`, both x86-only. aarch64 takes `yield` and `dmb oshst` - the outer-shareable store barrier is what makes host stores visible to a coherent device before the doorbell it polls |
| `src/core/graph.cpp`, `session.cpp`, `verify.cpp`, `src/kernels/cpu/pool.cpp` | use the two helpers above | four spin loops and three store fences, all previously x86-only |
| `src/kernels/cpu/portable.cpp` | **new**: stubs for every x86 kernel (Q2_0, i-quant, k-quant, EXL3, the IQ4_NL multi-token kernel) | a build must not silently compute something else. Each stub prints what is missing and stops; the engine refuses a Q2_0 pack at startup (`cpu_require_expert_support()`) instead of crashing at token 4000 |
| `include/strata/kernels/cpu/expert_layout.hpp`, `src/kernels/cpu/expert_layout.cpp` | `q2_own_kernels()`; the AVX-512 checks answer "no" on non-x86 | the same refusal, and no CPUID on ARM |
| `src/kernels/cpu/native_expert.cpp`, `src/core/expert_source.cpp` | gate the Q2_0 and IQ4_NL kernels on x86 | the native packs go through ggml-cpu's NEON dot products instead |
| `src/ds4/ds4_iq_cpu.cpp` | `IQ_X86` guard around the timing/variant code | the AVX-512 variants are x86-only |
| `src/core/glm_fast_path.cu`, `glm_fast_state.hpp` | **the unified-memory policy for CUDA**: a device whose memory is the system's sizes its pool from `MemAvailable`, and the RAM-tier headroom is 16 GB there | the runtime's free figure counts the model's page cache as used - it read ~2 GB free with the 90 GB pack cached, so the pool came out at 5 slots a layer and the start failed. `cudaDevAttrIntegrated` is 1 on a GB10, so this is now a device property, not an AMD-only branch |
| `CMakeLists.txt` | `STRATA_CPU_X86`; the x86-only sources, flags, tests and benchmarks gated on it; `GGML_NATIVE` off on ARM with `GGML_CPU_ARM_ARCH` | the x86 kernels are compiled with `-mavx512f...`; on aarch64 ggml-cpu needs its own arch string |
| `setup.py` | `ARM`, `arm_cpu_arch()` (from `/proc/cpuinfo`'s features), `cpu_info()`, and `unified_memory_pci_ids()` (a GPU with no memory BAR of its own) | the installer must recognise the machine, pick ggml's `-march` (`armv8.6-a+dotprod+i8mm+fp16` on the GB10) and report "120 GB shared with the system" instead of "120 GB of VRAM" |
| `maya.py` | the ARM branch in `check_pc`, the GPU label, and `-DGGML_CPU_ARM_ARCH` passed to the engine and vision builds | the setup's checks and messages |
| `tools/vision/CMakeLists.txt` | the same ggml arch handling | the image encoder builds for aarch64 too (verified: `build-vision/bin/maya-vision`) |
| `src/kernels/cpu/expert_multi_test.cpp`, `src/platform/memory_test.cpp`, `tests/exl3_test.cu` | skip where the kernel or the lock limit is missing | a test that cannot run must say so instead of failing |
| `serve/server.py` | `STRATA_HTTP_MAX_CONNECTIONS` (64) and `STRATA_HTTP_READ_TIMEOUT_S` (30 s) | a stuck client held a handler for ever; a stalled body is now a 408 and an over-limit request a 503 with `Retry-After: 1` |
| `serve/test_server.py` | tests for both limits | |
| `serve-maya.sh` | **new**: the local entrypoint - start / stop / status / logs | and it *finds a server it did not start*, which the pid file alone did not |
| `docs/SETTINGS.md` | the unified-memory paragraph, the headroom row, the two new server settings | |

**Measured on the result:** 16.8 tokens/s, 730 tokens/s on an 8K prompt, all 12096 experts resident, 99.97% cache
hit, a 42-second restart - matching @ericoh88's reported numbers on the same hardware.

### 4.2 The bring-up documentation and the EXL3 refusal (`8545eed`)

- `docs/DGX_SPARK.md` (**new**, 214 lines): the machine, the setup, running it, what the port changed, how the pool
  is sized on a unified device, the measured numbers, what is still missing, troubleshooting.
- `maya.py`: the setup now **refuses the DeepSeek EXL3 quants on an ARM CPU** (their expert kernels are x86-only),
  offering the GGUF quant instead - so nobody downloads ~300 GB of weights that cannot run.
- `README.md`, `docs/SETTINGS.md`: links.

### 4.3 The benchmark (`7ca21cf`)

- `tools/bench_tokens.py` (**new**): a sweep of the sizes the standard `--bench` does not do - the prompt read at
  256-32768 tokens with the time to its first token, and the answer written at 32-1024 tokens after a prompt of a
  given size. It drives the engine alone, the way the dashboard starts it.
- `BENCHMARK.md` (**new**): the numbers on this machine, and **the same GGUF on the same GPU with upstream
  llama.cpp** (`272aad8b9`, built here for `sm_121a`). Prefill: 754 vs 239 tokens/s at 32K tokens (**3.16x**).
  Decode: a tie at a short context (17.4 vs 17.7) and 1.03-1.07x Maya's way as the context grows (upstream loses
  ~12% of its rate between an empty context and 2K tokens, Maya ~4%).

### 4.4 Allocation-failure recovery (`e6f317c`, 16 files)

This one is not aarch64-specific: it is the "known gap" `docs/SETTINGS.md` had documented.

**What was broken.** An out-of-memory moment in the prompt path ended the *process*: the prefill's kernel launches
(28 sites), the MMQ kernels (8) and the prefill GEMMs called `std::exit(1)`, and the MMQ scratch pool's `cudaMalloc`
went through `CUDA_CHECK` into `std::abort()`. The client got a 503, the next request waited for the engine to load
again, and the expert cache had to be rebuilt from disk. Separately, `prefill_carve` carved its layout **past the
lendable region and only printed a warning** - an out-of-bounds pointer.

| File | Change |
| --- | --- |
| `include/strata/prefill/failure.hpp` | **new**: `strata::prefill::Failure` (a `std::runtime_error`), `fail()`, and `is_allocation_failure()` |
| `src/prefill/kernels.cu`, `moe_mmq.cu`, `gemm.cu` | every `std::exit(1)` becomes a thrown `Failure` |
| `src/prefill/ggml_cuda_host.cu` | the scratch pool's `cudaMalloc` failure throws (with the bytes it wanted and the bytes free) instead of aborting |
| `src/core/glm_prefill.cu` | `prefill_carve` returns false when the layout does not fit; `Glm5Model::prefill` reads the prompt again with a **smaller chunk** (down to 256 tokens) when it does not; `prefill_run` is guarded so the expert pool's tail is given back either way |
| `src/program/generate.cpp` | `guarded()` catches a request's failure; `cuda_context_broken()` still ends the engine for a sticky CUDA error; the serve loops end the *request* (not the process) and lower the chunk for the requests that follow |
| `src/prefill/prefill.cpp` | the same chunk-halving retry for the borrowed layout, with the `STRATA_PREALLOC_FAIL` test hook |
| `serve/server.py`, `serve/test_server.py` | a memory failure is a **503** (the client can try again), a wrong request still a **400** |
| `src/prefill/failure_test.cpp`, `tools/test_alloc_recovery.py` | **new tests**: the classification (no GPU), and both injected failures against the real engine - the request fails, the engine answers the next one |

Three `std::exit(1)` sites are deliberately left, because none of them is a memory failure: an MMQ type this build
does not cover (`moe_mmq.cu`), a build without the fused gate/up product (same file), and ggml's own init-time CUDA
checks (`ggml_cuda_error`, reachable only from `ggml_cuda_info` / `ggml_cuda_get_device` at start-up). The AMD-only
invariant checks in `moe_fused_rdna4.cu` are untouched.

---

## 5. How to verify any of this

```sh
git log --oneline f26d8ba..HEAD      # the four commits
git diff --stat f26d8ba..HEAD        # 43 files, +1960 / -132

./maya.sh --check                    # the PC checks: the GB10, CUDA 13, the ARM CPU, 120 GB shared
./serve-maya.sh start                 # ~42 s to load and warm the tiers
curl http://127.0.0.1:8080/v1/models

./serve-maya.sh stop
.venv/bin/python tools/bench_tokens.py --out BENCHMARK.md   # the sweep in BENCHMARK.md
.venv/bin/python -m unittest tools.test_alloc_recovery      # the recovery, on the real engine
```

The C++ tests (`build-tests`, `ctest`): 68 of 69 pass; `ple_parity` needs the maintainer's fixtures
(the Qwen3.8-Flash-Next GGUF and `bench/micro/ple_*.bin`), which no machine has outside the project. The Python
tests: 392 tools and 360 server tests pass.

## 6. What is still missing

- **Images**: the encoder compiles for aarch64, but its 1.1 GB of files were not downloaded on the machine this was
  brought up on - that install is text only.
- **`ple_parity`** needs the maintainer's fixtures (see above).
- The **canonical Q2_0 pack** and the **DeepSeek EXL3 quants** are x86-only (the setup refuses EXL3 on ARM now; a
  Q2_0 pack is refused at the engine's startup).

