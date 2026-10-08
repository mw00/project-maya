# Experimental Maya on RX 7900 XT / XTX and R9700 / RX 9070

This branch adds a Linux HIP build and installer path for Maya's GLM-5.3-Flash
engine on `gfx1100` and `gfx1201`. It uses one GPU and serves text. Images and HIP multi-GPU
inference are not enabled by this installer.

Use a system ROCm 7 installation with its HIP compiler and hipBLAS, Python
3.10+, CMake 3.24+, and a C++20 compiler. `ROCM_PATH` selects an installation
outside `/opt/rocm`. Everything Maya downloads goes into the project or its
model-data folder; setup does not install system packages.

```sh
./maya.sh --backend hip --gpu 0 --check
./maya.sh --backend hip --gpu 0 --context 8192 --no-vision --yes \
  --download-model --no-start --port 8099 --env STRATA_GLM_RAM_GB=60
./maya.sh --backend hip --port 8099
```

The first setup downloads about 96.5 GB of weights and verifies the SHA-256
hash of each shard. `--gguf-dir DIR` uses existing files instead. GPU numbers
are the kernel KFD topology order shown by `--check`; on the test machine the
7900 XT is GPU 0, an R9700 is GPU 1, and the integrated GPU is GPU 2. This
installer accepts `gfx1100` and `gfx1201` and compiles one binary for
`gfx1100;gfx1201;gfx1151`. The last target **builds, untested**: it is not
enabled by setup until the separate unified-memory work is validated.
Configs are named `maya-<quant>-hip.json` and
select the AMD device through `HIP_VISIBLE_DEVICES`.

The HIP config starts with an 8K context when requested above, 3 GiB of GPU
headroom, and 16 GiB of system-RAM headroom. The engine sizes the prompt chunk
from its prompt-memory budget. The example additionally caps the pinned expert
cache at 60 GiB. Change those settings with `--env KEY=VALUE` during setup.
Available RAM, rather than installed RAM, determines how much can be cached.
The engine reduces its RAM-tier allocation if ROCm cannot pin the requested
amount.

## RAM shadows (`STRATA_GLM_RAM_SHADOW=1`)

With the default exclusive tiers an expert lives in VRAM or in the RAM tier, never both: promoting it frees its RAM
copy, and when it is evicted from VRAM later it is either copied back or reread from the SSD. With
`STRATA_GLM_RAM_SHADOW=1` the RAM copy stays as a shadow while the RAM tier has room; shadows are the first RAM slots
reclaimed, and an evicted expert that still has one needs no copy back. It helps when the RAM tier is large enough to
hold most of the model (here a 90 GB tier). Measured on v1.0.11, Maya-S, greedy, 256-token answers:

| | exclusive | RAM shadows |
|---|---|---|
| one RX 7900 XT, answers | 15.2 tok/s | 18.5 tok/s |
| one RX 7900 XT, SSD read per request | 13.2 GB | 4.6 GB |
| R9700 + RX 7900 XT, answers | 33.2 tok/s | 35.6 tok/s |

## Prompt speed

Prompts run in chunks, and inside a chunk the mixers and the dense FFN run in
sub-batches. Two settings matter on AMD:

- `STRATA_GLM_PREFILL_SUB` sets the sub-batch size. It was a fixed 256; it is
  now a runtime setting with 256 as the default. Larger sub-batches give the
  matrix cores bigger GEMMs. Setup writes `1024` together with
  `STRATA_GLM_PREFILL_MB=4096` for cards with 20 GB or more; the extra prompt
  memory is borrowed from the expert pool only while a prompt runs.
- `STRATA_HIPBLASLT_TUNING` points at a table of hipBLASLt solutions for the
  prompt's FP16 projections, measured per architecture with
  `tools/hip/tune_hipblaslt` (`tools/hip/<arch>-glm-hipblaslt-<version>.txt`).
  Setup writes it when a table for the card exists. The engine accepts a table
  only for its own architecture and hipBLASLt version; otherwise, and for shapes
  without a row, it keeps plain hipBLAS. The tuner's absolute-error gate now
  scales with `sqrt(K / 4096)`, so long-K projections are no longer rejected.

Measured prompt speed, GLM-5.3-Flash Maya-S-v2 IQ2_XXS, ROCm 7.2:

| Setting | RX 7900 XT | Ryzen AI Max+ 395 (8060S) | R9700 |
|---|---|---|---|
| prompt chunk forced to 256 (the earlier config) | ~50 tok/s | ~70 | - |
| engine-sized chunk | ~198 | ~116 | - |
| + tuned hipBLASLt | ~253-259 | ~195 | - |
| + `STRATA_GLM_PREFILL_SUB=1024`, `STRATA_GLM_PREFILL_MB=4096` | ~413 | ~217 | ~490-560 |

The 8060S runs used the same prompt-path changes on a local build. The R9700
runs used a pinned RAM tier sized for every expert the VRAM pool does not hold. On
gfx1201, plain hipBLAS is already close to hipBLASLt for most shapes, so its
table keeps only the rows that were at least 2% faster.

## What changed

- Added HIP GPU detection, compilation, rebuild stamps, and serving configs to
  `maya.py`, with separate CUDA and HIP build directories.
- Completed the CUDA-shaped runtime/BLAS mappings needed by GLM, excluded
  NVIDIA's profiling ranges from HIP, and linked the GLM prompt path to HIP MMQ
  and hipBLAS.
- Replaced the GLM hyper-connection kernel's unsupported FP32 `__ldcg` with
  agent-scope acquire loads on HIP. CUDA keeps its existing implementation.
- Kept CUDA WMMA attention guarded to CUDA builds. HIP uses the F32 attention
  kernel; since v1.0.7 its 32-row tile keeps the latent rows FP16 in shared
  memory (34 KB), within RDNA's 64 KiB workgroup limit.
- Fixed the KDA parity fixture to initialize its recurrence state explicitly;
  its numerical tolerance is unchanged.
- The HIP post-store system fences in both shared doorbell kernels and Maya's
  fused routing signal (backported from Strata) arrived separately in v1.0.7
  (#2); the validation below includes them.

Upstream Strata was inspected at `d5ea713` (0.1.40.3). Its AMD backend documents
runtime selection, memory limits, desktop VRAM headroom, and newer packed-byte
optimizations. Maya's GLM engine is separate from Strata's current Qwen engine,
so Strata's model benchmarks do not establish Maya performance.

## Validation

Test host: RX 7900 XT 20 GiB, Ryzen 7 9800X3D, 192 GB installed DDR5, native
Ubuntu Linux, system ROCm 7.2.1 / Clang 22, plus AI PRO R9700 32 GiB. Maya base v1.0.10 (`e4bd57e`).

The HIP engine and selected test targets build. All 14 selected GPU checks
pass: device allocation, expert upload staging, packed-byte/shuffle
intrinsics, asynchronous mapped-memory handoff, IQ1_S arithmetic, GLM
hyper-connections, KDA, FFN, DSA, layer arithmetic, synthetic-model logits
against the committed reference, and HIP MMQ. Six isolated installer tests
pass. The fused HC decode kernel also agrees with the independent CPU
reference across eight consecutive 4096-wide steps, including in-place gate
updates, in all three variants (default HC3, forced HC2, and forced HC1).
The full-model runs are below.

The real batched MLA attention kernel agrees with a double-precision CPU
softmax reference (absolute tolerance `5e-5`), including empty and masked cell
lists, partial and multiple tiles, and multiple head groups. It reads the
FP16 latent cache added in Maya v1.0.4.

The GLM handoff test replays the real routing graph for 100 disk requests
and 100 CPU-lane requests with changing IDs, weights, inputs and answers. The
CPU observes each request without a driver query or stream synchronization.
The same multi-architecture engine and test build pass all 14 checks on the
7900 XT and R9700. HC1 and HC2 also pass on the R9700. The HIP intrinsics check
covers all 65,536 pairs of byte values with varying neighboring lanes and
permutation selectors, against per-byte CPU references. The runtime rejects
the uncompiled `gfx1036` integrated GPU with the actual device and compiled
architecture list in its message. See [R9700 build details](AMD_RDNA4.md) and
[the shared AMD backport audit](AMD_UPSTREAM.md).

```sh
python3 tools/test_maya_hip.py
LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=0 \
  ctest --test-dir build-hip --output-on-failure --timeout 60 \
  -R '^(glm_(hc|kda|ffn|dsa|layer)_parity|glm_model_test|hip_intrinsics|hip_(glm_handoff|glm_prefill_attention|handoff|device_selftest|prefill_mmq_parity|expert_cache_staging)|iq1_s_parity)$'
STRATA_GLM_HC2=1 LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=0 \
  build-hip/glm_hc_parity --selftest
STRATA_GLM_HC1=1 LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=0 \
  build-hip/glm_hc_parity --selftest
```

The relevant build targets must be built before running CTest. Build and test
logs are kept in `build-hip/`.

Full model (Maya-S, 8K context) on the RX 7900 XT with the config setup writes:
seven smoke requests (arithmetic, code, a two-turn conversation, three
400-token answers) and five back-to-back prompt+answer rounds complete, with
prompts at 307-385 tok/s on 1.8-2.5K-token prompts and the hipBLASLt table
loaded. The R9700 runs the full model with every expert in VRAM or a pinned
RAM tier; the prompt speeds above were measured there.
