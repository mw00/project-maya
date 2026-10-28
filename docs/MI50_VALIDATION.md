# Experimental MI50 (gfx906) source-build validation

## Status

This change adds experimental HIP source-build support for AMD Instinct MI50 (`gfx906`). It does not change Maya's installer. Build commands below assume a fresh source checkout and an existing ROCm installation that supplies `gfx906` rocBLAS/Tensile libraries. The necessary rocBLAS/Tensile support is supplied separately; this change does not rebuild or redistribute it.

The patch is based on Project Maya `c9683e31192c35a8e93469d4134a290eefb57eca` (v1.1.23). Its pinned GGML source is `ggml-org/llama.cpp` commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`; all 1,376 tracked regular files in the GGML subtree matched the pinned tree exactly.

## Validated model configurations

| Model | Family / quant | Status on this patch |
|---|---|---|
| Maya-M | GLM-5.3-Flash, IQ2_S | Four-MI50 bounded checks passed |
| Maya-L | GLM-5.3-Flash, IQ3_S | Four-MI50 bounded checks passed |
| Maya-S | DeepSeek-V4.1-Flash | Work in progress; not validated here |

These are fresh checks of this patch, separate from earlier custom-build reports. Both GLM configurations allocated 204,800-token contexts; functional inputs were approximately 8K tokens, not full-capacity prompts.

## Test system and build environment

| Component | Recorded system |
|---|---|
| GPUs | 4 × AMD Instinct MI50, 32 GiB HBM each (`gfx906`; some tools print “16GB”) |
| CPUs | 2 × AMD EPYC 7452 |
| Memory | 467 GiB RAM |
| Storage | U.2 NVMe |
| GPU links | PCIe 4.0 x16 on all four MI50s; no xGMI |
| Process affinity | CPU 0-31,64-95; NUMA memory node 0 |
| ROCm | 7.2.0 |
| HIP compiler | AMD Clang 22.0.0git |
| Host compiler | GCC 13.3.0 |
| CMake | 3.28.3 |

Unmodified upstream at the recorded base rejects `gfx906` at configure time. The comparison control changes only the architecture classification and keeps portable signed dot4; the patch additionally enables native signed dot4 and adds regression tests.

The source build used HIP for `gfx906`, disabled CUDA, enabled tests, prefill MMQ, and native experts, and used the pinned GGML checkout. The build produced Maya and the applicable HIP parity targets. No `ds4_gguf_test` result is implied by this build report.

For a fresh checkout, link the supplied `build-hip` directory only if its target was built from that same checkout; do not reuse a build directory from another source tree. A generic configure/build recipe is:

```sh
cmake -S . -B build-gfx906 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/cc \
  -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DCMAKE_HIP_PLATFORM=amd \
  -DBUILD_SHARED_LIBS=OFF \
  -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_ENABLE_HIP=ON \
  -DSTRATA_PREFILL_MMQ=ON \
  -DSTRATA_GGML_DIR=/path/to/pinned/llama.cpp \
  -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_PLE_FIXTURE_DIR=/path/to/checkout/bench/micro \
  -DSTRATA_PORTABLE=OFF \
  -DSTRATA_WERROR=OFF
cmake --build build-gfx906 --target maya --parallel 8
```

To reconstruct the matched control from the frozen implementation/corpus snapshot, retain the architecture registration but restore the intrinsic implementation and tests to the upstream base:

```sh
git worktree add --detach ../maya-mi50-control 72e7d76d677da8c46ad46bb8241cda5b20b1c3d5
git -C ../maya-mi50-control restore --source=c9683e31192c35a8e93469d4134a290eefb57eca -- \
  include/strata/hip_compat/intrinsics.hpp tests/hip/intrinsics.cpp
```

Build the control separately with the same recipe and use identical model/configuration inputs. Do not share build directories between arms.

With serving stopped and exclusive access to the four GPUs, build and run the selected checks from the checkout root. Configure your compatible ROCm library paths first, including `ROCBLAS_TENSILE_LIBPATH` if required:

```sh
set -e
cmake --build build-gfx906 --parallel 8 --target \
  maya hip_intrinsics glm_mv_bench hip_prefill_native_batch \
  hip_glm_prefill_attention hip_glm_handoff glm_hc_parity glm_kda_parity \
  glm_ffn_parity glm_dsa_parity glm_layer_parity glm_model_test \
  iq1_s_parity hip_handoff hip_expert_cache_staging hip_prefill_mmq_parity strata-device
for gpu in 0 1 2 3; do
  export HIP_VISIBLE_DEVICES=$gpu
  for test in hip_intrinsics hip_prefill_native_batch hip_glm_prefill_attention \
              hip_glm_handoff iq1_s_parity hip_handoff hip_expert_cache_staging \
              hip_prefill_mmq_parity; do
    build-gfx906/$test
  done
  for test in glm_hc_parity glm_kda_parity glm_ffn_parity glm_dsa_parity \
              glm_layer_parity glm_model_test strata-device; do
    build-gfx906/$test --selftest
  done
  build-gfx906/glm_mv_bench --parity-only
done
```

This is a selected GLM/HIP suite, not a blanket CTest or all-model qualification. DeepSeek-V4.1 / DS4 is outside this validation.

## Test results

| Check | Result |
|---|---:|
| HIP compatibility mapping check | Pass |
| Server tests | 342 run, 11 skipped |
| Installer/tools tests | 384 run, 28 skipped |
| Pinned GGML manifest | 1,376 of 1,376 tracked files byte-identical |
| Extended applicable parity set, each of four MI50s | 16 checks passed per card; 64 total |
| Same applicable parity set on the Radeon AI PRO R9700 regression build | 16 checks passed |
| Service restoration after validation windows | Pass |

A fused-prefill check printed `SKIP` on the R9700 run because that target is compiled only for `gfx1100`/`gfx1151`; it is not included in either applicable-suite count. This check does not establish fused-prefill support on `gfx906`.

## Model packs and run settings

The existing native packs were prepared with upstream `iq_pack.py --compat-bf16`, using standard nearest-even BF16 rounding. Maya-M uses IQ2_S; Maya-L uses IQ3_S. The shipped `data/expert-profile-glm.bin` matches the upstream profile data by hash. The performance rows below are for GLM-5.3-Flash Maya models. DeepSeek-V4.1 Maya-S is a distinct model and has no result in this validation.

The recorded control and candidate configuration used the same meaningful engine settings:

```text
--glm-pack <Maya-M-or-Maya-L-pack>
--max-context 204800 --max-new 32768
--prefill 1024 --kv int8
--layer-split auto --split-device 1,2,3
--expert-profile <project>/data/expert-profile-glm.bin

MAYA_GLM_WARM_PROMPT=1
STRATA_GLM_COMPLETE_POOL=0
STRATA_GLM_PARK_SLOTS=1
STRATA_GLM_PREFILL_MB=2048
STRATA_GLM_PREFILL_SUB=1024
STRATA_GLM_RESERVE_MB=1024
STRATA_GLM_SLOT_KEEP=0
STRATA_GLM_VRAM_EVICT=least
STRATA_GLM_SLOT_DIR=<writable-per-run-directory>

HIP_VISIBLE_DEVICES=0,1,2,3
CUDA_VISIBLE_DEVICES=0,1,2,3
HSA_ENABLE_SDMA=0
ROCBLAS_TENSILE_LIBPATH=<compatible-gfx906-rocblas-library-directory>

service defaults: temperature 1.0, top_p 0.95, reasoning effort high
validation requests override: temperature 0, reasoning effort none, max_tokens 3072
```

Processes were pinned to CPU 0-31,64-95 and NUMA memory node 0. Libraries resolved through `/opt/rocm/lib` and `/opt/rocm-7.2.0/lib`. Path-specific values for model files, executable, slot directory, logs, and tokenizer are intentionally omitted. Those paths do not affect the documented settings and are machine-local.

Recorded SHA-256 identifiers (metadata hashes identify the existing packs; full expert shards were not rehashed):

```text
control gfx906 executable: 845afab37550eeff7d1ca23552564e7fa39d11ca31ee2462360a671fba3f22e1
patch gfx906 executable: 19ded2f457293c23fd753217fe0afdc9abc4d137b329a90edcf3149ff1c87532
Maya-M index.txt: 73706db6223e225025b034fae6c69dd0f3188c14c0f83bf4dce9d5fa2d2882e1
Maya-M native_experts.txt: a53732d7d32ad3cb8f7b3174dfe01034f81238b3f482a54d34721e483efcd6d1
Maya-M compat-bf16.json: 7b2780dea25fbd3c12f4f34249ff95a3a8631db6c6f58ed1785730759899d0ac
Maya-L index.txt: 5d5690818008a6ed3cb265c94fad5dacacedca785c4a1a1c4a08721e18c160a7
Maya-L native_experts.txt: 348a9cec64bf30691d48b6df5a767b4cda174182c166d8d059bd1f1c344d6024
Maya-L compat-bf16.json: cc1593f841dc434b572709cbb0a10e926ffa8d55738126c45e82981567447b9e
expert-profile-glm.bin: 53b03a13482b6fedc599d149be2d69f88c38d9edc4611a740da46314ab1dc5a7
```

## Upstream benchmark protocol and results

Use Maya's standard benchmark while the server is stopped, then capture a report:

```sh
ln -s build-gfx906 build-hip  # fresh checkout only; do not replace an existing build
python maya.py --backend hip --config /path/to/mi50-config.json --bench
python maya.py --backend hip --config /path/to/mi50-config.json --report
```

The standard comparison uses greedy decoding, thinking off, three answers capped at 256 tokens, and a short warm-up; prefill is measured at 2,048 and 8,192 prompt tokens. The table below uses matched control/candidate runs. The benchmark used the document corpus committed at `72e7d76d677da8c46ad46bb8241cda5b20b1c3d5`; this results page was filled in afterward. Check out that snapshot to reproduce the measured corpus. Its SHA-256 was `66b02705575a889091f4accf20b8650568a015557c0d6ec9cfc1618ba445d3e0`.

| Model | Arm/order | 2K prefill (tok/s) | 8K prefill (tok/s) | Mean decode (tok/s) | Run status |
|---|---|---:|---:|---:|---|
| Maya-M, IQ2_S | Control | 215 | 356 | 30.9 | Recorded single run |
| Maya-M, IQ2_S | Patch | 218 | 359 | 36.7 | Recorded single run |
| Maya-L, IQ3_S | Patch | 236 | 384 | 33.6 | Recorded single run |
| Maya-L, IQ3_S | Control | 235 | 386 | 29.7 | Recorded single run |

The balanced order was Maya-M control then patch, followed by Maya-L patch then control. These are single-pass comparisons: no confidence interval is available. Do not claim an uplift or production recommendation from one benchmark pass. These are comparisons against an architecture-enabled portable-dot4 control, not against the deployed custom engine, which already uses native dot4.

The 204,800-token context is a configured capacity. The separate coding-task requests used about 8,191 input tokens; they did not test a 196K prompt. No full-context soak has been completed, and there is no production-promotion claim.

## Coding-task queue

| Model | Captured output comparison | Quality result | Run status |
|---|---|---|---|
| Maya-M, IQ2_S | Byte-identical source output | Patch/control: 10/10 each | Completed |
| Maya-L, IQ3_S | Source output differs | Patch/control: 10/10 each | Completed |

The fixed task generated a self-contained Python HTTP body-limit helper; ten independent standard-library tests checked boundaries, malformed input and read behavior. Its local prompt/corpus is not reproduced here, so this is a bounded smoke check, not a public coding benchmark. The request bytes were identical between arms within each model. Exact-repeat requests and synthetic append-only tool-message continuations completed without protocol errors; no external tool was executed. The standard benchmark discards answer text, so it cannot establish token identity or output parity. Only the separate captured coding response is compared above; no benchmark answer-identity claim is made.

## Representative patched-engine logs

These are sanitized engine lines from the standard benchmarks; the table uses `maya-bench.txt`, not these individual samples.

### Maya-M representative patched-engine lines
```text
glm split: layers [0, 13) on CUDA0, [13, 24) on CUDA1, [24, 35) on CUDA2, [35, 45) on CUDA3 (one 64 KB host hop per boundary per token)
glm fast: CUDA0 layers [0,13) expert pool 23.75 GB, 291-291 slots/layer (2910 total)
glm fast: CUDA1 layers [13,24) expert pool 25.68 GB, 291-291 slots/layer (3201 total)
glm fast: CUDA2 layers [24,35) expert pool 25.68 GB, 291-291 slots/layer (3201 total)
glm fast: CUDA3 layers [35,45) expert pool 26.98 GB, 291-291 slots/layer, the draft layer 291 (3201 total)
glm stat: decode 26.60 ms/tok (37.59 tok/s) | vram hit 100.00% | ram fetches 0.00/tok | disk 0.00 reads/tok in 0.00 waits/tok, 0.00 ms/tok | promotions 0.00/tok | cpu lane 0.00/tok in 0.00 ms/tok | prompt 45.6 ms/tok | vram 12384/12513 ram 0/681
glm spec: CUDA1 the head's recurrent-state backup (34 MB) before the expert pool
```

### Maya-L representative patched-engine lines
```text
glm split: layers [0, 13) on CUDA0, [13, 24) on CUDA1, [24, 35) on CUDA2, [35, 45) on CUDA3 (one 64 KB host hop per boundary per token)
glm fast: CUDA0 layers [0,13) expert pool 28.13 GB, 206-275 slots/layer (2466 total)
glm fast: CUDA1 layers [13,24) expert pool 28.70 GB, 227-255 slots/layer (2641 total)
glm fast: CUDA2 layers [24,35) expert pool 28.89 GB, 222-257 slots/layer (2658 total)
glm fast: CUDA3 layers [35,45) expert pool 28.05 GB, 188-236 slots/layer, the draft layer 219 (2416 total)
glm stat: decode 29.12 ms/tok (34.34 tok/s) | vram hit 93.37% | ram fetches 0.87/tok | disk 0.00 reads/tok in 0.01 waits/tok, 0.00 ms/tok | promotions 0.88/tok | cpu lane 22.72/tok in 5.88 ms/tok | prompt 49.9 ms/tok | vram 10053/10181 ram 2330/2906
glm spec: CUDA1 the head's recurrent-state backup (34 MB) before the expert pool
```

## DeepSeek-V4.1 scope note

No DS4 or DeepSeek-V4.1 Maya-S run was performed for this gfx906 patch. A historical pristine v1.1.21 run on gfx1201 recorded an existing DS4 parity assertion failure; that is not evidence of a v1.1.23 or MI50 failure. This validation makes no new DS4 correctness or compatibility claim.

## Separate custom-kernel follow-up (2026-10-26)

This section is reference material for users of four **32 GiB MI50s**, not additional validation of this PR's source revision. The follow-up used the older custom source `ff872914d17240177ea035e2f20e3ca68bc7286a` (v1.1.19 lineage). Its `hip_fast_signs`, optional P2P implementation and local test adapters are **not included in this PR**. Both timing arms already have native signed dot4, so these measurements do not measure the dot4 change above or an upgrade over the deployed custom engine.

### GLM-5.3-Flash Maya-L (IQ3_S)

The candidate retains the custom HIP bit-parallel sign expansion. The portable control changes only the five HIP sign-expansion selections in `dw<16>`, `dw<18>`, `dw<17>`, `dw<22>` and `dw<21>`; identical test helpers are present in both arms. The pinned build image was `sha256:db3b7ed02a9b70473eb18185180dfeee0b873faa141fc787eb35bc77453277db`. The same model pack, 204,800-token allocation, runtime settings, server source, toolchain and byte-identical 20-file benchmark corpus were used. `STRATA_GLM_P2P` was unset and trace options were absent in both arms.

The exhaustive host sign oracle passed all 128 packed 7-bit codes, all 256 raw 8-bit patterns and valid IQ grid bytes 1–62. Dedicated IQ3_S (type 21) and IQ2_XS (type 17) expert fixtures passed on each of four MI50s for both arms. The older source's attention test initially called an unsupported rocWMMA path on gfx906. Restoring its architecture applicability guard, without changing production kernels or supported-path numerical assertions, allowed the selected five-executable GLM/HIP suite to pass on all four cards (20 card/executable cases). This guard is already present in the source used by this PR's validation; the old test failure is not a PR regression.

Each cell ran the actual upstream `maya.py --bench` followed by `--report`: three greedy, thinking-off decode questions capped at 256 tokens, with 2K/8K prefill. Three post-repair matched pairs were retained:

| Pair / run order | Portable control decode | Sign candidate decode | Control 2K / 8K prefill | Candidate 2K / 8K prefill |
|---|---:|---:|---:|---:|
| 02: control, candidate | 33.1 tok/s | 36.6 tok/s | 237 / 386 tok/s | 237 / 385 tok/s |
| 03: candidate, control | 30.8 tok/s | 34.6 tok/s | 228 / 375 tok/s | 237 / 378 tok/s |
| 04: control, candidate | 33.6 tok/s | 36.6 tok/s | 238 / 387 tok/s | 237 / 388 tok/s |

Observed paired decode differences were +3.0 to +3.8 tok/s (mean +3.43). Median 2K/8K prefill differences were 0 / +1 tok/s. These are **exploratory observations**, not a repeatable-gain claim: the predeclared requirement of six clean pairs and a confidence interval excluding zero was not met. No production promotion follows. An earlier pair was excluded after a source-bundle hardlink error made its candidate source-root bytes at measurement time unprovable; the source checkout was restored and the retained pairs used independently verified source files.

After each standard benchmark, a separate internal coding fixture used 8,192 input tokens. All six retained outputs passed the same ten source tests, but output text and lengths differed. Candidate-minus-control sustained decode differences were +2.0, +2.0 and +2.5 tok/s; candidate TTFT was approximately 0.04–0.06 seconds higher. This fixture is not the upstream benchmark, broad coding qualification, an actual tool-execution loop, or a full-context soak. Exact-repeat and continuation findings in the earlier PR validation remain separate evidence. Every retained cell passed its thermal and restoration checks.

Recorded identifiers for the custom comparison:

```text
candidate engine SHA-256: 30c17c5d4e7aeffddf07a374f0ac03ab4a65be1af8b9f68e5d17573a25e5451e
portable control engine SHA-256: 41dc37fc979de6b6dd464656086fae8543939440244d7b0ef1497a83a9f84772
candidate glm_fast.cu overlay SHA-256: fac440913f676644e3608f795e4965642f6e11c9e418b12e65a7504aa2e6b092
control glm_fast.cu overlay SHA-256: d7e83b947cefe897cd9b5c33db025f073aaea84e0ef17771d906616005b380ce
benchmark corpus manifest SHA-256: 70fc3af400dda4c270b1742153999480efe6c91ce8f8b724e0fd965557653be4
```

For a separate sign-expansion contribution, rebase the independent kernel change and its direct expert tests onto the intended upstream revision, repeat the same-source comparison there, and retain the raw benchmark/report outputs. These older-source numbers cannot substitute for that qualification.

### DeepSeek-V4.1 Maya-S (`ds4_gguf_test`) & v1.1.29 Rebase Validation (2026-10-26)

On the rebased `v1.1.29` (`8b4b39f`) source:
1. **`ds4_gguf_test` (`gfx906` MI50 and `gfx1201` R9700)**: **100% PASS (`ALL OK (0 failed)`)** across all KV pack, sparse attention, WMMA, router, `mv`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_XS`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `MXFP4`, and CPU-lane tests:
   - The earlier `IQ3_S/IQ2_S` mismatch observed on older `v1.1.19` custom sources was resolved by upstream `v1.1.29` `ds4_iq.cu` sub-block scaling.
   - The remaining 1-ULP `IQ4_XS/Q4_K` window-vs-token (`NE=4` vs `NE=1`) discrepancy on HIP (`gfx906` and `gfx1201`) was traced to ROCm's `<hip/amd_detail/amd_hip_runtime.h>` defining `__fadd_rn`, `__fsub_rn`, `__fmul_rn`, and `__fmaf_rn` as plain inline operators inside the system header scope, bypassing lexical `#pragma clang fp contract(off)` in `ds4_iq.cu` and allowing AMDGPU instruction selection to fuse operations across `sub_dot` and `down_kernel` differently when unrolled for `NE=1` vs `NE=4`. Wrapping the `sub_dot`, `gate_up_kernel`, `down_kernel`, and `lanes_sum` reductions in explicit `#pragma clang fp contract(off)` helpers with `asm volatile` register barriers (`fadd_no_fma`, `fsub_no_fma`, `fmul_no_fma`, `fma_no_contract`) restores 100% bit-for-bit identity between token and window paths on both `gfx906` and `gfx1201`.
2. **GLM-5.3 HIP Microbenchmarks on `v1.1.29` (`gfx906` MI50 & `gfx1201` R9700)**:
   - `hip_intrinsics`: **PASS** (100% parity on signed `sdot4`/overflow, 65,536 packed-byte pairs, wave32/64 shuffles, ballot, sleep).
   - `hip_expert_kernel_bench --parity-only`: **PASS** (`bit_identical=yes` across all 6 production quantization pairs including `gu=21/down=17` `IQ3_S/IQ2_XS` and `gu=22/down=22` with `hip_fast_signs` and `STRATA_GLM_GU22_TUNED=1`).
   - `glm_attn_bench`: **PASS** (100% bit-exact parity on all `dsa_score` and `mla_attend` shapes with hardware intra-lane butterfly reductions `ds_swizzle` + DPP `quad_perm` on `gfx906` and `v_permlanex16` + `row_xmask` on `gfx1201`, delivering a **4.56x** speedup on `gfx906` `dsa_score` at 130K context: `150.3 us` vs `685.7 us`, and **3.84x** on `gfx1201`: `65.0 us` vs `249.4 us`).
   - `glm_mv_bench --parity-only`: **PASS** (`differ=0 max_abs=0.000e+00` across all `router_shexp`, `dense_gu`, `dense_down`, and `head` variants).

## Limits

- A mixed wave32/wave64 fat binary has not been validated.
- NVIDIA hardware validation remains with Maya's maintainers.

No production binary or configuration was promoted without passing all qualification gates; displaced services were restored with healthy endpoints and no retained benchmark lease.
