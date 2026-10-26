# R9700 / RX 9070: experimental Maya HIP

The Linux HIP installer accepts RX 7900 XT/XTX (`gfx1100`) and RX 9070 /
Radeon AI PRO R9700 (`gfx1201`). It uses one GPU, text input, and system ROCm 7.
The shared backend changes come from Strata
[`d5ea713`](https://github.com/Niko1221/Strata/tree/d5ea7133741e67743c0e886bb426c0ce8d69cf6c).
The R9700 runs the full model (Maya-S, about 20 tok/s answers and 490-560 tok/s
prompts, every expert in VRAM or a pinned RAM tier).

```sh
./maya.sh --backend hip --gpu 1 --check
```

Use the KFD GPU number printed by setup; the example machine has a 7900 XT
at GPU 0 and an R9700 at GPU 1. Setup chooses the supported card with the most
VRAM when `--gpu` is omitted. See [AMD_MAYA.md](AMD_MAYA.md) for model setup.

## One binary, three code targets

Setup compiles the architectures of the chosen cards (a 7900 XT + R9700: `gfx1100;gfx1201`); a manual build can
name all three, `gfx1100;gfx1201;gfx1151`. CMake also accepts space-separated
lists and architecture feature suffixes. The runtime strips the feature suffix
and checks exact membership in the compiled list, then requires wave32.
Its diagnostics print the actual architecture and the compiled targets.

`gfx1151` **builds, untested**. Code is included for later Strix Halo work, but
setup does not enable that APU here. Unified-memory accounting and hardware
validation are separate requirements.

A manual build, using Maya's existing pinned llama.cpp checkout:

```sh
maya_rocm_root="$(realpath "${ROCM_PATH:-/opt/rocm}")"
cmake -S . -B build-multi -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMAYA_ENABLE_HIP=ON -DMAYA_ENABLE_CUDA=OFF \
  -DMAYA_NATIVE_EXPERTS=ON -DMAYA_PREFILL_MMQ=ON \
  -DMAYA_BUILD_TESTS=OFF \
  '-DCMAKE_HIP_ARCHITECTURES=gfx1100;gfx1201;gfx1151' \
  -DCMAKE_HIP_COMPILER="$maya_rocm_root/llvm/bin/clang++" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$maya_rocm_root" \
  -DCMAKE_PREFIX_PATH="$maya_rocm_root" \
  -DMAYA_GGML_DIR="$PWD/third_party/llama.cpp"
cmake --build build-multi --target maya strata-device hip_expert_cache_staging \
  hip_intrinsics hip_handoff hip_glm_handoff hip_glm_prefill_attention iq1_s_parity \
  glm_hc_parity glm_kda_parity glm_ffn_parity glm_dsa_parity glm_layer_parity \
  glm_model_test hip_prefill_mmq_parity -j 3
```

The handoff target and the tested publication fences were merged in v1.0.7 (#2).

## Validation

Native Ubuntu, Ryzen 7 9800X3D, 192 GB DDR5, ROCm 7.2.1 / Clang 22;
RX 7900 XT 20 GiB and AI PRO R9700 32 GiB. The same engine binary contains
64 bundled code objects for each of gfx1100, gfx1201 and gfx1151.

```sh
python3 tools/test_maya_hip.py
python3 tools/test_hip_archs.py
env -u ROCR_VISIBLE_DEVICES LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=1 \
  ctest --test-dir build-multi --output-on-failure --timeout 60 \
  -R '^(glm_(hc|kda|ffn|dsa|layer)_parity|glm_model_test|hip_intrinsics|hip_(glm_handoff|glm_prefill_attention|handoff|device_selftest|prefill_mmq_parity|expert_cache_staging)|iq1_s_parity)$'
env -u ROCR_VISIBLE_DEVICES LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=1 \
  MAYA_GLM_HC1=1 build-multi/glm_hc_parity --selftest
env -u ROCR_VISIBLE_DEVICES LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=1 \
  MAYA_GLM_HC2=1 build-multi/glm_hc_parity --selftest
```

Both cards pass 14/14 GPU checks; HC1/HC2 also pass on the R9700. The intrinsics
test compares signed dot4, overflow, shuffles, and 65,536 packed-byte pairs and
permutations to CPU references. Six setup tests and five architecture-parser
tests pass. Selecting the uncompiled gfx1036 iGPU is refused before a kernel
launch. CUDA was not compile-checked because the test host has no `nvcc`.

The copied gfx1201 hipBLASLt tables match versions `100202` and `100500`.
They serve the inherited tuning loader. Maya's GLM prompt projections can now
use the separate `gfx1201-glm-hipblaslt-100202.txt` table through that loader;
shapes without a tuned row continue through hipBLAS. Its quantized expert path
still uses GGML MMQ.

## v1.1.29 R9700 Validation (2026-10-26)

On upstream `v1.1.29` (`8b4b39f`) on AI PRO R9700 (`gfx1201`):

- **`ds4_gguf_test`**: **100% PASS (`ALL OK (0 failed)`)** across all KV pack, sparse attention, WMMA, router, `mv`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_XS`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `MXFP4`, and CPU-lane tests. The 1-ULP `NE=4` vs `NE=1` discrepancy was resolved by wrapping `sub_dot`, `gate_up_kernel`, `down_kernel`, and `lanes_sum` reductions in explicit `#pragma clang fp contract(off)` helpers with `asm volatile` register barriers (`fadd_no_fma`, `fsub_no_fma`, `fmul_no_fma`, `fma_no_contract`).
- **`glm_attn_bench`**: **PASS** (100% bit-exact parity; `dsa_score` is **3.84x faster** on `gfx1201` at 130K context: `65.0 us` vs `249.4 us` via `v_permlanex16` + `row_xmask` intra-lane butterfly reduction).
- **`hip_expert_kernel_bench --parity-only`**: **PASS** (`bit_identical=yes` across all production quantization configurations including `gu=21/down=17` `IQ3_S/IQ2_XS` and `gu=22/down=22`).
- **`hip_intrinsics`**: **PASS** (100% parity).

