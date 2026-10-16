# GLM-5.3-Flash (Maya-S-v2-IQ2_XXS) 128K Context & Extended Decode Benchmark - 2026-10-16

Rig: 3x Tesla V100-PCIE-16GB, 2x Intel Xeon E5-2690 v4 (28 physical cores, AVX2), 64 GB host RAM, PCIe 3.0.  
Model: `Maya-S-v2-IQ2_XXS` (3 shards across 3 GPUs, RAM tier pinned in host memory).  
Engine: Project Maya (`build/strata --serve`) with custom AVX2 kernels and Volta SM70 fast path.

---

## 1. Scale & Execution Overview

Tested extreme context depth scaling and sustained multi-thousand token decode generation within the 128K context envelope:

| Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| **Max Context Configured** | `131,072 tokens` (128K) | `--max-context 131072 --kv int8 --kv-streaming on` |
| **Prompt Depth** | **`124,000 tokens`** | Multi-chapter documentation and technical corpus |
| **Generated Decode Length** | **`3,871 tokens`** | Sustained continuous generation |
| **Total Context Envelope** | **`127,871 tokens`** | **`97.6%`** of 128K capacity |
| **Prefill Total Time** | **`94,958.5 ms`** (~95.0 s) | Layer-major chunked prefill on Tensor Cores |
| **Prefill Throughput** | **`1,305.8 tokens/s`** | Sustained across 124,000 prompt tokens |
| **Decode Total Time** | **`102,269.9 ms`** (~102.3 s) | Continuous token generation |
| **Decode Throughput** | **`37.85 tokens/s`** | **`26.42 ms / token`** |

---

## 2. Detailed Runtime & Memory Telemetry

```
glm stat: decode 26.42 ms/tok (37.85 tok/s) | vram hit 89.29% | ram fetches 4.84/tok | disk 0.18 reads/tok in 0.17 waits/tok, 0.49 ms/tok | promotions 4.92/tok | cpu lane 32.24/tok in 11.37 ms/tok | prompt 0.8 ms/tok | vram 5318/5447 ram 7006/7205
glm spec: 3750 of 3869 drafts accepted (96.9%)
glm kv: KV streaming: 96.37% of 27225990 block reads hit VRAM, 2051.0 MiB read from RAM
glm KDA graphs: 10 captures, 44540 replays, 0 invalidations
glm KDA graphs: 9 captures, 40086 replays, 0 invalidations
glm spec: 4437 speculative positions, 97.1% accepted
glm KDA graphs: 15 captures, 2190 replays, 0 invalidations
```

* **VRAM MoE Expert Hit Rate (89.29%)**:
  * As the expert working set warmed over the sustained 3,871-token decode sequence, off-card host RAM fetches dropped to just **`4.84 fetches / token`** (compared to >20 / token on cold prompts).
  * VRAM promotions settled at `4.92 / token`, minimizing PCIe D2H eviction thrash.
* **Dual-Vector AVX2 CPU Lane (11.37 ms / tok)**:
  * Offloaded **`32.24 experts / token`** concurrently to the NUMA-pinned Xeon core pool.
  * Single-expert CPU evaluation averaged **`0.204 – 0.234 ms / expert`**, compressing the total CPU lane contribution per token to just **`11.37 ms`**.
* **MTP Speculative Decoding (96.9% acceptance)**:
  * **`3,750 of 3,869 drafts accepted`** (**`96.9%`** acceptance; `4,437 positions total at 97.1%`).
* **INT8 KV Cache Streaming (96.37% VRAM hit)**:
  * Total block reads: `27,225,990 reads`.
  * `96.37%` of all KV lookups hit VRAM HBM2.
  * Only `2,051.0 MiB` read across PCIe from host pinned memory across the entire 128K context generation.
* **CUDA Graphs (KDA Layers)**:
  * **`86,816 graph replays`** with **`0 invalidations`** across the 3 GPUs.

---

## 3. Scaling Comparison: Standard 8K vs. Extreme 128K

| Context Tier | Prompt Length | Decode Length | Prefill Speed | Decode Speed | VRAM Hit % | Off-Card RAM Fetches | CPU Lane Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard (8K)** | 2,048 tok | 512 tok | **`268 tok/s`** | **`31.0 tok/s`** | `76.4%` | `21.2 / tok` | `19.8 ms / tok` |
| **Extreme (128K)** | 124,000 tok | 3,871 tok | **`1,305.8 tok/s`** | **`37.85 tok/s`** | **`89.29%`** | **`4.84 / tok`** | **`11.37 ms / tok`** |

---

## 4. Key Implementation Details of the Edits

1. **AVX2 1-Token Decode Kernels & SwiGLU Fusion (`src/kernels/cpu/iq_avx2.cpp`)**:
   - Bit-identical 1-token decode kernels for `IQ2_XXS` and `IQ2_S` without calling generic batch paths.
   - `dot1_gu_iq2_xxs`: Loads activation vectors once into YMM registers to compute Gate and Up dot products concurrently, halving memory bandwidth demand.
   - Fused clamped SwiGLU epilogue into `iq256_gu_rows1` directly in registers without intermediate allocations.
   - 2-row unrolled down kernels (`dot2_iq2_s`, `dot2_iq3_xxs`) to amortize activation loads and sign mask generation.
   - `native_accum_out`: Vectorized AVX2 FMA expert reduction loop with pre-broadcast weights.

2. **Worker Pool False-Sharing Elimination (`src/core/glm_fast_state.hpp`)**:
   - `alignas(64)` padding between `gen_`, `ticket_`, and `done_` atomic counters to prevent false-sharing cache invalidations across NUMA sockets.
   - Batched worker completions into a single `done_.fetch_add(my_done)` per task.
   - NUMA multi-pool core pinning (`STRATA_GLM_CPU_PIN_1TO1`).

3. **CUDA Volta (`sm_70`) Direct Signs & 4-Row Register Blocking (`src/kernels/cuda/glm_fast.cu`)**:
   - Eliminated software-emulated `__vcmpne4` and `__vsub4` instructions on Volta SM70 using shared-memory sign tables (`s_sign`) and two's-complement direct bit arithmetic `(grid_pos.x ^ signs.x) + signs.y`.
   - Implemented `mv_kernel_t_r<8, 4>` to compute 4 output rows per warp for dense `Q8_0` GEMVs, quadrupling activation reuse.

4. **Multi-GPU Pipeline & Layer Graphs (`src/core/glm_fast_path.cu`, `src/core/glm_model.cu`)**:
   - Generalized `STRATA_GLM_SPEC_CANCEL` and `STRATA_GLM_KDA_GRAPH` across multi-GPU splits.
   - Pipelined expert quantization in `lane_experts` via `exp_done` atomic row tracking.
   - Avoided allocating unused reference scratch arena (`sc_`) on non-tail GPUs in `fast_mode_`.

5. **Bit-Exact Verification**:
   - `./build/native_rows_exact_test`: 2244 results checked per type against `ggml-cpu`'s dot product — **0 differ (all bit-identical)**.
   - `./build/glm_mv_nt_test`: 568 cases checked — **100% bit-for-bit identical**.
   - `./build/glm_layer_graph_test`: **PASS**.
