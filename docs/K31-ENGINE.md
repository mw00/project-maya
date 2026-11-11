# maya-k31: engine notes

Working notes for the K3.1 engine (`src/k31/`, `STRATA_BUILD_K31`). Each Phase 3 milestone adds the section it owns.
Paths are relative to the repository root, and line numbers are at `k31/phase3` (`kimi-k31` rebased on `v1.1.55`).

## 1. The k ≤ 8 inventory

K3.1 routes 16 of 896 experts per token, through a 3584-wide latent, on a 7168-wide model. Most of Maya's MoE code
was written for at most 8 picks, at most 512 experts and at most 4096 or 8192 wide. This section lists every such cap
in the code `maya-k31` compiles or calls. For each one it says whether K3.1 can reach it and what the engine does
about it. The rule from the plan stands: **the shared sites stay as they are**, and K3.1 either does not reach them,
reaches them through paired rows or a direct launch, or refuses the geometry first.

### 1.1 Scope and method

- **What `maya-k31` compiles.** It copies `maya-mm3`'s block (`CMakeLists.txt:821-833`): `src/ds4/ds4_iq.cu`,
  `ds4_lt.cu`, `ds4_iq_cpu.cpp` and `ds4_gguf_cpu.cpp`, built under `MAYA_DS4_IQ_ONLY`. It also has the
  `src/ds4/ds4_gguf.cu` build probe (M0). On the host it uses the tiering headers `ds4_place.hpp`, `ds4_store.hpp`,
  `ds4_pcie.hpp`, `ds4_ram_slice.hpp`, `ds4_pool.hpp` and `ds4_lane_parity.hpp`, and it calls kernels from
  `strata_kernels`: `glm_kda_*`, `glm_dsa_score` and `glm_dsa_select`.
- **The include closure** of those files was followed transitively, with include guards ignored so that it
  over-includes.
- **Two sweeps.**
  - The first pass read each file group by hand.
  - A second reader then grepped every file in the closure, plus the `strata_kernels` objects K3.1 calls, for
    `kPicks`, `[8]`, `* 8`, `<< 3`, `>> 3`, `& 7`, `<= 8` / `> 8` / `== 8`, `16 * 8`, `512`, `1024`, `4096`, `8192`,
    `32768` and `kMax*`. It filtered out byte sizes, SIMD widths and QK block sizes by reading each hit.
  - It found 6 sites the first pass had missed and corrected 15 rows. Both are folded in below.
- **Columns.**
  - "K3.1 reaches": *yes* means the planned call path hits the site; *only-if-misused* means it does only if code
    calls an API the plan avoids; *no* means it cannot.
  - "Plan": *not reached*, *direct launch* (T = 1, one grid row per expert, `groups = 16`), *paired rows* (a window
    token's latent copied into two 8-pick input rows), *refuse at geometry* (`k31_geo`), *new k31 code*, or
    *generalize* (a shared-file edit, allowed only additively and only when §1 of the plan allows it).

### 1.2 What this changes in the plan

1. **`kPicks` is not compiled into `maya-k31`.**
   - Its 126 uses sit in `src/ds4/ds4_kernels.cuh` (12), `ds4_kernels.cu` (19) and `ds4_model.cu` (95). Only
     `ds4_model.cu:26`, `ds4_kernels.cu:3`, `ds4_peer_calibrate.cu:2` and some tests include `ds4_kernels.cuh`; no
     file in the closure above does.
   - The M0 deliverable "the k ≤ 8 inventory of every site in the compiled-in DeepSeek sources: `kPicks` (126
     uses), …" therefore resolves to §1.8: all of them are *not compiled in*. (`tests/ds4_cpu_flow_test.cpp:51` has
     an unrelated local `kPicks = 6`.)
   - Keep it that way: no `src/k31` file may include `ds4_kernels.cuh` or `ds4_model.hpp`. A grep in the K3.1 CMake
     block or in `selftest_baseline.sh` can enforce this.
2. **The one compiled-in pick stride is a literal**, so a `kPicks` grep misses it. It is `src/ds4/ds4_iq.cu:856`,
   `xr = grp_ptr ? e >> 3 : 0`, documented at `ds4_iq.cuh:49-52`: in a window, entry `e` reads int8 input row `e / 8`.
   - Decode (T = 1, `grp_ptr` null) is k-agnostic: `iq::gate_up` / `iq::down` with `groups = 16`, plan row `g` = pick
     `g`, input row 0.
   - Windows (T > 1) use **paired rows**. Each token's latent is quantized once and copied to int8 rows `2t` and
     `2t + 1`, and the entries are numbered `(2t + h)·8 + i`. The h, hq and part rows then sit at `16t … 16t + 15`, so
     `k31_moe_combine` reads a plain 16-stride. This confirms the plan's "two 8-pick rows per token over a
     duplicated latent row".
3. **The fused IQ gate/up kernels compute only SwiGLU or swigluoai, never SiTU.**
   - `act_out<EPI>` (`ds4_iq.cu:708-712`) is fused into `iq::gate_up`, `set_gate_up`, the light kernels and
     `swiglu_f16`.
   - The dispatches at `:1754` (`gate_up`), `:1655` / `:1661` (`light_set`) and `:2197` (`set_gate_up`) send
     *every* `act != kActSwiglu` to the swigluoai EPIs.
   - So the plan's M2 "direct `iq::gate_up` / `iq::down` launches" and its prompt path "`set_gate_up` /
     `set_down` (k-agnostic)" would apply the wrong activation to K3.1 unless `kActSitu` exists. The plan adds it
     only at M3 ("SiTU fused (`kActSitu` = 2) only by microbenchmark").
   - The fix is one of:
     - (a) Through M2, gate and up run as raw GEMVs (`iq::mv`, or the `tu != tg` raw split) followed by
       `k31_situ`, and prompts use `f16_rows` + cuBLAS + `k31_situ`. Only `set_down` (activation-free) is reused.
     - (b) Bring `kActSitu` forward. That needs a SiTU EPI, plus all three dispatches changed to test
       `act == kActOai` explicitly, plus a mutation test that `kActSitu` does not compute swigluoai. It is an
       additive shared-file edit and belongs in the §1 list of shared edits.
   - Related sites:
     - Under SwiGLU, the gate's type is used for the up rows too (`ds4_iq.cu:1769`, `tu` is ignored), so `k31_fmt`
       must refuse `tg != tu` for any fused launch.
     - The lane-parity `swiglu_portable` (`ds4_par_exp.h:77`) and the CPU lane's fused gate/up
       (`native_expert.cpp:306`, `:324`, `:273`; `iq_cpu_gate_up`) are SwiGLU-only too.
     - So the K3.1 CPU lane runs `iq_cpu_rows` for gate and for up, then a k31 SiTU (with a portable variant for
       `MAYA_DS4_LANE_PARITY`), then `iq_cpu_rows` / `iq_cpu_down`.
4. **The prompt light sets stop at 512 experts and fall back silently** (`kLightMaxExp = 512`, `ds4_iq.cu:1239`;
   refusals at `:1785`, `:1789`).
   - With E = 896, `set_gate_up` / `set_down` still return true, but every expert takes the 64-row tiles. That is
     correct but slower for prompts that give most experts at most 16 rows.
   - Split each set into chunks of at most 512 experts (`wbase + i0·stride`, `bounds + i0`), or raise the constant
     to 1024 (128 MiB + 16 MiB of scratch a device stream; a shared-file edit).
   - The constant sits in an anonymous namespace, so `k31` mirrors it as `kIqLightMaxExp` with a `static_assert`.
5. **Every MoE wrapper in the `ds4_gguf.cu` probe is k ≤ 8.**
   - The sites: `mx_geometry` (`:1369-1375`), `Scratch part [8]` (`:1340-1344`), `mx_combine_kernel`'s `float p[8]`
     and `plan_w + t*8` (`:942-952`), `CombArgs [T][8]`, `win_experts`' `a.T * 8` h rows (`:1988`, `:2010`), and the
     window entry-to-token map `e >> 3` (`:757-759`, `:815`).
   - K3.1 never calls `moe_gate_up`, `moe_down`, `win_experts` or `MoeArgs` / `WinArgs`. MXFP4 and BF16 experts go
     through `gguf::mv` one matrix at a time (8 jobs, 16 columns a call), or through the host dequant + cuBLAS
     fallback.
   - Separately, `kMaxQK = 32768` caps Q8_0 and low-bit dense GEMV inputs (`:1339`, `:1764`, `:1819`). Layer 0's dense
     down has k = 33792, so it goes through `iq::mv`, which has no cap, or stays BF16, or is split into two k ≤ 16896
     jobs.
6. **CPU-lane byte caps.** `kNativeActBytes = 8192` (`native_expert.hpp:19`) allows a gate/up input of at most 7168
   at Q8_K (28 × 292 B = 8176). `kNativeHBytes = 4096` (`:22`) allows a down input of at most 3584 at Q8_K.
   - K3.1's routed experts (3584 / 3072) fit.
   - A 7168-wide input fits with 16 B to spare.
   - The shared experts' 6144-wide down input and the dense 33792-wide down do not, so `k31_geo` refuses them on
     the CPU lane, which keeps them on the GPU.
   - The token-count caps in this tier count tokens per call, not picks: `kMxCpuMaxN = 8`, lane parity's `kMaxT = 8`
     with unchecked stack arrays (`ds4_lane_parity.cpp:40`), and `kExl3CpuMaxN = 8`. K3.1 calls them with at most
     8 tokens and asserts that it does.
7. **`ds4_pcie.hpp` is the one host cost model with a pick cap.**
   - `split()` enumerates 2^n masks (`:48-67`; at k = 16, 65,535 masks × 16 ≈ 1M operations a call).
   - `streams_any()` clamps `n <= 8` (`:87`), `table()` defaults to `n_max = 8` (`:71`), and the masks are 32-bit
     (`1u << n` is undefined at n = 32, so at most 31 picks).
   - The cost depends only on how many cold and warm picks stream, so a count-based split is O((n + 1)²) and gives
     the same answers for n ≤ 8 with the lowest-index tie rule.
   - K3.1 uses that, in a k31 copy or as an additive overload, with `static_assert(kPicks31 <= 31)`.
8. **The GLM DSA kernels have no top-k cap, but an unchecked contract.**
   - `glm_dsa_select` (`glm_dsa.cu:98`) writes `top_pools · kpool` cells a row with no bound against `n_sel`, and
     `glm_dsa_score` indexes `n_tokens · n_pools` as an int (`:186-204`).
   - The K3.1 runner asserts `kpool = 1`, `top_pools = 2048`, no tail, `n_sel == 2048`, a score row
     `n_pools >= max pos + 1` and `T · n_keys < 2^31` (the prompt chunk rule) before every launch.
   - The GLM fast-path selection `dsa_topk.cuh` stops at 1024 (`:23-32`). M1 does not use it. The M3 fast path
     takes DeepSeek's segmented radix `topk` (any k) or a generalized copy.
9. **`iq::mv` refuses late** (`ds4_iq.cu:2242-2251`). It checks `n <= 64` up front, but the per-job `k % block` and
   `1 <= ncol <= 16` checks run per type after earlier types have launched. A refused job returns false with earlier
   outputs already written, so k31 validates every job first (or issues one type a call) and treats false as fatal.
   K3.1's decode needs 36 gate/up/shared jobs and 18 down jobs, well under 64.

### 1.3 The "experts truncated to 8" mutations (required from M1)

Each mutation below must make `k31_kernels_parity --selftest` (or, for the runner, the layer-composition test)
fail. The router-level ones are already in `tools/test_kimi_k31_golden.py`.

| Mutation | Where it bites | Caught by |
|---|---|---|
| `k31_route` keeps 8 picks (or writes at stride 8) | ids/weights against the golden at E = 896, k = 16 | `k31_kernels_parity` route vs `tools/kimi_k31_golden.py` (`check_route`: a set or length mismatch); already a negative test in `test_kimi_k31_golden.py` |
| Picks 8-15 get weight 0 (the combine loops to 8) | `k31_moe_combine` | combine vs ref dumps `ffn_moe_out` (the routed sum before the norm) |
| A window without the duplicated latent row (entries `t·16 + i` over T rows, so `e >> 3` reads the wrong token) | the paired-row builder plus `iq::gate_up`/`down` | window (T = 2, 4) vs token-by-token decode, bit for bit, on the big fixture |
| The CPU-lane split drops ranks 8-15 (an `n <= 8` clamp, as in `ds4_pcie.hpp:87`) | the K3.1 lane split | lane split vs brute force for n ≤ 12, then a layer composition with the CPU lane forced on |
| A fused gate/up with `act = kActSitu` running swigluoai (if option 3b is taken) | `ds4_iq.cu` dispatch | `k31_situ` golden (`check_situ` budget 10 ULP) through the fused path |

### 1.4 Compiled in and reached by K3.1 (27 rows)

| Site | What | Cap | K3.1 reaches | Plan |
|---|---|---|---|---|
| `src/ds4/ds4_iq.cu:856` | `xr[threadIdx.x] = a.grp_ptr != nullptr ? e >> 3 : 0;` in gate_up_kernel: a window entry e reads int8 input row e >> 3 (8 entries a token, DeepSeek's kPicks stride written as a literal) | 8-stride: 8 picks per input row | yes | paired rows |
| `src/ds4/ds4_iq.cuh:49-52` | iq::Args contract doc: grp_ptr null means group g = expert g of the plan, plan_ptr[g], entry g, input row 0; otherwise 'entry e's input row e >> 3' | 8-stride (window contract) | yes | direct launch (token) / paired rows (window) |
| `src/ds4/ds4_iq.cu:715-729` | group(): when grp_ptr == nullptr, blob = plan_ptr[grp] (0 = skip), entries m0 = grp, m1 = grp + 1; grid.y = groups (:1753, :1777) | none (groups ≤ 65535 grid.y) | yes | direct launch |
| `src/ds4/ds4_iq.cu:1567-1579` | gu_ne/dn_ne template ladder: ne ≤ 1 / ≤ 4 / ≤ 8 picks NE 1/4/8, else NE 16; `__shared__ int es[NE], xr[NE]` (:834, :923); loop `mm += NE` (:850, :939) | none (an expert's entries in passes of NE ≤ 16) | yes | direct launch (ne_max 1) / paired rows (ne_max = T) |
| `src/ds4/ds4_iq.cu:708-712` | act_out<EPI>: EPI 0 = DeepSeek clamped SwiGLU (pexp: swiglu_portable), EPI 1 = swigluoai; fused at :899 (token/window), :1068 (set_kernel), :1175 (tensor-core sets), :1329 (light). Args::act/SetArgs::act take only kActSwiglu = 0 … | activation ∈ {SwiGLU, swigluoai}; no raw or SiTU output | yes | a SiTU EPI is not enough on its own: gate_up (:1754), light_set (:1655/:1661) and set_gate_up (:2197) send every act != kActSwiglu to the swigluoai EPIs, so all three dispatches must test act == kActOai explicitly (plus a mutation test kActSitu != swigluoai);… |
| `src/ds4/ds4_iq.cu:1239` | `constexpr int kLightMaxExp = 512;` with light_gate_up/light_down `if (a.n_exp > kLightMaxExp ...) return false;` (:1785, :1789); scratch need = 512·16·max(k, 8192) (:1612) | E ≤ 512 per prompt light set | yes | split sets (≤ 512 experts a set_* call: wbase + i0·stride, bounds + i0) or generalize kLightMaxExp to 1024 (scratch 128 MiB + 16 MiB a device-stream) |
| `src/ds4/ds4_iq.cu:1238` | kLightRows = 16 (rows an expert in the light form): light_q8 `rows > kLightRows` (:1252), launch_set `max_rows <= 16` picks RM 16 (:1591-1593), launch_light 4/8/16 (:1643-1645), set_* `max_rows <= kLightRows` early return (:2193,… | none (light vs tile threshold on rows, not picks) | yes | not a cap (k-agnostic) |
| `src/ds4/ds4_iq.cu:2196` | set grids: z = n_exp, x = ceil(max_rows/64), y = n_ff/64 (gate_up :2196) or n_embd/64 (down :2233); guards n_ff % BN (:2189), n_embd % BN (:2227), BN = 64 (:979) | widths multiple of 64; E ≤ 65535 (grid z) | yes | refuse at geometry |
| `src/ds4/ds4_iq.cu:1752` | gate_up(): k_whole(n_embd) (n_embd % 256, IQ4_NL 32) and n_ff % (kWarps·kGuRW = 8); down() (:1776): n_ff % block and n_embd % (kWarps·kDnRW = 16) | divisibility only; no max width | yes | refuse at geometry |
| `src/ds4/ds4_iq.cu:2242-2243` | mv(): `bool done[64] = {}; if (n > 64) return false;`; kMaxJobs = 8 jobs a launch, looped (:1194-1196, :2249); `ncol < 1 \|\| ncol > 16` refused (:2251); NC ladder 1/4/16 (:1697-1702) | ≤ 64 jobs a call, ≤ 16 columns a job | yes | direct launch (one mv call a role a token: 16 gate + 16 up = 32, + 2 shared x 2 = 36 ≤ 64; down 16 + 2 = 18) |
| `src/ds4/ds4_iq.cu:1209` | mv_kernel: `nsb = b.k / 32`, lanes `kb += 32`, grid.x = ceil(max_n / 8) (:1698), size_t row/col offsets | none (k % block_values only) | yes | direct launch (no change) |
| `src/ds4/ds4_iq.cu:760` | `kGuRW = 1, kDnRW = 2` (rows a warp, commented for n_embd 5120 / 2304); kWarps = 8, kThreads = 256 (:61); lanes_sum `RW <= 8` (:740, :746); the IQ1 RP=4 variant is used only if grid.x % 4 == 0 (K3.1's 384 / 224 qualify), otherwis… | none (block shape; 8 = warps a block) | yes | not a cap (microbenchmark in tests/k31_iq_gpu_bench.cu, M0) |
| `src/ds4/ds4_iq.cu:63-87` | ggml block format internals: Ty<T>::TAB 256/512/1024/2048 (codebook words), kQK 256/32; Sub::q[8] (:181); `kb >> 3`, `kb & 7` (:193-194, :243-244, :506-507); ds4_iq1_codes.hpp:30-49 (4096 codes, g << 3) | none (8 sub-blocks of 32 in a 256-value block) | yes | not a cap |
| `src/ds4/ds4_iq.cu:2315` | grid_for ≤ 4096 blocks (f16_rows, gather_f16, swiglu_f16) and decode_values ≤ 1024 blocks (:2335), both grid-stride | none (grid-stride loops) | yes | not a cap |
| `src/ds4/ds4_iq.cu:1387` | tc_ok(): `static std::atomic<int> known[64]` (per device) | ≤ 64 devices (not a model cap) | yes | not a cap |
| `include/strata/kernels/cpu/native_expert.hpp:19` | `inline constexpr size_t kNativeActBytes = 8192;`, enforced at src/kernels/cpu/native_expert.cpp:230 inside native_fmt, which iq_cpu_fmt calls (ds4_iq_cpu.cpp:1073) | gate/up activation <= 8192 B, i.e. width <= 7168 at Q8_K (28 x 292 B) | yes | refuse at geometry |
| `include/strata/kernels/cpu/native_expert.hpp:22` | `inline constexpr size_t kNativeHBytes = 4096;`, enforced at native_expert.cpp:230 | down activation <= 4096 B, i.e. n_ff <= 3584 at Q8_K (14 x 292), <= 3840 at Q8_0 | yes | refuse at geometry |
| `src/ds4/ds4_iq_cpu.cpp:795` | iq1s_rows_n, also iq_rows_n :765 and x512_rows :711: `const block_q8_K* y[4]; float* o[4];`, t0 += 4, switch cases 1-3 / default 4 | 4 tokens a pass, any nt | yes | not reached |
| `src/ds4/ds4_iq_cpu.cpp:433` | iq1s_rows_avx2 (:424): `constexpr int S = 16; // blocks a segment`, sumi1 8 blocks a step `for (k0 = 0; k0 < ns; k0 += 8)` with `ka/kb = std::min(..., nb - 1)` (:461-465) | 16-block segment, 8-block sub-step; tail clamped | yes | not reached |
| `src/ds4/ds4_iq_cpu.cpp:850` | own_check `const int n = 2048, rows = 13;` / time_variant :891 `const int n = 4096, rows = 32;` | synthetic check and timing widths (not limits on the model) | yes | not reached |
| `src/ds4/ds4_cpu_prepare_rows.hpp:9` | detail::prepare_cpu_rows(type, fallback, own, one, multi) | none | yes | not reached |
| `src/kernels/cuda/glm_dsa.cu:186-204, 61-66, 80-106` | glm_dsa_score: no head/dim cap, but const int total = n_tokens * n_pools (:190, kernel :64); glm_dsa_select: no top_pools cap, rank loop O(n_vis^2) a thread, ties to the lower index, n_sel = kpool*top_pools + tail | int index: T*n_pools < 2^31; no top-k cap | yes | direct launch (kpool 1, top_pools 2048, no tail); prompt chunk rule T_chunk*n_keys < 2^31 |
| `src/kernels/cuda/glm_kda.cu:33, 166, 191, 170, 183; include/strata/kernels/glm_kda.hpp:45-75` | KD_MAX_D_CONV 16 (d_conv <= 17, exits); glm_kda_recurrence head_dim <= KD_THREADS 256 (exits); grid = n_head (no head cap); int totals d_inner*T and head_dim*n_head*T | head_dim <= 256, d_conv <= 17; no n_head cap | yes | direct launch |
| `src/ds4/ds4_iq.cu:1754` | gate_up(): `if (a.act != kActSwiglu) { ... gu_ne<T, 1> ... }`; the same catch-all test in light_set (:1655 guard, :1661) and set_gate_up (:2197); split_gu() requires `a.act == kActOai` (:1648). Any act that is not 0 runs the swig… | activation in {0 = clamped SwiGLU, anything else = swigluoai}; a new act value has no refusal | yes | generalize (a SiTU EPI, with all three dispatches testing act == kActOai explicitly and routing kActSitu to it, plus act_out, the light path and swiglu_f16) or new k31 code (iq::mv raw gate/up + k31_situ, the plan's M1) |
| `src/kernels/cpu/iq_avx512.cpp:357` | gu_rows_nt / dot_rows_nt: `default: for (t0 = 0; t0 < nt; t0 += 8)` (also :372 in passes of 4, iq_avx2.cpp:566 / :581 / :892, and kq_avx512.cpp:141 `TB = 8`). These are the native multi-token rows behind native_*_rows_split, reac… | none (8 or 4 tokens a pass, any nt) | yes | not a cap |
| `src/kernels/cuda/glm_dsa.cu:98` | glm_dsa_select_kernel: `if (rank < top_pools) for (m < kpool) row[rank * kpool + m] = p * kpool + m;` into row = cells + n_sel * t, with no check against n_sel. It reads score[p + n_pools * t] for p < n_vis = (pos[t] + 1) / kpool… | contract only: n_sel >= kpool*top_pools (+ kpool-1 with tail) and pos[t]+1 <= n_pools*kpool; an overflow writes past cells silently | yes | direct launch (kpool 1, top 2048, no tail, n_sel = 2048 exactly, score rows n_pools >= max pos + 1); the k31 runner asserts both |
| `src/kernels/cpu/native_expert.cpp:43` | rows_match_ggml: `const int n = 2048; // eight 256-value blocks`, nts {2, 5, 11}, 17 rows. This is the first-use parity check that decides whether the native multi-token rows are used (native_multi_rows_exact). | self-check width nb = 8 only (not a limit on the model) | yes | not a cap (M0 bench: bit-check nb = 12 and 14 against ggml's dot, as for own_check) |

### 1.5 Compiled in, reached only if misused or not at all (15 rows)

| Site | What | Cap | K3.1 reaches | Plan |
|---|---|---|---|---|
| `src/ds4/ds4_par_exp.h:77` | swiglu_portable / exp_portable (DS4_LANE_PARITY), reached via Args::pexp (ds4_iq.cuh:70, ds4_iq.cu:710); exp range y in [-87, 89) (:54-56) | parity path SwiGLU-only | only-if-misused | new k31 code (portable SiTU: tanh/sigmoid built on exp_portable for MAYA_DS4_LANE_PARITY) |
| `src/ds4/ds4_lt.cu:28-224` | hipBLASLt dense GEMM: Key{n,k,ldx,ldy,tb}, kCandidates 16, kWorkspace 64 MiB, g_dev[64] (:53, :113), kMinRows 64 (ds4_lt.hpp:16); CUDA build is a stub returning false (:225-229) | none (any n/k; ≥ 64 rows; ≤ 64 devices) | no | not a cap |
| `src/ds4/ds4_gguf_cpu.hpp:13` | `constexpr int kMxCpuMaxN = 8; // activations a call` (contract of mx_cpu_rows at :15-17) | tokens <= 8 per MXFP4 CPU-row call (not picks) | only-if-misused | not reached |
| `src/ds4/ds4_gguf_cpu.cpp:67` | rows_scalar / rows_avx2 (:101): `float acc[kMxCpuMaxN]` / `__m256 acc[kMxCpuMaxN]`, loops `i < n` without a check (mx_cpu_rows :339-345) | n <= 8 activations, else stack overflow | only-if-misused | not reached |
| `src/ds4/ds4_gguf_cpu.cpp:407` | mx_rows_tier scalar fallback: `for (i0 = 0; i0 < n; i0 += kMxCpuMaxN) rows_q_scalar(..., std::min(kMxCpuMaxN, n - i0), ...)`, with rows_q_scalar `acc[kMxCpuMaxN][8]` at :189 | 8 tokens a chunk, any n; [8] = MXFP4's 8-lane sum order | no | not reached |
| `src/ds4/ds4_mx_kern.inc:224-238` | rows(): `for (i0 = 0; i0 < n; i0 += 8)` dispatching rows_n<1..8> (:229-238) | 8 tokens a pass, any n | no | not reached |
| `src/ds4/ds4_mx_kern.inc:146` | `const int j = b & 7, lane = 8 * (b >> 3) + (j & 1) * 4 + (j >> 1);`, also act_lane / act_groups at ds4_gguf_cpu.cpp:153-155 and mx_act_bytes at ds4_gguf_cpu.hpp:54-57 | &7 / >>3 stride = groups of 8 MXFP4 blocks in the activation layout | no | not reached |
| `src/ds4/ds4_iq_cpu.cpp:1099` | iq_cpu_gate_up / iq_cpu_down (:1130-1139) / own_nt (:1059-1067): thread_local buffers sized 2*nt*nr, then own_rows (:813-829) or native_*_rows_split | none (nt = tokens per expert, any value; one call per expert) | only-if-misused | not reached: the K3.1 lane runs iq_cpu_rows (gate) + iq_cpu_rows (up) + k31 SiTU + iq_cpu_rows/iq_cpu_down (down); iq_cpu_gate_up fuses clamped SwiGLU |
| `src/ds4/ds4_iq512_kern.inc:63` | rows<TY, NT> / rows_iq1m<NT> (:137) for IQ2_XS / IQ3_XXS / IQ1_M, NT <= 4 via x512_rows; prefetch +2048 B (:76-77, :147-148) | NT <= 4 a pass (the caller splits into chunks) | no | not reached |
| `include/strata/kernels/cpu/exl3_cpu.hpp:53` | constexpr int kExl3CpuMaxN = 8 (band activations a call); the header is included by ds4_gguf_cpu.cpp:4, only exl3_cpu_tier() is called (:341) | tokens <= 8 per EXL3 band call | no | not reached |
| `src/ds4/ds4_iq_cpu.cpp:1146` | iq_cpu_rows(type, w, rb, k, ...): builds IqCpu{td, n_ff = k, d_row = rb} and calls iq_cpu_down, skipping native_fmt; iq_cpu_act_type (:1141) calls native_fmt(type, type, 256, 256) | none (any k; the caller sizes the activations) | no | not reached |
| `src/ds4/ds4_iq_cpu.cpp:1156` | dense_requant: `k % ggml_blck_size`, threads <= 16, kChunk = 32 (:1158-1185) | none on geometry (thread count is a setting) | no | not reached |
| `src/kernels/cuda/glm_dsa.cu:207-213` | glm_mla_attention: K = V = kv_lora absorbed latents, smem (n_sel + kv_lora) floats | K = V; n_sel + kv_lora <= ~12K floats of shared memory | no | new k31 code (k31_mla_attention: key 576 / value 512, output gate) |
| `src/ds4/ds4_iq.cu:1769` | gate_up's SwiGLU branch `GU_(T) gu_ne<T>(a, ...)` switches on a.tg only. The kernel reads the up rows as the gate's type (u = blob + tr[1] + row * row_bytes_t<T>, :863). The same happens in light_set LG_ (:1681) and set_gate_up S… | under SwiGLU, gate type == up type (tu is ignored with no refusal) | only-if-misused | refuse at geometry (k31_fmt: tg == tu for any direct iq::gate_up / set_gate_up launch), or generalize together with the SiTU EPI (EPI 2 raw gate, then a SiTU EPI 3-style pass on the up) |
| `src/kernels/cpu/native_expert.cpp:306` | native_gu_rows_split: `ff[t][r0 + r] = (g / (1.f + std::exp(-g))) * u` (also :324 for one token, and :273 for native_gu_rows), with the GLM clamp at :303-304 / :321-322. iq_cpu_gate_up falls through to it (ds4_iq_cpu.cpp:1123); i… | CPU-lane gate/up activation fixed to clamped SwiGLU | only-if-misused | not reached (K3.1's CPU lane uses iq_cpu_rows for gate and for up, then k31 SiTU, then iq_cpu_rows / iq_cpu_down: plan §3.1 k31_cpu) |

### 1.6 The ds4_gguf.cu probe and headers pulled in only through it (25 rows)

| Site | What | Cap | K3.1 reaches | Plan |
|---|---|---|---|---|
| `src/ds4/ds4_gguf.cu:2-6 (includes)` | #include ds4_gguf.cuh, ds4_pdl.cuh, ds4_iq.cuh, ds4_numerics.cuh; the only quoted include below them is ds4_gguf.cuh:15 -> strata/kernels/exl3.hpp, which has none; MAYA_DS4_IQ_ONLY makes pdl_on() false (ds4_pdl.cuh:32-33) | kPicks = 8 not visible here (0 uses); 126 = 12 (ds4_kernels.cuh) + 19 (ds4_kernels.cu) + 95 (ds4_model.cu) | no | not reached (verified claim) |
| `src/ds4/ds4_gguf.cu:1369-1375` | mx_geometry(): refuses k < 1 \|\| k > 8, n_embd % 64, n_ff % 64, n_embd > kMaxEmbd, n_ff % 32 | k <= 8; n_embd <= 8192; n_embd and n_ff multiples of 64 | only-if-misused | not reached for the k path (refuse at geometry in k31_geo upstream); set_* pass k = 1 |
| `src/ds4/ds4_gguf.cu:44` | constexpr int kMaxEnt = 16: a group's entries in one pass (a window's tokens at most) | window T <= 16 per win_experts call (:1968); Scratch xq [kMaxEnt][kMaxEmbd], hq [kMaxEnt*8][kMaxEmbd] (:1342, :1357) | only-if-misused | not reached (paired rows would need 2T <= 16, so <= 8 real tokens a call) |
| `src/ds4/ds4_gguf.cu:757-759, :815` | MxArgs comment 'entries e = t * 8 + i ... input x + (e >> 3) x_stride'; mx_gate_up_kernel xr = grp_ptr ? e >> 3 : 0 | 8-pick stride: window entry e maps to token e >> 3 | only-if-misused | paired rows (virtual token t' = 2t + half over a duplicated latent row, so e >> 3 lands on t'); otherwise not reached |
| `src/ds4/ds4_gguf.cu:912-924` | struct CombArgs: plan_ptr/plan_w [T][8], part [T * 8][n_embd], cpu_picks [T * 8][stride], cpu_pmask bit i = row t * 8 + i | 8 picks a token (stride 8), pick mask bits 0..7 | no | new k31 code (k31_moe_combine, stride kPicks31) |
| `src/ds4/ds4_gguf.cu:942-952` | mx_combine_kernel: float p[8]; for (i < a.k) { e = t * 8 + i; p[i] = ... }; combine(a.plan_w + t * 8, p, a.k, ...) | k <= 8 (p[8] overflows for k > 8; only mx_geometry upstream guards it) | no | new k31 code (k31_moe_combine with p[kPicks31] or a streaming fp32 sum, plan_w + t*kPicks31) |
| `src/ds4/ds4_gguf.cu:1338` | constexpr int kMaxEmbd = 8192: refused by mx_geometry (:1370); sizes Scratch part/xq/hq (:1357-1359) | expert-path n_embd <= 8192 (n_ff is NOT checked against it, though hq rows are n_ff wide) | yes | refuse at geometry (k31_geo width > 8192 must cover n_embd, latent and n_ff_exp) |
| `src/ds4/ds4_gguf.cu:1339, :1764, :1819, :1358, :1770, :1823` | constexpr int kMaxQK = 32768; gguf::mv refuses a kFmtGG job (m.k > kMaxQK, :1764) and a Q8_0 job ('Q8_0 inputs past the scratch', :1819); scratch mq slice i * kMaxMvCols * kMaxQK | k <= 32768 for Q8_0 and low-bit dense jobs (BF16 and MXFP4 jobs uncapped) | only-if-misused | direct launch: gguf::q8_round (:1618, no cap, caller buffers) or k31's own rounding into k31 scratch, then iq::mv; else split-K pair (second call's bias = first's y) |
| `src/ds4/ds4_gguf.cu:1340-1344, :1359` | struct Scratch { float* part; // [8][kMaxEmbd] ... }; cudaMalloc(&sc.part, 8 * kMaxEmbd * 4) | 8 token-path down outputs (parts) per stream | no | new k31 code (k31-owned scratch part [kPicks31][latent], hq [kPicks31][n_ff], xq rows) |
| `src/ds4/ds4_gguf.cu:1342, :1357` | Scratch hq/hd: h [kMaxEnt * 8] rows (nh = kMaxEnt * 8 * kMaxEmbd) | 16 tokens x 8 picks = 128 h rows | no | not reached (paired rows fit: 2T*8 <= 128) |
| `src/ds4/ds4_gguf.cu:1898-1922` | gguf::moe_gate_up: mx_geometry(k), then iq::gate_up(ia, a.k, 1) or launch_groups<GateUpK>(m, n_ff, a.k, 1); x rounded with iq_act_quant/q8_quant into Scratch row 0 | k <= 8 (mx_geometry); fused SwiGLU | only-if-misused | direct launch (iq::gate_up groups = 16 with k31 scratch); not reached |
| `src/ds4/ds4_gguf.cu:1925-1964` | gguf::moe_down: mx_geometry(k); q8/iq_act_quant(h, rows = a.k); iq::down(ia, a.k, 1) or launch_groups<DownK>; part = sc->part [8]; mx_combine_kernel | k <= 8 | only-if-misused | not reached (direct iq::down groups = 16 + k31_moe_combine) |
| `src/ds4/ds4_gguf.cu:1967-2032 (groups = a.T * a.k at :1988)` | gguf::win_experts: T <= kMaxEnt && mx_geometry(k); groups = a.T * a.k (:1988); h rounded as a.T * 8 rows (:2010); combine over plan [T][8] | k <= 8, T <= 16, h/part stride 8 | only-if-misused | not reached (the plan excludes the GGUF wrappers); paired rows possible in principle (T' = 2T <= 16, caller-owned a.part read by k31_moe_combine, sh_out/cpu null, out a throwaway) |
| `src/ds4/ds4_gguf.cu:27-1378 (mx_gate_up_kernel :796, mx_down_kernel :863, mx_combine_kernel :929)` | MXFP4 routed-expert kernels inside an anonymous namespace; iq::supported/block_bytes have no MXFP4 case (ds4_iq.cu:1706-1726) | MXFP4 experts reachable only through moe_* / win_experts (k <= 8) or one matrix at a time through gguf::mv | yes | M1: gguf::mv per expert matrix (k-agnostic, kMaxMvJobs chunks) or host dequant + cuBLAS; M2+ MXFP4 fast decode: new k31 code |
| `src/ds4/ds4_gguf.cu:178-184, :856, :1079, :1177, :1304; :1649-1667 iq_args; :2035-2049 iq_set` | swiglu(gv, uv, limit) hard-coded in every MXFP4 gate/up kernel (token/window, set, set_tc, set_wmma); iq_args/iq_set never set act, so it stays at iq::kActSwiglu (ds4_iq.cuh:33, :67, :102) | activation fixed to (clamped) SwiGLU; no SiTU; kActSitu does not exist yet | yes | direct launch (iq::set_gate_up / iq::gate_up with act = a new kActSitu) or separate gate/up GEMMs + k31_situ; use only gguf::set_down from this file |
| `src/ds4/ds4_gguf.cu:2052-2112, :2117-2172` | gguf::set_gate_up / set_down: mx_geometry(1, n_embd, n_ff); grid z = sa.n_exp; light kernels for experts of <= 16 rows (skip = 16) | k-agnostic; n_exp is a runtime value (grid z <= 65535); n_embd <= 8192, %64 | yes | reuse set_down as is (activation-free); set_gate_up see the swiglu row |
| `src/ds4/ds4_gguf.cu:1749, :1753, :76, :238, :260` | gguf::mv: n > x3::kMaxMvJobs refused; iq::MvJob IJ[kMaxMvJobs]; BfJobs j[kMaxMvJobs]; Q8Jobs j[kMaxMvJobs] | <= 8 jobs per mv call | yes | call-site chunking in the k31 runner (ceil(jobs/8) calls); not a picks cap |
| `src/ds4/ds4_gguf.cu:1764, :1790, :1726-1732` | ncol > x3::kMaxMvCols refused; mv_set tops out at mv_launch<16> | <= 16 activations (window columns) per job | yes | call-site chunking of T (prompt chunks use the cuBLAS/set paths) |
| `src/ds4/ds4_gguf.cu:1413-1610` | q8_batch_check: static const Spec spec[kMaxMvJobs] (8 fixed jobs, k 32..5120) | 8 jobs (self-test) | no | not reached (load-time check only when DS4_Q8_CHECK=1 and the runner calls it) |
| `src/ds4/ds4_gguf.cu:1876-1882` | launch_groups: ne_max <= 1/4/8 else Kern<16>; kernels loop entries in NE chunks (:808, :873) | none (entries per pass; larger groups loop); es[NE]/xr[NE] shared arrays | no | not reached |
| `src/ds4/ds4_gguf.cu:93-98, :164, :302-303, :433, :455, :462, :566, :697, :1001-1005, :1130-1147, :1194, :1209, :1252-12…` | 8-wide BF16 vectors, E2M1 LUT v[8], s_am[8]/s_at[8] (8 warps), byte offsets (o & 3) * 8, NT = NC > 8 ? 2 : 1 (column tiles), p[8] (8 warps' partials in q8_mv_tc), nibble/8-value tile chunks, WMMA row maps | none (layout or warp constants, not picks) | no | not reached (false positives) |
| `src/ds4/ds4_numerics.cuh:32, :36` | HIP E4M3 decode: e = (q >> 3) & 15u, m = q & 7u; bf16_rn, e4m3_value, e4m3_round | none (FP8 bit fields) | no | not reached (false positive) |
| `src/ds4/ds4_model.cu:21510, 21568; include/strata/kernels/exl3.hpp:121-127, 150` | ma.cpu_picks / wa.cpu_picks = lane_pick_rows(slot); x3::MoeArgs/WinArgs cpu_picks: 'pick i of token t at row t * 8 + i of cpu_picks' (exl3.hpp:124) | k stride 8 (literal) | only-if-misused | direct launch (iq::gate_up/iq::down; never build MoeArgs/WinArgs) |
| `include/strata/kernels/exl3.hpp:118-131, 137-163, 58 (exl3.cu)` | MoeArgs{k = 8, n_embd = 4096, n_ff = 2048 defaults}; WinArgs plan_ptr/plan_w [T][8], h [T*8][n_ff], part [T*8][n_embd]; EXL3_LEGACY_MOE when n_embd/n_ff <= 4096 | k stride 8; default widths 4096 | only-if-misused | never fill MoeArgs/WinArgs (direct iq launches instead) |
| `include/strata/kernels/exl3.hpp:75-76` | kMaxMvJobs = 8, kMaxMvCols = 16 (x3::MvJob batch; sizes ds4_gguf.cu's mv scratch at :1358) | <= 8 jobs a call, <= 16 activations a job | yes | direct launch (batch dense GEMVs at most 8 a call; window T <= 16) |

### 1.7 Host headers used only if K3.1 includes them (tiering, pool, lane parity, GGUF declarations) (16 rows)

| Site | What | Cap | K3.1 reaches | Plan |
|---|---|---|---|---|
| `src/ds4/ds4_gguf.cuh:95-102` | declarations: moe_gate_up/moe_down(x3::MoeArgs), win_experts(x3::WinArgs), set_gate_up/set_down(x3::SetArgs) | inherits the k <= 8 and SwiGLU limits above | only-if-misused | not reached for moe/win; set_down reused |
| `src/ds4/ds4_gguf.cuh:71-88, :104-107` | gguf::mv contract, PreQ8 {rows}, q8_round, act_q8k ('each 256's scale in its eight 32-blocks') | none (the 8 = 256/32) | yes | direct launch helpers: q8_round / act_q8k into k31 scratch before iq::mv / iq::gate_up |
| `include/strata/kernels/exl3.hpp:136-168` | struct WinArgs: plans [T][8] (:145-146), entries e = t * 8 + i (:138), h [T * 8][n_ff], part [T * 8][n_embd] (:141); T = 0, k = 8, n_embd = 4096, n_ff = 2048 (:162) | 8-pick stride per token; default widths | only-if-misused | paired rows if ever used (2 virtual tokens per token); otherwise not reached |
| `include/strata/kernels/exl3.hpp:184-197` | struct SetArgs: n_exp runtime, bounds n_exp + 1, max_rows; int n_embd = 4096, n_ff = 2048 (:194); skip_ge | no expert-count cap; default widths 4096/2048 | yes | reuse with explicit n_embd = 3584, n_ff = 3072 (k31_geo asserts they are set) |
| `include/strata/kernels/exl3.hpp:96-115` | ExpertLayout gg_type[3] per role; kMaxClasses = 32 MixedLayouts | 32 EXL3 mixed classes per layer (not experts, not picks) | no | not reached |
| `src/ds4/ds4_lane_parity.hpp:30` | par::gate_up / down 'nt <= 8 activations'; implementation ds4_lane_parity.cpp:40 `constexpr int kMaxT = 8;` with unchecked stack arrays (:164, :280-292, :433, :455) | tokens <= 8 per call, else stack overflow | no | not reached |
| `src/ds4/ds4_pcie.hpp:50` | split(): 'n picks (n <= 8)' (:48); `for (unsigned m = 1; m < (1u << n); ++m)`, exhaustive over 2^n masks, each split_time costing O(n) | documented k <= 8; cost 2^n (65,535 masks x 16 at k = 16, about 1M operations a call) | yes | generalize (with a split by counts: cost depends only on how many warm and cold picks stream, so O((n+1)^2); keep 'fewer streamed, then lower mask' by taking the lowest-index picks of each class; keep the exhaustive form as the test oracle up to n = 16) |
| `src/ds4/ds4_pcie.hpp:87` | streams_any(): `for (int n = 1; n <= n_max && n <= 8; ++n)` ... `split(k, n, (1u << n) - 1u, max_s)` | hard clamp n <= 8 | yes | generalize (drop the 8 and loop to n_max = 16; cheap once split works by counts) |
| `src/ds4/ds4_pcie.hpp:71` | table(): `int n_max = 8` default, `for (n = 0; n <= n_max; ++n) split(...)` | log table to 8 picks by default | yes | generalize (pass kPicks31 explicitly) |
| `src/ds4/ds4_pcie.hpp:31` | split_time(const Costs&, int n, unsigned cold, unsigned mask): bit i = pick i | 32-bit masks; `1u << n` is undefined at n = 32, so at most 31 picks | yes | generalize; static_assert(kPicks31 <= 31) where K3.1 builds the masks (16 is fine) |
| `src/ds4/ds4_pcie.hpp:116` | cal_get / cal_put (:132): `char line[4096]` | 4096-char line buffer for the lane-times file (not a geometry cap) | no | not reached |
| `src/ds4/ds4_place.hpp:38` | shares / queue (:63) / even (:100) / split (:136): vectors sized by n_exp at run time; recent_n is a parameter; spare = 1 (:141) | none (E taken at run time) | yes | not reached |
| `src/ds4/ds4_store.hpp:55` | Geo `int file[3]; uint64_t off[3], stride[3], bytes[3];` plus tr_[3] (:784); where_ / heat_ sized n_exp (:168-169); Hold vectors sized n (:196-200) | exactly 3 matrices an expert; no limit on picks or E | yes | not reached |
| `src/ds4/ds4_ram_slice.hpp:17` | `if (parts < 1 \|\| parts > 2 ...) return -1;` with std::array<RamSlice, 2> | NUMA parts <= 2 (not geometry) | no | not reached |
| `src/ds4/ds4_pool.hpp:64` | Pool ticket `(g << 32) \| 0x7fffffffu`: 32-bit generation, 31-bit job index | none in practice (jobs < 2^31) | yes | not reached |
| `src/ds4/ds4_os.hpp:506` | mbind `unsigned long mask[4]` with maxnode 256 (also :569-571); Windows affinity `cpu < 64` (:581, :591); kDirectAlign 4096 (:79) | OS-level only (256 NUMA nodes, 64 CPUs on Windows, 4 KiB O_DIRECT alignment) | yes | not reached |

### 1.8 Not compiled into maya-k31 (maya-ds4 runner, GLM fast path, other routers) (42 rows)

| Site | What | Cap | K3.1 reaches | Plan |
|---|---|---|---|---|
| `src/ds4/ds4_iq.cu:789-795` | `#if defined(STRATA_USE_HIP) \|\| defined(MAYA_DS4_IQ_ONLY)` makes DS4_IQ_PERSIST_BUILD 0, so kPersist (TAB·4 ≥ 8192: IQ1_S/IQ1_M/IQ2_S) and its known[64] tables (:1473, :1505) and the IQ1_M code table (ds4_iq1_codes.hpp:30 kIq1m… | none (feature compiled out) | no | not reached |
| `src/ds4/ds4_iq.cu:1825` | HIP MMQ prompt sets (block :1805-2069): kMmqCap = 32768; mmq_wanted `n_exp > 512` (:1835), `n_embd != 5120 \|\| n_ff != 2304` (:1841); ensure `ne > 512` (:1894), q8_bytes(nrows, 5120) (:1902), nrows·2·2304 (:1904), bounds 513 int… | E ≤ 512, rows ≤ 32768, DS4 geometry 5120/2304 only | no | not reached |
| `src/ds4/ds4_iq.cu:2114` | HIP fused RDNA4 prompt sets (block :2071-2178): fused_wanted `n_exp > 512`, `n_embd != 5120 \|\| n_ff != 2304 \|\| act != kActSwiglu` (:2117) | E ≤ 512, 5120/2304, SwiGLU only | no | not reached |
| `include/strata/prefill/moe_mmq.hpp:97` | `IDS_PAD = 512`; q8 rows padded to 512 values (:19); Product max_rows (:39, :49) | 512-value row padding / ids pad | no | not reached |
| `include/strata/prefill/moe_fused_rdna4.hpp:8` | 'Fixed GLM native expert geometry: E=4096, FF=2048'; Geometry n_embd/n_ff 0 = GLM 4096/2048, DeepSeek 5120/2304 (:13) | width 4096/5120, FF 2048/2304 | no | not reached |
| `src/ds4/ds4_kernels.cuh:22` | constexpr int kPicks = 8, and its other 11 lines in this header (Gather src[kWin*kPicks], Combine row/w[kWin][kPicks], Req ids/w[kWin][kPicks], ...) | k <= 8 | no | not reached |
| `src/ds4/ds4_kernels.cuh:355-372` | kRing 64, kMaxDim 8192; Req{ids[kWin][kPicks], w, ahead[kWin][kAhead], x[kWin][kMaxDim]}, Resp{cpu[kWin], pmask[kWin] (bit i a pick), part[kWin][kMaxDim]} - the host-mapped CPU-lane mailbox | k <= 8; dim <= 8192; window <= 16 | no | new k31 code (K3.1's own lane mailbox sized kPicks31) |
| `src/ds4/ds4_kernels.cu:865-914` | route_kernel: __shared__ sc[1024], bs[1024], pick[16]; logits row stride 1024; sqrt(softplus) scores; idx/w written at t*kPicks + r (:911-912) | E <= 1024 (implicit, not checked); k stored at stride 8 although pick[] holds 16 | no | new k31 code (k31_route) |
| `src/ds4/ds4_kernels.cu:2858-2916, 2934-2939, 2980-3032` | route_publish_kernel (sc[1024], pick[16], idx/w/plan_ptr at t*kPicks :2914-2916), route_ahead_kernel (bs[1024] :2939), route_pair_kernel (sc/bs[1024], pick[16] :2982-2985, a.idx/a.w/a.plan_ptr at t*kPicks :3030-3032) | E <= 1024; k stride 8 | no | new k31 code (k31_route) |
| `src/ds4/ds4_kernels.cu:3823-3826` | route_pair refusal: a.n > 1024 \|\| a.k > kPicks \|\| a.a_n > 1024 \|\| a.off + a.m > kAhead \|\| a.dim > kMaxDim \|\| a.T > kWin; also route_ahead n <= 1024 && off+m <= kAhead (:3820); resident_skip clamps STRATA_DS4_RESIDENT_TO… | k <= 8, E <= 1024, dim <= 8192, T <= 16 | no | not reached |
| `src/ds4/ds4_kernels.cu:3115-3134` | helper_take_kernel: for j < kWin*kPicks, t = j / kPicks, i = j % kPicks (:3133-3134); plan[t][i] rows kPicks apart | k stride 8 | no | not reached |
| `src/ds4/ds4_kernels.cu:3620-3680, 3924-3935` | moe_sort_kernel (vs[idx[t*kPicks+i]], pos[t*kPicks+i] :3635, :3647), moe_combine_kernel (pos/w at t*kPicks :3672-3680), moe_combine_x_kernel (:3933-3935) | k stride 8 (k is a parameter, but every row is kPicks apart) | no | new k31 code (a prompt-path pick sort and combine at stride kPicks31) |
| `src/ds4/ds4_kernels.cu:2801-2830, 4707-4718` | gather_kernel(Gather src[kWin*kPicks]), combine_kernel(Combine n/row/w[kWin][kPicks], cpu[kWin]) | k <= 8, T <= 16 | no | new k31 code (k31_moe_combine) |
| `src/ds4/ds4_kernels.cu:4047-4050, 4137-4176, 1571; ds4_kernels.cuh:214` | index_score refuses hd != 128 \|\| heads > 32; topk with any k (segmented: kTopkSeg 16384 at rows >= kTopkSegMin 32768); topk_merge kMergeCap 8192 (a cap above it falls back to the all-candidates path, same picks) | indexer <= 32 heads x 128; no top-k cap | no | new k31 code (copy into src/k31 for the M3/M4 fast path) |
| `src/ds4/ds4_kernels.cu:4533; ds4_kernels.cuh:221-248` | sparse_attn: if (d != 512 \|\| fmt not FP4/BF16) refuse; K = V with a sink per head; AttnScratch rows [max_t*cap][512] BF16 | head dim d == 512, K = V | no | generalize (M3: a k31 sibling with K 576 / V 512 / no sink) |
| `src/ds4/ds4_model.cu:3507` | load refusal: g.hd != 512 \|\| g.ix_hd != 128 \|\| g.k_exp > 8 \|\| g.hc > 8 -> 'geometry not covered' | k <= 8 (literal 8), head_dim 512 | no | not reached |
| `src/ds4/ds4_model.cu:3915, 1415-1417, 12217, 15237` | DSpark l.k_exp > kPicks refusal; route_fusable (n_exp <= 1024, k_exp <= kPicks, m <= kAhead, D <= kMaxDim); lookahead n.n_exp > 1024; PCIe MXFP4 stream k_exp <= kPicks | k <= 8, E <= 1024, dim <= 8192 | no | not reached |
| `src/ds4/ds4_model.cu:613, 4260-4277, 4527-4528, 5104-5105, 13419-13420, 13527, 17812, 21614-21615` | device/host routing scratch sized kWin*kPicks or Z*kPicks: d.rw/d.ridx, h_w/h_idx, chunk ridx/rw/row_tok/pos, route regions T*kPicks, c.ridx + r0*kPicks | k stride 8 | no | not reached |
| `src/ds4/ds4_model.cu:10087 (+10618, 10982-10989, 11110, 11194-11197, 16359)` | static constexpr int kSlots = kWin * kPicks (the CPU lane's (token, expert) rows a window can need); c_xh/c_xsb/c_acc rows (X*kSlots + sl); es/hi[kWin*kPicks]; n > kWin*kPicks refusal | <= 128 (token, pick) rows a window = 16 x 8 | no | new k31 code (the K3.1 CPU-lane split sized kWin31*kPicks31) |
| `src/ds4/ds4_model.cu:1567-1571, 4863, 14594-14604, 14883-15073, 21632-21691` | lane_picks [kRing][kWin][kPicks][dim] (lane_pick_rows); CPU-lane entry codes t*kPicks + i (wv/tok_of/pick_of/ord[kWin*kPicks], pk_rows + (t*kPicks+i)*D :15073); vex/vrow0/vn[kWin*kPicks], h_idx[t*kPicks+i] :21641, x3::MvJob j[2*k… | k stride 8 | no | not reached |
| `src/ds4/ds4_model.cu:1589-1593, 13988-13993, 14524-14525, 14578, 14662, 14748, 12434-12459` | helper GPU: helper_plan [kWin][kPicks], helper_w/hw/part [kWin*kPicks], ma.plan_ptr = helper_plan + t*kPicks; tmask 1u << (e / kPicks); held-expert probe p[kPicks], w[kPicks] | k <= 8 | no | not reached |
| `src/ds4/ds4_pcie.cuh:18-33; ds4_model.cu:1781-1786, 15180-15199, 15279, 15409-15417, 15438-15606, 15832-15868, 15998-15…` | kPullEnt = 16 * 8 (entry t*8 + i, signed char ent[kPullEnt] :28); kSlots = 16 (a pull's streamed experts); Pull src/ready/mat/dst[kSlots]; runner arrays const int (*ids)[kPicks], slot/ex/se[kPicks], went[kPullEnt], plan/plan2 [kP… | window entries 16 x 8; 16 streamed experts a pull | no | new k31 code (M2 streaming lane sized kPicks31; a token's 16 picks exactly fill pcie::kSlots, a window does not) |
| `src/ds4/ds4_model.cu:933, 12230, 17873-17874, 19650-19656, 2174-2175, 17845-17854` | lookahead/warm: last_picks kPicks a row, m = min(na, kAhead1 or kPicks); DS4_DUMP_R record header [.., kPicks] and picks [T kPicks] | k stride 8 | no | not reached |
| `src/ds4/ds4_model.cu:13279-13306, 13661, 21467, 14492-14500, 21548-21556` | route_pair(...) calls (DS4_ROUTE_FUSE); gf::MoeWin mw.k = l.k_exp; gf::moe_win_group(mw, s) for a window | k <= 8, T <= kWin 16 | no | not reached |
| `src/ds4/ds4_model.cu:10393, 18536` | (uint64_t) id * 8 + li (Engram cache key); hand_ack + (slot * 8 + mi) * 32 (micro-batch flags) | none (not pick-related) | no | not reached |
| `include/strata/core/glm_limits.hpp:33-40, 64, 67-68` | kMaxRouted = 8, kMaxExperts = 512, kMaxEmbd = 4096, kSlowMaxRouted = 64, kRankNone = 8 (static_asserts :39-40); slow_fits / check_fast_limits | k <= 8 fast, k <= 64 slow, E <= 512, n_embd <= 4096 | no | not reached (k31_geo keeps its own limits in the glm_limits_test pattern) |
| `include/strata/kernels/glm_fast.hpp:282-283, 289-299, 324, 328, 337-363, 368-384` | kRoutePicks = 8, kRouteExperts = 512; MoeDev plan_ptr/plan_w/plan_id/fetch_src/pf_* [8]; MoeRequest ids/w/promo_ptr/pred/pf_ids/pf_ptr/cpu_src[8], ahead[kAhead=4][8], near_ids[16], x[4096] (static_asserts :355-363); MoeResponse p… | k <= 8, E <= 512, n_embd <= 4096 | no | not reached (GLM fast-path rings and plans are not reused) |
| `include/strata/kernels/glm_fast.hpp:284-287; src/kernels/cuda/glm_fast.cu:3919, 4124, 4172, 4237` | cpu_plan nibble: (plan >> 4f) & 15 = how many of f RAM-tier experts go to the host; kPlanMaxF = 8; static_assert(4*kPlanMaxF + 4 <= 64) | 4-bit count, f <= 8 | no | not reached |
| `include/strata/kernels/glm_fast.hpp:389-396; src/kernels/cuda/glm_fast.cu:3917, 3929, 3956, 4169, 4231, 4257, 4270` | '8 = none' sentinels: skip_from = kRoutePicks, disk_skip_from = kRoutePicks (rank >= skip_from left out; kRoutePicks means nothing left out) | rank sentinel 8 | no | not reached |
| `src/kernels/cuda/glm_fast.cu:3968-3971, 5431-5440, 6621, 6683` | route kernel __shared__ s_p/s_sel[kRouteExperts], s_ids/s_w[kRoutePicks]; route_fits refuses k > 8 or E > 512 (a sticky launch error); moe_route; ahead n_expert <= kRouteExperts | k <= 8, E <= 512 | no | not reached (k31_route) |
| `src/kernels/cuda/glm_fast.cu:7095-7135, 7269, 7294, 7031; include/strata/kernels/glm_fast.hpp:403-423` | moe_win_group_kernel: kWinEnt = kMaxWinTokens*8 = 128, e = t*8 + i, pp[e] = (e & 7) < w.k ? plan : 0; route_fits(k) on moe_win_group; moe_win_experts refuses T > 16 \|\| k > 8; moe_experts_multi k > 8; MoeWin plan [T][8], scratch… | k <= 8, T <= 16 rows | yes | paired rows |
| `include/strata/kernels/glm_fast.hpp:44-50, 487, 490` | kMaxMvCols 16, kMaxMvJobs 8, kMaxMvNt 16 (glmf::mv); mtp_in / rms_q8 'n <= 4096' | n_embd <= 4096; 8 jobs, 16 cols | no | not reached (dense goes through gguf::mv / iq::mv / cuBLAS) |
| `src/kernels/cuda/dsa_topk.cuh:23-32, 103-125; glm_fast.cu:3093-3094, 6292-6296, 6314-6318; glm_batch.cu:682-699, 2451-2…` | select_top_pools: '1 <= top <= 1024', s_win[NT=1024], s_gt/s_eq[1024], bitonic sort over <= 1024 winners; glmf::dsa_select / dsa_select_rows and glmb::dsa_select pass top_pools without a guard | attention top-k <= 1024 | only-if-misused | not reached (glm_dsa_select O(n^2) for M1); for an O(n) fast path, generalize (2048 winners) or copy DeepSeek's topk |
| `src/kernels/cuda/glm_fast.cu:3497, 6476-6480, 6511` | MLA_KV = 512 (tensor-core rows); mla_rows_ok: kv_lora == MLA_KV; mla_split_attend fast branch kv_lora == 512 | latent row == 512, K = V | no | generalize (M3: a 'GLM mla_rows sibling' in src/k31 with K 576 / V 512) |
| `src/kernels/cuda/glm_batch.cu:319, 346-347, 397-398, 407; include/strata/kernels/glm_batch.hpp:58-71` | kda_rec / kda_rec_split / kda_prep: constexpr int HD = 128; q/k l2 norm rsqrtf(ssq + 1e-6f), the eps hard-coded | head_dim == 128; l2 eps 1e-6 | no | refuse at geometry (fast path only if kda.l2_norm_epsilon == 1e-6) or generalize (eps parameter) |
| `include/strata/kernels/glm_batch.hpp:15-18; src/kernels/cuda/glm_batch.cu:113-123` | glmb kRoutePicks = 8, kRouteExperts = 512; route_fits for route / moe_combine / moe_combine_add | k <= 8, E <= 512 | no | not reached |
| `src/kernels/cuda/router_sigmoid.cu:27, 81-88, 70` | RS_MAX_THREADS 512; k > 64 exits; n_expert > 512 exits; weights 1/fmax(sum, 6.1e-5), not +1e-20 | E <= 512, k <= 64 | no | new k31 code (k31_route) |
| `src/kernels/cuda/router_top10.cu:65, 194-201` | RT_MAX_THREADS 512; k > 64 exits; n_expert > 512*64 exits; softmax over all experts, no bias | k <= 64, E <= 32768 | no | not reached |
| `src/kernels/cuda/native_router.cu:49, 52, 111-112, 120-122` | logits stride 512, ids/weights 10 a token (fixed), float values[16]; validity checks on 512*4 / 10*4 buffers | E == 512, k == 10 (fixed) | no | not reached |
| `src/kernels/cuda/exl3.cu:2903-2908, 72, 2577` | moe_gate_up/moe_down: a.k < 1 \|\| a.k > 8 \|\| n_embd/n_ff > kMaxMoeK (8192) \|\| k*n_ff > kHFloats (8*8192) -> 'geometry not covered' | k <= 8; widths <= 8192 | no | not reached (direct iq launches) |
| `src/kernels/cuda/exl3.cu:3027-3031, 3037, 2578` | win_experts: T < 1 \|\| k > 8 refusal; T*8*n_ff > kUFloats (128*4096: 16 tokens x 8, n_ff <= 4096) | k <= 8; 16 x 8 entries; n_ff <= 4096 | no | not reached |
| `src/kernels/cuda/exl3.cu:863, 985-991, 1011, 1225, 872` | float p[8] combine registers (moe_down last block, moe_combine_kernel, win_combine_kernel); e = t*8 + i, plan_w + t*8 | k <= 8 (register array; guarded by the refusals above) | no | not reached |

### 1.9 `static_assert` list

Every structure K3.1 copies or creates that holds per-pick state carries `static_assert(kPicks31 >= 16)`. Where a
32-bit mask holds picks, it also carries `static_assert(kPicks31 <= 31)`. There must be no width or k defaults
inherited from DeepSeek's or GLM's structs.

| K3.1 structure | Replaces / mirrors | Assert |
|---|---|---|
| `k31_route` output `ids[T][kPicks31]`, `w[T][kPicks31]`, the in-block pick buffer | `route_kernel`'s stride-8 `idx`/`w` (`ds4_kernels.cu:911-912`), `pick[16]` | `kPicks31 >= 16`; stride kPicks31, never 8 |
| decode expert plan `plan_ptr[kPicks31]` + weights, `groups = kPicks31` | `iq::Args::plan_ptr` (`ds4_iq.cuh:49-52`) | `kPicks31 >= 16` |
| decode scratch `h[kPicks31][n_ff]`, `hq/hd`, `part[kPicks31][latent]` | gguf `Scratch part[8]` (`ds4_gguf.cu:1340-1362`) | `kPicks31 >= 16` |
| paired-row window builder (`grp_ptr`/`grp_start`/`grp_ent`, duplicated latent rows `[2T][latent]`), `kIqWinPicks = 8` | `ds4_iq.cu:856` `e >> 3`, `glmf::moe_win_group` | `kPicks31 == 2 * kIqWinPicks`; refuse `2T > 16` where a 16-entry window applies |
| `k31_moe_combine` per-token tables and accumulator | `mx_combine_kernel` `float p[8]`, `plan_w + t*8` (`ds4_gguf.cu:942-952`); exl3 `p[8]` | `kPicks31 >= 16` |
| the `iq::mv` job list `MvJob[2·kPicks31 + 2·n_shexp]` | `mv()`'s `done[64]` (`ds4_iq.cu:2243`) | `2 * kPicks31 + 4 <= 64` |
| the prompt set chunker with a mirrored `kIqLightMaxExp = 512` | `ds4_iq.cu:1239` | `kMaxExp31 <= 2 * kIqLightMaxExp` (two chunks at E ≤ 1024) |
| the CPU-lane split (count-based) and the lane request (ids, pick indices, cold/stream masks) | `ds4_pcie.hpp` `split`/`streams_any`/`table`; ds4 `Req`/`Resp` | `kPicks31 >= 16`, `kPicks31 <= 31`; `n_max = kPicks31` |
| CPU-lane per-pick activation and partial buffers | GLM's `8 * kNativeHBytes` (`glm_fast_path.cu:3087`, `:3482`) | `kPicks31 >= 16` |
| any lane-pick ring / gather / pull table forked from `ds4_model.cu` / `ds4_pcie.cuh` (`kSlots31 = kWin31 * kPicks31`, entry codes `t*kPicks31 + i`) | `kSlots` (`ds4_model.cu:10087`), `kPullEnt` | `kPicks31 >= 16` |
| any keep/skip rank sentinel ("none" = kPicks31) and packed RAM-tier count (≥ 5 bits a count) | GLM's "8 = none" and 4-bit plan nibbles (`glm_fast.hpp:284-287`, `:389-396`) | `kPicks31 >= 16` |
| the int8 dense scratch for `iq::mv` (layer-0 down k = 33792) | gguf `kMaxQK = 32768` (`ds4_gguf.cu:1339`) | `kMaxQK31 >= 33792`, or a refusal sized from the geometry |
| `k31_geo`'s refusal table (k > kPicks31, E > 1024, any width > 8192 including `n_ff_exp`) | `mx_geometry` (`ds4_gguf.cu:1370`), `glm_limits.hpp` | `kPicks31 >= 16` next to the k refusal, in the `glm_limits_test` pattern |
| parity dump/record headers and lookahead rows | `DS4_DUMP_R` header `[.., kPicks]` | `kPicks31 >= 16` |

## 2. The build (`STRATA_BUILD_K31`)

`maya-k31` is one CMake block (`CMakeLists.txt`, after maya-mm3's) behind `-DSTRATA_BUILD_K31=ON` (or `-DMAYA_BUILD_K31=ON`),
off by default and CUDA only for now (sm_70 and newer, as maya-mm3).

- **What it compiles.** `src/k31/k31_main.cpp`, the M1a kernels (`k31_route`, `k31_situ` + `k31_situ_cpu`,
  `k31_moe_combine`, `k31_attn_res`, `k31_mla`, `k31_kda`, `k31_index`), the header-only `k31_geo.hpp`, and
  `src/ds4/ds4_iq.cu`, `ds4_iq_cpu.cpp`, `ds4_gguf_cpu.cpp` under `MAYA_DS4_IQ_ONLY`. It links `strata_kernels` (the GLM
  KDA, GDN and DSA kernels), `strata_kernels_cpu`, cudart and cuBLAS. `ds4_lt.cu` and the `ds4_gguf.cu` probe are not
  in it yet.
- **Tests** (`STRATA_BUILD_TESTS`, label `k31`: `ctest -L k31`): `k31_<kernel>_test --selftest` for the seven kernels
  (cuBLAS for moe_combine, mla and kda; strata_kernels for kda), `k31_geo_test`, `k31_pack_test` (the pack reader and
  every refusal, §3), `k31_pack_roundtrip` (the tiny fixture packed by `tools/k31/pack_gguf.py` and read back against
  the GGUF and `k31::read`), `maya_k31_cli` (`--version` against the tree's release and build id, and the command
  line's refusals, `cmake/k31_cli_test.cmake`) and `k31_include_guard`.
- **Flags, on k31's own files only** (plan §10.8): `k31_situ_cpu.cpp` gets `-ffp-contract=off -fno-fast-math`,
  `k31_main.cpp`, `k31_geo_test.cpp` and `k31_pack_test.cpp` `-ffp-contract=off`, the `.cu` tests (float64 host
  references) `-Xcompiler=-ffp-contract=off`. The shared ds4 sources (`ds4_iq.cu`, `ds4_iq_cpu.cpp`, `ds4_gguf_cpu.cpp`)
  get no k31 flag: they build with maya-ds4's flags (its warnings, `strata_warnings`' `-Wall -Wextra` on the two CPU
  files, through `strata_kernels`, and so `-Werror` under `STRATA_WERROR=ON` as in maya-ds4) plus maya-mm3's
  `MAYA_DS4_IQ_ONLY`; only the include directories differ (`src/k31` and the build's `generated/`).
- **Guards.** At configure time, `cmake/k31_guard.cmake` refuses any `#include` of `ds4_kernels.cuh` or
  `ds4_model.hpp` in `src/k31/*` or `tests/k31_*` (§1.2 item 1), and a deferred check refuses fast math reaching any
  k31 source or target: `--use_fast_math`, `-use_fast_math` and its parts (`-ftz=true`, `--prec-div=false`,
  `--prec-sqrt=false`), `-ffast-math`, `-Ofast`, `-funsafe-math-optimizations`, `-ffinite-math-only`,
  `-fassociative-math` or `-freciprocal-math` in `CMAKE_{CUDA,CXX}_FLAGS[_<CONFIG>]`, a k31 target's options, what its
  linked targets hand down, or a k31 source's properties. `k31_include_guard` checks the compiled include closure (the
  compiler's dependency lists) of every k31 object.
- **Byte identity** (G-ID): `tools/k31/byte_ident.sh --build DIR` builds one Ninja tree OFF, ON and OFF again and fails
  on any non-k31 edge that runs, any output file whose sha256 moves, any build command or registered test that
  changes. `tools/k31/selftest_baseline.sh compare` then runs every other engine's tests against the recorded baseline.
- **`maya-k31 --pack-info FILE`** prints the geometry `k31::read` takes from the header and tensor directory (no
  tensor data, no GPU: a header-only file works), the kept experts' source ids (the `expert_prune` keys, else the
  `<file>.prune.json` sidecar), the tensor types with the kernel that takes each (`iq::mv`, asked of `ds4_iq`, or
  cuBLAS), and `k31::dump`'s lines, which equal `tools/k31/geo_ref.py`'s. A file it cannot run exits 1 with
  `k31::read`'s message. The engine's modes (`--serve`, `--score`, `--dump-raw-dir`, `--bench`, `--experts`) are
  refused until they exist.

## 3. The pack (`k31.txt`, `tensors.txt`)

`tools/k31/pack_gguf.py --model FILE.gguf [--out DIR] [--header-only]` writes a directory of text beside the weights
(default `<stem>.maya-k31`; no weights are copied), and `src/k31/k31_pack.hpp` (`k31::read_pack`) reads it back,
refusing again whatever the packer refuses. Every refusal reads `maya-k31: pack: ...` and names the record, key or
layer. Records are one a line, fields separated by spaces; a line's fields are all of it (extra fields are refused),
and every record but `layer`, `class` and `keep` appears once.

`k31.txt`:

```
format gguf
arch kimi-k31
data whole | header-only            header-only: the tensor data is not in the file (--pack-info and tests only)
n_layers N   n_embd N   vocab N   n_head N   context N   attn_res_block N
kda_heads N   kda_head_dim N   d_conv N   gate_lower_bound X (< 0)
mla_heads N   kv_lora N   qk_nope N   qk_rope N   v_head N   q_lora N
idx_heads N   idx_dim N   idx_topk N   idx_rope_dim 0   idx_topk_freq 1
rms_eps X   lora_eps X   kda_l2_eps X   idx_ln_eps X   router_renorm X      (each in (0, 1))
layer <L> attn kda|mla ffn dense|moe n_exp <E> k <k> edim <latent> eff <expert ff> sff <shared ff> norm_div <N>
class <L> <slot> <format> <count> blk.L.ffn_gate_exps.weight blk.L.ffn_up_exps.weight blk.L.ffn_down_exps.weight
keep <L> <source id of each of the layer's experts, in file order>
prune source_count <S> method <rest of the line>
model_file <the GGUF, relative to the pack directory>
```

(each scalar is its own `key value` record; the grouping above is for reading only.)

- One `layer` record a layer, `0..n_layers-1` in order, the dense layers first (`n_exp 0 k 0 edim 0 sff 0`). A MoE
  layer keeps the same `n_exp` as every other, picks `1 <= k <= min(n_exp, 16)`, and has widths above 0.
- One `class` a MoE layer, slot 0, holding all `n_exp` experts in one ggml format (gate, up and down alike), naming
  that layer's own three slabs. Its widths (`edim`, `eff`) are multiples of the format's block (256 values for the
  IQ and K-quant formats, 32 or 64 for the others; `tools/gguf_reader.py` `BLOCK_GEOMETRY`); a format outside that
  table is "not built in". A per-role mix (say a higher-bit down) or a second class a layer comes only if D-FMT picks
  it (plan §3.4), as do per-layer expert widths (thin experts), which the packer refuses for now.
- One `keep` a MoE layer: `n_exp` distinct ids in `[0, source_count)`. They come from the file's
  `kimi-k31.expert_prune.keep` / `.source_count` keys, else the `<FILE>.prune.json` sidecar (looked up beside the path
  as given, so a link to the weights finds the sidecar beside the link, as `k31::read` does), else the file's own
  `0..E-1` (`method none`, with a note); keys and sidecar present together must agree. The packer applies
  `k31::read`'s rules to both (§2's `--pack-info`), and `k31_pack_roundtrip` checks the two readings agree.
- `prune source_count` is at least `n_exp`. `blk.N.ssm_a` (A_log) must hold exactly `kda_heads` entries: it is sliced
  to the head count at conversion, as `k31::read` requires, so the pack carries no A_log record.

`tensors.txt`: `file <i> <path>` records (0, 1, ... in order; file 0 is `model_file`), then one
`t <name> <ggml type> <file> <offset> <bytes> <dims...>` a tensor: the offset absolute in that file, dims outermost
first (ggml's `ne` reversed), bytes = the dims' elements / block values x block bytes, rows (the last dim) whole
blocks, each name once, no two tensors overlapping, and every class's slabs present at
`[n_exp, eff, edim]` (gate, up) / `[n_exp, edim, eff]` (down) in the class's format. `read_pack` checks every tensor
ends inside its file unless the pack is `data header-only`; the packer refuses a file whose tensor data runs past
its end (a download or copy still running) unless given `--header-only`.

Not in the pack yet: `tokenizer/` (plan §2.4 step 6; `tools/mm3/pack_gguf.py` writes one for MiniMax) and
`profile.bin` (the expert-usage profile M2's placement reads). Both land with the milestones that read them.
