# RDNA decode expert kernels

For `n_embd=4096`, `n_ff=2048`, the HIP decode dispatcher selects kernels by
architecture:

| Architecture | IQ2_XXS gate/up (`gate_up<16>`) | IQ2_S down (`down<22>`) | IQ3_XXS down (`down<18>`) |
| --- | --- | --- | --- |
| gfx1100 | Optimized | Optimized | Optimized |
| gfx1151 | Optimized | Optimized | Original (legacy) |
| gfx1201 | Original (legacy) | Original (legacy) | Original (legacy) |

gfx1201 is unchanged and untuned by this expert-kernel work. Other shapes,
other quant types, other architectures and CUDA keep the original dispatch. Set
`MAYA_HIP_EXPERTS_LEGACY=1` before starting a process to force original HIP
expert kernels for an end-to-end comparison.

The main change is packed sign expansion: a multiply spreads four sign bits
into the high bit of four bytes, then masks and shifts produce the XOR masks
and two's-complement corrections. Codebook bytes are positive and nonzero, so
ordinary word addition introduces no carries between bytes. This replaces
HIP's emulated packed compare/subtract while retaining the original integer
scale divisions, FP32 accumulation order, clamping, q8_1 quantization and
expert combine order.

Gate/up retains sixteen wave32 waves per workgroup, two gate and two up rows
per wave, its 2 KiB codebook in LDS, and the fused shared-down branch. Down
uses two output rows per workgroup instead of four on optimized paths, reducing
its register budget. IQ2_S reads its 8 KiB codebook through global memory;
IQ3_XXS keeps its 1 KiB codebook in LDS. The grid selection and sign operations
are HIP-only.

The signed-codebook technique was adapted from
[Strata's moe_fused_iq.cu](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/src/prefill/moe_fused_iq.cu)
(MIT). Its HIP sign lookup tables were measured first; direct sign arithmetic
performed better here. The benchmark retains alternatives for reproducible
comparison: sign LUTs, global/LDS codebooks, eight/sixteen gate waves,
two/four/eight down rows, alignment-safe eight-byte IQ2_XXS/IQ3_XXS quant loads, a separate
shared-down launch, and an eight-wave occupancy constraint. Wider loads,
splitting shared down, and the occupancy constraint did not beat the selected
paths. This work does not fuse gate/up and down into one kernel.

## Reproduce

```bash
cmake -S . -B build-multi -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/rocm-7.2.1 \
  -DCMAKE_HIP_COMPILER=/opt/rocm-7.2.1/llvm/bin/clang++ \
  -DMAYA_ENABLE_HIP=ON -DMAYA_ENABLE_CUDA=OFF \
  -DMAYA_PREFILL_MMQ=ON -DMAYA_NATIVE_EXPERTS=ON \
  -DMAYA_BUILD_TESTS=OFF \
  -DMAYA_GGML_DIR=/ai/github/project-maya/third_party/llama.cpp \
  '-DCMAKE_HIP_ARCHITECTURES=gfx1100;gfx1201;gfx1151'
CCACHE_DIR=/tmp/maya-experts-ccache cmake --build build-multi -j 8 \
  --target maya hip_expert_kernel_bench glm_ffn_parity glm_layer_parity \
  glm_model_test iq1_s_parity hip_intrinsics
export HIP_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/opt/rocm/lib
build-multi/hip_expert_kernel_bench
build-multi/hip_expert_kernel_bench --shared
build-multi/hip_expert_kernel_bench --parity-only --sweep
build-multi/glm_ffn_parity --selftest
build-multi/glm_layer_parity --selftest
build-multi/glm_model_test --selftest
build-multi/iq1_s_parity
build-multi/hip_intrinsics
```

Run these commands from the repository root. `--sweep` also works with timing;
without it the benchmark compares original kernels against production
dispatch. Baseline launches bypass production dispatch, so both implementations
are measured in the same binary. `--shared` includes a Q6_K shared-down matrix
of shape 4096x2048 in the gate/up launch and a shared contribution in down.

Random finite quantized blocks use the actual GGML block layouts (66, 82 and
98 bytes). Eight expert blobs contain `[gate | up | down]`. Timing rotates
five independent pools: each kernel's weight working set exceeds the 7900 XT's
80 MiB cache, including down's smaller matrices. Requested weight and buffer allocations are below 300 MiB
for either timed fixture. Parity-only uses one pool and less than 80 MiB.
Each timing graph contains sixty launches, with seven alternating original/new
rounds after warmup. Reported latency is the median graph time per launch.
GB/s counts quantized weight bytes (plus shared-down weights with `--shared`),
not codebooks, activations or actual DRAM transaction bytes.

## RX 7900 XT results, 2026-10-08

ROCm 7.2.1, Release, all three requested architectures compiled. GPU 0 only;
the existing Maya server stayed running. No clocks or other processes were
changed. These are synthetic decode measurements, not Strix Halo results.

| Kernel | Original us/call | New us/call | Original GB/s | New GB/s | Latency reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| IQ2_XXS gate/up, IQ2_S fixture | 92.572 | 65.607 | 373.80 | 527.43 | 29.1% |
| IQ2_S down | 57.806 | 37.570 | 371.86 | 572.15 | 35.0% |
| IQ2_XXS gate/up, IQ3_XXS fixture | 93.209 | 68.616 | 371.24 | 504.30 | 26.4% |
| IQ3_XXS down | 48.220 | 43.506 | 532.77 | 590.49 | 9.8% |
| IQ2_XXS gate/up including shared down, IQ2_S fixture | 101.049 | 77.944 | 410.54 | 532.23 | 22.9% |

The full parity sweep passed all 620 comparisons with byte-identical q8_1
outputs and bit-identical FP32 outputs: maximum absolute error and relative L2
were both zero. Cases cover 1/3/8 routed experts, null expert slots, CPU/shared
contributions, zero activations, unclamped SwiGLU, and all tuning variants for
types 16/18/22. Legacy dispatch parity also passed. All five requested tests
passed: `glm_ffn_parity --selftest`, `glm_layer_parity --selftest`,
`glm_model_test --selftest`, `iq1_s_parity`, and `hip_intrinsics`.
Runtime tests used gfx1100; gfx1151/gfx1201 code objects were built, and
gfx1201 continues to select the original kernels. No CUDA toolkit was available
for a CUDA build; CUDA retains its original kernel template signatures and
arithmetic.

Full-model measurements reported after the initial synthetic tests showed
decode increasing from 15.1 to 16.4 tok/s (+8.6%) on RX 7900 XT (gfx1100),
with bit-identical parity. The two-GPU full-model configuration improved decode
by 3.9%.

## Strix Halo 8060S results (gfx1151), 2026-10-08

The following hardware measurements were reported for the original kernels
versus the optimized dispatch in commit `6f0bdf6`, before restoring legacy
IQ3_XXS down on gfx1151:

| Kernel / fixture | Original us/call | Optimized us/call | Speedup |
| --- | ---: | ---: | ---: |
| `gate_up<16>`, gu16/down22 fixture | 203.9 | 181.4 | 1.124x |
| `down<22>` (IQ2_S) | 131.4 | 126.6 | 1.038x |
| `gate_up<16>`, gu16/down18 fixture | 207.7 | 183.3 | 1.133x |
| `down<18>` (IQ3_XXS) | 126.5 | 130.2 | 0.972x (regression) |

Full-model decode on gfx1151 increased from 17.8 to 18.0 tok/s with that
dispatch. Because optimized `down<18>` was slower, production now retains the
original IQ3_XXS down kernel on gfx1151 while keeping optimized `gate_up<16>`
and `down<22>`. gfx1100 keeps optimized `down<18>`. The full-model result above
predates this selective fallback; decode after the fallback has not yet been
measured. gfx1201 continues to use the original kernels and remains untuned.

## Further measurements on Strix Halo

Run the benchmark, `--shared`, and `--parity-only --sweep` on gfx1151 using the
same ROCm build settings. Then compare the resident-pool Maya-S decode trace
with the same prompt, token count and routing settings, once with
`MAYA_HIP_EXPERTS_LEGACY=1` and once without it. Set that variable on the
separate profiling process; these changes do not require touching a running
server.

Record per-kernel call counts and mean/median durations, total expert
milliseconds per token, decode tokens/s, and dense q14 matvec time as a control.
For the supplied trace, a 20% latency reduction corresponds to gate/up <=183 us
and IQ2_S down <=96 us. Check whether expert effective bandwidth rises from
~165 GB/s toward 204–216 GB/s (85–90% of the supplied 240 GB/s peak). The
discrete-card speedups do not establish that APU bandwidth result; use the
resident decode trace to confirm it. `--sweep` allows checking whether Strix
prefers different codebook placement or row geometry before changing dispatch.
