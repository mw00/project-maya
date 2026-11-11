# Pascal GPUs (experimental)

Tesla P100 (sm_60) and, by the reports in #189 and #212, Tesla P40 / GTX 10 series (sm_61). The setup offers them as
experimental (#212). This page covers the P100 kernels below; the kernel numbers in it were measured on v1.1.12, the
end-to-end ones at the end on v1.1.12, v1.1.44 and v1.1.58.

**What was checked.** The tree builds for sm_60 and sm_61 (CUDA 12.9). On a Tesla P100-PCIE-16GB the engine's test suite
runs: the kernel parity tests, the GLM layer and model tests against the oracle (`glm_model_test`), the multi-column GEMV
and the prompt MMQ test, and `pascal_moe_gemm_test` (the prompt products below against ggml's dequantizer). The
end-to-end rates are at the end. sm_61 was only compiled here; its run evidence is #189 and #212.

## Build

CUDA 12.x is required (CUDA 13 dropped Pascal). With CUDA 13 also installed, point CMake at the 12.x toolkit:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=60 -DSTRATA_NATIVE_EXPERTS=ON \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda-12.9
cmake --build build --target maya -j8
```

For sm_61 use `-DCMAKE_CUDA_ARCHITECTURES=61` (the build then adds `-Xptxas -dlcm=ca`, #214).

## What differs on Pascal

- **No `__dp4a` on sm_60** (it starts at sm_61): `include/strata/kernels/dp4a.hpp` gives it as four `vmad` (#261).
- **No `__nanosleep` below sm_70:** the host-flag waits spin instead.
- **No tensor cores:** the two wmma attention kernels compile to nothing below sm_70 and `mla_attn` takes the F32 kernel
  (the log says `prompt attention on the F32 cores`); the tensor-core prompt attention (`qsa_prompt_attn`) needs sm_80 and refuses.
- **Prompt GEMMs:** for FP16 x FP16 -> F32 on sm_60, cuBLAS's default algorithm is a legacy kernel. `ALGO6` is used from 1024
  tokens up (`include/strata/cuda_compat/pascal_gemm.hpp`): through `Gemm::f16` on a P100, 11.4 -> 9.4 ms for 2048 x 4096 x 4096,
  34.3 -> 28.6 ms for 2048 x 12288 x 4096 and 23.3 -> 19.1 ms for 8192 x 4096 x 2048 (about +20%), the same output bits. Below
  512 tokens ALGO6 is slower, so smaller products keep the default. bf16 and the batched calls keep it too (they refuse an
  explicit algo).
- **48 KB of shared memory a block:** the EXL3 several-activation kernels need 58 KB, so a verify window's tokens run one at
  a time on Pascal (the log says so). `exl3_test` therefore fails its window cases on sm_60.
- **sm_61 (P40, GTX 10):** global loads are cached in L2 only, which starves the small loads of the GEMVs; #214 adds
  `-Xptxas -dlcm=ca` for sm_6x (1.7-3x decode on P40s). The P100 (sm_60) caches them in L1 already: `glm_fast.cu` built
  for sm_60 with and without the flag gives the same SASS.

## Measured on a Tesla P100 (`glm_mv_bench`, 8192 x 4096, one column)

| type | size | time | bandwidth |
| --- | ---: | ---: | ---: |
| Q8_0 | 34.0 MB | 120 us | 283 GB/s |
| Q4_K | 18.0 MB | 86 us | 209 GB/s |
| Q6_K | 26.2 MB | 162 us | 162 GB/s |
| bf16 | 64.0 MB | 128 us | 500 GB/s |

The P100 has about 730 GB/s (a plain stream of the same bytes reads 550-590 GB/s here). The bf16 GEMV, which has no integer
dot, gets 500; the quantised ones sit at a third to a half of that.

**It is not the emulated `dp4a`.** A build with `__dp4a` replaced by a free add (wrong answers, timing only) runs the same
GEMV in the same time (Q4_K 88.2 against 84.5 us, Q6_K 154.8 against 156.4, Q8_0 121.9 against 119.7). What limits them is
the arithmetic around the dot and, on sm_60, the integer multiply itself: there is no 32-bit IMAD, the compiler emits XMADs
(16 x 16 -> 32 bits, about 3 for one 32-bit multiply). Maya's 16-byte-load pattern with the arithmetic removed reaches 490-560 GB/s, so
the loads are not the limit either.

`mv_q4k_pascal` (Q4_K, 1-4 columns, `STRATA_GLM_PASCAL_MV=0` turns it off) takes two weights and two activations through one
multiply (`(a0 << 16) + a2` against `w0 | w2 << 16`: the high half is `a0*w0 + a2*w2`; 8 of them stay under 2^15 in both halves,
so the result is the exact integer dot), stages the activations in shared memory in a bank-conflict-free layout, and unpacks the
eight sub-block scales without a branch. 8192 x 4096, one column: 84 -> 55 us (210 -> 330 GB/s); two columns 92.5 -> 80.5 us;
four columns are slower than the tiled kernels (190 against 165 us), and each column is bit for bit the one-column result.
The IQ2_XXS / IQ2_S / IQ3_XXS dots flip the signs of four weights with `(g ^ m) + (m & 1)` and `(x * 0x00204081) & 0x01010101`
instead of `__vcmpne4` + `__vsub4` (about 16 emulated ops each): the same bits, the routed gate/up kernel 113 -> 139 GB/s
(`moe_gate_up`, 6 experts), `moe_down` 106 -> 119 GB/s (IQ2_S).

#261 adds the Q6_K one-column GEMV (`MAYA_GLM_MV_P100`) and a sign table in shared memory for the routed experts' IQ3_XXS /
IQ3_S / IQ2_S kernels (in place of the dot above there; IQ2_XXS and the other kernels keep it). See docs/SETTINGS.md.
On the P100 the table is the faster of the two there (decode profile, Maya-S24: `moe_down` 25.4 against 28.0, `moe_gate_up`
47.3 against 48.0). `STRATA_GLM_GU_STAGE=0` makes `moe_gate_up` faster still (45.5); decode moves less than its noise.

## End to end on four Tesla P100 16 GB (PCIe 3 behind a PEX 8749 switch), Maya-S24, i5-14500, 62 GB DDR4

Layer split 13/11/11/10, 32K context, MTP drafts (82-93% accepted), `--bench`-style 256-token answers:

| | tokens/s |
| --- | ---: |
| decode (three questions) | 21.7 / 22.1 / 23.5 |
| prompt, 2K tokens, `--prefill auto` (one 8192 chunk, the cards run in turn) | 105 |
| prompt, 2K / 8K tokens, `--prefill 512` | 133 / 220 |
| prompt, 2K / 8K tokens, `--prefill 1024` | 131 / 257 |
| prompt, 4K tokens: `--prefill 1024` / 1536 / 2048 | 175 / 173 / 160 |
| prompt, 4K tokens, `--prefill 1024` with `STRATA_GLM_PREFILL_CPU=0` / `STRATA_GLM_PRESTAGE=96` / `STRATA_GLM_PREFILL_SUB=512` | 147 / 169 / 175 |

Decode is bound by the host: the RAM tier's experts (about a fifth of the routes, 2-3 a layer) go through the CPU lane (0.32 ms an
expert, the host's 26 GB/s DDR read bandwidth) or over PCIe (1.1 ms), and the GPU waits for them (`moe_cpu_wait` is the largest
single category after the idle time of the layer split). So GEMV speed buys little: the Q4_K kernel above, 1.5x faster in isolation
(`kda_proj_mv` -34% in a profiled run), gave 22.31 -> 23.52 tokens/s (+5.4%) over three interleaved A/B rounds (all nine paired
questions faster; one unpaired A/B pair differed by less than the +-5% run-to-run noise, so compare interleaved runs). CPU-lane
threads (8 or 11 equal, 14 slower), `STRATA_GLM_PREFETCH_N` and `RAM_RESIDENT` (stops with "454 of 544 slots": the tier is 33 GB of
the 37.3 needed) did not help. More free RAM and more memory bandwidth are what would. Prompts are bound the same way: a 1024-token chunk stages or computes about 100 non-resident experts a layer, the CPU
lane's multi-token dots run at roughly 0.26 TFLOP/s, and the three idle cards wait for the working one.

**On v1.1.44** (the same four cards, 32K context, 256-token greedy answers to three questions, then a 4096-token prompt; three
interleaved rounds against the v1.1.12 build): decode 24.8 -> 28.2 tokens/s (all nine paired answers faster, drafts accepted
the same), prompt 203 -> 209. With `STRATA_GLM_PASCAL_MV=0` decode is 6% slower (26.2 -> 24.6, `kda_proj_mv` +50%); with
`STRATA_GLM_PREFILL_PASCAL=0` (MMQ) the prompt is 8% slower (189 -> 174). This prompt (docs text) sends 600-900 experts a
1024-token chunk to each card's CPU lane, so it reads slower than the 4K figure above.

**On v1.1.58** (`main` at a358252 against the same commit with this page's kernels; the same four cards behind one PCIe 3.0 x4
uplink, 26 GB RAM tier, 131072 context, `--parallel 4`, `--prefill auto`, no other settings; three interleaved rounds, temperature 0.7,
thinking off): one conversation's decode 23.5 -> 26.5 tokens/s (nine 400-token answers each, +13%), with 5K tokens of context
20.7 -> 22.2; a 2.9K-token prompt's first token after 31.5 -> 20.7 s (93 -> 143 tokens/s), a 6.2K-token one 57.3 -> 43.1 s
(109 -> 146). Four conversations at once decode the same (37.0 against 37.4 tokens/s in total, one run a round).

## Tests that do not pass on sm_60

- `exl3_test`: the several-activation window cases (48 KB of shared memory, above).
- `kv_hybrid_parity`: its last case expects the sm_80 tensor-core prompt attention to take the pools; below sm_80 it refuses
  by design. The other four cases of the test pass.
- `ple_parity`, `glm_model_big_*`: need a model file and numpy; unrelated to the GPU.
- `cuda_device_selftest`: `device.cu` still asks for compute capability 7.5; only this diagnostic calls it.
- The DeepSeek-V4 sources (`src/ds4/ds4_gguf.cu`, `ds4_iq.cu`) do not build for sm_60, so build the `maya` target, not `all`.
