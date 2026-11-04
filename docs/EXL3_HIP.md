# Experimental EXL3 prefill on HIP

The portable HIP implementations of `exl3::set_gate_up`, `set_down`, and `gemm_rows` reuse the existing eight-activation `block_gemv` decoder. They replace the previous no-op HIP branches. They use FP32 FMA arithmetic, not AMD matrix instructions; there is room for further optimization. The FP16 input adapter is separate from the original decode input structure. CUDA keeps its existing tensor-core implementation.

## Current limits

The setup offers DeepSeek's EXL3 quants on AMD as experimental (since v1.1.30), as it offers GLM's and MiniMax's EXL3 quants there: `--backend hip --model DeepSeek-V4.1-Flash-1017 --ds4-quant exl3` (or `exl3-3.5`) downloads, packs and configures them with a warning. The kernels pass `exl3_test` on gfx1201 and gfx1151; the whole model has had a smoke test, not a full-model parity or quality evaluation, and Vision and DSpark were disabled in that trial. Maya's GGUF quants (`--model deepseek-maya-s` / `deepseek-maya-xs`) remain the tested way on AMD. The manual steps below still work for a direct engine launch.

The EXL3 3.5bpw model has integer per-matrix K (3/4 for routed experts, 6 for most dense matrices). Its average bitrate does not require half-integer matrix encodings. Keep the original 1017 Engram shards beside the EXL3 shards; packing indexes the weights without copying them.

```sh
python tools/ds4/pack.py --model /path/to/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw
# After building the HIP engine; this starts its raw token-ID serving protocol.
DS4_SPLIT=36,4 build-hip/maya-ds4 \
  --pack /path/to/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw/maya-ds4 \
  --gpus 0,1,2 --tp 2 --max-context 262144 --reserve-mib 2048 \
  --mmap-experts off --prefill-chunk 2048 --vision off --dspark none --serve
```

The measured trial also enabled `DS4_SKIP_UNPICKED=1` from separate PR #297; this kernel change does not include or require that optimization to compile.

## Validation

On gfx1201 / ROCm 7.14.1, `exl3_test` passes with zero failures and zero launch errors. HIP now runs the previously skipped CPU-reference `gemm_rows` and grouped-expert tests, including empty experts, multiple eight-row groups and partial tails, instead of silently skipping those cases. Existing decode/window exactness tests also pass. In an appropriately configured HIP build:

```sh
cmake --build build-hip --target exl3_test maya-ds4 -j
ctest --test-dir build-hip -R '^exl3_test$' --output-on-failure
python tools/check_hip_compat.py
```

Full-model smoke tests produced coherent answers with short and fresh 1K-token prompts. This is not a full-model parity or quality evaluation.

## Practical setup measurements

Hardware: Ryzen 9950X, 256 GB RAM, two Radeon AI PRO R9700 32 GB cards plus a 16 GB RX 9070-family card, ROCm 7.14.1. TP2 pair plus singleton, `DS4_SPLIT=36,4`, context 262144, explicit RAM experts, reserve 2048 MiB. DSpark/profiling/flow-TP off.

| Fresh prompt | Previous Q8 setup | EXL3 setup |
| --- | --- | --- |
| 1043 tokens A | 17.988 s | 8.063 s |
| 1043 tokens B | 16.698 s | 6.975 s |
| 2067 tokens C | 19.558 s | 10.654 s |

Same fixed synthetic prompts, one output token, temperature zero, reasoning none; every response reports `cache_n=0`. Repeated 256-token generation: 18.6 tok/s EXL3 versus 18.2 Q8 (first runs: 12.1 versus 13.4). These are individual measurements, not confidence intervals.

The Q8 setup used `abc09c7` plus PR #297's optimization; EXL3 used `8d9a9b5` plus that optimization and these kernels. Quant, expert profiles and default chunk sizes differ (Q8 8192, EXL3 2048). These compare usable setups, not an isolated quant-only or kernel-only A/B. Unmodified HIP has no valid grouped EXL3 prefill result to time.

EXL3 allocation: each R9700 has 3.02 GB dense weights and 26.85 GB experts; the third GPU has 1.04 GB dense weights and 12.40 GB experts. Host experts occupy 173.1 GB. Engram tables remain disk-backed.
