# Configurable scalar EXL3 prompt tiles on RDNA4

A prompt often routes several tokens to the same expert. Larger scalar tiles unpack each compressed weight fragment once for more of those tokens, trading register/shared-memory occupancy for less repeated unpacking. The implementation keeps the current DOT2 sum order, four-lane reduction, rotations and FP16 rounding. It allocates no expanded-weight cache or extra persistent expert VRAM.

`MAYA_EXL3_HIP_SCALAR_ROWS=0` (default, or `8`) retains the original implementation. `12`, `16` and `24` select a fixed tile. `auto` selects gate/up's original 8 rows below `MAYA_EXL3_HIP_SCALAR_GU_MIN_ROWS` (48), otherwise 12; down uses 12 below `MAYA_EXL3_HIP_SCALAR_DOWN_MIN_ROWS` (128), otherwise 24. Each choice applies to one dispatched expert batch, using its existing `SetArgs.max_rows`; it is not an extra per-expert readback or learned predictor.

The cutoffs are configurable because occupancy and routed-row distributions differ between cards, model shapes and workloads. Capability is checked per visible GPU. Only gfx12, uniform integer 3/4-bit layouts with codebook 2 (mul1), on the scalar set path are selected. Other layouts, mixed expert sets and architectures retain the existing implementation. Existing matrix-kernel selection still wins. Decode and MTP-verification kernels are not changed. Native launches accept `STRATA_EXL3_HIP_SCALAR_*` and `EXL3_HIP_SCALAR_*` aliases too.

Example serving-config environment (explicit opt-in):

```json
{
  "MAYA_EXL3_HIP_SCALAR_ROWS": "auto",
  "MAYA_EXL3_HIP_SCALAR_GU_MIN_ROWS": "48",
  "MAYA_EXL3_HIP_SCALAR_DOWN_MIN_ROWS": "128"
}
```

## Measurements, 2026-10-28

On three R9700s, TP2 on two cards and a shared 20 GB helper/four-bit DSpark on the third, a matched **warm 4096-token prefill** screen measured:

| Mode | Two warm observations (ms) | Mean (ms) | Latency reduction |
| --- | --- | --- | --- |
| Original 8 | 9331.26, 9339.47 | 9335.365 | baseline |
| Fixed 12 | 9118.02, 9148.37 | 9133.195 | 2.17% |
| Auto, cutoffs 48/128 | 9135.97, 9061.22 | 9098.595 | 2.54% |

One loaded model, same binary, fixed expert residency/profile, warm order 8/12/auto/auto/12/8 after parity passes 8/12/auto/8. Controls drifted 0.088%. All 48 captured full-vocabulary states across three prompts were exact, including repeat controls; only the 4K prompt exercised the new set kernels. Routing/payload statistics matched across all ten 4K passes, with no KFD eviction-counter increase during the warm bracket and no process swapping. DSpark weights were loaded but drafting was not timed. This is a small prefill result, not a decode, broad-workload or default-setting claim; two samples do not establish auto as significantly better than fixed 12.

That model integration used the immutable serving v1.1.29 model objects with only its EXL3 kernel object replaced. Its kernel baseline was the same current DOT2 eight-row arithmetic, **not** the earlier scalar-FMA16 experiment. The public implementation is ported onto v1.1.34 (`249f9a8`); capability/configuration handling is new in this port. Do not describe the model timing as a fresh full-model v1.1.34 benchmark.

A prior screen on the v1.1.34 kernel base compared 8/12/16/24/32 rows using archived real routed-count shapes and generated weights/gather indices. A simple policy was fitted on seven training batches and frozen before sixteen held-out batches were replayed with two gather patterns on each main GPU. Held-out sums of expert-stage times fell 16.8–17.2% for fixed 12 and 17.6–18.1% for auto. Those sums are not workload-frequency-weighted model throughput. All 504 checked operator cases matched exactly. Fixed 32 was about 18% slower on the held-out sets and required 80 bytes/thread of private memory in gate/up, so it is not exposed here. Fixed 24 sometimes improved down while regressing gate/up; a global bigger-is-better policy was rejected.

## Reproducing the public port

```sh
cmake --build build-hip --target maya-ds4 exl3_test exl3_hip_scalar_policy_test -j 4
./build-hip/exl3_hip_scalar_policy_test
HIP_VISIBLE_DEVICES=0 ./build-hip/exl3_test --hip-scalar
HIP_VISIBLE_DEVICES=0 ./build-hip/exl3_test --hip-scalar-bench
```

The policy test needs no GPU. The GPU test checks every integer bitrate/codebook, half-integer fallbacks, distinct gate/up/down formats, empty experts, nonzero row bases and tails around tile/threshold boundaries. Each public mode is compared with the original path byte for byte, including output guards; the original small cases retain the CPU-reference tolerance checks. The optional benchmark times combined gate/up and down at model-sized D5120/F1152, with synthetic weights and three expert slots (one empty), in balanced mode order. Its purpose is a small repeatable tuning screen, not a replacement for representative full-model measurements. Measure one GPU at a time on an idle engine.
