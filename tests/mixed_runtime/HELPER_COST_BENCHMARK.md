# GLM helper: cost-based prefill admission

Measured 2026-11-06 on Maya v1.1.55, upstream main
`c4effee67cb95fb4303720919b075fddf04b4e69`, with the CUDA+HIP contribution at
`bac40a6a1c5eca8b730d0597bdf6259d47118f10` plus the cost-admission implementation.
The committed inference files match the tested candidate exactly. This is a
static-versus-adaptive helper comparison, not a new stock-versus-hybrid benchmark.

## Hardware and method

Ryzen AI Max+ 395 / Radeon 8060S (gfx1151), 128 GB physical unified memory,
96 GiB BIOS UMA, RTX 5090 32 GB (sm_120) over PCIe 4.0 x4 OCuLink. Ubuntu 26.04.1,
CUDA 13.1 and system HIP 7.1.52801-9999; the existing HIP executor was retained.
Model: GLM-5.3-Flash Maya-M IQ2_S, the same pack/tokenizer and initial expert profile
as [the original benchmark](BENCHMARK.md), which lists the model hashes.

Unchanged native `maya.py --bench`: three greedy answers up to 256 tokens,
thinking off, native warmups and 2048/8192-token prefill. Context 9216, native
INT8 latent KV, speculation/MTP off. CUDA expert pool sizing stays automatic.
Control/adaptive/adaptive/control order, two runs per arm, diagnostic tracing off.
Separate traced diagnostics are excluded from the timing means.

| Metric, tokens/s | Static runs | Adaptive runs | Static mean | Adaptive mean | Change |
|---|---|---|---:|---:|---:|
| Decode | 45.6, 45.7 | 45.6, 45.6 | 45.65 | 45.60 | −0.1% |
| Prefill 2048 | 330, 331 | 382, 380 | 330.5 | 381.0 | +15.3% |
| Prefill 8192 | 970, 969 | 983, 987 | 969.5 | 985.0 | +1.6% |

All four timed runs and both diagnostics had 2746 CUDA expert slots and the same
initial profile hash. Sampled whole-GPU CUDA peak was 31566 MiB (30.83 GiB) in
every run, including desktop/runtime use, sampled every 500 ms. This is not an
exact instantaneous or process-only peak. Two repetitions are a small sample;
the modest 8K improvement is not a general speed guarantee.

## Reproduce

Build the CUDA engine and external HIP module as described in
[EXPERT_PLUGIN.md](../../docs/EXPERT_PLUGIN.md). For a paired setup, rebuild the
pair after updating source. Use the original benchmark's no-speculation config:
`--max-context 9216 --expert-cache auto --prefill auto --kv-streaming on
--suffix-draft 0`, and the same initial `--expert-profile` in each run. With the
plugin library/device configured, common environment settings are:

```sh
STRATA_GLM_EXPERT_PLUGIN_MIB=90112
STRATA_GLM_EXPERT_PLUGIN_READ=1
STRATA_GLM_EXPERT_PLUGIN_PREFILL_SHARE=0.25
STRATA_GLM_EXPERT_PLUGIN_PRESTAGE=32
STRATA_GLM_HELPER_ZEROCOPY=1
STRATA_GLM_HELPER_CHECK=1
STRATA_GLM_CPU_LANE=0
STRATA_GLM_PREFILL_CPU=0
STRATA_GLM_RAM_GB=0
STRATA_GLM_SPLIT=0
STRATA_GLM_DISK_READERS=2
STRATA_GLM_PREFILL_PRED_T=0
STRATA_GLM_NO_REUSE=1
STRATA_GLM_NO_MTP=1
STRATA_GLM_NO_SPEC=1
```

Static: `STRATA_GLM_HELPER_COST=0`,
`STRATA_GLM_EXPERT_PLUGIN_PREFILL_MAX_ROWS=64`.
Adaptive: `STRATA_GLM_HELPER_COST=1`,
`STRATA_GLM_EXPERT_PLUGIN_PREFILL_MAX_ROWS=128`.
Put the settings in each configuration's `env` object, then run:

```sh
python maya.py --bench --config /absolute/path/to/config.json
```

Do not set `STRATA_GLM_HELPER_COST_TRACE` or `STRATA_GLM_PROF` for timing. Copy the
starting expert profile afresh for every run; retain the native report and log
before the next run overwrites them. Verify actual CUDA slot counts and loaded
runtime identities. The row-share setting is a ceiling, not a compute-time split.

## Interpretation and numerical checks

The estimator uses recent end-to-end helper time per row and CUDA staging span
per MiB, including contention and queue effects. Three prompt-size buckets and
an EWMA avoid mixing small warmups with large prompts. Admission reserves 15%
headroom relative to estimated remaining staging time; that margin is a heuristic.
Existing share/result-buffer limits and selected expert routes are preserved.
The implementation reads existing events after the next routing synchronization
and adds no new GPU wait. It changes no expert kernel, ABI, decoding or cache
promotion policy.

At 2K, sampled layers assigned about 29% fewer rows to HIP, reducing host waits
for helper completion while staging MORE expert weights for CUDA. At 8K, the
helper took somewhat more rows, with slightly less staging. The objective is
completion time, not minimizing transferred bytes. Diagnostic host waits are not
isolated GPU idle time; event spans are not pure PCIe copy time. Captured layer
samples omit pending final layers and are not complete prompt counters.

Validation:

- Fresh CUDA engine build; source, static libraries, module and actual loaded
  CUDA/HIP runtime hashes checked. HIP 7.1 observed in every run.
- Release CPU cost-estimator and production reader-lifetime tests pass on Mac and
  Ubuntu. The committed CI runs these without a GPU.
- Two 1025-token FP16 probes compare sequential, batched and repeated prefill.
  A fixed 22 GiB CUDA pool is used ONLY for numerical placement matching;
  throughput tests use automatic sizing. Both arms pass the existing probe's
  sequential/prefill and repeat KL gate of 0.1.
- Cross-policy full-vocabulary endpoints contain 154880 values. Sequential decode
  logits are bitwise equal. Batch/repeat KL is 1.888906e-6 / 1.196549e-6, below the
  experiment's predeclared 1e-4 comparison gate, with the same top token 1140.
  Batched logits are NOT bitwise equal: maximum absolute differences are
  6.3585 / 1.8756 despite the small probability-distribution KL.
- New static-control token/batch/repeat endpoints exactly match the saved
  historical HIP 7.1 fixed-placement endpoints. All native INT8 benchmark runs
  exit successfully; this is not a full INT8 teacher-forced numerical comparison.

This remains opt-in. No new MTP/DFlash, Windows, other-model/device, long-context
quality or GPU-sanitizer qualification. It has not been compared against an
optimally chosen smaller fixed share; that remains useful follow-up work.

## Tested artifact identities

```text
src/core/glm_prefill.cu: 178e5ff11cfbb58d5b0bb3e0236b907f6812641ab93a5a61c530cf1d0e91bc97
include/strata/core/helper_prefill_cost.hpp: f5967d65cb1d85ad1ae1b39ea8d53b8b6311940e1b36edc69b88d99414524785
CUDA engine: e8d9d801ef929e5e9b0cdbe1657964714a2d79c1e79e946e0cd28dfdd1a1d668
HIP module: 1214887e81056312221f5039a7dd1e3c7e4b091b576769733c7bf23b46139a6e
```
