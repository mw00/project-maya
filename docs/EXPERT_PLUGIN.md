# External GLM expert executor (experimental)

A CUDA Maya engine can use a separately compiled HIP helper in the same process.
The helper owns a bounded expert cache and computes selected decode experts.
Optional readback feeds Maya's CUDA prefill pipeline; optional grouped execution
shares prefill work with the helper. Maya retains routing, attention, dense/shared
layers, sampling and serving. No model weights or routing probabilities change.

The current capability is Linux, one primary GPU, native GGUF GLM experts with
width 4096, intermediate width 2048 and at most eight routes. EXL3, Windows and
other model shapes are unsupported by this adapter. Unset the plugin to retain
Maya's existing execution path. Unsupported plugin configurations report errors;
they do not silently switch to another backend.

## Build

Build Maya normally with its CUDA backend. Build the executor separately from the
same source tree with both CUDA and HIP toolchains installed:

```sh
bash tools/build_expert_modules.sh /absolute/path/to/modules
```

Defaults target `gfx1151` and `sm_120`; override `HIPCC`, `NVCC`, `HIP_ARCH` and
`CUDA_ARCH` as needed. Other architectures need separate validation. The builder
produces `libmaya_expert_hip.so` and a CUDA counterpart for comparison. It compiles
Maya's existing kernels and the persistent executor, with hidden implementation
symbols and three exported C entry points. `--with-probe` adds the stateless
diagnostic entry point used by the [fixtures](../tests/mixed_runtime/README.md).
Keep fixture and production outputs in separate directories.

## Configuration

Set these in the engine environment (or Maya configuration's `env` object):

```sh
export STRATA_GLM_SPLIT=0
export STRATA_GLM_EXPERT_PLUGIN=/absolute/path/to/modules/libmaya_expert_hip.so
export STRATA_GLM_EXPERT_PLUGIN_DEVICE=0
export STRATA_GLM_EXPERT_PLUGIN_MIB=90112
```

The device ordinal belongs to the **helper runtime** and is independent of the
primary CUDA ordinal. The budget is required, includes scratch, and accepts
1–131072 MiB. The example 90112 MiB is for the tested 96 GiB UMA machine; choose a
budget within the actual device's available memory. Do not also configure
`STRATA_GLM_EXPERT_HELPER` or layer splitting. The primary GPU keeps Maya's native
automatic VRAM sizing; the adapter adds no primary pool limit or reserve rule.

| Optional setting | Default | Effect |
|---|---|---|
| `STRATA_GLM_EXPERT_PLUGIN_READ` | off | `1` reads helper-owned weights into the existing host landing ring for CUDA prefill. Requires the readback extension. |
| `STRATA_GLM_EXPERT_PLUGIN_PREFILL_SHARE` | `0` | Fraction cap on helper-computed routed expert rows per layer, accepted range `0..0.49`. Requires the row extension. |
| `STRATA_GLM_EXPERT_PLUGIN_PREFILL_MAX_ROWS` | `16` | Per-expert admission limit, clamped to 1–256 rows. Admitted work still executes in tiles of at most 16 rows. |
| `STRATA_GLM_EXPERT_PLUGIN_PRESTAGE` | `0` | Maximum helper-backed experts admitted per layer to Maya's next-layer prestager. Requires readback; existing prestage slots and prediction rules still apply. |
| `STRATA_GLM_HELPER_ZEROCOPY` | off for this executor | `1` enables runtime-owned mapped input/plan/result buffers on an integrated helper. Discrete devices, insufficient budget or unavailable mapping retain copies. Grouped rows are unchanged. |

For GPU-only expert computation, set `STRATA_GLM_CPU_LANE=0` and
`STRATA_GLM_PREFILL_CPU=0`. CPU orchestration and host staging still run. With
readback enabled and helper share zero, CUDA computes prefill while helper weights
remain resident for subsequent dual-device decode. Positive share selects
helper-owned, nonresident experts lightest first within the row limit, fraction
cap and existing result-buffer capacity. CUDA handles the remaining work.
The fraction is an admission cap, not a measured compute-time ratio.

The tested HengHa machine preset uses share `0.25`, max rows `64`, prestage `32`,
readback and mapped buffers enabled. These are fixture settings, not universal
defaults. Increasing helper work may increase layer latency. Leave profiling off
for throughput tests; `STRATA_GLM_PROF` enables executor stage timings.

## Ownership and completion

The versioned [C ABI](../include/strata/core/expert_executor.h) passes host data
and opaque contexts. CUDA and HIP own their allocations separately; GPU pointers
and runtime handles never cross the boundary. Weight reads use host staging,
not assumed peer access. Unowned experts retain Maya's RAM/disk fallback.
Readback counters distinguish helper reads from the existing staged-disk category.

Calls serialize per context and return only after completion. Slot generations
reject stale jobs; output publication follows finite-result checks. Runtime
failures poison the context. Destruction is exclusive with workers, consumes the
handle and reports cleanup failures. Model adaptation failures require reload;
request reset cannot revive partially modified ownership.

Grouped prefill reserves 4 MiB per format group (otherwise 256 KiB); its 3 MiB
row scratch is allocated lazily within that budget. Compute and readback can
interleave between tiles. Prestaging uses at most two maximum-sized expert blobs
of pinned host storage per active part, waits for upload completion before reuse,
and joins its worker before consumption. Read failures drain workers and fail the
request; partial data is never consumed. Native helper-local replacement remains;
primary/helper hot-expert exchange is outside this contribution.

## Validation limits

CUDA/HIP real-weight fixtures, mapped/copy parity, failure paths and GLM model
inference have been tested on the Strix Halo/RTX 5090 Ubuntu fixture. This does
not establish general model quality, Windows support, device-loss recovery or
performance on other hardware. Record source, module, dependency and loaded
runtime identities with hardware results; host tests alone are insufficient.

See the [GLM Maya-M benchmark and reproduction settings](../tests/mixed_runtime/BENCHMARK.md)
for the stock CUDA+CPU and CUDA+HIP measurements, including their different UMA
allocations and numerical-validation limits.
