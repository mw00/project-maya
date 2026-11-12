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

Upstream's native helper EXL3 and `MAYA_GLM_HELPER_OVERLAP=1` paths are preserved.
The external executor does not implement their asynchronous enqueue/wait API;
requesting native helper overlap with the plugin fails explicitly. Leave that
setting unset for CUDA+HIP. This is separate from the plugin's grouped prefill
overlap, which remains supported.

## Paired setup and launcher

The normal installer already separates CUDA (`build/`) and HIP (`build-hip/`)
engines. A CUDA+HIP configuration needs a CUDA engine plus the external HIP
executor, not two full engines. On **Ubuntu 26.04 x86_64**, start the guided flow
with one command from a terminal:

```sh
./setup-cuda-hip.sh
```

It offers to install missing Ubuntu build packages (`sudo` may ask for your
password), creates Maya's private Python environment, installs Python/CMake/Ninja
dependencies through Maya's existing helper, and asks for a CUDA GLM config,
HIP device and memory budget. If no config exists, it invokes normal Maya model
setup first; choose a native GGUF GLM model. That first model preparation also
creates the ordinary CUDA build; the paired setup then builds its isolated copy.
Model downloads retain Maya's normal confirmation. Existing models are reused.
It builds the CUDA engine, HIP executor and CUDA comparison executor sequentially,
then generates the paired config/launcher and starts the dashboard.

Preview missing system packages without installing anything:

```sh
./setup-cuda-hip.sh --deps-plan
```

For a scripted installation with an existing model config:

```sh
./setup-cuda-hip.sh --install-deps --yes --config maya-my-glm.json \
    --hip-device 0 --helper-mib 90112 --no-start
```

`--yes` accepts dependency installation, not sudo authentication or model
downloads. Automatic system dependencies are limited to Ubuntu 26.04 x86_64:
`build-essential`, Python/venv, Git, pkg-config, CMake, Ninja,
`cuda-toolkit-13-1`, `hipcc` and `libamdhip64-dev`. Already installed packages
are preserved. The configured Ubuntu repositories must provide these packages;
the script does not add repositories or install/update GPU drivers. Working
NVIDIA and AMD drivers, device permissions and any required reboot come first.
A package plan that removes packages or changes graphics drivers stops before
installation. Other Linux systems can use preinstalled tools and omit
`--install-deps`; `--interactive` enables prompts independently.

The original explicit workflow remains available:

```sh
./maya.sh --backend cuda --setup --no-start
./setup-cuda-hip.sh --list-devices
./setup-cuda-hip.sh --config maya-my-glm.json --hip-device 0 --helper-mib 90112
```

Replace `maya-my-glm.json` with the configuration printed by normal setup.
The example budget requires enough HIP runtime memory (the performance fixture
used 96 GiB UMA). BIOS reservation alone does not determine HIP's allocation
limit: a historical 4 GiB BIOS boot reported about 59.6 GiB to HIP, still too little
for this 90112 MiB example. Stop other GPU workloads and choose a budget within
the helper's actual free memory. `--list-devices` shows NVIDIA indices separately
from **HIP runtime ordinals**; `--hip-device` uses the latter, without inherited
HIP/ROCR visibility filters. The primary defaults to the input config's one CUDA
GPU; `--cuda-device` explicitly changes it.

The paired installer reuses Maya's CUDA compiler setup and the executor builder.
It requires compatible CUDA `nvcc`, ROCm 7+ `hipcc`, g++ and CMake 3.24+.
Compiler discovery reuses Maya's CUDA version selection and finds HIP through
PATH, `ROCM_PATH` or `/opt/rocm`. It uses the existing pinned
llama.cpp checkout, downloading it through Maya's normal helper if missing.
`--ggml-dir` can name an existing checkout at the installer's pinned revision.
`--nvcc` and `--hipcc` override toolchain discovery. The HIP compilation target
is detected for the selected runtime device: its PCI identity is matched to the
kernel's KFD/DRM topology, independent of device enumeration order. Both sysfs
views must be accessible. `--list-devices` shows the architecture; an explicit
`--hip-arch` must match it, including during `--check`. Unknown targets and
`HSA_OVERRIDE_GFX_VERSION` overrides are rejected. The current WMMA build targets
are `gfx1100`, `gfx1151` and `gfx1201`; only `gfx1151` has hardware validation here.

Preflight reads the header of the source GGUF named by `native_experts.txt`, using
Maya's existing reader. It requires explicit GLM geometry: width 4096, intermediate
width 2048 and 1..8 routed experts, accepting both GLM metadata spellings and
scalar or array values within these limits. Missing, malformed or incompatible geometry is
rejected before compilation. Launch validation checks geometry again and matches
the visible HIP device to the pair's recorded compilation target. These checks
do not replace the engine's model loader or establish full artifact integrity.

Each successful run produces:

- `build-cuda-hip/pair-*/engine/maya` and its own CMake cache/build stamp;
- separately named HIP/CUDA executor objects and libraries under that pair's
  `modules/` directory, plus a source/artifact identity manifest;
- `maya-<model>-cuda-hip.json` and `run-maya-<model>-cuda-hip.sh` beside `maya.sh`.

Ordinary builds/configurations and previous pairs are preserved. A failed build
never replaces the existing paired configuration. Configuration publication is
atomic; running processes keep their old immutable pair. Old pair directories
are retained for rollback; remove an unused one only after its processes stop.
The production and diagnostic module builds should still use different output
directories when running the low-level builder manually.

Setup starts the normal Maya dashboard after success. `--no-start` only builds
and writes the configuration; `--check` validates the input, tools and available
HIP memory without building or changing files. The generated launcher supports
normal Maya arguments, including `--bench` and `--report`:

```sh
./run-maya-my-glm-cuda-hip.sh
./run-maya-my-glm-cuda-hip.sh --bench
./run-maya-my-glm-cuda-hip.sh --report
```

The new config retains model/drafter, sampling and serving choices. It selects
GPU-only expert computation, helper readback, mapped buffers, up to 64 admitted
rows per expert, and 32 prestaged experts; `--prefill-share` defaults to the tested
0.25 row cap and accepts 0..0.49. Set it to zero for CUDA-only prefill and dual-device
decode. The primary CUDA pool keeps its existing settings and native auto-sizing
when the input config uses `--expert-cache auto`. These are fixture defaults,
not a claim that this split is optimal for every GPU pair.

The Models page labels the configuration **CUDA + HIP**; its GPU selector still
refers to the CUDA primary. Change device assignments through paired setup. The
launcher and server verify the source, executable and helper identities, then
check visible HIP memory before starting the model. Switching models rejects a
stale pair before stopping the current model. Reports identify the HIP helper
budget separately; UMA is shared physical RAM, not additional system memory.
The existing live GPU monitor still charts the primary CUDA GPU only.

After pulling source updates, rerun `setup-cuda-hip.sh` to rebuild the matched
pair. Ordinary engine refresh cannot replace just its CUDA half. Automatic
CPU-lane and drafter calibration is skipped for this configuration; explicit
`--speed-calibration` can measure the retained drafter choice. The generated
workflow currently supports one CUDA primary and one conversation; unsupported
model families, layer splitting and parallel settings are rejected.

## Manual build

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
export MAYA_GLM_SPLIT=0
export MAYA_GLM_EXPERT_PLUGIN=/absolute/path/to/modules/libmaya_expert_hip.so
export MAYA_GLM_EXPERT_PLUGIN_DEVICE=0
export MAYA_GLM_EXPERT_PLUGIN_MIB=90112
```

The device ordinal belongs to the **helper runtime** and is independent of the
primary CUDA ordinal. The budget is required, includes scratch, and accepts
1–131072 MiB. The example 90112 MiB is for the tested 96 GiB UMA machine; choose a
budget within the actual device's available memory. Do not also configure
`MAYA_GLM_EXPERT_HELPER` or layer splitting. The primary GPU keeps Maya's native
automatic VRAM sizing; the adapter adds no primary pool limit or reserve rule.

| Optional setting | Default | Effect |
|---|---|---|
| `MAYA_GLM_EXPERT_PLUGIN_READ` | off | `1` reads helper-owned weights into the existing host landing ring for CUDA prefill. Requires the readback extension. |
| `MAYA_GLM_EXPERT_PLUGIN_PREFILL_SHARE` | `0` | Fraction cap on helper-computed routed expert rows per layer, accepted range `0..0.49`. Requires the row extension. |
| `MAYA_GLM_EXPERT_PLUGIN_PREFILL_MAX_ROWS` | `16` | Per-expert admission limit, clamped to 1–256 rows. Admitted work still executes in tiles of at most 16 rows. |
| `MAYA_GLM_HELPER_COST` | `0` | Adapt prefill admission to observed helper and CUDA staging costs; requires positive helper share. |
| `MAYA_GLM_HELPER_COST_TRACE` | `0` | Log per-layer costs, with either static or adaptive admission. Diagnostic only. |
| `MAYA_GLM_EXPERT_PLUGIN_PRESTAGE` | `0` | Maximum helper-backed experts admitted per layer to Maya's next-layer prestager. Requires readback; existing prestage slots and prediction rules still apply. |
| `MAYA_GLM_HELPER_ZEROCOPY` | off for this executor | `1` enables runtime-owned mapped input/plan/result buffers on an integrated helper. Discrete devices, insufficient budget or unavailable mapping retain copies. Grouped rows are unchanged. |

For GPU-only expert computation, set `MAYA_GLM_CPU_LANE=0` and
`MAYA_GLM_PREFILL_CPU=0`. CPU orchestration and host staging still run. With
readback enabled and helper share zero, CUDA computes prefill while helper weights
remain resident for subsequent dual-device decode. Positive share selects
helper-owned, nonresident experts lightest first within the row limit, fraction
cap and existing result-buffer capacity. CUDA handles the remaining work.
The fraction is an admission cap, not a measured compute-time ratio.

The tested HengHa machine preset uses share `0.25`, max rows `64`, prestage `32`,
readback and mapped buffers enabled. These are fixture settings, not universal
defaults. Increasing helper work may increase layer latency. Leave profiling off
for throughput tests; `MAYA_GLM_PROF` enables executor stage timings.

### Cost-based prefill admission

For an existing CUDA+HIP setup with readback and grouped prefill enabled, the
tested adaptive configuration changes only these two environment values:

```sh
export MAYA_GLM_HELPER_COST=1
export MAYA_GLM_EXPERT_PLUGIN_PREFILL_MAX_ROWS=128
```

In a Maya JSON configuration, put these values in its `env` object. Keep the
helper share at `0.25` for reproducing the reported test. Existing generated
pairs need rebuilding after updating the source, as for other inference changes.
The ordinary setup defaults and the static machine preset remain unchanged.

The scheduler learns helper worker time per row and CUDA plan-to-staging-end
time per nonresident MiB, separately for small, medium and large prompts. It
admits the lightest eligible experts while their estimated total helper time
fits within 85% of the remaining estimated staging time. This margin and the
estimates are heuristics, not isolated kernel or PCIe measurements. Calibration
uses existing events after the next required routing synchronization, without
adding a GPU wait. Before calibration, the admission limit is the smaller of
the configured limit and 64 rows; prompts shorter than 64 tokens use that same
fallback. The existing share ceiling, result-buffer limits and routing remain.

On the tested Maya-M fixture, this reduced HIP work at 2K and moved more expert
computation to CUDA, improving prefill despite transferring more weights. Two
native benchmark runs per arm measured +15.3% at 2K and +1.6% at 8K, with
effectively unchanged decode. This policy remains opt-in; MTP/DFlash, other
hardware and long-context quality were not qualified with it. See the
[cost-admission benchmark](../tests/mixed_runtime/HELPER_COST_BENCHMARK.md) for
settings, numerical differences and limits. Set `MAYA_GLM_HELPER_COST=0` and
restore the previous row limit to reproduce the static control.

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
