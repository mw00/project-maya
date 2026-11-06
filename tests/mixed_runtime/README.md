# CUDA/HIP executor validation fixtures

For the integrated adapter and production module build, see
[External GLM expert executor](../../docs/EXPERT_PLUGIN.md). These fixtures test
individual boundaries; they do not replace full-model validation or benchmarking.

## Runtime coexistence probe

Linux-only feasibility fixture. This is not an inference backend, expert kernel
validation, concurrency benchmark or proof of zero-copy interoperation.

Compile the HIP module separately from the CUDA executable:

```sh
tests/mixed_runtime/build.sh /absolute/build-probe
/absolute/build-probe/maya_mixed_probe /absolute/build-probe/libmaya_hip_probe.so 0 0
```

The final arguments explicitly select CUDA and HIP device ordinals in their own
runtime namespaces. Check the actual enumerated hardware before running. Toolchain
and architecture overrides: HIPCC, NVCC, HIP_ARCH, CUDA_ARCH.

The probe applies CUDA +1, HIP *2, CUDA +3, using host staging and completion waits
between every leg. It never shares GPU pointers or runtime allocation ownership.
It checks 100 jobs including unaligned tails, exact results and boundary canaries,
and rejects unknown ABI versions and capacity overflow. The HIP module exposes
one uniquely named C entry point and uses hidden visibility for its implementation.

Record source hashes, compiler versions, binary hashes and ldd/runtime-loaded
libraries with each run. The expert and adapter gates below exercise the real kernels; passing
this arithmetic fixture alone does not establish inference correctness.

## Real expert execution gate

`build_expert_modules.sh OUTPUT_DIRECTORY` builds separate CUDA and HIP shared
libraries containing the existing `glm_fast.cu` kernels. Hidden symbols, export
maps and linker garbage collection retain only the selected host entry path.
`-z defs` rejects unresolved dependencies. This avoids linking two complete Maya
engines or exposing C++/runtime handles across the module boundary.

`expert_module.cu` exposes a deliberately narrow synchronous diagnostic operation:
4096-wide, 2048-intermediate GLM experts, 1–8 routes, native supported GGUF formats,
explicit gate/up/down byte layout. Every call uploads weights and allocates scratch;
this is unsuitable for throughput measurement or a production cache. No scheduler
or inference behavior changes. Cleanup is best-effort in the diagnostic wrapper;
a production context needs explicit error-reporting destruction and device restore.

`expert_compare.py --help` lists fixture arguments. It requires numpy and a GGUF
reader. Use captured normalized activations, expert IDs and routing weights. The
runner reads expert bytes directly from GGUF without converting quantization. The
clamp is explicitly 8.0 for this fixture; this does not establish complete model
configuration equivalence. It compares unchanged Maya CUDA kernels with HIP and
4+4 mixed execution, checks repeatability, weighted reduction, zero input/routes,
output canaries and rejected ABI/blob descriptors. CUDA vs HIP errors are reported
without inventing a model-quality tolerance. The reduction consistency tolerance
is rtol=1e-4, atol=1e-5 and was set before the first execution.

The first recorded run used four captured layer-3 token inputs. Expert outputs
are not a full-layer, full-model or performance result. The persistent gate below covers context lifetime, supported formats and
serialized concurrent callers. Model configuration, cancellation and hardware
failure recovery need separate evaluation.


## Persistent context gate (PB2)

`include/strata/core/expert_executor.h` and `src/core/expert_executor.cu` add a versioned opaque-context interface.
The same source builds separately for CUDA and HIP. Create accepts an explicit
device, quantization, tested shape, slot count and byte budget. A single aligned
GPU arena contains expert slots and reusable scratch. Upload publishes a new slot
generation only after its copy completes. Compute validates every referenced
generation, serializes scratch usage, and copies results to caller memory only
after successful device completion and finite-output checks. GPU runtime failures
poison the context; subsequent jobs reject it. Successful create/operations and
teardown explicitly restore the caller's runtime device.

Calls are synchronous and serialized per context. Destruction must be exclusive
with all callers: it consumes/nulls the handle even if cleanup fails, and reports
GPU cleanup errors. Destroying a null handle is harmless. Callers must neither
reuse consumed handles nor invoke destroy concurrently with active/queued calls.
The integrated adapter uses this experimental ABI from include/strata/core.
Shapes remain capability-restricted to the tested GLM case.

Add `--persistent` to expert_compare.py to compare every persistent result with the
stateless reference. persistent.py also verifies bounded allocation accounting,
no expert upload during compute, budget rejection, bad slot/blob rejection,
stale upload/job rejection, actual replacement, context isolation, concurrent
callers and checked/idempotent teardown. CUDA/ROCm hardware-failure recovery and
restoration between multiple devices of one backend remain untested. Concurrent
callers are serialized; this is not evidence of overlapping GPU execution.


## Maya adapter gate

The persistent ABI/module now live under include/strata/core and src/core;
context_abi.h is a compatibility include. plugin_experts.hpp/.cpp provide the
host loader, bounded layout-group planning, insert/replace ownership and executor
calls. GlmExpertHelper delegates to this adapter only when explicitly enabled.
See docs/EXPERT_PLUGIN.md for configuration and current limitations.

expert_compare.py --adapter /absolute/libadapter_probe.so exercises the adapter
against both previously validated executor modules. Compile adapter_probe.cpp
with src/core/plugin_experts.cpp, include/ and tests/mixed_runtime on the include
path, shared/PIC/hidden visibility and -ldl. The test checks planned ownership,
budget rejection, duplicate insertion, unowned routes, replacement parity and
checked teardown on real GGUF data. The full engine smoke is recorded separately;
passing this fixture alone does not establish serving or model integration.


## Prefill reader lifetime regression

Run `python3 tests/mixed_runtime/test_prefill_reader.py` on the host. It compiles
the production reader loop with fake asynchronous I/O in Release mode and checks
that skipped predicted reads cannot make a host slot reusable before completion.
GPU fixtures still need the target toolchains, NumPy and the GGUF reader.
