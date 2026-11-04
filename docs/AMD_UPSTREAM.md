# Shared AMD backport audit

Compared Maya v1.0.4 (`70e0746`) plus this branch's GLM HIP port with Strata
[`d5ea713`](https://github.com/Niko1221/Strata/tree/d5ea7133741e67743c0e886bb426c0ce8d69cf6c).
The publication fixes below were merged in v1.0.7 (#2). Broader backend
updates require their own GLM validation.

## Publication fixes applied

Strata [issue #697](https://github.com/Niko1221/Strata/issues/697) reports that
the mapped-memory doorbell store can remain invisible to the CPU until stream
synchronization on gfx1201. Upstream uses a volatile ring store and a HIP-only
`__threadfence_system()` after the store in both shared doorbell kernels.
Maya now has those changes. The earlier payload fences remain.

Maya's `glm_fast.cu` has a separate routing request ring. Its `rq->seq` is
already volatile, but also lacked a post-store fence. The same HIP-only fence
now follows that signal. This is an application of the same publication rule
to Maya's code, not a copied Qwen kernel.

Validation on native Linux / system ROCm 7.2.1:

| Check | RX 7900 XT, gfx1100 | AI PRO R9700, gfx1201 |
| --- | --- | --- |
| Shared handoff: 100 CPU waits, 100 copy/rings, 100 fused publishes | Passed | Passed |
| Actual GLM routing graph: 100 disk requests and 100 CPU-lane requests | Passed | Passed |
| Same multi-architecture engine/GPU suite | 14/14 passed | 14/14 passed |
| 321B model request | Not verified | Not run |

Both cards now use the same `gfx1100;gfx1201;gfx1151` build. HC1/HC2 pass on
the R9700; `gfx1151` builds, untested. The installer enables the two discrete
architectures. These checks do not establish full-model correctness or speed.
See [AMD_RDNA4.md](AMD_RDNA4.md) for build and validation commands.

## Further integration requirements

| Shared component | Upstream change and integration requirement |
| --- | --- |
| `cmake/hip_backend.cmake` | Applied: upstream architecture lists, semicolon/space parsing, and compiled architecture metadata. Normalization also removes empty entries and duplicates before compiler use. |
| `src/core/device.cu` and `include/strata/core/device.hpp` | Applied: exact base-architecture checks against the compiled list, wave32 check, and actual/build architecture diagnostics. Broader upstream device-list helpers are separate. |
| `include/strata/hip_compat/` | Use the upstream headers as the base, retaining Maya's GLM mappings for occupancy, stream legacy, profiling, SGEMM/batched SGEMM, error codes and `__grid_constant__`. Replacing them without those additions breaks GLM compilation. |
| `intrinsics.hpp` | Applied: upstream RDNA4 dot4 guard, byte permutation, SWAR byte operations and older-HIP warp synchronization. Packed-byte and GLM reference checks pass on both discrete cards. |
| AMD setup detection | Applied: gfx1201 eligibility and fallback name; Maya uses the shared supported-architecture list. Unified-memory detection and wider installer helper consolidation are separate. |
| hipBLASLt table loader and tables | Added upstream gfx1201 tables for versions 100202/100500. The loader keeps its architecture/version checks. Maya's GLM prompt projections now reuse that loader with separate per-architecture GLM tables. |

The runtime query `hipblasLtGetVersion` on this host returns `100202`, or
**1.2.2**, and the installed header agrees. This is the value the loader
compares, regardless of a package's version label. Upstream includes both
`gfx1100-hipblaslt-100202.txt` and `gfx1201-hipblaslt-100202.txt`.

Maya's GLM dense prompt products in `src/core/glm_prefill.cu` still use SGEMM
and batched SGEMM through hipBLAS where appropriate. Its FP16 prompt
projections now reuse the inherited hipBLASLt loader in `src/prefill/gemm.cu`
when `MAYA_HIPBLASLT_TUNING` names a matching GLM table. Shapes absent from
the table fall back to GEMM Ex through hipBLAS, and the quantized expert path
continues to use GGML MMQ. The GLM tables were tuned and checked separately for
gfx1100, gfx1151 and gfx1201; the inherited Qwen tables are not interchangeable
with them.

## Compiler and model boundaries

Strata [issue #1180](https://github.com/Niko1221/Strata/issues/1180) concerns
its fused native expert prompt kernel and `waves_per_eu(8)`. That source is
absent from Maya, and no such setting was found in Maya's source tree. It is
not a reason to disable Maya's different GLM kernels without evidence.
The synthetic GPU checks do not establish full-model quantized-expert
or draft verification correctness. The first real request remains necessary.

Strata [issue #1389](https://github.com/Niko1221/Strata/issues/1389) measures
Qwen prompt performance. Its profiling and tuning method is useful once GLM
runs; its token rates are not Maya benchmarks.
