# Experimental MI50 support (gfx906)

MI50 support is experimental. This page records the build and validation work
for the current patch; **the current patch has not yet been validated on the
MI50 cards**. Historical results below apply only to their recorded source and
binary builds.

## Build

Use a ROCm 7 environment with HIP, hipBLAS, hipBLASLt when available, CMake,
Ninja, and a C++20 compiler. Configure for the MI50 architecture explicitly:

```sh
cmake -S . -B build-hip-mi50 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON \
  -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm \
  -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build-hip-mi50 --parallel 16
ctest --test-dir build-hip-mi50 --output-on-failure
```

The architecture uses the HIP compatibility layer's logical 32-lane mapping
over its 64-lane wavefront. The MI50 has no WMMA path; prompt attention uses
the portable kernel. Run the HIP intrinsic parity test and the full applicable
parity suite on the actual card before treating a build as validated.

## Model names

Use the published Maya model names, with the model family stated explicitly.
Maya-M and Maya-L refer here to the GLM-5.3-Flash quants. DeepSeek-V4.1-Flash
has Maya-S; it does not have Maya-M or Maya-L.

| Model | Family and published name | MI50 status for this patch |
|---|---|---|
| Maya-M | GLM-5.3-Flash Maya-M (IQ2_S) | Pending new-patch validation |
| Maya-L | GLM-5.3-Flash Maya-L (IQ3_S) | Pending new-patch validation |
| Maya-S | DeepSeek-V4.1-Flash-1017 Maya-S | Work in progress; not validated by this patch |

## Historical evidence (not validation of this patch)

- **CS-364:** Historical MI50 Maya-L serving samples on the then-current
  deployed build include 8K and long-context runs. These are operational
  observations, not results for this patch, and the campaign did not promote a
  change.
- **CS-358:** A single frozen game-generation request completed on the
  historical four-MI50 native wave64 lane using GLM-5.3-Flash Maya-M. The
  result establishes successful generation for that recorded lane only; it
  does not establish repeatable speed, broad quality, or validation of this
  patch.
- A separate historical gfx906 wave64 build record documents kernel/routing
  parity and an earlier Maya-M binary. Its source and patch chain differ from
  the current patch; see the MI50 wave64 build record and CS-358 result in the
  workspace evidence archive.

## New-patch validation record

| Check | Result |
|---|---|
| gfx906 configure and build | Pending |
| HIP intrinsic parity, including two logical warps | Pending |
| Full applicable parity suite on MI50 | Pending |
| Maya-M end-to-end run | Pending |
| Maya-L end-to-end run | Pending |
| DeepSeek-V4.1-Flash-1017 Maya-S end-to-end run | Outside this validation scope; work in progress |

Record the exact source revision, ROCm/compiler versions, GPU identity, test
results, model files, context, and engine command when these checks are run.
