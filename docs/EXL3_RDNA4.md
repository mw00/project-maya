# RDNA4 EXL3 matrix prefill

On HIP gfx12 devices, grouped EXL3 prefill can use matrix instructions for
`set_gate_up` and `set_down`: opt-in for now with `EXL3_WMMA_GFX12=1` (unset or `0`
keeps the portable path on gfx12); `EXL3_WMMA=0` disables matrix prefill on both gfx11 and gfx12.
`EXL3_WMMA_ROWS=16` (default) or `32` selects routed rows per block.

Earlier DeepSeek tests on three R9700s measured about 8–9% higher warm 4K
prefill throughput and 12% lower warm 16K prefill latency with this path.
Those whole-model measurements used an earlier integration;
they are not a speed guarantee for every model or machine.

## Guided setup

Use this with an EXL3 model and a HIP build containing this implementation,
compiled for the selected RDNA4 GPU (`gfx1201` for the R9700 / RX 9070 family).
It does not accelerate GGUF expert weights. A practical starting point is
16 routed rows per block; no particular GPU count or tensor-parallel layout
is required.

Turn it on with `EXL3_WMMA_GFX12=1`; 16 rows is the default tile. The explicit
settings below do that.

First check the GPUs with `./maya.sh --backend hip --check`. Stop any running
server before setting up again or editing its configuration. To save the
settings through setup, for example with DeepSeek's 3.5-bit EXL3 quant:

```sh
./maya.sh --backend hip --setup \
  --model DeepSeek-V4.1-Flash-1017 --ds4-quant exl3-3.5 \
  --env EXL3_WMMA_GFX12=1 --env EXL3_WMMA_ROWS=16 --env EXL3_WMMA=1 \
  --no-start
```

Keep your existing model-directory, GPU and context options when repeating
setup; choose your installed EXL3 quant if it differs from this example.
The example uses the normal setup workflow, which may build or download files;
it saves the configuration without starting the server. For an existing
installation, merging these entries into its configuration's `env` is enough:

```json
{
  "EXL3_WMMA_GFX12": "1",
  "EXL3_WMMA_ROWS": "16",
  "EXL3_WMMA": "1"
}
```

Preserve the other `env` entries. Use these exact `EXL3_*` names: the kernel
library reads them directly. The same names can be exported for a native
`maya-ds4` launch. They must be set before starting the engine.

### Verify and compare

1. Confirm setup selected a HIP build for your RDNA4 GPU. An older binary
   without this implementation cannot run the RDNA4 matrix path.
2. Check that the saved configuration contains the entries above, then start
   that configuration: `./maya.sh --config /path/to/maya-ds4-hip.json`.
   Use the actual config path printed by setup. A shell variable alone does
   not override a conflicting value already saved in the configuration.
3. To check kernel execution independently, build `exl3_test` and run the
   focused matrix tests below on an idle GPU. A saved flag alone is not proof
   that a request exercised the grouped EXL3 prefill path; short prompts may
   use the existing small-window kernels.
4. Compare representative long prompts with the same model, context, cache
   conditions and other settings. Record prefill and decode separately.
   For a portable-path control, set `EXL3_WMMA=0` in the same configuration
   and restart; keep the other settings fixed. Restore it to `1` and restart
   to re-enable matrix prefill. An environment change does not reconfigure
   an engine that is already loaded.

See [Setup](SETUP.md) for the installation workflow and
[Settings](SETTINGS.md#rdna4-exl3-matrix-prefill) for the switches.

## Implementation

The implementation decodes compressed weight tiles directly into matrix operand
registers, reusing each tile across the routed rows. It does not reconstruct the
whole expert or require a predecoded-weight cache. Activation rows use eight
bytes of shared-memory padding and two aligned 64-bit loads per lane operand.
The existing grouped-expert wrappers handle routing, channel scales, Hadamard
transforms, and clamped SwiGLU. Integer and supported half-integer bitrates use
upstream's K2 representation (twice the bitrate).

RDNA4 uses eight FP16 operand elements per lane and a different accumulator
layout from RDNA3. The existing gfx11 path, CUDA dispatch, and decode/verify paths
are retained. This is a focused port of the compressed matrix implementation
from fork experiment #28, not a port of the entire experimental engine stack.
The separate FMA16 and predecoded-fragment-cache experiments are excluded.

## Numerical and performance limits

Both RDNA4 paths use FP16 operands and FP32 accumulation. Matrix instructions
group products and additions differently from the portable DOT2 path.
The weights and operand formats stay the same. Different accumulation grouping
and rounding produce numerical differences that can affect later expert routing
and generated answers; bit-identical full-model logits are not required.
CPU-reference operator tests check numerical accuracy, while the reported
full-model logit differences do not by themselves measure answer-quality loss.

The previous experimental stack had full-model numerical differences. Its older
speed measurements do not establish an end-to-end gain for this current-upstream
port. Broader full-model quality, stability and matched performance evaluation
remain useful follow-ups. One earlier normal-cache/MTP comparison also
observed 3.24% slower decode despite faster prefill; generated continuations
and cache behavior differed. Measure both stages when choosing settings.

## Tests

Build `maya-ds4` and `exl3_test` with HIP enabled and gfx1201 in
`CMAKE_HIP_ARCHITECTURES`. Select a device with `HIP_VISIBLE_DEVICES`.
If normal setup built the engine with tests disabled, enable the test targets
in that existing HIP build directory first:

```sh
cmake -S . -B build-hip -DSTRATA_BUILD_TESTS=ON
cmake --build build-hip --target exl3_test -j 2
```

```sh
# Full suite, RDNA4 matrix path:
EXL3_WMMA_GFX12=1 ./build-hip/exl3_test
# Portable control (the default):
./build-hip/exl3_test
# Focused grouped-expert cases, both matrix tile sizes:
EXL3_WMMA_GFX12=1 EXL3_WMMA_ROWS=16 ./build-hip/exl3_test --prefill-only
EXL3_WMMA_GFX12=1 EXL3_WMMA_ROWS=32 ./build-hip/exl3_test --prefill-only
# Explicit global opt-out overrides forced RDNA4 enable:
EXL3_WMMA_GFX12=1 EXL3_WMMA=0 ./build-hip/exl3_test --prefill-only
```

The test prints `grouped matrix prefill: enabled` or `disabled` to verify
the selected dispatch in that test process.

The focused suite compares against the scalar CPU reference with the existing
`3e-3` relative-RMS tolerance and worst-element/finite checks. It covers integer
K=1..8 with all three codebooks, supported half-integer codes, different down
bitrates, mixed expert layouts, empty experts, nonzero row offsets, row counts
around 16/32 boundaries, and output guards. Tolerances are not widened for WMMA.
