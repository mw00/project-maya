# Opt-in RDNA4 EXL3 matrix prefill

On HIP gfx12 devices, `EXL3_WMMA_GFX12=1` enables matrix instructions for
`set_gate_up` and `set_down`. The default remains the existing portable path.
`EXL3_WMMA=0` overrides the new flag and disables matrix prefill.
`EXL3_WMMA_ROWS=16` (default) or `32` selects routed rows per block.

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

Matrix instructions group products and additions differently from scalar FP32
FMA. Full-model logits need not match the portable path bit for bit, and changed
logits can affect expert routing and generated answers. CPU-reference operator
tests do not establish full-model quality equivalence.

The previous experimental stack had full-model numerical differences. Its older
speed measurements do not establish an end-to-end gain for this current-upstream
port. Keep the path opt-in pending current-base full-model quality, stability,
and matched performance tests. No throughput improvement is claimed here.

## Tests

Build `maya-ds4` and `exl3_test` with HIP enabled and gfx1201 in
`CMAKE_HIP_ARCHITECTURES`. Select a device with `HIP_VISIBLE_DEVICES`.

```
# Existing full suite, portable default on gfx12:
./build-hip/exl3_test
# Focused grouped-expert cases, both matrix tile sizes:
EXL3_WMMA_GFX12=1 EXL3_WMMA_ROWS=16 ./build-hip/exl3_test --prefill-only
EXL3_WMMA_GFX12=1 EXL3_WMMA_ROWS=32 ./build-hip/exl3_test --prefill-only
# Explicit opt-out overrides the opt-in:
EXL3_WMMA_GFX12=1 EXL3_WMMA=0 ./build-hip/exl3_test --prefill-only
```

The focused suite compares against the scalar CPU reference with the existing
`3e-3` relative-RMS tolerance and worst-element/finite checks. It covers integer
K=1..8 with all three codebooks, supported half-integer codes, different down
bitrates, mixed expert layouts, empty experts, nonzero row offsets, row counts
around 16/32 boundaries, and output guards. Tolerances are not widened for WMMA.
