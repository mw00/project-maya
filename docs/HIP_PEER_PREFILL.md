# Automatic HIP prompt transport selection

For an eligible tensor-parallel pair, startup compares the ordinary host-mapped reduction, exact-F32 AR2, and direct peer-VRAM reduction. It selects the lowest median GPU time at the configured full prefill-chunk size. The peer path preserves member-0-plus-member-1 F32 addition order; it changes transport, not routing or weight precision. A faster collective is not a guarantee of the same percentage improvement in whole-model prefill.

`MAYA_DS4_HIP_PEER_AR` is off by default (opt-in until it is measured on a two-GPU AMD PC). Set it in a serving config's `env`; native launches use `DS4_HIP_PEER_AR`. The choices are:

- `auto`: compare compatible transports once during model loading.
- `0` or unset: retain the existing transport, with no peer allocation or timing probe.
- `1`: force compatible peer transport without timing it against alternatives.

Both ranks must support HIP peer access, fine-grained device allocations and a queryable steady wall clock. IDs come from the tensor-parallel group; no card models, bus addresses or GPU numbers are hard-coded. Unsupported pairs retain host transport. Explicit `MAYA_DS4_AR2=1/2` and flow groups retain their chosen transport without calibration; explicit AR2 `0` excludes AR2 from the comparison. When AR2 is unset, only its exact-F32 mode (`1`), not its FP16 mode (`2`), is a candidate.

Only two-member groups using the batch prompt loader are eligible. Single-GPU execution performs no peer queries, allocation or timing. Groups of three or more, decode/verification windows, logits gathering and pipeline handoffs keep their existing paths. `MAYA_DS4_AR2_DECODE` remains independent.

Automatic mode logs `PEER_AR_AUTO` with GPU IDs, payload size, sample count, candidate median times, selected transport, calibration duration and completion status. An active peer pair also logs `exact peer prefill all-reduce enabled`; do not infer activation from the environment setting alone.

## Startup calibration

The probe uses private synthetic inputs, one warmup per candidate and up to six rotated timing rounds, requiring at least three samples per candidate. Every result is checked exactly before it can be selected. Flags and sequence state are reset before inference; model activations, weights and expert placement are untouched. Peer buffers are released if host or AR2 wins. The chosen transport stays fixed for that model instance.

The target is roughly 0.5–2 seconds of startup overhead, with a shared three-second submission/polling budget across all eligible groups. A current R9700-pair primitive test completed calibration in 237 ms at 4096 x 5120 floats (host 18.16 ms, AR2 7.14 ms, peer 3.46 ms per reduction). These are collective timings, not end-to-end model speeds. An incomplete, safely drained probe keeps the existing transport. Incorrect results or undrained work refuse model loading. GPU driver calls cannot be forcibly interrupted by this userspace budget; this is not a hard recovery deadline for driver hangs.

## Memory and failure handling

Scratch is allocated after expert placement, without removing experts to make room. Each rank needs two receive slots of `prefill_chunk * model_dimension * sizeof(float)`, plus flags and error records: approximately 80 MiB at chunk 2048/dimension 5120, or 160 MiB at chunk 4096. Automatic calibration additionally needs one private input buffer (80 MiB per rank at chunk 4096/dimension 5120), and initialization leaves at least 64 MiB free. If either rank's initialization fails before a probe launches, both temporary allocations are released and the group keeps its previous transport.

Calls on each rank's compute stream have matching shapes and sequence numbers. Two receive slots prevent a producer from overwriting data the other rank still needs. Per-block completion flags publish payload stores after a system fence. Every output uses the same rank-order `__fadd_rn` as the exact host transport. Host-sequence-based graph replay is not supported.

A missing peer has a two-second device deadline. It uses `wall_clock64()` calibrated by `hipDeviceAttributeWallClockRate`, rather than a GPU core-cycle counter: a previous R9700 run recorded a negative core-counter interval that became a spurious immediate timeout after unsigned conversion. An unsupported clock query leaves host transport selected.

Timeouts set a sticky error and retain the first diagnostic (rank, sequence, observed sequence, block, payload size and elapsed/budget ticks). Later queued peer calls drain. The model checks transport status after prompt workers finish, before accepting the request. A launch error, timeout or other prompt-worker failure with an active peer group invalidates the model; reset does not clear this state, and reload is required. The port uses upstream's coordinated chunk-boundary cancellation; cancellation followed by model reuse still needs current-base validation. This does not establish recovery from arbitrary GPU/driver hangs.

## Tests and historical measurements

Build a HIP/gfx1201 tree, then:

```sh
cmake --build build-hip --target maya-ds4 ds4_peer_allreduce_test -j 4
HIP_VISIBLE_DEVICES=0,1 ./build-hip/ds4_peer_allreduce_test
```

Choose the two visible IDs appropriate to the machine. The standalone test needs about 800 MiB spare VRAM per rank and 640 MiB pinned host memory. It returns 77 when the selected pair lacks peer access, a queryable clock or sufficient spare VRAM. It compares 256 chained variable-size collectives against the existing host transport, with guarded outputs, signed zero, subnormals, extreme finite values, sequence wraparound and payloads through 8192 x 5120 floats. It also checks automatic selection against all three transports, a zero-budget fallback, explicit exclusion of AR2, first-sequence exact reuse after calibration, invalid argument rejection, an absent peer with a 50 ms physical deadline, and 32 subsequent poisoned calls preserving the first error record. The test's reset helper is only for isolated cases; the runtime never resets a failed transport.

Current-base full-model startup checks passed in auto, disabled, explicit AR2=1, and single-GPU modes. Automatic selection chose peer in 222 ms; auto/off/AR2 completed the same 1,213-token prompt with identical short greedy text. Disabled, explicit-AR2 and single-GPU runs emitted no calibration or peer-enable line. This smoke check does not establish full-logit parity or broad model quality.

Earlier full-model tests on our experimental EXL3 stack measured **9.2332% lower warm 2048-token prefill latency**, with a 95% interval of 8.9991–9.4673% across six balanced blocks/24 trials. Median latency was 7456.793 ms with host transport versus 6758.643 ms with peer transport. Cold results were inconclusive and no decode speedup was established. Scratch stayed allocated in both arms, keeping expert placement fixed; cold/warm/repeated logits and 80 greedy tokens matched independent references exactly.

Those tests used an older experimental engine with additional scalar EXL3, host-packing, predecode, copy-wait and transfer-pruning changes. They are evidence for this technique, **not a current-upstream end-to-end measurement**. Source and raw results are recorded in [fork PR #15](https://github.com/LostVector/project-maya/pull/15), with timeout diagnostics and the steady-clock fix in [#35](https://github.com/LostVector/project-maya/pull/35) and [#38](https://github.com/LostVector/project-maya/pull/38). Current-base end-to-end performance, cancellation/reuse and model-level failure-injection checks remain separate validation work.
