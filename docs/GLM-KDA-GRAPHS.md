# KDA layer graphs (GLM-5.3-Flash decode)

`MAYA_GLM_KDA_GRAPH` captures the recurrent layers of GLM-5.3-Flash's decode as
CUDA graphs and replays them token after token. Since v1.1.72 it is on by
default in NVIDIA builds (`0` turns it off; AMD builds keep direct launches
unless it is `1`; `MAYA_GLM_DEFER_CPU` keeps direct launches). The DSA layers'
attention stays direct, and their FFN side is captured too when
`MAYA_GLM_DSA_FFN_GRAPH` is on (its default). Speculative passes (the spec
gate), profiling, dumps, prefetch, lookahead, the DFlash2 drafter and a
request's `graphs=0` bypass capture. Expert selection, weights, quantization
and kernel arithmetic are unchanged; graph replay explicitly accounts for MoE
requests and residual-buffer swaps. CPU-plan/buffer-key changes (and a
request's `promote_min` / `miss_skip_*`) select their own captured graph. A
capture that fails falls back to direct launches for the engine's life.
Graphs are destroyed before their buffers.

## Checks for the default (v1.1.72)

- `tests/glm_pack_graph_test` on GLM-5.3-Flash Maya-S v2, one Tesla V100: PASS, 192
  full-vocabulary logit rows identical bit for bit with and without graphs (snapshots, reset), and 48
  steps across CPU-plan changes finite.
- Greedy answers (three prompts, `MAYA_GLM_REPEATABLE=1 MAYA_GLM_PREFILL_STABLE=1`, CPU lane off on the
  V100): the same tokens with graphs on and off, on the V100 (CUDA) and on a Radeon 8065S (HIP, `1`).
- Decode, default settings: +1% on one V100 that computes most of its RAM-tier experts on the CPU; no
  change on the Radeon 8065S, whose one-GPU MTP decode runs its verify windows without the graphs - AMD
  builds keep direct launches by default. #795 measured +8-9% on 3x V100 + RTX 3060.

## Measured thinking-Max screens (v1.0.21)

Windows, RTX 4090D 48 GB, Ryzen 5 3600, 128 GB DDR4-3200; unchanged
Maya-S-v2-IQ2_XXS weights, native FP16 cache, 262144 context capacity and six
CPU threads. Fixed 32755-token prompt, 128-token same-prompt warm-up, 2048
measured output tokens, temperature 1, top-p .95, seed 734621, MTP off.
Fresh engines and isolated copies of the same pre-test usage history; no
concurrent telemetry polling. Rates use native engine decode times.

| Order | Stock tok/s | KDA tok/s | Change |
|---|---:|---:|---:|
| Stock -> KDA | 30.7184 | 31.4429 | +2.36% |
| KDA -> stock | 30.7334 | 31.4516 | +2.34% |
| Arithmetic mean | 30.7259 | 31.4472 | +2.35% |

Both pairs have identical warm-up and measured streamed-output/reasoning hashes,
5292 VRAM / 8684 RAM slots, 32768 auto-prefill chunks and zero decode disk reads.
Mean decode time saved: 1.529 seconds per 2048 outputs. These are two repeats of
one seeded prompt, not a diverse quality benchmark or statistical confidence interval.

The control was v1.0.21 (`89a1336`), with only the common Windows environment-setter
build fix. The measured candidate was a sealed composition with **only KDA graphs
enabled**; its quota and growing-state switches were off. This PR contains neither
quota nor growing-state changes. Candidate/control executable SHA256:
`ce54343b9293456612d072863f80b49a6a7998e6167dd6d769f4ad08800ef647` /
`c059472e76f5876096f033645e945b95ad8d6f60c6b18963a87befaf6bdfdd69`.

This branch is rebased onto **v1.0.24 / e3660c2**. No v1.0.24 GPU, speed or
full native/quality result is claimed. The full v1.0.21 stock-oracle matrix was
deliberately interrupted after the synthetic gates to prioritize quick user-requested
screens; that interruption is not a numerical-parity pass or failure. Synthetic
graph checks passed on the rig, but current-main actual-pack and quality gates remain
pending and must be completed before merge or adoption. Historical v1.0.16 gains and the chart's progress
from a user-reported ~11 tok/s starting point are not this PR's isolated gain.

## Native validation

Build with `-DMAYA_BUILD_TESTS=ON`, then build `glm_layer_graph_test` and
`glm_pack_graph_test`. The first is a small
CUDA synthetic test; the second requires the real pack and GPU memory. Use an
isolated `MAYA_GLM_USAGE` and slot directory, MTP off, sufficient RAM tier and
the intended CPU lane. Do not run against live pack sidecars or alongside a
serving engine, and do not mistake compile-only checks for native execution.
The pack test compares the full vocabulary, not only argmax. Preserve any failed
gate; never turn a control's reset/re-prefill difference into a graph parity pass.
The actual-pack test first qualifies ordinary snapshot/reset repeats and refuses
loaded MTP; it does not attribute an unstable ordinary control to graph replay.
Check the final `glm KDA graphs:` line for nonzero captures and replays. Disable
profiling, dumps, prefetch and lookahead, which intentionally bypass capture.
No AMD GPU runtime validation is claimed by the CUDA-name mapping check.
