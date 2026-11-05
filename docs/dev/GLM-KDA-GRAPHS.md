# Experimental opt-in KDA layer graphs

`MAYA_GLM_KDA_GRAPH=1` captures recurrent layers of single-device GLM decode.
Default is the ordinary direct path. DSA layers remain direct. Split/MTP,
profiling, dumps, prefetch and lookahead bypass capture. Expert selection, weights,
quantization and kernel arithmetic are unchanged; graph replay explicitly accounts
for MoE requests and residual-buffer swaps. CPU-plan/buffer-key changes invalidate
the corresponding captured graph. Graphs are destroyed before their buffers.

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
