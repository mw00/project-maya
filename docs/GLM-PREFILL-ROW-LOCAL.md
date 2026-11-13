# Experimental NUMA row-local prefill scheduling

`MAYA_GLM_PREFILL_ROW_LOCAL=1` makes eligible GGML CPU prefill row blocks prefer
the NUMA half of their worker before stealing work from the other half. It is
off by default. This is an experimental scheduling change: the historical
combined-policy measurements below do not establish this switch's speed benefit
or whole-model answer quality. Production remains on the original engine.

## Scope

The existing `MAYA_GLM_ROW_HALVES=1` allocator places each RAM expert matrix's
low and high row halves on two NUMA nodes. Multi-token GGML prefill previously
used the ordinary shared job queue for those weights. This change records the
successful allocation's layout and nodes in its RAM class and uses two atomic
job queues for eligible gate/up and down batches.

Eligibility requires GLM-5 Next, more than one token, one registered unsplit RAM
mapping, matching matrix byte layout, aligned row-half boundaries, and every CPU
expert at a slot start in that mapping. Missing placement or any failed check
keeps the original queue. Non-Linux allocations do not establish this placement.
Enable both variables before engine startup:

```sh
export MAYA_GLM_ROW_HALVES=1
export MAYA_GLM_PREFILL_ROW_LOCAL=1
```

The original flat job IDs, 32-row gate/up blocks, 64-row down blocks, arithmetic,
quantization, two down batches and upload ordering are retained. Single-token
decode and EXL3 computation are unchanged. A one-time eligibility log is not
proof of physical page residency or of every later job's locality. The switch
can change timings and thus the adaptive CPU/GPU dispatch; CPU fixture parity
does not imply identical generated answers.

## Tests

The host-only fixture checks placement and slot guards, local-first order,
cross-half stealing, once-only job coverage, fallback order, split down ranges,
and concurrent independent invocations. Run without a GPU:

```sh
c++ -std=c++20 -O2 -DNDEBUG -Wall -Wextra -Werror -pthread -Isrc/core \
  tests/glm_prefill_row_local_test.cpp -o /tmp/glm_prefill_row_local_test
/tmp/glm_prefill_row_local_test
```

Linux CI is configured to run that host fixture. With `STRATA_BUILD_TESTS=ON`, CMake registers
`glm_prefill_row_local_test`; x86 builds with `STRATA_NATIVE_EXPERTS=ON` also link
the real CPU kernels and enable the numerical cases. Build that target and run
`ctest --test-dir <build> -R '^glm_prefill_row_local_test$' --output-on-failure`.

Historical Linux CPU tests passed bitwise comparisons of intermediate `c_ff`,
`c_hq` and final `out_h` for Q8_0, Q5_K, packed Q2_K (AVX-512), IQ3_S gate/up with
Q5_K down, and IQ3_S gate/up with IQ4_XS down. They also check sentinels, finite
outputs and disjoint writes between the two down batches. Numerical coverage
uses synthetic 256-by-256 matrices and a synchronous fixture pool; it does not
exercise full-model dimensions, the actual worker call chain or CPU/GPU parity.
The host scheduling cases also passed with native Windows MSVC and CTest.

The earlier candidate compiled for CUDA architecture 89 and ran the experiments
below. That binary used the historical source base identified below and a local
shared CPU-pool overlay; it was not built from this clean PR branch. This PR carries only the
four-file prefill patch, its tests and documentation. A complete CUDA build and
GPU run of the submitted branch, HIP validation and other NVIDIA architectures
remain pending.

## Historical combined-policy measurements

Measured on 2026-11-13: Linux, dual Xeon Platinum 8380, four RTX 4090 24 GB GPUs
at 255 W each, a shared 144-thread CPU pool, four service slots and a 160 GB RAM
tier limit. The model was GLM-5.3-Flash, with 42 GGML expert layers: eight using
IQ3_S gate/up with Q5_K down and 34 using IQ3_S gate/up with IQ4_XS down.

Each fresh process used the same frozen expert profile and eight fixed public
code/document tasks: eight sequential requests, then the same eight in two
waves of four clients. The 256-output-token experiment used greedy decoding,
request-manifest seed 1729, DFlash spec 8, CPU prefill and adaptive timing. The
greedy path did not forward that seed to the native engine. All completed samples
were retained. This was a custom workload, not `maya.sh --bench` or
`tools/agent_bench.py`, so these numbers are not directly comparable to the
README speed tables.

A used the original v1.1.61 deployment with automatic layer splitting. B used
the candidate binary, fixed layer boundaries `13,24,35`, `MAYA_GLM_ROW_HALVES=1`
and `MAYA_GLM_PREFILL_ROW_LOCAL=1`. The local shared-pool overlay was present in
both deployments and is not part of this PR. The source base of the historical
candidate was `aecfbf3f2e1c6dc008c9c64c0765aa27d9c5cab2` plus that overlay and this
prefill patch. Binary SHA256 identities:

- A: `d6f6c802984ae7cf5c6ab3e3faa6b5f6473dbc37ef6996f8d0e926730aa37ad1`
- B: `e3797da1fae1cb5ee159d0cc89d1a4858adcb8999008f5aa23cec347e09c4187`

Throughput is total output tokens divided by request-drain wall time, including
prompt processing; it is not native decode speed. Each table cell represents
eight requests and 2,048 output tokens.

| Arm, in run order | Sequential output tok/s | Four-client output tok/s |
| --- | ---: | ---: |
| A1 | 47.932 | 95.680 |
| B1 | 54.677 | 119.250 |
| B2 | 53.467 | 115.995 |
| A2 | 47.916 | 96.485 |

Across the four B/A pairings, observed gains were 11.55–14.11% sequential and
20.22–24.63% with four clients. Those are **combined-policy observations**, not
an isolated prefill-switch result. Binary, layer-split policy, allocation,
calibration, actual expert dispatch, drafts and generated text can affect the
work. The earlier same-binary comparison also enabled both row switches
together and stopped before completing ABBA when the next automatic split
changed; it does not supply the missing isolation.

## Quality and merge boundary

The 256-token screen failed its blinded relative quality gate on a lease-rules
task; the planned five independent validation groups were not run. Its scoring
had an interpretation ambiguity, so the supported conclusion is that quality
acceptance was not established, rather than that this patch caused degradation.

A separate 1,024-token natural-stop study retained all 64 answers: 59 stopped
naturally and five hit the cap. It also failed relative per-task checks for a
cache-concurrency answer and SQL retry handling, and one of four candidate
lease answers missed the required completeness score. Baseline answers also
varied and contained errors. Neither study identifies the row scheduler as the
cause; neither permits a quality-preserving speed claim.

Before promotion, the clean branch needs full engine build/runtime validation,
an isolated comparison with row-halves allocation enabled on both sides and
only this prefill switch varied, and predeclared answer-quality acceptance.
Keep the switch off by default and the PR in draft while those remain open.
