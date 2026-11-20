# GLM shared-pool affinity: one dual-Xeon observation

This is historical v1.1.57 configuration evidence from 2026-11-11 (Asia/Shanghai),
not a stock speed-table entry, current-main runtime acceptance or a universal
NUMA speedup. The tested 128/144 counts were operator choices for this host.
**144 is not a default, preset or recommendation.** Select and compare counts on
your own machine using the [affinity guide](../GLM_CPU_AFFINITY.md).

## Machine and method

- Two Xeon Platinum 8380 sockets, two NUMA nodes, 80 physical cores / 160 logical
  CPUs; approximately 251 GiB RAM; four RTX 4090 24 GB GPUs, each limited to 255 W.
- GLM-5.3-Flash, runtime model label `GLM-5.3 · Maya-L`; same model pack and initial
  expert profile in all arms. Context 1,048,576; INT8 KV, resident context 32,768;
  configured prefill chunk 5120; MTP enabled; DFlash disabled; serving capacity 4.
- All three arms used the same v1.1.57-based binary with a local patch allowing an
  explicit shared-pool count above 64, bounded by its allowed CPU mask. Binary
  SHA-256: `cef6db8b83601e2c361cdf529de67c9ecbed5ec295da1fccd89901e04e9d79e8`.
  No affinity-specific engine change was made between arms. The current PR
  confines that count override to the actual global shared-pool branch; this
  historical measurement does not validate a new current-main build.
- Each arm invoked the unmodified upstream `maya.py --config ... --bench`:
  warmup, three greedy/no-thinking answers capped at 256 tokens, prefill warmup,
  2048-token prefill and 8192-token prefill. All 21 commands completed. Requests
  were serialized, not four concurrent clients. The surrounding adapter imposed
  cooling/resource admission between commands; waits are outside native timings.
- Sequence: original unpinned 128, balanced worker list 128, balanced worker list
  144; one pass each. The two lists used the host's actual topology, physical
  cores before SMT, alternating nodes. Runtime masks confirmed 127 and 143 pinned
  background workers; callers retained their existing unpinned policy, and the
  20 ms worker spin was unchanged. Automatic CPU/PCIe lane retiming remained on.

## Results

| Configuration | Three-answer mean decode tok/s | Change | 2K prefill ms | 8K prefill ms | 8K prefill speed change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original 128 | 58.1691 | baseline | 3026.8 | 5833.8 | baseline |
| Balanced list 128 | 57.6495 | -0.893% | 2811.8 | 5462.3 | +6.801% |
| Balanced list 144 | 58.7472 | +0.994% | 2931.6 | 5413.4 | +7.766% |

All three decode answers produced exactly 256 tokens. Their individual native
rates were 59.1265 / 56.3306 / 59.0501, then 58.5853 / 55.8075 / 58.5558, then
58.0657 / 56.7363 / 61.4395 tok/s. The mean includes all three; warmups and the
single output token from a prefill measurement are excluded from that mean.
2K prefill speed changed by +7.646% and +3.247%, respectively.

The [sanitized sample CSV](glm-pool-affinity-20261111.csv) retains all 21 command
records, including warmups, native timings, token counts, input hashes, generated
token-ID hashes and speculative counters. Token-ID hashes encode an unsigned
64-bit little-endian token count followed by unsigned 32-bit little-endian IDs.

Against original128, balanced128 matched the generated token IDs of only the
first of three decode answers; balanced144 matched none. All three arms matched
the single output token from each prefill measurement, including its warmup.
This is not broad quality or greedy-output equivalence.

The observed prefill improvements did not establish a meaningful decode
improvement. There was no repeated/randomized A/B/A control; the 144 arm changes
both affinity and thread count. Output/routing differences and automatic retiming
preclude attributing the changes solely to worker affinity. The original128
configuration was restored; neither affinity candidate was promoted.

These observations do not test DeepSeek Engram page migration (PR #784), other
models, longer contexts, concurrent clients, different power limits or arbitrary
NUMA/SMT hardware. The new topology helper has synthetic fixture coverage; it has
not been benchmarked across those machines. No model requests were added to
prepare this PR.

## Retained source identities

SHA-256 values from the original evidence, verified again when preparing this PR:

| Artifact | SHA-256 |
| --- | --- |
| Upstream benchmark Python source | `1d6efa106997351118b69e953f7bba3547b305f15577505c7adc7ff4c9c7e113` |
| Shared input fixtures | `ba2b4d55d475ce4ea10d661f28ff8e4af9149a7cc7277dcb1447aa0d77b78a6f` |
| Original128 official report | `faf298b057253178a574659ebc8caa3bf5471dab5f8816f2875eeabc6a38817d` |
| Balanced128 official report | `e6c4b5382c54e6570a51fe7c3d899d67b1c881424d423c51ddb6e9663c6f8571` |
| Balanced144 official report | `962cdef94a950683a480c0ec69a181c40a2e381168ca6aa1ab946f340a7da8c1` |
| Completed/restored outcome | `af879256a256d7eecce3e676afbadef05aa6e18e16bc949085ddcfabffa49294` |
