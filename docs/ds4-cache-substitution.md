# Decode with cache-resident substitute experts (experimental, off by default)

On an earlier research build, this policy increased DeepSeek-V4.1-Flash-1017 EXL3 decode from **42.24 to
52.59 tok/s (+24.5%)** on three R9700s. Total answer-generation time fell 17.3%. It sometimes computes a
slightly lower-ranked expert already in GPU memory instead of an uncached expert on the CPU lane.
This is approximate inference: expert identities, routing weights, logits and answers can change.

Those numbers describe the research stack, not a promised speedup on current main or other hardware.
They include native MTP, TP2, a 20 GB helper pool, 256 GB RAM, a 3.5 bpw EXL3 target, and a four-bit drafter.
An estimated 82.8% reduction in logical packed-weight CPU reads per generated token accompanied the speedup;
that counter is not a physical DRAM or PCIe measurement.

## Enable explicitly

In a configuration's `env`, before loading the model:

```json
{
  "MAYA_DS4_CACHE_SUB_RATIO": "0.1",
  "MAYA_DS4_CACHE_SUB_N": "12",
  "MAYA_DS4_CACHE_SUB_RISK_BUDGET": "0.6"
}
```

The native engine also accepts the `DS4_` / `STRATA_DS4_` spellings. Ratio zero (the default) keeps the native
kernel and mailbox unchanged, without allocating the substitution sidebands. Reload to change the policy.
It applies to trunk decode and target verification, including batched conversations; prompt feeding and the
drafter retain their native routing. It works from cache residency, without an EXL3-only assumption or a fixed
number of GPUs. A healthy configured helper also counts as resident.

| Setting | Default | Meaning |
| --- | --- | --- |
| `CACHE_SUB_RATIO` | `0` | Minimum replacement/original **unbiased positive score** ratio; [0, 1]. Zero disables substitution. |
| `CACHE_SUB_N` | `12` | Candidate window: 12, 16 or 32, clipped to the layer's expert count. |
| `CACHE_SUB_RISK_BUDGET` | `0.6` | Nonnegative finite limit on normalized source-plus-replacement score mass per token/layer. |
| `CACHE_SUB_NORM` | `reference` | Only `reference` is supported: keep the original top-k score denominator. |

Each setting above has the `MAYA_DS4_` prefix. The budget is a heuristic contribution budget, **not** an
error probability, confidence in an answer, a learned safety predictor, or a guarantee about expert similarity.
Ratio 1 is still approximate: selection bias can rank a higher unbiased score below a native choice. Use ratio
zero for native routing.

## Selection and accounting

Candidates retain the model's authoritative `sqrt(softplus(logit)) + bias` order, including ID tie breaking.
Ratios use `sqrt(softplus(logit))`, never arbitrary logits. Preserve the first-ranked expert, the strongest
unbiased native choice, and every native choice already in primary or healthy-helper VRAM. A remaining miss
can take the highest-ranked unused resident candidate below the normal cutoff if it passes the score ratio.

Propose replacements first. Admit them in increasing order of
`original_score / original_topk_sum + replacement_score / original_topk_sum`, with stable tie breaking,
while the cumulative cost fits the budget. Rejected proposals are not reassigned. Keep the native number of
active experts and the native denominator: do not renormalize the replacements to recover lost mass.
Selection bias means the total routing mass can increase or decrease.

The window snapshots helper membership before GPU threads run, so TP members agree. A failed helper stops
being eligible in subsequent snapshots; the existing CPU fallback still executes already-selected expert IDs.
Original demand continues to update profile counts, cache heat and promotion requests. Executing a replacement
must not teach the cache that the displaced expert was never requested.

The feature refuses `DS4_SYNC` and resident/RAM/disk expert skipping combinations. When enabled, generation
uses the unfused router rather than silently bypassing substitution through `DS4_ROUTE_FUSE`; lookahead is
still published. No predictor, prefetch policy, cache eviction rule, weights or global defaults are changed.
The same request and cache history matters: residency-dependent routing is not reproducible from a sampling
seed alone. Batching and verification windows can also change those choices; speculative decoding does
not make this policy equivalent to the native model or a token-at-a-time cache history.

Native `STAT` counters, when active: `sub_picks`, `cache_substituted`, `sub_demand_gpu`, `sub_executed_gpu`,
`sub_removed_mass`, `sub_executed_mass`. GPU fractions compare original demand with executed picks; mass fields
are normalized per token/layer. They count target routing rows, including speculative rows later rejected,
not only emitted tokens. The ordinary cache-hit fraction describes executed routing and can rise mechanically.

## Quality evidence from the research stack

Twenty-four official hard LiveCodeBench questions, two paired sampling seeds, temperature 1 / top-p 1 /
top-k 64, high reasoning, 65,536 completion tokens. Private tests were used only by the grader.

| Outcome, 48 attempts per policy | Native | Substitution (0.1 / 12 / budget 0.6) |
| --- | ---: | ---: |
| Passed tests | 44 | 45 |
| Finished, wrong | 4 | 2 |
| Completion limit | 0 | 1 |

The [per-answer research measurements](benchmarks/ds4-cache-substitution-research.json) include question IDs,
seeds, outcomes, token counts, timings and logical read proxies, without model answers or private tests.

Among 47 pairs that both finished, one native pass became a failure and three native failures became passes.
These are **24 question clusters**, not 96 independent questions, an official full-benchmark score, a
contamination-free claim, evidence of improved accuracy, or proof of noninferiority. One full paired seed
was measured before a helper-board replacement and one after it, with both policies matched within each
hardware configuration; 22 unmatched candidate answers were retained separately and excluded.

Median generated length changed +0.19%; the earlier aggregate length increase was concentrated in outliers.
That does not establish the frequency of rare long or runaway completions. Broader agent trajectories,
strict schemas, long reasoning, other quants and other hardware remain important validation work.

## Tests

Build and run `ds4_cache_sub_test` for parser rejection, protections, native fallback and randomized selector
invariants. `ds4_route_sub_test` needs a CUDA/HIP GPU but no model: it compares actual kernels with the CPU
selector, checks bit-identical native/off routing, helper eligibility, reference mass, vision bias, windows
of 1/4/16 rows and different expert counts. Full-model integration checks and current-main performance
measurements must be reported separately from the earlier research result above.

## Current-main port validation

On `3c935e411cd6625a9af0f292dd0af52598e866d8` (v1.1.52), with the same source model/quant and three R9700s:

- Pristine upstream versus port disabled: all **83 x 129,280** saved FP32 logits were finite and bit-identical;
  32 generated token IDs matched. This includes 81 prompt rows and the last target verification window.
- Port enabled: all 81 prompt rows remained bit-identical; 1,733 of 7,440 decode picks were substituted,
  with zero eligible prefill picks.
- The native two-conversation check at context 4,096 passed: 14/14 logits rows bit-identical alone/together,
  with the same generated tokens in each of three MTP modes (serial, separate drafts, one-pass drafts).
- Two short API prompts, independent model loads in off/on/on/off order, greedy, a **512-token cap per answer**:
  pooled decode **36.25 -> 60.87 tok/s (+67.9%)**. All eight responses reached the cap. This is a small
  throughput smoke test, **not** completed-answer quality evidence or a general speedup estimate.

[Current-main measurements](benchmarks/ds4-cache-substitution-current-main.json) retain the prompts, timings,
counts and test configuration. Each load starts from identical profile/LT copies; ordinary adaptive expert
caching and MTP remain on for the API check. Both arms disable the prompt peer all-reduce identically, avoiding
the unrelated startup-probe drain issue tracked in #748; that patch is not included here. Other performance
settings remain the same in both arms. Normal serving was restored and its executable, configuration,
environment and health independently verified.

Local operator checks: 20,000 randomized CPU selector cases plus parser/protection/fallback cases passed,
including ASan/UBSan. The GPU test passed 2,916 cases on two R9700s, including native/off bit parity.
`tools/check_hip_compat.py` and all 48 dashboard tests passed. CUDA execution has not been measured on this
workstation. Model integration tests are bounded diagnostics, not proof for all contexts, batches or workloads.
