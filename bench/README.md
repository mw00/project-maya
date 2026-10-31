# bench

Harness knobs for the maya-paged-attention mirror (branch `paged-attn`). All
knobs are environment variables read by the `generate` binary's bench section
(`src/program/generate.cpp`, the bench block at line 2281+).

## STRATA_GLM_MULTI_BENCH=<n>,<L>,<G>[,<K>,<same_prompt>]

Multi-sequence bench mode. Five integers per the `sscanf` at
generate.cpp:2286-2287, in this order:

1. `n` - number of sequences benched together (default 2).
2. `L` - prompt token count per sequence (default 512).
3. `G` - greedy tokens generated per sequence (default 256).
4. `K` - boundary rounds, the sequence-lifecycle churn count (default 32).
5. `same_prompt` - when set, all n synthetic prompts are identical (fixed seed
   at generate.cpp:2298); when clear, each sequence gets its own stream.

The output block prints per-sequence agreement ("seq s vs seq 0 ... vs solo",
generate.cpp:2416-2424) with the throughput verdict in the `tiers(...)` line
(generate.cpp:2410, :2423): `together` on one GPU via `multi_rows`
(generate.cpp:2364/:2372), `pipelined` on the split via `multi_issue`.
`--tokens` with MULTI_BENCH runs the same REAL prompt for every sequence
(generate.cpp:2300).

## STRATA_GLM_MULTI_BENCH_PREFIX=<n>

Prefix probe, default OFF. Needs BOTH env vars set (`STRATA_GLM_MULTI_BENCH`
and `STRATA_GLM_MULTI_BENCH_PREFIX`); with the bench mode alone the default
path is byte-identical to the Task 0.3 goldens. When the probe is on, the
first `n` tokens of every sequence come from one shared stream and the
remainder is re-seeded per sequence (partial overlap, the case the existing
modes do NOT cover: `same_prompt` shares the whole prompt, plain MULTI_BENCH
shares nothing).

The probe is synthetic-prompts-only: with `--tokens` it would overwrite the
REAL prompt with synthetic tokens, so run the probe without `--tokens`.

The probe also records a per-sequence prefill token meter (`prefill_tok`,
capped at 256 sequences) - what the bench fed prefill for each sequence -
and prints it alongside the existing agreement block:

    prefix probe: <n> shared tokens, per-sequence prefill tokens: seq0=... seq1=...

P3's prefix-hit gate reads this meter against the shared count: the second
request of a shared prefix must skip >= 95% of the shared prefill. Any new
counter lives under the probe flag; the default path stays untouched.
