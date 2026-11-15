# bench

Knobs for the GLM multi-sequence bench. All are environment variables read by the
`generate` binary's bench section (`src/program/generate.cpp`, the `STRATA_GLM_MULTI_BENCH`
handler). The default path (no bench knobs set) is untouched by every one of them.

## STRATA_GLM_MULTI_BENCH=<n>,<L>,<G>[,<K>,<same_prompt>]

Multi-sequence bench mode. Five fields, parsed in this order:

1. `n` - number of sequences benched together (default 2).
2. `L` - prompt token count per sequence (default 512).
3. `G` - greedy tokens generated per sequence (default 256).
4. `K` - boundary rounds, the sequence-lifecycle churn count (default 32).
5. `same_prompt` - when set, all `n` synthetic prompts are identical (one fixed seed);
   when clear, each sequence gets its own stream.

The output block prints per-sequence agreement ("seq s vs seq 0 ... vs solo") with the
throughput verdict in the `tiers(...)` line: `together` on one GPU via the multi-rows path,
`pipelined` on a split via `multi_issue`. `--tokens` with this mode runs the same real prompt
for every sequence.

## STRATA_GLM_MULTI_BENCH_PREFIX=<n>

Prefix probe, off by default. Needs both `STRATA_GLM_MULTI_BENCH` and
`STRATA_GLM_MULTI_BENCH_PREFIX` set; with the bench mode alone the default path is
byte-identical to the plain multi-sequence run. When the probe is on, the first `n` tokens
of every sequence come from one shared stream and the remainder is re-seeded per sequence
(partial overlap, the case the other modes do not cover: `same_prompt` shares the whole
prompt, plain `MULTI_BENCH` shares nothing).

The probe is synthetic-prompts-only: with `--tokens` it would overwrite the real prompt with
synthetic tokens, so run it without `--tokens`.

It also records a per-sequence prefill token meter (`prefill_tok`, capped at 256 sequences)
and prints it alongside the agreement block:

    prefix probe: <n> shared tokens, per-sequence prefill tokens: seq0=... seq1=...

The meter is the signal a shared-prefix reuse gate would check: a second request of a shared
prefix must skip most of the shared prefill. Every new counter lives under the probe flag;
the default path stays untouched.
