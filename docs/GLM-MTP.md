# The MTP decode

GLM-5.3-Flash carries one MTP (NextN) block: a DSA attention layer and an MoE layer that read the model's last hidden
state and the next token, and predict the token after it. Maya decodes with it on one GPU or any split
(`src/core/glm_mtp.cu`). Each round, from the last emitted token `y` at position `p`:

1. **Draft** - the block runs at `p - 1` on (the model's hidden state of `p - 1`, `y`) and proposes `d1`; its own output
   state (the shared head norm's, which also feeds its head) and `d1` give `d2` at `p`, and so on, up to
   `STRATA_GLM_MTP_DRAFT` drafts (3). Each draft is the block head's argmax.
2. **Verify** - the window `[y, d1 .. dn]` goes through every layer of every GPU as one batched pass:
   - read once for all rows: the projections, the shared experts, the routers and the output head (`mv_rows`), and
     the mHC reads (`hc_rows`) and the recurrent layers' convs (`kda_prep_rows`) take one launch;
   - row after row with the one-token kernels: the recurrence (state in registers across the rows), the DSA attention
     and each row's routed experts through the tiers.
3. **Accept** - the model's own sample at row `t` (the request's sampler, draw `counter + t`) is the token of position
   `p + t + 1`. Draft `t + 1` stands while it equals that sample. The first mismatch is the new token; if every draft
   stands, the last row's sample is a bonus token.
4. **Commit** - the recurrent layers advance by the kept rows only: the verify read their states and wrote none, and
   the commit replays the kept rows' recurrence from inputs the verify kept. The conv histories come from the kept
   row's copy. The attention caches are written by position and need nothing.
5. **Cache** - the block's cache takes the kept rows with the model's hidden states (its cache-writing half: eh_proj,
   the attention norm, kv_a and the indexer's key and gate). The entries the drafts wrote from the block's own states
   are replaced. The prompt path and the token path write the same entries for the prompt.

## Exactness

Every row's arithmetic is the one-token path's: the multi-row kernels run each row's dot products in the same order,
and the recurrence keeps its state in registers instead of a round trip through memory. A greedy decode gives the same
tokens with any draft length. Checked on one V100 and on two (a split), with 0, 1 and 3 drafts a round: 200 greedy
tokens after a 3000-token prompt, identical. The CPU lane was off for this: the CPU computes its experts in its own
order.

Compare within one build and memory layout (`STRATA_GLM_MTP_FIXED=0` is the token path with the same layout). The
block's VRAM changes the expert layout, and the prompt path's arithmetic depends on it. A sampled decode draws each
emitted token at the counter it would have had token by token.

## The draft length

`STRATA_GLM_MTP_ADAPT` (on) measures the rounds of each length on this PC (their wall time, an EMA) and how often each
draft position is kept. It uses the length with the most tokens a second: `(1 + a1 + a1 a2 + ...) / round time`.

A length runs in stints. A round's time holds the tier work left over from the round before it (its routes' answers
and moves), so the first two rounds after a change of length, and a request's first two rounds, are not measured.

Every length is tried for 8 rounds first. Then a neighbour of the best gets a 4-round probe now and then; the gap
doubles while the best stays, up to 1024 rounds. Where the routed experts mostly come from RAM, a verify row costs its
experts in full, and the length can settle at 0 (token by token).

## Where it runs

The MTP decode runs where the expert pool holds a third of the routed experts or more. Below that, the experts
mostly stream from RAM, and a verify row costs its experts in full: on one V100 (about a quarter of Maya-S's experts in
VRAM), 1 draft a round gave 16.6 tok/s against 18.1 token by token. There the block stays out of the tiers, the
prompt path and the snapshots, its weights are freed, and the decode runs token by token as before. Setting
`STRATA_GLM_MTP_DRAFT` drafts anyway.

## Memory

- The block: ~173 MB of dense weights on the last GPU.
- Its expert slots (`STRATA_GLM_MTP_SLOT_SHARE`): a model layer's full claim where VRAM holds 45% of the experts or
  more, a quarter of it below that.
- Its RAM-tier experts after the model's.
- The window's buffers: ~60 MB a GPU for 4 rows.
- A prompt fills the block's cache 2048 rows at a time.
- Drafts skip the experts they miss beyond the best two (`STRATA_GLM_MTP_KEEP`), and any that are only on the SSD.

## Settings

`STRATA_GLM_MTP_DRAFT`, `STRATA_GLM_MTP_ADAPT`, `STRATA_GLM_MTP_FIXED`, `STRATA_GLM_MTP_KEEP`,
`STRATA_GLM_MTP_SLOT_SHARE`, `STRATA_GLM_NO_SPEC` (token by token), `STRATA_GLM_NO_MTP` (no block), and
`STRATA_GLM_MTP_PIPELINE` (the earlier two-GPU pipelined decode instead) - see the README's settings table.
`STRATA_GLM_MTP_TRACE=1` prints every round. Each request prints `glm mtp:` lines on stderr: the rounds, tokens a round,
the lengths used, and a round's host time per phase.
