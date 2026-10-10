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

The search starts at `STRATA_GLM_MTP_START` (2) and climbs: the start runs 8 rounds (6 measured), then each untried
neighbour of the best a 4-round probe (2 measured), until both neighbours of the best are measured and lose. A round's
time is the mean of its length's first ten rounds, then an EMA (one slow round does not stand for its length), and the
acceptance of a draft position is pulled toward the one before it while it has few rounds. The acceptance counts
decay over each position's last 64 comparisons (`STRATA_GLM_MTP_ACC_WINDOW`, 0 = every comparison since the load): with
whole-session counts a request's easy start spoke for its hard part (Maya-S kept 2 drafts through a 1536-token essay
with 57% of the drafts accepted, 21.3 tok/s, where 1 draft gives ~22). Every length used to be
tried for 8 rounds first, the longest first and token by token last, so a machine's first request paid for lengths it
would not use. The token path (length 0) is now only tried where length 1 is the best. Then a neighbour of the best
gets a 4-round probe now and then; the gap doubles while the best stays, up to 1024 rounds. Where the routed experts
mostly come from RAM, a verify row costs its experts in full, and the length can settle at 0 (token by token).

## The draft head

A draft only has to be a good guess - the verify decides every token - but each draft step ran the whole output head:
154880 x 4096 Q6_K, 0.52 GB, ~2.4 ms of a ~4 ms step on the Radeon 8065S. The draft head now runs over a **draft
vocabulary**, the full head's own rows for its tokens, so a draft inside it is the one the full head makes:

- GLM's byte-level BPE lays its vocabulary out by script: ids `[0, 98304)` are the Latin / code block, then the Han
  block (with the multi-digit numbers in it), then Cyrillic, Arabic, Kana and the rest, the control tokens last. The
  draft vocabulary is read from the GGUF's tokens: the leading block (the ids up to the first thousand that is mostly
  not ASCII), then the digit-only tokens and the control / user-defined ones after it (the roles, `<think>`, the
  tool-call tags). For GLM-5.3-Flash: 98,304 + 437 rows, 64% of the head (the 437 gathered, the rest read in place).
- The ids follow the BPE merge order, not the frequency: the first 32K ids missed 10% of the tokens of Maya-S's greedy
  bench answers (`' TCP'`, `` '```' ``, `'**:'`, `' dictionaries'` ...), the first 64K 4.4%, the vocabulary above
  0.07%. A miss costs the round its later drafts (~0.9% of a round's tokens for each 1% missed, with ~82% of the drafts
  accepted), so a 32K head would save ~4% of a round and lose ~9%.
- Where the answer leaves the vocabulary (an answer in another script) the drafts go back to the whole head by
  themselves: above 2% of the last ~64 emitted tokens outside it.

`STRATA_GLM_MTP_DRAFT_VOCAB=<K>` takes the first K ids as the leading block instead; `0` = the whole head.

## The chained drafts

A round's drafts used to go one by one: each step waited on the host for its token, then the host dequantized its
embedding row and copied it in, and the round synchronized with the device before its drafts and its NextN cache rows.
On Windows a host round trip also decides when the queued kernels are submitted. Now `token_embd` stays on the device
as stored (0.52 GB of Q6_K; `embed_tok` dequantizes a row to the host's floats), each step's argmax feeds the next
step's embedding there, and the host reads the round's drafts back once. The host's own rows (the first draft's token,
the cache rows) go through their own pinned buffers, so nothing in flight is overwritten and the round no longer waits
before them. `STRATA_GLM_MTP_CHAIN` is on where the device holds every routed expert and `token_embd` besides; `1`
forces it, `0` restores the step-by-step drafts.

### Measured (Maya-S, Radeon 8065S / Gorgon Halo, greedy)

A round of 2 drafts went from ~100 ms to ~95 ms (the draft vocabulary ~4 ms of it, the chain ~1 ms; the draft vocabulary
held every token of the bench's English and code answers, so no draft changed). Against the MTP decode before these
changes, same session:

| | before | after |
|---|---|---|
| 5 answers x 256 tokens, mean | 25.1 tok/s (first 25.0) | 27.8 tok/s (first 28.6) |
| the warm-up request (63 tokens, the length search) | 24.5 tok/s | 26.7 tok/s |
| 1536-token essay | 22.1 tok/s | 22.0 tok/s (1 draft a round: little to save) |
| 512 tokens after an 8K-token prompt | 22.0 tok/s | 23.6 tok/s |
| 3 x 2048 tokens + 1024 after an 8K prompt | 22.65 (20.9 / 24.2 / 22.5; 23.9) | 23.10 (21.9 / 24.8 / 22.6; 23.6) |

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
- The chained drafts' `token_embd` on the last GPU (0.52 GB for Maya's Q6_K), only where VRAM holds every routed expert
  and it besides; the draft head's gathered rows (~1.5 MB).
- A prompt fills the block's cache 2048 rows at a time.
- Drafts skip the experts they miss beyond the best two (`STRATA_GLM_MTP_KEEP`), and any that are only on the SSD.

## Settings

`STRATA_GLM_MTP_DRAFT`, `STRATA_GLM_MTP_ADAPT`, `STRATA_GLM_MTP_FIXED`, `STRATA_GLM_MTP_START`,
`STRATA_GLM_MTP_DRAFT_VOCAB`, `STRATA_GLM_MTP_CHAIN`, `STRATA_GLM_MTP_KEEP`, `STRATA_GLM_MTP_SLOT_SHARE`,
`STRATA_GLM_NO_SPEC` (token by token), `STRATA_GLM_NO_MTP` (no block), and `STRATA_GLM_MTP_PIPELINE` (the earlier
two-GPU pipelined decode instead) - see the README's settings table. `STRATA_GLM_MTP_TRACE=1` prints every round. Each
request prints `glm mtp:` lines on stderr: the rounds, tokens a round, the lengths used, a round's host time per phase,
and the emitted tokens outside the draft vocabulary.
