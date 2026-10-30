# The neko 4bpw experts-only pack - score and loading fidelity (2026-10-14, ruapehu 7x3090)

neko-legends' EXL3 4bpw (experts-only: the routed experts at 4 bits, every other tensor BF16/F32) packed
with `tools/exl3_pack.py` (#170).  The review asked what the score means and what the load-time requantization
costs: the tokens and the fidelity numbers below are the baseline.  `tools/twin_fidelity.py <pack>` recomputes
the table on any pack.

## The tokens

`bench/score/glm53-sharegpt-tail.tokens.txt`: 7,667 tokens of plain text (no chat template) - 14 English
ShareGPT conversations (a human turn and the model's answer each), taken from the tail of
ShareGPT_V3_unfiltered_cleaned_split.json so nothing near the front of the file (the part every quant recipe
touches) is used.  Conversation ids (in order): Da9D8J4_0, XZWuTmr_0, 5i0bddh_0, HfxzLdB_7, HfxzLdB_0,
iIcCycL_0, bEzr5XF_1, bEzr5XF_0, ndbWw7c_0, jhqx19T_0, pmkmGBZ_0, q1BFLM4_0, rBdlKIj_0, bzC8YtK_0.
Tokenized with the model's own tokenizer (the EXL3 repo's tokenizer.json, `add_special_tokens=False`).
Run: `STRATA_GLM_SCORE=2 strata --glm-pack <pack> --tokens-file bench/score/glm53-sharegpt-tail.tokens.txt
--max-context 16384` - the same file scores any engine change or any other GLM-5.3-Flash pack (the ids are
the fixture, not a prompt).

## Loading fidelity (tools/twin_fidelity.py on the pack)

At load the engine keeps one copy of each big BF16 row: the rows the fast path's quantized matmul (q()) reads
get a Q8_0 copy INSTEAD of the BF16 one; the rows b16()/q_or_b16() read keep only their exact BF16.  The
error the Q8_0 copy adds, measured row by row against the pack's own BF16 bits (the kernel's decode of the
twin, its FP16 scale included):

| rows | BF16 MB | Q8_0 MB | rel-RMS | worst row | max abs err |
| --- | ---: | ---: | ---: | ---: | ---: |
| attention/dense projections (158) | 11,156.8 | 5,927.1 | 0.00549 | 0.00659 | 0.0310 |
| shared experts (126) | 2,113.9 | 1,123.0 | 0.00561 | 0.00743 | 0.0078 |
| dense FFN (9) | 906.0 | 481.3 | 0.00561 | 0.00564 | 0.0009 |
| output head (1) | 1,268.8 | 674.0 | 0.00539 | 0.00539 | 0.0008 |
| routers, ssm/hc, indexer, k_b/v_b (368) | 935.1 | 935.1 | exact | 0 | 0 |
| attn_q_a / attn_kv_a_mqa (22) | 184.5 | 184.5 | exact | 0 | 0 |

The requantized rows sit ~0.55% RMS from their BF16 source (worst row 0.74%) - an order of magnitude finer
than the pack's own 4bpw experts (~2.5% RMS, the quantization the model was published with) and below the
engine's own expert-kernel arithmetic (~1.5% per expert, q8_1 activations).  The exact rows include
attn_q_a / attn_kv_a_mqa, which the first version of #170 requantized to Q8_0; they run from the exact
BF16 now (the review's point 2).

## The score, before and after the review fixes

Same tokens, same box, same pack; the pre-fix binary is #170 as reviewed (a Q8_0 twin beside every BF16 row,
q_a/kv_a requantized), the post-fix binary is this branch (one copy per row).  Dense rows = the per-device
`glm pack: dense weights` sums (STRATA_GLM_TIMING); residency is the engine's own expert-slot count at
12,096 routed experts.

| binary | nll | ppl | top-1 | ms/token | dense rows | expert slots in VRAM | pool |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| #170 as reviewed | 0.78025 | 2.1820 | 78.51% | 40.2 | 23.64 GB | 11,167 (92.3%) | 131.2 GB |
| this branch | 0.78023 | 2.1820 | 78.60% | 34.8 | 8.54 GB | 12,084 (99.96%) | 142.0 GB |

The requantization the loading code performs is end-to-end invisible (nll moves 2e-5, top-1 +0.09% from the
q_a/kv_a exactness fix) - on these tokens the neko pack through this engine scores ppl 2.18 / top-1 78.5%,
a normal number for conversational text, so a poor score on other token sets points at the tokens (or the
fine-tune), not at the loading path.  The one-copy rule's 15.1 GB of dense-row VRAM went to the expert pool:
every one of the 12,096 experts now sits in VRAM (99.96%, up from 92.3%), the RAM tier shrinks 32.5 -> 21.7 GB,
and the scored decode runs 13% faster (40.2 -> 34.8 ms/token, VRAM hit 97.57 -> 99.57%).

A cross-model baseline (the same tokens through turboderp's 4.05bpw, Maya-S, or STRATA_GLM_SCORE_TOPK against
the FP8 reference) did not run for this revision: the reference dumps from the Maya-S calibration are gone with
the box that made them, and every candidate model (turboderp 4.05bpw ~170 GB, Maya-S 96.5 GB, Maya-S24 94.7 GB)
exceeds ruapehu's 60 GB free alongside the live 225 GB source + packs.  The tokens file above makes the run a
one-liner on any box that has the model.
