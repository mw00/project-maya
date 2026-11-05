# Maya-S v1 against the FP8 model (2026-10-07, Uranus)

`GLM-5.3-Flash-Maya-S-IQ2_XXS` (90.1 GB, 2 shards): IQ2_XXS experts (gate/up with error-feedback rounding), IQ2_S down
in MoE layers 3-6 and 41-44, Q6_K attention / shared experts / dense layers / embeddings / output, Q8_0 small KDA
projections and the DSA indexer, F32 router and mixing weights, NextN layer included (Q2_K/Q3_K experts).
Recipe: `tools/maya_quant/recipes/maya-s-v1.json`. Made in 280 min on Uranus (1x V100, 6 cores) from the FP8 release.

## Distribution match on held-out text

The FP8 model's own top-64 log-probabilities (calib.py's reference: its forward in PyTorch, `hc_head` = the mean of
the four streams as in the official modelling code) against Maya-S run through the engine's one-token decode path
(`STRATA_GLM_SCORE_TOPK`, `tools/maya_quant/kl_eval.py`): 8 eval sequences never used for calibration, positions
64-1022, 7,672 tokens. KL is KL(FP8 || Maya-S) over the top 64 plus one bucket for the rest (the top 64 hold 98-99% of
FP8's probability). FP8's perplexity counts a token outside its top 64 at the 64th log-prob (a lower bound).

| Seq | Text | FP8 ppl | Maya-S ppl | KL | Same top token |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | chat | 5.26 | 5.68 | 0.335 | 84.6% |
| 3 | chat | 4.13 | 4.46 | 0.161 | 86.7% |
| 6 | reasoning trace | 4.02 | 4.31 | 0.492 | 80.7% |
| 9 | reasoning trace | 2.35 | 2.49 | 0.069 | 90.2% |
| 12 | web code (HTML/JS) | 1.97 | 2.28 | 0.286 | 88.5% |
| 15 | other code | 5.39 | 6.64 | 0.455 | 83.8% |
| 18 | tool calls | 6.57 | 7.08 | 1.026 | 75.0% |
| 21 | wikitext-2 (largely memorized by FP8) | 1.59 | 4.09 | 1.056 | 65.2% |
| **all** | | **3.51** | **4.31** | **0.485** | **81.8%** |
| all but wikitext | | 3.92 | 4.35 | 0.403 | 84.2% |

Top-1 accuracy on the actual next token: FP8 71.5%, Maya-S 68.1%.

Reading: the ~2.2 bits a weight cost most where FP8 is most certain - memorized text and the exact formats of tool
calls - and least in chat and reasoning. These numbers include the engine's own arithmetic (q8_1 activations in the
expert kernels, about 1.5% per expert), not only the quantization.

## Loops

`tools/maya_quant/loop_test.py`, 5 prompts (a three.js scene, a canvas animation, an explanation, a study plan in
Portuguese, a proof) at temperature 1.0 and greedy, Max thinking (no effort line - the template's default), 6,000 new
tokens: 10 of 10 answers without a repeated 64-token passage or a back-to-back repeat. None had closed its reasoning
by 6,000 tokens at Max effort; the tails read as coherent planning (orbital mechanics for the scene, colour and
layout choices for the canvas, a finished proof being polished).

At the dashboard's default thinking level (the template's High), with room for 14,000 new tokens: the three.js and
canvas prompts at temperature 1.0 and greedy - 4 of 4 without a loop, and 4 of 4 still planning when the budget ran
out (detailed, coherent design work: shader varyings, materials, orbital motion). Long thinking on creative web
prompts is the model's habit at High; a thinking budget in the server would bound the wait.

## Context fill

`tools/maya_quant/ctx_fill_test.py` on Uranus (1x V100 32 GB, 64 GB RAM): the project's documents as the context, a
fact planted at its very start, the question at its end (thinking Low, greedy).

| Prompt | Found the fact | Ended by itself | Loops | Decode | Prompt |
| --- | --- | --- | --- | ---: | ---: |
| 8,000 tokens | yes | yes | none | 15.9 tok/s | 4.0 ms/token |
| 16,000 tokens | yes | (the harness did not stop at end tokens then) | none | 14.8 tok/s | 4.1 ms/token |
| 30,000 tokens | yes | yes | none | 9.6 tok/s | 4.5 ms/token |

Each answer: the exact code and the one sentence about where it was, then the end-of-turn token.

## Speed at context depth (2026-10-07, Mercury)

Mercury (2x V100 32 GB, 30 GB RAM), through the dashboard: the repository's documents and kernels as the prompt, then
300 new tokens at temperature 1.0, thinking off, the MTP block drafting.  The DSA attention's pool selection used to
rank every pool against every other on each token (quadratic in the context: 21 ms per DSA layer at 64K); it is now
a radix select (0.07 ms at 64K, the same pools in the same order - tests/dsa_topk_parity.cu), and the decode no
longer slows down with the context:

| Prompt | 0.9K | 15-18K | 32K | 48K | 60K |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decode, tok/s | 32.8 | 31.7 | 30.5 | 29.5\* | 28.9\* |
| Prompt, tok/s | 165 | 384 | 385 | 357\* | 353\* |

\* with the image encoder resident beside the model (2.5 GB less expert cache on the first GPU); the other columns
with it on demand, as shipped.  The context-fill numbers above (Uranus) predate the fix: there the decode fell from
15.9 tok/s at 8K to 9.6 at 30K.

## Thinking budget

Long creative prompts at High effort planned past 14,000 tokens (above).  The server now caps a reasoning block at
32,768 tokens by default (`thinking_budget` in the config; 0 = none): the engine emits `</think>` in place of the next
sampled token and the answer follows in the same decode, nothing read again.

## What v2 targets

- Tool-call formats and memorized knowledge: more tool-call calibration, Q8_0 attention and shared experts.
- Per-layer sensitivity for the down projections (IQ2_S / IQ3_XXS where it pays), and per-expert bits in the Maya pack.
- Output-matching refinement of the quantized scales (the plan's scale-refinement step).
