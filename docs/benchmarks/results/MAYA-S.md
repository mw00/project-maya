# Maya-S against the FP8 model

`GLM-5.3-Flash-Maya-S` (96.5 GB, the files `Maya-S-v2-IQ2_XXS/` on Hugging Face): IQ2_XXS gate/up experts rounded with
error feedback (GPTQ-style, each expert with its own input statistics from the FP8 model), IQ2_S down projections and
IQ3_XXS in the first and last four MoE layers, Q6_K attention / shared experts / dense layers / embeddings / output,
Q8_0 small KDA projections and the DSA indexer, F32 router and mixing weights, the NextN (MTP) draft block.
Recipe: `tools/maya_quant/recipes/maya-s-v2.json`, made from Z.ai's FP8 release.

## Distribution match on held-out text

The FP8 model's top-64 log-probabilities on 8 held-out sequences never used for calibration (7,672 positions) against
Maya-S through the engine's one-token decode path (`tools/maya_quant/kl_eval.py`). KL is KL(FP8 || Maya-S) over the top
64 plus one bucket for the rest.

| Text | KL | Same top token |
| --- | ---: | ---: |
| chat | 0.290 | 85.8% |
| chat | 0.163 | 87.5% |
| reasoning trace | 0.422 | 83.3% |
| reasoning trace | 0.061 | 89.3% |
| web code (HTML/JS) | 0.256 | 90.1% |
| other code | 0.450 | 84.8% |
| tool calls | 0.850 | 76.5% |
| wikitext-2 (largely memorized by FP8) | 0.934 | 68.8% |
| **all** | **0.428** | **83.3%** |
| all but wikitext | 0.356 | 85.3% |

| | FP8 | Maya-S |
| --- | ---: | ---: |
| Perplexity | 3.51 | 4.19 |
| Top-1 accuracy on the actual next token | 71.5% | 68.8% |

## Task accuracy

| Task (zero-shot) | FP8 | Maya-S | Recovery |
| --- | ---: | ---: | ---: |
| ARC-Easy (acc) | 87.2 | 86.2 | 98.9% |
| ARC-Challenge (acc norm) | 71.0 | 68.2 | 96.1% |
| HellaSwag (acc norm) | 88.5 | 87.5 | 98.9% |
| WinoGrande (acc) | 78.5 | 75.5 | 96.2% |
| PIQA (acc norm) | 87.0 | 86.2 | 99.1% |
| **Average** | **82.5** | **80.8** | **97.9%** |

400 questions per task, the same questions for both models, scored the way lm-evaluation-harness scores them (the answer with the highest log-likelihood; length-normalized where the choices differ in length) - the FP8 model run layer by layer in PyTorch, Maya-S through Project Maya's engine (`tools/maya_quant/zs_*.py`).

## Long answers and long context

`tools/maya_quant/loop_test.py` (a three.js scene, a canvas animation, an explanation, a study plan in Portuguese, a
proof; temperature 1.0 and greedy, 6,000-14,000 new tokens) and `tools/maya_quant/ctx_fill_test.py` (a fact at the
start of an 8K, 16K and 30K-token context, asked about at the end), run on the Maya-S line with the same rounding of the
experts: no loop in 14 long answers, and the fact found at every size.

## Speed

Through the dashboard on 2x Tesla V100 32 GB with 30 GB of RAM: up to 40 tokens/s decode (answering) and up to 560 tokens/s
prefill (reading the prompt), and the speed holds at long context (the DSA attention's pool selection is linear in the context:
0.07 ms per layer at 64K tokens, where it was 21.5 ms).
