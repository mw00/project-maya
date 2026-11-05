# Maya-S v2 against the FP8 model (2026-10-07, Uranus)

`GLM-5.3-Flash-Maya-S-v2-IQ2_XXS` (96.5 GB, 3 shards): the same IQ2_XXS gate/up experts with error-feedback rounding as
v1, the down projections one step up everywhere - IQ3_XXS in the first and last four MoE layers (3-6, 41-44), IQ2_S in
the rest - and the rest as v1 (Q6_K attention, shared experts, dense layers, embeddings and output; Q8_0 small KDA
projections and the DSA indexer; F32 router and mixing weights; the NextN block with Q2_K/Q3_K experts).
Recipe: `tools/maya_quant/recipes/maya-s-v2.json`, from v1's calibration statistics; 255 min on Uranus.

## Distribution match on held-out text

The same yardstick as v1 ([MAYA-S-V1.md](MAYA-S-V1.md)): the FP8 model's top-64 log-probabilities on 8 held-out
sequences (7,672 positions), Maya-S through the engine's one-token decode path (`tools/maya_quant/kl_eval.py`).

| Seq | Text | KL v1 | KL v2 | Same top token v1 | v2 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | chat | 0.335 | **0.290** | 84.6% | **85.8%** |
| 3 | chat | **0.161** | 0.163 | 86.7% | **87.5%** |
| 6 | reasoning trace | 0.492 | **0.422** | 80.7% | **83.3%** |
| 9 | reasoning trace | 0.069 | **0.061** | **90.2%** | 89.3% |
| 12 | web code (HTML/JS) | 0.286 | **0.256** | 88.5% | **90.1%** |
| 15 | other code | 0.455 | **0.450** | 83.8% | **84.8%** |
| 18 | tool calls | 1.026 | **0.850** | 75.0% | **76.5%** |
| 21 | wikitext-2 (largely memorized by FP8) | 1.056 | **0.934** | 65.2% | **68.8%** |
| **all** | | 0.485 | **0.428** | 81.8% | **83.3%** |
| all but wikitext | | 0.403 | **0.356** | 84.2% | **85.3%** |

| | FP8 | v1 | v2 |
| --- | ---: | ---: | ---: |
| Perplexity | 3.51 | 4.31 | **4.19** |
| Top-1 accuracy on the actual next token | 71.5% | 68.1% | **68.8%** |
| Size | | 90.1 GB | 96.5 GB |

Reading: the down projections were where v1's error was largest for the size; one step up there (+6.4 GB) cuts the
KL by 12% overall and by 17% on tool calls, the format v1 matched worst.
