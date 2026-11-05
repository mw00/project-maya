# Maya-L - results

Maya-L (156.3 GB, `Maya-L/` on [Hugging Face](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF)) is Project
Maya's largest quant of GLM-5.3-Flash: Maya-M's recipe one step up, made from Z.ai's FP8 release with Maya-M's
statistics from the FP8 model itself:

| Tensors | Type |
| --- | --- |
| routed experts' gate/up | IQ3_S, error-feedback (GPTQ-style) rounding with the FP8 model's activation statistics |
| routed experts' down | IQ4_XS; Q5_K in the most sensitive MoE layers (the first and last four) |
| attention, shared experts, dense layers | Q6_K (the KDA gates and the indexer Q8_0, the router F32) |
| the NextN (MTP) draft block | Q4_K / Q5_K experts |

The calibration text is Maya-M's, weighted toward tool calls and front-end code (HTML/CSS/JS, three.js, canvas).
Recipe: `tools/maya_quant/recipes/maya-l.json`.

## Against the FP8 model, token by token

The same 8 held-out texts (7,672 tokens) through the engine, every next-token distribution compared with the FP8
model's own:

| | KL divergence vs FP8 (lower is better) | same top token as FP8 | top-1 on the actual next token |
| --- | ---: | ---: | ---: |
| FP8 | 0 | 100% | 71.5% |
| **Maya-L** (156.3 GB) | **0.188** | **90.0%** | **71.2%** |
| Maya-M (116 GB) | 0.291 | 86.8% | 70.3% |
| Maya-S (96.5 GB) | 0.428 | 83.3% | 68.8% |

Maya-L and Maya-M through the same engine (v1.0.15's), one after the other: **35% lower KL divergence** than
Maya-M. Maya-S's row is its own release's measurement, with an earlier engine.

## Zero-shot accuracy

**Maya-L keeps 99.2% of the full FP8 model's zero-shot accuracy**, with the same score as FP8 on HellaSwag and PIQA:

| Task (zero-shot) | FP8 | Maya-L | Recovery | Maya-M |
| --- | ---: | ---: | ---: | ---: |
| ARC-Easy (acc) | 87.2 | 86.0 | 98.6% | 86.5 |
| ARC-Challenge (acc norm) | 71.0 | 69.8 | 98.2% | 69.0 |
| HellaSwag (acc norm) | 88.5 | 88.5 | 100% | 86.8 |
| WinoGrande (acc) | 78.5 | 77.8 | 99.0% | 76.2 |
| PIQA (acc norm) | 87.0 | 87.0 | 100% | 85.0 |
| **Average** | **82.5** | **81.8** | **99.2%** | **80.7 (97.9%)** |

The same 400 questions per task for every model, scored the way lm-evaluation-harness scores them (the answer with
the highest log-likelihood; length-normalized where the choices differ in length) - the FP8 model run layer by layer
in PyTorch, Maya-L through Project Maya's engine (`tools/maya_quant/zs_*.py`).

## Long answers

No loops: a single-file three.js scene asked for at temperature 1.0 and greedily (greedy is where a quant loops first),
14,000 tokens each, with no repeated passage (longest repeat a single 32-token run, 0% repeated 64-token runs). Both
were still planning at 14,000 tokens - the test runs at GLM's Max thinking level.

## What it needs

Maya-L is the largest of the four and the most demanding: like every Maya quant it runs with the experts
spread over VRAM, RAM and the SSD, and it is fastest when VRAM and RAM together hold most of its 156 GB - what does
not fit is read from the SSD while it answers. With less memory, Maya-M (116 GB) or Maya-S (96.5 GB) answer faster.
