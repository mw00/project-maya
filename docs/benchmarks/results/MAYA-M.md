# Maya-M - results

Maya-M (116 GB, `Maya-M/` on [Hugging Face](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF)) is Project
Maya's larger quant of GLM-5.3-Flash, made from Z.ai's FP8 release with statistics from the FP8 model itself:

| Tensors | Type |
| --- | --- |
| routed experts' gate/up | IQ2_S, error-feedback (GPTQ-style) rounding with the FP8 model's activation statistics |
| routed experts' down | IQ3_XXS; IQ3_S in the most sensitive MoE layers (the first and last four) |
| attention, shared experts, dense layers | Q6_K (the KDA gates and the indexer Q8_0, the router F32) |
| the NextN (MTP) draft block | Q3_K / Q4_K experts |

The calibration text is weighted toward tool calls and front-end code (HTML/CSS/JS, three.js, canvas), the work
Maya-M is meant for. Recipe: `tools/maya_quant/recipes/maya-m.json`.

## Against the FP8 model, token by token

The same 8 held-out texts (7,672 tokens) through the engine, every next-token distribution compared with the FP8
model's own:

| | KL divergence vs FP8 (lower is better) | same top token as FP8 |
| --- | ---: | ---: |
| **Maya-M** (116 GB) | **0.329** | **86.2%** |
| Maya-S (96.5 GB) | 0.428 | 83.3% |

## Zero-shot accuracy

**Maya-M keeps 97.9% of the full FP8 model's zero-shot accuracy** - the same as Maya-S (97.9%): multiple-choice
tasks at 400 questions each do not separate the two (a point or two either way is within the test's noise), while
token by token (above) Maya-M is clearly closer to the FP8 model.

| Task (zero-shot) | FP8 | Maya-M | Recovery |
| --- | ---: | ---: | ---: |
| ARC-Easy (acc) | 87.2 | 86.5 | 99.1% |
| ARC-Challenge (acc norm) | 71.0 | 69.0 | 97.2% |
| HellaSwag (acc norm) | 88.5 | 86.8 | 98.0% |
| WinoGrande (acc) | 78.5 | 76.2 | 97.1% |
| PIQA (acc norm) | 87.0 | 85.0 | 97.7% |
| **Average** | **82.5** | **80.7** | **97.9%** |

The same 400 questions per task for every model, scored the way lm-evaluation-harness scores them (the answer with
the highest log-likelihood; length-normalized where the choices differ in length) - the FP8 model run layer by layer
in PyTorch, Maya-M through Project Maya's engine (`tools/maya_quant/zs_*.py`). Maya-S on the same questions:
[MAYA-S.md](MAYA-S.md).

## Tried and not kept

An ISTA-DASLab-style refinement on top (each routed expert's group scales refitted to match the FP8 layer's output,
codes held fixed, by least squares on calibration activations) lowered the layers' reconstruction error by 5-38% and
the KL a little (0.323), but predicted held-out text slightly worse (4 of 8 texts better, 4 worse). The released
Maya-M is the quant without it.
