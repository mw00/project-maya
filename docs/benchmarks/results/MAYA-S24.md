# Maya-S24 - results

Maya-S24 (94.7 GB, `Maya-S24/` on [Hugging Face](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF)) is
Maya-S with the weights every token uses in 4-bit instead of 6-bit, for GPUs with 24 GB or less.

| Tensors | Maya-S | Maya-S24 |
| --- | --- | --- |
| routed experts (95% of the model) | IQ2_XXS gate/up, IQ2_S / IQ3_XXS down | Maya-S's, byte for byte |
| attention projections (KDA q/k/v/output, MLA q_a/q_b/kv_a/output), shared experts | Q6_K | **Q4_K** |
| attention k_b / v_b (read as BF16 by the engine), dense layers, embeddings, output | Q6_K | Q6_K |
| KDA gates, the DSA indexer / router, norms | Q8_0 / F32 | Q8_0 / F32 |

**Why:** the weights every token uses stay on the GPU.
- In Maya-S they take 6.86 GB, 29% of a 24 GB card such as an RTX 3090 or 4090. That leaves fewer experts in VRAM, so more of each token's experts come from RAM or the SSD.
- In 4-bit they take about 1.5 GB less, room for about 200 more experts.

**How it was made:**
- The 4-bit tensors were quantized straight from Z.ai's FP8 release, with the same calibration statistics as Maya-S (an importance matrix from the FP8 model).
- They were then put in place of Maya-S's 6-bit ones; every other tensor is Maya-S's, byte for byte. Recipe: `tools/maya_quant/recipes/maya-s24.json`.

## Speed

One Tesla V100 32 GB, 64 GB RAM, the same engine for both models (`--bench`-style: five questions, 256 tokens each).
"24 GB" is the same card limited to 23.5 GB (`STRATA_GLM_VRAM_GB=23.5`):

| | Maya-S | Maya-S24 |
| --- | ---: | ---: |
| decode (writing the answer), 24 GB | 11.8 tokens/s | **13.5 tokens/s (+14%)** |
| decode, 32 GB | 17.2 tokens/s | **19.1 tokens/s (+11%)** |
| prefill (reading the prompt), 2k / 8k / 16k tokens, 32 GB | 267 / 373 / 362 tokens/s | 268 / 368 / 358 tokens/s |

## Against the FP8 model

| | KL divergence vs FP8 | same top token as FP8 | top-1 on the actual next token |
| --- | ---: | ---: | ---: |
| Maya-S | 0.421 | 83.3% | 68.6% |
| **Maya-S24** | **0.444** | **83.1%** | **67.9%** |

The same 8 held-out texts (7,672 tokens) for both, in the same run. (Maya-S measured 0.428 with an earlier engine; the
small difference is the engine's.)

| Task (zero-shot) | FP8 | Maya-S24 | Recovery |
| --- | ---: | ---: | ---: |
| ARC-Easy (acc) | 87.2 | 85.8 | 98.3% |
| ARC-Challenge (acc norm) | 71.0 | 69.0 | 97.2% |
| HellaSwag (acc norm) | 88.5 | 84.8 | 95.8% |
| WinoGrande (acc) | 78.5 | 77.0 | 98.1% |
| PIQA (acc norm) | 87.0 | 86.2 | 99.1% |
| **Average** | **82.5** | **80.6** | **97.7%** |

Maya-S keeps 97.9% on the same 2,000 questions ([MAYA-S.md](MAYA-S.md)). Questions and scoring as for Maya-S and
Maya-M: lm-evaluation-harness style, the FP8 model in PyTorch, the quants through Project Maya's engine.

**Which one:**
- **Maya-S24** on cards of 24 GB or less, where the speed matters most.
- **Maya-S** where the last fraction of quality matters more than about 11% decode speed.
- **Maya-M** where RAM and VRAM together hold 116 GB.
