# tools/ling_quant: Maya quants of Ling-3.2-air

Builds and checks quants of [inclusionAI/Ling-3.2-air](https://huggingface.co/inclusionAI/Ling-3.2-air) for
`maya-ling` (llama.cpp's `bailingmoe3` GGUF layout). It is the `tools/maya_quant` method applied to Ling:

- importance statistics per expert from Ling's own layer code;
- error-feedback (GPTQ-style) rounding of the experts' gate/up, and importance-weighted rounding of down;
- non-expert weights quantised from the FP8 release in fp32, not from a BF16 copy;
- every tier scored against the FP8 model itself.

## Tiers

| Tier | Size | Routed experts (gate/up, down) | Other weights | Aimed at |
| --- | ---: | --- | --- | --- |
| XS | 30.3 GB | IQ1_S, IQ2_XXS | Q5_K | 12 GB GPU + 16 GB RAM (RTX 3060) |
| S | 46.0 GB | IQ2_S, IQ3_XXS | Q6_K | 24 GB GPU + 32 GB RAM |
| M | 61.8 GB | IQ3_S, IQ4_XS | Q6_K | 24 GB GPU + 48 GB RAM |
| L | 77.6 GB | IQ4_XS, Q5_K | Q8_0 | 24 GB GPU + 64 GB RAM |

In every tier, the first two and last two MoE layers (2, 3, 40, 41) get one step more. The router, SSM/KDA scalars
and convolutions stay F32. The KDA beta, attention gate and MLA k_b/v_b are Q8_0. The MTP layer (blk.42) only drafts, so
its experts sit one step below the body's (the same as the body in XS). `recipes/ling-*.json` holds the exact rules: regex to type, first match wins.

## Accuracy

Weights-only, 32 held-out sequences of 2048 tokens (chat, multilingual, reasoning, web code, other code, tool calls,
wikitext). KL is against the FP8 reference's top 64 plus a tail bucket. Same top = how often the top token matches.

| | Size | KL | Same top |
| --- | ---: | ---: | ---: |
| FP8 through the eval path (noise floor) | 121 GB | 0.066 | 95.2% |
| BF16 GGUF | 255 GB | 0.078 | 94.9% |
| Q8_0 (llama-quantize) | 135.6 GB | 0.101 | 94.1% |
| L | 77.6 GB | 0.140 | 92.3% |
| M | 61.8 GB | 0.204 | 89.8% |
| S | 46.0 GB | 0.340 | 85.5% |
| XS | 30.3 GB | 0.654 | 78.7% |

The noise floor is not zero: Ling's 512-expert router has many near-ties, so tiny arithmetic differences change which
experts are chosen. Inside `maya-ling` itself (`ling_engine_kl.py`, 8 sequences), the tiers score L 0.089, M 0.144,
S 0.264 and XS 0.542. That is the same as the weights-only path on those sequences.

## Running it

```
LING_FP8=/path/to/Ling-3.2-air            # the FP8 release
MAYA_GGML_BASE=/path/to/libggml-base.so   # from a llama.cpp build (build/bin/)
LLAMA_CPP=/path/to/llama.cpp              # for convert_hf_to_gguf.py
LING_CALIB_DATA=/path/to/datasets         # the calibration sources, as tools/maya_quant/prep_calib.py reads them
tools/ling_quant/run.sh                   # [TIERS="s m"] [LING_WORK=ling-work] [THREADS=64]
```

`ling_paths.py` lists every environment variable the tools read. `run.sh` runs four steps and skips any whose output
already exists:

1. `prep_calib.py`: 256 calibration and 32 evaluation sequences of 2048 tokens, rendered with Ling's chat template
   (thinking on), from the same sources as `tools/maya_quant/prep_calib.py`.
2. `ling_calib.py`: the FP8 model in fp32, one layer at a time. It writes each expert's input statistics, each MoE
   layer's inputs (for the GPTQ pass), and the reference's top-64 log-probabilities on the evaluation set
   (`ref_topk.npz`, plus `ref_mtp_topk.npz` for the MTP layer). It was run on one 96 GB GPU, and the saved MoE inputs take
   about 115 GB of disk.
3. llama.cpp's `convert_hf_to_gguf.py`: the BF16 GGUF. Only the experts are read from it; everything else comes from
   FP8.
4. `ling_quantize.py` per tier, then `ling_eval.py`. Output over 48 GB is split into llama.cpp shards
   (`-00001-of-0000N`); `maya-ling` loads those with the split-GGUF support from alesha-pro/project-maya#4.

## Other checks

- `ling_engine_kl.py`: a GGUF scored inside `maya-ling` (`LING_LOGITS_OUT`), in the same way as `ling_eval.py`.
- `ling_zs.py`: zero-shot ARC-Easy/Challenge, HellaSwag, WinoGrande and PIQA (`tools/maya_quant/zs_tasks.py` requests),
  plus how often a quant's answer matches the FP8 model's.
- `ling_loop_test.py`: `tools/maya_quant/loop_test.py`'s long-answer prompts in `maya-ling`, sampled and greedy:
  finished, thinking closed, share of repeated text.
- `ling_map.py verify`: every GGUF tensor name and transform checked against the FP8 checkpoint.
- `ling_diff.py`: the reference and evaluation paths layer by layer, to find where two paths part.
- `ling_gptq_test.py`: error-feedback against plain importance rounding on one layer's experts.
- `by_category.py`: `ling_eval.py`'s per-sequence rows broken down by source.

The reference's details:

- The SwiGLU clamp is applied after SiLU: `min(silu(g), L) * clamp(u, ±L)`, with limit 4 on the experts of layers
  35-41 and 5 or 7 on the shared experts of layers 34-41.
- Routing is grouped: 8 groups, the best 4 by the sum of each group's top two scores, then the top 8 experts by
  sigmoid score plus bias. The weights are normalised and scaled by 2.5.
