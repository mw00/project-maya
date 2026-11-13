# Running Inkling-2-Small - phase 1, the specification

What an Inkling 2 engine in Maya starts from: the model exactly, the file format it reads, a reference forward
pass every kernel is parity-tested against, and how each op maps onto what Maya's engines already have. Written
2026-11-13 against `main` c5842c5 (v1.1.57); the request is issue #974.

## 1. The model, exactly

[thinkingmachines/Inkling-2-Small](https://huggingface.co/thinkingmachines/Inkling-2-Small) (Apache 2.0, released
2026-11-12), `Inkling2ForConditionalGeneration`, transformers' `inkling2` (in its main from 109a7d19bf, no release
yet):

| | |
| --- | --- |
| Total / active | 594 B / ~17 B a token (dense ~4.7 B, experts ~12.2 B) |
| Layers | 48, every one MoE; 40 sliding (window 1,024) and 8 full (5, 11, .., 47) |
| Attention | GQA 48 / 8 heads of 128; **no RoPE**: a relative bias (each head's 16 states mixed into a bias by distance, out to 1,024); a sink logit a head a token; a sigmoid gate on the output; q's scaled L2 clip and k's (22.6), v's weightless RMS norm; the full layers' log scaling of q and the relative states past 128K positions |
| Short convs | 4 taps on k and v (a full layer's dynamic: static taps + per-token taps from the input), after the attention and after the MoE |
| Attention residuals | blocks of 6 layers: a learned query (a full layer's 3: q, k, v) attends over the block states and the stream |
| MoE | 384 experts, 8 a token, no shared expert, width 2,304; sigmoid router with a selection bias, the weights normalized x 8; SwiGLU with softcaps (gate 4, up 25) |
| Head | logits softcapped at 30, the first 200,058 of 201,024 rows real |
| Weights shipped | experts NVFP4 (layers 0-39: e2m1 x e4m3 per 16 x amax / 2688 per expert) and MXFP8 (40-47); the rest BF16 |
| Extras | a DSpark-style drafter (`mtp.safetensors`, 5 layers); a vision tower |

## 2. The file: llama.cpp's `inkling2` GGUF

The format is llama.cpp's Inkling 2 conversion (ggml-org/llama.cpp PR #32187, draft, stacked on Inkling 1's
#25731), which is what [unsloth/Inkling-2-Small-GGUF](https://huggingface.co/unsloth/Inkling-2-Small-GGUF) ships
(UD-IQ1_S 172 GB to UD-Q8_K_XL 387 GB). An engine that reads it runs unsloth's files and any of Maya's own.

`tools/inkling2_gguf.py` writes that layout from the checkpoint (its docstring has the layout in full). It was written
from the PR's conversion and checked against unsloth's files:

- `--plan --against`: its 1,260 tensors are unsloth's UD-IQ1_S's, name for name and dim for dim; every tensor it
  writes as F32 is F32 there, and the reverse; its `inkling2.*` keys are theirs, value for value.
- Every F32 tensor unsloth keeps unquantized and the shards here hold - the folded convs (sliding and full, static
  and dynamic), norms, clips' scales, relative projections, residual queries, the router and its bias, the NVFP4
  experts' scales - compared bit for bit with unsloth's bytes (fetched by range): 653 tensors, all equal.

The conventions a kernel must follow (the conversion folds them in):

- **A short conv is a plain dot product.** `w [C, taps]`, tap taps-1 the current token, the residual and the output
  scale inside: `y[t] = sum_j w[c, j] x[t - 3 + j]`. A full layer's k / v add `sum_lag dyn[t, h(c) * 4 + lag]
  x[t - lag]` with `dyn = shortconv_{k,v}_dyn @ x_in` (lag 0 the current token, already scaled).
- **An NVFP4 expert tensor has a scale beside it.** `ffn_{gate,up,down}_exps.scale` (F32, one per expert), applied
  after the dot product - to gate and up before the activation. unsloth's requantized experts (IQ1_S, IQ2_XXS,
  IQ3_XXS) keep it for layers 0-39; layers 40-47 have none (the BF16 conversion folded MXFP8's exactly).
- `attn_res_score` (layer l's: the query before layer l's attention; a full layer's three), `ffn_res_score`,
  `output_res_score`: the attention residuals' queries. `exp_probs_b.bias`: the router's selection bias (for the
  pick only, not the weights).
- Not in the file: the drafter and the vision tower (the PR leaves them out; unsloth ships the tower as `mmproj`).

## 3. The reference: `ref/inkling2.py`

The forward pass over any `inkling2` GGUF, op by op in float64, in ref/glm.py's conventions (weights `(out, in)` from
`ref/load.py`, which now reads NVFP4 and unsloth's IQ types through gguf-py). Each intermediate a kernel produces is
recorded under a seam name (`blk.L.k_conv`, `.q`, `.k`, `.v`, `.r`, `.attn_gated`, `.attn_out`, `.picks`,
`.ffn_out`, `.out`, `logprobs`; `--dump-dir`), so a kernel is compared a seam at a time. Its docstring states the
forward in full.

`tests/inkling2_ref_test.py` holds it to transformers end to end: a small random Inkling 2 (12 layers, 2 full,
NVFP4 and MXFP8 experts as the checkpoint stores them, BF16 dense) written as the checkpoint, converted by
`tools/inkling2_gguf.py`, read back by the reference. Over 300 tokens against a window, a relative extent of 64 and
a log-scaling floor of 100: **max |dlogp| 6.4e-6, max KL 1.6e-12, top-1 100%** - fp32's rounding in the transformers
side. Each of these breaks fails it: the dynamic conv's taps by tap instead of lag, the sink dropped, the log
scaling dropped, an expert's scale dropped, the stream not restarted at a block boundary, the residual norm's eps,
the window off by one, the selection bias in the weights.

Needs transformers' main with `inkling2` (skipped without it) and gguf-py with NVFP4 (llama.cpp's).

### 3.1 On the real model

`tests/inkling2_real_check.py`: the real Inkling-2-Small, converted by `tools/inkling2_gguf.py` (11 files, 484 GB -
unsloth's BF16 is 484.4 GB; layers 40-47's MXFP8 experts doubled into BF16), through the reference, against
transformers' own decoder layers over the original checkpoint (fp32 on one RTX 5090, a layer at a time; every
expert decoded from the checkpoint's NVFP4 / MXFP8 bits). 256 tokens of text written for the test, all 48 layers:

| | transformers, the checkpoint | ref/inkling2.py, the GGUF |
| --- | --- | --- |
| perplexity (the next token right: 61.2%) | 3.597 | 3.597 |
| max \|dlogp\| against transformers | | 4.7e-5 |
| KL against transformers, mean / max | | 2.6e-12 / 1.4e-11 |
| the same top-1 token | | 100% of positions |
| time | 230 s (GPU) | 1,899 s (CPU, float64) |

And transformers' greedy answer, the same weights, to "In two sentences: why does a short convolution before
attention help a language model?" at thinking effort `none` (32 tokens, cut by the budget):

> <|message_model|><|content_text|>A short convolution before attention helps a language model by capturing local
> patterns and nearby token relationships that attention might otherwise miss. This gives the model a stronger

## 4. What maps onto Maya's engines

| Op | In Maya | New |
| --- | --- | --- |
| Dense matrices (Q6_K / Q8_0 / BF16 in unsloth's) | every engine's mat-vec / GEMM | - |
| Experts (IQ1_S / IQ2_XXS / IQ3_XXS / IQ4_XS; NVFP4 in the BF16 file) | the expert cache and its tiers (VRAM / RAM / SSD), the IQ kernels | the per-expert scale on gate / up / down; an NVFP4 decode for the BF16 / Q8 files |
| Router: sigmoid + selection bias, normalized x 8 | GLM / DeepSeek's `exp_probs_b` routers | - |
| SwiGLU with softcaps | the expert activation | 4 tanh(g / 4) sigmoid(g) 25 tanh(u / 25) |
| Attention: GQA 6:1, sliding window, sigmoid output gate | MiniMax-M3.1's GQA + window (mm3) | the relative bias (a head's 16 states . rel_proj[:, d]), a sink logit a head a token, the L2 clips, log scaling on full layers; no RoPE |
| Short convs | - | a 4-tap causal conv (state: 3 rows a conv a layer); the full layers' dynamic taps |
| Attention residuals | - | a softmax over <= 9 normed states a token (block states kept per token: 8 x 4,608) |
| Head | softcapped logits (GLM) | the 200,058-row slice |
| Drafter | DSpark (mm3_dspark, DeepSeek's) | Inkling's own weights (`mtp.safetensors`, not in the GGUF) |

The KV cache is small: the 40 sliding layers keep 1,024 positions (~170 MB at FP16 for all of them, any context);
the 8 full layers ~32 KB a token at FP16 - 1M context is 32 GB FP16, 16 GB INT8. A sequence's other state: four
convs' last 3 inputs a layer, and each position's block states (8 x 4,608) while it is in a prompt.

## 5. Memory

unsloth's smallest (UD-IQ1_S, 172 GB) is over 128 GB of VRAM + RAM, so as Maya does for DeepSeek the experts not
in memory are read from the SSD. A ~150 GB pack in the shape of Maya's DeepSeek XS (friendly expert types, the
dense weights GPTQ'd at 5-6 bits) is the target for a 160 GB pool; unsloth's KLD (0.092 against their BF16 file,
which keeps the experts' original bits) is the bar.

## 6. Next

1. Kernels for the new ops, each against the reference's seams: the attention (relative bias, sink, window,
   clips, log scaling), the convs, the attention residual, the expert scale + softcapped SwiGLU.
2. The engine: weights from the GGUF, the expert tiers, KV, decode and prefill.
