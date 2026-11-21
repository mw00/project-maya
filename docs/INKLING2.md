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
- **An absent `.scale` is 1.0, except beside NVFP4.** The scale only carries information beside NVFP4 experts, whose
  blocks have a fixed range. A requantized expert (IQ\*, Q\*_K) can fold the amax into its blocks, and some quants do:
  `peasantsmith/Inkling-2-Small-GGUF` (IQ2_XXS-mix) has 1,140 tensors and no `.scale`, where unsloth's have 1,260
  (3 x 40 layers more). A reader takes a missing `.scale` beside a non-NVFP4 expert as 1.0 - exact for such a file -
  and refuses an NVFP4 expert without one (its values would be wrong by its amax).
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

## 6. Phase 2: the engine, `maya-ik2`

`src/ik2/` runs unsloth's GGUFs in place (UD-Q4_K_XL: NVFP4 experts in layers 0-39; UD-Q3_K_XL: IQ3_XXS / IQ4_XS),
CUDA on Linux, Ampere on. Built by CMake beside the other engines (`maya-ik2`, and `ik2_kernels_test` with
`STRATA_BUILD_TESTS`); a 90 in `CMAKE_CUDA_ARCHITECTURES` builds sm_90a for the Hopper attention.

**Where the weights sit.** The dense weights (attention, router, head: ~4.7 B) are decoded once to FP16 in VRAM. The
routed experts stay as the GGUF holds them, all of them in pinned host memory (the files are read and registered
in place, a layer at a time: 48 s from NVMe for UD-Q4_K_XL), and the GPU keeps a cache of them in what VRAM is left
(`--cache-gb`). The host needs the experts' size in RAM: ~323 GiB for UD-Q4_K_XL, ~260 GiB for UD-Q3_K_XL.

**Prompts** go through layer by layer in passes of up to 16,384 tokens, so a pass copies each expert it misses
once; the next layer's likely experts are copied while this one runs, and a pass evicts its own experts first
(scan resistance) instead of flushing the cache. `IK2_WARM` fills the cache at load with the experts tokens pick
most (a profile written with `IK2_POP_OUT`). The attention is one fused kernel (relative bias, sinks, sliding window
and clips; FP16 MMA, and wgmma on Hopper).

**Decode** is bound by the PCIe link (~47 expert misses a token on the H200). `IK2_CPU=1` computes a missing
expert on the CPU when that is sooner than its copy (NVFP4 experts, AVX-512).

**Two GPUs** (`--gpus 2`): `IK2_EP=1` deals each layer's experts to both by how often they are picked
(`IK2_EP_SPLIT`, GPU 1's share), each GPU with its own cache and link; attention and the KV cache stay on GPU 0.
Without it the layers are split between the GPUs.

**Serving.** `--serve` speaks Maya's engine protocol (`serve/server.py`'s `StrataEngine`: `INFO`, `READY <ctx> stop`,
`GEN`, `PP`, `T`, `DONE`, `STOP`), keeps the conversation's cache for its next turn, and ends a request at
`<|content_model_end_sampling|>` and `<|endoftext|>` (`--eos` to change). The server side:

- `tools/ik2/chat_template.jinja` is thinkingmachines' own template (the GGUF carries the conversion's, which
  differs). `ChatTemplate` recognizes it (`.inkling`): the thinking level by its names (`none` .. `max`; thinking off
  is `none`, Auto its default 0.9), a past call's arguments given as a JSON string (OpenAI's form; the template
  refuses one) parsed into an object, `None` content as "".
- The model writes typed segments rather than tags - `<|content_thinking|>`, `<|content_text|>`, a call as
  `<|message_model|>NAME<|content_invoke_tool_json|>{"name": .., "args": {..}}<|end_message|>`.
  `InklingRewriter` turns them, as they stream, into what `OutputParser` reads (DeepSeek's and MiniMax's path).
- A required tool call (`tool_choice`) opens in its own form; a picture is refused in plain words (no vision
  tower yet).
- `tools/strata_tokenizer.py` reads the tokenizer under both names unsloth's files give its pre-tokenizer (`gpt-4o`,
  `inkling`): 442 strings against the model's own `tokenizer.json`, all equal, with both files.

To run it (no setup in `maya.py` yet):

    python tools/strata_tokenizer.py --gguf Inkling-2-Small-UD-Q4_K_XL-00001-of-00009.gguf --out ik2 \
        --chat-template tools/ik2/chat_template.jinja --check
    cat > ik2.json <<EOF
    {"exe": "build/maya-ik2", "args": ["--model", "Inkling-2-Small-UD-Q4_K_XL-00001-of-00009.gguf",
     "--max-context", "65536", "--cache-gb", "200"], "tokenizer": "ik2/tokenizer", "model_name": "inkling-2-small"}
    EOF
    python serve/server.py --engine maya --config ik2.json

### 6.1 Measured

UD-Q4_K_XL, KV in FP16, the experts in pinned host memory, PCIe 5:

| | H200 (141 GB) | RTX PRO 6000 (96 GB) | 2x RTX PRO 6000, `IK2_EP=1` |
| --- | --- | --- | --- |
| Expert cache | ~114 GiB | ~67 GiB | one on each GPU |
| Decode | 35-36 tok/s (`IK2_CPU=1`: 36.2) | 22 tok/s | 42 tok/s |
| A 21.8k-token prompt, fresh | 5.9 s | 7.7-8.0 s | 7.25-7.8 s |
| A 105k-token prompt | 19.3 s | | |

(Two GPUs read a prompt hardly faster: nsys shows each idle over a third of the time, waiting on its copies - the
link is busy only 61% - with the host's per-layer work between them. Decode nearly doubles.)

UD-Q3_K_XL on the H200: decode 44.6 tok/s (its experts are smaller: fewer misses, more of them cached).

Against `ref/inkling2.py` (float64) over the same GGUF, 48 tokens: layers 0-5 within 3e-4 relative, every top-1
token the same, KL mean 8.5e-4 (max 7.2e-3). With `IK2_KV=fp8` 5 of the 48 top-1 tokens differ (KL mean 8.3e-3):
Inkling's router is full of near-ties, and FP8's 1.5% in layer 0's attention output already changes 7 tokens'
experts there - so FP16 is the default.

`ik2_kernels_test` holds each kernel (attention in its forms, the convs, the attention residual, routing, the
expert decodes and their softcapped SwiGLU) to a double-precision reference on the CPU; no model needed.

What did not help (kept out): prefetching two or three layers ahead, or before the routing round trip (both crowd
the cache: a 21.8k prompt on an RTX PRO 6000 went 8.05 -> 9.7 s); predicting the next layer's experts from this
layer's input (46% precise); tensor parallel experts (`IK2_TP=1`: 42.8 vs EP's 41.3 tok/s but slower prompts; kept
as an option); the drafter (`--draft`: 2.4-2.7 tokens a step, but verifying 8 costs ~260 ms).

### 6.2 On AMD: `maya-inkling2`

A second engine for the same model, for AMD cards (HIP; `src/inkling2/`, built only with `STRATA_ENABLE_HIP`):
[INKLING2_AMD.md](INKLING2_AMD.md). It reads a pack of unsloth's UD-Q2_K_XL (`tools/inkling2/pack_gguf.py`) on
maya-ds4's ggml block-type kernels, the layers split across the cards, the experts VRAM does not hold computed on
the CPU, and it drafts with the model's own DSpark block ([INKLING2_DSPARK.md](INKLING2_DSPARK.md)): on 4x Radeon AI
PRO R9700, 37 tok/s, and 58 with drafts.

Both engines are the one family `ik2` to the server and the web app (`tools/maya_quants.py`'s `family_of`: by the
engine's name, `maya-ik2` or `maya-inkling2`), with everything in "Serving" above: this section's template, parser
and tokenizer. What the server gives each engine differs only in its command line (`engine_args`: the layer split
for `maya-inkling2`; `maya-ik2`'s GPUs are its own `--gpus`).

## 7. Next

1. Setup in `maya.py` (download, the RAM / VRAM check, the run config), as Hy4-Flash has.
2. Several requests at once; the vision tower (unsloth's `mmproj`).
3. A Maya quant of the experts in the shape of DeepSeek XS (section 5), for 160 GB of host memory.
