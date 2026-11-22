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
| Extras | a DSpark-style drafter (`mtp.safetensors`, 5 layers: drafting, set up by default); a vision tower |

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
CUDA on Linux, Volta (sm_70) on. Built by CMake beside the other engines (`maya-ik2`, and `ik2_kernels_test` with
`STRATA_BUILD_TESTS`); a 90 in `CMAKE_CUDA_ARCHITECTURES` builds sm_90a for the Hopper attention; `all` / `all-major`
take the compiler's list from sm_70 on. CUDA 12.x for a V100 (CUDA 13 has no sm_70).

**Kernels per architecture** (`k::arch_class()`; a fix for one class leaves the others' code as it is):

| GPU class | One token's attention | NVFP4 experts (one token) | Prompt attention | Verify window (drafting) |
| --- | --- | --- | --- | --- |
| Ampere and newer (sm_80+) | `mma.sync` m16n8k16 | `mma.sync` m16n8k16 | fused (`mma.sync`, `cp.async`; sm_90a: wgmma) | block path |
| Turing (sm_75) | Ampere's, m16n8k16 as two m16n8k8 | Ampere's, the same | cuBLAS (the fused kernel needs `cp.async` and more shared memory) | block path; drafting off (a NOTE) until its window is checked against the token path on a Turing card |
| Volta (sm_70) | its own: `mma.m8n8k4`, 32 keys a warp step | its own: `mma.m8n8k4` | cuBLAS | none: drafting off (a NOTE) |

Each class is exact within itself (the same greedy answers start after start); the classes differ from each other in
their sums' order, as different kernels do. `IK2_ARCH=70` / `75` takes an older class's kernels on a newer GPU (tests;
Turing's code only where the build has it - a `75-virtual` build).

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

**Two GPUs** (`--gpus 2`, outside the preview of section 8: with `IK2_UNSUPPORTED=1`): `IK2_EP=1` deals each layer's experts to both by how often they are picked
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
- A required tool call (`tool_choice`) opens in its own form; a picture is refused in plain words when the config
  has no image encoder (section 8: pictures).
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

## 7. Next

1. Setup in `maya.py` (download, the RAM / VRAM check, the run config), as Hy4-Flash has - done for Inkling-2-Mini as
   an experimental preview (`maya.py --setup --model inkling-2-mini`: docs/MODELS.md); Inkling-2-Small by hand still.
2. Several requests at once; the vision tower inside the engine (pictures work through `maya-vision`: section 8).
3. A Maya quant of the experts in the shape of DeepSeek XS (section 5), for 160 GB of host memory.

## 8. Inkling-2-Mini: the looped model, whole on one 24 GB card (smaller NVIDIA cards: the rest of the experts in RAM)

[Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (26 B, Apache 2.0) is the family's small
model with a loop in it: 24 layers of weights, of which layers 6..17 run three times - 48 layer visits a token.
Width 2304, 24 heads on 8 K / V heads (D 128), 128 experts a layer (8 a token, F 1152), a window of 1024 keys in all
layers but 5, 11, 17 and 23, the same relative bias, sinks, clips, short convs and attention residuals (block 6).

**The loop** (transformers PR #51050, llama.cpp PR #32695). Each pass of the core starts again from the stream and
the block states the core was first given. A pass after the first reads the pass before it through four messages a
layer - for q, k, v and the MoE - each an attention residual over the earlier pass's block states and its final
stream (`loop_res_query`, `loop_kv_res_query`), let into the layer's input by a gate of rank 64:
`(1 - tanh(F g)) x + tanh(G g) m`, `g = D [rms(x); rms(m)]`. The weights are a layer's; the KV cache and the conv
histories are a visit's. `ref/inkling2.py` and `maya-ik2` keep a list of visits; with `num_loops` 1 it is the layers.

**The file.** unsloth's Mini GGUFs are in the layout llama.cpp master reads (other tensor and key names, the convs
not folded, gate and up merged into `ffn_gate_up_exps`); the reference and the engine read both layouts.
UD-Q4_K_XL (17.6 GB) holds the experts of layers 0-19 in NVFP4 - their original bits - and those of layers 20-23
and the dense weights in Q8_0. (UD-Q8_K_XL keeps layers 20-23's experts in BF16: not read here.)
Against llama.cpp (b12590 with the Inkling 2 PRs) the reference agrees on 47 of 47 top-1 tokens, KL 2.5e-3; the
engine against the reference on 45-46 of 47, KL 6e-3..7e-3 - near ties of the router, as section 6.1 found; with
the messages left out the KL is 0.56.

**Everything in VRAM: the resident token path.** The experts are 14.7 GiB, the dense weights 1.1 GiB as the GGUF's
Q8_0 blocks (taken apart: the values and the scales each on their own, `mv_q8p`), so on a 24 GB card every expert
stays in VRAM and a token needs nothing from the host between its embedding and its logits: an expert's blocks are
found on the device (a table of pointers, `route_token`), the position is read on the device, no cuBLAS, no
synchronisation - and a token's launches replay as one CUDA graph (`IK2_FAST`, `IK2_GRAPH`). The experts are copied
to VRAM straight from the mapped files (a reload takes 3 s). What a token costs then is its weights' bytes (3.7 GB:
half the experts', half the dense) and its launches - each 2 us and more whatever it does - so:

- the short convs ride the products before them (`mv_q8p_conv`), the MoE's conv the experts' combine, the sinks
  the clips' launch; a looped layer's q, k and v products are one launch (`MvAlt`), its three norms one, its four
  messages one; the router's picks and their tables one (`route_token`): 1404 -> 781 launches a token;
- the attention of a token runs on the tensor cores, a warp at a time straight from the cache into the registers
  (`attn_decode_mma`: no shared memory, so sixteen and more warps an SM hide the memory's latency). It reads the
  cache at the memory's speed - 846 GB/s FP16, 776 GB/s FP8 on an RTX 3090 - and an FP8 cache, half the bytes, is
  the faster one deep in a context. Its bytes are made FP16 in the registers without a branch (`e4m3x4_h4`);
- sampling takes its candidates from the device (`cand_topk`): a thousand logits and a few more instead of the
  vocabulary's 200,058 every token.
- the NVFP4 experts of a token run on the tensor cores straight from their blocks (`experts_gate_up_swiglu_mma`,
  `experts_down_mma`, `IK2_EXPERT_MMA`): a code times its sub-block's scale is exact in FP16, the sums are FP32, and
  the gate / up input goes in as two FP16 limbs in two of the eight columns, which is the FP32 input's product. A
  warp alone on 16 rows was slower than the scalar kernel (100 against 118 tok/s): it waits for a block's bytes
  before it asks for the next ones. Eight warps on the same 16 rows, each a stride of the row, read 745 GB/s;
- the attention residuals ask for four states at once, and a pass's messages for all twelve core layers are one
  launch (the bank does not change within a pass): 781 -> 757 launches a token, the same sums bit for bit.

With these three, same card and file, FP8 cache: 124.8 tok/s 47 tokens deep (was 111.9), 120.7 at 4,096 (108.6),
103.2 after a 63,981-token document (94.1), 89.1 after 127,910 (81.4). The scalar expert path
(`IK2_EXPERT_MMA=0`) gives the logits of before bit for bit; over 399 forced tokens the tensor-core experts are a
median KL of 5.7e-5 from it with 394 of the top tokens the same (the tensor-core attention against its scalar
kernel on that text: 1.7e-4 and 393), and on 96 GSM8K questions they solve 74 where the scalar ones solve 73.

A prompt goes the cache path's way with every expert a hit; the attention's flash kernel reads an FP8 cache too
(the tile's bytes by `cp.async`, FP16 of them in shared memory), and a looped layer's gates are cuBLAS products
over a chunk. A short prompt - a chat turn, up to 96 tokens (`IK2_RG_SLOTS`) - has its experts read by groups, an
expert's blocks once for up to 8 of its tokens, instead of decoded to FP16 for a GEMM: 12 new tokens take 56 ms
(were 118).

**Measured**, one RTX 3090 (24 GB, 300 W), UD-Q4_K_XL, tok/s:

| | llama.cpp b12590 | `maya-ik2`, FP16 cache | FP8 cache |
| --- | --- | --- | --- |
| A token, 47 tokens deep | 57 | 112 | 110 |
| 4,096 deep | | 106 | 107 |
| 65,536 deep | | 84 | 91 |
| 131,072 deep (context 262,144) | | | 81 |
| A prompt of 4,096 / 65,536 tokens | 1052 / - | 2526 / 2499 | 2504 / 2387 |
| Sampling at temperature 1 | | 110 | |

**Context.** The eight visits of the layers without a window keep 32 KB a token in FP16, 16 KB in FP8. With an FP8
cache every expert stays in VRAM up to `--max-context 352256` (text only; 22.3 GiB at 262,144), and 262,144 fits
beside the image encoder. Past that the engine falls back to the cache path (64 tok/s). An FP8 cache is as far
from the FP16 logits whichever kernels read it (KL 1.9e-3 on the tensor cores, 2.0e-3 scalar, over 199 forced
tokens; FP16 on the tensor cores 2.2e-5) - but any two FP8 runs part within a few tokens: a difference of 1e-5
flips codes of a 3-bit mantissa.

**Pictures** go through `tools/vision`'s `maya-vision` built against a llama.cpp that has the Inkling 2 tower
(`PROJECTOR_TYPE_INKLING2V`; Maya's own build adds it to the pinned llama.cpp: `tools/vision/inkling2v_patch.py`,
and needs no text model for it - the config's "vision" entry names none) and unsloth's `mmproj-BF16.gguf`. The
setup sets them up; the encoder starts when a picture arrives (on the GPU while its free VRAM holds it, else on the
CPU) and ends after the request's pictures, so it keeps no VRAM. The server sends the encoder's rows with the request
(`GENI`), the engine puts them in place of the image tokens (`--image-token`, 200054) without the embedding's norm.
640 x 448 is 192 tokens and 22 ms.

**The drafter** (`--draft mtp.safetensors`, the official DSpark checkpoint: 5 layers; its taps the stream at
visits 5, 23 and 41 and the final hidden, `mtp_config.target_layer_ids`) drafts after the last chosen token, and the
model checks the window - the anchor and the drafts - in one go on the **block path** (`fast_block`): the resident
token path for 2 .. 8 tokens, every position read from the device, the picks grouped by expert on the device (an
expert's blocks read once for the tokens that picked it: 5 tokens need 2.7 times one token's experts, not 5), the
convs' histories and inputs recorded and taken back in one launch for the rejected tokens - one CUDA graph a window
size. Greedy keeps the agreeing drafts and the model's next token; sampling accepts a draft with the model's
probability of it and otherwise samples the model's distribution without it, which is the model's distribution
exactly.

**Drafting on = drafting off.** The window gives every token the one-token path's values to the bit (the **exact
window**, the default): its dense products are `mv_q8p`'s sums for each token with the weights read once
(`k::mv_q8p_cols`), its attention the token's own kernels a token at a time, its experts the token kernels' sums (the
tensor cores' NVFP4 blocks with the same warp split and order; the row dots - the Q8_0 layers, and every layer
where the tensor-core kernels are off - a member at a time, an expert read once for its tokens). So greedy answers
are the same with drafting and without it, and sampled ones follow the same distribution. `IK2_WINDOW_CHECK=n`
compares the first n windows with the one-token path row for row. `IK2_BLOCK_EXACT=0` takes the tensor cores' FP16
tiles and the window's own attention instead (a few 1e-5 off: greedy answers then differ with drafting on; the
contributor's figures below were measured that way).

**The prompt lookup** (`src/spec/suffix_drafter`) drafts the tokens that followed the latest earlier copy of the
text's end - edits, quoted code, repeated lines - with no weights; the same window checks them. `--drafting`: **auto**
(the setup's default) measures once on the PC, before the first answer, which of no drafting, the prompt lookup, the
drafter and both answers fastest (a chat, a code and a tool-call request, `IK2_DRAFT_IDS`), and keeps it in
`data/ik2-drafting.txt` (per GPU, driver, model files, context and cache; `maya --speed-calibration` measures again);
**on** drafts with the drafter and the prompt lookup, the controller choosing each step; **off** loads neither. A
request's `spec=1` asks for no drafting. The drafter takes room only where the start plan stays as it is without it
(every expert in VRAM, the same context and cache, and not the free VRAM the pictures' encoder would start in on the
GPU - `IK2_VISION_GIB`, the server's), else it is left out with a NOTE: on a 24 GB card that is above about 200K tokens of
context with the FP8 cache (`tools/ik2_fit.py` `draft_gib`: 0.88 GiB + 4.75 KiB a token). The Monitor's Speculation
shows the drafts kept (STAT `draft_acc` / `draft_off`, `draft_ms`, `verify_ms`).

Whether a window pays depends on the text: the drafts are taken 1.8 tokens a step in prose, 3 in new code, 4.3 in a
document's edit. A controller decides each step (`SpecCtl`): a window of B's expected tokens, (1 - a^B) / (1 - a)
with a the recent acceptance, over its measured time, against a plain token's; after 8 plain tokens it drafts once
to look again. One RTX 3090, greedy, against plain decode: prose 131 / 127 tok/s, new code 162 / 126, an edit 213 /
124; through the server new code 207 tok/s, at temperature 0.7 209. The drafter's context: its windowed layers a
ring of 2048 rows, the layer without a window the whole context in FP16 (`IK2_DRAFT_CTX` caps it: a 64K document
continues at 3.07 tokens a step with the whole context, 2.24 with the last 16K).

To run it (one GPU):

    python tools/strata_tokenizer.py --gguf Inkling-2-Mini-UD-Q4_K_XL.gguf --out ik2 \
        --chat-template tools/ik2/chat_template.jinja --check
    cat > ik2-mini.json <<EOF
    {"exe": "build/maya-ik2", "args": ["--model", "Inkling-2-Mini-UD-Q4_K_XL.gguf", "--max-context", "262144"],
     "env": {"IK2_KV": "fp8"}, "tokenizer": "ik2/tokenizer", "model_name": "inkling-2-mini",
     "vision": {"exe": "build-vision/bin/maya-vision", "mmproj": "mmproj-BF16.gguf", "gpu": true}}
    EOF
    python serve/server.py --engine maya --config ik2-mini.json

Thinking off (`reasoning_effort` `none`, `enable_thinking` false) has the prompt open the answer's text segment:
the model has no switch for its thinking, only the effort level, and at the lowest it still thinks a little.

**Smaller cards: the offload path.** Where the experts do not all fit (an 8, 12 or 16 GB card, or a 24 GB one past
its context), tokens still take the resident path's launches, one CUDA graph a token, with a tier of the experts in
VRAM - the ones Maya's profile (`data/expert-profile-ik2mini.bin`, prompts and answers over code, docs and a tool
chat) says are picked most a byte, the looped core's counted three times - and every other expert pinned in RAM: a
pick the tier does not hold is copied by the GPU itself into a staging slot (a small cache of its own) right after the
router, with no host step, and the same expert kernels read the same bytes, so the answers are the ones every expert
in VRAM gives. With `IK2_LANE=1` (lane parity) the CPU lane takes a share of the misses beside the copies, split by
measured costs; every expert then runs one exact arithmetic on either side (`src/ik2/ik2_exact.h`, held bit for bit
by `tests/ik2_off_test.cu`), so an answer does not depend on where an expert ran. Only the experts the tier does not
hold are pinned (on Linux): the tier is planned on the card's free VRAM first, copied in from the mapped files, then
the rest read and pinned - an 8 GB card's tier is 2.6 GiB of RAM the PC does not need (12.1 GiB pinned; 12 GB card
~8 GiB, 16 GB ~4 GiB). `IK2_PIN_ALL=1` pins every expert. The PC needs the model in RAM and VRAM together (about
24 GB, e.g. 16 GB of RAM with an 8 GB card; no SSD tier): the setup refuses less in one line. Drafting stays off on
the offload path (its window runs only where every expert is in VRAM; a NOTE says so).

**The preview.** Inkling-2-Mini comes in as an experimental preview: Linux, one NVIDIA GPU of 8 GB or more, sm_70
(Volta: V100) or newer; tested on RTX 3090s (sm_86) and V100s (sm_70); Turing (sm_75) kernel-tested only, Blackwell
(sm_100 / sm_120) compile-checked only. Elsewhere `maya-ik2` stops at the start with one line saying why (under
`--serve` also as a `NOTE` the server prints): no NVIDIA GPU, a GPU older than sm_70 (Pascal and before: none of the
tensor-core instructions its kernels use), a build with no kernels for the GPU's arch, less than 7 GiB of VRAM,
`--gpus 2` or more, Windows. `IK2_UNSUPPORTED=1` runs it anyway where it can run at all: two GPUs (`IK2_EP=1` or
`IK2_TP=1`, section 6), smaller cards with the experts streamed from RAM, @cobra91's Windows / CUDA 13 port -
untested here. CMake skips `maya-ik2` without a CUDA arch of 70 or more in the list, and on Windows unless
`-DSTRATA_IK2_WINDOWS=ON`.

**AMD (HIP, Linux).** A HIP build has `maya-ik2` for every arch of its list (not for AMD on Windows yet). The kernels
written for `mma.sync` / `ldmatrix` / `cp.async` run through AMD versions that keep their fragment layouts; the prompt's
hot paths have kernels of their own on RDNA 3 / 3.5 / 4's matrix cores (`src/ik2/ik2_rdna.inc`): a layer's routed
experts in three launches straight from their blocks (`IK2_RDNA_MOE=0`: off) and the fused attention in 24 KB of shared
memory (`IK2_RDNA_FLASH=0`: off); the other FP16 GEMMs take hipBLASLt's kernel measured once per shape, kept in
`lt-picks.txt` beside the GGUF (`IK2_LT_STORE` names another file). Tokens take the scalar expert kernels
(`IK2_EXPERT_MMA=1`: the mma layout). Answers repeat on a GPU (every sum in a fixed order) but are not CUDA's bits.
The engine refuses an AMD GPU older than RDNA 3 at its start; on an APU (Strix / Gorgon Halo) it counts the RAM the
system has available, less max(8 GiB, 10%), as the GPU's room. AMD keeps every expert in the GPU's memory (24 GB, no
offload path there yet). Run on a Radeon 8065S (Ubuntu, ROCm 7.1).

**Defaults.** Thinking Machines' `generation_config.json`: temperature 1.0 (no top_p, no top_k: 1.0 and off).
Thinking is the template's `reasoning_effort`, a name or a number in [0.0, 0.99]: `none` 0.0, `minimal` 0.1, `low`
0.2, `medium` 0.7, `high` 0.9 (the template's default), `xhigh` and `max` 0.99.

**Files** (Apache 2.0, the model and unsloth's GGUF alike):

| File | Repository | sha256 |
|---|---|---|
| `Inkling-2-Mini-UD-Q4_K_XL.gguf` (17.6 GB) | [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) | `e62d463c135ffe283df8d94b279a96ca145ac128b52ce854478613045bf4de15` |
| `mmproj-BF16.gguf` (0.6 GB, pictures) | same | `2c165c78790e78a3f45a2d1e9e931083e52ee69d45ff2a0dc9c281fdf1a666e7` |
| `mtp.safetensors` (0.6 GB, the drafter: the setup fetches it, not with `--drafting off`) | [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) | `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2` |

Outside the preview, two GPUs with a loop run as expert parallel (`IK2_EP=1`, its caches filled at the start:
checked on Windows, an RTX 5080 with an RTX 4060 Ti); the drafter is one GPU's.  Not done: the loop in the layer-major prompt passes (`IK2_SUP`: off
for a looped model), a second card for the KV cache (a million tokens is 16 GiB of it), the drafter's own launches
as a graph (2-3 ms a step now).
