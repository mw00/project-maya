# Project Maya quants of GLM-5.3-Flash - the plan

Two quants of our own, made from the official weights with error-feedback rounding and scale refinement sized to what we can run:

| | Maya-S (first) | Maya-L (later) |
|---|---|---|
| for | PCs with a smaller memory pool across RAM and VRAM (it runs from 32 GB of RAM, the rest streamed from the SSD) | quality first: larger memory pools |
| size | ~80-85 GB (experts ~74-78 GB) | ~135-145 GB (~3.5 bits a weight) |
| quality target | as close to the FP8 model as ~2 bits allow; no loops | KL <= 0.05 against the FP8 model |
| compare against | the FP8 model (Z.ai's release) | the FP8 model (Z.ai's release) |

## What we start from

- **Source**: `zai-org/GLM-5.3-Flash`: 62 safetensors shards, **328.4 GB**, block-wise FP8 (e4m3 weights, an fp32
  scale per 128x128 block). This is Z.ai's own deployment precision, so nothing is lost against BF16. The BF16 repo
  (640 GB) isn't needed.
- **Model**: 45 layers. The first 3 are dense (FFN 12288); the other 42 are MoE with 288 routed experts each (top-8,
  4096 x 2048, gate/up/down) plus one shared expert. Attention is 34 KDA (linear) layers and 11 DSA/MLA layers.
  There are mHC 4-stream residuals and one NextN (MTP) layer. Routed experts are 304.4B of the 321B parameters; one
  expert is 25.2M parameters.
- **The engine's development file** (before Maya's own quants): a 1.6-bit GGUF with experts of ~84 GB.

## Machines

- **EVO-X5 (arrives 2026-10-07)**: Ryzen AI Max+ 495, 16 cores, RDNA 3.5 40-CU GPU, 192 GB unified memory (up to
  160 GB for the GPU), 2-4 TB Gen4 NVMe. It does the heavy passes: one MoE layer in BF16 is 14.5 GB, so a layer, its
  Hessians and the calibration activations all fit at once. BF16 is native here (the V100s don't have it). Needs
  PyTorch for ROCm (gfx1151-class); check that first.
- **Mercury / Uranus**: the engine side.
  - Score every candidate quant in our engine (`MAYA_GLM_SCORE`: NLL, perplexity, top-1, and KL against a
    reference dump).
  - Add kernels for the new types.
  - Measure speed.

## Pipeline

### 0. Fetch (needs the owner's OK for the download)

- The 62 shards plus config and tokenizer, onto the EVO-X5's NVMe, resumable.
- Each file is checked against Hugging Face's LFS sha256.
- Time depends on the internet line: ~45 min at 1 Gbit/s, ~7.5 h at 100 Mbit/s.

### 1. Calibration and evaluation text

About 1M calibration tokens plus 100k held-out evaluation tokens, all in GLM's chat template, with thinking traces
where the model would think:

- **Chat**:
  - multi-turn conversations
  - instructions and writing
  - English, Portuguese (BR), Spanish, Chinese and a few others
- **Code**:
  - Python, JS/TS, C/C++, Rust, Go, SQL, shell
  - code reasoning traces
- **Math and science reasoning** (long `<think>` traces)
- **Tool calls / JSON / agent turns** (GLM's `<|observation|>` turns)
- **Long documents**: several 16k-32k-token samples, because the KDA state and the DSA selection behave differently there
- **The owner's own usage**, if they want it included: the dashboard conversations, or just the routing profile
  (`expert_usage.txt`)

Public datasets will be listed with exact files and sizes before anything is downloaded. Self-generated data
(answers from our current engine to ~500 written prompts) fills gaps in the chat format.

### 2. Reference and statistics pass

PyTorch on the EVO-X5, layer by layer, FP8 dequantized to BF16. It reads each layer once and keeps every token's
hidden state between layers (~1M x 4 streams x 4096 x 2 B = 32 GB). The reference forward uses `transformers`'
own `glm5_next` code (5.16+), checked against our engine on a few prompts first.

Per layer it records:

- **Activation statistics per expert**:
  - the diagonal, which is the imatrix
  - the full X^T X Hessians for gate/up (4096^2) and down (2048^2) inputs, accumulated in fp32 and kept on NVMe
    (~1 TB for all experts, so each layer's are used and dropped before the next)
- **Routing statistics per expert**:
  - frequency
  - REAP saliency (router weight x ||expert output||, averaged over the tokens routed to it)
  - the same per domain
- **The reference**: for the evaluation tokens, the final logits as the top-64 log-probs plus logsumexp at every
  position. This is the KL yardstick, in the format `MAYA_GLM_SCORE_REF` reads (extended to top-k).

Estimate: ~1 h compute (18B active parameters x 1M tokens; Hessian accumulation) plus reading 328 GB once.

### 3. Candidates

Every tensor is quantized at each candidate GGUF type with **error-feedback rounding**:

- **Method**: GPTQ over the full Hessian, block by block. Inside each block, llama.cpp's own quantizer chooses
  scales and grid points, weighted by the inverse-Hessian diagonal. The block's error then propagates to the
  remaining columns. This works for scalar K-quants and for the IQ lattice types, since a block is a block.
- **Candidates**:
  - routed experts: IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS
  - everything else: Q4_K, Q5_K, Q6_K, Q8_0
- **Recorded**: the Hessian-weighted output error per (tensor, type) and per expert group.
- **Cost control**: the error table needs a sample of experts per layer (~32 of 288). Only the final assembly
  quantizes everything.
- **Gumbel-softmax refinement**: the paper's Gumbel-softmax training of the grid choices and scales, published for
  scalar K-quants. It's a later add-on for any K-quant tensors the allocation picks, kept only if it measurably
  lowers KL.

### 4. Allocation (RCO-style)

- **Goal**: choose a type per tensor under the byte budget, minimizing predicted end-to-end damage (each tensor's
  error times its layer's sensitivity, with sensitivity from small perturbation runs). Then score the top
  2-3 allocations in our engine against the FP8 reference and keep the best. RCO proper uses gradients of the true
  KL; this approximates it with measured errors plus measured KL checks.
- **Maya-specific: per-expert bits.** Our engine loads its own pack, so experts don't all need the same type.
  - Hot experts (high routing frequency and saliency) get more bits; the long tail gets fewer.
  - The GGUF version keeps one type per tensor so it still loads in llama.cpp.
  - Usage on real text is very skewed (that is why Mercury's VRAM tier hits 97%), so this should buy more quality
    per GB than anything else.
- **Fixed floors against loops and drift**. These parts are only ~6 GB:
  - attention, MLA, KDA and shared experts at Q6_K-Q8_0
  - the 3 dense layers at Q8_0
  - output head Q6_K or better, embeddings Q6_K
  - down never below gate/up for the same expert

### 5. Pruning study (Maya-S only)

Pruning frees bytes, but it is not free:

- OpenMOSE's REAP prune of this exact model (60 of 288 experts per layer removed) measured KL 0.45 against the
  original, and perplexity +2.8%.
- The public 3.5 -> 3.0-bit step costs about 0.07 KL.

So: budget-matched runs at 0%, ~10% and ~20% pruning, each with the bits the freed bytes buy, judged by KL, task
checks and the loop test. My expectation is that light or no pruning wins and per-expert bits do the work.

### 6. Assembly

- **GGUF**: standard types, one type per tensor, loads in llama.cpp. Metadata comes from today's GGUF.
- **Maya pack**: what our engine loads, including per-expert types when used.
- **Vision**: the mmproj stays BF16 (1.2 GB) for the day the engine gets vision.

### 7. Evaluation (each candidate, the same text)

- **KL against the FP8 reference**, plus perplexity and top-1 agreement:
  - on the held-out mix
  - separately for chat, code, math and Portuguese
  - long context (16-32k)
- **Loop test**: 50 long generations at the default sampling (T=1.0, top-p 0.95) and 20 greedy ones. Measures
  repeated n-grams, runs that never end, and broken thinking-block structure.
- **Task accuracy against the FP8 model** (zero-shot ARC-Easy/Challenge, HellaSwag, WinoGrande, PIQA the way
  lm-evaluation-harness scores them: tools/maya_quant/zs_*.py), and spot checks:
  - GSM8K (100)
  - HumanEval subset
  - IFEval subset
  - MMLU-Pro (300)
  - a Portuguese chat set
- **Speed** on Uranus (64 + 32) and Mercury.

## Engine work (Mercury/Uranus)

- **Done 2026-10-06: the decode kernels** for IQ2_XS 17, IQ2_S 22, IQ3_S 21 and IQ1_M 29, beside IQ2_XXS 16,
  IQ3_XXS 18, IQ1_S 19, IQ4_XS 23, Q2_K 10 and Q3_K 11.
  - Checked on synthetic experts against a double-precision reference: every type lands at the same ~1.5% (expert
    kernels) and ~0.9% (dense GEMV), which is the q8_1 activation's share.
  - The prompt path's MMQ now also builds Q2_K-Q6_K.
  - The CPU lane goes through ggml, which covers every type.
- **IQ1_M has no MMQ in llama.cpp**, so a layer of it couldn't go through the batched prompt path. It stays out of
  the candidates unless that path gets a dequantize-and-GEMM fallback.
- Light kernels (the prompt path's sparsely routed experts) still cover only 16/18/19. Others go through MMQ, which is
  correct but slower for a handful of rows.
- Per-expert types: the route groups its 8 experts by type and launches per group.
- The KL reference as top-k log-probs, so a reference from the EVO-X5 can be scored on the V100s.

## Order of work

1. EVO-X5 setup: ROCm PyTorch; check the reference forward against our engine.
2. Download the source (with the OK).
3. Calibration text (exact dataset list first).
4. Pass 2 (statistics + reference).
5. Error table.
6. Allocation and pruning study.
7. Maya-S assembly + evaluation.
8. Maya-L.

The engine kernels go in parallel on the V100s.

## The owner's requirements for the headline quant (2026-10-06)

- **As close to FP8 as possible at the smallest size, no loops, no broken output.** Post-training is in scope:
  - Error-correcting rounding first.
  - Then, where the measurements still show damage, block/expert-wise output-matching refinement of the quantised
    scales and grid choices, and a short distillation of the small floating-point parts against the
    FP8 teacher's outputs.
  - Every candidate passes the loop test before it is called a release.
- **Made on Uranus** (the EVO-X5 slipped a day):
  - The FP8 source is on its PCIe 4 NVMe, and the 64 GB RAM holds the calibration activations.
  - The V100 does the statistics pass; its CPUs, plus Mercury's, quantise.
  - Mercury tests the two-GPU case.
- **Vision:** the release carries the vision tower.
  - The FP8 checkpoint's 347 vision tensors are converted into an mmproj GGUF (BF16; Q8_0 measured as an option).
  - The engine's GLM vision path is a release item.
- **Calibration and evaluation weighted toward what the community benchmarks:** single-file HTML/CSS/JS pages,
  three.js scenes and canvas/WebGL animations (plus shaders), beside chat, code, maths, tool calls, Portuguese and long
  documents.
  - Their experts get measured and bits accordingly.
  - The evaluation adds three.js / HTML generation prompts: does the page parse and run (headless browser), no loops.
- **QA: a context-fill test.**
  - Fill the context to its configured maximum (32k, then 64k / 128k) with real text and keep generating.
  - Check that the answer stays coherent, nothing crashes, and memory stays inside its budget.
  - Later on the roadmap: speeds at several context sizes, published in the repo.

## Later (the owner sets the start)

- **EXL3 quants** (ExLlamaV3's trellis/QTIP-style format): load and run them in the engine. Added to the roadmap
  2026-10-06. Not started; the owner decides when.
