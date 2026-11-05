# Running GLM-5.3-Flash - the study behind Maya's engine

What it would take for [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) to run on this
engine, component by component, with every number the decision needs. Written 2026-09-29 against engine
0.1.27 (`main`); all `file:line` references are to that tree.

The engine Maya started from is not a general inference engine: it is a native implementation of one architecture -
Qwen3.8-Flash-Next ("qwen4exp") - with its memory tiers (VRAM expert cache / pinned RAM / mapped SSD) and
its speculation stack built around that one shape. GLM-5.3-Flash is the same *kind* of model (a 3:1
linear-attention/sparse-attention hybrid MoE with hyper-connections and an MTP layer), which is why a port
is worth studying rather than dismissing. About two thirds of the engine's machinery applies unchanged; the
remaining third is the mixer (KDA vs GDN), the full-attention mixer (MLA vs plain QKV), the router scoring,
and the packaging.

---

## 1. The model, exactly

From the released `config.json` (`zai-org/GLM-5.3-Flash`, `Glm5NextForConditionalGeneration`,
`model_type: glm5-next`, MIT, released 2026-08-25):

| | |
| --- | --- |
| Total / active params | 320 B (321,323,031,390) / 18 B per token |
| Native weights | FP8 E4M3, block 128×128, dynamic activation scale; norms, gates, router, embeddings, `lm_head`, vision tower and indexer stay high-precision (BF16/F32) |
| Layers | 45 (`hidden_size` 4096); layer pattern 3 `linear_attention` then 1 `deepseek_sparse_attention`, so the full-attention layers are 3, 7, ..., 43 - **11 of them**, the same 4-interval as Strata's `qsa_interval` |
| Full attention (11 layers) | MLA, NoPE: 64 heads, `q_lora_rank` 1536, `kv_lora_rank` 512, `qk_nope_head_dim` 256, `qk_rope_head_dim` 0, `v_head_dim` 256, `mla_use_nope` true. Plus a DeepSeek-style sparse-attention indexer: `index_topk` 2048, `index_n_heads` 32, `index_head_dim` 128, `index_kpool` 4 (compressed index-key pool) |
| Linear attention (34 layers) | Kimi Delta Attention: 64 heads × head_dim 128, `short_conv_kernel_size` 4, `gate_lower_bound` -5.0 |
| Hyper-connections | `mhc: true`, `hc_mult` 4, `hc_sinkhorn_iters` 20 - the manifold-constrained variant (mHC) |
| MoE | 288 routed experts + 1 shared, top-8 per token, `moe_intermediate_size` 2048, sigmoid scoring, `noaux_tc` routing, `routed_scaling_factor` 2.5, router in float32; **layers 0-2 dense, 3-44 sparse (42 MoE layers)** |
| MTP | `num_nextn_predict_layers` 1 |
| Vision | 24-block ViT, hidden 1024, 16 heads, 448 px, patch 14, spatial merge 2, out 4096, projector intermediate 10240 |
| Vocab / context | 154,880 / 1,048,576 |
| Routed-expert parameter share | 288 × 42 × 3 × 4096 × 2048 ≈ **304.6 B params = 95 % of the model** |

GGUFs already exist: [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)
(dynamic quants; ~120 GB at their 3-bit, ~180 GB at 4-bit). llama.cpp has the building blocks this
architecture needs (KDA chunked + recurrent kernels, NoPE-MLA, DSA lightning indexer from the DeepSeek
V3.2 work, nextn/MTP), which matters twice: it proves the kernels are portable, and `third_party/ggml`
is the pinned dependency Maya builds its native path and `maya-vision` against.

## 2. What the engine already is

For reference, the implemented geometry (`include/strata/core/layout.hpp:27-59`): n_embd 2560, 48 layers,
full attention every 4th layer (12 QSA), GDN linear layers (16 k-heads / 48 v-heads, state 128, conv 4),
QSA full attention (24 heads, 2 KV heads, head_dim 256, indexer 4 q-heads over one shared 128-dim key),
gated residual with hc = 4 streams and a 320-wide bottleneck, MoE 512 experts × top-10 × intermediate 640,
a layer-1 PLE module backed by a 28.8 GB n-gram table, and an MTP draft layer. Per layer:

    gr_read (attn) -> GDN | QSA -> gr_write (attn) -> gr_read (ffn) -> MoE -> gr_write (ffn)

The engine's unit is a captured per-layer CUDA graph replayed by a host loop that overlaps a CPU expert
pool and a VRAM expert cache behind a doorbell (`include/strata/core/session.hpp`). Experts live in
`experts.bin` (fixed-size blobs, `layer * n_expert + expert`), served from a pinned arena by default or
from an mmap in low-RAM mode (`--mmap-experts`, `include/strata/core/expert_source.hpp:303-360`). Dense
weights load from the pack into VRAM. A "native pack" path runs directly from GGUF shards with per-layer
runtime geometry (`include/strata/kernels/cpu/native_expert.hpp`) - this is how the Coder (mixed i-quant
types, 256 experts/layer) works, and it is the path a GLM pack would ride.

## 3. Component-by-component gap analysis

### 3.1 Applies unchanged

| Piece | Why |
| --- | --- |
| 3:1 layer pattern | GLM's full-attention layers are `layer % 4 == 3`, identical to `is_qsa_layer` (`layout.hpp:63`). 45 layers → 11 QSA + 34 linear, and `n_qsa_layers() = 45/4 = 11` is already correct arithmetic. |
| Hyper-connection residual plumbing | `hc_mult` 4 = the implemented `hc = 4`. The stream stack, the per-half read/write structure, the BF16 weight contract and the fused-GR fast path all carry over. (What changes is *which weights* and one normalization - §3.4.) |
| Expert memory-tier machinery | VRAM expert cache, expert profile, pinning, PCIe overlap, multi-GPU expert split, the mmap source - all parameterized by `(n_layers, n_expert, blob_bytes)` at runtime. |
| KV cache formats and streaming | INT8 / Q4_0 / K8V4, KV streaming past 64K, checkpoints - all sized from `ModelGeometry`. |
| MTP speculation protocol | Verify windows, draft policy, prompt-lookup drafter. The drafter's *weights* are the model's own MTP layer (§3.6). |
| Prefill pipeline | Chunked MMQ prefill, `--prefill auto` - shapes come from the pack. |
| Serving | OpenAI/Anthropic APIs, MCP, monitor, tokenizer-free prompt caching - model-agnostic already. |
| PLE | GLM has no PLE module. `PleRun` is already optional (`layer.hpp:545`, "null when the PLE is not wired"), and a pack without `blk.1.ple_*` skips it - which is what GLM must do. No shard 2, no 29 GB n-gram table. |

### 3.2 The linear-attention mixer: GDN → KDA (moderate kernel work)

The implemented GDN (`ref/gdn.py::gdn_decode`, `layer.hpp:8-17`) has per-head scalars: `alpha`, `beta`,
`gate` are one float per v-head (`layer.hpp:73-75`), the gate is `softplus(walpha@x + dt) * ssm_a`.

KDA differs in two ways that matter:

1. **Fine-grained (per-channel) decay gates.** The decay is a *vector* of head_dim per head (64 × 128),
   applied elementwise to the state, not a scalar per head. In llama.cpp this is the
   `linear_attn_config`'s two decay-gate forms, selected by `gate_lower_bound`; GLM sets -5.0, so KDA's
   lower-bounded fine-grained gate form is the one to implement. The recurrent state shape itself
   (64 heads × 128 × 128 ≈ 1 M floats/layer, close to the implemented 128 × 128 × 48 ≈ 786 K) and the
   short conv (kernel 4 = `d_conv` 4) are the same shape family the `gdn.cu` / `native_gdn.cu` /
   `fused_gdn.cu` kernels already handle.
2. **Gate lower bound.** The -5.0 clamp on the gate logit is a one-line change *once* the gate is a
   vector, but it changes the numeric contract, so both the CUDA and the `ref/gdn.py` oracle must change
   together and re-pass `gdn_parity`.

Work: extend the gate buffers and the decay application in the chunked (prefill) and recurrent (decode)
GDN kernels to per-channel gates; add the lower-bound clamp; new BF16 `ssm_alpha`-class weights (shape
changes from (48) to (64×128) per token); update `check_layer` assertions. The activation-contract logic
(which weights take Q8_K vs BF16 activations) is per-tensor and already data-driven.

### 3.3 The full-attention mixer: QSA → NoPE-MLA + DSA (the largest piece)

The implemented QSA layer is plain QKV attention with RoPE: 24 heads, 2 KV heads × 256, cos/sin tables
(`layer.hpp:239-241`), KV pools paged per head (`k_pool` `[page][kv_head][page][dim]`), a block-pooled
indexer (4 q-heads, one shared 128-dim key) driving a top-k cell selection, and a per-head output gate.

GLM's 11 full-attention layers need:

1. **MLA with absorption, NoPE.** `qk_rope_head_dim = 0` means *no RoPE at all* in these layers - the
   rope tables, `pos_dev` rotation and M-RoPE position ids for attention disappear from the path (M-RoPE
   remains only as the vision position encoding if the vision path needs it). The KV cache becomes the
   compressed latent `c_kv` (512 floats/token/layer) - which is *cheaper* than the implemented KV: 11
   layers × 512 × 2 B ≈ **11.3 KB per context token in FP16**, about half of the implemented 2 × 256 × 2
   heads × 2 B = 24.6 KB, and 1.4 GB at 128 K context. Two implementation options:
   * *Absorbed:* precompute `q_hat = W_uk^T q` (a 512-vector per head) and score directly against the
     latent. The attention kernel then reads ONE 512-wide pool instead of per-head pools - a simpler and
     faster memory pattern than today's, at the cost of a per-token 64 × 512 GEMV to build `q_hat`.
   * *Expanded:* materialize per-head K (256) and V (256) into the existing pool layout at write time
     (3.2× the latent bytes, 36 KB/token) and reuse the existing decode/verify kernels unchanged.
     Absorbed is the right target (that is why MLA exists), but expanded is the honest first milestone -
     it isolates correctness from kernel novelty.
2. **The indexer.** GLM's indexer is 32 q-heads × 128 (vs implemented 4 × 128 shared-key), top-k 2048
   cells (the implemented selection width is already ~2 K - `kTopkMaxCells`, "8 queries x 2,051 cells",
   `layer.hpp:275`), plus `index_kpool` 4 - a compressed index-key pool whose exact form needs the
   converted GGUF (or the llama.cpp glm5 conversion) as the reference. The QSA indexer machinery
   (`native_qsa_indexer`, block scores, pooled keys) is the right skeleton; the head counts and the pool
   compression are the deltas.
3. **Per-head output gate** (`v_head_dim` 256, gated attention output) - the implemented `q_full`
   q|gate split (`layer.hpp:367-371`) already has this shape; dims change.

Work: new absorbed-MLA attention kernels (decode + prefill + verify-window) or the expanded interim path;
indexer re-parameterization; `QsaState` without rope tables; `check_layer` per-type rewrites. This is the
single largest kernel workstream, and it is also where numerical parity effort concentrates.

### 3.4 Hyper-connections: HC → mHC (small, but resolve one question)

The implemented GR (`ref/gr.py`, transcribed from llama.cpp's `qwen4exp` `build_hc_mix/combine`,
`gr.hpp:110-120`) uses per-half weights `w_norm` (hc·n_embd), a low-rank bottleneck `w_down`/`w_up`
(hc_lr 320), and `w_inject`; the write gate is `2*sigmoid(inject/hc)`.

GLM's mHC adds the manifold constraint: the mixing matrix is Sinkhorn-normalized
(`hc_sinkhorn_iters` 20, `hc_mult` 4). The question to resolve from the first converted GGUF is *where*
the Sinkhorn iteration lives:

* If the released weights are already normalized (the constraint enforced during training), mHC is a
  **weight-preprocessing step at pack time** and the inference kernels are bit-identical to today's GR.
* If the conversion expects the engine to normalize per forward (the config exposes the iteration count
  because inference needs it), the engine needs the 20-iteration Sinkhorn on the 4×4 matrix per layer per
  token - trivially cheap (16 numbers), but it must be inside the captured graph, and the oracle (`ref/gr.py`)
  must grow the same step.

Either way this is the *cheapest* architecture delta; the risk is only parity discipline.

### 3.5 MoE: shapes + scoring (small engine work, one hard wall)

* **Shapes.** 288 experts × top-8 × intermediate 2048 vs 512 × 10 × 640. All of `n_expert`, `k` and
  `n_ff` flow through `ModelGeometry` at runtime, the router top-k is already a parameter
  (`router_top10.hpp:17`, "k must be <= 64"), and the doorbell/expert dispatch carry `k` explicitly.
* **Dense layers 0-2.** The implemented stack assumes every layer is MoE. GLM's first three layers are
  dense MLPs. `core/native_dense.hpp` exists (dense GEMV machinery) - a dense-FFN layer type is needed in
  `block_layer` for layers 0-2, which is a small structural change plus weights plumbing.
* **Scoring.** The implemented router is softmax over all experts + renormalise (`router_top10.hpp:3-5`,
  `ref/moe.py::router`). GLM is **sigmoid scoring, `noaux_tc`, renormalise over the selected 8, × 2.5
  routed scaling, float32 router weights**. A scoring-function variant of the router kernel + the scaling
  factor - small, but it is a *selection semantics* change: parity must be checked at the router output
  (the BF16-GEMV → logits → top-8 boundary), because a flipped selection is a different expert, not a
  rounding error (the repo's own note, `layer.hpp:131-139`).
* **The hard wall: the canonical Q2_0 CPU expert kernel.** `include/strata/kernels/cpu/expert.hpp:34-45`
  bakes H = 2560, FF = 640, NE = 512 and BLOB = 1,382,400 as compile-time constants, and its contract
  REFUSES other shapes ("a different n_embd or a different expert width is REFUSED rather than
  mis-indexed", `expert_source.hpp:18-21`). GLM does not go through it: the **native path**
  (`native_expert.hpp`) is runtime-geometry (per-layer `NativeFmt` with `n_embd`, `n_ff`, ggml types from
  the GGUF), and that is the path GLM must ride - exactly as the Coder and the Orca compatibility pack did.
  Consequence for the study: GLM packs are native packs; the canonical Q2_0 pack path
  is out of scope for the port.
* **Expert blob size.** 3 × 4096 × 2048 = 25.2 M params/expert ≈ 3.2× the implemented expert. Blob sizes
  become per-layer-native (already supported: `layer_offsets_`/`layer_blob_bytes_`,
  `expert_source.hpp:345`). Per-token routed traffic at Q2_0-class rates ≈ 336 reads × ~7 MB ≈ 2.4 GB -
  ~3.6× the implemented 664 MB/token (`layer.hpp:526-527`), which is the real performance story, §5.

### 3.6 MTP, PLE-class extras, vision, tokenizer

* **MTP.** One nextn layer, same protocol. Two fixes: the drafter's geometry is hardcoded to canonical
  defaults (`static const strata::core::ModelGeometry draft_geometry{}`, `generate.cpp:1726`) and must
  follow the target model; the Qwen draft-vocab embed size is hardcoded at one call site
  (`generate.cpp:1374`, 248320) and must come from the pack. GLM's MTP tensors must be in the GGUF
  (llama.cpp's glm5 conversion carries nextn tensors) and `tools/mtp_pack.py` taught the tensor names.
  Draft acceptance is an empirical question per model.
* **Vision.** `maya-vision` is built from the pinned llama.cpp `mtmd` (`tools/vision/`). GLM's 24-block
  ViT at 448 px is a standard layout; support reduces to what the pinned llama.cpp supports for
  `glm5-next` vision + a converted `mmproj` GGUF, plus the family wiring in `setup.py`. If the pinned
  llama.cpp predates glm5 vision support, this is a submodule pin bump with a re-run of the parity tests.
* **Tokenizer.** The packer exports the model's own tokenizer from the GGUF (the rule: never share
  another model's tokenizer). GLM's BPE (154,880 vocab) flows through `gguf-py` like Qwen's; verify the
  pre-tokenizer regex is one `gguf-py` knows. A GLM chat template (`serve/chat_template.jinja` is
  Qwen's) is needed for the server, plus its image/video token ids (154854/154855).
* **Metadata.** The engine reads nine `qwen4exp.*` keys (§6). A `glm5-next.*` set must be recognized -
  the minimal change is in `generate.cpp:1466-1480`, which today reads only `expert_count` and
  `expert_used_count` from the model file.

### 3.7 Quantization and the pack

* Source weights are FP8 E4M3; the packer consumes standard GGUF quants through ggml-cpu
  (`ref/load.py`'s twelve-format table + the native path). The unsloth GGUFs (Q2_0/IQ2/IQ3/IQ4 dynamic)
  are the input; the canonical Q2_0 pack pipeline is not needed (§3.5).
* The `--compat-bf16` route (`tools/iq_pack.py`, built for ORCA) is exactly what a GLM pack needs for its
  high-precision projections: GLM's BF16 list differs from Qwen's (config `modules_to_not_convert`:
  attention projections on sparse layers, norms, gates, router, embeddings, `lm_head`, vision tower,
  indexer), so the compat list must be derived from the architecture's tensor names rather than copied.
* Per-tensor activation contracts (`docs/activation-contract.md`, Q8_0 vs Q8_K by source type) are
  data-driven already - new quant mixes ride along.

## 4. The mmap study (the memory-pool question)

**The headline number: the routed experts alone exceed 64 GB of RAM at every useful quantization.**

| Format | bpw (approx) | Per-expert | 288 × 42 experts | + dense ≈ | File |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q2_0-class | 2.25 | 7.1 MB | **~86 GB** | ~93 GB | ~100 GB |
| IQ2_XS-class | 2.06 | 6.5 MB | **~79 GB** | ~85 GB | ~95 GB |
| IQ3_XXS-class | 2.66 | 8.4 MB | **~101 GB** | ~108 GB | ~120 GB |
| 4-bit class | 4.25 | 13.4 MB | **~162 GB** | ~170 GB | ~180 GB |

(The original engine's Qwen numbers for comparison: 23-50 GB of experts *pinned in RAM* - the design assumption
"all experts resident in RAM" is dead for GLM on consumer hardware. This is the thing the user flagged.)

Decode traffic per token: 8 experts × 42 layers = 336 expert reads ≈ **2.4 GB/token at Q2_0 rates**
(3.6× Qwen's 664 MB/token). Where those bytes come from decides everything:

* VRAM cache (hot experts, profile-ranked): unchanged machinery, same hit-rate economics, but each hit
  now removes 3.2× more bytes from the RAM/SSD path.
* RAM (page cache or pinned): on 128 GB machines, ~86 GB of Q2_0-class experts fit alongside the OS -
  the mmap source with a warm page cache is then DRAM-speed and the current low-RAM mode *is* the main
  mode. On 64 GB machines roughly half the expert set fits; the rest pages from SSD, and a cache-miss
  heavy token costs ~1.2 GB of NVMe reads ≈ 100-150 ms - workable at ~5-8 tok/s cold, degraded but usable,
  and the expert profile is what keeps the miss rate down over time.
* SSD: NVMe random reads at blob granularity (~7 MB) are exactly the access pattern the existing
  mmap source and the PLE reader's unbuffered async path (`direct_file.hpp`) were built for.

What the engine already has for this: `FileExpertSource` (MapViewOfFile/mmap, blob = base + multiply,
`expert_source.hpp:303-360`), `pin_cache_complement` (pin the non-cached complement when RAM allows),
the VRAM tier with decayed-usage swapping, PCIe streaming of next-layer experts during prefill attention,
KV streaming, and the unbuffered async file layer.

One lesson the code already holds, and the study adopts: **retention is the whole game, and
random-access hints are the enemy.** `FILE_FLAG_RANDOM_ACCESS` was tried on the expert mapping and
measured at **1.93 GB/s - disk speed, 14x off** - because it tells Windows' cache manager to drop the
pages again quickly (`expert_source.cpp:332-343`). The expert access pattern is random per layer but
*repeated every token*, so the mapping deliberately carries no hint at all. The same reasoning rules out
`MADV_RANDOM` on Linux. What a 100 GB+ mapping does need (implemented on this branch, §7):

1. **Prefetch the routed blobs at `begin_layer`.** The source's own contract says it: "a source that
   reads from disk wants to start the read here so it overlaps the quantisation"
   (`expert_source.hpp:98`). When the router's ids for a layer are known, issuing
   `PrefetchVirtualMemory` (Windows) / `posix_fadvise(WILLNEED)` (Linux) for exactly those k blobs
   page-ins them while the pool quantizes the activation and the GPU finishes the previous layer -
   the mmap-path analogue of the PCIe expert streaming the prefill path already does. Unlike the
   random-access flag, prefetch is *retention-friendly*: it pulls the pages the router will actually
   touch into the standby list and says nothing about evicting the rest.
2. **Keep the mapping hint-free.** At GLM's scale (~86-162 GB of experts) the temptation to "fix"
   random access with hints is stronger and the failure mode is the same measured one. The mapping
   stays undecorated; the page cache decides retention by recency, which is what a hot working set
   inside a much larger mapped file wants.
3. **Never map-and-pin by accident.** `pin_cache_complement` remains opt-in and is now also the tool for
   the 128 GB-RAM case (pin the complement the VRAM cache does not hold, when it fits); on 64 GB it must
   stay off and the page cache must be allowed to reclaim.

## 5. Expected performance envelope (estimates, ±30 %)

The method: GPU part scaled by bandwidth and VRAM-held expert share,
CPU part by remaining expert bytes. Assume an RTX 3090-class card (24 GB) + DDR5, Q2_0-class pack:

* ~3,000 VRAM-cached experts cover ~70 % of reads (profile economics transfer; per-expert bytes are
  3.2× bigger, so a 24 GB card caches ~2.5-3 K of them). Remaining ~30 % = ~720 MB/token from
  DRAM at ~40 GB/s ≈ 18 ms → **decode ceiling ≈ 50-65 tok/s** if the CPU pool and the overlap hold.
* On 12 GB + 64 GB RAM: ~1.5 K experts cached (~55 % of reads); the misses split between page cache and
  SSD → **~25-40 tok/s** warm, lower cold.
* Prefill: batched expert MMQ over 8 K-token chunks is compute/weight-bound; expert bytes/token are
  amortized ~800× across a chunk, so **~1,000-1,800 tok/s** prefill on the same card class is plausible.
* Long context is *better* than Qwen's case: MLA-halved KV (11.3 KB/token fp16) means 128 K ≈ 1.4 GB,
  1 M ≈ 11 GB - 1 M context is feasible on 24 GB + 128 GB with KV streaming, where the implemented model
  saturates earlier.

These are study estimates, not measurements; the parity harness (§6) comes before any tuning.

## 6. Execution plan

**Phase 0 - reference and metadata (this branch).**
`generate.cpp` recognizes `general.architecture = glm5-next` and fails *loudly and specifically* until the
kernel work lands, instead of dying inside `check_all` with forty shape mismatches; the mmap improvements
from §4 (they serve the existing models' low-RAM mode too); this document.

**Phase 1 - the oracle (DONE, this branch).** `ref/glm.py`: GGUF → FP32 on demand (the
`ref/load.py` machinery), one full teacher-forced forward pass of glm5-next with the released
config as metadata defaults; every intermediate the engine will need as a parity seam is
recorded under llama.cpp's own `cb()` names (175 tensors on the synthetic model), and
`tools/test_glm_oracle.py` asserts the architectural invariants (MoE weight renormalization,
Sinkhorn dst-stochasticity, gate ranges, finite seams). The spec was pinned against
`ref/upstream-llama.cpp/glm5-next.cpp` (vendored, llama.cpp @ `272aad8b`), which also answered
the two open questions: **mHC Sinkhorn normalization runs per forward, inside the graph**
(softmax over dst, +eps, then alternating src/dst normalizations - a 4×4 matrix per layer per
token, but data-dependent, so the engine must compute it per token, not at pack time), and the
**k-pool indexer** averages each pool of 4 consecutive indexer keys with a per-key-dim softmax
over (gate + learned per-position additive embedding), scores pools with a ReLU'd multi-head
dot product against prescaled weights, selects top_k/kpool pools per query, and appends the
incomplete tail (`kpool_select_tail` = true in the released config). Also pinned: the KDA
gates are LOW-RANK decomposed (`f_a` rank head_dim → `f_b`, same for the output gate `g_a/g_b`)
with the lower-bound form `lb·sigmoid(−g·A)`, and the fused delta-net kernel applies the gate
on the KEY axis and bakes the `1/√S` output scale. Remaining known gap: llama.cpp has not
implemented the GLM5-Next MTP graph yet ("not implemented yet" in glm5-next.cpp L190), so the
draft layer's exact form stays a Phase 3 question.

**Phase 2 - the mixers (STARTED).** Two kernel families landed, each with a parity harness in the
repo's negative-test discipline (every wrong reading must differ materially before the right one is
judged):

* **mHC** (`include/strata/kernels/glm_hc.hpp`, `src/kernels/cuda/glm_hc.cu`, `glm_hc_parity`): the
  read half is three kernels - the joint-stack inverse RMS (ONE norm over all four streams, not
  gr's per-stream), the 24-row mixes GEMV, and a serial lane that forms the three gates, runs the
  Sinkhorn on the 4×4 comb (softmax over dst, +eps onto the values, dst-norm, then
  (iters-1)×[src-norm, dst-norm]), and forms `mixed = Σ_s pre_s·R_s`. The write half is one
  elementwise kernel: `out_dst = block_out·post_dst + Σ_src comb[dst,src]·residual_src`. The
  negative tests pin the joint-vs-per-stream norm, the src-major comb rows, the post-softmax +eps,
  sum-vs-mean, and the pre-gate floor (with hc_eps exaggerated to 3e-2 in the structural fixture,
  since the real 1e-6 sits below f32 noise). Weights are F32 for now - the BF16-input variant is a
  Phase 3 wiring decision, pinned against this F32 reference.
* **Router** (`include/strata/kernels/router_sigmoid.hpp`, `src/kernels/cuda/router_sigmoid.cu`,
  `router_sigmoid_parity`): one block per token, no reductions (sigmoid is bounded), the rank
  formula from router_top10 reproduced for the tie rule, bias applied to the SELECTION only, weights
  gathered from the unbiased sigmoid, renormalised with the 2**-14 clamp, scaled. The negative tests
  pin biased-weights (the noaux_tc trap), softmax-vs-sigmoid, missing renorm/scale, and tie order
  via a quantised fixture.

* **KDA** (`include/strata/kernels/glm_kda.hpp`, `src/kernels/cuda/glm_kda.cu`, `glm_kda_parity`): the
  linear-attention layer's four pieces, in ggml layouts end to end so Phase 3 wiring needs no
  transposes - the causal conv + SiLU with the history slide as a separate kernel (an in-place slide
  while earlier tokens' threads still read the old columns is a race), the low-rank gate's lower-bound
  squash, the RECURRENCE (one block per head, tokens looped inside the block: key-axis row scale, the
  `u = S^T k` reduction, the delta correction, the outer update, `o = S^T q` with the 1/sqrt(dv)
  baked in like ggml's kernel), and the normed-and-gated output. The negative tests pin the GDN
  readings: scalar per-head gate, value-axis gate, NO delta correction (plain linear attention - the
  shapes all agree), the softplus form, the missing output scale, and reversed conv taps.
* **Clamped SwiGLU** (`include/strata/kernels/glm_ffn.hpp`, `src/kernels/cuda/glm_ffn.cu`,
  `glm_ffn_parity`): the dense-lead, routed-expert and shared-expert activation -
  `silu(min(gate, limit)) * clamp(up, ±limit)`. The negative tests pin the one-sided gate clamp and
  the clamp-before-silu order, whose differences live in silu's damped tail and are bounded
  (~4.5e-3 with the fixture), so their thresholds are set from those bounds rather than guessed.
  This kernel also settled an accuracy contract: `__expf`'s argument-scaled ulp error is visible at
  |x| ~ 10, so these kernels use precise `expf`.

* **DSA indexer + absorbed-MLA attention** (`include/strata/kernels/glm_dsa.hpp`,
  `src/kernels/cuda/glm_dsa.cu`, `glm_dsa_parity`): the pool (per-key-dim softmax over
  gate+ape weighted member average), the score (relu'd multi-head dots against prescaled weights),
  the selection (the router rank formula over VISIBLE pools - pool p visible to query t only when
  its last member is <= t - with the incomplete tail appended), and the attention itself: one block
  per (query, head) against the 512-wide latent cache, q absorbed through wk_b, v absorbed through
  wv_b after the softmax, the 1/sqrt(qk_nope) scale, padding excluded from the softmax. Negative
  tests: a pool-wide softmax, unweighted pooling, no relu, no tail, and the wrong scale denominator.
* **The layer composition test** (`glm_layer_parity`): two complete synthetic-model layers (0: KDA +
  dense FFN; 3: DSA + MoE) through the kernel family, every seam held to `ref/glm.py`'s raw dumps
  (`data/glm5-synth-dumps/`). It caught three real bugs no single-kernel parity could see: the
  oracle's conv einsum labeled the two channel axes differently and computed garbage
  (fixed: `"tc,tck->ck"` - the kernels were right); the router bias read past its per-layer
  (n_expert,) allocation for tokens >= 1 (fixed: broadcast); and the test's own array-KV read
  (`expert_feed_forward_length` is an ARRAY - `MetaValue::u` is 0 for arrays). It also fixed
  glm.py's hc flat to ggml stream-major.

Phase 2 status: **COMPLETE** - all seven suites green on Mercury (glm_hc, router_sigmoid, glm_kda,
glm_ffn, glm_dsa, file_expert_source, glm_layer_parity). The last composition bug was the layer
test's own down-exps read: ffn_down_exps is {n_ff, n_embd, n_expert}, so ne0 is the INPUT dim and
down_e[o, v] lives at v + n_ff*o + n_ff*n_embd*e (the transpose of gate/up, whose ne0 is the
embedding axis).

Phase 3 status: the **Glm5Model streaming runner** (`src/core/glm_model.cu`) is complete and
verified - `glm_model_test` holds it to the oracle at every position of a 12-token sequence
(batched-vs-oracle 2e-3, all 8 prefix argmaxes, streaming-vs-batched 1e-5), derives the greedy
chain from the dump rather than a constant (a hardcoded Phase-1 chain went stale when the oracle's
conv einsum was repaired - four independent reviews converged on this diagnosis), and checks the
streaming continuation at positions 8-11. **8/8 suites green on Mercury in 2.1 s of GPU.** The
packer (`iq_pack.py`) recognizes `glm5-next` and carries the per-architecture BF16-preserve list
(the mHC fn/base/scale names, the glm5 indexer set, the router + noaux_tc bias, the KDA gates)
with 3-D tensors packing byte-exactly as (ne0*ne1, ne2) views; `strata_tokenizer` implements the
glm4/glm5 pre-tokenizer scheme (a verbatim CHATGLM4 transcription from llama-vocab.cpp: digit runs
in groups of at most 3, no `\p{M}` in the letter/punct classes, `ignore_merges` whole-vocab-hit
emission) and still refuses schemes it has not transcribed; the pack export ships the per-scheme
pattern and the ignore_merges flag. The driver gate now points at the verified runner.
Re-verified 2026-10-01 on a fresh Mercury extract: 8/8 suites in 2.0 s of GPU. Remaining for
Phase 4: quantized-kernel wiring for the production driver path, the installer FAMILIES entry
(deliberately deferred - it would advertise a model the driver still refuses to serve), and
serve/measure - the owner has ordered the **IQ1_M quant first**, end-to-end before optimizing
upward; §3.5's hard wall applies (GLM packs are native packs; the canonical Q2_0 expert kernel's
compile-time geometry does not fit GLM). llama.cpp has no GLM5-Next MTP graph; the draft layer
stays deferred.

**Phase 3 - pack and install.** `glm5-next` GGUF → native pack via `tools/iq_pack.py` with the GLM
compat-BF16 list; MTP pack; tokenizer export; `FAMILIES["glm53"]` in `setup.py` (the installer's family
table, `setup.py:80-104`, is one dict entry); GLM chat template; engine metadata keys
(`glm5-next.embedding_length`, `.block_count`, `.expert_count`, `.expert_used_count`, attention head
counts) read into `ModelGeometry`.

**Phase 4 - serve and measure.** Vision helper against the glm5 mmproj; end-to-end parity vs the oracle
per residual ladder (`dump_layers`); needle tests across 1 K-1 M; the benchmark matrix; then the
benchmark tables gain a GLM section.

Rough effort weighting: Phase 2 ≈ 60 %, Phase 1 ≈ 15 %, Phase 3 ≈ 15 %, Phase 4 ≈ 10 %. The critical
path runs through the MLA attention kernels and their parity harness.

## 7. What this branch contains

* `docs/GLM5-FLASH.md` - this study.
* The §4 mmap work in `FileExpertSource`: per-layer prefetch of the routed blobs from the existing
  `begin_layer` hook (Windows `PrefetchVirtualMemory`; Linux `posix_fadvise(WILLNEED)`), best-effort and
  retention-friendly - the mapping itself stays hint-free for the measured reason above. With a test.
* The §6 Phase-0 architecture gate in the driver.
* A build fix found by the clean-clone verification: `native_mmvq_multi` and `hit_cpu_order_parity` are
  now EXISTS-guarded like their siblings - their bench sources are not in the repository, and with
  `MAYA_BUILD_TESTS=ON` the generate step refused to run at all.
* Phase 1: `ref/glm.py` (the glm5-next oracle), `tools/glm_synth_gguf.py` (the synthetic model it and the
  kernel parity tests run against), `tools/test_glm_oracle.py` (7 invariant tests, no GPU), and
  `ref/upstream-llama.cpp/` (the vendored llama.cpp spec, @ `272aad8b`).
* Phase 2 complete: the five kernel families - mHC (`glm_hc_*`), the sigmoid router
  (`router_sigmoid_*`), KDA (`glm_kda_*`), the clamped SwiGLU + MoE tails (`glm_ffn_*`), and the
  DSA indexer + absorbed-MLA attention (`glm_dsa_*`) - each with its parity harness, plus
  `glm_layer_parity` (the full layer composition) and the binary fixtures under `data/`.
* Phase 3: the `Glm5Model` streaming runner (`src/core/glm_model.{hpp,cu}`, held to the oracle by
  `glm_model_test`); `iq_pack.py`'s glm5-next support (`tools/test_iq_pack_glm.py`) and the
  glm4/glm5 pre-tokenizer in `strata_tokenizer` (`tools/test_tokenizer_glm.py`).

A note on running the kernel parity tests on a 2x V100 machine (Mercury): its GPUs are V100s (sm_70), BELOW
the engine's sm_75 minimum, so the prebuilt archs produce binaries the V100 cannot launch ("no kernel
image"). The kernel sources themselves are arch-portable (the sm_80 features have the fp32 fallback),
so the local test tree relaxes the configure guard one line (`sed -i 's/_base LESS 75/_base LESS 70/'`)
and configures with `-DCMAKE_CUDA_ARCHITECTURES=70` - an uncommitted, build-tree-local change that
exists only to let the V100s execute the parity suite.

### 8.1 Offset-width audit (second-opinion item 7, 2026-10-03)

Every index product the glm5 kernels and the runner compute was checked against the REAL
dimensions (n_embd 4096, n_ff 12288/2048x288, 1M context).  Findings, none of them bugs:

* Host arena addressing is int64 end to end: the region vectors (`kda_S_`, `dsa_lat_`, ...),
  every scratch offset, and the expert slab base (`ffn_*_exps + slab*e`, 2.4e9 elements over
  INT32_MAX) all use int64/size_t.
* Device-side indexing is `(size_t)` wherever a product grows (latent cache row
  `kv_lora*cell` = 5.4e8 at 1M ctx; q_absorb; gemv rows).  The largest product any kernel
  forms in 32-bit `int` locals is bounded by one call's activation size - the runner is
  single-token and the parity harness 8 tokens - at most ~2e7, two orders below 2**31.
* The 2.2B-element-per-layer expert tensors never appear as a kernel-internal offset: the
  kernel sees one expert's slab (8.4M elements), the host adds the int64 layer base.

The audit's boundary: the engine's own ggml-side kernels (native_expert, the dense GEMV path)
are out of scope here - they run qwen4exp production models already.

### 8.2 The big synthetic and the llama.cpp cross-check (second-opinion item 7, 2026-10-03)

`tools/glm_synth_gguf.py --big` (seed 777) builds the second fixture: 45 layers in the released
cadence (DSA at il%4==3, layers 3..43), n_embd/q_lora/kv_lora/n_ff_exp multiples of 256 (the IQ
row width), top-8 of 32 experts, kpool 4 with top_k 256 (a 512-token sequence crosses 128 pools
and discards half), sinkhorn iterations 5, hot FFN scales (SwiGLU clamps fire) and +8 dt spikes
(KDA gates reach the lower bound).  tools/cover_audit.py PASSES on it - every branch the first
fixture left at zero fires here.

The cross-check: llama.cpp (master, CPU) streams the same 512 tokens; the f64 oracle teacher-forces
them.  Result, all 512 columns:

  * per-column max |logit diff|: p50 9.0e-05, p99 2.5e-03 (logit std 0.8);
  * KL(oracle || llama) on the softmax: p50 2.0e-09, p99 1.7e-07;
  * argmax agreement 510/512; BOTH flips are explained by MoE selection near-ties - the rank-8 vs
    rank-9 biased-score gap is 1.2e-5 (col 108) and 1.7e-6 (col 250), inside f32-vs-f64 noise, so
    the two implementations legitimately pick different 8th experts.  This is the second opinion's
    item 2 mechanism verbatim, and it is why the Phase-4 acceptance gates are top-1 agreement +
    mean/p99 KL, not a per-logit 2e-3 tolerance.

Phase-4 acceptance thresholds calibrated on this pair: KL p99 < 1e-5, top-1 agreement > 99%,
with every disagreement traceable to a selection-boundary gap below ~1e-4.  The 272aad8b pin is
unusable for the cross-check (its graph_reserve aborts on the scatter dead-slot bug master
fixed); /tmp/llama-ref on Mercury holds the master build + tools/llm_logits_dump.cpp.

### 8.2.1 The REAL-model cross-check (UD-IQ1_S vs llama.cpp, 2026-10-04) and the quantized gates

The §8.2 gates were calibrated F32-vs-F32 (llama on the F32 fixture).  The REAL-model cross-check
runs llama.cpp (upstream master, CPU) against the PACK (device MMVQ dense + CPU IQ experts) on the
same quantized GGUF — both sides quantized, different kernels.  Result on tokens 1..8 (seam-level
bisect via GLM_CB_DIR vs llama's LLM_CB_DIR, the full recipe in NEXT-STEPS.md):

  * layer 0: every seam within 3e-3 (fp noise); router chain (logits → biased probs → top-8 →
    renormalized weights) EQUAL at the probed layer; MoE selection never flips;
  * first divergence: L3 ffn_out (the first routed-expert layer), growing smoothly; the L18 ×68
    state explosion (real model behavior, both sides) amplifies it;
  * per-expert h/down vs an exact dequant+fp32 reference: ours 3.0–4.3e-2, llama 1.5–4.5e-2 —
    EQUALLY FAITHFUL; the gap is each side's activation-quantization error (ours q8_1 on the
    device dense path, llama Q8_K on CPU), re-rolled at every re-quantization of an already
    slightly-different stream;
  * aggregate (8 tokens): top-1 5/8, p50|Δlogit| 0.31–1.29, cos-sim 0.70–0.97 — the shape tracks
    the target distribution's peakedness (positions 5–7 recover to 0.95), NOT a cumulative bug.

CONCLUSION (owner-accepted 2026-10-04): no structural divergence; the difference is accumulated
quantized-kernel noise.  **Quantized-mode acceptance gates** (for pack-vs-llama comparisons on
the same quantized GGUF — the F32 gates above DO NOT apply):

  * every argmax/logit finite, same vocab, no NaN through all trunk layers;
  * p50|Δlogit| < 1.5 AND cosine(logits) > 0.65 per teacher-forced position;
  * MoE selection (top-8 ids) equal at every probed layer; expert compute within 2× of llama's
    distance to the exact dequant+fp32 reference;
  * no positional drift: the divergence must NOT grow monotonically with position (a state/cache
    bug does; noise recovers when the distribution sharpens).

Improving the seed is M2-adjacent (owner decision: fold into M2): switching the device dense path
to Q8_K-style activation quant would cut the divergence seed vs llama, at the cost of reworking
every MMVQ call site — to be decided where the path is rebuilt anyway (IQ1_S kernel, expert
tiering).

### 8.3 The tokenizer diff (second-opinion item 5, closed 2026-10-03)

The real GLM-5.3-Flash tokenizer.json (154,856 BPE tokens + 36 specials, 321,649 merges; the
CHATGLM4 pre-regex appears verbatim, `(?i:'s|'t|...)` included) is now an accepted fixture:
`tools/vocab_gguf.py` writes it as a vocab-only GGUF (written with llama.cpp's own gguf-py
GGUFWriter - my hand-rolled writer desyncs llama's reader on large string arrays), and
`tools/tokenizer_diff.py` compares `llama-tokenize` against `strata_tokenizer` id for id.

One real bug: the ignore_merges whole-vocab lookup ran on the RAW pre-token piece instead of the
byte-MAPPED one - identity for ASCII (every earlier test passed), wrong for any non-ASCII
exact-vocab piece.  Fixed to match llama-vocab.cpp L634 (map, then look up).

Result: exact agreement on the boundary corpus (504 ids) and its 60x repetition (30,360 ids) -
digits, combining marks, CRLF (the harness must read with newline=""), CJK, emoji, contractions,
specials, byte edges.  The transcription is verified at the id level, not just structurally.

### 8.4 The runner takes the big fixture (2026-10-03)

`glm_model_big_test` (a ctest fixture: tools/glm_big_fixture.py generates the 1.2 GB seed-777
GGUF into the build tree; the f64 expectation is committed under data/glm5-big-dumps/) streams
the same 512 tokens the llama.cpp cross-check used through the CUDA runner:

  * streaming vs batched at prefixes 100/250/511: bitwise 0;
  * state determinism (A, reset, B, reset, A): bitwise;
  * vs the f64 oracle over all 512 columns: **KL p50 2.0e-9, p99 5.1e-11-bounded, max 6.8e-9,
    argmax 512/512, median column |diff| 1.6e-5, zero columns beyond KL 1e-4** - i.e. on the
    real-cadence geometry with discards, clamps and gate saturation firing inside the runner,
    the runner matches the oracle BETTER than llama.cpp does (llama's 2 flips are its f32
    rounding, the runner's f32 order happens to match the f64 oracle's every argmax);
  * glm_model_test gains the advisor's two missing gates: state determinism and mutation
    sensitivity (a 4 KB mid-file weight corruption must move the logits > 1e-2, proving gate 1
    can catch a wrong model).

Suite: 10/10 (the 8 previous + the fixture generator + the big test).  One process note: the
commit-time expectation file was briefly TRANSPOSED (double transpose) - the gate caught it
immediately, which is the system working as designed.

## 9. Phase 4 - the source file and the wiring plan (opened 2026-10-03)

Source: unsloth/GLM-5.3-Flash-GGUF **UD-IQ1_S** (3 shards, 93.0 GB; sha256s recorded), downloaded
to the OS NVMe per the owner's instruction (the HDD and the second NVMe stream too slowly for
mmap serving; the second NVMe only stages).  The owner moved the Qwen IQ3_S quants (79 GB) to the
second NVMe to make room.

### 9.1 The file's actual recipe (parsed from shard 2's tensor directory)

* token_embd / output: **Q4_K**; dense-lead ffn_gate/up **Q5_K**, ffn_down **Q6_K**;
  attn_output **Q5_K** (Q6_K at blk.11), attn_q_a Q5_K/Q6_K;
* the absorbed MLA projections + the whole indexer attention set: **Q8_0**;
* ffn_gate_inp, indexer.proj, indexer_compressor_ape: **F32** (exactly the packer's F32 set);
* routed experts, DYNAMIC PER LAYER: gate and up ALWAYS share one type per layer (IQ1_S or
  IQ2_XXS), down is IQ3_XXS (IQ4_XS at blk.11/12) - so the pack's single gu_type per layer
  holds as designed; the types are **IQ1_S, IQ2_XXS, IQ3_XXS, IQ4_XS** - NOT the types the
  study's §3.5 assumed (IQ1_M/IQ2_XS/IQ2_S/IQ3_S gate-up, Q2_0/IQ4_NL down).  The advisor's
  "extend the native path to the source's types" is therefore resolved by the best possible
  answer: `native_fmt` is ggml-cpu trait-driven, so the four types work as-is (ggml-cpu has
  vec_dot for every i-quant); no new dequantizers.

### 9.2 The IQ1_S-first milestone: pack-backed Glm5Model

Keep the verified runner math (mHC/KDA/DSA/hc kernels) and teach it to serve the pack:

1. load index.txt + native_experts.txt; dequant the DENSE tensors to the existing F32 device
   arena at load (token_embd+output alone are 2x2.5 GB in F32 - fits alongside ~24 GB of
   untouched V100);
2. route the MoE loop through the engine's CPU native expert path (FileExpertSource mmap +
   prefetch, native_gu_rows/native_down_rows) instead of the device-gemv F32 slabs;
3. fix kNativeActBytes/kNativeHBytes: they are sized for qwen4exp's 2560/640-wide activations;
   GLM needs 4096/2048 (Q8_K ~4.2 KB / ~2.1 KB) - a hard overflow if left as-is;
4. un-refuse glm5-next in the driver gate once the above serves; FAMILIES entry last.

The full production-driver graph path (captured graphs, the server, the glm5-next WeightTable
in layout.cpp) stays a later milestone; the layout tables are qwen4exp-specific down to the
tensor names.
