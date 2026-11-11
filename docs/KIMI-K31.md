# Running Kimi K3.1 Pro - a port study for Maya's GLM engine

What it would take for [moonshotai/Kimi-K3.1-Pro](https://huggingface.co/moonshotai/Kimi-K3.1-Pro) to run on
this engine, op by op and tensor by tensor, with every number the decision needs and every place the
current GLM-5.3-Flash engine would break. Written 2026-11-04 against v1.1.53 (`7c4aa56`); all `file:line`
references are to that tree unless they name another repository.

K3.1 is the same *kind* of model GLM-5.3-Flash is - a 3:1 Kimi-Delta-Attention / sparse-attention hybrid
MoE with a NoPE MLA and a DeepSeek-style lightning indexer - which is why a port is worth studying rather
than dismissing. But it is not a bigger GLM. Four of its pieces have no counterpart in the GLM engine
(Block Attention Residuals instead of hyper-connections, a latent MoE, the SiTU activation, a per-token
indexer instead of GLM's key pools), three of its sizes overflow fixed arrays all over the GLM fast path
(16 routed experts a token, 896 experts a layer, a 7168-wide residual), and its experts ship as MXFP4,
which the GLM path cannot read. And at 2.78 T parameters it is roughly nine times GLM-5.3-Flash.

The study's conclusion, stated first: **a reference implementation and a refusal are cheap and are on
this branch's plan (Phases 1-2); serving it is kernel work on the scale of a new runner, not a
configuration of the GLM one (Phase 3).** The GLM engine must refuse a K3.1 file loudly; today it would take
the file's geometry from GLM-5.3 defaults and fail, if it fails at all, far from the cause (§4.8).

---

## 1. The model, exactly

K3.1 Pro is Kimi K3 with the full-attention layers' dense attention replaced by DeepSeek Sparse Attention:
the model card says "we replace the full-attention Gated MLA layers with Gated DeepSeek Sparse Attention
(Gated DSA) layers". Moonshot ships **no modeling code for K3.1**; its `auto_map` keeps only `AutoConfig`.
Everything outside the indexer is Kimi K3, whose code Moonshot does publish.

From the released `config.json` (`KimiK31ForConditionalGeneration`, `model_type: kimi_k31`, text config
`model_type: kimi_linear`, modified-MIT "Kimi K3.1-Pro" licence, released 2026-11-02):

| | |
| --- | --- |
| Total / active params | 2.78 T (2,780,110,363,392 counted from the shard headers, vision included) / 104.4 B per token (§3.1) |
| Native weights | routed experts **MXFP4** (compressed-tensors `mxfp4-pack-quantized`, group 32, E8M0 scales; quantization-aware training per the card, "MXFP4 weights / MXFP8 activations"); everything else BF16, with `A_log`, `dt_bias`, the conv weights, the KDA output-norm weight and the router bias in F32 |
| Layers | 93 (`hidden_size` 7168). Full-attention (DSA) layers, 0-indexed: **3, 7, 11, ..., 87, 91 and 92 - 24 of them**, i.e. `il % 4 == 3` plus the last layer; the other **69 are KDA**. The config's lists are 1-indexed (`full_attn_layers` = `dsa_layers` = 4, 8, ..., 92, 93; `is_kda_layer(i)` tests `(i + 1) in kda_layers`) |
| Residual stream | **Block Attention Residuals** (`attn_res_block_size` 12): no `x + f(x)` anywhere; each sublayer's input is a softmax mix over completed block summaries plus the current block's partial sum (§2.2) |
| Linear attention (69 layers) | Kimi Delta Attention: 96 heads × head_dim 128, `short_conv_kernel_size` 4, `gate_lower_bound` -5.0, `use_full_rank_gate` true (full-rank output gate) |
| Full attention (24 layers) | MLA, NoPE (`mla_use_nope` true - no RoPE anywhere in the model): 96 heads, `q_lora_rank` 1536, `kv_lora_rank` 512, `qk_nope_head_dim` 128, `qk_rope_head_dim` 64 (un-rotated, §5.5), `v_head_dim` 128, `mla_use_output_gate` true |
| Indexer (24 layers, `dsa_config`) | `index_num_heads` 32, `index_head_dim` 128, `index_topk` 2048, `index_rope_dim` 0, `index_k_norm_type` "layernorm", `index_topk_freq` 1, `index_skip_topk_offset` 1 - a per-token "lightning" indexer, no key pooling (§2.5) |
| MoE | 896 routed experts + 2 shared, **top-16**, `moe_intermediate_size` 3072 inside a **3584-wide latent** (`routed_expert_hidden_size`, `latent_moe_use_norm` true), sigmoid scoring, `noaux_tc` bias, renormalised, `routed_scaling_factor` 1.0; **layer 0 dense** (`intermediate_size` 33792), layers 1-92 MoE |
| Activation | SiTU-GLU, `activation_situ_beta` 4.0, `activation_situ_linear_beta` 25.0 (dense, routed and shared FFNs) |
| MTP | none (`num_nextn_predict_layers` 0) |
| Vision | MoonViT-V2 (401 M per the card) - out of scope here |
| Vocab / context | 163,840 (tiktoken-style, shared with Kimi K2/K3) / 1,048,576 |
| Routed-expert share | 896 × 92 × 3 × 3584 × 3072 = **2.72 T params = 97.9 % of the parameters, 92.7 % of the bytes** |

Note on `top_k: 50` in the text config: it is a transformers generation default written into every
config dict, not the MoE top-k (which is `num_experts_per_token` 16).

### 1.1 Sources and pins

| What | Pin | Role |
| --- | --- | --- |
| Moonshot Kimi-K3 code | HF `moonshotai/Kimi-K3` @ `f831ab66814297da540d832a5235f8e904f29d06` (`modeling_kimi_linear.py`, `modeling_kimi_k3.py`, `configuration_kimi_k3.py`) | the spec for everything outside the indexer. "ml.py:N" below = `modeling_kimi_linear.py` line N |
| K3.1 release | HF `moonshotai/Kimi-K3.1-Pro` @ `6d0a0202956796543364d666ab19c6ad1a494ee1` | config, shard headers, `configuration_kimi_k3.py` (differs from K3's only by storing `dsa_config` and `model_type`) |
| fla | `fla-core` 0.6.0 (`fla.ops.kda.chunk_kda`, `fused_recurrent_kda`, `fla.modules.ShortConvolution`, `FusedRMSNormGated`) | the kernels Moonshot's code calls; its KDA gate file says it is "modified and supported by the Moonshot AI Team" |
| **Indexer, primary** | SGLang branch `kimi-k3.1-support` @ `accf045c83fb361d7a8f07a4dadf45b2ea2b24b2` (the real change is `4cb6e857`, on main `8884b4b4`) | K3.1 support; every K3.1 switch is a config flag on SGLang's shared DSA indexer |
| **Indexer, independent** | vLLM PR #65685 "[Model] Kimi K3.1 Pro" @ `9d6ac73f3dc6a81a9c0b50fa602238f5244fd757` (`vllm/models/kimi_k3/nvidia/dsa.py`) | a second implementation; agrees with SGLang on every semantic point |
| Cross-check | SGLang `kimi-k3.1-day-0` @ `ed4b1752`; DeepSeek-V3.2-Exp `inference/model.py` @ `87e509a2` | same math; the DeepSeek reference is the comparison point in §2.5 |
| llama.cpp | `master` @ `5851a8e505f4d5fbb183f2847698eb6155d176b7`: `conversion/kimi_k3.py`, `src/models/kimi-k3.cpp` (K3 since `ad1de39e0`, #26185), `src/models/deepseek32.cpp` | the GGUF names and the K3 graph; no K3.1 support upstream |

Both indexer sources are **unmerged** (vLLM main @ `75f67614` still builds the K3 MLA with `indexer=None`;
neither engine's main registers `KimiK31ForConditionalGeneration`) and may be force-pushed: re-pin before
building on them. The SGLang cookbook names a branch `kimi-k31-pro` and a commit `9ab3c28176`; neither is
public.

### 1.2 K3 → K3.1, all of it

The two `config.json` files differ only in: the architecture and model-type names; `auto_map` (trimmed);
`text_config.dsa_config` (new); and `quantization_config.ignore`, which now also names the router gate and
the latent projections - tensors that were already BF16 in K3's checkpoint. The index lists 497,340 tensors
against K3's 497,220: **the 120 extra are the five indexer tensors on each of the 24 MLA layers**, and no
other name, shape or dtype differs (headers compared on shards 1, 2, 4, 13, 48, 93, 94). Consequence worth
remembering: **for any sequence of 2048 tokens or fewer, K3.1's forward pass is exactly K3's**, because the
indexer then selects every visible key (§2.5).

## 2. The forward pass, op by op

Notation: `x` is a token's 7168-vector, `RMS(v, w, eps) = w ⊙ v / sqrt(mean(v²) + eps)`. Moonshot's
`KimiRMSNorm` (ml.py:226-236) computes in fp32, casts to the input dtype, then multiplies by the weight;
**its default eps is 1e-6**, and two norms rely on that default (§5.2).

### 2.1 Embedding
`h = embed_tokens[ids]` (163,840 × 7168). No scale, no positional embedding; `lm_head` is untied.

### 2.2 Block Attention Residuals (ml.py:984-1046, 1075-1088)

Two things cross a layer boundary, both per token (nothing crosses tokens, so decode needs no cache for
them): `blocks`, a stack of completed block vectors, and `partial`, the sum of the current block's
sublayer outputs. Per layer `L`:

```
if L > 0:  a = mix(blocks + [partial], w_attn[L])        # skipped at layer 0: a = the embeddings
if L % 12 == 0:  blocks.append(partial); partial = None  # layer 0 pushes the embeddings as block 0
o = attn_L(input_layernorm(a))                           # KDA or MLA
partial = o if partial is None else partial + o
f = mix(blocks + [partial], w_ffn[L])
partial = partial + ffn_L(post_attention_layernorm(f))   # dense MLP at layer 0, MoE after
```

and after layer 92, `h = mix(blocks + [partial], w_out)` feeds the final norm. The mix (ml.py:1075-1088):

```
s_i = Σ_d w_d · v_{i,d} / sqrt(mean_d(v_i²) + 1e-5)     # w = res_norm.weight ⊙ res_proj.weight (fp32)
p   = softmax_i(s)
mix = Σ_i p_i · v_i                                      # the RAW candidates, in fp32
```

The query is a static learned vector per site; candidates are normalised only to be scored. Blocks are
pushed at layers 0, 12, 24, ..., 84, so a site mixes at most **9 candidates** (8 blocks + the partial, from
layer 85 on). After a push the partial restarts from that layer's attention output - the incoming residual
enters only through the block stack. Every layer carries both sites' weights in the checkpoint, including
layer 0's unused `self_attention_res_*`.

### 2.3 KDA layer (69 layers; ml.py:477-663)

```
q, k, v = W_q x, W_k x, W_v x                     # 7168 -> 12288 = 96 × 128 each
q, k, v = silu(causal_conv4(q)), ...              # depthwise, width 4, W[:, 3] on the current token
g    = W_fb (W_fa x)                              # 7168 -> 128 -> 12288, no activation between
beta = sigmoid(W_b x)                             # 96, one per head
gk[h,c] = -5 · sigmoid( exp(A_log[h]) · (g[h,c] + dt_bias[128h + c]) )      # in (-5, 0)
q̂ = q / sqrt(Σ q² + 1e-6) · 128^-0.5 ;  k̂ = k / sqrt(Σ k² + 1e-6)          # per head
per head, state S (128 × 128, fp32, zero at sequence start):
    S ← diag(exp(gk_t)) S                         # decay per KEY channel
    S ← S + k̂_t ⊗ beta_t (v_t − Sᵀ k̂_t)           # delta rule
    o_t = Sᵀ q̂_t
o   = RMS(o, o_norm.weight, 1e-5) ⊙ sigmoid(W_g x)    # per head, weight [128] shared; W_g full rank
out = W_o o                                       # 12288 -> 7168
```

The gate is the **lower-bound form**, not Kimi-Linear's `-exp(A_log)·softplus(g + dt_bias)`: fla's
`chunk_kda` and `fused_recurrent_kda` switch to `lower_bound · sigmoid(exp(A_log) · (g + dt_bias))` whenever
`lower_bound` is set (Moonshot passes -5.0 and `safe_gate=True`). `A_log` is per head; the checkpoint
stores 128 entries for the 96 heads (§5.1).

### 2.4 MLA layer, NoPE and gated (24 layers; ml.py:335-474)

```
qr     = RMS(W_qa x, q_a_layernorm, eps_lat)              # 7168 -> 1536; also the indexer's q input
q      = W_qb qr                                          # 96 heads × 192 = [q_nope 128 | q_rot 64]
[c, kr] = W_kva x                                         # 7168 -> 576 = [512 | 64]
c      = RMS(c, kv_a_layernorm, eps_lat)                  # kr is NOT normalised
[k_nope | v] = W_kvb c                                    # 96 × (128 + 128)
K_h    = [k_nope_h | kr]                                  # kr: one 64-d key shared by all heads
o_h    = Σ_s softmax_s( q_h · K_{h,s} / sqrt(192) + mask(t, s) ) v_{h,s}
out    = W_o ( o ⊙ sigmoid(W_g x) )                       # the "Gated" in Gated DSA; W_g: 7168 -> 12288
```

`rotary_emb` is `None` and `assert self.use_nope` holds (ml.py:396, 403): the 64 "rope" dims are plain key
dims and position ids are ignored, so weight absorption is exact. Absorbed, a query is
`[W_uk,hᵀ q_nope_h (512) | q_rot_h (64)]` against a **576-wide cache row** `[c (512) | kr (64)]`, and values
come from the 512 part only (`o_h = W_uv,h Σ p c`). `eps_lat` is 1e-6 in Moonshot's code and 1e-5 in every
engine (§5.2). In K3 `mask` is causal; in K3.1 it is causal plus the indexer's selection:

`mask(t, s) = 0` if `s ≤ t` and `s ∈ sel(t)`, else `-inf`.

### 2.5 The lightning indexer (24 layers)

Source: SGLang `kimi-k3.1-support` @ `accf045c` and vLLM PR #65685 @ `9d6ac73f`, which agree; Moonshot
publishes no indexer code. Per DSA layer, with `x` the same normed input the MLA projections read and `qr`
the post-norm q latent of §2.4:

```
iq = W_iqb qr                       -> [T, 32, 128]    no RoPE, no Hadamard rotation
ik = LayerNorm(W_ik x; k_norm.weight, k_norm.bias, eps 1e-5), fp32      -> [T, 128]
iw = (W_ip x) · 32^-0.5 · 128^-0.5  -> [T, 32]          (= /64)
score[t, s] = Σ_h iw[t,h] · ReLU(iq[t,h] · ik[s])
score[t, s] = -inf for s > t        (and for other sequences)
sel(t) = top-k(score[t], min(2048, t + 1))              a set; order unspecified
```

Pinned details:

* **Inputs.** `iq` reads `qr` *after* `q_a_layernorm` (SGLang `forward_mla.py` passes the normed `q` as
  `q_lora`; vLLM `layers/mla.py:191,234`). `ik` and `iw` read `x`.
* **Key norm.** A mean-subtracting LayerNorm with weight and bias, computed in fp32 (SGLang
  `dsa_indexer.py:389-402` with `index_k_norm_fp32`; vLLM `KimiK3DSAIndexerKNorm`, `dsa.py:80-101`), eps =
  `rms_norm_eps` 1e-5 in both. DeepSeek-V3.2's reference uses 1e-6.
* **Scales.** `softmax_scale = 128^-0.5` and the head scale `32^-0.5` multiply into exactly 1/64. The ReLU
  comes before the head weighting; the head weights are raw projections, so **scores can be negative**.
* **Causality.** A token's key is written before scoring, so query `t` sees keys `0..t` inclusive. With
  `t + 1 ≤ 2048` visible keys the selection is all of them and the layer is exactly dense causal MLA; to
  exercise the indexer at all, a test needs more visible keys than `index_topk`.
* **Application.** Engines gather the selected rows into a sparse-MLA kernel; DeepSeek's reference adds a
  `-inf` mask over the unselected keys on top of the causal mask. The two are the same function.
* **Nothing is always kept.** No sink, local window or forced self token (`index_init_tokens` /
  `index_local_tokens` are absent; SGLang's `_mask_init_and_local_tokens` is a no-op for K3.1).
* **`index_topk_freq` 1 / `index_skip_topk_offset` 1 are inert** (§5.4): every DSA layer runs its own
  indexer. Both engines refuse `index_topk_freq != 1` and `index_rope_dim != 0` (SGLang
  `configs/kimi_linear.py:137-173`; vLLM `dsa.py:603-609`); a port must refuse them too, not ignore them.
* Everything else in the engines - FP8 q/k with power-of-two scales, bf16 rounding of the head weights,
  an optional MXFP4 indexer cache in vLLM - is serving numerics and moves only near-ties at the cut.

Compared with DeepSeek-V3.2-Exp's indexer: 32 heads instead of 64, no RoPE on the first 64 dims, no
Hadamard rotation, LayerNorm eps 1e-5 instead of 1e-6, and only the MLA layers carry one.

### 2.6 FFNs: SiTU-GLU (ml.py:64-82)

```
SiTU(g, u) = 4 · tanh(g / 4) · sigmoid(g) · 25 · tanh(u / 25)       # fp32, cast back
dense (layer 0):  W_down SiTU(W_gate x, W_up x)                      # 7168 -> 33792 -> 7168
```

The "gate" half is `w1`/`gate_proj`, the "up" half `w3`/`up_proj`. SiTU's linear half is `25·tanh(u/25)`,
not `u`.

### 2.7 Latent MoE (layers 1-92; ml.py:666-874)

```
logits = W_r x                        # fp32: x and W_r are cast to float; W_r [896, 7168]
p   = sigmoid(logits)
sel = top16(p + e_score_correction_bias)          # the bias picks, it never weighs (noaux_tc)
w   = p[sel] / (Σ p[sel] + 1e-20) · 1.0           # renormalised, routed_scaling_factor 1.0
z   = W_ld x                          # routed_expert_down_proj: 7168 -> 3584 (the latent)
r   = Σ_k w_k · W2_e( SiTU(W1_e z, W3_e z) )      # experts live in the latent: 3584 -> 3072 -> 3584
r   = RMS(r, routed_expert_norm, 1e-5)            # AFTER the weighted sum
out = W_lu r + shared(x)              # routed_expert_up_proj 3584 -> 7168; shared: SiTU MLP 7168 -> 6144 -> 7168
```

The router and the two shared experts (fused to 6144) see the full-width `x`; only the routed experts see the
latent. The weighted sum is accumulated in fp32 (`moe_infer`, ml.py:867-873). There is no residual inside
the block: the residual is §2.2.

### 2.8 Head
`h = mix(blocks + [partial], w_out)`, `RMS(h, norm, 1e-5)`, `logits = lm_head h`. No softcap, no scaling.

### 2.9 Precision contract

Keep in fp32 whatever the reference loads: `A_log`, `dt_bias`, the three conv weights, `o_norm.weight`
(all F32 in the checkpoint), and **`e_score_correction_bias`** - rounding the router bias to bf16 can flip a
top-16 selection near a tie, which is a different expert, not a rounding error. The router logits, the
attention-residual scores and mix, SiTU, the MoE weighted sum and the indexer LayerNorm are fp32 in the
reference code. Moonshot's classes declare no `_keep_in_fp32_modules`, so a `torch_dtype=bfloat16` load of
its code would round the F32 tensors.

## 3. Tensors and sizes

### 3.1 Checkpoint totals (all 96 shard headers, no tensor data)

The sum matches `model.safetensors.index.json` (`total_size` 1,561,217,377,280, 497,340 tensors). Routed-expert
params count two fp4 values per packed byte; the scale tensors count as bytes only.

| Class | Tensors | Params | Bytes | Share of bytes |
| --- | ---: | ---: | ---: | ---: |
| routed experts, packed fp4 (`experts.E.w{1,2,3}.weight_packed`) | 247,296 | 2,722,740,830,208 | 1,361,370,415,104 | 87.20 % |
| routed experts, E8M0 scales (`...weight_scale`) | 247,296 | - | 85,085,650,944 | 5.45 % |
| **routed experts, total** | 494,592 | **2,722,740,830,208** | **1,446,456,066,048** | **92.65 %** |
| KDA attention, 69 layers | 966 | 30,618,088,704 | 61,258,257,408 | 3.92 % |
| shared experts, 92 layers | 276 | 12,155,092,992 | 24,310,185,984 | 1.56 % |
| MLA attention, 24 layers | 144 | 5,572,657,152 | 11,145,314,304 | 0.71 % |
| latent projections, 92 layers | 184 | 4,726,980,608 | 9,453,961,216 | 0.61 % |
| `lm_head` / `embed_tokens` | 1 + 1 | 2 × 1,174,405,120 | 2 × 2,348,810,240 | 0.30 % |
| dense layer-0 MLP | 3 | 726,663,168 | 1,453,326,336 | 0.09 % |
| router (`gate.weight` BF16 + `e_score_correction_bias` F32), 92 layers | 184 | 590,955,008 | 1,182,074,880 | 0.08 % |
| vision tower + projector | 168 | 447,358,976 | 894,717,952 | 0.06 % |
| **DSA indexer, 24 layers** | 120 | 178,526,208 | 357,052,416 | 0.02 % |
| attention-residual weights (93 × 2 sites + output) | 374 | 2,680,832 | 5,361,664 | - |
| other norms | 327 | 1,719,296 | 3,438,592 | - |
| **Total** | **497,340** | **2,780,110,363,392** | **1,561,217,377,280** | 100 % |

Derived: one expert is 3 × 3584 × 3072 = 33,030,144 params (17,547,264 bytes in MXFP4); the text model's
non-routed blocks are 54.57 B params, **55.75 B with `lm_head`** (the non-expert compute of every token),
56.92 B with the embeddings too. Routed params active per token: 16 × 92 × 33,030,144 = 48.62 B, so
**active = 104.37 B**, matching the card's 104 B. Layer `L` lives entirely in shard `L + 1`.

### 3.2 The file contract: `general.architecture = "kimi-k31"`

The GGUF a K3.1 port reads is upstream's `kimi-k3` layout (`conversion/kimi_k3.py`) under its own
architecture string, plus the indexer. A separate string rather than `kimi-k3` with optional tensors: it
needs a different memory layout (a per-token indexer cache), K3 files stay untouched, and an engine that
does not know K3.1 refuses the file instead of half-loading it. Tensor names (ggml `ne` order, source dtype;
the converter forces the router and `indexer.proj` to F32):

| HF (`language_model.model.`) | GGUF | ne | Source |
| --- | --- | --- | --- |
| `embed_tokens` / `lm_head` / `norm` | `token_embd` / `output` / `output_norm` | {7168, 163840} / {7168, 163840} / {7168} | BF16 |
| `output_attn_res_norm ⊙ _proj` | `output_res_score.weight` | {7168} | fused at conversion |
| every layer: `input_layernorm`, `post_attention_layernorm` | `blk.N.attn_norm`, `blk.N.ffn_norm` | {7168} | BF16 |
| every layer: `self_attention_res_*`, `mlp_res_*` (norm ⊙ proj) | `blk.N.attn_res_score.weight`, `blk.N.ffn_res_score.weight` | {7168} | fused |
| KDA: `self_attn.{q,k,v}_proj` | `attn_q`, `attn_k`, `attn_v` | {7168, 12288} | BF16 |
| KDA: `self_attn.{q,k,v}_conv1d` [12288, 1, 4] | `ssm_conv1d_q/k/v` | {4, 1, 12288} | F32 |
| KDA: `f_a_proj`, `f_b_proj` | `ssm_f_a`, `ssm_f_b` | {7168, 128}, {128, 12288} | BF16 |
| KDA: `b_proj` | `ssm_beta` | {7168, 96} | BF16 |
| KDA: `A_log` [**128**] | `ssm_a` = `-exp(A_log[:96])` | {96} | F32 |
| KDA: `dt_bias` | `ssm_dt.bias` | {12288} | F32 |
| KDA: `g_proj` (full-rank output gate) | `ssm_g` | {7168, 12288} | BF16 |
| KDA: `o_norm`, `o_proj` | `ssm_norm`, `attn_output` | {128}, {12288, 7168} | F32, BF16 |
| MLA: `q_a_proj`, `q_a_layernorm`, `q_b_proj` | `attn_q_a`, `attn_q_a_norm`, `attn_q_b` | {7168, 1536}, {1536}, {1536, 18432} | BF16 |
| MLA: `kv_a_proj_with_mqa`, `kv_a_layernorm` | `attn_kv_a_mqa`, `attn_kv_a_norm` | {7168, 576}, {512} | BF16 |
| MLA: `kv_b_proj` [24576, 512] | `attn_k_b` + `attn_v_b` (split, k_b transposed) | {128, 512, 96} + {512, 128, 96} | BF16 |
| MLA: `g_proj` (output gate), `o_proj` | `attn_gate`, `attn_output` | {7168, 12288}, {12288, 7168} | BF16 |
| **indexer: `self_attn.indexer.wq_b`** | **`blk.N.indexer.attn_q_b.weight`** | {1536, 4096} | BF16 |
| **indexer: `wk`** | **`blk.N.indexer.attn_k.weight`** | {7168, 128} | BF16 |
| **indexer: `k_norm.weight`, `k_norm.bias`** | **`blk.N.indexer.k_norm.weight` / `.bias`** | {128} | BF16 |
| **indexer: `weights_proj`** | **`blk.N.indexer.proj.weight`** | {7168, 32} | BF16 (F32 in the GGUF) |
| layer 0: `mlp.{gate,up,down}_proj` | `ffn_gate`, `ffn_up`, `ffn_down` | {7168, 33792} ×2, {33792, 7168} | BF16 |
| MoE: `block_sparse_moe.gate.weight`, `.e_score_correction_bias` | `ffn_gate_inp`, `exp_probs_b.bias` | {7168, 896}, {896} | BF16 (F32 in the GGUF), F32 |
| MoE: `routed_expert_down_proj`, `_up_proj`, `_norm` | `ffn_routed_down`, `ffn_routed_up`, `ffn_routed_norm` | {7168, 3584}, {3584, 7168}, {3584} | BF16 |
| MoE: `experts.E.w1` / `w3` / `w2` (packed + scale) | `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps` | {3584, 3072, 896} ×2 / {3072, 3584, 896} | MXFP4 (type 39), repacked losslessly |
| MoE: `shared_experts.{gate,up,down}_proj` | `ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp` | {7168, 6144} ×2, {6144, 7168} | BF16 |

The expert axis is outermost, so each expert is one contiguous slab. Metadata (`%s` = `kimi-k31`): upstream's
generic keys (`%s.vocab_size` 163840, `context_length`, `embedding_length` 7168, `block_count` 93,
`feed_forward_length` 33792, `expert_count` 896, `expert_used_count` 16, `attention.head_count` 96,
`attention.layer_norm_rms_epsilon` 1e-5) and its K3 set: `attention.head_count_kv` (a **93-entry array**:
1 on MLA layers, which are served as MQA, 0 on KDA layers), `ssm.conv_kernel` 4, `kda.head_dim` 128,
`kda.gate_lower_bound` -5, `attention.q_lora_rank` 1536, `attention.kv_lora_rank` 512, `attention.key_length`
576, `attention.value_length` 512, `attention.key_length_mla` 192, `attention.value_length_mla` 128,
`rope.dimension_count` 64 (**a width, not a rotation** - nothing is rotated), `expert_feed_forward_length`
3072, `expert_shared_count` 2, `leading_dense_block_count` 1, `expert_weights_scale` 1.0,
`expert_weights_norm` true, `expert_gating_func` 2 (sigmoid), `expert_latent_length` 3584,
`activation.situ_beta` 4, `activation.situ_linear_beta` 25, `attn_res.block_size` 12. **Plus, for K3.1:**
`%s.attention.indexer.head_count` 32, `%s.attention.indexer.key_length` 128, `%s.attention.indexer.top_k`
2048, and `%s.attention.layer_norm_epsilon` 1e-5 (the indexer LayerNorm's eps - an engine that leaves it
unset may run the LayerNorm with eps 0). No `indexer.kpool` and no `indexer.types` keys: there are no key
pools and no shared-selection layers.

### 3.3 Sizes, KV and traffic

Routed experts at common GGUF types (dense part extra; MB per expert in brackets):

| Experts kept / layer | IQ1_S (6.45) | IQ1_M (7.23) | IQ2_XXS (8.52) | Q2_K (10.84) | MXFP4 (17.55) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 896 (all) | 531.8 GB | 595.6 GB | 702.0 GB | 893.4 GB | 1,446.5 GB |
| 448 (50 %) | 265.9 GB | 297.8 GB | 351.0 GB | 446.7 GB | 723.2 GB |
| 224 (25 %) | 132.9 GB | 148.9 GB | 175.5 GB | 223.3 GB | 361.6 GB |

The non-routed part is 55.75 B params read every token: ~34 GB at Q4_K, ~61 GB at Q8_0, ~111 GB in BF16, which
alone is more than GLM-5.3-Flash's entire routed-expert set at 2-bit (~79-86 GB, GLM5-FLASH.md §4). Cache and
state: MLA latent 24 × 576 × 2 B = **27.6 KB per token** (f16), indexer keys 24 × 128 × 2 B = 6.1 KB per
token, so ~29 GB + ~6.4 GB at 1 M tokens; KDA state 69 × 96 × 128 × 128 × 4 B = **434 MB per sequence** plus
~30 MB of conv state. Decode reads 16 × 92 = **1,472 experts per token** (GLM-5.3-Flash: 336): 9.5 GB at
IQ1_S, 25.8 GB at MXFP4. As a study estimate, not a measurement: with every expert resident, expert bytes
alone cap decode near `bandwidth / 9.5 GB` at IQ1_S (about 10 tok/s from 100 GB/s of DRAM); paged from a 7
GB/s NVMe with nothing cached, under 1 tok/s. Below ~25 % of experts kept, quality is unmeasured: published
REAP-style pruning results stop at 50 % (the paper) and community Kimi-K3 reports at ~27 %.

## 4. Mapping onto Maya's GLM engine, and every gap

The engine K3.1 would ride is the GLM-5.3-Flash family (docs/GLM5-FLASH.md): `Glm5Geometry`
(include/strata/core/glm_model.hpp:66-93) read by `Glm5Geometry::from_gguf` (src/core/glm_model.cu:507-619),
the `Glm5Model` runner, its fast path (src/core/glm_fast_path.cu, src/kernels/cuda/glm_fast.cu) and the
prompt path (src/core/glm_prefill.cu, src/kernels/cuda/glm_batch.cu). Side by side:

| | GLM-5.3-Flash (implemented) | Kimi K3.1 Pro |
| --- | --- | --- |
| Layers / width | 45 / 4096 | 93 / **7168** |
| Mixers | 34 KDA + 11 DSA (`il % 4 == 3`) | 69 KDA + 24 DSA (`il % 4 == 3`, **plus 92**) |
| Residual | mHC: 4 streams, 20-step Sinkhorn | **Block Attention Residuals**: block 12, up to 9 candidates |
| KDA | 64 × 128, low-rank output gate (`ssm_g_a`/`ssm_g_b`) | 96 × 128, **full-rank output gate** (`ssm_g`) |
| MLA | 64 heads, nope 256, rope 0, v 256, no output gate, scale 1/√256 | 96 heads, nope 128 **+ 64 un-rotated**, v 128, **sigmoid output gate**, scale 1/√192 |
| Indexer | 32 × 128, **key pools of 4** (gate + APE), 512 pools + tail | 32 × 128, **per token**, top 2048 tokens |
| MoE | 288 + 1 shared, top-8, 2048 wide, clamped SwiGLU, ×2.5 | **896** + 2 shared, **top-16**, 3072 wide in a **3584 latent**, **SiTU**, norm after the sum, ×1.0 |
| Dense lead / MTP | 3 layers / 1 | 1 layer / none |
| Experts on disk | GGUF i-quants | **MXFP4** |
| Decode expert reads / token | 336 | 1,472 |

### 4.1 Applies unchanged (or nearly)

| Piece | Why |
| --- | --- |
| Per-layer KDA/DSA selection | the runner already reads the layer type from `attention.head_count_kv` per layer (0 = KDA), not from a fixed cadence - once the array is read in full (§4.8) |
| KDA recurrence | same gate form: Maya's oracle computes `lb · sigmoid(-(f_b f_a x + dt) · A)` with `A = ssm_a = -exp(A_log)` (ref/glm.py:332, 359-361), which is §2.3's `lb · sigmoid(exp(A_log)(g + dt))`; same key-axis decay, delta rule, l2-normalised q/k, 1/√128 scale, width-4 conv + SiLU, gated RMSNorm output |
| Router semantics | sigmoid, bias for selection only, renormalise, scale - the `router_sigmoid` contract (include/strata/kernels/router_sigmoid.hpp:5-17); E changes (the kernel stops at 512, src/kernels/cuda/router_sigmoid.cu:85-88) and so does the renormalisation floor (it clamps the sum at 2^-14, :71, where Moonshot adds 1e-20; §4.6); k = 16 is within its 64 (:81-84) |
| Shared experts | `n_ff_exp * n_shared` everywhere (glm_fast_path.cu:976) - two shared experts are k-generic; but the fast path adds their output inside `moe_down`'s routed sum (`sh_out`, src/kernels/cuda/glm_fast.cu:4926), and K3.1's are added only after that sum's norm and `W_lu` (ref/kimi_k31.py:975-984 on this branch), so that fusion goes |
| Expert memory tiers, mmap source, prefetch | parameterised by `(n_layers, n_expert, blob_bytes)` at runtime (GLM5-FLASH.md §3.1, §4) |
| CPU native activation buffers | the lane's gate/up input is the latent `z`, not `x` (ref/kimi_k31.py:946, 997, 1010 on this branch): 3584 values as Q8_K (4088 B) in `kNativeActBytes` 8192; `kNativeHBytes` 4096 holds 3072 (3504 B) (include/strata/kernels/cpu/native_expert.hpp:19-22) |
| Serving | the APIs are model-agnostic |

### 4.2 Residual: mHC → Block Attention Residuals (new kernel family, runner restructure)

The two have nothing in common but their position in the layer. mHC keeps four parallel streams per token,
mixes them through a data-dependent Sinkhorn-normalised 4×4 matrix and writes back through gates. Block
Attention Residuals keep a *growing* stack (1 to 9 vectors of 7168), score each candidate against a static
per-site vector after an RMS normalisation, softmax over the candidates, and mix the **raw** candidates;
the "write" is a plain addition to the partial sum, and every 12 layers the partial is pushed. None of the
`glm_hc` kernels apply. The kernel itself is small (≤ 9 dot products and a 9-way softmax per site); the cost
is in the runner, whose per-layer state becomes a variable-length stack rather than a fixed `(n_embd, hc)`
block, and in the prompt path, which holds `T × 9 × 7168` fp32 at the deepest layers.

### 4.3 KDA: 96 heads, full-rank output gate, `A_log` slice (small)

The `glm_kda` family carries over with three deltas: 96 heads (state 96 × 128 × 128 fp32 = 6.3 MB a layer
per sequence); the output gate is one full-rank 7168 → 12288 projection (`ssm_g`) instead of the low-rank
`ssm_g_a`/`ssm_g_b` pair; and the converter must slice `A_log` to the first 96 entries and check the rest is
zero (§5.1). Note `Glm5Geometry::d_inner()` is `kda_head_dim * n_head` - correct for K3.1 only because its KDA
and MLA head counts are both 96.

### 4.4 MLA: the 64-wide un-rotated tail, the output gate, the scale (moderate)

`from_gguf` refuses `attention.key_length != kv_lora_rank` - "nope-only MLA holds the bare latent; a rope
tail is not supported" (glm_model.cu:601-607) - and the oracle's and `glm_dsa`'s absorbed attention
score against a 512-wide latent with scale `1/sqrt(qk_nope)` (ref/glm.py:463). K3.1 needs the 576-wide
cache row of §2.4 (512 normed latent + 64 raw shared key), queries `[W_ukᵀ q_nope | q_rot]`, values from the
first 512 only, scale `1/sqrt(192)` (= `key_length_mla`), and `sigmoid(W_g x)` applied to the attention
output before `W_o`, which GLM's MLA does not have. The refusal at :601-607 is correct today and must stay
until that kernel exists. DeepSeek's `sparse_attn` is not that kernel either: it is K = V attention over one
512-wide row with a learned sink (src/ds4/ds4_kernels.cuh:222-224), and for any other head dim it prints a line
and returns without an error (src/ds4/ds4_kernels.cu:4487-4490); only its shape, a gather of the selected rows,
carries over.

### 4.5 Indexer: GLM's key pools at kpool = 1, with a wider selection (moderate)

GLM's indexer (`glm_dsa`, include/strata/kernels/glm_dsa.hpp; ref/glm.py:428-471) averages each pool of 4
keys through a per-key-dim softmax over gate + APE, scores pools, selects `top_k / kpool` = 512 pools
among `n_ctx / 4`, and appends the incomplete tail. K3.1's indexer is the same one at kpool = 1: a softmax
over a pool of one key is 1, so the pooled key is the key and the gate and APE drop out
(src/kernels/cuda/glm_dsa.cu:34-58); visibility `(pos + 1) / kpool` becomes `pos + 1` (:86); the tail loop
runs `kpool - 1` = 0 times (:101-106). The score (ReLU'd 32-head dot products against weights prescaled by
`1/sqrt(key_dim · heads)`, which is K3.1's 1/64 at 128 × 32, glm_dsa.hpp:13-14) and the key LayerNorm with
weight and bias (`layer_norm`, ref/glm.py:430-431) are ops GLM already runs, and DeepSeek's `index_score` is
the same formula over per-token keys, BF16 among its key formats (src/ds4/ds4_kernels.cuh:191-199). What is
new: no compressor gate or APE tensors; the geometry gate, which refuses kpool ≤ 1
(src/core/glm_model.cu:589-592); scores over `n_ctx` keys instead of `n_ctx / 4` pools; and the selection,
2048 of up to 1 M candidates per query per layer - four times the candidates and four times the picks of GLM's
pool selection. GLM's per-op select is O(n_vis²) (glm_dsa.cu:90-100), fine for fixtures; its fast radix
select holds at most 1024 winners in one block (src/kernels/cuda/dsa_topk.cuh:23-26). DeepSeek's segmented
radix `topk` takes any k, ties to the lower position (ds4_kernels.cuh:200-215). In a decode window a row of
32K positions or more is split into segments of ~16K (:214; src/ds4/ds4_kernels.cu:4109-4110), and at k = 2048
more picks than its merge's `kMergeCap` 8192 can pass once a row has five segments, past 64K positions (on a
device with 64 KB of shared memory a block the cap is 7680, reached from four, :4119); the merge then takes every
candidate instead, the same picks at a speed not yet measured (:1537-1544). Add a 128-wide key cache per token
per layer (§3.3). Its parity is defined by the tie rule of §5.3, not by index equality.

### 4.6 MoE: k = 16, E = 896, n_embd = 7168, latent, SiTU (the hard wall)

**Every fast-path MoE structure is sized for at most 8 routed experts a token, and the existing host guards
are fall-back switches, not refusals.** The k ≤ 8 sites, with what k = 16 does to each:

| Where | What | At k = 16 |
| --- | --- | --- |
| include/strata/kernels/glm_fast.hpp:327-343 | `MoeRequest` `ids[8]`, `w[8]`, `promo_ptr[8]`, `pred[8]`, `pf_ids/pf_ptr[8]`, `ahead[kAhead][8]`, `cpu_src[8]`, and `x[4096]` | host-mapped ring entries overflow into the next |
| glm_fast.hpp:349-361 | `MoeResponse` `ptr[8]`, `ready[8]`, `cpu_part[4096]` | overflow |
| glm_fast.hpp:398, 405-406 | `kMaxWinTokens` "16 x 8 entries"; `cpu_take` packs 4 bits per f, f = 0..8, in one 64-bit word | the plan encoding cannot hold k > 15 |
| src/kernels/cuda/glm_fast.cu:3954-3957 | `moe_route_kernel` `s_p[512]`, `s_sel[512]`, `s_ids[8]`, `s_w[8]` (plus `s_pred`, `s_ah`, `s_plan`, ... `[8]`) | shared-memory overflow; E = 896 overflows `s_p` too |
| glm_fast.cu:6611, 6617 | `moe_route()` turns off `rank_sel` and `fast` for k > 8 and runs the slow branch | the slow branch writes the same `[8]` arrays |
| glm_fast.cu:4896 | `moe_down_kernel`: 8 warps, warp w computes expert w | **experts 8-15 silently dropped** |
| glm_fast.cu:4638, 4278 | `moe_wait_fetch` `s_src[8]`; verify-window scratch claim over bits 0-7 | overflow |
| src/kernels/cuda/glm_batch.cu:2040-2041, 2145-2146, 2173-2174 | prompt-path `route_kernel` `s_p/s_sel[512]`, `s_ids[8]`; the combine kernels' `s_pos[8]`, `s_w[8]` | overflow |
| src/core/glm_fast_path.cu:1050-1058, 1181 | arena: `scratch`, `fetch_src`, `plan_ptr`, `plan_w`, `plan_id` 8 entries; 8 scratch slots | overflow |
| glm_fast_path.cu:4729-4730, 4835-4837 | `fast_service` `int ids[8]` filled for i < K; disk-miss `mi[8]`, `rd[8]`, ... | stack overflow |
| glm_fast_path.cu:5889-5901; src/core/glm_fast_state.hpp:858; src/program/generate.cpp:4531 | "8 = none" sentinels: `trunk_skip_from` clamps to [1, 8], `disk_skip_from = 8`, `GLM_DISK_SKIP_FROM` 0-8 | **silent quality loss**: ranks 8-15 treated as skippable |
| existing guards | `fast_helper_setup` errors (glm_fast_path.cu:2166-2170); `fast_cpu_lane_setup` silently turns the lane off (:3390); `moe_win_experts` (glm_fast.cu:7220) and HIP `moe_experts_multi` (:6982) return false; device experts off (glm_model.cu:3507); src/core/remote_experts.cpp:247 refuses `ne > 8`; EXL3 `moe_args` (src/kernels/cuda/exl3.cu:2897-2901) returns false and its `void` callers then skip the layer's experts | the right idea, in the wrong place: they fire after the route kernels already overflowed |

The adjacent caps fail the same way: **n_expert ≤ 512** (`moe_route_kernel` and `glmb::route_kernel`
shared arrays; `router_sigmoid` exits above 512, router_sigmoid.cu:85-88; the lookahead is silently off
above 512, glm_fast.cu:6645) and **n_embd ≤ 4096**, though not in the rings once a request carries `z`:
`MoeRequest::x[4096]` (the FFN input, glm_fast.hpp:343), `CpuAnswer::part[4096]` and
`MoeResponse::cpu_part[4096]` carry the experts' input and output rows, n_embd wide today (the fast path's
refusal names them, include/strata/core/glm_limits.hpp:35 on this branch). For K3.1 those rows are the
3584-wide latent (§2.7; ref/kimi_k31.py:946, 1006-1011 on this branch), so they would fit once the request
carries `z` rather than `x`. The 4096 limits 7168 does hit are the mHC kernels (`hc_kernel`'s
`s_mixed[4096]`, glm_fast.cu:2366; `hc3`, 4096 only, :2560; `head_prep_kernel`'s `sm[4096]`, :2708), which
Block Attention Residuals replace; `rms_q8`, one block for at most 4096 (:5291); and the guards that test
`n_embd` itself (glm_fast_path.cu:2166, :3390), which would have to test the latent. K3.1 is 16 of 896 at
7168, with 3584-wide expert rows. Raising the eights is not a constant change: the ring entry layout is
pinned by `static_assert`s (`offsetof(MoeRequest, x) % 8 == 0`, glm_fast.cu:4384-4385), the CPU-lane plan
nibbles cannot encode 16, the scratch-slot claim word uses bits 0-7 for slots and 8-15 for groups, and
`moe_down`'s warp-per-expert design needs a second pass or a different decomposition.

One path takes k = 16 already: GLM's per-op F32 path routes through `router_sigmoid`, which takes k ≤ 64
(router_sigmoid.cu:81-84; `kSlowMaxRouted`, include/strata/core/glm_limits.hpp:36 on this branch), still at
E ≤ 512 (router_sigmoid.cu:85-88). The DeepSeek engine does not: its route kernels hold `pick[16]` but write
the picks at stride `kPicks` = 8 (src/ds4/ds4_kernels.cu:846, 884-885; src/ds4/ds4_kernels.cuh:22), its MXFP4
kernels refuse k > 8 (§4.7), and its loader refuses more than 8 experts a token (src/ds4/ds4_model.cu:3506).

Beyond the counts, the MoE *shape* differs. The expert input is the 3584-wide latent `z = W_ld x`, not `x`;
the weighted sum is RMS-normalised and projected back up (`W_lu`) after the combine, and the shared experts
run on `x`. So the request carries the latent to the CPU lanes, the combine gains a norm and a GEMV, and the
router and the experts see different vectors. The norm needs all 16 experts' outputs (ref/kimi_k31.py:1014-1017,
975 on this branch), so the combine waits for the CPU lane's experts too. The activation is SiTU, not GLM's
clamped SwiGLU (`glm_ffn`): `4·tanh(g/4)·sigmoid(g) · 25·tanh(u/25)`, which needs precise `tanhf`/`expf`
for the same reason `glm_ffn` uses precise `expf` (GLM5-FLASH.md §6). One more numeric difference: Moonshot
renormalises with `+ 1e-20` in the denominator, and so does K3.1's oracle (ref/kimi_k31.py:348, 959 on this
branch); llama.cpp, GLM's oracle (ref/glm.py:531-532) and `router_sigmoid` (router_sigmoid.cu:71) clamp the
sum at 2^-14. They differ only if all 16 selected sigmoids are below ~3.8e-6.

### 4.7 MXFP4 (moderate; kernels exist elsewhere in the repo)

The release's experts are MXFP4, and the lossless conversion keeps them MXFP4 (ggml type 39, 32 values in
17 bytes = 4.25 bpw, the source's own size). The GLM device path cannot read it: `moe_supported()` lists no
type 39 (glm_fast.cu:5403-5406); the C++ `strata::block_geometry` has no type 39 (include/strata/artifact/
gguf_reader.hpp:118). The CPU lane can: its native experts take any type ggml-cpu has a dot product for
(src/kernels/cpu/native_expert.cpp:194-216), type 39 included. So can the reference: `ref/load.py` reads
MXFP4 and the low-bit i-quants of a K3.1 build, IQ1_S/M, IQ2_XXS/XS/S and IQ3_XXS/S, in `GEOM` and
`_GGUF_PY_NAME` beside IQ4_NL and IQ4_XS (`"MXFP4": (32, 17)`, ref/load.py:39-51). Its own numpy decoders
(`_DECODERS`, ref/load.py:81-82, 343-344) decode them, not gguf-py: ggml-quants.c's `dequantize_row_*` over
whole blocks, with the codebooks parsed from third_party/ggml/ggml-common.h (ref/load.py:113-181), held bit for
bit to gguf-py and to ggml's C `to_float` (tools/test_kimi_k31_load.py). Every other type gguf-py knows is
added from gguf-py when `gguf` is importable (ref/load.py:54-67). MXFP4 kernels already exist for the
DeepSeek engine: the device path in src/ds4/ds4_gguf.cu, which refuses k > 8 (`mx_geometry`,
ds4_gguf.cu:1360-1361) behind scratch sized for 8 (`part[8][kMaxEmbd]`, :1332) and is built only into maya-ds4
and its test (CMakeLists.txt:667-668, 760; MiniMax's engine takes ds4_iq.cu, ds4_lt.cu, ds4_iq_cpu.cpp and
ds4_gguf_cpu.cpp, not it, :817-818); the x86 CPU path in src/ds4/ds4_mx_kern.inc, which ds4_gguf_cpu.cpp
includes (src/ds4/ds4_gguf_cpu.cpp:126-141). Re-quantizing to i-quants is the other route (llama-quantize
needs `--allow-requantize` for an MXFP4 source, and it is then a second quantization: no BF16 expert source
exists).

### 4.8 Reading the file: the architecture check and the 64-item arrays (small, and first)

Two reader facts decide what happens to a K3.1 file *today*:

1. **`from_gguf` never reads `general.architecture`.** It picks its key prefix by probing
   `glm5next.block_count` and otherwise reads `glm5-next.` keys (glm_model.cu:508-510); every absent key
   silently takes the GLM-5.3 default (`num()`, :511-521). A `kimi-k31` file has no `glm5-next.` keys, so
   it is read as a **default GLM-5.3** - 8 of 288 experts at 4096 - and any refusal keyed on the geometry
   (k > 8, E > 512, n_embd > 4096) never fires. The architecture check has to land first, or in the same
   change. The accepted strings elsewhere are exactly `"glm5-next"` and `"glm5next"` (generate.cpp:5243,
   src/metal/glm_metal_load.cpp:152, maya.py:717, tools/iq_pack.py:99); the closest error-style precedent
   is src/core/glm_dflash.cu:782-787.
2. **The C++ GGUF reader keeps only the first 64 items of any metadata array** (`read_value(..., keep =
   64)`, gguf_reader.hpp:272; `parse()` uses the default). `from_gguf`'s `arr_i` accepts a per-layer array
   only when `items.size() >= n_layers` (glm_model.cu:522-531), so a 93-entry `attention.head_count_kv`
   is silently ignored and **every layer reads as KDA**. The fix that does not touch other models is
   `GgufFile::get_full(key, out)` (gguf_reader.hpp:364-383, already used at glm_mtp.cu:159-160) for
   per-layer arrays; until then, a per-layer array whose `count` exceeds what was kept must be refused, not
   skipped.

### 4.9 Tokenizer, template, MTP, vision

* **Tokenizer.** Kimi's tiktoken-style vocabulary (163,840, shared with Kimi K2) converts to llama.cpp's
  `kimi-k2` pre-tokenizer, which llama.cpp handles without BPE merges. `strata_tokenizer` refuses schemes it
  has not transcribed (tools/strata_tokenizer.py:144-145: qwen35, glm4, glm5, deepseek3, minimax); `kimi-k2`
  would be a new transcription, checked id for id as GLM5-FLASH.md §8.3 did. The GGUF's eos must be
  `<|end_of_msg|>` (163586), the turn terminator, not `[EOS]`; upstream's converter restores it.
* **Template.** Moonshot renders chats in Python (`encoding_k3.py`); K3.1's only semantic change to it is a
  `"medium"` thinking effort. llama.cpp's bundled `Kimi-K3.jinja` accepts only low/high/max.
* **MTP.** None in the release. Separately published draft models target specific serving engines and are
  out of scope.
* **Vision.** MoonViT-V2 is out of scope; the text model loads without it.

### 4.10 Hardware envelope

At full size the routed experts are 532 GB even at IQ1_S. The non-routed part is ~34 GB at Q4_K (§3.3), but
Q4_K is below Maya's quant floors (docs/QUANT-PLAN.md:113-117); at the floors - KDA, MLA, the shared experts,
the head and the embeddings at Q6_K, the dense layer 0 at Q8_0, with the latent projections at Q6_K too and the
router left in BF16 - it is ~47.8 GB (an estimate: §3.1's parameter counts × bits per weight), so a whole-model
install needs ~580 GB of RAM plus VRAM, or expert paging from SSD at 1,472 expert reads a token. The GLM
engine's tiering (VRAM cache, RAM, mmap with per-layer prefetch) is the right machinery, but each token asks it
for 4.4× as many experts as GLM-5.3 does.

The CPU, not DRAM, sets the speed of experts held in RAM. §3.3's `bandwidth / 9.5 GB` ("about 10 tok/s")
assumes the cores keep up, and they do not: the CPU lane computes IQ1_S with ggml-cpu's one-token dot (§6,
the CPU lane row), which is bound by instruction issue, not by memory (src/ds4/ds4_iq_cpu.cpp:411-412). The
128-expert llama.cpp build of this contract (below), on that dot, decodes at a measured 3.0 tok/s on a 16-core
machine, with an estimated 77 % of each token's 9.5 GB of experts on the CPU (the share not held in VRAM), so
its CPU runs IQ1_S at ~22 GB/s (an estimate: 3.0 × 0.77 × 9.5), against ~110 GB/s measured from that
machine's four DDR5-4800 channels (153.6 GB/s peak) - about 2.3 tok/s if every expert ran there (an estimate:
22 / 9.5). More cores, not more memory channels, would raise it.

A pruned build (§3.3) is not the realistic target for consumer machines that the first version of this section
called it. Measured since: a 128-expert (14 %) IQ1_S build loops and passes 1 of 46 agent tasks; the same prune,
unquantized and teacher-forced on held-out plain text, picks the full model's top token 72 % of the time and
raises perplexity from 2.33 to 5.33. That is consistent with the prune, not the port, doing most of the damage,
but does not separate it yet: the build's expert ranking and importance matrix came from a forward pass that kept
the residual block sums in bf16 and has since been found to lose context. A typical Maya PC - one 24 GB card and
64 GB of RAM or more, what Maya's GLM quants are sized for (73.4-156.3 GB, README.md:56-61; Maya-XS for 64 GB and
a 24 GB card, :58) - cannot hold K3.1 at any good quality. Quality between 14 % and 50 % of experts kept is
still open (§5.6).

Apple Silicon: Maya's Metal engines are GLM-5.3-Flash's (`maya-metal`, CHANGELOG.md:326-329;
src/metal/glm_metal.cpp:695) and MiniMax's (`maya-metal-mm3`, CHANGELOG.md:19). K3.1 would need one of its own,
which is not part of Phase 3.

## 5. Open questions

### 5.1 Open question 1 — resolved: `A_log` has 128 entries for 96 heads

**Answer: `A_log` is per head; use the first `linear_attn_config.num_heads` (96) entries, and assert the
rest is zero at conversion.** The checkpoint stores `layers.N.self_attn.A_log` as F32 **[128]** on every
KDA layer, in K3 and K3.1 alike; `b_proj` [96, 7168] pins the head count at 96, and 128 is the head dim.

* **Moonshot's module** allocates one value per head (ml.py:520-521):
  `self.A_log = torch.nn.Parameter(torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)))`.
  Loading the checkpoint into it fails on the shape unless the loader slices.
* **fla**, which Moonshot calls, reads it per head: `b_A = tl.load(A_log + i_h)` with `i_h < H`
  (`ops/kda/gate.py:133, 199`; `fused_recurrent.py:163`), and H comes from `g`/`v`, never from
  `A_log.shape`. A 128-entry tensor runs and only entries 0..95 are read. fla's naive gate does
  `A_log.view(H, 1)` and would raise on 128 - so a reference must slice explicitly.
* **vLLM** (`vllm/models/kimi_k3/nvidia/kda.py:86-108`) narrows from index 0:
  ```python
  shard_size = param.data.shape[shard_axis]
  start_idx = tp_rank * shard_size
  loaded_weight = loaded_weight.narrow(shard_axis, start_idx, shard_size)
  ```
  with `self.A_log = nn.Parameter(torch.empty(self.local_num_heads, dtype=torch.float32))` - the union over
  ranks is `[0, 96)`.
* **SGLang** (`python/sglang/srt/models/kimi_k3.py:1767-1790`): its loader comment reads "Checkpoint stores
  A_log as [head_dim]; the FLA kernel expects local_num_heads", and the loader narrows to the first
  `num_heads` before tensor-parallel sharding.
* **llama.cpp** (`conversion/kimi_k3.py:288-291`):
  ```python
  if name.endswith(".A_log"):
      n_head = self.hparams["num_attention_heads"]
      data_torch = -torch.exp(data_torch.float()[:n_head])
  ```
  This slices by the *MLA* head count; it is right only because both counts are 96. A K3.1 converter should
  slice by `linear_attn_config["num_heads"]`.
* **The tail.** A community pull request on the Kimi-K3 repository (Hugging Face discussion #150, a
  third-party report, not verified here) reports entries 96..127 are exactly 0.0 in all 69 KDA layers of K3.
  **For K3.1 it has now been read** (§7.4): in all 69 KDA layers, entries 96..127 are bitwise +0.0 and
  entries 0..95 are finite. Converters for this port still **assert it is zero** and fail
  otherwise (layer 0's whole `A_log` is the 512 bytes at offsets 2928-3439 of
  `model-00001-of-000096.safetensors`). A different discussion (#197) proposes a per-channel decay
  `A_log[d]`; that contradicts every kernel above and is not implemented.

### 5.2 Open question 2 — pinned, not resolved: the norm epsilons

| Norm | Moonshot code | vLLM / SGLang | llama.cpp `kimi-k3` | Proposed for this port |
| --- | --- | --- | --- | --- |
| `q_a_layernorm`, `kv_a_layernorm` | **1e-6** (`KimiRMSNorm` built without eps, ml.py:368, 383; default at :227) | 1e-5 (`rms_norm_eps`) | 1e-5 (`f_norm_rms_eps`) | a parameter; `ref/kimi_k31.py` defaults to Moonshot's 1e-6 (preset `moonshot`), preset `llama.cpp` takes the GGUF's 1e-5; the gate resolves the two (§7.2) |
| KDA q/k l2 norm | 1e-6 (fla: `x / sqrt(Σx² + 1e-6)`) | vLLM: in the kernel (`use_qk_l2norm_in_kernel`), eps not read here; SGLang not checked | 1e-5 (`build_gdn_l2_norm` with `f_norm_rms_eps`) | 1e-6 (the kernels Moonshot calls); the gate resolves it, narrowly (§7.2) |
| indexer `k_norm` LayerNorm | - (no K3.1 code) | 1e-5 (`rms_norm_eps`) | - (DeepSeek-V3.2 graph hard-codes 1e-6) | 1e-5, written as `attention.layer_norm_epsilon` |
| every other RMSNorm (layer norms, KDA `o_norm`, routed-expert norm, final, attn-res scoring) | 1e-5 | 1e-5 | 1e-5 | 1e-5 |

The latent-norm eps feeds `qr`, which feeds the indexer's queries, so the two choices produce slightly
different selections at near-ties; elsewhere the difference is far below f32 noise on activations of order
one. Which value matches training is not published. The indexer's 1e-5 is engine consensus, not the model
owner's specification. Any exact-match test states which values both sides used. On the synthetic fixture
the choice is precision-level: with the float64 reference the wrong q_a/kv_a eps moves the logits by
2.1e-4 at most and the wrong KDA eps by 1.5e-4 (|logit| ≤ 4), both under the gate's 3e-4, so only a seam
taken soon after the norm tells them apart (§7.2).

### 5.3 Open question 3 — pinned: indexer ties and the masking trap

* **Ties are real.** A key whose `iq[t,h] · ik[s] ≤ 0` for every head scores exactly 0, and so does every
  other such key; with fewer than 2048 positive scores in a row, the cut runs through a block of exact
  zeros. No engine specifies a tie rule (`torch.topk`, `ggml_top_k` and the CUDA kernels each pick
  differently; vLLM's own test accepts recall > 0.97 against an fp32 reference). The oracle's convention -
  descending score, ties toward the lower key index, like `top_k_desc` (ref/glm.py:192-196) - is a
  convention, not the model. **Parity compares selections as sets with a tie tolerance** (a key may
  differ only if its score is within tolerance of the row's k-th score).
* **Outputs cannot be excused row by row.** This section first proposed comparing attention outputs only
  on rows whose cut gap exceeds the tolerance. That does not work: a different tied key changes that
  row's attention output, and through the residual stream every later layer and position. On a first
  version of the fixture (4 indexer heads, 24 exact-zero tie rows at the cut), breaking the ties toward
  the later key instead left the set check satisfied but moved the logits by 2.6. So an output gate has
  to be **built tie-free**: the fixture's test sequences are chosen so that no cut lies near a tie (every
  golden cut margin at least 10× the score tolerance), the gate fails if the golden has such a row, and
  exact ties are exercised separately, on inputs built to tie, where only the sets are judged (§7.2).
* **Future keys are `-inf`, never 0.** Because the head weights can be negative, a valid score can be
  negative. SGLang's pure-torch logits fallback fills invalid cells with 0.0 (`layers/attention/dsa/utils.py`,
  `_fp8_mqa_logits_torch`) and is correct only because its top-k is length-limited per row; a port that
  copies the 0-fill and takes top-k over the whole row selects future keys. A negative test pins it.

### 5.4 Open question 4 — resolved: `index_topk_freq` 1 and `index_skip_topk_offset` 1 do nothing

The offset belongs to cross-layer index sharing. SGLang's rule (`configs/model_config.py:543-575`) is
`skip = max(layer_id - offset + 1, 0) % freq != 0`, always false at `freq = 1`; vLLM's PR refuses any other
frequency ("Kimi-K3.1 has no index-share mechanism: every DSA layer builds its own indexer"). All 24 DSA
layers carry their own indexer weights. A port refuses `freq != 1` rather than implement sharing.

### 5.5 Open question 5 — resolved: the 64 "rope" dims are un-rotated key data and cannot be folded

`kr` is the last 64 rows of `kv_a_proj_with_mqa` applied to `x`, outside `kv_a_layernorm`, shared by all
heads and never rotated (§2.4). It cannot fold into the normalised 512-wide latent (the norm is nonlinear),
and folding it into the query side would need `x` cached. So the cache row is 576 wide, exactly as
upstream's MQA view (`key_length` 576, `value_length` 512) has it, and Maya's 512-only refusal (§4.4) is a
kernel gap, not a conversion option.

### 5.6 Remaining open questions

1. **K3.1's `A_log` tail** (§5.1): read from the release (§7.4) and zero; still asserted at conversion.
2. **Training-time numerics**: which eps (§5.2), and whether training used the engines' FP8 indexer
   q/k and bf16 head-weight rounding (SGLang rounds twice, vLLM once). The card says K3.1 was trained with
   MXFP8 activations; the reference and the engines run bf16 activations. All of this moves near-ties only.
3. **Pins on moving targets**: both indexer sources are unmerged (§1.1).
4. **Moonshot's code on transformers 4.57.x** (the version K3.1's config was saved with) is untested; on
   5.22 it needs runtime shims (§6, Phase 1).
5. **Runner shape for Phase 3**: widen the GLM fast path to k = 16 / E = 896 / 7168 (touches every model
   on it) or build a separate K3.1 runner that reuses the kernels that fit (KDA, router, score kernel).
6. **Metadata arrays**: `get_full` for per-layer arrays (recommended) or a larger global sample.
7. **Pruned quality**: no published data below ~27 % of experts kept, and none for K3.1 at all.

## 6. Execution plan

**Phase 0 - this document.** The study; no engine code.

**Phase 1 - the oracle and its gate.** Following GLM5-FLASH.md §6 Phase 1 and `ref/glm.py`'s conventions
(features-first ggml layouts, f64 accumulation inside norms, softmaxes and the recurrence, F32 at seams,
`(out, in)` weights from `ref/load.py`, metadata defaults = the released config so a fixture writes only
what it changes, `ValueError` on contracts it does not implement):

* `ref/kimi_k31.py`: GGUF → one teacher-forced forward pass of §2, reading `kimi-k31.*` keys; every seam
  recorded under upstream `kimi-k3.cpp`'s `cb()` names (`attn_norm`, `kda_q_conv` ... `kda_out`,
  `mla_gated`, `mla_out`, `prefix_sum_attn`, `res_probs`, `ffn_moe_logits`, `ffn_moe_out`, `ffn_shexp`,
  `ffn_out`, `l_out`, `result_norm`, `result_output`) plus `deepseek32.cpp`'s indexer names
  (`indexer_q`, `indexer_k`, `indexer_weights`, `indexer_score`); refuses `index_rope_dim != 0`,
  `index_topk_freq != 1` and a non-LayerNorm key norm; the MLA latent-norm eps is a parameter. (As built,
  the GGUF of §3.2 carries no rope-dim or key-norm-type key, so those two are refused where the HF config
  is read - by the independent model's indexer patch, and by a converter; the reference refuses a
  shared-selection indexer (`attention.indexer.types`), a pooled one (`kpool`) and a missing tensor.)
* `tools/kimi_k31_synth_gguf.py`: a tiny random-weight model in the `tools/glm_synth_gguf.py` pattern (own
  F32 GGUF writer, dict geometry, seed, self-test re-read), written to a directory given on the command line:
  the F32 GGUF; the same weights as an HF checkpoint under K3.1's tensor names; a second HF copy with the
  experts in compressed-tensors MXFP4 (expert weights drawn on the MXFP4 grid so both copies hold identical
  values); and a manifest of seed, config and file hashes. It keeps every structural flag of §2: attention
  residuals with at least two block pushes, a dense layer 0, latent MoE with its norm, SiTU, the full-rank
  KDA gate at bound -5, NoPE MLA with an un-rotated tail and output gate, the last two layers both MLA,
  `A_log` padded to the head dim with a zero tail, and an `index_topk` well below the sequence length so
  selection discards keys. Two kernel constraints shape it: fla's Triton tiles want KDA head dims of at
  least 16, and sequences longer than fla's chunk of 64 exercise its inter-chunk path.
* Invariant tests with numpy only, so CI's tools step runs them (CI installs no torch and reaches `gguf`
  only through `tools/_paths.add_gguf_py()`). As built they are `tools/test_kimi_k31_synth.py` (the
  fixture's self-test, determinism, bf16/MXFP4 bit-identity worked by hand, the `ref/load.py` reader) and
  `tools/test_kimi_k31_ref.py`: config defaults and refusals; prefill = chunked prefill = token-by-token
  decode; rows with `t + 1 ≤ index_topk` select every visible key and later rows exactly `index_topk`; no
  future key is ever selected; with `index_topk ≥ T` the MLA is bit-identical to dense causal attention;
  absorbed MLA equals materialised MLA; and independent tokens-first transcriptions of the indexer formula,
  the KDA recurrence, the residual mix and both FFNs. The repo's negative-test discipline - each wrong
  reading must differ materially - moved into the gate's tests (`tools/test_kimi_k31_oracle.py`), where a
  wrong reading must fail the gate itself, not just differ (§7.2).
* **The gate against an independent model** (GPU, by hand, not in CI): Moonshot's own
  `modeling_kimi_linear.py` at `f831ab6`, run in fp32 with the **real fla 0.6.0 kernels**, loads the HF copy
  of the fixture. As built, "fp32" has to mean full-fp32 dots: fla as published runs its chunked prefill
  with TF32 dots on sm_80 and newer, which on the fixture moves the logits by 1.1 and flips 29 indexer and
  45 expert rows, so the golden is made with `TRITON_F32_DEFAULT=ieee` plus fla's one explicit TF32
  constant overridden (bf16 flipped routing on the fixture's first version, so there is no bf16 golden).
  The indexer, which Moonshot does not ship, is written separately from the oracle from the two engine
  sources of §2.5 and enters Moonshot's MLA as an additive `-inf` mask on top of its causal mask. Running it
  on transformers 5.22 takes four runtime shims (the moved `OutputRecorder`, the renamed
  `create_causal_mask` arguments, two cache methods) and switching the forced `flash_attention_2` to
  `sdpa` or `eager` after construction, with each model built from a fresh config (the setting is shared);
  `import fla` raises a Triton `RuntimeError` without a GPU driver, which Moonshot's `except ImportError` does
  not catch; and the module's `A_log` [96] must be loaded from the checkpoint's [128] by slicing. Its golden
  outputs (logits and the matching seams, plus fla's chunked prefill against its token-by-token recurrence as
  a self-check) are what the oracle is held to. The two sides share one written spec, so a misreading common
  to both would pass; the defences are the pinned sources, the negative tests and, for the indexer half
  (the one part not run from Moonshot's code), a replay of the golden's recorded indexer inputs through
  vLLM's and SGLang's own score functions (§7.2).

**Phase 2 - refusals in the GLM engine.** Make it refuse what it cannot run, loudly and specifically, before
it allocates (the house style: `bool f(..., std::string& err)`, `err = "<prefix>: ..."`, `return false`):

1. `Glm5Geometry::from_gguf`: read `general.architecture` first and accept only `"glm5-next"` / `"glm5next"`,
   naming the architecture it found (a `kimi-k31` file then stops here, before any default applies).
2. `Glm5Model::fast_setup`, right after `const Glm5Geometry& g = g_;` (glm_fast_path.cu:779): refuse
   `n_exp_used` outside 1..8, `n_expert > 512` and `n_embd > 4096`, naming the structures that hold those
   limits. The same check right after `from_gguf` in `load_pack` (glm_model.cu:2841) refuses before the dense
   upload: the dense weights are uploaded (and `new FastState()` runs, glm_fast_path.cu:766) before :779 is
   reached.
3. A per-layer metadata array that was truncated (`count` beyond the kept items) is refused instead of
   silently ignored, until per-layer arrays are read with `get_full`.
4. The qwen4exp driver gate (generate.cpp:5239-5253) and the Python-side architecture lists (maya.py, iq_pack)
   name `kimi-k31` in their refusals, as GLM5-FLASH.md's Phase 0 did for `glm5-next`.

Nothing in Phase 2 changes a GLM-5.3 file's behaviour; its parity suites and `tools/test_glm_oracle.py` must
stay green.

As built (§7.3): items 1 and 2 landed, with more than planned - the kernel launchers refuse too, as
counted failed launches, and the refusal does not come before *every* allocation: the first shard is
mapped to read its header, then the gate runs before the device is set, any other shard is mapped or
any row is uploaded. **Items 3 and 4 did not land.** Item 3 is not reachable by a K3.1 file now that
item 1 refuses it first, and GLM-5.3's per-layer arrays (one entry per block, 45 layers plus MTP) fit in
the 64 kept items; item 4 is open.

**Phase 3 - kernels and a runner (not on this branch).** The honest size of it, by kernel family, in the
GLM study's terms:

| Piece | Size | Why |
| --- | --- | --- |
| MoE at k = 16, E = 896, n_embd 7168 | **large** | §4.6: request/response ring, plans, scratch slots, verify windows, the "8 = none" sentinels, `moe_down`'s decomposition (the prompt path's route and combine and the CPU lane are the next two rows); or a new runner that avoids them |
| Prompt path (batched prefill) | **large** | batched Block Attention Residuals over the `T × 9 × 7168` fp32 stack (§4.2), the 576-wide MLA with its gate (§4.4), the kpool-1 selection at 2048 (§4.5), route and combine at 16 of 896 with the latent norm (glm_batch.cu:2040-2041, 2145-2146, 2173-2174; §4.6) |
| CPU lane at k = 16, on the latent | moderate | carries `z` (3584), not `x` (§4.1), SiTU, and a plan word for k = 16 (glm_fast.hpp:405-406, §4.6); IQ1_S rows of its own: it computes IQ1_S with ggml-cpu's one-token dot, since none of its own rows takes the type (src/kernels/cpu/iq_avx2.cpp:803-807, src/kernels/cpu/iq_avx512_one.cpp:152-154; multi-token, native_expert.hpp:49-51), while DeepSeek's CPU rows do (src/ds4/ds4_iq_cpu.cpp:408); the routed norm waits for its experts (§4.6), and decode runs at its speed (§4.10) |
| Per-token indexer + top-2048 selection | moderate | GLM's indexer at kpool = 1 (§4.5, glm_dsa.cu:34-106); new are the 2048-pick selection over up to 1 M candidates where GLM's fast select holds 1024 (dsa_topk.cuh:23-26; DeepSeek's segmented `topk` takes any k, ds4_kernels.cuh:200-215), the per-token key cache, prefill and decode variants, parity defined by ties (§5.3) |
| MLA with the 576-wide row and output gate | moderate | extends `glm_dsa`'s absorbed attention (§4.4) |
| MXFP4 experts in the GLM path | moderate | the device path: the DeepSeek engine's kernels refuse k > 8 (ds4_gguf.cu:1360-1361), so they come over widened, into `moe_gate_up` / `moe_down`; `block_geometry`; the CPU lane already takes type 39 (native_expert.cpp:194-216), and `ref/load.py` reads it and the low-bit i-quants (ref/load.py:39-51, §4.7) |
| Block Attention Residuals | small kernel, moderate runner | §4.2 |
| Latent MoE tails (`W_ld`, post-sum norm, `W_lu`) | small | GEMVs and a norm around the combine, and the shared experts' output moved out of `moe_down`'s sum (glm_fast.cu:4926, §4.1) |
| Router at E = 896, k = 16 | small | `router_sigmoid`'s 512-thread block (router_sigmoid.cu:85-88; k = 16 is within its 64, :81-84) and its 2^-14 sum floor (:71, §4.6); otherwise the same contract |
| SiTU FFN | small | replaces the clamped SwiGLU in `glm_ffn`'s three uses |
| KDA at 96 heads, full-rank gate | small | §4.3 |

Each family lands as GLM's did: a header + CUDA source + `*_parity --selftest` harness with negative tests
and thresholds set from measured bounds, then a layer-composition test against the oracle's raw dumps, then
the runner held to the oracle at every position. New CUDA calls need their AMD mapping
(`python tools/check_hip_compat.py`); new settings are `STRATA_GLM_*` variables, off by default.

**Phase 4 - pack, serve, measure.** `kimi-k31` in `tools/iq_pack.py` with a BF16/F32-preserve list (router
and its bias, the indexer, the residual scores, `ssm_a`, `ssm_dt`, the conv weights, norms); the `kimi-k2`
tokenizer transcription; the chat template; a cross-check against a llama.cpp build of the same GGUF
contract (§3.2); and only then a `FAMILIES` entry, which would otherwise advertise a model the driver refuses.

**Acceptance gates.**

* *Phase 1, oracle invariants* (`tools/test_kimi_k31_synth.py`, `tools/test_kimi_k31_ref.py`, CPU, CI):
  all invariants above pass.
* *Phase 1, oracle vs independent model* (fp32, GPU): per-seam and logit agreement with thresholds set from
  the measured gap rather than guessed; indexer selections equal as sets up to the tie tolerance of §5.3;
  both sides' epsilons recorded with the result. As built, the logit check is a max and a pooled rms bound
  on the full vocabulary rather than the GLM study's KL p99 / top-1 gates: on a tie-free golden (§5.3)
  every expert and indexer selection must match exactly, so there is no disagreement left to trace to a
  selection boundary; the logits bound (3e-4 float64, on logits up to |4|) sits four orders of magnitude
  under the smallest structural mutation's effect (§7.2).
* *Phase 1, MXFP4*: the MXFP4 copy of the fixture decodes to exactly the F32 copy's values.
* *Phase 2*: each refusal returns its specific message before the device is set or any row is uploaded;
  GLM-5.3 behaviour unchanged.
* *Phases 3-4* (future): the GLM gates - kernel parity with negative tests, layer composition, runner vs
  oracle (streaming vs batched, state determinism, mutation sensitivity), the F32 llama.cpp cross-check
  (GLM5-FLASH.md §8.2) and, on quantized files, the quantized-mode gates (§8.2.1).

Rough effort weighting: Phase 1 ≈ 10 %, Phase 2 ≈ 5 %, Phase 3 ≈ 65 % (half of it the k = 16 MoE, counting
its CPU lane and the prompt path's route and combine), Phase 4 ≈ 20 %. The critical path runs through the MoE
fast path, its CPU lane and the prompt path; the indexer is GLM's at kpool = 1 (§4.5, glm_dsa.cu:34-106) and
moderate. This branch stops at Phase 2.

## 7. What this branch contains

Based on v1.1.53 (`7c4aa56`). Nothing on it runs K3.1. For GLM-5.3 every limit it names is still 8, the GLM
parity self-tests print what v1.1.53's print, and the other GLM tests pass except one that cannot pass on
v1.1.53 either (§7.3); no real GLM-5.3 pack was run (§7.6).

### 7.1 Files

* `docs/KIMI-K31.md` - this study (Phase 0).
* **Phase 1** (Python; nothing new for CI to install - the numpy-only parts run in CI's tools step, the rest
  skips without torch):
  * `tools/kimi_k31_synth_gguf.py` + `tools/test_kimi_k31_synth.py` - the synthetic fixture: one seeded set
    of weights (seed 3101) written as an all-F32 `kimi-k31` GGUF (§3.2's tensors and keys), as an HF
    checkpoint under the release's names, as a second HF copy with the routed experts in compressed-tensors
    MXFP4 (bit-identical after dequantisation), and a manifest with the test sequences. Geometry (released
    values in brackets): vocab 320, hidden 256, 9 layers - 0-2 KDA, 3 MLA, 4-6 KDA, 7-8 MLA - with
    attention-residual block 3 [12], so blocks push at 0, 3 and 6; KDA 6 heads × 32 [96 × 128] with `A_log`
    stored at 32 entries, 6..31 zero; MLA 6 heads, q_lora 160, the 576-wide row (512 + 64 un-rotated), nope
    128, v 128, output gate; indexer 12 heads × 32, top-6 [2048]; dense layer 0; MoE 24 experts, top-10 (more
    than 8 on purpose), 2 shared, expert width 64 inside a 128-wide latent. Sizes a mix-up could swap are
    distinct (KDA head_dim 32 ≠ qk_rope 64, q_lora 160 ≠ the 192-wide q head, indexer q width 384 ≠ hidden
    256); KDA heads equal MLA heads, as released, because the format has one `attention.head_count`.
    `--geometry big` writes a second fixture (`big.gguf` and the two HF copies, ~2.9 GB, built on demand,
    nothing committed) for what the tiny one cannot reach: hidden 1024, latent 512, expert width 256, q_lora
    768, dense width 1792, vocab 1024, so every row an i-quant reads is a multiple of 256 (except `attn_k_b`,
    `ssm_f_b` and the conv taps, which are not in the release either); 64 experts, top-16, 2 shared; 10
    layers, MLA at 3, 8 and 9, attention-residual block 3 (pushes at 0, 3, 6 and 9); KDA and MLA 8 × 128;
    the indexer at its released shape, 32 × 128, with top-32; sequences of 96, 32 (= top-k: dense) and 300
    tokens (four of fla's chunk boundaries). It keeps the tiny fixture's layout rules (the last two layers
    MLA, as released; at least 8 indexer heads, §5.3). Its numbers are in §7.2.
  * `ref/kimi_k31.py` + `tools/test_kimi_k31_ref.py` - the reference forward pass of §2 in `ref/glm.py`'s
    conventions (§6 Phase 1), with `cb()` seams under the llama.cpp graph's names, float32 or float64, eps
    presets `moonshot` (default) and `llama.cpp` (§5.2), and refusals for what it does not implement.
    For real weights (Phase 3): `--layers N` (the first N layers, then the head as an N-layer file has it, or
    `--no-head`), lazy reads (only routed experts, each once per chunk of tokens; the embedding by rows; the
    head in slabs), the indexer scored in bounded query blocks, seams streamed to `--dump-dir` as they are
    made, `--act-q8 32` (iq::mv's int8 rounding of the inputs of every product with an iq::mv-type weight;
    not the CPU lane's Q8_K), `--tokens-file`, `--seams` and `--dry-run` (the read map of a run, a small-slice
    probe of each tensor, and the forward's memory estimate for the prompt); a truncated run adds the seam
    `res_blocks-(N-1)`, the block stack at the cut. `tools/test_kimi_k31_ref_layers.py` tests them, and
    `tools/kimi_k31_ref_pin.py` pins the oracle's bits on the fixture (`data/kimi-k31-ref-pin.json`).
  * `tools/kimi_k31_moonshot_oracle.py`, `tools/kimi_k31_moonshot_indexer.py`, `tools/kimi_k31_fla_standin.py`
    + `tools/test_kimi_k31_moonshot_oracle.py` - the independent model: Moonshot's K3 text model imported
    unchanged from a directory given on the command line (HF `moonshotai/Kimi-K3` @ `f831ab6`; the files'
    sha256 is checked and recorded; nothing is vendored), the lightning indexer as a patch on its MLA, and
    either the real fla 0.6.0 kernels (CUDA) or a pure-torch stand-in for the fla names Moonshot imports. It
    loads the HF copy strictly, with `A_log` sliced to `linear_attn_config.num_heads` and its tail asserted
    zero.
  * `tools/kimi_k31_gate.py`, `data/kimi-k31-synth-golden/` (`golden.npz` 1.2 MB, `golden.json`) +
    `tools/test_kimi_k31_oracle.py` - the gate: the independent model's outputs on the fixture, committed as
    the subset the comparison needs, so the gate itself needs numpy only. `.gitignore` excepts the one
    `.npz`.
* **Phase 3** (Python, numpy only; the engine's yardstick, §7.7):
  * `tools/kimi_k31_gate.py check-dumps` (alias `check-engine`) and `dump-ref` +
    `tools/test_kimi_k31_check_dumps.py` - an engine's seam dumps, under the reference's `cb()` names,
    judged rung by rung on the seam ladder against the golden and the float64 reference; `dump-ref` writes
    the reference's own dumps in the same layout.
* **Phase 2** (C++/CUDA; §6 items 1 and 2):
  * `include/strata/core/glm_limits.hpp` (host only): the limits (8 routed experts a token, 512 experts,
    `n_embd` 4096 on the fast path; 64 of 512 on the per-op path), the architecture check and the load
    gate `check_pack_limits`. `include/strata/core/glm_geometry.hpp`: `Glm5Geometry` and `from_gguf`, moved
    unchanged out of `glm_model.{hpp,cu}` so a host-only test can run the real reader.
  * `from_gguf` refuses any `general.architecture` but `glm5-next` / `glm5next` (and a missing or
    non-string one) before it reads a key, and a route of fewer than 1 or more than `expert_count`
    experts. `load_pack` runs the gate after reading the first shard's header and before `cudaSetDevice`,
    any other shard mapping or upload; `load` (the per-op runner) checks the per-op limits; `fast_setup`
    checks again (unreachable behind `load_pack`). A fast-path refusal says when `MAYA_GLM_SLOW=1` would
    take the route. A K3.1 file stops at the architecture check:
    `glm5_model: general.architecture is 'kimi-k31' - this runner reads GLM-5.3-Flash (glm5-next) GGUFs only: ...`.
  * The kernel launchers that hold a route - `glmf::moe_route`, `moe_fetch`, `moe_wait_fetch`,
    `moe_gate_up`, `moe_down`, `moe_win_group`; `glmb::route`, `moe_combine`, `moe_combine_add`; EXL3's
    `moe_gate_up` / `moe_down` in `exl3.cu` and `exl3_legacy.cu` - refuse what they cannot hold as a
    counted failed launch, so the forward reports an error instead of running past an array or silently
    dropping experts (§4.6). The two route launchers return `false` and their five callers stop the layer,
    so nothing reads an unwritten plan. The other void GLM launchers that refuse a geometry (`kda_rec`,
    `kda_layers`, `kda_gather`, `hc_finish`, `kda_conv`, the `mla_attn` variants) count it too.
    `static_assert`s prove the load gate's limits are within both kernel families' limits.
  * Separate names for what were literal 8s: `kRoutePicks` / `kRouteExperts` (what the kernels hold),
    `kPlanMaxF` (the CPU lane's plan word, all five writers), `kRankNone` ("leave no rank out" in
    `STRATA_GLM_TRUNK_KEEP`, `_MTP_KEEP`, `_DISK_SKIP_FROM`). All are still 8: no setting or answer changes.
  * Tests: `tests/glm_limits_test.cpp` (host only, a new CI step, through the real `from_gguf` and gate);
    the same file built as `glm_limits_engine_test` in GPU builds (through `Glm5Model::load` and
    `load_pack_env`); `tests/glm_moe_rows_test.cu` checks that each launcher refusal adds exactly one failed
    launch.

### 7.2 Phase 1: what was run, and the numbers

CPU work ran on x86-64 Linux (AMD EPYC 9124, Python 3.12, numpy 2.5); the golden and the live tests on one
NVIDIA RTX 3090 (sm_86) with torch 2.7.1+cu126, fla-core 0.6.0, triton 3.3.1 and transformers 5.22.1; the
fixture's first version also ran on an RTX 3060.

**Indexer source.** SGLang `kimi-k3.1-support` @ `accf045c` and vLLM PR #65685 @ `9d6ac73f`, both fetched
by commit id. The independent model's indexer follows vLLM's unfused path without its FP8 q/k
quantisation. `KimiK31IndexerEngines` then replays the golden's recorded indexer inputs (each MLA layer's
attention input and `q_a_layernorm` output) through vLLM's `_logits_ref` (copied with attribution) and
SGLang's `_fp8_mqa_logits_torch` semantics (with `-inf` masking and without its FP8-only bf16 cast): the two
agree with each other to 4.4e-16, with the reference to 8.7e-13 and with the golden to 7.6e-7, and give every
golden selection. Five misreadings of the formula each move the scores by at least 0.10 (ReLU after the head
sum 2.3, unscaled weights 45, `k_norm` without bias 0.10, no LayerNorm on k 0.68, q from the un-normed latent
0.69).

**The golden.** Real fla, fp32 with ieee dots (§6), eager attention, 315 MiB of GPU memory. Three sequences -
32 tokens (the indexer discards keys from position 6), 5 tokens (every row dense) and 80 tokens (across fla's
64-token chunk) - each as one prefill and as a prefill followed by one token a step through Moonshot's
cache. Recorded: full-vocabulary logits, the layer-0 KDA and layer-3 MLA attention outputs, every MoE
layer's expert ids and router cut margin, every MLA layer's indexer selections, cut margins and scores; and
a second run with the engines' q_a/kv_a eps 1e-5 for the layer-3 seam only.

**Ties.** The test sequences are grown a token at a time, keeping a token only if no indexer or router cut
at its position lies within 12× the gate's tolerance of a tie; the gate fails if a golden row has an indexer
cut margin under 10 × 3e-4 or a router cut margin under 10 × 5e-5. The golden's smallest are 4.18e-3 (14×)
and 6.26e-4 (12.5×), with no exact tie at any cut. Breaking every tie toward the later key or the later
expert passes the gate with every selection identical. `KimiK31Ties` builds exact ties on purpose (keys in
groups of four identical inputs, ten or more tied rows) and checks the reference, the independent indexer,
vLLM's scores with `torch.topk` and the gate's comparison agree on them as sets.

**Gate results** (`tools/kimi_k31_gate.py check`, the reference against the golden, 6 runs pooled):

| Check | float64 reference | float32 reference | Tolerance (f64 / f32) |
| --- | ---: | ---: | ---: |
| logits max, full vocabulary (\|logit\| ≤ 4) | 8.0e-5 | 1.9e-4 | 3e-4 / 5e-4 |
| logits rms | 5.6e-6 | 9.5e-6 | 1e-5 / 2e-5 |
| layer-3 MLA seam max / rms | 2.1e-6 / 2.9e-7 | 2.6e-6 / 4.4e-7 | 5e-6 / 7e-7 |
| layer-0 KDA seam max / rms | 2.3e-6 / 2.5e-7 | 2.1e-6 / 3.3e-7 | 3.7e-6 / 4.0e-7 (f32 4.7e-7) |
| indexer scores max (`-inf` exactly on future keys) | 6.6e-5 | 7.2e-5 | 3e-4 |
| expert sets, 1,872 rows | all equal | all equal | exact |
| indexer sets, 702 rows | all equal | all equal | exact up to ties |

The tolerances come from the independent model's own noise: real fla is not bit-reproducible (two fresh
runs differ by 4.7e-5 on the logits), so the float64 reference, exact to about 1e-13 on this fixture,
measures fla's error; against a second fresh run it is within 7.1e-5. The pure-torch stand-in against real
fla, all fp32: 1.1e-4 max on the logits (2.7e-5 relative) with the stand-in on CPU, 6.3e-5 with both on
CUDA, 9.1e-5 with the stand-in in fp64; prefill against decode inside one fla run, 5.3e-5. None of these
changes an expert set, an indexer set or an argmax. fla as published (TF32 dots) moves the logits by 1.1 and
flips 29 indexer rows, 45 expert rows and 11 argmaxes.

**The two epsilons.** The layer-3 seam resolves the q_a/kv_a eps in both directions: the wrong eps is 1.9e-6
rms on it against 2.9e-7 (f64) / 4.4e-7 (f32) matched, while on the logits it is 2.1e-4 max, under the
bound (only the f64 logits rms, 1.2e-5, trips). The layer-0 seam resolves the KDA q/k l2 eps (fla 1e-6,
llama.cpp's graph 1e-5), narrowly: wrong 6.5e-7 / 6.8e-7 rms against 2.5e-7 / 3.3e-7 matched, bounds about
1.6× (f64) and 1.4× (f32) from each side, with fla's run-to-run spread there (2.6e-8) more than ten times
smaller than either gap. Only that direction can be tested: fla's eps is fixed.

**Negative tests.** 20 source-level mutations, each applied to a fresh copy of the reference (or of the
fixture's converter) and run through the gate; a mutation whose text no longer matches the source errors
out. All 20 are caught - 19 fail the gate, and the reference refuses the last:

| Kind | Mutations | Caught by | Logits max |
| --- | --- | --- | ---: |
| structural (15) | the converter taking `A_log`'s last `num_heads` entries (the zero padding); attention-residual blocks one layer early / late; future keys 0-filled with a whole-row top-k; no ReLU in the indexer; indexer top-k ± 1; Kimi-Linear's softplus KDA gate; routed norm before the weighted sum; MLA with RoPE; experts truncated to 8; router bias dropped (also with the reference's own margins squeezed 1e4×, so its cut margins look like near-ties); expert weights from the biased score; SiTU betas swapped | logits, the layer-3 seam, experts, indexer sets and scores (the two KDA ones the layer-0 seam too) | 3.5 - 6.4 |
| precision-level (2) | q_a/kv_a eps 1e-5; KDA l2 eps 1e-5 | the layer-3 seam; both seams - each also the f64 logits rms | 2.1e-4; 1.5e-4 |
| no output change (2) | future keys 0-filled with the top-k still over visible keys; unscaled indexer head weights | the indexer-score check only | 8.0e-5 (baseline) |
| refused (1) | `A_log` sliced to the stored length | the reference refuses `ssm_a`'s shape | - |

**The big fixture** (`--geometry big`, §7.1; CPU only, no golden yet). Its noise is not tiny's, so its tie
bands come from its own (`tools/kimi_k31_gate.py gap --geometry big`, recorded in the big manifest as
`reference_gap`). The float32 reference against the float64 one on its three sequences, each one prefill, per
eps preset (moonshot / llama.cpp):

| Gap, float32 vs float64 reference | moonshot | llama.cpp |
| --- | ---: | ---: |
| logits max (\|logit\| ≤ 4.9) / rms | 1.45e-3 / 4.35e-5 | 1.50e-3 / 4.55e-5 |
| layer-3 MLA seam max / rms | 4.11e-6 / 3.27e-7 | 5.30e-6 / 3.64e-7 |
| layer-0 KDA seam max / rms | 1.60e-6 / 2.41e-7 | 1.71e-6 / 2.41e-7 |
| router scores max / largest cut-margin shift | 1.20e-4 / 5.80e-5 | 1.52e-4 / 7.23e-5 |
| indexer scores max (\|score\| ≤ 3.8) / largest cut-margin shift | 3.73e-4 / 2.20e-4 | 3.67e-4 / 1.74e-4 |
| expert sets (3,852 rows), indexer sets (1,284 rows) | all equal | all equal |

Tiny's float32 tolerances do not carry over: the logits gap alone is about 3× the gate's 5e-4, and the indexer
scores exceed its 3e-4, so an engine on this fixture is bounded at 2× these measured gaps, never at tiny's
numbers. The largest cut-margin shift any of the runs makes against float64 moonshot is 9.1e-5 (router) and
3.4e-4 (indexer). The big tie bands (`TIE_BANDS`) are 1.25e-4 and 4e-4 (1.4× and 1.2× that shift), so a golden
on these sequences must clear 1.25e-3 / 4e-3 (14× / 12× the shift). The sequences are picked at 1.5e-3 / 4.8e-3
in float64 under both presets, which leaves 2.8× / 2.3× the shift of room above that check. Their smallest
margins are 1.501e-3 and 4.80e-3 (seq0 took 620 draws, seq2 3,663 draws, 9 minutes). Tiny's multiples could not
be kept here: with bands of 1.9× / 3.8× the shift, only about one random candidate in 150 clears every cut at a
position where the indexer selects. fla's own noise on this fixture is not measured yet. A golden whose margins shift by more than a band
needs a wider band and a new pick. `selftest` pins the fixture's files and geometry to the ones the pick ran on.

**IQ1_S.** A llama.cpp build with `kimi-k31` support quantizes `big.gguf` to IQ1_S (`llama-quantize --imatrix
FILE big.gguf out.gguf IQ1_S`), where FILE is the all-ones importance matrix that `kimi_k31_synth_gguf.py
--unit-imatrix FILE` writes. The fixture has no text to compute a real one from, and llama-quantize refuses
IQ1_S without one. Exit 0, 1,236 MiB → 104 MiB (2.70 BPW).
All 27 routed-expert tensors are IQ1_S, along with the shared experts and the KDA/MLA projections. The rest follows
llama-quantize's usual IQ1_S mix: `attn_output` IQ2_XXS, `attn_v` Q4_K, `token_embd` and the dense `ffn_down`
Q2_K, `output` Q5_K. Routers, the indexer, `ssm_beta` and the conv taps stay F32, and only the ten 128-wide
tensors (`attn_k_b`, `ssm_f_b`) fall back, to IQ4_NL.

**Test runs on this tree.** The four K3.1 test files with torch available, live tests excluded: 73 passed,
11 subtests, in 53 s on CPU. The live tests (the oracle run fresh on the GPU, the reference gated against it,
and the fresh run checked against the committed golden), on the RTX 3090: 2 passed in 3 min. The fresh run
is within 4.7e-5 of the committed golden on the logits, 1.0e-6 on the seams and 2.6e-5 on the indexer scores,
and the float64 reference passes the gate against it (logits 7.1e-5 max, every selection equal). In CI's
configuration (§7.3) the same files run their numpy parts and skip the rest.

### 7.3 Phase 2: what was run

* **Build.** This tree, configured `-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON` against ggml from llama.cpp `3cf03257` (the
  installer's pin), GCC 15.2 and CUDA 12.4: all 465 steps from a clean build directory, no errors.
* **GLM parity self-tests**, on one RTX 3090: `glm_hc_parity`, `router_sigmoid_parity`, `glm_kda_parity`,
  `glm_ffn_parity`, `glm_dsa_parity`, `glm_kv_stream_parity` and `glm_layer_parity --selftest`,
  `glm_model_test --selftest` (batched-8 4.08e-6, streamed-8 0, 12-token 4.08e-6, continuation 2.77e-6) and
  `glm_memory_test`: all PASS, and their output is byte-identical to v1.1.53's (recorded from a base build on
  an RTX 3060), except `glm_kv_stream_parity`'s count of block reads that hit VRAM, which differs between
  runs of the same binary.
* **The GLM and EXL3 ctest set** (27 tests: the parity tests above, `glm_model_test`, `glm_memory_test`,
  `glm_split_devices_test`, `glm_limits_test`, `glm_limits_engine_test`, `glm_moe_rows_test`,
  `glm_rows_kernels_test`, `glm_rows_attn_test`, `glm_lat8_test`, `glm_layer_graph_test`, the big fixture's
  `glm_model_big_gen` + `glm_model_big_test`, `router_top10_parity`, `glm_mv_nt_test`, `glm_mv_nt_generic`,
  `glm_mmq_glu_test`, `exl3_test`, `exl3_exllamav3`, `exl3_mma_rows_test`, `exl3_hip_scalar_policy_test`),
  on the same GPU: 26 pass, `glm_model_big_test` (the 45-layer fixture, 8 of 32 experts, per-op path)
  among them. `glm_layer_graph_test` fails ("capture cache not exercised"); by its code it fails on
  v1.1.53 too (§7.6).
* **The refusal tests.** `glm_limits_test` (CI's host-only step, GCC 15.2 with `-Wall -Wextra -Werror`):
  PASS. Through the real `from_gguf` and gate, with `STRATA_GLM_SLOW` unset and then set, it refuses a
  `kimi-k31` header by its architecture (and shows the `glm5-next` reading of the same file would take
  GLM-5.3's defaults and pass every route gate - why the architecture check must come first), a missing or
  non-string architecture, a route wider than its layer or of no experts, and on the fast path 9 or 16 of
  288, 8 of 896, `n_embd` 7168, K3.1's 16 of 896 at 7168, 64 and 65 of 512 - each message naming the route;
  the per-op path takes routes up to 64 of 512 and refuses 65 of 512 and any layer of more than 512
  experts. GLM-5.3's 8 of 288 at 4096, the limits themselves (8 of 512 at 4096) and both spellings pass.
  `glm_limits_engine_test` (through `Glm5Model::load` and `load_pack_env`, on copies of
  `data/glm5-synth.gguf` with the route patched in place: 16 of 32, 16 of 12, 3 of 896, 3 of 12 at
  `n_embd` 8192, 65 of 512) passes with no GPU visible, so nothing before a refusal needs a device, and
  refuses the K3.1 fixture's `tiny.gguf` by its architecture. `glm_moe_rows_test` drives `moe_route` at 9
  of 512 and 8 of 513, `moe_fetch`, `moe_wait_fetch`, `moe_gate_up` and `moe_down` at 9 picks,
  `glmb::route`, `moe_combine`, `moe_combine_add` and EXL3's `moe_gate_up` / `moe_down` at 9: each adds
  exactly one counted failed launch. When these tests were written, eight deliberately broken copies
  of the two headers each made the CI test fail, and removing EXL3's counting made `glm_moe_rows_test` fail
  (`0 failed launch(es) counted, want 2`); those mutation runs were not repeated on this tree.
* **CI's checks**, as `.github/workflows/tests.yml` runs them, in a Python 3.12 venv with exactly CI's
  packages and gguf-py from llama.cpp `3cf03257`: `tools/check_hip_compat.py` passes (no new CUDA name
  lacks its mapping); the GLM memory, GLM limits and both DeepSeek row-preparation host tests pass; server
  tests 708 OK (21 skipped); installer and tools tests 793 OK (53 skipped; the 75 K3.1 tests among them,
  17 of those skipped for want of torch).

### 7.4 The `A_log` answer

`A_log` is per head: use the first `linear_attn_config.num_heads` (96) entries and assert the rest is zero
at conversion (§5.1). The fixture's converter and the independent model slice that way and check the tail;
the reference refuses an `ssm_a` of any other length.

In the release itself (`moonshotai/Kimi-K3.1-Pro` @ `6d0a0202`), read straight from the safetensors shards -
a header parse and a small positioned read per tensor, only from shards whose size matches both their own
header and the Hub's listing - all 69 KDA layers were checked: every `A_log` is F32 [128] with entries
96..127 bitwise +0.0 and entries 0..95 finite, from -2.47 to 2.46 (so `exp(A_log)` from 0.085 to 11.7);
every `dt_bias` is F32 [12288] = 96 × 128 and finite; by the checkpoint index, none of the 24 MLA layers
carries either. So K3.1's tail is the zero padding K3's is reported to be, and the slice loses nothing.

### 7.5 Corrections to this study

The first version of this document was written before Phases 1 and 2 were built. What they showed wrong
or incomplete is corrected above:

* §5.3: excusing attention outputs row by row at a tied cut does not work - a different tied key changes
  every later output - so the gate is built tie-free and ties are tested as sets on their own.
* §5.2: the reference defaults to Moonshot's q_a/kv_a eps 1e-6 rather than the GGUF's 1e-5 first
  proposed, and the gate now tells both that eps and the KDA l2 eps apart, at two seams; the study had
  left both open.
* §5.1 / §5.6: K3.1's `A_log` tail has been read (§7.4).
* §6 Phase 1: the test files' names; transformers 5.22 needs four runtime shims, not five (no `tie_weights`
  shim); an fp32 golden needs fla's dots forced to ieee; the negative tests live in the gate's tests; the
  reference cannot refuse `index_rope_dim` or the key-norm type, which the GGUF does not carry.
* §6 Phase 2: items 3 and 4 did not land, and the refusal comes after the first shard's header is mapped
  rather than before any allocation.
* §6 acceptance: the logit gate is a max / rms bound, not KL p99 and top-1.
* §4.6 and §4.8 describe v1.1.53. On this branch `from_gguf` lives in `glm_geometry.hpp`, and the
  "existing guards" of §4.6's table - EXL3's silent `moe_args` refusal among them - are counted refusals.

### 7.6 Not checked

* Real K3.1 weights through the reference, the independent model or the engine: only `A_log` and
  `dt_bias` were read from the release (§7.4). The hf-mxfp4 copy was never run through the oracle or the
  reference; `ref/load.py` reads MXFP4 now, and only its GGUF conversion's decode was checked against the F32
  copy (tools/test_kimi_k31_synth.py, KIMI_K31_MXFP4_GGUF). No bf16 run was made on this version of the fixture.
* Which epsilons K3.1 was trained with: the gate tells 1e-6 from 1e-5 at both norms, not which is right.
* SGLang's `_fp8_mqa_logits_torch` itself, with its bf16 cast and 0-fill - its semantics were reproduced for
  unquantised inputs. Both indexer sources are unmerged and were fetched by commit id; whether either
  branch has moved since was not checked.
* Moonshot's code on transformers 4.57.x; a KDA head count different from the MLA one (the format cannot
  express it, so slicing `A_log` by `num_attention_heads` instead of `linear_attn_config.num_heads` cannot be
  told apart on this fixture); long contexts (the fixture's top-k is 6 and its longest sequence 80 tokens).
* Phase 2 on a real pack: no `load_pack` + `fast_setup` + fast-decode run at k = 8 (no synthetic pack
  fixture exists; `glm_pack_test` and its siblings need a real GLM-5.3 pack), so the fast path is covered by
  the kernel tests above and by reading; no decode / prefill A/B of the per-launch checks; no HIP, Metal or
  Windows build.
* The new failed-launch counting in `kda_rec`, `kda_layers`, `kda_gather`, `hc_finish`, `kda_conv` and the
  `mla_attn` variants has no test of its own; `fast_setup`'s own check cannot be reached past `load_pack`'s.
* Left as found: `src/mm3/mm3_exl3.cu` (MiniMax M3's own copy of the EXL3 kernels) still refuses in
  `moe_args` without counting it; DeepSeek's main expert path does not read EXL3's launch-error count; the
  GLM fast path does not check `kda_head_dim` or `kv_lora_rank` at load (a pack with other values now gets a
  counted launch error at decode rather than wrong output); `glm_layer_graph_test` fails on this branch and,
  by its code, on v1.1.53 (it requires `LayerGraphs::invalidations > 0`, which nothing increments; this
  branch touches neither file); it was not run on a base build.
* CI itself: the suites ran in a local venv with CI's package list, not on `ubuntu-latest`. The fixture's
  byte-identical regeneration relies on numpy's seeded generator giving the same stream across versions.

### 7.7 Phase 3: the engine's seam dumps (`check-dumps`)

`tools/kimi_k31_gate.py check-dumps --dumps DIR [--dtype float32]` (also `check-engine`) reads an engine's dumps
of every golden run (`<seq>.<mode>/manifest.txt` + `<name>.f32` in ggml order, or `<seq>.<mode>.npz`) and judges
them in ladder order. Exit 0 pass, 1 fail, 2 refused. Dumps that cannot be judged are refused before anything is
compared: an unknown or missing run, a name the reference does not dump, a seam on a layer that has no such seam,
a non-canonical layer number, any shape other than the reference's (only the last axis of the expert, key and
residual-block selections may differ, and that is judged), a missing seam a rung needs, an unreadable file, a
`--gguf` that is no GGUF the reference reads (an unreadable header, a tensor type `ref/load.py` does not decode,
tensor data cut short) or is of another geometry (judged from its header and size, before any tensor is read or
the file hashed), a GGUF other than the fixture (checked by size before it is hashed), or `--golden` with
`--geometry big`. `dump-ref` refuses the same unreadable or other-geometry `--gguf` and the same `--golden`; it
does not check the fixture's identity (another GGUF of that geometry gets its own reference dumps). An error once
these checks have passed is a real one: it is raised (a traceback, exit 1), never printed as a refusal.

| Rung | Seams | Judged against | Bound, max / pooled rms | float32 reference |
| --- | --- | --- | ---: | ---: |
| 1 | `kda_out-0` | golden `layers.0.attn_out` | 3.7e-6 / 4.7e-7 | 2.1e-6 / 3.3e-7 |
| 2 | `l_out-0` (dense SiTU FFN) | float64 reference | 1.0e-5 / 7.0e-7 | 3.27e-6 / 3.27e-7 |
| 3 | `ffn_moe_out-1`; `ffn_moe_topk-1` | float64 reference; golden expert ids | 1.1e-5 / 7.0e-7; exact | 3.55e-6 / 3.36e-7; equal |
| 4 | `mla_out-3`; `top_k-3`, `indexer_score-3` | golden | 5e-6 / 7e-7; sets up to ties, scores 3e-4 | 2.6e-6 / 4.4e-7; equal, 7.2e-5 |
| 5 | `res_probs_ffn-3`, `res_mix_ffn-3` (the first mix after the push at 3) | float64 reference | 7.5e-6 / 1.1e-6, 4.1e-5 / 2.8e-6 | 2.42e-6 / 5.43e-7, 1.35e-5 / 1.37e-6 |
| 6 | `result_output`; every expert and indexer set | golden | 5e-4 / 2e-5; all 1,872 / 702 rows | 1.9e-4 / 9.5e-6; all equal |

The golden is used only when the engine's eps are its own (Moonshot's, or q_a/kv_a 1e-5 for the layer-3 seam);
otherwise, and with `--reference ref64`, every seam is judged against the float64 reference with the same bounds.
Indexer scores must be `-inf` exactly on the masked (future and padded) keys: a NaN, +inf or 0 there fails.

**The new bounds.** The golden does not carry rungs 2, 3 (`ffn_moe_out`) and 5, so they are judged against the
float64 reference. Each bound (`REF64_TOL`) is about 3× the float32 reference's measured max gap to the float64
one and 2× its pooled rms gap, over all six runs of the tiny fixture (the same margins as the golden's bounds;
the column above, measured the same with 1 and 6 BLAS threads). The self-test re-measures them and fails if a
bound falls under 2× / 1.5× the gap or rises above 5× / 3× of it, so a bound cannot be widened without a new
measurement. They hold for the tiny fixture only.

**The big fixture** (`--geometry big`, §7.2; plan M1a gate 2). It has no golden yet, so every seam is judged
against the float64 reference, on the big fixture's own tie-free sequences (seq0/1/2 of 96, 32 and 300 tokens,
each prefill and decode), with its tie bands (`TIE_BANDS`) and its own bounds (`DUMP_TOL`, `DUMP_REF64_TOL`): 2×
the float32 reference's gap to the float64 one, max and pooled rms over all six runs, the larger of the two eps
presets, rounded up. The ladder is the same (KDA and dense layer 0, MoE from layer 1, MLA at 3, 8 and 9, the push
at 3). The GGUF must be the one its sequences were picked on (`BIG_PICK`); its header and geometry are checked
first, so a real K3.1 GGUF is refused without being hashed. `--golden` is refused: the big fixture has none.
`dump-ref --geometry big` writes the reference's dumps of the same six runs.

| Seam | float32 reference, max moonshot / llama.cpp; rms | Bound, max / pooled rms |
| --- | ---: | ---: |
| `kda_out-0` | 1.60e-6 / 1.71e-6; 2.29e-7 | 3.5e-6 / 4.6e-7 |
| `l_out-0` | 9.94e-6 / 8.12e-6; 4.35e-7 | 2.0e-5 / 8.7e-7 |
| `ffn_moe_out-1` | 5.12e-6 / 5.98e-6; 3.99e-7 | 1.2e-5 / 8.0e-7 |
| `mla_out-3` | 4.11e-6 / 5.30e-6; 3.47e-7 | 1.1e-5 / 7.0e-7 |
| `res_probs_ffn-3`, `res_mix_ffn-3` | 1.04e-5 / 1.47e-5; 1.45e-6, 7.17e-5 / 9.25e-5; 2.62e-6 | 3.0e-5 / 2.9e-6, 1.9e-4 / 5.3e-6 |
| `result_output` (\|logit\| ≤ 4.9) | 1.45e-3 / 1.50e-3; 4.23e-5 | 3.1e-3 / 8.5e-5 |
| indexer scores, layers 3, 8, 9 (\|score\| ≤ 3.8) | 3.73e-4 / 3.67e-4 | 7.5e-4 |
| expert sets (7,704 rows), indexer sets (2,568 rows) | all equal | all equal |

The prefill-only maxima agree with §7.2's `gap` table. One preset's dumps judged under the other still fail,
but first on rung 4 (the q_a/kv_a eps), both ways: on this fixture the KDA l2 eps moves `kda_out-0` by less than
rung 1's bound. With `KIMI_K31_BIG_FIXTURE` set to the big build's directory,
`tools/test_kimi_k31_check_dumps.py` re-measures the gap and requires each bound to be 1.9-2.5× it (measured
2.00-2.07×; the 1.9 leaves room for the BLAS thread count moving the gap), checks that the reference's float32
dumps pass under both presets, that a 1e-3 change of a seam fails its rung and that each preset's dumps judged
under the other fail first on rung 4, and runs `dump-ref` and `check-engine --geometry big` through the CLI.

**Mutations.** The eleven M1a switches (ten applied to the float32 reference, the `A_log` tail slice to the
fixture's converter) plus indexer top-k + 1 each fail first on the expected rung, with every earlier rung passing:
the KDA softplus gate and the `A_log` tail slice on rung 1 (418,371× and 268,412× the bound), blocks pushed one
layer early or late and SiTU betas swapped on rung 2 (225,137×, 225,137×, 7,295×), routed norm before the sum,
experts truncated to 8 and router bias dropped on rung 3 (65,381×, 53,972×, 65,116×), and future keys 0-filled
over the whole row, no ReLU and top-k ± 1 on rung 4 (185,038×, 138,100×, 87,763×, 103,532×).
`tools/test_kimi_k31_check_dumps.py`: 40 tests, 36 on the tiny fixture (about 3 minutes on CPU) and 4 that
need `KIMI_K31_BIG_FIXTURE` (skipped without it).
