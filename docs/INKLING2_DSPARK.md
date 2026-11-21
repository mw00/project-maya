# Inkling 2 Small's DSpark drafter on Maya: the port spec (maya-ik2)

Working spec for speculative decoding in `maya-ik2` with Thinking Machines' own drafter (`mtp.safetensors`,
`mtp_config`). Companion to [INKLING2_AMD.md](INKLING2_AMD.md) (the target's forward pass, whose notation this
uses: `rms(x)`, `rmsg(x, g)`, eps 1e-6). The equations are read from the reference's source; the acceptance rate and the tap / RoPE questions are measured
on the reference (Measured); the cost model is arithmetic over measured routing, and a window's cost on our engine
is still unmeasured.

Sources:

- The reference: llama.cpp PR #32218 "Inkling 2 DSpark support" (open; head `842c10f617`, branch
  `pwilkin/inkling2-dspark` on base `pwilkin/inkling2`, last updated 2026-11-14). Diff:
  `<ref>/dspark-pr32218.diff`. Everything under "The forward" is `graph_inkling2` /
  `build_inkling2_head` in its `src/models/dflash.cpp` and the `has_rnn_context` paths of its `common/speculative.cpp`.
- The parts the diff builds on, read in `<llama.cpp>` (master b94959a64): the KV mask
  (`src/llama-kv-cache.cpp`, `block_uniform`), `is_masked_swa`, `llama_model_rope_type`, `examples/speculative-simple`.
- Thinking Machines' README (HF `thinkingmachines/Inkling-2-Small`) says nothing about the drafter. There is no
  first-party description to check the PR against: where the PR guesses, so do we (Open questions).
- The template in this repo: maya-ds4's DSpark path (`src/ds4/ds4_model.cu` `seed_drafter` / `draft` / `verify` /
  `rollback`, `src/ds4/ds4_main.cpp` `round`, `src/spec/draft_policy.cpp`).

## The files

`<models>/Inkling-2-Small-MTP/`:

| file | what |
|---|---|
| `mtp.safetensors`, `config.json` | Thinking Machines' checkpoint: 84 BF16 tensors, 3,423,420,768 bytes of weights |
| `Inkling-2-Small-DSpark-BF16.gguf` (3.44 GB) | the matrices as they are; every 1-D tensor and `dspark_fuse` F32. Checked against the checkpoint to the bit |
| `Inkling-2-Small-DSpark-Q8_0.gguf` (1.83 GB) | the matrices as ggml Q8_0 blocks, the rest F32. Checked: every value within its block's rounding |
| `maya-ik2-dspark-bf16/`, `maya-ik2-dspark-q8_0/` | the engine's index of each: `dspark.txt` (geometry, `key value`), `tensors.txt` (`file 0 <gguf>`, then `t <name> <type> <file> <offset> <bytes> <ne...>`, the format of `tools/ik2/pack_gguf.py`) |

Written by `tools/ik2/convert_dspark.py` (no torch; 33 s niced for both):

```sh
nice -n 19 python3 tools/ik2/convert_dspark.py --mtp <models>/Inkling-2-Small-MTP \
  --tokenizer-gguf <models>/Inkling-2-Small-GGUF/UD-Q2_K_XL/Inkling-2-Small-UD-Q2_K_XL-00001-of-00006.gguf
```

The GGUFs follow the PR's converter (`Inkling2DSparkModel`): architecture `dflash`, its tensor names, dims and keys
(`dflash.block_count 5`, `block_size 8`, `target_layers [5,11,17,23,29,35,41,48]`, `dspark.markov_rank 256`,
`sample_from_anchor false`, `has_confidence_head false`, `attention.causal false`, `attention.sliding_window 1024`,
`attention.sliding_window_pattern [T,T,F,T,T]`, `attention.key_length / value_length 128`, `rope.dimension_count 128`,
`rope.freq_base 1e6`, `final_logit_softcapping 30`, `vocab_size 201024`, `vocab_size_unpadded 200058`,
`tokenizer.ggml.mask_token_id 0`). The tokenizer keys are copied from the target's GGUF. Not yet loaded by a llama.cpp
build - the first step of the parity plan.

The engine should load the **Q8_0** file: `ds4_iq`'s GEMVs take Q8_0, while `ik2_hip_model.cu` expands BF16 to F32 in
VRAM (4 bytes a weight: 6.4 GB read a block instead of 1.7). One caveat: `joint_proj.weight` has small weights, and
46% of its Q8_0 blocks get an F16-subnormal step (coarser rounding, worst error 0.6% of a block's largest value).
It is 15 MB: take it (and `dspark_hidden_out`) from the BF16 file if the drafts look worse than the reference's.

## Constants (`mtp_config`)

hidden 4608; 5 layers; 48 query / 8 KV heads, head_dim 128 (GQA 6, as the target); FFN 18432 (SwiGLU, dense, no
experts); RMS eps 1e-6; RoPE theta 1e6 over all 128 dims; layers 0, 1, 3, 4 sliding-window 1024, layer 2 global;
block 8 (the anchor + 7 drafts); markov rank 256, `rnn_hidden`; no q / o bias; no confidence head.

## Tensors (checkpoint name -> GGUF name; shapes as the checkpoint's `[out, in]`, GGUF dims are the reverse)

Global:

| checkpoint | GGUF | shape | role |
|---|---|---|---|
| `fuse_weight` | `dspark_fuse.weight` (F32) | [6, 8] | row l (0-4): layer l's mixing logits over the 8 target taps; row 5: the markov head's |
| `mask_embed` | `mask_embd.weight` | [4608] | the input of draft rows 1-7 (one vector, the same for every row; used raw) |
| `norm.weight` | `output_norm.weight` | [4608] | the final RMS norm's gain |
| `markov_head.dflare_norm.weight` | `dspark_norm.weight` | [4608] | gain of the markov head's fused context |
| `markov_head.markov_w1.weight` | `markov_w1.weight` | [201024, 256] | previous-token embedding (a row a token) |
| `markov_head.rnn_proj.weight` / `.bias` | `joint_proj.weight` / `.bias` | [768, 9728] / [768] | the recurrence: 9728 = 256 state + 256 token + 4608 hidden + 4608 context; 768 = gate, candidate, latent |
| `markov_head.hidden_out.weight` / `.bias` | `dspark_hidden_out.weight` / `.bias` | [4608, 256] / [4608] | latent -> a correction in hidden space, which goes through the **target's** output matrix |

Each layer `layers.L.` -> `blk.L.`:

| checkpoint | GGUF | shape | role |
|---|---|---|---|
| `input_layernorm.weight` | `attn_norm.weight` | [4608] | pre-attention norm gain |
| `post_attention_layernorm.weight` | `ffn_norm.weight` | [4608] | pre-FFN norm gain |
| `dflare_norm.weight` | `dflare_norm.weight` | [4608] | gain of this layer's fused target context |
| `self_attn.q_proj.weight` | `attn_q.weight` | [6144, 4608] | queries of the block's rows |
| `self_attn.k_proj.weight`, `v_proj.weight` | `attn_k.weight`, `attn_v.weight` | [1024, 4608] | keys / values of the block's own rows |
| `self_attn.k_proj_ctx.weight`, `v_proj_ctx.weight` | `attn_k_ctx.weight`, `attn_v_ctx.weight` | [1024, 4608] | keys / values of the committed positions, from the fused target context |
| `self_attn.q_norm.weight`, `k_norm.weight`, `k_norm_ctx.weight` | `attn_q_norm`, `attn_k_norm`, `attn_k_ctx_norm` `.weight` | [128] | per-head RMS norm gains (one gain vector for all heads) |
| `self_attn.o_proj.weight` | `attn_output.weight` | [4608, 6144] | |
| `mlp.gate_proj.weight`, `up_proj.weight` | `ffn_gate.weight`, `ffn_up.weight` | [18432, 4608] | |
| `mlp.down_proj.weight` | `ffn_down.weight` | [4608, 18432] | |

Bytes (BF16): a layer's block-pass matrices 641.7 MB (q 56.6, k 9.4, v 9.4, o 56.6, FFN 3 x 169.9), ctx K + V 18.9 MB;
`markov_w1` 102.9 MB, `joint_proj` 14.9 MB, `hidden_out` 2.4 MB. Q8_0: x 0.531.

From the target, the drafter also uses `token_embd` + `token_embd_norm` (the anchor row) and `output` (the head, twice).

## The target taps

The drafter is fed, for every committed position p, 8 vectors of 4608 from the target's pass over the token at p:

- `tap_k`, k = 0..6 (ids 5, 11, 17, 23, 29, 35, 41): the residual stream `cur` **entering** layer id, i.e. the
  reference's `l_out-(id-1)` = `l_out-4, -10, -16, -22, -28, -34, -40`; in our engine `gp.cur` after layers 4, 10, ...,
  40 (the value the `"l_out"` tap already reports). It is the raw stream: before that layer's depth mix
  (`attn_res_query`) and before any norm. It is the current 6-layer block's stream only (each block starts a new
  one), and it is not one of `blocks[]` (those are `l_out-5, -11, ...`, after the global layer), so the GPU
  hand-off does not carry it already.
- `tap_7` (id 48 = n_layer): the head's input after the depth mix **and** the output norm,
  `rmsg(mix(blocks, cur, output_res_query), output_norm)` = the reference's `result_norm` (before `out_ids`), our
  `gp.xf`. The PR sets `t_layer_inp[n_layer] = cur` after `build_norm`.

Row RMS on the long reference prompt: taps 0-6 about 4.8, 4.0, 3.7, 2.4, 1.9, 1.8, 1.9; tap 7 about 4.6. They are
summed unnormalised (below), so a tap taken at the wrong point changes the mix visibly.

Engine work: the decode path copies 7 rows out at their layers (on whichever GPU holds them) and the head's row; the
**prompt path must produce all 8 for every prompt row**, which means the head's mix + norm for every row of a chunk
(today only the chunk's last token reaches the head; no output matmul is needed for the others).

## The forward

### 1. Context injection (once per committed position p, any number of rows at a time)

```
for l in 0..4:   c_l[p]   = rmsg( sum_k softmax(fuse[l])_k * tap_k[p],  blk.l.dflare_norm )
                 Kc_l[p]  = rope( rmsg_h( attn_k_ctx_l @ c_l[p], attn_k_ctx_norm_l ), p )      8 heads x 128
                 Vc_l[p]  =        attn_v_ctx_l @ c_l[p]                                       (no norm, no rope)
f[p] = rmsg( sum_k softmax(fuse[5])_k * tap_k[p],  dspark_norm )                               4608
```

`softmax(fuse[l])` is over the 8 taps (the checkpoint's values: layer 0 puts 0.49 on tap 5, layer 1 0.36 / 0.55 on
taps 4 / 5 and nothing on 6 / 7, layer 4 0.48 on tap 7; every tap matters somewhere). `rmsg_h`: RMS norm over each
head's 128 values, times the 128-value gain. `Kc_l`, `Vc_l` go into layer l's cache at position p; `f[p]` is kept
for the last committed position only (the PR keeps a ring of 9).

`rope(v, p)` for a head's 128 values, i = 0..63, theta_i = 1e6^(-i/64): pairs are **(v[i], v[i+64])** (ggml's NEOX
form): `v[i]' = v[i] cos(p theta_i) - v[i+64] sin(p theta_i)`, `v[i+64]' = v[i] sin(p theta_i) + v[i+64] cos(p theta_i)`.
(The PR does not set a rope type; `LLM_ARCH_DFLASH` without DeepSeek's or Mistral's markers gets NEOX in master's
`llama_model_rope_type`. Open question 3.)

### 2. One block (the backbone): 8 rows at positions n .. n+7

n = the position of `y`, the last sampled token, which the target has not read yet; contexts exist for 0 .. n-1.

```
x_0     = rmsg( token_embd[y], token_embd_norm )          the anchor row: the target's own normed embedding
x_i     = mask_embd                    i = 1..7           (raw, the same vector)
per layer l:
  a_i   = rmsg( x_i, attn_norm_l )
  Q_i   = rope( rmsg_h( attn_q_l @ a_i, attn_q_norm_l ), n+i )      48 x 128
  Kb_i  = rope( rmsg_h( attn_k_l @ a_i, attn_k_norm_l ), n+i )       8 x 128
  Vb_i  =               attn_v_l @ a_i
  keys of EVERY row i (the same set for the 8 rows):
     the block's own 8 rows (Kb, Vb) - all of them, both directions (non-causal)
     + the cached contexts (Kc_l[j], Vc_l[j]):  layers 0, 1, 3, 4:  n-1024 <= j <= n-1
                                               layer 2:            0 <= j <= n-1
  s_ihj = Q_ih . K_j,(h/6) / sqrt(128)          plain softmax over the key set (no sink, no bias, no clip)
  x_i   = x_i + attn_output_l @ ( sum_j softmax(s)_j V_j )
  b_i   = rmsg( x_i, ffn_norm_l )
  x_i   = x_i + ffn_down_l @ ( silu(ffn_gate_l @ b_i) * (ffn_up_l @ b_i) )
h_i     = rmsg( x_i, output_norm )                      (the drafter's norm.weight)
base_i  = output @ h_i                                  the TARGET's output matrix, 201024 raw logits, F32, uncapped
```

The window: the PR loads `n_swa = 1024 + 1` and gives the whole block the anchor's mask row (`block_uniform`), so a
key at j is masked when `n - j >= 1025`. The cache holds no context at n (the block's own anchor row is there), so
"+1" means: the anchor's cell plus the 1024 contexts before it - every row of the block sees the same 1024 contexts,
row 7 as far back as distance 1031. The block's own K / V are transient: they are not kept after the block
(llama.cpp writes them into the cache and removes them with `seq_rm` right after).

### 3. The markov head: draft tokens d_1 .. d_7, one at a time

```
s_0 = 0 (256);  t_0 = y;  c = f[n-1]
for i = 1..7:
  j     = [ s_{i-1} | markov_w1[t_{i-1}] | h_i | c ]                 9728
  z     = joint_proj @ j + joint_proj.bias                           768
  g     = sigmoid(z[0:256]);  cand = tanh(z[256:512]);  u = tanh(z[512:768])
  s_i   = cand + g * (s_{i-1} - cand)                                (= g s + (1 - g) cand)
  corr  = dspark_hidden_out @ u + dspark_hidden_out.bias             4608
  lg_i  = 30 * tanh( (base_i + output @ corr) / 30 )[0 : 200058]     softcap after the sum, then the vocab cut
  d_i   = t_i = argmax lg_i  (ties: the lowest id)   - or a draw, see Sampling
```

`u` comes from the same `z` as the state update (it sees `s_{i-1}`, not `s_i`). The state starts at zero for every
block. Row 0's logits are not used.

**The fold (ours, not the PR's).** The PR multiplies the target's whole output matrix (760 MB of Q6_K) by `corr` for
each of the 7 steps. Since it is linear: `output @ corr = M @ u + m0` with `M = output @ dspark_hidden_out`
([201024, 256], computed once at load, 103 MB as F16) and `m0 = output @ dspark_hidden_out.bias`. A step then reads
`joint_proj` + `M` (about 120 MB) instead of 780 MB. Same values up to float rounding; drafts need not be bit-exact.

## The drafting loop

State: the context ends at n (the target has read 0 .. n-1, contexts injected for them), `y` is the token for n.

1. `draft(y)`: one block at n .. n+7 -> d_1 .. d_7 and, as a confidence stand-in, each row's top probability.
2. Pick a window of k <= 7 drafts (Policy below). `verify`: the target reads `[y, d_1 .. d_k]` at n .. n+k in **one
   pass** (the prompt path's chunk of T = k+1 rows, from the saved position n), with the head on **every** row: row i
   gives the target's distribution for position n+i+1.
3. Accept: m = the leading drafts that stand (below); `next` = the target's pick at row m.
4. Roll the target back to n+m+1 (it keeps y and m drafts); inject the taps of rows 0 .. m into the drafter
   (positions n .. n+m); keep `f[n+m]`.
5. Emit d_1 .. d_m, `next`. Then y = next, n += m+1.

What the target hands over per pass: the 8 taps of each kept row (36,864 floats a row), and the T rows of logits.
A prompt: every row's taps (layer 2 is global, it needs all positions; the window layers need the last 1024).
While drafting rests, the taps of decoded tokens still have to be injected (or kept on the host, 147 KB a position,
and injected in one batch before the next draft).

What the drafter keeps: per layer a K / V cache keyed by position - layers 0, 1, 3, 4 a ring of 1024 + the rewind
margin, layer 2 a row a position up to `max_ctx` (F16: 4 KB a position and layer; 16K context: 151 MB in all) - and
`f` of the last committed position. No convolution history, no recurrent state: every row is a function of that
position's taps alone.

## Verification and acceptance

- **Greedy** (temperature 0): draft i stands while it equals the target's argmax at row i-1 (after the request's
  logit bias / penalties, each row counting the tokens before it, the drafts among them - as ds4's `pen_rows`).
  `next` is the argmax at row m. The answer is the token-by-token one, as far as the window pass is exact.
- **Sampled, drafts are the drafter's argmax** (ds4's default, llama.cpp's `--spec-draft-sampling greedy`): draft d
  at a row stands with probability p(d), p the request's own chain (temperature, top_k, top_p, min_p) on the target's
  row; otherwise the row's token is drawn from p with d removed. Past the first rejection the rows do not count.
  Each token is distributed as the target's own draw.
- **Sampled, drafts drawn** (ds4's `DS4_DSPARK_SAMPLE=1`, the PR's `probabilistic`): d_i is drawn from the drafter's
  q_i, the chain continues from the drawn token (`t_i` = the draw), and d stands with probability min(1, p(d)/q(d)),
  else a draw from max(0, p - q) normalised. The PR's q is `softmax(lg_i / temp)` over the whole vocabulary with the
  target's temperature - no top-k / top-p - and temperature 1 when the target's is 0. ds4 instead runs the request's
  own chain on the drafter's logits, which keeps more drafts; take ds4's, and never draw drafts for a greedy request.
- llama.cpp itself (`examples/speculative-simple`, the server) samples the target's row and compares, restoring a
  checkpoint and re-reading on a miss because Inkling's recurrent state cannot be `seq_rm`'d.

## Rollback and the engine's checkpoints

`Model::rewind(pos)` goes to the latest saved convolution history at or before pos (chunk starts, a prompt's end,
every 64th decoded token) and the caller reads on from there. That is too coarse for a rejected draft: re-reading the
kept rows would cost a second pass on every partial miss. A verify window needs its own undo:

- **Convolution histories** (4 streams a layer: K 1024, V 1024, attention-post 4608, FFN-post 4608 channels, 3 rows
  each): the history after row r is the pre-convolution inputs of rows r-2 .. r (the window's own rows, or the
  history from before the window where r < 2). The window pass keeps those inputs for its T rows; keeping m+1 rows
  sets each history from rows m-2 .. m. No recompute.
- **K / V caches** (rings and the global layers' rows): position-keyed; rows past n+m are dead once `end` is n+m+1
  and are overwritten when those positions come. The rings' rewind margin (`Options::rewind`, 4096 rows) already
  keeps the rows a window overwrites.
- **Expert cache counts / profile**: count the kept rows only.
- **The drafter**: nothing to undo - it is only ever given kept rows.
- `rewind()` proper (a regenerate, an edited history): the drafter's caches follow the same rule as the target's
  (position-keyed, the window rings with the same margin); `f` must be the one of the position before the new end, so
  either keep `f` at every saved checkpoint or re-derive it from the taps of the first row read after the rewind
  (drafting simply starts one token later).
- The window pass must give the tokens the token path gives ("chunked, exact" is the prompt path's claim; with T rows
  the head now matters for every row). Test: a greedy answer with drafting on equals the one with it off, on prompts
  where every expert is in VRAM.

## What maya-ds4 gives us, and what it does not

Reusable as is or nearly:

- `round()` in `src/ds4/ds4_main.cpp`: draft -> cut -> verify -> emit, the per-request keys (`spec`, `dspark_conf`,
  `dspark_cum`), `--dspark on|off|none`, the round log / `--spec-eval`, the per-position reached / kept counters.
- The serve protocol's fields (`drafts`, `draft_rounds`, `draft_ms`, `verify_ms`; `DONE`'s `drafts_acc drafts_off` are
  already in INKLING2_AMD.md's "Serving", `INFO dspark=0` becomes 1), `maya.py`'s speed calibration (`dspark_tok_s` against
  `no_drafting_tok_s`, keeps the faster).
- `strata::spec::DraftPolicy` (`src/spec/draft_policy.cpp`, DS4_DRAFT_COST): the window that commits the most tokens
  a millisecond from measured round times by rows and the drafts' acceptance. Engine-agnostic. For Inkling this
  should be the default rather than opt-in, because a row costs much more here (Cost model).
- `strata::spec::SuffixDrafter` (prompt lookup): `verify` takes drafts from any source.
- The acceptance rules: `strata::kernels::sample_tokens` / `sample_verify` / `SpecList` if the head's rows stay on
  the device; today maya-ik2 samples on the host (`mm3_sample.hpp`), where the same rules are a few lines.
- `ds4_iq`'s Q8_0 GEMVs for every drafter matrix; the engine's own window K / V rings, global cache and chunk
  attention GEMMs (the drafter has the target's 48 / 8 / 128 geometry and window).
- The shape of `seed_drafter` (taps -> context -> K / V -> ring, called from `verify` with the kept rows) and of
  `rollback(p0, T, keep)`.

Not reusable:

- ds4's drafter math: one `main_proj` over concatenated taps and one shared K/V projection with FP8 rows there; here a
  softmax mix per layer and separate `K_ctx` / `V_ctx`. Its taps are the mean of DeepSeek's hyper-connection streams.
- ds4's markov head (`logits[v] += E[tok] . H[v]`, `ds4_markov_q8.hpp`): a bilinear bias with no state. Inkling's is
  a GRU-style recurrence over (state, token, hidden, context) with the correction through the output matrix.
- **The confidence head.** ds4 cuts its window on a learned confidence (`--dspark-conf`, the running product
  `dspark_cum`); this drafter has none (`has_confidence_head false`). Stand-in: the draft row's own top probability
  `max softmax(lg_i)`, calibrated against measured acceptance per position before any threshold is trusted.
- The drafter's attention itself: RoPE and plain softmax instead of the target's relative bias, sinks and clips - a
  new (simpler) kernel variant; and two key sources per row (the block, the contexts).
- ds4's ring restore kernels (its compressed-KV layout).

## Cost model

Inputs: one decode pass ~30 ms today; dense weights 5.32 GB read once a pass; an expert 10.2 MB (layers 0-39) /
13.7 MB (40-47); 500 GB/s; 95% of routed experts in VRAM.

### Unique experts in a window (measured)

`<ref>/long/s0/ffn_moe_topk-L.bin`, 2698 prompt positions, every window of T consecutive positions
from position 64 on, all 48 layers (the `long-f32` dump gives the same to 0.05):

| T | unique experts a layer (mean) | of 8T | least / most shared layer | expert bytes a pass (mean / p90) |
|---|---|---|---|---|
| 1 | 8.00 | 100% | 8.0 / 8.0 | 4.14 GB |
| 2 | 13.74 | 86% | 11.2 / 15.4 | 7.09 / 7.64 GB |
| 3 | 18.70 | 78% | 13.6 / 22.2 | 9.64 / 10.52 GB |
| 4 | 23.15 | 72% | 15.6 / 28.4 | 11.93 / 13.08 GB |
| 5 | 27.28 | 68% | 17.4 / 34.5 | 14.04 / 15.43 GB |
| 6 | 31.11 | 65% | 19.0 / 40.2 | 16.01 / 17.64 GB |
| 7 | 34.74 | 62% | 20.5 / 45.8 | 17.86 / 19.74 GB |
| 8 | 38.18 | 60% | 22.0 / 51.1 | 19.63 / 21.68 GB |
| 9 | 41.46 | 58% | 23.3 / 56.2 | 21.31 / 23.48 GB |

Two neighbouring tokens share 2.26 of their 8 experts on average (0.6 in layer 2, 4.8 in layer 42). Routing is
close to independent from token to token: an 8-row window reads 4.7x the experts of one token. One document, read as
a prompt; generated text may differ a little.

### A verify pass of T rows

Bandwidth alone gives (5.32 + 4.14) / 0.5 = 18.9 ms for one token against the measured 30, so 11 ms are something
else. Three readings of that remainder, all fitted to 30 ms at T = 1:

- **A** (optimistic): the remainder is fixed per pass. t = 11.1 + (5.32 + E(T)) / 0.5 ms.
- **M** (middle): the 5% of experts outside VRAM are read at RAM speed (~80 GB/s), the rest of the remainder is
  fixed. t = 19.5 + 2.525 E(T) ms.
- **B** (pessimistic): everything scales with bytes (an effective 315 GB/s). t = (5.32 + E(T)) / 0.315 ms.

| T | A | M | B |
|---|---|---|---|
| 2 | 35.9 | 37.5 | 39.4 |
| 3 | 41.0 | 43.9 | 47.4 |
| 4 | 45.6 | 49.7 | 54.7 |
| 5 | 49.8 | 55.0 | 61.4 |
| 6 | 53.7 | 60.0 | 67.6 |
| 7 | 57.4 | 64.7 | 73.5 |
| 8 | 61.0 | 69.1 | 79.1 |

None of these is measured. ds4's own measured shape on an RTX 5070 (`kShape` in draft_policy.cpp: 1.35 / 2.05 / 3.6
of a token at T = 2 / 4 / 8) is steeper than all three (A 1.20 / 1.52 / 2.03, B 1.31 / 1.82 / 2.64). **Measuring
t(T) for T = 2..8 on the engine needs no drafter** (feed known text in windows with the head on every row) and
should come first.

### A draft round (one block of 8 rows + 7 head steps + injecting the kept rows)

| what | BF16 | Q8_0 |
|---|---|---|
| the 5 layers (q, k, v, o, FFN), read once for the 8 rows | 3.21 GB | 1.70 GB |
| the base head: the target's `output` (Q6_K) for rows 1-7 | 0.76 GB | 0.76 GB |
| 7 head steps, as the PR does them (`output` again each step) | 5.44 GB | 5.38 GB |
| 7 head steps, folded (`joint_proj` + `hidden_out` + M) | 0.84 GB | 0.45 GB |
| ctx K / V of the kept rows | 0.09 GB | 0.05 GB |
| **a round, folded** | **4.90 GB = 9.8 ms** | **2.96 GB = 5.9 ms** |
| a round, as the PR | 9.50 GB = 19.0 ms | 7.90 GB = 15.8 ms |

Plus launches and seven sequential head steps: take **7 ms (Q8_0, folded)** and **11 ms (BF16, folded)**. BF16 as
the engine loads it today (F32 in VRAM) would be ~16 ms. VRAM on the last GPU: 1.83 GB (Q8_0) + M 0.1 GB + caches
0.15 GB, i.e. about 150 of its 13.7 MB experts (0.8% of the model's) go to RAM.

### Break-even

A round with a window of T rows (T-1 drafts) costs t_draft + t_verify(T) and commits 1 + (accepted) tokens. With an
acceptance rate a per draft, independent by position, that is (1 - a^T) / (1 - a) tokens. Break-even a against 30 ms
a token:

| T | A, 7 ms | M, 7 ms | B, 7 ms | A, 11 ms | M, 11 ms | B, 11 ms | A, PR's head (20 ms) |
|---|---|---|---|---|---|---|---|
| 2 | 0.43 | 0.48 | 0.55 | 0.56 | 0.62 | 0.68 | 0.86 |
| 3 | 0.42 | 0.47 | 0.53 | 0.49 | 0.54 | 0.59 | 0.63 |
| 4 | 0.45 | 0.50 | 0.56 | 0.50 | 0.55 | 0.60 | 0.60 |
| 5 | 0.49 | 0.54 | 0.59 | 0.53 | 0.57 | 0.63 | 0.61 |
| 6 | 0.52 | 0.57 | 0.62 | 0.55 | 0.60 | 0.65 | 0.62 |
| 7 | 0.54 | 0.59 | 0.64 | 0.57 | 0.62 | 0.67 | 0.63 |
| 8 | 0.56 | 0.61 | 0.66 | 0.59 | 0.64 | 0.68 | 0.64 |

Expected speed against no drafting (Q8_0 folded, 7 ms), best window in brackets:

| acceptance | A | M | B |
|---|---|---|---|
| 0.5 | 1.09x (T 3) | 1.03x (T 3) | 0.97x (T 2) |
| 0.6 | 1.24x (T 4) | 1.16x (T 3) | 1.08x (T 3) |
| 0.7 | 1.46x (T 5) | 1.34x (T 5) | 1.23x (T 4) |
| 0.8 | 1.84x (T 7) | 1.65x (T 6-7) | 1.48x (T 6) |
| 0.9 | 2.51x (T 8) | 2.25x (T 8) | 1.98x (T 8) |

### Verdict

- Drafting pays only if at least about **half to two thirds of the drafts stand** (0.47-0.56 at the best window,
  depending on which reading of the 11 ms is right). Below that it is a loss at every window.
- The gain is smaller than on DeepSeek for the same acceptance, because Inkling routes 8 of 384 experts with little
  overlap between neighbours: at ds4's 77% acceptance on this PC (which gave it 1.62x) the model says about
  **1.3-1.6x** here, with windows of **5-6 rows**, not 8.
- A full 8-row window is right only above ~0.8 acceptance. At 0.6-0.7 the best window is **3-5 rows**; a fixed
  8-row window at 0.6 is slower than no drafting under M and B. So the window must be sized from measured cost
  (DraftPolicy) from the first version, with no confidence head to help.
- The fold and the Q8_0 copy are not optional: with the PR's head (20 ms a round) break-even rises to 0.60-0.64 and
  the gain at 0.7 falls to about 1.1x.
- The two numbers that decide it are unmeasured: the acceptance rate (parity plan, step 5) and t(T) on the engine
  (no drafter needed). Measure both before writing the drafter's kernels. If acceptance on the long prompt is under
  0.6, stop.

## Measured (2026-11-20, the reference run)

llama.cpp PR #32218's head (842c10f61) in a worktree (`<llama.cpp>-dspark`, built in the
ROCm container), with `tools/ik2/ref/inkling-dspark-eval.cpp` (`examples/inkling-dspark-eval` there),
`dspark-eval-worktree.patch` (env-switched variants; the extra tap candidates) and `run_dspark_eval.sh`. Our GGUFs
load in the PR's loader as they are. The PR head's target gives the reference's greedy tokens after the long prompt
(`13 198 12`), so its older Inkling 2 code is good enough for this.

Method: the target (UD-Q2_K_XL, auto-fit, 6.6-10.5 tok/s here) reads a prompt and decodes 300 greedy tokens once,
every position's taps kept (`<ref>/dspark-eval/<prompt>/`: `tokens.txt`, `tap-slot<id>.bin` F32
[positions][4608] - slots 5..41 and 48 are the PR's taps; `drafts-<variant>.txt`; `results.jsonl` - inputs for the
engine's drafter-only test). A greedy target does not depend on the drafter, so each drafter variant then runs alone:
a block after **every** generated position (293 a prompt), its drafts compared with the target's next 7 tokens; and
the real rounds (the next block after the kept drafts) replayed from that.

Prompts: `prose` = the 2698-token long prompt continued; `code` = a chat request for a Python module (the answer
starts with thinking, then code); `chat` = a chat question at the default thinking effort 0.9 (300 tokens of terse
thinking notes - the hardest text for a drafter).

**Acceptance by draft position** (kept / reached, a block after every position, BF16):

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | kept of 7 (every position) | accepted a real round (7 offered) |
|---|---|---|---|---|---|---|---|---|---|
| prose | 0.86 | 0.84 | 0.86 | 0.87 | 0.86 | 0.80 | 0.81 | 3.90 | 3.11 (72 rounds, 224 of 300 tokens drafted) |
| code | 0.84 | 0.80 | 0.76 | 0.81 | 0.81 | 0.85 | 0.84 | 3.27 | 2.78 (78 rounds, 217) |
| chat (thinking) | 0.61 | 0.58 | 0.45 | 0.49 | 0.48 | 0.45 | 0.20 | 1.26 | 1.19 (134 rounds, 159) |

A real round starts where the last one missed, a harder place than average: its first draft stands 0.75 / 0.79 / 0.60
of the time. Acceptance hardly falls with the position on prose and code (the head's recurrence works).

**Q8_0 = BF16**: 3.89 / 3.24 / 1.28 kept of 7, the same rounds and accepted counts (224 / 217 / 159). Use Q8_0.

**Variants** (mean kept of 7, prose / code / chat; base 3.90 / 3.27 / 1.26):

| variant | prose | code | chat | verdict |
|---|---|---|---|---|
| tap 48 = `l_out-47` raw | 3.59 | 3.12 | 1.02 | worse |
| tap 48 = the head's mix without the norm | 3.60 | 3.04 | 1.05 | worse: **the PR's `result_norm` is right** |
| taps 5-41 = the layer's mixed Q input | 3.73 | 2.95 | 1.21 | worse: **raw streams are right** |
| RoPE adjacent pairs (NORM) | 0.32 | 0.69 | 0.30 | collapses: **half-split (NEOX) is right** |
| window 1024 + 0 | 3.90 | 3.27 | 1.26 | identical counts |
| a mask row per block row | 3.89 | 3.27 | 1.26 | one draft of 2051 differs |

The "+1" and the shared mask row cannot be told apart by acceptance even at position 2700+; take the PR's.

**The draft's top probability is a usable confidence** (all prompts and positions, kept / reached by `max softmax`):
0.1-0.2: 0.16; 0.2-0.3: 0.31; 0.3-0.4: 0.28; 0.4-0.5: 0.42; 0.5-0.6: 0.49; 0.6-0.7: 0.63; 0.7-0.8: 0.71; 0.8-0.9:
0.79; 0.9-1.0: 0.95 (1777 of 3190 drafts). It is close to calibrated as it is.

**Speed against no drafting**, the cost model above over the replayed rounds (7 ms a draft round, 30 ms a token;
A / M / B = the three readings of a window's cost):

| window rule | prose | code | chat |
|---|---|---|---|
| always 7 drafts | 1.81 / 1.62 / 1.43 | 1.67 / 1.49 / 1.32 | 0.96 / 0.86 / 0.76 |
| always 4 | 1.78 / 1.63 / 1.48 | 1.71 / 1.57 / 1.42 | 1.15 / 1.05 / 0.95 |
| always 2 | 1.58 / 1.49 / 1.39 | 1.54 / 1.45 / 1.35 | 1.22 / 1.15 / 1.08 |
| while the product of top probabilities >= 0.5 (ds4's `dspark_cum`) | 1.92 / 1.81 / 1.68 | 1.90 / 1.76 / 1.61 | 1.22 / 1.16 / 1.10 |
| ... >= 0.3 | 1.96 / 1.81 / 1.65 | 1.85 / 1.68 / 1.52 | 1.25 / 1.18 / 1.10 |

**Go.** With the window cut on the product of the drafts' top probabilities at 0.5 (2.5-3.5 drafts verified a round
on prose and code, 1.5 on thinking text) the model gives about **1.6-1.9x on prose and code and 1.1-1.2x on terse
thinking text**, and never a loss; a fixed 7-draft window loses on the thinking text. Still unmeasured: t(T) on our
engine, which picks between A, M and B.

llama.cpp's own drafter costs 91 ms a block here (83 ms Q8_0): the PR's head as written is slower than three target
tokens of ours, which is why the fold and a device-side head loop matter. Injecting 2998 positions took 1.7 s.

## In the engine (2026-11-20: wired, measured)

`maya-ik2` drafts by default (`--dspark on|off|none`, maya-ds4's, and `idle` for tests; `--dspark-index DIR`, `--dspark-cum F`; a request's
`spec=0|1` and `dspark_cum=F` keys). The module loads on the last GPU with the engine's own output matrix
(`DsparkOptions::output_dev`); the engine keeps 3.09 GB of that GPU for it (`Options::dspark_reserve_gb`: what the
drafter reports at a 32k context - the setup's and the Models page's figure too, serve/models.py's `IK2_DSPARK_GB`),
so about 250 of its experts go to RAM.

**Taps.** Every row of every step leaves its 8 taps on the last GPU as F32 [rows][8][4608]: a layer whose output is
a tap (4, 10, ..., 40) publishes its rows to host memory the last GPU maps (the last GPU's own tap layer: a copy), the
head adds its input (the fused mix + norm, `result_norm`), one gather interleaves them. Two buffers by turns: the
drafter reads the last step's on its own stream while the next step writes the other. The order is by construction, not by
timing: `Dspark::feed_dev` records an event on the drafter's stream after its reads of a buffer, and the step that
writes that buffer again waits on it on its own stream first (no measurable cost: `--sweep IK2_HIP_TAP_EVENT=1,0`,
two rounds in ABBA order, prose 64.7 / 65.7 tok/s, code 55.6 / 56.0, thinking 38.4 / 38.3, within the runs' noise of
3 - 10 tok/s). They are fed
(`Dspark::feed_dev`) when the rows are kept - at the next draft, step or rewind - so `Model::keep` only shortens what
is pending. The prompt path does the same a chunk at a time (the taps through pinned host buffers, a chunk in flight
each; the head's mix + norm for every row), fed by the last GPU's stage. A rewind rolls the drafter back to the
context's end (`Dspark::rollback`), its rings as long as the engine's.

**A round** (`draft_round`, ik2_hip_main.cpp; the decode loop and `--serve` share it): `Model::draft(tok)` ->
`Model::window([tok, d_1 .. d_k])` -> the verdicts -> `Model::keep(m + 1)` -> out: `d_1 .. d_m`, next.

- The window: k maximizes (1 + sum_{i<=k} prod_{j<=i} p_j) / (t(k + 1) + t_draft), p the drafter's probabilities, under
  the drafter's own cut on their running product (0.5). t(T) = a + b (T - 1), a and b fitted as it decodes (a from
  the passes of one row, b from the windows; 30 and 7 ms to start), t_draft the draft's measured time. k = 0: a plain
  step.
- Greedy: a draft stands while it is the row's argmax (after the request's penalties). Sampled: ds4's rule for argmax
  drafts - it stands with probability p(d), p the request's own chain on the target's row, else the row's token is
  drawn from p without d. Never drawn drafts.
- Rest: a drafted round gains (kept drafts) x a - (t_draft + k b) ms; when its average over the last drafted rounds is
  under a tenth of a step, no drafts for 8 rounds (doubling to 32 while it keeps not paying, from 8 again with every
  request). The taps still flow. Rests are short on purpose: where drafts are break-even a rest gains little, and a
  long one sleeps through the code after a thinking block (measured through the server with rests up to 128 rounds).
- An end token among the kept drafts: the context is cut back to before it (`keep`), as an undrafted answer leaves it.

**The window's cost** (2.7k context, ms a pass; `--window-test`, `IK2_HIP_PHASES=1` for the parts):

| T | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| before | 30.4 | 43.1 | 50.7 | 58.4 | 68.1 | 76.4 | 83.8 | 93.5 | 100.0 |
| now | 29.9 | 38.9 | 45.5 | 51.3 | 59.2 | 65.5 | 71.2 | 77.4 | 83.7 |
| now, `--dspark none` (no taps) | 29.1 | 38.8 | 45.5 | 50.8 | 59.1 | 65.9 | 71.6 | 77.5 | 83.0 |

| part (ms a pass, with the phase marks) | T 1 before | T 1 now | T 2 before | T 2 now | T 4 before | T 4 now | T 8 before | T 8 now |
|---|---|---|---|---|---|---|---|---|
| mix + rms | 1.45 | 1.45 | 1.45 | 1.45 | 1.43 | 1.44 | 1.43 | 1.44 |
| input projections | 6.32 | 6.23 | 6.56 | 6.56 | 7.32 | 7.31 | 9.87 | 9.88 |
| convs + clips + cache rows | 0.60 | 0.61 | 1.82 | 0.64 | 1.85 | 0.63 | 1.93 | 0.71 |
| attention | 4.90 | 4.21 | 6.95 | 4.67 | 11.26 | 5.34 | 20.02 | 6.45 |
| gate + output + residual | 2.70 | 2.62 | 3.31 | 2.94 | 3.62 | 3.27 | 4.58 | 4.23 |
| mix + rms + router + publish + plan / grouping | 2.91 | 2.93 | 3.80 | 4.09 | 4.76 | 5.10 | 6.56 | 6.95 |
| GPU experts + sum | 8.45 | 8.45 | 14.61 | 14.62 | 23.35 | 23.49 | 41.14 | 41.43 |
| lane wait | 2.60 | 2.76 | 2.90 | 2.87 | 2.75 | 2.94 | 3.38 | 3.38 |
| lane sum + conv + residual | 0.35 | 0.35 | 0.68 | 0.36 | 0.68 | 0.37 | 0.71 | 0.38 |
| between layers / hand-offs / head / host | 2.92 | 2.82 | 2.64 | 2.34 | 3.00 | 2.45 | 4.01 | 2.95 |
| the pass | 33.2 | 32.4 | 44.7 | 40.6 | 60.0 | 52.3 | 93.6 | 77.8 |

(The grouping moved from the experts' part into the router's between the two columns.) What changed: the fused row
kernels (convs, clips, residuals: -2.5 ms a window), and the attention's scores - a thread took one (query, key) dot
and read the query from memory for each; it now takes a query and 4 keys (up to four rows) or 8, the query's value
read once for them: 2.2 -> 0.33 ms a row, and a token's own attention 4.9 -> 4.2 ms. The combine's exp a tile and a
thread was not the cost (measured: no change). The experts are the floor: 4.3 more unique experts a layer a row at
the cards' bandwidth, 4.7 ms.

**Exactness.** With the experts where they are (`--window-test` holds the cache still for it) a window's 9 rows are
the 9 token steps' logits to the bit, and so is a window cut back and read on. With the cache moving experts between
two passes the same token steps differ from themselves (0 of 9 rows the same, least cosine 0.9958): an expert's sum
on the CPU lane is not its sum on the GPU. `--ab --no-expert-cache`: the three prompts' 256 drafted tokens are the undrafted
ones. With the cache on a drafted answer leaves the undrafted one at a close call (prose token 21, the two best logits
0.057 apart; code 98 at 0.059; thinking 7 at 0.027) - as a second undrafted load does (code from token 165, thinking
from 7): docs/INKLING2_AMD.md, Status.

**Speed** (greedy, 256 tokens, a fresh load each side; tok/s):

| prompt | without drafts | with drafts | | tokens a pass | drafts kept | k = 0 .. 7 chosen |
|---|---|---|---|---|---|---|
| prose: the 2.7k-token prompt continued | 33.8 | 62.3 (66.0 another load) | 1.84x (1.95x) | 3.92 (4.18) | 190 of 207 | 9 13 14 5 4 0 5 15 |
| code: a chat request for a Python module | 36.3 | 53.3 (57.3) | 1.47x (1.58x) | 3.04 (3.49) | 171 of 243 | 17 15 12 6 10 9 4 11 |
| thinking: a chat question, effort 0.9 | 37.3 | 36.1 (35.3) | 0.97x (0.95x) | 1.26 (1.32) | 53 of 116 | 145 25 15 12 3 0 1 1 |

The brackets are a second fresh load (an earlier build of the same windows): the answers differ from load to load at
near-ties (the cache's moves), and with them the drafts kept. The thinking prompt run 13 times in one load, the cache
warm: 39.2 tok/s with drafts, 39.1 - 39.3 without - break-even, as the projection said; its fresh-load loss is the CPU
lane's share (8 - 13 % against 7 %: the drafter's 3.1 GB on the last GPU, and fewer cache moves a token). A draft costs
8.0 - 8.5 ms a round, a pass 26 - 30 ms + 7 - 8 a row as fitted while decoding (6.7 a row in `--window-test`).

Through serve/server.py (my_e2e.py: a 2.7k-token document and two follow-ups, temperature 1, thinking on; sampled,
so the answers differ): 38.2 / 40.5 / 45.7 tok/s after the first token with drafts, 32.0 / 33.8 / 32.1 without
(`--dspark none`, `off` then); drafts kept 194 of 363, 256 of 435, 297 of 397; tokens a pass 1.96 / 2.10 / 2.23.

Left: the draft (8 ms on the last GPU while the others idle) could overlap the first GPU's part of the pass; the
drafter's VRAM; a sampled window's logits go to the host a row (0.8 MB) for the verdict.

## Reference parity plan (steps 1-3 and 5 done as above; the tensor dumps of step 4 are not)

1. **A checkout of the PR's head, beside the existing one** (the saved diff does not apply to master b94959a64:
   8 of its files fail `git apply --check`; master has since split DSpark / DFlash2 into classes of their own and
   moved the graph to `src/models/dspark.cpp`):
   ```sh
   cd <llama.cpp>
   git fetch origin pull/32218/head:pr-32218
   git worktree add <llama.cpp>-dspark pr-32218
   ```
   Check first: that head's `src/models/inkling2.cpp` against master's (the PR predates the base model's merge and
   its follow-up #32429). If the target differs, the target-side dumps will not match `<ref>/long`;
   port the follow-up into the worktree or accept a new target reference from it. Also check there
   `llama_model_rope_type` for `LLM_ARCH_DFLASH` (NEOX expected) and the guard around `conf` in `draft()`.
2. **Build** in the ROCm container, as `build-hip` was (its cache: Ninja, Release, `GGML_HIP=ON`, `GGML_HIP_GRAPHS=ON`,
   `GGML_HIP_MMQ_MFMA=ON`, `GGML_HIP_NO_VMM=ON`, `AMDGPU_TARGETS=gfx1201`, `BUILD_SHARED_LIBS=ON`,
   `LLAMA_BUILD_EXAMPLES=ON`, `CMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang`):
   ```sh
   podman run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt seccomp=unconfined \
     -v <llama.cpp>-dspark:/llama:z --entrypoint /bin/bash <a ROCm image> -c '
     cd /llama && cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release \
       -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON -DAMDGPU_TARGETS=gfx1201 \
       -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_EXAMPLES=ON -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang &&
     cmake --build build-hip --target llama-inkling-dspark-dump llama-speculative-simple -j 16'
   ```
3. **Load check**: `llama-speculative-simple -m <target shard 1> -md
   /models/Inkling-2-Small-MTP/Inkling-2-Small-DSpark-BF16.gguf --spec-type draft-dspark -n 8 -p hi` (the PR may want
   `draft-dflash`; its `common_speculative` picks the driver from the draft's metadata). This is the first time our
   GGUF meets the PR's loader; a missing key or tensor shows here. If it balks, the fallback is the PR's own
   `convert_hf_to_gguf.py --dspark` on the checkpoint folder (it needs the model's tokenizer files beside
   `mtp.safetensors`), compared tensor by tensor with ours.
4. **The dump tool**: `tools/ik2/ref/inkling-dspark-dump.cpp`, built as `examples/inkling-dspark-dump` in the
   worktree: `examples/speculative-simple/speculative-simple.cpp` with `inkling-dump.cpp`'s `dump_cb` on both
   contexts (the draft context's `cb_eval` set in the `common_params` handed to
   `common_speculative_init_from_params`; its tensors written under `dft/`). Steps: the prompt as one batch
   (step 0: `common_speculative_process` injects every row), then N rounds of draft -> verify -> accept, greedy,
   `--spec-draft-n-max 7`. Saved per step, in the existing `index.jsonl` format:
   - target: `l_out-{4,10,16,22,28,34,40}`, `result_norm` for **every** row (take it before `out_ids`: name
     `t_layer_inp[48]` in the PR's inkling2.cpp with a `cb`), `ffn_moe_topk-L`, `result_output` of the verify rows;
   - injection (`dft/`): `dspark_context-L` (c_l), `dspark_f_rnn` (f) - the PR names these; add `cb`s for the ctx K
     after norm + rope and V (`dspark_kctx-L`, `dspark_vctx-L`);
   - block: `dspark_layer-L` (x after each layer, 8 rows), the final normed `h` (name it `dspark_h`), the base logits
     (`dspark_base`, rows 1-7), `dspark_chain_state` (s_i, padded to 4608), and per head step add `cb`s in
     `build_inkling2_head` for `z` (`dspark_joint-i`), `corr`, the capped `lg_i`; layer 0 and 2: `Qcur` / `Kcur`
     after norm + rope and `kqv_out`;
   - `rounds.jsonl`: per round n, y, the drafts, the target's picks, accepted.
   ```sh
   INK_DUMP_DIR=/ref/dspark-long INK_DUMP_ROUNDS=64 INK_DUMP_FILTER='^(l_out-(4|10|16|22|28|34|40)$|result_norm|layer_inp-48|ffn_moe_topk-|result_output|dspark_|Qcur-(0|2)$|Kcur-(0|2)$|kqv_out-(0|2)$)' \
   build-hip/bin/llama-inkling-dspark-dump -m /models/Inkling-2-Small-GGUF/UD-Q2_K_XL/Inkling-2-Small-UD-Q2_K_XL-00001-of-00006.gguf \
     -md /models/Inkling-2-Small-MTP/Inkling-2-Small-DSpark-BF16.gguf --spec-type draft-dspark --spec-draft-n-max 7 \
     -fa on -c 4096 -b 4096 -ub 4096 -t 16 --temp 0 -n 512 -f /prompt/prompt-long.txt
   ```
   (wrapped like `tools/ik2/ref/run_dump.sh`, a second script `run_dspark_dump.sh`; a short-prompt run first.)
5. **The acceptance rate**: the same binary without the filter (or `llama-speculative-simple`), 512 greedy tokens
   after the long prompt and after 2-3 chat / code prompts: `n_drafted`, `n_accept`, and from `rounds.jsonl` the
   kept / reached count **by draft position 1..7** - the table the break-even above needs. Then once with the Q8_0
   GGUF: how much acceptance the rounding costs. Then the tap variants of Open question 1 if the rate is poor.
6. **Our side**: a drafter-only test first (the reference's dumped taps in, `dspark_context`, `dspark_layer`, `dspark_h`,
   `lg_i` and the drafts out, compared with `compare_dumps.py`: cosine >= 0.999 a row on BF16, the same drafts), then
   the engine's own taps against the dumped ones, then end to end: the same drafts and the same accepted counts round
   by round on the long prompt.

## Open questions

Settled by the reference run (Measured): tap 48 is `result_norm`; taps 5-41 are the raw streams; RoPE pairs are
half-split; Q8_0 costs no acceptance; the top probability works as a confidence. The window "+1" / shared mask row
makes no measurable difference. The PR's `conf` read in the greedy path is unguarded (`common/speculative.cpp`,
`if (is_dspark)`: with `--spec-draft-p-min` > 0 it compares the markov state's first value with the threshold); the
default p_min is 0, so it is latent - do not copy it.

1. **t(T) on our engine** for T = 2..8, and whether the prompt path's chunk machinery has a fixed cost that the
   token path does not (a window may want the token path's kernels over T rows instead). It decides between 1.6x and
   1.9x.
2. Acceptance on more text: 300 tokens a kind is a first figure; answers after the thinking, tool calls and sampled
   requests are not covered.
3. The exactness of a window pass against token-by-token decoding when experts are split between VRAM and the CPU
   lane (ds4 documents that answers can differ there).
4. Tensor-level parity dumps of the drafter (step 4 of the plan) are still to be made; until then the engine's
   drafter-only test compares drafts (`drafts-bf16.txt`: position, kept, 7 x `token:top probability`) from the saved
   taps, not tensors.
