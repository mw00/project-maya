# Inkling 2 Small on AMD: `maya-ik2`'s AMD build

`maya-ik2`, the engine of [INKLING2.md](INKLING2.md) (the model, the file format, the reference, the server side: read
that first), on AMD (HIP). Its CUDA build's kernels are NVIDIA's own (inline PTX: `mma.sync`, `wgmma`, `cp.async`;
NVFP4), so on AMD the engine's model runs from `src/ik2/hip`, on maya-ds4's ggml block-type kernels
(`src/ds4/ds4_iq.cu`, `ds4_iq_cpu.cpp`, `ds4_gguf_cpu.cpp`, `MAYA_DS4_IQ_ONLY`; built the way `maya-hy4` is), from a
pack of unsloth's UD-Q2_K_XL, with the model's DSpark drafter ([INKLING2_DSPARK.md](INKLING2_DSPARK.md)). CMake
builds `src/ik2/hip` as `maya-ik2` with `STRATA_ENABLE_HIP`; the CUDA build (`src/ik2/*.cu`) is unchanged.

One engine to the server, the web app and the setup: family `ik2`, the chat template (thinkingmachines' own,
`tools/ik2/chat_template.jinja`), the answer's parser (`InklingRewriter`) and the tokenizer of INKLING2.md, and the
CUDA build's flags where both builds have the setting (`--model`, `--gpus N`, `--chunk`, `--max-context`; maya-ds4's
names for the AMD build's other settings). Its INFO line says `engine=ik2 engine_kind=ik2-hip`. "Serving" below has
what is the AMD build's own: its run config, its command line and the line protocol.

Inkling-2-Mini (INKLING2.md section 8, the CUDA build's experimental preview) does not run on this build yet: its
core layers run three times, each pass after the first let in through the four messages and their gates, and
`src/ik2/hip` has no loop. The rest of a Mini layer is a Small layer's, and unsloth's Mini GGUFs come in quants this
build's kernels read (UD-IQ2_XXS to UD-Q4_K_M), so the loop is what it needs.

## Status (2026-11-20)

On v1.1.70's tree (with #1079's `maya-ik2`, before this build was moved into it), measured again: parity OK x4,
the rewind test's five cases OK, short decode through `--serve` 37.3 tok/s without drafts and 59.1 with them (256
tokens), `--ab` prose 33.8 -> 62.1, code 36.1 -> 59.2, thinking 37.0 -> 39.3 tok/s; through serve/server.py with the
official chat template, my_e2e's three turns 42.1 / 38.9 / 50.7 tok/s. The table below is the run before the merge
(commit f620cbc: speculative decoding wired).

After the measurement phase (2026-11-21: the VRAM plan measured, the measured values - "VRAM", "The measured values"
below), the same checks: parity OK x4, the rewind test's five cases OK, short decode through `--serve` 37.6 tok/s
without drafts and 58.3 with them (the 2.7k prompt continued: 65.4), `--ab` prose 33.9 -> 59.5, code 37.1 -> 58.3,
thinking 37.3 -> 38.7 tok/s - within the runs' noise of the numbers above (drafted prose ranges 59 - 73 between runs).
At 32K 10,604 of the 18,432 experts are in VRAM (10,565 before, when 0.3 GB a GPU was left free instead of the
600 MiB reserve); prompts read as before (2.7k tokens at 946 tok/s).

Measured on this PC; `--max-context 32768`, the pack's expert profile loads by itself, the drafter on by default.
(The flags as they are named now: the drafter not loaded is `--dspark none`, maya-ds4's name; it was `off` then.)

| | without drafts (`--dspark none`) | with the DSpark drafter (default) |
|---|---|---|
| Decode, short context (12-token prompt, 512 tokens) | 37.5 tok/s | - |
| Prose: the 2.7k-token reference prompt continued, 256 tokens | 33.8 tok/s | 62.3 - 66.0 tok/s |
| Code: a chat request for a Python module, 256 tokens | 36.3 tok/s | 53.3 - 57.3 tok/s |
| Thinking: a chat question at effort 0.9, 256 tokens of thinking notes | 37.3 tok/s (39.2 with the cache warm) | 35.3 - 36.1 tok/s (39.2 with the cache warm) |
| Chat turns through serve/server.py (my_e2e: document, follow-up, code; temperature 1) | 32.0 / 33.8 / 32.1 tok/s | 38.2 / 40.5 / 45.7 tok/s |
| Prompt, 2698 tokens | ~940 tok/s | the same (every row's taps fed to the drafter) |
| Parity with llama.cpp | PARITY OK, short and long, natural and forced picks | |
| DeepSeek (maya-ds4) here, for comparison | 35.6 tok/s | 57.5 |

Done:
- Load: layer split, the expert pack, RAM experts pinned.
- Host-free decode step: device picks, a mailbox CPU lane, mapped hand-off.
- Dynamic VRAM/RAM expert cache: decayed counts, one mover per GPU, `--no-expert-cache` turns it off.
- Batched prompt path: RAM experts streamed into VRAM.
- Prefix reuse with checkpointed rewind (`--rewind`); F16 KV.
- `--serve` (maya-hy4's protocol: "Serving" below).
- **One step for T = 1 .. 9 rows** (`Impl::layer_win`): a token's step is a window of one row. Row versions of the fused
  kernels (`kv_finish_rows`, `clip2_rows`, `conv_add_rows`, `mix_rms_rows`), the attention's tiles at fixed positions
  with a sequential combine, the histories written from one buffer to another (the old one is what `keep()` cuts
  back from). A window's rows are **the token steps' logits to the bit** while the experts stay where they are
  (`--window-test`: 9 of 9 rows; `--window-bits` shows it part by part).
- The CPU lane over a window (`cpu_experts_rows`): a RAM expert several rows picked is read once for them.
- The drafter's taps: every row of every step and of every prompt chunk, F32 [rows][8][4608] on the last GPU, fed to
  `Dspark::feed_dev` once the rows are kept.
- Drafting in the decode loop and in `--serve` (`draft_round` in ik2_hip_main.cpp): draft -> window -> keep -> feed;
  the window by expected tokens a millisecond; greedy and sampled acceptance; `--dspark on|off|none` (maya-ds4's; `idle`, tests only),
  `--dspark-index`, `--dspark-cum`; INFO `dspark=1`, DONE `drafts_acc` / `drafts_off`. docs/INKLING2_DSPARK.md
  ("In the engine") has the numbers.

t(T) at 2.7k context, ms a pass of T rows (`--window-test`, the drafter's taps on):

| T | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| before (commit 85ef75e) | 30.4 | 43.1 | 50.7 | 58.4 | 68.1 | 76.4 | 83.8 | 93.5 | 100.0 |
| now | 29.9 | 38.9 | 45.5 | 51.3 | 59.2 | 65.5 | 71.2 | 77.4 | 83.7 |
| now, `--dspark none` (no taps) | 29.1 | 38.8 | 45.5 | 50.8 | 59.1 | 65.9 | 71.6 | 77.5 | 83.0 |

A row more costs 6.7 ms (8.7 before). What is left of it (`IK2_HIP_PHASES=1`, a line a window size): the routed
experts 4.7 ms (4.3 more unique experts a layer, at the cards' memory bandwidth: the floor), the input projections'
extra columns 0.5, the router / publish / grouping 0.45, the attention 0.33 (2.2 before), the output projection 0.23,
the lane 0.2.

Two passes over the same tokens differ when the expert cache moved an expert between them (its sum on the CPU lane is
not its sum on the GPU: logits cosine 0.995-0.9999, windows or not). A drafted run moves experts at other moments
than an undrafted one (a poll a pass, not a token), so:
- with `--no-expert-cache` the drafted greedy answer is the undrafted one token for token (`--ab --no-expert-cache`: three prompts,
  256 tokens each);
- with the cache on, the two part where the row's two best logits are close: prose at token 21 (gap 0.057), code at
  98 (0.059) or 111 (0.13), thinking at 7 (0.027) or 112 (0.99) - two drafted loads each, the gaps from the undrafted
  run (`IK2_HIP_MARGIN_LOG=1`; the median gap is 3.4 / 4.8 / 1.7, and an expert on the other side moves a logit by
  up to 1.4);
- two undrafted loads part the same way once the cache's moves fall differently: the prose prompt (the first after
  the load) gave the same 256 tokens in four loads, the code prompt after it the same 165, the thinking prompt 7.

Not done:
1. Nothing overlaps the draft (8 ms on the last GPU, the others idle) with the pass that follows it.
2. The drafter takes 3.3 GiB of the last GPU at 32K (measured: "VRAM" below): about 300 of that
   GPU's experts live in RAM instead.
3. Sampled requests copy every window row's logits to the host for the verdict (0.8 MB a row).
4. Thinking text is break-even (half of 1.5 offered drafts kept): the drafter rests there, it does not gain.

The experts VRAM does not hold (2026-11-21, "The experts VRAM does not hold" below): `--mmap-experts on|off|auto`,
maya-ds4's flag - the most routed that fit copied into RAM, the others read in place from the model's files - so a
context whose experts RAM cannot hold loads: 1M on this PC now loads with the drafter (2.7k-token prompt continued at
47.2 tok/s), where it stopped at its start before.

Engine flags (where maya-ds4 has the same setting, its name, unit, default and behaviour: `--max-context` 32768,
`--reserve-mib` 600 - the setup writes DeepSeek's AMD figure, 3072 on a discrete card -, `--cpu-threads` 0 (a thread a
physical core; N: at most N), `--expert-profile` (`<pack>/expert-profile.bin`, read and written back; a pack's
`default.prof` from before is read in its place), `--profile-readonly`, `--profile-out` (this run's routing alone),
`--no-expert-cache`, `--mmap-experts on|off|auto` (auto here as off: "The experts VRAM does not hold"), `--parallel N`
(1 .. 8, "Several conversations at once"), `--dspark on|off|none` (off: loaded, a request drafts when it says `spec=1`; none: not loaded;
`idle`, tests only, has no maya-ds4 counterpart: loaded and fed, never asked)): `--pack --max-context (--ctx)
--layer-split --gpus --reserve-mib --vram-experts --expert-profile (none) --profile-readonly --profile-out
--cpu-threads --log-floor --prefill-chunk --try-chunks --repeat --rewind --short-prompt --no-expert-cache --mmap-experts --parallel --eos --serve --tokens
--tokens-file --n-predict --layers --dspark --dspark-index --dspark-cum --verbose`; tests `--parity [--parity-force-picks
--parity-force-only --parity-decode --parity-detail --parity-max] --rewind-test --multi-check --short-bench --window-test
[--window-bits] --prompts --ab --temperature --seed --dspark-rest0`. Environment: `IK2_HIP_PHASES` (device phase
table, a line a window size), `IK2_HIP_ROUND_LOG` (a line a drafting round), `IK2_HIP_TIMELINE`, `IK2_HIP_PROF`,
`IK2_HIP_OLD_STEP`, `IK2_HIP_SWAP_BATCH`, `IK2_HIP_SWAP_BURST_MS`, `IK2_HIP_VERIFY_LOAD`, `IK2_HIP_EXPERT_CHECK`,
`IK2_HIP_NO_WARMUP`, `IK2_HIP_VRAM_LOG` (a line a GPU at each stage of the load and after each prompt, decode and
request: free VRAM, the engine's own, the rest); the measured values' overrides ("The measured values"):
`IK2_HIP_CACHE_DECAY` / `_MARGIN` / `_BONUS` / `_PROMPT_ROWS`, `IK2_HIP_SWAP_WIN_ROWS`, `IK2_HIP_LANE_SPIN_US`,
`IK2_HIP_SNAP_EVERY`, `IK2_HIP_BURST_ROUNDS` / `_MOVES` / `_MIN`, `IK2_HIP_DRAFT_T0` / `_A0` / `_B0` / `_REST` /
`_REST_MAX` / `_WARM` / `_REST_GAIN` / `_RESET_GAIN` / `_EMA` / `_B_MIN`, and at the load `IK2_HIP_STAGE`,
`IK2_HIP_TS_MAX`, `IK2_HIP_S_FLOOR_MI`, `IK2_HIP_WARM_TOKENS` (tests: `IK2_HIP_STAGE_USE`, `IK2_HIP_TS_USE`,
`IK2_HIP_S_USE_MI`, `IK2_HIP_TAP_EVENT`; `--sweep VAR=a,b,..`, `--sweep-rounds N`); `IK2_HIP_ATTN_OLD` / `IK2_HIP_ATTN_TK` now reach only the step with a tap or a pick hook.

## Goal and targets

- Text chat (thinking, tool calls) from unsloth's `UD-Q2_K_XL` GGUF set, served by `serve/server.py` like Hy4.
- This PC: 4x Radeon AI PRO R9700 (gfx1201, 32 GiB, PCIe 5.0 x16 each), Threadripper 9960X (24 cores, 4-channel DDR5),
  125 GB RAM, NVMe. maya-ds4 runs DeepSeek-V4.1-Flash Maya-S here at ~35 tok/s with 57% of its experts in VRAM.
- **Correctness first**: per-layer parity with llama.cpp's merged Inkling 2 implementation (the reference below).
- **Decode target: >= 25 tok/s** (stretch 35) on this PC, prompts >= 300 tok/s. Ceiling estimate: ~5.3 GB of dense
  weights + ~4.1 GB of experts read per token; layer split over 4 cards at ~600 GB/s -> ~60 tok/s if every pick were
  in VRAM.
- Not in the first version: vision (mmproj), tensor parallel. The drafter came later (INKLING2_DSPARK.md).

## The files

`<models>/Inkling-2-Small-GGUF/UD-Q2_K_XL/Inkling-2-Small-UD-Q2_K_XL-0000{1..6}-of-00006.gguf`
(shard 1 holds the metadata and tokenizer only; 1092 tensors in shards 2-6). 190.9 GiB of tensors: routed experts
185.3 GiB, everything else 5.55 GiB.

### Constants (GGUF `inkling2.*`)

| key | value |
|---|---|
| block_count | 48 (no MTP layers in the GGUF) |
| embedding_length | 4608 |
| attention.head_count / head_count_kv / key_length / value_length | 48 / 8 / 128 / 128 (GQA 6) |
| attention.layer_norm_rms_epsilon | 1e-6 |
| expert_count / expert_used_count / expert_feed_forward_length | 384 / 8 / 2304 (no shared expert) |
| expert_weights_scale | 8.0 |
| final_logit_softcapping | 30.0 |
| attention.sliding_window | 1024; pattern: layers 5, 11, 17, 23, 29, 35, 41, 47 are global, the other 40 SWA |
| d_rel (attention.relative_length) / rel_extent / rel_extent_swa | 16 / 1024 / 1024 |
| shortconv_kernel (shortconv.l_cache) | 4 |
| log_scaling_n_floor / log_scaling_alpha | **8192** / 0.1 (the GGUF's stale 128000 fixed by the pack: Open questions) |
| qk_clip_threshold / rel_clip_threshold | 22.627417 / 8.0 |
| attn_res.block_size | 6 |
| activation.situ_beta / situ_linear_beta (mlp_gate_softcap / mlp_up_softcap) | 4.0 / 25.0 |
| vocab_size / unpadded_vocab_size | 201024 / 200058 (the tokenizer list holds the 200058; rows above are padding) |
| tokenizer | `gpt2` BPE, pre `gpt-4o`; bos = eos = 200006, no BOS added; EOG also `<\|content_model_end_sampling\|>` |

### Tensors (ggml dims, ne0 first)

Global: `token_embd` (4608, 201024) Q5_K; `token_embd_norm` (4608) F32; `output_res_query` (4608) F32;
`output_norm` (4608) F32; `output` (4608, 201024) Q6_K.

Every layer `blk.L.`:

| tensor | dims | type (L0-39 / where it differs) |
|---|---|---|
| attn_norm (SWA layers only) | 4608 | F32 |
| attn_q_inp_norm, attn_k_inp_norm, attn_v_inp_norm (global only) | 4608 | F32 |
| attn_res_query (L >= 1) | (4608) SWA, (4608, 3) global | F32 |
| attn_q | (4608, 6144) | Q6_K (Q8_0 on global layers) |
| attn_gate | (4608, 6144) | Q6_K / Q8_0 |
| attn_k, attn_v | (4608, 1024) | Q8_0 |
| attn_k_conv_proj, attn_v_conv_proj (global only) | (4608, 32) | F32 |
| attn_rel | (4608, 768) | Q8_0 (48 heads x 16) |
| attn_rel_bias | (16, 1024) | F32 |
| attn_q_clip | (128, 48) | F32 (per head, per channel scale) |
| attn_rel_clip | (16, 48) | F32 |
| attn_sink_proj | (4608, 48) | BF16 |
| attn_output | (6144, 4608) | Q6_K / Q8_0 |
| attn_k_conv, attn_v_conv | (4, 1024) | F32 |
| attn_post_conv, ffn_post_conv | (4, 4608) | F32 |
| ffn_res_query | 4608 | F32 |
| ffn_norm | 4608 | F32 |
| ffn_gate_inp | (4608, 384) | F32 |
| exp_probs_b.bias | 384 | F32 (selection bias) |
| ffn_gate_up_exps | (4608, 4608, 384): rows 0..2303 gate, 2304..4607 up | IQ2_XS (L40-47: IQ3_XXS) |
| ffn_down_exps | (2304, 4608, 384) | IQ3_XXS (L40-47: IQ4_XS) |

One expert: 10.2 MB (L0-39), 13.7 MB (L40-47).

## The forward pass (one token at position p; the reference is `src/models/inkling2.cpp` in llama.cpp)

Reference checkout: `<llama.cpp>` (master b94959a64, Inkling 2 = PR #32217 + #32429).
`rms(x)` = x / sqrt(mean(x^2) + 1e-6) with no gain; `rmsg(x, g)` = rms(x) * g.

1. `x = rmsg(token_embd[id], token_embd_norm)`. `blocks = []` (a per-token list of saved residual streams, not state
   across tokens).
2. Per layer L (`swa = L % 6 != 5`):
   - `mix(blocks, cur, Q)` (attention over depth): states S = {rms(b) for b in blocks} + {rms(cur)}; for each query
     column q of Q (1 or 3 of them): w = softmax_s(S_s . q / sqrt(4608)); out_q = sum_s w_s S_s. (The order of the
     states does not matter.)
   - Attention inputs: L = 0: in_q = in_k = in_v = cur. L >= 1: m = mix(blocks, cur, attn_res_query); SWA: all three =
     m[0]; global: in_q, in_k, in_v = m[0], m[1], m[2]. Then SWA: all three = rmsg(., attn_norm); global: rmsg with
     attn_q_inp_norm / attn_k_inp_norm / attn_v_inp_norm.
   - `Q = attn_q @ in_q` (48 x 128), `K = conv_k(attn_k @ in_k)`, `V = conv_v(attn_v @ in_v)` (8 x 128 each),
     `R = attn_rel @ in_q` (48 x 16), `t = attn_sink_proj @ in_q` (48), `g = attn_gate @ in_q` (6144).
   - Short convolution `conv(y)` with state (the last 3 inputs y of this stream, per channel), weights w (4, C):
     c[ch] = sum_{k=0..3} w[k][ch] * y_{p-3+k}[ch] (k=3 is the current input). Global layers' K and V add dynamic taps:
     d = conv_proj @ in_k (or in_v), 32 = 4 lags x 8 heads, d[lag][h]; c[h*128+i] += sum_{lag=0..3} d[lag][h] *
     y_{p-lag}[h*128+i]. Output: (y_p + c) / sqrt(n*4 + 1), n = 2 with dynamic taps, else 1. The SWA layers' K and V
     use conv without dynamic taps too (attn_k_conv / attn_v_conv exist on every layer). Four streams a layer:
     K (1024 ch), V (1024), attention post (4608), FFN post (4608) - in llama.cpp's recurrent state at channel offsets
     0, 1024, 2048, 6656.
   - Clipping: `Q_h = clip(Q_h * attn_q_clip[:, h], 22.627417)`, `K_h = clip(K_h, 22.627417)`,
     `R_h = clip(R_h * attn_rel_clip[:, h], 8)`, where clip(v, T) = v * min(1, T / (||v||_2 + 1e-6)).
     `V_h = rms(V_h)` (value norm, no gain, eps 1e-6).
   - Global layers only: Q and R times tau(p) = 1 + 0.1 * log(max(1, (p + 1) / floor)), floor = log_scaling_n_floor.
   - Scores for query head h (kv head h / 6) and key at position j, distance d = p - j:
     s = Q_h . K_j / sqrt(128) + (d < 1024 ? R_h . attn_rel_bias[:, d] : 0) - t_h; SWA layers see d < 1024 only (the
     window: masked when p - j >= 1024); global layers see every j <= p. Softmax with a zero sink logit:
     a_j = exp(s_j) / (1 + sum_j exp(s_j)). out_h = sum_j a_j V_j.
   - `o = attn_output @ (out * sigmoid(g))`, `o = conv_post_attn(o)` (no dynamic taps).
   - Residual: if L % 6 == 0: blocks.append(cur_in) and cur = o (a new stream); else cur = cur_in + o.
   - FFN: h = rmsg(mix(blocks, cur, ffn_res_query), ffn_norm). Router: logits = ffn_gate_inp @ h (F32),
     prob = sigmoid(logits); picks = top8(prob + exp_probs_b); w = prob[picks] / max(sum, 6.1035e-5) * 8.0.
     Expert e: [gate | up] = gate_up_e @ h; a = 4 tanh(gate/4) * sigmoid(gate) * 25 tanh(up/25) (SITU);
     y = sum_k w_k down_{e_k} @ a_k. `y = conv_post_ffn(y)`; cur = cur + y.
3. Head: `h = rmsg(mix(blocks, cur, output_res_query), output_norm)`, logits = output @ h, logits = 30 tanh(logits/30),
   keep the first 200058.

## State for a conversation

- KV: SWA layers: a ring of the last 1024 positions (K and V after conv / clip / norm), 40 layers x 8 heads x 128 x 2;
  global layers: every position (8 layers). Distance-based bias needs the absolute positions of the cached keys.
- Conv histories: 3 rows x (1024 + 1024 + 4608 + 4608) channels a layer.
- At 32K context: global KV 8 x 32K x 2 x 1024 x 2 B (F16) = 1 GiB; SWA 40 x 1024 x 2 x 1024 x 2 B = 160 MiB.

## The engine (design)

- `src/ik2/hip/`, modeled on src/hy4: `ik2_hip_main.cpp` (args, `--serve` protocol from hy4_main.cpp),
  `ik2_hip_model.cu/.hpp` (load, place, forward), `ik2_hip_kernels.cu/.cuh` (Inkling-only kernels: mix, convs,
  clip/norm, rel-bias attention with sinks, router), `ik2_hip_lane.cpp` (CPU experts), a pack reader.
- Weights from the GGUF in place through a pack index (tools/ik2/pack_gguf.py, Hy4-style: GGUF names, stacked
  experts).
- **Layer split** over the GPUs (hidden-state hand-off between cards). Tensor parallel is out: the experts' 256-value
  blocks cannot split n_ff 2304 (maya-ds4 falls back the same way on this PC).
- **Experts**: per layer, the most-routed experts (by a profile, uint32 [48][384] routings, `--expert-profile` - Hy4's `--profile`) live
  in VRAM in ds4_iq's blob layout (gate | up | down, `tr[]` offsets, the fused GGUF gate_up maps to tr[1] = tr[0] +
  2304 * row_bytes); the rest are **copied into pinned RAM** (paging them from the GGUF on a miss is what made
  llama.cpp crawl here; since 2026-11-21 those RAM cannot hold are read in place - `--mmap-experts`) and computed on
  the CPU lane (24 threads). The engine counts routings and writes a profile
  (`--profile-out`), like Hy4. Budget: ~106 GiB of experts in VRAM (57%), ~79 GiB in RAM.
- Expert math on ds4_iq's kernels with a new activation `kActSitu` (below).
- Decode first (one token a step, the prompt fed token by token), then the batched prompt path, then speed (an
  asynchronous CPU lane like maya-ds4's mailbox, a dynamic expert cache).

### The SITU activation in ds4_iq

`ds4_iq.cuh`: a new activation id `kActSitu` beside `kActSwiglu` / `kActOai`, with two new `Args` fields
`situ_beta` (4.0) and `situ_lb` (25.0): `a = beta*tanh(g/beta)*sigmoid(g) * lb*tanh(u/lb)` (lb <= 0: up unchanged).
Every gate_up form the engine uses (token, window/set, tile, the prompt paths) and the CPU lane (`iq_cpu_gate_up`
epilogue / `native_gu_rows_split`) take it. Existing activations: bit-for-bit unchanged.

## The reference (llama.cpp tensor dumps)

Tool: `tools/ik2/ref/inkling-dump.cpp` (built as `llama-inkling-dump` in the reference checkout,
`examples/inkling-dump`), run in the ROCm container (`tools/ik2/ref/run_dump.sh`). It evaluates the prompt as
one batch (step 0), then 3-4 greedy single-token decodes (steps 1..N), and saves every graph tensor whose name
matches a regex.

- `<ref>/short`: prompt `The capital of France is Paris. The capital of Germany is` (12 tokens:
  `371 151451 149879 151194 149884 152717 13 149917 151451 149879 153067 149884`), greedy continuation
  `154328 13 149917 151451` (" Berlin. The capital"). Tensors: `l_out-L`, `ffn_inp-L`, `ffn_out-L`, `attn_out-L`
  (SWA layers), `ffn_moe_topk-L` (i32 picks), `ffn_moe_probs-L`, `ffn_moe_weights*-L`, `result_norm`,
  `result_output` (last row only), and layers 0 and 5: `Qcur`, `Kcur`, `Vcur`, `Rcur`, `kq_b`, `kqv_out`.
- `<ref>/long`: 2698-token prompt (`tools/ik2/ref/prompt-long.txt`), 3 steps: `l_out-L`,
  `ffn_moe_topk-L`, `attn_out-{0,5,47}`, `result_norm`, `result_output` - exercises the SWA ring and the distance cut.
- Format: `index.jsonl` (one line a tensor: step, name, file, type f32/i32, op, ne[4]); `s<step>/<name>.bin`
  contiguous, ne0 fastest; `tokens.txt` (prompt ids, then the greedy ids). Rows are tokens (ne1).
- `<ref>/long-f32`: the same prompt with `-fa off -ctk f32 -ctv f32` (the reference's own noise
  floor), `attn_out-L`, `ffn_inp-L`, `ffn_out-L`, `l_out-L`, `ffn_moe_topk-L` for every layer.
- **Parity, as measured.** The model is sensitive at a few positions: llama.cpp against itself (`long` vs `long-f32`,
  `tools/ik2/ref/compare_dumps.py`) drops to a worst `l_out` cosine of 0.23 (layer 35) with 89-95% of picks the
  same in the deep layers. The jumps come at the global layers (11, 17, 29, 35, 41), while the mean stays >= 0.97.
  So a per-row minimum of 0.999 cannot be met by any implementation on long prompts. The test is instead:
  - On the short reference with the reference's picks forced (`--parity-force-picks`): every row >= 0.999.
  - On the long reference: the same greedy tokens, a per-layer **mean** cosine >= 0.999 with picks forced, and a
    natural-pass worst cosine and pick agreement no worse than `long` vs `long-f32`.
  - maya-ik2 M1 met all of these. Short, forced: worst 0.99967. Long: natural worst 0.245 with 93.8% of picks
    the same; forced worst 0.70 with the mean >= 0.9993.

## Serving

`serve/server.py --engine maya --config <file>` runs the engine like any other. The setup makes everything, on AMD
under Linux only (`./maya.sh --setup --backend hip --gpus 0,1,2,3 --model inkling-2-small`, or Inkling 2 Small in its
model list; the Models page lists it under the other quants on an AMD install): the PC (VRAM + RAM should hold the
model, ~202 GB, `IK2_SMALL_MEMORY_GB`; below that the experts RAM does not hold are read in place from the model's files,
slower - the setup warns and goes on, as DeepSeek's does; `--mmap-experts` given to the setup is written and kept), the context (DeepSeek's: offered from 32K up to the
model's 1M, recommended from 64K up to 256K by what VRAM + RAM hold beyond the model), the engine
(`build-hip/maya-ik2`), unsloth's UD-Q2_K_XL set (`tools/ik2/download-small.json`, in
`<models>/Inkling-2-Small-GGUF/UD-Q2_K_XL`), Thinking Machines' `mtp.safetensors` and `config.json` in
`<models>/Inkling-2-Small-MTP` converted by `tools/ik2/convert_dspark.py --type q8_0`, the pack, its expert
profile (`expert-profile.bin`, copied from `data/expert-profile-ik2.bin` where the pack has none) and
`maya-inkling-2-small-hip.json` with `--dspark-index` and `--dspark none` until the speed calibration (DeepSeek's,
`tools/calibrate_ds4.py`: the config started without the drafter and with it, chat, code and tool calls) finds that
drafting pays on this PC (`--drafting measure|on|off`, `--speed-calibration`). Files already there are used as they are (a pack made by hand
included: `--repack` builds it again). `maya_quants.py`'s ids: `inkling-2-small`, `inkling-2-small-ud-q2_k_xl`
(`IK2_SMALL_IDS`; Inkling-2-Mini's are `IK2_IDS`, its CUDA preview's setup). On NVIDIA Inkling-2-Small has no setup
yet: the setup says so and stops.

By hand, the same steps:

    python tools/ik2/pack_gguf.py --model <the GGUF set's folder>   # <model>/maya-ik2: the pack, its tokenizer/

```json
{
 "exe": "<maya>/build-hip/maya-ik2",
 "args": ["--pack", "<models>/Inkling-2-Small-GGUF/UD-Q2_K_XL/maya-ik2", "--max-context", "32768",
          "--dspark-index", "<models>/Inkling-2-Small-MTP/maya-ik2-dspark-q8_0"],
 "cwd": "<maya>",
 "tokenizer": "<models>/Inkling-2-Small-GGUF/UD-Q2_K_XL/maya-ik2/tokenizer",
 "model_name": "inkling-2-small",
 "gpu": [0, 1, 2, 3],
 "backend": "hip",
 "lib_dirs": ["<rocm-sdk>/lib"],
 "port": 8080
}
```

- **Family.** `tools/maya_quants.py`'s `family_of` gives `ik2` by the engine's name (`maya-ik2*`, before the
  `--pack` check that would call it DeepSeek) or by `"installer": {"family": "ik2"}`, as for Inkling-2-Mini. A config
  is the AMD build's (`serve/models.py`'s `ik2_amd`) by its `"backend": "hip"` or its `--pack`; `is_ik2_mini` tells
  Mini's apart by its record (`ik2_quant` other than `ud-q2_k_xl`), so the AMD build's model key is `ik2`,
  Inkling-2-Small's. Sampling and the thinking level are its defaults (temperature 1.0, top_p 1.0, `auto`: the
  template's own effort, 0.9). The Models page gives the AMD build DeepSeek's rows (`IK2_AMD_KEYS`) and Mini's
  config its own (`IK2_KEYS`).
- **The template.** A pack made before `pack_gguf.py` wrote the official template holds the GGUF's (the llama.cpp
  conversion's) in `tokenizer/chat_template.jinja`; the server then renders with `tools/ik2/chat_template.jinja`
  and says so at its start (`chat_template_path`; `MAYA_PACK_TEMPLATE=1` keeps the pack's).
- **The command.** `Popen([exe, "--serve", *args])` with the config's args, plus `--layer-split auto` when `gpu`
  lists more than one GPU (`engine_args`; not on NVIDIA, where maya-ik2 stops at that flag), in `cwd`, with
  `HIP_VISIBLE_DEVICES` = the config's GPUs and `lib_dirs` first on the library path. The Models page changes
  `--max-context`, the GPUs and Drafting (`--dspark on|none` with `--dspark-index`, offered where the drafter's index is
  on this PC; the CUDA build drafts in its CLI only), and DeepSeek's rows for the settings the AMD build shares with maya-ds4:
  `--parallel` (Answers at once), `--reserve-mib`, `--mmap-experts` (Experts VRAM does not hold), `--prefill-chunk`, `--cpu-threads` and `--dspark-cum` (as DeepSeek's Draft confidence,
  `--dspark-conf`); every other flag stays as the config has it.
- **The drafter's index.** The engine drafts only with the index `--dspark-index` names (the setup writes it); without
  one it decodes without drafts and says so at its start.
- **What the server adds for the model** beyond INKLING2.md's list: a tool result headed by its tool's name
  (`<|message_tool|>NAME`, by its place after the assistant's calls: the server's messages carry no call ids);
  `<|message_user|>` / `<|message_system|>` / `<|message_tool|>` as stops beside the end of sampling and
  `<|endoftext|>`; the loop guard's view of a call (`<|content_invoke_tool_json|>` .. `<|end_message|>`) and the
  prompts it continues with; a tool result's picture as a text note (a user's picture is a plain 400: text only).

The line protocol (maya-hy4's, `src/hy4/hy4_main.cpp`'s `serve`):

- Start: `NOTE <text>` / `INFO key=value ...` lines (`INFO engine=inkling2 engine_kind=inkling2 model=Inkling-2-Small
  ctx=<n> vram_gb=<x> dspark=1 logit_bias=1 ...`; `dspark=0` with `--dspark off` / `none` or no drafter index), then
  `READY <max_context> stop`. stderr is the log.
- The server sends `GEN <max_new> [key=value ...] <id,id,...>`, the whole prompt each time (the engine reuses the
  prefix it holds). Keys: `temperature` (absent or 0: greedy), `top_p`, `top_k`, `min_p`, `seed`, `penalty_*`,
  `logit_bias=<id>:<v>,...` (the server's `ignore_eos` is a bias on the stop ids); for the drafter `spec=0|1` and
  `dspark_cum=F` / `spec_cum=F`. Unknown keys are ignored. `STOP`, `WRAP`, `QUIT`. No `GENI` (text only).
- The engine answers `PP <read> <total> <ms> <tok_s> <reused>` while it reads the prompt, `T <id>` a token (a round
  with drafts gives several at once), `STAT ...` now and then (with the drafter `draft_rounds= draft_ms= verify_ms=
  drafts= draft_acc= draft_off=`, maya-ds4's names), and ends with `DONE <generated> <prompt_tokens> <prompt_ms>
  <decode_ms> <finish> <drafts_acc> <drafts_off> <reused> 0 0` (finish `stop` | `length` | `cancel`) or `ERR
  <message>`. It stops at 200006 and 199999 (`--eos`); ids past 200057, the padded rows, are never sampled.
- `--parallel N` (N > 1, maya-ds4's): `READY <max_context> stop multi=N`; a GEN needs `rid=<id>` and every line of it
  carries the id after its keyword (`T <rid> <id>`, `PP <rid> ...`, `DONE <rid> ...`, `ERR <rid> ...`); `STOP <rid>`
  ends that request alone, a bare `STOP` every one ("Several conversations at once" below).

## Build and run

```sh
R=<rocm-sdk>
cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_HIP_COMPILER=$R/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=$R -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
cmake --build build-hip --target maya-ik2 -j
LD_LIBRARY_PATH=$R/lib build-hip/maya-ik2 --pack <pack> ...
```

`./maya.sh --setup` builds it as it builds the other engines. CMake hooks: `cmake/ik2_hip_engine.cmake`, `cmake/ik2_hip_kernels.cmake` and
`cmake/ik2_hip_dspark.cmake`, included from CMakeLists.txt after Hy4's block inside `if(STRATA_ENABLE_HIP)`
(`_strata_gpu_*` variables in scope; tests inside `if(STRATA_BUILD_TESTS)`).

## Open questions

- `log_scaling_n_floor` (settled: **8192**). The GGUF was converted from the first config.json, which said 128000.
  Thinking Machines' 2026-11-17 commit 1df2d97d45 changed only that value, to 8192, which is also NeMo AutoModel's
  inkling2 default. pack_gguf.py reads the stale 128000 as 8192 (`STALE_FLOOR`); `--log-floor` overrides it.
  llama.cpp with this GGUF still uses 128000, so past position 8192 maya-ik2 differs from it on purpose.
- The drafter (done: [INKLING2_DSPARK.md](INKLING2_DSPARK.md)). It is not in the GGUF set; Thinking Machines ships it
  as `mtp.safetensors` (3.4 GB, BF16) with its config.json, and the setup downloads it beside the model.
  - Shape (`mtp_config`): 5 dense layers (hidden 4608, 48/8 heads, head_dim 128, FFN 18432), layers 0/1/3/4 SWA 1024,
    RoPE theta 1e6, Q/K norms.
  - Block drafting, 8 tokens a block, fed by the target's hidden states at layers 5, 11, ..., 41, 48 through a fuse
    matrix and per-layer K_ctx/V_ctx projections.
  - A rank-256 GRU-style "markov" head whose correction goes through the target's own output matrix.
  - Reference: llama.cpp PR #32218 (open, on top of its DeepSeek-V4 DSpark "dflash" model). Its diff is saved at
    <ref>/dspark-pr32218.diff. maya-ds4's DSpark path is the template for block drafting and
    verification.
- Routing skew: on the long reference prompt the top 60% of a layer's experts take 97.5% of its picks (worst layer
  89%) - one document, so an upper bound; placement should also adapt per conversation.

## VRAM: the experts take what is left (measured, 2026-11-21)

The load places the routed experts last, as maya-ds4's `place_experts` does: each GPU first takes everything else -
the decode and prompt-path scratch, its layers' dense tensors, caches and histories, the head, the host-free step's
buffers, the drafter on the last GPU (its load's temporaries fit there, before the experts take the room) - then the
runtime's own share is taken by a warm-up (`Impl::warm_up`: a 256-token dummy chunk through every layer, each layer's
experts one zero blob in pinned memory, the drafter fed and asked once), and only then `Impl::place` reads the GPU's
free VRAM and fills it, less `--reserve-mib`, with its layers' experts (the most routed by the profile first, a spare
slot a layer). A GPU's experts are one allocation, rounded up to the runtime's granularity - measured at the load with
two probe allocations (here 2 MiB pages: `hipMalloc` of 1 MiB + 1 takes 2 MiB, of 2 MiB + 1 takes 4). A second
warm-up with the experts in place - a chunk and a quarter of tokens spread over the vocabulary, then decode windows of
1, 4 and 9 rows - takes no VRAM; it pays what the experts' four-minute load undid: that load's memory pressure sends
to swap (zram here) the host memory the first warm-up touched, and the first prompt then paged it back in - one of a
GPU's layers stalled 5.6 - 5.8 s, a 2.7k-token prompt took 7.4 - 8.9 s instead of 2.8, in 11 of 16 loads. Now the
load pays it (the second warm-up 1 - 7 s) and the first prompt reads as its repeat does. There is no estimate left in the plan: the guessed slacks (+256 B a tensor, +4096 B a layer, +32 MiB prompt
path, +64 MiB decode scratch) and the drafter's 3.09 GB figure are gone. `IK2_HIP_VRAM_LOG=1` prints a line a GPU at
each stage of the load and after each prompt, decode and served request: its free VRAM, the engine's own allocations
and the rest.

Where the VRAM outside the old plan went (GPU 0 at a 32K context, MiB; "runtime" is free VRAM lost beyond the
engine's own allocations):

| Stage | Engine's allocations | Runtime |
|---|---|---|
| The first stream (the device's start: before `free0`, never in a plan) | - | 146 |
| Decode scratch (31 allocations) | 56.2 | 1.8 (64 KiB minimum an allocation) |
| Prompt path scratch (41 allocations, the BLAS handle, the side stream) | 1299.7 | 70.3 (hipblasCreate 60, a stream 2, rounding) |
| 13 layers' dense tensors, caches, histories (371 allocations) | 1840.0 | 166.0 (2 MiB pages: a quarter to half a page an allocation) |
| The host-free step's buffers (57 allocations) | 15.6 | 4.4 |
| The warm-up (the BLAS's kernels, the kernels' code, their scratch) | - | 156.0 |
| The experts (one allocation) | 28186.4 | 1.6 |
| Last GPU: the drafter (its weights, caches, scratch and the engine's tap buffers) | 3162.5 | 151.5 (its BLAS handle, stream, kernels) |

Nothing grows after the load: prompts up to 80K tokens, drafted and undrafted decodes, and served requests leave the
free VRAM where the load left it, to the MiB. (A probe of this runtime, `hipMemGetInfo` around each call: the first
stream 146 MiB, each further stream 2, hipblasCreate 60, a first GemmEx 40, host memory - pinned, mapped or
registered, 4 GiB of it - none.)

Free VRAM after the load and after a long prompt, MiB a GPU (GPUs 0, 1, 2 / GPU 3):

| Context | Before (old plan, `--reserve-mib` R) | Now, `--reserve-mib 600`: after the load | after a long prompt |
|---|---|---|---|
| 32K | R - 324 / R - 642 (R = 1500: 1176 / 858) | 606 / 608 | 606 / 608 (30,000 tokens) |
| 256K | 310 - 340 with R = 1024 | 602 / 602 | 602 / 602 (80,000 tokens) |
| 512K ("Long contexts": the prompt path's buffers as before) | - | 600 / 608 | 600 / 608 (307,710 tokens) |
| 1M | - | did not load (the experts VRAM leaves: 131 GB, more than the RAM); with `--mmap-experts` (below): 604 / 604 | 604 / 604 (2,698 tokens) |

The rest above 600 is the experts' rounding: what no further expert fits (an expert is 9.6 - 12.1 MiB) and the
allocation's 2 MiB page. The experts in VRAM: 10,574 of 18,432 at 32K (10,565 before, when 0.3 GB a GPU was left
free instead of 0.6), 9,646 at 256K (9,630 before) - the old plan's slacks and the drafter's 3.09 GB over-counted
about as much as the runtime's share it missed. The load is about a minute longer than before (247 - 293 s against
166 - 225): the dense tensors and the experts are read in two passes over the files.

## Long contexts on this PC (measured, 2026-11-21)

**1M did not load here** before `--mmap-experts` ("The experts VRAM does not hold" below, which now loads it). At
`--max-context 1048576` the global layers' K / V rows take 8 GiB a GPU, the decode step's split-attention scratch 1.5
GiB a GPU (IK2_ATTN_BYTES: 16-key tiles over every key) and the drafter 9.6 GB of the last GPU, so the experts VRAM
leaves (6,454 of 18,432 stay) take 131 GB; this PC has 125 GB of RAM, 109 - 121 of it to be had, and the engine keeps
8 GiB for the rest (maya-ds4's RAM margin): it stopped at its start. Without the drafter it would still need about
121.5 GB. The largest context whose experts all fit in RAM with the drafter is about 630K (each position more takes 45
KB of VRAM a context: the K / V rows, the attention scratch, the drafter's rows - so about as much more RAM for
experts); the setup recommends up to 256K (S16).

**512K end to end** (`--max-context 524288`, the drafter on, the engine's defaults - the run named
`--prefill-chunk 512`, `IK2_HIP_STAGE=24` and `IK2_HIP_S_FLOOR_MI=44`, which the defaults now are; with the larger
2048 / 64 / 176 Mi of E9's measurement it did not load: the RAM experts 114 GB and the 8 GiB margin against 121 GB). 8,602 experts in VRAM, 99.8 GiB in RAM;
loaded in 396 s. The 2.7k-token prompt: 796 tok/s, 200 drafted tokens at 61.3 tok/s. Then a 307,710-token prompt -
the guide repeated 19 times with two maintainer's notes between the copies (OSPREY-4, port 51877; "Silver Lynx"), a
question about them at the end, thinking effort 0: read in 1106.7 s (278 tok/s); 133 tokens decoded at 7.9 tok/s
(30.8% of the picks on the CPU lane; drafting rests there: 1.31 tokens a pass, a pass 117 ms, a draft 19.2 ms). The
answer is coherent and about the document - it offers to explain how to add VRAM and RAM to see whether a model
fits, the "Recommended" column against "Whole model", and what the context costs - but declines to quote the notes
("internal maintainer details"): the long context is read and understood; the needle itself was not retrieved. The
free VRAM stayed at 600 / 608 MiB after both prompts and decodes. CONTEXT_TESTED["ik2-hip"] is 524288 (W2).

Earlier (2026-11-20, the drafter on, `--max-context 262144`, through `--serve`): a 262,000-token prompt read at 336
tok/s, 64 tokens at 10.0 tok/s, the decode path's rows the prompt path's.

**The speed calibration** (`./maya.sh --speed-calibration`'s `speed_calibration` -> tools/calibrate_ds4.py, run on a
scratch copy of the config the setup writes: 64K, `--reserve-mib 3072`, `--dspark none`): the DSpark drafter is
faster on this PC, 48.4 against 35.0 tokens/s combined (+38%; chat 36.7 against 34.3, code 55.0 against 35.0, tool
calls 60.5 against 35.8; 69% of the drafts accepted), so the config drafts. Its first run stopped at rendering the
tool calls' requests: it read the pack's chat template as it lies - for this pack the GGUF's, which refuses bare tool
declarations ("Tool declarations require a function object") and is not what the server renders with; it now takes
the server's own choice (serve/server.py's chat_template_path: thinkingmachines' template).

## The experts VRAM does not hold: `--mmap-experts` (2026-11-21)

maya-ds4's flag, values and default (`--mmap-experts on|off|auto`, ds4_model.cu's copy_experts): once VRAM's experts
are placed, the others are copied into RAM or read in place from the model's files (the page cache, the SSD where it
does not hold them). `off`: the most routed (routings a byte, the VRAM ones' order) that fit the RAM this process can
get beside the 8 GiB margin (E51) are copied, the others read in place; `on`: none copied; `auto`, the default:
maya-ds4's copies all of them when they fit with 8 GiB more to spare and else none (its #339: a part copied cost
decode speed on 32 / 64 GB) - here `auto` is `off`'s rule, measured below (E95). Where all of them fit - every
context up to the ~630K above on this PC - nothing changes: all copied, as before.

In place, a layer's RAM slots are its copied experts first (`ram`, as before), then the ones in the files (the CPU
lane's `CpuExperts::gu / dn`: the fused gate | up rows and the down rows where the GGUF has them). The CPU lane reads
those pages as they are; the prompt path gathers a staging pass's experts from the files into a pinned buffer of its
GPU (kStage blobs, 330 MB a GPU, allocated only where a layer has experts in the files) on up to 8 threads after an
`madvise(WILLNEED)` of each (E96), then copies them to the staging slots - not straight from the mapping, which the
load once saw land a stale page in VRAM (h2d's note); the expert cache's moves read an expert in through its bounce,
and a cold expert going back to the files is only a pointer. The pages of a layer's experts copied into VRAM or RAM
are let go of (`MADV_DONTNEED`), the in-place ones' kept. The Models page's "Experts VRAM does not hold" row (Auto /
From the files / In RAM) and the setup's `--mmap-experts` are DeepSeek's (W3, S31); below the model's memory the setup
now says the experts RAM does not hold are read from the SSD (S23).

Measured (the drafter on, the shipped profile read-only, `IK2_HIP_VRAM_LOG=1`):

| Run | Experts in VRAM / RAM / files | Load | 2.7k-token prompt | Its 200 tokens, drafted | CPU lane |
|---|---|---|---|---|---|
| 4K, `on` (parity) | 10,949 / 0 / 74.9 GiB | - | PARITY OK x4 (short, long; natural, forced) | - | 17.9 ms a token (short) |
| 1M, none copied (maya-ds4's auto) | 6,454 / 0 / 122.0 GiB | 140 s | 51.4 tok/s (52.5 s) | 32.2 tok/s (3.32 tokens a pass) | 16.2% of the picks, 20.2 ms a token |
| 1M, `off` (= auto here) | 6,454 / 100.0 GiB / 22.0 GiB | 392 s | 49.1 tok/s (54.9 s) | **47.2 tok/s** (3.49 a pass) | 16.3%, 8.8 ms a token |
| 512K before (all copied) | 8,602 / 99.8 GiB / 0 | 396 s | 796 tok/s | 61.3 tok/s | 9.9%, 4.4 ms a token |

The free VRAM after the 1M loads, prompts and decodes: 604 MiB on every GPU. So a part copied decodes 47% faster
than none on this PC (the CPU lane's share of the picks the same, 16.2 / 16.3%, its time a token 8.8 ms against
20.2), and `auto` takes `off`'s rule. Prompts are another
matter: every 512-token chunk streams almost every expert outside VRAM (37,496 for the 2.7k tokens' six chunks), and
with experts in the files a chunk takes about 9 s whether 22 GiB of them are there (`off`: the copies leave the page
cache little) or 122 GiB (none copied: 113 GB of page cache during the run) - 50 tok/s against 796 at 512K with all
of them copied and registered. That is what remains: long prompts at 1M are slow here.

The checks against the baseline (the default, all copied at these contexts, so the old path): PARITY OK x4, the rewind
test's five cases OK, `--ab` prose 33.9 -> 59.7, code 37.1 -> 58.1, thinking 37.4 -> 38.4 tok/s, `--serve` 37.7 tok/s
without drafts and 59.3 with them (the 2.7k prompt continued: 65.3) - the baseline's 33.9 / 59.5, 37.1 / 58.3, 37.3 /
38.7, 37.6 / 58.3 (65.4); the free VRAM after each decode 600 / 600 / 600 / 604 MiB.

## Several conversations at once: `--parallel N` (2026-11-21)

maya-ds4's flag and protocol (`--parallel N`, 1 .. 8, with `--serve`): the engine holds N conversations, each with its
own state - every layer's K / V caches (the global layers' `max_ctx` rows, the sliding-window rings), the short
convolutions' histories (and the other buffer a step writes them to), the saved histories a rewind goes back to, the
drafter's caches and context ring (`Dspark::add_states` / `bind`: the weights shared) - all of it allocated before the
experts are placed, so the experts take what is left (E2's rule). `READY <ctx> stop multi=N`; a GEN carries
`rid=<id>`, every line of a request carries it after its keyword (`T <rid> <id>`, `PP <rid> ...`, `DONE <rid> ...`,
`ERR <rid> ...`), `STOP <rid>` ends that request alone and a bare `STOP` every one - serve/server.py's `--parallel`
handling for maya-ds4 (`_generate_multi`, `_route`), unchanged.

- A request goes to the idle conversation whose tokens its prompt shares the most of (ties: the least recently used),
  else it waits for one (maya-ds4's rule, without its MAYA_DS4_SLOT_KEEP).
- Its prompt is read whole when its turn comes, in its own conversation (`Model::bind`: every single-conversation path
  - the prompt path, rewinds, the window, keep - works on the bound one).
- Between two prompts every running request writes. Alone, it decodes in rounds as with one conversation (the
  drafter's windows). Two or more: a token of each in one step through every layer (`Model::step_multi`): the
  projections, the router and the experts take the rows together - the weights read once, the CPU lane's expert read
  once for every row that picked it - and each row's short convolutions, K / V rows, attention and histories are its
  own conversation's (the T = 1 launches of the window's kernels, at the row's position). Each request's logit bias
  and sampler run on its own row's logits on the host. Each conversation's drafter is fed its row.
- `--multi-check DIR` (tests, with `--parallel N`): N conversations of a reference's prompt (37 tokens shorter each),
  K greedy tokens each alone, then the same through multi steps - each row's logits must be its own step's bits (the
  expert cache off).

Measured (`--parallel 8`, `--max-context 32768`, the drafter loaded, the shipped profile read-only): the other seven
conversations' state takes 4,940 MiB on each of GPUs 0 - 2 and 3,868 on GPU 3, their drafter states 1,940 MiB more
on GPU 3 - 2.9 GiB a conversation, so 8,630 of the 18,432 experts stay in VRAM (10,604 with one conversation); the
free VRAM after the load 604 / 604 / 604 / 602 MiB. Eight chat prompts (20 - 26 tokens, "Explain ... in detail.",
effort 0), 128 greedy tokens each, without drafts (`spec=0`):

| Requests at once | Tokens a second in all | Each request's | Against one request without --parallel (37.6) |
|---|---|---|---|
| 1 (each prompt alone) | 30.0 - 39.5 | - | - |
| 2 | 52.6 | 26.4 - 27.7 | 1.40x |
| 4 | 67.0 | 17.0 - 18.6 | 1.78x |
| 8 | 80.6 | 10.4 - 11.8 | 2.14x |

The answers next to their alone runs: the same for 2 - 128 of the 128 tokens - they part where the expert cache moved
an expert between the runs (as two loads of one prompt part: "Status"). `--multi-check` (each row's logits against its
own step's bits, the expert cache off), `--parallel 4`, 4 conversations x 32 tokens: 0 of 128 rows differ (worst
logits cosine 1.0000000), the same tokens in all 4; alone 28.8 ms a token (34.7 tok/s), the 4 together 61.9 ms a
step (64.6 tok/s in all) - MULTI OK.

What remains against maya-ds4's: it reads a long prompt a piece a turn while the others write, and drafts beside
several requests (its multi windows); here a prompt is read whole when its turn comes, and only a request alone drafts.

## The measured values (2026-11-21)

The measurement phase's list ("Values and their sources", the rows that were MEASURE), measured on this PC. Each value
has an `IK2_HIP_<NAME>` override (maya-ds4's `DS4_CACHE_*` way: read at the load); `--sweep VAR=a,b,..`
(tests, with `--prompts`) runs each prompt's drafted decode with each value in one load, after a warm run of the
prompt, `--sweep-rounds` times in ABBA order, so the expert cache's state favours no value. Workloads: `--ab`'s three
prompts (prose: the 2.7k-token document continued; code: the LRU-cache module; thinking: the sourdough question at
effort 0.9), 256 greedy tokens; prompts of 5K / 20K / 80K tokens of the long document for the prompt path; a six-turn
chat through serve/server.py for the cache and the rewind (the 2.7k document and five follow-ups, greedy, up to 3000
tokens an answer, each answer sent back without its reasoning, as OpenAI clients do). Every load since the profile
fix reads the shipped expert profile, read-only.

Decision rule: where maya-ds4 has a counterpart and its value is within noise of the best, maya-ds4's; else the
measured best; a value that does not matter within noise and has no counterpart is kept.

**Noise.** The drafted decode of the same prompt in one load: prose 59 - 73 tok/s (greedy text forks at near-ties, and
the drafts kept with it), code 54 - 63, thinking 37.3 - 41.0; the three prompts' mean, the default setting repeated
across nine sweeps, 52.9 - 56.3. The prompt path repeats within 0.5%. A chat's decode speed: 48.7 - 51.3 tok/s
between loads.

| # | Value | Old | New | Alternatives, results (tok/s unless said) | Noise | Source |
|---|---|---|---|---|---|---|
| E9 | Prompt chunk (`--prefill-chunk`) | 512 | 512 | 512 / 1024 / 2048 / 4096, prompt tok/s (one allocation for all): 5K 1020 / 1110 / 1056 / 841; 20K 1128 / 1350 / 1546 / 1548; 80K 914 / 1065 / 1193 / 1222. Decode, two loads each: 2048 1% slower (undrafted 33.67 / 35.80 / 36.47 against 33.76 / 36.22 / 36.96). Against it, with E42 and E44: the 2048 chunk, the 176 Mi floor and 64-expert staging together take about 2.85 GiB a GPU of the experts' room (1,140 experts at 32K): prompts of 67 / 82 / 300 / 2698 tokens then read at 114 - 117 / 102 - 105 / 297 - 299 / 883 tok/s instead of 130 - 134 / 111 - 114 / 330 - 333 / 1022 (a prompt streams each RAM expert it routes to; 2.7K tokens are also two chunks instead of six to pipeline), and the setup's own config - 64K, --reserve-mib 3072, the drafter - no longer loads on this PC (the experts outside VRAM 109 GB and the 8 GiB margin against 116 GB: the speed calibration's drafter side) | prompt 0.5%; undrafted decode 0.3 | MEASURED (maya-ds4's 4096: 20% slower at 5K) |
| E8, D2 | Rewind reach (`--rewind`) | 4096 | 4096 | The chat, prompt tokens read again past turn 1: 2048 11,486 (13.4 s), 4096 998 / 2,538 (3.1 / 5.6 s), 16384 745 (2.7 s); decode 49.8 / 50.0 / 50.9 | chat decode 48.7 - 51.3 | MEASURED (no maya-ds4 counterpart) |
| E15 | `--short-prompt` | 3 | **4** | Prompt path / decode path, ms, after 2.7K tokens: 3 tokens 118.7 / 95.9, 4 125.9 / 123.2, 5 129.8 / 148.1 | 3 ms | MEASURED |
| E27 | Draft time at start | 10 ms | 10 ms | 5 / 10 / 20: mean 54.9 / 55.0 / 55.0 | 52.9 - 56.3 | MEASURED (insensitive) |
| E28 | Rest length | 8 | 8 | 4 / 8 / 16: 54.8 / 54.0 / 53.6 | as above | MEASURED (insensitive) |
| E29 | Rest cap | 32 | 32 | 16 / 32 / 64: 55.9 / 54.9 / 53.6 | as above | MEASURED (insensitive) |
| E30 | Rounds before judging | 8 | 8 | 4 / 8 / 16: 53.8 / 55.4 / 53.9 | as above | MEASURED (insensitive) |
| E31 | Rest when gain < x·a | 0.1 | 0.1 | 0.05 / 0.1 / 0.2: 55.1 / 56.3 / 52.2 | as above | MEASURED (insensitive) |
| E32 | Rest length reset when gain > x·a | 0.3 | 0.3 | 0.2 / 0.3 / 0.5: 57.2 / 54.6 / 52.9 | as above | MEASURED (insensitive) |
| E33 | Averages' weight | 0.9 | 0.9 | 0.8 / 0.9 / 0.95: 51.8 / 52.9 / 53.8 | as above | MEASURED (insensitive) |
| E34 | Row cost floor | 0.5 ms | 0.5 ms | 0.25 / 0.5 / 1: 54.2 / 55.8 / 53.4 | as above | MEASURED (insensitive) |
| E42 | Score scratch floor (`kS_n`) | 44 Mi floats | 44 Mi | 80K prompt tok/s, 11 / 22 / 44 / 88 / 176 / 352 Mi: 853 / 1082 / 1247 / 1354 / 1409 / 1400; 20K within 1%; the larger floors' VRAM as E9 | 0.2% | MEASURED |
| E43 | Tokens an attention pass (`kTsMax`) | 512 | **128** | 2048-token chunks, 32 / 64 / 128 / 256 / 512: 20K 1806 / 1861 / 1875 / 1844 / 1841 tok/s, 80K 1483 / 1524 / 1532 / 1513 / 1511 (its rel-bias table 25 MB instead of 100) | 0.1% | MEASURED |
| E44 | Experts a staging pass (`kStage`) | 24 | 24 | 24 / 48 / 64 / 96: 20K 1579 / 1713 / 1812 / 1831 tok/s, 80K 1358 / 1436 / 1499 / 1509; the larger stages' VRAM (2 x 64 experts: 1.2 - 1.5 GiB a GPU) as E9 | 1% | MEASURED |
| E52 | The load's pinned stage | 256 MiB | 256 MiB | Pinned copies: 64 MiB 56.3 GB/s, 256 MiB 56.7, 1 GiB 56.9; the load reads its files at about 1 GB/s | - | MEASURED (insensitive) |
| E54 | Warm-up tokens | 256 | 256 | 16 / 64 / 256 all take the runtime's whole share (the free VRAM did not move after the load); the first 5K prompt after a 16-token warm-up 7% slower than its repeat, after 256 2% | - | MEASURED |
| E56, E57 | Expert cache: decay; swap rule | x0.995 a token; hot > 1.5 cold + 0.5 | **x0.9 every 256 tokens; hot > 1.25 cold + 1.0** | The chat, two loads each: maya-ds4's 51.3 / 51.0 tok/s, the CPU lane 4.0 / 4.3 %; the old 48.7 / 51.2, 4.1 / 3.9 % | chat decode 48.7 - 51.3 | MAYA (within noise) |
| E60, E61 | Burst size, trigger | 48 x 64, 64 tokens | kept | Inactive: the burst is off (E59) | - | MEASURED (inactive) |
| E62 | A window's moves count its rows as up to | 4 tokens | 4 | 2 / 4 / 8: mean 54.2 / 55.0 / 52.1 | as E27 | MEASURED (insensitive) |
| E63 | A prompt chunk's routing weight min(1, x/T) | 64 | 64 | 16 / 64 / 256: 52.5 / 52.1 / 53.5 | as E27 | MEASURED (insensitive) |
| E65 | Histories saved every | 64 tokens | 64 | 16 / 64 / 256: 52.9 / 55.0 / 53.1 (96 snapshots x 64 = 6144 decoded positions cover the 4096 rewind; at 16 they would not) | as E27 | MEASURED (insensitive) |
| E69 | The lane's spin | 20000 yields (1.8 ms) | **20 ms** | 200 µs / 1.8 ms / 20 ms: 57.0 / 55.0 / 55.2 | as E27 | MAYA (within noise) |
| E74 | Token attention tiles | 32 / 128 | kept | Not on the serving path (the tapped step only) | - | CHOSEN (trade) |

## Values and their sources

Every value of this engine, its drafter, the setup, the server's and the Models page's Inkling 2 rows and the tools,
with where it comes from (2026-11-20). Sources:

- **MODEL**: the model's own files (the GGUF set, config.json, vocab.json, the official chat template); **MODEL\***:
  equal to the reference implementation (llama.cpp's `src/models/inkling2.cpp`, PR #32218's converter).
- **MAYA**: what maya-ds4, DeepSeek's setup or another Maya family does - its name, default and behaviour.
- **MEASURED**: a recorded measurement (cited). **DERIVED**: computed from the above (the formula given).
- **CHOSEN (kind)**: an internal implementation constant with the reason in its code comment - a bound it must
  cover, what it trades, a method, a name, a scope.
- **MEASURE**: none left. The measurement phase (2026-11-21, "The measured values" above) measured each: MEASURED
  where the measured best was taken (with "insensitive" where no value mattered within noise and the value was kept,
  "inactive" where it never runs), MAYA where maya-ds4's value was within noise of the best; the VRAM slacks are gone
  ("VRAM" above).

### Engine

| # | Value | What it does | Source | Where it comes from |
|---|---|---|---|---|
| E1 | `max_ctx` 32768 | Positions the global layers' caches hold; the engine's context | MAYA | ds4_model.hpp `max_context = 32768`. Every recorded measurement used 32768 (INKLING2_AMD.md). The setup writes 64K or 256K instead (S15, S16). |
| E2 | `reserve_mib` 600 | VRAM each GPU keeps free: the experts take the GPU's free VRAM, measured once everything else is in place, less this | MAYA | ds4_model.hpp `reserve_mib = 600`; the plan is maya-ds4's place_experts rule ("VRAM" above): free after the load is 600 plus less than one expert, measured at 32K and 256K. The setup writes 3072 (S27), as DeepSeek's does. |
| E3 | `vram_experts` 1.0 | The largest share of a layer's experts kept in VRAM | MAYA | ds4_model.hpp `1.0` |
| E4 | `cpu_threads` 0: a thread a physical core, the caller's among them; N at most N | CPU lane workers | MAYA | ds4_model.hpp's cpu_threads (0 and its meaning). Not mirrored: maya-ds4's lane picks between the cores, the performance cores or one NUMA node by a measurement at load; this lane is hy4::Lane |
| E5 | Default profile `<pack>/expert-profile.bin` (a pack's `default.prof` read in its place) | Expert profile read when none is given | MAYA | ds4_main.cpp's `o.pack + "/expert-profile.bin"` |
| E6 | Written back after each request and at the end; `--profile-readonly`; `--profile-out` this run's alone | Whether routing counts are saved | MAYA | ds4_model.hpp's profile_readonly, ds4_model.cu's save_profile |
| E7 | `layer_split` empty: an equal share of all bytes per GPU | How the layers are split across the GPUs | CHOSEN (design) | Tensor parallel is ruled out (256-value blocks cannot split n_ff 2304: "The engine (design)" above); `--layer-split` sets it by hand |
| E8 | `rewind` 4096 | Extra rows the sliding-window rings keep, so a rewind can reach back that far without reading the prompt again | MEASURED | The six-turn chat ("The measured values" below), prompt tokens read again past the first turn: 2048 11,486 (turns after a 3000-token answer read their whole prompt again: 13.4 s), 4096 998 - 2,538 (3.1 - 5.6 s), 16384 745 (2.7 s); decode 49.8 / 50.0 / 50.9 tok/s, alike. 4096 holds this chat's answers; 16384 (maya-hy4's ring_slack) costs 0.5 GB a GPU more and reached no more here; no maya-ds4 counterpart |
| E9 | `prefill_chunk` 512 | Prompt tokens per prompt-path chunk | MEASURED | `--try-chunks` in one load, tok/s (two runs each, within 0.5%), 512 / 1024 / 2048 / 4096: 5K tokens 1020 / 1110 / 1056 / 841, 20K 1128 / 1350 / 1546 / 1548, 80K 914 / 1065 / 1193 / 1222 - one allocation for all, so the larger chunks' own cost is not in these: the 2048 chunk, the 176 Mi floor and 64-expert staging together take about 2.85 GiB a GPU of the experts' room (1,140 experts at 32K): prompts of 67 / 82 / 300 / 2698 tokens then read at 114 - 117 / 102 - 105 / 297 - 299 / 883 tok/s instead of 130 - 134 / 111 - 114 / 330 - 333 / 1022 (a prompt streams each RAM expert it routes to; 2.7K tokens are also two chunks instead of six to pipeline), and the setup's own config - 64K, --reserve-mib 3072, the drafter - no longer loads on this PC (the experts outside VRAM 109 GB and the 8 GiB margin against 116 GB: the speed calibration's drafter side). So 512, the best with what its VRAM costs counted. maya-ds4's auto rule, 4096 on several GPUs, is 20% slower at 5K even before that: not within noise |
| E10 | `log_floor` 0, so the pack's 8192 is used | log_scaling_n_floor | MODEL | config.json `text_config.log_scaling_n_floor 8192`; inkling2.txt 8192. The GGUF still says 128000 (see T1). |
| E11 | `dspark_cum` 0.5 | Floor on the running product of the drafts' probabilities | MEASURED | DSPARK.md: the ">= 0.5" row is the best of the projected rules. Also MAYA: ds4_main.cpp `0.5f` (measured there on a Radeon 8065S, +6.5%). |
| E12 | Removed (`dspark_reserve_gb` 3.09) | - | MEASURED | The drafter is loaded before the experts are placed, so the plan measures it in place: 3314 MiB at 32K (its 3162.5 MiB and its runtime's 151.5), 4794 at 256K ("VRAM" above) |
| E13 | `expert_cache` on, `--no-expert-cache` | Expert cache on by default | MAYA | ds4_model.hpp's expert_cache, ds4_main.cpp's flag |
| E14 | `dspark_index` empty: no drafter | The drafter loads only when its index folder is named | CHOSEN (design) | The drafter is a download of its own beside the GGUF set (not in the pack, as maya-ds4's is): the config names its index, the setup writes it |
| E15 | `--short-prompt` 4 | A prompt's unread tail of at most 4 tokens goes through the decode path | MEASURED | `--short-bench` after the 2.7k-token prompt, ms, prompt path / decode path (means of 5, each run within 3 ms): 3 tokens 118.7 / 95.9, **4 tokens 125.9 / 123.2**, 5 tokens 129.8 / 148.1, 8 tokens 146.4 / 231.7. Was 3; no maya-ds4 counterpart (its short prompts go through windows of 16) |
| E16 | `--dspark` "on" | Drafting by default | MEASURED | INKLING2_AMD.md and DSPARK.md: prose 1.84x, code 1.47x, thinking 0.97x on a fresh load. The ds4 engine also defaults to on (ds4_model.hpp); the setup default is a separate question (S28). |
| E17 | Stops `{200006, 199999}` | Ids that end an answer | MODEL | 200006 `<\|content_model_end_sampling\|>` (the GGUF's and config.json's eos_token_id), 199999 `<\|endoftext\|>` (vocab.json); the comment says so now |
| E18 | GEN's `top_k` mm3's default, 64, unless the request sets it | No top-k by default | MAYA | ds4_main.cpp's Req (top_k 64). The server sends top_k only where the request or the config sets it (0 as 64), the same for every family |
| E19 | 40 temperature-nats cut | Tokens more than 40·T below the top are left out of sampling | MAYA | hy4_main.cpp |
| E20 | seed `0x9E3779B97F4A7C15 * counter` | Default RNG seed for each request | MAYA | hy4_main.cpp; mm3_main.cpp |
| E21 | PP every 64 tokens | How often progress lines are printed | MAYA | ds4_main.cpp's prompt windows |
| E22 | STAT every 8 tokens | How often speed lines are printed | MAYA | ds4_main.cpp's `produced % 8`; maya-hy4's too. serve/ reads the latest line |
| E23 | stdin chunk 4096 B | Read buffer | CHOSEN (bound) | A read's size only: lines are put together across reads |
| E24 | `INFO ... model=Inkling-2-Small engine=inkling2 mtp=0` | Model name and protocol keys at start | MODEL | GGUF `general.name Inkling-2-Small`. The protocol is maya-hy4's (INKLING2_AMD.md). |
| E25 | `a` = 30 ms at start | Starting estimate of a one-row pass | MEASURED | INKLING2_AMD.md (t(1) = 29.9 ms at 2.7k context) |
| E26 | `b` = 7 ms at start | Starting estimate of one more window row | MEASURED | INKLING2_AMD.md (6.7 ms a row); DSPARK.md (7 - 8 fitted) |
| E27 | `t_draft` = 10 ms at start (IK2_HIP_DRAFT_T0) | Starting estimate of a draft's time | MEASURED (insensitive) | `--sweep` ("The measured values" below): 5 / 10 / 20 ms decode alike within noise |
| E28 | `rest_len` 8 (IK2_HIP_DRAFT_REST) | Rounds without drafts after a round that did not pay | MEASURED (insensitive) | 4 / 8 / 16 alike within noise (below) |
| E29 | Rest cap 32, doubling (IK2_HIP_DRAFT_REST_MAX) | Longest rest | MEASURED (insensitive) | 16 / 32 / 64 alike within noise (below) |
| E30 | 8 drafted rounds before judging (IK2_HIP_DRAFT_WARM) | Warm-up of the gain average | MEASURED (insensitive) | 4 / 8 / 16 alike within noise (below) |
| E31 | Rest when `gain < 0.1·a` (IK2_HIP_DRAFT_REST_GAIN) | When drafting rests | MEASURED (insensitive) | 0.05 / 0.1 / 0.2 alike within noise (below) |
| E32 | `gain > 0.3·a` resets the rest length (IK2_HIP_DRAFT_RESET_GAIN) | Hysteresis | MEASURED (insensitive) | 0.2 / 0.3 / 0.5 alike within noise (below) |
| E33 | EMA 0.9 / 0.1 (IK2_HIP_DRAFT_EMA) | Smoothing of a, b, gain and t_draft | MEASURED (insensitive) | 0.8 / 0.9 / 0.95 alike within noise (below) |
| E34 | `b` floor 0.5 ms (IK2_HIP_DRAFT_B_MIN) | Lower bound on the fitted row cost | MEASURED (insensitive) | 0.25 / 0.5 / 1 alike within noise (below) |
| E35 | `rest0` 0 | Rounds without a draft after k = 0 | MEASURED | main.cpp: "1 costs prose a sixth of its drafted rounds and gains thinking text nothing" |
| E36 | At most 7 drafts | Drafts per round | MODEL | config `mtp_config.block_size 8`: the anchor plus 7 drafts |
| E37 | Window sized by expected tokens per ms, on by default | How many drafts to verify | MEASURED | The rule's gain is in DSPARK.md. **Differs**: ds4's cost-aware window is off by default "until measured on GPUs" (ds4_main.cpp); ds4 uses only the fixed cum cut. |
| E38 | `ids[16]`, `tops[16]` | Window buffers | CHOSEN (bound) | At least window_max(), kWin 9 |
| E39 | Removed (+256 B per tensor) | - | MEASURED | The plan no longer estimates: it reads the free VRAM after the other allocations and the runtime's warm-up ("VRAM" above: each allocation's real size, 64 KiB at least and whole 2 MiB pages above 1 MiB, is in what it reads) |
| E40 | SWA ring = `n_swa + rewind + Tc` = 1024 + 4096 + 512 = 5632 rows | Rows of each sliding-window layer's cache | DERIVED | n_swa 1024 (GGUF `attention.sliding_window`) + E8 + E9 |
| E41 | Removed (+32 MiB) | - | MEASURED | The plan no longer estimates: it reads the free VRAM after the other allocations and the runtime's warm-up ("VRAM" above: each allocation's real size, 64 KiB at least and whole 2 MiB pages above 1 MiB, is in what it reads) |
| E42 | `S_n()` = max(44 Mi, n_head x max_keys) floats (IK2_HIP_S_FLOOR_MI) | Prompt-path attention score scratch | MEASURED (the floor); DERIVED (n_head x max_keys) | A pass of ts tokens over nk keys takes n_head x ts x nk scores and attention_rows halves ts until it fits: one token over every key must fit (44 Mi alone stops a prompt at about 961K keys). The floor sets how many tokens a long prompt's passes take: an 80K-token prompt, 2048-token chunks, tok/s at 11 / 22 / 44 / 88 / 176 / 352 Mi: 853 / 1082 / 1247 / 1354 / 1409 / 1400 (two runs each, within 0.2%); a 20K prompt within 1% at all of them. 176 Mi's 0.8 GiB a GPU more cost what E9 says, so 44 |
| E43 | `kTsMax` **128** (IK2_HIP_TS_MAX) | Tokens per pass of the prompt attention | MEASURED | 2048-token chunks, tok/s at 32 / 64 / 128 / 256 / 512 tokens a pass: a 20K prompt 1806 / 1861 / **1875** / 1844 / 1841, 80K 1483 / 1524 / 1532 / 1513 / 1511 ("The measured values" below); its rel-bias table kTsMax x 48 x 1024 floats, 25 MB. Was 512 (CHOSEN) |
| E44 | `kStage` 24 (IK2_HIP_STAGE) | Experts streamed per staging pass (two buffers) | MEASURED | 2048-token chunks, tok/s at 24 / 48 / 64 / 96: a 20K prompt 1579 / 1713 / 1812 / 1831, 80K 1358 / 1436 / 1499 / 1509 ("The measured values" above); 64's 0.8 GiB a GPU more cost what E9 says, so 24 |
| E45 | Removed (+4096 B per layer) | - | MEASURED | The plan no longer estimates: it reads the free VRAM after the other allocations and the runtime's warm-up ("VRAM" above: each allocation's real size, 64 KiB at least and whole 2 MiB pages above 1 MiB, is in what it reads) |
| E46 | Window scratch: 9 rows over up to 4096 keys | Attention scratch size for a window | CHOSEN (trade) | 9 = kWin rows; 4096 keys holds every SWA window (1024) and a global layer's first 4096 positions in one launch pair (58 MB a GPU); past it a window's rows run one at a time |
| E47 | Removed (+64 MiB) | - | MEASURED | The plan no longer estimates: it reads the free VRAM after the other allocations and the runtime's warm-up ("VRAM" above: each allocation's real size, 64 KiB at least and whole 2 MiB pages above 1 MiB, is in what it reads) |
| E48 | Removed from the engine; the setup's per-position cost `ctx_bytes` (D10) | Drafter VRAM at other contexts | MEASURED | 3314 MiB at 32K and 4794 at 256K: 1400 MiB for 229,376 positions, 6400 bytes each, D10 exactly |
| E49 | head + `8·n_embd` B | VRAM of output_norm + output_res_query (2 F32 vectors) | DERIVED | 2 x 4608 x 4 B |
| E50 | One spare expert slot per layer | Room for the cache's moves | CHOSEN (bound) | One move at a time a layer: the hot expert lands in the spare slot, the cold one's slot is the next spare |
| E51 | The RAM experts' copies leave `ram_margin()` 8 GiB (IK2_HIP_RAM_MARGIN_GB) of the RAM to be had | The copies that fit; the others read in place (E95) | MAYA | ds4_model.cu's ram_margin (8 GiB, DS4_RAM_MARGIN_GB); the setup's IK2_SMALL_MEMORY_GB counts the same 8 GiB (IK2_RAM_MARGIN_GIB). Before --mmap-experts the load stopped when they did not all fit |
| E52 | Staging pinned buffer of at least 256 MiB | Load-time copy buffer (host memory, not VRAM) | MEASURED (insensitive) | Pinned copies to an R9700: 64 MiB pieces 56.3 GB/s, 256 MiB 56.7, 1 GiB 56.9; the load reads the files at about 1 GB/s, so the piece's size is under 1% of it |
| E53 | 16 blobs: the movers' bounce buffer, and the most moves per poll | Expert-cache copy buffer | CHOSEN (bound) | swap_poll's most moves a round (16): the movers' bounce holds that many blobs |
| E54 | Warm-up chunk min(Tc, 256) tokens (IK2_HIP_WARM_TOKENS) | The runtime's share taken before the experts are placed; the first run's seconds | MEASURED | 16, 64 and 256 tokens all take the runtime's whole share (the free VRAM never moved after the load, prompts up to 80K); with 16 the first 5K prompt read 7% slower than its repeat (946 / 1020 tok/s), with 256 2% (996 / 1013). The 256 tokens cost 0.2 s of the load |
| E55 | token_embd as Q5_K / Q6_K / F32 | Embedding row types accepted | MODEL | The pack's tensors.txt: `token_embd.weight Q5_K` |
| E56 | The experts' heat x **0.9 every 256 tokens** (a token's share each token: 0.9^(1/256); IK2_HIP_CACHE_DECAY) | How fast an expert's heat decays | MAYA | maya-ds4's DS4_CACHE_DECAY 0.9 (ds4_model.cu ~1797), within noise of the earlier x0.995 a token: the six-turn chat, two loads each, 51.3 / 51.0 tok/s against 48.7 / 51.2, the CPU lane 4.0 / 4.3 % against 4.1 / 3.9 % |
| E57 | Move when `heat(hot) > 1.25·heat(cold) + 1.0` (IK2_HIP_CACHE_MARGIN / _BONUS) | The swap rule | MAYA | maya-ds4's DS4_CACHE_MARGIN 1.25 and _BONUS 1.0, measured with E56 (against 1.5 and 0.5) |
| E58 | `swap_batch` 4 | Moves per GPU between two steps | MEASURED | model.cu: "4 and 8 decode alike, the CPU lane at 2 - 3 % of the picks" |
| E59 | `burst_s` 0 (off) | Swap burst after a prompt | MEASURED | model.cu: "the first prompt after a load was seen to take 5 s longer with it" |
| E60 | 48 rounds x 64 moves (IK2_HIP_BURST_ROUNDS / _MOVES) | Burst size | MEASURED (inactive) | The burst is off (E59, measured): these never run; kept |
| E61 | Burst only after prompts of at least 64 tokens (IK2_HIP_BURST_MIN) | Burst trigger | MEASURED (inactive) | As E60 |
| E62 | Window poll: min(16, batch x min(T, 4)) moves (IK2_HIP_SWAP_WIN_ROWS) | Moves after a window | MEASURED (insensitive) | 2 / 4 / 8 alike within noise ("The measured values" below); at most the bounce's 16 |
| E63 | Prompt routings weighted min(1, 64/T) (IK2_HIP_CACHE_PROMPT_ROWS) | A prompt chunk's share of the heat | MEASURED (insensitive) | 16 / 64 / 256 alike within noise (below) |
| E64 | `IK2_HIP_SWAP_BATCH` clamped to 1..64 | Env bounds | CHOSEN (bound) | Env bounds: at least one move; above 16 swap_poll's bounce cap takes it |
| E65 | `kSnaps` 96, `kSnapEvery` 64 (IK2_HIP_SNAP_EVERY) | Convolution-history checkpoints: a ring of 96, one every 64 decoded tokens | MEASURED (insensitive) | Every 16 / 64 / 256 decoded tokens alike within noise (below); 96 x 64 = 6144 decoded positions, more than the 4096 rewind (E8): the bound holds (at 16 it would not) |
| E66 | `kWin` 9 | Most rows in a window | CHOSEN (bound) | A round takes at most 8 rows (the anchor and 7 drafts, mtp_config.block_size 8); the 9th only --window-test's 9-row window uses |
| E67 | Drafter taps 2..8, the last = n_layer | Check that the drafter fits the target | MODEL | config `mtp_config.target_layer_ids` (8 entries, the last 48) |
| E68 | `kLine` 16 unsigned (64 B) | Spacing of the mapped flags | CHOSEN (bound) | 64 bytes, a cache line: each flag on its own line |
| E69 | The lane's dispatcher spins 20 ms (IK2_HIP_LANE_SPIN_US), then sleeps 200 µs | CPU lane idle policy | MAYA | maya-ds4's DS4_LANE_SPIN_US 20000 us; measured within noise of the best: 200 / 1800 (the old 20000 yields) / 20000 us 57.0 / 55.0 / 55.2 tok/s (below) |
| E70 | Depth-mix RMS eps `1e-6f` (hard-coded) | Depth-mix norm epsilon | MODEL\* | llama.cpp inkling2.cpp `ggml_rms_norm(ctx0, states, 1e-6f)`; also config `rms_norm_eps 1e-06` |
| E71 | Router weight-sum eps `1e-20f` | Router normalization | MODEL\* | llama.cpp inkling2.cpp `expert_weights_norm_eps = 1e-20f` |
| E72 | `tau = 1 + alpha·log(max(1, (p+1)/floor))` | Log attention scaling | MODEL | alpha 0.1 (GGUF / config), floor 8192 (E10) |
| E73 | `kMaxStates` 16 | Cap on depth-mix states | CHOSEN (bound) | The model needs ceil(48 / 6) + 1 = 9 states (n_layers_per_attn_res_block 6) |
| E74 | `kTile` 32, `kSeg` 128 | Token attention tiles | CHOSEN (trade) | Not on the serving path: k::attention runs only in the step with a tap or a pick hook (parity, the tests); the served steps use the split attention (E78). Kept |
| E75 | `kATK` 16 | Keys per window-attention tile | MEASURED | 16-key tiles were the fastest of 64 / 32 / 16 at a 1024-key window (11.2 / 8.9 / 6.4 ms a token over the 48 layers: split_tk's comment); the window's tiles take the same size |
| E76 | `kAU` 4 for up to four rows, 8 from five | Keys per thread in window scores | MEASURED | kernels.cu (ms by T for 1 / 2 / 4 / 8 keys) |
| E77 | `kCombTiles` 8192 | Tiles per combine chunk (past 131072 keys) | CHOSEN (bound) | 8192 floats = 32 KB, a block's shared memory for the tiles' factors; past it (131072 keys at 16 a tile) the factors go a chunk at a time |
| E78 | `split_tk` 16 / 16 (`IK2_HIP_ATTN_TK`) | Split-attention tile keys | MEASURED | split_tk's comment: 64 keys 11.2, 32 keys 8.9, 16 keys 6.4 ms a token at a 1024-key window |
| E79 | Split scratch for at least 2048 keys, 16-key tiles | Attention scratch sizing | CHOSEN (bound) | At least 256 tiles: IK2_HIP_ATTN_TK's smallest tile (8) over 2048 keys |
| E80 | Clip eps `1e-6` | L2 clip denominator | MODEL\* | llama.cpp inkling2.cpp `ggml_scale_bias(norm, 1.0f, 1e-6f)` |
| E81 | Conv scale 1/sqrt(9), 1/sqrt(5) | Short-convolution output scale | MODEL\* | llama.cpp inkling2.cpp `1/sqrtf((dynamic ? 2 : 1)*n_conv + 1)`, n_conv = 4 (config `sconv_kernel_size`) |
| E82 | SITU beta 4.0, lb 25.0 | Expert activation | MODEL | GGUF `situ_beta 4.0` / `situ_linear_beta 25.0`; config `mlp_gate_softcap 4.0` / `mlp_up_softcap 25.0` |
| E83 | Fallbacks eps 1e-6, conv 4, res_block 6 | Used when the pack lacks the key | MODEL | config `rms_norm_eps`, `sconv_kernel_size`, `n_layers_per_attn_res_block` |
| E84 | No fallback: inkling2.txt must give expert_weights_scale, final_logit_softcapping, log_scaling_n_floor, bos, eos | Same | MODEL | The pack's values (8.0, 30.0, 8192, 200006); a pack without them is refused, as pack_gguf.py refuses a GGUF without them |
| E85 | Guards: n_embd % 256, head_dim ≤ 256, d_rel ≤ 256, conv == 4, n_used ≤ 16, n_expert ≤ 1024, states ≤ 16 | Geometries the engine refuses | CHOSEN (bound) | The kernels' limits: n_embd and n_ff_exp in whole 256-value blocks (ggml's k-quant super-blocks), head_dim and d_rel <= 256 (clip2 / attention_split), conv 4 (the 4-tap short convolution), n_used <= 16 and n_expert <= 1024 (hy4's router), states <= kMaxStates 16 |
| E86 | `kRowChunk` 128 | Rows per CPU work item | MAYA | hy4_lane.cpp (measured there: 64 and 32 slower on Zen 5) |
| E87 | `used[64]` | Rows tracked per CPU window | CHOSEN (bound) | At least kWin (9) rows |
| E88 | `inkling2.txt`, `tensors.txt` | Pack file names | MAYA | Hy4's pack layout (maya.py: "hy4.txt, tensors.txt, tokenizer/") |
| E89 | `--max-context --reserve-mib --vram-experts --cpu-threads --expert-profile --profile-out --prefill-chunk --eos` | Flag names | MAYA | ds4_main.cpp's flags; now also --no-expert-cache, --profile-readonly, --dspark on\|off\|none |
| E90 | `--ctx --log-floor --rewind --short-prompt --dspark-index --dspark-cum`; `--dspark idle` | Our own flag names | CHOSEN (names) | Settings maya-ds4 has no flag for (its cum floor: MAYA_DS4_DSPARK_CUM and the request's dspark_cum, the same key here); idle (tests) has no maya-ds4 counterpart |
| E91 | `IK2_HIP_*` env names | Test and diagnostic switches | CHOSEN (names) | The family's prefix, as DS4_* / HY4_* |
| E92 | Window-test pass: cosine ≥ 0.9995 | Test threshold | CHOSEN (bound) | Below the 9 of 9 rows found equal to the bit (above): room for a moved expert only |
| E93 | Rewind-test pass: cosine ≥ 0.9999 | Test threshold | CHOSEN (bound) | The exact cases resume at a chunk's start: the same chunks as the cold read, so only rounding is left |
| E94 | Parity: mean cosine ≥ 0.999; every row ≥ 0.999 on ≤ 64 positions | Test threshold | MEASURED | INKLING2_AMD.md (llama.cpp's own noise floor: a worst cosine of 0.23 on long prompts) |
| E95 | `--mmap-experts` auto (`Options::mmap_experts` -1); on / off | The experts VRAM does not hold: copied into RAM, or read in place from the files | MAYA (the flag, values, default, off and on); MEASURED (auto's rule) | maya-ds4's --mmap-experts (ds4_main.cpp, copy_experts). Its auto copies none unless all fit with 8 GiB to spare; here auto is off's - the most routed that fit: at 1M the 2.7k prompt continued at 47.2 tok/s with 100 GiB copied against 32.2 with none, prompts alike (49.1 / 51.4 tok/s) ("The experts VRAM does not hold") |
| E97 | `--parallel` 1 (`Options::seqs`), 1 .. 8 | Conversations held at once | MAYA | maya-ds4's --parallel (o.seqs, 1 .. 8, --serve), its protocol (READY multi=N, rid=, STOP <rid>) and its rule for a request's conversation (the idle one its prompt shares the most with, else the least recently used) |
| E98 | A multi step's rows: one a running request, at most 8 (<= kWin 9) | The step through every layer | CHOSEN (bound) | The window's buffers hold kWin rows; --parallel's 8 at most |
| E99 | A prompt read whole when its turn comes; a request alone drafts, several at once a token each | Scheduling of several requests | CHOSEN (trade) | maya-ds4 reads prompts a piece a turn and drafts beside several requests (its multi windows); neither is here yet ("Several conversations at once": what remains) |
| E96 | The prompt path's gather from the files: up to 8 threads, `MADV_WILLNEED` first, into a pinned buffer of kStage (E44) blobs a GPU | A staging pass's in-place experts | CHOSEN (method) | Pinned, as every copy to the device here (h2d's note: a copy straight from the mapping once landed a stale page); 8 threads: the four GPUs' gathers side by side within the 24 cores; kStage: a staging pass's experts. Not measured against other counts: the 1M prompts are bound by the SSD (E95's runs) |

### Drafter

| # | Value | What it does | Source | Where it comes from |
|---|---|---|---|---|
| D1 | `DsparkOptions::max_ctx` 32768 | Global-layer cache positions | MAYA | The engine's Options::max_ctx default (maya-ds4's), which the engine passes anyway |
| D2 | `rewind` 4096 (the engine passes `rewind + Tc`) | Ring margin | MEASURED | Follows E8 |
| D3 | `cum_min` 0.5 | Draft cut | MEASURED | As E11 (DSPARK.md; ds4_main.cpp) |
| D4 | `attn_tile` 8 | Keys per attention tile | MEASURED | dspark.hpp (8 / 16 / 24 keys at 300 / 2.7K / 16K) |
| D5 | `attn_gemm_min` 1024 | GEMM attention from this many keys on | MEASURED | dspark.hpp |
| D6 | `own_gemv` false | The block's GEMV kernel choice | MEASURED | dspark.hpp (within 3%) |
| D7 | `joint_index` empty: Q8_0 joint_proj | Where joint_proj comes from | MEASURED | DSPARK.md (Q8_0 = BF16 acceptance) |
| D8 | `kTaps` 8 | Taps per position | MODEL | config `mtp_config.target_layer_ids`, 8 entries |
| D9 | `kBlock` 8 | Block rows | MODEL | config `mtp_config.block_size 8` |
| D10 | `ctx_bytes` = 2·8·128·2 + 8·48·(4+2) = 6400 | VRAM per extra position | DERIVED | One global layer (`local_layer_ids [0,1,3,4]` of 5) x K, V x 8 KV heads x 128 x F16, plus block 8 x 48 heads x (F32 + F16) scores. The numbers are hard-coded, not read from dspark.txt. |
| D11 | `kFeed` 16 | Positions per feed step | MAYA | ds4_iq.cu (ncol ≤ 16) |
| D12 | `kMaxTiles` 4096 | Cap on attention tiles | CHOSEN (bound) | Past kMaxTiles tiles the tile doubles (the same sums, slower); the default GEMM form takes every context from 1,024 keys on (Sg / P16 sized from max_ctx), so the tiles see fewer. Without it (attn_gemm_min < 0) a context past the largest tile shared memory holds is refused at the load |
| D13 | `kPartAcc` 24 | Kernel accumulator bound | CHOSEN (bound) | = the 48 queries (8 rows x 6 heads a kv head) over the 2 groups of 128 threads; the load checks it |
| D14 | eps 1e-6, softcap 30 (fallbacks) | Norm eps; logit cap | MODEL | dspark.txt `eps 1e-06` (mtp_config.rms_norm_eps), `softcap 30.0` (text_config.final_logit_softcapping) |
| D15 | rope_theta fallback 1e6 | RoPE base | MODEL | mtp_config.rope_theta 1000000.0 |
| D16 | Guards NL ≤ 15, D ≤ 128, R 4..256 (multiple of 4) | Drafters it refuses | CHOSEN (bound) | The shapes the drafter's kernels were written and tested for; the model (5, 128, 256) is inside them |
| D17 | Tile clamp 4..64, else 8 | Option bounds | CHOSEN (bound) | 64: the largest tile whose shared memory fits (45 KB); 8 the measured default (D4) |
| D18 | SWA ring = n_swa + rewind + kFeed | Ring rows | DERIVED | 1024 (dspark.txt n_swa) + D2 + D11 |
| D19 | Context ring FR = min(max_ctx, rewind + 2·kFeed) | Markov context rows kept | CHOSEN (bound) | The rows a rewind can reach (D2) and two feeds |
| D20 | Markov state as int16 (x32767) | State quantization in the head | CHOSEN (method) | int16 x 32767 state in the head; acceptance matched llama.cpp's through the engine tests, no number for this scale alone |
| D21 | M rows int8, scale max/127 | Folded head matrix | CHOSEN (method) | Symmetric Q8 rows of the folded matrix |
| D22 | M built in 16-column chunks | GEMV batching | MAYA | ds4_iq.cu |
| D23 | `maya-ik2-dspark-<kind>/`, `dspark.txt`, `format maya-ik2-dspark 1`, `model_dir ..` | Index layout | CHOSEN (names) | tensors.txt in pack_gguf.py's format |
| D24 | The engine loads the Q8_0 copy | Drafter weight type | MEASURED | DSPARK.md |
| D25 | Taps = outputs of layers 4, 10, ..., 40 + `result_norm` | What the drafter is fed | MEASURED | DSPARK.md (the other tap variants are worse) and config `target_layer_ids` |
| D26 | RoPE NEOX (half-split) | Rotary pairing | MEASURED | DSPARK.md (adjacent pairs collapse to 0.32 / 0.69 / 0.30) |

### Setup

| # | Value | What it does | Source | Where it comes from |
|---|---|---|---|---|
| S1 | `IK2_EXE` `maya-ik2` (the CUDA build's target name too) | Engine binary | MAYA | `maya-<engine>`: DS4_EXE maya.py, HY4_EXE 804 |
| S2 | `IK2_SMALL` "Inkling-2-Small" | Model id | MODEL | GGUF `general.name` / `basename` |
| S3 | `IK2_SMALL_SHOWN` "Inkling 2 Small" | Name on screen | MODEL | GGUF `general.base_model.0.name` |
| S4 | `tools/ik2/download-small.json` | Manifest | MAYA | DS4_MANIFEST maya.py, HY4_MANIFEST 808 |
| S5 | `IK2_PACK` "maya-ik2" | Pack folder in the model's | MAYA | HY4_PACK "maya-hy4" maya.py |
| S6 | `IK2_SMALL_QUANT` "ud-q2_k_xl" | Quant key | MODEL | unsloth's folder `UD-Q2_K_XL` (download-small.json; the local folder) |
| S7 | `IK2_GGUF_DIR` "Inkling-2-Small-GGUF" | GGUF folder under models | MODEL | The repo name, `unsloth/Inkling-2-Small-GGUF` (download.json) |
| S8 | `IK2_MTP_DIR` "Inkling-2-Small-MTP" | Drafter folder | CHOSEN (names) | This PC's folder name for Thinking Machines' repo |
| S9 | `IK2_DSPARK` "maya-ik2-dspark-q8_0" | Drafter index folder | CHOSEN (names) | convert_dspark.py's Q8_0 index |
| S10 | Removed | - | - | It was not used |
| S11 | "maya-inkling-2-small-hip.json" (`maya_quants.config_stem` + "-hip") | Config file name | MAYA | "maya-inkling-2-mini.json", "maya-ds4-hip.json" maya.py |
| S12 | `IK2_SMALL_MODEL_NAME` "inkling-2-small" | API model name | MAYA | Lower-case ids: DS4_MODEL_NAME maya.py, HY4_MODEL_NAME 848 |
| S13 | `IK2_SMALL_CONTEXTS` top 1048576 | Largest context offered | MODEL | inkling2.txt `context_length`; config `model_max_length` |
| S14 | `IK2_SMALL_CONTEXTS` 32K .. 1M | Contexts offered | MAYA | DS4_CONTEXTS |
| S15 | `IK2_SMALL_DEFAULT_CONTEXT` 65536 | Fallback and lower bound of the recommendation | MAYA | DS4_DEFAULT_CONTEXT maya.py. It is not what the engine defaults to (E1, 32768), and nothing was measured at 64K (INKLING2_AMD.md). Measurable here. |
| S16 | `IK2_SMALL_RECOMMENDED_CONTEXT` 262144 | Highest recommendation | MAYA | DS4_RECOMMENDED_CONTEXT (DeepSeek's reason: its indexer's quadratic cost). maya-ik2 has been run end to end with a server at 256K and prompts up to 109,464 tokens (W2) |
| S17 | IK2_KV_BYTES 32768 + IK2_ATTN_BYTES 1560 on each GPU (+ IK2_DSPARK_KV_BYTES 6400 while it drafts) | Cache VRAM per context token | DERIVED | From the engine's allocations (serve/models.py says which): the global layers' K / V rows; each GPU's attention_split scratch, 48 x 130 floats a 16-key tile (past its 58 MB floor, 37,152 positions); the SWA rings are 7168 rows (E40) whatever the context. The setup and the page read the same constants |
| S18 | `IK2_DSPARK_KV_BYTES` 6400 | The drafter's per token | DERIVED | = D10 |
| S19 | `IK2_SMALL_MEMORY_GB` 203 (download-small.json the same) | VRAM + RAM that must hold the model | DERIVED | ceil(files 190.9 GiB + the engine's RAM margin 8 GiB + the drafter's 3.47 GB = 3.2 GiB) |
| S20 | `IK2_DSPARK_GB` 3.47 | Drafter's VRAM added to the need | MEASURED | The last GPU's free VRAM after the load's warm-up with the drafter and without it: 25124 against 28438 MiB at 32K ("VRAM" above) |
| S21 | `data/expert-profile-ik2.bin` | Shipped expert profile | MEASURED | The path follows DS4_PROFILE (maya.py). The contents equal the pack's default.prof byte for byte: 15,139,200 routings = 39,425 tokens x 48 x 8. **The corpus is not recorded** (commit 8d4e2c7 says only "the profile the AMD runs used"). |
| S22 | Set up on AMD under Linux only | Platform gate | CHOSEN (scope) | Inkling-2-Small on maya-ik2's CUDA build has no setup yet (Inkling-2-Mini's preview has) |
| S23 | VRAM + RAM < need: a warning, the setup goes on | Memory gate | MAYA | DeepSeek's check (check_pc); the warning says the experts RAM does not hold are then read from the SSD (--mmap-experts auto), slower |
| S24 | Recommendation pool `vram + ram - need` | Context recommendation | MAYA | ds4_recommended_context |
| S25 | 64K when RAM is unreadable or nothing fits | Fallback | MAYA | maya.py |
| S26 | Warn when a chosen context's cache does not fit | Warning | CHOSEN (warning) | A chosen context whose cache leaves the experts too little: said, on the same pool as S24 |
| S27 | `--reserve-mib` = `--vram-reserve-mib`, else hip_reserve_mib: 3072 (1024 on a Linux APU) | VRAM kept free | MAYA | DeepSeek's setup maya.py; hip_reserve_mib maya.py |
| S28 | `--dspark none` until the speed calibration; `--drafting measure \| on \| off` | Setup's drafting default | MAYA | DeepSeek's setup and tools/calibrate_ds4.py (now taking maya-ik2's --dspark-index) |
| S29 | A setup again keeps the config's --dspark; the calibration or --drafting then sets it | Re-setup rule | MAYA | write_config_ds4's keep_args |
| S30 | Port 8080 | Default port | MAYA | maya.py |
| S31 | Kept flags `--reserve-mib --vram-experts --layer-split --cpu-threads --expert-profile --prefill-chunk --dspark-cum --rewind --short-prompt --mmap-experts --parallel` | Hand-set flags a re-setup keeps | CHOSEN (list) | Every engine setting the setup does not ask, now with --dspark, --mmap-experts and --parallel (DeepSeek's list keeps them; the setup's --mmap-experts and --parallel are written, as DeepSeek's; PARALLEL_FAMILIES has ik2) |
| S32 | Sampling and effort from `MODEL_DEFAULTS["ik2"]` | Written into the config | MODEL | = T24 |
| S33 | "about a minute" | Conversion message | MEASURED | DSPARK.md (33 s niced, for both types) |
| S34 | `convert_dspark.py --type q8_0` | Only the Q8_0 copy is made | MEASURED | DSPARK.md |
| S35 | A shard 60 s newer than the pack makes it stale | Hand-made pack check | MAYA | maya.py (the same + 60) |
| S36 | Models folder `ROOT.parent/Maya-data/models` | Default location | MAYA | maya.py |
| S37 | Folders found from `--pack` (parents[1] == Inkling-2-Small-GGUF) and `--dspark-index` | Layout inference | DERIVED | From S7 / S8 / S9 |
| S38 | `IK2_PACK_NEEDS` files | Pack completeness check | MAYA | The pack layout of E88 |
| S39 | Profile copied only if its size = n_layer x n_expert x 4 (73,728) | Profile check | DERIVED | GGUF `block_count 48` x `expert_count 384` x 4 |
| S40 | Picker text: "384 experts, 8 a token", "about {IK2_SMALL_MEMORY_GB} GB", "up to a 1M context" | Model list row | MODEL | GGUF `expert_count` / `expert_used_count` / `context_length`; the 190 is S19 |
| S41 | The setup's drafting message is DeepSeek's (no figure) | Drafting message | - | The figure is IK2_DSPARK_GB where shown |

### Server and Models page

| # | Value | What it does | Source | Where it comes from |
|---|---|---|---|---|
| W1 | 32K .. 1M | Contexts on the Models page | MAYA | = S14 (models.py's DS4_CONTEXTS) |
| W2 | `CONTEXT_TESTED["ik2-hip"]` 524288 | Where the page starts saying "untested" | MEASURED | Run end to end at --max-context 524288 with a 307,710-token prompt ("Long contexts" above); 1M does not load on this PC (its RAM) |
| W3 | `IK2_KEYS` + parallel, reserve_mib, mmap, prefill_chunk, cpu_threads, dspark_cum (maya-ik2 only) | Settings the page edits | MAYA | DS4_KEYS; dspark_cum is dspark_conf's counterpart, in DeepSeek's Draft confidence row; mmap: DeepSeek's "Experts VRAM does not hold" row (auto: the flag dropped); parallel: DeepSeek's "Answers at once" row (1 .. 8, written above 1; the context's cost times it) |
| W4 | VRAM kept free (--reserve-mib) for maya-ik2 | - | MAYA | DeepSeek's row |
| W5 | `--dspark` read as "on" when absent; off and none read as Off | Current Drafting state | MAYA | The engine's default, as maya-ds4's; the page writes none (DeepSeek's) |
| W6 | Context fallback 32768 | Context when `--max-context` is absent | MAYA | = E1 (ds4_model.hpp) |
| W7 | Trained context from inkling2.txt `context_length` | Slider top | MODEL | inkling2.txt `context_length 1048576` |
| W8 | Drafter searched at `<models>/Inkling-2-Small-MTP/maya-ik2-dspark-q8_0` and `pack.parents[2]/...` | Finds the index | DERIVED | From S7 / S8 / S9 |
| W9 | The page's cost: the server's kv_bytes + kv_gpu_bytes x GPUs + dspark_kv_bytes while it drafts | Cache GB shown for a context | DERIVED | = S17, from serve/models.py |
| W10 | "About {IK2_DSPARK_GB} GB" (3.47) | Drafting hint | MEASURED | = S20 |
| W11 | Levels none, auto, low, medium, high, xhigh, max | Thinking levels offered | MODEL | The template's levels as the other PR's server takes them (frontend.py's INKLING_EFFORT): its effort_kwargs reads minimal as thinking off (0), so the page does not offer it beside Off |
| W12 | Default "auto" (the template's 0.9) | Default thinking level | MODEL | chat_template.jinja:5 |
| W13 | 0 / 0.2 / 0.7 / 0.9 / 0.99 in the titles | Labels | MODEL | chat_template.jinja:1 |
| W14 | "Inkling 2 Small (maya-ik2)", NAME_WORDS `inkling` | Labels | MODEL | GGUF base_model name |
| W15 | `INKLING_TEXT_ONLY` "Inkling-2-Small" | Name in picture errors | MODEL | GGUF `general.name` |
| W16 | Stops on `<|message_user|>`, `<|message_system|>`, `<|message_tool|>` | Turn-end guard | MODEL | chat_template.jinja:2 role tokens; vocab ids 200000 / 200002 / 200003. The template ends a model turn with `<\|content_model_end_sampling\|>` (124). Treating the role tokens as stops is our addition. |
| W17 | Call end `<|end_message|>` | Tool-call parsing | MODEL | chat_template.jinja:119 |
| W18 | GGUF-template detection: has `<|content_invoke_tool_json|>` and `<|message_model|>`, no `effort_map`; `MAYA_PACK_TEMPLATE=1` | Uses the official template | MODEL | Markers: chat_template.jinja:1, 112. The env name is CHOSEN. |
| W19 | maya-ik2 gets `--layer-split auto` (maya-ik2 does not) | Engine command | MAYA | The server's rule for any multi-GPU engine (server.py) |
| W20 | Call open `<|content_invoke_tool_(json\|text)|>`, close `<|end_message|>` | Loop guard | MODEL | chat_template.jinja:112, 119; vocab 200057 (`_text`) |
| W21 | A tool result is named by its position (the k-th result answers the k-th call) | Tool result header | CHOSEN (other PR) | frontend.py's inkling_messages (PR #1079's): OpenAI's and Anthropic's order |
| W22 | `IK2_OTHER` ("inkling-2-small-ud-q2_k_xl", "unsloth") | Download row | MODEL | download.json (unsloth) |
| W23 | Offered only on `backend == "hip"`; size = set + drafter | Download offer | CHOSEN (scope) | = S22; the size is download.json's sums |

### Tools

| # | Value | What it does | Source | Where it comes from |
|---|---|---|---|---|
| T1 | `STALE_FLOOR {128000: 8192}` | Fixes the GGUF's floor | MODEL | config.json `log_scaling_n_floor 8192` against GGUF `log_scaling_n_floor` / `attention.log_scale_floor` 128000. The Thinking Machines commit hash (pack_gguf.py; INKLING2_AMD.md) cannot be checked locally; config.json can. |
| T2 | Output `<model>/maya-ik2` | Pack folder | MAYA | = S5 |
| T3 | No fallback: the GGUF must give both keys | Used when the GGUF lacks the key | MODEL | pack_gguf.py's other required keys |
| T4 | `attention.log_scale_floor` read first, `log_scaling_n_floor` second | Key order | MODEL | Both are 128000 in this GGUF |
| T5 | Geometry guards | Refuses other shapes | CHOSEN (bound) | = E85 |
| T6 | Chat template: tools/ik2/chat_template.jinja over the GGUF's | Pack tokenizer | MODEL | The official template |
| T7 | `pre gpt-4o`, `add_bos False`, bos / eos 200006 | tokenizer.json | MODEL | GGUF `tokenizer.ggml.pre`, `add_bos_token`, `bos/eos_token_id` |
| T8 | `ARCH "dflash"` | Drafter GGUF architecture | MODEL\* | PR #32218's converter (DSPARK.md) |
| T9 | `N_TENSORS` 84 | Tensor count check | MODEL | mtp.safetensors has 84 BF16 tensors (DSPARK.md; expect_shapes builds them from mtp_config) |
| T10 | `block_size == 8` required | Config check | MODEL\* | "the PR's own checks"; config block_size 8 |
| T11 | `sample_from_anchor false`, `has_confidence_head false`, `attention.causal false` | GGUF keys | MODEL\* | PR converter (DSPARK.md) |
| T12 | eos fallback 200006 | Tokenizer key | MODEL | config `eos_token_id 200006` |
| T13 | eot = eos; `mask_token_id 0` | Tokenizer keys | MODEL\* | DSPARK.md (the PR's) |
| T14 | 1-D tensors and fuse_weight as F32 | Tensor types | MODEL\* | The PR forces the fuse (comment) |
| T15 | `--type` default "both" | Which copies are made | CHOSEN | Both copies for the tests; the setup passes q8_0 (S34) |
| T16 | Index names and format | Index layout | CHOSEN (names) | = D23 |
| T17 | Verify bound: \|d\|·(0.5 + 127·2^-11)·1.001 + 127·2^-25·1.001 | Q8_0 read-back tolerance | DERIVED | The x1.001 is CHOSEN slack over the derived bound |
| T18 | name "Inkling-2-Small DSpark", base_model URL | GGUF metadata | MODEL | download.json repo |
| T19 | unsloth repo, pinned revision, 6 shard sizes and sha256 | Model download | MODEL | The sizes match the local shards byte for byte (sha256 not recomputed) |
| T20 | 202 | Memory need | DERIVED | = S19 |
| T21 | Drafter repo, revision, mtp.safetensors 3,423,429,800 B, config.json 2,767 B | Drafter download | MODEL | Match the local files |
| T22 | Family `ik2`, MODEL / SHOWN names | Family table | MODEL | = S2 / S3 |
| T23 | OTHERS `ud-q4_k_xl`, `ud-q3_k_xl`, `ud-q2_k_xl` | Quant labels | MODEL / CHOSEN | UD-Q2_K_XL is on disk; Q4 / Q3 are maya-ik2's (the other PR's) labels |
| T24 | `MODEL_DEFAULTS["ik2"]` temperature 1.0, top_p 1.0, "auto" | Family sampling | CHOSEN (other PR) | maya_quants.py's MODEL_DEFAULTS (blame 1ef2ea1): "the card and files name no sampling" |
| T25 | `IK2_SMALL_IDS` aliases | Setup ids | CHOSEN (names) | Setup ids, as Inkling-2-Mini's `IK2_IDS` |
| T26 | `ik2_amd`, `is_ik2_mini` (the record's `ik2_quant`) | Which build and model a config runs | CHOSEN (names) | Family detection; the engine names are MAYA (S1) |

The measurement phase's list - the prompt chunk (E9), the rewind reach (E8, D2), the short prompt (E15), the drafting rounds' scheduler (E27 - E34), the expert cache's decay and swap rule (E56, E57), the burst and the warm-up (E54, E60, E61, E63, E65), the lane's spin (E69), E42's floor, E43, E44, E62, E74 and the VRAM slacks (E39, E41, E45, E47, E52) - is done: "The measured values" and "VRAM" above.