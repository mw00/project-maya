# MiniMax-M3.1-Flash on Apple Silicon (Metal) - port notes

Experimental. Back to [docs/METAL.md](METAL.md#minimax-m31-flash).

Started 2026-11-01 on Duckets Mac Studio (M5 Max, 128 GB unified) - the Maya team has no Mac;
this machine is the test hardware. Fork: Franzferdinan51/project-maya @ 465a981 (v1.1.48).

## Verified on real Mac hardware
- ./maya.sh --backend metal --check PASSES: M5 Max, ~109 GB GPU pool, clang 21, CMake 4.4, Python 3.15.
  (maya.sh needs Homebrew python first in PATH; system python3 is 3.9 and fails the check.)

## Why MiniMax does not run on Mac today
- src/metal is GLM-only: hand-written ggml graphs (glm_metal*.cpp) + expert pool + sampler + serve.
- MiniMax engine (src/mm3) is CUDA/EXL3-kernel only. The pinned llama.cpp fork has NO minimax arch.
- BUT Mayas own MiniMax quants are GGUF (peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF):
  dense in Q4_K/Q5_K/Q6_K/Q8_0, experts in IQ1_S/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS - all standard ggml
  types with ggml-metal kernels. No custom quant kernels needed. Activation is swigluoai.

## M3.1 architecture contract (src/mm3/mm3_geo.hpp - the kernels coverage checks)
- Every layer sparse attention: indexer (ix_heads = n_kv = 4, ix_dim 128) scores 128-token blocks,
  keeps topk (<=64) blocks + 1 local block; sparse_attention_freq all 1s in M3.1.
- Attention: n_head = 16 * n_kv, head_dim 128, partial rotary (n_rot <= hd), rope_theta, rms eps.
- KV cache (CUDA engine): BF16 [ctx][4][128] K and V + BF16 [ctx][128] index-key cache.
- MoE: <=256 experts, <=8/token, routed_scaling_factor; moe_layer_freq per layer; dense FFN
  (ff_dense) on non-MoE layers; shared experts (ff_shared).
- swigluoai: g=min(g,limit); u=clamp(u,-limit,limit); out = g/(1+exp(-1.702*g)) * (u+1). alpha=1.702.
- Oracles: src/mm3/mm3_cpu.cpp (CPU lane semantics), mm3_model.cu (full forward), ds4 IQ CPU rows.

## Plan
1. Build existing maya-metal (GLM) on this Mac - toolchain proof. [running]
2. mm3 metal engine beside glm_metal: GGUF loader (adapt glm_metal_load), graph builder for the
   sparse-indexer attention + MoE above, expert pool + sampler + serve reused.
3. Synthetic parity test in the style of metal_model_test (small fixture vs CPU reference).
4. Route MiniMax in maya.sh/setup/serve for --backend metal; update docs/METAL.md + docs/MODELS.md.
5. Real-model run + ./maya.sh --bench, file the results upstream (issue) and push to the fork.

## Blocker for step 5
- MiniMax Maya-XS GGUF = 93 GB. Sabrent: 52 GB free. Internal: 15 GB free. Needs ~41+ GB freed
  on Sabrent (or another disk) before the model can land.

## Milestone 2026-11-01: GLM engine verified on real Mac hardware
- cmake -DMAYA_ENABLE_METAL=ON build: BUILD-EXIT-0 (maya-metal + metal_model_test).
- ctest: 27/27 passed, 82 s total, CTEST-EXIT-0 (M5 Max). First full real-Mac pass of the suite.

## Progress — 2026-11-01 (Juno, on Duckets' Mac Studio)

**Milestone 1 DONE: the GLM Metal engine builds and passes its full parity suite on real Mac
hardware** — the first full real-Mac run the project has had (upstream has no Mac; CI only).
`ctest` in build-metal: 27/27 at that point.

**Milestone 2 DONE: the MiniMax-M3.1 Metal engine exists and passes parity.**
New files: `src/metal/mm3_metal.hpp`, `mm3_metal_impl.hpp`, `mm3_metal_load.cpp`, `mm3_metal.cpp`,
`mm3_main.cpp` (the `maya-metal-mm3` CLI), `src/metal/metal_mm3_test.cpp`,
`tools/mm3_metal_fixture.py`. CMake: mm3 sources join the `maya_metal` library; tests
`metal_mm3_gen` / `metal_mm3_test` / `metal_mm3_test_cpu` registered. Full suite now **30/30**.

Design as built (follows `tools/mm3/ref.py` op for op):
- Loader: minimax-m3 GGUF shards + the `-maya-indexers.bin` side file (dense layers' indexers),
  geometry from the `minimax-m3.*` keys exactly as `tools/mm3/pack_gguf.py`'s geometry() maps them.
  Norm weights are used as stored (llama.cpp's converter already folded the Gemma +1 in).
- Every tensor resident, routed experts included — the Maya quants are made to fit a 128 GB Mac
  whole, so there is no expert pool and **no graph cut at the router**: the router (sigmoid +
  correction bias + `ggml_argsort_top_k` + renormalised weights) runs in-graph and `mul_mat_id`
  consumes its top-k directly, over the stacked `ffn_*_exps.weight` tensors.
- Two graphs per layer: graph 1 computes q/k/v + indexer iq/ik (Gemma norms, NeoX partial RoPE),
  writes the BF16 KV and index-key caches, and reads the indexer scores `iq·ik` back to the host;
  the host picks kept blocks exactly as `ref.py`'s `msa_select` (block max, local blocks forced,
  top-k); graph 2 runs the masked sparse attention per KV group + the MLP (dense or MoE, all
  swigluoai) and writes the residual carry. The last layer's graph 2 also runs the head.
- Parity fixture: 3 layers (1 dense lead + 2 MoE), real head_dim/rotary/indexer widths, 2,176
  tokens = 17 index blocks so selection must actually drop a block; layer 0's indexers ride in
  the side file. Oracle = a numpy port of ref.py in the generator (k/v/iq/ik rounded to BF16 as
  the engines' caches do). Results (M5 Max): prompt logits rel L2 **4.1e-05 (CPU) / 4.2e-05
  (Metal)** vs the oracle, sliced == one-shot, greedy 8/8 tokens identical. Gates set at 1e-3.

Known v1 limits (deliberate, in order):
- Block selection is host-side from GPU scores: the scores readback is Tk×heads×T floats per
  layer. Correct and simple; an in-graph selection (ggml_top_k over block maxima) is the known
  next step when profiled on the real model.
- No `--serve` wiring yet: `maya-metal-mm3` is CLI-only (`--ids`); the serve protocol port is
  the next chunk, with the maya.sh/setup/docs wiring.
- The fixture is all-F32 (oracle exactness). The IQ1/IQ2 expert kernels the real quants use are
  ggml's standard mul_mat_id path (the same one every llama.cpp MoE GGUF takes), but a quantized
  fixture or the real model has not run yet — the real Maya-XS set is 93 GB and this Mac's
  Sabrent has ~52 GB free, so the real-model run waits on disk space (owner's call).

## Progress — 2026-11-01, later (milestone 3: serve + docs)

**The serve protocol is ported.** `src/metal/mm3_serve.cpp` implements serve.cpp's protocol for
Mm3Metal (GEN/STOP/PAUSE/WRAP/QUIT, PP/T/STAT/DONE lines, the shared sampler.cpp chain — same keys,
same Philox draws, seeded requests repeat). Deltas, all the model's rather than the protocol's: no
drafting (spec keys accepted; plain decoding writes the same tokens), GENI refused (vision tower not
wired), stop tokens from the GGUF's `tokenizer.ggml.eos_token_id`, and PAUSE/snapshots are nearly
free — Mm3Metal's caches are positional, so `snapshot_save/restore` are position save/rewind (added
to the engine for this). `tools/test_mm3_serve.py` (`metal_mm3_serve_test`) passes: a paused
request's follow-up reuses the kept prefix and writes the uninterrupted request's tokens, seeded
sampling repeats exactly, spec=1 changes nothing. Full suite: **31/31 passed** on this Mac (serial
ctest, 331 s — the disk-streaming GLM tests were slow on the Sabrent that day; no failures).

**Docs**: docs/METAL.md gained the MiniMax section (design, the resident-experts memory plan, why
Maya-XS is the Mac quant — Maya-S's 122 GB is past a 128 GB Mac's ~115 GB GPU working set), the
intro no longer calls MiniMax NVIDIA/AMD-only, the accuracy table has the mm3 row, and
docs/MODELS.md's Maya-XS row lists macOS/Apple Silicon.

**Still not done (deliberately)**: maya.py's setup/model-select integration — `./maya.sh --backend
metal --setup` does not offer MiniMax yet, and serve/server.py is not pointed at maya-metal-mm3 by
the setup. The engine speaks the server's protocol, so the remaining work is launcher plumbing in
maya.py (model routing, the GGUF download + pack step, recommended context), best done against
maya.py's own test suite (tools/test_maya.py, tools/test_maya_metal.py) rather than by feel.

## Progress — 2026-11-01, evening (milestone 4: the real model runs)

**Maya-XS runs on the Mac.** The full 93 GB set (peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF, Maya-XS:
two IQ1_S shards, 47.8 + 45.1 GB, plus the 12.5 MB indexers side file) was downloaded to
models/Maya-XS after freeing space on the Sabrent (121 GB free before the download, ~34 GB after).
All runs below are maya-metal-mm3 on the M5 Max with 8,192-token context state (1.16 GB) and
92.99 GB of weights resident.

- Load: 103–117 s (disk-bound).
- Chat prompt (the model's own template, 191 tokens): prefill 153 tok/s; the model answered
  "The capital of France is Paris." and emitted EOS. Decode 16.2 tok/s.
- Document prompt (3,227 tokens = 26 index blocks, so top-16 selection genuinely prunes): prefill
  242 tok/s; asked for the Maya-XS size "according to the document", the model reasoned over the
  document and quoted its line back ("Maya-XS (93 GB) fits into a 128 GB Mac with …") before the
  48-token budget cut its think block. Decode at ~3.2 k context: 6.7 tok/s — the sparse attention
  over ≤2,048 kept keys plus the per-layer host selection is the v1 cost, as designed.
- A raw completion prompt with no BOS and no chat template derailed into a repetition loop on this
  1.6-bit instruct quant — a prompt-format artifact, recorded so nobody re-diagnoses it: use the
  model's template (its tokenizer BOS is 200019, EOS 200020).

**One real bug found and fixed by the real set:** the loader derived the indexers side file's name
from the shard stem, but the real sets name it after the set, not the quant-tagged shard
(…-Maya-XS-maya-indexers.bin beside …-Maya-XS-IQ1_S-00001-of-00002.gguf), so the dense layers'
indexers were never found and the first real load failed on blk.0. The loader now discovers the
side file by scanning the shards' directory for the longest -maya-indexers.bin/.gguf name that
prefixes a shard name. Fixture parity unchanged (rel L2 4.24e-05); the four mm3 tests pass 4/4.

**Still not done (deliberately)**: maya.py launcher integration, as in milestone 3. Speed work
(fusing the per-layer host selection, batching decode graphs) is future work; v1 is
correctness-first, as the GLM engine was.
