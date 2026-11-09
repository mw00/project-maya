# data/glm-dsa: glm-dsa fixture hashes and reference logits

The glm-dsa (GLM-5.3) engine's test fixtures. The synthetic GGUFs themselves are not committed: `tools/glm_dsa_synth_gguf.py` regenerates them byte for byte (`--all --out-dir DIR`). Everything here is rewritten by `tools/strata/glm_dsa_refs.py`.

Token convention for every reference: token `i` is `(i * 37 + 11) % n_vocab` (n_vocab 512 for SMALL/LONG, 1024 for PACK/TIERS). Logit files are raw little-endian float32, row-major `(n_tokens, n_vocab)`, the layout `llm_logits_dump` writes.

| File | Content |
|---|---|
| `sha256.txt` | sha256 of the standard presets at the default seed (5300): SMALL, LONG, TIERS, PACK, PACK with `attn_k_b`/`attn_v_b` in Q4_0, IQ4_NL and F16, PACK9 |
| `oracle-hf-small-48.f32` | `ref/glm_dsa.py --mode hf` on SMALL, 48 tokens (top-8 selection active from position 8) |
| `oracle-hf-small-mtp-47.f32` | the NextN draft logits of the same run: row t uses the trunk's post-norm hidden at t and token t+1, position t |
| `llama-small-48.f32` | llama.cpp `aa94f208` CPU on SMALL, 48 tokens |
| `llama-long-2080-tail32.f32` | llama.cpp on LONG, 2080 tokens, rows 2048..2079 only (top-2048 selection active) |
| `llama-pack-12.f32`, `llama-tiers-12.f32` | llama.cpp on the quantized PACK and TIERS files, 12 tokens |

The llama.cpp references come from `tools/strata/glm_dsa_llama_logits.cpp` (one token per `llama_decode`, F32 K cache, flash attention off, no repacking buffer types; the source file says why each setting is needed).

## Measured agreement (2026-10-11, CPU container)

| Comparison | Max abs logit diff | Notes |
|---|---|---|
| oracle `--mode llama` vs llama.cpp, SMALL 48 tokens | 3.2e-5 | KL p99 3e-11, top-1 100% |
| oracle `--mode llama` vs llama.cpp, LONG 2080 tokens | 5.9e-6 (rows < 2048), 4.0e-6 (rows >= 2048) | the same oracle with selection disabled differs by 1.28 on rows >= 2048 |
| oracle vs llama.cpp, BIG geometry with F32 weights (8 layers, 64 heads, attn_k_b {192,512,64}) | 6.5e-6 | ad-hoc check, not committed |
| oracle vs llama.cpp, PACK / TIERS / WIDE (quantized weights) | 0.2 to 1.2 | llama.cpp quantizes activations to q8 for every quantized matmul; top-1 100%, mean KL 1e-3 to 5e-3 |
| oracle vs llama.cpp, BIG (78 layers, quantized) | 2.0 | the same activation quantization, compounded over 78 random layers (relative error 5e-3 after layer 0, 0.18 after layer 77); top-1 50% |
| `--mode hf` vs llama.cpp, SMALL | 9.1 | expected: SMALL's indexer keys have variance near 1e-6 on purpose, so HF's LayerNorm eps 1e-6 and llama.cpp's eps 0 select different cells |

Settings that matter for a tight llama.cpp comparison: an f16 K cache moves SMALL's logits by about 1e-2, and so does the CPU flash-attention kernel even with an F32 cache. On a host with AMX (this container: Sapphire Rapids) llama.cpp repacks the 3-D `attn_k_b` into the AMX buffer and then aborts in the scheduler (`ggml-backend.cpp` `GGML_ASSERT(*cur_backend_id != -1)`); Skylake CPUs have no AMX, but a GitHub runner might.
